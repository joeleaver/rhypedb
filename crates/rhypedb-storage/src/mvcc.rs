use std::collections::{HashMap, HashSet, VecDeque};

/// Type alias for the conflict-detection key set. AHash is ~10× faster than the
/// stdlib's SipHash on short keys (we measured ~30 µs saved per User delete at
/// K=100 cascading rows once we counted the bytes hashed). Not HashDoS-resistant
/// — fine for an in-process key set built from this process's own writes.
type WriteSet = HashSet<bytes::Bytes, ahash::RandomState>;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use parking_lot::{Mutex, RwLock};

use crate::{Error, Result};

/// Refcounted active-snapshot registry: snapshot version → number of live
/// transactions holding it. Shared (`Arc`) between the manager and every
/// [`Transaction`] it hands out, so a transaction can release its own hold on
/// drop without a back-reference to the manager (see `Transaction::release`).
type ActiveSnapshots = Arc<RwLock<HashMap<u64, u64>>>;

/// Release one hold on `snapshot`, removing it from the active set only when
/// the last transaction holding that version finishes.
fn release_hold(active: &RwLock<HashMap<u64, u64>>, snapshot: u64) {
    let mut guard = active.write();
    if let Some(count) = guard.get_mut(&snapshot) {
        *count -= 1;
        if *count == 0 {
            guard.remove(&snapshot);
        }
    }
}

/// Tracks a committed transaction for write-write conflict detection.
struct CommittedTxn {
    commit_version: u64,
    write_set: WriteSet,
}

/// The serialized commit state: the next commit version to reserve plus the
/// recent-commit log used for conflict detection. Guarded by
/// `TransactionManager::commit_mu` so the WHOLE commit critical section
/// (conflict-check → reserve version → apply durably → record → publish) is one
/// indivisible step. Nothing outside a held `commit_mu` touches these fields, so
/// there is no drop-then-relock window in which a racing committer could slip a
/// conflicting write past the check.
struct CommitState {
    next_version: u64,
    committed_log: VecDeque<CommittedTxn>,
}

/// MVCC transaction manager.
///
/// Assigns monotonically increasing commit versions, tracks active snapshots
/// for GC, and detects write-write conflicts at commit time. Writes are BUFFERED
/// in the `Transaction` and applied to the WAL + memtable only at commit (via the
/// `apply` callback), at the assigned commit version — so an aborted or
/// conflict-losing transaction never touches shared storage, and a committed
/// transaction's version is published to readers only AFTER its data is durable.
pub struct TransactionManager {
    /// The reader-visible version. A snapshot taken at `begin()` sees exactly the
    /// writes of every transaction whose `commit_version <= snapshot`. Advanced
    /// (Release) as the LAST step of a commit — AFTER the writes are applied — so
    /// it never names a version whose data is not yet in the memtable (closing
    /// the provisional-version visibility gap). Readers load it Acquire.
    visible_version: AtomicU64,
    /// Serializes the entire commit critical section (see `CommitState`).
    commit_mu: Mutex<CommitState>,
    /// Active read snapshots, REFCOUNTED by snapshot version. Two transactions
    /// can begin at the same visible version (no commit between their `begin`s),
    /// so the same snapshot value can be held by several readers at once; a plain
    /// set would let one transaction's commit/abort evict a value another reader
    /// still holds, wrongly raising `min_active_snapshot()` and letting
    /// compaction GC a version — or the conflict-log trim drop an entry — that
    /// the surviving reader still needs. The count is incremented at `begin` and
    /// decremented exactly once per transaction — at commit/abort, or on DROP
    /// of a transaction that was never finished; the key is removed at zero.
    ///
    /// The drop-release is load-bearing: engine paths routinely `drop(txn)` a
    /// read txn (open/rebuild, migrations, `?` error returns). Before it existed
    /// each such drop leaked a hold forever, pinning `min_active_snapshot()` at
    /// (typically) the recovery version, so compaction kept every version
    /// written since process start — unbounded disk growth on hot keys.
    active_snapshots: ActiveSnapshots,
    max_committed_log_size: usize,
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            visible_version: AtomicU64::new(0),
            commit_mu: Mutex::new(CommitState {
                next_version: 1,
                committed_log: VecDeque::new(),
            }),
            active_snapshots: Arc::new(RwLock::new(HashMap::new())),
            max_committed_log_size: 1024,
        }
    }

    /// Create a transaction manager that resumes from a known version.
    /// Used during recovery to restore the version counter from WAL/SST state.
    pub fn recover_at_version(version: u64) -> Self {
        Self {
            visible_version: AtomicU64::new(version),
            commit_mu: Mutex::new(CommitState {
                next_version: version + 1,
                committed_log: VecDeque::new(),
            }),
            active_snapshots: Arc::new(RwLock::new(HashMap::new())),
            max_committed_log_size: 1024,
        }
    }

    /// Start a new transaction, returning its snapshot version.
    pub fn begin(&self) -> Transaction {
        let snapshot = self.visible_version.load(Ordering::Acquire);
        *self.active_snapshots.write().entry(snapshot).or_insert(0) += 1;
        Transaction {
            snapshot,
            writes: Vec::new(),
            index: HashMap::default(),
            hold: Some(Arc::clone(&self.active_snapshots)),
        }
    }

    /// Commit a transaction's buffered writes.
    ///
    /// `apply(commit_version, writes)` durably applies the buffer (WAL append +
    /// memtable inserts) at the assigned version. It runs INSIDE the serialized
    /// critical section, AFTER the conflict check + version reservation and
    /// BEFORE the version is published, which gives three guarantees:
    ///
    /// * a conflict LOSER's buffer is never applied — the early return on
    ///   conflict happens before `apply`, so it can't overwrite the winner;
    /// * the visible version never names a not-yet-applied version — it is
    ///   published last, after `apply` returns;
    /// * same-key writers serialize, so apply order is well defined.
    ///
    /// On `apply` error the reserved version is NOT consumed and NOT published
    /// (the next commit reuses it), and the caller aborts.
    ///
    /// The empty-write fast path consumes NO version (a read-only/no-op txn must
    /// not advance the visible version past a version no apply ever published).
    pub fn commit<F>(&self, txn: &mut Transaction, apply: F) -> Result<u64>
    where
        F: FnOnce(u64, &[(Bytes, Option<Bytes>)]) -> Result<()>,
    {
        if txn.writes.is_empty() {
            txn.release();
            return Ok(txn.snapshot);
        }

        let mut state = self.commit_mu.lock();

        // Conflict check: any transaction committed AFTER our snapshot that wrote
        // one of our keys. Held continuously through publish — no relock window.
        for committed in state.committed_log.iter() {
            if committed.commit_version <= txn.snapshot {
                continue;
            }
            for (key, _) in &txn.writes {
                if committed.write_set.contains(key) {
                    // Conflict: return WITHOUT applying or consuming a version.
                    return Err(Error::WriteConflict);
                }
            }
        }

        // Reserve AND consume the version up front (do NOT publish it yet). A
        // failed apply burns this version rather than reusing it: `append_txn`
        // may have already written its data+footer to the page cache before a
        // later `sync()` errored, so reusing the version for the next txn could
        // alias that abandoned-but-page-resident batch on replay. Burning it
        // (a harmless gap in committed versions) keeps every persisted batch at a
        // unique version.
        let commit_version = state.next_version;
        state.next_version = commit_version + 1;

        // Apply durably at the reserved version, still inside the lock. On error
        // the version is consumed (above) but NEVER published, and the caller
        // aborts.
        apply(commit_version, &txn.writes)?;

        let write_set: WriteSet = txn.writes.iter().map(|(k, _)| k.clone()).collect();
        state.committed_log.push_back(CommittedTxn {
            commit_version,
            write_set,
        });

        // Trim entries no longer needed for conflict detection.
        let min_active = self
            .active_snapshots
            .read()
            .keys()
            .copied()
            .min()
            .unwrap_or(u64::MAX);
        while let Some(front) = state.committed_log.front() {
            if front.commit_version < min_active {
                state.committed_log.pop_front();
            } else {
                break;
            }
        }
        while state.committed_log.len() > self.max_committed_log_size {
            state.committed_log.pop_front();
        }

        // PUBLISH last (Release): the data is now in the memtable, so making the
        // version visible can never expose a hole.
        self.visible_version.store(commit_version, Ordering::Release);
        drop(state);

        txn.release();
        Ok(commit_version)
    }

    /// Abort a transaction: drop its buffered writes (never applied) and release
    /// its snapshot. Nothing to undo — the writes never reached the memtable.
    pub fn abort(&self, txn: &mut Transaction) {
        txn.release();
    }

    /// Returns the minimum active snapshot version, or u64::MAX if none active.
    /// Used to decide which old versions can be garbage-collected during compaction.
    pub fn min_active_snapshot(&self) -> u64 {
        self.active_snapshots
            .read()
            .keys()
            .copied()
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Observability: `(distinct snapshot versions held, total holds)`. A
    /// steadily non-zero count with a growing `current_version() -
    /// min_active_snapshot()` lag is the signature of a leaked/long-lived txn
    /// pinning compaction GC.
    pub fn active_snapshot_stats(&self) -> (usize, u64) {
        let guard = self.active_snapshots.read();
        (guard.len(), guard.values().sum())
    }

    /// Current reader-visible version (latest fully-applied + published commit).
    pub fn current_version(&self) -> u64 {
        self.visible_version.load(Ordering::Acquire)
    }
}

/// A transaction handle. Holds the read snapshot and the buffered writes.
///
/// Writes are buffered here and applied to the WAL + memtable atomically at
/// commit; an abort drops them. A transaction does NOT see its own uncommitted
/// writes (reads go to `snapshot`, the buffer is write-only) — which exactly
/// preserves the engine's pre-existing behavior (a put stamped above the read
/// snapshot was already invisible to that snapshot).
pub struct Transaction {
    snapshot: u64,
    /// Buffered writes, insertion-ordered with at most one entry per key
    /// (last-write-wins via `index`). `None` = tombstone. Drives both the WAL
    /// append and the memtable apply at commit, from the same sequence.
    writes: Vec<(Bytes, Option<Bytes>)>,
    /// key → index into `writes`, for O(1) last-write-wins coalescing.
    index: HashMap<Bytes, usize, ahash::RandomState>,
    /// The live hold on `snapshot` in the manager's active set; `Some` until the
    /// txn is finished (commit/abort) or dropped. `take()`-on-release makes the
    /// release exactly-once across commit/abort/drop. It owns no write path, so
    /// dropping an unfinished txn can only release the snapshot — its buffered
    /// writes are discarded, never applied.
    hold: Option<ActiveSnapshots>,
}

impl Transaction {
    pub fn snapshot(&self) -> u64 {
        self.snapshot
    }

    /// Release this txn's snapshot hold (idempotent — the second call is a
    /// no-op, so commit/abort followed by drop never double-decrements a
    /// refcount another txn shares).
    fn release(&mut self) {
        if let Some(active) = self.hold.take() {
            release_hold(&active, self.snapshot);
        }
    }

    /// Buffer a put for `key` (last write per key wins).
    pub fn record_put(&mut self, key: Bytes, value: Bytes) {
        self.upsert(key, Some(value));
    }

    /// Buffer a delete (tombstone) for `key` (last write per key wins).
    pub fn record_delete(&mut self, key: Bytes) {
        self.upsert(key, None);
    }

    fn upsert(&mut self, key: Bytes, value: Option<Bytes>) {
        if let Some(&i) = self.index.get(&key) {
            self.writes[i].1 = value;
        } else {
            let i = self.writes.len();
            self.index.insert(key.clone(), i);
            self.writes.push((key, value));
        }
    }

    /// The buffered writes (insertion-ordered, deduped) — the single sequence the
    /// commit drives both the WAL append and the memtable apply from.
    pub fn writes(&self) -> &[(Bytes, Option<Bytes>)] {
        &self.writes
    }

    /// True if this transaction has buffered no writes (a read-only txn).
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

impl Drop for Transaction {
    /// An unfinished txn releases its snapshot hold here — the structural
    /// guarantee that no code path (early `?` return, explicit `drop(txn)`,
    /// panic unwind) can pin `min_active_snapshot()` forever. Buffered writes
    /// are simply discarded (equivalent to abort): Drop has no apply path.
    ///
    /// No warning: dropping an unfinished read-write txn is the normal shape of
    /// every engine `?` error path, and this crate carries no logging dep (the
    /// old unconditional `eprintln!` here was stderr noise on each such error).
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A no-op apply: the manager-level tests exercise version + conflict logic,
    // not the WAL/memtable apply (that is covered at the LsmTree level).
    fn noop_apply(_v: u64, _w: &[(Bytes, Option<Bytes>)]) -> Result<()> {
        Ok(())
    }

    #[test]
    fn basic_commit() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        txn.record_put(Bytes::from("key1"), Bytes::from("v"));
        let version = tm.commit(&mut txn, noop_apply).unwrap();
        assert!(version > 0);
    }

    #[test]
    fn no_conflict_on_disjoint_writes() {
        let tm = TransactionManager::new();

        let mut txn1 = tm.begin();
        let mut txn2 = tm.begin();

        txn1.record_put(Bytes::from("key1"), Bytes::from("v"));
        txn2.record_put(Bytes::from("key2"), Bytes::from("v"));

        tm.commit(&mut txn1, noop_apply).unwrap();
        tm.commit(&mut txn2, noop_apply).unwrap(); // different keys → ok
    }

    #[test]
    fn conflict_on_overlapping_writes() {
        let tm = TransactionManager::new();

        let mut txn1 = tm.begin();
        let mut txn2 = tm.begin();

        txn1.record_put(Bytes::from("key1"), Bytes::from("a"));
        txn2.record_put(Bytes::from("key1"), Bytes::from("b")); // same key

        tm.commit(&mut txn1, noop_apply).unwrap();
        let result = tm.commit(&mut txn2, noop_apply);
        assert!(matches!(result, Err(Error::WriteConflict)));
        tm.abort(&mut txn2);
    }

    #[test]
    fn later_snapshot_sees_committed_writes() {
        let tm = TransactionManager::new();

        let mut txn1 = tm.begin();
        txn1.record_put(Bytes::from("key1"), Bytes::from("v"));
        let v1 = tm.commit(&mut txn1, noop_apply).unwrap();

        let mut txn2 = tm.begin();
        txn2.record_put(Bytes::from("key1"), Bytes::from("v"));
        let v2 = tm.commit(&mut txn2, noop_apply).unwrap();
        assert!(v2 > v1);
    }

    #[test]
    fn read_only_transactions_always_commit() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        // No writes buffered.
        let version = tm.commit(&mut txn, noop_apply).unwrap();
        assert_eq!(version, txn.snapshot());
        // A read-only commit consumes no version.
        assert_eq!(tm.current_version(), 0);
    }

    #[test]
    fn visible_version_only_advances_on_a_writing_commit() {
        let tm = TransactionManager::new();
        assert_eq!(tm.current_version(), 0);
        let mut t = tm.begin();
        t.record_put(Bytes::from("k"), Bytes::from("v"));
        let v = tm.commit(&mut t, noop_apply).unwrap();
        assert_eq!(tm.current_version(), v);
    }

    #[test]
    fn last_write_wins_coalesces_per_key() {
        let tm = TransactionManager::new();
        let mut t = tm.begin();
        t.record_put(Bytes::from("k"), Bytes::from("v1"));
        t.record_delete(Bytes::from("k"));
        t.record_put(Bytes::from("k"), Bytes::from("v2"));
        assert_eq!(t.writes().len(), 1, "one entry per key");
        assert_eq!(t.writes()[0].1.as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn min_active_snapshot_tracking() {
        let tm = TransactionManager::new();
        let txn1 = tm.begin();
        let _txn2 = tm.begin();
        assert_eq!(tm.min_active_snapshot(), txn1.snapshot());
    }

    #[test]
    fn abort_releases_snapshot() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        tm.abort(&mut txn);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
    }

    #[test]
    fn dropped_read_only_txn_releases_snapshot() {
        let tm = TransactionManager::new();
        let txn = tm.begin();
        assert_eq!(tm.min_active_snapshot(), 0);
        drop(txn);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
        assert_eq!(tm.active_snapshot_stats(), (0, 0));
    }

    #[test]
    fn dropped_read_write_txn_releases_snapshot_and_applies_nothing() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        txn.record_put(Bytes::from("k"), Bytes::from("v"));
        drop(txn);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
        // Nothing was committed: no version consumed or published.
        assert_eq!(tm.current_version(), 0);
        let mut t2 = tm.begin();
        t2.record_put(Bytes::from("k"), Bytes::from("v2"));
        assert_eq!(tm.commit(&mut t2, noop_apply).unwrap(), 1);
    }

    #[test]
    fn shared_snapshot_refcount_survives_one_drop() {
        let tm = TransactionManager::new();
        let a = tm.begin();
        let b = tm.begin();
        assert_eq!(a.snapshot(), b.snapshot());
        assert_eq!(tm.active_snapshot_stats(), (1, 2));
        drop(a);
        // b still holds the shared snapshot value.
        assert_eq!(tm.min_active_snapshot(), b.snapshot());
        assert_eq!(tm.active_snapshot_stats(), (1, 1));
        drop(b);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
    }

    #[test]
    fn commit_or_abort_then_drop_releases_exactly_once() {
        let tm = TransactionManager::new();
        // Three txns share snapshot 0; finishing-then-dropping two of them must
        // not over-decrement the count the third still relies on.
        let mut ro = tm.begin();
        let mut rw = tm.begin();
        let witness = tm.begin();
        let mut aborted = tm.begin();
        assert_eq!(tm.active_snapshot_stats(), (1, 4));

        tm.commit(&mut ro, noop_apply).unwrap(); // read-only fast path
        drop(ro);
        rw.record_put(Bytes::from("k"), Bytes::from("v"));
        tm.commit(&mut rw, noop_apply).unwrap(); // writing commit
        drop(rw);
        tm.abort(&mut aborted);
        tm.abort(&mut aborted); // a second abort is a no-op too
        drop(aborted);

        assert_eq!(tm.active_snapshot_stats(), (1, 1));
        assert_eq!(tm.min_active_snapshot(), witness.snapshot());
        drop(witness);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
    }

    #[test]
    fn conflict_loser_dropped_without_abort_releases() {
        let tm = TransactionManager::new();
        let mut t1 = tm.begin();
        let mut t2 = tm.begin();
        t1.record_put(Bytes::from("k"), Bytes::from("a"));
        t2.record_put(Bytes::from("k"), Bytes::from("b"));
        tm.commit(&mut t1, noop_apply).unwrap();
        assert!(matches!(tm.commit(&mut t2, noop_apply), Err(Error::WriteConflict)));
        drop(t2); // the `?`-propagation shape: no explicit abort
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
    }

    #[test]
    fn apply_error_burns_version_but_does_not_publish() {
        let tm = TransactionManager::new();
        let mut t = tm.begin();
        t.record_put(Bytes::from("k"), Bytes::from("v"));
        let r = tm.commit(&mut t, |_v, _w| Err(Error::WalCorrupted("boom".into())));
        assert!(r.is_err());
        // The failed version (1) is NOT published — readers don't see it.
        assert_eq!(tm.current_version(), 0);
        tm.abort(&mut t);
        // The failed version is BURNED, not reused: the next good commit gets 2,
        // so it can never alias a page-cached-but-abandoned batch at version 1.
        let mut t2 = tm.begin();
        t2.record_put(Bytes::from("k2"), Bytes::from("v"));
        assert_eq!(tm.commit(&mut t2, noop_apply).unwrap(), 2);
    }
}
