use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use parking_lot::{Mutex, RwLock};

use crate::{Error, Result};

/// Tracks committed transactions for write-write conflict detection.
struct CommittedTxn {
    commit_version: u64,
    write_set: HashSet<Bytes>,
}

/// MVCC transaction manager.
///
/// Assigns monotonically increasing versions, tracks active transactions,
/// and detects write-write conflicts at commit time.
pub struct TransactionManager {
    next_version: AtomicU64,
    active_snapshots: RwLock<HashSet<u64>>,
    committed_log: Mutex<VecDeque<CommittedTxn>>,
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
            next_version: AtomicU64::new(1),
            active_snapshots: RwLock::new(HashSet::new()),
            committed_log: Mutex::new(VecDeque::new()),
            max_committed_log_size: 1024,
        }
    }

    /// Create a transaction manager that resumes from a known version.
    /// Used during recovery to restore the version counter from WAL/SST state.
    pub fn recover_at_version(version: u64) -> Self {
        Self {
            next_version: AtomicU64::new(version + 1),
            active_snapshots: RwLock::new(HashSet::new()),
            committed_log: Mutex::new(VecDeque::new()),
            max_committed_log_size: 1024,
        }
    }

    /// Start a new transaction, returning its snapshot version.
    pub fn begin(&self) -> Transaction {
        let snapshot = self.next_version.load(Ordering::SeqCst) - 1;
        self.active_snapshots.write().insert(snapshot);
        Transaction {
            snapshot,
            write_set: HashSet::new(),
            committed: false,
        }
    }

    /// Attempt to commit a transaction. Returns the commit version on success.
    ///
    /// Checks for write-write conflicts: if any key in the transaction's write set
    /// was modified by a transaction that committed after this transaction's snapshot,
    /// the commit is rejected.
    pub fn commit(&self, txn: &mut Transaction) -> Result<u64> {
        if txn.write_set.is_empty() {
            txn.committed = true;
            self.active_snapshots.write().remove(&txn.snapshot);
            return Ok(txn.snapshot);
        }

        let committed_log = self.committed_log.lock();

        // Check for conflicts against all transactions committed after our snapshot.
        for committed in committed_log.iter() {
            if committed.commit_version <= txn.snapshot {
                continue;
            }
            for key in &txn.write_set {
                if committed.write_set.contains(key) {
                    // Conflict detected — don't modify any state, just return error.
                    // The caller will abort.
                    return Err(Error::WriteConflict);
                }
            }
        }

        drop(committed_log);

        // No conflict — assign a commit version.
        let commit_version = self.next_version.fetch_add(1, Ordering::SeqCst);

        // Record this transaction in the committed log.
        let mut committed_log = self.committed_log.lock();
        committed_log.push_back(CommittedTxn {
            commit_version,
            write_set: txn.write_set.clone(),
        });

        // Trim old entries that are no longer needed for conflict detection.
        let min_active = self.min_active_snapshot();
        while let Some(front) = committed_log.front() {
            if front.commit_version < min_active {
                committed_log.pop_front();
            } else {
                break;
            }
        }

        // Cap the log size as a safety measure.
        while committed_log.len() > self.max_committed_log_size {
            committed_log.pop_front();
        }

        drop(committed_log);

        txn.committed = true;
        self.active_snapshots.write().remove(&txn.snapshot);

        Ok(commit_version)
    }

    /// Abort a transaction, releasing its snapshot.
    pub fn abort(&self, txn: &mut Transaction) {
        txn.committed = true; // prevent double-release
        self.active_snapshots.write().remove(&txn.snapshot);
    }

    /// Returns the minimum active snapshot version, or u64::MAX if none active.
    /// Used for determining which old versions can be garbage-collected during compaction.
    pub fn min_active_snapshot(&self) -> u64 {
        self.active_snapshots
            .read()
            .iter()
            .copied()
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Current version (latest committed).
    pub fn current_version(&self) -> u64 {
        self.next_version.load(Ordering::SeqCst) - 1
    }
}

/// A transaction handle. Tracks the snapshot version and write set.
pub struct Transaction {
    snapshot: u64,
    write_set: HashSet<Bytes>,
    committed: bool,
}

impl Transaction {
    pub fn snapshot(&self) -> u64 {
        self.snapshot
    }

    /// Record a key as written by this transaction.
    pub fn record_write(&mut self, key: Bytes) {
        self.write_set.insert(key);
    }

    pub fn write_set(&self) -> &HashSet<Bytes> {
        &self.write_set
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.committed && !self.write_set.is_empty() {
            eprintln!(
                "WARNING: read-write transaction with snapshot {} dropped without commit/abort",
                self.snapshot
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_commit() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        txn.record_write(Bytes::from("key1"));
        let version = tm.commit(&mut txn).unwrap();
        assert!(version > 0);
    }

    #[test]
    fn no_conflict_on_disjoint_writes() {
        let tm = TransactionManager::new();

        let mut txn1 = tm.begin();
        let mut txn2 = tm.begin();

        txn1.record_write(Bytes::from("key1"));
        txn2.record_write(Bytes::from("key2"));

        tm.commit(&mut txn1).unwrap();
        tm.commit(&mut txn2).unwrap(); // should succeed — different keys
    }

    #[test]
    fn conflict_on_overlapping_writes() {
        let tm = TransactionManager::new();

        let mut txn1 = tm.begin();
        let mut txn2 = tm.begin();

        txn1.record_write(Bytes::from("key1"));
        txn2.record_write(Bytes::from("key1")); // same key

        tm.commit(&mut txn1).unwrap();
        let result = tm.commit(&mut txn2);
        assert!(matches!(result, Err(Error::WriteConflict)));
        tm.abort(&mut txn2);
    }

    #[test]
    fn later_snapshot_sees_committed_writes() {
        let tm = TransactionManager::new();

        // Commit a write.
        let mut txn1 = tm.begin();
        txn1.record_write(Bytes::from("key1"));
        let v1 = tm.commit(&mut txn1).unwrap();

        // New transaction starts after commit — no conflict.
        let mut txn2 = tm.begin();
        txn2.record_write(Bytes::from("key1"));
        let v2 = tm.commit(&mut txn2).unwrap();
        assert!(v2 > v1);
    }

    #[test]
    fn read_only_transactions_always_commit() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        // No writes recorded.
        let version = tm.commit(&mut txn).unwrap();
        assert_eq!(version, txn.snapshot);
    }

    #[test]
    fn min_active_snapshot_tracking() {
        let tm = TransactionManager::new();

        let txn1 = tm.begin();
        let _txn2 = tm.begin();

        assert_eq!(tm.min_active_snapshot(), txn1.snapshot());

        // Note: these leak because we don't abort, but the test only
        // checks the tracking logic.
    }

    #[test]
    fn abort_releases_snapshot() {
        let tm = TransactionManager::new();
        let mut txn = tm.begin();
        tm.abort(&mut txn);
        assert_eq!(tm.min_active_snapshot(), u64::MAX);
    }
}
