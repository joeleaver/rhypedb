//! Crash-recovery fuzz harness for the LSM storage engine (Overboard cmqshgpnx).
//!
//! Gated on the `crash-fuzz` feature so the whole file vanishes from a default
//! build. Run with:
//!
//! ```text
//! cargo test -p rhypedb-storage --features crash-fuzz --test crash_fuzz
//! ```
//!
//! - Inc 1 wired the injection sites and proved the simulated-crash → teardown →
//!   cold-reopen loop is faithful (`smoke_*`).
//! - Inc 2 (this) adds the seeded deterministic workload generator + the
//!   `verify_recovered` oracle, sweeps every WAL/flush boundary across seeds,
//!   and a TornTail power-loss test.
//!
//! ## Oracle model
//!
//! A workload commits transactions over a small fixed keyspace, recording the
//! last committed value per key in a SHADOW model (updated only after
//! `commit()` returns `Ok`). One injected crash interrupts either a commit (WAL
//! sites) or an explicit flush (flush sites). After teardown + cold reopen, the
//! oracle asserts:
//!
//! - **Flush-site crash:** the recovered keyspace equals the shadow EXACTLY — a
//!   flush is never allowed to lose, duplicate, or resurrect committed data.
//! - **WAL-site crash:** the in-flight transaction is ATOMIC — the recovered
//!   keyspace equals the shadow (txn reverted) OR the shadow with the in-flight
//!   writes applied (txn survived). Any other result is a torn/partial write and
//!   fails. We accept either outcome rather than predict which, because that is
//!   exactly the crash-atomicity property (and survival is payload-dependent —
//!   see `Site::WalAfterWriteBeforeFlush`).
//!
//! Plus: recovery is IDEMPOTENT (a second cold reopen yields the identical
//! state, catching double-replay), the recovered tree is writable and its
//! versioning advanced past recovery (a probe commit), and the data-dir lock was
//! released (the reopens succeed at all).
#![cfg(feature = "crash-fuzz")]

use bytes::Bytes;
use rhypedb_storage::crash_inject::{self, Caught, Mode, Site};
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_storage::SstCompression;
use std::panic::AssertUnwindSafe;
use std::path::Path;

/// Keys are `k0000`..`k0031` — a small bounded keyspace so the oracle can read
/// the entire recovered state by enumeration (catching lost AND resurrected
/// keys), and so updates/tombstones/resurrection are all exercised.
const KEYSPACE: u64 = 32;

/// Deterministic harness config: NO auto-flush (`memtable_flush_size =
/// usize::MAX`) and no background compaction — the workload drives commits and
/// flushes explicitly, so the only actor is the test thread and the crash point
/// is exactly controlled. fsync on every commit, so "torn ⇒ discarded" is a
/// deterministic outcome rather than depending on kernel writeback timing.
fn harness_config(dir: &Path) -> LsmConfig {
    LsmConfig {
        data_dir: dir.to_path_buf(),
        memtable_flush_size: usize::MAX, // never auto-flush; flushes are explicit
        compact_trigger_ssts: usize::MAX, // no auto-compaction in Inc 1/2
        zone_extractor: None,
        sync_on_commit: true,
        background_compaction: false,
        block_compression: SstCompression::None,
    }
}

fn key(i: u64) -> Vec<u8> {
    format!("k{i:04}").into_bytes()
}

// ---------------------------------------------------------------------------
// Deterministic workload generator (SplitMix64 — reproducible forever, no dep
// on an external RNG whose algorithm could change a stored corpus).
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A transaction's writes: `(key index, Some(value) | None==tombstone)`.
type Writes = Vec<(u64, Option<Bytes>)>;

fn gen_writes(rng: &mut Rng) -> Writes {
    let n = 1 + rng.below(3); // 1..=3 writes per txn
    let mut out = Writes::new();
    for _ in 0..n {
        let idx = rng.below(KEYSPACE);
        if rng.below(10) < 3 {
            out.push((idx, None)); // tombstone
        } else {
            let v = Bytes::from(format!("v{}", rng.next_u64()));
            out.push((idx, Some(v)));
        }
    }
    out
}

fn apply_writes(db: &LsmTree, writes: &Writes) {
    let mut txn = db.begin_txn();
    for (idx, val) in writes {
        match val {
            Some(v) => db.put(&mut txn, &key(*idx), v.clone()).unwrap(),
            None => db.delete(&mut txn, &key(*idx)).unwrap(),
        }
    }
    db.commit(&mut txn).unwrap();
}

/// Apply a committed transaction's writes to the shadow (same-key last-write
/// wins, matching the memtable + recovery semantics).
fn apply_to_shadow(shadow: &mut [Option<Bytes>], writes: &Writes) {
    for (idx, val) in writes {
        shadow[*idx as usize] = val.clone();
    }
}

fn commit_random_txn(db: &LsmTree, rng: &mut Rng, shadow: &mut [Option<Bytes>]) {
    let writes = gen_writes(rng);
    apply_writes(db, &writes);
    apply_to_shadow(shadow, &writes);
}

fn read_keyspace(db: &LsmTree) -> Vec<Option<Bytes>> {
    let snap = db.read_snapshot();
    (0..KEYSPACE).map(|i| db.get_at(snap, &key(i)).unwrap()).collect()
}

fn is_flush_site(s: Site) -> bool {
    matches!(
        s,
        Site::FlushAfterMemtableRotate
            | Site::FlushAfterSstFinish
            | Site::FlushAfterSstRegister
            | Site::FlushBeforeWalTruncate
            | Site::FlushAfterWalTruncate
    )
}

/// `Commit` footer record size: 21-byte header + 0-byte key + 8-byte count.
/// Tearing this many bytes off the WAL tail un-terminates the trailing batch.
const FOOTER_BYTES: u64 = 21 + 8;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// The expected fate of the in-flight transaction after recovery. Unlike a
/// lenient "reverted OR survived" disjunction, each fuzz case asserts the EXACT
/// arm its site + fault deterministically produce — so a recovery bug that picks
/// the WRONG atomic arm (drops a durable txn, or resurrects a discarded one) is
/// caught, not aliased onto an accepted alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// In-flight txn did NOT survive — recovered keyspace == shadow.
    Reverted,
    /// In-flight txn survived — recovered keyspace == shadow + in-flight writes.
    Survived,
}

/// Cold-reopen the data dir and assert recovery. The recovered keyspace must
/// EXACTLY equal the expected state (`expect` selects reverted vs survived);
/// recovery must be idempotent; and — the broad guard for the torn-WAL-tail
/// cleanup — a commit made AFTER recovery must survive a further reopen.
fn verify_recovered(
    dir: &Path,
    shadow: &[Option<Bytes>],
    in_flight: Option<&Writes>,
    expect: Expect,
    seed: u64,
    site: Site,
) {
    let want = {
        let mut s = shadow.to_vec();
        if let (Expect::Survived, Some(wf)) = (expect, in_flight) {
            apply_to_shadow(&mut s, wf);
        }
        s
    };

    // First cold reopen (the reopen succeeding proves the data-dir lock was
    // released by the simulated crash's teardown).
    let db1 = LsmTree::open(harness_config(dir)).unwrap();
    let rec = read_keyspace(&db1);
    assert_eq!(
        rec, want,
        "recovered state != expected {expect:?} (seed={seed} site={site:?})\n  rec ={rec:?}\n  want={want:?}"
    );
    drop(db1);

    // Second cold reopen: recovery must be idempotent (catches non-idempotent
    // double-replay, e.g. an SST-durable-but-WAL-not-truncated crash).
    let db2 = LsmTree::open(harness_config(dir)).unwrap();
    let rec2 = read_keyspace(&db2);
    assert_eq!(
        rec2, rec,
        "recovery is not idempotent across a second reopen (seed={seed} site={site:?})"
    );

    // Post-recovery durability: a commit made AFTER recovery must survive a
    // further cold reopen. A torn tail left in the WAL would conflate this
    // commit's framing and drop it on reopen — this asserts the tail was cleaned.
    // (`__after__` sorts before the `k####` keyspace, so it never perturbs the
    // enumeration; it also proves the recovered tree is writable + advancing.)
    let sentinel: &[u8] = b"__after__";
    {
        let mut t = db2.begin_txn();
        db2.put(&mut t, sentinel, Bytes::from_static(b"durable")).unwrap();
        db2.commit(&mut t).unwrap();
    }
    assert_eq!(
        db2.get_at(db2.read_snapshot(), sentinel).unwrap().as_deref(),
        Some(&b"durable"[..])
    );
    drop(db2);

    let db3 = LsmTree::open(harness_config(dir)).unwrap();
    assert_eq!(
        db3.get_at(db3.read_snapshot(), sentinel).unwrap().as_deref(),
        Some(&b"durable"[..]),
        "post-recovery commit lost on reopen — torn WAL tail not cleaned (seed={seed} site={site:?})"
    );
    assert_eq!(
        read_keyspace(&db3),
        want,
        "keyspace changed after a post-recovery commit+reopen (seed={seed} site={site:?})"
    );
}

/// One fuzz case: build a deterministic preamble of durable commits + flushes,
/// inject a crash at `site` during the next commit (WAL site) or flush (flush
/// site), tear down, and verify recovery against the EXACT expected outcome.
///
/// `torn` (WAL sites only) additionally lops the footer off the in-flight
/// batch's un-fsync'd tail, modelling power loss — so framed replay must discard
/// the torn batch. This is the only path that exercises the torn-batch-discard
/// branch (small batches are otherwise strictly all-or-nothing on disk).
fn run_case(seed: u64, site: Site, torn: bool) {
    let dir = tempfile::tempdir().unwrap();
    let mut shadow: Vec<Option<Bytes>> = vec![None; KEYSPACE as usize];
    let mut rng = Rng::new(seed);

    let db = LsmTree::open(harness_config(dir.path())).unwrap();

    // Preamble: deterministic durable history. Explicit flushes (~20%) move data
    // into SSTs and truncate the WAL, so recovery exercises SST+WAL reconciliation
    // — not just a flat WAL replay.
    let n_ops = 3 + rng.below(28);
    for _ in 0..n_ops {
        if rng.below(10) < 2 {
            db.flush().unwrap();
        } else {
            commit_random_txn(&db, &mut rng, &mut shadow);
        }
    }

    let wal_path = dir.path().join("wal.log");

    if is_flush_site(site) {
        assert!(!torn, "the torn axis is WAL-only");
        // Dirty the memtable so flush_locked gets past its is_empty short-circuit
        // and reaches the armed site.
        commit_random_txn(&db, &mut rng, &mut shadow);
        crash_inject::arm(site, 1, Mode::Crash);
        let outcome = crash_inject::catch_crash(AssertUnwindSafe(|| {
            let _ = db.flush();
            unreachable!("flush must have crashed at the armed site");
        }));
        assert_eq!(outcome, Caught::Crashed(site), "expected a crash at {site:?} (seed={seed})");
        crash_inject::disarm();
        db.discard_for_crash_recovery();
        drop(db);
        // A flush never loses, duplicates, or resurrects committed data.
        verify_recovered(dir.path(), &shadow, None, Expect::Reverted, seed, site);
    } else {
        let writes = gen_writes(&mut rng);
        let len_before = std::fs::metadata(&wal_path).unwrap().len();
        crash_inject::arm(site, 1, Mode::Crash);
        let outcome = crash_inject::catch_crash(AssertUnwindSafe(|| {
            apply_writes(&db, &writes);
            unreachable!("commit must have crashed at the armed site");
        }));
        assert_eq!(outcome, Caught::Crashed(site), "expected a crash at {site:?} (seed={seed})");
        crash_inject::disarm();
        // Faithful teardown: discard the un-flushed WAL buffer (a no-op once the
        // batch reached the page cache), then drop (releases the lock + fds).
        db.discard_for_crash_recovery();
        drop(db);

        let expect = if torn {
            // Power loss on the un-fsync'd tail: the in-flight batch (incl. footer)
            // reached the page cache at WalAfterFlushBeforeFsync; tearing the footer
            // off un-terminates it so framed replay discards the whole batch.
            let len_after = std::fs::metadata(&wal_path).unwrap().len();
            assert!(
                len_after >= len_before + FOOTER_BYTES,
                "in-flight batch too small to tear a footer (seed={seed})"
            );
            let f = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            f.set_len(len_after - FOOTER_BYTES).unwrap();
            drop(f);
            Expect::Reverted
        } else {
            match site {
                // Small batch sat in the un-flushed buffer -> discarded by teardown.
                Site::WalAfterWriteBeforeFlush => Expect::Reverted,
                // Batch reached the page cache (and fsync, for the latter) before
                // the in-process kill -> survives a SIGKILL-equivalent crash.
                Site::WalAfterFlushBeforeFsync | Site::WalAfterFsync => Expect::Survived,
                _ => unreachable!("non-WAL site in the WAL branch"),
            }
        };
        verify_recovered(dir.path(), &shadow, Some(&writes), expect, seed, site);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The full boundary sweep across a seed corpus, asserting the EXACT recovery
/// outcome per site. A failing case prints its seed + site for exact replay.
#[test]
fn fuzz_wal_and_flush_boundaries_recover() {
    const WAL_SITES: [Site; 3] = [
        Site::WalAfterWriteBeforeFlush,
        Site::WalAfterFlushBeforeFsync,
        Site::WalAfterFsync,
    ];
    const FLUSH_SITES: [Site; 5] = [
        Site::FlushAfterMemtableRotate,
        Site::FlushAfterSstFinish,
        Site::FlushAfterSstRegister,
        Site::FlushBeforeWalTruncate,
        Site::FlushAfterWalTruncate,
    ];
    for seed in 0..16u64 {
        for site in WAL_SITES {
            run_case(seed, site, false);
        }
        // Torn power-loss axis (WAL only): tear the footer off an un-fsync'd
        // in-flight batch so framed replay MUST discard it. This is the only path
        // that drives the torn-batch-discard branch — small batches are otherwise
        // strictly all-or-nothing on disk, so the sweep would never exercise it.
        run_case(seed, Site::WalAfterFlushBeforeFsync, true);
        for site in FLUSH_SITES {
            run_case(seed, site, false);
        }
    }
}

/// Power-loss: a transaction whose WAL bytes reached the OS page cache but were
/// NOT yet fsync'd (crash at `WalAfterFlushBeforeFsync`) is lost if power fails
/// before the kernel writes the tail to the platter. Model that by truncating
/// the un-fsync'd tail, then assert framed replay discards the torn transaction
/// WHOLE while every fsync'd commit before it survives.
#[test]
fn torn_tail_power_loss_discards_unfsynced_in_flight_txn() {
    let dir = tempfile::tempdir().unwrap();
    let mut shadow: Vec<Option<Bytes>> = vec![None; KEYSPACE as usize];
    let mut rng = Rng::new(0x70726E); // "prn"

    let db = LsmTree::open(harness_config(dir.path())).unwrap();
    // Durable, fsync'd committed history.
    for _ in 0..10 {
        commit_random_txn(&db, &mut rng, &mut shadow);
    }

    let wal_path = dir.path().join("wal.log");
    let len_before = std::fs::metadata(&wal_path).unwrap().len();

    // Crash after the in-flight txn's full batch reached the page cache (the
    // append_txn write_all + flush ran) but before sync_all.
    let writes = gen_writes(&mut rng);
    crash_inject::arm(Site::WalAfterFlushBeforeFsync, 1, Mode::Crash);
    let outcome = crash_inject::catch_crash(AssertUnwindSafe(|| {
        apply_writes(&db, &writes);
        unreachable!("commit must have crashed before fsync");
    }));
    assert_eq!(outcome, Caught::Crashed(Site::WalAfterFlushBeforeFsync));
    crash_inject::disarm();

    db.discard_for_crash_recovery();
    drop(db);

    // The in-flight batch (data records + footer) extended the WAL past the last
    // fsync. Power loss: lop off the footer so the trailing batch is unterminated.
    let len_after = std::fs::metadata(&wal_path).unwrap().len();
    assert!(
        len_after >= len_before + FOOTER_BYTES,
        "in-flight txn should have appended at least a footer ({len_before}..{len_after})"
    );
    let f = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
    f.set_len(len_after - FOOTER_BYTES).unwrap(); // tear off the footer, stay within the in-flight batch
    drop(f);

    // Recovery: the torn in-flight txn is dropped wholesale; everything fsync'd
    // before it survives exactly.
    let recovered = LsmTree::open(harness_config(dir.path())).unwrap();
    let rec = read_keyspace(&recovered);
    assert_eq!(
        rec, shadow,
        "torn power-loss tail must discard exactly the in-flight txn"
    );
}

/// Regression test for a data-loss bug FOUND BY THIS HARNESS: recovering from a
/// torn (un-footered) WAL left the orphaned records in the file, so the next
/// committed transaction was appended after them and a later replay folded them
/// into its footer-count check — silently dropping a durably-fsync'd commit (and
/// resurrecting the stale prior value) on a second crash. Fixed by truncating
/// the WAL to its valid prefix on recovery (Wal::replay_with_valid_len +
/// LsmTree::open). Without the fix, `k0007` reads back as a stale preamble value
/// after the second reopen.
#[test]
fn probe_commit_after_torn_recovery_survives_second_crash() {
    let dir = tempfile::tempdir().unwrap();
    let mut shadow: Vec<Option<Bytes>> = vec![None; KEYSPACE as usize];
    let mut rng = Rng::new(1);

    let db = LsmTree::open(harness_config(dir.path())).unwrap();
    for _ in 0..5 {
        commit_random_txn(&db, &mut rng, &mut shadow);
    }

    let wal_path = dir.path().join("wal.log");
    let len_before = std::fs::metadata(&wal_path).unwrap().len();

    // Produce a torn (un-footered) trailing batch on disk.
    let writes = gen_writes(&mut rng);
    crash_inject::arm(Site::WalAfterFlushBeforeFsync, 1, Mode::Crash);
    let outcome = crash_inject::catch_crash(AssertUnwindSafe(|| {
        apply_writes(&db, &writes);
        unreachable!();
    }));
    assert_eq!(outcome, Caught::Crashed(Site::WalAfterFlushBeforeFsync));
    crash_inject::disarm();
    db.discard_for_crash_recovery();
    drop(db);
    let len_after = std::fs::metadata(&wal_path).unwrap().len();
    assert!(len_after >= len_before + FOOTER_BYTES);
    let f = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
    f.set_len(len_after - FOOTER_BYTES).unwrap(); // tear off footer -> torn trailing batch on disk
    drop(f);

    // Recover from the torn WAL, then commit a NEW txn.
    let db2 = LsmTree::open(harness_config(dir.path())).unwrap();
    let mut t = db2.begin_txn();
    db2.put(&mut t, b"k0007", Bytes::from_static(b"AFTER_RECOVERY")).unwrap();
    db2.commit(&mut t).unwrap();
    assert_eq!(
        db2.get_at(db2.read_snapshot(), b"k0007").unwrap().as_deref(),
        Some(&b"AFTER_RECOVERY"[..]),
        "present before reopen"
    );
    drop(db2);

    // SECOND cold reopen (no flush happened): does the post-recovery commit survive?
    let db3 = LsmTree::open(harness_config(dir.path())).unwrap();
    let got = db3.get_at(db3.read_snapshot(), b"k0007").unwrap();
    assert_eq!(
        got.as_deref(),
        Some(&b"AFTER_RECOVERY"[..]),
        "post-torn-recovery commit LOST on second crash (got {got:?}) -- WAL torn-tail not cleaned on recovery"
    );
}

/// Inc-1 acceptance (kept): a crash injected mid-WAL-append unwinds the in-flight
/// commit, the teardown faithfully discards the un-flushed bytes (it must NOT
/// flush them — that would mask the data-loss class), the data-dir lock is
/// released so a cold reopen succeeds, and recovery keeps everything committed
/// before the crash while dropping the torn in-flight transaction.
#[test]
fn smoke_crash_in_wal_append_loses_only_the_in_flight_txn() {
    let dir = tempfile::tempdir().unwrap();

    let db = LsmTree::open(harness_config(dir.path())).unwrap();
    {
        let mut txn = db.begin_txn();
        db.put(&mut txn, b"a", Bytes::from_static(b"1")).unwrap();
        db.commit(&mut txn).unwrap(); // durably committed before the crash
    }

    // Crash on the next WAL append, before its BufWriter flush.
    crash_inject::arm(Site::WalAfterWriteBeforeFlush, 1, Mode::Crash);
    let outcome = crash_inject::catch_crash(AssertUnwindSafe(|| {
        let mut txn = db.begin_txn();
        db.put(&mut txn, b"b", Bytes::from_static(b"2")).unwrap();
        let _ = db.commit(&mut txn); // unwinds inside append_txn
        unreachable!("commit must have crashed at the armed site");
    }));
    assert_eq!(outcome, Caught::Crashed(Site::WalAfterWriteBeforeFlush));
    crash_inject::disarm();

    db.discard_for_crash_recovery();
    drop(db);

    // Cold reopen: the pre-crash commit survives; the in-flight txn is gone. Its
    // whole batch (a tiny payload, well under the 8 KiB BufWriter capacity) sat
    // in the un-flushed buffer the teardown discarded, so its footer never
    // reached the WAL and framed replay drops the txn whole. (A batch >= 8 KiB
    // could instead be page-cache-durable and survive — see the Site docs; the
    // fuzz oracle predicts from framing/atomicity, not payload size.)
    let recovered = LsmTree::open(harness_config(dir.path())).unwrap();
    let snap = recovered.read_snapshot();
    assert_eq!(recovered.get_at(snap, b"a").unwrap().as_deref(), Some(&b"1"[..]));
    assert_eq!(
        recovered.get_at(snap, b"b").unwrap(),
        None,
        "in-flight txn must not survive"
    );
}
