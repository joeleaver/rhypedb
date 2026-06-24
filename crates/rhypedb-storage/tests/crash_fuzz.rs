//! Crash-recovery fuzz harness for the LSM storage engine (Overboard cmqshgpnx).
//!
//! Gated on the `crash-fuzz` feature so the whole file vanishes from a default
//! build. Run with:
//!
//! ```text
//! cargo test -p rhypedb-storage --features crash-fuzz --test crash_fuzz
//! ```
//!
//! Inc 1 wires the injection sites and proves the simulated-crash → teardown →
//! cold-reopen loop is faithful (this file's single smoke test). Inc 2 adds the
//! seeded workload generator + the `verify_recovered` oracle on top of these
//! same helpers.
#![cfg(feature = "crash-fuzz")]

use bytes::Bytes;
use rhypedb_storage::crash_inject::{self, Caught, Mode, Site};
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_storage::SstCompression;
use std::path::Path;

/// Deterministic harness config: inline flush + no background compaction (the
/// workload thread is the only actor, so a crash is reproducible), fsync on
/// every commit (so "torn ⇒ discarded" is a deterministic outcome rather than
/// depending on the kernel's writeback timing), and a small memtable so a short
/// workload actually flushes.
fn harness_config(dir: &Path) -> LsmConfig {
    LsmConfig {
        data_dir: dir.to_path_buf(),
        memtable_flush_size: 1024,
        compact_trigger_ssts: usize::MAX, // no auto-compaction in Inc 1/2
        zone_extractor: None,
        sync_on_commit: true,
        background_compaction: false,
        block_compression: SstCompression::None,
    }
}

fn put(db: &LsmTree, key: &[u8], val: &[u8]) {
    let mut txn = db.begin_txn();
    db.put(&mut txn, key, Bytes::copy_from_slice(val)).unwrap();
    db.commit(&mut txn).unwrap();
}

fn get(db: &LsmTree, key: &[u8]) -> Option<Bytes> {
    db.get_at(db.read_snapshot(), key).unwrap()
}

/// The Inc-1 acceptance test: a crash injected mid-WAL-append unwinds the
/// in-flight commit, the teardown faithfully discards the un-flushed bytes (it
/// must NOT flush them — that would mask the data-loss class), the data-dir lock
/// is released so a cold reopen succeeds, and recovery keeps everything
/// committed before the crash while dropping the torn in-flight transaction.
#[test]
fn smoke_crash_in_wal_append_loses_only_the_in_flight_txn() {
    let dir = tempfile::tempdir().unwrap();

    let db = LsmTree::open(harness_config(dir.path())).unwrap();
    put(&db, b"a", b"1"); // durably committed before the crash

    // Crash on the next WAL append, before its BufWriter flush.
    crash_inject::arm(Site::WalAfterWriteBeforeFlush, 1, Mode::Crash);
    let outcome = crash_inject::catch_crash(|| {
        let mut txn = db.begin_txn();
        db.put(&mut txn, b"b", Bytes::from_static(b"2")).unwrap();
        let _ = db.commit(&mut txn); // unwinds inside append_txn
        unreachable!("commit must have crashed at the armed site");
    });
    assert_eq!(outcome, Caught::Crashed(Site::WalAfterWriteBeforeFlush));
    crash_inject::disarm();

    // Faithful teardown: discard the un-flushed WAL buffer, then drop (releases
    // the data-dir lock + closes fds without resurrecting the lost bytes).
    db.discard_for_crash_recovery();
    drop(db);

    // Cold reopen: the pre-crash commit survives; the in-flight txn is gone. Its
    // whole batch (a tiny payload, well under the 8 KiB BufWriter capacity) sat
    // in the un-flushed buffer the teardown discarded, so its footer never
    // reached the WAL and framed replay drops the txn whole. (A batch >= 8 KiB
    // could instead be page-cache-durable and survive — see the Site docs; Inc 2's
    // oracle predicts from framing, not payload size.)
    let recovered = LsmTree::open(harness_config(dir.path())).unwrap();
    assert_eq!(get(&recovered, b"a").as_deref(), Some(&b"1"[..]));
    assert_eq!(get(&recovered, b"b"), None, "in-flight txn must not survive");
}
