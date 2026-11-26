//! The `*_limited` scans count only LIVE keys toward their limit, across every
//! layer arrangement: tombstones in the same memtable as the rows they delete,
//! in a newer memtable or SST than those rows, and runs of them far longer than
//! the limit. Each case is checked against an unbounded scan of the same range.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;

use crate::lsm::{LsmConfig, LsmTree};
use crate::sst::SstCompression;

fn open(dir: &Path) -> Arc<LsmTree> {
    LsmTree::open(LsmConfig {
        data_dir: dir.to_path_buf(),
        // Large enough that only the explicit `flush()` calls below move data
        // out of the active memtable, so each test controls its layering.
        memtable_flush_size: 64 * 1024 * 1024,
        compact_trigger_ssts: usize::MAX,
        zone_extractor: None,
        sync_on_commit: false,
        background_compaction: false,
        block_compression: SstCompression::Lz4,
    })
    .unwrap()
}

/// A key under prefix `P:` whose 8-byte BE suffix is `id`, so key order is id
/// order (the shape of object and index keys: prefix-free, fixed width).
fn key(id: u64) -> Bytes {
    let mut k = b"P:".to_vec();
    k.extend_from_slice(&id.to_be_bytes());
    Bytes::from(k)
}

fn id_of(k: &[u8]) -> u64 {
    u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap())
}

fn put(tree: &LsmTree, ids: impl IntoIterator<Item = u64>) {
    for id in ids {
        let mut txn = tree.begin_txn();
        tree.put(&mut txn, &key(id), Bytes::from(format!("v{id}")))
            .unwrap();
        tree.commit(&mut txn).unwrap();
    }
}

fn delete(tree: &LsmTree, ids: impl IntoIterator<Item = u64>) {
    for id in ids {
        let mut txn = tree.begin_txn();
        tree.delete(&mut txn, &key(id)).unwrap();
        tree.commit(&mut txn).unwrap();
    }
}

fn snap(tree: &LsmTree) -> u64 {
    tree.txn_manager().current_version()
}

fn ids(rows: &[(Bytes, Bytes)]) -> Vec<u64> {
    rows.iter().map(|(k, _)| id_of(k)).collect()
}

/// The first `n` live ids of the whole `P:` range, from an unbounded scan.
fn first_live(tree: &LsmTree, n: usize) -> Vec<u64> {
    let all = tree.scan_prefix_at(snap(tree), b"P:").unwrap();
    ids(&all).into_iter().take(n).collect()
}

/// The first `n` live ids at or after `start`, from an unbounded scan.
fn first_live_from(tree: &LsmTree, start: u64, n: usize) -> Vec<u64> {
    let all = tree.scan_prefix_at(snap(tree), b"P:").unwrap();
    ids(&all)
        .into_iter()
        .filter(|id| *id >= start)
        .take(n)
        .collect()
}

fn limited(tree: &LsmTree, n: usize) -> Vec<u64> {
    ids(&tree.scan_prefix_at_limited(snap(tree), b"P:", n).unwrap())
}

fn limited_from(tree: &LsmTree, start: u64, n: usize) -> Vec<u64> {
    ids(&tree
        .scan_from_at_limited(snap(tree), b"P:", &key(start), n)
        .unwrap())
}

/// The reported bug's shape: rows in an SST, a few of the first ones deleted
/// later (tombstones in the memtable). `limit(N)` must still return N rows.
#[test]
fn tombstones_newer_than_their_rows_do_not_count_toward_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..204);
    tree.flush().unwrap();
    delete(&tree, [0, 1]);

    assert_eq!(limited(&tree, 201), (2..203).collect::<Vec<_>>());
    assert_eq!(limited(&tree, 300).len(), 202);
    assert_eq!(limited_from(&tree, 0, 10), (2..12).collect::<Vec<_>>());
}

/// Rows and their tombstones in the same (unflushed) memtable.
#[test]
fn tombstones_in_the_same_memtable_do_not_count_toward_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..20);
    delete(&tree, 0..15);

    assert_eq!(limited(&tree, 10), (15..20).collect::<Vec<_>>());
    assert_eq!(limited(&tree, 3), vec![15, 16, 17]);
}

/// Tombstones flushed into a NEWER SST than the rows they delete.
#[test]
fn tombstones_in_a_newer_sst_do_not_count_toward_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..50);
    tree.flush().unwrap();
    delete(&tree, 0..10);
    tree.flush().unwrap();

    assert_eq!(limited(&tree, 5), (10..15).collect::<Vec<_>>());
    assert_eq!(limited_from(&tree, 5, 5), (10..15).collect::<Vec<_>>());
}

/// A bounded scan may never return a later key while skipping an earlier live
/// one. Here the newest layer's window (tombstones 0 and 1, then live 50) ends
/// past keys 3 and 4, which only the older layer holds.
#[test]
fn never_skips_an_earlier_live_key_from_an_older_layer() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..20);
    tree.flush().unwrap();
    delete(&tree, [0, 1]);
    put(&tree, [50]);

    assert_eq!(limited(&tree, 3), vec![2, 3, 4]);
    assert_eq!(limited_from(&tree, 0, 3), vec![2, 3, 4]);
}

/// A tombstone run hundreds of times longer than the limit, split across
/// layers, still yields the first N live keys beyond it.
#[test]
fn walks_through_a_tombstone_run_far_longer_than_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..3000);
    tree.flush().unwrap();
    delete(&tree, 100..1500);
    tree.flush().unwrap();
    delete(&tree, 1500..2500);

    for n in [1, 5, 150, 700] {
        assert_eq!(limited(&tree, n), first_live(&tree, n), "limit {n}");
    }
    assert_eq!(limited_from(&tree, 100, 4), vec![2500, 2501, 2502, 2503]);
    assert_eq!(limited_from(&tree, 50, 60), first_live_from(&tree, 50, 60));
}

/// When the range holds fewer live keys than the limit, every one comes back,
/// and an all-tombstone range is empty.
#[test]
fn returns_every_live_key_when_the_limit_exceeds_them() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..40);
    tree.flush().unwrap();
    delete(&tree, (0..40).filter(|id| id % 3 != 0));

    assert_eq!(limited(&tree, 1000), first_live(&tree, 1000));
    assert_eq!(
        limited_from(&tree, 10, 1000),
        first_live_from(&tree, 10, 1000)
    );
    assert!(limited(&tree, 0).is_empty());

    delete(&tree, (0..40).filter(|id| id % 3 == 0));
    assert!(limited(&tree, 5).is_empty());
    assert!(limited_from(&tree, 0, 5).is_empty());
}

/// Only keys under `prefix` count, even when the seek point is near its end.
#[test]
fn stays_inside_the_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..10);
    let mut txn = tree.begin_txn();
    tree.put(&mut txn, b"Q:after", Bytes::from("x")).unwrap();
    tree.commit(&mut txn).unwrap();
    delete(&tree, 5..9);

    assert_eq!(limited_from(&tree, 4, 10), vec![4, 9]);
    assert_eq!(limited(&tree, 100).len(), 6);
}

/// Reads at an older snapshot see that snapshot's rows, not later deletes.
#[test]
fn honours_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    put(&tree, 0..10);
    let before = snap(&tree);
    delete(&tree, 0..5);

    let rows = tree.scan_prefix_at_limited(before, b"P:", 3).unwrap();
    assert_eq!(ids(&rows), vec![0, 1, 2]);
    assert_eq!(limited(&tree, 3), vec![5, 6, 7]);
}

/// Randomised: interleaved puts, deletes and flushes; after each round every
/// limit and seek point agrees with an unbounded scan.
#[test]
fn agrees_with_an_unbounded_scan_under_random_churn() {
    let dir = tempfile::tempdir().unwrap();
    let tree = open(dir.path());
    // xorshift64*: deterministic, no dev-dependency.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |bound: u64| {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D) % bound
    };
    for round in 0..12 {
        for _ in 0..150 {
            let id = next(400);
            if next(3) == 0 {
                put(&tree, [id])
            } else {
                delete(&tree, [id])
            }
        }
        put(&tree, (0..40).map(|_| next(400)));
        if round % 3 != 2 {
            tree.flush().unwrap();
        }
        for n in [1, 2, 7, 33, 120, 500] {
            assert_eq!(
                limited(&tree, n),
                first_live(&tree, n),
                "round {round} limit {n}"
            );
            let start = next(400);
            assert_eq!(
                limited_from(&tree, start, n),
                first_live_from(&tree, start, n),
                "round {round} start {start} limit {n}"
            );
        }
    }
}
