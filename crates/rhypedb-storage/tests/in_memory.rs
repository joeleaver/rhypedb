//! `LsmTree::open_in_memory`: flushes become SSTs held as bytes, compaction
//! runs inline, and reads see exactly what an on-disk tree would, with no
//! file created anywhere.

use bytes::Bytes;
use rhypedb_storage::lsm::{LsmConfig, LsmTree};

fn key(i: u32) -> Vec<u8> {
    format!("k{i:06}").into_bytes()
}

#[test]
fn in_memory_tree_flushes_compacts_and_reads() {
    let mut config = LsmConfig::new("must-not-be-created");
    config.memtable_flush_size = 16 * 1024;
    config.compact_trigger_ssts = 3;
    let tree = LsmTree::open_in_memory(config).unwrap();
    assert!(tree.is_in_memory());

    // Three rounds over the same keys: puts, overwrites, deletes of some.
    for round in 0..3u32 {
        for i in 0..2000u32 {
            let mut txn = tree.begin_txn();
            if round == 2 && i % 5 == 0 {
                tree.delete(&mut txn, &key(i)).unwrap();
            } else {
                tree.put(&mut txn, &key(i), Bytes::from(format!("v{round}-{i}"))).unwrap();
            }
            tree.commit(&mut txn).unwrap();
        }
    }
    assert!(tree.sst_count() >= 1, "flushes produced in-memory SSTs");

    let check = |tree: &LsmTree| {
        let snap = tree.read_snapshot();
        for i in 0..2000u32 {
            let got = tree.get_at(snap, &key(i)).unwrap();
            if i % 5 == 0 {
                assert_eq!(got, None, "key {i} was deleted");
            } else {
                assert_eq!(got, Some(Bytes::from(format!("v2-{i}"))));
            }
        }
        let live = tree.scan_prefix_at(snap, b"k").unwrap();
        assert_eq!(live.len(), 1600);
    };
    check(&tree);
    tree.flush().unwrap();
    tree.compact().unwrap();
    assert_eq!(tree.sst_count(), 1, "a full compaction leaves one table");
    check(&tree);

    assert!(tree.snapshot_to(std::path::Path::new("unused")).is_err());
    drop(tree);
    assert!(!std::path::Path::new("must-not-be-created").exists());
}
