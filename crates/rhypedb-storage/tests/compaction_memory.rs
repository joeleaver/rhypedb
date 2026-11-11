//! GH #28 regression guard: compaction's heap use must not scale with the
//! amount of data being merged.
//!
//! The old compaction collected every entry of every input SST into one `Vec`
//! (64 bytes per entry), sorted it, and kept a copy of every user key for the
//! bloom filter until the output was finished — ~100 bytes of heap per entry,
//! plus (for LZ4 files) every decompressed block kept alive by the entry
//! slices. A 275 MB database pushed the heap from 150 to 333 MB.
//!
//! This binary installs a counting global allocator and measures the PEAK heap
//! growth during `compact()` over ~400k entries in 4 SSTs: the old code would
//! need ~45 MB; the streaming merge needs roughly one block per input plus the
//! output's sparse index and bloom filter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_storage::sst::SstCompression;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let now = CURRENT.fetch_add(new_size - layout.size(), Ordering::Relaxed)
                    + (new_size - layout.size());
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const SSTS: u32 = 4;
const PER_SST: u32 = 100_000;

fn build(dir: &std::path::Path, compression: SstCompression) -> std::sync::Arc<LsmTree> {
    let mut cfg = LsmConfig::new(dir);
    cfg.memtable_flush_size = usize::MAX;
    cfg.compact_trigger_ssts = usize::MAX;
    cfg.background_compaction = false;
    cfg.sync_on_commit = false;
    cfg.block_compression = compression;
    let tree = LsmTree::open(cfg).unwrap();
    for s in 0..SSTS {
        for chunk in 0..(PER_SST / 10_000) {
            let mut txn = tree.begin_txn();
            for j in 0..10_000u32 {
                // Interleave the SSTs' key ranges so the merge really merges.
                let i = (chunk * 10_000 + j) * SSTS + s;
                let key = format!("o:{i:010}");
                tree.put(&mut txn, key.as_bytes(), Bytes::from(format!("value-{i:012}")))
                    .unwrap();
            }
            tree.commit(&mut txn).unwrap();
        }
        tree.flush().unwrap();
    }
    assert_eq!(tree.sst_count(), SSTS as usize);
    tree
}

fn check(compression: SstCompression) {
    let dir = tempfile::tempdir().unwrap();
    let tree = build(dir.path(), compression);

    let base = CURRENT.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    tree.compact().unwrap();
    let peak_growth = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let after = CURRENT.load(Ordering::Relaxed);
    assert_eq!(tree.sst_count(), 1);

    let entries = (SSTS * PER_SST) as usize;
    eprintln!(
        "{compression:?}: {entries} entries, compact() peak heap growth {:.1} MB, \
         retained after {:+.1} MB",
        peak_growth as f64 / 1e6,
        (after as f64 - base as f64) / 1e6
    );
    // Old implementation: >= 64 B/entry (Vec) + ~48 B/entry (bloom key copies)
    // ≈ 45 MB here. The streaming merge stays a small fraction of that.
    assert!(
        peak_growth < 8_000_000,
        "{compression:?}: compaction heap peak grew {peak_growth} bytes for {entries} entries"
    );
    // The merged reader's index/bloom replace the inputs' — no net growth.
    assert!(
        after <= base + 2_000_000,
        "{compression:?}: heap grew by {} bytes across compaction",
        after - base
    );

    // Data still intact.
    let snap = tree.read_snapshot();
    for i in [0u32, 1, 2, 3, 12_345, 399_999] {
        assert_eq!(
            tree.get_at(snap, format!("o:{i:010}").as_bytes()).unwrap().as_deref(),
            Some(format!("value-{i:012}").as_bytes())
        );
    }
}

// One test per binary run order would interleave allocator counts across
// threads; run both compressions sequentially in a single test.
#[test]
fn compaction_heap_does_not_scale_with_data() {
    check(SstCompression::None);
    check(SstCompression::Lz4);
}
