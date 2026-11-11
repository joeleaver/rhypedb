//! The memtable's size estimate drives the flush threshold, so it must track
//! the memory the memtable really holds. It used to count only key + value
//! bytes: for small rows (index entries, full-text postings) every entry's
//! skiplist node and separate key/value allocations went uncounted, and a
//! "4 MB" memtable really held several times that.
//!
//! A counting global allocator measures the true heap per entry and checks
//! `approximate_size` against it across small and large rows. (The allocator
//! sees requested sizes, not malloc's chunk headers, so the estimate — which
//! includes them — runs a little above this "real" figure for tiny rows.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use rhypedb_storage::memtable::MemTable;

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            CURRENT.fetch_add(layout.size(), Ordering::Relaxed);
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
                CURRENT.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// (real heap bytes, estimated bytes) for `n` puts of `key_len`/`val_len`.
fn measure(n: usize, key_len: usize, val_len: usize) -> (usize, usize) {
    // Build inputs first so only the memtable's own holdings are counted —
    // except the values, which the memtable keeps alive (it owns them once
    // the caller's transaction is gone), so they are made inside the window.
    let keys: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let mut k = format!("{i:0width$}", width = key_len).into_bytes();
            k.truncate(key_len);
            k
        })
        .collect();
    let base = CURRENT.load(Ordering::Relaxed);
    let mt = MemTable::new();
    for (i, k) in keys.iter().enumerate() {
        let value = Bytes::from(vec![b'v'; val_len]);
        mt.put(k, i as u64 + 1, value);
    }
    let real = CURRENT.load(Ordering::Relaxed) - base;
    let est = mt.approximate_size();
    drop(mt);
    (real, est)
}

#[test]
fn approximate_size_tracks_real_memtable_memory() {
    for (key_len, val_len) in [(24, 8), (40, 0), (30, 100), (32, 2_000), (20, 30_000)] {
        let n = if val_len > 1_000 { 2_000 } else { 50_000 };
        let (real, est) = measure(n, key_len, val_len);
        let ratio = est as f64 / real as f64;
        eprintln!(
            "key {key_len} B, value {val_len} B: real {:.1} B/entry, estimated {:.1} B/entry \
             (estimate/real = {ratio:.2})",
            real as f64 / n as f64,
            est as f64 / n as f64
        );
        // `MEASURE_ONLY=1` prints the per-entry numbers without asserting,
        // for re-deriving `ENTRY_OVERHEAD` after a skiplist/`Bytes` change.
        if std::env::var_os("MEASURE_ONLY").is_none() {
            assert!(
                (0.8..=1.25).contains(&ratio),
                "key {key_len} B / value {val_len} B: estimate {est} vs real {real} \
                 (ratio {ratio:.2}) — the flush threshold would not track memory"
            );
        }
    }
}
