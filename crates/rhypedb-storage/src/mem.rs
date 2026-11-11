//! Handing freed heap memory back to the operating system.
//!
//! An embedding application usually runs on the platform allocator. glibc's
//! malloc gives each thread that allocates its own arena (64 MB reservations
//! on 64-bit), and returns memory to the OS only from the TOP of an arena's
//! heap: free pages below a live allocation stay mapped, so after a transient
//! peak (a large transaction, a compaction, a backfill chunk) the process's
//! RSS settles far above what it is using. A real consumer measured ~1.6 GB of
//! RSS sitting in glibc thread arenas after a sync.
//!
//! `malloc_trim(0)` releases every whole free page in EVERY arena (glibc
//! 2.8 and later) back to the kernel (`madvise(MADV_DONTNEED)`). It frees nothing in
//! use and changes no behavior; it only costs a walk of the free lists. The
//! engine calls [`maybe_release_free_memory`] after phases that free a lot
//! (flush, compaction, cover-refresh rounds), throttled so a hot write path
//! never pays for it more than once per [`MIN_RELEASE_INTERVAL`].
//!
//! On other allocators / platforms this is a no-op: mimalloc (which the server
//! binary uses, with `purge_delay = 0`) purges on its own, and non-glibc libcs
//! have no `malloc_trim`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Minimum time between automatic releases.
pub const MIN_RELEASE_INTERVAL: Duration = Duration::from_secs(2);

/// Release free heap memory to the OS now. Returns `true` when the allocator
/// reported returning memory (glibc), `false` otherwise or when unsupported.
pub fn release_free_memory() -> bool {
    LAST_RELEASE_MS.store(now_ms(), Ordering::Relaxed);
    trim()
}

/// [`release_free_memory`], unless one ran within [`MIN_RELEASE_INTERVAL`].
pub fn maybe_release_free_memory() {
    let now = now_ms();
    let last = LAST_RELEASE_MS.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < MIN_RELEASE_INTERVAL.as_millis() as u64 {
        return;
    }
    // One caller wins the slot; concurrent callers skip.
    if LAST_RELEASE_MS
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        trim();
    }
}

/// Milliseconds since the first call (never 0, so 0 can mean "never").
fn now_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    start.elapsed().as_millis() as u64 + 1
}

static LAST_RELEASE_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim() -> bool {
    // SAFETY: `malloc_trim` takes a padding argument and is thread-safe (it
    // locks each arena while walking it); it never frees in-use memory.
    unsafe { libc::malloc_trim(0) != 0 }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim() -> bool {
    false
}

#[cfg(all(test, target_os = "linux", target_env = "gnu"))]
mod tests {
    use super::*;

    fn rss_bytes() -> u64 {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
        let resident: u64 = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
        resident * 4096
    }

    /// Free ~256 MB of small allocations on a worker thread (so they live in a
    /// glibc thread arena) while one allocation made AFTER them stays live —
    /// pinning the arena's top, so glibc's automatic top-trim can't return
    /// anything. Only `malloc_trim` can; RSS must drop by most of it.
    #[test]
    fn release_returns_freed_thread_arena_memory() {
        const CHUNK: usize = 1024;
        const COUNT: usize = 256 * 1024; // 256 MB
        let (pin_tx, pin_rx) = std::sync::mpsc::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            let mut chunks: Vec<Vec<u8>> = (0..COUNT).map(|i| vec![i as u8; CHUNK]).collect();
            let pin = vec![1u8; CHUNK]; // allocated last: sits at the arena top
            chunks.clear();
            chunks.shrink_to_fit();
            ready_tx.send(()).unwrap();
            pin_rx.recv().unwrap(); // keep `pin` alive until the measurement
            drop(pin);
        });
        ready_rx.recv().unwrap();
        let before = rss_bytes();
        release_free_memory();
        let after = rss_bytes();
        pin_tx.send(()).unwrap();
        worker.join().unwrap();
        let released = before.saturating_sub(after);
        assert!(
            released > 128 * 1024 * 1024,
            "malloc_trim released only {} MB (rss {} -> {} MB)",
            released / (1024 * 1024),
            before / (1024 * 1024),
            after / (1024 * 1024)
        );

        // Throttle (same test: the stamp is process-global): a release just
        // ran, so an automatic one within the interval is skipped.
        let stamp = LAST_RELEASE_MS.load(Ordering::Relaxed);
        maybe_release_free_memory();
        assert_eq!(LAST_RELEASE_MS.load(Ordering::Relaxed), stamp);
    }
}
