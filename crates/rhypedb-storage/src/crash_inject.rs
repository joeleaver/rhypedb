//! In-process crash-fault injection for the crash-recovery fuzz harness.
//!
//! GATED behind the `crash-fuzz` Cargo feature — OFF by default. With the
//! feature off, [`hit`] is an `#[inline(always)]` no-op that monomorphizes to
//! nothing (its [`Site`] argument is a compile-time constant), the arming API
//! ([`arm`]/[`disarm`]/[`catch_crash`]/[`Mode`]/[`Caught`]) does not exist, and
//! the shipped binary is byte-identical to a build that never mentioned this
//! module. **NEVER enable `crash-fuzz` in a release artifact.**
//!
//! # Model
//!
//! A test [`arm`]s a named [`Site`] to "fire after N hits" on the CURRENT
//! THREAD. Arming is thread-local, so parallel `cargo test` threads never
//! cross-fire and a single-threaded workload is deterministic. When the armed
//! site is hit for the Nth time the injector `panic!`s with a private crash
//! marker, unwinding out of the in-flight storage operation WITHOUT running it
//! to completion — exactly as a process killed at that instruction would. The
//! harness wraps the workload in [`catch_crash`], which catches ONLY the marker
//! (any other panic is a real bug and is re-raised unchanged), then performs a
//! faithful teardown and cold-reopens the data dir.
//!
//! ## Why unwinding is faithful
//!
//! The storage objects (`Wal`, memtables, SST readers) are owned by the harness
//! OUTSIDE the panicking closure, so the unwind drops only the in-flight
//! operation's locals (lock guards, scratch buffers) — never flushing or
//! finalizing anything a real crash would not. The one exception is the WAL's
//! `BufWriter`, whose `Drop` would flush its un-flushed user-space buffer (the
//! `Wal::append_txn` window between `write_all` and `flush`). The harness's
//! teardown discards that buffer without flushing it
//! (`LsmTree::discard_for_crash_recovery` → `Wal::forget_unflushed_buffer`),
//! then lets a normal `drop` release the data-dir lock and close fds — modelling
//! a process that died with dirty user-space buffers but whose already-`write`n
//! bytes remain in the OS page cache (i.e. SIGKILL semantics). Power-loss
//! (bytes that never reached the platter) is modelled separately by truncating
//! the WAL tail in the harness.

/// A named fault-injection point in the storage write / flush path. This enum
/// is ALWAYS compiled (it is the argument type of [`hit`]); only the *behavior*
/// of [`hit`] is feature-gated. Variants are append-only — tests refer to them
/// by name, and [`SITE_COUNT`] must stay in sync (asserted by a test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Site {
    /// `Wal::append_txn`: after the batch `write_all`, before the `BufWriter`
    /// flush. For a batch SMALLER than the `BufWriter` capacity (8 KiB) the
    /// bytes are still in the user-space buffer, so a crash here loses the whole
    /// in-flight transaction. A batch LARGER than the buffer is written straight
    /// through to the OS page cache by `write_all`, so its footer may already be
    /// durable and the txn can survive — still faithful to a real kill at this
    /// instruction. The recovery oracle therefore predicts the outcome from the
    /// on-disk WAL framing (is the footer present?), never from this site alone.
    WalAfterWriteBeforeFlush,
    /// `Wal::sync`: after the `BufWriter` flush (bytes now in the OS page
    /// cache), before `sync_all` — survives an in-process kill, lost only on a
    /// real power loss.
    WalAfterFlushBeforeFsync,
    /// `Wal::sync`: after `sync_all` returns — the transaction's WAL records are
    /// fully durable on the platter.
    WalAfterFsync,
    /// `LsmTree::flush_locked`: after the active memtable is rotated to the
    /// immutable set, before the SST is written. Nothing durable changed yet.
    FlushAfterMemtableRotate,
    /// `LsmTree::flush_locked`: after the SST `finish()` (durable on disk),
    /// before it is registered in `sst_files` and before the WAL is truncated.
    FlushAfterSstFinish,
    /// `LsmTree::flush_locked`: after the SST is registered for reads, before
    /// the WAL is truncated — reopen must reconcile the SST against a still-full
    /// WAL (idempotent double-replay).
    FlushAfterSstRegister,
    /// `LsmTree::flush_locked`: SST durable + registered, immediately BEFORE the
    /// WAL is truncated/recreated — the flush↔WAL-truncate race boundary.
    FlushBeforeWalTruncate,
    /// `LsmTree::flush_locked`: after the WAL has been truncated to a fresh
    /// header-only file. The flushed data lives only in the SST now.
    FlushAfterWalTruncate,
    /// `SstWriter::add`: partway through writing an SST's entries (flush OR
    /// compaction). A crash here leaves a partial `*.sst.tmp` that recovery must
    /// clean up — the data is still in the WAL (flush) or the un-deleted inputs
    /// (compaction).
    SstMidWrite,
    /// `LsmTree::compact_inner`: after the merged SST is published (renamed in)
    /// but before the in-RAM `sst_files` swap and before any input is unlinked —
    /// reopen sees the merged SST AND all inputs (redundant but correct).
    CompactionAfterMergedPublished,
    /// `LsmTree::compact_inner`: after the in-RAM swap, before the input-unlink
    /// loop — same on-disk state as above (the swap is in-RAM only).
    CompactionAfterSwap,
    /// `LsmTree::compact_inner`: partway through the input-unlink loop — some
    /// inputs deleted, some remain, merged present. Reopen must reconcile.
    CompactionMidUnlink,
}

/// Number of [`Site`] variants. Kept in sync with the enum by
/// `tests::site_count_matches_variants` (feature-on build).
pub const SITE_COUNT: usize = 12;

#[cfg(feature = "crash-fuzz")]
pub use imp::{arm, catch_crash, disarm, hit, Caught, Mode};

/// No-op when the feature is off: monomorphizes to nothing. Keeps every
/// injection call site (`crash_inject::hit(Site::X)`) compiling and free in the
/// shipped binary.
#[cfg(not(feature = "crash-fuzz"))]
#[inline(always)]
pub fn hit(_site: Site) {}

#[cfg(feature = "crash-fuzz")]
mod imp {
    use super::{Site, SITE_COUNT};
    use std::cell::RefCell;
    use std::sync::Once;

    /// How an armed [`Site`] fires when it is hit. Append-only.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Mode {
        /// Simulate process death at this instruction: panic-unwind carrying the
        /// private crash marker, which [`catch_crash`] recognizes.
        Crash,
    }

    /// Result of running a workload under [`catch_crash`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Caught {
        /// The workload hit the armed site and was unwound there.
        Crashed(Site),
        /// The workload ran to completion without hitting an armed site.
        Completed,
    }

    #[derive(Clone, Copy)]
    struct Armed {
        site: Site,
        after: u64,
        mode: Mode,
    }

    /// Private panic payload identifying an injected crash. Suppressed by the
    /// quiet hook so a fuzz corpus does not flood stderr; recognized by
    /// [`catch_crash`] (any other payload is re-raised as a real bug).
    struct CrashMarker {
        site: Site,
    }

    thread_local! {
        static ARMED: RefCell<Option<Armed>> = const { RefCell::new(None) };
        static COUNTERS: RefCell<[u64; SITE_COUNT]> = const { RefCell::new([0; SITE_COUNT]) };
    }

    /// Arm `site` to fire `mode` after it has been hit `after_hits` times ON
    /// THIS THREAD (`after_hits` is clamped to `>= 1`, so `1` fires on the very
    /// next hit). Replaces any prior arm and resets this thread's hit counters.
    /// Idempotently installs the quiet panic hook.
    ///
    /// Arm INSIDE the [`catch_crash`] closure that drives the workload: an
    /// injected crash that fires while no `catch_crash` is on the stack is an
    /// uncaught panic (the quiet hook merely suppresses its message).
    pub fn arm(site: Site, after_hits: u64, mode: Mode) {
        install_quiet_hook();
        COUNTERS.with(|c| *c.borrow_mut() = [0; SITE_COUNT]);
        ARMED.with(|a| {
            *a.borrow_mut() = Some(Armed {
                site,
                after: after_hits.max(1),
                mode,
            })
        });
    }

    /// Disarm this thread: no site fires until re-armed. Also clears the hit
    /// counters so a re-arm starts clean.
    pub fn disarm() {
        ARMED.with(|a| *a.borrow_mut() = None);
        COUNTERS.with(|c| *c.borrow_mut() = [0; SITE_COUNT]);
    }

    /// Record a hit at `site`; fire the injected fault if this thread armed
    /// `site` and the configured hit count has been reached.
    #[inline]
    pub fn hit(site: Site) {
        let armed = ARMED.with(|a| *a.borrow());
        let Some(armed) = armed else { return };
        if armed.site != site {
            return;
        }
        let n = COUNTERS.with(|c| {
            let mut c = c.borrow_mut();
            c[site as usize] += 1;
            c[site as usize]
        });
        if n == armed.after {
            fire(site, armed.mode);
        }
    }

    fn fire(site: Site, mode: Mode) -> ! {
        match mode {
            Mode::Crash => std::panic::panic_any(CrashMarker { site }),
        }
    }

    /// Run `workload`, catching ONLY an injected crash. Returns
    /// [`Caught::Crashed`] with the firing site if the workload was unwound by
    /// the injector, or [`Caught::Completed`] if it finished. Any non-injected
    /// panic (a real bug) is re-raised unchanged.
    pub fn catch_crash<F: FnOnce()>(workload: F) -> Caught {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        // AssertUnwindSafe: on a caught crash the harness DISCARDS the storage
        // handle and cold-reopens from disk, so the logical unwind-safety of any
        // in-memory state captured by `workload` is irrelevant.
        match catch_unwind(AssertUnwindSafe(workload)) {
            Ok(()) => Caught::Completed,
            Err(payload) => match payload.downcast::<CrashMarker>() {
                Ok(marker) => Caught::Crashed(marker.site),
                Err(other) => std::panic::resume_unwind(other),
            },
        }
    }

    /// Install (once per process) a panic hook that swallows the injected crash
    /// marker so a fuzz corpus does not spam stderr, and chains to the previous
    /// hook for every other panic. Safe under parallel tests: it only
    /// special-cases [`CrashMarker`].
    fn install_quiet_hook() {
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if info.payload().is::<CrashMarker>() {
                    return;
                }
                prev(info);
            }));
        });
    }
}

#[cfg(all(test, not(feature = "crash-fuzz")))]
mod inert_tests {
    use super::{hit, Site};

    #[test]
    fn hit_is_a_noop_when_feature_off() {
        // Must compile and do nothing at all in the default (no crash-fuzz)
        // build — this is the path the shipped binary takes.
        for _ in 0..1000 {
            hit(Site::WalAfterFsync);
            hit(Site::FlushBeforeWalTruncate);
        }
    }
}

#[cfg(all(test, feature = "crash-fuzz"))]
mod tests {
    use super::*;

    const ALL_SITES: [Site; SITE_COUNT] = [
        Site::WalAfterWriteBeforeFlush,
        Site::WalAfterFlushBeforeFsync,
        Site::WalAfterFsync,
        Site::FlushAfterMemtableRotate,
        Site::FlushAfterSstFinish,
        Site::FlushAfterSstRegister,
        Site::FlushBeforeWalTruncate,
        Site::FlushAfterWalTruncate,
        Site::SstMidWrite,
        Site::CompactionAfterMergedPublished,
        Site::CompactionAfterSwap,
        Site::CompactionMidUnlink,
    ];

    #[test]
    fn site_count_matches_variants() {
        // Every variant maps to a distinct in-bounds counter slot. If a variant
        // is added without bumping SITE_COUNT, the thread-local counter array
        // indexing would be out of bounds — catch it here instead.
        for (i, s) in ALL_SITES.iter().enumerate() {
            assert_eq!(*s as usize, i, "variant order must match index");
            assert!((*s as usize) < SITE_COUNT);
        }
    }

    #[test]
    fn arm_fires_after_n_hits_then_caught() {
        let outcome = catch_crash(|| {
            arm(Site::WalAfterFsync, 3, Mode::Crash);
            hit(Site::WalAfterFsync); // 1 — no fire
            hit(Site::WalAfterFsync); // 2 — no fire
            hit(Site::WalAfterFsync); // 3 — panics here
            unreachable!("third hit must have crashed");
        });
        assert_eq!(outcome, Caught::Crashed(Site::WalAfterFsync));
        disarm();
    }

    #[test]
    fn after_hits_zero_is_clamped_to_one() {
        let outcome = catch_crash(|| {
            arm(Site::FlushAfterSstFinish, 0, Mode::Crash);
            hit(Site::FlushAfterSstFinish); // first hit fires (clamped to 1)
            unreachable!();
        });
        assert_eq!(outcome, Caught::Crashed(Site::FlushAfterSstFinish));
        disarm();
    }

    #[test]
    fn unarmed_and_other_sites_do_not_fire() {
        let outcome = catch_crash(|| {
            arm(Site::WalAfterFsync, 1, Mode::Crash);
            // A different site never fires this arm, however many times it's hit.
            for _ in 0..100 {
                hit(Site::FlushAfterSstFinish);
            }
            disarm();
            // Disarmed: the armed site itself no longer fires.
            for _ in 0..100 {
                hit(Site::WalAfterFsync);
            }
        });
        assert_eq!(outcome, Caught::Completed);
    }

    #[test]
    fn completed_workload_reports_completed() {
        let outcome = catch_crash(|| {});
        assert_eq!(outcome, Caught::Completed);
    }

    #[test]
    fn real_panic_is_re_raised_not_swallowed() {
        // A non-injected panic inside the workload must propagate, not be
        // mistaken for an injected crash.
        let res = std::panic::catch_unwind(|| {
            catch_crash(|| panic!("a real bug"));
        });
        assert!(res.is_err(), "a real panic must escape catch_crash");
    }
}
