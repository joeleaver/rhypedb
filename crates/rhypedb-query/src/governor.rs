//! In-engine query resource governor.
//!
//! Bounds the cost of a single query so a hostile or fat-fingered request can't
//! exhaust a small (1–4 core / 512 MiB–1 GiB) deployment VM. All tenants are
//! untrusted; in the managed offering the DB runs co-located in the tenant's own
//! microVM, so a runaway query's blast radius is that tenant's own VM — but
//! "own-app abuse still tanks your own DB" is a real availability bug, and this is
//! the hard gate before any untrusted-client tier ever reaches the engine.
//!
//! The governor is OFF for embedded/library callers ([`ExecContext::new`] builds a
//! [`Governor::disabled`]); the server turns it on with limits resolved from the
//! `RHYPEDB_MAX_QUERY_*` env vars (see `rhypedb-server/config.rs`). A `0` on any
//! single limit disables that one dimension without disabling the whole governor.
//!
//! [`ExecContext::new`]: crate::executor::ExecContext::new
//!
//! ## What it bounds (and where)
//! - `.limit(N)` / vector `k` — clamped to `max_limit` ([`Governor::clamp_limit`]).
//! - Unindexed full scans — the caller materializes at most `max_rows_scanned + 1`
//!   objects and fails closed past the cap ([`Governor::scan_cap`]), rather than
//!   silently truncating (which would give wrong results for a filter).
//! - Rows EXAMINED across scans / traversals / mutations — a running budget
//!   ([`Governor::charge`]).
//! - Relationship-traversal depth ([`Governor::check_depth`]).
//! - Rows RETURNED ([`Governor::check_result_rows`]).
//! - Wall-clock per query ([`Governor::check_deadline`]).

use std::cell::Cell;
use std::time::{Duration, Instant};

use crate::error::{QueryError, QueryResult};

/// Per-query resource caps. `0` on any field means "unlimited" for that dimension,
/// so a single knob can be turned off without disabling the whole governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GovernorLimits {
    /// Clamp on `.limit(N)` and vector-search `k`. Prevents a query from asking the
    /// engine to buffer/return an absurd row count.
    pub max_limit: usize,
    /// Ceiling on the number of rows a single query may EXAMINE across all of its
    /// scans, traversals, and mutations. An unindexed full scan or a high-fan-out
    /// traversal past this fails closed rather than silently truncating.
    pub max_rows_scanned: usize,
    /// Maximum relationship-traversal depth (number of `.field` hops). Bounds the
    /// multiplicative fan-out of a deeply chained path.
    pub max_depth: usize,
    /// Ceiling on the number of objects a query may RETURN.
    pub max_result_rows: usize,
    /// Wall-clock budget for one query. [`Duration::ZERO`] = unlimited.
    pub max_duration: Duration,
}

impl GovernorLimits {
    /// Generous defaults that bound pathological cost while leaving real app queries
    /// untouched. The managed platform tightens these per VM size via the
    /// `RHYPEDB_MAX_QUERY_*` env vars.
    pub const DEFAULT: GovernorLimits = GovernorLimits {
        max_limit: 100_000,
        max_rows_scanned: 1_000_000,
        max_depth: 64,
        max_result_rows: 100_000,
        max_duration: Duration::from_secs(30),
    };

    /// Every dimension off — the governor is a no-op. Used by the embedded/library
    /// path and by an explicit server opt-out.
    pub const UNLIMITED: GovernorLimits = GovernorLimits {
        max_limit: 0,
        max_rows_scanned: 0,
        max_depth: 0,
        max_result_rows: 0,
        max_duration: Duration::ZERO,
    };
}

/// The live governor for one query execution: the resolved limits, a per-request
/// wall-clock deadline, and a running tally of rows examined.
///
/// Interior-mutable (a [`Cell`]) because it is shared by `&ExecContext` across the
/// executor's free-function call tree; a query executes on a single thread, so no
/// atomics are needed. This makes `Governor` (and thus `ExecContext`) `!Sync`,
/// which is fine — a context is created and consumed within one synchronous
/// `execute` call and never shared across threads.
pub struct Governor {
    limits: GovernorLimits,
    /// `None` when the wall-clock budget is unlimited or the governor is disabled.
    deadline: Option<Instant>,
    rows_examined: Cell<u64>,
    enabled: bool,
}

impl Governor {
    /// The disabled governor: no clamps, no deadline, no budget. `ExecContext::new`
    /// (embedded/library + tests) uses this so existing behaviour is unchanged.
    pub fn disabled() -> Self {
        Self {
            limits: GovernorLimits::UNLIMITED,
            deadline: None,
            rows_examined: Cell::new(0),
            enabled: false,
        }
    }

    /// An enabled governor with `limits`, arming the wall-clock deadline relative to
    /// `now`. `now` is a parameter (pass `Instant::now()` at request entry) so the
    /// type stays unit-testable without wall-clock nondeterminism.
    pub fn new(limits: GovernorLimits, now: Instant) -> Self {
        let deadline = (!limits.max_duration.is_zero()).then(|| now + limits.max_duration);
        Self {
            limits,
            deadline,
            rows_examined: Cell::new(0),
            enabled: true,
        }
    }

    /// Whether the governor is active. The embedded/library path stays on the exact
    /// pre-governor code paths when this is `false`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clamp a user-requested `.limit(N)` / vector `k` to `max_limit`.
    pub fn clamp_limit(&self, requested: usize) -> usize {
        if self.enabled && self.limits.max_limit != 0 {
            requested.min(self.limits.max_limit)
        } else {
            requested
        }
    }

    /// The scan ceiling (rows a single unindexed scan may materialize), or `None`
    /// when unbounded. Callers scan up to `cap + 1` and fail closed past `cap`.
    pub fn scan_cap(&self) -> Option<usize> {
        (self.enabled && self.limits.max_rows_scanned != 0).then_some(self.limits.max_rows_scanned)
    }

    /// Charge `n` rows against the examined-rows budget and re-check the deadline.
    /// Call this as rows are consumed by scans / traversals / mutations. Prefer
    /// charging in bulk (one call per batch) so the deadline's `Instant::now()`
    /// stays out of per-row inner loops.
    pub fn charge(&self, n: u64) -> QueryResult<()> {
        if !self.enabled {
            return Ok(());
        }
        let total = self.rows_examined.get().saturating_add(n);
        self.rows_examined.set(total);
        if self.limits.max_rows_scanned != 0 && total > self.limits.max_rows_scanned as u64 {
            return Err(QueryError::ResourceLimitExceeded(format!(
                "query examined more than {} rows; narrow it with an indexed filter or a smaller .limit()",
                self.limits.max_rows_scanned
            )));
        }
        self.check_deadline()
    }

    /// Fail closed if the wall-clock deadline has passed. Cheap enough to call
    /// between steps and in bulk loops (one `Instant::now()` + compare).
    pub fn check_deadline(&self) -> QueryResult<()> {
        match self.deadline {
            Some(deadline) if Instant::now() >= deadline => {
                Err(QueryError::ResourceLimitExceeded(format!(
                    "query exceeded its {} ms time budget",
                    self.limits.max_duration.as_millis()
                )))
            }
            _ => Ok(()),
        }
    }

    /// Fail closed if a traversal path is deeper than `max_depth` hops.
    pub fn check_depth(&self, depth: usize) -> QueryResult<()> {
        if self.enabled && self.limits.max_depth != 0 && depth > self.limits.max_depth {
            return Err(QueryError::ResourceLimitExceeded(format!(
                "query traverses more than {} relationship hops",
                self.limits.max_depth
            )));
        }
        Ok(())
    }

    /// Fail closed if the final result would return more than `max_result_rows`.
    pub fn check_result_rows(&self, rows: usize) -> QueryResult<()> {
        if self.enabled && self.limits.max_result_rows != 0 && rows > self.limits.max_result_rows {
            return Err(QueryError::ResourceLimitExceeded(format!(
                "query would return more than {} rows; page it with .offset()/.limit()",
                self.limits.max_result_rows
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> GovernorLimits {
        GovernorLimits {
            max_limit: 10,
            max_rows_scanned: 100,
            max_depth: 3,
            max_result_rows: 50,
            max_duration: Duration::from_secs(5),
        }
    }

    #[test]
    fn disabled_is_a_no_op() {
        let g = Governor::disabled();
        assert!(!g.is_enabled());
        // No clamp, no cap, no budget, no depth/result/deadline errors.
        assert_eq!(g.clamp_limit(1_000_000), 1_000_000);
        assert_eq!(g.scan_cap(), None);
        assert!(g.charge(u64::MAX).is_ok());
        assert!(g.check_depth(1_000).is_ok());
        assert!(g.check_result_rows(1_000_000).is_ok());
        assert!(g.check_deadline().is_ok());
    }

    #[test]
    fn clamp_limit_bounds_to_max() {
        let g = Governor::new(limits(), Instant::now());
        assert_eq!(g.clamp_limit(5), 5, "under the cap is unchanged");
        assert_eq!(g.clamp_limit(10), 10, "at the cap is unchanged");
        assert_eq!(g.clamp_limit(1_000), 10, "over the cap is clamped");
        // max_limit == 0 means that dimension is off.
        let off = Governor::new(
            GovernorLimits { max_limit: 0, ..limits() },
            Instant::now(),
        );
        assert_eq!(off.clamp_limit(1_000), 1_000);
    }

    #[test]
    fn scan_cap_reflects_row_budget() {
        assert_eq!(Governor::new(limits(), Instant::now()).scan_cap(), Some(100));
        let off = Governor::new(
            GovernorLimits { max_rows_scanned: 0, ..limits() },
            Instant::now(),
        );
        assert_eq!(off.scan_cap(), None);
    }

    #[test]
    fn charge_accumulates_and_fails_closed_over_budget() {
        let g = Governor::new(limits(), Instant::now());
        assert!(g.charge(60).is_ok());
        assert!(g.charge(40).is_ok(), "exactly at the 100-row budget is allowed");
        // One more row tips it over.
        let err = g.charge(1).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)));
        assert!(format!("{err}").contains("examined more than 100 rows"));
    }

    #[test]
    fn charge_saturates_without_overflow() {
        let g = Governor::new(
            GovernorLimits { max_rows_scanned: 0, ..limits() }, // budget off
            Instant::now(),
        );
        // max_rows_scanned == 0 disables the row budget even with a huge charge.
        assert!(g.charge(u64::MAX).is_ok());
        assert!(g.charge(u64::MAX).is_ok(), "saturating_add must not panic");
    }

    #[test]
    fn depth_budget_fails_closed() {
        let g = Governor::new(limits(), Instant::now());
        assert!(g.check_depth(3).is_ok(), "at the 3-hop cap is allowed");
        assert!(g.check_depth(4).is_err(), "past the cap fails closed");
    }

    #[test]
    fn result_row_ceiling_fails_closed() {
        let g = Governor::new(limits(), Instant::now());
        assert!(g.check_result_rows(50).is_ok());
        assert!(g.check_result_rows(51).is_err());
    }

    #[test]
    fn deadline_already_passed_fails_closed() {
        // Arm the deadline in the past so the check trips deterministically.
        let past = Instant::now() - Duration::from_secs(1);
        let g = Governor::new(
            GovernorLimits { max_duration: Duration::from_millis(1), ..limits() },
            past,
        );
        let err = g.check_deadline().unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)));
        assert!(format!("{err}").contains("time budget"));
        // And charge() also surfaces the deadline once the budget is otherwise fine.
        assert!(g.charge(1).is_err());
    }

    #[test]
    fn zero_duration_means_no_deadline() {
        let g = Governor::new(
            GovernorLimits { max_duration: Duration::ZERO, ..limits() },
            Instant::now() - Duration::from_secs(3600),
        );
        assert!(g.check_deadline().is_ok(), "ZERO duration disables the wall-clock budget");
    }
}
