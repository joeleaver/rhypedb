//! Guarded (compare-and-set) writes: `update_if`, `delete_if`, `link_if`,
//! `unlink_if`.
//!
//! The query layer evaluates a filter on a read snapshot and then writes each
//! matching object in its own transaction. Between the two, another writer may
//! commit, so the object a filter matched may no longer match by the time it
//! is written. Each guarded write therefore takes the filter as a guard and
//! decides again, inside the object's own transaction, on the version that
//! transaction reads:
//!
//! * the object is gone, or the guard rejects its current fields: nothing is
//!   written and the call reports "not applied" (`None` / `false`);
//! * the guard accepts: the write is staged, and the object's `o:` key is in
//!   the commit's conflict set (written by update and delete, pinned with a
//!   read guard by link and unlink, which do not write it), and so is every
//!   other key the guard read through its `GuardView` (`guard_view.rs`). A
//!   writer that committed to any of them after this transaction's snapshot
//!   makes the commit fail with `WriteConflict`, so the decision can never
//!   rest on a stale read.
//!
//! On `WriteConflict` the attempt is retried from scratch: a fresh snapshot, a
//! fresh read, a fresh guard decision. This is safe because nothing from the
//! failed attempt was applied, and because the guard is re-checked: a retry
//! applies only if the object still matches. Under contention on one object
//! the loser's retry sees the winner's write and (for a guard on the field the
//! winner changed) is rejected, so N racers yield exactly one applied write
//! and N-1 clean "not applied" results rather than errors.
//!
//! The unguarded `update` / `delete` / `link` / `unlink` are unchanged: no
//! re-check, no retry, `ObjectNotFound` for a missing object.

use super::DatabaseCore;
use crate::error::{EngineError, EngineResult};
use crate::object::{FieldMap, Object, deserialize_fields};
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::mvcc::Transaction;

/// A guard over an object's current committed version: `true` = still
/// matches, write it; `false` = skip it. It sees the object's fields and may
/// read other stored state (relations, other objects) through the
/// [`GuardView`](super::GuardView), all at the write's snapshot; whatever it
/// read joins the commit's conflict check. It runs once per attempt, so it
/// must decide from what it reads, not from what an earlier attempt saw.
pub type ObjectGuard<'a> = &'a dyn Fn(&super::GuardView<'_>) -> bool;

/// Attempts per guarded write before a `WriteConflict` is surfaced. Each
/// conflict means another transaction committed to a key this one touched,
/// so the system as a whole is making progress; the bound only stops one
/// caller retrying forever against a continuous stream of writers to the same
/// object whose writes keep the guard true.
pub(super) const MAX_GUARDED_ATTEMPTS: usize = 64;

/// Run `attempt` until it returns anything but `WriteConflict`, at most
/// [`MAX_GUARDED_ATTEMPTS`] times.
fn retry_on_conflict<T>(mut attempt: impl FnMut() -> EngineResult<T>) -> EngineResult<T> {
    let mut tries = 1;
    loop {
        match attempt() {
            // The competing commit has already landed (that is what the
            // conflict reports), so the retry can re-read at once.
            Err(EngineError::WriteConflict) if tries < MAX_GUARDED_ATTEMPTS => tries += 1,
            other => return other,
        }
    }
}

impl DatabaseCore {
    /// Update `object_id` only if it still exists and `guard` accepts its
    /// current committed fields, atomically with the write (compare-and-set).
    ///
    /// Returns the updated object, or `Ok(None)` when the object is missing or
    /// the guard rejects it (nothing written, no change event). A
    /// `WriteConflict` is retried internally with a fresh read and guard
    /// decision (see the module docs); other errors are returned as from
    /// [`update_with_origin`](Self::update_with_origin).
    pub fn update_if(
        &self,
        type_name: &str,
        object_id: u64,
        updates: &FieldMap,
        origin: Option<u64>,
        guard: ObjectGuard<'_>,
    ) -> EngineResult<Option<Object>> {
        retry_on_conflict(|| {
            self.update_once(type_name, object_id, updates.clone(), origin, Some(guard))
        })
    }

    /// Delete `object_id` (with its cascades) only if it still exists and
    /// `guard` accepts its current committed fields, atomically with the
    /// delete. Returns whether it was deleted. Conflicts are retried as for
    /// [`update_if`](Self::update_if).
    pub fn delete_if(
        &self,
        type_name: &str,
        object_id: u64,
        origin: Option<u64>,
        guard: ObjectGuard<'_>,
    ) -> EngineResult<bool> {
        retry_on_conflict(|| self.delete_once(type_name, object_id, origin, Some(guard)))
    }

    /// [`link`](Self::link) the source `source_id` only if it still exists and
    /// `guard` accepts its current committed fields, atomically with the edge
    /// write. Returns whether the link was written. A missing TARGET is still
    /// `ObjectNotFound`: the guard is about the source the filter matched.
    pub fn link_if(
        &self,
        source_type: &str,
        source_id: u64,
        field_name: &str,
        target_id: u64,
        edge_fields: Option<&FieldMap>,
        guard: ObjectGuard<'_>,
    ) -> EngineResult<bool> {
        retry_on_conflict(|| {
            self.link_once(
                source_type,
                source_id,
                field_name,
                target_id,
                edge_fields.cloned(),
                Some(guard),
            )
        })
    }

    /// [`unlink`](Self::unlink) under the same guard contract as
    /// [`link_if`](Self::link_if). Returns whether the guard accepted (the
    /// edge is then absent, whether or not it existed before).
    pub fn unlink_if(
        &self,
        source_type: &str,
        source_id: u64,
        field_name: &str,
        target_id: u64,
        guard: ObjectGuard<'_>,
    ) -> EngineResult<bool> {
        retry_on_conflict(|| {
            self.unlink_once(source_type, source_id, field_name, target_id, Some(guard))
        })
    }

    /// Whether the object exists at `txn`'s snapshot and `guard` accepts it.
    /// When it does, the object's `o:` key and everything the guard read
    /// through its [`GuardView`](super::GuardView) are pinned with read guards,
    /// so the commit conflicts with any later write to them (needed by callers
    /// that do not write the `o:` key themselves: link and unlink).
    pub(super) fn guard_accepts(
        &self,
        txn: &mut Transaction,
        (type_name, type_id): (&str, u64),
        object_id: u64,
        guard: ObjectGuard<'_>,
    ) -> EngineResult<bool> {
        let key = KeyBuilder::object(type_id, object_id);
        let Some(blob) = self.storage.get(txn, &key)? else {
            return Ok(false);
        };
        let accepted = self.run_guard(
            txn,
            (type_name, object_id),
            &deserialize_fields(&blob),
            guard,
        );
        if accepted {
            txn.record_read_guard(key);
        }
        Ok(accepted)
    }
}
