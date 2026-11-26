//! The per-object writes of `.update()`, `.delete()`, `.link()` and
//! `.unlink()`, and the guard each one re-checks.
//!
//! A write step receives the objects the pipeline before it selected, which
//! were read on a snapshot. When that selection came from a filter (or any set
//! source), each object is written through the engine's guarded API: inside
//! the object's own write transaction the engine re-reads it and re-evaluates
//! the filter predicates on its current committed version, and skips it (not
//! written, not returned) if it is gone or no longer matches. Conflicting
//! commits are retried by the engine on a fresh read. That makes
//! `T.filter(.k == v && .guard == old).update({ guard: new })` compare-and-set:
//! of any number of concurrent statements guarded on the same old value,
//! exactly one applies. See `docs/src/queries.md`, "Guarded writes".
//!
//! With security rules on, the write's rule is part of the same guard (see
//! `authz/write_gate.rs`), for point writes too: the rule is decided on the
//! version the write is based on, atomically with it, and re-decided on every
//! retry. A denied object is `PermissionDenied` with nothing written for it.
//!
//! The guarantee is per object, not per statement: a multi-object write is a
//! sequence of independent per-object transactions, each atomic with its own
//! re-check. (The rules' early check refuses a statement before any row is
//! written when the snapshot already shows a denied row; only a rule input
//! changing mid-statement can deny a later row after earlier rows landed.)

use rhypedb_engine::EngineError;
use rhypedb_engine::database::{Database, GuardView};
use rhypedb_engine::object::{FieldMap, Object};

use super::evaluate_predicate;
use crate::ast::{Predicate, Source, Step};
use crate::authz::WriteGate;
use crate::error::QueryResult;

/// How a write step re-checks each object it was handed.
#[derive(Debug, PartialEq)]
pub(super) enum WriteGuard<'q> {
    /// A point write (`T.get(id)` with no filter on the way): nothing to
    /// re-check, and a missing object is `ObjectNotFound` as before.
    Point,
    /// A set write: each object is written only if it still exists and
    /// satisfies every one of these predicates on its committed version.
    /// Empty for an unfiltered set (`T.all()`), which re-checks existence only.
    Recheck(Vec<&'q Predicate>),
}

impl WriteGuard<'_> {
    /// Whether `fields` (an object's current committed version) still matches.
    fn accepts(predicates: &[&Predicate], fields: &FieldMap) -> bool {
        predicates.iter().all(|p| evaluate_predicate(p, fields))
    }
}

/// The guard for the write step that follows `before` (the steps preceding it).
///
/// The predicates re-checked are the ones about the written objects
/// themselves: the source filter and every `.filter()` since the last step
/// that changed which objects flow (a traversal moves to other objects, and a
/// prior write may have changed the very fields a filter tested). A filter
/// before a traversal constrained OTHER objects, which this write does not
/// lock, so it is not re-checked.
pub(super) fn write_guard<'q>(source: &'q Source, before: &'q [Step]) -> WriteGuard<'q> {
    let (mut predicates, mut set): (Vec<&Predicate>, bool) = match source {
        Source::Filter { predicate, .. } => (vec![predicate], true),
        Source::All { .. } => (Vec::new(), true),
        Source::Get { .. } | Source::Create { .. } | Source::CreateBatch { .. } => {
            (Vec::new(), false)
        }
    };
    for step in before {
        match step {
            Step::Filter { predicate } => {
                predicates.push(predicate);
                set = true;
            }
            Step::Traverse { .. } => {
                predicates.clear();
                set = true;
            }
            Step::Update { .. } | Step::Delete | Step::Link { .. } | Step::Unlink { .. } => {
                predicates.clear();
            }
            // These narrow or rank the same objects; the filters still apply.
            Step::Similar { .. }
            | Step::Matches { .. }
            | Step::Limit { .. }
            | Step::Offset { .. } => {}
        }
    }
    if set {
        WriteGuard::Recheck(predicates)
    } else {
        WriteGuard::Point
    }
}

/// The engine guard for one guarded write: the step's predicates on the object's committed
/// fields, then (rules on) the write's security rule, both inside the write's transaction.
fn engine_guard<'a, 'q: 'a, 'g: 'a>(
    predicates: &'a [&'q Predicate],
    gate: Option<&'a WriteGate<'g>>,
) -> impl Fn(&GuardView<'_>) -> bool + 'a {
    move |view| {
        if let Some(g) = gate {
            g.begin_attempt();
        }
        // A row the filter no longer selects is skipped, never denied: the rule is only asked
        // about rows the write would actually touch.
        WriteGuard::accepts(predicates, view.fields()) && gate.is_none_or(|g| g.allows(view))
    }
}

/// The predicates a guard re-checks (none for a point write).
fn predicates<'a, 'q>(guard: &'a WriteGuard<'q>) -> &'a [&'q Predicate] {
    match guard {
        WriteGuard::Point => &[],
        WriteGuard::Recheck(p) => p,
    }
}

/// Settle a guarded write that `applied` or not: a rules denial is `PermissionDenied`, and a point
/// write whose object is gone is `ObjectNotFound` (as the unguarded engine call reports it) when
/// `missing_is_error`.
fn settle(
    guard: &WriteGuard<'_>,
    gate: Option<&WriteGate<'_>>,
    (type_name, id): (&str, u64),
    applied: bool,
    missing_is_error: bool,
) -> QueryResult<()> {
    if let Some(g) = gate {
        g.verdict(type_name)?;
    }
    if !applied && missing_is_error && *guard == WriteGuard::Point {
        return Err(EngineError::ObjectNotFound {
            type_name: type_name.into(),
            object_id: id,
        }
        .into());
    }
    Ok(())
}

/// Update one object; `None` = skipped by the guard.
pub(super) fn update_one(
    db: &Database,
    (guard, gate): (&WriteGuard<'_>, Option<&WriteGate<'_>>),
    type_name: &str,
    id: u64,
    fields: &FieldMap,
) -> QueryResult<Option<Object>> {
    if *guard == WriteGuard::Point && gate.is_none() {
        return Ok(Some(db.update(type_name, id, fields.clone())?));
    }
    let out = db.update_if(
        type_name,
        id,
        fields,
        None,
        &engine_guard(predicates(guard), gate),
    )?;
    settle(guard, gate, (type_name, id), out.is_some(), true)?;
    Ok(out)
}

/// Delete one object; `false` = skipped by the guard.
pub(super) fn delete_one(
    db: &Database,
    (guard, gate): (&WriteGuard<'_>, Option<&WriteGate<'_>>),
    type_name: &str,
    id: u64,
) -> QueryResult<bool> {
    if *guard == WriteGuard::Point && gate.is_none() {
        db.delete(type_name, id)?;
        return Ok(true);
    }
    let deleted = db.delete_if(type_name, id, None, &engine_guard(predicates(guard), gate))?;
    settle(guard, gate, (type_name, id), deleted, true)?;
    Ok(deleted)
}

/// Link one source object; `false` = skipped by the guard.
pub(super) fn link_one(
    db: &Database,
    (guard, gate): (&WriteGuard<'_>, Option<&WriteGate<'_>>),
    (source_type, id, field_name): (&str, u64, &str),
    target_id: u64,
    edge_fields: Option<&FieldMap>,
) -> QueryResult<bool> {
    if *guard == WriteGuard::Point && gate.is_none() {
        db.link(source_type, id, field_name, target_id, edge_fields.cloned())?;
        return Ok(true);
    }
    let check = engine_guard(predicates(guard), gate);
    let linked = db.link_if(source_type, id, field_name, target_id, edge_fields, &check)?;
    settle(guard, gate, (source_type, id), linked, true)?;
    Ok(linked)
}

/// Unlink one source object; `false` = skipped by the guard.
pub(super) fn unlink_one(
    db: &Database,
    (guard, gate): (&WriteGuard<'_>, Option<&WriteGate<'_>>),
    (source_type, id, field_name): (&str, u64, &str),
    target_id: u64,
) -> QueryResult<bool> {
    if *guard == WriteGuard::Point && gate.is_none() {
        db.unlink(source_type, id, field_name, target_id)?;
        return Ok(true);
    }
    let check = engine_guard(predicates(guard), gate);
    let unlinked = db.unlink_if(source_type, id, field_name, target_id, &check)?;
    // An unguarded unlink from a missing source is a no-op, not an error; a gated point unlink
    // keeps that (there is no edge to remove, so nothing to authorize).
    settle(guard, gate, (source_type, id), unlinked, false)?;
    Ok(unlinked || *guard == WriteGuard::Point)
}

#[cfg(test)]
mod gate_tests;
#[cfg(test)]
mod tests;
