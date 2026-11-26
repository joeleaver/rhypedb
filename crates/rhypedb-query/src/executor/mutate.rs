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
//! The guarantee is per object, not per statement: a multi-object write is a
//! sequence of independent per-object transactions, each atomic with its own
//! re-check.

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{FieldMap, Object};

use super::evaluate_predicate;
use crate::ast::{Predicate, Source, Step};
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

/// Update one object; `None` = skipped by the guard.
pub(super) fn update_one(
    db: &Database,
    guard: &WriteGuard<'_>,
    type_name: &str,
    id: u64,
    fields: &FieldMap,
) -> QueryResult<Option<Object>> {
    Ok(match guard {
        WriteGuard::Point => Some(db.update(type_name, id, fields.clone())?),
        WriteGuard::Recheck(p) => {
            db.update_if(type_name, id, fields, None, &|f| WriteGuard::accepts(p, f))?
        }
    })
}

/// Delete one object; `false` = skipped by the guard.
pub(super) fn delete_one(
    db: &Database,
    guard: &WriteGuard<'_>,
    type_name: &str,
    id: u64,
) -> QueryResult<bool> {
    Ok(match guard {
        WriteGuard::Point => {
            db.delete(type_name, id)?;
            true
        }
        WriteGuard::Recheck(p) => {
            db.delete_if(type_name, id, None, &|f| WriteGuard::accepts(p, f))?
        }
    })
}

/// Link one source object; `false` = skipped by the guard.
pub(super) fn link_one(
    db: &Database,
    guard: &WriteGuard<'_>,
    (source_type, id, field_name): (&str, u64, &str),
    target_id: u64,
    edge_fields: Option<&FieldMap>,
) -> QueryResult<bool> {
    Ok(match guard {
        WriteGuard::Point => {
            db.link(source_type, id, field_name, target_id, edge_fields.cloned())?;
            true
        }
        WriteGuard::Recheck(p) => {
            db.link_if(source_type, id, field_name, target_id, edge_fields, &|f| {
                WriteGuard::accepts(p, f)
            })?
        }
    })
}

/// Unlink one source object; `false` = skipped by the guard.
pub(super) fn unlink_one(
    db: &Database,
    guard: &WriteGuard<'_>,
    (source_type, id, field_name): (&str, u64, &str),
    target_id: u64,
) -> QueryResult<bool> {
    Ok(match guard {
        WriteGuard::Point => {
            db.unlink(source_type, id, field_name, target_id)?;
            true
        }
        WriteGuard::Recheck(p) => db.unlink_if(source_type, id, field_name, target_id, &|f| {
            WriteGuard::accepts(p, f)
        })?,
    })
}

#[cfg(test)]
mod tests;
