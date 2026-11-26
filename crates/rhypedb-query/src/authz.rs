//! P4 data-plane authorization: the bridge between query execution and the `rhypedb-authz` rules
//! evaluator. Everything here is a **strict no-op when `ExecContext.rules` is `None`** — a rules-off
//! deployment (every v1/P0–P2 tenant today) takes the exact pre-P4 execution path, so nothing about
//! its behavior changes. Rules are opt-in; turning them on is what closes the doors.
//!
//! Three enforcement points, each fed a live [`EngineResource`] the rules read through:
//! - **create** — gate the incoming write fields before `db.create`/`create_batch` (§[`gate_create`]).
//! - **write** (update/delete/link/unlink) — gate the **pre-mutation** object, so a rule like
//!   `request.auth.uid == resource.author` can't be defeated by the same write that flips `author`
//!   (P0-DBA-6). The rule is evaluated INSIDE each object's write transaction, on the version that
//!   write is based on, with everything the rule read in the commit's conflict check, so a
//!   concurrent change to a rule input can't slip a write the rules no longer allow (§[`WriteGate`]).
//! - **read** — filter the result of EVERY pipeline stage, dropping rows the principal may not read
//!   (Firestore semantics: you get the rows you may read, not an error), so no step — a filter, a
//!   ranked search, a limit, a traversal, a write gate — ever acts on a row the principal cannot
//!   see (§[`read_filter_stage`], [`filter_read`]).
//!
//! A denied *write* is a loud [`QueryError::PermissionDenied`]; a denied *read* row is silently
//! dropped. Relationship projections a rule needs (`resource.owners.uid`) walk edges via the
//! engine, bounded by both a hard cap and the query [`Governor`] so a rule can't become a fan-out
//! DoS (P0-DBA-7).

use rhypedb_authz::{Decision, Op, ResourceAccessor, RulesProgram};
use rhypedb_engine::database::Database;
use rhypedb_engine::object::{value_to_query_json, FieldMap, Object};

use crate::error::{QueryError, QueryResult};
use crate::executor::{ExecContext, QueryOutput};
use crate::governor::Governor;

mod write_gate;
pub(crate) use write_gate::WriteGate;

/// Hard cap on the linked targets one rule relationship-projection will enumerate. A relation
/// LARGER than this is treated as **indeterminate** (fails closed ⇒ deny), NOT truncated to a
/// partial set — a truncated deny-list (`!(x in resource.blocked.field)`) would fail OPEN for a
/// member past the cap. So membership rules (`x in resource.rel.field`) are only evaluable over a
/// relation of at most this many linked targets; beyond it, the rule denies (v1 limitation —
/// denormalize a scalar for larger sets). The governor also charges every edge, so a rule can't
/// outrun the query row budget either.
const MAX_RULE_RELATION_TARGETS: usize = 1024;

/// Filter a terminal result through the `read` rule when rules are on (no-op otherwise). ANY object
/// surfaced to the caller is a read, so this covers reads AND a mutation's returned post-image: a
/// multi-row `Objects` result is filtered row-wise, and a `Single` (a create/update return value)
/// the principal may not read collapses to `Done` (void, Firestore-style) rather than leaking. This
/// is what stops an `update` return value from bypassing the read rule.
pub(crate) fn apply_read_filter(ctx: &ExecContext<'_>, result: QueryOutput) -> QueryOutput {
    let Some(rules) = &ctx.rules else {
        return result;
    };
    match result {
        QueryOutput::Objects(mut objs) => {
            filter_read(ctx, &mut objs);
            QueryOutput::Objects(objs)
        }
        // A ranked result is filtered row-wise too; scores ride along by id
        // (ids are unique within one result) and the rank order is kept.
        QueryOutput::Scored(rows) => {
            let scores: std::collections::HashMap<u64, crate::executor::RowScore> =
                rows.iter().map(|(o, s)| (o.id, *s)).collect();
            let mut objs: Vec<Object> = rows.into_iter().map(|(o, _)| o).collect();
            filter_read(ctx, &mut objs);
            QueryOutput::Scored(
                objs.into_iter()
                    .map(|o| {
                        let s = scores[&o.id];
                        (o, s)
                    })
                    .collect(),
            )
        }
        QueryOutput::Single(mut obj) => {
            obj.ensure_fields_deserialized();
            let type_name = obj.type_name.clone();
            let res = EngineResource {
                db: ctx.db,
                governor: &ctx.governor,
                type_name: &type_name,
                object: Some(&obj),
                request: None,
            };
            if rules
                .evaluate(Op::Read, &type_name, &ctx.principal, &res)
                .is_allow()
            {
                QueryOutput::Single(obj)
            } else {
                QueryOutput::Done
            }
        }
        other => other,
    }
}

/// Read-filter an INTERMEDIATE pipeline result (rules on; a no-op otherwise), so every step sees
/// only rows the principal may read — unreadable rows behave exactly as if they did not exist.
///
/// Filtering only the terminal result let the steps in between work over unreadable rows, and
/// their observable effects became oracles: a write gate refusing `….filter(.body.contains(s))
/// .delete()` only when a hidden row matched, a traversal returning a readable neighbour of a
/// hidden match, a `.limit`/`k` cut crowded by hidden rows. The streaming id shapes are
/// materialized to evaluate the rule (charged to the governor) and keep their shape; a `Single`
/// the principal may not read collapses to `Done` like at the terminal.
pub(crate) fn read_filter_stage(ctx: &ExecContext<'_>, result: QueryOutput) -> QueryResult<QueryOutput> {
    if ctx.rules.is_none() {
        return Ok(result);
    }
    let filtered = match result {
        QueryOutput::IdSet { type_name, mut ids } => {
            let keep = readable_ids(ctx, &type_name, &ids)?;
            ids.retain(|id| keep.contains(id));
            QueryOutput::IdSet { type_name, ids }
        }
        QueryOutput::IdSetWithFields { type_name, mut items } => {
            let ids: Vec<u64> = items.iter().map(|(id, _)| *id).collect();
            let keep = readable_ids(ctx, &type_name, &ids)?;
            items.retain(|(id, _)| keep.contains(id));
            QueryOutput::IdSetWithFields { type_name, items }
        }
        other => {
            // Evaluating the rule is per-row work like any other — charge it.
            let rows = match &other {
                QueryOutput::Objects(o) => o.len(),
                QueryOutput::Scored(r) => r.len(),
                _ => 0,
            };
            ctx.governor.charge(rows as u64)?;
            apply_read_filter(ctx, other)
        }
    };
    ctx.governor.check_exhausted()?;
    Ok(filtered)
}

/// Objects evaluated per rule-check batch: the ids that pass are kept, the loaded objects are
/// dropped before the next batch, so checking a large candidate set never holds it all.
const READ_CHECK_BATCH: usize = 1024;

/// The ids among `ids` (of `type_name`) the principal may read, charged one row each, checked
/// in batches of [`READ_CHECK_BATCH`] objects.
pub(crate) fn readable_ids(
    ctx: &ExecContext<'_>,
    type_name: &str,
    ids: &[u64],
) -> QueryResult<std::collections::HashSet<u64>> {
    ctx.governor.charge(ids.len() as u64)?;
    let mut keep = std::collections::HashSet::with_capacity(ids.len());
    for batch in ids.chunks(READ_CHECK_BATCH) {
        let mut objs = ctx.db.get_many_lazy(type_name, batch).unwrap_or_default();
        filter_read(ctx, &mut objs);
        keep.extend(objs.iter().map(|o| o.id));
        ctx.governor.check_deadline()?;
    }
    Ok(keep)
}

/// A [`ResourceAccessor`] backed by the live engine: a stored (pre-mutation) object and/or the
/// incoming write fields. Relationship projections resolve the target type from the schema and read
/// linked objects through `db.get_links_many` + `db.get`, charging the governor.
pub(crate) struct EngineResource<'a> {
    db: &'a Database,
    governor: &'a Governor,
    type_name: &'a str,
    /// The stored object (fields already deserialized). `None` for a `create` (no prior resource).
    object: Option<&'a Object>,
    /// The incoming write fields (`request.<field>`). `None` for read/delete/link/unlink.
    request: Option<&'a FieldMap>,
}

/// The target type of relation field `rel` on `type_name`, or `None` if it is not a relation.
fn relation_target(db: &Database, type_name: &str, rel: &str) -> Option<String> {
    match &db.schema().get_type(type_name)?.get_field(rel)?.field_type {
        rhypedb_schema::FieldType::Relation(r) => Some(r.target_type.clone()),
        _ => None,
    }
}

/// Whether a rule may use a relation projection that examined `examined` edges: charges them to
/// the governor, and refuses (indeterminate ⇒ the caller fails closed) when that trips the query
/// row budget OR the relation is too large to enumerate fully. NEVER truncate to a partial set: a
/// partial deny-list would fail OPEN for a member past the cap (a review finding). `examined`
/// (pre-dedup edge count) is an upper bound on the id count, so `examined <= cap` guarantees the
/// full set fits.
fn within_rule_budget(governor: &Governor, examined: u64) -> bool {
    governor.charge(examined).is_ok() && examined <= MAX_RULE_RELATION_TARGETS as u64
}

impl<'a> EngineResource<'a> {
    fn relation_target(&self, rel: &str) -> Option<String> {
        relation_target(self.db, self.type_name, rel)
    }

    /// Linked target ids for `rel` off the current object. `None` = INDETERMINATE (no object,
    /// traversal error, or over-budget) — the caller must fail this closed (Absent/Unknown), NOT
    /// treat it as an empty set. `Some(ids)` is the determinate id list (`Some(empty)` = genuinely
    /// unlinked). Charges the FULL examined edge set (like the query traversal's `hop_edges`), so a
    /// rule over a huge to-many edge trips the row budget instead of evading it.
    fn linked_ids(&self, rel: &str) -> Option<Vec<u64>> {
        let obj = self.object?;
        let groups = self.db.get_links_many(self.type_name, &[obj.id], rel).ok()?;
        let examined: u64 = groups.iter().map(|g| g.len() as u64).sum();
        // Indeterminate ⇒ `None` ⇒ Absent ⇒ deny (see `within_rule_budget`).
        if !within_rule_budget(self.governor, examined) {
            return None;
        }
        Some(groups.into_iter().flatten().map(|(id, _)| id).collect())
    }

    fn target_scalar(&self, target_type: &str, id: u64, field: &str) -> Option<serde_json::Value> {
        let mut tobj = self.db.get(target_type, id).ok()?;
        tobj.ensure_fields_deserialized();
        tobj.fields.get(field).map(value_to_query_json)
    }
}

impl ResourceAccessor for EngineResource<'_> {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.object?.fields.get(field).map(value_to_query_json)
    }

    fn resource_relation_one(&self, rel: &str, target_field: &str) -> Option<serde_json::Value> {
        let target_type = self.relation_target(rel)?;
        // `None`/empty (indeterminate or unlinked) ⇒ None ⇒ Absent (fail closed).
        let id = *self.linked_ids(rel)?.first()?;
        self.target_scalar(&target_type, id, target_field)
    }

    fn resource_relation_many(&self, rel: &str, target_field: &str) -> Option<Vec<serde_json::Value>> {
        let target_type = self.relation_target(rel)?;
        // Propagate indeterminacy: `None` from linked_ids (over-budget) stays `None` ⇒ Absent, so a
        // deny-list rule fails closed rather than seeing a spurious empty set.
        let ids = self.linked_ids(rel)?;
        Some(
            ids.into_iter()
                .filter_map(|id| self.target_scalar(&target_type, id, target_field))
                .collect(),
        )
    }

    fn request_field(&self, field: &str) -> Option<serde_json::Value> {
        self.request?.get(field).map(value_to_query_json)
    }
}

/// Gate a `create` against the incoming fields. No-op if rules are off. `db.create` runs only if the
/// `create` rule allows.
pub(crate) fn gate_create(
    ctx: &ExecContext<'_>,
    type_name: &str,
    fields: &FieldMap,
) -> QueryResult<()> {
    let Some(rules) = &ctx.rules else { return Ok(()) };
    let res = EngineResource {
        db: ctx.db,
        governor: &ctx.governor,
        type_name,
        object: None,
        request: Some(fields),
    };
    decide(rules, Op::Create, type_name, ctx, &res)
}

/// The EARLY check of a write (update/delete/link/unlink) over `ids`: each **pre-mutation** object,
/// read on the query's snapshot, before any of them is written. No-op if rules are off. An id that
/// no longer exists is skipped (the engine call handles non-existence with its normal semantics).
/// Denied ⇒ [`QueryError::PermissionDenied`], with nothing written: a bulk write the rules refuse on
/// any row is refused whole.
///
/// This is NOT the authoritative check — the snapshot can be stale by the time a row is written.
/// [`WriteGate`] re-evaluates the rule inside each row's own write transaction (see `write_gate`).
fn gate_write(
    ctx: &ExecContext<'_>,
    op: Op,
    type_name: &str,
    ids: &[u64],
    request: Option<&FieldMap>,
) -> QueryResult<()> {
    let Some(rules) = &ctx.rules else { return Ok(()) };
    for &id in ids {
        // Fetch the object as it exists BEFORE the write (P0-DBA-6). A missing object is left to the
        // engine call's own behavior (no object ⇒ nothing to authorize here).
        let Ok(mut obj) = ctx.db.get(type_name, id) else {
            continue;
        };
        obj.ensure_fields_deserialized();
        let res = EngineResource {
            db: ctx.db,
            governor: &ctx.governor,
            type_name,
            object: Some(&obj),
            request,
        };
        decide(rules, op, type_name, ctx, &res)?;
    }
    Ok(())
}

/// Filter a terminal read result in place, dropping every object the principal may not read. No-op
/// if rules are off.
pub(crate) fn filter_read(ctx: &ExecContext<'_>, objs: &mut Vec<Object>) {
    let Some(rules) = &ctx.rules else { return };
    objs.retain_mut(|obj| {
        obj.ensure_fields_deserialized();
        let type_name = obj.type_name.clone();
        let res = EngineResource {
            db: ctx.db,
            governor: &ctx.governor,
            type_name: &type_name,
            object: Some(&*obj),
            request: None,
        };
        rules
            .evaluate(Op::Read, &type_name, &ctx.principal, &res)
            .is_allow()
    });
}

/// A [`ResourceAccessor`] over a change event's JSON field snapshot — no db, so relationship
/// projections are unavailable (a rule that traverses evaluates them as absent ⇒ its clause is
/// false). Used for the subscribe per-event filter when the live object is gone (a delete).
struct JsonSnapshot<'a> {
    fields: Option<&'a std::collections::HashMap<String, serde_json::Value>>,
}

impl ResourceAccessor for JsonSnapshot<'_> {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.fields?.get(field).cloned()
    }
    fn resource_relation_one(&self, _rel: &str, _f: &str) -> Option<serde_json::Value> {
        None
    }
    fn resource_relation_many(&self, _rel: &str, _f: &str) -> Option<Vec<serde_json::Value>> {
        None // relationships unavailable in a snapshot ⇒ Absent ⇒ deny (fail closed)
    }
    fn request_field(&self, _f: &str) -> Option<serde_json::Value> {
        None
    }
}

/// Subscribe per-event read gate (P0-DBA-5): may `principal` **read** the object this change event
/// is about? A subscription is a standing read, so an event is delivered only if the subscriber
/// passes the `read` rule (otherwise a sub is a read-rule bypass).
///
/// The rule is evaluated against the event's OWN field `snapshot` — i.e. exactly the data that will
/// be delivered — NOT a live re-fetch of the current object. This is load-bearing (a review finding):
/// re-fetching authorizes the *current* state while delivering the *commit-time* state, so an object
/// that transitions unreadable→readable across two writes would leak the earlier, private-era
/// snapshot. Evaluating against the delivered snapshot makes authorize-and-deliver self-consistent.
///
/// The snapshot carries the object's SCALAR fields only, so a **relationship-based** read rule
/// (`resource.owners.uid`) resolves its relation operands to absent ⇒ the clause is `Unknown` ⇒
/// **the event is denied** (fail-closed). v1 limitation: subscription-visible read rules should key
/// on scalar fields (denormalize an owner id) rather than edge traversals. Callers invoke this only
/// when a rules program is configured.
pub fn event_read_allowed(
    rules: &RulesProgram,
    principal: &rhypedb_authz::Principal,
    type_name: &str,
    snapshot: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> bool {
    let res = JsonSnapshot { fields: snapshot };
    rules.evaluate(Op::Read, type_name, principal, &res).is_allow()
}

fn decide(
    rules: &RulesProgram,
    op: Op,
    type_name: &str,
    ctx: &ExecContext<'_>,
    res: &EngineResource<'_>,
) -> QueryResult<()> {
    match rules.evaluate(op, type_name, &ctx.principal, res) {
        Decision::Allow => Ok(()),
        Decision::Deny => Err(denied(op, type_name)),
    }
}

/// The error a write the rules deny reports.
fn denied(op: Op, type_name: &str) -> QueryError {
    QueryError::PermissionDenied(format!(
        "{} on {type_name} denied by security rules",
        op.as_str()
    ))
}

#[cfg(test)]
mod tests;
