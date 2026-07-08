use std::collections::{HashMap, HashSet};

use bytes::Bytes;
use rhypedb_engine::database::Database;
use rhypedb_engine::object::{
    find_bytes_field_in_raw, find_u64_field_in_raw, FieldMap, Object, Value,
};
use rhypedb_engine::vectorizer::Vectorizer;

use crate::ast::*;
use crate::error::{QueryError, QueryResult};
use crate::governor::Governor;

/// Result of executing a query (intermediate or terminal).
///
/// `IdSet` is the streaming traversal variant: between `Step::Traverse` hops we
/// carry just (type, ids) and skip the per-object deserialize entirely. Every
/// step that doesn't need field data (Update, Delete, Link, Unlink, Limit,
/// Offset) consumes IdSet directly; only Filter materializes. The terminal
/// `execute()` call collapses any remaining IdSet to Objects so server
/// response code and tests see the same shape they always have.
///
/// `IdSetWithFields` extends IdSet for the inverse-traversal fusion path:
/// the reverse-edge index in this DB stores the source object's effective
/// fields as the entry value. After an inverse traversal we carry both the
/// dedup'd source ids AND those carried FieldMaps. A subsequent forward-1:1
/// `.field` traversal can then satisfy itself by reading the target id out
/// of those FieldMaps — saving a per-source forward-edge prefix scan, which
/// is the hot path that dominates multi-hop traversal at scale.
#[derive(Debug)]
pub enum QueryOutput {
    /// A list of objects (from get, filter, scan_type, materialized traversal).
    Objects(Vec<Object>),

    /// A single created/updated object.
    Single(Object),

    /// Void result (delete, link, unlink).
    Done,

    /// A dedup'd set of (type_name, ids) carried between traversal hops. Never
    /// reaches the wire — terminal-materialized in `execute()`.
    IdSet { type_name: String, ids: Vec<u64> },

    /// Like IdSet, but also carries per-id source fields (as raw
    /// serialized-FieldMap bytes) produced by an inverse traversal's
    /// covering reverse-edge values. Consumed by a subsequent forward-1:1
    /// traversal to skip the edge scan — the fusion path extracts the next
    /// hop's target id via `find_u64_field_in_raw` without building a
    /// HashMap. Falls back to plain IdSet semantics for any downstream step
    /// that doesn't take advantage of the fields; `Filter` / terminal
    /// materialize call `deserialize_fields` lazily.
    IdSetWithFields {
        type_name: String,
        items: Vec<(u64, Bytes)>,
    },
}

/// Context for query execution.
pub struct ExecContext<'a> {
    pub db: &'a Database,
    pub vectorizer: Option<&'a Vectorizer>,
    /// Server-wide default HNSW search width (`ef`) for `.similar` queries that
    /// omit an explicit `ef:`. `None` = use the engine's per-shape heuristic. An
    /// explicit per-query `ef:` always wins. The server sets this from
    /// `RHYPEDB_EF`; embedded/library callers leave it `None`.
    pub default_ef: Option<usize>,
    /// Server-wide default rerank pool size for `.similar` queries that omit an
    /// explicit `rerank:`. `None` (or `Some(0)`) = no full-precision rerank by
    /// default. An explicit per-query `rerank:` — including `rerank: 0` (off) —
    /// always wins. The server sets this from `RHYPEDB_RERANK`.
    pub default_rerank: Option<usize>,
    /// Per-query resource governor (row/depth/result/wall-clock caps + `.limit`/`k`
    /// clamps). [`Governor::disabled`] for embedded/library callers (the default);
    /// the server arms it from `RHYPEDB_MAX_QUERY_*`. See [`crate::governor`].
    pub governor: Governor,
    /// The verified end-user identity for this query (P4). The **anonymous** principal for
    /// embedded/library callers and for any request without a valid token — which, under a rules
    /// program, is subject to default-deny (never fail-open). See [`crate::authz`].
    pub principal: rhypedb_authz::Principal,
    /// The compiled default-deny security-rules program, or `None` = **authz disabled**: the
    /// executor takes its exact pre-P4 path (no gate, no principal check), so every rules-off
    /// deployment is byte-unchanged. `Some` ⇒ create/write/read are gated (P0-DBA-1).
    pub rules: Option<std::sync::Arc<rhypedb_authz::RulesProgram>>,
}

impl<'a> ExecContext<'a> {
    /// A context with no server-wide vector-search defaults and the governor
    /// disabled: `.similar` queries fall back to the engine's per-shape `ef`/
    /// `rerank` heuristics, and no resource caps apply. Server callers construct
    /// this then set `default_ef`/`default_rerank` and `governor` from config.
    pub fn new(db: &'a Database, vectorizer: Option<&'a Vectorizer>) -> Self {
        Self {
            db,
            vectorizer,
            default_ef: None,
            default_rerank: None,
            governor: Governor::disabled(),
            principal: rhypedb_authz::Principal::anonymous(),
            rules: None,
        }
    }
}

/// Execute a parsed query against the database.
pub fn execute(ctx: &ExecContext<'_>, query: &Query) -> QueryResult<QueryOutput> {
    // Bare `Type.similar(...)` is a global vector search. Handle it before the
    // generic source scan: `Source::All` would otherwise materialize the entire
    // type just to build an all-permissive candidate set, turning an O(k) k-NN
    // lookup into an O(type-size) scan.
    let (mut result, first_step) = match (&query.source, query.steps.first()) {
        (
            Source::All { type_name },
            Some(Step::Similar { field_name, query: sq, k, ef, rerank }),
        ) => (
            run_similar(ctx, type_name, field_name, sq, *k, None, *ef, *rerank)?,
            1,
        ),
        _ => (execute_source(ctx, &query.source, &query.steps)?, 0),
    };

    // Governor: bound the wall-clock and the traversal depth as the pipeline runs.
    // `depth` counts relationship hops (`Step::Traverse`), whose chained fan-out is
    // the multiplicative-cost risk; the deadline is re-checked before every step so
    // a long-running pipeline fails closed between stages (each step's inner loops
    // also charge rows, which re-checks the deadline in bulk).
    let mut depth = 0usize;
    for step in &query.steps[first_step..] {
        ctx.governor.check_deadline()?;
        if matches!(step, Step::Traverse { .. }) {
            depth += 1;
            ctx.governor.check_depth(depth)?;
        }
        result = execute_step(ctx, result, step, &query.source)?;
    }

    // Streaming-traversal: if the pipeline ended on an `IdSet` or
    // `IdSetWithFields`, materialize it to `Objects` so callers (server
    // response, tests) see the historical shape. This is the only `get`
    // cost we still pay for those IDs.
    match result {
        QueryOutput::IdSet { type_name, ids } => {
            // The final id set is about to be returned — enforce the result-row
            // ceiling before paying the per-id `get` to materialize it.
            ctx.governor.check_result_rows(ids.len())?;
            ctx.governor.charge(ids.len() as u64)?;
            result = QueryOutput::Objects(materialize_ids(ctx.db, &type_name, &ids));
        }
        QueryOutput::IdSetWithFields { type_name, items } => {
            // Terminal materialize: ignore the carried raw bytes (would only
            // hold edge_fields, not the target's object fields) and probe the
            // LSM for the full Object data.
            ctx.governor.check_result_rows(items.len())?;
            ctx.governor.charge(items.len() as u64)?;
            let ids: Vec<u64> = items.into_iter().map(|(id, _)| id).collect();
            result = QueryOutput::Objects(materialize_ids(ctx.db, &type_name, &ids));
        }
        _ => {}
    }

    // P4 read authz: filter the terminal result through the `read` rule (no-op when rules are off).
    // Applies to EVERY returned object shape — Objects AND a mutation's returned Single/post-image —
    // because any object surfaced to the caller is a read: a create/update that returns a row the
    // principal may not read must not leak it (a denied Single collapses to void). Denied rows are
    // dropped (Firestore semantics), so a point-`get` of an unreadable id returns an empty result.
    if ctx.rules.is_some() {
        result = crate::authz::apply_read_filter(ctx, result);
    }

    // A pipeline that produced `Objects` directly (filter, scan, similar, get)
    // rather than a streaming IdSet still owes the result-row ceiling.
    if let QueryOutput::Objects(ref objs) = result {
        ctx.governor.check_result_rows(objs.len())?;
    }

    Ok(result)
}

fn execute_source(
    ctx: &ExecContext<'_>,
    source: &Source,
    steps: &[Step],
) -> QueryResult<QueryOutput> {
    let db = ctx.db;
    match source {
        Source::Get { type_name, id } => {
            let obj = db.get(type_name, *id)?;
            Ok(QueryOutput::Objects(vec![obj]))
        }

        Source::Filter {
            type_name,
            predicate,
        } => {
            // Reject unsupported comparisons (e.g. Json ordering) up front, for
            // any predicate shape — before either the pushdown or the full scan.
            validate_predicate_for_type(db, type_name, predicate)?;
            // Rule-based access-path picker: a single comparison pushes straight
            // to the index/zone fast path (with an effective limit if the next
            // step is a bare `.limit(N)`); a top-level AND pushes its most
            // selective indexed conjunct and filters the residual in memory.
            // Anything it can't profitably narrow falls through to the full scan.
            // Clamp the pushed `.limit(N)` to the governor's `max_limit` so the
            // indexed fast path can't be asked to buffer an absurd match count.
            let pushed_limit = leading_limit(steps).map(|n| ctx.governor.clamp_limit(n));
            if let Some(mut objects) = plan_filter_scan(db, type_name, predicate, pushed_limit)? {
                // Fast-path objects may carry `raw_fields` — if a downstream
                // step inspects them via `obj.fields`, eagerly populate now.
                // No-op when raw_fields is None.
                for obj in &mut objects {
                    obj.ensure_fields_deserialized();
                }
                // Charge the narrowed match set against the examined-rows budget.
                ctx.governor.charge(objects.len() as u64)?;
                return Ok(QueryOutput::Objects(objects));
            }
            // No index/zone pushdown applied — this is an UNINDEXED full scan, the
            // sharpest DoS. The governor refuses it (fail-closed) if the type is
            // over budget, before materializing anything.
            let mut all = scan_all_objects(ctx, type_name)?;
            for obj in &mut all {
                obj.ensure_fields_deserialized();
            }
            let filtered = all
                .into_iter()
                .filter(|obj| evaluate_predicate(predicate, &obj.fields))
                .collect();
            Ok(QueryOutput::Objects(filtered))
        }

        Source::Create { type_name, fields } => {
            let field_map = literal_map_to_field_map(db, type_name, fields)?;
            // P4 create authz: gate the incoming fields before the write (no-op when rules off).
            crate::authz::gate_create(ctx, type_name, &field_map)?;
            let obj = db.create(type_name, field_map)?;
            Ok(QueryOutput::Single(obj))
        }

        Source::CreateBatch { type_name, rows } => {
            // Bound a bulk insert against the row budget (like the Update/Delete/
            // Link/Unlink mutation steps), so a giant createBatch is refused.
            ctx.governor.charge(rows.len() as u64)?;
            let field_maps: Vec<FieldMap> = rows
                .iter()
                .map(|r| literal_map_to_field_map(db, type_name, r))
                .collect::<QueryResult<_>>()?;
            // P4 create authz: every row must pass the `create` rule (no-op when rules off) — the
            // whole batch fails closed if any row is denied.
            for fm in &field_maps {
                crate::authz::gate_create(ctx, type_name, fm)?;
            }
            let objects = db.create_batch(type_name, field_maps)?;
            Ok(QueryOutput::Objects(objects))
        }

        Source::All { type_name } => {
            // A bare listing. The governor refuses it (fail-closed) if the type is
            // over the row budget; a leading `.limit(N)` is NOT pushed into the scan
            // (the underlying limited scan drops leading tombstones and would return
            // incomplete results), so a `.limit` on a huge unindexed listing is
            // refused — add an indexed filter to page a large type.
            let all = scan_all_objects(ctx, type_name)?;
            Ok(QueryOutput::Objects(all))
        }
    }
}

fn execute_step(
    ctx: &ExecContext<'_>,
    current: QueryOutput,
    step: &Step,
    source: &Source,
) -> QueryResult<QueryOutput> {
    let db = ctx.db;
    match step {
        Step::Traverse { field_name } => {
            // Extract the source type without consuming `current` yet — we
            // need to peek at IdSetWithFields contents for the fusion path
            // before falling through to ids_from_output.
            let source_type_str = output_type_name(&current, source).ok_or_else(|| {
                QueryError::InvalidArgument(format!(
                    "no relation field {field_name}: unknown source type"
                ))
            })?;

            let (target_type, is_inverse, is_one_to_one_forward) = db
                .schema()
                .get_type(&source_type_str)
                .and_then(|td| td.get_field(field_name))
                .and_then(|fd| match &fd.field_type {
                    rhypedb_schema::FieldType::Relation(rel) => Some((
                        rel.target_type.clone(),
                        fd.inverse().is_some(),
                        !rel.is_many && fd.inverse().is_none(),
                    )),
                    _ => None,
                })
                .ok_or_else(|| {
                    QueryError::InvalidArgument(format!(
                        "no relation field {field_name} on {source_type_str}"
                    ))
                })?;

            // Fusion fast path: input is either an IdSetWithFields (came
            // from an inverse traversal) OR an Objects set whose items
            // carry raw_fields (came from a previous fusion that landed
            // on covered data). Either way, the next-hop's id is embedded
            // in the input bytes, readable via `find_u64_field_in_raw`
            // — no HashMap build, no edge scan for this hop.
            //
            // Second-degree win: when the covering ALSO carries
            // `<field>__cover` (the target object's own serialized fields,
            // embedded at link time), we emit Objects whose raw_fields
            // ARE the target's object data — terminal materialize then
            // constructs Objects directly without an LSM probe per id.
            // Avoids the 700-user `multi_get` on the 2-hop bench shape.
            //
            // Third-degree (3-hop) win: when those Objects from a previous
            // fusion step feed into ANOTHER forward-1:1, their raw_fields
            // are the link-time cover, which `with_nested_forward_covers`
            // augments with `<next>__cover`. Re-entering this fast path
            // via the Objects branch extracts the 3rd-degree target from
            // raw_fields without another LSM probe.
            let fusion_input: Option<Vec<(u64, Bytes)>> = if is_one_to_one_forward {
                match &current {
                    QueryOutput::IdSetWithFields { items, .. }
                        if items.iter().any(|(_, bytes)| {
                            find_u64_field_in_raw(bytes, field_name.as_str()).is_some()
                        }) =>
                    {
                        Some(items.clone())
                    }
                    QueryOutput::Objects(objs)
                        if objs.iter().any(|o| {
                            o.raw_fields
                                .as_ref()
                                .and_then(|b| {
                                    find_u64_field_in_raw(b, field_name.as_str())
                                })
                                .is_some()
                        }) =>
                    {
                        Some(
                            objs.iter()
                                .filter_map(|o| {
                                    o.raw_fields.as_ref().map(|b| (o.id, b.clone()))
                                })
                                .collect(),
                        )
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some(items) = fusion_input {
                // Governor: charge this 1:1-forward hop's input against the row
                // budget (and re-check the deadline), like the streaming path
                // charges its edges. Output is 1:1-bounded by the input.
                ctx.governor.charge(items.len() as u64)?;
                let items = &items[..];
                let cover_field_name = format!("{field_name}__cover");
                let cover_v_field_name = format!("{field_name}__cover_v");
                let mut seen: HashSet<u64> = HashSet::with_capacity(items.len());
                let mut covered_items: Vec<(u64, Bytes)> = Vec::with_capacity(items.len());
                let mut id_only: Vec<u64> = Vec::new();
                let mut any_covered = false;
                for (_id, bytes) in items {
                    if let Some(tid) = find_u64_field_in_raw(bytes, field_name.as_str())
                        && seen.insert(tid)
                    {
                        // Cover staleness check: the cover writer stamped
                        // the target's generation at write time under
                        // `<field>__cover_v`. If the target has been updated
                        // since then, its current generation is higher; we
                        // route the id through the `id_only` bucket so a
                        // fresh probe replaces the stale blob.
                        let cover = find_bytes_field_in_raw(bytes, &cover_field_name);
                        let embedded_v =
                            find_u64_field_in_raw(bytes, cover_v_field_name.as_str())
                                .unwrap_or(0);
                        let live_v = db.object_version(target_type.as_str(), tid);
                        // `live_v != 0` is the existence guard. Every live
                        // object is born at generation >= 1, so version 0 means
                        // the target is either DELETED (its counter was
                        // forgotten) or predates the born-at-1 change. Trusting
                        // a cover whose target reads 0 would let a deleted
                        // object surface as a phantom (a never-updated target's
                        // cover_v is also 0, so `embedded_v == live_v` alone
                        // can't tell "alive at v0" from "deleted"). Falling
                        // through to a probe is correct for both; it costs the
                        // fast path only for legacy v0 covers, which self-heal
                        // on the next update/relink.
                        if let Some(c) = cover
                            && embedded_v == live_v
                            && live_v != 0
                        {
                            any_covered = true;
                            covered_items.push((tid, c));
                        } else {
                            id_only.push(tid);
                        }
                    }
                }
                if any_covered && id_only.is_empty() {
                    // Every dedup'd target carried fresh covering data —
                    // construct full Objects right now (with `raw_fields`
                    // populated for zero-copy wire encoding) and skip the
                    // terminal `multi_get` entirely. The single biggest
                    // 2-hop win: the 700-user point lookup pass vanishes.
                    let objects: Vec<Object> = covered_items
                        .into_iter()
                        .map(|(tid, cover)| {
                            Object::from_raw(target_type.clone(), tid, cover)
                        })
                        .collect();
                    return Ok(QueryOutput::Objects(objects));
                }
                if any_covered {
                    // Mixed case: some targets are fresh-covered, some are
                    // stale or never-covered. Build Objects directly for
                    // the fresh ones, and probe the LSM only for the
                    // remaining bucket — keeps the fast-path benefit for
                    // everything that's still good without rebuilding all
                    // rev_edges synchronously on the writer side.
                    let mut objects: Vec<Object> =
                        Vec::with_capacity(covered_items.len() + id_only.len());
                    for (tid, cover) in covered_items {
                        objects.push(Object::from_raw(target_type.clone(), tid, cover));
                    }
                    if !id_only.is_empty() {
                        let probed = db
                            .get_many_lazy(target_type.as_str(), &id_only)
                            .map_err(|e| QueryError::InvalidArgument(e.to_string()))?;
                        objects.extend(probed);
                    }
                    return Ok(QueryOutput::Objects(objects));
                }
                // Nothing covered (or every cover was stale) — emit id-only
                // and let the downstream materialize_ids/multi_get handle
                // the lookup. Same as pre-covering behaviour.
                let mut out = Vec::with_capacity(covered_items.len() + id_only.len());
                for (tid, _) in covered_items {
                    out.push(tid);
                }
                out.extend(id_only);
                return Ok(QueryOutput::IdSet {
                    type_name: target_type,
                    ids: out,
                });
            }

            // Streaming path: extract (type, ids), walk links in one batched
            // LSM pass.
            let (source_type, source_ids) = ids_from_output(current, source)?;
            let groups = db.get_links_many(&source_type, &source_ids, field_name)?;

            // Governor: charge the edges examined this hop (pre-dedup) against the
            // row budget and re-check the deadline. A chained traversal's
            // multiplicative fan-out fails closed here before it compounds into the
            // next hop. The seed scan cap already bounds `source_ids`, so one hop's
            // edge count is itself bounded.
            let hop_edges: u64 = groups.iter().map(|g| g.len() as u64).sum();
            ctx.governor.charge(hop_edges)?;

            // Inverse traversals: preserve the per-source covering bytes that
            // get_links_many returned (these carry the source's effective
            // fields so the next forward-1:1 hop can fuse via
            // `find_u64_field_in_raw`). Forward traversals just dedup IDs.
            if is_inverse {
                let mut seen: HashSet<u64> = HashSet::with_capacity(source_ids.len());
                let mut items: Vec<(u64, Bytes)> = Vec::new();
                for group in groups {
                    for (target_id, edge_bytes) in group {
                        if seen.insert(target_id) {
                            items.push((target_id, edge_bytes));
                        }
                    }
                }
                Ok(QueryOutput::IdSetWithFields {
                    type_name: target_type,
                    items,
                })
            } else {
                let mut seen: HashSet<u64> = HashSet::with_capacity(source_ids.len());
                let mut out: Vec<u64> = Vec::with_capacity(source_ids.len());
                for group in groups {
                    for (target_id, _edge_bytes) in group {
                        if seen.insert(target_id) {
                            out.push(target_id);
                        }
                    }
                }
                Ok(QueryOutput::IdSet {
                    type_name: target_type,
                    ids: out,
                })
            }
        }

        Step::Filter { predicate } => {
            // Reject unsupported comparisons (e.g. Json ordering) for the current
            // type before materializing, so the contract holds for a chained
            // `.filter()` after a traversal too.
            if let Some(tn) = output_type_name(&current, source) {
                validate_predicate_for_type(db, &tn, predicate)?;
            }
            // Filter is the one step that genuinely needs field data, so an
            // IdSet input must materialize here. Subsequent traversals will
            // re-collapse to IdSet via ids_from_output.
            //
            // For IdSetWithFields we still go through `materialize_ids`
            // rather than deserialize the carried covering bytes — those
            // hold the SOURCE object's fields (the previous hop), not the
            // current type's fields. The Filter predicate is against the
            // current type, so a fresh LSM probe is required.
            let mut objects = match current {
                QueryOutput::IdSet { type_name, ids } => {
                    materialize_ids(db, &type_name, &ids)
                }
                QueryOutput::IdSetWithFields { type_name, items } => {
                    let ids: Vec<u64> = items.into_iter().map(|(id, _)| id).collect();
                    materialize_ids(db, &type_name, &ids)
                }
                other => extract_objects(other)?,
            };
            // get_many emits Objects with raw_fields populated for the wire
            // shortcut — predicate evaluation needs the decoded FieldMap.
            for obj in &mut objects {
                obj.ensure_fields_deserialized();
            }
            let filtered = objects
                .into_iter()
                .filter(|obj| evaluate_predicate(predicate, &obj.fields))
                .collect();
            Ok(QueryOutput::Objects(filtered))
        }

        Step::Update { fields } => {
            // Update needs (type, id) only — work directly from IDs.
            let (type_name, ids) = ids_from_output(current, source)?;
            // Bound a bulk update against the row budget (and re-check the deadline).
            ctx.governor.charge(ids.len() as u64)?;
            let field_map = literal_map_to_field_map(db, &type_name, fields)?;
            // P4 write authz: gate each PRE-mutation object against the `update` rule, with the
            // incoming fields visible as `request.*` (no-op when rules off). P0-DBA-6: checking the
            // stored object before the write stops a rule like `resource.author == request.auth.uid`
            // from being defeated by the same update that rewrites `author`.
            crate::authz::gate_write(
                ctx,
                rhypedb_authz::Op::Update,
                &type_name,
                &ids,
                Some(&field_map),
            )?;
            let mut updated = Vec::with_capacity(ids.len());
            for id in &ids {
                updated.push(db.update(&type_name, *id, field_map.clone())?);
            }
            if updated.len() == 1 {
                Ok(QueryOutput::Single(updated.remove(0)))
            } else {
                Ok(QueryOutput::Objects(updated))
            }
        }

        Step::Delete => {
            let (type_name, ids) = ids_from_output(current, source)?;
            ctx.governor.charge(ids.len() as u64)?;
            // P4 write authz: gate each pre-mutation object against the `delete` rule (no-op off).
            crate::authz::gate_write(ctx, rhypedb_authz::Op::Delete, &type_name, &ids, None)?;
            for id in ids {
                db.delete(&type_name, id)?;
            }
            Ok(QueryOutput::Done)
        }

        Step::Link {
            target_type,
            target_id,
            edge_fields,
        } => {
            let (source_type, ids) = ids_from_output(current, source)?;
            ctx.governor.charge(ids.len() as u64)?;
            // P4 write authz: a link mutates the source object's relationships → gate it as an
            // `update` on the source type (no-op when rules off).
            crate::authz::gate_write(ctx, rhypedb_authz::Op::Update, &source_type, &ids, None)?;
            // Resolve the relation field once — every source row has the
            // same type at this point. (Resolved before coercing the edge
            // fields, which need the relation's declared edge-field types.)
            let field_name = resolve_relation_field(db, &source_type, target_type)?;
            let edge_map = if edge_fields.is_empty() {
                None
            } else {
                // Coerce each edge literal to its DECLARED edge-field type
                // (RelationType.edge_fields[].scalar_type), mirroring the
                // schema-directed coercion object fields get — otherwise an
                // `f64`/`u64`/`i32`/`i64` edge field is stored as the wrong
                // `Value` variant. `db.link()` validates the result.
                Some(edge_literal_map_to_field_map(
                    db,
                    &source_type,
                    &field_name,
                    edge_fields,
                )?)
            };
            for id in ids {
                db.link(&source_type, id, &field_name, *target_id, edge_map.clone())?;
            }
            Ok(QueryOutput::Done)
        }

        Step::Unlink {
            target_type,
            target_id,
        } => {
            let (source_type, ids) = ids_from_output(current, source)?;
            ctx.governor.charge(ids.len() as u64)?;
            // P4 write authz: an unlink mutates the source object's relationships → gate as `update`.
            crate::authz::gate_write(ctx, rhypedb_authz::Op::Update, &source_type, &ids, None)?;
            let field_name = resolve_relation_field(db, &source_type, target_type)?;
            for id in ids {
                db.unlink(&source_type, id, &field_name, *target_id)?;
            }
            Ok(QueryOutput::Done)
        }

        Step::Similar {
            field_name,
            query,
            k,
            ef,
            rerank,
        } => {
            // Type AND candidate set come from the live pipeline:
            // `A.filter(...).similar(...)` restricts to the filtered rows;
            // `A.bs.similar(...)` searches the traversed B type. (A bare
            // `Type.similar(...)` is handled in `execute` as a global search.)
            let (type_name, incoming_ids) = ids_from_output(current, source)?;
            let allowed: HashSet<u64> = incoming_ids.into_iter().collect();
            run_similar(
                ctx,
                &type_name,
                field_name,
                query,
                *k,
                Some(&allowed),
                *ef,
                *rerank,
            )
        }

        Step::Limit { count } => {
            // Clamp the requested limit to the governor's `max_limit` so a
            // `.limit(1_000_000_000)` can't be used to demand an absurd buffer.
            let count = ctx.governor.clamp_limit(*count);
            match current {
                QueryOutput::IdSet { type_name, mut ids } => {
                    ids.truncate(count);
                    Ok(QueryOutput::IdSet { type_name, ids })
                }
                QueryOutput::IdSetWithFields { type_name, mut items } => {
                    items.truncate(count);
                    Ok(QueryOutput::IdSetWithFields { type_name, items })
                }
                other => {
                    let mut objects = extract_objects(other)?;
                    objects.truncate(count);
                    Ok(QueryOutput::Objects(objects))
                }
            }
        }

        Step::Offset { count } => match current {
            QueryOutput::IdSet { type_name, ids } => {
                let ids: Vec<u64> = ids.into_iter().skip(*count).collect();
                Ok(QueryOutput::IdSet { type_name, ids })
            }
            QueryOutput::IdSetWithFields { type_name, items } => {
                let items: Vec<_> = items.into_iter().skip(*count).collect();
                Ok(QueryOutput::IdSetWithFields { type_name, items })
            }
            other => {
                let objects = extract_objects(other)?;
                let skipped = objects.into_iter().skip(*count).collect();
                Ok(QueryOutput::Objects(skipped))
            }
        },
    }
}

/// Upper bound on the HNSW search width (`ef`) and the candidate pool a single
/// `.similar()` query may request. Caps the cost of a hostile or fat-fingered
/// `ef:`/`rerank:` so one query can't exhaust a small (1–4 core) deployment VM.
const MAX_VECTOR_SEARCH_POOL: usize = 10_000;

/// Fill in the server-wide `ef`/`rerank` defaults for a `.similar` step, but
/// only where the query OMITTED the parameter. An explicit per-query value —
/// including `rerank: 0` (rerank off) — always takes priority over the default,
/// matching the "overridable per query" contract documented for `RHYPEDB_EF` /
/// `RHYPEDB_RERANK`.
fn resolve_similar_defaults(
    ef: Option<usize>,
    rerank: Option<usize>,
    default_ef: Option<usize>,
    default_rerank: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    (ef.or(default_ef), rerank.or(default_rerank))
}

/// The HNSW search parameters resolved for one `.similar` step: the candidate
/// pool to retrieve (`search_k`), the HNSW search width (`ef`), and whether to
/// run a full-precision rerank.
#[derive(Debug, PartialEq, Eq)]
struct ResolvedSearch {
    search_k: usize,
    ef: usize,
    rerank: bool,
}

/// Resolve the effective HNSW search parameters from a `.similar` step plus the
/// server-wide defaults. First fills in the `RHYPEDB_EF`/`RHYPEDB_RERANK`
/// defaults where the query OMITTED the parameter (an explicit per-query value —
/// including `rerank: 0` = off — always wins), then applies the same over-fetch
/// heuristic and safety clamps regardless of whether the value came from the
/// query or a default. So a configured default behaves exactly as if the caller
/// had written it into the query. `restricted` = an upstream filter narrowed the
/// candidate set (changes the over-fetch and the `ef` heuristic).
///
/// Pure and side-effect-free so the default-flow + clamp behaviour is unit-tested
/// deterministically, without the (non-deterministic) HNSW/quantizer path.
fn resolve_search(
    ef: Option<usize>,
    rerank: Option<usize>,
    default_ef: Option<usize>,
    default_rerank: Option<usize>,
    k: usize,
    restricted: bool,
) -> ResolvedSearch {
    // Server-wide defaults fill in only where the query omitted ef/rerank.
    let (ef, rerank) = resolve_similar_defaults(ef, rerank, default_ef, default_rerank);

    // `rerank: 0` (and the absent case) means "no full-precision rerank".
    let rerank = rerank.filter(|&r| r > 0);

    // Over-fetch only when restricting (post-filtering discards some hits); a
    // global search needs just k. For a SELECTIVE filter (a small restrict set)
    // the vectorizer takes an exact brute-force path that ignores this over-fetch
    // and `ef` entirely (it scores the whole set, so it never under-fills); the
    // over-fetch below only matters for the HNSW post-filter path used when the
    // restrict set is large (> EXACT_FILTER_MAX), where heavy filtering can still
    // yield < k (a documented residual). `rerank: N` raises the retrieved pool to
    // at least N so the full-precision re-score has N candidates to work over.
    let base_k = if restricted {
        k.saturating_mul(4).max(k)
    } else {
        k
    };
    let search_k = rerank
        .map_or(base_k, |r| r.max(base_k))
        .min(MAX_VECTOR_SEARCH_POOL);

    // User-supplied (or defaulted) `ef` overrides the heuristic. Floor it to
    // `search_k` (HNSW can't surface more than `ef` candidates) and cap it for
    // safety so a hostile/fat-fingered value can't exhaust a small VM.
    let heuristic_ef = if restricted {
        search_k.saturating_mul(2).max(64)
    } else {
        k.max(50)
    };
    let ef = ef
        .unwrap_or(heuristic_ef)
        .max(search_k)
        .min(MAX_VECTOR_SEARCH_POOL);

    ResolvedSearch {
        search_k,
        ef,
        rerank: rerank.is_some(),
    }
}

/// Run a vector similarity search over `type_name.field_name`.
///
/// `restrict = None` searches the whole type (a bare `Type.similar(...)`), so
/// no candidate set is materialized and `k` results are fetched directly.
/// `Some(set)` restricts results to the incoming pipeline ids (a preceding
/// `.filter()`/`.traverse()`), over-fetching to compensate for post-filtering.
/// An empty restriction short-circuits to no results — which also avoids
/// calling the vectorizer with a possibly-wrong type when an upstream step
/// emptied the pipeline after a traversal.
#[allow(clippy::too_many_arguments)]
fn run_similar(
    ctx: &ExecContext<'_>,
    type_name: &str,
    field_name: &str,
    query: &SimilarQuery,
    k: usize,
    restrict: Option<&HashSet<u64>>,
    ef: Option<usize>,
    rerank: Option<usize>,
) -> QueryResult<QueryOutput> {
    // An empty candidate set can never match — return early before touching the
    // vectorizer, so a pipeline that emptied after a traversal doesn't trigger a
    // wrong-type index lookup (and doesn't even require a vectorizer).
    if matches!(restrict, Some(set) if set.is_empty()) {
        return Ok(QueryOutput::Objects(Vec::new()));
    }

    // Governor: clamp `k` to `max_limit`. The retrieval pool is already bounded by
    // `MAX_VECTOR_SEARCH_POOL`, but an explicit `k` cap makes the returned-row
    // bound a stated invariant rather than an emergent one, and re-checks the
    // deadline before the (potentially heavy) HNSW search.
    let k = ctx.governor.clamp_limit(k);
    ctx.governor.check_deadline()?;

    let vectorizer = ctx.vectorizer.ok_or_else(|| {
        QueryError::Type("vector similarity search requires a vectorizer".into())
    })?;

    // Resolve the effective search width / pool / rerank from the query's
    // ef:/rerank: PLUS the server-wide RHYPEDB_EF / RHYPEDB_RERANK defaults (an
    // explicit per-query value always wins). See `resolve_search`.
    let ResolvedSearch {
        search_k,
        ef,
        rerank,
    } = resolve_search(
        ef,
        rerank,
        ctx.default_ef,
        ctx.default_rerank,
        k,
        restrict.is_some(),
    );

    let results = match query {
        SimilarQuery::Text(text) => vectorizer.search_text(
            type_name,
            field_name,
            text,
            search_k,
            ef,
            rerank,
            restrict,
        )?,
        SimilarQuery::Vector(vec) => vectorizer.search_vector(
            type_name,
            field_name,
            vec,
            search_k,
            ef,
            rerank,
            restrict,
        )?,
    };

    let objects: Vec<Object> = results
        .iter()
        .filter(|(id, _dist)| restrict.is_none_or(|set| set.contains(id)))
        .take(k)
        .filter_map(|(id, _dist)| ctx.db.get(type_name, *id).ok())
        .collect();

    Ok(QueryOutput::Objects(objects))
}

/// Read the type_name from any QueryOutput shape, without consuming it.
/// Used by `Step::Traverse` to look up schema metadata before deciding
/// whether to take the fusion fast-path or the streaming path.
fn output_type_name(output: &QueryOutput, source: &Source) -> Option<String> {
    match output {
        QueryOutput::IdSet { type_name, .. } => Some(type_name.clone()),
        QueryOutput::IdSetWithFields { type_name, .. } => Some(type_name.clone()),
        QueryOutput::Single(obj) => Some(obj.type_name.clone()),
        QueryOutput::Objects(objs) => objs
            .first()
            .map(|o| o.type_name.clone())
            .or_else(|| source_type_name(source)),
        QueryOutput::Done => source_type_name(source),
    }
}

/// Pull (type_name, ids) from any QueryOutput shape.
///
/// IdSet/IdSetWithFields: zero-copy. Objects/Single: extract `(type, id)`
/// pairs without touching field data. Done: empty set, type derived from
/// `source`.
fn ids_from_output(
    output: QueryOutput,
    source: &Source,
) -> QueryResult<(String, Vec<u64>)> {
    match output {
        QueryOutput::IdSet { type_name, ids } => Ok((type_name, ids)),
        QueryOutput::IdSetWithFields { type_name, items } => {
            Ok((type_name, items.into_iter().map(|(id, _)| id).collect()))
        }
        QueryOutput::Single(obj) => Ok((obj.type_name, vec![obj.id])),
        QueryOutput::Objects(objs) => {
            // Empty: derive type from source for a sensible error/no-op shape.
            if objs.is_empty() {
                let t = source_type_name(source).unwrap_or_default();
                return Ok((t, Vec::new()));
            }
            let type_name = objs[0].type_name.clone();
            let ids: Vec<u64> = objs.into_iter().map(|o| o.id).collect();
            Ok((type_name, ids))
        }
        QueryOutput::Done => {
            let t = source_type_name(source).unwrap_or_default();
            Ok((t, Vec::new()))
        }
    }
}

fn source_type_name(source: &Source) -> Option<String> {
    match source {
        Source::Get { type_name, .. }
        | Source::Filter { type_name, .. }
        | Source::Create { type_name, .. }
        | Source::CreateBatch { type_name, .. }
        | Source::All { type_name } => Some(type_name.clone()),
    }
}

/// Bulk materialize a list of IDs into Objects. Uses the engine's batched
/// `get_many_lazy` so each Object carries `raw_fields = Some(bytes)` — the
/// wire encoder ships the stored payload directly, skipping
/// `deserialize_fields` + HashMap construction + drop for objects that flow
/// straight from LSM to the TCP response. Consumers that read `obj.fields`
/// (Filter predicate, HTTP/JSON path) call `ensure_fields_deserialized`
/// first.
fn materialize_ids(db: &Database, type_name: &str, ids: &[u64]) -> Vec<Object> {
    db.get_many_lazy(type_name, ids).unwrap_or_default()
}

fn extract_objects(output: QueryOutput) -> QueryResult<Vec<Object>> {
    match output {
        QueryOutput::Objects(objs) => Ok(objs),
        QueryOutput::Single(obj) => Ok(vec![obj]),
        QueryOutput::Done => Ok(Vec::new()),
        QueryOutput::IdSet { .. } | QueryOutput::IdSetWithFields { .. } => {
            Err(QueryError::Type(
                "internal: extract_objects called on IdSet variant — should have \
                 been materialized first via ids_from_output or the terminal materialize"
                    .into(),
            ))
        }
    }
}

/// Find the relation field on `source_type` that points to `target_type`.
/// Returns the field name if exactly one match; errors on zero or multiple matches.
fn resolve_relation_field(
    db: &Database,
    source_type: &str,
    target_type: &str,
) -> QueryResult<String> {
    let type_def = db
        .schema()
        .get_type(source_type)
        .ok_or_else(|| crate::error::QueryError::Type(format!("unknown type {source_type}")))?;

    let matches: Vec<&str> = type_def
        .fields
        .iter()
        .filter_map(|f| match &f.field_type {
            rhypedb_schema::FieldType::Relation(rel) if rel.target_type == target_type => {
                Some(f.name.as_str())
            }
            _ => None,
        })
        .collect();

    match matches.as_slice() {
        [] => Err(crate::error::QueryError::InvalidArgument(format!(
            "no relation field on {source_type} points to {target_type}"
        ))),
        [single] => Ok((*single).to_string()),
        many => Err(crate::error::QueryError::InvalidArgument(format!(
            "ambiguous: {source_type} has multiple relation fields to {target_type}: {many:?}"
        ))),
    }
}

fn evaluate_predicate(predicate: &Predicate, fields: &FieldMap) -> bool {
    match predicate {
        Predicate::Compare {
            field_path,
            op,
            value,
        } => {
            let field_value = fields.get(field_path.as_str());
            match field_value {
                Some(fv) => compare_values(fv, op, value),
                None => false,
            }
        }
        Predicate::And(left, right) => {
            evaluate_predicate(left, fields) && evaluate_predicate(right, fields)
        }
        Predicate::Or(left, right) => {
            evaluate_predicate(left, fields) || evaluate_predicate(right, fields)
        }
    }
}

fn compare_values(field_val: &Value, op: &CompareOp, literal: &Literal) -> bool {
    match (field_val, literal) {
        (Value::String(a), Literal::String(b)) => compare_ord(a.as_str(), op, b.as_str()),
        (Value::U32(a), Literal::Int(b)) => compare_ord(*a as i64, op, *b),
        (Value::U64(a), Literal::Int(b)) => compare_ord(*a as i128, op, *b as i128),
        (Value::I32(a), Literal::Int(b)) => compare_ord(*a as i64, op, *b),
        (Value::I64(a), Literal::Int(b)) => compare_ord(*a, op, *b),
        (Value::F32(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::F64(a), Literal::Float(b)) => compare_ord(*a, op, *b),
        (Value::F32(a), Literal::Int(b)) => compare_ord(*a as f64, op, *b as f64 ),
        (Value::F64(a), Literal::Int(b)) => compare_ord(*a, op, *b as f64 ),
        (Value::U32(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::U64(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::I32(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::I64(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::Bool(a), Literal::Bool(b)) => match op {
            CompareOp::Eq => a == b,
            CompareOp::Ne => a != b,
            _ => false,
        },
        // DateTime compares as its i64 epoch-millis — against an int literal or
        // an RFC 3339 string (a malformed string never matches).
        (Value::DateTime(a), Literal::Int(b)) => compare_ord(*a, op, *b),
        (Value::DateTime(a), Literal::String(b)) => {
            match rhypedb_engine::object::datetime_millis_from_rfc3339(b) {
                Ok(bms) => compare_ord(*a, op, bms),
                Err(_) => false,
            }
        }
        // Json supports equality only (ordering is rejected before pushdown).
        (Value::Json(a), Literal::Json(b)) => json_eq(a, b, op),
        (Value::Json(a), Literal::String(b)) => {
            json_eq(a, &serde_json::Value::String(b.clone()), op)
        }
        (Value::Json(a), Literal::Int(b)) => json_eq(a, &serde_json::json!(b), op),
        (Value::Json(a), Literal::Float(b)) => json_eq(a, &serde_json::json!(b), op),
        (Value::Json(a), Literal::Bool(b)) => json_eq(a, &serde_json::Value::Bool(*b), op),
        (_, Literal::Null) => match op {
            CompareOp::Eq => matches!(field_val, Value::Null),
            CompareOp::Ne => !matches!(field_val, Value::Null),
            _ => false,
        },
        _ => false,
    }
}

/// Json equality (`==`/`!=`); any ordering op is `false` (no total order).
fn json_eq(a: &serde_json::Value, b: &serde_json::Value, op: &CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        _ => false,
    }
}

/// The declared scalar type of `type_name.field`, if it is a scalar field.
fn field_scalar_type(
    db: &Database,
    type_name: &str,
    field: &str,
) -> Option<rhypedb_schema::ScalarType> {
    db.schema()
        .get_type(type_name)?
        .fields
        .iter()
        .find(|f| f.name == field)
        .and_then(|f| match &f.field_type {
            rhypedb_schema::FieldType::Scalar(s) => Some(s.clone()),
            _ => None,
        })
}

/// Validate every leaf comparison in a filter predicate against `type_name`'s
/// schema, independent of `And`/`Or` nesting and of whether the comparison takes
/// the index pushdown — so each rule holds for a nested or post-traversal filter,
/// not only a single top-level comparison. Rejecting up front (rather than
/// letting the comparison fall through to a silent empty result) turns an
/// unsupported filter into a clear error. Current rules, by the field's declared
/// scalar type:
///   * `Json` — ordering (`<`/`>`/…) is rejected (no total order over arbitrary
///     JSON); descending into a key/path (`.meta.k`) is rejected (only
///     whole-value `==`/`!=` is supported — JSON path querying is a future card).
///   * `Bytes` — every comparison is rejected EXCEPT `== null` / `!= null`
///     (a blob has no useful query-language equality/ordering form; an exact
///     match would require base64-encoding the whole blob into the query, and
///     ordering is meaningless).
fn validate_predicate_for_type(
    db: &Database,
    type_name: &str,
    predicate: &Predicate,
) -> QueryResult<()> {
    use rhypedb_schema::ScalarType as ST;
    match predicate {
        Predicate::Compare { field_path, op, value } => {
            // For a dotted path the head segment names the field; a SCALAR head
            // with a sub-path means descending into a scalar (e.g. a JSON key).
            // A relation head returns `None` here, leaving relation traversals
            // untouched.
            let descends = field_path.contains('.');
            let head = field_path.split('.').next().unwrap_or(field_path);
            match field_scalar_type(db, type_name, head) {
                Some(ST::Json) => {
                    if descends {
                        return Err(QueryError::Type(
                            "querying into a Json field by key/path is not supported yet \
                             (a Json field supports whole-value `==` / `!=` only)"
                                .into(),
                        ));
                    }
                    if matches!(
                        op,
                        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
                    ) {
                        return Err(QueryError::Type(
                            "ordering comparisons are not supported on Json fields".into(),
                        ));
                    }
                }
                Some(ST::Bytes) => {
                    let null_check = !descends
                        && matches!(value, Literal::Null)
                        && matches!(op, CompareOp::Eq | CompareOp::Ne);
                    if !null_check {
                        return Err(QueryError::Type(
                            "comparisons on Bytes fields are not supported \
                             (only `== null` / `!= null`)"
                                .into(),
                        ));
                    }
                }
                _ => {}
            }
            Ok(())
        }
        Predicate::And(l, r) | Predicate::Or(l, r) => {
            validate_predicate_for_type(db, type_name, l)?;
            validate_predicate_for_type(db, type_name, r)
        }
    }
}


fn compare_ord<T: PartialOrd>(a: T, op: &CompareOp, b: T) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => a <= b,
        CompareOp::Gt => a > b,
        CompareOp::Ge => a >= b,
    }
}

/// Build a FieldMap from parsed literals, coercing each value into the
/// target type's declared scalar type. Schema-directed coercion is what makes
/// `f64`/`u64`/`i32`/`i64` fields reachable from the query language: the
/// parser only ever produces `Int`/`Float` literals, and the engine's
/// `validate_value` requires an exact `Value`/`ScalarType` match.
fn literal_map_to_field_map(
    db: &Database,
    type_name: &str,
    literals: &HashMap<String, Literal>,
) -> QueryResult<FieldMap> {
    let type_def = db.schema().get_type(type_name);
    let mut fields = FieldMap::new();
    for (name, lit) in literals {
        let target = type_def
            .and_then(|td| td.get_field(name))
            .and_then(|fd| match &fd.field_type {
                rhypedb_schema::FieldType::Scalar(st) => Some(st.clone()),
                _ => None,
            });
        fields.insert(name.clone(), literal_to_value(lit, target)?);
    }
    Ok(fields)
}

/// Build an edge-`FieldMap` from parsed literals, coercing each value into the
/// relation's DECLARED edge-field scalar type (`RelationType.edge_fields[].
/// scalar_type`). This is the edge-field analogue of [`literal_map_to_field_map`]:
/// without it, `A.link(B.get(2), { weight: 3.5 })` for a declared `f64` edge
/// field would store `Value::F32` (the best-effort default). An edge-field name
/// the relation doesn't declare gets `target = None` (best-effort) — `db.link()`
/// then rejects it as an unknown edge field, mirroring how `create`/`update`
/// reject unknown object fields.
fn edge_literal_map_to_field_map(
    db: &Database,
    source_type: &str,
    field_name: &str,
    literals: &HashMap<String, Literal>,
) -> QueryResult<FieldMap> {
    let edge_types: HashMap<&str, rhypedb_schema::ScalarType> = db
        .schema()
        .get_type(source_type)
        .and_then(|td| td.get_field(field_name))
        .and_then(|fd| match &fd.field_type {
            rhypedb_schema::FieldType::Relation(rel) => Some(rel),
            _ => None,
        })
        .map(|rel| {
            rel.edge_fields
                .iter()
                .map(|ef| (ef.name.as_str(), ef.scalar_type.clone()))
                .collect()
        })
        .unwrap_or_default();

    let mut fields = FieldMap::new();
    for (name, lit) in literals {
        let target = edge_types.get(name.as_str()).cloned();
        fields.insert(name.clone(), literal_to_value(lit, target)?);
    }
    Ok(fields)
}

/// Convert a literal into a `Value`. When the field's declared scalar type is
/// known (`target`), the literal is coerced to match it; otherwise (relation
/// target ids, edge fields the schema doesn't know) a best-effort mapping is
/// used.
fn literal_to_value(lit: &Literal, target: Option<rhypedb_schema::ScalarType>) -> QueryResult<Value> {
    use rhypedb_schema::ScalarType as ST;

    // Null is valid for any field type.
    if let Literal::Null = lit {
        return Ok(Value::Null);
    }

    let Some(st) = target else {
        return Ok(match lit {
            Literal::String(s) => Value::String(s.clone()),
            Literal::Int(i) if (0..=u32::MAX as i64).contains(i) => Value::U32(*i as u32),
            Literal::Int(i) => Value::I64(*i),
            Literal::Float(f) => Value::F32(*f as f32),
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Json(v) => Value::Json(v.clone()),
            Literal::Null => Value::Null,
        });
    };

    let mismatch = || QueryError::Type(format!("literal {lit:?} is not valid for a {st:?} field"));
    Ok(match (&st, lit) {
        (ST::String, Literal::String(s)) => Value::String(s.clone()),
        (ST::Bool, Literal::Bool(b)) => Value::Bool(*b),
        (ST::U32, Literal::Int(i)) if (0..=u32::MAX as i64).contains(i) => Value::U32(*i as u32),
        (ST::U64, Literal::Int(i)) if *i >= 0 => Value::U64(*i as u64),
        (ST::I32, Literal::Int(i)) if (i32::MIN as i64..=i32::MAX as i64).contains(i) => {
            Value::I32(*i as i32)
        }
        (ST::I64, Literal::Int(i)) => Value::I64(*i),
        (ST::F32, Literal::Float(f)) => Value::F32(*f as f32),
        (ST::F64, Literal::Float(f)) => Value::F64(*f),
        // Integer literals widen into float fields (`score: 5` for an `f64`).
        (ST::F32, Literal::Int(i)) => Value::F32(*i as f32),
        (ST::F64, Literal::Int(i)) => Value::F64(*i as f64),
        // Bytes: a base64 string literal decodes to raw bytes.
        (ST::Bytes, Literal::String(s)) => {
            Value::Bytes(rhypedb_engine::object::bytes_from_base64(s).map_err(QueryError::Type)?)
        }
        // DateTime: an RFC 3339 string, or an integer epoch-millis literal.
        (ST::DateTime, Literal::String(s)) => Value::DateTime(
            rhypedb_engine::object::datetime_millis_from_rfc3339(s).map_err(QueryError::Type)?,
        ),
        (ST::DateTime, Literal::Int(i)) => Value::DateTime(*i),
        // Json: a raw JSON container, or any scalar literal wrapped as JSON.
        (ST::Json, Literal::Json(v)) => Value::Json(v.clone()),
        (ST::Json, Literal::String(s)) => Value::Json(serde_json::Value::String(s.clone())),
        (ST::Json, Literal::Int(i)) => Value::Json(serde_json::json!(i)),
        (ST::Json, Literal::Float(f)) => Value::Json(serde_json::json!(f)),
        (ST::Json, Literal::Bool(b)) => Value::Json(serde_json::Value::Bool(*b)),
        _ => return Err(mismatch()),
    })
}

/// Full-type scan, bounded by the governor.
///
/// With the governor disabled (embedded/library path) this is byte-identical to
/// the old `db.scan_type(type_name)` — no cap, no error, no behaviour change.
///
/// When a row budget is set, it first takes a TOMBSTONE-CORRECT live count of the
/// type (a count-only keyspace scan that retains only a liveness bool per key, not
/// the object payloads) and FAILS CLOSED if that exceeds `max_rows_scanned` — so a
/// huge unindexed scan is refused BEFORE materializing any objects, without ever
/// silently truncating (which would return wrong results for a listing or a
/// post-scan filter). A type within budget is then materialized in full and
/// charged.
///
/// NB: a previous `scan_type_limited(cap+1)` bound was removed — its underlying
/// *limited* storage scan stops after N DISTINCT keys INCLUDING tombstones, so
/// leading deleted-but-uncompacted rows could make it return fewer live rows than
/// exist, silently corrupting `Type.limit(N)` and the filter fallback. The
/// count-then-scan here is correct because the count merges the WHOLE prefix.
fn scan_all_objects(ctx: &ExecContext<'_>, type_name: &str) -> QueryResult<Vec<Object>> {
    let g = &ctx.governor;
    if !g.is_enabled() {
        return Ok(ctx.db.scan_type(type_name)?);
    }
    if let Some(cap) = g.scan_cap() {
        // Refuse before materializing if the type is larger than the budget. The
        // count is bool-per-key (no object payloads), so this bounds peak memory of
        // an over-budget scan to the key-set rather than the (much larger) object
        // set — and it never truncates.
        let live = ctx.db.count_type(type_name)?;
        if live > cap as u64 {
            return Err(QueryError::ResourceLimitExceeded(format!(
                "query scans more than {cap} rows of type '{type_name}'; \
                 narrow it with an indexed filter or a smaller query"
            )));
        }
    }
    let objs = ctx.db.scan_type(type_name)?;
    g.charge(objs.len() as u64)?;
    Ok(objs)
}

/// Recognize the shape `Filter(Compare { int_field, op, int_literal })` and
/// push it down to `Database::filter_scan` so storage can use the secondary
/// index (when available) or zone-prune blocks (when not). Returns
/// `Ok(Some(objects))` on match, `Ok(None)` if the predicate is too complex
/// for the fast path (And/Or, string compare, missing field, etc.) — in
/// which case the caller falls back to the full scan + filter.
///
/// `limit` is the caller's request to stop after N matches (typically
/// extracted from a trailing `.limit(N)` step). Best-effort.
fn try_filter_scan(
    db: &Database,
    snapshot: u64,
    type_name: &str,
    predicate: &Predicate,
    limit: Option<usize>,
) -> QueryResult<Option<Vec<Object>>> {
    let Predicate::Compare { field_path, op, value } = predicate else {
        return Ok(None);
    };
    // No nested field paths for now — only top-level field.
    if field_path.contains('.') {
        return Ok(None);
    }
    // Alias to disambiguate from the query crate's CompareOp imported via
    // `use crate::ast::*`. The engine re-exports the storage enum so this
    // crate doesn't need to depend on storage directly.
    use rhypedb_engine::CompareOp as StorageOp;
    let storage_op = match op {
        CompareOp::Eq => StorageOp::Eq,
        CompareOp::Ne => StorageOp::Ne,
        CompareOp::Lt => StorageOp::Lt,
        CompareOp::Le => StorageOp::Le,
        CompareOp::Gt => StorageOp::Gt,
        CompareOp::Ge => StorageOp::Ge,
    };
    // DateTime and Json fields don't fit the generic literal dispatch below
    // (their stored `Value` variant isn't one of int/string/bool/float):
    //   * Json — no total order; equality is handled by the full scan + the
    //     engine's `json_eq`. (Ordering is already rejected up front by
    //     `validate_predicate_for_type`.) Route to the full scan.
    //   * DateTime — push down to the *integer* `filter_scan` after coercing
    //     the literal to epoch-millis (DateTime shares the I64 ordered
    //     index/zone encoding). An int literal IS millis; an RFC 3339 string
    //     is parsed to millis. A malformed string — or any other literal kind
    //     — routes to the full scan, so `compare_values` owns those semantics
    //     (a malformed string matches nothing, identically) and there is one
    //     source of truth.
    if let Some(st) = field_scalar_type(db, type_name, field_path) {
        use rhypedb_schema::ScalarType as ST;
        match st {
            ST::Json => return Ok(None),
            ST::DateTime => {
                let millis = match value {
                    Literal::Int(i) => *i,
                    Literal::String(s) => {
                        match rhypedb_engine::object::datetime_millis_from_rfc3339(s) {
                            Ok(ms) => ms,
                            Err(_) => return Ok(None),
                        }
                    }
                    _ => return Ok(None),
                };
                return Ok(Some(db.filter_scan_at(
                    snapshot, type_name, field_path, storage_op, millis, limit,
                )?));
            }
            _ => {}
        }
    }
    // Every scalar-typed literal can route to a typed filter_scan; null
    // falls through. Bytes-indexed predicates don't have a query-language
    // form yet (no Bytes literal at the parser level) — engine API users
    // call `Database::filter_scan_bytes` directly.
    match value {
        Literal::Int(i) => Ok(Some(db.filter_scan_at(
            snapshot, type_name, field_path, storage_op, *i, limit,
        )?)),
        Literal::String(s) => Ok(Some(db.filter_scan_str_at(
            snapshot, type_name, field_path, storage_op, s, limit,
        )?)),
        Literal::Bool(b) => Ok(Some(db.filter_scan_bool_at(
            snapshot, type_name, field_path, storage_op, *b, limit,
        )?)),
        Literal::Float(f) => Ok(Some(db.filter_scan_float_at(
            snapshot, type_name, field_path, storage_op, *f, limit,
        )?)),
        // A raw JSON container literal has no typed pushdown; the full scan +
        // `compare_values` handles it. (DateTime fields were already pushed
        // down, and Json fields routed to `Ok(None)`, by the field-type guard
        // above; this arm only fires for a Json literal on some other field.)
        Literal::Json(_) => Ok(None),
        Literal::Null => Ok(None),
    }
}

/// Rule-based access-path picker for a `Source::Filter`. Returns
/// `Ok(Some(objects))` when the filter can be served from a secondary / unique
/// index, or `Ok(None)` when the caller should do a full scan.
///
/// Strategies, most selective first:
///   * **single `Compare`** — a `@unique` equality is a ≤1-row `u:` point lookup;
///     otherwise the existing index/zone pushdown (`try_filter_scan`).
///   * **top-level `AND`** — a `@unique`-equality conjunct (≤1 row) > intersection
///     of two `@indexed` equality conjuncts > the single most-selective `@indexed`
///     conjunct; the FULL predicate is then re-filtered in memory.
///   * **top-level `OR`** — when EVERY disjunct is index-eligible, the union of
///     each disjunct's matches (else a full scan).
///
/// Behavior-preserving by construction: every materialized branch produces a
/// SUPERSET of the result, re-filters by the full predicate, and restores
/// object-id order — so the output equals `scan_all_objects + evaluate_predicate`,
/// incl. a trailing `.limit`/`.offset`. One snapshot is pinned for the whole
/// filter, so a multi-scan intersection / union is a single point-in-time view.
fn plan_filter_scan(
    db: &Database,
    type_name: &str,
    predicate: &Predicate,
    pushed_limit: Option<usize>,
) -> QueryResult<Option<Vec<Object>>> {
    // Pin ONE snapshot: every scan/probe below reads at the same point-in-time,
    // so an intersection / union can't blend several committed states.
    let snapshot = db.read_snapshot();

    // A single comparison: a @unique equality is the most selective path (≤1
    // row); otherwise the existing index/zone fast path (limit preserved).
    if matches!(predicate, Predicate::Compare { .. }) {
        if let Some(objs) = unique_eq_probe(db, snapshot, type_name, predicate)? {
            return Ok(Some(refilter_and_sort(predicate, objs)));
        }
        return try_filter_scan(db, snapshot, type_name, predicate, pushed_limit);
    }

    // Dispatch on the predicate ROOT so the OR path isn't swallowed by the AND
    // flatten (and vice-versa).
    match predicate {
        Predicate::And(..) => {
            let mut conjuncts: Vec<&Predicate> = Vec::new();
            flatten_and(predicate, &mut conjuncts);
            plan_and(db, snapshot, type_name, predicate, &conjuncts)
        }
        Predicate::Or(..) => {
            let mut disjuncts: Vec<&Predicate> = Vec::new();
            flatten_or(predicate, &mut disjuncts);
            plan_or_union(db, snapshot, type_name, predicate, &disjuncts)
        }
        // `Compare` is handled above; no other predicate shapes exist.
        Predicate::Compare { .. } => Ok(None),
    }
}

/// Plan a top-level `AND` (its flattened `conjuncts`). Priority: a `@unique`
/// equality conjunct (≤1 candidate) > an intersection of two `@indexed` equality
/// conjuncts > the single most-selective `@indexed` conjunct. Each conjunct is a
/// NECESSARY condition, so any one (or the intersection) yields a SUPERSET of the
/// result that the full re-filter narrows exactly.
fn plan_and(
    db: &Database,
    snapshot: u64,
    type_name: &str,
    predicate: &Predicate,
    conjuncts: &[&Predicate],
) -> QueryResult<Option<Vec<Object>>> {
    // 1) A @unique-equality conjunct yields ≤1 candidate — unbeatable.
    for c in conjuncts {
        if let Some(objs) = unique_eq_probe(db, snapshot, type_name, c)? {
            return Ok(Some(refilter_and_sort(predicate, objs)));
        }
    }

    // 2) Intersection of the first two @indexed EQUALITY conjuncts (rank 0:
    //    non-Bool, non-float Eq). Each Eq scan is a narrow per-value range, so
    //    intersecting two is cheap and yields a much smaller candidate set than
    //    pushing one. Bool/range conjuncts stay residual — poor narrowing terms,
    //    still enforced by the re-filter.
    let eq_generators: Vec<&Predicate> = conjuncts
        .iter()
        .copied()
        .filter(|c| conjunct_index_generator(db, type_name, c).is_some_and(|(rank, _)| rank == 0))
        .take(2)
        .collect();
    if eq_generators.len() == 2 {
        let mut sets: Vec<Vec<Object>> = Vec::with_capacity(2);
        for g in &eq_generators {
            if let Some(objs) = try_filter_scan(db, snapshot, type_name, g, None)? {
                sets.push(objs);
            }
        }
        if sets.len() == 2 {
            let intersected = intersect_by_id(sets);
            return Ok(Some(refilter_and_sort(predicate, intersected)));
        }
    }

    // 3) Single most-selective @indexed generator (v1), pushed UNBOUNDED so the
    //    candidate set is tombstone-sound and complete (a *bounded* index scan
    //    counts tombstones against its budget and could under-return).
    let Some(generator) = conjuncts
        .iter()
        .copied()
        .filter_map(|c| conjunct_index_generator(db, type_name, c))
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, c)| c)
    else {
        return Ok(None);
    };
    let Some(candidates) = try_filter_scan(db, snapshot, type_name, generator, None)? else {
        // Defensive: e.g. a malformed RFC 3339 DateTime literal pushes nothing.
        return Ok(None);
    };
    Ok(Some(refilter_and_sort(predicate, candidates)))
}

/// Plan a top-level `OR` (its flattened `disjuncts`) as a UNION of per-disjunct
/// index scans — but only when EVERY disjunct is index-eligible. If any disjunct
/// would need a full scan (non-indexed field, nested And/Or, Ne, float/Json/Null,
/// etc.), a row matching ONLY that disjunct could be missed, so we bail to the
/// full scan (`Ok(None)`).
fn plan_or_union(
    db: &Database,
    snapshot: u64,
    type_name: &str,
    predicate: &Predicate,
    disjuncts: &[&Predicate],
) -> QueryResult<Option<Vec<Object>>> {
    if disjuncts.len() < 2 {
        return Ok(None);
    }
    let mut unioned: Vec<Object> = Vec::new();
    for d in disjuncts {
        // A @unique-equality disjunct (≤1 row).
        if let Some(objs) = unique_eq_probe(db, snapshot, type_name, d)? {
            unioned.extend(objs);
            continue;
        }
        // An @indexed-generator disjunct (Eq / range).
        if conjunct_index_generator(db, type_name, d).is_some() {
            match try_filter_scan(db, snapshot, type_name, d, None)? {
                Some(objs) => unioned.extend(objs),
                None => return Ok(None), // defensive: eligible but unpushable
            }
            continue;
        }
        // Ineligible disjunct → can't union soundly.
        return Ok(None);
    }
    // `unioned` ⊇ result (the union of each disjunct's matches). Re-filter the
    // full predicate, dedup by id (an object can match several disjuncts), sort.
    Ok(Some(refilter_and_sort(predicate, unioned)))
}

/// If `compare` is `field == value` on a `@unique`, non-float, non-Json field
/// whose literal coerces to a non-null Value, return the ≤1 matching object via
/// the `u:` index. `Ok(None)` when it's not a usable unique-equality probe
/// (wrong op/type, not unique, Null/uncoercible literal).
///
/// The CALLER must still re-filter the full predicate over the returned object:
/// a `@unique` field updated to Null leaves a stale `u:` entry, and the probe is
/// often only one conjunct/disjunct of a larger predicate.
fn unique_eq_probe(
    db: &Database,
    snapshot: u64,
    type_name: &str,
    compare: &Predicate,
) -> QueryResult<Option<Vec<Object>>> {
    let Predicate::Compare { field_path, op, value } = compare else {
        return Ok(None);
    };
    if !matches!(op, CompareOp::Eq) || field_path.contains('.') {
        return Ok(None);
    }
    if !db.is_field_unique(type_name, field_path) {
        return Ok(None);
    }
    use rhypedb_schema::ScalarType as ST;
    let Some(st) = field_scalar_type(db, type_name, field_path) else {
        return Ok(None);
    };
    // Float Eq: the `u:` bytes for -0.0 differ from +0.0, but `compare_values`
    // treats them equal → a probe could MISS a stored -0.0. Json: serialized-byte
    // equality may diverge from `json_eq`. Both stay on the full-scan path.
    if !matches!(
        &st,
        ST::String | ST::U32 | ST::U64 | ST::I32 | ST::I64 | ST::DateTime | ST::Bool
    ) {
        return Ok(None);
    }
    // Coerce the literal to the field's declared Value so the probe's index bytes
    // match what was written. A coercion error (type mismatch) or a Null value
    // means "not a usable unique probe".
    let Ok(coerced) = literal_to_value(value, Some(st)) else {
        return Ok(None);
    };
    if matches!(coerced, Value::Null) {
        return Ok(None);
    }
    let found = db.find_by_unique_at(snapshot, type_name, field_path, &coerced)?;
    Ok(Some(found.into_iter().collect()))
}

/// Intersect candidate object sets BY id, returning the objects present in ALL
/// sets (taken from the smallest set to minimize work). Every input was scanned
/// at one snapshot, so an id present in every set denotes one consistent object.
fn intersect_by_id(mut sets: Vec<Vec<Object>>) -> Vec<Object> {
    sets.sort_by_key(|s| s.len()); // smallest base = least filtering work
    let mut iter = sets.into_iter();
    let Some(base) = iter.next() else {
        return Vec::new();
    };
    let id_sets: Vec<std::collections::HashSet<u64>> =
        iter.map(|s| s.iter().map(|o| o.id).collect()).collect();
    base.into_iter()
        .filter(|o| id_sets.iter().all(|ids| ids.contains(&o.id)))
        .collect()
}

/// Re-filter `candidates` (a superset of the result) by the FULL predicate, then
/// restore object-id order and dedup by id. Shared by every materialized planner
/// branch so each is a drop-in for `scan_all_objects + evaluate_predicate` (the
/// index yields value order, and a union can repeat an object across disjuncts).
fn refilter_and_sort(predicate: &Predicate, mut candidates: Vec<Object>) -> Vec<Object> {
    for obj in &mut candidates {
        obj.ensure_fields_deserialized();
    }
    let mut result: Vec<Object> = candidates
        .into_iter()
        .filter(|obj| evaluate_predicate(predicate, &obj.fields))
        .collect();
    result.sort_by_key(|obj| obj.id);
    result.dedup_by_key(|obj| obj.id);
    result
}

/// Collect the top-level AND conjuncts of `predicate`, flattening nested ANDs.
/// A `Compare` or an `Or(..)` subtree is one opaque conjunct (the latter is
/// never an index generator but is still enforced by the in-memory re-filter).
fn flatten_and<'a>(predicate: &'a Predicate, out: &mut Vec<&'a Predicate>) {
    match predicate {
        Predicate::And(left, right) => {
            flatten_and(left, out);
            flatten_and(right, out);
        }
        other => out.push(other),
    }
}

/// Collect the top-level OR disjuncts of `predicate`, flattening nested ORs.
/// A `Compare` or an `And(..)` subtree is one opaque disjunct.
fn flatten_or<'a>(predicate: &'a Predicate, out: &mut Vec<&'a Predicate>) {
    match predicate {
        Predicate::Or(left, right) => {
            flatten_or(left, out);
            flatten_or(right, out);
        }
        other => out.push(other),
    }
}

/// If `conjunct` is a `Compare` that can drive a genuine secondary-index probe,
/// return `(selectivity_rank, conjunct)` — lower rank = likelier more selective
/// (non-`Bool` `Eq` = 0, range = 1, `Bool` `Eq` = 2). Otherwise `None`. Excludes:
///   * nested field paths (`a.b`) — no index pushdown form;
///   * `Ne` — matches most rows, a poor narrowing term;
///   * non-`@indexed` fields — `filter_scan` would do a zone/full scan, not an
///     index scan, so pushing gains nothing over the residual filter;
///   * `Null`/`Json`/`*` literals — no typed pushdown;
///   * literal/field type mismatches and out-of-domain literals — `is_field_indexed`
///     is membership-only, so a mismatched literal (e.g. an `Int` against an
///     `@indexed String`) or an out-of-range int (e.g. a negative literal against
///     a `u32`) would silently route `filter_scan` to its non-index fallback (a
///     full scan). We admit a literal only when it matches the field's scalar
///     type AND fits its domain the way `try_filter_scan`/`filter_scan` route it,
///     guaranteeing the index path is actually taken;
///   * float `Eq` — the index key for `-0.0` differs from `+0.0`, but
///     `compare_values` treats them equal, so an Eq index probe would MISS a
///     stored `-0.0` (a non-superset). Float ranges are sound (the `-0.0`/`+0.0`
///     key order is consistent with `compare_values`, and the residual filter
///     drops any extras), so only float Eq is excluded — it stays correct in the
///     in-memory residual filter.
fn conjunct_index_generator<'a>(
    db: &Database,
    type_name: &str,
    conjunct: &'a Predicate,
) -> Option<(u8, &'a Predicate)> {
    let Predicate::Compare {
        field_path,
        op,
        value,
    } = conjunct
    else {
        return None;
    };
    if field_path.contains('.') || matches!(op, CompareOp::Ne) {
        return None;
    }
    if !db.is_field_indexed(type_name, field_path) {
        return None;
    }
    use rhypedb_schema::ScalarType as ST;
    let st = field_scalar_type(db, type_name, field_path)?;
    let hits_index = match (&st, value) {
        // Integer fields: the literal must fit the declared width, or
        // `filter_scan` takes its out-of-range fallback (a full scan).
        (ST::U32, Literal::Int(i)) => (0..=u32::MAX as i64).contains(i),
        (ST::U64, Literal::Int(i)) => *i >= 0,
        (ST::I32, Literal::Int(i)) => (i32::MIN as i64..=i32::MAX as i64).contains(i),
        (ST::I64, Literal::Int(_)) => true,
        // DateTime shares the i64 ordered encoding; an int literal IS millis and
        // an RFC 3339 string is coerced to millis by `try_filter_scan`.
        (ST::DateTime, Literal::Int(_) | Literal::String(_)) => true,
        (ST::String, Literal::String(_)) => true,
        (ST::Bool, Literal::Bool(_)) => true,
        // Float ranges are sound supersets; float Eq is excluded (see doc above).
        (ST::F32 | ST::F64, Literal::Float(_)) => !matches!(op, CompareOp::Eq),
        _ => false,
    };
    if !hits_index {
        return None;
    }
    // Selectivity heuristic (no statistics yet): equality is usually the most
    // selective, EXCEPT on a 2-valued `Bool` — rank that LAST so a sibling range
    // or higher-cardinality equality is preferred when both are available. (Only
    // a heuristic for *which* indexed conjunct to push; every choice is a correct
    // superset, so this never affects results.)
    let rank = match (op, &st) {
        (CompareOp::Eq, ST::Bool) => 2,
        (CompareOp::Eq, _) => 0,
        _ => 1, // ranges Lt/Le/Gt/Ge (Ne already excluded above)
    };
    Some((rank, conjunct))
}

/// If the query's first step is `.limit(N)`, return `Some(N)` so the filter
/// scan can stop after N matches.
///
/// Sound under the language's sequential pipeline semantics: a trailing
/// `.offset(M)` after the limit can't occur, because the parser rejects
/// `.limit(...).offset(...)` (offset must precede limit). A *leading*
/// `.offset(M)` instead makes `steps.first()` an `Offset`, so this returns
/// `None` and no limit is pushed. Steps that only shrink/transform the result
/// after a leading limit (a further filter or traverse) still honor the cap,
/// since the scan already returned at most N rows.
fn leading_limit(steps: &[Step]) -> Option<usize> {
    match steps.first()? {
        Step::Limit { count } => Some(*count),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_query;
    use rhypedb_schema::parser::parse_schema;

    fn test_db(dir: &std::path::Path) -> std::sync::Arc<Database> {
        let schema = parse_schema(
            r#"
            type User {
                name: String
                age: u32
                active: Bool
                friends: [User] @on_delete(remove)
            }
            "#,
        )
        .unwrap();
        Database::open(schema, dir).unwrap()
    }

    fn create_user(db: &Database, name: &str, age: u32) -> Object {
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String(name.into()));
        f.insert("age".into(), Value::U32(age));
        f.insert("active".into(), Value::Bool(true));
        db.create("User", f).unwrap()
    }

    fn product_db(dir: &std::path::Path) -> std::sync::Arc<Database> {
        let schema = parse_schema(
            r#"
            type Product {
                name: String
                price: f64
                big: u64
                rating: i32
                delta: i64
            }
            "#,
        )
        .unwrap();
        Database::open(schema, dir).unwrap()
    }

    #[test]
    fn resolve_similar_defaults_precedence() {
        // Omitted in the query -> the server-wide default fills in.
        assert_eq!(
            resolve_similar_defaults(None, None, Some(128), Some(40)),
            (Some(128), Some(40))
        );
        // An explicit per-query value always wins over the default.
        assert_eq!(
            resolve_similar_defaults(Some(256), Some(64), Some(128), Some(40)),
            (Some(256), Some(64))
        );
        // `rerank: 0` is an EXPLICIT "off" and must NOT be replaced by a default.
        assert_eq!(
            resolve_similar_defaults(None, Some(0), Some(128), Some(40)),
            (Some(128), Some(0))
        );
        // No default configured -> the query is unchanged (engine heuristics apply).
        assert_eq!(
            resolve_similar_defaults(None, None, None, None),
            (None, None)
        );
        // Mixed: ef explicit, rerank defaulted.
        assert_eq!(
            resolve_similar_defaults(Some(300), None, Some(128), Some(40)),
            (Some(300), Some(40))
        );
    }

    /// Deterministic proof that the server-wide defaults actually FLOW INTO the
    /// resolved HNSW parameters (the part a recall-based test can't show, because
    /// HNSW construction and the quantizer rotation are both non-deterministic).
    #[test]
    fn resolve_search_applies_defaults_and_clamps() {
        // Global (unrestricted), no query params, no defaults: the engine's own
        // heuristic ef (k.max(50) = 50), pool = k, no rerank.
        assert_eq!(
            resolve_search(None, None, None, None, 1, false),
            ResolvedSearch { search_k: 1, ef: 50, rerank: false }
        );
        // A server `RHYPEDB_EF` default REACHES the resolved ef — 200, not the
        // heuristic 50. This is exactly what would NOT happen if run_similar
        // failed to thread the default through.
        assert_eq!(
            resolve_search(None, None, Some(200), None, 1, false),
            ResolvedSearch { search_k: 1, ef: 200, rerank: false }
        );
        // A server `RHYPEDB_RERANK` default turns rerank ON and grows the pool to
        // at least the rerank size (ef stays at the heuristic, floored to search_k).
        assert_eq!(
            resolve_search(None, None, None, Some(10), 1, false),
            ResolvedSearch { search_k: 10, ef: 50, rerank: true }
        );
        // An explicit per-query value wins over BOTH defaults; `rerank: 0` is an
        // explicit "off" that the default must not resurrect.
        assert_eq!(
            resolve_search(Some(7), Some(0), Some(200), Some(10), 5, false),
            ResolvedSearch { search_k: 5, ef: 7, rerank: false }
        );
        // Restricted (post-filter) path: over-fetch base_k = 4*k, heuristic ef =
        // max(2*search_k, 64); the default ef still flows in and floors to search_k.
        assert_eq!(
            resolve_search(None, None, Some(200), None, 2, true),
            ResolvedSearch { search_k: 8, ef: 200, rerank: false }
        );
        // Safety cap: a default far above MAX_VECTOR_SEARCH_POOL is clamped (so a
        // server-wide default can't exhaust a small VM any more than a per-query
        // value could).
        assert_eq!(
            resolve_search(None, None, Some(1_000_000), Some(1_000_000), 1, false),
            ResolvedSearch {
                search_k: MAX_VECTOR_SEARCH_POOL,
                ef: MAX_VECTOR_SEARCH_POOL,
                rerank: true
            }
        );
    }

    /// End-to-end smoke that the full defaults path — `ExecContext` defaults →
    /// `execute()` → `run_similar` → `search_vector` → materialize — runs and
    /// returns the correct exact match with the defaults set. The deterministic
    /// proof that the defaults change the *resolved parameters* lives in
    /// `resolve_search_applies_defaults_and_clamps`; this test guards the wiring
    /// end-to-end (no panic / type error) over a real index. Uses a
    /// bring-your-own `Vector` field so no embed model is needed.
    #[test]
    fn similar_query_executes_with_server_defaults_set() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Doc {
                title: String
                v: Vector<4>
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema.clone(), dir.path()).unwrap();

        let vectorizer = Vectorizer::new(
            std::sync::Arc::clone(db.storage()),
            schema,
            db.type_ids().clone(),
            db.field_ids().clone(),
        )
        .unwrap();

        // Four orthonormal vectors, one per Doc.
        let basis = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut ids = Vec::new();
        for (i, vec) in basis.iter().enumerate() {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("doc-{i}")));
            let obj = db.create("Doc", f).unwrap();
            vectorizer.ingest_vector("Doc", obj.id, "v", vec).unwrap();
            ids.push(obj.id);
        }

        // Query equals the second basis vector -> Doc #1 is the unique nearest.
        let q = parse_query("Doc.similar(.v, [0.0, 1.0, 0.0, 0.0], k: 1)").unwrap();
        let top_id = |ctx: &ExecContext<'_>| match execute(ctx, &q).unwrap() {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 1);
                objs[0].id
            }
            other => panic!("expected Objects, got {other:?}"),
        };

        // No defaults: engine heuristics; explicit defaults via the context.
        let mut ctx = ExecContext::new(&db, Some(&vectorizer));
        let baseline = top_id(&ctx);
        ctx.default_ef = Some(200);
        ctx.default_rerank = Some(10);
        let with_defaults = top_id(&ctx);

        assert_eq!(baseline, ids[1]);
        assert_eq!(with_defaults, ids[1]);
    }

    #[test]
    fn create_coerces_literals_to_declared_scalar_types() {
        // Before the fix every Float -> F32 and every small Int -> U32, so
        // f64/u64/i32/i64 fields rejected the value with TypeMismatch. Now the
        // literal is coerced to the field's declared scalar type.
        let dir = tempfile::tempdir().unwrap();
        let db = product_db(dir.path());
        let ctx = ExecContext::new(&db, None);

        let q = parse_query(
            r#"Product.create({ name: "a", price: 3.5, big: 5, rating: -7, delta: 9 })"#,
        )
        .unwrap();
        let obj = match execute(&ctx, &q).unwrap() {
            QueryOutput::Single(o) => o,
            other => panic!("expected Single, got {other:?}"),
        };
        assert_eq!(obj.fields.get("price"), Some(&Value::F64(3.5)));
        assert_eq!(obj.fields.get("big"), Some(&Value::U64(5)));
        assert_eq!(obj.fields.get("rating"), Some(&Value::I32(-7)));
        assert_eq!(obj.fields.get("delta"), Some(&Value::I64(9)));
    }

    #[test]
    fn filter_int_field_with_float_literal_does_not_overmatch() {
        // `.rating > 4.5` on an i32 field used to return ALL rows
        // (filter_scan_float fell back to scan_type); now it filters
        // correctly via the per-row numeric fallback.
        let dir = tempfile::tempdir().unwrap();
        let db = product_db(dir.path());
        let ctx = ExecContext::new(&db, None);
        for (n, r) in [("a", 1), ("b", 5), ("c", 10)] {
            let q = parse_query(&format!(
                r#"Product.create({{ name: "{n}", price: 1.0, big: 1, rating: {r}, delta: 0 }})"#
            ))
            .unwrap();
            execute(&ctx, &q).unwrap();
        }
        let q = parse_query("Product.filter(.rating > 4.5)").unwrap();
        let objs = match execute(&ctx, &q).unwrap() {
            QueryOutput::Objects(o) => o,
            other => panic!("expected Objects, got {other:?}"),
        };
        let mut names: Vec<String> = objs
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn and_predicate_compares_int_field_against_float_literal() {
        // An And predicate bypasses the filter_scan fast path and evaluates
        // via compare_values, which previously lacked I32/I64-vs-Float arms
        // (so `.rating >= 5.0` silently matched nothing).
        let dir = tempfile::tempdir().unwrap();
        let db = product_db(dir.path());
        let ctx = ExecContext::new(&db, None);
        for (n, r, p) in [("a", 1, 10.0), ("b", 5, 20.0), ("c", 8, 200.0)] {
            let q = parse_query(&format!(
                r#"Product.create({{ name: "{n}", price: {p}, big: 1, rating: {r}, delta: 0 }})"#
            ))
            .unwrap();
            execute(&ctx, &q).unwrap();
        }
        let q = parse_query("Product.filter(.rating >= 5.0 && .price < 100.0)").unwrap();
        let objs = match execute(&ctx, &q).unwrap() {
            QueryOutput::Objects(o) => o,
            other => panic!("expected Objects, got {other:?}"),
        };
        let names: Vec<String> = objs
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["b".to_string()]);
    }

    #[test]
    fn similar_on_empty_filtered_set_returns_empty() {
        // A filter that matches nothing before .similar() yields an empty
        // result and must NOT require a vectorizer or trigger a wrong-type
        // index lookup (the empty candidate set short-circuits first).
        let dir = tempfile::tempdir().unwrap();
        let db = product_db(dir.path());
        let ctx = ExecContext::new(&db, None);
        execute(
            &ctx,
            &parse_query(
                r#"Product.create({ name: "a", price: 1.0, big: 1, rating: 1, delta: 0 })"#,
            )
            .unwrap(),
        )
        .unwrap();

        let q = parse_query(r#"Product.filter(.rating > 9999).similar(.name, "x", k: 5)"#).unwrap();
        match execute(&ctx, &q).unwrap() {
            QueryOutput::Objects(o) => assert!(o.is_empty()),
            other => panic!("expected empty Objects, got {other:?}"),
        }
    }

    #[test]
    fn execute_create_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let q = parse_query(r#"User.create({ name: "Alice", age: 30 })"#).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let obj = match result {
            QueryOutput::Single(o) => o,
            _ => panic!("expected Single"),
        };
        assert_eq!(obj.fields.get("name"), Some(&Value::String("Alice".into())));

        let q = parse_query(&format!("User.get({})", obj.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 1);
                assert_eq!(
                    objs[0].fields.get("name"),
                    Some(&Value::String("Alice".into()))
                );
            }
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn create_and_read_datetime_bytes_json() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"type Event { name: String  created: DateTime  blob: Bytes  meta: Json }"#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        // RFC 3339 datetime, base64 bytes, and a raw JSON object literal.
        let q = parse_query(
            r#"Event.create({ name: "launch", created: "1970-01-01T00:00:00.500Z", blob: "aGVsbG8=", meta: { "k": 1, "tags": ["a", "b"] } })"#,
        )
        .unwrap();
        let id = match execute(&ExecContext::new(&db, None), &q).unwrap() {
            QueryOutput::Single(o) => o.id,
            _ => panic!("expected Single"),
        };
        let obj = db.get("Event", id).unwrap();
        assert_eq!(obj.fields.get("created"), Some(&Value::DateTime(500)));
        assert_eq!(
            obj.fields.get("blob"),
            Some(&Value::Bytes(bytes::Bytes::from_static(b"hello")))
        );
        assert_eq!(
            obj.fields.get("meta"),
            Some(&Value::Json(serde_json::json!({"k": 1, "tags": ["a", "b"]})))
        );

        // An integer epoch-millis literal also coerces to DateTime.
        let q2 = parse_query(r#"Event.create({ name: "x", created: 1234 })"#).unwrap();
        let id2 = match execute(&ExecContext::new(&db, None), &q2).unwrap() {
            QueryOutput::Single(o) => o.id,
            _ => panic!(),
        };
        assert_eq!(
            db.get("Event", id2).unwrap().fields.get("created"),
            Some(&Value::DateTime(1234))
        );

        // A malformed datetime / base64 is a clean type error, not a panic.
        let bad_dt = parse_query(r#"Event.create({ created: "not-a-date" })"#).unwrap();
        assert!(execute(&ExecContext::new(&db, None), &bad_dt).is_err());
        let bad_b64 = parse_query(r#"Event.create({ blob: "@@@@" })"#).unwrap();
        assert!(execute(&ExecContext::new(&db, None), &bad_b64).is_err());
    }

    #[test]
    fn json_ordering_is_rejected_everywhere_equality_works() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Author { name: String  docs: [Doc] @inverse(Doc.author) }
            type Doc { name: String  meta: Json  author: Author }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let run = |q: &str| execute(&ExecContext::new(&db, None), &parse_query(q).unwrap());

        let a = match run(r#"Author.create({ name: "ada" })"#).unwrap() {
            QueryOutput::Single(o) => o.id,
            _ => panic!(),
        };
        run(&format!(
            r#"Doc.create({{ name: "d", meta: {{ "k": 1 }}, author: {a} }})"#
        ))
        .unwrap();

        // Ordering on a Json field is a hard error — NOT a silent empty result —
        // as a single comparison, inside a compound predicate, AND in a
        // post-traversal `.filter()` step.
        assert!(run(r#"Doc.filter(.meta > 5)"#).is_err());
        assert!(run(r#"Doc.filter(.meta > 5 && .name == "d")"#).is_err());
        assert!(run(r#"Doc.filter(.name == "d" || .meta >= 1)"#).is_err());
        assert!(run(&format!(r#"Author.get({a}).docs.filter(.meta >= 1)"#)).is_err());

        // Equality on a Json field is allowed (and matches the stored value).
        match run(r#"Doc.filter(.meta == { "k": 1 })"#).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 1),
            _ => panic!("expected Objects"),
        }
        match run(r#"Doc.filter(.meta == { "k": 2 })"#).unwrap() {
            QueryOutput::Objects(objs) => assert!(objs.is_empty()),
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn bytes_comparisons_and_json_path_filters_are_rejected_loudly() {
        // Two silent-empty footguns turned into hard errors:
        //   * a value comparison on a Bytes field (only `== null` / `!= null`
        //     is meaningful), and
        //   * descending into a Json key/path (whole-value `==`/`!=` only;
        //     real JSON path querying is a future card).
        // Both used to fall through to an empty result with no error.
        let dir = tempfile::tempdir().unwrap();
        let schema =
            parse_schema(r#"type Event { name: String  blob: Bytes  meta: Json }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let run = |q: &str| execute(&ExecContext::new(&db, None), &parse_query(q).unwrap());

        run(r#"Event.create({ name: "a", blob: "aGVsbG8=", meta: { "k": 1 } })"#).unwrap();
        run(r#"Event.create({ name: "b", meta: { "k": 2 } })"#).unwrap();

        // Bytes value comparisons are a hard error — as a single comparison,
        // inside a compound predicate, and for ordering ops.
        assert!(run(r#"Event.filter(.blob == "aGVsbG8=")"#).is_err());
        assert!(run(r#"Event.filter(.blob != "aGVsbG8=")"#).is_err());
        assert!(run(r#"Event.filter(.blob > "aGVsbG8=")"#).is_err());
        assert!(run(r#"Event.filter(.name == "a" && .blob == "aGVsbG8=")"#).is_err());
        // Even a malformed-base64 literal errors at validation — as an
        // unsupported Bytes comparison, before (and regardless of) any decode.
        assert!(run(r#"Event.filter(.blob == "@@@")"#).is_err());

        // Bytes null-checks remain allowed (the meaningful "is it set?").
        assert!(run(r#"Event.filter(.blob == null)"#).is_ok());
        assert!(run(r#"Event.filter(.blob != null)"#).is_ok());

        // Descending into a Json key/path is a hard error — single, nested,
        // and post-traversal-step shapes.
        assert!(run(r#"Event.filter(.meta.k == 1)"#).is_err());
        assert!(run(r#"Event.filter(.name == "a" || .meta.k == 1)"#).is_err());

        // Whole-value Json equality still works (regression guard).
        match run(r#"Event.filter(.meta == { "k": 1 })"#).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 1),
            _ => panic!("expected Objects"),
        }
    }

    // -----------------------------------------------------------------
    // DateTime range pushdown via try_filter_scan (card cmqn571cn)
    // -----------------------------------------------------------------

    fn datetime_event_db(dir: &std::path::Path, indexed: bool) -> std::sync::Arc<Database> {
        let decl = if indexed {
            "created: DateTime @indexed"
        } else {
            "created: DateTime"
        };
        let schema = parse_schema(&format!("type Event {{ name: String  {decl} }}")).unwrap();
        let db = Database::open(schema, dir).unwrap();
        for (i, ms) in [-10_000i64, 0, 500, 1000, 1500].iter().enumerate() {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("e{i}")));
            f.insert("created".into(), Value::DateTime(*ms));
            db.create("Event", f).unwrap();
        }
        db
    }

    /// The sorted `created` millis returned by running `q` against `db`.
    fn run_created(db: &Database, q: &str) -> Vec<i64> {
        match execute(&ExecContext::new(db, None), &parse_query(q).unwrap()).unwrap() {
            QueryOutput::Objects(objs) => {
                let mut v: Vec<i64> = objs
                    .iter()
                    .map(|o| match o.fields.get("created") {
                        Some(Value::DateTime(ms)) => *ms,
                        other => panic!("missing/bad created: {other:?}"),
                    })
                    .collect();
                v.sort_unstable();
                v
            }
            other => panic!("expected Objects, got {other:?}"),
        }
    }

    #[test]
    fn try_filter_scan_routes_datetime_literals() {
        let dir = tempfile::tempdir().unwrap();
        let db = datetime_event_db(dir.path(), true);

        let compare = |lit: Literal| Predicate::Compare {
            field_path: "created".into(),
            op: CompareOp::Gt,
            value: lit,
        };

        let snap = db.read_snapshot();
        // Int and RFC 3339 string literals push down (Some) and find the 3 rows > 0.
        let pushed_int =
            try_filter_scan(&db, snap, "Event", &compare(Literal::Int(0)), None).unwrap();
        assert_eq!(pushed_int.as_ref().map(|v| v.len()), Some(3));
        let pushed_str = try_filter_scan(
            &db,
            snap,
            "Event",
            &compare(Literal::String("1970-01-01T00:00:00Z".into())),
            None,
        )
        .unwrap();
        assert_eq!(pushed_str.as_ref().map(|v| v.len()), Some(3));

        // A malformed RFC 3339 string, and non-int/non-string literals, fall back
        // to the full scan (None) so `compare_values` owns those semantics.
        for lit in [
            Literal::String("not-a-date".into()),
            Literal::Float(1.0),
            Literal::Bool(true),
            Literal::Null,
        ] {
            assert!(
                try_filter_scan(&db, snap, "Event", &compare(lit.clone()), None)
                    .unwrap()
                    .is_none(),
                "literal {lit:?} should fall back to the full scan"
            );
        }
    }

    #[test]
    fn datetime_pushdown_matches_full_scan_and_is_literal_agnostic() {
        let dir_idx = tempfile::tempdir().unwrap();
        let dir_plain = tempfile::tempdir().unwrap();
        let db_idx = datetime_event_db(dir_idx.path(), true);
        let db_plain = datetime_event_db(dir_plain.path(), false);

        // For every query, the @indexed (secondary-index pushdown) and the plain
        // (zone-map) DB must return identical, correct result sets.
        let cases: &[(&str, Vec<i64>)] = &[
            (r#"Event.filter(.created > 0)"#, vec![500, 1000, 1500]),
            (r#"Event.filter(.created >= 0)"#, vec![0, 500, 1000, 1500]),
            (r#"Event.filter(.created < 1000)"#, vec![-10_000, 0, 500]),
            (r#"Event.filter(.created <= 0)"#, vec![-10_000, 0]),
            (r#"Event.filter(.created == 500)"#, vec![500]),
            (r#"Event.filter(.created != 500)"#, vec![-10_000, 0, 1000, 1500]),
            // RFC 3339 string literal == the equivalent int literal.
            (
                r#"Event.filter(.created > "1970-01-01T00:00:00Z")"#,
                vec![500, 1000, 1500],
            ),
            // Sub-millisecond precision truncates to 1000ms (so < 1000.5 == < 1000).
            (
                r#"Event.filter(.created < "1970-01-01T00:00:01.0005Z")"#,
                vec![-10_000, 0, 500],
            ),
            // Malformed RFC 3339 string matches nothing (full-scan semantics).
            (r#"Event.filter(.created < "not-a-date")"#, vec![]),
        ];
        for (q, expect) in cases {
            assert_eq!(&run_created(&db_idx, q), expect, "indexed: {q}");
            assert_eq!(&run_created(&db_plain, q), expect, "zone-map: {q}");
        }
    }

    #[test]
    fn execute_create_batch() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let q = parse_query(
            r#"User.create_batch([
                { name: "Alice", age: 25, active: true },
                { name: "Bob", age: 30, active: true },
                { name: "Carol", age: 35, active: false }
            ])"#,
        )
        .unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert_eq!(objs.len(), 3);
        let names: Vec<_> = objs
            .iter()
            .map(|o| o.fields.get("name").cloned().unwrap())
            .collect();
        assert!(names.contains(&Value::String("Alice".into())));
        assert!(names.contains(&Value::String("Bob".into())));
        assert!(names.contains(&Value::String("Carol".into())));

        // Round-trip: confirm we can read each back.
        for o in &objs {
            let q = parse_query(&format!("User.get({})", o.id)).unwrap();
            let result = execute(&ExecContext::new(&db, None), &q).unwrap();
            match result {
                QueryOutput::Objects(o2) => assert_eq!(o2.len(), 1),
                _ => panic!("expected Objects"),
            }
        }
    }

    #[test]
    fn execute_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        create_user(&db, "Alice", 25);
        create_user(&db, "Bob", 35);
        create_user(&db, "Carol", 20);

        let q = parse_query("User.filter(.age > 22)").unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 2);
                let names: Vec<_> = objs
                    .iter()
                    .map(|o| o.fields.get("name").unwrap().clone())
                    .collect();
                assert!(names.contains(&Value::String("Alice".into())));
                assert!(names.contains(&Value::String("Bob".into())));
            }
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn execute_update() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let alice = create_user(&db, "Alice", 25);

        let q = parse_query(&format!(
            r#"User.get({}).update({{ age: 26 }})"#,
            alice.id
        ))
        .unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();

        match result {
            QueryOutput::Single(obj) => {
                assert_eq!(obj.fields.get("age"), Some(&Value::U32(26)));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn execute_delete() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let alice = create_user(&db, "Alice", 25);

        let q = parse_query(&format!("User.get({}).delete()", alice.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        assert!(matches!(result, QueryOutput::Done));

        assert!(db.get("User", alice.id).is_err());
    }

    #[test]
    fn execute_traverse() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let alice = create_user(&db, "Alice", 25);
        let bob = create_user(&db, "Bob", 30);
        db.link("User", alice.id, "friends", bob.id, None)
            .unwrap();

        let q = parse_query(&format!("User.get({}).friends", alice.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();

        match result {
            QueryOutput::Objects(mut objs) => {
                assert_eq!(objs.len(), 1);
                // Terminal materialize uses `get_many_lazy` — the wire path
                // would emit `raw_fields` directly. Direct in-memory readers
                // call `ensure_fields_deserialized` to populate `fields`.
                objs[0].ensure_fields_deserialized();
                assert_eq!(
                    objs[0].fields.get("name"),
                    Some(&Value::String("Bob".into()))
                );
            }
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn execute_link_via_query() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let alice = create_user(&db, "Alice", 25);
        let bob = create_user(&db, "Bob", 30);

        // Link via the query language — resolver finds the friends field automatically.
        let q = parse_query(&format!(
            "User.get({}).link(User.get({}))",
            alice.id, bob.id
        ))
        .unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        assert!(matches!(result, QueryOutput::Done));

        // Verify the link landed via direct db query.
        let links = db.get_links("User", alice.id, "friends").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0, bob.id);
    }

    fn edge_typed_db(dir: &std::path::Path) -> std::sync::Arc<Database> {
        let schema = parse_schema(
            r#"
            type Tag { label: String }
            type Item {
                name: String
                tags: [Tag] {
                    score: f64
                    count: u64
                    rank: i32
                    big: i64
                    weight: f32
                } @on_delete(remove)
            }
            "#,
        )
        .unwrap();
        Database::open(schema, dir).unwrap()
    }

    fn make_item_and_tag(db: &Database) -> (u64, u64) {
        let mut itf = FieldMap::new();
        itf.insert("name".into(), Value::String("widget".into()));
        let item = db.create("Item", itf).unwrap();
        let mut tgf = FieldMap::new();
        tgf.insert("label".into(), Value::String("red".into()));
        let tag = db.create("Tag", tgf).unwrap();
        (item.id, tag.id)
    }

    #[test]
    fn link_edge_literals_coerce_to_declared_types() {
        let dir = tempfile::tempdir().unwrap();
        let db = edge_typed_db(dir.path());
        let (item, tag) = make_item_and_tag(&db);

        let q = parse_query(&format!(
            "Item.get({item}).link(Tag.get({tag}), \
             {{ score: 3.5, count: 5, rank: -7, big: 9000000000, weight: 1.25 }})"
        ))
        .unwrap();
        let res = execute(&ExecContext::new(&db, None), &q).unwrap();
        assert!(matches!(res, QueryOutput::Done));

        let links = db.get_links("Item", item, "tags").unwrap();
        assert_eq!(links.len(), 1);
        let e = &links[0].1;
        // Each edge literal must land in its DECLARED Value variant — the
        // best-effort default would store F32/U32/I64 here (the bug).
        assert_eq!(e.get("score"), Some(&Value::F64(3.5)), "f64 -> F64, not F32");
        assert_eq!(e.get("count"), Some(&Value::U64(5)), "u64 -> U64, not U32");
        assert_eq!(e.get("rank"), Some(&Value::I32(-7)), "i32 -> I32, not I64");
        assert_eq!(e.get("big"), Some(&Value::I64(9_000_000_000)), "i64 -> I64");
        assert_eq!(e.get("weight"), Some(&Value::F32(1.25)), "f32 -> F32");
    }

    #[test]
    fn link_edge_int_literal_widens_into_float_field() {
        let dir = tempfile::tempdir().unwrap();
        let db = edge_typed_db(dir.path());
        let (item, tag) = make_item_and_tag(&db);

        // `score: 5` (an Int literal) must widen into the declared f64 field.
        let q = parse_query(&format!(
            "Item.get({item}).link(Tag.get({tag}), {{ score: 5 }})"
        ))
        .unwrap();
        execute(&ExecContext::new(&db, None), &q).unwrap();

        let links = db.get_links("Item", item, "tags").unwrap();
        assert_eq!(links[0].1.get("score"), Some(&Value::F64(5.0)));
    }

    #[test]
    fn link_edge_null_literal_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let db = edge_typed_db(dir.path());
        let (item, tag) = make_item_and_tag(&db);

        // `null` is valid for any edge-field type (matches validate_edge_value).
        let q = parse_query(&format!(
            "Item.get({item}).link(Tag.get({tag}), {{ score: null }})"
        ))
        .unwrap();
        execute(&ExecContext::new(&db, None), &q).unwrap();
        assert_eq!(db.get_links("Item", item, "tags").unwrap()[0].1.get("score"), Some(&Value::Null));
    }

    #[test]
    fn link_edge_unknown_field_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let db = edge_typed_db(dir.path());
        let (item, tag) = make_item_and_tag(&db);

        let q = parse_query(&format!(
            "Item.get({item}).link(Tag.get({tag}), {{ nope: 1 }})"
        ))
        .unwrap();
        let res = execute(&ExecContext::new(&db, None), &q);
        assert!(res.is_err(), "an undeclared edge field must be rejected");
    }

    #[test]
    fn execute_limit() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }

        let q = parse_query("User.filter(.age >= 0).limit(2)").unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 2);
            }
            _ => panic!("expected Objects"),
        }
    }

    // --- Query governor: end-to-end enforcement through real queries. ---

    /// A context with the governor armed from `limits` (deadline relative to now).
    fn gov_ctx<'a>(
        db: &'a Database,
        limits: crate::governor::GovernorLimits,
    ) -> ExecContext<'a> {
        let mut ctx = ExecContext::new(db, None);
        ctx.governor = Governor::new(limits, std::time::Instant::now());
        ctx
    }

    #[test]
    fn governor_clamps_limit() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }
        let limits = crate::governor::GovernorLimits {
            max_limit: 2,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        let q = parse_query("User.limit(100)").unwrap();
        match execute(&gov_ctx(&db, limits), &q).unwrap() {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 2, ".limit(100) clamped to max_limit=2")
            }
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn governor_forbids_unindexed_scan_over_cap() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }
        let tight = crate::governor::GovernorLimits {
            max_rows_scanned: 3,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        // A type of 5 rows exceeds the 3-row budget -> fail closed BEFORE
        // materializing (the count-gate refuses it), never silently truncating.
        let err = execute(&gov_ctx(&db, tight), &parse_query("User").unwrap()).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "got {err:?}");
        // A leading `.limit(2)` does NOT rescue it — a `.limit` on an unindexed
        // listing is not pushed into the scan (the limited scan drops tombstones and
        // would return incomplete results), so it is refused too. Correctness over a
        // cheap-but-wrong pushdown.
        let err = execute(&gov_ctx(&db, tight), &parse_query("User.limit(2)").unwrap()).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "got {err:?}");
        // A type WITHIN the budget scans fully and correctly.
        let roomy = crate::governor::GovernorLimits {
            max_rows_scanned: 10,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        match execute(&gov_ctx(&db, roomy), &parse_query("User").unwrap()).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 5),
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn governor_scan_gate_never_truncates_across_tombstones() {
        // Regression for the adversarial-review HIGH: the old scan_type_limited
        // pushdown returned FEWER live rows than exist when leading (low-id) rows
        // were tombstoned. The count-then-scan gate must be tombstone-correct.
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let mut ids = Vec::new();
        for i in 0..20 {
            ids.push(create_user(&db, &format!("User{i}"), 20 + i).id);
        }
        // Delete the 5 LOWEST ids (tombstones, pre-compaction) — the exact shape
        // that made the limited scan under-return.
        for id in ids.iter().take(5) {
            db.delete("User", *id).unwrap();
        }
        // 15 live, budget 100 -> full correct scan returns ALL 15 (not 15 minus the
        // tombstones in the low slots).
        let g = crate::governor::GovernorLimits {
            max_rows_scanned: 100,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        match execute(&gov_ctx(&db, g), &parse_query("User").unwrap()).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 15, "no silent truncation"),
            _ => panic!("expected Objects"),
        }
        // And it matches the governor-disabled (embedded) result exactly.
        match execute(&ExecContext::new(&db, None), &parse_query("User").unwrap()).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 15),
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn governor_caps_traversal_depth() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let u = create_user(&db, "Root", 30);
        let limits = crate::governor::GovernorLimits {
            max_depth: 1,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        // Two hops exceeds the 1-hop budget (fires before the 2nd hop runs, so it
        // needs no graph data).
        let two_hop = format!("User.get({}).friends.friends", u.id);
        let err = execute(&gov_ctx(&db, limits), &parse_query(&two_hop).unwrap()).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "got {err:?}");
        // One hop is within budget.
        let one_hop = format!("User.get({}).friends", u.id);
        assert!(execute(&gov_ctx(&db, limits), &parse_query(&one_hop).unwrap()).is_ok());
    }

    #[test]
    fn governor_caps_result_rows() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }
        // Row budget off, result ceiling = 2. A 5-row listing exceeds it.
        let limits = crate::governor::GovernorLimits {
            max_result_rows: 2,
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        let err = execute(&gov_ctx(&db, limits), &parse_query("User").unwrap()).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "got {err:?}");
        assert!(execute(&gov_ctx(&db, limits), &parse_query("User.limit(2)").unwrap()).is_ok());
    }

    #[test]
    fn governor_enforces_wall_clock_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        for i in 0..3 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }
        // Arm the deadline in the past so any governor checkpoint (here: the scan's
        // row charge) trips it deterministically.
        let limits = crate::governor::GovernorLimits {
            max_duration: std::time::Duration::from_millis(1),
            ..crate::governor::GovernorLimits::UNLIMITED
        };
        let mut ctx = ExecContext::new(&db, None);
        ctx.governor =
            Governor::new(limits, std::time::Instant::now() - std::time::Duration::from_secs(1));
        let err = execute(&ctx, &parse_query("User").unwrap()).unwrap_err();
        assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "got {err:?}");
        assert!(format!("{err}").contains("time budget"));
    }

    #[test]
    fn governor_disabled_by_default_is_unbounded() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }
        // ExecContext::new => Governor::disabled(): a full listing returns all rows,
        // exactly as before the governor existed (embedded/library path unchanged).
        match execute(&ExecContext::new(&db, None), &parse_query("User").unwrap()).unwrap() {
            QueryOutput::Objects(objs) => assert_eq!(objs.len(), 5),
            _ => panic!("expected Objects"),
        }
    }

    #[test]
    fn fusion_returns_fresh_target_after_second_degree_update() {
        // End-to-end: bench-shape graph (User ↔ Rating ↔ Movie) and a 2-hop
        // covered fusion. The rev_edge on Movie's side embeds User's
        // serialized fields as `user__cover` plus a `user__cover_v` stamp.
        // Updating User.name afterwards bumps the per-object generation
        // counter; the next time we run the 2-hop fusion query, the
        // executor's `<field>__cover_v` vs `db.object_version(...)` check
        // detects mismatch and falls through to a fresh LSM probe via
        // `get_many_lazy`. The returned User must reflect the new name.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }

            type Movie {
                title: String
                ratings: [Rating] @inverse(Rating.movie)
            }

            type Rating {
                stars: u32
                user: User
                movie: Movie
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();

        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        let rating = db.create("Rating", rf).unwrap();

        db.link("Rating", rating.id, "user", alice.id, None).unwrap();
        db.link("Rating", rating.id, "movie", movie.id, None).unwrap();

        // Baseline: fusion should return Alice as-is via the fast covered
        // path (no LSM probe).
        let q = parse_query(&format!(
            "Movie.get({}).ratings.user",
            movie.id
        ))
        .unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert_eq!(objs.len(), 1);
        let mut returned = objs.into_iter().next().unwrap();
        returned.ensure_fields_deserialized();
        assert_eq!(
            returned.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );

        // Mutate Alice. The rev_edge on Movie's side still carries her
        // old serialized blob — only the per-object generation moves.
        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", alice.id, upd).unwrap();

        // Re-run the same fusion. Cover_v < live counter, so the executor
        // routes Alice through `get_many_lazy` instead of using the stale
        // cover — the returned name must be the post-update value.
        let q = parse_query(&format!(
            "Movie.get({}).ratings.user",
            movie.id
        ))
        .unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert_eq!(objs.len(), 1);
        let mut returned = objs.into_iter().next().unwrap();
        returned.ensure_fields_deserialized();
        assert_eq!(
            returned.fields.get("name"),
            Some(&Value::String("Renamed".into())),
            "fusion must fall through to a fresh probe when the embedded \
             cover_v is stale relative to the live generation counter"
        );
    }

    #[test]
    fn fusion_drops_target_deleted_without_prior_update() {
        // Phantom-read regression: a target that is CREATED-but-NEVER-UPDATED
        // gets `<field>__cover_v = object_version(target) = 0`. If that target
        // is then DELETED, `object_version` still returns 0 (no counter entry),
        // so the executor's `embedded_v == live_v` staleness check sees `0 == 0`
        // and treats the stale cover as fresh — serving the deleted object
        // straight from the embedded blob, with no LSM probe to catch the
        // deletion. The deleted target must NOT appear in results.
        //
        // `Rating.user @on_delete(remove)` lets us delete the User while the
        // Movie-side rev_edge that embeds `user__cover` survives — that
        // surviving cover is exactly what could phantom the deleted user.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }

            type Movie {
                title: String
                ratings: [Rating] @inverse(Rating.movie)
            }

            type Rating {
                stars: u32
                user: User @on_delete(remove)
                movie: Movie
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();

        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        let rating = db.create("Rating", rf).unwrap();

        db.link("Rating", rating.id, "user", alice.id, None).unwrap();
        db.link("Rating", rating.id, "movie", movie.id, None).unwrap();

        // Born-at-1: a live, never-updated object reads generation >= 1, so
        // generation 0 is reserved for "absent" (never-created or deleted).
        assert_eq!(db.object_version("User", alice.id), 1);

        // Baseline fusion returns Alice from the fresh cover.
        let q = parse_query(&format!("Movie.get({}).ratings.user", movie.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        assert!(
            matches!(&result, QueryOutput::Objects(o) if o.len() == 1),
            "baseline: expected 1 user, got {result:?}"
        );

        // Delete Alice (never updated). Rating.user is @on_delete(remove), so
        // the Rating survives and the Movie-side rev_edge keeps user__cover.
        db.delete("User", alice.id).unwrap();
        assert!(
            db.get("User", alice.id).is_err(),
            "Alice must be gone from the LSM after delete"
        );

        // Re-run the fusion. A deleted object must never surface.
        let q = parse_query(&format!("Movie.get({}).ratings.user", movie.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert!(
            objs.is_empty(),
            "deleted user surfaced as a phantom via a stale cover (cover_v 0 == live 0): {:?}",
            objs.into_iter()
                .map(|mut o| {
                    o.ensure_fields_deserialized();
                    o.fields.get("name").cloned()
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fusion_phantom_stays_closed_across_restart() {
        // The born bit is no longer a persisted `g:` key — it is reconstructed
        // at open() from the `o:*` object scan. This proves dropping that key
        // does not reintroduce the never-updated-then-deleted phantom across a
        // restart: a deleted target's `o:` key is gone, so the reopen scan
        // never re-seeds its generation, `object_version` reads 0, and the
        // surviving Movie-side `user__cover` is rejected.
        let dir = tempfile::tempdir().unwrap();
        let schema_text = r#"
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }
            type Movie {
                title: String
                ratings: [Rating] @inverse(Rating.movie)
            }
            type Rating {
                stars: u32
                user: User @on_delete(remove)
                movie: Movie
            }
        "#;

        let (alice_id, movie_id) = {
            let db = Database::open(parse_schema(schema_text).unwrap(), dir.path()).unwrap();
            let mut uf = FieldMap::new();
            uf.insert("name".into(), Value::String("Alice".into()));
            let alice = db.create("User", uf).unwrap();
            let mut mf = FieldMap::new();
            mf.insert("title".into(), Value::String("Aliens".into()));
            let movie = db.create("Movie", mf).unwrap();
            let mut rf = FieldMap::new();
            rf.insert("stars".into(), Value::U32(5));
            let rating = db.create("Rating", rf).unwrap();
            db.link("Rating", rating.id, "user", alice.id, None).unwrap();
            db.link("Rating", rating.id, "movie", movie.id, None).unwrap();
            // Delete Alice (never updated) while the Movie-side cover survives.
            db.delete("User", alice.id).unwrap();
            (alice.id, movie.id)
        };

        // Reopen: version_counters is rebuilt from disk (o:* seed + g:* override).
        let db = Database::open(parse_schema(schema_text).unwrap(), dir.path()).unwrap();
        assert_eq!(
            db.object_version("User", alice_id),
            0,
            "deleted target must read generation 0 after restart (not re-seeded)"
        );

        let q = parse_query(&format!("Movie.get({movie_id}).ratings.user")).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert!(
            objs.is_empty(),
            "deleted user phantomed via a stale cover after restart: {:?}",
            objs.into_iter()
                .map(|mut o| {
                    o.ensure_fields_deserialized();
                    o.fields.get("name").cloned()
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fusion_drops_nested_3hop_target_deleted_without_prior_update() {
        // Same phantom, one level deeper: the deleted target sits in a NESTED
        // 3-hop cover (`director__cover` embedded inside `movie__cover`). The
        // nested read reuses the same guarded fusion loop, so a never-updated
        // Director that is deleted must not surface through the nested cover.
        // `Movie.director @on_delete(remove)` lets us delete the Director while
        // the Movie (and the rev_edge carrying the nested cover) survives.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Director {
                name: String
            }
            type Movie {
                title: String
                director: Director @on_delete(remove)
                ratings: [Rating] @inverse(Rating.movie)
            }
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }
            type Rating {
                stars: u32
                user: User
                movie: Movie
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        mf.insert("director".into(), Value::U64(director.id));
        let movie = db.create("Movie", mf).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        // Baseline: the 3-hop fusion resolves the director.
        let q = parse_query(&format!("User.get({}).ratings.movie.director", user.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        assert!(
            matches!(&result, QueryOutput::Objects(o) if o.len() == 1),
            "baseline: expected 1 director, got {result:?}"
        );

        // Delete the never-updated Director; Movie survives (remove policy),
        // so the nested director__cover (cover_v == 1) lingers in the rev_edge.
        db.delete("Director", director.id).unwrap();

        let q = parse_query(&format!("User.get({}).ratings.movie.director", user.id)).unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        let objs = match result {
            QueryOutput::Objects(objs) => objs,
            other => panic!("expected Objects, got {other:?}"),
        };
        assert!(
            objs.is_empty(),
            "deleted director surfaced as a phantom via a stale NESTED 3-hop cover: {:?}",
            objs.into_iter()
                .map(|mut o| {
                    o.ensure_fields_deserialized();
                    o.fields.get("name").cloned()
                })
                .collect::<Vec<_>>()
        );
    }

    fn three_hop_db(dir: &std::path::Path) -> std::sync::Arc<Database> {
        let schema = parse_schema(
            r#"
            type Director {
                name: String
            }
            type Movie {
                title: String
                director: Director
                ratings: [Rating] @inverse(Rating.movie)
            }
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }
            type Rating {
                stars: u32
                user: User
                movie: Movie
            }
            "#,
        )
        .unwrap();
        Database::open(schema, dir).unwrap()
    }

    /// End-to-end 3-hop covering: `User.get(X).ratings.movie.director`.
    ///
    /// Pre-3-hop: hop 4 (`.director`) would fall through to a per-movie
    /// LSM probe because the executor's fast-path only handles 2-hop. With
    /// 3-hop covers embedded in rev_edge bytes, the second forward-1:1
    /// hop also routes through the fusion path — the Object's raw_fields
    /// carry the movie's serialized blob WITH `director` + `director__cover`
    /// inline, so the executor extracts the director directly.
    #[test]
    fn execute_three_hop_traverse_via_cover_fusion() {
        let dir = tempfile::tempdir().unwrap();
        let db = three_hop_db(dir.path());

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        mf.insert("director".into(), Value::U64(director.id));
        let movie = db.create("Movie", mf).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();

        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        let q = parse_query(&format!(
            "User.get({}).ratings.movie.director",
            user.id
        ))
        .unwrap();
        let result =
            execute(&ExecContext::new(&db, None), &q).unwrap();

        match result {
            QueryOutput::Objects(mut objs) => {
                assert_eq!(objs.len(), 1, "should resolve to one director");
                objs[0].ensure_fields_deserialized();
                assert_eq!(objs[0].type_name, "Director");
                assert_eq!(
                    objs[0].fields.get("name"),
                    Some(&Value::String("Scott".into())),
                    "director name should round-trip through 3-hop cover"
                );
            }
            other => panic!("expected Objects, got {other:?}"),
        }
    }

    #[test]
    fn execute_three_hop_cover_falls_through_on_stale_director() {
        // Director's cover_v stamp is checked at extract time. After we
        // update the director (bumping its generation), the embedded
        // `director__cover_v` in the movie cover is stale; the executor
        // routes through `get_many_lazy` to fetch the fresh director and
        // still returns correct data. (Sweeper is on by default; we
        // observe the slow-path fallback via a query that returns the
        // post-update state.)
        let dir = tempfile::tempdir().unwrap();
        let db = three_hop_db(dir.path());

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        mf.insert("director".into(), Value::U64(director.id));
        let movie = db.create("Movie", mf).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        // Bump the director — embedded cover is now stale until the
        // sweeper repairs OR a reader detects the cover_v mismatch.
        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("Director", director.id, upd).unwrap();

        let q = parse_query(&format!(
            "User.get({}).ratings.movie.director",
            user.id
        ))
        .unwrap();
        let result =
            execute(&ExecContext::new(&db, None), &q).unwrap();

        match result {
            QueryOutput::Objects(mut objs) => {
                assert_eq!(objs.len(), 1);
                objs[0].ensure_fields_deserialized();
                // The post-update name must be reflected — either via
                // sweeper repair (fast path) or cover_v fall-through
                // (slow path). Both are correct outcomes.
                assert_eq!(
                    objs[0].fields.get("name"),
                    Some(&Value::String("Renamed".into())),
                    "stale director cover must be detected + replaced"
                );
            }
            other => panic!("expected Objects, got {other:?}"),
        }
    }

    #[test]
    fn execute_boolean_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        create_user(&db, "Alice", 25);
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Inactive".into()));
        f.insert("age".into(), Value::U32(40));
        f.insert("active".into(), Value::Bool(false));
        db.create("User", f).unwrap();

        let q = parse_query("User.filter(.active == true)").unwrap();
        let result = execute(&ExecContext::new(&db, None), &q).unwrap();
        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 1);
                assert_eq!(
                    objs[0].fields.get("name"),
                    Some(&Value::String("Alice".into()))
                );
            }
            _ => panic!("expected Objects"),
        }
    }

    // ===== Query plan picker (AND-conjunct index pushdown) =====
    //
    // The planner must be behavior-preserving: every query returns exactly the
    // rows (and, for the AND path, the object-id order) that
    // `scan_all_objects + evaluate_predicate` would. These tests compare the
    // live `execute` path against an independent full-scan oracle across many
    // predicate shapes — including the counterexamples the design review raised:
    // value-order vs id-order under `.limit`, non-selective indexed conjuncts
    // past the probe cap, absent/null fields, and `Ne` residuals.

    fn planner_db(dir: &std::path::Path) -> std::sync::Arc<Database> {
        let schema = parse_schema(
            r#"
            type Person {
                name: String
                slug: String @unique
                age: u32 @indexed
                score: f64 @indexed
                active: Bool @indexed
                country: String @indexed
                nick: String
                rank: i64
            }
            "#,
        )
        .unwrap();
        Database::open(schema, dir).unwrap()
    }

    /// Insert a Person. `country`/`nick` = `None` omits the field entirely
    /// (absent), exercising the null/absent comparison semantics. Returns its id.
    #[allow(clippy::too_many_arguments)]
    fn person(
        db: &Database,
        name: &str,
        age: u32,
        score: f64,
        active: bool,
        country: Option<&str>,
        nick: Option<&str>,
        rank: i64,
    ) -> u64 {
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String(name.into()));
        // slug = name (the names in each test dataset are unique).
        f.insert("slug".into(), Value::String(name.into()));
        f.insert("age".into(), Value::U32(age));
        f.insert("score".into(), Value::F64(score));
        f.insert("active".into(), Value::Bool(active));
        if let Some(c) = country {
            f.insert("country".into(), Value::String(c.into()));
        }
        if let Some(n) = nick {
            f.insert("nick".into(), Value::String(n.into()));
        }
        f.insert("rank".into(), Value::I64(rank));
        db.create("Person", f).unwrap().id
    }

    fn seed_people(db: &Database) {
        // Insertion (= id) order deliberately DIFFERS from age order, so an
        // index (value-ordered) scan that forgot to restore id order would be
        // caught by the `.limit` parity test below.
        // name, age, score, active, country,    nick,     rank
        person(db, "Ann", 50, 1.5, true, Some("US"), Some("a"), 10);
        person(db, "Bob", 20, 2.5, true, Some("US"), None, 20);
        person(db, "Cy", 60, 3.5, true, Some("CA"), Some("c"), 30);
        person(db, "Dee", 30, 4.5, true, Some("US"), Some("d"), 40);
        person(db, "Eve", 70, 5.5, false, Some("CA"), None, 50);
        person(db, "Fay", 30, 2.5, true, None, Some("f"), 60); // country absent
        person(db, "Gus", 40, 9.9, false, Some("UK"), Some("g"), 70);
        person(db, "Hal", 20, 1.5, true, Some("US"), Some("h"), 80);
    }

    /// Full-scan oracle: ids (ascending) where `evaluate_predicate` holds —
    /// exactly the baseline the planner must reproduce.
    fn oracle_ids(db: &Database, predicate: &Predicate) -> Vec<u64> {
        // The oracle wants the full unbounded scan (the governor-disabled shape).
        let mut all = db.scan_type("Person").unwrap();
        for o in &mut all {
            o.ensure_fields_deserialized();
        }
        let mut ids: Vec<u64> = all
            .into_iter()
            .filter(|o| evaluate_predicate(predicate, &o.fields))
            .map(|o| o.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn predicate_of(filter_query: &str) -> Predicate {
        match parse_query(filter_query).unwrap().source {
            Source::Filter { predicate, .. } => predicate,
            other => panic!("expected a filter source, got {other:?}"),
        }
    }

    fn exec_ids(ctx: &ExecContext<'_>, query: &str) -> Vec<u64> {
        match execute(ctx, &parse_query(query).unwrap()).unwrap() {
            QueryOutput::Objects(objs) => objs.into_iter().map(|o| o.id).collect(),
            other => panic!("expected Objects, got {other:?}"),
        }
    }

    #[test]
    fn planner_result_set_parity_across_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db);
        let ctx = ExecContext::new(&db, None);

        let queries = [
            r#"Person.filter(.age > 25)"#,
            r#"Person.filter(.country == "US")"#,
            r#"Person.filter(.age > 25 && .country == "US")"#,
            r#"Person.filter(.country == "US" && .age > 25)"#,
            r#"Person.filter(.active == true && .age >= 30)"#,
            r#"Person.filter(.age >= 20 && .age <= 40)"#,
            r#"Person.filter(.country == "US" && .nick == "h")"#, // 2nd term non-indexed
            r#"Person.filter(.age > 25 && .rank < 45)"#,          // 2nd term non-indexed
            r#"Person.filter(.age > 100 && .country == "US")"#,   // empty result
            r#"Person.filter(.age != 20 && .country == "US")"#,   // Ne residual
            r#"Person.filter(.score > 2.0 && .active == true)"#,
            r#"Person.filter(.country == "US" || .country == "CA")"#, // OR → full scan
            r#"Person.filter(.age > 25 && (.country == "US" || .country == "CA"))"#, // nested OR
            r#"Person.filter(.age > 25 && .country == "US" && .active == true)"#, // 3-way AND
            r#"Person.filter(.nick == "a" && .rank > 5)"#, // no indexed term → full scan
        ];
        for q in queries {
            let want = oracle_ids(&db, &predicate_of(q));
            let mut got = exec_ids(&ctx, q);
            got.sort_unstable();
            assert_eq!(got, want, "set parity failed for `{q}`");
        }
    }

    #[test]
    fn planner_and_path_preserves_id_order_and_limit() {
        // Design-review counterexample: the AND path pushes a RANGE conjunct on
        // an @indexed field, whose index scan yields rows in value (age) order,
        // NOT id order. The planner must restore id order so `.limit(N)` returns
        // the same rows the full-scan baseline would.
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db);
        let ctx = ExecContext::new(&db, None);

        // `.rank < 1000` is true for everyone and is NON-indexed, so the only
        // generator is the @indexed RANGE `.age > 18` — forcing value-order.
        let base = r#"Person.filter(.age > 18 && .rank < 1000)"#;
        let pred = predicate_of(base);
        assert!(
            plan_filter_scan(&db, "Person", &pred, None)
                .unwrap()
                .is_some(),
            "planner should engage by pushing the @indexed range conjunct"
        );

        let want = oracle_ids(&db, &pred); // id-ascending
        assert_eq!(exec_ids(&ctx, base), want, "unbounded AND must be id-order");
        for k in 0..=want.len() + 1 {
            let got = exec_ids(&ctx, &format!("{base}.limit({k})"));
            let expect: Vec<u64> = want.iter().copied().take(k).collect();
            assert_eq!(got, expect, "limit({k}) parity failed");
        }
    }

    #[test]
    fn planner_null_and_ne_residual_parity() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db); // Fay has country absent
        let ctx = ExecContext::new(&db, None);

        for q in [
            r#"Person.filter(.age > 25 && .country != "US")"#, // absent country excluded by Ne
            r#"Person.filter(.country != "CA" && .age > 25)"#,
            r#"Person.filter(.age >= 30 && .country == "US")"#,
        ] {
            let want = oracle_ids(&db, &predicate_of(q));
            let mut got = exec_ids(&ctx, q);
            got.sort_unstable();
            assert_eq!(got, want, "null/Ne parity failed for `{q}`");
        }
    }

    #[test]
    fn planner_engages_or_falls_back_appropriately() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db);
        let engages = |q: &str| {
            plan_filter_scan(&db, "Person", &predicate_of(q), None)
                .unwrap()
                .is_some()
        };

        // Engages: AND with a selective @indexed conjunct (Eq or range).
        assert!(engages(r#"Person.filter(.country == "US" && .rank < 50)"#));
        assert!(engages(r#"Person.filter(.age > 18 && .nick == "a")"#));
        // Single Compare → existing fast path (delegated), still Some.
        assert!(engages(r#"Person.filter(.age > 18)"#));
        // v2 OR-union: engages when EVERY disjunct is index-eligible.
        assert!(engages(r#"Person.filter(.country == "US" || .age > 10)"#));
        // Falls back: an OR with a non-indexed disjunct (a row matching only it
        // would be missed by a union of index scans).
        assert!(!engages(r#"Person.filter(.country == "US" || .nick == "x")"#));
        // Falls back: no @indexed conjunct at all.
        assert!(!engages(r#"Person.filter(.nick == "a" && .rank > 5)"#));
        // Falls back: the only @indexed conjunct is `Ne` (excluded as generator).
        assert!(!engages(r#"Person.filter(.age != 20 && .nick == "x")"#));

        // Selectivity ranking: string Eq (0) < range (1) < bool Eq (2).
        let rank = |q: &str| {
            conjunct_index_generator(&db, "Person", &predicate_of(q))
                .unwrap()
                .0
        };
        let country_eq = rank(r#"Person.filter(.country == "US")"#);
        let age_range = rank(r#"Person.filter(.age > 18)"#);
        let active_eq = rank(r#"Person.filter(.active == true)"#);
        assert!(
            country_eq < age_range && age_range < active_eq,
            "expected string Eq < range < bool Eq, got {country_eq}/{age_range}/{active_eq}"
        );
    }

    #[test]
    fn planner_correct_across_index_tombstone_run() {
        // Regression for the bounded-probe blocker: a *bounded* index scan counts
        // tombstones against its row budget, so a long tombstone run at an indexed
        // value could make the AND-path probe under-return live rows and silently
        // drop results. The planner now scans the generator UNBOUNDED, which is
        // tombstone-sound. Build a tombstone run larger than any plausible probe
        // cap, then query a range that spans it plus a non-indexed residual.
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());

        // Insert many age==1 rows, then delete them all → a long run of index
        // tombstones at `i:Person:age:<1>:` (which sorts first in the keyspace).
        let bulk = 9000usize;
        let low: Vec<FieldMap> = (0..bulk)
            .map(|_| {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("low".into()));
                f.insert("age".into(), Value::U32(1));
                f.insert("score".into(), Value::F64(1.0));
                f.insert("active".into(), Value::Bool(true));
                f.insert("rank".into(), Value::I64(0));
                f
            })
            .collect();
        for o in db.create_batch("Person", low).unwrap() {
            db.delete("Person", o.id).unwrap();
        }
        // Live rows at age==1000, sorting AFTER the age==1 tombstone run.
        let n_live = 200usize;
        let high: Vec<FieldMap> = (0..n_live)
            .map(|i| {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("high".into()));
                f.insert("age".into(), Value::U32(1000));
                f.insert("score".into(), Value::F64(1.0));
                f.insert("active".into(), Value::Bool(true));
                f.insert("rank".into(), Value::I64(i as i64));
                f
            })
            .collect();
        db.create_batch("Person", high).unwrap();
        let ctx = ExecContext::new(&db, None);

        // Generator is the @indexed range `age >= 1`; its scan must not be
        // truncated by the age==1 tombstone run. `rank >= 0` is a non-indexed
        // residual so the AND path (not the single-Compare path) is exercised.
        let q = r#"Person.filter(.age >= 1 && .rank >= 0)"#;
        let pred = predicate_of(q);
        assert!(
            plan_filter_scan(&db, "Person", &pred, None)
                .unwrap()
                .is_some(),
            "planner should engage by pushing the @indexed range"
        );
        let want = oracle_ids(&db, &pred);
        assert_eq!(want.len(), n_live, "only the live age==1000 rows match");
        let mut got = exec_ids(&ctx, q);
        got.sort_unstable();
        assert_eq!(got, want, "tombstone-run parity failed (under-returned)");
    }

    #[test]
    fn planner_float_eq_residual_matches_negative_zero() {
        // `compare_values` treats -0.0 == 0.0, but the float index key for -0.0
        // differs from +0.0 — so float Eq must NOT be an index generator (it would
        // miss a stored -0.0). The planner pushes the @indexed range `age > 10`
        // instead and catches the -0.0 row in the in-memory residual filter.
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        let id_neg = person(&db, "Neg", 20, -0.0, true, Some("US"), None, 1);
        person(&db, "Pos", 20, 1.0, true, Some("US"), None, 2);
        let ctx = ExecContext::new(&db, None);

        // Float Eq must not qualify as an index generator (would miss -0.0).
        let score_eq = predicate_of(r#"Person.filter(.score == 0.0)"#);
        assert!(
            conjunct_index_generator(&db, "Person", &score_eq).is_none(),
            "float Eq must not be an index generator"
        );

        let q = r#"Person.filter(.score == 0.0 && .age > 10)"#;
        let pred = predicate_of(q);
        let want = oracle_ids(&db, &pred);
        assert_eq!(want, vec![id_neg], "only the -0.0 row matches score == 0.0");
        let mut got = exec_ids(&ctx, q);
        got.sort_unstable();
        assert_eq!(got, want, "float Eq / -0.0 parity failed");
    }

    // ===== v2: unique-Eq probe + multi-index intersection + OR-union =====

    #[test]
    fn planner_unique_eq_probe_parity_and_engages() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db); // slug == name, @unique
        let ctx = ExecContext::new(&db, None);

        // A @unique equality engages the unique probe and returns the one row.
        let q = r#"Person.filter(.slug == "Cy")"#;
        assert!(
            plan_filter_scan(&db, "Person", &predicate_of(q), None)
                .unwrap()
                .is_some(),
            "unique-eq should engage"
        );
        let want = oracle_ids(&db, &predicate_of(q));
        assert_eq!(want.len(), 1);
        let mut got = exec_ids(&ctx, q);
        got.sort_unstable();
        assert_eq!(got, want, "unique-eq parity failed");

        // A unique-equality CONJUNCT in an AND yields ≤1 candidate, re-filtered.
        for q in [
            r#"Person.filter(.slug == "Dee" && .age > 25)"#, // Dee age 30 → matches
            r#"Person.filter(.slug == "Bob" && .age > 25)"#, // Bob age 20 → residual drops it
            r#"Person.filter(.slug == "nope" && .age > 0)"#, // no such slug → empty
        ] {
            let want = oracle_ids(&db, &predicate_of(q));
            let mut got = exec_ids(&ctx, q);
            got.sort_unstable();
            assert_eq!(got, want, "unique-conjunct parity failed for `{q}`");
        }
    }

    #[test]
    fn planner_unique_probe_drops_stale_entry_after_update_to_null() {
        // A @unique field updated to Null leaves a stale `u:` entry pointing at
        // the (now null) object. The probe's re-filter must drop it, matching the
        // full-scan oracle (which sees the live null field). Without the re-filter
        // the planner would over-return the stale row.
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        let id = person(&db, "Zed", 33, 1.0, true, Some("US"), None, 1);
        let ctx = ExecContext::new(&db, None);

        let mut upd = FieldMap::new();
        upd.insert("slug".into(), Value::Null);
        db.update("Person", id, upd).unwrap();

        let q = r#"Person.filter(.slug == "Zed")"#;
        assert!(
            oracle_ids(&db, &predicate_of(q)).is_empty(),
            "no live row has slug == Zed"
        );
        assert!(
            exec_ids(&ctx, q).is_empty(),
            "planner must drop the stale u: entry via the re-filter"
        );
    }

    #[test]
    fn planner_unique_probe_excludes_non_eq_and_non_unique() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db);
        let snap = db.read_snapshot();
        // Non-Eq on a @unique field is not a point lookup.
        assert!(
            unique_eq_probe(&db, snap, "Person", &predicate_of(r#"Person.filter(.slug > "a")"#))
                .unwrap()
                .is_none()
        );
        // A non-unique field is not a unique probe.
        assert!(
            unique_eq_probe(
                &db,
                snap,
                "Person",
                &predicate_of(r#"Person.filter(.country == "US")"#)
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn planner_unique_probe_excludes_float_eq_negative_zero() {
        // A float @unique field: -0.0 and +0.0 hash to different `u:` bytes but
        // compare equal, so the probe must be excluded (it would MISS a stored
        // -0.0). The full-scan path stays correct via compare_values.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type M { ratio: f64 @unique }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("ratio".into(), Value::F64(-0.0));
        let id = db.create("M", f).unwrap().id;
        let ctx = ExecContext::new(&db, None);

        let q = r#"M.filter(.ratio == 0.0)"#;
        assert!(
            unique_eq_probe(&db, db.read_snapshot(), "M", &predicate_of(q))
                .unwrap()
                .is_none(),
            "float @unique Eq must be excluded from the unique probe"
        );
        // The full-scan path finds the -0.0 row (compare_values: -0.0 == 0.0).
        assert_eq!(exec_ids(&ctx, q), vec![id]);
    }

    #[test]
    fn planner_intersection_and_union_parity() {
        let dir = tempfile::tempdir().unwrap();
        let db = planner_db(dir.path());
        seed_people(&db);
        let ctx = ExecContext::new(&db, None);

        for q in [
            // AND of two @indexed equality conjuncts → intersection.
            r#"Person.filter(.country == "US" && .age == 20)"#,
            r#"Person.filter(.age == 30 && .country == "US")"#,
            // country Eq is the only rank-0 generator; active Eq (Bool) stays residual.
            r#"Person.filter(.country == "CA" && .active == true)"#,
            // Top-level OR of @indexed disjuncts → union (+ dedup where they overlap).
            r#"Person.filter(.country == "CA" || .age == 70)"#,
            r#"Person.filter(.age == 20 || .country == "CA")"#,
            // OR mixing a @unique disjunct with an @indexed disjunct.
            r#"Person.filter(.slug == "Ann" || .country == "CA")"#,
            // Nested OR inside an AND: the inner OR stays a residual conjunct
            // (NOT routed to the union path).
            r#"Person.filter(.active == true && (.country == "US" || .country == "CA"))"#,
        ] {
            assert!(
                plan_filter_scan(&db, "Person", &predicate_of(q), None)
                    .unwrap()
                    .is_some(),
                "planner should engage for `{q}`"
            );
            let want = oracle_ids(&db, &predicate_of(q));
            let mut got = exec_ids(&ctx, q);
            got.sort_unstable();
            assert_eq!(got, want, "intersection/union parity failed for `{q}`");
        }
    }
}
