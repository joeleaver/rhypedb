use std::collections::{HashMap, HashSet};

use bytes::Bytes;
use rhypedb_engine::database::Database;
use rhypedb_engine::object::{
    find_bytes_field_in_raw, find_u64_field_in_raw, FieldMap, Object, Value,
};
use rhypedb_engine::vectorizer::Vectorizer;

use crate::ast::*;
use crate::error::{QueryError, QueryResult};

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
}

impl<'a> ExecContext<'a> {
    /// A context with no server-wide vector-search defaults: `.similar` queries
    /// fall back to the engine's per-shape `ef`/`rerank` heuristics. Server
    /// callers construct this then set `default_ef`/`default_rerank` from config.
    pub fn new(db: &'a Database, vectorizer: Option<&'a Vectorizer>) -> Self {
        Self {
            db,
            vectorizer,
            default_ef: None,
            default_rerank: None,
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
        _ => (execute_source(ctx.db, &query.source, &query.steps)?, 0),
    };

    for step in &query.steps[first_step..] {
        result = execute_step(ctx, result, step, &query.source)?;
    }

    // Streaming-traversal: if the pipeline ended on an `IdSet` or
    // `IdSetWithFields`, materialize it to `Objects` so callers (server
    // response, tests) see the historical shape. This is the only `get`
    // cost we still pay for those IDs.
    match result {
        QueryOutput::IdSet { type_name, ids } => {
            result = QueryOutput::Objects(materialize_ids(ctx.db, &type_name, &ids));
        }
        QueryOutput::IdSetWithFields { type_name, items } => {
            // Terminal materialize: ignore the carried raw bytes (would only
            // hold edge_fields, not the target's object fields) and probe the
            // LSM for the full Object data.
            let ids: Vec<u64> = items.into_iter().map(|(id, _)| id).collect();
            result = QueryOutput::Objects(materialize_ids(ctx.db, &type_name, &ids));
        }
        _ => {}
    }

    Ok(result)
}

fn execute_source(
    db: &Database,
    source: &Source,
    steps: &[Step],
) -> QueryResult<QueryOutput> {
    match source {
        Source::Get { type_name, id } => {
            let obj = db.get(type_name, *id)?;
            Ok(QueryOutput::Objects(vec![obj]))
        }

        Source::Filter {
            type_name,
            predicate,
        } => {
            // Index / zone-map fast path: single integer comparison gets
            // pushed down to storage along with an effective limit (if the
            // next step is a bare `.limit(N)` we can stop scanning once N
            // matches land). Complex predicates (And/Or, string compares,
            // etc.) fall through to the full scan.
            let pushed_limit = leading_limit(steps);
            if let Some(mut objects) = try_filter_scan(db, type_name, predicate, pushed_limit)? {
                // Fast-path objects may carry `raw_fields` — if a downstream
                // step inspects them via `obj.fields`, eagerly populate now.
                // No-op when raw_fields is None.
                for obj in &mut objects {
                    obj.ensure_fields_deserialized();
                }
                return Ok(QueryOutput::Objects(objects));
            }
            let mut all = scan_all_objects(db, type_name)?;
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
            let obj = db.create(type_name, field_map)?;
            Ok(QueryOutput::Single(obj))
        }

        Source::CreateBatch { type_name, rows } => {
            let field_maps: Vec<FieldMap> = rows
                .iter()
                .map(|r| literal_map_to_field_map(db, type_name, r))
                .collect::<QueryResult<_>>()?;
            let objects = db.create_batch(type_name, field_maps)?;
            Ok(QueryOutput::Objects(objects))
        }

        Source::All { type_name } => {
            let all = scan_all_objects(db, type_name)?;
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
            let field_map = literal_map_to_field_map(db, &type_name, fields)?;
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

        Step::Limit { count } => match current {
            QueryOutput::IdSet { type_name, mut ids } => {
                ids.truncate(*count);
                Ok(QueryOutput::IdSet { type_name, ids })
            }
            QueryOutput::IdSetWithFields { type_name, mut items } => {
                items.truncate(*count);
                Ok(QueryOutput::IdSetWithFields { type_name, items })
            }
            other => {
                let mut objects = extract_objects(other)?;
                objects.truncate(*count);
                Ok(QueryOutput::Objects(objects))
            }
        },

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
        (_, Literal::Null) => match op {
            CompareOp::Eq => matches!(field_val, Value::Null),
            CompareOp::Ne => !matches!(field_val, Value::Null),
            _ => false,
        },
        _ => false,
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
        _ => return Err(mismatch()),
    })
}

fn scan_all_objects(db: &Database, type_name: &str) -> QueryResult<Vec<Object>> {
    Ok(db.scan_type(type_name)?)
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
    // Every scalar-typed literal can route to a typed filter_scan; null
    // falls through. Bytes-indexed predicates don't have a query-language
    // form yet (no Bytes literal at the parser level) — engine API users
    // call `Database::filter_scan_bytes` directly.
    match value {
        Literal::Int(i) => Ok(Some(db.filter_scan(
            type_name, field_path, storage_op, *i, limit,
        )?)),
        Literal::String(s) => Ok(Some(db.filter_scan_str(
            type_name, field_path, storage_op, s, limit,
        )?)),
        Literal::Bool(b) => Ok(Some(db.filter_scan_bool(
            type_name, field_path, storage_op, *b, limit,
        )?)),
        Literal::Float(f) => Ok(Some(db.filter_scan_float(
            type_name, field_path, storage_op, *f, limit,
        )?)),
        Literal::Null => Ok(None),
    }
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
}
