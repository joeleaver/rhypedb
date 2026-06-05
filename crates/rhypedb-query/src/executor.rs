use std::collections::{HashMap, HashSet};

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{FieldMap, Object, Value};
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

    /// Like IdSet, but also carries per-id source fields produced by an
    /// inverse traversal's covering reverse-edge values. Consumed by a
    /// subsequent forward-1:1 traversal to skip the edge scan. Falls back
    /// to plain IdSet semantics for any downstream step that doesn't take
    /// advantage of the fields.
    IdSetWithFields {
        type_name: String,
        items: Vec<(u64, rhypedb_engine::object::FieldMap)>,
    },
}

/// Context for query execution.
pub struct ExecContext<'a> {
    pub db: &'a Database,
    pub vectorizer: Option<&'a Vectorizer>,
}

/// Execute a parsed query against the database.
pub fn execute(ctx: &ExecContext<'_>, query: &Query) -> QueryResult<QueryOutput> {
    let mut result = execute_source(ctx.db, &query.source)?;

    for step in &query.steps {
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
            let ids: Vec<u64> = items.into_iter().map(|(id, _)| id).collect();
            result = QueryOutput::Objects(materialize_ids(ctx.db, &type_name, &ids));
        }
        _ => {}
    }

    Ok(result)
}

fn execute_source(db: &Database, source: &Source) -> QueryResult<QueryOutput> {
    match source {
        Source::Get { type_name, id } => {
            let obj = db.get(type_name, *id)?;
            Ok(QueryOutput::Objects(vec![obj]))
        }

        Source::Filter {
            type_name,
            predicate,
        } => {
            // Zone-map fast path: single integer comparison gets pushed down
            // to storage, which uses per-block min/max bounds to skip whole
            // groups of entries before any decode. Complex predicates
            // (And/Or, string compares, etc.) fall through to the full scan.
            if let Some(objects) = try_filter_scan(db, type_name, predicate)? {
                return Ok(QueryOutput::Objects(objects));
            }
            let all = scan_all_objects(db, type_name)?;
            let filtered = all
                .into_iter()
                .filter(|obj| evaluate_predicate(predicate, &obj.fields))
                .collect();
            Ok(QueryOutput::Objects(filtered))
        }

        Source::Create { type_name, fields } => {
            let field_map = literal_map_to_field_map(fields)?;
            let obj = db.create(type_name, field_map)?;
            Ok(QueryOutput::Single(obj))
        }

        Source::CreateBatch { type_name, rows } => {
            let field_maps: Vec<FieldMap> = rows
                .iter()
                .map(literal_map_to_field_map)
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

            // Fusion fast path: input is an IdSetWithFields (came from an
            // inverse traversal) AND we're now doing a forward-1:1 traversal
            // whose target id was captured in those source FieldMaps. Read it
            // straight out — no edge scan needed for this whole hop.
            if is_one_to_one_forward
                && let QueryOutput::IdSetWithFields { items, .. } = &current
                && items
                    .iter()
                    .any(|(_, f)| matches!(f.get(field_name.as_str()), Some(Value::U64(_))))
            {
                let mut seen: HashSet<u64> = HashSet::with_capacity(items.len());
                let mut out: Vec<u64> = Vec::with_capacity(items.len());
                for (_id, fields) in items {
                    if let Some(Value::U64(tid)) = fields.get(field_name.as_str())
                        && seen.insert(*tid)
                    {
                        out.push(*tid);
                    }
                }
                return Ok(QueryOutput::IdSet {
                    type_name: target_type,
                    ids: out,
                });
            }

            // Streaming path: extract (type, ids), walk links in one batched
            // LSM pass.
            let (source_type, source_ids) = ids_from_output(current, source)?;
            let groups = db.get_links_many(&source_type, &source_ids, field_name)?;

            // Inverse traversals: preserve the per-source FieldMaps that
            // get_links_many returned (these now carry covering source-field
            // data so the next forward-1:1 hop can fuse). Forward traversals
            // just dedup IDs.
            if is_inverse {
                let mut seen: HashSet<u64> = HashSet::with_capacity(source_ids.len());
                let mut items: Vec<(u64, FieldMap)> = Vec::new();
                for group in groups {
                    for (target_id, edge_fields) in group {
                        if seen.insert(target_id) {
                            items.push((target_id, edge_fields));
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
                    for (target_id, _edge_fields) in group {
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
            let objects = match current {
                QueryOutput::IdSet { type_name, ids } => {
                    materialize_ids(db, &type_name, &ids)
                }
                QueryOutput::IdSetWithFields { type_name, items } => {
                    let ids: Vec<u64> = items.into_iter().map(|(id, _)| id).collect();
                    materialize_ids(db, &type_name, &ids)
                }
                other => extract_objects(other)?,
            };
            let filtered = objects
                .into_iter()
                .filter(|obj| evaluate_predicate(predicate, &obj.fields))
                .collect();
            Ok(QueryOutput::Objects(filtered))
        }

        Step::Update { fields } => {
            // Update needs (type, id) only — work directly from IDs.
            let (type_name, ids) = ids_from_output(current, source)?;
            let field_map = literal_map_to_field_map(fields)?;
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
            let edge_map = if edge_fields.is_empty() {
                None
            } else {
                Some(literal_map_to_field_map(edge_fields)?)
            };
            // Resolve the relation field once — every source row has the
            // same type at this point.
            let field_name = resolve_relation_field(db, &source_type, target_type)?;
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
        } => {
            let vectorizer = ctx.vectorizer.ok_or_else(|| {
                QueryError::Type("vector similarity search requires a vectorizer".into())
            })?;

            let type_name = infer_type_from_objects(&[], source)
                .ok_or_else(|| QueryError::Type("cannot determine type for similar()".into()))?;

            let ef = (*k).max(50); // search width >= k
            let results = match query {
                SimilarQuery::Text(text) => {
                    vectorizer.search_text(&type_name, field_name, text, *k, ef)?
                }
                SimilarQuery::Vector(vec) => {
                    vectorizer.search_vector(&type_name, field_name, vec, *k, ef)?
                }
            };

            let objects: Vec<Object> = results
                .iter()
                .filter_map(|(id, _dist)| db.get(&type_name, *id).ok())
                .collect();

            Ok(QueryOutput::Objects(objects))
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
/// `get_many` so the whole set shares one read snapshot and the SSTs get
/// probed in sorted-key order.
fn materialize_ids(db: &Database, type_name: &str, ids: &[u64]) -> Vec<Object> {
    db.get_many(type_name, ids).unwrap_or_default()
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
        (Value::U64(a), Literal::Int(b)) => compare_ord(*a as i64, op, *b),
        (Value::I32(a), Literal::Int(b)) => compare_ord(*a as i64, op, *b),
        (Value::I64(a), Literal::Int(b)) => compare_ord(*a, op, *b),
        (Value::F32(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::F64(a), Literal::Float(b)) => compare_ord(*a, op, *b),
        (Value::F32(a), Literal::Int(b)) => compare_ord(*a as f64, op, *b as f64 ),
        (Value::F64(a), Literal::Int(b)) => compare_ord(*a, op, *b as f64 ),
        (Value::U32(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
        (Value::U64(a), Literal::Float(b)) => compare_ord(*a as f64, op, *b),
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

fn literal_map_to_field_map(
    literals: &HashMap<String, Literal>,
) -> QueryResult<FieldMap> {
    let mut fields = FieldMap::new();
    for (name, lit) in literals {
        let value = literal_to_value(lit)?;
        fields.insert(name.clone(), value);
    }
    Ok(fields)
}

fn literal_to_value(lit: &Literal) -> QueryResult<Value> {
    match lit {
        Literal::String(s) => Ok(Value::String(s.clone())),
        Literal::Int(i) => {
            if *i >= 0 && *i <= u32::MAX as i64 {
                Ok(Value::U32(*i as u32))
            } else {
                Ok(Value::I64(*i))
            }
        }
        Literal::Float(f) => Ok(Value::F32(*f as f32)),
        Literal::Bool(b) => Ok(Value::Bool(*b)),
        Literal::Null => Ok(Value::Null),
    }
}

fn infer_type_from_objects(objects: &[Object], source: &Source) -> Option<String> {
    if let Some(first) = objects.first() {
        return Some(first.type_name.clone());
    }
    source_type_name(source)
}

fn scan_all_objects(db: &Database, type_name: &str) -> QueryResult<Vec<Object>> {
    Ok(db.scan_type(type_name)?)
}

/// Recognize the shape `Filter(Compare { int_field, op, int_literal })` and
/// push it down to `Database::filter_scan` so storage can zone-prune blocks.
/// Returns `Ok(Some(objects))` on match, `Ok(None)` if the predicate is too
/// complex for the fast path (And/Or, string compare, missing field, etc.) —
/// in which case the caller falls back to the full scan + filter.
fn try_filter_scan(
    db: &Database,
    type_name: &str,
    predicate: &Predicate,
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
    // Only integer literals get the zone-map fast path. Strings, floats,
    // bools, and null fall through.
    let target = match value {
        Literal::Int(i) => *i,
        _ => return Ok(None),
    };
    Ok(Some(db.filter_scan(type_name, field_path, storage_op, target)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_query;
    use rhypedb_schema::parser::parse_schema;

    fn test_db(dir: &std::path::Path) -> Database {
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

    #[test]
    fn execute_create_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let q = parse_query(r#"User.create({ name: "Alice", age: 30 })"#).unwrap();
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
        let obj = match result {
            QueryOutput::Single(o) => o,
            _ => panic!("expected Single"),
        };
        assert_eq!(obj.fields.get("name"), Some(&Value::String("Alice".into())));

        let q = parse_query(&format!("User.get({})", obj.id)).unwrap();
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
            let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();

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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();

        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 1);
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
        assert!(matches!(result, QueryOutput::Done));

        // Verify the link landed via direct db query.
        let links = db.get_links("User", alice.id, "friends").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0, bob.id);
    }

    #[test]
    fn execute_limit() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        for i in 0..5 {
            create_user(&db, &format!("User{i}"), 20 + i);
        }

        let q = parse_query("User.filter(.age >= 0).limit(2)").unwrap();
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
        match result {
            QueryOutput::Objects(objs) => {
                assert_eq!(objs.len(), 2);
            }
            _ => panic!("expected Objects"),
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
        let result = execute(&ExecContext { db: &db, vectorizer: None }, &q).unwrap();
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
