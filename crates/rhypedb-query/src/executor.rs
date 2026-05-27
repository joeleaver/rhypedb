use std::collections::HashMap;

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{FieldMap, Object, Value};
use rhypedb_engine::vectorizer::Vectorizer;

use crate::ast::*;
use crate::error::{QueryError, QueryResult};

/// Result of executing a query.
#[derive(Debug)]
pub enum QueryOutput {
    /// A list of objects (from get, filter, traverse).
    Objects(Vec<Object>),

    /// A single created/updated object.
    Single(Object),

    /// Void result (delete, link, unlink).
    Done,
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
            let objects = extract_objects(current)?;
            let source_type = infer_type_from_objects(&objects, source);
            let mut results = Vec::new();

            for obj in &objects {
                let type_name = source_type.as_deref().unwrap_or(&obj.type_name);
                let links = db.get_links(type_name, obj.id, field_name)?;

                // Determine the target type from the schema.
                let target_type = db
                    .schema()
                    .get_type(type_name)
                    .and_then(|td| td.get_field(field_name))
                    .and_then(|fd| match &fd.field_type {
                        rhypedb_schema::FieldType::Relation(rel) => Some(rel.target_type.clone()),
                        _ => None,
                    });

                if let Some(target_type) = target_type {
                    for (target_id, _edge_fields) in links {
                        if let Ok(target_obj) = db.get(&target_type, target_id) {
                            results.push(target_obj);
                        }
                    }
                }
            }

            Ok(QueryOutput::Objects(results))
        }

        Step::Filter { predicate } => {
            let objects = extract_objects(current)?;
            let filtered = objects
                .into_iter()
                .filter(|obj| evaluate_predicate(predicate, &obj.fields))
                .collect();
            Ok(QueryOutput::Objects(filtered))
        }

        Step::Update { fields } => {
            let objects = extract_objects(current)?;
            let field_map = literal_map_to_field_map(fields)?;
            let mut updated = Vec::new();
            for obj in &objects {
                let result = db.update(&obj.type_name, obj.id, field_map.clone())?;
                updated.push(result);
            }
            if updated.len() == 1 {
                Ok(QueryOutput::Single(updated.remove(0)))
            } else {
                Ok(QueryOutput::Objects(updated))
            }
        }

        Step::Delete => {
            let objects = extract_objects(current)?;
            for obj in &objects {
                db.delete(&obj.type_name, obj.id)?;
            }
            Ok(QueryOutput::Done)
        }

        Step::Link {
            target_type,
            target_id,
            edge_fields,
        } => {
            let objects = extract_objects(current)?;
            let edge_map = if edge_fields.is_empty() {
                None
            } else {
                Some(literal_map_to_field_map(edge_fields)?)
            };

            // The link step comes after a traverse step, so we need to figure out
            // the relationship field name from the previous traverse. This is a
            // limitation — for now, we need to look at the query context.
            // The step itself needs the source type and field name.
            // In practice, the traverse before link sets up this context.
            //
            // For now: the link operates on the source objects from the previous step.
            // The field name was the traverse before this step.
            for obj in &objects {
                db.link(
                    &obj.type_name,
                    obj.id,
                    target_type, // This is actually the relationship field name in the current design
                    *target_id,
                    edge_map.clone(),
                )?;
            }

            Ok(QueryOutput::Done)
        }

        Step::Unlink {
            target_type,
            target_id,
        } => {
            let objects = extract_objects(current)?;
            for obj in &objects {
                db.unlink(&obj.type_name, obj.id, target_type, *target_id)?;
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

        Step::Limit { count } => {
            let mut objects = extract_objects(current)?;
            objects.truncate(*count);
            Ok(QueryOutput::Objects(objects))
        }

        Step::Offset { count } => {
            let objects = extract_objects(current)?;
            let skipped = objects.into_iter().skip(*count).collect();
            Ok(QueryOutput::Objects(skipped))
        }
    }
}

fn extract_objects(output: QueryOutput) -> QueryResult<Vec<Object>> {
    match output {
        QueryOutput::Objects(objs) => Ok(objs),
        QueryOutput::Single(obj) => Ok(vec![obj]),
        QueryOutput::Done => Ok(Vec::new()),
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
    match source {
        Source::Get { type_name, .. }
        | Source::Filter { type_name, .. }
        | Source::Create { type_name, .. }
        | Source::All { type_name } => Some(type_name.clone()),
    }
}

fn scan_all_objects(db: &Database, type_name: &str) -> QueryResult<Vec<Object>> {
    Ok(db.scan_type(type_name)?)
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
