//! Deterministic tests of the write guard: which predicates a write step
//! re-checks, and that a write handed a STALE selection (the filter ran on a
//! snapshot, then the row changed) skips the rows that no longer match.

use super::super::{ExecContext, QueryOutput, execute, execute_step};
use super::*;
use crate::governor::Governor;
use crate::parser::parse_query;
use rhypedb_engine::object::Value;
use rhypedb_schema::parser::parse_schema;

fn open(dir: &std::path::Path) -> std::sync::Arc<Database> {
    let schema = parse_schema(
        r#"
        type Account {
            handle: String
            step: u64
            peers: [Account]
        }
        "#,
    )
    .unwrap();
    Database::open(schema, dir).unwrap()
}

fn ctx(db: &Database) -> ExecContext<'_> {
    ExecContext {
        db,
        vectorizer: None,
        default_ef: None,
        default_rerank: None,
        governor: Governor::disabled(),
        principal: rhypedb_authz::Principal::anonymous(),
        rules: None,
    }
}

fn run(db: &Database, q: &str) -> QueryOutput {
    execute(&ctx(db), &parse_query(q).unwrap()).unwrap()
}

fn create(db: &Database, handle: &str, step: u64) -> u64 {
    match run(
        db,
        &format!(r#"Account.create({{ handle: "{handle}", step: {step} }})"#),
    ) {
        QueryOutput::Single(o) => o.id,
        other => panic!("{other:?}"),
    }
}

fn step_of(db: &Database, id: u64) -> u64 {
    let mut obj = db.get("Account", id).unwrap();
    obj.ensure_fields_deserialized();
    match obj.fields.get("step") {
        Some(Value::U64(s)) => *s,
        other => panic!("{other:?}"),
    }
}

/// Run the query's write step on `stale` (what its pipeline selected earlier)
/// instead of on a fresh selection: the interleaving where another writer
/// committed between the filter's snapshot and the write.
fn write_on_stale(db: &Database, q: &str, stale: QueryOutput) -> QueryOutput {
    let query = parse_query(q).unwrap();
    let (last, before) = query.steps.split_last().unwrap();
    let guard = write_guard(&query.source, before);
    execute_step(&ctx(db), stale, last, &query.source, &guard).unwrap()
}

/// What `q` without its final (write) step selects now.
fn selection(db: &Database, q: &str) -> QueryOutput {
    let mut query = parse_query(q).unwrap();
    query.steps.pop();
    execute(&ctx(db), &query).unwrap()
}

#[test]
fn guard_collects_the_predicates_about_the_written_objects() {
    let q = |s: &str| parse_query(s).unwrap();
    let guard_of = |s: &str| {
        let query = q(s);
        let n = query.steps.len() - 1;
        format!("{:?}", write_guard(&query.source, &query.steps[..n]))
    };
    assert_eq!(guard_of("Account.get(1).update({ step: 1 })"), "Point");
    assert_eq!(guard_of("Account.update({ step: 1 })"), "Recheck([])");
    let filtered =
        guard_of(r#"Account.filter(.handle == "a").filter(.step == 2).limit(1).delete()"#);
    assert!(
        filtered.contains("\"handle\"") && filtered.contains("\"step\""),
        "{filtered}"
    );
    // `get` followed by a filter is a guarded set write.
    assert!(guard_of("Account.get(1).filter(.step == 2).delete()").contains("\"step\""));
    // A filter before a traversal constrained the OTHER side: not re-checked.
    let traversed = guard_of(r#"Account.filter(.handle == "a").peers.filter(.step == 3).delete()"#);
    assert!(
        !traversed.contains("\"handle\"") && traversed.contains("\"step\""),
        "{traversed}"
    );
    assert_eq!(
        guard_of(r#"Account.filter(.handle == "a").peers.delete()"#),
        "Recheck([])"
    );
}

#[test]
fn update_on_a_stale_selection_skips_the_row_that_changed() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = create(&db, "a", 0);
    let b = create(&db, "b", 0);
    let q = r#"Account.filter(.step == 0).update({ step: 1 })"#;
    let stale = selection(&db, q);
    // A competing writer moves `a` on after the filter's snapshot.
    run(&db, &format!("Account.get({a}).update({{ step: 9 }})"));
    match write_on_stale(&db, q, stale) {
        QueryOutput::Single(o) => assert_eq!(o.id, b, "only the still-matching row is returned"),
        other => panic!("expected only b back, got {other:?}"),
    }
    assert_eq!((step_of(&db, a), step_of(&db, b)), (9, 1));
}

#[test]
fn update_on_a_stale_selection_of_one_row_returns_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = create(&db, "a", 0);
    let q = r#"Account.filter(.handle == "a" && .step == 0).update({ step: 1 })"#;
    let stale = selection(&db, q);
    run(&db, &format!("Account.get({a}).update({{ step: 1 }})"));
    assert!(matches!(write_on_stale(&db, q, stale), QueryOutput::Objects(o) if o.is_empty()));
    // And a row deleted since the snapshot is skipped, not an error.
    let stale = selection(&db, r#"Account.filter(.handle == "a").update({ step: 2 })"#);
    run(&db, &format!("Account.get({a}).delete()"));
    let out = write_on_stale(
        &db,
        r#"Account.filter(.handle == "a").update({ step: 2 })"#,
        stale,
    );
    assert!(
        matches!(&out, QueryOutput::Objects(o) if o.is_empty()),
        "{out:?}"
    );
}

#[test]
fn delete_link_and_unlink_on_a_stale_selection_skip_the_changed_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = create(&db, "a", 0);
    let peer = create(&db, "peer", 5);
    let guarded = r#"Account.filter(.handle == "a" && .step == 0)"#;
    let del = format!("{guarded}.delete()");
    let link = format!("{guarded}.link(Account.get({peer}))");
    let stale_del = selection(&db, &del);
    let stale_link = selection(&db, &link);
    run(&db, &format!("Account.get({a}).update({{ step: 1 }})"));
    write_on_stale(&db, &del, stale_del);
    write_on_stale(&db, &link, stale_link);
    assert_eq!(step_of(&db, a), 1, "deleted on a stale match");
    assert!(
        db.get_links("Account", a, "peers").unwrap().is_empty(),
        "linked on a stale match"
    );

    db.link("Account", a, "peers", peer, None).unwrap();
    let unlink =
        format!(r#"Account.filter(.handle == "a" && .step == 1).unlink(Account.get({peer}))"#);
    let stale_unlink = selection(&db, &unlink);
    run(&db, &format!("Account.get({a}).update({{ step: 2 }})"));
    write_on_stale(&db, &unlink, stale_unlink);
    assert_eq!(
        db.get_links("Account", a, "peers").unwrap().len(),
        1,
        "unlinked on a stale match"
    );
}

#[test]
fn point_writes_keep_their_not_found_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let q = parse_query("Account.get(424242).update({ step: 1 })").unwrap();
    assert!(execute(&ctx(&db), &q).is_err());
}
