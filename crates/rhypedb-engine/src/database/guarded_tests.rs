//! Guarded writes (`update_if` / `delete_if` / `link_if` / `unlink_if`) are
//! compare-and-set on the object's current committed version.
//!
//! The race they close: a filter matched the object on an older snapshot, and
//! another writer committed before the guarded write ran. These tests replay
//! it deterministically by committing the competing write from INSIDE the
//! guard, i.e. after the guarded attempt took its snapshot and read the object
//! but before it commits. That attempt must then fail its commit and be
//! retried on a fresh read, where the guard sees the competing write.

use std::cell::Cell;

use super::*;
use rhypedb_schema::parser::parse_schema;

fn open(dir: &Path) -> Arc<Database> {
    let schema = parse_schema(
        r#"
        type Account {
            step: u64
            peers: [Account]
        }
        "#,
    )
    .unwrap();
    Database::open(schema, dir).unwrap()
}

fn step_fields(step: u64) -> FieldMap {
    let mut f = FieldMap::new();
    f.insert("step".into(), Value::U64(step));
    f
}

fn create(db: &Database, step: u64) -> u64 {
    db.create("Account", step_fields(step)).unwrap().id
}

fn step_of(db: &Database, id: u64) -> u64 {
    let mut obj = db.get("Account", id).unwrap();
    obj.ensure_fields_deserialized();
    match obj.fields.get("step") {
        Some(Value::U64(s)) => *s,
        other => panic!("unexpected step {other:?}"),
    }
}

fn step_is(want: u64) -> impl Fn(&GuardView<'_>) -> bool {
    move |v| v.fields().get("step") == Some(&Value::U64(want))
}

/// A guard for `step == want` that, on its FIRST call only, commits
/// `step = competing` to `id` first: the write lands after the guarded
/// attempt's snapshot, exactly the stale-filter race.
fn racing_guard(
    db: &Database,
    id: u64,
    want: u64,
    competing: u64,
) -> impl Fn(&GuardView<'_>) -> bool {
    let raced = Cell::new(false);
    move |v| {
        if !raced.replace(true) {
            db.update("Account", id, step_fields(competing)).unwrap();
        }
        v.fields().get("step") == Some(&Value::U64(want))
    }
}

#[test]
fn update_if_applies_when_the_guard_accepts() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    let out = db
        .update_if("Account", id, &step_fields(1), None, &step_is(0))
        .unwrap();
    assert_eq!(out.map(|o| o.id), Some(id));
    assert_eq!(step_of(&db, id), 1);
}

#[test]
fn update_if_skips_a_rejected_or_missing_object() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 3);
    assert!(
        db.update_if("Account", id, &step_fields(9), None, &step_is(0))
            .unwrap()
            .is_none()
    );
    assert_eq!(step_of(&db, id), 3);
    assert_eq!(
        db.object_version("Account", id),
        1,
        "a skipped update must not bump the generation"
    );
    // A missing object no longer matches: skipped, not ObjectNotFound...
    assert!(
        db.update_if("Account", 9_999, &step_fields(1), None, &|_| true)
            .unwrap()
            .is_none()
    );
    // ...while the unguarded update keeps its contract.
    assert!(matches!(
        db.update("Account", 9_999, step_fields(1)),
        Err(EngineError::ObjectNotFound { .. })
    ));
}

#[test]
fn update_if_rechecks_after_a_competing_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    let guard = racing_guard(&db, id, 0, 5);
    let out = db
        .update_if("Account", id, &step_fields(1), None, &guard)
        .unwrap();
    assert!(out.is_none(), "the guard matched a stale version: {out:?}");
    assert_eq!(step_of(&db, id), 5, "the competing write must survive");
}

#[test]
fn update_if_applies_when_the_row_still_matches_after_a_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    // The competing write keeps `step == 0`: the retry still matches, so the
    // guarded update applies on top of it.
    let guard = racing_guard(&db, id, 0, 0);
    assert!(
        db.update_if("Account", id, &step_fields(1), None, &guard)
            .unwrap()
            .is_some()
    );
    assert_eq!(step_of(&db, id), 1);
}

#[test]
fn update_if_gives_up_after_bounded_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    let calls = Cell::new(0usize);
    // Every attempt loses to a fresh competing commit that keeps the guard true.
    let guard = |v: &GuardView<'_>| {
        calls.set(calls.get() + 1);
        db.update("Account", id, step_fields(0)).unwrap();
        v.fields().get("step") == Some(&Value::U64(0))
    };
    let out = db.update_if("Account", id, &step_fields(1), None, &guard);
    assert!(matches!(out, Err(EngineError::WriteConflict)), "{out:?}");
    assert_eq!(calls.get(), guarded::MAX_GUARDED_ATTEMPTS);
    assert_eq!(step_of(&db, id), 0);
}

#[test]
fn delete_if_rechecks_after_a_competing_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    assert!(!db.delete_if("Account", id, None, &step_is(7)).unwrap());
    let guard = racing_guard(&db, id, 0, 5);
    assert!(!db.delete_if("Account", id, None, &guard).unwrap());
    assert_eq!(step_of(&db, id), 5, "deleted on a stale match");
    assert!(db.delete_if("Account", id, None, &step_is(5)).unwrap());
    assert!(db.get("Account", id).is_err());
    // Gone: a second guarded delete is a clean no-op.
    assert!(!db.delete_if("Account", id, None, &|_| true).unwrap());
}

#[test]
fn link_if_rechecks_the_source_after_a_competing_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    let peer = create(&db, 0);
    // A link does not write the source's blob: only the read guard on it makes
    // the competing update conflict with this link.
    let guard = racing_guard(&db, id, 0, 5);
    assert!(
        !db.link_if("Account", id, "peers", peer, None, &guard)
            .unwrap()
    );
    assert!(db.get_links("Account", id, "peers").unwrap().is_empty());
    assert!(
        db.link_if("Account", id, "peers", peer, None, &step_is(5))
            .unwrap()
    );
    assert_eq!(db.get_links("Account", id, "peers").unwrap().len(), 1);
    // A missing target is still an error, not a skip.
    assert!(matches!(
        db.link_if("Account", id, "peers", 9_999, None, &|_| true),
        Err(EngineError::ObjectNotFound { .. })
    ));
}

#[test]
fn unlink_if_rechecks_the_source_after_a_competing_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, 0);
    let peer = create(&db, 0);
    db.link("Account", id, "peers", peer, None).unwrap();
    let guard = racing_guard(&db, id, 0, 5);
    assert!(!db.unlink_if("Account", id, "peers", peer, &guard).unwrap());
    assert_eq!(db.get_links("Account", id, "peers").unwrap().len(), 1);
    assert!(
        db.unlink_if("Account", id, "peers", peer, &step_is(5))
            .unwrap()
    );
    assert!(db.get_links("Account", id, "peers").unwrap().is_empty());
}
