//! A guard that reads OTHER stored state through its [`GuardView`] (a
//! relation's edge set, a linked object) has that state in the write's
//! conflict check: a competing commit to it after the write's snapshot fails
//! the attempt, and the retry decides again on a fresh read.
//!
//! Each test commits the competing change from INSIDE the guard on its first
//! call (after the attempt took its snapshot and read the state), the same
//! deterministic replay `guarded_tests.rs` uses for the object itself.

use std::cell::Cell;

use super::*;
use rhypedb_schema::parser::parse_schema;

fn open(dir: &Path) -> Arc<Database> {
    let schema = parse_schema(
        r#"
        type User {
            uid: String
            posts: [Post] @inverse(Post.editors)
            leading: [Post] @inverse(Post.lead)
        }
        type Post {
            title: String
            editors: [User]
            lead: User
        }
        "#,
    )
    .unwrap();
    Database::open(schema, dir).unwrap()
}

fn string_fields(field: &str, value: &str) -> FieldMap {
    let mut f = FieldMap::new();
    f.insert(field.into(), Value::String(value.into()));
    f
}

fn user(db: &Database, uid: &str) -> u64 {
    db.create("User", string_fields("uid", uid)).unwrap().id
}

fn post(db: &Database) -> u64 {
    db.create("Post", string_fields("title", "t")).unwrap().id
}

/// The `uid`s of the guarded post's editors, read through the view.
fn editor_uids(v: &GuardView<'_>) -> Vec<String> {
    v.link_ids("editors")
        .unwrap()
        .into_iter()
        .filter_map(|id| v.object_fields("User", id).unwrap())
        .filter_map(|f| match f.get("uid") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// A guard that runs `race` once, on its first call, then accepts only if
/// `decide` does; counts its calls.
fn racing<'a>(
    race: impl Fn() + 'a,
    decide: impl Fn(&GuardView<'_>) -> bool + 'a,
    calls: &'a Cell<u32>,
) -> impl Fn(&GuardView<'_>) -> bool + 'a {
    move |v| {
        calls.set(calls.get() + 1);
        if calls.get() == 1 {
            race();
        }
        decide(v)
    }
}

fn title_of(db: &Database, id: u64) -> FieldMap {
    let mut obj = db.get("Post", id).unwrap();
    obj.ensure_fields_deserialized();
    obj.fields
}

#[test]
fn an_edge_added_after_the_snapshot_is_seen_by_the_retry() {
    // The phantom case: a deny-list ("not if `bob` edits it") read an edge set
    // that a concurrent link then grows by a key the guard never saw.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (p, bob) = (post(&db), user(&db, "bob"));
    let calls = Cell::new(0);
    let guard = racing(
        || db.link("Post", p, "editors", bob, None).unwrap(),
        |v| !editor_uids(v).contains(&"bob".to_string()),
        &calls,
    );
    let out = db
        .update_if("Post", p, &string_fields("title", "x"), None, &guard)
        .unwrap();
    assert!(out.is_none(), "the retry sees bob and rejects");
    assert_eq!(calls.get(), 2, "the first attempt conflicted and retried");
    assert_eq!(
        title_of(&db, p).get("title"),
        Some(&Value::String("t".into()))
    );
}

#[test]
fn an_edge_removed_after_the_snapshot_is_seen_by_the_retry() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (p, alice) = (post(&db), user(&db, "alice"));
    db.link("Post", p, "editors", alice, None).unwrap();
    let calls = Cell::new(0);
    let guard = racing(
        || db.unlink("Post", p, "editors", alice).unwrap(),
        |v| editor_uids(v).contains(&"alice".to_string()),
        &calls,
    );
    assert!(!db.delete_if("Post", p, None, &guard).unwrap());
    assert_eq!(calls.get(), 2);
    assert!(db.get("Post", p).is_ok(), "nothing deleted");
}

#[test]
fn a_linked_object_changed_after_the_snapshot_is_seen_by_the_retry() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (p, u) = (post(&db), user(&db, "alice"));
    db.link("Post", p, "editors", u, None).unwrap();
    let other = user(&db, "carol");
    let calls = Cell::new(0);
    let guard = racing(
        || {
            db.update("User", u, string_fields("uid", "mallory"))
                .unwrap();
        },
        |v| editor_uids(v).contains(&"alice".to_string()),
        &calls,
    );
    // link_if does not write the post's blob: the pins alone protect it.
    assert!(
        !db.link_if("Post", p, "editors", other, None, &guard)
            .unwrap()
    );
    assert_eq!(calls.get(), 2);
    assert_eq!(db.get_link_ids("Post", p, "editors").unwrap(), vec![u]);
}

#[test]
fn an_inverse_edge_added_after_the_snapshot_is_seen_by_the_retry() {
    // The guarded object reads its `@inverse` set (reverse edges): a link
    // from ANOTHER object's side changes it.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (u, p) = (user(&db, "alice"), post(&db));
    let calls = Cell::new(0);
    let guard = racing(
        || db.link("Post", p, "editors", u, None).unwrap(),
        |v| v.link_ids("posts").unwrap().is_empty(),
        &calls,
    );
    let out = db
        .update_if("User", u, &string_fields("uid", "x"), None, &guard)
        .unwrap();
    assert!(out.is_none());
    assert_eq!(calls.get(), 2);
}

#[test]
fn an_inverse_edge_added_by_a_create_is_seen_by_the_retry() {
    // Creating a post already linked to the user adds a reverse edge too.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let u = user(&db, "alice");
    let calls = Cell::new(0);
    let guard = racing(
        || {
            let mut f = string_fields("title", "new");
            f.insert("lead".into(), Value::U64(u));
            db.create("Post", f).unwrap();
        },
        |v| v.link_ids("leading").unwrap().is_empty(),
        &calls,
    );
    let out = db
        .update_if("User", u, &string_fields("uid", "x"), None, &guard)
        .unwrap();
    assert!(out.is_none());
    assert_eq!(calls.get(), 2);
}

#[test]
fn unrelated_writes_do_not_force_a_retry() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (p, alice) = (post(&db), user(&db, "alice"));
    db.link("Post", p, "editors", alice, None).unwrap();
    let (q, bob) = (post(&db), user(&db, "bob"));
    let calls = Cell::new(0);
    let guard = racing(
        || {
            db.link("Post", q, "editors", bob, None).unwrap();
            db.update("User", bob, string_fields("uid", "bobby"))
                .unwrap();
        },
        |v| editor_uids(v).contains(&"alice".to_string()),
        &calls,
    );
    let out = db
        .update_if("Post", p, &string_fields("title", "x"), None, &guard)
        .unwrap();
    assert!(out.is_some());
    assert_eq!(calls.get(), 1, "no key the guard read changed");
}
