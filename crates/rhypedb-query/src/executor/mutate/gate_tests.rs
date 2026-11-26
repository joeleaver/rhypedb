//! The security rules are decided INSIDE each object's write transaction, on the version the write
//! is based on: a rule input changed after the early check, or mid-attempt, cannot let a write the
//! rules no longer allow land.
//!
//! Each deterministic test commits the competing change from inside the gate's first evaluation
//! ([`WriteGate::race_once`]): after the write attempt took its snapshot and read the object, before
//! it commits. The early (snapshot) check has already passed by then, which is exactly the old
//! time-of-check-to-time-of-use window. The attempt must conflict, retry, re-evaluate and deny.
//!
//! Every gated write path is covered, point (`T.get(id)`) and set (`T.filter(..)`/`T.all()`).

use std::sync::Arc;

use super::*;
use crate::error::QueryError;
use crate::executor::{ExecContext, execute};
use crate::parser::parse_query;
use rhypedb_authz::{Op, Principal, RulesProgram};
use rhypedb_engine::object::Value;
use rhypedb_schema::parser::parse_schema;

const SCHEMA: &str = r#"
    type User {
        uid: String
        edits: [Post] @inverse(Post.editors)
    }
    type Post {
        title: String
        ownerUid: String
        editors: [User]
        blocked: [User]
    }
"#;

fn open(dir: &std::path::Path) -> Arc<Database> {
    Database::open(parse_schema(SCHEMA).unwrap(), dir).unwrap()
}

fn strings(pairs: &[(&str, &str)]) -> FieldMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect()
}

fn user(db: &Database, uid: &str) -> u64 {
    db.create("User", strings(&[("uid", uid)])).unwrap().id
}

fn post(db: &Database, owner: &str) -> u64 {
    db.create("Post", strings(&[("title", "t"), ("ownerUid", owner)]))
        .unwrap()
        .id
}

fn field(db: &Database, ty: &str, id: u64, name: &str) -> Option<Value> {
    let mut obj = db.get(ty, id).ok()?;
    obj.ensure_fields_deserialized();
    obj.fields.get(name).cloned()
}

fn title(db: &Database, id: u64) -> Option<Value> {
    field(db, "Post", id, "title")
}

fn s(v: &str) -> Option<Value> {
    Some(Value::String(v.into()))
}

fn ctx<'a>(db: &'a Database, rules: &str, uid: &str) -> ExecContext<'a> {
    let mut c = ExecContext::new(db, None);
    c.principal = Principal::new_authenticated(uid, serde_json::Value::Null);
    c.rules = Some(Arc::new(RulesProgram::compile(rules, db.schema()).unwrap()));
    c
}

const OWNER_RULES: &str = "match Post { \
    allow update: if request.auth.uid == resource.ownerUid; \
    allow delete: if request.auth.uid == resource.ownerUid; }";

/// The two guards a write step can run under: a point write and an (unfiltered) set write.
fn both_guards() -> [WriteGuard<'static>; 2] {
    [WriteGuard::Point, WriteGuard::Recheck(Vec::new())]
}

fn is_denied<T: std::fmt::Debug>(r: QueryResult<T>) -> bool {
    matches!(r, Err(QueryError::PermissionDenied(_)))
}

#[test]
fn update_denied_when_ownership_changes_after_the_check() {
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let p = post(&db, "alice");
        let c = ctx(&db, OWNER_RULES, "alice");
        let fields = strings(&[("title", "alice-was-here")]);
        let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], Some(&fields))
            .unwrap()
            .unwrap();
        gate.race_once(|| {
            db.update("Post", p, strings(&[("ownerUid", "bob")]))
                .unwrap();
        });
        let r = update_one(&db, (&guard, Some(&gate)), "Post", p, &fields);
        assert!(is_denied(r), "{guard:?}");
        assert_eq!(title(&db, p), s("t"), "{guard:?}: nothing written");
        assert_eq!(field(&db, "Post", p, "ownerUid"), s("bob"));
    }
}

#[test]
fn delete_denied_when_ownership_changes_after_the_check() {
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let p = post(&db, "alice");
        let c = ctx(&db, OWNER_RULES, "alice");
        let gate = WriteGate::prepare(&c, Op::Delete, "Post", &[p], None)
            .unwrap()
            .unwrap();
        gate.race_once(|| {
            db.update("Post", p, strings(&[("ownerUid", "bob")]))
                .unwrap();
        });
        assert!(
            is_denied(delete_one(&db, (&guard, Some(&gate)), "Post", p)),
            "{guard:?}"
        );
        assert!(db.get("Post", p).is_ok(), "{guard:?}: not deleted");
    }
}

#[test]
fn link_and_unlink_denied_when_ownership_changes_after_the_check() {
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let (p, kept, added) = (post(&db, "alice"), user(&db, "k"), user(&db, "a"));
        db.link("Post", p, "editors", kept, None).unwrap();
        let c = ctx(&db, OWNER_RULES, "alice");

        let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], None)
            .unwrap()
            .unwrap();
        gate.race_once(|| {
            db.update("Post", p, strings(&[("ownerUid", "bob")]))
                .unwrap();
        });
        let r = link_one(
            &db,
            (&guard, Some(&gate)),
            ("Post", p, "editors"),
            added,
            None,
        );
        assert!(is_denied(r), "{guard:?}");
        // No side effect: neither the edge nor its reverse edge.
        assert_eq!(db.get_link_ids("Post", p, "editors").unwrap(), vec![kept]);
        assert!(db.get_link_ids("User", added, "edits").unwrap().is_empty());

        // Ownership back to alice, then lost again mid-unlink.
        db.update("Post", p, strings(&[("ownerUid", "alice")]))
            .unwrap();
        let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], None)
            .unwrap()
            .unwrap();
        gate.race_once(|| {
            db.update("Post", p, strings(&[("ownerUid", "bob")]))
                .unwrap();
        });
        let r = unlink_one(&db, (&guard, Some(&gate)), ("Post", p, "editors"), kept);
        assert!(is_denied(r), "{guard:?}");
        assert_eq!(db.get_link_ids("Post", p, "editors").unwrap(), vec![kept]);
    }
}

#[test]
fn a_retry_re_evaluates_and_applies_when_the_rule_still_allows() {
    // A competing write that keeps the rule true: the attempt conflicts, the retry re-evaluates on
    // the new version, and the write lands on top of it.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let p = post(&db, "alice");
    let c = ctx(&db, OWNER_RULES, "alice");
    let fields = strings(&[("title", "mine")]);
    let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], Some(&fields))
        .unwrap()
        .unwrap();
    gate.race_once(|| {
        db.update("Post", p, strings(&[("title", "theirs")]))
            .unwrap();
    });
    let out = update_one(&db, (&WriteGuard::Point, Some(&gate)), "Post", p, &fields).unwrap();
    assert!(out.is_some());
    assert_eq!(title(&db, p), s("mine"));
}

#[test]
fn a_relation_membership_revoked_after_the_check_denies() {
    let rules = "match Post { allow update: if request.auth.uid in resource.editors.uid; }";
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let (p, ed) = (post(&db, "o"), user(&db, "ed"));
        db.link("Post", p, "editors", ed, None).unwrap();
        let c = ctx(&db, rules, "ed");
        let fields = strings(&[("title", "x")]);
        let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], Some(&fields))
            .unwrap()
            .unwrap();
        gate.race_once(|| db.unlink("Post", p, "editors", ed).unwrap());
        assert!(is_denied(update_one(
            &db,
            (&guard, Some(&gate)),
            "Post",
            p,
            &fields
        )));
        assert_eq!(title(&db, p), s("t"), "{guard:?}");
    }
}

#[test]
fn a_deny_list_member_added_after_the_check_denies() {
    // The phantom: the concurrent link adds an edge key the first attempt never read.
    let rules = "match Post { allow update: if !(request.auth.uid in resource.blocked.uid); }";
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let (p, mallory) = (post(&db, "o"), user(&db, "mallory"));
        let c = ctx(&db, rules, "mallory");
        let fields = strings(&[("title", "x")]);
        let gate = WriteGate::prepare(&c, Op::Update, "Post", &[p], Some(&fields))
            .unwrap()
            .unwrap();
        gate.race_once(|| db.link("Post", p, "blocked", mallory, None).unwrap());
        assert!(is_denied(update_one(
            &db,
            (&guard, Some(&gate)),
            "Post",
            p,
            &fields
        )));
        assert_eq!(title(&db, p), s("t"), "{guard:?}");
    }
}

#[test]
fn a_linked_objects_field_changed_after_the_check_denies() {
    let rules = "match Post { allow delete: if request.auth.uid in resource.editors.uid; }";
    for guard in both_guards() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let (p, ed) = (post(&db, "o"), user(&db, "ed"));
        db.link("Post", p, "editors", ed, None).unwrap();
        let c = ctx(&db, rules, "ed");
        let gate = WriteGate::prepare(&c, Op::Delete, "Post", &[p], None)
            .unwrap()
            .unwrap();
        gate.race_once(|| {
            db.update("User", ed, strings(&[("uid", "someone-else")]))
                .unwrap();
        });
        assert!(is_denied(delete_one(&db, (&guard, Some(&gate)), "Post", p)));
        assert!(db.get("Post", p).is_ok(), "{guard:?}");
    }
}

#[test]
fn gated_point_writes_keep_their_missing_object_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = ctx(&db, OWNER_RULES, "alice");
    let missing = 9_999;
    let fields = strings(&[("title", "x")]);
    let gate = WriteGate::prepare(&c, Op::Update, "Post", &[missing], Some(&fields))
        .unwrap()
        .unwrap();
    let r = update_one(
        &db,
        (&WriteGuard::Point, Some(&gate)),
        "Post",
        missing,
        &fields,
    );
    assert!(matches!(
        r,
        Err(QueryError::Engine(
            rhypedb_engine::EngineError::ObjectNotFound { .. }
        ))
    ));
    // A set write skips it; a point unlink from a missing source stays a no-op.
    let set = WriteGuard::Recheck(Vec::new());
    assert!(
        update_one(&db, (&set, Some(&gate)), "Post", missing, &fields)
            .unwrap()
            .is_none()
    );
    let u = user(&db, "u");
    let gate = WriteGate::prepare(&c, Op::Update, "Post", &[missing], None)
        .unwrap()
        .unwrap();
    assert!(
        unlink_one(
            &db,
            (&WriteGuard::Point, Some(&gate)),
            ("Post", missing, "editors"),
            u
        )
        .unwrap()
    );
}

/// An unconditional write that retries its own write-write conflicts (the stress test's writers
/// contend on one object).
fn force(db: &Database, id: u64, fields: FieldMap) {
    db.update_if("Post", id, &fields, None, &|_| true)
        .unwrap()
        .unwrap();
}

/// Concurrency stress through the full query path, under a deny-list rule
/// (`!(uid in resource.blocked.uid)`). The main thread repeatedly blocks `mallory` (a link), then
/// clears `title`, waits, checks the title is still clear, and unblocks her; another thread keeps
/// writing `title` as mallory. Once the block has committed no write by mallory may commit, so
/// finding a title after the wait means a write the rules no longer allowed landed. The blocked
/// list holds a few hundred other users, so evaluating the rule takes long enough that, under the
/// pre-fix check (on a separate read before the write), nearly every round's block landed between
/// mallory's check and her write.
///
/// Every round is a real race: after unblocking, the flipper waits until one of mallory's writes
/// has applied before it blocks her again, so the block always lands while her tight loop is
/// already into its next attempt. A fixed sleep could not promise that: under load a single
/// evaluation can outlast it, and then the writer only ever saw the blocked state. The wait is
/// bounded by a generous deadline, so a writer that stops making progress fails the test rather
/// than hanging it.
#[test]
fn concurrent_rule_input_changes_never_let_a_denied_write_land() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    const ROUNDS: usize = 300;
    let db = Database::open_in_memory(
        parse_schema(SCHEMA).unwrap(),
        rhypedb_engine::database::OpenOptions::default(),
    )
    .unwrap();
    let p = post(&db, "o");
    for i in 0..300 {
        let u = user(&db, &format!("filler{i}"));
        db.link("Post", p, "blocked", u, None).unwrap();
    }
    let mallory = user(&db, "mallory");
    let rules = "match Post { allow update: if !(request.auth.uid in resource.blocked.uid); }";
    let (stop, applied) = (AtomicBool::new(false), AtomicUsize::new(0));
    // Long enough for a write that passed a stale check to land.
    let pause = Duration::from_millis(1);
    // How long one unblocked window may wait for mallory's next write to apply.
    let progress_deadline = Duration::from_secs(60);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let c = ctx(&db, rules, "mallory");
            let q =
                parse_query(&format!("Post.get({p}).update({{ title: \"mallory\" }})")).unwrap();
            while !stop.load(Ordering::Relaxed) {
                match execute(&c, &q) {
                    Ok(_) => {
                        applied.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(QueryError::PermissionDenied(_)) => {}
                    // Bounded retries exhausted under contention: nothing was written.
                    Err(QueryError::Engine(rhypedb_engine::EngineError::WriteConflict)) => {}
                    Err(e) => panic!("unexpected error: {e}"),
                }
            }
        });
        let fail = |msg: String| -> ! {
            stop.store(true, Ordering::Relaxed);
            panic!("{msg}");
        };
        for round in 0..ROUNDS {
            db.link("Post", p, "blocked", mallory, None).unwrap();
            force(&db, p, strings(&[("title", "")]));
            std::thread::sleep(pause);
            let seen = title(&db, p);
            if seen != s("") {
                fail(format!(
                    "round {round}: blocked mallory's write landed ({seen:?})"
                ));
            }
            let before = applied.load(Ordering::Relaxed);
            db.unlink("Post", p, "blocked", mallory).unwrap();
            let started = Instant::now();
            while applied.load(Ordering::Relaxed) == before {
                if started.elapsed() > progress_deadline {
                    fail(format!(
                        "round {round}: mallory's writes made no progress while unblocked"
                    ));
                }
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
    assert!(applied.load(Ordering::Relaxed) >= ROUNDS);
}
