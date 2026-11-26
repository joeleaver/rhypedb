//! Tests for the data-plane authorization bridge.

use super::*;
use crate::executor::{QueryOutput, execute};
use crate::parser::parse_query;
use rhypedb_authz::{Principal, RulesProgram};
use rhypedb_engine::object::Value;
use rhypedb_schema::parser::parse_schema;
use std::sync::Arc;

const SCHEMA: &str = r#"
    type User {
        uid: String
        role: String
    }
    type Post {
        title: String
        published: Bool
        ownerUid: String
        author: User
        editors: [User]
    }
"#;

fn db() -> (tempfile::TempDir, Arc<Database>) {
    let dir = tempfile::tempdir().unwrap();
    let schema = parse_schema(SCHEMA).unwrap();
    let db = Database::open(schema, dir.path()).unwrap();
    (dir, db)
}

fn mk_post(db: &Database, title: &str, published: bool, owner: &str) -> u64 {
    let mut f = FieldMap::new();
    f.insert("title".into(), Value::String(title.into()));
    f.insert("published".into(), Value::Bool(published));
    f.insert("ownerUid".into(), Value::String(owner.into()));
    db.create("Post", f).unwrap().id
}

fn mk_user(db: &Database, uid: &str) -> u64 {
    let mut f = FieldMap::new();
    f.insert("uid".into(), Value::String(uid.into()));
    db.create("User", f).unwrap().id
}

/// Build a rules-on ExecContext for `principal`.
fn ctx_with<'a>(
    db: &'a Database,
    rules: &Arc<RulesProgram>,
    principal: Principal,
) -> ExecContext<'a> {
    let mut c = ExecContext::new(db, None);
    c.principal = principal;
    c.rules = Some(rules.clone());
    c
}

fn rules(src: &str, db: &Database) -> Arc<RulesProgram> {
    Arc::new(RulesProgram::compile(src, db.schema()).unwrap())
}

fn user(uid: &str) -> Principal {
    Principal::new_authenticated(uid, serde_json::Value::Null)
}

fn run(ctx: &ExecContext<'_>, q: &str) -> QueryResult<QueryOutput> {
    execute(ctx, &parse_query(q).unwrap())
}

fn n_objects(out: QueryResult<QueryOutput>) -> usize {
    match out.unwrap() {
        QueryOutput::Objects(o) => o.len(),
        QueryOutput::Single(_) => 1,
        _ => 0,
    }
}

#[test]
fn read_filter_drops_unreadable_rows() {
    let (_d, db) = db();
    mk_post(&db, "public", true, "u1");
    mk_post(&db, "u1-private", false, "u1");
    mk_post(&db, "u2-private", false, "u2");
    let r = rules(
        "match Post { allow read: if resource.published || request.auth.uid == resource.ownerUid; }",
        &db,
    );
    // u1 sees the public post + their own private one.
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u1")), "Post")), 2);
    // u2 sees the public post + their own.
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u2")), "Post")), 2);
    // anonymous sees only the public post.
    assert_eq!(
        n_objects(run(&ctx_with(&db, &r, Principal::anonymous()), "Post")),
        1
    );
}

#[test]
fn point_get_of_unreadable_row_is_empty() {
    let (_d, db) = db();
    let secret = mk_post(&db, "secret", false, "u2");
    let r = rules(
        "match Post { allow read: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    let q = format!("Post.get({secret})");
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u1")), &q)), 0);
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u2")), &q)), 1);
}

#[test]
fn read_filter_via_relationship_traversal() {
    let (_d, db) = db();
    let author = mk_user(&db, "u1");
    let pid = mk_post(&db, "owned", false, "u1");
    db.link("Post", pid, "author", author, None).unwrap();
    let other = mk_post(&db, "other", false, "u9");
    let other_author = mk_user(&db, "u9");
    db.link("Post", other, "author", other_author, None)
        .unwrap();
    let r = rules(
        "match Post { allow read: if request.auth.uid == resource.author.uid; }",
        &db,
    );
    // u1 is the author of exactly one post → sees one.
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u1")), "Post")), 1);
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u9")), "Post")), 1);
    assert_eq!(
        n_objects(run(&ctx_with(&db, &r, user("nobody")), "Post")),
        0
    );
}

#[test]
fn create_gate() {
    let (_d, db) = db();
    let r = rules("match Post { allow create: if request.auth != null; }", &db);
    let q = r#"Post.create({ title: "x", published: true, ownerUid: "u1" })"#;
    // Authenticated → allowed.
    assert!(run(&ctx_with(&db, &r, user("u1")), q).is_ok());
    // Anonymous → denied.
    let err = run(&ctx_with(&db, &r, Principal::anonymous()), q).unwrap_err();
    assert!(matches!(err, QueryError::PermissionDenied(_)));
}

#[test]
fn update_gate_checks_pre_mutation_object() {
    let (_d, db) = db();
    let pid = mk_post(&db, "t", false, "u1"); // owned by u1
    let r = rules(
        "match Post { allow update: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    // Owner can update.
    assert!(
        run(
            &ctx_with(&db, &r, user("u1")),
            &format!("Post.get({pid}).update({{ title: \"new\" }})")
        )
        .is_ok()
    );
    // P0-DBA-6: u2 tries to hijack by also rewriting ownerUid — the gate sees the PRE-mutation
    // ownerUid ("u1") and denies before the write applies.
    let hijack = format!("Post.get({pid}).update({{ ownerUid: \"u2\", title: \"pwned\" }})");
    let err = run(&ctx_with(&db, &r, user("u2")), &hijack).unwrap_err();
    assert!(matches!(err, QueryError::PermissionDenied(_)));
    // And the write did NOT apply — ownerUid is still u1 (re-read with rules off).
    let plain = ExecContext::new(&db, None);
    let out = run(&plain, &format!("Post.get({pid})"));
    if let QueryOutput::Objects(o) = out.unwrap() {
        let mut obj = o.into_iter().next().unwrap();
        obj.ensure_fields_deserialized();
        assert_eq!(
            obj.fields.get("ownerUid"),
            Some(&Value::String("u1".into()))
        );
    } else {
        panic!("expected the post to still exist");
    }
}

#[test]
fn update_return_value_is_read_filtered() {
    // Review HIGH: an update returns the post-image; that return must pass the READ rule, else a
    // principal who may update-but-not-read exfiltrates the whole object via the update response.
    let (_d, db) = db();
    let pid = mk_post(&db, "secret", false, "u1"); // owned by u1
    let r = rules(
        "match Post { allow update: if request.auth != null; allow read: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    // u2 may update (authenticated) but NOT read (not owner) → the returned object is filtered to
    // void; nothing leaks.
    let out = run(
        &ctx_with(&db, &r, user("u2")),
        &format!("Post.get({pid}).update({{ title: \"x\" }})"),
    );
    assert_eq!(
        n_objects(out),
        0,
        "non-owner's update return is read-filtered to void"
    );
    // ...but the write still applied (side effect): title changed (re-read rules-off).
    let plain = ExecContext::new(&db, None);
    if let QueryOutput::Objects(o) = run(&plain, &format!("Post.get({pid})")).unwrap() {
        let mut obj = o.into_iter().next().unwrap();
        obj.ensure_fields_deserialized();
        assert_eq!(obj.fields.get("title"), Some(&Value::String("x".into())));
    } else {
        panic!("post should still exist");
    }
    // The owner CAN read their own update's return value.
    let out = run(
        &ctx_with(&db, &r, user("u1")),
        &format!("Post.get({pid}).update({{ title: \"y\" }})"),
    );
    assert_eq!(n_objects(out), 1, "owner sees their update return value");
}

#[test]
fn oversized_relation_denylist_fails_closed() {
    // Review verify residual: a to-many relation too large to fully enumerate is INDETERMINATE
    // ⇒ deny — never truncated to a partial set (which would fail OPEN for a deny-list member
    // past the enumeration cap). End-to-end against a real DB + a real deny-list rule.
    let (_d, db) = db();
    let pid = mk_post(&db, "t", true, "owner");
    let r = rules(
        "match Post { allow read: if !(request.auth.uid in resource.editors.uid); }",
        &db,
    );
    let getq = format!("Post.get({pid})");
    // Small editor set (≤ cap = determinate): a non-editor reads; an editor is denied.
    let ed1 = mk_user(&db, "ed1");
    db.link("Post", pid, "editors", ed1, None).unwrap();
    assert_eq!(
        n_objects(run(&ctx_with(&db, &r, user("stranger")), &getq)),
        1
    );
    assert_eq!(n_objects(run(&ctx_with(&db, &r, user("ed1")), &getq)), 0);
    // Grow the editor set past the enumeration cap (MAX_RULE_RELATION_TARGETS = 1024): the
    // relation is now indeterminate, so the deny-list denies EVERYONE (fail-closed) rather than
    // truncating and leaking to a would-be editor past the cap.
    for i in 0..1100 {
        let u = mk_user(&db, &format!("bulk{i}"));
        db.link("Post", pid, "editors", u, None).unwrap();
    }
    assert_eq!(
        n_objects(run(&ctx_with(&db, &r, user("stranger")), &getq)),
        0,
        "an un-enumerable deny-list must deny (fail closed), not truncate"
    );
}

#[test]
fn delete_gate() {
    let (_d, db) = db();
    let pid = mk_post(&db, "t", false, "u1");
    let r = rules(
        "match Post { allow delete: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    // Non-owner denied.
    let err = run(
        &ctx_with(&db, &r, user("u2")),
        &format!("Post.get({pid}).delete()"),
    )
    .unwrap_err();
    assert!(matches!(err, QueryError::PermissionDenied(_)));
    // Owner allowed.
    assert!(
        run(
            &ctx_with(&db, &r, user("u1")),
            &format!("Post.get({pid}).delete()")
        )
        .is_ok()
    );
}

#[test]
fn default_deny_unlisted_ops() {
    let (_d, db) = db();
    let pid = mk_post(&db, "t", true, "u1");
    // Only a read clause → create/update/delete are default-denied even for an authed owner.
    let r = rules("match Post { allow read: if true; }", &db);
    let c = ctx_with(&db, &r, user("u1"));
    assert!(matches!(
        run(
            &c,
            r#"Post.create({ title: "y", published: true, ownerUid: "u1" })"#
        )
        .unwrap_err(),
        QueryError::PermissionDenied(_)
    ));
    assert!(matches!(
        run(&c, &format!("Post.get({pid}).update({{ title: \"z\" }})")).unwrap_err(),
        QueryError::PermissionDenied(_)
    ));
    assert!(matches!(
        run(&c, &format!("Post.get({pid}).delete()")).unwrap_err(),
        QueryError::PermissionDenied(_)
    ));
    // read still works.
    assert_eq!(n_objects(run(&c, "Post")), 1);
}

#[test]
fn event_read_allowed_gates_subscription() {
    let (_d, db) = db();
    let r = rules(
        "match Post { allow read: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    // Authorizes against the event's own SNAPSHOT (the delivered data) — not a live re-fetch —
    // so authorize-and-deliver stay consistent (the TOCTOU-leak fix).
    let snap = std::collections::HashMap::from([("ownerUid".to_string(), serde_json::json!("u1"))]);
    assert!(super::event_read_allowed(
        &r,
        &user("u1"),
        "Post",
        Some(&snap)
    ));
    assert!(!super::event_read_allowed(
        &r,
        &user("u2"),
        "Post",
        Some(&snap)
    ));
    assert!(!super::event_read_allowed(
        &r,
        &Principal::anonymous(),
        "Post",
        Some(&snap)
    ));
    // No snapshot → fail-closed (can't confirm the principal could read it).
    assert!(!super::event_read_allowed(&r, &user("u1"), "Post", None));
    // A relationship-based read rule can't be satisfied by a scalar snapshot → fail-closed.
    let rel = rules(
        "match Post { allow read: if request.auth.uid == resource.author.uid; }",
        &db,
    );
    assert!(!super::event_read_allowed(
        &rel,
        &user("u1"),
        "Post",
        Some(&snap)
    ));
}

// ---- Per-stage read filtering + ranked search under rules (FTS review) ----

const FT_SCHEMA: &str = r#"
    type User {
        uid: String
        name: String
    }
    type Doc {
        body: String @fulltext
        ownerUid: String @indexed
        author: User
    }
"#;

/// Owner-only read + delete on Doc; Users are world-readable.
const FT_RULES: &str = "match Doc { allow read, delete: if request.auth.uid == resource.ownerUid; } \
                        match User { allow read: if true; }";

fn ft_db() -> (tempfile::TempDir, Arc<Database>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(parse_schema(FT_SCHEMA).unwrap(), dir.path()).unwrap();
    (dir, db)
}

fn mk_doc(db: &Database, body: &str, owner: &str) -> u64 {
    let mut f = FieldMap::new();
    f.insert("body".into(), Value::String(body.into()));
    f.insert("ownerUid".into(), Value::String(owner.into()));
    db.create("Doc", f).unwrap().id
}

fn row_ids(out: QueryResult<QueryOutput>) -> Vec<u64> {
    match out.unwrap() {
        QueryOutput::Objects(o) => o.into_iter().map(|o| o.id).collect(),
        QueryOutput::Scored(r) => r.into_iter().map(|(o, _)| o.id).collect(),
        QueryOutput::Single(o) => vec![o.id],
        _ => vec![],
    }
}

fn scores(out: QueryResult<QueryOutput>) -> Vec<(u64, f32)> {
    match out.unwrap() {
        QueryOutput::Scored(r) => r.into_iter().map(|(o, s)| (o.id, s.score)).collect(),
        other => panic!("expected a ranked result, got {other:?}"),
    }
}

#[test]
fn write_gate_is_not_an_oracle_for_unreadable_text() {
    let (_d, db) = ft_db();
    let secret = mk_doc(&db, "my api key is sk_live_ABC123", "u1");
    let r = rules(FT_RULES, &db);
    let attacker = ctx_with(&db, &r, user("u2"));
    // Before: `….delete()` was PermissionDenied exactly when a hidden row matched the guess.
    // Now the hidden row never reaches the write gate — right and wrong guesses both succeed
    // as no-ops, and nothing is deleted.
    for q in [
        r#"Doc.matches(.body, "sk_live_abc*", k: 10).delete()"#,
        r#"Doc.matches(.body, "sk_live_abd*", k: 10).delete()"#,
        r#"Doc.filter(.body.contains("sk_live_ABC1")).delete()"#,
        r#"Doc.filter(.body.contains("sk_live_ABD")).delete()"#,
        r#"Doc.filter(.ownerUid == "u1").limit(5).delete()"#,
    ] {
        assert!(run(&attacker, q).is_ok(), "{q}");
    }
    let get_filter =
        format!(r#"Doc.get({secret}).filter(.body >= "my api key is sk_live_A").delete()"#);
    assert!(run(&attacker, &get_filter).is_ok());
    assert!(db.get("Doc", secret).is_ok(), "nothing was deleted");
    // A point delete of a NAMED object is still gated by the delete rule itself (delete need
    // not imply read — Firestore semantics).
    assert!(matches!(
        run(&attacker, &format!("Doc.get({secret}).delete()")),
        Err(QueryError::PermissionDenied(_))
    ));
    // The owner can search-and-delete their own row.
    let owner = ctx_with(&db, &r, user("u1"));
    assert!(
        run(
            &owner,
            r#"Doc.matches(.body, "sk_live_abc*", k: 10).delete()"#
        )
        .is_ok()
    );
    assert!(db.get("Doc", secret).is_err());
}

#[test]
fn traversal_from_a_hidden_match_reveals_nothing() {
    let (_d, db) = ft_db();
    let author = mk_user(&db, "ceo");
    let memo = mk_doc(&db, "merger with acme next week", "u1");
    db.link("Doc", memo, "author", author, None).unwrap();
    let r = rules(FT_RULES, &db);
    let attacker = ctx_with(&db, &r, user("u2"));
    for q in [
        r#"Doc.matches(.body, "+merger +acme", k: 10).author"#,
        r#"Doc.matches(.body, "+merger +globex", k: 10).author"#,
        r#"Doc.filter(.body.contains("acme")).author"#,
        r#"Doc.filter(.ownerUid == "u1").author.limit(1)"#,
    ] {
        assert_eq!(row_ids(run(&attacker, q)), Vec::<u64>::new(), "{q}");
    }
    let owner = ctx_with(&db, &r, user("u1"));
    assert_eq!(
        row_ids(run(
            &owner,
            r#"Doc.matches(.body, "+merger +acme", k: 10).author"#
        )),
        vec![author]
    );
}

#[test]
fn hidden_rows_neither_crowd_nor_skew_what_a_caller_can_read() {
    let clean_dir = tempfile::tempdir().unwrap();
    let clean = Database::open(parse_schema(FT_SCHEMA).unwrap(), clean_dir.path()).unwrap();
    let (_nd, noisy) = ft_db();
    // The attacker's own row, identical in both databases.
    let mine = mk_doc(&noisy, "zebra crossing invoice", "u2");
    assert_eq!(mk_doc(&clean, "zebra crossing invoice", "u2"), mine);
    // Hidden rows full of the query terms, only in `noisy`.
    for _ in 0..5 {
        mk_doc(&noisy, "invoice invoice invoice zebra", "u1");
    }
    for i in 0..rhypedb_engine::fulltext::MAX_PREFIX_EXPANSION {
        mk_doc(&noisy, &format!("zq{i:03}"), "u1");
    }
    let r_noisy = rules(FT_RULES, &noisy);
    let r_clean = rules(FT_RULES, &clean);
    let (n, c) = (
        ctx_with(&noisy, &r_noisy, user("u2")),
        ctx_with(&clean, &r_clean, user("u2")),
    );
    // `k` is cut after the read rule: the hidden, higher-scoring rows don't crowd `k: 1`.
    assert_eq!(
        row_ids(run(&n, r#"Doc.matches(.body, "invoice", k: 1)"#)),
        vec![mine]
    );
    // The score is computed from readable documents only (no df/doc_count oracle).
    for q in [
        r#"Doc.matches(.body, "zebra", k: 3)"#,
        r#"Doc.matches(.body, "invoice zebra crossing", k: 3)"#,
    ] {
        assert_eq!(scores(run(&n, q)), scores(run(&c, q)), "{q}");
    }
    // The prefix cap counts terms in readable documents only.
    assert!(run(&n, r#"Doc.matches(.body, "zq*", k: 3)"#).is_ok());
    // Limits after a filter, pushed-down or not, page over readable rows only.
    assert_eq!(
        row_ids(run(&n, r#"Doc.filter(.body.contains("invoice")).limit(1)"#)),
        vec![mine]
    );
    assert_eq!(
        row_ids(run(&n, r#"Doc.filter(.ownerUid != "nobody").limit(1)"#)),
        vec![mine]
    );
    assert_eq!(row_ids(run(&n, "Doc.limit(1)")), vec![mine]);
    assert_eq!(row_ids(run(&n, "Doc.offset(0).limit(1)")), vec![mine]);
    // Rules off, the same query sees every row (the path is unchanged).
    let plain = ExecContext::new(&noisy, None);
    assert_eq!(
        row_ids(run(&plain, r#"Doc.matches(.body, "invoice", k: 10)"#)).len(),
        6
    );
}

/// A governed context for `principal` with `rules`.
fn gov_ctx_with<'a>(
    db: &'a Database,
    r: &Arc<RulesProgram>,
    principal: Principal,
    max_rows_scanned: usize,
) -> ExecContext<'a> {
    let mut c = ctx_with(db, r, principal);
    c.governor = crate::governor::Governor::new(
        crate::governor::GovernorLimits {
            max_rows_scanned,
            ..crate::governor::GovernorLimits::UNLIMITED
        },
        rhypedb_engine::time::Instant::now(),
    );
    c
}

#[test]
fn deleting_and_shrinking_large_fulltext_documents_is_charged() {
    let (_d, db) = ft_db();
    // 200 indexed words per doc, written with the governor off (as a trusted import).
    let body = (0..200)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let ids: Vec<u64> = (0..3).map(|_| mk_doc(&db, &body, "u1")).collect();
    let r = rules(
        "match Doc { allow read, delete, update: if request.auth.uid == resource.ownerUid; }",
        &db,
    );
    let owner = || user("u1");
    // Deleting them tombstones 600 postings: refused under a budget that only counts rows…
    let err = run(
        &gov_ctx_with(&db, &r, owner(), 100),
        r#"Doc.filter(.ownerUid == "u1").delete()"#,
    )
    .unwrap_err();
    assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "{err}");
    // …and the rows are still there (charged BEFORE any write).
    assert_eq!(row_ids(run(&ctx_with(&db, &r, owner()), "Doc")).len(), 3);
    // Shrinking a @fulltext field is charged the stored value it tombstones.
    let err = run(
        &gov_ctx_with(&db, &r, owner(), 100),
        &format!(r#"Doc.get({}).update({{ body: "z" }})"#, ids[0]),
    )
    .unwrap_err();
    assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "{err}");
    // A budget that covers the work succeeds.
    assert!(
        run(
            &gov_ctx_with(&db, &r, owner(), 100_000),
            r#"Doc.filter(.ownerUid == "u1").delete()"#
        )
        .is_ok()
    );
    assert_eq!(row_ids(run(&ctx_with(&db, &r, owner()), "Doc")).len(), 0);
}

#[test]
fn per_stage_read_filtering_is_charged_and_not_repeated_needlessly() {
    let (_d, db) = ft_db();
    for _ in 0..50 {
        mk_doc(&db, "text", "u1");
    }
    let r = rules(FT_RULES, &db);
    // Checking the rule on a row is per-row work, and it is charged: the 50-row listing
    // costs its scan AND its read check, so a 60-row budget can't cover both.
    let err = run(&gov_ctx_with(&db, &r, user("u1"), 60), "Doc").unwrap_err();
    assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "{err}");
    assert_eq!(
        row_ids(run(&gov_ctx_with(&db, &r, user("u1"), 250), "Doc")).len(),
        50
    );
    // `limit`/`offset`/`filter` only drop rows, so they never pay for a SECOND check —
    // 40 of them cost the same as none (before, each stage re-checked every row).
    let many = format!("Doc{}", ".limit(50)".repeat(40));
    assert_eq!(
        row_ids(run(&gov_ctx_with(&db, &r, user("u1"), 250), &many)).len(),
        50
    );
    // A traversal DOES introduce rows, so its output is checked (and charged) again.
    let author = mk_user(&db, "u1");
    let d = mk_doc(&db, "linked", "u1");
    db.link("Doc", d, "author", author, None).unwrap();
    let hop = run(
        &gov_ctx_with(&db, &r, user("u1"), 250),
        &format!("Doc.get({d}).author"),
    );
    assert_eq!(row_ids(hop), vec![author]);
}

#[test]
fn a_budget_exhausted_inside_the_read_filter_is_reported_not_silently_empty() {
    // A rule that traverses charges the edges it walks; when that charge fails the rule
    // evaluator treats the relation as indeterminate (deny). The query must then report the
    // limit rather than return a quietly shortened result.
    let (_d, db) = ft_db();
    let author = mk_user(&db, "u1");
    for _ in 0..40 {
        let d = mk_doc(&db, "text", "u1");
        db.link("Doc", d, "author", author, None).unwrap();
    }
    let r = rules(
        "match Doc { allow read: if request.auth.uid == resource.author.uid; } match User { allow read: if true; }",
        &db,
    );
    let err = run(&gov_ctx_with(&db, &r, user("u1"), 45), "Doc").unwrap_err();
    assert!(matches!(err, QueryError::ResourceLimitExceeded(_)), "{err}");
}

#[test]
fn building_index_error_hides_progress_counts_under_rules() {
    let dir = tempfile::tempdir().unwrap();
    {
        let plain = parse_schema(&FT_SCHEMA.replace("@fulltext", "")).unwrap();
        let db = Database::open(plain, dir.path()).unwrap();
        for i in 0..20 {
            mk_doc(&db, &format!("text {i}"), "u1");
        }
    }
    let db = Database::open_with_options(
        parse_schema(FT_SCHEMA).unwrap(),
        dir.path(),
        rhypedb_engine::database::OpenOptions {
            background_fulltext_build: false,
            ..Default::default()
        },
    )
    .unwrap();
    let q = r#"Doc.matches(.body, "text", k: 5)"#;
    let r = rules(FT_RULES, &db);
    let err = run(&ctx_with(&db, &r, user("u2")), q)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("still building") && !err.contains("objects indexed"),
        "{err}"
    );
    // Rules off (a trusted caller) still gets the progress.
    let err = run(&ExecContext::new(&db, None), q)
        .unwrap_err()
        .to_string();
    assert!(err.contains("objects indexed"), "{err}");
}

#[test]
fn rules_off_is_a_total_noop() {
    // The backward-compat guarantee: rules = None ⇒ the engine behaves exactly as pre-P4, even
    // for an anonymous principal — every op is allowed and no row is filtered.
    let (_d, db) = db();
    mk_post(&db, "a", false, "u1");
    mk_post(&db, "b", false, "u2");
    let c = ExecContext::new(&db, None); // anonymous, rules-off
    assert_eq!(n_objects(run(&c, "Post")), 2); // nothing filtered
    assert!(
        run(
            &c,
            r#"Post.create({ title: "c", published: false, ownerUid: "z" })"#
        )
        .is_ok()
    );
    assert_eq!(n_objects(run(&c, "Post")), 3);
    // delete works with no principal.
    let id = mk_post(&db, "d", false, "u3");
    assert!(run(&c, &format!("Post.get({id}).delete()")).is_ok());
}
