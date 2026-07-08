use super::{Decision, Op, ResourceAccessor, RulesProgram};
use crate::Principal;
use rhypedb_schema::parser::parse_schema;
use std::collections::HashMap;

const SCHEMA: &str = r#"
    type User {
        uid: String
        role: String
    }
    type Org {
        name: String
        members: [User]
    }
    type Post {
        title: String
        published: Bool
        rank: i64
        ownerUid: String
        tags: Json
        author: User
        editors: [User]
    }
"#;

fn schema() -> rhypedb_schema::Schema {
    parse_schema(SCHEMA).expect("schema parses")
}

fn compile(src: &str) -> RulesProgram {
    RulesProgram::compile(src, &schema()).expect("rules compile")
}

/// An in-memory accessor; keys for relations are "rel.field".
#[derive(Default)]
struct Mock {
    scalar: HashMap<String, serde_json::Value>,
    rel_one: HashMap<String, serde_json::Value>,
    rel_many: HashMap<String, Vec<serde_json::Value>>,
    request: HashMap<String, serde_json::Value>,
}

impl Mock {
    fn scalar(mut self, f: &str, v: serde_json::Value) -> Self {
        self.scalar.insert(f.into(), v);
        self
    }
    fn rel_one(mut self, rel: &str, f: &str, v: serde_json::Value) -> Self {
        self.rel_one.insert(format!("{rel}.{f}"), v);
        self
    }
    fn rel_many(mut self, rel: &str, f: &str, v: Vec<serde_json::Value>) -> Self {
        self.rel_many.insert(format!("{rel}.{f}"), v);
        self
    }
    fn request(mut self, f: &str, v: serde_json::Value) -> Self {
        self.request.insert(f.into(), v);
        self
    }
}

impl ResourceAccessor for Mock {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.scalar.get(field).cloned()
    }
    fn resource_relation_one(&self, rel: &str, target_field: &str) -> Option<serde_json::Value> {
        self.rel_one.get(&format!("{rel}.{target_field}")).cloned()
    }
    fn resource_relation_many(&self, rel: &str, target_field: &str) -> Vec<serde_json::Value> {
        self.rel_many
            .get(&format!("{rel}.{target_field}"))
            .cloned()
            .unwrap_or_default()
    }
    fn request_field(&self, field: &str) -> Option<serde_json::Value> {
        self.request.get(field).cloned()
    }
}

fn anon() -> Principal {
    Principal::anonymous()
}
fn user(uid: &str) -> Principal {
    Principal::new_authenticated(uid, serde_json::Value::Null)
}
fn user_claims(uid: &str, claims: serde_json::Value) -> Principal {
    Principal::new_authenticated(uid, claims)
}

fn eval(prog: &RulesProgram, op: Op, ty: &str, p: &Principal, r: &dyn ResourceAccessor) -> Decision {
    prog.evaluate(op, ty, p, r)
}

#[test]
fn default_deny_unknown_type_and_missing_op() {
    let prog = compile("match Post { allow read: if true; }");
    // Type with no match block → deny everything.
    assert_eq!(eval(&prog, Op::Read, "User", &user("u"), &Mock::default()), Decision::Deny);
    // Type with a match block but no clause for the op → deny.
    assert_eq!(eval(&prog, Op::Delete, "Post", &user("u"), &Mock::default()), Decision::Deny);
    // The one open op → allow.
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u"), &Mock::default()), Decision::Allow);
}

#[test]
fn read_public_or_owner() {
    let prog = compile(
        "match Post { allow read: if resource.published || request.auth.uid == resource.author.uid; }",
    );
    // Public post → anyone, incl. anonymous.
    let pub_post = Mock::default().scalar("published", serde_json::json!(true));
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &pub_post), Decision::Allow);
    // Private post owned by u1.
    let priv_post = Mock::default()
        .scalar("published", serde_json::json!(false))
        .rel_one("author", "uid", serde_json::json!("u1"));
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u1"), &priv_post), Decision::Allow);
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u2"), &priv_post), Decision::Deny);
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &priv_post), Decision::Deny);
}

#[test]
fn create_requires_auth_and_self_owner() {
    let prog = compile(
        "match Post { allow create: if request.auth != null && request.ownerUid == request.auth.uid; }",
    );
    // Authenticated writer setting ownerUid to their own uid → allow.
    let good = Mock::default().request("ownerUid", serde_json::json!("u1"));
    assert_eq!(eval(&prog, Op::Create, "Post", &user("u1"), &good), Decision::Allow);
    // Setting someone else's uid → deny.
    assert_eq!(eval(&prog, Op::Create, "Post", &user("u2"), &good), Decision::Deny);
    // Anonymous → deny (short-circuits on request.auth != null).
    assert_eq!(eval(&prog, Op::Create, "Post", &anon(), &good), Decision::Deny);
}

#[test]
fn update_checks_pre_mutation_owner() {
    let prog = compile("match Post { allow update: if request.auth.uid == resource.author.uid; }");
    // The stored (pre-mutation) object is owned by u1.
    let stored = Mock::default().rel_one("author", "uid", serde_json::json!("u1"));
    assert_eq!(eval(&prog, Op::Update, "Post", &user("u1"), &stored), Decision::Allow);
    assert_eq!(eval(&prog, Op::Update, "Post", &user("u2"), &stored), Decision::Deny);
}

#[test]
fn native_edge_membership() {
    let prog = compile("match Org { allow read: if request.auth.uid in resource.members.uid; }");
    let org = Mock::default().rel_many(
        "members",
        "uid",
        vec![serde_json::json!("a"), serde_json::json!("b")],
    );
    assert_eq!(eval(&prog, Op::Read, "Org", &user("a"), &org), Decision::Allow);
    assert_eq!(eval(&prog, Op::Read, "Org", &user("c"), &org), Decision::Deny);
    assert_eq!(eval(&prog, Op::Read, "Org", &anon(), &org), Decision::Deny);
    // No members linked → deny (empty set).
    assert_eq!(eval(&prog, Op::Read, "Org", &user("a"), &Mock::default()), Decision::Deny);
}

#[test]
fn json_array_membership() {
    let prog = compile(r#"match Post { allow read: if "vip" in resource.tags; }"#);
    let tagged = Mock::default().scalar("tags", serde_json::json!(["vip", "x"]));
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &tagged), Decision::Allow);
    let untagged = Mock::default().scalar("tags", serde_json::json!(["x"]));
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &untagged), Decision::Deny);
    // Field absent → not a set → deny (total).
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &Mock::default()), Decision::Deny);
}

#[test]
fn custom_claim_gate() {
    let prog = compile(r#"match Post { allow delete: if request.auth.claims.role == "admin"; }"#);
    let admin = user_claims("u", serde_json::json!({ "role": "admin" }));
    let plain = user_claims("u", serde_json::json!({ "role": "user" }));
    assert_eq!(eval(&prog, Op::Delete, "Post", &admin, &Mock::default()), Decision::Allow);
    assert_eq!(eval(&prog, Op::Delete, "Post", &plain, &Mock::default()), Decision::Deny);
    assert_eq!(eval(&prog, Op::Delete, "Post", &anon(), &Mock::default()), Decision::Deny);
}

#[test]
fn numeric_ordering() {
    let prog = compile("match Post { allow read: if resource.rank >= 10; }");
    assert_eq!(
        eval(&prog, Op::Read, "Post", &anon(), &Mock::default().scalar("rank", serde_json::json!(15))),
        Decision::Allow
    );
    assert_eq!(
        eval(&prog, Op::Read, "Post", &anon(), &Mock::default().scalar("rank", serde_json::json!(5))),
        Decision::Deny
    );
    // Missing rank → deny (total, no panic).
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &Mock::default()), Decision::Deny);
}

#[test]
fn write_expands_to_create_update_delete() {
    let prog = compile("match Post { allow write: if request.auth != null; }");
    for op in [Op::Create, Op::Update, Op::Delete] {
        assert_eq!(eval(&prog, op, "Post", &user("u"), &Mock::default()), Decision::Allow);
        assert_eq!(eval(&prog, op, "Post", &anon(), &Mock::default()), Decision::Deny);
    }
    // read got no clause → deny.
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u"), &Mock::default()), Decision::Deny);
}

#[test]
fn bare_bool_field_is_truthy() {
    let prog = compile("match Post { allow read: if resource.published; }");
    assert_eq!(
        eval(&prog, Op::Read, "Post", &anon(), &Mock::default().scalar("published", serde_json::json!(true))),
        Decision::Allow
    );
    assert_eq!(
        eval(&prog, Op::Read, "Post", &anon(), &Mock::default().scalar("published", serde_json::json!(false))),
        Decision::Deny
    );
}

#[test]
fn subscribe_is_a_distinct_op() {
    let prog = compile(
        "match Post { allow read: if true;  allow subscribe: if request.auth != null; }",
    );
    // read open to anon, subscribe requires auth — they don't leak into each other.
    assert_eq!(eval(&prog, Op::Read, "Post", &anon(), &Mock::default()), Decision::Allow);
    assert_eq!(eval(&prog, Op::Subscribe, "Post", &anon(), &Mock::default()), Decision::Deny);
    assert_eq!(eval(&prog, Op::Subscribe, "Post", &user("u"), &Mock::default()), Decision::Allow);
}

#[test]
fn has_clause_reflects_program() {
    let prog = compile("match Post { allow read, update: if true; }");
    assert!(prog.has_clause(Op::Read, "Post"));
    assert!(prog.has_clause(Op::Update, "Post"));
    assert!(!prog.has_clause(Op::Delete, "Post"));
    assert!(!prog.has_clause(Op::Read, "User"));
}

#[test]
fn comments_and_multiblock_merge() {
    let prog = compile(
        r#"
        # first block
        match Post { allow read: if resource.published; }
        // second block for the same type — clauses merge (any-true ⇒ allow)
        match Post { allow read: if request.auth.uid == resource.author.uid; }
        "#,
    );
    let priv_owned = Mock::default()
        .scalar("published", serde_json::json!(false))
        .rel_one("author", "uid", serde_json::json!("u1"));
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u1"), &priv_owned), Decision::Allow);
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u2"), &priv_owned), Decision::Deny);
}

#[test]
fn compile_fails_closed_on_bad_programs() {
    let s = schema();
    let bad = [
        "match Ghost { allow read: if true; }",              // unknown type
        "match Post { allow read: if resource.nope == 1; }", // unknown field
        "match Post { allow read: if resource.author == 1; }", // relation without projection
        "match Post { allow frobnicate: if true; }",         // unknown op
        "match Post { allow read: if request.auth == request.auth.uid; }", // auth vs non-null
        "match Post { allow read: if request.auth.uid == resource.author.ghost; }", // bad target field
        "match Post { allow read: if resource.title }",      // missing semicolon
        "match Post { allow read if true; }",                // missing colon
    ];
    for src in bad {
        assert!(
            RulesProgram::compile(src, &s).is_err(),
            "expected compile error for: {src}"
        );
    }
}

#[test]
fn empty_program_denies_all() {
    let prog = compile("");
    assert_eq!(eval(&prog, Op::Read, "Post", &user("u"), &Mock::default()), Decision::Deny);
    assert_eq!(eval(&prog, Op::Create, "Post", &user("u"), &Mock::default()), Decision::Deny);
}
