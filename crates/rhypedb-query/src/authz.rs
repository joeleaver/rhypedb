//! P4 data-plane authorization: the bridge between query execution and the `rhypedb-authz` rules
//! evaluator. Everything here is a **strict no-op when `ExecContext.rules` is `None`** — a rules-off
//! deployment (every v1/P0–P2 tenant today) takes the exact pre-P4 execution path, so nothing about
//! its behavior changes. Rules are opt-in; turning them on is what closes the doors.
//!
//! Three enforcement points, each fed a live [`EngineResource`] the rules read through:
//! - **create** — gate the incoming write fields before `db.create`/`create_batch` (§[`gate_create`]).
//! - **write** (update/delete/link/unlink) — gate the **pre-mutation** object before the engine
//!   mutates it, so a rule like `request.auth.uid == resource.author` can't be defeated by the same
//!   write that flips `author` (P0-DBA-6) (§[`gate_write`]).
//! - **read** — filter the terminal result set, dropping rows the principal may not read (Firestore
//!   semantics: you get the rows you may read, not an error) (§[`filter_read`]).
//!
//! A denied *write* is a loud [`QueryError::PermissionDenied`]; a denied *read* row is silently
//! dropped. Relationship projections a rule needs (`resource.owners.uid`) walk edges via the
//! engine, bounded by both a hard cap and the query [`Governor`] so a rule can't become a fan-out
//! DoS (P0-DBA-7).

use rhypedb_authz::{Decision, Op, ResourceAccessor, RulesProgram};
use rhypedb_engine::database::Database;
use rhypedb_engine::object::{value_to_query_json, FieldMap, Object};

use crate::error::{QueryError, QueryResult};
use crate::executor::{ExecContext, QueryOutput};
use crate::governor::Governor;

/// Hard cap on the linked targets one rule relationship-projection reads, so a predicate over a
/// huge to-many edge can't fan out unbounded. The governor *also* charges every edge/target, so a
/// rule additionally can't outrun the query's row budget.
const MAX_RULE_RELATION_TARGETS: usize = 1024;

/// Filter a terminal result through the `read` rule when rules are on (no-op otherwise). ANY object
/// surfaced to the caller is a read, so this covers reads AND a mutation's returned post-image: a
/// multi-row `Objects` result is filtered row-wise, and a `Single` (a create/update return value)
/// the principal may not read collapses to `Done` (void, Firestore-style) rather than leaking. This
/// is what stops an `update` return value from bypassing the read rule.
pub(crate) fn apply_read_filter(ctx: &ExecContext<'_>, result: QueryOutput) -> QueryOutput {
    let Some(rules) = &ctx.rules else {
        return result;
    };
    match result {
        QueryOutput::Objects(mut objs) => {
            filter_read(ctx, &mut objs);
            QueryOutput::Objects(objs)
        }
        QueryOutput::Single(mut obj) => {
            obj.ensure_fields_deserialized();
            let type_name = obj.type_name.clone();
            let res = EngineResource {
                db: ctx.db,
                governor: &ctx.governor,
                type_name: &type_name,
                object: Some(&obj),
                request: None,
            };
            if rules
                .evaluate(Op::Read, &type_name, &ctx.principal, &res)
                .is_allow()
            {
                QueryOutput::Single(obj)
            } else {
                QueryOutput::Done
            }
        }
        other => other,
    }
}

/// A [`ResourceAccessor`] backed by the live engine: a stored (pre-mutation) object and/or the
/// incoming write fields. Relationship projections resolve the target type from the schema and read
/// linked objects through `db.get_links_many` + `db.get`, charging the governor.
pub(crate) struct EngineResource<'a> {
    db: &'a Database,
    governor: &'a Governor,
    type_name: &'a str,
    /// The stored object (fields already deserialized). `None` for a `create` (no prior resource).
    object: Option<&'a Object>,
    /// The incoming write fields (`request.<field>`). `None` for read/delete/link/unlink.
    request: Option<&'a FieldMap>,
}

impl<'a> EngineResource<'a> {
    fn relation_target(&self, rel: &str) -> Option<String> {
        match &self.db.schema().get_type(self.type_name)?.get_field(rel)?.field_type {
            rhypedb_schema::FieldType::Relation(r) => Some(r.target_type.clone()),
            _ => None,
        }
    }

    /// Linked target ids for `rel` off the current object. `None` = INDETERMINATE (no object,
    /// traversal error, or over-budget) — the caller must fail this closed (Absent/Unknown), NOT
    /// treat it as an empty set. `Some(ids)` is the determinate id list (`Some(empty)` = genuinely
    /// unlinked). Charges the FULL examined edge set (like the query traversal's `hop_edges`), so a
    /// rule over a huge to-many edge trips the row budget instead of evading it.
    fn linked_ids(&self, rel: &str) -> Option<Vec<u64>> {
        let obj = self.object?;
        let groups = self.db.get_links_many(self.type_name, &[obj.id], rel).ok()?;
        let examined: u64 = groups.iter().map(|g| g.len() as u64).sum();
        if self.governor.charge(examined).is_err() {
            return None; // over budget ⇒ indeterminate ⇒ fail closed (NOT an empty set)
        }
        Some(
            groups
                .into_iter()
                .flatten()
                .map(|(id, _)| id)
                .take(MAX_RULE_RELATION_TARGETS)
                .collect(),
        )
    }

    fn target_scalar(&self, target_type: &str, id: u64, field: &str) -> Option<serde_json::Value> {
        let mut tobj = self.db.get(target_type, id).ok()?;
        tobj.ensure_fields_deserialized();
        tobj.fields.get(field).map(value_to_query_json)
    }
}

impl ResourceAccessor for EngineResource<'_> {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.object?.fields.get(field).map(value_to_query_json)
    }

    fn resource_relation_one(&self, rel: &str, target_field: &str) -> Option<serde_json::Value> {
        let target_type = self.relation_target(rel)?;
        // `None`/empty (indeterminate or unlinked) ⇒ None ⇒ Absent (fail closed).
        let id = *self.linked_ids(rel)?.first()?;
        self.target_scalar(&target_type, id, target_field)
    }

    fn resource_relation_many(&self, rel: &str, target_field: &str) -> Option<Vec<serde_json::Value>> {
        let target_type = self.relation_target(rel)?;
        // Propagate indeterminacy: `None` from linked_ids (over-budget) stays `None` ⇒ Absent, so a
        // deny-list rule fails closed rather than seeing a spurious empty set.
        let ids = self.linked_ids(rel)?;
        Some(
            ids.into_iter()
                .filter_map(|id| self.target_scalar(&target_type, id, target_field))
                .collect(),
        )
    }

    fn request_field(&self, field: &str) -> Option<serde_json::Value> {
        self.request?.get(field).map(value_to_query_json)
    }
}

/// Gate a `create` against the incoming fields. No-op if rules are off. `db.create` runs only if the
/// `create` rule allows.
pub(crate) fn gate_create(
    ctx: &ExecContext<'_>,
    type_name: &str,
    fields: &FieldMap,
) -> QueryResult<()> {
    let Some(rules) = &ctx.rules else { return Ok(()) };
    let res = EngineResource {
        db: ctx.db,
        governor: &ctx.governor,
        type_name,
        object: None,
        request: Some(fields),
    };
    decide(rules, Op::Create, type_name, ctx, &res)
}

/// Gate a write (update/delete/link/unlink) over `ids`, checking each **pre-mutation** object. No-op
/// if rules are off. An id that no longer exists is skipped (the engine call handles non-existence
/// with its normal semantics). Denied ⇒ [`QueryError::PermissionDenied`], before any mutation.
pub(crate) fn gate_write(
    ctx: &ExecContext<'_>,
    op: Op,
    type_name: &str,
    ids: &[u64],
    request: Option<&FieldMap>,
) -> QueryResult<()> {
    let Some(rules) = &ctx.rules else { return Ok(()) };
    for &id in ids {
        // Fetch the object as it exists BEFORE the write (P0-DBA-6). A missing object is left to the
        // engine call's own behavior (no object ⇒ nothing to authorize here).
        let Ok(mut obj) = ctx.db.get(type_name, id) else {
            continue;
        };
        obj.ensure_fields_deserialized();
        let res = EngineResource {
            db: ctx.db,
            governor: &ctx.governor,
            type_name,
            object: Some(&obj),
            request,
        };
        decide(rules, op, type_name, ctx, &res)?;
    }
    Ok(())
}

/// Filter a terminal read result in place, dropping every object the principal may not read. No-op
/// if rules are off.
pub(crate) fn filter_read(ctx: &ExecContext<'_>, objs: &mut Vec<Object>) {
    let Some(rules) = &ctx.rules else { return };
    objs.retain_mut(|obj| {
        obj.ensure_fields_deserialized();
        let type_name = obj.type_name.clone();
        let res = EngineResource {
            db: ctx.db,
            governor: &ctx.governor,
            type_name: &type_name,
            object: Some(&*obj),
            request: None,
        };
        rules
            .evaluate(Op::Read, &type_name, &ctx.principal, &res)
            .is_allow()
    });
}

/// A [`ResourceAccessor`] over a change event's JSON field snapshot — no db, so relationship
/// projections are unavailable (a rule that traverses evaluates them as absent ⇒ its clause is
/// false). Used for the subscribe per-event filter when the live object is gone (a delete).
struct JsonSnapshot<'a> {
    fields: Option<&'a std::collections::HashMap<String, serde_json::Value>>,
}

impl ResourceAccessor for JsonSnapshot<'_> {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.fields?.get(field).cloned()
    }
    fn resource_relation_one(&self, _rel: &str, _f: &str) -> Option<serde_json::Value> {
        None
    }
    fn resource_relation_many(&self, _rel: &str, _f: &str) -> Option<Vec<serde_json::Value>> {
        None // relationships unavailable in a snapshot ⇒ Absent ⇒ deny (fail closed)
    }
    fn request_field(&self, _f: &str) -> Option<serde_json::Value> {
        None
    }
}

/// Subscribe per-event read gate (P0-DBA-5): may `principal` **read** the object this change event
/// is about? A subscription is a standing read, so an event is delivered only if the subscriber
/// passes the `read` rule (otherwise a sub is a read-rule bypass).
///
/// The rule is evaluated against the event's OWN field `snapshot` — i.e. exactly the data that will
/// be delivered — NOT a live re-fetch of the current object. This is load-bearing (a review finding):
/// re-fetching authorizes the *current* state while delivering the *commit-time* state, so an object
/// that transitions unreadable→readable across two writes would leak the earlier, private-era
/// snapshot. Evaluating against the delivered snapshot makes authorize-and-deliver self-consistent.
///
/// The snapshot carries the object's SCALAR fields only, so a **relationship-based** read rule
/// (`resource.owners.uid`) resolves its relation operands to absent ⇒ the clause is `Unknown` ⇒
/// **the event is denied** (fail-closed). v1 limitation: subscription-visible read rules should key
/// on scalar fields (denormalize an owner id) rather than edge traversals. Callers invoke this only
/// when a rules program is configured.
pub fn event_read_allowed(
    rules: &RulesProgram,
    principal: &rhypedb_authz::Principal,
    type_name: &str,
    snapshot: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> bool {
    let res = JsonSnapshot { fields: snapshot };
    rules.evaluate(Op::Read, type_name, principal, &res).is_allow()
}

fn decide(
    rules: &RulesProgram,
    op: Op,
    type_name: &str,
    ctx: &ExecContext<'_>,
    res: &EngineResource<'_>,
) -> QueryResult<()> {
    match rules.evaluate(op, type_name, &ctx.principal, res) {
        Decision::Allow => Ok(()),
        Decision::Deny => Err(QueryError::PermissionDenied(format!(
            "{} on {type_name} denied by security rules",
            op.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{execute, QueryOutput};
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
    fn ctx_with<'a>(db: &'a Database, rules: &Arc<RulesProgram>, principal: Principal) -> ExecContext<'a> {
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
        assert_eq!(n_objects(run(&ctx_with(&db, &r, Principal::anonymous()), "Post")), 1);
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
        db.link("Post", other, "author", other_author, None).unwrap();
        let r = rules(
            "match Post { allow read: if request.auth.uid == resource.author.uid; }",
            &db,
        );
        // u1 is the author of exactly one post → sees one.
        assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u1")), "Post")), 1);
        assert_eq!(n_objects(run(&ctx_with(&db, &r, user("u9")), "Post")), 1);
        assert_eq!(n_objects(run(&ctx_with(&db, &r, user("nobody")), "Post")), 0);
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
        assert!(run(&ctx_with(&db, &r, user("u1")), &format!("Post.get({pid}).update({{ title: \"new\" }})")).is_ok());
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
            assert_eq!(obj.fields.get("ownerUid"), Some(&Value::String("u1".into())));
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
        let out = run(&ctx_with(&db, &r, user("u2")), &format!("Post.get({pid}).update({{ title: \"x\" }})"));
        assert_eq!(n_objects(out), 0, "non-owner's update return is read-filtered to void");
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
        let out = run(&ctx_with(&db, &r, user("u1")), &format!("Post.get({pid}).update({{ title: \"y\" }})"));
        assert_eq!(n_objects(out), 1, "owner sees their update return value");
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
        let err = run(&ctx_with(&db, &r, user("u2")), &format!("Post.get({pid}).delete()")).unwrap_err();
        assert!(matches!(err, QueryError::PermissionDenied(_)));
        // Owner allowed.
        assert!(run(&ctx_with(&db, &r, user("u1")), &format!("Post.get({pid}).delete()")).is_ok());
    }

    #[test]
    fn default_deny_unlisted_ops() {
        let (_d, db) = db();
        let pid = mk_post(&db, "t", true, "u1");
        // Only a read clause → create/update/delete are default-denied even for an authed owner.
        let r = rules("match Post { allow read: if true; }", &db);
        let c = ctx_with(&db, &r, user("u1"));
        assert!(matches!(
            run(&c, r#"Post.create({ title: "y", published: true, ownerUid: "u1" })"#).unwrap_err(),
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
        assert!(super::event_read_allowed(&r, &user("u1"), "Post", Some(&snap)));
        assert!(!super::event_read_allowed(&r, &user("u2"), "Post", Some(&snap)));
        assert!(!super::event_read_allowed(&r, &Principal::anonymous(), "Post", Some(&snap)));
        // No snapshot → fail-closed (can't confirm the principal could read it).
        assert!(!super::event_read_allowed(&r, &user("u1"), "Post", None));
        // A relationship-based read rule can't be satisfied by a scalar snapshot → fail-closed.
        let rel = rules(
            "match Post { allow read: if request.auth.uid == resource.author.uid; }",
            &db,
        );
        assert!(!super::event_read_allowed(&rel, &user("u1"), "Post", Some(&snap)));
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
        assert!(run(&c, r#"Post.create({ title: "c", published: false, ownerUid: "z" })"#).is_ok());
        assert_eq!(n_objects(run(&c, "Post")), 3);
        // delete works with no principal.
        let id = mk_post(&db, "d", false, "u3");
        assert!(run(&c, &format!("Post.get({id}).delete()")).is_ok());
    }
}
