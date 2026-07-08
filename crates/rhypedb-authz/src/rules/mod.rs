//! Default-deny, relationship-aware security rules (managed-DB P4, DECISION 1).
//!
//! A project ships an optional `rules.rhype` with a Firestore-familiar grammar that a Firebase dev
//! reads at a glance, but whose expression language traverses RhypeDB **relationships** natively
//! (a cheap pointer-follow, not a `get()` join):
//!
//! ```text
//! match Post {
//!   allow read:            if resource.published || request.auth.uid == resource.author;
//!   allow create:          if request.auth != null && request.author == request.auth.uid;
//!   allow update, delete:  if request.auth.uid == resource.author;      // pre-mutation object
//!   allow subscribe:       if request.auth != null;
//! }
//! match Org {
//!   allow read: if request.auth.uid in resource.members.uid;            // native edge traversal
//! }
//! ```
//!
//! **Default-deny (P0-DBA-1):** any op whose clauses don't evaluate true is denied; a type with no
//! `match` block denies every op. **Total (P0-DBA-7):** a missing field / type mismatch / absent
//! relation resolves to `null` and the clause is simply false — never a panic, never fail-open.
//! Rules are compiled **against the live schema** (unknown type/field/op ⇒ a load error, P0-DBA-8),
//! so a malformed program makes the server refuse to start rather than start open.
//!
//! The evaluator is engine-independent: it reaches object/relationship data through a
//! [`ResourceAccessor`] the executor implements (which is where the actual `db.get_links` traversal
//! and governor budget live), and it speaks `serde_json::Value` (the executor already has
//! `value_to_query_json` to convert engine `Value`s). This keeps the crate free of the engine's
//! ONNX/storage weight.

mod eval;
mod parse;

use std::collections::HashMap;

pub use parse::RulesError;

/// The five authorizable operations. `write` in the DSL expands to `create,update,delete`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Read,
    Create,
    Update,
    Delete,
    Subscribe,
}

impl Op {
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Read => "read",
            Op::Create => "create",
            Op::Update => "update",
            Op::Delete => "delete",
            Op::Subscribe => "subscribe",
        }
    }
}

/// The verdict for one (op, type, principal, resource) evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
}

impl Decision {
    pub fn is_allow(self) -> bool {
        self == Decision::Allow
    }
}

/// A compiled, schema-validated rules program. Cheap to clone-share behind an `Arc` on the
/// `ExecContext`. Built once at server boot (and on schema hot-reload) from a project's `rules.rhype`.
#[derive(Debug, Clone)]
pub struct RulesProgram {
    /// type name → per-op clause lists. A type absent from the map denies all ops (default-deny).
    types: HashMap<String, HashMap<Op, Vec<Expr>>>,
}

impl RulesProgram {
    /// Parse + compile `src` against `schema`. Fails closed on any parse error, unknown type/field/
    /// relation, bad op, or malformed path (P0-DBA-8) — the caller (server boot) treats an error as
    /// "refuse to start", never "start open".
    pub fn compile(src: &str, schema: &rhypedb_schema::Schema) -> Result<Self, RulesError> {
        parse::compile(src, schema)
    }

    /// Authorize `op` on `type_name` for `principal` against `resource`. Default-deny (P0-DBA-1):
    /// an unknown type, an op with no clause, or all-clauses-false ⇒ [`Decision::Deny`].
    pub fn evaluate(
        &self,
        op: Op,
        type_name: &str,
        principal: &crate::Principal,
        resource: &dyn ResourceAccessor,
    ) -> Decision {
        let Some(clauses) = self.types.get(type_name).and_then(|t| t.get(&op)) else {
            return Decision::Deny;
        };
        // Any clause true ⇒ allow.
        for clause in clauses {
            if eval::eval_bool(clause, principal, resource) {
                return Decision::Allow;
            }
        }
        Decision::Deny
    }

    /// Whether this program has ANY clause for `op` on `type_name`. Lets the executor's coarse
    /// pre-execution gate short-circuit a whole-verb denial (a type with no `write` clause can't be
    /// written regardless of the object) without materializing a resource.
    pub fn has_clause(&self, op: Op, type_name: &str) -> bool {
        self.types
            .get(type_name)
            .and_then(|t| t.get(&op))
            .is_some_and(|c| !c.is_empty())
    }
}

/// How the evaluator reads object/relationship/incoming data. Implemented by the executor (in
/// `rhypedb-query`), which owns the `Database` handle + the query [`Governor`] budget the relation
/// traversals must respect. All methods are **total**: an absent/inapplicable datum returns
/// `None`/empty, which the evaluator treats as `null` (⇒ the clause is false, never an error).
///
/// Values are `serde_json::Value` (the executor converts engine `Value`s via `value_to_query_json`).
pub trait ResourceAccessor {
    /// `resource.<scalarField>` of the **pre-mutation** stored object. `None` for a `create` (no
    /// prior resource) or an absent field.
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value>;

    /// `resource.<rel>.<targetField>` for a 1:1 relation — the linked object's scalar. `None` if
    /// unlinked/absent.
    fn resource_relation_one(&self, rel: &str, target_field: &str) -> Option<serde_json::Value>;

    /// `resource.<rel>.<targetField>` for a to-many relation — the projected scalar across every
    /// linked target (a bounded traversal). Empty if none/unlinked.
    fn resource_relation_many(&self, rel: &str, target_field: &str) -> Vec<serde_json::Value>;

    /// `request.<field>` — an incoming write value on a `create`/`update`. `None` if not present in
    /// the write or not applicable (e.g. a read).
    fn request_field(&self, field: &str) -> Option<serde_json::Value>;
}

/// An accessor with no data — every method returns "absent". Useful for the coarse pre-execution
/// gate (resource-independent clauses like `request.auth != null` still evaluate correctly) and for
/// `create` rules that only reference `request.*`/`request.auth` (pair it with a request-field map).
pub struct EmptyResource;
impl ResourceAccessor for EmptyResource {
    fn resource_scalar(&self, _f: &str) -> Option<serde_json::Value> {
        None
    }
    fn resource_relation_one(&self, _r: &str, _f: &str) -> Option<serde_json::Value> {
        None
    }
    fn resource_relation_many(&self, _r: &str, _f: &str) -> Vec<serde_json::Value> {
        Vec::new()
    }
    fn request_field(&self, _f: &str) -> Option<serde_json::Value> {
        None
    }
}

// ---- The compiled expression AST (produced by `parse::compile`, consumed by `eval`). ----

/// A comparison operator. `In` is scalar-in-set membership (relationship projection or JSON array).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
}

/// A literal in a rule expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Lit {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
}

/// A resolved (schema-checked) reference to runtime data. Paths are lowered here at compile time so
/// the evaluator needs no schema.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Operand {
    /// `request.auth` — only ever compared to `null` (compile-enforced). Resolves to a non-null
    /// marker when authenticated, `null` when anonymous.
    Auth,
    /// `request.auth.uid` — the token `sub`, or `null` when anonymous.
    AuthUid,
    /// `request.auth.claims[.a.b…]` — a (possibly nested) custom claim; `null` if absent.
    AuthClaim(Vec<String>),
    /// `resource.<scalarField>`.
    ResourceScalar(String),
    /// `resource.<rel>.<targetField>`, 1:1 relation → a scalar.
    ResourceRelationOne(String, String),
    /// `resource.<rel>.<targetField>`, to-many relation → a set (only valid as the RHS of `in`).
    ResourceRelationMany(String, String),
    /// `request.<field>` — an incoming write value.
    RequestField(String),
}

/// A compiled boolean expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Lit(Lit),
    Operand(Operand),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cmp {
        lhs: Box<Expr>,
        op: CmpOp,
        rhs: Box<Expr>,
    },
}

#[cfg(test)]
mod tests;
