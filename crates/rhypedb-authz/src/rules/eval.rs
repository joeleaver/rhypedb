//! Evaluate a compiled rule expression against a principal + a [`ResourceAccessor`].
//!
//! Everything is **total** (P0-DBA-7): an absent field / relation / claim resolves to `null`, a
//! cross-type comparison is simply `false`, and a non-boolean in boolean position is `false`. There
//! is no error path and no panic — so a clause can only ever *fail to grant*, never fail open.

use super::{CmpOp, Expr, Lit, Operand, ResourceAccessor};
use crate::Principal;

/// A resolved runtime value.
#[derive(Debug, Clone)]
enum RVal {
    Null,
    /// A non-null, non-comparable marker — `request.auth` when authenticated, or a nested claims
    /// object. Only meaningful against `null` (present ⇒ `!= null`).
    Present,
    Bool(bool),
    Num(f64),
    Str(String),
    /// A relationship projection or a JSON array — only used as the RHS of `in`.
    Set(Vec<RVal>),
}

/// Evaluate an expression as a boolean (the top of every clause is boolean).
pub(super) fn eval_bool(e: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> bool {
    match e {
        Expr::Lit(Lit::Bool(b)) => *b,
        // A bare non-boolean literal in boolean position is false (total).
        Expr::Lit(_) => false,
        Expr::Not(x) => !eval_bool(x, p, r),
        Expr::And(a, b) => eval_bool(a, p, r) && eval_bool(b, p, r),
        Expr::Or(a, b) => eval_bool(a, p, r) || eval_bool(b, p, r),
        Expr::Cmp { lhs, op, rhs } => eval_cmp(lhs, *op, rhs, p, r),
        // A bare operand is truthy iff it resolves to a true Bool or a Present (authenticated) marker.
        Expr::Operand(o) => match resolve(o, p, r) {
            RVal::Bool(b) => b,
            RVal::Present => true,
            _ => false,
        },
    }
}

fn eval_cmp(lhs: &Expr, op: CmpOp, rhs: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> bool {
    let lv = value_of(lhs, p, r);
    let rv = value_of(rhs, p, r);
    match op {
        CmpOp::Eq => rval_eq(&lv, &rv),
        CmpOp::Ne => !rval_eq(&lv, &rv),
        CmpOp::Lt => rval_ord(&lv, &rv).is_some_and(|o| o.is_lt()),
        CmpOp::Le => rval_ord(&lv, &rv).is_some_and(|o| o.is_le()),
        CmpOp::Gt => rval_ord(&lv, &rv).is_some_and(|o| o.is_gt()),
        CmpOp::Ge => rval_ord(&lv, &rv).is_some_and(|o| o.is_ge()),
        CmpOp::In => match &rv {
            RVal::Set(items) => items.iter().any(|it| rval_eq(&lv, it)),
            _ => false, // `in` over a non-set is never a match (total)
        },
    }
}

/// The value of an expression used as a comparison operand. A boolean sub-expression resolves to its
/// bool; a leaf resolves to its literal/operand value.
fn value_of(e: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> RVal {
    match e {
        Expr::Lit(l) => lit_to_rval(l),
        Expr::Operand(o) => resolve(o, p, r),
        Expr::Not(_) | Expr::And(_, _) | Expr::Or(_, _) | Expr::Cmp { .. } => {
            RVal::Bool(eval_bool(e, p, r))
        }
    }
}

fn lit_to_rval(l: &Lit) -> RVal {
    match l {
        Lit::Null => RVal::Null,
        Lit::Bool(b) => RVal::Bool(*b),
        Lit::Num(n) => RVal::Num(*n),
        Lit::Str(s) => RVal::Str(s.clone()),
    }
}

fn resolve(o: &Operand, p: &Principal, r: &dyn ResourceAccessor) -> RVal {
    match o {
        Operand::Auth => {
            if p.is_authenticated() {
                RVal::Present
            } else {
                RVal::Null
            }
        }
        Operand::AuthUid => p.uid().map(|s| RVal::Str(s.to_string())).unwrap_or(RVal::Null),
        Operand::AuthClaim(path) => {
            let mut cur = p.claims();
            for seg in path {
                match cur.get(seg) {
                    Some(v) => cur = v,
                    None => return RVal::Null,
                }
            }
            json_to_rval(cur)
        }
        Operand::ResourceScalar(f) => r.resource_scalar(f).map(|v| json_to_rval(&v)).unwrap_or(RVal::Null),
        Operand::ResourceRelationOne(rel, f) => r
            .resource_relation_one(rel, f)
            .map(|v| json_to_rval(&v))
            .unwrap_or(RVal::Null),
        Operand::ResourceRelationMany(rel, f) => {
            RVal::Set(r.resource_relation_many(rel, f).iter().map(json_to_rval).collect())
        }
        Operand::RequestField(f) => r.request_field(f).map(|v| json_to_rval(&v)).unwrap_or(RVal::Null),
    }
}

fn json_to_rval(v: &serde_json::Value) -> RVal {
    match v {
        serde_json::Value::Null => RVal::Null,
        serde_json::Value::Bool(b) => RVal::Bool(*b),
        serde_json::Value::Number(n) => n.as_f64().map(RVal::Num).unwrap_or(RVal::Null),
        serde_json::Value::String(s) => RVal::Str(s.clone()),
        serde_json::Value::Array(a) => RVal::Set(a.iter().map(json_to_rval).collect()),
        // A nested object is opaque — non-null but not scalar-comparable.
        serde_json::Value::Object(_) => RVal::Present,
    }
}

fn rval_eq(a: &RVal, b: &RVal) -> bool {
    match (a, b) {
        (RVal::Null, RVal::Null) => true,
        (RVal::Present, RVal::Present) => true,
        (RVal::Bool(x), RVal::Bool(y)) => x == y,
        (RVal::Num(x), RVal::Num(y)) => x == y,
        (RVal::Str(x), RVal::Str(y)) => x == y,
        // Sets compare only via `in`; every other cross-type pair is unequal (total).
        _ => false,
    }
}

fn rval_ord(a: &RVal, b: &RVal) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (RVal::Num(x), RVal::Num(y)) => x.partial_cmp(y),
        (RVal::Str(x), RVal::Str(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
