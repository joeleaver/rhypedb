//! Evaluate a compiled rule expression against a principal + a [`ResourceAccessor`].
//!
//! **Three-valued (SQL/CEL-style) logic so a missing value can never grant** (P0-DBA-7). A resolved
//! *absent* datum — a missing field, an unlinked relation, an absent claim, the anonymous
//! principal's uid, or an explicitly-null field value — is [`RVal::Absent`], distinct from the
//! intentional `null` literal ([`RVal::Null`]). ANY comparison with an `Absent` operand, and the
//! negation of an `Absent`/non-boolean operand, is [`Tri::Unknown`], and a clause grants only when
//! it evaluates to [`Tri::True`] — so absence and type-mismatch fail **closed**, never open. The
//! `request.auth == null` idiom still works because the `Auth` operand resolves to a concrete
//! `Null`/`Present`, not `Absent`. There is no error path and no panic.

use super::{CmpOp, Expr, Lit, Operand, ResourceAccessor};
use crate::Principal;

/// A resolved runtime value. `Null` is the *intentional* null (the `null` literal + the anonymous
/// `Auth` marker); `Absent` is an *unknown/missing* value (a missing field/claim/uid/relation, or a
/// null field value) — the two are deliberately distinct so absence fails closed while the
/// `request.auth == null` idiom stays decidable.
#[derive(Debug, Clone)]
enum RVal {
    /// Intentional null: the `null` literal and the anonymous `request.auth`.
    Null,
    /// Unknown/missing: any resolved-absent path (or a stored null). Never equal to anything;
    /// poisons its comparison to `Unknown`.
    Absent,
    /// A non-null, non-comparable marker — authenticated `request.auth`, or a nested claims object.
    Present,
    Bool(bool),
    Num(f64),
    Str(String),
    /// A relationship projection or a JSON array — only used as the RHS of `in`.
    Set(Vec<RVal>),
}

/// Three-valued truth. A clause grants only on [`Tri::True`]; `Unknown` (a missing/typeless operand)
/// and `False` both deny.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    fn from_bool(b: bool) -> Tri {
        if b {
            Tri::True
        } else {
            Tri::False
        }
    }
    fn not(self) -> Tri {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }
}

/// Whether a clause GRANTS: it must evaluate to exactly [`Tri::True`] (Unknown/False deny).
pub(super) fn eval_bool(e: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> bool {
    eval_tri(e, p, r) == Tri::True
}

fn eval_tri(e: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> Tri {
    match e {
        Expr::Lit(Lit::Bool(b)) => Tri::from_bool(*b),
        // A bare `null` is falsy; any other bare non-bool literal is Unknown (non-boolean).
        Expr::Lit(Lit::Null) => Tri::False,
        Expr::Lit(_) => Tri::Unknown,
        Expr::Not(x) => eval_tri(x, p, r).not(),
        Expr::And(a, b) => {
            // False dominates (short-circuit); otherwise Unknown if either side is Unknown.
            let av = eval_tri(a, p, r);
            if av == Tri::False {
                return Tri::False;
            }
            let bv = eval_tri(b, p, r);
            if av == Tri::True && bv == Tri::True {
                Tri::True
            } else if bv == Tri::False {
                Tri::False
            } else {
                Tri::Unknown
            }
        }
        Expr::Or(a, b) => {
            // True dominates (short-circuit); otherwise Unknown if either side is Unknown.
            let av = eval_tri(a, p, r);
            if av == Tri::True {
                return Tri::True;
            }
            let bv = eval_tri(b, p, r);
            if bv == Tri::True {
                Tri::True
            } else if av == Tri::False && bv == Tri::False {
                Tri::False
            } else {
                Tri::Unknown
            }
        }
        Expr::Cmp { lhs, op, rhs } => eval_cmp(lhs, *op, rhs, p, r),
        // A bare operand: a true Bool / an authenticated Present is True; an intentional Null is
        // False; a missing (Absent) or non-boolean (Str/Num/Set) operand is Unknown (never grants).
        Expr::Operand(o) => match resolve(o, p, r) {
            RVal::Bool(b) => Tri::from_bool(b),
            RVal::Present => Tri::True,
            RVal::Null => Tri::False,
            RVal::Absent | RVal::Str(_) | RVal::Num(_) | RVal::Set(_) => Tri::Unknown,
        },
    }
}

fn eval_cmp(lhs: &Expr, op: CmpOp, rhs: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> Tri {
    let lv = value_of(lhs, p, r);
    let rv = value_of(rhs, p, r);
    // SQL-style: ANY comparison with an unknown/missing operand is Unknown (never grants). This is
    // what stops `request.auth.uid == resource.ownerUid` from granting via `Absent == Absent` when
    // both are missing, and `!resource.flag` from granting when the flag is absent.
    if matches!(lv, RVal::Absent) || matches!(rv, RVal::Absent) {
        return Tri::Unknown;
    }
    let b = match op {
        CmpOp::Eq => rval_eq(&lv, &rv),
        CmpOp::Ne => !rval_eq(&lv, &rv),
        CmpOp::Lt => rval_ord(&lv, &rv).is_some_and(|o| o.is_lt()),
        CmpOp::Le => rval_ord(&lv, &rv).is_some_and(|o| o.is_le()),
        CmpOp::Gt => rval_ord(&lv, &rv).is_some_and(|o| o.is_gt()),
        CmpOp::Ge => rval_ord(&lv, &rv).is_some_and(|o| o.is_ge()),
        CmpOp::In => match &rv {
            RVal::Set(items) => items.iter().any(|it| rval_eq(&lv, it)),
            _ => false, // `in` over a non-set is never a match
        },
    };
    Tri::from_bool(b)
}

/// The value of an expression used as a comparison operand. A boolean sub-expression resolves to its
/// truth (Unknown ⇒ `Absent`, so it poisons the enclosing comparison); a leaf to its operand value.
fn value_of(e: &Expr, p: &Principal, r: &dyn ResourceAccessor) -> RVal {
    match e {
        Expr::Lit(l) => lit_to_rval(l),
        Expr::Operand(o) => resolve(o, p, r),
        Expr::Not(_) | Expr::And(_, _) | Expr::Or(_, _) | Expr::Cmp { .. } => {
            match eval_tri(e, p, r) {
                Tri::True => RVal::Bool(true),
                Tri::False => RVal::Bool(false),
                Tri::Unknown => RVal::Absent,
            }
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
        // Intentional: authenticated ⇒ Present, anonymous ⇒ Null (so `request.auth == null` is
        // decidable — NOT Absent).
        Operand::Auth => {
            if p.is_authenticated() {
                RVal::Present
            } else {
                RVal::Null
            }
        }
        // A missing uid (anonymous, or no sub) is Absent — an uid comparison then never grants.
        Operand::AuthUid => p.uid().map(|s| RVal::Str(s.to_string())).unwrap_or(RVal::Absent),
        Operand::AuthClaim(path) => {
            let mut cur = p.claims();
            for seg in path {
                match cur.get(seg) {
                    Some(v) => cur = v,
                    None => return RVal::Absent,
                }
            }
            json_to_rval(cur)
        }
        Operand::ResourceScalar(f) => r.resource_scalar(f).map(|v| json_to_rval(&v)).unwrap_or(RVal::Absent),
        Operand::ResourceRelationOne(rel, f) => r
            .resource_relation_one(rel, f)
            .map(|v| json_to_rval(&v))
            .unwrap_or(RVal::Absent),
        Operand::ResourceRelationMany(rel, f) => {
            RVal::Set(r.resource_relation_many(rel, f).iter().map(json_to_rval).collect())
        }
        Operand::RequestField(f) => r.request_field(f).map(|v| json_to_rval(&v)).unwrap_or(RVal::Absent),
    }
}

fn json_to_rval(v: &serde_json::Value) -> RVal {
    match v {
        // A stored/null field value carries no information → Absent (fails closed like a missing one).
        serde_json::Value::Null => RVal::Absent,
        serde_json::Value::Bool(b) => RVal::Bool(*b),
        serde_json::Value::Number(n) => n.as_f64().map(RVal::Num).unwrap_or(RVal::Absent),
        serde_json::Value::String(s) => RVal::Str(s.clone()),
        serde_json::Value::Array(a) => RVal::Set(a.iter().map(json_to_rval).collect()),
        // A nested object is opaque — non-null but not scalar-comparable.
        serde_json::Value::Object(_) => RVal::Present,
    }
}

/// Equality over concrete (non-Absent) values. Absent never reaches here — [`eval_cmp`] short-
/// circuits an Absent operand to `Unknown` first — so two missing values are NOT equal.
fn rval_eq(a: &RVal, b: &RVal) -> bool {
    match (a, b) {
        (RVal::Null, RVal::Null) => true,
        (RVal::Present, RVal::Present) => true,
        (RVal::Bool(x), RVal::Bool(y)) => x == y,
        (RVal::Num(x), RVal::Num(y)) => x == y,
        (RVal::Str(x), RVal::Str(y)) => x == y,
        // Sets compare only via `in`; every other pair (incl. any Absent that slipped in) is unequal.
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
