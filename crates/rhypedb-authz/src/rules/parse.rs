//! Lexer + recursive-descent parser + schema-driven compiler for `rules.rhype`.
//!
//! Two stages: parse `src` into an untyped AST (paths stay as `["request","auth","uid"]`), then
//! **compile** each path against the schema into a typed [`Operand`] (P0-DBA-8: an unknown type/
//! field/relation, or a path shape the model doesn't allow, is a hard error → the server refuses to
//! start). Keeping schema resolution at compile time means the evaluator carries no schema.

use rhypedb_schema::{FieldType, Schema};
use std::collections::HashMap;

use super::{CmpOp, Expr, Lit, Op, Operand};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RulesError {
    #[error("rules parse error at {pos}: {message}")]
    Parse { pos: usize, message: String },
    #[error("rules schema error: {0}")]
    Schema(String),
}

impl RulesError {
    fn parse(pos: usize, message: impl Into<String>) -> Self {
        RulesError::Parse {
            pos,
            message: message.into(),
        }
    }
    fn schema(message: impl Into<String>) -> Self {
        RulesError::Schema(message.into())
    }
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Match,
    Allow,
    If,
    In,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Colon,
    Semi,
    Comma,
    Dot,
    OrOr,
    AndAnd,
    Bang,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Ident(String),
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone)]
struct Spanned {
    tok: Tok,
    pos: usize,
}

fn lex(src: &str) -> Result<Vec<Spanned>, RulesError> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        let c = bytes[i];
        // Whitespace.
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Line comments: `//` and `#`.
        if c == b'#' || (c == b'/' && bytes.get(i + 1) == Some(&b'/')) {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        // Multi-char / punctuation operators.
        let two = |a: u8, b: u8| c == a && bytes.get(i + 1) == Some(&b);
        if two(b'|', b'|') {
            out.push(Spanned { tok: Tok::OrOr, pos: start });
            i += 2;
            continue;
        }
        if two(b'&', b'&') {
            out.push(Spanned { tok: Tok::AndAnd, pos: start });
            i += 2;
            continue;
        }
        if two(b'=', b'=') {
            out.push(Spanned { tok: Tok::EqEq, pos: start });
            i += 2;
            continue;
        }
        if two(b'!', b'=') {
            out.push(Spanned { tok: Tok::NotEq, pos: start });
            i += 2;
            continue;
        }
        if two(b'<', b'=') {
            out.push(Spanned { tok: Tok::Le, pos: start });
            i += 2;
            continue;
        }
        if two(b'>', b'=') {
            out.push(Spanned { tok: Tok::Ge, pos: start });
            i += 2;
            continue;
        }
        let single = match c {
            b'{' => Some(Tok::LBrace),
            b'}' => Some(Tok::RBrace),
            b'(' => Some(Tok::LParen),
            b')' => Some(Tok::RParen),
            b':' => Some(Tok::Colon),
            b';' => Some(Tok::Semi),
            b',' => Some(Tok::Comma),
            b'.' => Some(Tok::Dot),
            b'!' => Some(Tok::Bang),
            b'<' => Some(Tok::Lt),
            b'>' => Some(Tok::Gt),
            _ => None,
        };
        if let Some(t) = single {
            out.push(Spanned { tok: t, pos: start });
            i += 1;
            continue;
        }
        // String literal (double-quoted, with \" \\ \n \t escapes). Advance past the closing quote
        // honoring escapes, then decode the value UTF-8-correctly from the original slice.
        if c == b'"' {
            i += 1;
            loop {
                let Some(&ch) = bytes.get(i) else {
                    return Err(RulesError::parse(start, "unterminated string literal"));
                };
                match ch {
                    b'"' => {
                        i += 1;
                        break;
                    }
                    b'\\' => {
                        if bytes.get(i + 1).is_none() {
                            return Err(RulesError::parse(i, "dangling escape"));
                        }
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            out.push(Spanned { tok: Tok::Str(decode_str_span(src, start)?), pos: start });
            continue;
        }
        // Number literal (int/float, optional leading '-', scientific notation).
        if c.is_ascii_digit() || (c == b'-' && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit())) {
            let ns = i;
            if bytes[i] == b'-' {
                i += 1;
            }
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            let text = &src[ns..i];
            let n: f64 = text
                .parse()
                .map_err(|_| RulesError::parse(ns, format!("invalid number `{text}`")))?;
            out.push(Spanned { tok: Tok::Num(n), pos: ns });
            continue;
        }
        // Identifier / keyword.
        if c == b'_' || c.is_ascii_alphabetic() {
            let is = i;
            while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            let word = &src[is..i];
            let tok = match word {
                "match" => Tok::Match,
                "allow" => Tok::Allow,
                "if" => Tok::If,
                "in" => Tok::In,
                "true" => Tok::Bool(true),
                "false" => Tok::Bool(false),
                "null" => Tok::Null,
                _ => Tok::Ident(word.to_string()),
            };
            out.push(Spanned { tok, pos: is });
            continue;
        }
        return Err(RulesError::parse(start, format!("unexpected character `{}`", c as char)));
    }
    Ok(out)
}

/// Re-decode a `"..."` string literal from the raw source starting at the opening quote at `start`,
/// UTF-8-correct (the byte-wise lexer path is ASCII-only). Handles the same escapes.
fn decode_str_span(src: &str, start: usize) -> Result<String, RulesError> {
    let mut chars = src[start..].char_indices();
    // Skip the opening quote.
    let _ = chars.next();
    let mut s = String::new();
    let mut escaped = false;
    for (_, ch) in chars {
        if escaped {
            s.push(match ch {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                't' => '\t',
                other => {
                    return Err(RulesError::parse(start, format!("unknown escape \\{other}")));
                }
            });
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Ok(s),
            _ => s.push(ch),
        }
    }
    Err(RulesError::parse(start, "unterminated string literal"))
}

// ---------------------------------------------------------------------------
// Parser → untyped AST
// ---------------------------------------------------------------------------

/// Untyped expression: paths are not yet schema-resolved.
enum RawExpr {
    Lit(Lit),
    Path { segs: Vec<String>, pos: usize },
    Not(Box<RawExpr>),
    And(Box<RawExpr>, Box<RawExpr>),
    Or(Box<RawExpr>, Box<RawExpr>),
    Cmp(Box<RawExpr>, CmpOp, Box<RawExpr>),
}

struct RawAllow {
    ops: Vec<Op>,
    expr: RawExpr,
}

struct RawStmt {
    type_name: String,
    type_pos: usize,
    allows: Vec<RawAllow>,
}

struct Parser {
    toks: Vec<Spanned>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|s| &s.tok)
    }
    fn pos(&self) -> usize {
        self.toks.get(self.i).map(|s| s.pos).unwrap_or(usize::MAX)
    }
    fn bump(&mut self) -> Option<Spanned> {
        let t = self.toks.get(self.i).cloned();
        if t.is_some() {
            self.i += 1;
        }
        t
    }
    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), RulesError> {
        match self.bump() {
            Some(s) if &s.tok == want => Ok(()),
            Some(s) => Err(RulesError::parse(s.pos, format!("expected {what}"))),
            None => Err(RulesError::parse(usize::MAX, format!("expected {what}, got end of input"))),
        }
    }

    fn program(&mut self) -> Result<Vec<RawStmt>, RulesError> {
        let mut stmts = Vec::new();
        while self.peek().is_some() {
            stmts.push(self.stmt()?);
        }
        Ok(stmts)
    }

    fn stmt(&mut self) -> Result<RawStmt, RulesError> {
        self.expect(&Tok::Match, "`match`")?;
        let (type_name, type_pos) = self.ident("a type name")?;
        self.expect(&Tok::LBrace, "`{`")?;
        let mut allows = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            if self.peek().is_none() {
                return Err(RulesError::parse(usize::MAX, "unterminated `match` block"));
            }
            allows.push(self.allow()?);
        }
        self.expect(&Tok::RBrace, "`}`")?;
        Ok(RawStmt { type_name, type_pos, allows })
    }

    fn allow(&mut self) -> Result<RawAllow, RulesError> {
        self.expect(&Tok::Allow, "`allow`")?;
        let mut ops = Vec::new();
        loop {
            let (name, pos) = self.ident("an operation (read/create/update/delete/subscribe/write)")?;
            match name.as_str() {
                "read" => ops.push(Op::Read),
                "create" => ops.push(Op::Create),
                "update" => ops.push(Op::Update),
                "delete" => ops.push(Op::Delete),
                "subscribe" => ops.push(Op::Subscribe),
                "write" => ops.extend([Op::Create, Op::Update, Op::Delete]),
                other => {
                    return Err(RulesError::parse(pos, format!("unknown operation `{other}`")));
                }
            }
            if self.peek() == Some(&Tok::Comma) {
                self.bump();
                continue;
            }
            break;
        }
        self.expect(&Tok::Colon, "`:`")?;
        self.expect(&Tok::If, "`if`")?;
        let expr = self.expr()?;
        self.expect(&Tok::Semi, "`;`")?;
        Ok(RawAllow { ops, expr })
    }

    // expr := or
    fn expr(&mut self) -> Result<RawExpr, RulesError> {
        self.or()
    }

    fn or(&mut self) -> Result<RawExpr, RulesError> {
        let mut lhs = self.and()?;
        while self.peek() == Some(&Tok::OrOr) {
            self.bump();
            let rhs = self.and()?;
            lhs = RawExpr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn and(&mut self) -> Result<RawExpr, RulesError> {
        let mut lhs = self.cmp()?;
        while self.peek() == Some(&Tok::AndAnd) {
            self.bump();
            let rhs = self.cmp()?;
            lhs = RawExpr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn cmp(&mut self) -> Result<RawExpr, RulesError> {
        let lhs = self.unary()?;
        let op = match self.peek() {
            Some(Tok::EqEq) => CmpOp::Eq,
            Some(Tok::NotEq) => CmpOp::Ne,
            Some(Tok::Lt) => CmpOp::Lt,
            Some(Tok::Le) => CmpOp::Le,
            Some(Tok::Gt) => CmpOp::Gt,
            Some(Tok::Ge) => CmpOp::Ge,
            Some(Tok::In) => CmpOp::In,
            _ => return Ok(lhs),
        };
        self.bump();
        let rhs = self.unary()?;
        Ok(RawExpr::Cmp(Box::new(lhs), op, Box::new(rhs)))
    }

    fn unary(&mut self) -> Result<RawExpr, RulesError> {
        if self.peek() == Some(&Tok::Bang) {
            self.bump();
            let inner = self.unary()?;
            return Ok(RawExpr::Not(Box::new(inner)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<RawExpr, RulesError> {
        let pos = self.pos();
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.bump();
                let e = self.expr()?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Some(Tok::Null) => {
                self.bump();
                Ok(RawExpr::Lit(Lit::Null))
            }
            Some(Tok::Bool(b)) => {
                self.bump();
                Ok(RawExpr::Lit(Lit::Bool(b)))
            }
            Some(Tok::Num(n)) => {
                self.bump();
                Ok(RawExpr::Lit(Lit::Num(n)))
            }
            Some(Tok::Str(s)) => {
                self.bump();
                Ok(RawExpr::Lit(Lit::Str(s)))
            }
            Some(Tok::Ident(_)) => {
                // A dotted path: ident ("." ident)*
                let mut segs = Vec::new();
                let (first, _) = self.ident("an identifier")?;
                segs.push(first);
                while self.peek() == Some(&Tok::Dot) {
                    self.bump();
                    let (seg, _) = self.ident("an identifier after `.`")?;
                    segs.push(seg);
                }
                Ok(RawExpr::Path { segs, pos })
            }
            _ => Err(RulesError::parse(pos, "expected an expression")),
        }
    }

    fn ident(&mut self, what: &str) -> Result<(String, usize), RulesError> {
        match self.bump() {
            Some(Spanned { tok: Tok::Ident(s), pos }) => Ok((s, pos)),
            Some(s) => Err(RulesError::parse(s.pos, format!("expected {what}"))),
            None => Err(RulesError::parse(usize::MAX, format!("expected {what}, got end of input"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Compile: resolve paths against the schema → typed Expr
// ---------------------------------------------------------------------------

pub fn compile(src: &str, schema: &Schema) -> Result<super::RulesProgram, RulesError> {
    let toks = lex(src)?;
    let mut parser = Parser { toks, i: 0 };
    let stmts = parser.program()?;

    let mut types: HashMap<String, HashMap<Op, Vec<Expr>>> = HashMap::new();
    for stmt in stmts {
        let td = schema.get_type(&stmt.type_name).ok_or_else(|| {
            RulesError::Parse {
                pos: stmt.type_pos,
                message: format!("`match {}`: no such type in the schema", stmt.type_name),
            }
        })?;
        let entry = types.entry(stmt.type_name.clone()).or_default();
        for allow in stmt.allows {
            let compiled = compile_expr(&allow.expr, stmt.type_name.as_str(), td, schema)?;
            for op in allow.ops {
                entry.entry(op).or_default().push(compiled.clone());
            }
        }
    }
    Ok(super::RulesProgram { types })
}

fn compile_expr(
    e: &RawExpr,
    type_name: &str,
    td: &rhypedb_schema::TypeDef,
    schema: &Schema,
) -> Result<Expr, RulesError> {
    Ok(match e {
        RawExpr::Lit(l) => Expr::Lit(l.clone()),
        RawExpr::Not(x) => Expr::Not(Box::new(compile_expr(x, type_name, td, schema)?)),
        RawExpr::And(a, b) => Expr::And(
            Box::new(compile_expr(a, type_name, td, schema)?),
            Box::new(compile_expr(b, type_name, td, schema)?),
        ),
        RawExpr::Or(a, b) => Expr::Or(
            Box::new(compile_expr(a, type_name, td, schema)?),
            Box::new(compile_expr(b, type_name, td, schema)?),
        ),
        RawExpr::Cmp(a, op, b) => {
            let lhs = compile_expr(a, type_name, td, schema)?;
            let rhs = compile_expr(b, type_name, td, schema)?;
            // `request.auth` may only be compared to `null` (== / !=) — its non-null form isn't a
            // comparable value (use request.auth.uid for the id).
            let lhs_auth = matches!(lhs, Expr::Operand(Operand::Auth));
            let rhs_auth = matches!(rhs, Expr::Operand(Operand::Auth));
            if lhs_auth || rhs_auth {
                let other_is_null = if lhs_auth {
                    matches!(rhs, Expr::Lit(Lit::Null))
                } else {
                    matches!(lhs, Expr::Lit(Lit::Null))
                };
                let both_auth = lhs_auth && rhs_auth;
                if both_auth || !matches!(op, CmpOp::Eq | CmpOp::Ne) || !other_is_null {
                    return Err(RulesError::schema(
                        "`request.auth` may only be compared to `null` (use request.auth.uid for the id)",
                    ));
                }
            }
            Expr::Cmp { lhs: Box::new(lhs), op: *op, rhs: Box::new(rhs) }
        }
        RawExpr::Path { segs, pos } => {
            Expr::Operand(compile_path(segs, *pos, type_name, td, schema)?)
        }
    })
}

fn compile_path(
    segs: &[String],
    pos: usize,
    type_name: &str,
    td: &rhypedb_schema::TypeDef,
    schema: &Schema,
) -> Result<Operand, RulesError> {
    let err = |m: String| RulesError::Parse { pos, message: m };
    match segs {
        [root] if root == "request" || root == "resource" => Err(err(format!(
            "`{root}` needs a field (e.g. {root}.<field>)"
        ))),
        // request.auth[.uid | .claims[.a.b…]]
        [r, a] if r == "request" && a == "auth" => Ok(Operand::Auth),
        [r, a, f] if r == "request" && a == "auth" && f == "uid" => Ok(Operand::AuthUid),
        [r, a, c, rest @ ..] if r == "request" && a == "auth" && c == "claims" => {
            Ok(Operand::AuthClaim(rest.to_vec()))
        }
        [r, a, bad] if r == "request" && a == "auth" => Err(err(format!(
            "unknown `request.auth.{bad}` (use uid or claims.*)"
        ))),
        // request.<writeField> — must be a real scalar/relation field of this type.
        [r, field] if r == "request" => {
            let fd = td.get_field(field).ok_or_else(|| {
                err(format!("`request.{field}`: no field `{field}` on type `{type_name}`"))
            })?;
            match &fd.field_type {
                FieldType::Scalar(_) | FieldType::Relation(_) => Ok(Operand::RequestField(field.clone())),
                FieldType::Vector(_) => Err(err(format!(
                    "`request.{field}`: vector fields can't be referenced in rules"
                ))),
            }
        }
        // resource.<scalarField>
        [r, field] if r == "resource" => {
            let fd = td.get_field(field).ok_or_else(|| {
                err(format!("`resource.{field}`: no field `{field}` on type `{type_name}`"))
            })?;
            match &fd.field_type {
                FieldType::Scalar(_) => Ok(Operand::ResourceScalar(field.clone())),
                FieldType::Relation(_) => Err(err(format!(
                    "`resource.{field}` is a relationship — project a target field, e.g. resource.{field}.<field>"
                ))),
                FieldType::Vector(_) => Err(err(format!(
                    "`resource.{field}`: vector fields can't be referenced in rules"
                ))),
            }
        }
        // resource.<rel>.<targetField>
        [r, rel, target_field] if r == "resource" => {
            let fd = td.get_field(rel).ok_or_else(|| {
                err(format!("`resource.{rel}`: no field `{rel}` on type `{type_name}`"))
            })?;
            let FieldType::Relation(relation) = &fd.field_type else {
                return Err(err(format!(
                    "`resource.{rel}.{target_field}`: `{rel}` is not a relationship"
                )));
            };
            let target = schema.get_type(&relation.target_type).ok_or_else(|| {
                RulesError::schema(format!(
                    "relationship `{type_name}.{rel}` targets unknown type `{}`",
                    relation.target_type
                ))
            })?;
            let tf = target.get_field(target_field).ok_or_else(|| {
                err(format!(
                    "`resource.{rel}.{target_field}`: no field `{target_field}` on `{}`",
                    relation.target_type
                ))
            })?;
            if !matches!(tf.field_type, FieldType::Scalar(_)) {
                return Err(err(format!(
                    "`resource.{rel}.{target_field}` must project a scalar field"
                )));
            }
            if relation.is_many {
                Ok(Operand::ResourceRelationMany(rel.clone(), target_field.clone()))
            } else {
                Ok(Operand::ResourceRelationOne(rel.clone(), target_field.clone()))
            }
        }
        _ => Err(err(format!("unsupported path `{}`", segs.join(".")))),
    }
}
