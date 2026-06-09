use std::collections::HashMap;

use crate::ast::*;
use crate::error::{QueryError, QueryResult};

/// Parse a query string into a Query AST.
///
/// Grammar (informal):
/// ```text
/// query     = source ("." step)*
/// source    = IDENT ".get(" INT ")"
///           | IDENT ".filter(" predicate ")"
///           | IDENT ".create(" object ")"
///           | IDENT
/// step      = "filter(" predicate ")"
///           | "similar(" field "," vector "," "k:" INT ("," ("ef"|"rerank") ":" INT)* ")"
///           | "update(" object ")"
///           | "delete()"
///           | "link(" IDENT ".get(" INT ")" ["," object] ")"
///           | "unlink(" IDENT ".get(" INT ")" ")"
///           | "limit(" INT ")"
///           | "offset(" INT ")"
///           | IDENT                          (traverse)
/// predicate = "." FIELD_PATH OP literal
/// object    = "{" (IDENT ":" literal ",")* "}"
/// literal   = STRING | INT | FLOAT | "true" | "false" | "null"
/// ```
pub fn parse_query(input: &str) -> QueryResult<Query> {
    let mut p = Parser::new(input);
    let query = p.parse_query()?;
    p.skip_ws();
    if !p.is_eof() {
        return Err(p.error("unexpected trailing input"));
    }
    Ok(query)
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect_char(&mut self, expected: char) -> QueryResult<()> {
        self.skip_ws();
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(self.error(format!("expected '{expected}', got '{ch}'"))),
            None => Err(self.error(format!("expected '{expected}', got EOF"))),
        }
    }

    fn expect_str(&mut self, s: &str) -> QueryResult<()> {
        self.skip_ws();
        if self.input[self.pos..].starts_with(s) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(self.error(format!("expected '{s}'")))
        }
    }

    fn parse_ident(&mut self) -> QueryResult<String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.error("expected identifier"));
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_int(&mut self) -> QueryResult<i64> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some('-') {
            self.advance();
        }
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        if self.pos == start || (self.pos == start + 1 && self.input.as_bytes()[start] == b'-') {
            return Err(self.error("expected integer"));
        }
        self.input[start..self.pos]
            .parse()
            .map_err(|_| self.error("invalid integer"))
    }

    /// Parse a non-negative count (`limit` / `offset` / `k`). Rejects negatives,
    /// which would otherwise wrap to a huge `usize` (an unbounded limit, or a
    /// `k` that overflows the vectorizer's candidate math).
    fn parse_count(&mut self) -> QueryResult<usize> {
        let n = self.parse_int()?;
        if n < 0 {
            return Err(self.error("expected a non-negative integer"));
        }
        Ok(n as usize)
    }

    fn parse_number(&mut self) -> QueryResult<Literal> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some('-') {
            self.advance();
        }
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        if self.peek() == Some('.') {
            self.advance();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            let val: f64 = self.input[start..self.pos]
                .parse()
                .map_err(|_| self.error("invalid float"))?;
            Ok(Literal::Float(val))
        } else {
            let val: i64 = self.input[start..self.pos]
                .parse()
                .map_err(|_| self.error("invalid integer"))?;
            Ok(Literal::Int(val))
        }
    }

    fn parse_string_literal(&mut self) -> QueryResult<String> {
        self.skip_ws();
        self.expect_char('"')?;
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch == '"' {
                let s = self.input[start..self.pos].to_string();
                self.advance(); // consume closing quote
                return Ok(s);
            }
            if ch == '\\' {
                self.advance(); // skip escape
            }
            self.advance();
        }
        Err(self.error("unterminated string"))
    }

    fn parse_literal(&mut self) -> QueryResult<Literal> {
        self.skip_ws();
        match self.peek() {
            Some('"') => Ok(Literal::String(self.parse_string_literal()?)),
            Some(ch) if ch.is_ascii_digit() || ch == '-' => self.parse_number(),
            Some('t') => {
                self.expect_str("true")?;
                Ok(Literal::Bool(true))
            }
            Some('f') => {
                self.expect_str("false")?;
                Ok(Literal::Bool(false))
            }
            Some('n') => {
                self.expect_str("null")?;
                Ok(Literal::Null)
            }
            _ => Err(self.error("expected literal value")),
        }
    }

    fn parse_object(&mut self) -> QueryResult<HashMap<String, Literal>> {
        self.expect_char('{')?;
        let mut fields = HashMap::new();

        loop {
            self.skip_ws();
            if self.peek() == Some('}') {
                self.advance();
                break;
            }

            let name = self.parse_ident()?;
            self.expect_char(':')?;
            let value = self.parse_literal()?;
            fields.insert(name, value);

            self.skip_ws();
            if self.peek() == Some(',') {
                self.advance();
            }
        }

        Ok(fields)
    }

    /// Predicate grammar with standard boolean precedence (`&&` binds tighter
    /// than `||`) and parenthesized grouping, both left-associative:
    /// ```text
    /// predicate = and_term ("||" | "or" and_term)*
    /// and_term  = factor   ("&&" | "and" factor)*
    /// factor    = "(" predicate ")" | atom
    /// ```
    fn parse_predicate(&mut self) -> QueryResult<Predicate> {
        let mut left = self.parse_and_term()?;
        loop {
            self.skip_ws();
            let remaining = &self.input[self.pos..];
            if remaining.starts_with("||") {
                self.pos += 2;
            } else if remaining.starts_with("or ") {
                self.pos += 3;
            } else {
                break;
            }
            let right = self.parse_and_term()?;
            left = Predicate::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and_term(&mut self) -> QueryResult<Predicate> {
        let mut left = self.parse_predicate_factor()?;
        loop {
            self.skip_ws();
            let remaining = &self.input[self.pos..];
            if remaining.starts_with("&&") {
                self.pos += 2;
            } else if remaining.starts_with("and ") {
                self.pos += 4;
            } else {
                break;
            }
            let right = self.parse_predicate_factor()?;
            left = Predicate::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_predicate_factor(&mut self) -> QueryResult<Predicate> {
        self.skip_ws();
        if self.peek() == Some('(') {
            self.advance(); // consume '('
            let inner = self.parse_predicate()?;
            self.expect_char(')')?;
            Ok(inner)
        } else {
            self.parse_predicate_atom()
        }
    }

    fn parse_predicate_atom(&mut self) -> QueryResult<Predicate> {
        self.expect_char('.')?;

        // Parse field path (e.g., "name" or "tags.name").
        let mut path = self.parse_ident()?;
        while self.peek() == Some('.') {
            let saved = self.pos;
            self.advance();
            // Check if next is an ident (part of field path) or something else.
            if let Some(ch) = self.peek() {
                if ch.is_alphabetic() || ch == '_' {
                    let next = self.parse_ident()?;
                    // Check if the next thing is an operator — if so, this was the field path.
                    self.skip_ws();
                    let ahead = &self.input[self.pos..];
                    if ahead.starts_with("==")
                        || ahead.starts_with("!=")
                        || ahead.starts_with("<=")
                        || ahead.starts_with(">=")
                        || ahead.starts_with('<')
                        || ahead.starts_with('>')
                    {
                        path = format!("{path}.{next}");
                    } else {
                        // Not a comparison — backtrack, this dot was something else.
                        self.pos = saved;
                        break;
                    }
                } else {
                    self.pos = saved;
                    break;
                }
            } else {
                self.pos = saved;
                break;
            }
        }

        self.skip_ws();
        let op = self.parse_compare_op()?;
        let value = self.parse_literal()?;

        Ok(Predicate::Compare {
            field_path: path,
            op,
            value,
        })
    }

    fn parse_compare_op(&mut self) -> QueryResult<CompareOp> {
        self.skip_ws();
        let remaining = &self.input[self.pos..];

        if remaining.starts_with("==") {
            self.pos += 2;
            Ok(CompareOp::Eq)
        } else if remaining.starts_with("!=") {
            self.pos += 2;
            Ok(CompareOp::Ne)
        } else if remaining.starts_with("<=") {
            self.pos += 2;
            Ok(CompareOp::Le)
        } else if remaining.starts_with(">=") {
            self.pos += 2;
            Ok(CompareOp::Ge)
        } else if remaining.starts_with('<') {
            self.pos += 1;
            Ok(CompareOp::Lt)
        } else if remaining.starts_with('>') {
            self.pos += 1;
            Ok(CompareOp::Gt)
        } else {
            Err(self.error("expected comparison operator"))
        }
    }

    fn parse_vector_literal(&mut self) -> QueryResult<Vec<f32>> {
        self.expect_char('[')?;
        let mut values = Vec::new();

        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                self.advance();
                break;
            }

            let lit = self.parse_number()?;
            let val = match lit {
                Literal::Float(f) => f as f32,
                Literal::Int(i) => i as f32,
                _ => return Err(self.error("expected number in vector")),
            };
            values.push(val);

            self.skip_ws();
            if self.peek() == Some(',') {
                self.advance();
            }
        }

        Ok(values)
    }

    fn parse_query(&mut self) -> QueryResult<Query> {
        let source = self.parse_source()?;
        let mut steps = Vec::new();

        // Pipeline semantics: steps apply left-to-right, so `.limit(N).offset(M)`
        // means "take N then skip M" (= N-M rows) — almost always a pagination
        // mistake. Reject an offset that follows a limit on the SAME collection.
        // A `traverse`/`similar` in between changes which collection limit and
        // offset act on, so it resets the check. `.offset(M).limit(N)` reads as
        // "skip M, take N" and yields the expected page.
        let mut saw_limit = false;
        loop {
            self.skip_ws();
            if self.peek() != Some('.') {
                break;
            }

            self.advance(); // consume '.'

            self.skip_ws();
            let ident = self.parse_ident()?;

            self.skip_ws();
            let step = if self.peek() == Some('(') {
                self.parse_function_step(&ident)?
            } else {
                // Bare identifier = relationship traversal.
                Step::Traverse {
                    field_name: ident,
                }
            };

            match &step {
                Step::Limit { .. } => saw_limit = true,
                Step::Traverse { .. } | Step::Similar { .. } => saw_limit = false,
                Step::Offset { .. } if saw_limit => {
                    return Err(self.error(
                        "`.offset(...)` must come before `.limit(...)`: `.limit(N).offset(M)` \
                         takes N rows then skips M. For pagination use `.offset(M).limit(N)`.",
                    ));
                }
                _ => {}
            }

            steps.push(step);
        }

        Ok(Query { source, steps })
    }

    fn parse_source(&mut self) -> QueryResult<Source> {
        let type_name = self.parse_ident()?;

        self.skip_ws();
        if self.peek() != Some('.') {
            return Ok(Source::All { type_name });
        }

        let saved = self.pos;
        self.advance(); // consume '.'

        let method = self.parse_ident()?;
        self.skip_ws();

        match method.as_str() {
            "get" => {
                self.expect_char('(')?;
                let id = self.parse_int()? as u64;
                self.expect_char(')')?;
                Ok(Source::Get { type_name, id })
            }
            "filter" => {
                self.expect_char('(')?;
                let predicate = self.parse_predicate()?;
                self.expect_char(')')?;
                Ok(Source::Filter {
                    type_name,
                    predicate,
                })
            }
            "create" => {
                self.expect_char('(')?;
                let fields = self.parse_object()?;
                self.expect_char(')')?;
                Ok(Source::Create { type_name, fields })
            }
            "create_batch" => {
                // Type.create_batch([{...}, {...}, ...])
                self.expect_char('(')?;
                self.skip_ws();
                self.expect_char('[')?;
                let mut rows = Vec::new();
                self.skip_ws();
                if self.peek() != Some(']') {
                    rows.push(self.parse_object()?);
                    self.skip_ws();
                    while self.peek() == Some(',') {
                        self.advance();
                        self.skip_ws();
                        // Allow a trailing comma before ']'.
                        if self.peek() == Some(']') {
                            break;
                        }
                        rows.push(self.parse_object()?);
                        self.skip_ws();
                    }
                }
                self.expect_char(']')?;
                self.skip_ws();
                self.expect_char(')')?;
                Ok(Source::CreateBatch { type_name, rows })
            }
            _ => {
                // Not a source-level method — backtrack and treat as Type.traverse.
                self.pos = saved;
                Ok(Source::All { type_name })
            }
        }
    }

    fn parse_function_step(&mut self, name: &str) -> QueryResult<Step> {
        match name {
            "filter" => {
                self.expect_char('(')?;
                let predicate = self.parse_predicate()?;
                self.expect_char(')')?;
                Ok(Step::Filter { predicate })
            }
            "similar" => {
                self.expect_char('(')?;
                self.expect_char('.')?;
                let field_name = self.parse_ident()?;
                self.expect_char(',')?;

                // Detect text string vs raw vector.
                self.skip_ws();
                let query = if self.peek() == Some('"') {
                    SimilarQuery::Text(self.parse_string_literal()?)
                } else {
                    SimilarQuery::Vector(self.parse_vector_literal()?)
                };

                self.expect_char(',')?;
                self.skip_ws();
                self.expect_str("k:")?;
                let k = self.parse_count()?;

                // Optional trailing named params: `, ef: N` and `, rerank: N`,
                // in any order. A bare `.similar(.f, q, k: N)` parses exactly as
                // before (the loop sees no comma and exits immediately).
                let mut ef = None;
                let mut rerank = None;
                loop {
                    self.skip_ws();
                    if self.peek() != Some(',') {
                        break;
                    }
                    self.advance(); // consume ','
                    self.skip_ws();
                    if self.peek() == Some(')') {
                        break; // tolerate a trailing comma
                    }
                    let name = self.parse_ident()?;
                    self.skip_ws();
                    self.expect_char(':')?;
                    let n = self.parse_count()?;
                    match name.as_str() {
                        "ef" => {
                            if n == 0 {
                                return Err(self.error("ef must be >= 1"));
                            }
                            ef = Some(n);
                        }
                        "rerank" => rerank = Some(n),
                        other => {
                            return Err(self.error(format!(
                                "unknown similar() parameter '{other}' \
                                 (expected 'ef' or 'rerank')"
                            )));
                        }
                    }
                }
                self.expect_char(')')?;
                Ok(Step::Similar {
                    field_name,
                    query,
                    k,
                    ef,
                    rerank,
                })
            }
            "update" => {
                self.expect_char('(')?;
                let fields = self.parse_object()?;
                self.expect_char(')')?;
                Ok(Step::Update { fields })
            }
            "delete" => {
                self.expect_char('(')?;
                self.expect_char(')')?;
                Ok(Step::Delete)
            }
            "link" => {
                self.expect_char('(')?;
                let target_type = self.parse_ident()?;
                self.expect_char('.')?;
                self.expect_str("get")?;
                self.expect_char('(')?;
                let target_id = self.parse_int()? as u64;
                self.expect_char(')')?;

                self.skip_ws();
                let edge_fields = if self.peek() == Some(',') {
                    self.advance();
                    self.parse_object()?
                } else {
                    HashMap::new()
                };
                self.expect_char(')')?;

                Ok(Step::Link {
                    target_type,
                    target_id,
                    edge_fields,
                })
            }
            "unlink" => {
                self.expect_char('(')?;
                let target_type = self.parse_ident()?;
                self.expect_char('.')?;
                self.expect_str("get")?;
                self.expect_char('(')?;
                let target_id = self.parse_int()? as u64;
                self.expect_char(')')?;
                self.expect_char(')')?;
                Ok(Step::Unlink {
                    target_type,
                    target_id,
                })
            }
            "limit" => {
                self.expect_char('(')?;
                let count = self.parse_count()?;
                self.expect_char(')')?;
                Ok(Step::Limit { count })
            }
            "offset" => {
                self.expect_char('(')?;
                let count = self.parse_count()?;
                self.expect_char(')')?;
                Ok(Step::Offset { count })
            }
            _ => Err(QueryError::UnknownFunction(name.into())),
        }
    }

    fn error(&self, msg: impl Into<String>) -> QueryError {
        QueryError::Parse {
            pos: self.pos,
            message: msg.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_get() {
        let q = parse_query("User.get(42)").unwrap();
        assert_eq!(
            q.source,
            Source::Get {
                type_name: "User".into(),
                id: 42
            }
        );
        assert!(q.steps.is_empty());
    }

    #[test]
    fn parse_filter() {
        let q = parse_query("User.filter(.age > 18)").unwrap();
        match &q.source {
            Source::Filter { type_name, predicate } => {
                assert_eq!(type_name, "User");
                match predicate {
                    Predicate::Compare { field_path, op, value } => {
                        assert_eq!(field_path, "age");
                        assert_eq!(*op, CompareOp::Gt);
                        assert_eq!(*value, Literal::Int(18));
                    }
                    _ => panic!("expected Compare"),
                }
            }
            _ => panic!("expected Filter source"),
        }
    }

    #[test]
    fn parse_create() {
        let q = parse_query(r#"User.create({ name: "Alice", age: 30 })"#).unwrap();
        match &q.source {
            Source::Create { type_name, fields } => {
                assert_eq!(type_name, "User");
                assert_eq!(fields.get("name"), Some(&Literal::String("Alice".into())));
                assert_eq!(fields.get("age"), Some(&Literal::Int(30)));
            }
            _ => panic!("expected Create source"),
        }
    }

    #[test]
    fn parse_traverse() {
        let q = parse_query("User.get(1).friends").unwrap();
        assert_eq!(q.steps.len(), 1);
        assert_eq!(
            q.steps[0],
            Step::Traverse {
                field_name: "friends".into()
            }
        );
    }

    #[test]
    fn parse_chained_traverse_and_filter() {
        let q = parse_query(r#"User.get(1).friends.posts.filter(.title == "hello")"#).unwrap();
        assert_eq!(q.steps.len(), 3);
        assert_eq!(
            q.steps[0],
            Step::Traverse {
                field_name: "friends".into()
            }
        );
        assert_eq!(
            q.steps[1],
            Step::Traverse {
                field_name: "posts".into()
            }
        );
        match &q.steps[2] {
            Step::Filter { predicate } => match predicate {
                Predicate::Compare { field_path, op, value } => {
                    assert_eq!(field_path, "title");
                    assert_eq!(*op, CompareOp::Eq);
                    assert_eq!(*value, Literal::String("hello".into()));
                }
                _ => panic!("expected Compare"),
            },
            _ => panic!("expected Filter step"),
        }
    }

    #[test]
    fn parse_similar_with_vector() {
        let q = parse_query("Post.similar(.embedding, [1.0, 2.0, 3.0], k: 10)").unwrap();
        assert_eq!(q.steps.len(), 1);
        match &q.steps[0] {
            Step::Similar {
                field_name,
                query,
                k,
                ef,
                rerank,
            } => {
                assert_eq!(field_name, "embedding");
                assert_eq!(*query, SimilarQuery::Vector(vec![1.0, 2.0, 3.0]));
                assert_eq!(*k, 10);
                assert_eq!(*ef, None);
                assert_eq!(*rerank, None);
            }
            _ => panic!("expected Similar step"),
        }
    }

    #[test]
    fn parse_similar_with_text() {
        let q = parse_query(r#"Post.similar(.embedding, "distributed systems", k: 5)"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        match &q.steps[0] {
            Step::Similar {
                field_name,
                query,
                k,
                ef,
                rerank,
            } => {
                assert_eq!(field_name, "embedding");
                assert_eq!(
                    *query,
                    SimilarQuery::Text("distributed systems".into())
                );
                assert_eq!(*k, 5);
                assert_eq!(*ef, None);
                assert_eq!(*rerank, None);
            }
            _ => panic!("expected Similar step"),
        }
    }

    #[test]
    fn parse_similar_with_ef_and_rerank() {
        // Both params, vector query.
        let q = parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, ef: 200, rerank: 50)")
            .unwrap();
        match &q.steps[0] {
            Step::Similar { k, ef, rerank, .. } => {
                assert_eq!(*k, 10);
                assert_eq!(*ef, Some(200));
                assert_eq!(*rerank, Some(50));
            }
            _ => panic!("expected Similar step"),
        }

        // Reversed order, text query, rerank then ef.
        let q = parse_query(r#"Post.similar(.embedding, "hello", k: 3, rerank: 40, ef: 128)"#)
            .unwrap();
        match &q.steps[0] {
            Step::Similar { ef, rerank, .. } => {
                assert_eq!(*ef, Some(128));
                assert_eq!(*rerank, Some(40));
            }
            _ => panic!("expected Similar step"),
        }

        // Only ef.
        let q = parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, ef: 75)").unwrap();
        match &q.steps[0] {
            Step::Similar { ef, rerank, .. } => {
                assert_eq!(*ef, Some(75));
                assert_eq!(*rerank, None);
            }
            _ => panic!("expected Similar step"),
        }

        // Trailing comma is tolerated.
        let q = parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, ef: 75,)").unwrap();
        assert!(matches!(&q.steps[0], Step::Similar { ef: Some(75), .. }));
    }

    #[test]
    fn parse_similar_rejects_bad_params() {
        // Unknown parameter name.
        assert!(parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, foo: 5)").is_err());
        // ef must be >= 1.
        assert!(parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, ef: 0)").is_err());
        // Negative count is rejected by parse_count.
        assert!(parse_query("Post.similar(.embedding, [1.0, 0.0], k: 10, rerank: -1)").is_err());
    }

    #[test]
    fn parse_update() {
        let q = parse_query(r#"User.get(1).update({ name: "Bob" })"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        match &q.steps[0] {
            Step::Update { fields } => {
                assert_eq!(fields.get("name"), Some(&Literal::String("Bob".into())));
            }
            _ => panic!("expected Update step"),
        }
    }

    #[test]
    fn parse_delete() {
        let q = parse_query("Movie.get(42).delete()").unwrap();
        assert_eq!(q.steps.len(), 1);
        assert_eq!(q.steps[0], Step::Delete);
    }

    #[test]
    fn parse_link_with_edge_props() {
        let q = parse_query(
            r#"User.get(1).favorite_movies.link(Movie.get(42), { rating: 4.5 })"#,
        )
        .unwrap();
        assert_eq!(q.steps.len(), 2);
        assert_eq!(
            q.steps[0],
            Step::Traverse {
                field_name: "favorite_movies".into()
            }
        );
        match &q.steps[1] {
            Step::Link {
                target_type,
                target_id,
                edge_fields,
            } => {
                assert_eq!(target_type, "Movie");
                assert_eq!(*target_id, 42);
                assert_eq!(edge_fields.get("rating"), Some(&Literal::Float(4.5)));
            }
            _ => panic!("expected Link step"),
        }
    }

    #[test]
    fn parse_unlink() {
        let q = parse_query("User.get(1).friends.unlink(User.get(5))").unwrap();
        assert_eq!(q.steps.len(), 2);
        match &q.steps[1] {
            Step::Unlink {
                target_type,
                target_id,
            } => {
                assert_eq!(target_type, "User");
                assert_eq!(*target_id, 5);
            }
            _ => panic!("expected Unlink step"),
        }
    }

    #[test]
    fn parse_limit_offset() {
        // Pagination idiom: offset before limit ("skip 20, take 10").
        let q = parse_query("User.filter(.age > 18).offset(20).limit(10)").unwrap();
        assert_eq!(q.steps.len(), 2);
        assert_eq!(q.steps[0], Step::Offset { count: 20 });
        assert_eq!(q.steps[1], Step::Limit { count: 10 });
    }

    #[test]
    fn limit_before_offset_is_rejected() {
        // `.limit(N).offset(M)` is the pagination footgun under sequential
        // pipeline semantics (take N then skip M = N-M rows); the parser
        // rejects it and points at the `.offset(M).limit(N)` idiom.
        let err = parse_query("User.filter(.age > 18).limit(10).offset(20)").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("offset") && msg.contains("limit"),
            "expected an offset/limit ordering error, got: {msg}"
        );
    }

    #[test]
    fn limit_then_traverse_then_offset_is_allowed() {
        // A traverse between limit and offset changes which collection they act
        // on ("take 10 users, go to friends, skip 5 friends"), so it is NOT the
        // pagination footgun and must parse.
        let q = parse_query("User.filter(.age > 18).limit(10).friends.offset(5)").unwrap();
        assert_eq!(q.steps.len(), 3);
        assert_eq!(q.steps[0], Step::Limit { count: 10 });
        assert!(matches!(q.steps[1], Step::Traverse { .. }));
        assert_eq!(q.steps[2], Step::Offset { count: 5 });
    }

    #[test]
    fn negative_count_is_rejected() {
        // limit/offset/k wrap to a huge usize if parsed as a raw cast; reject.
        assert!(parse_query("User.limit(-1)").is_err());
        assert!(parse_query("User.offset(-5)").is_err());
    }

    #[test]
    fn predicate_and_binds_tighter_than_or() {
        // `a && b || c` parses as `Or(And(a, b), c)` (standard precedence).
        let q = parse_query("User.filter(.a > 1 && .b > 2 || .c > 3)").unwrap();
        let pred = match &q.source {
            Source::Filter { predicate, .. } => predicate,
            _ => panic!("expected filter source"),
        };
        match pred {
            Predicate::Or(left, right) => {
                assert!(matches!(**left, Predicate::And(_, _)), "left of Or should be And(a,b), got {left:?}");
                assert!(matches!(**right, Predicate::Compare { .. }), "right of Or should be c, got {right:?}");
            }
            other => panic!("expected Or at top, got {other:?}"),
        }
    }

    #[test]
    fn predicate_or_then_and_groups_correctly() {
        // `a || b && c` parses as `Or(a, And(b, c))`.
        let q = parse_query("User.filter(.a > 1 || .b > 2 && .c > 3)").unwrap();
        let pred = match &q.source {
            Source::Filter { predicate, .. } => predicate,
            _ => panic!("expected filter source"),
        };
        match pred {
            Predicate::Or(left, right) => {
                assert!(matches!(**left, Predicate::Compare { .. }), "left of Or should be a, got {left:?}");
                assert!(matches!(**right, Predicate::And(_, _)), "right of Or should be And(b,c), got {right:?}");
            }
            other => panic!("expected Or at top, got {other:?}"),
        }
    }

    #[test]
    fn predicate_parentheses_override_precedence() {
        // `(a || b) && c` parses as `And(Or(a, b), c)`.
        let q = parse_query("User.filter((.a > 1 || .b > 2) && .c > 3)").unwrap();
        let pred = match &q.source {
            Source::Filter { predicate, .. } => predicate,
            _ => panic!("expected filter source"),
        };
        match pred {
            Predicate::And(left, right) => {
                assert!(matches!(**left, Predicate::Or(_, _)), "left of And should be Or(a,b), got {left:?}");
                assert!(matches!(**right, Predicate::Compare { .. }), "right of And should be c, got {right:?}");
            }
            other => panic!("expected And at top, got {other:?}"),
        }
    }

    #[test]
    fn parse_boolean_literal() {
        let q = parse_query("User.filter(.active == true)").unwrap();
        match &q.source {
            Source::Filter { predicate, .. } => match predicate {
                Predicate::Compare { value, .. } => {
                    assert_eq!(*value, Literal::Bool(true));
                }
                _ => panic!("expected Compare"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parse_nested_field_path() {
        let q = parse_query(r#"User.get(1).friends.posts.filter(.tags.name == "rust")"#).unwrap();
        match &q.steps[2] {
            Step::Filter { predicate } => match predicate {
                Predicate::Compare { field_path, .. } => {
                    assert_eq!(field_path, "tags.name");
                }
                _ => panic!("expected Compare"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parse_and_predicate() {
        let q = parse_query("User.filter(.age > 18 && .age < 65)").unwrap();
        match &q.source {
            Source::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::And(_, _)));
            }
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parse_negative_number() {
        let q = parse_query("User.filter(.score > -10)").unwrap();
        match &q.source {
            Source::Filter { predicate, .. } => match predicate {
                Predicate::Compare { value, .. } => {
                    assert_eq!(*value, Literal::Int(-10));
                }
                _ => panic!("expected Compare"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parse_float_literal() {
        let q = parse_query("User.filter(.score > 2.5)").unwrap();
        match &q.source {
            Source::Filter { predicate, .. } => match predicate {
                Predicate::Compare { value, .. } => {
                    assert_eq!(*value, Literal::Float(2.5));
                }
                _ => panic!("expected Compare"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parse_complex_query() {
        let q = parse_query(
            r#"User.get(1).friends.favorite_movies.filter(.genre == "sci-fi").similar(.embedding, [0.1, 0.2], k: 5).limit(10)"#,
        )
        .unwrap();

        assert_eq!(
            q.source,
            Source::Get {
                type_name: "User".into(),
                id: 1
            }
        );
        assert_eq!(q.steps.len(), 5);
        assert!(matches!(&q.steps[0], Step::Traverse { field_name } if field_name == "friends"));
        assert!(matches!(&q.steps[1], Step::Traverse { field_name } if field_name == "favorite_movies"));
        assert!(matches!(&q.steps[2], Step::Filter { .. }));
        assert!(matches!(&q.steps[3], Step::Similar { k: 5, .. }));
        assert_eq!(q.steps[4], Step::Limit { count: 10 });
    }
}
