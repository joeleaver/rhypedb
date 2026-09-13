//! The `.matches` query mini-language.
//!
//! ```text
//! query   = clause*
//! clause  = ["+"] ( '"' phrase-text '"' | bare-text )
//! ```
//!
//! * Bare text is analyzed with the field's analyzer; every token it yields
//!   becomes its own **term clause** (so `invoice-4471` is the two terms
//!   `invoice`, `4471`). Term clauses are OR-ed: a document matches when it
//!   contains any of them, and each one it contains adds to its score.
//! * `+` marks a clause **required**: a document must match every required
//!   clause. Required clauses still contribute to the score.
//! * `"quoted words"` is a **phrase clause**: the analyzed tokens must occur
//!   consecutively in the document (needs stored positions). A phrase that
//!   analyzes to a single token is just a term clause.
//! * Clauses that analyze to nothing (punctuation, emoji) are dropped; a
//!   query left with no clauses is an error rather than a silent empty set.

use super::analyzer::Analyzer;

/// One parsed clause. `terms.len() == 1` is a term clause, `> 1` a phrase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub required: bool,
    pub terms: Vec<String>,
}

impl Clause {
    pub fn is_phrase(&self) -> bool {
        self.terms.len() > 1
    }
}

/// A parsed, analyzed query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQuery {
    pub clauses: Vec<Clause>,
}

impl ParsedQuery {
    pub fn has_required(&self) -> bool {
        self.clauses.iter().any(|c| c.required)
    }

    pub fn needs_positions(&self) -> bool {
        self.clauses.iter().any(Clause::is_phrase)
    }

    /// Every distinct term across all clauses, in first-seen order.
    pub fn distinct_terms(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for c in &self.clauses {
            for t in &c.terms {
                if !out.contains(&t.as_str()) {
                    out.push(t.as_str());
                }
            }
        }
        out
    }
}

/// Query-syntax errors. Rendered verbatim to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuerySyntaxError {
    /// A `"` was opened and never closed.
    UnterminatedPhrase,
    /// After analysis nothing searchable remained.
    NoSearchableTerms,
}

impl std::fmt::Display for QuerySyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedPhrase => write!(f, "unterminated phrase (missing closing quote)"),
            Self::NoSearchableTerms => write!(
                f,
                "query contains no searchable terms (only punctuation, or every clause was empty)"
            ),
        }
    }
}

/// Parse + analyze a `.matches` query string.
pub fn parse_query(raw: &str, analyzer: Analyzer) -> Result<ParsedQuery, QuerySyntaxError> {
    let mut clauses = Vec::new();
    let mut chars = raw.char_indices().peekable();
    while let Some(&(i, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut required = false;
        let mut start = i;
        if c == '+' {
            required = true;
            chars.next();
            // `+` followed by whitespace/end is a stray marker: skip it.
            match chars.peek() {
                Some(&(j, nc)) if !nc.is_whitespace() => start = j,
                _ => continue,
            }
        }
        let (_, first) = *chars.peek().expect("peeked non-whitespace");
        if first == '"' {
            chars.next();
            let text_start = start + 1;
            let mut text_end = None;
            for (j, ch) in chars.by_ref() {
                if ch == '"' {
                    text_end = Some(j);
                    break;
                }
            }
            let Some(end) = text_end else {
                return Err(QuerySyntaxError::UnterminatedPhrase);
            };
            let tokens = analyzer.analyze(&raw[text_start..end]);
            let terms: Vec<String> = tokens.into_iter().map(|t| t.term).collect();
            if !terms.is_empty() {
                clauses.push(Clause { required, terms });
            }
        } else {
            let mut end = raw.len();
            while let Some(&(j, ch)) = chars.peek() {
                if ch.is_whitespace() || ch == '"' {
                    end = j;
                    break;
                }
                chars.next();
            }
            // A bare word is one OR-ed term per analyzed token (no implicit
            // phrase — `e-mail` finds documents with either word).
            for t in analyzer.analyze(&raw[start..end]) {
                clauses.push(Clause {
                    required,
                    terms: vec![t.term],
                });
            }
        }
    }
    if clauses.is_empty() {
        return Err(QuerySyntaxError::NoSearchableTerms);
    }
    Ok(ParsedQuery { clauses })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &str) -> ParsedQuery {
        parse_query(raw, Analyzer::Simple).unwrap()
    }
    fn term(t: &str, required: bool) -> Clause {
        Clause {
            required,
            terms: vec![t.into()],
        }
    }
    fn phrase(ts: &[&str], required: bool) -> Clause {
        Clause {
            required,
            terms: ts.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn bare_terms_are_optional_and_analyzed() {
        let q = parse("Invoice 4471 Café");
        assert_eq!(
            q.clauses,
            vec![term("invoice", false), term("4471", false), term("cafe", false)]
        );
        assert!(!q.has_required());
        assert!(!q.needs_positions());
        assert_eq!(q.distinct_terms(), vec!["invoice", "4471", "cafe"]);
    }

    #[test]
    fn plus_marks_required_and_hyphenated_words_split_into_terms() {
        let q = parse("+invoice draft +e-mail");
        assert_eq!(
            q.clauses,
            vec![
                term("invoice", true),
                term("draft", false),
                term("e", true),
                term("mail", true)
            ]
        );
        assert!(q.has_required());
        // A stray `+` is ignored; `++x` is still just required x.
        assert_eq!(parse("+ x ++y").clauses, vec![term("x", false), term("y", true)]);
    }

    #[test]
    fn quoted_text_is_a_phrase() {
        let q = parse(r#"+"Distributed Consensus" raft "single""#);
        assert_eq!(
            q.clauses,
            vec![
                phrase(&["distributed", "consensus"], true),
                term("raft", false),
                // One-token phrase degrades to a term clause.
                term("single", false),
            ]
        );
        assert!(q.needs_positions());
        // Adjacent phrase without a space before it still parses.
        assert_eq!(
            parse(r#"a"b c"d"#).clauses,
            vec![term("a", false), phrase(&["b", "c"], false), term("d", false)]
        );
        assert_eq!(parse(r#""" x"#).clauses, vec![term("x", false)]);
    }

    #[test]
    fn distinct_terms_dedupes_across_clauses() {
        let q = parse(r#"a "a b" b +a"#);
        assert_eq!(q.distinct_terms(), vec!["a", "b"]);
    }

    #[test]
    fn errors_on_unterminated_phrase_and_empty_query() {
        assert_eq!(
            parse_query(r#"open "phrase"#, Analyzer::Simple),
            Err(QuerySyntaxError::UnterminatedPhrase)
        );
        for raw in ["", "   ", "... !!!", "+", "\"\"", "\"...\"", "🎉"] {
            assert_eq!(
                parse_query(raw, Analyzer::Simple),
                Err(QuerySyntaxError::NoSearchableTerms),
                "{raw:?}"
            );
        }
    }
}
