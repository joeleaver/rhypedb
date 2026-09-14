//! The `.matches` query mini-language.
//!
//! ```text
//! query   = clause*
//! clause  = ["+"] ( '"' phrase-text '"' | bare-text ["*"] )
//! ```
//!
//! * Bare text is analyzed with the field's analyzer; every token it yields
//!   becomes its own **term clause** (so `invoice-4471` is the two terms
//!   `invoice`, `4471`). Term clauses are OR-ed: a document matches when it
//!   contains any of them, and each one it contains adds to its score.
//! * `+` marks a clause **required**: a document must match every required
//!   clause. Required clauses still contribute to the score.
//! * `"quoted words"` is a **phrase clause**: the analyzed tokens must occur
//!   in the document at the same relative positions they have in the query
//!   (needs stored positions) — consecutively for plain words, and with the
//!   same gaps where the analyzer dropped a word (a stop word under
//!   `english`: `"state of the art"` is `state` +0, `art` +3). A phrase that
//!   analyzes to a single token is just a term clause.
//! * Bare text ending in `*` is a **prefix clause** (`camera*`): it matches
//!   every indexed term that starts with the analyzed text before the `*`
//!   — analyzed like any other term, so under `english` `cameras*` is the
//!   prefix `camera`, which is what the index holds. The expansion scores
//!   as ONE term (its postings merged, see `search`), so a rare misspelling
//!   among the expansions cannot dominate. The prefix must be at least two
//!   characters after analysis; `*` alone, a one-character prefix, and a
//!   `*` inside a phrase are errors. Bare text that analyzes to several
//!   tokens (`e-mail*`) yields plain terms for all but the last, which is
//!   the prefix. Under a stemming analyzer the prefix is stemmed too and
//!   matched against stored STEMS, so a partial word whose stem is shorter
//!   than what was typed (`securit*` vs the stem `secur`) finds nothing —
//!   an inherent stem+prefix trade-off (a dictionary sweep put stemming the
//!   prefix at 88% of partial prefixes matched vs 83% for leaving it
//!   unstemmed, so this is the better single strategy); `simple` gives
//!   exact character-by-character type-ahead.
//! * Clauses that analyze to nothing (punctuation, emoji) are dropped; a
//!   query left with no clauses is an error rather than a silent empty set —
//!   UNLESS what was dropped were stop words (`of my`, under `english`): the
//!   user typed real words, so that is an empty RESULT, not a syntax error.
//!   A required stop word (`+of`) is dropped like any other and constrains
//!   nothing.

use super::analyzer::Analyzer;

/// Shortest prefix (in characters, after analysis) a prefix clause accepts.
pub const MIN_PREFIX_CHARS: usize = 2;

/// One parsed clause. `terms.len() == 1` is a term clause (or, with
/// `prefix`, a prefix clause), `> 1` a phrase.
///
/// A prefix clause's single entry is its **posting key**: the analyzed
/// prefix with the `*` kept (`"camera*"`). The key is what the search layer
/// looks the merged expansion up by, and it can never collide with a real
/// term — `*` is punctuation to UAX#29, so no analyzer emits it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub required: bool,
    pub terms: Vec<String>,
    /// `terms[0]` is a prefix key (`"camera*"`), not an exact term.
    pub prefix: bool,
    /// For a phrase: each term's position relative to the first
    /// (`offsets[0] == 0`, strictly increasing) — consecutive words are
    /// `0, 1, 2, …`; a gap marks a word the analyzer dropped between them.
    /// Empty for term and prefix clauses.
    pub offsets: Vec<u32>,
}

impl Clause {
    pub fn is_phrase(&self) -> bool {
        self.terms.len() > 1
    }

    /// A term clause.
    pub fn term(term: String, required: bool) -> Self {
        Self {
            required,
            terms: vec![term],
            prefix: false,
            offsets: Vec::new(),
        }
    }

    /// A phrase clause with consecutive offsets `0, 1, …`.
    pub fn phrase(terms: Vec<String>, required: bool) -> Self {
        let offsets = (0..terms.len() as u32).collect();
        Self {
            required,
            terms,
            prefix: false,
            offsets,
        }
    }

    /// The relative position of the `i`-th phrase term (`i` itself when the
    /// clause was built without explicit offsets).
    pub fn offset(&self, i: usize) -> u32 {
        self.offsets.get(i).copied().unwrap_or(i as u32)
    }

    /// The analyzed prefix text of a prefix clause (`"camera"` for the key
    /// `"camera*"`); `None` for term and phrase clauses.
    pub fn prefix_text(&self) -> Option<&str> {
        if self.prefix {
            self.terms[0].strip_suffix('*')
        } else {
            None
        }
    }
}

/// A parsed, analyzed query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQuery {
    pub clauses: Vec<Clause>,
}

impl ParsedQuery {
    /// No clauses at all: every word was a stop word. Matches nothing.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    pub fn has_required(&self) -> bool {
        self.clauses.iter().any(|c| c.required)
    }

    pub fn needs_positions(&self) -> bool {
        self.clauses.iter().any(Clause::is_phrase)
    }

    /// Whether any clause is a prefix clause.
    pub fn has_prefix(&self) -> bool {
        self.clauses.iter().any(|c| c.prefix)
    }

    /// Every distinct posting key across all clauses, in first-seen order:
    /// exact terms, plus one `"prefix*"` key per distinct prefix clause.
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
    /// A prefix clause whose analyzed prefix is shorter than
    /// [`MIN_PREFIX_CHARS`] — includes a bare `*`, text that analyzes to
    /// nothing (`!!*`, an over-long word), and multi-word text whose LAST
    /// word is short (`東京*`: one ideograph per word, so the prefix is `京`).
    /// `raw` is the text before the `*` (truncated for the message),
    /// `prefix` the analyzed last word (possibly empty).
    PrefixTooShort { raw: String, prefix: String },
    /// A `*` inside a quoted phrase; phrases take exact terms only.
    PrefixInPhrase { phrase: String },
}

impl std::fmt::Display for QuerySyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedPhrase => write!(f, "unterminated phrase (missing closing quote)"),
            Self::NoSearchableTerms => write!(
                f,
                "query contains no searchable terms (only punctuation, or every clause was empty)"
            ),
            Self::PrefixTooShort { raw, prefix } => {
                let shown: String = if raw.chars().count() > 40 {
                    format!("{}…", raw.chars().take(40).collect::<String>())
                } else {
                    raw.clone()
                };
                let n = prefix.chars().count();
                write!(
                    f,
                    "prefix term \"{shown}*\" is too short: the word before the * analyzes to \
                     \"{prefix}\" ({n} character{}); a prefix needs at least {MIN_PREFIX_CHARS}",
                    if n == 1 { "" } else { "s" },
                )
            }
            Self::PrefixInPhrase { phrase } => write!(
                f,
                "prefix terms (word*) are not supported inside a phrase: \"{phrase}\""
            ),
        }
    }
}

/// Parse + analyze a `.matches` query string.
pub fn parse_query(raw: &str, analyzer: Analyzer) -> Result<ParsedQuery, QuerySyntaxError> {
    let mut clauses = Vec::new();
    let mut stop_words_dropped = 0u32;
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
            let text = &raw[text_start..end];
            if text.contains('*') {
                // The analyzer would silently drop the `*` (it is
                // punctuation) and the user would get an exact phrase they
                // did not ask for.
                return Err(QuerySyntaxError::PrefixInPhrase {
                    phrase: text.to_string(),
                });
            }
            let analyzed = analyzer.analyze_opts(text, false);
            stop_words_dropped += analyzed.stop_words_dropped;
            let tokens = analyzed.tokens;
            if let Some(first) = tokens.first() {
                // Offsets are relative to the first KEPT token: a leading
                // dropped word shifts nothing, an inner one leaves a gap.
                let base = first.position;
                let offsets: Vec<u32> = tokens.iter().map(|t| t.position - base).collect();
                let terms: Vec<String> = tokens.into_iter().map(|t| t.term).collect();
                clauses.push(Clause {
                    required,
                    terms,
                    prefix: false,
                    offsets: if offsets.len() > 1 { offsets } else { Vec::new() },
                });
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
            let word = &raw[start..end];
            // A trailing `*` (one or more) makes the LAST analyzed token a
            // prefix clause. Anywhere else `*` is punctuation to the
            // analyzer and simply splits words (`cam*era` → `cam`, `era`).
            let stem = word.trim_end_matches('*');
            let is_prefix = stem.len() != word.len();
            // The prefix text keeps stop words (`the*` asks for words that
            // START with "the"); only the plain words before it drop them.
            let analyzed = analyzer.analyze_opts(stem, is_prefix);
            stop_words_dropped += analyzed.stop_words_dropped;
            let mut tokens = analyzed.tokens;
            let prefix_token = if is_prefix {
                let last = tokens.pop();
                match last {
                    Some(t) if t.term.chars().count() >= MIN_PREFIX_CHARS => Some(t.term),
                    other => {
                        return Err(QuerySyntaxError::PrefixTooShort {
                            raw: stem.to_string(),
                            prefix: other.map(|t| t.term).unwrap_or_default(),
                        });
                    }
                }
            } else {
                None
            };
            // A bare word is one OR-ed term per analyzed token (no implicit
            // phrase — `e-mail` finds documents with either word).
            for t in tokens {
                clauses.push(Clause::term(t.term, required));
            }
            if let Some(prefix) = prefix_token {
                clauses.push(Clause {
                    required,
                    terms: vec![format!("{prefix}*")],
                    prefix: true,
                    offsets: Vec::new(),
                });
            }
        }
    }
    if clauses.is_empty() && stop_words_dropped == 0 {
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
        Clause::term(t.into(), required)
    }
    fn phrase(ts: &[&str], required: bool) -> Clause {
        Clause::phrase(ts.iter().map(|s| s.to_string()).collect(), required)
    }
    fn prefix(p: &str, required: bool) -> Clause {
        Clause {
            required,
            terms: vec![format!("{p}*")],
            prefix: true,
            offsets: Vec::new(),
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
    fn trailing_star_makes_a_prefix_clause() {
        let q = parse("cam* +Invoice* draft");
        assert_eq!(
            q.clauses,
            vec![prefix("cam", false), prefix("invoice", true), term("draft", false)]
        );
        assert!(q.has_prefix());
        assert!(q.has_required());
        assert!(!q.needs_positions());
        assert_eq!(q.clauses[0].prefix_text(), Some("cam"));
        assert_eq!(q.clauses[2].prefix_text(), None);
        // The key carries the star, so a term and a prefix of the same text
        // are distinct posting keys.
        assert_eq!(parse("cam cam*").distinct_terms(), vec!["cam", "cam*"]);
        assert_eq!(parse("cam* cam*").distinct_terms(), vec!["cam*"]);
        // The prefix is analyzed: folded like a term.
        assert_eq!(parse("CAFÉ*").clauses, vec![prefix("cafe", false)]);
        assert_eq!(
            parse_query("Cameras*", Analyzer::English).unwrap().clauses,
            vec![prefix("camera", false)]
        );
        // Several stars are one star; multi-token text: last token is the prefix.
        assert_eq!(parse("cam**").clauses, vec![prefix("cam", false)]);
        assert_eq!(parse("e-mail*").clauses, vec![term("e", false), prefix("mail", false)]);
        // A star that is not trailing is just punctuation (splits the word).
        assert_eq!(parse("cam*era").clauses, vec![term("cam", false), term("era", false)]);
        // Exactly the minimum length is accepted; a stray `+` before is still stray.
        assert_eq!(parse("ca*").clauses, vec![prefix("ca", false)]);
        assert_eq!(parse("+ ca*").clauses, vec![prefix("ca", false)]);
    }

    #[test]
    fn prefix_errors_are_loud() {
        for raw in ["*", "**", "c*", "é*", "!!*", "+*", "x c*", "\"ok\" *"] {
            assert!(
                matches!(parse_query(raw, Analyzer::Simple), Err(QuerySyntaxError::PrefixTooShort { .. })),
                "{raw:?} → {:?}",
                parse_query(raw, Analyzer::Simple)
            );
        }
        let err = parse_query("c*", Analyzer::Simple).unwrap_err();
        assert_eq!(
            err.to_string(),
            "prefix term \"c*\" is too short: the word before the * analyzes to \"c\" (1 character); a prefix needs at least 2"
        );
        // Multi-word text names the analyzed LAST word, so a two-ideograph
        // CJK prefix explains itself; nothing-analyzable says so too.
        assert_eq!(
            parse_query("東京*", Analyzer::Simple).unwrap_err().to_string(),
            "prefix term \"東京*\" is too short: the word before the * analyzes to \"京\" (1 character); a prefix needs at least 2"
        );
        assert_eq!(
            parse_query("!!*", Analyzer::Simple).unwrap_err().to_string(),
            "prefix term \"!!*\" is too short: the word before the * analyzes to \"\" (0 characters); a prefix needs at least 2"
        );
        // An over-long word (dropped by the analyzer) is echoed truncated.
        let long = "x".repeat(300);
        let msg = parse_query(&format!("{long}*"), Analyzer::Simple).unwrap_err().to_string();
        assert!(msg.starts_with(&format!("prefix term \"{}…*\"", "x".repeat(40))), "{msg}");
        assert!(msg.len() < 200, "{msg}");
        // `*` inside a phrase.
        for raw in ["\"security cam*\"", "\"* x\"", "+\"a*b\""] {
            assert!(
                matches!(parse_query(raw, Analyzer::Simple), Err(QuerySyntaxError::PrefixInPhrase { .. })),
                "{raw:?}"
            );
        }
        assert_eq!(
            parse_query("\"security cam*\"", Analyzer::Simple).unwrap_err().to_string(),
            "prefix terms (word*) are not supported inside a phrase: \"security cam*\""
        );
        // Length is counted in characters after analysis, not bytes: a
        // two-byte, one-character prefix is too short; two folded
        // characters are enough. CJK ideographs are one token each under
        // UAX#29, so `東京*` is the one-character prefix `京`.
        assert!(matches!(parse_query("é*", Analyzer::Simple), Err(QuerySyntaxError::PrefixTooShort { .. })));
        assert_eq!(parse("ça*").clauses, vec![prefix("ca", false)]);
        assert!(matches!(parse_query("東京*", Analyzer::Simple), Err(QuerySyntaxError::PrefixTooShort { .. })));
    }

    #[test]
    fn english_stop_words_are_dropped_from_queries_not_errors() {
        let en = |raw: &str| parse_query(raw, Analyzer::English);
        // Acceptance: `of my house` is the query `house`.
        assert_eq!(en("of my house").unwrap(), en("house").unwrap());
        assert_eq!(en("of my house").unwrap().clauses, vec![term("hous", false)]);
        // A required stop word constrains nothing.
        assert_eq!(en("+of house").unwrap().clauses, vec![term("hous", false)]);
        // Only stop words: an EMPTY query (matches nothing), not an error…
        let q = en("of my").unwrap();
        assert!(q.is_empty() && q.clauses.is_empty());
        assert!(en("+the \"of the\"").unwrap().is_empty());
        // …but nothing searchable at all is still an error, under english too.
        for raw in ["", "!!!", "\"...\"", "🎉"] {
            assert_eq!(en(raw), Err(QuerySyntaxError::NoSearchableTerms), "{raw:?}");
        }
        // A phrase keeps the gap where a stop word was.
        let q = en("\"state of the art\"").unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert_eq!(q.clauses[0].terms, vec!["state", "art"]);
        assert_eq!(q.clauses[0].offsets, vec![0, 3]);
        assert!(q.needs_positions());
        // A leading stop word shifts nothing; a phrase left with one word is
        // a term.
        let q = en("\"the state art\"").unwrap();
        assert_eq!((q.clauses[0].terms.clone(), q.clauses[0].offsets.clone()), (vec!["state".to_string(), "art".to_string()], vec![0, 1]));
        assert_eq!(en("\"of the art\"").unwrap().clauses, vec![term("art", false)]);
        // Prefix text keeps stop words: `the*` asks for words starting with "the".
        assert_eq!(en("the*").unwrap().clauses, vec![prefix("the", false)]);
        assert_eq!(en("of the*").unwrap().clauses, vec![prefix("the", false)]);
        // `simple` is untouched.
        assert_eq!(
            parse("of my house").clauses,
            vec![term("of", false), term("my", false), term("house", false)]
        );
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
