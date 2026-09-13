//! Text analyzers for `@fulltext` fields.
//!
//! An analyzer maps a string to a sequence of [`Token`]s — a normalized term
//! plus its 0-based position in the token stream. Positions are what phrase
//! queries (`"a b"`) intersect on, so two adjacent words in the source text
//! MUST get consecutive positions even when one of them is dropped from the
//! index (an over-long token still consumes its position).
//!
//! The analyzer name is part of a field's index identity: the SDL stores it
//! (`@fulltext(analyzer: "simple")`), the catalog digest hashes it, and the
//! index build marker records it so a change rebuilds the index rather than
//! mixing token streams produced by two different analyzers.

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;
use unicode_segmentation::UnicodeSegmentation;

/// Longest term (in UTF-8 bytes, after normalization) that is indexed. Longer
/// tokens — base64 blobs, minified source, URLs glued into one word — are
/// dropped: they are never meaningful search terms and would bloat the key
/// space. Same cap as Lucene's `StandardAnalyzer` (255).
pub const MAX_TERM_BYTES: usize = 255;

/// One indexed term with its position in the token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The normalized term exactly as it is keyed in the index.
    pub term: String,
    /// 0-based position of the source word in the token stream. Dropped
    /// over-long words still consume a position (see the module doc).
    pub position: u32,
}

/// A named text analyzer. `Copy` so the write path can hold one per field
/// without allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    /// The default and (today) only analyzer:
    ///
    /// 1. UAX#29 word segmentation (`unicode_words`) — splits on whitespace
    ///    and punctuation, keeps `don't`, `3.14`, `e-mail` → `e`,`mail`;
    ///    CJK ideographs are one word each.
    /// 2. Unicode lowercase.
    /// 3. NFKD decomposition, then every combining mark is dropped — this is
    ///    the ASCII folding of diacritics (`café` → `cafe`, `naïve` →
    ///    `naive`) and also folds compatibility forms (`ﬁ` → `fi`, full-width
    ///    `１２３` → `123`).
    ///
    /// No stemming, no stop words.
    Simple,
}

impl Analyzer {
    /// Resolve an analyzer by its SDL name. The schema parser only admits
    /// names from `FulltextDef::KNOWN_ANALYZERS`, so `None` here means the
    /// engine and schema crates disagree — callers treat it as a hard error.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "simple" => Some(Self::Simple),
            _ => None,
        }
    }

    /// The SDL name of this analyzer.
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "simple",
        }
    }

    /// Tokenize `text`. Deterministic: the same input always yields the same
    /// token stream, on every platform (the crates are table-driven, not
    /// locale-driven).
    pub fn analyze(self, text: &str) -> Vec<Token> {
        match self {
            Self::Simple => analyze_simple(text),
        }
    }
}

fn analyze_simple(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut position: u32 = 0;
    for word in text.unicode_words() {
        let term = fold_simple(word);
        // Position is consumed whether or not the term is kept, so phrase
        // adjacency in the source text is preserved around a dropped token.
        let pos = position;
        position = position.saturating_add(1);
        if term.is_empty() || term.len() > MAX_TERM_BYTES {
            continue;
        }
        out.push(Token {
            term,
            position: pos,
        });
    }
    out
}

/// Lowercase, NFKD-decompose and strip combining marks from one word.
fn fold_simple(word: &str) -> String {
    let lowered = word.to_lowercase();
    lowered.nfkd().filter(|c| !is_combining_mark(*c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(text: &str) -> Vec<String> {
        Analyzer::Simple
            .analyze(text)
            .into_iter()
            .map(|t| t.term)
            .collect()
    }

    fn positions(text: &str) -> Vec<(String, u32)> {
        Analyzer::Simple
            .analyze(text)
            .into_iter()
            .map(|t| (t.term, t.position))
            .collect()
    }

    #[test]
    fn name_round_trips() {
        assert_eq!(Analyzer::from_name("simple"), Some(Analyzer::Simple));
        assert_eq!(Analyzer::Simple.name(), "simple");
        assert_eq!(Analyzer::from_name("english"), None);
        // Every name the schema parser admits resolves here — the two crates
        // must agree.
        for name in rhypedb_schema::FulltextDef::KNOWN_ANALYZERS {
            assert!(Analyzer::from_name(name).is_some(), "{name}");
        }
    }

    #[test]
    fn splits_on_whitespace_and_punctuation_and_lowercases() {
        assert_eq!(
            terms("The note that mentions Invoice 4471!"),
            vec!["the", "note", "that", "mentions", "invoice", "4471"]
        );
        assert_eq!(terms("e-mail, Invoice/4471;(draft)"), vec!["e", "mail", "invoice", "4471", "draft"]);
        // UAX#29 treats `:` and `.` between letters as word-internal (MidLetter),
        // exactly like Lucene's StandardTokenizer: "re:invoice" is ONE term.
        assert_eq!(terms("re:Invoice a.b.c"), vec!["re:invoice", "a.b.c"]);
        assert_eq!(terms("  multiple   spaces\tand\nnewlines "), vec!["multiple", "spaces", "and", "newlines"]);
    }

    #[test]
    fn keeps_uax29_words_intact() {
        // Apostrophes and decimal points are word-internal under UAX#29.
        assert_eq!(terms("don't stop at 3.14"), vec!["don't", "stop", "at", "3.14"]);
        // Underscore is word-internal too (ExtendNumLet).
        assert_eq!(terms("snake_case_id"), vec!["snake_case_id"]);
        // Email splits at `@`, keeps the dotted host.
        assert_eq!(terms("me@example.com"), vec!["me", "example.com"]);
    }

    #[test]
    fn folds_diacritics_and_compatibility_forms() {
        assert_eq!(terms("Café naïve résumé Ærø"), vec!["cafe", "naive", "resume", "ærø"]);
        assert_eq!(terms("ÉCOLE Ångström"), vec!["ecole", "angstrom"]);
        // Precomposed and decomposed input fold to the same term.
        assert_eq!(terms("caf\u{e9}"), terms("cafe\u{301}"));
        // Compatibility decomposition: ligature + full-width digits.
        assert_eq!(terms("ﬁle １２３"), vec!["file", "123"]);
        // Turkish dotted capital I lowercases to i + combining dot → "i".
        assert_eq!(terms("İstanbul"), vec!["istanbul"]);
        // ß has no diacritic to strip; it stays (no case-fold expansion).
        assert_eq!(terms("Straße"), vec!["straße"]);
    }

    #[test]
    fn cjk_ideographs_are_one_token_each() {
        assert_eq!(terms("東京都"), vec!["東", "京", "都"]);
        // Mixed script keeps Latin words whole.
        assert_eq!(terms("東京 tokyo"), vec!["東", "京", "tokyo"]);
    }

    #[test]
    fn positions_are_consecutive_and_survive_dropped_tokens() {
        assert_eq!(
            positions("a b  c"),
            vec![("a".into(), 0), ("b".into(), 1), ("c".into(), 2)]
        );
        // A word longer than MAX_TERM_BYTES is dropped but still consumes its
        // position, so "a" and "c" are NOT adjacent (phrase "a c" must not
        // match).
        let long = "x".repeat(MAX_TERM_BYTES + 1);
        assert_eq!(
            positions(&format!("a {long} c")),
            vec![("a".into(), 0), ("c".into(), 2)]
        );
        // Exactly the cap is kept.
        let cap = "y".repeat(MAX_TERM_BYTES);
        assert_eq!(terms(&cap), vec![cap.clone()]);
        // The cap is measured AFTER folding (in bytes): 128 two-byte chars
        // that fold to 128 one-byte chars fit.
        let folded = "é".repeat(128);
        assert_eq!(terms(&folded), vec!["e".repeat(128)]);
    }

    #[test]
    fn empty_and_symbol_only_input_yield_no_tokens() {
        assert!(terms("").is_empty());
        assert!(terms("   \n\t ").is_empty());
        assert!(terms("... !!! --- ***").is_empty());
        // Emoji are not words.
        assert!(terms("🎉🎉").is_empty());
        assert_eq!(terms("party 🎉 time"), vec!["party", "time"]);
    }

    #[test]
    fn is_deterministic() {
        let text = "Déjà vu: the SAME text, analyzed twice — 3.14 東京";
        assert_eq!(Analyzer::Simple.analyze(text), Analyzer::Simple.analyze(text));
    }
}
