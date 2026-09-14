//! Text analyzers for `@fulltext` fields.
//!
//! An analyzer maps a string to a sequence of [`Token`]s — a normalized term
//! plus its 0-based position in the token stream. Positions are what phrase
//! queries (`"a b"`) intersect on, so two adjacent words in the source text
//! MUST get consecutive positions even when one of them is dropped from the
//! index (an over-long token still consumes its position).
//!
//! The analyzer is part of a field's index identity: the SDL stores its name
//! (`@fulltext(analyzer: "simple")`), the catalog digest hashes it, and the
//! index build marker records its [`Analyzer::definition`] — the name plus a
//! version of its behaviour — so a change of analyzer OR of what an analyzer
//! does (a new stop-word list, say) rebuilds the index rather than mixing
//! token streams produced by two different tokenizations.

use rust_stemmers::{Algorithm, Stemmer};
use unicode_normalization::UnicodeNormalization;
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
    /// The default analyzer:
    ///
    /// 1. UAX#29 word segmentation (`unicode_words`) — splits on whitespace
    ///    and punctuation, keeps `don't`, `3.14`, `e-mail` → `e`,`mail`;
    ///    CJK ideographs are one word each.
    /// 2. Unicode lowercase.
    /// 3. NFKD decomposition, then the **diacritic** combining marks are
    ///    dropped (the Combining Diacritical Marks blocks: U+0300–036F,
    ///    U+1AB0–1AFF, U+1DC0–1DFF, U+20D0–20FF) — this is the ASCII folding
    ///    of diacritics (`café` → `cafe`, `naïve` → `naive`) and also folds
    ///    compatibility forms (`ﬁ` → `fi`, full-width `１２３` → `123`).
    ///    Script-bearing combining marks (Devanagari matras and virama, Thai
    ///    vowels/tones, Arabic harakat, Hebrew niqqud, …) are NOT diacritics
    ///    and are kept — stripping them would merge distinct words.
    /// 4. Default-ignorable format characters (soft hyphen, zero-width
    ///    space/joiners, bidi marks, BOM, variation selectors) are dropped so
    ///    pasted text matches typed text.
    ///
    /// No stemming, no stop words.
    Simple,
    /// [`Simple`](Self::Simple), then English stemming, so the inflections of
    /// a word share one term (`camera`, `cameras`, `Camera's` → `camera`;
    /// `run`, `running`, `runs` → `run`):
    ///
    /// 5. Typographic apostrophes (`’` U+2019, `‘` U+2018, `ʼ` U+02BC) become
    ///    the ASCII apostrophe — UAX#29 keeps them word-internal exactly like
    ///    `'`, and the stemmer's possessive rule only knows `'`.
    /// 6. The Snowball English stemmer (Porter2): strips `'s`, then the
    ///    plural / `-ing` / `-ed` / `-ly` / `-ness` / `-ation` … suffix
    ///    families by the reference algorithm. Words shorter than three
    ///    letters, numbers and non-Latin words pass through unchanged.
    ///
    /// The stem is what the index holds, so the same function runs on query
    /// terms — including every word of a phrase and the text of a prefix
    /// term. Only English inflection is modelled: a French or German word
    /// gets the English rules applied, which is harmless but not stemming.
    ///
    /// 7. **Stop words** ([`ENGLISH_STOP_WORDS`], the Snowball list) are
    ///    dropped — on indexed text and on query text alike, so `of my
    ///    house` searches for `house` and `of`/`my` never enter the
    ///    postings. Checked after the fold and apostrophe normalization and
    ///    BEFORE stemming (the list holds surface forms: `don't`, `it's`,
    ///    `having`). A dropped stop word still consumes its position, exactly
    ///    like an over-long token, so a phrase keeps its shape: `"state of
    ///    the art"` is `state` at +0 and `art` at +3, and matches a document
    ///    with the same gap. The text of a prefix term (`the*`) is exempt —
    ///    a prefix asks for words STARTING with those letters, not the word.
    English,
}

/// The Snowball English stop-word list (the one Lucene's `EnglishAnalyzer`,
/// Postgres's `english` configuration and SQLite FTS5's porter setups all
/// derive from): pronouns, auxiliaries, determiners, prepositions,
/// conjunctions and their common contractions — 174 surface forms. Sorted,
/// so membership is a binary search.
pub const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
];

/// Whether `word` (already folded and apostrophe-normalized) is an English
/// stop word.
pub fn is_english_stop_word(word: &str) -> bool {
    ENGLISH_STOP_WORDS.binary_search(&word).is_ok()
}

/// The token stream of one analysis plus what was dropped on the way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analyzed {
    pub tokens: Vec<Token>,
    /// Words dropped because they are stop words (positions consumed). Lets
    /// a query distinguish "nothing but stop words" (an empty result) from
    /// "nothing searchable at all" (an error).
    pub stop_words_dropped: u32,
}

impl Analyzer {
    /// Resolve an analyzer by its SDL name. The schema parser only admits
    /// names from `FulltextDef::KNOWN_ANALYZERS`, so `None` here means the
    /// engine and schema crates disagree — callers treat it as a hard error.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "simple" => Some(Self::Simple),
            "english" => Some(Self::English),
            _ => None,
        }
    }

    /// The SDL name of this analyzer.
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::English => "english",
        }
    }

    /// The identity the index build marker records: the SDL name, plus a
    /// suffix whenever the analyzer's BEHAVIOUR has changed since it first
    /// shipped, so an index built by the previous behaviour no longer
    /// matches and is rebuilt into a new generation (see
    /// `BuildMarker::matches_config`). Bump the suffix — never edit an
    /// existing analyzer's behaviour without one; the index would silently
    /// stop matching the query side.
    ///
    /// * `simple`: unchanged since it shipped.
    /// * `english/2`: `english` gained stop-word removal (issue #20). A
    ///   marker that says plain `english` is a #17-era index.
    pub fn definition(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::English => "english/2",
        }
    }

    /// Tokenize `text`. Deterministic: the same input always yields the same
    /// token stream, on every platform (the crates are table-driven, not
    /// locale-driven). Stop words (english) are dropped.
    pub fn analyze(self, text: &str) -> Vec<Token> {
        self.analyze_opts(text, false).tokens
    }

    /// [`Self::analyze`] with control over stop words and a count of what
    /// was dropped. `keep_stop_words` is for the text of a prefix term: a
    /// prefix asks for words that START with those letters, so `the*` must
    /// expand to `theory`, `theatre`, … rather than vanish.
    pub fn analyze_opts(self, text: &str, keep_stop_words: bool) -> Analyzed {
        match self {
            Self::Simple => analyze_words(text, |w| Some(fold_simple(w))),
            Self::English => analyze_words(text, |w| fold_english(w, keep_stop_words)),
        }
    }
}

/// Split `text` into UAX#29 words, fold each one into its term with `fold`
/// (`None` = a stop word), and stamp positions. Shared by every analyzer so
/// the position and length-cap rules can never drift between them.
fn analyze_words(text: &str, fold: impl Fn(&str) -> Option<String>) -> Analyzed {
    let mut out = Vec::new();
    let mut stop_words_dropped = 0u32;
    let mut position: u32 = 0;
    for word in text.unicode_words() {
        let term = fold(word);
        // Position is consumed whether or not the term is kept, so phrase
        // adjacency in the source text is preserved around a dropped token —
        // an over-long word, a word that folds to nothing, or a stop word.
        let pos = position;
        position = position.saturating_add(1);
        let Some(term) = term else {
            stop_words_dropped += 1;
            continue;
        };
        if term.is_empty() || term.len() > MAX_TERM_BYTES {
            continue;
        }
        out.push(Token {
            term,
            position: pos,
        });
    }
    Analyzed {
        tokens: out,
        stop_words_dropped,
    }
}

/// Lowercase, NFKD-decompose, strip diacritic marks and ignorable format
/// characters from one word.
fn fold_simple(word: &str) -> String {
    let lowered = word.to_lowercase();
    lowered
        .nfkd()
        .filter(|&c| !is_diacritic_mark(c) && !is_default_ignorable(c))
        .collect()
}

/// [`fold_simple`], then apostrophe normalization, the stop-word check
/// (`None` = dropped as a stop word, unless `keep_stop_words`) and the
/// Snowball English stemmer. The length cap is applied by the caller on
/// the RESULT, so a stem is measured, not the surface form.
fn fold_english(word: &str, keep_stop_words: bool) -> Option<String> {
    let folded: String = fold_simple(word)
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' => '\'',
            c => c,
        })
        .collect();
    if folded.is_empty() {
        return Some(folded);
    }
    if !keep_stop_words && is_english_stop_word(&folded) {
        return None;
    }
    // `Stemmer` is a plain function-pointer wrapper; creating one is free
    // and keeps the analyzer `Copy` with no shared state.
    Some(match Stemmer::create(Algorithm::English).stem(&folded) {
        std::borrow::Cow::Borrowed(_) => folded,
        std::borrow::Cow::Owned(stemmed) => stemmed,
    })
}

/// The four Combining Diacritical Marks blocks (UTR#30 "diacritic folding"
/// scope). Deliberately NOT `is_combining_mark`, which also covers the vowel
/// signs of abugida scripts.
fn is_diacritic_mark(c: char) -> bool {
    matches!(
        c,
        '\u{0300}'..='\u{036F}' | '\u{1AB0}'..='\u{1AFF}' | '\u{1DC0}'..='\u{1DFF}' | '\u{20D0}'..='\u{20FF}'
    )
}

/// Default_Ignorable_Code_Point members that show up in pasted text: soft
/// hyphen, combining grapheme joiner, Arabic letter mark, Hangul fillers,
/// Khmer/Mongolian format controls, zero-width space / ZWNJ / ZWJ / bidi
/// marks, bidi embeddings, word joiner + invisible operators, variation
/// selectors, BOM, interlinear annotation controls, tag characters.
/// ZWNJ (U+200C) is stripped on purpose: Persian/Indic text is written both
/// with and without it, and folding both spellings to one term is the
/// search-friendly choice.
fn is_default_ignorable(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{034F}'
            | '\u{061C}'
            | '\u{115F}'..='\u{1160}'
            | '\u{17B4}'..='\u{17B5}'
            | '\u{180B}'..='\u{180F}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{206F}'
            | '\u{3164}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
            | '\u{FFA0}'
            | '\u{FFF0}'..='\u{FFF8}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0000}'..='\u{E0FFF}'
    )
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

    fn english(text: &str) -> Vec<String> {
        Analyzer::English
            .analyze(text)
            .into_iter()
            .map(|t| t.term)
            .collect()
    }

    #[test]
    fn name_round_trips() {
        assert_eq!(Analyzer::from_name("simple"), Some(Analyzer::Simple));
        assert_eq!(Analyzer::Simple.name(), "simple");
        assert_eq!(Analyzer::from_name("english"), Some(Analyzer::English));
        assert_eq!(Analyzer::English.name(), "english");
        assert_eq!(Analyzer::from_name("English"), None);
        assert_eq!(Analyzer::from_name("klingon"), None);
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
    fn keeps_script_vowel_marks_but_drops_diacritics() {
        // Devanagari matras + virama are Mn/Mc but NOT diacritics: four
        // different words must stay four different terms.
        assert_eq!(terms("कल काल कील कुल"), vec!["कल", "काल", "कील", "कुल"]);
        assert_eq!(terms("हिन्दी"), vec!["हिन्दी"]);
        // Thai has no word boundaries and UAX#29 has no dictionary
        // segmentation, so `simple` yields one token per grapheme cluster —
        // but every vowel/tone mark stays attached to its base (Mn marks are
        // NOT stripped): the tokens concatenate back to the exact input.
        assert_eq!(terms("ที่"), vec!["ที่"]);
        assert_eq!(terms("ภาษาไทย").concat(), "ภาษาไทย");
        // Arabic harakat and Hebrew niqqud survive (no language-specific
        // normalization in `simple`).
        assert_eq!(terms("مُحَمَّد"), vec!["مُحَمَّد"]);
        assert_eq!(terms("שָׁלוֹם"), vec!["שָׁלוֹם"]);
        // …while Latin/Greek/Cyrillic diacritics still fold.
        assert_eq!(terms("Ñandú Ελληνικά Ёлка"), vec!["nandu", "ελληνικα", "елка"]);
        // Combining-only input folds away to nothing rather than panicking.
        assert!(terms("\u{301}\u{308}").is_empty());
    }

    #[test]
    fn drops_invisible_format_characters() {
        assert_eq!(terms("hello\u{200F} co\u{AD}operate a\u{200D}b \u{FEFF}bom"), vec!["hello", "cooperate", "ab", "bom"]);
        // ZWNJ folds both Persian spellings to one term.
        assert_eq!(terms("می\u{200C}خواهم"), terms("میخواهم"));
        // Emoji variation selectors don't survive into a term either.
        assert_eq!(terms("x\u{FE0F}y"), vec!["xy"]);
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
        assert_eq!(Analyzer::English.analyze(text), Analyzer::English.analyze(text));
    }

    // ---- english ----

    #[test]
    fn english_folds_inflections_and_possessives_to_one_term() {
        // The issue's acceptance case: every spelling is the same term.
        assert_eq!(english("cameras camera Camera's CAMERAS'"), vec!["camera"; 4]);
        assert_eq!(english("run running runs"), vec!["run"; 3]);
        assert_eq!(english("fly flying flies"), vec!["fli"; 3]);
        assert_eq!(english("table tables"), vec!["tabl"; 2]);
        assert_eq!(english("invoice invoices"), vec!["invoic"; 2]);
        // Snowball's reference vector.
        assert_eq!(english("fruitlessly"), vec!["fruitless"]);
        // A phrase stems word by word, positions intact.
        assert_eq!(
            positions_of(Analyzer::English, "security cameras"),
            vec![("secur".into(), 0), ("camera".into(), 1)]
        );
        assert_eq!(english("\"security camera\""), english("security cameras"));
    }

    #[test]
    fn english_runs_the_simple_fold_first() {
        // Case + diacritics fold before stemming, so the stem is of the
        // folded form.
        assert_eq!(english("Cafés"), vec!["cafe"]);
        assert_eq!(english("RÉSUMÉS résumé"), vec!["resum", "resum"]);
        assert_eq!(english("ﬁles"), vec!["file"]);
        // Everything the simple analyzer drops, english drops too.
        assert_eq!(english("... !!! 🎉"), Vec::<String>::new());
        assert_eq!(english("co\u{AD}operate"), vec!["cooper"]);
        // Same tokenization: hyphens split, apostrophes stay word-internal
        // (`don't` is one word — and a stop word; `won't`-style contractions
        // are on the list as surface forms).
        assert_eq!(english("e-mail don't"), vec!["e", "mail"]);
        assert_eq!(english("e-mail dont"), vec!["e", "mail", "dont"]);
    }

    #[test]
    fn english_normalizes_typographic_apostrophes_before_the_possessive_rule() {
        // UAX#29 keeps ’ / ‘ / ʼ word-internal like ', but Snowball only
        // strips an ASCII 's — without normalization "camera’s" would be
        // its own term.
        assert_eq!(english("camera\u{2019}s"), vec!["camera"]);
        assert_eq!(english("camera\u{2018}s"), vec!["camera"]);
        assert_eq!(english("camera\u{02BC}s"), vec!["camera"]);
        assert_eq!(english("camera\u{2019}s"), english("camera's"));
        // Full-width apostrophe folds via NFKD already.
        assert_eq!(english("camera\u{FF07}s"), vec!["camera"]);
        // `simple` is untouched by this rule (its index must not shift).
        assert_eq!(terms("camera\u{2019}s"), vec!["camera\u{2019}s"]);
    }

    #[test]
    fn english_leaves_numbers_short_words_and_other_scripts_alone() {
        assert_eq!(english("3.14 4471 v2"), vec!["3.14", "4471", "v2"]);
        // Short words pass the stemmer unchanged — but `a`, `is`, `the` are
        // stop words and go; `us` is not on the list.
        assert_eq!(english("a is us the"), vec!["us"]);
        assert_eq!(english("us"), vec!["us"]);
        assert_eq!(english("東京都 कल Ελληνικά"), vec!["東", "京", "都", "कल", "ελληνικα"]);
        assert_eq!(english("Straße"), vec!["straße"]);
        // Non-English Latin words get the English rules — harmless, but
        // documented as not-stemming.
        assert_eq!(english("données"), vec!["donne"]);
    }

    #[test]
    fn english_length_cap_measures_the_stem_and_keeps_positions() {
        // The cap is on the indexed term, i.e. the stem: a 257-byte surface
        // form whose stem fits is kept ("a" + b×252 + "ings" → step 1a drops
        // the "s", step 1b the "ing").
        let long = format!("a{}ings", "b".repeat(MAX_TERM_BYTES - 3));
        assert_eq!(long.len(), MAX_TERM_BYTES + 2);
        assert_eq!(terms(&long), Vec::<String>::new(), "simple drops it");
        let stem = english(&long);
        assert_eq!(stem.len(), 1, "english keeps the stem");
        assert!(stem[0].len() <= MAX_TERM_BYTES && stem[0].starts_with("abbb"), "{}", stem[0]);
        // Over the cap even after stemming: dropped, position consumed.
        let huge = "y".repeat(MAX_TERM_BYTES + 10);
        assert_eq!(
            positions_of(Analyzer::English, &format!("zeta {huge} c")),
            vec![("zeta".into(), 0), ("c".into(), 2)]
        );
    }

    fn positions_of(a: Analyzer, text: &str) -> Vec<(String, u32)> {
        a.analyze(text).into_iter().map(|t| (t.term, t.position)).collect()
    }

    // ---- english stop words (issue #20) ----

    #[test]
    fn stop_word_list_is_sorted_lowercase_and_unique() {
        for w in ENGLISH_STOP_WORDS {
            assert_eq!(*w, w.to_lowercase(), "{w}");
            assert!(!w.is_empty());
        }
        for pair in ENGLISH_STOP_WORDS.windows(2) {
            assert!(pair[0] < pair[1], "not sorted/unique at {pair:?}");
        }
        assert_eq!(ENGLISH_STOP_WORDS.len(), 174);
        assert!(is_english_stop_word("the") && is_english_stop_word("don't"));
        assert!(!is_english_stop_word("house") && !is_english_stop_word("us"));
    }

    #[test]
    fn english_drops_stop_words_before_stemming_and_keeps_their_positions() {
        // The issue's acceptance case: `of my house` searches for `house`.
        assert_eq!(english("of my house"), vec!["hous"]);
        assert_eq!(english("surveillance video of my house"), vec!["surveil", "video", "hous"]);
        // Checked on the folded, apostrophe-normalized surface form, before
        // stemming: `having` is on the list (its stem `have` is too, but that
        // is not what is checked), and the typographic apostrophe folds.
        assert_eq!(english("Having HAVING don\u{2019}t It's"), Vec::<String>::new());
        // Position is consumed: the phrase keeps its shape.
        assert_eq!(
            positions_of(Analyzer::English, "state of the art"),
            vec![("state".into(), 0), ("art".into(), 3)]
        );
        // Only stop words: nothing left, and the count says why.
        let a = Analyzer::English.analyze_opts("of the", false);
        assert_eq!((a.tokens.len(), a.stop_words_dropped), (0, 2));
        let a = Analyzer::English.analyze_opts("!!!", false);
        assert_eq!((a.tokens.len(), a.stop_words_dropped), (0, 0));
        // Kept on request (prefix text): the surface stop word is stemmed
        // like any word.
        let a = Analyzer::English.analyze_opts("the", true);
        assert_eq!((terms_of(&a.tokens), a.stop_words_dropped), (vec!["the".to_string()], 0));
        // `simple` never drops them.
        assert_eq!(terms("of my house"), vec!["of", "my", "house"]);
        assert_eq!(Analyzer::Simple.analyze_opts("of the", false).stop_words_dropped, 0);
    }

    #[test]
    fn definitions_version_behaviour_changes() {
        assert_eq!(Analyzer::Simple.definition(), "simple");
        assert_eq!(Analyzer::English.definition(), "english/2");
        // The definition starts with the SDL name, so a marker is readable.
        for a in [Analyzer::Simple, Analyzer::English] {
            assert!(a.definition().starts_with(a.name()));
        }
    }

    fn terms_of(tokens: &[Token]) -> Vec<String> {
        tokens.iter().map(|t| t.term.clone()).collect()
    }
}
