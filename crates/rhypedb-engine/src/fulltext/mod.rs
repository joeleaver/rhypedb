//! Full-text search (`@fulltext` fields, `Type.matches(.field, "terms", k: N)`).
//!
//! * [`analyzer`] — turns a `String` value (or a query clause) into a
//!   position-stamped token stream (`simple`, or `english` = simple +
//!   stemming). The SAME analyzer runs on the write path and the query path;
//!   any drift silently breaks matching, so there is exactly one
//!   implementation per analyzer.
//! * [`posting`] — the posting payload codec (`doc_len`, `tf`, positions).
//! * [`query`] — the `.matches` query mini-language (`+term`, `"phrase"`).
//! * [`search`] — BM25 scoring over decoded postings.
//!
//! The inverted index itself lives in the LSM under the `f:` (postings) and
//! `l:` (per-document token count) prefixes and is maintained synchronously
//! inside each object's transaction by the create / update / delete paths in
//! `database.rs` — exactly like `@unique` and `@indexed`. See
//! `rhypedb_storage::key::KeyPrefix::Fulltext` for the key layout.
//!
//! ## Corpus statistics live in memory, deliberately
//!
//! BM25 needs the field's document count and average length. Those are NOT a
//! persisted row: MVCC is first-committer-wins on any shared key, so a
//! single `stats` row would make every pair of concurrent writers to the
//! type conflict. Instead each field keeps a [`FulltextStats`] of atomics,
//! updated after a successful commit and rebuilt from the `l:` rows at open.
//! A crash can never leave them stale (they are re-derived), and a
//! concurrent commit can trail them by one — BM25 tolerates that (the scorer
//! clamps `N ≥ df`).

pub mod analyzer;
pub mod build;
pub mod posting;
pub mod query;
pub mod search;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use analyzer::{Analyzer, MAX_TERM_BYTES, Token};
pub use posting::Posting;
pub use query::{Clause, MIN_PREFIX_CHARS, ParsedQuery, QuerySyntaxError};
pub use build::{BuildProgress, BuildState, FulltextIndexStatus};
pub use search::{CorpusStats, FulltextHit};

#[cfg(test)]
mod tests;

// Crash-recovery fuzz for the index (needs the storage injector): WAL-site
// sweeps over a write workload + backfill-chunk sites, each cold-reopened and
// checked against the objects themselves. See the vectorizer's sibling harness.
#[cfg(all(test, feature = "crash-fuzz"))]
mod crash_fuzz;

/// Per-field corpus statistics (see the module doc for why they are in
/// memory). Both counters move together after every commit that adds,
/// removes or resizes an indexed document.
#[derive(Debug, Default)]
pub struct FulltextStats {
    doc_count: AtomicU64,
    total_tokens: AtomicU64,
}

impl FulltextStats {
    pub fn new(doc_count: u64, total_tokens: u64) -> Self {
        Self {
            doc_count: AtomicU64::new(doc_count),
            total_tokens: AtomicU64::new(total_tokens),
        }
    }

    pub fn snapshot(&self) -> CorpusStats {
        CorpusStats {
            doc_count: self.doc_count.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
        }
    }

    /// Apply a signed delta. Saturates at zero: a delta can only drive a
    /// counter negative if it was rebuilt from a snapshot that already
    /// excluded the document, and a clamped zero is the harmless outcome.
    pub fn apply(&self, docs: i64, tokens: i64) {
        fn add(counter: &AtomicU64, delta: i64) {
            if delta == 0 {
                return;
            }
            let mut cur = counter.load(Ordering::Relaxed);
            loop {
                let next = (cur as i128 + delta as i128).max(0) as u64;
                match counter.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return,
                    Err(actual) => cur = actual,
                }
            }
        }
        add(&self.doc_count, docs);
        add(&self.total_tokens, tokens);
    }
}

/// Write-path metadata for one `@fulltext` field, pre-resolved at open so
/// create / update / delete never re-walk the schema.
#[derive(Debug, Clone)]
pub struct FulltextField {
    /// Field name (the FieldMap key).
    pub name: String,
    /// Stable catalog field id (keys are by id, so a rename is a no-op).
    pub field_id: u64,
    /// Index generation namespacing this build's keys.
    pub generation: u32,
    pub analyzer: Analyzer,
    /// Whether postings store positions (phrase queries need them).
    pub positions: bool,
    pub stats: Arc<FulltextStats>,
    /// Live build state + backfill progress (see [`build`]).
    pub progress: Arc<BuildProgress>,
}

/// Corpus-stat deltas accumulated while staging a transaction, keyed by
/// `(type_id, field_id)`. Applied only after the commit lands.
#[derive(Debug, Default)]
pub struct StatsDelta {
    deltas: HashMap<(u64, u64), (i64, i64)>,
}

impl StatsDelta {
    pub fn add(&mut self, type_id: u64, field_id: u64, docs: i64, tokens: i64) {
        if docs == 0 && tokens == 0 {
            return;
        }
        let e = self.deltas.entry((type_id, field_id)).or_insert((0, 0));
        e.0 += docs;
        e.1 += tokens;
    }

    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(u64, u64), &(i64, i64))> {
        self.deltas.iter()
    }
}

/// One field value analyzed for indexing: its token count and its distinct
/// terms with ascending positions (the write path's unit of work).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedDoc {
    pub doc_len: u32,
    pub terms: Vec<(String, Vec<u32>)>,
}

/// Analyze `text` with the field's analyzer into its indexable form.
pub fn tokenize_for_index(ff: &FulltextField, text: &str) -> IndexedDoc {
    let tokens = ff.analyzer.analyze(text);
    let terms = posting::term_map(&tokens)
        .into_iter()
        .map(|(t, p)| (t.to_string(), p))
        .collect();
    IndexedDoc {
        doc_len: tokens.len() as u32,
        terms,
    }
}

/// Key bytes for a term: the engine's `\x00`-escaped, `\x00\x00`-terminated
/// encoding (shared with the var-length secondary index) so a term-prefix
/// scan is exact.
pub fn encode_term(term: &str) -> Vec<u8> {
    crate::database::encode_str_for_index(term)
}

/// Key bytes covering every term that STARTS with `prefix`: the same
/// escaping as [`encode_term`] without the terminator. The escape rewrites
/// bytes one at a time (`\x00` → `\x00\x01`, everything else verbatim), so
/// `encode_term(t)` starts with `encode_term_prefix(p)` exactly when `t`
/// starts with `p`.
pub fn encode_term_prefix(prefix: &str) -> Vec<u8> {
    let mut bytes = encode_term(prefix);
    bytes.truncate(bytes.len() - 2);
    bytes
}

/// Most distinct indexed terms a prefix clause (`cam*`) may expand to. A
/// longer expansion is refused with a clear error instead of turning a
/// two-letter prefix into a scan of half the index.
pub const MAX_PREFIX_EXPANSION: usize = 64;

/// Value of an `l:` row: the document's token count as a varint.
pub fn encode_doc_len(doc_len: u32) -> bytes::Bytes {
    let mut out = Vec::with_capacity(5);
    posting::put_varint(&mut out, doc_len);
    bytes::Bytes::from(out)
}

/// Decode an `l:` row value; must be exactly one varint.
pub fn decode_doc_len(bytes: &[u8]) -> Result<u32, posting::PostingError> {
    let mut cur = bytes;
    let v = posting::get_varint(&mut cur)?;
    if !cur.is_empty() {
        return Err(posting::PostingError::TrailingBytes);
    }
    Ok(v)
}

/// Stage every index row for a freshly indexed value: one posting per
/// distinct term plus the `l:` doc row. Appends to `puts` (flushed by the
/// caller's `put_batch`, committed with the object).
pub fn stage_doc_puts(
    type_id: u64,
    ff: &FulltextField,
    object_id: u64,
    doc: &IndexedDoc,
    puts: &mut Vec<(bytes::Bytes, bytes::Bytes)>,
) {
    puts.reserve(doc.terms.len() + 1);
    for (term, positions) in &doc.terms {
        let key = rhypedb_storage::key::KeyBuilder::fulltext_posting(
            type_id,
            ff.field_id,
            ff.generation,
            &encode_term(term),
            object_id,
        );
        puts.push((key, posting::encode_posting(doc.doc_len, positions, ff.positions)));
    }
    puts.push((
        rhypedb_storage::key::KeyBuilder::fulltext_doc(type_id, ff.field_id, ff.generation, object_id),
        encode_doc_len(doc.doc_len),
    ));
}

/// Result of a full-text search: the ranked hits plus how many posting rows
/// were examined (charged against the query governor's row budget).
#[derive(Debug, Clone, PartialEq)]
pub struct FulltextSearchResult {
    pub hits: Vec<FulltextHit>,
    pub postings_scanned: u64,
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    #[test]
    fn stats_apply_saturates_at_zero_and_tracks_both_counters() {
        let s = FulltextStats::new(2, 10);
        s.apply(1, 5);
        assert_eq!(s.snapshot(), CorpusStats { doc_count: 3, total_tokens: 15 });
        s.apply(-1, -5);
        assert_eq!(s.snapshot(), CorpusStats { doc_count: 2, total_tokens: 10 });
        s.apply(-5, -50);
        assert_eq!(s.snapshot(), CorpusStats { doc_count: 0, total_tokens: 0 });
        s.apply(0, 0);
        assert_eq!(s.snapshot(), CorpusStats { doc_count: 0, total_tokens: 0 });
    }

    #[test]
    fn term_prefix_encoding_is_prefix_preserving() {
        assert_eq!(encode_term_prefix(""), Vec::<u8>::new());
        assert_eq!(encode_term_prefix("cam"), b"cam".to_vec());
        assert!(encode_term("camera").starts_with(&encode_term_prefix("cam")));
        assert!(encode_term("cam").starts_with(&encode_term_prefix("cam")));
        assert!(!encode_term("ca").starts_with(&encode_term_prefix("cam")));
        assert!(!encode_term("dam").starts_with(&encode_term_prefix("cam")));
        // Escaped NULs stay prefix-preserving.
        assert_eq!(encode_term_prefix("a\0"), b"a\0\x01".to_vec());
        assert!(encode_term("a\0b").starts_with(&encode_term_prefix("a\0")));
        assert!(!encode_term("a").starts_with(&encode_term_prefix("a\0")));
    }

    #[test]
    fn delta_merges_per_field_and_drops_zeroes() {
        let mut d = StatsDelta::default();
        assert!(d.is_empty());
        d.add(1, 2, 0, 0);
        assert!(d.is_empty());
        d.add(1, 2, 1, 4);
        d.add(1, 2, -1, 3);
        d.add(1, 3, 1, 1);
        let mut got: Vec<_> = d.iter().map(|(k, v)| (*k, *v)).collect();
        got.sort();
        assert_eq!(got, vec![((1, 2), (0, 7)), ((1, 3), (1, 1))]);
    }
}
