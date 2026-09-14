//! BM25 scoring over decoded postings.
//!
//! Pure: takes the parsed query, one posting list per distinct term, the
//! corpus statistics and an optional candidate restriction, and returns the
//! top-`k` hits. The storage-facing half (fetching the posting lists at a
//! snapshot) lives on `Database::fulltext_search`, so this can be tested
//! exhaustively without an LSM.
//!
//! Formula (Lucene's BM25Similarity):
//!
//! ```text
//! idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))
//! tfnorm(t,d) = tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl / avgdl))
//! score(d)    = Σ_{clauses matched} Σ_{t ∈ clause} idf(t) · tfnorm(t, d)
//! ```
//!
//! with `k1 = 1.2`, `b = 0.75`. A phrase clause contributes the sum of its
//! DISTINCT terms' scores when (and only when) the terms occur consecutively.
//! Ties break on ascending object id so results are deterministic.
//!
//! A prefix clause (`cam*`) is scored as ONE term: the caller hands in its
//! expansion already merged into a single posting list under the clause's
//! key (`"cam*"`) — per document `tf` is the sum over the matching terms
//! and `df` the number of distinct documents — so a prefix behaves like a
//! word with several spellings. Expanding into one clause per term instead
//! would let a rare misspelling among the expansions dominate with a huge
//! idf (Lucene's `SCORING_BOOLEAN_REWRITE` pathology).

use std::collections::{HashMap, HashSet};

use super::posting::Posting;
use super::query::{Clause, ParsedQuery};

pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// One search hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FulltextHit {
    pub object_id: u64,
    /// BM25 score; higher is better. Always finite and > 0.
    pub score: f32,
}

/// Corpus statistics for one field: document count and total token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusStats {
    pub doc_count: u64,
    pub total_tokens: u64,
}

/// Posting list for one term: `(object_id, posting)`, any order.
pub type PostingList = Vec<(u64, Posting)>;

/// Score `query` and return the top `k` hits (score desc, id asc).
///
/// `postings` maps every distinct term of the query to its posting list (a
/// missing entry means the term has no postings). `restrict`, when given,
/// limits candidates to that id set — applied BEFORE the top-k cut so the
/// caller gets `k` results from within the set, not `k` minus the
/// filtered-out ones.
pub fn score_query(
    query: &ParsedQuery,
    postings: &HashMap<&str, PostingList>,
    stats: CorpusStats,
    restrict: Option<&HashSet<u64>>,
    k: usize,
) -> Vec<FulltextHit> {
    if k == 0 || stats.doc_count == 0 {
        return Vec::new();
    }
    let required_count = query.clauses.iter().filter(|c| c.required).count() as u32;

    // idf per term. `N` is clamped to at least df: the in-memory doc count
    // can trail a concurrent commit by one, and a negative numerator would
    // flip the score's sign.
    let empty: PostingList = Vec::new();
    let idf: HashMap<&str, f32> = query
        .distinct_terms()
        .into_iter()
        .map(|t| {
            let df = postings.get(t).map_or(0, |p| p.len()) as f64;
            let n = (stats.doc_count as f64).max(df);
            (t, (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32)
        })
        .collect();
    let avgdl = if stats.total_tokens == 0 {
        1.0
    } else {
        stats.total_tokens as f32 / stats.doc_count as f32
    };
    let tfnorm = |p: &Posting| -> f32 {
        let tf = p.tf as f32;
        let dl = p.doc_len as f32;
        tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl))
    };
    let allowed = |id: u64| restrict.is_none_or(|set| set.contains(&id));

    #[derive(Default)]
    struct Acc {
        score: f32,
        required_hits: u32,
    }
    let mut acc: HashMap<u64, Acc> = HashMap::new();
    let mut add = |id: u64, contribution: f32, clause: &Clause| {
        let a = acc.entry(id).or_default();
        a.score += contribution;
        if clause.required {
            a.required_hits += 1;
        }
    };

    for clause in &query.clauses {
        if clause.is_phrase() {
            score_phrase(clause, postings, &idf, &tfnorm, &allowed, &mut add);
        } else {
            let term = clause.terms[0].as_str();
            let list = postings.get(term).unwrap_or(&empty);
            let w = idf[term];
            for (id, p) in list {
                if allowed(*id) {
                    add(*id, w * tfnorm(p), clause);
                }
            }
        }
    }

    let mut hits: Vec<FulltextHit> = acc
        .into_iter()
        .filter(|(_, a)| a.required_hits == required_count && a.score > 0.0)
        .map(|(object_id, a)| FulltextHit {
            object_id,
            score: a.score,
        })
        .collect();
    hits.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.object_id.cmp(&b.object_id))
    });
    hits.truncate(k);
    hits
}

/// Phrase clause: a document matches when every term is present and some
/// occurrence of term₀ at position `p` is followed by termᵢ at
/// `p + offsetᵢ` (see `Clause::offsets` — consecutive words are `1, 2, …`; a
/// larger step is a word the analyzer dropped, e.g. a stop word, which the
/// document must also have a word in place of). Repeated terms inside a
/// phrase ("to be or not to be") are handled naturally by the per-term
/// position lists.
fn score_phrase(
    clause: &Clause,
    postings: &HashMap<&str, PostingList>,
    idf: &HashMap<&str, f32>,
    tfnorm: &dyn Fn(&Posting) -> f32,
    allowed: &dyn Fn(u64) -> bool,
    add: &mut dyn FnMut(u64, f32, &Clause),
) {
    // Per-term id → posting lookups; drive from the shortest list.
    let mut maps: Vec<HashMap<u64, &Posting>> = Vec::with_capacity(clause.terms.len());
    for t in &clause.terms {
        let Some(list) = postings.get(t.as_str()) else {
            return; // a term with no postings → the phrase matches nothing
        };
        maps.push(list.iter().map(|(id, p)| (*id, p)).collect());
    }
    let (driver_idx, _) = maps
        .iter()
        .enumerate()
        .min_by_key(|(_, m)| m.len())
        .expect("phrase has ≥ 2 terms");
    let driver: Vec<u64> = maps[driver_idx].keys().copied().collect();
    'docs: for id in driver {
        if !allowed(id) {
            continue;
        }
        let mut per_term: Vec<&Posting> = Vec::with_capacity(maps.len());
        for m in &maps {
            match m.get(&id) {
                Some(p) => per_term.push(p),
                None => continue 'docs,
            }
        }
        // Shape: walk term₀'s positions; each later term must sit at
        // p + its offset. Position lists are ascending, so a binary search
        // keeps this O(tf₀ · Σ log tfᵢ).
        let matched = per_term[0].positions.iter().any(|&p| {
            per_term.iter().enumerate().skip(1).all(|(i, post)| {
                p.checked_add(clause.offset(i))
                    .is_some_and(|want| post.positions.binary_search(&want).is_ok())
            })
        });
        if matched {
            // Sum over the DISTINCT terms of the phrase: "to be or not to be"
            // scores `to` and `be` once each, not once per occurrence.
            let mut seen: Vec<&str> = Vec::with_capacity(clause.terms.len());
            let contribution: f32 = clause
                .terms
                .iter()
                .zip(&per_term)
                .filter(|(t, _)| {
                    if seen.contains(&t.as_str()) {
                        false
                    } else {
                        seen.push(t.as_str());
                        true
                    }
                })
                .map(|(t, p)| idf[t.as_str()] * tfnorm(p))
                .sum();
            add(id, contribution, clause);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fulltext::Analyzer;
    use crate::fulltext::posting::term_map;
    use crate::fulltext::query::parse_query;

    /// Build posting lists for a tiny corpus of `(id, text)` documents.
    fn corpus(docs: &[(u64, &str)]) -> (HashMap<String, PostingList>, CorpusStats) {
        let mut postings: HashMap<String, PostingList> = HashMap::new();
        let mut total = 0u64;
        for (id, text) in docs {
            let tokens = Analyzer::Simple.analyze(text);
            total += tokens.len() as u64;
            for (term, positions) in term_map(&tokens) {
                postings.entry(term.to_string()).or_default().push((
                    *id,
                    Posting {
                        doc_len: tokens.len() as u32,
                        tf: positions.len() as u32,
                        positions,
                    },
                ));
            }
        }
        (
            postings,
            CorpusStats {
                doc_count: docs.len() as u64,
                total_tokens: total,
            },
        )
    }

    fn run(docs: &[(u64, &str)], q: &str, k: usize, restrict: Option<&HashSet<u64>>) -> Vec<FulltextHit> {
        let (owned, stats) = corpus(docs);
        let postings: HashMap<&str, PostingList> =
            owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let query = parse_query(q, Analyzer::Simple).unwrap();
        score_query(&query, &postings, stats, restrict, k)
    }
    fn ids(hits: &[FulltextHit]) -> Vec<u64> {
        hits.iter().map(|h| h.object_id).collect()
    }

    const DOCS: &[(u64, &str)] = &[
        (1, "the invoice for the invoice run"),          // invoice ×2, len 6
        (2, "invoice 4471 is overdue"),                   // invoice ×1, len 4
        (3, "consensus in distributed systems"),          // len 4
        (4, "distributed consensus is hard, consensus"),  // phrase present, len 5
        (5, "hard systems"),                              // len 2
    ];

    #[test]
    fn single_term_ranks_by_tf_and_length() {
        let hits = run(DOCS, "invoice", 10, None);
        assert_eq!(ids(&hits), vec![1, 2], "tf 2 beats tf 1");
        assert!(hits[0].score > hits[1].score);
        assert!(hits.iter().all(|h| h.score.is_finite() && h.score > 0.0));
        // Same tf: the shorter document wins (length normalization).
        let hits = run(DOCS, "hard", 10, None);
        assert_eq!(ids(&hits), vec![5, 4]);
    }

    #[test]
    fn terms_are_or_ed_and_scores_add() {
        let hits = run(DOCS, "invoice systems", 10, None);
        assert_eq!(hits.len(), 4);
        let set: HashSet<u64> = ids(&hits).into_iter().collect();
        assert_eq!(set, [1u64, 2, 3, 5].into_iter().collect());
        // A rarer term (4471, df 1) outweighs a commoner one at equal tf.
        let hits = run(DOCS, "4471 consensus", 10, None);
        assert_eq!(hits[0].object_id, 2);
    }

    #[test]
    fn required_terms_intersect_but_optional_still_score() {
        assert_eq!(ids(&run(DOCS, "+invoice +4471", 10, None)), vec![2]);
        assert!(run(DOCS, "+invoice +nowhere", 10, None).is_empty());
        // Required consensus, optional hard: doc 4 (has both) beats doc 3.
        let hits = run(DOCS, "+consensus hard", 10, None);
        assert_eq!(ids(&hits), vec![4, 3]);
        // A required clause alone still counts toward the score.
        let a = run(DOCS, "+invoice", 10, None);
        let b = run(DOCS, "invoice", 10, None);
        assert_eq!(a, b);
    }

    #[test]
    fn phrase_requires_adjacency_in_order() {
        assert_eq!(ids(&run(DOCS, "\"distributed consensus\"", 10, None)), vec![4]);
        // Reverse order is not the phrase.
        assert!(run(DOCS, "\"consensus distributed\"", 10, None).is_empty());
        // Words present but not adjacent (doc 3: consensus … distributed).
        assert!(run(DOCS, "\"consensus systems\"", 10, None).is_empty());
        // Three-term phrase and a repeated-word phrase.
        assert_eq!(ids(&run(DOCS, "\"the invoice run\"", 10, None)), vec![1]);
        assert_eq!(ids(&run(DOCS, "\"invoice for the invoice\"", 10, None)), vec![1]);
        // A repeated term inside a phrase is scored once: the phrase with the
        // repeat scores the same as its distinct-term set would.
        let repeated = run(DOCS, "\"invoice for the invoice\"", 10, None)[0].score;
        let distinct = run(DOCS, "+invoice +for +the", 10, None)
            .iter()
            .find(|h| h.object_id == 1)
            .unwrap()
            .score;
        assert!((repeated - distinct).abs() < 1e-6, "{repeated} vs {distinct}");
        // Phrase as an optional clause alongside a term: union.
        let hits = run(DOCS, "\"distributed consensus\" overdue", 10, None);
        assert_eq!(ids(&hits).len(), 2);
        // Required phrase + optional term: intersection on the phrase.
        assert_eq!(ids(&run(DOCS, "+\"distributed consensus\" overdue", 10, None)), vec![4]);
        // A phrase term absent from the corpus matches nothing (no panic).
        assert!(run(DOCS, "\"distributed nowhere\"", 10, None).is_empty());
    }

    #[test]
    fn phrase_offsets_require_the_same_gap_in_the_document() {
        // doc 1: state@0 of@1 the@2 art@3 · doc 2: state@0 art@1 (simple
        // analyzer keeps every word, so the positions are literal).
        let docs: &[(u64, &str)] = &[(1, "state of the art"), (2, "state art")];
        let (owned, stats) = corpus(docs);
        let postings: HashMap<&str, PostingList> =
            owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let phrase = |offsets: Vec<u32>| ParsedQuery {
            clauses: vec![Clause {
                required: false,
                terms: vec!["state".into(), "art".into()],
                prefix: false,
                offsets,
            }],
        };
        // What the english analyzer produces for "state of the art".
        assert_eq!(ids(&score_query(&phrase(vec![0, 3]), &postings, stats, None, 10)), vec![1]);
        // Adjacent: only the adjacent document.
        assert_eq!(ids(&score_query(&phrase(vec![0, 1]), &postings, stats, None, 10)), vec![2]);
        // No offsets = consecutive (the pre-#20 contract).
        assert_eq!(ids(&score_query(&phrase(vec![]), &postings, stats, None, 10)), vec![2]);
        // An empty query (every word was a stop word) matches nothing.
        let empty = ParsedQuery { clauses: vec![] };
        assert!(score_query(&empty, &postings, stats, None, 10).is_empty());
    }

    #[test]
    fn prefix_clause_scores_its_merged_expansion_as_one_term() {
        // Corpus: 1 "camera", 2 "cameras", 3 "camera cameras" (both), 4 "cam".
        let (owned, stats) = corpus(&[(1, "camera x"), (2, "cameras x"), (3, "camera cameras"), (4, "cam x")]);
        let mut postings: HashMap<&str, PostingList> =
            owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        // What the storage layer produces for `cam*`: the union of camera /
        // cameras / cam merged per document (tf summed, doc_len shared).
        let mut merged: HashMap<u64, Posting> = HashMap::new();
        for t in ["camera", "cameras", "cam"] {
            for (id, p) in &owned[t] {
                let e = merged.entry(*id).or_insert(Posting { doc_len: p.doc_len, tf: 0, positions: vec![] });
                e.tf += p.tf;
            }
        }
        postings.insert("cam*", merged.into_iter().collect());

        let query = parse_query("cam*", Analyzer::Simple).unwrap();
        let hits = score_query(&query, &postings, stats, None, 10);
        // Every document matches; doc 3 (tf 2) ranks first; the rest tie on
        // tf 1 / len 2 and break on id.
        assert_eq!(ids(&hits), vec![3, 1, 2, 4]);
        assert!(hits[0].score > hits[1].score);
        assert_eq!(hits[1].score, hits[2].score);
        // ONE idf for the whole expansion: the score of doc 1 equals what a
        // plain term with df 4 / tf 1 would get — i.e. the same as querying
        // a synthetic term with the merged list.
        let mut only: HashMap<&str, PostingList> = HashMap::new();
        only.insert("cam", postings["cam*"].clone());
        let plain = score_query(&parse_query("cam", Analyzer::Simple).unwrap(), &only, stats, None, 10);
        assert_eq!(plain, hits);
        // Required prefix + optional term: everything with the prefix
        // matches; the three that also have `x` outrank doc 3, which has
        // the prefix twice but no `x`.
        let hits = score_query(&parse_query("+cam* x", Analyzer::Simple).unwrap(), &postings, stats, None, 10);
        assert_eq!(ids(&hits), vec![1, 2, 4, 3]);
        // Required term the prefix docs lack: intersection empties.
        let hits = score_query(&parse_query("cam* +zzz", Analyzer::Simple).unwrap(), &postings, stats, None, 10);
        assert!(hits.is_empty());
        // A prefix and the exact term of the same text are different keys.
        let hits = score_query(&parse_query("+cam +cam*", Analyzer::Simple).unwrap(), &postings, stats, None, 10);
        assert_eq!(ids(&hits), vec![4]);
        // An unexpanded prefix (nothing in the index) matches nothing.
        let query = parse_query("zzz*", Analyzer::Simple).unwrap();
        assert!(score_query(&query, &postings, stats, None, 10).is_empty());
    }

    #[test]
    fn k_truncates_and_ties_break_on_id() {
        let docs = &[(9, "same same"), (3, "same same"), (5, "same same"), (1, "other")];
        let hits = run(docs, "same", 2, None);
        assert_eq!(ids(&hits), vec![3, 5]);
        assert_eq!(hits[0].score, hits[1].score);
        assert!(run(docs, "same", 0, None).is_empty());
    }

    #[test]
    fn restrict_filters_before_the_top_k_cut() {
        let docs = &[(1, "x x x"), (2, "x x"), (3, "x"), (4, "x")];
        let restrict: HashSet<u64> = [3u64, 4].into_iter().collect();
        let hits = run(docs, "x", 2, Some(&restrict));
        assert_eq!(ids(&hits), vec![3, 4]);
        let none: HashSet<u64> = HashSet::new();
        assert!(run(docs, "x", 2, Some(&none)).is_empty());
    }

    #[test]
    fn unknown_term_and_empty_corpus() {
        assert!(run(DOCS, "zzzz", 10, None).is_empty());
        assert!(run(&[], "invoice", 10, None).is_empty());
        // Stats lagging behind df never yields a negative / NaN score.
        let (owned, _) = corpus(DOCS);
        let postings: HashMap<&str, PostingList> =
            owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let query = parse_query("invoice", Analyzer::Simple).unwrap();
        let lagging = CorpusStats {
            doc_count: 1,
            total_tokens: 1,
        };
        let hits = score_query(&query, &postings, lagging, None, 10);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.score.is_finite() && h.score > 0.0));
    }
}
