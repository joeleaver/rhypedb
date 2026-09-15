//! The two phases of a full-text search, split so a caller can decide which
//! documents may take part BEFORE anything is ranked.
//!
//! [`Database::fulltext_candidates`](crate::database::Database::fulltext_candidates)
//! reads every posting the query needs at one snapshot into a
//! [`FulltextCandidates`]; [`FulltextCandidates::score`] ranks them over the
//! whole field (the plain `.matches` path), and
//! [`FulltextCandidates::score_visible`] ranks them over a caller-chosen set
//! of VISIBLE documents only.
//!
//! `score_visible` exists for security rules: a document the caller may not
//! read must not influence what they get back. So it drops invisible
//! postings before anything is counted — every term's `df`, the corpus
//! `doc_count` / `total_tokens` (taken from the visible candidates, not the
//! field-wide stats), the prefix-expansion cap and the top-`k` cut all see
//! only visible documents. Otherwise a BM25 score, the number of rows under
//! `k`, or a "prefix matches too many terms" error would each be an oracle
//! for text in documents the caller cannot read.

use std::collections::{HashMap, HashSet};

use super::posting::Posting;
use super::query::ParsedQuery;
use super::search::{CorpusStats, FulltextHit, PostingList, score_query};
use crate::error::{EngineError, EngineResult};

/// Everything a query's postings scan produced, ready to rank.
#[derive(Debug)]
pub struct FulltextCandidates {
    pub(crate) type_name: String,
    pub(crate) field_name: String,
    pub(crate) parsed: ParsedQuery,
    /// Posting key → list: exact terms, plus prefix keys (`"cam*"`) whose
    /// expansion was merged and capped during the scan.
    pub(crate) postings: HashMap<String, PostingList>,
    /// Prefix keys whose cap was DEFERRED to scoring: one list per expanded
    /// term, unmerged, so the cap can count only terms with a visible posting.
    pub(crate) prefix_terms: HashMap<String, Vec<PostingList>>,
    /// Field-wide corpus statistics at the scan.
    pub(crate) stats: CorpusStats,
    /// Posting rows examined, tombstones included.
    pub postings_scanned: u64,
}

impl FulltextCandidates {
    /// Every document with at least one posting for the query, ascending.
    pub fn candidate_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .postings
            .values()
            .chain(self.prefix_terms.values().flatten())
            .flat_map(|list| list.iter().map(|(id, _)| *id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Rows scoring will walk: every clause visits every posting of each of
    /// its terms (a phrase once per word, repeats included). A governor
    /// charges this BEFORE ranking so clauses × postings is budgeted, not
    /// just the postings fetched once per distinct term.
    pub fn scoring_work(&self) -> u64 {
        let len = |key: &str| -> u64 {
            self.postings
                .get(key)
                .map(|l| l.len() as u64)
                .or_else(|| self.prefix_terms.get(key).map(|ls| ls.iter().map(|l| l.len() as u64).sum()))
                .unwrap_or(0)
        };
        self.parsed
            .clauses
            .iter()
            .flat_map(|c| c.terms.iter())
            .map(|t| len(t))
            .fold(0u64, u64::saturating_add)
    }

    /// Rank over the whole field: field-wide statistics, `restrict` (when
    /// given) applied before the top-`k` cut.
    pub fn score(&self, restrict: Option<&HashSet<u64>>, k: usize) -> EngineResult<Vec<FulltextHit>> {
        let merged = self.merge_prefixes(None)?;
        let postings: HashMap<&str, &PostingList> = self
            .postings
            .iter()
            .map(|(key, list)| (key.as_str(), list))
            .chain(merged.iter().map(|(key, list)| (*key, list)))
            .collect();
        Ok(score_query(&self.parsed, &postings, self.stats, restrict, k))
    }

    /// Rank over `visible` documents only (see the module doc): invisible
    /// postings are dropped first, and `df`, the corpus statistics, the
    /// prefix-expansion cap and the top-`k` cut are all computed from what
    /// is left.
    pub fn score_visible(&self, visible: &HashSet<u64>, k: usize) -> EngineResult<Vec<FulltextHit>> {
        let merged = self.merge_prefixes(Some(visible))?;
        let postings: HashMap<&str, PostingList> = self
            .postings
            .iter()
            .map(|(key, list)| {
                let kept = list.iter().filter(|(id, _)| visible.contains(id)).cloned().collect();
                (key.as_str(), kept)
            })
            .chain(merged.into_iter())
            .collect();
        // One doc_len per document (every posting of a document carries it).
        let mut doc_lens: HashMap<u64, u32> = HashMap::new();
        for list in postings.values() {
            for (id, p) in list {
                doc_lens.entry(*id).or_insert(p.doc_len);
            }
        }
        let stats = CorpusStats {
            doc_count: doc_lens.len() as u64,
            total_tokens: doc_lens.values().map(|&l| l as u64).sum(),
        };
        Ok(score_query(&self.parsed, &postings, stats, None, k))
    }

    /// Merge each deferred prefix expansion into one list per document
    /// (`tf` summed), counting toward the cap only terms with at least one
    /// posting in `visible` (all terms when `None`).
    fn merge_prefixes(&self, visible: Option<&HashSet<u64>>) -> EngineResult<HashMap<&str, PostingList>> {
        let mut out = HashMap::new();
        for (key, lists) in &self.prefix_terms {
            let mut merged: HashMap<u64, Posting> = HashMap::new();
            let mut terms = 0usize;
            for list in lists {
                let mut counted = false;
                for (id, p) in list.iter().filter(|(id, _)| visible.is_none_or(|v| v.contains(id))) {
                    if !counted {
                        counted = true;
                        terms += 1;
                        if terms > super::MAX_PREFIX_EXPANSION {
                            return Err(EngineError::FulltextQuery(prefix_cap_message(key)));
                        }
                    }
                    match merged.entry(*id) {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(Posting { doc_len: p.doc_len, tf: p.tf, positions: Vec::new() });
                        }
                        std::collections::hash_map::Entry::Occupied(mut o) => {
                            if o.get().doc_len != p.doc_len {
                                return Err(EngineError::FulltextIndexCorrupt {
                                    type_name: self.type_name.clone(),
                                    field: self.field_name.clone(),
                                    detail: format!(
                                        "object {id} has postings with doc_len {} and {} under prefix {key:?}",
                                        o.get().doc_len,
                                        p.doc_len
                                    ),
                                });
                            }
                            o.get_mut().tf = o.get().tf.saturating_add(p.tf);
                        }
                    }
                }
            }
            out.insert(key.as_str(), merged.into_iter().collect());
        }
        Ok(out)
    }
}

/// The "prefix expands too far" message for posting key `key` (`"cam*"`).
pub(crate) fn prefix_cap_message(key: &str) -> String {
    format!(
        "prefix term \"{key}\" matches more than {} indexed terms; use a longer prefix",
        super::MAX_PREFIX_EXPANSION
    )
}
