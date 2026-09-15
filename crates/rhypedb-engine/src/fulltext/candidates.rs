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
use super::search::{CorpusStats, FulltextHit, PostingList, RestrictScope, score_query_with};
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

    /// Work scoring will do, in rows: every clause walks every posting of each of its
    /// terms (a phrase once per word, repeats included), and a phrase additionally walks
    /// its first word's positions in every document, probing each later word — Σ tf₀ ×
    /// (words − 1). A governor charges this BEFORE ranking, so a query of many phrases
    /// over high-frequency documents is budgeted, not just the postings fetched once.
    pub fn scoring_work(&self) -> u64 {
        let list = |key: &str| self.postings.get(key);
        let len = |key: &str| -> u64 {
            list(key)
                .map(|l| l.len() as u64)
                .or_else(|| self.prefix_terms.get(key).map(|ls| ls.iter().map(|l| l.len() as u64).sum()))
                .unwrap_or(0)
        };
        self.parsed
            .clauses
            .iter()
            .map(|c| {
                let walks = c.terms.iter().map(|t| len(t)).fold(0u64, u64::saturating_add);
                let positions = if c.is_phrase() {
                    let tf0: u64 = list(&c.terms[0]).map_or(0, |l| l.iter().map(|(_, p)| p.tf as u64).sum());
                    tf0.saturating_mul(c.terms.len() as u64 - 1)
                } else {
                    0
                };
                walks.saturating_add(positions)
            })
            .fold(0u64, u64::saturating_add)
    }

    /// Rank over the whole field: field-wide statistics, `restrict` (when
    /// given) applied before the top-`k` cut. Fails with
    /// `FulltextDeadlineExceeded` past `deadline`.
    pub fn score(
        &self,
        restrict: Option<&HashSet<u64>>,
        k: usize,
        deadline: Option<std::time::Instant>,
    ) -> EngineResult<Vec<FulltextHit>> {
        let merged = self.merge_prefixes(None)?;
        let postings = self.lists(&merged);
        score_query_with(&self.parsed, &postings, self.stats, restrict, RestrictScope::RankOnly, k, deadline)
            .map_err(|_| self.deadline_error())
    }

    /// Rank over `visible` documents only (see the module doc): the visible set
    /// is the corpus — `df`, the statistics, the prefix-expansion cap and the
    /// top-`k` cut are all computed from it. Works on the collected lists in
    /// place (no copy of the postings).
    pub fn score_visible(
        &self,
        visible: &HashSet<u64>,
        k: usize,
        deadline: Option<std::time::Instant>,
    ) -> EngineResult<Vec<FulltextHit>> {
        let merged = self.merge_prefixes(Some(visible))?;
        let postings = self.lists(&merged);
        // One doc_len per visible document (every posting of a document carries it).
        let mut doc_lens: HashMap<u64, u32> = HashMap::new();
        for list in postings.values() {
            for (id, p) in list.iter().filter(|(id, _)| visible.contains(id)) {
                doc_lens.entry(*id).or_insert(p.doc_len);
            }
        }
        let stats = CorpusStats {
            doc_count: doc_lens.len() as u64,
            total_tokens: doc_lens.values().map(|&l| l as u64).sum(),
        };
        score_query_with(&self.parsed, &postings, stats, Some(visible), RestrictScope::Corpus, k, deadline)
            .map_err(|_| self.deadline_error())
    }

    /// Every posting list by key: the collected exact terms plus `merged` prefix lists.
    fn lists<'a>(&'a self, merged: &'a HashMap<&'a str, PostingList>) -> HashMap<&'a str, &'a PostingList> {
        self.postings
            .iter()
            .map(|(key, list)| (key.as_str(), list))
            .chain(merged.iter().map(|(key, list)| (*key, list)))
            .collect()
    }

    fn deadline_error(&self) -> EngineError {
        EngineError::FulltextDeadlineExceeded {
            type_name: self.type_name.clone(),
            field: self.field_name.clone(),
        }
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
                            return Err(EngineError::FulltextQuery(prefix_cap_message(key, super::MAX_PREFIX_EXPANSION)));
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

/// The "prefix expands too far" message for posting key `key` (`"cam*"`) and the `cap` hit.
pub(crate) fn prefix_cap_message(key: &str, cap: usize) -> String {
    format!("prefix term \"{key}\" matches more than {cap} indexed terms; use a longer prefix")
}
