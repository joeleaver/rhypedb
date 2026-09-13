//! Full-text search (`@fulltext` fields, `Type.matches(.field, "terms", k: N)`).
//!
//! This module owns everything text-search specific that is not a write-path
//! hook in `database.rs`:
//!
//! * [`analyzer`] — the text analyzers that turn a `String` value (or a query
//!   clause) into a position-stamped token stream. The SAME analyzer runs on
//!   the write path and the query path; any drift between the two silently
//!   breaks matching, so there is exactly one implementation.
//!
//! Later increments add the posting codec, the write-path index maintenance,
//! BM25 scoring and the background backfill here.

pub mod analyzer;

pub use analyzer::{Analyzer, Token, MAX_TERM_BYTES};
