//! Parsed query cache.
//!
//! Avoids re-parsing the same query string repeatedly. Bounded by entry count;
//! when full, the cache clears all entries on the next insert (generation-based
//! eviction). Adequate for the typical workload of a small set of repeated
//! queries; degrades gracefully under high-cardinality query strings.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use rhypedb_query::ast::Query;
use rhypedb_query::error::QueryError;
use rhypedb_query::parser::parse_query;

pub const DEFAULT_CACHE_SIZE: usize = 256;

pub struct QueryCache {
    inner: RwLock<HashMap<String, Arc<Query>>>,
    max_size: usize,
}

impl QueryCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            max_size,
        }
    }

    /// Look up or parse a query. Returns a shared reference to the parsed AST.
    /// On parse error, the error is returned and nothing is cached.
    pub fn get_or_parse(&self, query_text: &str) -> Result<Arc<Query>, QueryError> {
        // Fast path: read lock, return cached.
        {
            let cache = self.inner.read();
            if let Some(q) = cache.get(query_text) {
                return Ok(Arc::clone(q));
            }
        }

        // Slow path: parse before acquiring the write lock to avoid holding it
        // across the parse.
        let parsed = parse_query(query_text)?;
        let arc = Arc::new(parsed);

        {
            let mut cache = self.inner.write();
            // Re-check after acquiring write lock — another thread may have
            // inserted while we were parsing.
            if let Some(q) = cache.get(query_text) {
                return Ok(Arc::clone(q));
            }
            if cache.len() >= self.max_size {
                cache.clear();
            }
            cache.insert(query_text.to_string(), Arc::clone(&arc));
        }

        Ok(arc)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_miss_then_hit_returns_same_arc() {
        let cache = QueryCache::new(8);
        let q1 = cache.get_or_parse("User.get(1)").unwrap();
        assert_eq!(cache.len(), 1);
        let q2 = cache.get_or_parse("User.get(1)").unwrap();
        // Same Arc — pointer equality.
        assert!(Arc::ptr_eq(&q1, &q2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn different_queries_are_separate_entries() {
        let cache = QueryCache::new(8);
        let q1 = cache.get_or_parse("User.get(1)").unwrap();
        let q2 = cache.get_or_parse("User.get(2)").unwrap();
        assert!(!Arc::ptr_eq(&q1, &q2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn parse_error_not_cached() {
        let cache = QueryCache::new(8);
        let err = cache.get_or_parse("NOT VALID");
        assert!(err.is_err());
        assert!(cache.is_empty());
    }

    #[test]
    fn overflow_clears_cache() {
        let cache = QueryCache::new(3);
        cache.get_or_parse("User.get(1)").unwrap();
        cache.get_or_parse("User.get(2)").unwrap();
        cache.get_or_parse("User.get(3)").unwrap();
        assert_eq!(cache.len(), 3);
        cache.get_or_parse("User.get(4)").unwrap();
        // After overflow, cache is cleared and only the new entry is present.
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn concurrent_access_is_safe() {
        use std::thread;
        let cache = Arc::new(QueryCache::new(64));
        let mut handles = Vec::new();
        for thread_id in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let q = format!("User.get({})", thread_id * 100 + i);
                    cache.get_or_parse(&q).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Cache should have entries (may have been cleared multiple times due
        // to overflow, but should still hold the most recent batch).
        assert!(!cache.is_empty());
    }
}
