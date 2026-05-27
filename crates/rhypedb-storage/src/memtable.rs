use bytes::Bytes;
use crossbeam_skiplist::SkipMap;

use crate::key::InternalKey;

/// Value stored in the memtable. `None` represents a tombstone (deletion).
pub type MemValue = Option<Bytes>;

/// In-memory sorted store backed by a lock-free skip list.
///
/// Supports concurrent reads and writes without locking. Entries are
/// keyed by `InternalKey` (user_key + inverted version) so that iterating
/// forward yields the newest version of each key first.
pub struct MemTable {
    map: SkipMap<Bytes, MemValue>,
    size_bytes: std::sync::atomic::AtomicUsize,
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MemTable {
    pub fn new() -> Self {
        Self {
            map: SkipMap::new(),
            size_bytes: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Insert a versioned key-value pair.
    pub fn put(&self, user_key: &[u8], version: u64, value: Bytes) {
        let ik = InternalKey::new(user_key, version);
        let added = ik.as_bytes().len() + value.len();
        self.map
            .insert(Bytes::copy_from_slice(ik.as_bytes()), Some(value));
        self.size_bytes
            .fetch_add(added, std::sync::atomic::Ordering::Relaxed);
    }

    /// Insert a tombstone for a versioned key.
    pub fn delete(&self, user_key: &[u8], version: u64) {
        let ik = InternalKey::new(user_key, version);
        let added = ik.as_bytes().len();
        self.map
            .insert(Bytes::copy_from_slice(ik.as_bytes()), None);
        self.size_bytes
            .fetch_add(added, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the latest value for a user key that is visible at `version`.
    ///
    /// Returns `Some(Some(value))` if found, `Some(None)` if deleted (tombstone),
    /// or `None` if no entry exists for this key at or before `version`.
    pub fn get(&self, user_key: &[u8], version: u64) -> Option<MemValue> {
        // We want the first internal key whose user_key matches and whose
        // version <= the requested version. Because versions are inverted in
        // the encoding, the scan key with our target version is the lower bound.
        let scan_key = InternalKey::new(user_key, version);
        let scan_bytes = Bytes::copy_from_slice(scan_key.as_bytes());

        // Scan forward from the target version.
        for entry in self.map.range(scan_bytes..) {
            let ik = InternalKey::new(&[], 0); // placeholder
            let entry_key = entry.key();

            // Decode the user key portion (everything except last 8 bytes).
            if entry_key.len() < 8 {
                continue;
            }
            let entry_user_key = &entry_key[..entry_key.len() - 8];

            if entry_user_key != user_key {
                // Passed our key range.
                return None;
            }

            // This entry's version is <= our target version (due to inverted ordering).
            drop(ik);
            return Some(entry.value().clone());
        }

        None
    }

    /// Approximate size in bytes of all entries in this memtable.
    pub fn approximate_size(&self) -> usize {
        self.size_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns true if the memtable has no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate all entries in sorted order. Used for flushing to SST.
    pub fn iter(&self) -> impl Iterator<Item = (Bytes, MemValue)> + '_ {
        self.map.iter().map(|e| (e.key().clone(), e.value().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let mt = MemTable::new();
        let key = b"o:test";
        mt.put(key, 1, Bytes::from("hello"));

        let val = mt.get(key, 1);
        assert_eq!(val, Some(Some(Bytes::from("hello"))));
    }

    #[test]
    fn get_returns_none_for_missing_key() {
        let mt = MemTable::new();
        assert_eq!(mt.get(b"missing", 1), None);
    }

    #[test]
    fn get_respects_version_visibility() {
        let mt = MemTable::new();
        let key = b"o:versioned";

        mt.put(key, 1, Bytes::from("v1"));
        mt.put(key, 5, Bytes::from("v5"));
        mt.put(key, 10, Bytes::from("v10"));

        // Reading at version 1 should see v1
        assert_eq!(mt.get(key, 1), Some(Some(Bytes::from("v1"))));

        // Reading at version 7 should see v5 (latest <= 7)
        assert_eq!(mt.get(key, 7), Some(Some(Bytes::from("v5"))));

        // Reading at version 10 should see v10
        assert_eq!(mt.get(key, 10), Some(Some(Bytes::from("v10"))));

        // Reading at version 0 should see nothing
        assert_eq!(mt.get(key, 0), None);
    }

    #[test]
    fn delete_returns_tombstone() {
        let mt = MemTable::new();
        let key = b"o:deleted";

        mt.put(key, 1, Bytes::from("alive"));
        mt.delete(key, 5);

        assert_eq!(mt.get(key, 1), Some(Some(Bytes::from("alive"))));
        assert_eq!(mt.get(key, 5), Some(None)); // tombstone
        assert_eq!(mt.get(key, 10), Some(None)); // still deleted
    }

    #[test]
    fn iter_is_sorted() {
        let mt = MemTable::new();
        mt.put(b"c", 1, Bytes::from("3"));
        mt.put(b"a", 1, Bytes::from("1"));
        mt.put(b"b", 1, Bytes::from("2"));

        let keys: Vec<_> = mt.iter().map(|(k, _)| k).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn approximate_size_grows() {
        let mt = MemTable::new();
        assert_eq!(mt.approximate_size(), 0);

        mt.put(b"key", 1, Bytes::from("value"));
        assert!(mt.approximate_size() > 0);
    }
}
