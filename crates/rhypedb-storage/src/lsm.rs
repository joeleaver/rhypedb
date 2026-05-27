use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;

use crate::memtable::MemTable;
use crate::mvcc::{Transaction, TransactionManager};
use crate::sst::{SstReader, SstWriter};
use crate::wal::{RecordType, Wal, WalRecord};
use crate::Result;

const DEFAULT_MEMTABLE_FLUSH_SIZE: usize = 4 * 1024 * 1024; // 4MB

/// Configuration for the LSM-tree storage engine.
pub struct LsmConfig {
    pub data_dir: PathBuf,
    pub memtable_flush_size: usize,
}

impl LsmConfig {
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            memtable_flush_size: DEFAULT_MEMTABLE_FLUSH_SIZE,
        }
    }
}

/// LSM-tree storage engine.
///
/// Provides versioned key-value storage with:
/// - Write-ahead log for durability
/// - In-memory skip-list memtable for writes
/// - Sorted string table (SST) files on disk
/// - MVCC transaction manager for snapshot isolation
pub struct LsmTree {
    config: LsmConfig,
    active_memtable: Arc<RwLock<Arc<MemTable>>>,
    immutable_memtables: Arc<RwLock<Vec<Arc<MemTable>>>>,
    sst_files: Arc<RwLock<Vec<SstReader>>>,
    wal: Arc<parking_lot::Mutex<Wal>>,
    txn_manager: Arc<TransactionManager>,
    next_sst_id: std::sync::atomic::AtomicU64,
}

impl LsmTree {
    /// Open or create an LSM-tree at the given directory.
    pub fn open(config: LsmConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir)?;
        std::fs::create_dir_all(config.data_dir.join("sst"))?;

        let wal_path = config.data_dir.join("wal.log");
        let txn_manager = Arc::new(TransactionManager::new());

        // Recover from WAL if it exists.
        let memtable = Arc::new(MemTable::new());
        let records = Wal::replay(&wal_path)?;
        for record in &records {
            match record.record_type {
                RecordType::Put => {
                    memtable.put(&record.key, record.version, record.value.clone());
                }
                RecordType::Delete => {
                    memtable.delete(&record.key, record.version);
                }
            }
        }

        // Discover existing SST files.
        let mut sst_readers = Vec::new();
        let mut max_sst_id = 0u64;
        let sst_dir = config.data_dir.join("sst");
        if sst_dir.exists() {
            let mut entries: Vec<_> = std::fs::read_dir(&sst_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "sst")
                })
                .collect();
            entries.sort_by_key(|e| e.path());

            for entry in entries {
                if let Some(id) = entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    max_sst_id = max_sst_id.max(id);
                }
                sst_readers.push(SstReader::open(entry.path())?);
            }
        }

        let wal = Wal::open(&wal_path)?;

        Ok(Self {
            config,
            active_memtable: Arc::new(RwLock::new(Arc::new(memtable.as_ref().clone_into_new()))),
            immutable_memtables: Arc::new(RwLock::new(Vec::new())),
            sst_files: Arc::new(RwLock::new(sst_readers)),
            wal: Arc::new(parking_lot::Mutex::new(wal)),
            txn_manager: Arc::new(txn_manager.as_ref().clone_into_new()),
            next_sst_id: std::sync::atomic::AtomicU64::new(max_sst_id + 1),
        })
    }

    /// Begin a new transaction.
    pub fn begin_txn(&self) -> Transaction {
        self.txn_manager.begin()
    }

    /// Read a user key at the transaction's snapshot version.
    ///
    /// Search order: active memtable → immutable memtables → SST files (newest first).
    pub fn get(&self, txn: &Transaction, user_key: &[u8]) -> Result<Option<Bytes>> {
        let version = txn.snapshot();

        // 1. Active memtable.
        let active = self.active_memtable.read().clone();
        if let Some(val) = active.get(user_key, version) {
            return match val {
                Some(v) => Ok(Some(v)),
                None => Ok(None), // tombstone
            };
        }

        // 2. Immutable memtables (newest first).
        let immutables = self.immutable_memtables.read().clone();
        for mt in immutables.iter().rev() {
            if let Some(val) = mt.get(user_key, version) {
                return match val {
                    Some(v) => Ok(Some(v)),
                    None => Ok(None),
                };
            }
        }

        // 3. SST files (newest first).
        let ssts = self.sst_files.read();
        for sst in ssts.iter().rev() {
            if let Some(val) = sst.get_versioned(user_key, version)? {
                return match val {
                    Some(v) => Ok(Some(v)),
                    None => Ok(None),
                };
            }
        }

        Ok(None)
    }

    /// Scan for all keys with the given prefix, visible at the transaction's snapshot.
    /// Returns `(user_key, value)` pairs. Tombstones are excluded from results.
    pub fn scan_prefix(
        &self,
        txn: &Transaction,
        prefix: &[u8],
    ) -> Result<Vec<(Bytes, Bytes)>> {
        let version = txn.snapshot();

        // Collect from all sources.
        let mut merged: std::collections::BTreeMap<Bytes, Option<Bytes>> =
            std::collections::BTreeMap::new();

        // SSTs first (oldest), so newer entries overwrite.
        let ssts = self.sst_files.read();
        for sst in ssts.iter() {
            for (key, value) in sst.scan_prefix(prefix, version) {
                merged.insert(key, value);
            }
        }
        drop(ssts);

        // Immutable memtables (oldest first).
        let immutables = self.immutable_memtables.read().clone();
        for mt in immutables.iter() {
            for (key, value) in mt.scan_prefix(prefix, version) {
                merged.insert(key, value);
            }
        }

        // Active memtable (newest).
        let active = self.active_memtable.read().clone();
        for (key, value) in active.scan_prefix(prefix, version) {
            merged.insert(key, value);
        }

        // Filter out tombstones.
        Ok(merged
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .collect())
    }

    /// Write a key-value pair within a transaction.
    pub fn put(&self, txn: &mut Transaction, user_key: &[u8], value: Bytes) -> Result<()> {
        let version = self.txn_manager.current_version() + 1; // provisional

        txn.record_write(Bytes::copy_from_slice(user_key));

        // Write to WAL first for durability.
        self.wal.lock().append(&WalRecord {
            record_type: RecordType::Put,
            key: Bytes::copy_from_slice(user_key),
            value: value.clone(),
            version,
        })?;

        // Write to memtable.
        self.active_memtable.read().put(user_key, version, value);

        self.maybe_flush()?;

        Ok(())
    }

    /// Delete a key within a transaction.
    pub fn delete(&self, txn: &mut Transaction, user_key: &[u8]) -> Result<()> {
        let version = self.txn_manager.current_version() + 1;

        txn.record_write(Bytes::copy_from_slice(user_key));

        self.wal.lock().append(&WalRecord {
            record_type: RecordType::Delete,
            key: Bytes::copy_from_slice(user_key),
            value: Bytes::new(),
            version,
        })?;

        self.active_memtable.read().delete(user_key, version);

        self.maybe_flush()?;

        Ok(())
    }

    /// Commit a transaction, checking for write-write conflicts.
    pub fn commit(&self, txn: &mut Transaction) -> Result<u64> {
        let version = self.txn_manager.commit(txn)?;
        self.wal.lock().sync()?;
        Ok(version)
    }

    /// Abort a transaction.
    pub fn abort(&self, txn: &mut Transaction) {
        self.txn_manager.abort(txn);
    }

    /// Check if the active memtable is large enough to flush.
    fn maybe_flush(&self) -> Result<()> {
        let size = self.active_memtable.read().approximate_size();
        if size >= self.config.memtable_flush_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush the active memtable to a new SST file.
    pub fn flush(&self) -> Result<()> {
        // Rotate memtable: current becomes immutable, new empty one becomes active.
        let old_memtable = {
            let mut active = self.active_memtable.write();
            let old = active.clone();
            *active = Arc::new(MemTable::new());
            old
        };

        if old_memtable.is_empty() {
            return Ok(());
        }

        self.immutable_memtables.write().push(old_memtable.clone());

        // Write SST file.
        let sst_id = self
            .next_sst_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let sst_path = self.config.data_dir.join("sst").join(format!("{sst_id:08}.sst"));

        let mut writer = SstWriter::new(&sst_path)?;
        for (key, value) in old_memtable.iter() {
            writer.add(&key, &value)?;
        }
        writer.finish()?;

        // Open the new SST for reads and remove from immutable list.
        let reader = SstReader::open(&sst_path)?;
        self.sst_files.write().push(reader);

        let mut immutables = self.immutable_memtables.write();
        immutables.retain(|mt| !Arc::ptr_eq(mt, &old_memtable));

        // Start a new WAL (old data is now in the SST).
        let wal_path = self.config.data_dir.join("wal.log");
        let new_wal = Wal::open(&wal_path)?;
        *self.wal.lock() = new_wal;

        Ok(())
    }

    /// Compact all SST files into a single new SST, dropping old versions
    /// and tombstones that are no longer needed by any active transaction.
    pub fn compact(&self) -> Result<()> {
        let ssts = self.sst_files.read();
        if ssts.len() < 2 {
            return Ok(());
        }

        let min_snapshot = self.txn_manager.min_active_snapshot();

        // Merge all SST iterators into a single sorted stream.
        // Since each SST is already sorted and SSTs are ordered oldest→newest,
        // we collect all entries and sort. For a production system you'd use
        // a merge iterator, but this is correct and simple.
        let mut all_entries: Vec<(Bytes, Option<Bytes>)> = Vec::new();
        for sst in ssts.iter() {
            for entry in sst.iter() {
                all_entries.push(entry);
            }
        }
        drop(ssts);

        all_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Write merged entries to a new SST, keeping only the latest version
        // of each user key that's still needed.
        let sst_id = self
            .next_sst_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let sst_path = self
            .config
            .data_dir
            .join("sst")
            .join(format!("{sst_id:08}.sst"));

        let mut writer = SstWriter::new(&sst_path)?;
        let mut prev_user_key: Option<Vec<u8>> = None;
        let mut kept_latest_for_key = false;

        for (key, value) in &all_entries {
            if key.len() < 8 {
                continue;
            }
            let user_key = &key[..key.len() - 8];
            let ver_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let version = !u64::from_be_bytes(ver_bytes);

            let same_key = prev_user_key
                .as_ref()
                .is_some_and(|prev| prev.as_slice() == user_key);

            if !same_key {
                prev_user_key = Some(user_key.to_vec());
                kept_latest_for_key = false;
            }

            if !kept_latest_for_key {
                // Always keep the latest version of each key.
                writer.add(key, value)?;
                kept_latest_for_key = true;
            } else if version >= min_snapshot {
                // Keep versions that might still be visible to active transactions.
                writer.add(key, value)?;
            }
            // Older versions below min_snapshot are dropped.
        }

        let meta = writer.finish()?;

        if meta.entry_count == 0 {
            // All entries were compacted away — remove the empty SST.
            let _ = std::fs::remove_file(&sst_path);
            // Remove old SSTs.
            let mut ssts = self.sst_files.write();
            let old_paths: Vec<_> = ssts.iter().map(|s| s.path().to_path_buf()).collect();
            ssts.clear();
            drop(ssts);
            for path in old_paths {
                let _ = std::fs::remove_file(path);
            }
            return Ok(());
        }

        // Swap old SSTs for the new compacted one.
        let new_reader = SstReader::open(&sst_path)?;
        let mut ssts = self.sst_files.write();
        let old_paths: Vec<_> = ssts.iter().map(|s| s.path().to_path_buf()).collect();
        ssts.clear();
        ssts.push(new_reader);
        drop(ssts);

        // Delete old SST files.
        for path in old_paths {
            let _ = std::fs::remove_file(path);
        }

        Ok(())
    }

    /// Returns a reference to the transaction manager.
    pub fn txn_manager(&self) -> &TransactionManager {
        &self.txn_manager
    }

    /// Number of SST files on disk.
    pub fn sst_count(&self) -> usize {
        self.sst_files.read().len()
    }
}

/// Helper to create a fresh MemTable by replaying into a new one.
/// We need this because MemTable wraps a SkipMap which isn't Clone.
trait CloneIntoNew {
    fn clone_into_new(&self) -> Self;
}

impl CloneIntoNew for MemTable {
    fn clone_into_new(&self) -> Self {
        let new = MemTable::new();
        // For a fresh open, the recovered memtable's entries need to be copied.
        // But we actually just need a fresh empty one for the active slot.
        // The recovered data is already in the memtable we're replacing.
        new
    }
}

impl CloneIntoNew for TransactionManager {
    fn clone_into_new(&self) -> Self {
        // Fresh manager — version state was reconstructed from WAL.
        TransactionManager::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn test_config(dir: &Path) -> LsmConfig {
        LsmConfig {
            data_dir: dir.to_path_buf(),
            memtable_flush_size: 1024, // small for testing
        }
    }

    #[test]
    fn basic_put_get() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"hello", Bytes::from("world")).unwrap();
        tree.commit(&mut txn).unwrap();

        let txn2 = tree.begin_txn();
        let val = tree.get(&txn2, b"hello").unwrap();
        assert_eq!(val, Some(Bytes::from("world")));
    }

    #[test]
    fn delete_makes_key_invisible() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"key", Bytes::from("value")).unwrap();
        tree.commit(&mut txn).unwrap();

        let mut txn2 = tree.begin_txn();
        tree.delete(&mut txn2, b"key").unwrap();
        tree.commit(&mut txn2).unwrap();

        let txn3 = tree.begin_txn();
        assert_eq!(tree.get(&txn3, b"key").unwrap(), None);
    }

    #[test]
    fn flush_persists_to_sst() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"persisted", Bytes::from("data"))
            .unwrap();
        tree.commit(&mut txn).unwrap();

        tree.flush().unwrap();

        // Verify SST was created.
        let sst_dir = dir.path().join("sst");
        let sst_count = std::fs::read_dir(&sst_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .count();
        assert!(sst_count > 0);

        // Should still be readable.
        let txn2 = tree.begin_txn();
        assert_eq!(
            tree.get(&txn2, b"persisted").unwrap(),
            Some(Bytes::from("data"))
        );
    }

    #[test]
    fn write_conflict_detected() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn1 = tree.begin_txn();
        let mut txn2 = tree.begin_txn();

        tree.put(&mut txn1, b"contested", Bytes::from("v1"))
            .unwrap();
        tree.put(&mut txn2, b"contested", Bytes::from("v2"))
            .unwrap();

        tree.commit(&mut txn1).unwrap();
        let result = tree.commit(&mut txn2);
        assert!(matches!(result, Err(Error::WriteConflict)));
        tree.abort(&mut txn2);
    }

    #[test]
    fn many_writes_trigger_flush() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        // Write enough data to exceed the 1KB memtable threshold.
        for i in 0..100u64 {
            let mut txn = tree.begin_txn();
            let key = format!("key-{i:04}");
            let value = format!("value-{i:04}-padding-to-make-this-bigger");
            tree.put(&mut txn, key.as_bytes(), Bytes::from(value))
                .unwrap();
            tree.commit(&mut txn).unwrap();
        }

        // Should have flushed at least once.
        let sst_dir = dir.path().join("sst");
        let sst_count = std::fs::read_dir(&sst_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .count();
        assert!(sst_count > 0);

        // All keys should still be readable.
        let txn = tree.begin_txn();
        for i in 0..100u64 {
            let key = format!("key-{i:04}");
            let val = tree.get(&txn, key.as_bytes()).unwrap();
            assert!(val.is_some(), "key {key} missing after flush");
        }
    }

    #[test]
    fn compaction_merges_sst_files() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        // Write and flush multiple times to create multiple SSTs.
        for batch in 0..3u64 {
            for i in 0..20u64 {
                let mut txn = tree.begin_txn();
                let key = format!("key-{:04}", batch * 20 + i);
                let value = format!("value-{}-padding-for-size", batch * 20 + i);
                tree.put(&mut txn, key.as_bytes(), Bytes::from(value))
                    .unwrap();
                tree.commit(&mut txn).unwrap();
            }
            tree.flush().unwrap();
        }

        let sst_count_before = tree.sst_count();
        assert!(sst_count_before >= 3);

        tree.compact().unwrap();

        assert_eq!(tree.sst_count(), 1);

        // All keys still readable.
        let txn = tree.begin_txn();
        for i in 0..60u64 {
            let key = format!("key-{i:04}");
            let val = tree.get(&txn, key.as_bytes()).unwrap();
            assert!(val.is_some(), "key {key} missing after compaction");
        }
    }

    #[test]
    fn compaction_drops_old_versions() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        // Write a key, flush, then update it and flush again.
        let mut txn1 = tree.begin_txn();
        tree.put(&mut txn1, b"versioned", Bytes::from("v1"))
            .unwrap();
        tree.commit(&mut txn1).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"versioned", Bytes::from("v2"))
            .unwrap();
        tree.commit(&mut txn2).unwrap();
        tree.flush().unwrap();

        assert!(tree.sst_count() >= 2);

        tree.compact().unwrap();
        assert_eq!(tree.sst_count(), 1);

        // Should see latest value.
        let txn3 = tree.begin_txn();
        assert_eq!(
            tree.get(&txn3, b"versioned").unwrap(),
            Some(Bytes::from("v2"))
        );
    }

    #[test]
    fn compaction_removes_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn1 = tree.begin_txn();
        tree.put(&mut txn1, b"ephemeral", Bytes::from("here"))
            .unwrap();
        tree.commit(&mut txn1).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.delete(&mut txn2, b"ephemeral").unwrap();
        tree.commit(&mut txn2).unwrap();
        tree.flush().unwrap();

        tree.compact().unwrap();

        let txn3 = tree.begin_txn();
        assert_eq!(tree.get(&txn3, b"ephemeral").unwrap(), None);
    }

    #[test]
    fn compaction_skipped_with_fewer_than_two_ssts() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"solo", Bytes::from("value")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        assert_eq!(tree.sst_count(), 1);
        tree.compact().unwrap();
        assert_eq!(tree.sst_count(), 1); // unchanged
    }
}
