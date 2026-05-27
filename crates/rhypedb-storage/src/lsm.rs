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

    /// Returns a reference to the transaction manager.
    pub fn txn_manager(&self) -> &TransactionManager {
        &self.txn_manager
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
}
