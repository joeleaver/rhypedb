use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bytes::{BufMut, Bytes, BytesMut};

use crate::key::InternalKey;
use crate::memtable::MemValue;
use crate::{Error, Result};

/// Magic bytes at the start of every SST file.
const SST_MAGIC: &[u8; 4] = b"RSST";

/// SST file format version.
const SST_VERSION: u32 = 1;

/// On-disk footer at the end of the SST file.
///
/// ```text
/// [index_offset: 8 bytes]
/// [index_count:  4 bytes]
/// [magic:        4 bytes]
/// ```
const FOOTER_SIZE: usize = 8 + 4 + 4;

/// A single entry in the SST data block.
///
/// ```text
/// [key_len:   4 bytes]
/// [value_len: 4 bytes]  (u32::MAX = tombstone)
/// [key:       key_len bytes]
/// [value:     value_len bytes]
/// ```
const TOMBSTONE_MARKER: u32 = u32::MAX;

/// In-memory representation of an SST index entry.
///
/// On-disk format: `[key_len: 4][key: key_len][offset: 8]`
#[derive(Debug, Clone)]
struct IndexEntry {
    key: Bytes,
    offset: u64,
}

/// Writer for creating SST files from sorted key-value pairs.
pub struct SstWriter {
    writer: BufWriter<File>,
    index: Vec<IndexEntry>,
    current_offset: u64,
    entry_count: usize,
    path: PathBuf,
}

impl SstWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write file header.
        writer.write_all(SST_MAGIC)?;
        writer.write_all(&SST_VERSION.to_be_bytes())?;

        Ok(Self {
            writer,
            index: Vec::new(),
            current_offset: (SST_MAGIC.len() + 4) as u64, // past header
            entry_count: 0,
            path,
        })
    }

    /// Add a sorted entry. Keys MUST be added in ascending order.
    pub fn add(&mut self, internal_key: &[u8], value: &MemValue) -> Result<()> {
        let key_len = internal_key.len() as u32;
        let (value_len, value_bytes): (u32, &[u8]) = match value {
            Some(v) => (v.len() as u32, v.as_ref()),
            None => (TOMBSTONE_MARKER, &[]),
        };

        // Record index entry for every Nth entry (sparse index).
        if self.entry_count.is_multiple_of(16) {
            self.index.push(IndexEntry {
                key: Bytes::copy_from_slice(internal_key),
                offset: self.current_offset,
            });
        }

        let entry_size = 4 + 4 + internal_key.len() + value_bytes.len();
        let mut buf = BytesMut::with_capacity(entry_size);
        buf.put_u32(key_len);
        buf.put_u32(value_len);
        buf.put_slice(internal_key);
        buf.put_slice(value_bytes);

        self.writer.write_all(&buf)?;
        self.current_offset += entry_size as u64;
        self.entry_count += 1;

        Ok(())
    }

    /// Finalize the SST file by writing the index block and footer.
    pub fn finish(mut self) -> Result<SstMeta> {
        let index_offset = self.current_offset;

        // Write index entries.
        for entry in &self.index {
            let key_len = entry.key.len() as u32;
            let mut buf = BytesMut::with_capacity(4 + entry.key.len() + 8);
            buf.put_u32(key_len);
            buf.put_slice(&entry.key);
            buf.put_u64(entry.offset);
            self.writer.write_all(&buf)?;
        }

        // Write footer.
        let mut footer = BytesMut::with_capacity(FOOTER_SIZE);
        footer.put_u64(index_offset);
        footer.put_u32(self.index.len() as u32);
        footer.put_slice(SST_MAGIC);
        self.writer.write_all(&footer)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        let first_key = self.index.first().map(|e| e.key.clone());
        let last_key = self
            .index
            .last()
            .map(|e| e.key.clone());

        Ok(SstMeta {
            path: self.path,
            entry_count: self.entry_count,
            first_key,
            last_key,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Metadata about a completed SST file.
#[derive(Debug, Clone)]
pub struct SstMeta {
    pub path: PathBuf,
    pub entry_count: usize,
    pub first_key: Option<Bytes>,
    pub last_key: Option<Bytes>,
}

/// Reader for querying SST files.
pub struct SstReader {
    data: Vec<u8>,
    index: Vec<IndexEntry>,
    path: PathBuf,
}

impl SstReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        if data.len() < SST_MAGIC.len() + 4 + FOOTER_SIZE {
            return Err(Error::SstCorrupted("file too small".into()));
        }

        // Validate header.
        if &data[..4] != SST_MAGIC {
            return Err(Error::SstCorrupted("bad magic".into()));
        }
        let version = u32::from_be_bytes(data[4..8].try_into().unwrap());
        if version != SST_VERSION {
            return Err(Error::SstCorrupted(format!("unsupported version: {version}")));
        }

        // Read footer.
        let footer_start = data.len() - FOOTER_SIZE;
        if &data[footer_start + 12..] != SST_MAGIC {
            return Err(Error::SstCorrupted("bad footer magic".into()));
        }
        let index_offset =
            u64::from_be_bytes(data[footer_start..footer_start + 8].try_into().unwrap()) as usize;
        let index_count =
            u32::from_be_bytes(data[footer_start + 8..footer_start + 12].try_into().unwrap())
                as usize;

        // Parse index entries.
        let mut index = Vec::with_capacity(index_count);
        let mut pos = index_offset;
        for _ in 0..index_count {
            if pos + 4 > footer_start {
                return Err(Error::SstCorrupted("index truncated".into()));
            }
            let key_len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let key = Bytes::copy_from_slice(&data[pos..pos + key_len]);
            pos += key_len;
            let offset = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            index.push(IndexEntry { key, offset });
        }

        Ok(Self {
            data,
            index,
            path,
        })
    }

    /// Look up a specific internal key. Returns `Some(Some(value))` for a put,
    /// `Some(None)` for a tombstone, or `None` if the key isn't in this SST.
    pub fn get(&self, target_key: &[u8]) -> Result<Option<MemValue>> {
        // Binary search the sparse index to find the block to scan.
        let block_idx = match self.index.binary_search_by(|e| e.key.as_ref().cmp(target_key)) {
            Ok(i) => i,
            Err(0) => return Ok(None), // target is before all entries
            Err(i) => i - 1,
        };

        let start = self.index[block_idx].offset as usize;
        let end = if block_idx + 1 < self.index.len() {
            self.index[block_idx + 1].offset as usize
        } else {
            // Scan until the index block.
            let footer_start = self.data.len() - FOOTER_SIZE;
            
            u64::from_be_bytes(self.data[footer_start..footer_start + 8].try_into().unwrap())
                    as usize
        };

        // Linear scan through the data block.
        let mut pos = start;
        while pos < end {
            let key_len =
                u32::from_be_bytes(self.data[pos..pos + 4].try_into().unwrap()) as usize;
            let value_len_raw =
                u32::from_be_bytes(self.data[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;

            let key = &self.data[pos..pos + key_len];
            pos += key_len;

            let is_tombstone = value_len_raw == TOMBSTONE_MARKER;
            let value_len = if is_tombstone { 0 } else { value_len_raw as usize };
            let value_slice = &self.data[pos..pos + value_len];
            pos += value_len;

            if key == target_key {
                return if is_tombstone {
                    Ok(Some(None))
                } else {
                    Ok(Some(Some(Bytes::copy_from_slice(value_slice))))
                };
            }

            if key > target_key {
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Find the latest visible version of a user key at the given snapshot version.
    /// Scans through versioned entries to find the newest version <= snapshot.
    pub fn get_versioned(&self, user_key: &[u8], version: u64) -> Result<Option<MemValue>> {
        if self.index.is_empty() {
            return Ok(None);
        }

        // Use the user key with version 0 (largest inverted bytes) as the
        // search target. This is the LARGEST internal key for this user key,
        // so binary search finds the block at or before where entries end.
        let search_key = InternalKey::new(user_key, 0);
        let search_bytes = search_key.as_bytes();

        let block_idx = match self
            .index
            .binary_search_by(|e| e.key.as_ref().cmp(search_bytes))
        {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        };

        // Scan from the found block through all remaining data, since
        // entries for one user key may span multiple sparse index blocks.
        let start = self.index[block_idx].offset as usize;
        let footer_start = self.data.len() - FOOTER_SIZE;
        let data_end =
            u64::from_be_bytes(self.data[footer_start..footer_start + 8].try_into().unwrap())
                as usize;

        let mut pos = start;
        while pos < data_end {
            let key_len =
                u32::from_be_bytes(self.data[pos..pos + 4].try_into().unwrap()) as usize;
            let value_len_raw =
                u32::from_be_bytes(self.data[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;

            let key = &self.data[pos..pos + key_len];
            pos += key_len;

            let is_tombstone = value_len_raw == TOMBSTONE_MARKER;
            let value_len = if is_tombstone { 0 } else { value_len_raw as usize };
            let value_slice = &self.data[pos..pos + value_len];
            pos += value_len;

            if key.len() < 8 {
                continue;
            }
            let entry_user_key = &key[..key.len() - 8];

            if entry_user_key != user_key {
                if entry_user_key > user_key {
                    return Ok(None);
                }
                continue;
            }

            // Same user key — decode the version.
            let ver_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let entry_version = !u64::from_be_bytes(ver_bytes);

            if entry_version <= version {
                return if is_tombstone {
                    Ok(Some(None))
                } else {
                    Ok(Some(Some(Bytes::copy_from_slice(value_slice))))
                };
            }
        }

        Ok(None)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Find the maximum version stored in this SST file.
    /// Used during recovery to restore the transaction version counter.
    pub fn max_version(&self) -> u64 {
        let mut max_ver = 0u64;
        for (key, _) in self.iter() {
            if key.len() >= 8 {
                let ver_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
                let version = !u64::from_be_bytes(ver_bytes);
                max_ver = max_ver.max(version);
            }
        }
        max_ver
    }

    /// Scan for entries whose user key starts with `prefix`, returning the latest
    /// visible version per user key at the given snapshot version.
    pub fn scan_prefix(&self, prefix: &[u8], version: u64) -> Vec<(Bytes, Option<Bytes>)> {
        let mut results = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;

        for (key, value) in self.iter() {
            if key.len() < 8 {
                continue;
            }
            let user_key = &key[..key.len() - 8];

            // Skip entries before the prefix range.
            if user_key < prefix {
                continue;
            }

            // Stop past the prefix range.
            if !user_key.starts_with(prefix) {
                break;
            }

            let ver_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let entry_version = !u64::from_be_bytes(ver_bytes);

            if entry_version > version {
                continue;
            }

            let same_key = last_user_key
                .as_ref()
                .is_some_and(|prev| prev.as_slice() == user_key);
            if same_key {
                continue;
            }

            last_user_key = Some(user_key.to_vec());
            results.push((Bytes::copy_from_slice(user_key), value));
        }

        results
    }

    /// Iterate all entries in the SST in sorted order.
    pub fn iter(&self) -> SstIterator<'_> {
        let header_size = SST_MAGIC.len() + 4;
        let footer_start = self.data.len() - FOOTER_SIZE;
        let index_offset =
            u64::from_be_bytes(self.data[footer_start..footer_start + 8].try_into().unwrap())
                as usize;

        SstIterator {
            data: &self.data,
            pos: header_size,
            end: index_offset,
        }
    }
}

/// Iterator over SST entries.
pub struct SstIterator<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Iterator for SstIterator<'a> {
    type Item = (Bytes, MemValue);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }

        let key_len =
            u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap()) as usize;
        let value_len_raw =
            u32::from_be_bytes(self.data[self.pos + 4..self.pos + 8].try_into().unwrap());
        self.pos += 8;

        let key = Bytes::copy_from_slice(&self.data[self.pos..self.pos + key_len]);
        self.pos += key_len;

        let is_tombstone = value_len_raw == TOMBSTONE_MARKER;
        let value_len = if is_tombstone { 0 } else { value_len_raw as usize };

        let value = if is_tombstone {
            None
        } else {
            Some(Bytes::copy_from_slice(
                &self.data[self.pos..self.pos + value_len],
            ))
        };
        self.pos += value_len;

        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::KeyBuilder;

    #[test]
    fn write_and_read_sst() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("test.sst");

        // Write entries.
        let mut writer = SstWriter::new(&sst_path).unwrap();
        for i in 0u64..100 {
            let key = InternalKey::new(&KeyBuilder::object(1, i), 1);
            let value = Bytes::from(format!("value-{i}"));
            writer.add(key.as_bytes(), &Some(value)).unwrap();
        }
        let meta = writer.finish().unwrap();
        assert_eq!(meta.entry_count, 100);

        // Read back.
        let reader = SstReader::open(&sst_path).unwrap();
        let key = InternalKey::new(&KeyBuilder::object(1, 42), 1);
        let result = reader.get(key.as_bytes()).unwrap();
        assert_eq!(result, Some(Some(Bytes::from("value-42"))));
    }

    #[test]
    fn tombstones_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("tomb.sst");

        let mut writer = SstWriter::new(&sst_path).unwrap();
        let key = InternalKey::new(b"deleted", 1);
        writer.add(key.as_bytes(), &None).unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let result = reader.get(key.as_bytes()).unwrap();
        assert_eq!(result, Some(None)); // tombstone
    }

    #[test]
    fn missing_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("sparse.sst");

        let mut writer = SstWriter::new(&sst_path).unwrap();
        let key = InternalKey::new(b"exists", 1);
        writer.add(key.as_bytes(), &Some(Bytes::from("yes"))).unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let missing = InternalKey::new(b"missing", 1);
        assert_eq!(reader.get(missing.as_bytes()).unwrap(), None);
    }

    #[test]
    fn versioned_get() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("ver.sst");

        let user_key = b"mykey";
        let mut writer = SstWriter::new(&sst_path).unwrap();

        // Add versions in sorted order (version 10 sorts before version 5 due to inversion).
        let k10 = InternalKey::new(user_key, 10);
        let k5 = InternalKey::new(user_key, 5);
        let k1 = InternalKey::new(user_key, 1);

        writer
            .add(k10.as_bytes(), &Some(Bytes::from("v10")))
            .unwrap();
        writer
            .add(k5.as_bytes(), &Some(Bytes::from("v5")))
            .unwrap();
        writer
            .add(k1.as_bytes(), &Some(Bytes::from("v1")))
            .unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();

        assert_eq!(
            reader.get_versioned(user_key, 10).unwrap(),
            Some(Some(Bytes::from("v10")))
        );
        assert_eq!(
            reader.get_versioned(user_key, 7).unwrap(),
            Some(Some(Bytes::from("v5")))
        );
        assert_eq!(
            reader.get_versioned(user_key, 1).unwrap(),
            Some(Some(Bytes::from("v1")))
        );
        assert_eq!(reader.get_versioned(user_key, 0).unwrap(), None);
    }

    #[test]
    fn iterate_all_entries() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("iter.sst");

        let mut writer = SstWriter::new(&sst_path).unwrap();
        for i in 0u64..50 {
            let key = InternalKey::new(&KeyBuilder::object(1, i), 1);
            writer
                .add(key.as_bytes(), &Some(Bytes::from(format!("v{i}"))))
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let entries: Vec<_> = reader.iter().collect();
        assert_eq!(entries.len(), 50);
    }
}
