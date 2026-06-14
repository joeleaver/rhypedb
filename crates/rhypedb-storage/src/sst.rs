use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bytes::{BufMut, Bytes, BytesMut};

use crate::bloom::{BloomFilter, HashAlgo};
use crate::key::InternalKey;
use crate::memtable::MemValue;
use crate::zone::{ZoneBuilder, ZoneFieldExtractor, ZoneMap};
use crate::{Error, Result};

/// Magic bytes at the start of every SST file.
const SST_MAGIC: &[u8; 4] = b"RSST";

/// SST file format version.
/// - v1: data + sparse index + footer.
/// - v2: + bloom filter block (FNV-1a, fixed format).
/// - v3: + algorithm byte at the start of the bloom block (so xxh3 and
///   future hash variants can be selected per file).
/// - v4: + zone map block before the bloom block, footer extended with
///   `[zonemap_offset: u64][zonemap_size: u32]`. Per-block min/max for
///   opted-in integer fields enables `scan_prefix_filtered` to skip whole
///   blocks before any entry decode. Zone columns keyed by FNV-1a of
///   field name — vulnerable to `rename_field`.
/// - v5: zone map keyed by stable catalog `field_id` instead of FNV(name).
///   Wire format identical to v4 (still `u32` per column key); only the
///   *meaning* of the bits changes, so the version bump is the safety
///   marker. v4 files keep opening for read but their zone map is
///   discarded (block pruning falls back to full scan) — natural
///   compaction rewrites them as v5.
const SST_VERSION: u32 = 5;

/// v1 footer: [index_offset: 8][index_count: 4][magic: 4]
const FOOTER_V1_SIZE: usize = 8 + 4 + 4;

/// v2/v3 footer: [bloom_offset: 8][bloom_size: 4][index_offset: 8][index_count: 4][magic: 4].
/// The layout is identical between v2 and v3; only the bloom block's internal
/// format changes (v3 prepends a 1-byte hash-algorithm tag).
const FOOTER_V2_SIZE: usize = 8 + 4 + 8 + 4 + 4;

/// v4/v5 footer: [zonemap_offset: 8][zonemap_size: 4] + v2/v3 footer.
const FOOTER_V4_SIZE: usize = 8 + 4 + FOOTER_V2_SIZE;

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
    /// Collected user keys for the bloom filter. The bloom filter is
    /// constructed in `finish()` once we know the entry count.
    user_keys: Vec<Bytes>,
    /// Optional callback that pulls zone-mapped (field-hash, encoded-bytes)
    /// pairs out of each entry's value bytes. `None` disables the zone-map
    /// block — useful for SSTs that carry only edges or other non-object
    /// data. The engine sets this at `LsmTree` construction.
    zone_extractor: Option<ZoneFieldExtractor>,
    /// Accumulator for per-block min/max bounds. Mirrors the sparse-index
    /// block layout: a new block starts every 16 entries.
    zone_builder: ZoneBuilder,
}

/// Sparse-index block size (entries per block). Used both to decide when to
/// emit a sparse-index entry and to compute the block_idx for zone maps.
const ENTRIES_PER_BLOCK: usize = 16;

impl SstWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_zone_extractor(path, None)
    }

    pub fn new_with_zone_extractor(
        path: impl AsRef<Path>,
        zone_extractor: Option<ZoneFieldExtractor>,
    ) -> Result<Self> {
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
            user_keys: Vec::new(),
            zone_extractor,
            zone_builder: ZoneBuilder::new(),
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
        if self.entry_count.is_multiple_of(ENTRIES_PER_BLOCK) {
            self.index.push(IndexEntry {
                key: Bytes::copy_from_slice(internal_key),
                offset: self.current_offset,
            });
        }

        // Update zone bounds for opted-in integer fields. Skipped for tombstones
        // (no value to extract from) and when no extractor is configured.
        if let (Some(extractor), Some(_)) = (&self.zone_extractor, value) {
            let zone_fields = extractor(internal_key, value_bytes);
            if !zone_fields.is_empty() {
                let block_idx = self.entry_count / ENTRIES_PER_BLOCK;
                self.zone_builder.record(block_idx, &zone_fields);
            }
        }

        // Collect user key for the bloom filter. Same user key at different
        // versions only needs to be added once, so de-dup against the last
        // one we saw (keys arrive in sorted order, so duplicates are adjacent).
        if internal_key.len() >= 8 {
            let user_key = &internal_key[..internal_key.len() - 8];
            let is_new = self
                .user_keys
                .last()
                .is_none_or(|prev| prev.as_ref() != user_key);
            if is_new {
                self.user_keys.push(Bytes::copy_from_slice(user_key));
            }
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

    /// Finalize the SST file by writing the index, zone map, bloom filter, and footer.
    pub fn finish(mut self) -> Result<SstMeta> {
        let index_offset = self.current_offset;

        // Write index entries.
        let mut bytes_written = 0u64;
        for entry in &self.index {
            let key_len = entry.key.len() as u32;
            let mut buf = BytesMut::with_capacity(4 + entry.key.len() + 8);
            buf.put_u32(key_len);
            buf.put_slice(&entry.key);
            buf.put_u64(entry.offset);
            self.writer.write_all(&buf)?;
            bytes_written += buf.len() as u64;
        }

        // Zone map block (v4): per-block min/max for opted-in fields. When
        // no extractor is configured, this writes just the empty header
        // (num_blocks=0, num_fields=0) — still a valid zero-overhead v4 file.
        let zonemap_offset = index_offset + bytes_written;
        let mut zone_buf = Vec::new();
        self.zone_builder.write_to(&mut zone_buf)?;
        let zonemap_size = zone_buf.len() as u32;
        self.writer.write_all(&zone_buf)?;

        // Build the bloom filter from collected user keys.
        let mut bloom = BloomFilter::new(self.user_keys.len());
        for k in &self.user_keys {
            bloom.add(k);
        }

        // Bloom block (v3 layout: [algo_byte][bloom.write_to payload]).
        let bloom_offset = zonemap_offset + zonemap_size as u64;
        let mut bloom_buf = Vec::with_capacity(6 + bloom.byte_size());
        bloom_buf.push(bloom.algo().to_byte());
        bloom.write_to(&mut bloom_buf)?;
        let bloom_size = bloom_buf.len() as u32;
        self.writer.write_all(&bloom_buf)?;

        // v4 footer.
        let mut footer = BytesMut::with_capacity(FOOTER_V4_SIZE);
        footer.put_u64(zonemap_offset);
        footer.put_u32(zonemap_size);
        footer.put_u64(bloom_offset);
        footer.put_u32(bloom_size);
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
    /// File contents, held as a `Bytes` so the hot-path key/value slices we
    /// return to callers can be zero-copy ref-counted views (`data.slice(..)`)
    /// instead of allocating + memcpy via `Bytes::copy_from_slice`.
    data: Bytes,
    index: Vec<IndexEntry>,
    /// Bloom filter for user keys. v1 files don't have one — `None` means
    /// "no filter, assume might contain anything".
    bloom: Option<BloomFilter>,
    /// End of the data block (where the sparse index begins).
    data_end: usize,
    /// First user key in the file (no version suffix). `None` for empty SST.
    /// Used to skip the whole SST in a batched lookup when every needle is
    /// below this — avoids ~700 bloom calls per query that would all reject.
    first_user_key: Option<Bytes>,
    /// Last user key in the file (no version suffix). `None` for empty SST.
    /// Computed by walking the final data block at `open()` so it's exact —
    /// the sparse index only records each block's *first* key, not its last.
    last_user_key: Option<Bytes>,
    /// Per-block min/max bounds for opted-in integer fields. Empty for v1-v3
    /// files. v4 files always carry one (possibly with `num_fields=0` if no
    /// extractor was configured at write time).
    zone_map: Option<ZoneMap>,
    path: PathBuf,
}

impl SstReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // mmap the file contents instead of read-to-end. The kernel page cache
        // owns the actual memory; cold pages can be evicted under pressure.
        // Wrapping in `Bytes::from_owner` lets the hot-path `data.slice(..)`
        // calls keep returning zero-copy refcounted views — when the last
        // `Bytes` ref drops, the `Mmap` (and its mapping) is released.
        //
        // SAFETY: SST files are written-once via `SstWriter::finish()` to a
        // fresh path before the SstReader opens them; no path ever has both
        // an active writer and reader, and we never truncate / modify a file
        // that's been opened for reading. Mapping a written-once file is the
        // canonical safe use of `Mmap::map`.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // Hint random access — bloom-then-seek scans don't benefit from the
        // kernel's read-ahead heuristic, and on big SSTs the read-ahead can
        // waste page cache and disk bandwidth.
        let _ = mmap.advise(memmap2::Advice::Random);
        let data: Bytes = Bytes::from_owner(mmap);

        if data.len() < SST_MAGIC.len() + 4 + FOOTER_V1_SIZE {
            return Err(Error::SstCorrupted("file too small".into()));
        }

        // Validate header.
        if &data[..4] != SST_MAGIC {
            return Err(Error::SstCorrupted("bad magic".into()));
        }
        let version = u32::from_be_bytes(data[4..8].try_into().unwrap());
        if !(1..=5).contains(&version) {
            return Err(Error::SstCorrupted(format!("unsupported version: {version}")));
        }

        let (index_offset, index_count, bloom, zone_map) = if version == 4 || version == 5 {
            if data.len() < SST_MAGIC.len() + 4 + FOOTER_V4_SIZE {
                return Err(Error::SstCorrupted("v4/v5 footer truncated".into()));
            }
            let footer_start = data.len() - FOOTER_V4_SIZE;
            // Magic check: footer_start + 36 (zonemap u64+u32 + bloom u64+u32 + index u64+u32 = 36).
            if &data[footer_start + 36..] != SST_MAGIC {
                return Err(Error::SstCorrupted("bad v4/v5 footer magic".into()));
            }
            let zonemap_offset = u64::from_be_bytes(
                data[footer_start..footer_start + 8].try_into().unwrap(),
            ) as usize;
            let zonemap_size = u32::from_be_bytes(
                data[footer_start + 8..footer_start + 12].try_into().unwrap(),
            ) as usize;
            let bloom_offset = u64::from_be_bytes(
                data[footer_start + 12..footer_start + 20].try_into().unwrap(),
            ) as usize;
            let bloom_size = u32::from_be_bytes(
                data[footer_start + 20..footer_start + 24].try_into().unwrap(),
            ) as usize;
            let index_offset = u64::from_be_bytes(
                data[footer_start + 24..footer_start + 32].try_into().unwrap(),
            ) as usize;
            let index_count = u32::from_be_bytes(
                data[footer_start + 32..footer_start + 36].try_into().unwrap(),
            ) as usize;

            if zonemap_offset + zonemap_size > footer_start {
                return Err(Error::SstCorrupted("zonemap extends past footer".into()));
            }
            // v5 columns are keyed by stable field_id; v4 columns are keyed
            // by FNV-1a(field_name) and would be miskeyed under the new
            // convention (zero pruning hits, or worse, false matches if a
            // hash happens to collide with a field_id). The zone map block
            // is still parsed/skipped over for layout consistency, but only
            // wired into the reader for v5. v4 SSTs serve correct results
            // without block pruning until natural compaction rewrites them.
            let zone_map = if version == 5 {
                let zone_slice = &data[zonemap_offset..zonemap_offset + zonemap_size];
                Some(ZoneMap::read_from(&mut &zone_slice[..])?)
            } else {
                None
            };

            if bloom_offset + bloom_size > footer_start {
                return Err(Error::SstCorrupted("bloom block extends past footer".into()));
            }
            let bloom_slice = &data[bloom_offset..bloom_offset + bloom_size];
            if bloom_slice.is_empty() {
                return Err(Error::SstCorrupted("v4/v5 bloom missing algo byte".into()));
            }
            let algo = HashAlgo::from_byte(bloom_slice[0]).ok_or_else(|| {
                Error::SstCorrupted(format!("unknown bloom algo byte: {}", bloom_slice[0]))
            })?;
            let bloom = BloomFilter::read_from(&mut &bloom_slice[1..], algo)
                .map_err(|e| Error::SstCorrupted(format!("bloom: {e}")))?;

            (index_offset, index_count, Some(bloom), zone_map)
        } else if version == 2 || version == 3 {
            if data.len() < SST_MAGIC.len() + 4 + FOOTER_V2_SIZE {
                return Err(Error::SstCorrupted("v2/v3 footer truncated".into()));
            }
            let footer_start = data.len() - FOOTER_V2_SIZE;
            if &data[footer_start + 24..] != SST_MAGIC {
                return Err(Error::SstCorrupted("bad footer magic".into()));
            }
            let bloom_offset = u64::from_be_bytes(
                data[footer_start..footer_start + 8].try_into().unwrap(),
            ) as usize;
            let bloom_size = u32::from_be_bytes(
                data[footer_start + 8..footer_start + 12].try_into().unwrap(),
            ) as usize;
            let index_offset = u64::from_be_bytes(
                data[footer_start + 12..footer_start + 20].try_into().unwrap(),
            ) as usize;
            let index_count = u32::from_be_bytes(
                data[footer_start + 20..footer_start + 24].try_into().unwrap(),
            ) as usize;

            // Parse bloom filter. v2: no algo byte, assume FNV-1a. v3: read
            // the leading algo byte, then parse the bloom payload.
            if bloom_offset + bloom_size > footer_start {
                return Err(Error::SstCorrupted("bloom block extends past footer".into()));
            }
            let bloom_slice = &data[bloom_offset..bloom_offset + bloom_size];
            let bloom = if version == 3 {
                if bloom_slice.is_empty() {
                    return Err(Error::SstCorrupted("v3 bloom missing algo byte".into()));
                }
                let algo = HashAlgo::from_byte(bloom_slice[0]).ok_or_else(|| {
                    Error::SstCorrupted(format!("unknown bloom algo byte: {}", bloom_slice[0]))
                })?;
                BloomFilter::read_from(&mut &bloom_slice[1..], algo)
                    .map_err(|e| Error::SstCorrupted(format!("bloom: {e}")))?
            } else {
                BloomFilter::read_from(&mut &bloom_slice[..], HashAlgo::Fnv1a)
                    .map_err(|e| Error::SstCorrupted(format!("bloom: {e}")))?
            };

            (index_offset, index_count, Some(bloom), None)
        } else {
            // version == 1
            let footer_start = data.len() - FOOTER_V1_SIZE;
            if &data[footer_start + 12..] != SST_MAGIC {
                return Err(Error::SstCorrupted("bad footer magic".into()));
            }
            let index_offset = u64::from_be_bytes(
                data[footer_start..footer_start + 8].try_into().unwrap(),
            ) as usize;
            let index_count = u32::from_be_bytes(
                data[footer_start + 8..footer_start + 12].try_into().unwrap(),
            ) as usize;
            (index_offset, index_count, None, None)
        };

        // Parse index entries.
        let mut index = Vec::with_capacity(index_count);
        let mut pos = index_offset;
        for _ in 0..index_count {
            if pos + 4 > data.len() {
                return Err(Error::SstCorrupted("index truncated".into()));
            }
            let key_len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let key = data.slice(pos..pos + key_len);
            pos += key_len;
            let offset = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            index.push(IndexEntry { key, offset });
        }

        let data_end = index_offset;

        // First user key: strip the 8-byte version suffix off the first
        // sparse-index entry (which is the first entry's full internal key).
        let first_user_key = index.first().and_then(|e| {
            if e.key.len() >= 8 {
                Some(e.key.slice(..e.key.len() - 8))
            } else {
                None
            }
        });

        // Last user key: walk the FINAL data block (sparse index only knows
        // each block's first key, not its last). One-time cost at open();
        // gives an exact upper bound for the per-batch range-pruning check.
        let last_user_key = if let Some(last_index) = index.last() {
            let block_start = last_index.offset as usize;
            let mut pos = block_start;
            let mut last_uk: Option<Bytes> = None;
            while pos < data_end {
                if pos + 8 > data_end {
                    break;
                }
                let key_len =
                    u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                let value_len_raw =
                    u32::from_be_bytes(data[pos + 4..pos + 8].try_into().unwrap());
                let key_start = pos + 8;
                let key_end = key_start + key_len;
                if key_end > data_end {
                    break;
                }
                if key_len >= 8 {
                    last_uk = Some(data.slice(key_start..key_end - 8));
                }
                let val_len = if value_len_raw == TOMBSTONE_MARKER {
                    0
                } else {
                    value_len_raw as usize
                };
                pos = key_end + val_len;
            }
            last_uk
        } else {
            None
        };

        Ok(Self {
            data,
            index,
            bloom,
            data_end,
            first_user_key,
            last_user_key,
            zone_map,
            path,
        })
    }

    /// Borrow the optional zone map (v4 SSTs only).
    pub fn zone_map(&self) -> Option<&ZoneMap> {
        self.zone_map.as_ref()
    }

    /// Quick batched bounds check: returns `false` when no key in
    /// `sorted_user_keys` could possibly be in this SST (entirely below the
    /// first or entirely above the last). Lets `LsmTree::multi_get_at` skip
    /// the whole SST — and therefore N bloom calls + the cursor walk —
    /// without per-key work.
    ///
    /// `sorted_user_keys` MUST be sorted ascending. Returns `true` for empty
    /// SSTs only when the input is non-empty (treated as "unknown" — caller
    /// will fall through to the existing path, which handles empty SST).
    pub fn range_overlaps_sorted(&self, sorted_user_keys: &[&[u8]]) -> bool {
        if sorted_user_keys.is_empty() {
            return false;
        }
        let (first, last) = match (&self.first_user_key, &self.last_user_key) {
            (Some(f), Some(l)) => (f.as_ref(), l.as_ref()),
            _ => return false, // empty SST — nothing to overlap with
        };
        // Smallest needle > our last entry → all needles are above our range.
        if sorted_user_keys[0] > last {
            return false;
        }
        // Largest needle < our first entry → all needles are below our range.
        if *sorted_user_keys.last().unwrap() < first {
            return false;
        }
        true
    }

    /// Prefix-batched bounds check: skip the SST when no entry's user key
    /// could possibly start with any of `sorted_prefixes`. Two safe-skip
    /// conditions:
    ///
    /// - **Smallest prefix sorts strictly above `last_user_key`**: every
    ///   prefix is bigger than every entry; no entry can start with one.
    /// - **Largest prefix is `< first_user_key` AND `first_user_key` does NOT
    ///   start with that prefix**: even the largest prefix can't reach any
    ///   entry — any key starting with `largest` would be `< first_user_key`
    ///   (since their differing byte is smaller), so it can't be in the SST.
    ///
    /// Other cases keep the SST in play (conservative — never skip an SST
    /// that could have a match).
    pub fn prefix_range_overlaps_sorted(&self, sorted_prefixes: &[&[u8]]) -> bool {
        if sorted_prefixes.is_empty() {
            return false;
        }
        let (first, last) = match (&self.first_user_key, &self.last_user_key) {
            (Some(f), Some(l)) => (f.as_ref(), l.as_ref()),
            _ => return false,
        };
        if sorted_prefixes[0] > last {
            return false;
        }
        let largest = *sorted_prefixes.last().unwrap();
        if largest < first && !first.starts_with(largest) {
            return false;
        }
        true
    }

    /// Check whether the SST might contain the given user key.
    /// Returns `false` for "definitely not present", `true` for "might be present".
    /// v1 files without a bloom filter always return `true`.
    pub fn might_contain(&self, user_key: &[u8]) -> bool {
        match &self.bloom {
            Some(filter) => filter.contains(user_key),
            None => true,
        }
    }

    /// Look up a specific internal key. Returns `Some(Some(value))` for a put,
    /// `Some(None)` for a tombstone, or `None` if the key isn't in this SST.
    pub fn get(&self, target_key: &[u8]) -> Result<Option<MemValue>> {
        // Bloom filter check: if the bloom filter says the user key is
        // definitely not in this SST, skip the read entirely.
        if target_key.len() >= 8 {
            let user_key = &target_key[..target_key.len() - 8];
            if !self.might_contain(user_key) {
                return Ok(None);
            }
        }

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
            self.data_end
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
            let value_start = pos;
            pos += value_len;

            if key == target_key {
                return if is_tombstone {
                    Ok(Some(None))
                } else {
                    Ok(Some(Some(self.data.slice(value_start..value_start + value_len))))
                };
            }

            if key > target_key {
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Batch point lookup: walk the SST ONCE for a pre-sorted list of user keys
    /// and return their latest visible versions at the given snapshot. Saves the
    /// per-key sparse-index binary search when needles fall in the same block,
    /// and uses a threshold gallop (re-binary-search at block boundaries) to
    /// skip whole blocks when needles are scattered.
    ///
    /// `sorted_user_keys` MUST be sorted ascending by byte order — for object
    /// keys (`o:<type_be>:<id_be>`) within one type, sorted-by-id is equivalent.
    /// Returns one entry per input in the same order:
    /// - `None` — needle is not in this SST (look in older layers).
    /// - `Some(None)` — tombstone is the latest visible version.
    /// - `Some(Some(bytes))` — put is the latest visible version.
    pub fn multi_get_versioned(
        &self,
        sorted_user_keys: &[&[u8]],
        version: u64,
    ) -> Result<Vec<Option<MemValue>>> {
        let n = sorted_user_keys.len();
        let mut out: Vec<Option<MemValue>> = vec![None; n];
        if n == 0 || self.index.is_empty() {
            return Ok(out);
        }

        let data_end = self.data_end;
        let mut i = 0usize;
        let mut pos = 0usize;
        let mut block_idx = 0usize;
        let mut seeded = false;

        while i < n {
            let needle = sorted_user_keys[i];

            // Bloom-skip needles that definitely aren't in this SST.
            if !self.might_contain(needle) {
                i += 1;
                continue;
            }

            // Seed the cursor to the first live needle's block.
            if !seeded {
                block_idx = self.locate_block_from(needle, 0);
                pos = self.index[block_idx].offset as usize;
                seeded = true;
            }

            if pos >= data_end {
                // Past end-of-data: every remaining needle is a miss in this SST.
                i += 1;
                continue;
            }

            // Decode entry at pos.
            let key_len =
                u32::from_be_bytes(self.data[pos..pos + 4].try_into().unwrap()) as usize;
            let value_len_raw =
                u32::from_be_bytes(self.data[pos + 4..pos + 8].try_into().unwrap());
            let key_start = pos + 8;
            let key_end = key_start + key_len;
            let is_tomb = value_len_raw == TOMBSTONE_MARKER;
            let val_len = if is_tomb { 0 } else { value_len_raw as usize };
            let val_start = key_end;
            let val_end = val_start + val_len;

            if key_len < 8 {
                pos = val_end;
                continue;
            }

            let entry_user = &self.data[key_start..key_end - 8];

            match entry_user.cmp(needle) {
                std::cmp::Ordering::Less => {
                    pos = val_end;
                    // Threshold gallop: if pos crossed into the next sparse block,
                    // re-binary-search the index to skip over blocks whose first key
                    // is still less than needle.
                    if block_idx + 1 < self.index.len()
                        && pos >= self.index[block_idx + 1].offset as usize
                    {
                        block_idx = self.locate_block_from(needle, block_idx + 1);
                        pos = self.index[block_idx].offset as usize;
                    }
                }
                std::cmp::Ordering::Equal => {
                    let entry_ver = !u64::from_be_bytes(
                        self.data[key_end - 8..key_end].try_into().unwrap(),
                    );
                    if entry_ver <= version {
                        out[i] = Some(if is_tomb {
                            None
                        } else {
                            Some(self.data.slice(val_start..val_end))
                        });
                        // Skip past any older versions of this same user key so the
                        // cursor lands on the next user key for the next needle.
                        pos = val_end;
                        while pos < data_end {
                            let kl2 = u32::from_be_bytes(
                                self.data[pos..pos + 4].try_into().unwrap(),
                            ) as usize;
                            if kl2 < 8 {
                                break;
                            }
                            let vlr2 = u32::from_be_bytes(
                                self.data[pos + 4..pos + 8].try_into().unwrap(),
                            );
                            let next_user = &self.data[pos + 8..pos + 8 + kl2 - 8];
                            if next_user != needle {
                                break;
                            }
                            let vl2 = if vlr2 == TOMBSTONE_MARKER {
                                0
                            } else {
                                vlr2 as usize
                            };
                            pos += 8 + kl2 + vl2;
                        }
                        // Keep block_idx in sync with pos.
                        while block_idx + 1 < self.index.len()
                            && pos >= self.index[block_idx + 1].offset as usize
                        {
                            block_idx += 1;
                        }
                        i += 1;
                    } else {
                        // Newest version visible here is after the snapshot — walk
                        // forward through same-user-key versions looking for an
                        // older visible one (or a different user_key, which exits
                        // via the Greater case on the next iteration).
                        pos = val_end;
                        while block_idx + 1 < self.index.len()
                            && pos >= self.index[block_idx + 1].offset as usize
                        {
                            block_idx += 1;
                        }
                    }
                }
                std::cmp::Ordering::Greater => {
                    // Needle is absent from this SST (entries already past it).
                    // Leave pos where it is — the next needle may match this entry.
                    i += 1;
                }
            }
        }

        Ok(out)
    }

    /// Find the sparse-index block at or before `user_key`'s FIRST
    /// (highest-version) entry, restricted to the suffix `self.index[start..]`.
    /// Returns an absolute index into `self.index`.
    ///
    /// Seeds with the SMALLEST internal key for `user_key` (version=u64::MAX,
    /// inverted bytes = 0x00). Seeding with the largest (version=0) is WRONG:
    /// when a user key's versions straddle a sparse-index block boundary it
    /// lands on the block holding only the OLDER version, and the forward scan
    /// starts there — returning a stale read. Same bug class as `get_versioned`
    /// (fixed); this is the `multi_get` / batched-lookup seek.
    fn locate_block_from(&self, user_key: &[u8], start: usize) -> usize {
        let mut search_key = Vec::with_capacity(user_key.len() + 8);
        search_key.extend_from_slice(user_key);
        search_key.extend_from_slice(&[0u8; 8]);

        let slice = &self.index[start..];
        let rel = match slice.binary_search_by(|e| e.key.as_ref().cmp(&search_key)) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        start + rel
    }

    /// Sparse-index sibling of `locate_block_from` for prefix scans: returns
    /// the latest block whose first key is <= the smallest internal key whose
    /// user key starts with `prefix` (version=u64::MAX, inverted bytes = 0).
    fn prefix_start_block_from(&self, prefix: &[u8], start: usize) -> usize {
        let mut search_key = Vec::with_capacity(prefix.len() + 8);
        search_key.extend_from_slice(prefix);
        search_key.extend_from_slice(&[0u8; 8]);

        let slice = &self.index[start..];
        let rel = match slice.binary_search_by(|e| e.key.as_ref().cmp(&search_key)) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        start + rel
    }

    /// Batch prefix scan: walk the SST ONCE for a pre-sorted list of disjoint
    /// prefixes and return matches per prefix at the given snapshot. Same
    /// idea as `multi_get_versioned` (cursor + threshold gallop) but emitting
    /// many entries per prefix.
    ///
    /// `sorted_prefixes` MUST be sorted ascending by byte order AND must
    /// represent disjoint ranges (no prefix may be a true prefix of another).
    /// For edge prefixes (`e:<src_be>:<rel_be>:`) this is trivially true.
    ///
    /// Per-prefix semantics match `scan_prefix`: deduped to one entry per
    /// user key (the newest visible version), tombstones preserved as
    /// `Some(None)` in the returned vector so layer merging can suppress
    /// older puts.
    pub fn multi_scan_prefix_versioned(
        &self,
        sorted_prefixes: &[&[u8]],
        version: u64,
    ) -> Vec<Vec<(Bytes, MemValue)>> {
        let n = sorted_prefixes.len();
        let mut out: Vec<Vec<(Bytes, MemValue)>> = (0..n).map(|_| Vec::new()).collect();
        if n == 0 || self.index.is_empty() {
            return out;
        }

        let data_end = self.data_end;
        let mut i = 0usize;
        let mut pos = 0usize;
        let mut block_idx = 0usize;
        let mut seeded = false;
        let mut last_user_key: Option<Vec<u8>> = None;

        while i < n {
            let prefix = sorted_prefixes[i];

            // Seed cursor at the first prefix's starting block.
            if !seeded {
                block_idx = self.prefix_start_block_from(prefix, 0);
                pos = self.index[block_idx].offset as usize;
                seeded = true;
                last_user_key = None;
            }

            if pos >= data_end {
                // Past end-of-data: remaining prefixes contribute nothing.
                i += 1;
                last_user_key = None;
                continue;
            }

            // Decode entry at pos.
            let key_len =
                u32::from_be_bytes(self.data[pos..pos + 4].try_into().unwrap()) as usize;
            let value_len_raw =
                u32::from_be_bytes(self.data[pos + 4..pos + 8].try_into().unwrap());
            let key_start = pos + 8;
            let key_end = key_start + key_len;
            let is_tomb = value_len_raw == TOMBSTONE_MARKER;
            let val_len = if is_tomb { 0 } else { value_len_raw as usize };
            let val_start = key_end;
            let val_end = val_start + val_len;

            if key_len < 8 {
                pos = val_end;
                continue;
            }

            let entry_user = &self.data[key_start..key_end - 8];

            if entry_user < prefix {
                // Before the active prefix's range — advance entry cursor.
                pos = val_end;
                // Threshold gallop: re-seek for the active prefix on block crossing.
                if block_idx + 1 < self.index.len()
                    && pos >= self.index[block_idx + 1].offset as usize
                {
                    block_idx = self.prefix_start_block_from(prefix, block_idx + 1);
                    pos = self.index[block_idx].offset as usize;
                }
            } else if entry_user.starts_with(prefix) {
                // In range — match per scan_prefix's rules.
                let entry_ver = !u64::from_be_bytes(
                    self.data[key_end - 8..key_end].try_into().unwrap(),
                );
                if entry_ver <= version {
                    let same_key = last_user_key
                        .as_ref()
                        .is_some_and(|prev| prev.as_slice() == entry_user);
                    if !same_key {
                        last_user_key = Some(entry_user.to_vec());
                        let value = if is_tomb {
                            None
                        } else {
                            Some(self.data.slice(val_start..val_end))
                        };
                        // Zero-copy user_key clone: the underlying memory is
                        // already in self.data, so .slice() is a refcount bump.
                        out[i].push((self.data.slice(key_start..key_end - 8), value));
                    }
                }
                pos = val_end;
                while block_idx + 1 < self.index.len()
                    && pos >= self.index[block_idx + 1].offset as usize
                {
                    block_idx += 1;
                }
            } else {
                // entry_user > prefix and doesn't start with it — done with this
                // prefix. Advance to the next prefix without moving pos so the
                // same entry can be re-evaluated against it.
                i += 1;
                last_user_key = None;
            }
        }

        out
    }

    /// Find the latest visible version of a user key at the given snapshot version.
    /// Scans through versioned entries to find the newest version <= snapshot.
    pub fn get_versioned(&self, user_key: &[u8], version: u64) -> Result<Option<MemValue>> {
        if self.index.is_empty() {
            return Ok(None);
        }

        // Bloom filter check: skip the read entirely if the user key is
        // definitely not in this SST.
        if !self.might_contain(user_key) {
            return Ok(None);
        }

        // Seek to the user key's SMALLEST internal key (highest version =
        // inverted bytes 0x00..00), so the block we land on is at or before
        // this user key's FIRST (highest-version) entry. We then scan forward
        // and return the first entry whose version is <= the requested one.
        //
        // Using the LARGEST internal key (version 0) here is WRONG: when a
        // user key's versions straddle a sparse-index block boundary, the
        // largest-key search lands on the block holding the OLDER version and
        // the forward scan starts there, silently skipping the newer version
        // in the previous block — a stale-read data-loss bug. Surfaced by the
        // chunked field-type migration, which flushes a memtable where every
        // object carries two versions (create + convert).
        let search_key = InternalKey::new(user_key, u64::MAX);
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
        let data_end = self.data_end;

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
            let value_start = pos;
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
                    Ok(Some(Some(self.data.slice(value_start..value_start + value_len))))
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

    /// Like `scan_prefix`, but additionally accepts a `FieldPredicate` whose
    /// per-block satisfiability is checked against the v4 zone map. Blocks
    /// whose `min..=max` bounds for the predicate field rule out a match are
    /// skipped wholesale — no entry decode, no value compare.
    ///
    /// Caller is still responsible for re-evaluating the predicate on the
    /// returned `(user_key, value)` pairs: zone maps are a coarse-grained
    /// filter and don't guarantee every entry in a candidate block matches.
    ///
    /// For SSTs without a zone map (legacy v1-v3 files, or v4 files written
    /// with no extractor), falls back to the standard `scan_prefix` path.
    pub fn scan_prefix_filtered(
        &self,
        prefix: &[u8],
        version: u64,
        predicate: &crate::zone::FieldPredicate,
    ) -> Vec<(Bytes, Option<Bytes>)> {
        let Some(zone_map) = &self.zone_map else {
            return self.scan_prefix(prefix, version);
        };

        let mut results = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;
        let start = self.prefix_start_offset(prefix);

        // We walk the same entry stream as `scan_prefix`, but track the
        // current sparse-index block so we can consult the zone map. The
        // zone check fires ONCE per block (when first entering it), not per
        // entry — checking per entry was a real ~10 ms-per-query regression
        // on the 100 K filter bench because the HashMap lookup ran 25 K
        // times instead of 6 K times per SST.
        let mut block_idx = self.block_idx_for_offset(start);
        let mut pos = start;
        let data_end = self.data_end;
        let mut current_block_passed = usize::MAX;

        while pos < data_end {
            // First-touch zone check for this block. Cached so subsequent
            // entries in the same block skip the HashMap lookup.
            if block_idx != current_block_passed {
                if !zone_map.block_could_match(
                    block_idx,
                    predicate.field_id,
                    predicate.op,
                    predicate.target,
                ) {
                    // Jump to the start of the next block (or end-of-data).
                    pos = self.block_end_offset(block_idx);
                    block_idx += 1;
                    last_user_key = None;
                    continue;
                }
                current_block_passed = block_idx;
            }

            // Decode entry. If we cross a block boundary mid-scan, advance
            // block_idx so the next iteration sees up-to-date zone bounds.
            let key_len =
                u32::from_be_bytes(self.data[pos..pos + 4].try_into().unwrap()) as usize;
            let value_len_raw =
                u32::from_be_bytes(self.data[pos + 4..pos + 8].try_into().unwrap());
            let key_start = pos + 8;
            let key_end = key_start + key_len;
            let is_tomb = value_len_raw == TOMBSTONE_MARKER;
            let val_len = if is_tomb { 0 } else { value_len_raw as usize };
            let val_end = key_end + val_len;
            if key_len < 8 {
                pos = val_end;
                continue;
            }

            let user_key = &self.data[key_start..key_end - 8];
            if user_key < prefix {
                pos = val_end;
                while block_idx + 1 < self.index.len()
                    && pos >= self.index[block_idx + 1].offset as usize
                {
                    block_idx += 1;
                }
                continue;
            }
            if !user_key.starts_with(prefix) {
                break;
            }

            let ver_bytes: [u8; 8] = self.data[key_end - 8..key_end].try_into().unwrap();
            let entry_version = !u64::from_be_bytes(ver_bytes);
            if entry_version > version {
                pos = val_end;
                while block_idx + 1 < self.index.len()
                    && pos >= self.index[block_idx + 1].offset as usize
                {
                    block_idx += 1;
                }
                continue;
            }

            let same_key = last_user_key
                .as_ref()
                .is_some_and(|prev| prev.as_slice() == user_key);
            if !same_key {
                last_user_key = Some(user_key.to_vec());
                let value = if is_tomb {
                    None
                } else {
                    Some(self.data.slice(key_start + key_len..val_end))
                };
                results.push((self.data.slice(key_start..key_end - 8), value));
            }
            pos = val_end;
            while block_idx + 1 < self.index.len()
                && pos >= self.index[block_idx + 1].offset as usize
            {
                block_idx += 1;
            }
        }

        results
    }

    /// Block index containing `offset`. Linear in the worst case but cheap —
    /// only called when seeding a filtered scan (typically once per call).
    fn block_idx_for_offset(&self, offset: usize) -> usize {
        for (i, entry) in self.index.iter().enumerate() {
            if entry.offset as usize > offset {
                return i.saturating_sub(1);
            }
        }
        self.index.len().saturating_sub(1)
    }

    /// First-byte offset of the block AFTER `block_idx` (or `data_end` if
    /// `block_idx` is the last block). Used to jump past zone-excluded blocks.
    fn block_end_offset(&self, block_idx: usize) -> usize {
        if block_idx + 1 < self.index.len() {
            self.index[block_idx + 1].offset as usize
        } else {
            self.data_end
        }
    }

    /// Scan for entries whose user key starts with `prefix`, returning the latest
    /// visible version per user key at the given snapshot version.
    ///
    /// Uses the sparse index to jump directly to the relevant byte range
    /// instead of scanning from the start of the file.
    pub fn scan_prefix(&self, prefix: &[u8], version: u64) -> Vec<(Bytes, Option<Bytes>)> {
        self.scan_prefix_impl(prefix, version, usize::MAX)
    }

    /// Seek-then-scan variant of `scan_prefix_max`: skip entries whose
    /// user_key is less than `start_user_key`, then emit at most
    /// `max_distinct` distinct user keys within the `prefix` range. Powers
    /// `Gt` / `Ge` range queries on a secondary index, where the lower
    /// bound is the predicate's target value (not the field prefix itself).
    pub fn scan_from_max(
        &self,
        prefix: &[u8],
        start_user_key: &[u8],
        version: u64,
        max_distinct: usize,
    ) -> Vec<(Bytes, Option<Bytes>)> {
        if max_distinct == 0 || !start_user_key.starts_with(prefix) {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;

        // The sparse-index binary search uses the lowest-version-tagged
        // internal-key form of the seek point — same trick as
        // `prefix_start_offset`.
        let start_offset = self.user_key_start_offset(start_user_key);

        for (key, value) in self.iter_from(start_offset) {
            if key.len() < 8 {
                continue;
            }
            let user_key = &key[..key.len() - 8];

            if user_key < start_user_key {
                continue;
            }
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
            let user_key_len = key.len() - 8;
            results.push((key.slice(..user_key_len), value));
            if results.len() >= max_distinct {
                break;
            }
        }

        results
    }

    /// Binary-search the sparse index for the byte offset of the block that
    /// could contain `start_user_key`. Sibling of `prefix_start_offset` for
    /// the case where the caller has a full user key (not just a prefix).
    fn user_key_start_offset(&self, start_user_key: &[u8]) -> usize {
        let header_size = SST_MAGIC.len() + 4;
        if self.index.is_empty() {
            return header_size;
        }
        let mut search_key = Vec::with_capacity(start_user_key.len() + 8);
        search_key.extend_from_slice(start_user_key);
        search_key.extend_from_slice(&[0u8; 8]);
        let block_idx = match self
            .index
            .binary_search_by(|e| e.key.as_ref().cmp(&search_key))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        self.index[block_idx].offset as usize
    }

    /// Bounded variant: stop after collecting `max_distinct` user keys. Used
    /// by the LSM bounded range-scan primitive that powers secondary-index
    /// `LIMIT N` push-down — cuts an O(prefix_size) per-SST walk down to
    /// O(N).
    pub fn scan_prefix_max(
        &self,
        prefix: &[u8],
        version: u64,
        max_distinct: usize,
    ) -> Vec<(Bytes, Option<Bytes>)> {
        self.scan_prefix_impl(prefix, version, max_distinct)
    }

    fn scan_prefix_impl(
        &self,
        prefix: &[u8],
        version: u64,
        max_distinct: usize,
    ) -> Vec<(Bytes, Option<Bytes>)> {
        if max_distinct == 0 {
            return Vec::new();
        }
        let mut results = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;

        let start = self.prefix_start_offset(prefix);

        for (key, value) in self.iter_from(start) {
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
            // Zero-copy slice into the iterator's key Bytes (which already
            // shares the SstReader's data refcount).
            let user_key_len = key.len() - 8;
            results.push((key.slice(..user_key_len), value));
            if results.len() >= max_distinct {
                break;
            }
        }

        results
    }

    /// Find the byte offset where scanning should start for entries whose
    /// user key begins with `prefix`. Uses the sparse index for an O(log N)
    /// jump instead of scanning from the start of the file.
    fn prefix_start_offset(&self, prefix: &[u8]) -> usize {
        let header_size = SST_MAGIC.len() + 4;
        if self.index.is_empty() {
            return header_size;
        }

        // The smallest InternalKey whose user key starts with `prefix` has
        // the version inverted to all zeros (version = u64::MAX).
        let mut search_key = Vec::with_capacity(prefix.len() + 8);
        search_key.extend_from_slice(prefix);
        search_key.extend_from_slice(&[0u8; 8]);

        let block_idx = match self
            .index
            .binary_search_by(|e| e.key.as_ref().cmp(&search_key))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };

        self.index[block_idx].offset as usize
    }

    /// Iterate all entries in the SST in sorted order.
    pub fn iter(&self) -> SstIterator<'_> {
        let header_size = SST_MAGIC.len() + 4;
        self.iter_from(header_size)
    }

    /// Iterate entries starting at the given byte offset. The offset must
    /// point to the start of an entry header.
    pub fn iter_from(&self, start: usize) -> SstIterator<'_> {
        SstIterator {
            data: &self.data,
            pos: start,
            end: self.data_end,
        }
    }
}

/// Iterator over SST entries. Holds a reference to the `Bytes` owner so each
/// emitted key/value is a zero-copy `data.slice(...)` instead of an allocating
/// `Bytes::copy_from_slice`.
pub struct SstIterator<'a> {
    data: &'a Bytes,
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

        let key = self.data.slice(self.pos..self.pos + key_len);
        self.pos += key_len;

        let is_tombstone = value_len_raw == TOMBSTONE_MARKER;
        let value_len = if is_tombstone { 0 } else { value_len_raw as usize };

        let value = if is_tombstone {
            None
        } else {
            Some(self.data.slice(self.pos..self.pos + value_len))
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

    /// Build an SST where each type_id (1..type_count) has `per_type` objects.
    /// Returns the path. Keys are written in sorted order.
    fn build_multi_type_sst(
        path: &Path,
        type_count: u64,
        per_type: u64,
    ) -> usize {
        let mut writer = SstWriter::new(path).unwrap();
        for type_id in 1..=type_count {
            for obj_id in 0..per_type {
                let key = InternalKey::new(&KeyBuilder::object(type_id, obj_id), 1);
                writer
                    .add(
                        key.as_bytes(),
                        &Some(Bytes::from(format!("t{type_id}-o{obj_id}"))),
                    )
                    .unwrap();
            }
        }
        let meta = writer.finish().unwrap();
        meta.entry_count
    }

    #[test]
    fn scan_prefix_returns_matching_entries() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("prefix.sst");
        build_multi_type_sst(&sst_path, 5, 100);

        let reader = SstReader::open(&sst_path).unwrap();

        // Scan for objects of type 3 only.
        let prefix = KeyBuilder::object_prefix(3);
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert_eq!(results.len(), 100, "should find all 100 objects of type 3");

        // Verify all results belong to type 3.
        for (user_key, _) in &results {
            assert!(
                user_key.starts_with(&prefix),
                "result {:?} does not start with prefix",
                user_key
            );
        }
    }

    #[test]
    fn scan_prefix_uses_sparse_index() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("sparse_scan.sst");
        build_multi_type_sst(&sst_path, 5, 200);

        let reader = SstReader::open(&sst_path).unwrap();

        // Scanning for type 5 (the last block) should start past the first
        // sparse index entry — not at offset 8.
        let prefix = KeyBuilder::object_prefix(5);
        let start = reader.prefix_start_offset(&prefix);
        assert!(
            start > (SST_MAGIC.len() + 4),
            "expected sparse index to skip past header for last-type prefix, got offset {start}"
        );

        // And it should still return the right results.
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert_eq!(results.len(), 200);
    }

    #[test]
    fn scan_prefix_at_start_of_file() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("first.sst");
        build_multi_type_sst(&sst_path, 5, 100);

        let reader = SstReader::open(&sst_path).unwrap();
        let prefix = KeyBuilder::object_prefix(1);
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn scan_prefix_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("nomatch.sst");
        build_multi_type_sst(&sst_path, 5, 100);

        let reader = SstReader::open(&sst_path).unwrap();
        // Type 99 doesn't exist.
        let prefix = KeyBuilder::object_prefix(99);
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert!(results.is_empty());
    }

    #[test]
    fn scan_prefix_before_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("before.sst");
        // type ids 10..=14
        let mut writer = SstWriter::new(&sst_path).unwrap();
        for type_id in 10u64..=14 {
            for obj_id in 0u64..50 {
                let key = InternalKey::new(&KeyBuilder::object(type_id, obj_id), 1);
                writer.add(key.as_bytes(), &Some(Bytes::from("x"))).unwrap();
            }
        }
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        // Looking for type 1 — sorts before everything.
        let prefix = KeyBuilder::object_prefix(1);
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert!(results.is_empty());
    }

    #[test]
    fn scan_prefix_empty_sst() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("empty.sst");
        let writer = SstWriter::new(&sst_path).unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let prefix = KeyBuilder::object_prefix(1);
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert!(results.is_empty());
    }

    #[test]
    fn scan_prefix_only_returns_latest_version_per_key() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("versioned.sst");

        let mut writer = SstWriter::new(&sst_path).unwrap();
        let user_key = KeyBuilder::object(1, 0);
        // Versions in sorted order: highest first (smallest InternalKey bytes).
        for &v in &[10u64, 5, 1] {
            let key = InternalKey::new(&user_key, v);
            writer
                .add(key.as_bytes(), &Some(Bytes::from(format!("v{v}"))))
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let prefix = KeyBuilder::object_prefix(1);
        let results = reader.scan_prefix(&prefix, 7);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, Some(Bytes::from("v5")));
    }

    #[test]
    fn scan_prefix_speedup_via_sparse_index() {
        // Build a 10K-entry SST with 100 types × 100 objects.
        // Scan for the last type (100) — should jump past ~99% of the file.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("bench.sst");
        let total = build_multi_type_sst(&sst_path, 100, 100);
        assert_eq!(total, 10_000);

        let reader = SstReader::open(&sst_path).unwrap();
        let prefix = KeyBuilder::object_prefix(100);

        let header_size = SST_MAGIC.len() + 4;
        let data_size = reader.data_end - header_size;

        let start = reader.prefix_start_offset(&prefix);
        let bytes_skipped = start - header_size;
        let skipped_pct = (bytes_skipped as f64 / data_size as f64) * 100.0;

        eprintln!(
            "scan_prefix sparse index: skipped {bytes_skipped} / {data_size} bytes ({skipped_pct:.1}%)"
        );

        // For a prefix matching only the last type, we should skip the vast
        // majority of the file (>90%).
        assert!(
            skipped_pct > 90.0,
            "expected to skip >90% of the file, got {skipped_pct:.1}%"
        );

        // And results must be correct.
        let results = reader.scan_prefix(&prefix, u64::MAX);
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn bloom_filter_persists_in_sst() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("bloom.sst");

        let mut writer = SstWriter::new(&sst_path).unwrap();
        let user_key = KeyBuilder::object(1, 42);
        let key = InternalKey::new(&user_key, 1);
        writer.add(key.as_bytes(), &Some(Bytes::from("hi"))).unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        // Present key — bloom says might-contain.
        assert!(reader.might_contain(&user_key));
        // Absent key — bloom should say definitely not.
        let absent = KeyBuilder::object(1, 99);
        assert!(!reader.might_contain(&absent));
    }

    #[test]
    fn bloom_filter_no_false_negatives_at_scale() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("bloom_scale.sst");
        let count = build_multi_type_sst(&sst_path, 10, 100);
        assert_eq!(count, 1000);

        let reader = SstReader::open(&sst_path).unwrap();

        // All inserted keys must report might-contain (no false negatives).
        for type_id in 1u64..=10 {
            for obj_id in 0u64..100 {
                let key = KeyBuilder::object(type_id, obj_id);
                assert!(
                    reader.might_contain(&key),
                    "false negative for type {type_id} obj {obj_id}"
                );
            }
        }
    }

    #[test]
    fn bloom_filter_rejects_absent_keys() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("bloom_absent.sst");
        build_multi_type_sst(&sst_path, 5, 100);

        let reader = SstReader::open(&sst_path).unwrap();

        // Keys from types not in the SST should mostly be rejected.
        let mut rejected = 0;
        let total = 500;
        for type_id in 100u64..600 {
            let key = KeyBuilder::object(type_id, 0);
            if !reader.might_contain(&key) {
                rejected += 1;
            }
        }
        let rejection_rate = rejected as f64 / total as f64;
        eprintln!(
            "bloom rejection rate for absent keys: {:.3} ({rejected}/{total})",
            rejection_rate
        );
        // With ~1% FPR, we should reject ~99% of absent keys.
        assert!(
            rejection_rate > 0.95,
            "bloom rejected only {rejection_rate} of absent keys, expected > 0.95"
        );
    }

    #[test]
    fn get_short_circuits_via_bloom() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("short_circuit.sst");
        build_multi_type_sst(&sst_path, 3, 50);

        let reader = SstReader::open(&sst_path).unwrap();

        // Key for a type not in the SST.
        let absent_user = KeyBuilder::object(99, 0);
        let absent_internal = InternalKey::new(&absent_user, 1);
        assert_eq!(reader.get(absent_internal.as_bytes()).unwrap(), None);
        assert_eq!(reader.get_versioned(&absent_user, u64::MAX).unwrap(), None);

        // Present key still works.
        let present_user = KeyBuilder::object(2, 10);
        let present_internal = InternalKey::new(&present_user, 1);
        let result = reader.get(present_internal.as_bytes()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn empty_sst_has_no_bloom_false_match() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("empty_bloom.sst");
        let writer = SstWriter::new(&sst_path).unwrap();
        writer.finish().unwrap();

        let reader = SstReader::open(&sst_path).unwrap();
        let key = KeyBuilder::object(1, 0);
        // Empty bloom filter must reject everything.
        assert!(!reader.might_contain(&key));
    }

    // --- multi_get_versioned (sorted-batch SST sweep) -----------------------

    /// Build an SST with `count` objects of type 1, ids 0..count, version 1.
    /// Value is `b"v-{id}"`. Returns the path.
    fn build_dense_object_sst(path: &Path, count: u64) -> SstReader {
        let mut writer = SstWriter::new(path).unwrap();
        for id in 0..count {
            let user = KeyBuilder::object(1, id);
            let key = InternalKey::new(&user, 1);
            writer
                .add(key.as_bytes(), &Some(Bytes::from(format!("v-{id}"))))
                .unwrap();
        }
        writer.finish().unwrap();
        SstReader::open(path).unwrap()
    }

    fn ok_value(s: &str) -> Option<MemValue> {
        Some(Some(Bytes::from(s.to_string())))
    }

    #[test]
    fn multi_get_versioned_empty_batch() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("e.sst"), 10);
        let results = reader.multi_get_versioned(&[], u64::MAX).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn multi_get_versioned_all_hits_within_one_block() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("dense.sst"), 32);

        // 5 contiguous needles in the second block (ids 16..21).
        let keys: Vec<Bytes> = (16u64..21).map(|i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let results = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        assert_eq!(results.len(), 5);
        for (offset, val) in results.into_iter().enumerate() {
            let id = 16 + offset as u64;
            assert_eq!(val, ok_value(&format!("v-{id}")));
        }
    }

    #[test]
    fn multi_get_versioned_scattered_triggers_gallop() {
        // 256 objects = 16 sparse blocks. Needles spread across blocks force
        // the threshold gallop to re-seek the sparse index between matches.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("scattered.sst"), 256);

        let ids: Vec<u64> = vec![3, 47, 100, 175, 230];
        let keys: Vec<Bytes> = ids.iter().map(|&i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let results = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        assert_eq!(results.len(), ids.len());
        for (idx, &id) in ids.iter().enumerate() {
            assert_eq!(results[idx], ok_value(&format!("v-{id}")));
        }
    }

    #[test]
    fn multi_get_versioned_mixed_hits_and_misses() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("mixed.sst"), 100);

        // Mix present (in 0..100) with absent (>=100). Sorted ascending.
        let ids: Vec<u64> = vec![5, 42, 99, 150, 200];
        let keys: Vec<Bytes> = ids.iter().map(|&i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let results = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        assert_eq!(results[0], ok_value("v-5"));
        assert_eq!(results[1], ok_value("v-42"));
        assert_eq!(results[2], ok_value("v-99"));
        assert_eq!(results[3], None); // absent
        assert_eq!(results[4], None); // absent
    }

    #[test]
    fn multi_get_versioned_misses_only() {
        // All-misses path: bloom rejects every needle, fast exit.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("only_low.sst"), 20);

        let ids: Vec<u64> = vec![1000, 2000, 3000];
        let keys: Vec<Bytes> = ids.iter().map(|&i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let results = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        assert_eq!(results, vec![None, None, None]);
    }

    #[test]
    fn multi_get_versioned_tombstone_returned_as_some_none() {
        // Tombstone is the latest visible entry for a key. multi_get_versioned
        // must report Some(None) so the LsmTree caller knows to STOP searching
        // older layers (matching get_versioned's contract).
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("tomb.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();

        // Sorted order: ids 1..3, with id=2 having a tombstone at version 2 and a put at version 1.
        // InternalKey ordering: higher version first per user key, so tombstone (v2) comes before put (v1).
        let k1 = InternalKey::new(&KeyBuilder::object(1, 1), 1);
        writer.add(k1.as_bytes(), &Some(Bytes::from("alive"))).unwrap();

        let k2_tomb = InternalKey::new(&KeyBuilder::object(1, 2), 2);
        writer.add(k2_tomb.as_bytes(), &None).unwrap();
        let k2_put = InternalKey::new(&KeyBuilder::object(1, 2), 1);
        writer.add(k2_put.as_bytes(), &Some(Bytes::from("dead"))).unwrap();

        let k3 = InternalKey::new(&KeyBuilder::object(1, 3), 1);
        writer.add(k3.as_bytes(), &Some(Bytes::from("third"))).unwrap();

        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let keys: Vec<Bytes> = (1u64..=3).map(|i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let results = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        assert_eq!(results[0], ok_value("alive"));
        assert_eq!(results[1], Some(None), "tombstone must surface as Some(None)");
        assert_eq!(results[2], ok_value("third"));
    }

    #[test]
    fn multi_get_versioned_respects_snapshot_version() {
        // Same user key at v1, v5, v10. Snapshots see the right one.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("ver.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();

        let user = KeyBuilder::object(1, 42);
        for &v in &[10u64, 5, 1] {
            let key = InternalKey::new(&user, v);
            writer
                .add(key.as_bytes(), &Some(Bytes::from(format!("v{v}"))))
                .unwrap();
        }
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let refs: Vec<&[u8]> = vec![user.as_ref()];

        assert_eq!(
            reader.multi_get_versioned(&refs, 10).unwrap()[0],
            ok_value("v10")
        );
        assert_eq!(
            reader.multi_get_versioned(&refs, 7).unwrap()[0],
            ok_value("v5")
        );
        assert_eq!(
            reader.multi_get_versioned(&refs, 3).unwrap()[0],
            ok_value("v1")
        );
        assert_eq!(reader.multi_get_versioned(&refs, 0).unwrap()[0], None);
    }

    // --- multi_scan_prefix_versioned (sorted-batch SST prefix sweep) -------

    /// Build an SST holding `per_src` edges from each source id in 1..=`src_count`
    /// on rel_id 7. Edge value is `b"edge-{src}-{tgt}"`. Returns the reader.
    fn build_edge_sst(path: &Path, src_count: u64, per_src: u64) -> SstReader {
        let mut writer = SstWriter::new(path).unwrap();
        for src in 1..=src_count {
            for tgt in 0..per_src {
                let user = KeyBuilder::edge(src, 7, tgt);
                let key = InternalKey::new(&user, 1);
                writer
                    .add(
                        key.as_bytes(),
                        &Some(Bytes::from(format!("edge-{src}-{tgt}"))),
                    )
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        SstReader::open(path).unwrap()
    }

    #[test]
    fn multi_scan_prefix_versioned_empty_batch() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 5, 3);
        let results = reader.multi_scan_prefix_versioned(&[], u64::MAX);
        assert!(results.is_empty());
    }

    #[test]
    fn multi_scan_prefix_versioned_single_prefix_matches_scan_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 5, 4);

        let p = KeyBuilder::edge_prefix(3, 7);
        let refs: Vec<&[u8]> = vec![p.as_ref()];
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);
        let single = reader.scan_prefix(&p, u64::MAX);

        assert_eq!(batched.len(), 1);
        assert_eq!(batched[0], single);
        assert_eq!(batched[0].len(), 4);
    }

    #[test]
    fn multi_scan_prefix_versioned_multiple_disjoint_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 10, 5);

        // Sorted by src id (= sorted bytewise since edge prefix is fixed-width BE).
        let prefixes: Vec<Bytes> = vec![1u64, 3, 7]
            .into_iter()
            .map(|s| KeyBuilder::edge_prefix(s, 7))
            .collect();
        let refs: Vec<&[u8]> = prefixes.iter().map(|p| p.as_ref()).collect();
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);

        assert_eq!(batched.len(), 3);
        for (idx, src) in [1u64, 3, 7].iter().enumerate() {
            assert_eq!(batched[idx].len(), 5, "src {src} should have 5 entries");
            for (i, (key, _)) in batched[idx].iter().enumerate() {
                let want = KeyBuilder::edge(*src, 7, i as u64);
                assert_eq!(key.as_ref(), want.as_ref());
            }
        }
    }

    #[test]
    fn multi_scan_prefix_versioned_mixed_present_and_absent() {
        // src 1, 5, 9 exist; src 100 and 500 don't. Sorted ascending.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 10, 3);

        let prefixes: Vec<Bytes> = vec![1u64, 5, 9, 100, 500]
            .into_iter()
            .map(|s| KeyBuilder::edge_prefix(s, 7))
            .collect();
        let refs: Vec<&[u8]> = prefixes.iter().map(|p| p.as_ref()).collect();
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);

        assert_eq!(batched[0].len(), 3); // src 1
        assert_eq!(batched[1].len(), 3); // src 5
        assert_eq!(batched[2].len(), 3); // src 9
        assert_eq!(batched[3].len(), 0); // absent
        assert_eq!(batched[4].len(), 0); // absent
    }

    #[test]
    fn multi_scan_prefix_versioned_scattered_triggers_gallop() {
        // Many SST entries to span multiple sparse blocks; a few far-apart
        // prefixes exercise the threshold gallop in the Less branch.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 200, 2);

        let prefixes: Vec<Bytes> = vec![1u64, 50, 150, 199]
            .into_iter()
            .map(|s| KeyBuilder::edge_prefix(s, 7))
            .collect();
        let refs: Vec<&[u8]> = prefixes.iter().map(|p| p.as_ref()).collect();
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);

        for (idx, src) in [1u64, 50, 150, 199].iter().enumerate() {
            assert_eq!(batched[idx].len(), 2, "src {src}");
        }
    }

    #[test]
    fn multi_scan_prefix_versioned_tombstone_preserved_as_some_none() {
        // Latest version for an edge is a tombstone; multi_scan_prefix must
        // surface it as Some(None) so the LsmTree merge can suppress an
        // older put from the same or an older layer.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("e.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();

        // src 1 has two edges: (1,7,10) live; (1,7,20) tombstoned at v2, put at v1.
        let k_a = InternalKey::new(&KeyBuilder::edge(1, 7, 10), 1);
        writer.add(k_a.as_bytes(), &Some(Bytes::from("live"))).unwrap();

        let k_b_tomb = InternalKey::new(&KeyBuilder::edge(1, 7, 20), 2);
        writer.add(k_b_tomb.as_bytes(), &None).unwrap();
        let k_b_put = InternalKey::new(&KeyBuilder::edge(1, 7, 20), 1);
        writer
            .add(k_b_put.as_bytes(), &Some(Bytes::from("old")))
            .unwrap();

        // src 3 just to ensure the cursor advances past src 1 cleanly.
        let k_c = InternalKey::new(&KeyBuilder::edge(3, 7, 0), 1);
        writer
            .add(k_c.as_bytes(), &Some(Bytes::from("other")))
            .unwrap();
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let p = KeyBuilder::edge_prefix(1, 7);
        let refs: Vec<&[u8]> = vec![p.as_ref()];
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);

        assert_eq!(batched[0].len(), 2);
        // Entries sorted by user_key: (1,7,10) first, then (1,7,20).
        assert_eq!(batched[0][0].1, Some(Bytes::from("live")));
        assert_eq!(batched[0][1].1, None, "tombstone surfaces as Some(None)");
    }

    #[test]
    fn multi_scan_prefix_versioned_respects_snapshot_version() {
        // Edge (1,7,0) at v10 and v5; snapshots at 3, 7, 10 pick the right one.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("e.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();

        let user = KeyBuilder::edge(1, 7, 0);
        for &v in &[10u64, 5] {
            let key = InternalKey::new(&user, v);
            writer
                .add(key.as_bytes(), &Some(Bytes::from(format!("v{v}"))))
                .unwrap();
        }
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let p = KeyBuilder::edge_prefix(1, 7);
        let refs: Vec<&[u8]> = vec![p.as_ref()];

        let r10 = reader.multi_scan_prefix_versioned(&refs, 10);
        assert_eq!(r10[0][0].1, Some(Bytes::from("v10")));

        let r7 = reader.multi_scan_prefix_versioned(&refs, 7);
        assert_eq!(r7[0][0].1, Some(Bytes::from("v5")));

        let r3 = reader.multi_scan_prefix_versioned(&refs, 3);
        assert!(r3[0].is_empty(), "no version <= 3 should be visible");
    }

    #[test]
    fn multi_scan_prefix_versioned_matches_scan_prefix_at_scale() {
        // Differential test: for many prefixes at once, the batch result must
        // exactly equal calling scan_prefix per prefix.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_edge_sst(&dir.path().join("e.sst"), 200, 8);

        let mut srcs: Vec<u64> = (1u64..=200).step_by(5).collect();
        srcs.extend([1, 7, 13, 137, 199]);
        srcs.sort_unstable();
        srcs.dedup();

        let prefixes: Vec<Bytes> = srcs
            .iter()
            .map(|s| KeyBuilder::edge_prefix(*s, 7))
            .collect();
        let refs: Vec<&[u8]> = prefixes.iter().map(|p| p.as_ref()).collect();
        let batched = reader.multi_scan_prefix_versioned(&refs, u64::MAX);

        for (idx, p) in prefixes.iter().enumerate() {
            let single = reader.scan_prefix(p, u64::MAX);
            assert_eq!(
                batched[idx], single,
                "mismatch at prefix idx {idx} (src={})",
                srcs[idx]
            );
        }
    }

    // --- range pruning (first_user_key / last_user_key + overlap checks) ---

    #[test]
    fn first_and_last_user_key_are_accurate() {
        // Write 50 entries; first user key should be obj(1,0), last obj(1,49).
        // last_user_key is computed by walking the last data block (not just
        // the last sparse-index entry, which only records each block's first
        // key — so a naive impl would return obj(1,48) here).
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("range.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();
        for id in 0..50u64 {
            let key = InternalKey::new(&KeyBuilder::object(1, id), 1);
            writer.add(key.as_bytes(), &Some(Bytes::from("v"))).unwrap();
        }
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let first = KeyBuilder::object(1, 0);
        let last = KeyBuilder::object(1, 49);
        assert_eq!(reader.first_user_key.as_deref(), Some(first.as_ref()));
        assert_eq!(reader.last_user_key.as_deref(), Some(last.as_ref()));
    }

    #[test]
    fn first_and_last_user_key_for_empty_sst() {
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("empty.sst");
        SstWriter::new(&sst_path).unwrap().finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();
        assert!(reader.first_user_key.is_none());
        assert!(reader.last_user_key.is_none());
    }

    #[test]
    fn range_overlaps_sorted_skips_above_and_below() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("e.sst"), 100);
        // Range is obj(1,0)..obj(1,99).

        // Entirely below.
        let below: Vec<Bytes> = vec![KeyBuilder::object(0, 5), KeyBuilder::object(0, 9)];
        let refs: Vec<&[u8]> = below.iter().map(|k| k.as_ref()).collect();
        assert!(
            !reader.range_overlaps_sorted(&refs),
            "all-below batch should be pruned"
        );

        // Entirely above.
        let above: Vec<Bytes> = vec![KeyBuilder::object(2, 0), KeyBuilder::object(2, 9)];
        let refs: Vec<&[u8]> = above.iter().map(|k| k.as_ref()).collect();
        assert!(
            !reader.range_overlaps_sorted(&refs),
            "all-above batch should be pruned"
        );

        // Straddling — must not be pruned.
        let straddle: Vec<Bytes> = vec![KeyBuilder::object(0, 5), KeyBuilder::object(1, 50)];
        let refs: Vec<&[u8]> = straddle.iter().map(|k| k.as_ref()).collect();
        assert!(reader.range_overlaps_sorted(&refs));

        // Entirely inside.
        let inside: Vec<Bytes> = vec![KeyBuilder::object(1, 10), KeyBuilder::object(1, 20)];
        let refs: Vec<&[u8]> = inside.iter().map(|k| k.as_ref()).collect();
        assert!(reader.range_overlaps_sorted(&refs));
    }

    #[test]
    fn range_overlaps_sorted_empty_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("e.sst"), 10);
        // Empty needle batch: never overlaps (nothing to look up).
        assert!(!reader.range_overlaps_sorted(&[]));

        // Empty SST: never overlaps (no entries to match).
        let empty_sst_path = dir.path().join("empty.sst");
        SstWriter::new(&empty_sst_path).unwrap().finish().unwrap();
        let empty_reader = SstReader::open(&empty_sst_path).unwrap();
        let needle = KeyBuilder::object(1, 0);
        assert!(!empty_reader.range_overlaps_sorted(&[needle.as_ref()]));
    }

    #[test]
    fn prefix_range_overlaps_sorted_skips_above_and_below() {
        // SST has edges from src 10..20 on rel 7. Range is e:10..7.. to e:19..7..N.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("e.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();
        for src in 10u64..20 {
            for tgt in 0u64..3 {
                let key = InternalKey::new(&KeyBuilder::edge(src, 7, tgt), 1);
                writer.add(key.as_bytes(), &Some(Bytes::from("v"))).unwrap();
            }
        }
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        // All prefixes target src > 19 — entirely above.
        let above: Vec<Bytes> =
            vec![KeyBuilder::edge_prefix(50, 7), KeyBuilder::edge_prefix(100, 7)];
        let refs: Vec<&[u8]> = above.iter().map(|p| p.as_ref()).collect();
        assert!(
            !reader.prefix_range_overlaps_sorted(&refs),
            "all-above prefix batch should be pruned"
        );

        // All prefixes target src < 10 — entirely below.
        let below: Vec<Bytes> =
            vec![KeyBuilder::edge_prefix(1, 7), KeyBuilder::edge_prefix(5, 7)];
        let refs: Vec<&[u8]> = below.iter().map(|p| p.as_ref()).collect();
        assert!(
            !reader.prefix_range_overlaps_sorted(&refs),
            "all-below prefix batch should be pruned"
        );

        // Prefix that lands inside the range — must not be pruned.
        let inside: Vec<Bytes> = vec![KeyBuilder::edge_prefix(15, 7)];
        let refs: Vec<&[u8]> = inside.iter().map(|p| p.as_ref()).collect();
        assert!(reader.prefix_range_overlaps_sorted(&refs));
    }

    // --- v5 zone map + scan_prefix_filtered --------------------------------

    /// Synthetic field_id for the `year` column in zone-map tests. In
    /// production this comes from the catalog (`c:F:` rows); here we use a
    /// small constant so the test is hermetic.
    const YEAR_FIELD_ID: u32 = 1;

    /// Build an SST of "movie" entries (one int field "year") with the given
    /// (id, year) pairs, in id-sorted order. Returns the reader.
    fn build_movie_sst(path: &Path, entries: &[(u64, u32)]) -> SstReader {
        use crate::zone::ZoneFieldExtractor;

        // Naive extractor: parse our test format (just the year as the value).
        let extractor: ZoneFieldExtractor = std::sync::Arc::new(
            move |_internal_key: &[u8], value: &[u8]| -> Vec<(u32, [u8; 8])> {
                // Test value format: 4 bytes BE u32 of `year`.
                if value.len() != 4 {
                    return Vec::new();
                }
                let year = u32::from_be_bytes(value.try_into().unwrap()) as u64;
                vec![(YEAR_FIELD_ID, year.to_be_bytes())]
            },
        );

        let mut writer = SstWriter::new_with_zone_extractor(path, Some(extractor)).unwrap();
        for &(id, year) in entries {
            let user_key = KeyBuilder::object(1, id);
            let internal_key = InternalKey::new(&user_key, 1);
            let value = Bytes::copy_from_slice(&year.to_be_bytes());
            writer.add(internal_key.as_bytes(), &Some(value)).unwrap();
        }
        writer.finish().unwrap();
        SstReader::open(path).unwrap()
    }

    #[test]
    fn v5_sst_carries_zone_map() {
        let dir = tempfile::tempdir().unwrap();
        let entries: Vec<(u64, u32)> = (0u64..32).map(|i| (i, 1900 + i as u32)).collect();
        let reader = build_movie_sst(&dir.path().join("v5.sst"), &entries);

        let zone = reader.zone_map().expect("v5 SST must carry zone map");
        // 32 entries / 16-per-block = 2 blocks. Block 0: years 1900-1915.
        // Block 1: years 1916-1931.
        let (min0, max0) = zone.bounds(0, YEAR_FIELD_ID).unwrap();
        assert_eq!(min0, 1900);
        assert_eq!(max0, 1915);
        let (min1, max1) = zone.bounds(1, YEAR_FIELD_ID).unwrap();
        assert_eq!(min1, 1916);
        assert_eq!(max1, 1931);
    }

    #[test]
    fn scan_prefix_filtered_skips_blocks_that_cant_match() {
        use crate::zone::{CompareOp, FieldPredicate};

        let dir = tempfile::tempdir().unwrap();
        // 4 blocks of 16 entries each. Years are tightly clustered per block:
        // block 0: 1900-1915, 1: 1916-1931, 2: 1932-1947, 3: 1948-1963.
        let entries: Vec<(u64, u32)> = (0u64..64).map(|i| (i, 1900 + i as u32)).collect();
        let reader = build_movie_sst(&dir.path().join("filt.sst"), &entries);

        // year > 1940 must skip blocks 0 and 1 entirely (their max < 1940).
        // Predicate target = 1940 encoded as u64.
        let predicate = FieldPredicate {
            field_id: YEAR_FIELD_ID,
            op: CompareOp::Gt,
            target: 1940,
        };
        let prefix = KeyBuilder::object_prefix(1);
        let results = reader.scan_prefix_filtered(&prefix, u64::MAX, &predicate);
        // Block 2 entries 32..47 → years 1932..1947, of which 1941..1947 match (>1940 → 7 entries).
        // Block 3 entries 48..63 → years 1948..1963, all 16 match.
        // Caller re-evaluates predicate; for now we just check the candidate
        // set was correctly narrowed to entries in blocks 2 & 3 = 32 entries
        // (the filter doesn't apply the predicate itself).
        assert_eq!(results.len(), 32, "should include entries from blocks 2+3 only");
    }

    #[test]
    fn scan_prefix_filtered_falls_back_when_no_zone_map() {
        use crate::zone::{CompareOp, FieldPredicate};

        // SST built WITHOUT an extractor — no zone map (well, an empty one).
        // scan_prefix_filtered should behave like scan_prefix.
        let dir = tempfile::tempdir().unwrap();
        let sst_path = dir.path().join("nofilt.sst");
        let mut writer = SstWriter::new(&sst_path).unwrap();
        for id in 0u64..10 {
            let user_key = KeyBuilder::object(1, id);
            let internal_key = InternalKey::new(&user_key, 1);
            writer
                .add(internal_key.as_bytes(), &Some(Bytes::from("v")))
                .unwrap();
        }
        writer.finish().unwrap();
        let reader = SstReader::open(&sst_path).unwrap();

        let predicate = FieldPredicate {
            field_id: YEAR_FIELD_ID,
            op: CompareOp::Gt,
            target: 1900,
        };
        let prefix = KeyBuilder::object_prefix(1);
        // Empty zone map → can't rule out anything → returns everything.
        let results = reader.scan_prefix_filtered(&prefix, u64::MAX, &predicate);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn multi_get_versioned_matches_get_versioned_at_scale() {
        // Differential test: for a random-ish batch, every result must match
        // calling get_versioned independently. Catches off-by-ones in the
        // gallop and the same-user-key advance.
        let dir = tempfile::tempdir().unwrap();
        let reader = build_dense_object_sst(&dir.path().join("diff.sst"), 1024);

        let ids: Vec<u64> = (0u64..1024).step_by(7).chain([5, 11, 17, 999, 1023, 1500]).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();

        let keys: Vec<Bytes> = sorted.iter().map(|&i| KeyBuilder::object(1, i)).collect();
        let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let batched = reader.multi_get_versioned(&refs, u64::MAX).unwrap();

        for (idx, &id) in sorted.iter().enumerate() {
            let user = KeyBuilder::object(1, id);
            let single = reader.get_versioned(&user, u64::MAX).unwrap();
            assert_eq!(
                batched[idx], single,
                "mismatch at id {id}: batched={:?} single={:?}",
                batched[idx], single
            );
        }
    }
}
