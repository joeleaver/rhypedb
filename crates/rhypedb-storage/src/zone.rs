//! Per-block zone maps for integer fields.
//!
//! A zone map records the min/max value of opted-in integer fields within
//! each sparse-index block of an SST. At query time, a filter predicate like
//! `Movie.year > 2010` can skip whole blocks whose `year` range falls entirely
//! on the wrong side of the predicate — no entry decode, no value compare.
//!
//! The storage layer stays schema-agnostic: it takes pre-encoded 8-byte values
//! (engine flips signed-int MSBs / widens narrow ints so byte order matches
//! numeric order) and a `field_id` — the stable u32 catalog ID for the field.
//! `field_id` (not a name hash) is the column key: it survives `rename_field`
//! by construction, so an SST written before a field rename still prunes
//! correctly after the rename.
//!
//! **Format on disk (v5 SST):**
//!
//! ```text
//! [num_blocks: u32 BE]
//! [num_fields: u32 BE]
//! [field_ids: u32 BE × num_fields]
//! [per (block, field): [min: 8 bytes BE][max: 8 bytes BE]]
//! ```
//!
//! Per-field per-block cost: 16 bytes. For a 100 K-entry SST with one
//! zone-mapped field, that's 6250 blocks × 16 = 100 KB sidecar — a small
//! tax for the block-skipping speedup.
//!
//! **No-data sentinel:** a fresh block's bounds initialize to
//! `(min: u64::MAX, max: u64::MIN)`. Any check that finds `min > max`
//! interprets the block as "no data for this field — could match, must scan."
//! When an entry with the field arrives, `min = min(min, v)` and
//! `max = max(max, v)` populate real bounds.
//!
//! **Historical note:** v4 SSTs keyed columns by `FNV-1a(field_name)`. A
//! `rename_field` left those columns stranded under the old name's hash so
//! queries against the new name skipped every block. v5 fixes this by
//! switching to `field_id`. The v4 reader path still opens the file (data
//! and bloom are unaffected) but discards the zone map — those SSTs serve
//! correct results without block pruning until natural compaction rewrites
//! them as v5.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

use bytes::{BufMut, BytesMut};

use crate::{Error, Result};

/// Closure that pulls zone-mapped (field_id, encoded-8-byte) tuples
/// out of an SST entry's value bytes. Signature is `(internal_key, value_bytes)
/// -> Vec<(u32, [u8; 8])>` so the closure can short-circuit on non-object keys
/// (edges, unique-index entries, etc.) without decoding their values.
///
/// `field_id` is the stable catalog field_id (truncated to u32 — the engine
/// asserts catalog IDs fit). It is invariant across `rename_field`, so a zone
/// map written before a rename is still readable after.
pub type ZoneFieldExtractor = Arc<dyn Fn(&[u8], &[u8]) -> Vec<(u32, [u8; 8])> + Send + Sync>;

/// Writer-side accumulator for per-block zone bounds. Updated entry-by-entry
/// inside `SstWriter::add`; serialized as the zone-map block in `finish()`.
///
/// Only blocks that actually saw a zone-field value are held (in block
/// order); every other block up to `num_blocks` is written as the no-data
/// sentinel at serialization time. In an SST of mostly edges, index entries
/// and postings, ~1% of blocks carry bounds, so a dense in-memory table
/// (a `Vec` per block) cost ~50x what the data needs — tens of MB per
/// compaction of a large file. The serialized bytes are unchanged.
pub struct ZoneBuilder {
    /// field_id → index in `field_ids`. Built lazily as new fields appear.
    field_id_to_idx: HashMap<u32, usize>,
    /// Insertion-ordered list of field IDs; defines column order in the
    /// per-block bounds tables.
    field_ids: Vec<u32>,
    /// One past the highest block index recorded (the serialized block count).
    num_blocks: usize,
    /// `(block_idx, per-field (min, max))` for blocks with any recorded value,
    /// ascending by block. A row may be shorter than `field_ids` (fields first
    /// seen later); missing columns are the `(u64::MAX, u64::MIN)` sentinel.
    rows: Vec<(usize, Vec<(u64, u64)>)>,
}

const NO_DATA: (u64, u64) = (u64::MAX, u64::MIN);

impl ZoneBuilder {
    pub fn new() -> Self {
        Self {
            field_id_to_idx: HashMap::new(),
            field_ids: Vec::new(),
            num_blocks: 0,
            rows: Vec::new(),
        }
    }

    /// Record an entry's zone-field values into the block at `block_idx`.
    /// Fields not present in `zone_fields` simply leave their (min, max) at
    /// the previous value (or the no-data sentinel if this is the first entry
    /// in the block).
    pub fn record(&mut self, block_idx: usize, zone_fields: &[(u32, [u8; 8])]) {
        self.num_blocks = self.num_blocks.max(block_idx + 1);
        if zone_fields.is_empty() {
            return;
        }
        // The writer records blocks in order, so the row is almost always the
        // last one or a new last one.
        let row_pos = match self.rows.last() {
            Some((b, _)) if *b == block_idx => self.rows.len() - 1,
            Some((b, _)) if *b > block_idx => {
                match self.rows.binary_search_by_key(&block_idx, |(b, _)| *b) {
                    Ok(i) => i,
                    Err(i) => {
                        self.rows.insert(i, (block_idx, Vec::new()));
                        i
                    }
                }
            }
            _ => {
                self.rows.push((block_idx, Vec::new()));
                self.rows.len() - 1
            }
        };
        for &(field_id, bytes) in zone_fields {
            // Allocate a column for first-time field IDs.
            let field_idx = match self.field_id_to_idx.get(&field_id) {
                Some(&idx) => idx,
                None => {
                    let idx = self.field_ids.len();
                    self.field_ids.push(field_id);
                    self.field_id_to_idx.insert(field_id, idx);
                    idx
                }
            };
            let row = &mut self.rows[row_pos].1;
            if row.len() <= field_idx {
                row.resize(field_idx + 1, NO_DATA);
            }
            let value = u64::from_be_bytes(bytes);
            let (cur_min, cur_max) = row[field_idx];
            row[field_idx] = (cur_min.min(value), cur_max.max(value));
        }
    }

    /// `(min, max)` of every field (in column order) for `block_idx`; the
    /// sentinel for fields without a value there.
    pub fn block_bounds(&self, block_idx: usize) -> Vec<(u64, u64)> {
        let mut out = vec![NO_DATA; self.field_ids.len()];
        if let Ok(i) = self.rows.binary_search_by_key(&block_idx, |(b, _)| *b) {
            out[..self.rows[i].1.len()].copy_from_slice(&self.rows[i].1);
        }
        out
    }

    /// Serialize the accumulated zone map, streaming it in small chunks (the
    /// dense form can be tens of MB). Format is documented at the top of this
    /// module.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        let num_fields = self.field_ids.len();
        let mut buf = BytesMut::with_capacity(64 * 1024);
        buf.put_u32(self.num_blocks as u32);
        buf.put_u32(num_fields as u32);
        for &id in &self.field_ids {
            buf.put_u32(id);
        }
        let mut rows = self.rows.iter().peekable();
        for block in 0..self.num_blocks {
            let row = match rows.peek() {
                Some((b, r)) if *b == block => {
                    rows.next();
                    r.as_slice()
                }
                _ => &[],
            };
            for f in 0..num_fields {
                let (min, max) = row.get(f).copied().unwrap_or(NO_DATA);
                buf.put_u64(min);
                buf.put_u64(max);
            }
            if buf.len() >= 60 * 1024 {
                w.write_all(&buf)?;
                buf.clear();
            }
        }
        w.write_all(&buf)
    }
}

impl Default for ZoneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader-side view of a serialized zone map. Lookup is `(block_idx,
/// field_id) -> Option<(min, max)>` in O(1) via the id → column index map.
///
/// The bounds are read in place from `raw` — for an open SST, a slice of the
/// mapped file — rather than decoded into a heap array at open: a large file
/// has a bound pair per field for every 16-entry block (tens of MB), while a
/// query touches only the blocks it prunes.
#[derive(Debug, Clone, Default)]
pub struct ZoneMap {
    field_id_to_idx: HashMap<u32, usize>,
    num_fields: usize,
    /// Flat big-endian u64 array, exactly as written:
    /// `[block_idx * num_fields * 2 + field_idx * 2 + {0=min, 1=max}]`.
    raw: bytes::Bytes,
    /// Number of u64 bounds in `raw`.
    num_bounds: usize,
}

impl ZoneMap {
    /// Parse a serialized zone map from a reader (copies it once).
    pub fn read_from(r: &mut dyn Read) -> Result<Self> {
        let mut all = Vec::new();
        r.read_to_end(&mut all).map_err(io_err)?;
        Self::from_bytes(bytes::Bytes::from(all))
    }

    /// View a serialized zone map held in `data` (e.g. a slice of a mapped
    /// SST) without copying its bounds. Trailing bytes past the bounds are
    /// ignored, as `read_from` always did.
    pub fn from_bytes(data: bytes::Bytes) -> Result<Self> {
        let truncated = || Error::SstCorrupted("zone map truncated".into());
        if data.len() < 8 {
            return Err(truncated());
        }
        let num_blocks = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
        let num_fields = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let header = num_fields.checked_mul(4).and_then(|n| n.checked_add(8)).ok_or_else(truncated)?;
        if data.len() < header {
            return Err(truncated());
        }
        let mut field_id_to_idx = HashMap::with_capacity(num_fields);
        for idx in 0..num_fields {
            let at = 8 + idx * 4;
            let field_id = u32::from_be_bytes(data[at..at + 4].try_into().unwrap());
            field_id_to_idx.insert(field_id, idx);
        }
        // num_blocks × num_fields × 2 u64s.
        let num_bounds = num_blocks
            .checked_mul(num_fields)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(truncated)?;
        let end = num_bounds
            .checked_mul(8)
            .and_then(|n| n.checked_add(header))
            .ok_or_else(truncated)?;
        if data.len() < end {
            return Err(truncated());
        }
        Ok(Self {
            field_id_to_idx,
            num_fields,
            raw: data.slice(header..end),
            num_bounds,
        })
    }

    fn bound(&self, i: usize) -> u64 {
        u64::from_be_bytes(self.raw[i * 8..i * 8 + 8].try_into().unwrap())
    }

    /// Number of blocks this map covers. Should equal the SST's sparse-index
    /// length; mismatches indicate format corruption (caller validates).
    pub fn num_blocks(&self) -> usize {
        if self.num_fields == 0 {
            // Zone maps with zero fields don't carry per-block data — treat
            // as covering whatever the SST has (always returns true on check).
            // There are no bounds when `num_fields == 0`, so this is 0.
            return self.num_bounds;
        }
        self.num_bounds / (self.num_fields * 2)
    }

    /// Returns `(min, max)` for `field_id` at `block_idx`, or `None` if
    /// the field isn't tracked in this zone map. The caller treats `None`
    /// (and `min > max`) as "no info — must scan the block."
    pub fn bounds(&self, block_idx: usize, field_id: u32) -> Option<(u64, u64)> {
        let field_idx = *self.field_id_to_idx.get(&field_id)?;
        let pos = (block_idx * self.num_fields + field_idx) * 2;
        if pos + 1 >= self.num_bounds {
            return None;
        }
        Some((self.bound(pos), self.bound(pos + 1)))
    }

    /// `true` iff the block could contain a key whose `field_id` value
    /// satisfies `op` against `target_value_bytes` (encoded the same way).
    /// Conservative — returns `true` whenever bounds are unknown or the no-
    /// data sentinel is in effect.
    pub fn block_could_match(
        &self,
        block_idx: usize,
        field_id: u32,
        op: CompareOp,
        target: u64,
    ) -> bool {
        let Some((min, max)) = self.bounds(block_idx, field_id) else {
            return true; // field not tracked — can't skip
        };
        if min > max {
            return true; // no entry for this field in the block — can't skip
        }
        match op {
            // For value-bytes encoded so that u64 order = numeric order,
            // each predicate translates to a half-open interval test on
            // [min, max].
            CompareOp::Eq => target >= min && target <= max,
            CompareOp::Ne => true, // can't skip: at least one entry might differ
            CompareOp::Lt => min < target,
            CompareOp::Le => min <= target,
            CompareOp::Gt => max > target,
            CompareOp::Ge => max >= target,
        }
    }
}

/// Comparison operator for filter predicates. Mirrors the query AST but lives
/// in the storage layer so SST scans don't depend on query crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A pushed-down predicate against a single zone-mapped field.
#[derive(Debug, Clone, Copy)]
pub struct FieldPredicate {
    pub field_id: u32,
    pub op: CompareOp,
    /// Target value, encoded the same way as on-disk zone bounds (u64 from
    /// big-endian bytes with the engine's encoding rules applied).
    pub target: u64,
}

fn io_err(e: io::Error) -> Error {
    Error::SstCorrupted(format!("zone map: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic field IDs for tests. In production these come from the
    // catalog (`c:F:` rows); here we use small constants for readability.
    const YEAR_ID: u32 = 1;
    const AGE_ID: u32 = 2;

    #[test]
    fn builder_record_grows_blocks_and_fields() {
        let mut b = ZoneBuilder::new();
        b.record(0, &[(YEAR_ID, 100u64.to_be_bytes())]);
        b.record(0, &[(YEAR_ID, 50u64.to_be_bytes())]);
        b.record(1, &[(AGE_ID, 30u64.to_be_bytes()), (YEAR_ID, 200u64.to_be_bytes())]);

        assert_eq!(b.num_blocks, 2);
        assert_eq!(b.field_ids.len(), 2);
        let year_idx = b.field_id_to_idx[&YEAR_ID];
        let age_idx = b.field_id_to_idx[&AGE_ID];
        assert_eq!(b.block_bounds(0)[year_idx], (50, 100));
        // Block 0 never saw age — sentinel still in place.
        assert_eq!(b.block_bounds(0)[age_idx], (u64::MAX, u64::MIN));
        // Block 1 has year=200, age=30 — both singletons.
        assert_eq!(b.block_bounds(1)[year_idx], (200, 200));
        assert_eq!(b.block_bounds(1)[age_idx], (30, 30));
    }

    /// The sparse builder must serialize byte-for-byte what the original dense
    /// builder (reproduced here verbatim in behaviour) wrote, over random
    /// histories: blocks skipped, fields first seen late, blocks recorded with
    /// no fields, out-of-order block indices.
    #[test]
    fn sparse_builder_writes_the_dense_builders_bytes() {
        struct Dense {
            ids: Vec<u32>,
            idx: HashMap<u32, usize>,
            bounds: Vec<Vec<(u64, u64)>>,
        }
        impl Dense {
            fn record(&mut self, block_idx: usize, zone_fields: &[(u32, [u8; 8])]) {
                while self.bounds.len() <= block_idx {
                    self.bounds.push(vec![(u64::MAX, u64::MIN); self.ids.len()]);
                }
                for &(field_id, bytes) in zone_fields {
                    let fi = match self.idx.get(&field_id) {
                        Some(&i) => i,
                        None => {
                            let i = self.ids.len();
                            self.ids.push(field_id);
                            self.idx.insert(field_id, i);
                            for b in &mut self.bounds {
                                b.push((u64::MAX, u64::MIN));
                            }
                            i
                        }
                    };
                    let v = u64::from_be_bytes(bytes);
                    let (lo, hi) = self.bounds[block_idx][fi];
                    self.bounds[block_idx][fi] = (lo.min(v), hi.max(v));
                }
            }
            fn bytes(&self) -> Vec<u8> {
                let mut out = Vec::new();
                out.extend_from_slice(&(self.bounds.len() as u32).to_be_bytes());
                out.extend_from_slice(&(self.ids.len() as u32).to_be_bytes());
                for id in &self.ids {
                    out.extend_from_slice(&id.to_be_bytes());
                }
                for b in &self.bounds {
                    for (lo, hi) in b {
                        out.extend_from_slice(&lo.to_be_bytes());
                        out.extend_from_slice(&hi.to_be_bytes());
                    }
                }
                out
            }
        }
        for seed in 1..=200u64 {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut next = move |n: u64| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng % n
            };
            let mut dense = Dense { ids: Vec::new(), idx: HashMap::new(), bounds: Vec::new() };
            let mut sparse = ZoneBuilder::new();
            let mut block = 0usize;
            for _ in 0..next(300) {
                block += next(40) as usize; // gaps, often long
                let b = if next(10) == 0 { block.saturating_sub(next(50) as usize) } else { block };
                let fields: Vec<(u32, [u8; 8])> = (0..next(3))
                    .map(|_| (next(5) as u32 + 1, next(1000).to_be_bytes()))
                    .collect();
                dense.record(b, &fields);
                sparse.record(b, &fields);
            }
            let mut got = Vec::new();
            sparse.write_to(&mut got).unwrap();
            assert_eq!(got, dense.bytes(), "seed {seed}");
        }
    }

    #[test]
    fn builder_roundtrip_via_zonemap() {
        let mut b = ZoneBuilder::new();
        b.record(0, &[(YEAR_ID, 1900u64.to_be_bytes())]);
        b.record(0, &[(YEAR_ID, 1950u64.to_be_bytes())]);
        b.record(1, &[(YEAR_ID, 2000u64.to_be_bytes())]);

        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let zmap = ZoneMap::read_from(&mut &buf[..]).unwrap();

        assert_eq!(zmap.num_blocks(), 2);
        assert_eq!(zmap.bounds(0, YEAR_ID), Some((1900, 1950)));
        assert_eq!(zmap.bounds(1, YEAR_ID), Some((2000, 2000)));
        // Unrelated field id: not tracked.
        assert_eq!(zmap.bounds(0, 0xdeadbeef), None);
    }

    #[test]
    fn from_bytes_reads_bounds_in_place_and_rejects_truncation() {
        let mut b = ZoneBuilder::new();
        b.record(0, &[(YEAR_ID, 1900u64.to_be_bytes()), (AGE_ID, 30u64.to_be_bytes())]);
        b.record(2, &[(YEAR_ID, 2020u64.to_be_bytes())]);
        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let full = bytes::Bytes::from(buf);
        let z = ZoneMap::from_bytes(full.clone()).unwrap();
        assert_eq!(z.num_blocks(), 3);
        assert_eq!(z.bounds(0, YEAR_ID), Some((1900, 1900)));
        assert_eq!(z.bounds(0, AGE_ID), Some((30, 30)));
        assert_eq!(z.bounds(2, YEAR_ID), Some((2020, 2020)));
        assert_eq!(z.bounds(3, YEAR_ID), None, "past the last block");
        // Same answers as the copying reader.
        let copied = ZoneMap::read_from(&mut &full[..]).unwrap();
        for blk in 0..4 {
            for f in [YEAR_ID, AGE_ID, 7] {
                assert_eq!(z.bounds(blk, f), copied.bounds(blk, f));
            }
        }
        for cut in [0, 7, 8 + 4, full.len() - 1] {
            assert!(ZoneMap::from_bytes(full.slice(..cut)).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn block_could_match_skips_outside_range() {
        let mut b = ZoneBuilder::new();
        // Block 0: years 1990..2000. Block 1: years 2010..2020.
        b.record(0, &[(YEAR_ID, 1990u64.to_be_bytes())]);
        b.record(0, &[(YEAR_ID, 2000u64.to_be_bytes())]);
        b.record(1, &[(YEAR_ID, 2010u64.to_be_bytes())]);
        b.record(1, &[(YEAR_ID, 2020u64.to_be_bytes())]);

        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let zmap = ZoneMap::read_from(&mut &buf[..]).unwrap();

        // year > 2005: block 0 must skip (max 2000), block 1 must scan.
        assert!(!zmap.block_could_match(0, YEAR_ID, CompareOp::Gt, 2005));
        assert!(zmap.block_could_match(1, YEAR_ID, CompareOp::Gt, 2005));

        // year == 2000: block 0 in range, block 1 not.
        assert!(zmap.block_could_match(0, YEAR_ID, CompareOp::Eq, 2000));
        assert!(!zmap.block_could_match(1, YEAR_ID, CompareOp::Eq, 2000));

        // year != X: never skip.
        assert!(zmap.block_could_match(0, YEAR_ID, CompareOp::Ne, 2000));

        // year < 1995: block 0 in range (min 1990), block 1 not.
        assert!(zmap.block_could_match(0, YEAR_ID, CompareOp::Lt, 1995));
        assert!(!zmap.block_could_match(1, YEAR_ID, CompareOp::Lt, 1995));
    }

    #[test]
    fn block_could_match_unknown_field_returns_true() {
        let b = ZoneBuilder::new();
        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let zmap = ZoneMap::read_from(&mut &buf[..]).unwrap();
        // No fields tracked — caller must scan.
        assert!(zmap.block_could_match(0, YEAR_ID, CompareOp::Gt, 0));
    }

    #[test]
    fn block_could_match_no_data_sentinel_returns_true() {
        let mut b = ZoneBuilder::new();
        // Field tracked in block 1 only.
        b.record(0, &[]); // entry in block 0 has no year
        b.record(1, &[(YEAR_ID, 2000u64.to_be_bytes())]);

        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let zmap = ZoneMap::read_from(&mut &buf[..]).unwrap();

        // Block 0 has no year — must scan (sentinel min > max).
        assert!(zmap.block_could_match(0, YEAR_ID, CompareOp::Gt, 0));
        // Block 1 has year=2000 — gt 1900 yes, gt 2500 no.
        assert!(zmap.block_could_match(1, YEAR_ID, CompareOp::Gt, 1900));
        assert!(!zmap.block_could_match(1, YEAR_ID, CompareOp::Gt, 2500));
    }
}
