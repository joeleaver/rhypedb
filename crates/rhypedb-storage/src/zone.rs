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
pub struct ZoneBuilder {
    /// field_id → index in `field_ids`. Built lazily as new fields appear.
    field_id_to_idx: HashMap<u32, usize>,
    /// Insertion-ordered list of field IDs; defines column order in the
    /// per-block bounds tables.
    field_ids: Vec<u32>,
    /// Per-block per-field (min, max). `bounds[block_idx][field_idx]`.
    /// `(u64::MAX, u64::MIN)` is the "no entry with this field yet" sentinel.
    bounds: Vec<Vec<(u64, u64)>>,
}

impl ZoneBuilder {
    pub fn new() -> Self {
        Self {
            field_id_to_idx: HashMap::new(),
            field_ids: Vec::new(),
            bounds: Vec::new(),
        }
    }

    /// Record an entry's zone-field values into the block at `block_idx`.
    /// Fields not present in `zone_fields` simply leave their (min, max) at
    /// the previous value (or the no-data sentinel if this is the first entry
    /// in the block).
    pub fn record(&mut self, block_idx: usize, zone_fields: &[(u32, [u8; 8])]) {
        // Grow the per-block bounds list to cover `block_idx`.
        while self.bounds.len() <= block_idx {
            self.bounds
                .push(vec![(u64::MAX, u64::MIN); self.field_ids.len()]);
        }
        for &(field_id, bytes) in zone_fields {
            // Allocate a column for first-time field IDs; pad existing
            // blocks with the no-data sentinel.
            let field_idx = match self.field_id_to_idx.get(&field_id) {
                Some(&idx) => idx,
                None => {
                    let idx = self.field_ids.len();
                    self.field_ids.push(field_id);
                    self.field_id_to_idx.insert(field_id, idx);
                    for block in &mut self.bounds {
                        block.push((u64::MAX, u64::MIN));
                    }
                    idx
                }
            };
            let value = u64::from_be_bytes(bytes);
            let (cur_min, cur_max) = self.bounds[block_idx][field_idx];
            self.bounds[block_idx][field_idx] = (cur_min.min(value), cur_max.max(value));
        }
    }

    /// Serialize the accumulated zone map. Format is documented at the top
    /// of this module.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        let num_blocks = self.bounds.len() as u32;
        let num_fields = self.field_ids.len() as u32;

        let mut buf = BytesMut::with_capacity(
            8 + (num_fields as usize) * 4 + (num_blocks as usize) * (num_fields as usize) * 16,
        );
        buf.put_u32(num_blocks);
        buf.put_u32(num_fields);
        for &id in &self.field_ids {
            buf.put_u32(id);
        }
        for block in &self.bounds {
            for &(min, max) in block {
                buf.put_u64(min);
                buf.put_u64(max);
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
#[derive(Debug, Clone, Default)]
pub struct ZoneMap {
    field_id_to_idx: HashMap<u32, usize>,
    num_fields: usize,
    /// Per-block per-field flat array: `bounds[block_idx * num_fields * 2 + field_idx * 2 + {0=min, 1=max}]`.
    /// Flat representation is more cache-friendly than `Vec<Vec<(u64, u64)>>`.
    bounds: Vec<u64>,
}

impl ZoneMap {
    pub fn read_from(r: &mut dyn Read) -> Result<Self> {
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4).map_err(io_err)?;
        let num_blocks = u32::from_be_bytes(buf4) as usize;
        r.read_exact(&mut buf4).map_err(io_err)?;
        let num_fields = u32::from_be_bytes(buf4) as usize;

        let mut field_id_to_idx = HashMap::with_capacity(num_fields);
        for idx in 0..num_fields {
            r.read_exact(&mut buf4).map_err(io_err)?;
            let field_id = u32::from_be_bytes(buf4);
            field_id_to_idx.insert(field_id, idx);
        }

        // num_blocks × num_fields × 2 u64s.
        let bound_count = num_blocks * num_fields * 2;
        let mut bounds = Vec::with_capacity(bound_count);
        let mut buf8 = [0u8; 8];
        for _ in 0..bound_count {
            r.read_exact(&mut buf8).map_err(io_err)?;
            bounds.push(u64::from_be_bytes(buf8));
        }

        Ok(Self {
            field_id_to_idx,
            num_fields,
            bounds,
        })
    }

    /// Number of blocks this map covers. Should equal the SST's sparse-index
    /// length; mismatches indicate format corruption (caller validates).
    pub fn num_blocks(&self) -> usize {
        if self.num_fields == 0 {
            // Zone maps with zero fields don't carry per-block data — treat
            // as covering whatever the SST has (always returns true on check).
            // `bounds` is empty when `num_fields == 0` (it was built with
            // `num_blocks * num_fields * 2 == 0` capacity), so this is 0.
            return self.bounds.len();
        }
        self.bounds.len() / (self.num_fields * 2)
    }

    /// Returns `(min, max)` for `field_id` at `block_idx`, or `None` if
    /// the field isn't tracked in this zone map. The caller treats `None`
    /// (and `min > max`) as "no info — must scan the block."
    pub fn bounds(&self, block_idx: usize, field_id: u32) -> Option<(u64, u64)> {
        let field_idx = *self.field_id_to_idx.get(&field_id)?;
        let pos = (block_idx * self.num_fields + field_idx) * 2;
        if pos + 1 >= self.bounds.len() {
            return None;
        }
        Some((self.bounds[pos], self.bounds[pos + 1]))
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

        assert_eq!(b.bounds.len(), 2);
        assert_eq!(b.field_ids.len(), 2);
        let year_idx = b.field_id_to_idx[&YEAR_ID];
        let age_idx = b.field_id_to_idx[&AGE_ID];
        assert_eq!(b.bounds[0][year_idx], (50, 100));
        // Block 0 never saw age — sentinel still in place.
        assert_eq!(b.bounds[0][age_idx], (u64::MAX, u64::MIN));
        // Block 1 has year=200, age=30 — both singletons.
        assert_eq!(b.bounds[1][year_idx], (200, 200));
        assert_eq!(b.bounds[1][age_idx], (30, 30));
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
