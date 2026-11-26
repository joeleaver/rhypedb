//! Bounded (`LIMIT`-style) and resumable range scans over an [`LsmTree`].
//!
//! Every scan here reads each layer (SSTs oldest first, then the immutable
//! memtables, then the active memtable) for only a bounded window of user keys
//! and merges the windows newest-wins. The windows count RAW keys, tombstones
//! included, so the merge is only trustworthy for the globally-smallest
//! `window` raw keys (see [`LsmTree::scan_chunk_raw`]). [`ChunkScan`] exposes
//! exactly that; the `*_limited` scans drive it until they hold enough LIVE
//! keys, so deleted-but-uncompacted rows never count toward a caller's limit.
//!
//! All of them need a PREFIX-FREE keyspace under `prefix`; see
//! [`crate::key::debug_assert_prefix_free`].

use bytes::Bytes;

use super::LsmTree;
use crate::Result;

/// One bounded, tombstone-aware step of a resumable range scan
/// (`LsmTree::scan_chunk_raw`). Built for chunked work that must walk an
/// entire key range exactly once across crashes — e.g. shadow-field
/// field-type migrations.
#[derive(Debug, Clone)]
pub struct ChunkScan {
    /// Live (non-tombstone) entries in this chunk, ascending by key. Drawn
    /// from the globally-smallest `max_distinct` RAW keys at/after the start
    /// point; may be shorter than `max_distinct` (or empty) when the chunk
    /// straddles a run of tombstones.
    pub live: Vec<(Bytes, Bytes)>,
    /// Highest RAW user key visited this chunk, INCLUDING tombstones. `None`
    /// iff no key — live or tombstoned — exists at/after the start point
    /// within `prefix` (the range is exhausted). A resumable caller advances
    /// its cursor to this key so a tombstone run longer than `max_distinct`
    /// cannot strand live keys beyond it (which a `scan_from_at_limited`
    /// caller, seeing only live keys, would do).
    pub high_water: Option<Bytes>,
    /// `true` when more keys may exist strictly past `high_water`: either a
    /// source saturated its `max_distinct` window, or the merged set held
    /// more than `max_distinct` distinct keys. `false` PROVES the range ends
    /// at `high_water`. NEVER infer end-of-range from `live.len() <
    /// max_distinct`; only `!more` (or `high_water == None`) is sound.
    pub more: bool,
    /// RAW keys taken this chunk, live AND tombstoned (`>= live.len()`). A
    /// caller that budgets work must charge this, not `live.len()`: a long
    /// tombstone run is real merge work that yields no live rows.
    pub visited: usize,
}
impl LsmTree {
    /// Bounded prefix scan: the first `max_live` LIVE user keys under
    /// `prefix` (each at its latest version visible at `version`), ascending.
    /// Built for `LIMIT N` push-down on secondary indexes, whose keys sort by
    /// (encoded value, object id).
    ///
    /// Deleted-but-uncompacted keys never count toward `max_live`: the result
    /// holds exactly `min(max_live, live keys in range)` entries, the same as
    /// the first `max_live` of an unbounded [`LsmTree::scan_prefix_at`]. The
    /// work is proportional to `max_live` plus the tombstones walked past,
    /// never the whole prefix (see [`LsmTree::scan_from_at_limited`]).
    pub fn scan_prefix_at_limited(
        &self,
        version: u64,
        prefix: &[u8],
        max_live: usize,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        self.scan_from_at_limited(version, prefix, prefix, max_live)
    }

    /// Seek-then-scan bounded range scan: the first `max_live` LIVE user keys
    /// at or after `start_user_key` (inclusive) within `prefix`, ascending.
    /// Tombstones never count toward `max_live`, as in
    /// [`LsmTree::scan_prefix_at_limited`].
    ///
    /// It walks [`LsmTree::scan_chunk_raw`] windows from the seek point,
    /// resuming strictly after each window's tombstone-inclusive high-water
    /// key, until it holds `max_live` live keys or the range is exhausted.
    /// The first window is `max_live` keys wide, so a range with no
    /// tombstones costs the single window the old one-shot scan did. A
    /// resumed window asks for the live keys still missing, widened to twice
    /// the tombstones the previous window met: a long tombstone run grows the
    /// windows geometrically, so walking it costs O(run) work in
    /// O(log run) steps rather than one tiny step per tombstone.
    pub fn scan_from_at_limited(
        &self,
        version: u64,
        prefix: &[u8],
        start_user_key: &[u8],
        max_live: usize,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        let mut out: Vec<(Bytes, Bytes)> = Vec::new();
        if max_live == 0 {
            return Ok(out);
        }
        let mut start = Bytes::copy_from_slice(start_user_key);
        let mut window = max_live;
        loop {
            let chunk = self.scan_chunk_raw(version, prefix, &start, window)?;
            let met_tombstones = chunk.visited - chunk.live.len();
            let missing = max_live - out.len();
            out.extend(chunk.live.into_iter().take(missing));
            let missing = max_live - out.len();
            let (Some(high_water), true) = (chunk.high_water, chunk.more) else {
                return Ok(out);
            };
            if missing == 0 {
                return Ok(out);
            }
            start = successor(&high_water);
            window = missing.max(met_tombstones.saturating_mul(2));
        }
    }

    /// Tombstone-aware bounded range scan for resumable chunked work.
    ///
    /// Unlike the `*_limited` scans, which return only live entries (so their
    /// result can't tell a caller where a tombstone run ended), this returns the live entries PLUS a tombstone-inclusive
    /// `high_water` and a sound `more` flag. A caller advances its cursor to
    /// `high_water` each step and stops only when `!more` (or `high_water`
    /// is `None`). See [`ChunkScan`].
    ///
    /// `start_user_key` is inclusive (same seek semantics as
    /// `scan_from_at_limited`); pass the key strictly after the last cursor
    /// to avoid re-visiting it.
    ///
    /// Requires a PREFIX-FREE keyspace under `prefix` (no user key a strict
    /// prefix of another): fixed-length ids or escaped-and-terminated values.
    /// See [`crate::key::debug_assert_prefix_free`] for why, and for the
    /// debug-build guard that enforces it.
    pub fn scan_chunk_raw(
        &self,
        version: u64,
        prefix: &[u8],
        start_user_key: &[u8],
        max_distinct: usize,
    ) -> Result<ChunkScan> {
        if max_distinct == 0 {
            return Ok(ChunkScan {
                live: Vec::new(),
                high_water: None,
                more: false,
                visited: 0,
            });
        }
        let mut merged: std::collections::BTreeMap<Bytes, Option<Bytes>> =
            std::collections::BTreeMap::new();
        // A source is "saturated" when it returns a full `max_distinct`
        // window — it may hold more keys past the window we can't see yet.
        let mut source_saturated = false;

        let ssts = self.sst_files.read();
        for sst in ssts.iter() {
            let rows = sst.scan_from_max(prefix, start_user_key, version, max_distinct);
            source_saturated |= rows.len() == max_distinct;
            for (key, value) in rows {
                merged.insert(key, value);
            }
        }
        drop(ssts);

        let immutables = self.immutable_memtables.read().clone();
        for mt in immutables.iter() {
            let rows = mt.scan_from_max(prefix, start_user_key, version, max_distinct);
            source_saturated |= rows.len() == max_distinct;
            for (key, value) in rows {
                merged.insert(key, value);
            }
        }

        let active = self.active_memtable.read().clone();
        let rows = active.scan_from_max(prefix, start_user_key, version, max_distinct);
        source_saturated |= rows.len() == max_distinct;
        for (key, value) in rows {
            merged.insert(key, value);
        }

        // More keys may lie past `high_water` if either a source saturated
        // its window OR the merged (deduped) set already holds more than we
        // will take — the latter catches the case where several unsaturated
        // sources together overflow `max_distinct` distinct keys (a pure
        // any-source-saturated test would falsely report exhaustion here and
        // strand the overflow).
        let more = source_saturated || merged.len() > max_distinct;

        // Take the globally-smallest `max_distinct` RAW keys (tombstones
        // included). Each source contributed its own smallest `max_distinct`
        // keys, so every key in the true global smallest-`max_distinct` is
        // present: a key with < `max_distinct` keys below it across all
        // sources has < `max_distinct` keys below it in whichever source
        // holds it, so it sits inside that source's returned window.
        let mut high_water: Option<Bytes> = None;
        let mut live: Vec<(Bytes, Bytes)> = Vec::new();
        let mut visited = 0usize;
        for (k, v) in merged.into_iter().take(max_distinct) {
            visited += 1;
            match v {
                Some(val) => {
                    high_water = Some(k.clone());
                    live.push((k, val));
                }
                None => high_water = Some(k),
            }
        }

        Ok(ChunkScan {
            live,
            high_water,
            more,
            visited,
        })
    }
}

/// The smallest user key strictly greater than `key` (`key ‖ 0x00`). In a
/// prefix-free keyspace no user key lies between the two, so resuming an
/// inclusive seek here skips exactly `key`.
fn successor(key: &[u8]) -> Bytes {
    let mut next = Vec::with_capacity(key.len() + 1);
    next.extend_from_slice(key);
    next.push(0);
    Bytes::from(next)
}

#[cfg(test)]
mod tests;
