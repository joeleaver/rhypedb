use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;

use crate::memtable::MemTable;
use crate::mvcc::{Transaction, TransactionManager};
use crate::sst::{SstReader, SstWriter};
use crate::wal::{RecordType, Wal};
use crate::Result;

const DEFAULT_MEMTABLE_FLUSH_SIZE: usize = 4 * 1024 * 1024; // 4MB
const DEFAULT_COMPACT_TRIGGER_SSTS: usize = 4;

/// Configuration for the LSM-tree storage engine.
pub struct LsmConfig {
    pub data_dir: PathBuf,
    pub memtable_flush_size: usize,
    /// Auto-compact when the SST count reaches this many. Compaction merges
    /// all SSTs into a single file; reduces read amplification (fewer bloom
    /// checks, fewer cross-layer BTreeMap merges per scan).
    pub compact_trigger_ssts: usize,
    /// Optional closure that pulls zone-mapped integer field values out of
    /// each SST entry's value bytes. Set by the engine at construction time
    /// (it knows the schema and can decode FieldMaps). When `None`, SSTs are
    /// written with an empty zone map (v4 format, zero overhead). When set,
    /// every flushed SST captures per-block min/max bounds for opted-in
    /// integer fields, enabling `scan_prefix_filtered_at` to skip blocks.
    pub zone_extractor: Option<crate::zone::ZoneFieldExtractor>,
    /// `true` (default): every `LsmTree::commit` calls `fsync(2)` on the WAL,
    /// guaranteeing the writes survive a power loss. `false`: skip the
    /// fsync — the kernel still gets the bytes via `write(2)` so a clean
    /// process crash is recoverable from the WAL, but a power loss can
    /// silently drop the last N writes. Matches Postgres's `fsync=off`
    /// mode for apples-to-apples benchmarking (and is occasionally useful
    /// for bulk imports where the operator accepts the risk).
    pub sync_on_commit: bool,
    /// `true` (default): auto-triggered compaction (the threshold crossing
    /// inside `maybe_flush`) is signalled to a background worker thread,
    /// letting the writer that triggered the flush return immediately.
    /// `false`: compact inline on the writer that crossed the threshold,
    /// blocking the write until the merge completes — the legacy behavior.
    /// Explicit `LsmTree::compact()` calls are always synchronous either
    /// way (they serialize against the worker via the compaction mutex).
    pub background_compaction: bool,
}

impl LsmConfig {
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            memtable_flush_size: DEFAULT_MEMTABLE_FLUSH_SIZE,
            compact_trigger_ssts: DEFAULT_COMPACT_TRIGGER_SSTS,
            zone_extractor: None,
            sync_on_commit: true,
            background_compaction: true,
        }
    }
}

/// Compaction worker coordination — shared between the LsmTree (which
/// signals new work) and the worker thread (which waits on the condvar
/// and runs `compact_inner`).
///
/// Three flags drive the worker:
/// * `pending` — `maybe_flush` set this when crossing the SST threshold;
///   worker clears it when it picks up the job.
/// * `running` — set while `compact_inner` is in flight; cleared after.
///   Lets `wait_for_compaction` block until idle.
/// * `shutdown` — set by `Drop`; tells the worker to exit its loop.
#[derive(Default)]
struct CompactionState {
    pending: bool,
    running: bool,
    shutdown: bool,
}

/// LSM-tree storage engine.
///
/// Provides versioned key-value storage with:
/// - Write-ahead log for durability
/// - In-memory skip-list memtable for writes
/// - Sorted string table (SST) files on disk
/// - MVCC transaction manager for snapshot isolation
/// - Optional background compaction worker (`background_compaction` config)
pub struct LsmTree {
    config: LsmConfig,
    active_memtable: Arc<RwLock<Arc<MemTable>>>,
    immutable_memtables: Arc<RwLock<Vec<Arc<MemTable>>>>,
    sst_files: Arc<RwLock<Vec<SstReader>>>,
    wal: Arc<parking_lot::Mutex<Wal>>,
    txn_manager: Arc<TransactionManager>,
    /// Coordinates concurrent writers against `flush()`. Every writer holds
    /// `.read()` across its `[WAL append + memtable write]` pair; `flush()`
    /// holds `.write()` across its `[rotate memtable + write SST + truncate
    /// WAL]` pass. This makes a write's WAL record and memtable entry land in
    /// ONE flush generation: without it, a flush rotating the memtable between
    /// a concurrent writer's WAL append and its memtable write strands that
    /// write's row in the new (not-yet-flushed) memtable while its WAL record
    /// is truncated — silently losing committed data across a reopen. Writers
    /// still run concurrently with each other (shared read lock); only `flush`
    /// excludes them, for its duration. (Perf note: `flush` holds the write
    /// lock across its SST write I/O, a brief periodic write-stall; a future
    /// segmented-WAL change could move that I/O out of the lock.)
    flush_lock: parking_lot::RwLock<()>,
    next_sst_id: std::sync::atomic::AtomicU64,
    /// Serializes simultaneous compaction attempts. The background worker
    /// AND any caller of public `compact()` both acquire this — guarantees
    /// at most one merge is reading SSTs / writing the temp SST / racing
    /// for the final `sst_files.write()` swap.
    compaction_mutex: parking_lot::Mutex<()>,
    /// Worker coordination. Wrapped in `Arc` so the worker thread holds a
    /// clone independent of LsmTree's lifetime (it's the channel through
    /// which `Drop` signals shutdown).
    compaction_state:
        Arc<(parking_lot::Mutex<CompactionState>, parking_lot::Condvar)>,
    /// Background-worker join handle. Taken on drop.
    compaction_handle:
        parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

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
}

impl LsmTree {
    /// Open or create an LSM-tree at the given directory.
    ///
    /// Recovery procedure:
    /// 1. Discover and open existing SST files, tracking max version
    /// 2. Replay WAL records into a fresh memtable, tracking max version
    /// 3. Initialize the transaction manager at the recovered max version
    /// 4. Spawn the compaction worker (unless `background_compaction` is
    ///    disabled) and hand it a `Weak<Self>`
    ///
    /// Returns `Arc<Self>` because the worker thread needs a weak handle
    /// back to the tree; callers can treat the `Arc` like an `LsmTree`
    /// thanks to deref.
    pub fn open(config: LsmConfig) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&config.data_dir)?;
        std::fs::create_dir_all(config.data_dir.join("sst"))?;

        let mut max_version = 0u64;

        // Discover existing SST files and find max version across all of them.
        let mut sst_readers = Vec::new();
        let mut max_sst_id = 0u64;
        let sst_dir = config.data_dir.join("sst");
        if sst_dir.exists() {
            let entries: Vec<_> = std::fs::read_dir(&sst_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "sst")
                })
                .collect();

            // Open every SST, then order them oldest→newest by the highest
            // version each contains (tie-broken by file id). A pure file-id sort
            // is WRONG after a compaction: the merged SST is written with a high
            // file id but holds OLD data (its max version ≤ the pre-compaction
            // snapshot), so by id it would outrank SSTs flushed concurrently
            // during the merge — inverting version shadowing across a restart and
            // serving stale reads. Ordering by max version restores true age (and
            // matches the in-memory order `compact_inner` maintains: merged
            // oldest-first, concurrently-flushed newer SSTs after).
            let mut with_id: Vec<(u64, SstReader)> = Vec::with_capacity(entries.len());
            for entry in entries {
                let id = entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                max_sst_id = max_sst_id.max(id);
                let reader = SstReader::open(entry.path())?;
                max_version = max_version.max(reader.max_version());
                with_id.push((id, reader));
            }
            with_id.sort_by(|a, b| {
                a.1.max_version()
                    .cmp(&b.1.max_version())
                    .then(a.0.cmp(&b.0))
            });
            sst_readers = with_id.into_iter().map(|(_, r)| r).collect();
        }

        // Replay WAL into a memtable, tracking max version.
        let wal_path = config.data_dir.join("wal.log");
        // A pre-framing (legacy, un-magic'd) WAL is replayed with old per-record
        // semantics, then converted to framed below so no framed appends are ever
        // mixed into an un-magic'd file (which replay can't disambiguate from a
        // torn framed batch).
        let wal_was_legacy = Wal::file_is_legacy(&wal_path)?;
        let memtable = Arc::new(MemTable::new());
        let records = Wal::replay(&wal_path)?;
        for record in &records {
            max_version = max_version.max(record.version);
            match record.record_type {
                RecordType::Put => {
                    memtable.put(&record.key, record.version, record.value.clone());
                }
                RecordType::Delete => {
                    memtable.delete(&record.key, record.version);
                }
                // `replay` resolves commit framing internally and never yields a
                // `Commit` footer — only the data records of completed txns.
                RecordType::Commit => {}
            }
        }

        // Initialize transaction manager at the recovered version.
        let txn_manager = Arc::new(TransactionManager::recover_at_version(max_version));

        // Open WAL for new writes (append to existing — records are still
        // needed until the next flush persists them to SST).
        let wal = Wal::open(&wal_path)?;

        let background_compaction = config.background_compaction;
        let tree = Arc::new(Self {
            config,
            active_memtable: Arc::new(RwLock::new(memtable)),
            immutable_memtables: Arc::new(RwLock::new(Vec::new())),
            sst_files: Arc::new(RwLock::new(sst_readers)),
            wal: Arc::new(parking_lot::Mutex::new(wal)),
            txn_manager,
            flush_lock: parking_lot::RwLock::new(()),
            next_sst_id: std::sync::atomic::AtomicU64::new(max_sst_id + 1),
            compaction_mutex: parking_lot::Mutex::new(()),
            compaction_state: Arc::new((
                parking_lot::Mutex::new(CompactionState::default()),
                parking_lot::Condvar::new(),
            )),
            compaction_handle: parking_lot::Mutex::new(None),
        });

        // One-time upgrade: persist the legacy-WAL records to an SST and
        // re-create the WAL with the framing magic (flush() calls create_fresh).
        // The recovered records are non-empty (a non-empty legacy WAL always
        // rebuilds a non-empty memtable), so the flush converts the file.
        if wal_was_legacy {
            tree.flush()?;
        }

        if background_compaction {
            let weak = Arc::downgrade(&tree);
            let state = Arc::clone(&tree.compaction_state);
            let handle = std::thread::Builder::new()
                .name("rhypedb-compaction".into())
                .spawn(move || compaction_worker(state, weak))?;
            *tree.compaction_handle.lock() = Some(handle);
        }

        Ok(tree)
    }

    /// Begin a new transaction.
    pub fn begin_txn(&self) -> Transaction {
        self.txn_manager.begin()
    }

    /// Take a snapshot for a read-only operation. Returns the latest committed
    /// version as a `u64`. Pair with [`get_at`] / [`scan_prefix_at`] for the
    /// read-only fast path that skips MVCC active-set registration.
    ///
    /// Use this for reads only. A transaction is still required for writes
    /// because conflict detection depends on the active set.
    pub fn read_snapshot(&self) -> u64 {
        self.txn_manager.current_version()
    }

    /// Read a user key at the transaction's snapshot version.
    ///
    /// Search order: active memtable → immutable memtables → SST files (newest first).
    pub fn get(&self, txn: &Transaction, user_key: &[u8]) -> Result<Option<Bytes>> {
        self.get_at(txn.snapshot(), user_key)
    }

    /// Read a user key at the given snapshot version, without requiring a
    /// `Transaction`. Used by the read-only fast path.
    pub fn get_at(&self, version: u64, user_key: &[u8]) -> Result<Option<Bytes>> {
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

    /// Batch point lookup at the given snapshot version.
    ///
    /// Sorts the input keys once, then walks each layer (active memtable →
    /// immutables newest-first → SSTs newest-first) with its batched primitive:
    /// `MemTable::multi_get` / `SstReader::multi_get_versioned`. The SST path
    /// is the meaningful win — sorted-batch + threshold-gallop turns N
    /// independent sparse-index seeks into ~one block-read per ~6 dense
    /// lookups. After each layer, needles that resolved (hit OR tombstone) are
    /// pruned so older layers don't redo their work.
    ///
    /// Returns a `Vec<Option<Bytes>>` in the same order as `user_keys`. `None`
    /// entries are misses (or tombstones — same observable result to callers).
    pub fn multi_get_at(
        &self,
        version: u64,
        user_keys: &[&[u8]],
    ) -> Result<Vec<Option<Bytes>>> {
        let n = user_keys.len();
        let mut out: Vec<Option<Bytes>> = vec![None; n];
        if n == 0 {
            return Ok(out);
        }

        // Sort by user-key bytes, remembering original index for reordering.
        let mut remaining: Vec<(usize, &[u8])> = user_keys.iter().copied().enumerate().collect();
        remaining.sort_unstable_by(|a, b| a.1.cmp(b.1));

        // Snapshot the memtable/SST handles once. Cheap (Arc clones); avoids
        // re-acquiring the RwLock per layer iteration.
        let active = self.active_memtable.read().clone();
        let immutables = self.immutable_memtables.read().clone();
        let ssts_guard = self.sst_files.read();

        // Layer probe helper: takes a per-key lookup closure that returns
        // `Option<MemValue>` (None=miss, Some(_) = hit-or-tombstone). Records
        // hits/tombstones in `out` and prunes them from `remaining`.
        macro_rules! probe_layer {
            ($lookup:expr) => {{
                let lookup = $lookup;
                let keys: Vec<&[u8]> = remaining.iter().map(|(_, k)| *k).collect();
                let vals: Vec<Option<crate::memtable::MemValue>> = lookup(&keys)?;
                let mut next = Vec::with_capacity(remaining.len());
                for ((orig_idx, key), val) in remaining.into_iter().zip(vals.into_iter()) {
                    match val {
                        // Hit (put) or tombstone: stop looking; tombstones
                        // surface as `None` in the final output, matching the
                        // single-key `get_at` semantics.
                        Some(memval) => out[orig_idx] = memval,
                        None => next.push((orig_idx, key)),
                    }
                }
                remaining = next;
            }};
        }

        probe_layer!(|keys: &[&[u8]]| -> Result<_> { Ok(active.multi_get(keys, version)) });

        for mt in immutables.iter().rev() {
            if remaining.is_empty() {
                break;
            }
            probe_layer!(|keys: &[&[u8]]| -> Result<_> { Ok(mt.multi_get(keys, version)) });
        }

        for sst in ssts_guard.iter().rev() {
            if remaining.is_empty() {
                break;
            }
            // Range prune: skip SSTs whose [first_user_key, last_user_key]
            // doesn't overlap the batch. For 20 SSTs created by sequential
            // memtable flushes, most have non-overlapping ranges — pruning
            // cuts ~90% of per-needle bloom calls on a typical traversal.
            let keys: Vec<&[u8]> = remaining.iter().map(|(_, k)| *k).collect();
            if !sst.range_overlaps_sorted(&keys) {
                continue;
            }
            let vals = sst.multi_get_versioned(&keys, version)?;
            let mut next = Vec::with_capacity(remaining.len());
            for ((orig_idx, key), val) in remaining.into_iter().zip(vals) {
                match val {
                    Some(memval) => out[orig_idx] = memval,
                    None => next.push((orig_idx, key)),
                }
            }
            remaining = next;
        }

        Ok(out)
    }

    /// Scan for all keys with the given prefix, visible at the transaction's snapshot.
    /// Returns `(user_key, value)` pairs. Tombstones are excluded from results.
    pub fn scan_prefix(
        &self,
        txn: &Transaction,
        prefix: &[u8],
    ) -> Result<Vec<(Bytes, Bytes)>> {
        self.scan_prefix_at(txn.snapshot(), prefix)
    }

    /// Batched prefix scan: sorts the input prefixes once, then walks each
    /// layer with its batched primitive (`MemTable::multi_scan_prefix` /
    /// `SstReader::multi_scan_prefix_versioned`). The SST primitive is the
    /// payoff — one cursor sweep with threshold gallop turns N independent
    /// sparse-index seeks per SST into ~one block-read per ~6 dense prefixes,
    /// the same shape that landed for point lookups.
    ///
    /// Layer-merge: SSTs oldest-first → immutable memtables oldest-first →
    /// active memtable last. Per-prefix accumulator keeps the newest entry
    /// per user_key; tombstones surface as `None` in the accumulator and
    /// drop in the final filter. Input order restored at the end.
    ///
    /// Replaces the prior v1 implementation that only amortized RwLock
    /// acquisitions across the batch.
    pub fn multi_scan_prefix_at(
        &self,
        version: u64,
        prefixes: &[&[u8]],
    ) -> Result<Vec<Vec<(Bytes, Bytes)>>> {
        let n = prefixes.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Sort prefixes by bytes, remember original index for output reordering.
        let mut indexed: Vec<(usize, &[u8])> = prefixes.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| a.1.cmp(b.1));
        let sorted_prefixes: Vec<&[u8]> = indexed.iter().map(|(_, p)| *p).collect();

        // Per-sorted-prefix accumulator. BTreeMap so the per-prefix output is
        // returned in user-key order, matching `scan_prefix_at`.
        let mut merged: Vec<std::collections::BTreeMap<Bytes, Option<Bytes>>> =
            (0..n).map(|_| std::collections::BTreeMap::new()).collect();

        let ssts_guard = self.sst_files.read();
        let immutables = self.immutable_memtables.read().clone();
        let active = self.active_memtable.read().clone();

        // SSTs (oldest first). Newer entries overwrite older ones in the
        // per-prefix accumulator. Range-prune SSTs that can't possibly match
        // any prefix in the batch — same insight as `multi_get_at`, but the
        // overlap predicate has to account for prefix-of-first-key cases.
        for sst in ssts_guard.iter() {
            if !sst.prefix_range_overlaps_sorted(&sorted_prefixes) {
                continue;
            }
            let layer = sst.multi_scan_prefix_versioned(&sorted_prefixes, version);
            for (sp_idx, entries) in layer.into_iter().enumerate() {
                for (key, value) in entries {
                    merged[sp_idx].insert(key, value);
                }
            }
        }

        // Immutable memtables (oldest first).
        for mt in immutables.iter() {
            let layer = mt.multi_scan_prefix(&sorted_prefixes, version);
            for (sp_idx, entries) in layer.into_iter().enumerate() {
                for (key, value) in entries {
                    merged[sp_idx].insert(key, value);
                }
            }
        }

        // Active memtable (newest).
        let layer = active.multi_scan_prefix(&sorted_prefixes, version);
        for (sp_idx, entries) in layer.into_iter().enumerate() {
            for (key, value) in entries {
                merged[sp_idx].insert(key, value);
            }
        }

        // Filter tombstones and restore input order.
        let mut out: Vec<Vec<(Bytes, Bytes)>> = (0..n).map(|_| Vec::new()).collect();
        for (sp_idx, (orig_idx, _)) in indexed.iter().enumerate() {
            out[*orig_idx] = std::mem::take(&mut merged[sp_idx])
                .into_iter()
                .filter_map(|(k, v)| v.map(|val| (k, val)))
                .collect();
        }
        Ok(out)
    }

    /// Filtered prefix scan: walks the same prefix as `scan_prefix_at` but
    /// passes a `FieldPredicate` to each SST so blocks whose zone bounds
    /// rule out a match can be skipped wholesale (no entry decode).
    ///
    /// **Caller still re-evaluates the predicate** on the returned entries —
    /// zone maps are a coarse-grained pre-filter, not an exact one. For the
    /// 100 K filter-scan bench, the typical pattern is "this block's max year
    /// is 1985, predicate wants year > 2010 — skip 16 entries with one compare."
    ///
    /// Memtables don't have zone maps, so the active + immutable layers fall
    /// back to the full `scan_prefix` path. The win comes entirely from the
    /// SST layer where most data lives at scale.
    pub fn scan_prefix_filtered_at(
        &self,
        version: u64,
        prefix: &[u8],
        predicate: &crate::zone::FieldPredicate,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        let mut merged: std::collections::BTreeMap<Bytes, Option<Bytes>> =
            std::collections::BTreeMap::new();

        // SSTs first (oldest), so newer entries overwrite.
        let ssts = self.sst_files.read();
        for sst in ssts.iter() {
            for (key, value) in sst.scan_prefix_filtered(prefix, version, predicate) {
                merged.insert(key, value);
            }
        }
        drop(ssts);

        // Immutable memtables (oldest first). No zone map — full scan_prefix.
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

        Ok(merged
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .collect())
    }

    /// Bounded prefix scan: returns at most `max_distinct` user keys, each
    /// the latest visible version. Designed for `LIMIT N` push-down on
    /// secondary indexes where the result set is sorted by (encoded_value,
    /// object_id) and the caller only wants the first N matches.
    ///
    /// **Per-layer scan cap.** Each layer is asked for up to `max_distinct`
    /// user keys. For workloads where the same key is rarely tombstoned, the
    /// merged set already contains the first `max_distinct` results — the
    /// per-layer walk stops after ~N entries instead of scanning the full
    /// prefix. For workloads where the top layer tombstones the first N
    /// entries of a lower layer, the merged result may contain fewer than
    /// `max_distinct` live entries. Callers tolerant of that (secondary
    /// indexes on append-mostly data) should prefer this; transactional
    /// reads should not.
    pub fn scan_prefix_at_limited(
        &self,
        version: u64,
        prefix: &[u8],
        max_distinct: usize,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        if max_distinct == 0 {
            return Ok(Vec::new());
        }
        let mut merged: std::collections::BTreeMap<Bytes, Option<Bytes>> =
            std::collections::BTreeMap::new();

        let ssts = self.sst_files.read();
        for sst in ssts.iter() {
            for (key, value) in sst.scan_prefix_max(prefix, version, max_distinct) {
                merged.insert(key, value);
            }
        }
        drop(ssts);

        let immutables = self.immutable_memtables.read().clone();
        for mt in immutables.iter() {
            for (key, value) in mt.scan_prefix_max(prefix, version, max_distinct) {
                merged.insert(key, value);
            }
        }

        let active = self.active_memtable.read().clone();
        for (key, value) in active.scan_prefix_max(prefix, version, max_distinct) {
            merged.insert(key, value);
        }

        Ok(merged
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .take(max_distinct)
            .collect())
    }

    /// Seek-then-scan bounded range scan. Skip everything below
    /// `start_user_key`, then emit at most `max_distinct` live user keys
    /// from within the `prefix` range. Same tombstone-shadow caveat as
    /// `scan_prefix_at_limited`.
    pub fn scan_from_at_limited(
        &self,
        version: u64,
        prefix: &[u8],
        start_user_key: &[u8],
        max_distinct: usize,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        if max_distinct == 0 {
            return Ok(Vec::new());
        }
        let mut merged: std::collections::BTreeMap<Bytes, Option<Bytes>> =
            std::collections::BTreeMap::new();

        let ssts = self.sst_files.read();
        for sst in ssts.iter() {
            for (key, value) in sst.scan_from_max(prefix, start_user_key, version, max_distinct) {
                merged.insert(key, value);
            }
        }
        drop(ssts);

        let immutables = self.immutable_memtables.read().clone();
        for mt in immutables.iter() {
            for (key, value) in mt.scan_from_max(prefix, start_user_key, version, max_distinct) {
                merged.insert(key, value);
            }
        }

        let active = self.active_memtable.read().clone();
        for (key, value) in active.scan_from_max(prefix, start_user_key, version, max_distinct) {
            merged.insert(key, value);
        }

        Ok(merged
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .take(max_distinct)
            .collect())
    }

    /// Tombstone-aware bounded range scan for resumable chunked work.
    ///
    /// Unlike `scan_from_at_limited`, which drops tombstones BEFORE applying
    /// the `max_distinct` cap (so neither its result count nor its max live
    /// key can detect end-of-range across a tombstone run longer than the
    /// cap), this returns the live entries PLUS a tombstone-inclusive
    /// `high_water` and a sound `more` flag. A caller advances its cursor to
    /// `high_water` each step and stops only when `!more` (or `high_water`
    /// is `None`). See [`ChunkScan`].
    ///
    /// `start_user_key` is inclusive (same seek semantics as
    /// `scan_from_at_limited`); pass the key strictly after the last cursor
    /// to avoid re-visiting it.
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
        for (k, v) in merged.into_iter().take(max_distinct) {
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
        })
    }

    /// Scan for all keys with the given prefix, visible at the given snapshot
    /// version, without requiring a `Transaction`. Used by the read-only fast path.
    pub fn scan_prefix_at(
        &self,
        version: u64,
        prefix: &[u8],
    ) -> Result<Vec<(Bytes, Bytes)>> {
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
        // Buffer into the transaction. The write is applied to the WAL + memtable
        // atomically at commit (see `commit`); it is NOT durable or visible (even
        // to this txn) until then.
        txn.record_put(Bytes::copy_from_slice(user_key), value);
        Ok(())
    }

    /// Delete a key within a transaction.
    pub fn delete(&self, txn: &mut Transaction, user_key: &[u8]) -> Result<()> {
        // Buffer a tombstone; applied at commit (see `put` / `commit`).
        txn.record_delete(Bytes::copy_from_slice(user_key));
        Ok(())
    }

    /// Batched put within a transaction. Equivalent to calling `put` N times
    /// but takes the WAL mutex and the memtable read-lock exactly once and
    /// collapses the N WAL `write_all` syscalls into one. Used by paths that
    /// know they're about to issue many writes back-to-back (`create_batch`,
    /// link cascades), where per-call lock acquire + syscall overhead
    /// dominates the actual storage work.
    pub fn put_batch(
        &self,
        txn: &mut Transaction,
        entries: &[(Bytes, Bytes)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        // Buffer the batch; applied as one framed WAL txn + memtable apply at
        // commit (see `commit`).
        for (key, value) in entries {
            txn.record_put(key.clone(), value.clone());
        }
        Ok(())
    }

    /// Batched delete within a transaction. Mirrors `put_batch`: one WAL
    /// mutex acquire + one `write_all` syscall for the whole tombstone
    /// batch, one memtable read-lock acquire for the inserts. Used by the
    /// cascade-delete path where a User-delete at K=100 ratings issues
    /// ~1,000 single deletes today — the per-call WAL + memtable lock
    /// overhead dominates, not the actual tombstone work.
    pub fn delete_batch(
        &self,
        txn: &mut Transaction,
        keys: &[Bytes],
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        // Buffer the tombstone batch; applied at commit (see `commit`).
        for key in keys {
            txn.record_delete(key.clone());
        }
        Ok(())
    }

    /// Commit a transaction, checking for write-write conflicts. With the
    /// default `sync_on_commit = true` config, also fsyncs the WAL before
    /// returning — guarantees the writes survive a power loss. When the
    /// config opts out, the bytes still reach the kernel via the WAL's
    /// `write_all` (so a clean crash is recoverable) but no fsync syscall
    /// runs; matches Postgres's `fsync=off` mode.
    pub fn commit(&self, txn: &mut Transaction) -> Result<u64> {
        // Apply the transaction's buffered writes durably, then publish.
        //
        // Lock order: `flush_lock.read()` (coordinate with flush's WAL truncate,
        // Step A) is acquired OUTSIDE the serialized commit critical section. The
        // apply closure runs INSIDE the txn_manager's `commit_mu`, AFTER the
        // conflict check + version reservation and BEFORE the version is
        // published — so a conflict loser's buffer is never applied, and the
        // reader-visible version never names a not-yet-applied version. Holding
        // `flush_lock.read()` across the WAL append + memtable apply means a
        // concurrent `flush()` (`flush_lock.write()`) cannot rotate the memtable +
        // truncate the WAL between them. `maybe_flush()` runs only AFTER both
        // locks are released (it needs `flush_lock.write()`).
        let version = {
            let _fg = self.flush_lock.read();
            self.txn_manager.commit(txn, |commit_version, writes| {
                {
                    let mut wal = self.wal.lock();
                    wal.append_txn(writes, commit_version)?;
                    if self.config.sync_on_commit {
                        wal.sync()?;
                    }
                }
                let mt = self.active_memtable.read();
                for (key, value) in writes {
                    match value {
                        Some(v) => mt.put(key, commit_version, v.clone()),
                        None => mt.delete(key, commit_version),
                    }
                }
                Ok(())
            })?
        };
        self.maybe_flush()?;
        Ok(version)
    }

    /// Abort a transaction.
    pub fn abort(&self, txn: &mut Transaction) {
        self.txn_manager.abort(txn);
    }

    /// Check if the active memtable is large enough to flush. If a flush
    /// happens and the SST count crosses the auto-compaction threshold,
    /// either signal the background worker (when `background_compaction` is
    /// on) or run a compaction inline. Both are cheap to skip when the
    /// thresholds aren't crossed (just an atomic-ish read).
    fn maybe_flush(&self) -> Result<()> {
        let size = self.active_memtable.read().approximate_size();
        if size >= self.config.memtable_flush_size {
            self.flush()?;
            // Trigger compaction when SSTs pile up. The cost is per-flush
            // (rare), but it pays back on every read path that probes blooms
            // and merges across layers. Keeps the read amplification flat.
            let sst_count = self.sst_files.read().len();
            if sst_count >= self.config.compact_trigger_ssts {
                if self.config.background_compaction {
                    self.signal_compaction();
                } else {
                    self.compact()?;
                }
            }
        }
        Ok(())
    }

    /// Mark a compaction as pending and wake the worker. Cheap: a mutex
    /// flip + one condvar notify. Idempotent — if a compaction is already
    /// pending or running, this just sets the flag again (the worker
    /// coalesces by re-checking the SST count when it picks up the job).
    fn signal_compaction(&self) {
        let mut guard = self.compaction_state.0.lock();
        guard.pending = true;
        self.compaction_state.1.notify_one();
    }

    /// Block until no compaction is pending or running. Used by tests,
    /// benchmarks, and callers that want to observe a quiescent SST set
    /// before reading. Returns immediately when the worker is idle.
    pub fn wait_for_compaction(&self) {
        let mut guard = self.compaction_state.0.lock();
        while guard.pending || guard.running {
            self.compaction_state.1.wait(&mut guard);
        }
    }

    /// Flush the active memtable to a new SST file.
    pub fn flush(&self) -> Result<()> {
        // Exclude all writers for the whole pass: the memtable rotate and the
        // WAL truncate must be atomic w.r.t. any concurrent writer's
        // `[WAL append + memtable write]` (which holds `flush_lock.read()`),
        // else a writer can land its row in the new memtable while its WAL
        // record is truncated → committed data lost across reopen.
        let _fg = self.flush_lock.write();

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

        let mut writer = SstWriter::new_with_zone_extractor(
            &sst_path,
            self.config.zone_extractor.clone(),
        )?;
        for (key, value) in old_memtable.iter() {
            writer.add(&key, &value)?;
        }
        writer.finish()?;

        // Open the new SST for reads and remove from immutable list.
        let reader = SstReader::open(&sst_path)?;
        self.sst_files.write().push(reader);

        let mut immutables = self.immutable_memtables.write();
        immutables.retain(|mt| !Arc::ptr_eq(mt, &old_memtable));

        // Truncate the WAL — flushed data is now durable in the SST.
        let wal_path = self.config.data_dir.join("wal.log");
        let new_wal = Wal::create_fresh(&wal_path)?;
        *self.wal.lock() = new_wal;

        Ok(())
    }

    /// Compact all SST files into a single new SST, dropping old versions
    /// and tombstones that are no longer needed by any active transaction.
    ///
    /// Synchronous: blocks until the merge is done. Acquires
    /// `compaction_mutex` so it serializes against the background worker
    /// (avoids two concurrent merges racing on the final swap).
    pub fn compact(&self) -> Result<()> {
        let _guard = self.compaction_mutex.lock();
        self.compact_inner()
    }

    /// The compaction body — the bytes-moving work. Called by both the
    /// public sync `compact()` and the background worker, both of which
    /// hold `compaction_mutex` so this body never runs concurrently with
    /// itself.
    fn compact_inner(&self) -> Result<()> {
        let ssts = self.sst_files.read();
        if ssts.len() < 2 {
            return Ok(());
        }

        let min_snapshot = self.txn_manager.min_active_snapshot();

        // Merge all SST iterators into a single sorted stream.
        // Since each SST is already sorted and SSTs are ordered oldest→newest,
        // we collect all entries and sort. For a production system you'd use
        // a merge iterator, but this is correct and simple.
        // Capture the EXACT set of SSTs being merged while still holding the
        // read lock. `flush()` only ever APPENDS to `sst_files`, and the read
        // lock blocks it from doing so right now — so any SST that appears after
        // we `drop(ssts)` is a flush that raced the (lock-free) merge below and
        // carries committed data we must NEITHER merge NOR delete. The old code
        // replaced the whole list (`clear()`) and deleted every file, silently
        // destroying those concurrently-flushed SSTs.
        let merged_paths: Vec<PathBuf> =
            ssts.iter().map(|s| s.path().to_path_buf()).collect();

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

        let mut writer = SstWriter::new_with_zone_extractor(
            &sst_path,
            self.config.zone_extractor.clone(),
        )?;
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

        // The exact set of files we merged (captured under the read lock). Any
        // SST NOT in this set was flushed concurrently and must be preserved.
        let merged: HashSet<&Path> = merged_paths.iter().map(|p| p.as_path()).collect();

        if meta.entry_count == 0 {
            // Everything we merged was compacted away — drop the empty output
            // and the merged SSTs, but KEEP any concurrently-flushed SSTs.
            let _ = std::fs::remove_file(&sst_path);
            {
                let mut ssts = self.sst_files.write();
                let kept: Vec<SstReader> = std::mem::take(&mut *ssts)
                    .into_iter()
                    .filter(|s| !merged.contains(s.path()))
                    .collect();
                *ssts = kept;
            }
            for path in &merged_paths {
                let _ = std::fs::remove_file(path);
            }
            return Ok(());
        }

        // Swap the merged SSTs for the new compacted one, PRESERVING any SST
        // flushed while the merge ran. The merged SST holds the oldest data, so
        // it goes first; concurrently-flushed (newer) SSTs follow, keeping
        // `sst_files` oldest→newest so point reads' newest-first `.rev()` and the
        // scan readers' oldest-first BTreeMap merge both resolve the newest
        // version of every key. `open()` reconstructs this exact order by max
        // version (not file id), so it survives a restart too — see `open()`.
        let new_reader = SstReader::open(&sst_path)?;
        {
            let mut ssts = self.sst_files.write();
            let kept: Vec<SstReader> = std::mem::take(&mut *ssts)
                .into_iter()
                .filter(|s| !merged.contains(s.path()))
                .collect();
            ssts.push(new_reader);
            ssts.extend(kept);
        }

        // Delete ONLY the files we actually merged.
        for path in &merged_paths {
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

    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
    }
}

impl Drop for LsmTree {
    /// Tell the compaction worker to exit and wait for it. The shutdown
    /// flag + notify wakes it whether it's blocked on the condvar or
    /// mid-merge (it re-checks `shutdown` after each iteration). Joining
    /// here keeps the tree alive until the worker has released its
    /// `compaction_mutex` and stopped touching SST state.
    ///
    /// Self-join guard: the worker temporarily holds an `Arc<LsmTree>`
    /// (upgraded from its `Weak`) while running `compact_inner`. If the
    /// external holder drops their `Arc` during that window, the
    /// worker's Arc becomes the last one — its end-of-scope drop fires
    /// `LsmTree::drop` on the worker thread, where `handle.join()`
    /// would deadlock against the current thread. Skip the join in that
    /// case; the worker is already finishing and will exit on the next
    /// condvar check (we just set `shutdown = true`).
    fn drop(&mut self) {
        {
            let mut guard = self.compaction_state.0.lock();
            guard.shutdown = true;
            self.compaction_state.1.notify_all();
        }
        if let Some(handle) = self.compaction_handle.lock().take()
            && handle.thread().id() != std::thread::current().id()
        {
            let _ = handle.join();
        }
    }
}

/// Background compaction worker. Owns a `Weak<LsmTree>` so the tree's
/// lifetime isn't extended by this thread — `Drop` flips the shutdown
/// flag, the worker breaks out, the upgrade afterwards (if any) returns
/// `None` because the strong count has already dropped to 0.
fn compaction_worker(
    state: Arc<(parking_lot::Mutex<CompactionState>, parking_lot::Condvar)>,
    weak: std::sync::Weak<LsmTree>,
) {
    loop {
        // Phase A: wait for work. Block on the condvar until something
        // sets `pending` or `shutdown`. Coalescing falls out for free —
        // many `signal_compaction()` calls between two iterations all
        // collapse to one `pending = true`.
        let mut guard = state.0.lock();
        while !guard.pending && !guard.shutdown {
            state.1.wait(&mut guard);
        }
        if guard.shutdown {
            break;
        }
        guard.pending = false;
        guard.running = true;
        drop(guard);

        // Phase B: do the work. If the tree has dropped underneath us,
        // exit cleanly — the shutdown notify would have arrived too.
        if let Some(lsm) = weak.upgrade() {
            let _outer_guard = lsm.compaction_mutex.lock();
            // Errors are logged (best-effort) but don't kill the worker —
            // an SST write failure on one cycle shouldn't prevent us from
            // trying again on the next signal.
            let _ = lsm.compact_inner();
        }

        // Phase C: mark idle and wake any `wait_for_compaction` callers.
        let mut guard = state.0.lock();
        guard.running = false;
        state.1.notify_all();
    }
    // Even after a clean shutdown exit, wake any waiters so they don't
    // spin forever (e.g. a test calling `wait_for_compaction` right
    // before the tree drops).
    let mut guard = state.0.lock();
    guard.pending = false;
    guard.running = false;
    state.1.notify_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn test_config(dir: &Path) -> LsmConfig {
        LsmConfig {
            data_dir: dir.to_path_buf(),
            memtable_flush_size: 1024, // small for testing
            // Don't auto-compact in unit tests — they assert against specific
            // SST counts mid-flow. Tests that want compaction call it
            // explicitly via `tree.compact()`.
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: true,
            // Tests run inline so the existing assertions on SST counts
            // immediately after `compact()` stay deterministic. Targeted
            // tests below opt this back on to exercise the worker.
            background_compaction: false,
        }
    }

    // --- scan_chunk_raw (resumable chunked range scan) -------------------

    // A test key under prefix `P:` whose 8-byte BE suffix == `id`, so
    // lexicographic order matches numeric order (like real object keys).
    fn ckey(id: u64) -> Bytes {
        let mut k = b"P:".to_vec();
        k.extend_from_slice(&id.to_be_bytes());
        Bytes::from(k)
    }

    fn ckey_id(k: &[u8]) -> u64 {
        u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap())
    }

    // Drive scan_chunk_raw to completion exactly as the migration worker
    // does: advance the cursor to `high_water`, stop on `!more` or an empty
    // range. Returns every live id seen, in scan order.
    fn collect_via_chunks(tree: &LsmTree, chunk_size: usize) -> Vec<u64> {
        let prefix = b"P:";
        let mut out = Vec::new();
        let mut cursor: Option<u64> = None;
        loop {
            let start = match cursor {
                None => Bytes::from(prefix.to_vec()),
                Some(c) => ckey(c + 1),
            };
            let snap = tree.txn_manager().current_version();
            let chunk = tree.scan_chunk_raw(snap, prefix, &start, chunk_size).unwrap();
            let Some(hw) = chunk.high_water.clone() else {
                break;
            };
            for (k, _v) in &chunk.live {
                out.push(ckey_id(k));
            }
            cursor = Some(ckey_id(&hw));
            if !chunk.more {
                break;
            }
        }
        out
    }

    // The load-bearing correctness property: a run of tombstones LONGER than
    // the chunk size must not strand the live keys beyond it. A
    // `scan_from_at_limited` caller (which sees only live keys) would land
    // its cursor inside the run, get an empty result, and wrongly conclude
    // end-of-range. scan_chunk_raw's tombstone-inclusive high_water walks
    // straight through.
    #[test]
    fn scan_chunk_raw_walks_through_long_tombstone_run() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        for id in 0..100u64 {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, &ckey(id), Bytes::from("v")).unwrap();
            tree.commit(&mut txn).unwrap();
        }
        // Delete a contiguous run of 50 (ids 20..70) — far longer than the
        // chunk size of 8.
        for id in 20..70u64 {
            let mut txn = tree.begin_txn();
            tree.delete(&mut txn, &ckey(id)).unwrap();
            tree.commit(&mut txn).unwrap();
        }

        let got = collect_via_chunks(&tree, 8);
        let expected: Vec<u64> = (0..20).chain(70..100).collect();
        assert_eq!(got, expected, "long tombstone run stranded live keys");
    }

    // Validates the `more` signal across MULTIPLE sources that each stay
    // UNDER the cap but together overflow it. A naive `more = any source
    // saturated` would report exhaustion after the first chunk and drop the
    // overflow; the correct `more = saturated || merged_len > cap` continues.
    #[test]
    fn scan_chunk_raw_more_reflects_cross_source_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        // Even ids -> flushed to an SST.
        for id in [0u64, 2, 4, 6, 8] {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, &ckey(id), Bytes::from("v")).unwrap();
            tree.commit(&mut txn).unwrap();
        }
        tree.flush().unwrap();
        // Odd ids -> live in the active memtable.
        for id in [1u64, 3, 5, 7, 9] {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, &ckey(id), Bytes::from("v")).unwrap();
            tree.commit(&mut txn).unwrap();
        }

        // cap 8: SST returns 5 (<8), memtable returns 5 (<8) — neither
        // saturates — but merged is 10 distinct > 8, so the first chunk MUST
        // set more=true.
        let snap = tree.txn_manager().current_version();
        let first = tree
            .scan_chunk_raw(snap, b"P:", b"P:", 8)
            .unwrap();
        assert_eq!(first.live.len(), 8);
        assert!(first.more, "cross-source overflow must report more=true");

        let got = collect_via_chunks(&tree, 8);
        assert_eq!(got, (0..10).collect::<Vec<_>>());
    }

    // Regression: a multi-version user key whose two versions STRADDLE a
    // sparse-index block boundary (ENTRIES_PER_BLOCK = 16) must point-read
    // the LATEST after a flush. We place 15 single-version keys (SST entries
    // 0..14) ahead of the target, so the target's two versions land at
    // entries 15 (newest) and 16 (oldest) — across the block boundary. The
    // old `get_versioned` seek (largest internal key) landed on block 1,
    // started its scan at the OLD version, and returned it — a stale-read
    // data-loss bug the chunked field-type migration surfaced.
    #[test]
    fn flush_of_multi_version_key_straddling_block_boundary_reads_latest() {
        let dir = tempfile::tempdir().unwrap();
        let target = b"B:target";
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            // 15 keys sorting BEFORE the target -> SST entries 0..14.
            for i in 0..15u64 {
                let mut t = tree.begin_txn();
                tree.put(&mut t, format!("A:{i:06}").as_bytes(), Bytes::from("a"))
                    .unwrap();
                tree.commit(&mut t).unwrap();
            }
            // Target, two versions -> entries 15 (v2, newest) and 16 (v1).
            let mut t = tree.begin_txn();
            tree.put(&mut t, target, Bytes::from("v1")).unwrap();
            tree.commit(&mut t).unwrap();
            let mut t = tree.begin_txn();
            tree.put(&mut t, target, Bytes::from("v2")).unwrap();
            tree.commit(&mut t).unwrap();
            // A few keys sorting AFTER, so the target isn't the last entry.
            for i in 0..5u64 {
                let mut t = tree.begin_txn();
                tree.put(&mut t, format!("C:{i:06}").as_bytes(), Bytes::from("c"))
                    .unwrap();
                tree.commit(&mut t).unwrap();
            }
            tree.flush().unwrap();
        }
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        assert_eq!(
            tree.get_at(tree.read_snapshot(), target).unwrap(),
            Some(Bytes::from("v2")),
            "straddling multi-version key read the STALE version"
        );
    }

    // Same straddle as above, but through the BATCHED point-lookup path
    // (`multi_get_at` -> SstReader::multi_get / locate_block_from). The seek
    // there had the identical lowest-version seed bug; `get_many` /
    // traversals must agree with `get`.
    #[test]
    fn multi_get_of_multi_version_key_straddling_block_boundary_reads_latest() {
        let dir = tempfile::tempdir().unwrap();
        let target: &[u8] = b"B:target";
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            for i in 0..15u64 {
                let mut t = tree.begin_txn();
                tree.put(&mut t, format!("A:{i:06}").as_bytes(), Bytes::from("a"))
                    .unwrap();
                tree.commit(&mut t).unwrap();
            }
            let mut t = tree.begin_txn();
            tree.put(&mut t, target, Bytes::from("v1")).unwrap();
            tree.commit(&mut t).unwrap();
            let mut t = tree.begin_txn();
            tree.put(&mut t, target, Bytes::from("v2")).unwrap();
            tree.commit(&mut t).unwrap();
            for i in 0..5u64 {
                let mut t = tree.begin_txn();
                tree.put(&mut t, format!("C:{i:06}").as_bytes(), Bytes::from("c"))
                    .unwrap();
                tree.commit(&mut t).unwrap();
            }
            tree.flush().unwrap();
        }
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let snap = tree.read_snapshot();
        let got = tree.multi_get_at(snap, &[target]).unwrap();
        assert_eq!(
            got,
            vec![Some(Bytes::from("v2"))],
            "batched multi_get returned the STALE version (must match get_at)"
        );
        // Differential: both paths agree.
        assert_eq!(got[0], tree.get_at(snap, target).unwrap());
    }

    // An empty range yields high_water=None (the worker's terminal signal).
    #[test]
    fn scan_chunk_raw_empty_range_high_water_none() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let snap = tree.txn_manager().current_version();
        let chunk = tree.scan_chunk_raw(snap, b"P:", b"P:", 8).unwrap();
        assert!(chunk.live.is_empty());
        assert!(chunk.high_water.is_none());
        assert!(!chunk.more);
    }

    // Regression: background compaction must not drop committed data. With
    // small flushes + background compaction, the writer flushes new SSTs WHILE
    // the compaction worker is merging an earlier snapshot. The buggy
    // `compact_inner` `clear()`d the whole SST list and deleted every file at
    // swap time, destroying those concurrently-flushed SSTs (and their data was
    // already out of the WAL) — so both point reads AND scans lost ~77% of keys.
    // This drives that race hard and asserts every key survives, including
    // across a reopen (durability).
    #[test]
    fn compaction_preserves_concurrently_flushed_data() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 16 * 1024, // force many flushes
            compact_trigger_ssts: 4,        // and background compaction -> merge
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: true,
        };
        let tree = LsmTree::open(cfg).unwrap();

        let n: u64 = 20_000;
        let val = Bytes::from(vec![7u8; 256]);
        let key = |id: u64| {
            let mut k = b"v:".to_vec();
            k.extend_from_slice(&id.to_be_bytes());
            k
        };
        // Commit in batches, like the ingest path.
        for chunk in (0..n).collect::<Vec<_>>().chunks(2000) {
            let mut txn = tree.begin_txn();
            for &id in chunk {
                tree.put(&mut txn, &key(id), val.clone()).unwrap();
            }
            tree.commit(&mut txn).unwrap();
        }

        let count_misses = || {
            let snap = tree.read_snapshot();
            let mut miss_single = 0;
            for id in 0..n {
                if tree.get_at(snap, &key(id)).unwrap().is_none() {
                    miss_single += 1;
                }
            }
            let keys: Vec<Vec<u8>> = (0..n).map(key).collect();
            let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
            let batch = tree.multi_get_at(snap, &refs).unwrap();
            let miss_batch = batch.iter().filter(|v| v.is_none()).count();
            (miss_single, miss_batch)
        };

        // Read immediately (possibly mid-compaction) — the query does not wait.
        let (ms_imm, mb_imm) = count_misses();
        // Then quiesce compaction and read again.
        tree.wait_for_compaction();
        let (ms_settled, mb_settled) = count_misses();
        // Scans must also see every key (data is on disk, not just point-read).
        let snap = tree.read_snapshot();
        let scanned = tree.scan_prefix_at(snap, b"v:").unwrap().len();

        assert_eq!(
            (ms_imm, mb_imm, ms_settled, mb_settled),
            (0, 0, 0, 0),
            "point reads lost committed keys after background compaction"
        );
        assert_eq!(scanned, n as usize, "scan lost committed keys after compaction");

        // Durability: reopen the tree from disk and confirm nothing was lost.
        drop(tree);
        let reopened = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 16 * 1024,
            compact_trigger_ssts: 4,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        })
        .unwrap();
        let snap = reopened.read_snapshot();
        let reopened_scanned = reopened.scan_prefix_at(snap, b"v:").unwrap().len();
        assert_eq!(
            reopened_scanned, n as usize,
            "keys lost across reopen after background compaction"
        );
    }

    // === Concurrent-writer contract (gating card 3 / parallel migration) ===
    //
    // Card 3 (parallel migration workers) spawns N threads, each converting a
    // DISJOINT object-id sub-range with its own write txn, all committing
    // concurrently while flushes fire mid-flight. The engine was built for
    // effectively-serialized writers (`put` stamps the SHARED active memtable
    // at a provisional version `current_version()+1` BEFORE the commit-time
    // conflict check), and no test ever exercised multiple concurrent writer
    // THREADS. These tests establish — empirically — whether the disjoint
    // concurrent-writer model loses or corrupts COMMITTED data. They are the
    // gate: if the gate passes, parallel backfill is safe to build on this
    // substrate; if it fails, the write path must be hardened first.

    // GATE: N threads writing disjoint key ranges, committing in small batches
    // with a tiny flush size + background compaction so flushes/compactions
    // fire CONSTANTLY while other threads have in-flight (put-but-not-yet-
    // committed) txns. Every committed key must survive: point read, scan, AND
    // a reopen from disk. A single lost/torn key fails the gate.
    #[test]
    fn concurrent_disjoint_writers_no_committed_loss() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 4 * 1024, // tiny -> flush mid-flight, constantly
            compact_trigger_ssts: 4,       // + background compaction races writers
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: true,
        };
        let tree = LsmTree::open(cfg).unwrap();

        const THREADS: u64 = 12;
        const PER_THREAD: u64 = 2_000;
        const BATCH: u64 = 20; // small -> many commits -> wide in-flight overlap

        // Globally-disjoint key: `o:<thread BE><local BE>`; value encodes the
        // global id so a torn/wrong value (not just a miss) is also caught.
        let key = |t: u64, local: u64| -> Bytes {
            let mut k = b"o:".to_vec();
            k.extend_from_slice(&t.to_be_bytes());
            k.extend_from_slice(&local.to_be_bytes());
            Bytes::from(k)
        };
        let val = |t: u64, local: u64| -> Bytes {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&t.to_be_bytes());
            v.extend_from_slice(&local.to_be_bytes());
            Bytes::from(v)
        };

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS as usize));
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let tree = std::sync::Arc::clone(&tree);
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait(); // release all writers at once -> maximal overlap
                let mut local = 0u64;
                while local < PER_THREAD {
                    let end = (local + BATCH).min(PER_THREAD);
                    let entries: Vec<(Bytes, Bytes)> =
                        (local..end).map(|l| (key(t, l), val(t, l))).collect();
                    // Retry on WriteConflict (disjoint ranges shouldn't conflict
                    // with each other, but the engine is optimistic — be honest).
                    loop {
                        let mut txn = tree.begin_txn();
                        tree.put_batch(&mut txn, &entries).unwrap();
                        match tree.commit(&mut txn) {
                            Ok(_) => break,
                            Err(Error::WriteConflict) => {
                                tree.abort(&mut txn);
                                continue;
                            }
                            Err(e) => panic!("unexpected commit error: {e:?}"),
                        }
                    }
                    local = end;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let total = (THREADS * PER_THREAD) as usize;

        // Every committed key present + correct via point read.
        let check_all_present = |tree: &LsmTree| {
            let snap = tree.read_snapshot();
            let mut misses = 0u64;
            let mut wrong = 0u64;
            for t in 0..THREADS {
                for l in 0..PER_THREAD {
                    match tree.get_at(snap, &key(t, l)).unwrap() {
                        None => misses += 1,
                        Some(v) if v != val(t, l) => wrong += 1,
                        Some(_) => {}
                    }
                }
            }
            (misses, wrong)
        };

        let (misses, wrong) = check_all_present(&tree);
        assert_eq!(
            (misses, wrong),
            (0, 0),
            "concurrent disjoint writers lost/corrupted committed keys (point read)"
        );
        // Quiesce compaction, then scan must also see every key.
        tree.wait_for_compaction();
        let snap = tree.read_snapshot();
        let scanned = tree.scan_prefix_at(snap, b"o:").unwrap().len();
        assert_eq!(scanned, total, "scan lost committed keys");

        // Durability across reopen.
        drop(tree);
        let reopened = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 4 * 1024,
            compact_trigger_ssts: 4,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        })
        .unwrap();
        let (misses, wrong) = check_all_present(&reopened);
        assert_eq!(
            (misses, wrong),
            (0, 0),
            "concurrent disjoint writers lost/corrupted committed keys across reopen"
        );
        let snap = reopened.read_snapshot();
        let scanned = reopened.scan_prefix_at(snap, b"o:").unwrap().len();
        assert_eq!(scanned, total, "scan lost committed keys across reopen");
    }

    // Concurrent writers contending for the SAME keys: every key must end
    // present with a WELL-FORMED value written by a real thread (no missing
    // key, no byte-torn/partial value), and the conflict detector must reject
    // overlapping commits. This checks structural integrity under same-key
    // contention; it deliberately does NOT assert committed-only survivorship —
    // that property is a KNOWN defect (a conflict loser's value can overwrite
    // the committed winner's) pinned by the `#[ignore]`d deterministic test
    // `same_key_conflict_loser_must_not_overwrite_committed_winner` and fixed
    // by storage Step B. (Asserting survivorship here would fail today.)
    #[test]
    fn concurrent_conflicting_writers_no_torn_values() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 4 * 1024,
            compact_trigger_ssts: 4,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: true,
        };
        let tree = LsmTree::open(cfg).unwrap();

        const THREADS: u64 = 8;
        const KEYS: u64 = 400;
        const ROUNDS: u64 = 30;

        let key = |k: u64| -> Bytes {
            let mut b = b"k:".to_vec();
            b.extend_from_slice(&k.to_be_bytes());
            Bytes::from(b)
        };
        // value = [thread u8 | round u8-ish] — first byte is the writing thread.
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS as usize));
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let tree = std::sync::Arc::clone(&tree);
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for r in 0..ROUNDS {
                    for k in 0..KEYS {
                        let v = Bytes::from(vec![t as u8, (r & 0xff) as u8, 0xAB]);
                        loop {
                            let mut txn = tree.begin_txn();
                            tree.put(&mut txn, &key(k), v.clone()).unwrap();
                            match tree.commit(&mut txn) {
                                Ok(_) => break,
                                Err(Error::WriteConflict) => {
                                    tree.abort(&mut txn);
                                    continue;
                                }
                                Err(e) => panic!("unexpected: {e:?}"),
                            }
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Every key must be present with a well-formed value written by a real
        // thread (no missing key, no torn/partial value).
        tree.wait_for_compaction();
        let snap = tree.read_snapshot();
        for k in 0..KEYS {
            let v = tree
                .get_at(snap, &key(k))
                .unwrap()
                .unwrap_or_else(|| panic!("key {k} missing after concurrent contention"));
            assert_eq!(v.len(), 3, "key {k} has a torn value: {v:?}");
            assert!((v[0] as u64) < THREADS, "key {k} value from no real thread: {v:?}");
            assert_eq!(v[2], 0xAB, "key {k} value sentinel corrupted: {v:?}");
        }
    }

    // An ABORTED txn's writes never reach storage: they are buffered in the
    // Transaction and dropped on abort (Step B), so they are invisible to reads.
    #[test]
    fn aborted_txn_data_visibility_probe() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1024,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        };
        let tree = LsmTree::open(cfg).unwrap();

        let key = |k: u64| -> Bytes {
            let mut b = b"a:".to_vec();
            b.extend_from_slice(&k.to_be_bytes());
            Bytes::from(b)
        };

        // Write 200 keys in a txn, then ABORT (never commit).
        let mut txn = tree.begin_txn();
        for k in 0..200u64 {
            tree.put(&mut txn, &key(k), Bytes::from(vec![9u8; 32])).unwrap();
        }
        tree.abort(&mut txn);

        let snap = tree.read_snapshot();
        let visible = (0..200u64)
            .filter(|&k| tree.get_at(snap, &key(k)).unwrap().is_some())
            .count();

        // Buffered + dropped on abort → never visible.
        assert_eq!(
            visible, 0,
            "ABORTED txn rows visible at the current snapshot ({visible}/200)"
        );
    }

    // Regression (Step B fixed the old "resurrection" defect): an aborted txn's
    // write is buffered and dropped, never applied to the memtable, so a LATER
    // commit of a DIFFERENT key cannot resurrect it. (Pre-Step-B, both shared a
    // provisional version and the ghost reappeared once current_version advanced.)
    #[test]
    fn aborted_txn_data_does_not_resurrect_after_later_commit() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        };
        let tree = LsmTree::open(cfg).unwrap();

        let ghost = b"g:ghost";
        let other = b"g:other";

        // Txn A writes the ghost key, then aborts (buffer dropped).
        let mut a = tree.begin_txn();
        tree.put(&mut a, ghost, Bytes::from_static(b"GHOST")).unwrap();
        tree.abort(&mut a);

        let snap0 = tree.read_snapshot();
        assert!(tree.get_at(snap0, ghost).unwrap().is_none(), "ghost visible pre-commit");

        // Txn B writes a DIFFERENT key and commits.
        let mut b = tree.begin_txn();
        tree.put(&mut b, other, Bytes::from_static(b"OTHER")).unwrap();
        tree.commit(&mut b).unwrap();

        // The ghost stays gone — no resurrection.
        let snap1 = tree.read_snapshot();
        assert!(
            tree.get_at(snap1, ghost).unwrap().is_none(),
            "aborted ghost resurrected after a later commit (Step-B regression)"
        );
    }

    // Regression (Step B): on SAME-key concurrent writes, the WriteConflict
    // LOSER must NOT overwrite the committed WINNER. Pre-Step-B, both txns wrote
    // the shared memtable at the same provisional version before the conflict
    // check, so the loser's row physically replaced the winner's. With
    // buffer-in-txn the loser's buffer is discarded on conflict and never
    // applied, so the winner survives (incl. across reopen).
    #[test]
    fn same_key_conflict_loser_must_not_overwrite_committed_winner() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        };
        let tree = LsmTree::open(cfg).unwrap();
        let k: &[u8] = b"k:dup";

        // A and B both write the SAME key; both `put` before either commits.
        let mut a = tree.begin_txn();
        let mut b = tree.begin_txn();
        tree.put(&mut a, k, Bytes::from_static(b"WINNER")).unwrap();
        tree.put(&mut b, k, Bytes::from_static(b"LOSER")).unwrap();

        // A commits (no prior conflict); B then conflicts with A and aborts.
        tree.commit(&mut a).unwrap();
        assert!(
            matches!(tree.commit(&mut b), Err(Error::WriteConflict)),
            "B must conflict with A on the shared key"
        );
        tree.abort(&mut b);

        // The committed winner's value must survive — not the aborted loser's.
        let snap = tree.read_snapshot();
        assert_eq!(
            tree.get_at(snap, k).unwrap().as_deref(),
            Some(&b"WINNER"[..]),
            "committed winner's value was overwritten by the aborted loser"
        );
    }

    // Last-write-wins within a single txn must hold across the WAL too: the
    // buffer coalesces put/delete/put on one key to its final value, and the WAL
    // append + memtable apply are driven from that same deduped sequence, so a
    // reopen (WAL replay) reconstructs the same final value.
    #[test]
    fn same_key_multiple_writes_in_one_txn_last_wins_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20, // keep it in the WAL (force replay on reopen)
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: true,
            background_compaction: false,
        };
        let tree = LsmTree::open(cfg).unwrap();
        let k: &[u8] = b"k:multi";

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, k, Bytes::from_static(b"v1")).unwrap();
        tree.delete(&mut txn, k).unwrap();
        tree.put(&mut txn, k, Bytes::from_static(b"v2")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snap = tree.read_snapshot();
        assert_eq!(tree.get_at(snap, k).unwrap().as_deref(), Some(&b"v2"[..]));

        // Reopen → replay the WAL → same final value.
        drop(tree);
        let reopened = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: true,
            background_compaction: false,
        })
        .unwrap();
        let snap = reopened.read_snapshot();
        assert_eq!(
            reopened.get_at(snap, k).unwrap().as_deref(),
            Some(&b"v2"[..]),
            "last-write-wins not preserved across WAL replay"
        );
    }

    // A multi-record transaction is atomic across reopen even when a flush races
    // its commit: every committed multi-key txn is either wholly present or
    // (if torn) wholly absent — never a partial set of its keys.
    #[test]
    fn multi_record_txn_atomic_across_reopen_with_concurrent_flush() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 4 * 1024, // force flushes mid-stream
            compact_trigger_ssts: 4,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: true,
        };
        let tree = LsmTree::open(cfg).unwrap();

        // Each txn writes a primary + two "index" keys sharing a txn id; the
        // invariant: a txn's three keys are all-present or all-absent.
        const TXNS: u64 = 4000;
        let pk = |t: u64| Bytes::from(format!("p:{t:08}"));
        let i1 = |t: u64| Bytes::from(format!("i1:{t:08}"));
        let i2 = |t: u64| Bytes::from(format!("i2:{t:08}"));
        for t in 0..TXNS {
            let mut txn = tree.begin_txn();
            tree.put_batch(
                &mut txn,
                &[
                    (pk(t), Bytes::from_static(b"x")),
                    (i1(t), Bytes::from_static(b"x")),
                    (i2(t), Bytes::from_static(b"x")),
                ],
            )
            .unwrap();
            tree.commit(&mut txn).unwrap();
        }
        tree.wait_for_compaction();
        drop(tree);

        let reopened = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 4 * 1024,
            compact_trigger_ssts: 4,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        })
        .unwrap();
        let snap = reopened.read_snapshot();
        for t in 0..TXNS {
            let p = reopened.get_at(snap, &pk(t)).unwrap().is_some();
            let a = reopened.get_at(snap, &i1(t)).unwrap().is_some();
            let b = reopened.get_at(snap, &i2(t)).unwrap().is_some();
            assert!(
                p == a && a == b,
                "txn {t} torn across reopen: p={p} i1={a} i2={b}"
            );
        }
        // All txns committed cleanly (no crash injected), so all must be present.
        assert!(
            reopened.get_at(snap, &pk(TXNS - 1)).unwrap().is_some(),
            "committed txns lost across reopen"
        );
    }

    // Upgrade path: a pre-framing (legacy, un-magic'd) WAL is recovered with old
    // per-record semantics AND converted to a framed (magic'd) WAL on open, so
    // subsequent framed appends are never mixed into an un-magic'd file.
    #[test]
    fn legacy_wal_converted_to_framed_on_open() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sst")).unwrap();
        let wal_path = dir.path().join("wal.log");

        // Hand-build a legacy WAL: raw un-framed records, NO magic header.
        let mut raw = Vec::new();
        for i in 0..3u64 {
            crate::wal::WalRecord {
                record_type: RecordType::Put,
                key: Bytes::from(format!("lk{i}")),
                value: Bytes::from_static(b"v"),
                version: i + 1,
            }
            .encode_into(&mut raw);
        }
        std::fs::write(&wal_path, &raw).unwrap();
        assert!(crate::wal::Wal::file_is_legacy(&wal_path).unwrap());

        let cfg = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: true,
            background_compaction: false,
        };
        let tree = LsmTree::open(cfg).unwrap();
        // Legacy records recovered.
        let snap = tree.read_snapshot();
        for i in 0..3u64 {
            assert!(
                tree.get_at(snap, format!("lk{i}").as_bytes()).unwrap().is_some(),
                "legacy record lk{i} lost on upgrade open"
            );
        }
        // WAL converted to framed (magic now present, no longer legacy).
        assert!(
            !crate::wal::Wal::file_is_legacy(&wal_path).unwrap(),
            "legacy WAL was not converted to framed on open"
        );

        // A subsequent committed write + reopen still works on the framed WAL.
        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"lk_new", Bytes::from_static(b"v")).unwrap();
        tree.commit(&mut txn).unwrap();
        drop(tree);
        let reopened = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 1 << 20,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: true,
            background_compaction: false,
        })
        .unwrap();
        let snap = reopened.read_snapshot();
        assert!(reopened.get_at(snap, b"lk0").unwrap().is_some());
        assert!(reopened.get_at(snap, b"lk_new").unwrap().is_some());
    }

    // Recovery must order SSTs by the versions they hold, NOT by file id. A
    // concurrent flush during compaction can leave OLD data in a higher-id file
    // and NEW data in a lower-id file (the merged output gets a fresh high id but
    // holds pre-snapshot data). A file-id sort would then shadow the new value
    // with the old one across a restart. This reproduces that exact on-disk
    // state directly and asserts the newest version still wins.
    #[test]
    fn recovery_orders_ssts_by_version_not_file_id() {
        let dir = tempfile::tempdir().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let key: &[u8] = b"v:\x00\x00\x00\x00\x00\x00\x00\x01";
        let write_sst = |id: u64, version: u64, value: &str| {
            let path = sst_dir.join(format!("{id:08}.sst"));
            let mut w = SstWriter::new(&path).unwrap();
            // Internal key = user_key || !version (big-endian), matching the
            // engine's inverted-version encoding (newer versions sort first).
            let mut ik = key.to_vec();
            ik.extend_from_slice(&(!version).to_be_bytes());
            let mv = Some(Bytes::copy_from_slice(value.as_bytes()));
            w.add(&ik, &mv).unwrap();
            w.finish().unwrap();
        };

        // NEWER value (v=200) in the LOWER file id; OLDER value (v=100) in the
        // HIGHER file id — the post-inversion state.
        write_sst(3, 200, "new");
        write_sst(5, 100, "old");

        let tree = LsmTree::open(LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_flush_size: 64 * 1024,
            compact_trigger_ssts: usize::MAX,
            zone_extractor: None,
            sync_on_commit: false,
            background_compaction: false,
        })
        .unwrap();

        let snap = tree.read_snapshot();
        let new = Some(Bytes::copy_from_slice(b"new"));
        assert_eq!(
            tree.get_at(snap, key).unwrap(),
            new,
            "point read must resolve the newest version regardless of file id"
        );
        assert_eq!(
            tree.multi_get_at(snap, &[key]).unwrap(),
            vec![new.clone()],
            "batch read must too"
        );
        let scanned = tree.scan_prefix_at(snap, b"v:").unwrap();
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].1, Bytes::copy_from_slice(b"new"), "scan must too");
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

    /// Same config as `test_config` but with the background compaction
    /// worker turned ON. Tests in this group exercise the signaling +
    /// worker thread path; `wait_for_compaction` makes the assertions
    /// deterministic without a fixed sleep.
    fn test_config_with_bg_compaction(dir: &Path) -> LsmConfig {
        let mut c = test_config(dir);
        c.background_compaction = true;
        // Auto-compact at 3 SSTs so we can drive the trigger from the test
        // without forcing the legacy 4-SST default.
        c.compact_trigger_ssts = 3;
        c
    }

    #[test]
    fn background_compaction_worker_merges_ssts_on_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config_with_bg_compaction(dir.path())).unwrap();

        for i in 0..3 {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, format!("k{i}").as_bytes(), Bytes::from(format!("v{i}")))
                .unwrap();
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        // Crossing the threshold should have signalled the worker; the
        // explicit flush doesn't call maybe_flush, so kick it.
        tree.signal_compaction();
        tree.wait_for_compaction();

        assert_eq!(
            tree.sst_count(),
            1,
            "background worker should have merged the 3 SSTs into one"
        );

        // Data still readable post-merge.
        let txn = tree.begin_txn();
        for i in 0..3 {
            let v = tree.get(&txn, format!("k{i}").as_bytes()).unwrap();
            assert_eq!(v, Some(Bytes::from(format!("v{i}"))));
        }
    }

    #[test]
    fn background_compaction_signals_coalesce() {
        // Multiple signal_compaction() calls before the worker picks up
        // the first one should NOT trigger multiple separate merges —
        // they coalesce into one. We assert quiescence via
        // wait_for_compaction and observe the merge count via sst_count.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config_with_bg_compaction(dir.path())).unwrap();

        for i in 0..3 {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, format!("k{i}").as_bytes(), Bytes::from(format!("v{i}")))
                .unwrap();
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        // Many signals before quiescing — the worker should still end in
        // a single merged SST.
        for _ in 0..10 {
            tree.signal_compaction();
        }
        tree.wait_for_compaction();
        assert_eq!(tree.sst_count(), 1);
    }

    #[test]
    fn drop_joins_compaction_worker_cleanly() {
        // Dropping a tree with the worker alive must not hang. We start
        // the worker, optionally let it process one round, then drop.
        // No assertion beyond "returns within a reasonable time" — if the
        // drop deadlocks, the test will hang and the harness will time out.
        let dir = tempfile::tempdir().unwrap();
        {
            let tree =
                LsmTree::open(test_config_with_bg_compaction(dir.path())).unwrap();
            for i in 0..3 {
                let mut txn = tree.begin_txn();
                tree.put(
                    &mut txn,
                    format!("k{i}").as_bytes(),
                    Bytes::from(format!("v{i}")),
                )
                .unwrap();
                tree.commit(&mut txn).unwrap();
                tree.flush().unwrap();
            }
            tree.signal_compaction();
            // No explicit wait — Drop must handle a pending worker too.
        }
        // If we got here, Drop returned. Reopen and verify state is sane.
        let tree2 = LsmTree::open(test_config(dir.path())).unwrap();
        let txn = tree2.begin_txn();
        let v = tree2.get(&txn, b"k0").unwrap();
        assert_eq!(v, Some(Bytes::from("v0")));
    }

    #[test]
    fn sync_compact_serializes_with_background_worker() {
        // Even with the worker enabled, an explicit `compact()` call runs
        // to completion synchronously thanks to the compaction mutex.
        // No double-merge / no race.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config_with_bg_compaction(dir.path())).unwrap();

        for i in 0..3 {
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, format!("k{i}").as_bytes(), Bytes::from(format!("v{i}")))
                .unwrap();
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        tree.signal_compaction();
        tree.compact().unwrap();
        tree.wait_for_compaction();

        assert_eq!(tree.sst_count(), 1);
    }

    #[test]
    fn recovery_from_wal_after_reopen() {
        let dir = tempfile::tempdir().unwrap();

        // Write data, don't flush (stays in WAL only).
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"wal-key", Bytes::from("wal-value"))
                .unwrap();
            tree.commit(&mut txn).unwrap();
        }
        // Tree dropped — simulates process exit.

        // Reopen — should recover from WAL.
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let txn = tree.begin_txn();
            let val = tree.get(&txn, b"wal-key").unwrap();
            assert_eq!(val, Some(Bytes::from("wal-value")));
        }
    }

    #[test]
    fn recovery_from_sst_after_reopen() {
        let dir = tempfile::tempdir().unwrap();

        // Write and flush (data in SST).
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"sst-key", Bytes::from("sst-value"))
                .unwrap();
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        // Reopen — should find data in SST.
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let txn = tree.begin_txn();
            let val = tree.get(&txn, b"sst-key").unwrap();
            assert_eq!(val, Some(Bytes::from("sst-value")));
        }
    }

    #[test]
    fn version_counter_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        let version_after_writes;
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            for i in 0..10u64 {
                let mut txn = tree.begin_txn();
                tree.put(&mut txn, format!("k{i}").as_bytes(), Bytes::from("v"))
                    .unwrap();
                tree.commit(&mut txn).unwrap();
            }
            version_after_writes = tree.txn_manager().current_version();
        }

        // Reopen — version counter should resume, not reset.
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let recovered_version = tree.txn_manager().current_version();
            assert!(
                recovered_version >= version_after_writes,
                "version counter went backwards: {recovered_version} < {version_after_writes}"
            );

            // New writes should get higher versions.
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"new-key", Bytes::from("new-val"))
                .unwrap();
            let new_version = tree.commit(&mut txn).unwrap();
            assert!(
                new_version > version_after_writes,
                "new version {new_version} should be > {version_after_writes}"
            );
        }
    }

    #[test]
    fn wal_truncated_after_flush() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");

        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"before-flush", Bytes::from("val"))
                .unwrap();
            tree.commit(&mut txn).unwrap();

            // WAL should have records before flush.
            let records_before = Wal::replay(&wal_path).unwrap();
            assert!(!records_before.is_empty());

            tree.flush().unwrap();

            // WAL should be empty after flush.
            let records_after = Wal::replay(&wal_path).unwrap();
            assert!(
                records_after.is_empty(),
                "WAL should be empty after flush, has {} records",
                records_after.len()
            );
        }
    }

    #[test]
    fn new_writes_after_reopen_dont_conflict() {
        let dir = tempfile::tempdir().unwrap();

        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"key", Bytes::from("v1")).unwrap();
            tree.commit(&mut txn).unwrap();
        }

        // Reopen and overwrite the same key — should succeed, not conflict.
        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put(&mut txn, b"key", Bytes::from("v2")).unwrap();
            tree.commit(&mut txn).unwrap();

            let txn2 = tree.begin_txn();
            assert_eq!(tree.get(&txn2, b"key").unwrap(), Some(Bytes::from("v2")));
        }
    }

    #[test]
    fn read_snapshot_sees_committed_writes() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"key", Bytes::from("value")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snapshot = tree.read_snapshot();
        assert_eq!(
            tree.get_at(snapshot, b"key").unwrap(),
            Some(Bytes::from("value"))
        );
    }

    #[test]
    fn read_snapshot_scan_prefix_works() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"a:1", Bytes::from("one")).unwrap();
        tree.put(&mut txn, b"a:2", Bytes::from("two")).unwrap();
        tree.put(&mut txn, b"b:1", Bytes::from("other")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snapshot = tree.read_snapshot();
        let results = tree.scan_prefix_at(snapshot, b"a:").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn read_snapshot_does_not_register_in_active_set() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let before = tree.txn_manager().min_active_snapshot();

        let _snapshot = tree.read_snapshot();
        let _snapshot2 = tree.read_snapshot();
        let _snapshot3 = tree.read_snapshot();

        // Read snapshots should NOT show up in active_snapshots.
        // min_active_snapshot is u64::MAX when nothing is registered.
        let after = tree.txn_manager().min_active_snapshot();
        assert_eq!(before, after);
    }

    // --- multi_get_at (sorted-batch across layers) --------------------------

    #[test]
    fn multi_get_at_returns_input_order_with_unsorted_keys() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"k:1", Bytes::from("v1")).unwrap();
        tree.put(&mut txn, b"k:2", Bytes::from("v2")).unwrap();
        tree.put(&mut txn, b"k:3", Bytes::from("v3")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snapshot = tree.read_snapshot();
        // Pass keys out of order; output must still match input positions.
        let inputs: Vec<&[u8]> = vec![b"k:3", b"k:1", b"k:2", b"k:missing"];
        let results = tree.multi_get_at(snapshot, &inputs).unwrap();
        assert_eq!(
            results,
            vec![
                Some(Bytes::from("v3")),
                Some(Bytes::from("v1")),
                Some(Bytes::from("v2")),
                None,
            ]
        );
    }

    #[test]
    fn multi_get_at_memtable_shadows_sst() {
        // Write+flush a value to SST, then overwrite in memtable. The newer
        // memtable entry must win for that key while the others come from SST.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"k:1", Bytes::from("old1")).unwrap();
        tree.put(&mut txn, b"k:2", Bytes::from("old2")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"k:1", Bytes::from("new1")).unwrap();
        tree.commit(&mut txn2).unwrap();

        let snapshot = tree.read_snapshot();
        let inputs: Vec<&[u8]> = vec![b"k:1", b"k:2"];
        let results = tree.multi_get_at(snapshot, &inputs).unwrap();
        assert_eq!(
            results,
            vec![Some(Bytes::from("new1")), Some(Bytes::from("old2"))]
        );
    }

    #[test]
    fn multi_get_at_tombstone_in_newer_layer_shadows_put_in_sst() {
        // SST has a put; memtable has a tombstone for the same key. The
        // tombstone must win — caller sees None, NOT the old SST value.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"alive", Bytes::from("a")).unwrap();
        tree.put(&mut txn, b"dead", Bytes::from("rip")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.delete(&mut txn2, b"dead").unwrap();
        tree.commit(&mut txn2).unwrap();

        let snapshot = tree.read_snapshot();
        let inputs: Vec<&[u8]> = vec![b"alive", b"dead"];
        let results = tree.multi_get_at(snapshot, &inputs).unwrap();
        assert_eq!(results, vec![Some(Bytes::from("a")), None]);
    }

    #[test]
    fn multi_get_at_empty_input() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let snapshot = tree.read_snapshot();
        let results = tree.multi_get_at(snapshot, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn multi_get_at_duplicate_keys_in_input() {
        // Same key appears twice in input. Both output positions must receive
        // the same value — the sort/reorder must not drop one.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"k", Bytes::from("v")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snapshot = tree.read_snapshot();
        let inputs: Vec<&[u8]> = vec![b"k", b"missing", b"k"];
        let results = tree.multi_get_at(snapshot, &inputs).unwrap();
        assert_eq!(
            results,
            vec![Some(Bytes::from("v")), None, Some(Bytes::from("v"))]
        );
    }

    // --- multi_scan_prefix_at (sorted-batch prefix across layers) ----------

    #[test]
    fn multi_scan_prefix_at_empty_input() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let snapshot = tree.read_snapshot();
        let results = tree.multi_scan_prefix_at(snapshot, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn multi_scan_prefix_at_returns_input_order_with_unsorted_prefixes() {
        // Pass prefixes in non-sorted order; output positions must match input.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"a:1", Bytes::from("a1")).unwrap();
        tree.put(&mut txn, b"b:1", Bytes::from("b1")).unwrap();
        tree.put(&mut txn, b"b:2", Bytes::from("b2")).unwrap();
        tree.put(&mut txn, b"c:1", Bytes::from("c1")).unwrap();
        tree.commit(&mut txn).unwrap();

        let snapshot = tree.read_snapshot();
        // Input order: c, a, b — sorted is a, b, c. Reordering must restore.
        let prefixes: Vec<&[u8]> = vec![b"c:", b"a:", b"b:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        assert_eq!(results.len(), 3);
        // results[0] corresponds to input "c:"
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].1, Bytes::from("c1"));
        // results[1] corresponds to input "a:"
        assert_eq!(results[1].len(), 1);
        assert_eq!(results[1][0].1, Bytes::from("a1"));
        // results[2] corresponds to input "b:"
        assert_eq!(results[2].len(), 2);
    }

    #[test]
    fn multi_scan_prefix_at_memtable_overwrites_sst() {
        // Write+flush a value to SST, then overwrite a same-prefix key in
        // memtable. The memtable entry must win for that user key while
        // siblings (same prefix, different key) still come from SST.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"p:1", Bytes::from("old1")).unwrap();
        tree.put(&mut txn, b"p:2", Bytes::from("old2")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"p:1", Bytes::from("new1")).unwrap();
        tree.commit(&mut txn2).unwrap();

        let snapshot = tree.read_snapshot();
        let prefixes: Vec<&[u8]> = vec![b"p:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        assert_eq!(results[0].len(), 2);
        // BTreeMap output is user-key-sorted.
        assert_eq!(results[0][0].0.as_ref(), b"p:1");
        assert_eq!(results[0][0].1, Bytes::from("new1"));
        assert_eq!(results[0][1].0.as_ref(), b"p:2");
        assert_eq!(results[0][1].1, Bytes::from("old2"));
    }

    #[test]
    fn multi_scan_prefix_at_tombstone_in_newer_layer_drops_sst_entry() {
        // SST has p:1 = "v1" and p:2 = "v2". Memtable tombstones p:1. The
        // prefix scan must return only p:2.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"p:1", Bytes::from("v1")).unwrap();
        tree.put(&mut txn, b"p:2", Bytes::from("v2")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.delete(&mut txn2, b"p:1").unwrap();
        tree.commit(&mut txn2).unwrap();

        let snapshot = tree.read_snapshot();
        let prefixes: Vec<&[u8]> = vec![b"p:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].0.as_ref(), b"p:2");
        assert_eq!(results[0][0].1, Bytes::from("v2"));
    }

    #[test]
    fn multi_scan_prefix_at_merges_across_multiple_ssts() {
        // Two flushes → two SSTs. One prefix has entries split across both
        // SSTs; the merge must union them without duplication.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"p:1", Bytes::from("v1")).unwrap();
        tree.put(&mut txn, b"q:1", Bytes::from("q1")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"p:2", Bytes::from("v2")).unwrap();
        tree.put(&mut txn2, b"r:1", Bytes::from("r1")).unwrap();
        tree.commit(&mut txn2).unwrap();
        tree.flush().unwrap();

        let snapshot = tree.read_snapshot();
        let prefixes: Vec<&[u8]> = vec![b"p:", b"q:", b"r:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        // p: spans both SSTs (p:1 in older, p:2 in newer).
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 1);
        assert_eq!(results[2].len(), 1);
    }

    #[test]
    fn multi_scan_prefix_at_spans_layers_with_overlapping_entries() {
        // p:1 lives in SST. p:2 lives in active memtable.
        // p:1 then has a fresh put in active memtable — that must overwrite the SST.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"p:1", Bytes::from("sst1")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"p:1", Bytes::from("memnew")).unwrap();
        tree.put(&mut txn2, b"p:2", Bytes::from("mem2")).unwrap();
        tree.commit(&mut txn2).unwrap();

        let snapshot = tree.read_snapshot();
        let prefixes: Vec<&[u8]> = vec![b"p:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        assert_eq!(results[0].len(), 2);
        assert_eq!(results[0][0].0.as_ref(), b"p:1");
        assert_eq!(results[0][0].1, Bytes::from("memnew"));
        assert_eq!(results[0][1].0.as_ref(), b"p:2");
        assert_eq!(results[0][1].1, Bytes::from("mem2"));
    }

    #[test]
    fn multi_get_at_correct_across_many_non_overlapping_ssts() {
        // Make 10 SSTs whose key ranges don't overlap (sequential block of
        // user keys per flush). Then look up a mix of present + absent keys
        // that spans multiple SSTs. The range-pruning fast path must produce
        // the same answer as the per-key scan would.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        for block in 0u64..10 {
            let mut txn = tree.begin_txn();
            for i in 0u64..10 {
                let id = block * 100 + i; // ids 0..9, 100..109, 200..209, ...
                let key = format!("k:{id:08}");
                let val = format!("v{id}");
                tree.put(&mut txn, key.as_bytes(), Bytes::from(val)).unwrap();
            }
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        let snapshot = tree.read_snapshot();

        // Mix of present (one per block) + absent (gaps + above the top).
        let present_ids: Vec<u64> = (0u64..10).map(|b| b * 100 + 5).collect();
        let absent_ids: Vec<u64> = vec![50, 150, 250, 5000];
        let mut all: Vec<String> = present_ids.iter().map(|i| format!("k:{i:08}")).collect();
        all.extend(absent_ids.iter().map(|i| format!("k:{i:08}")));

        let refs: Vec<&[u8]> = all.iter().map(|s| s.as_bytes()).collect();
        let results = tree.multi_get_at(snapshot, &refs).unwrap();

        for (i, id) in present_ids.iter().enumerate() {
            assert_eq!(
                results[i],
                Some(Bytes::from(format!("v{id}"))),
                "missing present id {id} at idx {i}"
            );
        }
        for (offset, absent_id) in absent_ids.iter().enumerate() {
            let idx = present_ids.len() + offset;
            assert!(
                results[idx].is_none(),
                "absent id {absent_id} at idx {idx} unexpectedly returned {:?}",
                results[idx]
            );
        }
    }

    #[test]
    fn multi_scan_prefix_at_correct_across_many_non_overlapping_ssts() {
        // Similar shape, but for prefix scans. 5 SSTs, each holding a few
        // distinct prefix-groups; queries hit a subset; range-prune path
        // must produce the same answers.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        for block in 0u64..5 {
            let mut txn = tree.begin_txn();
            for sub in 0u64..5 {
                // user_key format: "p:<block>:<sub>:item"
                let key = format!("p:{block:02}:{sub:02}:item");
                let val = format!("b{block}s{sub}");
                tree.put(&mut txn, key.as_bytes(), Bytes::from(val)).unwrap();
            }
            tree.commit(&mut txn).unwrap();
            tree.flush().unwrap();
        }

        let snapshot = tree.read_snapshot();

        // Scan three prefixes: one from block 1, one from block 3, one absent.
        let prefixes: Vec<&[u8]> = vec![b"p:01:", b"p:03:", b"p:99:"];
        let results = tree.multi_scan_prefix_at(snapshot, &prefixes).unwrap();

        assert_eq!(results[0].len(), 5, "p:01: should yield 5 entries");
        assert_eq!(results[1].len(), 5, "p:03: should yield 5 entries");
        assert_eq!(results[2].len(), 0, "p:99: should yield nothing");
    }

    #[test]
    fn multi_get_at_spans_multiple_ssts() {
        // Two flushes → two SSTs. Newer SST has the winning version for one
        // key, older SST is the only source for another. Batch must merge
        // newest-first across SSTs.
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        let mut txn = tree.begin_txn();
        tree.put(&mut txn, b"a", Bytes::from("a-old")).unwrap();
        tree.put(&mut txn, b"b", Bytes::from("b-only")).unwrap();
        tree.commit(&mut txn).unwrap();
        tree.flush().unwrap();

        let mut txn2 = tree.begin_txn();
        tree.put(&mut txn2, b"a", Bytes::from("a-new")).unwrap();
        tree.put(&mut txn2, b"c", Bytes::from("c-only")).unwrap();
        tree.commit(&mut txn2).unwrap();
        tree.flush().unwrap();

        let snapshot = tree.read_snapshot();
        let inputs: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let results = tree.multi_get_at(snapshot, &inputs).unwrap();
        assert_eq!(
            results,
            vec![
                Some(Bytes::from("a-new")),
                Some(Bytes::from("b-only")),
                Some(Bytes::from("c-only")),
                None,
            ]
        );
    }

    #[test]
    fn put_batch_matches_individual_puts() {
        // Differential test: same N writes via the batch primitive vs N
        // single-call puts must land the same logical state. Catches any
        // version-stamp / write-set mismatch between the two paths.
        let dir_a = tempfile::tempdir().unwrap();
        let tree_a = LsmTree::open(test_config(dir_a.path())).unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let tree_b = LsmTree::open(test_config(dir_b.path())).unwrap();

        let entries: Vec<(Bytes, Bytes)> = (0..50u64)
            .map(|i| {
                (
                    Bytes::from(format!("k{i:03}")),
                    Bytes::from(format!("v{i:03}-pad")),
                )
            })
            .collect();

        // Tree A: single-op loop.
        let mut txn_a = tree_a.begin_txn();
        for (k, v) in &entries {
            tree_a.put(&mut txn_a, k, v.clone()).unwrap();
        }
        tree_a.commit(&mut txn_a).unwrap();

        // Tree B: one put_batch call.
        let mut txn_b = tree_b.begin_txn();
        tree_b.put_batch(&mut txn_b, &entries).unwrap();
        tree_b.commit(&mut txn_b).unwrap();

        let snap_a = tree_a.read_snapshot();
        let snap_b = tree_b.read_snapshot();
        for (k, v) in &entries {
            assert_eq!(tree_a.get_at(snap_a, k).unwrap(), Some(v.clone()));
            assert_eq!(tree_b.get_at(snap_b, k).unwrap(), Some(v.clone()));
        }
    }

    #[test]
    fn delete_batch_matches_individual_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();

        // Seed 30 keys.
        let mut txn = tree.begin_txn();
        let entries: Vec<(Bytes, Bytes)> = (0..30u64)
            .map(|i| {
                (
                    Bytes::from(format!("k{i:03}")),
                    Bytes::from(format!("v{i:03}")),
                )
            })
            .collect();
        for (k, v) in &entries {
            tree.put(&mut txn, k, v.clone()).unwrap();
        }
        tree.commit(&mut txn).unwrap();

        // Delete the first 10 single-op and the next 10 via delete_batch;
        // last 10 stay live.
        let mut txn = tree.begin_txn();
        for (k, _) in &entries[..10] {
            tree.delete(&mut txn, k).unwrap();
        }
        let batch_keys: Vec<Bytes> = entries[10..20].iter().map(|(k, _)| k.clone()).collect();
        tree.delete_batch(&mut txn, &batch_keys).unwrap();
        tree.commit(&mut txn).unwrap();

        let snap = tree.read_snapshot();
        for (k, _) in &entries[..20] {
            assert_eq!(tree.get_at(snap, k).unwrap(), None, "should be tombstoned: {k:?}");
        }
        for (k, v) in &entries[20..] {
            assert_eq!(
                tree.get_at(snap, k).unwrap(),
                Some(v.clone()),
                "should remain live: {k:?}"
            );
        }
    }

    #[test]
    fn batch_writes_survive_replay() {
        // WAL replay path: a crash after batch writes (no flush) must
        // restore exactly the keys/values the batch staged.
        let dir = tempfile::tempdir().unwrap();
        let entries: Vec<(Bytes, Bytes)> = (0..20u64)
            .map(|i| (Bytes::from(format!("k{i}")), Bytes::from(format!("v{i}"))))
            .collect();

        {
            let tree = LsmTree::open(test_config(dir.path())).unwrap();
            let mut txn = tree.begin_txn();
            tree.put_batch(&mut txn, &entries).unwrap();
            tree.commit(&mut txn).unwrap();
            // Drop without flush — WAL replay must rebuild memtable state.
        }

        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let snap = tree.read_snapshot();
        for (k, v) in &entries {
            assert_eq!(tree.get_at(snap, k).unwrap(), Some(v.clone()));
        }
    }

    #[test]
    fn batch_empty_input_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let tree = LsmTree::open(test_config(dir.path())).unwrap();
        let mut txn = tree.begin_txn();
        tree.put_batch(&mut txn, &[]).unwrap();
        tree.delete_batch(&mut txn, &[]).unwrap();
        tree.commit(&mut txn).unwrap();
    }
}
