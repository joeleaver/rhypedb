use std::collections::{HashMap, HashSet};
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};

use rhypedb_embed::{Embedder, Reranker};
#[cfg(feature = "fastembed")]
use rhypedb_embed::{FastEmbedder, FastReranker};
use rhypedb_schema::{DistanceMetric, FieldType, IndexDef, QuantizationType, Schema, VectorizeDef};
use rhypedb_storage::crash_inject;
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::LsmTree;
use rhypedb_vector::distance::{compute_distance, Metric};
use rhypedb_vector::index::QuantizedIndex;
use rhypedb_vector::quantize::TurboQuantConfig;
use rhypedb_vector::hnsw::HnswConfig;

use crate::object::{deserialize_fields, Value};
use crate::EngineResult;

/// Aggregate indexing status.
#[derive(Debug, Clone)]
pub struct IndexingStatus {
    pub pending: usize,
    pub index_stats: Vec<IndexStat>,
}

#[derive(Debug, Clone)]
pub struct IndexStat {
    pub name: String,
    pub vectors: usize,
}

/// State of a vector field on an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VectorState {
    Pending = 0,
    Indexed = 1,
    Failed = 2,
}

impl From<u8> for VectorState {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Indexed,
            2 => Self::Failed,
            _ => Self::Pending,
        }
    }
}

/// A vectorization job stored in the persistent queue.
#[derive(Debug, Clone)]
pub struct VectorizeJob {
    pub type_name: String,
    pub object_id: u64,
    pub source_field: String,
    pub vector_field: String,
    pub model: String,
}

impl VectorizeJob {
    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();
        put_string(&mut buf, &self.type_name);
        buf.put_u64(self.object_id);
        put_string(&mut buf, &self.source_field);
        put_string(&mut buf, &self.vector_field);
        put_string(&mut buf, &self.model);
        buf.freeze()
    }

    fn deserialize(data: &[u8]) -> Option<Self> {
        let mut pos = 0;
        let type_name = read_string(data, &mut pos)?;
        if pos + 8 > data.len() {
            return None;
        }
        let object_id = u64::from_be_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let source_field = read_string(data, &mut pos)?;
        let vector_field = read_string(data, &mut pos)?;
        let model = read_string(data, &mut pos)?;
        Some(Self {
            type_name,
            object_id,
            source_field,
            vector_field,
            model,
        })
    }
}

fn put_string(buf: &mut BytesMut, s: &str) {
    buf.put_u16(s.len() as u16);
    buf.put_slice(s.as_bytes());
}

fn read_string(data: &[u8], pos: &mut usize) -> Option<String> {
    if *pos + 2 > data.len() {
        return None;
    }
    let len = u16::from_be_bytes(data[*pos..*pos + 2].try_into().ok()?) as usize;
    *pos += 2;
    if *pos + len > data.len() {
        return None;
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len]).ok()?.to_string();
    *pos += len;
    Some(s)
}

const BATCH_SIZE: usize = 256;

/// Manages vector indexes and the async vectorization pipeline.
pub struct Vectorizer {
    storage: Arc<LsmTree>,
    schema: Schema,
    type_ids: HashMap<String, u64>,
    field_ids: HashMap<String, u64>,
    indexes: parking_lot::RwLock<HashMap<String, Arc<QuantizedIndex>>>,
    embedders: parking_lot::Mutex<HashMap<String, Box<dyn Embedder>>>,
    reranker: parking_lot::Mutex<Option<Box<dyn Reranker>>>,
    next_job_id: AtomicU64,
    running: Arc<AtomicBool>,
    worker_handles: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
    claim_mutex: parking_lot::Mutex<()>,
}

/// Effective index config for a Vector field with no (or a partial) `@index`
/// directive. These MUST equal what deployed indexes were built with: a Vector
/// field that omits a parameter resolves to exactly this, so an existing
/// snapshot does not spuriously mismatch and trigger a rebuild on upgrade.
/// Changing any of these is a breaking migration — it rebuilds every index that
/// relied on the default. (Note: `HnswConfig::default()` uses ef_construction
/// 200, which is NOT our legacy value — resolve against these constants, never
/// `Default::default()`.)
const LEGACY_QUANT_BITS: u8 = 4;
const LEGACY_HNSW_M: usize = 16;
const LEGACY_HNSW_EF_CONSTRUCTION: usize = 100;
const LEGACY_METRIC: Metric = Metric::Cosine;

/// Restrict-set size at or below which a filtered `.similar()` is answered by
/// EXACT brute-force over the set instead of HNSW + post-filter. HNSW
/// post-filtering under-fills when a selective filter's matches mostly fall
/// outside the global top-k; brute-forcing a small set is both exact (recall
/// 1.0 within the filter) and, at this size, cheaper than the graph. Above this
/// the filter is non-selective enough that the over-fetch path suffices. The
/// brute path reads each member's f32 from the LSM `v:` keyspace, so its cost
/// scales with the set size, not the corpus. Measured at 384-d
/// (`bench_brute_force_restricted_latency`): restrict 100 -> 0.13 ms, 1k -> 1.3
/// ms, 10k -> 15 ms per query (~linear). 15 ms is the worst case (a filter
/// selecting ~10k) and buys exactness over the cheaper-but-lossy HNSW
/// post-filter — the point of the knob; typical selective filters (10s-1000s)
/// stay sub-2 ms.
const EXACT_FILTER_MAX: usize = 10_000;

/// Resolve a Vector field's effective `(HnswConfig, TurboQuantConfig)` from its
/// `@index` directive, filling any omitted parameter from the legacy defaults.
/// `m_max0` is always derived as `2*m` (not a user knob). The schema is already
/// validated (`validate_schema` rejects `quantization: none`, `m`/`ef_construction`
/// of 0, and `@index` on non-Vector fields), so this mapping is total; the
/// residual `None`-enum arm defaults to 4-bit defensively for an unvalidated
/// (e.g. programmatically built) schema rather than panicking.
fn resolve_index_config(dimensions: u32, index: Option<&IndexDef>) -> (HnswConfig, TurboQuantConfig) {
    let bits = match index.and_then(|i| i.quantization.as_ref()) {
        Some(QuantizationType::TurboQuant2Bit) => 2,
        Some(QuantizationType::TurboQuant3Bit) => 3,
        Some(QuantizationType::TurboQuant4Bit) => 4,
        Some(QuantizationType::None) | None => LEGACY_QUANT_BITS,
    };
    let metric = match index.and_then(|i| i.metric.as_ref()) {
        Some(DistanceMetric::Cosine) => Metric::Cosine,
        Some(DistanceMetric::L2) => Metric::L2,
        Some(DistanceMetric::DotProduct) => Metric::DotProduct,
        None => LEGACY_METRIC,
    };
    let m = index.and_then(|i| i.m).map_or(LEGACY_HNSW_M, |m| m as usize);
    let ef_construction = index
        .and_then(|i| i.ef_construction)
        .map_or(LEGACY_HNSW_EF_CONSTRUCTION, |e| e as usize);
    let hnsw_config = HnswConfig {
        m,
        // HNSW convention: layer-0 capacity is 2*m. Derived, not a user knob.
        m_max0: m * 2,
        ef_construction,
        metric,
    };
    (hnsw_config, TurboQuantConfig::new(dimensions, bits))
}

/// Compare a freshly-constructed `target` index (carrying the config the schema's
/// `@index` directive resolves to) against a `loaded` snapshot. Returns
/// `Some(human-readable reason)` for the first parameter that differs, or `None`
/// if every parameter matches. Any difference means the snapshot must be rebuilt
/// from the LSM to honor the schema (see [`Vectorizer::rebuild_indexes`]). All
/// four parameters are persisted in the snapshot, so this comparison is exact;
/// `m_max0` is omitted because it is always `2*m` (a difference in `m` covers it).
fn index_config_mismatch(target: &QuantizedIndex, loaded: &QuantizedIndex) -> Option<String> {
    if loaded.quant_bits() != target.quant_bits() {
        Some(format!(
            "quantization bits {} -> {}",
            loaded.quant_bits(),
            target.quant_bits()
        ))
    } else if loaded.metric() != target.metric() {
        Some(format!(
            "metric {:?} -> {:?}",
            loaded.metric(),
            target.metric()
        ))
    } else if loaded.hnsw_m() != target.hnsw_m() {
        Some(format!("m {} -> {}", loaded.hnsw_m(), target.hnsw_m()))
    } else if loaded.hnsw_ef_construction() != target.hnsw_ef_construction() {
        Some(format!(
            "ef_construction {} -> {}",
            loaded.hnsw_ef_construction(),
            target.hnsw_ef_construction()
        ))
    } else {
        None
    }
}

impl Vectorizer {
    pub fn new(
        storage: Arc<LsmTree>,
        schema: Schema,
        type_ids: HashMap<String, u64>,
        field_ids: HashMap<String, u64>,
    ) -> EngineResult<Self> {
        // Create HNSW indexes for each @vectorize field.
        let mut indexes = HashMap::new();

        // Build an HNSW index for EVERY Vector field. A `@vectorize` field is a
        // Vector field whose values come from auto-embedding a source text; a
        // bare Vector field (no `@vectorize`) holds caller-supplied vectors via
        // `ingest_vector` (bring-your-own-embeddings). Both share one index +
        // the `v:` keyspace, so both rebuild from the LSM on restart.
        for type_def in schema.types.values() {
            for field in &type_def.fields {
                if let FieldType::Vector(vt) = &field.field_type {
                    let index_key = format!("{}.{}", type_def.name, field.name);
                    // Effective config comes from the field's `@index` directive,
                    // defaulting to the legacy hardcoded values for any omitted
                    // parameter (so existing indexes are unchanged).
                    let (hnsw_config, quant_config) =
                        resolve_index_config(vt.dimensions, field.index());
                    let index = QuantizedIndex::new(hnsw_config, quant_config);
                    indexes.insert(index_key, Arc::new(index));
                }
            }
        }

        // Recover next_job_id by scanning existing queue entries.
        let mut max_job_id = 0u64;
        let snapshot = storage.read_snapshot();
        let prefix = KeyBuilder::queue_prefix();
        if let Ok(entries) = storage.scan_prefix_at(snapshot, &prefix) {
            for (key, _) in &entries {
                if key.len() >= 10 {
                    let id_bytes: [u8; 8] = key[2..10].try_into().unwrap();
                    let job_id = u64::from_be_bytes(id_bytes);
                    max_job_id = max_job_id.max(job_id);
                }
            }
        }

        let vectorizer = Self {
            storage,
            schema,
            type_ids,
            field_ids,
            indexes: parking_lot::RwLock::new(indexes),
            embedders: parking_lot::Mutex::new(HashMap::new()),
            reranker: parking_lot::Mutex::new(None),
            next_job_id: AtomicU64::new(max_job_id + 1),
            running: Arc::new(AtomicBool::new(false)),
            worker_handles: parking_lot::Mutex::new(Vec::new()),
            claim_mutex: parking_lot::Mutex::new(()),
        };

        vectorizer.rebuild_indexes()?;
        vectorizer.reconcile_pending_jobs()?;

        Ok(vectorizer)
    }

    /// Re-enqueue vectorize jobs orphaned by a crash between `claim_batch`
    /// (which deletes + commits the queue entry up front) and `store_and_index`
    /// (which writes the `v:` vector and flips the state to `Indexed`). Such a
    /// job leaves a durable `vector_state == Pending` with no queue entry and no
    /// vector, and nothing else re-creates it — the HNSW rebuild only sweeps the
    /// `v:` keyspace — so without this the object stays unindexed until its
    /// source row is mutated again (silent loss across a hard kill / jkbase
    /// Pause→SIGKILL). Runs once at open, single-threaded, before the worker
    /// starts, so it cannot race a live claim.
    ///
    /// Only `@vectorize` fields are auto-embedded; bare (BYO) `Vector` fields are
    /// written `Indexed` directly by `ingest_vectors`, so they never sit at
    /// `Pending` and are excluded. A `Pending` state whose queue entry still
    /// exists is a normal backlog job (not an orphan) and is left alone — so a
    /// routine restart never double-enqueues. A `Pending` state whose object no
    /// longer exists (or whose source field is no longer a string) is a true
    /// orphan the worker would only drop; it is cleaned up so the scan converges.
    fn reconcile_pending_jobs(&self) -> EngineResult<()> {
        struct Tmpl {
            type_name: String,
            source_field: String,
            vector_field: String,
            model: String,
        }
        // (type_id, field_id) -> job template, for every @vectorize field.
        let mut templates: HashMap<(u64, u64), Tmpl> = HashMap::new();
        for type_def in self.schema.types.values() {
            let Some(&type_id) = self.type_ids.get(&type_def.name) else {
                continue;
            };
            for field in &type_def.fields {
                if !matches!(field.field_type, FieldType::Vector(_)) {
                    continue;
                }
                let Some(vd) = field.vectorize() else {
                    continue;
                };
                let field_key = format!("{}.{}", type_def.name, field.name);
                let Some(&field_id) = self.field_ids.get(&field_key) else {
                    continue;
                };
                templates.insert(
                    (type_id, field_id),
                    Tmpl {
                        type_name: type_def.name.clone(),
                        source_field: vd.source_field.clone(),
                        vector_field: field.name.clone(),
                        model: vd.model.clone(),
                    },
                );
            }
        }
        if templates.is_empty() {
            return Ok(());
        }

        let snapshot = self.storage.read_snapshot();

        // Objects already on the queue (legitimately pending, not orphaned), keyed
        // by (type_id, object_id, field_id). Skip these so a normal restart with a
        // pending backlog doesn't double-enqueue (and double-embed) every job.
        let mut queued: HashSet<(u64, u64, u64)> = HashSet::new();
        for (_, value) in &self
            .storage
            .scan_prefix_at(snapshot, &KeyBuilder::queue_prefix())?
        {
            if let Some(job) = VectorizeJob::deserialize(value) {
                let field_key = format!("{}.{}", job.type_name, job.vector_field);
                if let (Some(&tid), Some(&fid)) = (
                    self.type_ids.get(&job.type_name),
                    self.field_ids.get(&field_key),
                ) {
                    queued.insert((tid, job.object_id, fid));
                }
                // A queued job whose stored type/field name no longer resolves
                // (e.g. renamed after enqueue) is simply omitted from `queued`.
                // Benign: the live worker would itself drop that stale-name job
                // (process_batch resolves by current name), and reconcile keys the
                // state keyspace by the stable field_id, so the object is just
                // re-enqueued under its current name instead.
            }
        }

        let type_ids: HashSet<u64> = templates.keys().map(|(t, _)| *t).collect();
        let mut to_enqueue: Vec<VectorizeJob> = Vec::new();
        let mut stale_states: Vec<Bytes> = Vec::new();

        for type_id in type_ids {
            let prefix = KeyBuilder::vector_state_type_prefix(type_id);
            for (key, value) in &self.storage.scan_prefix_at(snapshot, &prefix)? {
                // Only Pending states are orphan candidates.
                if value.first().copied().map(VectorState::from) != Some(VectorState::Pending) {
                    continue;
                }
                // Key: s:<type_id>:<object_id>:<field_id> — object_id [11..19],
                // field_id [20..28].
                if key.len() < 28 {
                    continue;
                }
                let object_id = u64::from_be_bytes(key[11..19].try_into().unwrap());
                let field_id = u64::from_be_bytes(key[20..28].try_into().unwrap());
                let Some(tmpl) = templates.get(&(type_id, field_id)) else {
                    continue; // a Vector field without @vectorize — not auto-embedded
                };
                if queued.contains(&(type_id, object_id, field_id)) {
                    continue; // still on the queue → the worker will handle it
                }

                // Mirror process_batch's filter: re-enqueue only if the source row
                // still exists and carries a string source value; otherwise the
                // state is a true orphan (object deleted / source cleared) and is
                // cleaned up so it isn't re-scanned every restart.
                let obj_key = KeyBuilder::object(type_id, object_id);
                let has_source_text = match self.storage.get_at(snapshot, &obj_key) {
                    Ok(Some(data)) => {
                        matches!(
                            deserialize_fields(&data).get(&tmpl.source_field),
                            Some(Value::String(_))
                        )
                    }
                    _ => false,
                };

                if has_source_text {
                    to_enqueue.push(VectorizeJob {
                        type_name: tmpl.type_name.clone(),
                        object_id,
                        source_field: tmpl.source_field.clone(),
                        vector_field: tmpl.vector_field.clone(),
                        model: tmpl.model.clone(),
                    });
                } else {
                    stale_states.push(key.clone());
                }
            }
        }

        // Clean up true orphans (object gone / source cleared) in one txn.
        if !stale_states.is_empty() {
            let mut txn = self.storage.begin_txn();
            for key in &stale_states {
                self.storage.delete(&mut txn, key)?;
            }
            self.storage.commit(&mut txn).map_err(|e| match e {
                rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
                other => crate::EngineError::Storage(other),
            })?;
        }

        // Re-enqueue recoverable orphans (each writes a fresh queue entry, and
        // re-asserts the Pending state). Idempotent: a later store_and_index for
        // an already-indexed object just re-inserts the same object_id.
        for job in to_enqueue {
            self.enqueue(job)?;
        }

        Ok(())
    }

    fn snapshot_path(&self, index_key: &str) -> std::path::PathBuf {
        let sanitized = index_key.replace('.', "_");
        self.storage.data_dir().join(format!("hnsw_{sanitized}.bin"))
    }

    /// Rebuild HNSW indexes from snapshots or persisted vectors.
    /// Tries loading a serialized snapshot first (O(n) sequential read).
    /// Falls back to full rebuild from f32 vectors in the LSM if no
    /// snapshot exists or the snapshot is corrupt.
    fn rebuild_indexes(&self) -> EngineResult<()> {
        let mut indexes = self.indexes.write();
        if indexes.is_empty() {
            return Ok(());
        }

        let index_keys: Vec<String> = indexes.keys().cloned().collect();

        for index_key in &index_keys {
            let snapshot_path = self.snapshot_path(index_key);

            // Try loading from snapshot.
            if snapshot_path.exists() {
                match std::fs::File::open(&snapshot_path) {
                    Ok(file) => {
                        let mut reader = std::io::BufReader::new(file);
                        match QuantizedIndex::load(&mut reader) {
                            Ok(loaded) => {
                                // The schema is the source of truth: the fresh
                                // index already in the map carries the config the
                                // current `@index` directive resolves to. If a
                                // persisted snapshot was built with a different
                                // bits/metric/m/ef_construction, it can't be
                                // reconciled in place — drop it and rebuild from
                                // the LSM f32 (which `save_single_snapshot` then
                                // persists with the new config, so the next
                                // restart matches and does not rebuild again).
                                let mismatch = {
                                    let target = indexes.get(index_key).unwrap();
                                    index_config_mismatch(target, &loaded)
                                };
                                if let Some(reason) = mismatch {
                                    eprintln!(
                                        "HNSW snapshot for {index_key} was built with a different \
                                         @index config ({reason}); rebuilding from LSM to match the schema"
                                    );
                                    // Fall through to the full LSM rebuild below,
                                    // which uses the fresh target-config index.
                                } else {
                                    let loaded = Arc::new(loaded);

                                    // Delta rebuild: insert any vectors in LSM missing from the snapshot.
                                    let delta = self.insert_delta_vectors(
                                        index_key, &loaded,
                                    )?;

                                    indexes.insert(index_key.clone(), loaded);
                                    eprintln!(
                                        "loaded HNSW snapshot for {index_key} ({} vectors, {delta} delta)",
                                        indexes[index_key].len()
                                    );
                                    continue;
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "corrupt HNSW snapshot for {index_key}, rebuilding: {e}"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("failed to open HNSW snapshot for {index_key}: {e}");
                    }
                }
            }

            // Full rebuild from f32 vectors in LSM.
            let index = indexes.get(index_key).unwrap();
            let count = self.rebuild_index_from_lsm(index_key, index)?;

            if count > 0 {
                eprintln!("rebuilt HNSW index for {index_key}: {count} vectors");
                self.save_single_snapshot(index_key, index);
            }
        }

        Ok(())
    }

    /// Insert any f32 vectors from the LSM that are missing from a loaded index.
    /// Returns the number of delta vectors inserted.
    fn insert_delta_vectors(
        &self,
        index_key: &str,
        index: &QuantizedIndex,
    ) -> EngineResult<usize> {
        let (type_id, field_id) = match self.resolve_index_ids(index_key) {
            Some(ids) => ids,
            None => return Ok(0),
        };

        let vectors = self.scan_vectors_for_field(type_id, field_id)?;
        // Only the vectors not already in the loaded index; insert them in parallel.
        let missing: Vec<(u64, Vec<f32>)> = vectors
            .into_iter()
            .filter(|(object_id, _)| !index.contains_id(*object_id))
            .collect();
        let delta = missing.len();
        index.insert_parallel(&missing);

        Ok(delta)
    }

    /// Rebuild an index fully from f32 vectors in the LSM.
    /// Returns the number of vectors inserted.
    fn rebuild_index_from_lsm(
        &self,
        index_key: &str,
        index: &QuantizedIndex,
    ) -> EngineResult<usize> {
        let (type_id, field_id) = match self.resolve_index_ids(index_key) {
            Some(ids) => ids,
            None => return Ok(0),
        };

        let vectors = self.scan_vectors_for_field(type_id, field_id)?;
        let count = vectors.len();
        index.insert_parallel(&vectors);

        Ok(count)
    }

    fn resolve_index_ids(&self, index_key: &str) -> Option<(u64, u64)> {
        let mut parts = index_key.splitn(2, '.');
        let type_name = parts.next()?;
        let type_id = *self.type_ids.get(type_name)?;
        let field_id = *self.field_ids.get(index_key)?;
        Some((type_id, field_id))
    }

    fn scan_vectors_for_field(
        &self,
        type_id: u64,
        field_id: u64,
    ) -> EngineResult<Vec<(u64, Vec<f32>)>> {
        let prefix = KeyBuilder::vector_prefix(type_id);
        let snapshot = self.storage.read_snapshot();
        let entries = self.storage.scan_prefix_at(snapshot, &prefix)?;

        let mut results = Vec::new();
        for (key, data) in &entries {
            // Cold-reopen rebuild boundary: a crash partway through this scan must
            // still converge on the next reopen (the rebuild is purely derived
            // from the durable `v:` keyspace and writes nothing).
            crash_inject::hit(crash_inject::Site::VectorizeRebuildMidScan);
            if key.len() < 2 + 8 + 1 + 8 + 1 + 8 {
                continue;
            }
            let key_field_id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let key_field_id = u64::from_be_bytes(key_field_id_bytes);
            if key_field_id != field_id {
                continue;
            }
            let obj_id_start = key.len() - 8 - 1 - 8;
            let obj_id_bytes: [u8; 8] =
                key[obj_id_start..obj_id_start + 8].try_into().unwrap();
            let object_id = u64::from_be_bytes(obj_id_bytes);

            if let Some(vector) = deserialize_f32_vec(data) {
                results.push((object_id, vector));
            }
        }

        Ok(results)
    }

    /// Save all HNSW index snapshots to disk.
    pub fn save_snapshots(&self) {
        let indexes = self.indexes.read();
        for (index_key, index) in indexes.iter() {
            self.save_single_snapshot(index_key, index);
        }
    }

    fn save_single_snapshot(&self, index_key: &str, index: &QuantizedIndex) {
        if index.is_empty() {
            return;
        }
        let path = self.snapshot_path(index_key);
        let tmp_path = path.with_extension("bin.tmp");
        match std::fs::File::create(&tmp_path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                if let Err(e) = index.save(&mut writer) {
                    eprintln!("failed to write HNSW snapshot for {index_key}: {e}");
                    let _ = std::fs::remove_file(&tmp_path);
                    return;
                }
                if let Err(e) = std::io::Write::flush(&mut writer) {
                    eprintln!("failed to flush HNSW snapshot for {index_key}: {e}");
                    let _ = std::fs::remove_file(&tmp_path);
                    return;
                }
                if let Err(e) = std::fs::rename(&tmp_path, &path) {
                    eprintln!("failed to rename HNSW snapshot for {index_key}: {e}");
                    let _ = std::fs::remove_file(&tmp_path);
                }
            }
            Err(e) => {
                eprintln!("failed to create HNSW snapshot file for {index_key}: {e}");
            }
        }
    }

    /// Enqueue a vectorization job for an object.
    pub fn enqueue(&self, job: VectorizeJob) -> EngineResult<()> {
        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
        let key = KeyBuilder::queue_entry(job_id);
        let value = job.serialize();

        // Set vector state to pending.
        let type_id = self.type_ids[&job.type_name];
        let field_key = format!("{}.{}", job.type_name, job.vector_field);
        let field_id = self.field_ids[&field_key];
        let state_key = KeyBuilder::vector_state(type_id, job.object_id, field_id);

        let mut txn = self.storage.begin_txn();
        self.storage
            .put(&mut txn, &key, value)?;
        self.storage.put(
            &mut txn,
            &state_key,
            Bytes::from(vec![VectorState::Pending as u8]),
        )?;
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
            other => crate::EngineError::Storage(other),
        })?;

        Ok(())
    }

    /// Get the vectorization state of a specific vector field on an object.
    pub fn get_state(
        &self,
        type_name: &str,
        object_id: u64,
        vector_field: &str,
    ) -> EngineResult<VectorState> {
        let type_id = self.type_ids[type_name];
        let field_key = format!("{type_name}.{vector_field}");
        let field_id = self.field_ids[&field_key];
        let state_key = KeyBuilder::vector_state(type_id, object_id, field_id);

        let snapshot = self.storage.read_snapshot();
        match self.storage.get_at(snapshot, &state_key)? {
            Some(data) if !data.is_empty() => Ok(VectorState::from(data[0])),
            _ => Ok(VectorState::Pending),
        }
    }

    /// Get aggregate indexing status: pending queue depth, indexed count, and
    /// vectors loaded in each HNSW index.
    pub fn status(&self) -> IndexingStatus {
        // Count pending jobs in the queue.
        let snapshot = self.storage.read_snapshot();
        let prefix = KeyBuilder::queue_prefix();
        let pending = self
            .storage
            .scan_prefix_at(snapshot, &prefix)
            .map(|e| e.len())
            .unwrap_or(0);

        // Count vectors in each HNSW index.
        let indexes = self.indexes.read();
        let mut index_stats = Vec::new();
        for (name, index) in indexes.iter() {
            index_stats.push(IndexStat {
                name: name.clone(),
                vectors: index.len(),
            });
        }

        IndexingStatus {
            pending,
            index_stats,
        }
    }

    /// Process pending jobs using the shared embedder (for single-threaded use / tests).
    pub fn process_pending(&self) -> EngineResult<usize> {
        let batch = self.claim_batch()?;
        if batch.is_empty() {
            return Ok(0);
        }
        self.process_batch(batch)
    }

    /// Atomically claim a batch of jobs from the queue.
    /// Deletes queue entries upfront so parallel workers don't double-claim.
    fn claim_batch(&self) -> EngineResult<Vec<(VectorizeJob, u64)>> {
        let _lock = self.claim_mutex.lock();

        let mut txn = self.storage.begin_txn();
        let prefix = KeyBuilder::queue_prefix();
        let entries = self.storage.scan_prefix(&txn, &prefix)?;

        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let jobs: Vec<(Bytes, VectorizeJob)> = entries
            .into_iter()
            .filter_map(|(key, value)| {
                let job = VectorizeJob::deserialize(&value)?;
                Some((key, job))
            })
            .take(BATCH_SIZE)
            .collect();

        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        // Delete queue entries to claim them.
        for (key, _) in &jobs {
            self.storage.delete(&mut txn, key)?;
        }
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
            other => crate::EngineError::Storage(other),
        })?;

        // The queue entries are now durably deleted but no `v:`/`Indexed` exists
        // yet — the orphan window the crash-fuzz harness recovers from.
        crash_inject::hit(crash_inject::Site::VectorizeAfterClaimCommit);

        Ok(jobs
            .into_iter()
            .map(|(_, job)| {
                let object_id = job.object_id;
                (job, object_id)
            })
            .collect())
    }

    /// Process a claimed batch of jobs. Embedding uses the shared
    /// `self.embedders` (one model instance across the worker and query paths,
    /// ~124MB rather than one copy each), locked only for the embed call so it
    /// doesn't block query-path embeds during the insert/commit phase.
    fn process_batch(
        &self,
        jobs: Vec<(VectorizeJob, u64)>,
    ) -> EngineResult<usize> {
        // Group by model.
        let mut jobs_by_model: HashMap<String, Vec<VectorizeJob>> = HashMap::new();
        for (job, _) in jobs {
            jobs_by_model
                .entry(job.model.clone())
                .or_default()
                .push(job);
        }

        let mut processed = 0;

        for (model_name, batch_jobs) in &jobs_by_model {
            let texts: Vec<String> = batch_jobs
                .iter()
                .filter_map(|job| {
                    let type_id = self.type_ids.get(&job.type_name)?;
                    let obj_key = KeyBuilder::object(*type_id, job.object_id);
                    let snapshot = self.storage.read_snapshot();
                    let data = self.storage.get_at(snapshot, &obj_key).ok()??;
                    let fields = deserialize_fields(&data);
                    match fields.get(&job.source_field)? {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    }
                })
                .collect();

            if texts.is_empty() {
                continue;
            }

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // Lock the shared embedder only for the embed; release before the
            // insert/commit phase so concurrent query-path embeds don't block.
            let embed_result = {
                let mut embedders = self.embedders.lock();
                // Lazily load the fastembed-backed embedder for this model. When
                // built without the `fastembed` feature there's no built-in
                // embedder to construct, so an absent entry yields a clear error.
                #[cfg(feature = "fastembed")]
                embedders.entry(model_name.clone()).or_insert_with(|| {
                    Box::new(FastEmbedder::new(model_name).expect("failed to load model"))
                });
                match embedders.get_mut(model_name) {
                    Some(embedder) => embedder.embed(&text_refs),
                    None => Err(rhypedb_embed::EmbedError::Model(
                        "no embedder available (built without the `fastembed` feature)".into(),
                    )),
                }
            };
            let embeddings = match embed_result {
                Ok(e) => e,
                Err(e) => {
                    self.mark_batch_failed(batch_jobs, &format!("{e}"))?;
                    continue;
                }
            };

            for (emb_idx, job) in batch_jobs.iter().enumerate() {
                if emb_idx >= embeddings.len() {
                    break;
                }

                let embedding = &embeddings[emb_idx];
                self.store_and_index(
                    &job.type_name,
                    job.object_id,
                    &job.vector_field,
                    embedding,
                )?;
                processed += 1;
            }
        }

        Ok(processed)
    }

    /// Search a vector index with a text query (encodes text first).
    #[allow(clippy::too_many_arguments)]
    pub fn search_text(
        &self,
        type_name: &str,
        vector_field: &str,
        query_text: &str,
        k: usize,
        ef: usize,
        rerank: bool,
        restrict: Option<&HashSet<u64>>,
    ) -> EngineResult<Vec<(u64, f32)>> {
        let index_key = format!("{type_name}.{vector_field}");
        let index = self
            .indexes
            .read()
            .get(&index_key)
            .cloned()
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })?;

        // Find the model for this field.
        let model = self
            .schema
            .get_type(type_name)
            .and_then(|td| td.get_field(vector_field))
            .and_then(|fd| fd.vectorize())
            .map(|v| v.model.clone())
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })?;

        let query_vec = {
            let mut embedders = self.embedders.lock();
            #[cfg(feature = "fastembed")]
            embedders.entry(model.clone()).or_insert_with(|| {
                Box::new(FastEmbedder::new(&model).expect("failed to load model"))
            });
            match embedders.get_mut(&model) {
                Some(embedder) => embedder
                    .embed(&[query_text])
                    .map_err(|e| crate::EngineError::TypeNotFound(e.to_string()))?,
                None => {
                    return Err(crate::EngineError::TypeNotFound(
                        "no embedder available (built without the `fastembed` feature)".into(),
                    ))
                }
            }
        };

        if query_vec.is_empty() {
            return Ok(Vec::new());
        }

        // Whether cross-encoder reranking is active. RHYPEDB_DISABLE_RERANK
        // turns it off entirely (no reranker model is loaded) — raw HNSW
        // results, much faster and a far smaller memory/image footprint.
        let rerank_disabled = std::env::var_os("RHYPEDB_DISABLE_RERANK").is_some();

        // How many HNSW candidates to retrieve. With reranking off we only need
        // the top k. With it on we over-retrieve a *bounded* pool to feed the
        // cross-encoder — it runs one forward pass per candidate, so this is the
        // dominant query cost. Capped (overridable via RHYPEDB_RERANK_CANDIDATES)
        // rather than the old uncapped `k * 10`, which reranked ~120 verses for
        // a 12-result query.
        let retrieval_k = if rerank_disabled {
            k
        } else {
            std::env::var("RHYPEDB_RERANK_CANDIDATES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or((k * 3).min(48))
                .max(k)
        };
        // Exact small-set path: a selective filter restricts retrieval to a
        // bounded set — score it exactly (over the LSM f32) instead of via HNSW,
        // which under-fills for selective filters. The cross-encoder rerank below
        // then runs over these candidates, truncated to the SAME bounded
        // `retrieval_k` pool the global path uses (the cross-encoder is one
        // forward pass per candidate, so the pool is deliberately capped). So a
        // very large filter only sends its top-`retrieval_k` by exact vector
        // distance to the cross-encoder. `restrict = None` (a global search,
        // incl. every deployed bible-app query) takes the untouched HNSW path.
        let use_brute = restrict.is_some_and(|s| s.len() <= EXACT_FILTER_MAX);
        let mut candidates = if use_brute {
            let set = restrict.unwrap();
            if std::env::var_os("RHYPEDB_DEBUG_RERANK").is_some() {
                eprintln!(
                    "[brute] key={index_key} restrict={} retrieval_k={retrieval_k} (exact small-set path)",
                    set.len()
                );
            }
            let mut c = self.brute_force_restricted(&index_key, &index, &query_vec[0], set);
            c.truncate(retrieval_k);
            c
        } else {
            index.search(&query_vec[0], retrieval_k, ef.max(retrieval_k))
        };

        // Full-precision rerank: replace the TurboQuant estimates with exact
        // distances against the f32 vectors in the LSM. When a cross-encoder
        // reranker also runs below it re-sorts by semantic score (so this is a
        // no-op for that path); when it does not, the exactly-reranked order is
        // what we return. Skipped on the brute path (already exact).
        if rerank && !use_brute {
            candidates = self.rerank_candidates(&index_key, &index, &query_vec[0], candidates);
        }

        // Find the source field for this vector field.
        let source_field = self
            .schema
            .get_type(type_name)
            .and_then(|td| td.get_field(vector_field))
            .and_then(|fd| fd.vectorize())
            .map(|v| v.source_field.clone());

        // Rerank if enabled and we can read the source text.
        if let Some(source_field) = source_field.filter(|_| !rerank_disabled) {
            let type_id = self.type_ids.get(type_name).copied();

            // Fetch original text for each candidate, all under ONE snapshot.
            let mut candidate_texts: Vec<(u64, String)> = Vec::new();
            if let Some(type_id) = type_id {
                let snapshot = self.storage.read_snapshot();
                for (obj_id, _dist) in &candidates {
                    let obj_key = KeyBuilder::object(type_id, *obj_id);
                    if let Ok(Some(data)) = self.storage.get_at(snapshot, &obj_key) {
                        let fields = deserialize_fields(&data);
                        if let Some(Value::String(text)) = fields.get(&source_field) {
                            candidate_texts.push((*obj_id, text.clone()));
                        }
                    }
                }
            }

            if !candidate_texts.is_empty() {
                // Lazily initialize the reranker.
                let mut reranker = self.reranker.lock();
                if reranker.is_none() {
                    #[cfg(feature = "fastembed")]
                    match FastReranker::new() {
                        Ok(r) => *reranker = Some(Box::new(r)),
                        Err(_) => {
                            // Reranker unavailable — return HNSW results as-is.
                            return Ok(candidates.into_iter().take(k).collect());
                        }
                    }
                    // Still none — either the load above failed, or this build has
                    // no `fastembed` feature (no built-in reranker). Return the
                    // raw HNSW results rather than reranking.
                    if reranker.is_none() {
                        return Ok(candidates.into_iter().take(k).collect());
                    }
                }

                if let Some(ref mut ranker) = *reranker {
                    let doc_refs: Vec<&str> =
                        candidate_texts.iter().map(|(_, t)| t.as_str()).collect();

                    if let Ok(reranked) = ranker.rerank(query_text, &doc_refs, k) {
                        let mut out: Vec<(u64, f32)> = reranked
                            .into_iter()
                            .map(|r| (candidate_texts[r.index].0, r.score))
                            .collect();
                        // The cross-encoder can only rank candidates whose source
                        // text was readable; a candidate with missing/non-string
                        // text is invisible to it. Don't silently drop those —
                        // append any not already chosen (in candidate order, i.e.
                        // exact-distance order on the brute path) up to k, so a
                        // reranked result never under-fills relative to the
                        // no-reranker fallback. No-op when every candidate has
                        // text (the normal @vectorize case), so global searches
                        // are unaffected.
                        if out.len() < k {
                            let chosen: HashSet<u64> = out.iter().map(|(id, _)| *id).collect();
                            for (id, dist) in &candidates {
                                if out.len() >= k {
                                    break;
                                }
                                if !chosen.contains(id) {
                                    out.push((*id, *dist));
                                }
                            }
                        }
                        return Ok(out);
                    }
                }
            }
        }

        // Fallback: return HNSW results without reranking.
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Search a vector index with a raw vector.
    ///
    /// When `rerank` is set, the `k` ANN candidates are re-scored against the
    /// full-precision f32 vectors in the LSM and returned sorted by exact
    /// distance (see [`Vectorizer::rerank_candidates`]). The caller is expected
    /// to have sized `k` to the desired rerank pool and to trim to the final
    /// top-k itself.
    ///
    /// A non-empty `restrict` of at most [`EXACT_FILTER_MAX`] ids takes the exact
    /// brute-force path over that set (see [`Vectorizer::brute_force_restricted`]),
    /// ignoring `ef`/`rerank` (the result is already exact); `restrict = None`
    /// leaves the global HNSW path unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn search_vector(
        &self,
        type_name: &str,
        vector_field: &str,
        query_vec: &[f32],
        k: usize,
        ef: usize,
        rerank: bool,
        restrict: Option<&HashSet<u64>>,
    ) -> EngineResult<Vec<(u64, f32)>> {
        let index_key = format!("{type_name}.{vector_field}");
        let index = self
            .indexes
            .read()
            .get(&index_key)
            .cloned()
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })?;

        // Exact small-set path: a selective filter restricts the search to a
        // bounded set — brute-force exact distances over just those vectors.
        // Exact recall, never under-fills, and (for a small set) cheaper than
        // the graph. `ef`/`rerank` are moot here (the result is already exact).
        if let Some(set) = restrict
            && set.len() <= EXACT_FILTER_MAX
        {
            if std::env::var_os("RHYPEDB_DEBUG_RERANK").is_some() {
                eprintln!("[brute] key={index_key} restrict={} (exact small-set path)", set.len());
            }
            return Ok(self.brute_force_restricted(&index_key, &index, query_vec, set));
        }

        let results = index.search(query_vec, k, ef);
        if rerank {
            Ok(self.rerank_candidates(&index_key, &index, query_vec, results))
        } else {
            Ok(results)
        }
    }

    /// Re-score ANN candidates against the full-precision f32 vectors stored in
    /// the LSM `v:` keyspace and return them sorted ascending by EXACT distance.
    ///
    /// The TurboQuant estimator reconstructs original-scale distances from each
    /// vector's stored norm (see `quantize.rs` `distance_estimate_prepared`), so
    /// the exact `compute_distance(index.metric(), query, f32)` computed here is
    /// the same quantity the ANN approximates — and is exactly what
    /// `brute_force_knn` (the recall ground truth) uses. Reranking therefore
    /// only ever moves results toward the exact top-k.
    ///
    /// Robustness: candidates whose f32 is missing, corrupt, or a mismatched
    /// dimension are scored `f32::INFINITY` so they sort last — they are never
    /// dropped, so a rerank can never under-fill relative to the ANN result. A
    /// single consistent snapshot is used for the whole batch. On a storage
    /// error the ANN order is returned unchanged (rerank is a best-effort recall
    /// boost, not a hard guarantee — failing the query would be worse).
    /// Score a set of object ids against `query_vec` using the index `metric`
    /// and the FULL-PRECISION f32 in the LSM `v:` keyspace. Missing / corrupt /
    /// dimension-mismatched vectors score `f32::INFINITY` (kept, sorted last —
    /// never dropped, so a caller can never under-fill relative to its input).
    /// Sorted ascending by `(distance, id)`: the id tie-break keeps the order
    /// deterministic across f32 ties (common at low bit-widths) and unordered id
    /// sources (a filter's `HashSet`). One consistent LSM snapshot covers the
    /// whole batch. Returns `None` ONLY on a storage error, so the caller can
    /// pick its own fallback (the ANN order, for rerank). Shared by
    /// [`Self::rerank_candidates`] and [`Self::brute_force_restricted`].
    fn exact_rescore(
        &self,
        type_id: u64,
        field_id: u64,
        metric: Metric,
        query_vec: &[f32],
        ids: &[u64],
    ) -> Option<Vec<(u64, f32)>> {
        let snapshot = self.storage.read_snapshot();
        let keys: Vec<Bytes> = ids
            .iter()
            .map(|id| KeyBuilder::vector(type_id, *id, field_id))
            .collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|b| b.as_ref()).collect();
        let vals = self.storage.multi_get_at(snapshot, &key_refs).ok()?;

        let mut scored: Vec<(u64, f32)> = ids
            .iter()
            .zip(vals)
            .map(|(&id, val)| {
                let dist = val
                    .as_deref()
                    .and_then(deserialize_f32_vec)
                    .filter(|v| v.len() == query_vec.len())
                    .map(|v| compute_distance(metric, query_vec, &v))
                    .unwrap_or(f32::INFINITY);
                (id, dist)
            })
            .collect();
        // `total_cmp` is a total order (finite ascending, `INFINITY` last); the
        // `id` tie-break makes the result deterministic.
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Some(scored)
    }

    /// Exact brute-force k-NN over a *restricted* id set (a selective filter).
    /// HNSW post-filtering under-fills when few graph hits fall inside the set;
    /// for a small set it is both exact and cheaper to score every member
    /// directly. Reuses [`Self::exact_rescore`] (same missing→INFINITY +
    /// `(distance, id)` order as rerank), so a `.filter().similar()` over a small
    /// set returns the TRUE top-k within the filter. Returns the full sorted set
    /// — the caller trims to k. On a storage error returns the ids unscored, in
    /// id order (best-effort; never fails the query).
    fn brute_force_restricted(
        &self,
        index_key: &str,
        index: &QuantizedIndex,
        query_vec: &[f32],
        restrict: &HashSet<u64>,
    ) -> Vec<(u64, f32)> {
        let (type_id, field_id) = match self.resolve_index_ids(index_key) {
            Some(ids) => ids,
            None => return Vec::new(),
        };
        let ids: Vec<u64> = restrict.iter().copied().collect();
        self.exact_rescore(type_id, field_id, index.metric(), query_vec, &ids)
            .unwrap_or_else(|| {
                let mut v: Vec<(u64, f32)> =
                    ids.into_iter().map(|id| (id, f32::INFINITY)).collect();
                v.sort_by_key(|(id, _)| *id);
                v
            })
    }

    fn rerank_candidates(
        &self,
        index_key: &str,
        index: &QuantizedIndex,
        query_vec: &[f32],
        candidates: Vec<(u64, f32)>,
    ) -> Vec<(u64, f32)> {
        if candidates.is_empty() {
            return candidates;
        }
        let (type_id, field_id) = match self.resolve_index_ids(index_key) {
            Some(ids) => ids,
            None => return candidates,
        };
        let ids: Vec<u64> = candidates.iter().map(|(id, _)| *id).collect();
        // On a storage error keep the ANN order (best-effort — failing the query
        // would be worse than returning the approximate distances).
        let scored = match self.exact_rescore(type_id, field_id, index.metric(), query_vec, &ids) {
            Some(s) => s,
            None => return candidates,
        };
        if std::env::var_os("RHYPEDB_DEBUG_RERANK").is_some() {
            let missing = scored.iter().filter(|(_, d)| d.is_infinite()).count();
            let finite: Vec<f32> = scored
                .iter()
                .map(|(_, d)| *d)
                .filter(|d| d.is_finite())
                .collect();
            let (min, max) = (
                finite.first().copied().unwrap_or(f32::NAN),
                finite.last().copied().unwrap_or(f32::NAN),
            );
            eprintln!(
                "[rerank] key={index_key} qdim={} n={} missing={missing} dist_min={min:.5} dist_max={max:.5} top3={:?}",
                query_vec.len(),
                scored.len(),
                &scored[..scored.len().min(3)]
            );
        }
        scored
    }

    /// Store one vector under its `v:` key, mark it `Indexed`, and insert it
    /// into the in-memory HNSW index. The single point where a vector becomes
    /// durable + searchable; shared by the embed worker and `ingest_vector`.
    fn store_and_index(
        &self,
        type_name: &str,
        object_id: u64,
        vector_field: &str,
        vector: &[f32],
    ) -> EngineResult<()> {
        let index_key = format!("{type_name}.{vector_field}");
        let index = self
            .indexes
            .read()
            .get(&index_key)
            .cloned()
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })?;
        // `HnswIndex::insert` is safe to call concurrently for the graph structure
        // (per-node locks, no global graph lock), so the embed worker and any
        // in-flight `ingest_vectors` may insert at once without a serializing lock.
        // Re-inserting the same object_id (an update) tombstones the prior node and
        // the new vector wins (HnswIndex::insert handles this atomically) — so
        // dropping the old insert_lock doesn't change update semantics.
        index.insert(object_id, vector);

        let type_id = self.type_ids[type_name];
        let field_id = self.field_ids[&index_key];
        let vector_key = KeyBuilder::vector(type_id, object_id, field_id);
        let state_key = KeyBuilder::vector_state(type_id, object_id, field_id);

        let mut txn = self.storage.begin_txn();
        // The HNSW (in-RAM, rebuilt from `v:` on reopen) is now ahead of the LSM:
        // nothing durable for this object yet. A crash here must recover via the
        // still-`Pending` state being re-enqueued on reopen.
        crash_inject::hit(crash_inject::Site::VectorizeBeforeStoreCommit);
        self.storage
            .put(&mut txn, &vector_key, serialize_f32_vec(vector))?;
        self.storage.put(
            &mut txn,
            &state_key,
            Bytes::from(vec![VectorState::Indexed as u8]),
        )?;
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
            other => crate::EngineError::Storage(other),
        })?;
        // The object is fully durable (`v:` + `Indexed`); a reopen is a no-op.
        crash_inject::hit(crash_inject::Site::VectorizeAfterStoreCommit);
        Ok(())
    }

    /// Expected dimension of a Vector field, or an error if the field isn't a
    /// Vector field on a known type.
    fn vector_field_dim(&self, type_name: &str, vector_field: &str) -> EngineResult<usize> {
        self.schema
            .get_type(type_name)
            .and_then(|td| td.get_field(vector_field))
            .and_then(|fd| match &fd.field_type {
                FieldType::Vector(vt) => Some(vt.dimensions as usize),
                _ => None,
            })
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })
    }

    /// Ingest a caller-supplied (precomputed) vector for an object's Vector
    /// field — bring-your-own-embeddings, no embedding step. Validates the
    /// dimension against the schema, then stores + indexes it synchronously.
    pub fn ingest_vector(
        &self,
        type_name: &str,
        object_id: u64,
        vector_field: &str,
        vector: &[f32],
    ) -> EngineResult<()> {
        let expected = self.vector_field_dim(type_name, vector_field)?;
        validate_vector(vector_field, vector, expected)?;
        self.store_and_index(type_name, object_id, vector_field, vector)
    }

    /// Batch form of [`ingest_vector`]: validate every row's dimension up front
    /// (so a bad row leaves nothing partially applied), then index all rows and
    /// persist their `v:`/state keys in a SINGLE transaction. Returns the count
    /// ingested. Callers chunk large loads across calls to bound txn size.
    pub fn ingest_vectors(
        &self,
        type_name: &str,
        vector_field: &str,
        rows: &[(u64, Vec<f32>)],
    ) -> EngineResult<usize> {
        if rows.is_empty() {
            return Ok(0);
        }
        let index_key = format!("{type_name}.{vector_field}");
        let expected = self.vector_field_dim(type_name, vector_field)?;
        // Validate EVERY row (dimension + finiteness) before any mutation, so a
        // bad row leaves nothing partially applied. Finiteness matters: a NaN/inf
        // component would otherwise reach the quantizer (NaN silently coerced to
        // an all-zero vector; inf -> NaN -> a panic backstopped in quantize.rs).
        for (_object_id, vector) in rows {
            validate_vector(vector_field, vector, expected)?;
        }
        let index = self
            .indexes
            .read()
            .get(&index_key)
            .cloned()
            .ok_or_else(|| crate::EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: vector_field.into(),
            })?;
        let type_id = self.type_ids[type_name];
        let field_id = self.field_ids[&index_key];

        // Build the HNSW edges for the whole batch in parallel across cores
        // (inserts are concurrency-safe), then persist outside the index.
        index.insert_parallel(rows);
        let mut txn = self.storage.begin_txn();
        for (object_id, vector) in rows {
            self.storage.put(
                &mut txn,
                &KeyBuilder::vector(type_id, *object_id, field_id),
                serialize_f32_vec(vector),
            )?;
            self.storage.put(
                &mut txn,
                &KeyBuilder::vector_state(type_id, *object_id, field_id),
                Bytes::from(vec![VectorState::Indexed as u8]),
            )?;
        }
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
            other => crate::EngineError::Storage(other),
        })?;
        Ok(rows.len())
    }

    /// Start background worker threads for vectorization.
    /// Each worker loads its own embedding model (~300MB per worker).
    pub fn start_worker(self: &Arc<Self>, num_workers: usize) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        let mut handles = self.worker_handles.lock();
        for worker_id in 0..num_workers {
            let vectorizer = Arc::clone(self);
            let handle = std::thread::spawn(move || {
                while vectorizer.running.load(Ordering::SeqCst) {
                    let batch = match vectorizer.claim_batch() {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("vectorizer worker {worker_id} claim error: {e}");
                            std::thread::sleep(std::time::Duration::from_millis(500));
                            continue;
                        }
                    };

                    if batch.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }

                    match vectorizer.process_batch(batch) {
                        Ok(_) => {
                            vectorizer.save_snapshots();
                        }
                        Err(e) => {
                            eprintln!("vectorizer worker {worker_id} error: {e}");
                            std::thread::sleep(std::time::Duration::from_millis(500));
                        }
                    }
                }
            });
            handles.push(handle);
        }
    }

    /// Stop all background worker threads and save index snapshots.
    pub fn stop_worker(&self) {
        self.running.store(false, Ordering::SeqCst);
        let handles: Vec<_> = self.worker_handles.lock().drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
        self.save_snapshots();
    }

    /// Check which types/fields have @vectorize configured.
    pub fn vectorize_fields(&self) -> Vec<(String, String, VectorizeDef)> {
        let mut result = Vec::new();
        for type_def in self.schema.types.values() {
            for field in &type_def.fields {
                if let Some(vec_def) = field.vectorize() {
                    result.push((
                        type_def.name.clone(),
                        field.name.clone(),
                        vec_def.clone(),
                    ));
                }
            }
        }
        result
    }

    fn mark_batch_failed(
        &self,
        jobs: &[VectorizeJob],
        _error: &str,
    ) -> EngineResult<()> {
        for job in jobs {
            let type_id = self.type_ids[&job.type_name];
            let field_key = format!("{}.{}", job.type_name, job.vector_field);
            let field_id = self.field_ids[&field_key];
            let state_key = KeyBuilder::vector_state(type_id, job.object_id, field_id);

            let mut txn = self.storage.begin_txn();
            self.storage.put(
                &mut txn,
                &state_key,
                Bytes::from(vec![VectorState::Failed as u8]),
            )?;
            self.storage.commit(&mut txn).map_err(|e| match e {
                rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
                other => crate::EngineError::Storage(other),
            })?;
        }
        Ok(())
    }
}

impl Drop for Vectorizer {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

/// Validate a caller-supplied vector before ingest: correct dimension and all
/// components finite (no NaN/inf). Rejecting non-finite values up front keeps
/// them out of the quantizer/HNSW — a NaN would be silently coerced to an
/// all-zero vector, and an inf would become a NaN inside the quantizer.
fn validate_vector(vector_field: &str, vector: &[f32], expected: usize) -> EngineResult<()> {
    if vector.len() != expected {
        return Err(crate::EngineError::TypeMismatch {
            field: vector_field.into(),
            expected: format!("vector of dimension {expected}"),
            got: format!("vector of dimension {}", vector.len()),
        });
    }
    if !vector.iter().all(|x| x.is_finite()) {
        return Err(crate::EngineError::TypeMismatch {
            field: vector_field.into(),
            expected: "vector with all-finite components".into(),
            got: "vector containing NaN or infinity".into(),
        });
    }
    Ok(())
}

fn serialize_f32_vec(vec: &[f32]) -> Bytes {
    let mut buf = BytesMut::with_capacity(vec.len() * 4);
    for &v in vec {
        buf.put_f32(v);
    }
    buf.freeze()
}

fn deserialize_f32_vec(data: &[u8]) -> Option<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return None;
    }
    let mut vec = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        vec.push(f32::from_be_bytes(chunk.try_into().ok()?));
    }
    Some(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_embed::{EmbedResult, RerankResult};
    use rhypedb_schema::parser::parse_schema;
    use rhypedb_storage::lsm::LsmConfig;

    fn test_setup(dir: &std::path::Path) -> (Arc<LsmTree>, Schema, HashMap<String, u64>, HashMap<String, u64>) {
        let schema = parse_schema(
            r#"
            type Post {
                title: String
                body: String
                embedding: Vector<384> @vectorize(source: "body", model: "all-MiniLM-L6-v2")
            }
            "#,
        )
        .unwrap();

        let config = LsmConfig::new(dir);
        let storage = LsmTree::open(config).unwrap();

        let mut type_ids = HashMap::new();
        type_ids.insert("Post".into(), 1u64);

        let mut field_ids = HashMap::new();
        field_ids.insert("Post.title".into(), 1u64);
        field_ids.insert("Post.body".into(), 2u64);
        field_ids.insert("Post.embedding".into(), 3u64);

        (storage, schema, type_ids, field_ids)
    }

    fn store_object(storage: &LsmTree, type_id: u64, object_id: u64, body: &str) {
        let mut fields = crate::object::FieldMap::new();
        fields.insert("body".into(), Value::String(body.into()));
        let serialized = crate::object::serialize_fields(&fields);
        let key = KeyBuilder::object(type_id, object_id);
        let mut txn = storage.begin_txn();
        storage.put(&mut txn, &key, serialized).unwrap();
        storage.commit(&mut txn).unwrap();
    }

    #[test]
    fn job_serialization_roundtrip() {
        let job = VectorizeJob {
            type_name: "Post".into(),
            object_id: 42,
            source_field: "body".into(),
            vector_field: "embedding".into(),
            model: "all-MiniLM-L6-v2".into(),
        };
        let data = job.serialize();
        let recovered = VectorizeJob::deserialize(&data).unwrap();
        assert_eq!(recovered.type_name, "Post");
        assert_eq!(recovered.object_id, 42);
        assert_eq!(recovered.source_field, "body");
        assert_eq!(recovered.vector_field, "embedding");
        assert_eq!(recovered.model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn enqueue_sets_pending_state() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
        let vectorizer = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        vectorizer
            .enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 1,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();

        let state = vectorizer.get_state("Post", 1, "embedding").unwrap();
        assert_eq!(state, VectorState::Pending);
    }

    /// The crash window: a job is claimed (queue entry deleted + committed) but
    /// the process dies before `store_and_index`, leaving a durable `Pending`
    /// state with no queue entry and no vector. A restart must re-enqueue it
    /// (no manual re-mutation), since the HNSW rebuild only sweeps `v:` keys.
    #[test]
    fn reconcile_re_enqueues_orphaned_pending_job_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
        store_object(&storage, 1, 1, "the quick brown fox");

        {
            let vz = Vectorizer::new(
                Arc::clone(&storage),
                schema.clone(),
                type_ids.clone(),
                field_ids.clone(),
            )
            .unwrap();
            vz.enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 1,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();
            // Claim (deletes + commits the queue entry) WITHOUT processing.
            let claimed = vz.claim_batch().unwrap();
            assert_eq!(claimed.len(), 1, "claim removes the queue entry");
            let snap = storage.read_snapshot();
            let q = storage
                .scan_prefix_at(snap, &KeyBuilder::queue_prefix())
                .unwrap();
            assert!(q.is_empty(), "queue is empty after the claim (the orphan window)");
            assert_eq!(
                vz.get_state("Post", 1, "embedding").unwrap(),
                VectorState::Pending
            );
        }

        // Restart: reconcile must put the orphaned job back on the queue.
        let vz2 = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();
        let snap = storage.read_snapshot();
        let q = storage
            .scan_prefix_at(snap, &KeyBuilder::queue_prefix())
            .unwrap();
        assert_eq!(q.len(), 1, "reconcile re-enqueued the orphaned Pending job");
        let job = VectorizeJob::deserialize(&q[0].1).unwrap();
        assert_eq!(job.object_id, 1);
        assert_eq!(job.type_name, "Post");
        assert_eq!(job.source_field, "body");
        assert_eq!(job.vector_field, "embedding");
        // And it drains cleanly through the normal claim path.
        assert_eq!(vz2.claim_batch().unwrap().len(), 1);
    }

    /// A `Pending` state whose queue entry still exists is a normal backlog job,
    /// not an orphan — a routine restart must NOT duplicate it.
    #[test]
    fn reconcile_does_not_double_enqueue_a_still_queued_job() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
        store_object(&storage, 1, 1, "still pending");
        {
            let vz = Vectorizer::new(
                Arc::clone(&storage),
                schema.clone(),
                type_ids.clone(),
                field_ids.clone(),
            )
            .unwrap();
            vz.enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 1,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();
            // Do NOT claim — the job is legitimately queued.
        }
        let _vz2 = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();
        let snap = storage.read_snapshot();
        let q = storage
            .scan_prefix_at(snap, &KeyBuilder::queue_prefix())
            .unwrap();
        assert_eq!(
            q.len(),
            1,
            "a still-queued pending job must not be duplicated by reconcile"
        );
    }

    /// A `Pending` state whose source object no longer exists (deleted while
    /// pending) is a true orphan the worker would only drop — reconcile must NOT
    /// re-enqueue it, and must clean up the stale state so the scan converges.
    #[test]
    fn reconcile_skips_and_cleans_orphan_for_missing_object() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
        // Object 7 is never stored (simulates a deleted source row).
        {
            let vz = Vectorizer::new(
                Arc::clone(&storage),
                schema.clone(),
                type_ids.clone(),
                field_ids.clone(),
            )
            .unwrap();
            vz.enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 7,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();
            vz.claim_batch().unwrap(); // orphan: queue empty, Pending for a missing object
        }
        let _vz2 = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();
        let snap = storage.read_snapshot();
        let q = storage
            .scan_prefix_at(snap, &KeyBuilder::queue_prefix())
            .unwrap();
        assert!(q.is_empty(), "must not re-enqueue a job for a missing object");
        // The stale Pending state key (type 1, object 7, field 3) was cleaned up.
        let state_key = KeyBuilder::vector_state(1, 7, 3);
        assert!(
            storage.get_at(snap, &state_key).unwrap().is_none(),
            "stale orphan state should be cleaned up"
        );
    }

    // Drives the real fastembed embedding pipeline; only runs with the feature.
    #[cfg(feature = "fastembed")]
    #[test]
    fn process_pending_embeds_and_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());

        // Store an object in the LSM.
        store_object(&storage, 1, 1, "the quick brown fox jumps over the lazy dog");

        let vectorizer = Vectorizer::new(
            Arc::clone(&storage),
            schema,
            type_ids,
            field_ids,
        )
        .unwrap();

        vectorizer
            .enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 1,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();

        let processed = vectorizer.process_pending().unwrap();
        assert_eq!(processed, 1);

        let state = vectorizer.get_state("Post", 1, "embedding").unwrap();
        assert_eq!(state, VectorState::Indexed);
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn search_after_indexing() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());

        // Store multiple objects.
        store_object(&storage, 1, 1, "machine learning and neural networks");
        store_object(&storage, 1, 2, "cooking pasta with tomato sauce");
        store_object(&storage, 1, 3, "deep learning for natural language processing");

        let vectorizer = Vectorizer::new(
            Arc::clone(&storage),
            schema,
            type_ids,
            field_ids,
        )
        .unwrap();

        for id in 1..=3 {
            vectorizer
                .enqueue(VectorizeJob {
                    type_name: "Post".into(),
                    object_id: id,
                    source_field: "body".into(),
                    vector_field: "embedding".into(),
                    model: "all-MiniLM-L6-v2".into(),
                })
                .unwrap();
        }

        vectorizer.process_pending().unwrap();

        // Search for ML-related content.
        let results = vectorizer
            .search_text("Post", "embedding", "artificial intelligence", 2, 50, false, None)
            .unwrap();

        assert_eq!(results.len(), 2);
        // The ML-related posts (1 and 3) should rank above the cooking post (2).
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.contains(&1) || ids.contains(&3),
            "expected ML-related posts in top 2, got {ids:?}"
        );
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn queue_survives_restart() {
        let dir = tempfile::tempdir().unwrap();

        // Enqueue a job, then drop the vectorizer (simulating restart).
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            store_object(&storage, 1, 1, "hello world");

            let vectorizer =
                Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();

            vectorizer
                .enqueue(VectorizeJob {
                    type_name: "Post".into(),
                    object_id: 1,
                    source_field: "body".into(),
                    vector_field: "embedding".into(),
                    model: "all-MiniLM-L6-v2".into(),
                })
                .unwrap();
        }

        // Reopen — the job should still be in the queue.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            let vectorizer =
                Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();

            let processed = vectorizer.process_pending().unwrap();
            assert_eq!(processed, 1);
        }
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn indexed_vectors_survive_restart() {
        let dir = tempfile::tempdir().unwrap();

        // First run: index a document.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            store_object(&storage, 1, 1, "machine learning and neural networks");
            store_object(&storage, 1, 2, "cooking pasta with tomato sauce");

            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            for id in 1..=2 {
                vectorizer
                    .enqueue(VectorizeJob {
                        type_name: "Post".into(),
                        object_id: id,
                        source_field: "body".into(),
                        vector_field: "embedding".into(),
                        model: "all-MiniLM-L6-v2".into(),
                    })
                    .unwrap();
            }

            let processed = vectorizer.process_pending().unwrap();
            assert_eq!(processed, 2);

            // Verify search works.
            let results = vectorizer
                .search_text("Post", "embedding", "artificial intelligence", 1, 50, false, None)
                .unwrap();
            assert!(!results.is_empty(), "search should return results before restart");
        }

        // Second run: vectors should be rebuilt from LSM without re-encoding.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            // Search should work immediately — no need to re-process.
            let results = vectorizer
                .search_text("Post", "embedding", "artificial intelligence", 1, 50, false, None)
                .unwrap();
            assert!(
                !results.is_empty(),
                "search should return results after restart (vectors rebuilt from LSM)"
            );

            // The ML document should rank above the cooking document.
            assert_eq!(results[0].0, 1, "ML document should be the top result");
        }
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn background_worker_processes_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = test_setup(dir.path());

        store_object(&storage, 1, 1, "test document for background worker");

        let vectorizer = Arc::new(
            Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap(),
        );

        vectorizer
            .enqueue(VectorizeJob {
                type_name: "Post".into(),
                object_id: 1,
                source_field: "body".into(),
                vector_field: "embedding".into(),
                model: "all-MiniLM-L6-v2".into(),
            })
            .unwrap();

        vectorizer.start_worker(1);

        // Wait for the worker to process the job.
        let mut attempts = 0;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let state = vectorizer.get_state("Post", 1, "embedding").unwrap();
            if state == VectorState::Indexed {
                break;
            }
            attempts += 1;
            assert!(attempts < 50, "worker didn't process job within 10 seconds");
        }

        vectorizer.stop_worker();
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn snapshot_speeds_up_restart() {
        let dir = tempfile::tempdir().unwrap();

        // First run: index documents, save snapshot.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            store_object(&storage, 1, 1, "machine learning and neural networks");
            store_object(&storage, 1, 2, "cooking pasta with tomato sauce");
            store_object(&storage, 1, 3, "deep learning for natural language processing");

            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            for id in 1..=3 {
                vectorizer
                    .enqueue(VectorizeJob {
                        type_name: "Post".into(),
                        object_id: id,
                        source_field: "body".into(),
                        vector_field: "embedding".into(),
                        model: "all-MiniLM-L6-v2".into(),
                    })
                    .unwrap();
            }

            vectorizer.process_pending().unwrap();
            vectorizer.save_snapshots();

            // Verify snapshot file was created.
            let snapshot_path = dir.path().join("hnsw_Post_embedding.bin");
            assert!(snapshot_path.exists(), "snapshot file should exist");
            assert!(
                snapshot_path.metadata().unwrap().len() > 0,
                "snapshot file should not be empty"
            );
        }

        // Second run: should load from snapshot.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            let results = vectorizer
                .search_text("Post", "embedding", "artificial intelligence", 2, 50, false, None)
                .unwrap();
            assert_eq!(results.len(), 2);
            let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(
                ids.contains(&1) || ids.contains(&3),
                "ML posts should rank high after snapshot restore, got {ids:?}"
            );
        }
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn snapshot_delta_rebuild() {
        let dir = tempfile::tempdir().unwrap();

        // First run: index 2 docs, save snapshot.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            store_object(&storage, 1, 1, "machine learning and neural networks");
            store_object(&storage, 1, 2, "cooking pasta with tomato sauce");

            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            for id in 1..=2 {
                vectorizer
                    .enqueue(VectorizeJob {
                        type_name: "Post".into(),
                        object_id: id,
                        source_field: "body".into(),
                        vector_field: "embedding".into(),
                        model: "all-MiniLM-L6-v2".into(),
                    })
                    .unwrap();
            }
            vectorizer.process_pending().unwrap();
            vectorizer.save_snapshots();
        }

        // Index a 3rd doc without saving a new snapshot.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            store_object(&storage, 1, 3, "deep learning for NLP");

            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            vectorizer
                .enqueue(VectorizeJob {
                    type_name: "Post".into(),
                    object_id: 3,
                    source_field: "body".into(),
                    vector_field: "embedding".into(),
                    model: "all-MiniLM-L6-v2".into(),
                })
                .unwrap();
            vectorizer.process_pending().unwrap();
            // Intentionally NOT saving snapshot.
        }

        // Third run: snapshot has 2, LSM has 3. Delta rebuild should pick up the 3rd.
        {
            let (storage, schema, type_ids, field_ids) = test_setup(dir.path());
            let vectorizer = Vectorizer::new(
                Arc::clone(&storage),
                schema,
                type_ids,
                field_ids,
            )
            .unwrap();

            let status = vectorizer.status();
            let index_stat = status
                .index_stats
                .iter()
                .find(|s| s.name == "Post.embedding")
                .unwrap();
            assert_eq!(
                index_stat.vectors, 3,
                "should have 3 vectors after delta rebuild"
            );
        }
    }

    // --- Bring-your-own-vector (BYO) ingest: a bare Vector field, no embedder ---

    fn byo_setup(
        dir: &std::path::Path,
    ) -> (Arc<LsmTree>, Schema, HashMap<String, u64>, HashMap<String, u64>) {
        // A BARE Vector field — no @vectorize, so no embedder is involved.
        let schema = parse_schema(
            r#"
            type Doc {
                embedding: Vector<4>
            }
            "#,
        )
        .unwrap();
        let storage = LsmTree::open(LsmConfig::new(dir)).unwrap();
        let mut type_ids = HashMap::new();
        type_ids.insert("Doc".into(), 1u64);
        let mut field_ids = HashMap::new();
        field_ids.insert("Doc.embedding".into(), 1u64);
        (storage, schema, type_ids, field_ids)
    }

    #[test]
    fn byo_ingest_and_search_bare_vector_field() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = byo_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        // A bare Vector field gets an HNSW index even without @vectorize.
        v.ingest_vectors(
            "Doc",
            "embedding",
            &[
                (1, vec![1.0, 0.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0, 0.0]),
                (3, vec![0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();

        let results = v
            .search_vector("Doc", "embedding", &[0.9, 0.1, 0.0, 0.0], 1, 16, false, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 1,
            "nearest to [0.9,0.1,0,0] must be id 1, got {results:?}"
        );
        assert_eq!(
            v.get_state("Doc", 1, "embedding").unwrap(),
            VectorState::Indexed
        );
    }

    #[test]
    fn byo_ingest_rebuilds_index_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = byo_setup(dir.path());
        {
            let v = Vectorizer::new(
                Arc::clone(&storage),
                schema.clone(),
                type_ids.clone(),
                field_ids.clone(),
            )
            .unwrap();
            v.ingest_vectors(
                "Doc",
                "embedding",
                &[(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])],
            )
            .unwrap();
        }
        // New vectorizer over the same storage: rebuild_indexes must restore the
        // index from the `v:` keys — no @vectorize, no embedder, no re-ingest.
        let v2 = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();
        let results = v2
            .search_vector("Doc", "embedding", &[0.1, 0.9, 0.0, 0.0], 1, 16, false, None)
            .unwrap();
        assert_eq!(
            results[0].0, 2,
            "after restart, nearest to [0.1,0.9,0,0] must be id 2, got {results:?}"
        );
    }

    #[test]
    fn byo_ingest_rejects_dim_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = byo_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
        // Field is Vector<4>; a 3-dim vector must be rejected.
        assert!(
            v.ingest_vector("Doc", 1, "embedding", &[1.0, 0.0, 0.0]).is_err(),
            "3-dim vector into a Vector<4> field must error"
        );
    }

    #[test]
    fn byo_batch_validates_all_dims_before_applying() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = byo_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
        // Second row has the wrong dim — the whole batch is rejected up front,
        // with nothing partially applied.
        assert!(
            v.ingest_vectors(
                "Doc",
                "embedding",
                &[(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0])],
            )
            .is_err(),
            "batch with a bad-dim row must error"
        );
        let results = v
            .search_vector("Doc", "embedding", &[1.0, 0.0, 0.0, 0.0], 1, 16, false, None)
            .unwrap();
        assert!(
            results.is_empty(),
            "no rows should have been applied, got {results:?}"
        );
    }

    #[test]
    fn byo_ingest_rejects_non_finite() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = byo_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        // Untrusted callers can supply NaN/inf — these must be rejected before
        // reaching the quantizer (inf would panic it; NaN would silently store
        // an all-zero vector), with nothing applied.
        assert!(
            v.ingest_vector("Doc", 1, "embedding", &[f32::INFINITY, 0.0, 0.0, 0.0])
                .is_err(),
            "an +inf component must be rejected"
        );
        assert!(
            v.ingest_vector("Doc", 2, "embedding", &[0.0, f32::NAN, 0.0, 0.0])
                .is_err(),
            "a NaN component must be rejected"
        );
        // Batch: one bad row rejects the whole batch, nothing applied.
        assert!(
            v.ingest_vectors(
                "Doc",
                "embedding",
                &[
                    (1, vec![1.0, 0.0, 0.0, 0.0]),
                    (2, vec![0.0, f32::NEG_INFINITY, 0.0, 0.0]),
                ],
            )
            .is_err(),
            "a batch containing a non-finite row must be rejected"
        );
        let results = v
            .search_vector("Doc", "embedding", &[1.0, 0.0, 0.0, 0.0], 1, 16, false, None)
            .unwrap();
        assert!(
            results.is_empty(),
            "no rows should have been applied after rejection, got {results:?}"
        );
    }

    // --- Full-precision rerank ---

    fn rerank_setup(
        dir: &std::path::Path,
    ) -> (Arc<LsmTree>, Schema, HashMap<String, u64>, HashMap<String, u64>) {
        let schema = parse_schema(
            r#"
            type Doc {
                embedding: Vector<16>
            }
            "#,
        )
        .unwrap();
        let storage = LsmTree::open(LsmConfig::new(dir)).unwrap();
        let mut type_ids = HashMap::new();
        type_ids.insert("Doc".into(), 1u64);
        let mut field_ids = HashMap::new();
        field_ids.insert("Doc.embedding".into(), 1u64);
        (storage, schema, type_ids, field_ids)
    }

    /// Deterministic, tie-free 16-d vectors so the test is reproducible.
    fn synth_vec(seed: f32) -> Vec<f32> {
        (0..16).map(|j| (seed * 0.7 + j as f32 * 1.3).sin()).collect()
    }

    #[test]
    fn rerank_reproduces_exact_cosine_order() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        let n = 40u64;
        let rows: Vec<(u64, Vec<f32>)> = (1..=n).map(|i| (i, synth_vec(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let query = synth_vec(3.5);

        // Ground truth: exact cosine distance over ALL vectors (the index metric
        // is hardcoded Cosine — see Vectorizer::new). This is precisely what
        // rerank must reproduce.
        let mut exact: Vec<(u64, f32)> = rows
            .iter()
            .map(|(id, vec)| (*id, compute_distance(Metric::Cosine, &query, vec)))
            .collect();
        exact.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Retrieve a pool covering every vector (k=n, large ef) and rerank it.
        // With the pool >= n the reranked order IS the exact sort, so the whole
        // distance sequence must match the brute-force ground truth. Comparing
        // distances (not ids) is immune to ambiguous ordering at exact ties.
        let reranked = v
            .search_vector("Doc", "embedding", &query, n as usize, 256, true, None)
            .unwrap();
        assert_eq!(reranked.len(), n as usize, "rerank must keep the full pool");
        for (r, e) in reranked.iter().zip(exact.iter()) {
            assert!(
                (r.1 - e.1).abs() < 1e-6,
                "reranked distance {} != exact {}",
                r.1,
                e.1
            );
        }
        assert_eq!(reranked[0].0, exact[0].0, "top-1 id must match the exact NN");
    }

    // Reproduces the benchmark conditions in-process: a higher-dim index, a
    // candidate pool SMALLER than the index, and recall measured against an
    // in-test brute-force ground truth. Rerank over a pool must NEVER score
    // below the plain ANN top-k drawn from the same exploration width.
    #[test]
    fn rerank_recall_beats_or_matches_ann_on_subset_pool() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema("type Doc {\n embedding: Vector<64>\n}").unwrap();
        let storage = LsmTree::open(LsmConfig::new(dir.path())).unwrap();
        let mut type_ids = HashMap::new();
        type_ids.insert("Doc".into(), 1u64);
        let mut field_ids = HashMap::new();
        field_ids.insert("Doc.embedding".into(), 1u64);
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        // 2000 deterministic unit vectors in 64-d (a few loose clusters).
        let dims = 64usize;
        let n = 2000u64;
        let unit = |seed: f32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dims)
                .map(|j| (seed * 0.13 + j as f32 * 0.37).sin() + (seed * 0.05).cos())
                .collect();
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.iter().map(|x| x / norm).collect()
        };
        let rows: Vec<(u64, Vec<f32>)> = (1..=n).map(|i| (i, unit(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let k = 10usize;
        let pool = 200usize; // candidate pool << n, like the bench
        let mut q_seed = 0.5f32;
        let (mut ann_hits, mut rr_hits, mut total) = (0usize, 0usize, 0usize);
        for _ in 0..40 {
            q_seed += 1.7;
            let query = unit(q_seed);

            // Brute-force cosine ground truth over ALL vectors.
            let mut gt: Vec<(u64, f32)> = rows
                .iter()
                .map(|(id, vec)| (*id, compute_distance(Metric::Cosine, &query, vec)))
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_set: std::collections::HashSet<u64> =
                gt.iter().take(k).map(|(id, _)| *id).collect();

            // Plain ANN top-k vs rerank over a `pool`-sized candidate set, both
            // at the SAME exploration width (ef = pool).
            let ann = v
                .search_vector("Doc", "embedding", &query, k, pool, false, None)
                .unwrap();
            let rr = v
                .search_vector("Doc", "embedding", &query, pool, pool, true, None)
                .unwrap();
            ann_hits += ann.iter().take(k).filter(|(id, _)| gt_set.contains(id)).count();
            rr_hits += rr.iter().take(k).filter(|(id, _)| gt_set.contains(id)).count();
            total += k;
        }
        let ann_recall = ann_hits as f32 / total as f32;
        let rr_recall = rr_hits as f32 / total as f32;
        eprintln!("subset-pool recall: ann={ann_recall:.4} rerank={rr_recall:.4}");
        assert!(
            rr_recall >= ann_recall - 1e-6,
            "rerank recall {rr_recall} must not drop below plain ANN {ann_recall}"
        );
    }

    #[test]
    fn rerank_never_underfills_when_pool_exceeds_index() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        let rows: Vec<(u64, Vec<f32>)> = (1..=5u64).map(|i| (i, synth_vec(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        // Ask for a far larger pool than the index holds: rerank must return all
        // 5 (no panic, no drops), sorted by exact distance. The query equals
        // id 2's vector, so id 2 (cosine distance 0) must rank first.
        let reranked = v
            .search_vector("Doc", "embedding", &synth_vec(2.0), 100, 200, true, None)
            .unwrap();
        assert_eq!(reranked.len(), 5);
        assert_eq!(reranked[0].0, 2, "exact nearest to query(seed=2) is id 2");
    }

    // --- @index SDL directive → per-index config (Knob A) ---

    /// `Doc { embedding: Vector<16> ... }` with a caller-supplied `@index`.
    fn index_setup(
        dir: &std::path::Path,
        schema_str: &str,
    ) -> (Arc<LsmTree>, Schema, HashMap<String, u64>, HashMap<String, u64>) {
        let schema = parse_schema(schema_str).unwrap();
        let storage = LsmTree::open(LsmConfig::new(dir)).unwrap();
        let mut type_ids = HashMap::new();
        type_ids.insert("Doc".into(), 1u64);
        let mut field_ids = HashMap::new();
        field_ids.insert("Doc.embedding".into(), 1u64);
        (storage, schema, type_ids, field_ids)
    }

    fn quant_bits_of(v: &Vectorizer, key: &str) -> u8 {
        v.indexes.read().get(key).unwrap().quant_bits()
    }

    #[test]
    fn resolve_index_config_defaults_to_legacy() {
        // Absent @index MUST resolve to exactly the legacy hardcoded config, or
        // every deployed index would mismatch its snapshot and rebuild on upgrade.
        let (hnsw, quant) = resolve_index_config(384, None);
        assert_eq!(quant.bits, LEGACY_QUANT_BITS);
        assert_eq!(quant.dimensions, 384);
        assert_eq!(hnsw.m, LEGACY_HNSW_M);
        assert_eq!(hnsw.m_max0, LEGACY_HNSW_M * 2);
        assert_eq!(hnsw.ef_construction, LEGACY_HNSW_EF_CONSTRUCTION);
        assert_eq!(hnsw.metric, LEGACY_METRIC);
    }

    #[test]
    fn index_directive_sets_quant_bits() {
        for (q, want) in [
            ("turboquant_2bit", 2u8),
            ("turboquant_3bit", 3),
            ("turboquant_4bit", 4),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let schema =
                format!("type Doc {{ embedding: Vector<16> @index(hnsw, quantization: {q}) }}");
            let (storage, schema, type_ids, field_ids) = index_setup(dir.path(), &schema);
            let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
            assert_eq!(quant_bits_of(&v, "Doc.embedding"), want, "quantization {q}");
        }
    }

    #[test]
    fn index_directive_sets_metric_and_hnsw_params() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = index_setup(
            dir.path(),
            "type Doc { embedding: Vector<16> @index(hnsw, metric: l2, m: 8, ef_construction: 64) }",
        );
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
        let idx = v.indexes.read().get("Doc.embedding").unwrap().clone();
        assert_eq!(idx.metric(), Metric::L2);
        assert_eq!(idx.hnsw_m(), 8);
        assert_eq!(idx.hnsw_ef_construction(), 64);
        // bits omitted → legacy default
        assert_eq!(idx.quant_bits(), LEGACY_QUANT_BITS);
    }

    #[test]
    fn index_config_mismatch_detects_difference() {
        // Two indexes built from the same (default) config never mismatch.
        let (h1, q1) = resolve_index_config(16, None);
        let (h2, q2) = resolve_index_config(16, None);
        let a = QuantizedIndex::new(h1, q1);
        let b = QuantizedIndex::new(h2, q2);
        assert!(index_config_mismatch(&a, &b).is_none());

        // A different bit-width is detected as a mismatch.
        let c = QuantizedIndex::new(
            HnswConfig { m: 16, m_max0: 32, ef_construction: 100, metric: Metric::Cosine },
            TurboQuantConfig::new(16, 2),
        );
        assert!(index_config_mismatch(&a, &c).is_some());

        // So is a different metric.
        let d = QuantizedIndex::new(
            HnswConfig { m: 16, m_max0: 32, ef_construction: 100, metric: Metric::L2 },
            TurboQuantConfig::new(16, 4),
        );
        assert!(index_config_mismatch(&a, &d).is_some());
    }

    #[test]
    fn config_change_rebuilds_index_from_lsm() {
        let dir = tempfile::tempdir().unwrap();
        let rows: Vec<(u64, Vec<f32>)> = (1..=20u64).map(|i| (i, synth_vec(i as f32))).collect();

        // First run: default (4-bit) index, ingest 20 vectors, save the snapshot.
        {
            let (storage, schema, type_ids, field_ids) =
                index_setup(dir.path(), "type Doc { embedding: Vector<16> }");
            let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
            assert_eq!(quant_bits_of(&v, "Doc.embedding"), 4);
            v.ingest_vectors("Doc", "embedding", &rows).unwrap();
            v.save_snapshots();
        }

        // Second run: the SDL now asks for 2-bit. The 4-bit snapshot mismatches
        // the schema, so the index must be dropped and rebuilt from the LSM f32
        // at 2-bit — without losing any vectors.
        {
            let (storage, schema, type_ids, field_ids) = index_setup(
                dir.path(),
                "type Doc { embedding: Vector<16> @index(hnsw, quantization: turboquant_2bit) }",
            );
            let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();
            assert_eq!(
                quant_bits_of(&v, "Doc.embedding"),
                2,
                "index should have been rebuilt at the new 2-bit width"
            );
            let stat = v
                .status()
                .index_stats
                .into_iter()
                .find(|s| s.name == "Doc.embedding")
                .unwrap();
            assert_eq!(stat.vectors, 20, "all vectors must survive the rebuild");
            // The rebuilt index is functional: the self-match query ranks first.
            let hits = v
                .search_vector("Doc", "embedding", &synth_vec(5.0), 3, 64, false, None)
                .unwrap();
            assert!(hits.iter().any(|(id, _)| *id == 5), "got {hits:?}");
        }
    }

    // --- Exact small-set brute-force for filtered .similar() (Knob B) ---

    #[test]
    fn filtered_search_is_exact_within_restrict_set() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        let rows: Vec<(u64, Vec<f32>)> = (1..=30u64).map(|i| (i, synth_vec(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let query = synth_vec(3.5);
        let restrict: HashSet<u64> = [8, 17, 26].into_iter().collect();

        // Independently compute the exact order over the restrict set.
        let mut expected: Vec<(u64, f32)> = restrict
            .iter()
            .map(|&id| (id, compute_distance(Metric::Cosine, &query, &synth_vec(id as f32))))
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let got = v
            .search_vector("Doc", "embedding", &query, 2, 64, false, Some(&restrict))
            .unwrap();

        let got_ids: Vec<u64> = got.iter().map(|(id, _)| *id).collect();
        let exp_ids: Vec<u64> = expected.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            got_ids, exp_ids,
            "filtered brute-force order must equal the exact order"
        );
    }

    #[test]
    fn filtered_search_does_not_underfill_when_global_topk_misses_the_set() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        let rows: Vec<(u64, Vec<f32>)> = (1..=30u64).map(|i| (i, synth_vec(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let query = synth_vec(3.5);
        // synth_vec is periodic in seed (period ~8.98); {8,17,26} are ~anti-phase
        // to 3.5 → the farthest vectors, so a global ANN search never surfaces
        // them. This is exactly the selective filter that the old over-fetch +
        // post-filter path under-fills on.
        let restrict: HashSet<u64> = [8, 17, 26].into_iter().collect();

        let global = v
            .search_vector("Doc", "embedding", &query, 8, 64, false, None)
            .unwrap();
        let global_ids: HashSet<u64> = global.iter().map(|(id, _)| *id).collect();
        assert!(
            restrict.is_disjoint(&global_ids),
            "precondition: restrict set must lie outside the global top-8, got {global_ids:?}"
        );

        // The exact small-set path still returns every member of the set (the
        // caller, run_similar, then trims to the final k). No under-fill.
        let got = v
            .search_vector("Doc", "embedding", &query, 2, 64, false, Some(&restrict))
            .unwrap();
        let got_ids: HashSet<u64> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            got_ids, restrict,
            "filtered search must return the set's members, not under-fill to the global top-k"
        );
    }

    #[test]
    fn filtered_search_keeps_missing_vector_member_as_infinity() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        // Ingest 1..=5; the restrict set also names id 99, which has no vector.
        let rows: Vec<(u64, Vec<f32>)> = (1..=5u64).map(|i| (i, synth_vec(i as f32))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let query = synth_vec(2.0);
        let restrict: HashSet<u64> = [2, 4, 99].into_iter().collect();
        let got = v
            .search_vector("Doc", "embedding", &query, 3, 64, false, Some(&restrict))
            .unwrap();

        // All three present (never dropped); id 99 (no vector) scores INFINITY
        // and sorts last — parity with rerank's never-under-fill rule. id 2 is
        // the exact self-match to the query.
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, 2, "self-match is nearest");
        assert_eq!(got.last().unwrap().0, 99, "missing-vector member sorts last");
        assert!(got.last().unwrap().1.is_infinite());
    }

    #[test]
    fn filtered_search_is_deterministic_on_ties() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = rerank_setup(dir.path());
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        // ids 10 and 20 share the SAME vector → identical distance to any query.
        let shared = synth_vec(7.0);
        let mut rows: Vec<(u64, Vec<f32>)> = (1..=5u64).map(|i| (i, synth_vec(i as f32))).collect();
        rows.push((10, shared.clone()));
        rows.push((20, shared.clone()));
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let restrict: HashSet<u64> = [10, 20].into_iter().collect();
        // The (distance, id) tie-break must order id 10 before id 20 every run,
        // despite the unordered HashSet source and the exact distance tie.
        for _ in 0..8 {
            let got = v
                .search_vector("Doc", "embedding", &shared, 2, 64, false, Some(&restrict))
                .unwrap();
            let ids: Vec<u64> = got.iter().map(|(id, _)| *id).collect();
            assert_eq!(ids, vec![10, 20], "tie-break by id must be deterministic");
        }
    }

    /// Latency of the exact small-set path at 384-d (the bible-app dimension)
    /// across restrict-set sizes, to justify `EXACT_FILTER_MAX`. Not a CI gate.
    /// Run: `cargo test -p rhypedb-engine --release bench_brute_force_restricted_latency -- --ignored --nocapture`
    #[test]
    #[ignore = "latency benchmark; run in --release with --ignored --nocapture"]
    fn bench_brute_force_restricted_latency() {
        use std::time::Instant;
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) =
            index_setup(dir.path(), "type Doc { embedding: Vector<384> }");
        let v = Vectorizer::new(storage, schema, type_ids, field_ids).unwrap();

        let dim = 384usize;
        let vec_at = |seed: u64| -> Vec<f32> {
            (0..dim).map(|j| ((seed as f32) * 0.7 + j as f32 * 1.3).sin()).collect()
        };
        let n = 10_000u64;
        let rows: Vec<(u64, Vec<f32>)> = (1..=n).map(|i| (i, vec_at(i))).collect();
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();

        let query = vec_at(424_242);
        for &size in &[100usize, 1_000, 10_000] {
            let restrict: HashSet<u64> = (1..=size as u64).collect();
            let _ = v
                .search_vector("Doc", "embedding", &query, 10, 64, false, Some(&restrict))
                .unwrap(); // warm
            let iters = 20;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = v
                    .search_vector("Doc", "embedding", &query, 10, 64, false, Some(&restrict))
                    .unwrap();
            }
            let per_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!("brute restrict={size:>6} dim={dim}: {per_ms:.3} ms/query");
        }
    }

    // --- Filtered text search through the cross-encoder (mock models) ---
    //
    // A mock embedder + reranker drive `search_text`'s cross-encoder path
    // deterministically WITHOUT loading any ONNX model, by injecting into the
    // (test-visible) `embedders` / `reranker` fields.

    struct MockEmbedder {
        vec: Vec<f32>,
    }
    impl Embedder for MockEmbedder {
        fn embed(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| self.vec.clone()).collect())
        }
        fn dimensions(&self) -> usize {
            self.vec.len()
        }
        fn model_name(&self) -> &str {
            "mock"
        }
    }

    /// Ranks documents in the order received (index 0 best) — so the result
    /// order is the candidate (exact-distance) order, deterministic + model-free.
    struct MockReranker;
    impl Reranker for MockReranker {
        fn rerank(
            &mut self,
            _query: &str,
            documents: &[&str],
            top_k: usize,
        ) -> EmbedResult<Vec<RerankResult>> {
            let n = documents.len().min(top_k);
            Ok((0..n)
                .map(|i| RerankResult {
                    index: i,
                    score: (documents.len() - i) as f32,
                })
                .collect())
        }
    }

    fn vectorize_setup(
        dir: &std::path::Path,
    ) -> (Arc<LsmTree>, Schema, HashMap<String, u64>, HashMap<String, u64>) {
        let schema = parse_schema(
            r#"
            type Doc {
                body: String
                embedding: Vector<4> @vectorize(source: "body", model: "mock")
            }
            "#,
        )
        .unwrap();
        let storage = LsmTree::open(LsmConfig::new(dir)).unwrap();
        let mut type_ids = HashMap::new();
        type_ids.insert("Doc".into(), 1u64);
        let mut field_ids = HashMap::new();
        field_ids.insert("Doc.body".into(), 1u64);
        field_ids.insert("Doc.embedding".into(), 2u64);
        (storage, schema, type_ids, field_ids)
    }

    fn inject_mocks(v: &Vectorizer, query: Vec<f32>) {
        v.embedders
            .lock()
            .insert("mock".into(), Box::new(MockEmbedder { vec: query }));
        *v.reranker.lock() = Some(Box::new(MockReranker));
    }

    #[test]
    fn filtered_text_search_keeps_textless_candidate_via_append() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = vectorize_setup(dir.path());
        let v = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();

        // ids 1,2,3 all get vectors; only 1 and 2 get source text. id 3 has a
        // vector but NO body → invisible to the cross-encoder.
        let rows = vec![
            (1u64, vec![1.0f32, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ];
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();
        store_object(&storage, 1, 1, "first doc");
        store_object(&storage, 1, 2, "second doc");
        // id 3: intentionally no object/body.

        inject_mocks(&v, vec![1.0, 0.0, 0.0, 0.0]);

        let restrict: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let got = v
            .search_text("Doc", "embedding", "q", 3, 64, false, Some(&restrict))
            .unwrap();

        let ids: HashSet<u64> = got.iter().map(|(id, _)| *id).collect();
        // Without the append-fill, the cross-encoder would drop id 3 (no text)
        // and return only {1,2}. The result must still contain the textless
        // member, so a reranked result never under-fills.
        assert_eq!(
            ids, restrict,
            "textless candidate (id 3) must be appended, not dropped"
        );
    }

    #[test]
    fn filtered_text_search_runs_cross_encoder_over_brute_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let (storage, schema, type_ids, field_ids) = vectorize_setup(dir.path());
        let v = Vectorizer::new(Arc::clone(&storage), schema, type_ids, field_ids).unwrap();

        let rows = vec![
            (1u64, vec![1.0f32, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
            (4, vec![0.0, 0.0, 0.0, 1.0]),
        ];
        v.ingest_vectors("Doc", "embedding", &rows).unwrap();
        for (id, _) in &rows {
            store_object(&storage, 1, *id, &format!("doc {id}"));
        }
        inject_mocks(&v, vec![1.0, 0.0, 0.0, 0.0]);

        // Restrict to a selective subset; the cross-encoder ranks exactly those.
        let restrict: HashSet<u64> = [2, 4].into_iter().collect();
        let got = v
            .search_text("Doc", "embedding", "q", 5, 64, false, Some(&restrict))
            .unwrap();
        let ids: HashSet<u64> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, restrict, "filtered text search returns the filter's members");
    }
}

// Crash-recovery fuzz harness for the vectorize queue + HNSW rebuild. It is a
// submodule of `vectorizer` (so it reaches the module-private key/serialization
// helpers and the `embedders` field directly — no public surface widening) and
// only compiles with both `cfg(test)` and the `crash-fuzz` feature, which makes
// the four `Vectorize*` injection sites live and `catch_crash`/`arm` available.
#[cfg(all(test, feature = "crash-fuzz"))]
#[path = "vectorizer_crash_fuzz.rs"]
mod vectorizer_crash_fuzz;
