use std::collections::HashMap;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};

use rhypedb_embed::{Embedder, FastEmbedder, FastReranker, Reranker};
use rhypedb_schema::{FieldType, Schema, VectorizeDef};
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::LsmTree;
use rhypedb_vector::distance::Metric;
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

impl Vectorizer {
    pub fn new(
        storage: Arc<LsmTree>,
        schema: Schema,
        type_ids: HashMap<String, u64>,
        field_ids: HashMap<String, u64>,
    ) -> EngineResult<Self> {
        // Create HNSW indexes for each @vectorize field.
        let mut indexes = HashMap::new();

        for type_def in schema.types.values() {
            for field in &type_def.fields {
                if let Some(_vec_def) = field.vectorize()
                    && let FieldType::Vector(vt) = &field.field_type {
                        let index_key = format!("{}.{}", type_def.name, field.name);
                        let hnsw_config = HnswConfig {
                            m: 16,
                            m_max0: 32,
                            ef_construction: 100,
                            metric: Metric::Cosine,
                        };
                        let quant_config = TurboQuantConfig::new(vt.dimensions, 3);
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

        Ok(vectorizer)
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
        let mut delta = 0usize;

        for (object_id, vector) in &vectors {
            if !index.contains_id(*object_id) {
                index.insert(*object_id, vector);
                delta += 1;
            }
        }

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
        for (object_id, vector) in &vectors {
            index.insert(*object_id, vector);
        }

        Ok(vectors.len())
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
        let mut embedders = self.embedders.lock();
        self.process_batch(batch, &mut embedders)
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

        Ok(jobs
            .into_iter()
            .map(|(_, job)| {
                let object_id = job.object_id;
                (job, object_id)
            })
            .collect())
    }

    /// Process a claimed batch of jobs with the given embedders.
    fn process_batch(
        &self,
        jobs: Vec<(VectorizeJob, u64)>,
        embedders: &mut HashMap<String, Box<dyn Embedder>>,
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

            let embedder = embedders
                .entry(model_name.clone())
                .or_insert_with(|| {
                    Box::new(FastEmbedder::new(model_name).expect("failed to load model"))
                });
            let embeddings = match embedder.embed(&text_refs) {
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

                let index_key = format!("{}.{}", job.type_name, job.vector_field);
                if let Some(index) = self.indexes.read().get(&index_key) {
                    index.insert(job.object_id, embedding);
                }

                let type_id = self.type_ids[&job.type_name];
                let field_key = format!("{}.{}", job.type_name, job.vector_field);
                let field_id = self.field_ids[&field_key];

                let vector_key = KeyBuilder::vector(type_id, job.object_id, field_id);
                let vector_bytes = serialize_f32_vec(embedding);
                let state_key =
                    KeyBuilder::vector_state(type_id, job.object_id, field_id);

                let mut txn = self.storage.begin_txn();
                self.storage.put(&mut txn, &vector_key, vector_bytes)?;
                self.storage.put(
                    &mut txn,
                    &state_key,
                    Bytes::from(vec![VectorState::Indexed as u8]),
                )?;
                self.storage.commit(&mut txn).map_err(|e| match e {
                    rhypedb_storage::Error::WriteConflict => crate::EngineError::WriteConflict,
                    other => crate::EngineError::Storage(other),
                })?;

                processed += 1;
            }
        }

        Ok(processed)
    }

    /// Search a vector index with a text query (encodes text first).
    pub fn search_text(
        &self,
        type_name: &str,
        vector_field: &str,
        query_text: &str,
        k: usize,
        ef: usize,
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
            let embedder = embedders
                .entry(model.clone())
                .or_insert_with(|| {
                    Box::new(FastEmbedder::new(&model).expect("failed to load model"))
                });
            embedder
                .embed(&[query_text])
                .map_err(|e| crate::EngineError::TypeNotFound(e.to_string()))?
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
        let candidates = index.search(&query_vec[0], retrieval_k, ef.max(retrieval_k));

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

            // Fetch original text for each candidate.
            let mut candidate_texts: Vec<(u64, String)> = Vec::new();
            if let Some(type_id) = type_id {
                for (obj_id, _dist) in &candidates {
                    let obj_key = KeyBuilder::object(type_id, *obj_id);
                    let snapshot = self.storage.read_snapshot();
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
                    match FastReranker::new() {
                        Ok(r) => *reranker = Some(Box::new(r)),
                        Err(_) => {
                            // Reranker unavailable — return HNSW results as-is.
                            return Ok(candidates.into_iter().take(k).collect());
                        }
                    }
                }

                if let Some(ref mut ranker) = *reranker {
                    let doc_refs: Vec<&str> =
                        candidate_texts.iter().map(|(_, t)| t.as_str()).collect();

                    if let Ok(reranked) = ranker.rerank(query_text, &doc_refs, k) {
                        return Ok(reranked
                            .into_iter()
                            .map(|r| (candidate_texts[r.index].0, r.score))
                            .collect());
                    }
                }
            }
        }

        // Fallback: return HNSW results without reranking.
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Search a vector index with a raw vector.
    pub fn search_vector(
        &self,
        type_name: &str,
        vector_field: &str,
        query_vec: &[f32],
        k: usize,
        ef: usize,
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

        Ok(index.search(query_vec, k, ef))
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
                let mut local_embedders: HashMap<String, Box<dyn Embedder>> =
                    HashMap::new();
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

                    match vectorizer.process_batch(batch, &mut local_embedders) {
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
        let storage = Arc::new(LsmTree::open(config).unwrap());

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
            .search_text("Post", "embedding", "artificial intelligence", 2, 50)
            .unwrap();

        assert_eq!(results.len(), 2);
        // The ML-related posts (1 and 3) should rank above the cooking post (2).
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.contains(&1) || ids.contains(&3),
            "expected ML-related posts in top 2, got {ids:?}"
        );
    }

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
                .search_text("Post", "embedding", "artificial intelligence", 1, 50)
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
                .search_text("Post", "embedding", "artificial intelligence", 1, 50)
                .unwrap();
            assert!(
                !results.is_empty(),
                "search should return results after restart (vectors rebuilt from LSM)"
            );

            // The ML document should rank above the cooking document.
            assert_eq!(results[0].0, 1, "ML document should be the top result");
        }
    }

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
                .search_text("Post", "embedding", "artificial intelligence", 2, 50)
                .unwrap();
            assert_eq!(results.len(), 2);
            let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(
                ids.contains(&1) || ids.contains(&3),
                "ML posts should rank high after snapshot restore, got {ids:?}"
            );
        }
    }

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
}
