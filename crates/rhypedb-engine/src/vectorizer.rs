use std::collections::HashMap;
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
    worker_handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
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
        let txn = storage.begin_txn();
        let prefix = KeyBuilder::queue_prefix();
        if let Ok(entries) = storage.scan_prefix(&txn, &prefix) {
            for (key, _) in &entries {
                if key.len() >= 10 {
                    let id_bytes: [u8; 8] = key[2..10].try_into().unwrap();
                    let job_id = u64::from_be_bytes(id_bytes);
                    max_job_id = max_job_id.max(job_id);
                }
            }
        }
        drop(txn);

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
            worker_handle: parking_lot::Mutex::new(None),
        };

        vectorizer.rebuild_indexes()?;

        Ok(vectorizer)
    }

    /// Rebuild HNSW indexes from persisted vectors in the LSM.
    /// Called during startup to restore the in-memory HNSW graph from
    /// vectors that were previously indexed and durably stored.
    fn rebuild_indexes(&self) -> EngineResult<()> {
        let indexes = self.indexes.read();
        if indexes.is_empty() {
            return Ok(());
        }

        // For each vectorize field, scan its persisted vectors and reinsert into HNSW.
        for type_def in self.schema.types.values() {
            for field in &type_def.fields {
                if field.vectorize().is_none() {
                    continue;
                }
                let index_key = format!("{}.{}", type_def.name, field.name);
                let index = match indexes.get(&index_key) {
                    Some(idx) => idx,
                    None => continue,
                };

                let type_id = match self.type_ids.get(&type_def.name) {
                    Some(&id) => id,
                    None => continue,
                };
                let field_key = format!("{}.{}", type_def.name, field.name);
                let field_id = match self.field_ids.get(&field_key) {
                    Some(&id) => id,
                    None => continue,
                };

                // Scan all vector entries for this type.
                let prefix = KeyBuilder::vector_prefix(type_id);
                let txn = self.storage.begin_txn();
                let entries = self.storage.scan_prefix(&txn, &prefix)?;
                drop(txn);

                let mut count = 0usize;
                for (key, data) in &entries {
                    // Vector key: v:<type_id>:<object_id>:<field_id>
                    // Extract object_id and field_id from the key.
                    // Key structure after prefix: <object_id (8 bytes)>:<field_id (8 bytes)>
                    if key.len() < 2 + 8 + 1 + 8 + 1 + 8 {
                        continue;
                    }

                    // Parse out the field_id from the last 8 bytes to verify it matches.
                    let key_field_id_bytes: [u8; 8] =
                        key[key.len() - 8..].try_into().unwrap();
                    let key_field_id = u64::from_be_bytes(key_field_id_bytes);
                    if key_field_id != field_id {
                        continue;
                    }

                    // Extract object_id: 8 bytes before the separator + field_id.
                    let obj_id_start = key.len() - 8 - 1 - 8;
                    let obj_id_bytes: [u8; 8] =
                        key[obj_id_start..obj_id_start + 8].try_into().unwrap();
                    let object_id = u64::from_be_bytes(obj_id_bytes);

                    // Deserialize the f32 vector.
                    if let Some(vector) = deserialize_f32_vec(data) {
                        index.insert(object_id, &vector);
                        count += 1;
                    }
                }

                if count > 0 {
                    eprintln!(
                        "rebuilt HNSW index for {}.{}: {} vectors",
                        type_def.name, field.name, count
                    );
                }
            }
        }

        Ok(())
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

        let txn = self.storage.begin_txn();
        match self.storage.get(&txn, &state_key)? {
            Some(data) if !data.is_empty() => Ok(VectorState::from(data[0])),
            _ => Ok(VectorState::Pending),
        }
    }

    /// Process all pending jobs in the queue. Called by the background worker
    /// or directly for synchronous processing in tests.
    pub fn process_pending(&self) -> EngineResult<usize> {
        let txn = self.storage.begin_txn();
        let prefix = KeyBuilder::queue_prefix();
        let entries = self.storage.scan_prefix(&txn, &prefix)?;
        drop(txn);

        if entries.is_empty() {
            return Ok(0);
        }

        // Group jobs by model for batch encoding.
        let mut jobs_by_model: HashMap<String, Vec<(Bytes, VectorizeJob)>> = HashMap::new();
        for (key, value) in entries {
            if let Some(job) = VectorizeJob::deserialize(&value) {
                jobs_by_model
                    .entry(job.model.clone())
                    .or_default()
                    .push((key, job));
            }
        }

        let mut processed = 0;

        for (model_name, jobs) in &jobs_by_model {
            // Get or create embedder for this model.
            let texts: Vec<String> = jobs
                .iter()
                .filter_map(|(_, job)| {
                    let type_id = self.type_ids.get(&job.type_name)?;
                    let obj_key = KeyBuilder::object(*type_id, job.object_id);
                    let txn = self.storage.begin_txn();
                    let data = self.storage.get(&txn, &obj_key).ok()??;
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

            // Embed the batch.
            let embeddings = {
                let mut embedders = self.embedders.lock();
                let embedder = embedders
                    .entry(model_name.clone())
                    .or_insert_with(|| {
                        Box::new(FastEmbedder::new(model_name).expect("failed to load model"))
                    });
                embedder.embed(&text_refs)
            };

            let embeddings = match embeddings {
                Ok(e) => e,
                Err(e) => {
                    self.mark_jobs_failed(jobs, &format!("{e}"))?;
                    continue;
                }
            };

            // Insert each embedding into the HNSW index, persist to LSM, and update state.
            for (emb_idx, (queue_key, job)) in jobs.iter().enumerate() {
                if emb_idx >= embeddings.len() {
                    break;
                }

                let embedding = &embeddings[emb_idx];

                // Insert into HNSW index.
                let index_key = format!("{}.{}", job.type_name, job.vector_field);
                if let Some(index) = self.indexes.read().get(&index_key) {
                    index.insert(job.object_id, embedding);
                }

                // Persist the raw embedding to LSM so it survives restart.
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
                self.storage.delete(&mut txn, queue_key)?;
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

        // Over-retrieve: get more candidates than k for reranking.
        let retrieval_k = k * 10;
        let candidates = index.search(&query_vec[0], retrieval_k, ef.max(retrieval_k));

        // Find the source field for this vector field.
        let source_field = self
            .schema
            .get_type(type_name)
            .and_then(|td| td.get_field(vector_field))
            .and_then(|fd| fd.vectorize())
            .map(|v| v.source_field.clone());

        // Rerank if we have a reranker and can read the source text.
        if let Some(source_field) = source_field {
            let type_id = self.type_ids.get(type_name).copied();

            // Fetch original text for each candidate.
            let mut candidate_texts: Vec<(u64, String)> = Vec::new();
            if let Some(type_id) = type_id {
                for (obj_id, _dist) in &candidates {
                    let obj_key = KeyBuilder::object(type_id, *obj_id);
                    let txn = self.storage.begin_txn();
                    if let Ok(Some(data)) = self.storage.get(&txn, &obj_key) {
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

    /// Start the background worker thread.
    pub fn start_worker(self: &Arc<Self>) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // already running
        }

        let vectorizer = Arc::clone(self);
        let handle = std::thread::spawn(move || {
            while vectorizer.running.load(Ordering::SeqCst) {
                match vectorizer.process_pending() {
                    Ok(0) => {
                        // No work — sleep briefly before checking again.
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    Ok(_) => {
                        // Processed some jobs — immediately check for more.
                    }
                    Err(e) => {
                        eprintln!("vectorizer worker error: {e}");
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                }
            }
        });

        *self.worker_handle.lock() = Some(handle);
    }

    /// Stop the background worker thread.
    pub fn stop_worker(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.worker_handle.lock().take() {
            let _ = handle.join();
        }
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

    fn mark_jobs_failed(
        &self,
        jobs: &[(Bytes, VectorizeJob)],
        _error: &str,
    ) -> EngineResult<()> {
        for (queue_key, job) in jobs {
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
            self.storage.delete(&mut txn, queue_key)?;
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

        vectorizer.start_worker();

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
}
