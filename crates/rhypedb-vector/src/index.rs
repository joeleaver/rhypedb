use std::io;

use crate::distance::{self, Metric};
use crate::hnsw::{DistanceProvider, HnswConfig, HnswIndex, IndexMemory};
use crate::quantize::{CompressedVector, PreparedQuery, TurboQuantConfig, TurboQuantizer};
use crate::serial::{read_u8, write_u8};

const QUANTIZED_INDEX_MAGIC: &[u8; 4] = b"RQNT";
const QUANTIZED_INDEX_VERSION: u32 = 1;

/// Distance provider that uses TurboQuant's unbiased estimator.
/// All distance computations — during both graph construction and search —
/// go through TurboQuant compression.
pub struct TurboQuantDistance {
    quantizer: TurboQuantizer,
    metric: Metric,
}

impl DistanceProvider for TurboQuantDistance {
    type Stored = CompressedVector;
    type Query = PreparedQuery;

    fn prepare(&self, query: &[f32]) -> PreparedQuery {
        self.quantizer.prepare_query(query)
    }

    fn distance(&self, query: &PreparedQuery, stored: &CompressedVector) -> f32 {
        self.quantizer.distance_estimate_prepared(query, stored, self.metric)
    }

    fn distance_stored(&self, a: &CompressedVector, b: &CompressedVector) -> f32 {
        // For stored-vs-stored, project `a` (no full decompress — see
        // TurboQuantizer::prepare_stored) and use the asymmetric estimator.
        let prepared = self.quantizer.prepare_stored(a);
        self.quantizer
            .distance_estimate_prepared(&prepared, b, self.metric)
    }

    fn prepare_stored(&self, stored: &CompressedVector) -> PreparedQuery {
        self.quantizer.prepare_stored(stored)
    }

    fn prepare_stored_into(&self, stored: &CompressedVector, buf: &mut PreparedQuery) {
        self.quantizer.prepare_stored_into(stored, buf);
    }

    fn store(&self, vector: &[f32]) -> CompressedVector {
        self.quantizer.compress(vector)
    }

    fn write_stored(&self, stored: &CompressedVector, w: &mut dyn io::Write) -> io::Result<()> {
        stored.write_to(w)
    }

    fn read_stored(&self, r: &mut dyn io::Read) -> io::Result<CompressedVector> {
        CompressedVector::read_from(r)
    }

    fn stored_bytes(stored: &CompressedVector) -> usize {
        // Heap allocation only (codes + QJL signs share one buffer); the inline
        // struct (the boxed-slice header + the two f32 norms) is node_overhead.
        stored.heap_bytes()
    }
}

/// A vector index that uses TurboQuant compression for ALL operations.
///
/// Unlike the previous implementation which built the HNSW graph with
/// full-precision vectors and only used TurboQuant for reranking, this
/// index stores ONLY compressed vectors. Every distance computation
/// during graph construction and search uses TurboQuant's unbiased
/// estimator. This reflects production behavior where full-precision
/// vectors are not kept in memory.
pub struct QuantizedIndex {
    hnsw: HnswIndex<TurboQuantDistance>,
    metric: Metric,
}

impl QuantizedIndex {
    pub fn new(hnsw_config: HnswConfig, quant_config: TurboQuantConfig) -> Self {
        let metric = hnsw_config.metric;
        let distance = TurboQuantDistance {
            quantizer: TurboQuantizer::new(quant_config),
            metric,
        };
        Self {
            hnsw: HnswIndex::with_distance(hnsw_config, distance),
            metric,
        }
    }

    /// Insert a vector. It is immediately compressed — the full-precision
    /// vector is not retained.
    pub fn insert(&self, id: u64, vector: &[f32]) {
        self.hnsw.insert(id, vector);
    }

    /// Search for the k nearest neighbors using TurboQuant distance estimates.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.hnsw.search(query, k, ef)
    }

    pub fn delete(&self, id: u64) -> bool {
        self.hnsw.delete(id)
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn contains_id(&self, id: u64) -> bool {
        self.hnsw.contains_id(id)
    }

    /// Precise in-memory footprint of the index (codes + graph + overhead),
    /// excluding any LSM/durability layer. See [`HnswIndex::memory_bytes`].
    pub fn memory_bytes(&self) -> IndexMemory {
        self.hnsw.memory_bytes()
    }

    pub fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        w.write_all(QUANTIZED_INDEX_MAGIC)?;
        crate::serial::write_u32(w, QUANTIZED_INDEX_VERSION)?;
        write_u8(w, self.metric.to_u8())?;
        self.hnsw.distance().quantizer.write_to(w)?;
        self.hnsw.save(w)
    }

    pub fn load(r: &mut dyn io::Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != *QUANTIZED_INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid quantized index snapshot magic",
            ));
        }
        let version = crate::serial::read_u32(r)?;
        if version != QUANTIZED_INDEX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported quantized index snapshot version {version}"),
            ));
        }

        let metric = Metric::from_u8(read_u8(r)?).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown metric byte")
        })?;
        let quantizer = TurboQuantizer::read_from(r)?;
        let distance = TurboQuantDistance { quantizer, metric };
        let hnsw = HnswIndex::load(r, distance)?;

        Ok(Self { hnsw, metric })
    }
}

/// Compute recall@k: fraction of true k-nearest neighbors found.
pub fn recall_at_k(found: &[(u64, f32)], truth: &[(u64, f32)], k: usize) -> f32 {
    let truth_set: std::collections::HashSet<u64> =
        truth.iter().take(k).map(|(id, _)| *id).collect();
    let found_set: std::collections::HashSet<u64> =
        found.iter().take(k).map(|(id, _)| *id).collect();
    truth_set.intersection(&found_set).count() as f32 / k as f32
}

/// Brute-force exact k-nearest neighbors for ground truth.
pub fn brute_force_knn(
    query: &[f32],
    vectors: &[(u64, Vec<f32>)],
    k: usize,
    metric: Metric,
) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, vec)| (*id, distance::compute_distance(metric, query, vec)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

/// Compute Spearman rank correlation between two distance orderings.
pub fn rank_correlation(exact_order: &[u64], estimated_order: &[u64]) -> f32 {
    let n = exact_order.len().min(estimated_order.len());
    if n < 2 {
        return 1.0;
    }

    let exact_rank: std::collections::HashMap<u64, usize> = exact_order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut d_squared_sum = 0.0f64;
    let mut counted = 0usize;

    for (est_rank, &id) in estimated_order.iter().enumerate() {
        if let Some(&ex_rank) = exact_rank.get(&id) {
            let d = est_rank as f64 - ex_rank as f64;
            d_squared_sum += d * d;
            counted += 1;
        }
    }

    if counted < 2 {
        return 0.0;
    }

    let n = counted as f64;
    (1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1.0))) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::TurboQuantizer;
    use rand::Rng;

    fn random_vectors(count: usize, dims: usize) -> Vec<(u64, Vec<f32>)> {
        let mut rng = rand::rng();
        (0..count)
            .map(|i| {
                let vec: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
                (i as u64, vec)
            })
            .collect()
    }

    #[test]
    fn quantized_search_recall_4bit() {
        let dims = 64u32;
        let n = 500;
        let k = 10;

        let hnsw_config = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let quant_config = TurboQuantConfig::new(dims, 4);
        let index = QuantizedIndex::new(hnsw_config, quant_config);

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;
            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search(query, k, 100);
            total_recall += recall_at_k(&found, &truth, k);
        }

        let avg_recall = total_recall / n_queries as f32;
        eprintln!("4-bit NATIVE quantized recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
        assert!(
            avg_recall >= 0.4,
            "4-bit quantized recall@{k} = {avg_recall} (expected >= 0.4)"
        );
    }

    #[test]
    fn quantized_search_recall_3bit() {
        let dims = 64u32;
        let n = 500;
        let k = 10;

        let hnsw_config = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let quant_config = TurboQuantConfig::new(dims, 3);
        let index = QuantizedIndex::new(hnsw_config, quant_config);

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;
            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search(query, k, 100);
            total_recall += recall_at_k(&found, &truth, k);
        }

        let avg_recall = total_recall / n_queries as f32;
        eprintln!("3-bit NATIVE quantized recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
        assert!(
            avg_recall >= 0.3,
            "3-bit quantized recall@{k} = {avg_recall} (expected >= 0.3)"
        );
    }

    #[test]
    fn quantized_search_recall_2bit() {
        let dims = 64u32;
        let n = 500;
        let k = 10;

        let hnsw_config = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let quant_config = TurboQuantConfig::new(dims, 2);
        let index = QuantizedIndex::new(hnsw_config, quant_config);

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;
            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search(query, k, 100);
            total_recall += recall_at_k(&found, &truth, k);
        }

        let avg_recall = total_recall / n_queries as f32;
        eprintln!("2-bit NATIVE quantized recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
        // 2-bit with quantized graph construction will be lower.
        assert!(
            avg_recall >= 0.15,
            "2-bit quantized recall@{k} = {avg_recall} (expected >= 0.15)"
        );
    }

    #[test]
    fn exact_vs_quantized_recall_comparison() {
        let dims = 64u32;
        let n = 500;
        let k = 10;

        let vectors = random_vectors(n, dims as usize);
        let query = &vectors[0].1;
        let truth = brute_force_knn(query, &vectors, k, Metric::L2);

        // Exact HNSW.
        let exact_index = HnswIndex::new(HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        });
        for (id, vec) in &vectors {
            exact_index.insert(*id, vec);
        }
        let exact_recall = recall_at_k(&exact_index.search(query, k, 100), &truth, k);

        // Quantized HNSW (all operations through TurboQuant).
        let quant_index = QuantizedIndex::new(
            HnswConfig {
                m: 16,
                m_max0: 32,
                ef_construction: 100,
                metric: Metric::L2,
            },
            TurboQuantConfig::new(dims, 4),
        );
        for (id, vec) in &vectors {
            quant_index.insert(*id, vec);
        }
        let quant_recall = recall_at_k(&quant_index.search(query, k, 100), &truth, k);

        eprintln!(
            "exact HNSW recall: {exact_recall:.3}, NATIVE quantized recall: {quant_recall:.3}"
        );

        assert!(
            quant_recall >= exact_recall - 0.5,
            "quantized recall {quant_recall} too far below exact {exact_recall}"
        );
    }

    #[test]
    fn ranking_fidelity() {
        let dims = 64u32;
        let n = 200;

        let quant_config = TurboQuantConfig::new(dims, 4);
        let quantizer = TurboQuantizer::new(quant_config);

        let mut rng = rand::rng();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let vectors = random_vectors(n, dims as usize);

        let mut exact_ranking: Vec<(u64, f32)> = vectors
            .iter()
            .map(|(id, v)| (*id, distance::l2_squared(&query, v)))
            .collect();
        exact_ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let compressed: Vec<(u64, CompressedVector)> = vectors
            .iter()
            .map(|(id, v)| (*id, quantizer.compress(v)))
            .collect();
        let mut estimated_ranking: Vec<(u64, f32)> = compressed
            .iter()
            .map(|(id, cv)| (*id, quantizer.distance_estimate(&query, cv, Metric::L2)))
            .collect();
        estimated_ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let exact_order: Vec<u64> = exact_ranking.iter().map(|(id, _)| *id).collect();
        let estimated_order: Vec<u64> = estimated_ranking.iter().map(|(id, _)| *id).collect();

        let rho = rank_correlation(&exact_order, &estimated_order);
        eprintln!("Spearman rank correlation (4-bit, {dims}d): {rho:.3}");

        assert!(
            rho > 0.5,
            "rank correlation {rho} too low"
        );
    }

    #[test]
    fn cosine_metric_quantized_search() {
        let dims = 64u32;
        let n = 300;
        let k = 10;

        let index = QuantizedIndex::new(
            HnswConfig {
                m: 16,
                m_max0: 32,
                ef_construction: 100,
                metric: Metric::Cosine,
            },
            TurboQuantConfig::new(dims, 4),
        );

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let query = &vectors[0].1;
        let truth = brute_force_knn(query, &vectors, k, Metric::Cosine);
        let found = index.search(query, k, 100);
        let recall = recall_at_k(&found, &truth, k);

        eprintln!("cosine 4-bit NATIVE quantized recall@{k}: {recall:.3}");
        assert!(recall >= 0.3, "cosine recall {recall} too low");
    }

    #[test]
    fn quantized_index_save_load_roundtrip() {
        let dims = 32u32;
        let n = 100;

        let hnsw_config = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            metric: Metric::L2,
        };
        let quant_config = TurboQuantConfig::new(dims, 3);
        let index = QuantizedIndex::new(hnsw_config, quant_config);

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let query = &vectors[0].1;
        let original_results = index.search(query, 10, 50);

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();

        let loaded = QuantizedIndex::load(&mut &buf[..]).unwrap();

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.metric(), Metric::L2);

        let loaded_results = loaded.search(query, 10, 50);
        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, load) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, load.0);
            assert!((orig.1 - load.1).abs() < 1e-6);
        }
    }

    #[test]
    fn quantized_index_save_load_preserves_recall() {
        let dims = 64u32;
        let n = 300;
        let k = 10;

        let index = QuantizedIndex::new(
            HnswConfig {
                m: 16,
                m_max0: 32,
                ef_construction: 100,
                metric: Metric::L2,
            },
            TurboQuantConfig::new(dims, 4),
        );

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let query = &vectors[0].1;
        let truth = brute_force_knn(query, &vectors, k, Metric::L2);
        let original_recall = recall_at_k(&index.search(query, k, 100), &truth, k);

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();
        let loaded = QuantizedIndex::load(&mut &buf[..]).unwrap();

        let loaded_recall = recall_at_k(&loaded.search(query, k, 100), &truth, k);
        assert_eq!(
            original_recall, loaded_recall,
            "recall changed after save/load"
        );
    }

    #[test]
    fn quantized_index_insert_after_load() {
        let dims = 32u32;

        let index = QuantizedIndex::new(
            HnswConfig {
                m: 8,
                m_max0: 16,
                ef_construction: 50,
                metric: Metric::Cosine,
            },
            TurboQuantConfig::new(dims, 3),
        );

        let vectors = random_vectors(50, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();
        let loaded = QuantizedIndex::load(&mut &buf[..]).unwrap();

        // Insert new vectors into the loaded index.
        let extra = random_vectors(20, dims as usize);
        for (id, vec) in &extra {
            loaded.insert(50 + id, vec);
        }

        assert_eq!(loaded.len(), 70);
        let results = loaded.search(&vectors[0].1, 5, 50);
        assert_eq!(results.len(), 5);
    }
}
