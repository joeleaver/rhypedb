use crate::distance::{self, Metric};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::quantize::{CompressedVector, TurboQuantConfig, TurboQuantizer};

/// A vector index that combines HNSW for graph navigation with TurboQuant
/// for compressed storage and distance estimation.
///
/// During search, the HNSW graph structure guides traversal while
/// TurboQuant's unbiased estimator computes approximate distances
/// without decompressing full vectors.
pub struct QuantizedIndex {
    hnsw: HnswIndex,
    quantizer: TurboQuantizer,
    compressed: parking_lot::RwLock<std::collections::HashMap<u64, CompressedVector>>,
    metric: Metric,
}

impl QuantizedIndex {
    pub fn new(hnsw_config: HnswConfig, quant_config: TurboQuantConfig) -> Self {
        let metric = hnsw_config.metric;
        Self {
            hnsw: HnswIndex::new(hnsw_config),
            quantizer: TurboQuantizer::new(quant_config),
            compressed: parking_lot::RwLock::new(std::collections::HashMap::new()),
            metric,
        }
    }

    /// Insert a vector. The full-precision vector is used for HNSW graph
    /// construction (ensuring high-quality connections), while a compressed
    /// copy is stored for search-time distance estimation.
    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let compressed = self.quantizer.compress(&vector);
        self.compressed.write().insert(id, compressed);
        self.hnsw.insert(id, vector);
    }

    /// Search using TurboQuant's unbiased distance estimator.
    ///
    /// Phase 1: HNSW graph traversal finds candidate neighbors using
    /// full-precision distances (the graph was built with full vectors).
    ///
    /// Phase 2: Rerank candidates using TurboQuant's unbiased estimator
    /// to simulate what production search would look like with only
    /// compressed vectors stored.
    pub fn search_quantized(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        // Get more candidates than needed from HNSW (full-precision graph).
        let candidates = self.hnsw.search(query, ef, ef);

        // Rerank using TurboQuant distance estimates.
        let compressed = self.compressed.read();
        let mut reranked: Vec<(u64, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _exact_dist)| {
                let cv = compressed.get(&id)?;
                let estimated_dist = self.quantizer.distance_estimate(query, cv, self.metric);
                Some((id, estimated_dist))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked.truncate(k);
        reranked
    }

    /// Search using exact full-precision distances (for comparison).
    pub fn search_exact(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.hnsw.search(query, k, ef)
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
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
/// Returns a value in [-1, 1] where 1 = perfect agreement.
pub fn rank_correlation(exact_order: &[u64], estimated_order: &[u64]) -> f32 {
    let n = exact_order.len().min(estimated_order.len());
    if n < 2 {
        return 1.0;
    }

    // Build rank maps.
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
            index.insert(*id, vec.clone());
        }

        // Test multiple queries.
        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;

            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search_quantized(query, k, 100);
            let recall = recall_at_k(&found, &truth, k);
            total_recall += recall;
        }

        let avg_recall = total_recall / n_queries as f32;
        assert!(
            avg_recall >= 0.6,
            "4-bit quantized recall@{k} = {avg_recall} (expected >= 0.6)"
        );
        eprintln!("4-bit recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
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
            index.insert(*id, vec.clone());
        }

        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;
            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search_quantized(query, k, 100);
            total_recall += recall_at_k(&found, &truth, k);
        }

        let avg_recall = total_recall / n_queries as f32;
        assert!(
            avg_recall >= 0.4,
            "3-bit quantized recall@{k} = {avg_recall} (expected >= 0.4)"
        );
        eprintln!("3-bit recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
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
            index.insert(*id, vec.clone());
        }

        let mut total_recall = 0.0f32;
        let n_queries = 20;

        for q in 0..n_queries {
            let query = &vectors[q * 10].1;
            let truth = brute_force_knn(query, &vectors, k, Metric::L2);
            let found = index.search_quantized(query, k, 100);
            total_recall += recall_at_k(&found, &truth, k);
        }

        let avg_recall = total_recall / n_queries as f32;
        // 2-bit is aggressive — recall will be lower.
        eprintln!("2-bit recall@{k}: {avg_recall:.3} (n={n}, dims={dims})");
        assert!(
            avg_recall >= 0.2,
            "2-bit quantized recall@{k} = {avg_recall} (expected >= 0.2)"
        );
    }

    #[test]
    fn exact_vs_quantized_recall_comparison() {
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
            index.insert(*id, vec.clone());
        }

        let query = &vectors[0].1;
        let truth = brute_force_knn(query, &vectors, k, Metric::L2);

        let exact_recall = recall_at_k(&index.search_exact(query, k, 100), &truth, k);
        let quant_recall =
            recall_at_k(&index.search_quantized(query, k, 100), &truth, k);

        eprintln!("exact HNSW recall: {exact_recall:.3}, quantized recall: {quant_recall:.3}");

        // Quantized should not be drastically worse than exact HNSW.
        // (Both are approximate — the gap should be bounded.)
        assert!(
            quant_recall >= exact_recall - 0.4,
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
        let vectors: Vec<(u64, Vec<f32>)> = random_vectors(n, dims as usize);

        // Rank by exact distance.
        let mut exact_ranking: Vec<(u64, f32)> = vectors
            .iter()
            .map(|(id, v)| (*id, distance::l2_squared(&query, v)))
            .collect();
        exact_ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Rank by TurboQuant estimated distance.
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
            "rank correlation {rho} too low — distance ordering not preserved"
        );
    }

    #[test]
    fn cosine_metric_quantized_search() {
        let dims = 64u32;
        let n = 300;
        let k = 10;

        let hnsw_config = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::Cosine,
        };
        let quant_config = TurboQuantConfig::new(dims, 4);
        let index = QuantizedIndex::new(hnsw_config, quant_config);

        let vectors = random_vectors(n, dims as usize);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        let query = &vectors[0].1;
        let truth = brute_force_knn(query, &vectors, k, Metric::Cosine);
        let found = index.search_quantized(query, k, 100);
        let recall = recall_at_k(&found, &truth, k);

        eprintln!("cosine 4-bit recall@{k}: {recall:.3}");
        assert!(recall >= 0.4, "cosine recall {recall} too low");
    }
}
