use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

use parking_lot::RwLock;
use rand::Rng;

use crate::distance::{compute_distance, Metric};

/// Configuration for HNSW index construction.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,               // max neighbors per node per layer
    pub m_max0: usize,          // max neighbors at layer 0 (typically 2*m)
    pub ef_construction: usize, // search width during construction
    pub metric: Metric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            metric: Metric::Cosine,
        }
    }
}

/// A node in the HNSW graph.
struct Node {
    id: u64,
    vector: Vec<f32>,
    neighbors: Vec<Vec<u64>>, // neighbors[layer] = list of neighbor IDs
    deleted: bool,
}

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    config: HnswConfig,
    nodes: RwLock<Vec<Node>>,
    id_to_idx: RwLock<std::collections::HashMap<u64, usize>>,
    entry_point: RwLock<Option<usize>>,
    max_layer: RwLock<usize>,
    ml: f64, // normalization factor for level generation
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        let ml = 1.0 / (config.m as f64).ln();
        Self {
            config,
            nodes: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(std::collections::HashMap::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            ml,
        }
    }

    /// Insert a vector with the given ID.
    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let level = self.random_level();
        let num_layers = level + 1;

        // Create the node.
        let node = Node {
            id,
            vector,
            neighbors: vec![Vec::new(); num_layers],
            deleted: false,
        };

        let mut nodes = self.nodes.write();
        let node_idx = nodes.len();
        nodes.push(node);
        drop(nodes);

        self.id_to_idx.write().insert(id, node_idx);

        let entry_point = *self.entry_point.read();

        if entry_point.is_none() {
            *self.entry_point.write() = Some(node_idx);
            *self.max_layer.write() = level;
            return;
        }

        let ep = entry_point.unwrap();
        let current_max_layer = *self.max_layer.read();

        // Phase 1: Greedily traverse from top to the node's insertion level.
        let mut current_ep = ep;
        let nodes = self.nodes.read();
        for layer in (num_layers..=current_max_layer).rev() {
            current_ep = self.greedy_closest(&nodes, &nodes[node_idx].vector, current_ep, layer);
        }
        drop(nodes);

        // Phase 2: At each layer from insertion level down to 0, find neighbors and connect.
        for layer in (0..num_layers).rev() {
            let ef = self.config.ef_construction;
            let nodes = self.nodes.read();
            let candidates = self.search_layer(&nodes, &nodes[node_idx].vector, current_ep, ef, layer);
            drop(nodes);

            let max_neighbors = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Select the closest candidates as neighbors.
            let neighbors: Vec<u64> = candidates
                .iter()
                .take(max_neighbors)
                .map(|&(_, idx)| {
                    let nodes = self.nodes.read();
                    nodes[idx].id
                })
                .collect();

            // Set neighbors for this node at this layer.
            {
                let mut nodes = self.nodes.write();
                nodes[node_idx].neighbors[layer] = neighbors.clone();
            }

            // Add bidirectional connections.
            for &neighbor_id in &neighbors {
                let neighbor_idx = self.id_to_idx.read()[&neighbor_id];
                let mut nodes = self.nodes.write();

                if layer < nodes[neighbor_idx].neighbors.len() {
                    nodes[neighbor_idx].neighbors[layer].push(id);

                    // Prune if over capacity.
                    if nodes[neighbor_idx].neighbors[layer].len() > max_neighbors {
                        let node_vec = nodes[node_idx].vector.clone();
                        let _ = node_vec; // used below
                        let mut scored: Vec<(f32, u64)> = nodes[neighbor_idx].neighbors[layer]
                            .iter()
                            .map(|&nid| {
                                let nidx = self.id_to_idx.read()[&nid];
                                let dist = compute_distance(
                                    self.config.metric,
                                    &nodes[neighbor_idx].vector,
                                    &nodes[nidx].vector,
                                );
                                (dist, nid)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        scored.truncate(max_neighbors);
                        nodes[neighbor_idx].neighbors[layer] =
                            scored.into_iter().map(|(_, nid)| nid).collect();
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].1;
            }
        }

        // Update entry point if this node has a higher level.
        if level > current_max_layer {
            *self.entry_point.write() = Some(node_idx);
            *self.max_layer.write() = level;
        }
    }

    /// Search for the k nearest neighbors of the query vector.
    /// Returns (id, distance) pairs sorted by distance (ascending).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        let entry_point = *self.entry_point.read();
        let ep = match entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let max_layer = *self.max_layer.read();
        let nodes = self.nodes.read();

        // Greedy descent from top layer.
        let mut current_ep = ep;
        for layer in (1..=max_layer).rev() {
            current_ep = self.greedy_closest(&nodes, query, current_ep, layer);
        }

        // Search at layer 0 with ef candidates.
        let search_ef = ef.max(k);
        let candidates = self.search_layer(&nodes, query, current_ep, search_ef, 0);

        // Return top-k, filtering deleted nodes.
        candidates
            .into_iter()
            .filter(|&(_, idx)| !nodes[idx].deleted)
            .take(k)
            .map(|(dist, idx)| (nodes[idx].id, dist))
            .collect()
    }

    /// Mark a vector as deleted (tombstone). It will be skipped in search results
    /// but remains in the graph for connectivity until compaction.
    pub fn delete(&self, id: u64) -> bool {
        let idx = match self.id_to_idx.read().get(&id).copied() {
            Some(idx) => idx,
            None => return false,
        };
        self.nodes.write()[idx].deleted = true;
        true
    }

    /// Number of vectors in the index (including deleted).
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Number of active (non-deleted) vectors.
    pub fn active_count(&self) -> usize {
        self.nodes.read().iter().filter(|n| !n.deleted).count()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random();
        (-r.ln() * self.ml).floor() as usize
    }

    fn greedy_closest(&self, nodes: &[Node], query: &[f32], start: usize, layer: usize) -> usize {
        let mut current = start;
        let mut current_dist = compute_distance(self.config.metric, query, &nodes[current].vector);

        loop {
            let mut changed = false;

            if layer < nodes[current].neighbors.len() {
                for &neighbor_id in &nodes[current].neighbors[layer] {
                    if let Some(&neighbor_idx) = self.id_to_idx.read().get(&neighbor_id) {
                        let dist =
                            compute_distance(self.config.metric, query, &nodes[neighbor_idx].vector);
                        if dist < current_dist {
                            current = neighbor_idx;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search a single layer, returning up to `ef` candidates sorted by distance.
    fn search_layer(
        &self,
        nodes: &[Node],
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        let entry_dist = compute_distance(self.config.metric, query, &nodes[entry_point].vector);

        let mut visited = HashSet::new();
        visited.insert(entry_point);

        // Min-heap of candidates to explore.
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>> = BinaryHeap::new();
        candidates.push(Reverse((OrderedFloat(entry_dist), entry_point)));

        // Max-heap of results (worst first).
        let mut results: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry_point));

        while let Some(Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            let worst_dist = results.peek().map(|(OrderedFloat(d), _)| *d).unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            if layer < nodes[c_idx].neighbors.len() {
                for &neighbor_id in &nodes[c_idx].neighbors[layer] {
                    if let Some(&neighbor_idx) = self.id_to_idx.read().get(&neighbor_id)
                        && visited.insert(neighbor_idx) {
                            let dist = compute_distance(
                                self.config.metric,
                                query,
                                &nodes[neighbor_idx].vector,
                            );
                            let worst_dist =
                                results.peek().map(|(OrderedFloat(d), _)| *d).unwrap_or(f32::MAX);

                            if dist < worst_dist || results.len() < ef {
                                candidates.push(Reverse((OrderedFloat(dist), neighbor_idx)));
                                results.push((OrderedFloat(dist), neighbor_idx));

                                if results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                }
            }
        }

        // Convert to sorted vec.
        let mut result_vec: Vec<(f32, usize)> =
            results.into_iter().map(|(OrderedFloat(d), idx)| (d, idx)).collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result_vec
    }
}

/// Wrapper for f32 that implements Ord (for BinaryHeap).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn insert_and_search_basic() {
        let config = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(100, 32);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        assert_eq!(index.len(), 100);

        // Search should return results.
        let query = &vectors[0].1;
        let results = index.search(query, 5, 50);
        assert_eq!(results.len(), 5);

        // The query vector itself should be the closest match.
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn search_returns_correct_k() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config);

        let vectors = random_vectors(50, 16);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        let results = index.search(&vectors[0].1, 10, 50);
        assert_eq!(results.len(), 10);

        let results = index.search(&vectors[0].1, 3, 50);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_results_sorted_by_distance() {
        let config = HnswConfig {
            metric: Metric::L2,
            ..Default::default()
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(200, 32);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        let results = index.search(&vectors[0].1, 20, 100);
        for window in results.windows(2) {
            assert!(window[0].1 <= window[1].1, "results not sorted by distance");
        }
    }

    #[test]
    fn recall_is_reasonable() {
        let dims = 32;
        let n = 500;
        let k = 10;

        let config = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(n, dims);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        let query = &vectors[0].1;

        // Brute-force ground truth.
        let mut exact: Vec<(f32, u64)> = vectors
            .iter()
            .map(|(id, vec)| (crate::distance::l2_squared(query, vec), *id))
            .collect();
        exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: HashSet<u64> = exact.iter().take(k).map(|(_, id)| *id).collect();

        // HNSW search.
        let results = index.search(query, k, 100);
        let found: HashSet<u64> = results.iter().map(|(id, _)| *id).collect();

        let recall = truth.intersection(&found).count() as f32 / k as f32;
        assert!(
            recall >= 0.7,
            "recall {recall} too low (found {found:?} vs truth {truth:?})"
        );
    }

    #[test]
    fn delete_excludes_from_results() {
        let config = HnswConfig {
            metric: Metric::L2,
            ..Default::default()
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(50, 16);
        for (id, vec) in &vectors {
            index.insert(*id, vec.clone());
        }

        // Delete a specific vector.
        let deleted_id = 5u64;
        assert!(index.delete(deleted_id));
        assert_eq!(index.active_count(), 49);

        // Search should not return the deleted vector.
        let results = index.search(&vectors[0].1, 50, 100);
        assert!(!results.iter().any(|(id, _)| *id == deleted_id));
    }

    #[test]
    fn empty_index_search() {
        let index = HnswIndex::new(HnswConfig::default());
        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn single_vector() {
        let config = HnswConfig {
            metric: Metric::L2,
            ..Default::default()
        };
        let index = HnswIndex::new(config);
        index.insert(0, vec![1.0, 2.0, 3.0]);

        let results = index.search(&[1.0, 2.0, 3.0], 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn cosine_metric_works() {
        let config = HnswConfig {
            metric: Metric::Cosine,
            ..Default::default()
        };
        let index = HnswIndex::new(config);

        // Insert two vectors: one similar to query, one orthogonal.
        index.insert(0, vec![1.0, 0.0]);
        index.insert(1, vec![0.9, 0.1]);
        index.insert(2, vec![0.0, 1.0]);

        let results = index.search(&[1.0, 0.0], 3, 10);
        // [1.0, 0.0] should be closest, [0.9, 0.1] second, [0.0, 1.0] last.
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
    }
}
