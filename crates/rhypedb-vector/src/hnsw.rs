use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use std::io;

use parking_lot::RwLock;
use rand::Rng;

use crate::distance::{compute_distance, Metric};

const HNSW_MAGIC: &[u8; 4] = b"RHNS";
// v2: neighbor lists store internal node indices (u32) instead of external IDs
// (u64). v1 snapshots are rejected on load; the engine then rebuilds the index
// from the f32 vectors in the LSM `v:` keyspace (the source of truth), so the
// format change is safe across upgrades and rollbacks.
const HNSW_VERSION: u32 = 2;

use crate::serial::{
    read_f32_vec, read_i64, read_u32, read_u64, read_u8, write_f32_slice, write_i64, write_u32,
    write_u64, write_u8,
};

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

/// Trait for computing distances between a query and stored representations.
/// This is what makes HNSW work with both full-precision and compressed vectors.
pub trait DistanceProvider: Send + Sync {
    /// The stored representation of a vector (f32 slice, compressed, etc.)
    type Stored: Send + Sync;

    /// A query prepared once for repeated distance computations against many
    /// stored vectors (e.g. precomputed projections). Built via [`Self::prepare`].
    type Query: Send + Sync;

    /// Prepare a full-precision query for repeated distance computations.
    fn prepare(&self, query: &[f32]) -> Self::Query;

    /// Compute distance between a prepared query and a stored vector.
    fn distance(&self, query: &Self::Query, stored: &Self::Stored) -> f32;

    /// Compute distance between two stored vectors (for neighbor pruning).
    fn distance_stored(&self, a: &Self::Stored, b: &Self::Stored) -> f32;

    /// Prepare a *stored* vector as a query, so pruning can score many candidates
    /// against one fixed stored vector with the projection matvecs hoisted out of
    /// the loop. Equivalent to preparing its reconstruction.
    fn prepare_stored(&self, stored: &Self::Stored) -> Self::Query;

    /// Convert a full-precision vector into the stored representation.
    fn store(&self, vector: &[f32]) -> Self::Stored;

    /// Serialize a stored vector to a writer.
    fn write_stored(&self, stored: &Self::Stored, w: &mut dyn io::Write) -> io::Result<()>;

    /// Deserialize a stored vector from a reader.
    fn read_stored(&self, r: &mut dyn io::Read) -> io::Result<Self::Stored>;

    /// Resident byte size of one stored vector, including its heap allocations,
    /// for memory accounting. (Used by [`HnswIndex::memory_bytes`].)
    fn stored_bytes(stored: &Self::Stored) -> usize;
}

/// Breakdown of an HNSW index's in-memory footprint. `stored_bytes` is the
/// vector data (compressed codes or raw f32); `graph_bytes` the neighbor lists;
/// the rest is per-node structs and the id→index map. This is the RAM the index
/// needs to *serve* queries — it excludes any durability/LSM layer above it.
#[derive(Debug, Clone, Copy)]
pub struct IndexMemory {
    pub nodes: usize,
    pub stored_bytes: usize,
    pub graph_bytes: usize,
    pub node_overhead: usize,
    pub id_map_bytes: usize,
}

impl IndexMemory {
    pub fn total(&self) -> usize {
        self.stored_bytes + self.graph_bytes + self.node_overhead + self.id_map_bytes
    }
}

/// Default distance provider using full-precision f32 vectors.
pub struct ExactDistance {
    pub metric: Metric,
}

impl DistanceProvider for ExactDistance {
    type Stored = Vec<f32>;
    type Query = Vec<f32>;

    fn prepare(&self, query: &[f32]) -> Vec<f32> {
        query.to_vec()
    }

    fn distance(&self, query: &Vec<f32>, stored: &Vec<f32>) -> f32 {
        compute_distance(self.metric, query, stored)
    }

    fn distance_stored(&self, a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        compute_distance(self.metric, a, b)
    }

    fn prepare_stored(&self, stored: &Vec<f32>) -> Vec<f32> {
        stored.clone()
    }

    fn store(&self, vector: &[f32]) -> Vec<f32> {
        vector.to_vec()
    }

    fn write_stored(&self, stored: &Vec<f32>, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, stored.len() as u32)?;
        write_f32_slice(w, stored)
    }

    fn read_stored(&self, r: &mut dyn io::Read) -> io::Result<Vec<f32>> {
        let len = read_u32(r)? as usize;
        read_f32_vec(r, len)
    }

    fn stored_bytes(stored: &Vec<f32>) -> usize {
        // Heap only — the inline Vec header is counted in node_overhead.
        stored.capacity() * std::mem::size_of::<f32>()
    }
}

/// A node in the HNSW graph. Layer-0 neighbors (every node has them, and they are
/// the bulk of the graph) live in the parent [`Graph`]'s shared arena, not here —
/// only the count (`len0`) is stored inline. Higher layers (rare: ~6% of nodes at
/// m=16) keep their own small `Vec`s.
struct Node<S> {
    id: u64,
    stored: S,
    len0: u32,            // # of layer-0 neighbors in this node's arena slot
    upper: Vec<Vec<u32>>, // layers 1..; empty (no heap alloc) for single-layer nodes
    deleted: bool,
}

/// The HNSW graph: node metadata plus a single contiguous arena holding every
/// node's layer-0 neighbor list. Node `i`'s layer-0 neighbors occupy
/// `layer0[i*m_max0 .. i*m_max0 + nodes[i].len0]` (fixed stride `m_max0`). This
/// turns the dominant per-node `Vec<u32>` (one allocation each) into a single big
/// allocation — slashing allocator overhead (mimalloc per-allocation rounding +
/// metadata) at scale, where the in-memory index footprint is dominated by the
/// graph. Higher layers stay as per-node `Vec`s (rare). The on-disk snapshot
/// format is unchanged (still per-layer neighbor lists) — the arena is purely an
/// in-memory layout, so v2 snapshots load directly.
struct Graph<S> {
    nodes: Vec<Node<S>>,
    layer0: Vec<u32>,
    m_max0: usize,
}

impl<S> Graph<S> {
    fn new(m_max0: usize) -> Self {
        Self { nodes: Vec::new(), layer0: Vec::new(), m_max0 }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append a node, allocating its (empty) layer-0 arena slot. Returns its index.
    fn push_node(&mut self, node: Node<S>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.layer0.resize(self.layer0.len() + self.m_max0, 0);
        idx
    }

    /// Number of layers node `i` participates in (always >= 1).
    fn num_layers(&self, i: usize) -> usize {
        1 + self.nodes[i].upper.len()
    }

    /// Neighbor list of node `i` at `layer` (layer 0 from the arena slot).
    fn neighbors(&self, i: usize, layer: usize) -> &[u32] {
        if layer == 0 {
            let s = i * self.m_max0;
            &self.layer0[s..s + self.nodes[i].len0 as usize]
        } else {
            &self.nodes[i].upper[layer - 1]
        }
    }

    fn neighbor_len(&self, i: usize, layer: usize) -> usize {
        if layer == 0 {
            self.nodes[i].len0 as usize
        } else {
            self.nodes[i].upper[layer - 1].len()
        }
    }

    /// Replace node `i`'s neighbor list at `layer`. For layer 0, `list.len()` must
    /// be <= m_max0 (the slot stride).
    fn set_neighbors(&mut self, i: usize, layer: usize, list: &[u32]) {
        if layer == 0 {
            debug_assert!(list.len() <= self.m_max0);
            let s = i * self.m_max0;
            self.layer0[s..s + list.len()].copy_from_slice(list);
            self.nodes[i].len0 = list.len() as u32;
        } else {
            let v = &mut self.nodes[i].upper[layer - 1];
            v.clear();
            v.extend_from_slice(list);
        }
    }

    /// Append one neighbor to node `i`'s `layer`. Caller guarantees room (layer 0:
    /// `len0 < m_max0`).
    fn push_neighbor(&mut self, i: usize, layer: usize, n: u32) {
        if layer == 0 {
            let len = self.nodes[i].len0 as usize;
            debug_assert!(len < self.m_max0);
            self.layer0[i * self.m_max0 + len] = n;
            self.nodes[i].len0 += 1;
        } else {
            self.nodes[i].upper[layer - 1].push(n);
        }
    }
}

/// HNSW index for approximate nearest neighbor search, generic over
/// the distance computation and storage strategy.
pub struct HnswIndex<D: DistanceProvider = ExactDistance> {
    config: HnswConfig,
    distance: D,
    nodes: RwLock<Graph<D::Stored>>,
    id_to_idx: RwLock<std::collections::HashMap<u64, usize>>,
    entry_point: RwLock<Option<usize>>,
    max_layer: RwLock<usize>,
    ml: f64,
}

impl HnswIndex<ExactDistance> {
    /// Create an HNSW index with exact full-precision distance computation.
    pub fn new(config: HnswConfig) -> Self {
        let distance = ExactDistance {
            metric: config.metric,
        };
        Self::with_distance(config, distance)
    }
}

impl<D: DistanceProvider> HnswIndex<D> {
    /// Create an HNSW index with a custom distance provider.
    pub fn with_distance(config: HnswConfig, distance: D) -> Self {
        let ml = 1.0 / (config.m as f64).ln();
        let m_max0 = config.m_max0;
        Self {
            config,
            distance,
            nodes: RwLock::new(Graph::new(m_max0)),
            id_to_idx: RwLock::new(std::collections::HashMap::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            ml,
        }
    }

    /// Insert a vector with the given ID.
    pub fn insert(&self, id: u64, vector: &[f32]) {
        let level = self.random_level();
        let num_layers = level + 1;

        let stored = self.distance.store(vector);

        let node = Node {
            id,
            stored,
            len0: 0,
            upper: vec![Vec::new(); num_layers - 1],
            deleted: false,
        };

        let mut nodes = self.nodes.write();
        let node_idx = nodes.len();
        // Neighbor lists are stored as u32 internal indices; a single index must
        // fit in u32. 2^32 nodes would need ~860 GB just for codes, so this is a
        // structural invariant, never a real limit — but assert to rule out silent
        // graph corruption from a wrapped index.
        assert!(
            node_idx <= u32::MAX as usize,
            "HNSW index exceeds u32 node capacity ({node_idx} nodes)"
        );
        nodes.push_node(node);
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
        let prepared = self.distance.prepare(vector);

        // Phase 1: Greedily traverse from top to the node's insertion level.
        let mut current_ep = ep;
        let nodes = self.nodes.read();
        for layer in (num_layers..=current_max_layer).rev() {
            current_ep = self.greedy_closest(&nodes, &prepared, current_ep, layer);
        }
        drop(nodes);

        // Phase 2: At each layer from insertion level down to 0, find neighbors.
        for layer in (0..num_layers).rev() {
            let ef = self.config.ef_construction;
            let nodes = self.nodes.read();
            let candidates = self.search_layer(&nodes, &prepared, current_ep, ef, layer);
            drop(nodes);

            let max_neighbors = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // `candidates` already holds node *indices*; store them directly as
            // the neighbor list — no id resolution, no per-candidate lock.
            let neighbors: Vec<u32> = candidates
                .iter()
                .take(max_neighbors)
                .map(|&(_, idx)| idx as u32)
                .collect();

            {
                let mut g = self.nodes.write();
                g.set_neighbors(node_idx, layer, &neighbors);
            }

            // Add bidirectional connections with pruning.
            for &neighbor_idx in &neighbors {
                let neighbor_idx = neighbor_idx as usize;
                let mut g = self.nodes.write();

                if layer < g.num_layers(neighbor_idx) {
                    if g.neighbor_len(neighbor_idx, layer) < max_neighbors {
                        g.push_neighbor(neighbor_idx, layer, node_idx as u32);
                    } else {
                        // Slot full: score the existing edges plus the new candidate
                        // against the (fixed) neighbor vector and keep the closest
                        // max_neighbors. Equivalent to push-then-prune, but never
                        // exceeds the arena slot stride. prepare_stored hoists the
                        // projection matvecs out of the inner loop.
                        let prepared =
                            self.distance.prepare_stored(&g.nodes[neighbor_idx].stored);
                        let mut cand: Vec<u32> = g.neighbors(neighbor_idx, layer).to_vec();
                        cand.push(node_idx as u32);
                        let mut scored: Vec<(f32, u32)> = cand
                            .iter()
                            .map(|&nidx| {
                                let dist = self
                                    .distance
                                    .distance(&prepared, &g.nodes[nidx as usize].stored);
                                (dist, nidx)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        scored.truncate(max_neighbors);
                        let kept: Vec<u32> = scored.into_iter().map(|(_, nidx)| nidx).collect();
                        g.set_neighbors(neighbor_idx, layer, &kept);
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].1;
            }
        }

        if level > current_max_layer {
            *self.entry_point.write() = Some(node_idx);
            *self.max_layer.write() = level;
        }
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        let entry_point = *self.entry_point.read();
        let ep = match entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let prepared = self.distance.prepare(query);
        let max_layer = *self.max_layer.read();
        let nodes = self.nodes.read();

        let mut current_ep = ep;
        for layer in (1..=max_layer).rev() {
            current_ep = self.greedy_closest(&nodes, &prepared, current_ep, layer);
        }

        let search_ef = ef.max(k);
        let candidates = self.search_layer(&nodes, &prepared, current_ep, search_ef, 0);

        candidates
            .into_iter()
            .filter(|&(_, idx)| !nodes.nodes[idx].deleted)
            .take(k)
            .map(|(dist, idx)| (nodes.nodes[idx].id, dist))
            .collect()
    }

    pub fn delete(&self, id: u64) -> bool {
        let idx = match self.id_to_idx.read().get(&id).copied() {
            Some(idx) => idx,
            None => return false,
        };
        self.nodes.write().nodes[idx].deleted = true;
        true
    }

    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    pub fn active_count(&self) -> usize {
        self.nodes.read().nodes.iter().filter(|n| !n.deleted).count()
    }

    /// Precise in-memory footprint of the index (codes + graph + overhead),
    /// computed by walking the structure — not via process RSS. This is the RAM
    /// needed to serve k-NN; it excludes any durability layer (e.g. the LSM the
    /// server persists the vectors into for restart).
    pub fn memory_bytes(&self) -> IndexMemory {
        let g = self.nodes.read();
        let node_struct = std::mem::size_of::<Node<D::Stored>>();
        let vec_hdr = std::mem::size_of::<Vec<u32>>();
        let mut stored_bytes = 0usize;
        let mut graph_bytes = 0usize;
        for node in &g.nodes {
            stored_bytes += D::stored_bytes(&node.stored);
            // Higher layers are per-node Vecs (rare); layer 0 lives in the shared
            // arena, counted once below.
            graph_bytes += node.upper.capacity() * vec_hdr;
            for layer in &node.upper {
                graph_bytes += layer.capacity() * std::mem::size_of::<u32>();
            }
        }
        // Layer-0 arena: one allocation, fixed stride m_max0/node — count the full
        // capacity (including unused slot tails) as honest resident bytes.
        graph_bytes += g.layer0.capacity() * std::mem::size_of::<u32>();
        let n = g.nodes.len();
        // HashMap<u64, usize>: a bucket per slot at current capacity, ~1 control
        // byte + the (key, value) pair; a reasonable resident estimate.
        let entry = std::mem::size_of::<u64>() + std::mem::size_of::<usize>() + 1;
        let id_map_bytes = self.id_to_idx.read().capacity() * entry;
        IndexMemory {
            nodes: n,
            stored_bytes,
            graph_bytes,
            node_overhead: n * node_struct,
            id_map_bytes,
        }
    }

    pub fn contains_id(&self, id: u64) -> bool {
        self.id_to_idx.read().contains_key(&id)
    }

    pub fn distance(&self) -> &D {
        &self.distance
    }

    pub fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        w.write_all(HNSW_MAGIC)?;
        write_u32(w, HNSW_VERSION)?;

        write_u32(w, self.config.m as u32)?;
        write_u32(w, self.config.m_max0 as u32)?;
        write_u32(w, self.config.ef_construction as u32)?;
        write_u8(w, self.config.metric.to_u8())?;

        let g = self.nodes.read();
        let entry_point = *self.entry_point.read();
        let max_layer = *self.max_layer.read();

        write_u64(w, g.len() as u64)?;
        write_i64(w, entry_point.map_or(-1, |ep| ep as i64))?;
        write_u32(w, max_layer as u32)?;

        // Per-layer neighbor lists — the same on-disk layout as before the arena
        // (the arena is an in-memory detail), so the format/version is unchanged.
        for i in 0..g.nodes.len() {
            write_u64(w, g.nodes[i].id)?;
            write_u8(w, u8::from(g.nodes[i].deleted))?;
            let num_layers = g.num_layers(i);
            write_u32(w, num_layers as u32)?;
            for layer in 0..num_layers {
                let nbrs = g.neighbors(i, layer);
                write_u32(w, nbrs.len() as u32)?;
                for &nidx in nbrs {
                    write_u32(w, nidx)?;
                }
            }
            self.distance.write_stored(&g.nodes[i].stored, w)?;
        }

        Ok(())
    }

    pub fn load(r: &mut dyn io::Read, distance: D) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != *HNSW_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid HNSW snapshot magic",
            ));
        }
        let version = read_u32(r)?;
        if version != HNSW_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported HNSW snapshot version {version}"),
            ));
        }

        let m = read_u32(r)? as usize;
        let m_max0 = read_u32(r)? as usize;
        let ef_construction = read_u32(r)? as usize;
        let metric = Metric::from_u8(read_u8(r)?).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown metric byte")
        })?;
        let config = HnswConfig {
            m,
            m_max0,
            ef_construction,
            metric,
        };
        let ml = 1.0 / (m as f64).ln();

        let num_nodes = read_u64(r)? as usize;
        let ep_val = read_i64(r)?;
        let entry_point = if ep_val < 0 {
            None
        } else {
            Some(ep_val as usize)
        };
        // A corrupt/partially-written snapshot (no checksum) could carry an
        // entry point or neighbor index that does not address a real node.
        // Neighbor indices are used as unchecked slice indices in the hot loops,
        // so reject any out-of-range index here with InvalidData — that routes
        // the engine to its LSM rebuild fallback instead of panicking on the
        // first traversal.
        if let Some(ep) = entry_point
            && ep >= num_nodes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HNSW entry point out of range",
            ));
        }
        let max_layer = read_u32(r)? as usize;

        let mut nodes = Vec::with_capacity(num_nodes);
        // Layer-0 arena: one slot of m_max0 indices per node.
        let mut layer0 = vec![0u32; num_nodes * m_max0];
        let mut id_to_idx = std::collections::HashMap::with_capacity(num_nodes);

        for idx in 0..num_nodes {
            let id = read_u64(r)?;
            let deleted = read_u8(r)? != 0;
            let num_layers = read_u32(r)? as usize;
            if num_layers == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HNSW node has zero layers",
                ));
            }
            // Layer 0 -> this node's arena slot.
            let l0_count = read_u32(r)? as usize;
            if l0_count > m_max0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HNSW layer-0 neighbor count exceeds m_max0",
                ));
            }
            let slot = idx * m_max0;
            for j in 0..l0_count {
                let nidx = read_u32(r)?;
                if nidx as usize >= num_nodes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "HNSW neighbor index out of range",
                    ));
                }
                layer0[slot + j] = nidx;
            }
            let len0 = l0_count as u32;
            // Higher layers -> per-node Vecs.
            let mut upper = Vec::with_capacity(num_layers - 1);
            for _ in 1..num_layers {
                let num_neighbors = read_u32(r)? as usize;
                let mut layer_neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    let nidx = read_u32(r)?;
                    if nidx as usize >= num_nodes {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "HNSW neighbor index out of range",
                        ));
                    }
                    layer_neighbors.push(nidx);
                }
                upper.push(layer_neighbors);
            }
            let stored = distance.read_stored(r)?;

            id_to_idx.insert(id, idx);
            nodes.push(Node {
                id,
                stored,
                len0,
                upper,
                deleted,
            });
        }

        Ok(Self {
            config,
            distance,
            nodes: RwLock::new(Graph {
                nodes,
                layer0,
                m_max0,
            }),
            id_to_idx: RwLock::new(id_to_idx),
            entry_point: RwLock::new(entry_point),
            max_layer: RwLock::new(max_layer),
            ml,
        })
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random();
        (-r.ln() * self.ml).floor() as usize
    }

    fn greedy_closest(
        &self,
        g: &Graph<D::Stored>,
        query: &D::Query,
        start: usize,
        layer: usize,
    ) -> usize {
        let mut current = start;
        let mut current_dist = self.distance.distance(query, &g.nodes[current].stored);

        loop {
            let mut changed = false;

            if layer < g.num_layers(current) {
                for &neighbor_idx in g.neighbors(current, layer) {
                    let neighbor_idx = neighbor_idx as usize;
                    let dist = self.distance.distance(query, &g.nodes[neighbor_idx].stored);
                    if dist < current_dist {
                        current = neighbor_idx;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    fn search_layer(
        &self,
        g: &Graph<D::Stored>,
        query: &D::Query,
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        let entry_dist = self.distance.distance(query, &g.nodes[entry_point].stored);

        let mut visited = HashSet::new();
        visited.insert(entry_point);

        let mut candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>> = BinaryHeap::new();
        candidates.push(Reverse((OrderedFloat(entry_dist), entry_point)));

        let mut results: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry_point));

        while let Some(Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            let worst_dist = results.peek().map(|(OrderedFloat(d), _)| *d).unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            if layer < g.num_layers(c_idx) {
                for &neighbor_idx in g.neighbors(c_idx, layer) {
                    let neighbor_idx = neighbor_idx as usize;
                    if visited.insert(neighbor_idx) {
                        let dist = self
                            .distance
                            .distance(query, &g.nodes[neighbor_idx].stored);
                        let worst_dist = results
                            .peek()
                            .map(|(OrderedFloat(d), _)| *d)
                            .unwrap_or(f32::MAX);

                        if dist < worst_dist || results.len() < ef {
                            candidates
                                .push(Reverse((OrderedFloat(dist), neighbor_idx)));
                            results.push((OrderedFloat(dist), neighbor_idx));

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<(f32, usize)> = results
            .into_iter()
            .map(|(OrderedFloat(d), idx)| (d, idx))
            .collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result_vec
    }
}

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
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
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
            index.insert(*id, vec);
        }

        assert_eq!(index.len(), 100);

        let query = &vectors[0].1;
        let results = index.search(query, 5, 50);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn search_returns_correct_k() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config);

        let vectors = random_vectors(50, 16);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
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
            index.insert(*id, vec);
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
            index.insert(*id, vec);
        }

        let query = &vectors[0].1;

        let mut exact: Vec<(f32, u64)> = vectors
            .iter()
            .map(|(id, vec)| (crate::distance::l2_squared(query, vec), *id))
            .collect();
        exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: HashSet<u64> = exact.iter().take(k).map(|(_, id)| *id).collect();

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
            index.insert(*id, vec);
        }

        let deleted_id = 5u64;
        assert!(index.delete(deleted_id));
        assert_eq!(index.active_count(), 49);

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
        index.insert(0, &[1.0, 2.0, 3.0]);

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

        index.insert(0, &[1.0, 0.0]);
        index.insert(1, &[0.9, 0.1]);
        index.insert(2, &[0.0, 1.0]);

        let results = index.search(&[1.0, 0.0], 3, 10);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
    }

    #[test]
    fn save_load_roundtrip() {
        let config = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(200, 32);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let query = &vectors[0].1;
        let original_results = index.search(query, 10, 50);

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();

        let distance = ExactDistance { metric: Metric::L2 };
        let loaded = HnswIndex::load(&mut &buf[..], distance).unwrap();

        assert_eq!(loaded.len(), 200);
        let loaded_results = loaded.search(query, 10, 50);
        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, load) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, load.0);
            assert!((orig.1 - load.1).abs() < 1e-6);
        }
    }

    #[test]
    fn save_load_preserves_deletes() {
        let config = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(50, 16);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }
        index.delete(5);

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();

        let distance = ExactDistance { metric: Metric::L2 };
        let loaded = HnswIndex::load(&mut &buf[..], distance).unwrap();

        assert_eq!(loaded.len(), 50);
        assert_eq!(loaded.active_count(), 49);
        let results = loaded.search(&vectors[0].1, 50, 100);
        assert!(!results.iter().any(|(id, _)| *id == 5));
    }

    #[test]
    fn save_load_empty_index() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config);

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();

        let distance = ExactDistance {
            metric: Metric::Cosine,
        };
        let loaded = HnswIndex::load(&mut &buf[..], distance).unwrap();
        assert_eq!(loaded.len(), 0);
        assert!(loaded.search(&[1.0, 2.0, 3.0], 5, 50).is_empty());
    }

    #[test]
    fn load_rejects_bad_magic() {
        let buf = b"BAD_MAGIC_AND_SOME_PADDING";
        let distance = ExactDistance { metric: Metric::L2 };
        let result = HnswIndex::load(&mut &buf[..], distance);
        assert!(result.is_err());
    }

    #[test]
    fn load_rejects_v1_snapshot() {
        // v1 stored neighbor lists as u64 external IDs; v2 stores u32 internal
        // indices. A v1 snapshot must be rejected (the version check fires before
        // any node bytes are read) so the engine rebuilds the index from the LSM's
        // f32 vectors rather than misinterpreting the on-disk layout.
        let mut buf = Vec::new();
        buf.extend_from_slice(HNSW_MAGIC);
        write_u32(&mut buf, 1).unwrap();
        let distance = ExactDistance { metric: Metric::L2 };
        let result = HnswIndex::load(&mut &buf[..], distance);
        assert!(result.is_err(), "v1 HNSW snapshot must be rejected");
    }

    #[test]
    fn load_rejects_neighbor_index_out_of_range() {
        // A structurally well-formed v2 snapshot whose stored neighbor index is
        // >= num_nodes (a partial write / bit-flip — snapshots have no checksum)
        // must be rejected with an error, NOT loaded so it panics on the first
        // traversal. load() returning Err routes the engine to its LSM rebuild
        // fallback (vectorizer::rebuild_indexes).
        let mut buf = Vec::new();
        buf.extend_from_slice(HNSW_MAGIC);
        write_u32(&mut buf, 2).unwrap(); // version 2
        write_u32(&mut buf, 8).unwrap(); // m
        write_u32(&mut buf, 16).unwrap(); // m_max0
        write_u32(&mut buf, 50).unwrap(); // ef_construction
        write_u8(&mut buf, Metric::L2.to_u8()).unwrap();
        write_u64(&mut buf, 1).unwrap(); // num_nodes = 1
        write_i64(&mut buf, 0).unwrap(); // entry_point = 0 (in range)
        write_u32(&mut buf, 0).unwrap(); // max_layer
        // node 0
        write_u64(&mut buf, 42).unwrap(); // id
        write_u8(&mut buf, 0).unwrap(); // not deleted
        write_u32(&mut buf, 1).unwrap(); // 1 layer
        write_u32(&mut buf, 1).unwrap(); // layer 0: 1 neighbor
        write_u32(&mut buf, 5).unwrap(); // neighbor index 5 >= num_nodes(1): invalid
        write_u32(&mut buf, 2).unwrap(); // stored: len 2
        write_f32_slice(&mut buf, &[1.0, 2.0]).unwrap();

        let distance = ExactDistance { metric: Metric::L2 };
        let result = HnswIndex::load(&mut &buf[..], distance);
        assert!(
            result.is_err(),
            "v2 snapshot with an out-of-range neighbor index must be rejected"
        );
    }

    #[test]
    fn save_load_insert_after_load() {
        let config = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(config);

        let vectors = random_vectors(50, 16);
        for (id, vec) in &vectors {
            index.insert(*id, vec);
        }

        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();

        let distance = ExactDistance { metric: Metric::L2 };
        let loaded = HnswIndex::load(&mut &buf[..], distance).unwrap();

        // Insert more vectors after loading.
        let mut rng = rand::rng();
        for id in 50..60 {
            let vec: Vec<f32> = (0..16).map(|_| rng.random_range(-1.0..1.0)).collect();
            loaded.insert(id, &vec);
        }

        assert_eq!(loaded.len(), 60);
        let results = loaded.search(&vectors[0].1, 5, 50);
        assert_eq!(results.len(), 5);
    }
}
