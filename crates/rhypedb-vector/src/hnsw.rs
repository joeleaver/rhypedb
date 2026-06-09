use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

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

    /// Like [`Self::prepare_stored`] but reuses `buf` instead of allocating a
    /// fresh query. The build-time prune loop prepares up to ~2·m stored vectors
    /// per insert; reusing one scratch query removes those allocations. The
    /// default just overwrites via `prepare_stored`; providers with reusable
    /// internal buffers (e.g. projection vectors) should override it.
    fn prepare_stored_into(&self, stored: &Self::Stored, buf: &mut Self::Query) {
        *buf = self.prepare_stored(stored);
    }

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

/// Debug-only enforcement of the lock discipline that keeps the per-node-locked
/// graph deadlock-free: at most ONE node `links` lock is held at a time, and the
/// leaf locks (`entry`, `id_to_idx`) are never taken while a node lock is held.
/// Compiled to nothing in release builds.
#[cfg(debug_assertions)]
mod lockcheck {
    use std::cell::Cell;
    thread_local! {
        static NODE_LOCKS: Cell<u32> = const { Cell::new(0) };
    }
    /// RAII scope marking one node `links` lock held; asserts none was already held
    /// (taking a second would risk an A-locks-B / B-locks-A deadlock).
    pub(super) struct NodeLockScope;
    impl NodeLockScope {
        pub(super) fn enter() -> Self {
            NODE_LOCKS.with(|c| {
                assert_eq!(c.get(), 0, "two node `links` locks held at once — deadlock risk");
                c.set(1);
            });
            NodeLockScope
        }
    }
    impl Drop for NodeLockScope {
        fn drop(&mut self) {
            NODE_LOCKS.with(|c| c.set(c.get() - 1));
        }
    }
    /// Assert no node lock is held — call before taking a leaf lock (`entry`,
    /// `id_to_idx`) so the leaf/node lock order can never form a cycle.
    pub(super) fn assert_no_node_lock() {
        NODE_LOCKS.with(|c| {
            assert_eq!(c.get(), 0, "leaf lock taken while holding a node `links` lock — lock-order violation");
        });
    }
}
#[cfg(not(debug_assertions))]
mod lockcheck {
    pub(super) struct NodeLockScope;
    impl NodeLockScope {
        #[inline(always)]
        pub(super) fn enter() -> Self {
            NodeLockScope
        }
    }
    #[inline(always)]
    pub(super) fn assert_no_node_lock() {}
}

/// The graph's entry point and its layer, kept as ONE value behind a single lock
/// so concurrent inserts always observe a consistent `(node, max_layer)` pair and
/// update it atomically. `None` = empty index.
#[derive(Debug, Clone, Copy)]
struct EntryState {
    node: usize,
    max_layer: usize,
}

/// A node in the HNSW graph. `stored` (the compressed/raw vector) is IMMUTABLE
/// after construction, so distance computations read it LOCK-FREE via the stable
/// address `boxcar::Vec` guarantees. Only the neighbor lists are mutable, behind
/// the per-node `links` lock. `ready` is set (Release) once the node is fully
/// linked at all layers; concurrent traversals skip a node until then, so they
/// never route through a half-built node.
struct Node<S> {
    id: u64,
    stored: S,
    links: RwLock<NodeLinks>,
    deleted: std::sync::atomic::AtomicBool,
    ready: std::sync::atomic::AtomicBool,
}

impl<S> Node<S> {
    /// Run `f` with a brief read lock on the neighbor lists. The closure must NOT
    /// lock another node's `links` (the scope's debug assert enforces it); copy the
    /// indices out and release before any further graph traversal.
    #[inline]
    fn with_links_read<R>(&self, f: impl FnOnce(&NodeLinks) -> R) -> R {
        let _scope = lockcheck::NodeLockScope::enter();
        let g = self.links.read();
        f(&g)
    }

    /// Run `f` with a write lock on the neighbor lists — used for the atomic
    /// read-modify-write of a back-edge prune. Same one-lock-at-a-time rule.
    #[inline]
    fn with_links_write<R>(&self, f: impl FnOnce(&mut NodeLinks) -> R) -> R {
        let _scope = lockcheck::NodeLockScope::enter();
        let mut g = self.links.write();
        f(&mut g)
    }
}

/// A node's mutable neighbor lists (guarded by [`Node::links`]). Layer 0 is a
/// fixed-capacity `Box<[u32]>` of length `m_max0` with an inline count — one small
/// allocation per node, no `Vec` header/growth slack, recovering most of the old
/// shared arena's compactness while being trivially safe under per-node locking.
/// Higher layers (rare: ~6% of nodes at m=16) are small `Vec`s.
struct NodeLinks {
    layer0: Box<[u32]>, // length == m_max0; first `len0` entries are the neighbors
    len0: u32,
    upper: Vec<Vec<u32>>, // upper[l-1] = layer l neighbors; empty for single-layer nodes
}

impl NodeLinks {
    fn new(num_layers: usize, m_max0: usize) -> Self {
        Self {
            layer0: vec![0u32; m_max0].into_boxed_slice(),
            len0: 0,
            upper: vec![Vec::new(); num_layers - 1],
        }
    }

    fn num_layers(&self) -> usize {
        1 + self.upper.len()
    }

    fn neighbors(&self, layer: usize) -> &[u32] {
        if layer == 0 {
            &self.layer0[..self.len0 as usize]
        } else {
            &self.upper[layer - 1]
        }
    }

    fn neighbor_len(&self, layer: usize) -> usize {
        if layer == 0 {
            self.len0 as usize
        } else {
            self.upper[layer - 1].len()
        }
    }

    fn set(&mut self, layer: usize, list: &[u32]) {
        if layer == 0 {
            debug_assert!(list.len() <= self.layer0.len());
            self.layer0[..list.len()].copy_from_slice(list);
            self.len0 = list.len() as u32;
        } else {
            let v = &mut self.upper[layer - 1];
            v.clear();
            v.extend_from_slice(list);
        }
    }

    fn push(&mut self, layer: usize, n: u32) {
        if layer == 0 {
            let len = self.len0 as usize;
            debug_assert!(len < self.layer0.len());
            self.layer0[len] = n;
            self.len0 += 1;
        } else {
            self.upper[layer - 1].push(n);
        }
    }
}

/// HNSW index for approximate nearest neighbor search, generic over
/// the distance computation and storage strategy.
pub struct HnswIndex<D: DistanceProvider = ExactDistance> {
    config: HnswConfig,
    distance: D,
    /// Append-only, never-reallocating node store. Gives STABLE element addresses
    /// (so `&node.stored` is read lock-free) and lock-free concurrent `push`/`get`.
    /// Each node carries its own `links` lock; there is no global graph lock.
    nodes: boxcar::Vec<Node<D::Stored>>,
    id_to_idx: RwLock<std::collections::HashMap<u64, usize>>,
    entry: RwLock<Option<EntryState>>,
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
        // m_max0 is the fixed capacity of each node's layer-0 list; must be >= 1.
        assert!(config.m_max0 >= 1, "HnswConfig.m_max0 must be >= 1");
        let ml = 1.0 / (config.m as f64).ln();
        Self {
            config,
            distance,
            nodes: boxcar::Vec::new(),
            id_to_idx: RwLock::new(std::collections::HashMap::new()),
            entry: RwLock::new(None),
            ml,
        }
    }

    /// Look up node `idx` that we KNOW is published (our own freshly-pushed node,
    /// or an index from a synchronized edge). Panics on a fabricated/in-flight
    /// index — callers traversing neighbor lists must use `self.nodes.get` + skip
    /// `None` instead (the boxcar in-flight window).
    #[inline]
    fn own_node(&self, idx: usize) -> &Node<D::Stored> {
        self.nodes
            .get(idx)
            .expect("node index must be published before use")
    }

    /// Insert a vector with the given ID. Safe to call concurrently from many
    /// threads (see [`Self::insert_parallel`]): the graph has no global lock — each
    /// node carries its own `links` lock, taken ONE AT A TIME, and the entry point
    /// is updated atomically under a single leaf lock. A node is only published as
    /// reachable (`ready`) after it is fully linked at all its layers.
    ///
    /// Re-inserting an existing ID (an "update", whether serial or a concurrent
    /// same-ID race) creates a new node and tombstones the prior one: the new
    /// vector wins (matching the LSM `v:` last-writer semantics) and search never
    /// returns a duplicate ID. The tombstoned node lingers as a routing-only graph
    /// member until the next rebuild reclaims it. Distinct IDs (the bulk-build case)
    /// never hit this path.
    pub fn insert(&self, id: u64, vector: &[f32]) {
        let level = self.random_level();
        let num_layers = level + 1;
        let m_max0 = self.config.m_max0;

        let stored = self.distance.store(vector);
        let node = Node {
            id,
            stored,
            links: RwLock::new(NodeLinks::new(num_layers, m_max0)),
            deleted: AtomicBool::new(false),
            ready: AtomicBool::new(false),
        };
        // Lock-free append; returns a stable, never-reused index.
        let node_idx = self.nodes.push(node);
        // u32 internal-index invariant (2^32 nodes ≈ 860 GB of codes — structural,
        // never a real limit, but assert to rule out a wrapped index).
        assert!(
            node_idx <= u32::MAX as usize,
            "HNSW index exceeds u32 node capacity ({node_idx} nodes)"
        );

        lockcheck::assert_no_node_lock();
        // Claim the id atomically (the RwLock serializes concurrent same-id inserts).
        // If it was already indexed (a re-insert/update, or a same-id race), tombstone
        // the prior node so search won't return a duplicate id — the new node wins.
        if let Some(prev) = self.id_to_idx.write().insert(id, node_idx)
            && let Some(old) = self.nodes.get(prev)
        {
            old.deleted.store(true, Ordering::Release);
        }

        // Atomic empty-init: if the index is still empty, become the entry point and
        // finish (the first node has no edges but is where every search starts). Two
        // concurrent inserts into an empty index can't both early-return: one wins
        // the `entry` write lock, the other observes it and links normally.
        lockcheck::assert_no_node_lock();
        let (ep, current_max_layer) = {
            let mut entry = self.entry.write();
            match *entry {
                None => {
                    self.own_node(node_idx).ready.store(true, Ordering::Release);
                    *entry = Some(EntryState {
                        node: node_idx,
                        max_layer: level,
                    });
                    return;
                }
                Some(st) => (st.node, st.max_layer),
            }
        };

        let prepared = self.distance.prepare(vector);
        let mut scratch: Vec<u32> = Vec::with_capacity(m_max0);

        // Phase 1: greedy descent from the top down to the node's insertion level.
        let mut current_ep = ep;
        for layer in (num_layers..=current_max_layer).rev() {
            current_ep = self.greedy_closest(&prepared, current_ep, layer, &mut scratch);
        }

        // Reusable projected-query buffer for the prune step (one scratch query
        // refilled per full-slot neighbor instead of allocating each time).
        let mut prune_q: Option<D::Query> = None;

        // Phase 2: at each layer from insertion level to 0, find + link neighbors.
        for layer in (0..num_layers).rev() {
            let candidates =
                self.search_layer(&prepared, current_ep, self.config.ef_construction, layer, &mut scratch);
            let max_neighbors = if layer == 0 { m_max0 } else { self.config.m };

            // Our forward edges = closest candidates (skip self + dedup defensively).
            let mut neighbors: Vec<u32> = Vec::with_capacity(max_neighbors);
            for &(_, idx) in &candidates {
                if idx == node_idx {
                    continue;
                }
                let u = idx as u32;
                if !neighbors.contains(&u) {
                    neighbors.push(u);
                    if neighbors.len() >= max_neighbors {
                        break;
                    }
                }
            }

            // Write our OWN edges (lock self only, then release).
            self.own_node(node_idx)
                .with_links_write(|l| l.set(layer, &neighbors));

            // Back-edges: lock ONE neighbor at a time across its FULL read-modify-
            // write (re-read its list, prune, write back) — releasing between the
            // read and write-back would let a concurrent insert clobber our edge and
            // progressively disconnect the graph. Hoist only the neighbor's own
            // (immutable) projection out of the lock.
            for &nbr_u in &neighbors {
                let nbr = nbr_u as usize;
                let nbr_node = match self.nodes.get(nbr) {
                    Some(n) => n,
                    None => continue, // in-flight; can't yet be a useful graph member
                };
                match prune_q.as_mut() {
                    Some(buf) => self.distance.prepare_stored_into(&nbr_node.stored, buf),
                    None => prune_q = Some(self.distance.prepare_stored(&nbr_node.stored)),
                }
                let prepared_nbr = prune_q.as_ref().unwrap();
                nbr_node.with_links_write(|l| {
                    if layer >= l.num_layers() {
                        return; // neighbor doesn't participate at this layer
                    }
                    let already = l.neighbors(layer).contains(&(node_idx as u32));
                    if already {
                        return;
                    }
                    if l.neighbor_len(layer) < max_neighbors {
                        l.push(layer, node_idx as u32);
                        return;
                    }
                    // Slot full: score current members + the new node against the
                    // neighbor (distances read immutable `stored` lock-free — no
                    // second links lock), keep the closest `max_neighbors`.
                    let mut scored: Vec<(f32, u32)> = l
                        .neighbors(layer)
                        .iter()
                        .map(|&c| {
                            let d = self
                                .nodes
                                .get(c as usize)
                                .map(|n| self.distance.distance(prepared_nbr, &n.stored))
                                .unwrap_or(f32::MAX);
                            (d, c)
                        })
                        .collect();
                    let d_new = self
                        .distance
                        .distance(prepared_nbr, &self.own_node(node_idx).stored);
                    scored.push((d_new, node_idx as u32));
                    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    scored.truncate(max_neighbors);
                    let kept: Vec<u32> = scored.into_iter().map(|(_, c)| c).collect();
                    l.set(layer, &kept);
                });
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].1;
            }
        }

        // Publish: fully linked at all layers, now safe for others to traverse to.
        self.own_node(node_idx).ready.store(true, Ordering::Release);

        // Raise the entry point if this node is taller than the current max — under
        // the single `entry` lock, re-checking in case a concurrent insert already
        // raised it past our level.
        if level > current_max_layer {
            lockcheck::assert_no_node_lock();
            let mut entry = self.entry.write();
            match *entry {
                Some(st) if level <= st.max_layer => {}
                _ => {
                    *entry = Some(EntryState {
                        node: node_idx,
                        max_layer: level,
                    })
                }
            }
        }
    }

    /// Insert a batch of vectors, fanning the work across a scoped thread pool.
    /// `insert` is concurrency-safe (per-node locks, no global graph lock), so this
    /// runs chunks on separate threads. The resulting graph is topologically
    /// equivalent to a serial build (insertion order is nondeterministic either way,
    /// like HNSW's random levels), so recall is preserved within run-to-run variance.
    pub fn insert_parallel(&self, items: &[(u64, Vec<f32>)]) {
        if items.is_empty() {
            return;
        }
        let cores = std::env::var("RHYPE_BUILD_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        const MIN_PARALLEL: usize = 256;
        if cores <= 1 || items.len() < MIN_PARALLEL {
            for (id, v) in items {
                self.insert(*id, v);
            }
            return;
        }
        // Seed the entry point with one insert before fanning out, so workers build
        // against a non-empty graph (not required for correctness — empty-init is
        // atomic — just avoids every thread racing to initialize at once).
        let start = if self.entry.read().is_none() {
            self.insert(items[0].0, &items[0].1);
            1
        } else {
            0
        };
        let rest = &items[start..];
        if rest.is_empty() {
            return;
        }
        let n_threads = cores.min(rest.len());
        let chunk = rest.len().div_ceil(n_threads);
        std::thread::scope(|s| {
            for c in rest.chunks(chunk) {
                s.spawn(move || {
                    for (id, v) in c {
                        self.insert(*id, v);
                    }
                });
            }
        });
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        lockcheck::assert_no_node_lock();
        let (ep, max_layer) = match *self.entry.read() {
            Some(st) => (st.node, st.max_layer),
            None => return Vec::new(),
        };

        let prepared = self.distance.prepare(query);
        let mut scratch: Vec<u32> = Vec::with_capacity(self.config.m_max0);

        let mut current_ep = ep;
        for layer in (1..=max_layer).rev() {
            current_ep = self.greedy_closest(&prepared, current_ep, layer, &mut scratch);
        }

        let search_ef = ef.max(k);
        let candidates = self.search_layer(&prepared, current_ep, search_ef, 0, &mut scratch);

        // Filter deletes (routing-only nodes) at the result stage, then take k.
        candidates
            .into_iter()
            .filter_map(|(dist, idx)| {
                let n = self.nodes.get(idx)?;
                if n.deleted.load(Ordering::Acquire) {
                    None
                } else {
                    Some((n.id, dist))
                }
            })
            .take(k)
            .collect()
    }

    pub fn delete(&self, id: u64) -> bool {
        lockcheck::assert_no_node_lock();
        let idx = match self.id_to_idx.read().get(&id).copied() {
            Some(idx) => idx,
            None => return false,
        };
        match self.nodes.get(idx) {
            Some(n) => {
                n.deleted.store(true, Ordering::Release);
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.count()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.count() == 0
    }

    pub fn active_count(&self) -> usize {
        // boxcar's iterator only yields published elements (never the in-flight
        // window), so this is safe to call concurrently with insert.
        self.nodes
            .iter()
            .filter(|(_, n)| !n.deleted.load(Ordering::Acquire))
            .count()
    }

    /// Precise in-memory footprint of the index (codes + graph + overhead),
    /// computed by walking the structure — not via process RSS. This is the RAM
    /// needed to serve k-NN; it excludes any durability layer (e.g. the LSM the
    /// server persists the vectors into for restart). Intended for quiescent use.
    pub fn memory_bytes(&self) -> IndexMemory {
        let node_struct = std::mem::size_of::<Node<D::Stored>>();
        let vec_hdr = std::mem::size_of::<Vec<u32>>();
        let u32_sz = std::mem::size_of::<u32>();
        let mut stored_bytes = 0usize;
        let mut graph_bytes = 0usize;
        let mut n = 0usize;
        for (_, node) in self.nodes.iter() {
            n += 1;
            stored_bytes += D::stored_bytes(&node.stored);
            node.with_links_read(|l| {
                // Layer-0 fixed Box<[u32]> (m_max0 entries) + higher-layer Vecs.
                graph_bytes += l.layer0.len() * u32_sz;
                graph_bytes += l.upper.capacity() * vec_hdr;
                for layer in &l.upper {
                    graph_bytes += layer.capacity() * u32_sz;
                }
            });
        }
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
        lockcheck::assert_no_node_lock();
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

        let entry = *self.entry.read();

        // On-disk layout unchanged: node count, entry-point index (-1 if empty),
        // max_layer, then per-node per-layer neighbor lists. boxcar's iterator
        // yields nodes in index order (0..n), which is what neighbor indices
        // reference. Intended for quiescent (non-concurrent-insert) save.
        write_u64(w, self.nodes.count() as u64)?;
        write_i64(w, entry.map_or(-1, |st| st.node as i64))?;
        write_u32(w, entry.map_or(0, |st| st.max_layer) as u32)?;

        for (_, node) in self.nodes.iter() {
            write_u64(w, node.id)?;
            write_u8(w, u8::from(node.deleted.load(Ordering::Acquire)))?;
            // Snapshot the neighbor lists under the per-node lock, then write outside.
            let layers: Vec<Vec<u32>> =
                node.with_links_read(|l| (0..l.num_layers()).map(|ly| l.neighbors(ly).to_vec()).collect());
            write_u32(w, layers.len() as u32)?;
            for nbrs in &layers {
                write_u32(w, nbrs.len() as u32)?;
                for &nidx in nbrs {
                    write_u32(w, nidx)?;
                }
            }
            self.distance.write_stored(&node.stored, w)?;
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

        // Bound m_max0: it sizes a per-node `Box<[u32]>`, so a corrupt huge value
        // would OOM before the body could fail. Real configs are m_max0 = 2*m,
        // well under 2^16. 0 is invalid (every node needs a layer-0 slot).
        if m_max0 == 0 || m_max0 > (1 << 16) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HNSW m_max0 out of range",
            ));
        }

        let num_nodes = read_u64(r)? as usize;
        let ep_val = read_i64(r)?;
        let entry_node = if ep_val < 0 { None } else { Some(ep_val as usize) };
        // A corrupt/partially-written snapshot (no checksum) could carry an entry
        // point or neighbor index that doesn't address a real node. Reject any
        // out-of-range index with InvalidData so the engine routes to its LSM
        // rebuild fallback instead of panicking on the first traversal.
        if let Some(ep) = entry_node
            && ep >= num_nodes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HNSW entry point out of range",
            ));
        }
        let max_layer = read_u32(r)? as usize;
        // Bound max_layer: it drives the top-down descent loops in search()/insert()
        // (`for layer in (1..=max_layer).rev()`), so a corrupt huge value would hang
        // the process rather than fail fast. Real max_layer is ~log_m(N), well under
        // 64 (same bound as per-node num_layers).
        if max_layer > 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HNSW max_layer out of range",
            ));
        }

        let nodes: boxcar::Vec<Node<D::Stored>> = boxcar::Vec::new();
        // Cap the id-map reserve: `num_nodes` is unchecked, so a corrupt count must
        // not pre-allocate a giant map. A valid larger snapshot just grows it.
        let mut id_to_idx = std::collections::HashMap::with_capacity(num_nodes.min(4096));

        for idx in 0..num_nodes {
            let id = read_u64(r)?;
            let deleted = read_u8(r)? != 0;
            let num_layers = read_u32(r)? as usize;
            // Real max_layer is ~log_m(N) (well under 64); bound it so a corrupt
            // count can't allocate a huge `upper` vec.
            if num_layers == 0 || num_layers > 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HNSW node layer count out of range",
                ));
            }
            // Layer 0 -> this node's fixed Box<[u32]> slot.
            let l0_count = read_u32(r)? as usize;
            if l0_count > m_max0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HNSW layer-0 neighbor count exceeds m_max0",
                ));
            }
            let mut layer0 = vec![0u32; m_max0].into_boxed_slice();
            for slot in layer0.iter_mut().take(l0_count) {
                let nidx = read_u32(r)?;
                if nidx as usize >= num_nodes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "HNSW neighbor index out of range",
                    ));
                }
                *slot = nidx;
            }
            let len0 = l0_count as u32;
            // Higher layers -> per-node Vecs. Do NOT pre-reserve from the unchecked
            // count; push and let a truncated stream fail fast on read_u32.
            let mut upper = Vec::with_capacity(num_layers - 1);
            for _ in 1..num_layers {
                let count = read_u32(r)? as usize;
                let mut layer_neighbors = Vec::new();
                for _ in 0..count {
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
            let pushed = nodes.push(Node {
                id,
                stored,
                links: RwLock::new(NodeLinks { layer0, len0, upper }),
                deleted: AtomicBool::new(deleted),
                ready: AtomicBool::new(true), // loaded nodes are fully linked
            });
            debug_assert_eq!(pushed, idx);
        }

        Ok(Self {
            config,
            distance,
            nodes,
            id_to_idx: RwLock::new(id_to_idx),
            entry: RwLock::new(entry_node.map(|node| EntryState { node, max_layer })),
            ml,
        })
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Copy node `idx`'s layer-`layer` neighbor indices into `out` under a brief
    /// per-node read lock, then RELEASE — so distance computation against those
    /// neighbors happens lock-free (immutable `stored`) and never holds two node
    /// locks. `out` is cleared first. No-op if the node is absent (in-flight) or
    /// doesn't reach `layer`.
    #[inline]
    fn copy_neighbors(&self, idx: usize, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        if let Some(node) = self.nodes.get(idx) {
            node.with_links_read(|l| {
                if layer < l.num_layers() {
                    out.extend_from_slice(l.neighbors(layer));
                }
            });
        }
    }

    fn greedy_closest(
        &self,
        query: &D::Query,
        start: usize,
        layer: usize,
        scratch: &mut Vec<u32>,
    ) -> usize {
        let mut current = start;
        let mut current_dist = match self.nodes.get(current) {
            Some(n) => self.distance.distance(query, &n.stored),
            None => return current,
        };

        loop {
            self.copy_neighbors(current, layer, scratch);
            let mut changed = false;
            // Distances are computed AFTER the lock is released (immutable stored).
            for &nbr_u in scratch.iter() {
                let nbr = nbr_u as usize;
                let n = match self.nodes.get(nbr) {
                    Some(n) => n,
                    None => continue, // in-flight; skip
                };
                if !n.ready.load(Ordering::Acquire) {
                    continue; // mid-construction; not a safe routing target yet
                }
                let dist = self.distance.distance(query, &n.stored);
                if dist < current_dist {
                    current = nbr;
                    current_dist = dist;
                    changed = true;
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
        query: &D::Query,
        entry_point: usize,
        ef: usize,
        layer: usize,
        scratch: &mut Vec<u32>,
    ) -> Vec<(f32, usize)> {
        let entry_dist = match self.nodes.get(entry_point) {
            Some(n) => self.distance.distance(query, &n.stored),
            None => return Vec::new(),
        };

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

            // Snapshot c_idx's neighbors under a brief read lock, then score them
            // lock-free.
            self.copy_neighbors(c_idx, layer, scratch);
            for &nbr_u in scratch.iter() {
                let nbr = nbr_u as usize;
                if !visited.insert(nbr) {
                    continue;
                }
                let n = match self.nodes.get(nbr) {
                    Some(n) => n,
                    None => continue, // in-flight; skip
                };
                if !n.ready.load(Ordering::Acquire) {
                    continue; // mid-construction; not a safe routing target yet
                }
                let dist = self.distance.distance(query, &n.stored);
                let worst_dist = results
                    .peek()
                    .map(|(OrderedFloat(d), _)| *d)
                    .unwrap_or(f32::MAX);
                if dist < worst_dist || results.len() < ef {
                    candidates.push(Reverse((OrderedFloat(dist), nbr)));
                    results.push((OrderedFloat(dist), nbr));
                    if results.len() > ef {
                        results.pop();
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

    /// A parallel build must produce a complete, well-connected index whose recall
    /// is on par with a serial build (topologically equivalent up to the
    /// nondeterministic insertion order). Run in DEBUG too, so the lock-discipline
    /// guard (one node lock at a time, leaf-lock ordering) is actively checked.
    #[test]
    fn parallel_build_matches_serial_recall() {
        let dims = 32;
        let n = 3000;
        let k = 10;
        let cfg = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let vectors = random_vectors(n, dims);

        let serial = HnswIndex::new(cfg.clone());
        for (id, v) in &vectors {
            serial.insert(*id, v);
        }

        let parallel = HnswIndex::new(cfg.clone());
        parallel.insert_parallel(&vectors);

        assert_eq!(parallel.len(), n);
        assert_eq!(parallel.active_count(), n);
        for (id, _) in &vectors {
            assert!(parallel.contains_id(*id), "missing id {id} after parallel build");
        }

        let recall = |idx: &HnswIndex<ExactDistance>| -> f32 {
            let nq = 40;
            let mut total = 0.0f32;
            for qi in 0..nq {
                let query = &vectors[qi * 13 % n].1;
                let mut exact: Vec<(f32, u64)> = vectors
                    .iter()
                    .map(|(id, v)| (crate::distance::l2_squared(query, v), *id))
                    .collect();
                exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let truth: HashSet<u64> = exact.iter().take(k).map(|(_, id)| *id).collect();
                let found: HashSet<u64> =
                    idx.search(query, k, 100).iter().map(|(id, _)| *id).collect();
                total += truth.intersection(&found).count() as f32 / k as f32;
            }
            total / nq as f32
        };
        let serial_recall = recall(&serial);
        let parallel_recall = recall(&parallel);
        assert!(
            parallel_recall >= serial_recall - 0.1,
            "parallel recall {parallel_recall} too far below serial {serial_recall}"
        );
    }

    /// Stress the concurrent-insert paths directly: many threads calling `insert`
    /// on a FRESH index (hammering the atomic empty-init), then on a shared one.
    /// Asserts completeness + connectivity; the in-debug lock-discipline guard
    /// turns any nested/mis-ordered lock into a panic here. A `save`/`load`
    /// round-trip confirms the concurrently-built graph serializes correctly.
    #[test]
    fn concurrent_insert_stress() {
        let cfg = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(cfg.clone());
        let vectors = random_vectors(800, 16);

        std::thread::scope(|s| {
            for chunk in vectors.chunks(25) {
                let index = &index;
                s.spawn(move || {
                    for (id, v) in chunk {
                        index.insert(*id, v);
                    }
                });
            }
        });

        assert_eq!(index.len(), 800);
        assert_eq!(index.active_count(), 800);
        for (id, _) in &vectors {
            assert!(index.contains_id(*id), "missing id {id} after concurrent build");
        }
        // Connectivity: a search should reach a large fraction of the graph.
        let results = index.search(&vectors[0].1, 50, 200);
        assert!(
            results.len() >= 45,
            "concurrent build left the graph poorly connected: search returned {}",
            results.len()
        );

        // save/load round-trip of the concurrently-built graph.
        let mut buf = Vec::new();
        index.save(&mut buf).unwrap();
        let loaded = HnswIndex::load(&mut &buf[..], ExactDistance { metric: Metric::L2 }).unwrap();
        assert_eq!(loaded.len(), 800);
        assert_eq!(loaded.search(&vectors[0].1, 50, 200).len(), results.len());
    }

    /// Re-inserting an existing id (an update, incl. a concurrent same-id race)
    /// must tombstone the prior node so search returns the id at most once with the
    /// NEW vector — not a stale/duplicate result.
    #[test]
    fn reinsert_same_id_tombstones_old() {
        let cfg = HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            metric: Metric::L2,
        };
        let index = HnswIndex::new(cfg);
        let vectors = random_vectors(200, 16);
        for (id, v) in &vectors {
            index.insert(*id, v);
        }
        // Update id 5 to sit right next to id 7 (so it stays well-connected and
        // findable — the point here is the tombstone, not recall on an outlier).
        let new5 = vectors[7].1.clone();
        index.insert(5, &new5);

        assert!(index.contains_id(5));
        assert_eq!(index.len(), 201, "re-insert pushes a new node");
        assert_eq!(index.active_count(), 200, "the prior id-5 node is tombstoned");

        // The updated id 5 is found (co-located with id 7) and appears AT MOST once
        // — the old node is tombstoned, so no duplicate / stale id 5.
        let res = index.search(&new5, 10, 100);
        let count5 = res.iter().filter(|(id, _)| *id == 5).count();
        assert!(count5 <= 1, "id 5 duplicated in results (tombstone failed): {count5}");
        assert!(
            res.iter().any(|(id, _)| *id == 5),
            "updated id 5 (now co-located with id 7) should be found"
        );
    }

    /// A corrupt snapshot with an implausible `max_layer` must be rejected (it drives
    /// the descent loops; a huge value would hang rather than fail fast).
    #[test]
    fn load_rejects_huge_max_layer() {
        let mut buf = Vec::new();
        buf.extend_from_slice(HNSW_MAGIC);
        write_u32(&mut buf, 2).unwrap(); // version 2
        write_u32(&mut buf, 8).unwrap(); // m
        write_u32(&mut buf, 16).unwrap(); // m_max0
        write_u32(&mut buf, 50).unwrap(); // ef_construction
        write_u8(&mut buf, Metric::L2.to_u8()).unwrap();
        write_u64(&mut buf, 1).unwrap(); // num_nodes
        write_i64(&mut buf, 0).unwrap(); // entry_point = 0
        write_u32(&mut buf, 1000).unwrap(); // max_layer = 1000 (>> 64): invalid
        let distance = ExactDistance { metric: Metric::L2 };
        assert!(
            HnswIndex::load(&mut &buf[..], distance).is_err(),
            "snapshot with out-of-range max_layer must be rejected"
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
