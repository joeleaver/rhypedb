//! In-memory footprint of the TurboQuant HNSW index, ISOLATED from the server's
//! LSM/durability layer. The full rhypedb-server RSS additionally includes a copy
//! of the RAW f32 vectors in the LSM (the `v:` keyspace, the durable source for
//! rebuild-on-restart) — but those live in mmap'd SST files as reclaimable,
//! file-backed page cache (there is NO block cache), so they inflate VmRSS yet
//! are NOT part of the non-reclaimable serving heap and yield under memory
//! pressure. k-NN search only touches the in-memory HNSW index (nodes hold the
//! `CompressedVector` inline). This bench measures just that index — the honest
//! "RAM to serve" (RssAnon) number — to compare against pgvector's HNSW index size.
//!
//! It reports the index's footprint computed PRECISELY (walking the structure —
//! codes + graph + overhead), not via process RSS, which proved unreliable for an
//! in-process delta. Process RSS is printed too, only as a sanity cross-check.
//!
//! Run:  cargo run --release --example index_mem_bench -p rhypedb-vector -- <N> [bits] [dims]

use rhypedb_vector::distance::Metric;
use rhypedb_vector::hnsw::HnswConfig;
use rhypedb_vector::index::QuantizedIndex;
use rhypedb_vector::quantize::TurboQuantConfig;

fn rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:")
            && let Some(tok) = rest.split_whitespace().next()
        {
            return tok.parse().unwrap_or(0);
        }
    }
    0
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let bits: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let dims: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(384);

    let mut rng = Lcg::new(0xA11CE ^ n as u64);
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dims).map(|_| rng.next_f32()).collect()).collect();

    let cfg = TurboQuantConfig::new(dims as u32, bits);
    let codes = cfg.total_size();

    let rss_before = rss_kb();
    let index = QuantizedIndex::new(
        HnswConfig { m: 16, m_max0: 32, ef_construction: 100, metric: Metric::Cosine },
        cfg,
    );
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v);
    }
    let rss_after = rss_kb();

    let mem = index.memory_bytes();
    let total = mem.total();
    let bpv = total as f64 / n as f64;
    let f32_bpv = (dims * 4) as f64;

    println!("--- index footprint  n={n} dims={dims} bits={bits} ---");
    println!("  codes (compressed vectors): {:>10} B  ({:.0} B/vec)", mem.stored_bytes, mem.stored_bytes as f64 / n as f64);
    println!("  graph (neighbor lists):     {:>10} B  ({:.0} B/vec)", mem.graph_bytes, mem.graph_bytes as f64 / n as f64);
    println!("  node structs:               {:>10} B  ({:.0} B/vec)", mem.node_overhead, mem.node_overhead as f64 / n as f64);
    println!("  id->idx map:                {:>10} B  ({:.0} B/vec)", mem.id_map_bytes, mem.id_map_bytes as f64 / n as f64);
    println!("  TOTAL index:                {:>10} B  ({bpv:.0} B/vec)", total);
    println!("  per-vector codes config: {codes} B  |  raw f32 would be: {f32_bpv:.0} B/vec");
    println!(
        "  => codes {:.1}x smaller than raw f32; whole index {:.2}x vs a raw-f32+graph store",
        f32_bpv / codes as f64,
        (f32_bpv * n as f64 + mem.graph_bytes as f64 + mem.node_overhead as f64 + mem.id_map_bytes as f64) / total as f64
    );
    println!("  [cross-check] process RSS delta: {} KB ({:.0} B/vec)",
        rss_after.saturating_sub(rss_before),
        rss_after.saturating_sub(rss_before) as f64 * 1024.0 / n as f64);
}
