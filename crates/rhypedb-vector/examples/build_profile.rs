//! Decompose the HNSW build cost. Separates the per-vector quantization work
//! (compress + prepare_query matvecs) from the graph-construction work
//! (search_layer distance calls + neighbor pruning/linking + locks + node mgmt),
//! so we know which lever matters: SIMD/scratch on the matvecs, vs the graph ops
//! (where parallelism + cheaper neighbor lookups would help).
//!
//! Run:  cargo run --release --example build_profile -p rhypedb-vector -- <N> [bits] [dims]

use std::hint::black_box;
use std::time::Instant;

use rhypedb_vector::distance::Metric;
use rhypedb_vector::hnsw::HnswConfig;
use rhypedb_vector::index::QuantizedIndex;
use rhypedb_vector::quantize::{TurboQuantConfig, TurboQuantizer};

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg(s | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let bits: u8 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let dims: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(384);

    let mut rng = Lcg::new(0xBEEF ^ n as u64);
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dims).map(|_| rng.next_f32()).collect()).collect();
    let cfg = TurboQuantConfig::new(dims as u32, bits);

    // 1) compress() alone (2 matvecs + quantize per vector).
    let q = TurboQuantizer::new(cfg.clone());
    let t = Instant::now();
    for v in &vecs {
        black_box(q.compress(v));
    }
    let t_compress = t.elapsed().as_secs_f64();

    // 2) prepare_query() alone (2 matvecs per vector — the per-insert query proj).
    let t = Instant::now();
    for v in &vecs {
        black_box(q.prepare_query(v));
    }
    let t_prepare = t.elapsed().as_secs_f64();

    // 3) full SERIAL index build.
    let hnsw_cfg = HnswConfig { m: 16, m_max0: 32, ef_construction: 100, metric: Metric::Cosine };
    let index = QuantizedIndex::new(hnsw_cfg.clone(), cfg.clone());
    let t = Instant::now();
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v);
    }
    let t_build = t.elapsed().as_secs_f64();

    // 4) full PARALLEL index build (per-node-locked fine-grained insert).
    let items: Vec<(u64, Vec<f32>)> =
        vecs.iter().enumerate().map(|(i, v)| (i as u64, v.clone())).collect();
    let index_p = QuantizedIndex::new(hnsw_cfg, cfg);
    let t = Instant::now();
    index_p.insert_parallel(&items);
    let t_build_p = t.elapsed().as_secs_f64();
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    assert_eq!(index_p.len(), n, "parallel build dropped vectors");

    let graph = t_build - t_compress - t_prepare;
    let us = |s: f64| s / n as f64 * 1e6;
    println!("--- build profile  n={n} dims={dims} bits={bits} ---");
    println!("  compress():        {t_compress:8.2}s  ({:7.1} us/vec)", us(t_compress));
    println!("  prepare_query():   {t_prepare:8.2}s  ({:7.1} us/vec)", us(t_prepare));
    println!("  full build SERIAL: {t_build:8.2}s  ({:7.1} us/vec)  = {:.0} ins/s", us(t_build), n as f64 / t_build);
    println!("  graph ops (build - compress - prepare): {graph:8.2}s  ({:7.1} us/vec, {:.0}%)",
        us(graph), 100.0 * graph / t_build);
    println!("  full build PARALLEL ({cores} cores): {t_build_p:8.2}s  = {:.0} ins/s  (speedup {:.2}x)",
        n as f64 / t_build_p, t_build / t_build_p);
}
