//! Microbenchmark for the TurboQuant distance kernel and end-to-end search.
//!
//! Run with:  cargo run --release --example kernel_bench -p rhypedb-vector
//!
//! Measures two things, before/after kernel optimization:
//!   1. The isolated hot kernel `inner_product_estimate_prepared` — ns/call.
//!      This is the function called once per candidate during HNSW search and
//!      construction; it currently heap-allocates twice per call.
//!   2. End-to-end `QuantizedIndex::search` throughput (queries/sec) and recall,
//!      so we can confirm the optimization preserves accuracy and moves the
//!      number that actually matters.
//!
//! Determinism: uses a fixed-seed LCG so before/after runs see identical data.

use std::hint::black_box;
use std::time::Instant;

use rhypedb_vector::distance::Metric;
use rhypedb_vector::hnsw::HnswConfig;
use rhypedb_vector::index::{brute_force_knn, recall_at_k, QuantizedIndex};
use rhypedb_vector::quantize::{TurboQuantConfig, TurboQuantizer};

/// Tiny deterministic PRNG (xorshift64*) so the bench is reproducible without a
/// dependency on rand's thread RNG.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        // map top 24 bits to [-1, 1)
        let u = (v >> 40) as f32 / (1u32 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

fn random_vec(rng: &mut Lcg, dims: usize) -> Vec<f32> {
    (0..dims).map(|_| rng.next_f32()).collect()
}

fn bench_kernel(dims: usize, bits: u8, n: usize, rounds: usize) {
    let mut rng = Lcg::new(0xC0FFEE ^ (dims as u64) ^ ((bits as u64) << 32));
    let quantizer = TurboQuantizer::new(TurboQuantConfig::new(dims as u32, bits));

    let compressed: Vec<_> = (0..n)
        .map(|_| quantizer.compress(&random_vec(&mut rng, dims)))
        .collect();
    let query = random_vec(&mut rng, dims);
    let prepared = quantizer.prepare_query(&query);

    // Warmup.
    let mut acc = 0.0f32;
    for cv in &compressed {
        acc += quantizer.inner_product_estimate_prepared(&prepared, cv);
    }
    black_box(acc);

    let mut best_ns_per_call = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        let mut acc = 0.0f32;
        for cv in &compressed {
            acc += quantizer.inner_product_estimate_prepared(black_box(&prepared), black_box(cv));
        }
        black_box(acc);
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / n as f64;
        if ns_per_call < best_ns_per_call {
            best_ns_per_call = ns_per_call;
        }
    }

    let calls_per_sec = 1e9 / best_ns_per_call;
    println!(
        "  kernel  dims={dims:<5} bits={bits}  n={n:<6}  {best_ns_per_call:>8.2} ns/call  \
         {:>8.2} M calls/s",
        calls_per_sec / 1e6
    );
}

fn bench_search(dims: usize, bits: u8, n: usize, n_queries: usize) {
    let mut rng = Lcg::new(0xBEEF ^ (dims as u64) ^ ((bits as u64) << 16));
    let hnsw_config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        metric: Metric::L2,
    };
    let index = QuantizedIndex::new(hnsw_config, TurboQuantConfig::new(dims as u32, bits));

    let vectors: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| (i as u64, random_vec(&mut rng, dims)))
        .collect();

    let build_start = Instant::now();
    for (id, v) in &vectors {
        index.insert(*id, v);
    }
    let build_secs = build_start.elapsed().as_secs_f64();

    let k = 10;
    let ef = 100;
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|q| vectors[q * 7 % n].1.clone()).collect();

    // Warmup.
    for q in &queries {
        black_box(index.search(q, k, ef));
    }

    let search_start = Instant::now();
    let mut sink = 0usize;
    for q in &queries {
        let r = index.search(black_box(q), k, ef);
        sink += r.len();
    }
    let search_secs = search_start.elapsed().as_secs_f64();
    black_box(sink);

    // Recall vs brute force.
    let mut total_recall = 0.0f32;
    for q in queries.iter().take(20) {
        let truth = brute_force_knn(q, &vectors, k, Metric::L2);
        let found = index.search(q, k, ef);
        total_recall += recall_at_k(&found, &truth, k);
    }
    let recall = total_recall / queries.len().min(20) as f32;

    let qps = n_queries as f64 / search_secs;
    let inserts_per_sec = n as f64 / build_secs;
    println!(
        "  search  dims={dims:<5} bits={bits}  n={n:<6}  build={inserts_per_sec:>9.0} ins/s  \
         {qps:>8.1} qps  recall@{k}={recall:.3}"
    );
}

fn main() {
    println!("=== TurboQuant kernel + build microbench ===");
    // Isolated search kernel — unchanged by the build-path work; run small as a
    // sanity check that the distance estimate didn't move.
    println!("-- isolated kernel (sanity) --");
    for &bits in &[2u8, 3, 4] {
        bench_kernel(384, bits, 10_000, 5);
    }
    // Build + search. `build=… ins/s` is the figure of interest for the build
    // optimization (prepare_stored): HNSW construction is dominated by
    // `prepare_stored`'s O(d²) matmuls during neighbor pruning. `qps` and
    // `recall@10` confirm search throughput and accuracy are preserved. Two
    // dims so the O(d²) win is visible; sizes kept modest for a quick A/B.
    println!("-- build + search (QuantizedIndex) --");
    bench_search(384, 3, 6_000, 300);
    bench_search(384, 4, 6_000, 300);
    bench_search(768, 4, 2_000, 300);
}
