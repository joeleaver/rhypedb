//! SST per-block LZ4 compression benchmark (v5 uncompressed vs v6 LZ4).
//!
//! Writes the SAME synthetic dataset twice — once with `SstCompression::None`
//! (v5) and once with `Lz4` (v6) — then reports:
//!   * on-disk file size + compression ratio,
//!   * point-read latency (`get_versioned` over random keys, warm cache),
//!   * full-scan throughput (`iter`).
//!
//! Warm-cache numbers isolate the decompression CPU cost; the disk-size win is
//! what drives the cold-cache / page-cache-density benefit in production.
//!
//! Run (release for realistic numbers):
//!   cargo run --release -p rhypedb-storage --example sst_compression_bench
//!   cargo run --release -p rhypedb-storage --example sst_compression_bench -- 200000
//!
//! Args: [num_entries]. Default: 100000. Deterministic (splitmix64).

use bytes::Bytes;
use rhypedb_storage::key::{InternalKey, KeyBuilder};
use rhypedb_storage::sst::{SstCompression, SstReader, SstWriter};
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A realistic-ish object value: a small "JSON-like" record with repeated field
/// names (compressible, as real serialized FieldMaps are) plus some entropy.
fn make_value(id: u64, rng: &mut u64) -> Bytes {
    let city = ["London", "Paris", "Tokyo", "Berlin", "Madrid"][(splitmix64(rng) % 5) as usize];
    let score = splitmix64(rng) % 1000;
    Bytes::from(format!(
        "{{\"type\":\"User\",\"id\":{id},\"name\":\"user_{id}\",\"city\":\"{city}\",\
         \"active\":true,\"score\":{score},\"bio\":\"the quick brown fox jumps over the lazy dog\"}}"
    ))
}

fn build(path: &std::path::Path, n: u64, compression: SstCompression) {
    let mut w = SstWriter::new_with_options(path, None, compression).unwrap();
    let mut rng = 0x1234_5678u64;
    for id in 0..n {
        let key = InternalKey::new(&KeyBuilder::object(1, id), 1);
        w.add(key.as_bytes(), &Some(make_value(id, &mut rng))).unwrap();
    }
    w.finish().unwrap();
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dir = std::env::temp_dir().join(format!("rhypedb-sst-comp-bench-{n}"));
    let _ = std::fs::create_dir_all(&dir);
    let p5 = dir.join("v5.sst");
    let p6 = dir.join("v6.sst");

    println!("Building {n} entries (v5 uncompressed + v6 lz4)...");
    build(&p5, n, SstCompression::None);
    build(&p6, n, SstCompression::Lz4);

    let s5 = std::fs::metadata(&p5).unwrap().len();
    let s6 = std::fs::metadata(&p6).unwrap().len();
    println!("\n=== file size ===");
    println!("  v5 (none): {:>12} bytes", s5);
    println!("  v6 (lz4):  {:>12} bytes", s6);
    println!("  ratio:     {:.2}x smaller", s5 as f64 / s6 as f64);

    let r5 = SstReader::open(&p5).unwrap();
    let r6 = SstReader::open(&p6).unwrap();

    // Point reads: random keys, warm cache (run twice, report the second).
    let probes: Vec<Vec<u8>> = {
        let mut rng = 0xDEAD_BEEFu64;
        (0..10_000)
            .map(|_| KeyBuilder::object(1, splitmix64(&mut rng) % n).to_vec())
            .collect()
    };
    let bench_get = |r: &SstReader| -> f64 {
        for p in &probes {
            let _ = r.get_versioned(p, 1).unwrap();
        }
        let t = Instant::now();
        for p in &probes {
            std::hint::black_box(r.get_versioned(p, 1).unwrap());
        }
        t.elapsed().as_secs_f64() * 1e9 / probes.len() as f64
    };
    println!("\n=== point read (get_versioned, warm) ===");
    println!("  v5: {:>8.0} ns/op", bench_get(&r5));
    println!("  v6: {:>8.0} ns/op", bench_get(&r6));

    // Full scan throughput.
    let bench_scan = |r: &SstReader| -> f64 {
        let t = Instant::now();
        let mut c = 0usize;
        for (k, v) in r.iter() {
            c += k.len() + v.map(|b| b.len()).unwrap_or(0);
        }
        std::hint::black_box(c);
        t.elapsed().as_secs_f64() * 1e6
    };
    println!("\n=== full scan (iter) ===");
    println!("  v5: {:>8.1} us", bench_scan(&r5));
    println!("  v6: {:>8.1} us", bench_scan(&r6));

    let _ = std::fs::remove_dir_all(&dir);
}
