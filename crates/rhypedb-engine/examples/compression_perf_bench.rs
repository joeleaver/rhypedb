//! Compression / compaction / traversal / string-index performance bench.
//!
//! Answers: how much does the v6 per-block LZ4 default cost on the hot paths
//! (compaction, multi-hop traversal reads, @indexed String filter scans), and
//! does background compaction keep the writer's tail latency flat? Every read/
//! compaction section runs the SAME workload twice — `None` (v5 uncompressed,
//! zero-copy) and `Lz4` (v6) — and prints both plus the ratio.
//!
//! Sections:
//!   1. Compaction cost (storage-level KV): write N rows across many SSTs, time
//!      `compact()`. None vs Lz4. (Card: LZ4 compaction cost + async-compaction.)
//!   2. Async-compaction tail (storage-level): per-commit latency with
//!      `background_compaction` on vs off. (Card cmq5gow93.)
//!   3. Multi-hop traversal (engine): user -> ratings -> movies, reads served
//!      from cover blobs in SSTs. None vs Lz4. (Card cmq5goufq, rhypedb side.)
//!   4. @indexed String filter scan (engine). None vs Lz4. (Card cmq5gozng.)
//!
//! Run (release — debug numbers are meaningless):
//!   cargo run --release -p rhypedb-engine --example compression_perf_bench
//!   cargo run --release -p rhypedb-engine --example compression_perf_bench -- 1000000
//!
//! Arg: [compaction_rows]. Default 300000. Deterministic (splitmix64).

use bytes::Bytes;
use rhypedb_engine::database::{Database, OpenOptions};
use rhypedb_engine::object::{FieldMap, Value};
use rhypedb_schema::parser::parse_schema;
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_storage::SstCompression;
use std::sync::Arc;
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = p.metadata() {
                total += m.len();
            }
        }
    }
    total
}

fn lsm_config(dir: &std::path::Path, compression: SstCompression, background: bool) -> LsmConfig {
    LsmConfig {
        data_dir: dir.to_path_buf(),
        memtable_flush_size: 4 * 1024 * 1024,
        compact_trigger_ssts: 4,
        zone_extractor: None,
        sync_on_commit: false, // measure compaction/throughput, not fsync
        background_compaction: background,
        block_compression: compression,
    }
}

/// A compressible KV value (repeated structure, like a serialized FieldMap).
fn kv_value(id: u64, rng: &mut u64) -> Bytes {
    let tag = ["alpha", "bravo", "charlie", "delta", "echo"][(splitmix64(rng) % 5) as usize];
    Bytes::from(format!(
        "{{\"id\":{id},\"name\":\"item_{id}\",\"tag\":\"{tag}\",\"flag\":true,\"note\":\"lorem ipsum dolor sit amet\"}}"
    ))
}

// ---------------------------------------------------------------------------
// Section 1: compaction cost (storage-level), None vs Lz4.
// ---------------------------------------------------------------------------
fn bench_compaction(n: u64, compression: SstCompression) -> (f64, f64, u64) {
    let dir = std::env::temp_dir().join(format!("rhypedb-cperf-compact-{compression:?}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sst")).unwrap();
    // background OFF so we control exactly when compaction runs (clean timing).
    let tree = LsmTree::open(lsm_config(&dir, compression, false)).unwrap();

    let flush_every = 25_000u64;
    let mut rng = 0xABCD_1234u64;
    let t_write = Instant::now();
    for i in 0..n {
        let mut txn = tree.begin_txn();
        let key = format!("k:{i:012}");
        tree.put(&mut txn, key.as_bytes(), kv_value(i, &mut rng)).unwrap();
        tree.commit(&mut txn).unwrap();
        if (i + 1) % flush_every == 0 {
            tree.flush().unwrap();
        }
    }
    tree.flush().unwrap();
    let write_s = t_write.elapsed().as_secs_f64();

    let ssts_before = tree.sst_count();
    let t_compact = Instant::now();
    tree.compact().unwrap();
    let compact_s = t_compact.elapsed().as_secs_f64();
    let on_disk = dir_size(&dir);
    println!(
        "  {compression:?}: write {write_s:.2}s ({:.0} rows/s) | compact {ssts_before} SSTs in {compact_s:.3}s | on-disk {:.1} MB",
        n as f64 / write_s,
        on_disk as f64 / 1e6
    );
    drop(tree);
    let _ = std::fs::remove_dir_all(&dir);
    (write_s, compact_s, on_disk)
}

// ---------------------------------------------------------------------------
// Section 2: async-compaction tail latency (storage-level), bg on vs off.
// ---------------------------------------------------------------------------
fn bench_async_tail(n: u64, background: bool) {
    let dir = std::env::temp_dir().join(format!("rhypedb-cperf-async-{background}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sst")).unwrap();
    // Small memtable so flushes (and the 4-SST compaction trigger) fire often.
    let mut cfg = lsm_config(&dir, SstCompression::Lz4, background);
    cfg.memtable_flush_size = 512 * 1024;
    let tree = LsmTree::open(cfg).unwrap();

    let mut rng = 0x5151_2727u64;
    let mut max_commit_us = 0.0f64;
    let mut over_10ms = 0u64;
    let t = Instant::now();
    for i in 0..n {
        let mut txn = tree.begin_txn();
        let key = format!("k:{i:012}");
        tree.put(&mut txn, key.as_bytes(), kv_value(i, &mut rng)).unwrap();
        let c = Instant::now();
        tree.commit(&mut txn).unwrap();
        let us = c.elapsed().as_secs_f64() * 1e6;
        if us > max_commit_us {
            max_commit_us = us;
        }
        if us > 10_000.0 {
            over_10ms += 1;
        }
    }
    tree.wait_for_compaction();
    let total_s = t.elapsed().as_secs_f64();
    let label = if background { "bg ON " } else { "bg OFF" };
    println!(
        "  {label} (lz4): total {total_s:.2}s ({:.0} rows/s) | worst commit {max_commit_us:.0} us | commits >10ms: {over_10ms}",
        n as f64 / total_s
    );
    drop(tree);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Engine sections: shared graph setup.
// ---------------------------------------------------------------------------
const TRAVERSAL_SCHEMA: &str = r#"
type User {
    name: String
    city: String @indexed
    ratings: [Rating] @inverse(Rating.user)
}
type Movie {
    title: String
    ratings: [Rating] @inverse(Rating.movie)
}
type Rating {
    user: User @on_delete(cascade)
    movie: Movie @on_delete(cascade)
    stars: f32
}
"#;

fn open_engine(dir: &std::path::Path, compression: SstCompression) -> Arc<Database> {
    let schema = parse_schema(TRAVERSAL_SCHEMA).unwrap();
    Database::open_with_options(
        schema,
        dir,
        OpenOptions {
            sync_on_commit: false,
            block_compression: compression,
            ..Default::default()
        },
    )
    .unwrap()
}

fn city(i: u64) -> String {
    format!("city-{:03}", i % 200)
}

/// Build a User/Movie/Rating graph and settle it into compacted SSTs.
fn build_graph(
    db: &Database,
    n_users: u64,
    n_movies: u64,
    ratings_per_user: u64,
) -> (Vec<u64>, ()) {
    // Users (with an @indexed city) + movies, via bulk batches.
    let mut user_ids = Vec::with_capacity(n_users as usize);
    for chunk in (0..n_users).collect::<Vec<_>>().chunks(5000) {
        let rows: Vec<FieldMap> = chunk
            .iter()
            .map(|&i| {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String(format!("user_{i}")));
                f.insert("city".into(), Value::String(city(i)));
                f
            })
            .collect();
        for o in db.create_batch("User", rows).unwrap() {
            user_ids.push(o.id);
        }
    }
    let mut movie_ids = Vec::with_capacity(n_movies as usize);
    for chunk in (0..n_movies).collect::<Vec<_>>().chunks(5000) {
        let rows: Vec<FieldMap> = chunk
            .iter()
            .map(|&i| {
                let mut f = FieldMap::new();
                f.insert("title".into(), Value::String(format!("movie_{i}")));
                f
            })
            .collect();
        for o in db.create_batch("Movie", rows).unwrap() {
            movie_ids.push(o.id);
        }
    }
    // Ratings: each links a user to a movie (forward `movie` + inverse fills
    // User.ratings / Movie.ratings).
    let mut rng = 0x9999_7777u64;
    for &uid in &user_ids {
        for _ in 0..ratings_per_user {
            let mid = movie_ids[(splitmix64(&mut rng) % movie_ids.len() as u64) as usize];
            let mut f = FieldMap::new();
            f.insert("stars".into(), Value::F32((splitmix64(&mut rng) % 5 + 1) as f32));
            let rating = db.create("Rating", f).unwrap();
            db.link("Rating", rating.id, "user", uid, None).unwrap();
            db.link("Rating", rating.id, "movie", mid, None).unwrap();
        }
    }
    // Settle into SSTs + compact so reads exercise the (decompressing) SST path.
    db.storage().flush().unwrap();
    db.storage().compact().unwrap();
    db.storage().wait_for_compaction();
    (user_ids, ())
}

fn bench_engine(compression: SstCompression, n_users: u64, n_movies: u64, rpu: u64) {
    let dir = std::env::temp_dir().join(format!("rhypedb-cperf-engine-{compression:?}"));
    let _ = std::fs::remove_dir_all(&dir);
    let db = open_engine(&dir, compression);

    let t_build = Instant::now();
    let (user_ids, _) = build_graph(&db, n_users, n_movies, rpu);
    let build_s = t_build.elapsed().as_secs_f64();
    let on_disk = dir_size(&dir);

    // --- Traversal: user -> ratings -> movies, over a sample of users. ---
    let sample: Vec<u64> = user_ids.iter().copied().step_by((n_users / 1000).max(1) as usize).collect();
    // Warm, then time.
    let mut touched = 0usize;
    let run = |touched: &mut usize| {
        for &uid in &sample {
            for (rid, _rf) in db.get_links("User", uid, "ratings").unwrap() {
                for (_mid, mf) in db.get_links("Rating", rid, "movie").unwrap() {
                    *touched += mf.len();
                }
            }
        }
    };
    run(&mut touched);
    let t_trav = Instant::now();
    touched = 0;
    run(&mut touched);
    let trav_s = t_trav.elapsed().as_secs_f64();
    std::hint::black_box(touched);

    // --- @indexed String filter scan: city == "city-042". ---
    use rhypedb_storage::zone::CompareOp;
    let mut hits = 0usize;
    for _ in 0..2 {
        hits = db
            .filter_scan_str("User", "city", CompareOp::Eq, "city-042", None)
            .unwrap()
            .len();
    }
    let t_filt = Instant::now();
    let iters = 200u32;
    for _ in 0..iters {
        std::hint::black_box(
            db.filter_scan_str("User", "city", CompareOp::Eq, "city-042", None)
                .unwrap()
                .len(),
        );
    }
    let filt_us = t_filt.elapsed().as_secs_f64() * 1e6 / iters as f64;

    println!(
        "  {compression:?}: build {build_s:.2}s, on-disk {:.1} MB | traversal({} users) {:.2} ms | filter_scan_str(city) {filt_us:.0} us ({hits} hits)",
        on_disk as f64 / 1e6,
        sample.len(),
        trav_s * 1e3,
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let n_compact: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000);

    println!("=== Section 1: compaction cost (storage KV, {n_compact} rows, bg off) ===");
    let (_, c_none, sz_none) = bench_compaction(n_compact, SstCompression::None);
    let (_, c_lz4, sz_lz4) = bench_compaction(n_compact, SstCompression::Lz4);
    println!(
        "  -> compaction lz4/none = {:.2}x time; on-disk none/lz4 = {:.2}x",
        c_lz4 / c_none,
        sz_none as f64 / sz_lz4 as f64
    );

    let n_async = (n_compact / 3).max(50_000);
    println!("\n=== Section 2: async-compaction tail latency ({n_async} rows, lz4) ===");
    bench_async_tail(n_async, false);
    bench_async_tail(n_async, true);

    // Engine graph: moderate scale (link() is per-edge). ~n_users*rpu ratings.
    let n_users = 10_000u64;
    let n_movies = 1_000u64;
    let rpu = 5u64;
    println!(
        "\n=== Sections 3+4: engine traversal + string index ({n_users} users, {n_movies} movies, {} ratings) ===",
        n_users * rpu
    );
    bench_engine(SstCompression::None, n_users, n_movies, rpu);
    bench_engine(SstCompression::Lz4, n_users, n_movies, rpu);
}
