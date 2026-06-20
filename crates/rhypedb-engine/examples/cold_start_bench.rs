//! Cold-start latency benchmark — Phase 6 (scale-to-zero readiness).
//!
//! Measures how long a server takes to become query-ready after a restart,
//! isolating the two costs that dominate a vector DB's cold start:
//!   1. `Database::open` — LSM open + WAL replay.
//!   2. `Vectorizer::new` — HNSW index materialization, which is EITHER a fast
//!      load from an `hnsw_*.bin` snapshot (saved on graceful shutdown) OR a full
//!      rebuild from the LSM `v:` keys when the snapshot is absent/mismatched.
//!
//! The snapshot-vs-rebuild gap is exactly what graceful-shutdown's snapshot save
//! buys us, and it informs the object-storage restore decision: a snapshot is a
//! plain file that ships with a physical backup, so restoring it avoids the
//! rebuild on wake.
//!
//! Run (release for realistic numbers):
//!   cargo run --release -p rhypedb-engine --example cold_start_bench
//!   cargo run --release -p rhypedb-engine --example cold_start_bench -- 1000,10000,100000 384
//!
//! Args: [sizes (comma-separated)] [dim]. Defaults: 1000,10000,100000  384.
//! Uses bring-your-own `Vector` fields, so NO embedding model is needed.

use rhypedb_engine::database::Database;
use rhypedb_engine::vectorizer::Vectorizer;
use rhypedb_schema::Schema;
use rhypedb_schema::parser::parse_schema;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const INGEST_CHUNK: usize = 5_000;

fn schema_for(dim: usize) -> Schema {
    parse_schema(&format!(
        "type Doc {{\n    v: Vector<{dim}>\n}}\n"
    ))
    .expect("schema parse")
}

fn open(dir: &Path, schema: &Schema) -> (Arc<Database>, Vectorizer, f64, f64) {
    // A prior handle on this dir may still be releasing its data-dir flock: at
    // large N the build phase leaves a background compaction running that holds
    // the LsmTree (and its LOCK) alive transiently past `drop()`. In production a
    // cold start is a fresh PROCESS (the OS releases the lock on exit), so this
    // only affects this in-process bench. Retry the open until the lock is free,
    // and time ONLY the successful attempt (the wait happens before it starts).
    let db = loop {
        let t = Instant::now();
        match Database::open(schema.clone(), dir) {
            Ok(db) => break (db, t.elapsed().as_secs_f64() * 1e3),
            Err(e) if e.to_string().contains("is locked") => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => panic!("db open: {e}"),
        }
    };
    let (db, db_open_ms) = db;

    let t = Instant::now();
    let vectorizer = Vectorizer::new(
        Arc::clone(db.storage()),
        schema.clone(),
        db.type_ids().clone(),
        db.field_ids().clone(),
    )
    .expect("vectorizer new");
    let vec_ms = t.elapsed().as_secs_f64() * 1e3;
    (db, vectorizer, db_open_ms, vec_ms)
}

/// Deterministic pseudo-random unit-ish vector generator (splitmix64), so the
/// benchmark is reproducible run-to-run without a `rand` dependency.
fn gen_chunk(start_id: u64, count: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count as u64 {
        let id = start_id + i;
        let mut state = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // map to [-1, 1)
            ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        };
        let v: Vec<f32> = (0..dim).map(|_| next()).collect();
        out.push((id, v));
    }
    out
}

fn dir_size_bytes(dir: &Path, prefix: &str, suffix: &str) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(prefix)
                && name.ends_with(suffix)
                && let Ok(m) = entry.metadata()
            {
                total += m.len();
            }
        }
    }
    total
}

fn delete_snapshots(dir: &Path) -> usize {
    let mut n = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("hnsw_")
                && name.ends_with(".bin")
                && std::fs::remove_file(entry.path()).is_ok()
            {
                n += 1;
            }
        }
    }
    n
}

fn run_size(n: usize, dim: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path();
    let schema = schema_for(dim);

    // ---- Build phase: ingest N vectors, then save the HNSW snapshot. ----
    let (db, vectorizer, _, _) = open(path, &schema);
    let t = Instant::now();
    let mut ingested = 0u64;
    while (ingested as usize) < n {
        let count = INGEST_CHUNK.min(n - ingested as usize);
        let rows = gen_chunk(ingested + 1, count, dim);
        ingested += vectorizer
            .ingest_vectors("Doc", "v", &rows)
            .expect("ingest") as u64;
    }
    let ingest_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    vectorizer.save_snapshots();
    let snap_save_ms = t.elapsed().as_secs_f64() * 1e3;
    let snap_bytes = dir_size_bytes(path, "hnsw_", ".bin");
    let sst_bytes = dir_size_bytes(&path.join("sst"), "", ".sst");
    drop(vectorizer);
    drop(db);

    // ---- Reopen A: snapshot present -> fast load + zero delta. ----
    let (db, vectorizer, db_open_a, vec_load_ms) = open(path, &schema);
    drop(vectorizer);
    drop(db);

    // ---- Reopen B: snapshot deleted -> full HNSW rebuild from the LSM. ----
    let removed = delete_snapshots(path);
    assert!(removed >= 1, "expected at least one snapshot to delete");
    let (db, vectorizer, db_open_b, vec_rebuild_ms) = open(path, &schema);
    drop(vectorizer);
    drop(db);

    let db_open_ms = (db_open_a + db_open_b) / 2.0;
    let speedup = if vec_load_ms > 0.0 {
        vec_rebuild_ms / vec_load_ms
    } else {
        f64::INFINITY
    };

    println!(
        "{n:>9} | {dim:>4} | {ingest_ms:>10.1} | {db_open_ms:>9.2} | {vec_load_ms:>11.2} | {vec_rebuild_ms:>12.2} | {speedup:>7.1}x | {snap_save:>9.1} | {snap_mb:>8.2} | {sst_mb:>7.2}",
        snap_save = snap_save_ms,
        snap_mb = snap_bytes as f64 / 1e6,
        sst_mb = sst_bytes as f64 / 1e6,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1_000, 10_000, 100_000]);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(384);

    println!("rhypedb cold-start benchmark (dim={dim}, BYO vectors, release recommended)");
    println!(
        "  reopen times are: Database::open (LSM+WAL) and Vectorizer::new (HNSW load vs full rebuild)\n"
    );
    println!(
        "{:>9} | {:>4} | {:>10} | {:>9} | {:>11} | {:>12} | {:>8} | {:>9} | {:>8} | {:>7}",
        "vectors", "dim", "ingest ms", "open ms", "hnsw load", "hnsw rebld", "rebld/ld", "snap ms", "snap MB", "sst MB"
    );
    println!("{}", "-".repeat(118));
    for n in sizes {
        run_size(n, dim);
    }
}
