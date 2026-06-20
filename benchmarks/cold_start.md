# Cold-start latency (scale-to-zero readiness)

Phase 6 item 3. How long does a `rhypedb-server` take to become query-ready after
a restart, and what dominates that time?

A vector DB's cold start has two costs:

1. **`Database::open`** — LSM open + WAL replay. Cheap.
2. **`Vectorizer::new`** — HNSW index materialization. This is EITHER a fast load
   from an `hnsw_*.bin` snapshot (saved on graceful shutdown) OR a full rebuild
   from the LSM `v:` keys when the snapshot is absent/stale/config-mismatched.

The rebuild-vs-load gap is exactly what the graceful-shutdown snapshot save buys.

## How to run

```sh
cargo run --release -p rhypedb-engine --example cold_start_bench
# custom sizes + dim:
cargo run --release -p rhypedb-engine --example cold_start_bench -- 1000,10000,100000,1000000 384
```

The bench (`crates/rhypedb-engine/examples/cold_start_bench.rs`) ingests N
bring-your-own vectors (no embedding model needed), saves the HNSW snapshot, then
reopens twice — once with the snapshot present (load path) and once with it
deleted (full rebuild path) — timing `Database::open` and `Vectorizer::new`
separately. Vectors are deterministic (splitmix64), so runs are reproducible.

## Results (dim 384, release, 1–4 core dev box)

| vectors | ingest | `Database::open` (LSM+WAL) | HNSW **load** (snapshot) | HNSW **rebuild** (from LSM) | rebuild ÷ load | snapshot file | sst |
|--------:|-------:|--------------------------:|-------------------------:|----------------------------:|---------------:|--------------:|----:|
| 1,000 | 51 ms | 1.6 ms | 29 ms | 65 ms | 2.2× | 0.9 MB | — |
| 10,000 | 0.54 s | 3.0 ms | 36 ms | 455 ms | 12.8× | 4.1 MB | 16 MB |
| 100,000 | 8.6 s | 34 ms | 137 ms | 7.87 s | 57.3× | 36 MB | 163 MB |
| 1,000,000 | 135 s | 166 ms | **1.49 s** | **121 s** | **81.3×** | 350 MB | 1.67 GB |

(dim 384, release, single dev box; deterministic vectors so runs are reproducible.)

## Takeaways

1. **HNSW rebuild dominates cold start; LSM open + WAL replay does not.** Even at
   1M vectors, opening the LSM and replaying the WAL is ~166 ms. The HNSW rebuild
   is ~121 s (two minutes) — ~700× larger, and it grows worse than linearly with
   vector count (58× the rebuild cost for 10× the vectors, 100k→1M).
2. **The HNSW snapshot is the lever.** Loading the snapshot is fast and scales far
   better than rebuilding: 137 ms vs 7.9 s at 100k (57×), and **1.5 s vs 121 s at
   1M (81×)**. Graceful shutdown already saves these snapshots; the cold start
   after a clean stop is therefore ~1.5 s at 1M vs ~2 minutes without.
3. **Implication for backup/restore (object-storage decision).** A physical
   backup already copies `hnsw_*.bin` alongside the SSTs (see
   `Database::backup_to`). Restoring those snapshot files means a restored
   instance skips the rebuild and wakes fast. **An object-storage restore should
   ship the HNSW snapshot, not just the SSTs** — otherwise every wake pays the
   full rebuild (multi-second per 100k vectors).
4. **"Acceptably fast restore on wake" (the acceptance criterion) is met IF the
   snapshot is present.** The risk cases are a SIGKILL (no graceful shutdown, so no
   fresh snapshot → rebuild on next open) and a restore that drops the `.bin`
   files. Budget roughly: rebuild ≈ the numbers above; load ≈ a few hundred ms even
   at 100k.

## Where this feeds

- Validates Phase-6 item 3 ("survives scale-to-zero: correct + acceptably-fast
  restore on wake").
- Informs the jkbase object-storage-vs-volume decision: snapshots must travel with
  the data, and the wake-time budget is dominated by whether the snapshot is present.
