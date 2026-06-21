# SST per-block LZ4 compression (v6) — findings

Card cmpzqwhse. The SST data region can be compressed per 16-entry sparse-index
block with LZ4 (file format **v6**). Reads decompress one block at a time; the
compressed file stays memory-mapped (reclaimable page cache). v1–v5 files stay
readable and zero-copy; a tree can hold a mix, and compaction migrates files to
the configured format over time. Default: **on** (`block_compression = "lz4"`).

## Harness

`cargo run --release -p rhypedb-storage --example sst_compression_bench [-- N]`

Writes the same N synthetic objects (small JSON-like records — repeated field
names plus some entropy, representative of serialized FieldMaps) twice, as v5
(uncompressed) and v6 (LZ4), then measures file size, point-read latency
(`get_versioned` over 10k random keys, warm cache), and full-scan time (`iter`).
Deterministic (splitmix64).

## Result (N = 100,000, release, warm cache)

| Metric | v5 (none) | v6 (lz4) | Δ |
| --- | --- | --- | --- |
| File size | 17.80 MB | 3.98 MB | **4.47× smaller** |
| Point read (`get_versioned`) | 344 ns/op | 751 ns/op | ~2.2× slower |
| Full scan (`iter`) | 4296 µs | 5106 µs | ~1.19× slower |

## Reading the numbers

- **Disk / page cache (the win).** 4.47× smaller files exceed the ~3-4× target.
  Smaller files mean far more of the working set fits in the OS page cache, so
  the dominant production cost — *cold* reads that hit the disk — drops. The
  benchmark is warm-cache, so it does **not** show this win directly; it isolates
  the CPU cost paid to get the disk win.
- **Point reads pay a per-block decompress.** Each `get_versioned` decompresses
  one ~16-entry block (a few KB) to find one key, so warm-cache point reads are
  ~2× slower in absolute terms (+~400 ns). That is dwarfed by a single avoided
  cold read (µs–ms), which the denser page cache buys back many times over.
- **Scans barely move (+19%).** The block decompress is amortized across all 16
  entries, so sequential scans pay little.

## Tradeoff / when to turn it off

v6 reads return *copies* out of the decompressed block rather than zero-copy mmap
views (v5 behavior). For a latency-sensitive, fully-in-RAM workload that does
mostly point reads and is not page-cache-bound, `block_compression = "none"`
keeps the zero-copy path. For the common case (working set larger than RAM, or a
managed/scale-to-zero deployment where disk size and page-cache density matter),
`lz4` is the better default.

## End-to-end cost: compaction, traversal, string index (None vs Lz4)

`cargo run --release -p rhypedb-engine --example compression_perf_bench [-- N]`
runs the same workloads on both compression modes (storage-level KV for
compaction; engine-level graph for reads). "Did we lose much by defaulting LZ4
on?" — measured, release:

### Compaction cost (storage KV, background OFF for clean timing)

| Rows | compact (None) | compact (Lz4) | Δ | on-disk None | on-disk Lz4 | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 300k | 52 ms | 76 ms | 1.45× | 38.8 MB | 10.2 MB | 3.80× |
| 1M | 158 ms | 241 ms | 1.52× | 130.2 MB | 34.6 MB | 3.76× |

Compaction is ~1.5× more CPU under LZ4 (compress every merged block), but in
absolute terms it's tiny (241 ms to merge 40 SSTs / 1M rows) and it runs in the
background. Bulk-write throughput is ~5% lower (≈0.94M vs 0.99M rows/s). Disk is
3.8× smaller.

### Async compaction keeps the writer's tail flat (card cmq5gow93)

Per-commit latency with a small memtable so flushes + the 4-SST compaction
trigger fire often (LZ4):

| Rows | mode | total | worst commit | commits >10 ms |
| --- | --- | --- | --- | --- |
| 333k | bg OFF | 1.21 s | **66.7 ms** | 22 |
| 333k | bg ON | **0.58 s** | **18.5 ms** | 4 |

Background compaction (the default) cuts the worst-case commit ~3.6× and, at this
scale, halves total ingest time — because an inline compaction otherwise stalls
the writer synchronously, and LZ4's extra compaction CPU makes that stall *worse*,
so offloading it matters even more. The +50% compaction cost above is paid by a
background thread the writer never blocks on.

### Read paths (engine, data settled in compacted SSTs)

10k users / 1k movies / 50k ratings:

| Metric | None | Lz4 | Δ |
| --- | --- | --- | --- |
| 2-hop traversal (user→ratings→movies, 1000 users) | 14.1 ms | 14.6 ms | +4% |
| `filter_scan_str` on `@indexed` String (city ==) | 15 µs | 16 µs | ~noise |
| on-disk | 20.7 MB | 7.1 MB | 2.9× smaller |

At this small, cache-resident scale the read cost looked tiny (a few %). **That
estimate did not survive scale** — see below.

### ⚠ At scale, LZ4 costs 3.7× on multi-hop traversal (vs Postgres, 1M rows)

Running the real harness (`benchmarks/suite1/scenario_08_3hop.py`, rhypedb-tcp vs
Postgres) at **1M ratings** (100k users × 10k movies × 1k directors × 10
ratings/user), 3-hop traversal, mean per-op:

| 1M 3-hop traversal | mean op | vs PG-optimal (69 µs) | vs PG-idiomatic (139 µs) |
| --- | --- | --- | --- |
| rhypedb **None** (v5) | **42.9 µs** | **1.6× faster** | **3.2× faster** |
| rhypedb **Lz4** (v6) | **157.7 µs** | 2.3× slower | 1.13× slower |

Same data, same queries, only the compression flag differs: **LZ4 is 3.67×
slower than None** here, and it flips a decisive win over even hand-tuned
Postgres into a loss. Why so much worse than the micro-bench's "+4%"? A graph
traversal does **many scattered, point-like cover-blob reads** (each `get_links`
hop reads rev-edge / forward-edge entries spread across the keyspace), and each
one decompresses a whole 16-entry block to extract a *single* entry — the
opposite of a sequential scan, where the block decompress amortizes over 16
entries. The micro-bench graph was small and cache-resident, hiding this.

**Not everything regresses:** the 100k `@indexed` filter scan with LZ4 still beat
Postgres (82.8 µs vs 94.6 µs) — a range scan reads many entries per decompressed
block. The damage is concentrated in scatter-read workloads (traversal, point
lookups).

**Decision (2026-06-21): default flipped to `None`.** A 3.7× regression on a core
read workload, by default, is the wrong trade — especially against the project's
"don't trade function for memory" rule. LZ4 is now **opt-in** for deployments
where disk size / cold-cache density dominates and reads are scan-heavy or rare. A
future thread-safe decompressed-block cache could make LZ4 viable as a default by
amortizing repeated hot-block reads within a traversal (deferred).

**Background compaction still earns its keep regardless** (and absorbs LZ4's
~1.5× compaction CPU when it is enabled): worst-case commit 66.7→18.5 ms,
total ingest halved at 1M. See the async-compaction section above.
