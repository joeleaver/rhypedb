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
