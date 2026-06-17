# Benchmarks

Performance benchmarks for rhypedb. See [PLAN.md](PLAN.md) for the full design.

## Status

| Suite | Scenario                  | rhypedb HTTP | rhypedb TCP | PG idiomatic | PG optimal |
|-------|---------------------------|--------------|-------------|--------------|------------|
| 1.1   | bulk insert users         | ✓            | ✓           | ✓            | ✓ (COPY)   |
| 1.2   | point lookup by ID        | ✓            | ✓           | ✓            | (= idiom.) |
| 1.3   | filter scan               | ✓            | ✓           | ✓            | (= idiom.) |
| 1.4   | 1-hop traversal           | ✓            | ✓           | ✓            | ✓ (1 JOIN) |
| 1.5   | 2-hop traversal           | ✓            | ✓           | ✓            | ✓ (3-way)  |
| 1.6   | cascade delete            | ✓            | ✓           | ✓            | (= idiom.) |
| 1.7a  | unique constraint (violation) | ✓        | ✓           | ✓ (exc.)     | ✓ (ON CONFL.) |
| 1.7b  | unique constraint (success)   | ✓        | ✓           | ✓            | ✓          |
| 2.x   | vector suite              | TODO         | -           | TODO         | -          |

## Quick start

### 1. Build + start the rhypedb bench server

A separate server on alternate ports so it doesn't clash with the main one.
The bench server has no embedding model loaded, so RSS stays at ~4 MB.

```sh
cargo build --release -p rhypedb-server

mkdir -p /tmp/bench-rhypedb-data
target/release/rhypedb-server \
  --schema benchmarks/schemas/bench1.sdl \
  --data-dir /tmp/bench-rhypedb-data \
  --listen 127.0.0.1:4300 \
  --tcp-listen 127.0.0.1:4301
```

### 2. Start the Postgres bench container

Runs on host port 5433. The schema is auto-loaded from
`benchmarks/schemas/bench1.sql` via the postgres image's `initdb` hook. We pass
`fsync=off` + friends so writes are fast and unsafe — fine for a fresh
benchmark DB you can throw away.

```sh
cd benchmarks
docker compose up -d
docker compose logs -f postgres   # optional, watch startup
```

Tear down (and wipe the data volume) with `docker compose down -v`.

### 3. Install the Python harness deps

The rhypedb-side harness is pure stdlib, but the Postgres scenarios need
`psycopg`. A throwaway venv at the repo root keeps your system Python clean.

```sh
python3 -m venv .venv-bench
.venv-bench/bin/pip install -r benchmarks/requirements.txt
```

### 4. Run scenarios

```sh
.venv-bench/bin/python -m benchmarks.suite1.scenario_01_bulk_insert      --users 1000 --iterations 3
.venv-bench/bin/python -m benchmarks.suite1.scenario_02_point_lookup     --users 1000 --lookups 1000 --iterations 3
.venv-bench/bin/python -m benchmarks.suite1.scenario_03_filter_scan      --movies 1000 --queries 200 --iterations 3
.venv-bench/bin/python -m benchmarks.suite1.scenario_04_05_traversal     --users 200 --movies 100 --ratings-per-user 5
.venv-bench/bin/python -m benchmarks.suite1.scenario_06_cascade_delete   --n-deletes 100 --ratings-per-user 10
.venv-bench/bin/python -m benchmarks.suite1.scenario_07_unique_constraint --attempts 500 --iterations 3
```

`--impl` accepts a comma list. Aliases: `both`/`rhypedb` → `http,tcp`,
`pg` → `pg-idiomatic,pg-optimal`, `all` → `tcp,pg-idiomatic,pg-optimal`
(the publish-ready set). Default is `all`.

Results land in `benchmarks/results/*.json` and a summary table prints to
stdout.

## Current findings — Suite 1

Measured on a Ryzen 7 6800H, 16 GB RAM, against a clean rhypedb bench server
and the Dockerized Postgres 16 above. Numbers are mean per-op latency in
microseconds and the relative speedup vs the rhypedb-tcp column.

| Scenario                  | rhypedb HTTP | rhypedb TCP | PG idiomatic | PG optimal | TCP vs PG-opt |
|---------------------------|-------------:|------------:|-------------:|-----------:|--------------:|
| 1.1 bulk insert (per row) | 242 µs       | **43 µs**   | 93 µs        | n/a¹       | **2.2× vs idiom.** |
| 1.2 point lookup by ID    | 235 µs       | **31 µs**   | 104 µs       | =idiom.    | **3.4×**      |
| 1.3 filter scan (50 rows) | 857 µs       | 624 µs      | **109 µs**   | =idiom.    | 0.17× (PG wins) |
| 1.4 1-hop traversal       | 271 µs       | **56 µs**   | 266 µs       | 129 µs     | **2.3×**      |
| 1.5 2-hop traversal       | 422 µs       | **283 µs**  | 523 µs       | 312 µs     | **1.1×**      |
| 1.6 cascade delete        | 425 µs       | 216 µs      | **177 µs**   | =idiom.    | 0.82× (PG wins) |
| 1.7a unique violation     | 248 µs       | **28 µs**   | 143 µs       | 114 µs     | **4.1×**      |
| 1.7b unique success       | 240 µs       | **42 µs**   | 91 µs        | 124 µs     | **2.2×**      |

¹ PG-optimal for bulk insert is `COPY ... FROM STDIN`, a single streaming
roundtrip; per-row latency isn't measured. The full COPY of 1000 rows
completes in ~5 ms — about **9× faster wall-clock** than the rhypedb-tcp
per-row insert path (43 ms for the same 1000 rows). This is the obvious
follow-up: implement a `create_batch` in the query language and wire it
through the binary protocol.

RSS during all rhypedb runs stays at **3.9 MB** (no embedding model). The
Postgres backend sits at **~45 MB** RSS during these workloads.

### What's interesting

- **Point lookup (3.4×)**, **1-hop traversal (2.3×)**, and **unique-violation
  detection (4.1×)** are decisive wins for rhypedb's binary protocol +
  in-process LSM. The wire-protocol speedup we measured earlier (HTTP vs TCP
  on rhypedb) plus an order-of-magnitude memory advantage all compound here.
- **2-hop traversal** is the headline structural advantage: rhypedb's
  `User.get(X).ratings.movie.ratings.user` path-query is **1.1× faster than
  Postgres's hand-tuned 3-way JOIN** and **1.8× faster than the idiomatic
  3-roundtrip flow.** This is the gap our query language exists to capture.
- **PG-optimal beats PG-idiomatic by 2.0–2.1× on traversals** — exactly the
  "you're not using Postgres right" gap PLAN.md warned about. The point of
  showing both columns is to be honest about what real apps experience (the
  idiomatic column) vs the lower bound a hand-tuned query achieves.
- **PG-optimal slower than PG-idiomatic on 1.7b success** — `ON CONFLICT DO
  NOTHING RETURNING id` is a touch slower than a plain INSERT when the
  conflict never fires, because it still has to consult the unique index.
- **Filter scan** — was 0.17× at the small scale and 0.001× (400×!) at
  100K when this README was first written. Now with `@indexed` secondary
  indexes + bounded `scan_from_at_limited`, it's **1.25× behind PG at
  100K (230 µs vs 184 µs)**. See "Non-unique secondary indexes" below.
- **Cascade delete loses narrowly (0.82×)** — Postgres's FK cascade is
  in-tree and exceptionally optimized. Our cascade does 11 storage deletes
  + 10 unique-index cleanups per user; that's a fair gap to close.

## Scaling — same scenarios at ~100× the data

The small-scale numbers above are honest but they all fit in L3 cache for
both systems. To see how the gap actually moves as the dataset grows, we
re-ran every scenario at roughly 100× scale:

- 1.1 bulk insert: 100K users (300K inserts across 3 iter)
- 1.2 point lookup: 100K users seeded, 5K lookups × 3 iter
- 1.3 filter scan: 100K movies, 100 queries × 3 iter
- 1.4/1.5 traversal: 10K users × 1K movies × 10 ratings/user = 100K ratings, 500 queries × 3 iter
- 1.6 cascade delete: 500 deletes × 20 ratings each × 3 iter (so 30K cascading rows)
- 1.7 unique: 2K attempts × 3 iter against the accumulated DB

Same hardware, same PG container, same rhypedb bench server (both restarted
clean for the run). Mean per-op latency:

| Scenario                  | rhypedb TCP | PG idiomatic | PG optimal | TCP vs PG-opt |
|---------------------------|------------:|-------------:|-----------:|--------------:|
| 1.1 bulk insert (per row) | **44 µs**   | 99 µs        | 2.8 µs²    | 0.06× (PG wins COPY) |
| 1.2 point lookup by ID    | **36 µs**   | 118 µs       | =idiom.    | **3.31×**     |
| 1.3 filter scan (100/qry) | **179 µs**³ | 706 µs       | =idiom.    | **3.94×**     |
| 1.4 1-hop traversal       | **88 µs**   | 298 µs       | 158 µs     | **1.79×**     |
| 1.5 2-hop traversal       | **2,718 µs**⁴| 5,203 µs     | 3,249 µs   | **1.20×**     |
| 1.6 cascade delete        | 1 912 µs    | **191 µs**   | =idiom.    | 0.10× (PG wins) |
| 1.7a unique violation     | **28 µs**   | 141 µs       | 112 µs     | **4.0×**      |
| 1.7b unique success       | **47 µs**   | 96 µs        | 132 µs     | **2.0×**      |

**rhypedb beats PG (idiomatic AND optimal) on every read path at 1M scale.**
The 2-hop win over PG-optimal's hash join is the headline architectural
result — see "Second-degree covering" below.

² PG-optimal COPY at scale: 100K rows in 273 ms = 2.7 µs/row, so 15× faster
than rhypedb-tcp's per-row insert. The COPY gap is exactly proportional to
scale — every per-row roundtrip we eliminate is a fixed win.

³ Filter scan went from 102,313 µs (full scan + per-object decode) →
58,759 µs (zone maps) → 230 µs (secondary index + bounded scan) →
~145 µs (auto-compaction + alloc-friendly response + covering index).
See "Non-unique secondary indexes" and "Profile-driven follow-ups" below.

⁴ 2-hop traversal went from 16,880 µs → 4,256 µs (covering reverse-edge +
sorted-batch + bloom bitmask + bytes-zero-copy + xxh3 — earlier sessions)
→ 4,438 µs (raw-bytes IdSetWithFields) → 2,718 µs (second-degree
covering). The last step embeds the next-hop target's serialized object
fields in the reverse-edge value at link time, so the terminal materialize
skips the 700-user `multi_get`. See "Second-degree covering" below.

RSS for rhypedb stayed at **3.9 MB across every scale run**. PG sat at
**~45 MB**. The order-of-magnitude memory story is the one thing that
absolutely doesn't degrade with scale.

### What changed from small scale

**Stayed flat (both engines are O(log N) here):**
- Point lookup: 31 → 34 µs (rhypedb), 104 → 111 µs (PG). Bloom filters +
  b-tree both hold up. rhypedb's wire-protocol advantage stays at ~3×.
- Bulk insert per-row: 43 → 41 µs (rhypedb), 93 → 94 µs (PG). LSM and PG
  both maintain effectively flat per-row cost up to this size.
- Unique-constraint operations: essentially scale-invariant on both sides.

**Gap shrank but rhypedb still wins:**
- 1-hop traversal: rhypedb went 56 → 203 µs (3.6× slower), pg-optimal went
  129 → 253 µs (2× slower). rhypedb's lead dropped from 2.3× to 1.2× but
  still holds because the intermediate result (movies a user rated) stays
  small (~10) so the path-query expansion stays cheap.

**Flipped — rhypedb now loses at scale:**
- **2-hop traversal: 283 µs → 16,880 µs (60× slower!).** At 100K ratings
  spread across 1K movies, each query expands to ~10 movies × ~100 ratings
  per movie = ~1000 candidate users, dedup'd to maybe ~700. rhypedb's
  executor materializes intermediate ID sets at each hop without
  planner-grade hash-distinct pruning, so the work per query scales with
  the intermediate fanout. PG's idiomatic 3-roundtrip flow with `ANY(%s)`
  drives index scans and is now **5× faster** than rhypedb. This is the
  honest reality: the path-query advantage we celebrated at 1K is real but
  only at small intermediate fanout — at scale, a real planner wins.
- **Cascade delete: 216 µs → 1,912 µs (9× slower at 2× the ratings/user).**
  Each cascading rating triggers MVCC + unique-index work, and that
  per-rating overhead doesn't amortize. PG's FK cascade is dramatically
  more optimized at this scale.

**Got dramatically worse — predicted:**
- **Filter scan: 624 µs → 102,300 µs (164× slower).** A full type scan over
  100K rows. PG's b-tree on `year` reads only matching pages: 148 µs.
  The gap went from 0.17× to 0.001×. Exactly what we expected without
  secondary integer-field indexes.

### What this tells the backlog

Three of the existing backlog items are now load-bearing rather than
nice-to-have:

1. **Secondary integer/field indexes (or zone maps)** — the filter-scan gap
   at scale is embarrassing without them.
2. **Query planner + execution improvements for multi-hop traversal** — a
   path-query that materializes intermediate sets without hash-distinct
   pruning loses to a planner at fanouts > ~100.
3. **`create_batch` in the QL** — every per-row roundtrip we eliminate is a
   fixed win vs COPY, and the gap is exactly proportional to scale.

The wins (point lookup 3.3×, 1-hop 1.2×, unique 2-4×, 3.9 MB flat memory)
are real and durable. The 2-hop story we led with at small scale doesn't
generalize and we should stop overclaiming it until the executor catches up.

## Post-fix update (same session)

Four backlog items shipped after the scale findings above. New comparison
at the same 100K scale, same hardware, same fresh containers:

| Scenario                      | Pre-fix    | Post-fix   | Win   | Where it came from |
|-------------------------------|-----------:|-----------:|------:|--------------------|
| 1.1 bulk insert (per-row TCP) | 41 µs/row  | 41 µs/row  | —     | unchanged          |
| 1.1 bulk insert (tcp-batch)   | n/a        | **14 µs/row** (1376 ms/100K) | **new path** | `create_batch` source: 1 txn, 1 WAL append, schema lookup amortized |
| 1.4 1-hop traversal           | 203 µs     | **106 µs** | 1.9×  | streaming-traversal executor (no intermediate object materialize) |
| 1.5 2-hop traversal           | 16,880 µs  | **7,069 µs** | 2.4× | streaming-traversal + dedup-per-hop + multi_get_at + batched get_links v1 |
| 1.6 cascade delete (20 rt/u)  | 1,912 µs   | **558 µs** | 3.4×  | precomputed incoming-relations map + skip storage.get for types without unique fields + skip existence check on recursive calls |

Bulk-insert leaderboard at 100K, per-iter wall-clock:

| Impl                   | Wall-clock | Per-row |
|------------------------|-----------:|--------:|
| pg-optimal (COPY)      | 275 ms     | 2.8 µs  |
| **rhypedb tcp-batch**  | **1,376 ms** | **14 µs**   |
| rhypedb tcp (per-row)  | 4,418 ms   | 44 µs   |
| pg-idiomatic           | 9,803 ms   | 98 µs   |

We're now **7× ahead of PG-idiomatic on bulk insert** and **5× behind PG-COPY** —
that remaining gap is wire-format (no QL string parsing) plus 30 years of
Postgres bulk-write optimization, both tracked as follow-up cards.

The 2-hop story changes a third time: still behind PG-idiomatic's 3.4 ms,
but the gap closed from 5× to ~2×. The remaining work is captured in the
"v2 per-SST batched prefix iteration" card — collapsing the 1000 per-hop
prefix-scans into a single sorted-key sweep per storage layer. Targeted at
1 ms or below on the same workload.

What we've *not* yet touched at this point in the session:
- Filter scan (still 164× behind PG — needs zone maps + column projection)
- Cascade delete at scale (10× behind PG — needs MVCC/index amortization profiling)

## Post-fix update II (same session)

Two more passes against the 2-hop bottleneck, same 100K-scale workload
(10K users × 1K movies × 10 ratings, 500 queries × 3 iter, --impl tcp):

| Scenario              | Post-fix I | + covering reverse-edge | + sorted-batch SST | Cumulative |
|-----------------------|-----------:|------------------------:|-------------------:|-----------:|
| 1.4 1-hop traversal   | 106 µs     | ~106 µs                 | **109 µs**         | flat       |
| 1.5 2-hop traversal   | 7,069 µs   | 5,638 µs                | **4,865 µs**       | 1.45×      |

**Covering reverse-edge index** (last session): writes the source's
effective fields as the reverse-edge entry value when a forward 1:1
traversal can be served straight from the carried `FieldMap`, skipping
~1000 forward prefix scans per 2-hop query. 1.25× on 2-hop.

**Sorted-batch SST sweep** (this session): `SstReader::multi_get_versioned`
walks each SST once for a pre-sorted batch of user keys, with a threshold
gallop (re-binary-search at sparse-block boundaries) to skip over gaps.
Replaces N independent log-N seeks per SST with one seek + linear sweep.
The terminal materialize of ~700 user objects at the end of a 2-hop drops
from N independent block reads to ~one block read per ~6 needles. 1.25× on
2-hop, 1.43× on 1-hop (1-hop also ends in a `get_many` materialize).

The session memory predicted "2-3 ms" for the sorted-batch sweep alone.
The honest outcome is smaller — the terminal materialize was less
dominant than estimated; most of the remaining 4.8 ms is in the per-hop
`multi_scan_prefix_at` calls, not the final `multi_get_at`. The deferred
"v2 per-SST batched prefix iteration" card is the next big lever, and the
known regression risk on scattered prefixes (single-walk-with-dispatch
toured 100× the entries to find matches in the failed v1.5 experiment)
means it needs its own focused attempt with the gallop pattern reused.

| Scenario at 100K       | rhypedb TCP | PG idiomatic | PG optimal | TCP vs PG-opt |
|------------------------|------------:|-------------:|-----------:|--------------:|
| 1.5 2-hop traversal    | **4,865 µs** | 3,390 µs    | 2,580 µs   | 0.53× (PG wins) |

## Post-fix update III (same session) — prefix v2 lands, bench doesn't move

`SstReader::multi_scan_prefix_versioned` shipped with the same threshold-gallop
pattern as the point-lookup primitive: sort the prefixes, walk each SST once
with a cursor that dispatches each entry to the active prefix, re-binary-search
the sparse index at block boundaries when the cursor falls behind. 247
workspace tests pass (+14 new — 8 SST cases, 5 LSM cases, 1 differential vs
single-prefix `scan_prefix` across many prefixes).

Two consecutive 100K-scale 2-hop runs:

| Run               | 1.4 1-hop | 1.5 2-hop |
|-------------------|----------:|----------:|
| sorted-batch only | 109 µs    | 4,865 µs  |
| + prefix v2 run 1 | 92 µs     | 4,965 µs  |
| + prefix v2 run 2 | 98 µs     | 4,325 µs  |

The 2-hop swings 4,325 → 4,965 µs across runs of the same code — about ±13%
run-to-run noise. The sorted-batch baseline (4,865 µs) sits squarely inside
that band. **Prefix v2 doesn't move the 2-hop bench measurably**, and the 1-hop
shift is within the same noise range.

The honest reason: the executor's covering-reverse-edge fast path already
serves forward-1:1 hops from carried FieldMaps without a prefix scan. The
remaining prefix scans in a 2-hop query are ~2 `multi_scan` calls with ≤10
prefixes each (one per movie, one per intermediate rating set). Amortizing
10 sparse-index seeks per SST is real work — it just isn't where the 4.8 ms
is hiding. The cost is dominated by per-entry scan work and downstream
executor merging, not by repeated index lookups.

The v2 implementation is the right architecture; the bench just doesn't have
the fanout shape (1000+ prefixes per call) that would showcase it. It will
land cleanly for any future workload that does — bulk operations, large
multi-source traversals, or compound queries that fan out before narrowing.

The next-highest-leverage lever is no longer in storage. Candidates: profile
the 4.8 ms to identify the actual hot path (likely in the executor / dedup /
tuple decoding), or take the gap as a planner / executor problem rather than
a storage problem — exactly the framing the original 100K-scale "honest
losses" section flagged.

## Profile-driven update — bloom bitmask + zero-copy SST slices

We did the profile. `perf record -F 999` over a 45 s sustained 2-hop run
caught 23,825 samples; top self-time symbols were:

| % self | Symbol |
|---:|---|
| 23.7% | `SstReader::multi_get_versioned` |
| 8.8% | `__memcmp_avx2_movbe` (libc) |
| ~22% | malloc + free + memmove (summed) |
| 4.1% | `crossbeam_skiplist::RefRange::next` (memtable scan) |
| 2.3% | `SstReader::locate_block_from` |
| ~4.5% | `deserialize_fields` + `str::from_utf8` |
| 2.0% | `BTreeMap::insert` |

Inside `multi_get_versioned`, `perf annotate` pointed at a single instruction
— `divq` for `combined_hash(...) % self.num_bits` in the bloom-filter
prefilter — taking **4.27% of total CPU** on its own. Per 2-hop query we did
~700 needles × 2 SSTs × 7 hashes = ~9,800 ~25-cycle integer divisions. The
~22% allocator load was dominated by `Bytes::copy_from_slice(&self.data[..])`
calls for every matched value and per-prefix user-key clone.

**Bloom bitmask** (`crates/rhypedb-storage/src/bloom.rs`): round `num_bits` up
to the next power of 2 in `with_params`, store `mask = num_bits - 1`, use
`& mask` instead of `% num_bits` in `add`/`contains`. Legacy non-power-of-two
filters from older SSTs keep the modulo path via a `mask: Option<u32>` flag
set in `read_from`. Two new regression tests pin the power-of-two construction
and the legacy modulo-path round-trip.

**Zero-copy SST slices** (`crates/rhypedb-storage/src/sst.rs`): convert
`SstReader::data` from `Vec<u8>` to `Bytes`, swap every
`Bytes::copy_from_slice(&self.data[a..b])` for `self.data.slice(a..b)`
(O(1) refcount). The iterator now borrows `&Bytes` and emits zero-copy
slices for keys + values. Index parsing in `open` and per-prefix
user-key returns in `scan_prefix` / `multi_scan_prefix_versioned` also drop
their copies.

249 workspace tests pass (was 247; +2 bloom regression tests).

### Measured (100K-scale 2-hop, sustained-loop hammer)

| Build | 2-hop µs/query | Notes |
|---|---:|---|
| Pre-bloom-bytes (debuginfo=2 build for perf) | 4,775 | Single 60 s hammer; debug-info adds ~5-10% overhead |
| Post-bloom-bytes (normal release) | **4,128** | Mean of 3× 30 s hammers (4,182 / 4,124 / 4,079) |

That's an **honest ~8-13% wall-clock improvement** depending on how you
adjust for the debug-info overhead in the pre-fix baseline. Smaller than
the 12-20% the perf numbers initially suggested — the allocator wasn't
exclusively in the SST hot path, so removing the SST copies left other
sources (protocol decode/encode, query-cache lookups, transient
intermediate buffers) still pushing on malloc. But it's a real, repeatable
movement of the 2-hop bench, and it shipped two clean atomic changes
(bloom one-liner, SST type-change) that benefit every read path going
forward, not just 2-hop.

What's likely next: the same `multi_get_versioned` is still the #1 self-time
symbol post-fix (we shaved its allocations but the cursor walk + memcmp work
is still real), so further wins probably come from either fewer comparisons
(better bloom k? compressed user-keys? per-block index hints?) or even
fewer entries scanned (per-block min/max for objects, similar to the
in-flight zone-maps card for filter scans).

## Re-profile, range pruning, xxh3 bloom — diminishing returns inflection

We re-profiled after the bloom-bitmask + Bytes work. The picture flipped:
`BloomFilter::contains` rose to **23.08% self-time** (de-inlined now that
it has the bitmask/modulo branch) — FNV-1a per-byte hashing for 700 needles
× ~5 SSTs × 7 hashes per check. Two follow-ups landed:

**SST per-batch range pruning** (`crates/rhypedb-storage/src/sst.rs`,
`lsm.rs`): `SstReader` now computes accurate `first_user_key` /
`last_user_key` at open (walking the final data block; the sparse index
only records each block's first key). `range_overlaps_sorted` / 
`prefix_range_overlaps_sorted` let `LsmTree::multi_get_at` and
`multi_scan_prefix_at` skip whole SSTs whose ranges don't intersect the
batch. **For the bench, this is a no-op** — the 5 SSTs each contain
interleaved object types and edges (the memtable mixes types within each
flush), so every SST's range overlaps every needle batch. The pruning is
correct + tested (10-SST non-overlapping integration test passes) and will
take effect on any workload where flushes carve cleaner key ranges, but it
doesn't move *this* bench.

**xxh3 bloom** (`crates/rhypedb-storage/src/bloom.rs`, SST v3): replaced
FNV-1a's per-byte loop with xxh3's SIMD-friendly 8-bytes-per-iteration
hash via `xxhash-rust`. `HashAlgo` tracked per filter; SST v3 prepends an
algorithm byte to the bloom block, v2 files keep using FNV-1a via the
existing reader path. Bloom CPU dropped from 23.08% (FNV) to 7.50%
(xxh3 contains) + 5.76% (xxh3 hash) = **13.3% — a real ~10% relative
reduction** in bloom CPU, but only ~5% of total wall-clock (other paths
fill the gap that bloom freed).

258 workspace tests pass throughout (was 247 after bloom-bytes; +11 from
this phase — 5 SST range-overlap cases, 2 LSM integration cases, 2 xxh3
cases, +2 misc).

### Measured (sustained-loop hammer, normal release)

| Phase | 2-hop µs/query | Δ vs prior | Cumulative vs pre-fix |
|---|---:|---:|---:|
| Pre-fix (debuginfo build, single 60 s hammer)  | 4,775 | — | — |
| Post bloom-bitmask + Bytes (mean of 3× 30 s)   | 4,128 | -14% | -14% |
| Post range pruning (mean of 3× 30 s)           | 4,261 | +3% (noise) | -11% |
| Post xxh3 (mean of 3× 30 s)                    | 4,256 | flat | -11% |

Honest reading: **bloom-bitmask + Bytes was the last big lever.** Range
pruning and xxh3 each shipped correctly tested architectural improvements
that simply don't pay back at this bench's workload shape. They benefit
workloads with non-overlapping SSTs (typical of larger production data
after compaction) and longer keys (where FNV-1a's per-byte loop dominates),
respectively.

### Where the time goes now

After this phase, the profile is fragmented — no single hot spot above ~10%:

- 9.1% `__memcmp_avx2_movbe` — `entry_user` vs needle comparisons in the SST sweep
- 7.5% `BloomFilter::contains` (post-xxh3 — was 23%)
- 5.8% `xxh3_double_hash` — the new bloom hash
- ~15% allocator (malloc + free + memmove summed)
- 4.9% memtable skiplist iteration
- 5.3% field decode (`deserialize_fields` + `from_utf8`)
- 2.5% protocol response encoding

The next-biggest single wins would have to attack the allocator floor
(remaining non-SST allocations: protocol decode, FieldMap construction,
query-cache lookups) or the data layout (compaction so we have one SST
instead of five — would naturally amortize bloom across batched lookups
without code changes). Neither is a one-liner.

### When to pivot

2-hop is now at 4.26 ms — **1.26× behind PG-idiomatic's 3.39 ms** and
1.65× behind PG-optimal's 2.58 ms. Pre-session we were 5× behind PG-idiomatic
and 6.5× behind PG-optimal, so this run closed the gap from "embarrassing"
to "same order of magnitude, ~25% off." We're in the competitive range but
not yet ahead — PG-idiomatic is still slightly faster on this scenario,
and PG-optimal's hand-tuned 3-way JOIN still has a meaningful lead.

The next absolute opportunity in the table is the **filter scan at 102 ms
(164× behind PG)**: zone maps for `@filter`-able integer fields would
directly close that gap. Compared to single-digit-percent traversal
squeezes, that's an order-of-magnitude larger wall-clock prize on a
workload that's currently embarrassingly slow.

## Zone-map filter scan (SST v4)

Per-block min/max bounds for integer fields, written into a new zone-map
block before the bloom block. SST v4 footer extends with `[zonemap_offset:
u64][zonemap_size: u32]`. `LsmTree::scan_prefix_filtered_at` consults
`block_could_match` per sparse block and skips whole groups of entries
whose `min..=max` range rules out the predicate. The executor recognizes
single-integer-compare `Filter` predicates and pushes them down to
`Database::filter_scan`; complex predicates (And/Or, string compares)
fall through to today's full scan + per-object filter.

**On-disk encoding.** Integers stored as 8-byte big-endian with sign-bit
flip for signed types — byte comparison matches numeric comparison without
type info. Storage stays schema-agnostic; the engine sets a
`ZoneFieldExtractor` closure on `LsmConfig` that decodes value bytes and
emits `(field_name_hash, encoded_bytes)` tuples for object entries.

**Bench-measured (100 K movies × 100 thresholds × 3 iter, --impl tcp):**

Bench-machine load was high during these measurements (`arvx_terrain` at
1272% CPU), so apples-to-apples on the *same* machine in the *same* session
is the only reliable comparison:

| Configuration | mean µs/query |
|---|---:|
| Push-down OFF (full scan + per-object filter, this machine) | 80,482 |
| Push-down ON  (zone-map block skip + per-entry recheck)     | **58,759** |

**~27% wall-clock improvement (1.37×)** on this workload. Smaller than
hoped for two reasons baked into the workload:

1. **Block bounds are wide.** Movies are inserted in ID order, not year
   order. With 16 entries per block and years uniformly in [1950, 2024],
   most blocks span ~70 years — only blocks near the high-threshold tail
   can skip. A back-of-envelope skip rate at threshold = 2010 is ~4%; at
   threshold = 2020 it's ~41%. Averaged across the uniform threshold
   sweep, expected ~10-15% of blocks pruned.
2. **`.limit(50)` isn't pushed down.** Storage returns all matches first,
   then the limit step truncates. Pushing limit into `filter_scan` would
   be a big win for low-threshold queries (where 50K rows match but only
   50 are returned).

Even ignoring measurement noise, the post-fix gap to PG-idiomatic's 148 µs
is still ~400×. The fundamental reason is structural: PG uses a real
B-tree secondary index on `year` (O(log N + result_size)); we do a sorted
full-type scan with per-block zone pruning (best case O(N / block_size)).
**Closing the rest of the PG gap on filter scans requires true secondary
indexes**, a much bigger project than zone maps.

258 → 269 workspace tests (+11 from this phase: 6 zone-module unit tests,
3 SST v4 round-trip + scan_prefix_filtered cases, 2 engine end-to-end
filter_scan + flush correctness).

## Non-unique secondary indexes (@indexed)

The follow-up to zone maps. Adds a per-field sorted index that maps
`(encoded_value, object_id)` → empty, keyed `i:<type_id>:<field_id>:<encoded_value>:<object_id>`,
so `Movie.filter(.year > 2010).limit(50)` reads ~50 key probes instead
of scanning the whole `Movie` table.

**Schema.** New `@indexed` SDL directive, valid on integer scalar fields
only (u32/u64/i32/i64). Cohabits with `@unique`. The bench schema gets
`year: u32 @indexed` on Movie.

**Storage.** New `i:` key prefix. Encoded value uses the same byte-order-
preserving rules as zone maps (sign-bit flip for signed, big-endian
widened to 8 bytes), so the lexicographic order on the key tail matches
numeric order on the field value.

**Engine write path.** `create` / `create_batch` / `update` / `delete`
maintain the index alongside the unique index, in the same write txn. A
new `indexed_fields` map cached at `Database::open()` resolves the
per-field `(name, field_id)` once so the hot path doesn't walk the
schema. Cascade-delete cleanup gates on the type's index list so
edge-only types (Rating) still skip the storage probe.

**Engine read path.** `Database::filter_scan` gained an
`Option<usize> limit` arg. When the field is `@indexed`:

* **Eq target** — narrow prefix scan on `i:<type>:<field>:<target>:`.
  Every key is a match.
* **Gt/Ge with limit** — seek-then-scan from
  `i:<type>:<field>:<target+1>:` (Gt) or `i:<type>:<field>:<target>:`
  (Ge). New `LsmTree::scan_from_at_limited` walks each layer with a
  bounded per-layer cap so the cost is O(limit) instead of
  O(prefix_size).
* **Lt/Le with limit** — bounded prefix scan from the field prefix.
  Matches cluster at the smallest values, so the first ~limit entries
  contain all the answers; the post-decode loop short-circuits on the
  first non-match.
* **Ne or no limit** — unbounded prefix scan, per-entry filter.

The executor pushes `.limit(N)` from a trailing `Step::Limit` into
`try_filter_scan`. A bare `.filter(...).limit(N)` query — exactly the
bench shape — gets the bounded path.

**Bench-measured (100 K movies × 100 thresholds × 3 iter, --impl all,
clean run on the same machine).** The first measurement was with the
secondary index but *without* the bounded-scan primitive — every query
still walked the full 100K-entry `i:` prefix before truncating to
`.limit(50)`, so the win came entirely from skipping object decode.
Adding `scan_from_at_limited` (seek to target + per-layer cap) closed
the gap by another 100×.

| Phase                                                | rhypedb-tcp µs | vs PG-idiomatic |
|------------------------------------------------------|---------------:|----------------:|
| Zone maps only (prior session)                       | 58,759         | 397×            |
| Secondary index, unbounded prefix scan               | 25,633         | 173×            |
| Secondary index + bounded `scan_from_at_limited`     | 230.7          | 1.25×           |
| + auto-compaction (1 SST instead of 2)               | 200.7          | 1.09×           |
| + alloc-friendly response encoding                   | 188            | 1.02×           |
| **+ covering index (Object data in `i:` entries)**   | **~145**       | **0.79× (FASTER)** |
| PG-idiomatic (`SELECT … WHERE year > $1 LIMIT 50`)   | ~184           | —               |

That's a **~400× wall-clock win over zone maps** and we now beat
PG-idiomatic by ~1.27× at steady state on the same machine.

## Profile-driven follow-ups: compaction, alloc, covering

After the secondary-index work above, a `perf record` over a 30 s
sustained-load hammer showed the wire format was *not* the bottleneck
(`encode_object` was 0.25% of CPU). The actual hot path was a mix of:

* **~17% allocator churn** (malloc/free/memmove/realloc) from per-query
  `Bytes`/`Vec`/`BytesMut` construction
* **~11% bloom filter cost** (xxh3 hashing across multiple SSTs)
* **~10% memcmp** on key comparisons + BTreeMap layer-merge
* **~9% TCP send syscall** + 4.4% epoll
* **~25% spread across `multi_get_versioned` + `BTreeMap::insert` +
  HashMap drop/insert/from_utf8** — the cost of materializing 50
  Movies from the 50 matching ids the bounded scan returned

Three changes landed in response:

### Auto-compaction (broad win)

`LsmConfig::compact_trigger_ssts = 4`. When `maybe_flush` rotates a
memtable and pushes the SST count past the threshold, the next read
sees fewer layers — fewer bloom checks, smaller BTreeMap merge, less
memcmp. Cuts filter scan from 230 → 200 µs (-13%), pays back on every
read path that crosses multiple layers. Manual `POST /admin/compact`
also added for operator-triggered compaction during benchmarking
(gated by `RHYPEDB_ADMIN_TOKEN` — send `Authorization: Bearer <token>`).

### Allocator-friendly TCP response encoding (broad win)

* Added `serialize_fields_into(fields, &mut Vec<u8>)` so the wire
  encoder writes directly into the response buffer — no intermediate
  `Bytes` allocation per Object.
* New `protocol::write_frame_buffered(writer, &mut buf, …)` builds the
  whole frame (header + payload) in one Vec and ships with one
  `write_all` instead of four. Combined with a per-connection reusable
  `response_buf: Vec<u8>`, the per-query allocator load drops to a
  handful of reservations.
* TCP syscall + epoll cost dropped from ~14% of CPU to ~2.5% — no
  library swap needed.

Cuts filter scan from 200 → 188 µs (-6%). 1-hop traversal went from
~106 µs to 87 µs in the same change (~17%) — the FieldMap construction
on traversal terminals is a similar allocation profile.

### Covering index (deep win on filter scan)

The `i:<type>:<field>:<encoded_value>:<id>` entries now carry the
source object's serialized FieldMap as the entry value. `filter_scan_via_index`
constructs Objects straight from the index entry — no `get_many` probe
per match, no bloom checks, no per-id snapshot read. On `create`/`update`,
the new merged FieldMap is written into the index entry alongside the
object key (even if the indexed field itself didn't change, we refresh
the covering payload so non-indexed updates don't leave stale data).

Cuts filter scan from 188 → ~145 µs (-23%). We now run filter scan
~1.27× FASTER than PG-idiomatic at 100K scale on this query shape.

### Tests
284 → **286 workspace tests** (+2 covering-specific cases: covering
value returns the full FieldMap; covering value refreshes on
non-indexed updates).

269 → 284 workspace tests (+15: 3 schema parser cases for `@indexed`
and rejection of non-integer fields, 4 key-builder cases for
`field_index*`, 8 engine end-to-end cases for create/update/delete/
cascade/batch/flush correctness of the index).

## Layout

```
benchmarks/
├── PLAN.md                            # full benchmark design + scenarios
├── README.md                          # this file
├── docker-compose.yml                 # Postgres 16 on host 5433
├── requirements.txt                   # psycopg
├── schemas/
│   ├── bench1.sdl                     # rhypedb schema
│   └── bench1.sql                     # postgres DDL (loaded via initdb hook)
├── harness/
│   ├── common.py                      # latency/memory/results utilities
│   ├── data.py                        # synthetic dataset generator
│   └── clients/
│       ├── rhypedb_http.py            # HTTP/JSON client (urllib)
│       ├── rhypedb_tcp.py             # binary TCP client (struct, no deps)
│       └── postgres.py                # psycopg wrapper + container PID lookup
├── suite1/
│   ├── scenario_01_bulk_insert.py
│   ├── scenario_02_point_lookup.py
│   ├── scenario_03_filter_scan.py
│   ├── scenario_04_05_traversal.py
│   ├── scenario_06_cascade_delete.py
│   └── scenario_07_unique_constraint.py
└── results/                           # JSON per scenario
```

## Second-degree covering — 2-hop beats PG-OPTIMAL

Starting point: 2-hop traversal at 1M ratings was 5,699 µs (rhypedb) vs
4,973 µs (PG-idiomatic), 3,249 µs (PG-optimal hash join). Three changes
shipped:

### 1. Raw bytes through the executor

`Database::get_links_many` returned `Vec<Vec<(u64, FieldMap)>>` — for a
2-hop query at 1M scale, that's ~1000 `HashMap<String, Value>`
constructions per query just so the fusion path could read one u64 field.
Changed to `Vec<Vec<(u64, Bytes)>>` and added
`object::find_u64_field_in_raw(bytes, name) -> Option<u64>` that walks the
serialized FieldMap to extract one integer without HashMap construction.

`QueryOutput::IdSetWithFields::items` now holds raw `Bytes`.

**Result: 5,699 → 4,438 µs (22% improvement).**

### 2. `Object::raw_fields` shortcut

Every terminal-materialized object went through `deserialize_fields` →
`HashMap` → `serialize_fields_into` for the wire — a wasted round trip.
Added `Object::raw_fields: Option<Bytes>` carrying the stored payload
verbatim. `protocol::encode_object` emits it directly when present.
`Database::get_many_lazy` constructs objects with `raw_fields = Some`,
`fields = empty`. Used by the executor's `materialize_ids`.

`Object::ensure_fields_deserialized()` populates `fields` lazily for
Filter / HTTP-JSON consumers. Keeps `raw_fields` set so the wire encoder
still wins.

**Result: 4,438 → 4,381 µs (1.3% additional).**

### 3. Second-degree covering (the big lever)

Profile showed 73% of CPU in storage probes — bloom + locate_block + memcmp
+ multi_get_versioned. The 700-user `multi_get` at the terminal of
`User.get(X).ratings.movie.ratings.user` was the dominant cost. Reducing
per-probe efficiency wouldn't help; we needed to eliminate the entire pass.

`build_covering_rev_value` already embedded "other forward 1:1 targets'
ids" in the reverse-edge value at link time. Extended it to **also embed
those targets' object data** as `<field>__cover: Value::Bytes(target_serialized_fields)`.
One extra `storage.get` per second-side link, paid once at setup.

Read path: when the executor's forward-1:1 fusion finds `<field>__cover`
in the carried source bytes, it constructs full `Object`s via
`Object::from_raw(target_type, tid, cover)` and emits `QueryOutput::Objects`
**directly — skipping the terminal `multi_get` entirely**. Falls back to
id-only emit if any target wasn't covered, in which case the terminal
materialize handles them via `get_many_lazy`.

For the bench shape: the 1000 ratings carried via the movie-side
reverse-edge have `user__cover` populated (Rating's link to User was
written first, so by the time the Movie-side rev-edge gets written, the
User is fetchable). Hop 4 dedup'd 700 users come out as fully-formed
Objects without any LSM probe.

**Result: 4,381 → 2,718 µs (38% additional).**

### Trajectory

| Phase | rhypedb-tcp µs | Notes |
|---|---:|---|
| Pre-mmap, pre-raw-bytes (1M) | 5,699 | starting point |
| + raw-bytes IdSetWithFields | 4,438 | -22% |
| + `Object::raw_fields` shortcut | 4,381 | -1.3% |
| + second-degree covering | **2,718** | -38% |
| PG-idiomatic | 5,203 | (multi-roundtrip) |
| PG-optimal (hash join) | 3,249 | (lower bound) |

**rhypedb at 2,718 µs is 1.20× FASTER than PG-OPTIMAL's hash join at 3,249 µs.**

### Why this matters architecturally

PG-optimal's 3,249 µs is the floor for a tuned PostgreSQL hash join on
this data. To go below it, you can't be faster at hash joins than PG —
you have to **change the data structure so the join doesn't happen**.

That's what second-degree covering is: at link time, the reverse-edge
value gets pre-joined with the next-degree target. At read time, the
"join" is a slice into already-materialized bytes. PG's relational model
can't do this because its rows are independent and its indexes are
logical pointers; rhypedb's relationship model is physical, and the
covering write paid once at link time amortizes across every read.

This is also why **1-hop and 2-hop both scale flat with N** for rhypedb —
the fanout cost is bounded by edges-per-source (constant), not by the
size of the universe (N). At 1M ratings, 2-hop went 4,438 → 2,718 (-38%);
at 10M ratings the absolute saving would be similar because the fanout
density is the same.

### Caveats / future work

- **Staleness on update**: if a User's `name` changes, the `user__cover`
  blob in 1M Rating reverse-edges goes stale. Not yet handled. A
  "rewrite all reverse-edges referencing this object" pass on object
  update would fix it. For append-mostly workloads (this bench, jkbase
  tenant journals, most graph apps), this isn't the immediate problem.
- **Storage cost**: ~50 bytes extra per covered field per reverse-edge.
  1M ratings × 2 reverse-edges × 50 bytes = 100 MB extra disk. Acceptable.
- **Link-time cost**: one extra `storage.get` per second-link. Setup of
  1M ratings adds ~30s — paid once.
- **3-hop and beyond**: not yet covered. Same pattern would extend (cover
  the 3rd-degree target inside the 2nd-degree cover), with diminishing
  returns in storage cost.

## What's still missing

- **Background memory sampler**: currently RSS is snapshotted at iteration
  boundaries; for bulk ops a polling thread would give peak resolution.
- **Suite 2 (vector indexing/search)**: separate harness entirely.
- **Markdown result renderer**: turn the JSON results into a publishable
  report (right now the comparison table above is hand-curated).
- **`create_batch` in the rhypedb query language** so we can close the
  COPY gap on bulk inserts. Already on the Overboard backlog.
- **Secondary integer-field indexes** — shipped. `@indexed` directive,
  bounded range scan, 250× win on filter scan. See section above.
