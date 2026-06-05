# Benchmark Plan

A reference for the rhypedb performance benchmark suite. The goal is to produce
numbers we can publish — fair, reproducible, and honest about what each system
is and isn't built for.

## Goals & non-goals

**Goals**
- Establish a baseline for rhypedb performance vs. an established peer (Postgres)
- Measure the axes we care about: throughput, latency distribution, memory, disk
- Capture rhypedb's structural advantages (single-query path traversal, integrated
  vectors, low memory) honestly
- Create a reproducible harness we can re-run after every optimization

**Non-goals**
- "Rhypedb wins everything" framing. We expect Postgres to win raw single-op
  throughput on simple operations until our binary protocol + prepared statements
  land. The point is to measure where we stand, not to cherry-pick wins.
- Microbenchmarks of internal operations. This is end-to-end client→server→storage
  measurement.

## Two suites

The suites are independent so they can be run separately. Each lives in its own
docker-compose stack.

### Suite 1 — Relational / graph

**What rhypedb is built for that Postgres can also do:** typed entities, relationships,
filter scans, traversals, cascade deletes, unique constraints.

**Workload:** social-graph-shaped dataset (users + movies + ratings + friend edges).
MovieLens 1M or synthetic.

**Scenarios:** see _Scenarios — Suite 1_ below.

### Suite 2 — Vector indexing / search

**What rhypedb is built for that Postgres+pgvector can also do:** semantic search
over a corpus, end-to-end including embedding generation.

**Workload:** document corpus (Bible verses, or BEIR-style — TBD when we wire it
up). Same embedding model on both sides (all-MiniLM-L6-v2).

**Scenarios:** see _Scenarios — Suite 2_ below.

## Measurement methodology

### Unit of measurement: task wall-clock

The benchmark unit is "how long does the task take, end-to-end, from the client's
perspective." Not per-query throughput.

This matters because rhypedb's path query can express things in one query +
roundtrip that take 2-5 queries + roundtrips in Postgres. Measuring per-query
throughput would hide that.

### Three implementations per task (where applicable)

For each scenario in Suite 1, we time three implementations:

1. **rhypedb** — idiomatic path query
2. **Postgres idiomatic** — what a typical app would do via an ORM-style driver:
   multiple queries with separate roundtrips
3. **Postgres optimal** — single hand-tuned query (recursive CTE, complex joins)
   as a lower bound on what's achievable

This prevents two unfair framings:
- "You're not using Postgres right" (we are — the optimal column)
- "Nobody writes Postgres queries that complicated" (right — the idiomatic column
  shows what real apps experience)

Suite 2 scenarios mostly have a single sensible implementation per system.

### Latency reporting

- Full distribution via HDR histogram (or equivalent)
- Report p50, p95, p99, p99.9
- Don't report averages alone — they hide tail latency

### Memory axes

All measured via cgroup `memory.peak` (most accurate) or `/proc/PID/status` RSS
sampled at intervals.

- **Cold RSS** — server started, no data loaded
- **Post-load RSS** — after the dataset is loaded
- **Peak RSS during bulk op** — high water mark during indexing / bulk insert
- **Steady-state RSS during queries** — running a query workload for N minutes
- **Disk usage at rest** — `du -sh` of the data directory

### Throughput

For sustained workloads we also report ops/sec. But it's secondary to wall-clock
and latency; this isn't a TPC-C-style throughput benchmark.

### Iterations & error bars

- Each scenario runs N=10 iterations (configurable)
- Drop the warmest 2 iterations (warmup)
- Report median + IQR or full distribution

## Harness architecture

```
benchmarks/
├── PLAN.md                # this doc
├── docker-compose.yml     # rhypedb + postgres services
├── harness/               # Python driver
│   ├── common/            # shared utilities (memory sampling, HDR, reporting)
│   ├── suite1/            # relational/graph scenarios
│   └── suite2/            # vector scenarios
├── data/                  # downloaded/generated datasets
└── results/               # output JSON + rendered markdown
```

**Driver language:** Python — easy to write, the bottleneck is the systems being
measured, not the driver. Use:
- `httpx` (rhypedb HTTP) → swap for binary protocol client once that lands
- `psycopg` v3 (Postgres)
- `psutil` (memory sampling)
- `hdrhistogram` (latency)

**Isolation:** each scenario runs in a fresh container to avoid cross-contamination
(memtable carryover, buffer cache warmth from previous tests). The harness:
1. Starts both DB containers
2. Loads the dataset
3. Runs the scenario N times
4. Captures memory at each phase
5. Tears down containers
6. Repeats for the next scenario

**Hardware:** document the host CPU/RAM/disk, run all scenarios on the same box,
avoid running anything else concurrently.

## Scenarios — Suite 1 (relational / graph)

Schema (rhypedb SDL):

```
type User {
    id: u64
    name: String
    email: String @unique
    friends: [User] @on_delete(remove)
    ratings: [Rating] @inverse(Rating.user)
}

type Movie {
    title: String
    year: u32
    ratings: [Rating] @inverse(Rating.movie)
}

type Rating {
    user: User @on_delete(cascade)
    movie: Movie @on_delete(cascade)
    stars: f32
}
```

Postgres equivalent: `users`, `movies`, `ratings`, `friendships` tables with the
usual PK/FK indexes.

### Scenarios

| # | Task | rhypedb expression | PG idiomatic | PG optimal |
|---|------|----|----|----|
| 1.1 | Bulk insert 10K users + 1K movies + 100K ratings | sequence of `create` (later: `create_batch`) | parameterized INSERT loop | `COPY ... FROM STDIN` |
| 1.2 | Point lookup by ID | `User.get(id)` | `SELECT FROM users WHERE id=$1` | same |
| 1.3 | Filter scan: ratings by user X | `Rating.filter(.user.id == X)` | `SELECT FROM ratings WHERE user_id=$1` | same |
| 1.4 | 1-hop: movies user X has rated | `User.get(X).ratings.movie` | 2 queries (ratings, then movies by id) | single JOIN |
| 1.5 | 2-hop: other users who rated movies user X rated | `User.get(X).ratings.movie.ratings.user` | 3-4 queries with roundtrips | single 3-way JOIN |
| 1.6 | Cascade delete a user → all ratings gone | `User.get(X).delete()` | `DELETE FROM users WHERE id=$1` with FK CASCADE | same |
| 1.7 | Unique constraint: insert duplicate email | `User.create({email: existing})` should fail | `INSERT ... ON CONFLICT` | same |

**Reports per scenario:** task wall-clock distribution per implementation, memory
at start/post-load/during-op/end, disk at rest.

## Scenarios — Suite 2 (vector indexing / search)

Schema (rhypedb SDL):

```
type Doc {
    text: String
    embedding: Vector<384> @vectorize(source: "text", model: "all-MiniLM-L6-v2")
                          @index(hnsw, metric: cosine, quantization: turboquant_3bit)
}
```

Postgres equivalent: `docs(id, text, embedding vector(384))` with a pgvector HNSW
index, plus a sidecar Python service (sentence-transformers) for embedding
generation since pgvector doesn't generate embeddings.

### Scenarios

| # | Task | Notes |
|---|------|----|
| 2.1 | Indexing throughput: docs/sec end-to-end | includes embedding generation |
| 2.2 | Time-to-searchable: insert → first queryable | for the async pipeline |
| 2.3 | k-NN query latency, k=10 | p50/p95/p99 over N queries from a held-out set |
| 2.4 | k-NN query latency, k=100 | same |
| 2.5 | Recall@10 vs. brute-force ground truth | quality, not just speed |
| 2.6 | Memory during indexing (peak) | both DB and sidecar (where applicable) |
| 2.7 | Memory at rest with N vectors | total system memory |

**Fairness notes for Suite 2:**

- Both sides use the same embedding model (all-MiniLM-L6-v2)
- Both sides include the embedding sidecar/process in the memory total — rhypedb's
  in-process ONNX vs PG's external sidecar
- Both sides use HNSW with comparable parameters (m=16, ef_construction=100)
- pgvector doesn't have built-in quantization at the recall@10 level we have with
  TurboQuant 3-bit; we report both raw recall and disk footprint
- For "time-to-searchable" we measure when the inserted doc appears in a query —
  this favors rhypedb's async pipeline architecture if pgvector requires synchronous
  indexing

## Reporting

Each run produces a JSON file in `results/`. A renderer turns it into Markdown
with tables. Format:

```
results/2026-06-04-suite1.json
results/2026-06-04-suite1.md
```

For publication, we render a single markdown comparison document with:
- One table per scenario showing wall-clock distributions
- One table for memory comparison
- One paragraph of interpretation per scenario — honest about who wins what

## Honest framing for the published report

The intro should explicitly say:
1. Postgres is mature, optimized over decades, has a real query planner. We're a
   pre-MVP storage engine.
2. We expect to lose raw throughput on most simple ops today (HTTP protocol,
   no prepared statements, no planner).
3. We expect to win on: integrated vectors+rerank in one process, lower baseline
   memory, faster cold start, cleaner traversal queries.
4. The point is the **gap** — measuring it lets us close it.

## Open questions

- **Dataset for Suite 1:** MovieLens 1M, or synthetic? MovieLens is realistic but
  doesn't have friend edges; we'd synthesize those.
- **Dataset for Suite 2:** Bible verses (reuse what we have), or BEIR? BEIR has
  ground-truth relevance labels.
- **Where to publish:** README appendix, separate site, blog post?
- **CI integration:** should benchmarks run on every push, or only when manually
  invoked? Probably the latter — the runs take real time and need stable hardware.
