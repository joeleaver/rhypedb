# Vector Search

Vectors are a native field type in rhypedb, not an add-on. You declare a `Vector<N>` field, optionally index it for approximate nearest-neighbour (ANN) search, and optionally have the server compute embeddings for you from text. Then you query it with `.similar(...)`.

## Declaring a vector field

```
type Post {
    body: String
    embedding: Vector<384>
}
```

`Vector<N>` stores an `N`-dimensional `f32` vector. The dimension `N` is fixed by the schema. On its own, this field stores whatever vectors you write to it; to search it efficiently you add an index.

## Two ways to get vectors in

### 1. Bring your own vectors

Compute embeddings in your application and write them like any other field:

```
Post.create({ body: "hello world", embedding: [0.12, -0.04, 0.98, ...] })
```

The vector literal must have exactly `N` elements. For bulk ingestion of precomputed vectors, the binary TCP protocol has a dedicated `VectorBatch` message (see the [API Reference](api-reference.md)).

### 2. Server-side embedding with `@vectorize`

Let the server embed text for you. Point `@vectorize` at a `String` field and a model:

```
type Post {
    body: String
    embedding: Vector<384> @vectorize(source: "body", model: "all-MiniLM-L6-v2")
}
```

Now you only supply the text — the server fills in the vector on create and re-embeds on update:

```
Post.create({ body: "distributed systems are hard" })
```

The embedding is computed asynchronously by a background worker. `GET /status` reports how many embeddings are pending, how many vectors each index holds, and whether the embedding model is currently loaded:

```json
{
  "vectorizer": {
    "pending": 3,
    "indexes": [ { "name": "Post.embedding", "vectors": 1000 } ],
    "model_loaded": true,
    "model_error": null
  }
}
```

`model_loaded` is `true` once the model has loaded successfully at least once. `model_error` carries the most recent load-failure message while one is in effect (for example, the model is still downloading, or this binary was built without the `fastembed` feature) and goes back to `null` as soon as a load succeeds. A model-load failure does not lose or fail any pending jobs — they stay queued and are retried with a backoff; see [Embedding pipeline settings](#embedding-pipeline-settings) below.

> The `Vector<N>` dimension must match the model's output size (e.g. `all-MiniLM-L6-v2` produces 384-dimensional vectors).

## Embedding pipeline settings

The background embed worker (and the query-time embedder used by a text `.similar`) can be tuned via the `[vectorizer]` table in `rhypedb.toml` (see [Running rhypedb](operations.md#configuration-file)):

| Key | Default | What it costs |
| --- | --- | --- |
| `batch_size` | `32` | Jobs claimed and embedded per pass. Higher batches trade memory for throughput — at 512-token texts, a batch of 256 (the old hard-coded value) could take a desktop process to 15 GB resident; 32 keeps peak memory bounded. |
| `max_length` | `256` | Token limit per text. MiniLM/BGE-small were trained at 256; attention memory grows with the *square* of this, so raising it is expensive. |
| `intra_threads` | half the CPU cores | ONNX Runtime intra-op threads used by each lazily-loaded model. |
| `quantized` | `true` | Prefer the int8-quantized model variant when fastembed has one (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`); smaller and faster, a small quality cost. Base/large BGE models have no quantized variant and always use fp32 regardless of this setting. |
| `cache_dir` | fastembed's own default | Where downloaded model files are cached on disk. |
| `cross_encoder` | `"off"` | `"off"` disables cross-encoder reranking of `.similar` text search; any other string names the reranker model to turn it on with (see [Cross-encoder reranking](#cross-encoder-reranking) below). A second model, off by default. |

If the model fails to load — a network hiccup mid-download, a cold cache on first boot, or simply the time it takes to download and initialize a ~100MB+ ONNX model — the background worker does **not** fail or drop the affected jobs. It puts them back on the queue, records the failure under `vectorizer.model_error` on `GET /status`, and retries with an exponential backoff (starting at 2s, doubling up to a 60s cap) until a load succeeds, at which point `model_error` clears and the backoff resets. A `.similar` text query made while the model is unavailable gets a clear `ModelUnavailable` error rather than panicking: the first query after a failure attempts the load itself (and waits for that attempt, which for a download can be the network timeout); queries during the backoff window are refused immediately without retrying, so a known-bad model doesn't cost every query a download attempt. Embedded callers driving `process_pending` see the same failure as an `Err(ModelUnavailable)` when a batch could not be embedded at all (its jobs are re-queued), so a "loop until 0" driver can back off rather than spin.

> **Upgrading from a build before this setting existed:** the cross-encoder used to run by default and was opted *out* of with `RHYPEDB_DISABLE_RERANK`. It is now **off unless you set `cross_encoder`** — a deployment that relied on the old default silently loses reranking after upgrading (queries still succeed; `GET /status` shows `reranker_loaded: false`). Add `[vectorizer] cross_encoder = "bge-reranker-base"` to keep it. `RHYPEDB_DISABLE_RERANK` is no longer read (the server warns at startup if it, or `RHYPEDB_RERANKER_DIR` with the cross-encoder off, is set).

### Cross-encoder reranking

`.similar` text search can optionally run a SECOND pass over the ANN
candidates: a cross-encoder model scores each candidate's source text
directly against your query text, which typically ranks results more
accurately than vector distance alone — at the cost of one extra model
(~280MB) and one forward pass per candidate. It is **off by default**
(`cross_encoder = "off"`); turn it on by naming the reranker model — today
exactly one is supported:

```toml
[vectorizer]
cross_encoder = "bge-reranker-base"
```

This is a **separate knob from the per-query `rerank:` argument** (see
[Searching — `.similar`](#searching--similar) below) — `rerank: N` always
means "re-score `N` candidates against the exact `f32` vectors," a cheap
step that is always available. The cross-encoder runs *in addition to* that,
whenever it is turned on here AND the field has a readable `@vectorize`
source text; a raw-vector `.similar` query never uses it (there is no query
text to score against). Like the embedder, a cross-encoder load failure is
fail-soft: `.similar` falls back to returning the ANN/rescored order rather
than erroring, and the failure is visible under `vectorizer.reranker_error`
on `GET /status` (`vectorizer.reranker_loaded` mirrors `model_loaded`).

Reranker environment knobs (all optional; the config key above is the only
switch that turns the cross-encoder on):

| Variable | Effect |
| --- | --- |
| `RHYPEDB_RERANKER_DIR` | Load the reranker from a local directory (`model.onnx` plus the four tokenizer files) instead of downloading it. |
| `RHYPEDB_RERANKER_FP32` | Use the full-precision `bge-reranker-base` instead of the default int8-quantized build (larger, slower, marginally more accurate). |
| `RHYPEDB_RERANK_CANDIDATES` | How many ANN candidates the cross-encoder scores per query (default `min(k * 3, 48)`); this is the dominant query cost when reranking is on. |
| `RHYPEDB_DEBUG_RERANK` | Log the search path taken (brute-force / ANN / rerank pool) to stderr. |

## Indexing for search — `@index(hnsw, ...)`

To run similarity search at scale, add an HNSW index:

```
embedding: Vector<384> @vectorize(source: "body", model: "all-MiniLM-L6-v2")
                       @index(hnsw, metric: cosine, quantization: turboquant_4bit, m: 16, ef_construction: 200)
```

| Parameter | Values | Default | Notes |
| --- | --- | --- | --- |
| `metric` | `cosine`, `l2`, `dot_product` | `cosine` | distance function |
| `quantization` | `turboquant_2bit`, `turboquant_3bit`, `turboquant_4bit` | `turboquant_4bit` | in-index compression |
| `m` | ≥ 1 | engine default | graph fan-out |
| `ef_construction` | ≥ 1 | engine default | build-time accuracy/effort |

The index stores **quantized** vectors (TurboQuant, 2–4 bits per component) for speed and a small memory footprint; the **raw `f32` vectors are kept losslessly at rest** and are what get re-ranked and what survive a backup or logical export. `quantization: none` is rejected — there is always an index-side quantization.

The index is persisted as `hnsw_<field>.bin` in the data directory and rebuilt from the raw vectors if missing or stale (for example, after a logical import).

## Searching — `.similar`

```
Post.similar(.embedding, "vector databases", k: 10)
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `.field` | the `Vector` field to search |
| query | a `"text"` string (only if the field has `@vectorize`) or a raw `[f32, ...]` vector |
| `k:` | number of nearest neighbours to return |
| `ef:` | (optional) HNSW search width — higher = better recall, more work |
| `rerank:` | (optional) re-score this many ANN candidates with full-precision vectors before returning the top `k` |

A text query is embedded with the same model as the field's `@vectorize`. A raw-vector query works on any vector field and must match the field dimension.

```
// text query (field must have @vectorize)
Post.similar(.embedding, "distributed consensus", k: 5)

// raw vector query
Post.similar(.embedding, [0.01, 0.42, ...], k: 5)

// tune recall: widen the search and rerank the top 50 with exact vectors
Post.similar(.embedding, "distributed consensus", k: 10, ef: 200, rerank: 50)

// narrow the candidate set with a filter first
Post.filter(.published == true).similar(.embedding, "rust async", k: 10)
```

Every `.similar` result is **ranked** and each carries a `score`, read as `Row.score`. Ordinarily `score` is the index's distance under the field's metric (**lower** is closer) and results come back nearest-first. When [cross-encoder reranking](#cross-encoder-reranking) is on and the reranker actually scored a row, `score` is instead the cross-encoder's relevance score (**higher** is more relevant) and those rows come back most-relevant-first — the two scales are not comparable, so know which one your deployment uses. Rows the cross-encoder could not score (a raw-vector query, the reranker failed to load, or the object has no readable source text) keep a distance; see [Ranked results](queries.md#ranked-results) for the exact rules. See [Ranked results](queries.md#ranked-results) and, for keyword search over the same objects, [`.matches`](queries.md#full-text-search--matchesfield-query-k-n).

## Tuning recall vs. latency

Three knobs trade accuracy for speed:

- **`quantization`** (schema, per field) — more bits = higher base recall, larger index. `turboquant_4bit` is a good default; drop to `3bit`/`2bit` to shrink the index and lean on `rerank`.
- **`ef`** (per query) — a larger HNSW search width explores more of the graph. Raises recall at a latency cost.
- **`rerank`** (per query) — pull a larger candidate pool from the (quantized) index, then re-score those candidates with the exact `f32` vectors and return the best `k`. This recovers most of the precision lost to quantization for a modest cost.

A common high-recall pattern is a moderate `ef` with `rerank` set to a few × `k`:

```
Post.similar(.embedding, "query text", k: 10, ef: 200, rerank: 50)
```

Server-wide defaults for `ef` and `rerank` can also be set via the `RHYPEDB_EF` and `RHYPEDB_RERANK` environment variables (see [Running rhypedb](operations.md)); they apply only to queries that omit the corresponding argument, and an explicit per-query `ef:`/`rerank:` (including `rerank: 0` to force rerank off) always overrides them.

## What's preserved across backups

Because the raw `f32` vectors are stored losslessly, both a physical backup and a logical export preserve your vectors exactly. A logical export ships the raw vectors and the HNSW graph is **rebuilt on import** (the graph itself holds only lossy quantized codes and is cheap to regenerate). See **[Backup & Recovery](backup-recovery.md)**.
