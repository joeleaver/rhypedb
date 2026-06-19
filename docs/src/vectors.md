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

The embedding is computed asynchronously by a background worker. `GET /status` reports how many embeddings are pending and how many vectors each index holds:

```json
{ "vectorizer": { "pending": 3, "indexes": [ { "name": "Post.embedding", "vectors": 1000 } ] } }
```

> The `Vector<N>` dimension must match the model's output size (e.g. `all-MiniLM-L6-v2` produces 384-dimensional vectors).

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

## Tuning recall vs. latency

Three knobs trade accuracy for speed:

- **`quantization`** (schema, per field) — more bits = higher base recall, larger index. `turboquant_4bit` is a good default; drop to `3bit`/`2bit` to shrink the index and lean on `rerank`.
- **`ef`** (per query) — a larger HNSW search width explores more of the graph. Raises recall at a latency cost.
- **`rerank`** (per query) — pull a larger candidate pool from the (quantized) index, then re-score those candidates with the exact `f32` vectors and return the best `k`. This recovers most of the precision lost to quantization for a modest cost.

A common high-recall pattern is a moderate `ef` with `rerank` set to a few × `k`:

```
Post.similar(.embedding, "query text", k: 10, ef: 200, rerank: 50)
```

Server-wide defaults for `ef` and `rerank` can also be set via the `RHYPEDB_EF` and `RHYPEDB_RERANK` environment variables (see [Running rhypedb](operations.md)); per-query arguments override them.

## What's preserved across backups

Because the raw `f32` vectors are stored losslessly, both a physical backup and a logical export preserve your vectors exactly. A logical export ships the raw vectors and the HNSW graph is **rebuilt on import** (the graph itself holds only lossy quantized codes and is cheap to regenerate). See **[Backup & Recovery](backup-recovery.md)**.
