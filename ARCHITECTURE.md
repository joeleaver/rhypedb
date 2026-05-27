# rhypedb Architecture

## Overview

rhypedb is a strongly-typed object database with first-class relationships and first-class vector support. It is written in Rust and designed as the default database offering for the [jkbase](https://github.com/joeleaver/jkbase) cloud platform, running inside Firecracker microVMs with tight memory constraints and scale-to-zero requirements.

## Data Model

### Types, Not Tables

Users define object types via a Schema Definition Language (SDL). Each type has typed properties and typed relationships. Objects conform to their type's schema — no half-populated records, no schema-on-read ambiguity.

### Relationships Are Properties

Relationships are declared in the schema as typed links between objects. The database stores, indexes, traverses, and maintains them — they are not foreign keys or JOINs. This eliminates junction tables, makes traversal a pointer-follow rather than a hash-join, and lets the database maintain referential integrity automatically.

Relationships can carry typed edge properties (e.g. a `rating: f32` on a `favorite_movies` link).

### Vectors Are a Native Type

`Vector<N>` is a first-class property type alongside `String`, `u32`, `Bool`, etc. Vectors are compressed at rest via TurboQuant (adapted from the ICLR 2026 KV cache compression technique), indexed via HNSW, and queryable with similarity operators.

### Schema Definition Language (SDL)

```
type User {
  name: String
  email: String @unique
  reputation: u32
  friends: [User] @on_delete(remove)
  posts: [Post] @inverse(Post.author)
  embedding: Vector<1536>
}

type Post {
  title: String
  body: String
  author: User @on_delete(cascade)
  tags: [Tag] @on_delete(remove)
  embedding: Vector<1536>
}

type Tag {
  name: String @unique
}
```

SDL is the source of truth. Client SDKs (Rust, JS) are generated from it.

### Referential Integrity

The database guarantees no dangling references. Delete policies are declared per-relationship:

- `@on_delete(remove)` — edge silently removed (default for to-many)
- `@on_delete(cascade)` — deletion cascades to this object
- `@on_delete(deny)` — deletion rejected if references exist

All side-effects execute atomically within a single transaction.

## Storage Engine

### LSM-Tree + Write-Ahead Log

The storage engine is a custom LSM-tree with write-ahead log, built for:

- **Write-heavy edge operations** — relationship creation/deletion is frequent
- **Natural MVCC** — multiple versions coexist in the LSM until compaction
- **WAL-based subscriptions** — every mutation is sequenced, CDC stream is nearly free
- **Firecracker-friendly lifecycle** — flush memtable before VM snapshot, restore from WAL on wake
- **Future sharding** — SST files with key ranges map to shard boundaries

### Three Logical Stores

All stores share one LSM keyspace with prefixed keys:

```
o:<TypeID>:<ObjectID>                        → serialized properties
e:<SourceID>:<RelName>:<TargetID>            → edge properties (or empty)
r:<TargetID>:<RelName>:<SourceID>            → reverse edge (empty value)
v:<TypeID>:<ObjectID>:<FieldName>            → TurboQuant-compressed vector
```

- **Object Store** — keyed by `(TypeID, ObjectID)`, stores serialized property values
- **Edge Store** — forward index keyed by `(SourceID, RelationshipName, TargetID)`, reverse index keyed by `(TargetID, RelationshipName, SourceID)` for inverse traversal and cascade lookups
- **Vector Store** — TurboQuant-compressed vectors stored separately for cache efficiency, with memory-mapped HNSW indexes

### Components

- **WAL** (`wal.rs`) — Append-only log with CRC32 integrity checks. Supports crash recovery via replay with truncated-tail handling.
- **Memtable** (`memtable.rs`) — Lock-free skip list (crossbeam-skiplist) with MVCC-aware versioned reads and tombstone support.
- **SST** (`sst.rs`) — Sorted String Table file format with sparse index, versioned point lookups, and full iteration.
- **LSM** (`lsm.rs`) — Orchestrates WAL + memtable + SST flush + compaction + multi-level reads.
- **MVCC** (`mvcc.rs`) — Transaction manager with snapshot isolation and write-write conflict detection.

## Vector Engine

### TurboQuant Adaptation

Adapted from the ICLR 2026 KV cache compression technique ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). CPU-only implementation using Rust SIMD.

Quantization pipeline at ingest:
1. **Orthogonal rotation** — distributes variance evenly across dimensions
2. **Lloyd-Max scalar quantization** — 2-4 bits per dimension using optimal codebooks calibrated per-type
3. **QJL residual encoding** — preserves information lost in quantization (1 bit per dimension)
4. **Bit-packing** — 4 values/byte at 2-bit, 2/byte at 4-bit

The combined estimator is mathematically unbiased, preserving search recall without systematic skew. Net compression: ~4-8x for typical embedding dimensions.

### HNSW Index

Hierarchical Navigable Small World graph for approximate nearest neighbor search:
- Separate memory-mapped index per vector field per type
- Asymmetric search: full-precision queries against quantized stored vectors
- Tombstone-based deletes with periodic graph repair during compaction
- Configurable via SDL: `@index(hnsw, metric: cosine, quantization: turboquant_3bit)`

## Vectorizer

### Server-Side Encoding

rhypedb encodes text into vectors server-side — clients never need to interact with an embedding model directly. This is configured in the schema via the `@vectorize` directive:

```
type Post {
  body: String
  embedding: Vector<384> @vectorize(
    source: "body",
    model: "all-MiniLM-L6-v2"
  ) @index(hnsw, metric: cosine, quantization: turboquant_3bit)
}
```

When a Post is created or its `body` field is updated, rhypedb automatically encodes the text into a 384-dimensional vector, compresses it with TurboQuant, and inserts it into the HNSW index. Similarity queries accept text strings that are encoded transparently at query time:

```
Post.similar(.embedding, "distributed systems", k: 10)
```

### Async Vectorization Pipeline

Embedding inference takes 10-30ms per document on CPU. To avoid blocking the write path, vectorization is asynchronous:

```
Client: Post.create({ body: "hello world" })
  │
  ├─ Immediate: object stored in LSM, return to client
  │
  └─ Background: encode text → TurboQuant compress → HNSW insert
```

Objects are queryable by scalar fields and relationships the moment `create` returns. Vectors become searchable after the background worker processes them.

**Components:**

1. **Vectorization queue** — persistent queue stored in the LSM (survives crash and scale-to-zero). When an object with a `@vectorize` field is created or its source field is updated, a job is enqueued.

2. **Background worker** — dedicated thread pool that pulls from the queue, runs the embedding model (via ONNX Runtime), compresses with TurboQuant, and inserts into the HNSW index. Batches documents for efficient inference.

3. **Vector state** — each vector field tracks its state: `pending` (queued, not yet searchable), `indexed` (compressed and in HNSW), or `failed` (encoding error). Similarity queries only search indexed vectors.

4. **Backpressure** — if the queue grows during burst inserts, the worker batches efficiently. Embedding models are faster with larger batches. The queue depth is observable for monitoring.

### Default Embedding Model

The default model is `all-MiniLM-L6-v2`:
- 384 dimensions, ~80MB model size
- <30ms per document on CPU
- Good quality for general-purpose semantic search
- Available as quantized ONNX model via the `fastembed` crate (`ort` ONNX Runtime bindings)

With TurboQuant 3-bit compression, each 384-dim vector is stored in ~150 bytes.

### Scale-to-Zero Behavior

On Firecracker VM snapshot, the vectorization queue is persisted in the WAL. On wake, the background worker resumes processing any pending jobs. No vectors are lost.

## Transaction Model

### MVCC with Snapshot Isolation

- **Monotonic transaction counter** — atomic u64 logical clock
- **Snapshot reads** — each transaction sees a consistent point-in-time. Readers never block writers.
- **Write-write conflict detection** — at commit time, check for overlapping write sets with concurrently committed transactions. Conflict → abort + retry.
- **Snapshot isolation by default, serializable opt-in.** Snapshot isolation tracks write sets (cheap). Serializable also tracks read sets (more aborts, rarely needed).

### Version Cleanup

Old versions accumulate in the LSM. During compaction, versions older than the oldest active transaction's snapshot are discarded.

## Query Language

Path-based query language with method chaining, relationship traversal, and vector similarity as first-class operators:

```
// Traverse relationships with filtering and vector search
User.get(1)
  .friends
  .favorite_movies { rating, added_at }
  .filter(.genre == "sci-fi")
  .similar(.embedding, query_vec, k: 10)

// Mutations
User.create({ name: "Alice", email: "alice@example.com" })
User.get(1).favorite_movies.link(Movie.get(42), { rating: 4.5 })
Movie.get(42).delete()  // triggers @on_delete policies
```

## Wire Protocol

- **Binary protocol** (primary) — custom binary over TCP with multiplexed request/response and streaming for subscriptions
- **HTTP + WebSocket gateway** — JSON-over-HTTP for queries/mutations, WebSocket for subscriptions. Designed for jkbase WASM functions and JS clients.

## Real-Time Subscriptions

Built on the WAL commit stream. Subscriptions are registered query patterns — clients get semantically meaningful change events, not raw key mutations.

```
subscribe(User.filter(.reputation > 100))
subscribe(User.get(1).friends.posts)
```

## jkbase Integration

- Single binary, no external dependencies
- Runs as a persistent service inside per-project Firecracker microVMs
- Scale-to-zero: flush memtable to SST, Firecracker snapshots the VM. On wake (~5ms), reconstruct from WAL tail.
- Schema and config in `jkbase.toml`, credentials via jkbase secrets system

## Crate Structure

```
crates/
  rhypedb-storage/    # LSM-tree, WAL, memtable, SST, MVCC
  rhypedb-schema/     # SDL parser, type system (planned)
  rhypedb-server/     # Main binary
```
