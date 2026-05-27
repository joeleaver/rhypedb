# rhypedb

A strongly-typed object database with first-class relationships and first-class vector support. Written in Rust.

## What is rhypedb?

rhypedb is a database that treats objects, relationships, and vectors as core primitives — not afterthoughts bolted onto a relational or document model.

- **Objects, not rows.** Define typed schemas. Objects conform to their type. No impedance mismatch between your data model and your domain model.
- **Relationships are properties.** Declare links between objects in your schema. The database stores, traverses, and maintains them. No foreign keys, no junction tables, no JOINs.
- **Vectors are native.** `Vector<N>` is a type like `String` or `u32`. Compressed at rest with [TurboQuant](https://github.com/0xSero/turboquant)-adapted quantization, indexed with HNSW, queryable with similarity operators.
- **Real-time subscriptions.** Subscribe to query patterns and get semantically meaningful change events.

## Schema

```
type User {
  name: String
  email: String @unique
  friends: [User] @on_delete(remove)
  posts: [Post] @inverse(Post.author)
  embedding: Vector<1536>
}

type Post {
  title: String
  author: User @on_delete(cascade)
  tags: [Tag] @on_delete(remove)
  embedding: Vector<1536>
}
```

## Query Language

Path-based traversal with filtering and vector search:

```
// Find sci-fi movies similar to a query, favorited by friends
User.get(1)
  .friends
  .favorite_movies { rating, added_at }
  .filter(.genre == "sci-fi")
  .similar(.embedding, query_vec, k: 10)
```

## Design Goals

- **Low memory.** Designed to run inside Firecracker microVMs. TurboQuant compresses vectors 4-8x. Scale-to-zero via VM snapshots.
- **Referential integrity.** The database guarantees no dangling references. Delete policies (`remove`, `cascade`, `deny`) are declared in the schema and enforced atomically.
- **MVCC.** Snapshot isolation with write-write conflict detection. Readers never block writers.
- **Built for jkbase.** Default database for the [jkbase](https://github.com/joeleaver/jkbase) cloud platform. Single binary, no external dependencies.

## Status

Early development. The storage engine (LSM-tree + WAL + MVCC) is implemented with 36 passing tests. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Building

```
cargo build
cargo test
```

## License

MIT
