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

One path-based language for lookups, filters, relationship traversal, mutations, and vector search:

```
// Traverse relationships
User.get(1).friends.posts.filter(.title != "")

// Create and relate
Post.create({ title: "Hello", author: User.get(1) })
User.get(1).favorite_movies.link(Movie.get(42), { rating: 4.5 })

// Vector search, optionally narrowed by a filter
Post.filter(.published == true).similar(.embedding, "vector databases", k: 10)
```

See the [Query Language](docs/src/queries.md) reference for the full grammar.

## Design Goals

- **Low memory.** Designed to run inside Firecracker microVMs. TurboQuant compresses vectors 4-8x. Scale-to-zero via VM snapshots.
- **Referential integrity.** The database guarantees no dangling references. Delete policies (`remove`, `cascade`, `deny`) are declared in the schema and enforced atomically.
- **MVCC.** Snapshot isolation with write-write conflict detection. Readers never block writers.
- **Built for jkbase.** Default database for the [jkbase](https://github.com/joeleaver/jkbase) cloud platform. Single binary, no external dependencies.

## Documentation

📖 **[Read the docs → joeleaver.github.io/rhypedb](https://joeleaver.github.io/rhypedb/)**

Built with [mdBook](https://rust-lang.github.io/mdBook/) (sources in [`docs/`](docs/); `mdbook serve docs` to preview locally):

- [Getting Started](docs/src/getting-started.md) — define a schema, run the server, query it.
- [Schema Reference](docs/src/schema.md) — types, relationships, directives.
- [Query Language](docs/src/queries.md) — sources, steps, operators, mutations.
- [Vector Search](docs/src/vectors.md) — `Vector<N>`, `@vectorize`, HNSW, recall tuning.
- [Schema Migrations](docs/src/migrations.md) — adding/removing fields, renames, online type changes.
- [Running rhypedb](docs/src/operations.md) — configuration, auth, monitoring.
- [Backup & Recovery](docs/src/backup-recovery.md) — physical snapshots and logical export/import.
- [API Reference](docs/src/api-reference.md) — HTTP, CLI, and the binary TCP protocol.

To read it as a rendered site: `mdbook serve docs` (or `mdbook build docs`).

## Quick start

```bash
# build the server and CLI
cargo build --release -p rhypedb-server -p rhypedb-cli

# run a server on the example schema
target/release/rhypedb-server --schema examples/blog.rhype --data-dir ./data
```

In another terminal, talk to it with the CLI (`-e` runs one query; with no args it starts a REPL):

```bash
# create
rhypedb-cli -e 'User.create({ name: "Alice", email: "alice@example.com", age: 30 })'
rhypedb-cli -e 'Post.create({ title: "Hello", body: "first post", author: User.get(1) })'

# read & traverse relationships
rhypedb-cli -e 'User.filter(.age > 18).limit(10)'
rhypedb-cli -e 'User.get(1).posts'

# mutate
rhypedb-cli -e 'User.get(1).update({ age: 31 })'
```

The same queries work over HTTP: `curl -s localhost:4200/query -d '{"query":"User.get(1)"}'`.

### Evolving the schema

**Additive changes need no migration.** Add a field and reapply the schema — existing objects just read the new field as null until it's set:

```diff
  type User {
      name: String
      email: String @unique
      age: u32
+     bio: String
  }
```

```bash
# apply without a restart (or simply restart the server on the edited file)
curl -X POST localhost:4200/admin/reload \
  -H "Authorization: Bearer $RHYPEDB_ADMIN_TOKEN" --data-binary @examples/blog.rhype
```

**Type changes run as an online migration.** Changing a field's *type* can't be inferred from a schema diff — it rewrites every stored value behind a live double-write, backfilling in the background and cutting over with no downtime. You drive it with a named converter (e.g. widening an `i64` field to `f64`):

```bash
rhypedb-cli --admin-token "$RHYPEDB_ADMIN_TOKEN" \
  migrate start --type User --field score --to f64 --converter widen_int_to_f64
rhypedb-cli --admin-token "$RHYPEDB_ADMIN_TOKEN" migrate status   # watch it backfill, then cut over
```

Renames, removals, error policies, pause/resume/cancel, and dry-runs are covered in **[Schema Migrations](docs/src/migrations.md)**.

See [Getting Started](docs/src/getting-started.md) for the full walkthrough.

## Status

Early development. The storage engine (LSM-tree + WAL + MVCC), schema engine, query language, vector search, online migrations, and backup/restore are all implemented and tested across the workspace. APIs may still change. See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal design.

## License

MIT
