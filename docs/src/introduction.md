# rhypedb

**A strongly-typed object database with first-class relationships and native vector search. Written in Rust.**

rhypedb treats objects, relationships, and vectors as core primitives — not afterthoughts bolted onto a relational or document model.

- **Objects, not rows.** You define typed schemas; stored objects conform to their type. No impedance mismatch between your data model and your domain model.
- **Relationships are properties.** You declare links between objects in the schema. The database stores, traverses, and maintains them — and enforces referential integrity. No foreign keys, no junction tables, no `JOIN`s.
- **Vectors are native.** `Vector<N>` is a field type like `String` or `i64`. Vectors are compressed at rest with [TurboQuant](https://github.com/0xSero/turboquant)-adapted quantization, indexed with HNSW, and queried with a `.similar(...)` operator — optionally with server-side text embedding so you never have to compute embeddings yourself.
- **One path-based query language** for lookups, filters, relationship traversal, mutations, and vector similarity.
- **Real-time subscriptions.** Subscribe to query patterns and get semantically meaningful change events.

## Design goals

- **Low memory.** rhypedb is built to run inside Firecracker microVMs. TurboQuant compresses vectors 4–8×, and steady-state RSS stays in single-digit megabytes on the benchmark workloads. Scale-to-zero via VM snapshots.
- **Referential integrity.** No dangling references. Delete policies (`remove`, `cascade`, `deny`) are declared in the schema and enforced atomically.
- **MVCC.** Snapshot isolation with write-write conflict detection. Readers never block writers.
- **Single binary, no external dependencies.** rhypedb is the default database for the [jkbase](https://github.com/joeleaver/jkbase) cloud platform.

## How it fits together

rhypedb ships three binaries:

| Binary | Role |
| --- | --- |
| `rhypedb-server` | The database server. Speaks HTTP (default `127.0.0.1:4200`) and a binary TCP protocol (default `127.0.0.1:4201`). |
| `rhypedb-cli` | Interactive client and operator tool — run queries, drive migrations, take backups, export data. |
| `rhypedb-import` | Offline importer for logical (NDJSON) exports. |

A running server is configured by exactly one thing you write: a **schema file** (`.rhype`). Everything else — types, indexes, vector fields, relationships, delete policies — is declared there.

## Where to go next

- New here? Start with **[Getting Started](getting-started.md)** — define a schema, run the server, and execute your first queries.
- Writing a schema? See the **[Schema Reference](schema.md)**.
- Querying? See the **[Query Language](queries.md)** and **[Vector Search](vectors.md)**.
- Deploying or operating a server? See **[Running rhypedb](operations.md)** and **[Backup & Recovery](backup-recovery.md)**.
- Evolving a live schema? See **[Schema Migrations](migrations.md)**.

> **Status: early development.** The data model, query language, storage engine, vector search, migrations, and backup/restore are all implemented and tested, but APIs may still change. This documentation describes the current behavior of the `master` branch.

For the internal design (storage engine, MVCC, the vector pipeline), see **[Architecture](architecture.md)**.
