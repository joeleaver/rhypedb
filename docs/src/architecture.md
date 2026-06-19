# Architecture

This documentation covers how to *use* rhypedb. For how it *works* inside — the LSM-tree storage engine, the write-ahead log, MVCC snapshot isolation, the TurboQuant/HNSW vector pipeline, the wire protocol, and the crate layout — see the architecture document in the repository:

**→ [ARCHITECTURE.md](https://github.com/joeleaver/rhypedb/blob/master/ARCHITECTURE.md)**

A short orientation to the workspace crates:

| Crate | Role |
| --- | --- |
| `rhypedb-storage` | LSM-tree, WAL, memtable, SSTs, MVCC, the single-writer data-dir lock. |
| `rhypedb-schema` | The schema DSL parser and type system. |
| `rhypedb-engine` | The database engine: catalog, object model, migrations, the vectorizer queue. |
| `rhypedb-vector` | HNSW indexing, TurboQuant quantization, distance metrics. |
| `rhypedb-query` | Query language AST, parser, and executor. |
| `rhypedb-embed` | Text embedding models (via ONNX Runtime) for `@vectorize`. |
| `rhypedb-subscribe` | Change events and subscription filters. |
| `rhypedb-server` | The server binary: HTTP + TCP, admin API. Also ships `rhypedb-import`. |
| `rhypedb-cli` | The client and operator tool. |
