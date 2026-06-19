# API Reference

Three ways to talk to a `rhypedb-server`: the **HTTP API**, the **`rhypedb-cli`** tool, and the **binary TCP protocol**. All three execute the same [query language](queries.md).

## HTTP API

Base URL defaults to `http://127.0.0.1:4200`.

### Data plane (open)

#### `POST /query`

Execute a query or mutation.

**Request:**

```json
{ "query": "User.get(1)" }
```

**Responses** (`200 OK`), depending on the query:

| Query kind | Body |
| --- | --- |
| Returns many objects | `{ "objects": [ { ... }, ... ] }` |
| Returns one object | `{ "object": { ... } }` |
| Mutation with no return (update/delete) | `{ "ok": true }` |

**Errors:** `400`/`500` with `{ "error": "message" }`.

```bash
curl -s http://127.0.0.1:4200/query \
  -H 'content-type: application/json' \
  -d '{"query":"User.filter(.age > 18).limit(10)"}'
```

#### `GET /health`

Liveness check. `200 OK` with a short status string.

#### `GET /status`

Operational snapshot.

```json
{
  "subscriptions": 2,
  "vectorizer": { "pending": 0, "indexes": [ { "name": "Post.embedding", "vectors": 12000 } ] }
}
```

### Admin plane (gated by `RHYPEDB_ADMIN_TOKEN`)

All routes below require `Authorization: Bearer <RHYPEDB_ADMIN_TOKEN>`. With the token unset they return `403`; with a wrong token, `401`. See [Running rhypedb](operations.md#authentication).

| Method & path | Purpose |
| --- | --- |
| `POST /admin/migrations` | Start a field-type migration. |
| `GET /admin/migrations` | List migrations (filter with `?status=&type=`). |
| `GET /admin/migrations/{id}` | Migration detail / progress. |
| `POST /admin/migrations/{id}/pause` | Pause. |
| `POST /admin/migrations/{id}/resume` | Resume. |
| `POST /admin/migrations/{id}/cancel` | Cancel (before cutover only). |
| `POST /admin/migrations/{id}/cutover` | Force cutover. |
| `GET /admin/migrations/{id}/quarantine` | List quarantined rows. |
| `POST /admin/migrations/{id}/quarantine/retry` | Retry quarantined rows. |
| `GET /admin/migrations/{id}/events` | Live event stream (Server-Sent Events). |
| `POST /admin/backup` | Write a physical snapshot on the server. |
| `GET /admin/backup/stream` | Download a physical snapshot as a tar. |
| `POST /admin/export` | Write a logical NDJSON export on the server. |
| `GET /admin/export/stream` | Download a logical export (`?types=&vectors=`). |
| `POST /admin/compact` | Force flush + full compaction. |
| `POST /admin/reload` | Hot-reload the schema (SDL in the request body). |

**Start a migration:**

```json
POST /admin/migrations
{
  "type": "User", "field": "score", "to": "f64",
  "converter": "widen_int_to_f64", "converter_version": 1,
  "chunk": 1000, "parallel": 4, "policy": "stop",
  "quarantine_cap": 100000, "dry_run": false
}
→ { "migration_id": 1, "created_at_ms": 1750000000000 }
```

See **[Schema Migrations](migrations.md)** and **[Backup & Recovery](backup-recovery.md)** for the workflows these endpoints serve.

## CLI reference

```
rhypedb-cli [GLOBAL FLAGS] [SUBCOMMAND]
```

### Global flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-H`, `--host <url>` | `http://127.0.0.1:4200` | Server URL. |
| `-e`, `--execute <query>` | — | Run one query and exit. |
| `--admin-token <token>` | `$RHYPEDB_ADMIN_TOKEN` | Token for admin subcommands. |

With no subcommand and no `-e`, the CLI starts an interactive REPL (`quit`/`exit` to leave).

### Subcommands

```
migrate start   --type T --field F --to TYPE --converter NAME
                [--converter-version V] [--chunk N] [--parallel P]
                [--policy stop|skip|quarantine] [--quarantine-cap N] [--dry-run]
migrate status  [ID]
migrate pause | resume | cancel | cutover   ID
migrate quarantine list   ID
migrate quarantine retry  ID --new-converter NAME
migrate events  ID

backup  --dest <server-path> [--label L]
backup  --download <local-dir> [--label L]
restore <snapshot-dir> <data-dir> [--force]
verify  <snapshot-dir>

export  --dest <server-path>  [--label L] [--types A,B] [--vectors raw|none]
export  --download <local-file> [--types A,B] [--vectors raw|none]
verify-export <export-file>
```

`backup`, `export`, and the `migrate` family talk to the admin API and need `--admin-token`. `restore` and `verify`/`verify-export` are offline/local and do not. (Logical *import* is a separate binary — see below.)

## Offline binaries

### `rhypedb-import`

Import a logical export into a fresh data directory (server stopped):

```
rhypedb-import <export-file> --data-dir <path> [--force] [--vectors raw|none]
```

## Binary TCP protocol

For high-throughput clients, the server speaks a length-prefixed binary protocol (default `127.0.0.1:4201`, `--tcp-listen`). It executes the same queries as `POST /query` with less per-request overhead, and adds a bulk vector-ingest message.

### Frame format

```
[ len: u32 BE ] [ req_id: u32 BE ] [ type: u8 ] [ payload: (len - 5) bytes ]
```

`len` counts `req_id` + `type` + `payload`. Responses echo the `req_id`.

### Client message types

| Type | Name | Payload |
| --- | --- | --- |
| `0x01` | Query | `[ q_len: u32 BE ][ utf-8 query ]` |
| `0x02` | Ping | *(empty)* |
| `0x03` | VectorBatch | bulk vector ingest for one `Vector` field |

### Server response types

| Type | Name | Payload |
| --- | --- | --- |
| `0x80` | Objects | `[ count: u32 BE ]` then encoded objects |
| `0x81` | Single | one encoded object |
| `0x82` | Done | *(empty)* |
| `0x83` | Error | `[ msg_len: u32 BE ][ utf-8 message ]` |
| `0x84` | Pong | *(empty)* |

`VectorBatch` (`0x03`) bulk-ingests caller-supplied `f32` vectors for one type's `Vector` field — the path used by `rhypedb-import` and the recommended way to load precomputed vectors at scale.
