# Running rhypedb

This page covers running and operating a `rhypedb-server` in practice: configuration, the data directory, authentication, hot reload, compaction, and monitoring.

## Starting the server

```bash
rhypedb-server --schema schema.rhype --data-dir /var/lib/rhypedb
```

### Command-line flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-s`, `--schema <path>` | *(required)* | Path to the SDL schema file. |
| `-d`, `--data-dir <path>` | `./rhypedb-data` | Storage directory (created if absent). |
| `--listen <addr>` | `127.0.0.1:4200` | HTTP listen address. |
| `--tcp-listen <addr>` | `127.0.0.1:4201` | Binary TCP listen address. |
| `--no-sync` | off | Skip the WAL `fsync` at commit. Faster, but a power loss can drop the last few writes. Equivalent to Postgres `fsync=off`. **Don't use in production.** |

### Environment variables

| Variable | Meaning |
| --- | --- |
| `RHYPEDB_ADMIN_TOKEN` | Bearer token that gates all `/admin/*` endpoints. If unset, admin routes return `403`. |
| `RHYPEDB_EF` | Default HNSW search width (`ef`) for `.similar` queries that omit `ef:` (must be `≥ 1`; overridable per query). An invalid value is ignored with a warning. |
| `RHYPEDB_RERANK` | Default rerank pool size for `.similar` queries that omit `rerank:` (`0` = off; overridable per query). An invalid value is ignored with a warning. |

## The data directory

A data directory is self-contained. After running, `/var/lib/rhypedb` contains:

| Entry | What it is |
| --- | --- |
| `sst/` | LSM-tree sorted-string-table files (the bulk of the data). |
| `wal.log` | Write-ahead log for durability between flushes. |
| `schema.rhype` | The schema (written here by `restore`/`import`; the server otherwise reads `--schema`). |
| `hnsw_<field>.bin` | Persisted HNSW vector index, one per indexed `Vector` field. Rebuilt from the raw vectors if missing or stale. |
| `LOCK` | The single-writer advisory lock (see below). |

### One writer per data directory

A data directory must be opened by **one process at a time**. Two writers on one directory would corrupt the LSM, so on open the server takes an advisory lock (`flock`) on the `LOCK` file, held for the lifetime of the process and released automatically when it exits.

If you try to start a second server — or run an offline tool like `rhypedb-import` or `restore` — against a directory that's already in use, it fails fast:

```
data directory is locked: /var/lib/rhypedb/LOCK is locked (held by pid 1234 on host db-1). Stop the server or other tool using this data directory before opening it.
```

This is a same-host guard. It does **not** make it safe to point two machines at the same shared/network-mounted directory — never do that. (On a network filesystem the lock may not engage at all; in that case rhypedb still detects a second opener at the next write and halts loudly rather than corrupting data, but the supported model is one host, one local data directory.)

## Authentication

The data-plane endpoints (`POST /query`, `GET /status`, `GET /health`, `GET /schema`) are **open** — put them behind your own network boundary or proxy as needed.

The administrative endpoints (`/admin/*` — migrations, backup, export, compaction, reload) are gated by `RHYPEDB_ADMIN_TOKEN`:

- Token **unset** → admin routes return `403 Forbidden`.
- Token set, request missing/mismatched `Authorization: Bearer <token>` → `401 Unauthorized`.
- Token set and matched → request proceeds.

```bash
curl -s -X POST http://127.0.0.1:4200/admin/compact \
  -H "Authorization: Bearer $RHYPEDB_ADMIN_TOKEN"
```

The CLI reads the token from `--admin-token` or the `RHYPEDB_ADMIN_TOKEN` environment variable.

## Hot schema reload

After you change the schema file, you can apply it without restarting the process:

```bash
curl -s -X POST http://127.0.0.1:4200/admin/reload \
  -H "Authorization: Bearer $RHYPEDB_ADMIN_TOKEN" \
  --data-binary @schema.rhype
```

The server rebuilds its in-memory handle on the same underlying storage and swaps it in atomically — in-flight queries are not interrupted. Reload is refused while an online field-type migration is armed (the migration owns the schema until it finishes). After a `change_field_type` migration cuts over, the server reloads automatically. See **[Schema Migrations](migrations.md)**.

## Compaction

rhypedb compacts its LSM automatically in the background. To force a flush + full compaction on demand (an expensive, mutating operation — hence admin-gated):

```bash
curl -s -X POST http://127.0.0.1:4200/admin/compact \
  -H "Authorization: Bearer $RHYPEDB_ADMIN_TOKEN"
```

```json
{ "flush_ok": true, "flush_ms": 50, "compact_ok": true, "compact_ms": 200 }
```

## Monitoring

- **`GET /health`** — liveness. Returns `200 OK` with a short status string.
- **`GET /status`** — operational snapshot: active subscriptions, pending embeddings, and per-index vector counts.
- **`GET /schema`** — live schema introspection (JSON + canonical SDL) for tooling and typed-client codegen. See the [API reference](api-reference.md#get-schema).

```bash
curl -s http://127.0.0.1:4200/status
```

```json
{
  "subscriptions": 2,
  "vectorizer": { "pending": 0, "indexes": [ { "name": "Post.embedding", "vectors": 12000 } ] }
}
```

Neither requires authentication, and `/status` exposes only operational metrics (no data), so it is safe to scrape.

## Backups

See **[Backup & Recovery](backup-recovery.md)** for taking consistent online backups (physical snapshots and portable logical exports) and restoring them.
