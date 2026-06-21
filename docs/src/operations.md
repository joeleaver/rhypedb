# Running rhypedb

This page covers running and operating a `rhypedb-server` in practice: configuration, the data directory, authentication, hot reload, compaction, and monitoring.

## Starting the server

```bash
rhypedb-server --schema schema.rhype --data-dir /var/lib/rhypedb
```

### Command-line flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-s`, `--schema <path>` | *(required, except with `--restore-from`)* | Path to the SDL schema file. Optional when restoring (the snapshot carries its own authoritative schema). |
| `-d`, `--data-dir <path>` | `./rhypedb-data` | Storage directory (created if absent). |
| `--restore-from <dir>` | off | Restore a physical backup snapshot into `--data-dir` **before** serving (see [Restore on boot](#restore-on-boot)). Idempotent. |
| `--restore-force` | off | Allow `--restore-from` to overwrite an existing, *different* database in `--data-dir`. |
| `--listen <addr>` | `127.0.0.1:4200` | HTTP listen address. |
| `--tcp-listen <addr>` | `127.0.0.1:4201` | Binary TCP listen address. |
| `--no-sync` | off | Skip the WAL `fsync` at commit. Faster, but a power loss can drop the last few writes. Equivalent to Postgres `fsync=off`. **Don't use in production.** |

### Environment variables

| Variable | Meaning |
| --- | --- |
| `RHYPEDB_ADMIN_TOKEN` | Bearer token that gates all `/admin/*` endpoints. If unset, admin routes return `403`. |
| `RHYPEDB_EF` | Default HNSW search width (`ef`) for `.similar` queries that omit `ef:` (must be `≥ 1`; overridable per query). An invalid value is ignored with a warning. |
| `RHYPEDB_RERANK` | Default rerank pool size for `.similar` queries that omit `rerank:` (`0` = off; overridable per query). An invalid value is ignored with a warning. |
| `RHYPEDB_RESTORE_FROM` | Snapshot directory to restore on boot (same as `--restore-from`; the flag wins if both are set). |
| `RHYPEDB_RESTORE_FROM_FORCE` | `1`/`true`/`yes`/`on` ⇒ same as `--restore-force`. |
| `RHYPEDB_CONFIG` | Path to a TOML config file (same as `--config`; the flag wins if both are set). |
| `RHYPEDB_BLOCK_COMPRESSION` | `none` (default) or `lz4` — per-block SST compression for new files (same as `--block-compression`). |

## Configuration file

Instead of (or alongside) flags and env vars, point the server at a TOML config
file with `--config <path>` or `RHYPEDB_CONFIG`. There is no auto-discovery — the
file is loaded only from the explicit path.

**Precedence (most-specific wins):**

```text
CLI flag  >  env var  >  config file  >  built-in default
```

So a config file is a *fallback layer*: anything you also pass as a flag or env var
overrides it, and with no `--config` the server behaves exactly as before.

The keys map one-to-one to the flags/env vars (flat, `snake_case`):

| Key | Type | Default | Equivalent flag / env |
| --- | --- | --- | --- |
| `schema` | path | *(required unless `restore_from`)* | `--schema` |
| `data_dir` | path | `./rhypedb-data` | `--data-dir` |
| `listen` | string | `127.0.0.1:4200` | `--listen` |
| `tcp_listen` | string | `127.0.0.1:4201` | `--tcp-listen` |
| `no_sync` | bool | `false` | `--no-sync` |
| `admin_token` | string | *(unset → admin off)* | `RHYPEDB_ADMIN_TOKEN` |
| `ef` | int ≥ 1 | *(heuristic)* | `RHYPEDB_EF` |
| `rerank` | int (`0` = off) | `off` | `RHYPEDB_RERANK` |
| `restore_from` | path | *(off)* | `--restore-from` |
| `restore_from_force` | bool | `false` | `--restore-force` |
| `cache_max_entries` | int ≥ 1 | `256` | *(file only)* |
| `graceful_drain_secs` | int ≥ 1 | `20` | *(file only)* |
| `worker_quiesce_budget_secs` | int ≥ 1 | `10` | *(file only)* |
| `block_compression` | `"none"` \| `"lz4"` | `"none"` | `--block-compression` / `RHYPEDB_BLOCK_COMPRESSION` |

**SST block compression.** `block_compression` controls how newly written SST
files (memtable flushes and compactions) store their data region. `"none"` (the
default) keeps the uncompressed layout, whose reads are zero-copy views into the
memory-mapped file. `"lz4"` compresses each ~16-entry block with LZ4 — roughly
3.8× smaller files, so more of the working set fits in the OS page cache — but it
is **opt-in** because a benchmark showed it costs ~3.7× on a 1M-row multi-hop
graph traversal: each scattered cover-blob read decompresses a whole block to
pull a single entry, which doesn't amortize the way a sequential scan does. Turn
`"lz4"` on where **disk size / cold-cache density matters and reads are
scan-heavy or rare** (range/filter scans amortize the block decompress well);
keep `"none"` for point-lookup- and traversal-heavy workloads. **Both formats are
always readable regardless of this setting** — it only affects new writes — so a
data directory can hold a mix, and changing the value lets natural compaction
migrate files to the new format over time (no offline rewrite needed). An
unrecognized value is warned about and ignored (falls back to the lower layer /
the default).

**Validation.** A typo'd or unknown key, invalid TOML, a wrong-typed value (e.g.
`ef = "x"`), or an unreadable `--config` path is a **fatal startup error** (exit 1)
that names the file. Out-of-range *tuning* values (`ef`/`rerank`/`cache`/drain
below their minimum) are **warned about and ignored** — they fall back to the lower
layer or the default, so a fat-fingered tuning hint never bricks an unattended
start. The boolean flags (`--no-sync`, `--restore-force`) can only force a value
*on*; to keep one off, leave it unset everywhere.

Relative paths (`schema`, `data_dir`, `restore_from`) resolve against the server's
**current working directory**, not the config file's location — prefer absolute
paths in a config file. The effective resolved config is logged at startup (the
`admin_token` is **never** printed — only whether admin is enabled). Keep any
config file containing `admin_token` out of version control. A commented example with every key and its
default ships at [`docs/examples/rhypedb.toml`](https://github.com/joeleaver/rhypedb/blob/master/docs/examples/rhypedb.toml).

## Restore on boot

For a managed deployment (where the platform controls the instance lifecycle and
fetches backups from object storage), the server can restore a [physical
backup](backup-recovery.md) snapshot into its `--data-dir` **at startup, before it
opens the database or binds a listener**, then serve normally:

```sh
rhypedb-server --restore-from /restore/snapshot --data-dir /var/lib/rhypedb
```

The typical scale-to-zero wake flow is: the platform stops the instance (a clean
`SIGTERM` flushes the memtable and saves the HNSW snapshots), fetches a backup from
object storage to a local path, and starts the server pointed at it.

What it does, in order: validate the snapshot's `MANIFEST.json` (a typo'd or
incomplete source is refused **before** anything is cleared) → take the
single-writer data-dir lock → clear stale LSM data → copy the SSTs, WAL,
`schema.rhype`, and the `hnsw_*.bin` index snapshots → open and serve.

Key points:

- **The snapshot's schema wins.** A restored data dir carries its own
  `schema.rhype`, which is authoritative (the SSTs were written for it). `--schema`
  is therefore optional under `--restore-from`; if you pass it, it must match the
  snapshot's schema exactly or startup fails.
- **The HNSW snapshots travel with the backup.** Restoring the `hnsw_*.bin` files
  lets a vector index wake in ~hundreds of ms instead of rebuilding from the LSM
  (which is multi-second per 100k vectors — see the cold-start benchmark). A
  missing index snapshot is a warning, not an error (it rebuilds on open).
- **Idempotent.** `--restore-from` records the snapshot identity, so a restart with
  the same snapshot already in place is a no-op — you can leave
  `RHYPEDB_RESTORE_FROM` set across restarts without re-clobbering live data.
- **Overwrite protection.** Restoring over an existing, *different* database
  requires `--restore-force` (or `RHYPEDB_RESTORE_FROM_FORCE=1`). A stale `LOCK`
  file from a crashed instance does **not** count as existing data.
- **Same host only.** Like all of rhypedb, the single-writer lock is same-host; the
  managed platform must guarantee the previous instance is stopped first.

This is the in-server counterpart to the offline `rhypedb-cli restore`, which
remains available for manual use.

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
