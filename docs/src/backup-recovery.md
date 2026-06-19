# Backup & Recovery

rhypedb offers two complementary ways to get your data out and back in:

| | Physical backup | Logical export |
| --- | --- | --- |
| **What it is** | A byte-level snapshot of the storage files. | A portable NDJSON dump of objects, relationships, and vectors. |
| **Format** | LSM SSTs + WAL + indexes. | `rhypedb-logical-export-v1` text stream. |
| **Speed** | Very fast (hard-links files). | Slower (serializes every object). |
| **Portability** | Same storage format/version. | Version-independent; survives format changes. |
| **Use it for** | Fast local snapshots, point-in-time recovery, VM scale-to-zero. | Migrating between versions, moving data, archival, inspection. |

Both are taken **online** from a running server against a single consistent snapshot, and both are gated by `RHYPEDB_ADMIN_TOKEN`.

## Physical backup

A physical backup flushes the memtable, hard-links the immutable SST files into the destination, copies the WAL and vector indexes, and writes a `MANIFEST.json` last as a completeness marker.

### Take one

Write it on the server's filesystem:

```bash
rhypedb-cli --admin-token "$TOKEN" backup --dest /backups --label nightly
```

Or stream it to your local machine (downloaded as a tar and unpacked into a directory):

```bash
rhypedb-cli --admin-token "$TOKEN" backup --download ./snap-2026-06-19
```

The equivalent HTTP endpoints are `POST /admin/backup` and `GET /admin/backup/stream`.

### Verify one

```bash
rhypedb-cli verify ./snap-2026-06-19
```

This checks that `MANIFEST.json` is present and every file it lists exists.

### Restore one

Restore is an **offline** operation — the target server must be stopped (the data-dir lock will refuse it otherwise).

```bash
rhypedb-cli restore ./snap-2026-06-19 /var/lib/rhypedb --force
```

`restore`:

- validates the snapshot's manifest and completeness before touching the target,
- clears any stale `sst/`, `wal.log`, and `hnsw_*.bin` from the target first (so you never mix two databases),
- copies the snapshot in, and
- prints the command to start the server on the restored directory.

`--force` is required to overwrite a non-empty target directory.

## Logical export

A logical export is a single self-describing NDJSON stream — header, schema, objects, relationships, vectors, trailer — read at one consistent snapshot. The trailer (with per-type counts and a `complete` marker) is written last, so a truncated dump is detectable. Because it carries the schema and typed values rather than raw storage bytes, it imports into any version of rhypedb.

Values are encoded losslessly: 64-bit integers as decimal strings (to survive JSON), floats as the exact IEEE-754 bit pattern, bytes as base64, and **raw `f32` vectors** verbatim. The HNSW index is *not* exported — it is rebuilt from the raw vectors on import.

### Take one

```bash
# write on the server
rhypedb-cli --admin-token "$TOKEN" export --dest /exports --label v2

# or stream to a local file
rhypedb-cli --admin-token "$TOKEN" export --download ./dump.ndjson
```

Options:

| Flag | Meaning |
| --- | --- |
| `--types A,B` | Export only these types (relationships to excluded types are dropped and counted). |
| `--vectors raw\|none\|reembed` | `raw` (default) ships the f32 vectors; `none` omits them; `reembed` omits `@vectorize` vectors so the importer regenerates them from source text. |

The equivalent HTTP endpoints are `POST /admin/export` and `GET /admin/export/stream`. An export is refused while a field-type migration is in progress.

### Verify one

```bash
rhypedb-cli verify-export ./dump.ndjson
```

This validates the header, the trailer, and the per-type counts.

## Logical import

Importing a logical export is done by the dedicated **`rhypedb-import`** binary, offline (server stopped):

```bash
rhypedb-import ./dump.ndjson --data-dir /var/lib/rhypedb
```

It:

- validates the file (header, trailer, counts) and parses the embedded schema **before** touching anything,
- builds the entire import in a fresh staging directory and **atomically swaps** it into place only on full success — a failed import never wipes or corrupts the target,
- preserves object ids exactly (relationships and vectors reference ids by value, so no remapping is needed), and
- writes `schema.rhype` and tells you how to start the server.

```
imported 3 type(s): 1000 objects, 500 edges, 100 vectors
start the server with:
  rhypedb-server --schema /var/lib/rhypedb/schema.rhype --data-dir /var/lib/rhypedb
(the HNSW vector index rebuilds from the imported vectors on first start)
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--data-dir <path>` | Target directory (must be empty unless `--force`). |
| `--force` | Overwrite a non-empty target. |
| `--vectors raw\|none\|reembed` | How to handle vectors (`raw` = default). `reembed` re-derives `@vectorize` vectors from source text by loading the embedding model — it fails cleanly if the model is unavailable, and is lossy/non-deterministic vs. the originals. |

## Which should I use?

- **Fast local snapshot, same version, point-in-time recovery, or VM snapshotting** → physical backup/restore.
- **Moving data between rhypedb versions, archiving, or inspecting the contents** → logical export/import.

> All restore and import operations are offline by design and protected by the [single-writer data-dir lock](operations.md#one-writer-per-data-directory): they refuse to run against a directory a live server still holds.
