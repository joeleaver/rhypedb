# Schema Migrations

Changing the schema of a database that already holds data is a real operation with rules. rhypedb sorts schema changes into four kinds, by how much work and risk each carries:

| Change | How it happens | Cost |
| --- | --- | --- |
| **Add** a field or type | Implicit — update the schema and reload | Free, instant |
| **Remove** a field or type | Implicit, but **opt-in and one-way** | Instant (data tombstoned) |
| **Rename** a field or type | Explicit migration verb | Instant (catalog only) |
| **Change a field's type** | Explicit **online migration** with a converter | Proportional to row count |

The reason renames and type changes need explicit operations is that they cannot be inferred safely from a schema diff: a rename looks identical to "drop the old field, add a new one," and a type change requires rewriting every stored value. Adds and removes have no such ambiguity.

## Adding fields and types

Additive changes are the easy case. Add the field or type to your schema file and apply the new schema — existing objects simply don't have the new field (it reads back as absent/null until set), and existing ids are never renumbered.

To apply a new schema to a running server without a restart, `POST /admin/reload` with the new SDL (see [Running rhypedb](operations.md)); otherwise restart the server pointing at the updated schema file.

Fields are not "required," so adding one to a populated type is always safe.

## Removing fields and types

Removal is also driven by the schema diff, but it is **refused by default** and is **one-way**. Retiring a field or type tombstones it (the catalog entry is marked retired; the bytes in existing object blobs age out as objects are rewritten) and a retired name cannot simply be re-added. Because it is destructive, it requires an explicit opt-in at the engine level (`allow_schema_shrink`) rather than happening silently on reload.

## Renames

A rename preserves the underlying field/type identity and all its data and indexes — only the name in the catalog changes, in a single atomic step. Renames are expressed as migration verbs (`rename_type`, `rename_field`) through the engine's migration API.

## Changing a field's type (online migration)

This is the substantial one: changing a field's encoding — say `i64` → `f64` — across a table that may hold millions of rows, while the server keeps serving reads and writes. It runs as a driven, multi-phase **online migration** and is fully exposed over the admin API and the CLI.

### The pieces

- A **converter**: a named, versioned function that turns an old value into the new type. The server ships a set of **built-in converters** (it prints the available list in its startup banner — e.g. `widen_int_to_f64`). User-supplied/network-supplied converters are not yet available; you migrate using a built-in converter that matches your change.
- A **target type**: the new scalar type for the field.
- Tuning: chunk size, parallelism, an error policy, and an optional dry run.

### Running one with the CLI

```bash
rhypedb-cli --admin-token "$RHYPEDB_ADMIN_TOKEN" \
  migrate start --type User --field score --to f64 \
  --converter widen_int_to_f64 --chunk 1000 --parallel 4 --policy stop
```

This returns a migration id. Watch its progress:

```bash
rhypedb-cli migrate status 1            # list all, or detail for id 1
rhypedb-cli migrate events 1            # live event stream until it settles
```

And control it:

```bash
rhypedb-cli migrate pause 1
rhypedb-cli migrate resume 1
rhypedb-cli migrate cancel 1            # only before cutover
rhypedb-cli migrate cutover 1           # force cutover for a parked-but-done plan
```

`migrate start` flags:

| Flag | Meaning |
| --- | --- |
| `--type`, `--field` | the field to migrate |
| `--to` | target scalar type (`f64`, `i64`, `String`, …) |
| `--converter`, `--converter-version` | the named converter and its version |
| `--chunk` | objects converted per commit (`0` = default) |
| `--parallel` | number of partition workers |
| `--policy` | `stop` (default), `skip`, or `quarantine` — what to do on a converter error |
| `--quarantine-cap` | max quarantined rows before failing (with `quarantine`) |
| `--dry-run` | run a full preflight that writes nothing |

The same operations exist as admin HTTP endpoints (`POST /admin/migrations`, `GET /admin/migrations/{id}`, `.../pause|resume|cancel|cutover`, `.../events`); see the [API Reference](api-reference.md).

### What happens under the hood

1. **Arm.** A double-write hook is installed so every live write to the field is also carried forward into the new encoding while the migration runs.
2. **Backfill.** Worker threads convert existing rows in chunks, committing durably and recording per-partition cursors (so the work is crash-resumable).
3. **Cutover.** Once every row is converted, the new encoding is promoted to the live field, indexes are reconciled, and the hook is disarmed. **This is the point of no return** — cancel is refused from here on.
4. **Done.** The migration settles as completed.

### Three things to know

- **Cancel is lossless before cutover, impossible after.** Cancelling rolls back the in-progress conversion (the source field was never overwritten during backfill). Once cutover starts promoting the converted values, there's nothing to roll back to.
- **It survives restarts.** If the server stops mid-migration, it auto-resumes on startup from the durable per-partition cursors. After a `change_field_type` cutover completes, the server hot-reloads its in-memory handle automatically so it serves the new type without a restart.
- **Pick an error policy up front.** `stop` (default) fails the whole migration on the first converter error. `skip` counts and skips bad rows. `quarantine` sets bad rows aside so you can inspect and retry them (`migrate quarantine list <id>` / `migrate quarantine retry <id>`) before cutover proceeds. `--dry-run` does a full conversion pass that writes nothing, so you can validate before committing.
