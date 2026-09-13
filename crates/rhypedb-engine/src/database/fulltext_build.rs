//! Background full-text index builder (a child module of `database` so it can
//! reach the engine's private write-path helpers).
//!
//! Computed at every open / reload from the persisted build markers
//! (`crate::fulltext::build`): a list of [`FulltextTask`]s — backfills for
//! fields whose marker is `Building`, sweeps of stale generations after an
//! analyzer/positions change, and full drops for markers whose field lost the
//! directive — run sequentially on one detached thread.
//!
//! ## Convergence with live writers
//!
//! The write paths maintain the CURRENT generation's rows for every object
//! they touch, always. The backfill visits objects in id order at a pinned
//! snapshot per chunk and indexes only those WITHOUT an `l:` row, committing
//! `[postings + l: rows …, marker LAST]` so a torn tail loses at most the
//! cursor advance. A foreground write to an object inside the chunk window
//! also writes that object's `l:` row, so the two commits conflict on it and
//! the loser retries: the backfill re-reads (now indexed → skipped); a
//! foreground loser surfaces `WriteConflict` to its caller exactly like the
//! field-type migration's backfill does today. Nothing can be indexed twice
//! with different content, and nothing is left un-indexed. The backfill never
//! gives up to foreground traffic: every conflict halves its chunk (down to
//! one object), and a one-object chunk can only conflict with a write that
//! just indexed that very object — which the retry then skips — so progress
//! is guaranteed under any write load.
//!
//! The thread holds only a `Weak<Database>` between chunks (upgraded per
//! chunk, like the migration driver), so dropping the last external `Arc`
//! stops it at the next chunk boundary; `Database::drop` signals + joins.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bytes::Bytes;
use rhypedb_schema::{Schema, SchemaError};
use rhypedb_storage::crash_inject;
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::LsmTree;

use super::Database;
use crate::error::{EngineError, EngineResult};
use crate::fulltext::build::{BuildMarker, BuildProgress, BuildState, FulltextIndexStatus};
use crate::fulltext::{Analyzer, FulltextField, FulltextStats, StatsDelta, tokenize_for_index};
use crate::object::Value;
use std::collections::HashMap;

/// Objects visited per backfill chunk (one commit each).
pub(super) const FULLTEXT_BUILD_CHUNK: usize = 512;
/// Rows deleted per drop-sweep commit.
const FULLTEXT_DROP_CHUNK: usize = 2048;
const WRITE_CONFLICT_RETRIES: u32 = 8;

/// One unit of background work.
#[derive(Debug, Clone)]
pub(super) enum FulltextTask {
    /// Backfill `field` of `type_name` from its marker's cursor.
    Build {
        type_name: String,
        type_id: u64,
        field: FulltextField,
    },
    /// Delete every `f:`/`l:` row of `(type_id, field_id)` whose generation
    /// is not `keep` (`None` = every row), then clear the marker's stale flag
    /// (`keep = Some`) or delete the marker (`keep = None`).
    Drop {
        type_id: u64,
        field_id: u64,
        keep: Option<u32>,
    },
}

/// Shared handle between a `Database` and its builder thread.
#[derive(Debug)]
pub(super) struct FulltextBuilder {
    pub(super) tasks: parking_lot::Mutex<Vec<FulltextTask>>,
    /// Tasks not yet completed (a stopped build stays pending for the next open).
    pub(super) pending: AtomicUsize,
    pub(super) stop: AtomicBool,
    pub(super) handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl FulltextBuilder {
    pub(super) fn new(tasks: Vec<FulltextTask>) -> Self {
        Self {
            pending: AtomicUsize::new(tasks.len()),
            tasks: parking_lot::Mutex::new(tasks),
            stop: AtomicBool::new(false),
            handle: parking_lot::Mutex::new(None),
        }
    }
}

/// What `plan_fulltext_open` hands back to `rebuild_with_arc_storage`.
pub(super) struct FulltextOpenPlan {
    pub(super) fields: HashMap<String, Vec<FulltextField>>,
    pub(super) stats: HashMap<(u64, u64), Arc<FulltextStats>>,
    pub(super) tasks: Vec<FulltextTask>,
}

fn read_marker(storage: &LsmTree, snapshot: u64, type_id: u64, field_id: u64) -> EngineResult<Option<BuildMarker>> {
    Ok(storage
        .get_at(snapshot, &KeyBuilder::catalog_fulltext_marker(type_id, field_id))?
        .and_then(|v| BuildMarker::decode(&v)))
}

/// Commit `writes` in one txn, retrying a bounded number of `WriteConflict`s.
fn commit_writes(storage: &LsmTree, puts: &[(Bytes, Bytes)], deletes: &[Bytes]) -> EngineResult<()> {
    let mut attempts = 0;
    loop {
        let mut txn = storage.begin_txn();
        if !puts.is_empty() {
            storage.put_batch(&mut txn, puts)?;
        }
        if !deletes.is_empty() {
            storage.delete_batch(&mut txn, deletes)?;
        }
        match storage.commit(&mut txn) {
            Ok(_) => return Ok(()),
            Err(rhypedb_storage::Error::WriteConflict) if attempts < WRITE_CONFLICT_RETRIES => {
                storage.abort(&mut txn);
                attempts += 1;
            }
            Err(e) => {
                storage.abort(&mut txn);
                return Err(match e {
                    rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                    other => EngineError::Storage(other),
                });
            }
        }
    }
}

/// Reconcile every `@fulltext` field's build marker against the schema and
/// derive the write-path tables + the background task list. Runs on every
/// open and reload, BEFORE the `Database` exists.
///
/// Decisions per field:
/// * no marker (fresh directive, or a torn-write wipe) → generation 0,
///   `Building` from cursor 0 — or `Built` immediately when the type has no
///   objects at all;
/// * marker with the same analyzer + positions → resume as recorded;
/// * marker with a different identity → generation + 1, `Building`, the old
///   generation flagged stale (swept in the background).
///
/// Markers whose `(type_id, field_id)` is no longer a `@fulltext` field in
/// the schema (directive removed, field/type dropped) become `Dropping`.
pub(super) fn plan_fulltext_open(
    storage: &LsmTree,
    schema: &Schema,
    type_ids: &HashMap<String, u64>,
    field_ids: &HashMap<String, u64>,
) -> EngineResult<FulltextOpenPlan> {
    let snapshot = storage.read_snapshot();
    let mut existing: HashMap<(u64, u64), BuildMarker> = HashMap::new();
    let mut malformed: Vec<Bytes> = Vec::new();
    for (key, value) in storage.scan_prefix_at(snapshot, &KeyBuilder::catalog_fulltext_marker_prefix())? {
        match (KeyBuilder::catalog_fulltext_marker_ids(&key), BuildMarker::decode(&value)) {
            (Some(ids), Some(m)) => {
                existing.insert(ids, m);
            }
            // A malformed marker is dropped and treated as absent: the rows
            // it described are re-derivable and the backfill is idempotent.
            _ => malformed.push(key),
        }
    }

    let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
    let mut builds: Vec<FulltextTask> = Vec::new();
    let mut drops: Vec<FulltextTask> = Vec::new();
    let mut fields: HashMap<String, Vec<FulltextField>> = HashMap::new();
    let mut stats: HashMap<(u64, u64), Arc<FulltextStats>> = HashMap::new();
    let mut live: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();

    let mut type_names: Vec<&String> = schema.types.keys().collect();
    type_names.sort();
    for type_name in type_names {
        let type_def = &schema.types[type_name];
        let type_id = type_ids[type_name];
        let mut list = Vec::new();
        for field in &type_def.fields {
            let Some(ft) = field.fulltext() else {
                continue;
            };
            let analyzer = Analyzer::from_name(&ft.analyzer).ok_or_else(|| {
                EngineError::Schema(SchemaError::Validation(format!(
                    "@fulltext on '{type_name}.{}': analyzer {:?} is not known to this engine build",
                    field.name, ft.analyzer
                )))
            })?;
            let field_id = field_ids[&format!("{type_name}.{}", field.name)];
            live.insert((type_id, field_id));

            let prior = existing.remove(&(type_id, field_id));
            let (marker, changed) = match prior {
                Some(m) if m.matches_config(&ft.analyzer, ft.positions) && m.state != BuildState::Dropping => {
                    (m, false)
                }
                Some(m) => (
                    // Identity changed (or the field was mid-drop and came
                    // back): a new generation, old rows flagged stale.
                    BuildMarker {
                        state: BuildState::Building,
                        generation: m.generation.wrapping_add(1),
                        cursor: 0,
                        positions: ft.positions,
                        stale_generations: true,
                        analyzer: ft.analyzer.clone(),
                    },
                    true,
                ),
                None => (
                    BuildMarker {
                        state: BuildState::Building,
                        generation: 0,
                        cursor: 0,
                        positions: ft.positions,
                        stale_generations: false,
                        analyzer: ft.analyzer.clone(),
                    },
                    true,
                ),
            };
            let mut marker = marker;
            // A type with no objects at all needs no backfill: mark Built now
            // so `.matches` works immediately on a fresh database. (Provable
            // emptiness only — `high_water == None` means no key, live or
            // tombstoned, exists under the prefix.)
            let mut changed = changed;
            if marker.state == BuildState::Building && marker.cursor == 0 {
                let prefix = KeyBuilder::object_prefix(type_id);
                let probe = storage.scan_chunk_raw(snapshot, &prefix, &prefix, 1)?;
                if probe.high_water.is_none() {
                    marker.state = BuildState::Built;
                    changed = true;
                }
            }
            if changed {
                puts.push((KeyBuilder::catalog_fulltext_marker(type_id, field_id), marker.encode()));
            }

            // Corpus stats for the CURRENT generation (see `crate::fulltext`).
            let mut doc_count = 0u64;
            let mut total_tokens = 0u64;
            for (doc_key, value) in storage.scan_prefix_at(
                snapshot,
                &KeyBuilder::fulltext_doc_prefix(type_id, field_id, marker.generation),
            )? {
                doc_count += 1;
                match crate::fulltext::decode_doc_len(&value) {
                    Ok(len) => total_tokens += len as u64,
                    // Derived state: count the doc at length 0 and keep
                    // opening rather than take the database offline; the
                    // posting rows still fail the SEARCH that reads them.
                    Err(e) => eprintln!(
                        "warning: full-text doc-length row for {type_name}.{} object {} is corrupt \
                         ({e}); counting it at length 0 — remove and re-add @fulltext to rebuild",
                        field.name,
                        KeyBuilder::fulltext_object_id(&doc_key).unwrap_or(0),
                    ),
                }
            }
            let field_stats = Arc::new(FulltextStats::new(doc_count, total_tokens));
            stats.insert((type_id, field_id), Arc::clone(&field_stats));
            let ff = FulltextField {
                name: field.name.clone(),
                field_id,
                generation: marker.generation,
                analyzer,
                positions: ft.positions,
                stats: field_stats,
                progress: Arc::new(BuildProgress::new(marker.state)),
            };
            if marker.state == BuildState::Building {
                builds.push(FulltextTask::Build {
                    type_name: type_name.clone(),
                    type_id,
                    field: ff.clone(),
                });
            }
            if marker.stale_generations {
                drops.push(FulltextTask::Drop {
                    type_id,
                    field_id,
                    keep: Some(marker.generation),
                });
            }
            list.push(ff);
        }
        if !list.is_empty() {
            fields.insert(type_name.clone(), list);
        }
    }

    // Orphaned markers: the field lost its directive (or was dropped with
    // its type). Flag Dropping (durably) and sweep every generation.
    let mut orphans: Vec<((u64, u64), BuildMarker)> = existing.into_iter().collect();
    orphans.sort_by_key(|(ids, _)| *ids);
    for ((type_id, field_id), mut m) in orphans {
        debug_assert!(!live.contains(&(type_id, field_id)));
        if m.state != BuildState::Dropping {
            m.state = BuildState::Dropping;
            puts.push((KeyBuilder::catalog_fulltext_marker(type_id, field_id), m.encode()));
        }
        drops.push(FulltextTask::Drop {
            type_id,
            field_id,
            keep: None,
        });
    }

    if !puts.is_empty() || !malformed.is_empty() {
        commit_writes(storage, &puts, &malformed)?;
    }

    let mut tasks = builds;
    tasks.extend(drops);
    Ok(FulltextOpenPlan {
        fields,
        stats,
        tasks,
    })
}

/// Thread body: run every task in order; stop between chunks when asked.
pub(super) fn fulltext_builder_main(weak: std::sync::Weak<Database>, builder: Arc<FulltextBuilder>) {
    let tasks = std::mem::take(&mut *builder.tasks.lock());
    for task in tasks {
        if builder.stop.load(Ordering::Acquire) {
            return;
        }
        let outcome = match &task {
            FulltextTask::Build {
                type_name,
                type_id,
                field,
            } => run_build(&weak, &builder, type_name, *type_id, field),
            FulltextTask::Drop {
                type_id,
                field_id,
                keep,
            } => run_drop(&weak, &builder, *type_id, *field_id, *keep),
        };
        match outcome {
            Ok(true) => {
                builder.pending.fetch_sub(1, Ordering::AcqRel);
            }
            // Stopped (or the database went away): leave the remaining tasks
            // pending; the next open re-derives and resumes them.
            Ok(false) => return,
            Err(e) => {
                // Durable state is untouched by a failed chunk (its txn was
                // aborted); the marker stays Building/Dropping so `.matches`
                // keeps refusing with progress and the next open retries.
                eprintln!("full-text builder: {task:?} failed: {e}");
            }
        }
    }
}

/// Backfill one field. `Ok(true)` = finished (marker Built), `Ok(false)` =
/// stopped early (marker left Building at the last committed cursor).
fn run_build(
    weak: &std::sync::Weak<Database>,
    builder: &FulltextBuilder,
    type_name: &str,
    type_id: u64,
    field: &FulltextField,
) -> EngineResult<bool> {
    let marker_key = KeyBuilder::catalog_fulltext_marker(type_id, field.field_id);
    let (mut cursor, stale, total) = {
        let Some(db) = weak.upgrade() else {
            return Ok(false);
        };
        let snapshot = db.storage.read_snapshot();
        let marker = read_marker(&db.storage, snapshot, type_id, field.field_id)?;
        // A marker that already reads Built (raced by a reload that finished
        // the job) is done.
        if marker.as_ref().is_some_and(|m| m.state == BuildState::Built) {
            field.progress.set_state(BuildState::Built);
            return Ok(true);
        }
        let cursor = marker.as_ref().map_or(0, |m| m.cursor);
        let stale = marker.as_ref().is_some_and(|m| m.stale_generations);
        let total = db
            .storage
            .count_prefix_at(snapshot, &KeyBuilder::object_prefix(type_id))?;
        (cursor, stale, total)
    };
    field.progress.set_total(total);

    // Adaptive chunk: halved on every conflict, restored after a clean commit.
    let mut chunk_size = FULLTEXT_BUILD_CHUNK;
    loop {
        if builder.stop.load(Ordering::Acquire) {
            return Ok(false);
        }
        let Some(db) = weak.upgrade() else {
            return Ok(false);
        };
        // Never race a cutover / rename / reload (they hold the write side).
        let _migration_guard = db.migration_lock.read();
        let (visited, next_cursor, done) = loop {
            if builder.stop.load(Ordering::Acquire) {
                return Ok(false);
            }
            let snapshot = db.storage.read_snapshot();
            let chunk = db.scan_chunk(type_name, snapshot, cursor, chunk_size)?;
            let mut txn = db.storage.begin_txn();
            let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
            let mut delta = StatsDelta::default();
            for obj in &chunk.objects {
                let Some(Value::String(text)) = obj.fields.get(&field.name) else {
                    continue;
                };
                // Already indexed (a live write got there first, or a prior
                // pass of this chunk committed before a torn tail).
                let doc_key = KeyBuilder::fulltext_doc(type_id, field.field_id, field.generation, obj.id);
                if db.storage.get(&txn, &doc_key)?.is_some() {
                    continue;
                }
                let doc = tokenize_for_index(field, text);
                crate::fulltext::stage_doc_puts(type_id, field, obj.id, &doc, &mut puts);
                delta.add(type_id, field.field_id, 1, doc.doc_len as i64);
            }
            let done = !chunk.more || chunk.next_cursor.is_none();
            let next_cursor = chunk.next_cursor.unwrap_or(u64::MAX);
            // Marker LAST in the batch (commit order = batch order), so a torn
            // tail drops only the cursor advance and the chunk is redone
            // idempotently.
            let marker = BuildMarker {
                state: if done { BuildState::Built } else { BuildState::Building },
                generation: field.generation,
                cursor: next_cursor,
                positions: field.positions,
                stale_generations: stale,
                analyzer: field.analyzer.name().to_string(),
            };
            puts.push((marker_key.clone(), marker.encode()));
            db.storage.put_batch(&mut txn, &puts)?;
            crash_inject::hit(crash_inject::Site::FulltextBuildBeforeChunkCommit);
            match db.storage.commit(&mut txn) {
                Ok(_) => {
                    crash_inject::hit(crash_inject::Site::FulltextBuildAfterChunkCommit);
                    db.apply_fulltext_delta(&delta);
                    chunk_size = FULLTEXT_BUILD_CHUNK;
                    break (chunk.objects.len() as u64, next_cursor, done);
                }
                Err(rhypedb_storage::Error::WriteConflict) => {
                    // A live write inside this chunk's window landed first:
                    // re-scan at a fresh snapshot (its object now has an `l:`
                    // row and is skipped) with a smaller window, so a busy
                    // writer shrinks our steps instead of stalling them.
                    db.storage.abort(&mut txn);
                    chunk_size = (chunk_size / 2).max(1);
                }
                Err(e) => {
                    db.storage.abort(&mut txn);
                    return Err(match e {
                        rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                        other => EngineError::Storage(other),
                    });
                }
            }
        };
        field.progress.add_visited(visited);
        cursor = next_cursor;
        if done {
            field.progress.set_state(BuildState::Built);
            return Ok(true);
        }
    }
}

/// Sweep index rows of `(type_id, field_id)` whose generation is not `keep`
/// (all of them when `None`), then clear the stale flag / delete the marker.
fn run_drop(
    weak: &std::sync::Weak<Database>,
    builder: &FulltextBuilder,
    type_id: u64,
    field_id: u64,
    keep: Option<u32>,
) -> EngineResult<bool> {
    for prefix in [
        KeyBuilder::fulltext_field_all_generations_prefix(type_id, field_id),
        KeyBuilder::fulltext_doc_all_generations_prefix(type_id, field_id),
    ] {
        let mut start = prefix.clone();
        loop {
            if builder.stop.load(Ordering::Acquire) {
                return Ok(false);
            }
            let Some(db) = weak.upgrade() else {
                return Ok(false);
            };
            let _migration_guard = db.migration_lock.read();
            let snapshot = db.storage.read_snapshot();
            let chunk = db.storage.scan_chunk_raw(snapshot, &prefix, &start, FULLTEXT_DROP_CHUNK)?;
            let deletes: Vec<Bytes> = chunk
                .live
                .iter()
                .filter(|(k, _)| keep.is_none_or(|g| KeyBuilder::fulltext_generation(k) != Some(g)))
                .map(|(k, _)| k.clone())
                .collect();
            if !deletes.is_empty() {
                // A conflict here means a live writer touched one of these
                // keys after our snapshot — only possible for the CURRENT
                // generation, which we never delete — so a retry re-scans and
                // simply sees the same stale rows again.
                commit_writes(&db.storage, &[], &deletes)?;
            }
            match chunk.high_water {
                Some(hw) if chunk.more => {
                    // Resume strictly after the highest key visited.
                    let mut next = hw.to_vec();
                    next.push(0);
                    start = Bytes::from(next);
                }
                _ => break,
            }
        }
    }
    // Finalize the marker.
    let Some(db) = weak.upgrade() else {
        return Ok(false);
    };
    let marker_key = KeyBuilder::catalog_fulltext_marker(type_id, field_id);
    match keep {
        Some(_) => {
            let snapshot = db.storage.read_snapshot();
            if let Some(mut m) = read_marker(&db.storage, snapshot, type_id, field_id)? {
                m.stale_generations = false;
                commit_writes(&db.storage, &[(marker_key, m.encode())], &[])?;
            }
        }
        None => commit_writes(&db.storage, &[], &[marker_key])?,
    }
    Ok(true)
}

impl Database {
    /// Spawn the builder thread for this handle's pending tasks (no-op when
    /// there are none). Called once at the end of open / reload.
    pub(super) fn spawn_fulltext_builder(self: &Arc<Self>) -> EngineResult<()> {
        if self.fulltext_builder.tasks.lock().is_empty() {
            return Ok(());
        }
        let weak = Arc::downgrade(self);
        let builder = Arc::clone(&self.fulltext_builder);
        let handle = std::thread::Builder::new()
            .name("rhypedb-fulltext-build".into())
            .spawn(move || fulltext_builder_main(weak, builder))
            .map_err(|e| EngineError::Storage(rhypedb_storage::Error::Io(e)))?;
        *self.fulltext_builder.handle.lock() = Some(handle);
        Ok(())
    }

    /// Ask the builder to stop at its next chunk boundary and wait for it.
    /// Safe to call from any thread (self-join guarded) and more than once.
    pub(crate) fn stop_fulltext_builder(&self) {
        self.fulltext_builder.stop.store(true, Ordering::Release);
        if let Some(handle) = self.fulltext_builder.handle.lock().take()
            && handle.thread().id() != std::thread::current().id()
        {
            let _ = handle.join();
        }
    }

    /// Test hook: run this handle's pending tasks ON THE CALLING THREAD (the
    /// crash-fuzz injector arms a thread-local, so the builder must run where
    /// the harness armed it). Requires the handle to have been opened with
    /// `background_fulltext_build: false` (otherwise the thread owns them).
    #[cfg(test)]
    pub(crate) fn run_fulltext_tasks_inline(self: &Arc<Self>) {
        fulltext_builder_main(Arc::downgrade(self), Arc::clone(&self.fulltext_builder));
    }

    /// Per-field build state + progress, sorted by `Type.field` (for
    /// `GET /status`).
    pub fn fulltext_status(&self) -> Vec<FulltextIndexStatus> {
        let mut out: Vec<FulltextIndexStatus> = self
            .fulltext_fields
            .iter()
            .flat_map(|(type_name, fields)| {
                fields.iter().map(move |ff| FulltextIndexStatus {
                    name: format!("{type_name}.{}", ff.name),
                    state: ff.progress.state(),
                    generation: ff.generation,
                    documents: ff.stats.snapshot().doc_count,
                    indexed: ff.progress.visited(),
                    total: ff.progress.total(),
                })
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Block until every background full-text task of this handle has
    /// completed (builds AND drops), or `timeout` elapses. `true` = idle.
    /// A handle opened with `background_fulltext_build: false` never runs
    /// its tasks, so this returns `false` unless there were none.
    pub fn wait_for_fulltext_builds(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.fulltext_builder.pending.load(Ordering::Acquire) == 0 {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}
