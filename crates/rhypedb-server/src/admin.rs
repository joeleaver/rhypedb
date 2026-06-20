//! Card 5: HTTP admin surface for field-type migrations.
//!
//! Routes live under `/admin/*` (migrations, `/admin/reload`, `/admin/compact`)
//! and are gated by a single static token (`RHYPEDB_ADMIN_TOKEN`, read once at
//! startup): unset → 403, mismatch → 401. A quick safety net; real RBAC is a
//! separate epic. Engine calls are blocking, so each handler hops onto
//! `spawn_blocking` (the established pattern from `handle_admin_compact`).
//! Responses are built with `serde_json::json!` over the engine's public structs
//! so the engine stays serde-free.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use rhypedb_engine::database::{
    Database, MigrationEvent, MigrationFilter, MigrationPlanSpec, MigrationProgress,
    MigrationSummary, QuarantineEntry,
};
use rhypedb_engine::logical::{LogicalExportOptions, VectorMode};
use rhypedb_engine::{EngineError, ErrorPolicy, MigrationPhase, MigrationStatus};
use rhypedb_schema::parser::parse_schema;
use rhypedb_schema::{FieldType, ScalarType, Schema};

use crate::AppState;

pub(crate) fn admin_router(state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .route("/admin/migrations", post(start).get(list))
        .route("/admin/migrations/{id}", get(detail))
        .route("/admin/migrations/{id}/pause", post(pause))
        .route("/admin/migrations/{id}/resume", post(resume))
        .route("/admin/migrations/{id}/cancel", post(cancel))
        .route("/admin/migrations/{id}/cutover", post(cutover))
        .route("/admin/migrations/{id}/quarantine", get(quarantine))
        .route("/admin/migrations/{id}/quarantine/retry", post(retry_quarantine))
        .route("/admin/migrations/{id}/events", get(events))
        // In-place schema hot-reload (post-cutover stale handle, or general SDL
        // drift). Body = the updated SDL; refused while a migration is in flight.
        .route("/admin/reload", post(reload))
        // Force-flush + compact. Mutating + expensive, so gated like the rest of
        // the admin surface. The handler lives in lib.rs next to /query.
        .route("/admin/compact", post(crate::handle_admin_compact))
        // Physical backup: snapshot the live data dir. `POST /admin/backup`
        // writes a snapshot dir to a server path; `GET /admin/backup/stream`
        // streams the snapshot back as a tar.
        .route("/admin/backup", post(backup))
        .route("/admin/backup/stream", get(backup_stream))
        // Logical export: portable, version-independent NDJSON dump.
        // `POST /admin/export` writes it to a server path; `GET
        // /admin/export/stream` streams it back.
        .route("/admin/export", post(export))
        .route("/admin/export/stream", get(export_stream))
        // Logical import: apply a portable NDJSON dump to the LIVE database
        // (additive, non-atomic, upsert-by-id). Streamed body; refused with 409
        // while a field-type migration is in flight.
        .route("/admin/import/stream", post(import_stream))
        // Auth applies to these routes only (NOT /query, /status, /health).
        .route_layer(middleware::from_fn_with_state(state, admin_auth))
}

/// Gate every admin request on `RHYPEDB_ADMIN_TOKEN`. Unset → 403 (admin
/// disabled); present but the `Authorization: Bearer <token>` header is missing
/// or wrong → 401.
async fn admin_auth(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let Some(token) = state.admin_token.as_deref() else {
        return (
            StatusCode::FORBIDDEN,
            "admin endpoints disabled: set RHYPEDB_ADMIN_TOKEN\n",
        )
            .into_response();
    };
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    if presented != Some(token) {
        return (
            StatusCode::UNAUTHORIZED,
            "missing or invalid admin token\n",
        )
            .into_response();
    }
    next.run(req).await
}

// ---------------------------------------------------------------------------
// Request body for POST /admin/migrations
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CreateMigrationRequest {
    #[serde(rename = "type")]
    type_name: String,
    field: String,
    /// Target scalar kind name, SDL spelling (e.g. "f64", "i64", "String").
    to: String,
    converter: String,
    #[serde(default)]
    converter_version: u32,
    #[serde(default)]
    chunk: u64,
    #[serde(default)]
    parallel: Option<u8>,
    #[serde(default)]
    policy: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    quarantine_cap: u64,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn start(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateMigrationRequest>,
) -> Response {
    let Some(target) = field_type_from_str(&req.to) else {
        return bad_request(&format!("unknown target type {:?}", req.to));
    };
    let policy = match req.policy.as_deref().map(policy_from_str) {
        None => ErrorPolicy::Stop,
        Some(Some(p)) => p,
        Some(None) => return bad_request("policy must be one of stop|skip|quarantine"),
    };
    let spec = MigrationPlanSpec {
        type_name: req.type_name,
        field_name: req.field,
        target_field_type: target,
        converter_name: req.converter,
        converter_version: req.converter_version,
        chunk_size: req.chunk,
        error_policy: policy,
        dry_run: req.dry_run,
        quarantine_cap: req.quarantine_cap,
        parallel_degree: req.parallel,
    };
    // Hold the schema-epoch read guard across the arm so a hot-reload (write
    // guard) can't swap the handle between create and the hook being armed.
    let _epoch = state.reload_lock.read().await;
    let db = state.db();
    match spawn_blocking_engine(move || db.start_field_type_migration_async(spec)).await {
        Ok(handle) => {
            // Register the completion watcher so the live handle hot-reloads to
            // the post-cutover schema when this plan finishes — no restart.
            ensure_reload_watcher(&state, handle.plan_id);
            (
                StatusCode::OK,
                Json(json!({
                    "migration_id": handle.plan_id,
                    "created_at_ms": handle.created_at_ms,
                })),
            )
                .into_response()
        }
        Err(e) => err_response(e),
    }
}

#[derive(Deserialize)]
struct ListQuery {
    status: Option<String>,
    #[serde(rename = "type")]
    type_name: Option<String>,
}

async fn list(State(state): State<Arc<AppState>>, Query(q): Query<ListQuery>) -> Response {
    let status = match q.status.as_deref().map(str_to_status) {
        None => None,
        Some(Some(s)) => Some(s),
        Some(None) => return bad_request("unknown status filter"),
    };
    let filter = MigrationFilter {
        status,
        type_name: q.type_name,
    };
    let db = state.db();
    match spawn_blocking_engine(move || db.list_migrations_filtered(&filter)).await {
        Ok(rows) => {
            let arr: Vec<JsonValue> = rows.iter().map(summary_json).collect();
            (StatusCode::OK, Json(json!({ "migrations": arr }))).into_response()
        }
        Err(e) => err_response(e),
    }
}

async fn detail(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let db = state.db();
    match spawn_blocking_engine(move || db.query_migration_progress(id)).await {
        Ok(p) => (StatusCode::OK, Json(progress_json(&p))).into_response(),
        Err(e) => err_response(e),
    }
}

async fn pause(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.pause_migration(id)).await
}

async fn resume(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    // Hold the schema-epoch read guard across the resume so a hot-reload (write
    // guard) can't swap the handle while resume arms the hook on it — the same
    // arm-vs-swap fence `start` uses. (`resume_field_type_migration` drives the
    // plan inline to completion, so this also serializes the whole resume vs a
    // reload; a concurrent reload waits, then sees the settled plan.)
    let _epoch = state.reload_lock.read().await;
    let db = state.db();
    match spawn_blocking_engine(move || db.resume_field_type_migration(id)).await {
        Ok(()) => {
            // A resumed plan may now be Completed (it drives inline) — arm a
            // watcher whose subscribe-then-recheck reloads the live handle even
            // though completion already happened. Idempotent (stash consumed once).
            ensure_reload_watcher(&state, id);
            (StatusCode::OK, Json(json!({ "ok": true }))).into_response()
        }
        Err(e) => err_response(e),
    }
}

async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.cancel_migration(id)).await
}

async fn cutover(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.cutover_migration(id)).await
}

async fn quarantine(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let db = state.db();
    match spawn_blocking_engine(move || db.list_quarantined(id)).await {
        Ok(rows) => {
            let arr: Vec<JsonValue> = rows.iter().map(quarantine_json).collect();
            (StatusCode::OK, Json(json!({ "quarantined": arr }))).into_response()
        }
        Err(e) => err_response(e),
    }
}

#[derive(Deserialize)]
struct RetryQuarantineRequest {
    new_converter: String,
    /// Specific object ids to retry; empty / omitted → every quarantined row.
    #[serde(default)]
    ids: Vec<u64>,
}

async fn retry_quarantine(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<RetryQuarantineRequest>,
) -> Response {
    let db = state.db();
    let resolved = spawn_blocking_engine(move || {
        // Empty ids → retry the whole current quarantine set.
        let ids = if req.ids.is_empty() {
            db.list_quarantined(id)?
                .into_iter()
                .map(|q| q.object_id)
                .collect::<Vec<_>>()
        } else {
            req.ids
        };
        db.retry_quarantined(id, &ids, &req.new_converter)
    })
    .await;
    match resolved {
        Ok(n) => (StatusCode::OK, Json(json!({ "resolved": n }))).into_response(),
        Err(e) => err_response(e),
    }
}

/// SSE stream of `MigrationEvent`s for one plan. Bridges the engine's blocking
/// `std::sync::mpsc` channel to an async stream: a `spawn_blocking` forwarder
/// drains `recv_timeout(2s)` into a bounded tokio channel and exits on a
/// terminal event, on the client disconnecting (tokio sender closed), or on the
/// hub dropping the engine sender — so it never parks a thread forever.
async fn events(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let engine_rx = state.db().subscribe_migration_events(id);
    let (tx, rx) = tokio::sync::mpsc::channel::<MigrationEvent>(64);
    tokio::task::spawn_blocking(move || loop {
        match engine_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(ev) => {
                // A terminal event ENDS the stream (CutoverDone is NOT terminal —
                // a StatusChanged(Completed) always follows it). Mirrors the
                // engine's MigrationEvent::is_terminal.
                let terminal = ev.is_terminal();
                // try_send (never block): a slow/hung-but-connected client must
                // not park this blocking-pool thread on a full channel. Events are
                // a best-effort, non-replayed live stream, so a dropped frame under
                // backpressure is acceptable; only a CLOSED receiver (client gone)
                // ends the forwarder.
                match tx.try_send(ev) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
                }
                if terminal {
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if tx.is_closed() {
                    break; // client gone; stop parking on recv
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    });
    let stream = ReceiverStream::new(rx).map(|ev| {
        Ok::<Event, std::convert::Infallible>(
            Event::default()
                .event(event_name(&ev))
                .json_data(event_json(&ev))
                .unwrap_or_else(|_| Event::default().data("{}")),
        )
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------
// In-place schema hot-reload
// ---------------------------------------------------------------------------

/// `POST /admin/reload` — body is the updated SDL. Hot-reloads the live handle in
/// place so the operator never has to restart the server to pick up a schema
/// change (e.g. after a `change_field_type` migration cuts over). `400` on a
/// parse error; `409` if a migration is still in flight; on a rebuild error the
/// old handle stays live and serving.
///
/// The reload runs the same catalog reconcile as a reopen, with its limits: a
/// field-kind change with no settled plan is refused (`FieldKindChanged`) and a
/// type/field drop needs the shrink opt-in. NOTE: like a reopen, NEWLY adding
/// `@indexed`/`@unique` to an already-populated field does NOT backfill the
/// secondary index for existing rows — only rows written after the reload are
/// indexed. The intended use is picking up a post-cutover field-kind flip, not
/// arbitrary index/constraint changes on populated data.
async fn reload(State(state): State<Arc<AppState>>, body: String) -> Response {
    let schema = match parse_schema(&body) {
        Ok(s) => s,
        Err(e) => return bad_request(&format!("schema parse error: {e}")),
    };
    match do_reload(&state, schema).await {
        Ok(new) => {
            let types: Vec<JsonValue> = new
                .schema()
                .types
                .iter()
                .map(|(name, td)| json!({ "type": name, "fields": td.fields.len() }))
                .collect();
            (StatusCode::OK, Json(json!({ "ok": true, "schema": types }))).into_response()
        }
        Err(e) => err_response(e),
    }
}

/// Swap the live `Database` handle for a fresh one built on the SAME storage
/// under `target_schema`. The schema-epoch WRITE guard drains in-flight
/// schema-driven ops (query execute, migration `start`) and blocks new ones for
/// the swap, so nothing straddles it. The engine refuses
/// (`ReloadBlockedByActiveMigration` → 409) if any migration hook is still
/// armed; a failed rebuild leaves the old handle untouched (no brick). Returns
/// the new handle.
async fn do_reload(state: &Arc<AppState>, target_schema: Schema) -> Result<Arc<Database>, EngineError> {
    let _epoch = state.reload_lock.write().await;
    let db = state.db();
    let new = spawn_blocking_engine(move || db.reload_handle(target_schema)).await?;
    state.db.store(new.clone());
    Ok(new)
}

/// Look up `plan_id` and, if the live handle is stale w.r.t. it, arm a reload
/// watcher. Used by `start` / `resume` after a verb returns.
fn ensure_reload_watcher(state: &Arc<AppState>, plan_id: u64) {
    let db = state.db();
    match db.list_migrations() {
        Ok(all) => {
            if let Some(s) = all.iter().find(|s| s.plan_id == plan_id) {
                arm_reload_watcher(state, s);
            }
        }
        Err(e) => eprintln!("reload watcher: could not read plan {plan_id}: {e}"),
    }
}

/// If the live handle is STALE w.r.t. this plan's target kind, stash the target
/// schema and spawn a completion watcher. No-op when:
/// - the plan can never flip the catalog (dry-run / Cancelled / DryRunCompleted),
/// - the target kind is non-scalar (`change_field_type` never produces this), or
/// - the live handle ALREADY declares the target kind (opened with the target
///   SDL, or already reloaded — keeps this idempotent and skips the common
///   already-correct case).
///
/// Otherwise the watcher (see [`spawn_reload_watcher`]) hot-reloads on completion
/// — covering the still-Running, the just-`Completed` (e.g. a synchronous
/// `resume`), and the post-restart cases uniformly.
fn arm_reload_watcher(state: &Arc<AppState>, s: &MigrationSummary) {
    if s.dry_run
        || matches!(
            s.status,
            MigrationStatus::Cancelled | MigrationStatus::DryRunCompleted
        )
    {
        return;
    }
    let Some(target) = s.target_field_type.clone() else {
        eprintln!(
            "reload watcher: plan {} has a non-scalar target; skipping auto-reload",
            s.plan_id
        );
        return;
    };
    let db = state.db();
    let already_target = db
        .schema()
        .get_type(&s.type_name)
        .and_then(|t| t.get_field(&s.field_name))
        .map(|f| f.field_type == target)
        .unwrap_or(false);
    if already_target {
        return;
    }
    let Some(schema) = db.schema().with_field_type(&s.type_name, &s.field_name, target) else {
        return;
    };
    state
        .pending_reload_schemas
        .lock()
        .unwrap()
        .insert(s.plan_id, schema);
    spawn_reload_watcher(state.clone(), s.plan_id);
}

/// Watch one plan and hot-reload the live handle to the stashed target schema
/// when it reaches `Completed`. The engine event hub is **non-replayed** (a
/// subscriber attached after an event was published never sees it), and the
/// driver runs concurrently with — or, on the `resume`/startup paths, BEFORE —
/// this call. So after subscribing we RE-CHECK the plan's status: if it already
/// `Completed`, reload directly. This subscribe-then-recheck closes the race
/// where a fast/empty migration (or a synchronous resume) settles before we
/// subscribe. Otherwise we wait for the live `Completed` event.
fn spawn_reload_watcher(state: Arc<AppState>, plan_id: u64) {
    // Subscribe FIRST, so any event published from here on is delivered; the
    // recheck below then covers anything published just before the subscribe.
    let engine_rx = state.db().subscribe_migration_events(plan_id);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<MigrationEvent>(16);
    tokio::task::spawn_blocking(move || loop {
        match engine_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(ev) => {
                let terminal = ev.is_terminal();
                if tx.blocking_send(ev).is_err() {
                    break; // consumer gone
                }
                if terminal {
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if tx.is_closed() {
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    });
    tokio::spawn(async move {
        // Recheck: catch a completion that landed before we subscribed.
        if migration_completed(&state, plan_id).await {
            try_reload_from_stash(&state, plan_id).await;
            state.pending_reload_schemas.lock().unwrap().remove(&plan_id);
            return;
        }
        while let Some(ev) = rx.recv().await {
            if matches!(
                ev,
                MigrationEvent::StatusChanged {
                    status: MigrationStatus::Completed,
                    ..
                }
            ) {
                try_reload_from_stash(&state, plan_id).await;
            }
            if ev.is_terminal() {
                // Cancelled / Failed (or an already-handled Completed): clear any
                // remaining stash and stop watching.
                state.pending_reload_schemas.lock().unwrap().remove(&plan_id);
                break;
            }
        }
    });
}

/// Consume the stashed target schema for `plan_id` (removing it, so the reload
/// fires at most once even with a duplicate watcher) and hot-reload to it.
async fn try_reload_from_stash(state: &Arc<AppState>, plan_id: u64) {
    let schema = state.pending_reload_schemas.lock().unwrap().remove(&plan_id);
    if let Some(schema) = schema
        && let Err(e) = do_reload(state, schema).await
    {
        eprintln!("auto-reload after migration {plan_id} completed failed: {e}");
    }
}

/// Current status of `plan_id` is `Completed` (best-effort; `false` on any error).
async fn migration_completed(state: &Arc<AppState>, plan_id: u64) -> bool {
    let db = state.db();
    matches!(
        spawn_blocking_engine(move || db.query_migration_progress(plan_id)).await,
        Ok(p) if p.status == MigrationStatus::Completed
    )
}

/// Re-arm reload watchers at startup for every migration plan a prior run left
/// behind. `arm_reload_watcher` self-filters: it arms only plans whose target
/// kind the live handle does not yet declare (i.e. the handle is stale), so a
/// plan that finished its backfill post-restart — including one that completes in
/// the `resume_inflight` window between open and here — still hot-reloads, while
/// already-correct plans are skipped.
pub(crate) fn resume_reload_watchers(state: &Arc<AppState>) {
    let db = state.db();
    match db.list_migrations() {
        Ok(plans) => {
            for s in &plans {
                arm_reload_watcher(state, s);
            }
        }
        Err(e) => eprintln!("reload watchers: could not list plans at startup: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Run a unit-returning engine verb on a blocking thread, mapping the result to
/// `{"ok": true}` / an error response.
async fn verb<F>(state: Arc<AppState>, f: F) -> Response
where
    F: FnOnce(&rhypedb_engine::database::Database) -> Result<(), EngineError> + Send + 'static,
{
    let db = state.db();
    match spawn_blocking_engine(move || f(&db)).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "ok": true }))).into_response(),
        Err(e) => err_response(e),
    }
}

/// Run a blocking engine closure on the blocking pool, flattening a join panic
/// into a generic storage error.
async fn spawn_blocking_engine<T, F>(f: F) -> Result<T, EngineError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EngineError> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r,
        Err(join) => Err(EngineError::Storage(rhypedb_storage::Error::Io(
            std::io::Error::other(format!("admin task panicked: {join}")),
        ))),
    }
}

// ===================================================================
// Physical backup (Overboard cmqiioa2y).
// ===================================================================

fn io_err(e: std::io::Error) -> EngineError {
    EngineError::Storage(rhypedb_storage::Error::Io(e))
}

/// fsync a file's contents OR a directory's entries (a read-only fd is fine for
/// fsync). Used to make a backup crash-durable.
fn fsync_path(p: &std::path::Path) -> std::io::Result<()> {
    std::fs::File::open(p)?.sync_all()
}

/// Sync the CONTENTS of every snapshot data file, then the directories holding
/// them (so the dir entries — hard links + copies — persist). Hard-linked SSTs
/// already have durable inode data; the sync_all is then a cheap no-op fsync.
fn sync_backup_data(dir: &std::path::Path) -> std::io::Result<()> {
    let sst_dir = dir.join("sst");
    if let Ok(rd) = std::fs::read_dir(&sst_dir) {
        for e in rd.flatten() {
            fsync_path(&e.path())?;
        }
    }
    for name in ["wal.log", "schema.rhype"] {
        let p = dir.join(name);
        if p.is_file() {
            fsync_path(&p)?;
        }
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let n = e.file_name();
            let n = n.to_string_lossy();
            if n.starts_with("hnsw_") && n.ends_with(".bin") {
                fsync_path(&e.path())?;
            }
        }
    }
    fsync_path(&sst_dir)?;
    fsync_path(dir)?;
    Ok(())
}

/// Remove orphaned `.rhypedb-backup-stream-*` temp dirs left in `data_dir` by a
/// previous process that was HARD-killed mid-stream (TempDirGuard never ran).
/// They hard-link SSTs, so leaving them pins old inodes and leaks disk. Called
/// once at startup; the pid in each name belongs to a dead process.
pub(crate) fn reap_backup_temp_dirs(data_dir: &std::path::Path) {
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        for e in rd.flatten() {
            if e.file_name()
                .to_string_lossy()
                .starts_with(".rhypedb-backup-stream-")
            {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
}

/// Freeze a complete physical backup into `dir` (which must be fresh): the LSM
/// SSTs + WAL + `hnsw_*.bin` (via [`Database::backup_to`]), the SDL schema copied
/// in (it is NOT in the data dir but is required to open the restore), and a
/// `MANIFEST.json` written LAST — a reader treats its presence as "complete".
/// Returns the manifest as JSON. NOTE: backup does NOT hold `reload_lock` — a hot
/// reload swaps the handle to a fresh one on the SAME `Arc<LsmTree>`, and the
/// captured `db` keeps working on that shared storage; `flush_lock` (on the
/// storage) already serializes the snapshot vs writers.
fn backup_into_dir(
    db: &Database,
    schema_path: &std::path::Path,
    dir: &std::path::Path,
) -> Result<JsonValue, EngineError> {
    let manifest = db.backup_to(dir)?;
    // The schema is mandatory — a data-dir-only backup will not open.
    std::fs::copy(schema_path, dir.join("schema.rhype")).map_err(io_err)?;
    let in_flight: Vec<JsonValue> = manifest
        .in_flight_migrations
        .iter()
        .map(|(id, conv)| json!({ "plan_id": id, "converter": conv }))
        .collect();
    let manifest_json = json!({
        "format": "rhypedb-physical-backup-v1",
        "created_at_ms": manifest.created_at_ms,
        "max_version": manifest.max_version,
        "wal_bytes": manifest.wal_bytes,
        "ssts": manifest.sst_names,
        "hnsw_files": manifest.hnsw_files,
        "schema_file": "schema.rhype",
        "in_flight_migrations": in_flight,
    });
    let bytes = serde_json::to_vec_pretty(&manifest_json)
        .map_err(|e| io_err(std::io::Error::other(e.to_string())))?;
    // Crash-durability: sync every data file's CONTENTS + the dirs, THEN write +
    // sync the manifest, THEN fsync the dir LAST — so a durable "MANIFEST.json
    // present" implies every file it vouches for is durable too (the backup is
    // reached for precisely after a crash).
    sync_backup_data(dir).map_err(io_err)?;
    std::fs::write(dir.join("MANIFEST.json"), bytes).map_err(io_err)?;
    fsync_path(&dir.join("MANIFEST.json")).map_err(io_err)?;
    fsync_path(dir).map_err(io_err)?;
    Ok(manifest_json)
}

fn sanitize_label(l: &str) -> String {
    l.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn backup_dir_name(label: Option<&str>) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    match label {
        Some(l) if !l.is_empty() => format!("rhypedb-backup-{ts}-{}", sanitize_label(l)),
        _ => format!("rhypedb-backup-{ts}"),
    }
}

#[derive(Deserialize)]
struct BackupReq {
    /// Directory on the SERVER's filesystem to place the snapshot under. A
    /// timestamped `rhypedb-backup-<ms>[-<label>]/` subdir is created in it.
    dest: String,
    label: Option<String>,
}

/// `POST /admin/backup` — write a physical snapshot to a path on the server.
async fn backup(State(state): State<Arc<AppState>>, Json(req): Json<BackupReq>) -> Response {
    let db = state.db();
    let schema_path = state.schema_path.clone();
    let dest = std::path::PathBuf::from(req.dest);
    let label = req.label;
    match spawn_blocking_engine(move || {
        let dir = dest.join(backup_dir_name(label.as_deref()));
        std::fs::create_dir_all(&dir).map_err(io_err)?;
        match backup_into_dir(&db, &schema_path, &dir) {
            Ok(manifest_json) => Ok(json!({
                "ok": true,
                "path": dir.to_string_lossy(),
                "manifest": manifest_json,
            })),
            Err(e) => {
                // Don't leave a partial (manifest-less) dir behind on failure.
                let _ = std::fs::remove_dir_all(&dir);
                Err(e)
            }
        }
    })
    .await
    {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => err_response(e),
    }
}

/// Removes its directory on drop (best-effort temp cleanup).
struct TempDirGuard(std::path::PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `std::io::Write` that forwards each chunk to an async mpsc channel so a
/// blocking tar build can stream out a response body without buffering.
struct ChannelWriter {
    tx: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
}
impl std::io::Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.tx
            .blocking_send(Ok(buf.to_vec()))
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "client disconnected")
            })?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn unique_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos)
}

/// `GET /admin/backup/stream` — freeze a snapshot into a temp dir on the data
/// dir's filesystem (so SSTs hard-link), then stream it back as a `tar` archive
/// (chunked, never buffered — OOM-safe). The temp dir is removed afterward.
///
/// A failure AFTER the 200 response begins (e.g. a snapshot/tar error mid-stream)
/// can only truncate the body, so clients MUST validate the tar's end-of-archive
/// marker before trusting a download (the CLI's `tar::Archive::unpack` does).
async fn backup_stream(State(state): State<Arc<AppState>>) -> Response {
    let db = state.db();
    let schema_path = state.schema_path.clone();
    let data_dir = state.data_dir.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(16);

    tokio::task::spawn_blocking(move || {
        // Temp snapshot dir INSIDE the data dir → guaranteed same filesystem, so
        // the SST hard-links work. Removed on drop (incl. the error path).
        let tmp = data_dir.join(format!(".rhypedb-backup-stream-{}", unique_suffix()));
        let _guard = TempDirGuard(tmp.clone());
        let build = (|| -> Result<(), std::io::Error> {
            std::fs::create_dir_all(&tmp)?;
            backup_into_dir(&db, &schema_path, &tmp)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut builder = tar::Builder::new(ChannelWriter { tx: tx.clone() });
            builder.append_dir_all(".", &tmp)?;
            builder.finish()?;
            Ok(())
        })();
        if let Err(e) = build {
            let _ = tx.blocking_send(Err(e));
        }
    });

    let body =
        axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-tar")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"rhypedb-backup.tar\"",
        )
        .body(body)
        .expect("valid backup-stream response")
}

// ===================================================================
// Logical export (Overboard cmqikqug4).
// ===================================================================

#[derive(Deserialize)]
struct ExportQuery {
    /// Comma-separated type names; empty/absent = every type.
    types: Option<String>,
    /// Vector handling: "raw" (default) | "none".
    vectors: Option<String>,
}

/// Build [`LogicalExportOptions`] from query/body inputs. Errors on an unknown
/// vectors mode; the type set itself is validated by the engine (unknown type
/// → 404).
fn build_export_opts(
    types_csv: Option<&str>,
    types_list: Option<Vec<String>>,
    vectors: Option<&str>,
) -> Result<LogicalExportOptions, String> {
    let types = match (types_csv, types_list) {
        (_, Some(list)) => {
            let v: Vec<String> = list
                .into_iter()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            (!v.is_empty()).then_some(v)
        }
        (Some(csv), None) => {
            let v: Vec<String> = csv
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            (!v.is_empty()).then_some(v)
        }
        (None, None) => None,
    };
    let vectors = match vectors {
        None | Some("raw") => VectorMode::Raw,
        Some("none") => VectorMode::None,
        Some("reembed") => VectorMode::Reembed,
        Some(other) => {
            return Err(format!(
                "unknown vectors mode '{other}' (expected raw|none|reembed)"
            ));
        }
    };
    Ok(LogicalExportOptions { types, vectors })
}

fn export_vectors_tag(opts: &LogicalExportOptions) -> &'static str {
    match opts.vectors {
        VectorMode::Raw => "raw",
        VectorMode::None => "none",
        VectorMode::Reembed => "reembed",
    }
}

/// `GET /admin/export/stream` — stream a logical NDJSON dump (chunked, never
/// buffered — OOM-safe). A pre-flight refuses mid field-type migration with a
/// clean 409. A failure AFTER the 200 begins can only truncate the body, so
/// clients MUST validate the trailer line before trusting the download (the
/// CLI's `verify-export` / post-download check does).
async fn export_stream(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ExportQuery>,
) -> Response {
    let opts = match build_export_opts(q.types.as_deref(), None, q.vectors.as_deref()) {
        Ok(o) => o,
        Err(m) => return bad_request(&m),
    };
    let db = state.db();
    // Pre-flight before committing to a 200 stream: an unknown type is a clean
    // 404 and an in-flight field-type migration a clean 409, rather than a
    // truncated body the client only discovers via the missing trailer.
    if let Some(types) = &opts.types {
        let schema = db.schema();
        for t in types {
            if schema.get_type(t).is_none() {
                return err_response(EngineError::TypeNotFound(t.clone()));
            }
        }
    }
    let migrating = db.migrating_fields();
    if migrating > 0 {
        return err_response(EngineError::ExportWhileMigrating {
            migrating_fields: migrating,
        });
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(16);
    tokio::task::spawn_blocking(move || {
        let mut writer = ChannelWriter { tx: tx.clone() };
        if let Err(e) = db.logical_export_stream(&mut writer, &opts) {
            let _ = tx.blocking_send(Err(std::io::Error::other(e.to_string())));
        }
    });

    let body = axum::body::Body::from_stream(ReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"rhypedb-export.ndjson\"",
        )
        .body(body)
        .expect("valid export-stream response")
}

#[derive(Deserialize)]
struct ExportReq {
    /// Directory on the SERVER's filesystem to place the export under. A
    /// timestamped `rhypedb-export-<ms>[-<label>]/` subdir is created in it.
    dest: String,
    label: Option<String>,
    types: Option<Vec<String>>,
    vectors: Option<String>,
}

/// `POST /admin/export` — write a logical export to a path on the server.
async fn export(State(state): State<Arc<AppState>>, Json(req): Json<ExportReq>) -> Response {
    let opts = match build_export_opts(None, req.types, req.vectors.as_deref()) {
        Ok(o) => o,
        Err(m) => return bad_request(&m),
    };
    let db = state.db();
    let dest = std::path::PathBuf::from(req.dest);
    let label = req.label;
    match spawn_blocking_engine(move || {
        let dir = dest.join(export_dir_name(label.as_deref()));
        std::fs::create_dir_all(&dir).map_err(io_err)?;
        match export_into_dir(&db, &dir, &opts) {
            Ok(manifest_json) => Ok(json!({
                "ok": true,
                "path": dir.to_string_lossy(),
                "manifest": manifest_json,
            })),
            Err(e) => {
                // Don't leave a partial (manifest-less) dir behind on failure.
                let _ = std::fs::remove_dir_all(&dir);
                Err(e)
            }
        }
    })
    .await
    {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => err_response(e),
    }
}

/// Write a logical export into `dir` (fresh): `export.ndjson` (whose own
/// trailer is its completeness sentinel) plus a `MANIFEST.json` written LAST.
/// Crash-durability: fsync the data file, then write + fsync the manifest, then
/// fsync the dir — so a durable MANIFEST.json implies a durable, complete data
/// file.
fn export_into_dir(
    db: &Database,
    dir: &std::path::Path,
    opts: &LogicalExportOptions,
) -> Result<JsonValue, EngineError> {
    let data_path = dir.join("export.ndjson");
    let mut file = std::fs::File::create(&data_path).map_err(io_err)?;
    let summary = db.logical_export_stream(&mut file, opts)?;
    file.sync_all().map_err(io_err)?;

    let mut counts = serde_json::Map::new();
    for (t, c) in &summary.counts {
        counts.insert(
            t.clone(),
            json!({
                "objects": c.objects,
                "edges": c.edges,
                "vectors": c.vectors,
                "dangling_edges_skipped": c.dangling_edges_skipped,
            }),
        );
    }
    let created_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let manifest_json = json!({
        "format": "rhypedb-logical-export-v1",
        "created_at_ms": created_at_ms,
        "data_file": "export.ndjson",
        "vectors": export_vectors_tag(opts),
        "counts": JsonValue::Object(counts),
        "complete": true,
    });
    let bytes = serde_json::to_vec_pretty(&manifest_json)
        .map_err(|e| io_err(std::io::Error::other(e.to_string())))?;
    std::fs::write(dir.join("MANIFEST.json"), bytes).map_err(io_err)?;
    fsync_path(&dir.join("MANIFEST.json")).map_err(io_err)?;
    fsync_path(dir).map_err(io_err)?;
    Ok(manifest_json)
}

fn export_dir_name(label: Option<&str>) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    match label {
        Some(l) if !l.is_empty() => format!("rhypedb-export-{ts}-{}", sanitize_label(l)),
        _ => format!("rhypedb-export-{ts}"),
    }
}

#[derive(Deserialize)]
struct ImportQuery {
    vectors: Option<String>,
}

/// Per-process monotonic counter making each online-import temp file unique even
/// when two imports read the clock in the same instant.
static IMPORT_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Removes a temp file on drop unless disarmed — so a cancelled handler (client
/// disconnect) does not leak the streamed file into the data dir.
struct TempFileGuard(std::path::PathBuf);
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn server_error(msg: &str) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": msg }))).into_response()
}

/// `POST /admin/import/stream` — apply a logical NDJSON dump to the LIVE
/// database (additive, insert-only, refuses id collisions; see
/// `run_online_import`). The body is streamed to a temp file (OOM-safe) before
/// being applied on the blocking pool. Refused with 409 while a field-type
/// migration is in flight; a concurrent hot-reload is fenced for the apply.
async fn import_stream(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ImportQuery>,
    body: axum::body::Body,
) -> Response {
    let vectors = match q.vectors.as_deref() {
        None | Some("raw") => crate::import::VectorImportMode::Raw,
        Some("none") => crate::import::VectorImportMode::None,
        Some("reembed") => crate::import::VectorImportMode::Reembed,
        Some(other) => {
            return bad_request(&format!(
                "unknown vectors mode '{other}' (expected raw|none|reembed)"
            ));
        }
    };

    // Unique temp path (pid + monotonic nonce + clock) so concurrent imports
    // never collide on the file. The guard removes it on EVERY exit, including
    // handler cancellation.
    let nonce = IMPORT_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = state.data_dir.join(format!(
        ".rhypedb-online-import-{}-{nonce}-{nanos}.ndjson",
        std::process::id()
    ));
    let _tmp_guard = TempFileGuard(tmp.clone());

    // Stream the body to the temp file FIRST — no locks held during the
    // (potentially large) upload. A write failure here is a server-side IO error.
    if let Err(m) = stream_body_to_file(body, &tmp).await {
        return server_error(&m);
    }

    // Fence a concurrent hot-reload (like /query) and refuse a migration in
    // flight, then apply on the blocking pool. (The fence is held for the whole
    // apply — a long import blocks a reload for its duration, which is intended:
    // the import needs a stable schema.)
    let _epoch = state.reload_lock.read().await;
    let db = state.db();
    let migrating = db.migrating_fields();
    if migrating > 0 {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": format!(
                    "cannot import while a field-type migration is in progress ({migrating} field(s))"
                )
            })),
        )
            .into_response();
    }

    let vectorizer = state.vectorizer.clone();
    let apply_path = tmp.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::import::run_online_import(&apply_path, &db, vectors, vectorizer.as_deref())
    })
    .await;

    match result {
        Ok(Ok(report)) => (
            StatusCode::OK,
            Json(json!({
                "ok": true,
                "types": report.types,
                "objects": report.objects,
                "edges": report.edges,
                "vectors": report.vectors,
                "reembed_enqueued": report.reembed_enqueued,
            })),
        )
            .into_response(),
        // Import errors are dump/schema/collision issues (client-facing).
        Ok(Err(msg)) => bad_request(&msg),
        Err(join) => server_error(&format!("import task failed: {join}")),
    }
}

/// Stream a request body to `path`, bounded memory (one chunk at a time).
async fn stream_body_to_file(
    body: axum::body::Body,
    path: &std::path::Path,
) -> Result<(), String> {
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;
    let mut file = tokio::fs::File::create(path)
        .await
        .map_err(|e| format!("create temp import file: {e}"))?;
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read request body: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write temp import file: {e}"))?;
    }
    file.flush()
        .await
        .map_err(|e| format!("flush temp import file: {e}"))?;
    Ok(())
}

fn err_response(e: EngineError) -> Response {
    let code = match &e {
        EngineError::MigrationPlanNotFound { .. } | EngineError::TypeNotFound(_) => {
            StatusCode::NOT_FOUND
        }
        EngineError::MigrationCancelledCannotCutover { .. }
        | EngineError::MigrationCannotCancelInCutover { .. }
        | EngineError::MigrationCannotCancelSettled { .. }
        | EngineError::MigrationAlreadyRunning { .. }
        | EngineError::ReloadBlockedByActiveMigration { .. }
        | EngineError::ExportWhileMigrating { .. }
        | EngineError::MigrationCutoverHasErrors { .. } => StatusCode::CONFLICT,
        EngineError::ConverterNotRegistered { .. }
        | EngineError::MigrationResumeSchemaMismatch { .. }
        | EngineError::MigrationFieldConverterUnresolved { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (code, Json(json!({ "error": e.to_string() }))).into_response()
}

fn bad_request(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response()
}

fn summary_json(s: &MigrationSummary) -> JsonValue {
    json!({
        "plan_id": s.plan_id,
        "type": s.type_name,
        "field": s.field_name,
        "status": status_str(s.status),
        "objects_converted": s.objects_converted,
        "chunk_size": s.chunk_size,
        "converter": s.converter_name,
        "converter_version": s.converter_version,
        "created_at_ms": s.created_at_ms,
        "error_count": s.error_count,
        "error_policy": policy_str(s.error_policy),
        "dry_run": s.dry_run,
    })
}

fn progress_json(p: &MigrationProgress) -> JsonValue {
    let parts: Vec<JsonValue> = p
        .partitions
        .iter()
        .map(|pp| {
            json!({
                "idx": pp.idx,
                "lo": pp.lo,
                "hi": pp.hi,
                "cursor": pp.cursor,
                "objects_converted": pp.objects_converted,
                "errors": pp.errors,
                "done": pp.done,
            })
        })
        .collect();
    json!({
        "plan_id": p.plan_id,
        "type": p.type_name,
        "field": p.field_name,
        "status": status_str(p.status),
        "phase": phase_str(p.phase),
        "dry_run": p.dry_run,
        "parallel_degree": p.parallel_degree,
        "total_objects": p.total_objects,
        "objects_converted": p.objects_converted,
        "errors": p.errors,
        "created_at_ms": p.created_at_ms,
        "now_ms": p.now_ms,
        "objects_per_sec": p.objects_per_sec,
        "eta_unix_ms": p.eta_unix_ms,
        "partitions": parts,
    })
}

fn quarantine_json(q: &QuarantineEntry) -> JsonValue {
    json!({
        "object_id": q.object_id,
        "error": q.error_msg,
        "errored_at_ms": q.errored_at_ms,
        "attempted_converter": q.attempted_converter_name,
    })
}

fn event_name(ev: &MigrationEvent) -> &'static str {
    match ev {
        MigrationEvent::ChunkCompleted { .. } => "chunk_completed",
        MigrationEvent::PartitionDone { .. } => "partition_done",
        MigrationEvent::CutoverStarted { .. } => "cutover_started",
        MigrationEvent::CutoverDone { .. } => "cutover_done",
        MigrationEvent::RollbackStarted { .. } => "rollback_started",
        MigrationEvent::StatusChanged { .. } => "status_changed",
        MigrationEvent::Failed { .. } => "failed",
    }
}

fn event_json(ev: &MigrationEvent) -> JsonValue {
    match ev {
        MigrationEvent::ChunkCompleted {
            plan_id,
            partition_idx,
            cursor,
            objects_converted,
        } => json!({
            "type": "chunk_completed", "plan_id": plan_id, "partition_idx": partition_idx,
            "cursor": cursor, "objects_converted": objects_converted,
        }),
        MigrationEvent::PartitionDone {
            plan_id,
            partition_idx,
        } => json!({ "type": "partition_done", "plan_id": plan_id, "partition_idx": partition_idx }),
        MigrationEvent::CutoverStarted { plan_id } => {
            json!({ "type": "cutover_started", "plan_id": plan_id })
        }
        MigrationEvent::CutoverDone { plan_id } => {
            json!({ "type": "cutover_done", "plan_id": plan_id })
        }
        MigrationEvent::RollbackStarted { plan_id } => {
            json!({ "type": "rollback_started", "plan_id": plan_id })
        }
        MigrationEvent::StatusChanged { plan_id, status } => {
            json!({ "type": "status_changed", "plan_id": plan_id, "status": status_str(*status) })
        }
        MigrationEvent::Failed { plan_id, message } => {
            json!({ "type": "failed", "plan_id": plan_id, "message": message })
        }
    }
}

fn status_str(s: MigrationStatus) -> &'static str {
    match s {
        MigrationStatus::Pending => "Pending",
        MigrationStatus::Running => "Running",
        MigrationStatus::Completed => "Completed",
        MigrationStatus::Cancelled => "Cancelled",
        MigrationStatus::Failed => "Failed",
        MigrationStatus::AwaitingConverter => "AwaitingConverter",
        MigrationStatus::DryRunCompleted => "DryRunCompleted",
    }
}

fn str_to_status(s: &str) -> Option<MigrationStatus> {
    Some(match s.to_ascii_lowercase().as_str() {
        "pending" => MigrationStatus::Pending,
        "running" => MigrationStatus::Running,
        "completed" => MigrationStatus::Completed,
        "cancelled" | "canceled" => MigrationStatus::Cancelled,
        "failed" => MigrationStatus::Failed,
        "awaitingconverter" => MigrationStatus::AwaitingConverter,
        "dryruncompleted" | "dryrun" => MigrationStatus::DryRunCompleted,
        _ => return None,
    })
}

fn phase_str(p: MigrationPhase) -> &'static str {
    match p {
        MigrationPhase::Converting => "Converting",
        MigrationPhase::CuttingOver => "CuttingOver",
        MigrationPhase::RollingBack => "RollingBack",
    }
}

fn policy_str(p: ErrorPolicy) -> &'static str {
    match p {
        ErrorPolicy::Stop => "stop",
        ErrorPolicy::SkipAndLog => "skip",
        ErrorPolicy::Quarantine => "quarantine",
    }
}

fn policy_from_str(s: &str) -> Option<ErrorPolicy> {
    Some(match s.to_ascii_lowercase().as_str() {
        "stop" => ErrorPolicy::Stop,
        "skip" | "skipandlog" => ErrorPolicy::SkipAndLog,
        "quarantine" => ErrorPolicy::Quarantine,
        _ => return None,
    })
}

/// Map an SDL scalar-type name to a `FieldType`. Migrations only ever target a
/// scalar kind, so relations/vectors are deliberately not accepted.
fn field_type_from_str(name: &str) -> Option<FieldType> {
    let scalar = match name {
        "String" => ScalarType::String,
        "u32" => ScalarType::U32,
        "u64" => ScalarType::U64,
        "i32" => ScalarType::I32,
        "i64" => ScalarType::I64,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        "Bool" => ScalarType::Bool,
        "DateTime" => ScalarType::DateTime,
        "Bytes" => ScalarType::Bytes,
        "Json" => ScalarType::Json,
        _ => return None,
    };
    Some(FieldType::Scalar(scalar))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_cache::QueryCache;
    use rhypedb_engine::database::Database;
    use rhypedb_engine::object::{FieldMap, Value};
    use rhypedb_schema::parser::parse_schema;

    /// Build the admin app over a fresh temp DB with `n` User rows, serve it on an
    /// ephemeral port, and return the base URL. The temp dir + server task are
    /// leaked for the lifetime of the test process (fine for a unit test).
    async fn spawn(token: Option<&str>, rows: i64) -> String {
        spawn_app(token, rows).await.0
    }

    /// Like [`spawn`] but also returns the `Arc<AppState>` so a test can inspect
    /// the live handle directly (e.g. assert it hot-reloaded). The temp dir +
    /// server task are leaked for the test process lifetime.
    async fn spawn_app(token: Option<&str>, rows: i64) -> (String, Arc<AppState>) {
        let dir = tempfile::tempdir().unwrap();
        let sdl = r#"type User { score: i64 }"#;
        let schema_path = dir.path().join("schema.rhype");
        std::fs::write(&schema_path, sdl).unwrap();
        let db = Database::open(parse_schema(sdl).unwrap(), dir.path()).unwrap();
        crate::converters::register_builtins(&db);
        for i in 0..rows {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let data_dir = dir.path().to_path_buf();
        std::mem::forget(dir); // keep the data dir alive
        let state = Arc::new(AppState {
            db: arc_swap::ArcSwap::from(db),
            vectorizer: None,
            query_cache: QueryCache::new(16),
            admin_token: token.map(|s| s.to_string()),
            reload_lock: tokio::sync::RwLock::new(()),
            pending_reload_schemas: std::sync::Mutex::new(std::collections::HashMap::new()),
            data_dir,
            schema_path,
            default_ef: None,
            default_rerank: None,
        });
        let app = Router::new()
            .merge(admin_router(state.clone()))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), state)
    }

    /// The live handle's declared type for `Type.field`.
    fn field_kind(state: &Arc<AppState>, ty: &str, field: &str) -> FieldType {
        state
            .db()
            .schema()
            .get_type(ty)
            .unwrap()
            .get_field(field)
            .unwrap()
            .field_type
            .clone()
    }

    /// AC `admin_endpoints_return_403_when_token_unset`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_endpoints_return_403_when_token_unset() {
        let base = spawn(None, 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("{base}/admin/migrations"))
                .call()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(403));
    }

    /// AC `admin_endpoints_return_401_on_token_mismatch`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_endpoints_return_401_on_token_mismatch() {
        let base = spawn(Some("s3cret"), 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("{base}/admin/migrations"))
                .header("Authorization", "Bearer wrong")
                .call()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(401));
    }

    /// Auth also gates the POST routes (not just GET): unset token → 403, wrong
    /// token → 401 on POST /admin/migrations/:id/cancel.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_post_routes_are_auth_gated() {
        let base_403 = spawn(None, 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_403}/admin/migrations/1/cancel"))
                .send_empty()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(403), "POST with no server token must be 403");

        let base_401 = spawn(Some("s3cret"), 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_401}/admin/migrations/1/cancel"))
                .header("Authorization", "Bearer wrong")
                .send_empty()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(401), "POST with wrong token must be 401");
    }

    /// `/admin/compact` is operational (mutating + expensive: force-flush +
    /// full compaction) and must be gated like the migration routes — it was
    /// previously mounted on the OPEN router, so anyone could trigger a
    /// compaction unauthenticated. Unset token → 403, wrong token → 401, correct
    /// token → 200 with the handler actually running.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_compact_is_auth_gated() {
        // No server token → admin disabled → 403.
        let base_403 = spawn(None, 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_403}/admin/compact"))
                .send_empty()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(403), "compact with no server token must be 403");

        // Wrong token → 401.
        let base_401 = spawn(Some("s3cret"), 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_401}/admin/compact"))
                .header("Authorization", "Bearer wrong")
                .send_empty()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(401), "compact with wrong token must be 401");

        // Correct token → 200, and the handler runs end-to-end (flush + compact)
        // — proves moving the route into admin_router didn't break it.
        let base_ok = spawn(Some("tok"), 4).await;
        let body = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_ok}/admin/compact"))
                .header("Authorization", "Bearer tok")
                .send_empty()
                .unwrap()
                .body_mut()
                .read_json::<serde_json::Value>()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(body["flush_ok"], true, "compact must run with a valid token");
        assert_eq!(body["compact_ok"], true);
    }

    #[test]
    fn reap_removes_orphaned_stream_temp_dirs_only() {
        let dir = tempfile::tempdir().unwrap();
        // An orphaned streaming-backup temp dir + real data alongside it.
        std::fs::create_dir_all(dir.path().join(".rhypedb-backup-stream-123-456").join("sst"))
            .unwrap();
        std::fs::create_dir_all(dir.path().join("sst")).unwrap();
        std::fs::write(dir.path().join("sst").join("00000001.sst"), b"x").unwrap();
        std::fs::write(dir.path().join("wal.log"), b"x").unwrap();

        reap_backup_temp_dirs(dir.path());

        assert!(
            !dir.path().join(".rhypedb-backup-stream-123-456").exists(),
            "orphaned temp dir must be reaped"
        );
        assert!(
            dir.path().join("sst").join("00000001.sst").exists(),
            "real data must be untouched"
        );
        assert!(dir.path().join("wal.log").exists());
    }

    /// `/admin/backup` is gated like the rest of the admin surface.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backup_endpoints_are_auth_gated() {
        let base_403 = spawn(None, 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_403}/admin/backup"))
                .send_json(serde_json::json!({ "dest": "/tmp/nope" }))
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(403), "backup with no server token must be 403");

        let base_401 = spawn(Some("s3cret"), 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_401}/admin/backup"))
                .header("Authorization", "Bearer wrong")
                .send_json(serde_json::json!({ "dest": "/tmp/nope" }))
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(401), "backup with wrong token must be 401");
    }

    /// `POST /admin/backup` writes a complete, openable snapshot dir.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backup_path_writes_openable_snapshot() {
        let (base, _state) = spawn_app(Some("tok"), 3).await;
        let dest = tempfile::tempdir().unwrap();
        let dest_str = dest.path().to_string_lossy().to_string();
        let b2 = base.clone();
        let resp: serde_json::Value = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b2}/admin/backup"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({ "dest": dest_str, "label": "nightly" }))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["manifest"]["schema_file"], "schema.rhype");
        assert!(
            !resp["manifest"]["ssts"].as_array().unwrap().is_empty(),
            "manifest lists at least one SST"
        );
        let path = std::path::PathBuf::from(resp["path"].as_str().unwrap());
        assert!(path.join("MANIFEST.json").is_file());
        assert!(path.join("schema.rhype").is_file());
        assert!(path.join("sst").is_dir());
        // The snapshot opens as an independent database with the 3 rows.
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            &path,
        )
        .unwrap();
        assert_eq!(
            db.get("User", 1).unwrap().fields.get("score"),
            Some(&Value::I64(0))
        );
        std::mem::forget(dest);
    }

    /// `GET /admin/backup/stream` streams a tar that unpacks to an openable dir.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backup_stream_returns_openable_tar() {
        let (base, _state) = spawn_app(Some("tok"), 2).await;
        let b2 = base.clone();
        let bytes: Vec<u8> = tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut resp = ureq::get(&format!("{b2}/admin/backup/stream"))
                .header("Authorization", "Bearer tok")
                .call()
                .unwrap();
            let mut buf = Vec::new();
            resp.body_mut().as_reader().read_to_end(&mut buf).unwrap();
            buf
        })
        .await
        .unwrap();
        assert!(!bytes.is_empty(), "tar stream is non-empty");

        let out = tempfile::tempdir().unwrap();
        tar::Archive::new(&bytes[..]).unpack(out.path()).unwrap();
        assert!(out.path().join("MANIFEST.json").is_file());
        assert!(out.path().join("schema.rhype").is_file());
        assert!(out.path().join("sst").is_dir());
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            out.path(),
        )
        .unwrap();
        assert_eq!(
            db.get("User", 1).unwrap().fields.get("score"),
            Some(&Value::I64(0))
        );
    }

    /// `POST /admin/import/stream` applies a streamed dump to the LIVE database.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn online_import_via_http_lands_data() {
        let (base, state) = spawn_app(Some("tok"), 0).await; // empty live server
        // Build an export of 3 User rows from a source DB (same schema).
        let src_dir = tempfile::tempdir().unwrap();
        let src = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            src_dir.path(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for i in 0..3i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(100 + i));
            ids.push(src.create("User", f).unwrap().id);
        }
        let mut buf = Vec::new();
        src.logical_export_stream(
            &mut buf,
            &rhypedb_engine::logical::LogicalExportOptions::default(),
        )
        .unwrap();
        drop(src);

        let url = format!("{base}/admin/import/stream");
        let resp: serde_json::Value = tokio::task::spawn_blocking(move || {
            ureq::post(&url)
                .header("Authorization", "Bearer tok")
                .header("Content-Type", "application/x-ndjson")
                .send(&buf[..])
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["objects"].as_u64(), Some(3));

        // The 3 imported users are now in the LIVE database.
        for id in ids {
            let obj = state.db().get("User", id).unwrap();
            assert!(matches!(obj.fields.get("score"), Some(Value::I64(_))));
        }
    }

    /// `GET /admin/export/stream` returns a complete, well-ordered NDJSON dump.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_stream_returns_complete_ndjson() {
        let (base, _state) = spawn_app(Some("tok"), 3).await;
        let b2 = base.clone();
        let text: String = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("{b2}/admin/export/stream"))
                .header("Authorization", "Bearer tok")
                .call()
                .unwrap()
                .body_mut()
                .read_to_string()
                .unwrap()
        })
        .await
        .unwrap();

        let lines: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["kind"], "header");
        assert_eq!(lines[0]["format"], "rhypedb-logical-export-v1");
        assert_eq!(lines[0]["vectors"], "raw");
        assert_eq!(lines[1]["kind"], "schema");
        assert!(parse_schema(lines[1]["sdl"].as_str().unwrap()).is_ok());
        let objects = lines.iter().filter(|l| l["kind"] == "object").count();
        assert_eq!(objects, 3, "3 User rows exported");
        let trailer = lines.last().unwrap();
        assert_eq!(trailer["kind"], "trailer");
        assert_eq!(trailer["complete"], true);
        assert_eq!(trailer["counts"]["User"]["objects"], 3);
    }

    /// Both export endpoints are behind the admin-token gate (GET + POST).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_endpoints_are_auth_gated() {
        let status_of = |url: String, header: Option<&'static str>| async move {
            tokio::task::spawn_blocking(move || {
                let mut req = ureq::get(&url);
                if let Some(h) = header {
                    req = req.header("Authorization", h);
                }
                req.call().err().and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
            })
            .await
            .unwrap()
        };

        let base_403 = spawn(None, 0).await;
        assert_eq!(
            status_of(format!("{base_403}/admin/export/stream"), None).await,
            Some(403),
            "GET stream with no server token must be 403"
        );
        let base_401 = spawn(Some("s3cret"), 0).await;
        assert_eq!(
            status_of(format!("{base_401}/admin/export/stream"), Some("Bearer wrong")).await,
            Some(401),
            "GET stream with wrong token must be 401"
        );

        let base_403p = spawn(None, 0).await;
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_403p}/admin/export"))
                .send_json(serde_json::json!({ "dest": "/tmp/nope" }))
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(403), "POST export with no server token must be 403");
    }

    /// `POST /admin/export` writes a complete, verifiable export file.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_path_writes_verifiable_file() {
        let (base, _state) = spawn_app(Some("tok"), 2).await;
        let dest = tempfile::tempdir().unwrap();
        let dest_str = dest.path().to_string_lossy().to_string();
        let b2 = base.clone();
        let resp: serde_json::Value = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b2}/admin/export"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({ "dest": dest_str, "label": "nightly" }))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["manifest"]["complete"], true);
        assert_eq!(resp["manifest"]["counts"]["User"]["objects"], 2);
        let path = std::path::PathBuf::from(resp["path"].as_str().unwrap());
        assert!(path.join("MANIFEST.json").is_file());
        assert!(path.join("export.ndjson").is_file());
        std::mem::forget(dest);
    }

    /// An unknown vectors mode is a clean 400; an unknown type a clean 404 —
    /// both BEFORE the streaming 200 commits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_stream_rejects_bad_params_before_streaming() {
        let base = spawn(Some("tok"), 0).await;
        let b2 = base.clone();
        let bad_vectors = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("{b2}/admin/export/stream?vectors=bogus"))
                .header("Authorization", "Bearer tok")
                .call()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(bad_vectors, Some(400), "unknown vectors mode → 400");

        let unknown_type = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("{base}/admin/export/stream?types=Nope"))
                .header("Authorization", "Bearer tok")
                .call()
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(unknown_type, Some(404), "unknown type → 404");
    }

    /// HTTP round-trip: start a migration, then read its detail — exercises the
    /// auth-passing path + the start/detail handlers + JSON shapes end-to-end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_start_and_detail_round_trip() {
        let base = spawn(Some("tok"), 20).await;
        let body = tokio::task::spawn_blocking(move || {
            // start
            let start: serde_json::Value = ureq::post(&format!("{base}/admin/migrations"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({
                    "type": "User", "field": "score", "to": "f64",
                    "converter": "widen_int_to_f64", "converter_version": 1, "chunk": 4
                }))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap();
            let id = start["migration_id"].as_u64().unwrap();
            // detail (poll until terminal-ish or a few tries)
            let mut last = serde_json::Value::Null;
            for _ in 0..50 {
                last = ureq::get(&format!("{base}/admin/migrations/{id}"))
                    .header("Authorization", "Bearer tok")
                    .call()
                    .unwrap()
                    .body_mut()
                    .read_json()
                    .unwrap();
                if last["status"] == "Completed" {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            last
        })
        .await
        .unwrap();
        assert_eq!(body["type"], "User");
        assert_eq!(body["field"], "score");
        assert_eq!(body["status"], "Completed");
        assert_eq!(body["total_objects"].as_u64().unwrap(), 20);
    }

    /// The headline feature: after an online migration completes, the live server
    /// handle hot-reloads to the post-cutover schema WITHOUT a restart. Drives a
    /// real i64→f64 migration over the admin HTTP API, then asserts the in-memory
    /// schema kind flipped on its own.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hotreload_auto_swaps_handle_after_migration_completes() {
        let (base, state) = spawn_app(Some("tok"), 12).await;
        assert_eq!(
            field_kind(&state, "User", "score"),
            FieldType::Scalar(ScalarType::I64),
            "live handle starts at the source kind"
        );

        let b2 = base.clone();
        tokio::task::spawn_blocking(move || {
            let start: serde_json::Value = ureq::post(&format!("{b2}/admin/migrations"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({
                    "type": "User", "field": "score", "to": "f64",
                    "converter": "widen_int_to_f64", "converter_version": 1, "chunk": 4
                }))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap();
            let id = start["migration_id"].as_u64().unwrap();
            for _ in 0..100 {
                let last: serde_json::Value = ureq::get(&format!("{b2}/admin/migrations/{id}"))
                    .header("Authorization", "Bearer tok")
                    .call()
                    .unwrap()
                    .body_mut()
                    .read_json()
                    .unwrap();
                if last["status"] == "Completed" {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            panic!("migration did not complete");
        })
        .await
        .unwrap();

        // The completion watcher hot-reloads the live handle to f64. It fires
        // asynchronously after the Completed event, so poll for it.
        let mut reloaded = false;
        for _ in 0..150 {
            if field_kind(&state, "User", "score") == FieldType::Scalar(ScalarType::F64) {
                reloaded = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(reloaded, "live handle should hot-reload to f64 with no restart");
        assert!(
            state.pending_reload_schemas.lock().unwrap().is_empty(),
            "the per-plan target schema is consumed once the reload lands"
        );
    }

    /// Regression for the subscribe-after-complete race: with an EMPTY table the
    /// migration settles almost instantly — likely BEFORE the watcher subscribes.
    /// The subscribe-then-recheck path must still hot-reload the live handle (a
    /// pure event-driven watcher would silently miss it and leave the handle
    /// stale forever).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hotreload_auto_fires_when_migration_completes_before_subscribe() {
        let (base, state) = spawn_app(Some("tok"), 0).await; // empty table
        assert_eq!(
            field_kind(&state, "User", "score"),
            FieldType::Scalar(ScalarType::I64)
        );

        let b2 = base.clone();
        tokio::task::spawn_blocking(move || {
            let start: serde_json::Value = ureq::post(&format!("{b2}/admin/migrations"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({
                    "type": "User", "field": "score", "to": "f64",
                    "converter": "widen_int_to_f64", "converter_version": 1
                }))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap();
            let id = start["migration_id"].as_u64().unwrap();
            for _ in 0..100 {
                let last: serde_json::Value = ureq::get(&format!("{b2}/admin/migrations/{id}"))
                    .header("Authorization", "Bearer tok")
                    .call()
                    .unwrap()
                    .body_mut()
                    .read_json()
                    .unwrap();
                if last["status"] == "Completed" {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            panic!("migration did not complete");
        })
        .await
        .unwrap();

        let mut reloaded = false;
        for _ in 0..150 {
            if field_kind(&state, "User", "score") == FieldType::Scalar(ScalarType::F64) {
                reloaded = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            reloaded,
            "empty-table migration must still hot-reload via subscribe-then-recheck"
        );
    }

    /// `POST /admin/reload` with an unparseable SDL → 400, and the server keeps
    /// serving on the untouched handle (no brick). A subsequent valid reload to
    /// the same schema succeeds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reload_rejects_bad_sdl_and_stays_live() {
        let (base, state) = spawn_app(Some("tok"), 3).await;

        let b2 = base.clone();
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b2}/admin/reload"))
                .header("Authorization", "Bearer tok")
                .send("type User { this is not valid")
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(400));
        // Untouched + live.
        assert_eq!(
            field_kind(&state, "User", "score"),
            FieldType::Scalar(ScalarType::I64)
        );

        let b3 = base.clone();
        let ok = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b3}/admin/reload"))
                .header("Authorization", "Bearer tok")
                .send("type User { score: i64 }")
                .is_ok()
        })
        .await
        .unwrap();
        assert!(ok, "a valid reload with no migration in flight succeeds");
    }

    /// `POST /admin/reload` is refused with 409 while a migration is in flight —
    /// reloading then would silently disarm the double-write hook (data loss). A
    /// converter that blocks on the first row holds the plan provably armed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reload_returns_409_while_migration_in_flight() {
        use rhypedb_engine::object::Value as EngineValue;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;

        let (base, state) = spawn_app(Some("tok"), 3).await;
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let entered_tx = std::sync::Mutex::new(entered_tx);
        let release_rx = std::sync::Mutex::new(release_rx);
        let blocked = AtomicBool::new(false);
        state.db().register_converter("blocker", 1, move |_oid, v| {
            if !blocked.swap(true, Ordering::SeqCst) {
                entered_tx.lock().unwrap().send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
            match v {
                EngineValue::I64(i) => Ok(EngineValue::F64(*i as f64)),
                _ => unreachable!(),
            }
        });

        let b2 = base.clone();
        tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b2}/admin/migrations"))
                .header("Authorization", "Bearer tok")
                .send_json(serde_json::json!({
                    "type": "User", "field": "score", "to": "f64",
                    "converter": "blocker", "converter_version": 1, "chunk": 1, "parallel": 1
                }))
                .unwrap()
                .body_mut()
                .read_json::<serde_json::Value>()
                .unwrap();
        })
        .await
        .unwrap();
        // Wait until a worker is inside the converter → the plan is armed.
        tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
            .await
            .unwrap();

        let b3 = base.clone();
        let code = tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{b3}/admin/reload"))
                .header("Authorization", "Bearer tok")
                .send("type User { score: f64 }")
                .err()
                .and_then(|e| match e {
                    ureq::Error::StatusCode(c) => Some(c),
                    _ => None,
                })
        })
        .await
        .unwrap();
        assert_eq!(code, Some(409), "reload must be refused mid-migration");

        // Unblock so the migration (and its background driver) can finish.
        release_tx.send(()).unwrap();
    }
}
