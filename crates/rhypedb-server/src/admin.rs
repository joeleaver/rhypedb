//! Card 5: HTTP admin surface for field-type migrations.
//!
//! Routes live under `/admin/migrations*` and are gated by a single static
//! token (`RHYPEDB_ADMIN_TOKEN`, read once at startup): unset → 403, mismatch →
//! 401. A quick safety net; real RBAC is a separate epic. Engine calls are
//! blocking, so each handler hops onto `spawn_blocking` (the established pattern
//! from `handle_admin_compact`). Responses are built with `serde_json::json!`
//! over the engine's public structs so the engine stays serde-free.

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
    MigrationEvent, MigrationFilter, MigrationPlanSpec, MigrationProgress, MigrationSummary,
    QuarantineEntry,
};
use rhypedb_engine::{EngineError, ErrorPolicy, MigrationPhase, MigrationStatus};
use rhypedb_schema::{FieldType, ScalarType};

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
        // Auth applies to these routes only (NOT /query, /health, /admin/compact).
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
    let db = state.db.clone();
    match spawn_blocking_engine(move || db.start_field_type_migration_async(spec)).await {
        Ok(handle) => (
            StatusCode::OK,
            Json(json!({
                "migration_id": handle.plan_id,
                "created_at_ms": handle.created_at_ms,
            })),
        )
            .into_response(),
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
    let db = state.db.clone();
    match spawn_blocking_engine(move || db.list_migrations_filtered(&filter)).await {
        Ok(rows) => {
            let arr: Vec<JsonValue> = rows.iter().map(summary_json).collect();
            (StatusCode::OK, Json(json!({ "migrations": arr }))).into_response()
        }
        Err(e) => err_response(e),
    }
}

async fn detail(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let db = state.db.clone();
    match spawn_blocking_engine(move || db.query_migration_progress(id)).await {
        Ok(p) => (StatusCode::OK, Json(progress_json(&p))).into_response(),
        Err(e) => err_response(e),
    }
}

async fn pause(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.pause_migration(id)).await
}

async fn resume(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.resume_field_type_migration(id)).await
}

async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.cancel_migration(id)).await
}

async fn cutover(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    verb(state, move |db| db.cutover_migration(id)).await
}

async fn quarantine(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let db = state.db.clone();
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
    let db = state.db.clone();
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
    let engine_rx = state.db.subscribe_migration_events(id);
    let (tx, rx) = tokio::sync::mpsc::channel::<MigrationEvent>(64);
    tokio::task::spawn_blocking(move || loop {
        match engine_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(ev) => {
                let terminal = matches!(
                    ev,
                    MigrationEvent::CutoverDone { .. }
                        | MigrationEvent::Failed { .. }
                        | MigrationEvent::StatusChanged {
                            status: MigrationStatus::Cancelled
                                | MigrationStatus::Completed
                                | MigrationStatus::DryRunCompleted,
                            ..
                        }
                );
                if tx.blocking_send(ev).is_err() {
                    break; // client disconnected
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
// Shared helpers
// ---------------------------------------------------------------------------

/// Run a unit-returning engine verb on a blocking thread, mapping the result to
/// `{"ok": true}` / an error response.
async fn verb<F>(state: Arc<AppState>, f: F) -> Response
where
    F: FnOnce(&rhypedb_engine::database::Database) -> Result<(), EngineError> + Send + 'static,
{
    let db = state.db.clone();
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

fn err_response(e: EngineError) -> Response {
    let code = match &e {
        EngineError::MigrationPlanNotFound { .. } | EngineError::TypeNotFound(_) => {
            StatusCode::NOT_FOUND
        }
        EngineError::MigrationCancelledCannotCutover { .. }
        | EngineError::MigrationCannotCancelInCutover { .. }
        | EngineError::MigrationCannotCancelSettled { .. }
        | EngineError::MigrationAlreadyRunning { .. }
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
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        crate::converters::register_builtins(&db);
        for i in 0..rows {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        std::mem::forget(dir); // keep the data dir alive
        let state = Arc::new(AppState {
            db,
            vectorizer: None,
            query_cache: QueryCache::new(16),
            admin_token: token.map(|s| s.to_string()),
        });
        let app = Router::new()
            .merge(admin_router(state.clone()))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
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
}
