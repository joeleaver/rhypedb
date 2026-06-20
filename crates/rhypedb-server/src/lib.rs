//! The rhypedb server, exposed as a library so it can be embedded by a thin
//! binary (this crate's own `main.rs`) or by another crate that just wants to
//! launch the server (e.g. an app that builds it via a buildpack). The actual
//! entry point is [`run`]; the `#[global_allocator]` lives in the binary crate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use arc_swap::ArcSwap;
use rhypedb_engine::database::Database;
use rhypedb_engine::object::{Object, Value};
use rhypedb_engine::vectorizer::Vectorizer;
use rhypedb_query::executor::{ExecContext, QueryOutput};
use rhypedb_schema::parser::parse_schema;
use rhypedb_schema::{
    DistanceMetric, FieldDef, FieldType, IndexDef, OnDeletePolicy, QuantizationType, ScalarType,
    Schema,
};

mod admin;
mod converters;
pub mod import;
mod protocol;
mod query_cache;
mod restore;

use query_cache::QueryCache;

#[derive(Parser)]
#[command(name = "rhypedb", about = "rhypedb database server")]
struct Cli {
    /// Path to the SDL schema file. Required, EXCEPT with `--restore-from`: a
    /// restored data dir carries its own authoritative `schema.rhype`, so the
    /// flag is optional there (and, if given, must match the snapshot's schema).
    #[arg(short, long)]
    schema: Option<PathBuf>,

    /// Data directory for storage.
    #[arg(short, long, default_value = "./rhypedb-data")]
    data_dir: PathBuf,

    /// Restore a physical backup snapshot into `--data-dir` BEFORE serving, then
    /// open it. The snapshot's `schema.rhype` becomes authoritative. Idempotent:
    /// a restart with the same snapshot already in place is a no-op (so this can
    /// stay set across restarts). Also settable via `RHYPEDB_RESTORE_FROM`.
    #[arg(long)]
    restore_from: Option<PathBuf>,

    /// Allow `--restore-from` to overwrite an existing, different database in
    /// `--data-dir`. Not needed to re-apply the same snapshot (that's a no-op) or
    /// to clear a stale LOCK. Also settable via `RHYPEDB_RESTORE_FROM_FORCE=1`.
    #[arg(long)]
    restore_force: bool,

    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:4200")]
    listen: String,

    /// Binary TCP listen address.
    #[arg(long, default_value = "127.0.0.1:4201")]
    tcp_listen: String,

    /// Skip the WAL fsync at commit time — kernel still has the bytes via
    /// write_all, so clean process crashes are recoverable, but a power
    /// loss can drop the last N writes. Matches Postgres's
    /// `fsync=off + synchronous_commit=off` mode (used by the bench
    /// harness). Off by default for safety.
    #[arg(long)]
    no_sync: bool,
}

pub(crate) struct AppState {
    /// The live database handle. Wrapped in `ArcSwap` so an in-place hot-reload
    /// (after a `change_field_type` migration cuts over, or via `/admin/reload`)
    /// can swap in a fresh handle on the SAME storage under the post-cutover
    /// schema — no process restart. Read it via [`AppState::db`].
    pub(crate) db: ArcSwap<Database>,
    vectorizer: Option<Arc<Vectorizer>>,
    query_cache: QueryCache,
    /// Card 5: the `RHYPEDB_ADMIN_TOKEN` env value, read ONCE at startup.
    /// `None` → the `/admin/migrations*` routes return 403 (admin disabled);
    /// `Some` → a request must present a matching `Authorization: Bearer <token>`
    /// or get 401. A quick safety net; real RBAC is a separate epic.
    pub(crate) admin_token: Option<String>,
    /// Schema-epoch lock for in-place hot-reload. Every operation that uses the
    /// handle for schema-driven work (query execute, migration `start`) takes
    /// `.read()` for that ONE operation; a reload takes `.write()`, which drains
    /// in-flight readers, swaps `db`, then releases — so nothing straddles the
    /// swap. Uncontended in steady state (writers appear only on reload).
    pub(crate) reload_lock: tokio::sync::RwLock<()>,
    /// Target schemas captured at migration-create time, keyed by `plan_id`, so
    /// the per-plan completion watcher can hot-reload to the post-cutover schema
    /// without reconstructing it from the catalog (no `catalog → Schema` exists).
    /// Entries are removed when the plan settles.
    pub(crate) pending_reload_schemas: std::sync::Mutex<HashMap<u64, Schema>>,
    /// The `--data-dir` the engine opened. Used by `/admin/backup/stream` to
    /// place the temporary snapshot dir on the SAME filesystem (so SSTs hard-link).
    pub(crate) data_dir: std::path::PathBuf,
    /// The `--schema` SDL file. There is no `Schema → SDL` serializer, so a backup
    /// copies this file in so the restored data dir is openable.
    pub(crate) schema_path: std::path::PathBuf,
    /// Server-wide default HNSW search width (`ef`) for `.similar` queries that
    /// omit `ef:`, read ONCE from `RHYPEDB_EF` at startup. `None` = use the
    /// engine's per-shape heuristic. Threaded into every query's `ExecContext`.
    pub(crate) default_ef: Option<usize>,
    /// Server-wide default rerank pool size for `.similar` queries that omit
    /// `rerank:`, read ONCE from `RHYPEDB_RERANK`. `None` = no full-precision
    /// rerank by default. A per-query `ef:`/`rerank:` always overrides these.
    pub(crate) default_rerank: Option<usize>,
}

impl AppState {
    /// Load the current database handle (a cheap `ArcSwap` load). Operations that
    /// do schema-driven work should hold [`AppState::reload_lock`]`.read()` around
    /// the load + use so a hot-reload cannot swap the handle mid-operation.
    pub(crate) fn db(&self) -> Arc<Database> {
        self.db.load_full()
    }
}

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
}

#[derive(Serialize)]
struct QueryResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    objects: Option<Vec<ObjectJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    object: Option<ObjectJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct ObjectJson {
    #[serde(rename = "type")]
    type_name: String,
    id: u64,
    fields: HashMap<String, serde_json::Value>,
}

impl From<Object> for ObjectJson {
    fn from(mut obj: Object) -> Self {
        // HTTP/JSON response needs the decoded FieldMap; honor the lazy-
        // deserialize shortcut populated by `Database::get_many`.
        obj.ensure_fields_deserialized();
        let fields = obj
            .fields
            .into_iter()
            .map(|(k, v)| (k, value_to_json(v)))
            .collect();
        ObjectJson {
            type_name: obj.type_name,
            id: obj.id,
            fields,
        }
    }
}

fn value_to_json(v: Value) -> serde_json::Value {
    match v {
        Value::String(s) => serde_json::Value::String(s),
        Value::U32(n) => serde_json::json!(n),
        Value::U64(n) => serde_json::json!(n),
        Value::I32(n) => serde_json::json!(n),
        Value::I64(n) => serde_json::json!(n),
        Value::F32(n) => serde_json::json!(n),
        Value::F64(n) => serde_json::json!(n),
        Value::Bool(b) => serde_json::Value::Bool(b),
        Value::Bytes(b) => serde_json::json!(format!("<{} bytes>", b.len())),
        Value::Null => serde_json::Value::Null,
    }
}

/// Parse the `RHYPEDB_EF` / `RHYPEDB_RERANK` env values into server-wide
/// `.similar` defaults. Both are optional tuning knobs: an absent, empty,
/// non-integer, or out-of-range value is IGNORED with a warning (the server
/// must not refuse to start over a fat-fingered tuning hint). `ef` must be
/// `>= 1` (matching the per-query parser); `rerank: 0` means "off" and is
/// normalised to `None` (no default rerank). Takes the raw strings so it is
/// unit-testable without mutating the process environment.
fn parse_vector_search_defaults(
    ef: Option<&str>,
    rerank: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    fn parse_knob(name: &str, raw: Option<&str>, min: usize) -> Option<usize> {
        let raw = raw.map(str::trim).filter(|s| !s.is_empty())?;
        match raw.parse::<usize>() {
            Ok(n) if n >= min => Some(n),
            Ok(n) => {
                eprintln!(
                    "WARNING: {name}={n} is below the minimum of {min}; ignoring it."
                );
                None
            }
            Err(_) => {
                eprintln!(
                    "WARNING: {name}=\"{raw}\" is not a non-negative integer; ignoring it."
                );
                None
            }
        }
    }
    // ef must be >= 1 (a width of 0 explores nothing). rerank accepts 0 = off,
    // which collapses to "no default rerank".
    let default_ef = parse_knob("RHYPEDB_EF", ef, 1);
    let default_rerank = parse_knob("RHYPEDB_RERANK", rerank, 0).filter(|&n| n > 0);
    (default_ef, default_rerank)
}

async fn handle_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> (StatusCode, Json<QueryResponse>) {
    let query = match state.query_cache.get_or_parse(&req.query) {
        Ok(q) => q,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(QueryResponse {
                    objects: None,
                    object: None,
                    ok: None,
                    error: Some(format!("parse error: {e}")),
                }),
            );
        }
    };

    // Hold the schema-epoch read guard only across execute (the schema-driven
    // work); a hot-reload (write guard) can't swap the handle mid-query.
    // Materialized results decode self-describingly afterward, so the guard need
    // not extend over response building.
    let result = {
        let _epoch = state.reload_lock.read().await;
        let db = state.db();
        let ctx = ExecContext {
            db: &db,
            vectorizer: state.vectorizer.as_deref(),
            default_ef: state.default_ef,
            default_rerank: state.default_rerank,
        };
        rhypedb_query::executor::execute(&ctx, &query)
    };

    match result {
        Ok(QueryOutput::Objects(objs)) => (
            StatusCode::OK,
            Json(QueryResponse {
                objects: Some(objs.into_iter().map(ObjectJson::from).collect()),
                object: None,
                ok: None,
                error: None,
            }),
        ),
        Ok(QueryOutput::Single(obj)) => {
            // Enqueue vectorization for created/updated objects.
            if let Some(vectorizer) = &state.vectorizer {
                enqueue_vectorize(vectorizer, &state.db(), &obj);
            }
            (
                StatusCode::OK,
                Json(QueryResponse {
                    objects: None,
                    object: Some(ObjectJson::from(obj)),
                    ok: None,
                    error: None,
                }),
            )
        }
        Ok(QueryOutput::Done) => (
            StatusCode::OK,
            Json(QueryResponse {
                objects: None,
                object: None,
                ok: Some(true),
                error: None,
            }),
        ),
        // IdSet is an internal traversal carrier; the executor materializes it
        // before returning. Reaching this arm would be a bug in execute().
        Ok(QueryOutput::IdSet { .. }) | Ok(QueryOutput::IdSetWithFields { .. }) => unreachable!(
            "QueryOutput::IdSet variants should be materialized to Objects before leaving execute()"
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(QueryResponse {
                objects: None,
                object: None,
                ok: None,
                error: Some(format!("{e}")),
            }),
        ),
    }
}

fn enqueue_vectorize(vectorizer: &Vectorizer, db: &Database, obj: &Object) {
    let schema = db.schema();
    if let Some(type_def) = schema.get_type(&obj.type_name) {
        for field in &type_def.fields {
            if let Some(vec_def) = field.vectorize() {
                let _ = vectorizer.enqueue(
                    rhypedb_engine::vectorizer::VectorizeJob {
                        type_name: obj.type_name.clone(),
                        object_id: obj.id,
                        source_field: vec_def.source_field.clone(),
                        vector_field: field.name.clone(),
                        model: vec_def.model.clone(),
                    },
                );
            }
        }
    }
}

async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> String {
    format!(
        "ok (subscriptions: {})",
        state.db().subscriptions().subscription_count()
    )
}

/// Force-flush the active memtable + compact all SST files into one.
/// Operational (mutating + expensive), so gated by `RHYPEDB_ADMIN_TOKEN`
/// alongside the migration admin routes — the route is registered inside
/// `admin::admin_router`, not on the open router. For benchmarking and manual
/// triggering; auto-compaction is the proper long-term answer.
pub(crate) async fn handle_admin_compact(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let storage = state.db().storage().clone();
    let t0 = std::time::Instant::now();
    let flush_result = tokio::task::spawn_blocking({
        let storage = storage.clone();
        move || storage.flush()
    })
    .await;
    let flush_ms = t0.elapsed().as_millis();

    let t1 = std::time::Instant::now();
    let compact_result = tokio::task::spawn_blocking({
        let storage = storage.clone();
        move || storage.compact()
    })
    .await;
    let compact_ms = t1.elapsed().as_millis();

    let flush_ok = matches!(&flush_result, Ok(Ok(_)));
    let compact_ok = matches!(&compact_result, Ok(Ok(_)));
    let flush_err = match &flush_result {
        Ok(Err(e)) => Some(e.to_string()),
        Err(e) => Some(e.to_string()),
        Ok(Ok(_)) => None,
    };
    let compact_err = match &compact_result {
        Ok(Err(e)) => Some(e.to_string()),
        Err(e) => Some(e.to_string()),
        Ok(Ok(_)) => None,
    };

    Json(serde_json::json!({
        "flush_ok": flush_ok,
        "flush_ms": flush_ms,
        "flush_error": flush_err,
        "compact_ok": compact_ok,
        "compact_ms": compact_ms,
        "compact_error": compact_err,
    }))
}

async fn handle_status(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut result = serde_json::json!({
        "subscriptions": state.db().subscriptions().subscription_count(),
    });

    if let Some(vectorizer) = &state.vectorizer {
        let status = vectorizer.status();
        let indexes: Vec<serde_json::Value> = status
            .index_stats
            .iter()
            .map(|s| serde_json::json!({ "name": s.name, "vectors": s.vectors }))
            .collect();
        result["vectorizer"] = serde_json::json!({
            "pending": status.pending,
            "indexes": indexes,
        });
    }

    Json(result)
}

/// `GET /schema` — a structured JSON introspection of the LIVE schema (types,
/// fields, directives) plus the canonical SDL. Open + read-only like `/status`:
/// the data plane (`/query`) is already open and you need the schema to use it,
/// and SDK-codegen consumers shouldn't need the operator admin token. Trivially
/// moved behind `admin_router` later if a deployment wants it gated.
async fn handle_schema(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(schema_introspection(state.db().schema()))
}

/// Build the codegen-friendly introspection payload: format-tagged, types sorted
/// by name (HashMap order is nondeterministic), fields in declaration order, plus
/// the canonical SDL (`emit_schema`) so a consumer can parse or round-trip it.
fn schema_introspection(schema: &Schema) -> serde_json::Value {
    let mut names: Vec<&String> = schema.types.keys().collect();
    names.sort();
    let types: Vec<serde_json::Value> = names
        .iter()
        .map(|name| {
            let td = &schema.types[*name];
            let fields: Vec<serde_json::Value> =
                td.fields.iter().map(field_introspection).collect();
            serde_json::json!({ "name": name, "fields": fields })
        })
        .collect();
    serde_json::json!({
        "format": "rhypedb-schema-introspection-v1",
        "types": types,
        "sdl": rhypedb_schema::emit_schema(schema),
    })
}

/// One field → its introspection object. `kind` discriminates the shape: a
/// `scalar` carries `scalar`/`unique`/`indexed`; a `vector` carries `dimensions`
/// plus optional `vectorize`/`index`; a `relation` carries `target`/`many` plus
/// optional `onDelete`/`inverse`/`edgeFields`.
fn field_introspection(f: &FieldDef) -> serde_json::Value {
    let mut field = serde_json::json!({ "name": f.name });
    match &f.field_type {
        FieldType::Scalar(s) => {
            field["kind"] = "scalar".into();
            field["scalar"] = scalar_name(s).into();
            field["unique"] = f.is_unique().into();
            field["indexed"] = f.is_indexed().into();
        }
        FieldType::Vector(v) => {
            field["kind"] = "vector".into();
            field["dimensions"] = v.dimensions.into();
            if let Some(vz) = f.vectorize() {
                field["vectorize"] =
                    serde_json::json!({ "source": vz.source_field, "model": vz.model });
            }
            if let Some(ix) = f.index() {
                field["index"] = index_introspection(ix);
            }
        }
        FieldType::Relation(rel) => {
            field["kind"] = "relation".into();
            field["target"] = rel.target_type.as_str().into();
            field["many"] = rel.is_many.into();
            if let Some(p) = f.on_delete() {
                field["onDelete"] = on_delete_name(p).into();
            }
            if let Some(inv) = f.inverse() {
                field["inverse"] =
                    serde_json::json!({ "type": inv.type_name, "field": inv.field_name });
            }
            if !rel.edge_fields.is_empty() {
                let efs: Vec<serde_json::Value> = rel
                    .edge_fields
                    .iter()
                    .map(|ef| {
                        serde_json::json!({ "name": ef.name, "scalar": scalar_name(&ef.scalar_type) })
                    })
                    .collect();
                field["edgeFields"] = efs.into();
            }
        }
    }
    field
}

fn index_introspection(ix: &IndexDef) -> serde_json::Value {
    // IndexType::Hnsw is the only variant today.
    let mut out = serde_json::json!({ "type": "hnsw" });
    if let Some(m) = &ix.metric {
        out["metric"] = metric_name(m).into();
    }
    if let Some(q) = &ix.quantization {
        out["quantization"] = quantization_name(q).into();
    }
    if let Some(m) = ix.m {
        out["m"] = m.into();
    }
    if let Some(ef) = ix.ef_construction {
        out["efConstruction"] = ef.into();
    }
    out
}

/// Canonical SDL spellings — mirror rhypedb-schema's emitter so the introspection
/// and the embedded SDL agree.
fn scalar_name(s: &ScalarType) -> &'static str {
    match s {
        ScalarType::String => "String",
        ScalarType::U32 => "u32",
        ScalarType::U64 => "u64",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Bool => "Bool",
        ScalarType::DateTime => "DateTime",
        ScalarType::Bytes => "Bytes",
        ScalarType::Json => "Json",
    }
}

fn on_delete_name(p: &OnDeletePolicy) -> &'static str {
    match p {
        OnDeletePolicy::Remove => "remove",
        OnDeletePolicy::Cascade => "cascade",
        OnDeletePolicy::Deny => "deny",
    }
}

fn metric_name(m: &DistanceMetric) -> &'static str {
    match m {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "l2",
        DistanceMetric::DotProduct => "dot_product",
    }
}

fn quantization_name(q: &QuantizationType) -> &'static str {
    match q {
        QuantizationType::TurboQuant2Bit => "turboquant_2bit",
        QuantizationType::TurboQuant3Bit => "turboquant_3bit",
        QuantizationType::TurboQuant4Bit => "turboquant_4bit",
        QuantizationType::None => "none",
    }
}

/// An env var set to a truthy value (`1`/`true`/`yes`/`on`, case-insensitive).
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Read + parse an SDL schema file, exiting the process with a legible message on
/// any error (the only sensible behaviour at startup).
fn read_and_parse_schema(path: &std::path::Path) -> Schema {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("failed to read schema file {path:?}: {e}");
        std::process::exit(1);
    });
    parse_schema(&text).unwrap_or_else(|e| {
        eprintln!("schema error: {e}");
        std::process::exit(1);
    })
}

/// Parse the CLI, open the database, and serve until shutdown. Callers provide
/// the async runtime (this crate's `main` uses `#[tokio::main]`).
pub async fn run() {
    let cli = Cli::parse();

    // Restore-on-boot config. CLI flag wins over env (clap has no `env` feature
    // here, matching the RHYPEDB_ADMIN_TOKEN / EF / RERANK pattern).
    let restore_from = cli
        .restore_from
        .clone()
        .or_else(|| std::env::var_os("RHYPEDB_RESTORE_FROM").map(PathBuf::from));
    let restore_force = cli.restore_force || env_truthy("RHYPEDB_RESTORE_FROM_FORCE");

    // If asked, restore a snapshot into the data dir BEFORE opening it, and take
    // the schema from the restored dir (the snapshot's schema.rhype is
    // authoritative — opening restored SSTs under a different schema would
    // reconcile/mutate the on-disk catalog).
    let schema_path: PathBuf = if let Some(snapshot) = &restore_from {
        match restore::restore_from_snapshot(snapshot, &cli.data_dir, restore_force) {
            Ok(report) => {
                if report.skipped {
                    println!(
                        "restore: {} already holds this snapshot ({}) — serving existing data",
                        cli.data_dir.display(),
                        snapshot.display()
                    );
                } else {
                    println!(
                        "restored {} SSTs + WAL + {} HNSW snapshot(s) from {} into {}",
                        report.sst_count,
                        report.hnsw_count,
                        snapshot.display(),
                        cli.data_dir.display()
                    );
                }
                for (plan_id, converter) in &report.in_flight {
                    eprintln!(
                        "WARNING: restored backup was MID-MIGRATION (plan {plan_id}, converter \
                         {converter}); register that converter before serving."
                    );
                }
            }
            Err(e) => {
                eprintln!("restore failed: {e}");
                std::process::exit(1);
            }
        }
        let restored_schema = cli.data_dir.join("schema.rhype");
        // If --schema was ALSO given, it must match the snapshot's exactly. The
        // snapshot still wins; this is a cheap guard against an operator pointing
        // at the wrong schema.
        if let Some(explicit) = &cli.schema
            && read_and_parse_schema(&restored_schema) != read_and_parse_schema(explicit)
        {
            eprintln!(
                "--schema {explicit:?} does not match the restored snapshot's schema \
                 ({restored_schema:?}); the snapshot's schema is authoritative — omit \
                 --schema or pass the matching file."
            );
            std::process::exit(1);
        }
        restored_schema
    } else {
        cli.schema.clone().unwrap_or_else(|| {
            eprintln!("--schema <path> is required (unless --restore-from is given)");
            std::process::exit(1);
        })
    };

    let schema = read_and_parse_schema(&schema_path);

    let db = Database::open_with_options(
        schema.clone(),
        &cli.data_dir,
        rhypedb_engine::database::OpenOptions {
            sync_on_commit: !cli.no_sync,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("failed to open database: {e}");
        std::process::exit(1);
    });
    if cli.no_sync {
        eprintln!(
            "WARNING: --no-sync is on. WAL writes will not fsync; power loss can drop \
             the last N records. Equivalent to Postgres fsync=off."
        );
    }

    // Card 5: register the built-in named converters so operators can start
    // migrations by name over HTTP/CLI, then resume any in-flight migration left
    // by a prior run (open() armed its hook + completed rollbacks/cutovers that
    // need no converter; a Converting plan needs its converter, now registered).
    converters::register_builtins(&db);
    converters::resume_inflight(&db);

    let has_vectorize = schema
        .types
        .values()
        .any(|td| td.fields.iter().any(|f| f.vectorize().is_some()));
    // Build the vectorizer if the schema has ANY Vector field — bare Vector
    // fields (no @vectorize) hold caller-supplied vectors via the binary
    // vector-batch op and still need their HNSW index. The embed worker only
    // runs when there are @vectorize fields to auto-embed.
    let has_vector_field = schema
        .types
        .values()
        .any(|td| td.vector_fields().next().is_some());

    let vectorizer = if has_vector_field {
        let vectorizer = match Vectorizer::new(
            Arc::clone(db.storage()),
            schema.clone(),
            db.type_ids().clone(),
            db.field_ids().clone(),
        ) {
            Ok(v) => v,
            // A misconfigured vector index (e.g. an invalid `@index` directive)
            // should fail startup with a legible message, not a panic backtrace.
            Err(e) => {
                eprintln!("failed to initialize vector indexes: {e}");
                std::process::exit(1);
            }
        };
        let v = Arc::new(vectorizer);
        if has_vectorize {
            v.start_worker(1);
        }
        Some(v)
    } else {
        None
    };

    // Card 5: read the admin token ONCE. Unset → admin endpoints return 403.
    let admin_token = std::env::var("RHYPEDB_ADMIN_TOKEN").ok().filter(|t| !t.is_empty());
    let admin_enabled = admin_token.is_some();

    // Server-wide `.similar` defaults, read ONCE. A per-query `ef:`/`rerank:`
    // overrides these; an invalid value is warned-about and ignored (above).
    let (default_ef, default_rerank) = parse_vector_search_defaults(
        std::env::var("RHYPEDB_EF").ok().as_deref(),
        std::env::var("RHYPEDB_RERANK").ok().as_deref(),
    );

    let state = Arc::new(AppState {
        db: ArcSwap::from(db),
        vectorizer,
        query_cache: QueryCache::new(query_cache::DEFAULT_CACHE_SIZE),
        admin_token,
        reload_lock: tokio::sync::RwLock::new(()),
        pending_reload_schemas: std::sync::Mutex::new(HashMap::new()),
        data_dir: cli.data_dir.clone(),
        schema_path: schema_path.clone(),
        default_ef,
        default_rerank,
    });

    // Re-register completion watchers for any migration left in flight by a prior
    // run, so a plan that finishes its backfill post-restart still hot-reloads the
    // live handle to the target schema instead of waiting for an operator.
    admin::resume_reload_watchers(&state);

    // Sweep orphaned `.rhypedb-backup-stream-*` temp dirs from a previously
    // hard-killed run (their TempDirGuard never ran) so leaked hard-linked SST
    // inodes don't accumulate on the volume.
    admin::reap_backup_temp_dirs(&cli.data_dir);

    let app = Router::new()
        .route("/query", post(handle_query))
        .route("/status", get(handle_status))
        .route("/health", get(handle_health))
        .route("/schema", get(handle_schema))
        // All admin/operational routes (/admin/compact, /admin/reload,
        // /admin/migrations*) are gated by RHYPEDB_ADMIN_TOKEN inside admin_router.
        .merge(admin::admin_router(state.clone()))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(&cli.listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {}: {e}", cli.listen);
            std::process::exit(1);
        });

    let tcp_listener = TcpListener::bind(&cli.tcp_listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {}: {e}", cli.tcp_listen);
            std::process::exit(1);
        });

    println!("rhypedb HTTP listening on {}", cli.listen);
    println!("rhypedb binary TCP listening on {}", cli.tcp_listen);
    println!("  POST /query     — execute queries");
    println!("  GET  /health    — health check");
    println!("  GET  /schema    — schema introspection (JSON + SDL)");
    if default_ef.is_some() || default_rerank.is_some() {
        println!(
            "  vector .similar defaults: ef={} rerank={} (per-query args override)",
            default_ef.map_or_else(|| "heuristic".to_string(), |n| n.to_string()),
            default_rerank.map_or_else(|| "off".to_string(), |n| n.to_string()),
        );
    }
    if admin_enabled {
        println!("  *    /admin/* (compact, reload, migrations*) — admin (RHYPEDB_ADMIN_TOKEN set)");
        println!(
            "       built-in converters: {}",
            converters::BUILTIN_CONVERTER_NAMES.join(", ")
        );
    } else {
        println!("  *    /admin/* (compact, reload, migrations*) — DISABLED (set RHYPEDB_ADMIN_TOKEN to enable; returns 403)");
    }

    // Serve both listeners until a shutdown signal (the platform's SIGTERM on
    // scale-to-zero, or Ctrl-C), then drain in-flight requests and flush the
    // memtable so the next cold-start replays an empty WAL.
    serve(state, app, listener, tcp_listener, shutdown_signal()).await;
}

/// How long in-flight requests are allowed to drain after a shutdown signal
/// before we flush and exit regardless. Kept comfortably under a typical 30s
/// platform stop window (e.g. Kubernetes `terminationGracePeriodSeconds`) so the
/// flush always lands before the platform's SIGKILL backstop.
const GRACEFUL_DRAIN: std::time::Duration = std::time::Duration::from_secs(20);

/// How long to wait for the vectorizer embed worker to finish its in-flight batch
/// and stop before flushing. Bounded so a slow embed batch can't eat the stop
/// window — on timeout we flush anyway (the un-stored batch is WAL-durable and
/// replays on restart).
const WORKER_QUIESCE_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// Resolve on the first shutdown signal: SIGTERM (the platform's graceful stop)
/// or SIGINT / Ctrl-C. On non-unix only Ctrl-C is wired. A SIGTERM-handler install
/// failure on unix is fatal: the managed-service stop contract IS SIGTERM, so
/// silently degrading to Ctrl-C-only would kill graceful shutdown on exactly the
/// platform that depends on it — better to fail loud at startup.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                eprintln!("FATAL: could not install SIGTERM handler: {e}");
                std::process::exit(1);
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Serve the HTTP and binary-TCP listeners until `shutdown` resolves, then stop
/// accepting, drain in-flight requests (bounded by [`GRACEFUL_DRAIN`]), and flush
/// the active memtable to an SST so the next cold-start replays an empty WAL.
/// Factored out of [`run`] so a test can drive shutdown with its own future
/// instead of a real signal.
///
/// Without this the process is SIGKILLed mid-request on scale-to-zero: committed
/// data is still WAL-durable (no loss), but every restart pays a full WAL replay
/// and the LSM compaction worker never gets to join.
async fn serve(
    state: Arc<AppState>,
    app: Router,
    http_listener: TcpListener,
    tcp_listener: TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) {
    // A single signal flips this watch; both listeners observe it. `watch` retains
    // the latest value, so a receiver that subscribes after the flip still sees it.
    let (shutdown_tx, _) = tokio::sync::watch::channel(false);
    {
        let shutdown_tx = shutdown_tx.clone();
        tokio::spawn(async move {
            shutdown.await;
            eprintln!("shutdown signal received — draining (up to {GRACEFUL_DRAIN:?})");
            let _ = shutdown_tx.send(true);
        });
    }

    // HTTP: axum's own graceful shutdown stops accepting and waits for in-flight
    // handlers to finish before the future resolves. A fatal serve error flips the
    // shutdown watch so the whole server drains + flushes + exits (the platform
    // then restarts it) rather than silently running with a dead HTTP listener —
    // the old code `.unwrap()`'d here, crashing the process on the same error.
    let http_task = {
        let mut rx = shutdown_tx.subscribe();
        let err_tx = shutdown_tx.clone();
        tokio::spawn(async move {
            let signalled = async move {
                let _ = rx.wait_for(|flagged| *flagged).await;
            };
            if let Err(e) = axum::serve(http_listener, app)
                .with_graceful_shutdown(signalled)
                .await
            {
                eprintln!("http serve error: {e} — initiating shutdown");
                let _ = err_tx.send(true);
            }
        })
    };

    // Binary TCP: select the accept loop against the shutdown watch. On shutdown
    // stop accepting and let in-flight connections finish (bounded by the outer
    // drain timeout below). `conns.join_next()` reaps finished connections so the
    // set can't grow unbounded while accepting.
    let tcp_state = state.clone();
    let mut tcp_shutdown = shutdown_tx.subscribe();
    let tcp_task = tokio::spawn(async move {
        let mut conns = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                accepted = tcp_listener.accept() => match accepted {
                    Ok((socket, _addr)) => {
                        let conn_state = tcp_state.clone();
                        let conn_shutdown = tcp_shutdown.clone();
                        conns.spawn(async move {
                            handle_tcp_connection(socket, conn_state, conn_shutdown).await;
                        });
                    }
                    Err(e) => {
                        eprintln!("tcp accept error: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                },
                Some(_) = conns.join_next() => {}
                _ = tcp_shutdown.changed() => break,
            }
        }
        // Stop accepting new connections; drain the in-flight ones.
        drop(tcp_listener);
        while conns.join_next().await.is_some() {}
    });

    // Serve until the shutdown signal arrives.
    let _ = shutdown_tx.subscribe().wait_for(|flagged| *flagged).await;

    // Bound the drain so the flush always lands before the platform's SIGKILL.
    let drain = async {
        let _ = http_task.await;
        let _ = tcp_task.await;
    };
    if tokio::time::timeout(GRACEFUL_DRAIN, drain).await.is_err() {
        eprintln!("in-flight drain exceeded {GRACEFUL_DRAIN:?} — flushing and exiting anyway");
    }

    // Quiesce the vectorizer embed worker BEFORE flushing: stop_worker joins the
    // worker thread, so any in-flight embed batch finishes its store-and-index
    // commit first — closing the claim→store window that would otherwise orphan
    // an embedding on exit — and it saves the HNSW snapshots for a faster
    // cold-start. Those vector commits then land in the flush below. (Engine-
    // internal workers — migration backfill, cover refresh — aren't owned by the
    // server; any in-flight commits of theirs are WAL-durable and replay on
    // restart, so the empty-WAL cold-start is guaranteed for request-path +
    // vectorizer writes and best-effort otherwise.)
    if let Some(vectorizer) = state.vectorizer.clone() {
        let stop = tokio::task::spawn_blocking(move || vectorizer.stop_worker());
        if tokio::time::timeout(WORKER_QUIESCE_BUDGET, stop).await.is_err() {
            eprintln!(
                "vectorizer worker did not stop within {WORKER_QUIESCE_BUDGET:?}; flushing anyway"
            );
        }
    }

    // Flush the active memtable to an SST. Mirrors /admin/compact: flush() is
    // blocking, so run it on a blocking thread. A flush failure is non-fatal —
    // committed data is WAL-durable and replays on restart.
    let storage = state.db().storage().clone();
    match tokio::task::spawn_blocking(move || storage.flush()).await {
        Ok(Ok(())) => eprintln!("flushed memtable on shutdown; exiting cleanly"),
        Ok(Err(e)) => {
            eprintln!("shutdown flush failed: {e} (data is WAL-durable; restart will replay)")
        }
        Err(e) => eprintln!("shutdown flush task panicked: {e}"),
    }
}

async fn handle_tcp_connection(
    socket: tokio::net::TcpStream,
    state: Arc<AppState>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let (read, write) = socket.into_split();
    let mut reader = tokio::io::BufReader::new(read);
    let mut writer = tokio::io::BufWriter::new(write);
    handle_connection_stream(&mut reader, &mut writer, state, shutdown).await;
}

async fn handle_connection_stream<R, W>(
    reader: &mut R,
    writer: &mut W,
    state: Arc<AppState>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // Per-connection response buffer. Holds the entire wire frame (header +
    // payload) and gets cleared between responses. Eliminates the per-query
    // `Vec::new()` for the response payload and lets the encode + write
    // collapse to one `write_all` instead of four.
    let mut response_buf: Vec<u8> = Vec::with_capacity(4 * 1024);

    loop {
        // Only check shutdown at this request boundary: an in-flight request has
        // already passed `read_frame` and runs to its write before the next loop,
        // so a graceful stop never truncates a response — it just stops reading
        // new frames and closes an otherwise-idle connection promptly (matching
        // axum's HTTP graceful-shutdown semantics). `read_frame` is dropped if
        // shutdown wins, which is fine: we close the socket either way.
        let frame = tokio::select! {
            read = protocol::read_frame(reader) => match read {
                Ok(f) => f,
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        eprintln!("tcp read_frame error: {e}");
                    }
                    return;
                }
            },
            _ = shutdown.changed() => return,
        };

        match frame.kind {
            protocol::REQ_PING => {
                if let Err(e) = protocol::write_frame_buffered(
                    writer,
                    &mut response_buf,
                    frame.req_id,
                    protocol::RESP_PONG,
                    |_| {},
                )
                .await
                {
                    eprintln!("tcp pong write error: {e}");
                    return;
                }
            }
            protocol::REQ_QUERY => {
                let query_text = match protocol::decode_query_payload(&frame.payload) {
                    Ok(q) => q,
                    Err(e) => {
                        let msg = format!("decode: {e}");
                        let _ = protocol::write_frame_buffered(
                            writer,
                            &mut response_buf,
                            frame.req_id,
                            protocol::RESP_ERROR,
                            |buf| protocol::encode_error_payload_into(&msg, buf),
                        )
                        .await;
                        continue;
                    }
                };

                // Schema-epoch read guard around execute only (see handle_query);
                // released before the frame write, which touches no handle state.
                let response = {
                    let _epoch = state.reload_lock.read().await;
                    execute_query(&state, &query_text)
                };
                let write_result = match response {
                    Ok(QueryOutput::Objects(objs)) => {
                        protocol::write_frame_buffered(
                            writer,
                            &mut response_buf,
                            frame.req_id,
                            protocol::RESP_OBJECTS,
                            |buf| protocol::encode_objects_payload_into(&objs, buf),
                        )
                        .await
                    }
                    Ok(QueryOutput::Single(obj)) => {
                        if let Some(vectorizer) = &state.vectorizer {
                            enqueue_vectorize(vectorizer, &state.db(), &obj);
                        }
                        protocol::write_frame_buffered(
                            writer,
                            &mut response_buf,
                            frame.req_id,
                            protocol::RESP_SINGLE,
                            |buf| protocol::encode_object(&obj, buf),
                        )
                        .await
                    }
                    Ok(QueryOutput::Done) => protocol::write_frame_buffered(
                        writer,
                        &mut response_buf,
                        frame.req_id,
                        protocol::RESP_DONE,
                        |_| {},
                    )
                    .await,
                    Ok(QueryOutput::IdSet { .. })
                    | Ok(QueryOutput::IdSetWithFields { .. }) => unreachable!(
                        "QueryOutput::IdSet variants should be materialized to Objects before leaving execute()"
                    ),
                    Err(msg) => protocol::write_frame_buffered(
                        writer,
                        &mut response_buf,
                        frame.req_id,
                        protocol::RESP_ERROR,
                        |buf| protocol::encode_error_payload_into(&msg, buf),
                    )
                    .await,
                };

                if let Err(e) = write_result {
                    eprintln!("tcp write_frame error: {e}");
                    return;
                }
            }
            protocol::REQ_VECTOR_BATCH => {
                let result = match protocol::decode_vector_batch_payload(&frame.payload) {
                    Ok(batch) => match &state.vectorizer {
                        Some(v) => v
                            .ingest_vectors(&batch.type_name, &batch.field_name, &batch.rows)
                            .map_err(|e| format!("{e}")),
                        None => Err(
                            "server has no vector index (schema declares no Vector field)"
                                .to_string(),
                        ),
                    },
                    Err(e) => Err(format!("decode: {e}")),
                };
                let write_result = match result {
                    Ok(_n) => {
                        protocol::write_frame_buffered(
                            writer,
                            &mut response_buf,
                            frame.req_id,
                            protocol::RESP_DONE,
                            |_| {},
                        )
                        .await
                    }
                    Err(msg) => {
                        protocol::write_frame_buffered(
                            writer,
                            &mut response_buf,
                            frame.req_id,
                            protocol::RESP_ERROR,
                            |buf| protocol::encode_error_payload_into(&msg, buf),
                        )
                        .await
                    }
                };
                if let Err(e) = write_result {
                    eprintln!("tcp vector-batch write error: {e}");
                    return;
                }
            }
            other => {
                let msg = format!("unknown request type 0x{other:02x}");
                let _ = protocol::write_frame_buffered(
                    writer,
                    &mut response_buf,
                    frame.req_id,
                    protocol::RESP_ERROR,
                    |buf| protocol::encode_error_payload_into(&msg, buf),
                )
                .await;
            }
        }
    }
}

/// Parse and execute a query, returning either the result or an error message.
fn execute_query(state: &AppState, query_text: &str) -> Result<QueryOutput, String> {
    let query = state
        .query_cache
        .get_or_parse(query_text)
        .map_err(|e| format!("parse error: {e}"))?;
    let db = state.db();
    let ctx = ExecContext {
        db: &db,
        vectorizer: state.vectorizer.as_deref(),
        default_ef: state.default_ef,
        default_rerank: state.default_rerank,
    };
    rhypedb_query::executor::execute(&ctx, &query).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tcp_tests {
    use super::*;
    use tokio::io::duplex;

    fn test_state() -> Arc<AppState> {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                age: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let data_dir = dir.path().to_path_buf();
        // Leak the tempdir — it will live for the test process lifetime.
        std::mem::forget(dir);
        let schema_path = data_dir.join("schema.rhype");
        Arc::new(AppState {
            db: ArcSwap::from(db),
            vectorizer: None,
            query_cache: QueryCache::new(query_cache::DEFAULT_CACHE_SIZE),
            admin_token: None,
            reload_lock: tokio::sync::RwLock::new(()),
            pending_reload_schemas: std::sync::Mutex::new(HashMap::new()),
            data_dir,
            schema_path,
            default_ef: None,
            default_rerank: None,
        })
    }

    #[test]
    fn parse_vector_search_defaults_normalises() {
        // Valid values pass through.
        assert_eq!(
            parse_vector_search_defaults(Some("128"), Some("40")),
            (Some(128), Some(40))
        );
        // Unset -> no defaults.
        assert_eq!(parse_vector_search_defaults(None, None), (None, None));
        // Empty / whitespace-only -> ignored; surrounding whitespace trimmed.
        assert_eq!(
            parse_vector_search_defaults(Some("   "), Some(" 50 ")),
            (None, Some(50))
        );
        // ef must be >= 1; ef=0 is ignored. rerank=0 means "off" -> None.
        assert_eq!(
            parse_vector_search_defaults(Some("0"), Some("0")),
            (None, None)
        );
        // Garbage / negative -> ignored (server still starts).
        assert_eq!(
            parse_vector_search_defaults(Some("abc"), Some("-5")),
            (None, None)
        );
    }

    #[tokio::test]
    async fn ping_pong() {
        let state = test_state();
        let (mut client, server) = duplex(4096);

        let handler = tokio::spawn(async move {
            let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state, shutdown_rx).await;
        });

        protocol::write_frame(&mut client, 1, protocol::REQ_PING, &[]).await.unwrap();
        let resp = protocol::read_frame(&mut client).await.unwrap();
        assert_eq!(resp.req_id, 1);
        assert_eq!(resp.kind, protocol::RESP_PONG);
        assert!(resp.payload.is_empty());

        drop(client);
        let _ = handler.await;
    }

    #[tokio::test]
    async fn create_and_get_via_tcp() {
        let state = test_state();
        let (mut client, server) = duplex(8192);

        let handler = tokio::spawn(async move {
            let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state, shutdown_rx).await;
        });

        // Create a user via the binary protocol.
        let create_payload = protocol::encode_query_payload(
            r#"User.create({ name: "Alice", age: 30 })"#,
        );
        protocol::write_frame(&mut client, 1, protocol::REQ_QUERY, &create_payload)
            .await
            .unwrap();
        let resp = protocol::read_frame(&mut client).await.unwrap();
        assert_eq!(resp.req_id, 1);
        assert_eq!(resp.kind, protocol::RESP_SINGLE);
        let created = protocol::decode_single_payload(&resp.payload).unwrap();
        assert_eq!(created.type_name, "User");
        let user_id = created.id;

        // Now fetch it back — User.get returns Objects (list of 1).
        let get_payload =
            protocol::encode_query_payload(&format!("User.get({user_id})"));
        protocol::write_frame(&mut client, 2, protocol::REQ_QUERY, &get_payload)
            .await
            .unwrap();
        let resp = protocol::read_frame(&mut client).await.unwrap();
        assert_eq!(resp.req_id, 2);
        assert_eq!(resp.kind, protocol::RESP_OBJECTS);
        let objs = protocol::decode_objects_payload(&resp.payload).unwrap();
        assert_eq!(objs.len(), 1);
        let fetched = &objs[0];
        assert_eq!(fetched.id, user_id);
        assert_eq!(
            fetched.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(fetched.fields.get("age"), Some(&Value::U32(30)));

        drop(client);
        let _ = handler.await;
    }

    #[tokio::test]
    async fn error_on_parse_failure() {
        let state = test_state();
        let (mut client, server) = duplex(4096);

        let handler = tokio::spawn(async move {
            let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state, shutdown_rx).await;
        });

        let bad_query = protocol::encode_query_payload("THIS IS NOT VALID");
        protocol::write_frame(&mut client, 7, protocol::REQ_QUERY, &bad_query)
            .await
            .unwrap();
        let resp = protocol::read_frame(&mut client).await.unwrap();
        assert_eq!(resp.req_id, 7);
        assert_eq!(resp.kind, protocol::RESP_ERROR);
        let msg = protocol::decode_error_payload(&resp.payload).unwrap();
        assert!(msg.contains("parse error"), "expected parse error, got {msg}");

        drop(client);
        let _ = handler.await;
    }

    #[tokio::test]
    async fn many_queries_on_one_connection() {
        let state = test_state();
        let (mut client, server) = duplex(16384);

        let handler = tokio::spawn(async move {
            let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state, shutdown_rx).await;
        });

        // Send 10 creates in sequence.
        for i in 0..10u32 {
            let payload = protocol::encode_query_payload(&format!(
                r#"User.create({{ name: "User{i}", age: {} }})"#,
                20 + i
            ));
            protocol::write_frame(&mut client, i, protocol::REQ_QUERY, &payload)
                .await
                .unwrap();
            let resp = protocol::read_frame(&mut client).await.unwrap();
            assert_eq!(resp.req_id, i);
            assert_eq!(resp.kind, protocol::RESP_SINGLE);
        }

        drop(client);
        let _ = handler.await;
    }

    fn count_ssts(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().is_some_and(|x| x == "sst"))
                    .count()
            })
            .unwrap_or(0)
    }

    /// A shutdown signal must drain and then flush the active memtable to an SST,
    /// so the next cold-start replays an empty WAL instead of being SIGKILLed
    /// mid-request with the memtable un-persisted.
    #[tokio::test]
    async fn graceful_shutdown_flushes_memtable() {
        use std::time::Duration;

        let state = test_state();
        let sst_dir = state.data_dir.join("sst");
        let before = count_ssts(&sst_dir);

        // Write a row so the active memtable is non-empty — flush is a no-op on an
        // empty memtable, so without this the test would prove nothing.
        execute_query(&state, r#"User.create({ name: "Persist", age: 7 })"#).unwrap();

        // Ephemeral ports so the test never collides with a fixed bind.
        let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tcp_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let app = Router::new()
            .route("/health", get(handle_health))
            .with_state(state.clone());

        // Drive shutdown with our own oneshot instead of a real SIGTERM.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let serve_task = tokio::spawn(serve(
            state.clone(),
            app,
            http_listener,
            tcp_listener,
            async move {
                let _ = rx.await;
            },
        ));

        // Trigger graceful shutdown; serve() should drain, flush, and return.
        tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), serve_task)
            .await
            .expect("serve did not shut down within 10s")
            .expect("serve task panicked");

        let after = count_ssts(&sst_dir);
        assert!(
            after > before,
            "graceful shutdown should flush the memtable to a new SST (before={before}, after={after})"
        );
    }

    /// An idle binary-protocol connection (the common pooled/keep-alive case) must
    /// close promptly when shutdown fires, rather than blocking on `read_frame`
    /// and holding the drain open for the full timeout.
    #[tokio::test]
    async fn idle_tcp_connection_closes_on_shutdown() {
        use std::time::Duration;

        let state = test_state();
        let (mut client, server) = duplex(4096);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let handler = tokio::spawn(async move {
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state, shutdown_rx).await;
        });

        // Prove the connection is live, then leave it idle (send no more frames).
        protocol::write_frame(&mut client, 1, protocol::REQ_PING, &[]).await.unwrap();
        let resp = protocol::read_frame(&mut client).await.unwrap();
        assert_eq!(resp.kind, protocol::RESP_PONG);

        // Shutdown must close the idle connection without waiting on a next frame.
        shutdown_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(5), handler)
            .await
            .expect("idle connection did not close promptly on shutdown")
            .expect("connection handler panicked");

        drop(client);
    }

    #[test]
    fn schema_introspection_covers_scalar_relation_vector() {
        let schema = parse_schema(
            r#"
            type User {
                name: String @unique
                age: i64 @indexed
                posts: [Post] @inverse(Post.author) @on_delete(remove)
                favorites: [Post] {
                    rating: f32,
                    added_at: DateTime
                } @on_delete(cascade)
                embedding: Vector<384> @vectorize(source: "name", model: "all-MiniLM-L6-v2") @index(hnsw, metric: cosine, quantization: turboquant_3bit, m: 16, ef_construction: 200)
            }
            type Post {
                title: String @indexed
                author: User @on_delete(deny)
            }
            "#,
        )
        .unwrap();

        let v = schema_introspection(&schema);
        assert_eq!(v["format"], "rhypedb-schema-introspection-v1");
        // Types sorted by name: Post before User.
        let types = v["types"].as_array().unwrap();
        assert_eq!(types[0]["name"], "Post");
        assert_eq!(types[1]["name"], "User");

        let field = |ty: &str, name: &str| -> serde_json::Value {
            types
                .iter()
                .find(|t| t["name"] == ty)
                .unwrap()["fields"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["name"] == name)
                .unwrap()
                .clone()
        };

        let name = field("User", "name");
        assert_eq!(name["kind"], "scalar");
        assert_eq!(name["scalar"], "String");
        assert_eq!(name["unique"], true);
        assert_eq!(name["indexed"], false);

        let age = field("User", "age");
        assert_eq!(age["scalar"], "i64");
        assert_eq!(age["indexed"], true);

        let posts = field("User", "posts");
        assert_eq!(posts["kind"], "relation");
        assert_eq!(posts["target"], "Post");
        assert_eq!(posts["many"], true);
        assert_eq!(posts["onDelete"], "remove");
        assert_eq!(posts["inverse"]["type"], "Post");
        assert_eq!(posts["inverse"]["field"], "author");

        let favorites = field("User", "favorites");
        assert_eq!(favorites["onDelete"], "cascade");
        let efs = favorites["edgeFields"].as_array().unwrap();
        assert!(efs.iter().any(|e| e["name"] == "rating" && e["scalar"] == "f32"));
        assert!(efs.iter().any(|e| e["name"] == "added_at" && e["scalar"] == "DateTime"));

        let emb = field("User", "embedding");
        assert_eq!(emb["kind"], "vector");
        assert_eq!(emb["dimensions"], 384);
        assert_eq!(emb["vectorize"]["source"], "name");
        assert_eq!(emb["vectorize"]["model"], "all-MiniLM-L6-v2");
        assert_eq!(emb["index"]["type"], "hnsw");
        assert_eq!(emb["index"]["metric"], "cosine");
        assert_eq!(emb["index"]["quantization"], "turboquant_3bit");
        assert_eq!(emb["index"]["m"], 16);
        assert_eq!(emb["index"]["efConstruction"], 200);

        // The canonical SDL is embedded and reparsable.
        assert!(v["sdl"].as_str().unwrap().contains("type User"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn schema_endpoint_serves_introspection() {
        let state = test_state();
        let app = Router::new()
            .route("/schema", get(handle_schema))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let text = tokio::task::spawn_blocking(move || {
            ureq::get(&format!("http://{addr}/schema"))
                .call()
                .unwrap()
                .body_mut()
                .read_to_string()
                .unwrap()
        })
        .await
        .unwrap();
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["format"], "rhypedb-schema-introspection-v1");
        assert!(body["types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "User"));
        assert!(body["sdl"].as_str().unwrap().contains("type User"));
    }
}
