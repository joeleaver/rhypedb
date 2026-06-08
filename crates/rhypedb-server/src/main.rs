use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// Swap the global allocator. mimalloc consistently beats glibc malloc on
// small-object workloads (~20-30% faster on 24-32 byte allocs from the
// cascade hot path) without giving up determinism. `secure` feature is
// off — we're not on a hostile-input boundary inside the process.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{Object, Value};
use rhypedb_engine::vectorizer::Vectorizer;
use rhypedb_query::executor::{ExecContext, QueryOutput};
use rhypedb_schema::parser::parse_schema;

mod protocol;
mod query_cache;

use query_cache::QueryCache;

#[derive(Parser)]
#[command(name = "rhypedb", about = "rhypedb database server")]
struct Cli {
    /// Path to the SDL schema file.
    #[arg(short, long)]
    schema: PathBuf,

    /// Data directory for storage.
    #[arg(short, long, default_value = "./rhypedb-data")]
    data_dir: PathBuf,

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

struct AppState {
    db: Arc<Database>,
    vectorizer: Option<Arc<Vectorizer>>,
    query_cache: QueryCache,
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

    let ctx = ExecContext {
        db: &state.db,
        vectorizer: state.vectorizer.as_deref(),
    };

    match rhypedb_query::executor::execute(&ctx, &query) {
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
                enqueue_vectorize(vectorizer, &state.db, &obj);
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
        state.db.subscriptions().subscription_count()
    )
}

/// Force-flush the active memtable + compact all SST files into one.
/// Operational only — no auth, no admin gating. For benchmarking and
/// manual triggering during development. Auto-compaction is the proper
/// long-term answer.
async fn handle_admin_compact(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let storage = state.db.storage();
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
        "subscriptions": state.db.subscriptions().subscription_count(),
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

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let schema_text = std::fs::read_to_string(&cli.schema).unwrap_or_else(|e| {
        eprintln!("failed to read schema file {:?}: {e}", cli.schema);
        std::process::exit(1);
    });

    let schema = parse_schema(&schema_text).unwrap_or_else(|e| {
        eprintln!("schema error: {e}");
        std::process::exit(1);
    });

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

    let has_vectorize = schema
        .types
        .values()
        .any(|td| td.fields.iter().any(|f| f.vectorize().is_some()));

    let vectorizer = if has_vectorize {
        let v = Arc::new(
            Vectorizer::new(
                Arc::clone(db.storage()),
                schema.clone(),
                db.type_ids().clone(),
                db.field_ids().clone(),
            )
            .unwrap(),
        );
        v.start_worker(1);
        Some(v)
    } else {
        None
    };

    let state = Arc::new(AppState {
        db,
        vectorizer,
        query_cache: QueryCache::new(query_cache::DEFAULT_CACHE_SIZE),
    });

    let app = Router::new()
        .route("/query", post(handle_query))
        .route("/status", get(handle_status))
        .route("/health", get(handle_health))
        .route("/admin/compact", post(handle_admin_compact))
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

    // Spawn the binary TCP accept loop.
    let tcp_state = state.clone();
    tokio::spawn(async move {
        loop {
            match tcp_listener.accept().await {
                Ok((socket, _addr)) => {
                    let conn_state = tcp_state.clone();
                    tokio::spawn(async move {
                        handle_tcp_connection(socket, conn_state).await;
                    });
                }
                Err(e) => {
                    eprintln!("tcp accept error: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    });

    axum::serve(listener, app).await.unwrap();
}

async fn handle_tcp_connection(socket: tokio::net::TcpStream, state: Arc<AppState>) {
    let (read, write) = socket.into_split();
    let mut reader = tokio::io::BufReader::new(read);
    let mut writer = tokio::io::BufWriter::new(write);
    handle_connection_stream(&mut reader, &mut writer, state).await;
}

async fn handle_connection_stream<R, W>(reader: &mut R, writer: &mut W, state: Arc<AppState>)
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // Per-connection response buffer. Holds the entire wire frame (header +
    // payload) and gets cleared between responses. Eliminates the per-query
    // `Vec::new()` for the response payload and lets the encode + write
    // collapse to one `write_all` instead of four.
    let mut response_buf: Vec<u8> = Vec::with_capacity(4 * 1024);

    loop {
        let frame = match protocol::read_frame(reader).await {
            Ok(f) => f,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    eprintln!("tcp read_frame error: {e}");
                }
                return;
            }
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

                let response = execute_query(&state, &query_text);
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
                            enqueue_vectorize(vectorizer, &state.db, &obj);
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
    let ctx = ExecContext {
        db: &state.db,
        vectorizer: state.vectorizer.as_deref(),
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
        // Leak the tempdir — it will live for the test process lifetime.
        std::mem::forget(dir);
        Arc::new(AppState {
            db,
            vectorizer: None,
            query_cache: QueryCache::new(query_cache::DEFAULT_CACHE_SIZE),
        })
    }

    #[tokio::test]
    async fn ping_pong() {
        let state = test_state();
        let (mut client, server) = duplex(4096);

        let handler = tokio::spawn(async move {
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state).await;
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
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state).await;
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
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state).await;
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
            let (read, write) = tokio::io::split(server);
            let mut reader = tokio::io::BufReader::new(read);
            let mut writer = tokio::io::BufWriter::new(write);
            handle_connection_stream(&mut reader, &mut writer, state).await;
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
}
