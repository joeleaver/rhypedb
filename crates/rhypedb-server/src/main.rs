use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{Object, Value};
use rhypedb_engine::vectorizer::Vectorizer;
use rhypedb_query::executor::{ExecContext, QueryOutput};
use rhypedb_query::parser::parse_query;
use rhypedb_schema::parser::parse_schema;
use rhypedb_subscribe::{ChangeKind, SubscriptionFilter};

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
}

struct AppState {
    db: Database,
    vectorizer: Option<Arc<Vectorizer>>,
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
    fn from(obj: Object) -> Self {
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
    let query = match parse_query(&req.query) {
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

/// WebSocket subscription endpoint.
/// Query params: ?type=User&id=5&kind=create,update
#[derive(Deserialize)]
struct SubscribeParams {
    #[serde(rename = "type")]
    type_name: Option<String>,
    id: Option<u64>,
    kind: Option<String>,
}

async fn handle_subscribe(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SubscribeParams>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state, params))
}

async fn handle_ws_connection(
    mut socket: WebSocket,
    state: Arc<AppState>,
    params: SubscribeParams,
) {
    let mut filter = match (&params.type_name, params.id) {
        (Some(tn), Some(id)) => SubscriptionFilter::for_object(tn.clone(), id),
        (Some(tn), None) => SubscriptionFilter::for_type(tn.clone()),
        _ => SubscriptionFilter::all(),
    };

    if let Some(kind_str) = &params.kind {
        filter.kinds = kind_str
            .split(',')
            .filter_map(|k| match k.trim() {
                "create" => Some(ChangeKind::Create),
                "update" => Some(ChangeKind::Update),
                "delete" => Some(ChangeKind::Delete),
                _ => None,
            })
            .collect();
    }

    let (_sub_id, rx) = state.db.subscriptions().subscribe(filter);

    let (tx_async, mut rx_async) = tokio::sync::mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        while let Ok(event) = rx.recv() {
            if tx_async.send(event).is_err() {
                break;
            }
        }
    });

    while let Some(event) = rx_async.recv().await {
        let json = serde_json::to_string(&event).unwrap();
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }
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

    let db = Database::open(schema.clone(), &cli.data_dir).unwrap_or_else(|e| {
        eprintln!("failed to open database: {e}");
        std::process::exit(1);
    });

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
        v.start_worker();
        Some(v)
    } else {
        None
    };

    let state = Arc::new(AppState { db, vectorizer });

    let app = Router::new()
        .route("/query", post(handle_query))
        .route("/subscribe", get(handle_subscribe))
        .route("/health", get(handle_health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cli.listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {}: {e}", cli.listen);
            std::process::exit(1);
        });

    println!("rhypedb listening on {}", cli.listen);
    println!("  POST /query     — execute queries");
    println!("  GET  /subscribe — WebSocket subscriptions");
    println!("  GET  /health    — health check");

    axum::serve(listener, app).await.unwrap();
}
