use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
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
use rhypedb_storage::lsm::LsmTree;

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
        Ok(QueryOutput::Single(obj)) => (
            StatusCode::OK,
            Json(QueryResponse {
                objects: None,
                object: Some(ObjectJson::from(obj)),
                ok: None,
                error: None,
            }),
        ),
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

async fn handle_health() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
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

    // Set up vectorizer if any fields have @vectorize.
    let has_vectorize = schema.types.values().any(|td| {
        td.fields.iter().any(|f| f.vectorize().is_some())
    });

    let vectorizer = if has_vectorize {
        let storage = Arc::new(
            LsmTree::open(rhypedb_storage::lsm::LsmConfig::new(
                cli.data_dir.join("vectorizer"),
            ))
            .unwrap(),
        );

        let mut type_ids = HashMap::new();
        let mut field_ids = HashMap::new();
        let mut next_field_id = 1u64;
        let mut type_names: Vec<_> = schema.types.keys().cloned().collect();
        type_names.sort();
        for (type_id, name) in (1u64..).zip(type_names.iter()) {
            type_ids.insert(name.clone(), type_id);
            let type_def = &schema.types[name];
            for field in &type_def.fields {
                let field_key = format!("{name}.{}", field.name);
                field_ids.insert(field_key, next_field_id);
                next_field_id += 1;
            }
        }

        let v = Arc::new(
            Vectorizer::new(storage, schema.clone(), type_ids, field_ids).unwrap(),
        );
        v.start_worker();
        Some(v)
    } else {
        None
    };

    let state = Arc::new(AppState { db, vectorizer });

    let app = Router::new()
        .route("/query", post(handle_query))
        .route("/health", get(handle_health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cli.listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {}: {e}", cli.listen);
            std::process::exit(1);
        });

    println!("rhypedb listening on {}", cli.listen);

    axum::serve(listener, app).await.unwrap();
}
