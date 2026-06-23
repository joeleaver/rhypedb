//! Increment-5 acceptance test: the official sync [`rhypedb_client::Client`] —
//! with codegen-generated typed seeds — driven against a REAL in-process server
//! over a real TCP socket. This is the end-to-end the earlier increments deferred:
//! it proves the whole stack composes (generated code → client → wire → server →
//! engine) for the full surface — connect/ping, typed create/fetch/fetch_one/get,
//! untyped query, prepared statements, raw update + delete, BYO-vector ingest +
//! similar search, and a live change-event subscription.
//!
//! It lives in-crate (not under `tests/`) so it can reach the private server
//! internals (`AppState`, `handle_tcp_connection`, `Vectorizer`) to stand a real
//! server up; `rhypedb-client` / `rhypedb-codegen` are dev-deps.

use std::net::SocketAddr;
use std::sync::Arc;

use super::*;
use rhypedb_client::{ChangeKind, Client, Notification, Query, QueryResult, SubscriptionFilter};

// The codegen-generated typed client for `E2E_SDL`. Compiling it here proves the
// retargeted codegen output builds against `rhypedb-client`; `generated_fixture_is_in_sync`
// keeps the committed file byte-identical to the generator.
#[path = "client_e2e_generated.rs"]
mod generated;
use generated::{Doc, User};

/// The schema the fixture was generated from. Kept byte-identical to the SDL fed
/// to `rhypedb-codegen` (asserted by `generated_fixture_is_in_sync`).
const E2E_SDL: &str = r#"type User {
  name: String @unique
  age: u32 @indexed
  active: Bool
}
type Post {
  title: String
  author: User
}
type Doc {
  label: String
  embedding: Vector<4>
}
"#;

/// Build a real `AppState` (a fresh temp data dir) including a vectorizer for the
/// `Doc.embedding` field, so the BYO-vector ingest path is live.
fn build_state() -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let schema = parse_schema(E2E_SDL).unwrap();
    let db = Database::open(schema.clone(), dir.path()).unwrap();
    let vectorizer = Arc::new(
        Vectorizer::new(
            Arc::clone(db.storage()),
            schema.clone(),
            db.type_ids().clone(),
            db.field_ids().clone(),
        )
        .unwrap(),
    );
    let data_dir = dir.path().to_path_buf();
    // Leak the tempdir for the test-process lifetime (the server holds the data dir).
    std::mem::forget(dir);
    let schema_path = data_dir.join("schema.rhype");
    Arc::new(AppState {
        db: ArcSwap::from(db),
        vectorizer: Some(vectorizer),
        query_cache: QueryCache::new(query_cache::DEFAULT_CACHE_SIZE),
        admin_token: None,
        reload_lock: tokio::sync::RwLock::new(()),
        pending_reload_schemas: std::sync::Mutex::new(HashMap::new()),
        data_dir,
        schema_path,
        default_ef: None,
        default_rerank: None,
        graceful_drain: std::time::Duration::from_secs(20),
        worker_quiesce_budget: std::time::Duration::from_secs(10),
        network_subs: Arc::new(AtomicUsize::new(0)),
        events_dropped: Arc::new(AtomicU64::new(0)),
    })
}

/// Bind an ephemeral TCP port and accept connections forever, serving each with
/// the real `handle_tcp_connection`. The accept task owns a never-fired shutdown
/// sender so per-connection receivers stay live; abort the returned handle to stop.
async fn start_server(state: Arc<AppState>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        // Held for the task's lifetime so connection receivers never see the
        // sender drop (which would look like a shutdown signal).
        let (shutdown_tx, _keep) = tokio::sync::watch::channel(false);
        while let Ok((socket, _)) = listener.accept().await {
            tokio::spawn(handle_tcp_connection(
                socket,
                state.clone(),
                shutdown_tx.subscribe(),
            ));
        }
    });
    (addr, task)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_client_surface_against_real_server() {
    let state = build_state();
    let (addr, accept_task) = start_server(state).await;

    // The sync client blocks, so drive it on a blocking thread while the server
    // tasks run on the runtime's workers. A panic propagates through the join.
    tokio::task::spawn_blocking(move || run_client_flow(addr))
        .await
        .expect("client flow thread panicked");

    accept_task.abort();
}

/// Exercise the full client surface against the real server. Panics on any
/// assertion failure (surfaced through the blocking join).
fn run_client_flow(addr: SocketAddr) {
    let client = Client::connect(addr).unwrap();

    // --- liveness ---
    client.ping().unwrap();

    // --- typed create (generated User::create) ---
    let ada = client
        .create(&User::create(&User {
            name: Some("Ada".into()),
            age: Some(30),
            active: Some(true),
        }))
        .unwrap();
    assert_eq!(ada.data.name.as_deref(), Some("Ada"));
    assert_eq!(ada.data.age, Some(30));
    assert_eq!(ada.data.active, Some(true));
    let _bob = client
        .create(&User::create(&User {
            name: Some("Bob".into()),
            age: Some(20),
            active: Some(false),
        }))
        .unwrap();

    // --- typed fetch (list) ---
    let all = client.fetch(&User::all()).unwrap();
    assert_eq!(all.len(), 2);

    // --- typed fetch_one via filter (indexed predicate) ---
    let adult = client
        .fetch_one(&User::all().filter(".age > 25"))
        .unwrap()
        .expect("a matching row");
    assert_eq!(adult.data.name.as_deref(), Some("Ada"));

    // --- typed get by id ---
    let got = client.fetch_one(&User::get(ada.id)).unwrap().expect("a row");
    assert_eq!(got.id, ada.id);
    assert_eq!(got.data.name.as_deref(), Some("Ada"));

    // --- untyped query ---
    match client.query("User").unwrap() {
        QueryResult::Objects(v) => assert_eq!(v.len(), 2),
        other => panic!("expected a list, got {other:?}"),
    }

    // --- prepared statement: prepare once, execute typed ---
    let stmt = client.prepare(&User::all()).unwrap();
    assert_eq!(client.fetch_prepared(&stmt).unwrap().len(), 2);
    assert_eq!(client.fetch_prepared(&stmt).unwrap().len(), 2); // re-runs with no re-parse

    // --- raw update, then confirm via a typed get ---
    client
        .execute(&Query::<User>::raw(format!(
            "User.get({}).update({{ age: 31 }})",
            ada.id
        )))
        .unwrap();
    let updated = client.fetch_one(&User::get(ada.id)).unwrap().expect("a row");
    assert_eq!(updated.data.age, Some(31));

    // --- BYO-vector ingest + similar search ---
    let d1 = client
        .create(&Doc::create(&Doc { label: Some("d1".into()) }))
        .unwrap();
    let d2 = client
        .create(&Doc::create(&Doc { label: Some("d2".into()) }))
        .unwrap();
    assert_eq!(
        client
            .ingest_vectors("Doc", "embedding", &[(d1.id, vec![1.0f32, 0.0, 0.0, 0.0])])
            .unwrap(),
        1
    );
    assert_eq!(
        client
            .ingest_vectors("Doc", "embedding", &[(d2.id, vec![0.0f32, 1.0, 0.0, 0.0])])
            .unwrap(),
        1
    );
    let hits = client
        .fetch::<Doc>(&Query::raw("Doc.similar(.embedding, [1.0, 0.0, 0.0, 0.0], k: 1)"))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, d1.id, "nearest to [1,0,0,0] must be d1");

    // --- live subscription: subscribe, mutate, receive the pushed event ---
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).unwrap();
    let carol = client
        .create(&User::create(&User {
            name: Some("Carol".into()),
            age: Some(40),
            active: Some(true),
        }))
        .unwrap();
    match sub.next_event().unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Create);
            assert_eq!(c.type_name, "User");
            assert_eq!(c.id, carol.id);
        }
        other => panic!("expected a User create event, got {other:?}"),
    }
    sub.unsubscribe().unwrap();

    // --- raw delete, then confirm it's gone ---
    client
        .execute(&Query::<User>::raw(format!("User.get({}).delete()", ada.id)))
        .unwrap();
    let remaining = client.fetch(&User::all()).unwrap();
    assert!(
        remaining.iter().all(|r| r.id != ada.id),
        "Ada must be deleted"
    );
    // Created Ada + Bob + Carol, deleted Ada → Bob + Carol remain.
    assert_eq!(remaining.len(), 2);
}

#[test]
fn generated_fixture_is_in_sync_with_codegen() {
    // The committed fixture must match what rhypedb-codegen produces for E2E_SDL,
    // so the compiled-and-exercised generated module can't silently drift from the
    // generator. Regenerate with `rhypedb-codegen` if this fails.
    let schema = parse_schema(E2E_SDL).unwrap();
    let regenerated = rhypedb_codegen::generate_rust(&schema);
    assert_eq!(
        regenerated,
        include_str!("client_e2e_generated.rs"),
        "client_e2e_generated.rs is stale — regenerate it from E2E_SDL"
    );
}
