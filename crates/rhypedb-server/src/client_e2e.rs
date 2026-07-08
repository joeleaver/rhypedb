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
use rhypedb_client::{
    AsyncClient, ChangeKind, Client, Notification, Query, QueryResult, SubscriptionFilter,
};

// The codegen-generated typed client for `E2E_SDL`. Compiling it here proves the
// retargeted codegen output builds against `rhypedb-client`; `generated_fixture_is_in_sync`
// keeps the committed file byte-identical to the generator.
#[path = "client_e2e_generated.rs"]
mod generated;
use generated::{Doc, Thing, User};

/// The schema the fixture was generated from. Kept byte-identical to the SDL fed
/// to `rhypedb-codegen` (asserted by `generated_fixture_is_in_sync`). `Thing`
/// exercises DateTime/Bytes/Json round-trips; `Tag` is relation-only (its
/// generated `create()` has no scalar fields — it proves that path compiles
/// warning-free, exercised at compile time even though the test never builds a Tag).
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
type Thing {
  created: DateTime
  blob: Bytes
  meta: Json
  label: String
}
type Tag {
  owner: User
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
        query_governor: None,
        queries_total: Arc::new(AtomicU64::new(0)),
        metering_cache: std::sync::Mutex::new(None),
        rules: None,
        principal_source: None,
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
    // execute_prepared exposes the untyped result shape.
    match client.execute_prepared(&stmt).unwrap() {
        QueryResult::Objects(v) => assert_eq!(v.len(), 2),
        other => panic!("expected a list, got {other:?}"),
    }
    // fetch_one_prepared (the Option path).
    let get_ada = client.prepare(&User::get(ada.id)).unwrap();
    assert_eq!(
        client.fetch_one_prepared(&get_ada).unwrap().unwrap().id,
        ada.id
    );
    // create_prepared (the into_typed_single path, distinct from fetch_prepared).
    let mk_doc = client
        .prepare(&Doc::create(&Doc { label: Some("prepared-doc".into()) }))
        .unwrap();
    let pd = client.create_prepared(&mk_doc).unwrap();
    assert_eq!(pd.data.label.as_deref(), Some("prepared-doc"));

    // --- DateTime / Bytes / Json typed create + read-back over the real socket ---
    // (the silent-wrong-answer-prone path: create-literal escaping + server
    // coercion + read rendering must all agree.)
    let thing = client
        .create(&Thing::create(&Thing {
            created: Some("2021-01-01T00:00:00Z".into()), // RFC 3339
            blob: Some("AAEC".into()),                    // base64 of [0,1,2]
            meta: Some(serde_json::json!({ "k": 1, "tags": ["a", "b"] })),
            label: Some("t1".into()),
        }))
        .unwrap();
    let back = client.fetch_one(&Thing::get(thing.id)).unwrap().expect("a row");
    assert_eq!(back.data.created.as_deref(), Some("2021-01-01T00:00:00Z"));
    assert_eq!(back.data.blob.as_deref(), Some("AAEC"));
    assert_eq!(back.data.meta, Some(serde_json::json!({ "k": 1, "tags": ["a", "b"] })));
    assert_eq!(back.data.label.as_deref(), Some("t1"));

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

    // --- live subscription: cover Create / Update / Delete + the fields payload ---
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).unwrap();
    let carol = client
        .create(&User::create(&User {
            name: Some("Carol".into()),
            age: Some(40),
            active: Some(true),
        }))
        .unwrap();
    // Create — also assert the best-effort fields payload carries the scalar.
    match sub.next_event().unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Create);
            assert_eq!(c.type_name, "User");
            assert_eq!(c.id, carol.id);
            assert_eq!(
                c.fields.as_ref().and_then(|m| m.get("name")),
                Some(&serde_json::json!("Carol")),
                "create event must carry the scalar fields"
            );
        }
        other => panic!("expected a User create event, got {other:?}"),
    }
    // Update.
    client
        .execute(&Query::<User>::raw(format!(
            "User.get({}).update({{ age: 41 }})",
            carol.id
        )))
        .unwrap();
    match sub.next_event().unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Update);
            assert_eq!(c.id, carol.id);
        }
        other => panic!("expected a User update event, got {other:?}"),
    }
    // Delete.
    client
        .execute(&Query::<User>::raw(format!("User.get({}).delete()", carol.id)))
        .unwrap();
    match sub.next_event().unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Delete);
            assert_eq!(c.id, carol.id);
        }
        other => panic!("expected a User delete event, got {other:?}"),
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
    // Created Ada + Bob + Carol; deleted Carol (in the subscription block) and
    // Ada → only Bob remains.
    assert_eq!(remaining.len(), 1);
}

/// Issue #13: an engine-stamped write origin survives the FULL network seam —
/// engine publish → hub → `ConnEventSink` → wire (decimal string) →
/// `ChangeNotification::from_wire` — and reaches a real network subscriber
/// intact, losslessly past 2^53. The unit tests cover the halves in isolation
/// (wire `from_change_event`, client decode of a hand-crafted frame); this joins
/// them against a real engine event. Writes are stamped via the ENGINE handle
/// because the network verbs cannot tag origin (inbound origin is out of scope).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn write_origin_reaches_network_subscriber() {
    let state = build_state();
    let db_state = Arc::clone(&state);
    let (addr, accept_task) = start_server(state).await;

    tokio::task::spawn_blocking(move || {
        use rhypedb_engine::object::FieldMap;
        const ORIGIN: u64 = 9_007_199_254_740_993; // 2^53 + 1: lossless only as a string

        let client = Client::connect(addr).unwrap();
        // Subscribe first: the handshake ack means the sink is registered
        // server-side before any write is stamped, so no event can be missed.
        let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).unwrap();
        let db = db_state.db();

        // Tagged create → origin surfaces on the network subscriber, lossless.
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Origin-Ada".into()));
        let obj = db.create_with_origin("User", f, Some(ORIGIN)).unwrap();
        match sub.next_event().unwrap() {
            Notification::Change(c) => {
                assert_eq!((c.kind, c.id), (ChangeKind::Create, obj.id));
                assert_eq!(
                    c.origin,
                    Some(ORIGIN),
                    "engine-stamped origin must survive the wire losslessly past 2^53"
                );
            }
            other => panic!("expected a tagged create event, got {other:?}"),
        }

        // Untagged engine write → origin is None over the same seam.
        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Plain-Bob".into()));
        db.create("User", f2).unwrap();
        match sub.next_event().unwrap() {
            Notification::Change(c) => {
                assert_eq!(c.origin, None, "untagged write → no origin on the wire")
            }
            other => panic!("expected an untagged create event, got {other:?}"),
        }

        // Delete carries the origin through the same seam too.
        db.delete_with_origin("User", obj.id, Some(ORIGIN)).unwrap();
        match sub.next_event().unwrap() {
            Notification::Change(c) => {
                assert_eq!((c.kind, c.id, c.origin), (ChangeKind::Delete, obj.id, Some(ORIGIN)));
            }
            other => panic!("expected a tagged delete event, got {other:?}"),
        }

        sub.unsubscribe().unwrap();
    })
    .await
    .expect("client flow thread panicked");

    accept_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_async_client_surface_against_real_server() {
    let state = build_state();
    let (addr, accept_task) = start_server(state).await;

    // The async client runs natively on the runtime — no spawn_blocking needed.
    run_async_client_flow(addr).await;

    accept_task.abort();
}

/// The async parallel to [`run_client_flow`]: exercise the full `AsyncClient`
/// surface against the real server. Panics on any assertion failure.
async fn run_async_client_flow(addr: SocketAddr) {
    use tokio_stream::StreamExt;

    let client = AsyncClient::connect(addr).await.unwrap();

    // --- liveness ---
    client.ping().await.unwrap();

    // --- typed create (generated User::create) ---
    let ada = client
        .create(&User::create(&User {
            name: Some("Ada".into()),
            age: Some(30),
            active: Some(true),
        }))
        .await
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
        .await
        .unwrap();

    // --- typed fetch (list) ---
    assert_eq!(client.fetch(&User::all()).await.unwrap().len(), 2);

    // --- typed fetch_one via filter (indexed predicate) ---
    let adult = client
        .fetch_one(&User::all().filter(".age > 25"))
        .await
        .unwrap()
        .expect("a matching row");
    assert_eq!(adult.data.name.as_deref(), Some("Ada"));

    // --- typed get by id ---
    let got = client.fetch_one(&User::get(ada.id)).await.unwrap().expect("a row");
    assert_eq!((got.id, got.data.name.as_deref()), (ada.id, Some("Ada")));

    // --- untyped query ---
    match client.query("User").await.unwrap() {
        QueryResult::Objects(v) => assert_eq!(v.len(), 2),
        other => panic!("expected a list, got {other:?}"),
    }

    // --- prepared statements: full family ---
    let stmt = client.prepare(&User::all()).await.unwrap();
    assert_eq!(client.fetch_prepared(&stmt).await.unwrap().len(), 2);
    assert_eq!(client.fetch_prepared(&stmt).await.unwrap().len(), 2); // re-runs, no re-parse
    match client.execute_prepared(&stmt).await.unwrap() {
        QueryResult::Objects(v) => assert_eq!(v.len(), 2),
        other => panic!("expected a list, got {other:?}"),
    }
    let get_ada = client.prepare(&User::get(ada.id)).await.unwrap();
    assert_eq!(
        client.fetch_one_prepared(&get_ada).await.unwrap().unwrap().id,
        ada.id
    );
    let mk_doc = client
        .prepare(&Doc::create(&Doc { label: Some("prepared-doc".into()) }))
        .await
        .unwrap();
    assert_eq!(
        client.create_prepared(&mk_doc).await.unwrap().data.label.as_deref(),
        Some("prepared-doc")
    );

    // --- DateTime / Bytes / Json typed create + read-back over the real socket ---
    let thing = client
        .create(&Thing::create(&Thing {
            created: Some("2021-01-01T00:00:00Z".into()),
            blob: Some("AAEC".into()),
            meta: Some(serde_json::json!({ "k": 1, "tags": ["a", "b"] })),
            label: Some("t1".into()),
        }))
        .await
        .unwrap();
    let back = client.fetch_one(&Thing::get(thing.id)).await.unwrap().expect("a row");
    assert_eq!(back.data.created.as_deref(), Some("2021-01-01T00:00:00Z"));
    assert_eq!(back.data.blob.as_deref(), Some("AAEC"));
    assert_eq!(back.data.meta, Some(serde_json::json!({ "k": 1, "tags": ["a", "b"] })));
    assert_eq!(back.data.label.as_deref(), Some("t1"));

    // --- raw update, then confirm via a typed get ---
    client
        .execute(&Query::<User>::raw(format!("User.get({}).update({{ age: 31 }})", ada.id)))
        .await
        .unwrap();
    let updated = client.fetch_one(&User::get(ada.id)).await.unwrap().expect("a row");
    assert_eq!(updated.data.age, Some(31));

    // --- BYO-vector ingest + similar search ---
    let d1 = client.create(&Doc::create(&Doc { label: Some("d1".into()) })).await.unwrap();
    let d2 = client.create(&Doc::create(&Doc { label: Some("d2".into()) })).await.unwrap();
    assert_eq!(
        client.ingest_vectors("Doc", "embedding", &[(d1.id, vec![1.0f32, 0.0, 0.0, 0.0])]).await.unwrap(),
        1
    );
    assert_eq!(
        client.ingest_vectors("Doc", "embedding", &[(d2.id, vec![0.0f32, 1.0, 0.0, 0.0])]).await.unwrap(),
        1
    );
    let hits = client
        .fetch::<Doc>(&Query::raw("Doc.similar(.embedding, [1.0, 0.0, 0.0, 0.0], k: 1)"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, d1.id, "nearest to [1,0,0,0] must be d1");

    // --- live subscription via the async Stream: Create / Update / Delete ---
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).await.unwrap();
    let carol = client
        .create(&User::create(&User {
            name: Some("Carol".into()),
            age: Some(40),
            active: Some(true),
        }))
        .await
        .unwrap();
    // Drive the Stream surface (StreamExt::next) against real server pushes.
    match sub.next().await.expect("a stream item").unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Create);
            assert_eq!((c.type_name.as_str(), c.id), ("User", carol.id));
            assert_eq!(
                c.fields.as_ref().and_then(|m| m.get("name")),
                Some(&serde_json::json!("Carol")),
                "create event must carry the scalar fields"
            );
        }
        other => panic!("expected a User create event, got {other:?}"),
    }
    client
        .execute(&Query::<User>::raw(format!("User.get({}).update({{ age: 41 }})", carol.id)))
        .await
        .unwrap();
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Update, carol.id)),
        other => panic!("expected a User update event, got {other:?}"),
    }
    client
        .execute(&Query::<User>::raw(format!("User.get({}).delete()", carol.id)))
        .await
        .unwrap();
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Delete, carol.id)),
        other => panic!("expected a User delete event, got {other:?}"),
    }
    sub.unsubscribe().await.unwrap();

    // --- raw delete, then confirm it's gone ---
    client
        .execute(&Query::<User>::raw(format!("User.get({}).delete()", ada.id)))
        .await
        .unwrap();
    let remaining = client.fetch(&User::all()).await.unwrap();
    assert!(remaining.iter().all(|r| r.id != ada.id), "Ada must be deleted");
    assert_eq!(remaining.len(), 1); // only Bob remains
}

#[test]
fn generated_fixture_is_in_sync_with_codegen() {
    // The committed fixture must match what rhypedb-codegen produces for E2E_SDL,
    // so the compiled-and-exercised generated module can't silently drift from the
    // generator. Regenerate with `rhypedb-codegen` if this fails.
    let schema = parse_schema(E2E_SDL).unwrap();
    let regenerated = rhypedb_codegen::generate_rust(&schema);
    // Normalize line endings: the generator emits `\n`, but a CRLF checkout (e.g.
    // Windows autocrlf) would materialize the committed file with `\r\n`. Compare
    // on content, not platform line-ending policy.
    assert_eq!(
        regenerated.replace("\r\n", "\n"),
        include_str!("client_e2e_generated.rs").replace("\r\n", "\n"),
        "client_e2e_generated.rs is stale — regenerate it from E2E_SDL"
    );
}
