//! Async-client end-to-end tests over a real TCP socket against a mock server
//! that speaks the binary protocol via the shared `rhypedb-wire` codecs.
//!
//! The mock server is plain sync (a std thread) — only the [`AsyncClient`] side
//! is async, which is the point: TCP is TCP. Because the wire format is
//! single-sourced in `rhypedb-wire`, exercising the async client against a
//! wire-faithful mock proves its full I/O path (connect, frame, request/response,
//! decode-to-typed, error mapping, dead-latch) the same way the sync `e2e.rs`
//! does for `Client`. Run with `cargo test -p rhypedb-client --features async`.
#![cfg(feature = "async")]

use std::collections::HashMap;
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rhypedb_client::{AsyncClient, Error, Query, QueryResult};
use rhypedb_wire::object::{FieldMap, Object, Value};
use rhypedb_wire::protocol::{self, sync as wire_sync};

#[derive(serde::Deserialize, Debug, PartialEq)]
struct User {
    name: Option<String>,
    age: Option<u32>,
}

fn user_obj(id: u64, name: &str, age: u32) -> Object {
    let mut fields = FieldMap::new();
    fields.insert("name".into(), Value::String(name.into()));
    fields.insert("age".into(), Value::U32(age));
    Object {
        type_name: "User".into(),
        id,
        fields,
        raw_fields: None,
    }
}

fn single(stream: &mut TcpStream, req_id: u32, obj: Object) {
    let mut buf = Vec::new();
    protocol::encode_object(&obj, &mut buf);
    wire_sync::write_frame(stream, req_id, protocol::RESP_SINGLE, &buf).unwrap();
}

fn objects(stream: &mut TcpStream, req_id: u32, objs: &[Object]) {
    let payload = protocol::encode_objects_payload(objs);
    wire_sync::write_frame(stream, req_id, protocol::RESP_OBJECTS, &payload).unwrap();
}

/// Reply to a query string with the wire-faithful response shape (shared by the
/// `REQ_QUERY` and `REQ_EXECUTE` paths).
fn respond_query(stream: &mut TcpStream, req: u32, q: &str) {
    if q == "User.get(1)" {
        single(stream, req, user_obj(1, "Alice", 30));
    } else if q == "User.get(666)" {
        // Valid object header but truncated fields (tag 3 / U64 with no value
        // bytes). The client must surface an error, never panic.
        let mut bad = Vec::new();
        bad.extend_from_slice(&1u16.to_be_bytes()); // type_len
        bad.push(b'U'); // type "U"
        bad.extend_from_slice(&1u64.to_be_bytes()); // id
        bad.extend_from_slice(&1u16.to_be_bytes()); // fields count = 1
        bad.extend_from_slice(&1u16.to_be_bytes()); // name_len = 1
        bad.push(b'x'); // name
        bad.push(3); // tag U64 — needs 8 value bytes, none present
        wire_sync::write_frame(stream, req, protocol::RESP_SINGLE, &bad).unwrap();
    } else if q == "User" {
        objects(stream, req, &[user_obj(1, "Alice", 30), user_obj(2, "Bob", 25)]);
    } else if q == "User.get(99)" {
        objects(stream, req, &[]);
    } else if q.starts_with("User.create") {
        single(stream, req, user_obj(2, "Bob", 25));
    } else if q.contains("delete") {
        wire_sync::write_frame(stream, req, protocol::RESP_DONE, &[]).unwrap();
    } else {
        let p = protocol::encode_error_payload("no such type: Bad");
        wire_sync::write_frame(stream, req, protocol::RESP_ERROR, &p).unwrap();
    }
}

/// A mock server: one connection, replies to each frame by the wire codecs.
/// Prepared statements are per-connection state, mirroring the real server.
fn spawn_mock() -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut prepared: HashMap<u64, String> = HashMap::new();
        let mut next_stmt_id: u64 = 1;
        while let Ok(frame) = wire_sync::read_frame(&mut stream) {
            let req = frame.req_id;
            match frame.kind {
                protocol::REQ_PING => {
                    wire_sync::write_frame(&mut stream, req, protocol::RESP_PONG, &[]).unwrap();
                }
                protocol::REQ_QUERY => {
                    let q = protocol::decode_query_payload(&frame.payload).unwrap();
                    respond_query(&mut stream, req, &q);
                }
                protocol::REQ_PREPARE => {
                    let q = protocol::decode_query_payload(&frame.payload).unwrap();
                    if q == "PARSEFAIL" {
                        let p = protocol::encode_error_payload("parse error: PARSEFAIL");
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_ERROR, &p).unwrap();
                    } else if q == "SHORTPREP" {
                        // RESP_PREPARED with a truncated stmt_id (< 8 bytes): framing
                        // intact → a decode error that must NOT latch the connection.
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_PREPARED, &[0u8; 4])
                            .unwrap();
                    } else {
                        let id = next_stmt_id;
                        next_stmt_id += 1;
                        prepared.insert(id, q);
                        let p = protocol::encode_prepared_payload(id);
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_PREPARED, &p)
                            .unwrap();
                    }
                }
                protocol::REQ_EXECUTE => {
                    let stmt_id = protocol::decode_execute_payload(&frame.payload).unwrap();
                    match prepared.get(&stmt_id) {
                        Some(q) => {
                            let q = q.clone();
                            respond_query(&mut stream, req, &q);
                        }
                        None => {
                            let p = protocol::encode_error_payload(&format!(
                                "unknown statement id {stmt_id}"
                            ));
                            wire_sync::write_frame(&mut stream, req, protocol::RESP_ERROR, &p)
                                .unwrap();
                        }
                    }
                }
                protocol::REQ_VECTOR_BATCH => {
                    let batch = protocol::decode_vector_batch_payload(&frame.payload).unwrap();
                    if batch.type_name == "Nope" {
                        let p = protocol::encode_error_payload(
                            "server has no vector index (schema declares no Vector field)",
                        );
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_ERROR, &p).unwrap();
                    } else {
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_DONE, &[]).unwrap();
                    }
                }
                _ => {
                    let p = protocol::encode_error_payload("mock: unsupported request");
                    wire_sync::write_frame(&mut stream, req, protocol::RESP_ERROR, &p).unwrap();
                }
            }
        }
    });
    (addr, handle)
}

#[tokio::test]
async fn full_request_response_surface() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    client.ping().await.unwrap();

    let one = client
        .fetch_one::<User>(&Query::get("User", 1))
        .await
        .unwrap()
        .expect("a row");
    assert_eq!(one.id, 1);
    assert_eq!(
        one.data,
        User {
            name: Some("Alice".into()),
            age: Some(30),
        }
    );

    let all = client.fetch::<User>(&Query::all("User")).await.unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!((all[0].id, all[1].id), (1, 2));

    assert!(
        client
            .fetch_one::<User>(&Query::get("User", 99))
            .await
            .unwrap()
            .is_none()
    );

    let created = client
        .create::<User>(&Query::<User>::raw("User.create({ name: \"Bob\" })"))
        .await
        .unwrap();
    assert_eq!(created.id, 2);
    assert_eq!(created.data.name.as_deref(), Some("Bob"));

    assert!(matches!(
        client.execute(&Query::<User>::raw("User.delete(1)")).await.unwrap(),
        QueryResult::Done
    ));

    match client.query("Bad.thing").await {
        Err(Error::Server(msg)) => assert!(msg.contains("no such type"), "got: {msg}"),
        other => panic!("expected a server error, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn malformed_reply_errors_without_panic_and_keeps_connection() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    // A malformed object payload (framing intact) → an error, NOT a panic.
    let err = client
        .fetch_one::<User>(&Query::get("User", 666))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Io(_)), "got: {err:?}");

    // Frame boundaries were intact → the connection is still usable.
    let ok = client
        .fetch_one::<User>(&Query::get("User", 1))
        .await
        .unwrap()
        .expect("a row");
    assert_eq!(ok.id, 1);

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn io_error_latches_connection_closed() {
    // A mock that reads one request then drops the connection without replying →
    // the client's read sees EOF (an I/O error), which must latch the connection
    // closed so the NEXT call fails fast instead of desyncing.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = wire_sync::read_frame(&mut stream); // consume one request, then drop
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    assert!(matches!(client.ping().await, Err(Error::Io(_))));
    assert!(matches!(client.ping().await, Err(Error::Closed)));
    assert!(matches!(client.query("User").await, Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn prepared_statements_roundtrip() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    let get1 = client.prepare(&Query::<User>::get("User", 1)).await.unwrap();
    for _ in 0..3 {
        let row = client.fetch_one_prepared(&get1).await.unwrap().expect("a row");
        assert_eq!(row.id, 1);
        assert_eq!(row.data.name.as_deref(), Some("Alice"));
    }

    let all = client.prepare(&Query::<User>::all("User")).await.unwrap();
    assert_eq!(client.fetch_prepared(&all).await.unwrap().len(), 2);

    let mk = client
        .prepare(&Query::<User>::raw("User.create({ name: \"Bob\" })"))
        .await
        .unwrap();
    let created = client.create_prepared(&mk).await.unwrap();
    assert_eq!(created.id, 2);
    assert_eq!(created.data.name.as_deref(), Some("Bob"));

    let del = client.prepare(&Query::<User>::raw("User.delete(1)")).await.unwrap();
    assert!(matches!(
        client.execute_prepared(&del).await.unwrap(),
        QueryResult::Done
    ));

    match client.prepare(&Query::<User>::raw("PARSEFAIL")).await {
        Err(Error::Server(m)) => assert!(m.contains("parse error"), "got: {m}"),
        other => panic!("expected a prepare-time server error, got {other:?}"),
    }

    let bad = client.prepare(&Query::<User>::raw("Bad.thing")).await.unwrap();
    match client.execute_prepared(&bad).await {
        Err(Error::Server(m)) => assert!(m.contains("no such type"), "got: {m}"),
        other => panic!("expected an execute-time server error, got {other:?}"),
    }
    assert!(client.ping().await.is_ok(), "usable after a server error");

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn prepared_statement_from_another_client_is_rejected() {
    // A Prepared is branded with its issuing client; using it on a different
    // client must fail CLIENT-SIDE (no I/O). The brand counter is shared with the
    // sync client, so this also guards accidental sync↔async handle mixing.
    let (addr_a, handle_a) = spawn_mock();
    let (addr_b, handle_b) = spawn_mock();
    let client_a = AsyncClient::connect(addr_a).await.unwrap();
    let client_b = AsyncClient::connect(addr_b).await.unwrap();

    let stmt = client_a.prepare(&Query::<User>::get("User", 1)).await.unwrap();
    assert!(matches!(
        client_b.execute_prepared(&stmt).await,
        Err(Error::ForeignStatement)
    ));
    assert!(client_b.ping().await.is_ok());
    assert_eq!(
        client_a.fetch_one_prepared(&stmt).await.unwrap().unwrap().id,
        1
    );

    drop(client_a);
    drop(client_b);
    handle_a.join().unwrap();
    handle_b.join().unwrap();
}

#[tokio::test]
async fn ingest_vectors_roundtrip() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    let rows: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.5, -1.0, 2.0]), (7, vec![3.0, 4.0, 5.0])];
    assert_eq!(client.ingest_vectors("Doc", "embedding", &rows).await.unwrap(), 2);

    // Borrowed-slice rows encode identically (generic over AsRef<[f32]>).
    let v1 = [0.1f32, 0.2, 0.3];
    let v2 = [0.4f32, 0.5, 0.6];
    let borrowed: Vec<(u64, &[f32])> = vec![(2, &v1[..]), (3, &v2[..])];
    assert_eq!(
        client.ingest_vectors("Doc", "embedding", &borrowed).await.unwrap(),
        2
    );

    // Empty batch → 0, no validation (matches the engine's short-circuit).
    let empty: Vec<(u64, Vec<f32>)> = Vec::new();
    assert_eq!(client.ingest_vectors("Doc", "embedding", &empty).await.unwrap(), 0);

    match client.ingest_vectors("Nope", "embedding", &rows).await {
        Err(Error::Server(m)) => assert!(m.contains("no vector index"), "got: {m}"),
        other => panic!("expected a server error, got {other:?}"),
    }
    assert!(client.ping().await.is_ok());

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn ingest_vectors_rejects_oversized_batch_without_sending() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    let dim = protocol::MAX_FRAME_PAYLOAD / 4 + 1;
    let huge: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.0f32; dim])];
    match client.ingest_vectors("Doc", "embedding", &huge).await {
        Err(Error::BatchTooLarge { rows, bytes }) => {
            assert_eq!(rows, 1);
            assert!(bytes > protocol::MAX_FRAME_PAYLOAD);
        }
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }
    // Nothing was sent → the connection is pristine and still serves requests.
    assert!(client.ping().await.is_ok());
    let small: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0, 2.0])];
    assert_eq!(client.ingest_vectors("Doc", "embedding", &small).await.unwrap(), 1);

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn torn_vector_batch_latches_connection_closed() {
    // The dead-latch applies to the new request kinds too: a torn reply to a
    // VECTOR_BATCH yields Error::Io, then any further call fails fast as Closed.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = wire_sync::read_frame(&mut stream); // consume the batch, then drop
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let rows: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0, 2.0])];
    assert!(matches!(
        client.ingest_vectors("Doc", "embedding", &rows).await,
        Err(Error::Io(_))
    ));
    assert!(matches!(client.ping().await, Err(Error::Closed)));
    assert!(matches!(
        client.prepare(&Query::<User>::all("User")).await,
        Err(Error::Closed)
    ));

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn malformed_prepared_reply_errors_without_latching() {
    let (addr, handle) = spawn_mock();
    let client = AsyncClient::connect(addr).await.unwrap();

    // RESP_PREPARED with a truncated stmt_id: framing intact → a payload-decode
    // error (Error::Io) that must NOT latch the connection.
    let err = client.prepare(&Query::<User>::raw("SHORTPREP")).await.unwrap_err();
    assert!(matches!(err, Error::Io(_)), "got: {err:?}");

    let ok = client
        .fetch_one::<User>(&Query::get("User", 1))
        .await
        .unwrap()
        .expect("a row");
    assert_eq!(ok.id, 1);

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn cancelled_round_trip_latches_connection_closed() {
    // Unlike the sync client, an async call can be cancelled mid-flight (a
    // `tokio::time::timeout`, a `select!`, an aborted task). A round_trip future
    // dropped after the write but before the full reply is read leaves an unread
    // reply tail in the socket; the NEXT call would otherwise mis-correlate it
    // (the protocol echoes no checkable request id). The pessimistic dead-latch
    // must instead fail every subsequent call fast as Closed.
    //
    // The mock dribbles the PONG reply out in two halves with a gap so a short
    // timeout drops the ping() future while only the first half is on the wire.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let frame = wire_sync::read_frame(&mut stream).unwrap();
        assert_eq!(frame.kind, protocol::REQ_PING);
        let mut framed = Vec::new();
        wire_sync::write_frame(&mut framed, frame.req_id, protocol::RESP_PONG, &[]).unwrap();
        let mid = framed.len() / 2;
        stream.write_all(&framed[..mid]).unwrap();
        stream.flush().unwrap();
        thread::sleep(Duration::from_millis(150));
        stream.write_all(&framed[mid..]).unwrap();
        stream.flush().unwrap();
        // Hold the connection open briefly so the test, not the peer, drives timing.
        thread::sleep(Duration::from_millis(200));
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    // The timeout fires mid-reply, dropping the ping() future.
    let timed_out = tokio::time::timeout(Duration::from_millis(30), client.ping()).await;
    assert!(timed_out.is_err(), "the timeout must fire before the full reply arrives");

    // Latched dead → every subsequent call fails fast as Closed (never a stale,
    // mis-correlated reply). With a latch-only-on-Err design this would instead
    // return a mis-parsed frame, so this asserts the pessimistic latch.
    assert!(matches!(client.ping().await, Err(Error::Closed)));
    assert!(matches!(client.query("User").await, Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn shared_behind_arc_serializes_concurrent_calls() {
    // AsyncClient is Send + Sync: share it behind an Arc and fire many concurrent
    // requests. The tokio mutex serializes them on the one connection, so every
    // reply is correctly correlated (no interleaving / mis-correlation).
    let (addr, handle) = spawn_mock();
    let client = std::sync::Arc::new(AsyncClient::connect(addr).await.unwrap());

    let mut tasks = Vec::new();
    for _ in 0..16 {
        let c = client.clone();
        tasks.push(tokio::spawn(async move {
            let row = c.fetch_one::<User>(&Query::get("User", 1)).await.unwrap().unwrap();
            assert_eq!(row.id, 1);
            assert_eq!(row.data.name.as_deref(), Some("Alice"));
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }

    drop(client);
    handle.join().unwrap();
}
