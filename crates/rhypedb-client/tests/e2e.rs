//! End-to-end client tests over a real TCP socket against a mock server that
//! speaks the binary protocol via the shared `rhypedb-wire` codecs.
//!
//! Because the wire format is single-sourced in `rhypedb-wire` (the real server
//! uses the same codecs), exercising the client against a wire-faithful mock
//! proves the full client I/O path — connect, frame, request/response,
//! decode-to-typed, error mapping — without pulling in the engine or a tokio
//! server. A real-server E2E rides in with the codegen retarget increment.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread::{self, JoinHandle};

use rhypedb_client::{Client, Error, Query, QueryResult};
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
/// `REQ_QUERY` and `REQ_EXECUTE` paths, since a prepared statement re-runs the
/// exact same query the server would have parsed inline).
fn respond_query(stream: &mut TcpStream, req: u32, q: &str) {
    if q == "User.get(1)" {
        single(stream, req, user_obj(1, "Alice", 30));
    } else if q == "User.get(666)" {
        // A RESP_SINGLE whose object has a valid header but truncated fields
        // (tag 3 / U64 with no value bytes). The client must surface an error,
        // never panic.
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
        // No match → empty list.
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
/// Prepared statements are per-connection state (a `stmt_id -> query` map),
/// mirroring the real server.
fn spawn_mock() -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut prepared: HashMap<u64, String> = HashMap::new();
        let mut next_stmt_id: u64 = 1;
        // Loop until the client disconnects (read_frame errs) — a clean exit.
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
                        // A RESP_PREPARED whose stmt_id payload is truncated (< 8
                        // bytes). The framing is intact, so the client must report
                        // a decode error WITHOUT latching the connection dead.
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
                        // The real server replies DONE (the ingested count is not
                        // echoed); the client reports the rows it submitted.
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

#[test]
fn full_request_response_surface() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // Liveness.
    client.ping().unwrap();

    // get → single, typed.
    let one = client
        .fetch_one::<User>(&Query::get("User", 1))
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

    // all → list, typed.
    let all = client.fetch::<User>(&Query::all("User")).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].id, 1);
    assert_eq!(all[1].id, 2);

    // empty list → fetch_one is None.
    assert!(
        client
            .fetch_one::<User>(&Query::get("User", 99))
            .unwrap()
            .is_none()
    );

    // create → the affected single object.
    let created = client
        .create::<User>(&Query::<User>::raw("User.create({ name: \"Bob\" })"))
        .unwrap();
    assert_eq!(created.id, 2);
    assert_eq!(created.data.name.as_deref(), Some("Bob"));

    // delete → Done.
    assert!(matches!(
        client
            .execute(&Query::<User>::raw("User.delete(1)"))
            .unwrap(),
        QueryResult::Done
    ));

    // server error → Error::Server with the server's message.
    match client.query("Bad.thing") {
        Err(Error::Server(msg)) => assert!(msg.contains("no such type"), "got: {msg}"),
        other => panic!("expected a server error, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[test]
fn malformed_reply_errors_without_panic_and_keeps_connection() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // A malformed object payload surfaces as an error — crucially NOT a panic
    // (the framing was intact, so only the payload decode failed).
    let err = client
        .fetch_one::<User>(&Query::get("User", 666))
        .unwrap_err();
    assert!(matches!(err, Error::Io(_)), "got: {err:?}");

    // The frame boundaries were intact, so the connection is still usable.
    let ok = client
        .fetch_one::<User>(&Query::get("User", 1))
        .unwrap()
        .expect("a row");
    assert_eq!(ok.id, 1);

    drop(client);
    handle.join().unwrap();
}

#[test]
fn io_error_latches_connection_closed() {
    // A mock that reads one request then drops the connection without replying
    // → the client's read sees EOF (an I/O error), which must latch the
    // connection closed so the NEXT call fails fast instead of desyncing.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = wire_sync::read_frame(&mut stream); // consume one request, then drop
    });

    let client = Client::connect(addr).unwrap();
    // First call: the peer hung up mid-exchange → an I/O error.
    assert!(matches!(client.ping(), Err(Error::Io(_))));
    // Subsequent calls fail fast as Closed (no silent mis-correlation).
    assert!(matches!(client.ping(), Err(Error::Closed)));
    assert!(matches!(client.query("User"), Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn prepared_statements_roundtrip() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // Prepare once, run several times — each re-run re-sends only the stmt id.
    let get1 = client.prepare(&Query::<User>::get("User", 1)).unwrap();
    for _ in 0..3 {
        let row = client.fetch_one_prepared(&get1).unwrap().expect("a row");
        assert_eq!(row.id, 1);
        assert_eq!(row.data.name.as_deref(), Some("Alice"));
    }

    // A list-shaped prepared query.
    let all = client.prepare(&Query::<User>::all("User")).unwrap();
    assert_eq!(client.fetch_prepared(&all).unwrap().len(), 2);

    // A write-shaped prepared query → create_prepared yields the single object.
    let mk = client
        .prepare(&Query::<User>::raw("User.create({ name: \"Bob\" })"))
        .unwrap();
    let created = client.create_prepared(&mk).unwrap();
    assert_eq!(created.id, 2);
    assert_eq!(created.data.name.as_deref(), Some("Bob"));

    // execute_prepared exposes the untyped result shape (a delete → Done).
    let del = client
        .prepare(&Query::<User>::raw("User.delete(1)"))
        .unwrap();
    assert!(matches!(
        client.execute_prepared(&del).unwrap(),
        QueryResult::Done
    ));

    // A statement that errors at PREPARE time surfaces the server's message.
    match client.prepare(&Query::<User>::raw("PARSEFAIL")) {
        Err(Error::Server(m)) => assert!(m.contains("parse error"), "got: {m}"),
        other => panic!("expected a prepare-time server error, got {other:?}"),
    }

    // A statement that prepares fine but errors at EXECUTE time (unknown type)
    // surfaces the server error — and the connection stays usable afterwards.
    let bad = client.prepare(&Query::<User>::raw("Bad.thing")).unwrap();
    match client.execute_prepared(&bad) {
        Err(Error::Server(m)) => assert!(m.contains("no such type"), "got: {m}"),
        other => panic!("expected an execute-time server error, got {other:?}"),
    }
    assert!(client.ping().is_ok(), "connection still usable after a server error");

    drop(client);
    handle.join().unwrap();
}

#[test]
fn prepared_statement_from_another_client_is_rejected() {
    // A Prepared is branded with its issuing client; using it on a different
    // client must fail CLIENT-SIDE (no I/O), so it can't accidentally run the
    // other connection's unrelated statement of the same id.
    let (addr_a, handle_a) = spawn_mock();
    let (addr_b, handle_b) = spawn_mock();
    let client_a = Client::connect(addr_a).unwrap();
    let client_b = Client::connect(addr_b).unwrap();

    let stmt = client_a.prepare(&Query::<User>::get("User", 1)).unwrap();
    assert!(matches!(
        client_b.execute_prepared(&stmt),
        Err(Error::ForeignStatement)
    ));
    // client_b is untouched by the rejected call and still works.
    assert!(client_b.ping().is_ok());
    // The owning client still runs it fine.
    assert_eq!(
        client_a.fetch_one_prepared(&stmt).unwrap().unwrap().id,
        1
    );

    drop(client_a);
    drop(client_b);
    handle_a.join().unwrap();
    handle_b.join().unwrap();
}

#[test]
fn ingest_vectors_roundtrip() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // Owned Vec<f32> rows.
    let rows: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.5, -1.0, 2.0]), (7, vec![3.0, 4.0, 5.0])];
    assert_eq!(client.ingest_vectors("Doc", "embedding", &rows).unwrap(), 2);

    // Borrowed-slice rows encode identically (generic over AsRef<[f32]>).
    let v1 = [0.1f32, 0.2, 0.3];
    let v2 = [0.4f32, 0.5, 0.6];
    let borrowed: Vec<(u64, &[f32])> = vec![(2, &v1[..]), (3, &v2[..])];
    assert_eq!(
        client.ingest_vectors("Doc", "embedding", &borrowed).unwrap(),
        2
    );

    // An empty batch is accepted and reports 0 ingested. NOTE: the real engine
    // short-circuits `if rows.is_empty() { return Ok(0); }` BEFORE checking that
    // the type/field exists, so an empty batch is a no-op even for a typo'd
    // type/field — it does NOT validate. (A non-empty batch does validate.)
    let empty: Vec<(u64, Vec<f32>)> = Vec::new();
    assert_eq!(client.ingest_vectors("Doc", "embedding", &empty).unwrap(), 0);

    // A server-side error (no vector index) surfaces as Error::Server.
    match client.ingest_vectors("Nope", "embedding", &rows) {
        Err(Error::Server(m)) => assert!(m.contains("no vector index"), "got: {m}"),
        other => panic!("expected a server error, got {other:?}"),
    }
    // Connection still usable after the error.
    assert!(client.ping().is_ok());

    drop(client);
    handle.join().unwrap();
}

#[test]
fn ingest_vectors_rejects_oversized_batch_without_sending() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // One row whose vector alone exceeds the 16 MiB frame cap. The client must
    // refuse it client-side (BatchTooLarge) rather than write an oversized frame
    // the server would reject by tearing the connection.
    let dim = protocol::MAX_FRAME_PAYLOAD / 4 + 1;
    let huge: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.0f32; dim])];
    match client.ingest_vectors("Doc", "embedding", &huge) {
        Err(Error::BatchTooLarge { rows, bytes }) => {
            assert_eq!(rows, 1);
            assert!(bytes > protocol::MAX_FRAME_PAYLOAD);
        }
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }
    // Nothing was sent, so the connection is pristine and still serves requests.
    assert!(client.ping().is_ok());
    let small: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0, 2.0])];
    assert_eq!(client.ingest_vectors("Doc", "embedding", &small).unwrap(), 1);

    drop(client);
    handle.join().unwrap();
}

#[test]
fn torn_vector_batch_latches_connection_closed() {
    // The dead-latch must apply to the increment's NEW request kinds too: a torn
    // reply to a VECTOR_BATCH yields Error::Io, then the connection refuses every
    // further call (any kind) as Closed — no mis-correlation across ops.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = wire_sync::read_frame(&mut stream); // consume the batch, then drop
    });

    let client = Client::connect(addr).unwrap();
    let rows: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0, 2.0])];
    assert!(matches!(
        client.ingest_vectors("Doc", "embedding", &rows),
        Err(Error::Io(_))
    ));
    // Latched: a subsequent op of a DIFFERENT kind also fails fast as Closed.
    assert!(matches!(client.ping(), Err(Error::Closed)));
    assert!(matches!(
        client.prepare(&Query::<User>::all("User")),
        Err(Error::Closed)
    ));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn malformed_prepared_reply_errors_without_latching() {
    let (addr, handle) = spawn_mock();
    let client = Client::connect(addr).unwrap();

    // A RESP_PREPARED whose stmt_id payload is truncated: the framing is intact,
    // so this is a payload-DECODE error (Error::Io) that must NOT latch the
    // connection — only true I/O errors latch.
    let err = client
        .prepare(&Query::<User>::raw("SHORTPREP"))
        .unwrap_err();
    assert!(matches!(err, Error::Io(_)), "got: {err:?}");

    // Frame boundaries intact → the connection is still usable.
    let ok = client
        .fetch_one::<User>(&Query::get("User", 1))
        .unwrap()
        .expect("a row");
    assert_eq!(ok.id, 1);

    drop(client);
    handle.join().unwrap();
}
