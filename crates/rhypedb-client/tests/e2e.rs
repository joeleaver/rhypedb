//! End-to-end client tests over a real TCP socket against a mock server that
//! speaks the binary protocol via the shared `rhypedb-wire` codecs.
//!
//! Because the wire format is single-sourced in `rhypedb-wire` (the real server
//! uses the same codecs), exercising the client against a wire-faithful mock
//! proves the full client I/O path — connect, frame, request/response,
//! decode-to-typed, error mapping — without pulling in the engine or a tokio
//! server. A real-server E2E rides in with the codegen retarget increment.

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

/// A mock server: one connection, replies to each frame by the wire codecs.
fn spawn_mock() -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Loop until the client disconnects (read_frame errs) — a clean exit.
        while let Ok(frame) = wire_sync::read_frame(&mut stream) {
            match frame.kind {
                protocol::REQ_PING => {
                    wire_sync::write_frame(&mut stream, frame.req_id, protocol::RESP_PONG, &[])
                        .unwrap();
                }
                protocol::REQ_QUERY => {
                    let q = protocol::decode_query_payload(&frame.payload).unwrap();
                    let req = frame.req_id;
                    if q == "User.get(1)" {
                        single(&mut stream, req, user_obj(1, "Alice", 30));
                    } else if q == "User.get(666)" {
                        // A RESP_SINGLE whose object has a valid header but
                        // truncated fields (tag 3 / U64 with no value bytes).
                        // The client must surface an error, never panic.
                        let mut bad = Vec::new();
                        bad.extend_from_slice(&1u16.to_be_bytes()); // type_len
                        bad.push(b'U'); // type "U"
                        bad.extend_from_slice(&1u64.to_be_bytes()); // id
                        bad.extend_from_slice(&1u16.to_be_bytes()); // fields count = 1
                        bad.extend_from_slice(&1u16.to_be_bytes()); // name_len = 1
                        bad.push(b'x'); // name
                        bad.push(3); // tag U64 — needs 8 value bytes, none present
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_SINGLE, &bad)
                            .unwrap();
                    } else if q == "User" {
                        objects(&mut stream, req, &[user_obj(1, "Alice", 30), user_obj(2, "Bob", 25)]);
                    } else if q == "User.get(99)" {
                        // No match → empty list.
                        objects(&mut stream, req, &[]);
                    } else if q.starts_with("User.create") {
                        single(&mut stream, req, user_obj(2, "Bob", 25));
                    } else if q.contains("delete") {
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_DONE, &[]).unwrap();
                    } else {
                        let p = protocol::encode_error_payload("no such type: Bad");
                        wire_sync::write_frame(&mut stream, req, protocol::RESP_ERROR, &p).unwrap();
                    }
                }
                _ => {
                    let p = protocol::encode_error_payload("mock: unsupported request");
                    wire_sync::write_frame(&mut stream, frame.req_id, protocol::RESP_ERROR, &p)
                        .unwrap();
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
