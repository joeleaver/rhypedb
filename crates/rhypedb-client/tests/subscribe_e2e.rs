//! End-to-end subscription tests over real TCP sockets against a mock server
//! that speaks the binary protocol via the shared `rhypedb-wire` codecs.
//!
//! `Client::subscribe` opens a SECOND, dedicated connection (separate from the
//! client's query connection), so every mock here accepts two connections: the
//! idle query connection first, then the subscription connection it drives.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread::{self, JoinHandle};

use rhypedb_client::{ChangeKind, Client, Error, Notification, SubscriptionFilter};
use rhypedb_wire::protocol::{self, sync as wire_sync, WireEvent};

/// Build a `RESP_EVENT` payload (canonical `WireEvent` JSON) for one change.
fn event_payload(kind: &str, type_name: &str, id: u64, version: u64) -> Vec<u8> {
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), serde_json::json!("Ada"));
    let we = WireEvent {
        v: protocol::EVENT_FORMAT_TAG.to_string(),
        kind: kind.to_string(),
        type_name: type_name.to_string(),
        id: id.to_string(),
        version: version.to_string(),
        fields: Some(fields),
    };
    serde_json::to_vec(&we).unwrap()
}

/// Spawn a mock that accepts the client's (idle) query connection, then the
/// subscription connection, and hands the latter to `script` to drive. The query
/// connection is held open until `script` returns.
fn spawn_sub_server<F>(script: F) -> (SocketAddr, JoinHandle<()>)
where
    F: FnOnce(TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        // 1) the Client's request/response connection (idle in these tests).
        let (query_conn, _) = listener.accept().unwrap();
        // 2) the dedicated subscription connection.
        let (sub, _) = listener.accept().unwrap();
        script(sub);
        drop(query_conn);
    });
    (addr, handle)
}

/// Read the SUBSCRIBE frame and ack it with RESP_SUBSCRIBED (the common prelude).
fn accept_subscribe(stream: &mut TcpStream) -> u32 {
    let frame = wire_sync::read_frame(stream).unwrap();
    assert_eq!(frame.kind, protocol::REQ_SUBSCRIBE);
    // The filter must decode (proves the client encoded a valid one).
    protocol::decode_subscribe_filter(&frame.payload).unwrap();
    wire_sync::write_frame(stream, frame.req_id, protocol::RESP_SUBSCRIBED, &[]).unwrap();
    frame.req_id
}

#[test]
fn receives_events_and_lagged_then_ends_on_close() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 42, 7)).unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_SUBLAGGED, &[]).unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("delete", "User", 42, 9)).unwrap();
        // Drop `sub` → the client sees EOF on its next read.
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).unwrap();

    match sub.next_event().unwrap() {
        Notification::Change(c) => {
            assert_eq!(c.kind, ChangeKind::Create);
            assert_eq!((c.type_name.as_str(), c.id, c.version), ("User", 42, 7));
            assert_eq!(
                c.fields.as_ref().and_then(|m| m.get("name")),
                Some(&serde_json::json!("Ada"))
            );
        }
        other => panic!("expected a create event, got {other:?}"),
    }
    assert_eq!(sub.next_event().unwrap(), Notification::Lagged);
    match sub.next_event().unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id, c.version), (ChangeKind::Delete, 42, 9)),
        other => panic!("expected a delete event, got {other:?}"),
    }
    // The server closed → the next read is EOF, surfaced once as Io, then ended.
    assert!(matches!(sub.next_event(), Err(Error::Io(_))));
    assert!(matches!(sub.next_event(), Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn iterator_yields_events_then_the_terminating_error_then_none() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("update", "User", 1, 2)).unwrap();
        // then drop → EOF
    });

    let client = Client::connect(addr).unwrap();
    let sub = client.subscribe(SubscriptionFilter::all()).unwrap();

    let mut items = sub.into_iter();
    assert!(matches!(items.next(), Some(Ok(Notification::Change(_)))));
    // EOF surfaces once as Some(Err(..)) ...
    assert!(matches!(items.next(), Some(Err(Error::Io(_)))));
    // ... then the iterator is done.
    assert!(items.next().is_none());
    assert!(items.next().is_none());

    drop(client);
    handle.join().unwrap();
}

#[test]
fn subscribe_error_is_surfaced() {
    // The server rejects the SUBSCRIBE (e.g. an old server that doesn't support
    // subscriptions, or a cap breach) → Client::subscribe returns Error::Server.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let frame = wire_sync::read_frame(&mut sub).unwrap();
        assert_eq!(frame.kind, protocol::REQ_SUBSCRIBE);
        let p = protocol::encode_error_payload("subscriptions are not supported");
        wire_sync::write_frame(&mut sub, frame.req_id, protocol::RESP_ERROR, &p).unwrap();
    });

    let client = Client::connect(addr).unwrap();
    match client.subscribe(SubscriptionFilter::all()) {
        Err(Error::Server(m)) => assert!(m.contains("not supported"), "got: {m}"),
        other => panic!("expected a server error, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[test]
fn unsubscribe_drains_inflight_events_then_acks() {
    // After the client sends UNSUBSCRIBE, the server may still have a pushed
    // event queued ahead of the DONE ack. unsubscribe() must discard it and
    // return Ok on the DONE.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        // Wait for the UNSUBSCRIBE, then reply with an in-flight EVENT *then* DONE.
        let frame = wire_sync::read_frame(&mut sub).unwrap();
        assert_eq!(frame.kind, protocol::REQ_UNSUBSCRIBE);
        assert_eq!(protocol::decode_unsubscribe_payload(&frame.payload).unwrap(), h);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 5, 1)).unwrap();
        wire_sync::write_frame(&mut sub, frame.req_id, protocol::RESP_DONE, &[]).unwrap();
    });

    let client = Client::connect(addr).unwrap();
    let sub = client.subscribe(SubscriptionFilter::all()).unwrap();
    sub.unsubscribe().expect("unsubscribe drains the in-flight event and sees DONE");

    drop(client);
    handle.join().unwrap();
}

#[test]
fn malformed_event_errors_but_keeps_subscription_live() {
    // A syntactically-framed but semantically-bad event must surface an error
    // WITHOUT ending the subscription (the framing is intact) — the next valid
    // event still arrives.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        // (a) invalid JSON body.
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, b"not json").unwrap();
        // (b) valid envelope but unknown kind.
        let bad_kind = serde_json::to_vec(&WireEvent {
            v: protocol::EVENT_FORMAT_TAG.to_string(),
            kind: "frobnicate".to_string(),
            type_name: "User".to_string(),
            id: "1".to_string(),
            version: "1".to_string(),
            fields: None,
        })
        .unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &bad_kind).unwrap();
        // (c) a good event — must still be delivered.
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 9, 3)).unwrap();
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();

    // (a) bad JSON → an error, subscription still live.
    assert!(matches!(sub.next_event(), Err(Error::Io(_))));
    // (b) unknown kind → a deserialize error, subscription still live.
    assert!(matches!(sub.next_event(), Err(Error::Deserialize(_))));
    // (c) the good event arrives — proves the subscription never latched ended.
    match sub.next_event().unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Create, 9)),
        other => panic!("expected the good event, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[test]
fn unknown_event_format_tag_is_rejected() {
    // The client rejects an unrecognized envelope version loudly rather than
    // silently misinterpreting it.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        let future = serde_json::to_vec(&WireEvent {
            v: "rhypedb-event-v999".to_string(),
            kind: "create".to_string(),
            type_name: "User".to_string(),
            id: "1".to_string(),
            version: "1".to_string(),
            fields: None,
        })
        .unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &future).unwrap();
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();
    match sub.next_event() {
        Err(Error::Deserialize(m)) => assert!(m.contains("format tag"), "got: {m}"),
        other => panic!("expected a format-tag rejection, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[test]
fn shutdown_pushes_lagged_then_closes() {
    // The real server's graceful shutdown pushes a best-effort SUBLAGGED to each
    // live subscription, then closes — i.e. the LAST frame is SUBLAGGED, then EOF.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_SUBLAGGED, &[]).unwrap();
        // drop → EOF
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();
    assert_eq!(sub.next_event().unwrap(), Notification::Lagged);
    assert!(matches!(sub.next_event(), Err(Error::Io(_))));
    assert!(matches!(sub.next_event(), Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn query_connection_is_unaffected_by_subscription_pushes() {
    // The dedicated-connection design guarantees the client's request/response
    // connection never sees a push. Drive a ping on the query connection WHILE
    // the subscription connection streams events, and assert the ping gets a
    // correctly-correlated PONG (never an Event/SubLagged frame).
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        // 1) query connection: serve a ping.
        let (mut query_conn, _) = listener.accept().unwrap();
        // 2) subscription connection: ack + push events.
        let (mut sub, _) = listener.accept().unwrap();
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 1, 1)).unwrap();
        // Now answer the ping on the query connection.
        let frame = wire_sync::read_frame(&mut query_conn).unwrap();
        assert_eq!(frame.kind, protocol::REQ_PING);
        wire_sync::write_frame(&mut query_conn, frame.req_id, protocol::RESP_PONG, &[]).unwrap();
        // One more push, then close the subscription.
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("update", "User", 1, 2)).unwrap();
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();

    // First event on the sub connection.
    assert!(matches!(sub.next_event().unwrap(), Notification::Change(_)));
    // A ping on the QUERY connection returns a PONG — not a stray push.
    client.ping().expect("query connection serves a correctly-correlated PONG");
    // The subscription keeps delivering on its own connection.
    assert!(matches!(sub.next_event().unwrap(), Notification::Change(_)));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn stopper_unblocks_a_parked_next_event_on_another_thread() {
    // The flagship cross-thread cancel: a worker blocks in next_event (the mock
    // sends nothing after the ack); the main thread stops it via the stopper,
    // which must unblock the parked read and end the subscription.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        accept_subscribe(&mut sub);
        // Block until the client closes the socket (via the stopper).
        let _ = wire_sync::read_frame(&mut sub);
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();
    let stopper = sub.stopper().unwrap();

    let worker = thread::spawn(move || sub.next_event());
    // Stop from this thread. shutdown(Both) unblocks the worker's parked read
    // regardless of whether it stops before or after the read parks.
    stopper.stop().unwrap();
    let result = worker.join().unwrap();
    assert!(matches!(result, Err(Error::Io(_))), "got: {result:?}");

    drop(client);
    handle.join().unwrap();
}

#[test]
fn stopper_closes_the_connection() {
    // The stopper shuts the subscription's socket down; a subsequent read ends
    // the subscription. (shutdown(Both) also unblocks a concurrently-parked
    // next_event — OS-guaranteed — which is how a worker thread is stopped.)
    let (addr, handle) = spawn_sub_server(|mut sub| {
        accept_subscribe(&mut sub);
        // Block forever waiting for a request that never comes; the client closes
        // the socket via the stopper, so this read returns and the thread exits.
        let _ = wire_sync::read_frame(&mut sub);
    });

    let client = Client::connect(addr).unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).unwrap();
    let stopper = sub.stopper().unwrap();
    stopper.stop().unwrap();
    // The socket is shut down → the next read ends the subscription.
    assert!(matches!(sub.next_event(), Err(Error::Io(_))));

    drop(client);
    handle.join().unwrap();
}
