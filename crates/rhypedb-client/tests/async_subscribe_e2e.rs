//! Async subscription end-to-end tests over real TCP sockets against a sync mock
//! server that speaks the binary protocol via the shared `rhypedb-wire` codecs.
//!
//! `AsyncClient::subscribe` opens a SECOND, dedicated connection (separate from
//! the client's query connection), so every mock here accepts two connections:
//! the idle query connection first, then the subscription connection it drives.
//! Run with `cargo test -p rhypedb-client --features async`.
#![cfg(feature = "async")]

use std::collections::HashMap;
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rhypedb_client::{AsyncClient, ChangeKind, Error, Notification, SubscriptionFilter};
use rhypedb_wire::protocol::{self, sync as wire_sync, WireEvent};
use tokio_stream::StreamExt;

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
        origin: None,
    };
    serde_json::to_vec(&we).unwrap()
}

/// Spawn a mock that accepts the client's (idle) query connection, then the
/// subscription connection, and hands the latter to `script` to drive.
fn spawn_sub_server<F>(script: F) -> (SocketAddr, JoinHandle<()>)
where
    F: FnOnce(TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (query_conn, _) = listener.accept().unwrap();
        let (sub, _) = listener.accept().unwrap();
        script(sub);
        drop(query_conn);
    });
    (addr, handle)
}

fn accept_subscribe(stream: &mut TcpStream) -> u32 {
    let frame = wire_sync::read_frame(stream).unwrap();
    assert_eq!(frame.kind, protocol::REQ_SUBSCRIBE);
    protocol::decode_subscribe_filter(&frame.payload).unwrap();
    wire_sync::write_frame(stream, frame.req_id, protocol::RESP_SUBSCRIBED, &[]).unwrap();
    frame.req_id
}

#[tokio::test]
async fn next_event_receives_events_and_lagged_then_ends_on_close() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 42, 7)).unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_SUBLAGGED, &[]).unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("delete", "User", 42, 9)).unwrap();
        // Drop `sub` → the client sees EOF on its next read.
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).await.unwrap();

    match sub.next_event().await.unwrap() {
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
    assert_eq!(sub.next_event().await.unwrap(), Notification::Lagged);
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id, c.version), (ChangeKind::Delete, 42, 9)),
        other => panic!("expected a delete event, got {other:?}"),
    }
    // EOF surfaces once as Io, then Closed.
    assert!(matches!(sub.next_event().await, Err(Error::Io(_))));
    assert!(matches!(sub.next_event().await, Err(Error::Closed)));

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn stream_yields_events_then_the_terminating_error_then_none() {
    // The Stream impl: drive it with tokio_stream::StreamExt::next.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("update", "User", 1, 2)).unwrap();
        // then drop → EOF
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).await.unwrap();

    assert!(matches!(sub.next().await, Some(Ok(Notification::Change(_)))));
    // EOF surfaces once as Some(Err(..)) ...
    assert!(matches!(sub.next().await, Some(Err(Error::Io(_)))));
    // ... then the stream is done.
    assert!(sub.next().await.is_none());
    assert!(sub.next().await.is_none());

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn subscribe_error_is_surfaced() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let frame = wire_sync::read_frame(&mut sub).unwrap();
        assert_eq!(frame.kind, protocol::REQ_SUBSCRIBE);
        let p = protocol::encode_error_payload("subscriptions are not supported");
        wire_sync::write_frame(&mut sub, frame.req_id, protocol::RESP_ERROR, &p).unwrap();
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    match client.subscribe(SubscriptionFilter::all()).await {
        Err(Error::Server(m)) => assert!(m.contains("not supported"), "got: {m}"),
        other => panic!("expected a server error, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn unsubscribe_drains_inflight_events_then_acks() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        let frame = wire_sync::read_frame(&mut sub).unwrap();
        assert_eq!(frame.kind, protocol::REQ_UNSUBSCRIBE);
        assert_eq!(protocol::decode_unsubscribe_payload(&frame.payload).unwrap(), h);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 5, 1)).unwrap();
        wire_sync::write_frame(&mut sub, frame.req_id, protocol::RESP_DONE, &[]).unwrap();
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let sub = client.subscribe(SubscriptionFilter::all()).await.unwrap();
    sub.unsubscribe().await.expect("unsubscribe drains the in-flight event and sees DONE");

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn malformed_event_errors_but_keeps_subscription_live() {
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
            origin: None,
        })
        .unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &bad_kind).unwrap();
        // (c) a good event — must still be delivered.
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 9, 3)).unwrap();
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).await.unwrap();

    assert!(matches!(sub.next_event().await, Err(Error::Io(_))));
    assert!(matches!(sub.next_event().await, Err(Error::Deserialize(_))));
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Create, 9)),
        other => panic!("expected the good event, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn unknown_event_format_tag_is_rejected() {
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        let future = serde_json::to_vec(&WireEvent {
            v: "rhypedb-event-v999".to_string(),
            kind: "create".to_string(),
            type_name: "User".to_string(),
            id: "1".to_string(),
            version: "1".to_string(),
            fields: None,
            origin: None,
        })
        .unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &future).unwrap();
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).await.unwrap();
    match sub.next_event().await {
        Err(Error::Deserialize(m)) => assert!(m.contains("format tag"), "got: {m}"),
        other => panic!("expected a format-tag rejection, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn query_connection_is_unaffected_by_subscription_pushes() {
    // The dedicated-connection design guarantees the query connection never sees
    // a push. Ping on the query connection WHILE the subscription streams events.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut query_conn, _) = listener.accept().unwrap();
        let (mut sub, _) = listener.accept().unwrap();
        let h = accept_subscribe(&mut sub);
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 1, 1)).unwrap();
        let frame = wire_sync::read_frame(&mut query_conn).unwrap();
        assert_eq!(frame.kind, protocol::REQ_PING);
        wire_sync::write_frame(&mut query_conn, frame.req_id, protocol::RESP_PONG, &[]).unwrap();
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("update", "User", 1, 2)).unwrap();
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::all()).await.unwrap();

    assert!(matches!(sub.next_event().await.unwrap(), Notification::Change(_)));
    client.ping().await.expect("query connection serves a correctly-correlated PONG");
    assert!(matches!(sub.next_event().await.unwrap(), Notification::Change(_)));

    drop(client);
    handle.join().unwrap();
}

#[tokio::test]
async fn dropping_a_next_future_midframe_is_cancel_safe() {
    // The flagship async-specific guarantee: the Stream read goes through the
    // cancel-safe FrameReader, so a `next()` future dropped while a frame is only
    // PARTIALLY on the wire loses nothing — the next poll resumes and delivers the
    // whole frame. The mock writes the second event split in two halves with a gap
    // between; the client times out (dropping the future) inside that gap.
    let (addr, handle) = spawn_sub_server(|mut sub| {
        let h = accept_subscribe(&mut sub);
        // Event 1: delivered whole.
        wire_sync::write_frame(&mut sub, h, protocol::RESP_EVENT, &event_payload("create", "User", 1, 1)).unwrap();
        // Event 2: build the frame bytes, then dribble them out in two halves with
        // a delay so the client's short timeout fires mid-frame.
        let mut framed = Vec::new();
        wire_sync::write_frame(&mut framed, h, protocol::RESP_EVENT, &event_payload("update", "User", 2, 2)).unwrap();
        let mid = framed.len() / 2;
        sub.write_all(&framed[..mid]).unwrap();
        sub.flush().unwrap();
        thread::sleep(Duration::from_millis(150));
        sub.write_all(&framed[mid..]).unwrap();
        sub.flush().unwrap();
        // Keep the connection open a moment so the client can read the full frame.
        thread::sleep(Duration::from_millis(150));
    });

    let client = AsyncClient::connect(addr).await.unwrap();
    let mut sub = client.subscribe(SubscriptionFilter::for_type("User")).await.unwrap();

    // Event 1 arrives whole.
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Create, 1)),
        other => panic!("expected event 1, got {other:?}"),
    }

    // Now race event 2 against a short timeout that fires while only the first
    // half is on the wire. The timed-out `next_event` future is DROPPED mid-frame.
    let timed_out = tokio::time::timeout(Duration::from_millis(30), sub.next_event()).await;
    assert!(timed_out.is_err(), "the timeout must fire before the full frame arrives");

    // Resume: the FrameReader kept the buffered first half, so the next read
    // completes event 2 intact — proving cancel-safety (no desync, no loss).
    match sub.next_event().await.unwrap() {
        Notification::Change(c) => assert_eq!((c.kind, c.id), (ChangeKind::Update, 2)),
        other => panic!("expected event 2 intact after a mid-frame cancel, got {other:?}"),
    }

    drop(client);
    handle.join().unwrap();
}
