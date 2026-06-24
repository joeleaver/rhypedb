//! rhypedb-client — the official client for rhypedb.
//!
//! Speaks the binary TCP protocol over a single persistent connection, using
//! the shared [`rhypedb_wire`] codecs (so the wire format is single-sourced with
//! the server). No engine dependency.
//!
//! The default build is the **synchronous** [`Client`] and links zero of tokio.
//! A native `AsyncClient` (same wire codecs, same typed [`Query`]/[`Row`]
//! surface; its `subscribe` returns an `AsyncSubscription` that is a
//! `futures`-style `Stream`) is available behind the `async` cargo feature.
//!
//! ```no_run
//! use rhypedb_client::{Client, Query};
//! # #[derive(serde::Deserialize)] struct User { name: Option<String>, age: Option<u32> }
//! let client = Client::connect("127.0.0.1:4201")?;
//! let users = client.fetch::<User>(&Query::all("User").filter(".age > 18"))?;
//! for row in &users {
//!     println!("#{} {:?}", row.id, row.data.name);
//! }
//! # Ok::<(), rhypedb_client::Error>(())
//! ```
//!
//! For a query you run repeatedly, [`prepare`](Client::prepare) it once (the
//! server parses + caches it for the connection's lifetime) and re-run it with
//! [`fetch_prepared`](Client::fetch_prepared) / friends — no re-parse, no
//! re-sent query text. Precomputed vectors are bulk-loaded with
//! [`ingest_vectors`](Client::ingest_vectors). Live change events stream from
//! [`subscribe`](Client::subscribe), which returns a blocking [`Subscription`]
//! iterator over its own dedicated connection.
//!
//! A native async client (sharing this protocol core) will be added behind a
//! feature in a later increment — sync is the substrate, async is additive.

use std::marker::PhantomData;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use rhypedb_wire::object::Object;
use rhypedb_wire::protocol::{self, sync as wire_sync};
use serde::de::DeserializeOwned;

#[cfg(feature = "async")]
mod async_client;
mod query;
mod subscription;
#[cfg(feature = "async")]
pub use async_client::{AsyncClient, AsyncSubscription};
pub use query::{Query, Row};
pub use subscription::{ChangeNotification, Notification, Subscription, SubscriptionStopper};
// Re-export the filter model so callers can build subscriptions without naming
// `rhypedb-subscribe` (which is tokio-free, so the client stays runtime-free).
pub use rhypedb_subscribe::{ChangeKind, SubscriptionFilter};
// `Prepared` is a `pub` item in this crate root (it is branded with the issuing
// client), so it is exported as `rhypedb_client::Prepared` directly.

/// An error from the client or the server.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failed to establish the TCP connection.
    #[error("connect failed: {0}")]
    Connect(#[source] std::io::Error),
    /// Socket / framing I/O error during a request or response.
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),
    /// The server returned an error response (parse / validation / engine error).
    #[error("server error: {0}")]
    Server(String),
    /// The server replied with a frame kind that is not valid for this request.
    #[error("unexpected response frame kind: 0x{0:02x}")]
    UnexpectedResponse(u8),
    /// A query expected exactly one object (e.g. `create`) but the server
    /// returned a different result shape.
    #[error("expected a single object, got {0}")]
    UnexpectedShape(&'static str),
    /// An object's fields could not be deserialized into the requested type.
    #[error("deserialize error: {0}")]
    Deserialize(String),
    /// A previous request failed mid-flight (I/O error), so the connection's
    /// request/response framing may be desynced. The connection refuses further
    /// requests rather than risk returning a mis-correlated reply — reconnect.
    #[error("connection is closed after a previous I/O error; reconnect")]
    Closed,
    /// A vector batch's encoded payload exceeds the protocol frame cap
    /// ([`rhypedb_wire::protocol::MAX_FRAME_PAYLOAD`]). Split the rows across
    /// multiple [`ingest_vectors`](Client::ingest_vectors) calls.
    #[error(
        "vector batch too large: {rows} rows encode to {bytes} bytes (frame cap {cap})",
        cap = protocol::MAX_FRAME_PAYLOAD
    )]
    BatchTooLarge {
        /// Number of rows in the offending batch.
        rows: usize,
        /// Encoded payload size in bytes.
        bytes: usize,
    },
    /// A [`Prepared`] statement was used with a different [`Client`] than the one
    /// that prepared it. Statement ids are per-connection, so prepare it on this
    /// client first. (This is caught client-side before any I/O.)
    #[error("prepared statement belongs to a different client; prepare it on this client first")]
    ForeignStatement,
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Connection options. Defaults: TCP_NODELAY on, no timeouts.
#[derive(Debug, Clone, Default)]
pub struct ClientConfig {
    /// Bound on the initial TCP connect. `None` = OS default.
    pub connect_timeout: Option<Duration>,
    /// Per-read socket timeout. `None` = block indefinitely.
    pub read_timeout: Option<Duration>,
    /// Per-write socket timeout. `None` = block indefinitely.
    pub write_timeout: Option<Duration>,
}

/// The decoded result of a query, before typing.
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// A list result (`RESP_OBJECTS`) — filters, scans, `.similar`, etc.
    Objects(Vec<Object>),
    /// A single object (`RESP_SINGLE`) — `get`, `create`, `update`.
    Single(Object),
    /// A void result (`RESP_DONE`) — `delete`, `link`, `unlink`.
    Done,
}

impl QueryResult {
    /// Flatten to a list: `Single` → one element, `Done` → empty.
    pub fn into_objects(self) -> Vec<Object> {
        match self {
            QueryResult::Objects(v) => v,
            QueryResult::Single(o) => vec![o],
            QueryResult::Done => Vec::new(),
        }
    }

    fn shape(&self) -> &'static str {
        match self {
            QueryResult::Objects(_) => "a list",
            QueryResult::Single(_) => "a single object",
            QueryResult::Done => "no result",
        }
    }

    /// Materialize every object into `T` (`fetch`-shaped results).
    fn into_typed_rows<T: DeserializeOwned>(self) -> Result<Vec<Row<T>>> {
        self.into_objects().iter().map(Row::from_object).collect()
    }

    /// Materialize a result that must be exactly one object into `T`
    /// (`create`/`update`-shaped results).
    fn into_typed_single<T: DeserializeOwned>(self) -> Result<Row<T>> {
        match self {
            QueryResult::Single(o) => Row::from_object(&o),
            QueryResult::Objects(mut v) if v.len() == 1 => Row::from_object(&v.remove(0)),
            other => Err(Error::UnexpectedShape(other.shape())),
        }
    }
}

/// One persistent connection plus its request-id counter. Guarded by the
/// client's mutex; the binary protocol is strictly request/response on a
/// non-subscription connection, so the frame read after a write IS the reply.
struct Connection {
    stream: TcpStream,
    next_req_id: u32,
    /// Set once an I/O error interrupts a round-trip: the stream may hold a
    /// partial/unread reply, so reusing it could mis-correlate. Latches the
    /// connection closed (fail-fast) instead of silently desyncing.
    dead: bool,
}

impl Connection {
    /// Send one request frame and read the one reply frame. On any I/O failure
    /// the connection is marked dead (see [`Connection::dead`]).
    fn round_trip(&mut self, kind: u8, payload: &[u8]) -> Result<protocol::Frame> {
        if self.dead {
            return Err(Error::Closed);
        }
        let req_id = self.next_req_id;
        self.next_req_id = self.next_req_id.wrapping_add(1);
        // Any I/O error here can leave the stream framing desynced — latch dead.
        let result = wire_sync::write_frame(&mut self.stream, req_id, kind, payload)
            .map_err(Error::Io)
            .and_then(|()| wire_sync::read_frame(&mut self.stream).map_err(Error::Io));
        if result.is_err() {
            self.dead = true;
        }
        result
    }
}

/// Process-wide source of per-`Client` identities, used to brand [`Prepared`]
/// statements so one client can't run another's statement id (see [`Prepared`]).
static NEXT_CLIENT_ID: AtomicU64 = AtomicU64::new(1);

/// A statement prepared on a [`Client`]'s connection (see [`Client::prepare`]).
///
/// The server parses + caches the query once and hands back this handle; running
/// it via [`Client::execute_prepared`] / [`Client::fetch_prepared`] / … re-runs
/// it with **no re-parse** and **without re-sending the query text**. There are
/// no bind parameters — a prepared statement is a *fixed* query, worthwhile when
/// you run the exact same query many times.
///
/// Statement ids are scoped to the connection that created them and are released
/// when that `Client` is dropped. A `Prepared` is branded with its originating
/// client, so handing it to a *different* `Client` is rejected with
/// [`Error::ForeignStatement`] rather than silently running that client's
/// unrelated statement of the same id.
#[derive(Debug, Clone, Copy)]
pub struct Prepared<T> {
    stmt_id: u64,
    client_id: u64,
    _row: PhantomData<fn() -> T>,
}

/// A connected rhypedb client. `Send + Sync` — share it behind an `Arc`; calls
/// serialize on the single underlying connection.
pub struct Client {
    conn: Mutex<Connection>,
    /// This client's brand, stamped into every [`Prepared`] it issues.
    client_id: u64,
    /// The resolved server address this client connected to, so
    /// [`subscribe`](Client::subscribe) can dial a fresh dedicated connection.
    peer_addr: SocketAddr,
    /// Connect/write timeouts reused for subscription connections. (The
    /// `read_timeout` is intentionally not reused — a subscription blocks waiting
    /// for events; see [`subscribe`](Client::subscribe).)
    connect_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
}

impl Client {
    /// Connect to a rhypedb server's binary TCP port with default options.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        Self::connect_with(addr, &ClientConfig::default())
    }

    /// Connect with explicit [`ClientConfig`] options.
    pub fn connect_with<A: ToSocketAddrs>(addr: A, config: &ClientConfig) -> Result<Self> {
        let stream = match config.connect_timeout {
            Some(timeout) => {
                // connect_timeout needs a resolved SocketAddr; try each until one
                // connects, mirroring TcpStream::connect's resolution.
                let addrs = addr.to_socket_addrs().map_err(Error::Connect)?;
                let mut last = None;
                let mut connected = None;
                for sa in addrs {
                    match TcpStream::connect_timeout(&sa, timeout) {
                        Ok(s) => {
                            connected = Some(s);
                            break;
                        }
                        Err(e) => last = Some(e),
                    }
                }
                connected.ok_or_else(|| {
                    Error::Connect(last.unwrap_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "no addresses to connect to",
                        )
                    }))
                })?
            }
            None => TcpStream::connect(addr).map_err(Error::Connect)?,
        };
        stream.set_nodelay(true).map_err(Error::Connect)?;
        stream
            .set_read_timeout(config.read_timeout)
            .map_err(Error::Connect)?;
        stream
            .set_write_timeout(config.write_timeout)
            .map_err(Error::Connect)?;
        let peer_addr = stream.peer_addr().map_err(Error::Connect)?;
        Ok(Client {
            conn: Mutex::new(Connection {
                stream,
                next_req_id: 1,
                dead: false,
            }),
            client_id: NEXT_CLIENT_ID.fetch_add(1, Ordering::Relaxed),
            peer_addr,
            connect_timeout: config.connect_timeout,
            write_timeout: config.write_timeout,
        })
    }

    /// Liveness check: `PING` → `PONG`.
    pub fn ping(&self) -> Result<()> {
        let frame = self.round_trip(protocol::REQ_PING, &[])?;
        match frame.kind {
            protocol::RESP_PONG => Ok(()),
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Run a raw query string, returning the untyped [`QueryResult`].
    pub fn query(&self, query: &str) -> Result<QueryResult> {
        let frame = self.round_trip(protocol::REQ_QUERY, &protocol::encode_query_payload(query))?;
        decode_query_frame(&frame)
    }

    /// Run a typed [`Query`] and materialize every row into `T`.
    pub fn fetch<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Vec<Row<T>>> {
        self.query(query.as_str())?.into_typed_rows()
    }

    /// Run a typed [`Query`] and return the first row, if any.
    pub fn fetch_one<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Option<Row<T>>> {
        Ok(self.fetch(query)?.into_iter().next())
    }

    /// Run a typed write that returns the affected object (`create` / `update`),
    /// materializing it into `T`.
    pub fn create<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Row<T>> {
        self.query(query.as_str())?.into_typed_single()
    }

    /// Run a query for its effect, discarding the result shape (`delete`,
    /// `link`, `unlink`, or a write whose returned object you don't need).
    pub fn execute<T>(&self, query: &Query<T>) -> Result<QueryResult> {
        self.query(query.as_str())
    }

    /// Prepare a typed [`Query`] for repeated execution on this connection. The
    /// server parses + caches it and returns a [`Prepared`] handle; run it with
    /// [`execute_prepared`](Self::execute_prepared) / [`fetch_prepared`](Self::fetch_prepared)
    /// / friends with no re-parse and no re-sent query text. The handle is valid
    /// until this client is dropped.
    pub fn prepare<T>(&self, query: &Query<T>) -> Result<Prepared<T>> {
        let frame =
            self.round_trip(protocol::REQ_PREPARE, &protocol::encode_prepare_payload(query.as_str()))?;
        match frame.kind {
            protocol::RESP_PREPARED => {
                let stmt_id = protocol::decode_prepared_payload(&frame.payload).map_err(Error::Io)?;
                Ok(Prepared {
                    stmt_id,
                    client_id: self.client_id,
                    _row: PhantomData,
                })
            }
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Run a [`Prepared`] statement, returning the untyped [`QueryResult`].
    pub fn execute_prepared<T>(&self, stmt: &Prepared<T>) -> Result<QueryResult> {
        decode_query_frame(&self.run_prepared(stmt)?)
    }

    /// Run a [`Prepared`] statement and materialize every row into `T`.
    pub fn fetch_prepared<T: DeserializeOwned>(&self, stmt: &Prepared<T>) -> Result<Vec<Row<T>>> {
        self.execute_prepared(stmt)?.into_typed_rows()
    }

    /// Run a [`Prepared`] statement and return the first row, if any.
    pub fn fetch_one_prepared<T: DeserializeOwned>(
        &self,
        stmt: &Prepared<T>,
    ) -> Result<Option<Row<T>>> {
        Ok(self.fetch_prepared(stmt)?.into_iter().next())
    }

    /// Run a [`Prepared`] write that returns the affected object, materializing
    /// it into `T`.
    pub fn create_prepared<T: DeserializeOwned>(&self, stmt: &Prepared<T>) -> Result<Row<T>> {
        self.execute_prepared(stmt)?.into_typed_single()
    }

    /// Bulk-load precomputed vectors into one `Vector` field. `rows` is a slice
    /// of `(object_id, vector)`; the vector storage is generic (`Vec<f32>`,
    /// `&[f32]`, …). The server validates and applies the whole batch atomically:
    /// on success every row is ingested (returns the row count), otherwise
    /// nothing is. Returns [`Error::BatchTooLarge`] if a single batch would
    /// exceed the protocol frame cap — split it across calls (each call is its
    /// own atomic batch).
    pub fn ingest_vectors<V: AsRef<[f32]>>(
        &self,
        type_name: &str,
        field_name: &str,
        rows: &[(u64, V)],
    ) -> Result<usize> {
        let payload = protocol::encode_vector_batch_payload(type_name, field_name, rows);
        if payload.len() > protocol::MAX_FRAME_PAYLOAD {
            return Err(Error::BatchTooLarge {
                rows: rows.len(),
                bytes: payload.len(),
            });
        }
        let frame = self.round_trip(protocol::REQ_VECTOR_BATCH, &payload)?;
        match frame.kind {
            protocol::RESP_DONE => Ok(rows.len()),
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Open a change-event [`Subscription`] on a **dedicated** connection to the
    /// same server. It is independent of this client's query connection — the
    /// server makes a subscribed connection refuse queries, so pushed events
    /// never race query replies — and is a blocking iterator of [`Notification`]s.
    /// Each call opens its own connection; reconcile a [`Notification::Lagged`]
    /// by re-querying with this client.
    ///
    /// The subscription connection is dialed to the address this client resolved
    /// to at connect time and honors the configured `connect_timeout`. The
    /// configured `read_timeout` is intentionally NOT applied — a subscription
    /// blocks waiting for sparse events; cancel a blocked read with a
    /// [`SubscriptionStopper`](crate::SubscriptionStopper).
    pub fn subscribe(&self, filter: SubscriptionFilter) -> Result<Subscription> {
        Subscription::open(self.peer_addr, self.connect_timeout, self.write_timeout, &filter)
    }

    /// Verify a [`Prepared`] belongs to this client (statement ids are
    /// per-connection), then round-trip an `Execute`. Branding is checked before
    /// any I/O, so a foreign handle never touches the wire.
    fn run_prepared<T>(&self, stmt: &Prepared<T>) -> Result<protocol::Frame> {
        if stmt.client_id != self.client_id {
            return Err(Error::ForeignStatement);
        }
        self.round_trip(protocol::REQ_EXECUTE, &protocol::encode_execute_payload(stmt.stmt_id))
    }

    fn round_trip(&self, kind: u8, payload: &[u8]) -> Result<protocol::Frame> {
        let mut conn = self
            .conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        conn.round_trip(kind, payload)
    }
}

/// Map a query/execute reply frame to a [`QueryResult`] or an [`Error`].
fn decode_query_frame(frame: &protocol::Frame) -> Result<QueryResult> {
    match frame.kind {
        protocol::RESP_OBJECTS => Ok(QueryResult::Objects(
            protocol::decode_objects_payload(&frame.payload).map_err(Error::Io)?,
        )),
        protocol::RESP_SINGLE => {
            let (obj, _) = protocol::decode_object(&frame.payload, 0).map_err(Error::Io)?;
            Ok(QueryResult::Single(obj))
        }
        protocol::RESP_DONE => Ok(QueryResult::Done),
        protocol::RESP_ERROR => Err(server_error(&frame.payload)),
        other => Err(Error::UnexpectedResponse(other)),
    }
}

/// Decode a `RESP_ERROR` payload into [`Error::Server`] (falling back if the
/// message itself is malformed — still surface it as a server error, not I/O).
pub(crate) fn server_error(payload: &[u8]) -> Error {
    match protocol::decode_error_payload(payload) {
        Ok(msg) => Error::Server(msg),
        Err(_) => Error::Server(String::from_utf8_lossy(payload).into_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_wire::object::{FieldMap, Value};

    fn frame(kind: u8, payload: Vec<u8>) -> protocol::Frame {
        protocol::Frame {
            req_id: 1,
            kind,
            payload,
        }
    }

    fn obj(id: u64) -> Object {
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("x".into()));
        Object {
            type_name: "T".into(),
            id,
            fields,
            raw_fields: None,
        }
    }

    #[test]
    fn decode_query_frame_maps_each_response_kind() {
        // RESP_OBJECTS → Objects
        let payload = protocol::encode_objects_payload(&[obj(1), obj(2)]);
        match decode_query_frame(&frame(protocol::RESP_OBJECTS, payload)).unwrap() {
            QueryResult::Objects(v) => assert_eq!(v.len(), 2),
            other => panic!("{other:?}"),
        }
        // RESP_SINGLE → Single
        let mut buf = Vec::new();
        protocol::encode_object(&obj(7), &mut buf);
        match decode_query_frame(&frame(protocol::RESP_SINGLE, buf)).unwrap() {
            QueryResult::Single(o) => assert_eq!(o.id, 7),
            other => panic!("{other:?}"),
        }
        // RESP_DONE → Done
        assert!(matches!(
            decode_query_frame(&frame(protocol::RESP_DONE, vec![])).unwrap(),
            QueryResult::Done
        ));
        // RESP_ERROR → Error::Server with the decoded message
        let payload = protocol::encode_error_payload("boom");
        match decode_query_frame(&frame(protocol::RESP_ERROR, payload)) {
            Err(Error::Server(m)) => assert_eq!(m, "boom"),
            other => panic!("{other:?}"),
        }
        // A reply kind that is not valid for a query → UnexpectedResponse
        match decode_query_frame(&frame(protocol::RESP_PONG, vec![])) {
            Err(Error::UnexpectedResponse(k)) => assert_eq!(k, protocol::RESP_PONG),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn server_error_falls_back_on_malformed_payload() {
        // A truncated error payload still surfaces as a server error, not I/O.
        match server_error(&[0xFF]) {
            Error::Server(_) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn into_objects_flattens() {
        assert_eq!(QueryResult::Single(obj(1)).into_objects().len(), 1);
        assert_eq!(QueryResult::Done.into_objects().len(), 0);
        assert_eq!(
            QueryResult::Objects(vec![obj(1), obj(2)]).into_objects().len(),
            2
        );
    }
}
