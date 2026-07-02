//! The native-async client ([`AsyncClient`]) and its subscription stream
//! ([`AsyncSubscription`]), behind the `async` cargo feature.
//!
//! This is a thin async parallel to the synchronous [`Client`](crate::Client):
//! it shares the wire codecs, the typed [`Query`]/[`Row`] surface, the
//! [`QueryResult`] decode, the [`Prepared`] branding, and the [`Notification`]
//! model — only the I/O plumbing (a tokio [`TcpStream`] behind a tokio
//! [`Mutex`](tokio::sync::Mutex), and a [`Stream`] subscription) is distinct. The
//! request/response semantics (one persistent connection; latch-dead on a
//! mid-flight I/O error; per-connection statement ids; one frame per vector
//! batch; a dedicated connection per subscription) are identical to the sync
//! client; see its docs for the rationale.

use std::future::Future;
use std::io;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};
use std::time::Duration;

use rhypedb_subscribe::SubscriptionFilter;
use rhypedb_wire::protocol::{self, FrameReader};
use serde::de::DeserializeOwned;
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio::sync::Mutex;
use tokio_stream::Stream;

use crate::{
    decode_query_frame, server_error, ChangeNotification, ClientConfig, Error, Notification,
    Prepared, Query, QueryResult, Result, Row, NEXT_CLIENT_ID,
};

/// Run `fut`, failing with a `TimedOut` I/O error if `dur` elapses first. `None`
/// means no deadline (await to completion). On timeout the caller treats it like
/// any other I/O error — i.e. it latches the connection dead (a partial
/// read/write may have desynced framing).
async fn with_timeout<F, T>(dur: Option<Duration>, fut: F) -> io::Result<T>
where
    F: Future<Output = io::Result<T>>,
{
    match dur {
        Some(d) => match tokio::time::timeout(d, fut).await {
            Ok(r) => r,
            Err(_elapsed) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "operation timed out",
            )),
        },
        None => fut.await,
    }
}

/// Connect a tokio [`TcpStream`], honoring an optional connect deadline.
async fn dial<A: ToSocketAddrs>(addr: A, connect_timeout: Option<Duration>) -> Result<TcpStream> {
    let connect = TcpStream::connect(addr);
    let stream = match connect_timeout {
        Some(d) => tokio::time::timeout(d, connect)
            .await
            .map_err(|_elapsed| {
                Error::Connect(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "connect timed out",
                ))
            })?
            .map_err(Error::Connect)?,
        None => connect.await.map_err(Error::Connect)?,
    };
    stream.set_nodelay(true).map_err(Error::Connect)?;
    Ok(stream)
}

/// One persistent async connection plus its request-id counter. Guarded by the
/// client's tokio mutex; the binary protocol is strictly request/response on a
/// non-subscription connection, so the frame read after a write IS the reply.
struct AsyncConnection {
    stream: TcpStream,
    next_req_id: u32,
    /// Latches once an I/O error interrupts a round-trip (a partial/unread reply
    /// may have desynced framing) so the connection fails fast instead of
    /// risking a mis-correlated reply. Mirrors the sync client.
    dead: bool,
}

impl AsyncConnection {
    /// Send one request frame and read the one reply frame. On any I/O failure
    /// (including a timeout) the connection is marked dead.
    ///
    /// The dead-latch is **pessimistic across the await**: we set `dead` *before*
    /// the round-trip future and clear it only on a fully-completed success. This
    /// is what makes the latch hold under *cancellation* — an async caller can
    /// drop this future mid-flight (`tokio::time::timeout`, `select!`, an aborted
    /// task), and a dropped read is not cancel-safe (it may have consumed part of
    /// the reply from the socket). If we latched only on the `Err` path, a dropped
    /// future would leave `dead == false` with an unread reply tail buffered in
    /// the socket, and the next request would silently mis-correlate that stale
    /// reply (the protocol does not echo a checkable request id). Assuming the
    /// worst across the await closes that hole: a cancelled call leaves the
    /// connection `Closed`, so the next call fails fast — reconnect. (The
    /// synchronous `Client` cannot be cancelled mid-call, so it has no such hole.)
    async fn round_trip(
        &mut self,
        kind: u8,
        payload: &[u8],
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
    ) -> Result<protocol::Frame> {
        if self.dead {
            return Err(Error::Closed);
        }
        let req_id = self.next_req_id;
        self.next_req_id = self.next_req_id.wrapping_add(1);
        // Arm the latch before the await; a dropped/cancelled future then leaves
        // the connection dead. Only a clean, fully-read reply disarms it.
        self.dead = true;
        let result = self
            .do_round_trip(req_id, kind, payload, read_timeout, write_timeout)
            .await;
        if result.is_ok() {
            self.dead = false;
        }
        result
    }

    async fn do_round_trip(
        &mut self,
        req_id: u32,
        kind: u8,
        payload: &[u8],
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
    ) -> Result<protocol::Frame> {
        // Build the whole frame in one buffer → a single write_all (matches the
        // sync client's contiguous `sync::write_frame`).
        let mut buf = Vec::with_capacity(9 + payload.len());
        with_timeout(
            write_timeout,
            protocol::write_frame_buffered(&mut self.stream, &mut buf, req_id, kind, |b| {
                b.extend_from_slice(payload)
            }),
        )
        .await
        .map_err(Error::Io)?;
        with_timeout(read_timeout, protocol::read_frame(&mut self.stream))
            .await
            .map_err(Error::Io)
    }
}

/// A connected async rhypedb client. `Send + Sync` — share it behind an `Arc`;
/// calls serialize on the single underlying connection (held across the await by
/// a tokio mutex). The async parallel to [`Client`](crate::Client) with the same
/// surface; see the module docs.
pub struct AsyncClient {
    conn: Mutex<AsyncConnection>,
    /// This client's brand, stamped into every [`Prepared`] it issues. Drawn from
    /// the same process-global counter as the sync client, so brands never alias.
    client_id: u64,
    /// The resolved server address, so [`subscribe`](Self::subscribe) can dial a
    /// fresh dedicated connection.
    peer_addr: SocketAddr,
    connect_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
}

impl AsyncClient {
    /// Connect to a rhypedb server's binary TCP port with default options.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        Self::connect_with(addr, &ClientConfig::default()).await
    }

    /// Connect with explicit [`ClientConfig`] options. (The async client applies
    /// `read_timeout`/`write_timeout` per round-trip via [`tokio::time::timeout`]
    /// rather than as socket options.)
    pub async fn connect_with<A: ToSocketAddrs>(addr: A, config: &ClientConfig) -> Result<Self> {
        let stream = dial(addr, config.connect_timeout).await?;
        let peer_addr = stream.peer_addr().map_err(Error::Connect)?;
        Ok(AsyncClient {
            conn: Mutex::new(AsyncConnection {
                stream,
                next_req_id: 1,
                dead: false,
            }),
            client_id: NEXT_CLIENT_ID.fetch_add(1, Ordering::Relaxed),
            peer_addr,
            connect_timeout: config.connect_timeout,
            read_timeout: config.read_timeout,
            write_timeout: config.write_timeout,
        })
    }

    /// Liveness check: `PING` → `PONG`.
    pub async fn ping(&self) -> Result<()> {
        let frame = self.round_trip(protocol::REQ_PING, &[]).await?;
        match frame.kind {
            protocol::RESP_PONG => Ok(()),
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Run a raw query string, returning the untyped [`QueryResult`].
    pub async fn query(&self, query: &str) -> Result<QueryResult> {
        let frame = self
            .round_trip(protocol::REQ_QUERY, &protocol::encode_query_payload(query))
            .await?;
        decode_query_frame(&frame)
    }

    /// Run a typed [`Query`] and materialize every row into `T`.
    pub async fn fetch<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Vec<Row<T>>> {
        self.query(query.as_str()).await?.into_typed_rows()
    }

    /// Run a typed [`Query`] and return the first row, if any.
    pub async fn fetch_one<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Option<Row<T>>> {
        Ok(self.fetch(query).await?.into_iter().next())
    }

    /// Run a typed write that returns the affected object (`create` / `update`),
    /// materializing it into `T`.
    pub async fn create<T: DeserializeOwned>(&self, query: &Query<T>) -> Result<Row<T>> {
        self.query(query.as_str()).await?.into_typed_single()
    }

    /// Run a query for its effect, discarding the result shape (`delete`, `link`,
    /// `unlink`, or a write whose returned object you don't need).
    pub async fn execute<T>(&self, query: &Query<T>) -> Result<QueryResult> {
        self.query(query.as_str()).await
    }

    /// Prepare a typed [`Query`] for repeated execution on this connection (server
    /// parses + caches it, returns a [`Prepared`] handle). See the sync
    /// [`Client::prepare`](crate::Client::prepare) for the semantics — the handle
    /// is branded with this client and valid until it is dropped.
    pub async fn prepare<T>(&self, query: &Query<T>) -> Result<Prepared<T>> {
        let frame = self
            .round_trip(
                protocol::REQ_PREPARE,
                &protocol::encode_prepare_payload(query.as_str()),
            )
            .await?;
        match frame.kind {
            protocol::RESP_PREPARED => {
                let stmt_id =
                    protocol::decode_prepared_payload(&frame.payload).map_err(Error::Io)?;
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
    pub async fn execute_prepared<T>(&self, stmt: &Prepared<T>) -> Result<QueryResult> {
        decode_query_frame(&self.run_prepared(stmt).await?)
    }

    /// Run a [`Prepared`] statement and materialize every row into `T`.
    pub async fn fetch_prepared<T: DeserializeOwned>(
        &self,
        stmt: &Prepared<T>,
    ) -> Result<Vec<Row<T>>> {
        self.execute_prepared(stmt).await?.into_typed_rows()
    }

    /// Run a [`Prepared`] statement and return the first row, if any.
    pub async fn fetch_one_prepared<T: DeserializeOwned>(
        &self,
        stmt: &Prepared<T>,
    ) -> Result<Option<Row<T>>> {
        Ok(self.fetch_prepared(stmt).await?.into_iter().next())
    }

    /// Run a [`Prepared`] write that returns the affected object, materializing it
    /// into `T`.
    pub async fn create_prepared<T: DeserializeOwned>(&self, stmt: &Prepared<T>) -> Result<Row<T>> {
        self.execute_prepared(stmt).await?.into_typed_single()
    }

    /// Bulk-load precomputed vectors into one `Vector` field (one atomic batch per
    /// call, one frame). See the sync
    /// [`Client::ingest_vectors`](crate::Client::ingest_vectors) — returns
    /// [`Error::BatchTooLarge`] (before any write) if the encoded batch would
    /// exceed the protocol frame cap.
    pub async fn ingest_vectors<V: AsRef<[f32]>>(
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
        let frame = self.round_trip(protocol::REQ_VECTOR_BATCH, &payload).await?;
        match frame.kind {
            protocol::RESP_DONE => Ok(rows.len()),
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Open a change-event [`AsyncSubscription`] on a **dedicated** connection to
    /// the same server (independent of this client's query connection — the server
    /// makes a subscribed connection refuse queries). It is a `Stream` of
    /// [`Notification`]s; reconcile a [`Notification::Lagged`] by re-querying.
    ///
    /// The subscription connection is dialed to the address this client resolved
    /// to and honors the configured `connect_timeout`. The configured
    /// `read_timeout` is intentionally NOT applied to the event read — a
    /// subscription waits for sparse events; to stop it, stop polling / drop the
    /// stream (drop closes the connection, which the server treats as an implicit
    /// unsubscribe) or call [`AsyncSubscription::unsubscribe`].
    pub async fn subscribe(&self, filter: SubscriptionFilter) -> Result<AsyncSubscription> {
        AsyncSubscription::open(
            self.peer_addr,
            self.connect_timeout,
            self.write_timeout,
            &filter,
        )
        .await
    }

    /// Verify a [`Prepared`] belongs to this client (statement ids are
    /// per-connection), then round-trip an `Execute`. Branding is checked before
    /// any I/O, so a foreign handle never touches the wire.
    async fn run_prepared<T>(&self, stmt: &Prepared<T>) -> Result<protocol::Frame> {
        if stmt.client_id != self.client_id {
            return Err(Error::ForeignStatement);
        }
        self.round_trip(
            protocol::REQ_EXECUTE,
            &protocol::encode_execute_payload(stmt.stmt_id),
        )
        .await
    }

    async fn round_trip(&self, kind: u8, payload: &[u8]) -> Result<protocol::Frame> {
        let mut conn = self.conn.lock().await;
        conn.round_trip(kind, payload, self.read_timeout, self.write_timeout)
            .await
    }
}

/// A live change-event subscription on a dedicated connection, as an async
/// [`Stream`] of `Result<Notification>`.
///
/// Poll it with `tokio_stream::StreamExt::next` (or any `futures` `StreamExt`), or
/// call [`next_event`](Self::next_event) directly. The first stream error or
/// end-of-stream surfaces once as `Some(Err(..))`, after which the stream yields
/// `None`; a malformed *event payload* surfaces as an error but does NOT end the
/// stream (the framing is intact), mirroring the sync [`Subscription`].
///
/// To cancel, stop polling and drop it: dropping the `AsyncSubscription` closes
/// its connection, which the server treats as an implicit unsubscribe. The read
/// is cancel-safe (it goes through the wire crate's [`FrameReader`]), so dropping
/// a `next()` future mid-frame is safe. For an explicit, acknowledged teardown,
/// [`unsubscribe`](Self::unsubscribe).
///
/// [`Subscription`]: crate::Subscription
#[derive(Debug)]
pub struct AsyncSubscription {
    stream: TcpStream,
    /// Cancel-safe inbound reader; retains partially-read bytes across a dropped
    /// poll, so racing this stream in a `select!` never desyncs the framing.
    reader: FrameReader,
    /// The subscription handle (the `SUBSCRIBE` frame's `req_id`); the server tags
    /// every pushed frame with it and it is the `UNSUBSCRIBE` target.
    handle: u32,
    /// Reused for the `UNSUBSCRIBE` write (the event read deliberately has no
    /// deadline — see [`AsyncClient::subscribe`]).
    write_timeout: Option<Duration>,
    /// The filter's `exclude_origin`, applied CLIENT-SIDE (the wire subscribe
    /// filter can't carry it): `Change` events whose `origin` matches are dropped
    /// by both [`next_event`](Self::next_event) and the [`Stream`] impl, so
    /// [`SubscriptionFilter::excluding_origin`] works over the network too.
    exclude_origin: Option<u64>,
    /// Latches once the stream ends (EOF or a terminal frame) so it stops after
    /// surfacing the cause exactly once.
    ended: bool,
}

impl AsyncSubscription {
    /// Dial a fresh connection, run the `SUBSCRIBE → RESP_SUBSCRIBED` handshake
    /// (the ack precedes any event, as in the sync path), and return the live
    /// subscription. (Crate-internal; reached via [`AsyncClient::subscribe`].)
    pub(crate) async fn open(
        addr: SocketAddr,
        connect_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
        filter: &SubscriptionFilter,
    ) -> Result<Self> {
        let mut stream = dial(addr, connect_timeout).await?;

        // One subscription per connection, so a fixed handle of 1 is sufficient.
        const HANDLE: u32 = 1;
        let payload = protocol::encode_subscribe_filter(filter);
        let mut buf = Vec::with_capacity(9 + payload.len());
        with_timeout(
            write_timeout,
            protocol::write_frame_buffered(&mut stream, &mut buf, HANDLE, protocol::REQ_SUBSCRIBE, |b| {
                b.extend_from_slice(&payload)
            }),
        )
        .await
        .map_err(Error::Io)?;

        // Read the ack through the FrameReader from the start, so any bytes read
        // ahead of the ack stay buffered for the event stream.
        let mut reader = FrameReader::new();
        let frame = reader.next_frame(&mut stream).await.map_err(Error::Io)?;
        match frame.kind {
            protocol::RESP_SUBSCRIBED => Ok(AsyncSubscription {
                stream,
                reader,
                handle: HANDLE,
                write_timeout,
                exclude_origin: filter.exclude_origin,
                ended: false,
            }),
            // An old server replies ERROR to an unknown SUBSCRIBE kind ⇒
            // "subscriptions unsupported"; cap/validation failures land here too.
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Await the next [`Notification`]. Cancel-safe (the only await is the
    /// cancel-safe [`FrameReader`] read). Returns [`Error::Io`] on EOF / socket
    /// error (latching the subscription ended); a malformed event payload errors
    /// but leaves the subscription live. After it has ended, returns
    /// [`Error::Closed`].
    pub async fn next_event(&mut self) -> Result<Notification> {
        if self.ended {
            return Err(Error::Closed);
        }
        loop {
            let frame = match self.reader.next_frame(&mut self.stream).await {
                Ok(f) => f,
                Err(e) => {
                    self.ended = true;
                    return Err(Error::Io(e));
                }
            };
            let notification = self.classify(frame)?;
            // Client-side loop-breaker: skip our OWN-origin events and await the
            // next (the wire subscribe filter can't carry exclude_origin).
            if self.is_excluded(&notification) {
                continue;
            }
            return Ok(notification);
        }
    }

    /// Whether a notification is an own-origin `Change` to drop under the filter's
    /// client-side [`exclude_origin`](AsyncSubscription::exclude_origin).
    fn is_excluded(&self, n: &Notification) -> bool {
        match self.exclude_origin {
            Some(ex) => matches!(n, Notification::Change(c) if c.origin == Some(ex)),
            None => false,
        }
    }

    /// Map a pushed frame to a [`Notification`] (or an error). Sets `ended` for a
    /// terminal frame; a malformed *event payload* (framing intact) errors but
    /// leaves the subscription live. Shared by [`next_event`](Self::next_event)
    /// and the [`Stream`] impl.
    fn classify(&mut self, frame: protocol::Frame) -> Result<Notification> {
        match frame.kind {
            protocol::RESP_EVENT => {
                let wire = protocol::decode_event_payload(&frame.payload).map_err(Error::Io)?;
                Ok(Notification::Change(ChangeNotification::from_wire(wire)?))
            }
            protocol::RESP_SUBLAGGED => Ok(Notification::Lagged),
            // Neither is part of the post-subscribe push protocol; treat as terminal.
            protocol::RESP_ERROR => {
                self.ended = true;
                Err(server_error(&frame.payload))
            }
            other => {
                self.ended = true;
                Err(Error::UnexpectedResponse(other))
            }
        }
    }

    /// Cleanly close the subscription: send `UNSUBSCRIBE`, drain any events the
    /// server had already pushed ahead of the ack, and await the `DONE`
    /// acknowledgement. (Dropping the subscription instead just closes the
    /// connection, which the server also treats as an unsubscribe — without the
    /// explicit ack.)
    pub async fn unsubscribe(mut self) -> Result<()> {
        let payload = protocol::encode_unsubscribe_payload(self.handle);
        let mut buf = Vec::with_capacity(9 + payload.len());
        with_timeout(
            self.write_timeout,
            protocol::write_frame_buffered(
                &mut self.stream,
                &mut buf,
                self.handle,
                protocol::REQ_UNSUBSCRIBE,
                |b| b.extend_from_slice(&payload),
            ),
        )
        .await
        .map_err(Error::Io)?;
        // Events queued before the server processed the UNSUBSCRIBE arrive ahead
        // of the DONE ack — discard them until the ack (or an error).
        loop {
            let frame = self.reader.next_frame(&mut self.stream).await.map_err(Error::Io)?;
            match frame.kind {
                protocol::RESP_DONE => return Ok(()),
                protocol::RESP_EVENT | protocol::RESP_SUBLAGGED => continue,
                protocol::RESP_ERROR => return Err(server_error(&frame.payload)),
                other => return Err(Error::UnexpectedResponse(other)),
            }
        }
    }
}

impl Stream for AsyncSubscription {
    type Item = Result<Notification>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `AsyncSubscription` is `Unpin` (TcpStream + FrameReader are), so we can
        // take a plain `&mut`.
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        loop {
            match this.reader.poll_next_frame(cx, &mut this.stream) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(frame)) => {
                    let n = this.classify(frame);
                    // Skip our OWN-origin events (client-side exclude_origin) and
                    // poll the next frame; a buffered frame yields Ready again.
                    if let Ok(ref notif) = n
                        && this.is_excluded(notif)
                    {
                        continue;
                    }
                    return Poll::Ready(Some(n));
                }
                Poll::Ready(Err(e)) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(Error::Io(e))));
                }
            }
        }
    }
}
