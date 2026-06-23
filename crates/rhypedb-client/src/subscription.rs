//! Change-event subscriptions over a dedicated connection.
//!
//! A subscription runs on its **own** TCP connection, separate from the
//! [`Client`](crate::Client)'s request/response connection: the server turns a
//! connection that issues `SUBSCRIBE` into a *subscription connection* that
//! rejects queries, so pushed events can never race a query reply. Open one with
//! [`Client::subscribe`](crate::Client::subscribe); it returns a [`Subscription`],
//! a **blocking iterator** of [`Notification`]s.
//!
//! Events are NOTIFICATIONS, not authoritative state: `fields` are best-effort
//! (the `/query` read form — `Bytes` as base64, `DateTime` as RFC 3339, `Json`
//! inline) and a `Delete` carries the deleted object's last-known scalars (only a
//! cascade-deleted edge-only row omits them). A [`Notification::Lagged`] means the
//! server's bounded buffer overflowed (or it is shutting down) and ≥1 event was
//! dropped — reconcile by re-querying current state with the [`Client`](crate::Client).

use std::collections::HashMap;
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::time::Duration;

use rhypedb_wire::protocol::{self, sync as wire_sync, WireEvent};
use rhypedb_subscribe::{ChangeKind, SubscriptionFilter};

use crate::{server_error, Error, Result};

/// One message delivered over a [`Subscription`].
#[derive(Debug, Clone, PartialEq)]
pub enum Notification {
    /// A change event (create / update / delete).
    Change(ChangeNotification),
    /// The server's bounded per-subscription buffer overflowed and at least one
    /// event was dropped (or the server is shutting down). The subscription is
    /// still live; reconcile by re-querying current state.
    Lagged,
}

/// A single change event. Scalar `fields` are best-effort (see the module docs);
/// re-query via the [`Client`](crate::Client) for authoritative typed state.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeNotification {
    /// Whether the object was created, updated, or deleted.
    pub kind: ChangeKind,
    /// The object's type name.
    pub type_name: String,
    /// The object id (decoded from the wire's lossless decimal-string form).
    pub id: u64,
    /// The commit version (decoded from the wire's lossless decimal-string form).
    pub version: u64,
    /// Best-effort scalar fields in the `/query` read form, or `None` for a
    /// cascade-deleted edge-only join row. Large 64-bit values may have lost
    /// precision in transit — re-query for exact values.
    pub fields: Option<HashMap<String, serde_json::Value>>,
}

impl ChangeNotification {
    fn from_wire(w: WireEvent) -> Result<Self> {
        // Reject an unknown event-envelope version loudly rather than risk
        // misinterpreting a future, differently-shaped format (the `v` tag exists
        // precisely so a consumer can detect the format — see protocol docs).
        if w.v != protocol::EVENT_FORMAT_TAG {
            return Err(Error::Deserialize(format!(
                "unsupported event format tag {:?} (expected {:?}) — upgrade the client",
                w.v,
                protocol::EVENT_FORMAT_TAG
            )));
        }
        let kind = match w.kind.as_str() {
            "create" => ChangeKind::Create,
            "update" => ChangeKind::Update,
            "delete" => ChangeKind::Delete,
            other => return Err(Error::Deserialize(format!("unknown event kind {other:?}"))),
        };
        let id = w
            .id
            .parse::<u64>()
            .map_err(|_| Error::Deserialize(format!("invalid event id {:?}", w.id)))?;
        let version = w
            .version
            .parse::<u64>()
            .map_err(|_| Error::Deserialize(format!("invalid event version {:?}", w.version)))?;
        Ok(ChangeNotification {
            kind,
            type_name: w.type_name,
            id,
            version,
            fields: w.fields,
        })
    }
}

/// A live change-event subscription on a dedicated connection.
///
/// Iterate it (it is an [`Iterator`] of `Result<Notification>`) or call
/// [`next_event`](Self::next_event) directly; both **block** until the next event
/// arrives. The first stream error or end-of-stream surfaces once as a
/// `Some(Err(..))` (then the iterator yields `None`); a malformed *event payload*
/// surfaces as an error but does NOT end the subscription (the framing is intact).
///
/// `Subscription` is `Send`, so move it onto a worker thread. To stop a
/// subscription whose thread is blocked in [`next_event`](Self::next_event),
/// obtain a [`SubscriptionStopper`] first (it can close the connection from
/// another thread).
///
/// Dropping a `Subscription` closes its connection, which the server treats as an
/// implicit unsubscribe — **with one caveat**: a [`SubscriptionStopper`] holds a
/// duplicated socket handle, so if one is still alive the OS keeps the connection
/// open until it too is dropped. For deterministic teardown while a stopper
/// exists, call [`SubscriptionStopper::stop`] (it actively shuts the connection
/// down) or [`unsubscribe`](Self::unsubscribe).
#[derive(Debug)]
pub struct Subscription {
    stream: TcpStream,
    /// The subscription handle (the `SUBSCRIBE` frame's `req_id`); the server
    /// tags every pushed frame with it and it is the `UNSUBSCRIBE` target.
    handle: u32,
    /// Latches once the stream ends (EOF or a terminal error) so the iterator
    /// stops after surfacing the cause exactly once.
    ended: bool,
}

impl Subscription {
    /// Dial a fresh connection, run the `SUBSCRIBE` handshake, and return the
    /// live subscription. (Crate-internal; reached via
    /// [`Client::subscribe`](crate::Client::subscribe).)
    ///
    /// `connect_timeout` bounds establishing the connection (like
    /// [`Client::connect`](crate::Client::connect)). The configured `read_timeout`
    /// is deliberately NOT applied: a subscription blocks waiting for sparse
    /// server pushes, so a finite per-read deadline would tear it down between
    /// events. Cancel a blocked read with a [`SubscriptionStopper`] instead.
    pub(crate) fn open(
        addr: SocketAddr,
        connect_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
        filter: &SubscriptionFilter,
    ) -> Result<Self> {
        let mut stream = match connect_timeout {
            Some(t) => TcpStream::connect_timeout(&addr, t).map_err(Error::Connect)?,
            None => TcpStream::connect(addr).map_err(Error::Connect)?,
        };
        stream.set_nodelay(true).map_err(Error::Connect)?;
        // The write timeout guards the SUBSCRIBE/UNSUBSCRIBE sends; the READ side
        // stays blocking with no timeout — a subscription waits for events, and a
        // blocked read is cancelled via a SubscriptionStopper, not a deadline.
        stream
            .set_write_timeout(write_timeout)
            .map_err(Error::Connect)?;

        // The subscribe frame's req_id is the subscription HANDLE. One
        // subscription per connection, so a fixed handle of 1 is sufficient.
        const HANDLE: u32 = 1;
        let payload = protocol::encode_subscribe_filter(filter);
        wire_sync::write_frame(&mut stream, HANDLE, protocol::REQ_SUBSCRIBE, &payload)
            .map_err(Error::Io)?;

        // The ack is guaranteed to precede any event: the server registers the
        // sink and writes RESP_SUBSCRIBED within one inbound-frame handler, before
        // it ever drains the event channel.
        let frame = wire_sync::read_frame(&mut stream).map_err(Error::Io)?;
        match frame.kind {
            protocol::RESP_SUBSCRIBED => Ok(Subscription {
                stream,
                handle: HANDLE,
                ended: false,
            }),
            // An old server replies ERROR to an unknown SUBSCRIBE kind ⇒
            // "subscriptions unsupported"; cap/validation failures land here too.
            protocol::RESP_ERROR => Err(server_error(&frame.payload)),
            other => Err(Error::UnexpectedResponse(other)),
        }
    }

    /// Block until the next [`Notification`] arrives.
    ///
    /// This blocks **indefinitely** waiting for a server push (no read deadline —
    /// see [`open`](Self::open)); to interrupt a blocked call from another thread,
    /// use a [`SubscriptionStopper`]. Returns [`Error::Io`] on end-of-stream or a
    /// socket error (and latches the subscription ended); a malformed event
    /// payload also returns an error but leaves the subscription live (framing
    /// intact). After the subscription has ended, returns [`Error::Closed`].
    pub fn next_event(&mut self) -> Result<Notification> {
        if self.ended {
            return Err(Error::Closed);
        }
        let frame = match wire_sync::read_frame(&mut self.stream) {
            Ok(f) => f,
            Err(e) => {
                self.ended = true;
                return Err(Error::Io(e));
            }
        };
        match frame.kind {
            protocol::RESP_EVENT => {
                // Payload-decode failure (framing intact) → surface but stay live.
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

    /// Get a handle that can stop this subscription from another thread by
    /// closing its connection — unblocking a thread parked in
    /// [`next_event`](Self::next_event) (a closed read ends the iterator) and
    /// prompting the server to unregister the subscription.
    pub fn stopper(&self) -> Result<SubscriptionStopper> {
        Ok(SubscriptionStopper {
            stream: self.stream.try_clone().map_err(Error::Io)?,
        })
    }

    /// Cleanly close the subscription: send `UNSUBSCRIBE`, drain any events the
    /// server had already pushed ahead of the ack, and wait for the `DONE`
    /// acknowledgement. (Dropping the `Subscription` instead just closes the
    /// connection, which the server also treats as an unsubscribe — but without
    /// the explicit ack.)
    ///
    /// Like [`next_event`](Self::next_event), the drain reads block until the
    /// server responds or closes the connection; on a hung server it blocks until
    /// a [`SubscriptionStopper`] (obtained beforehand) closes the socket.
    pub fn unsubscribe(mut self) -> Result<()> {
        wire_sync::write_frame(
            &mut self.stream,
            self.handle,
            protocol::REQ_UNSUBSCRIBE,
            &protocol::encode_unsubscribe_payload(self.handle),
        )
        .map_err(Error::Io)?;
        // Events queued before the server processed the UNSUBSCRIBE arrive ahead
        // of the DONE ack — discard them until the ack (or an error).
        loop {
            let frame = wire_sync::read_frame(&mut self.stream).map_err(Error::Io)?;
            match frame.kind {
                protocol::RESP_DONE => return Ok(()),
                protocol::RESP_EVENT | protocol::RESP_SUBLAGGED => continue,
                protocol::RESP_ERROR => return Err(server_error(&frame.payload)),
                other => return Err(Error::UnexpectedResponse(other)),
            }
        }
    }
}

impl Iterator for Subscription {
    type Item = Result<Notification>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        // A terminal result sets `ended`, so the NEXT call returns None; a
        // non-terminal error (bad event payload) leaves `ended` false and the
        // iterator keeps going.
        Some(self.next_event())
    }
}

/// A handle to stop a [`Subscription`] from another thread (see
/// [`Subscription::stopper`]). Closing it shuts down the subscription's
/// connection, so a blocked [`next_event`](Subscription::next_event) returns and
/// the server unregisters the subscription.
#[derive(Debug)]
pub struct SubscriptionStopper {
    stream: TcpStream,
}

impl SubscriptionStopper {
    /// Stop the subscription by closing its connection. Idempotent enough for a
    /// shutdown path: a second call (or a call after the peer already closed)
    /// returns an [`Error::Io`].
    pub fn stop(&self) -> Result<()> {
        self.stream.shutdown(Shutdown::Both).map_err(Error::Io)
    }
}
