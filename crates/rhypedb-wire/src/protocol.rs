#![allow(dead_code)]
//! Binary TCP wire protocol for rhypedb.
//!
//! # Frame format
//!
//! ```text
//! [len: u32 BE]       bytes that follow (req_id + type + payload)
//! [req_id: u32 BE]    request ID (echoed in response)
//! [type: u8]          message type
//! [payload: len-5 bytes]
//! ```
//!
//! # Client request types
//!
//! | byte | name        | payload                                  |
//! |------|-------------|------------------------------------------|
//! | 0x01 | Query       | `[q_len: u32 BE][utf8 query]`            |
//! | 0x02 | Ping        | empty                                    |
//! | 0x03 | VectorBatch | see `decode_vector_batch_payload`        |
//! | 0x04 | Prepare     | `[q_len: u32 BE][utf8 query]` (same as Query) |
//! | 0x05 | Execute     | `[stmt_id: u64 BE]`                      |
//! | 0x06 | Subscribe   | filter (see `decode_subscribe_filter`)   |
//! | 0x07 | Unsubscribe | `[sub_req_id: u32 BE]`                    |
//!
//! A `Prepare` parses + caches the query for the LIFETIME OF THE CONNECTION and
//! returns a `Prepared(stmt_id)`; subsequent `Execute(stmt_id)` re-run it with no
//! re-parse and no re-sent query string. Statement IDs are per-connection (like a
//! Postgres session) and are dropped when the connection closes.
//!
//! # Server response types
//!
//! | byte | name      | payload                                            |
//! |------|-----------|----------------------------------------------------|
//! | 0x80 | Objects   | `[count: u32 BE]` then `count` encoded objects     |
//! | 0x81 | Single    | one encoded object                                 |
//! | 0x82 | Done      | empty                                              |
//! | 0x83 | Error     | `[msg_len: u32 BE][utf8 msg]`                      |
//! | 0x84 | Pong      | empty                                              |
//! | 0x85 | Prepared  | `[stmt_id: u64 BE]`                                |
//! | 0x86 | Subscribed| empty (reply to Subscribe; ack by `req_id`)        |
//! | 0x87 | Event     | UTF-8 JSON of a `WireEvent` (server-PUSHED)        |
//! | 0x88 | SubLagged | empty (server-PUSHED; see Subscriptions)           |
//!
//! # Subscriptions (server-initiated pushes)
//!
//! A `Subscribe(req_id=R, filter)` registers a change-event stream on this
//! connection and is acked with `Subscribed` echoing `R`. Thereafter the server
//! PUSHES `Event` / `SubLagged` frames tagged with `req_id=R` — these are
//! **unsolicited**: they may arrive at any time, interleaved with request acks.
//!
//! Therefore a client on a subscribed connection MUST demultiplex frames by
//! **kind first**, never assume "the next frame answers my last request", and
//! route `Event`/`SubLagged` (0x87/0x88) by `req_id` to the originating
//! subscription. To keep the simple synchronous clients safe, a connection that
//! has issued a `Subscribe` becomes a **subscription connection**: it serves
//! only `Subscribe`/`Unsubscribe`/`Ping`, and rejects `Query`/`Prepare`/
//! `Execute`/`VectorBatch` with an `Error`. Run queries on a separate connection.
//! A connection that never subscribes NEVER receives a push, so existing
//! request/response clients are unaffected. An old server replies `Error` to a
//! `Subscribe` (unknown kind) — a new client treats that as "subscriptions
//! unsupported".
//!
//! `Unsubscribe(target=R)` removes the subscription whose handle is `R`; it is
//! acked with `Done`. `SubLagged(R)` means the bounded per-connection buffer
//! overflowed and ≥1 event for `R` was dropped — the client must reconcile by
//! re-querying. The server also pushes a best-effort `SubLagged(R)` to each live
//! subscription on graceful shutdown, so a backend reconciles across a deploy.
//!
//! `Event` payload is a format-tagged JSON `WireEvent` (`object_id`/`version` as
//! decimal strings so 64-bit ids survive a JS `JSON.parse`). Its `fields` are
//! **best-effort / a notification**: large 64-bit scalar field values may lose
//! precision in a JS `JSON.parse` — re-query for authoritative state. `Create`,
//! `Update`, AND `Delete` events all carry the object's scalar fields (a `Delete`
//! reports the deleted object's last-known values, so a subscriber learns *which*
//! object — by slug/identifier — went away). The one exception: a cascade-deleted
//! pure edge-only join row (a type with no scalar fields) carries no `fields`.
//! (`Bytes` fields arrive as base64, `DateTime` as RFC 3339, `Json` inline,
//! matching the `/query` read form.)
//! # Object encoding
//!
//! ```text
//! [type_name_len: u16 BE][type_name: utf8]
//! [id: u64 BE]
//! [fields: same format as engine::serialize_fields]
//! ```

use std::collections::HashMap;
use std::io;

use serde::{Deserialize, Serialize};
#[cfg(feature = "tokio")]
use std::pin::Pin;
#[cfg(feature = "tokio")]
use std::task::{Context, Poll};
#[cfg(feature = "tokio")]
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, ReadBuf};

use crate::object::{serialize_fields_into, try_deserialize_fields, Object};
use rhypedb_subscribe::{ChangeEvent, ChangeKind, SubscriptionFilter};

/// Maximum payload size per frame (16 MB). Defensive limit to prevent
/// runaway allocations from malformed clients.
pub const MAX_FRAME_PAYLOAD: usize = 16 * 1024 * 1024;

// Request types
pub const REQ_QUERY: u8 = 0x01;
pub const REQ_PING: u8 = 0x02;
/// Bulk ingest of caller-supplied (precomputed) vectors for one Vector field.
pub const REQ_VECTOR_BATCH: u8 = 0x03;
/// Parse + cache a query for the connection's lifetime; replies `Prepared(id)`.
pub const REQ_PREPARE: u8 = 0x04;
/// Run a previously-prepared statement by id; replies like `Query`.
pub const REQ_EXECUTE: u8 = 0x05;
/// Open a change-event subscription on this connection (payload = filter).
/// The frame's `req_id` becomes the subscription handle; replies `Subscribed`.
pub const REQ_SUBSCRIBE: u8 = 0x06;
/// Close a subscription (payload = `[sub_req_id: u32 BE]`); replies `Done`.
pub const REQ_UNSUBSCRIBE: u8 = 0x07;

// Response types
pub const RESP_OBJECTS: u8 = 0x80;
pub const RESP_SINGLE: u8 = 0x81;
pub const RESP_DONE: u8 = 0x82;
pub const RESP_ERROR: u8 = 0x83;
pub const RESP_PONG: u8 = 0x84;
/// Reply to `Prepare`: the assigned per-connection statement id.
pub const RESP_PREPARED: u8 = 0x85;
/// Reply to `Subscribe`: ack (empty payload; correlated by the subscribe req_id).
pub const RESP_SUBSCRIBED: u8 = 0x86;
/// Server-PUSHED change event: `req_id` = the subscription handle; payload =
/// UTF-8 JSON of a [`WireEvent`].
pub const RESP_EVENT: u8 = 0x87;
/// Server-PUSHED lag notice: `req_id` = the subscription handle; empty payload.
/// At least one event for that subscription was dropped — reconcile by re-query.
pub const RESP_SUBLAGGED: u8 = 0x88;

/// Format tag stamped into every [`WireEvent`] so a consumer can detect the
/// envelope version (mirrors the logical-export `FORMAT_TAG` convention).
pub const EVENT_FORMAT_TAG: &str = "rhypedb-event-v1";

/// A parsed inbound frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub req_id: u32,
    pub kind: u8,
    pub payload: Vec<u8>,
}

/// Read one frame from an async reader. Returns Err on connection close or
/// malformed framing.
#[cfg(feature = "tokio")]
pub async fn read_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<Frame> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len < 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame too small: {len} bytes"),
        ));
    }
    if len > MAX_FRAME_PAYLOAD + 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame too large: {len} bytes"),
        ));
    }

    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).await?;

    let req_id = u32::from_be_bytes(body[0..4].try_into().unwrap());
    let kind = body[4];
    let payload = body[5..].to_vec();

    Ok(Frame {
        req_id,
        kind,
        payload,
    })
}

/// A **cancellation-safe** inbound frame reader.
///
/// [`read_frame`] is built on `read_exact`, which is NOT cancel-safe: if its
/// future is dropped after it has copied some bytes out of the underlying reader
/// but before completing, those bytes are lost and the stream desyncs. That is
/// fine when the only competing `select!` arm closes the connection, but the
/// subscription connection loop also races server pushes and lag wake-ups that
/// `continue` the loop — so a partially-read inbound control frame must survive a
/// cancelled read.
///
/// `FrameReader` owns a reusable accumulation buffer and parses whole frames out
/// of it. Each await is a single [`AsyncReadExt::read_buf`] (which IS cancel-safe
/// — a dropped poll leaves the buffer untouched), and the bytes read so far live
/// in the task-owned buffer, so dropping a [`FrameReader::next_frame`] future
/// mid-frame loses nothing; the next call resumes. It also tolerates a read that
/// returns more than one frame's worth of bytes (TCP coalescing / read-ahead).
#[cfg(feature = "tokio")]
#[derive(Debug, Default)]
pub struct FrameReader {
    buf: Vec<u8>,
    /// Parse cursor into `buf`; bytes before it are already-returned frames.
    start: usize,
}

#[cfg(feature = "tokio")]
impl FrameReader {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(8 * 1024),
            start: 0,
        }
    }

    /// Try to parse one complete frame out of the already-buffered bytes, without
    /// any I/O. Returns `Ok(None)` when more bytes are needed; `Err` on a framing
    /// violation (length out of bounds). Advances the parse cursor on success and
    /// resets the buffer once fully drained. Shared by the async [`next_frame`]
    /// and the poll-based [`poll_next_frame`] so the bounds-checked frame parser
    /// is single-sourced.
    fn try_parse(&mut self) -> io::Result<Option<Frame>> {
        let avail = self.buf.len() - self.start;
        if avail < 4 {
            return Ok(None);
        }
        let off = self.start;
        let len = u32::from_be_bytes(self.buf[off..off + 4].try_into().unwrap()) as usize;
        if len < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("frame too small: {len} bytes"),
            ));
        }
        if len > MAX_FRAME_PAYLOAD + 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("frame too large: {len} bytes"),
            ));
        }
        if avail < 4 + len {
            return Ok(None);
        }
        let body = off + 4;
        let req_id = u32::from_be_bytes(self.buf[body..body + 4].try_into().unwrap());
        let kind = self.buf[body + 4];
        let payload = self.buf[body + 5..off + 4 + len].to_vec();
        self.start += 4 + len;
        // Fully drained → reset to keep the buffer from growing.
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        }
        Ok(Some(Frame {
            req_id,
            kind,
            payload,
        }))
    }

    /// Compact the already-consumed prefix so the buffer stays bounded by at most
    /// one in-flight frame (+ any read-ahead) before reading more bytes.
    fn compact(&mut self) {
        if self.start > 0 {
            self.buf.drain(0..self.start);
            self.start = 0;
        }
    }

    /// Read the next complete frame. Cancel-safe across `select!` iterations.
    pub async fn next_frame<R: AsyncReadExt + Unpin>(&mut self, reader: &mut R) -> io::Result<Frame> {
        loop {
            if let Some(frame) = self.try_parse()? {
                return Ok(frame);
            }
            self.compact();
            // Cancel-safe append: a dropped future leaves `buf` untouched.
            let n = reader.read_buf(&mut self.buf).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed mid-frame",
                ));
            }
        }
    }

    /// Poll-based counterpart to [`next_frame`], for hand-rolled [`Stream`] /
    /// `poll_*` implementations (e.g. the async client's subscription stream).
    ///
    /// Cancel-safe by construction: buffered bytes live in `self`, and each read
    /// goes through a stack scratch buffer that is only appended to `self.buf`
    /// once `poll_read` reports `Ready(Ok)`, so a `Pending`/dropped poll loses
    /// nothing. EOF mid-frame surfaces as [`io::ErrorKind::UnexpectedEof`].
    ///
    /// [`Stream`]: https://docs.rs/futures-core/latest/futures_core/stream/trait.Stream.html
    pub fn poll_next_frame<R: AsyncRead + Unpin>(
        &mut self,
        cx: &mut Context<'_>,
        reader: &mut R,
    ) -> Poll<io::Result<Frame>> {
        loop {
            match self.try_parse() {
                Ok(Some(frame)) => return Poll::Ready(Ok(frame)),
                Ok(None) => {}
                Err(e) => return Poll::Ready(Err(e)),
            }
            self.compact();
            // Read into a stack scratch buffer; only commit to `self.buf` once the
            // read actually yields bytes, keeping the poll cancel-safe.
            let mut scratch = [0u8; 16 * 1024];
            let mut read_buf = ReadBuf::new(&mut scratch);
            match Pin::new(&mut *reader).poll_read(cx, &mut read_buf) {
                Poll::Ready(Ok(())) => {
                    let filled = read_buf.filled();
                    if filled.is_empty() {
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "connection closed mid-frame",
                        )));
                    }
                    self.buf.extend_from_slice(filled);
                }
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Write one frame to an async writer. Issues 4 successive `write_all` calls
/// which the underlying BufWriter coalesces before flushing.
///
/// Hot callers should prefer `write_frame_buffered` so the entire frame —
/// header + payload — is built in one contiguous buffer and shipped with a
/// single `write_all`, sidestepping the BufWriter's intermediate state
/// machine.
#[cfg(feature = "tokio")]
pub async fn write_frame<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    req_id: u32,
    kind: u8,
    payload: &[u8],
) -> io::Result<()> {
    let len = (5 + payload.len()) as u32;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(&req_id.to_be_bytes()).await?;
    writer.write_all(&[kind]).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Build a complete frame (header + payload) in the provided buffer and ship
/// it with one `write_all` + `flush`. The buffer is reused across responses
/// — caller is responsible for clearing it before the next call. Avoids
/// per-response allocation AND turns the 4-write-all-then-flush sequence
/// into a single syscall path through the BufWriter.
///
/// `payload_builder` writes the response payload (the bytes after the 5-byte
/// header) into the buffer. The header is prepended by this function.
#[cfg(feature = "tokio")]
pub async fn write_frame_buffered<W, F>(
    writer: &mut W,
    buf: &mut Vec<u8>,
    req_id: u32,
    kind: u8,
    payload_builder: F,
) -> io::Result<()>
where
    W: AsyncWriteExt + Unpin,
    F: FnOnce(&mut Vec<u8>),
{
    buf.clear();
    // Frame layout on the wire: [len:4][req_id:4][kind:1][payload].
    // We reserve the 9-byte header up front; once the payload is built we
    // backfill `len` (= 5 + payload_len, matching the existing protocol
    // where the length excludes itself).
    buf.extend_from_slice(&[0u8; 9]);
    payload_builder(buf);

    let payload_len = buf.len() - 9;
    let total_len = (5 + payload_len) as u32;
    buf[0..4].copy_from_slice(&total_len.to_be_bytes());
    buf[4..8].copy_from_slice(&req_id.to_be_bytes());
    buf[8] = kind;

    writer.write_all(buf).await?;
    writer.flush().await?;
    Ok(())
}

/// Blocking frame I/O over `std::io::Read`/`Write`, for the synchronous client.
///
/// Mirrors the async [`read_frame`]/[`write_frame`] byte-for-byte (same
/// `[len:u32][req_id:u32][kind:u8][payload]` layout, same length bounds), but
/// requires no async runtime — a sync client links zero of tokio. Cancellation
/// safety is irrelevant here (a blocking call is never raced by a `select!`), so
/// no sync `FrameReader` is needed.
pub mod sync {
    use super::{Frame, MAX_FRAME_PAYLOAD};
    use std::io::{self, Read, Write};

    /// Read one frame from a blocking reader. Errs on EOF or malformed framing.
    pub fn read_frame<R: Read>(reader: &mut R) -> io::Result<Frame> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("frame too small: {len} bytes"),
            ));
        }
        if len > MAX_FRAME_PAYLOAD + 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("frame too large: {len} bytes"),
            ));
        }

        let mut body = vec![0u8; len];
        reader.read_exact(&mut body)?;
        let req_id = u32::from_be_bytes(body[0..4].try_into().unwrap());
        let kind = body[4];
        let payload = body[5..].to_vec();
        Ok(Frame {
            req_id,
            kind,
            payload,
        })
    }

    /// Write one frame to a blocking writer as a single contiguous buffer, then
    /// flush. (Caller should wrap the socket in a `BufWriter` if batching.)
    pub fn write_frame<W: Write>(
        writer: &mut W,
        req_id: u32,
        kind: u8,
        payload: &[u8],
    ) -> io::Result<()> {
        let total_len = (5 + payload.len()) as u32;
        let mut buf = Vec::with_capacity(4 + 5 + payload.len());
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(&req_id.to_be_bytes());
        buf.push(kind);
        buf.extend_from_slice(payload);
        writer.write_all(&buf)?;
        writer.flush()
    }
}

/// Encode a single Object to bytes (for inclusion in Objects / Single responses).
/// Writes directly into `out` — no intermediate Bytes allocation. If the
/// Object carries `raw_fields` (came straight from an LSM read via
/// `Object::from_raw`), the wire emits those bytes verbatim and skips the
/// `serialize_fields_into` pass — the on-disk format and the wire FieldMap
/// format are identical, so no transformation is needed.
pub fn encode_object(obj: &Object, out: &mut Vec<u8>) {
    let type_bytes = obj.type_name.as_bytes();
    out.extend_from_slice(&(type_bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(type_bytes);
    out.extend_from_slice(&obj.id.to_be_bytes());
    if let Some(raw) = &obj.raw_fields {
        out.extend_from_slice(raw);
    } else {
        serialize_fields_into(&obj.fields, out);
    }
}

/// Decode a single Object starting at `pos`. Returns the decoded object and
/// the new position.
pub fn decode_object(data: &[u8], mut pos: usize) -> io::Result<(Object, usize)> {
    if pos + 2 > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "object header truncated"));
    }
    let type_len = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    if pos + type_len > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "object type_name truncated",
        ));
    }
    let type_name = std::str::from_utf8(&data[pos..pos + type_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("type_name utf8: {e}")))?
        .to_string();
    pos += type_len;

    if pos + 8 > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "object id truncated"));
    }
    let id = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // The fields encoding is self-describing in length: count, then each field
    // with its own length-prefixed name and tagged value. `try_deserialize_fields`
    // walks it fallibly (bounds- + UTF-8-checked) — so malformed wire bytes
    // surface as `InvalidData`, never a panic — and returns the end offset.
    let (fields, end) = try_deserialize_fields(data, pos)?;
    pos = end;

    Ok((
        Object {
            type_name,
            id,
            fields,
            raw_fields: None,
        },
        pos,
    ))
}

/// Encode an Objects response payload into `out`. Caller can pre-size the
/// buffer or reuse it across responses to avoid per-query allocation.
pub fn encode_objects_payload_into(objects: &[Object], out: &mut Vec<u8>) {
    // Heuristic pre-size: ~64 bytes per object (typical small objects with
    // a few small fields). Saves the first 2-3 Vec::reserve doublings on
    // the hot 50-object response shape.
    out.reserve(4 + objects.len() * 64);
    out.extend_from_slice(&(objects.len() as u32).to_be_bytes());
    for obj in objects {
        encode_object(obj, out);
    }
}

/// Convenience wrapper. Hot callers should pre-allocate and call
/// `encode_objects_payload_into`.
pub fn encode_objects_payload(objects: &[Object]) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_objects_payload_into(objects, &mut buf);
    buf
}

/// Decode an Objects response payload.
pub fn decode_objects_payload(data: &[u8]) -> io::Result<Vec<Object>> {
    if data.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "objects count missing"));
    }
    let count = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    // Don't pre-allocate to a hostile `count` (it's an unbounded u32, while the
    // payload is ≤ 16 MiB): cap the reservation and let the Vec grow. Each
    // decode_object validates, so a bogus count fails fast when bytes run out.
    let mut objects = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let (obj, new_pos) = decode_object(data, pos)?;
        objects.push(obj);
        pos = new_pos;
    }
    Ok(objects)
}

/// Encode a Single response payload.
pub fn encode_single_payload(object: &Object) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_object(object, &mut buf);
    buf
}

/// Decode a Single response payload.
pub fn decode_single_payload(data: &[u8]) -> io::Result<Object> {
    let (obj, _) = decode_object(data, 0)?;
    Ok(obj)
}

/// Encode a Query request payload: length-prefixed UTF-8.
pub fn encode_query_payload(query: &str) -> Vec<u8> {
    let q_bytes = query.as_bytes();
    let mut buf = Vec::with_capacity(4 + q_bytes.len());
    buf.extend_from_slice(&(q_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(q_bytes);
    buf
}

/// Decode a Query request payload.
/// A decoded `REQ_VECTOR_BATCH` request: which Vector field to ingest into, and
/// the `(object_id, vector)` rows.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorBatch {
    pub type_name: String,
    pub field_name: String,
    pub rows: Vec<(u64, Vec<f32>)>,
}

/// Decode a `REQ_VECTOR_BATCH` payload:
/// `[type_len:u16 BE][type utf8][field_len:u16 BE][field utf8][count:u32 BE]`
/// then `count` rows of `[object_id:u64 BE][dim:u32 BE][f32 x dim, little-endian]`.
pub fn decode_vector_batch_payload(data: &[u8]) -> io::Result<VectorBatch> {
    fn err(m: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, m.into())
    }
    let mut pos = 0usize;

    let read_str = |data: &[u8], pos: &mut usize, what: &str| -> io::Result<String> {
        if data.len() < *pos + 2 {
            return Err(err(format!("vector-batch: {what} length missing")));
        }
        let n = u16::from_be_bytes(data[*pos..*pos + 2].try_into().unwrap()) as usize;
        *pos += 2;
        if data.len() < *pos + n {
            return Err(err(format!("vector-batch: {what} truncated")));
        }
        let s = std::str::from_utf8(&data[*pos..*pos + n])
            .map_err(|e| err(format!("vector-batch: {what} utf8: {e}")))?
            .to_string();
        *pos += n;
        Ok(s)
    };

    let type_name = read_str(data, &mut pos, "type")?;
    let field_name = read_str(data, &mut pos, "field")?;

    if data.len() < pos + 4 {
        return Err(err("vector-batch: count missing"));
    }
    let count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut rows = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        if data.len() < pos + 12 {
            return Err(err("vector-batch: row header truncated"));
        }
        let object_id = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let dim = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let byte_len = dim * 4;
        if data.len() < pos + byte_len {
            return Err(err("vector-batch: vector data truncated"));
        }
        let mut v = Vec::with_capacity(dim);
        for c in data[pos..pos + byte_len].chunks_exact(4) {
            v.push(f32::from_le_bytes(c.try_into().unwrap()));
        }
        pos += byte_len;
        rows.push((object_id, v));
    }
    Ok(VectorBatch {
        type_name,
        field_name,
        rows,
    })
}

/// Encode a `REQ_VECTOR_BATCH` payload — the exact inverse of
/// [`decode_vector_batch_payload`]:
/// `[type_len:u16 BE][type utf8][field_len:u16 BE][field utf8][count:u32 BE]`
/// then `count` rows of `[object_id:u64 BE][dim:u32 BE][f32 x dim, little-endian]`.
///
/// `rows` is generic over the per-row vector storage (`Vec<f32>`, `&[f32]`,
/// `[f32; N]`, …) so callers need not clone into owned `Vec`s. The byte layout is
/// identical regardless, so the server decodes any of them interchangeably.
///
/// Rows of differing dimensions still encode correctly (each row carries its own
/// `dim`); a caller that wants a uniform-dimension guarantee must check first.
pub fn encode_vector_batch_payload<V: AsRef<[f32]>>(
    type_name: &str,
    field_name: &str,
    rows: &[(u64, V)],
) -> Vec<u8> {
    let t = type_name.as_bytes();
    let f = field_name.as_bytes();
    // Capacity is a hint; use the first row's dim as the per-row estimate and
    // saturate so a pathological input can't overflow the size computation.
    let est_dim = rows.first().map(|(_, v)| v.as_ref().len()).unwrap_or(0);
    let per_row = 12usize.saturating_add(est_dim.saturating_mul(4));
    let cap = (2 + t.len() + 2 + f.len() + 4).saturating_add(rows.len().saturating_mul(per_row));
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&(t.len() as u16).to_be_bytes());
    buf.extend_from_slice(t);
    buf.extend_from_slice(&(f.len() as u16).to_be_bytes());
    buf.extend_from_slice(f);
    buf.extend_from_slice(&(rows.len() as u32).to_be_bytes());
    for (object_id, vector) in rows {
        let v = vector.as_ref();
        buf.extend_from_slice(&object_id.to_be_bytes());
        buf.extend_from_slice(&(v.len() as u32).to_be_bytes());
        for component in v {
            buf.extend_from_slice(&component.to_le_bytes());
        }
    }
    buf
}

pub fn decode_query_payload(data: &[u8]) -> io::Result<String> {
    if data.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "query length missing"));
    }
    let len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    if data.len() < 4 + len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "query body truncated"));
    }
    std::str::from_utf8(&data[4..4 + len])
        .map(|s| s.to_string())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("query utf8: {e}")))
}

/// Encode a `Prepare` request payload — identical to a `Query` payload.
pub fn encode_prepare_payload(query: &str) -> Vec<u8> {
    encode_query_payload(query)
}

/// Encode an `Execute` request payload: `[stmt_id: u64 BE]`.
pub fn encode_execute_payload(stmt_id: u64) -> Vec<u8> {
    stmt_id.to_be_bytes().to_vec()
}

/// Decode an `Execute` request payload into the statement id.
pub fn decode_execute_payload(data: &[u8]) -> io::Result<u64> {
    if data.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "execute: statement id missing",
        ));
    }
    Ok(u64::from_be_bytes(data[0..8].try_into().unwrap()))
}

/// Encode a `Prepared` response into a caller-provided buffer: `[stmt_id: u64 BE]`.
pub fn encode_prepared_payload_into(stmt_id: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&stmt_id.to_be_bytes());
}

/// Encode a `Prepared` response payload: `[stmt_id: u64 BE]`.
pub fn encode_prepared_payload(stmt_id: u64) -> Vec<u8> {
    stmt_id.to_be_bytes().to_vec()
}

/// Decode a `Prepared` response payload into the statement id.
pub fn decode_prepared_payload(data: &[u8]) -> io::Result<u64> {
    if data.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "prepared: statement id missing",
        ));
    }
    Ok(u64::from_be_bytes(data[0..8].try_into().unwrap()))
}

/// Encode an Error response payload: length-prefixed UTF-8 message.
pub fn encode_error_payload(msg: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + msg.len());
    encode_error_payload_into(msg, &mut buf);
    buf
}

/// Encode an Error response into a caller-provided buffer.
pub fn encode_error_payload_into(msg: &str, out: &mut Vec<u8>) {
    let bytes = msg.as_bytes();
    out.reserve(4 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Decode an Error response payload.
pub fn decode_error_payload(data: &[u8]) -> io::Result<String> {
    if data.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "error length missing"));
    }
    let len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    if data.len() < 4 + len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "error body truncated"));
    }
    std::str::from_utf8(&data[4..4 + len])
        .map(|s| s.to_string())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("error utf8: {e}")))
}

// ---------------------------------------------------------------------------
// Subscriptions
// ---------------------------------------------------------------------------

const FILTER_FLAG_TYPE: u8 = 0x01;
const FILTER_FLAG_OBJECT: u8 = 0x02;
/// Bits beyond the two defined flags must be zero (forward-compat fail-closed).
const FILTER_FLAG_RESERVED: u8 = !(FILTER_FLAG_TYPE | FILTER_FLAG_OBJECT);

const KIND_BIT_CREATE: u8 = 0x01;
const KIND_BIT_UPDATE: u8 = 0x02;
const KIND_BIT_DELETE: u8 = 0x04;
const KIND_BITS_RESERVED: u8 = !(KIND_BIT_CREATE | KIND_BIT_UPDATE | KIND_BIT_DELETE);

fn kind_bit(kind: ChangeKind) -> u8 {
    match kind {
        ChangeKind::Create => KIND_BIT_CREATE,
        ChangeKind::Update => KIND_BIT_UPDATE,
        ChangeKind::Delete => KIND_BIT_DELETE,
    }
}

/// Encode a `Subscribe` filter payload:
/// `[flags:u8 bit0=has_type bit1=has_object_id]`
/// `[if has_type: type_len:u16 BE, type utf8]`
/// `[if has_object_id: object_id:u64 BE]`
/// `[kinds:u8 bitmask bit0=Create bit1=Update bit2=Delete; 0 = all kinds]`.
///
/// Two "match-all" encodings to keep distinct: clearing `has_type` matches all
/// types (vs. an empty type name, which is rejected on decode); a `0` kinds
/// bitmask matches all kinds (vs. naming specific bits).
pub fn encode_subscribe_filter(filter: &SubscriptionFilter) -> Vec<u8> {
    let mut out = Vec::new();
    // Treat an empty type name as "no type filter": the decoder rejects a
    // zero-length name (the only "match-all" encoding clears the flag), so
    // emitting one here would produce a payload our own decoder refuses. This
    // keeps encode→decode a faithful round-trip.
    let type_name = filter.type_name.as_deref().filter(|s| !s.is_empty());
    let mut flags = 0u8;
    if type_name.is_some() {
        flags |= FILTER_FLAG_TYPE;
    }
    if filter.object_id.is_some() {
        flags |= FILTER_FLAG_OBJECT;
    }
    out.push(flags);
    if let Some(tn) = type_name {
        let b = tn.as_bytes();
        out.extend_from_slice(&(b.len() as u16).to_be_bytes());
        out.extend_from_slice(b);
    }
    if let Some(oid) = filter.object_id {
        out.extend_from_slice(&oid.to_be_bytes());
    }
    let mut kinds = 0u8;
    for k in &filter.kinds {
        kinds |= kind_bit(*k);
    }
    out.push(kinds);
    out
}

/// Decode a `Subscribe` filter payload. STRICT: rejects reserved flag/kind bits,
/// a `has_type` flag with a zero-length name, truncation, and trailing bytes —
/// so a malformed or forward-incompatible filter fails loud rather than silently
/// subscribing to the wrong (or empty) stream.
pub fn decode_subscribe_filter(data: &[u8]) -> io::Result<SubscriptionFilter> {
    fn err(m: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, m.into())
    }
    let mut pos = 0usize;
    if data.is_empty() {
        return Err(err("subscribe: flags byte missing"));
    }
    let flags = data[pos];
    pos += 1;
    if flags & FILTER_FLAG_RESERVED != 0 {
        return Err(err("subscribe: reserved flag bits set"));
    }

    let type_name = if flags & FILTER_FLAG_TYPE != 0 {
        if pos + 2 > data.len() {
            return Err(err("subscribe: type length missing"));
        }
        let n = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if n == 0 {
            return Err(err("subscribe: has_type set but type name is empty"));
        }
        if pos + n > data.len() {
            return Err(err("subscribe: type name truncated"));
        }
        let s = std::str::from_utf8(&data[pos..pos + n])
            .map_err(|e| err(format!("subscribe: type utf8: {e}")))?
            .to_string();
        pos += n;
        Some(s)
    } else {
        None
    };

    let object_id = if flags & FILTER_FLAG_OBJECT != 0 {
        if pos + 8 > data.len() {
            return Err(err("subscribe: object_id missing"));
        }
        let oid = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        Some(oid)
    } else {
        None
    };

    if pos + 1 > data.len() {
        return Err(err("subscribe: kinds byte missing"));
    }
    let kinds_byte = data[pos];
    pos += 1;
    if kinds_byte & KIND_BITS_RESERVED != 0 {
        return Err(err("subscribe: reserved kind bits set"));
    }
    let mut kinds = Vec::new();
    if kinds_byte & KIND_BIT_CREATE != 0 {
        kinds.push(ChangeKind::Create);
    }
    if kinds_byte & KIND_BIT_UPDATE != 0 {
        kinds.push(ChangeKind::Update);
    }
    if kinds_byte & KIND_BIT_DELETE != 0 {
        kinds.push(ChangeKind::Delete);
    }

    if pos != data.len() {
        return Err(err("subscribe: trailing bytes after filter"));
    }

    Ok(SubscriptionFilter {
        type_name,
        object_id,
        kinds,
    })
}

/// Encode an `Unsubscribe` request payload: `[sub_req_id: u32 BE]`.
pub fn encode_unsubscribe_payload(sub_req_id: u32) -> Vec<u8> {
    sub_req_id.to_be_bytes().to_vec()
}

/// Decode an `Unsubscribe` request payload into the target subscription handle.
pub fn decode_unsubscribe_payload(data: &[u8]) -> io::Result<u32> {
    if data.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsubscribe: subscription handle missing",
        ));
    }
    Ok(u32::from_be_bytes(data[0..4].try_into().unwrap()))
}

/// The wire form of a [`ChangeEvent`] (the `RESP_EVENT` JSON payload).
///
/// It is a NOTIFICATION envelope, not a faithful state snapshot:
/// - `id`/`version` are decimal STRINGS so 64-bit values survive a JS
///   `JSON.parse` (which would coerce a bare number to an f64 and lose
///   precision past 2^53) — `id` is the key you re-query by, so it must be exact.
/// - `fields` are BEST-EFFORT: they use the `/query` read form (`Bytes` as
///   base64, `DateTime` as RFC 3339, `Json` inline), but large 64-bit scalar
///   values may lose precision in a JS `JSON.parse`. `Delete` events carry the
///   deleted object's last-known scalar fields (same shape as create/update);
///   only a cascade-deleted pure edge-only join row (no scalar fields) omits
///   them. For authoritative values, re-query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireEvent {
    /// Format tag, [`EVENT_FORMAT_TAG`].
    pub v: String,
    /// `"create"` | `"update"` | `"delete"`.
    pub kind: String,
    #[serde(rename = "type")]
    pub type_name: String,
    /// The object id, as a decimal string.
    pub id: String,
    /// The commit version, as a decimal string.
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<HashMap<String, serde_json::Value>>,
}

impl WireEvent {
    pub fn from_change_event(ev: &ChangeEvent) -> Self {
        let kind = match ev.kind {
            ChangeKind::Create => "create",
            ChangeKind::Update => "update",
            ChangeKind::Delete => "delete",
        };
        WireEvent {
            v: EVENT_FORMAT_TAG.to_string(),
            kind: kind.to_string(),
            type_name: ev.type_name.clone(),
            id: ev.object_id.to_string(),
            version: ev.version.to_string(),
            fields: ev.fields.clone(),
        }
    }
}

/// Encode a `RESP_EVENT` payload (JSON of a [`WireEvent`]) into `out`.
pub fn encode_event_payload_into(ev: &ChangeEvent, out: &mut Vec<u8>) {
    // Serialization of a well-formed WireEvent cannot fail (all fields are
    // plain JSON-representable); fall back to an empty object defensively.
    match serde_json::to_vec(&WireEvent::from_change_event(ev)) {
        Ok(bytes) => out.extend_from_slice(&bytes),
        Err(_) => out.extend_from_slice(b"{}"),
    }
}

/// Decode a `RESP_EVENT` payload.
pub fn decode_event_payload(data: &[u8]) -> io::Result<WireEvent> {
    serde_json::from_slice(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("event json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{FieldMap, Value};

    fn sample_object() -> Object {
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("age".into(), Value::U32(30));
        fields.insert("happy".into(), Value::Bool(true));
        fields.insert("score".into(), Value::F32(99.5));
        Object {
            type_name: "User".into(),
            id: 42,
            fields,
            raw_fields: None,
        }
    }

    #[test]
    fn execute_and_prepared_payload_roundtrip() {
        for id in [0u64, 1, 42, u64::MAX] {
            assert_eq!(
                decode_execute_payload(&encode_execute_payload(id)).unwrap(),
                id
            );
            assert_eq!(
                decode_prepared_payload(&encode_prepared_payload(id)).unwrap(),
                id
            );
        }
        // `encode_prepare_payload` is a Query payload — round-trips via the query decoder.
        assert_eq!(
            decode_query_payload(&encode_prepare_payload("User.get(1)")).unwrap(),
            "User.get(1)"
        );
    }

    #[test]
    fn execute_and_prepared_reject_short_buffers() {
        assert!(decode_execute_payload(&[0u8; 7]).is_err());
        assert!(decode_prepared_payload(&[0u8; 4]).is_err());
        assert!(decode_execute_payload(&[]).is_err());
    }

    #[test]
    fn vector_batch_decode_roundtrip() {
        // Build a payload exactly as the Python client does and decode it.
        let type_name = "Doc";
        let field_name = "embedding";
        let rows: Vec<(u64, Vec<f32>)> =
            vec![(1, vec![0.5, -1.0, 2.0]), (7, vec![3.0, 4.0, 5.0])];

        let mut payload = Vec::new();
        payload.extend_from_slice(&(type_name.len() as u16).to_be_bytes());
        payload.extend_from_slice(type_name.as_bytes());
        payload.extend_from_slice(&(field_name.len() as u16).to_be_bytes());
        payload.extend_from_slice(field_name.as_bytes());
        payload.extend_from_slice(&(rows.len() as u32).to_be_bytes());
        for (id, vec) in &rows {
            payload.extend_from_slice(&id.to_be_bytes());
            payload.extend_from_slice(&(vec.len() as u32).to_be_bytes());
            for f in vec {
                payload.extend_from_slice(&f.to_le_bytes());
            }
        }

        let batch = decode_vector_batch_payload(&payload).unwrap();
        assert_eq!(batch.type_name, "Doc");
        assert_eq!(batch.field_name, "embedding");
        assert_eq!(batch.rows, rows);

        // The encoder must produce byte-identical output to the hand-built
        // payload above (and round-trip through the decoder).
        let encoded = encode_vector_batch_payload(type_name, field_name, &rows);
        assert_eq!(encoded, payload, "encoder must match the canonical layout");
        let back = decode_vector_batch_payload(&encoded).unwrap();
        assert_eq!(back, batch);

        // The encoder is generic over the vector storage: borrowed slices encode
        // identically to owned Vecs.
        let borrowed: Vec<(u64, &[f32])> =
            rows.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        assert_eq!(
            encode_vector_batch_payload(type_name, field_name, &borrowed),
            payload
        );

        // An empty batch encodes a well-formed zero-count payload.
        let empty: Vec<(u64, Vec<f32>)> = Vec::new();
        let enc_empty = encode_vector_batch_payload("Doc", "embedding", &empty);
        let dec_empty = decode_vector_batch_payload(&enc_empty).unwrap();
        assert!(dec_empty.rows.is_empty());
        assert_eq!(dec_empty.type_name, "Doc");

        // Rows of DIFFERING dimensions round-trip (each row carries its own dim),
        // which is the documented guarantee and the one thing distinguishing this
        // from a flat fixed-stride layout.
        let mixed: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.5, 1.0]), (2, vec![3.0, 4.0, 5.0])];
        let dec_mixed = decode_vector_batch_payload(&encode_vector_batch_payload(
            "Doc", "embedding", &mixed,
        ))
        .unwrap();
        assert_eq!(dec_mixed.rows, mixed);

        // Truncated payloads are rejected, not panicked on.
        assert!(decode_vector_batch_payload(&payload[..payload.len() - 1]).is_err());
        assert!(decode_vector_batch_payload(&[0, 1]).is_err());
    }

    #[test]
    fn object_roundtrip() {
        let obj = sample_object();
        let mut buf = Vec::new();
        encode_object(&obj, &mut buf);
        let (decoded, end) = decode_object(&buf, 0).unwrap();
        assert_eq!(end, buf.len());
        assert_eq!(decoded.type_name, obj.type_name);
        assert_eq!(decoded.id, obj.id);
        assert_eq!(decoded.fields, obj.fields);
    }

    #[test]
    fn objects_payload_roundtrip() {
        let objs = vec![sample_object(), sample_object()];
        let payload = encode_objects_payload(&objs);
        let decoded = decode_objects_payload(&payload).unwrap();
        assert_eq!(decoded.len(), 2);
        for d in &decoded {
            assert_eq!(d.type_name, objs[0].type_name);
            assert_eq!(d.fields, objs[0].fields);
        }
    }

    #[test]
    fn empty_objects_payload() {
        let payload = encode_objects_payload(&[]);
        let decoded = decode_objects_payload(&payload).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn query_payload_roundtrip() {
        let q = "User.get(1)";
        let payload = encode_query_payload(q);
        let decoded = decode_query_payload(&payload).unwrap();
        assert_eq!(decoded, q);
    }

    #[test]
    fn error_payload_roundtrip() {
        let msg = "parse error at position 5";
        let payload = encode_error_payload(msg);
        let decoded = decode_error_payload(&payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn subscribe_filter_roundtrip_all_shapes() {
        let cases = vec![
            SubscriptionFilter::all(),
            SubscriptionFilter::for_type("User"),
            SubscriptionFilter::for_object("Post", 12345678901234567u64),
            SubscriptionFilter {
                type_name: Some("User".into()),
                object_id: None,
                kinds: vec![ChangeKind::Create, ChangeKind::Delete],
            },
            SubscriptionFilter {
                type_name: None,
                object_id: None,
                kinds: vec![ChangeKind::Update],
            },
        ];
        for f in cases {
            let bytes = encode_subscribe_filter(&f);
            let back = decode_subscribe_filter(&bytes).unwrap();
            assert_eq!(back.type_name, f.type_name);
            assert_eq!(back.object_id, f.object_id);
            assert_eq!(back.kinds, f.kinds);
        }
    }

    #[test]
    fn subscribe_filter_strict_rejections() {
        // has_type set but zero-length name.
        assert!(decode_subscribe_filter(&[FILTER_FLAG_TYPE, 0, 0, 0]).is_err());
        // reserved flag bit set.
        assert!(decode_subscribe_filter(&[0x80, 0]).is_err());
        // reserved kind bit set (bit3 = 0x08).
        assert!(decode_subscribe_filter(&[0x00, 0x08]).is_err());
        // empty payload.
        assert!(decode_subscribe_filter(&[]).is_err());
        // missing kinds byte (flags only).
        assert!(decode_subscribe_filter(&[0x00]).is_err());
        // truncated type.
        assert!(decode_subscribe_filter(&[FILTER_FLAG_TYPE, 0, 4, b'U', b's']).is_err());
        // trailing bytes after a valid all-filter.
        assert!(decode_subscribe_filter(&[0x00, 0x00, 0xFF]).is_err());
    }

    #[test]
    fn encode_filter_normalises_empty_type_name() {
        // Some("") must not emit a zero-length name the strict decoder rejects;
        // the encoder normalises it to "no type filter".
        let f = SubscriptionFilter {
            type_name: Some(String::new()),
            object_id: None,
            kinds: vec![],
        };
        let bytes = encode_subscribe_filter(&f);
        let back = decode_subscribe_filter(&bytes).expect("must round-trip");
        assert_eq!(back.type_name, None);
    }

    #[test]
    fn sync_frame_roundtrip_and_bounds() {
        use std::io::Cursor;
        // write_frame then read_frame round-trips req_id / kind / payload — this
        // is the synchronous client's actual frame path (no tokio feature).
        let payload = encode_query_payload("User.get(7)");
        let mut wire = Vec::new();
        sync::write_frame(&mut wire, 99, REQ_QUERY, &payload).unwrap();
        let frame = sync::read_frame(&mut Cursor::new(wire)).unwrap();
        assert_eq!(frame.req_id, 99);
        assert_eq!(frame.kind, REQ_QUERY);
        assert_eq!(decode_query_payload(&frame.payload).unwrap(), "User.get(7)");

        // An empty-payload frame (Ping-like) round-trips.
        let mut wire = Vec::new();
        sync::write_frame(&mut wire, 1, REQ_PING, &[]).unwrap();
        let f = sync::read_frame(&mut Cursor::new(wire)).unwrap();
        assert_eq!((f.req_id, f.kind, f.payload.len()), (1, REQ_PING, 0));

        // A declared length below the 5-byte header minimum is rejected.
        assert!(sync::read_frame(&mut Cursor::new(3u32.to_be_bytes().to_vec())).is_err());
        // An oversized declared length is rejected before allocating the body.
        let too_big = (MAX_FRAME_PAYLOAD as u32 + 6).to_be_bytes().to_vec();
        assert!(sync::read_frame(&mut Cursor::new(too_big)).is_err());
        // A truncated body (header promises more than is present) errs, no panic.
        let mut short = 9u32.to_be_bytes().to_vec(); // len=9 → 4 payload bytes expected
        short.extend_from_slice(&[0, 0, 0, 1, REQ_PING]); // req_id+kind, zero payload
        assert!(sync::read_frame(&mut Cursor::new(short)).is_err());
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn frame_reader_handles_segmentation_and_coalescing() {
        use tokio::io::duplex;
        let (mut client, mut server) = duplex(64 * 1024);

        // Build three frames.
        let f1 = encode_query_payload("User.get(1)");
        let f2 = encode_query_payload("User.get(2)");
        let mut two = Vec::new();
        // Frame 3 carries an empty payload (Ping-like).
        let mk = |req: u32, kind: u8, p: &[u8]| {
            let mut v = Vec::new();
            v.extend_from_slice(&((5 + p.len()) as u32).to_be_bytes());
            v.extend_from_slice(&req.to_be_bytes());
            v.push(kind);
            v.extend_from_slice(p);
            v
        };
        let frame1 = mk(1, REQ_QUERY, &f1);
        let frame2 = mk(2, REQ_QUERY, &f2);
        let frame3 = mk(3, REQ_PING, &[]);

        let mut reader = FrameReader::new();
        let reader_task = tokio::spawn(async move {
            let a = reader.next_frame(&mut server).await.unwrap();
            let b = reader.next_frame(&mut server).await.unwrap();
            let c = reader.next_frame(&mut server).await.unwrap();
            (a, b, c)
        });

        // Frame 1: write it ONE BYTE AT A TIME (worst-case segmentation).
        for byte in &frame1 {
            tokio::io::AsyncWriteExt::write_all(&mut client, &[*byte]).await.unwrap();
            tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();
        }
        // Frames 2 and 3 in a SINGLE write (coalesced / read-ahead).
        two.extend_from_slice(&frame2);
        two.extend_from_slice(&frame3);
        tokio::io::AsyncWriteExt::write_all(&mut client, &two).await.unwrap();
        tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();

        let (a, b, c) = reader_task.await.unwrap();
        assert_eq!((a.req_id, a.kind), (1, REQ_QUERY));
        assert_eq!(decode_query_payload(&a.payload).unwrap(), "User.get(1)");
        assert_eq!((b.req_id, b.kind), (2, REQ_QUERY));
        assert_eq!(decode_query_payload(&b.payload).unwrap(), "User.get(2)");
        assert_eq!((c.req_id, c.kind, c.payload.len()), (3, REQ_PING, 0));
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn frame_reader_survives_cancellation_mid_frame() {
        use tokio::io::duplex;
        let (mut client, mut server) = duplex(64 * 1024);
        let payload = encode_query_payload("User.get(7)");
        let mut frame = Vec::new();
        frame.extend_from_slice(&((5 + payload.len()) as u32).to_be_bytes());
        frame.extend_from_slice(&9u32.to_be_bytes());
        frame.push(REQ_QUERY);
        frame.extend_from_slice(&payload);

        let mut reader = FrameReader::new();
        // Write the first half, then repeatedly CANCEL next_frame (drop the
        // future via select! with an immediately-ready branch) — a cancel-unsafe
        // reader would lose the buffered half and desync.
        let half = frame.len() / 2;
        tokio::io::AsyncWriteExt::write_all(&mut client, &frame[..half]).await.unwrap();
        tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();
        for _ in 0..5 {
            tokio::select! {
                biased;
                _ = std::future::ready(()) => {} // always wins → cancels next_frame
                _ = reader.next_frame(&mut server) => panic!("frame not complete yet"),
            }
        }
        // Now write the rest; the frame must still parse intact.
        tokio::io::AsyncWriteExt::write_all(&mut client, &frame[half..]).await.unwrap();
        tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();
        let f = reader.next_frame(&mut server).await.unwrap();
        assert_eq!((f.req_id, f.kind), (9, REQ_QUERY));
        assert_eq!(decode_query_payload(&f.payload).unwrap(), "User.get(7)");
    }

    #[test]
    fn unsubscribe_payload_roundtrip() {
        for id in [0u32, 1, 42, u32::MAX] {
            assert_eq!(
                decode_unsubscribe_payload(&encode_unsubscribe_payload(id)).unwrap(),
                id
            );
        }
        assert!(decode_unsubscribe_payload(&[0u8; 3]).is_err());
    }

    #[test]
    fn wire_event_roundtrip_and_lossless_ids() {
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), serde_json::json!("Alice"));
        let ev = ChangeEvent {
            version: 9_007_199_254_740_993, // 2^53 + 1: would lose precision as a JS number
            kind: ChangeKind::Update,
            type_name: "User".into(),
            object_id: 18_446_744_073_709_551_615, // u64::MAX
            fields: Some(fields),
        };
        let mut buf = Vec::new();
        encode_event_payload_into(&ev, &mut buf);
        let wire = decode_event_payload(&buf).unwrap();
        assert_eq!(wire.v, EVENT_FORMAT_TAG);
        assert_eq!(wire.kind, "update");
        assert_eq!(wire.type_name, "User");
        // Ids are decimal strings, exact even past 2^53.
        assert_eq!(wire.id, "18446744073709551615");
        assert_eq!(wire.version, "9007199254740993");
        assert_eq!(
            wire.fields.unwrap().get("name"),
            Some(&serde_json::json!("Alice"))
        );

        // An event with no captured fields (e.g. a cascade-deleted edge-only
        // join row) omits the `fields` key entirely (`skip_serializing_if`).
        let del_no_fields = ChangeEvent {
            version: 7,
            kind: ChangeKind::Delete,
            type_name: "User".into(),
            object_id: 5,
            fields: None,
        };
        let mut dbuf = Vec::new();
        encode_event_payload_into(&del_no_fields, &mut dbuf);
        let json = String::from_utf8(dbuf.clone()).unwrap();
        assert!(!json.contains("fields"), "fields-less event must omit fields: {json}");
        assert_eq!(decode_event_payload(&dbuf).unwrap().kind, "delete");

        // A delete event WITH captured scalar fields carries them on the wire,
        // exactly like create/update — a subscriber learns which object went.
        let del_with_fields = ChangeEvent {
            version: 8,
            kind: ChangeKind::Delete,
            type_name: "User".into(),
            object_id: 5,
            fields: Some({
                let mut m = HashMap::new();
                m.insert("name".into(), serde_json::json!("Alice"));
                m
            }),
        };
        let mut dbuf2 = Vec::new();
        encode_event_payload_into(&del_with_fields, &mut dbuf2);
        let decoded = decode_event_payload(&dbuf2).unwrap();
        assert_eq!(decoded.kind, "delete");
        assert_eq!(
            decoded.fields.unwrap().get("name"),
            Some(&serde_json::json!("Alice"))
        );
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn frame_roundtrip_async() {
        use tokio::io::duplex;
        let (mut a, mut b) = duplex(4096);

        let payload = encode_query_payload("User.limit(10)");
        write_frame(&mut a, 42, REQ_QUERY, &payload).await.unwrap();

        let frame = read_frame(&mut b).await.unwrap();
        assert_eq!(frame.req_id, 42);
        assert_eq!(frame.kind, REQ_QUERY);
        let query = decode_query_payload(&frame.payload).unwrap();
        assert_eq!(query, "User.limit(10)");
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn rejects_oversize_frame() {
        use tokio::io::duplex;
        let (mut a, mut b) = duplex(8);
        // Write a frame header claiming 32 MB.
        let bogus_len: u32 = 32 * 1024 * 1024;
        tokio::io::AsyncWriteExt::write_all(&mut a, &bogus_len.to_be_bytes())
            .await
            .unwrap();
        // Reader should reject before allocating.
        let result = read_frame(&mut b).await;
        assert!(result.is_err());
    }
}
