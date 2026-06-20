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
//!
//! A `Prepare` parses + caches the query for the LIFETIME OF THE CONNECTION and
//! returns a `Prepared(stmt_id)`; subsequent `Execute(stmt_id)` re-run it with no
//! re-parse and no re-sent query string. Statement IDs are per-connection (like a
//! Postgres session) and are dropped when the connection closes.
//!
//! # Server response types
//!
//! | byte | name     | payload                                            |
//! |------|----------|----------------------------------------------------|
//! | 0x80 | Objects  | `[count: u32 BE]` then `count` encoded objects     |
//! | 0x81 | Single   | one encoded object                                 |
//! | 0x82 | Done     | empty                                              |
//! | 0x83 | Error    | `[msg_len: u32 BE][utf8 msg]`                      |
//! | 0x84 | Pong     | empty                                              |
//! | 0x85 | Prepared | `[stmt_id: u64 BE]`                                |
//!
//! # Object encoding
//!
//! ```text
//! [type_name_len: u16 BE][type_name: utf8]
//! [id: u64 BE]
//! [fields: same format as engine::serialize_fields]
//! ```

use std::io;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use rhypedb_engine::object::{deserialize_fields, serialize_fields_into, Object};

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

// Response types
pub const RESP_OBJECTS: u8 = 0x80;
pub const RESP_SINGLE: u8 = 0x81;
pub const RESP_DONE: u8 = 0x82;
pub const RESP_ERROR: u8 = 0x83;
pub const RESP_PONG: u8 = 0x84;
/// Reply to `Prepare`: the assigned per-connection statement id.
pub const RESP_PREPARED: u8 = 0x85;

/// A parsed inbound frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub req_id: u32,
    pub kind: u8,
    pub payload: Vec<u8>,
}

/// Read one frame from an async reader. Returns Err on connection close or
/// malformed framing.
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

/// Write one frame to an async writer. Issues 4 successive `write_all` calls
/// which the underlying BufWriter coalesces before flushing.
///
/// Hot callers should prefer `write_frame_buffered` so the entire frame —
/// header + payload — is built in one contiguous buffer and shipped with a
/// single `write_all`, sidestepping the BufWriter's intermediate state
/// machine.
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
    // with its own length-prefixed name and tagged value. We walk it to know
    // where it ends.
    let fields_start = pos;
    let fields_end = scan_fields_end(data, pos)?;
    let fields = deserialize_fields(&data[fields_start..fields_end]);
    pos = fields_end;

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

/// Walk through a fields encoding to find its end offset. The fields encoding
/// (matching engine::serialize_fields) is:
///   [count: u16][ (name_len: u16)(name)(tag: u8)(value...) ]*
fn scan_fields_end(data: &[u8], start: usize) -> io::Result<usize> {
    let mut pos = start;
    if pos + 2 > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "fields count truncated"));
    }
    let count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    for _ in 0..count {
        if pos + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "field name truncated"));
        }
        let name_len = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2 + name_len;

        if pos + 1 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "field tag missing"));
        }
        let tag = data[pos];
        pos += 1;

        let value_len = match tag {
            0 => 0,                                       // Null
            1 => read_u32_len(data, &mut pos)?,           // String
            2 => 4,                                       // U32
            3 => 8,                                       // U64
            4 => 4,                                       // I32
            5 => 8,                                       // I64
            6 => 4,                                       // F32
            7 => 8,                                       // F64
            8 => 1,                                       // Bool
            9 => read_u32_len(data, &mut pos)?,           // Bytes
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown value tag {tag}"),
                ))
            }
        };
        pos += value_len;
    }
    Ok(pos)
}

fn read_u32_len(data: &[u8], pos: &mut usize) -> io::Result<usize> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "length prefix truncated"));
    }
    let len = u32::from_be_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    Ok(len)
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
    let mut objects = Vec::with_capacity(count);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_engine::object::{FieldMap, Value};

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
