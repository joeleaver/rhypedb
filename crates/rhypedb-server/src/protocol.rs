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
//! | byte | name  | payload                                  |
//! |------|-------|------------------------------------------|
//! | 0x01 | Query | `[q_len: u32 BE][utf8 query]`            |
//! | 0x02 | Ping  | empty                                    |
//!
//! # Server response types
//!
//! | byte | name    | payload                                            |
//! |------|---------|----------------------------------------------------|
//! | 0x80 | Objects | `[count: u32 BE]` then `count` encoded objects     |
//! | 0x81 | Single  | one encoded object                                 |
//! | 0x82 | Done    | empty                                              |
//! | 0x83 | Error   | `[msg_len: u32 BE][utf8 msg]`                      |
//! | 0x84 | Pong    | empty                                              |
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

use rhypedb_engine::object::{deserialize_fields, serialize_fields, Object};

/// Maximum payload size per frame (16 MB). Defensive limit to prevent
/// runaway allocations from malformed clients.
pub const MAX_FRAME_PAYLOAD: usize = 16 * 1024 * 1024;

// Request types
pub const REQ_QUERY: u8 = 0x01;
pub const REQ_PING: u8 = 0x02;

// Response types
pub const RESP_OBJECTS: u8 = 0x80;
pub const RESP_SINGLE: u8 = 0x81;
pub const RESP_DONE: u8 = 0x82;
pub const RESP_ERROR: u8 = 0x83;
pub const RESP_PONG: u8 = 0x84;

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

/// Write one frame to an async writer.
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

/// Encode a single Object to bytes (for inclusion in Objects / Single responses).
pub fn encode_object(obj: &Object, out: &mut Vec<u8>) {
    let type_bytes = obj.type_name.as_bytes();
    out.extend_from_slice(&(type_bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(type_bytes);
    out.extend_from_slice(&obj.id.to_be_bytes());
    let fields_bytes = serialize_fields(&obj.fields);
    out.extend_from_slice(&fields_bytes);
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

/// Encode an Objects response payload.
pub fn encode_objects_payload(objects: &[Object]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(objects.len() as u32).to_be_bytes());
    for obj in objects {
        encode_object(obj, &mut buf);
    }
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

/// Encode an Error response payload: length-prefixed UTF-8 message.
pub fn encode_error_payload(msg: &str) -> Vec<u8> {
    let bytes = msg.as_bytes();
    let mut buf = Vec::with_capacity(4 + bytes.len());
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(bytes);
    buf
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
        }
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
