use std::collections::HashMap;

use bytes::{BufMut, Bytes, BytesMut};

/// A dynamically-typed value for object fields.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Bytes(Bytes),
    Null,
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::String(_) => "String",
            Value::U32(_) => "u32",
            Value::U64(_) => "u64",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "Bool",
            Value::Bytes(_) => "Bytes",
            Value::Null => "Null",
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => write!(f, "{s}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            Value::Null => write!(f, "null"),
        }
    }
}

/// A set of field values for an object. Used for create/update operations
/// and for query results.
pub type FieldMap = HashMap<String, Value>;

/// A stored object with its identity.
///
/// `raw_fields` is an optional shortcut populated when the object came
/// straight from an LSM read (`Database::get_many`) — the bytes are the
/// already-serialized FieldMap and can be shipped to the TCP wire encoder
/// directly without a deserialize-then-reserialize round trip. When set,
/// `fields` is empty until a consumer calls `ensure_fields_deserialized`.
/// Hot path on terminal materialize: 50+ object encodes per query save the
/// `deserialize_fields` + `HashMap` construction + drop chain.
#[derive(Debug, Clone)]
pub struct Object {
    pub type_name: String,
    pub id: u64,
    pub fields: FieldMap,
    pub raw_fields: Option<Bytes>,
}

impl Object {
    /// Construct a "lazy" Object whose serialized FieldMap is held in
    /// `raw_fields`. `fields` stays empty until something asks for it.
    pub fn from_raw(type_name: String, id: u64, raw_fields: Bytes) -> Self {
        Self {
            type_name,
            id,
            fields: FieldMap::new(),
            raw_fields: Some(raw_fields),
        }
    }

    /// Populate `fields` from `raw_fields` if not already populated.
    /// Idempotent. Keeps `raw_fields` set so a subsequent wire encode still
    /// gets the fast-path (emit raw bytes directly, skip re-serialize) —
    /// the small memory cost of holding both is worth it for code paths
    /// that need both (Filter predicate then wire-emit, HTTP JSON path).
    pub fn ensure_fields_deserialized(&mut self) {
        if !self.fields.is_empty() {
            return;
        }
        if let Some(raw) = &self.raw_fields {
            self.fields = deserialize_fields(raw);
        }
    }
}

/// Serialization tags for the binary field encoding.
#[repr(u8)]
enum ValueTag {
    Null = 0,
    String = 1,
    U32 = 2,
    U64 = 3,
    I32 = 4,
    I64 = 5,
    F32 = 6,
    F64 = 7,
    Bool = 8,
    Bytes = 9,
}

/// Serialize a FieldMap to bytes for storage. Convenience wrapper around
/// `serialize_fields_into` that allocates a fresh `BytesMut` — callers that
/// can pre-size or reuse a buffer (TCP response encoder, batched writes)
/// should call the `_into` variant directly.
pub fn serialize_fields(fields: &FieldMap) -> Bytes {
    let mut buf = BytesMut::with_capacity(estimate_fields_size(fields));
    serialize_fields_into_bytesmut(fields, &mut buf);
    buf.freeze()
}

/// Serialize a FieldMap directly into a caller-provided `Vec<u8>`. Avoids
/// the intermediate `Bytes` allocation that callers on the hot encode path
/// would otherwise pay (TCP response, log emission, etc.).
pub fn serialize_fields_into(fields: &FieldMap, out: &mut Vec<u8>) {
    out.reserve(estimate_fields_size(fields));
    out.extend_from_slice(&(fields.len() as u16).to_be_bytes());
    for (name, value) in fields {
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.extend_from_slice(name.as_bytes());
        write_value_into(value, out);
    }
}

fn serialize_fields_into_bytesmut(fields: &FieldMap, buf: &mut BytesMut) {
    buf.put_u16(fields.len() as u16);
    for (name, value) in fields {
        buf.put_u16(name.len() as u16);
        buf.put_slice(name.as_bytes());
        match value {
            Value::Null => buf.put_u8(ValueTag::Null as u8),
            Value::String(s) => {
                buf.put_u8(ValueTag::String as u8);
                buf.put_u32(s.len() as u32);
                buf.put_slice(s.as_bytes());
            }
            Value::U32(v) => {
                buf.put_u8(ValueTag::U32 as u8);
                buf.put_u32(*v);
            }
            Value::U64(v) => {
                buf.put_u8(ValueTag::U64 as u8);
                buf.put_u64(*v);
            }
            Value::I32(v) => {
                buf.put_u8(ValueTag::I32 as u8);
                buf.put_i32(*v);
            }
            Value::I64(v) => {
                buf.put_u8(ValueTag::I64 as u8);
                buf.put_i64(*v);
            }
            Value::F32(v) => {
                buf.put_u8(ValueTag::F32 as u8);
                buf.put_f32(*v);
            }
            Value::F64(v) => {
                buf.put_u8(ValueTag::F64 as u8);
                buf.put_f64(*v);
            }
            Value::Bool(v) => {
                buf.put_u8(ValueTag::Bool as u8);
                buf.put_u8(u8::from(*v));
            }
            Value::Bytes(b) => {
                buf.put_u8(ValueTag::Bytes as u8);
                buf.put_u32(b.len() as u32);
                buf.put_slice(b);
            }
        }
    }
}

fn write_value_into(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Null => out.push(ValueTag::Null as u8),
        Value::String(s) => {
            out.push(ValueTag::String as u8);
            out.extend_from_slice(&(s.len() as u32).to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Value::U32(v) => {
            out.push(ValueTag::U32 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::U64(v) => {
            out.push(ValueTag::U64 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::I32(v) => {
            out.push(ValueTag::I32 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::I64(v) => {
            out.push(ValueTag::I64 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::F32(v) => {
            out.push(ValueTag::F32 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::F64(v) => {
            out.push(ValueTag::F64 as u8);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::Bool(v) => {
            out.push(ValueTag::Bool as u8);
            out.push(u8::from(*v));
        }
        Value::Bytes(b) => {
            out.push(ValueTag::Bytes as u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
    }
}

/// Estimate the serialized size of a FieldMap for buffer pre-sizing. Rough
/// upper bound — over-estimates on Null/numeric fields, under-estimates on
/// huge string/bytes. Goal is to avoid the first few `Vec::reserve` doublings
/// on small-object payloads.
fn estimate_fields_size(fields: &FieldMap) -> usize {
    // 2 bytes count + per-field (name_len 2 + name + tag 1 + ~8 value bytes)
    let mut total = 2;
    for (name, value) in fields {
        total += 2 + name.len() + 1;
        total += match value {
            Value::String(s) => 4 + s.len(),
            Value::Bytes(b) => 4 + b.len(),
            Value::Null | Value::Bool(_) => 0,
            Value::U32(_) | Value::I32(_) | Value::F32(_) => 4,
            Value::U64(_) | Value::I64(_) | Value::F64(_) => 8,
        };
    }
    total
}

/// Scan a serialized FieldMap (same format as `serialize_fields`) for a
/// single field by name, returning its raw `Bytes` payload (the Value::Bytes
/// inner slice) if present, or `None` if the field is absent / wrong type /
/// malformed. Caller-provided `data_owner` is used to slice into; pass the
/// same Bytes the lookup is happening on so the returned slice shares its
/// owner.
///
/// Hot path: 2-hop fusion reads the next-hop target's *covered object data*
/// from the source's reverse-edge value — emitting an Object directly
/// without an LSM probe at terminal materialize.
pub fn find_bytes_field_in_raw(data: &Bytes, field_name: &str) -> Option<Bytes> {
    if data.len() < 2 {
        return None;
    }
    let mut pos = 0;
    let count = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
    pos += 2;
    let needle = field_name.as_bytes();

    for _ in 0..count {
        if pos + 2 > data.len() {
            return None;
        }
        let name_len = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
        pos += 2;
        if pos + name_len + 1 > data.len() {
            return None;
        }
        let name_match = name_len == needle.len() && &data[pos..pos + name_len] == needle;
        pos += name_len;
        let tag = data[pos];
        pos += 1;

        let value_len = match tag {
            0 => 0,
            1 => {
                if pos + 4 > data.len() { return None; }
                let l = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                l
            }
            2 => 4,
            3 => 8,
            4 => 4,
            5 => 8,
            6 => 4,
            7 => 8,
            8 => 1,
            9 => {
                if pos + 4 > data.len() { return None; }
                let l = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                l
            }
            _ => return None,
        };

        if name_match {
            if pos + value_len > data.len() {
                return None;
            }
            if tag == 9 {
                // Value::Bytes — zero-copy slice into the owning Bytes.
                return Some(data.slice(pos..pos + value_len));
            }
            return None;
        }
        pos += value_len;
    }
    None
}

/// Scan a serialized FieldMap (same format as `serialize_fields`) for a
/// single field by name, returning its U64 value if present and tagged as
/// U32 / U64 / I32 / I64 (each widened to u64). Returns `None` if the
/// field is absent, has a different type, or the data is malformed.
///
/// Avoids the full `deserialize_fields` allocation when the caller knows it
/// only needs one integer field — most notably the forward-1:1 fusion path
/// in 2-hop traversals, which extracts the next-hop target id from carried
/// covering reverse-edge values.
pub fn find_u64_field_in_raw(data: &[u8], field_name: &str) -> Option<u64> {
    if data.len() < 2 {
        return None;
    }
    let mut pos = 0;
    let count = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
    pos += 2;
    let needle = field_name.as_bytes();

    for _ in 0..count {
        if pos + 2 > data.len() {
            return None;
        }
        let name_len = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
        pos += 2;
        if pos + name_len + 1 > data.len() {
            return None;
        }
        let name_match = name_len == needle.len() && &data[pos..pos + name_len] == needle;
        pos += name_len;
        let tag = data[pos];
        pos += 1;

        let value_len = match tag {
            0 => 0,                                      // Null
            1 => {
                if pos + 4 > data.len() { return None; }
                let l = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                l
            }
            2 => 4, // U32
            3 => 8, // U64
            4 => 4, // I32
            5 => 8, // I64
            6 => 4, // F32
            7 => 8, // F64
            8 => 1, // Bool
            9 => {
                if pos + 4 > data.len() { return None; }
                let l = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                l
            }
            _ => return None,
        };

        if name_match {
            // Decode if it's an integer tag the fusion path can use.
            if pos + value_len > data.len() {
                return None;
            }
            return match tag {
                2 => Some(u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as u64),
                3 => Some(u64::from_be_bytes(data[pos..pos + 8].try_into().ok()?)),
                4 => Some(i32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as u64),
                5 => Some(i64::from_be_bytes(data[pos..pos + 8].try_into().ok()?) as u64),
                _ => None,
            };
        }
        pos += value_len;
    }
    None
}

/// Deserialize a FieldMap from stored bytes.
pub fn deserialize_fields(data: &[u8]) -> FieldMap {
    let mut fields = HashMap::new();
    let mut pos = 0;

    if data.len() < 2 {
        return fields;
    }

    let count = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    for _ in 0..count {
        let name_len = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .unwrap()
            .to_string();
        pos += name_len;

        let tag = data[pos];
        pos += 1;

        let value = match tag {
            0 => Value::Null,
            1 => {
                let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                let s = std::str::from_utf8(&data[pos..pos + len])
                    .unwrap()
                    .to_string();
                pos += len;
                Value::String(s)
            }
            2 => {
                let v = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
                Value::U32(v)
            }
            3 => {
                let v = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                Value::U64(v)
            }
            4 => {
                let v = i32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
                Value::I32(v)
            }
            5 => {
                let v = i64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                Value::I64(v)
            }
            6 => {
                let v = f32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
                Value::F32(v)
            }
            7 => {
                let v = f64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                Value::F64(v)
            }
            8 => {
                let v = data[pos] != 0;
                pos += 1;
                Value::Bool(v)
            }
            9 => {
                let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                let b = Bytes::copy_from_slice(&data[pos..pos + len]);
                pos += len;
                Value::Bytes(b)
            }
            _ => Value::Null,
        };

        fields.insert(name, value);
    }

    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_roundtrip() {
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("age".into(), Value::U32(30));
        fields.insert("score".into(), Value::F64(95.5));
        fields.insert("active".into(), Value::Bool(true));

        let encoded = serialize_fields(&fields);
        let decoded = deserialize_fields(&encoded);

        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(decoded.get("age"), Some(&Value::U32(30)));
        assert_eq!(decoded.get("score"), Some(&Value::F64(95.5)));
        assert_eq!(decoded.get("active"), Some(&Value::Bool(true)));
    }

    #[test]
    fn serialize_empty() {
        let fields = FieldMap::new();
        let encoded = serialize_fields(&fields);
        let decoded = deserialize_fields(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn find_u64_field_in_raw_finds_each_int_tag() {
        let mut fields = FieldMap::new();
        fields.insert("u32_field".into(), Value::U32(42));
        fields.insert("u64_field".into(), Value::U64(0xdead_beef_cafe));
        fields.insert("i32_field".into(), Value::I32(-7));
        fields.insert("i64_field".into(), Value::I64(-9_000_000_000));
        let bytes = serialize_fields(&fields);

        assert_eq!(find_u64_field_in_raw(&bytes, "u32_field"), Some(42));
        assert_eq!(find_u64_field_in_raw(&bytes, "u64_field"), Some(0xdead_beef_cafe));
        assert_eq!(find_u64_field_in_raw(&bytes, "i32_field"), Some(-7i64 as u64));
        assert_eq!(
            find_u64_field_in_raw(&bytes, "i64_field"),
            Some(-9_000_000_000i64 as u64)
        );
    }

    #[test]
    fn find_u64_field_in_raw_misses_non_int_and_missing() {
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("hi".into()));
        fields.insert("x".into(), Value::U64(7));
        let bytes = serialize_fields(&fields);

        assert_eq!(find_u64_field_in_raw(&bytes, "name"), None, "string tag isn't an int");
        assert_eq!(find_u64_field_in_raw(&bytes, "x"), Some(7));
        assert_eq!(find_u64_field_in_raw(&bytes, "nope"), None);
    }

    #[test]
    fn find_u64_field_in_raw_handles_empty() {
        assert_eq!(find_u64_field_in_raw(&[], "x"), None);
        let empty = serialize_fields(&FieldMap::new());
        assert_eq!(find_u64_field_in_raw(&empty, "x"), None);
    }

    #[test]
    fn serialize_bytes_value() {
        let mut fields = FieldMap::new();
        fields.insert("data".into(), Value::Bytes(Bytes::from_static(b"binary")));

        let encoded = serialize_fields(&fields);
        let decoded = deserialize_fields(&encoded);

        assert_eq!(
            decoded.get("data"),
            Some(&Value::Bytes(Bytes::from_static(b"binary")))
        );
    }
}
