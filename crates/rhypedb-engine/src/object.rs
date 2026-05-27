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
#[derive(Debug, Clone)]
pub struct Object {
    pub type_name: String,
    pub id: u64,
    pub fields: FieldMap,
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

/// Serialize a FieldMap to bytes for storage.
pub fn serialize_fields(fields: &FieldMap) -> Bytes {
    let mut buf = BytesMut::new();

    // Number of fields.
    buf.put_u16(fields.len() as u16);

    for (name, value) in fields {
        // Field name (length-prefixed).
        buf.put_u16(name.len() as u16);
        buf.put_slice(name.as_bytes());

        // Value.
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

    buf.freeze()
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
