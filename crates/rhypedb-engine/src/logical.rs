//! Logical export/import — a portable, version-independent dump format.
//!
//! The physical backup (`Database::backup_to`) is fast and complete but ties
//! the artifact to the on-disk SST/WAL/catalog layout. A *logical* dump reads
//! data out through the object layer into a self-describing NDJSON stream that
//! survives storage-format and version changes, can move between systems, is
//! human-inspectable, and supports selective per-type export.
//!
//! This module is the pure foundation: the typed `{"t","v"}` value envelope
//! that round-trips every [`Value`] variant losslessly. The streaming
//! export/import drivers build on top of it.
//!
//! # Why a typed envelope, not bare JSON
//!
//! Naive JSON loses fidelity the database cannot afford:
//! * a JSON number is an IEEE-754 `f64`, so a `u64`/`i64` past `2^53` silently
//!   rounds — every 64-bit integer is therefore a **decimal string**;
//! * a JSON number cannot represent `NaN`, `±∞`, or distinguish `-0.0`, and an
//!   `f32` widened to `f64` and back can drift — every float is therefore the
//!   **lowercase hex of its IEEE-754 bit pattern** (8 nibbles for `f32`, 16 for
//!   `f64`), which is exact and total;
//! * `Bytes` is standard base64.
//!
//! `String`, `Bool`, and `Null` are carried as native JSON for readability.
//! The `t` tag drives decoding, so each value rebuilds the *exact* original
//! [`Value`] variant — `F32` and `F64` never collapse together even when their
//! decimal text would collide.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::Value as Json;

use crate::object::{FieldMap, Value};

/// An error decoding a logical-export value or field map.
#[derive(Debug, thiserror::Error)]
pub enum LogicalError {
    #[error("malformed export value: {0}")]
    MalformedValue(String),
}

type LResult<T> = std::result::Result<T, LogicalError>;

fn malformed(msg: impl Into<String>) -> LogicalError {
    LogicalError::MalformedValue(msg.into())
}

/// Encode a single [`Value`] to its typed `{"t":<tag>,"v":<repr>}` envelope.
/// Lossless and parser-agnostic — see the module docs for the encoding rules.
pub fn value_to_json(v: &Value) -> Json {
    match v {
        Value::Null => env("null", Json::Null),
        Value::String(s) => env("str", Json::String(s.clone())),
        Value::U32(n) => env("u32", Json::String(n.to_string())),
        Value::U64(n) => env("u64", Json::String(n.to_string())),
        Value::I32(n) => env("i32", Json::String(n.to_string())),
        Value::I64(n) => env("i64", Json::String(n.to_string())),
        Value::F32(x) => env("f32", Json::String(format!("{:08x}", x.to_bits()))),
        Value::F64(x) => env("f64", Json::String(format!("{:016x}", x.to_bits()))),
        Value::Bool(b) => env("bool", Json::Bool(*b)),
        Value::Bytes(b) => env("bytes", Json::String(B64.encode(b))),
    }
}

/// Decode a typed `{"t","v"}` envelope back to the exact [`Value`] variant.
pub fn value_from_json(j: &Json) -> LResult<Value> {
    let obj = j
        .as_object()
        .ok_or_else(|| malformed("value envelope must be a JSON object"))?;
    let tag = obj
        .get("t")
        .and_then(Json::as_str)
        .ok_or_else(|| malformed("value envelope missing string field `t`"))?;
    let v = obj.get("v").unwrap_or(&Json::Null);

    match tag {
        "null" => Ok(Value::Null),
        "str" => Ok(Value::String(as_str(v, tag)?.to_owned())),
        "u32" => Ok(Value::U32(parse_int(v, tag)?)),
        "u64" => Ok(Value::U64(parse_int(v, tag)?)),
        "i32" => Ok(Value::I32(parse_int(v, tag)?)),
        "i64" => Ok(Value::I64(parse_int(v, tag)?)),
        "f32" => Ok(Value::F32(f32::from_bits(parse_hex(v, tag, 8)? as u32))),
        "f64" => Ok(Value::F64(f64::from_bits(parse_hex(v, tag, 16)?))),
        "bool" => Ok(Value::Bool(
            v.as_bool()
                .ok_or_else(|| malformed("`bool` value must be a JSON boolean"))?,
        )),
        "bytes" => {
            let s = as_str(v, tag)?;
            let bytes = B64
                .decode(s)
                .map_err(|e| malformed(format!("invalid base64 `bytes`: {e}")))?;
            Ok(Value::Bytes(bytes::Bytes::from(bytes)))
        }
        other => Err(malformed(format!("unknown value tag `{other}`"))),
    }
}

/// Encode a [`FieldMap`] to a JSON object of `name -> {"t","v"}` envelopes.
pub fn fields_to_json(fields: &FieldMap) -> Json {
    let mut m = serde_json::Map::with_capacity(fields.len());
    for (name, value) in fields {
        m.insert(name.clone(), value_to_json(value));
    }
    Json::Object(m)
}

/// Decode a `name -> {"t","v"}` JSON object back to a [`FieldMap`].
pub fn fields_from_json(j: &Json) -> LResult<FieldMap> {
    let obj = j
        .as_object()
        .ok_or_else(|| malformed("fields must be a JSON object"))?;
    let mut out = FieldMap::with_capacity(obj.len());
    for (name, raw) in obj {
        let value = value_from_json(raw).map_err(|e| malformed(format!("field `{name}`: {e}")))?;
        out.insert(name.clone(), value);
    }
    Ok(out)
}

fn env(tag: &str, v: Json) -> Json {
    let mut m = serde_json::Map::with_capacity(2);
    m.insert("t".to_owned(), Json::String(tag.to_owned()));
    m.insert("v".to_owned(), v);
    Json::Object(m)
}

fn as_str<'a>(v: &'a Json, tag: &str) -> LResult<&'a str> {
    v.as_str()
        .ok_or_else(|| malformed(format!("`{tag}` value must be a JSON string")))
}

fn parse_int<T>(v: &Json, tag: &str) -> LResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let s = as_str(v, tag)?;
    s.parse::<T>()
        .map_err(|e| malformed(format!("invalid `{tag}` integer {s:?}: {e}")))
}

/// Parse a fixed-width lowercase-hex IEEE-754 bit pattern. `width` is the
/// expected nibble count (8 for `f32`, 16 for `f64`) — enforced so a short or
/// over-long string is rejected rather than silently zero-extended.
fn parse_hex(v: &Json, tag: &str, width: usize) -> LResult<u64> {
    let s = as_str(v, tag)?;
    if s.len() != width {
        return Err(malformed(format!(
            "`{tag}` expects {width} hex digits, got {} ({s:?})",
            s.len()
        )));
    }
    u64::from_str_radix(s, 16).map_err(|e| malformed(format!("invalid `{tag}` hex {s:?}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a value and assert it decodes to the identical variant.
    /// Floats are compared by bit pattern so `NaN` and `-0.0` are exact.
    fn assert_round_trips(v: Value) {
        let json = value_to_json(&v);
        let back = value_from_json(&json).expect("decodes");
        match (&v, &back) {
            (Value::F32(a), Value::F32(b)) => {
                assert_eq!(a.to_bits(), b.to_bits(), "f32 bits differ for {json}")
            }
            (Value::F64(a), Value::F64(b)) => {
                assert_eq!(a.to_bits(), b.to_bits(), "f64 bits differ for {json}")
            }
            _ => assert_eq!(v, back, "round trip differs via {json}"),
        }
    }

    #[test]
    fn every_variant_round_trips() {
        for v in [
            Value::Null,
            Value::String(String::new()),
            Value::String("héllo \" world\n🦀".to_owned()),
            Value::Bool(true),
            Value::Bool(false),
            Value::U32(0),
            Value::U32(u32::MAX),
            Value::U64(0),
            Value::U64(u64::MAX),
            Value::I32(i32::MIN),
            Value::I32(i32::MAX),
            Value::I64(i64::MIN),
            Value::I64(i64::MAX),
            Value::F32(0.0),
            Value::F32(-0.0),
            Value::F32(core::f32::consts::PI),
            Value::F32(f32::INFINITY),
            Value::F32(f32::NEG_INFINITY),
            Value::F32(f32::NAN),
            Value::F64(0.0),
            Value::F64(-0.0),
            Value::F64(core::f64::consts::PI),
            Value::F64(f64::INFINITY),
            Value::F64(f64::NAN),
            Value::Bytes(bytes::Bytes::new()),
            Value::Bytes(bytes::Bytes::from_static(&[0, 1, 2, 255, 254, 0, 128])),
        ] {
            assert_round_trips(v);
        }
    }

    #[test]
    fn u64_past_f64_precision_survives() {
        // 2^53 + 1 is the canonical value a JSON-number encoding would corrupt.
        let v = Value::U64((1u64 << 53) + 1);
        let json = value_to_json(&v);
        assert_eq!(json["v"], Json::String("9007199254740993".to_owned()));
        assert_eq!(value_from_json(&json).unwrap(), v);
    }

    #[test]
    fn f32_and_f64_do_not_collapse() {
        // 0.1 has different bit patterns (and different decimal-shortest text)
        // as f32 vs f64; the tag must keep them distinct variants.
        let a = value_to_json(&Value::F32(0.1));
        let b = value_to_json(&Value::F64(0.1));
        assert_ne!(a, b);
        assert!(matches!(value_from_json(&a).unwrap(), Value::F32(_)));
        assert!(matches!(value_from_json(&b).unwrap(), Value::F64(_)));
    }

    #[test]
    fn field_map_round_trips() {
        let mut fm = FieldMap::new();
        fm.insert("name".to_owned(), Value::String("Ada".to_owned()));
        fm.insert("age".to_owned(), Value::U32(36));
        fm.insert("score".to_owned(), Value::F64(98.6));
        fm.insert("blob".to_owned(), Value::Bytes(bytes::Bytes::from_static(b"\x00\xff")));
        fm.insert("maybe".to_owned(), Value::Null);

        let json = fields_to_json(&fm);
        let back = fields_from_json(&json).expect("decodes");
        assert_eq!(fm, back);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        // Not an object.
        assert!(value_from_json(&Json::String("x".into())).is_err());
        // Missing tag.
        assert!(value_from_json(&serde_json::json!({"v": "1"})).is_err());
        // Unknown tag.
        assert!(value_from_json(&serde_json::json!({"t": "u128", "v": "1"})).is_err());
        // Integer not a string.
        assert!(value_from_json(&serde_json::json!({"t": "u64", "v": 1})).is_err());
        // Integer out of range for the declared width.
        assert!(value_from_json(&serde_json::json!({"t": "u32", "v": "4294967296"})).is_err());
        // Float hex wrong width.
        assert!(value_from_json(&serde_json::json!({"t": "f32", "v": "abc"})).is_err());
        assert!(value_from_json(&serde_json::json!({"t": "f64", "v": "0badc0de"})).is_err());
        // Bad base64.
        assert!(value_from_json(&serde_json::json!({"t": "bytes", "v": "!!!!"})).is_err());
        // Bool not a boolean.
        assert!(value_from_json(&serde_json::json!({"t": "bool", "v": "true"})).is_err());
    }
}
