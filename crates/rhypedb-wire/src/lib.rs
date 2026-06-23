//! rhypedb-wire — the transport- and runtime-agnostic wire layer shared by the
//! rhypedb server and the official client library.
//!
//! - [`object`] — the value model ([`object::Value`], [`object::Object`],
//!   [`object::FieldMap`]) and the field-serialization codec used on the wire
//!   and in the LSM. Pure: depends only on `base64`, `bytes`, `time`, and
//!   `serde_json`.
//! - [`protocol`] — the binary protocol constants and payload codecs. Payload
//!   encode/decode are pure sync functions over buffers. Sync framing lives in
//!   [`protocol::sync`]; async framing (over tokio's `AsyncRead`/`AsyncWrite`)
//!   is behind the optional `tokio` feature and used by the server.
//!
//! The engine and server re-export these (`rhypedb_engine::object`,
//! `rhypedb_server::protocol`) so existing paths are unchanged.

pub mod object;
pub mod protocol;
