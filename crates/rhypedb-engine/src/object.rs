//! The value model and field-serialization codec. Moved to the shared
//! [`rhypedb_wire::object`] crate so the official client can decode wire objects
//! without depending on the engine; re-exported here so every existing
//! `rhypedb_engine::object::*` / `crate::object::*` path is unchanged.

pub use rhypedb_wire::object::*;
