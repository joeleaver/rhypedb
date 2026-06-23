//! The binary wire protocol: constants, payload codecs, and framing.
//!
//! The implementation moved to the shared [`rhypedb_wire::protocol`] crate so
//! the official client can speak the protocol without depending on the server
//! (and so there is a single source of truth for the byte formats). Re-exported
//! here so every existing `protocol::*` path in this crate is unchanged.
//!
//! The async framing (`read_frame`/`write_frame`/`write_frame_buffered`/
//! `FrameReader`, over tokio's `AsyncRead`/`AsyncWrite`) is provided by
//! `rhypedb-wire`'s `tokio` feature, which this crate enables in its manifest.

pub use rhypedb_wire::protocol::*;
