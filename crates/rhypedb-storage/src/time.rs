//! The clock every rhypedb crate reads.
//!
//! On every target with a clock this is `std::time`, unchanged. On
//! `wasm32-unknown-unknown` `std::time::Instant::now` and `SystemTime::now`
//! panic, so there the same names come from `web-time`, which reads the
//! JavaScript host's clock (`performance.now()` / `Date.now()`, both present in
//! a Web Worker). The types have std's API, so code is written once against
//! `rhypedb_storage::time::{Instant, SystemTime}` (the engine re-exports it as
//! `rhypedb_engine::time`).

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use web_time::{Instant, SystemTime, UNIX_EPOCH};
