//! Full-text index build markers and progress.
//!
//! Every `@fulltext` field owns one persisted **build marker**
//! (`c:X:<type_id><field_id>`, see `KeyBuilder::catalog_fulltext_marker`)
//! recording the field's live index identity and build state:
//!
//! ```text
//! [format u8 = 1][state u8][generation u32 BE][cursor u64 BE]
//! [positions u8][stale_generations u8][analyzer_len u16 BE][analyzer utf-8]
//! ```
//!
//! * `generation` namespaces the field's `f:`/`l:` keys. It is bumped when
//!   the analyzer or `positions` setting changes, so the live writers and the
//!   rebuild write fresh keys while the previous generation's rows are
//!   dropped in the background — no reader ever sees a mix of two token
//!   streams.
//! * `state` — `Building` (the backfill has not covered every object yet;
//!   `.matches` refuses with progress), `Built`, or `Dropping` (the directive
//!   was removed / the field or type dropped; every row of every generation
//!   is being deleted, then the marker itself).
//! * `cursor` — the object id the backfill resumes after (the write paths
//!   keep indexing new writes regardless, and the backfill skips any object
//!   that already has an `l:` row, so resuming is idempotent).
//! * `stale_generations` — set on a generation bump until the background
//!   sweep has deleted every row of the older generations.
//!
//! The marker is the ONLY durable build state; in-memory progress
//! ([`BuildProgress`]) is rebuilt from it at open.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use bytes::Bytes;

/// Build state of one field's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuildState {
    Building = 0,
    Built = 1,
    Dropping = 2,
}

impl BuildState {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Building),
            1 => Some(Self::Built),
            2 => Some(Self::Dropping),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Building => "building",
            Self::Built => "built",
            Self::Dropping => "dropping",
        }
    }
}

const MARKER_FORMAT: u8 = 1;

/// The persisted build marker of one `@fulltext` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildMarker {
    pub state: BuildState,
    pub generation: u32,
    pub cursor: u64,
    pub positions: bool,
    pub stale_generations: bool,
    pub analyzer: String,
}

impl BuildMarker {
    pub fn encode(&self) -> Bytes {
        let mut out = Vec::with_capacity(1 + 1 + 4 + 8 + 1 + 1 + 2 + self.analyzer.len());
        out.push(MARKER_FORMAT);
        out.push(self.state as u8);
        out.extend_from_slice(&self.generation.to_be_bytes());
        out.extend_from_slice(&self.cursor.to_be_bytes());
        out.push(self.positions as u8);
        out.push(self.stale_generations as u8);
        out.extend_from_slice(&(self.analyzer.len() as u16).to_be_bytes());
        out.extend_from_slice(self.analyzer.as_bytes());
        Bytes::from(out)
    }

    /// Decode a marker value. `None` = malformed (treated by the caller as
    /// "absent": a fresh build, which is always safe — the rows are
    /// re-derivable and the backfill skips already-indexed objects).
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 18 || bytes[0] != MARKER_FORMAT {
            return None;
        }
        let state = BuildState::from_byte(bytes[1])?;
        let generation = u32::from_be_bytes(bytes[2..6].try_into().ok()?);
        let cursor = u64::from_be_bytes(bytes[6..14].try_into().ok()?);
        let positions = match bytes[14] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let stale_generations = match bytes[15] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let len = u16::from_be_bytes(bytes[16..18].try_into().ok()?) as usize;
        if bytes.len() != 18 + len {
            return None;
        }
        let analyzer = std::str::from_utf8(&bytes[18..]).ok()?.to_string();
        Some(Self {
            state,
            generation,
            cursor,
            positions,
            stale_generations,
            analyzer,
        })
    }

    /// Whether this marker describes the same index identity as the schema's
    /// directive (same analyzer and positions setting).
    pub fn matches_config(&self, analyzer: &str, positions: bool) -> bool {
        self.analyzer == analyzer && self.positions == positions
    }
}

/// Live, in-memory view of one field's build, mirrored from the marker and
/// advanced by the builder thread. Read by `fulltext_search` (the gate) and
/// `GET /status` (progress).
#[derive(Debug)]
pub struct BuildProgress {
    state: AtomicU8,
    /// Objects the backfill has visited so far (indexed or already indexed).
    visited: AtomicU64,
    /// Objects of the type when the backfill started (0 until it starts).
    total: AtomicU64,
}

impl BuildProgress {
    pub fn new(state: BuildState) -> Self {
        Self {
            state: AtomicU8::new(state as u8),
            visited: AtomicU64::new(0),
            total: AtomicU64::new(0),
        }
    }

    pub fn state(&self) -> BuildState {
        BuildState::from_byte(self.state.load(Ordering::Acquire)).unwrap_or(BuildState::Building)
    }

    pub fn set_state(&self, state: BuildState) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub fn visited(&self) -> u64 {
        self.visited.load(Ordering::Relaxed)
    }

    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn add_visited(&self, n: u64) {
        self.visited.fetch_add(n, Ordering::Relaxed);
    }
}

/// One line of `GET /status`'s `fulltext.indexes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FulltextIndexStatus {
    /// `Type.field`.
    pub name: String,
    pub state: BuildState,
    pub generation: u32,
    /// Documents currently in the index (corpus stat).
    pub documents: u64,
    /// Backfill progress: objects visited / objects at backfill start. Both
    /// 0 once a build finished at open on an empty type.
    pub indexed: u64,
    pub total: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_round_trips_and_rejects_malformed() {
        let m = BuildMarker {
            state: BuildState::Building,
            generation: 7,
            cursor: 123_456,
            positions: false,
            stale_generations: true,
            analyzer: "simple".into(),
        };
        let enc = m.encode();
        assert_eq!(BuildMarker::decode(&enc), Some(m.clone()));
        assert!(m.matches_config("simple", false));
        assert!(!m.matches_config("simple", true));
        assert!(!m.matches_config("english", false));
        // Every state.
        for st in [BuildState::Building, BuildState::Built, BuildState::Dropping] {
            let m2 = BuildMarker { state: st, ..m.clone() };
            assert_eq!(BuildMarker::decode(&m2.encode()).unwrap().state, st);
        }
        // Malformed shapes decode to None (caller treats as absent).
        assert_eq!(BuildMarker::decode(&[]), None);
        assert_eq!(BuildMarker::decode(&enc[..10]), None);
        let mut bad_format = enc.to_vec();
        bad_format[0] = 9;
        assert_eq!(BuildMarker::decode(&bad_format), None);
        let mut bad_state = enc.to_vec();
        bad_state[1] = 9;
        assert_eq!(BuildMarker::decode(&bad_state), None);
        let mut trailing = enc.to_vec();
        trailing.push(0);
        assert_eq!(BuildMarker::decode(&trailing), None);
    }

    #[test]
    fn progress_tracks_state_and_counters() {
        let p = BuildProgress::new(BuildState::Building);
        assert_eq!(p.state(), BuildState::Building);
        p.set_total(10);
        p.add_visited(4);
        p.add_visited(6);
        assert_eq!((p.visited(), p.total()), (10, 10));
        p.set_state(BuildState::Built);
        assert_eq!(p.state(), BuildState::Built);
        assert_eq!(BuildState::Dropping.as_str(), "dropping");
    }
}
