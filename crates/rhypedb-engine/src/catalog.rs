//! Persisted schema catalog. See card 1/5.
//!
//! ## What this module solves
//!
//! Before this catalog existed, type/field/relation numeric IDs were assigned
//! at [`Database::open`] time by sorting `schema.types.keys()` alphabetically
//! and walking a single global counter. The IDs were baked into every storage
//! key (`o:<type_id>:<obj_id>`, `r:<obj_id>:<rel_id>:…`, index keys) but
//! never persisted. Any edit to the schema — adding a type, renaming a
//! field, dropping a relation — caused the sort order to shift, every ID
//! to renumber, and every existing object to silently decode as garbage.
//!
//! ## Keyspace
//!
//! All catalog rows live under [`KeyPrefix::Catalog`] (`b'c'`). The
//! subtype tag is a single ASCII byte followed by `:`; entry payloads
//! use `\x00` as the inner separator (NUL is rejected by the schema
//! validator, so it never collides with a real identifier).
//!
//! ```text
//! c:F:                              format-version sentinel
//! c:I:                              initialized marker (presence = catalog is whole)
//! c:M:                              metadata header (reserved for phases 2+)
//! c:D:                              schema digest (drives reconcile fast-path)
//!
//! c:T:<type>                        type id-entry
//! c:E:<type>\x00<field>             field id-entry
//! c:R:<type>\x00<field>             relation id-entry
//!
//! c:N:T  c:N:E  c:N:R               next-id counters
//! ```
//!
//! ## Value envelope
//!
//! Every catalog value carries a 2-byte prefix: byte 0 is the
//! record-format version (current = 0x01), byte 1 is a value-kind
//! discriminant (0x01 id-entry, 0x02 counter, 0x03 marker). Id-entry
//! bodies are TLV (tag, u16 BE len, value) so future cards can add
//! tombstone status / aliases / per-row metadata without a format
//! break. Decoders preserve unknown TLV tags so a future-binary's row
//! survives a phase-1 reopen-and-rewrite cycle byte-for-byte.
//!
//! Phase-1 TLV tags written and required for id-entries:
//!
//! | Tag  | Name          | Required | Notes                                                       |
//! |------|---------------|----------|-------------------------------------------------------------|
//! | 0x01 | `id`          | always   | u64 BE                                                      |
//! | 0x02 | `assigned_at` | always   | unix millis when first persisted                            |
//! | 0x03 | `assigned_by` | always   | 0 = backfill, 1 = fresh additive                            |
//! | 0x04 | `kind`        | for `E`/`R` only | discriminant per [`kind_byte`] — closes scalar↔relation flip |
//!
//! Reserved tags (decoder must preserve verbatim on rewrite): 0x10-0x1F
//! tombstones (card 2/5), 0x20-0x2F aliases (card 3/5), 0x30-0x3F type
//! details (card 4/5), 0xF0-0xFF experimental.

use crate::error::{CatalogError, EngineError, EngineResult};
use bytes::Bytes;
use parking_lot::Mutex;
use rhypedb_schema::{FieldType, ScalarType, Schema};
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::LsmTree;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-wide mutex serialising catalog open/reconcile operations.
///
/// The storage layer's `put_batch` makes writes visible to other
/// transactions at memtable-write time, before `commit` runs its
/// write-write conflict check. That means two concurrent reconcile
/// commits — even ones the conflict detector eventually rejects —
/// both leave their writes in the memtable, producing duplicate IDs
/// in the catalog. Until the storage layer adopts true rollback-on-
/// conflict, we serialise catalog work here.
///
/// Cost is negligible: catalog work happens once per `Database::open`,
/// and the lock is uncontended in single-Database deployments (the
/// overwhelming common case). The lock does NOT protect cross-process
/// races — the storage layer doesn't lock the data directory, so
/// multi-process opens of the same dir are already racy regardless.
static CATALOG_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Catalog format versions. v1 is the phase-1 layout (no tombstones). v2
/// is bumped the first time a tombstone is written. A v1 catalog with NO
/// tombstoned rows opens cleanly under a v2 binary and stays at v1; if
/// the operator runs a shrink with `allow_schema_shrink: true`, the
/// reconcile commit bumps `c:F:` to v2 in the same batch that writes the
/// first tombstone TLV. A v1 binary refuses to open a v2 catalog.
pub(crate) const CATALOG_FORMAT_V1: u64 = 1;
pub(crate) const CATALOG_FORMAT_V2: u64 = 2;
/// Bumped the first time a row is renamed. v1 catalogs that never
/// shrank stay at v1; v2 catalogs that never renamed stay at v2.
pub(crate) const CATALOG_FORMAT_V3: u64 = 3;
/// Bumped the first time a field-type change is applied. Earlier-format
/// catalogs that never change a field type stay at their current version.
pub(crate) const CATALOG_FORMAT_V4: u64 = 4;
/// Set by the first successful `rename_field` (card 3/5 phase 2). Pure
/// marker — `c:F:` only — so v4 readers refuse to open a v5 catalog (the
/// pre-rename binary cannot resolve `<old>` to a field that has been
/// renamed away).
pub(crate) const CATALOG_FORMAT_V5: u64 = 5;
pub(crate) const CATALOG_FORMAT_CURRENT: u64 = CATALOG_FORMAT_V5;

/// Per-row record format version. Future phases may bump for rows that
/// carry semantic-meaning TLVs phase 1 can't interpret.
const RECORD_FORMAT_V1: u8 = 0x01;

// Value-kind discriminants (byte 1 of every catalog value).
const KIND_ID_ENTRY: u8 = 0x01;
const KIND_COUNTER: u8 = 0x02;
const KIND_MARKER: u8 = 0x03;
/// Value kind for a shadow-field migration plan record (`c:P:<id>`,
/// card 1/5). Its own decoder; `decode_id_entry`/`decode_counter` reject
/// it via the WrongValueKind arm so a stray plan row can't be misread.
const KIND_MIGRATION_PLAN: u8 = 0x04;
/// Value kind for a per-partition migration cursor (`c:S:<plan><idx>`,
/// card 3/5). A fixed-layout framed value (no TLV) — a torn write
/// decode-FAILS cleanly rather than silently misparsing the cursor.
const KIND_PARTITION_CURSOR: u8 = 0x05;
/// Value kind for a quarantine sidecar record (`c:Q:<plan><object_id>`,
/// card 4/5) — a row whose converter failed under the Quarantine policy.
const KIND_QUARANTINE: u8 = 0x06;

// TLV tags inside id-entry bodies (phase 1).
const TLV_ID: u8 = 0x01;
const TLV_ASSIGNED_AT: u8 = 0x02;
const TLV_ASSIGNED_BY: u8 = 0x03;
const TLV_KIND: u8 = 0x04;

// TLV tags inside id-entry bodies (phase 2 — tombstones).
const TLV_STATUS:         u8 = 0x10;
const TLV_RETIRED_AT_MS:  u8 = 0x11;
const TLV_RETIRED_REASON: u8 = 0x13;
// 0x12 (`retired_at_version`), 0x14 (`retired_by_format`), 0x15
// (`retired_by_actor`), 0x16 (`retired_note`) are reserved for follow-on
// cards. The decoder treats every tag from 0x10-0x1F as a known range —
// if a tag we don't recognise appears here, it's preserved verbatim in
// `unknown_tlvs` and round-tripped on rewrite, so a future binary's row
// survives a phase-2 reopen-and-rewrite cycle byte-for-byte.

// `assigned_by` payload values.
const ASSIGNED_BY_BACKFILL: u8 = 0;
const ASSIGNED_BY_FRESH: u8 = 1;

// Tombstone status payload values for TLV_STATUS (0x10).
const STATUS_LIVE:       u8 = 0x00;
const STATUS_TOMBSTONED: u8 = 0x01;
// 0x02 reserved for future "RetiringInProgress"; the decoder refuses
// any other value rather than guess.

// Retirement reason payload values for TLV_RETIRED_REASON (0x13).
const REASON_EXPLICIT_SHRINK:        u8 = 0x01;
const REASON_CASCADE_PARENT_RETIRED: u8 = 0x02;
// 0x03 reserved for card 4/5 kind-change forced retirement.

// TLV tags inside id-entry bodies (phase 3 — renames).
//
// 0x20 carries a packed `previous_names` chain: u8 count followed by
// `count` RenameRecord structs (each: u16 from_len + utf8 + u16 to_len
// + utf8 + u64 BE timestamp). Most-recent first. Capped at
// MAX_RENAME_HISTORY entries.
// 0x21 mirrors the head-of-chain timestamp for cheap "last touched"
// reads (u64 BE unix millis).
// 0x22-0x2F reserved for follow-on cards (renamed_by_actor, the field-
// rename verb that lands once zone-map keys are field-id-keyed, etc.).
const TLV_PREVIOUS_NAMES:      u8 = 0x20;
const TLV_LAST_RENAMED_AT_MS:  u8 = 0x21;

// TLV tags inside id-entry bodies (phase 4 — field-type changes).
//
// 0x30 carries a packed `type_change_history` chain: u8 count followed
// by `count` TypeChangeRecord structs (each: u8 from_kind + u8 to_kind
// + u64 BE wall_time_unix_ms). Most-recent first. Capped at
// MAX_TYPE_CHANGE_HISTORY entries. Card 4/5 only writes this tag on
// scalar field rows; relation fields' kind byte is RELATION and never
// changes via this verb.
//
// 0x31 mirrors the head-of-chain timestamp for cheap "last touched"
// reads (u64 BE unix millis).
//
// 0x32-0x3F reserved for follow-on cards (resumable cursor, error
// policy, parallelism state, etc. — see the design synthesis).
const TLV_TYPE_CHANGE_HISTORY: u8 = 0x30;
const TLV_LAST_TYPE_CHANGE_AT_MS: u8 = 0x31;

/// Per-row cap on the type-change audit chain. Operators in practice
/// migrate a single field type once or twice across a database's
/// lifetime; 16 is generous and keeps the TLV body small.
pub(crate) const MAX_TYPE_CHANGE_HISTORY: usize = 16;

/// Per-row cap on the rename audit chain. Operators in practice rename
/// a single type a handful of times across a database's lifetime; 32 is
/// generous and keeps each TLV body under ~3KB at maximum-length names.
pub(crate) const MAX_RENAME_HISTORY: usize = 32;

// Bounded retry budget on WriteConflict during catalog commits.
// Concurrent opens fight over the digest write (always present in
// reconcile commits) and the catalog header writes (always present in
// backfill commits). 8 retries is far more than realistic contention.
const COMMIT_RETRY_BUDGET: usize = 8;

// =====================================================================
// Shadow-field migration plan records (card 1/5)
//
// `c:P:<plan_id BE>` → [RECORD_FORMAT_V1][KIND_MIGRATION_PLAN]
//   [u16 BE body_len][TLV body]. One row per chunked field-type
// migration; survives restart to drive auto-resume. The TLV body is
// unknown-tag preserving (mirroring the id-entry codec) so cards 2/4 can
// add fields (double-write armed, error policy, quarantine cursor)
// without a record-format bump or a card-1 binary mangling the row.
// =====================================================================
const TLV_MP_TYPE_NAME: u8 = 0x01;
const TLV_MP_FIELD_NAME: u8 = 0x02;
const TLV_MP_FIELD_ID: u8 = 0x03;
const TLV_MP_SRC_KIND: u8 = 0x04;
const TLV_MP_TARGET_KIND: u8 = 0x05;
const TLV_MP_STATUS: u8 = 0x06;
const TLV_MP_CURSOR: u8 = 0x07;
const TLV_MP_CHUNK_SIZE: u8 = 0x08;
const TLV_MP_CREATED_AT_MS: u8 = 0x09;
const TLV_MP_CONVERTER_NAME: u8 = 0x0A;
const TLV_MP_CONVERTER_VERSION: u8 = 0x0B;
const TLV_MP_OBJECTS_CONVERTED: u8 = 0x0C;
// Card 2 (online migration) additions, in the 0x20-0x3F reserved range so a
// card-1 binary round-trips them verbatim and decodes their absence as the
// card-1 defaults (phase=Converting, cutover_cursor=0).
const TLV_MP_PHASE: u8 = 0x20;
const TLV_MP_CUTOVER_CURSOR: u8 = 0x21;
// Card 3 (parallel workers) — coordinated band split so card 4 doesn't collide:
//   0x22-0x23 = card 3 (parallel_degree, id_upper_bound)
//   0x24-0x2F = reserved for card 4 (error_policy, quarantine cursor)
// `parallel_degree` PRESENCE is the discriminator between a parallel (card-3)
// plan and a legacy single-cursor (card-1/2) plan — NOT a U==0 sentinel. A
// card-1/2 row has neither tag and decodes as `parallel_degree=None`,
// `id_upper_bound=0`. Unknown tags are preserved verbatim and round-tripped.
const TLV_MP_PARALLEL_DEGREE: u8 = 0x22;
const TLV_MP_ID_UPPER_BOUND: u8 = 0x23;
// Card 4 (ErrorPolicy + quarantine + dry-run) — 0x24-0x2F band. Absent on a
// card-1/2/3 row → decode as Stop / not-dry-run / 0 errors / default cap.
const TLV_MP_ERROR_POLICY: u8 = 0x24;
const TLV_MP_DRY_RUN: u8 = 0x25;
const TLV_MP_ERROR_COUNT: u8 = 0x26;
const TLV_MP_QUARANTINE_CAP: u8 = 0x27;

// MigrationStatus payload bytes for TLV_MP_STATUS. The decoder refuses
// any other value rather than guess.
const MP_STATUS_PENDING: u8 = 0x00;
const MP_STATUS_RUNNING: u8 = 0x01;
const MP_STATUS_COMPLETED: u8 = 0x02;
const MP_STATUS_CANCELLED: u8 = 0x03;
const MP_STATUS_FAILED: u8 = 0x04;
const MP_STATUS_AWAITING_CONVERTER: u8 = 0x05;
// Card 4: a dry-run finished — settled + terminal, but the catalog kind was NOT
// flipped (distinct from Completed so no reader mistakes a preflight for a real
// migration). Non-quiescing (the hook was never armed for a dry-run).
const MP_STATUS_DRY_RUN_COMPLETED: u8 = 0x06;

// ErrorPolicy payload bytes for TLV_MP_ERROR_POLICY (card 4).
const MP_ERROR_POLICY_STOP: u8 = 0x00;
const MP_ERROR_POLICY_SKIP_AND_LOG: u8 = 0x01;
const MP_ERROR_POLICY_QUARANTINE: u8 = 0x02;

// MigrationPhase payload bytes for TLV_MP_PHASE.
const MP_PHASE_CONVERTING: u8 = 0x00;
const MP_PHASE_CUTTING_OVER: u8 = 0x01;
// Card 5: a terminal cancel is rolling back (stripping `<field>__shadow`
// siblings). Durable so a crash mid-rollback resumes the strip, not the
// forward migration.
const MP_PHASE_ROLLING_BACK: u8 = 0x02;

/// Default per-migration quarantine cap (card 4). Exceeding it auto-STOPS the
/// migration (parks `Failed`) — a tripwire against a runaway error field (e.g. a
/// forgotten NULL case) flooding `c:Q:`. Overridable per migration (0 → this).
pub(crate) const DEFAULT_QUARANTINE_CAP: u64 = 100_000;

/// Default objects processed per chunk commit. Bounds each chunk's
/// `put_batch` + fsync while amortizing the per-chunk lock/scan overhead;
/// cancel latency is `<= chunk_size` rows. Overridable per migration.
pub(crate) const DEFAULT_MIGRATION_CHUNK_SIZE: u64 = 1024;

/// Upper clamp on a migration's parallel backfill degree (card 3/5). One worker
/// thread per partition; the object-id range `[1, id_upper_bound)` is split into
/// this many contiguous spans. Bounded so a hostile/garbage value can't spawn an
/// unreasonable thread count or overflow the `u8` partition index.
pub(crate) const MAX_PARALLEL_DEGREE: u8 = 64;

/// Discriminants stored in TLV 0x04 of every field/relation id-entry.
/// Catches scalar-type swaps (Int → String) and scalar↔relation flips
/// at open time instead of letting on-disk values be silently
/// reinterpreted under the new kind.
pub(crate) mod kind_byte {
    pub const UNSET: u8 = 0x00;
    pub const SCALAR_STRING: u8 = 0x01;
    pub const SCALAR_U32: u8 = 0x02;
    pub const SCALAR_U64: u8 = 0x03;
    pub const SCALAR_I32: u8 = 0x04;
    pub const SCALAR_I64: u8 = 0x05;
    pub const SCALAR_F32: u8 = 0x06;
    pub const SCALAR_F64: u8 = 0x07;
    pub const SCALAR_BOOL: u8 = 0x08;
    pub const SCALAR_DATETIME: u8 = 0x09;
    pub const SCALAR_BYTES: u8 = 0x0A;
    pub const SCALAR_JSON: u8 = 0x0B;
    pub const VECTOR: u8 = 0x40;
    pub const RELATION: u8 = 0x80;
}

/// Whether a catalog row is live (the schema still names it) or
/// tombstoned (retired forever — its numeric ID will never be reused).
/// Stored on disk in TLV `0x10` of every id-entry row; phase-2 binaries
/// write `Tombstoned` only via the shrink-allowed reconcile path. The
/// implicit value when TLV `0x10` is absent is `Live` — a v1-format row
/// has no status TLV and decodes cleanly as `Live`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TombstoneStatus {
    Live,
    Tombstoned,
}

/// Why a catalog row was retired. Stored on disk in TLV `0x13`. Only
/// meaningful on rows whose status is `Tombstoned`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetireReason {
    /// The operator explicitly removed this entry from the schema.
    ExplicitShrink,
    /// This field or relation was retired because its parent type was
    /// retired in the same reconcile commit.
    CascadeParentRetired,
}

/// One step in a catalog row's rename audit chain. Stored on disk in
/// TLV `0x20` as part of the `previous_names` vector. Most-recent rename
/// is at index 0; the oldest is at `chain.len() - 1`. Card 3/5 only
/// writes records for **type** renames (rename_field is deferred to a
/// follow-on card that addresses zone-map name-hash keying).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameRecord {
    pub from: String,
    pub to: String,
    pub wall_time_unix_ms: u64,
}

/// One step in a catalog row's type-change audit chain. Stored on disk
/// in TLV `0x30` as part of the `type_change_history` vector. Most
/// recent at index 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeChangeRecord {
    pub from_kind: u8,
    pub to_kind: u8,
    pub wall_time_unix_ms: u64,
}

/// A single decoded id-entry row. `unknown_tlvs` carries every TLV the
/// decoder didn't recognise, in original byte order, so a rewrite from
/// a phase-1 binary preserves phase-2+ state verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IdEntry {
    pub id: u64,
    pub assigned_at: u64,
    pub assigned_by: u8,
    pub kind: u8,
    /// `Live` for v1 rows (TLV 0x10 absent) and unretired v2 rows.
    /// `Tombstoned` only on rows the reconcile shrink path wrote.
    pub status: TombstoneStatus,
    /// Unix millis when the tombstone was written. `None` on `Live`
    /// rows. Cheap observability for "when was this retired?" without
    /// loading the (deferred) migration log.
    pub retired_at_ms: Option<u64>,
    /// Why this entry was retired. `None` on `Live` rows.
    pub retired_reason: Option<RetireReason>,
    /// Rename audit chain. Empty for never-renamed entries. The most
    /// recent rename is at index 0. Encoded on disk under TLV 0x20.
    pub previous_names: Vec<RenameRecord>,
    /// Mirror of the head-of-chain `wall_time_unix_ms` for cheap
    /// "last touched" reads. None for never-renamed entries.
    pub last_renamed_at_ms: Option<u64>,
    /// Field-type change audit chain. Empty for fields whose kind has
    /// never been migrated. Encoded on disk under TLV 0x30.
    pub type_change_history: Vec<TypeChangeRecord>,
    /// Mirror of the head-of-chain `wall_time_unix_ms` for the type
    /// change chain. None when `type_change_history` is empty.
    pub last_type_change_at_ms: Option<u64>,
    pub unknown_tlvs: Vec<(u8, Bytes)>,
}

impl IdEntry {
    fn fresh(id: u64, kind: u8, now_ms: u64) -> Self {
        Self {
            id,
            assigned_at: now_ms,
            assigned_by: ASSIGNED_BY_FRESH,
            kind,
            status: TombstoneStatus::Live,
            retired_at_ms: None,
            retired_reason: None,
            previous_names: Vec::new(),
            last_renamed_at_ms: None,
            type_change_history: Vec::new(),
            last_type_change_at_ms: None,
            unknown_tlvs: Vec::new(),
        }
    }

    fn backfilled(id: u64, kind: u8, now_ms: u64) -> Self {
        Self {
            id,
            assigned_at: now_ms,
            assigned_by: ASSIGNED_BY_BACKFILL,
            kind,
            status: TombstoneStatus::Live,
            retired_at_ms: None,
            retired_reason: None,
            previous_names: Vec::new(),
            last_renamed_at_ms: None,
            type_change_history: Vec::new(),
            last_type_change_at_ms: None,
            unknown_tlvs: Vec::new(),
        }
    }

    /// Mark a live entry as tombstoned. Caller is responsible for
    /// re-encoding and committing the row.
    fn tombstone(&mut self, now_ms: u64, reason: RetireReason) {
        self.status = TombstoneStatus::Tombstoned;
        self.retired_at_ms = Some(now_ms);
        self.retired_reason = Some(reason);
    }
}

/// Retirement metadata for a single tombstoned catalog entry. Reserved
/// for the operator-facing `Database::last_retirement_report()` API
/// which will land in a follow-on card; today every retired entry is
/// also recorded on its `IdEntry` via the on-disk TLVs.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RetiredEntry {
    pub id: u64,
    pub name: String,
    pub reason: RetireReason,
    pub retired_at_ms: u64,
}

/// Result of a single shrink-allowed reconcile commit. See `RetiredEntry`.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RetirementReport {
    pub retired_types: Vec<RetiredEntry>,
    pub retired_fields: Vec<RetiredEntry>,
    pub retired_relations: Vec<RetiredEntry>,
    pub catalog_format_before: u64,
    pub catalog_format_after: u64,
}

impl RetirementReport {
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.retired_types.is_empty()
            && self.retired_fields.is_empty()
            && self.retired_relations.is_empty()
    }
}

/// The in-memory catalog the engine consults during open(). The
/// `*_ids` projections include every ID ever allocated (live or
/// tombstoned); the parallel `tombstoned_*` sets carry only the retired
/// subset. Hot read paths check the negative case (id is NOT in the
/// tombstoned set → live → proceed) so the common case is one
/// `HashSet::contains` against an empty/tiny set.
#[derive(Debug, Default)]
pub(crate) struct Catalog {
    pub type_ids: HashMap<String, u64>,
    pub field_ids: HashMap<String, u64>,
    pub rel_ids: HashMap<String, u64>,
    pub field_kinds: HashMap<String, u8>,

    pub type_entries: HashMap<String, IdEntry>,
    pub field_entries: HashMap<String, IdEntry>,
    pub rel_entries: HashMap<String, IdEntry>,

    /// Tombstoned numeric IDs, partitioned by row kind. Populated by
    /// `load_existing` from on-disk status TLVs and by reconcile when
    /// it tombstones a row in the current commit. Retired IDs ALSO
    /// remain in `type_ids` / `field_ids` / `rel_ids` so the
    /// duplicate-ID scan still sees them and counter self-heal still
    /// respects them — these sets are "the subset of the ids maps that
    /// is retired."
    pub tombstoned_type_ids: HashSet<u64>,
    pub tombstoned_field_ids: HashSet<u64>,
    pub tombstoned_rel_ids: HashSet<u64>,

    /// Tombstoned names, partitioned the same way. Used by reconcile's
    /// name-reuse refusal (refusing to allocate a fresh ID for a name
    /// that's already retired — preserves the invariant that a retired
    /// name's data can never be silently re-bound).
    pub tombstoned_type_names: HashSet<String>,
    pub tombstoned_field_quals: HashSet<String>,
    pub tombstoned_rel_quals: HashSet<String>,

    /// The on-disk catalog format version. `1` for legacy / fresh
    /// catalogs that never shrank; `2` once any tombstone has been
    /// written.
    pub format_version: u64,

    pub next_type: u64,
    pub next_field: u64,
    pub next_rel: u64,
}

impl Catalog {
    fn empty() -> Self {
        Self {
            next_type: 1,
            next_field: 1,
            next_rel: 1,
            format_version: CATALOG_FORMAT_V1,
            ..Self::default()
        }
    }

    fn insert_type(&mut self, name: String, entry: IdEntry) {
        self.type_ids.insert(name.clone(), entry.id);
        if entry.status == TombstoneStatus::Tombstoned {
            self.tombstoned_type_ids.insert(entry.id);
            self.tombstoned_type_names.insert(name.clone());
        }
        self.type_entries.insert(name, entry);
    }

    fn insert_field(&mut self, qual: String, entry: IdEntry) {
        self.field_ids.insert(qual.clone(), entry.id);
        self.field_kinds.insert(qual.clone(), entry.kind);
        if entry.status == TombstoneStatus::Tombstoned {
            self.tombstoned_field_ids.insert(entry.id);
            self.tombstoned_field_quals.insert(qual.clone());
        }
        self.field_entries.insert(qual, entry);
    }

    fn insert_rel(&mut self, qual: String, entry: IdEntry) {
        self.rel_ids.insert(qual.clone(), entry.id);
        if entry.status == TombstoneStatus::Tombstoned {
            self.tombstoned_rel_ids.insert(entry.id);
            self.tombstoned_rel_quals.insert(qual.clone());
        }
        self.rel_entries.insert(qual, entry);
    }

    /// Tombstone an existing live entry. Updates in-memory state to
    /// match what the caller is about to commit to disk. Idempotent:
    /// re-tombstoning an already-tombstoned entry is a no-op.
    ///
    /// Currently unused — `reconcile_into_txn` mutates `IdEntry` and
    /// the tombstoned sets directly because it also encodes and stages
    /// the row in the same loop. Kept for follow-on cards (rename API,
    /// field-type-change migration) that may tombstone an entry from a
    /// different code path.
    #[allow(dead_code)]
    fn mark_type_tombstoned(&mut self, name: &str, now_ms: u64, reason: RetireReason) {
        if let Some(entry) = self.type_entries.get_mut(name) {
            if entry.status == TombstoneStatus::Tombstoned {
                return;
            }
            entry.tombstone(now_ms, reason);
            self.tombstoned_type_ids.insert(entry.id);
            self.tombstoned_type_names.insert(name.to_string());
        }
    }

    #[allow(dead_code)]
    fn mark_field_tombstoned(&mut self, qual: &str, now_ms: u64, reason: RetireReason) {
        if let Some(entry) = self.field_entries.get_mut(qual) {
            if entry.status == TombstoneStatus::Tombstoned {
                return;
            }
            entry.tombstone(now_ms, reason);
            self.tombstoned_field_ids.insert(entry.id);
            self.tombstoned_field_quals.insert(qual.to_string());
        }
    }

    #[allow(dead_code)]
    fn mark_rel_tombstoned(&mut self, qual: &str, now_ms: u64, reason: RetireReason) {
        if let Some(entry) = self.rel_entries.get_mut(qual) {
            if entry.status == TombstoneStatus::Tombstoned {
                return;
            }
            entry.tombstone(now_ms, reason);
            self.tombstoned_rel_ids.insert(entry.id);
            self.tombstoned_rel_quals.insert(qual.to_string());
        }
    }
}

// =====================================================================
// PUBLIC ENTRY POINT
// =====================================================================

/// Open-time hook. Returns the [`Catalog`] the engine should use.
/// Internally chooses between three paths:
///
/// 1. **Fresh / legacy DB** (no `c:F:`, no `c:I:`): run deterministic
///    backfill from the schema's alphabetical/declaration order, write
///    every catalog row in one atomic commit. For a legacy DB with
///    existing `o:` data the IDs come out byte-equal to what the
///    pre-catalog allocator produced; data round-trips untouched.
///
/// 2. **Partial catalog** (`c:F:` xor `c:I:`): a torn WAL during a
///    previous backfill left half-state. Clear every `c:*` row in one
///    txn and re-run backfill. Idempotent because backfill is
///    deterministic on schema.
///
/// 3. **Initialized catalog** (`c:F:` and `c:I:` both present): load
///    the existing rows, validate counters, fast-path skip if the
///    schema digest matches, otherwise reconcile (detect drops →
///    detect kind changes → additive allocation, write a single
///    atomic commit at the end).
pub(crate) fn load_or_initialize(
    storage: &LsmTree,
    schema: &Schema,
    allow_schema_shrink: bool,
) -> EngineResult<Catalog> {
    // Validate identifier safety once up front — catches phase-1
    // tampering or unusual schema-builder consumers that bypass the
    // parser's checks. The encoder also has debug_assert!s but those
    // are silent in release.
    for (type_name, type_def) in &schema.types {
        check_identifier(type_name)?;
        for field in &type_def.fields {
            check_identifier(&field.name)?;
        }
    }

    // Hold the process-wide catalog lock for the entire attempt-and-
    // retry loop. Concurrent threads in the same process see this lock
    // and serialise; a subsequent thread's open() will observe the
    // earlier thread's writes when it enters.
    let _guard = CATALOG_INIT_LOCK.lock();

    for attempt in 0..COMMIT_RETRY_BUDGET {
        match load_or_initialize_attempt(storage, schema, allow_schema_shrink) {
            Ok(cat) => return Ok(cat),
            Err(EngineError::Storage(rhypedb_storage::Error::WriteConflict)) => {
                // With the process-wide lock held, write-conflict from
                // concurrent threads in this process is impossible.
                // Any conflict here is a multi-process race against
                // the storage dir — out of scope for the catalog and
                // also out of scope for the storage layer (which
                // doesn't lock the data dir either). Surface it after
                // the bounded retry budget so the caller can decide
                // whether to back off.
                if attempt + 1 == COMMIT_RETRY_BUDGET {
                    return Err(EngineError::Catalog(CatalogError::ConcurrentInit));
                }
                continue;
            }
            Err(other) => return Err(other),
        }
    }
    Err(EngineError::Catalog(CatalogError::ConcurrentInit))
}

fn load_or_initialize_attempt(
    storage: &LsmTree,
    schema: &Schema,
    allow_schema_shrink: bool,
) -> EngineResult<Catalog> {
    // Begin the txn before any catalog read. The txn's snapshot
    // anchors the MVCC conflict check to the point our DECISIONS were
    // made; if another opener commits between our reads and our
    // commit, their `commit_version > txn.snapshot`, our overlapping
    // write keys (counters and digest) trip the conflict, and the
    // outer retry loop re-reads and tries again. Taking the snapshot
    // any later (e.g. only at begin_txn() right before commit) lets a
    // concurrent commit slip through with the same snapshot value and
    // produce duplicate IDs — the exact failure mode adversary A0
    // flagged in the design review.
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();

    let result = (|| -> EngineResult<(Catalog, bool)> {
        let format_present = storage.get(&txn, &KeyBuilder::catalog_format())?;
        let initialized = storage.get(&txn, &KeyBuilder::catalog_initialized())?;

        match (format_present.as_ref(), initialized.as_ref()) {
            (None, None) => {
                // Fresh DB or pre-catalog legacy DB. Both routes
                // converge on backfill — for a fresh DB it's the first
                // commit; for a legacy DB it produces IDs byte-equal
                // to what the pre-catalog algorithm computed, so
                // existing `o:`/`r:`/`i:`/etc. keys still decode.
                let cat = backfill_into_txn(storage, schema, &mut txn)?;
                Ok((cat, true))
            }
            (Some(_), Some(_)) => {
                let v = decode_format_version("c:F:", format_present.as_ref().unwrap())?;
                if v > CATALOG_FORMAT_CURRENT {
                    return Err(EngineError::Catalog(CatalogError::UnsupportedFormat {
                        got: v,
                        max_supported: CATALOG_FORMAT_CURRENT,
                    }));
                }
                let mut cat = load_existing(storage, snap, v)?;

                let stored_digest = storage
                    .get(&txn, &KeyBuilder::catalog_digest())?
                    .ok_or_else(|| {
                        EngineError::Catalog(CatalogError::MissingRequiredKey {
                            key_debug: "c:D:".into(),
                        })
                    })?;
                let want_digest = compute_schema_digest(schema);
                if stored_digest.as_ref() == &want_digest[..] {
                    // Fast path: no writes, abort the txn.
                    return Ok((cat, false));
                }
                reconcile_into_txn(
                    storage,
                    schema,
                    &mut cat,
                    allow_schema_shrink,
                    &want_digest,
                    &mut txn,
                )?;
                Ok((cat, true))
            }
            _ => {
                let cat = recover_partial_into_txn(storage, schema, &mut txn, snap)?;
                Ok((cat, true))
            }
        }
    })();

    match result {
        Ok((cat, wrote)) => {
            if wrote {
                storage.commit(&mut txn)?;
            } else {
                storage.abort(&mut txn);
            }
            Ok(cat)
        }
        Err(e) => {
            storage.abort(&mut txn);
            Err(e)
        }
    }
}

// =====================================================================
// RENAME — type rename verb (card 3/5)
//
// Scope: card 3/5 ships rename_type only. rename_field is deferred to
// a follow-on card that also addresses zone-map name-hash keying —
// renaming a field today would orphan the zone-map columns embedded
// in existing SSTs (which are hashed by field-name string), producing
// silent empty results from queries that should match. Type rename
// has no such problem: field names (and therefore zone-map keys, cover
// blob keys, and index hashes) stay unchanged.
// =====================================================================

/// One rename verb in a migration plan. Card 3/5 phase 1 shipped only
/// the `Type` variant (rename_field was deferred because SST v4 zone-map
/// columns were keyed by FNV(field_name) — a field rename would have
/// silently corrupted zone pruning). With SST v5 keying zone columns by
/// the stable catalog field_id, `Field` is now safe to land.
#[derive(Debug, Clone)]
pub(crate) enum RenameVerb {
    Type { old: String, new: String },
    Field {
        type_name: String,
        old: String,
        new: String,
    },
}

/// A field-type change verb. Card 4/5 phase 1 only supports plain
/// scalar fields (no `@indexed`, `@unique`, `@vectorize`, no relation
/// kinds). Each of those carries follow-on work the synthesis spelled
/// out in detail.
pub(crate) struct FieldTypeChangeVerb {
    pub type_name: String,
    pub field_name: String,
    pub target_kind: u8,
    /// User-supplied per-row converter. `Fn(object_id, old_value)`.
    pub converter: RowConverter,
}

/// Per-row value converter for a field-type change:
/// `Fn(object_id, old_value) -> new_value`.
pub(crate) type RowConverter =
    Box<dyn Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value> + Send + Sync>;

/// Shared, resumable converter held in the per-`Database` registry
/// (shadow-field card 1). `Arc` (not `Box`) so the same converter resolves
/// at create AND at resume after restart, and is carried across the
/// `_consuming` rebuild. Resolved by `(name, version)` pinned in the plan.
pub(crate) type RegisteredConverter = std::sync::Arc<
    dyn Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value> + Send + Sync,
>;

/// One entry in the report a successful migrate returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenamePair {
    pub from: String,
    pub to: String,
    pub id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    pub renamed_types: Vec<RenamePair>,
    pub renamed_fields: Vec<FieldRenamePair>,
    pub field_type_changes: Vec<FieldTypeChangePair>,
    pub catalog_format_before: u64,
    pub catalog_format_after: u64,
}

/// Lifecycle status of a chunked field-type migration plan (`c:P:<id>`).
///
/// While `quiesces()` is true the plan is unsettled: the card-2 double-write
/// hook stays armed (writes to the migrating field still need a shadow stamped,
/// or fail closed if the converter is unresolved). `is_terminal` means the
/// worker is not running and will not auto-resume on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStatus {
    /// Committed; worker not yet started (or not yet observed). Hook armed.
    Pending,
    /// Worker is backfilling shadows (or cutting over). Hook armed.
    Running,
    /// All rows converted + cut over, catalog kind flipped, hook disarmed. Terminal.
    Completed,
    /// Operator-cancelled. Hook disarmed; field left partially
    /// converted (read-tolerant). Terminal.
    Cancelled,
    /// Worker hit an unrecoverable error. Hook stays armed (writes to the field
    /// keep failing closed) for inspection. Terminal (no auto-resume).
    Failed,
    /// Resume found the pinned converter missing or version-changed.
    /// Parked until re-registered or cancelled. Hook armed; resumable.
    AwaitingConverter,
    /// Card 4: a `dry_run` migration finished. Settled + terminal, but the
    /// catalog kind was NOT flipped and no `o:`/`c:Q:` writes happened — distinct
    /// from `Completed` so no reader mistakes a preflight for a real migration.
    /// Non-quiescing (the hook was never armed for a dry-run, so nothing to keep
    /// armed); the record is kept only for observability of the estimate.
    DryRunCompleted,
}

#[allow(dead_code)] // is_terminal wired into auto-resume (increment 4)
impl MigrationStatus {
    fn to_byte(self) -> u8 {
        match self {
            MigrationStatus::Pending => MP_STATUS_PENDING,
            MigrationStatus::Running => MP_STATUS_RUNNING,
            MigrationStatus::Completed => MP_STATUS_COMPLETED,
            MigrationStatus::Cancelled => MP_STATUS_CANCELLED,
            MigrationStatus::Failed => MP_STATUS_FAILED,
            MigrationStatus::AwaitingConverter => MP_STATUS_AWAITING_CONVERTER,
            MigrationStatus::DryRunCompleted => MP_STATUS_DRY_RUN_COMPLETED,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            MP_STATUS_PENDING => MigrationStatus::Pending,
            MP_STATUS_RUNNING => MigrationStatus::Running,
            MP_STATUS_COMPLETED => MigrationStatus::Completed,
            MP_STATUS_CANCELLED => MigrationStatus::Cancelled,
            MP_STATUS_FAILED => MigrationStatus::Failed,
            MP_STATUS_AWAITING_CONVERTER => MigrationStatus::AwaitingConverter,
            MP_STATUS_DRY_RUN_COMPLETED => MigrationStatus::DryRunCompleted,
            _ => return None,
        })
    }

    /// Worker is not running and will not auto-resume on its own.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            MigrationStatus::Completed
                | MigrationStatus::Cancelled
                | MigrationStatus::Failed
                | MigrationStatus::DryRunCompleted
        )
    }

    /// True while the plan is unsettled — the double-write hook must stay armed.
    /// `Completed`/`Cancelled`/`DryRunCompleted` settle it; `Failed` stays
    /// unsettled so a half-converted field keeps failing writes closed before
    /// inspection. (Name kept for back-compat; card 2 replaced type-wide quiesce
    /// with the field-scoped double-write hook.)
    pub fn quiesces(self) -> bool {
        !matches!(
            self,
            MigrationStatus::Completed
                | MigrationStatus::Cancelled
                | MigrationStatus::DryRunCompleted
        )
    }

    /// Eligible for the worker to (re)drive: `Pending` or `Running`.
    pub fn is_drivable(self) -> bool {
        matches!(self, MigrationStatus::Pending | MigrationStatus::Running)
    }
}

/// Per-migration policy for per-row CONVERTER failures (card 4/5). Governs ONLY
/// `FieldTypeChangeConverterFailed`; a structural `MigrationRowUnexpectedKind`
/// (on-disk vs catalog kind disagreement) or `FieldTypeChangeConverterReturnedWrongKind`
/// (converter-contract violation) ALWAYS halts regardless of policy. Immutable
/// after create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorPolicy {
    /// First converter failure halts all partitions; no cutover. (Default —
    /// matches card-1/2/3 single-commit semantics.)
    Stop,
    /// Converter failure is counted, the row is left unchanged, the migration
    /// continues. Cutover may proceed (errored rows stay source-shape).
    SkipAndLog,
    /// Converter failure is recorded to `c:Q:` (source preserved) and the
    /// migration continues. Cutover refuses until every quarantine row is
    /// resolved (retried) or cleared.
    Quarantine,
}

impl ErrorPolicy {
    fn to_byte(self) -> u8 {
        match self {
            ErrorPolicy::Stop => MP_ERROR_POLICY_STOP,
            ErrorPolicy::SkipAndLog => MP_ERROR_POLICY_SKIP_AND_LOG,
            ErrorPolicy::Quarantine => MP_ERROR_POLICY_QUARANTINE,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            MP_ERROR_POLICY_STOP => ErrorPolicy::Stop,
            MP_ERROR_POLICY_SKIP_AND_LOG => ErrorPolicy::SkipAndLog,
            MP_ERROR_POLICY_QUARANTINE => ErrorPolicy::Quarantine,
            _ => return None,
        })
    }
}

/// A persisted chunked field-type migration (`c:P:<plan_id>`), the card-2
/// state machine's durable record. Survives restart; auto-resume rebuilds
/// the worker + the double-write hook from these rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MigrationPlan {
    pub plan_id: u64,
    pub type_name: String,
    pub field_name: String,
    /// Stable catalog field id, pinned at create so a concurrent rename
    /// can't misdirect the migration.
    pub field_id: u64,
    pub src_kind: u8,
    pub target_kind: u8,
    pub status: MigrationStatus,
    /// Highest `object_id` whose conversion is durably committed. Resume
    /// re-scans strictly after this id; `0` = not started. Because a torn
    /// chunk write can leave converted blobs with the cursor un-advanced,
    /// re-scan from here relies on per-row idempotency-on-target.
    pub cursor: u64,
    pub chunk_size: u64,
    pub created_at_ms: u64,
    /// Name the converter is resolved by from the per-`Database` registry
    /// at create AND resume. `version` is pinned so a same-name converter
    /// with changed semantics parks `AwaitingConverter` instead of running.
    pub converter_name: String,
    pub converter_version: u32,
    /// Observability only — completion is proven by a positive exhaustion
    /// scan, never by this counter (a torn re-scan can double-count).
    pub objects_converted: u64,
    /// Card 2: which phase of the online migration this plan is in.
    /// `Converting` = the worker is backfilling `<field>__shadow` siblings (or
    /// done backfilling, awaiting cutover). `CuttingOver` = the cutover pass is
    /// renaming `<field>__shadow` → `<field>` per object. A crash mid-cutover
    /// re-resumes the rename pass, NOT the converter (they are not
    /// interchangeable). Card-1 rows (no TLV) decode as `Converting`.
    pub phase: MigrationPhase,
    /// Card 2: highest `object_id` whose CUTOVER rename is durably committed —
    /// a cursor dedicated to the cutover pass, distinct from `cursor` (the
    /// conversion scan). `0` = cutover not started. Card-1 rows decode as `0`.
    pub cutover_cursor: u64,
    /// Card 3: number of parallel backfill partitions (`1..=64`). `Some(n)` marks
    /// a parallel plan whose Converting-phase cursors live in `c:S:<plan><idx>`
    /// keys (the legacy `cursor` field is unused, left 0). `None` = a legacy
    /// card-1/2 single-worker plan whose cursor is the `cursor` field. PINNED at
    /// create; resume recomputes identical partition boundaries from it, so the
    /// operator can't change it mid-migration.
    pub parallel_degree: Option<u8>,
    /// Card 3: exclusive upper bound on pre-existing object ids — the snapshot of
    /// `next_object_id` taken (under `migration_lock.write()`, after the hook is
    /// armed) at create. The backfill partitions cover `[1, id_upper_bound)`;
    /// objects created during the migration get ids `>= id_upper_bound` and are
    /// born with the shadow via the double-write hook, so no worker touches them.
    /// `0` on a legacy plan (no partitions).
    pub id_upper_bound: u64,
    /// Card 4: per-row converter-failure policy. PINNED at create (immutable —
    /// the SkipAndLog cutover-skip soundness depends on it). Card-1/2/3 rows
    /// decode as `Stop`.
    pub error_policy: ErrorPolicy,
    /// Card 4: a preflight that runs the converter over every row, counting
    /// `objects_converted`/`error_count`, but writes NOTHING to `o:`/`c:Q:` and
    /// never cuts over (no hook armed). Card-1/2/3 rows decode as `false`.
    pub dry_run: bool,
    /// Card 4: number of rows whose converter failed (summed from the durable
    /// per-partition `c:S:` `errors` at finalize — re-scan-proof like
    /// `objects_converted`, NOT a free-running counter). Observability + the
    /// cutover gate's coarse signal. Card-1/2/3 rows decode as `0`.
    pub error_count: u64,
    /// Card 4: cap on quarantined/errored rows; exceeding it parks the migration
    /// `Failed` (tripwire vs a runaway error field). `0` → `DEFAULT_QUARANTINE_CAP`.
    pub quarantine_cap: u64,
    /// Forward-compat: TLV tags this binary doesn't recognise, preserved
    /// verbatim so a card-2/4 row round-trips through a card-1 binary.
    pub unknown_tlvs: Vec<(u8, Bytes)>,
}

/// Phase of an online (card-2) chunked field-type migration. Persisted in the
/// `c:P:` plan record so a crash resumes the correct pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPhase {
    /// Backfilling `<field>__shadow` siblings (or done, awaiting cutover).
    Converting,
    /// Renaming `<field>__shadow` → `<field>` per object (cutover in progress).
    CuttingOver,
    /// Card 5: a terminal cancel is in progress — stripping every
    /// `<field>__shadow` sibling back off the `o:` blobs (the source field is
    /// untouched, so the migration rolls back losslessly). The hook stays armed
    /// until the strip completes, then the plan settles `Cancelled`.
    RollingBack,
}

impl MigrationPhase {
    fn to_byte(self) -> u8 {
        match self {
            MigrationPhase::Converting => MP_PHASE_CONVERTING,
            MigrationPhase::CuttingOver => MP_PHASE_CUTTING_OVER,
            MigrationPhase::RollingBack => MP_PHASE_ROLLING_BACK,
        }
    }
    fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            MP_PHASE_CONVERTING => MigrationPhase::Converting,
            MP_PHASE_CUTTING_OVER => MigrationPhase::CuttingOver,
            MP_PHASE_ROLLING_BACK => MigrationPhase::RollingBack,
            _ => return None,
        })
    }
}

/// One field rename, included in the report. `type_name` is the (possibly
/// already-renamed) parent type the field hangs off of. `field_id` is the
/// stable catalog id, preserved by the rename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldRenamePair {
    pub type_name: String,
    pub from: String,
    pub to: String,
    pub field_id: u64,
    /// PER-VERB count of object FieldMaps rewritten. In a multi-verb plan that
    /// chains renames of one type, the SAME physical object is counted once per
    /// verb that touches it — this is an operation count, not a distinct-object
    /// count (summing across a chain's pairs over-counts). Same for
    /// `covers_rewritten`.
    pub objects_rewritten: u64,
    /// Count of `r:*` reverse-edge cover blobs whose embedded source-side
    /// FieldMap was rewritten in the same atomic batch as the catalog row
    /// and object rewrites. Each rev_edge from an object of this type via a
    /// forward 1:1 or 1:N relation gets rewritten exactly once.
    pub covers_rewritten: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldTypeChangePair {
    pub type_name: String,
    pub field_name: String,
    pub from_kind: u8,
    pub to_kind: u8,
    pub field_id: u64,
    pub objects_converted: u64,
}

/// Apply a migration plan against the current catalog state. Holds
/// `CATALOG_INIT_LOCK` for the entire call so a concurrent reconcile
/// or other migrate cannot race. Commits exactly one atomic LSM batch
/// per `apply` call. Returns a structured report or a typed error.
/// Cover-less convenience wrapper used by catalog-only tests (no `Database`
/// maintainer in hand). Production callers go through `apply_migration_with_cover`
/// so a field rename can refresh `@indexed` covers.
#[cfg(test)]
pub(crate) fn apply_migration(
    storage: &LsmTree,
    schema: &Schema,
    verbs: &[RenameVerb],
) -> EngineResult<MigrationReport> {
    apply_migration_with_cover(storage, schema, verbs, None)
}

/// `apply_migration` plus a [`FieldCoverMaintainer`] so a field rename can
/// refresh the renamed type's `@indexed` covering payloads in the same atomic
/// batch (the covers embed field names, which change under a rename). Callers
/// that hold a `Database` (or a `MigrationContext` carrying one) thread it in;
/// catalog-only callers pass `None`.
/// In-memory view of the writes a multi-verb rename plan has buffered SO FAR,
/// layered over the plan's storage snapshot. Storage has NO read-your-own-writes
/// (a `Transaction`'s puts/deletes only reach the memtable at commit), so within
/// `apply_migration_with_cover` a later verb reading at `snap` cannot otherwise
/// see an earlier verb's object/cover rewrites — committing both would leave the
/// type's objects/covers half-renamed. This overlay closes that gap: it is folded
/// forward one verb at a time (`absorb`) and consulted at every storage read site
/// (`get_at` / `scan_prefix_at`).
///
/// It is ALSO the authoritative net write-set for the commit (`net_sets`): folding
/// verb-ordered last-write-wins resolves every key to a single `Some(put)` /
/// `None(delete)`, so a key never lands in BOTH the put and the delete batch.
/// That matters because storage applies all puts THEN all deletes — for a chain
/// `A→B→C` the intermediate `c:E:T\0B` is put by verb 1 then deleted by verb 2
/// (net = delete, correct), while a name-reuse `[A→B, X→A]` re-PUTs `c:E:T\0A`
/// after deleting it (net = put, correct); the raw put-then-delete batches would
/// mis-resolve the reuse case and drop the field.
///
/// Only built for multi-verb plans (`verbs.len() > 1`); single-verb plans leave it
/// empty and never call `net_sets`, so the common production path is byte-for-byte
/// unchanged.
#[derive(Default)]
struct WriteOverlay {
    /// Net state per key: `Some(value)` = put, `None` = delete. A `BTreeMap`
    /// (not a hash map) so a prefix scan is a `range`, not a full-map walk.
    map: std::collections::BTreeMap<Bytes, Option<Bytes>>,
}

impl WriteOverlay {
    fn new() -> Self {
        Self::default()
    }

    /// Fold one verb's freshly-appended writes into the net state, in order
    /// (later wins: a delete after a put tombstones the key; a put after a delete
    /// revives it — matching the sequential composition of the verbs).
    fn absorb(&mut self, puts: &[(Bytes, Bytes)], deletes: &[Bytes]) {
        for (k, v) in puts {
            self.map.insert(k.clone(), Some(v.clone()));
        }
        for k in deletes {
            self.map.insert(k.clone(), None);
        }
    }

    /// Point read layered over `snap`: an overlay hit wins (`Some` = value,
    /// `None` = deleted); otherwise fall through to the snapshot.
    fn get_at(&self, storage: &LsmTree, snap: u64, key: &[u8]) -> EngineResult<Option<Bytes>> {
        match self.map.get(key) {
            Some(Some(v)) => Ok(Some(v.clone())),
            Some(None) => Ok(None),
            None => Ok(storage.get_at(snap, key)?),
        }
    }

    /// Prefix scan layered over `snap`: take the snapshot scan, then apply every
    /// overlay entry under `prefix` (`Some` overrides/adds, `None` removes).
    ///
    /// The `None`-removal branch is currently UNREACHABLE through the rename verbs
    /// (the only scanned prefixes are `o:`/`e:`, both put-only; catalog `c:` rows
    /// are the only deletes and are read from the in-memory `Catalog`, never
    /// scanned through storage). It is implemented for generality and pinned by a
    /// unit test so it can't silently rot.
    fn scan_prefix_at(
        &self,
        storage: &LsmTree,
        snap: u64,
        prefix: &[u8],
    ) -> EngineResult<Vec<(Bytes, Bytes)>> {
        let base = storage.scan_prefix_at(snap, prefix)?;
        let lower = Bytes::copy_from_slice(prefix);
        // Fast path: no buffered write under this prefix → return the snapshot
        // scan verbatim (avoids rebuilding a BTreeMap for the per-edge `e:` scans,
        // which never have overlay entries — keeps the scan from going quadratic).
        let any_overlay = self
            .map
            .range(lower.clone()..)
            .next()
            .is_some_and(|(k, _)| k.starts_with(prefix));
        if !any_overlay {
            return Ok(base);
        }
        let mut merged: std::collections::BTreeMap<Bytes, Bytes> = base.into_iter().collect();
        for (k, v) in self.map.range(lower..) {
            if !k.starts_with(prefix) {
                break;
            }
            match v {
                Some(val) => {
                    merged.insert(k.clone(), val.clone());
                }
                None => {
                    merged.remove(k);
                }
            }
        }
        Ok(merged.into_iter().collect())
    }

    /// The net write-set for the commit: every key resolved to exactly ONE of a
    /// put (`Some`) or a delete (`None`), so the put and delete batches are
    /// disjoint (the storage commit's put-then-delete ordering becomes moot).
    fn net_sets(&self) -> (Vec<(Bytes, Bytes)>, Vec<Bytes>) {
        let mut puts = Vec::new();
        let mut deletes = Vec::new();
        for (k, v) in &self.map {
            match v {
                Some(val) => puts.push((k.clone(), val.clone())),
                None => deletes.push(k.clone()),
            }
        }
        (puts, deletes)
    }
}

pub(crate) fn apply_migration_with_cover(
    storage: &LsmTree,
    schema: &Schema,
    verbs: &[RenameVerb],
    cover: Option<&dyn FieldCoverMaintainer>,
) -> EngineResult<MigrationReport> {
    let _guard = CATALOG_INIT_LOCK.lock();

    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();

    let result = (|| -> EngineResult<(MigrationReport, bool)> {
        // Read current format. We support v1, v2, v3 catalogs as input.
        let fv_raw = storage
            .get(&txn, &KeyBuilder::catalog_format())?
            .ok_or_else(|| {
                EngineError::Catalog(CatalogError::MissingRequiredKey {
                    key_debug: "c:F:".into(),
                })
            })?;
        let format_before = decode_format_version("c:F:", &fv_raw)?;
        if format_before > CATALOG_FORMAT_CURRENT {
            return Err(EngineError::Catalog(CatalogError::UnsupportedFormat {
                got: format_before,
                max_supported: CATALOG_FORMAT_CURRENT,
            }));
        }
        let mut cat = load_existing(storage, snap, format_before)?;

        let mut report = MigrationReport {
            catalog_format_before: format_before,
            catalog_format_after: format_before,
            ..MigrationReport::default()
        };

        // Phase 3 lifted the former pre-flight directive refusal (renaming a field
        // carrying @indexed / @unique / @vectorize, or a relation field, is
        // supported). The multi-verb-same-type refusal it left behind is ALSO
        // lifted now (Overboard cmqgvlf6b): a single plan may chain renames of one
        // type (`A→B→C`) or rename several of its fields. Every verb still reads
        // from ONE storage snapshot (`snap`) with no read-your-own-writes, so a
        // later verb is handed a `WriteOverlay` (`overlay`) layering the prior
        // verbs' buffered object/cover rewrites over `snap`. For a multi-verb plan
        // the overlay's net state is also the commit write-set, so chained/reused
        // catalog keys resolve correctly (see `WriteOverlay`). Production still only
        // ever builds single-verb plans (each `Database` / `MigrationContext` rename
        // is its own commit), which skip the overlay entirely.
        let multi_verb = verbs.len() > 1;
        let mut overlay = WriteOverlay::new();

        // Narrow refusal retained: a TYPE rename combined with a FIELD rename of
        // that same type (either order). Field-only multi-verb plans are the lifted
        // case; this one stays refused because a field verb resolves its @indexed
        // cover maintainer (the live handle) by type NAME, which a same-plan type
        // rename makes stale — the index covering payload would silently not
        // refresh. Match the field's type against BOTH the old and new names of
        // every type verb. Out of scope for cmqgvlf6b (field renames); split it.
        if multi_verb {
            let mut renamed_type_names: std::collections::HashSet<&str> =
                std::collections::HashSet::new();
            for verb in verbs {
                if let RenameVerb::Type { old, new } = verb {
                    renamed_type_names.insert(old.as_str());
                    renamed_type_names.insert(new.as_str());
                }
            }
            if !renamed_type_names.is_empty() {
                for verb in verbs {
                    if let RenameVerb::Field { type_name, .. } = verb
                        && renamed_type_names.contains(type_name.as_str())
                    {
                        return Err(EngineError::Catalog(
                            CatalogError::RenameTypeWithFieldSamePlan {
                                type_name: type_name.clone(),
                            },
                        ));
                    }
                }
            }
        }

        // Refuse renaming a type or field that has an UNSETTLED chunked
        // field-type migration plan. A rename re-keys the catalog entry by
        // name (preserving the numeric id), but the plan and its name-keyed
        // lookups (`finalize_migration_cutover`, `active_plan_for_type`)
        // still reference the OLD name — the cutover would silently never
        // land and a later reopen would hard-error `FieldKindChanged`.
        // Symmetric to the offline `change_field_type` interlock.
        let plans = scan_migration_plans(storage, snap)?;
        for verb in verbs {
            let blocked = match verb {
                RenameVerb::Type { old, .. } => plans
                    .iter()
                    .find(|p| p.status.quiesces() && &p.type_name == old)
                    .map(|p| (old.clone(), p.plan_id)),
                RenameVerb::Field {
                    type_name, old, ..
                } => plans
                    .iter()
                    .find(|p| {
                        p.status.quiesces() && &p.type_name == type_name && &p.field_name == old
                    })
                    .map(|p| (format!("{type_name}.{old}"), p.plan_id)),
            };
            if let Some((qualified, plan_id)) = blocked {
                return Err(EngineError::Catalog(
                    CatalogError::MigrationFieldHasActivePlan { qualified, plan_id },
                ));
            }
        }

        // Per-verb pre-flight + mutation. The whole plan succeeds or fails as a
        // unit before any storage write happens. Each verb's effect on the
        // IN-MEMORY catalog (`cat`) is visible to the next; its STORAGE rewrites
        // are buffered (not visible through `snap`) and made visible to later
        // verbs via the `overlay`, folded forward after each verb.
        let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
        let mut deletes: Vec<Bytes> = Vec::new();
        let now_ms = now_unix_millis();
        for verb in verbs {
            let puts_before = puts.len();
            let deletes_before = deletes.len();
            match verb {
                RenameVerb::Type { old, new } => {
                    apply_type_rename_verb(
                        &mut cat,
                        old,
                        new,
                        now_ms,
                        &mut puts,
                        &mut deletes,
                        &mut report,
                    )?;
                }
                RenameVerb::Field {
                    type_name,
                    old,
                    new,
                } => {
                    // Relation fields live in a separate catalog namespace
                    // (`rel_entries`) and require edge-cover rewrites instead of
                    // an object-payload rewrite. Route a LIVE relation to its own
                    // verb; everything else (scalar/vector, or a not-found field)
                    // goes through the scalar verb, which also surfaces the
                    // "source not found" error.
                    let old_qual = format!("{type_name}.{old}");
                    let is_live_relation = cat
                        .rel_entries
                        .get(&old_qual)
                        .map(|e| e.status != TombstoneStatus::Tombstoned)
                        .unwrap_or(false);
                    if is_live_relation {
                        apply_relation_rename_verb(
                            storage,
                            schema,
                            snap,
                            &overlay,
                            &mut cat,
                            type_name,
                            old,
                            new,
                            now_ms,
                            &mut puts,
                            &mut deletes,
                            &mut report,
                        )?;
                    } else {
                        apply_field_rename_verb(
                            storage,
                            schema,
                            snap,
                            &overlay,
                            &mut cat,
                            type_name,
                            old,
                            new,
                            now_ms,
                            &mut puts,
                            &mut deletes,
                            &mut report,
                            cover,
                        )?;
                    }
                }
            }
            // Fold this verb's buffered writes forward so the NEXT verb's reads —
            // and, for a multi-verb plan, the final commit set (`net_sets` below) —
            // observe them. Single-verb plans have no next verb and commit the raw
            // batches unchanged, so skip the fold.
            if multi_verb {
                overlay.absorb(&puts[puts_before..], &deletes[deletes_before..]);
            }
        }

        if report.renamed_types.is_empty() && report.renamed_fields.is_empty() {
            // No-op plan. Abort the txn; don't bump format or touch
            // digest.
            return Ok((report, false));
        }

        // For a multi-verb plan the overlay's net state IS the commit write-set:
        // it collapses per-key last-write-wins across verbs so the put and delete
        // batches are disjoint (no put-then-delete mis-ordering for a chained or
        // reused catalog key). MUST run before the catalog-format put below, which
        // is a unique key never produced by a verb and is appended onto the rebuilt
        // `puts`. Single-verb plans keep their raw batches (overlay untouched).
        if multi_verb {
            let (net_puts, net_deletes) = overlay.net_sets();
            puts = net_puts;
            deletes = net_deletes;
        }

        // Bump catalog format. A rename_type plan minimally bumps to v3
        // (the historical type-rename marker). A rename_field plan that
        // lands at all forces v5 — v4 readers can't safely resolve a
        // renamed field to its pre-rename name (the schema names are now
        // out of sync with the v4 binary's expectations).
        let mut target_format = cat.format_version.max(CATALOG_FORMAT_V3);
        if !report.renamed_fields.is_empty() {
            target_format = target_format.max(CATALOG_FORMAT_V5);
        }
        if cat.format_version < target_format {
            cat.format_version = target_format;
            puts.push((
                KeyBuilder::catalog_format(),
                encode_format_version(target_format),
            ));
        }
        report.catalog_format_after = cat.format_version;

        // Leave c:D: in place. The next open with the post-rename
        // schema computes a different digest (names contribute to the
        // hash), the digest comparison mismatches, reconcile runs, and
        // — because the in-memory catalog is already keyed by post-
        // rename names — reconcile finds no drops, no adds, no kind
        // changes, and rewrites the digest in its tail commit. Net
        // effect: one fast-path miss on the first post-rename open,
        // then steady fast-path forever after.
        storage.put_batch(&mut txn, &puts)?;
        if !deletes.is_empty() {
            storage.delete_batch(&mut txn, &deletes)?;
        }
        Ok((report, true))
    })();

    match result {
        Ok((report, wrote)) => {
            if wrote {
                storage.commit(&mut txn)?;
            } else {
                storage.abort(&mut txn);
            }
            Ok(report)
        }
        Err(e) => {
            storage.abort(&mut txn);
            Err(e)
        }
    }
}

fn apply_type_rename_verb(
    cat: &mut Catalog,
    old: &str,
    new: &str,
    now_ms: u64,
    puts: &mut Vec<(Bytes, Bytes)>,
    deletes: &mut Vec<Bytes>,
    report: &mut MigrationReport,
) -> EngineResult<()> {
    // No-op rejection.
    if old == new {
        return Err(EngineError::Catalog(CatalogError::RenameNoOp {
            kind: "type",
            name: old.into(),
        }));
    }
    // Identifier safety on the new name.
    check_identifier(new)?;

    // Source must exist and be live.
    let entry = cat
        .type_entries
        .get(old)
        .ok_or_else(|| EngineError::Catalog(CatalogError::RenameSourceNotFound {
            kind: "type",
            name: old.into(),
        }))?;
    if entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(CatalogError::RenameSourceRetired {
            kind: "type",
            name: old.into(),
            retired_id: entry.id,
            retired_at_ms: entry.retired_at_ms.unwrap_or(0),
        }));
    }

    // Target must not exist (live or tombstoned).
    if let Some(existing) = cat.type_entries.get(new) {
        if existing.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(
                CatalogError::RenameTargetIsRetired {
                    kind: "type",
                    name: new.into(),
                    retired_id: existing.id,
                },
            ));
        }
        return Err(EngineError::Catalog(CatalogError::RenameTargetCollision {
            kind: "type",
            name: new.into(),
            existing_id: existing.id,
        }));
    }
    if cat.tombstoned_type_names.contains(new) {
        // Tombstoned but absent from type_entries — would only happen
        // if the row was deleted out-of-band; treat as retired.
        return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
            kind: "type",
            name: new.into(),
            retired_id: 0,
        }));
    }

    // History-cap check.
    if entry.previous_names.len() + 1 > MAX_RENAME_HISTORY {
        return Err(EngineError::Catalog(
            CatalogError::RenameHistoryCapExceeded {
                kind: "type",
                name: old.into(),
                cap: MAX_RENAME_HISTORY,
            },
        ));
    }

    // Mutate: rewrite the type entry's key from `old` to `new`,
    // preserving every existing field (including unknown TLVs) and
    // appending one RenameRecord.
    let mut new_entry = entry.clone();
    let preserved_id = new_entry.id;
    new_entry.previous_names.insert(
        0,
        RenameRecord {
            from: old.into(),
            to: new.into(),
            wall_time_unix_ms: now_ms,
        },
    );
    new_entry.last_renamed_at_ms = Some(now_ms);

    // Stage catalog row mutation: put the new key, delete the old.
    puts.push((
        KeyBuilder::catalog_type(new),
        encode_id_entry(&new_entry),
    ));
    deletes.push(KeyBuilder::catalog_type(old));

    // Cascade: rewrite every c:E:<old>\x00field and c:R:<old>\x00field
    // row to use the new type name. The TLV body is preserved verbatim,
    // including tombstoned children's retirement metadata and any
    // unknown TLVs from a future binary.
    let mut child_field_quals: Vec<String> = Vec::new();
    for qual in cat.field_entries.keys() {
        let (t, _) = split_qualified(qual);
        if t == old {
            child_field_quals.push(qual.clone());
        }
    }
    for qual in &child_field_quals {
        let (_, f) = split_qualified(qual);
        let f_owned = f.to_string();
        if let Some(field_entry) = cat.field_entries.get(qual).cloned() {
            puts.push((
                KeyBuilder::catalog_field(new, &f_owned),
                encode_id_entry(&field_entry),
            ));
            deletes.push(KeyBuilder::catalog_field(old, &f_owned));
            // Re-key in-memory maps.
            let new_qual = format!("{}.{}", new, f_owned);
            if let Some(id) = cat.field_ids.remove(qual) {
                cat.field_ids.insert(new_qual.clone(), id);
            }
            if let Some(kind) = cat.field_kinds.remove(qual) {
                cat.field_kinds.insert(new_qual.clone(), kind);
            }
            cat.field_entries.remove(qual);
            cat.field_entries.insert(new_qual.clone(), field_entry);
            // Re-key tombstoned name set if applicable.
            if cat.tombstoned_field_quals.remove(qual) {
                cat.tombstoned_field_quals.insert(new_qual);
            }
        }
    }

    let mut child_rel_quals: Vec<String> = Vec::new();
    for qual in cat.rel_entries.keys() {
        let (t, _) = split_qualified(qual);
        if t == old {
            child_rel_quals.push(qual.clone());
        }
    }
    for qual in &child_rel_quals {
        let (_, f) = split_qualified(qual);
        let f_owned = f.to_string();
        if let Some(rel_entry) = cat.rel_entries.get(qual).cloned() {
            puts.push((
                KeyBuilder::catalog_rel(new, &f_owned),
                encode_id_entry(&rel_entry),
            ));
            deletes.push(KeyBuilder::catalog_rel(old, &f_owned));
            let new_qual = format!("{}.{}", new, f_owned);
            if let Some(id) = cat.rel_ids.remove(qual) {
                cat.rel_ids.insert(new_qual.clone(), id);
            }
            cat.rel_entries.remove(qual);
            cat.rel_entries.insert(new_qual.clone(), rel_entry);
            if cat.tombstoned_rel_quals.remove(qual) {
                cat.tombstoned_rel_quals.insert(new_qual);
            }
        }
    }

    // Re-key the type maps. Tombstoned name set is NOT touched: the
    // old name is now FREE to be reused (the type wasn't retired, it
    // was renamed). The new name is alive at the same numeric id.
    cat.type_ids.remove(old);
    cat.type_ids.insert(new.into(), preserved_id);
    cat.type_entries.remove(old);
    cat.type_entries.insert(new.into(), new_entry);

    report.renamed_types.push(RenamePair {
        from: old.into(),
        to: new.into(),
        id: preserved_id,
    });
    Ok(())
}

/// rename_field verb (card 3/5 phase 2). Mirrors `apply_type_rename_verb`
/// for catalog mutation, then additionally rewrites every object's
/// serialized FieldMap so the field appears under the new name.
///
/// **Scope of rewrite (all in one atomic batch):**
/// * Every `o:<type_id>:*` object — required for correctness because
///   FieldMap entries are keyed by field name natively (see
///   `crates/rhypedb-engine/src/object.rs:130 serialize_fields_into`).
/// * Every `r:<target>:<rel>:<source>` reverse-edge value whose source
///   is an object of this type — the embedded source-side FieldMap is
///   rewritten with the new field name. Without this the executor's
///   cover-fusion fast path returns Objects with the OLD field name
///   (the `cover_v` stamp matches because rename doesn't bump it, so
///   the existing staleness fall-through never fires). Bounded by
///   O(edges from source type via forward relations).
/// * Catalog row `c:E:<type>\x00<old>` → `c:E:<type>\x00<new>`,
///   preserving `field_id` and appending a `RenameRecord`.
///
/// **Atomicity:** the entire rewrite + catalog mutation lands in one
/// LSM batch under `CATALOG_INIT_LOCK`. A 1 M-object rename produces one
/// fsync — fine for planned maintenance, not for hot online use. The
/// resumable-cursor shadow-field design (Epic #3) is the right answer
/// for the latter.
///
/// **Directives (phase 3):** a field carrying `@indexed` / `@unique` /
/// `@vectorize` is supported. `u:`/`i:`/`v:`/`s:` keys are `field_id`-keyed
/// (stable), so only the `@indexed` covering VALUES (full FieldMaps embedding
/// names) are refreshed — via the `cover` maintainer, in the same batch.
/// `@unique` and `@vectorize` need no on-disk rewrite. Relation-field renames
/// go through `apply_relation_rename_verb` instead (separate catalog namespace).
#[allow(clippy::too_many_arguments)]
fn apply_field_rename_verb(
    storage: &LsmTree,
    _schema: &Schema,
    snap: u64,
    overlay: &WriteOverlay,
    cat: &mut Catalog,
    type_name: &str,
    old: &str,
    new: &str,
    now_ms: u64,
    puts: &mut Vec<(Bytes, Bytes)>,
    deletes: &mut Vec<Bytes>,
    report: &mut MigrationReport,
    cover: Option<&dyn FieldCoverMaintainer>,
) -> EngineResult<()> {
    // No-op rejection.
    if old == new {
        return Err(EngineError::Catalog(CatalogError::RenameNoOp {
            kind: "field",
            name: format!("{type_name}.{old}"),
        }));
    }
    // Identifier safety on the new name (reuses the same validator that
    // gates schema-side identifier names — rejects `\x00`, `:`, `__`,
    // empty, etc.).
    check_identifier(new)?;

    // Parent type must exist and be live.
    let type_entry = cat
        .type_entries
        .get(type_name)
        .ok_or_else(|| {
            EngineError::Catalog(CatalogError::RenameSourceNotFound {
                kind: "type",
                name: type_name.into(),
            })
        })?
        .clone();
    if type_entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(CatalogError::RenameSourceRetired {
            kind: "type",
            name: type_name.into(),
            retired_id: type_entry.id,
            retired_at_ms: type_entry.retired_at_ms.unwrap_or(0),
        }));
    }

    let old_qual = format!("{type_name}.{old}");
    let new_qual = format!("{type_name}.{new}");

    // Source field must exist and be live.
    let field_entry = cat
        .field_entries
        .get(&old_qual)
        .ok_or_else(|| EngineError::Catalog(CatalogError::RenameSourceNotFound {
            kind: "field",
            name: old_qual.clone(),
        }))?
        .clone();
    if field_entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(CatalogError::RenameSourceRetired {
            kind: "field",
            name: old_qual.clone(),
            retired_id: field_entry.id,
            retired_at_ms: field_entry.retired_at_ms.unwrap_or(0),
        }));
    }

    // Target field must not exist (live or tombstoned within this type).
    // We also check the rel_entries side so e.g. `User.movie` (relation
    // field) doesn't clash with a `User.movie` scalar field that gets
    // renamed in.
    if let Some(existing) = cat.field_entries.get(&new_qual) {
        if existing.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
                kind: "field",
                name: new_qual.clone(),
                retired_id: existing.id,
            }));
        }
        return Err(EngineError::Catalog(CatalogError::RenameTargetCollision {
            kind: "field",
            name: new_qual.clone(),
            existing_id: existing.id,
        }));
    }
    if let Some(existing) = cat.rel_entries.get(&new_qual) {
        if existing.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
                kind: "field",
                name: new_qual.clone(),
                retired_id: existing.id,
            }));
        }
        return Err(EngineError::Catalog(CatalogError::RenameTargetCollision {
            kind: "field",
            name: new_qual.clone(),
            existing_id: existing.id,
        }));
    }
    if cat.tombstoned_field_quals.contains(&new_qual) {
        return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
            kind: "field",
            name: new_qual.clone(),
            retired_id: 0,
        }));
    }

    // History-cap check.
    if field_entry.previous_names.len() + 1 > MAX_RENAME_HISTORY {
        return Err(EngineError::Catalog(
            CatalogError::RenameHistoryCapExceeded {
                kind: "field",
                name: old_qual.clone(),
                cap: MAX_RENAME_HISTORY,
            },
        ));
    }

    // This verb handles scalar/vector fields. Relation fields are routed to
    // `apply_relation_rename_verb` by the dispatch in `apply_migration_with_cover`
    // (they live in `rel_entries`, a different catalog namespace). `@indexed` /
    // `@unique` / `@vectorize` are supported here (phase 3): their keys are
    // field_id-keyed, so only the `@indexed` covers (refreshed below via `cover`)
    // need rewriting.

    // ---- Object FieldMap rewrite ----------------------------------
    // Every existing object of this type encodes the old field name in
    // its serialized FieldMap. Rewrite each one to use the new name
    // before catalog commit, all in the same atomic batch. Skipping
    // this would make reads of the renamed field return None for every
    // pre-rename object.
    let type_id = *cat.type_ids.get(type_name).ok_or_else(|| {
        EngineError::Catalog(CatalogError::RenameSourceNotFound {
            kind: "type",
            name: type_name.into(),
        })
    })?;

    // Map each of this type's fields to its CURRENT bare name keyed by STABLE
    // field_id, for the @indexed cover refresh below. `cat.field_ids` already
    // reflects EARLIER verbs' renames; THIS verb's rename (old→new) is applied as
    // an override because `cat` isn't re-keyed for it until after the object loop
    // (the rewritten object blob already carries `new`). A single old→new remap
    // can't resolve a field that an earlier verb in the same plan renamed, which
    // is why the maintainer takes this map instead.
    let mut current_field_name: std::collections::HashMap<u64, String> =
        std::collections::HashMap::new();
    for (qual, &fid) in &cat.field_ids {
        let (t, f) = split_qualified(qual);
        if t == type_name {
            current_field_name.insert(fid, f.to_string());
        }
    }
    current_field_name.insert(field_entry.id, new.to_string());

    let prefix = KeyBuilder::object_prefix(type_id);
    let entries = overlay.scan_prefix_at(storage, snap, &prefix)?;
    let mut objects_rewritten: u64 = 0;
    // Collect source object IDs for the rev_edge cover rewrite pass
    // below — every rev_edge whose source is one of these objects
    // carries a copy of the source's FieldMap and must be rewritten in
    // the same atomic batch.
    let mut source_object_ids: Vec<u64> = Vec::with_capacity(entries.len());
    for (key, data) in entries {
        // Object key tail is the u64 BE object_id.
        let object_id = if key.len() >= 8 {
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let id = u64::from_be_bytes(id_bytes);
            source_object_ids.push(id);
            Some(id)
        } else {
            None
        };
        let mut fields = crate::object::deserialize_fields(&data);
        // Only rewrite when the field is present — an object may have
        // been written before the field existed in the schema (legacy
        // path) and just lacks the entry, in which case there's nothing
        // to rename.
        if let Some(value) = fields.remove(old) {
            fields.insert(new.into(), value);
            let new_blob = crate::object::serialize_fields(&fields);
            // Refresh this type's @indexed covering payloads with the renamed
            // blob. A cover is a full copy of the object FieldMap (field names
            // embedded), so a rename stales every i: entry on the type — the
            // renamed field's own AND any sibling's — and a covered
            // `filter_scan_via_index` would otherwise return the old name.
            // field_id is stable, so each rebuilt key overwrites in place. Same
            // atomic batch; no-op without a maintainer or @indexed fields.
            if let (Some(cover), Some(object_id)) = (cover, object_id) {
                puts.extend(cover.rename_index_cover_puts(
                    type_name,
                    object_id,
                    &fields,
                    &new_blob,
                    &current_field_name,
                ));
            }
            puts.push((key.clone(), new_blob));
            objects_rewritten += 1;
        }
    }

    // ---- Reverse-edge cover blob rewrite --------------------------
    // Every rev_edge `r:<target>:<rel>:<source>` whose source is an
    // object of `type_name` carries a copy of the source's FieldMap as
    // its value (the "cover blob" the executor reads on path traversal
    // fast paths). The renamed field lives inside that FieldMap by
    // name, so unless we rewrite the cover in this same atomic batch
    // the executor returns Objects with the OLD field name post-rename
    // (the cover_v stamp matches because rename doesn't bump it, so
    // the existing staleness fall-through never fires).
    //
    // Bounded work: O(edges where source = type_name's objects via a
    // forward relation). Scans `e:<obj_id>:<rel_id>:` to enumerate
    // targets, then reads + rewrites each `r:<tgt>:<rel>:<obj_id>` in
    // place. This verb only ever renames a scalar/vector field (relation
    // fields are routed to `apply_relation_rename_verb`), so the renamed
    // name appears in the cover as the bare `<old>` key only — no
    // `<old>__cover` / `<old>__cover_v` sidecars to touch (those belong to
    // relation fields).
    // Which forward relations' rev-edge covers carry this type's object payloads:
    // derive from the catalog by STABLE rel_id (split each rel qual's type), NOT
    // from the immutable plan `schema`'s field NAMES. In a multi-verb plan an
    // earlier relation rename re-keys `cat.rel_ids`, so a name-based lookup would
    // miss and skip those covers, leaving this scalar field's name stale in them.
    // Mirrors `apply_relation_rename_verb`; inverse relations have no forward edges
    // so the `e:` scan below finds nothing for them (same covers as before).
    let mut forward_rel_ids: Vec<u64> = Vec::new();
    for (qual, &rel_id) in &cat.rel_ids {
        let (t, _) = split_qualified(qual);
        if t == type_name {
            forward_rel_ids.push(rel_id);
        }
    }

    let mut covers_rewritten: u64 = 0;
    for src_id in &source_object_ids {
        for &rel_id in &forward_rel_ids {
            let edge_prefix = KeyBuilder::edge_prefix(*src_id, rel_id);
            let edges = overlay.scan_prefix_at(storage, snap, &edge_prefix)?;
            for (edge_key, _) in edges {
                // Edge key layout: `e:<src 8 BE>:<rel 8 BE>:<tgt 8 BE>`.
                // The last 8 bytes are the target_id.
                if edge_key.len() < 8 {
                    continue;
                }
                let tgt_bytes: [u8; 8] =
                    edge_key[edge_key.len() - 8..].try_into().unwrap();
                let tgt_id = u64::from_be_bytes(tgt_bytes);
                let rev_key = KeyBuilder::reverse_edge(tgt_id, rel_id, *src_id);
                let Some(rev_bytes) = overlay.get_at(storage, snap, &rev_key)? else {
                    continue;
                };
                // Empty cover means "use fall-through" — nothing to rewrite.
                if rev_bytes.is_empty() {
                    continue;
                }
                let mut cover_fields = crate::object::deserialize_fields(&rev_bytes);
                if let Some(v) = cover_fields.remove(old) {
                    cover_fields.insert(new.into(), v);
                    let new_cover = crate::object::serialize_fields(&cover_fields);
                    puts.push((rev_key, new_cover));
                    covers_rewritten += 1;
                }
            }
        }
    }

    // ---- Catalog row mutation -------------------------------------
    let mut new_entry = field_entry.clone();
    let preserved_id = new_entry.id;
    new_entry.previous_names.insert(
        0,
        RenameRecord {
            from: old.into(),
            to: new.into(),
            wall_time_unix_ms: now_ms,
        },
    );
    new_entry.last_renamed_at_ms = Some(now_ms);

    puts.push((
        KeyBuilder::catalog_field(type_name, new),
        encode_id_entry(&new_entry),
    ));
    deletes.push(KeyBuilder::catalog_field(type_name, old));

    // Re-key in-memory catalog maps so subsequent verbs in this same
    // plan observe the new name. Tombstoned-quals set is NOT touched:
    // a rename leaves the old name free for reuse (the field wasn't
    // retired, it was renamed).
    cat.field_ids.remove(&old_qual);
    cat.field_ids.insert(new_qual.clone(), preserved_id);
    if let Some(kind) = cat.field_kinds.remove(&old_qual) {
        cat.field_kinds.insert(new_qual.clone(), kind);
    }
    cat.field_entries.remove(&old_qual);
    cat.field_entries
        .insert(new_qual.clone(), new_entry);

    report.renamed_fields.push(FieldRenamePair {
        type_name: type_name.into(),
        from: old.into(),
        to: new.into(),
        field_id: preserved_id,
        objects_rewritten,
        covers_rewritten,
    });
    Ok(())
}

/// rename_field verb for a RELATION field (phase 3). Relation fields live in
/// the `rel_entries` catalog namespace (not `field_entries`) and — unlike
/// scalar fields — are NOT stored in object payloads. They live in the edge
/// index: forward edges `e:<src>:<rel_id>:<tgt>` and reverse edges
/// `r:<tgt>:<rel_id>:<src>`, both keyed by the STABLE `rel_id`. So a relation
/// rename rewrites no object blobs and no edge KEYS. What DOES embed the
/// relation field NAME is:
///   * the catalog row `c:R:<type>\x00<name>` (re-keyed, `field_id` preserved,
///     `RenameRecord` appended), and
///   * reverse-edge COVER blobs — `<name>: U64(target)` for the linked relation
///     and, for every OTHER forward-1:1 relation of the source, the peer
///     sidecars `<name>__cover` / `<name>__cover_v`.
///
/// Both are rewritten here in one atomic batch. In-memory name-keyed caches
/// (`incoming_relations`, `cascade_meta`) are rebuilt on the mandatory
/// post-rename reopen, so cascade-delete then resolves the new name correctly.
#[allow(clippy::too_many_arguments)]
fn apply_relation_rename_verb(
    storage: &LsmTree,
    _schema: &Schema,
    snap: u64,
    overlay: &WriteOverlay,
    cat: &mut Catalog,
    type_name: &str,
    old: &str,
    new: &str,
    now_ms: u64,
    puts: &mut Vec<(Bytes, Bytes)>,
    deletes: &mut Vec<Bytes>,
    report: &mut MigrationReport,
) -> EngineResult<()> {
    // No-op rejection.
    if old == new {
        return Err(EngineError::Catalog(CatalogError::RenameNoOp {
            kind: "field",
            name: format!("{type_name}.{old}"),
        }));
    }
    check_identifier(new)?;

    // Parent type must exist and be live.
    let type_entry = cat
        .type_entries
        .get(type_name)
        .ok_or_else(|| {
            EngineError::Catalog(CatalogError::RenameSourceNotFound {
                kind: "type",
                name: type_name.into(),
            })
        })?
        .clone();
    if type_entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(CatalogError::RenameSourceRetired {
            kind: "type",
            name: type_name.into(),
            retired_id: type_entry.id,
            retired_at_ms: type_entry.retired_at_ms.unwrap_or(0),
        }));
    }

    let old_qual = format!("{type_name}.{old}");
    let new_qual = format!("{type_name}.{new}");

    // Source relation must exist and be live.
    let rel_entry = cat
        .rel_entries
        .get(&old_qual)
        .ok_or_else(|| {
            EngineError::Catalog(CatalogError::RenameSourceNotFound {
                kind: "field",
                name: old_qual.clone(),
            })
        })?
        .clone();
    if rel_entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(CatalogError::RenameSourceRetired {
            kind: "field",
            name: old_qual.clone(),
            retired_id: rel_entry.id,
            retired_at_ms: rel_entry.retired_at_ms.unwrap_or(0),
        }));
    }

    // Target name must not collide with a live/retired field OR relation.
    if let Some(existing) = cat.field_entries.get(&new_qual) {
        if existing.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
                kind: "field",
                name: new_qual.clone(),
                retired_id: existing.id,
            }));
        }
        return Err(EngineError::Catalog(CatalogError::RenameTargetCollision {
            kind: "field",
            name: new_qual.clone(),
            existing_id: existing.id,
        }));
    }
    if let Some(existing) = cat.rel_entries.get(&new_qual) {
        if existing.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
                kind: "field",
                name: new_qual.clone(),
                retired_id: existing.id,
            }));
        }
        return Err(EngineError::Catalog(CatalogError::RenameTargetCollision {
            kind: "field",
            name: new_qual.clone(),
            existing_id: existing.id,
        }));
    }
    if cat.tombstoned_field_quals.contains(&new_qual)
        || cat.tombstoned_rel_quals.contains(&new_qual)
    {
        return Err(EngineError::Catalog(CatalogError::RenameTargetIsRetired {
            kind: "field",
            name: new_qual.clone(),
            retired_id: 0,
        }));
    }

    // History-cap check.
    if rel_entry.previous_names.len() + 1 > MAX_RENAME_HISTORY {
        return Err(EngineError::Catalog(
            CatalogError::RenameHistoryCapExceeded {
                kind: "field",
                name: old_qual.clone(),
                cap: MAX_RENAME_HISTORY,
            },
        ));
    }

    // ---- Reverse-edge cover rewrite -------------------------------
    // The relation NAME appears inside reverse-edge cover blobs in two roles:
    //   (a) the renamed relation's OWN rev_edges carry `<old>: U64(target)`;
    //   (b) every OTHER forward-1:1 relation's rev_edge cover carries the peer
    //       sidecars `<old>`, `<old>__cover`, `<old>__cover_v`.
    // Scan every live relation of this type from the CATALOG (schema-shape
    // independent — robust whether the caller passed a pre- or post-rename
    // schema), walk each source object's forward edges, and rewrite the
    // matching keys in each target's rev_edge value. Inverse relations have no
    // forward edges (the scan finds nothing); many-relations have empty covers
    // (skipped). Bounded by O(edges out of this type's objects).
    let type_id = *cat.type_ids.get(type_name).ok_or_else(|| {
        EngineError::Catalog(CatalogError::RenameSourceNotFound {
            kind: "type",
            name: type_name.into(),
        })
    })?;
    let mut rel_ids: Vec<u64> = Vec::new();
    for (qual, &rid) in &cat.rel_ids {
        let (t, _) = split_qualified(qual);
        if t == type_name {
            rel_ids.push(rid);
        }
    }
    let obj_prefix = KeyBuilder::object_prefix(type_id);
    let obj_entries = overlay.scan_prefix_at(storage, snap, &obj_prefix)?;
    let mut source_ids: Vec<u64> = Vec::with_capacity(obj_entries.len());
    for (key, _) in &obj_entries {
        if key.len() >= 8 {
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            source_ids.push(u64::from_be_bytes(id_bytes));
        }
    }

    let old_cover = format!("{old}__cover");
    let old_cover_v = format!("{old}__cover_v");
    let new_cover = format!("{new}__cover");
    let new_cover_v = format!("{new}__cover_v");

    let mut covers_rewritten: u64 = 0;
    for src_id in &source_ids {
        for &rel_id in &rel_ids {
            let edge_prefix = KeyBuilder::edge_prefix(*src_id, rel_id);
            let edges = overlay.scan_prefix_at(storage, snap, &edge_prefix)?;
            for (edge_key, _) in edges {
                if edge_key.len() < 8 {
                    continue;
                }
                let tgt_bytes: [u8; 8] = edge_key[edge_key.len() - 8..].try_into().unwrap();
                let tgt_id = u64::from_be_bytes(tgt_bytes);
                let rev_key = KeyBuilder::reverse_edge(tgt_id, rel_id, *src_id);
                let Some(rev_bytes) = overlay.get_at(storage, snap, &rev_key)? else {
                    continue;
                };
                if rev_bytes.is_empty() {
                    continue;
                }
                let mut cover_fields = crate::object::deserialize_fields(&rev_bytes);
                let mut changed = false;
                if let Some(v) = cover_fields.remove(old) {
                    cover_fields.insert(new.into(), v);
                    changed = true;
                }
                if let Some(v) = cover_fields.remove(&old_cover) {
                    cover_fields.insert(new_cover.clone(), v);
                    changed = true;
                }
                if let Some(v) = cover_fields.remove(&old_cover_v) {
                    cover_fields.insert(new_cover_v.clone(), v);
                    changed = true;
                }
                if changed {
                    let new_blob = crate::object::serialize_fields(&cover_fields);
                    puts.push((rev_key, new_blob));
                    covers_rewritten += 1;
                }
            }
        }
    }

    // ---- Catalog row mutation -------------------------------------
    // A relation field has TWO catalog rows: the generic field row (`c:E:`,
    // tracked in field_entries/field_ids/field_kinds) AND the relation row
    // (`c:R:`, tracked in rel_entries/rel_ids). Re-key BOTH, each preserving its
    // own id and appending a RenameRecord. The Tombstoned-quals sets are NOT
    // touched: a rename leaves the old name free for reuse.
    let preserved_field_id = if let Some(field_entry) = cat.field_entries.get(&old_qual).cloned() {
        let mut new_field_entry = field_entry;
        let id = new_field_entry.id;
        new_field_entry.previous_names.insert(
            0,
            RenameRecord {
                from: old.into(),
                to: new.into(),
                wall_time_unix_ms: now_ms,
            },
        );
        new_field_entry.last_renamed_at_ms = Some(now_ms);
        puts.push((
            KeyBuilder::catalog_field(type_name, new),
            encode_id_entry(&new_field_entry),
        ));
        deletes.push(KeyBuilder::catalog_field(type_name, old));
        cat.field_ids.remove(&old_qual);
        cat.field_ids.insert(new_qual.clone(), id);
        if let Some(kind) = cat.field_kinds.remove(&old_qual) {
            cat.field_kinds.insert(new_qual.clone(), kind);
        }
        cat.field_entries.remove(&old_qual);
        cat.field_entries.insert(new_qual.clone(), new_field_entry);
        id
    } else {
        // A relation with no generic field row should not happen, but report
        // the rel_id rather than panic.
        rel_entry.id
    };

    let mut new_rel_entry = rel_entry.clone();
    let preserved_rel_id = new_rel_entry.id;
    new_rel_entry.previous_names.insert(
        0,
        RenameRecord {
            from: old.into(),
            to: new.into(),
            wall_time_unix_ms: now_ms,
        },
    );
    new_rel_entry.last_renamed_at_ms = Some(now_ms);
    puts.push((
        KeyBuilder::catalog_rel(type_name, new),
        encode_id_entry(&new_rel_entry),
    ));
    deletes.push(KeyBuilder::catalog_rel(type_name, old));
    cat.rel_ids.remove(&old_qual);
    cat.rel_ids.insert(new_qual.clone(), preserved_rel_id);
    cat.rel_entries.remove(&old_qual);
    cat.rel_entries.insert(new_qual.clone(), new_rel_entry);

    report.renamed_fields.push(FieldRenamePair {
        type_name: type_name.into(),
        from: old.into(),
        to: new.into(),
        field_id: preserved_field_id,
        objects_rewritten: 0,
        covers_rewritten,
    });
    Ok(())
}

// =====================================================================
// FIELD-TYPE CHANGE — re-encode every object's value for a field whose
// kind is changing (card 4/5 phase 1).
//
// Scope: plain scalar fields only. `@indexed`, `@unique`, `@vectorize`,
// and relation fields are refused with typed errors — each requires
// follow-on work (index rebuild under the new encoding, uniqueness
// re-check, embedding pipeline, schema-typed relation kinds).
//
// Atomicity: the whole migration commits in one LSM batch under
// `CATALOG_INIT_LOCK`. A 1M-row migration produces a single fsync —
// fine for planned-maintenance use cases, NOT fine for online use.
// The synthesis's shadow-field + double-write + resumable cursor
// design is the right answer for the latter; deferred to a follow-on
// card so we can ship this verb today.
// =====================================================================

/// Supplies the per-object secondary-index covering-payload re-puts an offline
/// field-type change must stage so a covered query on a SIBLING `@indexed`
/// field stops serving the migrated field's stale source value. The covering
/// payload is a full copy of the object blob with NO generation stamp, so the
/// `stage_generation_bump` this verb already does can't invalidate it — it must
/// be REWRITTEN. The catalog can't reach the engine's `@indexed` metadata +
/// index-key encoding (they live on `Database`), so it asks an implementor (the
/// `Database`) for the extra puts — inverting the dependency so catalog stays
/// the lower layer. Mirrors what the online cutover does via
/// `rewrite_object_and_maintain_covers`.
pub(crate) trait FieldCoverMaintainer {
    /// For a converted object — now serialized (`serialized`) with its migrated
    /// field at the target kind — return the `(key, value)` puts that overwrite
    /// each sibling `@indexed` field's covering payload with the fresh blob.
    /// `fields` is the object's full FieldMap (the indexed values are unchanged
    /// by the migration, so they reproduce the existing index keys).
    fn sibling_index_cover_puts(
        &self,
        type_name: &str,
        object_id: u64,
        fields: &crate::object::FieldMap,
        serialized: &Bytes,
    ) -> Vec<(Bytes, Bytes)>;

    /// Rename-time variant: refresh EVERY `@indexed` field's covering payload
    /// for this object with the post-rename blob `serialized`. A covering blob
    /// is a full copy of the object FieldMap (every field's NAME embedded), so a
    /// field rename stales EVERY index cover on the type — not just the
    /// renamed field's own — and `filter_scan_via_index` would otherwise hand
    /// back objects whose `fields.get(<current name>)` is `None`.
    ///
    /// `current_field_name` maps each field's STABLE `field_id` to its CURRENT
    /// bare name in `fields` — the caller folds in earlier verbs' renames AND this
    /// verb's own old→new. The in-memory index metadata (`indexed_fields`) still
    /// carries pre-plan names, so the maintainer resolves each indexed field's
    /// lookup name through this map by `field_id`. This is correct even when an
    /// EARLIER verb in the same plan already renamed an indexed field (a single
    /// old→new remap cannot resolve that). `field_id`/`kind` are stable across a
    /// rename, so each existing `i:` key is reproduced exactly and the `put`
    /// overwrites the stale-name blob in place. No-op when the type has no
    /// `@indexed` fields.
    fn rename_index_cover_puts(
        &self,
        type_name: &str,
        object_id: u64,
        fields: &crate::object::FieldMap,
        serialized: &Bytes,
        current_field_name: &std::collections::HashMap<u64, String>,
    ) -> Vec<(Bytes, Bytes)>;
}

pub(crate) fn apply_field_type_change(
    storage: &LsmTree,
    schema: &Schema,
    verb: FieldTypeChangeVerb,
    cover: Option<&dyn FieldCoverMaintainer>,
) -> EngineResult<MigrationReport> {
    let _guard = CATALOG_INIT_LOCK.lock();

    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();

    let result = (|| -> EngineResult<(MigrationReport, bool)> {
        let fv_raw = storage
            .get(&txn, &KeyBuilder::catalog_format())?
            .ok_or_else(|| {
                EngineError::Catalog(CatalogError::MissingRequiredKey {
                    key_debug: "c:F:".into(),
                })
            })?;
        let format_before = decode_format_version("c:F:", &fv_raw)?;
        if format_before > CATALOG_FORMAT_CURRENT {
            return Err(EngineError::Catalog(CatalogError::UnsupportedFormat {
                got: format_before,
                max_supported: CATALOG_FORMAT_CURRENT,
            }));
        }
        let mut cat = load_existing(storage, snap, format_before)?;

        let qual = format!("{}.{}", verb.type_name, verb.field_name);

        // Offline-surface interlock (shadow-field card 1): refuse a
        // single-commit field-type change while ANY chunked migration plan is
        // unsettled on the same TYPE. The chunked worker rewrites the whole
        // object blob with `migration_lock` dropped, so even a change on a
        // DIFFERENT field of the type would clobber its `o:<type>:*` writes
        // (and two flippers on one field double-write type_change_history).
        // This one site covers all three offline reach paths
        // (Database::change_field_type, *_consuming, and
        // MigrationContext::change_field_type via run_migrations) since they
        // all funnel here. Settled plans (Completed/Cancelled) don't block.
        let plans = scan_migration_plans(storage, snap)?;
        if let Some((qualified, plan_id)) = active_plan_for_type(&plans, &verb.type_name) {
            return Err(EngineError::Catalog(
                CatalogError::MigrationFieldHasActivePlan { qualified, plan_id },
            ));
        }

        // ---- Validation (shared with the chunked migration create path) -
        let (field_entry, type_id) = validate_field_type_change(
            &cat,
            schema,
            &verb.type_name,
            &verb.field_name,
            verb.target_kind,
        )?;

        // ---- Object scan + closure + re-serialize ---------------------
        let prefix = KeyBuilder::object_prefix(type_id);
        let entries = storage.scan_prefix_at(snap, &prefix)?;

        let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(entries.len() + 4);
        let mut objects_converted: u64 = 0;
        for (key, data) in entries {
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);

            let mut fields = crate::object::deserialize_fields(&data);
            if let Some(old_value) = fields.get(&verb.field_name).cloned() {
                let new_value = (verb.converter)(object_id, &old_value).map_err(|e| {
                    EngineError::Catalog(CatalogError::FieldTypeChangeConverterFailed {
                        qualified: qual.clone(),
                        object_id,
                        reason: e.to_string(),
                    })
                })?;
                let got_kind = value_to_kind_byte(&new_value);
                if got_kind != verb.target_kind {
                    return Err(EngineError::Catalog(
                        CatalogError::FieldTypeChangeConverterReturnedWrongKind {
                            qualified: qual.clone(),
                            object_id,
                            got_kind: kind_name(got_kind),
                            want_kind: kind_name(verb.target_kind),
                        },
                    ));
                }
                // Bump the generation BEFORE the converted blob so a covered
                // read re-probes the new value instead of serving the stale
                // cover (see stage_generation_bump).
                stage_generation_bump(storage, &txn, type_id, object_id, &mut puts)?;
                fields.insert(verb.field_name.clone(), new_value);
                let new_blob = crate::object::serialize_fields(&fields);
                // Refresh sibling @indexed covering payloads with the new blob
                // (they carry no generation stamp, so the bump above can't fix
                // them). Same atomic batch as the o: blob. No-op when there are
                // no @indexed siblings.
                if let Some(cover) = cover {
                    puts.extend(cover.sibling_index_cover_puts(
                        &verb.type_name,
                        object_id,
                        &fields,
                        &new_blob,
                    ));
                }
                puts.push((key.clone(), new_blob));
                objects_converted += 1;
            }
        }

        // ---- Catalog mutation -----------------------------------------
        let now_ms = now_unix_millis();
        let from_kind = field_entry.kind;
        let mut new_entry = field_entry.clone();
        new_entry.kind = verb.target_kind;
        new_entry.type_change_history.insert(
            0,
            TypeChangeRecord {
                from_kind,
                to_kind: verb.target_kind,
                wall_time_unix_ms: now_ms,
            },
        );
        new_entry.last_type_change_at_ms = Some(now_ms);

        puts.push((
            KeyBuilder::catalog_field(&verb.type_name, &verb.field_name),
            encode_id_entry(&new_entry),
        ));
        cat.field_kinds.insert(qual.clone(), verb.target_kind);
        cat.field_entries.insert(qual.clone(), new_entry);

        // Bump catalog format to v4 the first time a type change lands.
        let mut format_after = cat.format_version;
        if format_after < CATALOG_FORMAT_V4 {
            format_after = CATALOG_FORMAT_V4;
            cat.format_version = format_after;
            puts.push((
                KeyBuilder::catalog_format(),
                encode_format_version(CATALOG_FORMAT_V4),
            ));
        }

        // Leave c:D: stale — the next open with the post-change schema
        // computes a different digest (kind contributes to the hash),
        // reconcile runs once, finds no drops/adds/kind-mismatches
        // (we just updated the catalog kind to match), refreshes the
        // digest in its tail commit.

        storage.put_batch(&mut txn, &puts)?;

        let mut report = MigrationReport {
            catalog_format_before: format_before,
            catalog_format_after: format_after,
            ..MigrationReport::default()
        };
        report.field_type_changes.push(FieldTypeChangePair {
            type_name: verb.type_name.clone(),
            field_name: verb.field_name.clone(),
            from_kind,
            to_kind: verb.target_kind,
            field_id: field_entry.id,
            objects_converted,
        });
        Ok((report, true))
    })();

    match result {
        Ok((report, wrote)) => {
            if wrote {
                storage.commit(&mut txn)?;
            } else {
                storage.abort(&mut txn);
            }
            Ok(report)
        }
        Err(e) => {
            storage.abort(&mut txn);
            Err(e)
        }
    }
}

/// Shared validation for a scalar field-type change, used by BOTH the
/// offline single-commit path (`apply_field_type_change`) and the chunked
/// migration create path. Runs every gate EXCEPT the active-plan interlock
/// (each caller does its own `active_plan_for_type` check — both refuse while
/// any unsettled plan covers the same type). Returns the
/// validated source field entry (cloned) and its owning type id. The error
/// order is preserved verbatim so the offline-path regression tests stay
/// byte-identical.
fn validate_field_type_change(
    cat: &Catalog,
    schema: &Schema,
    type_name: &str,
    field_name: &str,
    target_kind: u8,
) -> EngineResult<(IdEntry, u64)> {
    let qual = format!("{type_name}.{field_name}");
    let field_entry = cat.field_entries.get(&qual).cloned().ok_or_else(|| {
        EngineError::Catalog(CatalogError::FieldTypeChangeSourceNotFound {
            qualified: qual.clone(),
        })
    })?;
    if field_entry.status == TombstoneStatus::Tombstoned {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeSourceRetired {
                qualified: qual.clone(),
                field_id: field_entry.id,
                retired_at_ms: field_entry.retired_at_ms.unwrap_or(0),
            },
        ));
    }
    if field_entry.kind == target_kind {
        return Err(EngineError::Catalog(CatalogError::FieldTypeChangeNoOp {
            qualified: qual,
        }));
    }
    // Phase 1 only supports SCALAR → SCALAR.
    if !is_scalar_kind(field_entry.kind) || !is_scalar_kind(target_kind) {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeNonScalar {
                qualified: qual,
                current_kind: kind_name(field_entry.kind),
                requested_kind: kind_name(target_kind),
            },
        ));
    }
    // Refuse a target kind the engine has no writable `Value` for. Every scalar
    // (including DateTime/Json) now has a variant, so this only fires for a
    // non-scalar / unset target — a converter could never return a value whose
    // kind byte matches, so the migration would fail every row. (A converter
    // that targets DateTime/Json must actually produce that Value variant; the
    // per-row contract check still enforces that.) Also closes the latent
    // offline-path bug where, with zero objects, this would vacuously flip the
    // catalog to an unrepresentable kind.
    if !is_representable_target_kind(target_kind) {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeUnrepresentableTarget {
                qualified: qual,
                kind: kind_name(target_kind),
            },
        ));
    }
    // Refuse @indexed / @unique / @vectorize. We consult the schema's field
    // definition for the source field: the field exists in cat (just
    // validated above), so it must exist in schema (caller passed the
    // in-process Schema).
    let type_def = schema.types.get(type_name).ok_or_else(|| {
        EngineError::Catalog(CatalogError::FieldTypeChangeSourceNotFound {
            qualified: qual.clone(),
        })
    })?;
    let field_def = type_def
        .fields
        .iter()
        .find(|f| f.name == field_name)
        .ok_or_else(|| {
            EngineError::Catalog(CatalogError::FieldTypeChangeSourceNotFound {
                qualified: qual.clone(),
            })
        })?;
    if field_def.is_indexed() {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeDirectiveUnsupported {
                qualified: qual,
                directive: "@indexed",
                planned_phase: "follow-on",
            },
        ));
    }
    if field_def.is_unique() {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeDirectiveUnsupported {
                qualified: qual,
                directive: "@unique",
                planned_phase: "follow-on",
            },
        ));
    }
    if field_def.vectorize().is_some() {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeDirectiveUnsupported {
                qualified: qual,
                directive: "@vectorize",
                planned_phase: "follow-on",
            },
        ));
    }
    if field_entry.type_change_history.len() + 1 > MAX_TYPE_CHANGE_HISTORY {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeHistoryCapExceeded {
                qualified: qual,
                cap: MAX_TYPE_CHANGE_HISTORY,
            },
        ));
    }
    let type_id = *cat.type_ids.get(type_name).ok_or_else(|| {
        EngineError::Catalog(CatalogError::FieldTypeChangeSourceNotFound {
            qualified: qual.clone(),
        })
    })?;
    Ok((field_entry, type_id))
}

/// Stage a generation bump for a migrated object so rev-edge COVERS that
/// embed its old field value are invalidated.
///
/// Rev-edge 1:1-forward covers cache a copy of the target's FieldMap plus a
/// `<field>__cover_v` stamp = the target's generation at cover-write time. The
/// fusion reader serves the cached cover iff `cover_v == object_version(target)`
/// (executor.rs). A field-type change rewrites the object's `o:` blob WITHOUT
/// going through the normal update path, so the generation never moves — and
/// because of born-at-1 EVERY live object is generation >= 1, so `cover_v ==
/// object_version` still holds and a covered read serves the STALE source
/// value. Bumping the generation here (write the `g:` object_version key =
/// current + 1; born-at-1 default of 1 when the key is absent) makes the
/// staleness check re-probe the converted `o:` blob.
///
/// MUST be staged BEFORE the converted blob in the same batch: with per-entry
/// WAL framing a torn tail drops a suffix, so "blob converted, generation not
/// bumped" (which idempotent-on-target resume would then SKIP, stranding a
/// stale cover forever) must be impossible. Over-bumping on a re-do is
/// harmless — the generation is a monotonic staleness counter, not a count.
/// The live handle's in-memory `version_counters` is left stale (the migrate
/// handle is stale post-change → reopen rebuilds it from these `g:` keys).
fn stage_generation_bump(
    storage: &LsmTree,
    txn: &rhypedb_storage::mvcc::Transaction,
    type_id: u64,
    object_id: u64,
    puts: &mut Vec<(Bytes, Bytes)>,
) -> EngineResult<()> {
    let g_key = KeyBuilder::object_version(type_id, object_id);
    let current = match storage.get(txn, &g_key)? {
        Some(b) if b.len() == 8 => u64::from_be_bytes(b[..].try_into().unwrap()),
        _ => 1, // born-at-1: a live object with no g: key is generation 1
    };
    puts.push((
        g_key,
        Bytes::copy_from_slice(&current.saturating_add(1).to_be_bytes()),
    ));
    Ok(())
}

fn is_scalar_kind(k: u8) -> bool {
    use kind_byte::*;
    matches!(
        k,
        SCALAR_STRING
            | SCALAR_U32
            | SCALAR_U64
            | SCALAR_I32
            | SCALAR_I64
            | SCALAR_F32
            | SCALAR_F64
            | SCALAR_BOOL
            | SCALAR_DATETIME
            | SCALAR_BYTES
            | SCALAR_JSON
    )
}

/// Target kinds a converter can actually PRODUCE — exactly the kinds
/// `value_to_kind_byte` can emit (every scalar kind now has a `Value` variant).
fn is_representable_target_kind(k: u8) -> bool {
    is_scalar_kind(k)
}

fn value_to_kind_byte(v: &crate::object::Value) -> u8 {
    use crate::object::Value as V;
    use kind_byte::*;
    match v {
        V::String(_) => SCALAR_STRING,
        V::U32(_) => SCALAR_U32,
        V::U64(_) => SCALAR_U64,
        V::I32(_) => SCALAR_I32,
        V::I64(_) => SCALAR_I64,
        V::F32(_) => SCALAR_F32,
        V::F64(_) => SCALAR_F64,
        V::Bool(_) => SCALAR_BOOL,
        V::DateTime(_) => SCALAR_DATETIME,
        V::Bytes(_) => SCALAR_BYTES,
        V::Json(_) => SCALAR_JSON,
        V::Null => UNSET,
    }
}

// =====================================================================
// MIGRATION LOG — versioned schema + named migration replay (card 5/5)
//
// Card 5/5 layers an idempotent migration framework on top of the
// verbs cards 1-4 already provide. The operator passes a Vec of named
// `Migration`s; we track which ordinals have been applied via two new
// catalog keys:
//   * `c:V:` — u64 BE current applied count (next-ordinal-to-apply)
//   * `c:G:<u64 BE ordinal>` — applied migration record (utf8 name + ms)
//
// Validation on each `run_migrations` call:
//   * `migrations.len() < current_version` → `MigrationListShorterThanApplied`
//   * `stored_name(i) != migrations[i].name` for i < current_version
//        → `MigrationNameMismatch`
//
// Pending migrations are applied one-at-a-time: each migration's
// closure runs verbs (rename_type, change_field_type) and the log
// record is written in the same single-commit flow each verb already
// uses. If a migration's verb fails, the migration aborts and the
// log entry is NOT written; the version counter is not bumped.
// =====================================================================

/// One named, ordered schema migration. Card 5/5 supports closures that
/// run any combination of `rename_type` / `change_field_type` verbs
/// through the `MigrationContext`. Each migration's effect is bounded
/// by whatever those verbs commit (each is its own atomic commit; a
/// multi-verb migration is multiple commits — see the `auto_resume`
/// note in the synthesis).
pub struct Migration {
    pub name: String,
    pub up: MigrationUp,
}

/// The `up` closure of a [`Migration`]: applies the migration's verbs through
/// the `MigrationContext`. Consumed (FnOnce) when the migration runs.
pub type MigrationUp = Box<dyn FnOnce(&MigrationContext) -> EngineResult<()> + Send + Sync>;

impl Migration {
    pub fn new<S, F>(name: S, up: F) -> Self
    where
        S: Into<String>,
        F: FnOnce(&MigrationContext) -> EngineResult<()> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            up: Box::new(up),
        }
    }
}

/// Per-migration handle the closure uses to run verbs against the
/// open catalog. Each method delegates to the corresponding card's
/// verb (rename_type → card 3, change_field_type → card 4). The
/// schema reference is the **final** schema the operator will open
/// `Database` with after `run_migrations` completes — migrations
/// that change field types need it to validate `@indexed` etc.
pub struct MigrationContext<'a> {
    storage: &'a LsmTree,
    schema: &'a Schema,
    /// Cover maintainer threaded from `Database::run_migrations_inner` so a
    /// field-type change verb inside a migration list refreshes sibling
    /// `@indexed` covering payloads (see [`FieldCoverMaintainer`]).
    cover: Option<&'a dyn FieldCoverMaintainer>,
}

impl MigrationContext<'_> {
    pub fn rename_type(&self, old: &str, new: &str) -> EngineResult<()> {
        let verbs = [RenameVerb::Type {
            old: old.into(),
            new: new.into(),
        }];
        apply_migration_with_cover(self.storage, self.schema, &verbs, self.cover)?;
        Ok(())
    }

    pub fn rename_field(
        &self,
        type_name: &str,
        old: &str,
        new: &str,
    ) -> EngineResult<()> {
        let verbs = [RenameVerb::Field {
            type_name: type_name.into(),
            old: old.into(),
            new: new.into(),
        }];
        apply_migration_with_cover(self.storage, self.schema, &verbs, self.cover)?;
        Ok(())
    }

    pub fn change_field_type<F>(
        &self,
        type_name: &str,
        field_name: &str,
        target_field_type: FieldType,
        converter: F,
    ) -> EngineResult<()>
    where
        F: Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value>
            + Send
            + Sync
            + 'static,
    {
        let target_kind = schema_kind_byte(&target_field_type);
        let verb = FieldTypeChangeVerb {
            type_name: type_name.into(),
            field_name: field_name.into(),
            target_kind,
            converter: Box::new(converter),
        };
        apply_field_type_change(self.storage, self.schema, verb, self.cover)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct MigrationLogReport {
    pub applied: Vec<AppliedMigration>,
    pub version_before: u64,
    pub version_after: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedMigration {
    pub ordinal: u64,
    pub name: String,
    pub applied_at_unix_ms: u64,
}

/// Apply pending migrations from `migrations`. Idempotent across
/// repeated calls — previously-applied migrations are skipped.
/// Validates that the migration list hasn't been reordered or renamed
/// at any already-applied ordinal.
pub(crate) fn run_migrations(
    storage: &LsmTree,
    schema: &Schema,
    migrations: Vec<Migration>,
    cover: Option<&dyn FieldCoverMaintainer>,
) -> EngineResult<MigrationLogReport> {
    // NOTE: we do NOT hold `CATALOG_INIT_LOCK` here. Each verb the
    // migration closure calls (rename_type, change_field_type) acquires
    // the lock itself, and parking_lot::Mutex is non-reentrant —
    // taking it here would deadlock. Migration-log read + log entry
    // writes between verbs are protected by their own short txns; if
    // a concurrent migrate runs against the same DB it'll serialise
    // verb-by-verb the same way concurrent reconcile + migrate do.

    // STEP 1: read current applied count.
    let snap = storage.read_snapshot();
    let current_version = match storage.get_at(snap, &KeyBuilder::catalog_migration_version())? {
        Some(bytes) => {
            if bytes.len() != 8 {
                return Err(EngineError::Catalog(CatalogError::Truncated {
                    key_debug: "c:V:".into(),
                    len: bytes.len(),
                    min: 8,
                }));
            }
            u64::from_be_bytes(bytes[..].try_into().unwrap())
        }
        None => 0,
    };

    // STEP 2: detect DB-ahead-of-code.
    if (migrations.len() as u64) < current_version {
        return Err(EngineError::Catalog(
            CatalogError::MigrationListShorterThanApplied {
                code_count: migrations.len() as u64,
                catalog_count: current_version,
            },
        ));
    }

    // STEP 3: detect rename/reorder of already-applied migrations.
    for (i, mig) in migrations.iter().enumerate() {
        let ord = i as u64;
        if ord >= current_version {
            break;
        }
        let applied = read_migration_log_entry(storage, snap, ord)?;
        if applied.name != mig.name {
            return Err(EngineError::Catalog(CatalogError::MigrationNameMismatch {
                ordinal: ord,
                code_name: mig.name.clone(),
                catalog_name: applied.name,
            }));
        }
    }

    // STEP 4: apply pending migrations.
    let mut report = MigrationLogReport {
        version_before: current_version,
        version_after: current_version,
        ..MigrationLogReport::default()
    };
    let mut version_cursor = current_version;
    let ctx = MigrationContext {
        storage,
        schema,
        cover,
    };
    for (i, mig) in migrations.into_iter().enumerate() {
        let ord = i as u64;
        if ord < current_version {
            continue;
        }
        let name = mig.name.clone();
        (mig.up)(&ctx).map_err(|e| {
            EngineError::Catalog(CatalogError::MigrationVerbFailed {
                ordinal: ord,
                name: name.clone(),
                reason: e.to_string(),
            })
        })?;
        // Persist the log entry + bump version in one commit.
        let now_ms = now_unix_millis();
        let mut txn = storage.begin_txn();
        let entry_bytes = encode_migration_log_entry(&name, now_ms);
        storage.put(&mut txn, &KeyBuilder::catalog_migration_log(ord), entry_bytes)?;
        let next = ord + 1;
        storage.put(
            &mut txn,
            &KeyBuilder::catalog_migration_version(),
            Bytes::copy_from_slice(&next.to_be_bytes()),
        )?;
        storage.commit(&mut txn)?;
        version_cursor = next;
        report.applied.push(AppliedMigration {
            ordinal: ord,
            name,
            applied_at_unix_ms: now_ms,
        });
    }
    report.version_after = version_cursor;
    Ok(report)
}

fn read_migration_log_entry(
    storage: &LsmTree,
    snap: u64,
    ordinal: u64,
) -> EngineResult<AppliedMigration> {
    let raw = storage
        .get_at(snap, &KeyBuilder::catalog_migration_log(ordinal))?
        .ok_or_else(|| {
            EngineError::Catalog(CatalogError::MissingRequiredKey {
                key_debug: format!("c:G:{}", ordinal),
            })
        })?;
    decode_migration_log_entry(ordinal, &raw)
}

/// Migration log entry format:
/// ```text
/// byte 0..2  : u16 BE name_len
/// byte 2..N  : utf8 name (name_len bytes)
/// byte N..N+8: u64 BE applied_at_unix_ms
/// ```
fn encode_migration_log_entry(name: &str, applied_at_unix_ms: u64) -> Bytes {
    let mut buf: Vec<u8> = Vec::with_capacity(2 + name.len() + 8);
    debug_assert!(name.len() <= u16::MAX as usize);
    buf.extend_from_slice(&(name.len() as u16).to_be_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&applied_at_unix_ms.to_be_bytes());
    Bytes::from(buf)
}

fn decode_migration_log_entry(ordinal: u64, bytes: &[u8]) -> EngineResult<AppliedMigration> {
    let key_debug = format!("c:G:{}", ordinal);
    if bytes.len() < 2 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug,
            len: bytes.len(),
            min: 2,
        }));
    }
    let name_len = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
    if bytes.len() < 2 + name_len + 8 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug,
            len: bytes.len(),
            min: 2 + name_len + 8,
        }));
    }
    let name = std::str::from_utf8(&bytes[2..2 + name_len])
        .map_err(|_| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: key_debug.clone(),
            })
        })?
        .to_string();
    let ts_bytes: [u8; 8] = bytes[2 + name_len..2 + name_len + 8].try_into().unwrap();
    Ok(AppliedMigration {
        ordinal,
        name,
        applied_at_unix_ms: u64::from_be_bytes(ts_bytes),
    })
}

// =====================================================================
// BACKFILL — fresh / legacy DB → initialized catalog
// =====================================================================

fn backfill_into_txn(
    storage: &LsmTree,
    schema: &Schema,
    txn: &mut rhypedb_storage::mvcc::Transaction,
) -> EngineResult<Catalog> {
    let now_ms = now_unix_millis();
    let mut cat = Catalog::empty();

    let mut type_names: Vec<String> = schema.types.keys().cloned().collect();
    type_names.sort(); // identical to legacy std String sort

    let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(8 + type_names.len() * 8);

    puts.push((
        KeyBuilder::catalog_format(),
        encode_format_version(CATALOG_FORMAT_V1),
    ));

    for name in &type_names {
        let tid = cat.next_type;
        cat.next_type = checked_bump("type", cat.next_type)?;
        let entry = IdEntry::backfilled(tid, kind_byte::UNSET, now_ms);
        puts.push((KeyBuilder::catalog_type(name), encode_id_entry(&entry)));
        cat.insert_type(name.clone(), entry);

        let type_def = &schema.types[name];
        for field in &type_def.fields {
            let qual = format!("{}.{}", name, field.name);
            let kind = schema_kind_byte(&field.field_type);

            let fid = cat.next_field;
            cat.next_field = checked_bump("field", cat.next_field)?;
            let entry = IdEntry::backfilled(fid, kind, now_ms);
            puts.push((
                KeyBuilder::catalog_field(name, &field.name),
                encode_id_entry(&entry),
            ));
            cat.insert_field(qual.clone(), entry);

            if matches!(field.field_type, FieldType::Relation(_)) {
                let rid = cat.next_rel;
                cat.next_rel = checked_bump("rel", cat.next_rel)?;
                let entry = IdEntry::backfilled(rid, kind_byte::RELATION, now_ms);
                puts.push((
                    KeyBuilder::catalog_rel(name, &field.name),
                    encode_id_entry(&entry),
                ));
                cat.insert_rel(qual.clone(), entry);
            }
        }
    }

    puts.push((
        KeyBuilder::catalog_next_type(),
        encode_counter(cat.next_type),
    ));
    puts.push((
        KeyBuilder::catalog_next_field(),
        encode_counter(cat.next_field),
    ));
    puts.push((KeyBuilder::catalog_next_rel(), encode_counter(cat.next_rel)));

    let digest = compute_schema_digest(schema);
    puts.push((
        KeyBuilder::catalog_digest(),
        Bytes::copy_from_slice(&digest),
    ));

    // `c:I:` is LAST in the batch. WAL framing is per-record, so any
    // replay containing `c:I:` has every preceding record; any replay
    // missing it is a torn-write the recovery branches above clean up.
    puts.push((KeyBuilder::catalog_initialized(), encode_marker(now_ms)));

    storage.put_batch(txn, &puts)?;
    Ok(cat)
}

fn recover_partial_into_txn(
    storage: &LsmTree,
    schema: &Schema,
    txn: &mut rhypedb_storage::mvcc::Transaction,
    snap: u64,
) -> EngineResult<Catalog> {
    // Blanket-clear the catalog keyspace, THEN re-backfill from the live
    // schema. EXEMPT the shadow-field migration keys (`c:P:` plans, the
    // `c:S:` per-partition cursors, the `c:Q:` quarantine sidecars, and the
    // `c:N:M` id counter): they are not schema-derived, so backfill can't
    // recreate them, and a torn-init reopen that hits this branch while a
    // migration is in flight would otherwise
    // silently wipe the plan + per-partition cursors + the counter — losing
    // crash-resume (re-converting every partition from scratch) and letting a
    // freed id be reissued.
    let plan_prefix = KeyBuilder::catalog_migration_plan_prefix();
    let partition_prefix = KeyBuilder::catalog_partition_cursor_prefix();
    let quarantine_prefix = KeyBuilder::catalog_quarantine_prefix();
    let counter_key = KeyBuilder::catalog_next_migration();
    let stale = storage.scan_prefix_at(snap, &KeyBuilder::catalog_prefix_all())?;
    let deletes: Vec<Bytes> = stale
        .into_iter()
        .map(|(k, _)| k)
        .filter(|k| {
            !k.starts_with(&plan_prefix)
                && !k.starts_with(&partition_prefix)
                && !k.starts_with(&quarantine_prefix)
                && k != &counter_key
        })
        .collect();
    if !deletes.is_empty() {
        storage.delete_batch(txn, &deletes)?;
    }
    backfill_into_txn(storage, schema, txn)
}

// =====================================================================
// LOAD EXISTING — initialized catalog → in-memory Catalog struct
// =====================================================================

fn load_existing(storage: &LsmTree, snap: u64, format_version: u64) -> EngineResult<Catalog> {
    let mut cat = Catalog {
        format_version,
        ..Catalog::default()
    };

    // c:T:* — types
    let type_prefix = KeyBuilder::catalog_prefix_type();
    let mut seen_type_ids: HashMap<u64, String> = HashMap::new();
    for (key, value) in storage.scan_prefix_at(snap, &type_prefix)? {
        let name = std::str::from_utf8(&key[type_prefix.len()..])
            .map_err(|_| {
                EngineError::Catalog(CatalogError::MalformedKey {
                    key_debug: debug_key(&key),
                })
            })?
            .to_string();
        let entry = decode_id_entry(&format!("c:T:{}", name), &value)?;
        // A v1 catalog that has a tombstoned row is internally
        // inconsistent — v1 binaries never write tombstones, so this
        // is either manual tampering or a binary that bumped the row
        // format without bumping `c:F:`. Refuse to open.
        if format_version == CATALOG_FORMAT_V1 && entry.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::TombstoneOnV1Catalog {
                row_kind: "type",
                row_id: entry.id,
            }));
        }
        if let Some(prev) = seen_type_ids.insert(entry.id, name.clone()) {
            return Err(EngineError::Catalog(CatalogError::DuplicateId {
                kind: "type",
                id: entry.id,
                first: prev,
                second: name,
            }));
        }
        cat.insert_type(name, entry);
    }

    // c:E:* — fields
    let field_prefix = KeyBuilder::catalog_prefix_field();
    let mut seen_field_ids: HashMap<u64, String> = HashMap::new();
    for (key, value) in storage.scan_prefix_at(snap, &field_prefix)? {
        let payload = &key[field_prefix.len()..];
        let nul = payload.iter().position(|&b| b == 0).ok_or_else(|| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let type_name = std::str::from_utf8(&payload[..nul]).map_err(|_| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let field_name = std::str::from_utf8(&payload[nul + 1..]).map_err(|_| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let qual = format!("{}.{}", type_name, field_name);
        let entry = decode_id_entry(&format!("c:E:{}.{}", type_name, field_name), &value)?;
        if format_version == CATALOG_FORMAT_V1 && entry.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::TombstoneOnV1Catalog {
                row_kind: "field",
                row_id: entry.id,
            }));
        }
        if let Some(prev) = seen_field_ids.insert(entry.id, qual.clone()) {
            return Err(EngineError::Catalog(CatalogError::DuplicateId {
                kind: "field",
                id: entry.id,
                first: prev,
                second: qual,
            }));
        }
        cat.insert_field(qual, entry);
    }

    // c:R:* — relations
    let rel_prefix = KeyBuilder::catalog_prefix_rel();
    let mut seen_rel_ids: HashMap<u64, String> = HashMap::new();
    for (key, value) in storage.scan_prefix_at(snap, &rel_prefix)? {
        let payload = &key[rel_prefix.len()..];
        let nul = payload.iter().position(|&b| b == 0).ok_or_else(|| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let type_name = std::str::from_utf8(&payload[..nul]).map_err(|_| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let field_name = std::str::from_utf8(&payload[nul + 1..]).map_err(|_| {
            EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            })
        })?;
        let qual = format!("{}.{}", type_name, field_name);
        let entry = decode_id_entry(&format!("c:R:{}.{}", type_name, field_name), &value)?;
        if format_version == CATALOG_FORMAT_V1 && entry.status == TombstoneStatus::Tombstoned {
            return Err(EngineError::Catalog(CatalogError::TombstoneOnV1Catalog {
                row_kind: "relation",
                row_id: entry.id,
            }));
        }
        if let Some(prev) = seen_rel_ids.insert(entry.id, qual.clone()) {
            return Err(EngineError::Catalog(CatalogError::DuplicateId {
                kind: "relation",
                id: entry.id,
                first: prev,
                second: qual,
            }));
        }
        cat.insert_rel(qual, entry);
    }

    // Counters. Self-heal a stale on-disk value to `max(allocated)+1`
    // so torn-WAL tails that dropped a counter bump can't cause silent
    // ID reuse on the next additive allocation.
    let on_disk_type = read_counter(storage, snap, "c:N:T", KeyBuilder::catalog_next_type)?;
    let on_disk_field = read_counter(storage, snap, "c:N:E", KeyBuilder::catalog_next_field)?;
    let on_disk_rel = read_counter(storage, snap, "c:N:R", KeyBuilder::catalog_next_rel)?;

    cat.next_type = self_heal_counter(on_disk_type, cat.type_ids.values().copied().max());
    cat.next_field = self_heal_counter(on_disk_field, cat.field_ids.values().copied().max());
    cat.next_rel = self_heal_counter(on_disk_rel, cat.rel_ids.values().copied().max());

    Ok(cat)
}

fn read_counter(
    storage: &LsmTree,
    snap: u64,
    key_debug: &str,
    key_fn: fn() -> Bytes,
) -> EngineResult<u64> {
    let raw = storage.get_at(snap, &key_fn())?.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredKey {
            key_debug: key_debug.into(),
        })
    })?;
    decode_counter(key_debug, &raw)
}

fn self_heal_counter(on_disk: u64, max_assigned: Option<u64>) -> u64 {
    match max_assigned {
        Some(m) => on_disk.max(m.saturating_add(1)),
        None => on_disk.max(1),
    }
}

// =====================================================================
// RECONCILE — schema diff against existing catalog
// =====================================================================

fn reconcile_into_txn(
    storage: &LsmTree,
    schema: &Schema,
    cat: &mut Catalog,
    allow_schema_shrink: bool,
    new_digest: &[u8; 32],
    txn: &mut rhypedb_storage::mvcc::Transaction,
) -> EngineResult<()> {
    let now_ms = now_unix_millis();

    // PASS 1 — detect drops. Drops are catalog entries the new schema
    // doesn't name. Already-tombstoned entries are excluded; they're
    // not "drops" — they're already retired.
    let mut dropped_types: Vec<String> = cat
        .type_ids
        .keys()
        .filter(|t| !schema.types.contains_key(t.as_str()))
        .filter(|t| !cat.tombstoned_type_names.contains(t.as_str()))
        .cloned()
        .collect();
    dropped_types.sort();

    let mut dropped_fields: Vec<String> = Vec::new();
    let mut dropped_rels: Vec<String> = Vec::new();
    for qual in cat.field_ids.keys() {
        if cat.tombstoned_field_quals.contains(qual.as_str()) {
            continue;
        }
        let (t, f) = split_qualified(qual);
        match schema.types.get(t) {
            None => {} // already captured as a type drop
            Some(td) if td.fields.iter().all(|fd| fd.name != f) => {
                dropped_fields.push(qual.clone());
            }
            _ => {}
        }
    }
    for qual in cat.rel_ids.keys() {
        if cat.tombstoned_rel_quals.contains(qual.as_str()) {
            continue;
        }
        let (t, f) = split_qualified(qual);
        match schema.types.get(t) {
            None => {}
            Some(td) => {
                match td.fields.iter().find(|x| x.name == f) {
                    None => dropped_rels.push(qual.clone()),
                    Some(fd) if !matches!(fd.field_type, FieldType::Relation(_)) => {
                        // Relation became a scalar. The catalog drop
                        // surfaces here; the accompanying kind change
                        // on the matching field also fires in PASS 2.
                        dropped_rels.push(qual.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    dropped_fields.sort();
    dropped_rels.sort();

    let has_drops =
        !dropped_types.is_empty() || !dropped_fields.is_empty() || !dropped_rels.is_empty();

    if has_drops && !allow_schema_shrink {
        return Err(EngineError::Catalog(CatalogError::SchemaShrinkRequiresOptIn {
            dropped_types,
            dropped_fields,
            dropped_rels,
        }));
    }

    // PASS 1a — refuse re-binding a retired name. If the schema names
    // a type/field/relation that the catalog has tombstoned, using it
    // would either silently resurrect the retired numeric ID's data
    // (if we used the old ID) or — worse — allocate a fresh ID and
    // strand the retired data. Refuse loudly. Runs UNCONDITIONAL of
    // `allow_schema_shrink` because the shrink flag governs DROPS,
    // not REBINDS — even a non-shrinking reopen must not rebind a
    // retired name.
    for schema_type in schema.types.keys() {
        if cat.tombstoned_type_names.contains(schema_type.as_str()) {
            let retired_id = cat
                .type_entries
                .get(schema_type.as_str())
                .map(|e| e.id)
                .unwrap_or(0);
            return Err(EngineError::Catalog(
                CatalogError::NameReuseOfRetiredEntry {
                    kind: "type",
                    name: schema_type.clone(),
                    retired_id,
                },
            ));
        }
    }
    for (type_name, type_def) in &schema.types {
        for field in &type_def.fields {
            let qual = format!("{}.{}", type_name, field.name);
            let is_rel = matches!(field.field_type, FieldType::Relation(_));
            if cat.tombstoned_field_quals.contains(qual.as_str()) {
                let retired_id = cat
                    .field_entries
                    .get(qual.as_str())
                    .map(|e| e.id)
                    .unwrap_or(0);
                return Err(EngineError::Catalog(
                    CatalogError::NameReuseOfRetiredEntry {
                        kind: "field",
                        name: qual,
                        retired_id,
                    },
                ));
            }
            if is_rel && cat.tombstoned_rel_quals.contains(qual.as_str()) {
                let retired_id = cat
                    .rel_entries
                    .get(qual.as_str())
                    .map(|e| e.id)
                    .unwrap_or(0);
                return Err(EngineError::Catalog(
                    CatalogError::NameReuseOfRetiredEntry {
                        kind: "relation",
                        name: qual,
                        retired_id,
                    },
                ));
            }
        }
    }

    // PASS 1b — cascade closure: for every dropped type, add ALL of
    // its catalog-known fields and relations to the dropped sets so we
    // tombstone the whole subtree in one commit. Tag them as
    // CascadeParentRetired so observability tells the right story.
    let mut cascaded_fields: HashSet<String> = HashSet::new();
    let mut cascaded_rels: HashSet<String> = HashSet::new();
    if !dropped_types.is_empty() {
        let dropped_type_set: HashSet<&str> =
            dropped_types.iter().map(String::as_str).collect();
        for qual in cat.field_ids.keys() {
            if cat.tombstoned_field_quals.contains(qual.as_str()) {
                continue;
            }
            let (t, _) = split_qualified(qual);
            if dropped_type_set.contains(t) && !dropped_fields.contains(qual) {
                cascaded_fields.insert(qual.clone());
            }
        }
        for qual in cat.rel_ids.keys() {
            if cat.tombstoned_rel_quals.contains(qual.as_str()) {
                continue;
            }
            let (t, _) = split_qualified(qual);
            if dropped_type_set.contains(t) && !dropped_rels.contains(qual) {
                cascaded_rels.insert(qual.clone());
            }
        }
    }

    // PASS 2 — detect kind changes.
    //
    // A kind mismatch is normally a fatal `FieldKindChanged` (the operator
    // changed a field's type without a migration verb). BUT during an
    // in-flight chunked field-type migration the catalog kind is still the
    // SOURCE while the operator has reopened with the TARGET schema — the
    // legitimate "resume me" state. Recognise a *drivable* plan migrating
    // exactly `cat_kind -> want_kind` for this field and accept it; the
    // open path's `auto_resume_migrations` arms the double-write hook and
    // (once the converter is registered) drives the cutover that flips the
    // catalog to match. Only a drivable plan licenses the mismatch — a `Failed` /
    // `AwaitingConverter` plan does NOT (it's parked, not in motion).
    let migration_plans = scan_migration_plans(storage, txn.snapshot())?;
    for (qual, &cat_kind) in &cat.field_kinds {
        let (t, f) = split_qualified(qual);
        let Some(td) = schema.types.get(t) else {
            continue;
        };
        let Some(fd) = td.fields.iter().find(|x| x.name == f) else {
            continue;
        };
        let want_kind = schema_kind_byte(&fd.field_type);
        if cat_kind != want_kind
            && resumable_plan_for_kind_change(&migration_plans, t, f, cat_kind, want_kind).is_none()
        {
            return Err(EngineError::Catalog(CatalogError::FieldKindChanged {
                qualified: qual.clone(),
                was: kind_name(cat_kind),
                now: kind_name(want_kind),
            }));
        }
    }

    // PASS 3 — additive allocation. Deterministic order so two racing
    // additive opens with the same schema delta compute identical IDs.
    let mut puts: Vec<(Bytes, Bytes)> = Vec::new();

    let mut new_type_names: Vec<String> = schema
        .types
        .keys()
        .filter(|t| !cat.type_ids.contains_key(t.as_str()))
        .cloned()
        .collect();
    new_type_names.sort();
    for name in &new_type_names {
        let id = cat.next_type;
        cat.next_type = checked_bump("type", cat.next_type)?;
        let entry = IdEntry::fresh(id, kind_byte::UNSET, now_ms);
        puts.push((KeyBuilder::catalog_type(name), encode_id_entry(&entry)));
        cat.insert_type(name.clone(), entry);
    }

    let mut all_type_names: Vec<&String> = schema.types.keys().collect();
    all_type_names.sort();
    for type_name in all_type_names {
        let td = &schema.types[type_name.as_str()];
        for field in &td.fields {
            let qual = format!("{}.{}", type_name, field.name);
            let kind = schema_kind_byte(&field.field_type);

            if !cat.field_ids.contains_key(&qual) {
                let id = cat.next_field;
                cat.next_field = checked_bump("field", cat.next_field)?;
                let entry = IdEntry::fresh(id, kind, now_ms);
                puts.push((
                    KeyBuilder::catalog_field(type_name, &field.name),
                    encode_id_entry(&entry),
                ));
                cat.insert_field(qual.clone(), entry);
            }

            if matches!(field.field_type, FieldType::Relation(_))
                && !cat.rel_ids.contains_key(&qual)
            {
                let id = cat.next_rel;
                cat.next_rel = checked_bump("rel", cat.next_rel)?;
                let entry = IdEntry::fresh(id, kind_byte::RELATION, now_ms);
                puts.push((
                    KeyBuilder::catalog_rel(type_name, &field.name),
                    encode_id_entry(&entry),
                ));
                cat.insert_rel(qual.clone(), entry);
            }
        }
    }

    // PASS 4 — write tombstones. For each dropped (or cascaded) entry,
    // rewrite its catalog row with `status = Tombstoned` and the
    // retirement metadata TLVs. Re-encoding preserves `unknown_tlvs`
    // verbatim so a future-format binary's payload survives.
    let mut bumped_format = false;
    for name in &dropped_types {
        if let Some(entry) = cat.type_entries.get_mut(name) {
            entry.tombstone(now_ms, RetireReason::ExplicitShrink);
            cat.tombstoned_type_ids.insert(entry.id);
            cat.tombstoned_type_names.insert(name.clone());
            puts.push((
                KeyBuilder::catalog_type(name),
                encode_id_entry(entry),
            ));
            bumped_format = true;
        }
    }
    for qual in &dropped_fields {
        if let Some(entry) = cat.field_entries.get_mut(qual) {
            entry.tombstone(now_ms, RetireReason::ExplicitShrink);
            cat.tombstoned_field_ids.insert(entry.id);
            cat.tombstoned_field_quals.insert(qual.clone());
            let (t, f) = split_qualified(qual);
            puts.push((KeyBuilder::catalog_field(t, f), encode_id_entry(entry)));
            bumped_format = true;
        }
    }
    for qual in &dropped_rels {
        if let Some(entry) = cat.rel_entries.get_mut(qual) {
            entry.tombstone(now_ms, RetireReason::ExplicitShrink);
            cat.tombstoned_rel_ids.insert(entry.id);
            cat.tombstoned_rel_quals.insert(qual.clone());
            let (t, f) = split_qualified(qual);
            puts.push((KeyBuilder::catalog_rel(t, f), encode_id_entry(entry)));
            bumped_format = true;
        }
    }
    // Cascade-tagged fields/relations whose parent type was dropped.
    let mut cascaded_fields_sorted: Vec<&String> = cascaded_fields.iter().collect();
    cascaded_fields_sorted.sort();
    for qual in cascaded_fields_sorted {
        if let Some(entry) = cat.field_entries.get_mut(qual.as_str()) {
            entry.tombstone(now_ms, RetireReason::CascadeParentRetired);
            cat.tombstoned_field_ids.insert(entry.id);
            cat.tombstoned_field_quals.insert(qual.clone());
            let (t, f) = split_qualified(qual);
            puts.push((KeyBuilder::catalog_field(t, f), encode_id_entry(entry)));
            bumped_format = true;
        }
    }
    let mut cascaded_rels_sorted: Vec<&String> = cascaded_rels.iter().collect();
    cascaded_rels_sorted.sort();
    for qual in cascaded_rels_sorted {
        if let Some(entry) = cat.rel_entries.get_mut(qual.as_str()) {
            entry.tombstone(now_ms, RetireReason::CascadeParentRetired);
            cat.tombstoned_rel_ids.insert(entry.id);
            cat.tombstoned_rel_quals.insert(qual.clone());
            let (t, f) = split_qualified(qual);
            puts.push((KeyBuilder::catalog_rel(t, f), encode_id_entry(entry)));
            bumped_format = true;
        }
    }

    // Bump the catalog format version to v2 the first time we write a
    // tombstone. A v1 catalog that never shrinks stays at v1, so a
    // future v1 binary can still open it.
    if bumped_format && cat.format_version < CATALOG_FORMAT_V2 {
        cat.format_version = CATALOG_FORMAT_V2;
        puts.push((
            KeyBuilder::catalog_format(),
            encode_format_version(CATALOG_FORMAT_V2),
        ));
    }

    // Always refresh counters and digest. Two consequences:
    //   (a) Any future open with the same schema takes the digest
    //       fast-path and skips reconcile entirely.
    //   (b) The c:D: write makes EVERY reconcile commit's write-set
    //       overlap with every concurrent reconcile's write-set, so
    //       the LSM's write-write conflict detection serialises us
    //       cleanly even when the additive deltas are disjoint.
    puts.push((
        KeyBuilder::catalog_next_type(),
        encode_counter(cat.next_type),
    ));
    puts.push((
        KeyBuilder::catalog_next_field(),
        encode_counter(cat.next_field),
    ));
    puts.push((KeyBuilder::catalog_next_rel(), encode_counter(cat.next_rel)));
    puts.push((
        KeyBuilder::catalog_digest(),
        Bytes::copy_from_slice(new_digest),
    ));

    storage.put_batch(txn, &puts)?;
    Ok(())
}

// =====================================================================
// SCHEMA DIGEST
// =====================================================================

pub(crate) fn compute_schema_digest(schema: &Schema) -> [u8; 32] {
    let mut hasher = Sha256::new();
    let mut type_names: Vec<&String> = schema.types.keys().collect();
    type_names.sort();
    hasher.update((type_names.len() as u32).to_be_bytes());
    for type_name in type_names {
        let td = &schema.types[type_name.as_str()];
        hasher.update((type_name.len() as u32).to_be_bytes());
        hasher.update(type_name.as_bytes());
        hasher.update((td.fields.len() as u32).to_be_bytes());
        for field in &td.fields {
            hasher.update((field.name.len() as u32).to_be_bytes());
            hasher.update(field.name.as_bytes());
            hasher.update([schema_kind_byte(&field.field_type)]);

            // Directive flags: bit 0 indexed, bit 1 unique, bit 2 inverse,
            // bit 3 vectorize. Other directives (on_delete) are not part
            // of the digest — they don't affect catalog allocation.
            let mut flags: u8 = 0;
            if field.is_indexed() {
                flags |= 0b0001;
            }
            if field.is_unique() {
                flags |= 0b0010;
            }
            if field.inverse().is_some() {
                flags |= 0b0100;
            }
            if field.vectorize().is_some() {
                flags |= 0b1000;
            }
            hasher.update([flags]);

            // For relations, include the target type name so a target
            // swap surfaces as a digest change (which forces reconcile,
            // which detects the kind change via the kind byte today —
            // and via target-type TLVs in card 4/5).
            if let FieldType::Relation(rel) = &field.field_type {
                hasher.update((rel.target_type.len() as u32).to_be_bytes());
                hasher.update(rel.target_type.as_bytes());
                hasher.update([rel.is_many as u8]);
            }
        }
    }
    hasher.finalize().into()
}

// =====================================================================
// ENCODING / DECODING
// =====================================================================

fn encode_id_entry(entry: &IdEntry) -> Bytes {
    let mut body: Vec<u8> = Vec::with_capacity(64);
    write_tlv(&mut body, TLV_ID, &entry.id.to_be_bytes());
    write_tlv(&mut body, TLV_ASSIGNED_AT, &entry.assigned_at.to_be_bytes());
    write_tlv(&mut body, TLV_ASSIGNED_BY, &[entry.assigned_by]);
    write_tlv(&mut body, TLV_KIND, &[entry.kind]);

    // Tombstone TLVs are only written when `status == Tombstoned`. A
    // live entry under v2 is byte-identical to a v1 entry, preserving
    // the digest fast-path for DBs that never shrink.
    if entry.status == TombstoneStatus::Tombstoned {
        write_tlv(&mut body, TLV_STATUS, &[STATUS_TOMBSTONED]);
        if let Some(ms) = entry.retired_at_ms {
            write_tlv(&mut body, TLV_RETIRED_AT_MS, &ms.to_be_bytes());
        }
        if let Some(reason) = entry.retired_reason {
            let reason_byte = match reason {
                RetireReason::ExplicitShrink => REASON_EXPLICIT_SHRINK,
                RetireReason::CascadeParentRetired => REASON_CASCADE_PARENT_RETIRED,
            };
            write_tlv(&mut body, TLV_RETIRED_REASON, &[reason_byte]);
        }
    }

    // Rename TLVs are only written when the row has a non-empty rename
    // chain. A never-renamed entry under v3 is byte-identical to a v2
    // entry, preserving the digest fast-path for DBs that never rename.
    if !entry.previous_names.is_empty() {
        let mut chain_buf: Vec<u8> = Vec::with_capacity(entry.previous_names.len() * 32);
        debug_assert!(entry.previous_names.len() <= MAX_RENAME_HISTORY);
        chain_buf.push(entry.previous_names.len() as u8);
        for r in &entry.previous_names {
            chain_buf.extend_from_slice(&(r.from.len() as u16).to_be_bytes());
            chain_buf.extend_from_slice(r.from.as_bytes());
            chain_buf.extend_from_slice(&(r.to.len() as u16).to_be_bytes());
            chain_buf.extend_from_slice(r.to.as_bytes());
            chain_buf.extend_from_slice(&r.wall_time_unix_ms.to_be_bytes());
        }
        write_tlv(&mut body, TLV_PREVIOUS_NAMES, &chain_buf);
        if let Some(ms) = entry.last_renamed_at_ms {
            write_tlv(&mut body, TLV_LAST_RENAMED_AT_MS, &ms.to_be_bytes());
        }
    }

    // Type-change TLVs only written when the row has a non-empty
    // type-change chain. A never-migrated field under v4 is byte-
    // identical to a v3 entry — the digest fast-path stays hot.
    if !entry.type_change_history.is_empty() {
        let mut chain_buf: Vec<u8> =
            Vec::with_capacity(1 + entry.type_change_history.len() * 10);
        debug_assert!(entry.type_change_history.len() <= MAX_TYPE_CHANGE_HISTORY);
        chain_buf.push(entry.type_change_history.len() as u8);
        for r in &entry.type_change_history {
            chain_buf.push(r.from_kind);
            chain_buf.push(r.to_kind);
            chain_buf.extend_from_slice(&r.wall_time_unix_ms.to_be_bytes());
        }
        write_tlv(&mut body, TLV_TYPE_CHANGE_HISTORY, &chain_buf);
        if let Some(ms) = entry.last_type_change_at_ms {
            write_tlv(&mut body, TLV_LAST_TYPE_CHANGE_AT_MS, &ms.to_be_bytes());
        }
    }

    for (tag, value) in &entry.unknown_tlvs {
        write_tlv(&mut body, *tag, value);
    }
    debug_assert!(body.len() <= u16::MAX as usize);

    let mut out: Vec<u8> = Vec::with_capacity(4 + body.len());
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_ID_ENTRY);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(&body);
    Bytes::from(out)
}

fn write_tlv(out: &mut Vec<u8>, tag: u8, value: &[u8]) {
    out.push(tag);
    out.extend_from_slice(&(value.len() as u16).to_be_bytes());
    out.extend_from_slice(value);
}

fn decode_id_entry(key_debug: &str, bytes: &[u8]) -> EngineResult<IdEntry> {
    if bytes.len() < 4 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 4,
        }));
    }
    if bytes[0] != RECORD_FORMAT_V1 {
        return Err(EngineError::Catalog(
            CatalogError::UnsupportedRecordFormat {
                row: key_debug.into(),
                got: bytes[0],
                max_supported: RECORD_FORMAT_V1,
            },
        ));
    }
    match bytes[1] {
        KIND_ID_ENTRY => {}
        KIND_COUNTER | KIND_MARKER | KIND_MIGRATION_PLAN => {
            return Err(EngineError::Catalog(CatalogError::WrongValueKind {
                key_debug: key_debug.into(),
                expected: "IdEntry",
                got_tag: bytes[1],
            }));
        }
        other => {
            return Err(EngineError::Catalog(CatalogError::UnknownValueKind {
                key_debug: key_debug.into(),
                tag: other,
            }));
        }
    }
    let body_len = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    if bytes.len() < 4 + body_len {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 4 + body_len,
        }));
    }
    let body = &bytes[4..4 + body_len];

    let mut id: Option<u64> = None;
    let mut assigned_at: Option<u64> = None;
    let mut assigned_by: Option<u8> = None;
    let mut kind: Option<u8> = None;
    let mut status: TombstoneStatus = TombstoneStatus::Live;
    let mut retired_at_ms: Option<u64> = None;
    let mut retired_reason: Option<RetireReason> = None;
    let mut previous_names: Vec<RenameRecord> = Vec::new();
    let mut last_renamed_at_ms: Option<u64> = None;
    let mut type_change_history: Vec<TypeChangeRecord> = Vec::new();
    let mut last_type_change_at_ms: Option<u64> = None;
    let mut unknown_tlvs: Vec<(u8, Bytes)> = Vec::new();
    let mut seen_tags: u128 = 0; // bitset for duplicate detection (tags are u8)

    let mut cur = 0;
    while cur < body.len() {
        if body.len() - cur < 3 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: bytes.len(),
                min: 4 + cur + 3,
            }));
        }
        let tag = body[cur];
        let len = u16::from_be_bytes([body[cur + 1], body[cur + 2]]) as usize;
        cur += 3;
        if body.len() - cur < len {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: bytes.len(),
                min: 4 + cur + len,
            }));
        }
        let value = &body[cur..cur + len];
        cur += len;

        if tag < 128 {
            let bit = 1u128 << tag;
            if seen_tags & bit != 0 {
                return Err(EngineError::Catalog(CatalogError::DuplicateTlv {
                    key_debug: key_debug.into(),
                    tag,
                }));
            }
            seen_tags |= bit;
        }

        match tag {
            TLV_ID => {
                if value.len() != 8 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 8,
                    }));
                }
                id = Some(u64::from_be_bytes(value.try_into().unwrap()));
            }
            TLV_ASSIGNED_AT => {
                if value.len() != 8 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 8,
                    }));
                }
                assigned_at = Some(u64::from_be_bytes(value.try_into().unwrap()));
            }
            TLV_ASSIGNED_BY => {
                if value.len() != 1 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 1,
                    }));
                }
                assigned_by = Some(value[0]);
            }
            TLV_KIND => {
                if value.len() != 1 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 1,
                    }));
                }
                kind = Some(value[0]);
            }
            TLV_STATUS => {
                if value.len() != 1 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 1,
                    }));
                }
                status = match value[0] {
                    STATUS_LIVE => TombstoneStatus::Live,
                    STATUS_TOMBSTONED => TombstoneStatus::Tombstoned,
                    other => {
                        return Err(EngineError::Catalog(
                            CatalogError::UnknownTombstoneStatus {
                                row: key_debug.into(),
                                status: other,
                            },
                        ));
                    }
                };
            }
            TLV_RETIRED_AT_MS => {
                if value.len() != 8 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 8,
                    }));
                }
                retired_at_ms = Some(u64::from_be_bytes(value.try_into().unwrap()));
            }
            TLV_RETIRED_REASON => {
                if value.len() != 1 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 1,
                    }));
                }
                retired_reason = Some(match value[0] {
                    REASON_EXPLICIT_SHRINK => RetireReason::ExplicitShrink,
                    REASON_CASCADE_PARENT_RETIRED => RetireReason::CascadeParentRetired,
                    other => {
                        return Err(EngineError::Catalog(
                            CatalogError::UnknownRetireReason {
                                row: key_debug.into(),
                                reason: other,
                            },
                        ));
                    }
                });
            }
            TLV_PREVIOUS_NAMES => {
                previous_names = decode_rename_chain(key_debug, value)?;
            }
            TLV_LAST_RENAMED_AT_MS => {
                if value.len() != 8 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 8,
                    }));
                }
                last_renamed_at_ms = Some(u64::from_be_bytes(value.try_into().unwrap()));
            }
            TLV_TYPE_CHANGE_HISTORY => {
                type_change_history = decode_type_change_chain(key_debug, value)?;
            }
            TLV_LAST_TYPE_CHANGE_AT_MS => {
                if value.len() != 8 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 8,
                    }));
                }
                last_type_change_at_ms =
                    Some(u64::from_be_bytes(value.try_into().unwrap()));
            }
            other => {
                unknown_tlvs.push((other, Bytes::copy_from_slice(value)));
            }
        }
    }

    let id = id.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredTlv {
            key_debug: key_debug.into(),
            tag: TLV_ID,
        })
    })?;
    let assigned_at = assigned_at.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredTlv {
            key_debug: key_debug.into(),
            tag: TLV_ASSIGNED_AT,
        })
    })?;
    let assigned_by = assigned_by.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredTlv {
            key_debug: key_debug.into(),
            tag: TLV_ASSIGNED_BY,
        })
    })?;
    let kind = kind.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredTlv {
            key_debug: key_debug.into(),
            tag: TLV_KIND,
        })
    })?;

    // Internal-consistency check: a Tombstoned row MUST carry
    // retired_at_ms and retired_reason. Encoders only produce that
    // combination together; a missing one indicates either external
    // tampering or a torn write that landed the status but not the
    // payload. Refuse rather than guess.
    if status == TombstoneStatus::Tombstoned {
        if retired_at_ms.is_none() {
            return Err(EngineError::Catalog(CatalogError::MissingRequiredTlv {
                key_debug: key_debug.into(),
                tag: TLV_RETIRED_AT_MS,
            }));
        }
        if retired_reason.is_none() {
            return Err(EngineError::Catalog(CatalogError::MissingRequiredTlv {
                key_debug: key_debug.into(),
                tag: TLV_RETIRED_REASON,
            }));
        }
    }

    Ok(IdEntry {
        id,
        assigned_at,
        assigned_by,
        kind,
        status,
        retired_at_ms,
        retired_reason,
        previous_names,
        last_renamed_at_ms,
        type_change_history,
        last_type_change_at_ms,
        unknown_tlvs,
    })
}

fn encode_counter(value: u64) -> Bytes {
    let mut out: Vec<u8> = Vec::with_capacity(10);
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_COUNTER);
    out.extend_from_slice(&value.to_be_bytes());
    Bytes::from(out)
}

fn decode_counter(key_debug: &str, bytes: &[u8]) -> EngineResult<u64> {
    if bytes.len() < 10 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 10,
        }));
    }
    if bytes[0] != RECORD_FORMAT_V1 {
        return Err(EngineError::Catalog(
            CatalogError::UnsupportedRecordFormat {
                row: key_debug.into(),
                got: bytes[0],
                max_supported: RECORD_FORMAT_V1,
            },
        ));
    }
    if bytes[1] != KIND_COUNTER {
        return Err(EngineError::Catalog(CatalogError::WrongValueKind {
            key_debug: key_debug.into(),
            expected: "Counter",
            got_tag: bytes[1],
        }));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[2..10]);
    Ok(u64::from_be_bytes(buf))
}

fn encode_marker(now_ms: u64) -> Bytes {
    let mut out: Vec<u8> = Vec::with_capacity(10);
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_MARKER);
    out.extend_from_slice(&now_ms.to_be_bytes());
    Bytes::from(out)
}

fn encode_format_version(v: u64) -> Bytes {
    Bytes::copy_from_slice(&v.to_be_bytes())
}

fn decode_format_version(key_debug: &str, bytes: &[u8]) -> EngineResult<u64> {
    if bytes.len() != 8 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 8,
        }));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(bytes);
    Ok(u64::from_be_bytes(buf))
}

// =====================================================================
// Shadow-field migration plan codec + id allocation (card 1/5)
// =====================================================================

fn encode_migration_plan(plan: &MigrationPlan) -> Bytes {
    let mut body: Vec<u8> = Vec::with_capacity(128);
    write_tlv(&mut body, TLV_MP_TYPE_NAME, plan.type_name.as_bytes());
    write_tlv(&mut body, TLV_MP_FIELD_NAME, plan.field_name.as_bytes());
    write_tlv(&mut body, TLV_MP_FIELD_ID, &plan.field_id.to_be_bytes());
    write_tlv(&mut body, TLV_MP_SRC_KIND, &[plan.src_kind]);
    write_tlv(&mut body, TLV_MP_TARGET_KIND, &[plan.target_kind]);
    write_tlv(&mut body, TLV_MP_STATUS, &[plan.status.to_byte()]);
    write_tlv(&mut body, TLV_MP_CURSOR, &plan.cursor.to_be_bytes());
    write_tlv(&mut body, TLV_MP_CHUNK_SIZE, &plan.chunk_size.to_be_bytes());
    write_tlv(&mut body, TLV_MP_CREATED_AT_MS, &plan.created_at_ms.to_be_bytes());
    write_tlv(&mut body, TLV_MP_CONVERTER_NAME, plan.converter_name.as_bytes());
    write_tlv(
        &mut body,
        TLV_MP_CONVERTER_VERSION,
        &plan.converter_version.to_be_bytes(),
    );
    write_tlv(
        &mut body,
        TLV_MP_OBJECTS_CONVERTED,
        &plan.objects_converted.to_be_bytes(),
    );
    // Card 2: phase + cutover cursor.
    write_tlv(&mut body, TLV_MP_PHASE, &[plan.phase.to_byte()]);
    write_tlv(
        &mut body,
        TLV_MP_CUTOVER_CURSOR,
        &plan.cutover_cursor.to_be_bytes(),
    );
    // Card 3: parallel degree (only when parallel) + id upper bound.
    if let Some(n) = plan.parallel_degree {
        write_tlv(&mut body, TLV_MP_PARALLEL_DEGREE, &[n]);
        write_tlv(
            &mut body,
            TLV_MP_ID_UPPER_BOUND,
            &plan.id_upper_bound.to_be_bytes(),
        );
    }
    // Card 4: error policy / dry-run / error count / quarantine cap (always
    // written; absent on a card-1/2/3 row → decode to Stop/false/0/default).
    write_tlv(&mut body, TLV_MP_ERROR_POLICY, &[plan.error_policy.to_byte()]);
    write_tlv(&mut body, TLV_MP_DRY_RUN, &[plan.dry_run as u8]);
    write_tlv(&mut body, TLV_MP_ERROR_COUNT, &plan.error_count.to_be_bytes());
    write_tlv(
        &mut body,
        TLV_MP_QUARANTINE_CAP,
        &plan.quarantine_cap.to_be_bytes(),
    );
    // Preserve forward-compat tags verbatim.
    for (tag, value) in &plan.unknown_tlvs {
        write_tlv(&mut body, *tag, value);
    }
    debug_assert!(body.len() <= u16::MAX as usize);

    let mut out: Vec<u8> = Vec::with_capacity(4 + body.len());
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_MIGRATION_PLAN);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(&body);
    Bytes::from(out)
}

fn decode_migration_plan(plan_id: u64, key_debug: &str, bytes: &[u8]) -> EngineResult<MigrationPlan> {
    if bytes.len() < 4 {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 4,
        }));
    }
    if bytes[0] != RECORD_FORMAT_V1 {
        return Err(EngineError::Catalog(
            CatalogError::UnsupportedRecordFormat {
                row: key_debug.into(),
                got: bytes[0],
                max_supported: RECORD_FORMAT_V1,
            },
        ));
    }
    match bytes[1] {
        KIND_MIGRATION_PLAN => {}
        KIND_ID_ENTRY | KIND_COUNTER | KIND_MARKER => {
            return Err(EngineError::Catalog(CatalogError::WrongValueKind {
                key_debug: key_debug.into(),
                expected: "MigrationPlan",
                got_tag: bytes[1],
            }));
        }
        other => {
            return Err(EngineError::Catalog(CatalogError::UnknownValueKind {
                key_debug: key_debug.into(),
                tag: other,
            }));
        }
    }
    let body_len = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    if bytes.len() < 4 + body_len {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: 4 + body_len,
        }));
    }
    let body = &bytes[4..4 + body_len];

    let mut type_name: Option<String> = None;
    let mut field_name: Option<String> = None;
    let mut field_id: Option<u64> = None;
    let mut src_kind: Option<u8> = None;
    let mut target_kind: Option<u8> = None;
    let mut status: Option<MigrationStatus> = None;
    let mut cursor: Option<u64> = None;
    let mut chunk_size: Option<u64> = None;
    let mut created_at_ms: Option<u64> = None;
    let mut converter_name: Option<String> = None;
    let mut converter_version: Option<u32> = None;
    let mut objects_converted: Option<u64> = None;
    let mut phase: Option<MigrationPhase> = None;
    let mut cutover_cursor: Option<u64> = None;
    let mut parallel_degree: Option<u8> = None;
    let mut id_upper_bound: Option<u64> = None;
    let mut error_policy: Option<ErrorPolicy> = None;
    let mut dry_run: Option<bool> = None;
    let mut error_count: Option<u64> = None;
    let mut quarantine_cap: Option<u64> = None;
    let mut unknown_tlvs: Vec<(u8, Bytes)> = Vec::new();
    let mut seen_tags: u128 = 0;

    let read_u64 = |value: &[u8]| -> EngineResult<u64> {
        if value.len() != 8 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: 8,
            }));
        }
        Ok(u64::from_be_bytes(value.try_into().unwrap()))
    };
    let read_u8 = |value: &[u8]| -> EngineResult<u8> {
        if value.len() != 1 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: 1,
            }));
        }
        Ok(value[0])
    };
    let read_str = |value: &[u8]| -> EngineResult<String> {
        std::str::from_utf8(value)
            .map(|s| s.to_string())
            .map_err(|_| {
                EngineError::Catalog(CatalogError::MalformedMigrationPlan {
                    row: key_debug.into(),
                    reason: "non-utf8 string TLV",
                })
            })
    };

    let mut cur = 0;
    while cur < body.len() {
        if body.len() - cur < 3 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: bytes.len(),
                min: 4 + cur + 3,
            }));
        }
        let tag = body[cur];
        let len = u16::from_be_bytes([body[cur + 1], body[cur + 2]]) as usize;
        cur += 3;
        if body.len() - cur < len {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: bytes.len(),
                min: 4 + cur + len,
            }));
        }
        let value = &body[cur..cur + len];
        cur += len;

        if tag < 128 {
            let bit = 1u128 << tag;
            if seen_tags & bit != 0 {
                return Err(EngineError::Catalog(CatalogError::DuplicateTlv {
                    key_debug: key_debug.into(),
                    tag,
                }));
            }
            seen_tags |= bit;
        }

        match tag {
            TLV_MP_TYPE_NAME => type_name = Some(read_str(value)?),
            TLV_MP_FIELD_NAME => field_name = Some(read_str(value)?),
            TLV_MP_FIELD_ID => field_id = Some(read_u64(value)?),
            TLV_MP_SRC_KIND => src_kind = Some(read_u8(value)?),
            TLV_MP_TARGET_KIND => target_kind = Some(read_u8(value)?),
            TLV_MP_STATUS => {
                let b = read_u8(value)?;
                status = Some(MigrationStatus::from_byte(b).ok_or_else(|| {
                    EngineError::Catalog(CatalogError::UnknownMigrationStatus {
                        row: key_debug.into(),
                        status: b,
                    })
                })?);
            }
            TLV_MP_CURSOR => cursor = Some(read_u64(value)?),
            TLV_MP_CHUNK_SIZE => chunk_size = Some(read_u64(value)?),
            TLV_MP_CREATED_AT_MS => created_at_ms = Some(read_u64(value)?),
            TLV_MP_CONVERTER_NAME => converter_name = Some(read_str(value)?),
            TLV_MP_CONVERTER_VERSION => {
                if value.len() != 4 {
                    return Err(EngineError::Catalog(CatalogError::Truncated {
                        key_debug: key_debug.into(),
                        len: value.len(),
                        min: 4,
                    }));
                }
                converter_version = Some(u32::from_be_bytes(value.try_into().unwrap()));
            }
            TLV_MP_OBJECTS_CONVERTED => objects_converted = Some(read_u64(value)?),
            TLV_MP_PHASE => {
                let b = read_u8(value)?;
                phase = Some(MigrationPhase::from_byte(b).ok_or_else(|| {
                    EngineError::Catalog(CatalogError::MalformedMigrationPlan {
                        row: key_debug.into(),
                        reason: "unknown migration phase byte",
                    })
                })?);
            }
            TLV_MP_CUTOVER_CURSOR => cutover_cursor = Some(read_u64(value)?),
            TLV_MP_PARALLEL_DEGREE => {
                let n = read_u8(value)?;
                // Reject out-of-range degrees rather than decode a plan whose
                // partition math can't be reconstructed (fail closed).
                if n == 0 || n > MAX_PARALLEL_DEGREE {
                    return Err(EngineError::Catalog(CatalogError::MalformedMigrationPlan {
                        row: key_debug.into(),
                        reason: "parallel_degree out of range (1..=64)",
                    }));
                }
                parallel_degree = Some(n);
            }
            TLV_MP_ID_UPPER_BOUND => id_upper_bound = Some(read_u64(value)?),
            TLV_MP_ERROR_POLICY => {
                let b = read_u8(value)?;
                error_policy = Some(ErrorPolicy::from_byte(b).ok_or_else(|| {
                    EngineError::Catalog(CatalogError::MalformedMigrationPlan {
                        row: key_debug.into(),
                        reason: "unknown error policy byte",
                    })
                })?);
            }
            TLV_MP_DRY_RUN => dry_run = Some(read_u8(value)? != 0),
            TLV_MP_ERROR_COUNT => error_count = Some(read_u64(value)?),
            TLV_MP_QUARANTINE_CAP => quarantine_cap = Some(read_u64(value)?),
            other => unknown_tlvs.push((other, Bytes::copy_from_slice(value))),
        }
    }

    let require_tag = |tag: u8| {
        EngineError::Catalog(CatalogError::MissingRequiredTlv {
            key_debug: key_debug.into(),
            tag,
        })
    };
    Ok(MigrationPlan {
        plan_id,
        type_name: type_name.ok_or_else(|| require_tag(TLV_MP_TYPE_NAME))?,
        field_name: field_name.ok_or_else(|| require_tag(TLV_MP_FIELD_NAME))?,
        field_id: field_id.ok_or_else(|| require_tag(TLV_MP_FIELD_ID))?,
        src_kind: src_kind.ok_or_else(|| require_tag(TLV_MP_SRC_KIND))?,
        target_kind: target_kind.ok_or_else(|| require_tag(TLV_MP_TARGET_KIND))?,
        status: status.ok_or_else(|| require_tag(TLV_MP_STATUS))?,
        cursor: cursor.ok_or_else(|| require_tag(TLV_MP_CURSOR))?,
        chunk_size: chunk_size.ok_or_else(|| require_tag(TLV_MP_CHUNK_SIZE))?,
        created_at_ms: created_at_ms.ok_or_else(|| require_tag(TLV_MP_CREATED_AT_MS))?,
        converter_name: converter_name.ok_or_else(|| require_tag(TLV_MP_CONVERTER_NAME))?,
        converter_version: converter_version.ok_or_else(|| require_tag(TLV_MP_CONVERTER_VERSION))?,
        objects_converted: objects_converted.ok_or_else(|| require_tag(TLV_MP_OBJECTS_CONVERTED))?,
        // Card-1 rows have no phase/cutover TLV → default to the pre-card-2
        // semantics (Converting, cutover not started).
        phase: phase.unwrap_or(MigrationPhase::Converting),
        cutover_cursor: cutover_cursor.unwrap_or(0),
        // Card 3: absent → legacy single-worker plan (parallel_degree None,
        // id_upper_bound 0). `parallel_degree` presence is the discriminator.
        parallel_degree,
        id_upper_bound: id_upper_bound.unwrap_or(0),
        // Card 4: absent on a card-1/2/3 row → Stop / not-dry-run / 0 errors /
        // default cap.
        error_policy: error_policy.unwrap_or(ErrorPolicy::Stop),
        dry_run: dry_run.unwrap_or(false),
        error_count: error_count.unwrap_or(0),
        quarantine_cap: quarantine_cap.unwrap_or(DEFAULT_QUARANTINE_CAP),
        unknown_tlvs,
    })
}

/// Read the `c:N:M` migration-id counter within `txn`; absent ⇒ 0.
fn read_migration_counter(
    storage: &LsmTree,
    txn: &rhypedb_storage::mvcc::Transaction,
) -> EngineResult<u64> {
    match storage.get(txn, &KeyBuilder::catalog_next_migration())? {
        Some(bytes) => decode_counter("c:N:M", &bytes),
        None => Ok(0),
    }
}

// =====================================================================
// Per-partition migration cursor (`c:S:<plan><idx>`, card 3/5)
// =====================================================================

/// One parallel backfill partition's durable progress. A worker advances ONLY
/// its own partition's cursor (disjoint `c:S:` key → no inter-worker conflict);
/// the plan-level aggregate is summed over partitions at finalize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartitionCursor {
    /// Highest `object_id` in this partition whose shadow is durably committed.
    /// `0` = not started; the worker resumes from `max(cursor + 1, lo)`.
    pub cursor: u64,
    /// Objects this partition has converted (observability + skew surfacing).
    pub objects_converted: u64,
    /// Card 4: rows in this partition whose converter FAILED under a non-Stop
    /// policy (SkipAndLog/Quarantine). Committed atomically with the cursor
    /// advance (same chunk batch), so a torn/conflicted chunk drops the count
    /// alongside its converts — a re-scan recomputes it idempotently (the
    /// re-scan-proof analogue of `objects_converted`, NOT a free-running counter).
    pub errors: u64,
    /// True once this partition's whole `[lo, hi)` range is exhausted.
    pub done: bool,
}

/// Fixed framed layout: `[RECORD_FORMAT_V1][KIND_PARTITION_CURSOR][cursor u64 BE]
/// [objects_converted u64 BE][errors u64 BE][done u8]` (card-4 v2; the card-3
/// 19-byte v1 is unshipped on this branch so there is no compat to keep). A torn
/// write decode-FAILS cleanly (wrong length / kind / done byte) rather than
/// silently misparsing the cursor.
const PARTITION_CURSOR_LEN: usize = 2 + 8 + 8 + 8 + 1;

pub(crate) fn encode_partition_cursor(pc: &PartitionCursor) -> Bytes {
    let mut out = Vec::with_capacity(PARTITION_CURSOR_LEN);
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_PARTITION_CURSOR);
    out.extend_from_slice(&pc.cursor.to_be_bytes());
    out.extend_from_slice(&pc.objects_converted.to_be_bytes());
    out.extend_from_slice(&pc.errors.to_be_bytes());
    out.push(pc.done as u8);
    Bytes::from(out)
}

pub(crate) fn decode_partition_cursor(
    key_debug: &str,
    bytes: &[u8],
) -> EngineResult<PartitionCursor> {
    if bytes.len() != PARTITION_CURSOR_LEN {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: bytes.len(),
            min: PARTITION_CURSOR_LEN,
        }));
    }
    if bytes[0] != RECORD_FORMAT_V1 {
        return Err(EngineError::Catalog(CatalogError::UnsupportedRecordFormat {
            row: key_debug.into(),
            got: bytes[0],
            max_supported: RECORD_FORMAT_V1,
        }));
    }
    if bytes[1] != KIND_PARTITION_CURSOR {
        return Err(EngineError::Catalog(CatalogError::WrongValueKind {
            key_debug: key_debug.into(),
            expected: "PartitionCursor",
            got_tag: bytes[1],
        }));
    }
    let cursor = u64::from_be_bytes(bytes[2..10].try_into().unwrap());
    let objects_converted = u64::from_be_bytes(bytes[10..18].try_into().unwrap());
    let errors = u64::from_be_bytes(bytes[18..26].try_into().unwrap());
    let done = match bytes[26] {
        0 => false,
        1 => true,
        _ => {
            return Err(EngineError::Catalog(CatalogError::MalformedMigrationPlan {
                row: key_debug.into(),
                reason: "partition cursor done byte not in {0,1}",
            }));
        }
    };
    Ok(PartitionCursor {
        cursor,
        objects_converted,
        errors,
        done,
    })
}

// =====================================================================
// Quarantine sidecar (`c:Q:<plan><object_id>`, card 4/5)
// =====================================================================

/// Cap on the stored converter error message (bytes). Keeps a runaway error
/// string from bloating the sidecar; the operator gets the gist.
const MAX_QUARANTINE_ERROR_MSG: usize = 1024;

/// A decoded quarantine record. `source_value` is the serialized 1-field
/// `FieldMap` `{field_name: <source value>}` (reuses the object codec) so a
/// retry can recover the exact value the converter choked on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QuarantineRecord {
    pub source_value: Bytes,
    pub errored_at_ms: u64,
    pub error_msg: String,
    pub attempted_converter_name: String,
}

/// Framed layout: `[fmt][KIND_QUARANTINE][errored_at_ms u64 BE]
/// [source_value_len u32 BE][source_value][error_msg_len u16 BE][error_msg]
/// [converter_name_len u16 BE][converter_name]`. The error message is truncated
/// to `MAX_QUARANTINE_ERROR_MSG` bytes (on a char boundary).
pub(crate) fn encode_quarantine_record(
    source_value: &[u8],
    errored_at_ms: u64,
    error_msg: &str,
    attempted_converter_name: &str,
) -> Bytes {
    let mut msg = error_msg;
    if msg.len() > MAX_QUARANTINE_ERROR_MSG {
        // Truncate on a UTF-8 char boundary at or below the cap.
        let mut end = MAX_QUARANTINE_ERROR_MSG;
        while end > 0 && !msg.is_char_boundary(end) {
            end -= 1;
        }
        msg = &msg[..end];
    }
    let cname = attempted_converter_name.as_bytes();
    let mut out =
        Vec::with_capacity(2 + 8 + 4 + source_value.len() + 2 + msg.len() + 2 + cname.len());
    out.push(RECORD_FORMAT_V1);
    out.push(KIND_QUARANTINE);
    out.extend_from_slice(&errored_at_ms.to_be_bytes());
    out.extend_from_slice(&(source_value.len() as u32).to_be_bytes());
    out.extend_from_slice(source_value);
    out.extend_from_slice(&(msg.len() as u16).to_be_bytes());
    out.extend_from_slice(msg.as_bytes());
    out.extend_from_slice(&(cname.len() as u16).to_be_bytes());
    out.extend_from_slice(cname);
    Bytes::from(out)
}

pub(crate) fn decode_quarantine_record(
    key_debug: &str,
    bytes: &[u8],
) -> EngineResult<QuarantineRecord> {
    let malformed = |reason: &'static str| {
        EngineError::Catalog(CatalogError::MalformedMigrationPlan {
            row: key_debug.into(),
            reason,
        })
    };
    if bytes.len() < 2 + 8 + 4 {
        return Err(malformed("quarantine record too short"));
    }
    if bytes[0] != RECORD_FORMAT_V1 {
        return Err(EngineError::Catalog(CatalogError::UnsupportedRecordFormat {
            row: key_debug.into(),
            got: bytes[0],
            max_supported: RECORD_FORMAT_V1,
        }));
    }
    if bytes[1] != KIND_QUARANTINE {
        return Err(EngineError::Catalog(CatalogError::WrongValueKind {
            key_debug: key_debug.into(),
            expected: "QuarantineRecord",
            got_tag: bytes[1],
        }));
    }
    let errored_at_ms = u64::from_be_bytes(bytes[2..10].try_into().unwrap());
    let mut cur = 10;
    let mut read_lp = |width: usize| -> EngineResult<&[u8]> {
        if bytes.len() - cur < width {
            return Err(malformed("quarantine length-prefix truncated"));
        }
        let n = match width {
            4 => u32::from_be_bytes(bytes[cur..cur + 4].try_into().unwrap()) as usize,
            2 => u16::from_be_bytes(bytes[cur..cur + 2].try_into().unwrap()) as usize,
            _ => unreachable!(),
        };
        cur += width;
        if bytes.len() - cur < n {
            return Err(malformed("quarantine field truncated"));
        }
        let s = &bytes[cur..cur + n];
        cur += n;
        Ok(s)
    };
    let source_value = Bytes::copy_from_slice(read_lp(4)?);
    let error_msg = std::str::from_utf8(read_lp(2)?)
        .map_err(|_| malformed("quarantine error_msg not utf-8"))?
        .to_string();
    let attempted_converter_name = std::str::from_utf8(read_lp(2)?)
        .map_err(|_| malformed("quarantine converter_name not utf-8"))?
        .to_string();
    Ok(QuarantineRecord {
        source_value,
        errored_at_ms,
        error_msg,
        attempted_converter_name,
    })
}

/// Serialize a single field's value as a 1-entry `FieldMap` blob (reusing the
/// object codec) so the quarantine sidecar can store + later recover the exact
/// source value the converter failed on. Returns `None` if the field is absent.
fn single_field_value_blob(blob: &[u8], field_name: &str) -> Option<Bytes> {
    let fields = crate::object::deserialize_fields(blob);
    let value = fields.get(field_name)?.clone();
    let mut one = crate::object::FieldMap::new();
    one.insert(field_name.to_string(), value);
    Some(crate::object::serialize_fields(&one))
}

/// True for the ONE per-row error class governed by `ErrorPolicy` — a converter
/// that returned `Err` on an otherwise-valid row. A structural
/// `MigrationRowUnexpectedKind` (on-disk vs catalog kind disagreement) or
/// `FieldTypeChangeConverterReturnedWrongKind` (converter-contract violation)
/// signals a setup/programming defect, not a bad row, and ALWAYS halts.
fn is_policy_governed_error(e: &EngineError) -> bool {
    matches!(
        e,
        EngineError::Catalog(CatalogError::FieldTypeChangeConverterFailed { .. })
    )
}

/// Would the cutover still treat this object's migrating field as an UNCONVERTED
/// source row (card 4)? True iff the source is present, at `src_kind`, AND has no
/// current-version `<field>__shadow`. A row whose field became Null/target-kind,
/// or which gained a current shadow (via the double-write hook self-healing it,
/// or `retry_quarantined`), is RESOLVED. Mirrors `run_cutover`'s own per-row
/// decision so the quarantine gate and the cutover agree.
fn quarantine_unresolved(
    blob: &[u8],
    field_name: &str,
    src_kind: u8,
    target_kind: u8,
    converter_version: u32,
) -> bool {
    let fields = crate::object::deserialize_fields(blob);
    let Some(value) = fields.get(field_name) else {
        return false; // field absent → cutover skips it
    };
    let got = value_to_kind_byte(value);
    if got == target_kind || got == kind_byte::UNSET {
        return false; // already target / Null → resolved
    }
    if got != src_kind {
        return false; // some other kind — not a clean src row (cutover handles separately)
    }
    // Source still at src_kind: resolved IFF a current-version shadow exists.
    let shadow_name = format!("{field_name}__shadow");
    let shadow_cv_name = format!("{field_name}__shadow_cv");
    let has_current_shadow = fields.contains_key(&shadow_name)
        && matches!(
            fields.get(&shadow_cv_name),
            Some(crate::object::Value::U32(v)) if *v == converter_version
        );
    !has_current_shadow
}

/// The cutover error gate (card 4): how many of a plan's quarantine rows are
/// still UNRESOLVED, after reaping the resolved ones? Returns `0` for any policy
/// other than `Quarantine` (Stop never reaches cutover with errors; SkipAndLog's
/// errored rows are accepted as source-shape). For `Quarantine` it scans
/// `c:Q:<plan>`, and for each row checks the LIVE object blob: a row whose object
/// gained a current shadow (the double-write hook self-healed it, or
/// `retry_quarantined` ran) is RESOLVED — DELETE its now-stale sidecar — and one
/// still source-at-src-no-shadow is UNRESOLVED. So the gate self-corrects against
/// hook self-heals instead of trusting a free-running counter. Caller holds
/// `migration_lock.write()` (no concurrent `c:Q:`/blob writer).
pub(crate) fn cutover_quarantine_gate(
    storage: &LsmTree,
    plan: &MigrationPlan,
    type_id: u64,
) -> EngineResult<u64> {
    if plan.error_policy != ErrorPolicy::Quarantine {
        return Ok(0);
    }
    let prefix = KeyBuilder::catalog_quarantine_plan_prefix(plan.plan_id);
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();
    let rows = storage.scan_prefix_at(snap, &prefix)?;
    let mut unresolved = 0u64;
    let mut resolved_keys: Vec<Bytes> = Vec::new();
    for (qkey, _) in rows {
        let object_id = object_id_from_key(&qkey);
        let obj_key = KeyBuilder::object(type_id, object_id);
        let is_unresolved = match storage.get_at(snap, &obj_key)? {
            None => false, // object deleted → nothing to convert → resolved
            Some(blob) => quarantine_unresolved(
                &blob,
                &plan.field_name,
                plan.src_kind,
                plan.target_kind,
                plan.converter_version,
            ),
        };
        if is_unresolved {
            unresolved += 1;
        } else {
            resolved_keys.push(qkey);
        }
    }
    if resolved_keys.is_empty() {
        storage.abort(&mut txn);
    } else {
        storage.delete_batch(&mut txn, &resolved_keys)?;
        storage.commit(&mut txn)?;
    }
    Ok(unresolved)
}

/// Re-run a (now-fixed) converter over the named quarantine rows (card 4).
/// For each `object_id`: if there is no `c:Q:` row it is already resolved
/// (skip, not counted); otherwise re-read the LIVE object blob and run
/// `new_converter` via the shared `convert_row_for_backfill` — which is
/// idempotent (a row the double-write hook already shadowed at the current
/// version yields `Ok(None)`, so the hook's write wins) — writing the shadow on
/// success and DELETING the `c:Q:` row. A row whose converter still fails keeps
/// its `c:Q:` row (not counted). Returns the count of rows newly resolved.
/// Caller holds `migration_lock.write()` so this serializes against the
/// read-locked hook + `run_cutover` (no last-write-wins blob clobber). Does NOT
/// change `plan.error_count` (a historical count; the cutover gate reads the live
/// `c:Q:` state instead).
pub(crate) fn retry_quarantined(
    storage: &LsmTree,
    plan: &MigrationPlan,
    type_id: u64,
    ids: &[u64],
    new_converter: &RegisteredConverter,
) -> EngineResult<u64> {
    const WRITE_CONFLICT_RETRIES: u32 = 8;
    let field_name = &plan.field_name;
    let shadow_name = format!("{field_name}__shadow");
    let shadow_cv_name = format!("{field_name}__shadow_cv");
    let mut retried = 0u64;
    for &object_id in ids {
        let qkey = KeyBuilder::catalog_quarantine(plan.plan_id, object_id);
        let obj_key = KeyBuilder::object(type_id, object_id);
        let mut attempts = 0u32;
        let resolved = loop {
            let mut txn = storage.begin_txn();
            if storage.get(&txn, &qkey)?.is_none() {
                storage.abort(&mut txn); // no quarantine row → already resolved
                break false;
            }
            // Decide the write from the LIVE blob (absent object → just drop the
            // stale sidecar; else re-run the converter, idempotent vs a current shadow).
            let new_blob = match storage.get(&txn, &obj_key)? {
                None => None, // object deleted → nothing to convert; clear the sidecar
                Some(blob) => match convert_row_for_backfill(
                    object_id,
                    &blob,
                    &plan.type_name,
                    field_name,
                    &shadow_name,
                    &shadow_cv_name,
                    plan.src_kind,
                    plan.target_kind,
                    plan.converter_version,
                    new_converter,
                    plan.plan_id,
                ) {
                    Ok(Some(nb)) => Some(nb),  // converted → write shadow + clear sidecar
                    Ok(None) => None,          // already current/Null/target → clear sidecar
                    Err(_) => {
                        storage.abort(&mut txn); // still failing → keep the sidecar
                        break false;
                    }
                },
            };
            if let Some(nb) = new_blob {
                storage.put(&mut txn, &obj_key, nb)?;
            }
            storage.delete(&mut txn, &qkey)?;
            match storage.commit(&mut txn) {
                Ok(_) => break true,
                Err(rhypedb_storage::Error::WriteConflict) if attempts < WRITE_CONFLICT_RETRIES => {
                    storage.abort(&mut txn);
                    attempts += 1;
                    continue;
                }
                Err(e) => return Err(EngineError::Storage(e)),
            }
        };
        if resolved {
            retried += 1;
        }
    }
    Ok(retried)
}

/// Delete ALL of a plan's `c:Q:` rows (card 4) — the operator accepts that the
/// remaining quarantined rows stay source-shape; cutover will then leave them.
/// Returns the count deleted. Caller holds `migration_lock.write()`.
pub(crate) fn clear_quarantine(storage: &LsmTree, plan_id: u64) -> EngineResult<u64> {
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();
    let keys: Vec<Bytes> = storage
        .scan_prefix_at(snap, &KeyBuilder::catalog_quarantine_plan_prefix(plan_id))?
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let n = keys.len() as u64;
    if keys.is_empty() {
        storage.abort(&mut txn);
    } else {
        storage.delete_batch(&mut txn, &keys)?;
        storage.commit(&mut txn)?;
    }
    Ok(n)
}

/// Half-open object-id range `[lo, hi)` owned by partition `idx` when the domain
/// `[1, id_upper_bound)` is split into `n` contiguous spans (card 3/5).
/// Deterministic in `(n, id_upper_bound, idx)` so resume recomputes identical
/// boundaries (cursors stay sound). Contiguous so each worker does a pure
/// forward range scan (no read amplification). The union over `idx in 0..n` is
/// exactly `[1, id_upper_bound)` and the ranges are disjoint; a partition past
/// the populated tail (under id skew) is empty (`lo == hi`).
pub(crate) fn partition_range(n: u8, id_upper_bound: u64, idx: u8) -> (u64, u64) {
    let n = n.max(1) as u64;
    let idx = idx as u64;
    let width = id_upper_bound.saturating_sub(1); // count of ids in [1, U)
    if width == 0 {
        let end = id_upper_bound.max(1);
        return (end, end); // empty domain → every partition empty
    }
    let span = width.div_ceil(n).max(1);
    let lo = idx.saturating_mul(span).saturating_add(1).min(id_upper_bound);
    let hi = idx
        .saturating_add(1)
        .saturating_mul(span)
        .saturating_add(1)
        .min(id_upper_bound);
    (lo, hi)
}

/// Decode every `c:P:<id>` plan record visible at `snap`, ascending by id.
pub(crate) fn scan_migration_plans(
    storage: &LsmTree,
    snap: u64,
) -> EngineResult<Vec<MigrationPlan>> {
    let prefix = KeyBuilder::catalog_migration_plan_prefix();
    let mut plans = Vec::new();
    for (key, value) in storage.scan_prefix_at(snap, &prefix)? {
        let id_bytes = &key[prefix.len()..];
        if id_bytes.len() != 8 {
            return Err(EngineError::Catalog(CatalogError::MalformedKey {
                key_debug: debug_key(&key),
            }));
        }
        let plan_id = u64::from_be_bytes(id_bytes.try_into().unwrap());
        let kd = debug_key(&key);
        plans.push(decode_migration_plan(plan_id, &kd, &value)?);
    }
    Ok(plans)
}

/// Allocate the next plan id, self-healing the persisted counter against
/// the highest plan id actually on disk (incl. terminal plans) so a torn
/// counter bump can never reissue a live id. Caller persists
/// `c:N:M = returned id` alongside the new plan in one commit.
fn next_migration_id(counter: u64, plans: &[MigrationPlan]) -> u64 {
    let max_plan = plans.iter().map(|p| p.plan_id).max().unwrap_or(0);
    counter.max(max_plan).saturating_add(1)
}

/// `(qualified, plan_id)` of an *unsettled* (`status.quiesces()`) migration
/// on `type_name` — ANY field. Used by the offline-surface interlock and the
/// chunked-create interlock: a field-type change (offline OR chunked) and a
/// rename are refused while any unsettled plan covers the same TYPE.
///
/// TYPE-scoped, not field-scoped: the migration worker rewrites the WHOLE
/// object blob (deserialize all fields, convert one, re-serialize), so two
/// migrations on different fields of one type — or an offline change on field
/// B while a chunked plan drives field A with `migration_lock` dropped —
/// clobber each other's `o:<type>:*` writes. Settled plans (Completed /
/// Cancelled) don't block — the field is final.
fn active_plan_for_type(plans: &[MigrationPlan], type_name: &str) -> Option<(String, u64)> {
    plans
        .iter()
        .find(|p| p.status.quiesces() && p.type_name == type_name)
        .map(|p| (format!("{}.{}", p.type_name, p.field_name), p.plan_id))
}

/// The plan id of an *unsettled* (`status.quiesces()`) migration on
/// `type_name.field_name` whose src/target kinds exactly match an observed
/// catalog→schema kind delta, if any. Used by reconcile PASS 2 to recognise
/// an in-flight field-type cutover and SKIP the `FieldKindChanged` hard error
/// instead of refusing to open. Licenses the mismatch for ANY resumable plan
/// (`Running`/`Pending`/`Failed`/`AwaitingConverter`) migrating exactly
/// `cat_kind → want_kind`: a `Failed`/`AwaitingConverter` plan still needs the
/// operator to reopen with the TARGET schema to resume it — refusing the open
/// outright would brick the type (it can't be opened with either schema).
fn resumable_plan_for_kind_change(
    plans: &[MigrationPlan],
    type_name: &str,
    field_name: &str,
    cat_kind: u8,
    want_kind: u8,
) -> Option<u64> {
    plans
        .iter()
        .find(|p| {
            p.status.quiesces()
                && p.type_name == type_name
                && p.field_name == field_name
                && p.src_kind == cat_kind
                && p.target_kind == want_kind
        })
        .map(|p| p.plan_id)
}

// =====================================================================
// Chunked field-type migration worker (card 2/5)
//
// The synchronous run-to-completion path: create a durable plan, arm the
// double-write hook, backfill `<field>__shadow` siblings in crash-safe chunks,
// then cut over (promote shadow → source + flip the catalog kind). Writes stay
// ONLINE throughout via the hook. Async / cancel / pause are card 5.
// =====================================================================

/// Outcome of `create_migration_plan`: the durable plan id + its owning
/// type id, so the Database orchestrator can arm the double-write hook and
/// drive without re-loading the catalog.
pub(crate) struct CreatedMigration {
    pub plan_id: u64,
    pub type_id: u64,
}

/// Validate, allocate, and persist a new chunked field-type migration plan
/// in ONE commit (`c:P:<id>` + the `c:N:M` counter). Caller holds
/// `migration_lock.write()`. Refuses if an unsettled plan already covers the
/// field (same interlock the offline path uses). Status starts `Running`,
/// cursor `0`; the caller arms the double-write hook then drives.
// Each arg is an independent plan field with no natural grouping; a params
// struct would just move the noise to the single call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_migration_plan(
    storage: &LsmTree,
    schema: &Schema,
    type_name: &str,
    field_name: &str,
    target_kind: u8,
    converter_name: &str,
    converter_version: u32,
    chunk_size: u64,
    // Card 3: `Some(n)` makes this a parallel plan over `[1, id_upper_bound)`
    // (cursors in `c:S:`); `None` is a legacy single-worker plan (`cursor` field).
    parallel_degree: Option<u8>,
    id_upper_bound: u64,
    // Card 4: per-row failure policy (immutable), dry-run preflight flag, and the
    // quarantine cap (0 → DEFAULT_QUARANTINE_CAP).
    error_policy: ErrorPolicy,
    dry_run: bool,
    quarantine_cap: u64,
) -> EngineResult<CreatedMigration> {
    let _guard = CATALOG_INIT_LOCK.lock();
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();

    let result = (|| -> EngineResult<CreatedMigration> {
        let fv_raw = storage
            .get(&txn, &KeyBuilder::catalog_format())?
            .ok_or_else(|| {
                EngineError::Catalog(CatalogError::MissingRequiredKey {
                    key_debug: "c:F:".into(),
                })
            })?;
        let format = decode_format_version("c:F:", &fv_raw)?;
        let cat = load_existing(storage, snap, format)?;

        // Refuse a second migration on the same TYPE (any field) while one is
        // unsettled — the worker rewrites the whole object blob, so concurrent
        // plans on one type clobber each other (see active_plan_for_type).
        let plans = scan_migration_plans(storage, snap)?;
        if let Some((qualified, plan_id)) = active_plan_for_type(&plans, type_name) {
            return Err(EngineError::Catalog(
                CatalogError::MigrationFieldHasActivePlan { qualified, plan_id },
            ));
        }

        let (field_entry, type_id) =
            validate_field_type_change(&cat, schema, type_name, field_name, target_kind)?;

        let counter = read_migration_counter(storage, &txn)?;
        let plan_id = next_migration_id(counter, &plans);
        let plan = MigrationPlan {
            plan_id,
            type_name: type_name.to_string(),
            field_name: field_name.to_string(),
            field_id: field_entry.id,
            src_kind: field_entry.kind,
            target_kind,
            status: MigrationStatus::Running,
            cursor: 0,
            chunk_size,
            created_at_ms: now_unix_millis(),
            converter_name: converter_name.to_string(),
            converter_version,
            objects_converted: 0,
            phase: MigrationPhase::Converting,
            cutover_cursor: 0,
            parallel_degree,
            id_upper_bound,
            error_policy,
            dry_run,
            error_count: 0,
            quarantine_cap: if quarantine_cap == 0 {
                DEFAULT_QUARANTINE_CAP
            } else {
                quarantine_cap
            },
            unknown_tlvs: Vec::new(),
        };
        let puts = vec![
            (
                KeyBuilder::catalog_migration_plan(plan_id),
                encode_migration_plan(&plan),
            ),
            (KeyBuilder::catalog_next_migration(), encode_counter(plan_id)),
        ];
        storage.put_batch(&mut txn, &puts)?;
        Ok(CreatedMigration { plan_id, type_id })
    })();

    match result {
        Ok(created) => {
            storage.commit(&mut txn)?;
            Ok(created)
        }
        Err(e) => {
            storage.abort(&mut txn);
            Err(e)
        }
    }
}

/// Point-load a single plan record `c:P:<plan_id>` within `txn`.
fn require_migration_plan(
    storage: &LsmTree,
    txn: &rhypedb_storage::mvcc::Transaction,
    plan_id: u64,
) -> EngineResult<MigrationPlan> {
    let key = KeyBuilder::catalog_migration_plan(plan_id);
    let bytes = storage.get(txn, &key)?.ok_or_else(|| {
        EngineError::Catalog(CatalogError::MissingRequiredKey {
            key_debug: debug_key(&key),
        })
    })?;
    decode_migration_plan(plan_id, &debug_key(&key), &bytes)
}

/// Point-load a single migration plan within `txn`. Public so the card-2
/// cutover (a `Database` method — it must reach the index/rev-edge cover
/// maintenance that lives on `Database`, not `LsmTree`) can read the plan
/// it drives.
pub(crate) fn load_migration_plan(
    storage: &LsmTree,
    txn: &rhypedb_storage::mvcc::Transaction,
    plan_id: u64,
) -> EngineResult<MigrationPlan> {
    require_migration_plan(storage, txn, plan_id)
}

/// The `(key, encoded-bytes)` pair for a plan record, so the card-2 cutover
/// loop can stage the plan-record advance (cutover cursor / phase) LAST in its
/// own per-chunk batch — the same crash-safe ordering `run_migration_chunks`
/// uses.
pub(crate) fn migration_plan_record(plan: &MigrationPlan) -> (Bytes, Bytes) {
    (
        KeyBuilder::catalog_migration_plan(plan.plan_id),
        encode_migration_plan(plan),
    )
}

/// Park a plan `Failed` AND rewind it to a clean `Converting` start, in its own
/// txn. Used by the CUTOVER refusal arms (missing / stale shadow): unlike the
/// backfill's plain `park_migration_failed` (which leaves `phase=Converting` +
/// a mid-range backfill cursor so resume continues the converter), a cutover
/// refusal must send resume BACK through the backfill so the missing/stale
/// shadow is re-stamped — otherwise `drive_migration_to_completion` (which only
/// runs the converter while `phase==Converting`) would skip straight back to the
/// cutover and re-refuse forever, holding quiesce. Resets `phase=Converting`,
/// `cursor=0` (re-scan the whole keyspace; already-current shadows are skipped
/// idempotently), and `cutover_cursor=0`.
pub(crate) fn park_migration_failed_rewind(storage: &LsmTree, plan_id: u64) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    plan.status = MigrationStatus::Failed;
    plan.phase = MigrationPhase::Converting;
    plan.cursor = 0;
    plan.cutover_cursor = 0;
    // Card 3 (load-bearing for parallel plans): the rewind sends resume BACK
    // through the backfill, but an N>1 plan's backfill is gated by the
    // per-partition `c:S:<plan><idx>` `done` flags — `run_migration_partition`
    // fast-returns `Done` the instant `done==true`. If the cursors survived the
    // rewind, the re-driven backfill would fast-return Done for every partition,
    // `all_partitions_done` would report the range covered, and cutover would
    // re-refuse the SAME missing/stale shadow forever. Deleting every `c:S:`
    // cursor forces a full re-backfill (already-current shadows are
    // idempotency-skipped). Done in the SAME txn as the plan rewrite, deletes
    // FIRST so a torn tail can only drop the plan-status advance. Safe vs
    // concurrent writers: the only caller is `run_cutover`, which holds
    // `migration_lock.write()`, so no live writer/worker can add a `c:S:` key
    // here. (The plain `park_migration_failed_keep_cursors` — used by a
    // CONVERTING-phase worker error — does NOT reset the cursors so resume
    // continues each partition from where it stopped.)
    let cursor_keys: Vec<Bytes> = storage
        .scan_prefix_at(snap, &KeyBuilder::catalog_partition_cursor_plan_prefix(plan_id))?
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    if !cursor_keys.is_empty() {
        storage.delete_batch(&mut txn, &cursor_keys)?;
    }
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// Park a plan `Failed` WITHOUT resetting any cursor (card 3). Used by a
/// CONVERTING-phase backfill error (single-worker `run_migration_chunks` or a
/// parallel `run_migration_partition` worker): the failing chunk was aborted
/// before commit, so every `c:S:<plan><idx>` (and the legacy `cursor`) stays at
/// its last durable position — resume continues each partition from there and
/// re-converts idempotently. Distinct from `park_migration_failed_rewind`, which
/// a CUTOVER refusal uses to force a full re-backfill by deleting the cursors.
pub(crate) fn park_migration_failed_keep_cursors(
    storage: &LsmTree,
    plan_id: u64,
) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    plan.status = MigrationStatus::Failed;
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// Card 5: durably mark a plan `RollingBack` (a terminal cancel) in its own txn,
/// resetting `cutover_cursor=0` so the rollback strip starts from the top. The
/// status is LEFT as-is (Running/Pending/etc.) — it stays `quiesces()`+drivable
/// so a crash mid-rollback re-arms the hook and resumes the strip via
/// auto-resume. Idempotent (already RollingBack → no-op). The caller holds
/// `migration_lock.write()`, and partition workers never write `c:P:`, so this
/// can't race a concurrent backfill commit. MUST run BEFORE the in-memory CANCEL
/// control store so a winding-down driver's re-load observes it.
pub(crate) fn set_plan_phase_rolling_back(storage: &LsmTree, plan_id: u64) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    if plan.phase == MigrationPhase::RollingBack {
        storage.abort(&mut txn);
        return Ok(());
    }
    plan.phase = MigrationPhase::RollingBack;
    plan.cutover_cursor = 0; // reused as the rollback cursor — start from the top
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// Card 5: settle a rolled-back plan `Cancelled`. Deletes the plan's per-partition
/// cursors (`c:S:`) and quarantine sidecars (`c:Q:`) FIRST (they are exempt from
/// `recover_partial_into_txn`, so leaving them orphans them), then writes
/// `status=Cancelled` (plan record LAST), all in ONE txn. Does NOT flip the
/// catalog kind (the field stays at `src_kind`) and does NOT bump the format
/// floor. Idempotent.
///
/// Crash-safety: the caller (`run_cancel_rollback_locked`) has ALREADY stripped
/// every `<field>__shadow` from the `o:` blobs BEFORE this runs. Once this commit
/// lands, `status=Cancelled` does NOT `quiesces()`, so a reopen re-arms no hook —
/// but no shadow remains on disk, so there is no window where a shadow is present
/// while the hook is gone (the leak the deferral warned about). Disarm itself is
/// in-memory (the `migrating_fields` ArcSwap) and is done by the caller under the
/// still-held write lock, mirroring `finalize_migration_cutover`.
pub(crate) fn finalize_migration_cancelled(storage: &LsmTree, plan_id: u64) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    // Delete c:S: + c:Q: for this plan FIRST (dead state — the plan is settled).
    let mut dead: Vec<Bytes> = storage
        .scan_prefix_at(snap, &KeyBuilder::catalog_partition_cursor_plan_prefix(plan_id))?
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    dead.extend(
        storage
            .scan_prefix_at(snap, &KeyBuilder::catalog_quarantine_plan_prefix(plan_id))?
            .into_iter()
            .map(|(k, _)| k),
    );
    if !dead.is_empty() {
        storage.delete_batch(&mut txn, &dead)?;
    }
    plan.status = MigrationStatus::Cancelled;
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// The trailing 8 bytes of an object key are its `object_id` (BE).
fn object_id_from_key(key: &[u8]) -> u64 {
    u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap())
}

/// Backfill the `<field>__shadow` siblings for a plan's `Converting` phase
/// (shadow-field card 2). Loads the plan, resolves the type id once, then scans
/// the object keyspace in chunks. For each row it converts the SOURCE value and
/// writes a `<field>__shadow` + `<field>__shadow_cv` (= the converter version)
/// sibling into the blob, **leaving the source field intact** — reads continue
/// to serve the source until the separate cutover pass promotes the shadow.
///
/// Each chunk commits as `[shadowed object blobs..., updated plan record LAST]`,
/// so a torn WAL tail can only drop the cursor advance — never leave the cursor
/// AHEAD of its durable blobs. Combined with per-row idempotency (a row whose
/// `<field>__shadow` is already present AND stamped with the current converter
/// version is skipped), re-running from the last durable cursor after a crash is
/// correct. A `<field>__shadow` left by an OLD converter version is re-stamped.
///
/// The per-chunk commit is wrapped in a bounded `WriteConflict` retry that
/// re-snapshots and re-scans the chunk: this backfill runs WITHOUT
/// `migration_lock` (so writes to other types proceed), and a concurrent live
/// double-writer (card 2d) may have advanced a row's shadow on the same `o:`
/// key — the retry re-reads and the idempotency skip drops the now-current row.
///
/// Does NOT bump the object generation (the source is unchanged, so covers stay
/// correct during `Converting`) and does NOT flip the catalog kind — the caller
/// invokes the cutover after this returns `Ok`. On a converter error, a
/// converter that returns the wrong kind, or an on-disk row whose source kind is
/// neither source nor target, the plan is parked `Failed` (the double-write hook
/// stays armed so writes to the field keep failing closed) and the error is returned.
/// Convert one object's blob for a field-type migration backfill. Returns the
/// rewritten blob (source kept + `<field>__shadow`/`__shadow_cv` added) when the
/// row needs conversion, or `None` to skip (field absent / Null / source already
/// at target kind / shadow already current). A terminal `Err` (on-disk source
/// kind != catalog `src_kind`, converter failure, converter returned the wrong
/// kind) means the caller must abort the in-flight chunk and park the plan
/// `Failed`. Shared by the single-worker (`run_migration_chunks`) and
/// per-partition (`run_migration_partition`) backfills so the conversion
/// semantics — and thus the produced blobs (AC4 byte-for-byte) — are identical.
#[allow(clippy::too_many_arguments)]
fn convert_row_for_backfill(
    object_id: u64,
    blob: &[u8],
    type_name: &str,
    field_name: &str,
    shadow_name: &str,
    shadow_cv_name: &str,
    src_kind: u8,
    target_kind: u8,
    converter_version: u32,
    converter: &RegisteredConverter,
    plan_id: u64,
) -> EngineResult<Option<Bytes>> {
    let mut fields = crate::object::deserialize_fields(blob);
    let Some(old_value) = fields.get(field_name).cloned() else {
        return Ok(None); // field absent on this row — nothing to convert
    };
    let src_got = value_to_kind_byte(&old_value);
    // Source already holds a target-kind value (anomalous / predates the source
    // schema) → cutover treats source==target+no-shadow as already-cut; or Null
    // → carries no shadow (mirrors the write hook). Either way: skip.
    if src_got == target_kind || src_got == kind_byte::UNSET {
        return Ok(None);
    }
    if src_got != src_kind {
        // On-disk source disagrees with the catalog — park Failed rather than guess.
        return Err(EngineError::Catalog(CatalogError::MigrationRowUnexpectedKind {
            plan_id,
            object_id,
            got_kind: kind_name(src_got),
            expected_kind: kind_name(src_kind),
        }));
    }
    // Idempotency: a `<field>__shadow` present AND stamped with the CURRENT
    // converter version is up to date — skip. A shadow from an OLD version (or
    // with no `_cv` stamp) is stale → fall through and re-convert.
    if fields.contains_key(shadow_name)
        && matches!(
            fields.get(shadow_cv_name),
            Some(crate::object::Value::U32(v)) if *v == converter_version
        )
    {
        return Ok(None);
    }
    let new_value = converter(object_id, &old_value).map_err(|e| {
        EngineError::Catalog(CatalogError::FieldTypeChangeConverterFailed {
            qualified: format!("{type_name}.{field_name}"),
            object_id,
            reason: e.to_string(),
        })
    })?;
    let out_kind = value_to_kind_byte(&new_value);
    if out_kind != target_kind {
        return Err(EngineError::Catalog(
            CatalogError::FieldTypeChangeConverterReturnedWrongKind {
                qualified: format!("{type_name}.{field_name}"),
                object_id,
                got_kind: kind_name(out_kind),
                want_kind: kind_name(target_kind),
            },
        ));
    }
    // Write the shadow sibling + its converter-version stamp, LEAVE the source.
    // No generation bump: the source is unchanged so every cover that embeds it
    // stays correct during Converting.
    fields.insert(shadow_name.to_string(), new_value);
    fields.insert(
        shadow_cv_name.to_string(),
        crate::object::Value::U32(converter_version),
    );
    Ok(Some(crate::object::serialize_fields(&fields)))
}

pub(crate) fn run_migration_chunks(
    storage: &LsmTree,
    plan_id: u64,
    converter: &RegisteredConverter,
) -> EngineResult<()> {
    // Load the plan + resolve the owning type id once (the per-chunk commits
    // touch only `o:` + `c:P:<id>`, never the catalog, so no further catalog
    // reads are needed inside the loop).
    let (mut plan, type_id) = {
        let txn = storage.begin_txn();
        let snap = txn.snapshot();
        let plan = require_migration_plan(storage, &txn, plan_id)?;
        let fv_raw = storage
            .get(&txn, &KeyBuilder::catalog_format())?
            .ok_or_else(|| {
                EngineError::Catalog(CatalogError::MissingRequiredKey {
                    key_debug: "c:F:".into(),
                })
            })?;
        let format = decode_format_version("c:F:", &fv_raw)?;
        let cat = load_existing(storage, snap, format)?;
        let type_id = *cat.type_ids.get(&plan.type_name).ok_or_else(|| {
            EngineError::Catalog(CatalogError::FieldTypeChangeSourceNotFound {
                qualified: format!("{}.{}", plan.type_name, plan.field_name),
            })
        })?;
        (plan, type_id)
    };

    // Card 4: the legacy single-worker path is STOP-only and resumes only
    // pre-card-4 plans. A card-4 plan (non-Stop policy or dry-run) is always
    // PARALLEL — refuse rather than silently downgrade its declared policy.
    if plan.error_policy != ErrorPolicy::Stop || plan.dry_run {
        return Err(EngineError::Catalog(CatalogError::MalformedMigrationPlan {
            row: format!("c:P:{plan_id}"),
            reason: "non-Stop / dry-run plan must run via the parallel backfill path, not run_migration_chunks",
        }));
    }

    // Bound on per-chunk WriteConflict retries before giving up (a hot key
    // under concurrent live writes in card-2 inc 2d). Dormant in inc 2c.
    const WRITE_CONFLICT_RETRIES: u32 = 8;

    let field_name = plan.field_name.clone();
    let shadow_name = format!("{field_name}__shadow");
    let shadow_cv_name = format!("{field_name}__shadow_cv");
    let src_kind = plan.src_kind;
    let target_kind = plan.target_kind;
    let converter_version = plan.converter_version;
    let chunk_size = if plan.chunk_size == 0 {
        DEFAULT_MIGRATION_CHUNK_SIZE
    } else {
        plan.chunk_size
    } as usize;
    let object_prefix = KeyBuilder::object_prefix(type_id);
    let mut cursor = plan.cursor;

    loop {
        // Resume strictly AFTER the cursor (idempotency makes re-including it
        // harmless, but skipping it is cheaper).
        let start = if cursor == 0 {
            object_prefix.clone()
        } else {
            match cursor.checked_add(1) {
                Some(next) => KeyBuilder::object(type_id, next),
                // Cursor at u64::MAX — no possible key past it. Exhausted.
                None => break,
            }
        };

        // Process one chunk, retrying its commit on a WriteConflict by
        // re-snapshotting + re-scanning (the idempotency skip drops rows a
        // racing writer already shadowed). `None` = data exhausted; `Some` =
        // committed (carries the advanced plan + whether more chunks remain).
        let mut attempts = 0u32;
        let committed: Option<(MigrationPlan, bool)> = loop {
            let mut txn = storage.begin_txn();
            let snap = txn.snapshot();
            let chunk = storage.scan_chunk_raw(snap, &object_prefix, &start, chunk_size)?;
            let Some(high_water) = chunk.high_water.clone() else {
                // No key — live or tombstoned — past the cursor: data done.
                storage.abort(&mut txn);
                break None;
            };
            let more = chunk.more;
            // Advance the cursor to the tombstone-inclusive high-water mark so
            // a tombstone run longer than the chunk can't strand live keys.
            let next_cursor = object_id_from_key(&high_water);

            let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(chunk.live.len() + 1);
            let mut converted_this_chunk: u64 = 0;
            for (key, blob) in &chunk.live {
                let object_id = object_id_from_key(key);
                match convert_row_for_backfill(
                    object_id,
                    blob,
                    &plan.type_name,
                    &field_name,
                    &shadow_name,
                    &shadow_cv_name,
                    src_kind,
                    target_kind,
                    converter_version,
                    converter,
                    plan_id,
                ) {
                    Ok(Some(new_blob)) => {
                        puts.push((key.clone(), new_blob));
                        converted_this_chunk += 1;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // Abort this chunk's uncommitted blobs (the cursor stays
                        // where the last committed chunk left it), park Failed
                        // keeping the cursor so resume continues converting.
                        storage.abort(&mut txn);
                        park_migration_failed_keep_cursors(storage, plan_id)?;
                        return Err(e);
                    }
                }
            }

            let mut plan_after = plan.clone();
            plan_after.cursor = next_cursor;
            plan_after.objects_converted =
                plan.objects_converted.saturating_add(converted_this_chunk);
            plan_after.status = MigrationStatus::Running;
            // Phase stays Converting throughout this loop.
            // CRASH-SAFE ORDER: object blobs FIRST, plan record LAST.
            puts.push((
                KeyBuilder::catalog_migration_plan(plan_id),
                encode_migration_plan(&plan_after),
            ));
            storage.put_batch(&mut txn, &puts)?;
            match storage.commit(&mut txn) {
                Ok(_) => break Some((plan_after, more)),
                Err(rhypedb_storage::Error::WriteConflict)
                    if attempts < WRITE_CONFLICT_RETRIES =>
                {
                    // Release the conflicted txn's snapshot before re-scanning,
                    // else each retry orphans a snapshot that pins the GC floor.
                    storage.abort(&mut txn);
                    attempts += 1;
                    continue; // re-snapshot + re-scan + re-skip already-shadowed
                }
                Err(e) => {
                    storage.abort(&mut txn);
                    return Err(EngineError::Storage(e));
                }
            }
        };

        match committed {
            None => break, // data exhausted
            Some((plan_after, more)) => {
                plan = plan_after;
                cursor = plan.cursor;
                // `!more` is the ONLY sound exhaustion signal (live.len() <
                // chunk cannot be — a tombstone run shrinks live below the cap).
                if !more {
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Control byte for an in-flight parallel migration, shared (`Arc<AtomicU8>`)
/// across the driver + every partition worker; workers poll it BETWEEN chunks
/// (card 3/5).
pub(crate) mod migration_control {
    pub const RUN: u8 = 0;
    pub const PAUSE: u8 = 1;
    pub const CANCEL: u8 = 2;
}

/// Why a partition worker returned (card 3/5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionDriveOutcome {
    /// The partition's whole `[lo, hi)` range is converted (cursor `done`).
    Done,
    /// Stopped at a chunk boundary on a PAUSE signal — resumable from the cursor.
    Paused,
    /// Stopped at a chunk boundary on a CANCEL signal — resumable from the cursor.
    Cancelled,
    /// Card 4: the GLOBAL quarantine/error count crossed the cap — the worker
    /// committed its in-flight chunk (so the `c:Q:` rows + counts are durable)
    /// then stopped. The driver parks the plan `Failed` once.
    CapExceeded,
}

/// Back-fill `<field>__shadow` for ONE partition's contiguous object-id range
/// `[lo, hi)` (card 3/5). Mirrors `run_migration_chunks` but scans only its
/// range and advances its OWN `c:S:<plan><idx>` cursor — disjoint from every
/// other partition's keys AND from the `c:P:` plan record, so N workers never
/// WriteConflict with each other on the hot path (only with a live foreground
/// writer touching a row in this range). Resumes from the persisted partition
/// cursor; polls `control` between chunks (pause/cancel stop at a chunk
/// boundary, leaving the partition resumable). On a conversion/data error it
/// aborts the in-flight chunk and returns the error WITHOUT parking the plan —
/// the driver joins all partitions then parks `Failed` once. Per-chunk commit
/// order `[o: blobs FIRST, c:S:<plan><idx> LAST]`; a torn tail drops only the
/// cursor advance and resume re-converts idempotently.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_migration_partition(
    storage: &LsmTree,
    plan_id: u64,
    type_id: u64,
    partition_idx: u8,
    lo: u64,
    hi: u64,
    type_name: &str,
    field_name: &str,
    src_kind: u8,
    target_kind: u8,
    converter_version: u32,
    chunk_size: u64,
    converter: &RegisteredConverter,
    control: &std::sync::atomic::AtomicU8,
    // Card 4: per-row failure policy, dry-run (count-only), the GLOBAL error
    // counter shared across all partitions (soft cap tripwire), the cap, and the
    // converter name recorded in `c:Q:` rows.
    error_policy: ErrorPolicy,
    dry_run: bool,
    error_counter: &std::sync::atomic::AtomicU64,
    quarantine_cap: u64,
    attempted_converter_name: &str,
    // Card 5: live event sink (per-chunk ChunkCompleted + a PartitionDone when
    // this run exhausts the partition's range). `None` → no publishing.
    events: Option<&crate::database::MigrationEventHub>,
) -> EngineResult<PartitionDriveOutcome> {
    use std::sync::atomic::Ordering;
    const WRITE_CONFLICT_RETRIES: u32 = 8;
    let publish = |ev: crate::database::MigrationEvent| {
        if let Some(hub) = events {
            hub.publish(ev);
        }
    };
    enum ChunkResult {
        Exhausted,
        // (advanced cursor, errors committed in THIS chunk).
        Committed(PartitionCursor, u64),
    }

    let shadow_name = format!("{field_name}__shadow");
    let shadow_cv_name = format!("{field_name}__shadow_cv");
    let chunk_size = if chunk_size == 0 {
        DEFAULT_MIGRATION_CHUNK_SIZE
    } else {
        chunk_size
    } as usize;
    let object_prefix = KeyBuilder::object_prefix(type_id);
    let cursor_key = KeyBuilder::catalog_partition_cursor(plan_id, partition_idx);
    let cursor_dbg = debug_key(&cursor_key);

    // Load this partition's persisted cursor (absent → start at `lo`).
    let mut pc = {
        let txn = storage.begin_txn();
        match storage.get(&txn, &cursor_key)? {
            Some(bytes) => decode_partition_cursor(&cursor_dbg, &bytes)?,
            None => PartitionCursor {
                cursor: 0,
                objects_converted: 0,
                errors: 0,
                done: false,
            },
        }
    };
    if pc.done {
        return Ok(PartitionDriveOutcome::Done);
    }

    loop {
        match control.load(Ordering::Relaxed) {
            migration_control::PAUSE => return Ok(PartitionDriveOutcome::Paused),
            migration_control::CANCEL => return Ok(PartitionDriveOutcome::Cancelled),
            _ => {}
        }

        // Resume strictly after the cursor, but never before `lo` (a fresh
        // cursor of 0 starts the scan at `lo`).
        let start_id = pc.cursor.max(lo.saturating_sub(1)).saturating_add(1);
        if start_id >= hi {
            // The whole [lo, hi) range is covered (incl. an empty partition where
            // lo >= hi) — mark done idempotently.
            if !pc.done {
                pc.done = true;
                let mut txn = storage.begin_txn();
                storage.put(&mut txn, &cursor_key, encode_partition_cursor(&pc))?;
                storage.commit(&mut txn)?;
            }
            publish(crate::database::MigrationEvent::PartitionDone {
                plan_id,
                partition_idx,
            });
            return Ok(PartitionDriveOutcome::Done);
        }
        let start = KeyBuilder::object(type_id, start_id);

        let mut attempts = 0u32;
        let result = loop {
            let mut txn = storage.begin_txn();
            let snap = txn.snapshot();
            let chunk = storage.scan_chunk_raw(snap, &object_prefix, &start, chunk_size)?;
            let Some(high_water) = chunk.high_water.clone() else {
                // No key past `start` in the WHOLE type prefix → range exhausted.
                storage.abort(&mut txn);
                break ChunkResult::Exhausted;
            };
            let more = chunk.more;
            let high_water_id = object_id_from_key(&high_water);

            let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(chunk.live.len() + 1);
            let mut converted_this_chunk: u64 = 0;
            let mut errors_this_chunk: u64 = 0;
            for (key, blob) in &chunk.live {
                let object_id = object_id_from_key(key);
                if object_id >= hi {
                    continue; // belongs to the next partition — never write it here
                }
                match convert_row_for_backfill(
                    object_id,
                    blob,
                    type_name,
                    field_name,
                    &shadow_name,
                    &shadow_cv_name,
                    src_kind,
                    target_kind,
                    converter_version,
                    converter,
                    plan_id,
                ) {
                    Ok(Some(new_blob)) => {
                        // Dry-run counts the convert but writes NO `o:` blob.
                        if !dry_run {
                            puts.push((key.clone(), new_blob));
                        }
                        converted_this_chunk += 1;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // A STRUCTURAL error (kind mismatch / converter-contract
                        // violation) or the Stop policy halts regardless: abort the
                        // chunk, propagate (the driver parks Failed keeping cursors).
                        if error_policy == ErrorPolicy::Stop || !is_policy_governed_error(&e) {
                            storage.abort(&mut txn);
                            return Err(e);
                        }
                        // SkipAndLog / Quarantine: count + continue. The errored row
                        // keeps its source value (NO shadow written) so cutover can
                        // leave it source-shape (SkipAndLog) or the operator triages
                        // it (Quarantine).
                        errors_this_chunk += 1;
                        if error_policy == ErrorPolicy::Quarantine && !dry_run {
                            let source_value =
                                single_field_value_blob(blob, field_name).unwrap_or_default();
                            puts.push((
                                KeyBuilder::catalog_quarantine(plan_id, object_id),
                                encode_quarantine_record(
                                    &source_value,
                                    now_unix_millis(),
                                    &e.to_string(),
                                    attempted_converter_name,
                                ),
                            ));
                        }
                    }
                }
            }

            // Termination is decided AFTER the commit, not as a pre-loop guard:
            // done once the type is exhausted OR the scan reached/passed `hi`.
            // The cursor never advances past `hi - 1` (the next partition owns
            // ids >= hi), so resume can't re-scan into the neighbour's range.
            let done = !more || high_water_id >= hi;
            let next_cursor = high_water_id.min(hi - 1);
            let pc_after = PartitionCursor {
                cursor: next_cursor,
                objects_converted: pc.objects_converted.saturating_add(converted_this_chunk),
                // errors are committed atomically with the cursor (and the
                // `o:`/`c:Q:` puts) so a torn/conflicted chunk drops them too —
                // re-scan recomputes idempotently.
                errors: pc.errors.saturating_add(errors_this_chunk),
                done,
            };
            puts.push((cursor_key.clone(), encode_partition_cursor(&pc_after)));
            storage.put_batch(&mut txn, &puts)?;
            match storage.commit(&mut txn) {
                Ok(_) => break ChunkResult::Committed(pc_after, errors_this_chunk),
                Err(rhypedb_storage::Error::WriteConflict)
                    if attempts < WRITE_CONFLICT_RETRIES =>
                {
                    storage.abort(&mut txn);
                    attempts += 1;
                    continue;
                }
                Err(e) => {
                    storage.abort(&mut txn);
                    return Err(EngineError::Storage(e));
                }
            }
        };

        match result {
            ChunkResult::Exhausted => {
                pc.done = true;
                let mut txn = storage.begin_txn();
                storage.put(&mut txn, &cursor_key, encode_partition_cursor(&pc))?;
                storage.commit(&mut txn)?;
                publish(crate::database::MigrationEvent::PartitionDone {
                    plan_id,
                    partition_idx,
                });
                return Ok(PartitionDriveOutcome::Done);
            }
            ChunkResult::Committed(pc_after, errs) => {
                pc = pc_after;
                // Card 5: a chunk is durable — emit progress (cursor + count).
                publish(crate::database::MigrationEvent::ChunkCompleted {
                    plan_id,
                    partition_idx,
                    cursor: pc.cursor,
                    objects_converted: pc.objects_converted,
                });
                // Roll this chunk's (now durable) errors into the GLOBAL counter
                // AFTER the commit, so a WriteConflict re-scan never double-adds.
                if errs > 0 {
                    error_counter.fetch_add(errs, Ordering::SeqCst);
                }
                // Soft cap tripwire — stop (the in-flight chunk is already durable)
                // once the GLOBAL error count crosses the cap; the driver parks Failed.
                if error_policy != ErrorPolicy::Stop
                    && error_counter.load(Ordering::SeqCst) > quarantine_cap
                {
                    return Ok(PartitionDriveOutcome::CapExceeded);
                }
                if pc.done {
                    publish(crate::database::MigrationEvent::PartitionDone {
                        plan_id,
                        partition_idx,
                    });
                    return Ok(PartitionDriveOutcome::Done);
                }
            }
        }
    }
}

/// Disposition of a parallel backfill pass (card 3b/2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackfillDisposition {
    /// Every partition's `[lo,hi)` is converted (all `c:S:` `done` flags
    /// satisfied) — the caller proceeds to the single-threaded cutover.
    AllDone,
    /// A worker stopped at a chunk boundary on a PAUSE/CANCEL signal (or because
    /// `Database::drop` paused it). The plan stays Running/Converting, resumable.
    Paused,
}

/// Summed per-partition progress (card 3b/2 + card 4). Re-reads the DURABLE
/// `c:S:<plan><idx>` cursors — the authoritative source (not in-memory worker
/// outcomes), so a fresh run and a resume that spawned a subset of workers reach
/// the identical decision. Returns `(objects_converted, errors, all_done)`:
/// One partition's live progress, for `Database::query_migration_progress`
/// (card 5). `lo`/`hi` are the `[lo, hi)` id range the worker owns; the cursor
/// fields come from the durable `c:S:<plan><idx>` record (defaulted to zero /
/// `done = lo>=hi` when the worker has not yet committed a cursor).
pub(crate) struct PartitionProgressRow {
    pub idx: u8,
    pub lo: u64,
    pub hi: u64,
    pub cursor: u64,
    pub objects_converted: u64,
    pub errors: u64,
    pub done: bool,
}

/// Read every partition's `c:S:<plan><idx>` cursor into a per-partition
/// progress row (card 5 observability). Mirrors `sum_partition_counts`'s scan
/// but keeps the per-partition detail. An empty range (`lo>=hi`) or an
/// unwritten cursor is reported as zero-progress; `done` reflects the durable
/// flag OR an immediately-complete empty range.
pub(crate) fn read_partition_progress(
    storage: &LsmTree,
    plan_id: u64,
    parallel_degree: u8,
    id_upper_bound: u64,
) -> EngineResult<Vec<PartitionProgressRow>> {
    let n = parallel_degree.max(1);
    let txn = storage.begin_txn();
    let mut rows = Vec::with_capacity(n as usize);
    for idx in 0..n {
        let (lo, hi) = partition_range(n, id_upper_bound, idx);
        let key = KeyBuilder::catalog_partition_cursor(plan_id, idx);
        let (cursor, objects_converted, errors, done) = match storage.get(&txn, &key)? {
            Some(bytes) => {
                let pc = decode_partition_cursor(&debug_key(&key), &bytes)?;
                (pc.cursor, pc.objects_converted, pc.errors, pc.done || lo >= hi)
            }
            None => (0, 0, 0, lo >= hi),
        };
        rows.push(PartitionProgressRow {
            idx,
            lo,
            hi,
            cursor,
            objects_converted,
            errors,
            done,
        });
    }
    Ok(rows)
}

/// `all_done` is true iff every partition's cursor has `done==true` OR its
/// `partition_range` is empty (`lo>=hi`, immediately complete). The counts sum
/// over partitions regardless of `all_done` (so the cap-exceed/dry-run paths can
/// roll them up, and a resume can seed the soft cap counter from them).
pub(crate) fn sum_partition_counts(
    storage: &LsmTree,
    plan_id: u64,
    parallel_degree: u8,
    id_upper_bound: u64,
) -> EngineResult<(u64, u64, bool)> {
    let n = parallel_degree.max(1);
    let txn = storage.begin_txn();
    let mut converted: u64 = 0;
    let mut errors: u64 = 0;
    let mut all_done = true;
    for idx in 0..n {
        let (lo, hi) = partition_range(n, id_upper_bound, idx);
        if lo >= hi {
            continue; // empty partition — nothing to convert, treated as done
        }
        let key = KeyBuilder::catalog_partition_cursor(plan_id, idx);
        match storage.get(&txn, &key)? {
            Some(bytes) => {
                let pc = decode_partition_cursor(&debug_key(&key), &bytes)?;
                converted = converted.saturating_add(pc.objects_converted);
                errors = errors.saturating_add(pc.errors);
                if !pc.done {
                    all_done = false;
                }
            }
            None => all_done = false, // never written → not done
        }
    }
    Ok((converted, errors, all_done))
}

/// Fan out `parallel_degree` partition workers over `[1, id_upper_bound)` for a
/// plan's Converting phase and JOIN them all (card 3b/2). Worker idx 0 runs on
/// THIS thread; idx `1..N` run on scoped threads — so this returns only after
/// every partition has stopped (no detached worker survives into the caller's
/// cutover; the join barrier is airtight via `std::thread::scope`). Each worker
/// advances only its own `c:S:<plan><idx>` cursor + its `o:` blobs (disjoint
/// keys → no inter-worker conflict). The caller must have already confirmed the
/// plan is in the `Converting` phase and pass its fields.
///
/// Returns `AllDone` only when the AUTHORITATIVE gate (`all_partitions_done`,
/// re-reading the persisted `done` flags) is satisfied. On any worker error or
/// PANIC: parks the plan `Failed` KEEPING the `c:S:` cursors (the failing chunk
/// aborted before commit, so resume continues each partition idempotently) and
/// returns the error. On any PAUSE/CANCEL outcome: returns `Paused` (CANCEL is
/// folded into Paused for card 3b/2 — terminal cancel + rollback is card 5).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_parallel_backfill(
    storage: &LsmTree,
    plan_id: u64,
    type_id: u64,
    parallel_degree: u8,
    id_upper_bound: u64,
    type_name: &str,
    field_name: &str,
    src_kind: u8,
    target_kind: u8,
    converter_version: u32,
    chunk_size: u64,
    converter: &RegisteredConverter,
    control: &std::sync::atomic::AtomicU8,
    // Card 4: the immutable failure policy, dry-run flag, cap, and the converter
    // name recorded into `c:Q:` rows.
    error_policy: ErrorPolicy,
    dry_run: bool,
    quarantine_cap: u64,
    converter_name: &str,
    // Card 5: per-plan live event sink (ChunkCompleted / PartitionDone). `None`
    // suppresses publishing (e.g. a unit test driving a worker directly). The
    // `&MigrationEventHub` is `Sync`, so it is shared across the scoped workers.
    events: Option<&crate::database::MigrationEventHub>,
) -> EngineResult<BackfillDisposition> {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::AtomicU64;
    let n = parallel_degree.max(1);

    // Soft GLOBAL error counter shared across all partition workers — only a cap
    // tripwire, NEVER the source of plan.error_count (that is summed from the
    // durable c:S: errors). Seed it from the durable errors so a RESUMED run's
    // cap check accounts for errors already committed in a prior run.
    let (_seed_conv, seed_errors, _seed_done) =
        sum_partition_counts(storage, plan_id, n, id_upper_bound)?;
    let error_counter = AtomicU64::new(seed_errors);

    // catch_unwind each worker so a panic can't unwind past the scope join
    // (which would skip the driver's done-signal + deregister → wait-forever +
    // a wedged registry entry). The panicking worker's in-flight txn aborts
    // cleanly on drop (buffered-not-applied since Step B).
    type WorkerOut = Result<EngineResult<PartitionDriveOutcome>, ()>;
    let run_one = |idx: u8| -> WorkerOut {
        let (lo, hi) = partition_range(n, id_upper_bound, idx);
        catch_unwind(AssertUnwindSafe(|| {
            run_migration_partition(
                storage,
                plan_id,
                type_id,
                idx,
                lo,
                hi,
                type_name,
                field_name,
                src_kind,
                target_kind,
                converter_version,
                chunk_size,
                converter,
                control,
                error_policy,
                dry_run,
                &error_counter,
                quarantine_cap,
                converter_name,
                events,
            )
        }))
        .map_err(|_| ())
    };

    let outcomes: Vec<WorkerOut> = std::thread::scope(|scope| {
        let handles: Vec<_> = (1..n).map(|idx| scope.spawn(move || run_one(idx))).collect();
        let mut out = Vec::with_capacity(n as usize);
        out.push(run_one(0)); // worker 0 on this thread
        for h in handles {
            // The closure already catch_unwinds, so a scope join can itself only
            // panic if `run_one`'s own frame did — fold that to `Err(())` too.
            out.push(h.join().unwrap_or(Err(())));
        }
        out
    });

    let mut any_paused = false;
    let mut any_cap_exceeded = false;
    let mut first_err: Option<EngineError> = None;
    for o in outcomes {
        match o {
            Err(()) => {
                first_err.get_or_insert(EngineError::Catalog(
                    CatalogError::MigrationWorkerPanicked { plan_id },
                ));
            }
            Ok(Err(e)) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Ok(Ok(PartitionDriveOutcome::Done)) => {}
            Ok(Ok(PartitionDriveOutcome::Paused)) | Ok(Ok(PartitionDriveOutcome::Cancelled)) => {
                any_paused = true;
            }
            Ok(Ok(PartitionDriveOutcome::CapExceeded)) => {
                any_cap_exceeded = true;
            }
        }
    }
    if let Some(e) = first_err {
        park_migration_failed_keep_cursors(storage, plan_id)?;
        return Err(e);
    }
    // Roll the DURABLE per-partition counts into the plan (authoritative
    // error_count), regardless of disposition — observability + the cutover gate.
    let (converted, errors, all_done) = sum_partition_counts(storage, plan_id, n, id_upper_bound)?;
    set_plan_counts(storage, plan_id, converted, errors)?;

    if any_cap_exceeded {
        // The cap tripwire fired — park Failed (hook stays armed, resumable for
        // triage). The cutover gate (unresolved c:Q: count) keeps it blocked until
        // the operator retries/clears or fixes the converter and resumes.
        park_migration_failed_keep_cursors(storage, plan_id)?;
        return Err(EngineError::MigrationQuarantineCapExceeded {
            plan_id,
            cap: quarantine_cap,
        });
    }
    if any_paused {
        return Ok(BackfillDisposition::Paused);
    }
    if all_done {
        Ok(BackfillDisposition::AllDone)
    } else {
        // Defensive: all workers reported Done yet a `c:S:` flag is unset — a
        // worker-contract violation. Park keeping cursors rather than cut over
        // a possibly-incomplete range; resume re-checks.
        park_migration_failed_keep_cursors(storage, plan_id)?;
        Err(EngineError::Catalog(
            CatalogError::MigrationPartitionGateInconsistent { plan_id },
        ))
    }
}

/// Roll the per-partition `c:S:` counts up into the plan record's observational
/// `objects_converted` + `error_count` (card 3b/2 + 4). Called once the backfill
/// stops; the plan record has no other concurrent `c:P:` writer at that point
/// (all workers joined, cutover not started).
fn set_plan_counts(
    storage: &LsmTree,
    plan_id: u64,
    objects_converted: u64,
    error_count: u64,
) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    plan.objects_converted = objects_converted;
    plan.error_count = error_count;
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// Finalize a DRY-RUN migration (card 4): delete the per-partition `c:S:`
/// cursors and mark the plan `DryRunCompleted` (settled + terminal, catalog kind
/// NOT flipped), in ONE commit. A dry-run never armed the double-write hook and
/// wrote no `o:`/`c:Q:`, so there is nothing else to clean and no hook to disarm
/// — this is storage-only (no `Database` needed). The plan record is KEPT so
/// `list_migrations` reports the preflight's `objects_converted`/`error_count`.
pub(crate) fn finalize_dry_run(storage: &LsmTree, plan_id: u64) -> EngineResult<()> {
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();
    let mut plan = match require_migration_plan(storage, &txn, plan_id) {
        Ok(p) => p,
        Err(e) => {
            storage.abort(&mut txn);
            return Err(e);
        }
    };
    // Delete every c:S:<plan> cursor (transient dry-run progress) in the same txn.
    let cursor_keys: Vec<Bytes> = storage
        .scan_prefix_at(snap, &KeyBuilder::catalog_partition_cursor_plan_prefix(plan_id))?
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    if !cursor_keys.is_empty() {
        storage.delete_batch(&mut txn, &cursor_keys)?;
    }
    plan.status = MigrationStatus::DryRunCompleted;
    let (key, value) = migration_plan_record(&plan);
    storage.put(&mut txn, &key, value)?;
    storage.commit(&mut txn)?;
    Ok(())
}

/// Flip the catalog field kind to the plan's target and mark the plan
/// `Completed`, in ONE commit ordered `[field-entry flip FIRST, plan record
/// LAST]`. A torn tail thus drops only the plan-status advance (the flip
/// persists), so resume re-runs this idempotently. Idempotent: if the kind
/// is already the target (a prior finalize's plan record was torn off), it
/// skips the flip + history entry and just records `Completed`. Caller holds
/// `migration_lock.write()`; this takes `CATALOG_INIT_LOCK`. Returns the
/// plan's `objects_converted`.
pub(crate) fn finalize_migration_cutover(storage: &LsmTree, plan_id: u64) -> EngineResult<u64> {
    let _guard = CATALOG_INIT_LOCK.lock();
    let mut txn = storage.begin_txn();
    let snap = txn.snapshot();

    let result = (|| -> EngineResult<u64> {
        let mut plan = require_migration_plan(storage, &txn, plan_id)?;
        let fv_raw = storage
            .get(&txn, &KeyBuilder::catalog_format())?
            .ok_or_else(|| {
                EngineError::Catalog(CatalogError::MissingRequiredKey {
                    key_debug: "c:F:".into(),
                })
            })?;
        let format_before = decode_format_version("c:F:", &fv_raw)?;
        let cat = load_existing(storage, snap, format_before)?;
        let qual = format!("{}.{}", plan.type_name, plan.field_name);

        let mut puts: Vec<(Bytes, Bytes)> = Vec::new();

        // Idempotent flip — only if the catalog isn't already at the target
        // (a torn prior finalize, or auto-resume re-running the cutover).
        if let Some(field_entry) = cat.field_entries.get(&qual)
            && field_entry.kind != plan.target_kind
        {
            let now_ms = now_unix_millis();
            let from_kind = field_entry.kind;
            let mut new_entry = field_entry.clone();
            new_entry.kind = plan.target_kind;
            new_entry.type_change_history.insert(
                0,
                TypeChangeRecord {
                    from_kind,
                    to_kind: plan.target_kind,
                    wall_time_unix_ms: now_ms,
                },
            );
            new_entry.last_type_change_at_ms = Some(now_ms);
            // Field-entry flip FIRST in the batch.
            puts.push((
                KeyBuilder::catalog_field(&plan.type_name, &plan.field_name),
                encode_id_entry(&new_entry),
            ));
            // Leave c:D: stale — the next open recomputes the digest (kind
            // contributes), reconcile finds no drift, refreshes it.
        }

        // Bump the catalog format FLOOR to V4 — a completed field-type change
        // implies V4. INDEPENDENT of the flip guard above: if a torn prior
        // finalize persisted the flip but dropped this bump (and the plan
        // record), the idempotent re-run skips the flip yet must STILL restore
        // the floor, or the version-gate at open is permanently lost. Never
        // downgrades a V5+ catalog (`max(current, V4)`).
        if format_before < CATALOG_FORMAT_V4 {
            puts.push((
                KeyBuilder::catalog_format(),
                encode_format_version(CATALOG_FORMAT_V4),
            ));
        }

        plan.status = MigrationStatus::Completed;
        // Plan record LAST.
        puts.push((
            KeyBuilder::catalog_migration_plan(plan_id),
            encode_migration_plan(&plan),
        ));
        storage.put_batch(&mut txn, &puts)?;
        Ok(plan.objects_converted)
    })();

    match result {
        Ok(n) => {
            storage.commit(&mut txn)?;
            Ok(n)
        }
        Err(e) => {
            storage.abort(&mut txn);
            Err(e)
        }
    }
}

// =====================================================================
// HELPERS
// =====================================================================

/// Public wrapper so `Database::change_field_type` can convert the
/// caller's `FieldType` to the on-disk kind discriminant.
pub(crate) fn schema_kind_byte_public(ft: &FieldType) -> u8 {
    schema_kind_byte(ft)
}

/// Public wrapper for the on-disk kind-byte → human name mapping, so
/// `Database` can build typed migration errors without duplicating the table.
pub(crate) fn kind_name_public(k: u8) -> &'static str {
    kind_name(k)
}

/// Public wrapper so the card-2 double-write hook (in `database.rs`) can
/// validate that a converter's output matches the migration's target kind.
pub(crate) fn value_to_kind_byte_public(v: &crate::object::Value) -> u8 {
    value_to_kind_byte(v)
}

fn schema_kind_byte(ft: &FieldType) -> u8 {
    use kind_byte::*;
    match ft {
        FieldType::Scalar(s) => match s {
            ScalarType::String => SCALAR_STRING,
            ScalarType::U32 => SCALAR_U32,
            ScalarType::U64 => SCALAR_U64,
            ScalarType::I32 => SCALAR_I32,
            ScalarType::I64 => SCALAR_I64,
            ScalarType::F32 => SCALAR_F32,
            ScalarType::F64 => SCALAR_F64,
            ScalarType::Bool => SCALAR_BOOL,
            ScalarType::DateTime => SCALAR_DATETIME,
            ScalarType::Bytes => SCALAR_BYTES,
            ScalarType::Json => SCALAR_JSON,
        },
        FieldType::Vector(_) => VECTOR,
        FieldType::Relation(_) => RELATION,
    }
}

/// Inverse of the scalar arm of [`schema_kind_byte`]: map a persisted kind byte
/// back to its `ScalarType`, or `None` for a non-scalar / unset kind. Lets the
/// engine surface a migration plan's target field type (e.g. so the server can
/// build the post-cutover schema for an in-place hot-reload).
pub(crate) fn scalar_type_from_kind(k: u8) -> Option<ScalarType> {
    use kind_byte::*;
    Some(match k {
        SCALAR_STRING => ScalarType::String,
        SCALAR_U32 => ScalarType::U32,
        SCALAR_U64 => ScalarType::U64,
        SCALAR_I32 => ScalarType::I32,
        SCALAR_I64 => ScalarType::I64,
        SCALAR_F32 => ScalarType::F32,
        SCALAR_F64 => ScalarType::F64,
        SCALAR_BOOL => ScalarType::Bool,
        SCALAR_DATETIME => ScalarType::DateTime,
        SCALAR_BYTES => ScalarType::Bytes,
        SCALAR_JSON => ScalarType::Json,
        _ => return None,
    })
}

fn kind_name(k: u8) -> &'static str {
    use kind_byte::*;
    match k {
        UNSET => "Unset",
        SCALAR_STRING => "Scalar(String)",
        SCALAR_U32 => "Scalar(U32)",
        SCALAR_U64 => "Scalar(U64)",
        SCALAR_I32 => "Scalar(I32)",
        SCALAR_I64 => "Scalar(I64)",
        SCALAR_F32 => "Scalar(F32)",
        SCALAR_F64 => "Scalar(F64)",
        SCALAR_BOOL => "Scalar(Bool)",
        SCALAR_DATETIME => "Scalar(DateTime)",
        SCALAR_BYTES => "Scalar(Bytes)",
        SCALAR_JSON => "Scalar(Json)",
        VECTOR => "Vector",
        RELATION => "Relation",
        _ => "Unknown",
    }
}

fn split_qualified(qual: &str) -> (&str, &str) {
    // Qualified keys are constructed by load_existing via
    // `format!("{type}.{field}")` where both halves were already
    // validated by check_identifier() — neither contains `.`. So
    // splitting on the first `.` is unambiguous.
    let dot = qual.find('.').unwrap_or(qual.len());
    let (a, b) = qual.split_at(dot);
    (a, b.strip_prefix('.').unwrap_or(b))
}

fn checked_bump(kind: &'static str, cur: u64) -> EngineResult<u64> {
    cur.checked_add(1)
        .ok_or(EngineError::Catalog(CatalogError::CounterOverflow { kind }))
}

fn check_identifier(name: &str) -> EngineResult<()> {
    for &b in name.as_bytes() {
        if b == 0 || b == b':' || b == b'.' {
            return Err(EngineError::Catalog(
                CatalogError::ReservedByteInIdentifier {
                    name: name.into(),
                    byte: b,
                },
            ));
        }
    }
    // Reject the engine-internal sidecar namespace anywhere in the name.
    // Engine writes `<field>__cover`, `<field>__cover_v`, `<field>__shadow`
    // (suffix) and `__cover_<id>`, `__rel_<id>` (prefix). A user identifier
    // containing `__` at any position can collide with one of these
    // sidecar keys inside a serialized FieldMap (cover blobs in rev_edge
    // values, shadow fields during in-flight field-type migrations).
    if name.contains("__") {
        return Err(EngineError::Catalog(
            CatalogError::ReservedDoubleUnderscoreInIdentifier { name: name.into() },
        ));
    }
    Ok(())
}

fn decode_rename_chain(key_debug: &str, value: &[u8]) -> EngineResult<Vec<RenameRecord>> {
    if value.is_empty() {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: 0,
            min: 1,
        }));
    }
    let count = value[0] as usize;
    if count == 0 {
        return Err(EngineError::Catalog(CatalogError::EmptyRenameChain {
            row: key_debug.into(),
        }));
    }
    if count > MAX_RENAME_HISTORY {
        return Err(EngineError::Catalog(CatalogError::RenameChainOverCap {
            row: key_debug.into(),
            count,
            cap: MAX_RENAME_HISTORY,
        }));
    }
    let mut chain = Vec::with_capacity(count);
    let mut cur = 1;
    for _ in 0..count {
        if value.len() - cur < 2 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: cur + 2,
            }));
        }
        let from_len = u16::from_be_bytes([value[cur], value[cur + 1]]) as usize;
        cur += 2;
        if value.len() - cur < from_len {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: cur + from_len,
            }));
        }
        let from = std::str::from_utf8(&value[cur..cur + from_len])
            .map_err(|_| {
                EngineError::Catalog(CatalogError::MalformedKey {
                    key_debug: key_debug.into(),
                })
            })?
            .to_string();
        cur += from_len;

        if value.len() - cur < 2 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: cur + 2,
            }));
        }
        let to_len = u16::from_be_bytes([value[cur], value[cur + 1]]) as usize;
        cur += 2;
        if value.len() - cur < to_len {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: cur + to_len,
            }));
        }
        let to = std::str::from_utf8(&value[cur..cur + to_len])
            .map_err(|_| {
                EngineError::Catalog(CatalogError::MalformedKey {
                    key_debug: key_debug.into(),
                })
            })?
            .to_string();
        cur += to_len;

        if value.len() - cur < 8 {
            return Err(EngineError::Catalog(CatalogError::Truncated {
                key_debug: key_debug.into(),
                len: value.len(),
                min: cur + 8,
            }));
        }
        let wall_time_unix_ms =
            u64::from_be_bytes(value[cur..cur + 8].try_into().unwrap());
        cur += 8;

        chain.push(RenameRecord {
            from,
            to,
            wall_time_unix_ms,
        });
    }
    Ok(chain)
}

fn decode_type_change_chain(
    key_debug: &str,
    value: &[u8],
) -> EngineResult<Vec<TypeChangeRecord>> {
    if value.is_empty() {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: 0,
            min: 1,
        }));
    }
    let count = value[0] as usize;
    if count == 0 {
        return Err(EngineError::Catalog(CatalogError::EmptyTypeChangeChain {
            row: key_debug.into(),
        }));
    }
    if count > MAX_TYPE_CHANGE_HISTORY {
        return Err(EngineError::Catalog(CatalogError::TypeChangeChainOverCap {
            row: key_debug.into(),
            count,
            cap: MAX_TYPE_CHANGE_HISTORY,
        }));
    }
    let needed = 1 + count * 10;
    if value.len() < needed {
        return Err(EngineError::Catalog(CatalogError::Truncated {
            key_debug: key_debug.into(),
            len: value.len(),
            min: needed,
        }));
    }
    let mut chain = Vec::with_capacity(count);
    let mut cur = 1;
    for _ in 0..count {
        let from_kind = value[cur];
        let to_kind = value[cur + 1];
        let wall_time_unix_ms =
            u64::from_be_bytes(value[cur + 2..cur + 10].try_into().unwrap());
        cur += 10;
        chain.push(TypeChangeRecord {
            from_kind,
            to_kind,
            wall_time_unix_ms,
        });
    }
    Ok(chain)
}

pub(crate) fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn debug_key(key: &Bytes) -> String {
    let mut s = String::new();
    for &b in key.iter() {
        if (0x20..0x7f).contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("\\x{:02x}", b));
        }
    }
    s
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_schema::{
        Directive, FieldDef, FieldType, RelationType, ScalarType, Schema, TypeDef,
    };
    use rhypedb_storage::lsm::LsmConfig;
    use std::collections::HashMap;
    use tempfile::TempDir;

    // -----------------------------------------------------------------
    // Builders
    // -----------------------------------------------------------------

    fn schema_with(types: Vec<(&str, Vec<FieldDef>)>) -> Schema {
        let mut t = HashMap::new();
        for (name, fields) in types {
            t.insert(
                name.to_string(),
                TypeDef {
                    name: name.to_string(),
                    fields,
                },
            );
        }
        Schema { types: t }
    }

    fn scalar(name: &str, st: ScalarType) -> FieldDef {
        FieldDef {
            name: name.into(),
            field_type: FieldType::Scalar(st),
            directives: vec![],
        }
    }
    fn scalar_indexed(name: &str, st: ScalarType) -> FieldDef {
        FieldDef {
            name: name.into(),
            field_type: FieldType::Scalar(st),
            directives: vec![Directive::Indexed],
        }
    }
    fn relation(name: &str, target: &str, is_many: bool) -> FieldDef {
        FieldDef {
            name: name.into(),
            field_type: FieldType::Relation(RelationType {
                target_type: target.into(),
                is_many,
                edge_fields: vec![],
            }),
            directives: vec![],
        }
    }

    fn scalar_unique(name: &str, st: ScalarType) -> FieldDef {
        FieldDef {
            name: name.into(),
            field_type: FieldType::Scalar(st),
            directives: vec![Directive::Unique],
        }
    }

    fn vector_vectorize(name: &str, dimensions: u32, source: &str) -> FieldDef {
        FieldDef {
            name: name.into(),
            field_type: FieldType::Vector(rhypedb_schema::VectorType { dimensions }),
            directives: vec![Directive::Vectorize(rhypedb_schema::VectorizeDef {
                source_field: source.into(),
                model: "test-model".into(),
            })],
        }
    }

    fn open_lsm(dir: &TempDir) -> std::sync::Arc<LsmTree> {
        let config = LsmConfig::new(dir.path());
        LsmTree::open(config).unwrap()
    }

    // -----------------------------------------------------------------
    // 9.1 Encoding unit tests
    // -----------------------------------------------------------------

    #[test]
    fn encode_decode_id_entry_roundtrip() {
        let entry = IdEntry::fresh(42, kind_byte::SCALAR_I64, 1_700_000_000_000);
        let bytes = encode_id_entry(&entry);
        let back = decode_id_entry("c:T:X", &bytes).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn encode_decode_id_entry_preserves_unknown_tlvs() {
        let mut entry = IdEntry::fresh(7, kind_byte::SCALAR_STRING, 100);
        // Tag 0xF7 is in the experimental range; 0x16 is reserved in
        // the tombstone range for a future free-form retire-note. Both
        // must round-trip verbatim through a current binary.
        entry.unknown_tlvs = vec![
            (0xF7, Bytes::from_static(b"future")),
            (0x16, Bytes::from_static(b"note")),
        ];
        let first = encode_id_entry(&entry);
        let back = decode_id_entry("c:T:X", &first).unwrap();
        assert_eq!(back, entry);
        // Round-trip-future-tag: a current binary must rewrite byte-for-byte.
        let second = encode_id_entry(&back);
        assert_eq!(first, second);
    }

    #[test]
    fn decode_id_entry_rejects_unknown_record_format() {
        let mut bytes = vec![0x99, KIND_ID_ENTRY, 0, 0];
        bytes.extend_from_slice(&encode_id_entry(&IdEntry::fresh(1, 0, 0))[4..]);
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::UnsupportedRecordFormat { got: 0x99, .. })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_unknown_value_kind() {
        let bytes = [RECORD_FORMAT_V1, 0x77, 0, 0];
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::UnknownValueKind { tag: 0x77, .. })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_wrong_value_kind() {
        let counter = encode_counter(42);
        let err = decode_id_entry("c:T:X", &counter).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::WrongValueKind {
                expected: "IdEntry",
                ..
            })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_truncated_body() {
        let mut bytes = encode_id_entry(&IdEntry::fresh(1, 0, 0)).to_vec();
        bytes.truncate(bytes.len() - 1);
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::Truncated { .. })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_duplicate_tlv() {
        // Manually build a body with TLV_ID twice.
        let mut body: Vec<u8> = Vec::new();
        write_tlv(&mut body, TLV_ID, &1u64.to_be_bytes());
        write_tlv(&mut body, TLV_ID, &2u64.to_be_bytes());
        write_tlv(&mut body, TLV_ASSIGNED_AT, &0u64.to_be_bytes());
        write_tlv(&mut body, TLV_ASSIGNED_BY, &[0]);
        write_tlv(&mut body, TLV_KIND, &[0]);
        let mut bytes: Vec<u8> = vec![RECORD_FORMAT_V1, KIND_ID_ENTRY];
        bytes.extend_from_slice(&(body.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&body);
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::DuplicateTlv { tag: TLV_ID, .. })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_missing_required_tlv_id() {
        let mut body: Vec<u8> = Vec::new();
        write_tlv(&mut body, TLV_ASSIGNED_AT, &0u64.to_be_bytes());
        write_tlv(&mut body, TLV_ASSIGNED_BY, &[0]);
        write_tlv(&mut body, TLV_KIND, &[0]);
        let mut bytes: Vec<u8> = vec![RECORD_FORMAT_V1, KIND_ID_ENTRY];
        bytes.extend_from_slice(&(body.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&body);
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::MissingRequiredTlv { tag: TLV_ID, .. })
        ));
    }

    #[test]
    fn encode_decode_counter_roundtrip() {
        for v in [0u64, 1, u64::MAX / 2, u64::MAX] {
            let bytes = encode_counter(v);
            let back = decode_counter("c:N:T", &bytes).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn decode_counter_rejects_truncated() {
        let err = decode_counter("c:N:T", &[1, 2, 3, 4]).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::Truncated { .. })
        ));
    }

    // --- Shadow-field migration plan codec (card 1/5) ---

    fn sample_plan() -> MigrationPlan {
        MigrationPlan {
            plan_id: 7,
            type_name: "User".to_string(),
            field_name: "age".to_string(),
            field_id: 42,
            src_kind: 0x04,
            target_kind: 0x05,
            status: MigrationStatus::Running,
            cursor: 1000,
            chunk_size: DEFAULT_MIGRATION_CHUNK_SIZE,
            created_at_ms: 1_700_000_000_000,
            converter_name: "widen_i32_to_i64".to_string(),
            converter_version: 1,
            objects_converted: 999,
            phase: MigrationPhase::Converting,
            cutover_cursor: 0,
            parallel_degree: None,
            id_upper_bound: 0,
            error_policy: ErrorPolicy::Stop,
            dry_run: false,
            error_count: 0,
            quarantine_cap: DEFAULT_QUARANTINE_CAP,
            unknown_tlvs: Vec::new(),
        }
    }

    /// Hand-rolled encoder so a test can plant a chosen raw status byte
    /// without searching the encoded bytes.
    fn encode_plan_with_raw_status(status_byte: u8) -> Vec<u8> {
        let mut body = Vec::new();
        write_tlv(&mut body, TLV_MP_TYPE_NAME, b"User");
        write_tlv(&mut body, TLV_MP_FIELD_NAME, b"age");
        write_tlv(&mut body, TLV_MP_FIELD_ID, &42u64.to_be_bytes());
        write_tlv(&mut body, TLV_MP_SRC_KIND, &[0x04]);
        write_tlv(&mut body, TLV_MP_TARGET_KIND, &[0x05]);
        write_tlv(&mut body, TLV_MP_STATUS, &[status_byte]);
        write_tlv(&mut body, TLV_MP_CURSOR, &0u64.to_be_bytes());
        write_tlv(&mut body, TLV_MP_CHUNK_SIZE, &1024u64.to_be_bytes());
        write_tlv(&mut body, TLV_MP_CREATED_AT_MS, &0u64.to_be_bytes());
        write_tlv(&mut body, TLV_MP_CONVERTER_NAME, b"c");
        write_tlv(&mut body, TLV_MP_CONVERTER_VERSION, &1u32.to_be_bytes());
        write_tlv(&mut body, TLV_MP_OBJECTS_CONVERTED, &0u64.to_be_bytes());
        let mut out = vec![RECORD_FORMAT_V1, KIND_MIGRATION_PLAN];
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn migration_plan_roundtrip() {
        let p = sample_plan();
        let bytes = encode_migration_plan(&p);
        let back = decode_migration_plan(p.plan_id, "c:P:7", &bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn migration_plan_roundtrip_parallel() {
        // Card 3: a parallel plan carries parallel_degree + id_upper_bound.
        let mut p = sample_plan();
        p.parallel_degree = Some(8);
        p.id_upper_bound = 1_000_000;
        p.cursor = 0; // parallel plans leave the legacy cursor at 0
        let bytes = encode_migration_plan(&p);
        let back = decode_migration_plan(p.plan_id, "c:P:7", &bytes).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.parallel_degree, Some(8));
        assert_eq!(back.id_upper_bound, 1_000_000);
    }

    #[test]
    fn legacy_plan_decodes_as_non_parallel() {
        // A card-1/2 row (sample_plan has no parallel tags) decodes to None/0.
        let back = decode_migration_plan(7, "c:P:7", &encode_migration_plan(&sample_plan())).unwrap();
        assert_eq!(back.parallel_degree, None);
        assert_eq!(back.id_upper_bound, 0);
    }

    #[test]
    fn migration_plan_roundtrip_card4_fields() {
        // Card 4: non-default error_policy / dry_run / error_count / cap round-trip.
        let mut p = sample_plan();
        p.error_policy = ErrorPolicy::Quarantine;
        p.dry_run = true;
        p.error_count = 1234;
        p.quarantine_cap = 50_000;
        p.status = MigrationStatus::DryRunCompleted;
        let back = decode_migration_plan(p.plan_id, "c:P:7", &encode_migration_plan(&p)).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.error_policy, ErrorPolicy::Quarantine);
        assert!(back.dry_run);
        assert_eq!(back.error_count, 1234);
        assert_eq!(back.quarantine_cap, 50_000);
        assert_eq!(back.status, MigrationStatus::DryRunCompleted);
    }

    #[test]
    fn card1_plan_decodes_card4_defaults() {
        // A pre-card-4 row (the hand-rolled encoder writes no 0x24-0x27 tags)
        // decodes to Stop / not-dry-run / 0 errors / the default cap.
        let raw = encode_plan_with_raw_status(MP_STATUS_RUNNING);
        let back = decode_migration_plan(7, "c:P:7", &raw).unwrap();
        assert_eq!(back.error_policy, ErrorPolicy::Stop);
        assert!(!back.dry_run);
        assert_eq!(back.error_count, 0);
        assert_eq!(back.quarantine_cap, DEFAULT_QUARANTINE_CAP);
    }

    #[test]
    fn quarantine_record_roundtrip_and_truncates_long_msg() {
        let src = b"\x00\x01some-serialized-value";
        let rec = decode_quarantine_record(
            "c:Q:1#2",
            &encode_quarantine_record(src, 1_700_000_000_000, "converter blew up", "widen"),
        )
        .unwrap();
        assert_eq!(&rec.source_value[..], &src[..]);
        assert_eq!(rec.errored_at_ms, 1_700_000_000_000);
        assert_eq!(rec.error_msg, "converter blew up");
        assert_eq!(rec.attempted_converter_name, "widen");
        // A >1 KiB message is truncated (on a char boundary).
        let long = "x".repeat(5000);
        let rec2 = decode_quarantine_record(
            "c:Q:1#2",
            &encode_quarantine_record(b"v", 0, &long, "c"),
        )
        .unwrap();
        assert_eq!(rec2.error_msg.len(), MAX_QUARANTINE_ERROR_MSG);
    }

    #[test]
    fn decode_migration_plan_rejects_out_of_range_parallel_degree() {
        // Hand-plant a parallel_degree TLV (0x22) with an illegal value 0.
        let mut p = sample_plan();
        p.parallel_degree = Some(8);
        p.id_upper_bound = 10;
        let mut bytes = encode_migration_plan(&p).to_vec();
        // Find the 0x22 TLV in the body (after the 4-byte record header) and
        // overwrite its 1-byte payload with 0. Body starts at offset 4; each TLV
        // is [tag u8][len u16 BE][payload].
        let body = &mut bytes[4..];
        let mut i = 0;
        let mut patched = false;
        while i + 3 <= body.len() {
            let tag = body[i];
            let len = u16::from_be_bytes([body[i + 1], body[i + 2]]) as usize;
            if tag == TLV_MP_PARALLEL_DEGREE {
                body[i + 3] = 0; // illegal degree
                patched = true;
                break;
            }
            i += 3 + len;
        }
        assert!(patched, "parallel_degree TLV not found");
        assert!(decode_migration_plan(7, "c:P:7", &bytes).is_err());
    }

    #[test]
    fn partition_cursor_roundtrip_and_rejects_corruption() {
        let pc = PartitionCursor {
            cursor: 123456,
            objects_converted: 789,
            errors: 12,
            done: true,
        };
        let bytes = encode_partition_cursor(&pc);
        assert_eq!(bytes.len(), PARTITION_CURSOR_LEN);
        assert_eq!(decode_partition_cursor("c:S:1#0", &bytes).unwrap(), pc);

        // Wrong length, wrong kind, bad done byte all fail cleanly.
        assert!(decode_partition_cursor("x", &bytes[..bytes.len() - 1]).is_err());
        let mut bad_kind = bytes.to_vec();
        bad_kind[1] = 0xFF;
        assert!(decode_partition_cursor("x", &bad_kind).is_err());
        let mut bad_done = bytes.to_vec();
        bad_done[PARTITION_CURSOR_LEN - 1] = 2;
        assert!(decode_partition_cursor("x", &bad_done).is_err());
    }

    #[test]
    fn partition_range_covers_domain_exactly_once() {
        // For several (n, U), every id in [1, U) lands in exactly one partition.
        for &u in &[1u64, 2, 7, 100, 1000, 1001, 4096] {
            for &n in &[1u8, 2, 3, 4, 8, 16, 64] {
                let ranges: Vec<(u64, u64)> =
                    (0..n).map(|i| partition_range(n, u, i)).collect();
                // Disjoint + ascending, and within the domain.
                for w in ranges.windows(2) {
                    assert!(w[0].1 <= w[1].0, "overlap at n={n} u={u}: {ranges:?}");
                }
                for &(lo, hi) in &ranges {
                    assert!(lo <= hi && hi <= u.max(1), "bad range n={n} u={u}: ({lo},{hi})");
                }
                // Every id in [1, U) covered exactly once.
                for id in 1..u {
                    let hits = ranges.iter().filter(|&&(lo, hi)| id >= lo && id < hi).count();
                    assert_eq!(hits, 1, "id {id} hit {hits}× at n={n} u={u}: {ranges:?}");
                }
            }
        }
    }

    #[test]
    fn migration_plan_preserves_unknown_tlvs_byte_identical() {
        // Simulate a future-card row carrying a reserved tag (0x30, still in
        // the unallocated 0x22-0x3F range) that this binary doesn't understand.
        // It must survive a decode→encode cycle verbatim so a reopen-and-rewrite
        // doesn't mangle a newer row.
        let mut p = sample_plan();
        p.unknown_tlvs.push((0x30, Bytes::from_static(&[0xAB, 0xCD])));
        let bytes = encode_migration_plan(&p);
        let back = decode_migration_plan(p.plan_id, "c:P:7", &bytes).unwrap();
        assert_eq!(
            back.unknown_tlvs,
            vec![(0x30u8, Bytes::from_static(&[0xAB, 0xCD]))]
        );
        assert_eq!(encode_migration_plan(&back), bytes);
    }

    #[test]
    fn decode_migration_plan_rejects_wrong_value_kind() {
        let counter = encode_counter(5); // KIND_COUNTER, not a plan
        let err = decode_migration_plan(1, "c:P:1", &counter).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::WrongValueKind { .. })
        ));
    }

    #[test]
    fn decode_id_entry_rejects_migration_plan_kind() {
        // A stray plan row landing where an id-entry is expected must error,
        // not be silently misread.
        let bytes = encode_migration_plan(&sample_plan());
        let err = decode_id_entry("c:E:User\0age", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::WrongValueKind {
                expected: "IdEntry",
                ..
            })
        ));
    }

    #[test]
    fn decode_migration_plan_rejects_unknown_status() {
        let bytes = encode_plan_with_raw_status(0x7F);
        let err = decode_migration_plan(7, "c:P:7", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::UnknownMigrationStatus { status: 0x7F, .. })
        ));
        // Sanity: a valid status byte decodes.
        let ok = decode_migration_plan(7, "c:P:7", &encode_plan_with_raw_status(MP_STATUS_RUNNING));
        assert_eq!(ok.unwrap().status, MigrationStatus::Running);
    }

    #[test]
    fn decode_migration_plan_rejects_missing_required_tlv() {
        let mut body: Vec<u8> = Vec::new();
        write_tlv(&mut body, TLV_MP_TYPE_NAME, b"User"); // only one of many
        let mut out = vec![RECORD_FORMAT_V1, KIND_MIGRATION_PLAN];
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(&body);
        let err = decode_migration_plan(1, "c:P:1", &out).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::MissingRequiredTlv { .. })
        ));
    }

    #[test]
    fn migration_status_byte_roundtrips_and_semantics() {
        for s in [
            MigrationStatus::Pending,
            MigrationStatus::Running,
            MigrationStatus::Completed,
            MigrationStatus::Cancelled,
            MigrationStatus::Failed,
            MigrationStatus::AwaitingConverter,
        ] {
            assert_eq!(MigrationStatus::from_byte(s.to_byte()), Some(s));
        }
        assert_eq!(MigrationStatus::from_byte(0x7F), None);
        // Quiesce released ONLY by Completed + Cancelled; Failed holds it.
        assert!(!MigrationStatus::Completed.quiesces());
        assert!(!MigrationStatus::Cancelled.quiesces());
        assert!(MigrationStatus::Failed.quiesces());
        assert!(MigrationStatus::AwaitingConverter.quiesces());
        assert!(MigrationStatus::Running.quiesces());
        // Terminal: Completed, Cancelled, Failed (no auto-resume).
        assert!(MigrationStatus::Failed.is_terminal());
        assert!(!MigrationStatus::AwaitingConverter.is_terminal());
        assert!(MigrationStatus::Running.is_drivable());
        assert!(MigrationStatus::Pending.is_drivable());
        assert!(!MigrationStatus::Failed.is_drivable());
    }

    #[test]
    fn next_migration_id_self_heals_against_max_plan() {
        let mut p1 = sample_plan();
        p1.plan_id = 1;
        let mut p5 = sample_plan();
        p5.plan_id = 5;
        // Counter behind the real max plan id (torn counter bump) → heal to max+1.
        assert_eq!(next_migration_id(2, &[p1.clone(), p5.clone()]), 6);
        // Counter ahead of all plans → counter+1.
        assert_eq!(next_migration_id(10, &[p1, p5]), 11);
        // Fresh DB: no plans, no counter.
        assert_eq!(next_migration_id(0, &[]), 1);
    }

    #[test]
    fn active_plan_for_type_matches_unsettled_any_field() {
        let plan = |id: u64, field: &str, status: MigrationStatus| {
            let mut p = sample_plan();
            p.plan_id = id;
            p.type_name = "User".into();
            p.field_name = field.into();
            p.status = status;
            p
        };
        // Unsettled blocks the TYPE regardless of field (the worker rewrites
        // the whole blob); settled (Completed / Cancelled) does not.
        assert_eq!(
            active_plan_for_type(&[plan(1, "age", MigrationStatus::Running)], "User"),
            Some(("User.age".into(), 1))
        );
        // A different field of the same type still blocks.
        assert_eq!(
            active_plan_for_type(&[plan(2, "score", MigrationStatus::AwaitingConverter)], "User"),
            Some(("User.score".into(), 2))
        );
        assert_eq!(
            active_plan_for_type(&[plan(3, "age", MigrationStatus::Completed)], "User"),
            None
        );
        assert_eq!(
            active_plan_for_type(&[plan(4, "age", MigrationStatus::Cancelled)], "User"),
            None
        );
        // Different type → no match.
        assert_eq!(
            active_plan_for_type(&[plan(5, "age", MigrationStatus::Running)], "Post"),
            None
        );
    }

    #[test]
    fn resumable_plan_for_kind_change_licenses_unsettled_matching_direction() {
        let plan = |id: u64, status: MigrationStatus, src: u8, target: u8| {
            let mut p = sample_plan();
            p.plan_id = id;
            p.type_name = "User".into();
            p.field_name = "age".into();
            p.status = status;
            p.src_kind = src;
            p.target_kind = target;
            p
        };
        use kind_byte::{SCALAR_F64, SCALAR_I64};
        // Any unsettled status (incl. Failed/AwaitingConverter) licenses the
        // exact src->target mismatch — otherwise a Failed plan bricks the type
        // (can't reopen with target schema).
        for st in [
            MigrationStatus::Running,
            MigrationStatus::Pending,
            MigrationStatus::Failed,
            MigrationStatus::AwaitingConverter,
        ] {
            assert_eq!(
                resumable_plan_for_kind_change(
                    &[plan(1, st, SCALAR_I64, SCALAR_F64)],
                    "User",
                    "age",
                    SCALAR_I64,
                    SCALAR_F64
                ),
                Some(1),
                "{st:?} should license its own kind change"
            );
        }
        // Settled plan does not license.
        assert_eq!(
            resumable_plan_for_kind_change(
                &[plan(2, MigrationStatus::Completed, SCALAR_I64, SCALAR_F64)],
                "User",
                "age",
                SCALAR_I64,
                SCALAR_F64
            ),
            None
        );
        // Direction mismatch does not license (guards against masking a real
        // accidental kind change).
        assert_eq!(
            resumable_plan_for_kind_change(
                &[plan(3, MigrationStatus::Running, SCALAR_I64, SCALAR_F64)],
                "User",
                "age",
                SCALAR_I64,
                kind_byte::SCALAR_U32
            ),
            None
        );
    }

    #[test]
    fn schema_digest_is_stable_across_runs() {
        let s = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("age", ScalarType::I64),
            ],
        )]);
        let a = compute_schema_digest(&s);
        let b = compute_schema_digest(&s);
        assert_eq!(a, b);
    }

    #[test]
    fn schema_digest_changes_when_field_type_changes() {
        let int_schema = schema_with(vec![("User", vec![scalar("age", ScalarType::I64)])]);
        let str_schema = schema_with(vec![("User", vec![scalar("age", ScalarType::String)])]);
        assert_ne!(
            compute_schema_digest(&int_schema),
            compute_schema_digest(&str_schema)
        );
    }

    #[test]
    fn schema_digest_changes_when_indexed_directive_added() {
        let plain = schema_with(vec![("User", vec![scalar("age", ScalarType::I64)])]);
        let indexed = schema_with(vec![("User", vec![scalar_indexed("age", ScalarType::I64)])]);
        assert_ne!(
            compute_schema_digest(&plain),
            compute_schema_digest(&indexed)
        );
    }

    #[test]
    fn schema_digest_changes_when_relation_target_changes() {
        let to_user = schema_with(vec![
            ("User", vec![]),
            ("Movie", vec![]),
            ("Rating", vec![relation("user", "User", false)]),
        ]);
        let to_movie = schema_with(vec![
            ("User", vec![]),
            ("Movie", vec![]),
            ("Rating", vec![relation("user", "Movie", false)]),
        ]);
        assert_ne!(
            compute_schema_digest(&to_user),
            compute_schema_digest(&to_movie)
        );
    }

    // -----------------------------------------------------------------
    // 9.2 Backfill correctness
    // -----------------------------------------------------------------

    /// The single most important guarantee in this module: backfill
    /// must produce byte-identical IDs to the legacy alphabetical
    /// algorithm, otherwise every existing object on disk decodes as
    /// garbage. We replicate the legacy 4-line loop inline as the
    /// oracle.
    #[test]
    fn backfill_from_alphabetical_matches_legacy_ids() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = bench_like_schema();

        let cat = load_or_initialize(&storage, &schema, false).unwrap();

        let (legacy_type, legacy_field, legacy_rel) = legacy_assign(&schema);
        assert_eq!(cat.type_ids, legacy_type);
        assert_eq!(cat.field_ids, legacy_field);
        assert_eq!(cat.rel_ids, legacy_rel);
    }

    fn legacy_assign(
        schema: &Schema,
    ) -> (
        HashMap<String, u64>,
        HashMap<String, u64>,
        HashMap<String, u64>,
    ) {
        let mut type_ids = HashMap::new();
        let mut field_ids = HashMap::new();
        let mut rel_ids = HashMap::new();
        let mut next_field = 1u64;
        let mut next_rel = 1u64;
        let mut names: Vec<_> = schema.types.keys().cloned().collect();
        names.sort();
        for (type_id, name) in (1u64..).zip(names.iter()) {
            type_ids.insert(name.clone(), type_id);
            let td = &schema.types[name];
            for field in &td.fields {
                let qual = format!("{}.{}", name, field.name);
                field_ids.insert(qual.clone(), next_field);
                next_field += 1;
                if matches!(field.field_type, FieldType::Relation(_)) {
                    rel_ids.insert(qual, next_rel);
                    next_rel += 1;
                }
            }
        }
        (type_ids, field_ids, rel_ids)
    }

    fn bench_like_schema() -> Schema {
        schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    scalar_indexed("birth_year", ScalarType::I64),
                    relation("ratings", "Rating", true),
                ],
            ),
            (
                "Movie",
                vec![
                    scalar("title", ScalarType::String),
                    scalar("year", ScalarType::I64),
                    relation("ratings", "Rating", true),
                    relation("director", "Director", false),
                ],
            ),
            ("Director", vec![scalar("name", ScalarType::String)]),
            (
                "Rating",
                vec![
                    scalar("stars", ScalarType::I64),
                    relation("user", "User", false),
                    relation("movie", "Movie", false),
                ],
            ),
        ])
    }

    #[test]
    fn backfill_writes_format_initialized_digest_and_counters() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = bench_like_schema();
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let snap = storage.read_snapshot();

        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_format())
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_initialized())
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_digest())
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_next_type())
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_next_field())
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .get_at(snap, &KeyBuilder::catalog_next_rel())
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn backfill_idempotent_on_repeat_open() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = bench_like_schema();

        let first = load_or_initialize(&storage, &schema, false).unwrap();
        let second = load_or_initialize(&storage, &schema, false).unwrap();
        assert_eq!(first.type_ids, second.type_ids);
        assert_eq!(first.field_ids, second.field_ids);
        assert_eq!(first.rel_ids, second.rel_ids);
        assert_eq!(first.next_type, second.next_type);
        assert_eq!(first.next_field, second.next_field);
        assert_eq!(first.next_rel, second.next_rel);
    }

    #[test]
    fn backfill_is_deterministic_modulo_timestamp() {
        let s = bench_like_schema();
        let d1 = TempDir::new().unwrap();
        let d2 = TempDir::new().unwrap();
        let s1 = open_lsm(&d1);
        let s2 = open_lsm(&d2);
        let c1 = load_or_initialize(&s1, &s, false).unwrap();
        let c2 = load_or_initialize(&s2, &s, false).unwrap();
        assert_eq!(c1.type_ids, c2.type_ids);
        assert_eq!(c1.field_ids, c2.field_ids);
        assert_eq!(c1.rel_ids, c2.rel_ids);
        assert_eq!(c1.field_kinds, c2.field_kinds);
    }

    // -----------------------------------------------------------------
    // 9.3 Additive drift — no renumbering
    // -----------------------------------------------------------------

    #[test]
    fn add_type_does_not_renumber_existing() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![
            ("Movie", vec![scalar("title", ScalarType::String)]),
            ("User", vec![scalar("name", ScalarType::String)]),
        ]);
        let before = load_or_initialize(&storage, &base, false).unwrap();
        let movie_id_before = before.type_ids["Movie"];
        let user_id_before = before.type_ids["User"];

        let extended = schema_with(vec![
            ("Movie", vec![scalar("title", ScalarType::String)]),
            ("User", vec![scalar("name", ScalarType::String)]),
            // 'Director' sorts before 'Movie' and 'User' alphabetically.
            // Pre-catalog code would renumber every existing type ID.
            ("Director", vec![scalar("name", ScalarType::String)]),
        ]);
        let after = load_or_initialize(&storage, &extended, false).unwrap();
        assert_eq!(after.type_ids["Movie"], movie_id_before);
        assert_eq!(after.type_ids["User"], user_id_before);
        assert!(after.type_ids["Director"] > movie_id_before.max(user_id_before));
    }

    #[test]
    fn add_field_does_not_renumber_existing() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let before = load_or_initialize(&storage, &base, false).unwrap();
        let name_id = before.field_ids["User.name"];

        let extended = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("age", ScalarType::I64),
            ],
        )]);
        let after = load_or_initialize(&storage, &extended, false).unwrap();
        assert_eq!(after.field_ids["User.name"], name_id);
        assert!(after.field_ids["User.age"] > name_id);
    }

    #[test]
    fn add_relation_does_not_renumber_existing() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let before = load_or_initialize(&storage, &base, false).unwrap();
        let user_id = before.type_ids["User"];

        let extended = schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    relation("favourite", "Movie", false),
                ],
            ),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let after = load_or_initialize(&storage, &extended, false).unwrap();
        assert_eq!(after.type_ids["User"], user_id);
        assert!(after.field_ids.contains_key("User.favourite"));
        assert!(after.rel_ids.contains_key("User.favourite"));
    }

    #[test]
    fn add_indexed_directive_does_not_renumber_existing() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![("User", vec![scalar("age", ScalarType::I64)])]);
        let before = load_or_initialize(&storage, &base, false).unwrap();
        let age_id = before.field_ids["User.age"];

        let with_index = schema_with(vec![("User", vec![scalar_indexed("age", ScalarType::I64)])]);
        let after = load_or_initialize(&storage, &with_index, false).unwrap();
        assert_eq!(after.field_ids["User.age"], age_id);
    }

    // -----------------------------------------------------------------
    // 9.4 Shrink refusal
    // -----------------------------------------------------------------

    #[test]
    fn drop_type_refused_without_flag() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let big = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &big, false).unwrap();
        let smaller = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let err = load_or_initialize(&storage, &smaller, false).unwrap_err();
        let EngineError::Catalog(CatalogError::SchemaShrinkRequiresOptIn {
            dropped_types, ..
        }) = err
        else {
            panic!("expected SchemaShrinkRequiresOptIn, got {err}");
        };
        assert_eq!(dropped_types, vec!["Movie".to_string()]);
    }

    /// With `allow_schema_shrink: true`, dropping a type now writes
    /// tombstones rather than erroring. Phase 2 behavior — phase 1
    /// would have errored with SchemaShrinkNotYetSupported here.
    #[test]
    fn drop_type_with_flag_tombstones_the_type() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let big = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &big, false).unwrap();
        let smaller = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let cat = load_or_initialize(&storage, &smaller, true).unwrap();
        // Movie's id is still in type_ids (retired IDs are never removed
        // — only marked) AND is in the tombstoned set.
        assert!(cat.type_ids.contains_key("Movie"));
        assert!(cat.tombstoned_type_names.contains("Movie"));
        let movie_id = cat.type_ids["Movie"];
        assert!(cat.tombstoned_type_ids.contains(&movie_id));
        assert_eq!(cat.format_version, CATALOG_FORMAT_V2);
    }

    /// Dropping a type cascades to ALL of its fields and relations in
    /// the same commit. Each cascaded entry is tagged with reason
    /// `CascadeParentRetired` so observability tells the right story.
    #[test]
    fn drop_type_cascades_to_its_fields_and_relations() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![
            (
                "Movie",
                vec![
                    scalar("title", ScalarType::String),
                    scalar("year", ScalarType::I64),
                    relation("director", "Director", false),
                ],
            ),
            ("Director", vec![scalar("name", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &base, false).unwrap();
        let smaller = schema_with(vec![(
            "Director",
            vec![scalar("name", ScalarType::String)],
        )]);
        let cat = load_or_initialize(&storage, &smaller, true).unwrap();
        // Type tombstoned + all child fields + the relation tombstoned.
        assert!(cat.tombstoned_type_names.contains("Movie"));
        assert!(cat.tombstoned_field_quals.contains("Movie.title"));
        assert!(cat.tombstoned_field_quals.contains("Movie.year"));
        assert!(cat.tombstoned_field_quals.contains("Movie.director"));
        assert!(cat.tombstoned_rel_quals.contains("Movie.director"));
        // Director is still live.
        assert!(!cat.tombstoned_type_names.contains("Director"));
        assert!(!cat.tombstoned_field_quals.contains("Director.name"));
        // Cascade reason recorded.
        let movie_entry = &cat.type_entries["Movie"];
        assert_eq!(movie_entry.retired_reason, Some(RetireReason::ExplicitShrink));
        let title_entry = &cat.field_entries["Movie.title"];
        assert_eq!(
            title_entry.retired_reason,
            Some(RetireReason::CascadeParentRetired)
        );
    }

    /// Re-applying the same shrunk schema is idempotent — the second
    /// open is a no-op via the digest fast-path. Tombstone metadata
    /// (retired_at_ms etc.) is NOT re-stamped.
    #[test]
    fn reapplying_shrunk_schema_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let big = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &big, false).unwrap();
        let smaller = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let first = load_or_initialize(&storage, &smaller, true).unwrap();
        let first_retired_at = first.type_entries["Movie"].retired_at_ms;
        // Sleep is forbidden; just open again and check the timestamp is unchanged.
        let second = load_or_initialize(&storage, &smaller, true).unwrap();
        assert_eq!(
            second.type_entries["Movie"].retired_at_ms,
            first_retired_at,
            "re-applying shrunk schema must not re-stamp retired_at_ms"
        );
    }

    /// Reusing the name of a retired type/field/relation in a fresh
    /// schema is refused with `NameReuseOfRetiredEntry`. The operator
    /// must pick a different name or restore from a backup.
    #[test]
    fn rebinding_retired_type_name_is_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let big = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &big, false).unwrap();
        // Retire Movie.
        let shrunk = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &shrunk, true).unwrap();
        // Operator tries to re-add Movie with the same name. Refuse.
        let resurrection = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let err = load_or_initialize(&storage, &resurrection, true).unwrap_err();
        let EngineError::Catalog(CatalogError::NameReuseOfRetiredEntry { kind, name, .. }) = err
        else {
            panic!("expected NameReuseOfRetiredEntry, got {err}");
        };
        assert_eq!(kind, "type");
        assert_eq!(name, "Movie");
    }

    /// The first tombstone write bumps the catalog format to v2. A
    /// v1-only binary must refuse to open the result.
    #[test]
    fn first_tombstone_bumps_catalog_format_to_v2() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let big = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_pre = load_or_initialize(&storage, &big, false).unwrap();
        assert_eq!(cat_pre.format_version, CATALOG_FORMAT_V1);
        let smaller = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let cat_post = load_or_initialize(&storage, &smaller, true).unwrap();
        assert_eq!(cat_post.format_version, CATALOG_FORMAT_V2);
    }

    /// A v1 catalog whose row carries `status=Tombstoned` is internally
    /// inconsistent; the loader refuses rather than guessing.
    #[test]
    fn v1_catalog_with_tombstoned_row_is_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        // Seed a v1 catalog.
        let schema = schema_with(vec![("X", vec![scalar("a", ScalarType::I64)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        // Manually inject a tombstoned type row WITHOUT bumping c:F:.
        let now = 1_000_000_000_000u64;
        let mut bad = IdEntry::backfilled(99, kind_byte::UNSET, now);
        bad.tombstone(now, RetireReason::ExplicitShrink);
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_type("Bad"),
                encode_id_entry(&bad),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();
        let err = load_or_initialize(&storage, &schema, false).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::TombstoneOnV1Catalog { .. })
        ));
    }

    /// Tombstone TLVs round-trip through encode/decode.
    #[test]
    fn tombstone_tlvs_roundtrip() {
        let mut entry = IdEntry::fresh(5, kind_byte::SCALAR_I64, 100);
        entry.tombstone(1_700_000_000_000, RetireReason::CascadeParentRetired);
        let bytes = encode_id_entry(&entry);
        let back = decode_id_entry("c:T:X", &bytes).unwrap();
        assert_eq!(back, entry);
        assert_eq!(back.status, TombstoneStatus::Tombstoned);
        assert_eq!(back.retired_at_ms, Some(1_700_000_000_000));
        assert_eq!(
            back.retired_reason,
            Some(RetireReason::CascadeParentRetired)
        );
    }

    /// A live entry under V2 is byte-identical to a V1 entry — critical
    /// for the digest fast-path to stay hot across the format bump.
    #[test]
    fn live_entry_under_v2_is_byte_equal_to_v1() {
        let live = IdEntry::fresh(42, kind_byte::SCALAR_STRING, 99);
        let bytes_v1 = encode_id_entry(&live);
        // The encoder writes tombstone TLVs only when status == Tombstoned.
        // A live entry under V2 must encode to exactly the same bytes.
        let mut as_v2 = live.clone();
        as_v2.status = TombstoneStatus::Live; // unchanged
        let bytes_v2 = encode_id_entry(&as_v2);
        assert_eq!(bytes_v1, bytes_v2);
    }

    /// An unknown tombstone-status byte (e.g. `0x02` reserved) is
    /// refused rather than silently treated as live.
    #[test]
    fn decode_id_entry_rejects_unknown_tombstone_status() {
        // Hand-build a payload with TLV_STATUS=0x02.
        let mut body: Vec<u8> = Vec::new();
        write_tlv(&mut body, TLV_ID, &7u64.to_be_bytes());
        write_tlv(&mut body, TLV_ASSIGNED_AT, &0u64.to_be_bytes());
        write_tlv(&mut body, TLV_ASSIGNED_BY, &[0]);
        write_tlv(&mut body, TLV_KIND, &[kind_byte::SCALAR_I64]);
        write_tlv(&mut body, TLV_STATUS, &[0x02]);
        let mut bytes: Vec<u8> = vec![RECORD_FORMAT_V1, KIND_ID_ENTRY];
        bytes.extend_from_slice(&(body.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&body);
        let err = decode_id_entry("c:T:X", &bytes).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::UnknownTombstoneStatus { status: 0x02, .. })
        ));
    }

    // -----------------------------------------------------------------
    // Rename (card 3/5) — encoding + apply_migration tests
    // -----------------------------------------------------------------

    /// previous_names TLV roundtrips through encode/decode unchanged.
    #[test]
    fn rename_chain_roundtrip() {
        let mut entry = IdEntry::fresh(7, kind_byte::UNSET, 100);
        entry.previous_names = vec![
            RenameRecord { from: "User".into(), to: "Account".into(), wall_time_unix_ms: 1_700_000_000_000 },
            RenameRecord { from: "Person".into(), to: "User".into(), wall_time_unix_ms: 1_600_000_000_000 },
        ];
        entry.last_renamed_at_ms = Some(1_700_000_000_000);
        let bytes = encode_id_entry(&entry);
        let back = decode_id_entry("c:T:Account", &bytes).unwrap();
        assert_eq!(back, entry);
    }

    /// A never-renamed entry under v3 must encode byte-identical to a
    /// pre-rename entry — preserves digest fast-path locality.
    #[test]
    fn never_renamed_entry_under_v3_is_byte_equal_to_pre_v3() {
        let entry = IdEntry::fresh(42, kind_byte::SCALAR_STRING, 99);
        let bytes_a = encode_id_entry(&entry);
        let mut other = entry.clone();
        other.previous_names = Vec::new();
        other.last_renamed_at_ms = None;
        let bytes_b = encode_id_entry(&other);
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn rename_type_preserves_type_id() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema_before = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_before = load_or_initialize(&storage, &schema_before, false).unwrap();
        let user_id = cat_before.type_ids["User"];

        let report = apply_migration(
            &storage,
            &schema_before,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Account".into(),
            }],
        )
        .unwrap();
        assert_eq!(report.renamed_types.len(), 1);
        assert_eq!(report.renamed_types[0].id, user_id);
        assert_eq!(report.catalog_format_after, CATALOG_FORMAT_V3);

        // Reopen with post-rename schema. Account inherits User's ID.
        let schema_after = schema_with(vec![
            ("Account", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_after = load_or_initialize(&storage, &schema_after, false).unwrap();
        assert_eq!(cat_after.type_ids["Account"], user_id);
        // Old name is gone from type_ids.
        assert!(!cat_after.type_ids.contains_key("User"));
    }

    #[test]
    fn rename_type_cascades_child_field_and_relation_rows() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema_before = schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    relation("favourite", "Movie", false),
                ],
            ),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_before = load_or_initialize(&storage, &schema_before, false).unwrap();
        let name_id = cat_before.field_ids["User.name"];
        let fav_field_id = cat_before.field_ids["User.favourite"];
        let fav_rel_id = cat_before.rel_ids["User.favourite"];

        apply_migration(
            &storage,
            &schema_before,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Account".into(),
            }],
        )
        .unwrap();

        // Reopen with the post-rename schema. Field & relation IDs are
        // preserved under the new qualified names.
        let schema_after = schema_with(vec![
            (
                "Account",
                vec![
                    scalar("name", ScalarType::String),
                    relation("favourite", "Movie", false),
                ],
            ),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_after = load_or_initialize(&storage, &schema_after, false).unwrap();
        assert_eq!(cat_after.field_ids["Account.name"], name_id);
        assert_eq!(cat_after.field_ids["Account.favourite"], fav_field_id);
        assert_eq!(cat_after.rel_ids["Account.favourite"], fav_rel_id);
        // Old qualified names are GONE — no orphaned children.
        assert!(!cat_after.field_ids.contains_key("User.name"));
        assert!(!cat_after.field_ids.contains_key("User.favourite"));
        assert!(!cat_after.rel_ids.contains_key("User.favourite"));
    }

    #[test]
    fn rename_type_records_audit_chain() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Account".into(),
            }],
        )
        .unwrap();
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "Account".into(),
                new: "Member".into(),
            }],
        )
        .unwrap();

        // Re-open. The Member row's previous_names chain shows the
        // full audit history.
        let schema_after =
            schema_with(vec![("Member", vec![scalar("name", ScalarType::String)])]);
        let cat = load_or_initialize(&storage, &schema_after, false).unwrap();
        let entry = &cat.type_entries["Member"];
        assert_eq!(entry.previous_names.len(), 2);
        // Most recent first.
        assert_eq!(entry.previous_names[0].from, "Account");
        assert_eq!(entry.previous_names[0].to, "Member");
        assert_eq!(entry.previous_names[1].from, "User");
        assert_eq!(entry.previous_names[1].to, "Account");
    }

    #[test]
    fn rename_type_source_not_found_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "Missing".into(),
                new: "Other".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameSourceNotFound { kind: "type", .. })
        ));
    }

    #[test]
    fn rename_type_target_collision_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Movie".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameTargetCollision { kind: "type", .. })
        ));
    }

    #[test]
    fn rename_type_target_is_retired_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        // Retire Movie.
        let smaller =
            schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &smaller, true).unwrap();
        // Try to rename User → Movie (Movie is tombstoned).
        let err = apply_migration(
            &storage,
            &smaller,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Movie".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameTargetIsRetired { kind: "type", .. })
        ));
    }

    #[test]
    fn rename_type_no_op_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "User".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameNoOp { kind: "type", .. })
        ));
    }

    #[test]
    fn rename_type_first_rename_bumps_format_to_v3() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        assert_eq!(cat_pre.format_version, CATALOG_FORMAT_V1);
        let report = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Account".into(),
            }],
        )
        .unwrap();
        assert_eq!(report.catalog_format_before, CATALOG_FORMAT_V1);
        assert_eq!(report.catalog_format_after, CATALOG_FORMAT_V3);
    }

    /// Renaming a type after the source-type was retired refuses
    /// cleanly — retired entries cannot be renamed.
    #[test]
    fn rename_type_source_retired_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![
            ("User", vec![scalar("name", ScalarType::String)]),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let smaller =
            schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let _ = load_or_initialize(&storage, &smaller, true).unwrap();
        // Movie is tombstoned. Try to rename it.
        let err = apply_migration(
            &storage,
            &smaller,
            &[RenameVerb::Type {
                old: "Movie".into(),
                new: "Film".into(),
            }],
        )
        .unwrap_err();
        // The entry exists in type_entries but is tombstoned, so we
        // get RenameSourceRetired (not RenameSourceNotFound).
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameSourceRetired { kind: "type", .. })
        ));
    }

    /// After rename, the old name is FREE — operator can add a new
    /// type at the old name and it gets a fresh numeric id.
    #[test]
    fn old_name_is_free_after_rename_and_allocates_fresh_id_if_readded() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema_before =
            schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let cat_pre = load_or_initialize(&storage, &schema_before, false).unwrap();
        let original_user_id = cat_pre.type_ids["User"];

        apply_migration(
            &storage,
            &schema_before,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "Account".into(),
            }],
        )
        .unwrap();

        // Re-open with a schema that has BOTH the renamed-to name AND
        // a fresh type at the old name.
        let schema_after = schema_with(vec![
            ("Account", vec![scalar("name", ScalarType::String)]),
            ("User", vec![scalar("name", ScalarType::String)]),
        ]);
        let cat_after = load_or_initialize(&storage, &schema_after, false).unwrap();
        assert_eq!(cat_after.type_ids["Account"], original_user_id);
        // The new User must have a fresh id, NOT the original.
        assert_ne!(cat_after.type_ids["User"], original_user_id);
        // No tombstone — rename doesn't tombstone.
        assert!(!cat_after.tombstoned_type_names.contains("User"));
    }

    // -----------------------------------------------------------------
    // rename_field (card 3/5 phase 2) — apply_migration tests
    // -----------------------------------------------------------------

    #[test]
    fn rename_field_preserves_field_id_and_bumps_format_to_v5() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema_before = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("age", ScalarType::U32),
            ],
        )]);
        let cat_before = load_or_initialize(&storage, &schema_before, false).unwrap();
        let name_id = cat_before.field_ids["User.name"];

        let report = apply_migration(
            &storage,
            &schema_before,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "handle".into(),
            }],
        )
        .unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        let rep = &report.renamed_fields[0];
        assert_eq!(rep.from, "name");
        assert_eq!(rep.to, "handle");
        assert_eq!(rep.field_id, name_id);
        assert_eq!(report.catalog_format_after, CATALOG_FORMAT_V5);

        // Reopen with the post-rename schema. Field ID is preserved under
        // the new qualified name; the old qualified name is gone.
        let schema_after = schema_with(vec![(
            "User",
            vec![
                scalar("handle", ScalarType::String),
                scalar("age", ScalarType::U32),
            ],
        )]);
        let cat_after = load_or_initialize(&storage, &schema_after, false).unwrap();
        assert_eq!(cat_after.field_ids["User.handle"], name_id);
        assert!(!cat_after.field_ids.contains_key("User.name"));
    }

    #[test]
    fn rename_field_records_audit_chain() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "handle".into(),
            }],
        )
        .unwrap();
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "handle".into(),
                new: "username".into(),
            }],
        )
        .unwrap();

        // Re-open with the post-rename schema; the field's
        // previous_names chain shows the full audit history.
        let schema_after = schema_with(vec![(
            "User",
            vec![scalar("username", ScalarType::String)],
        )]);
        let cat = load_or_initialize(&storage, &schema_after, false).unwrap();
        let entry = &cat.field_entries["User.username"];
        assert_eq!(entry.previous_names.len(), 2);
        // Most recent first.
        assert_eq!(entry.previous_names[0].from, "handle");
        assert_eq!(entry.previous_names[0].to, "username");
        assert_eq!(entry.previous_names[1].from, "name");
        assert_eq!(entry.previous_names[1].to, "handle");
    }

    #[test]
    fn rename_field_source_not_found_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "missing".into(),
                new: "other".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameSourceNotFound { kind: "field", .. })
        ));
    }

    #[test]
    fn rename_field_parent_type_not_found_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "Missing".into(),
                old: "name".into(),
                new: "handle".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameSourceNotFound { kind: "type", .. })
        ));
    }

    #[test]
    fn rename_field_target_collision_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("nickname", ScalarType::String),
            ],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "nickname".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameTargetCollision { kind: "field", .. })
        ));
    }

    #[test]
    fn rename_field_no_op_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "name".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameNoOp { kind: "field", .. })
        ));
    }

    #[test]
    fn rename_field_new_name_with_double_underscore_refused() {
        // `check_identifier` must reject `__` anywhere — otherwise
        // rename_field can mint a name that the schema parser will refuse
        // on the next open, locking the operator out. Regression for the
        // PR #6 adversarial-review finding.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "__internal".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::ReservedDoubleUnderscoreInIdentifier { .. })
        ));
    }

    #[test]
    fn rename_type_new_name_with_double_underscore_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let err = apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Type {
                old: "User".into(),
                new: "User__internal".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::ReservedDoubleUnderscoreInIdentifier { .. })
        ));
    }

    /// Chain of field renames in a single `apply_migration` plan must
    /// refuse the entire plan up front if the underlying field carries
    /// `@indexed` — even when the schema only shows the CHAIN-TERMINAL
    /// name (and so the OLD per-verb directive check would miss the
    /// intermediate verb whose `old`/`new` aren't in the schema).
    /// Regression for PR #6 adversarial-review finding #5.
    #[test]
    fn rename_field_chained_indexed_succeeds() {
        // Phase 3: a field can be renamed THROUGH a chain when each link is a
        // SEPARATE migration — the safe pattern, since each commits on its own
        // snapshot. (A single plan with both verbs is refused — see
        // `rename_plan_multi_verb_same_type_refused`.) field_id is preserved.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let pre = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar_indexed("year", ScalarType::U32),
            ],
        )]);
        let cat_pre = load_or_initialize(&storage, &pre, false).unwrap();
        let orig_id = cat_pre.field_ids["Movie.year"];

        let mid = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar_indexed("released_in", ScalarType::U32),
            ],
        )]);
        let post = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar_indexed("year_released", ScalarType::U32),
            ],
        )]);
        apply_migration(
            &storage,
            &mid,
            &[RenameVerb::Field {
                type_name: "Movie".into(),
                old: "year".into(),
                new: "released_in".into(),
            }],
        )
        .unwrap();
        apply_migration(
            &storage,
            &post,
            &[RenameVerb::Field {
                type_name: "Movie".into(),
                old: "released_in".into(),
                new: "year_released".into(),
            }],
        )
        .unwrap();
        let cat_after = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat_after.field_ids.contains_key("Movie.year"));
        assert!(!cat_after.field_ids.contains_key("Movie.released_in"));
        assert_eq!(cat_after.field_ids["Movie.year_released"], orig_id);
    }

    #[test]
    fn rename_plan_multi_verb_two_fields_same_type_succeeds() {
        // Overboard cmqgvlf6b: a single plan may now rename TWO fields of one
        // type. (Formerly refused with RenameMultiVerbSameType.) Both land and
        // each field_id is preserved. Catalog-only check (no objects/covers); the
        // object + @indexed-cover behaviour is exercised in the database tests.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar("year", ScalarType::U32),
            ],
        )]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        let year_id = cat_pre.field_ids["Movie.year"];
        let title_id = cat_pre.field_ids["Movie.title"];
        apply_migration(
            &storage,
            &schema,
            &[
                RenameVerb::Field {
                    type_name: "Movie".into(),
                    old: "year".into(),
                    new: "released_in".into(),
                },
                RenameVerb::Field {
                    type_name: "Movie".into(),
                    old: "title".into(),
                    new: "name".into(),
                },
            ],
        )
        .unwrap();
        // Load with the TERMINAL schema (the rename left the names re-keyed).
        let post = schema_with(vec![(
            "Movie",
            vec![
                scalar("name", ScalarType::String),
                scalar("released_in", ScalarType::U32),
            ],
        )]);
        let cat = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat.field_ids.contains_key("Movie.year"));
        assert!(!cat.field_ids.contains_key("Movie.title"));
        assert_eq!(cat.field_ids["Movie.released_in"], year_id);
        assert_eq!(cat.field_ids["Movie.name"], title_id);
    }

    #[test]
    fn rename_type_and_field_same_type_one_plan_refused() {
        // Out of cmqgvlf6b's scope (field renames). A TYPE rename + a FIELD rename
        // of that type in ONE plan stays refused (either order): the field verb's
        // @indexed cover maintainer keys by the pre-rename type name, which the
        // same-plan type rename makes stale. Split into separate migrations.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("Movie", vec![scalar("year", ScalarType::U32)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        // Type first, field on the NEW type name.
        let err = apply_migration(
            &storage,
            &schema,
            &[
                RenameVerb::Type {
                    old: "Movie".into(),
                    new: "Film".into(),
                },
                RenameVerb::Field {
                    type_name: "Film".into(),
                    old: "year".into(),
                    new: "released_in".into(),
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameTypeWithFieldSamePlan { .. })
        ));

        // Field first (on the OLD type name), type second — also refused.
        let err2 = apply_migration(
            &storage,
            &schema,
            &[
                RenameVerb::Field {
                    type_name: "Movie".into(),
                    old: "year".into(),
                    new: "released_in".into(),
                },
                RenameVerb::Type {
                    old: "Movie".into(),
                    new: "Film".into(),
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(
            err2,
            EngineError::Catalog(CatalogError::RenameTypeWithFieldSamePlan { .. })
        ));

        // Nothing committed: original names intact.
        let cat = load_or_initialize(&storage, &schema, false).unwrap();
        assert!(cat.type_ids.contains_key("Movie"));
        assert!(cat.field_ids.contains_key("Movie.year"));
    }

    #[test]
    fn write_overlay_layers_reads_and_net_sets() {
        // Unit test for the WriteOverlay mechanics, incl. the None-in-snap-scan
        // removal branch that the rename verbs don't otherwise exercise (object/
        // cover/rev-edge keys are put-only).
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        // Seed three keys under a common prefix at a snapshot.
        let pfx: &[u8] = b"o:test:";
        let k1 = Bytes::from_static(b"o:test:1");
        let k2 = Bytes::from_static(b"o:test:2");
        let k3 = Bytes::from_static(b"o:test:3");
        let mut txn = storage.begin_txn();
        storage
            .put_batch(
                &mut txn,
                &[
                    (k1.clone(), Bytes::from_static(b"v1")),
                    (k2.clone(), Bytes::from_static(b"v2")),
                    (k3.clone(), Bytes::from_static(b"v3")),
                ],
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();
        let snap = storage.begin_txn().snapshot();

        let mut overlay = WriteOverlay::new();
        // verb-1-style writes: override k1, tombstone k2, leave k3, add k4.
        let k4 = Bytes::from_static(b"o:test:4");
        overlay.absorb(
            &[
                (k1.clone(), Bytes::from_static(b"v1b")),
                (k4.clone(), Bytes::from_static(b"v4")),
            ],
            std::slice::from_ref(&k2),
        );

        // get_at: override wins, tombstone reads None, untouched falls through.
        assert_eq!(
            overlay.get_at(&storage, snap, &k1).unwrap().as_deref(),
            Some(&b"v1b"[..])
        );
        assert_eq!(overlay.get_at(&storage, snap, &k2).unwrap(), None);
        assert_eq!(
            overlay.get_at(&storage, snap, &k3).unwrap().as_deref(),
            Some(&b"v3"[..])
        );
        assert_eq!(
            overlay.get_at(&storage, snap, &k4).unwrap().as_deref(),
            Some(&b"v4"[..])
        );

        // scan_prefix_at: k1 overridden, k2 removed (tombstone branch), k3 base,
        // k4 added — sorted by key.
        let scanned = overlay.scan_prefix_at(&storage, snap, pfx).unwrap();
        assert_eq!(
            scanned,
            vec![
                (k1.clone(), Bytes::from_static(b"v1b")),
                (k3.clone(), Bytes::from_static(b"v3")),
                (k4.clone(), Bytes::from_static(b"v4")),
            ]
        );

        // net_sets: puts sorted, the tombstoned key in deletes.
        let (puts, deletes) = overlay.net_sets();
        assert_eq!(
            puts,
            vec![
                (k1, Bytes::from_static(b"v1b")),
                (k4, Bytes::from_static(b"v4")),
            ]
        );
        assert_eq!(deletes, vec![k2]);
    }

    #[test]
    fn rename_field_indexed_succeeds() {
        // Phase 3: renaming an @indexed field is allowed. The i: keys are
        // field_id-keyed (preserved); covering payloads are refreshed by the
        // Database maintainer (exercised end-to-end in the database tests). At
        // the catalog layer there are no objects, so assert the catalog re-key.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar_indexed("year", ScalarType::U32),
            ],
        )]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        let orig_id = cat_pre.field_ids["Movie.year"];
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "Movie".into(),
                old: "year".into(),
                new: "released_in".into(),
            }],
        )
        .unwrap();
        let post = schema_with(vec![(
            "Movie",
            vec![
                scalar("title", ScalarType::String),
                scalar_indexed("released_in", ScalarType::U32),
            ],
        )]);
        let cat_after = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat_after.field_ids.contains_key("Movie.year"));
        assert_eq!(cat_after.field_ids["Movie.released_in"], orig_id);
    }

    #[test]
    fn rename_field_unique_succeeds() {
        // Phase 3: renaming a @unique field is allowed. u: keys are field_id-
        // keyed with the object_id as value — nothing on disk references the
        // name, so it's a pure catalog re-key.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar_unique("email", ScalarType::String),
            ],
        )]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        let orig_id = cat_pre.field_ids["User.email"];
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "email".into(),
                new: "email_addr".into(),
            }],
        )
        .unwrap();
        let post = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar_unique("email_addr", ScalarType::String),
            ],
        )]);
        let cat_after = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat_after.field_ids.contains_key("User.email"));
        assert_eq!(cat_after.field_ids["User.email_addr"], orig_id);
    }

    #[test]
    fn rename_field_vectorize_succeeds() {
        // Phase 3: renaming a Vector field carrying @vectorize is allowed.
        // v:/s: keys are field_id-keyed (stable); the directive's `source`
        // ref points at another field by name and is unchanged. Pure re-key.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "Doc",
            vec![
                scalar("body", ScalarType::String),
                vector_vectorize("embedding", 8, "body"),
            ],
        )]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        let orig_id = cat_pre.field_ids["Doc.embedding"];
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "Doc".into(),
                old: "embedding".into(),
                new: "embedding_v2".into(),
            }],
        )
        .unwrap();
        let post = schema_with(vec![(
            "Doc",
            vec![
                scalar("body", ScalarType::String),
                vector_vectorize("embedding_v2", 8, "body"),
            ],
        )]);
        let cat_after = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat_after.field_ids.contains_key("Doc.embedding"));
        assert_eq!(cat_after.field_ids["Doc.embedding_v2"], orig_id);
    }

    #[test]
    fn rename_field_relation_succeeds() {
        // Phase 3: relation fields can be renamed (routed to
        // apply_relation_rename_verb). rel_id is preserved; old name freed.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    relation("favourite", "Movie", false),
                ],
            ),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_pre = load_or_initialize(&storage, &schema, false).unwrap();
        let orig_rel_id = cat_pre.rel_ids["User.favourite"];
        apply_migration(
            &storage,
            &schema,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "favourite".into(),
                new: "pick".into(),
            }],
        )
        .unwrap();
        let post = schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    relation("pick", "Movie", false),
                ],
            ),
            ("Movie", vec![scalar("title", ScalarType::String)]),
        ]);
        let cat_after = load_or_initialize(&storage, &post, false).unwrap();
        assert!(!cat_after.rel_ids.contains_key("User.favourite"));
        assert_eq!(cat_after.rel_ids["User.pick"], orig_rel_id);
    }

    #[test]
    fn rename_field_old_name_freed_for_reuse() {
        // After rename, the old name should be free to allocate a fresh
        // field at — same precedent as type rename. We don't tombstone.
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema_before = schema_with(vec![(
            "User",
            vec![scalar("name", ScalarType::String)],
        )]);
        let cat_pre = load_or_initialize(&storage, &schema_before, false).unwrap();
        let orig_field_id = cat_pre.field_ids["User.name"];

        apply_migration(
            &storage,
            &schema_before,
            &[RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "handle".into(),
            }],
        )
        .unwrap();

        // Re-open with a schema that has BOTH the renamed-to name AND
        // a fresh field at the old name.
        let schema_after = schema_with(vec![(
            "User",
            vec![
                scalar("handle", ScalarType::String),
                scalar("name", ScalarType::String),
            ],
        )]);
        let cat_after = load_or_initialize(&storage, &schema_after, false).unwrap();
        assert_eq!(cat_after.field_ids["User.handle"], orig_field_id);
        // The new `name` must have a fresh id, NOT the original.
        assert_ne!(cat_after.field_ids["User.name"], orig_field_id);
    }

    #[test]
    fn rename_chain_cap_overflow_refused() {
        // Hand-construct an entry whose chain is already at the cap.
        let mut entry = IdEntry::fresh(1, kind_byte::UNSET, 0);
        entry.previous_names = (0..MAX_RENAME_HISTORY)
            .map(|i| RenameRecord {
                from: format!("Name{i}"),
                to: format!("Name{}", i + 1),
                wall_time_unix_ms: i as u64,
            })
            .collect();
        // Build a Catalog with this one type at name "Name32" (the last `to`).
        let mut cat = Catalog::empty();
        cat.format_version = CATALOG_FORMAT_V3;
        cat.insert_type("Name32".to_string(), entry);
        let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
        let mut deletes: Vec<Bytes> = Vec::new();
        let mut report = MigrationReport::default();
        let err = apply_type_rename_verb(
            &mut cat,
            "Name32",
            "Name33",
            999,
            &mut puts,
            &mut deletes,
            &mut report,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameHistoryCapExceeded {
                kind: "type",
                cap: MAX_RENAME_HISTORY,
                ..
            })
        ));
    }

    #[test]
    fn rename_chain_decoder_rejects_count_zero() {
        let body = vec![0u8]; // count = 0
        let err = decode_rename_chain("c:T:X", &body).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::EmptyRenameChain { .. })
        ));
    }

    #[test]
    fn rename_chain_decoder_rejects_count_over_cap() {
        let body = vec![(MAX_RENAME_HISTORY as u8) + 1];
        let err = decode_rename_chain("c:T:X", &body).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::RenameChainOverCap { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Type-change history (card 4/5) — encoding tests
    // -----------------------------------------------------------------

    #[test]
    fn type_change_chain_roundtrip() {
        let mut entry = IdEntry::fresh(7, kind_byte::SCALAR_I64, 100);
        entry.type_change_history = vec![
            TypeChangeRecord {
                from_kind: kind_byte::SCALAR_F64,
                to_kind: kind_byte::SCALAR_STRING,
                wall_time_unix_ms: 200,
            },
            TypeChangeRecord {
                from_kind: kind_byte::SCALAR_I64,
                to_kind: kind_byte::SCALAR_F64,
                wall_time_unix_ms: 100,
            },
        ];
        entry.last_type_change_at_ms = Some(200);
        let bytes = encode_id_entry(&entry);
        let back = decode_id_entry("c:E:User.x", &bytes).unwrap();
        assert_eq!(back, entry);
    }

    /// A never-migrated field under v4 encodes byte-identical to a
    /// pre-v4 field — keeps the digest fast-path hot.
    #[test]
    fn never_migrated_field_under_v4_is_byte_equal_to_pre_v4() {
        let entry = IdEntry::fresh(42, kind_byte::SCALAR_I64, 99);
        let bytes_a = encode_id_entry(&entry);
        let mut other = entry.clone();
        other.type_change_history.clear();
        other.last_type_change_at_ms = None;
        let bytes_b = encode_id_entry(&other);
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn type_change_chain_decoder_rejects_count_zero() {
        let body = vec![0u8];
        let err = decode_type_change_chain("c:E:User.x", &body).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::EmptyTypeChangeChain { .. })
        ));
    }

    #[test]
    fn type_change_chain_decoder_rejects_count_over_cap() {
        let body = vec![(MAX_TYPE_CHANGE_HISTORY as u8) + 1];
        let err = decode_type_change_chain("c:E:User.x", &body).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::TypeChangeChainOverCap { .. })
        ));
    }

    #[test]
    fn drop_field_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let base = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("age", ScalarType::I64),
            ],
        )]);
        let _ = load_or_initialize(&storage, &base, false).unwrap();
        let smaller = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let err = load_or_initialize(&storage, &smaller, false).unwrap_err();
        let EngineError::Catalog(CatalogError::SchemaShrinkRequiresOptIn {
            dropped_fields, ..
        }) = err
        else {
            panic!("expected SchemaShrinkRequiresOptIn, got {err}");
        };
        assert_eq!(dropped_fields, vec!["User.age".to_string()]);
    }

    // -----------------------------------------------------------------
    // 9.5 Kind-change detection
    // -----------------------------------------------------------------

    #[test]
    fn field_kind_change_int_to_string_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let int_schema = schema_with(vec![("User", vec![scalar("age", ScalarType::I64)])]);
        let _ = load_or_initialize(&storage, &int_schema, false).unwrap();
        let str_schema = schema_with(vec![("User", vec![scalar("age", ScalarType::String)])]);
        let err = load_or_initialize(&storage, &str_schema, false).unwrap_err();
        let EngineError::Catalog(CatalogError::FieldKindChanged {
            qualified,
            was,
            now,
        }) = err
        else {
            panic!("expected FieldKindChanged, got {err}");
        };
        assert_eq!(qualified, "User.age");
        assert_eq!(was, "Scalar(I64)");
        assert_eq!(now, "Scalar(String)");
    }

    #[test]
    fn field_kind_change_scalar_to_relation_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let scalar_schema = schema_with(vec![
            ("User", vec![scalar("friend", ScalarType::I64)]),
            ("Other", vec![]),
        ]);
        let _ = load_or_initialize(&storage, &scalar_schema, false).unwrap();
        let rel_schema = schema_with(vec![
            ("User", vec![relation("friend", "Other", false)]),
            ("Other", vec![]),
        ]);
        let err = load_or_initialize(&storage, &rel_schema, false).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::FieldKindChanged { .. })
        ));
    }

    // -----------------------------------------------------------------
    // 9.6 Concurrency
    // -----------------------------------------------------------------

    #[test]
    fn concurrent_open_is_safe() {
        let dir = TempDir::new().unwrap();
        let storage = std::sync::Arc::new(open_lsm(&dir));
        let schema = std::sync::Arc::new(bench_like_schema());

        // Backfill once so both threads enter the load path.
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        // Two threads each add a DIFFERENT field. Their additive
        // deltas don't overlap on the new c:E:* rows, but both rewrite
        // c:D: and bump c:N:E, which forces write-set intersection and
        // triggers the internal retry path.
        let s1 = std::sync::Arc::clone(&storage);
        let s2 = std::sync::Arc::clone(&storage);
        let h1 = std::thread::spawn(move || {
            let extended = schema_with(vec![
                (
                    "User",
                    vec![
                        scalar("name", ScalarType::String),
                        scalar_indexed("birth_year", ScalarType::I64),
                        scalar("alpha", ScalarType::I64),
                        relation("ratings", "Rating", true),
                    ],
                ),
                (
                    "Movie",
                    vec![
                        scalar("title", ScalarType::String),
                        scalar("year", ScalarType::I64),
                        relation("ratings", "Rating", true),
                        relation("director", "Director", false),
                    ],
                ),
                ("Director", vec![scalar("name", ScalarType::String)]),
                (
                    "Rating",
                    vec![
                        scalar("stars", ScalarType::I64),
                        relation("user", "User", false),
                        relation("movie", "Movie", false),
                    ],
                ),
            ]);
            load_or_initialize(&s1, &extended, false)
        });
        let h2 = std::thread::spawn(move || {
            let extended = schema_with(vec![
                (
                    "User",
                    vec![
                        scalar("name", ScalarType::String),
                        scalar_indexed("birth_year", ScalarType::I64),
                        scalar("beta", ScalarType::I64),
                        relation("ratings", "Rating", true),
                    ],
                ),
                (
                    "Movie",
                    vec![
                        scalar("title", ScalarType::String),
                        scalar("year", ScalarType::I64),
                        relation("ratings", "Rating", true),
                        relation("director", "Director", false),
                    ],
                ),
                ("Director", vec![scalar("name", ScalarType::String)]),
                (
                    "Rating",
                    vec![
                        scalar("stars", ScalarType::I64),
                        relation("user", "User", false),
                        relation("movie", "Movie", false),
                    ],
                ),
            ]);
            load_or_initialize(&s2, &extended, false)
        });

        // One or both may succeed; the loser may exhaust retries or
        // succeed after retry. Either way, the final catalog must
        // have no duplicate IDs and counters must be self-consistent.
        let _ = (h1.join().unwrap(), h2.join().unwrap());

        // Reopen with the union schema and validate.
        let union = schema_with(vec![
            (
                "User",
                vec![
                    scalar("name", ScalarType::String),
                    scalar_indexed("birth_year", ScalarType::I64),
                    scalar("alpha", ScalarType::I64),
                    scalar("beta", ScalarType::I64),
                    relation("ratings", "Rating", true),
                ],
            ),
            (
                "Movie",
                vec![
                    scalar("title", ScalarType::String),
                    scalar("year", ScalarType::I64),
                    relation("ratings", "Rating", true),
                    relation("director", "Director", false),
                ],
            ),
            ("Director", vec![scalar("name", ScalarType::String)]),
            (
                "Rating",
                vec![
                    scalar("stars", ScalarType::I64),
                    relation("user", "User", false),
                    relation("movie", "Movie", false),
                ],
            ),
        ]);
        let cat = load_or_initialize(&storage, &union, false).unwrap();

        // No duplicate type / field / rel IDs.
        let type_ids: std::collections::HashSet<u64> = cat.type_ids.values().copied().collect();
        assert_eq!(type_ids.len(), cat.type_ids.len());
        let field_ids: std::collections::HashSet<u64> = cat.field_ids.values().copied().collect();
        assert_eq!(field_ids.len(), cat.field_ids.len());
        let rel_ids: std::collections::HashSet<u64> = cat.rel_ids.values().copied().collect();
        assert_eq!(rel_ids.len(), cat.rel_ids.len());
    }

    // -----------------------------------------------------------------
    // 9.7 Format / decoder safety
    // -----------------------------------------------------------------

    #[test]
    fn unknown_format_version_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("X", vec![])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        // Overwrite c:F: with a future version. We don't go through
        // backfill again — we open afresh.
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_format(),
                Bytes::copy_from_slice(&99u64.to_be_bytes()),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();

        let err = load_or_initialize(&storage, &schema, false).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::UnsupportedFormat { got: 99, .. })
        ));
    }

    // -----------------------------------------------------------------
    // 9.8 Digest fast-path (instrumented via a smoke test rather than
    //     an explicit counter — the fast path is taken iff reconcile
    //     would otherwise allocate; the absence of new entries proves
    //     no work happened)
    // -----------------------------------------------------------------

    #[test]
    fn reopen_unchanged_schema_does_not_change_catalog_state() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = bench_like_schema();
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let snap1 = storage.read_snapshot();
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let snap2 = storage.read_snapshot();
        // No additional commit happened on the no-op reopen.
        assert_eq!(snap1, snap2);
    }

    #[test]
    fn digest_mismatch_triggers_reconcile_and_self_corrects_digest() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("X", vec![scalar("a", ScalarType::I64)])]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        // Corrupt the digest in place.
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_digest(),
                Bytes::copy_from_slice(&[0u8; 32]),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();

        // Reopen with the same schema — reconcile runs (digest
        // mismatched) but no drops or kind changes; it succeeds and
        // rewrites the digest to the correct value.
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        let snap = storage.read_snapshot();
        let stored = storage
            .get_at(snap, &KeyBuilder::catalog_digest())
            .unwrap()
            .unwrap();
        assert_eq!(stored.as_ref(), &compute_schema_digest(&schema)[..]);

        // And the next open after that takes the fast path.
        let snap_before = storage.read_snapshot();
        let _ = load_or_initialize(&storage, &schema, false).unwrap();
        let snap_after = storage.read_snapshot();
        assert_eq!(snap_before, snap_after);
    }

    // -----------------------------------------------------------------
    // 9.9 Crash safety / partial catalog recovery
    // -----------------------------------------------------------------

    #[test]
    fn partial_catalog_recovers_via_clear_and_rebackfill() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        // Inject a torn write: only c:F: present, no c:I:.
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_format(),
                encode_format_version(CATALOG_FORMAT_V1),
            )
            .unwrap();
        // Also write a stray catalog type row so the recovery has to
        // actually delete something.
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_type("Garbage"),
                encode_id_entry(&IdEntry::fresh(99, kind_byte::UNSET, 0)),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();

        let schema = schema_with(vec![("User", vec![scalar("name", ScalarType::String)])]);
        let cat = load_or_initialize(&storage, &schema, false).unwrap();
        assert!(cat.type_ids.contains_key("User"));
        assert!(!cat.type_ids.contains_key("Garbage"));
    }

    // -----------------------------------------------------------------
    // Adversary fixes — counter self-heal + duplicate ID detection
    // -----------------------------------------------------------------

    #[test]
    fn load_existing_self_heals_stale_counter() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![(
            "User",
            vec![
                scalar("name", ScalarType::String),
                scalar("age", ScalarType::I64),
            ],
        )]);
        let _ = load_or_initialize(&storage, &schema, false).unwrap();

        // Roll back c:N:E by hand — simulates a torn tail that
        // dropped the counter-bump record. Self-heal must clamp to
        // max(allocated)+1 on next open.
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_next_field(),
                encode_counter(0),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();

        let cat = load_or_initialize(&storage, &schema, false).unwrap();
        let max_field = *cat.field_ids.values().max().unwrap();
        assert!(
            cat.next_field > max_field,
            "next_field {} must exceed max {}",
            cat.next_field,
            max_field
        );
    }

    #[test]
    fn duplicate_type_id_detected_on_load() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        // Write a self-consistent catalog header but two type rows
        // sharing the same ID — only reachable via external tampering.
        let now = 0;
        let mut txn = storage.begin_txn();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_format(),
                encode_format_version(CATALOG_FORMAT_V1),
            )
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_initialized(),
                encode_marker(now),
            )
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_digest(),
                Bytes::copy_from_slice(&[0u8; 32]),
            )
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_next_type(),
                encode_counter(5),
            )
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_next_field(),
                encode_counter(1),
            )
            .unwrap();
        storage
            .put(&mut txn, &KeyBuilder::catalog_next_rel(), encode_counter(1))
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_type("Alpha"),
                encode_id_entry(&IdEntry::backfilled(1, kind_byte::UNSET, now)),
            )
            .unwrap();
        storage
            .put(
                &mut txn,
                &KeyBuilder::catalog_type("Beta"),
                encode_id_entry(&IdEntry::backfilled(1, kind_byte::UNSET, now)),
            )
            .unwrap();
        storage.commit(&mut txn).unwrap();

        let schema = schema_with(vec![("Alpha", vec![]), ("Beta", vec![])]);
        let err = load_or_initialize(&storage, &schema, false).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::DuplicateId {
                kind: "type",
                id: 1,
                ..
            })
        ));
    }

    // -----------------------------------------------------------------
    // Identifier safety
    // -----------------------------------------------------------------

    #[test]
    fn reserved_byte_in_type_name_refused() {
        let dir = TempDir::new().unwrap();
        let storage = open_lsm(&dir);
        let schema = schema_with(vec![("Bad:Name", vec![])]);
        let err = load_or_initialize(&storage, &schema, false).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(CatalogError::ReservedByteInIdentifier { .. })
        ));
    }
}
