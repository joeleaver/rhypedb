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
pub(crate) const CATALOG_FORMAT_CURRENT: u64 = CATALOG_FORMAT_V2;

/// Per-row record format version. Future phases may bump for rows that
/// carry semantic-meaning TLVs phase 1 can't interpret.
const RECORD_FORMAT_V1: u8 = 0x01;

// Value-kind discriminants (byte 1 of every catalog value).
const KIND_ID_ENTRY: u8 = 0x01;
const KIND_COUNTER: u8 = 0x02;
const KIND_MARKER: u8 = 0x03;

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

// Bounded retry budget on WriteConflict during catalog commits.
// Concurrent opens fight over the digest write (always present in
// reconcile commits) and the catalog header writes (always present in
// backfill commits). 8 retries is far more than realistic contention.
const COMMIT_RETRY_BUDGET: usize = 8;

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
    let stale = storage.scan_prefix_at(snap, &KeyBuilder::catalog_prefix_all())?;
    if !stale.is_empty() {
        let deletes: Vec<Bytes> = stale.into_iter().map(|(k, _)| k).collect();
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
    for (qual, &cat_kind) in &cat.field_kinds {
        let (t, f) = split_qualified(qual);
        let Some(td) = schema.types.get(t) else {
            continue;
        };
        let Some(fd) = td.fields.iter().find(|x| x.name == f) else {
            continue;
        };
        let want_kind = schema_kind_byte(&fd.field_type);
        if cat_kind != want_kind {
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
        KIND_COUNTER | KIND_MARKER => {
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
// HELPERS
// =====================================================================

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
    Ok(())
}

fn now_unix_millis() -> u64 {
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
