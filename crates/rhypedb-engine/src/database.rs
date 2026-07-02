use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use bytes::Bytes;

use rhypedb_schema::{FieldType, OnDeletePolicy, ScalarType, Schema};
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_subscribe::{ChangeEvent, ChangeKind, SubscriptionHub};

use crate::error::{EngineError, EngineResult};
use crate::object::{
    FieldMap, Object, Value, deserialize_fields, deserialize_fields_projected, extract_field,
    serialize_fields,
};

/// One bounded, OOM-safe step of an ascending object scan
/// ([`Database::scan_chunk`]). Mirrors storage's [`rhypedb_storage::lsm::ChunkScan`]
/// but yields decoded [`Object`]s instead of raw key/value pairs.
#[derive(Debug, Clone)]
pub struct ObjectChunk {
    /// Live objects in this chunk, ascending by object id. May be shorter than
    /// the requested `max_distinct` (or empty) when the chunk straddles a run
    /// of tombstoned ids — do NOT treat a short vec as end-of-range.
    pub objects: Vec<Object>,
    /// Highest object id visited this chunk, tombstones included. `None` iff
    /// the range is exhausted (no key at/after the cursor). Pass it back as the
    /// next `cursor` to resume.
    pub next_cursor: Option<u64>,
    /// `true` when more objects may exist strictly past `next_cursor`. `false`
    /// PROVES the scan is complete. Only `!more` (or `next_cursor == None`) is
    /// a sound stop condition.
    pub more: bool,
}

/// Objects (and vector entries) materialized per chunk during a logical
/// export. Bounds peak memory regardless of how large a type is.
const LOGICAL_EXPORT_CHUNK_SIZE: usize = 1024;

/// Serialize one NDJSON line (a `serde_json::Value`) and write it followed by a
/// newline. `Value` serialization is infallible here — every map key is a
/// string and every numeric field is encoded as a string or a plain integer
/// (no NaN floats) — so only the write itself can fail.
fn write_export_line(writer: &mut dyn std::io::Write, val: &serde_json::Value) -> EngineResult<()> {
    let mut buf = serde_json::to_vec(val).expect("export line Value serialization is infallible");
    buf.push(b'\n');
    writer.write_all(&buf).map_err(export_io_err)
}

/// The byte string immediately after `k` in unsigned-byte order — `k` with a
/// trailing `0x00`. Advances a raw chunked scan strictly past an inclusive
/// high-water key whose tail is not a decodable object id (e.g. the `v:` vector
/// keyspace, where the trailing 8 bytes are the field id).
fn successor_key(k: &[u8]) -> Bytes {
    let mut buf = bytes::BytesMut::with_capacity(k.len() + 1);
    buf.extend_from_slice(k);
    buf.extend_from_slice(&[0u8]);
    buf.freeze()
}

fn export_io_err(e: std::io::Error) -> EngineError {
    EngineError::Storage(rhypedb_storage::Error::Io(e))
}

/// One row of the precomputed reverse-relation index used by cascade delete.
/// `is_many` distinguishes forward 1:1 incoming relations (where the source's
/// rev_edge cover embeds the target's data) from many-relations (where the
/// target isn't embedded). The cover-refresh sweeper consults it to decide
/// whether an incoming source S needs Phase 1 re-run after T is updated.
#[derive(Debug, Clone)]
struct IncomingRelation {
    source_type_id: u64,
    source_type: String,
    source_field: String,
    rel_id: u64,
    is_many: bool,
    policy: OnDeletePolicy,
}

/// Arena for tombstone keys produced by a single `delete` cascade walk.
/// All key bytes go into ONE owned `Vec<u8>` and each push records the
/// `(start, end)` byte range. When the cascade walk is done, the caller
/// freezes the buffer into a `Bytes` and slices each range into a
/// refcount-only view to hand to `storage.delete_batch`.
///
/// Replaces the historical `Vec<Bytes>` accumulator where every key was
/// a separate small `BytesMut::with_capacity` allocation — at K=100
/// cascading ratings that's ~500 mallocs per User-delete swapped for a
/// single buffer allocation.
struct TombstoneArena {
    buf: Vec<u8>,
    ranges: Vec<(u32, u32)>,
}

impl TombstoneArena {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(16 * 1024),
            ranges: Vec::with_capacity(512),
        }
    }

    fn push_object(&mut self, type_id: u64, object_id: u64) {
        let r = KeyBuilder::object_into(&mut self.buf, type_id, object_id);
        self.ranges.push(r);
    }

    fn push_edge(&mut self, source_id: u64, rel_id: u64, target_id: u64) {
        let r = KeyBuilder::edge_into(&mut self.buf, source_id, rel_id, target_id);
        self.ranges.push(r);
    }

    fn push_reverse_edge(&mut self, target_id: u64, rel_id: u64, source_id: u64) {
        let r = KeyBuilder::reverse_edge_into(&mut self.buf, target_id, rel_id, source_id);
        self.ranges.push(r);
    }

    fn push_object_version(&mut self, type_id: u64, object_id: u64) {
        let r = KeyBuilder::object_version_into(&mut self.buf, type_id, object_id);
        self.ranges.push(r);
    }

    fn push_unique_index(&mut self, type_id: u64, field_id: u64, value_bytes: &[u8]) {
        let r = KeyBuilder::unique_index_into(&mut self.buf, type_id, field_id, value_bytes);
        self.ranges.push(r);
    }

    fn push_field_index(
        &mut self,
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) {
        let r = KeyBuilder::field_index_into(
            &mut self.buf,
            type_id,
            field_id,
            encoded_value,
            object_id,
        );
        self.ranges.push(r);
    }

    fn push_field_index_var(
        &mut self,
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8],
        object_id: u64,
    ) {
        let r = KeyBuilder::field_index_var_into(
            &mut self.buf,
            type_id,
            field_id,
            encoded_value,
            object_id,
        );
        self.ranges.push(r);
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Reserve capacity for `n` additional ranges. Used at points where
    /// we know the inbound scan size to avoid Vec doublings.
    fn reserve(&mut self, n: usize) {
        self.ranges.reserve(n);
    }

    /// Roll back to a previous range count — used when a deny violation
    /// aborts mid-cascade and we need the parent's earlier-staged keys
    /// to remain but our own to disappear. Truncating the byte buffer to
    /// match keeps the arena tight even on the error path.
    fn truncate(&mut self, ranges_len: usize) {
        let new_buf_len = if ranges_len == 0 {
            0
        } else {
            self.ranges[ranges_len - 1].1 as usize
        };
        self.ranges.truncate(ranges_len);
        self.buf.truncate(new_buf_len);
    }

    /// Consume the arena and produce one `Bytes` per recorded range.
    /// `Bytes::from(buf)` is O(1) (owns the Vec) and `.slice(range)` is
    /// refcount-only — no per-key heap allocation.
    fn into_keys(self) -> Vec<Bytes> {
        let buf_bytes = Bytes::from(self.buf);
        self.ranges
            .iter()
            .map(|&(s, e)| buf_bytes.slice(s as usize..e as usize))
            .collect()
    }
}

/// Pre-resolved per-type metadata the cascade hot path consults to avoid
/// `format!("Type.field")` + `HashMap::get` per relation per cascaded
/// object. For a User-delete at K=100 ratings, the bench used to do ~10
/// HashMap lookups + 4-6 `format!` allocations × 100 cascaded Ratings =
/// ~1000 hot-path allocations. With this struct each cascade call does
/// one `HashMap::get(&type_id)` and iterates a contiguous `Vec<ForwardRelMeta>`.
#[derive(Debug, Clone)]
struct CascadeMeta {
    type_name: String,
    has_unique: bool,
    has_indexed: bool,
    /// True when the type declares at least one scalar field — i.e. its
    /// `o:` blob carries identifying data worth surfacing on a Delete
    /// change event. Used to decide whether `delete_inner` should read the
    /// object payload for a cascade-deleted row: scalar-bearing types get a
    /// read (so the Delete event carries the same scalar fields create/update
    /// emit), while pure edge-only join rows skip it and keep the zero-read
    /// cascade fast path. `has_scalar` is a superset of `has_unique` /
    /// `has_indexed` (those are always scalar fields).
    has_scalar: bool,
    /// Forward (non-inverse) relations on this type, with rel_id pre-
    /// resolved. Used by the outbound-tombstone walk.
    forward_relations: Vec<ForwardRelMeta>,
}

#[derive(Debug, Clone)]
struct ForwardRelMeta {
    field_name: String,
    rel_id: u64,
    is_many: bool,
}

/// The rhypedb database engine.
///
/// Manages typed objects and relationships backed by the LSM storage engine,
/// enforcing the schema's type constraints and referential integrity.
pub struct Database {
    schema: Schema,
    storage: Arc<LsmTree>,
    type_ids: HashMap<String, u64>,
    rel_ids: HashMap<String, u64>,
    field_ids: HashMap<String, u64>,
    /// Retired numeric IDs, partitioned by row kind. Populated from the
    /// catalog at open(). Hot paths check the negative case (id is NOT
    /// in the tombstoned set) so the common case is one `HashSet::contains`
    /// against an empty/tiny set.
    tombstoned_type_ids: std::collections::HashSet<u64>,
    tombstoned_field_ids: std::collections::HashSet<u64>,
    tombstoned_rel_ids: std::collections::HashSet<u64>,
    /// Retired names, partitioned the same way. Used by resolve helpers
    /// to surface a typed `*Retired` error when a caller names a retired
    /// entity (e.g. `db.get("RetiredType", id)`). Type and relation
    /// name sets are reserved for the follow-on rename / migration
    /// cards; field quals are the hot path consulted by every read.
    #[allow(dead_code)]
    tombstoned_type_names: std::collections::HashSet<String>,
    tombstoned_field_quals: std::collections::HashSet<String>,
    #[allow(dead_code)]
    tombstoned_rel_quals: std::collections::HashSet<String>,
    /// Per-type set of retired field NAMES (NOT qualified). The FieldMap
    /// strip path uses `get(type_name)` once per object; for the common
    /// case (no retired fields on this type) the lookup misses and the
    /// strip is essentially free.
    retired_field_names_by_type: HashMap<String, std::collections::HashSet<String>>,
    /// Retirement timestamps (Unix millis) keyed by numeric id. Used to
    /// build the `*Retired` error variants without consulting the
    /// catalog at every error site.
    retired_at_ms_by_type_id: HashMap<u64, u64>,
    retired_at_ms_by_field_id: HashMap<u64, u64>,
    retired_at_ms_by_rel_id: HashMap<u64, u64>,
    /// `Arc`-shared so the `_consuming` migrate variants hand the SAME
    /// `AtomicU64` to the rebuilt handle. Without sharing, the OLD
    /// handle's `fetch_add` after carry-snapshot and the NEW handle's
    /// `fetch_add` from the snapshot can return the same id —
    /// collision + silent overwrite on the shared `Arc<LsmTree>`.
    next_object_id: Arc<AtomicU64>,
    /// Subscription hub for change events. Wrapped in `Arc` so the
    /// `_consuming` migrate variants can hand the same hub to the new
    /// `Database` handle without dropping live subscribers' channels.
    subscriptions: Arc<SubscriptionHub>,
    /// target_type_id → list of relations that point at it. Built once at
    /// open() so cascade delete doesn't iterate the whole schema per
    /// recursive call. Keyed by type_id to skip a String-keyed lookup on
    /// every cascade call.
    incoming_relations: HashMap<u64, Vec<IncomingRelation>>,
    /// type_id → per-type cascade metadata used by `delete_inner` to
    /// avoid repeated schema walks + `format!()` calls per relation per
    /// cascaded object.
    cascade_meta_by_id: HashMap<u64, CascadeMeta>,
    /// type_id → type name. Reverse of `type_ids`, used by cascade to
    /// resolve names for subscription events without consulting the schema.
    type_name_by_id: HashMap<u64, String>,
    /// type_name → list of @indexed scalar fields, with their pre-resolved
    /// (field_name, field_id). Cached so the create/update/delete write
    /// paths don't re-traverse the schema per object.
    indexed_fields: HashMap<String, Vec<IndexedField>>,
    /// Per-object monotonic generation counter, bumped on every successful
    /// `update`. Lives in-memory for cheap reads (cover-write stamps the
    /// target's current generation into `<name>__cover_v`; executor fusion
    /// compares against the live generation to detect stale covers). Backed
    /// by `g:<type_id>:<object_id>` keys for restart durability — the map is
    /// repopulated by scanning that prefix in `open()`.
    /// `Arc`-shared so the `_consuming` migrate variants hand the SAME
    /// RwLock to the rebuilt handle. Without sharing, the OLD handle's
    /// `update()` between carry-snapshot and the new handle taking over
    /// would persist to LSM but not to NEW's in-memory map, causing
    /// cover_v generation divergence on subsequent NEW updates.
    version_counters: Arc<RwLock<HashMap<(u64, u64), u64>>>,
    /// Cheap lockless "is `version_counters` non-empty?" check. Cascade
    /// delete uses it to skip the per-cascaded-object `version_counters.read()
    /// .contains_key()` when no object has ever been updated. For the bench
    /// (insert + delete, no updates) this saves 100 RwLock acquires per
    /// User-delete at K=100. Updated alongside every map mutation.
    version_counter_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Outbound channel to the cover-refresh worker. Each successful
    /// `update()` pushes the bumped target's `(type_id, object_id)` here;
    /// the worker scans `r:<target>:*` for incoming 1:1 forward sources
    /// and re-runs Phase 1 for each (rewriting their outbound rev_edges
    /// with the target's fresh cover).
    ///
    /// Wrapped in `Mutex<Option<...>>` so `Drop` can `take()` the sender,
    /// closing the channel and prompting the worker to exit cleanly.
    cover_refresh_tx: parking_lot::Mutex<Option<std::sync::mpsc::Sender<(u64, u64)>>>,
    /// Join handle for the cover-refresh worker thread. Taken on drop.
    cover_refresh_handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// A `Weak` to this very `Database`, stashed right after `Arc::new` so a
    /// `&self` method (`create_field_type_migration`) can hand a detached
    /// migration driver thread a `Weak<Database>` WITHOUT changing the public
    /// `&self` signature. The driver upgrades it transiently (only for the
    /// cutover stage) so it never extends the database's lifetime — when external
    /// `Arc`s drop, the upgrade returns `None` and the driver leaves the plan
    /// resumable for the next open. Empty until set in `rebuild_with_arc_storage`.
    self_weak: parking_lot::Mutex<std::sync::Weak<Database>>,
    /// Per-plan registry of in-flight migration drivers (shadow-field card 3/5).
    /// Keyed by `plan_id` (unique, monotonic — never reused). The single gate
    /// enforcing "at most one driver per plan" (async create OR inline
    /// resume/auto-resume register here before driving; a second registration
    /// fails `MigrationAlreadyRunning`). `Drop` drains this, signals every
    /// `control` to PAUSE, and joins each `Some(handle)` (mirrors the
    /// `cover_refresh_*` teardown, with the same self-join-skip guard).
    migration_drivers: parking_lot::Mutex<HashMap<u64, MigrationDriver>>,
    /// Card 5: per-plan live event stream (ChunkCompleted / PartitionDone /
    /// CutoverStarted / CutoverDone / RollbackStarted / StatusChanged / Failed).
    /// `Arc` so the detached driver thread can publish independent of this
    /// handle's lifetime (mirrors `subscriptions`). Carried across a
    /// `_consuming` rebuild so live subscribers keep their channels.
    migration_events: Arc<MigrationEventHub>,
    /// Write barrier excluding user-facing mutations during the catalog
    /// migration verbs. Read-locked by `create` / `create_batch` /
    /// `update` / `delete` / `link` / `unlink`; write-locked by
    /// `rename_type` / `rename_field` / `change_field_type` /
    /// `run_migrations` (and their `_consuming` siblings).
    ///
    /// Without this lock, a concurrent writer can commit a new object
    /// with the OLD field-name layout while the migration is mid-scan;
    /// the migration's MVCC write_set doesn't intersect the new object's
    /// key, so the conflict goes undetected and the object lands in the
    /// post-rename catalog era with stale-named FieldMap entries.
    /// Wrapped in `Arc` so the `_consuming` migrate variants carry the
    /// same lock instance through to the rebuilt handle (both old and
    /// new Databases serialize through it).
    migration_lock: Arc<parking_lot::RwLock<()>>,
    /// The `OpenOptions` this handle was constructed with. Stored so the
    /// `_consuming` migrate variants can re-spawn workers with matching
    /// settings on the rebuilt handle.
    opts: OpenOptions,
    /// `true` once one of the `_consuming` migrate verbs has consumed
    /// this handle and returned a fresh `Arc<Database>`. Set under
    /// `Release` so any caller racing a method call observes the
    /// poison via `Acquire`. Public read/write entrypoints check this
    /// and surface `DatabaseMigratedAway` instead of returning stale-
    /// cache results.
    migrated: std::sync::atomic::AtomicBool,
    /// `(type_id, field_name) -> field_id` lookup used by the zone-map
    /// extractor (write path) and `filter_scan` predicate builder (read
    /// path). Wrapped in `Arc<ArcSwap<...>>` so:
    /// * the extractor closure can capture it BEFORE the catalog loads
    ///   (chicken-and-egg: `LsmConfig::zone_extractor` must be set before
    ///   `LsmTree::open`, but the catalog only loads after); and
    /// * a future migrate verb can rebuild and atomically swap the table
    ///   without touching `LsmConfig` or rebuilding the closure (PR B).
    zone_field_id_lookup: Arc<arc_swap::ArcSwap<ZoneFieldIdLookup>>,
    /// Per-`Database` named converter registry for chunked field-type
    /// migrations (shadow-field card 1). `name -> (version, converter)`.
    /// Per-`Database` (NOT a process-global) so two DBs in one process —
    /// the multi-tenant-in-VM deployment — can't collide on a converter
    /// name and bind the wrong body at resume. `Arc` so it is shared with
    /// the `_consuming` rebuilt handle. A migration pins `(name, version)`
    /// in its plan and re-resolves here at create and at resume; a missing
    /// or version-changed name parks the plan `AwaitingConverter`.
    converters: Arc<parking_lot::RwLock<HashMap<String, (u32, crate::catalog::RegisteredConverter)>>>,
    /// Per-`(type_id, field_name)` double-write hooks for in-flight chunked
    /// field-type migrations (shadow-field card 2). While a field is migrating,
    /// every write to it ALSO stamps a converted `<field>__shadow` sibling so
    /// writes proceed during the migration (card 2d — no type-wide quiesce). A
    /// hook with an unresolved converter is REJECTING: a write to its field
    /// fails closed (`MigrationFieldConverterUnresolved`). Nested
    /// `type_id -> {field_name -> hook}` so the hot-path probe for a
    /// non-migrating type is a single Copy-`u64` miss. Lock-free read cache
    /// (ArcSwap), mutated ONLY under `migration_lock.write()` via
    /// `arm_field_hook`/`disarm_field_hook`, rebuilt from the `c:P:` plans on
    /// open/create. A non-zero `migrating_field_count` is the fast-path gate.
    migrating_fields: arc_swap::ArcSwap<
        std::collections::HashMap<u64, std::collections::HashMap<String, Arc<MigratingFieldHook>>>,
    >,
    /// Fast-path gate: total live hook count. Producers load this (`Relaxed`)
    /// and skip ALL hook work — no lock, no map probe, no `String` — when it's
    /// `0` (the common case, no migration active). Kept in sync with
    /// `migrating_fields` under `migration_lock.write()`.
    migrating_field_count: std::sync::atomic::AtomicUsize,
}

/// Completion signal for a migration driver (shadow-field card 3/5). `finished`
/// flips true when the driver stops (any disposition); `wait_take_error` blocks
/// on the condvar until then and TAKES the terminal error (worker convert error
/// / cutover refusal) the ASYNC create driver recorded — the inline resume path
/// propagates its error through the normal return, so it leaves the error
/// `None`. The `finished` atomic doubles as a lock-free check for the
/// registration gate (reap a finished leftover without taking the inner mutex).
/// Shared (`Arc`) between the registry entry and the driver, so the driver can
/// signal completion even after `Database::drop` has drained the registry.
struct MigrationSignal {
    finished: std::sync::atomic::AtomicBool,
    error: parking_lot::Mutex<Option<EngineError>>,
    cv: parking_lot::Condvar,
}

impl MigrationSignal {
    fn new() -> Self {
        Self {
            finished: std::sync::atomic::AtomicBool::new(false),
            error: parking_lot::Mutex::new(None),
            cv: parking_lot::Condvar::new(),
        }
    }

    /// Driver: record the terminal error (if any) + wake every waiter. Setting
    /// `finished` UNDER the error mutex (the same one `wait_take_error` parks on)
    /// is what rules out a lost wakeup.
    fn mark_done(&self, error: Option<EngineError>) {
        let mut g = self.error.lock();
        *g = error;
        self.finished
            .store(true, std::sync::atomic::Ordering::Release);
        self.cv.notify_all();
    }

    /// Lock-free "the driver has stopped" check for the registration gate.
    fn is_finished(&self) -> bool {
        self.finished.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Waiter: block until the driver stops, then TAKE the terminal error (so a
    /// second waiter sees `None` — the durable plan status is the multi-waiter
    /// source of truth).
    fn wait_take_error(&self) -> Option<EngineError> {
        let mut g = self.error.lock();
        while !self.finished.load(std::sync::atomic::Ordering::Acquire) {
            self.cv.wait(&mut g);
        }
        g.take()
    }
}

/// A registered in-flight migration driver (shadow-field card 3/5). One per
/// `plan_id` (the registration is the double-driver gate). The ASYNC create
/// driver does NOT remove its own entry on exit — it just `mark_done`s the
/// signal and returns; the entry (with its still-joinable handle) is reaped by
/// `wait_for_migration` (join + remove), by the registration gate (a finished
/// leftover), or by `Database::drop` (drain + join). Keeping the handle until a
/// JOIN is what makes a `wait; drop; reopen` sequence race-free.
struct MigrationDriver {
    /// Run/Pause/Cancel byte (`catalog::migration_control`), polled by every
    /// partition worker BETWEEN chunks. `pause_migration`/`cancel_migration`
    /// store into it; `Database::drop` stores PAUSE.
    control: Arc<std::sync::atomic::AtomicU8>,
    /// Completion signal — `wait_for_migration` blocks on it.
    signal: Arc<MigrationSignal>,
    /// `Some` for the async create driver (its detached thread, joined by
    /// `wait_for_migration` / `Database::drop`); `None` for an inline resume /
    /// auto-resume drive (which runs on the calling thread — no thread to join,
    /// and its `InlineDriveGuard` removes the entry synchronously on exit).
    handle: Option<std::thread::JoinHandle<()>>,
}

/// A per-`(type, field)` double-write hook for an in-flight chunked field-type
/// migration (shadow-field card 2). A writer touching the field stamps a
/// converted `<field>__shadow` sibling via this hook so the write carries the
/// migration forward instead of being rejected.
pub(crate) struct MigratingFieldHook {
    pub field_name: String,
    /// `None` = the plan's pinned converter is not registered in this
    /// `Database` yet (e.g. a fresh open before `register_converter`). A write
    /// to the field then FAILS CLOSED (`MigrationFieldConverterUnresolved`,
    /// card 2b) rather than land source-only with no shadow — which cutover
    /// would later refuse. Resolves to `Some` once the converter is registered
    /// and the hook is re-armed.
    pub converter: Option<crate::catalog::RegisteredConverter>,
    /// On-disk kind the converter must produce (validates converter output).
    pub target_kind: u8,
    /// Pinned converter version; stamped onto each `<field>__shadow_cv` sibling
    /// so cutover can refuse a shadow written by a stale converter (card 2c).
    pub converter_version: u32,
    pub plan_id: u64,
}

/// True if `key` is a card-2 migration shadow sibling (`<field>__shadow` or
/// `<field>__shadow_cv`) — siblings written into the object blob during a
/// migration that must never leak to callers. (The `__`-infix namespace is
/// reserved, like the `__cover` covers.)
pub(crate) fn is_shadow_sibling_key(key: &str) -> bool {
    key.ends_with("__shadow") || key.ends_with("__shadow_cv")
}

/// One @indexed scalar field on a type, with everything the write path needs
/// to emit/withdraw its `idx:` entry without re-resolving from the schema.
/// `field_id` is the same stable per-`{Type.field}` u64 the unique index uses.
/// `kind` decides which encoder + key layout the field uses:
///
///   * `Integer` / `Bool` / `Float` → fixed 8-byte sortable encoding,
///     written into the legacy `KeyBuilder::field_index` layout.
///   * `String` / `Bytes` → variable-length escape-encoded + NUL-NUL
///     terminator, written into `KeyBuilder::field_index_var`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedKind {
    Integer,
    Bool,
    Float,
    String,
    Bytes,
}

#[derive(Debug, Clone)]
struct IndexedField {
    name: String,
    field_id: u64,
    kind: IndexedKind,
}

/// Bundle of in-memory state the `_consuming` migrate variants carry
/// from the old `Database` into the rebuilt one. Skips re-scanning the
/// LSM for state that is invariant across the migration (object IDs
/// haven't changed; the generation counters haven't changed; the
/// subscription hub keeps live channel receivers).
pub(crate) struct CarryState {
    pub subscriptions: Arc<SubscriptionHub>,
    /// SAME hub the OLD handle holds, so a `_consuming` rebuild keeps live
    /// migration-event subscribers' channels (card 5).
    pub migration_events: Arc<MigrationEventHub>,
    /// SAME `Arc<AtomicU64>` the OLD handle holds. Any `fetch_add` on
    /// either handle's pointer increments the same counter.
    pub next_object_id: Arc<AtomicU64>,
    /// SAME `Arc<RwLock<...>>` the OLD handle holds — see
    /// `Database::version_counters` for the rationale.
    pub version_counters: Arc<RwLock<HashMap<(u64, u64), u64>>>,
    pub version_counter_count: Arc<std::sync::atomic::AtomicUsize>,
    pub migration_lock: Arc<parking_lot::RwLock<()>>,
    /// SAME registry Arc the OLD handle holds, so converters registered
    /// before a `_consuming` verb stay resolvable on the new handle.
    pub converters: Arc<parking_lot::RwLock<HashMap<String, (u32, crate::catalog::RegisteredConverter)>>>,
}

/// Operator-tunable knobs that don't depend on schema. Pass to
/// `Database::open_with_options` to override per-deployment behaviour.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    /// `true` (default): every commit fsyncs the WAL — durable against
    /// power loss. `false`: skip the fsync syscall — the kernel still has
    /// the bytes, so a clean process crash is recoverable, but a hard
    /// power loss can drop the last N writes. Matches Postgres's
    /// `fsync=off + synchronous_commit=off` mode for benchmarking, and
    /// is useful for bulk imports that accept the risk.
    pub sync_on_commit: bool,
    /// `true` (default): spawn the cover-refresh worker thread. Every
    /// `update()` that bumps an object's generation queues the target;
    /// the worker scans for incoming 1:1 forward sources and rewrites
    /// their outbound rev_edges so embedded `<name>__cover` blobs stay
    /// current. Disable for benchmarks that don't want the background
    /// CPU or for tests that need deterministic stale-cover state.
    pub background_cover_refresh: bool,
    /// Reserved for the tombstone-migration phase (card 2/5). In phase
    /// 1 this flag is REJECTED at the schema-shrink gate regardless of
    /// its value: opening with a shrinking schema returns
    /// `CatalogError::SchemaShrink` (flag `false`) or
    /// `CatalogError::SchemaShrinkNotYetSupported` (flag `true`). The
    /// field exists so callers can wire the intent through their
    /// config today and the gate flips when phase 2 ships, without a
    /// breaking API change. Defaults to `false`.
    pub allow_schema_shrink: bool,
    /// Per-block compression for newly written SST files (flush + compaction).
    /// Defaults to `None` (v5 zero-copy layout). `Lz4` (v6) gives ~3.8x smaller
    /// files but a benchmark showed it costs ~3.7x on a 1M-row multi-hop graph
    /// traversal (each scattered cover-blob read decompresses a whole block for
    /// one entry), so it is opt-in: enable it where disk size / cold-cache
    /// density matters and reads are scan-heavy or rare. Both formats are always
    /// readable; compaction migrates files to whichever is set.
    pub block_compression: rhypedb_storage::SstCompression,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            sync_on_commit: true,
            background_cover_refresh: true,
            allow_schema_shrink: false,
            block_compression: rhypedb_storage::SstCompression::None,
        }
    }
}

/// Spec for a chunked field-type migration (shadow-field card 1/5). The
/// converter must already be registered via [`Database::register_converter`]
/// under `(converter_name, converter_version)` — the pair is pinned in the
/// persisted plan so a restart re-resolves the same body (a missing or
/// version-skewed converter parks the plan `AwaitingConverter` on resume
/// rather than running the wrong logic).
#[derive(Debug, Clone)]
pub struct MigrationPlanSpec {
    pub type_name: String,
    pub field_name: String,
    pub target_field_type: rhypedb_schema::FieldType,
    pub converter_name: String,
    pub converter_version: u32,
    /// Objects converted per chunk commit; `0` uses the engine default
    /// (`DEFAULT_MIGRATION_CHUNK_SIZE`). Smaller = more frequent cursor
    /// commits (lower re-do on crash) at more fsyncs.
    pub chunk_size: u64,
    /// Card 4: per-row converter-failure policy (default `Stop`).
    pub error_policy: crate::catalog::ErrorPolicy,
    /// Card 4: when `true`, run the converter over every row to estimate
    /// `objects_converted`/`error_count` but write NOTHING and never cut over —
    /// a preflight. Default `false`.
    pub dry_run: bool,
    /// Card 4: cap on quarantined/errored rows; exceeding it auto-stops the
    /// migration (`Failed`). `0` → `DEFAULT_QUARANTINE_CAP` (100K).
    pub quarantine_cap: u64,
    /// Card 5: override the number of parallel backfill workers. `None` →
    /// auto (one per CPU, capped at 8). Clamped into `1..=MAX_PARALLEL_DEGREE`.
    pub parallel_degree: Option<u8>,
}

impl Default for MigrationPlanSpec {
    /// Defaults for the non-identifying knobs (card 4): `Stop` policy, not a
    /// dry-run, engine-default chunk size + quarantine cap. The identifying
    /// fields (type/field/target/converter) are placeholders the caller MUST
    /// override — provided only so `..Default::default()` can supply the card-4
    /// knobs without every call site spelling them out.
    fn default() -> Self {
        Self {
            type_name: String::new(),
            field_name: String::new(),
            target_field_type: rhypedb_schema::FieldType::Scalar(rhypedb_schema::ScalarType::I64),
            converter_name: String::new(),
            converter_version: 0,
            chunk_size: 0,
            error_policy: crate::catalog::ErrorPolicy::Stop,
            dry_run: false,
            quarantine_cap: 0,
            parallel_degree: None,
        }
    }
}

/// Operator-facing snapshot of a persisted migration plan
/// ([`Database::list_migrations`]). Kind bytes and forward-compat TLVs are
/// deliberately not exposed.
/// What [`Database::backup_to`] captured. Plain data (the engine stays
/// serde-free); the caller serializes this into the on-disk `MANIFEST.json`.
#[derive(Debug, Clone)]
pub struct BackupManifest {
    /// SST file names hard-linked/copied into `<dst>/sst/`.
    pub sst_names: Vec<String>,
    /// Highest committed version across the captured SSTs.
    pub max_version: u64,
    /// Byte size of the captured (post-flush, header-only) `wal.log`.
    pub wal_bytes: u64,
    /// `hnsw_*.bin` vector-index snapshots copied alongside (may be empty).
    pub hnsw_files: Vec<String>,
    /// `(plan_id, converter_name)` of migrations in flight at backup time.
    pub in_flight_migrations: Vec<(u64, String)>,
    /// Wall-clock backup time (ms since epoch); 0 if the clock is unavailable.
    pub created_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct MigrationSummary {
    pub plan_id: u64,
    pub type_name: String,
    pub field_name: String,
    /// The field type this plan migrates the field TO. `None` only for a
    /// non-scalar target (which `change_field_type` rejects, so in practice
    /// always `Some`). Lets the server build the post-cutover schema for an
    /// in-place hot-reload without a `catalog → Schema` reconstructor.
    pub target_field_type: Option<FieldType>,
    pub status: crate::catalog::MigrationStatus,
    /// Highest object id whose conversion is durably committed.
    pub cursor: u64,
    /// Observability counter — completion is proven by an exhaustion scan,
    /// not by this value (a torn re-scan can double-count).
    pub objects_converted: u64,
    pub chunk_size: u64,
    pub converter_name: String,
    pub converter_version: u32,
    pub created_at_ms: u64,
    /// Card 4: rows whose converter failed during backfill (a historical count;
    /// the live unresolved-quarantine count is `list_quarantined().len()`).
    pub error_count: u64,
    /// Card 4: per-row failure policy.
    pub error_policy: crate::catalog::ErrorPolicy,
    /// Card 4: a dry-run preflight (wrote nothing, never cut over).
    pub dry_run: bool,
}

/// Operator-facing view of one quarantined row (card 4,
/// [`Database::list_quarantined`]).
#[derive(Debug, Clone)]
pub struct QuarantineEntry {
    pub object_id: u64,
    pub error_msg: String,
    pub errored_at_ms: u64,
    pub attempted_converter_name: String,
}

/// One partition's live progress within a parallel migration (card 5,
/// [`Database::query_migration_progress`]). A legacy single-worker plan is
/// reported as one synthetic partition over `[1, id_upper_bound)`.
#[derive(Debug, Clone)]
pub struct PartitionProgress {
    pub idx: u8,
    /// Inclusive low / exclusive high object-id bound the worker owns.
    pub lo: u64,
    pub hi: u64,
    /// Highest object id whose shadow is durably committed in this partition.
    pub cursor: u64,
    pub objects_converted: u64,
    pub errors: u64,
    pub done: bool,
}

/// Live progress snapshot of one migration plan (card 5,
/// [`Database::query_migration_progress`]). Aggregates the per-partition
/// `c:S:` cursors and derives an ETA from the durable `created_at_ms` and the
/// converted count.
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    pub plan_id: u64,
    pub type_name: String,
    pub field_name: String,
    pub status: crate::catalog::MigrationStatus,
    pub phase: crate::catalog::MigrationPhase,
    pub dry_run: bool,
    pub parallel_degree: Option<u8>,
    /// `id_upper_bound - 1` — the count of object ids in `[1, U)` at plan
    /// creation. The progress denominator. Overcounts if rows in the range
    /// were deleted (converted may never reach this), which is acceptable for a
    /// "wait or come back" ETA.
    pub total_objects: u64,
    pub objects_converted: u64,
    pub errors: u64,
    pub created_at_ms: u64,
    pub now_ms: u64,
    /// Conversion rate over the whole run (`converted / elapsed`). `None`
    /// unless the plan is `Running` with at least one converted row.
    pub objects_per_sec: Option<f64>,
    /// Projected completion wall-clock (unix ms): `now + remaining/rate`.
    /// `None` unless the plan is `Running` with a positive rate.
    pub eta_unix_ms: Option<u64>,
    pub partitions: Vec<PartitionProgress>,
}

/// Filter for [`Database::list_migrations_filtered`] (card 5). A `None` field
/// matches everything; set fields are ANDed.
#[derive(Debug, Clone, Default)]
pub struct MigrationFilter {
    pub status: Option<crate::catalog::MigrationStatus>,
    pub type_name: Option<String>,
}

/// Handle returned by [`Database::start_field_type_migration_async`] (card 5).
/// The operator-facing surface is keyed by `plan_id`; `created_at_ms` is
/// captured once at creation (immutable thereafter) for client-side ETA math.
#[derive(Debug, Clone, Copy)]
pub struct MigrationHandle {
    pub plan_id: u64,
    pub created_at_ms: u64,
}

/// One live event on a migration's progress stream (card 5,
/// [`Database::subscribe_migration_events`]). Events are best-effort and
/// non-replayed: a subscriber attached after an event was published never sees
/// it (poll [`Database::query_migration_progress`] for the current state).
/// Every variant carries `plan_id` so the hub can filter per subscriber.
#[derive(Debug, Clone)]
pub enum MigrationEvent {
    /// A partition worker committed one chunk of shadow backfill.
    ChunkCompleted {
        plan_id: u64,
        partition_idx: u8,
        cursor: u64,
        objects_converted: u64,
    },
    /// A partition exhausted its `[lo, hi)` range (durable `done`).
    PartitionDone { plan_id: u64, partition_idx: u8 },
    /// The cutover pass began (all partitions backfilled).
    CutoverStarted { plan_id: u64 },
    /// The cutover pass completed; the catalog kind is flipped.
    CutoverDone { plan_id: u64 },
    /// A terminal cancel's rollback pass began (card 5 — stripping shadows).
    RollbackStarted { plan_id: u64 },
    /// The plan reached a new durable status (Completed / Cancelled /
    /// DryRunCompleted / Failed).
    StatusChanged {
        plan_id: u64,
        status: crate::catalog::MigrationStatus,
    },
    /// The driver surfaced a terminal error (the plan parked `Failed`).
    Failed { plan_id: u64, message: String },
}

impl MigrationEvent {
    pub fn plan_id(&self) -> u64 {
        match self {
            MigrationEvent::ChunkCompleted { plan_id, .. }
            | MigrationEvent::PartitionDone { plan_id, .. }
            | MigrationEvent::CutoverStarted { plan_id }
            | MigrationEvent::CutoverDone { plan_id }
            | MigrationEvent::RollbackStarted { plan_id }
            | MigrationEvent::StatusChanged { plan_id, .. }
            | MigrationEvent::Failed { plan_id, .. } => *plan_id,
        }
    }

    /// True for an event that ENDS a plan's stream — a settled `StatusChanged`
    /// or a `Failed`. (`CutoverDone` is always followed by a `Completed`
    /// `StatusChanged`, so it is not itself terminal.) The hub drops a plan's
    /// subscribers after a terminal event.
    pub fn is_terminal(&self) -> bool {
        match self {
            MigrationEvent::Failed { .. } => true,
            MigrationEvent::StatusChanged { status, .. } => matches!(
                status,
                crate::catalog::MigrationStatus::Completed
                    | crate::catalog::MigrationStatus::Cancelled
                    | crate::catalog::MigrationStatus::DryRunCompleted
            ),
            _ => false,
        }
    }
}

struct MigrationEventSub {
    id: u64,
    plan_id: u64,
    sender: std::sync::mpsc::Sender<MigrationEvent>,
}

/// Per-plan migration event fan-out (card 5). Mirrors `SubscriptionHub`
/// (`rhypedb-subscribe`): `std::sync::mpsc` channels (runtime-agnostic — the
/// engine has no tokio dependency; the server bridges to async for SSE).
/// `publish` is non-blocking (unbounded channel) and never holds a storage txn
/// or `migration_lock` across the send, so a slow/absent consumer cannot
/// backpressure a migration worker. Dead subscribers are reaped lazily on the
/// first failed send (the receiver was dropped).
pub(crate) struct MigrationEventHub {
    subs: parking_lot::RwLock<Vec<MigrationEventSub>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl MigrationEventHub {
    pub(crate) fn new() -> Self {
        Self {
            subs: parking_lot::RwLock::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub(crate) fn subscribe(
        &self,
        plan_id: u64,
    ) -> (u64, std::sync::mpsc::Receiver<MigrationEvent>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.subs.write().push(MigrationEventSub {
            id,
            plan_id,
            sender,
        });
        (id, receiver)
    }

    pub(crate) fn publish(&self, event: MigrationEvent) {
        let plan = event.plan_id();
        // A terminal event ends the plan's stream — after delivering it, drop ALL
        // of that plan's subscribers (no further events will ever come, so an
        // un-dropped receiver would otherwise linger in `subs` forever; lazy
        // send-failure reaping only fires on a FUTURE publish that never happens).
        let terminal = event.is_terminal();
        let mut dead = Vec::new();
        {
            let subs = self.subs.read();
            for s in subs.iter() {
                if s.plan_id == plan && s.sender.send(event.clone()).is_err() {
                    dead.push(s.id);
                }
            }
        }
        if terminal {
            self.subs.write().retain(|s| s.plan_id != plan);
        } else if !dead.is_empty() {
            self.subs.write().retain(|s| !dead.contains(&s.id));
        }
    }
}

impl Database {
    /// Open a database with the given schema and data directory.
    ///
    /// Returns an `Arc<Database>` because the engine owns a background
    /// cover-refresh worker thread that holds a `Weak<Database>` reference
    /// — the Arc wrapper is required for the worker's lifetime to track
    /// the database's. All callers can treat the `Arc` like a `Database`
    /// directly thanks to deref.
    pub fn open(schema: Schema, data_dir: impl AsRef<Path>) -> EngineResult<Arc<Self>> {
        Self::open_with_options(schema, data_dir, OpenOptions::default())
    }

    /// Open with explicit options. The engine still owns the
    /// `zone_extractor` wiring (it needs schema-aware decoding); options
    /// carries the durability + throughput knobs that operators / bench
    /// harnesses tune from the outside.
    pub fn open_with_options(
        schema: Schema,
        data_dir: impl AsRef<Path>,
        options: OpenOptions,
    ) -> EngineResult<Arc<Self>> {
        let mut config = LsmConfig::new(data_dir);
        // Wire a zone-field extractor that pulls integer field values out of
        // object entries' FieldMap blobs at SST flush/compaction time. Lets
        // `filter_scan` skip blocks whose min/max bounds rule out the
        // predicate without per-entry decode + compare.
        //
        // Two-phase init: the extractor needs a `(type_id, field_name) ->
        // field_id` table that only exists after `catalog::load_or_initialize`
        // runs, but `LsmConfig::zone_extractor` must be installed BEFORE
        // `LsmTree::open`. Capture an empty `ArcSwap<HashMap>` here and swap
        // in the populated table after the catalog loads. No flush can fire
        // before population because nothing has been written yet.
        let zone_field_id_lookup: Arc<arc_swap::ArcSwap<ZoneFieldIdLookup>> = Arc::new(
            arc_swap::ArcSwap::from_pointee(ZoneFieldIdLookup::new()),
        );
        let extractor_lookup = Arc::clone(&zone_field_id_lookup);
        config.zone_extractor = Some(Arc::new(move |internal_key, value| {
            let snapshot = extractor_lookup.load();
            do_extract_zone_fields(&snapshot, internal_key, value)
        }));
        config.sync_on_commit = options.sync_on_commit;
        config.block_compression = options.block_compression;
        let storage = LsmTree::open(config)?;
        Self::rebuild_with_arc_storage(
            storage,
            schema,
            options,
            zone_field_id_lookup,
            None,
        )
    }

    /// Shared post-`LsmTree::open` body. Used by:
    /// * `open_with_options` (`carry = None`) — fresh start, scans
    ///   `o:*` for `next_object_id`, scans `g:*` for `version_counters`,
    ///   creates a fresh `SubscriptionHub`.
    /// * The `_consuming` migrate variants (`carry = Some(...)`) —
    ///   reuse the existing `Arc<LsmTree>`, carry forward in-memory
    ///   counters + subscribers from the old handle, rebuild derived
    ///   schema-shaped caches (`type_ids`, `cascade_meta_by_id`,
    ///   `indexed_fields`, …) under the post-migration schema.
    ///
    /// The `zone_field_id_lookup` `Arc` is shared with the extractor
    /// closure baked into `LsmConfig` — when migrate calls this fn it
    /// passes the same `Arc` so the table swap takes effect for the
    /// already-installed closure.
    pub(crate) fn rebuild_with_arc_storage(
        storage: Arc<LsmTree>,
        schema: Schema,
        options: OpenOptions,
        zone_field_id_lookup: Arc<arc_swap::ArcSwap<ZoneFieldIdLookup>>,
        carry: Option<CarryState>,
    ) -> EngineResult<Arc<Self>> {
        // Pull stable numeric IDs from the persisted schema catalog
        // (see `catalog.rs`). For a legacy / fresh database the
        // catalog backfill produces the same IDs the prior
        // alphabetical algorithm did, so existing on-disk data is
        // untouched; for an extended schema the catalog allocates new
        // IDs from persisted counters without renumbering anything.
        let cat =
            crate::catalog::load_or_initialize(&storage, &schema, options.allow_schema_shrink)?;
        let type_ids = cat.type_ids;
        let rel_ids = cat.rel_ids;
        let field_ids = cat.field_ids;
        let tombstoned_type_ids = cat.tombstoned_type_ids;
        let tombstoned_field_ids = cat.tombstoned_field_ids;
        let tombstoned_rel_ids = cat.tombstoned_rel_ids;
        let tombstoned_type_names = cat.tombstoned_type_names;
        let tombstoned_field_quals = cat.tombstoned_field_quals;
        let tombstoned_rel_quals = cat.tombstoned_rel_quals;

        // Retirement timestamps keyed by numeric id, populated from
        // catalog entries' `retired_at_ms` TLV. Used to build the
        // `*Retired` error variants without re-loading the catalog.
        let mut retired_at_ms_by_type_id: HashMap<u64, u64> = HashMap::new();
        let mut retired_at_ms_by_field_id: HashMap<u64, u64> = HashMap::new();
        let mut retired_at_ms_by_rel_id: HashMap<u64, u64> = HashMap::new();
        for entry in cat.type_entries.values() {
            if let Some(ms) = entry.retired_at_ms {
                retired_at_ms_by_type_id.insert(entry.id, ms);
            }
        }
        for entry in cat.field_entries.values() {
            if let Some(ms) = entry.retired_at_ms {
                retired_at_ms_by_field_id.insert(entry.id, ms);
            }
        }
        for entry in cat.rel_entries.values() {
            if let Some(ms) = entry.retired_at_ms {
                retired_at_ms_by_rel_id.insert(entry.id, ms);
            }
        }

        // Per-type set of retired field NAMES (not qualified). Used by
        // the FieldMap strip path — for a type with no retired fields,
        // the HashMap lookup misses and the strip is a single hash.
        let mut retired_field_names_by_type: HashMap<String, std::collections::HashSet<String>> =
            HashMap::new();
        for qual in &tombstoned_field_quals {
            if let Some(dot) = qual.find('.') {
                let (t, f) = (&qual[..dot], &qual[dot + 1..]);
                retired_field_names_by_type
                    .entry(t.to_string())
                    .or_default()
                    .insert(f.to_string());
            }
        }

        // Recover the max object ID by scanning existing objects —
        // skipped when carry provides the shared `Arc<AtomicU64>`
        // (migrate doesn't change existing object IDs, and the SAME
        // atomic counter is carried verbatim through to the new
        // handle).
        // One `o:*` scan recovers two things: the high-water object id (for
        // `next_object_id`) and the born bit for every live object. Each
        // existing `o:` key seeds `version_counters` to generation 1 — this is
        // what lets a create skip a persisted `g:` key entirely: an object's
        // existence on disk IS its born bit, reconstructed here at open. The
        // `g:*` override scan below only bumps the few objects that were
        // UPDATED (generation >= 2). The carry path skips this — it clones the
        // in-memory counters from the old handle verbatim.
        let (max_object_id, mut version_counters): (u64, HashMap<(u64, u64), u64>) =
            if carry.is_some() {
                // Unused: carry path constructs `next_object_id` from
                // `Arc::clone(&c.next_object_id)` and clones `version_counters`.
                (0u64, HashMap::new())
            } else {
                let mut max_object_id = 0u64;
                let mut seed: HashMap<(u64, u64), u64> = HashMap::new();
                let txn = storage.begin_txn();
                for &type_id in type_ids.values() {
                    let prefix = KeyBuilder::object_prefix(type_id);
                    if let Ok(entries) = storage.scan_prefix(&txn, &prefix) {
                        for (key, _) in &entries {
                            // Object key: o:<type_id>:<object_id> — last 8 bytes are the object ID.
                            if key.len() >= 8 {
                                let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
                                let object_id = u64::from_be_bytes(id_bytes);
                                max_object_id = max_object_id.max(object_id);
                                seed.insert((type_id, object_id), 1);
                            }
                        }
                    }
                }
                drop(txn);
                (max_object_id, seed)
            };

        // Precompute the reverse-relation index used by cascade delete.
        // Without this, every delete_inner call walks the entire schema
        // looking for inbound edges. We deliberately skip @inverse fields:
        // those don't allocate their own forward edges (reads route through
        // the underlying forward field's rev_edges), so a cascade-time
        // `scan_prefix(r:<id>:<inverse_rel_id>:)` is guaranteed to return
        // empty and pay only bloom/sparse-index overhead. For the bench
        // schema that's two wasted scans per cascaded Rating (User.ratings
        // and Movie.ratings), or ~200 wasted scans per User-delete at
        // K=100 cascading ratings — a measurable cascade tax.
        let mut incoming_relations: HashMap<u64, Vec<IncomingRelation>> = HashMap::new();
        for (source_type, type_def) in &schema.types {
            let source_type_id = type_ids[source_type];
            for field in &type_def.fields {
                if let FieldType::Relation(rel) = &field.field_type {
                    if field.inverse().is_some() {
                        continue;
                    }
                    let rel_key = format!("{source_type}.{}", field.name);
                    let rel_id = rel_ids[&rel_key];
                    let policy = field.on_delete().cloned().unwrap_or(if rel.is_many {
                        OnDeletePolicy::Remove
                    } else {
                        OnDeletePolicy::Deny
                    });
                    let target_type_id = type_ids[&rel.target_type];
                    incoming_relations
                        .entry(target_type_id)
                        .or_default()
                        .push(IncomingRelation {
                            source_type_id,
                            source_type: source_type.clone(),
                            source_field: field.name.clone(),
                            rel_id,
                            is_many: rel.is_many,
                            policy,
                        });
                }
            }
        }

        let types_with_unique: std::collections::HashSet<String> = schema
            .types
            .iter()
            .filter_map(|(name, td)| {
                if td.fields.iter().any(|f| f.is_unique()) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        // type_id → name reverse map (for subscription publish + error
        // messages without re-walking type_ids each time).
        let mut type_name_by_id: HashMap<u64, String> = HashMap::new();
        for (name, id) in &type_ids {
            type_name_by_id.insert(*id, name.clone());
        }

        // Per-type cascade metadata. Lets `delete_inner` iterate a Vec of
        // pre-resolved (field_name, rel_id, is_many) tuples instead of
        // walking the schema + format!()-ing field keys + HashMap-looking-up
        // rel_ids on every cascade call. For K=100 cascading Ratings this
        // cuts ~6 allocations + ~10 HashMap lookups per Rating cascade →
        // ~1,600 fewer ops per User delete.
        let mut cascade_meta_by_id: HashMap<u64, CascadeMeta> = HashMap::new();
        for (name, type_def) in &schema.types {
            let type_id = type_ids[name];
            let mut forward_relations = Vec::new();
            for field in &type_def.fields {
                if let FieldType::Relation(rel) = &field.field_type {
                    if field.inverse().is_some() {
                        continue;
                    }
                    let rel_key = format!("{name}.{}", field.name);
                    let rel_id = rel_ids[&rel_key];
                    forward_relations.push(ForwardRelMeta {
                        field_name: field.name.clone(),
                        rel_id,
                        is_many: rel.is_many,
                    });
                }
            }
            cascade_meta_by_id.insert(
                type_id,
                CascadeMeta {
                    type_name: name.clone(),
                    has_unique: types_with_unique.contains(name),
                    has_indexed: type_def.fields.iter().any(|f| f.is_indexed()),
                    has_scalar: type_def
                        .fields
                        .iter()
                        .any(|f| matches!(f.field_type, FieldType::Scalar(_))),
                    forward_relations,
                },
            );
        }

        // Precompute the @indexed scalar fields per type. Each entry resolves
        // to the same `{Type.field}` → u64 ID we use for unique indexes so
        // write-path code can build idx: keys without re-traversing the schema.
        let mut indexed_fields: HashMap<String, Vec<IndexedField>> = HashMap::new();
        for (type_name, type_def) in &schema.types {
            let mut list = Vec::new();
            for field in &type_def.fields {
                if field.is_indexed() {
                    let key = format!("{type_name}.{}", field.name);
                    let field_id = field_ids[&key];
                    let kind = match &field.field_type {
                        FieldType::Scalar(ScalarType::String) => IndexedKind::String,
                        FieldType::Scalar(ScalarType::Bytes) => IndexedKind::Bytes,
                        FieldType::Scalar(ScalarType::Bool) => IndexedKind::Bool,
                        FieldType::Scalar(ScalarType::F32 | ScalarType::F64) => IndexedKind::Float,
                        // DateTime indexes through the Integer path — it shares
                        // the I64 MSB-flip ordered encoding in
                        // `encode_int_for_zone` / `build_field_index_key`.
                        FieldType::Scalar(ScalarType::DateTime) => IndexedKind::Integer,
                        _ => IndexedKind::Integer,
                    };
                    list.push(IndexedField {
                        name: field.name.clone(),
                        field_id,
                        kind,
                    });
                }
            }
            if !list.is_empty() {
                indexed_fields.insert(type_name.clone(), list);
            }
        }

        // Override the born-bit seed with persisted generations for objects
        // that have been UPDATED at least once. A `g:` key is written only on
        // update (generation >= 2), so this scan touches just those objects;
        // never-updated objects keep their seeded generation of 1. The carry
        // path skips this entirely — the SAME `Arc<RwLock<HashMap>>` is carried
        // verbatim to the new handle, so old and new observe one counter map.
        if carry.is_none() {
            let txn2 = storage.begin_txn();
            if let Ok(entries) = storage.scan_prefix(&txn2, &KeyBuilder::object_version_prefix()) {
                for (key, value) in entries {
                    if key.len() < 1 + 1 + 8 + 1 + 8 || value.len() != 8 {
                        continue;
                    }
                    let type_id_bytes: [u8; 8] = key[2..10].try_into().unwrap();
                    let obj_id_bytes: [u8; 8] = key[11..19].try_into().unwrap();
                    let v_bytes: [u8; 8] = value[..].try_into().unwrap();
                    version_counters.insert(
                        (
                            u64::from_be_bytes(type_id_bytes),
                            u64::from_be_bytes(obj_id_bytes),
                        ),
                        u64::from_be_bytes(v_bytes),
                    );
                }
            }
            drop(txn2);
        }

        // Populate the zone-field-id lookup now that the catalog has
        // assigned IDs. The extractor closure already holds a clone of
        // this `Arc<ArcSwap>` and will pick up the new table on its
        // next load.
        //
        // **Subtlety:** the previous version of this comment claimed
        // "no flush has fired yet" — that's wrong. `LsmTree::open`
        // replays the WAL into the memtable, and then
        // `catalog::load_or_initialize` does `put_batch` of catalog
        // rows which can push the memtable over the flush threshold.
        // That flush calls the extractor while the lookup is still
        // empty, producing an SST with `num_blocks=0` in its zone map.
        // Correctness holds (per-entry filter fallback still runs) but
        // block pruning is dead on that SST until natural compaction
        // rewrites it.
        //
        // We force a follow-up flush below to recover pruning on any
        // such SST: anything left in the memtable now gets rewritten
        // under the populated extractor. Anything that already flushed
        // during the catalog load with an empty zone map will be
        // healed by the next compaction pass.
        zone_field_id_lookup
            .store(Arc::new(build_zone_field_id_lookup(&schema, &type_ids, &field_ids)));
        // Best-effort warmup flush. If the LSM is configured against
        // flushes (rare) or returns an io error, surface it — opening
        // would otherwise hide a real durability issue.
        storage.flush()?;

        let db = Arc::new(Self {
            schema,
            storage,
            type_ids,
            rel_ids,
            field_ids,
            tombstoned_type_ids,
            tombstoned_field_ids,
            tombstoned_rel_ids,
            tombstoned_type_names,
            tombstoned_field_quals,
            tombstoned_rel_quals,
            retired_field_names_by_type,
            retired_at_ms_by_type_id,
            retired_at_ms_by_field_id,
            retired_at_ms_by_rel_id,
            next_object_id: match &carry {
                Some(c) => Arc::clone(&c.next_object_id),
                None => Arc::new(AtomicU64::new(max_object_id + 1)),
            },
            subscriptions: match &carry {
                Some(c) => Arc::clone(&c.subscriptions),
                None => Arc::new(SubscriptionHub::new()),
            },
            migration_events: match &carry {
                Some(c) => Arc::clone(&c.migration_events),
                None => Arc::new(MigrationEventHub::new()),
            },
            incoming_relations,
            cascade_meta_by_id,
            type_name_by_id,
            indexed_fields,
            version_counter_count: match &carry {
                Some(c) => Arc::clone(&c.version_counter_count),
                None => Arc::new(std::sync::atomic::AtomicUsize::new(version_counters.len())),
            },
            version_counters: match &carry {
                Some(c) => Arc::clone(&c.version_counters),
                None => Arc::new(RwLock::new(version_counters)),
            },
            cover_refresh_tx: parking_lot::Mutex::new(None),
            cover_refresh_handle: parking_lot::Mutex::new(None),
            self_weak: parking_lot::Mutex::new(std::sync::Weak::new()),
            migration_drivers: parking_lot::Mutex::new(HashMap::new()),
            migration_lock: match &carry {
                Some(c) => Arc::clone(&c.migration_lock),
                None => Arc::new(parking_lot::RwLock::new(())),
            },
            zone_field_id_lookup,
            // Card 2: double-write hooks — start empty, rebuilt from `c:P:` by
            // `auto_resume_migrations` on the open path (below) / armed by
            // create_field_type_migration under migration_lock.write(). (Not
            // carried via CarryState: a `_consuming` rebuild is refused while a
            // plan is unsettled, so carry is always migration-free.)
            migrating_fields: arc_swap::ArcSwap::from_pointee(std::collections::HashMap::new()),
            migrating_field_count: std::sync::atomic::AtomicUsize::new(0),
            converters: match &carry {
                Some(c) => Arc::clone(&c.converters),
                None => Arc::new(parking_lot::RwLock::new(HashMap::new())),
            },
            opts: options.clone(),
            migrated: std::sync::atomic::AtomicBool::new(false),
        });

        // Stash a `Weak` to ourselves so a `&self` verb can hand a detached
        // migration driver a `Weak<Database>` (card 3/5) without an `Arc<Self>`
        // signature. Set before any worker spawns / auto-resume runs.
        *db.self_weak.lock() = Arc::downgrade(&db);

        // Spawn the cover-refresh worker now that `db` lives inside an Arc
        // we can downgrade. The worker holds a `Weak<Database>` so it
        // doesn't extend the database's lifetime — once external Arcs are
        // dropped, our Drop impl closes the channel and joins. Skipped when
        // the caller opts out via `OpenOptions::background_cover_refresh`.
        if options.background_cover_refresh {
            let (tx, rx) = std::sync::mpsc::channel::<(u64, u64)>();
            let weak = Arc::downgrade(&db);
            let handle = std::thread::Builder::new()
                .name("rhypedb-cover-refresh".into())
                .spawn(move || cover_refresh_worker(rx, weak))
                .map_err(|e| EngineError::Storage(rhypedb_storage::Error::Io(e)))?;
            *db.cover_refresh_tx.lock() = Some(tx);
            *db.cover_refresh_handle.lock() = Some(handle);
        }

        // Auto-resume in-flight chunked field-type migrations (shadow-field
        // card 1) — ONLY on the genuine open path (`carry.is_none()`). The
        // `_consuming` rebuild path (`carry.is_some()`) is reached while the
        // caller holds `migration_lock.write()`, and auto-resume re-takes it
        // (via finalize) — gating on `carry` avoids that self-deadlock and is
        // also correct: a `_consuming` verb is a schema op, not a fresh open.
        if carry.is_none() {
            db.auto_resume_migrations()?;
        }

        Ok(db)
    }

    /// Current generation of the object `(type_name, object_id)`. Returns 0
    /// when the object has never been updated (no entry in the counter map).
    /// Public so the query executor can compare against `<name>__cover_v`
    /// embedded in rev_edge values without pulling more `Database` internals.
    pub fn object_version(&self, type_name: &str, object_id: u64) -> u64 {
        let Some(&type_id) = self.type_ids.get(type_name) else {
            return 0;
        };
        self.version_counters
            .read()
            .get(&(type_id, object_id))
            .copied()
            .unwrap_or(0)
    }

    fn bump_version(&self, type_id: u64, object_id: u64) -> u64 {
        let mut map = self.version_counters.write();
        let inserted = !map.contains_key(&(type_id, object_id));
        let entry = map.entry((type_id, object_id)).or_insert(0);
        *entry += 1;
        let v = *entry;
        if inserted {
            self.version_counter_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        v
    }

    fn rollback_version(&self, type_id: u64, object_id: u64) {
        let mut map = self.version_counters.write();
        if let Some(v) = map.get_mut(&(type_id, object_id)) {
            if *v > 0 {
                *v -= 1;
            }
            if *v == 0 {
                map.remove(&(type_id, object_id));
                self.version_counter_count
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    fn forget_version(&self, type_id: u64, object_id: u64) {
        if self
            .version_counters
            .write()
            .remove(&(type_id, object_id))
            .is_some()
        {
            self.version_counter_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------
    // Tombstone resolution + strip helpers. Wrap every read/write
    // entrypoint that previously did `type_ids.get(name).ok_or(TypeNotFound)`
    // — for a retired entity, return the typed `*Retired` variant
    // INSTEAD of `*NotFound`. The retired entity is still in the
    // catalog (and so in `type_ids` / `field_ids` / `rel_ids`); the
    // distinction matters because the operator's mental model is
    // different: "you removed it from the schema" vs. "you typoed".
    // -----------------------------------------------------------------

    /// Resolve a type name to its numeric id. Returns `TypeRetired` if
    /// the type was tombstoned, `TypeNotFound` if it never existed.
    ///
    /// Also serves as the poison-check chokepoint: every public read
    /// or write method goes through `resolve_type_id` at least once,
    /// so checking `migrated` here gates the whole user-facing API
    /// without an inline branch at each entrypoint.
    fn resolve_type_id(&self, type_name: &str) -> EngineResult<u64> {
        self.check_not_migrated()?;
        let type_id = self
            .type_ids
            .get(type_name)
            .copied()
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        if self.tombstoned_type_ids.contains(&type_id) {
            return Err(EngineError::TypeRetired {
                name: type_name.into(),
                id: type_id,
                retired_at_unix_ms: self
                    .retired_at_ms_by_type_id
                    .get(&type_id)
                    .copied()
                    .unwrap_or(0),
            });
        }
        Ok(type_id)
    }

    /// Resolve a `Type.field` pair to `(type_id, field_id)`. Surfaces
    /// `TypeRetired` first (if the parent type itself was retired),
    /// then `FieldRetired`, then `FieldNotFound`. Reserved for the
    /// follow-on cards (rename / field-type change) — currently call
    /// sites use `field_retired_error` + `schema.get_field` directly
    /// because they also need the `FieldDef` for type-aware encoding.
    #[allow(dead_code)]
    fn resolve_field_id(&self, type_name: &str, field: &str) -> EngineResult<(u64, u64)> {
        let type_id = self.resolve_type_id(type_name)?;
        let qual = format!("{}.{}", type_name, field);
        let field_id = self
            .field_ids
            .get(&qual)
            .copied()
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: field.into(),
            })?;
        if self.tombstoned_field_ids.contains(&field_id) {
            return Err(EngineError::FieldRetired {
                type_name: type_name.into(),
                field: field.into(),
                field_id,
                retired_at_unix_ms: self
                    .retired_at_ms_by_field_id
                    .get(&field_id)
                    .copied()
                    .unwrap_or(0),
            });
        }
        Ok((type_id, field_id))
    }

    /// Resolve a `Type.relation` pair to `(type_id, rel_id)`. Surfaces
    /// `TypeRetired` first, then `RelationRetired`.
    #[allow(dead_code)]
    fn resolve_relation_id(&self, type_name: &str, relation: &str) -> EngineResult<(u64, u64)> {
        let type_id = self.resolve_type_id(type_name)?;
        let qual = format!("{}.{}", type_name, relation);
        let rel_id =
            self.rel_ids
                .get(&qual)
                .copied()
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: relation.into(),
                })?;
        if self.tombstoned_rel_ids.contains(&rel_id) {
            return Err(EngineError::RelationRetired {
                type_name: type_name.into(),
                relation: relation.into(),
                relation_id: rel_id,
                retired_at_unix_ms: self
                    .retired_at_ms_by_rel_id
                    .get(&rel_id)
                    .copied()
                    .unwrap_or(0),
            });
        }
        Ok((type_id, rel_id))
    }

    /// Strip every retired field from a decoded `FieldMap` BEFORE it
    /// leaves the engine boundary. Called from every `deserialize_fields`
    /// site so on-disk bytes are preserved verbatim but callers never
    /// see a field name their current schema doesn't know. For the
    /// common case (a type with no retired fields) the HashMap lookup
    /// misses and the strip is a single hash + branch.
    fn strip_tombstoned_fields(&self, type_name: &str, fields: &mut FieldMap) {
        if let Some(retired) = self.retired_field_names_by_type.get(type_name)
            && !retired.is_empty()
        {
            fields.retain(|name, _| !retired.contains(name));
        }
        // Card 2: never expose the migration shadow siblings (`<field>__shadow`,
        // `<field>__shadow_cv`) to callers. UNCONDITIONAL (not gated on an
        // active migration): a reader on a pre-cutover MVCC snapshot can
        // deserialize a shadow-bearing blob and then race a `disarm_field_hook`
        // that flips the count to 0 before the gate is checked — gating here
        // would leak the siblings on that read. `__`-suffixed siblings are a
        // reserved namespace that must never reach a caller regardless, and the
        // retain is cheap next to the deserialize this always follows. (The
        // lazy/raw wire path stays count-gated — its zero-copy fast path can't
        // afford an always-on deserialize, and it only ships verbatim bytes a
        // migration could have written.)
        fields.retain(|name, _| !is_shadow_sibling_key(name));
    }

    // -----------------------------------------------------------------
    // Schema migration verbs (card 3/5).
    //
    // `rename_type` shipped in card 3/5 phase 1. `rename_field` lands
    // in phase 2: SST v5 zone maps are keyed by stable catalog
    // field_id (rename-invariant), and FieldMap rewrites at rename
    // time keep `o:` object data and `r:` cover blobs consistent.
    // -----------------------------------------------------------------

    /// Rename a live type from `old` to `new`. The numeric type_id is
    /// preserved; every child catalog row (`c:E:<type>\\x00*` and
    /// `c:R:<type>\\x00*`) is cascaded to the new type name in the
    /// same atomic commit. Object data (`o:<type_id>:*`), index data
    /// (`i:<type_id>:*`), edge data (`r:` / `e:`), and cover blobs
    /// are NOT touched — they're keyed by `type_id` (stable) and by
    /// field-name strings (unchanged by a type rename).
    ///
    /// On-disk effect: deletes `c:T:<old>` and `c:E:<old>\\x00*` and
    /// `c:R:<old>\\x00*`; puts `c:T:<new>` and the renamed child rows;
    /// deletes `c:D:` (digest) so the next open recomputes it; bumps
    /// `c:F:` to v3 the first time.
    ///
    /// **Caller MUST drop this `Database` handle after a successful
    /// rename and re-open with the post-rename schema.** Several
    /// derived caches (`incoming_relations`, `cascade_meta_by_id`,
    /// `indexed_fields`) are built at `open()` and not rebuilt in
    /// place; this `Database`'s in-memory `type_ids` and `schema` are
    /// also stale after the call. The follow-on card may add an
    /// `Arc<Self>`-consuming variant that returns a fresh `Arc<Database>`
    /// in one call.
    pub fn rename_type(&self, old: &str, new: &str) -> EngineResult<crate::catalog::MigrationReport> {
        // Reject calls on a handle that's been migrated away (a held
        // Arc::clone after a `_consuming` verb produced a new handle).
        // The catalog state on disk would be valid but the in-memory
        // schema / type_ids / field_ids of this stale handle don't
        // match it — running another verb against it would corrupt.
        self.check_not_migrated()?;
        // Write-lock excludes concurrent `create` / `update` / `link` /
        // `unlink` / `delete` so they can't commit OLD-shape data while
        // the catalog migration runs and atomically rewrites the
        // affected on-disk state. Closes the during-verb race window
        // the adversarial review surfaced.
        let _migration_guard = self.migration_lock.write();
        self.rename_type_inner(old, new)
    }

    /// Inner body of `rename_type` without locking. Callers MUST hold
    /// `migration_lock.write()` (and have already run the
    /// `check_not_migrated` guard if appropriate). Used by the
    /// `_consuming` variant so the lock + poison + rebuild can be one
    /// atomic critical section.
    fn rename_type_inner(&self, old: &str, new: &str) -> EngineResult<crate::catalog::MigrationReport> {
        let verbs = [crate::catalog::RenameVerb::Type {
            old: old.into(),
            new: new.into(),
        }];
        crate::catalog::apply_migration_with_cover(&self.storage, &self.schema, &verbs, Some(self))
    }

    /// Rename a live field from `type_name.old` to `type_name.new`. The
    /// numeric `field_id` is preserved; every existing object's
    /// serialized FieldMap (`o:<type_id>:<obj_id>` value) is rewritten
    /// to use the new name in the same atomic LSM batch as the catalog
    /// row update.
    ///
    /// What stays consistent by construction (all in the same atomic
    /// LSM batch):
    /// * Object reads of the new name — every existing object has been
    ///   rewritten.
    /// * Reverse-edge cover blobs (`r:<target>:<rel>:<source>` values)
    ///   whose source is an object of this type — the embedded
    ///   source-side FieldMap is rewritten to use the new name. Without
    ///   this, the executor's covering fast-path would return Objects
    ///   with the OLD field name (the `cover_v` stamp matches because
    ///   rename doesn't bump it, so the existing staleness fall-through
    ///   never fires).
    /// * Unique index entries (`u:<type>:<field_id>:…`) — keyed by
    ///   `field_id` with the object_id as value, untouched by the rename.
    /// * Secondary index entries (`i:<type>:<field_id>:…`) — the KEYS are
    ///   keyed by `field_id` (untouched), but each entry's covering value is
    ///   a full object FieldMap embedding field names, so it is rewritten with
    ///   the new name in the same atomic batch (any sibling `@indexed` field's
    ///   cover too, since it also embeds the renamed field).
    /// * SST zone-map pruning — v5 zone columns are also keyed by
    ///   `field_id`, so existing block bounds keep pruning correctly
    ///   under the new name.
    ///
    /// **Caller MUST drop this `Database` handle after a successful
    /// rename and re-open with the post-rename schema** — the in-memory
    /// `schema`, `field_ids`, derived caches, and the zone-field-id
    /// lookup are all stale until reopened. PR B's `Arc<Self>`-consuming
    /// `migrate` variant lifts this requirement.
    pub fn rename_field(
        &self,
        type_name: &str,
        old: &str,
        new: &str,
    ) -> EngineResult<crate::catalog::MigrationReport> {
        self.check_not_migrated()?;
        let _migration_guard = self.migration_lock.write();
        self.rename_field_inner(type_name, old, new)
    }

    /// Lock-less inner body of `rename_field`. See `rename_type_inner`.
    fn rename_field_inner(
        &self,
        type_name: &str,
        old: &str,
        new: &str,
    ) -> EngineResult<crate::catalog::MigrationReport> {
        let verbs = [crate::catalog::RenameVerb::Field {
            type_name: type_name.into(),
            old: old.into(),
            new: new.into(),
        }];
        crate::catalog::apply_migration_with_cover(&self.storage, &self.schema, &verbs, Some(self))
    }

    /// Apply pending migrations from the supplied list, idempotent
    /// across repeated calls. Validates that the list hasn't been
    /// reordered or renamed at any already-applied ordinal.
    ///
    /// Typical usage flow (card 5/5):
    ///
    /// ```ignore
    /// let migrations = vec![
    ///     Migration::new("001_rename_user", |m| m.rename_type("User", "Account")),
    ///     Migration::new("002_score_to_float", |m| m.change_field_type(
    ///         "Account", "score", FieldType::Scalar(ScalarType::F64),
    ///         |_, v| Ok(Value::F64(/* ... */)),
    ///     )),
    /// ];
    /// let db = Database::open(initial_schema, dir)?;
    /// let report = db.run_migrations(migrations)?;
    /// drop(db);
    /// let db = Database::open(post_migration_schema, dir)?;
    /// ```
    pub fn run_migrations(
        &self,
        migrations: Vec<crate::catalog::Migration>,
    ) -> EngineResult<crate::catalog::MigrationLogReport> {
        self.check_not_migrated()?;
        let _migration_guard = self.migration_lock.write();
        self.run_migrations_inner(migrations)
    }

    fn run_migrations_inner(
        &self,
        migrations: Vec<crate::catalog::Migration>,
    ) -> EngineResult<crate::catalog::MigrationLogReport> {
        crate::catalog::run_migrations(&self.storage, &self.schema, migrations, Some(self))
    }

    /// Change a scalar field's type. The closure converts each
    /// existing object's value to the target type; the call holds
    /// `CATALOG_INIT_LOCK` and commits every re-encoded object plus
    /// the catalog kind-byte update as one atomic LSM batch.
    ///
    /// **Card 4/5 phase 1 scope.** Supports plain scalar fields only.
    /// `@indexed`, `@unique`, `@vectorize`, and relation kinds are
    /// refused with typed errors — each requires follow-on work
    /// (index rebuild under the new encoding, uniqueness re-check,
    /// embedding pipeline). The single-commit design is appropriate
    /// for offline / maintenance-window migrations of small to medium
    /// databases; large online migrations need the resumable
    /// shadow-field machinery the synthesis design spelled out, which
    /// is deferred.
    ///
    /// Like `rename_type`, the caller MUST drop this `Database` handle
    /// after a successful call and re-open with the post-change schema
    /// — the in-memory `field_kinds` / `schema` are stale otherwise.
    pub fn change_field_type<F>(
        &self,
        type_name: &str,
        field_name: &str,
        target_field_type: rhypedb_schema::FieldType,
        converter: F,
    ) -> EngineResult<crate::catalog::MigrationReport>
    where
        F: Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value>
            + Send
            + Sync
            + 'static,
    {
        self.check_not_migrated()?;
        let _migration_guard = self.migration_lock.write();
        self.change_field_type_inner(type_name, field_name, target_field_type, converter)
    }

    fn change_field_type_inner<F>(
        &self,
        type_name: &str,
        field_name: &str,
        target_field_type: rhypedb_schema::FieldType,
        converter: F,
    ) -> EngineResult<crate::catalog::MigrationReport>
    where
        F: Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value>
            + Send
            + Sync
            + 'static,
    {
        let target_kind = crate::catalog::schema_kind_byte_public(&target_field_type);
        let verb = crate::catalog::FieldTypeChangeVerb {
            type_name: type_name.into(),
            field_name: field_name.into(),
            target_kind,
            converter: Box::new(converter),
        };
        crate::catalog::apply_field_type_change(&self.storage, &self.schema, verb, Some(self))
    }

    // -----------------------------------------------------------------
    // Arc<Self>-consuming migrate variants (PR B).
    //
    // The non-consuming `rename_type` / `rename_field` / `change_field_type`
    // / `run_migrations` methods leave the caller responsible for
    // dropping the Database and reopening with the post-migration
    // schema. The `_consuming` variants take `self: Arc<Self>`, apply
    // the verb against the shared `Arc<LsmTree>`, rebuild the engine-
    // level derived state under the post-migration schema, and return a
    // fresh `Arc<Self>` — sharing storage, compaction worker,
    // subscription hub, in-memory generation counters, and
    // `next_object_id` with the old handle. The old handle is marked
    // `migrated` so any retained `Arc` surfaces `DatabaseMigratedAway`
    // instead of returning stale-cache results from the OLD schema's
    // `type_ids` / `indexed_fields`.
    //
    // What is reused (no rescan, no thread churn):
    // * `Arc<LsmTree>` — the storage stays open. WAL replay / SST
    //   rediscovery / compaction-worker thread all skipped.
    // * `Arc<SubscriptionHub>` — live subscriber channel receivers
    //   keep working across the migrate.
    // * `next_object_id` — preserved verbatim (no migration changes
    //   existing object IDs).
    // * In-memory `version_counters` map (cloned, since the old handle
    //   may still drop concurrently with the new handle running).
    // * `zone_field_id_lookup` Arc — the SAME `ArcSwap` instance the
    //   `LsmConfig::zone_extractor` closure already captured; the
    //   rebuild swaps in a new per-type table under the post-migration
    //   schema. No need to recreate or rewire the closure.
    //
    // What is fresh on the new handle:
    // * Cover-refresh worker. The old handle's worker exits naturally
    //   when the caller's old `Arc<Database>` drops (via the Drop impl's
    //   self-join guard); the new handle spawns its own worker if
    //   `options.background_cover_refresh` is set.
    // * All schema-shaped derived caches: `type_ids`, `field_ids`,
    //   `rel_ids`, `indexed_fields`, `incoming_relations`,
    //   `cascade_meta_by_id`, `type_name_by_id`, `retired_*`.
    // -----------------------------------------------------------------

    /// `Arc<Self>`-consuming variant of `rename_type`. Renames the type
    /// in the catalog (single LSM commit), then rebuilds engine-level
    /// derived state under `post_schema` and returns a fresh
    /// `Arc<Database>` sharing storage + subscribers + counters with
    /// the consumed handle. The returned report is the same one
    /// `rename_type` would return.
    pub fn rename_type_consuming(
        self: &Arc<Self>,
        old: &str,
        new: &str,
        post_schema: Schema,
    ) -> EngineResult<(crate::catalog::MigrationReport, Arc<Self>)> {
        // Take `migration_lock.write()` once and HOLD IT through the
        // verb + rebuild + poison atomic sequence. Without the held
        // lock, the window between the verb's commit (catalog migrated
        // on disk) and the poison flag being set could let a concurrent
        // writer's `create()` (waiting on `migration_lock.read()`)
        // acquire the lock and commit OLD-shape data into the post-
        // migration catalog era. With the lock held, the concurrent
        // writer remains blocked until the new handle exists and the
        // poison fires.
        //
        // `self: &Arc<Self>` — caller's Arc isn't consumed by the call.
        // On Err, the caller still owns their original Arc; on Ok they
        // get both the (now poisoned) original AND the new handle.
        // This sidesteps the "destroy the only Arc on benign validation
        // error" hazard the adversarial review flagged for the prior
        // `self: Arc<Self>` signature.
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        let report = self.rename_type_inner(old, new)?;
        let new_db = self.clone_into_new_handle(post_schema)?;
        Ok((report, new_db))
    }

    /// `Arc<Self>`-consuming variant of `rename_field`. See
    /// `rename_type_consuming` for the shared semantics.
    pub fn rename_field_consuming(
        self: &Arc<Self>,
        type_name: &str,
        old: &str,
        new: &str,
        post_schema: Schema,
    ) -> EngineResult<(crate::catalog::MigrationReport, Arc<Self>)> {
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        let report = self.rename_field_inner(type_name, old, new)?;
        let new_db = self.clone_into_new_handle(post_schema)?;
        Ok((report, new_db))
    }

    /// `Arc<Self>`-consuming variant of `change_field_type`. See
    /// `rename_type_consuming` for the shared semantics.
    pub fn change_field_type_consuming<F>(
        self: &Arc<Self>,
        type_name: &str,
        field_name: &str,
        target_field_type: rhypedb_schema::FieldType,
        converter: F,
        post_schema: Schema,
    ) -> EngineResult<(crate::catalog::MigrationReport, Arc<Self>)>
    where
        F: Fn(u64, &crate::object::Value) -> EngineResult<crate::object::Value>
            + Send
            + Sync
            + 'static,
    {
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        let report =
            self.change_field_type_inner(type_name, field_name, target_field_type, converter)?;
        let new_db = self.clone_into_new_handle(post_schema)?;
        Ok((report, new_db))
    }

    /// `Arc<Self>`-consuming variant of `run_migrations`. See
    /// `rename_type_consuming` for the shared semantics.
    pub fn run_migrations_consuming(
        self: &Arc<Self>,
        migrations: Vec<crate::catalog::Migration>,
        post_schema: Schema,
    ) -> EngineResult<(crate::catalog::MigrationLogReport, Arc<Self>)> {
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        let report = self.run_migrations_inner(migrations)?;
        let new_db = self.clone_into_new_handle(post_schema)?;
        Ok((report, new_db))
    }

    /// Build a fresh `Arc<Database>` sharing `Arc<LsmTree>`,
    /// `Arc<SubscriptionHub>`, `next_object_id`, the `version_counters`
    /// map, and the `zone_field_id_lookup` `Arc<ArcSwap>` with `self`;
    /// then mark `self` as migrated so any retained `Arc` to the old
    /// handle surfaces `DatabaseMigratedAway` on next use.
    ///
    /// Private — every public entry point that constructs a successor
    /// handle goes through one of the `_consuming` verbs above.
    fn clone_into_new_handle(
        self: &Arc<Self>,
        post_schema: Schema,
    ) -> EngineResult<Arc<Self>> {
        // Carry the SHARED Arc instances — not snapshots. This makes
        // `next_object_id`/`version_counters`/`migration_lock`/etc. one
        // and the same across OLD and NEW handles, closing the carry-
        // race window where the OLD handle could fetch_add a new id
        // (or bump a version_counter) after the snapshot but before
        // the poison and end up colliding with the NEW handle's first
        // operation.
        let carry = CarryState {
            subscriptions: Arc::clone(&self.subscriptions),
            migration_events: Arc::clone(&self.migration_events),
            next_object_id: Arc::clone(&self.next_object_id),
            version_counters: Arc::clone(&self.version_counters),
            version_counter_count: Arc::clone(&self.version_counter_count),
            migration_lock: Arc::clone(&self.migration_lock),
            converters: Arc::clone(&self.converters),
        };
        // Poison the OLD handle BEFORE rebuilding. If `rebuild_with_arc_storage`
        // succeeds, the OLD handle is already marked migrated so any
        // racing call to a public read/write method through `resolve_type_id`
        // surfaces `DatabaseMigratedAway` instead of stale-cache results.
        // If rebuild FAILS, the catalog change has already committed on
        // disk — the old handle is functionally dead (its in-memory
        // schema disagrees with the on-disk catalog), so poisoning it
        // is the safer failure mode than leaving it un-poisoned and
        // letting the caller observe inconsistent reads. The contract
        // becomes: on Err, drop the OLD handle and reopen.
        self.migrated
            .store(true, std::sync::atomic::Ordering::Release);
        Self::rebuild_with_arc_storage(
            Arc::clone(&self.storage),
            post_schema,
            self.opts.clone(),
            Arc::clone(&self.zone_field_id_lookup),
            Some(carry),
        )
    }

    /// Hot schema-reload primitive (server `/admin/reload` + auto-reload on
    /// migration completion). Rebuilds a fresh `Arc<Database>` under
    /// `post_schema`, sharing the SAME `Arc<LsmTree>` / subscribers / counters /
    /// converters as `self` (no second LSM/WAL open, no rescan) — i.e. exactly
    /// the `clone_into_new_handle` carry — but WITHOUT making a catalog change.
    ///
    /// Two deliberate differences from `clone_into_new_handle`:
    ///
    /// * **Non-poisoning.** `self.migrated` is left `false`, so the OLD handle
    ///   stays fully live. If the rebuild FAILS (e.g. the supplied SDL disagrees
    ///   with the on-disk catalog → `FieldKindChanged`), the caller still has a
    ///   working handle and the server keeps serving — a bad reload can't brick
    ///   it. The server quiesces in-flight requests around the swap (a
    ///   write-locked schema epoch), so nothing observes the old handle's stale
    ///   caches mid-request; the only other old-handle holders (the draining
    ///   cover-refresh worker, the vectorizer's schema snapshot) are benign for a
    ///   scalar field-type reload.
    ///
    /// * **Refuses while a migration is in flight.** The rebuilt handle does NOT
    ///   carry `migrating_fields` (CarryState omits it — the `_consuming`
    ///   invariant is "no unsettled plan"), so reloading mid-migration would
    ///   silently disarm the double-write hook → source-only writes that cutover
    ///   later loses/refuses. Holding `migration_lock.write()` across the check +
    ///   rebuild makes the guard race-free vs a concurrent migration arm/disarm.
    pub fn reload_handle(self: &Arc<Self>, post_schema: Schema) -> EngineResult<Arc<Self>> {
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        let armed = self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if armed > 0 {
            return Err(EngineError::ReloadBlockedByActiveMigration { armed });
        }
        let carry = CarryState {
            subscriptions: Arc::clone(&self.subscriptions),
            migration_events: Arc::clone(&self.migration_events),
            next_object_id: Arc::clone(&self.next_object_id),
            version_counters: Arc::clone(&self.version_counters),
            version_counter_count: Arc::clone(&self.version_counter_count),
            migration_lock: Arc::clone(&self.migration_lock),
            converters: Arc::clone(&self.converters),
        };
        // NON-poisoning: self.migrated stays false (see doc comment).
        Self::rebuild_with_arc_storage(
            Arc::clone(&self.storage),
            post_schema,
            self.opts.clone(),
            Arc::clone(&self.zone_field_id_lookup),
            Some(carry),
        )
    }

    /// Guardrail for read/write entry points. Returns
    /// `DatabaseMigratedAway` if this handle has been consumed by one
    /// of the `_consuming` migrate verbs.
    fn check_not_migrated(&self) -> EngineResult<()> {
        if self
            .migrated
            .load(std::sync::atomic::Ordering::Acquire)
        {
            Err(EngineError::DatabaseMigratedAway)
        } else {
            Ok(())
        }
    }

    /// If `Type.field` is tombstoned, return the typed `FieldRetired`
    /// error so callers using `schema.get_field` can short-circuit to
    /// a precise "this was retired" message before they encounter the
    /// inevitable `FieldNotFound` from the schema. Returns `None` if
    /// the field is live (or never existed — that's the caller's
    /// `FieldNotFound` path).
    fn field_retired_error(&self, type_name: &str, field: &str) -> Option<EngineError> {
        let qual = format!("{}.{}", type_name, field);
        if !self.tombstoned_field_quals.contains(&qual) {
            return None;
        }
        let field_id = self.field_ids.get(&qual).copied().unwrap_or(0);
        Some(EngineError::FieldRetired {
            type_name: type_name.into(),
            field: field.into(),
            field_id,
            retired_at_unix_ms: self
                .retired_at_ms_by_field_id
                .get(&field_id)
                .copied()
                .unwrap_or(0),
        })
    }

    /// Install (or replace) the card-2 double-write hook for `type_id`'s
    /// `hook.field_name`. MUST hold `migration_lock.write()`. Clone-on-write
    /// swap of the ArcSwap; keeps `migrating_field_count` in sync. Replacing an
    /// existing hook for the same field (e.g. re-arming with a now-resolved
    /// converter) does not change the count.
    pub(crate) fn arm_field_hook(&self, type_id: u64, hook: MigratingFieldHook) {
        let mut m = (**self.migrating_fields.load()).clone();
        let by_field = m.entry(type_id).or_default();
        let is_new = !by_field.contains_key(&hook.field_name);
        by_field.insert(hook.field_name.clone(), Arc::new(hook));
        self.migrating_fields.store(Arc::new(m));
        if is_new {
            self.migrating_field_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Remove the card-2 hook(s) for `plan_id` on `type_id` (its migration
    /// completed/cancelled). Keyed by plan id so callers needn't thread the
    /// field name. Same locking contract as `arm_field_hook`.
    pub(crate) fn disarm_field_hook(&self, type_id: u64, plan_id: u64) {
        let mut m = (**self.migrating_fields.load()).clone();
        let mut removed = 0usize;
        if let Some(by_field) = m.get_mut(&type_id) {
            let before = by_field.len();
            by_field.retain(|_, h| h.plan_id != plan_id);
            removed = before - by_field.len();
            if by_field.is_empty() {
                m.remove(&type_id);
            }
        }
        self.migrating_fields.store(Arc::new(m));
        if removed > 0 {
            self.migrating_field_count
                .fetch_sub(removed, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Card-2 double-write producer hook. Called on the create + update paths
    /// immediately BEFORE the object blob is serialized: for every field of
    /// `type_id` currently mid-migration that is present (non-Null) in `fields`,
    /// stamp a converted `<field>__shadow` sibling (+ `<field>__shadow_cv` =
    /// the converter version) so the write carries the migration forward. The
    /// `migration_field_count` gate (load `Relaxed`, early-return on 0) keeps a
    /// non-migrating database at one atomic load and no lock/map/String work.
    /// FAILS CLOSED (`MigrationFieldConverterUnresolved`) if a migrating field's
    /// converter isn't registered — never lands a source-only value that
    /// cutover would later refuse. Caller holds `migration_lock.read()` so the
    /// hook set is stable for the whole op.
    fn apply_migrating_field_hook(
        &self,
        type_id: u64,
        type_name: &str,
        object_id: u64,
        fields: &mut FieldMap,
    ) -> EngineResult<()> {
        if self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return Ok(());
        }
        let map = self.migrating_fields.load();
        let Some(by_field) = map.get(&type_id) else {
            return Ok(());
        };
        let mut shadows: Vec<(String, Value)> = Vec::new();
        for (field_name, hook) in by_field.iter() {
            let Some(value) = fields.get(field_name) else {
                continue; // this write doesn't touch the migrating field
            };
            if matches!(value, Value::Null) {
                continue; // Null/absent source carries no shadow (mirrors the worker)
            }
            let converter = hook.converter.as_ref().ok_or_else(|| {
                EngineError::MigrationFieldConverterUnresolved {
                    type_name: type_name.to_string(),
                    field: field_name.clone(),
                    plan_id: hook.plan_id,
                }
            })?;
            let new_value = converter(object_id, value)?;
            let got = crate::catalog::value_to_kind_byte_public(&new_value);
            if got != hook.target_kind {
                return Err(EngineError::Catalog(
                    crate::CatalogError::FieldTypeChangeConverterReturnedWrongKind {
                        qualified: format!("{type_name}.{field_name}"),
                        object_id,
                        got_kind: crate::catalog::kind_name_public(got),
                        want_kind: crate::catalog::kind_name_public(hook.target_kind),
                    },
                ));
            }
            shadows.push((format!("{field_name}__shadow"), new_value));
            shadows.push((
                format!("{field_name}__shadow_cv"),
                Value::U32(hook.converter_version),
            ));
        }
        for (k, v) in shadows {
            fields.insert(k, v);
        }
        Ok(())
    }

    /// Register a named row converter for chunked field-type migrations
    /// (shadow-field card 1). A migration is created against a converter
    /// `name`; the `(name, version)` pair is pinned in the persisted plan
    /// and re-resolved here at create and after a restart. Bumping
    /// `version` for a changed converter body makes an in-flight plan park
    /// `AwaitingConverter` rather than silently run the new logic over rows
    /// already converted by the old one. Per-`Database`, so converters in
    /// one tenant's DB never resolve in another's.
    pub fn register_converter<F>(&self, name: &str, version: u32, converter: F)
    where
        F: Fn(u64, &Value) -> EngineResult<Value> + Send + Sync + 'static,
    {
        self.converters
            .write()
            .insert(name.to_string(), (version, Arc::new(converter)));
    }

    /// Resolve a converter by `(name, version)`. `None` if absent or the
    /// registered version differs (→ the caller parks `AwaitingConverter`).
    fn resolve_converter(&self, name: &str, version: u32) -> Option<crate::catalog::RegisteredConverter> {
        self.converters
            .read()
            .get(name)
            .and_then(|(v, c)| (*v == version).then(|| Arc::clone(c)))
    }

    /// Create and run a chunked, crash-resumable field-type migration
    /// (shadow-field card 2/5). Synchronous: backfills a converted
    /// `<field>__shadow` sibling for every object in per-chunk commits, then
    /// cuts over (promote shadow → source + reconcile covers + flip the catalog
    /// kind), returning the durable plan id. ONLINE: concurrent writes proceed
    /// throughout via the double-write hook — a write to the migrating field
    /// whose converter is unresolved is the only one rejected
    /// (`MigrationFieldConverterUnresolved`, fail-closed).
    ///
    /// Unlike the single-commit [`Database::change_field_type`], this never
    /// holds `CATALOG_INIT_LOCK` across the scan-and-rewrite, commits at
    /// chunk boundaries, and resumes from the last durable cursor after a
    /// crash (see `open` auto-resume). The converter must be registered
    /// first via [`Database::register_converter`].
    ///
    /// Like the other migrate verbs, on success this handle's in-memory
    /// schema is STALE (the catalog kind changed underneath it): drop it and
    /// reopen with the schema where the field has the target type before
    /// issuing further writes to the migrated type.
    /// Create a chunked field-type migration and START it ASYNCHRONOUSLY,
    /// returning the plan id IMMEDIATELY (shadow-field card 3/5). A detached
    /// driver thread fans out `N` parallel partition workers over the
    /// pre-existing object range `[1, U)`, then runs the single-threaded cutover.
    ///
    /// CONTRACT CHANGE vs card 2: the verb NO LONGER drives to completion before
    /// returning. The plan + double-write hook are committed synchronously (so
    /// every write after this returns carries the migration forward), but the
    /// backfill + cutover run in the background. Use `wait_for_migration(plan_id)`
    /// to block until the driver finishes, `pause_migration` / `resume_field_type_migration`
    /// to control it. As with the other migrate verbs, on completion this handle's
    /// in-memory schema is stale (the catalog kind flipped underneath it) — reopen
    /// with the target schema before further writes to the migrated type.
    pub fn create_field_type_migration(&self, spec: MigrationPlanSpec) -> EngineResult<u64> {
        self.check_not_migrated()?;
        let target_kind = crate::catalog::schema_kind_byte_public(&spec.target_field_type);
        // Resolve the converter up front: fail fast rather than persist a
        // plan that can never run. (The resume-after-restart path, where the
        // plan already exists, parks `AwaitingConverter` instead.)
        let converter = self
            .resolve_converter(&spec.converter_name, spec.converter_version)
            .ok_or_else(|| EngineError::ConverterNotRegistered {
                name: spec.converter_name.clone(),
                version: spec.converter_version,
            })?;

        // Resolve the parallel degree before the lock (cheap): an explicit
        // `spec.parallel_degree` override (card 5), else one worker per CPU capped
        // at 8. Always clamped into `1..=MAX_PARALLEL_DEGREE`.
        let parallel_degree = match spec.parallel_degree {
            Some(n) => (n as usize).clamp(1, crate::catalog::MAX_PARALLEL_DEGREE as usize) as u8,
            None => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .clamp(1, 8)
                .min(crate::catalog::MAX_PARALLEL_DEGREE as usize) as u8,
        };

        // Setup under the migration write-barrier: validate, allocate + persist
        // the plan (with the parallel degree + the pre-existing id upper bound),
        // and arm the double-write hook atomically so no writer slips a
        // create/update into the migrating field between plan and hook (which
        // would land source-only with no shadow that cutover then refuses).
        let created = {
            let _guard = self.migration_lock.write();
            // U = `next_object_id` snapshot. Writers (`create`/`create_batch`)
            // take `migration_lock.read()` THEN `fetch_add`, so under this
            // write-lock the counter is FROZEN — U is exact and stays exact until
            // the lock releases. Every pre-existing object has id `< U` (a worker
            // converts it); every object created after the lock releases gets id
            // `>= U` and is born WITH the shadow via the now-armed hook (a worker
            // never touches it). Persisting U atomically with the plan (below)
            // closes any torn-U window.
            let id_upper_bound = self.next_object_id.load(Ordering::SeqCst);
            let created = crate::catalog::create_migration_plan(
                &self.storage,
                &self.schema,
                &spec.type_name,
                &spec.field_name,
                target_kind,
                &spec.converter_name,
                spec.converter_version,
                spec.chunk_size,
                Some(parallel_degree),
                id_upper_bound,
                spec.error_policy,
                spec.dry_run,
                spec.quarantine_cap,
            )?;
            // Install the double-write hook so live writes to the migrating field
            // carry it forward (card 2d — no quiesce; writes proceed). Card 4: a
            // DRY-RUN does NOT arm the hook — it's a preflight that writes nothing
            // and never cuts over, so live writes proceed normally (no shadow,
            // nothing to disarm or leak).
            if !spec.dry_run {
                self.arm_field_hook(
                    created.type_id,
                    MigratingFieldHook {
                        field_name: spec.field_name.clone(),
                        converter: Some(Arc::clone(&converter)),
                        target_kind,
                        converter_version: spec.converter_version,
                        plan_id: created.plan_id,
                    },
                );
            }
            created
        };

        // Spawn the detached driver WITHOUT holding `migration_lock` so live
        // writes (to this type and others) proceed during the (potentially long)
        // backfill; the double-write hook keeps every write's shadow current. On
        // a spawn failure, fall back to driving inline (synchronous) so the
        // migration still completes (a fresh plan id, so the inline drive's
        // registration can't collide).
        if self
            .spawn_migration_driver(created.plan_id, created.type_id, converter.clone())
            .is_err()
        {
            self.drive_migration_to_completion(created.plan_id, created.type_id, Some(&converter))?;
        }
        Ok(created.plan_id)
    }

    /// The double-driver gate (card 3/5), run while HOLDING the registry lock:
    /// refuse if an ACTIVE driver already owns `plan_id`, else REAP a finished
    /// leftover (its thread already `mark_done`d — drop its handle, detaching the
    /// spent thread). The `is_finished` check is lock-free (the signal's atomic),
    /// so this never nests the registry lock under the signal mutex. Callers then
    /// insert the fresh entry under the SAME lock (so the async path's handle is
    /// stored atomically with the entry — no window where `Database::drop` /
    /// `wait_for_migration` could see a `None` handle and fail to join the live
    /// thread).
    fn gate_locked(
        reg: &mut HashMap<u64, MigrationDriver>,
        plan_id: u64,
    ) -> EngineResult<()> {
        if let Some(existing) = reg.get(&plan_id) {
            if !existing.signal.is_finished() {
                return Err(EngineError::MigrationAlreadyRunning { plan_id });
            }
            reg.remove(&plan_id); // finished leftover — reap (detach its spent thread)
        }
        Ok(())
    }

    /// Register + spawn the detached async migration driver for `plan_id` (card
    /// 3/5), inserting the entry WITH its join handle atomically under the
    /// registry lock. The driver does NOT remove its own entry on exit — it
    /// `mark_done`s and returns, leaving the still-joinable handle for
    /// `wait_for_migration` / `Database::drop` to join (this is what makes a
    /// `wait; drop; reopen` race-free, AND ensures `Database::drop` always joins
    /// a live driver). Returns `MigrationAlreadyRunning` if an active driver
    /// already owns the plan, or a spawn IO error. Holding the registry lock
    /// across the spawn is deadlock-free: `migration_driver_main` never takes the
    /// registry lock (Design C — it never deregisters itself).
    fn spawn_migration_driver(
        &self,
        plan_id: u64,
        type_id: u64,
        converter: crate::catalog::RegisteredConverter,
    ) -> EngineResult<()> {
        use std::sync::atomic::AtomicU8;
        let weak = self.self_weak.lock().clone();
        let storage = Arc::clone(&self.storage);
        let events = Arc::clone(&self.migration_events);
        let mut reg = self.migration_drivers.lock();
        Self::gate_locked(&mut reg, plan_id)?;
        let control = Arc::new(AtomicU8::new(crate::catalog::migration_control::RUN));
        let signal = Arc::new(MigrationSignal::new());
        let handle = std::thread::Builder::new()
            .name("rhypedb-migration-driver".into())
            .spawn({
                let control = Arc::clone(&control);
                let signal = Arc::clone(&signal);
                move || {
                    migration_driver_main(
                        weak, storage, converter, control, signal, plan_id, type_id, events,
                    )
                }
            })
            .map_err(|e| EngineError::Storage(rhypedb_storage::Error::Io(e)))?;
        reg.insert(
            plan_id,
            MigrationDriver {
                control,
                signal,
                handle: Some(handle),
            },
        );
        Ok(())
    }

    /// Register an INLINE migration drive (resume / auto-resume) for `plan_id`
    /// (card 3/5). Same gate as the async spawn, but `handle` stays `None` (the
    /// drive runs on the calling thread). Returns the shared `(control, signal)`
    /// so the caller can thread the control into the workers and an
    /// `InlineDriveGuard` can signal/deregister on exit.
    fn register_inline_driver(
        &self,
        plan_id: u64,
    ) -> EngineResult<(Arc<std::sync::atomic::AtomicU8>, Arc<MigrationSignal>)> {
        use std::sync::atomic::AtomicU8;
        let mut reg = self.migration_drivers.lock();
        Self::gate_locked(&mut reg, plan_id)?;
        let control = Arc::new(AtomicU8::new(crate::catalog::migration_control::RUN));
        let signal = Arc::new(MigrationSignal::new());
        reg.insert(
            plan_id,
            MigrationDriver {
                control: Arc::clone(&control),
                signal: Arc::clone(&signal),
                handle: None,
            },
        );
        Ok((control, signal))
    }

    /// Block until the migration driver for `plan_id` finishes — the async
    /// create driver reaches a terminal disposition (Completed/Failed) or stops
    /// (Paused), or an inline resume drive returns — then JOIN its thread so it
    /// has fully exited (released its storage `Arc`) before returning, making a
    /// following `drop(db)` + reopen race-free. Returns the async driver's
    /// terminal error to the FIRST waiter (later waiters / the durable plan
    /// status are the multi-waiter source of truth). Returns immediately if no
    /// driver is registered (already reaped, or never started). Card 3/5: tests +
    /// operators use this to bridge the now-ASYNC create contract.
    pub fn wait_for_migration(&self, plan_id: u64) -> EngineResult<()> {
        // Take the handle + signal under the registry lock. Taking the handle
        // means only the first waiter joins; a later waiter still blocks on the
        // signal but finds `handle == None`.
        let (handle, signal) = {
            let mut reg = self.migration_drivers.lock();
            match reg.get_mut(&plan_id) {
                Some(d) => (d.handle.take(), Arc::clone(&d.signal)),
                None => return Ok(()), // no driver → already finished / never ran
            }
        };
        let err = signal.wait_take_error();
        if let Some(h) = handle
            && h.thread().id() != std::thread::current().id()
        {
            let _ = h.join(); // the driver `mark_done`s then returns immediately
        }
        // Reap: a finished async entry is not self-removed, so drop it now that we
        // have joined (idempotent if the gate / Drop got there first).
        self.migration_drivers.lock().remove(&plan_id);
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Request a PAUSE of the in-flight driver for `plan_id` (card 3/5). Its
    /// partition workers stop at the next chunk boundary (AC6: within one chunk),
    /// leaving the plan resumable (Running/Converting). No-op success if no
    /// driver is registered. Resume via `resume_field_type_migration`.
    pub fn pause_migration(&self, plan_id: u64) -> EngineResult<()> {
        if let Some(d) = self.migration_drivers.lock().get(&plan_id) {
            d.control.store(
                crate::catalog::migration_control::PAUSE,
                Ordering::SeqCst,
            );
        }
        Ok(())
    }

    /// TERMINAL cancel of a field-type migration (card 5). Rolls the plan back:
    /// stops any running workers, strips every partial `<field>__shadow` sibling
    /// back off the `o:` blobs (the source field is untouched in the Converting
    /// phase, so this restores the pre-migration shape losslessly), settles the
    /// plan `Cancelled`, and disarms the double-write hook.
    ///
    /// Refused once cutover has begun (`MigrationCannotCancelInCutover`) — the
    /// point of no return, since some source values have already been promoted to
    /// the converted value. Idempotent on an already-`Cancelled` plan; refused on
    /// a settled `Completed`/`DryRunCompleted` plan (`MigrationCannotCancelSettled`).
    ///
    /// Async-initiated when a driver is active: the durable `RollingBack` marker +
    /// the CANCEL control byte are set and the call returns; the winding-down
    /// driver runs the rollback (await via `wait_for_migration` / poll status).
    /// With no active driver (paused / crashed / `Failed` / never started) the
    /// rollback runs INLINE before returning.
    pub fn cancel_migration(&self, plan_id: u64) -> EngineResult<()> {
        // Decide + durably mark intent UNDER `migration_lock.write()` so this can
        // never race `run_cutover_locked` (which flips a plan to CuttingOver under
        // the SAME lock): the loser observes the winner's durable phase and never
        // acts on a half-decided plan.
        let inline = {
            let _guard = self.migration_lock.write();
            let plan = {
                let txn = self.storage.begin_txn();
                crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
            };
            match plan.status {
                crate::catalog::MigrationStatus::Cancelled => return Ok(()), // idempotent
                crate::catalog::MigrationStatus::Completed
                | crate::catalog::MigrationStatus::DryRunCompleted => {
                    return Err(EngineError::MigrationCannotCancelSettled { plan_id });
                }
                _ => {}
            }
            if plan.phase == crate::catalog::MigrationPhase::CuttingOver {
                return Err(EngineError::MigrationCannotCancelInCutover { plan_id });
            }
            let type_id = *self.type_ids.get(&plan.type_name).ok_or_else(|| {
                EngineError::TypeNotFound(plan.type_name.clone())
            })?;
            // Durably mark RollingBack BEFORE signalling CANCEL: a winding-down
            // driver that observes the (later) CANCEL store then re-loads the plan
            // is guaranteed to see this committed phase (commit happens-before the
            // SeqCst store; the worker's CANCEL read happens-after it).
            crate::catalog::set_plan_phase_rolling_back(&self.storage, plan_id)?;
            match self.migration_drivers.lock().get(&plan_id) {
                Some(d) => {
                    d.control
                        .store(crate::catalog::migration_control::CANCEL, Ordering::SeqCst);
                    None // an active driver completes the rollback on winddown
                }
                None => Some(type_id), // no driver → we roll back inline below
            }
        };
        if let Some(type_id) = inline {
            // No active driver: drive the rollback inline (routes RollingBack →
            // run_cancel_rollback). `drive_migration_to_completion` registers an
            // inline driver (gated against a concurrent driver) + takes the write
            // lock inside the terminal pass, so the lock released above is re-taken
            // there — no self-deadlock.
            self.drive_migration_to_completion(plan_id, type_id, None)?;
        }
        Ok(())
    }

    /// Explicitly run a plan's cutover (card 5 — the `/admin/migrations/:id/cutover`
    /// surface). For an AllDone-but-parked plan (e.g. after clearing a quarantine
    /// block); the normal flow cuts over automatically at the end of the backfill.
    /// Refuses a cancelled / rolling-back plan with `MigrationCancelledCannotCutover`.
    pub fn cutover_migration(&self, plan_id: u64) -> EngineResult<()> {
        let _guard = self.migration_lock.write();
        let plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        if plan.status == crate::catalog::MigrationStatus::Cancelled
            || plan.phase == crate::catalog::MigrationPhase::RollingBack
        {
            return Err(EngineError::MigrationCancelledCannotCutover { plan_id });
        }
        let type_id = *self
            .type_ids
            .get(&plan.type_name)
            .ok_or_else(|| EngineError::TypeNotFound(plan.type_name.clone()))?;
        self.guard_resume_schema(&plan)?;
        self.run_cutover_locked(plan, type_id)
    }

    /// Drive a plan from its current phase to completion (shadow-field card 2).
    ///
    /// Phase `Converting`: backfill every `<field>__shadow` sibling WITHOUT
    /// holding `migration_lock` so writers to OTHER types proceed during the
    /// (potentially long) scan. Then transition to the cutover. Phase
    /// `CuttingOver` (a resumed plan): the converter is NOT re-run — only the
    /// rename pass resumes.
    ///
    /// The cutover (`run_cutover`) promotes the shadows to the source field,
    /// reconciles every cover/index, flips the catalog kind, and disarms the
    /// double-write hook — holding `migration_lock.write()` for the whole pass.
    /// On a converter / data / shadow error the plan is left `Failed` (the hook
    /// stays armed, so writes to the migrating field keep failing closed until
    /// it is resolved + resumed); the hook is disarmed only on a clean `Completed`.
    fn drive_migration_to_completion(
        &self,
        plan_id: u64,
        type_id: u64,
        converter: Option<&crate::catalog::RegisteredConverter>,
    ) -> EngineResult<()> {
        let plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        // Card 5: a plan already flipped to RollingBack (an operator cancel, or a
        // crashed/auto-resumed rollback) skips the backfill entirely — the
        // terminal pass strips its shadows. No converter required (rollback never
        // converts).
        if plan.phase == crate::catalog::MigrationPhase::RollingBack {
            return self.run_terminal_pass(plan_id, type_id);
        }
        // The converter is needed ONLY to backfill shadows in the Converting
        // phase. A plan resumed in the CuttingOver phase (a crash mid-cutover)
        // is a pure rename pass — don't demand a converter the operator has no
        // reason to re-register.
        if plan.phase == crate::catalog::MigrationPhase::Converting {
            let converter = converter.ok_or_else(|| EngineError::ConverterNotRegistered {
                name: plan.converter_name.clone(),
                version: plan.converter_version,
            })?;
            match plan.parallel_degree {
                Some(n) => {
                    // Card 3: parallel backfill INLINE (on this thread — resume /
                    // auto-resume are synchronous). Register a control so a
                    // concurrent pause/cancel/wait works during the drive AND to
                    // gate a second driver on this plan. The `InlineDriveGuard`
                    // signals done + deregisters on EVERY exit path (incl. `?` and
                    // the cutover below). Boundaries recompute from the PINNED
                    // `(n, id_upper_bound)` so they match the persisted `c:S:`
                    // cursors exactly.
                    let (control, signal) = self.register_inline_driver(plan_id)?;
                    let _guard = InlineDriveGuard {
                        db: self,
                        plan_id,
                        signal,
                    };
                    // Capture the disposition WITHOUT `?` — a cancel that landed
                    // mid-backfill must roll back even if a worker errored first.
                    let disp_result = crate::catalog::run_parallel_backfill(
                        &self.storage,
                        plan_id,
                        type_id,
                        n.max(1),
                        plan.id_upper_bound,
                        &plan.type_name,
                        &plan.field_name,
                        plan.src_kind,
                        plan.target_kind,
                        plan.converter_version,
                        plan.chunk_size,
                        converter,
                        &control,
                        plan.error_policy,
                        plan.dry_run,
                        plan.quarantine_cap,
                        &plan.converter_name,
                        Some(&self.migration_events),
                    );
                    // B8 (card 5): re-load AFTER the backfill returns, BEFORE
                    // propagating any error. A `cancel_migration` that landed while
                    // the workers were stopping (or that raced the AllDone→cutover
                    // handoff, or that beat a Stop-policy worker error that parked
                    // the plan Failed) has durably flipped the plan to RollingBack
                    // under `migration_lock.write()`. Routing through the terminal
                    // pass means the cancel is never lost — the rollback completes
                    // and supersedes a backfill error (the partial state is being
                    // discarded anyway); cutover never runs on a cancelled plan.
                    let post = {
                        let txn = self.storage.begin_txn();
                        crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
                    };
                    if post.phase == crate::catalog::MigrationPhase::RollingBack {
                        return self.run_terminal_pass(plan_id, type_id);
                    }
                    // No cancel → surface a genuine backfill error (already parked
                    // Failed) now.
                    let disp = disp_result?;
                    if matches!(disp, crate::catalog::BackfillDisposition::Paused) {
                        return Ok(()); // genuine pause — plan left resumable
                    }
                    // AllDone. A dry-run cleans up + marks DryRunCompleted (no
                    // cutover); a real plan cuts over (still under `_guard`, so a
                    // waiter wakes only after cutover finishes/fails).
                    if plan.dry_run {
                        crate::catalog::finalize_dry_run(&self.storage, plan_id)?;
                        self.migration_events.publish(MigrationEvent::StatusChanged {
                            plan_id,
                            status: crate::catalog::MigrationStatus::DryRunCompleted,
                        });
                        return Ok(());
                    }
                    return self.run_terminal_pass(plan_id, type_id);
                }
                None => {
                    // Legacy card-1/2 single-worker plan (no partitions): the
                    // pre-3b/2 path, only reached by resuming an old plan.
                    crate::catalog::run_migration_chunks(&self.storage, plan_id, converter)?;
                }
            }
        }
        self.run_terminal_pass(plan_id, type_id)
    }

    /// Run the terminal pass for a plan whose backfill is done (card 5). Takes
    /// `migration_lock.write()` ONCE, loads the plan UNDER the lock, and routes
    /// by phase: `RollingBack` → `run_cancel_rollback_locked` (strip shadows),
    /// else → `run_cutover_locked` (promote shadows). Routing under the single
    /// write-lock acquisition is what makes cutover-vs-cancel race-free:
    /// `cancel_migration` flips the plan to `RollingBack` under the SAME lock, so
    /// the loser of that lock observes the winner's durable decision and never
    /// runs the wrong pass on a half-decided plan.
    fn run_terminal_pass(&self, plan_id: u64, type_id: u64) -> EngineResult<()> {
        let _guard = self.migration_lock.write();
        let plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        if plan.phase == crate::catalog::MigrationPhase::RollingBack {
            self.run_cancel_rollback_locked(plan, type_id)
        } else {
            self.run_cutover_locked(plan, type_id)
        }
    }

    /// Cutover pass (shadow-field card 2): promote every `<field>__shadow`
    /// sibling to the source field, reconcile ALL covers/indexes via the shared
    /// `rewrite_object_and_maintain_covers`, then flip the catalog kind and
    /// disarm the double-write hook.
    ///
    /// `_locked`: the caller (`run_terminal_pass` / `cutover_migration`) HOLDS
    /// `migration_lock.write()` for the WHOLE pass and has loaded `plan` under
    /// it. The write-lock spans every commit, so once it is acquired no in-flight
    /// writer (card-2 inc 2d) AND no background cover-refresh pass (which also
    /// takes `migration_lock.read()`) can add a shadow or stamp a stale `cover_v`
    /// racing a rename, and a reader (which needs no lock) sees each row
    /// transition source→target atomically (each row's rename is one commit).
    /// Reads stay available throughout — only writes are drained.
    ///
    /// Per-chunk commit order mirrors the backfill worker: `[promoted o: blob,
    /// i:/r:/g: cover maintenance, plan record (cutover_cursor) LAST]`, so a
    /// torn tail drops only the cursor advance and resume re-does the chunk
    /// idempotently (a row already promoted — source at target kind, no shadow —
    /// is skipped). The bounded `WriteConflict` retry is belt-and-suspenders for
    /// any same-keyspace racer; the generation over-bump on a retry is harmless
    /// (monotonic staleness counter).
    fn run_cutover_locked(
        &self,
        mut plan: crate::catalog::MigrationPlan,
        type_id: u64,
    ) -> EngineResult<()> {
        const WRITE_CONFLICT_RETRIES: u32 = 8;
        let plan_id = plan.plan_id;
        // Defense in depth (card 5): never cut over a cancelled / rolling-back
        // plan. `run_terminal_pass` routes a RollingBack plan to the rollback;
        // `cutover_migration` refuses earlier — this guards any direct path so a
        // promotion can never overwrite the source of a plan being abandoned.
        if plan.status == crate::catalog::MigrationStatus::Cancelled
            || plan.phase == crate::catalog::MigrationPhase::RollingBack
        {
            return Err(EngineError::MigrationCancelledCannotCutover { plan_id });
        }
        let type_name = plan.type_name.clone();
        let field_name = plan.field_name.clone();
        let target_kind = plan.target_kind;
        let converter_version = plan.converter_version;
        let error_policy = plan.error_policy;
        let shadow_name = format!("{field_name}__shadow");
        let shadow_cv_name = format!("{field_name}__shadow_cv");

        // Card 4 cutover gate: a Quarantine plan with UNRESOLVED quarantine rows
        // cannot cut over — the operator must `retry_quarantined` / `clear_quarantine`
        // first (the gate self-corrects against hook self-heals + reaps resolved
        // sidecars). Park `Failed` (hook armed, resumable) so the block is observable;
        // resume re-checks. (SkipAndLog/Stop → gate returns 0 and proceeds.)
        let unresolved = crate::catalog::cutover_quarantine_gate(&self.storage, &plan, type_id)?;
        if unresolved > 0 {
            crate::catalog::park_migration_failed_keep_cursors(&self.storage, plan_id)?;
            return Err(EngineError::MigrationCutoverHasErrors {
                plan_id,
                error_count: unresolved,
            });
        }

        // Durably mark CuttingOver BEFORE the first rename so a crash resumes the
        // rename pass, not the converter. Idempotent (already CuttingOver → skip).
        if plan.phase != crate::catalog::MigrationPhase::CuttingOver {
            plan.phase = crate::catalog::MigrationPhase::CuttingOver;
            let mut txn = self.storage.begin_txn();
            let (k, v) = crate::catalog::migration_plan_record(&plan);
            self.storage.put(&mut txn, &k, v)?;
            self.storage.commit(&mut txn)?;
        }
        self.migration_events
            .publish(MigrationEvent::CutoverStarted { plan_id });

        let chunk_size = if plan.chunk_size == 0 {
            crate::catalog::DEFAULT_MIGRATION_CHUNK_SIZE
        } else {
            plan.chunk_size
        } as usize;
        let object_prefix = KeyBuilder::object_prefix(type_id);
        let mut cursor = plan.cutover_cursor;

        loop {
            let start = if cursor == 0 {
                object_prefix.clone()
            } else {
                match cursor.checked_add(1) {
                    Some(next) => KeyBuilder::object(type_id, next),
                    None => break,
                }
            };

            let mut attempts = 0u32;
            let committed: Option<(crate::catalog::MigrationPlan, bool)> = loop {
                let mut txn = self.storage.begin_txn();
                let snap = txn.snapshot();
                let chunk = self.storage.scan_chunk_raw(snap, &object_prefix, &start, chunk_size)?;
                let Some(high_water) = chunk.high_water.clone() else {
                    self.storage.abort(&mut txn);
                    break None;
                };
                let more = chunk.more;
                let next_cursor =
                    u64::from_be_bytes(high_water[high_water.len() - 8..].try_into().unwrap());

                // Objects whose in-memory generation this attempt bumped (via
                // rewrite_object_and_maintain_covers). On any path that does NOT
                // commit — a refusal park, a WriteConflict retry, or a terminal
                // storage error — these must be rolled back, mirroring update()'s
                // commit-failure handling, so the live handle's generation
                // counters don't drift ahead of the durable `g:` keys.
                let mut bumped: Vec<u64> = Vec::new();
                for (key, blob) in &chunk.live {
                    let object_id =
                        u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap());
                    let mut fields = deserialize_fields(blob);
                    match fields.get(&shadow_name).cloned() {
                        None => {
                            // No shadow. Skip rows that are already cut over
                            // (source at target kind), absent, or Null (Null
                            // carries no shadow); refuse a source-at-src-kind
                            // row the converter never reached.
                            match fields.get(&field_name) {
                                None | Some(Value::Null) => continue,
                                Some(v)
                                    if crate::catalog::value_to_kind_byte_public(v)
                                        == target_kind =>
                                {
                                    continue;
                                }
                                // Card 4: under a non-Stop policy this is an
                                // ACCEPTED error row (SkipAndLog left it, or a
                                // Quarantine row the operator cleared) — the gate
                                // already confirmed no UNRESOLVED quarantine rows
                                // remain, so LEAVE it source-shape. Under Stop a
                                // missing shadow is a genuine torn/unreached row →
                                // refuse + rewind.
                                Some(_)
                                    if error_policy != crate::catalog::ErrorPolicy::Stop =>
                                {
                                    continue;
                                }
                                Some(_) => {
                                    self.storage.abort(&mut txn);
                                    for oid in &bumped {
                                        self.rollback_version(type_id, *oid);
                                    }
                                    // Rewind to Converting so resume re-backfills
                                    // the missing shadow rather than re-refusing.
                                    crate::catalog::park_migration_failed_rewind(
                                        &self.storage,
                                        plan_id,
                                    )?;
                                    return Err(EngineError::MigrationCutoverShadowMissing {
                                        plan_id,
                                        object_id,
                                    });
                                }
                            }
                        }
                        Some(shadow_val) => {
                            // Refuse a shadow written by a stale converter.
                            let found_cv = match fields.get(&shadow_cv_name) {
                                Some(Value::U32(v)) => *v,
                                _ => 0,
                            };
                            if found_cv != converter_version {
                                self.storage.abort(&mut txn);
                                for oid in &bumped {
                                    self.rollback_version(type_id, *oid);
                                }
                                crate::catalog::park_migration_failed_rewind(
                                    &self.storage,
                                    plan_id,
                                )?;
                                return Err(EngineError::MigrationCutoverShadowStale {
                                    plan_id,
                                    object_id,
                                    found_cv,
                                    want_cv: converter_version,
                                });
                            }
                            // Snapshot indexed-field values BEFORE the rename
                            // (the migrating field is non-indexed, so siblings
                            // are unchanged — this drives the covering re-put).
                            let old_indexed_snapshot: Vec<Option<Value>> =
                                if let Some(idx_fields) = self.indexed_fields.get(&type_name) {
                                    idx_fields
                                        .iter()
                                        .map(|ifd| fields.get(&ifd.name).cloned())
                                        .collect()
                                } else {
                                    Vec::new()
                                };
                            // Promote: source := shadow value, drop both shadow
                            // siblings → shadow-free blob.
                            fields.insert(field_name.clone(), shadow_val);
                            fields.remove(&shadow_name);
                            fields.remove(&shadow_cv_name);
                            let serialized = serialize_fields(&fields);
                            // Cover/index maintenance FIRST, the promoted o: blob
                            // LAST: these are individual puts (not one atomic
                            // put_batch), so the `o:` write — the one that flips
                            // this row's "already cut?" decision — must be last.
                            // A torn tail that drops `o:` leaves the shadow in
                            // place, so resume re-promotes the row and redoes the
                            // (idempotent) cover maintenance; a torn tail that
                            // keeps `o:` necessarily kept every earlier write too.
                            self.rewrite_object_and_maintain_covers(
                                &mut txn,
                                &type_name,
                                type_id,
                                object_id,
                                &fields,
                                &serialized,
                                &old_indexed_snapshot,
                                true,
                            )?;
                            bumped.push(object_id);
                            self.storage.put(&mut txn, key, serialized.clone())?;
                        }
                    }
                }

                let mut plan_after = plan.clone();
                plan_after.cutover_cursor = next_cursor;
                // phase stays CuttingOver, status stays Running.
                let (k, v) = crate::catalog::migration_plan_record(&plan_after);
                self.storage.put(&mut txn, &k, v)?;
                match self.storage.commit(&mut txn) {
                    Ok(_) => break Some((plan_after, more)),
                    Err(rhypedb_storage::Error::WriteConflict)
                        if attempts < WRITE_CONFLICT_RETRIES =>
                    {
                        // Release the conflicted txn's snapshot + undo this
                        // attempt's generation bumps before re-scanning.
                        self.storage.abort(&mut txn);
                        for oid in &bumped {
                            self.rollback_version(type_id, *oid);
                        }
                        attempts += 1;
                        continue;
                    }
                    Err(e) => {
                        self.storage.abort(&mut txn);
                        for oid in &bumped {
                            self.rollback_version(type_id, *oid);
                        }
                        return Err(match e {
                            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                            other => EngineError::Storage(other),
                        });
                    }
                }
            };

            match committed {
                None => break,
                Some((plan_after, more)) => {
                    plan = plan_after;
                    cursor = plan.cutover_cursor;
                    if !more {
                        break;
                    }
                }
            }
        }

        // Flip the catalog kind + mark Completed (idempotent), then disarm the
        // double-write hook. Already holding migration_lock.write().
        crate::catalog::finalize_migration_cutover(&self.storage, plan_id)?;
        self.disarm_field_hook(type_id, plan_id);
        self.migration_events
            .publish(MigrationEvent::CutoverDone { plan_id });
        self.migration_events
            .publish(MigrationEvent::StatusChanged {
                plan_id,
                status: crate::catalog::MigrationStatus::Completed,
            });
        Ok(())
    }

    /// Terminal-cancel rollback pass (card 5) — the INVERSE of
    /// `run_cutover_locked`. Strips every `<field>__shadow`/`<field>__shadow_cv`
    /// sibling back off the `o:` blobs (the source field is never mutated in the
    /// Converting phase, so this restores the pre-migration shape losslessly),
    /// reconciling covers/indexes via the shared
    /// `rewrite_object_and_maintain_covers` (the gen-bump invalidates any cover
    /// embedding the row so it re-fetches the now-shadow-free blob after the hook
    /// disarms). Then settles the plan `Cancelled` and disarms the hook.
    ///
    /// `_locked`: the caller (`run_terminal_pass`) HOLDS `migration_lock.write()`
    /// for the whole pass + loaded `plan` under it. Same crash-safe per-chunk
    /// order as cutover: `[cover-maint, stripped o: blob, plan(cutover_cursor)
    /// LAST]` — a torn tail drops only the cursor advance and resume re-strips the
    /// chunk idempotently (a row already shadow-free is skipped). Reuses
    /// `cutover_cursor` as the rollback cursor.
    ///
    /// Why the hook stays armed until the end: while armed,
    /// `migration_in_flight()` is true, so every cover builder strips shadows — no
    /// cover can bake one in. Disarming runs LAST (after every `o:` is
    /// shadow-free), so there is never a window where a shadow is on disk with the
    /// hook gone (the leak the deferral warned about).
    fn run_cancel_rollback_locked(
        &self,
        mut plan: crate::catalog::MigrationPlan,
        type_id: u64,
    ) -> EngineResult<()> {
        const WRITE_CONFLICT_RETRIES: u32 = 8;
        let plan_id = plan.plan_id;
        let type_name = plan.type_name.clone();
        let field_name = plan.field_name.clone();
        let shadow_name = format!("{field_name}__shadow");
        let shadow_cv_name = format!("{field_name}__shadow_cv");

        // Establish the durable RollingBack marker if a direct call reached here
        // without `cancel_migration` having set it (idempotent otherwise).
        if plan.phase != crate::catalog::MigrationPhase::RollingBack {
            crate::catalog::set_plan_phase_rolling_back(&self.storage, plan_id)?;
            plan.phase = crate::catalog::MigrationPhase::RollingBack;
            plan.cutover_cursor = 0;
        }
        self.migration_events
            .publish(MigrationEvent::RollbackStarted { plan_id });

        let chunk_size = if plan.chunk_size == 0 {
            crate::catalog::DEFAULT_MIGRATION_CHUNK_SIZE
        } else {
            plan.chunk_size
        } as usize;
        let object_prefix = KeyBuilder::object_prefix(type_id);
        let mut cursor = plan.cutover_cursor;

        loop {
            let start = if cursor == 0 {
                object_prefix.clone()
            } else {
                match cursor.checked_add(1) {
                    Some(next) => KeyBuilder::object(type_id, next),
                    None => break,
                }
            };

            let mut attempts = 0u32;
            let committed: Option<(crate::catalog::MigrationPlan, bool)> = loop {
                let mut txn = self.storage.begin_txn();
                let snap = txn.snapshot();
                let chunk = self.storage.scan_chunk_raw(snap, &object_prefix, &start, chunk_size)?;
                let Some(high_water) = chunk.high_water.clone() else {
                    self.storage.abort(&mut txn);
                    break None;
                };
                let more = chunk.more;
                let next_cursor =
                    u64::from_be_bytes(high_water[high_water.len() - 8..].try_into().unwrap());

                let mut bumped: Vec<u64> = Vec::new();
                for (key, blob) in &chunk.live {
                    let object_id =
                        u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap());
                    let mut fields = deserialize_fields(blob);
                    // Skip a row with no shadow: either already rolled back, or it
                    // never carried one (the converter never reached it, or a Null
                    // source). The source field is left exactly as-is.
                    if !fields.contains_key(&shadow_name)
                        && !fields.contains_key(&shadow_cv_name)
                    {
                        continue;
                    }
                    // Snapshot indexed-field values BEFORE stripping (the migrating
                    // field is non-indexed, so the indexed siblings are unchanged —
                    // this drives the covering re-put / gen-bump).
                    let old_indexed_snapshot: Vec<Option<Value>> =
                        if let Some(idx_fields) = self.indexed_fields.get(&type_name) {
                            idx_fields
                                .iter()
                                .map(|ifd| fields.get(&ifd.name).cloned())
                                .collect()
                        } else {
                            Vec::new()
                        };
                    // Strip both shadow siblings; KEEP the source field.
                    fields.remove(&shadow_name);
                    fields.remove(&shadow_cv_name);
                    let serialized = serialize_fields(&fields);
                    // Cover/index maintenance FIRST, stripped o: blob LAST (the
                    // write that flips this row's "already stripped?" decision).
                    self.rewrite_object_and_maintain_covers(
                        &mut txn,
                        &type_name,
                        type_id,
                        object_id,
                        &fields,
                        &serialized,
                        &old_indexed_snapshot,
                        true,
                    )?;
                    bumped.push(object_id);
                    self.storage.put(&mut txn, key, serialized.clone())?;
                }

                let mut plan_after = plan.clone();
                plan_after.cutover_cursor = next_cursor;
                let (k, v) = crate::catalog::migration_plan_record(&plan_after);
                self.storage.put(&mut txn, &k, v)?;
                match self.storage.commit(&mut txn) {
                    Ok(_) => break Some((plan_after, more)),
                    Err(rhypedb_storage::Error::WriteConflict)
                        if attempts < WRITE_CONFLICT_RETRIES =>
                    {
                        self.storage.abort(&mut txn);
                        for oid in &bumped {
                            self.rollback_version(type_id, *oid);
                        }
                        attempts += 1;
                        continue;
                    }
                    Err(e) => {
                        self.storage.abort(&mut txn);
                        for oid in &bumped {
                            self.rollback_version(type_id, *oid);
                        }
                        return Err(match e {
                            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                            other => EngineError::Storage(other),
                        });
                    }
                }
            };

            match committed {
                None => break,
                Some((plan_after, more)) => {
                    plan = plan_after;
                    cursor = plan.cutover_cursor;
                    if !more {
                        break;
                    }
                }
            }
        }

        // Settle Cancelled (delete c:S:/c:Q:, NO kind flip), then disarm the hook.
        // Already holding migration_lock.write(); the strip above completed first,
        // so once Cancelled is durable no shadow remains on disk.
        crate::catalog::finalize_migration_cancelled(&self.storage, plan_id)?;
        self.disarm_field_hook(type_id, plan_id);
        self.migration_events
            .publish(MigrationEvent::StatusChanged {
                plan_id,
                status: crate::catalog::MigrationStatus::Cancelled,
            });
        Ok(())
    }

    /// Refuse to DRIVE a plan unless the open schema declares the field at the
    /// plan's TARGET kind (shadow-field card 1, blocker F3). Driving flips the
    /// catalog to the target; if this handle still validates writes against
    /// the source kind (operator reopened with the OLD schema), finishing the
    /// migration would silently corrupt. The operator must reopen with the
    /// target schema first.
    fn guard_resume_schema(&self, plan: &crate::catalog::MigrationPlan) -> EngineResult<()> {
        self.guard_resume_schema_for(plan, plan.target_kind)
    }

    /// Card 5: phase-aware variant of [`guard_resume_schema`]. A forward
    /// migration (Converting/CuttingOver) demands the TARGET kind; a RollingBack
    /// plan demands the SOURCE kind (the operator abandoned the migration, so
    /// their SDL reverted the field to `src_kind`). Driving the wrong direction
    /// against a mismatched schema would corrupt, so refuse.
    fn guard_resume_schema_for(
        &self,
        plan: &crate::catalog::MigrationPlan,
        want_kind: u8,
    ) -> EngineResult<()> {
        let want = self
            .schema
            .get_type(&plan.type_name)
            .and_then(|td| td.fields.iter().find(|f| f.name == plan.field_name))
            .map(|fd| crate::catalog::schema_kind_byte_public(&fd.field_type));
        if want != Some(want_kind) {
            return Err(EngineError::MigrationResumeSchemaMismatch {
                plan_id: plan.plan_id,
                expected: crate::catalog::kind_name_public(want_kind),
                found: want
                    .map(crate::catalog::kind_name_public)
                    .unwrap_or("<absent>"),
            });
        }
        Ok(())
    }

    /// Open-path hook (shadow-field card 2): re-establish the double-write hook
    /// from the persisted `c:P:` plans and resume any drivable migration that
    /// can proceed (a Converting plan whose converter is already registered, or
    /// any CuttingOver plan — a rename-only pass). Runs ONLY on a genuine open
    /// (not the `_consuming` rebuild — see `rebuild_with_arc_storage`).
    ///
    /// At a fresh open the per-`Database` converter registry is empty (the
    /// operator registers converters AFTER open), so a Converting plan is armed
    /// but NOT driven here — the operator calls `resume_field_type_migration`
    /// after registering. Every unsettled plan re-arms the field hook: writes to
    /// the migrating field whose converter is unresolved FAIL CLOSED
    /// (`MigrationFieldConverterUnresolved`) across the restart, while all other
    /// writes proceed (card 2d — no type-wide quiesce).
    fn auto_resume_migrations(&self) -> EngineResult<()> {
        let plans = {
            let txn = self.storage.begin_txn();
            let snap = txn.snapshot();
            crate::catalog::scan_migration_plans(&self.storage, snap)?
        };
        for plan in plans {
            if !plan.status.quiesces() {
                continue; // Completed / Cancelled — settled, no hook needed
            }
            let Some(&type_id) = self.type_ids.get(&plan.type_name) else {
                continue; // type no longer exists — nothing to re-arm
            };
            // Card 4: a dry-run preflight NEVER arms the double-write hook (it
            // writes no shadows and never flips the catalog kind), so arming it on
            // reopen would brick live writes to the field with no recovery. Re-drive
            // the preflight to completion when its converter is available (settles it
            // `DryRunCompleted`); otherwise leave it Running for an explicit
            // `resume_field_type_migration` — it still does NOT brick writes (no hook).
            if plan.dry_run {
                let converter =
                    self.resolve_converter(&plan.converter_name, plan.converter_version);
                if plan.status.is_drivable() && converter.is_some() {
                    self.drive_migration_to_completion(plan.plan_id, type_id, converter.as_ref())?;
                }
                continue;
            }
            // Converter is empty at a fresh open (operator registers AFTER
            // open) → the hook arms in a REJECTING (converter: None) state, so a
            // live write to the migrating field fails closed until it resolves.
            let converter = self.resolve_converter(&plan.converter_name, plan.converter_version);
            {
                let _guard = self.migration_lock.write();
                // Rebuild the double-write hook from the plan.
                self.arm_field_hook(
                    type_id,
                    MigratingFieldHook {
                        field_name: plan.field_name.clone(),
                        converter: converter.clone(),
                        target_kind: plan.target_kind,
                        converter_version: plan.converter_version,
                        plan_id: plan.plan_id,
                    },
                );
            }
            // Drive a drivable plan when we can: a Converting plan needs its
            // converter registered (carried across a _consuming rebuild, or a
            // re-open where the operator registered before open); a CuttingOver
            // plan (crashed mid-cutover) is a pure rename pass; a RollingBack plan
            // (card 5 — a crashed terminal cancel) is a pure strip pass — both
            // resume with NO converter, so a reopen finishes them even though the
            // per-`Database` converter registry is empty after restart.
            let is_cutting = plan.phase == crate::catalog::MigrationPhase::CuttingOver;
            let is_rolling_back = plan.phase == crate::catalog::MigrationPhase::RollingBack;
            // A RollingBack plan MUST always complete its rollback — even if a
            // worker error parked it `Failed` before the cancel's strip ran (Failed
            // is not `is_drivable`, so it is gated separately here). Forward passes
            // (Converting/CuttingOver) still require a drivable status.
            let drive = is_rolling_back
                || (plan.status.is_drivable() && (converter.is_some() || is_cutting));
            if drive {
                // A RollingBack plan validates against the SOURCE kind (the
                // operator abandoned the migration); forward passes need TARGET.
                if is_rolling_back {
                    self.guard_resume_schema_for(&plan, plan.src_kind)?;
                } else {
                    self.guard_resume_schema(&plan)?;
                }
                self.drive_migration_to_completion(
                    plan.plan_id,
                    type_id,
                    converter.as_ref(),
                )?;
            }
        }
        Ok(())
    }

    /// Resume an in-flight chunked field-type migration after a restart (or
    /// after it parked waiting for its converter). Resolves the plan's pinned
    /// `(converter_name, converter_version)` from this `Database`'s registry —
    /// register it first — then drives the plan to completion and cuts over.
    /// No-op for an already-settled (`Completed`/`Cancelled`) plan.
    ///
    /// Like the other migrate verbs, on success this handle's in-memory schema
    /// is stale (the catalog kind flipped underneath it): reopen with the
    /// target schema before issuing further writes to the migrated type.
    pub fn resume_field_type_migration(&self, plan_id: u64) -> EngineResult<()> {
        self.check_not_migrated()?;
        let plan = {
            let txn = self.storage.begin_txn();
            let snap = txn.snapshot();
            crate::catalog::scan_migration_plans(&self.storage, snap)?
                .into_iter()
                .find(|p| p.plan_id == plan_id)
                .ok_or(EngineError::MigrationPlanNotFound { plan_id })?
        };
        if !plan.status.quiesces() {
            return Ok(()); // already settled
        }
        let type_id = *self.type_ids.get(&plan.type_name).ok_or_else(|| {
            EngineError::TypeNotFound(plan.type_name.clone())
        })?;
        // The converter is required ONLY to backfill in the Converting phase. A
        // plan that crashed mid-cutover (CuttingOver) resumes as a pure rename
        // pass and needs no converter — don't force the operator to re-register
        // one. (drive_migration_to_completion re-checks this for the Converting
        // branch, but failing here avoids arming for a plan we can't drive.)
        let converter = self.resolve_converter(&plan.converter_name, plan.converter_version);
        // Card 5: a RollingBack plan (a crashed terminal cancel) resumes as a
        // pure strip pass — no converter, validated against the SOURCE kind. Arm
        // the hook (a None converter is fine: the strip runs under the write lock
        // so no live write hits the rejecting hook), then drive → rollback.
        if plan.phase == crate::catalog::MigrationPhase::RollingBack {
            self.guard_resume_schema_for(&plan, plan.src_kind)?;
            {
                let _guard = self.migration_lock.write();
                self.arm_field_hook(
                    type_id,
                    MigratingFieldHook {
                        field_name: plan.field_name.clone(),
                        converter: converter.clone(),
                        target_kind: plan.target_kind,
                        converter_version: plan.converter_version,
                        plan_id,
                    },
                );
            }
            return self.drive_migration_to_completion(plan_id, type_id, converter.as_ref());
        }
        if plan.phase == crate::catalog::MigrationPhase::Converting && converter.is_none() {
            return Err(EngineError::ConverterNotRegistered {
                name: plan.converter_name.clone(),
                version: plan.converter_version,
            });
        }
        // Card 4: a dry-run preflight runs against (and stays on) the SOURCE
        // schema and never flips the catalog kind, so the F3 target-schema guard
        // doesn't apply; and it arms no hook. Just drive the preflight to
        // completion (→ DryRunCompleted) on this source-schema handle.
        if plan.dry_run {
            return self.drive_migration_to_completion(plan_id, type_id, converter.as_ref());
        }
        self.guard_resume_schema(&plan)?;
        {
            let _guard = self.migration_lock.write();
            // Re-arm the hook with the now-resolved converter (open may have
            // armed it REJECTING when the converter wasn't registered yet).
            self.arm_field_hook(
                type_id,
                MigratingFieldHook {
                    field_name: plan.field_name.clone(),
                    converter: converter.clone(),
                    target_kind: plan.target_kind,
                    converter_version: plan.converter_version,
                    plan_id,
                },
            );
        }
        self.drive_migration_to_completion(plan_id, type_id, converter.as_ref())
    }

    /// Snapshot every persisted migration plan (`c:P:*`), newest semantics
    /// from the durable record. Cheap observability for tests and operators.
    pub fn list_migrations(&self) -> EngineResult<Vec<MigrationSummary>> {
        let _guard = self.migration_lock.read();
        let txn = self.storage.begin_txn();
        let snap = txn.snapshot();
        let plans = crate::catalog::scan_migration_plans(&self.storage, snap)?;
        Ok(plans
            .into_iter()
            .map(|p| MigrationSummary {
                plan_id: p.plan_id,
                type_name: p.type_name,
                field_name: p.field_name,
                target_field_type: crate::catalog::scalar_type_from_kind(p.target_kind)
                    .map(FieldType::Scalar),
                status: p.status,
                cursor: p.cursor,
                objects_converted: p.objects_converted,
                chunk_size: p.chunk_size,
                converter_name: p.converter_name,
                converter_version: p.converter_version,
                created_at_ms: p.created_at_ms,
                error_count: p.error_count,
                error_policy: p.error_policy,
                dry_run: p.dry_run,
            })
            .collect())
    }

    /// Like [`list_migrations`](Self::list_migrations) but filtered (card 5).
    /// `filter.status` / `filter.type_name` are ANDed; a `None` field matches
    /// everything. Cheap — filters the in-memory snapshot.
    pub fn list_migrations_filtered(
        &self,
        filter: &MigrationFilter,
    ) -> EngineResult<Vec<MigrationSummary>> {
        let all = self.list_migrations()?;
        Ok(all
            .into_iter()
            .filter(|s| {
                filter.status.is_none_or(|st| s.status == st)
                    && filter
                        .type_name
                        .as_ref()
                        .is_none_or(|t| &s.type_name == t)
            })
            .collect())
    }

    /// Live progress + ETA for one migration plan (card 5). Aggregates the
    /// durable per-partition `c:S:` cursors (parallel plans) or the plan's own
    /// cursor (legacy single-worker), and derives a rate-based ETA from
    /// `created_at_ms`. The ETA is `None` unless the plan is `Running` with at
    /// least one converted row (a settled / not-yet-started plan has no
    /// meaningful projection).
    pub fn query_migration_progress(&self, plan_id: u64) -> EngineResult<MigrationProgress> {
        let _guard = self.migration_lock.read();
        let plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        let total_objects = plan.id_upper_bound.saturating_sub(1);
        let (objects_converted, errors, partitions) = match plan.parallel_degree {
            Some(n) => {
                let rows = crate::catalog::read_partition_progress(
                    &self.storage,
                    plan_id,
                    n,
                    plan.id_upper_bound,
                )?;
                let mut converted = 0u64;
                let mut errs = 0u64;
                let parts: Vec<PartitionProgress> = rows
                    .into_iter()
                    .map(|r| {
                        converted = converted.saturating_add(r.objects_converted);
                        errs = errs.saturating_add(r.errors);
                        PartitionProgress {
                            idx: r.idx,
                            lo: r.lo,
                            hi: r.hi,
                            cursor: r.cursor,
                            objects_converted: r.objects_converted,
                            errors: r.errors,
                            done: r.done,
                        }
                    })
                    .collect();
                (converted, errs, parts)
            }
            None => {
                // Legacy card-1/2 single-worker plan: synthesize one partition
                // over the whole range from the plan's own cursor/counters.
                let done = !plan.status.quiesces();
                let part = PartitionProgress {
                    idx: 0,
                    lo: 1,
                    hi: plan.id_upper_bound.max(1),
                    cursor: plan.cursor,
                    objects_converted: plan.objects_converted,
                    errors: plan.error_count,
                    done,
                };
                (plan.objects_converted, plan.error_count, vec![part])
            }
        };
        let now_ms = crate::catalog::now_unix_millis();
        let elapsed_ms = now_ms.saturating_sub(plan.created_at_ms);
        let running = plan.status == crate::catalog::MigrationStatus::Running;
        let (objects_per_sec, eta_unix_ms) = if running && objects_converted > 0 && elapsed_ms > 0 {
            let rate = (objects_converted as f64) * 1000.0 / (elapsed_ms as f64);
            // eta = now + remaining/rate = now + remaining*elapsed/converted.
            // Computed in f64 then clamped: an integer `remaining*elapsed` can
            // overflow u64 at extreme scale, and `saturating_mul` would then cap
            // the product at u64::MAX and divide it down to an astronomically
            // wrong (but finite) estimate. f64 keeps the ratio meaningful.
            let remaining = total_objects.saturating_sub(objects_converted) as f64;
            let projected_ms = remaining * (elapsed_ms as f64) / (objects_converted as f64);
            let eta = now_ms.saturating_add(projected_ms.min(u64::MAX as f64) as u64);
            (Some(rate), Some(eta))
        } else {
            (None, None)
        };
        Ok(MigrationProgress {
            plan_id,
            type_name: plan.type_name,
            field_name: plan.field_name,
            status: plan.status,
            phase: plan.phase,
            dry_run: plan.dry_run,
            parallel_degree: plan.parallel_degree,
            total_objects,
            objects_converted,
            errors,
            created_at_ms: plan.created_at_ms,
            now_ms,
            objects_per_sec,
            eta_unix_ms,
            partitions,
        })
    }

    /// Subscribe to a migration's live event stream (card 5). Returns an
    /// `mpsc::Receiver` that yields [`MigrationEvent`]s for `plan_id` until the
    /// receiver is dropped (which lazily deregisters it on the next publish).
    /// Best-effort + non-replayed: subscribe BEFORE starting the migration to
    /// observe the full sequence; a late subscriber misses earlier events and
    /// should poll [`query_migration_progress`](Self::query_migration_progress).
    pub fn subscribe_migration_events(
        &self,
        plan_id: u64,
    ) -> std::sync::mpsc::Receiver<MigrationEvent> {
        self.migration_events.subscribe(plan_id).1
    }

    /// Async operator entry point (card 5): start a field-type migration and
    /// return a [`MigrationHandle`] (the plan id + its immutable
    /// `created_at_ms`) immediately. Thin wrapper over
    /// [`create_field_type_migration`](Self::create_field_type_migration) (which
    /// already arms the hook + spawns the detached driver) — kept separate so
    /// the existing synchronous-completion callers (cards 1-4 tests) are
    /// untouched. The post-create load is race-free: `created_at_ms` never
    /// changes after the plan record is first written.
    pub fn start_field_type_migration_async(
        &self,
        spec: MigrationPlanSpec,
    ) -> EngineResult<MigrationHandle> {
        let plan_id = self.create_field_type_migration(spec)?;
        let created_at_ms = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?.created_at_ms
        };
        Ok(MigrationHandle {
            plan_id,
            created_at_ms,
        })
    }

    /// List a migration plan's quarantined rows (card 4) — rows whose converter
    /// failed under the `Quarantine` policy, for operator triage. NOTE: a row the
    /// double-write hook has since self-healed (a live write converted it) still
    /// appears until the cutover gate / `retry_quarantined` / `clear_quarantine`
    /// reaps its sidecar.
    pub fn list_quarantined(&self, plan_id: u64) -> EngineResult<Vec<QuarantineEntry>> {
        let _guard = self.migration_lock.read();
        let txn = self.storage.begin_txn();
        let snap = txn.snapshot();
        let prefix = rhypedb_storage::key::KeyBuilder::catalog_quarantine_plan_prefix(plan_id);
        let mut out = Vec::new();
        for (qkey, qval) in self.storage.scan_prefix_at(snap, &prefix)? {
            let object_id = u64::from_be_bytes(qkey[qkey.len() - 8..].try_into().unwrap());
            let rec = crate::catalog::decode_quarantine_record(
                &format!("c:Q:{plan_id}:{object_id}"),
                &qval,
            )?;
            out.push(QuarantineEntry {
                object_id,
                error_msg: rec.error_msg,
                errored_at_ms: rec.errored_at_ms,
                attempted_converter_name: rec.attempted_converter_name,
            });
        }
        Ok(out)
    }

    /// Re-run a (now-fixed) converter over the named quarantined rows (card 4),
    /// writing the shadow + deleting the `c:Q:` sidecar on success. `new_converter_name`
    /// must be registered at the plan's pinned converter version. Returns the count
    /// newly resolved. Does NOT auto-unblock cutover (resume re-checks the gate);
    /// does NOT change the historical `error_count`. Takes `migration_lock.write()`
    /// to serialize against the read-locked double-write hook + `run_cutover`.
    pub fn retry_quarantined(
        &self,
        plan_id: u64,
        ids: &[u64],
        new_converter_name: &str,
    ) -> EngineResult<u64> {
        self.check_not_migrated()?;
        let plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        let converter = self
            .resolve_converter(new_converter_name, plan.converter_version)
            .ok_or_else(|| EngineError::ConverterNotRegistered {
                name: new_converter_name.to_string(),
                version: plan.converter_version,
            })?;
        let type_id = *self.type_ids.get(&plan.type_name).ok_or_else(|| {
            EngineError::TypeNotFound(plan.type_name.clone())
        })?;
        let _guard = self.migration_lock.write();
        crate::catalog::retry_quarantined(&self.storage, &plan, type_id, ids, &converter)
    }

    /// Delete ALL of a plan's quarantine rows (card 4): the operator accepts the
    /// remaining quarantined rows stay source-shape, unblocking cutover (resume
    /// then leaves them source-shape). Returns the count cleared. Takes
    /// `migration_lock.write()`.
    pub fn clear_quarantine(&self, plan_id: u64) -> EngineResult<u64> {
        self.check_not_migrated()?;
        let _guard = self.migration_lock.write();
        crate::catalog::clear_quarantine(&self.storage, plan_id)
    }

    /// Create a new object of the given type.
    ///
    /// `fields` may include forward (non-inverse) relation fields whose
    /// value is an integer target id — the engine then writes the forward
    /// edge AND rev_edge as part of the same txn, with symmetric covers
    /// built from the in-memory FieldMap (no per-link `scan_prefix` for
    /// other targets, no extra commits). This collapses the historical
    /// `Type.create + link + link` 3-txn dance into one batched txn.
    /// Create an object. Untagged: delegates to
    /// [`create_with_origin`](Self::create_with_origin) with `origin = None`.
    pub fn create(&self, type_name: &str, fields: FieldMap) -> EngineResult<Object> {
        self.create_with_origin(type_name, fields, None)
    }

    /// Create an object, tagging the emitted [`ChangeEvent`] with `origin`.
    ///
    /// `origin` is an opaque, caller-owned token surfaced on
    /// [`ChangeEvent::origin`]; pass `None` for an untagged write. A subscriber
    /// that also writes in reaction to the change feed uses it — via
    /// [`SubscriptionFilter::exclude_origin`] — to skip its own writes and avoid
    /// a write → event → react → write loop.
    pub fn create_with_origin(
        &self,
        type_name: &str,
        fields: FieldMap,
        origin: Option<u64>,
    ) -> EngineResult<Object> {
        // Block under the migration write-barrier: if a rename / change
        // / run_migrations is in flight, wait until it commits so this
        // create observes the post-migration schema (and writes a
        // FieldMap whose keys match the catalog's view of the field).
        let _migration_guard = self.migration_lock.read();
        // Check catalog state FIRST — a retired type isn't in `self.schema`
        // anymore (the operator removed it), so falling through to
        // `schema.get_type` would yield `TypeNotFound` for retired
        // entities. Surfacing `TypeRetired` instead tells the operator
        // "you removed this, not typoed it."
        let type_id = self.resolve_type_id(type_name)?;
        // Card 2d: writes to a migrating type are NO LONGER quiesced — the
        // double-write hook (apply_migrating_field_hook) stamps the converted
        // shadow inline, so the write carries the migration forward. A migrating
        // field whose converter is unresolved still FAILS CLOSED inside the hook.
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);

        let mut txn = self.storage.begin_txn();
        let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
        // Single object — no cross-row uniqueness to track, but the helper
        // takes a staged set, so hand it a throwaway one.
        let mut staged_unique: HashMap<Bytes, u64> = HashMap::new();

        let scalar_fields = self.stage_create_writes(
            &mut txn, type_name, type_def, type_id, object_id, &fields, &mut puts,
            &mut staged_unique,
        )?;

        // `stage_create_writes` bumped this object's in-memory generation to 1
        // (born-at-1). If the write doesn't land, undo it so a future create
        // reusing nothing — and any cover stamping — sees a consistent counter.
        self.storage.put_batch(&mut txn, &puts).map_err(|e| {
            self.rollback_version(type_id, object_id);
            EngineError::Storage(e)
        })?;
        let version = self.storage.commit(&mut txn).map_err(|e| {
            self.rollback_version(type_id, object_id);
            match e {
                rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                other => EngineError::Storage(other),
            }
        })?;

        self.subscriptions.publish(ChangeEvent {
            version,
            kind: ChangeKind::Create,
            type_name: type_name.into(),
            object_id,
            fields: Some(fields_to_json(&scalar_fields)),
            origin,
        });

        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields: scalar_fields,
            raw_fields: None,
        })
    }

    /// Stage all writes for one create-with-inline-relations row.
    ///
    /// Splits `fields` into scalar fields (which form the object payload)
    /// and relation fields (which expand into forward edge + rev_edge
    /// writes). For every forward 1:1 relation being set in this row we
    /// fetch the target's serialized data so the rev_edge's `<name>__cover`
    /// can be populated symmetrically — both sides of a pair of 1:1 links
    /// land with full covers, instead of the historical sequence-dependent
    /// "second link gets a cover, first stays empty" pattern. Cover_v
    /// stamps are read from the in-memory `version_counters` map (a live
    /// never-updated target reads 1; an absent/deleted target reads 0).
    ///
    /// Unique-index puts are issued inline (not buffered into `puts`) so they
    /// commit atomically with the rest of the txn; intra-batch duplicate values
    /// are caught via `staged`, which the caller threads across every row of a
    /// `create_batch` (a buffered put can't be seen by a later row's
    /// `storage.get`). All other writes accumulate into `puts` for the caller to
    /// flush via `put_batch`. Returns the scalar-only `FieldMap` for the response.
    #[allow(clippy::too_many_arguments)]
    fn stage_create_writes(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        type_def: &rhypedb_schema::TypeDef,
        type_id: u64,
        object_id: u64,
        fields: &FieldMap,
        puts: &mut Vec<(Bytes, Bytes)>,
        staged: &mut HashMap<Bytes, u64>,
    ) -> EngineResult<FieldMap> {
        // First pass: validate, split scalars from relations.
        let mut scalar_fields = FieldMap::new();
        // (field_name, target_id, target_type, target_data, target_version, rel_id, is_1to1_forward)
        let mut links: Vec<(String, u64, String, Bytes, u64, u64, bool)> = Vec::new();

        for (field_name, value) in fields {
            let field_def =
                type_def
                    .get_field(field_name)
                    .ok_or_else(|| EngineError::FieldNotFound {
                        type_name: type_name.into(),
                        field: field_name.clone(),
                    })?;
            validate_value(field_def, value)?;

            match &field_def.field_type {
                FieldType::Relation(rel) => {
                    if field_def.inverse().is_some() {
                        // Inverse relations are virtual — they don't have
                        // their own forward edges, so a value here would
                        // have no place to land.
                        return Err(EngineError::TypeMismatch {
                            field: field_def.name.clone(),
                            expected: "scalar (inverse fields are virtual)".into(),
                            got: value.type_name().into(),
                        });
                    }
                    if matches!(value, Value::Null) {
                        continue;
                    }
                    let target_id: u64 = match value {
                        Value::U64(v) => *v,
                        Value::U32(v) => *v as u64,
                        Value::I32(v) if *v >= 0 => *v as u64,
                        Value::I64(v) if *v >= 0 => *v as u64,
                        _ => {
                            // validate_value already rejected non-integer
                            // values for Relation; this arm only fires on
                            // negative signed integers.
                            return Err(EngineError::TypeMismatch {
                                field: field_def.name.clone(),
                                expected: "relation target id (non-negative integer)".into(),
                                got: value.type_name().into(),
                            });
                        }
                    };
                    let target_type_id = *self
                        .type_ids
                        .get(&rel.target_type)
                        .ok_or_else(|| EngineError::TypeNotFound(rel.target_type.clone()))?;
                    let target_key = KeyBuilder::object(target_type_id, target_id);
                    let target_data = self.storage.get(txn, &target_key)?.ok_or_else(|| {
                        EngineError::ObjectNotFound {
                            type_name: rel.target_type.clone(),
                            object_id: target_id,
                        }
                    })?;
                    let target_version = self.object_version(&rel.target_type, target_id);
                    let rel_key = format!("{type_name}.{field_name}");
                    let rel_id =
                        *self
                            .rel_ids
                            .get(&rel_key)
                            .ok_or_else(|| EngineError::FieldNotFound {
                                type_name: type_name.into(),
                                field: field_name.clone(),
                            })?;
                    let is_1to1_forward = !rel.is_many;
                    links.push((
                        field_name.clone(),
                        target_id,
                        rel.target_type.clone(),
                        target_data,
                        target_version,
                        rel_id,
                        is_1to1_forward,
                    ));
                }
                FieldType::Scalar(_) => {
                    scalar_fields.insert(field_name.clone(), value.clone());
                }
                FieldType::Vector(_) => {
                    return Err(EngineError::TypeMismatch {
                        field: field_def.name.clone(),
                        expected: "scalar or relation".into(),
                        got: value.type_name().into(),
                    });
                }
            }
        }

        // Card 2: double-write the shadow for any field mid-migration into the
        // SERIALIZED blob (shared by the object entry AND every covering-index
        // entry below). Apply it to a clone so `scalar_fields` itself stays
        // shadow-free — the unique-index loop, the in-memory covers, the change
        // event, and the returned Object all iterate `scalar_fields` and must
        // see only real fields. Cloned only while a migration is in flight.
        let serialized = if self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            let mut with_shadow = scalar_fields.clone();
            self.apply_migrating_field_hook(type_id, type_name, object_id, &mut with_shadow)?;
            serialize_fields(&with_shadow)
        } else {
            serialize_fields(&scalar_fields)
        };

        // Unique-index writes for scalar fields. Issued inline (committed with
        // the txn); `staged` carries claimed values across rows so a duplicate
        // later in the same `create_batch` is rejected.
        for (field_name, value) in &scalar_fields {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() && !matches!(value, Value::Null) {
                self.check_unique_and_insert(
                    txn, type_name, type_id, field_name, value, object_id, staged,
                )?;
            }
        }

        // Secondary index entries — covering payload = serialized scalars.
        if let Some(idx_fields) = self.indexed_fields.get(type_name) {
            for ifd in idx_fields {
                if let Some(value) = scalar_fields.get(&ifd.name)
                    && !matches!(value, Value::Null)
                    && let Some(key) =
                        build_field_index_key(type_id, ifd.field_id, ifd.kind, value, object_id)
                {
                    puts.push((key, serialized.clone()));
                }
            }
        }

        // Object key.
        puts.push((KeyBuilder::object(type_id, object_id), serialized.clone()));

        // Forward edges + rev_edges with in-memory-built covers. The set
        // of "other forward-1:1 targets" is just the other entries in
        // `links` (no LSM scan_prefix). Symmetric: every rev_edge for this
        // row carries the full peer set, instead of the historical
        // "first link empty, second link covers" asymmetry.
        for (i, link) in links.iter().enumerate() {
            let (field_name, target_id, _, _, _, rel_id, _) = link;
            puts.push((
                KeyBuilder::edge(object_id, *rel_id, *target_id),
                Bytes::new(),
            ));

            let rev_value =
                build_inflight_cover(self, txn, &scalar_fields, field_name, *target_id, &links, i)?;
            puts.push((
                KeyBuilder::reverse_edge(*target_id, *rel_id, object_id),
                rev_value,
            ));
        }

        // Born at generation 1 — IN MEMORY ONLY, no persisted `g:` key.
        // Generation 0 means "absent": a live, never-updated object reads
        // version >= 1, while `delete` (via `forget_version`) drops it back to
        // 0, so the fusion reader's `cover_v == object_version && != 0` check
        // rejects a cover for a deleted target (its `cover_v`, taken while
        // alive, can never equal the post-delete live version 0) — closing the
        // never-updated-then-deleted phantom.
        //
        // The born bit is NOT persisted here: an object's existence is already
        // on disk as its `o:` key, and `open()` reconstructs generation 1 for
        // every live object from the `o:*` scan it already runs for
        // `next_object_id` (see `rebuild_with_arc_storage`). A `g:` key is
        // written only when an object is UPDATED (generation >= 2), where the
        // higher value can't be derived from existence alone — so creates and
        // never-updated deletes carry no generation write-amp. Done last so a
        // staging error above leaves no in-memory bump to undo; the caller
        // rolls this back if the commit doesn't land.
        self.bump_version(type_id, object_id);

        Ok(scalar_fields)
    }

    /// Bulk-create N objects in ONE transaction (one WAL append + commit).
    /// Schema/type lookup is amortized; per-row work mirrors `create` exactly
    /// (validate → build key → serialize → write unique-index entries → put).
    /// Subscription events fire after commit, one per object.
    ///
    /// If any row fails (validation, unique violation, write conflict), the
    /// whole batch rolls back — none of the rows land. This is intentional:
    /// callers reaching for the bulk path want the all-or-nothing shape of
    /// `COPY ... FROM STDIN`, not a partial insert.
    pub fn create_batch(&self, type_name: &str, rows: Vec<FieldMap>) -> EngineResult<Vec<Object>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let _migration_guard = self.migration_lock.read();
        let type_id = self.resolve_type_id(type_name)?;
        // Card 2d: no quiesce — the per-row double-write hook in
        // stage_create_writes carries each migrating field forward.
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

        let mut txn = self.storage.begin_txn();
        let mut object_ids: Vec<u64> = Vec::with_capacity(rows.len());
        // Every put across the whole batch — object payload, secondary
        // index entries, forward + rev edges — accumulates here and
        // flushes via one `put_batch` at the end. Unique-index puts are
        // issued inline (committed with the txn). Per-row scalar FieldMaps
        // are reconstructed post-batch for the published events / returned
        // Objects.
        let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(rows.len() * 2);
        let mut scalar_rows: Vec<FieldMap> = Vec::with_capacity(rows.len());
        // Tracks the @unique values claimed so far in THIS batch. A buffered
        // unique-index put is invisible to a later row's `storage.get` (reads
        // resolve at the txn snapshot), so without this two rows sharing a
        // `@unique` value would both commit. Threaded into every row.
        let mut staged_unique: HashMap<Bytes, u64> = HashMap::new();

        for fields in &rows {
            let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);
            match self.stage_create_writes(
                &mut txn, type_name, type_def, type_id, object_id, fields, &mut puts,
                &mut staged_unique,
            ) {
                Ok(scalar_fields) => {
                    // stage_create_writes bumped this object to generation 1.
                    scalar_rows.push(scalar_fields);
                    object_ids.push(object_id);
                }
                Err(e) => {
                    // This row never bumped (bump is the last step), but every
                    // successfully-staged earlier row did — undo them all.
                    for id in &object_ids {
                        self.rollback_version(type_id, *id);
                    }
                    return Err(e);
                }
            }
        }

        // born-at-1 rollback on a failed batch — none of the rows land, so no
        // in-memory generation should survive either.
        let rollback_all = |db: &Self| {
            for id in &object_ids {
                db.rollback_version(type_id, *id);
            }
        };
        if let Err(e) = self.storage.put_batch(&mut txn, &puts) {
            rollback_all(self);
            return Err(EngineError::Storage(e));
        }
        let version = match self.storage.commit(&mut txn) {
            Ok(v) => v,
            Err(e) => {
                rollback_all(self);
                return Err(match e {
                    rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                    other => EngineError::Storage(other),
                });
            }
        };

        // Build the returned Objects + publish events after commit.
        // Events report only the scalar fields (relation values went into
        // the edge index, not the object payload).
        let mut out = Vec::with_capacity(scalar_rows.len());
        for (id, scalar_fields) in object_ids.into_iter().zip(scalar_rows) {
            self.subscriptions.publish(ChangeEvent {
                version,
                kind: ChangeKind::Create,
                type_name: type_name.into(),
                object_id: id,
                fields: Some(fields_to_json(&scalar_fields)),
                origin: None,
            });
            out.push(Object {
                type_name: type_name.into(),
                id,
                fields: scalar_fields,
                raw_fields: None,
            });
        }
        Ok(out)
    }

    /// Bulk-insert objects with CALLER-SUPPLIED ids, preserving them exactly —
    /// the import counterpart of [`create_batch`](Self::create_batch), which
    /// instead assigns ids from `next_object_id`. Reuses `stage_create_writes`,
    /// so @unique / @indexed covering entries (and any inline-relation edges)
    /// are rebuilt identically, then advances `next_object_id` PAST the highest
    /// restored id via `fetch_max` — the same monotonic invariant `open()`
    /// reconstructs from the `o:*` scan — so a later `create` can never reuse a
    /// restored id. Because ids are preserved, edges and vectors import verbatim
    /// with no remap table.
    ///
    /// All-or-nothing per call (one txn, one commit), like `create_batch`: any
    /// row that fails validation / @unique rolls the whole call back. Each
    /// `FieldMap` is the object's SCALAR fields (a logical import recreates
    /// relations separately as edges, after every object exists); a relation
    /// field present here is staged inline and so requires its target to exist.
    pub fn restore_objects(
        &self,
        type_name: &str,
        rows: Vec<(u64, FieldMap)>,
        reject_existing: bool,
    ) -> EngineResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let _migration_guard = self.migration_lock.read();
        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

        let mut txn = self.storage.begin_txn();
        let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(rows.len() * 2);
        let mut staged_unique: HashMap<Bytes, u64> = HashMap::new();
        let mut staged: Vec<(u64, FieldMap)> = Vec::with_capacity(rows.len());
        let mut max_id = 0u64;

        for (object_id, fields) in &rows {
            // Additive (online) restore: refuse an id that already exists rather
            // than overwriting it — restore_objects is an insert path and would
            // leave the prior object's unique/index/edge entries stale. A
            // deleted id reads as absent here, so it is reusable.
            if reject_existing {
                let key = KeyBuilder::object(type_id, *object_id);
                if self.storage.get(&txn, &key).map_err(EngineError::Storage)?.is_some() {
                    for (id, _) in &staged {
                        self.rollback_version(type_id, *id);
                    }
                    return Err(EngineError::RestoreObjectExists {
                        type_name: type_name.into(),
                        object_id: *object_id,
                    });
                }
            }
            match self.stage_create_writes(
                &mut txn, type_name, type_def, type_id, *object_id, fields, &mut puts,
                &mut staged_unique,
            ) {
                Ok(scalar_fields) => {
                    max_id = max_id.max(*object_id);
                    staged.push((*object_id, scalar_fields));
                }
                Err(e) => {
                    // This row never bumped (bump is the last step); undo the
                    // earlier rows' in-memory generation bumps.
                    for (id, _) in &staged {
                        self.rollback_version(type_id, *id);
                    }
                    return Err(e);
                }
            }
        }

        let rollback_all = |db: &Self| {
            for (id, _) in &staged {
                db.rollback_version(type_id, *id);
            }
        };
        if let Err(e) = self.storage.put_batch(&mut txn, &puts) {
            rollback_all(self);
            return Err(EngineError::Storage(e));
        }
        let version = match self.storage.commit(&mut txn) {
            Ok(v) => v,
            Err(e) => {
                rollback_all(self);
                return Err(match e {
                    rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                    other => EngineError::Storage(other),
                });
            }
        };

        // Advance the id counter past every restored id so a future create
        // cannot collide. fetch_max is monotonic, so chunked calls compose.
        self.next_object_id.fetch_max(max_id + 1, Ordering::SeqCst);

        // Publish create events (one per object), mirroring create_batch.
        for (object_id, scalar_fields) in &staged {
            self.subscriptions.publish(ChangeEvent {
                version,
                kind: ChangeKind::Create,
                type_name: type_name.into(),
                object_id: *object_id,
                fields: Some(fields_to_json(scalar_fields)),
                origin: None,
            });
        }
        Ok(())
    }

    /// Bulk-write raw vector values for a `Vector` field, preserving object ids.
    /// Each `bytes` is the verbatim `v:` payload (big-endian f32, exactly what
    /// the export shipped), so this is a lossless restore.
    ///
    /// It writes ONLY the `v:<type_id>:<object_id>:<field_id>` source-of-truth
    /// keys — NOT the HNSW graph. The graph is rebuilt from these keys on the
    /// next open ONLY when the `.bin` snapshot is absent or its config mismatches
    /// (a full `rebuild_index_from_lsm`); against a warm dir whose snapshot
    /// already holds these ids the open takes the delta path and SKIPS ids
    /// already in the graph, so an OVERWRITE would not reach the graph. Callers
    /// overwriting vectors must therefore clear `hnsw_*.bin` first — the offline
    /// import does (it stages into a fresh dir). So a logical import reconstructs
    /// vector search with no vectorizer/embedder at import time; search comes
    /// back recall-equivalent after the rebuild.
    ///
    /// Each payload's length must be `dims * 4` for the field's declared
    /// `Vector<dims>`. All-or-nothing per call (one txn / commit).
    pub fn restore_vectors(
        &self,
        type_name: &str,
        field_name: &str,
        rows: &[(u64, Bytes)],
    ) -> EngineResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let _migration_guard = self.migration_lock.read();
        let (type_id, field_id) = self.resolve_field_id(type_name, field_name)?;

        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let field_def = type_def
            .get_field(field_name)
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: field_name.into(),
            })?;
        let expected_len = match &field_def.field_type {
            FieldType::Vector(v) => v.dimensions as usize * 4,
            _ => {
                return Err(EngineError::TypeMismatch {
                    field: field_name.into(),
                    expected: "Vector field".into(),
                    got: "non-vector field".into(),
                });
            }
        };

        let mut txn = self.storage.begin_txn();
        let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(rows.len());
        for (object_id, bytes) in rows {
            if bytes.len() != expected_len {
                return Err(EngineError::TypeMismatch {
                    field: field_name.into(),
                    expected: format!("{expected_len} bytes (dims*4)"),
                    got: format!("{} bytes", bytes.len()),
                });
            }
            puts.push((KeyBuilder::vector(type_id, *object_id, field_id), bytes.clone()));
        }
        self.storage
            .put_batch(&mut txn, &puts)
            .map_err(EngineError::Storage)?;
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;
        Ok(())
    }

    /// Get an object by type and ID.
    pub fn get(&self, type_name: &str, object_id: u64) -> EngineResult<Object> {
        let type_id = self.resolve_type_id(type_name)?;

        let key = KeyBuilder::object(type_id, object_id);
        let snapshot = self.storage.read_snapshot();
        let data =
            self.storage
                .get_at(snapshot, &key)?
                .ok_or_else(|| EngineError::ObjectNotFound {
                    type_name: type_name.into(),
                    object_id,
                })?;

        let mut fields = deserialize_fields(&data);
        self.strip_tombstoned_fields(type_name, &mut fields);
        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields,
            raw_fields: None,
        })
    }

    /// Batch point lookup for N objects of one type. Acquires the storage
    /// memtable/SST locks ONCE for the whole batch (via `multi_get_at`), and
    /// probes IDs in sorted order so SST sparse-index entries are touched
    /// monotonically (better cache locality).
    ///
    /// Skips IDs that don't exist instead of erroring — at the terminal hop
    /// of a traversal, a missing target is "the edge pointed to a deleted
    /// object" and should drop silently, not abort the whole query.
    pub fn get_many(&self, type_name: &str, ids: &[u64]) -> EngineResult<Vec<Object>> {
        self.get_many_at(self.storage.read_snapshot(), type_name, ids)
    }

    /// [`get_many`](Self::get_many) reading at a CALLER-supplied snapshot. Used
    /// by the `filter_scan_*_at` legacy non-covering fallback so the per-id
    /// materialization shares the scan's snapshot — keeping a planner
    /// intersection / union a single point-in-time view even on pre-covering
    /// index data.
    pub fn get_many_at(
        &self,
        snapshot: u64,
        type_name: &str,
        ids: &[u64],
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;

        // Sort + dedup. Streaming traversal already dedups across hops;
        // this is the belt-and-suspenders pass for direct external callers.
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        // Build key buffers once, borrow them as &[u8] for the batch call.
        let key_bufs: Vec<_> = sorted
            .iter()
            .map(|id| KeyBuilder::object(type_id, *id))
            .collect();
        let key_refs: Vec<&[u8]> = key_bufs.iter().map(|k| k.as_ref()).collect();

        let values = self.storage.multi_get_at(snapshot, &key_refs)?;

        // Public API: return Objects with `fields` populated. Callers that
        // accept the lazy shortcut should use `get_many_lazy` instead.
        let mut out = Vec::with_capacity(sorted.len());
        for (id, value) in sorted.into_iter().zip(values) {
            if let Some(data) = value {
                let mut fields = deserialize_fields(&data);
                self.strip_tombstoned_fields(type_name, &mut fields);
                out.push(Object {
                    type_name: type_name.into(),
                    id,
                    fields,
                    raw_fields: None,
                });
            }
        }
        Ok(out)
    }

    /// Lazy variant of `get_many`: returns Objects with `raw_fields = Some(bytes)`
    /// and `fields` empty. The wire encoder (`encode_object`) can ship the
    /// stored payload directly — no `deserialize_fields` + HashMap + drop
    /// cycle. Consumers that read `obj.fields` (predicates, vectorize hook,
    /// HTTP/JSON path) must call `ensure_fields_deserialized` first.
    ///
    /// Used by the executor's terminal materialize, where Objects flow
    /// straight from the LSM to the TCP response without intermediate
    /// inspection. Saves ~50% of per-object materialize cost at 50+ objects
    /// per query (2-hop traversal terminal, filter scan covering path).
    pub fn get_many_lazy(&self, type_name: &str, ids: &[u64]) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;

        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let key_bufs: Vec<_> = sorted
            .iter()
            .map(|id| KeyBuilder::object(type_id, *id))
            .collect();
        let key_refs: Vec<&[u8]> = key_bufs.iter().map(|k| k.as_ref()).collect();

        let snapshot = self.storage.read_snapshot();
        let values = self.storage.multi_get_at(snapshot, &key_refs)?;

        // While a field-type migration is in flight, the `o:` blob carries
        // `<field>__shadow` siblings the wire path must never ship. The lazy
        // path ships `raw_fields` verbatim (protocol.rs `encode_object`), so it
        // can't rely on the eager `strip_tombstoned_fields` chokepoint —
        // deserialize + strip + fall back to an eager Object here. Paid only
        // during a migration; the zero-copy fast path is untouched otherwise.
        let migrating = self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0;

        let mut out = Vec::with_capacity(sorted.len());
        for (id, value) in sorted.into_iter().zip(values) {
            if let Some(data) = value {
                if migrating {
                    let mut fields = deserialize_fields(&data);
                    self.strip_tombstoned_fields(type_name, &mut fields);
                    out.push(Object {
                        type_name: type_name.into(),
                        id,
                        fields,
                        raw_fields: None,
                    });
                } else {
                    out.push(Object::from_raw(type_name.into(), id, data));
                }
            }
        }
        Ok(out)
    }

    /// Scan all objects of a given type. Uses the LSM prefix scan on the
    /// object key prefix, so this is a real index scan — not a brute-force probe.
    pub fn scan_type(&self, type_name: &str) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;

        let prefix = KeyBuilder::object_prefix(type_id);
        let snapshot = self.storage.read_snapshot();
        let entries = self.storage.scan_prefix_at(snapshot, &prefix)?;

        let mut objects = Vec::new();
        for (key, data) in entries {
            // Object key: o:<type_id>:<object_id> — extract object_id from last 8 bytes.
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);

            let mut fields = deserialize_fields(&data);
            self.strip_tombstoned_fields(type_name, &mut fields);
            objects.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields,
                raw_fields: None,
            });
        }

        Ok(objects)
    }

    /// Tombstone-CORRECT live count of one type's objects (a count-only keyspace
    /// scan over `o:<type_id>:` that retains only a liveness bool per key, not the
    /// object payloads). Used by the query governor to refuse an over-budget
    /// unindexed scan BEFORE materializing any objects — much lighter than the
    /// object set, and (unlike a *limited* scan) never under-counts across leading
    /// tombstones.
    pub fn count_type(&self, type_name: &str) -> EngineResult<u64> {
        let type_id = self.resolve_type_id(type_name)?;
        let prefix = KeyBuilder::object_prefix(type_id);
        let snapshot = self.storage.read_snapshot();
        Ok(self.storage.count_prefix_at(snapshot, &prefix)?)
    }

    /// Metering: total live objects across ALL types — a count-only keyspace scan
    /// (no field deserialize, no value retention). O(n) in object count; for the
    /// infrequently-polled `/status` metering counters. The host cannot compute
    /// this without the engine (it must never parse the guest keyspace).
    pub fn count_objects(&self) -> EngineResult<u64> {
        let snapshot = self.storage.read_snapshot();
        let prefix = KeyBuilder::all_objects_prefix();
        Ok(self.storage.count_prefix_at(snapshot, &prefix)?)
    }

    /// Metering: total live forward edges (relationship links) across all
    /// relationships. Reverse-edge index entries (`r:`) are excluded, so each link
    /// counts once. Same cost profile as [`count_objects`](Self::count_objects).
    pub fn count_edges(&self) -> EngineResult<u64> {
        let snapshot = self.storage.read_snapshot();
        let prefix = KeyBuilder::all_edges_prefix();
        Ok(self.storage.count_prefix_at(snapshot, &prefix)?)
    }

    /// Column-projected point lookup: like [`get`](Self::get) but deserializes
    /// only `fields` instead of the whole object, skipping the `String`-key +
    /// `Value` allocations for every other field. The returned Object's
    /// `fields` holds exactly the requested-and-present columns.
    ///
    /// A building block for predicate pushdown and intermediate-hop projection;
    /// callers that need the full object should use `get`.
    pub fn get_projected(
        &self,
        type_name: &str,
        object_id: u64,
        fields: &[&str],
    ) -> EngineResult<Object> {
        let type_id = self.resolve_type_id(type_name)?;

        let key = KeyBuilder::object(type_id, object_id);
        let snapshot = self.storage.read_snapshot();
        let data =
            self.storage
                .get_at(snapshot, &key)?
                .ok_or_else(|| EngineError::ObjectNotFound {
                    type_name: type_name.into(),
                    object_id,
                })?;

        let mut projected = deserialize_fields_projected(&data, fields);
        self.strip_tombstoned_fields(type_name, &mut projected);
        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields: projected,
            raw_fields: None,
        })
    }

    /// Column-projected type scan: like [`scan_type`](Self::scan_type) but
    /// deserializes only `fields` per object. The win scales with the ratio of
    /// skipped columns to kept ones — a wide type scanned for two columns pays
    /// a fraction of the full-deserialize cost.
    pub fn scan_type_projected(
        &self,
        type_name: &str,
        fields: &[&str],
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;

        let prefix = KeyBuilder::object_prefix(type_id);
        let snapshot = self.storage.read_snapshot();
        let entries = self.storage.scan_prefix_at(snapshot, &prefix)?;

        let mut objects = Vec::with_capacity(entries.len());
        for (key, data) in entries {
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);

            let mut projected = deserialize_fields_projected(&data, fields);
            self.strip_tombstoned_fields(type_name, &mut projected);
            objects.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields: projected,
                raw_fields: None,
            });
        }

        Ok(objects)
    }

    /// One bounded, OOM-safe step of an ascending scan over every live object
    /// of `type_name`, visible at the caller-pinned `snapshot`.
    ///
    /// Unlike [`scan_type`](Self::scan_type) — which materializes the whole
    /// type in memory — this walks the `o:<type_id>:` keyspace in chunks of at
    /// most `max_distinct` distinct keys, so a caller (logical export, bulk
    /// re-index) can stream an arbitrarily large type with bounded peak memory.
    ///
    /// Resume protocol: start with `cursor == 0`; each call returns
    /// [`ObjectChunk::next_cursor`] (the highest object id visited this chunk,
    /// tombstones included) — pass it back verbatim as the next `cursor`. Stop
    /// when [`ObjectChunk::more`] is `false` (or `next_cursor` is `None`).
    /// NEVER infer end-of-range from a short `objects` vec: a chunk may straddle
    /// a run of tombstoned object ids, returning few (or zero) live objects
    /// while more live objects remain past `next_cursor`.
    ///
    /// Pin `snapshot` once via [`read_snapshot`](rhypedb_storage::lsm::LsmTree::read_snapshot)
    /// across the whole walk so every chunk sees one MVCC point-in-time view.
    pub fn scan_chunk(
        &self,
        type_name: &str,
        snapshot: u64,
        cursor: u64,
        max_distinct: usize,
    ) -> EngineResult<ObjectChunk> {
        let type_id = self.resolve_type_id(type_name)?;
        let prefix = KeyBuilder::object_prefix(type_id);

        // `cursor == 0` starts at the prefix; otherwise seek strictly after the
        // last visited object id. A cursor at u64::MAX has no successor key, so
        // the range is already exhausted (matches the migration driver).
        let start = if cursor == 0 {
            prefix.clone()
        } else {
            match cursor.checked_add(1) {
                Some(next) => KeyBuilder::object(type_id, next),
                None => {
                    return Ok(ObjectChunk {
                        objects: Vec::new(),
                        next_cursor: None,
                        more: false,
                    });
                }
            }
        };

        let chunk = self.storage.scan_chunk_raw(snapshot, &prefix, &start, max_distinct)?;

        // Object key is o:<type_id>:<object_id>; the object id is the trailing
        // 8 bytes. high_water lives in the same keyspace, so its tail decodes
        // to the highest id visited (live or tombstoned).
        let next_cursor = chunk
            .high_water
            .as_ref()
            .map(|hw| u64::from_be_bytes(hw[hw.len() - 8..].try_into().unwrap()));

        let mut objects = Vec::with_capacity(chunk.live.len());
        for (key, data) in &chunk.live {
            if key.len() < 8 {
                continue;
            }
            let object_id = u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap());
            let mut fields = deserialize_fields(data);
            self.strip_tombstoned_fields(type_name, &mut fields);
            objects.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields,
                raw_fields: None,
            });
        }

        Ok(ObjectChunk {
            objects,
            next_cursor,
            more: chunk.more,
        })
    }

    /// True if `type_name.field_name` has a secondary (`@indexed`) index — i.e.
    /// a `filter_scan` on it takes the real index fast path (`filter_scan_via_index`)
    /// rather than the zone-map-pruned full prefix scan. This reads the same
    /// `indexed_fields` map `filter_scan` consults, so it is authoritative.
    ///
    /// The query planner uses it to decide whether pushing a predicate conjunct
    /// down yields a genuinely sublinear scan (worth doing) versus a full scan
    /// in disguise (better left to the in-memory residual filter).
    pub fn is_field_indexed(&self, type_name: &str, field_name: &str) -> bool {
        self.indexed_fields
            .get(type_name)
            .is_some_and(|fields| fields.iter().any(|f| f.name == field_name))
    }

    /// A read snapshot handle (MVCC version) for the current committed state.
    /// Pass it to the `*_at` read methods so a sequence of reads observes ONE
    /// point-in-time view — e.g. the query planner pins one per filter and
    /// threads it through every scan in a multi-index intersection / union.
    pub fn read_snapshot(&self) -> u64 {
        self.storage.read_snapshot()
    }

    /// True if `type_name.field_name` carries a `@unique` constraint — i.e. an
    /// exact `field == value` lookup can be served by [`find_by_unique`](Self::find_by_unique)
    /// (a single `u:` point read yielding ≤1 object) rather than an index/zone
    /// scan. Reads the schema, the same source the `u:` writer consults.
    pub fn is_field_unique(&self, type_name: &str, field_name: &str) -> bool {
        self.schema
            .get_type(type_name)
            .and_then(|td| td.get_field(field_name))
            .is_some_and(|fd| fd.is_unique())
    }

    /// Exact lookup of the one object whose `@unique` `field_name` equals
    /// `value`, or `None` when no row has that value (or the field isn't unique
    /// / doesn't exist). See [`find_by_unique_at`](Self::find_by_unique_at).
    pub fn find_by_unique(
        &self,
        type_name: &str,
        field_name: &str,
        value: &Value,
    ) -> EngineResult<Option<Object>> {
        self.find_by_unique_at(self.storage.read_snapshot(), type_name, field_name, value)
    }

    /// [`find_by_unique`](Self::find_by_unique) at a CALLER-supplied snapshot, so
    /// a planner OR-union mixing a unique probe with index scans reads them all
    /// at one point-in-time. Reads the `u:<type>:<field>:<value>` entry for the
    /// object id, then the object — both under the SAME snapshot. Shadow/retired
    /// fields are stripped as on any other read.
    ///
    /// The query planner uses this as its most selective access path: an
    /// equality on a unique field yields at most one candidate.
    pub fn find_by_unique_at(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        value: &Value,
    ) -> EngineResult<Option<Object>> {
        let type_id = self.resolve_type_id(type_name)?;
        let field_key = format!("{type_name}.{field_name}");
        let Some(&field_id) = self.field_ids.get(&field_key) else {
            return Ok(None);
        };
        let value_bytes = value_to_index_bytes(value);
        let unique_key = KeyBuilder::unique_index(type_id, field_id, &value_bytes);
        let Some(id_bytes) = self.storage.get_at(snapshot, &unique_key)? else {
            return Ok(None);
        };
        if id_bytes.len() < 8 {
            return Ok(None);
        }
        let object_id = u64::from_be_bytes(id_bytes[..8].try_into().unwrap());
        let obj_key = KeyBuilder::object(type_id, object_id);
        let Some(data) = self.storage.get_at(snapshot, &obj_key)? else {
            // Dangling `u:` entry (object gone) — can't happen within one
            // snapshot since both are written atomically, but stay defensive.
            return Ok(None);
        };
        let mut fields = deserialize_fields(&data);
        self.strip_tombstoned_fields(type_name, &mut fields);
        Ok(Some(Object {
            type_name: type_name.into(),
            id: object_id,
            fields,
            raw_fields: None,
        }))
    }

    /// Filtered scan: pushes a single-field integer comparison down to storage.
    ///
    /// Two fast paths are layered:
    ///   1. **Secondary index (`@indexed` field)** — prefix scan on the
    ///      `i:<type>:<field>:` key range yields `(encoded_value, id)` pairs
    ///      directly from the key. For `Eq` we further narrow to the
    ///      value-specific prefix; for ranges we filter encoded_values from
    ///      the field prefix. No object decode happens for non-matching ids.
    ///   2. **Zone-map fallback** — the field isn't indexed. Walks the
    ///      object-key prefix and skips SST blocks whose per-field min/max
    ///      bounds rule out the predicate, then re-evaluates per entry.
    ///
    /// `target` is the raw query-level integer; this method looks up the
    /// field's schema type (U32 / U64 / I32 / I64) and re-encodes the target
    /// to match the on-disk byte-order encoding. Out-of-range targets (e.g.,
    /// negative literal against a U32 field with `<` op) fall back to a
    /// `scan_type` for safety.
    ///
    /// `limit` is best-effort early termination: once `limit` ids match, the
    /// per-entry walk stops. Storage still returns the full prefix-scan
    /// result set; the win is skipping object materialization beyond the limit.
    ///
    /// Returns `Err(FieldNotFound)` for unknown fields and falls back to
    /// `scan_type` for non-integer field types.
    pub fn filter_scan(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: i64,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        self.filter_scan_at(
            self.storage.read_snapshot(),
            type_name,
            field_name,
            op,
            target,
            limit,
        )
    }

    /// [`filter_scan`](Self::filter_scan) reading at a CALLER-supplied snapshot
    /// instead of taking its own. The query planner pins one snapshot per filter
    /// and threads it through every scan in a multi-index intersection / union,
    /// so the combined result is a single point-in-time view (rather than a blend
    /// of several snapshots).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_scan_at(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: i64,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_storage::zone::FieldPredicate;

        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let field_def = type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(type_name, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })
        })?;

        // Cast the raw query target to the field's actual integer type so the
        // encoding matches what's on disk. Bail out to `scan_type` (no perf
        // gain, but correct) for non-integer scalar fields.
        let target_value = match &field_def.field_type {
            FieldType::Scalar(ScalarType::U32) if (0..=u32::MAX as i64).contains(&target) => {
                Value::U32(target as u32)
            }
            FieldType::Scalar(ScalarType::U64) if target >= 0 => Value::U64(target as u64),
            FieldType::Scalar(ScalarType::I32)
                if (i32::MIN as i64..=i32::MAX as i64).contains(&target) =>
            {
                Value::I32(target as i32)
            }
            FieldType::Scalar(ScalarType::I64) => Value::I64(target),
            // DateTime is i64 epoch-millis; the caller (executor) has already
            // coerced an RFC 3339 / int literal to millis, so `target` IS the
            // millis to compare. Encodes like I64 via `encode_int_for_zone`.
            FieldType::Scalar(ScalarType::DateTime) => Value::DateTime(target),
            // Non-integer field, or an integer field whose declared type can't
            // represent `target` (out of range). The typed index/zone fast
            // path doesn't apply — compare per row so the answer stays correct
            // (and empty for non-numeric fields) instead of returning the whole
            // table.
            _ => {
                return self.filter_scan_fallback(snapshot, type_name, field_name, limit, |v| match v {
                    Value::U32(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::U64(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::I32(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::I64(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    // Defensive: a DateTime value can only reach here under a
                    // non-DateTime declared field (impossible via the schema),
                    // but compare it as its i64 millis to keep parity rather
                    // than silently dropping it.
                    Value::DateTime(ms) => Some(compare_partial(*ms as i128, op, target as i128)),
                    Value::F64(f) => Some(compare_partial(*f, op, target as f64)),
                    Value::F32(f) => Some(compare_partial(*f as f64, op, target as f64)),
                    _ => Some(false),
                });
            }
        };

        let target_bytes = encode_int_for_zone(&target_value).unwrap();
        let target_u64 = u64::from_be_bytes(target_bytes);

        // === Secondary-index fast path ===
        if let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
        {
            return self.filter_scan_via_index(
                snapshot,
                type_name,
                type_id,
                ifd.field_id,
                op,
                &target_bytes,
                limit,
            );
        }

        // === Zone-map fallback ===
        // Resolve the field's stable catalog ID for the zone-map predicate.
        // The lookup table is the same one the extractor consulted at
        // write time, so producer and consumer agree by construction —
        // crucially, that agreement survives `rename_field` because
        // field_id is preserved (only the name in the catalog row
        // changes).
        let lookup_guard = self.zone_field_id_lookup.load();
        let field_id = lookup_guard
            .get(&type_id)
            .and_then(|entries| entries.iter().find(|(n, _)| n == field_name).map(|(_, id)| *id))
            // Field isn't enrolled (non-integer, retired, or never
            // existed) — predicate with an unmapped field is harmless:
            // `ZoneMap::bounds()` returns None and every block must scan.
            .unwrap_or(u32::MAX);
        let predicate = FieldPredicate {
            field_id,
            op,
            target: target_u64,
        };

        let prefix = KeyBuilder::object_prefix(type_id);
        let entries = self
            .storage
            .scan_prefix_filtered_at(snapshot, &prefix, &predicate)?;

        // Re-evaluate the predicate per entry (zone maps are block-level).
        // Stop accumulating once we've hit the caller's limit — the scan
        // can't terminate early but we can at least skip the object copy.
        //
        // Column projection: read only the predicate field to re-check the
        // coarse block filter; rows that fail never pay the full FieldMap
        // deserialize. The survivors (typically a small fraction) get the
        // full `deserialize_fields` so the returned Object is complete.
        //
        // The survivor path deliberately walks the blob twice — once in
        // `extract_field` (a partial walk that stops at the predicate field,
        // no allocation) and once in `deserialize_fields`. Folding both into a
        // single predicate-aware deserializer would only save the short prefix
        // re-walk on the *matching* rows, at the cost of coupling predicate
        // evaluation into the deserializer and losing `extract_field` as a
        // clean, reusable projection primitive — not worth it.
        let cap = limit.unwrap_or(usize::MAX);
        let mut objects = Vec::new();
        for (key, data) in entries {
            if objects.len() >= cap {
                break;
            }
            if key.len() < 8 {
                continue;
            }
            let passes = extract_field(&data, field_name)
                .map(|v| value_passes_int_predicate(&v, op, target_u64))
                .unwrap_or(false);
            if !passes {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);

            let mut fields = deserialize_fields(&data);
            self.strip_tombstoned_fields(type_name, &mut fields);
            objects.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields,
                raw_fields: None,
            });
        }

        Ok(objects)
    }

    /// Walk the `i:<type>:<field>:` index range for a single integer-op
    /// predicate and emit up to `limit` matching objects, reading their
    /// fields straight from the covering index entry value — no per-id LSM
    /// probe for materialization.
    ///
    /// Three storage shapes, in increasing scan-size order:
    ///
    /// * **Eq** — narrow prefix scan on `i:<type>:<field>:<target>:`. Every
    ///   returned key is a match. O(matches).
    /// * **Ge/Gt with `limit`** — seek-then-scan from the smallest passing
    ///   key, bounded by the limit. O(limit).
    /// * **Lt/Le/Ne or no limit** — bounded prefix scan from the field's
    ///   natural start. Lt/Le's matches cluster at the start of the prefix,
    ///   so a bounded scan still serves the limit in O(limit) work. Without
    ///   a limit we fall through to the full prefix scan.
    ///
    /// **Covering index.** Each `i:` entry's value is the source object's
    /// serialized FieldMap, written at create/update time. The materialize
    /// step is just a per-entry `deserialize_fields` + Object construction —
    /// no `get_many` call into the LSM, no bloom probes, no per-id snapshot
    /// reads. Entries with empty values (legacy / non-covering) fall back to
    /// the historical id-collect + `get_many` path so older databases stay
    /// readable.
    #[allow(clippy::too_many_arguments)]
    fn filter_scan_via_index(
        &self,
        snapshot: u64,
        type_name: &str,
        type_id: u64,
        field_id: u64,
        op: rhypedb_storage::zone::CompareOp,
        target_bytes: &[u8; 8],
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_storage::zone::CompareOp;

        let cap = limit.unwrap_or(usize::MAX);
        let target_u64 = u64::from_be_bytes(*target_bytes);

        // === Eq fast path — narrow value-prefix scan ===
        if matches!(op, CompareOp::Eq) {
            let prefix = KeyBuilder::field_index_value_prefix(type_id, field_id, target_bytes);
            let entries = if let Some(n) = limit {
                self.storage.scan_prefix_at_limited(snapshot, &prefix, n)?
            } else {
                self.storage.scan_prefix_at(snapshot, &prefix)?
            };
            let mut out = Vec::with_capacity(entries.len().min(cap));
            let mut fallback_ids: Vec<u64> = Vec::new();
            for (key, value) in entries {
                if out.len() + fallback_ids.len() >= cap {
                    break;
                }
                if key.len() < 8 {
                    continue;
                }
                let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
                let object_id = u64::from_be_bytes(id_bytes);
                if value.is_empty() {
                    fallback_ids.push(object_id);
                } else {
                    let mut fields = deserialize_fields(&value);
                    self.strip_tombstoned_fields(type_name, &mut fields);
                    out.push(Object {
                        type_name: type_name.into(),
                        id: object_id,
                        fields,
                        raw_fields: None,
                    });
                }
            }
            if !fallback_ids.is_empty() {
                out.extend(self.get_many_at(snapshot, type_name, &fallback_ids)?);
            }
            return Ok(out);
        }

        let prefix = KeyBuilder::field_index_prefix(type_id, field_id);
        let plen = prefix.len();

        // === Bounded range path: seek-then-scan for Gt/Ge, bounded prefix
        //     scan for Lt/Le. Skipped when no limit was pushed down. ===
        let entries = match (op, limit) {
            (CompareOp::Gt, Some(n)) => {
                if target_u64 == u64::MAX {
                    return Ok(Vec::new());
                }
                let seek_bytes = (target_u64 + 1).to_be_bytes();
                let mut start_key = bytes::BytesMut::with_capacity(prefix.len() + 8);
                start_key.extend_from_slice(&prefix);
                start_key.extend_from_slice(&seek_bytes);
                self.storage
                    .scan_from_at_limited(snapshot, &prefix, &start_key, n)?
            }
            (CompareOp::Ge, Some(n)) => {
                let mut start_key = bytes::BytesMut::with_capacity(prefix.len() + 8);
                start_key.extend_from_slice(&prefix);
                start_key.extend_from_slice(target_bytes);
                self.storage
                    .scan_from_at_limited(snapshot, &prefix, &start_key, n)?
            }
            (CompareOp::Lt, Some(n)) | (CompareOp::Le, Some(n)) => {
                self.storage.scan_prefix_at_limited(snapshot, &prefix, n)?
            }
            _ => self.storage.scan_prefix_at(snapshot, &prefix)?,
        };

        let mut out = Vec::with_capacity(entries.len().min(cap));
        let mut fallback_ids: Vec<u64> = Vec::new();
        for (key, value) in entries {
            if out.len() + fallback_ids.len() >= cap {
                break;
            }
            if key.len() != plen + 8 + 1 + 8 {
                continue;
            }
            let value_slice = &key[plen..plen + 8];
            let value_u64 = u64::from_be_bytes(value_slice.try_into().unwrap());
            let pass = match op {
                CompareOp::Lt => value_u64 < target_u64,
                CompareOp::Le => value_u64 <= target_u64,
                CompareOp::Gt => value_u64 > target_u64,
                CompareOp::Ge => value_u64 >= target_u64,
                CompareOp::Ne => value_u64 != target_u64,
                CompareOp::Eq => unreachable!("Eq handled above"),
            };
            if !pass {
                if matches!(op, CompareOp::Lt | CompareOp::Le) {
                    break;
                }
                continue;
            }
            let id_bytes: [u8; 8] = key[plen + 9..plen + 17].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);
            if value.is_empty() {
                fallback_ids.push(object_id);
            } else {
                let mut fields = deserialize_fields(&value);
                self.strip_tombstoned_fields(type_name, &mut fields);
                out.push(Object {
                    type_name: type_name.into(),
                    id: object_id,
                    fields,
                    raw_fields: None,
                });
            }
        }

        if !fallback_ids.is_empty() {
            out.extend(self.get_many_at(snapshot, type_name, &fallback_ids)?);
        }
        Ok(out)
    }

    /// Bool-valued filtered scan. Routes to the secondary index when the
    /// field is `@indexed`; otherwise falls back to a typed scan + per-row
    /// comparison.
    pub fn filter_scan_bool(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: bool,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        self.filter_scan_bool_at(
            self.storage.read_snapshot(),
            type_name,
            field_name,
            op,
            target,
            limit,
        )
    }

    /// [`filter_scan_bool`](Self::filter_scan_bool) at a caller-supplied
    /// snapshot. See [`filter_scan_at`](Self::filter_scan_at).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_scan_bool_at(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: bool,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        // Validate the field exists (errors as Retired / NotFound). A non-bool
        // field never matches a bool literal: the fallback closure below
        // returns an empty set, not the whole table.
        type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(type_name, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })
        })?;

        if let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
            && ifd.kind == IndexedKind::Bool
        {
            let encoded = encode_bool_for_index(&Value::Bool(target)).unwrap();
            return self.filter_scan_via_index(
                snapshot,
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encoded,
                limit,
            );
        }
        self.filter_scan_fallback(snapshot, type_name, field_name, limit, |v| match v {
            Value::Bool(b) => Some(compare_bool(*b, op, target)),
            _ => Some(false),
        })
    }

    /// Float-valued filtered scan. `target` is interpreted as `f64`; both
    /// `f32` and `f64` index entries share the same 8-byte sortable layout
    /// (`f32` values widen on write).
    pub fn filter_scan_float(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: f64,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        self.filter_scan_float_at(
            self.storage.read_snapshot(),
            type_name,
            field_name,
            op,
            target,
            limit,
        )
    }

    /// [`filter_scan_float`](Self::filter_scan_float) at a caller-supplied
    /// snapshot. See [`filter_scan_at`](Self::filter_scan_at).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_scan_float_at(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: f64,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let field_def = type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(type_name, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })
        })?;
        let is_float = matches!(
            field_def.field_type,
            FieldType::Scalar(ScalarType::F32 | ScalarType::F64)
        );

        // Float index fast path — only when the field actually is a float.
        if is_float
            && let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
            && ifd.kind == IndexedKind::Float
        {
            let encoded = encode_f64_for_index(target);
            return self.filter_scan_via_index(
                snapshot,
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encoded,
                limit,
            );
        }
        // Per-row fallback: compare ANY numeric field value against the f64
        // target, so an int field compared to a float literal filters
        // correctly. Non-numeric fields never match — an empty result, not the
        // whole table.
        self.filter_scan_fallback(snapshot, type_name, field_name, limit, |v| match v {
            Value::F64(f) => Some(compare_partial(*f, op, target)),
            Value::F32(f) => Some(compare_partial(*f as f64, op, target)),
            Value::U32(n) => Some(compare_partial(*n as f64, op, target)),
            Value::U64(n) => Some(compare_partial(*n as f64, op, target)),
            Value::I32(n) => Some(compare_partial(*n as f64, op, target)),
            Value::I64(n) => Some(compare_partial(*n as f64, op, target)),
            _ => Some(false),
        })
    }

    /// Bytes-valued filtered scan. Mirrors `filter_scan_str` but for the
    /// `Bytes` scalar — the encoder, key layout, and parsing are identical
    /// (variable-length escape + `\x00\x00` terminator).
    pub fn filter_scan_bytes(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: &[u8],
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        // Validate the field exists (errors as Retired / NotFound). A non-bytes
        // field never matches a bytes literal: the fallback closure below
        // returns an empty set, not the whole table.
        type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(type_name, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })
        })?;

        let snapshot = self.storage.read_snapshot();
        if let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
            && ifd.kind == IndexedKind::Bytes
        {
            return self.filter_scan_via_index_var(
                snapshot,
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encode_bytes_for_index(target),
                limit,
            );
        }
        self.filter_scan_fallback(snapshot, type_name, field_name, limit, |v| match v {
            Value::Bytes(b) => Some(compare_ord(b.as_ref(), op, target)),
            _ => Some(false),
        })
    }

    /// String-valued filtered scan. Mirrors `filter_scan` but for `String`
    /// scalars: pushes a single-field comparison against a string literal
    /// down to the secondary index when one is declared on the field. The
    /// fast path uses the variable-length encoded value layout
    /// (`KeyBuilder::field_index_var`).
    ///
    /// Falls back to `scan_type` + per-row comparison when the field isn't
    /// indexed or isn't a `String` scalar — strings have no zone-map
    /// acceleration today, so the non-indexed path is a full scan.
    pub fn filter_scan_str(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: &str,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        self.filter_scan_str_at(
            self.storage.read_snapshot(),
            type_name,
            field_name,
            op,
            target,
            limit,
        )
    }

    /// [`filter_scan_str`](Self::filter_scan_str) at a caller-supplied snapshot.
    /// See [`filter_scan_at`](Self::filter_scan_at).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_scan_str_at(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: &str,
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        let type_id = self.resolve_type_id(type_name)?;
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        // Validate the field exists (errors as Retired / NotFound). A
        // non-string field never matches a string literal: the fallback
        // closure below returns an empty set, not the whole table.
        type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(type_name, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })
        })?;

        // === Indexed fast path ===
        if let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
            && ifd.kind == IndexedKind::String
        {
            return self.filter_scan_via_index_var(
                snapshot,
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encode_str_for_index(target),
                limit,
            );
        }

        // === Non-indexed fallback: full type scan, post-filter ===
        self.filter_scan_fallback(snapshot, type_name, field_name, limit, |v| match v {
            Value::String(s) => Some(compare_ord(s.as_str(), op, target)),
            _ => Some(false),
        })
    }

    /// Per-row fallback when no index serves the predicate. Walks the object-
    /// key prefix and applies `predicate` to the field's value, capping at
    /// `limit` matches. `predicate` returns `Some(bool)` (pass / fail) per
    /// value; `None` is treated as fail.
    ///
    /// Column projection: the predicate is evaluated against *only* the named
    /// field, extracted in O(1)-skip fashion from each row's raw bytes. The
    /// full `deserialize_fields` (HashMap + every `Value`) runs only for rows
    /// that pass — so a selective filter over a wide type stops paying to
    /// materialize the columns it discards. `field_name` is always a live
    /// (non-retired) field here: every public `filter_scan_*` entry point
    /// validates it before falling through, so reading it pre-strip matches
    /// the post-strip value the old `scan_type` path saw.
    fn filter_scan_fallback<F>(
        &self,
        snapshot: u64,
        type_name: &str,
        field_name: &str,
        limit: Option<usize>,
        mut predicate: F,
    ) -> EngineResult<Vec<Object>>
    where
        F: FnMut(&Value) -> Option<bool>,
    {
        let type_id = self.resolve_type_id(type_name)?;
        let prefix = KeyBuilder::object_prefix(type_id);
        let entries = self.storage.scan_prefix_at(snapshot, &prefix)?;

        let cap = limit.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        for (key, data) in entries {
            if out.len() >= cap {
                break;
            }
            if key.len() < 8 {
                continue;
            }
            let pass = extract_field(&data, field_name)
                .as_ref()
                .and_then(&mut predicate)
                .unwrap_or(false);
            if !pass {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);
            let mut fields = deserialize_fields(&data);
            self.strip_tombstoned_fields(type_name, &mut fields);
            out.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields,
                raw_fields: None,
            });
        }
        Ok(out)
    }

    /// Index-backed fast path for variable-length encoded values (String,
    /// Bytes). The on-disk key layout is
    /// `i:<type>:<field>:<escaped_value>\x00\x00<object_id>`. `target_encoded`
    /// is the caller's value run through the same encoder used at write
    /// time (`encode_bytes_for_index` / `encode_str_for_index`).
    ///
    /// Eq narrows to the value-prefix scan; range/Ne walk the field's full
    /// prefix and compare encoded value bytes (which preserve sort order).
    /// Covering payload semantics match the fixed-width variant — empty
    /// value falls back to per-id `get_many` for legacy entries.
    #[allow(clippy::too_many_arguments)]
    fn filter_scan_via_index_var(
        &self,
        snapshot: u64,
        type_name: &str,
        type_id: u64,
        field_id: u64,
        op: rhypedb_storage::zone::CompareOp,
        target_encoded: &[u8],
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_storage::zone::CompareOp;

        let cap = limit.unwrap_or(usize::MAX);

        // === Eq fast path — narrow value-prefix scan ===
        if matches!(op, CompareOp::Eq) {
            let prefix =
                KeyBuilder::field_index_var_value_prefix(type_id, field_id, target_encoded);
            let entries = if let Some(n) = limit {
                self.storage.scan_prefix_at_limited(snapshot, &prefix, n)?
            } else {
                self.storage.scan_prefix_at(snapshot, &prefix)?
            };
            let mut out = Vec::with_capacity(entries.len().min(cap));
            let mut fallback_ids: Vec<u64> = Vec::new();
            for (key, value) in entries {
                if out.len() + fallback_ids.len() >= cap {
                    break;
                }
                if key.len() < 8 {
                    continue;
                }
                let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
                let object_id = u64::from_be_bytes(id_bytes);
                if value.is_empty() {
                    fallback_ids.push(object_id);
                } else {
                    let mut fields = deserialize_fields(&value);
                    self.strip_tombstoned_fields(type_name, &mut fields);
                    out.push(Object {
                        type_name: type_name.into(),
                        id: object_id,
                        fields,
                        raw_fields: None,
                    });
                }
            }
            if !fallback_ids.is_empty() {
                out.extend(self.get_many_at(snapshot, type_name, &fallback_ids)?);
            }
            return Ok(out);
        }

        // === Range/Ne path — scan the whole field prefix ===
        let prefix = KeyBuilder::field_index_prefix(type_id, field_id);
        let plen = prefix.len();
        let entries = self.storage.scan_prefix_at(snapshot, &prefix)?;

        let mut out = Vec::with_capacity(entries.len().min(cap));
        let mut fallback_ids: Vec<u64> = Vec::new();
        for (key, value) in entries {
            if out.len() + fallback_ids.len() >= cap {
                break;
            }
            // Need plen + at least one 0x00 0x00 terminator + 8 byte id.
            if key.len() < plen + 2 + 8 {
                continue;
            }
            // Find the embedded `\x00\x00` terminator. Encoded values cannot
            // contain it (every embedded NUL is followed by 0x01), so the
            // first occurrence at or after `plen` is the boundary.
            let mut term_at: Option<usize> = None;
            let mut i = plen;
            while i + 1 < key.len() - 8 {
                if key[i] == 0 && key[i + 1] == 0 {
                    term_at = Some(i);
                    break;
                }
                i += 1;
            }
            let Some(term) = term_at else { continue };
            // Encoded value (with terminator) for byte-wise compare; range
            // checks compare against target_encoded which carries the same
            // shape, so sort order is preserved.
            let value_with_term = &key[plen..term + 2];
            let cmp = value_with_term.cmp(target_encoded);
            let pass = match op {
                CompareOp::Lt => cmp.is_lt(),
                CompareOp::Le => cmp.is_le(),
                CompareOp::Gt => cmp.is_gt(),
                CompareOp::Ge => cmp.is_ge(),
                CompareOp::Ne => cmp.is_ne(),
                CompareOp::Eq => unreachable!("Eq handled above"),
            };
            if !pass {
                // The scan is sorted ascending by encoded value, so once
                // Lt/Le starts failing we can stop. Gt/Ge fail at the
                // start until they begin passing — can't short-circuit.
                if matches!(op, CompareOp::Lt | CompareOp::Le) {
                    break;
                }
                continue;
            }
            let id_bytes: [u8; 8] = key[term + 2..term + 10].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);
            if value.is_empty() {
                fallback_ids.push(object_id);
            } else {
                let mut fields = deserialize_fields(&value);
                self.strip_tombstoned_fields(type_name, &mut fields);
                out.push(Object {
                    type_name: type_name.into(),
                    id: object_id,
                    fields,
                    raw_fields: None,
                });
            }
        }

        if !fallback_ids.is_empty() {
            out.extend(self.get_many_at(snapshot, type_name, &fallback_ids)?);
        }
        Ok(out)
    }

    /// Update an object's fields. Only the provided fields are updated;
    /// unmentioned fields are preserved.
    /// Update an object. Untagged: delegates to
    /// [`update_with_origin`](Self::update_with_origin) with `origin = None`.
    pub fn update(
        &self,
        type_name: &str,
        object_id: u64,
        updates: FieldMap,
    ) -> EngineResult<Object> {
        self.update_with_origin(type_name, object_id, updates, None)
    }

    /// Update an object, tagging the emitted [`ChangeEvent`] with `origin`
    /// (see [`create_with_origin`](Self::create_with_origin) for the rationale).
    pub fn update_with_origin(
        &self,
        type_name: &str,
        object_id: u64,
        updates: FieldMap,
        origin: Option<u64>,
    ) -> EngineResult<Object> {
        let _migration_guard = self.migration_lock.read();
        let type_id = self.resolve_type_id(type_name)?;
        // Card 2d: no quiesce — the double-write hook re-stamps the migrating
        // field's shadow over the merged blob (apply_migrating_field_hook).
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

        // Validate update fields. Retired fields surface as
        // `FieldRetired` so the operator learns their schema is stale,
        // not "you typoed a field name."
        for (field_name, value) in &updates {
            let field_def = type_def.get_field(field_name).ok_or_else(|| {
                self.field_retired_error(type_name, field_name)
                    .unwrap_or_else(|| EngineError::FieldNotFound {
                        type_name: type_name.into(),
                        field: field_name.clone(),
                    })
            })?;
            validate_value(field_def, value)?;
        }

        let key = KeyBuilder::object(type_id, object_id);
        let mut txn = self.storage.begin_txn();

        let existing_data =
            self.storage
                .get(&txn, &key)?
                .ok_or_else(|| EngineError::ObjectNotFound {
                    type_name: type_name.into(),
                    object_id,
                })?;

        let mut fields = deserialize_fields(&existing_data);

        // Check unique constraints for updated fields. This touches a SINGLE
        // object, so there are no earlier rows whose buffered unique puts a
        // later row would miss — the intra-batch hazard simply doesn't arise.
        // Hand the helper a throwaway staged set to satisfy its signature.
        let mut staged_unique: HashMap<Bytes, u64> = HashMap::new();
        for (field_name, value) in &updates {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() {
                // Remove the old `u:` entry whenever the field HELD a non-null
                // value — independent of the new value. Gating the removal on
                // the NEW value being non-null (as a single combined condition
                // once did) skipped cleanup on an update-to-Null, dangling the
                // `u:<type>:<field>:<old_value>` key → a false UniqueViolation
                // when the freed value is later reused. This mirrors the
                // keyspace contract create (insert-only) and delete
                // (remove-if-stored-non-null) already honour.
                if let Some(old_value) = fields.get(field_name)
                    && !matches!(old_value, Value::Null)
                {
                    self.remove_unique_index(&mut txn, type_name, type_id, field_name, old_value)?;
                }
                // Claim the new value only when it is non-null (Null carries no
                // uniqueness constraint — many rows may be Null at once).
                if !matches!(value, Value::Null) {
                    self.check_unique_and_insert(
                        &mut txn, type_name, type_id, field_name, value, object_id,
                        &mut staged_unique,
                    )?;
                }
            }
        }

        // Build the NEW field set by merging updates into the old fields.
        // We need both the old values (to look up index entries to remove)
        // and the merged set (to build the covering payload AND the new
        // object entry).
        let old_indexed_snapshot: Vec<Option<Value>> =
            if let Some(idx_fields) = self.indexed_fields.get(type_name) {
                idx_fields
                    .iter()
                    .map(|ifd| fields.get(&ifd.name).cloned())
                    .collect()
            } else {
                Vec::new()
            };

        let any_update = !updates.is_empty();
        for (k, v) in updates {
            fields.insert(k, v);
        }

        // Card 2: double-write the shadow over the MERGED field set into the
        // SERIALIZED blob. Apply to a clone so `fields` stays shadow-free — the
        // cover maintenance, change event, and returned Object all use `fields`
        // and must see only real fields. Cloned only while a migration runs.
        let serialized = if self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            let mut with_shadow = fields.clone();
            self.apply_migrating_field_hook(type_id, type_name, object_id, &mut with_shadow)?;
            serialize_fields(&with_shadow)
        } else {
            serialize_fields(&fields)
        };

        // Maintain secondary-index covering payloads, refresh outbound rev-edge
        // covers, and bump this object's generation — shared with the card-2
        // cutover via `rewrite_object_and_maintain_covers` so cutover provably
        // touches every keyspace a normal update touches (see that method).
        self.rewrite_object_and_maintain_covers(
            &mut txn,
            type_name,
            type_id,
            object_id,
            &fields,
            &serialized,
            &old_indexed_snapshot,
            any_update,
        )?;

        self.storage.put(&mut txn, &key, serialized)?;
        let version = self.storage.commit(&mut txn).map_err(|e| {
            // The commit didn't land — undo the in-memory bump so future
            // updates don't skip a generation and so the stamped versions
            // in existing rev_edges still match a successful next write.
            self.rollback_version(type_id, object_id);
            match e {
                rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
                other => EngineError::Storage(other),
            }
        })?;

        // Enqueue cover refresh for this target. Every other rev_edge that
        // embedded this object as `<name>__cover` (under a different source)
        // now has a stale snapshot; the worker will scan `r:<target>:*` to
        // find those sources and re-run Phase 1 for each. Send failure means
        // the channel was closed (Drop in progress) — fall through, the
        // next read still detects staleness via cover_v.
        if let Some(tx) = self.cover_refresh_tx.lock().as_ref() {
            let _ = tx.send((type_id, object_id));
        }

        self.subscriptions.publish(ChangeEvent {
            version,
            kind: ChangeKind::Update,
            type_name: type_name.into(),
            object_id,
            fields: Some(fields_to_json(&fields)),
            origin,
        });

        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields,
            raw_fields: None,
        })
    }

    /// Stage all cover/index maintenance for an object whose `o:` blob is being
    /// (re)written: rewrite the secondary-index covering payload for every
    /// `@indexed` field, refresh this object's outbound rev-edge covers, and
    /// bump its generation so incoming `<name>__cover` snapshots re-probe.
    ///
    /// Shared by `update()` and the card-2 cutover (`cutover_field_type_migration`).
    /// The migrated field's value otherwise goes stale in the `i:` covering
    /// payloads — which carry no `<field>__cover_v` generation stamp, so the
    /// cutover generation-bump alone can NOT invalidate them (the covered
    /// `filter_scan_via_index` reads the payload verbatim). Routing both writers
    /// through one helper guarantees cutover touches exactly the keyspaces a
    /// normal update touches.
    ///
    /// Stages into `txn`; the CALLER stages the `o:` blob put and commits, and
    /// owns `rollback_version` on a commit failure (this bumps the in-memory
    /// generation). Does NOT enqueue the background cover-refresh or publish a
    /// change event — those are live-write concerns, not part of a cutover.
    #[allow(clippy::too_many_arguments)]
    fn rewrite_object_and_maintain_covers(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        type_id: u64,
        object_id: u64,
        new_fields: &FieldMap,
        serialized: &Bytes,
        old_indexed_snapshot: &[Option<Value>],
        any_update: bool,
    ) -> EngineResult<()> {
        // Maintain secondary index entries for any @indexed fields. Covering
        // value = the new serialized fields, so subsequent filter_scans can
        // read full Objects from the index without per-id LSM probes.
        //
        // Three sub-cases per indexed field:
        //   * Value changed (old != new): delete old entry, insert new (with covering).
        //   * Value unchanged but ANY field changed: re-write the entry's
        //     covering payload (same key, fresh value bytes).
        //   * No update at all: nothing to do (skipped at the outer level).
        if let Some(idx_fields) = self.indexed_fields.get(type_name) {
            for (ifd, old_value_opt) in idx_fields.iter().zip(old_indexed_snapshot.iter()) {
                let new_value_opt = new_fields.get(&ifd.name).cloned();
                let value_changed = old_value_opt != &new_value_opt;
                if value_changed {
                    if let Some(old_v) = old_value_opt
                        && !matches!(old_v, Value::Null)
                    {
                        self.remove_field_index(
                            txn,
                            type_id,
                            ifd.field_id,
                            ifd.kind,
                            old_v,
                            object_id,
                        )?;
                    }
                    if let Some(new_v) = &new_value_opt
                        && !matches!(new_v, Value::Null)
                    {
                        self.insert_field_index(
                            txn,
                            type_id,
                            ifd.field_id,
                            ifd.kind,
                            new_v,
                            object_id,
                            serialized.clone(),
                        )?;
                    }
                } else if any_update
                    && let Some(new_v) = &new_value_opt
                    && !matches!(new_v, Value::Null)
                {
                    // Same key — `put` overwrites the value with the new
                    // covering payload.
                    self.insert_field_index(
                        txn,
                        type_id,
                        ifd.field_id,
                        ifd.kind,
                        new_v,
                        object_id,
                        serialized.clone(),
                    )?;
                }
            }
        }

        // Phase 1: refresh covering reverse-edges on THIS object's own
        // outbound forward relations. The rev_edge value stored at each
        // linked target carries the source object's effective fields (and
        // covers for OTHER forward-1:1 targets) — both go stale when the
        // source's blob is rewritten. Bounded cost: number of outbound
        // relation endpoints.
        self.refresh_outbound_rev_edges(txn, type_name, object_id, Some(serialized))?;

        // Phase 2: bump this object's generation. Every rev_edge that
        // embedded us as `<name>__cover` earlier now has a stale snapshot;
        // the executor's fusion path detects mismatch via a HashMap lookup
        // against this counter and falls through to a fresh LSM probe for
        // those specific targets. Bounded write cost (one in-memory bump +
        // one persisted `g:` put) regardless of how many incoming references
        // this object has — that fan-in could be millions.
        let new_version = self.bump_version(type_id, object_id);
        self.storage.put(
            txn,
            &KeyBuilder::object_version(type_id, object_id),
            Bytes::copy_from_slice(&new_version.to_be_bytes()),
        )?;
        Ok(())
    }

    /// Delete an object, enforcing @on_delete policies on all inbound relationships.
    /// Cascades are recursive — if deleting A cascades to B, and B has its own
    /// cascade relationships, those are followed too.
    /// Delete an object (cascading to its owned edges). Untagged: delegates to
    /// [`delete_with_origin`](Self::delete_with_origin) with `origin = None`.
    pub fn delete(&self, type_name: &str, object_id: u64) -> EngineResult<()> {
        self.delete_with_origin(type_name, object_id, None)
    }

    /// Delete an object, tagging EVERY emitted [`ChangeEvent`] — the top-level
    /// delete and every cascaded delete — with the SAME `origin` (see
    /// [`create_with_origin`](Self::create_with_origin) for the rationale).
    pub fn delete_with_origin(
        &self,
        type_name: &str,
        object_id: u64,
        origin: Option<u64>,
    ) -> EngineResult<()> {
        let _migration_guard = self.migration_lock.read();
        let type_id = self.resolve_type_id(type_name)?;

        let mut txn = self.storage.begin_txn();
        // type_id keyed instead of (String, u64) — drops a String alloc
        // per cascaded object. At K=100 that's 100 fewer allocations per
        // User delete. The value is the deleted object's scalar fields,
        // captured by `delete_inner` for the Delete change event (so a
        // subscriber learns *which* object went away — esp. its slug/
        // identifying fields — the same way create/update report them).
        // `None` means no scalar payload: a pure edge-only join row (no
        // scalar fields — capture is skipped even on a top-level delete that
        // read the blob for its existence check), or an object already gone
        // in a circular cascade.
        let mut deleted: HashMap<(u64, u64), Option<HashMap<String, serde_json::Value>>> =
            HashMap::with_capacity(128);
        // Arena instead of `Vec<Bytes>` — every tombstone key lives in one
        // pre-sized buffer; `into_keys` produces refcount-only slices
        // when we hand them to `delete_batch`. At K=100 that's ~500
        // mallocs replaced by ONE.
        let mut arena = TombstoneArena::new();

        // Top-level delete: verify existence (per public API contract).
        // Pass `None` for the cascade context — there's no parent rev_edge
        // value to extract peer targets from.
        self.delete_inner(
            &mut txn,
            type_id,
            object_id,
            true,
            &mut deleted,
            &mut arena,
            None,
        )?;

        let tombstones = arena.into_keys();
        self.storage.delete_batch(&mut txn, &tombstones)?;

        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        // Drop the in-memory version-counter entries for everything the
        // commit just removed. The persisted `g:` keys were already
        // tombstoned inside the txn.
        for (del_type_id, del_id) in deleted.keys() {
            self.forget_version(*del_type_id, *del_id);
        }

        for ((del_type_id, del_id), fields) in &deleted {
            let type_name = self
                .type_name_by_id
                .get(del_type_id)
                .cloned()
                .unwrap_or_default();
            self.subscriptions.publish(ChangeEvent {
                version,
                kind: ChangeKind::Delete,
                type_name,
                object_id: *del_id,
                fields: fields.clone(),
                origin,
            });
        }

        Ok(())
    }

    /// Internal recursive delete. The `deleted` set prevents infinite loops
    /// if there are circular cascade relationships.
    ///
    /// `verify_exists`: only the top-level public delete needs the
    /// existence check. Cascade-recursive calls receive IDs from the
    /// reverse-edge scan we just did — those objects provably exist, so the
    /// LSM probe is pure waste.
    /// `cascade_ctx` carries the (parent_rel_id, source_cover_bytes) pair
    /// captured by the parent's inbound scan. When present:
    ///   * `parent_rel_id` is the relation the parent used to find us
    ///     (e.g. `Rating.user` when cascading from User → Rating). The
    ///     parent already staged tombstones for both directions of that
    ///     edge, so this call must NOT re-stage them.
    ///   * The cover bytes are the rev_edge value the parent's scan
    ///     returned. With symmetric covers (inline-relations create or
    ///     update-time refresh), this blob carries every OTHER forward 1:1
    ///     target id directly — extract via `find_u64_field_in_raw` and
    ///     stage tombstones without a per-relation `scan_prefix`. At
    ///     K=100 cascading Ratings this drops 200 LSM scans per User
    ///     delete.
    // Internal staging helper: the args are cohesive (txn + identity +
    // delete-mode flags); grouping them into a struct would add indirection
    // without clarity.
    #[allow(clippy::too_many_arguments)]
    fn delete_inner(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_id: u64,
        object_id: u64,
        verify_exists: bool,
        deleted: &mut HashMap<(u64, u64), Option<HashMap<String, serde_json::Value>>>,
        arena: &mut TombstoneArena,
        cascade_ctx: Option<(u64, Bytes)>,
    ) -> EngineResult<()> {
        if deleted.contains_key(&(type_id, object_id)) {
            return Ok(()); // already deleted in this cascade chain
        }
        // Reserve the slot now (no payload yet); the blob read below fills it
        // in when this type carries scalar fields. A plain insert here would
        // clobber a prior capture on a circular-cascade re-hit — but the
        // `contains_key` guard above already returned for that case.
        deleted.insert((type_id, object_id), None);

        // One HashMap lookup → all the per-type schema info the cascade
        // walk needs. Saves repeated `schema.get_type` / `rel_ids.get` /
        // `format!("Type.field")` per cascaded object.
        let meta = self.cascade_meta_by_id.get(&type_id).ok_or_else(|| {
            EngineError::TypeNotFound(
                self.type_name_by_id
                    .get(&type_id)
                    .cloned()
                    .unwrap_or_else(|| format!("type_id={type_id}")),
            )
        })?;

        // Card 2d: deletes (incl. cascades) into a migrating type are now
        // ALLOWED — the object and its `<field>__shadow` siblings are dropped
        // together with the rest of the object blob, so the migration stays
        // consistent (a deleted row simply never reaches cutover).

        // Read the object payload when we need it for: unique-index cleanup,
        // secondary-index cleanup, the top-level existence check, OR capturing
        // the deleted object's scalar fields for the Delete change event
        // (`has_scalar`). It's skipped entirely for pure edge-only join rows
        // (Rating in the bench) — they carry no scalar payload to clean up or
        // report, so the cascade never touches their object blob. `has_scalar`
        // is a superset of `has_unique`/`has_indexed`, so this gate is
        // effectively "scalar-bearing type OR a verify".
        let type_idx_fields = self.indexed_fields.get(&meta.type_name);
        if meta.has_unique || meta.has_indexed || meta.has_scalar || verify_exists {
            let obj_key = KeyBuilder::object(type_id, object_id);
            let obj_data = self.storage.get(txn, &obj_key)?;
            if obj_data.is_none() && verify_exists {
                return Err(EngineError::ObjectNotFound {
                    type_name: meta.type_name.clone(),
                    object_id,
                });
            }
            // Cascade-recursive call against an object that's already
            // gone (e.g. a circular cascade chain). Continue silently.
            if let Some(data) = &obj_data {
                let mut fields = deserialize_fields(data);
                if meta.has_unique
                    && let Some(type_def) = self.schema.get_type(&meta.type_name)
                {
                    for field_def in &type_def.fields {
                        if field_def.is_unique()
                            && let Some(value) = fields.get(&field_def.name)
                            && !matches!(value, Value::Null)
                        {
                            let field_key = format!("{}.{}", meta.type_name, field_def.name);
                            let field_id = self.field_ids[&field_key];
                            let value_bytes = value_to_index_bytes(value);
                            arena.push_unique_index(type_id, field_id, &value_bytes);
                        }
                    }
                }
                if let Some(idx_fields) = type_idx_fields {
                    for ifd in idx_fields {
                        let Some(value) = fields.get(&ifd.name) else {
                            continue;
                        };
                        if matches!(value, Value::Null) {
                            continue;
                        }
                        match ifd.kind {
                            IndexedKind::Integer => {
                                if let Some(encoded) = encode_int_for_zone(value) {
                                    arena.push_field_index(
                                        type_id,
                                        ifd.field_id,
                                        &encoded,
                                        object_id,
                                    );
                                }
                            }
                            IndexedKind::Bool => {
                                if let Some(encoded) = encode_bool_for_index(value) {
                                    arena.push_field_index(
                                        type_id,
                                        ifd.field_id,
                                        &encoded,
                                        object_id,
                                    );
                                }
                            }
                            IndexedKind::Float => {
                                if let Some(encoded) = encode_float_for_index(value) {
                                    arena.push_field_index(
                                        type_id,
                                        ifd.field_id,
                                        &encoded,
                                        object_id,
                                    );
                                }
                            }
                            IndexedKind::String => {
                                if let Value::String(s) = value {
                                    let encoded = encode_str_for_index(s);
                                    arena.push_field_index_var(
                                        type_id,
                                        ifd.field_id,
                                        &encoded,
                                        object_id,
                                    );
                                }
                            }
                            IndexedKind::Bytes => {
                                if let Value::Bytes(b) = value {
                                    let encoded = encode_bytes_for_index(b);
                                    arena.push_field_index_var(
                                        type_id,
                                        ifd.field_id,
                                        &encoded,
                                        object_id,
                                    );
                                }
                            }
                        }
                    }
                }

                // Capture the deleted object's scalar fields for the Delete
                // change event — the same payload create/update emit, so a
                // subscriber learns *which* object went away (esp. its
                // slug/identifying fields), not just an opaque id. Strip the
                // migration shadow siblings and retired-field names first: they
                // live in the on-disk blob but must NEVER reach a caller (a
                // reserved namespace / a name the current schema doesn't know).
                // `fields` is otherwise scalar-only — relations are edges, not
                // blob entries — so this mirrors create/update's scalar set.
                //
                // Gated on `has_scalar` (NOT just "the blob was read"): a
                // top-level delete reads the blob for the existence check even
                // for an edge-only type, but such a type has no identifying
                // scalar data — leave its slot `None` so a DIRECT delete of an
                // edge-only row agrees with a CASCADE delete of the same type
                // (both `None`) instead of emitting an empty `Some({})`.
                if meta.has_scalar {
                    self.strip_tombstoned_fields(&meta.type_name, &mut fields);
                    if let Some(slot) = deleted.get_mut(&(type_id, object_id)) {
                        *slot = Some(fields_to_json(&fields));
                    }
                }
            }
        }

        // Inbound relationships. `scan_prefix_raw` returns rev_edge values
        // verbatim so the cover blob can ride along to the recursive call.
        // Stage tombstones directly into the arena as we scan — historical
        // `edges_to_remove` Vec was a 200-element intermediate at K=100,
        // paid for by Vec doublings + a second loop that just copied the
        // same tuples. Tracking `arena_start` lets us truncate on a deny
        // violation.
        let arena_start = arena.len();
        // (source_type_id, source_id, source_rel_id, source_cover_bytes)
        let mut objects_to_cascade: Vec<(u64, u64, u64, Bytes)> = Vec::new();
        let mut deny_info: Option<(String, String)> = None;

        if let Some(incoming) = self.incoming_relations.get(&type_id) {
            'incoming: for inc in incoming {
                let rev_prefix = KeyBuilder::reverse_edge_prefix(object_id, inc.rel_id);
                let reverse_edges = self.scan_prefix_raw(txn, &rev_prefix)?;
                if reverse_edges.is_empty() {
                    continue;
                }
                match inc.policy {
                    OnDeletePolicy::Deny => {
                        deny_info = Some((inc.source_type.clone(), inc.source_field.clone()));
                        break 'incoming;
                    }
                    OnDeletePolicy::Remove => {
                        arena.reserve(2 * reverse_edges.len());
                        for (source_id, _) in reverse_edges {
                            arena.push_edge(source_id, inc.rel_id, object_id);
                            arena.push_reverse_edge(object_id, inc.rel_id, source_id);
                        }
                    }
                    OnDeletePolicy::Cascade => {
                        arena.reserve(2 * reverse_edges.len());
                        objects_to_cascade.reserve(reverse_edges.len());
                        for (source_id, source_cover) in reverse_edges {
                            arena.push_edge(source_id, inc.rel_id, object_id);
                            arena.push_reverse_edge(object_id, inc.rel_id, source_id);
                            objects_to_cascade.push((
                                inc.source_type_id,
                                source_id,
                                inc.rel_id,
                                source_cover,
                            ));
                        }
                    }
                }
            }
        }

        if let Some((ref_type, ref_field)) = deny_info {
            arena.truncate(arena_start);
            return Err(EngineError::DeleteDenied {
                type_name: meta.type_name.clone(),
                object_id,
                referencing_type: ref_type,
                referencing_field: ref_field,
            });
        }

        // Recursively delete cascade targets. The IDs came from a reverse-
        // edge scan we just did inside this same txn, so they provably
        // exist — skip the existence check on the recursive call. The
        // rev_edge value (cover blob) goes with each child, paired with
        // the rel_id that walked us there.
        for (cascade_type_id, cascade_id, cascade_rel_id, cascade_cover) in objects_to_cascade {
            self.delete_inner(
                txn,
                cascade_type_id,
                cascade_id,
                false,
                deleted,
                arena,
                Some((cascade_rel_id, cascade_cover)),
            )?;
        }

        // Outbound edge tombstones. Two halves:
        //   1) Cover-extract: pull every forward 1:1 peer target id out
        //      of the parent-supplied cover blob via
        //      `find_u64_field_in_raw`. Skip parent's rel_id (handled
        //      above). Mark covered.
        //   2) Fallback `scan_prefix_raw` for relations not covered
        //      (many-relations always; forward 1:1 with no peer at the
        //      cover-write time).
        // Tiny Vec — typical type has ≤ 4 forward relations, linear-scan
        // `contains` is faster than a HashSet at that size.
        let mut covered_rel_ids: Vec<u64> = Vec::with_capacity(4);
        if let Some((parent_rel_id, ref cover)) = cascade_ctx {
            covered_rel_ids.push(parent_rel_id);
            if !cover.is_empty() {
                for rel in &meta.forward_relations {
                    if rel.is_many || rel.rel_id == parent_rel_id {
                        continue;
                    }
                    if let Some(target_id) =
                        crate::object::find_u64_field_in_raw(cover, &rel.field_name)
                    {
                        arena.push_edge(object_id, rel.rel_id, target_id);
                        arena.push_reverse_edge(target_id, rel.rel_id, object_id);
                        covered_rel_ids.push(rel.rel_id);
                    }
                }
            }
        }

        for rel in &meta.forward_relations {
            if covered_rel_ids.contains(&rel.rel_id) {
                continue;
            }
            let edge_prefix = KeyBuilder::edge_prefix(object_id, rel.rel_id);
            let outbound = self.scan_prefix_raw(txn, &edge_prefix)?;
            for (target_id, _) in &outbound {
                arena.push_edge(object_id, rel.rel_id, *target_id);
                arena.push_reverse_edge(*target_id, rel.rel_id, object_id);
            }
        }

        // Stage the object's own tombstone.
        arena.push_object(type_id, object_id);

        // Drop the persisted `g:` generation key — but only for objects that
        // were UPDATED (generation >= 2). A never-updated object has no `g:`
        // key on disk (born-at-1 lives in memory; existence comes from the
        // `o:` key), so tombstoning it would be pure WAL+SST bloat. The atomic
        // `version_counter_count` skips even the RwLock acquire when the
        // counter map is empty.
        if self
            .version_counter_count
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0
        {
            let was_updated = self
                .version_counters
                .read()
                .get(&(type_id, object_id))
                .is_some_and(|&v| v > 1);
            if was_updated {
                arena.push_object_version(type_id, object_id);
            }
        }

        Ok(())
    }

    /// Create a relationship (edge) between two objects.
    pub fn link(
        &self,
        source_type: &str,
        source_id: u64,
        field_name: &str,
        target_id: u64,
        edge_fields: Option<FieldMap>,
    ) -> EngineResult<()> {
        let _migration_guard = self.migration_lock.read();
        let source_type_id = self.resolve_type_id(source_type)?;
        let type_def = self
            .schema
            .get_type(source_type)
            .ok_or_else(|| EngineError::TypeNotFound(source_type.into()))?;

        let field = type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(source_type, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                })
        })?;

        let rel = match &field.field_type {
            FieldType::Relation(r) => r,
            _ => {
                return Err(EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                });
            }
        };

        // Validate edge-field values against the relation's declared edge
        // fields — the symmetric guarantee to validate_value() for object
        // fields: every edge field must be declared, and its Value variant must
        // match the declared scalar type. The query language already coerces
        // edge literals to the declared type; this also guards direct callers,
        // so a wrong variant errors instead of silently round-tripping back
        // wrong.
        if let Some(ref ef) = edge_fields {
            for (name, value) in ef {
                let edge_def = rel
                    .edge_fields
                    .iter()
                    .find(|e| e.name == *name)
                    .ok_or_else(|| EngineError::FieldNotFound {
                        type_name: format!("{source_type}.{field_name}"),
                        field: name.clone(),
                    })?;
                validate_edge_value(edge_def, value)?;
            }
        }

        // Verify the target type isn't tombstoned.
        let target_type_id = self.resolve_type_id(&rel.target_type)?;

        let mut txn = self.storage.begin_txn();

        let source_key = KeyBuilder::object(source_type_id, source_id);
        let source_data = self.storage.get(&txn, &source_key)?;
        if source_data.is_none() {
            return Err(EngineError::ObjectNotFound {
                type_name: source_type.into(),
                object_id: source_id,
            });
        }

        let target_key = KeyBuilder::object(target_type_id, target_id);
        if self.storage.get(&txn, &target_key)?.is_none() {
            return Err(EngineError::ObjectNotFound {
                type_name: rel.target_type.clone(),
                object_id: target_id,
            });
        }

        let rel_key = format!("{source_type}.{field_name}");
        let rel_id = self.rel_ids[&rel_key];
        if self.tombstoned_rel_ids.contains(&rel_id) {
            return Err(EngineError::RelationRetired {
                type_name: source_type.into(),
                relation: field_name.into(),
                relation_id: rel_id,
                retired_at_unix_ms: self
                    .retired_at_ms_by_rel_id
                    .get(&rel_id)
                    .copied()
                    .unwrap_or(0),
            });
        }

        // Write edge + reverse edge.
        let edge_key = KeyBuilder::edge(source_id, rel_id, target_id);
        let rev_key = KeyBuilder::reverse_edge(target_id, rel_id, source_id);

        let edge_value = match edge_fields {
            Some(ef) => serialize_fields(&ef),
            None => Bytes::new(),
        };

        // Covering reverse-edge value: serialize the source's effective fields
        // — its explicit object fields, plus the THIS link's target, plus any
        // other forward 1:1 outbound targets already in the edge index.
        // Inverse-traversal fusion in the executor reads these to satisfy a
        // subsequent forward-1:1 hop without an extra prefix scan per source.
        //
        // We do this on the write path so the reverse-edge scan in the hot
        // executor path stays a single LSM operation. Cost: a few extra
        // prefix scans per link() call (one per OTHER forward 1:1 field on
        // the source type) — paid once at setup, amortized across every
        // subsequent traversal.
        let rev_value = build_covering_rev_value(
            self,
            &txn,
            source_type,
            source_id,
            source_data.as_deref(),
            field_name,
            target_id,
        )?;

        self.storage.put(&mut txn, &edge_key, edge_value)?;
        self.storage.put(&mut txn, &rev_key, rev_value)?;

        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        Ok(())
    }

    /// Remove a relationship between two objects.
    pub fn unlink(
        &self,
        source_type: &str,
        source_id: u64,
        field_name: &str,
        target_id: u64,
    ) -> EngineResult<()> {
        let _migration_guard = self.migration_lock.read();
        let _ = self.resolve_type_id(source_type)?;
        let rel_key = format!("{source_type}.{field_name}");
        let rel_id = *self
            .rel_ids
            .get(&rel_key)
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            })?;
        if self.tombstoned_rel_ids.contains(&rel_id) {
            return Err(EngineError::RelationRetired {
                type_name: source_type.into(),
                relation: field_name.into(),
                relation_id: rel_id,
                retired_at_unix_ms: self
                    .retired_at_ms_by_rel_id
                    .get(&rel_id)
                    .copied()
                    .unwrap_or(0),
            });
        }

        let mut txn = self.storage.begin_txn();

        let edge_key = KeyBuilder::edge(source_id, rel_id, target_id);
        let rev_key = KeyBuilder::reverse_edge(target_id, rel_id, source_id);

        self.storage.delete(&mut txn, &edge_key)?;
        self.storage.delete(&mut txn, &rev_key)?;

        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        Ok(())
    }

    /// Get all targets of a relationship from a source object.
    /// Returns (target_id, edge_fields) pairs.
    ///
    /// If the field has an @inverse directive, this transparently uses the
    /// reverse edge index of the referenced relationship.
    pub fn get_links(
        &self,
        source_type: &str,
        source_id: u64,
        field_name: &str,
    ) -> EngineResult<Vec<(u64, FieldMap)>> {
        let _ = self.resolve_type_id(source_type)?;
        let type_def = self
            .schema
            .get_type(source_type)
            .ok_or_else(|| EngineError::TypeNotFound(source_type.into()))?;

        let field = type_def.get_field(field_name).ok_or_else(|| {
            self.field_retired_error(source_type, field_name)
                .unwrap_or_else(|| EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                })
        })?;

        let snapshot = self.storage.read_snapshot();

        // If this field has @inverse, traverse via the reverse edge index
        // of the referenced relationship.
        if let Some(inv) = field.inverse() {
            let inv_rel_key = format!("{}.{}", inv.type_name, inv.field_name);
            let inv_rel_id =
                *self
                    .rel_ids
                    .get(&inv_rel_key)
                    .ok_or_else(|| EngineError::FieldNotFound {
                        type_name: inv.type_name.clone(),
                        field: inv.field_name.clone(),
                    })?;

            let prefix = KeyBuilder::reverse_edge_prefix(source_id, inv_rel_id);
            return self.scan_prefix_at(snapshot, &prefix);
        }

        // Direct forward edge scan.
        let rel_key = format!("{source_type}.{field_name}");
        let rel_id = *self
            .rel_ids
            .get(&rel_key)
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            })?;

        let prefix = KeyBuilder::edge_prefix(source_id, rel_id);
        self.scan_prefix_at(snapshot, &prefix)
    }

    /// Batched variant of `get_links`: walk the relation for N source IDs
    /// against ONE memtable/SST lock acquisition. Returns one inner Vec per
    /// input source id (same order), each holding the `(target_id,
    /// edge_fields)` pairs.
    ///
    /// At a single traversal hop this collapses N independent prefix-scan
    /// stacks into one — the per-source iteration still happens, but the
    /// surrounding lock dance and per-call schema/relation lookups are paid
    /// exactly once.
    pub fn get_links_many(
        &self,
        source_type: &str,
        source_ids: &[u64],
        field_name: &str,
    ) -> EngineResult<Vec<Vec<(u64, Bytes)>>> {
        if source_ids.is_empty() {
            return Ok(Vec::new());
        }

        let type_def = self
            .schema
            .get_type(source_type)
            .ok_or_else(|| EngineError::TypeNotFound(source_type.into()))?;

        let field = type_def
            .get_field(field_name)
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            })?;

        // Resolve the relation ID once (forward or inverse) — outside the
        // per-source loop the original get_links was paying.
        let (rel_id, use_inverse) = if let Some(inv) = field.inverse() {
            let inv_rel_key = format!("{}.{}", inv.type_name, inv.field_name);
            let id = *self
                .rel_ids
                .get(&inv_rel_key)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: inv.type_name.clone(),
                    field: inv.field_name.clone(),
                })?;
            (id, true)
        } else {
            let rel_key = format!("{source_type}.{field_name}");
            let id = *self
                .rel_ids
                .get(&rel_key)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                })?;
            (id, false)
        };

        // Build prefixes for the batch. Bytes::clone is cheap (Arc), but the
        // multi_scan API takes `&[u8]` so we keep owned buffers alive
        // alongside the slice views.
        let prefix_bufs: Vec<_> = source_ids
            .iter()
            .map(|sid| {
                if use_inverse {
                    KeyBuilder::reverse_edge_prefix(*sid, rel_id)
                } else {
                    KeyBuilder::edge_prefix(*sid, rel_id)
                }
            })
            .collect();
        let prefix_refs: Vec<&[u8]> = prefix_bufs.iter().map(|p| p.as_ref()).collect();

        let snapshot = self.storage.read_snapshot();
        let raw = self.storage.multi_scan_prefix_at(snapshot, &prefix_refs)?;

        // Return raw (id, value-bytes) pairs — no FieldMap construction. The
        // executor either hands the bytes to the fusion path's
        // `find_u64_field_in_raw` (forward-1:1 next hop) or to
        // `deserialize_fields` lazily (Filter / terminal materialize). For
        // 2-hop at 1M ratings this avoids ~1000 FieldMap allocations per
        // query.
        Ok(raw
            .into_iter()
            .map(|entries| {
                let mut out = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    if key.len() < 8 {
                        continue;
                    }
                    let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
                    out.push((u64::from_be_bytes(id_bytes), value));
                }
                out
            })
            .collect())
    }

    /// Scan for keys with a given prefix within a transaction (used by
    /// write paths that need to see uncommitted state).
    fn scan_prefix(
        &self,
        txn: &rhypedb_storage::mvcc::Transaction,
        prefix: &[u8],
    ) -> EngineResult<Vec<(u64, FieldMap)>> {
        let entries = self.storage.scan_prefix(txn, prefix)?;
        Ok(Self::decode_edge_entries(entries))
    }

    /// Like `scan_prefix` but returns the raw entry value bytes instead of
    /// a decoded `FieldMap`. Used by cascade delete: each rev_edge value
    /// carries the source object's covering blob, and the cascade only
    /// needs to pull a few `u64` fields out of it via the byte-level
    /// `find_u64_field_in_raw` helper — way cheaper than running
    /// `deserialize_fields` on every blob just to read two u64s.
    fn scan_prefix_raw(
        &self,
        txn: &rhypedb_storage::mvcc::Transaction,
        prefix: &[u8],
    ) -> EngineResult<Vec<(u64, Bytes)>> {
        let entries = self.storage.scan_prefix(txn, prefix)?;
        let mut out = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            out.push((u64::from_be_bytes(id_bytes), value));
        }
        Ok(out)
    }

    /// Scan for keys with a given prefix at a snapshot version (used by the
    /// read-only fast path).
    fn scan_prefix_at(&self, snapshot: u64, prefix: &[u8]) -> EngineResult<Vec<(u64, FieldMap)>> {
        let entries = self.storage.scan_prefix_at(snapshot, prefix)?;
        Ok(Self::decode_edge_entries(entries))
    }

    /// Decode (user_key, value) pairs into (extracted_id, edge_fields).
    /// The ID is the last 8 bytes of the user key.
    fn decode_edge_entries(entries: Vec<(Bytes, Bytes)>) -> Vec<(u64, FieldMap)> {
        let mut results = Vec::new();
        for (key, value) in entries {
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let id = u64::from_be_bytes(id_bytes);

            let edge_fields = if value.is_empty() {
                FieldMap::new()
            } else {
                deserialize_fields(&value)
            };

            results.push((id, edge_fields));
        }
        results
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn subscriptions(&self) -> &SubscriptionHub {
        &self.subscriptions
    }

    /// Clone the `Arc<SubscriptionHub>` so a long-lived consumer (e.g. a network
    /// connection) can hold the hub directly. The hub `Arc` is preserved verbatim
    /// across a hot-reload (`reload_handle`/`clone_into_new_handle` carry the same
    /// `Arc`), so a subscription registered on the returned handle keeps receiving
    /// events from post-reload commits.
    pub fn subscriptions_arc(&self) -> Arc<SubscriptionHub> {
        Arc::clone(&self.subscriptions)
    }

    pub fn storage(&self) -> &Arc<LsmTree> {
        &self.storage
    }

    /// Take a consistent physical backup of this database into `dst` (a fresh,
    /// empty directory): the LSM SSTs + WAL (via [`LsmTree::snapshot_to`], which
    /// flushes first and holds the right locks) plus a copy of every
    /// `hnsw_*.bin` vector-index snapshot. The result is openable via
    /// `Database::open(<the schema this DB was opened with>, dst)`.
    ///
    /// Two things are deliberately the CALLER's job: (1) freshening the HNSW
    /// snapshots — the server owns the vectorizer and calls `save_snapshots()`
    /// first; here the `.bin` files are copied as-is and a missing/stale one is
    /// rebuilt from the LSM on open. (2) Writing the on-disk `MANIFEST.json` —
    /// the engine stays serde-free, so the caller serializes the returned
    /// [`BackupManifest`] and writes it LAST, its presence marking the backup
    /// complete.
    pub fn backup_to(&self, dst: &std::path::Path) -> EngineResult<BackupManifest> {
        // SSTs + WAL — the consistent, load-bearing part.
        let sst = self.storage.snapshot_to(dst)?;

        // Copy each hnsw_*.bin (skip the *.bin.tmp the writer renames from).
        let data_dir = self.storage.data_dir();
        let mut hnsw_files = Vec::new();
        for entry in std::fs::read_dir(data_dir).map_err(rhypedb_storage::Error::Io)? {
            let entry = entry.map_err(rhypedb_storage::Error::Io)?;
            let fname = entry.file_name();
            let name = fname.to_string_lossy();
            if name.starts_with("hnsw_") && name.ends_with(".bin") {
                std::fs::copy(entry.path(), dst.join(&*name))
                    .map_err(rhypedb_storage::Error::Io)?;
                hnsw_files.push(name.into_owned());
            }
        }

        // In-flight (non-terminal) migrations + their converter names — so an
        // operator restoring a mid-migration backup knows it will auto-resume
        // and which converters must be registered first.
        let in_flight_migrations = self
            .list_migrations()?
            .into_iter()
            .filter(|m| !m.status.is_terminal())
            .map(|m| (m.plan_id, m.converter_name))
            .collect();

        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(BackupManifest {
            sst_names: sst.sst_names,
            max_version: sst.max_version,
            wal_bytes: sst.wal_bytes,
            hnsw_files,
            in_flight_migrations,
            created_at_ms,
        })
    }

    /// Stream a portable, version-independent **logical** dump of this database
    /// to `writer` as NDJSON. Unlike [`backup_to`](Self::backup_to) (which
    /// hard-links the on-disk SSTs/WAL and is tied to the storage format), this
    /// reads data out through the object layer into a self-describing stream
    /// that survives format/version changes and can be inspected by hand.
    ///
    /// Line order is fixed and dependency-safe so a single forward pass can
    /// import it: `header` → `schema` → every type's `object` lines → every
    /// type's forward `edge` lines → every type's raw `vector` lines →
    /// `trailer`. The trailer (with per-type counts and `complete:true`) is
    /// written LAST and is the completeness sentinel — NDJSON has no
    /// end-of-archive marker, so a truncated download is detected by its
    /// absence.
    ///
    /// The whole dump is read at ONE pinned MVCC snapshot. It is refused while
    /// a field-type migration is in flight (stored values diverge from the
    /// declared schema mid-migration). Memory is bounded: objects, edges, and
    /// vectors all stream in chunks; nothing materializes a whole type.
    ///
    /// The embedded schema is the LIVE [`schema`](Self::schema) rendered to SDL
    /// — never the operator's on-disk file, which goes stale after a completed
    /// rename/change-type migration.
    pub fn logical_export_stream(
        &self,
        writer: &mut dyn std::io::Write,
        opts: &crate::logical::LogicalExportOptions,
    ) -> EngineResult<crate::logical::LogicalExportSummary> {
        // Refuse mid field-type migration: stored values diverge from the
        // declared schema until cutover completes, so the dump would be
        // internally inconsistent. The server maps this to HTTP 409.
        let migrating = self.migrating_field_count.load(Ordering::SeqCst);
        if migrating > 0 {
            return Err(EngineError::ExportWhileMigrating {
                migrating_fields: migrating,
            });
        }

        // Resolve + validate the type set, in a deterministic sorted order.
        let mut type_names: Vec<String> = match &opts.types {
            Some(list) if !list.is_empty() => {
                for t in list {
                    if self.schema.get_type(t).is_none() {
                        return Err(EngineError::TypeNotFound(t.clone()));
                    }
                }
                list.clone()
            }
            _ => self.schema.type_names().map(|s| s.to_owned()).collect(),
        };
        type_names.sort();
        type_names.dedup();

        // Pin a REGISTERED read snapshot for the whole dump. begin_txn() inserts
        // it into the MVCC active set, so a concurrent flush+compaction cannot GC
        // versions visible at this snapshot out from under the long-running,
        // multi-pass export and silently drop objects from a dump still marked
        // complete:true. read_snapshot() only reads current_version() and is NOT
        // registered, so it would not hold compaction back. Transaction's Drop
        // does NOT unregister — abort() must run on every path, so the fallible
        // body lives in export_sections and we abort unconditionally after it.
        let mut txn = self.storage.begin_txn();
        let snapshot = txn.snapshot();
        let result = self.export_sections(writer, opts, &type_names, snapshot);
        self.storage.abort(&mut txn);
        result
    }

    /// Write the header → schema → objects → edges → vectors → trailer sections
    /// of a logical export at the caller-pinned `snapshot`. Split out from
    /// [`logical_export_stream`](Self::logical_export_stream) so its snapshot
    /// transaction is reliably aborted on every return path — including the `?`
    /// early-returns when the writer (a disconnected client, a full disk) fails.
    fn export_sections(
        &self,
        writer: &mut dyn std::io::Write,
        opts: &crate::logical::LogicalExportOptions,
        type_names: &[String],
        snapshot: u64,
    ) -> EngineResult<crate::logical::LogicalExportSummary> {
        use crate::logical::{self, TypeCounts, VectorMode};

        let included: std::collections::HashSet<&str> =
            type_names.iter().map(String::as_str).collect();
        let chunk_size = LOGICAL_EXPORT_CHUNK_SIZE;

        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let vectors_tag = match opts.vectors {
            VectorMode::Raw => "raw",
            VectorMode::None => "none",
            VectorMode::Reembed => "reembed",
        };

        // (1) HEADER.
        write_export_line(
            writer,
            &serde_json::json!({
                "kind": "header",
                "format": logical::FORMAT_TAG,
                "created_at_ms": created_at_ms,
                "export_version": snapshot,
                "vectors": vectors_tag,
                "types": type_names,
            }),
        )?;

        // (2) SCHEMA — the LIVE schema rendered to SDL.
        write_export_line(
            writer,
            &serde_json::json!({
                "kind": "schema",
                "sdl": rhypedb_schema::emit_schema(&self.schema),
            }),
        )?;

        let mut counts: std::collections::BTreeMap<String, TypeCounts> =
            std::collections::BTreeMap::new();
        for t in type_names {
            counts.insert(t.clone(), TypeCounts::default());
        }

        // (3) OBJECT lines — every type, ascending object id, scalars only.
        for type_name in type_names {
            let mut cursor = 0u64;
            loop {
                let chunk = self.scan_chunk(type_name, snapshot, cursor, chunk_size)?;
                for obj in &chunk.objects {
                    write_export_line(
                        writer,
                        &serde_json::json!({
                            "kind": "object",
                            "type": type_name,
                            "id": obj.id.to_string(),
                            "fields": logical::fields_to_json(&obj.fields),
                        }),
                    )?;
                    counts.get_mut(type_name).unwrap().objects += 1;
                }
                match chunk.next_cursor {
                    Some(next) if chunk.more => cursor = next,
                    _ => break,
                }
            }
        }

        // (4) EDGE lines — forward (non-@inverse) relations only, after ALL
        // objects so both endpoints exist on a single-pass import. @inverse
        // fields are the reverse view of another type's forward edge and would
        // duplicate them. Read at the pinned snapshot (not get_links, which
        // takes its own).
        for type_name in type_names {
            let type_def = self.schema.get_type(type_name).unwrap();
            // (field name, rel_id, target-type-included?) for each forward rel.
            let forward: Vec<(&str, u64, bool)> = type_def
                .fields
                .iter()
                .filter(|f| {
                    matches!(f.field_type, FieldType::Relation(_)) && f.inverse().is_none()
                })
                .filter_map(|f| {
                    let rel_id = *self.rel_ids.get(&format!("{type_name}.{}", f.name))?;
                    let FieldType::Relation(rel) = &f.field_type else {
                        return None;
                    };
                    Some((f.name.as_str(), rel_id, included.contains(rel.target_type.as_str())))
                })
                .collect();
            if forward.is_empty() {
                continue;
            }

            let mut cursor = 0u64;
            loop {
                let chunk = self.scan_chunk(type_name, snapshot, cursor, chunk_size)?;
                for obj in &chunk.objects {
                    for (field_name, rel_id, target_included) in &forward {
                        let prefix = KeyBuilder::edge_prefix(obj.id, *rel_id);
                        let links = self.scan_prefix_at(snapshot, &prefix)?;
                        if links.is_empty() {
                            continue;
                        }
                        if !target_included {
                            counts.get_mut(type_name).unwrap().dangling_edges_skipped +=
                                links.len() as u64;
                            continue;
                        }
                        for (dst, edge_fields) in links {
                            write_export_line(
                                writer,
                                &serde_json::json!({
                                    "kind": "edge",
                                    "type": type_name,
                                    "src": obj.id.to_string(),
                                    "field": field_name,
                                    "dst": dst.to_string(),
                                    "edge_fields": logical::fields_to_json(&edge_fields),
                                }),
                            )?;
                            counts.get_mut(type_name).unwrap().edges += 1;
                        }
                    }
                }
                match chunk.next_cursor {
                    Some(next) if chunk.more => cursor = next,
                    _ => break,
                }
            }
        }

        // (5) VECTOR lines (Raw or Reembed) — streamed over the v:<type_id>:
        // keyspace. The stored value is already big-endian f32, shipped
        // verbatim as base64; import decodes it and rebuilds the HNSW graph.
        // In Reembed mode the @vectorize fields are omitted (regenerated from
        // source text on import); BYO vector fields still ship raw.
        if opts.vectors != VectorMode::None {
            for type_name in type_names {
                let type_def = self.schema.get_type(type_name).unwrap();
                // field_id -> field name for this type's LIVE vector fields;
                // leftover vectors of a retired field are skipped (no field to
                // import them into).
                let mut vec_field_names: HashMap<u64, &str> = HashMap::new();
                for f in type_def.vector_fields() {
                    // Reembed: skip @vectorize fields (regenerated on import);
                    // a BYO field has no source text and must still ship raw.
                    if opts.vectors == VectorMode::Reembed && f.vectorize().is_some() {
                        continue;
                    }
                    if let Some(fid) = self.field_ids.get(&format!("{type_name}.{}", f.name)) {
                        vec_field_names.insert(*fid, f.name.as_str());
                    }
                }
                if vec_field_names.is_empty() {
                    continue;
                }

                let type_id = self.resolve_type_id(type_name)?;
                let v_prefix = KeyBuilder::vector_prefix(type_id);
                let mut start = v_prefix.clone();
                loop {
                    let chunk =
                        self.storage
                            .scan_chunk_raw(snapshot, &v_prefix, &start, chunk_size)?;
                    for (key, value) in &chunk.live {
                        // v:<type_id>:<object_id>:<field_id> — field_id is the
                        // trailing 8 bytes; object_id the 8 before its separator.
                        if key.len() < 17 {
                            continue;
                        }
                        let field_id =
                            u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap());
                        let Some(field_name) = vec_field_names.get(&field_id) else {
                            continue;
                        };
                        let object_id = u64::from_be_bytes(
                            key[key.len() - 17..key.len() - 9].try_into().unwrap(),
                        );
                        write_export_line(
                            writer,
                            &serde_json::json!({
                                "kind": "vector",
                                "type": type_name,
                                "id": object_id.to_string(),
                                "field": field_name,
                                "dims": value.len() / 4,
                                "f32": logical::encode_bytes(value),
                            }),
                        )?;
                        counts.get_mut(type_name).unwrap().vectors += 1;
                    }
                    match &chunk.high_water {
                        Some(hw) if chunk.more => start = successor_key(hw),
                        _ => break,
                    }
                }
            }
        }

        // (6) TRAILER — last line, the completeness sentinel.
        let mut counts_obj = serde_json::Map::new();
        for (t, c) in &counts {
            counts_obj.insert(
                t.clone(),
                serde_json::json!({
                    "objects": c.objects,
                    "edges": c.edges,
                    "vectors": c.vectors,
                    "dangling_edges_skipped": c.dangling_edges_skipped,
                }),
            );
        }
        write_export_line(
            writer,
            &serde_json::json!({
                "kind": "trailer",
                "complete": true,
                "counts": serde_json::Value::Object(counts_obj),
            }),
        )?;
        writer.flush().map_err(export_io_err)?;

        Ok(crate::logical::LogicalExportSummary { counts })
    }

    pub fn type_ids(&self) -> &HashMap<String, u64> {
        &self.type_ids
    }

    pub fn field_ids(&self) -> &HashMap<String, u64> {
        &self.field_ids
    }

    pub fn rel_ids(&self) -> &HashMap<String, u64> {
        &self.rel_ids
    }

    /// Number of fields currently undergoing a field-type migration. Non-zero
    /// means stored values may diverge from the declared schema, so a logical
    /// export is refused (see [`logical_export_stream`](Self::logical_export_stream)).
    /// A cheap lock-free read; lets the server pre-flight a clean 409 before
    /// committing to a streamed 200 response.
    pub fn migrating_fields(&self) -> usize {
        self.migrating_field_count.load(Ordering::SeqCst)
    }

    /// Check that a unique value doesn't already exist, and insert the index entry.
    ///
    /// `staged` records every unique key this transaction has already claimed.
    /// It is required because a buffered `put` is invisible to `storage.get`
    /// (reads resolve at the txn snapshot; the write buffer is write-only), so
    /// without it a second row in the SAME `create_batch` carrying a duplicate
    /// `@unique` value would slip past the committed-data check and both rows
    /// would commit. Callers driving a single object pass a throwaway map.
    #[allow(clippy::too_many_arguments)]
    fn check_unique_and_insert(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        type_id: u64,
        field_name: &str,
        value: &Value,
        object_id: u64,
        staged: &mut HashMap<Bytes, u64>,
    ) -> EngineResult<()> {
        let field_key = format!("{type_name}.{field_name}");
        let field_id = self.field_ids[&field_key];
        let value_bytes = value_to_index_bytes(value);
        let unique_key = KeyBuilder::unique_index(type_id, field_id, &value_bytes);

        let violation = || EngineError::UniqueViolation {
            type_name: type_name.into(),
            field: field_name.into(),
            value: value.to_string(),
        };

        // Intra-txn check FIRST: catch a duplicate value staged by an earlier
        // row in this same transaction, which the committed-data probe below
        // cannot see (the txn write buffer is write-only).
        if let Some(&staged_id) = staged.get(&unique_key)
            && staged_id != object_id
        {
            return Err(violation());
        }

        if let Some(existing) = self.storage.get(txn, &unique_key)? {
            let existing_id = u64::from_be_bytes(existing[..8].try_into().unwrap());
            if existing_id != object_id {
                return Err(violation());
            }
        }

        let mut id_buf = bytes::BytesMut::with_capacity(8);
        bytes::BufMut::put_u64(&mut id_buf, object_id);
        self.storage.put(txn, &unique_key, id_buf.freeze())?;
        staged.insert(unique_key, object_id);

        Ok(())
    }

    /// Insert a non-unique secondary index entry. The on-disk shape depends
    /// on `kind`:
    ///
    ///   * `Integer` → fixed-width `i:<type>:<field>:<8-byte encoded>:<id>`.
    ///   * `String`  → variable-length `i:<type>:<field>:<escaped>\x00\x00<id>`.
    ///
    /// The `covering` bytes get stored as the entry value — when non-empty,
    /// this is the source object's serialized FieldMap, which lets
    /// `filter_scan_via_index` reconstruct Objects without an extra `get_many`
    /// probe per match. Pass `Bytes::new()` for legacy / non-covering mode.
    ///
    /// Caller must hold a write txn. Silently no-ops on value-kind mismatches
    /// (e.g. null, or wrong scalar type) — there's nothing to index.
    #[allow(clippy::too_many_arguments)]
    fn insert_field_index(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_id: u64,
        field_id: u64,
        kind: IndexedKind,
        value: &Value,
        object_id: u64,
        covering: Bytes,
    ) -> EngineResult<()> {
        if let Some(key) = build_field_index_key(type_id, field_id, kind, value, object_id) {
            self.storage.put(txn, &key, covering)?;
        }
        Ok(())
    }

    /// Remove a secondary index entry. Same dispatch as `insert_field_index`.
    fn remove_field_index(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_id: u64,
        field_id: u64,
        kind: IndexedKind,
        value: &Value,
        object_id: u64,
    ) -> EngineResult<()> {
        if let Some(key) = build_field_index_key(type_id, field_id, kind, value, object_id) {
            self.storage.delete(txn, &key)?;
        }
        Ok(())
    }

    /// Remove a unique index entry.
    fn remove_unique_index(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        type_id: u64,
        field_name: &str,
        value: &Value,
    ) -> EngineResult<()> {
        let field_key = format!("{type_name}.{field_name}");
        let field_id = self.field_ids[&field_key];
        let value_bytes = value_to_index_bytes(value);
        let unique_key = KeyBuilder::unique_index(type_id, field_id, &value_bytes);
        self.storage.delete(txn, &unique_key)?;
        Ok(())
    }

    /// Rewrite every outbound rev_edge whose source is `(type_name, object_id)`.
    /// For each forward (non-inverse) relation, walk every linked target in
    /// the edge index and `put` a freshly built `build_covering_rev_value`
    /// payload at `r:<target>:<rel>:<source>`. This is the Phase 1 work
    /// `update()` does for the source-side cover refresh; the cover-refresh
    /// worker calls the same code to repair stale covers in OTHER sources
    /// after a target's data changes.
    ///
    /// `source_data` provides the source object's effective scalar bytes —
    /// `Some(b)` for `update()` (where the new bytes haven't been committed
    /// yet) and the current persisted bytes for the worker path.
    fn refresh_outbound_rev_edges(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        object_id: u64,
        source_data: Option<&[u8]>,
    ) -> EngineResult<()> {
        let Some(type_def) = self.schema.get_type(type_name) else {
            return Ok(());
        };
        let has_forward_1to1 = type_def.fields.iter().any(|f| {
            matches!(&f.field_type, FieldType::Relation(rel) if !rel.is_many)
                && f.inverse().is_none()
        });
        if !has_forward_1to1 {
            return Ok(());
        }
        for field in &type_def.fields {
            if !matches!(field.field_type, FieldType::Relation(_)) {
                continue;
            }
            if field.inverse().is_some() {
                continue;
            }
            let rel_key = format!("{type_name}.{}", field.name);
            let Some(&rel_id) = self.rel_ids.get(&rel_key) else {
                continue;
            };
            let prefix = KeyBuilder::edge_prefix(object_id, rel_id);
            let entries = self.scan_prefix(txn, &prefix)?;
            for (target_id, _edge_value) in entries {
                let rev_value = build_covering_rev_value(
                    self,
                    txn,
                    type_name,
                    object_id,
                    source_data,
                    &field.name,
                    target_id,
                )?;
                let rev_key = KeyBuilder::reverse_edge(target_id, rel_id, object_id);
                self.storage.put(txn, &rev_key, rev_value)?;
            }
        }
        Ok(())
    }

    /// Refresh every stale embedded cover whose underlying target is
    /// `(target_type_id, target_id)`. Used by the cover-refresh worker after
    /// a target update has bumped its generation.
    ///
    /// Algorithm:
    ///   1. Scan every incoming 1:1 forward rev_edge of the target —
    ///      `r:<target>:<rel>:*` for each relation listed in
    ///      `incoming_relations` whose source field is 1:1. Each match gives
    ///      us a source object S that has the target embedded as one of S's
    ///      other-target covers.
    ///   2. For each S, re-run Phase 1 — `refresh_outbound_rev_edges` —
    ///      which rewrites every outbound rev_edge of S with fresh covers
    ///      pulled from each peer's current state (including the target's
    ///      newly written data + bumped generation).
    ///
    /// All rewrites happen in one txn so partial work can't make the index
    /// temporarily inconsistent. A commit failure (write conflict) is
    /// surfaced; the next bump re-enqueues the target so retry is cheap.
    ///
    /// Card 2: takes `migration_lock.read()` for the whole pass. This worker
    /// runs on a background thread and writes rev-edge cover blobs that embed
    /// other objects' fields + a `<name>__cover_v` stamp from the LIVE
    /// generation counter — neither of which the MVCC write-set conflict check
    /// would order against an in-flight cutover (different keyspaces). Without
    /// the lock, a refresh racing `run_cutover` (which holds
    /// `migration_lock.write()`) could (a) bake a `<field>__shadow` into a cover
    /// if the migration disarms between the blob read and the strip decision, or
    /// (b) stamp `cover_v` = the post-bump generation onto a stale (pre-cutover)
    /// blob, defeating the cutover's generation-bump invalidation. The read
    /// guard makes the cutover's write pass mutually exclude this worker. Safe:
    /// only the worker calls this (the enqueue at `update()` is async), so the
    /// guard is never re-entrant.
    fn refresh_covers_for_target(&self, target_type_id: u64, target_id: u64) -> EngineResult<()> {
        let _migration_guard = self.migration_lock.read();
        let Some(incoming) = self.incoming_relations.get(&target_type_id) else {
            return Ok(());
        };

        // Phase A: read pass — find every (S_type_id, S_id) that links to
        // the target via a 1:1 forward relation. Done in a read-only scan
        // outside the write txn so the prefix scan doesn't see in-flight
        // writes from this same worker pass.
        let read_txn = self.storage.begin_txn();
        let mut sources: Vec<(u64, u64)> = Vec::new();
        for inc in incoming {
            if inc.is_many {
                continue;
            }
            let rev_prefix = KeyBuilder::reverse_edge_prefix(target_id, inc.rel_id);
            let entries = self.scan_prefix_raw(&read_txn, &rev_prefix)?;
            for (source_id, _value) in entries {
                sources.push((inc.source_type_id, source_id));
            }
        }
        drop(read_txn);

        if sources.is_empty() {
            return Ok(());
        }

        // Phase B: write pass — for each source, re-run Phase 1 in a fresh
        // txn so commit is atomic. We collect the source bytes inside the
        // same txn that does the put so a concurrent S-update doesn't get
        // overwritten with stale cover content.
        let mut txn = self.storage.begin_txn();
        for (s_type_id, s_id) in sources {
            let Some(s_type_name) = self.type_name_by_id.get(&s_type_id) else {
                continue;
            };
            let obj_key = KeyBuilder::object(s_type_id, s_id);
            let Some(s_bytes) = self.storage.get(&txn, &obj_key)? else {
                continue;
            };
            let s_type_name = s_type_name.clone();
            self.refresh_outbound_rev_edges(&mut txn, &s_type_name, s_id, Some(&s_bytes))?;
        }
        self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;
        Ok(())
    }
}

impl Drop for Database {
    /// Tear down the cover-refresh worker. Dropping the sender closes the
    /// channel — the worker's blocked `recv()` returns `Err`, the loop
    /// breaks, the thread exits. We then join so this `Drop` doesn't
    /// return while the worker is still touching the LSM.
    ///
    /// Self-join guard: the worker temporarily holds an `Arc<Database>`
    /// during each iteration (via `Weak::upgrade`). If the external
    /// holder drops their `Arc` while the worker is processing, the
    /// worker's local Arc can become the last one — its end-of-scope
    /// drop fires `Database::drop` on the worker thread itself, where
    /// `handle.join()` would deadlock against the current thread. In
    /// that case the worker is already finishing its iteration and will
    /// exit naturally on the next `rx.recv()` (which now returns `Err`
    /// because we just dropped the sender), so we just skip the join.
    fn drop(&mut self) {
        *self.cover_refresh_tx.lock() = None;
        if let Some(handle) = self.cover_refresh_handle.lock().take()
            && handle.thread().id() != std::thread::current().id()
        {
            let _ = handle.join();
        }

        // Stop in-flight migration drivers (card 3/5). Drain the registry under
        // the lock, RELEASE it, then signal PAUSE + join — never hold the
        // registry lock across a join (a driver self-deregisters under that same
        // lock on its normal exit). PAUSE (not CANCEL) = "stop at a chunk
        // boundary, leave the plan resumable"; the next open's auto-resume
        // re-drives it. `Drop` runs only at refcount 0, so every `weak.upgrade`
        // inside a driver returns `None` now — a driver cannot re-enter the
        // database or take the registry lock, so there is no cycle. The same
        // self-join-skip guard the cover-refresh worker uses covers the case
        // where the LAST `Arc` was a driver's transient cutover upgrade, so this
        // `Drop` is itself running on a driver thread.
        let drivers: Vec<MigrationDriver> =
            std::mem::take(&mut *self.migration_drivers.lock())
                .into_values()
                .collect();
        for d in &drivers {
            d.control.store(
                crate::catalog::migration_control::PAUSE,
                std::sync::atomic::Ordering::SeqCst,
            );
        }
        for d in drivers {
            if let Some(handle) = d.handle
                && handle.thread().id() != std::thread::current().id()
            {
                let _ = handle.join();
            }
        }
    }
}

/// RAII cleanup for an INLINE migration drive (resume / auto-resume, card 3/5).
/// On drop — EVERY exit path of `drive_migration_to_completion`'s parallel
/// branch, including `?` propagation and a panic unwind — it wakes any
/// `wait_for_migration` waiter and removes the registry entry. (The async create
/// driver does this manually in `migration_driver_main`, since it holds only a
/// `Weak<Database>`.) The inline drive runs under the operator's `Arc<Database>`,
/// so `Database::drop` cannot run until the drive returns and the guard fires —
/// no race against the registry drain.
struct InlineDriveGuard<'a> {
    db: &'a Database,
    plan_id: u64,
    signal: Arc<MigrationSignal>,
}

impl Drop for InlineDriveGuard<'_> {
    fn drop(&mut self) {
        // Inline drive surfaces its error through the normal return path, so the
        // signal carries no error here — just mark finished + wake waiters, then
        // remove the entry. Both run on the calling thread, so there is no
        // separate thread to join.
        self.signal.mark_done(None);
        self.db.migration_drivers.lock().remove(&self.plan_id);
    }
}

/// Detached ASYNC migration driver entrypoint (shadow-field card 3/5). Owns the
/// `N` partition workers for one plan's Converting phase, then the
/// single-threaded cutover. Holds a `Weak<Database>` (+ a storage `Arc`) — it
/// upgrades to a strong `Arc<Database>` ONLY transiently for the cutover, so it
/// never extends the database's lifetime: if external `Arc`s drop mid-backfill,
/// `Database::drop` stores PAUSE + joins this thread, the cutover upgrade returns
/// `None`, and the plan is left resumable for the next open's auto-resume.
///
/// Does NOT touch the registry (it never removes its own entry) — it just
/// `mark_done`s the signal and returns, leaving the still-joinable handle for
/// `wait_for_migration` / `Database::drop` to join. The whole drive runs under a
/// `catch_unwind` so a worker/cutover panic can't skip the `mark_done`
/// (parking_lot locks don't poison, so an unwound `run_cutover` releases
/// `migration_lock` cleanly).
#[allow(clippy::too_many_arguments)]
fn migration_driver_main(
    weak: std::sync::Weak<Database>,
    storage: Arc<LsmTree>,
    converter: crate::catalog::RegisteredConverter,
    control: Arc<std::sync::atomic::AtomicU8>,
    signal: Arc<MigrationSignal>,
    plan_id: u64,
    type_id: u64,
    events: Arc<MigrationEventHub>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drive_async_migration(&weak, &storage, &converter, &control, plan_id, type_id, &events)
    }));
    // `Ok(Err(e))` = a parked Failed terminal (surfaced to the first waiter);
    // `Ok(Ok(()))` = completed or paused (both clean); `Err(_)` = a caught panic
    // (the durable plan status is the source of truth — no EngineError to carry).
    let error = match result {
        Ok(Err(e)) => Some(e),
        _ => None,
    };
    // Card 5: surface a terminal driver error on the event stream (the plan was
    // already parked Failed by run_parallel_backfill / run_cutover).
    if let Some(e) = &error {
        events.publish(MigrationEvent::Failed {
            plan_id,
            message: e.to_string(),
        });
    }
    signal.mark_done(error);
}

/// The async driver's drive body: backfill (Converting) then cutover. Storage-
/// only for the backfill (never needs the `Database`); upgrades the `Weak` just
/// for the cutover. Returns the terminal error (which `run_parallel_backfill` /
/// `run_cutover` have ALREADY parked the plan `Failed` for) so the driver can
/// surface it to a `wait_for_migration` waiter; a clean completion, a pause, or
/// a dropped-DB upgrade-`None` all return `Ok(())` (the plan is left for the next
/// open in the drop case).
fn drive_async_migration(
    weak: &std::sync::Weak<Database>,
    storage: &LsmTree,
    converter: &crate::catalog::RegisteredConverter,
    control: &std::sync::atomic::AtomicU8,
    plan_id: u64,
    type_id: u64,
    events: &MigrationEventHub,
) -> EngineResult<()> {
    let plan = {
        let txn = storage.begin_txn();
        match crate::catalog::load_migration_plan(storage, &txn, plan_id) {
            Ok(p) => p,
            Err(_) => return Ok(()), // plan vanished — nothing to drive
        }
    };
    let mut disp_paused = false;
    let mut backfill_err: Option<EngineError> = None;
    if plan.phase == crate::catalog::MigrationPhase::Converting {
        let n = plan.parallel_degree.unwrap_or(1).max(1);
        // Capture the result WITHOUT `?` — a cancel that landed mid-backfill must
        // roll back even if a worker errored first (Stop policy / I/O).
        match crate::catalog::run_parallel_backfill(
            storage,
            plan_id,
            type_id,
            n,
            plan.id_upper_bound,
            &plan.type_name,
            &plan.field_name,
            plan.src_kind,
            plan.target_kind,
            plan.converter_version,
            plan.chunk_size,
            converter,
            control,
            plan.error_policy,
            plan.dry_run,
            plan.quarantine_cap,
            &plan.converter_name,
            Some(events),
        ) {
            Ok(crate::catalog::BackfillDisposition::AllDone) => {}
            Ok(crate::catalog::BackfillDisposition::Paused) => disp_paused = true,
            Err(e) => backfill_err = Some(e),
        }
    }
    // B8 (card 5): re-load AFTER the backfill, BEFORE propagating any error. A
    // `cancel_migration` that landed while the workers were stopping (or that
    // raced the AllDone→cutover handoff, or that beat a Stop-policy worker error
    // that parked the plan Failed) has durably flipped the plan to RollingBack.
    // The rollback completes and supersedes a backfill error (the partial state
    // is being discarded anyway); cutover never runs on a cancelled plan.
    let post = {
        let txn = storage.begin_txn();
        match crate::catalog::load_migration_plan(storage, &txn, plan_id) {
            Ok(p) => p,
            Err(_) => return Ok(()),
        }
    };
    if post.phase == crate::catalog::MigrationPhase::RollingBack {
        // Needs the `Database` (cover/index maintenance). If it is dropping, the
        // next open's auto-resume finishes the rollback.
        if let Some(db) = weak.upgrade() {
            db.run_terminal_pass(plan_id, type_id)?;
        }
        return Ok(());
    }
    // No cancel → surface a genuine backfill error (already parked Failed) now.
    if let Some(e) = backfill_err {
        return Err(e);
    }
    if disp_paused {
        return Ok(()); // genuine pause — resumable
    }
    // A dry-run preflight cleans up + marks DryRunCompleted (storage-only — no
    // hook was armed, no cutover).
    if plan.dry_run {
        crate::catalog::finalize_dry_run(storage, plan_id)?;
        events.publish(MigrationEvent::StatusChanged {
            plan_id,
            status: crate::catalog::MigrationStatus::DryRunCompleted,
        });
        return Ok(());
    }
    // Single-threaded cutover — skip if the DB is dropping (the next open's
    // auto-resume finishes it).
    if let Some(db) = weak.upgrade() {
        db.run_terminal_pass(plan_id, type_id)?;
    }
    Ok(())
}

/// Cover-refresh worker. Loops on the per-database channel, popping each
/// target `(type_id, object_id)` that just had its generation bumped and
/// asking the database to repair every embedded cover for that target.
///
/// Holds a `Weak<Database>` so the database's lifetime isn't extended by
/// this thread — when external `Arc<Database>` refs all drop, our upgrade
/// returns `None` and we exit. The channel-close signal from `Drop` covers
/// the case where the worker is blocked in `recv()` at that moment.
fn cover_refresh_worker(
    rx: std::sync::mpsc::Receiver<(u64, u64)>,
    weak: std::sync::Weak<Database>,
) {
    while let Ok((type_id, object_id)) = rx.recv() {
        let Some(db) = weak.upgrade() else { break };
        // Best-effort: a refresh failure (write conflict, IO error) is
        // self-healing. The next update on the target re-enqueues it; the
        // reader fall-through continues to detect staleness via cover_v
        // in the meantime.
        let _ = db.refresh_covers_for_target(type_id, object_id);
    }
}

/// Build the serialized FieldMap that gets stored as the reverse-edge entry's
/// value. The map contains:
///   * the source object's explicit fields (deserialized from its blob), and
///   * `field_name → Value::U64(target_id)` for the link being written, and
///   * one entry per OTHER forward 1:1 field on the source type whose target
///     can be found in the current edge index — discovered via a per-field
///     prefix scan inside the same transaction.
///
/// The cost is a few extra (cheap) edge prefix scans per link() call. The
/// payoff is that an inverse traversal can now satisfy a subsequent
/// forward-1:1 hop without scanning the forward edge index at all.
/// Build a rev_edge covering value from in-memory state — used by the
/// inline-relations create path where every forward 1:1 target's data is
/// already on hand from the txn's existence checks. Mirrors what
/// `build_covering_rev_value` does at link() time, but skips the
/// `scan_prefix` for "other forward 1:1 targets" because the in-memory
/// `links` slice already enumerates the full peer set.
///
/// `this_idx` is the offset of THIS rev_edge's link in `links`; everything
/// else in the slice that's `is_1to1_forward` becomes a covered peer.
/// Returns `Bytes::new()` if there are no other 1:1 peers (matches the
/// existing convention so the SST doesn't carry empty cover blobs).
/// True when a field-type migration is in flight (the `<field>__shadow`
/// double-write is active). Gates the shadow-strip in the cover builders so a
/// non-migrating database pays one atomic load and no scan.
fn migration_in_flight(db: &Database) -> bool {
    // Acquire pairs with the SeqCst store in `arm_field_hook`/`disarm_field_hook`
    // so a cover builder running under `migration_lock.read()` (the background
    // cover-refresh worker is NOT serialized against a disarm that happens
    // outside the write-lock) can never observe a half-published count.
    db.migrating_field_count
        .load(std::sync::atomic::Ordering::Acquire)
        > 0
}

/// Drop the card-2 `<field>__shadow`/`<field>__shadow_cv` siblings from a
/// FieldMap that is about to be embedded in a cover blob. Covers must be
/// SHADOW-FREE: cutover never re-derives cover keyspaces from the shadows, and
/// a shadow baked into a cover would leak via the executor's fusion fast path
/// and never reconcile. Gated by the caller on `migration_in_flight`.
fn strip_shadow_from_cover(map: &mut FieldMap) {
    map.retain(|k, _| !is_shadow_sibling_key(k));
}

/// Return a serialized blob guaranteed shadow-free: if a migration is in flight
/// and `data` carries shadow siblings, deserialize + strip + re-serialize;
/// otherwise hand back the original `Bytes` (refcount only, no copy).
fn shadow_stripped_blob(db: &Database, data: Bytes) -> Bytes {
    if !migration_in_flight(db) {
        return data;
    }
    let mut fields = deserialize_fields(&data);
    let before = fields.len();
    strip_shadow_from_cover(&mut fields);
    if fields.len() == before {
        data
    } else {
        serialize_fields(&fields)
    }
}

fn build_inflight_cover(
    db: &Database,
    txn: &rhypedb_storage::mvcc::Transaction,
    scalar_fields: &FieldMap,
    this_field: &str,
    this_target: u64,
    links: &[(String, u64, String, Bytes, u64, u64, bool)],
    this_idx: usize,
) -> EngineResult<Bytes> {
    let other_1to1: Vec<&(String, u64, String, Bytes, u64, u64, bool)> = links
        .iter()
        .enumerate()
        .filter_map(|(j, l)| if j == this_idx || !l.6 { None } else { Some(l) })
        .collect();
    if other_1to1.is_empty() {
        return Ok(Bytes::new());
    }

    let mut effective = scalar_fields.clone();
    // Card 2: never bake the source's `<field>__shadow` siblings into a cover.
    if migration_in_flight(db) {
        strip_shadow_from_cover(&mut effective);
    }
    effective.insert(this_field.to_string(), Value::U64(this_target));
    for link in &other_1to1 {
        effective.insert(link.0.clone(), Value::U64(link.1));
    }
    for link in &other_1to1 {
        // Augment the in-memory target_data with 3rd-degree covers — same
        // recursion as build_covering_rev_value does at link() time. The
        // type of the link's target is stored as link.2.
        let nested = with_nested_forward_covers(db, txn, &link.2, link.3.clone(), link.1)?;
        effective.insert(format!("{}__cover", link.0), Value::Bytes(nested));
        effective.insert(format!("{}__cover_v", link.0), Value::U64(link.4));
    }
    Ok(serialize_fields(&effective))
}

fn build_covering_rev_value(
    db: &Database,
    txn: &rhypedb_storage::mvcc::Transaction,
    source_type: &str,
    source_id: u64,
    source_data: Option<&[u8]>,
    field_name: &str,
    target_id: u64,
) -> EngineResult<Bytes> {
    // Find OTHER forward 1:1 relation targets already in the edge index.
    // Only return a non-empty covering value when at least one such target
    // exists — otherwise the carried fields can't satisfy any fusion lookup,
    // and writing them just bloats the reverse-edge index for no win. (At
    // 100K-scale bench setup this halves the covering data we'd otherwise
    // emit — first link of every pair becomes a normal empty rev edge.)
    let mut other_targets: Vec<(String, u64)> = Vec::new();
    if let Some(type_def) = db.schema.get_type(source_type) {
        for other in &type_def.fields {
            if other.name == field_name {
                continue;
            }
            let rel = match &other.field_type {
                FieldType::Relation(r) => r,
                _ => continue,
            };
            if rel.is_many || other.inverse().is_some() {
                continue;
            }
            let rel_key = format!("{source_type}.{}", other.name);
            let Some(other_rel_id) = db.rel_ids.get(&rel_key).copied() else {
                continue;
            };
            let prefix = KeyBuilder::edge_prefix(source_id, other_rel_id);
            let entries = db.scan_prefix(txn, &prefix)?;
            if let Some((existing_target, _)) = entries.first() {
                other_targets.push((other.name.clone(), *existing_target));
            }
        }
    }

    if other_targets.is_empty() {
        return Ok(Bytes::new());
    }

    // Otherwise, serialize source's explicit fields + this link's target +
    // each discovered other forward target. Each other target also has its
    // OBJECT FIELDS embedded under `<name>__cover` — second-degree covering.
    // A subsequent traversal that hops *through* this source to one of those
    // other targets (e.g. 2-hop `movie.ratings.user`) can extract the
    // target's fields straight from the covering and skip the per-id LSM
    // probe at terminal materialize.
    //
    // Cost: one extra `storage.get` per other_target at link() time. For the
    // bench's setup (1M ratings × 1 extra probe per second link), that's
    // ~30 seconds added to load — paid once, amortized across every read.
    let mut effective = match source_data {
        Some(bytes) => deserialize_fields(bytes),
        None => FieldMap::new(),
    };
    // Card 2: never bake the source's `<field>__shadow` siblings into the cover.
    if migration_in_flight(db) {
        strip_shadow_from_cover(&mut effective);
    }
    effective.insert(field_name.to_string(), Value::U64(target_id));
    for (name, tid) in &other_targets {
        effective.insert(name.clone(), Value::U64(*tid));
    }

    // Look up each other_target's object fields and embed under `__cover`.
    // Alongside the blob, stamp the target's current generation under
    // `<name>__cover_v` so the executor's fusion path can detect when the
    // embedded snapshot is stale (target has been updated since this
    // rev_edge was written) and fall back to a fresh LSM probe for that
    // specific target — instead of forcing an unbounded rev_edge rewrite
    // on every update to a hot key.
    //
    // Third-degree (3-hop) covering: when the other-target is itself the
    // source of an outgoing 1:1 forward relation, that next-level target's
    // data + cover_v stamp get embedded INSIDE the other-target's serialized
    // form via `with_nested_forward_covers`. A query like `S.<other>.<next>`
    // can then extract `next` straight from the rev_edge bytes — no LSM
    // probe at either hop. Bounded by one extra storage.get per nested
    // 1:1 forward field per other-target.
    if let Some(type_def) = db.schema.get_type(source_type) {
        for (name, tid) in &other_targets {
            let Some(field) = type_def.get_field(name) else {
                continue;
            };
            let rel = match &field.field_type {
                FieldType::Relation(r) => r,
                _ => continue,
            };
            let Some(target_type_id) = db.type_ids.get(&rel.target_type).copied() else {
                continue;
            };
            let target_key = KeyBuilder::object(target_type_id, *tid);
            if let Ok(Some(target_data)) = db.storage.get(txn, &target_key) {
                let nested =
                    with_nested_forward_covers(db, txn, &rel.target_type, target_data, *tid)?;
                effective.insert(format!("{name}__cover"), Value::Bytes(nested));
                let target_version = db.object_version(&rel.target_type, *tid);
                effective.insert(format!("{name}__cover_v"), Value::U64(target_version));
            }
        }
    }

    Ok(serialize_fields(&effective))
}

/// Augment a target's serialized fields with 3rd-degree cover data: for
/// each 1:1 forward (non-inverse, non-many) relation on `target_type`,
/// fetch that next-hop target's data and embed under `<field>__cover` /
/// `<field>__cover_v` directly inside the target's FieldMap before
/// serialization. Also writes `<field>: Value::U64(next_tid)` so the
/// executor's `find_u64_field_in_raw` can locate the next-hop id without
/// a second edge scan.
///
/// This is the engine-side recursion that turns 2-hop covering into
/// 3-hop covering. Depth is capped at one extra level — recursion would
/// be unsound for cyclic 1:1 schemas (e.g. User.partner: User) without
/// cycle detection, and the storage cost compounds. Bench schemas with
/// chained 1:1 forward relations get the win; schemas without any
/// outgoing 1:1 on the target (the existing Movie/User shape) get
/// `target_data` back verbatim.
fn with_nested_forward_covers(
    db: &Database,
    txn: &rhypedb_storage::mvcc::Transaction,
    target_type: &str,
    target_data: Bytes,
    target_id: u64,
) -> EngineResult<Bytes> {
    let Some(type_def) = db.schema.get_type(target_type) else {
        // Card 2: even the verbatim-passthrough must not leak `<field>__shadow`.
        return Ok(shadow_stripped_blob(db, target_data));
    };

    // Cheap pre-check: does this type even have any 1:1 forward outgoing
    // relations? If not, skip the deserialize+reserialize round trip and
    // return the original bytes (refcount-only, no copy — unless a migration
    // means it carries shadow siblings to strip).
    let has_any_forward_1to1 = type_def.fields.iter().any(|f| {
        matches!(&f.field_type, FieldType::Relation(rel) if !rel.is_many) && f.inverse().is_none()
    });
    if !has_any_forward_1to1 {
        return Ok(shadow_stripped_blob(db, target_data));
    }

    let mut effective = deserialize_fields(&target_data);
    if migration_in_flight(db) {
        strip_shadow_from_cover(&mut effective);
    }
    let mut wrote_any = false;

    for field in &type_def.fields {
        let rel = match &field.field_type {
            FieldType::Relation(r) => r,
            _ => continue,
        };
        if rel.is_many || field.inverse().is_some() {
            continue;
        }
        let rel_key = format!("{target_type}.{}", field.name);
        let Some(rel_id) = db.rel_ids.get(&rel_key).copied() else {
            continue;
        };
        // Find the next-hop target via the forward edge index.
        let prefix = KeyBuilder::edge_prefix(target_id, rel_id);
        let entries = db.scan_prefix(txn, &prefix)?;
        let Some(&(next_tid, _)) = entries.first() else {
            continue;
        };
        let Some(next_type_id) = db.type_ids.get(&rel.target_type).copied() else {
            continue;
        };
        let next_key = KeyBuilder::object(next_type_id, next_tid);
        let Ok(Some(next_data)) = db.storage.get(txn, &next_key) else {
            continue;
        };
        wrote_any = true;
        effective.insert(field.name.clone(), Value::U64(next_tid));
        // Card 2: the next-hop blob is embedded verbatim — strip its shadows.
        effective.insert(
            format!("{}__cover", field.name),
            Value::Bytes(shadow_stripped_blob(db, next_data)),
        );
        let next_v = db.object_version(&rel.target_type, next_tid);
        effective.insert(format!("{}__cover_v", field.name), Value::U64(next_v));
    }

    if !wrote_any {
        // No outgoing 1:1 edges actually populated (type has the fields
        // but this instance hasn't been linked yet). Return original bytes
        // unchanged so we don't pay reserialization cost for no benefit
        // (still shadow-stripped if a migration is in flight).
        return Ok(shadow_stripped_blob(db, target_data));
    }
    Ok(serialize_fields(&effective))
}

fn value_to_index_bytes(value: &Value) -> Vec<u8> {
    match value {
        Value::String(s) => s.as_bytes().to_vec(),
        Value::U32(v) => v.to_be_bytes().to_vec(),
        Value::U64(v) => v.to_be_bytes().to_vec(),
        Value::I32(v) => v.to_be_bytes().to_vec(),
        Value::I64(v) => v.to_be_bytes().to_vec(),
        Value::F32(v) => v.to_be_bytes().to_vec(),
        Value::F64(v) => v.to_be_bytes().to_vec(),
        Value::Bool(v) => vec![u8::from(*v)],
        Value::Bytes(b) => b.to_vec(),
        // DateTime indexes by its i64 epoch-millis (big-endian, like I64).
        Value::DateTime(ms) => ms.to_be_bytes().to_vec(),
        // Json indexes by its compact serialized bytes (exact-match only).
        Value::Json(j) => serde_json::to_vec(j).unwrap_or_default(),
        Value::Null => vec![],
    }
}

fn fields_to_json(fields: &FieldMap) -> HashMap<String, serde_json::Value> {
    fields
        .iter()
        .map(|(k, v)| (k.clone(), crate::object::value_to_query_json(v)))
        .collect()
}

/// Validate a relation edge-field value against its declared scalar type. The
/// edge-field analogue of the `FieldType::Scalar` arm of [`validate_value`] —
/// an exact `Value`/`ScalarType` variant match (Null is always allowed).
fn validate_edge_value(
    edge_def: &rhypedb_schema::EdgeFieldDef,
    value: &Value,
) -> EngineResult<()> {
    if matches!(value, Value::Null) {
        return Ok(());
    }
    let ok = matches!(
        (&edge_def.scalar_type, value),
        (ScalarType::String, Value::String(_))
            | (ScalarType::U32, Value::U32(_))
            | (ScalarType::U64, Value::U64(_))
            | (ScalarType::I32, Value::I32(_))
            | (ScalarType::I64, Value::I64(_))
            | (ScalarType::F32, Value::F32(_))
            | (ScalarType::F64, Value::F64(_))
            | (ScalarType::Bool, Value::Bool(_))
            | (ScalarType::Bytes, Value::Bytes(_))
            | (ScalarType::DateTime, Value::DateTime(_))
            | (ScalarType::Json, Value::Json(_))
    );
    if !ok {
        return Err(EngineError::TypeMismatch {
            field: edge_def.name.clone(),
            expected: format!("{:?}", edge_def.scalar_type),
            got: value.type_name().into(),
        });
    }
    Ok(())
}

fn validate_value(field_def: &rhypedb_schema::FieldDef, value: &Value) -> EngineResult<()> {
    if matches!(value, Value::Null) {
        return Ok(());
    }

    match &field_def.field_type {
        FieldType::Scalar(scalar) => {
            let ok = matches!(
                (scalar, value),
                (ScalarType::String, Value::String(_))
                    | (ScalarType::U32, Value::U32(_))
                    | (ScalarType::U64, Value::U64(_))
                    | (ScalarType::I32, Value::I32(_))
                    | (ScalarType::I64, Value::I64(_))
                    | (ScalarType::F32, Value::F32(_))
                    | (ScalarType::F64, Value::F64(_))
                    | (ScalarType::Bool, Value::Bool(_))
                    | (ScalarType::Bytes, Value::Bytes(_))
                    | (ScalarType::DateTime, Value::DateTime(_))
                    | (ScalarType::Json, Value::Json(_))
            );
            if !ok {
                return Err(EngineError::TypeMismatch {
                    field: field_def.name.clone(),
                    expected: format!("{scalar:?}"),
                    got: value.type_name().into(),
                });
            }
        }
        FieldType::Relation(_) => {
            // Relation values at create/update time encode the target object
            // id — accept any unsigned integer literal that fits in u64.
            // Inline-relation creates use this path so the engine can stage
            // edges + rev_edges in the same txn as the object put.
            let ok = matches!(
                value,
                Value::U64(_) | Value::U32(_) | Value::I32(_) | Value::I64(_)
            );
            if !ok {
                return Err(EngineError::TypeMismatch {
                    field: field_def.name.clone(),
                    expected: "relation target id (integer)".into(),
                    got: value.type_name().into(),
                });
            }
        }
        FieldType::Vector(_) => {
            return Err(EngineError::TypeMismatch {
                field: field_def.name.clone(),
                expected: "scalar".into(),
                got: value.type_name().into(),
            });
        }
    }
    Ok(())
}

/// `type_id` → list of `(field_name, field_id)` for every integer scalar
/// field on that type. The zone-extractor consults this at SST flush time
/// to translate a FieldMap's string-keyed entries into the `field_id` zone
/// columns expect (SST v5+); the predicate builder in `filter_scan` uses
/// the same table on the read path.
///
/// Rename-safety: `field_id` is the catalog's stable u64 truncated to u32
/// — invariant across `rename_field`. So an SST written before a rename is
/// still pruned correctly after, and the table is rebuilt without renumbering.
pub(crate) type ZoneFieldIdLookup = HashMap<u64, Vec<(String, u32)>>;

/// Build the zone-field lookup table from the loaded catalog. One entry per
/// type that has at least one zone-eligible scalar field (the integer scalars
/// plus `DateTime`); the engine never enrolls the others in zone maps
/// (`encode_int_for_zone` returns `None` for them).
pub(crate) fn build_zone_field_id_lookup(
    schema: &Schema,
    type_ids: &HashMap<String, u64>,
    field_ids: &HashMap<String, u64>,
) -> ZoneFieldIdLookup {
    let mut out: ZoneFieldIdLookup = HashMap::new();
    for (type_name, type_def) in &schema.types {
        let Some(&type_id) = type_ids.get(type_name) else {
            continue;
        };
        let mut entries: Vec<(String, u32)> = Vec::new();
        for field in &type_def.fields {
            if !field_is_zone_eligible(field) {
                continue;
            }
            let qual = format!("{type_name}.{}", field.name);
            let Some(&fid_u64) = field_ids.get(&qual) else {
                continue;
            };
            // Catalog IDs start at 1 and increment by 1 (see `c:N:E`); a
            // schema with >2^32 fields is unreachable in practice but the
            // debug_assert flags any future producer of larger IDs.
            debug_assert!(fid_u64 <= u32::MAX as u64);
            entries.push((field.name.clone(), fid_u64 as u32));
        }
        if !entries.is_empty() {
            out.insert(type_id, entries);
        }
    }
    out
}

/// Whether a field is eligible to be enrolled in an SST zone map. Mirrors
/// `encode_int_for_zone`'s match arms (the integer scalars plus `DateTime`,
/// which shares the I64 ordered encoding).
fn field_is_zone_eligible(field: &rhypedb_schema::FieldDef) -> bool {
    matches!(
        &field.field_type,
        FieldType::Scalar(
            ScalarType::U32
                | ScalarType::U64
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::DateTime
        )
    )
}

/// Zone-field extractor passed to `LsmConfig::zone_extractor`. Pulls integer
/// field values out of an object entry's serialized FieldMap so the SST
/// writer can record per-block min/max bounds.
///
/// Returns empty when the entry isn't an object key (edges, reverse edges,
/// unique-index entries, etc.) since their values aren't FieldMaps. Object
/// keys are `o:<type>:<id>` so the prefix check is a 2-byte compare;
/// `type_id` lives at bytes 2..10 (big-endian u64, per `KeyBuilder::object`).
pub(crate) fn do_extract_zone_fields(
    lookup: &ZoneFieldIdLookup,
    internal_key: &[u8],
    value: &[u8],
) -> Vec<(u32, [u8; 8])> {
    if internal_key.len() < 2 + 8 || internal_key[0] != b'o' || internal_key[1] != b':' {
        return Vec::new();
    }
    let type_id = u64::from_be_bytes(internal_key[2..10].try_into().unwrap());
    let Some(field_entries) = lookup.get(&type_id) else {
        return Vec::new();
    };
    let fields = deserialize_fields(value);
    let mut out: Vec<(u32, [u8; 8])> = Vec::with_capacity(field_entries.len());
    for (name, fid) in field_entries {
        if let Some(val) = fields.get(name.as_str())
            && let Some(encoded) = encode_int_for_zone(val)
        {
            out.push((*fid, encoded));
        }
    }
    out
}

/// Per-entry re-check for `filter_scan`'s zone-map path. The block-level zone
/// filter is coarse; entries that survived may still individually fail the
/// predicate. Takes the single predicate-field `Value` (extracted via
/// `extract_field` — no full deserialize) and re-applies the integer compare.
/// A non-integer / absent value never matches.
pub(crate) fn value_passes_int_predicate(
    value: &Value,
    op: rhypedb_storage::zone::CompareOp,
    target_u64: u64,
) -> bool {
    use rhypedb_storage::zone::CompareOp;

    let Some(bytes) = encode_int_for_zone(value) else {
        return false;
    };
    let entry = u64::from_be_bytes(bytes);
    match op {
        CompareOp::Eq => entry == target_u64,
        CompareOp::Ne => entry != target_u64,
        CompareOp::Lt => entry < target_u64,
        CompareOp::Le => entry <= target_u64,
        CompareOp::Gt => entry > target_u64,
        CompareOp::Ge => entry >= target_u64,
    }
}

/// Encode an integer-typed `Value` into 8 bytes whose lexicographic order
/// matches numeric order. Signed types flip the MSB so negatives sort below
/// positives; narrow types widen to 64 bits first. `DateTime` encodes like
/// `I64` (its i64 epoch-millis). Returns `None` for the value types with no
/// ordered fixed-width slot here (strings, floats, bools, nulls, bytes, json)
/// — those aren't zone-mapped.
/// Sort-preserving variable-length encoding for `String` and `Bytes`
/// secondary index entries.
///
/// Every byte is copied through with one escape rule — `0x00` becomes
/// `0x00 0x01` — and a terminator `0x00 0x00` is appended. Properties:
///
///   * Byte-wise lexicographic order on the encoded form matches byte-wise
///     order on the original. For UTF-8 strings this means code-point order
///     for ASCII and raw byte order otherwise, matching PG's default `text`
///     collation = `C`.
///   * The terminator `0x00 0x00` can never appear inside a valid encoded
///     value, because every embedded `0x00` is followed by `0x01`. So a
///     prefix scan keyed by `encoded || 0x00 0x00` matches exactly the keys
///     whose encoded value equals `encoded` — no false positives from
///     longer values that share the prefix.
///   * Empty values still produce two bytes (the terminator), so empties
///     sort before any non-empty value starting with `0x00 0x01` and after
///     every other value.
pub(crate) fn encode_bytes_for_index(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() + 2);
    for &x in b {
        if x == 0 {
            out.push(0);
            out.push(1);
        } else {
            out.push(x);
        }
    }
    out.push(0);
    out.push(0);
    out
}

pub(crate) fn encode_str_for_index(s: &str) -> Vec<u8> {
    encode_bytes_for_index(s.as_bytes())
}

/// Sort-preserving fixed-width encoding for IEEE-754 floats.
///
/// Both `f32` and `f64` widen to `f64` so they share one 8-byte slot in
/// `KeyBuilder::field_index`. The bit transformation is the standard
/// sortable-float trick:
///
///   * Positive (or +0): flip the sign bit. Positives now have a leading
///     `1`, so they sort above negatives.
///   * Negative (or -0): flip all bits. Negatives reverse their magnitude
///     ordering so larger-magnitude negatives sort first.
///
/// After the transformation, byte-wise lexicographic order matches numeric
/// order for every non-NaN value. NaN values map deterministically based on
/// their bit pattern and sort at one end — fine for our index (we don't
/// distinguish NaN from itself for Eq), but the exact NaN position isn't a
/// stable API guarantee.
pub(crate) fn encode_f64_for_index(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    let xform = if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000_0000_0000
    };
    xform.to_be_bytes()
}

fn encode_float_for_index(value: &Value) -> Option<[u8; 8]> {
    let v = match value {
        Value::F32(v) => *v as f64,
        Value::F64(v) => *v,
        _ => return None,
    };
    Some(encode_f64_for_index(v))
}

fn encode_bool_for_index(value: &Value) -> Option<[u8; 8]> {
    let Value::Bool(b) = value else { return None };
    let mut out = [0u8; 8];
    out[7] = u8::from(*b);
    Some(out)
}

/// Apply a `CompareOp` to any pair of `Ord` values. Used by the
/// non-indexed fallback for strings and bytes.
fn compare_ord<T: Ord + ?Sized>(a: &T, op: rhypedb_storage::zone::CompareOp, b: &T) -> bool {
    use rhypedb_storage::zone::CompareOp;
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => a <= b,
        CompareOp::Gt => a > b,
        CompareOp::Ge => a >= b,
    }
}

/// Apply a `CompareOp` to `PartialOrd` values (i.e. floats). NaN
/// comparisons return `false` for every op except `Ne`, matching IEEE 754
/// and SQL semantics.
fn compare_partial<T: PartialOrd>(a: T, op: rhypedb_storage::zone::CompareOp, b: T) -> bool {
    use rhypedb_storage::zone::CompareOp;
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => a <= b,
        CompareOp::Gt => a > b,
        CompareOp::Ge => a >= b,
    }
}

fn compare_bool(a: bool, op: rhypedb_storage::zone::CompareOp, b: bool) -> bool {
    // false < true; defer to `compare_ord` via u8.
    compare_ord(&u8::from(a), op, &u8::from(b))
}

impl crate::catalog::FieldCoverMaintainer for Database {
    /// Offline field-type change (`change_field_type` / `run_migrations`) cover
    /// maintenance: overwrite every SIBLING `@indexed` field's covering payload
    /// with the converted blob, so a covered `filter_scan` on that sibling stops
    /// returning the migrated field's stale source value. The indexed values are
    /// unchanged by the migration (the migrating field is never @indexed), so
    /// each existing index key is reproduced exactly and `put` overwrites the
    /// stale value in place. Mirrors the `any_update` re-put branch the online
    /// cutover hits via `rewrite_object_and_maintain_covers`.
    fn sibling_index_cover_puts(
        &self,
        type_name: &str,
        object_id: u64,
        fields: &FieldMap,
        serialized: &Bytes,
    ) -> Vec<(Bytes, Bytes)> {
        let mut out = Vec::new();
        let Some(idx_fields) = self.indexed_fields.get(type_name) else {
            return out;
        };
        let Some(&type_id) = self.type_ids.get(type_name) else {
            return out;
        };
        for ifd in idx_fields {
            if let Some(value) = fields.get(&ifd.name)
                && !matches!(value, Value::Null)
                && let Some(key) =
                    build_field_index_key(type_id, ifd.field_id, ifd.kind, value, object_id)
            {
                out.push((key, serialized.clone()));
            }
        }
        out
    }

    fn rename_index_cover_puts(
        &self,
        type_name: &str,
        object_id: u64,
        fields: &FieldMap,
        serialized: &Bytes,
        current_field_name: &std::collections::HashMap<u64, String>,
    ) -> Vec<(Bytes, Bytes)> {
        let mut out = Vec::new();
        let Some(idx_fields) = self.indexed_fields.get(type_name) else {
            return out;
        };
        let Some(&type_id) = self.type_ids.get(type_name) else {
            return out;
        };
        for ifd in idx_fields {
            // `indexed_fields` still carries the pre-plan name (the live handle
            // hasn't reopened), so resolve each indexed field's CURRENT name via
            // its stable field_id. This tracks a field renamed by an EARLIER verb
            // in the same plan — which a single old→new remap could not — and the
            // caller folds in THIS verb's own rename too. field_id/kind are stable,
            // so the rebuilt i: key matches the existing entry and overwrites it.
            let Some(lookup) = current_field_name.get(&ifd.field_id) else {
                continue;
            };
            if let Some(value) = fields.get(lookup.as_str())
                && !matches!(value, Value::Null)
                && let Some(key) =
                    build_field_index_key(type_id, ifd.field_id, ifd.kind, value, object_id)
            {
                out.push((key, serialized.clone()));
            }
        }
        out
    }
}

/// Build the secondary-index key for one `(type_id, field_id, value, object_id)`
/// tuple, picking the encoder + key layout that matches the indexed field's
/// `kind`. Returns `None` when the value isn't representable in the chosen
/// encoder (null, mismatched scalar type) — caller treats that as "nothing
/// to index" and skips the write/delete.
fn build_field_index_key(
    type_id: u64,
    field_id: u64,
    kind: IndexedKind,
    value: &Value,
    object_id: u64,
) -> Option<Bytes> {
    match kind {
        IndexedKind::Integer => {
            let encoded = encode_int_for_zone(value)?;
            Some(KeyBuilder::field_index(
                type_id, field_id, &encoded, object_id,
            ))
        }
        IndexedKind::Bool => {
            let encoded = encode_bool_for_index(value)?;
            Some(KeyBuilder::field_index(
                type_id, field_id, &encoded, object_id,
            ))
        }
        IndexedKind::Float => {
            let encoded = encode_float_for_index(value)?;
            Some(KeyBuilder::field_index(
                type_id, field_id, &encoded, object_id,
            ))
        }
        IndexedKind::String => {
            let Value::String(s) = value else { return None };
            let encoded = encode_str_for_index(s);
            Some(KeyBuilder::field_index_var(
                type_id, field_id, &encoded, object_id,
            ))
        }
        IndexedKind::Bytes => {
            let Value::Bytes(b) = value else { return None };
            let encoded = encode_bytes_for_index(b);
            Some(KeyBuilder::field_index_var(
                type_id, field_id, &encoded, object_id,
            ))
        }
    }
}

pub(crate) fn encode_int_for_zone(value: &Value) -> Option<[u8; 8]> {
    let bits: u64 = match value {
        Value::U32(v) => *v as u64,
        Value::U64(v) => *v,
        Value::I32(v) => {
            // Sign-extend to i64, bit-cast, flip MSB.
            let widened = *v as i64;
            (widened as u64) ^ 0x8000_0000_0000_0000
        }
        Value::I64(v) => (*v as u64) ^ 0x8000_0000_0000_0000,
        // DateTime is i64 epoch-millis — same MSB-flip ordered encoding as
        // I64, so an ordered secondary index / zone map sorts timestamps
        // (including pre-epoch negatives) correctly.
        Value::DateTime(v) => (*v as u64) ^ 0x8000_0000_0000_0000,
        _ => return None,
    };
    Some(bits.to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::find_bytes_field_in_raw;
    use rhypedb_schema::parser::parse_schema;

    fn open_with_shrink(schema: Schema, dir: &std::path::Path) -> Arc<Database> {
        let opts = OpenOptions {
            allow_schema_shrink: true,
            ..Default::default()
        };
        Database::open_with_options(schema, dir, opts).unwrap()
    }

    // -----------------------------------------------------------------
    // Tombstone gating — read paths
    // -----------------------------------------------------------------

    /// A type that was retired via schema shrink must error with
    /// `TypeRetired` (not `TypeNotFound`) when named in `get`. The
    /// distinction matters because the operator's mental model is
    /// different: "you removed it" vs. "you typoed."
    #[test]
    fn get_on_retired_type_returns_type_retired() {
        let dir = tempfile::tempdir().unwrap();
        // Open with two types, insert into both, retire one, reopen.
        let big = parse_schema(
            r#"
            type User { name: String }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        let mut fields = FieldMap::new();
        fields.insert("title".into(), Value::String("Inception".into()));
        let movie = db.create("Movie", fields).unwrap();
        drop(db);

        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let err = db2.get("Movie", movie.id).unwrap_err();
        let EngineError::TypeRetired { name, .. } = err else {
            panic!("expected TypeRetired, got {err}");
        };
        assert_eq!(name, "Movie");
    }

    /// A field that was retired on a still-live type is stripped from
    /// the returned FieldMap. The on-disk bytes are preserved.
    #[test]
    fn get_strips_retired_field_from_returned_field_map() {
        let dir = tempfile::tempdir().unwrap();
        let big = parse_schema(
            r#"
            type User {
                name: String
                nickname: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("nickname".into(), Value::String("Ally".into()));
        let user = db.create("User", fields).unwrap();
        drop(db);

        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let fetched = db2.get("User", user.id).unwrap();
        assert!(fetched.fields.contains_key("name"));
        assert!(
            !fetched.fields.contains_key("nickname"),
            "retired field must be stripped from returned FieldMap"
        );
    }

    /// `scan_type` on a retired type errors; on a live type, the
    /// returned FieldMaps have any retired fields stripped.
    #[test]
    fn scan_type_strips_retired_fields_and_rejects_retired_types() {
        let dir = tempfile::tempdir().unwrap();
        let big = parse_schema(
            r#"
            type User {
                name: String
                nickname: String
            }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        for i in 0..3 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("u{i}")));
            f.insert("nickname".into(), Value::String(format!("nick{i}")));
            db.create("User", f).unwrap();
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("m{i}")));
            db.create("Movie", f).unwrap();
        }
        drop(db);

        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let users = db2.scan_type("User").unwrap();
        assert_eq!(users.len(), 3);
        for u in &users {
            assert!(u.fields.contains_key("name"));
            assert!(!u.fields.contains_key("nickname"));
        }
        let err = db2.scan_type("Movie").unwrap_err();
        assert!(matches!(err, EngineError::TypeRetired { .. }));
    }

    // -----------------------------------------------------------------
    // Tombstone gating — write paths
    // -----------------------------------------------------------------

    #[test]
    fn create_on_retired_type_returns_type_retired() {
        let dir = tempfile::tempdir().unwrap();
        let big = parse_schema(
            r#"
            type User { name: String }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        drop(db);
        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("x".into()));
        let err = db2.create("Movie", f).unwrap_err();
        assert!(matches!(err, EngineError::TypeRetired { .. }));
    }

    #[test]
    fn update_with_retired_field_returns_field_retired() {
        let dir = tempfile::tempdir().unwrap();
        let big = parse_schema(
            r#"
            type User {
                name: String
                nickname: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("nickname".into(), Value::String("Ally".into()));
        let user = db.create("User", f).unwrap();
        drop(db);

        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let mut updates = FieldMap::new();
        updates.insert("nickname".into(), Value::String("Newname".into()));
        let err = db2.update("User", user.id, updates).unwrap_err();
        let EngineError::FieldRetired { field, .. } = err else {
            panic!("expected FieldRetired, got {err}");
        };
        assert_eq!(field, "nickname");
    }

    #[test]
    fn delete_on_retired_type_returns_type_retired() {
        let dir = tempfile::tempdir().unwrap();
        let big = parse_schema(
            r#"
            type User { name: String }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(big, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("x".into()));
        let movie = db.create("Movie", f).unwrap();
        drop(db);
        let smaller = parse_schema(r#"type User { name: String }"#).unwrap();
        let db2 = open_with_shrink(smaller, dir.path());
        let err = db2.delete("Movie", movie.id).unwrap_err();
        assert!(matches!(err, EngineError::TypeRetired { .. }));
    }

    // -----------------------------------------------------------------
    // Rename verb (card 3/5) — integration tests
    // -----------------------------------------------------------------

    /// Renaming a type via `Database::rename_type` preserves object
    /// data. Insert under the old name, rename, drop the handle, open
    /// with the new schema, fetch the same object by id under the new
    /// name. Bytes are identical.
    #[test]
    fn rename_type_preserves_object_data_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
                email: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("email".into(), Value::String("a@x".into()));
        let user = db.create("User", f).unwrap();
        let original_id = user.id;
        let report = db.rename_type("User", "Account").unwrap();
        assert_eq!(report.renamed_types.len(), 1);
        assert_eq!(report.renamed_types[0].from, "User");
        assert_eq!(report.renamed_types[0].to, "Account");
        drop(db);

        let schema_after = parse_schema(
            r#"
            type Account {
                name: String
                email: String
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        let fetched = db2.get("Account", original_id).unwrap();
        assert_eq!(fetched.type_name, "Account");
        assert_eq!(
            fetched.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(
            fetched.fields.get("email"),
            Some(&Value::String("a@x".into()))
        );
    }

    /// After rename, the old name returns TypeNotFound (NOT
    /// TypeRetired — rename is distinct from retirement).
    #[test]
    fn old_name_after_rename_returns_type_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", f).unwrap();
        db.rename_type("User", "Account").unwrap();
        drop(db);

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        let err = db2.get("User", user.id).unwrap_err();
        assert!(
            matches!(err, EngineError::TypeNotFound(_)),
            "expected TypeNotFound on old name, got {err}"
        );
    }

    /// Re-opening with the post-rename schema picks up the renamed
    /// type seamlessly — no `SchemaShrinkRequiresOptIn` because the
    /// old name was renamed (not dropped).
    #[test]
    fn reopen_with_post_rename_schema_succeeds_without_shrink_flag() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        db.rename_type("User", "Account").unwrap();
        drop(db);

        // Open WITHOUT allow_schema_shrink — the rename leaves no
        // dropped entries, so this succeeds.
        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let _ = Database::open(schema_after, dir.path()).unwrap();
    }

    // -----------------------------------------------------------------
    // rename_field (card 3/5 phase 2) — integration tests
    // -----------------------------------------------------------------

    /// Insert two objects under the old name, rename, reopen with the
    /// post-rename schema, and assert both objects round-trip the value
    /// under the new field name. Proves the in-batch FieldMap rewrite
    /// landed for every object.
    #[test]
    fn rename_field_rewrites_object_fieldmaps_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
                age: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        f1.insert("age".into(), Value::U32(30));
        let u1 = db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        f2.insert("age".into(), Value::U32(25));
        let u2 = db.create("User", f2).unwrap();

        let report = db.rename_field("User", "name", "handle").unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        assert_eq!(report.renamed_fields[0].objects_rewritten, 2);
        drop(db);

        let schema_after = parse_schema(
            r#"
            type User {
                handle: String
                age: u32
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        let r1 = db2.get("User", u1.id).unwrap();
        assert_eq!(
            r1.fields.get("handle"),
            Some(&Value::String("Alice".into())),
            "object 1 must expose the value under the new name"
        );
        assert_eq!(
            r1.fields.get("age"),
            Some(&Value::U32(30)),
            "untouched fields preserved"
        );
        assert!(
            !r1.fields.contains_key("name"),
            "the old name must NOT remain in the rewritten FieldMap"
        );
        let r2 = db2.get("User", u2.id).unwrap();
        assert_eq!(
            r2.fields.get("handle"),
            Some(&Value::String("Bob".into()))
        );
    }

    /// A field rename preserves the SST zone-map pruning — block bounds
    /// are keyed by stable field_id (SST v5), so a `filter_scan` on the
    /// renamed field still uses the bounds laid down before the rename.
    ///
    /// The original version of this test only asserted result-count
    /// correctness, which would pass even with the zone extractor
    /// stubbed to return empty Vec (per-entry filter fallback masks
    /// zone-map breakage). The adversarial review surfaced this gap;
    /// this rewrite directly inspects the on-disk SST's zone map via
    /// `SstReader` to assert pruning bounds are present under the new
    /// field_id post-rename.
    #[test]
    fn rename_field_preserves_zone_map_pruning() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type Movie {
                title: String
                year: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        // Insert enough movies to flush a multi-block SST. The default
        // sparse-index block size is 16 entries, so 256 entries → 16
        // blocks (more than enough to confirm zone-map column data).
        for i in 0u32..256 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("Film {i}")));
            f.insert("year".into(), Value::U32(1900 + i));
            db.create("Movie", f).unwrap();
        }
        // Force a flush so the zone map gets baked into a v5 SST.
        db.storage.flush().unwrap();

        // Capture the year field_id BEFORE the rename — it's invariant
        // across the rename, so v5 SST zone columns under this id are
        // exactly what we expect to look up post-rename.
        let year_field_id_u64 = db.field_ids()["Movie.year"];
        assert!(year_field_id_u64 <= u32::MAX as u64);
        let year_field_id = year_field_id_u64 as u32;

        db.rename_field("Movie", "year", "released_in").unwrap();
        drop(db);

        let schema_after = parse_schema(
            r#"
            type Movie {
                title: String
                released_in: u32
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();

        // ---- DIRECT zone-map inspection ------------------------------
        // Walk the data dir, find the SST(s), open via `SstReader`, and
        // assert that the zone map carries per-block bounds under the
        // (stable) field_id. If pruning had been broken by the rename
        // (e.g. SST stuck at v4 with FNV(name) keying, or extractor
        // running with empty lookup), the column wouldn't be present
        // and `bounds()` would return None for every block.
        use rhypedb_storage::sst::SstReader;
        let mut sst_paths: Vec<std::path::PathBuf> = Vec::new();
        let sst_dir = dir.path().join("sst");
        for entry in std::fs::read_dir(&sst_dir).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("sst") {
                sst_paths.push(p);
            }
        }
        assert!(!sst_paths.is_empty(), "expected at least one SST on disk");

        let mut blocks_with_bounds: usize = 0;
        for path in &sst_paths {
            let reader = SstReader::open(path).unwrap();
            let Some(zone) = reader.zone_map() else {
                continue; // v4 SST with no usable zone map — skip
            };
            for block_idx in 0..zone.num_blocks() {
                if let Some((min, max)) = zone.bounds(block_idx, year_field_id)
                    && min != u64::MAX
                    && max != u64::MIN
                {
                    // Real bounds (not the "no-data sentinel").
                    // Years are u32 widened to u64 in the encoder.
                    assert!(
                        (1900..=2155).contains(&(min as u32)),
                        "block {block_idx} min={min} out of expected range",
                    );
                    assert!(
                        (1900..=2155).contains(&(max as u32)),
                        "block {block_idx} max={max} out of expected range",
                    );
                    blocks_with_bounds += 1;
                }
            }
        }
        assert!(
            blocks_with_bounds > 0,
            "no SST block carried zone-map bounds under field_id {year_field_id} — pruning is silently broken",
        );

        // ---- Result-correctness check --------------------------------
        // Pruning works AND the object rewrite landed → exact result.
        let results = db2
            .filter_scan(
                "Movie",
                "released_in",
                rhypedb_storage::zone::CompareOp::Gt,
                2100,
                None,
            )
            .unwrap();
        // Years 2101..2155 inclusive: 55 entries.
        assert_eq!(results.len(), 55, "results: {}", results.len());
    }

    /// `Database::rename_field` errors propagate as typed catalog
    /// errors. Smoke-test that the indexed-field refusal surfaces
    /// without a catalog state mutation.
    #[test]
    fn rename_field_indexed_filter_scan_correct_post_rename() {
        // Phase 3 @indexed lift: after renaming an @indexed field, a COVERED
        // filter_scan on the new name returns objects whose FieldMap carries
        // the new name. The i: covering payloads (full object FieldMaps) were
        // refreshed in the rename batch; without that, the cover fast-path
        // would hand back the OLD field name.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie {
                title: String
                year: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        for (t, y) in [("A", 2000u32), ("B", 2010), ("C", 2000)] {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(t.into()));
            f.insert("year".into(), Value::U32(y));
            db.create("Movie", f).unwrap();
        }
        db.storage.flush().unwrap();
        let report = db.rename_field("Movie", "year", "released_in").unwrap();
        assert_eq!(report.renamed_fields[0].objects_rewritten, 3);
        drop(db);

        let schema_after = parse_schema(
            r#"
            type Movie {
                title: String
                released_in: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        let results = db2
            .filter_scan(
                "Movie",
                "released_in",
                rhypedb_storage::zone::CompareOp::Eq,
                2000,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 2, "two movies match released_in == 2000");
        for obj in &results {
            assert_eq!(
                obj.fields.get("released_in"),
                Some(&Value::U32(2000)),
                "covered result must expose the value under the NEW name; got {:?}",
                obj.fields
            );
            assert!(
                !obj.fields.contains_key("year"),
                "covered result must NOT retain the old name; got {:?}",
                obj.fields
            );
        }
    }

    #[test]
    fn chained_rename_with_objects_via_separate_migrations() {
        // The production path: chaining renames as SEPARATE migrations (each its
        // own commit + reopen) correctly carries objects to the terminal name —
        // no half-renamed objects. (A single multi-verb plan is refused.)
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie {
                title: String
                year: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("Aliens".into()));
        f.insert("year".into(), Value::U32(1986));
        let m = db.create("Movie", f).unwrap();
        db.rename_field("Movie", "year", "released_in").unwrap();
        drop(db);

        let mid = parse_schema(
            r#"
            type Movie {
                title: String
                released_in: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(mid, dir.path()).unwrap();
        db.rename_field("Movie", "released_in", "year_released").unwrap();
        drop(db);

        let after = parse_schema(
            r#"
            type Movie {
                title: String
                year_released: u32
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        let got = db2.get("Movie", m.id).unwrap();
        assert_eq!(
            got.fields.get("year_released"),
            Some(&Value::U32(1986)),
            "object must carry the terminal name after a chained rename; got {:?}",
            got.fields
        );
        assert!(!got.fields.contains_key("year"));
        assert!(!got.fields.contains_key("released_in"));
    }

    #[test]
    fn rename_chain_and_field_one_plan_indexed_cover_correct() {
        // Overboard cmqgvlf6b: a SINGLE plan chains an @indexed field
        // (year→released_in→year_made) AND renames a sibling (title→name) on one
        // type. Verifies the objects carry the terminal names AND the @indexed
        // COVERING payload is refreshed to the fully-renamed blob — a covered
        // filter_scan returns `name`, not the stale `title`. The old single
        // old→new cover remap could not resolve this across a chain.
        use crate::catalog::RenameVerb;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie {
                title: String
                year: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("Aliens".into()));
        f.insert("year".into(), Value::U32(1986));
        let m = db.create("Movie", f).unwrap();
        db.storage.flush().unwrap();

        let verbs = [
            RenameVerb::Field {
                type_name: "Movie".into(),
                old: "year".into(),
                new: "released_in".into(),
            },
            RenameVerb::Field {
                type_name: "Movie".into(),
                old: "released_in".into(),
                new: "year_made".into(),
            },
            RenameVerb::Field {
                type_name: "Movie".into(),
                old: "title".into(),
                new: "name".into(),
            },
        ];
        let report =
            crate::catalog::apply_migration_with_cover(&db.storage, &db.schema, &verbs, Some(&*db))
                .unwrap();
        // Per-verb counters: the chain rewrites the ONE object once per step
        // (3 verbs → 3), it is not an object COUNT.
        let total: u64 = report
            .renamed_fields
            .iter()
            .map(|r| r.objects_rewritten)
            .sum();
        assert_eq!(total, 3, "one object rewritten once per verb");
        drop(db);

        let after = parse_schema(
            r#"
            type Movie {
                name: String
                year_made: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        let got = db2.get("Movie", m.id).unwrap();
        assert_eq!(got.fields.get("year_made"), Some(&Value::U32(1986)));
        assert_eq!(
            got.fields.get("name"),
            Some(&Value::String("Aliens".into()))
        );
        assert!(!got.fields.contains_key("year"));
        assert!(!got.fields.contains_key("released_in"));
        assert!(!got.fields.contains_key("title"));

        let results = db2
            .filter_scan(
                "Movie",
                "year_made",
                rhypedb_storage::zone::CompareOp::Eq,
                1986,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].fields.get("name"),
            Some(&Value::String("Aliens".into())),
            "covered result exposes the renamed sibling under its NEW name; got {:?}",
            results[0].fields
        );
        assert!(
            !results[0].fields.contains_key("title"),
            "covered result must not retain the old sibling name; got {:?}",
            results[0].fields
        );
        assert!(!results[0].fields.contains_key("year"));
    }

    #[test]
    fn rename_two_different_types_one_plan() {
        // A multi-verb plan touching DIFFERENT types in one apply was already
        // allowed (the old guard only refused a REPEATED type); the new overlay
        // path now runs it, so pin the disjoint case end-to-end.
        use crate::catalog::RenameVerb;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie { year: u32 @indexed }
            type Actor { age: u32 @indexed }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("year".into(), Value::U32(1986));
        let m = db.create("Movie", mf).unwrap();
        let mut af = FieldMap::new();
        af.insert("age".into(), Value::U32(42));
        let a = db.create("Actor", af).unwrap();
        db.storage.flush().unwrap();

        let verbs = [
            RenameVerb::Field {
                type_name: "Movie".into(),
                old: "year".into(),
                new: "released_in".into(),
            },
            RenameVerb::Field {
                type_name: "Actor".into(),
                old: "age".into(),
                new: "years".into(),
            },
        ];
        crate::catalog::apply_migration_with_cover(&db.storage, &db.schema, &verbs, Some(&*db))
            .unwrap();
        drop(db);

        let after = parse_schema(
            r#"
            type Movie { released_in: u32 @indexed }
            type Actor { years: u32 @indexed }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        assert_eq!(
            db2.get("Movie", m.id).unwrap().fields.get("released_in"),
            Some(&Value::U32(1986))
        );
        assert_eq!(
            db2.get("Actor", a.id).unwrap().fields.get("years"),
            Some(&Value::U32(42))
        );
        let mr = db2
            .filter_scan(
                "Movie",
                "released_in",
                rhypedb_storage::zone::CompareOp::Eq,
                1986,
                None,
            )
            .unwrap();
        assert_eq!(mr.len(), 1);
        assert_eq!(mr[0].fields.get("released_in"), Some(&Value::U32(1986)));
        let ar = db2
            .filter_scan(
                "Actor",
                "years",
                rhypedb_storage::zone::CompareOp::Eq,
                42,
                None,
            )
            .unwrap();
        assert_eq!(ar.len(), 1);
        assert_eq!(ar[0].fields.get("years"), Some(&Value::U32(42)));
    }

    #[test]
    fn rename_type_and_field_different_types_one_plan() {
        // The type+field guard is NARROW: it only refuses the SAME type. A type
        // rename PLUS a field rename of a DIFFERENT type in one plan is allowed and
        // correct (the field verb's cover maintainer keys by the un-renamed type).
        use crate::catalog::RenameVerb;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie { title: String }
            type Actor { age: u32 @indexed }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Heat".into()));
        let m = db.create("Movie", mf).unwrap();
        let mut af = FieldMap::new();
        af.insert("age".into(), Value::U32(42));
        let a = db.create("Actor", af).unwrap();
        db.storage.flush().unwrap();

        let verbs = [
            RenameVerb::Type {
                old: "Movie".into(),
                new: "Film".into(),
            },
            RenameVerb::Field {
                type_name: "Actor".into(),
                old: "age".into(),
                new: "years".into(),
            },
        ];
        crate::catalog::apply_migration_with_cover(&db.storage, &db.schema, &verbs, Some(&*db))
            .unwrap();
        drop(db);

        let after = parse_schema(
            r#"
            type Film { title: String }
            type Actor { years: u32 @indexed }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        assert_eq!(
            db2.get("Film", m.id).unwrap().fields.get("title"),
            Some(&Value::String("Heat".into()))
        );
        assert_eq!(
            db2.get("Actor", a.id).unwrap().fields.get("years"),
            Some(&Value::U32(42))
        );
        let ar = db2
            .filter_scan(
                "Actor",
                "years",
                rhypedb_storage::zone::CompareOp::Eq,
                42,
                None,
            )
            .unwrap();
        assert_eq!(ar.len(), 1);
        assert_eq!(ar[0].fields.get("years"), Some(&Value::U32(42)));
    }

    #[test]
    fn backup_to_roundtrip_reopens_with_objects() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Movie { title: String year: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut ids = Vec::new();
        for (t, y) in [("Aliens", 1986u32), ("Heat", 1995)] {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(t.into()));
            f.insert("year".into(), Value::U32(y));
            ids.push(db.create("Movie", f).unwrap().id);
        }

        // Back up to a fresh dir (same filesystem → hard-linked SSTs).
        let backup_dir = tempfile::tempdir().unwrap();
        let manifest = db.backup_to(backup_dir.path()).unwrap();
        assert!(
            !manifest.sst_names.is_empty(),
            "backup must capture at least one SST"
        );
        assert!(manifest.in_flight_migrations.is_empty());

        // Reopen the BACKUP as an independent database — all objects present.
        let schema2 = parse_schema(r#"type Movie { title: String year: u32 }"#).unwrap();
        let restored = Database::open(schema2, backup_dir.path()).unwrap();
        assert_eq!(
            restored.get("Movie", ids[0]).unwrap().fields.get("title"),
            Some(&Value::String("Aliens".into()))
        );
        assert_eq!(
            restored.get("Movie", ids[1]).unwrap().fields.get("year"),
            Some(&Value::U32(1995))
        );
    }

    #[test]
    fn scan_chunk_visits_every_object_ascending_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Item { n: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        // Far more objects than the chunk cap, so the walk spans many chunks.
        let count = 250u32;
        for n in 0..count {
            let mut f = FieldMap::new();
            f.insert("n".into(), Value::U32(n));
            db.create("Item", f).unwrap();
        }

        let snapshot = db.storage.read_snapshot();
        let max_distinct = 16usize;
        let mut cursor = 0u64;
        let mut collected: Vec<(u64, u32)> = Vec::new();
        let mut chunks = 0;
        loop {
            let chunk = db.scan_chunk("Item", snapshot, cursor, max_distinct).unwrap();
            assert!(chunk.objects.len() <= max_distinct, "chunk respects the cap");
            for obj in &chunk.objects {
                let Some(Value::U32(n)) = obj.fields.get("n") else {
                    panic!("missing/bad field on {}", obj.id)
                };
                collected.push((obj.id, *n));
            }
            chunks += 1;
            match chunk.next_cursor {
                Some(next) if chunk.more => cursor = next,
                _ => break,
            }
        }

        assert!(chunks > 1, "walk should span multiple chunks, got {chunks}");
        assert!(
            collected.windows(2).all(|w| w[0].0 < w[1].0),
            "ascending, de-duplicated ids"
        );
        assert_eq!(collected.len() as u32, count);

        // Identical set to the materializing scan_type.
        let mut want: Vec<(u64, u32)> = db
            .scan_type("Item")
            .unwrap()
            .into_iter()
            .map(|o| match o.fields.get("n") {
                Some(Value::U32(n)) => (o.id, *n),
                _ => panic!("bad object"),
            })
            .collect();
        want.sort_by_key(|(id, _)| *id);
        assert_eq!(collected, want, "chunked scan == scan_type set");
    }

    #[test]
    fn scan_chunk_empty_type_terminates_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Item { n: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let snapshot = db.storage.read_snapshot();
        let chunk = db.scan_chunk("Item", snapshot, 0, 16).unwrap();
        assert!(chunk.objects.is_empty());
        assert_eq!(chunk.next_cursor, None);
        assert!(!chunk.more, "exhausted range must report more=false");
    }

    #[test]
    fn scan_chunk_advances_past_long_tombstone_run() {
        // A contiguous deleted run longer than the chunk cap must NOT strand the
        // live objects beyond it — the sound stop condition is `!more`, not a
        // short/empty chunk. This is the invariant scan_chunk_raw was built for.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Item { n: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let count = 100u32;
        let mut ids = Vec::new();
        for n in 0..count {
            let mut f = FieldMap::new();
            f.insert("n".into(), Value::U32(n));
            ids.push(db.create("Item", f).unwrap().id);
        }
        // Delete a 40-wide contiguous run in the middle; chunk cap is 8, so some
        // chunk lands entirely inside the run (zero live, more=true).
        let max_distinct = 8usize;
        for id in &ids[20..60] {
            db.delete("Item", *id).unwrap();
        }

        let snapshot = db.storage.read_snapshot();
        let mut cursor = 0u64;
        let mut seen = std::collections::BTreeSet::new();
        let mut saw_empty_nonfinal_chunk = false;
        loop {
            let chunk = db.scan_chunk("Item", snapshot, cursor, max_distinct).unwrap();
            if chunk.objects.is_empty() && chunk.more {
                saw_empty_nonfinal_chunk = true;
            }
            for o in &chunk.objects {
                seen.insert(o.id);
            }
            match chunk.next_cursor {
                Some(next) if chunk.more => cursor = next,
                _ => break,
            }
        }

        let expect: std::collections::BTreeSet<u64> =
            ids[..20].iter().chain(ids[60..].iter()).copied().collect();
        assert_eq!(seen, expect, "no live object stranded behind the tombstone run");
        assert!(
            saw_empty_nonfinal_chunk,
            "a chunk should land entirely inside the tombstone run"
        );
    }

    /// Build a small DB with two types, a forward relation carrying an edge
    /// field, an @inverse back-reference, and raw vectors. Returns the db plus
    /// the created ids for assertions.
    fn export_fixture() -> (tempfile::TempDir, Arc<Database>, u64, u64, u64, u64) {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String @unique
                age: u32
                posts: [Post] @inverse(Post.author)
                embedding: Vector<4>
            }
            type Post {
                title: String
                rating: f64
                author: User { weight: f32 }
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut u1f = FieldMap::new();
        u1f.insert("name".into(), Value::String("Ada".into()));
        u1f.insert("age".into(), Value::U32(36));
        let u1 = db.create("User", u1f).unwrap().id;
        let mut u2f = FieldMap::new();
        u2f.insert("name".into(), Value::String("Alan".into()));
        u2f.insert("age".into(), Value::U32(41));
        let u2 = db.create("User", u2f).unwrap().id;

        let mut p1f = FieldMap::new();
        p1f.insert("title".into(), Value::String("On Computable Numbers".into()));
        p1f.insert("rating".into(), Value::F64(4.5));
        let p1 = db.create("Post", p1f).unwrap().id;
        let mut ef = FieldMap::new();
        ef.insert("weight".into(), Value::F32(0.5));
        db.link("Post", p1, "author", u1, Some(ef)).unwrap();

        let mut p2f = FieldMap::new();
        p2f.insert("title".into(), Value::String("Mind".into()));
        p2f.insert("rating".into(), Value::F64(3.0));
        let p2 = db.create("Post", p2f).unwrap().id;
        db.link("Post", p2, "author", u2, None).unwrap();

        // Raw vectors straight to the v: keyspace (no embedder in-test).
        let user_type_id = *db.type_ids().get("User").unwrap();
        let emb_field_id = *db.field_ids().get("User.embedding").unwrap();
        let mut txn = db.storage.begin_txn();
        for (id, arr) in [(u1, [1.0f32, 2.0, 3.0, 4.0]), (u2, [5.0, 6.0, 7.0, 8.0])] {
            let key = KeyBuilder::vector(user_type_id, id, emb_field_id);
            let mut buf = bytes::BytesMut::new();
            for v in arr {
                buf.extend_from_slice(&v.to_be_bytes());
            }
            db.storage.put(&mut txn, &key, buf.freeze()).unwrap();
        }
        db.storage.commit(&mut txn).unwrap();

        (dir, db, u1, u2, p1, p2)
    }

    fn parse_ndjson(out: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(out)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn logical_export_stream_emits_ordered_complete_dump() {
        use crate::logical::LogicalExportOptions;
        let (_dir, db, u1, _u2, p1, _p2) = export_fixture();

        let mut out = Vec::new();
        let summary = db
            .logical_export_stream(&mut out, &LogicalExportOptions::default())
            .unwrap();
        let lines = parse_ndjson(&out);

        // Header first; types sorted.
        assert_eq!(lines[0]["kind"], "header");
        assert_eq!(lines[0]["format"], crate::logical::FORMAT_TAG);
        assert_eq!(lines[0]["vectors"], "raw");
        assert_eq!(lines[0]["types"], serde_json::json!(["Post", "User"]));
        // Schema second; re-parses.
        assert_eq!(lines[1]["kind"], "schema");
        assert!(parse_schema(lines[1]["sdl"].as_str().unwrap()).is_ok());

        // Global section order: all objects, then all edges, then all vectors,
        // then trailer LAST.
        let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
        let last_object = kinds.iter().rposition(|k| *k == "object").unwrap();
        let first_edge = kinds.iter().position(|k| *k == "edge").unwrap();
        let last_edge = kinds.iter().rposition(|k| *k == "edge").unwrap();
        let first_vector = kinds.iter().position(|k| *k == "vector").unwrap();
        assert!(last_object < first_edge, "all objects precede any edge");
        assert!(last_edge < first_vector, "all edges precede any vector");
        assert_eq!(*kinds.last().unwrap(), "trailer");

        let objects: Vec<_> = lines.iter().filter(|l| l["kind"] == "object").collect();
        let edges: Vec<_> = lines.iter().filter(|l| l["kind"] == "edge").collect();
        let vectors: Vec<_> = lines.iter().filter(|l| l["kind"] == "vector").collect();
        assert_eq!(objects.len(), 4, "2 users + 2 posts");
        assert_eq!(edges.len(), 2, "forward Post.author only, NOT the User.posts inverse");
        assert_eq!(vectors.len(), 2);
        assert!(
            edges.iter().all(|e| e["type"] == "Post" && e["field"] == "author"),
            "no inverse-edge duplication"
        );

        // ids are emitted as decimal strings; hoist them out of the closures.
        let (u1s, p1s) = (u1.to_string(), p1.to_string());

        // Edge fields survive with type fidelity.
        let e1 = edges.iter().find(|e| e["src"] == p1s).unwrap();
        assert_eq!(e1["dst"], u1s);
        assert_eq!(e1["edge_fields"]["weight"]["t"], "f32");

        // Scalar fidelity on an object line (u32 carried as a decimal string).
        let ada = objects
            .iter()
            .find(|o| o["type"] == "User" && o["id"] == u1s)
            .unwrap();
        assert_eq!(ada["fields"]["name"]["v"], "Ada");
        assert_eq!(ada["fields"]["age"]["t"], "u32");
        assert_eq!(ada["fields"]["age"]["v"], "36");

        // Vector bytes are byte-equal to the stored v: value.
        let v1 = vectors.iter().find(|v| v["id"] == u1s).unwrap();
        assert_eq!(v1["dims"], 4);
        assert_eq!(v1["field"], "embedding");
        let decoded = crate::logical::decode_bytes(v1["f32"].as_str().unwrap()).unwrap();
        let mut expect = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            expect.extend_from_slice(&v.to_be_bytes());
        }
        assert_eq!(decoded, expect);

        // Trailer counts + summary agree.
        let trailer = lines.last().unwrap();
        assert_eq!(trailer["complete"], true);
        assert_eq!(trailer["counts"]["User"]["objects"], 2);
        assert_eq!(trailer["counts"]["Post"]["objects"], 2);
        assert_eq!(trailer["counts"]["Post"]["edges"], 2);
        assert_eq!(trailer["counts"]["User"]["vectors"], 2);
        assert_eq!(summary.counts["Post"].edges, 2);
        assert_eq!(summary.counts["User"].vectors, 2);
        assert_eq!(summary.total_dangling_skipped(), 0);
    }

    #[test]
    fn logical_export_refuses_while_field_migrating() {
        use crate::logical::LogicalExportOptions;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type T { a: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        // Simulate an in-flight field-type migration (the gate the export reads).
        db.migrating_field_count.store(1, Ordering::SeqCst);
        let mut out = Vec::new();
        let err = db
            .logical_export_stream(&mut out, &LogicalExportOptions::default())
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::ExportWhileMigrating { migrating_fields: 1 }
        ));
        assert!(out.is_empty(), "nothing written when refused");

        // Clearing the gate lets it through.
        db.migrating_field_count.store(0, Ordering::SeqCst);
        let mut ok = Vec::new();
        assert!(db
            .logical_export_stream(&mut ok, &LogicalExportOptions::default())
            .is_ok());
    }

    #[test]
    fn logical_export_selective_types_skips_dangling_edges() {
        use crate::logical::{LogicalExportOptions, VectorMode};
        let (_dir, db, _u1, _u2, _p1, _p2) = export_fixture();

        // Export only Post; its forward author -> User edges are now dangling
        // (target type excluded) and must be skipped + counted, not emitted.
        let opts = LogicalExportOptions {
            types: Some(vec!["Post".into()]),
            vectors: VectorMode::None,
        };
        let mut out = Vec::new();
        let summary = db.logical_export_stream(&mut out, &opts).unwrap();
        let lines = parse_ndjson(&out);

        assert_eq!(lines[0]["types"], serde_json::json!(["Post"]));
        assert!(
            lines.iter().all(|l| l["kind"] != "edge"),
            "dangling edges are not emitted"
        );
        assert!(
            lines
                .iter()
                .filter(|l| l["kind"] == "object")
                .all(|o| o["type"] == "Post"),
            "only the selected type's objects"
        );
        assert!(
            lines.iter().all(|l| l["kind"] != "vector"),
            "vectors=none drops embeddings"
        );
        assert_eq!(summary.total_dangling_skipped(), 2, "both posts' author edges");
        assert_eq!(lines.last().unwrap()["complete"], true);
    }

    #[test]
    fn logical_export_survives_concurrent_compaction() {
        use crate::logical::LogicalExportOptions;

        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Item { n: u32 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut ids = Vec::new();
        for n in 0..20u32 {
            let mut f = FieldMap::new();
            f.insert("n".into(), Value::U32(n));
            ids.push(db.create("Item", f).unwrap().id);
        }
        // Flush the originals to SST1 so a later flush+compact has >= 2 SSTs to
        // merge (compact() is a no-op below 2). The victim is NOT the last-written
        // object, so its committed version is strictly below the export snapshot.
        db.storage.flush().unwrap();
        let victim = ids[5];

        // A sink that, on the first byte written (the header line — before any
        // object is scanned), performs a concurrent update + flush + compaction
        // on the same DB: the worst-case interleave for a long-lived export.
        // Without the registered snapshot + floor-preserving compaction, the
        // victim's pinned version is GC'd and it vanishes from a dump still
        // marked complete:true.
        struct CompactMidStream {
            db: Arc<Database>,
            victim: u64,
            fired: bool,
            out: Vec<u8>,
        }
        impl std::io::Write for CompactMidStream {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if !self.fired {
                    self.fired = true;
                    let mut f = FieldMap::new();
                    f.insert("n".into(), Value::U32(99999));
                    self.db.update("Item", self.victim, f).unwrap();
                    self.db.storage.flush().unwrap();
                    self.db.storage.compact().unwrap();
                }
                self.out.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let mut sink = CompactMidStream {
            db: db.clone(),
            victim,
            fired: false,
            out: Vec::new(),
        };
        let summary = db
            .logical_export_stream(&mut sink, &LogicalExportOptions::default())
            .unwrap();

        let lines = parse_ndjson(&sink.out);
        let objects: Vec<_> = lines.iter().filter(|l| l["kind"] == "object").collect();
        assert_eq!(objects.len(), 20, "no object GC'd out from under the pinned export");
        let victim_id = victim.to_string();
        let victim_line = objects
            .iter()
            .find(|o| o["id"] == victim_id)
            .expect("victim survived");
        assert_eq!(
            victim_line["fields"]["n"]["v"], "5",
            "export reads the pinned-snapshot value, not the concurrent update"
        );
        assert_eq!(summary.counts["Item"].objects, 20);
    }

    #[test]
    fn restore_objects_preserves_ids_and_rebuilds_indexes() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String @unique  score: i64 @indexed }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        // Restore with non-contiguous, caller-supplied ids (as a logical import
        // would, preserving the source ids).
        let rows = vec![
            (5u64, {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("Ada".into()));
                f.insert("score".into(), Value::I64(10));
                f
            }),
            (9u64, {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("Alan".into()));
                f.insert("score".into(), Value::I64(20));
                f
            }),
        ];
        db.restore_objects("User", rows, false).unwrap();

        // Ids preserved exactly.
        assert_eq!(
            db.get("User", 5).unwrap().fields.get("name"),
            Some(&Value::String("Ada".into()))
        );
        assert_eq!(
            db.get("User", 9).unwrap().fields.get("score"),
            Some(&Value::I64(20))
        );

        // A later create does NOT collide with a restored id (fetch_max moved
        // next_object_id past 9).
        let bob = {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("Bob".into()));
            f.insert("score".into(), Value::I64(30));
            db.create("User", f).unwrap()
        };
        assert!(bob.id > 9, "new id {} must exceed the max restored id 9", bob.id);

        // @unique is enforced post-restore: re-inserting "Ada" is rejected.
        let dup = {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("Ada".into()));
            f.insert("score".into(), Value::I64(99));
            db.create("User", f)
        };
        assert!(matches!(dup, Err(EngineError::UniqueViolation { .. })), "got {dup:?}");

        // @indexed covering entries were rebuilt: the index scan finds the
        // restored row by its id.
        let hits = db
            .filter_scan("User", "score", CompareOp::Eq, 10, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 5);
    }

    #[test]
    fn restore_objects_is_all_or_nothing_on_unique_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String @unique }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        // Two rows in one call sharing a @unique value — the whole call must
        // roll back (a buffered unique put is invisible to a later row's read,
        // so staged-dedup is what catches it).
        let rows = vec![
            (1u64, {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("dup".into()));
                f
            }),
            (2u64, {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("dup".into()));
                f
            }),
        ];
        let r = db.restore_objects("User", rows, false);
        assert!(matches!(r, Err(EngineError::UniqueViolation { .. })), "got {r:?}");
        // Nothing landed.
        assert!(db.get("User", 1).is_err());
        assert!(db.get("User", 2).is_err());
        // And next_object_id was NOT advanced (commit never happened), so a
        // fresh create still starts low.
        let o = {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("ok".into()));
            db.create("User", f).unwrap()
        };
        assert_eq!(o.id, 1, "no id burned by the rolled-back restore");
    }

    #[test]
    fn restore_objects_survives_reopen() {
        // Restored ids + values persist across a close/reopen, and the reopened
        // handle's next_object_id is seeded past them (no collision).
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Item { n: u32 }"#).unwrap();
        {
            let db = Database::open(schema.clone(), dir.path()).unwrap();
            let rows: Vec<(u64, FieldMap)> = [3u64, 7, 42]
                .into_iter()
                .map(|id| {
                    let mut f = FieldMap::new();
                    f.insert("n".into(), Value::U32(id as u32));
                    (id, f)
                })
                .collect();
            db.restore_objects("Item", rows, false).unwrap();
        }
        let db = Database::open(schema, dir.path()).unwrap();
        assert_eq!(db.get("Item", 42).unwrap().fields.get("n"), Some(&Value::U32(42)));
        let next = {
            let mut f = FieldMap::new();
            f.insert("n".into(), Value::U32(0));
            db.create("Item", f).unwrap()
        };
        assert!(next.id > 42, "reopened next_object_id must clear the restored max");
    }

    #[test]
    fn restore_vectors_writes_exact_v_keys() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type Doc { embedding: Vector<3> }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        db.restore_objects("Doc", vec![(7, FieldMap::new())], false).unwrap();

        let mut buf = bytes::BytesMut::new();
        for f in [1.0f32, 2.0, 3.0] {
            buf.extend_from_slice(&f.to_be_bytes());
        }
        let payload = buf.freeze();
        db.restore_vectors("Doc", "embedding", &[(7, payload.clone())]).unwrap();

        // The v: key holds the verbatim payload.
        let type_id = *db.type_ids().get("Doc").unwrap();
        let field_id = *db.field_ids().get("Doc.embedding").unwrap();
        let key = KeyBuilder::vector(type_id, 7, field_id);
        let got = db.storage.get_at(db.storage.read_snapshot(), &key).unwrap();
        assert_eq!(got.as_deref(), Some(&payload[..]));

        // A wrong-length payload is rejected (dims*4 enforced).
        let bad = db.restore_vectors(
            "Doc",
            "embedding",
            &[(7, bytes::Bytes::from_static(&[0, 1, 2, 3]))],
        );
        assert!(matches!(bad, Err(EngineError::TypeMismatch { .. })), "got {bad:?}");
    }

    #[test]
    fn metering_counts_objects_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                favourites: [Movie] @on_delete(remove)
            }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        assert_eq!(db.count_objects().unwrap(), 0);
        assert_eq!(db.count_edges().unwrap(), 0);

        let mut m = FieldMap::new();
        m.insert("title".into(), Value::String("A".into()));
        let movie1 = db.create("Movie", m).unwrap();
        let mut m = FieldMap::new();
        m.insert("title".into(), Value::String("B".into()));
        let movie2 = db.create("Movie", m).unwrap();
        let mut u = FieldMap::new();
        u.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", u).unwrap();
        assert_eq!(db.count_objects().unwrap(), 3, "3 live objects");
        assert_eq!(db.count_edges().unwrap(), 0, "no links yet");

        db.link("User", user.id, "favourites", movie1.id, None).unwrap();
        db.link("User", user.id, "favourites", movie2.id, None).unwrap();
        assert_eq!(
            db.count_edges().unwrap(),
            2,
            "two forward links (reverse `r:` entries are not counted)"
        );

        // A delete drops the object count and, via @on_delete(remove), its inbound
        // edge — counts reflect live rows only, never tombstones.
        db.delete("Movie", movie1.id).unwrap();
        assert_eq!(db.count_objects().unwrap(), 2, "one object deleted");
        assert_eq!(db.count_edges().unwrap(), 1, "the removed movie's edge is gone");
    }

    #[test]
    fn rename_relation_then_scalar_same_type_cover_correct() {
        // Regression for the design-review BLOCKER: renaming a RELATION field
        // BEFORE a SCALAR field of the SAME type in one plan. The scalar verb must
        // still rewrite the rev-edge covers for that relation — it enumerates
        // forward relations by stable rel_id from the catalog, not by the (already
        // re-keyed) field name — so the scalar field's NEW name lands in the cover.
        // Needs TWO forward-1:1 relations so the rev covers are populated at all
        // (the recommendation rev cover carries the source's scalar fields + the
        // favourite peer sidecars).
        use crate::catalog::RenameVerb;
        use rhypedb_storage::key::KeyBuilder;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                favourite: Movie
                recommendation: Movie
            }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut m1 = FieldMap::new();
        m1.insert("title".into(), Value::String("Inception".into()));
        let movie1 = db.create("Movie", m1).unwrap();
        let mut m2 = FieldMap::new();
        m2.insert("title".into(), Value::String("Tenet".into()));
        let movie2 = db.create("Movie", m2).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        db.link("User", user.id, "favourite", movie1.id, None).unwrap();
        db.link("User", user.id, "recommendation", movie2.id, None)
            .unwrap();
        // recommendation is the 2nd-linked 1:1 relation, so ITS OWN rev cover is
        // the populated one (carries the user's `name` + favourite's peer
        // sidecars). Rename THAT relation so its non-empty cover is what the scalar
        // verb must refresh — the only shape that fails under name-based relation
        // enumeration.
        let rec_id = db.rel_ids()["User.recommendation"];

        // Relation rename FIRST, then the scalar rename — the order that broke.
        // verb 1 re-keys cat.rel_ids (recommendation→suggested); verb 2 (scalar)
        // must STILL find that relation by rel_id to refresh `name`→`handle` in its
        // cover (fix 3), AND read verb 1's rewrite through the overlay so it does
        // not clobber the recommendation→suggested rename (fix 1).
        let verbs = [
            RenameVerb::Field {
                type_name: "User".into(),
                old: "recommendation".into(),
                new: "suggested".into(),
            },
            RenameVerb::Field {
                type_name: "User".into(),
                old: "name".into(),
                new: "handle".into(),
            },
        ];
        crate::catalog::apply_migration_with_cover(&db.storage, &db.schema, &verbs, Some(&*db))
            .unwrap();

        let rev_key = KeyBuilder::reverse_edge(movie2.id, rec_id, user.id);
        let txn = db.storage().begin_txn();
        let rev_val = db
            .storage()
            .get(&txn, &rev_key)
            .unwrap()
            .expect("populated rev_edge cover exists");
        drop(txn);
        let cover = crate::object::deserialize_fields(&rev_val);
        // fix 3: the scalar's NEW name reached the renamed relation's own cover.
        assert!(
            cover.contains_key("handle"),
            "renamed scalar present in cover: {cover:?}"
        );
        assert!(
            !cover.contains_key("name"),
            "old scalar name gone from cover: {cover:?}"
        );
        // fix 1: verb 1's relation rename survived verb 2's overlay-read rewrite.
        assert!(
            cover.contains_key("suggested"),
            "renamed relation key present: {cover:?}"
        );
        assert!(
            !cover.contains_key("recommendation"),
            "old relation key gone: {cover:?}"
        );
    }

    #[test]
    fn rename_two_relations_one_plan() {
        // Two forward-1:1 relations of one type renamed in ONE plan: the second
        // verb must read the first verb's cover rewrite through the overlay (the
        // rev cover carries the OTHER relation's peer sidecars), else verb 2 would
        // clobber verb 1's rename. Both renamed keys must survive in the cover.
        use crate::catalog::RenameVerb;
        use rhypedb_storage::key::KeyBuilder;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                favourite: Movie
                recommendation: Movie
            }
            type Movie { title: String }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut m1 = FieldMap::new();
        m1.insert("title".into(), Value::String("Inception".into()));
        let movie1 = db.create("Movie", m1).unwrap();
        let mut m2 = FieldMap::new();
        m2.insert("title".into(), Value::String("Tenet".into()));
        let movie2 = db.create("Movie", m2).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        db.link("User", user.id, "favourite", movie1.id, None).unwrap();
        db.link("User", user.id, "recommendation", movie2.id, None)
            .unwrap();
        let rec_id = db.rel_ids()["User.recommendation"];

        let verbs = [
            RenameVerb::Field {
                type_name: "User".into(),
                old: "favourite".into(),
                new: "top_pick".into(),
            },
            RenameVerb::Field {
                type_name: "User".into(),
                old: "recommendation".into(),
                new: "suggested".into(),
            },
        ];
        crate::catalog::apply_migration_with_cover(&db.storage, &db.schema, &verbs, Some(&*db))
            .unwrap();

        // suggested's (=recommendation's) rev cover holds top_pick's (=favourite's)
        // peer sidecars AND its own renamed key. Both renames must survive.
        let rev_key = KeyBuilder::reverse_edge(movie2.id, rec_id, user.id);
        let txn = db.storage().begin_txn();
        let rev_val = db
            .storage()
            .get(&txn, &rev_key)
            .unwrap()
            .expect("rev cover exists");
        drop(txn);
        let cover = crate::object::deserialize_fields(&rev_val);
        assert!(
            cover.contains_key("suggested"),
            "own renamed key present: {cover:?}"
        );
        assert!(
            cover.contains_key("top_pick") || cover.contains_key("top_pick__cover"),
            "peer renamed sidecar present: {cover:?}"
        );
        assert!(
            !cover.contains_key("favourite") && !cover.contains_key("favourite__cover"),
            "old peer keys gone: {cover:?}"
        );
        assert!(
            !cover.contains_key("recommendation"),
            "old own key gone: {cover:?}"
        );
    }

    #[test]
    fn rename_plain_field_refreshes_sibling_indexed_cover() {
        // Latent phase-2 bug, fixed in phase 3: renaming a PLAIN scalar field
        // must refresh SIBLING @indexed covers — a covering blob is a full
        // object FieldMap, so it embeds the renamed plain field's name too.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type Movie {
                title: String
                year: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("Aliens".into()));
        f.insert("year".into(), Value::U32(1986));
        db.create("Movie", f).unwrap();
        db.storage.flush().unwrap();
        // Rename the PLAIN field `title`; `year` stays @indexed.
        db.rename_field("Movie", "title", "name").unwrap();
        drop(db);

        let after = parse_schema(
            r#"
            type Movie {
                name: String
                year: u32 @indexed
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        // A covered filter_scan on the SIBLING @indexed `year` must return the
        // object with the renamed plain field under its NEW name.
        let results = db2
            .filter_scan(
                "Movie",
                "year",
                rhypedb_storage::zone::CompareOp::Eq,
                1986,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].fields.get("name"),
            Some(&Value::String("Aliens".into())),
            "sibling @indexed cover must carry the renamed plain field under its new name; got {:?}",
            results[0].fields
        );
        assert!(
            !results[0].fields.contains_key("title"),
            "old name must be gone from the sibling cover: {:?}",
            results[0].fields
        );
    }

    #[test]
    fn rename_field_unique_preserves_constraint_post_rename() {
        // Phase 3 @unique lift: u: keys are field_id-keyed with object_id as
        // value, so the constraint survives a rename untouched.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                email: String @unique
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("email".into(), Value::String("a@x.com".into()));
        db.create("User", f).unwrap();
        db.rename_field("User", "email", "email_addr").unwrap();
        drop(db);

        let schema_after = parse_schema(
            r#"
            type User {
                name: String
                email_addr: String @unique
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        // The pre-rename value is still indexed under the same field_id, so a
        // colliding insert under the NEW name is rejected.
        let mut dup = FieldMap::new();
        dup.insert("name".into(), Value::String("Bob".into()));
        dup.insert("email_addr".into(), Value::String("a@x.com".into()));
        assert!(matches!(
            db2.create("User", dup),
            Err(EngineError::UniqueViolation { .. })
        ));
        // A different value still inserts.
        let mut ok = FieldMap::new();
        ok.insert("name".into(), Value::String("Carol".into()));
        ok.insert("email_addr".into(), Value::String("c@x.com".into()));
        db2.create("User", ok).unwrap();
    }

    #[test]
    fn rename_relation_field_rewrites_rev_edge_covers() {
        // Phase 3 relation lift: renaming a relation field rewrites both the
        // bare peer key AND the __cover / __cover_v sidecars embedded in OTHER
        // forward-1:1 relations' rev_edge covers.
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
                favourite: Movie
                recommendation: Movie
            }
            type Movie {
                title: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut m1 = FieldMap::new();
        m1.insert("title".into(), Value::String("Inception".into()));
        let movie1 = db.create("Movie", m1).unwrap();
        let mut m2 = FieldMap::new();
        m2.insert("title".into(), Value::String("Tenet".into()));
        let movie2 = db.create("Movie", m2).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        db.link("User", user.id, "favourite", movie1.id, None).unwrap();
        db.link("User", user.id, "recommendation", movie2.id, None)
            .unwrap();

        // Rename the `favourite` RELATION field. Its peer sidecars
        // (favourite__cover / __cover_v) live inside `recommendation`'s
        // rev_edge cover; its bare key lives in its own rev_edge cover.
        let report = db.rename_field("User", "favourite", "top_pick").unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        assert!(
            report.renamed_fields[0].covers_rewritten >= 1,
            "expected at least one rev_edge cover rewrite, got {}",
            report.renamed_fields[0].covers_rewritten
        );

        use rhypedb_storage::key::KeyBuilder;
        // recommendation's rev_edge cover embeds the `favourite` peer sidecars.
        let rec_rel_id = db.rel_ids()["User.recommendation"];
        let rev_key = KeyBuilder::reverse_edge(movie2.id, rec_rel_id, user.id);
        let txn = db.storage().begin_txn();
        let rev_val = db
            .storage()
            .get(&txn, &rev_key)
            .unwrap()
            .expect("rev_edge for the recommendation link must exist");
        drop(txn);
        let cover = crate::object::deserialize_fields(&rev_val);
        assert!(
            cover.contains_key("top_pick"),
            "renamed peer key present in cover: {cover:?}"
        );
        assert!(
            cover.contains_key("top_pick__cover"),
            "renamed peer __cover present in cover: {cover:?}"
        );
        assert!(
            !cover.contains_key("favourite") && !cover.contains_key("favourite__cover"),
            "old peer keys must be gone from cover: {cover:?}"
        );
    }

    #[test]
    fn cascade_delete_via_renamed_relation_field() {
        // Phase 3 relation lift: an @on_delete(cascade) relation still fires
        // after the relation field is renamed — the rebuilt cascade cache +
        // rewritten rev_edge cover resolve the NEW name.
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Hello".into()));
        pf.insert("author".into(), Value::U64(user.id));
        let post = db.create("Post", pf).unwrap();

        let report = db.rename_field("Post", "author", "writer").unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        drop(db);

        let schema_after = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                writer: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        assert!(db2.get("Post", post.id).is_ok(), "post survives the rename");
        // Deleting the referenced user cascades to delete the post, resolving
        // the relationship via the renamed `writer` field.
        db2.delete("User", user.id).unwrap();
        assert!(
            db2.get("Post", post.id).is_err(),
            "post must be cascade-deleted via the renamed relation"
        );
        assert!(db2.get("User", user.id).is_err());
    }

    #[test]
    fn rename_many_relation_field_preserves_links() {
        // A MANY relation has no rev_edge covers (covers_rewritten == 0), but
        // the dual catalog re-key + reopen must keep forward links resolvable
        // under the NEW name (edges are rel_id-keyed and stable).
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
                friends: [User]
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut a = FieldMap::new();
        a.insert("name".into(), Value::String("A".into()));
        let ua = db.create("User", a).unwrap();
        let mut b = FieldMap::new();
        b.insert("name".into(), Value::String("B".into()));
        let ub = db.create("User", b).unwrap();
        db.link("User", ua.id, "friends", ub.id, None).unwrap();

        let report = db.rename_field("User", "friends", "buddies").unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        assert_eq!(
            report.renamed_fields[0].covers_rewritten, 0,
            "a many relation carries no rev_edge covers to rewrite"
        );
        drop(db);

        let after = parse_schema(
            r#"
            type User {
                name: String
                buddies: [User]
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(after, dir.path()).unwrap();
        let links = db2.get_links("User", ua.id, "buddies").unwrap();
        assert_eq!(links.len(), 1, "the link must resolve under the new name");
        assert_eq!(links[0].0, ub.id);
    }

    /// rename_field rewrites the embedded source-side FieldMap inside
    /// every `r:<target>:<rel>:<source>` reverse-edge cover blob whose
    /// source is an object of the renamed type. Without this, the
    /// executor's covering-fast-path reads stale field names directly
    /// out of the cover bytes (the `cover_v` stamp matches because
    /// rename doesn't bump it, so the staleness fall-through never
    /// fires). Regression for the PR #6 adversarial-review blocker.
    ///
    /// Setup: User with TWO forward 1:1 relations (`favourite` +
    /// `recommendation`) so `build_covering_rev_value` writes a
    /// non-empty cover (the empty-when-no-peer optimization at
    /// `database.rs::build_covering_rev_value` line 3391-3393 keeps
    /// the cover empty for single-relation types).
    #[test]
    fn rename_field_rewrites_rev_edge_cover_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(
            r#"
            type User {
                name: String
                favourite: Movie
                recommendation: Movie
            }
            type Movie {
                title: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let mut m1 = FieldMap::new();
        m1.insert("title".into(), Value::String("Inception".into()));
        let movie1 = db.create("Movie", m1).unwrap();
        let mut m2 = FieldMap::new();
        m2.insert("title".into(), Value::String("Tenet".into()));
        let movie2 = db.create("Movie", m2).unwrap();

        let mut user_fields = FieldMap::new();
        user_fields.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", user_fields).unwrap();

        // Link both forward 1:1 — the SECOND link triggers
        // `build_covering_rev_value` to write a non-empty cover for
        // BOTH rev_edges (it re-emits the first one as part of the
        // covering pass).
        db.link("User", user.id, "favourite", movie1.id, None).unwrap();
        db.link("User", user.id, "recommendation", movie2.id, None).unwrap();

        let report = db.rename_field("User", "name", "handle").unwrap();
        assert_eq!(report.renamed_fields.len(), 1);
        let pair = &report.renamed_fields[0];
        assert_eq!(pair.objects_rewritten, 1);
        assert!(
            pair.covers_rewritten >= 1,
            "expected at least 1 rev_edge cover rewrite, got {}",
            pair.covers_rewritten,
        );

        // Read the rev_edge for the second link directly and assert
        // its embedded FieldMap has `handle` (not `name`).
        use rhypedb_storage::key::KeyBuilder;
        let rec_rel_id = db.rel_ids()["User.recommendation"];
        let rev_key = KeyBuilder::reverse_edge(movie2.id, rec_rel_id, user.id);
        let txn = db.storage().begin_txn();
        let rev_val = db
            .storage()
            .get(&txn, &rev_key)
            .unwrap()
            .expect("rev_edge for the linked pair must exist");
        drop(txn);
        assert!(
            !rev_val.is_empty(),
            "rev_edge value must be a non-empty cover (two forward 1:1 → coverable)",
        );
        let cover_fields = crate::object::deserialize_fields(&rev_val);
        assert_eq!(
            cover_fields.get("handle"),
            Some(&Value::String("Alice".into())),
            "rev_edge cover must carry the value under the NEW field name post-rename; got {cover_fields:?}",
        );
        assert!(
            !cover_fields.contains_key("name"),
            "rev_edge cover must NOT retain the OLD field name post-rename; got {cover_fields:?}",
        );
    }

    /// The migration write barrier excludes concurrent `create` calls
    /// from running during `rename_field`. Without it, a `create()` mid-
    /// migration would commit a new object with the OLD field-name in
    /// its serialized FieldMap, and that name would NOT be in the
    /// migration's write-set — MVCC misses the conflict, and the object
    /// lands stale in the post-rename catalog era.
    ///
    /// Setup: spawn N writer threads doing `create()` in a tight loop;
    /// from the main thread, call `rename_field`; wait for writers; then
    /// scan all objects and assert EVERY one has the new field name.
    #[test]
    fn rename_field_excludes_concurrent_writers() {
        use std::sync::atomic::{AtomicBool, Ordering as AOrd};
        use std::sync::Arc as StdArc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                age: u32
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let stop = StdArc::new(AtomicBool::new(false));
        let mut writer_handles = Vec::new();
        for _ in 0..4 {
            let db = StdArc::clone(&db);
            let stop = StdArc::clone(&stop);
            writer_handles.push(thread::spawn(move || {
                while !stop.load(AOrd::Relaxed) {
                    let mut f = FieldMap::new();
                    f.insert("name".into(), Value::String("Alice".into()));
                    f.insert("age".into(), Value::U32(30));
                    let _ = db.create("User", f);
                }
            }));
        }

        // Let writers warm up briefly so there's a queue of `create()`
        // calls blocked on `migration_lock.read()` when the rename
        // takes the write lock.
        std::thread::sleep(std::time::Duration::from_millis(20));

        db.rename_field("User", "name", "handle").unwrap();

        stop.store(true, AOrd::Relaxed);
        for h in writer_handles {
            h.join().unwrap();
        }

        // After the barrier, all subsequent creates have the new
        // schema — but the barrier doesn't update self.schema (still
        // post-verb stale). So writes through this OLD handle still
        // write under the OLD name. Drop and reopen to validate.
        drop(db);
        let schema_after = parse_schema(
            r#"
            type User {
                handle: String
                age: u32
            }
            "#,
        )
        .unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        let all = db2.scan_type("User").unwrap();
        // Every object should have `handle` (rewritten from `name` for
        // pre-rename objects; the writers may have committed before OR
        // after the rename. Writes before the rename see the OLD lock
        // state, write `name`, get rewritten by the migration. Writes
        // after the rename, on the OLD handle, write `name` again under
        // the in-memory stale schema — the barrier doesn't help that
        // case; PR B's poison flag does. For PR A the assertion is
        // milder: PRE-rename writes (under the barrier) end up with
        // `handle` after the verb. We can't easily separate the two
        // populations without timestamp injection, so just assert the
        // count is non-zero and that NO object has BOTH name AND
        // handle (would indicate a partial rewrite).
        for obj in &all {
            assert!(
                !(obj.fields.contains_key("name") && obj.fields.contains_key("handle")),
                "object {} has both old and new field names: {:?}",
                obj.id,
                obj.fields,
            );
        }
    }

    /// `Database::rename_field` is idempotent when wrapped in
    /// `run_migrations` — a second call with the same migration list
    /// is a no-op rather than re-running the verb (which would refuse
    /// with `RenameSourceNotFound` because the old name is gone).
    #[test]
    fn rename_field_idempotent_via_run_migrations() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let u = db.create("User", f).unwrap();

        db.run_migrations(vec![crate::catalog::Migration::new(
            "001_rename_name_to_handle",
            |m| m.rename_field("User", "name", "handle"),
        )])
        .unwrap();
        drop(db);

        // Re-open with post-rename schema; replay should be a no-op.
        let schema_after = parse_schema(r#"type User { handle: String }"#).unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        db2.run_migrations(vec![crate::catalog::Migration::new(
            "001_rename_name_to_handle",
            |_| Ok(()),
        )])
        .unwrap();
        let r = db2.get("User", u.id).unwrap();
        assert_eq!(
            r.fields.get("handle"),
            Some(&Value::String("Alice".into()))
        );
    }

    // -----------------------------------------------------------------
    // _consuming migrate variants (PR B) — integration tests
    // -----------------------------------------------------------------

    /// The consuming variant returns a new Arc that observes the
    /// post-rename schema immediately. The old handle still resolves to
    /// the OLD schema's view (and is poisoned with
    /// `DatabaseMigratedAway` for read/write APIs).
    #[test]
    fn rename_type_consuming_returns_handle_observing_new_name() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let u = db.create("User", f).unwrap();

        let old_arc = Arc::clone(&db);
        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (report, db2) = db.rename_type_consuming("User", "Account", schema_after).unwrap();
        assert_eq!(report.renamed_types.len(), 1);

        // New handle resolves the new name immediately, no reopen.
        let r = db2.get("Account", u.id).unwrap();
        assert_eq!(r.type_name, "Account");
        assert_eq!(
            r.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );

        // Old handle is poisoned: any read/write surfaces
        // DatabaseMigratedAway via the `resolve_type_id` chokepoint.
        let err = old_arc.get("User", u.id).unwrap_err();
        assert!(
            matches!(err, EngineError::DatabaseMigratedAway),
            "old handle should be poisoned, got {err}"
        );
    }

    /// rename_field via the consuming variant exposes the renamed
    /// field on the new handle without needing a drop+reopen.
    #[test]
    fn rename_field_consuming_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String age: u32 }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("age".into(), Value::U32(30));
        let u = db.create("User", f).unwrap();

        let schema_after = parse_schema(r#"type User { handle: String age: u32 }"#).unwrap();
        let (_report, db2) = db.rename_field_consuming("User", "name", "handle", schema_after).unwrap();

        let r = db2.get("User", u.id).unwrap();
        assert_eq!(
            r.fields.get("handle"),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(r.fields.get("age"), Some(&Value::U32(30)));
    }

    /// run_migrations via the consuming variant returns the log report
    /// AND a fresh handle.
    #[test]
    fn run_migrations_consuming_returns_report_and_handle() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (report, db2) = db
            .run_migrations_consuming(
                vec![crate::catalog::Migration::new("001_rename", |m| {
                    m.rename_type("User", "Account")
                })],
                schema_after,
            )
            .unwrap();
        assert_eq!(report.applied.len(), 1);

        // New handle works under the post-migration schema.
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Bob".into()));
        db2.create("Account", f).unwrap();
    }

    /// The new handle shares the same `Arc<LsmTree>` (no WAL replay,
    /// no SST rediscovery, no compaction-worker thread churn). We can
    /// observe this by checking that `Arc::ptr_eq` on the storage
    /// references holds across the migrate.
    #[test]
    fn migrate_reuses_arc_lsm_tree() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let old_storage = Arc::clone(db.storage());
        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, db2) = db.rename_type_consuming("User", "Account", schema_after).unwrap();
        assert!(
            Arc::ptr_eq(&old_storage, db2.storage()),
            "migrate must reuse the existing Arc<LsmTree> — no reopen"
        );
    }

    /// The next_object_id counter is carried forward so the new handle
    /// doesn't restart numbering. Without this carry, a delete + create
    /// across migrate would resurface a previously-used id.
    #[test]
    fn migrate_carries_next_object_id() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let u1 = db.create("User", f).unwrap();
        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        let u2 = db.create("User", f2).unwrap();

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, db2) = db.rename_type_consuming("User", "Account", schema_after).unwrap();

        let mut f3 = FieldMap::new();
        f3.insert("name".into(), Value::String("Carol".into()));
        let u3 = db2.create("Account", f3).unwrap();
        assert!(
            u3.id > u1.id && u3.id > u2.id,
            "migrate must carry next_object_id (u1={}, u2={}, u3={})",
            u1.id,
            u2.id,
            u3.id,
        );
    }

    /// Live subscribers stay connected across a migrate — their
    /// `Receiver`s are still valid because the `Arc<SubscriptionHub>` is
    /// shared between the old and new `Database` instances.
    #[test]
    fn migrate_keeps_live_subscribers_connected() {
        use rhypedb_subscribe::SubscriptionFilter;

        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let (_id, rx) = db.subscriptions().subscribe(SubscriptionFilter {
            type_name: Some("Account".into()),
            kinds: Vec::new(),
            object_id: None,
        });

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, db2) = db.rename_type_consuming("User", "Account", schema_after).unwrap();

        // Trigger an event via the NEW handle.
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        db2.create("Account", f).unwrap();

        // The subscriber's receiver, opened against the OLD hub, sees
        // the event from the NEW handle — proving the hub is shared.
        let evt = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(evt.type_name, "Account");
    }

    /// `next_object_id` must be SHARED (not snapshotted) between OLD
    /// and NEW handles. Otherwise a concurrent `create()` on OLD that
    /// races the migration's snapshot can mint the same ID NEW's first
    /// `create()` will also mint — silent overwrite on the shared
    /// `Arc<LsmTree>`. Regression for PR #7 adversarial-review blocker.
    #[test]
    fn migrate_shares_next_object_id_arc() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, db2) =
            db.rename_type_consuming("User", "Account", schema_after).unwrap();

        // Both handles must point to the SAME AtomicU64 — `Arc::ptr_eq`
        // is the structural guarantee.
        assert!(
            Arc::ptr_eq(&db.next_object_id, &db2.next_object_id),
            "next_object_id Arc must be shared between OLD and NEW handles",
        );
    }

    /// Same as above for `version_counters` and `migration_lock` —
    /// PR #7 blockers + majors collapse into "carry the shared Arcs".
    #[test]
    fn migrate_shares_version_counters_and_migration_lock() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, db2) =
            db.rename_type_consuming("User", "Account", schema_after).unwrap();

        assert!(
            Arc::ptr_eq(&db.version_counters, &db2.version_counters),
            "version_counters Arc must be shared",
        );
        assert!(
            Arc::ptr_eq(&db.version_counter_count, &db2.version_counter_count),
            "version_counter_count Arc must be shared",
        );
        assert!(
            Arc::ptr_eq(&db.migration_lock, &db2.migration_lock),
            "migration_lock Arc must be shared so writers on either handle serialize against migrations on either handle",
        );
    }

    /// A `_consuming` verb that errors out must NOT consume the
    /// caller's Arc. With the prior `self: Arc<Self>` signature the Arc
    /// was moved into the function and dropped on `?` early-return,
    /// destroying the caller's only handle on benign validation errors
    /// (no-op, source-not-found, etc.). The `self: &Arc<Self>` switch
    /// preserves the caller's Arc through any failure path.
    #[test]
    fn consuming_verb_preserves_arc_on_validation_error() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        // No-op rename (old == new) is rejected as RenameNoOp.
        let post_schema = parse_schema(r#"type User { name: String }"#).unwrap();
        match db.rename_type_consuming("User", "User", post_schema) {
            Err(EngineError::Catalog(crate::CatalogError::RenameNoOp { .. })) => {}
            Ok(_) => panic!("expected RenameNoOp"),
            Err(e) => panic!("expected RenameNoOp, got {e}"),
        }

        // The caller's `db` Arc is still alive AND usable — the verb
        // didn't poison it because the verb itself errored before
        // reaching the rebuild + poison sequence.
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        db.create("User", f).unwrap();
    }

    /// Non-consuming migrate verbs must reject calls on a handle that
    /// was already migrated away via a `_consuming` verb. Otherwise a
    /// stale `Arc::clone` could run another migration against the OLD
    /// in-memory schema (which doesn't match the on-disk catalog
    /// anymore) and corrupt.
    #[test]
    fn non_consuming_verbs_reject_after_consuming_migrate() {
        let dir = tempfile::tempdir().unwrap();
        let schema_before = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let old_handle = Arc::clone(&db);

        let schema_after = parse_schema(r#"type Account { name: String }"#).unwrap();
        let (_report, _new_db) =
            db.rename_type_consuming("User", "Account", schema_after).unwrap();

        // `old_handle` is the same Arc — it's now poisoned.
        // Direct `rename_type` (non-consuming) must reject.
        let err = old_handle.rename_type("Account", "Member").unwrap_err();
        assert!(
            matches!(err, EngineError::DatabaseMigratedAway),
            "non-consuming rename_type on poisoned handle must error DatabaseMigratedAway, got {err}",
        );
    }

    // -----------------------------------------------------------------
    // change_field_type (card 4/5) — integration tests
    // -----------------------------------------------------------------

    /// Convert i64 age values to f64. All existing rows are re-encoded
    /// in one atomic batch; reopening with the new schema reads them
    /// back as floats.
    #[test]
    fn change_field_type_int_to_float_round_trip() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let schema_before =
            parse_schema(r#"type User { name: String  score: i64 }"#).unwrap();
        let db = Database::open(schema_before, dir.path()).unwrap();
        let scores = [5i64, 10, 15, 20];
        let mut ids = Vec::new();
        for s in scores {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("u{s}")));
            f.insert("score".into(), Value::I64(s));
            ids.push(db.create("User", f).unwrap().id);
        }
        let report = db
            .change_field_type(
                "User",
                "score",
                FieldType::Scalar(ScalarType::F64),
                |_oid, v| {
                    let i = match v {
                        Value::I64(i) => *i,
                        other => {
                            return Err(EngineError::Catalog(
                                crate::CatalogError::FieldTypeChangeConverterFailed {
                                    qualified: "User.score".into(),
                                    object_id: 0,
                                    reason: format!("expected I64, got {}", other.type_name()),
                                },
                            ));
                        }
                    };
                    Ok(Value::F64(i as f64))
                },
            )
            .unwrap();
        assert_eq!(report.field_type_changes.len(), 1);
        assert_eq!(report.field_type_changes[0].objects_converted, 4);
        assert_eq!(report.catalog_format_after, 4); // CATALOG_FORMAT_V4
        drop(db);

        // Reopen with the post-change schema.
        let schema_after =
            parse_schema(r#"type User { name: String  score: f64 }"#).unwrap();
        let db2 = Database::open(schema_after, dir.path()).unwrap();
        for (id, expected) in ids.iter().zip(scores.iter()) {
            let obj = db2.get("User", *id).unwrap();
            match obj.fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, *expected as f64),
                other => panic!("expected F64, got {other:?}"),
            }
        }
    }

    /// Trying to change the type of an @indexed field is refused.
    /// Index rebuild is deferred to a follow-on card.
    #[test]
    fn change_field_type_on_indexed_field_refused() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { age: i64 @indexed }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let err = db
            .change_field_type(
                "User",
                "age",
                FieldType::Scalar(ScalarType::F64),
                |_, _| Ok(Value::F64(0.0)),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeDirectiveUnsupported {
                directive: "@indexed",
                ..
            })
        ));
    }

    /// Same source and target kind → NoOp refusal.
    #[test]
    fn change_field_type_same_kind_is_no_op_refused() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { age: i64 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let err = db
            .change_field_type(
                "User",
                "age",
                FieldType::Scalar(ScalarType::I64),
                |_, v| Ok(v.clone()),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeNoOp { .. })
        ));
    }

    /// Converter returning the wrong kind is caught.
    #[test]
    fn change_field_type_converter_wrong_kind_refused() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { age: i64 }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("age".into(), Value::I64(42));
        db.create("User", f).unwrap();
        // Closure returns String even though target is F64.
        let err = db
            .change_field_type(
                "User",
                "age",
                FieldType::Scalar(ScalarType::F64),
                |_, _| Ok(Value::String("nope".into())),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterReturnedWrongKind { .. }
            )
        ));
    }

    /// Converter returning Err aborts the migration; catalog state is
    /// unchanged.
    #[test]
    fn change_field_type_converter_error_aborts_atomically() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { age: i64 }"#).unwrap();
        let db = Database::open(schema.clone(), dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("age".into(), Value::I64(42));
        let user = db.create("User", f).unwrap();
        let err = db
            .change_field_type(
                "User",
                "age",
                FieldType::Scalar(ScalarType::F64),
                |_, _| {
                    Err(EngineError::Catalog(
                        crate::CatalogError::FieldTypeChangeConverterFailed {
                            qualified: "User.age".into(),
                            object_id: 0,
                            reason: "user said no".into(),
                        },
                    ))
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeConverterFailed { .. })
        ));
        drop(db);
        // Reopen — object is unchanged.
        let db2 = Database::open(schema, dir.path()).unwrap();
        let obj = db2.get("User", user.id).unwrap();
        assert_eq!(obj.fields.get("age"), Some(&Value::I64(42)));
    }

    /// Offline regression: a field-type change with zero objects flips the
    /// catalog kind, reports `objects_converted == 0`, bumps format to V4,
    /// and creates NO migration plan (the offline path is plan-free).
    #[test]
    fn apply_field_type_change_no_objects_noop_is_unchanged() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let report = db
            .change_field_type(
                "User",
                "score",
                FieldType::Scalar(ScalarType::F64),
                |_, v| match v {
                    Value::I64(i) => Ok(Value::F64(*i as f64)),
                    _ => Ok(Value::F64(0.0)),
                },
            )
            .unwrap();
        assert_eq!(report.field_type_changes[0].objects_converted, 0);
        assert_eq!(report.catalog_format_after, 4);
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Catalog flipped (reopen with f64 succeeds); offline path persists
        // no plan.
        assert!(db2.list_migrations().unwrap().is_empty());
    }

    // -----------------------------------------------------------------
    // Chunked field-type migration (shadow-field card 1/5, increment 3)
    // -----------------------------------------------------------------

    /// A field-type change must BUMP each converted object's generation so a
    /// rev-edge cover embedding the old field value is invalidated (the fusion
    /// reader serves a cover iff `cover_v == object_version`; born-at-1 makes
    /// every live object generation >= 1, so without a bump `cover_v ==
    /// object_version` keeps holding and a covered read serves the STALE source
    /// value). Verified for BOTH the offline single-commit path and the
    /// chunked migration path. Fails (object_version stays 1) without the
    /// stage_generation_bump fix.
    #[test]
    fn field_type_change_bumps_generation_to_invalidate_covers() {
        use rhypedb_schema::{FieldType, ScalarType};
        // --- offline single-commit path ---
        let dir = tempfile::tempdir().unwrap();
        let off_id = {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(5));
            let id = db.create("User", f).unwrap().id;
            assert_eq!(db.object_version("User", id), 1, "born-at-1");
            db.change_field_type(
                "User",
                "score",
                FieldType::Scalar(ScalarType::F64),
                |_, v| match v {
                    Value::I64(i) => Ok(Value::F64(*i as f64)),
                    _ => Ok(Value::F64(0.0)),
                },
            )
            .unwrap();
            id
        };
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert!(
            db.object_version("User", off_id) > 1,
            "offline change_field_type must bump generation (covers would serve stale source); got {}",
            db.object_version("User", off_id)
        );

        // --- chunked migration path ---
        let dir2 = tempfile::tempdir().unwrap();
        let chunk_id = {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir2.path(),
            )
            .unwrap();
            db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(7));
            let id = db.create("User", f).unwrap().id;
            let plan_id = db
                .create_field_type_migration(MigrationPlanSpec {
                    type_name: "User".into(),
                    field_name: "score".into(),
                    target_field_type: FieldType::Scalar(ScalarType::F64),
                    converter_name: "widen".into(),
                    converter_version: 1,
                    chunk_size: 4, ..Default::default()
                })
                .unwrap();
            db.wait_for_migration(plan_id).unwrap();
            id
        };
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir2.path(),
        )
        .unwrap();
        assert!(
            db2.object_version("User", chunk_id) > 1,
            "chunked migration must bump generation; got {}",
            db2.object_version("User", chunk_id)
        );
    }

    // A converter that widens I64 -> F64; anything else is a hard error
    // (so a double-conversion or wrong-kind row surfaces, not silently
    // coerces).
    fn widen_i64_to_f64(name: &str) -> impl Fn(u64, &Value) -> EngineResult<Value> {
        let q = name.to_string();
        move |oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            other => Err(EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterFailed {
                    qualified: q.clone(),
                    object_id: oid,
                    reason: format!("expected I64, got {}", other.type_name()),
                },
            )),
        }
    }

    #[test]
    fn create_field_type_migration_converts_all_and_completes() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { name: String  score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let scores = [5i64, 10, 15, 20, 25, 30, 35];
        let mut ids = Vec::new();
        for s in scores {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("u{s}")));
            f.insert("score".into(), Value::I64(s));
            ids.push(db.create("User", f).unwrap().id);
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 2, ..Default::default()
            })
            .unwrap();
        assert_eq!(plan_id, 1);
        db.wait_for_migration(plan_id).unwrap();
        let summary = db.list_migrations().unwrap();
        assert_eq!(summary.len(), 1);
        assert_eq!(
            summary[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        assert_eq!(summary[0].objects_converted, scores.len() as u64);
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type User { name: String  score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for (id, s) in ids.iter().zip(scores.iter()) {
            match db2.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, *s as f64),
                other => panic!("expected F64, got {other:?}"),
            }
        }
    }

    /// Card 5 (5a): `start_field_type_migration_async` returns a handle;
    /// `query_migration_progress` aggregates the per-partition cursors and
    /// reports a settled plan with no ETA; `list_migrations_filtered` ANDs the
    /// status + type filters.
    #[test]
    fn card5_query_progress_and_filter() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        for s in [1i64, 2, 3, 4, 5] {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(s));
            db.create("User", f).unwrap();
        }
        let handle = db
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 2,
                ..Default::default()
            })
            .unwrap();
        assert!(handle.created_at_ms > 0, "handle carries created_at_ms");
        db.wait_for_migration(handle.plan_id).unwrap();

        let p = db.query_migration_progress(handle.plan_id).unwrap();
        assert_eq!(p.status, crate::catalog::MigrationStatus::Completed);
        assert_eq!(p.objects_converted, 5);
        assert_eq!(p.total_objects, 5, "U-1 over [1,U) with 5 objects");
        assert!(p.eta_unix_ms.is_none(), "settled plan has no ETA");
        assert!(p.objects_per_sec.is_none());
        assert!(!p.partitions.is_empty());
        assert!(p.partitions.iter().all(|pp| pp.done));
        assert_eq!(
            p.partitions.iter().map(|pp| pp.objects_converted).sum::<u64>(),
            5
        );

        // Filters.
        assert!(
            db.list_migrations_filtered(&MigrationFilter {
                status: Some(crate::catalog::MigrationStatus::Running),
                ..Default::default()
            })
            .unwrap()
            .is_empty()
        );
        assert_eq!(
            db.list_migrations_filtered(&MigrationFilter {
                status: Some(crate::catalog::MigrationStatus::Completed),
                type_name: Some("User".into()),
            })
            .unwrap()
            .len(),
            1
        );
        assert!(
            db.list_migrations_filtered(&MigrationFilter {
                type_name: Some("Other".into()),
                ..Default::default()
            })
            .unwrap()
            .is_empty()
        );
    }

    /// Hot schema-reload: after an async `change_field_type` settles, the live
    /// handle's in-memory schema is stale (source kind). `reload_handle` swaps in
    /// a fresh handle on the SAME storage under the target schema — proving the
    /// in-place fix for the post-cutover stale handle (the offline analog
    /// drop()+reopen is `migrate_change_field_type_i64_to_f64_reopen`). Also
    /// proves it is NON-poisoning: a failing reload leaves the handle live.
    #[test]
    fn reload_handle_swaps_schema_in_place_shares_storage_no_poison() {
        use rhypedb_schema::{FieldType, ScalarType};
        // A second, non-migrated type lets us exercise the OLD handle after the
        // reload without touching the migrated field — a poisoned handle would
        // return `DatabaseMigratedAway` for any resolve.
        const SRC: &str = r#"type User { score: i64 } type Tag { label: String }"#;
        const DST: &str = r#"type User { score: f64 } type Tag { label: String }"#;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(SRC).unwrap(), dir.path()).unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let scores = [5i64, 10, 15, 20, 25];
        let mut ids = Vec::new();
        for s in scores {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(s));
            ids.push(db.create("User", f).unwrap().id);
        }
        let handle = db
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 2,
                ..Default::default()
            })
            .unwrap();
        db.wait_for_migration(handle.plan_id).unwrap();

        // The reload: same storage, target schema. Migrated rows decode as F64
        // through the new handle without any drop()/reopen().
        let reloaded = db.reload_handle(parse_schema(DST).unwrap()).unwrap();
        for (id, s) in ids.iter().zip(scores.iter()) {
            match reloaded.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, *s as f64),
                other => panic!("expected F64 after reload, got {other:?}"),
            }
        }
        // Shared storage + next_object_id: a create on the reloaded handle
        // continues the id sequence from the OLD handle (5 User objects → id 6).
        let mut t = FieldMap::new();
        t.insert("label".into(), Value::String("a".into()));
        assert_eq!(reloaded.create("Tag", t).unwrap().id, 6);

        // NON-poisoning: the OLD handle is still live (it was never marked
        // migrated). Resolving a non-migrated type succeeds where a poisoned
        // handle would return `DatabaseMigratedAway`.
        let mut t2 = FieldMap::new();
        t2.insert("label".into(), Value::String("b".into()));
        let old_tag = db
            .create("Tag", t2)
            .expect("old handle must stay live after a non-poisoning reload");
        assert_eq!(old_tag.id, 7, "old + new handles share next_object_id");
    }

    /// Hot schema-reload MUST refuse while a migration is in flight: the rebuilt
    /// handle does not carry `migrating_fields`, so reloading mid-backfill would
    /// silently disarm the double-write hook → source-only writes that cutover
    /// loses. A converter that blocks on the first row holds the plan provably
    /// armed-but-unsettled while we attempt the reload.
    #[test]
    fn reload_handle_refused_while_migration_armed() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let entered_tx = std::sync::Mutex::new(entered_tx);
        let release_rx = std::sync::Mutex::new(release_rx);
        let blocked_once = AtomicBool::new(false);
        db.register_converter("widen", 1, move |_oid, v| {
            // Block only on the FIRST row so the plan is armed-but-unsettled while
            // the test attempts the reload; subsequent rows pass straight through.
            if !blocked_once.swap(true, Ordering::SeqCst) {
                entered_tx.lock().unwrap().send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
            match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                _ => unreachable!(),
            }
        });
        for s in [1i64, 2, 3] {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(s));
            db.create("User", f).unwrap();
        }
        let handle = db
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 1,
                parallel_degree: Some(1),
                ..Default::default()
            })
            .unwrap();
        // A worker is now inside the converter → hook armed, plan running.
        entered_rx.recv().unwrap();
        let err = match db.reload_handle(parse_schema(r#"type User { score: f64 }"#).unwrap()) {
            Ok(_) => panic!("reload must be refused while a migration is armed"),
            Err(e) => e,
        };
        assert!(
            matches!(err, EngineError::ReloadBlockedByActiveMigration { armed } if armed > 0),
            "reload must refuse a mid-flight migration, got {err:?}"
        );
        // Release; once settled, the same reload succeeds.
        release_tx.send(()).unwrap();
        db.wait_for_migration(handle.plan_id).unwrap();
        let reloaded = db
            .reload_handle(parse_schema(r#"type User { score: f64 }"#).unwrap())
            .unwrap();
        assert!(reloaded.get("User", 1).is_ok());
    }

    /// Card 5 AC `progress_eta_within_25pct_of_actual_on_uniform_load`: with a
    /// converter that takes a fixed time per row (uniform load), the rate-based
    /// ETA sampled mid-flight predicts the actual completion within 25%.
    /// Single-partition for deterministic linear pacing.
    #[test]
    fn card5_eta_within_25pct_on_uniform_load() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::time::{Duration, Instant};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        // ~2ms per row → ~800ms over 400 rows; uniform load.
        db.register_converter("widen", 1, |_oid, v| {
            std::thread::sleep(Duration::from_millis(2));
            match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                _ => unreachable!(),
            }
        });
        const N: i64 = 400;
        for i in 0..N {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let t0 = Instant::now();
        let handle = db
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 8,
                parallel_degree: Some(1), // single worker → linear pacing
                ..Default::default()
            })
            .unwrap();
        // Sample once a stable fraction (20–70%) has converted.
        let mut sample = None;
        for _ in 0..5000 {
            let p = db.query_migration_progress(handle.plan_id).unwrap();
            if p.status == crate::catalog::MigrationStatus::Running
                && (80..=280).contains(&p.objects_converted)
                && p.eta_unix_ms.is_some()
            {
                sample = Some(p);
                break;
            }
            if !p.status.quiesces() {
                break; // finished too fast — fail below
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        db.wait_for_migration(handle.plan_id).unwrap();
        let actual_ms = t0.elapsed().as_millis() as u64;
        let p = sample.expect("should have caught an in-flight sample");
        let predicted_total_ms = p.eta_unix_ms.unwrap().saturating_sub(p.created_at_ms);
        let lo = actual_ms * 75 / 100;
        let hi = actual_ms * 125 / 100;
        assert!(
            predicted_total_ms >= lo && predicted_total_ms <= hi,
            "ETA predicted total {predicted_total_ms}ms not within 25% of actual {actual_ms}ms \
             (sampled at {} / {} converted)",
            p.objects_converted,
            p.total_objects
        );
    }

    /// Card 5 (5b) AC `events_stream_emits_chunkcompleted_partitiondone_and_cutoverdone`:
    /// a subscriber attached before start sees >=1 ChunkCompleted, >=1
    /// PartitionDone, and exactly 1 CutoverDone (+ the Completed StatusChanged).
    #[test]
    fn card5_events_stream_emits_expected_sequence() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        for s in 0i64..10 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(s));
            db.create("User", f).unwrap();
        }
        // Subscribe BEFORE starting so the full sequence is observed.
        let plan_id = 1u64;
        let rx = db.subscribe_migration_events(plan_id);
        let started = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 3,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(started, plan_id);
        db.wait_for_migration(plan_id).unwrap();

        // Drain (sender side is closed once the driver thread exits + the hub is
        // dropped at db drop; here we just collect what arrived).
        let mut chunk = 0usize;
        let mut part_done = 0usize;
        let mut cutover_started = 0usize;
        let mut cutover_done = 0usize;
        let mut completed = 0usize;
        while let Ok(ev) = rx.recv_timeout(std::time::Duration::from_secs(2)) {
            match ev {
                MigrationEvent::ChunkCompleted { plan_id: p, .. } => {
                    assert_eq!(p, plan_id);
                    chunk += 1;
                }
                MigrationEvent::PartitionDone { .. } => part_done += 1,
                MigrationEvent::CutoverStarted { .. } => cutover_started += 1,
                MigrationEvent::CutoverDone { .. } => cutover_done += 1,
                MigrationEvent::StatusChanged { status, .. } => {
                    if status == crate::catalog::MigrationStatus::Completed {
                        completed += 1;
                        break; // terminal
                    }
                }
                _ => {}
            }
        }
        assert!(chunk >= 1, "expected >=1 ChunkCompleted, got {chunk}");
        assert!(part_done >= 1, "expected >=1 PartitionDone, got {part_done}");
        assert_eq!(cutover_started, 1, "exactly one CutoverStarted");
        assert_eq!(cutover_done, 1, "exactly one CutoverDone");
        assert_eq!(completed, 1, "exactly one Completed StatusChanged");
    }

    /// Acceptance: both passes commit at chunk boundaries — `ceil(rows/chunk)`
    /// commits each for the shadow backfill AND the cutover, NOT one big commit.
    #[test]
    fn create_field_type_migration_chunks_commit_at_chunk_boundary() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        // Disable the cover-refresh worker so commit counting is
        // deterministic (it never fires here — no updates — but be explicit).
        let opts = OpenOptions {
            background_cover_refresh: false,
            ..OpenOptions::default()
        };
        let db = Database::open_with_options(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
            opts,
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        for i in 0..10i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let before = db.storage.txn_manager().current_version();
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        // Must wait for the ASYNC driver to finish before counting commits.
        db.wait_for_migration(plan_id).unwrap();
        let after = db.storage.txn_manager().current_version();
        let chunks = 10u64.div_ceil(4);
        assert!(chunks > 1, "test must exercise multiple chunks");
        // Card-3 online flow with the ASYNC parallel driver, per-chunk commits
        // throughout (NOT one big batch):
        //   1 plan-create
        // + the shadow backfill: each of the N partitions commits per chunk +
        //   a `done` marker, so the backfill commit count is DEGREE-dependent
        //   (splitting only adds partial-chunk commits) — at LEAST `chunks`
        //   (the single-partition floor on a 1-CPU host).
        // + 1 phase-flip commit (Converting → CuttingOver)
        // + `chunks` single-threaded cutover commits (deterministic)
        // + 1 finalize commit (catalog kind flip + Completed)
        // Assert the incremental LOWER BOUND — proves chunked commits in both
        // passes regardless of the resolved parallel degree.
        let min_commits = 1 + chunks + 1 + chunks + 1;
        assert!(
            after - before >= min_commits,
            "expected per-chunk commits in both passes (got {}, want >= {min_commits}), not a single batch",
            after - before
        );
    }

    /// Acceptance: crash mid-migration, then resume from the persisted
    /// cursor — every row ends converted exactly once. Drives the catalog
    /// backfill worker + the cutover directly (the synchronous create path's
    /// seam) so we can inject a converter that fails partway, leaving a
    /// `Running`→`Failed` plan with a mid-range cursor, then re-drive with a
    /// good converter. Under the card-2 shadow model the backfill writes
    /// `<field>__shadow` siblings and LEAVES the source — reads return the
    /// SOURCE until the cutover promotes the shadows.
    #[test]
    fn create_field_type_migration_resumes_from_cursor_after_simulated_crash() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for i in 0..12i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 4, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();

        // "Crash": a converter that errors once it reaches the back half.
        let cutoff = ids[ids.len() / 2];
        let bad: crate::catalog::RegisteredConverter = Arc::new(move |oid: u64, v: &Value| {
            if oid >= cutoff {
                return Err(EngineError::Catalog(
                    crate::CatalogError::FieldTypeChangeConverterFailed {
                        qualified: "User.score".into(),
                        object_id: oid,
                        reason: "boom".into(),
                    },
                ));
            }
            match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                _ => unreachable!(),
            }
        });
        let err =
            crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &bad).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeConverterFailed { .. })
        ));
        let mid = db.list_migrations().unwrap();
        assert_eq!(mid[0].status, crate::catalog::MigrationStatus::Failed);
        assert!(
            mid[0].cursor > 0 && mid[0].cursor < *ids.last().unwrap(),
            "cursor should be mid-range, got {}",
            mid[0].cursor
        );
        let partial = mid[0].objects_converted;
        assert!(partial > 0 && partial < 12, "partial progress: {partial}");

        // Resume the backfill with a good converter from the persisted cursor.
        // Already-shadowed rows are idempotently skipped (the good converter
        // reads the SOURCE i64, never an f64 — the source is left intact).
        let good: crate::catalog::RegisteredConverter = Arc::new(|_oid: u64, v: &Value| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!("the converter reads the source; it never sees an f64"),
        });
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &good).unwrap();

        // Mid-migration (Converting done, NOT yet cut over): reads still return
        // the SOURCE value — the shadow is invisible and the kind is unchanged.
        match db.get("User", ids[0]).unwrap().fields.get("score") {
            Some(Value::I64(0)) => {}
            other => panic!("pre-cutover read must return source I64(0), got {other:?}"),
        }

        // Cut over: promote every shadow to the source field + flip the kind.
        db.run_terminal_pass(created.plan_id, created.type_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        drop(db);

        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for (id, i) in ids.iter().zip(0i64..) {
            match db2.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, i as f64),
                other => panic!("id {id}: expected F64, got {other:?}"),
            }
        }
    }

    /// A present `Value::Null` (kind UNSET) in the migrating field is skipped
    /// exactly like an absent field — it must NOT brick the migration.
    #[test]
    fn create_field_type_migration_skips_null_rows() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let normal = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(7));
            db.create("User", f).unwrap().id
        };
        // Overwrite a second row's blob with a present-Null score, bypassing
        // create() validation, to exercise the UNSET-skip arm.
        let null_id = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(9));
            db.create("User", f).unwrap().id
        };
        let tid = db.resolve_type_id("User").unwrap();
        let mut nf = FieldMap::new();
        nf.insert("score".into(), Value::Null);
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, null_id),
                    crate::object::serialize_fields(&nf),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();

        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        // The Null row converted nothing; only the normal row counts.
        assert_eq!(db.list_migrations().unwrap()[0].objects_converted, 1);
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert!(matches!(
            db2.get("User", normal).unwrap().fields.get("score"),
            Some(Value::F64(_))
        ));
        // The Null row is left Null (read-tolerant), not converted.
        assert!(matches!(
            db2.get("User", null_id).unwrap().fields.get("score"),
            Some(Value::Null) | None
        ));
    }

    /// G1 REGRESSION (card 2): a covered query on a SIBLING `@indexed` field
    /// must return the migrated field at its TARGET kind after cutover. The
    /// secondary-index covering payload embeds a full copy of the object incl.
    /// the (non-indexed) migrating field and carries NO `<field>__cover_v`
    /// generation stamp — so the cutover generation-bump alone can't invalidate
    /// it; cutover must REWRITE the covering payload. Without that fix this read
    /// serves the stale SOURCE value/kind. (Also a latent card-1 / offline bug.)
    #[test]
    fn cutover_refreshes_index_cover_on_sibling_indexed_field() {
        use rhypedb_schema::{FieldType, ScalarType};
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type T { x: i64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("T.x"));
        for i in 0..6i64 {
            let mut f = FieldMap::new();
            f.insert("x".into(), Value::I64(i * 10));
            f.insert("y".into(), Value::I64(i));
            db.create("T", f).unwrap();
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "T".into(),
                field_name: "x".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type T { x: f64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Covered read through the @indexed sibling `y` (filter_scan_via_index
        // materializes straight from the i: covering payload).
        let results = db2.filter_scan("T", "y", CompareOp::Ge, 0, None).unwrap();
        assert_eq!(results.len(), 6, "covered scan must see every row");
        for obj in &results {
            let y = match obj.fields.get("y") {
                Some(Value::I64(v)) => *v,
                other => panic!("y missing/wrong: {other:?}"),
            };
            match obj.fields.get("x") {
                Some(Value::F64(f)) => assert_eq!(*f, (y * 10) as f64, "stale x value"),
                other => panic!("covered read served stale/source x: {other:?}"),
            }
            assert!(
                obj.fields.keys().all(|k| !is_shadow_sibling_key(k)),
                "shadow leaked into covered read: {:?}",
                obj.fields.keys().collect::<Vec<_>>()
            );
        }
    }

    /// G2 REGRESSION (offline): the single-commit `change_field_type` must also
    /// refresh sibling `@indexed` covering payloads, or a covered query on the
    /// sibling returns the migrated field's stale source value/kind (latent
    /// since card 1). Mirrors `cutover_refreshes_index_cover_on_sibling_indexed_field`
    /// for the offline path.
    #[test]
    fn offline_field_type_change_refreshes_index_cover() {
        use rhypedb_schema::{FieldType, ScalarType};
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type T { x: i64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for i in 0..5i64 {
            let mut f = FieldMap::new();
            f.insert("x".into(), Value::I64(i * 10));
            f.insert("y".into(), Value::I64(i));
            db.create("T", f).unwrap();
        }
        db.change_field_type(
            "T",
            "x",
            FieldType::Scalar(ScalarType::F64),
            |_oid, v| match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                _ => unreachable!(),
            },
        )
        .unwrap();
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type T { x: f64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let results = db2.filter_scan("T", "y", CompareOp::Ge, 0, None).unwrap();
        assert_eq!(results.len(), 5);
        for obj in &results {
            let y = match obj.fields.get("y") {
                Some(Value::I64(v)) => *v,
                other => panic!("y: {other:?}"),
            };
            match obj.fields.get("x") {
                Some(Value::F64(f)) => assert_eq!(*f, (y * 10) as f64, "stale x via offline change"),
                other => panic!("offline covered read served stale/source x: {other:?}"),
            }
        }
    }

    /// Cutover refuses (and parks `Failed`) an object whose source is still at
    /// the source kind with NO shadow — the converter never reached it.
    #[test]
    fn cutover_refuses_missing_shadow() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for i in 0..3i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        // Cut over WITHOUT backfilling any shadows → first row refuses.
        let err = db.run_terminal_pass(created.plan_id, created.type_id).unwrap_err();
        assert!(
            matches!(err, EngineError::MigrationCutoverShadowMissing { .. }),
            "got {err:?}"
        );
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// Cutover refuses (and parks `Failed`) an object whose `<field>__shadow_cv`
    /// stamp doesn't match the plan's pinned converter version.
    #[test]
    fn cutover_refuses_stale_shadow_cv() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let id = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(7));
            db.create("User", f).unwrap().id
        };
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        // Plan pins converter_version = 2.
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 2, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        // Craft a blob with a shadow stamped at the WRONG converter version (1).
        let tid = db.resolve_type_id("User").unwrap();
        let mut nf = FieldMap::new();
        nf.insert("score".into(), Value::I64(7));
        nf.insert("score__shadow".into(), Value::F64(7.0));
        nf.insert("score__shadow_cv".into(), Value::U32(1));
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, id),
                    crate::object::serialize_fields(&nf),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();
        let err = db.run_terminal_pass(created.plan_id, created.type_id).unwrap_err();
        assert!(
            matches!(
                err,
                EngineError::MigrationCutoverShadowStale {
                    found_cv: 1,
                    want_cv: 2,
                    ..
                }
            ),
            "got {err:?}"
        );
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// Card 3 worker: N partitions over `[1, U)` back-fill the shadow for every
    /// object exactly once, each advancing its own `c:S:` cursor; re-running is
    /// idempotent (done partitions return immediately, convert nothing more).
    #[test]
    fn parallel_partition_backfill_covers_range_idempotently() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::atomic::{AtomicU8, Ordering};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();

        const N: i64 = 250;
        for i in 0..N {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let i64_kind = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::I64));
        let target = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let u = db.next_object_id.load(Ordering::SeqCst); // exclusive id upper bound
        let degree = 4u8;
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, Some(degree), u, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let converter: crate::catalog::RegisteredConverter =
            std::sync::Arc::new(|_oid: u64, v: &Value| match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                other => Err(EngineError::Catalog(
                    crate::CatalogError::FieldTypeChangeConverterFailed {
                        qualified: "User.score".into(),
                        object_id: 0,
                        reason: format!("unexpected {other:?}"),
                    },
                )),
            });
        let control = AtomicU8::new(crate::catalog::migration_control::RUN);
        let errs = std::sync::atomic::AtomicU64::new(0);

        let run_all = || {
            for idx in 0..degree {
                let (lo, hi) = crate::catalog::partition_range(degree, u, idx);
                let outcome = crate::catalog::run_migration_partition(
                    &db.storage, created.plan_id, created.type_id, idx, lo, hi,
                    "User", "score", i64_kind, target, 1, 16, &converter, &control,
                    crate::catalog::ErrorPolicy::Stop, false, &errs, 0, "widen", None,
                )
                .unwrap();
                assert_eq!(outcome, crate::catalog::PartitionDriveOutcome::Done);
            }
        };

        run_all();

        // Every object carries a converted shadow (score i64 → shadow f64, cv 1).
        let tid = created.type_id;
        let snap = db.storage.read_snapshot();
        let mut shadows = 0usize;
        for id in 1..u {
            let key = rhypedb_storage::key::KeyBuilder::object(tid, id);
            let blob = db.storage.get_at(snap, &key).unwrap().expect("object missing");
            let fields = crate::object::deserialize_fields(&blob);
            match (
                fields.get("score"),
                fields.get("score__shadow"),
                fields.get("score__shadow_cv"),
            ) {
                (Some(Value::I64(s)), Some(Value::F64(sh)), Some(Value::U32(1))) => {
                    assert_eq!(*sh, *s as f64);
                    shadows += 1;
                }
                other => panic!("object {id} missing/bad shadow: {other:?}"),
            }
        }
        assert_eq!(shadows, N as usize, "every object backfilled exactly once");

        // Sum of per-partition converted counts == N (counted once), all done.
        let mut total = 0u64;
        for idx in 0..degree {
            let key = rhypedb_storage::key::KeyBuilder::catalog_partition_cursor(created.plan_id, idx);
            let txn = db.storage.begin_txn();
            let bytes = db.storage.get(&txn, &key).unwrap().expect("partition cursor missing");
            let pc = crate::catalog::decode_partition_cursor("c:S", &bytes).unwrap();
            assert!(pc.done, "partition {idx} not done");
            total += pc.objects_converted;
        }
        assert_eq!(total, N as u64, "objects_converted summed over partitions");

        // Idempotent re-run: done partitions return Done, convert nothing more.
        run_all();
        let mut total2 = 0u64;
        for idx in 0..degree {
            let key = rhypedb_storage::key::KeyBuilder::catalog_partition_cursor(created.plan_id, idx);
            let txn = db.storage.begin_txn();
            let pc = crate::catalog::decode_partition_cursor(
                "c:S",
                &db.storage.get(&txn, &key).unwrap().unwrap(),
            )
            .unwrap();
            total2 += pc.objects_converted;
        }
        assert_eq!(total2, N as u64, "re-run must not double-count");
    }

    /// AC4: a `parallel_degree == 1` backfill produces the SAME converted object
    /// content as the legacy single-worker `run_migration_chunks` — both go
    /// through the one shared `convert_row_for_backfill`, and a single partition
    /// covers the identical `[1, U)` range. (Compared as deserialized field maps,
    /// since the on-disk byte order of a `FieldMap` is hash-nondeterministic and
    /// not part of the contract.)
    #[test]
    fn parallel_degree_1_backfill_matches_single_worker_byte_for_byte() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::atomic::{AtomicU8, Ordering};
        use std::sync::Arc;
        let i64k = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::I64));
        let f64k = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let converter: crate::catalog::RegisteredConverter = Arc::new(|_o: u64, v: &Value| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        let seed = |db: &Database| -> Vec<u64> {
            let mut ids = Vec::new();
            for i in 0..20i64 {
                let mut f = FieldMap::new();
                f.insert("score".into(), Value::I64(i * 3 - 7));
                ids.push(db.create("User", f).unwrap().id);
            }
            ids
        };

        // A: degree-1 parallel backfill.
        let dir_a = tempfile::tempdir().unwrap();
        let db_a =
            Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir_a.path())
                .unwrap();
        let ids_a = seed(&db_a);
        let u_a = db_a.next_object_id.load(Ordering::SeqCst);
        let created_a = crate::catalog::create_migration_plan(
            &db_a.storage,
            &db_a.schema,
            "User",
            "score",
            f64k,
            "widen",
            1,
            4,
            Some(1),
            u_a, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let ctrl = AtomicU8::new(crate::catalog::migration_control::RUN);
        let disp = crate::catalog::run_parallel_backfill(
            &db_a.storage,
            created_a.plan_id,
            created_a.type_id,
            1,
            u_a,
            "User",
            "score",
            i64k,
            f64k,
            1,
            4,
            &converter,
            &ctrl,
            crate::catalog::ErrorPolicy::Stop,
            false,
            0,
            "widen",
            None,
        )
        .unwrap();
        assert_eq!(disp, crate::catalog::BackfillDisposition::AllDone);

        // B: legacy single-worker backfill.
        let dir_b = tempfile::tempdir().unwrap();
        let db_b =
            Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir_b.path())
                .unwrap();
        let ids_b = seed(&db_b);
        let created_b = crate::catalog::create_migration_plan(
            &db_b.storage,
            &db_b.schema,
            "User",
            "score",
            f64k,
            "widen",
            1,
            4,
            None,
            0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        crate::catalog::run_migration_chunks(&db_b.storage, created_b.plan_id, &converter).unwrap();

        // Identical ids (fresh DBs, same create order) → compare converted content.
        assert_eq!(ids_a, ids_b);
        assert_eq!(created_a.type_id, created_b.type_id);
        let snap_a = db_a.storage.read_snapshot();
        let snap_b = db_b.storage.read_snapshot();
        for &id in &ids_a {
            let key = rhypedb_storage::key::KeyBuilder::object(created_a.type_id, id);
            let blob_a = db_a.storage.get_at(snap_a, &key).unwrap().unwrap();
            let blob_b = db_b.storage.get_at(snap_b, &key).unwrap().unwrap();
            let fields_a = crate::object::deserialize_fields(&blob_a);
            let fields_b = crate::object::deserialize_fields(&blob_b);
            assert_eq!(
                fields_a, fields_b,
                "object {id} converted content differs (degree-1 vs single-worker)"
            );
        }
    }

    /// Card 3 BLOCKER regression: a CUTOVER refusal on a parallel (N>1) plan must
    /// reset the per-partition `c:S:` cursors (via `park_migration_failed_rewind`),
    /// else the re-driven backfill fast-returns `Done` for every partition and
    /// cutover re-refuses the same missing shadow FOREVER. Backfill, delete one
    /// row's shadow to force `MigrationCutoverShadowMissing`, confirm the rewind
    /// cleared the cursors, then re-backfill + cutover and assert it COMPLETES.
    #[test]
    fn parallel_cutover_refusal_resets_cursors_and_completes() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::atomic::{AtomicU8, Ordering};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let mut ids = Vec::new();
        for i in 0..30i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let i64k = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::I64));
        let f64k = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let converter: crate::catalog::RegisteredConverter = Arc::new(|_o: u64, v: &Value| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        let degree = 4u8;
        let u = db.next_object_id.load(Ordering::SeqCst);
        let created = crate::catalog::create_migration_plan(
            &db.storage,
            &db.schema,
            "User",
            "score",
            f64k,
            "widen",
            1,
            4,
            Some(degree),
            u, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let plan_id = created.plan_id;
        let tid = created.type_id;
        let ctrl = AtomicU8::new(crate::catalog::migration_control::RUN);
        let backfill = || {
            crate::catalog::run_parallel_backfill(
                &db.storage, plan_id, tid, degree, u, "User", "score", i64k, f64k, 1, 4, &converter,
                &ctrl, crate::catalog::ErrorPolicy::Stop, false, 0, "widen", None,
            )
        };
        assert_eq!(
            backfill().unwrap(),
            crate::catalog::BackfillDisposition::AllDone
        );

        // Corrupt: strip the shadow from one row so cutover refuses it.
        let victim = ids[ids.len() / 2];
        {
            let key = rhypedb_storage::key::KeyBuilder::object(tid, victim);
            let snap = db.storage.read_snapshot();
            let blob = db.storage.get_at(snap, &key).unwrap().unwrap();
            let mut fields = crate::object::deserialize_fields(&blob);
            fields.remove("score__shadow");
            fields.remove("score__shadow_cv");
            let mut txn = db.storage.begin_txn();
            db.storage
                .put(&mut txn, &key, crate::object::serialize_fields(&fields))
                .unwrap();
            db.storage.commit(&mut txn).unwrap();
        }

        // Cutover refuses the missing shadow → park_migration_failed_rewind.
        let err = db.run_terminal_pass(plan_id, tid).unwrap_err();
        assert!(matches!(
            err,
            EngineError::MigrationCutoverShadowMissing { .. }
        ));

        // BLOCKER fix: the rewind deleted every c:S: cursor.
        for idx in 0..degree {
            let key = rhypedb_storage::key::KeyBuilder::catalog_partition_cursor(plan_id, idx);
            let txn = db.storage.begin_txn();
            assert!(
                db.storage.get(&txn, &key).unwrap().is_none(),
                "c:S cursor {idx} must be deleted by the rewind"
            );
        }

        // Re-backfill (re-stamps the deleted shadow) then cutover — must COMPLETE
        // (not loop). The full-range re-scan re-converts idempotently.
        assert_eq!(
            backfill().unwrap(),
            crate::catalog::BackfillDisposition::AllDone
        );
        db.run_terminal_pass(plan_id, tid).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
    }

    /// Register a `widen` converter that BLOCKS every call until `release()` is
    /// invoked, signalling (once) on a channel when the FIRST worker reaches it.
    /// This makes a pause/reopen test DETERMINISTIC instead of sleep-timed: the
    /// test waits for the start signal (the backfill is provably mid-flight, with
    /// NO chunk committed yet because the first row of the first chunk is
    /// blocked), sets the control byte, then releases — so every worker stops
    /// after exactly its first chunk, well before completion, on every host.
    fn register_gated_widen(
        db: &Database,
    ) -> (std::sync::mpsc::Receiver<()>, impl Fn() + use<>) {
        use std::sync::{Arc, Condvar, Mutex};
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let tx = Mutex::new(Some(tx));
        let rel = Arc::clone(&release);
        db.register_converter("widen", 1, move |_oid, v| {
            if let Some(t) = tx.lock().unwrap().take() {
                let _ = t.send(());
            }
            let (m, cv) = &*rel;
            let mut g = m.lock().unwrap();
            while !*g {
                g = cv.wait(g).unwrap();
            }
            drop(g);
            match v {
                Value::I64(i) => Ok(Value::F64(*i as f64)),
                _ => unreachable!(),
            }
        });
        let rel2 = Arc::clone(&release);
        let release_fn = move || {
            let (m, cv) = &*rel2;
            *m.lock().unwrap() = true;
            cv.notify_all();
        };
        (rx, release_fn)
    }

    /// AC6: a pause request stops the parallel backfill BEFORE completion (every
    /// worker polls the control byte between chunks), leaving the plan resumable;
    /// resume then completes it. Deterministic via a gated converter: the pause is
    /// set while a worker is provably blocked on the first row of its first chunk
    /// (no chunk committed yet), so every worker stops after exactly one chunk.
    #[test]
    fn pause_migration_stops_before_completion_then_resumes() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (started, release) = register_gated_widen(&db);
        let mut ids = Vec::new();
        for i in 0..150i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        // A worker has reached the (blocked) converter → backfill is mid-flight,
        // nothing committed. Pause now, THEN release: every worker stops after its
        // first chunk.
        started.recv().unwrap();
        db.pause_migration(plan_id).unwrap();
        release();
        db.wait_for_migration(plan_id).unwrap();

        let st = db.list_migrations().unwrap()[0].status;
        assert!(
            st.quiesces() && st != crate::catalog::MigrationStatus::Completed,
            "expected a paused/resumable plan, got {st:?}"
        );
        // Strictly fewer than all rows converted (proves it stopped early).
        let tid = db.resolve_type_id("User").unwrap();
        let snap = db.storage.read_snapshot();
        let converted = ids
            .iter()
            .filter(|&&id| {
                let blob = db
                    .storage
                    .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, id))
                    .unwrap()
                    .unwrap();
                crate::object::deserialize_fields(&blob).contains_key("score__shadow")
            })
            .count();
        assert!(
            converted < ids.len(),
            "pause must stop before converting all rows (converted {converted}/{})",
            ids.len()
        );
        drop(db);

        // Resume requires reopening with the TARGET schema (the F3 guard refuses
        // to drive a plan whose handle still validates against the source kind).
        let db2 = Database::open(parse_schema(r#"type User { score: f64 }"#).unwrap(), dir.path())
            .unwrap();
        db2.register_converter("widen", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        db2.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db2.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        for (id, i) in ids.iter().zip(0i64..) {
            assert_eq!(
                db2.get("User", *id).unwrap().fields.get("score"),
                Some(&Value::F64(i as f64)),
                "row {id} must be converted exactly once"
            );
        }
    }

    /// Card 5 (5c): a TERMINAL cancel rolls the migration back — strips every
    /// partial `<field>__shadow` sibling, leaves the source value intact, settles
    /// `Cancelled`, and disarms the hook (migrating_field_count → 0). Deterministic
    /// via the gated converter: cancel is issued while a worker is provably
    /// mid-backfill, so some shadows exist to strip.
    #[test]
    fn card5_cancel_rolls_back_strips_shadows_and_disarms() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (started, release) = register_gated_widen(&db);
        let mut ids = Vec::new();
        for i in 0..120i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        // Backfill is provably mid-flight (a worker blocked in the converter).
        started.recv().unwrap();
        // An active driver owns the plan → cancel marks RollingBack + CANCEL and
        // returns; the driver completes the rollback on winddown.
        db.cancel_migration(plan_id).unwrap();
        release();
        db.wait_for_migration(plan_id).unwrap();

        // Settled Cancelled, the catalog kind NOT flipped (still i64).
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Cancelled
        );
        // Hook disarmed.
        assert_eq!(
            db.migrating_field_count
                .load(std::sync::atomic::Ordering::Acquire),
            0,
            "cancel must disarm the double-write hook"
        );
        // NO shadow remains on any object; the source value is intact (I64).
        let tid = db.resolve_type_id("User").unwrap();
        let snap = db.storage.read_snapshot();
        for (id, i) in ids.iter().zip(0i64..) {
            let blob = db
                .storage
                .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, *id))
                .unwrap()
                .unwrap();
            let fields = crate::object::deserialize_fields(&blob);
            assert!(
                !fields.contains_key("score__shadow")
                    && !fields.contains_key("score__shadow_cv"),
                "row {id} still carries a shadow after rollback"
            );
            assert_eq!(fields.get("score"), Some(&Value::I64(i)), "source row {id} mutated");
        }
        // Reads return the source value (no kind flip, handle still valid).
        assert_eq!(
            db.get("User", ids[0]).unwrap().fields.get("score"),
            Some(&Value::I64(0))
        );
        // A subsequent cancel is idempotent.
        db.cancel_migration(plan_id).unwrap();
    }

    /// Card 5 (5c): a terminal cancel that CRASHES mid-rollback resumes on the
    /// next open. The durable `RollingBack` phase survives; auto-resume re-drives
    /// the strip from a fresh open with the SOURCE schema (no converter), settling
    /// `Cancelled` and stripping every shadow. Deterministic: a paused plan with
    /// partial shadows is durably flipped to RollingBack, the handle dropped, then
    /// reopened.
    #[test]
    fn card5_cancel_rollback_resumes_after_reopen() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (started, release) = register_gated_widen(&db);
        let mut ids = Vec::new();
        for i in 0..120i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        // Pause mid-flight so SOME shadows are durably written, then quiesce.
        started.recv().unwrap();
        db.pause_migration(plan_id).unwrap();
        release();
        db.wait_for_migration(plan_id).unwrap();
        // Simulate a cancel whose rollback crashed before completing AND whose
        // backfill had errored: park Failed, then durably mark RollingBack WITHOUT
        // driving it. A RollingBack plan must complete its rollback on reopen EVEN
        // when Failed (Failed is not normally drivable; the rollback is gated in
        // separately).
        crate::catalog::park_migration_failed_keep_cursors(&db.storage, plan_id).unwrap();
        crate::catalog::set_plan_phase_rolling_back(&db.storage, plan_id).unwrap();
        drop(db);

        // Reopen with the SOURCE schema (the operator abandoned the migration).
        // auto-resume must complete the rollback even with no converter registered.
        let db2 = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        assert_eq!(
            db2.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Cancelled,
            "auto-resume must complete the rollback (even from Failed)"
        );
        assert_eq!(
            db2.migrating_field_count
                .load(std::sync::atomic::Ordering::Acquire),
            0,
            "rollback must disarm the hook"
        );
        let tid = db2.resolve_type_id("User").unwrap();
        let snap = db2.storage.read_snapshot();
        for (id, i) in ids.iter().zip(0i64..) {
            let blob = db2
                .storage
                .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, *id))
                .unwrap()
                .unwrap();
            let fields = crate::object::deserialize_fields(&blob);
            assert!(
                !fields.contains_key("score__shadow")
                    && !fields.contains_key("score__shadow_cv"),
                "row {id} still carries a shadow after resumed rollback"
            );
            assert_eq!(fields.get("score"), Some(&Value::I64(i)));
        }
    }

    /// Card 5 (5c, impl-review regression): a cancel that RACES a Stop-policy
    /// backfill error still rolls back. The worker's converter fails (parking the
    /// plan Failed + returning Err) AFTER cancel durably set RollingBack — the
    /// driver must complete the rollback (→ Cancelled, shadows stripped, hook
    /// disarmed) and NOT surface the backfill error or wedge Failed+RollingBack.
    #[test]
    fn card5_cancel_then_backfill_error_still_rolls_back() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::{Arc, Condvar, Mutex};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        // A gated converter that, once released, FAILS (Stop policy → park Failed).
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let (tx, started) = std::sync::mpsc::channel::<()>();
        let tx = Mutex::new(Some(tx));
        let rel = Arc::clone(&release);
        db.register_converter("widen", 1, move |oid, _v| {
            if let Some(t) = tx.lock().unwrap().take() {
                let _ = t.send(());
            }
            let (m, cv) = &*rel;
            let mut g = m.lock().unwrap();
            while !*g {
                g = cv.wait(g).unwrap();
            }
            drop(g);
            Err(EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterFailed {
                    qualified: "User.score".into(),
                    object_id: oid,
                    reason: "boom".into(),
                },
            ))
        });
        let mut ids = Vec::new();
        for i in 0..40i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        // Worker is blocked in the converter; cancel durably marks RollingBack,
        // THEN release so the converter fails (parks Failed + returns Err).
        started.recv().unwrap();
        db.cancel_migration(plan_id).unwrap();
        {
            let (m, cv) = &*release;
            *m.lock().unwrap() = true;
            cv.notify_all();
        }
        // Must NOT surface the backfill error — the rollback supersedes it.
        db.wait_for_migration(plan_id).unwrap();

        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Cancelled,
            "cancel must win over the Stop-policy backfill error"
        );
        assert_eq!(
            db.migrating_field_count
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        let tid = db.resolve_type_id("User").unwrap();
        let snap = db.storage.read_snapshot();
        for (id, i) in ids.iter().zip(0i64..) {
            let blob = db
                .storage
                .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, *id))
                .unwrap()
                .unwrap();
            let fields = crate::object::deserialize_fields(&blob);
            assert!(
                !fields.contains_key("score__shadow")
                    && !fields.contains_key("score__shadow_cv")
            );
            assert_eq!(fields.get("score"), Some(&Value::I64(i)));
        }
    }

    /// Card 5 (5c) AC `cancel_does_not_run_cutover`: an explicit cutover of a
    /// cancelled plan is refused with `MigrationCancelledCannotCutover`.
    #[test]
    fn card5_cancel_does_not_run_cutover() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (started, release) = register_gated_widen(&db);
        for i in 0..40i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        started.recv().unwrap();
        db.cancel_migration(plan_id).unwrap();
        release();
        db.wait_for_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Cancelled
        );
        assert!(matches!(
            db.cutover_migration(plan_id),
            Err(EngineError::MigrationCancelledCannotCutover { plan_id: p }) if p == plan_id
        ));
    }

    /// Card 5 (5c) AC `start_pause_resume_cancel_progress_via_public_api`: drive
    /// the full operator surface through the PUBLIC verbs. Plan A exercises
    /// start_async → progress → pause → resume → complete; plan B exercises
    /// cancel. Deterministic via the gated converter.
    #[test]
    fn card5_start_pause_resume_cancel_progress_via_public_api() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (started, release) = register_gated_widen(&db);
        let mut ids = Vec::new();
        for i in 0..150i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let handle = db
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        started.recv().unwrap();
        // progress while Running.
        let p = db.query_migration_progress(handle.plan_id).unwrap();
        assert_eq!(p.status, crate::catalog::MigrationStatus::Running);
        assert_eq!(p.total_objects, 150);
        // pause → release → wait (resumable, partial).
        db.pause_migration(handle.plan_id).unwrap();
        release();
        db.wait_for_migration(handle.plan_id).unwrap();
        let st = db.list_migrations().unwrap()[0].status;
        assert!(st.quiesces() && st != crate::catalog::MigrationStatus::Completed);
        drop(db);

        // resume to completion (reopen with the TARGET schema + a plain converter).
        let db2 = Database::open(parse_schema(r#"type User { score: f64 }"#).unwrap(), dir.path())
            .unwrap();
        db2.register_converter("widen", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        db2.resume_field_type_migration(handle.plan_id).unwrap();
        let done = db2.query_migration_progress(handle.plan_id).unwrap();
        assert_eq!(done.status, crate::catalog::MigrationStatus::Completed);
        assert!(done.eta_unix_ms.is_none(), "completed plan has no ETA");
        drop(db2);

        // Plan B: cancel via the public verb (fresh db, source schema).
        let dir_b = tempfile::tempdir().unwrap();
        let db_b =
            Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir_b.path())
                .unwrap();
        let (started_b, release_b) = register_gated_widen(&db_b);
        for i in 0..40i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db_b.create("User", f).unwrap();
        }
        let hb = db_b
            .start_field_type_migration_async(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
                ..Default::default()
            })
            .unwrap();
        started_b.recv().unwrap();
        db_b.cancel_migration(hb.plan_id).unwrap();
        release_b();
        db_b.wait_for_migration(hb.plan_id).unwrap();
        assert_eq!(
            db_b.query_migration_progress(hb.plan_id).unwrap().status,
            crate::catalog::MigrationStatus::Cancelled
        );
    }

    /// The double-driver gate (card 3/5): at most one ACTIVE driver per plan id
    /// (a second registration is refused `MigrationAlreadyRunning`), but a
    /// FINISHED leftover is reaped so a later resume can register cleanly.
    #[test]
    fn migration_driver_gate_rejects_second_and_reaps_finished() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        let (_ctrl, sig) = db.register_inline_driver(7).unwrap();
        // A second registration while the first is active is refused.
        assert!(matches!(
            db.register_inline_driver(7),
            Err(EngineError::MigrationAlreadyRunning { plan_id: 7 })
        ));
        // Once the first signals finished, the gate reaps it and admits a new one.
        sig.mark_done(None);
        assert!(db.register_inline_driver(7).is_ok());
    }

    /// AC2: a parallel plan paused mid-backfill resumes PER-PARTITION across a
    /// REOPEN (simulated restart) — each partition continues from its persisted
    /// `c:S:` cursor — and every row ends converted exactly once.
    #[test]
    fn parallel_migration_resumes_per_partition_across_reopen() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let plan_id;
        let ids: Vec<u64>;
        {
            let db =
                Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
                    .unwrap();
            // Gated converter so the pause lands mid-backfill deterministically
            // (a partial set of partitions converted → resume must finish them).
            let (started, release) = register_gated_widen(&db);
            let mut v = Vec::new();
            for i in 0..150i64 {
                let mut f = FieldMap::new();
                f.insert("score".into(), Value::I64(i));
                v.push(db.create("User", f).unwrap().id);
            }
            ids = v;
            plan_id = db
                .create_field_type_migration(MigrationPlanSpec {
                    type_name: "User".into(),
                    field_name: "score".into(),
                    target_field_type: FieldType::Scalar(ScalarType::F64),
                    converter_name: "widen".into(),
                    converter_version: 1,
                    chunk_size: 4, ..Default::default()
                })
                .unwrap();
            started.recv().unwrap();
            db.pause_migration(plan_id).unwrap();
            release();
            db.wait_for_migration(plan_id).unwrap();
            assert_ne!(
                db.list_migrations().unwrap()[0].status,
                crate::catalog::MigrationStatus::Completed
            );
            drop(db);
        }
        // Reopen with the TARGET schema (simulated restart); register the
        // converter, then resume — each partition continues from its c:S: cursor.
        let db = Database::open(parse_schema(r#"type User { score: f64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        db.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        for (id, i) in ids.iter().zip(0i64..) {
            assert_eq!(
                db.get("User", *id).unwrap().fields.get("score"),
                Some(&Value::F64(i as f64)),
                "row {id} must be converted exactly once across the reopen"
            );
        }
    }

    // =====================================================================
    // Card 4/5: ErrorPolicy + quarantine sidecar + dry-run
    // =====================================================================

    /// A widen converter that ERRORS on a sentinel value (`poison`) and widens
    /// every other `I64` to `F64`. Models a row the operator's converter can't
    /// handle (e.g. a forgotten edge case).
    fn widen_or_fail_on(poison: i64) -> impl Fn(u64, &Value) -> EngineResult<Value> + Send + Sync {
        move |oid, v| match v {
            Value::I64(i) if *i == poison => Err(EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterFailed {
                    qualified: "User.score".into(),
                    object_id: oid,
                    reason: format!("poison value {poison}"),
                },
            )),
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            other => Err(EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterFailed {
                    qualified: "User.score".into(),
                    object_id: oid,
                    reason: format!("unexpected {}", other.type_name()),
                },
            )),
        }
    }

    /// Seed `n` User rows with `score = i`, planting `score = -1` (the poison
    /// value) at `poison_ids` positions. Returns the created object ids.
    fn seed_scores(db: &Database, n: i64, poison_idxs: &[i64]) -> Vec<u64> {
        let mut ids = Vec::new();
        for i in 0..n {
            let mut f = FieldMap::new();
            let v = if poison_idxs.contains(&i) { -1 } else { i };
            f.insert("score".into(), Value::I64(v));
            ids.push(db.create("User", f).unwrap().id);
        }
        ids
    }

    fn card4_spec(error_policy: crate::catalog::ErrorPolicy, dry_run: bool) -> MigrationPlanSpec {
        use rhypedb_schema::{FieldType, ScalarType};
        MigrationPlanSpec {
            type_name: "User".into(),
            field_name: "score".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 4,
            error_policy,
            dry_run,
            ..Default::default()
        }
    }

    /// AC: `Stop` (the default) halts the whole migration on the first converter
    /// failure — parked `Failed`, no cutover.
    #[test]
    fn error_policy_stop_halts_all_partitions_on_first_failure() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        seed_scores(&db, 30, &[15]); // one poison row
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Stop, false))
            .unwrap();
        let err = db.wait_for_migration(plan_id).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeConverterFailed { .. })
        ));
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// AC: `SkipAndLog` counts the failure, leaves the row source-shape, and the
    /// migration continues + cuts over (the errored rows stay source-shape).
    #[test]
    fn error_policy_skip_and_log_continues_and_increments_error_count() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        let ids = seed_scores(&db, 30, &[5, 15, 25]); // 3 poison rows
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::SkipAndLog, false))
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        let summary = &db.list_migrations().unwrap()[0];
        assert_eq!(summary.status, crate::catalog::MigrationStatus::Completed);
        assert_eq!(summary.error_count, 3);
        assert_eq!(summary.objects_converted, 27);
        // A non-poison row is converted (f64); a poison row stays source-shape.
        drop(db);
        let db2 = Database::open(parse_schema(r#"type User { score: f64 }"#).unwrap(), dir.path())
            .unwrap();
        assert_eq!(
            db2.get("User", ids[0]).unwrap().fields.get("score"),
            Some(&Value::F64(0.0))
        );
        // The poison row (id at index 5) was LEFT source-shape (still I64(-1)).
        assert_eq!(
            db2.get("User", ids[5]).unwrap().fields.get("score"),
            Some(&Value::I64(-1))
        );
    }

    /// AC: `Quarantine` writes a `c:Q:` sidecar per failed row + continues the
    /// backfill (the good rows are converted).
    #[test]
    fn error_policy_quarantine_writes_sidecar_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        let ids = seed_scores(&db, 30, &[5, 15, 25]);
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Quarantine, false))
            .unwrap();
        // The backfill continues + quarantines; the driver then refuses cutover
        // (unresolved quarantine) — but the SIDECARS prove the backfill continued.
        let _ = db.wait_for_migration(plan_id);
        let q = db.list_quarantined(plan_id).unwrap();
        assert_eq!(q.len(), 3, "one sidecar per failed row");
        assert!(q.iter().all(|e| e.error_msg.contains("poison")));
        assert_eq!(db.list_migrations().unwrap()[0].error_count, 3);
        // A good row was converted (shadow present) despite the errors.
        let tid = db.resolve_type_id("User").unwrap();
        let snap = db.storage.read_snapshot();
        let blob = db
            .storage
            .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, ids[0]))
            .unwrap()
            .unwrap();
        assert!(crate::object::deserialize_fields(&blob).contains_key("score__shadow"));
    }

    /// AC: cutover refuses a `Quarantine` plan with unresolved quarantine rows
    /// (`MigrationCutoverHasErrors`), parking `Failed`.
    #[test]
    fn cutover_refuses_with_non_skip_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        seed_scores(&db, 30, &[5, 15, 25]);
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Quarantine, false))
            .unwrap();
        let err = db.wait_for_migration(plan_id).unwrap_err();
        assert!(
            matches!(
                err,
                EngineError::MigrationCutoverHasErrors { error_count: 3, .. }
            ),
            "got {err:?}"
        );
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// AC: a `dry_run` invokes the converter on every row but writes NO `o:` /
    /// `c:Q:` / lingering `c:S:` rows, and never flips the catalog kind.
    #[test]
    fn dry_run_invokes_converter_without_storage_writes() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let ids = seed_scores(&db, 20, &[]);
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Stop, true))
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        let summary = &db.list_migrations().unwrap()[0];
        assert_eq!(summary.status, crate::catalog::MigrationStatus::DryRunCompleted);
        assert!(summary.dry_run);
        assert_eq!(summary.objects_converted, 20, "counted, not written");
        // NO shadow on any object.
        let tid = db.resolve_type_id("User").unwrap();
        let snap = db.storage.read_snapshot();
        for id in &ids {
            let blob = db
                .storage
                .get_at(snap, &rhypedb_storage::key::KeyBuilder::object(tid, *id))
                .unwrap()
                .unwrap();
            let f = crate::object::deserialize_fields(&blob);
            assert!(!f.contains_key("score__shadow"), "dry-run wrote a shadow");
            assert!(matches!(f.get("score"), Some(Value::I64(_))), "source untouched");
        }
        // c:S: cursors were cleaned up.
        let cs = db
            .storage
            .scan_prefix_at(
                snap,
                &rhypedb_storage::key::KeyBuilder::catalog_partition_cursor_plan_prefix(plan_id),
            )
            .unwrap();
        assert!(cs.is_empty(), "dry-run left c:S: cursors");
        // Catalog kind NOT flipped — reopening with the SOURCE schema still works.
        drop(db);
        let db2 = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        assert_eq!(
            db2.get("User", ids[0]).unwrap().fields.get("score"),
            Some(&Value::I64(0))
        );
    }

    /// AC: a dry-run's `error_count` matches a real run over the same data.
    #[test]
    fn dry_run_error_count_matches_actual_run() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        seed_scores(&db, 30, &[5, 15, 25]); // 3 poison rows
        // Dry run (preflight) — SkipAndLog so the converter failures are counted.
        let dry = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::SkipAndLog, true))
            .unwrap();
        db.wait_for_migration(dry).unwrap();
        let dry_errors = db.list_migrations().unwrap()[0].error_count;
        assert_eq!(dry_errors, 3);
        // A real run over the SAME data (the settled dry-run plan doesn't block it).
        let real = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::SkipAndLog, false))
            .unwrap();
        db.wait_for_migration(real).unwrap();
        let real_errors = db
            .list_migrations()
            .unwrap()
            .into_iter()
            .find(|m| m.plan_id == real)
            .unwrap()
            .error_count;
        assert_eq!(real_errors, dry_errors, "preflight estimate must be accurate");
    }

    /// A dry-run paused/crashed mid-flight must NOT brick writes to the field on
    /// reopen — it never arms the double-write hook, and it remains resumable on
    /// the SOURCE-schema handle (no F3 target-schema guard). (impl-review BLOCKER)
    #[test]
    fn paused_dry_run_does_not_brick_writes_and_resumes() {
        use std::sync::atomic::Ordering;
        let dir = tempfile::tempdir().unwrap();
        let plan_id;
        {
            let db =
                Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
                    .unwrap();
            let (started, release) = register_gated_widen(&db);
            seed_scores(&db, 120, &[]);
            plan_id = db
                .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Stop, true))
                .unwrap();
            // A dry-run never arms the hook — even mid-flight.
            assert_eq!(db.migrating_field_count.load(Ordering::SeqCst), 0);
            started.recv().unwrap();
            db.pause_migration(plan_id).unwrap();
            release();
            db.wait_for_migration(plan_id).unwrap();
            assert_ne!(
                db.list_migrations().unwrap()[0].status,
                crate::catalog::MigrationStatus::DryRunCompleted
            );
            drop(db);
        }
        // Reopen on the SOURCE schema (a dry-run never flipped the kind).
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        // auto_resume must NOT have armed the hook for the dry-run.
        assert_eq!(db.migrating_field_count.load(Ordering::SeqCst), 0);
        // A write to the migrating field SUCCEEDS (not failed-closed).
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::I64(999));
        db.create("User", f).unwrap();
        // Resume completes the preflight on the source-schema handle (no F3 guard).
        db.register_converter("widen", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        db.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations()
                .unwrap()
                .into_iter()
                .find(|m| m.plan_id == plan_id)
                .unwrap()
                .status,
            crate::catalog::MigrationStatus::DryRunCompleted
        );
    }

    /// AC: exceeding the quarantine cap auto-STOPS the migration (parks `Failed`).
    #[test]
    fn quarantine_cap_auto_stops_migration() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        // A converter that fails on EVERY row → a runaway error field.
        db.register_converter("widen", 1, |oid, _v| -> EngineResult<Value> {
            Err(EngineError::Catalog(
                crate::CatalogError::FieldTypeChangeConverterFailed {
                    qualified: "User.score".into(),
                    object_id: oid,
                    reason: "always fails".into(),
                },
            ))
        });
        seed_scores(&db, 200, &[]);
        let mut spec = card4_spec(crate::catalog::ErrorPolicy::Quarantine, false);
        spec.quarantine_cap = 10; // tiny cap
        let plan_id = db.create_field_type_migration(spec).unwrap();
        let err = db.wait_for_migration(plan_id).unwrap_err();
        assert!(
            matches!(err, EngineError::MigrationQuarantineCapExceeded { cap: 10, .. }),
            "got {err:?}"
        );
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// AC: `retry_quarantined` with a fixed converter clears the `c:Q:` keys +
    /// writes the shadow; the plan then cuts over cleanly.
    #[test]
    fn retry_quarantined_rerunning_with_fixed_converter_clears_q_keys() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_or_fail_on(-1));
        let ids = seed_scores(&db, 30, &[5, 15, 25]);
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Quarantine, false))
            .unwrap();
        let _ = db.wait_for_migration(plan_id); // cutover refused; 3 quarantined
        let poison_ids: Vec<u64> = vec![ids[5], ids[15], ids[25]];
        assert_eq!(db.list_quarantined(plan_id).unwrap().len(), 3);
        // Register a GOOD converter at the plan's pinned name+version and retry.
        db.register_converter("good", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        let n = db.retry_quarantined(plan_id, &poison_ids, "good").unwrap();
        assert_eq!(n, 3);
        assert!(
            db.list_quarantined(plan_id).unwrap().is_empty(),
            "retry must clear the c:Q: keys"
        );
        // Resume (reopen with target schema; the pinned converter name is "widen",
        // re-register it with a good body) → cutover now passes → Completed.
        drop(db);
        let db2 = Database::open(parse_schema(r#"type User { score: f64 }"#).unwrap(), dir.path())
            .unwrap();
        db2.register_converter("widen", 1, |_oid, v| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => unreachable!(),
        });
        db2.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db2.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
    }

    /// A STRUCTURAL error (on-disk kind mismatch) ALWAYS halts, regardless of
    /// policy — it is never quarantined/skipped.
    #[test]
    fn structural_kind_mismatch_always_stops_under_quarantine() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema(r#"type User { score: i64 }"#).unwrap(), dir.path())
            .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let ids = seed_scores(&db, 12, &[]);
        // Overwrite one row's score with a String (kind SCALAR_STRING) — neither
        // src (i64) nor target (f64).
        let tid = db.resolve_type_id("User").unwrap();
        let mut sf = FieldMap::new();
        sf.insert("score".into(), Value::String("oops".into()));
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, ids[6]),
                    crate::object::serialize_fields(&sf),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();
        let plan_id = db
            .create_field_type_migration(card4_spec(crate::catalog::ErrorPolicy::Quarantine, false))
            .unwrap();
        let err = db.wait_for_migration(plan_id).unwrap_err();
        assert!(
            matches!(
                err,
                EngineError::Catalog(crate::CatalogError::MigrationRowUnexpectedKind { .. })
            ),
            "structural mismatch must halt, not quarantine; got {err:?}"
        );
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
        // It was NOT quarantined.
        assert!(db.list_quarantined(plan_id).unwrap().is_empty());
    }

    /// G3: while a migration is in flight the lazy/raw wire path
    /// (`get_many_lazy`) must NOT leak the worker-written `<field>__shadow`
    /// siblings — it ships `raw_fields` verbatim, bypassing the eager strip.
    #[test]
    fn lazy_path_strips_shadow_during_migration() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for i in 0..4i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        // Arm the hook so migrating_field_count > 0 (drives the strip gate).
        db.arm_field_hook(
            created.type_id,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: None,
                target_kind: target,
                converter_version: 1,
                plan_id: created.plan_id,
            },
        );
        // Backfill shadows (Converting), leaving source intact.
        let good = widen_i64_to_f64("User.score");
        let conv: crate::catalog::RegisteredConverter = std::sync::Arc::new(good);
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &conv).unwrap();

        let lazy = db.get_many_lazy("User", &ids).unwrap();
        assert_eq!(lazy.len(), ids.len());
        for mut obj in lazy {
            obj.ensure_fields_deserialized();
            assert!(
                obj.fields.keys().all(|k| !is_shadow_sibling_key(k)),
                "lazy path leaked shadow: {:?}",
                obj.fields.keys().collect::<Vec<_>>()
            );
            // Source still served (not yet cut over).
            assert!(matches!(obj.fields.get("score"), Some(Value::I64(_))));
        }
    }

    /// The backfill worker is idempotent across a resume: a second pass over
    /// already-shadowed rows (shadow present, current converter version) must
    /// skip them — it never re-invokes the converter.
    #[test]
    fn worker_shadow_backfill_idempotent_on_resume() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for i in 0..5i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            db.create("User", f).unwrap();
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let good: crate::catalog::RegisteredConverter =
            Arc::new(widen_i64_to_f64("User.score"));
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &good).unwrap();
        // Second pass: every row is already shadowed at the current version, so
        // this converter (which panics if called) must never run.
        let poison: crate::catalog::RegisteredConverter = Arc::new(|_oid, _v: &Value| {
            panic!("converter must not be re-invoked for an already-shadowed row")
        });
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &poison).unwrap();
        // Cutover still completes correctly.
        db.run_terminal_pass(created.plan_id, created.type_id).unwrap();
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let all = db2.scan_type("User").unwrap();
        assert_eq!(all.len(), 5);
        for obj in all {
            assert!(matches!(obj.fields.get("score"), Some(Value::F64(_))));
        }
    }

    /// A2 (review): a cutover that refuses a missing shadow must REWIND the
    /// plan to Converting (not leave it stuck at CuttingOver) so a resume
    /// re-backfills the missing shadow and completes — rather than re-refusing
    /// forever with quiesce held.
    #[test]
    fn cutover_missing_shadow_rewinds_and_recovers_via_resume() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for i in 0..4i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let conv: crate::catalog::RegisteredConverter = Arc::new(widen_i64_to_f64("User.score"));
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &conv).unwrap();
        // Corrupt one row back to source-only (drop its shadow) so cutover hits
        // the missing-shadow refusal.
        let tid = db.resolve_type_id("User").unwrap();
        let mut src_only = FieldMap::new();
        src_only.insert("score".into(), Value::I64(1));
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, ids[1]),
                    crate::object::serialize_fields(&src_only),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();

        let err = db.run_terminal_pass(created.plan_id, created.type_id).unwrap_err();
        assert!(matches!(err, EngineError::MigrationCutoverShadowMissing { .. }));
        // REWOUND to a clean Converting start (status Failed, cursors reset).
        let plans = crate::catalog::scan_migration_plans(&db.storage, db.storage.begin_txn().snapshot())
            .unwrap();
        let p = plans.iter().find(|p| p.plan_id == created.plan_id).unwrap();
        assert_eq!(p.status, crate::catalog::MigrationStatus::Failed);
        assert_eq!(p.phase, crate::catalog::MigrationPhase::Converting);
        assert_eq!(p.cursor, 0);
        assert_eq!(p.cutover_cursor, 0);
        drop(db);

        // Recover: reopen at the target schema (licensed mid-migration), register
        // the converter, resume → re-backfills the missing shadow + cuts over.
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db2.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        db2.resume_field_type_migration(created.plan_id).unwrap();
        assert_eq!(
            db2.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        for (id, i) in ids.iter().zip(0i64..) {
            match db2.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, i as f64),
                other => panic!("id {id}: expected F64, got {other:?}"),
            }
        }
    }

    /// A3 (review): crash mid-cutover, then reopen — auto-resume must finish the
    /// CuttingOver pass from the persisted `cutover_cursor` WITHOUT a converter
    /// (a rename-only pass), re-scanning already-promoted rows idempotently (the
    /// `source already target, no shadow → skip` arm).
    #[test]
    fn cutover_resumes_from_partial_via_reopen() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for i in 0..6i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 2, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        let conv: crate::catalog::RegisteredConverter = Arc::new(widen_i64_to_f64("User.score"));
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &conv).unwrap();

        // Simulate a crash PART WAY through the cutover: promote rows 0 and 1 by
        // hand (source := f64, drop the shadow) and mark the plan CuttingOver,
        // BUT leave cutover_cursor = 0 so resume re-scans from the start and must
        // skip the already-promoted rows.
        let tid = db.resolve_type_id("User").unwrap();
        let mut txn = db.storage.begin_txn();
        for &promoted in &ids[..2] {
            let mut f = FieldMap::new();
            let orig = ids.iter().position(|x| *x == promoted).unwrap() as f64;
            f.insert("score".into(), Value::F64(orig));
            db.storage
                .put_batch(
                    &mut txn,
                    &[(
                        rhypedb_storage::key::KeyBuilder::object(tid, promoted),
                        crate::object::serialize_fields(&f),
                    )],
                )
                .unwrap();
        }
        let mut plan = crate::catalog::load_migration_plan(&db.storage, &txn, created.plan_id)
            .unwrap();
        plan.phase = crate::catalog::MigrationPhase::CuttingOver;
        plan.cutover_cursor = 0;
        let (k, v) = crate::catalog::migration_plan_record(&plan);
        db.storage.put(&mut txn, &k, v).unwrap();
        db.storage.commit(&mut txn).unwrap();
        drop(db);

        // Reopen at the target schema; the converter registry is EMPTY after
        // restart. auto-resume must still finish the CuttingOver rename pass.
        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert_eq!(
            db2.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        for (id, i) in ids.iter().zip(0i64..) {
            match db2.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(f)) => assert_eq!(*f, i as f64),
                other => panic!("id {id}: expected F64, got {other:?}"),
            }
        }
    }

    /// A7 (review): a Null-source (and an already-target) row reads back
    /// correctly through a COVERED scan on a sibling @indexed field after
    /// cutover — the cutover's `None => continue` arm leaves the create-time
    /// covering payload, which is already cutover-correct for those rows.
    #[test]
    fn cutover_covered_read_of_null_and_target_rows() {
        use rhypedb_schema::{FieldType, ScalarType};
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type T { x: i64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("T.x"));
        // Normal rows.
        for i in 0..3i64 {
            let mut f = FieldMap::new();
            f.insert("x".into(), Value::I64(i * 10));
            f.insert("y".into(), Value::I64(i));
            db.create("T", f).unwrap();
        }
        // A row with a Null migrating field. Created through the normal path so
        // its o: blob AND its sibling-y covering payload are written
        // consistently (both Null) — the worker + cutover both skip it, leaving
        // those create-time entries, which must already read back as Null.
        let null_y = 99i64;
        {
            let mut f = FieldMap::new();
            f.insert("x".into(), Value::Null);
            f.insert("y".into(), Value::I64(null_y));
            db.create("T", f).unwrap();
        }

        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "T".into(),
                field_name: "x".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 8, ..Default::default()
            })
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type T { x: f64  y: i64 @indexed }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let results = db2.filter_scan("T", "y", CompareOp::Ge, 0, None).unwrap();
        assert_eq!(results.len(), 4);
        for obj in &results {
            assert!(obj.fields.keys().all(|k| !is_shadow_sibling_key(k)));
            let y = match obj.fields.get("y") {
                Some(Value::I64(v)) => *v,
                other => panic!("y: {other:?}"),
            };
            match obj.fields.get("x") {
                // The Null row stays Null; the rest are the converted f64.
                Some(Value::Null) | None if y == null_y => {}
                Some(Value::F64(f)) if y != null_y => assert_eq!(*f, (y * 10) as f64),
                other => panic!("covered read y={y} served wrong x: {other:?}"),
            }
        }
    }

    /// An on-disk row whose migrating-field kind is neither source nor target
    /// parks the plan `Failed` (quiesce held) rather than guessing.
    #[test]
    fn create_field_type_migration_unexpected_kind_fails() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let id = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(1));
            db.create("User", f).unwrap().id
        };
        // Overwrite with a String value: kind SCALAR_STRING, neither I64
        // (src) nor F64 (target).
        let tid = db.resolve_type_id("User").unwrap();
        let mut sf = FieldMap::new();
        sf.insert("score".into(), Value::String("oops".into()));
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, id),
                    crate::object::serialize_fields(&sf),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();

        // Async create returns Ok immediately; the unexpected-kind error surfaces
        // on the driver thread and is delivered to the first `wait_for_migration`.
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        let err = db.wait_for_migration(plan_id).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationRowUnexpectedKind { .. })
        ));
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Failed
        );
    }

    /// Creating a migration against an unregistered converter fails fast and
    /// persists NO plan.
    #[test]
    fn create_field_type_migration_unregistered_converter_refused() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let err = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "missing".into(),
                converter_version: 1,
                chunk_size: 0, ..Default::default()
            })
            .unwrap_err();
        assert!(matches!(err, EngineError::ConverterNotRegistered { .. }));
        assert!(db.list_migrations().unwrap().is_empty());
    }

    /// Plan ids are monotonic and never reused across a restart (the c:N:M
    /// counter self-heals against the max persisted plan id).
    #[test]
    fn migration_id_counter_monotonic_across_restart() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { a: i64  b: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User"));
        let mut f = FieldMap::new();
        f.insert("a".into(), Value::I64(1));
        f.insert("b".into(), Value::I64(2));
        db.create("User", f).unwrap();
        let p1 = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "a".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 0, ..Default::default()
            })
            .unwrap();
        assert_eq!(p1, 1);
        db.wait_for_migration(p1).unwrap();
        drop(db);
        let db2 = Database::open(
            parse_schema(r#"type User { a: f64  b: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db2.register_converter("widen", 1, widen_i64_to_f64("User"));
        let p2 = db2
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "b".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 0, ..Default::default()
            })
            .unwrap();
        assert_eq!(p2, 2, "plan id must not reset or reuse across restart");
        db2.wait_for_migration(p2).unwrap();
    }

    /// A second migration on a field that already has an unsettled plan is
    /// refused.
    #[test]
    fn create_field_type_migration_refused_with_active_plan() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::I64(3));
        db.create("User", f).unwrap();
        // Persist a Running plan directly (don't drive it to completion).
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let _created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 4, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        // A create against the same field now refuses.
        let err = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationFieldHasActivePlan { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Auto-resume on open (shadow-field card 1/5, increment 4)
    // -----------------------------------------------------------------

    // Persist a Running plan WITHOUT driving it — simulates a crash after
    // create_migration_plan committed the plan but before the chunk loop
    // finished. Returns the plan id.
    fn persist_running_plan(db: &Database, type_name: &str, field: &str) -> u64 {
        use rhypedb_schema::{FieldType, ScalarType};
        let target = crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        crate::catalog::create_migration_plan(
            &db.storage, &db.schema, type_name, field, target, "widen", 1, 4, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap()
        .plan_id
    }

    /// Open scans `c:P:`, re-arms the double-write hook for a Running plan
    /// (writes to the migrating field fail closed until the converter is
    /// registered), and — once registered — `resume_field_type_migration`
    /// drives it to completion. Also exercises the plan-aware reconcile (open
    /// with the TARGET schema while the catalog is still the source kind).
    #[test]
    fn database_open_auto_resumes_running_migrations() {
        let dir = tempfile::tempdir().unwrap();
        let mut ids = Vec::new();
        {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            for i in 0..6i64 {
                let mut f = FieldMap::new();
                f.insert("score".into(), Value::I64(i));
                ids.push(db.create("User", f).unwrap().id);
            }
            // Crash after the plan is persisted Running, before any drive.
            persist_running_plan(&db, "User", "score");
        }

        // Reopen with the TARGET schema: plan-aware reconcile accepts the
        // still-source catalog kind because a drivable plan covers it.
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Card 2d: open re-armed the double-write hook in a REJECTING state
        // (converter not registered yet) — a write to the MIGRATING field fails
        // closed, while other writes proceed.
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::F64(1.0));
        assert!(matches!(
            db.create("User", f),
            Err(EngineError::MigrationFieldConverterUnresolved { .. })
        ));
        let plan_id = db.list_migrations().unwrap()[0].plan_id;
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Running
        );

        // Register the converter and resume; the migration completes.
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        db.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        drop(db);

        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for (id, i) in ids.iter().zip(0i64..) {
            match db.get("User", *id).unwrap().fields.get("score") {
                Some(Value::F64(v)) => assert_eq!(*v, i as f64),
                other => panic!("id {id}: expected F64, got {other:?}"),
            }
        }
        // Quiesce released — writes flow again.
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::F64(9.0));
        assert!(db.create("User", f).is_ok());
    }

    /// The converter is NOT persisted — only its `(name, version)` is, in the
    /// plan. After a restart the operator re-registers the same name/version
    /// and resume resolves it. A version skew leaves the plan unresumable.
    #[test]
    fn converter_registry_round_trip_through_restart() {
        let dir = tempfile::tempdir().unwrap();
        let plan_id;
        {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(3));
            db.create("User", f).unwrap();
            plan_id = persist_running_plan(&db, "User", "score");
        }
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Wrong version registered -> resume can't resolve -> refused.
        db.register_converter("widen", 2, widen_i64_to_f64("User.score"));
        assert!(matches!(
            db.resume_field_type_migration(plan_id),
            Err(EngineError::ConverterNotRegistered { version: 1, .. })
        ));
        // Correct (name, version) -> resolves and completes.
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        db.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
    }

    /// Reopening with the SOURCE schema while a migration is in flight lets
    /// open succeed (catalog == schema), but driving the migration is refused
    /// — finishing it would flip the catalog to the target under a handle
    /// still validating against the source kind (blocker F3).
    #[test]
    fn resume_refused_when_reopened_with_source_schema() {
        let dir = tempfile::tempdir().unwrap();
        let plan_id;
        {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(3));
            db.create("User", f).unwrap();
            plan_id = persist_running_plan(&db, "User", "score");
        }
        // Reopen with the OLD (source) schema — open succeeds, hook re-armed.
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        assert!(matches!(
            db.resume_field_type_migration(plan_id),
            Err(EngineError::MigrationResumeSchemaMismatch { .. })
        ));
    }

    /// A torn-init reopen (exactly one of c:F:/c:I: present) blanket-clears the
    /// catalog keyspace and re-backfills — but the migration plan (`c:P:`) and
    /// the id counter (`c:N:M`) must survive (they aren't schema-derived), or
    /// crash-resume is lost and a freed id could be reissued (blocker G).
    #[test]
    fn recover_partial_preserves_migration_plan_and_counter() {
        let dir = tempfile::tempdir().unwrap();
        let plan_id;
        {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(3));
            db.create("User", f).unwrap();
            plan_id = persist_running_plan(&db, "User", "score");
            // Force the torn-init recovery branch on the next open by removing
            // the `c:I:` initialized marker (c:F: stays).
            let mut txn = db.storage.begin_txn();
            db.storage
                .delete_batch(
                    &mut txn,
                    &[rhypedb_storage::key::KeyBuilder::catalog_initialized()],
                )
                .unwrap();
            db.storage.commit(&mut txn).unwrap();
        }
        // Reopen (source schema) -> recover_partial runs -> plan + counter
        // must survive.
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let migs = db.list_migrations().unwrap();
        assert_eq!(migs.len(), 1, "plan wiped by recover_partial");
        assert_eq!(migs[0].plan_id, plan_id);
        // Drop this handle before reopening: the data-dir guard holds an
        // exclusive lock for a handle's lifetime, so the next open must not
        // overlap it (a real reopen follows a process exit, which releases it).
        drop(db);
        // Counter survived: a new plan on a DIFFERENT type gets the NEXT id,
        // not a reissued one. (A second plan on the SAME type is refused by
        // the type-scoped interlock, so use Post.)
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }  type Post { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let next = persist_running_plan(&db, "Post", "score");
        assert!(next > plan_id, "counter reissued a freed id: {next} <= {plan_id}");
    }

    /// A rename verb (here a type rename) is refused while an unsettled plan
    /// covers the type — the plan's name-keyed cutover would go stale
    /// (blocker F2).
    #[test]
    fn rename_refused_while_migration_plan_unsettled() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::I64(3));
        db.create("User", f).unwrap();
        persist_running_plan(&db, "User", "score");
        let err = db
            .run_migrations(vec![Migration::new("rename_user", |m| {
                m.rename_type("User", "Account")
            })])
            .unwrap_err();
        // run_migrations wraps the verb failure; the underlying reason is the
        // active-plan refusal.
        match err {
            EngineError::Catalog(crate::CatalogError::MigrationVerbFailed {
                reason, ..
            }) => assert!(
                reason.contains("active migration plan"),
                "unexpected reason: {reason}"
            ),
            other => panic!("expected MigrationVerbFailed, got {other:?}"),
        }
    }

    /// End-to-end migration where the objects live in an SST (not just the
    /// memtable): forces a flush between create and migrate so the chunk scan
    /// reads from disk, then reads results back through BOTH get() and
    /// get_many() — guards the SST scan + multi_get block-straddle paths at
    /// the engine level.
    #[test]
    fn migration_over_sst_flush_reads_via_get_and_get_many() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let mut ids = Vec::new();
        for i in 0..40i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        // Push the originals to an SST so the migration scan + the converted
        // writes produce multi-version keys on disk after the cutover.
        db.storage.flush().unwrap();
        let plan_id = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap();
        db.wait_for_migration(plan_id).unwrap();
        drop(db);
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // get()
        for (id, i) in ids.iter().zip(0i64..) {
            assert_eq!(
                db.get("User", *id).unwrap().fields.get("score"),
                Some(&Value::F64(i as f64)),
                "get() id {id}"
            );
        }
        // get_many() (multi_get_at path)
        let objs = db.get_many("User", &ids).unwrap();
        assert_eq!(objs.len(), ids.len());
        for obj in objs {
            assert!(
                matches!(obj.fields.get("score"), Some(Value::F64(_))),
                "get_many id {} stale: {:?}",
                obj.id,
                obj.fields.get("score")
            );
        }
    }

    /// A converter that fails mid-run leaves the plan `Failed` (quiesce held).
    /// The operator fixes the converter (same name/version), reopens with the
    /// target schema (reconcile must accept the still-source catalog for a
    /// Failed plan — BUG-2), and `resume_field_type_migration` completes it.
    #[test]
    fn failed_migration_recovers_via_public_resume() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let mut ids = Vec::new();
        let plan_id;
        {
            let db = Database::open(
                parse_schema(r#"type User { score: i64 }"#).unwrap(),
                dir.path(),
            )
            .unwrap();
            for i in 0..8i64 {
                let mut f = FieldMap::new();
                f.insert("score".into(), Value::I64(i));
                ids.push(db.create("User", f).unwrap().id);
            }
            // Converter that errors once it reaches the back half of the ids.
            let cutoff = ids[ids.len() / 2];
            let bad_calls = Arc::new(AtomicU64::new(0));
            let bc = Arc::clone(&bad_calls);
            db.register_converter("widen", 1, move |oid, v| {
                bc.fetch_add(1, Ordering::SeqCst);
                if oid >= cutoff {
                    return Err(EngineError::Catalog(
                        crate::CatalogError::FieldTypeChangeConverterFailed {
                            qualified: "User.score".into(),
                            object_id: oid,
                            reason: "transient".into(),
                        },
                    ));
                }
                match v {
                    Value::I64(i) => Ok(Value::F64(*i as f64)),
                    _ => unreachable!(),
                }
            });
            // Async create returns Ok; the converter failure surfaces on the
            // driver thread and is delivered to the first `wait_for_migration`.
            plan_id = db
                .create_field_type_migration(MigrationPlanSpec {
                    type_name: "User".into(),
                    field_name: "score".into(),
                    target_field_type: FieldType::Scalar(ScalarType::F64),
                    converter_name: "widen".into(),
                    converter_version: 1,
                    chunk_size: 2, ..Default::default()
                })
                .unwrap();
            let err = db.wait_for_migration(plan_id).unwrap_err();
            assert!(matches!(
                err,
                EngineError::Catalog(crate::CatalogError::FieldTypeChangeConverterFailed { .. })
            ));
            assert_eq!(
                db.list_migrations().unwrap()[0].status,
                crate::catalog::MigrationStatus::Failed
            );
        }
        // Reopen with the TARGET schema — reconcile must accept the Failed
        // plan's still-source catalog kind (else the type is bricked).
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Re-register the SAME (name, version) with a WORKING body.
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        db.resume_field_type_migration(plan_id).unwrap();
        assert_eq!(
            db.list_migrations().unwrap()[0].status,
            crate::catalog::MigrationStatus::Completed
        );
        drop(db);
        let db = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        for (id, i) in ids.iter().zip(0i64..) {
            assert_eq!(
                db.get("User", *id).unwrap().fields.get("score"),
                Some(&Value::F64(i as f64))
            );
        }
    }

    /// A migration on User.age must block a field-type change on User.score
    /// (a DIFFERENT field of the SAME type) — the worker rewrites the whole
    /// object blob, so concurrent plans on one type clobber each other
    /// (BUG-3, type-scoped interlock).
    #[test]
    fn change_field_type_refused_on_sibling_field_during_migration() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { age: i64  score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let mut f = FieldMap::new();
        f.insert("age".into(), Value::I64(1));
        f.insert("score".into(), Value::I64(2));
        db.create("User", f).unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User"));
        // Unsettled plan on User.age.
        persist_running_plan(&db, "User", "age");

        // Offline change on the sibling field is refused.
        let err = db
            .change_field_type(
                "User",
                "score",
                FieldType::Scalar(ScalarType::F64),
                |_, v| match v {
                    Value::I64(i) => Ok(Value::F64(*i as f64)),
                    _ => Ok(Value::F64(0.0)),
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationFieldHasActivePlan { .. })
        ));

        // A chunked migration on the sibling field is also refused.
        let err = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4, ..Default::default()
            })
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationFieldHasActivePlan { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Double-write producer hook + reader strip (isolation tests — the hook is
    // exercised directly and the strip via a hand-written shadow blob, so they
    // pin the mechanism independent of the live create/update path)
    // -----------------------------------------------------------------

    fn f64_conv() -> crate::catalog::RegisteredConverter {
        std::sync::Arc::new(|_oid: u64, v: &Value| match v {
            Value::I64(i) => Ok(Value::F64(*i as f64)),
            _ => Ok(Value::F64(0.0)),
        })
    }

    fn f64_target() -> u8 {
        use rhypedb_schema::{FieldType, ScalarType};
        crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64))
    }

    #[test]
    fn migrating_field_hook_stamps_shadow_and_cv() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64  name: String }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let tid = db.resolve_type_id("User").unwrap();
        db.arm_field_hook(
            tid,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: Some(f64_conv()),
                target_kind: f64_target(),
                converter_version: 7,
                plan_id: 1,
            },
        );
        let mut fields = FieldMap::new();
        fields.insert("score".into(), Value::I64(5));
        fields.insert("name".into(), Value::String("x".into()));
        db.apply_migrating_field_hook(tid, "User", 42, &mut fields)
            .unwrap();
        // Source preserved; shadow + cv stamped; unrelated field untouched.
        assert_eq!(fields.get("score"), Some(&Value::I64(5)));
        assert_eq!(fields.get("score__shadow"), Some(&Value::F64(5.0)));
        assert_eq!(fields.get("score__shadow_cv"), Some(&Value::U32(7)));
        assert_eq!(fields.get("name"), Some(&Value::String("x".into())));
    }

    #[test]
    fn migrating_field_hook_fails_closed_when_converter_unresolved() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let tid = db.resolve_type_id("User").unwrap();
        db.arm_field_hook(
            tid,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: None, // not registered yet
                target_kind: f64_target(),
                converter_version: 1,
                plan_id: 3,
            },
        );
        let mut fields = FieldMap::new();
        fields.insert("score".into(), Value::I64(5));
        assert!(matches!(
            db.apply_migrating_field_hook(tid, "User", 1, &mut fields),
            Err(EngineError::MigrationFieldConverterUnresolved { plan_id: 3, .. })
        ));
        // No shadow leaked on the fail-closed path.
        assert!(!fields.contains_key("score__shadow"));
    }

    #[test]
    fn migrating_field_hook_skips_null_and_absent_source() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let tid = db.resolve_type_id("User").unwrap();
        db.arm_field_hook(
            tid,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: Some(f64_conv()),
                target_kind: f64_target(),
                converter_version: 1,
                plan_id: 1,
            },
        );
        // Null source → no shadow.
        let mut nf = FieldMap::new();
        nf.insert("score".into(), Value::Null);
        db.apply_migrating_field_hook(tid, "User", 1, &mut nf).unwrap();
        assert!(!nf.contains_key("score__shadow"));
        // Field absent from this write → no shadow.
        let mut af = FieldMap::new();
        af.insert("other".into(), Value::I64(1));
        db.apply_migrating_field_hook(tid, "User", 1, &mut af).unwrap();
        assert!(!af.contains_key("score__shadow"));
    }

    #[test]
    fn reader_strips_shadow_siblings() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let id = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(5));
            db.create("User", f).unwrap().id
        };
        // Hand-write a blob carrying shadow siblings (simulates a double-write).
        let tid = db.resolve_type_id("User").unwrap();
        let mut blob = FieldMap::new();
        blob.insert("score".into(), Value::I64(5));
        blob.insert("score__shadow".into(), Value::F64(5.0));
        blob.insert("score__shadow_cv".into(), Value::U32(1));
        let mut txn = db.storage.begin_txn();
        db.storage
            .put_batch(
                &mut txn,
                &[(
                    rhypedb_storage::key::KeyBuilder::object(tid, id),
                    crate::object::serialize_fields(&blob),
                )],
            )
            .unwrap();
        db.storage.commit(&mut txn).unwrap();
        // Arm a hook so the strip is active (count > 0).
        db.arm_field_hook(
            tid,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: Some(f64_conv()),
                target_kind: f64_target(),
                converter_version: 1,
                plan_id: 1,
            },
        );
        // get() and get_many() must NOT expose the shadow siblings.
        let obj = db.get("User", id).unwrap();
        assert_eq!(obj.fields.get("score"), Some(&Value::I64(5)));
        assert!(!obj.fields.contains_key("score__shadow"));
        assert!(!obj.fields.contains_key("score__shadow_cv"));
        let many = db.get_many("User", &[id]).unwrap();
        assert_eq!(many.len(), 1);
        assert!(!many[0].fields.contains_key("score__shadow"));
        assert!(!many[0].fields.contains_key("score__shadow_cv"));
    }

    // -----------------------------------------------------------------
    // Migration log (card 5/5)
    // -----------------------------------------------------------------

    /// Running a list of named migrations applies them in order, records
    /// each in the log, and bumps the catalog's migration version.
    #[test]
    fn run_migrations_applies_and_records() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let report = db
            .run_migrations(vec![
                Migration::new("001_rename_user_to_account", |m| {
                    m.rename_type("User", "Account")
                }),
            ])
            .unwrap();
        assert_eq!(report.version_before, 0);
        assert_eq!(report.version_after, 1);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].ordinal, 0);
        assert_eq!(report.applied[0].name, "001_rename_user_to_account");
    }

    /// Re-running the same list is a no-op — already-applied
    /// migrations are skipped.
    #[test]
    fn run_migrations_is_idempotent_across_reopens() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema.clone(), dir.path()).unwrap();
        let _ = db
            .run_migrations(vec![Migration::new("001", |m| {
                m.rename_type("User", "Account")
            })])
            .unwrap();
        drop(db);

        // Reopen with the post-rename schema and re-run the same
        // migration list. Nothing should happen.
        let post_schema = parse_schema(r#"type Account { name: String }"#).unwrap();
        let db2 = Database::open(post_schema, dir.path()).unwrap();
        let report = db2
            .run_migrations(vec![Migration::new("001", |_m| Ok(()))])
            .unwrap();
        assert_eq!(report.version_before, 1);
        assert_eq!(report.version_after, 1);
        assert!(report.applied.is_empty());
    }

    /// Renaming an applied migration is refused.
    #[test]
    fn run_migrations_refuses_renamed_migration() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let _ = db
            .run_migrations(vec![Migration::new("original_name", |m| {
                m.rename_type("User", "Account")
            })])
            .unwrap();
        drop(db);

        // Same logic but with a different name at ordinal 0.
        let post_schema = parse_schema(r#"type Account { name: String }"#).unwrap();
        let db2 = Database::open(post_schema, dir.path()).unwrap();
        let err = db2
            .run_migrations(vec![Migration::new("renamed_after_applied", |_| Ok(()))])
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationNameMismatch {
                ordinal: 0,
                ..
            })
        ));
    }

    /// Code list shorter than catalog → DB-ahead-of-code error.
    #[test]
    fn run_migrations_refuses_when_db_is_ahead_of_code() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let _ = db
            .run_migrations(vec![Migration::new("001", |m| {
                m.rename_type("User", "Account")
            })])
            .unwrap();
        drop(db);

        let post_schema = parse_schema(r#"type Account { name: String }"#).unwrap();
        let db2 = Database::open(post_schema, dir.path()).unwrap();
        let err = db2.run_migrations(vec![]).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationListShorterThanApplied {
                code_count: 0,
                catalog_count: 1,
            })
        ));
    }

    /// Adding one more migration runs only the new one.
    #[test]
    fn run_migrations_only_runs_new_entries() {
        use crate::catalog::Migration;
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let _ = db
            .run_migrations(vec![Migration::new("001", |m| {
                m.rename_type("User", "Account")
            })])
            .unwrap();
        drop(db);

        let post = parse_schema(r#"type Account { name: String }"#).unwrap();
        let db2 = Database::open(post, dir.path()).unwrap();
        let report = db2
            .run_migrations(vec![
                Migration::new("001", |_| Ok(())),
                Migration::new("002", |m| m.rename_type("Account", "Member")),
            ])
            .unwrap();
        assert_eq!(report.version_before, 1);
        assert_eq!(report.version_after, 2);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].ordinal, 1);
        assert_eq!(report.applied[0].name, "002");
    }

    fn test_schema() -> Schema {
        parse_schema(
            r#"
            type User {
                name: String
                email: String @unique
                age: u32
            }

            type Post {
                title: String
                body: String
                author: User @on_delete(cascade)
            }

            type Tag {
                name: String @unique
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn create_and_get_object() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("email".into(), Value::String("alice@example.com".into()));
        fields.insert("age".into(), Value::U32(30));

        let user = db.create("User", fields).unwrap();
        assert_eq!(user.type_name, "User");
        assert_eq!(
            user.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );

        let fetched = db.get("User", user.id).unwrap();
        assert_eq!(fetched.fields.get("name"), user.fields.get("name"));
        assert_eq!(fetched.fields.get("email"), user.fields.get("email"));
        assert_eq!(fetched.fields.get("age"), user.fields.get("age"));
    }

    #[test]
    fn scan_type_returns_all_objects() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        for i in 0..5 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("User{i}")));
            db.create("User", f).unwrap();
        }

        let all = db.scan_type("User").unwrap();
        assert_eq!(all.len(), 5);

        let names: Vec<_> = all
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"User0".to_string()));
        assert!(names.contains(&"User4".to_string()));
    }

    #[test]
    fn scan_type_excludes_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        db.create("User", f2).unwrap();

        db.delete("User", alice.id).unwrap();

        let all = db.scan_type("User").unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(
            all[0].fields.get("name"),
            Some(&Value::String("Bob".into()))
        );
    }

    #[test]
    fn get_projected_returns_only_requested_fields() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("email".into(), Value::String("alice@example.com".into()));
        fields.insert("age".into(), Value::U32(30));
        let user = db.create("User", fields).unwrap();

        let proj = db.get_projected("User", user.id, &["name", "age"]).unwrap();
        assert_eq!(proj.id, user.id);
        assert_eq!(proj.fields.len(), 2);
        assert_eq!(proj.fields.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(proj.fields.get("age"), Some(&Value::U32(30)));
        assert!(!proj.fields.contains_key("email"), "email was not projected");

        // Projected values agree with a full get.
        let full = db.get("User", user.id).unwrap();
        assert_eq!(proj.fields.get("name"), full.fields.get("name"));
        assert_eq!(proj.fields.get("age"), full.fields.get("age"));
    }

    #[test]
    fn get_projected_missing_object_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();
        let err = db.get_projected("User", 9999, &["name"]).unwrap_err();
        assert!(
            matches!(err, EngineError::ObjectNotFound { .. }),
            "expected ObjectNotFound, got {err:?}"
        );
    }

    #[test]
    fn scan_type_projected_returns_only_requested_fields() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        for i in 0..5 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("User{i}")));
            f.insert("email".into(), Value::String(format!("u{i}@x.com")));
            f.insert("age".into(), Value::U32(20 + i));
            db.create("User", f).unwrap();
        }

        let projected = db.scan_type_projected("User", &["name"]).unwrap();
        assert_eq!(projected.len(), 5);
        for obj in &projected {
            assert_eq!(obj.fields.len(), 1, "only `name` should be materialized");
            assert!(matches!(obj.fields.get("name"), Some(Value::String(_))));
        }

        // Same id set + same name values as a full scan.
        let mut proj_names: Vec<_> = projected
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        let mut full_names: Vec<_> = db
            .scan_type("User")
            .unwrap()
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        proj_names.sort();
        full_names.sort();
        assert_eq!(proj_names, full_names);
    }

    #[test]
    fn update_object() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("age".into(), Value::U32(30));

        let user = db.create("User", fields).unwrap();

        let mut updates = FieldMap::new();
        updates.insert("age".into(), Value::U32(31));

        let updated = db.update("User", user.id, updates).unwrap();
        assert_eq!(updated.fields.get("age"), Some(&Value::U32(31)));
        assert_eq!(
            updated.fields.get("name"),
            Some(&Value::String("Alice".into()))
        );
    }

    #[test]
    fn delete_object() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Bob".into()));

        let user = db.create("User", fields).unwrap();
        db.delete("User", user.id).unwrap();

        let result = db.get("User", user.id);
        assert!(matches!(result, Err(EngineError::ObjectNotFound { .. })));
    }

    #[test]
    fn reject_invalid_field_type() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::U32(42)); // wrong type

        let result = db.create("User", fields);
        assert!(matches!(result, Err(EngineError::TypeMismatch { .. })));
    }

    #[test]
    fn reject_unknown_field() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut fields = FieldMap::new();
        fields.insert("nonexistent".into(), Value::String("nope".into()));

        let result = db.create("User", fields);
        assert!(matches!(result, Err(EngineError::FieldNotFound { .. })));
    }

    #[test]
    fn reject_unknown_type() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let result = db.create("NonExistent", FieldMap::new());
        assert!(matches!(result, Err(EngineError::TypeNotFound(_))));
    }

    #[test]
    fn link_and_get_links() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                friends: [User] @on_delete(remove)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        let bob = db.create("User", f2).unwrap();

        db.link("User", alice.id, "friends", bob.id, None).unwrap();

        let links = db.get_links("User", alice.id, "friends").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0, bob.id);
    }

    #[test]
    fn link_multiple_targets() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                friends: [User] @on_delete(remove)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let alice = db
            .create("User", {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("Alice".into()));
                f
            })
            .unwrap();
        let bob = db
            .create("User", {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("Bob".into()));
                f
            })
            .unwrap();
        let carol = db
            .create("User", {
                let mut f = FieldMap::new();
                f.insert("name".into(), Value::String("Carol".into()));
                f
            })
            .unwrap();

        db.link("User", alice.id, "friends", bob.id, None).unwrap();
        db.link("User", alice.id, "friends", carol.id, None)
            .unwrap();

        let links = db.get_links("User", alice.id, "friends").unwrap();
        assert_eq!(links.len(), 2);
        let ids: Vec<_> = links.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&bob.id));
        assert!(ids.contains(&carol.id));
    }

    #[test]
    fn unlink_removes_edge() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                friends: [User] @on_delete(remove)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        let bob = db.create("User", f2).unwrap();

        db.link("User", alice.id, "friends", bob.id, None).unwrap();
        db.unlink("User", alice.id, "friends", bob.id).unwrap();

        let links = db.get_links("User", alice.id, "friends").unwrap();
        assert_eq!(links.len(), 0);
    }

    #[test]
    fn link_with_edge_properties() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                favorite_movies: [Movie] {
                    rating: f32
                } @on_delete(remove)
            }
            type Movie {
                title: String
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        let alien = db.create("Movie", mf).unwrap();

        let mut edge_props = FieldMap::new();
        edge_props.insert("rating".into(), Value::F32(4.5));

        db.link(
            "User",
            alice.id,
            "favorite_movies",
            alien.id,
            Some(edge_props),
        )
        .unwrap();

        let links = db.get_links("User", alice.id, "favorite_movies").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0, alien.id);
        assert_eq!(links[0].1.get("rating"), Some(&Value::F32(4.5)));
    }

    fn edge_validation_db(
        dir: &std::path::Path,
        rating_type: &str,
    ) -> (std::sync::Arc<Database>, u64, u64) {
        let schema = parse_schema(&format!(
            r#"
            type User {{
                name: String
                favorite_movies: [Movie] {{ rating: {rating_type} }} @on_delete(remove)
            }}
            type Movie {{ title: String }}
            "#
        ))
        .unwrap();
        let db = Database::open(schema, dir).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        let alien = db.create("Movie", mf).unwrap();
        (db, alice.id, alien.id)
    }

    #[test]
    fn link_rejects_wrong_edge_value_variant() {
        let dir = tempfile::tempdir().unwrap();
        // Declared f64, but a caller passes an F32 value — must be rejected,
        // not silently stored as the wrong variant.
        let (db, user, movie) = edge_validation_db(dir.path(), "f64");
        let mut bad = FieldMap::new();
        bad.insert("rating".into(), Value::F32(4.5));
        let err = db.link("User", user, "favorite_movies", movie, Some(bad));
        assert!(
            matches!(err, Err(EngineError::TypeMismatch { .. })),
            "got {err:?}"
        );
        // The matching variant is accepted.
        let mut good = FieldMap::new();
        good.insert("rating".into(), Value::F64(4.5));
        db.link("User", user, "favorite_movies", movie, Some(good)).unwrap();
    }

    #[test]
    fn link_rejects_undeclared_edge_field() {
        let dir = tempfile::tempdir().unwrap();
        let (db, user, movie) = edge_validation_db(dir.path(), "f32");
        let mut bad = FieldMap::new();
        bad.insert("bogus".into(), Value::F32(1.0));
        let err = db.link("User", user, "favorite_movies", movie, Some(bad));
        assert!(
            matches!(err, Err(EngineError::FieldNotFound { .. })),
            "got {err:?}"
        );
    }

    #[test]
    fn delete_with_on_delete_deny() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                author: User @on_delete(deny)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Hello World".into()));
        let post = db.create("Post", pf).unwrap();

        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // Deleting alice should be denied because a post references her.
        let result = db.delete("User", alice.id);
        assert!(matches!(result, Err(EngineError::DeleteDenied { .. })));

        // Alice should still exist.
        assert!(db.get("User", alice.id).is_ok());
    }

    #[test]
    fn delete_with_on_delete_cascade() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Hello World".into()));
        let post = db.create("Post", pf).unwrap();

        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // Deleting alice should cascade-delete the post.
        db.delete("User", alice.id).unwrap();

        assert!(matches!(
            db.get("User", alice.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
        assert!(matches!(
            db.get("Post", post.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
    }

    /// Issue #13: the `*_with_origin` write verbs stamp their opaque origin onto
    /// every emitted `ChangeEvent`; the plain verbs stamp `None`; and a cascade
    /// delete fans the SAME origin onto every event it produces.
    #[test]
    fn write_origin_stamps_change_events() {
        use rhypedb_subscribe::{ChangeKind, SubscriptionFilter};
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User { name: String }
            type Post { title: String  author: User @on_delete(cascade) }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let (_id, rx) = db.subscriptions().subscribe(SubscriptionFilter::all());
        let next = |rx: &std::sync::mpsc::Receiver<rhypedb_subscribe::ChangeEvent>| {
            rx.recv_timeout(Duration::from_secs(1)).expect("an event")
        };

        // create_with_origin → Create event carries the origin.
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create_with_origin("User", uf, Some(42)).unwrap();
        let e = next(&rx);
        assert_eq!((e.kind, e.type_name.as_str(), e.origin), (ChangeKind::Create, "User", Some(42)));

        // plain create → untagged.
        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Hello".into()));
        let post = db.create("Post", pf).unwrap();
        let e = next(&rx);
        assert_eq!((e.kind, e.origin), (ChangeKind::Create, None));

        // link emits no change event today, so the next event is the update.
        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // update_with_origin → Update event carries the origin.
        let mut uf2 = FieldMap::new();
        uf2.insert("name".into(), Value::String("Alicia".into()));
        db.update_with_origin("User", alice.id, uf2, Some(43)).unwrap();
        let e = next(&rx);
        assert_eq!((e.kind, e.origin), (ChangeKind::Update, Some(43)));

        // delete_with_origin → the top-level AND cascaded Delete events all
        // carry the ONE origin passed to the call.
        db.delete_with_origin("User", alice.id, Some(44)).unwrap();
        let mut deletes = Vec::new();
        while let Ok(e) = rx.recv_timeout(Duration::from_millis(300)) {
            deletes.push(e);
        }
        assert_eq!(deletes.len(), 2, "User + cascaded Post = 2 delete events");
        assert!(deletes.iter().all(|e| e.kind == ChangeKind::Delete));
        assert!(
            deletes.iter().all(|e| e.origin == Some(44)),
            "every cascade delete event must carry the same origin"
        );
        let types: std::collections::HashSet<&str> =
            deletes.iter().map(|e| e.type_name.as_str()).collect();
        assert!(types.contains("User") && types.contains("Post"));
    }

    // --- Shadow-field card 2d: live writes during migration (no quiesce) ---

    fn quiesce_cascade_db() -> (Arc<Database>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        (db, dir)
    }

    fn named(name: &str) -> FieldMap {
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String(name.into()));
        f
    }

    /// Card 2d: live writes to a migrating type PROCEED (no quiesce). The
    /// double-write hook stamps a converted shadow on each create/update, the
    /// backfill worker skips those already-shadowed rows, and the cutover
    /// promotes everything — so a row written DURING the migration ends at the
    /// target with its latest value.
    #[test]
    fn writes_proceed_and_double_write_during_migration() {
        use rhypedb_schema::{FieldType, ScalarType};
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        db.register_converter("widen", 1, widen_i64_to_f64("User.score"));
        let mut ids = Vec::new();
        for i in 0..4i64 {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(i));
            ids.push(db.create("User", f).unwrap().id);
        }
        // Begin a migration (plan + hook armed) but do NOT cut over yet.
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let converter: crate::catalog::RegisteredConverter =
            Arc::new(widen_i64_to_f64("User.score"));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        db.arm_field_hook(
            created.type_id,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: Some(Arc::clone(&converter)),
                target_kind: target,
                converter_version: 1,
                plan_id: created.plan_id,
            },
        );

        // Live writes mid-migration PROCEED + double-write a shadow; the
        // returned Object is shadow-free.
        let live_created = {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(100));
            let obj = db.create("User", f).unwrap();
            assert!(obj.fields.keys().all(|k| !is_shadow_sibling_key(k)));
            obj.id
        };
        {
            let mut f = FieldMap::new();
            f.insert("score".into(), Value::I64(999));
            let obj = db.update("User", ids[0], f).unwrap();
            assert!(obj.fields.keys().all(|k| !is_shadow_sibling_key(k)));
        }

        // Backfill the remaining rows + cut over.
        crate::catalog::run_migration_chunks(&db.storage, created.plan_id, &converter).unwrap();
        db.run_terminal_pass(created.plan_id, created.type_id).unwrap();
        drop(db);

        let db2 = Database::open(
            parse_schema(r#"type User { score: f64 }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        // Live-updated row reflects the NEW value, converted.
        assert!(matches!(
            db2.get("User", ids[0]).unwrap().fields.get("score"),
            Some(Value::F64(f)) if *f == 999.0
        ));
        // Live-created row converted.
        assert!(matches!(
            db2.get("User", live_created).unwrap().fields.get("score"),
            Some(Value::F64(f)) if *f == 100.0
        ));
        // Untouched originals converted.
        for (id, i) in ids[1..].iter().zip(1i64..) {
            assert!(matches!(
                db2.get("User", *id).unwrap().fields.get("score"),
                Some(Value::F64(f)) if *f == i as f64
            ));
        }
    }

    /// Card 2d open-to-register window: a write that TOUCHES a migrating field
    /// whose converter is unresolved FAILS CLOSED — never lands a source-only
    /// value the cutover would later refuse. A write that doesn't touch the
    /// migrating field proceeds.
    #[test]
    fn write_to_migrating_field_with_unresolved_converter_fails_closed() {
        use rhypedb_schema::{FieldType, ScalarType};
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { score: i64  tag: String }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let target =
            crate::catalog::schema_kind_byte_public(&FieldType::Scalar(ScalarType::F64));
        let created = crate::catalog::create_migration_plan(
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16, None, 0, crate::catalog::ErrorPolicy::Stop, false, 0,
        )
        .unwrap();
        db.arm_field_hook(
            created.type_id,
            MigratingFieldHook {
                field_name: "score".into(),
                converter: None,
                target_kind: target,
                converter_version: 1,
                plan_id: created.plan_id,
            },
        );
        let mut f = FieldMap::new();
        f.insert("score".into(), Value::I64(5));
        assert!(matches!(
            db.create("User", f),
            Err(EngineError::MigrationFieldConverterUnresolved { .. })
        ));
        let mut f = FieldMap::new();
        f.insert("tag".into(), Value::String("ok".into()));
        assert!(db.create("User", f).is_ok());
    }

    /// Card 2d: a delete (incl. a cascade) into a migrating type is now ALLOWED
    /// — the object and any `<field>__shadow` siblings drop together.
    #[test]
    fn delete_cascade_into_migrating_type_proceeds() {
        let (db, _dir) = quiesce_cascade_db();
        let alice = db.create("User", named("Alice")).unwrap();
        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Hello".into()));
        let post = db.create("Post", pf).unwrap();
        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // Arm a migration hook on the CHILD (Post). Deleting the PARENT (User)
        // cascades into Post — under card 1 this was rejected; under 2d it
        // proceeds and removes BOTH objects. (delete drops the whole blob, so
        // it never invokes the hook's converter.)
        let post_tid = db.resolve_type_id("Post").unwrap();
        db.arm_field_hook(
            post_tid,
            MigratingFieldHook {
                field_name: "title".into(),
                converter: None,
                target_kind: crate::catalog::schema_kind_byte_public(
                    &rhypedb_schema::FieldType::Scalar(rhypedb_schema::ScalarType::Bytes),
                ),
                converter_version: 1,
                plan_id: 9,
            },
        );
        db.delete("User", alice.id).unwrap();
        assert!(matches!(
            db.get("User", alice.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
        assert!(matches!(
            db.get("Post", post.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
    }

    /// Card 2d regression (review #1/#2): the background cover-refresh worker
    /// MUST take `migration_lock.read()`, so the cutover's
    /// `migration_lock.write()` pass mutually excludes it. Without the lock a
    /// refresh racing the cutover could bake a `<field>__shadow` into a cover
    /// (disarm between blob-read and strip-decision) or stamp a post-bump
    /// `cover_v` onto a stale blob (defeating the cutover generation-bump).
    /// Deterministic lock-discipline check: while a `write()` guard is held the
    /// worker call must BLOCK; it completes only once the guard drops.
    #[test]
    fn cover_refresh_excluded_under_migration_write_lock() {
        use std::sync::mpsc;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(
            parse_schema(r#"type User { name: String }"#).unwrap(),
            dir.path(),
        )
        .unwrap();
        let tid = db.resolve_type_id("User").unwrap();
        let guard = db.migration_lock.write();
        let (tx, rx) = mpsc::channel();
        let db2 = Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            tx.send(()).unwrap(); // signal: about to take migration_lock.read()
            db2.refresh_covers_for_target(tid, 1)
        });
        rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            !handle.is_finished(),
            "cover-refresh ran while migration_lock.write() was held — it must take migration_lock.read()"
        );
        drop(guard);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn converter_registry_resolves_by_name_and_version() {
        let (db, _dir) = quiesce_cascade_db();
        db.register_converter("widen", 1, |_id, v| Ok(v.clone()));
        assert!(db.resolve_converter("widen", 1).is_some()); // exact match
        assert!(db.resolve_converter("widen", 2).is_none()); // version skew → park
        assert!(db.resolve_converter("nope", 1).is_none()); // missing → park
    }

    #[test]
    fn converter_registry_survives_consuming_rebuild() {
        // Per-Database registry must ride the _consuming carry so a converter
        // registered before a migrate verb is still resolvable on the new
        // handle. clone_into_new_handle is the carry path every _consuming
        // verb funnels through.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema("type User { name: String }").unwrap();
        let db = Database::open(schema.clone(), dir.path()).unwrap();
        db.register_converter("widen", 1, |_id, v| Ok(v.clone()));
        let db2 = db.clone_into_new_handle(schema).unwrap();
        assert!(db2.resolve_converter("widen", 1).is_some());
    }

    #[test]
    fn change_field_type_accepts_datetime_target() {
        // DateTime/Json now have writable `Value` variants, so they are
        // representable migration targets — the up-front "unrepresentable target"
        // refusal no longer applies, and a converter that yields a DateTime value
        // is accepted.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema("type User { age: i64 }").unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let res = db.change_field_type(
            "User",
            "age",
            rhypedb_schema::FieldType::Scalar(rhypedb_schema::ScalarType::DateTime),
            |_id, v| {
                Ok(match v {
                    Value::I64(ms) => Value::DateTime(*ms),
                    other => other.clone(),
                })
            },
        );
        assert!(
            res.is_ok(),
            "DateTime is now a representable migration target: {res:?}"
        );
    }

    #[test]
    fn inverse_relationship_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                posts: [Post] @inverse(Post.author)
            }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf1 = FieldMap::new();
        pf1.insert("title".into(), Value::String("First Post".into()));
        let post1 = db.create("Post", pf1).unwrap();

        let mut pf2 = FieldMap::new();
        pf2.insert("title".into(), Value::String("Second Post".into()));
        let post2 = db.create("Post", pf2).unwrap();

        // Link posts to alice via Post.author
        db.link("Post", post1.id, "author", alice.id, None).unwrap();
        db.link("Post", post2.id, "author", alice.id, None).unwrap();

        // Traverse via User.posts (which is @inverse of Post.author)
        let posts = db.get_links("User", alice.id, "posts").unwrap();
        assert_eq!(posts.len(), 2);
        let ids: Vec<_> = posts.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&post1.id));
        assert!(ids.contains(&post2.id));
    }

    #[test]
    fn unique_constraint_on_create() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        f1.insert("email".into(), Value::String("alice@example.com".into()));
        db.create("User", f1).unwrap();

        // Same email should fail.
        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        f2.insert("email".into(), Value::String("alice@example.com".into()));
        let result = db.create("User", f2);
        assert!(matches!(result, Err(EngineError::UniqueViolation { .. })));
    }

    #[test]
    fn unique_constraint_allows_different_values() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        f1.insert("email".into(), Value::String("alice@example.com".into()));
        db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        f2.insert("email".into(), Value::String("bob@example.com".into()));
        db.create("User", f2).unwrap(); // should succeed
    }

    #[test]
    fn find_by_unique_returns_matching_object_or_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("email".into(), Value::String("alice@example.com".into()));
        let alice = db.create("User", f).unwrap();

        // Hit: the unique email resolves to exactly Alice.
        let got = db
            .find_by_unique("User", "email", &Value::String("alice@example.com".into()))
            .unwrap();
        assert_eq!(got.map(|o| o.id), Some(alice.id));

        // Miss: no row with that email.
        assert!(
            db.find_by_unique("User", "email", &Value::String("nobody@example.com".into()))
                .unwrap()
                .is_none()
        );

        // After delete, the unique lookup no longer finds it.
        db.delete("User", alice.id).unwrap();
        assert!(
            db.find_by_unique("User", "email", &Value::String("alice@example.com".into()))
                .unwrap()
                .is_none()
        );

        // is_field_unique reflects the schema.
        assert!(db.is_field_unique("User", "email"));
        assert!(!db.is_field_unique("User", "name"));
    }

    #[test]
    fn unique_constraint_on_update() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        f1.insert("email".into(), Value::String("alice@example.com".into()));
        db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        f2.insert("email".into(), Value::String("bob@example.com".into()));
        let bob = db.create("User", f2).unwrap();

        // Updating bob's email to alice's should fail.
        let mut updates = FieldMap::new();
        updates.insert("email".into(), Value::String("alice@example.com".into()));
        let result = db.update("User", bob.id, updates);
        assert!(matches!(result, Err(EngineError::UniqueViolation { .. })));
    }

    #[test]
    fn update_unique_field_to_null_frees_value_for_reuse() {
        // Regression: updating a `@unique` field FROM a value TO Null used to
        // skip the `u:` removal entirely (the old guard gated removal on the
        // NEW value being non-null), dangling `u:<type>:<field>:<old>` -> the
        // freed value could no longer be re-created/re-assigned (false
        // UniqueViolation) and a bare `find_by_unique` probe returned the stale
        // row. After update-to-null the value must be fully released.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("email".into(), Value::String("alice@example.com".into()));
        let alice = db.create("User", f).unwrap();

        // Null out the unique field.
        let mut updates = FieldMap::new();
        updates.insert("email".into(), Value::Null);
        db.update("User", alice.id, updates).unwrap();

        // (b/c) The bare unique probe no longer resolves the freed value — i.e.
        // the `u:` key is actually gone (find_by_unique does NOT re-filter, so a
        // hit here would mean a dangling key still on disk).
        assert!(
            db.find_by_unique("User", "email", &Value::String("alice@example.com".into()))
                .unwrap()
                .is_none(),
            "the old unique key must be removed when the field is set to Null"
        );

        // (a) The freed value can be claimed by a brand-new row.
        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Bob".into()));
        f2.insert("email".into(), Value::String("alice@example.com".into()));
        let bob = db
            .create("User", f2)
            .expect("re-creating the freed unique value must succeed");

        // ...and the probe now resolves to the NEW owner.
        assert_eq!(
            db.find_by_unique("User", "email", &Value::String("alice@example.com".into()))
                .unwrap()
                .map(|o| o.id),
            Some(bob.id)
        );

        // The nulled row is still live, just no longer holding the value.
        let alice_now = db.get("User", alice.id).unwrap();
        assert_ne!(
            alice_now.fields.get("email"),
            Some(&Value::String("alice@example.com".into()))
        );
    }

    #[test]
    fn update_unique_field_can_be_claimed_again_via_update() {
        // The freed value must also be reusable via UPDATE (not only create),
        // and the uniqueness constraint must re-apply once it is re-claimed.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("email".into(), Value::String("a@x.com".into()));
        let a = db.create("User", f).unwrap();

        let mut f = FieldMap::new();
        f.insert("email".into(), Value::String("b@x.com".into()));
        let b = db.create("User", f).unwrap();

        // a -> Null releases "a@x.com".
        let mut up = FieldMap::new();
        up.insert("email".into(), Value::Null);
        db.update("User", a.id, up).unwrap();

        // b can now take the freed value.
        let mut up = FieldMap::new();
        up.insert("email".into(), Value::String("a@x.com".into()));
        db.update("User", b.id, up)
            .expect("re-claiming the freed value via update must succeed");
        assert_eq!(
            db.find_by_unique("User", "email", &Value::String("a@x.com".into()))
                .unwrap()
                .map(|o| o.id),
            Some(b.id)
        );

        // Constraint re-applies: a trying to re-take it now collides with b.
        let mut up = FieldMap::new();
        up.insert("email".into(), Value::String("a@x.com".into()));
        assert!(matches!(
            db.update("User", a.id, up),
            Err(EngineError::UniqueViolation { .. })
        ));
    }

    #[test]
    fn update_to_null_on_one_unique_field_leaves_a_sibling_unique_field_intact() {
        // Two @unique fields on one type: nulling ONE in an update must free
        // exactly that value and leave the other field's claim untouched (the
        // per-field loop must not over-remove). Uses a local schema so the
        // shared test_schema fixture stays single-unique.
        let schema = parse_schema(
            r#"
            type Account {
                email: String @unique
                handle: String @unique
            }
            "#,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("email".into(), Value::String("e@x.com".into()));
        f.insert("handle".into(), Value::String("h1".into()));
        let acc = db.create("Account", f).unwrap();

        // Null ONLY email; handle is left as-is in the same update call.
        let mut up = FieldMap::new();
        up.insert("email".into(), Value::Null);
        db.update("Account", acc.id, up).unwrap();

        // email freed...
        assert!(
            db.find_by_unique("Account", "email", &Value::String("e@x.com".into()))
                .unwrap()
                .is_none()
        );
        let mut f2 = FieldMap::new();
        f2.insert("email".into(), Value::String("e@x.com".into()));
        db.create("Account", f2)
            .expect("the nulled unique value must be reusable");

        // ...but handle is STILL held by acc (not collaterally removed).
        assert_eq!(
            db.find_by_unique("Account", "handle", &Value::String("h1".into()))
                .unwrap()
                .map(|o| o.id),
            Some(acc.id)
        );
        let mut f3 = FieldMap::new();
        f3.insert("handle".into(), Value::String("h1".into()));
        assert!(
            matches!(
                db.create("Account", f3),
                Err(EngineError::UniqueViolation { .. })
            ),
            "the sibling unique value must remain claimed"
        );
    }

    #[test]
    fn unique_constraint_within_create_batch() {
        // Rows in ONE create_batch carrying the same @unique value must be
        // rejected wherever the collision sits. Regression: before the
        // staged-set fix, a buffered unique-index put was invisible to a later
        // row's storage read, so the dup rows committed silently (only
        // collisions vs ALREADY-committed rows were caught).
        let row = |name: &str, email: &str| {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(name.into()));
            f.insert("email".into(), Value::String(email.into()));
            f
        };

        // The duplicate's position inside the batch must not matter: adjacent,
        // non-adjacent, and a triple all have to fail. Each runs in a fresh db
        // so the atomicity assertion below is independent.
        let layouts: Vec<Vec<FieldMap>> = vec![
            // adjacent (rows 0,1)
            vec![row("A", "dup@x.com"), row("B", "dup@x.com")],
            // non-adjacent (rows 0,2; row 1 distinct)
            vec![row("A", "dup@x.com"), row("B", "mid@x.com"), row("C", "dup@x.com")],
            // triple identical
            vec![row("A", "dup@x.com"), row("B", "dup@x.com"), row("C", "dup@x.com")],
        ];
        for layout in layouts {
            let dir = tempfile::tempdir().unwrap();
            let db = Database::open(test_schema(), dir.path()).unwrap();
            let result = db.create_batch("User", layout);
            assert!(
                matches!(result, Err(EngineError::UniqueViolation { .. })),
                "a duplicate @unique value anywhere in one batch must violate, got {result:?}"
            );
            // Atomicity: the batch failed, so NO row may have landed — not even
            // the distinct ones, otherwise the slot would now be taken.
            assert!(
                db.scan_type("User").unwrap().is_empty(),
                "a failed create_batch must leave no rows behind"
            );
            // And the value is still free: a single create with it now succeeds.
            db.create("User", row("Late", "dup@x.com")).unwrap();
        }
    }

    #[test]
    fn create_batch_rejects_dup_against_both_committed_and_staged() {
        // Both checks must fire in the same batch: a value already committed
        // (caught by the storage probe) AND an intra-batch dup (caught by the
        // staged set).
        let row = |name: &str, email: &str| {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(name.into()));
            f.insert("email".into(), Value::String(email.into()));
            f
        };
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        // Pre-commit one row.
        db.create("User", row("Alice", "alice@example.com")).unwrap();

        // A batch colliding with the committed row (storage probe path).
        assert!(matches!(
            db.create_batch("User", vec![row("Eve", "alice@example.com")]),
            Err(EngineError::UniqueViolation { .. })
        ));
        // A batch with a fresh-but-internally-duplicated value (staged path).
        assert!(matches!(
            db.create_batch("User", vec![row("Bob", "bob@example.com"), row("Bob2", "bob@example.com")]),
            Err(EngineError::UniqueViolation { .. })
        ));
        // Only Alice ever committed.
        assert_eq!(db.scan_type("User").unwrap().len(), 1);
    }

    #[test]
    fn create_batch_allows_distinct_unique_values() {
        // The staged-set check must NOT reject a batch whose @unique values are
        // all distinct.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let row = |name: &str, email: &str| {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(name.into()));
            f.insert("email".into(), Value::String(email.into()));
            f
        };

        let rows = vec![
            row("Alice", "alice@example.com"),
            row("Bob", "bob@example.com"),
            row("Carol", "carol@example.com"),
        ];
        let out = db.create_batch("User", rows).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(db.scan_type("User").unwrap().len(), 3);

        // A second batch colliding with an ALREADY-committed value still fails.
        let collide = vec![row("Dave", "alice@example.com")];
        assert!(matches!(
            db.create_batch("User", collide),
            Err(EngineError::UniqueViolation { .. })
        ));
    }

    #[test]
    fn recursive_cascade_delete() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            type Comment {
                body: String
                post: Post @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("My Post".into()));
        let post = db.create("Post", pf).unwrap();
        db.link("Post", post.id, "author", alice.id, None).unwrap();

        let mut cf = FieldMap::new();
        cf.insert("body".into(), Value::String("Great post!".into()));
        let comment = db.create("Comment", cf).unwrap();
        db.link("Comment", comment.id, "post", post.id, None)
            .unwrap();

        // Deleting alice should cascade to post, which cascades to comment.
        db.delete("User", alice.id).unwrap();

        assert!(matches!(
            db.get("User", alice.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
        assert!(matches!(
            db.get("Post", post.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
        assert!(matches!(
            db.get("Comment", comment.id),
            Err(EngineError::ObjectNotFound { .. })
        ));
    }

    #[test]
    fn object_id_recovery_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let schema = test_schema();

        let first_id;
        {
            let db = Database::open(schema.clone(), dir.path()).unwrap();
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("Alice".into()));
            let obj = db.create("User", f).unwrap();
            first_id = obj.id;
        }

        // Reopen — new objects should get IDs after the existing ones.
        {
            let db = Database::open(schema, dir.path()).unwrap();
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("Bob".into()));
            let obj = db.create("User", f).unwrap();
            assert!(
                obj.id > first_id,
                "new ID {} should be > existing ID {}",
                obj.id,
                first_id
            );

            // Original object should still exist.
            let alice = db.get("User", first_id).unwrap();
            assert_eq!(
                alice.fields.get("name"),
                Some(&Value::String("Alice".into()))
            );
        }
    }

    #[test]
    fn unique_index_cleaned_on_cascade_delete() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
            }
            type Post {
                slug: String @unique
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf = FieldMap::new();
        pf.insert("slug".into(), Value::String("hello-world".into()));
        let post = db.create("Post", pf).unwrap();
        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // Delete alice — cascades to post.
        db.delete("User", alice.id).unwrap();

        // Now creating a new post with the same slug should succeed
        // because the unique index was cleaned up.
        let mut uf2 = FieldMap::new();
        uf2.insert("name".into(), Value::String("Bob".into()));
        let bob = db.create("User", uf2).unwrap();

        let mut pf2 = FieldMap::new();
        pf2.insert("slug".into(), Value::String("hello-world".into()));
        let post2 = db.create("Post", pf2).unwrap();
        db.link("Post", post2.id, "author", bob.id, None).unwrap();

        assert_eq!(
            db.get("Post", post2.id).unwrap().fields.get("slug"),
            Some(&Value::String("hello-world".into()))
        );
    }

    #[test]
    fn subscription_receives_create_event() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let (_id, rx) = db
            .subscriptions()
            .subscribe(rhypedb_subscribe::SubscriptionFilter::for_type("User"));

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", f).unwrap();

        let event = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(event.kind, rhypedb_subscribe::ChangeKind::Create);
        assert_eq!(event.type_name, "User");
        assert_eq!(event.object_id, user.id);
    }

    #[test]
    fn subscription_receives_update_event() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", f).unwrap();

        let (_id, rx) =
            db.subscriptions()
                .subscribe(rhypedb_subscribe::SubscriptionFilter::for_object(
                    "User", user.id,
                ));

        let mut updates = FieldMap::new();
        updates.insert("name".into(), Value::String("Bob".into()));
        db.update("User", user.id, updates).unwrap();

        let event = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(event.kind, rhypedb_subscribe::ChangeKind::Update);
        assert_eq!(event.object_id, user.id);
    }

    #[test]
    fn subscription_receives_delete_event() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", f).unwrap();

        let (_id, rx) = db
            .subscriptions()
            .subscribe(rhypedb_subscribe::SubscriptionFilter::for_type("User"));
        // Drain the create event.
        let _ = rx.recv_timeout(std::time::Duration::from_secs(1));

        db.delete("User", user.id).unwrap();

        let event = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(event.kind, rhypedb_subscribe::ChangeKind::Delete);
        assert_eq!(event.object_id, user.id);
        // The Delete event now carries the deleted object's scalar fields,
        // the same payload create/update emit — a subscriber learns *which*
        // object went away, not just an opaque id.
        let fields = event
            .fields
            .expect("delete event carries the deleted object's scalar fields");
        assert_eq!(fields.get("name").and_then(|v| v.as_str()), Some("Alice"));
    }

    #[test]
    fn subscription_receives_cascade_delete_events() {
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User { name: String }
            type Post {
                title: String
                author: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let mut pf = FieldMap::new();
        pf.insert("title".into(), Value::String("Post 1".into()));
        let post = db.create("Post", pf).unwrap();
        db.link("Post", post.id, "author", alice.id, None).unwrap();

        // Subscribe to all delete events.
        let mut filter = rhypedb_subscribe::SubscriptionFilter::all();
        filter.kinds = vec![rhypedb_subscribe::ChangeKind::Delete];
        let (_id, rx) = db.subscriptions().subscribe(filter);

        db.delete("User", alice.id).unwrap();

        // Should receive delete events for both User and Post.
        let mut by_type: std::collections::HashMap<
            String,
            Option<std::collections::HashMap<String, serde_json::Value>>,
        > = std::collections::HashMap::new();
        for _ in 0..2 {
            if let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(1)) {
                by_type.insert(event.type_name.clone(), event.fields.clone());
            }
        }
        let mut deleted_types: Vec<String> = by_type.keys().cloned().collect();
        deleted_types.sort();
        assert_eq!(deleted_types, vec!["Post", "User"]);

        // The directly-deleted User carries its scalar fields...
        let user_fields = by_type
            .get("User")
            .unwrap()
            .as_ref()
            .expect("User delete carries fields");
        assert_eq!(
            user_fields.get("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
        // ...and so does the CASCADE-deleted Post (Option B: scalar-bearing
        // types get their blob read on the cascade path so the Delete event is
        // as informative as a direct delete's).
        let post_fields = by_type
            .get("Post")
            .unwrap()
            .as_ref()
            .expect("cascade-deleted Post carries fields");
        assert_eq!(
            post_fields.get("title").and_then(|v| v.as_str()),
            Some("Post 1")
        );
    }

    #[test]
    fn cascade_delete_event_for_edge_only_type_has_no_fields() {
        // An edge-only join row (only relation fields, no scalars) keeps the
        // zero-read cascade fast path: its blob is never read, so its Delete
        // event carries `fields: None`. This is the deliberate Option-B
        // boundary — there's no identifying scalar data to report anyway.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User { name: String }
            type Membership {
                user: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        let membership = db.create("Membership", FieldMap::new()).unwrap();
        db.link("Membership", membership.id, "user", alice.id, None)
            .unwrap();

        let mut filter = rhypedb_subscribe::SubscriptionFilter::all();
        filter.kinds = vec![rhypedb_subscribe::ChangeKind::Delete];
        let (_id, rx) = db.subscriptions().subscribe(filter);

        db.delete("User", alice.id).unwrap();

        let mut by_type: std::collections::HashMap<
            String,
            Option<std::collections::HashMap<String, serde_json::Value>>,
        > = std::collections::HashMap::new();
        for _ in 0..2 {
            if let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(1)) {
                by_type.insert(event.type_name.clone(), event.fields.clone());
            }
        }

        // User (scalar-bearing, direct) carries fields.
        assert_eq!(
            by_type
                .get("User")
                .unwrap()
                .as_ref()
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str()),
            Some("Alice")
        );
        // Membership (edge-only, cascade) carries NO fields — blob never read.
        assert_eq!(
            *by_type
                .get("Membership")
                .expect("Membership delete event delivered"),
            None
        );
    }

    #[test]
    fn direct_delete_event_for_edge_only_type_has_no_fields() {
        // A TOP-LEVEL delete of an edge-only type must agree with the cascade
        // case: `fields: None`. The top-level path reads the blob for the
        // existence check even though the type has no scalar fields, so the
        // capture is gated on `has_scalar` (not merely "the blob was read") to
        // avoid emitting an empty `Some({})` that would disagree with a cascade
        // delete of the same type.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User { name: String }
            type Membership {
                user: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();
        let membership = db.create("Membership", FieldMap::new()).unwrap();
        db.link("Membership", membership.id, "user", alice.id, None)
            .unwrap();

        let (_id, rx) = db
            .subscriptions()
            .subscribe(rhypedb_subscribe::SubscriptionFilter::for_type("Membership"));

        // Delete the Membership DIRECTLY (top-level, verify_exists=true).
        db.delete("Membership", membership.id).unwrap();

        let event = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(event.kind, rhypedb_subscribe::ChangeKind::Delete);
        assert_eq!(event.object_id, membership.id);
        assert_eq!(
            event.fields, None,
            "a direct delete of an edge-only type must carry no fields (not Some({{}}))"
        );
    }

    #[test]
    fn filter_scan_matches_full_scan_then_filter() {
        // 50 users with ages 1..=50. Filter "age > 30" should return 20.
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        for i in 1u32..=50 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("User{i}")));
            f.insert("email".into(), Value::String(format!("u{i}@example.com")));
            f.insert("age".into(), Value::U32(i));
            db.create("User", f).unwrap();
        }

        let gt = db
            .filter_scan("User", "age", CompareOp::Gt, 30, None)
            .unwrap();
        assert_eq!(gt.len(), 20, "age > 30 should match users with age 31..=50");
        for u in &gt {
            match u.fields.get("age") {
                Some(Value::U32(v)) => assert!(*v > 30, "stray user with age {v}"),
                other => panic!("missing/bad age: {:?}", other),
            }
        }

        let eq = db
            .filter_scan("User", "age", CompareOp::Eq, 25, None)
            .unwrap();
        assert_eq!(eq.len(), 1);
        assert!(matches!(eq[0].fields.get("age"), Some(Value::U32(25))));

        let lt = db
            .filter_scan("User", "age", CompareOp::Lt, 5, None)
            .unwrap();
        assert_eq!(lt.len(), 4, "age < 5 should match users 1..=4");
    }

    fn indexed_schema() -> Schema {
        parse_schema(
            r#"
            type Movie {
                title: String
                year: u32 @indexed
            }
            "#,
        )
        .unwrap()
    }

    /// Count the `i:` (secondary-index) entries currently visible in the LSM.
    /// Walks the global field-index prefix; used to verify create/update/
    /// delete maintain the index without leaking entries.
    fn count_index_entries(db: &Database) -> usize {
        // Field-index prefix is just `i:` — works regardless of type/field.
        let snapshot = db.storage().read_snapshot();
        let entries = db.storage().scan_prefix_at(snapshot, b"i:").unwrap();
        entries.len()
    }

    #[test]
    fn indexed_field_create_writes_index_entry() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("Alien".into()));
        f.insert("year".into(), Value::U32(1979));
        db.create("Movie", f).unwrap();

        assert_eq!(count_index_entries(&db), 1);
    }

    #[test]
    fn indexed_field_filter_scan_eq_uses_index() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        // Three at 2000, two at 1990, one at 2010.
        for _ in 0..3 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String("M".into()));
            f.insert("year".into(), Value::U32(2000));
            db.create("Movie", f).unwrap();
        }
        for _ in 0..2 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String("M".into()));
            f.insert("year".into(), Value::U32(1990));
            db.create("Movie", f).unwrap();
        }
        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("M".into()));
        f.insert("year".into(), Value::U32(2010));
        db.create("Movie", f).unwrap();

        let hits = db
            .filter_scan("Movie", "year", CompareOp::Eq, 2000, None)
            .unwrap();
        assert_eq!(hits.len(), 3);
        for h in &hits {
            assert_eq!(h.fields.get("year"), Some(&Value::U32(2000)));
        }
    }

    #[test]
    fn indexed_field_filter_scan_range_with_limit() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        for y in 1950u32..=2020 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("M{y}")));
            f.insert("year".into(), Value::U32(y));
            db.create("Movie", f).unwrap();
        }

        let gt = db
            .filter_scan("Movie", "year", CompareOp::Gt, 2010, None)
            .unwrap();
        assert_eq!(gt.len(), 10, "years 2011..=2020 should match");

        let gt_limited = db
            .filter_scan("Movie", "year", CompareOp::Gt, 2010, Some(3))
            .unwrap();
        assert_eq!(gt_limited.len(), 3);

        let lt = db
            .filter_scan("Movie", "year", CompareOp::Lt, 1955, None)
            .unwrap();
        assert_eq!(lt.len(), 5, "years 1950..=1954 should match");

        let le = db
            .filter_scan("Movie", "year", CompareOp::Le, 1952, None)
            .unwrap();
        assert_eq!(le.len(), 3, "years 1950..=1952 should match");
    }

    // -----------------------------------------------------------------
    // DateTime ordered secondary index + range pushdown (card cmqn571cn)
    // -----------------------------------------------------------------

    fn datetime_indexed_schema() -> Schema {
        parse_schema(r#"type Event { name: String  created: DateTime @indexed }"#).unwrap()
    }

    fn datetime_plain_schema() -> Schema {
        // No @indexed -> exercises the zone-map fallback path of filter_scan.
        parse_schema(r#"type Event { name: String  created: DateTime }"#).unwrap()
    }

    fn make_event(db: &Database, name: &str, ms: i64) {
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String(name.into()));
        f.insert("created".into(), Value::DateTime(ms));
        db.create("Event", f).unwrap();
    }

    #[test]
    fn indexed_datetime_create_writes_index_entry() {
        // @indexed DateTime now builds an `i:` secondary-index entry (previously
        // it silently built nothing, which is why @indexed DateTime was rejected
        // at the schema parser).
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(datetime_indexed_schema(), dir.path()).unwrap();
        make_event(&db, "a", 1000);
        assert_eq!(count_index_entries(&db), 1);
    }

    /// Run all six comparison ops against a DateTime field and return the set of
    /// matching millis (sorted). Shared by the indexed and zone-map parity test.
    fn datetime_op_results(db: &Database, target: i64) -> Vec<(rhypedb_storage::zone::CompareOp, Vec<i64>)> {
        use rhypedb_storage::zone::CompareOp;
        let ops = [
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
        ];
        ops.into_iter()
            .map(|op| {
                let mut got: Vec<i64> = db
                    .filter_scan("Event", "created", op, target, None)
                    .unwrap()
                    .iter()
                    .map(|o| match o.fields.get("created") {
                        Some(Value::DateTime(ms)) => *ms,
                        other => panic!("missing/bad created: {other:?}"),
                    })
                    .collect();
                got.sort_unstable();
                (op, got)
            })
            .collect()
    }

    #[test]
    fn indexed_datetime_ordering_including_negative_millis() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(datetime_indexed_schema(), dir.path()).unwrap();

        // Pre-epoch (negative), epoch, and post-epoch timestamps.
        for (n, ms) in [("pre", -10_000i64), ("epoch", 0), ("post", 10_000)] {
            make_event(&db, n, ms);
        }

        let results = datetime_op_results(&db, 0);
        let expect = |op: CompareOp| {
            results.iter().find(|(o, _)| *o == op).map(|(_, v)| v.clone()).unwrap()
        };
        // The MSB-flip encoding sorts negatives below positives.
        assert_eq!(expect(CompareOp::Eq), vec![0]);
        assert_eq!(expect(CompareOp::Ne), vec![-10_000, 10_000]);
        assert_eq!(expect(CompareOp::Lt), vec![-10_000]);
        assert_eq!(expect(CompareOp::Le), vec![-10_000, 0]);
        assert_eq!(expect(CompareOp::Gt), vec![10_000]);
        assert_eq!(expect(CompareOp::Ge), vec![0, 10_000]);

        // Gt against a negative target picks up epoch + post.
        let gt_neg = db
            .filter_scan("Event", "created", CompareOp::Gt, -10_000, None)
            .unwrap();
        assert_eq!(gt_neg.len(), 2);
    }

    #[test]
    fn indexed_datetime_gt_i64_max_is_empty() {
        // i64::MAX encodes to u64::MAX; the Gt seek path guards against the
        // `target_u64 + 1` overflow and returns empty.
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(datetime_indexed_schema(), dir.path()).unwrap();
        make_event(&db, "a", 0);
        make_event(&db, "b", 1_000_000);
        // With a limit (triggers the seek-then-scan path) and without.
        assert!(
            db.filter_scan("Event", "created", CompareOp::Gt, i64::MAX, Some(10))
                .unwrap()
                .is_empty()
        );
        assert!(
            db.filter_scan("Event", "created", CompareOp::Gt, i64::MAX, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn datetime_indexed_and_zone_map_paths_agree() {
        // Same dataset + same queries on an @indexed DateTime field (secondary
        // index path) and a plain DateTime field (zone-map fallback path) must
        // return identical result sets.
        let data = [-50_000i64, -1, 0, 1, 1000, 999_999, i64::MAX];
        let target = 1000;

        let dir_idx = tempfile::tempdir().unwrap();
        let db_idx = Database::open(datetime_indexed_schema(), dir_idx.path()).unwrap();
        let dir_plain = tempfile::tempdir().unwrap();
        let db_plain = Database::open(datetime_plain_schema(), dir_plain.path()).unwrap();
        for (i, ms) in data.iter().enumerate() {
            make_event(&db_idx, &format!("e{i}"), *ms);
            make_event(&db_plain, &format!("e{i}"), *ms);
        }
        // Sanity: the indexed DB actually built index entries; the plain one did not.
        assert_eq!(count_index_entries(&db_idx), data.len());
        assert_eq!(count_index_entries(&db_plain), 0);

        assert_eq!(
            datetime_op_results(&db_idx, target),
            datetime_op_results(&db_plain, target),
            "indexed and zone-map paths must agree"
        );
    }

    #[test]
    fn unique_datetime_still_works() {
        // @unique DateTime uses the plain-BE `u:` keyspace, independent of the
        // new sign-flipped `i:` index keyspace; duplicates are still rejected.
        let dir = tempfile::tempdir().unwrap();
        let schema =
            parse_schema(r#"type Event { name: String  created: DateTime @unique }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        make_event(&db, "first", 5000);
        let mut dup = FieldMap::new();
        dup.insert("name".into(), Value::String("second".into()));
        dup.insert("created".into(), Value::DateTime(5000));
        let res = db.create("Event", dup);
        assert!(
            matches!(res, Err(EngineError::UniqueViolation { .. })),
            "duplicate @unique DateTime should be rejected, got {res:?}"
        );
        // A different timestamp is fine.
        make_event(&db, "third", 6000);
    }

    #[test]
    fn indexed_datetime_update_and_delete_maintain_index() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(datetime_indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("a".into()));
        f.insert("created".into(), Value::DateTime(1000));
        let ev = db.create("Event", f).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        // Update the timestamp — the old index entry drops, a new one appears.
        let mut upd = FieldMap::new();
        upd.insert("created".into(), Value::DateTime(2000));
        db.update("Event", ev.id, upd).unwrap();
        assert_eq!(count_index_entries(&db), 1);
        assert!(
            db.filter_scan("Event", "created", CompareOp::Eq, 1000, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            db.filter_scan("Event", "created", CompareOp::Eq, 2000, None)
                .unwrap()
                .len(),
            1
        );

        // Delete removes the index entry.
        db.delete("Event", ev.id).unwrap();
        assert_eq!(count_index_entries(&db), 0);
    }

    #[test]
    fn indexed_field_update_updates_index_entry() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("M".into()));
        f.insert("year".into(), Value::U32(1979));
        let movie = db.create("Movie", f).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        // Update the year — old idx entry drops, new one appears.
        let mut upd = FieldMap::new();
        upd.insert("year".into(), Value::U32(1986));
        db.update("Movie", movie.id, upd).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        let old = db
            .filter_scan("Movie", "year", CompareOp::Eq, 1979, None)
            .unwrap();
        assert_eq!(old.len(), 0, "old year value should no longer be indexed");

        let new = db
            .filter_scan("Movie", "year", CompareOp::Eq, 1986, None)
            .unwrap();
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].id, movie.id);
    }

    #[test]
    fn indexed_field_delete_removes_index_entry() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("title".into(), Value::String("A".into()));
        f1.insert("year".into(), Value::U32(2000));
        let a = db.create("Movie", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("title".into(), Value::String("B".into()));
        f2.insert("year".into(), Value::U32(2000));
        db.create("Movie", f2).unwrap();

        assert_eq!(count_index_entries(&db), 2);

        db.delete("Movie", a.id).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        let hits = db
            .filter_scan("Movie", "year", CompareOp::Eq, 2000, None)
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "deleted entry must not reappear via the index"
        );
    }

    #[test]
    fn indexed_field_cascade_delete_removes_index_entries() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User { name: String }
            type Movie {
                title: String
                year: u32 @indexed
                owner: User @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", uf).unwrap();

        for y in [1979u32, 1986, 1993] {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("M{y}")));
            f.insert("year".into(), Value::U32(y));
            let m = db.create("Movie", f).unwrap();
            db.link("Movie", m.id, "owner", alice.id, None).unwrap();
        }
        assert_eq!(count_index_entries(&db), 3);

        db.delete("User", alice.id).unwrap();
        assert_eq!(
            count_index_entries(&db),
            0,
            "cascade-deleted movies must also drop their secondary-index entries"
        );

        let hits = db
            .filter_scan("Movie", "year", CompareOp::Eq, 1986, None)
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn indexed_field_batch_writes_index_entries() {
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        let rows: Vec<FieldMap> = (1950u32..1960)
            .map(|y| {
                let mut f = FieldMap::new();
                f.insert("title".into(), Value::String(format!("M{y}")));
                f.insert("year".into(), Value::U32(y));
                f
            })
            .collect();
        db.create_batch("Movie", rows).unwrap();

        assert_eq!(count_index_entries(&db), 10);

        let mid = db
            .filter_scan("Movie", "year", CompareOp::Eq, 1955, None)
            .unwrap();
        assert_eq!(mid.len(), 1);
    }

    #[test]
    fn indexed_field_covering_returns_full_fieldmap() {
        // Covering index: filter_scan should return Movies with their full
        // FieldMap (title + year) populated from the index entry value,
        // without doing a per-id get_many probe.
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        for y in 2010u32..=2015 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("Movie of {y}")));
            f.insert("year".into(), Value::U32(y));
            db.create("Movie", f).unwrap();
        }

        let hits = db
            .filter_scan("Movie", "year", CompareOp::Eq, 2013, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].fields.get("year"), Some(&Value::U32(2013)));
        assert_eq!(
            hits[0].fields.get("title"),
            Some(&Value::String("Movie of 2013".into())),
            "covering index must surface the title field, not just the indexed value"
        );

        let range = db
            .filter_scan("Movie", "year", CompareOp::Gt, 2012, Some(3))
            .unwrap();
        assert_eq!(range.len(), 3);
        for h in &range {
            assert!(h.fields.contains_key("title"));
            assert!(h.fields.contains_key("year"));
        }
    }

    #[test]
    fn indexed_field_covering_value_refreshes_on_non_indexed_update() {
        // Covering value is the full FieldMap. An update to a NON-indexed
        // field (title) must also rewrite the covering value, otherwise the
        // filter_scan reads back stale data.
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("title".into(), Value::String("Old Title".into()));
        f.insert("year".into(), Value::U32(1999));
        let movie = db.create("Movie", f).unwrap();

        // Update the title; year stays the same.
        let mut upd = FieldMap::new();
        upd.insert("title".into(), Value::String("New Title".into()));
        db.update("Movie", movie.id, upd).unwrap();

        // filter_scan via index should see the NEW title, not the stale Old Title.
        let hits = db
            .filter_scan("Movie", "year", CompareOp::Eq, 1999, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].fields.get("title"),
            Some(&Value::String("New Title".into())),
            "covering value must be refreshed even when only a non-indexed field changed"
        );
    }

    #[test]
    fn indexed_field_correct_after_flush() {
        // SST + memtable path: half the data is on disk, half is in memory.
        // Both layers must contribute index entries to the prefix scan.
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(indexed_schema(), dir.path()).unwrap();

        for y in 1950u32..1960 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("M{y}")));
            f.insert("year".into(), Value::U32(y));
            db.create("Movie", f).unwrap();
        }
        db.storage().flush().unwrap();

        for y in 1960u32..1970 {
            let mut f = FieldMap::new();
            f.insert("title".into(), Value::String(format!("M{y}")));
            f.insert("year".into(), Value::U32(y));
            db.create("Movie", f).unwrap();
        }

        let gt = db
            .filter_scan("Movie", "year", CompareOp::Gt, 1954, None)
            .unwrap();
        assert_eq!(
            gt.len(),
            15,
            "should include 1955..=1969 across both flushed SST and active memtable"
        );
    }

    fn string_indexed_schema() -> Schema {
        parse_schema(
            r#"
            type User {
                name: String @indexed
                bio: String
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn string_indexed_create_writes_index_entry() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        assert_eq!(count_index_entries(&db), 0);

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("bio".into(), Value::String("hi".into()));
        db.create("User", f).unwrap();

        assert_eq!(count_index_entries(&db), 1);
    }

    #[test]
    fn string_indexed_filter_scan_eq_uses_index() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        for n in &["Alice", "Bob", "Carol", "Alice", "Dave"] {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String((*n).into()));
            db.create("User", f).unwrap();
        }

        let hits = db
            .filter_scan_str("User", "name", CompareOp::Eq, "Alice", None)
            .unwrap();
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert_eq!(h.fields.get("name"), Some(&Value::String("Alice".into())));
        }
    }

    #[test]
    fn string_indexed_terminator_disambiguates_prefix() {
        // "ab"'s value-prefix must NOT match keys for "abc" — the embedded
        // \x00\x00 terminator is what makes the prefix unambiguous.
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        for n in &["ab", "abc", "abcd", "b"] {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String((*n).into()));
            db.create("User", f).unwrap();
        }

        let ab = db
            .filter_scan_str("User", "name", CompareOp::Eq, "ab", None)
            .unwrap();
        assert_eq!(ab.len(), 1);
        assert_eq!(ab[0].fields.get("name"), Some(&Value::String("ab".into())));
    }

    #[test]
    fn string_indexed_values_with_embedded_nul_roundtrip() {
        // The escape rule (0x00 -> 0x00 0x01) is what keeps embedded NUL
        // values distinct from the terminator. Verify both Eq lookup and
        // sort order with embedded NULs.
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        let with_nul: String = "ab\0c".into();
        let without_nul: String = "ab".into();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String(with_nul.clone()));
        db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String(without_nul.clone()));
        db.create("User", f2).unwrap();

        let hit = db
            .filter_scan_str("User", "name", CompareOp::Eq, &with_nul, None)
            .unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(
            hit[0].fields.get("name"),
            Some(&Value::String(with_nul.clone()))
        );

        // "ab" alone must NOT match the "ab\0c" entry.
        let hit_short = db
            .filter_scan_str("User", "name", CompareOp::Eq, "ab", None)
            .unwrap();
        assert_eq!(hit_short.len(), 1);
        assert_eq!(
            hit_short[0].fields.get("name"),
            Some(&Value::String(without_nul))
        );
    }

    #[test]
    fn string_indexed_filter_scan_range() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        for n in &["alice", "bob", "carol", "dave", "eve"] {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String((*n).into()));
            db.create("User", f).unwrap();
        }

        let lt = db
            .filter_scan_str("User", "name", CompareOp::Lt, "carol", None)
            .unwrap();
        let mut lt_names: Vec<_> = lt
            .iter()
            .filter_map(|o| match o.fields.get("name") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        lt_names.sort();
        assert_eq!(lt_names, vec!["alice".to_string(), "bob".into()]);

        let ge = db
            .filter_scan_str("User", "name", CompareOp::Ge, "carol", None)
            .unwrap();
        assert_eq!(ge.len(), 3);

        let ne = db
            .filter_scan_str("User", "name", CompareOp::Ne, "carol", None)
            .unwrap();
        assert_eq!(ne.len(), 4);

        let lt_limited = db
            .filter_scan_str("User", "name", CompareOp::Lt, "carol", Some(1))
            .unwrap();
        assert_eq!(lt_limited.len(), 1);
    }

    #[test]
    fn string_indexed_update_updates_index_entry() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", f).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Bob".into()));
        db.update("User", user.id, upd).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        let old = db
            .filter_scan_str("User", "name", CompareOp::Eq, "Alice", None)
            .unwrap();
        assert_eq!(old.len(), 0);

        let new = db
            .filter_scan_str("User", "name", CompareOp::Eq, "Bob", None)
            .unwrap();
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].id, user.id);
    }

    #[test]
    fn string_indexed_delete_removes_index_entry() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        let mut f1 = FieldMap::new();
        f1.insert("name".into(), Value::String("Alice".into()));
        let a = db.create("User", f1).unwrap();

        let mut f2 = FieldMap::new();
        f2.insert("name".into(), Value::String("Alice".into()));
        db.create("User", f2).unwrap();
        assert_eq!(count_index_entries(&db), 2);

        db.delete("User", a.id).unwrap();
        assert_eq!(count_index_entries(&db), 1);

        let hits = db
            .filter_scan_str("User", "name", CompareOp::Eq, "Alice", None)
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn string_indexed_covering_returns_full_fieldmap() {
        // Covering: the filter_scan result must include all the source
        // object's scalar fields, not just the indexed column.
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(string_indexed_schema(), dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        f.insert("bio".into(), Value::String("hello world".into()));
        db.create("User", f).unwrap();

        let hits = db
            .filter_scan_str("User", "name", CompareOp::Eq, "Alice", None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].fields.get("bio"),
            Some(&Value::String("hello world".into())),
            "covering payload must include non-indexed scalars"
        );
    }

    fn multi_indexed_schema() -> Schema {
        // One type with every indexable scalar type so the dispatch code
        // gets exercised across all encoders in one place.
        parse_schema(
            r#"
            type Item {
                name: String @indexed
                active: Bool @indexed
                rating: f32 @indexed
                weight: f64 @indexed
                hash: Bytes @indexed
                age: u32 @indexed
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn bool_indexed_eq_uses_index() {
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(multi_indexed_schema(), dir.path()).unwrap();

        for active in [true, true, false, true, false] {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("x".into()));
            f.insert("active".into(), Value::Bool(active));
            f.insert("rating".into(), Value::F32(0.0));
            f.insert("weight".into(), Value::F64(0.0));
            f.insert("hash".into(), Value::Bytes(bytes::Bytes::from_static(b"")));
            f.insert("age".into(), Value::U32(0));
            db.create("Item", f).unwrap();
        }

        let actives = db
            .filter_scan_bool("Item", "active", CompareOp::Eq, true, None)
            .unwrap();
        assert_eq!(actives.len(), 3);
        for h in &actives {
            assert_eq!(h.fields.get("active"), Some(&Value::Bool(true)));
        }

        let inactives = db
            .filter_scan_bool("Item", "active", CompareOp::Eq, false, None)
            .unwrap();
        assert_eq!(inactives.len(), 2);
    }

    #[test]
    fn float_indexed_range_preserves_order() {
        // The sortable-float encoding must keep numeric order across the
        // sign boundary (negatives sort below positives) AND within each
        // sign (smaller magnitudes closer to zero).
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(multi_indexed_schema(), dir.path()).unwrap();

        for r in [-3.5f32, -0.5, 0.0, 0.25, 1.5, 4.0] {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("x".into()));
            f.insert("active".into(), Value::Bool(false));
            f.insert("rating".into(), Value::F32(r));
            f.insert("weight".into(), Value::F64(0.0));
            f.insert("hash".into(), Value::Bytes(bytes::Bytes::from_static(b"")));
            f.insert("age".into(), Value::U32(0));
            db.create("Item", f).unwrap();
        }

        let lt0 = db
            .filter_scan_float("Item", "rating", CompareOp::Lt, 0.0, None)
            .unwrap();
        assert_eq!(lt0.len(), 2, "values < 0.0 should match -3.5 and -0.5");

        let ge_half = db
            .filter_scan_float("Item", "rating", CompareOp::Ge, 0.5, None)
            .unwrap();
        assert_eq!(ge_half.len(), 2, "values >= 0.5 should match 1.5 and 4.0");

        let eq_zero = db
            .filter_scan_float("Item", "rating", CompareOp::Eq, 0.0, None)
            .unwrap();
        assert_eq!(eq_zero.len(), 1);
    }

    #[test]
    fn bytes_indexed_eq_and_range() {
        // Bytes uses the same variable-length escape encoding as String,
        // including round-trip through 0x00 bytes.
        use rhypedb_storage::zone::CompareOp;
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(multi_indexed_schema(), dir.path()).unwrap();

        let payloads: &[&[u8]] = &[b"\x00ab", b"abc", b"abcd", b"abc\x00xyz", b"zzz"];
        for &p in payloads {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String("x".into()));
            f.insert("active".into(), Value::Bool(false));
            f.insert("rating".into(), Value::F32(0.0));
            f.insert("weight".into(), Value::F64(0.0));
            f.insert(
                "hash".into(),
                Value::Bytes(bytes::Bytes::copy_from_slice(p)),
            );
            f.insert("age".into(), Value::U32(0));
            db.create("Item", f).unwrap();
        }

        // "abc" Eq should match only the exact "abc" payload — not "abcd"
        // and not "abc\x00xyz".
        let hits = db
            .filter_scan_bytes("Item", "hash", CompareOp::Eq, b"abc", None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].fields.get("hash"),
            Some(&Value::Bytes(bytes::Bytes::from_static(b"abc")))
        );

        // "abc" with embedded NUL should round-trip via Eq.
        let with_nul = db
            .filter_scan_bytes("Item", "hash", CompareOp::Eq, b"abc\x00xyz", None)
            .unwrap();
        assert_eq!(with_nul.len(), 1);

        // Range: < "abc" should hit only "\x00ab".
        let lt = db
            .filter_scan_bytes("Item", "hash", CompareOp::Lt, b"abc", None)
            .unwrap();
        assert_eq!(lt.len(), 1);
    }

    fn covering_schema() -> Schema {
        // Bench-shape: Rating has two forward 1:1 relations (user, movie).
        // Inverse fields on User/Movie let `get_links_many` surface the
        // covering reverse-edge values directly so a test can inspect them.
        parse_schema(
            r#"
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }

            type Movie {
                title: String
                ratings: [Rating] @inverse(Rating.movie)
            }

            type Rating {
                stars: u32
                user: User
                movie: Movie
            }
            "#,
        )
        .unwrap()
    }

    /// Set up one User u, one Movie m, one Rating r linked to both.
    /// Returns (db, user_id, movie_id, rating_id, tempdir).
    fn build_one_rating(schema: Schema) -> (Arc<Database>, u64, u64, u64, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();

        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        let rating = db.create("Rating", rf).unwrap();

        // Link order matters: first link writes empty rev_edge cover,
        // second link writes the cover with the other-target's data.
        db.link("Rating", rating.id, "user", user.id, None).unwrap();
        db.link("Rating", rating.id, "movie", movie.id, None)
            .unwrap();

        (db, user.id, movie.id, rating.id, dir)
    }

    #[test]
    fn inline_relations_at_create_skip_separate_link_calls() {
        // The whole point of accepting relation fields in `create`: writing
        // Rating + its forward edges to User and Movie happens in ONE txn,
        // and `get_links` round-trips show the edges landed.
        let (db, uid, mid, rid, _dir) = {
            let dir = tempfile::tempdir().unwrap();
            let db = Database::open(covering_schema(), dir.path()).unwrap();
            let mut uf = FieldMap::new();
            uf.insert("name".into(), Value::String("Alice".into()));
            let user = db.create("User", uf).unwrap();
            let mut mf = FieldMap::new();
            mf.insert("title".into(), Value::String("Aliens".into()));
            let movie = db.create("Movie", mf).unwrap();
            let mut rf = FieldMap::new();
            rf.insert("stars".into(), Value::U32(5));
            rf.insert("user".into(), Value::U64(user.id));
            rf.insert("movie".into(), Value::U64(movie.id));
            let rating = db.create("Rating", rf).unwrap();
            (db, user.id, movie.id, rating.id, dir)
        };

        // Rating object only carries scalars (relations went to edge index).
        let fetched = db.get("Rating", rid).unwrap();
        assert_eq!(fetched.fields.get("stars"), Some(&Value::U32(5)));
        assert!(!fetched.fields.contains_key("user"));
        assert!(!fetched.fields.contains_key("movie"));

        // Forward edges visible to get_links.
        let user_links = db.get_links("Rating", rid, "user").unwrap();
        assert_eq!(user_links.len(), 1);
        assert_eq!(user_links[0].0, uid);

        let movie_links = db.get_links("Rating", rid, "movie").unwrap();
        assert_eq!(movie_links.len(), 1);
        assert_eq!(movie_links[0].0, mid);
    }

    #[test]
    fn inline_relations_yield_symmetric_covers() {
        // Sequential link() writes asymmetric covers: only the second link
        // gets a `<peer>__cover` blob. Inline-relations at create build
        // covers from in-memory state, so BOTH rev_edges land with full
        // peer covers. Test verifies the user-side rev_edge now carries
        // `movie__cover` (which the historical flow left empty).
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(covering_schema(), dir.path()).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        let user_side = db.get_links_many("User", &[user.id], "ratings").unwrap();
        assert_eq!(user_side[0].len(), 1);
        let (_rid, user_side_cover) = &user_side[0][0];
        // movie__cover should be present in user-side rev_edge cover.
        let movie_cover = find_bytes_field_in_raw(user_side_cover, "movie__cover").expect(
            "inline-relations create should write symmetric covers — \
                 movie__cover missing from user-side rev_edge",
        );
        let movie_fields = deserialize_fields(&movie_cover);
        assert_eq!(
            movie_fields.get("title"),
            Some(&Value::String("Aliens".into()))
        );

        // Movie-side rev_edge should also carry user__cover.
        let movie_side = db.get_links_many("Movie", &[movie.id], "ratings").unwrap();
        let (_rid, movie_side_cover) = &movie_side[0][0];
        let user_cover = find_bytes_field_in_raw(movie_side_cover, "user__cover")
            .expect("user__cover missing from movie-side rev_edge");
        let user_fields = deserialize_fields(&user_cover);
        assert_eq!(
            user_fields.get("name"),
            Some(&Value::String("Alice".into()))
        );
    }

    #[test]
    fn inline_relations_reject_inverse_field() {
        // Inverse fields are virtual — setting one at create time would
        // have no edge index to land in. Engine must reject before any
        // mutation.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(covering_schema(), dir.path()).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        // User.ratings is @inverse(Rating.user) — virtual.
        uf.insert("ratings".into(), Value::U64(42));
        let result = db.create("User", uf);
        assert!(matches!(result, Err(EngineError::TypeMismatch { .. })));
    }

    #[test]
    fn inline_relations_reject_missing_target() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(covering_schema(), dir.path()).unwrap();
        // Reference a User that doesn't exist.
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(99_999));
        let result = db.create("Rating", rf);
        assert!(matches!(result, Err(EngineError::ObjectNotFound { .. })));
    }

    #[test]
    fn cascade_extracts_peer_targets_from_cover_blob() {
        // Cover-extract optimization: when cascading from User → Rating,
        // the parent's inbound scan returns the rev_edge value which
        // (with symmetric covers from inline-relations) embeds the
        // movie target id. The recursive delete extracts it without a
        // forward-edge `scan_prefix` and tombstones the movie-side rev
        // edge. Test verifies the Movie's rev_edge for the Rating is
        // gone after the User cascade — proves the cover-extract path
        // staged the right keys.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(
            r#"
            type User {
                name: String
                email: String @unique
                ratings: [Rating] @inverse(Rating.user)
            }
            type Movie {
                title: String
                ratings: [Rating] @inverse(Rating.movie)
            }
            type Rating {
                stars: u32
                user: User @on_delete(cascade)
                movie: Movie @on_delete(cascade)
            }
            "#,
        )
        .unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        uf.insert("email".into(), Value::String("alice@x".into()));
        let user = db.create("User", uf).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        let rating = db.create("Rating", rf).unwrap();

        // Sanity: Movie has the Rating in its rev_edge index.
        let before = db.get_links_many("Movie", &[movie.id], "ratings").unwrap();
        assert_eq!(before[0].len(), 1, "rating should be linked to movie");

        // Cascade-delete the User. Rating cascades. The Movie's rev_edge
        // entry for the rating should be tombstoned via the cover-extract
        // path (not via an outbound scan, but the bench observable is the
        // same: gone).
        db.delete("User", user.id).unwrap();

        let after = db.get_links_many("Movie", &[movie.id], "ratings").unwrap();
        assert!(
            after[0].is_empty(),
            "Movie's rev_edge to the cascaded Rating must be tombstoned \
             after the User cascade (cover-extract path)"
        );
        // Rating itself must be gone.
        assert!(db.get("Rating", rating.id).is_err());
    }

    #[test]
    fn create_batch_inline_relations_lands_all_edges() {
        // create_batch with inline relations: 3 ratings in one txn each
        // linking to the same user + movie; verify each rating's forward
        // edges resolve correctly via get_links.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(covering_schema(), dir.path()).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();

        let rows: Vec<FieldMap> = (1u32..=3)
            .map(|s| {
                let mut r = FieldMap::new();
                r.insert("stars".into(), Value::U32(s));
                r.insert("user".into(), Value::U64(user.id));
                r.insert("movie".into(), Value::U64(movie.id));
                r
            })
            .collect();
        let ratings = db.create_batch("Rating", rows).unwrap();
        assert_eq!(ratings.len(), 3);

        // Three rev_edges hanging off the movie.
        let movie_side = db.get_links_many("Movie", &[movie.id], "ratings").unwrap();
        assert_eq!(movie_side[0].len(), 3);

        // Each rating's forward edge to the user.
        for r in &ratings {
            let user_links = db.get_links("Rating", r.id, "user").unwrap();
            assert_eq!(user_links.len(), 1);
            assert_eq!(user_links[0].0, user.id);
        }
    }

    #[test]
    fn covered_rev_edge_refreshes_after_source_field_update() {
        // Phase 1 staleness: the rev_edge stored on Movie's side for this
        // Rating carries the Rating's effective fields verbatim (stars=5).
        // Updating Rating.stars must rewrite that rev_edge value, or a
        // downstream consumer reading the source covering sees stale data.
        let (db, _uid, mid, rid, _dir) = build_one_rating(covering_schema());

        let mut upd = FieldMap::new();
        upd.insert("stars".into(), Value::U32(1));
        db.update("Rating", rid, upd).unwrap();

        let groups = db.get_links_many("Movie", &[mid], "ratings").unwrap();
        let group = &groups[0];
        assert_eq!(group.len(), 1);
        let (got_rid, cover) = &group[0];
        assert_eq!(*got_rid, rid);

        // Decode the rev_edge value (it's a serialized FieldMap carrying the
        // Rating's effective fields) and assert stars reflects the update.
        let cover_fields = deserialize_fields(cover);
        assert_eq!(
            cover_fields.get("stars"),
            Some(&Value::U32(1)),
            "Phase 1: source object's update must propagate to its own \
             covering rev_edge values"
        );
    }

    /// Open the covering-schema database with the background cover-refresh
    /// sweeper disabled. Used by tests that need to observe the cover_v
    /// mismatch BEFORE the sweeper would otherwise repair it.
    fn build_one_rating_no_sweeper(
        schema: Schema,
    ) -> (Arc<Database>, u64, u64, u64, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open_with_options(
            schema,
            dir.path(),
            OpenOptions {
                background_cover_refresh: false,
                ..Default::default()
            },
        )
        .unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Aliens".into()));
        let movie = db.create("Movie", mf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        let rating = db.create("Rating", rf).unwrap();
        db.link("Rating", rating.id, "user", user.id, None).unwrap();
        db.link("Rating", rating.id, "movie", movie.id, None)
            .unwrap();
        (db, user.id, movie.id, rating.id, dir)
    }

    #[test]
    fn second_degree_cover_staleness_is_detectable_via_versions() {
        // Phase 2 invalidation-tombstone design: User.update does NOT
        // synchronously rewrite the rev_edge values that embedded the user
        // under `user__cover`. Instead it bumps the per-object generation
        // counter; cover writers stamp the target's version into
        // `<name>__cover_v`. Readers compare against the live counter and
        // fall through to a fresh LSM probe when they disagree — bounded
        // write cost regardless of fan-in.
        //
        // (Phase 3, the background cover-refresh sweeper, ASYNCHRONOUSLY
        // repairs the stale covers later. This test runs with the sweeper
        // disabled so the post-update inspection is deterministic.)
        let (db, uid, mid, _rid, _dir) = build_one_rating_no_sweeper(covering_schema());

        assert_eq!(
            db.object_version("User", uid),
            1,
            "born-at-1: a freshly-created object has generation 1 (0 is reserved for absent)"
        );

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", uid, upd).unwrap();

        assert_eq!(
            db.object_version("User", uid),
            2,
            "successful update must bump the per-object generation (1 -> 2)"
        );

        let groups = db.get_links_many("Movie", &[mid], "ratings").unwrap();
        let group = &groups[0];
        assert_eq!(group.len(), 1);
        let (_rid, cover) = &group[0];

        // The cover bytes still contain the OLD user data (we did not
        // rewrite the rev_edge — that's the whole point of the tombstone).
        let user_cover_bytes = find_bytes_field_in_raw(cover, "user__cover")
            .expect("user__cover should be embedded in the rev_edge");
        let user_fields = deserialize_fields(&user_cover_bytes);
        assert_eq!(
            user_fields.get("name"),
            Some(&Value::String("Alice".into())),
            "rev_edge value is not rewritten on a Phase 2 source-target update"
        );

        // …but the stamped `user__cover_v` is below the live counter,
        // which is exactly the signal the executor uses to fall through.
        let stamped_v = crate::object::find_u64_field_in_raw(cover, "user__cover_v")
            .expect("cover_v stamp should be present");
        assert_eq!(
            stamped_v, 1,
            "stamp records the target's generation (born-at-1) as of cover-write time"
        );
        assert!(
            stamped_v < db.object_version("User", uid),
            "stamp < live counter triggers reader fall-through"
        );
    }

    /// Helper: poll the `r:<movie>:movie:<rating>` rev_edge value and read
    /// the embedded `user__cover_v` stamp. Returns the stamp or `None` if
    /// the rev_edge isn't present or doesn't carry a cover.
    fn read_movie_side_user_cover_v(db: &Database, movie_id: u64) -> Option<u64> {
        let groups = db.get_links_many("Movie", &[movie_id], "ratings").ok()?;
        let group = groups.into_iter().next()?;
        let (_rid, cover) = group.into_iter().next()?;
        crate::object::find_u64_field_in_raw(&cover, "user__cover_v")
    }

    fn read_movie_side_user_name(db: &Database, movie_id: u64) -> Option<String> {
        let groups = db.get_links_many("Movie", &[movie_id], "ratings").ok()?;
        let group = groups.into_iter().next()?;
        let (_rid, cover) = group.into_iter().next()?;
        let inner = crate::object::find_bytes_field_in_raw(&cover, "user__cover")?;
        let fields = deserialize_fields(&inner);
        match fields.get("name")? {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Poll predicate up to `attempts` × 10ms. Returns true if the predicate
    /// held within the window. Used by sweeper tests to wait for the
    /// background thread to repair a stale cover without sleeping for a
    /// fixed (potentially flaky) duration.
    fn poll_until(mut p: impl FnMut() -> bool, attempts: u32) -> bool {
        for _ in 0..attempts {
            if p() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        p()
    }

    #[test]
    fn cover_refresh_worker_repairs_stale_user_cover() {
        // End-to-end check that the background sweeper rewrites the
        // movie-side rev_edge `user__cover` after the user updates. Verifies:
        //   1. Pre-update: cover_v stamp == 1 (born-at-1, user never updated).
        //   2. Post-update + post-sweep: cover_v stamp matches the new
        //      live generation AND the embedded cover bytes reflect the new
        //      user.name.
        let (db, uid, mid, _rid, _dir) = build_one_rating(covering_schema());

        let stamp_before =
            read_movie_side_user_cover_v(&db, mid).expect("cover_v should be present pre-update");
        assert_eq!(stamp_before, 1);

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", uid, upd).unwrap();
        let new_v = db.object_version("User", uid);
        assert_eq!(new_v, 2);

        // Sweeper runs asynchronously; poll until repair lands or fail.
        let repaired = poll_until(
            || read_movie_side_user_cover_v(&db, mid) == Some(new_v),
            200, // ≤ 2 seconds; sweeper should take micros
        );
        assert!(
            repaired,
            "cover-refresh worker did not stamp the new cover_v within 2s"
        );

        assert_eq!(
            read_movie_side_user_name(&db, mid).as_deref(),
            Some("Renamed"),
            "embedded user__cover bytes must contain the post-update name"
        );
    }

    #[test]
    fn cover_refresh_idempotent_under_repeated_bumps() {
        // Multiple updates in quick succession enqueue multiple sweeps.
        // Each sweep is independently safe: the final rev_edge state still
        // reflects the LAST update.
        let (db, uid, mid, _rid, _dir) = build_one_rating(covering_schema());

        for new_name in ["B", "C", "D", "E"] {
            let mut upd = FieldMap::new();
            upd.insert("name".into(), Value::String(new_name.into()));
            db.update("User", uid, upd).unwrap();
        }
        let final_v = db.object_version("User", uid);

        let landed = poll_until(
            || {
                read_movie_side_user_cover_v(&db, mid) == Some(final_v)
                    && read_movie_side_user_name(&db, mid).as_deref() == Some("E")
            },
            200,
        );
        assert!(landed, "sweeper did not converge on the final state");
    }

    #[test]
    fn cover_refresh_worker_disabled_leaves_cover_stale() {
        // With the background sweeper opted out, an update never rewrites
        // the embedded cover — the OnlyHand path is the reader's cover_v
        // fall-through. This guards against the sweeper being silently
        // re-enabled by future refactors.
        let (db, uid, mid, _rid, _dir) = build_one_rating_no_sweeper(covering_schema());

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", uid, upd).unwrap();

        // Give a real sweeper a generous window to PROVE it isn't running.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let stamp = read_movie_side_user_cover_v(&db, mid).unwrap();
        assert_eq!(
            stamp, 1,
            "no sweeper means cover_v stays at the born-at-1 value stamped at link time"
        );
        assert_eq!(
            read_movie_side_user_name(&db, mid).as_deref(),
            Some("Alice"),
            "no sweeper means embedded user.name must stay at the pre-update value"
        );
    }

    /// Schema with a chained 1:1 forward relation: `Rating.movie` → Movie,
    /// `Movie.director` → Director. Used by the 3-hop covering tests below
    /// to verify the recursive cover embed.
    fn three_hop_schema() -> Schema {
        parse_schema(
            r#"
            type Director {
                name: String
            }
            type Movie {
                title: String
                director: Director
                ratings: [Rating] @inverse(Rating.movie)
            }
            type User {
                name: String
                ratings: [Rating] @inverse(Rating.user)
            }
            type Rating {
                stars: u32
                user: User
                movie: Movie
            }
            "#,
        )
        .unwrap()
    }

    /// Verify the cover writer embeds the 3rd-degree target's data inside
    /// the 2nd-degree cover blob. Inspecting the rev_edge bytes directly:
    /// `r:user:rel:rating` value should carry `movie__cover` which itself
    /// is a serialized FieldMap containing `director: U64(...)` and
    /// `director__cover: Bytes(...)` with the director's data.
    #[test]
    fn three_hop_cover_embeds_director_inside_movie_cover() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(three_hop_schema(), dir.path()).unwrap();

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();

        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        mf.insert("director".into(), Value::U64(director.id));
        let movie = db.create("Movie", mf).unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();

        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        // Walk the user-side rev_edge → extract the movie_cover →
        // extract director from inside that cover.
        let groups = db.get_links_many("User", &[user.id], "ratings").unwrap();
        let (_rid, rating_cover) = &groups[0][0];
        let movie_cover_bytes =
            crate::object::find_bytes_field_in_raw(rating_cover, "movie__cover")
                .expect("rating rev_edge should embed movie__cover");
        let director_in_movie =
            crate::object::find_u64_field_in_raw(&movie_cover_bytes, "director")
                .expect("movie__cover should carry director id (3-hop)");
        assert_eq!(director_in_movie, director.id);
        let director_cover_bytes =
            crate::object::find_bytes_field_in_raw(&movie_cover_bytes, "director__cover")
                .expect("movie__cover should carry director__cover (3-hop)");
        let director_fields = deserialize_fields(&director_cover_bytes);
        assert_eq!(
            director_fields.get("name"),
            Some(&Value::String("Scott".into()))
        );
        let stamp = crate::object::find_u64_field_in_raw(&movie_cover_bytes, "director__cover_v")
            .expect("movie__cover should carry director__cover_v stamp");
        assert_eq!(stamp, 1, "born-at-1: a freshly-created director has generation 1");
    }

    #[test]
    fn three_hop_cover_omitted_when_target_has_no_forward_1to1() {
        // The Movie-side rev_edge's `user__cover` should NOT carry any
        // <next>__cover — User has no forward 1:1 relations to embed
        // (only the @inverse `ratings`). Pre-check in
        // `with_nested_forward_covers` should short-circuit and return
        // the original target bytes unchanged.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(three_hop_schema(), dir.path()).unwrap();

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        mf.insert("director".into(), Value::U64(director.id));
        let movie = db.create("Movie", mf).unwrap();
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        rf.insert("user".into(), Value::U64(user.id));
        rf.insert("movie".into(), Value::U64(movie.id));
        db.create("Rating", rf).unwrap();

        // The Movie-side rev_edge contains user__cover. User has no
        // forward 1:1 — that cover should be User's bare scalars.
        let groups = db.get_links_many("Movie", &[movie.id], "ratings").unwrap();
        let (_rid, rating_cover) = &groups[0][0];
        let user_cover_bytes = crate::object::find_bytes_field_in_raw(rating_cover, "user__cover")
            .expect("movie-side rev_edge should embed user__cover");
        let user_fields = deserialize_fields(&user_cover_bytes);
        assert_eq!(
            user_fields.get("name"),
            Some(&Value::String("Alice".into()))
        );
        // No nested cover fields should appear since User has no outgoing 1:1.
        assert!(
            !user_fields
                .keys()
                .any(|k| k.ends_with("__cover") || k.ends_with("__cover_v")),
            "user__cover should not embed any nested __cover entries"
        );
    }

    #[test]
    fn three_hop_cover_picked_up_after_separate_link_call() {
        // Inline-relations create above goes through build_inflight_cover.
        // The legacy `create + link` flow goes through build_covering_rev_value.
        // Verify the 3-hop nesting works for that path too. Link order
        // matters: the first link sees no peers and writes an empty
        // rev_edge; the second link embeds the first link's target. So
        // we link movie FIRST and user SECOND — the user-side rev_edge
        // (the second one written) ends up with `movie__cover`, which
        // is where 3-hop embeds `director` + `director__cover`.
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(three_hop_schema(), dir.path()).unwrap();

        let mut df = FieldMap::new();
        df.insert("name".into(), Value::String("Scott".into()));
        let director = db.create("Director", df).unwrap();
        let mut mf = FieldMap::new();
        mf.insert("title".into(), Value::String("Alien".into()));
        let movie = db.create("Movie", mf).unwrap();
        db.link("Movie", movie.id, "director", director.id, None)
            .unwrap();

        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alice".into()));
        let user = db.create("User", uf).unwrap();
        let mut rf = FieldMap::new();
        rf.insert("stars".into(), Value::U32(5));
        let rating = db.create("Rating", rf).unwrap();
        db.link("Rating", rating.id, "movie", movie.id, None)
            .unwrap();
        db.link("Rating", rating.id, "user", user.id, None).unwrap();

        let groups = db.get_links_many("User", &[user.id], "ratings").unwrap();
        let (_rid, rating_cover) = &groups[0][0];
        let movie_cover_bytes =
            crate::object::find_bytes_field_in_raw(rating_cover, "movie__cover")
                .expect("rating rev_edge should embed movie__cover (legacy link path)");
        let director_in_movie =
            crate::object::find_u64_field_in_raw(&movie_cover_bytes, "director");
        assert_eq!(
            director_in_movie,
            Some(director.id),
            "3-hop embed should kick in on the link() path too"
        );
    }

    #[test]
    fn filter_scan_correct_after_flush() {
        // Force a flush mid-create so the predicate spans both memtable
        // (no zone map) and SST (zone map). Result must still be correct.
        use rhypedb_storage::zone::CompareOp;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(test_schema(), dir.path()).unwrap();

        for i in 1u32..=20 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("U{i}")));
            f.insert("email".into(), Value::String(format!("u{i}@x")));
            f.insert("age".into(), Value::U32(i));
            db.create("User", f).unwrap();
        }
        db.storage().flush().unwrap();

        // Add 20 more after the flush — these live in the memtable, with
        // no zone map, so filter_scan must still find them via the
        // memtable's full-scan fallback path.
        for i in 21u32..=40 {
            let mut f = FieldMap::new();
            f.insert("name".into(), Value::String(format!("U{i}")));
            f.insert("email".into(), Value::String(format!("u{i}@x")));
            f.insert("age".into(), Value::U32(i));
            db.create("User", f).unwrap();
        }

        let gt = db
            .filter_scan("User", "age", CompareOp::Gt, 15, None)
            .unwrap();
        assert_eq!(gt.len(), 25, "should include 16..=40");
    }

    #[test]
    fn create_writes_no_generation_key_update_does() {
        // born-at-1 lives in memory only: a create persists NO `g:` key
        // (existence is the born bit, reconstructed at open from the `o:*`
        // scan). A `g:` key appears only once an object is UPDATED.
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema(r#"type User { name: String }"#).unwrap();
        let db = Database::open(schema, dir.path()).unwrap();

        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("Alice".into()));
        let alice = db.create("User", f).unwrap();
        assert_eq!(
            db.object_version("User", alice.id),
            1,
            "born at generation 1 in memory"
        );

        let g_prefix = KeyBuilder::object_version_prefix();
        let txn = db.storage().begin_txn();
        let g_after_create = db.storage().scan_prefix(&txn, &g_prefix).unwrap();
        drop(txn);
        assert!(
            g_after_create.is_empty(),
            "create must persist no g: key, found {}",
            g_after_create.len()
        );

        // An update bumps to generation 2 AND persists exactly one g: key.
        let mut uf = FieldMap::new();
        uf.insert("name".into(), Value::String("Alicia".into()));
        db.update("User", alice.id, uf).unwrap();
        assert_eq!(db.object_version("User", alice.id), 2);

        let txn2 = db.storage().begin_txn();
        let g_after_update = db.storage().scan_prefix(&txn2, &g_prefix).unwrap();
        drop(txn2);
        assert_eq!(
            g_after_update.len(),
            1,
            "update must persist exactly one g: key"
        );
    }

    #[test]
    fn born_bit_reconstructed_from_object_scan_on_reopen() {
        // Dropping the persisted create-time `g:` key must not lose the born
        // bit across a restart: open() re-seeds generation 1 for every live
        // object from the `o:*` scan. A live never-updated object reads 1 after
        // reopen; a deleted one reads 0 (its `o:` key is gone, so never seeded).
        let dir = tempfile::tempdir().unwrap();
        let schema_text = r#"type User { name: String }"#;

        let (live_id, dead_id) = {
            let db = Database::open(parse_schema(schema_text).unwrap(), dir.path()).unwrap();
            let mut a = FieldMap::new();
            a.insert("name".into(), Value::String("Live".into()));
            let live = db.create("User", a).unwrap();
            let mut b = FieldMap::new();
            b.insert("name".into(), Value::String("Dead".into()));
            let dead = db.create("User", b).unwrap();
            db.delete("User", dead.id).unwrap();
            (live.id, dead.id)
        };

        // Reopen: version_counters rebuilt purely from disk (o:* seed, no g:).
        let db = Database::open(parse_schema(schema_text).unwrap(), dir.path()).unwrap();
        assert_eq!(
            db.object_version("User", live_id),
            1,
            "live never-updated object must reconstruct generation 1 from existence"
        );
        assert_eq!(
            db.object_version("User", dead_id),
            0,
            "deleted object must read generation 0 after reopen (o: key gone, never seeded)"
        );
    }
}
