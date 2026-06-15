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
    /// writes proceed during the migration instead of being quiesced. Nested
    /// `type_id -> {field_name -> hook}` so the hot-path probe for a
    /// non-migrating type is a single Copy-`u64` miss. Lock-free read cache
    /// (ArcSwap), mutated ONLY under `migration_lock.write()` via
    /// `arm_field_hook`/`disarm_field_hook`, rebuilt from the `c:P:` plans on
    /// open/create alongside `migrating`. **Card 2a plumbs it; writers don't
    /// consume it until card 2b** (quiesce still rejects in 2a).
    migrating_fields: arc_swap::ArcSwap<
        std::collections::HashMap<u64, std::collections::HashMap<String, Arc<MigratingFieldHook>>>,
    >,
    /// Fast-path gate: total live hook count. Producers load this (`Relaxed`)
    /// and skip ALL hook work — no lock, no map probe, no `String` — when it's
    /// `0` (the common case, no migration active). Kept in sync with
    /// `migrating_fields` under `migration_lock.write()`.
    migrating_field_count: std::sync::atomic::AtomicUsize,
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
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            sync_on_commit: true,
            background_cover_refresh: true,
            allow_schema_shrink: false,
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
}

/// Operator-facing snapshot of a persisted migration plan
/// ([`Database::list_migrations`]). Kind bytes and forward-compat TLVs are
/// deliberately not exposed.
#[derive(Debug, Clone)]
pub struct MigrationSummary {
    pub plan_id: u64,
    pub type_name: String,
    pub field_name: String,
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
        // `<field>__shadow_cv`) to callers. Gated on an active migration so a
        // non-migrating database pays one atomic load and no scan. This is the
        // single eager-read chokepoint (get/get_many/scan/filter_scan all call
        // it); the lazy/raw wire path is handled separately in card 2c.
        if self
            .migrating_field_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            fields.retain(|name, _| !is_shadow_sibling_key(name));
        }
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
        crate::catalog::apply_migration(&self.storage, &self.schema, &verbs)
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
    /// * Secondary index entries (`i:<type>:<field_id>:…`) and unique
    ///   index entries (`u:<type>:<field_id>:…`) — both keyed by
    ///   `field_id`, untouched by the rename.
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
        crate::catalog::apply_migration(&self.storage, &self.schema, &verbs)
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

        // Setup under the migration write-barrier: validate, allocate +
        // persist the plan, and arm quiesce atomically so no writer slips a
        // create/update into the migrating type between plan and quiesce.
        let created = {
            let _guard = self.migration_lock.write();
            let created = crate::catalog::create_migration_plan(
                &self.storage,
                &self.schema,
                &spec.type_name,
                &spec.field_name,
                target_kind,
                &spec.converter_name,
                spec.converter_version,
                spec.chunk_size,
            )?;
            // Install the double-write hook so live writes to the migrating
            // field carry it forward (card 2d — no quiesce; writes proceed).
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
            created
        };

        // Drive WITHOUT holding `migration_lock` so live writes (to this type
        // and others) proceed during the (potentially long) backfill; the
        // double-write hook keeps every write's shadow current. The cutover
        // re-takes `migration_lock.write()` to drain writers for its pass.
        self.drive_migration_to_completion(created.plan_id, created.type_id, Some(&converter))?;
        Ok(created.plan_id)
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
    /// reconciles every cover/index, flips the catalog kind, and releases
    /// quiesce — holding `migration_lock.write()` for the whole pass. On a
    /// converter / data / shadow error the plan is left `Failed` (quiesce HELD)
    /// and the error propagates; quiesce is released only on a clean `Completed`.
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
        // The converter is needed ONLY to backfill shadows in the Converting
        // phase. A plan resumed in the CuttingOver phase (a crash mid-cutover)
        // is a pure rename pass — don't demand a converter the operator has no
        // reason to re-register.
        if plan.phase == crate::catalog::MigrationPhase::Converting {
            let converter = converter.ok_or_else(|| EngineError::ConverterNotRegistered {
                name: plan.converter_name.clone(),
                version: plan.converter_version,
            })?;
            crate::catalog::run_migration_chunks(&self.storage, plan_id, converter)?;
        }
        self.run_cutover(plan_id, type_id)
    }

    /// Cutover pass (shadow-field card 2): promote every `<field>__shadow`
    /// sibling to the source field, reconcile ALL covers/indexes via the shared
    /// `rewrite_object_and_maintain_covers`, then flip the catalog kind and
    /// release quiesce.
    ///
    /// Holds `migration_lock.write()` for the WHOLE pass. The write-lock spans
    /// every commit, so once it is acquired no in-flight writer (card-2 inc 2d)
    /// can add a shadow racing a rename, and a reader (which needs no lock) sees
    /// each row transition source→target atomically (each row's rename is one
    /// commit). Reads stay available throughout — only writes are drained.
    ///
    /// Per-chunk commit order mirrors the backfill worker: `[promoted o: blob,
    /// i:/r:/g: cover maintenance, plan record (cutover_cursor) LAST]`, so a
    /// torn tail drops only the cursor advance and resume re-does the chunk
    /// idempotently (a row already promoted — source at target kind, no shadow —
    /// is skipped). A `WriteConflict` (the background cover-refresh worker holds
    /// no `migration_lock`) retries the chunk; the generation over-bump on a
    /// retry is harmless (monotonic staleness counter).
    fn run_cutover(&self, plan_id: u64, type_id: u64) -> EngineResult<()> {
        const WRITE_CONFLICT_RETRIES: u32 = 8;
        let _guard = self.migration_lock.write();

        let mut plan = {
            let txn = self.storage.begin_txn();
            crate::catalog::load_migration_plan(&self.storage, &txn, plan_id)?
        };
        let type_name = plan.type_name.clone();
        let field_name = plan.field_name.clone();
        let target_kind = plan.target_kind;
        let converter_version = plan.converter_version;
        let shadow_name = format!("{field_name}__shadow");
        let shadow_cv_name = format!("{field_name}__shadow_cv");

        // Durably mark CuttingOver BEFORE the first rename so a crash resumes the
        // rename pass, not the converter. Idempotent (already CuttingOver → skip).
        if plan.phase != crate::catalog::MigrationPhase::CuttingOver {
            plan.phase = crate::catalog::MigrationPhase::CuttingOver;
            let mut txn = self.storage.begin_txn();
            let (k, v) = crate::catalog::migration_plan_record(&plan);
            self.storage.put(&mut txn, &k, v)?;
            self.storage.commit(&mut txn)?;
        }

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
        Ok(())
    }

    /// Refuse to DRIVE a plan unless the open schema declares the field at the
    /// plan's TARGET kind (shadow-field card 1, blocker F3). Driving flips the
    /// catalog to the target; if this handle still validates writes against
    /// the source kind (operator reopened with the OLD schema), finishing the
    /// migration would silently corrupt. The operator must reopen with the
    /// target schema first.
    fn guard_resume_schema(&self, plan: &crate::catalog::MigrationPlan) -> EngineResult<()> {
        let want = self
            .schema
            .get_type(&plan.type_name)
            .and_then(|td| td.fields.iter().find(|f| f.name == plan.field_name))
            .map(|fd| crate::catalog::schema_kind_byte_public(&fd.field_type));
        if want != Some(plan.target_kind) {
            return Err(EngineError::MigrationResumeSchemaMismatch {
                plan_id: plan.plan_id,
                expected: crate::catalog::kind_name_public(plan.target_kind),
                found: want
                    .map(crate::catalog::kind_name_public)
                    .unwrap_or("<absent>"),
            });
        }
        Ok(())
    }

    /// Open-path hook (shadow-field card 1, inc 4): re-establish the quiesce
    /// set from the persisted `c:P:` plans and resume any drivable migration
    /// whose converter is already registered. Runs ONLY on a genuine open
    /// (not the `_consuming` rebuild — see `rebuild_with_arc_storage`).
    ///
    /// At a fresh open the per-`Database` converter registry is empty (the
    /// operator registers converters AFTER open), so drivable plans are armed
    /// but NOT driven here — the operator calls `resume_field_type_migration`
    /// after registering. Every quiescing plan (incl. `Failed` /
    /// `AwaitingConverter`) still re-arms quiesce so writes stay rejected
    /// across the restart.
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
            // plan (crashed mid-cutover) is a pure rename pass and resumes with
            // NO converter — so a reopen finishes the cutover even though the
            // per-`Database` converter registry is empty after restart.
            let is_cutting = plan.phase == crate::catalog::MigrationPhase::CuttingOver;
            if plan.status.is_drivable() && (converter.is_some() || is_cutting) {
                self.guard_resume_schema(&plan)?;
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
        if plan.phase == crate::catalog::MigrationPhase::Converting && converter.is_none() {
            return Err(EngineError::ConverterNotRegistered {
                name: plan.converter_name.clone(),
                version: plan.converter_version,
            });
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
                status: p.status,
                cursor: p.cursor,
                objects_converted: p.objects_converted,
                chunk_size: p.chunk_size,
                converter_name: p.converter_name,
                converter_version: p.converter_version,
                created_at_ms: p.created_at_ms,
            })
            .collect())
    }

    /// Create a new object of the given type.
    ///
    /// `fields` may include forward (non-inverse) relation fields whose
    /// value is an integer target id — the engine then writes the forward
    /// edge AND rev_edge as part of the same txn, with symmetric covers
    /// built from the in-memory FieldMap (no per-link `scan_prefix` for
    /// other targets, no extra commits). This collapses the historical
    /// `Type.create + link + link` 3-txn dance into one batched txn.
    pub fn create(&self, type_name: &str, fields: FieldMap) -> EngineResult<Object> {
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

        let scalar_fields = self.stage_create_writes(
            &mut txn, type_name, type_def, type_id, object_id, &fields, &mut puts,
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
    /// Unique-index puts are issued inline (the next row in `create_batch`
    /// must see them through MVCC to detect intra-batch dup values). All
    /// other writes accumulate into `puts` for the caller to flush via
    /// `put_batch`. Returns the scalar-only `FieldMap` for the response.
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

        // Unique-index writes for scalar fields (inline — required for
        // cross-row uniqueness detection inside a single `create_batch`).
        for (field_name, value) in &scalar_fields {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() && !matches!(value, Value::Null) {
                self.check_unique_and_insert(
                    txn, type_name, type_id, field_name, value, object_id,
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
        // flushes via one `put_batch` at the end. Unique-index puts stay
        // inline because the next row's uniqueness check inside the same
        // txn must see them. Per-row scalar FieldMaps are reconstructed
        // post-batch for the published events / returned Objects.
        let mut puts: Vec<(Bytes, Bytes)> = Vec::with_capacity(rows.len() * 2);
        let mut scalar_rows: Vec<FieldMap> = Vec::with_capacity(rows.len());

        for fields in &rows {
            let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);
            match self.stage_create_writes(
                &mut txn, type_name, type_def, type_id, object_id, fields, &mut puts,
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

        let snapshot = self.storage.read_snapshot();
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
            // Non-integer field, or an integer field whose declared type can't
            // represent `target` (out of range). The typed index/zone fast
            // path doesn't apply — compare per row so the answer stays correct
            // (and empty for non-numeric fields) instead of returning the whole
            // table.
            _ => {
                return self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
                    Value::U32(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::U64(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::I32(n) => Some(compare_partial(*n as i128, op, target as i128)),
                    Value::I64(n) => Some(compare_partial(*n as i128, op, target as i128)),
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
        let snapshot = self.storage.read_snapshot();
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
    fn filter_scan_via_index(
        &self,
        type_name: &str,
        type_id: u64,
        field_id: u64,
        op: rhypedb_storage::zone::CompareOp,
        target_bytes: &[u8; 8],
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_storage::zone::CompareOp;

        let snapshot = self.storage.read_snapshot();
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
                out.extend(self.get_many(type_name, &fallback_ids)?);
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
            out.extend(self.get_many(type_name, &fallback_ids)?);
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
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encoded,
                limit,
            );
        }
        self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
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
        self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
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

        if let Some(idx_fields) = self.indexed_fields.get(type_name)
            && let Some(ifd) = idx_fields.iter().find(|f| f.name == field_name)
            && ifd.kind == IndexedKind::Bytes
        {
            return self.filter_scan_via_index_var(
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encode_bytes_for_index(target),
                limit,
            );
        }
        self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
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
                type_name,
                type_id,
                ifd.field_id,
                op,
                &encode_str_for_index(target),
                limit,
            );
        }

        // === Non-indexed fallback: full type scan, post-filter ===
        self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
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
        let snapshot = self.storage.read_snapshot();
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
    fn filter_scan_via_index_var(
        &self,
        type_name: &str,
        type_id: u64,
        field_id: u64,
        op: rhypedb_storage::zone::CompareOp,
        target_encoded: &[u8],
        limit: Option<usize>,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_storage::zone::CompareOp;

        let snapshot = self.storage.read_snapshot();
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
                out.extend(self.get_many(type_name, &fallback_ids)?);
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
            out.extend(self.get_many(type_name, &fallback_ids)?);
        }
        Ok(out)
    }

    /// Update an object's fields. Only the provided fields are updated;
    /// unmentioned fields are preserved.
    pub fn update(
        &self,
        type_name: &str,
        object_id: u64,
        updates: FieldMap,
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

        // Check unique constraints for updated fields.
        for (field_name, value) in &updates {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() && !matches!(value, Value::Null) {
                // Remove old unique index entry if the field had a value.
                if let Some(old_value) = fields.get(field_name)
                    && !matches!(old_value, Value::Null)
                {
                    self.remove_unique_index(&mut txn, type_name, type_id, field_name, old_value)?;
                }
                self.check_unique_and_insert(
                    &mut txn, type_name, type_id, field_name, value, object_id,
                )?;
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
    pub fn delete(&self, type_name: &str, object_id: u64) -> EngineResult<()> {
        let _migration_guard = self.migration_lock.read();
        let type_id = self.resolve_type_id(type_name)?;

        let mut txn = self.storage.begin_txn();
        // type_id keyed instead of (String, u64) — drops a String alloc
        // per cascaded object. At K=100 that's 100 fewer allocations per
        // User delete.
        let mut deleted: std::collections::HashSet<(u64, u64)> =
            std::collections::HashSet::with_capacity(128);
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
        for (del_type_id, del_id) in &deleted {
            self.forget_version(*del_type_id, *del_id);
        }

        for (del_type_id, del_id) in &deleted {
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
                fields: None,
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
        deleted: &mut std::collections::HashSet<(u64, u64)>,
        arena: &mut TombstoneArena,
        cascade_ctx: Option<(u64, Bytes)>,
    ) -> EngineResult<()> {
        if !deleted.insert((type_id, object_id)) {
            return Ok(()); // already deleted in this cascade chain
        }

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

        // Unique-index + secondary-index cleanup. Skipped entirely when the
        // type has neither — for edge-only types (Rating in the bench) the
        // cascade never touches the object payload.
        let type_idx_fields = self.indexed_fields.get(&meta.type_name);
        if meta.has_unique || meta.has_indexed || verify_exists {
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
                let fields = deserialize_fields(data);
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

    pub fn storage(&self) -> &Arc<LsmTree> {
        &self.storage
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

    /// Check that a unique value doesn't already exist, and insert the index entry.
    fn check_unique_and_insert(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        type_id: u64,
        field_name: &str,
        value: &Value,
        object_id: u64,
    ) -> EngineResult<()> {
        let field_key = format!("{type_name}.{field_name}");
        let field_id = self.field_ids[&field_key];
        let value_bytes = value_to_index_bytes(value);
        let unique_key = KeyBuilder::unique_index(type_id, field_id, &value_bytes);

        if let Some(existing) = self.storage.get(txn, &unique_key)? {
            let existing_id = u64::from_be_bytes(existing[..8].try_into().unwrap());
            if existing_id != object_id {
                return Err(EngineError::UniqueViolation {
                    type_name: type_name.into(),
                    field: field_name.into(),
                    value: value.to_string(),
                });
            }
        }

        let mut id_buf = bytes::BytesMut::with_capacity(8);
        bytes::BufMut::put_u64(&mut id_buf, object_id);
        self.storage.put(txn, &unique_key, id_buf.freeze())?;

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
    fn refresh_covers_for_target(&self, target_type_id: u64, target_id: u64) -> EngineResult<()> {
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
    }
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
    db.migrating_field_count
        .load(std::sync::atomic::Ordering::Relaxed)
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
        Value::Null => vec![],
    }
}

fn fields_to_json(fields: &FieldMap) -> HashMap<String, serde_json::Value> {
    fields
        .iter()
        .map(|(k, v)| {
            let json_val = match v {
                Value::String(s) => serde_json::Value::String(s.clone()),
                Value::U32(n) => serde_json::json!(n),
                Value::U64(n) => serde_json::json!(n),
                Value::I32(n) => serde_json::json!(n),
                Value::I64(n) => serde_json::json!(n),
                Value::F32(n) => serde_json::json!(n),
                Value::F64(n) => serde_json::json!(n),
                Value::Bool(b) => serde_json::Value::Bool(*b),
                Value::Bytes(b) => serde_json::json!(format!("<{} bytes>", b.len())),
                Value::Null => serde_json::Value::Null,
            };
            (k.clone(), json_val)
        })
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
/// type that has at least one integer scalar field; the engine never enrolls
/// non-integer fields in zone maps (`encode_int_for_zone` returns `None`
/// for them).
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
/// `encode_int_for_zone`'s match arms.
fn field_is_zone_eligible(field: &rhypedb_schema::FieldDef) -> bool {
    matches!(
        &field.field_type,
        FieldType::Scalar(
            ScalarType::U32 | ScalarType::U64 | ScalarType::I32 | ScalarType::I64
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
/// positives; narrow types widen to 64 bits first. Returns `None` for non-
/// integer values (strings, floats, bools, nulls, bytes) — those aren't
/// zone-mapped.
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
    fn rename_field_indexed_field_refused_at_db_layer() {
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
        let err = db.rename_field("Movie", "year", "released_in").unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::RenameFieldDirectiveUnsupported {
                directive: "@indexed",
                ..
            })
        ));
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
            db.create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
            })
            .unwrap();
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
                chunk_size: 2,
            })
            .unwrap();
        assert_eq!(plan_id, 1);
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
        db.create_field_type_migration(MigrationPlanSpec {
            type_name: "User".into(),
            field_name: "score".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 4,
        })
        .unwrap();
        let after = db.storage.txn_manager().current_version();
        let chunks = 10u64.div_ceil(4);
        assert!(chunks > 1, "test must exercise multiple chunks");
        // Card-2 online flow, per-chunk commits throughout:
        //   1 plan-create
        // + `chunks` shadow-backfill commits (run_migration_chunks)
        // + 1 phase-flip commit (Converting → CuttingOver)
        // + `chunks` cutover commits (promote shadow + maintain covers)
        // + 1 finalize commit (catalog kind flip + Completed)
        assert_eq!(
            after - before,
            1 + chunks + 1 + chunks + 1,
            "expected per-chunk commits in both passes, not a single batch"
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 4,
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
        db.run_cutover(created.plan_id, created.type_id).unwrap();
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

        db.create_field_type_migration(MigrationPlanSpec {
            type_name: "User".into(),
            field_name: "score".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 4,
        })
        .unwrap();
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
        db.create_field_type_migration(MigrationPlanSpec {
            type_name: "T".into(),
            field_name: "x".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 4,
        })
        .unwrap();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
        )
        .unwrap();
        // Cut over WITHOUT backfilling any shadows → first row refuses.
        let err = db.run_cutover(created.plan_id, created.type_id).unwrap_err();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 2, 16,
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
        let err = db.run_cutover(created.plan_id, created.type_id).unwrap_err();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
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
        db.run_cutover(created.plan_id, created.type_id).unwrap();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
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

        let err = db.run_cutover(created.plan_id, created.type_id).unwrap_err();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 2,
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

        db.create_field_type_migration(MigrationPlanSpec {
            type_name: "T".into(),
            field_name: "x".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 8,
        })
        .unwrap();
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

        let err = db
            .create_field_type_migration(MigrationPlanSpec {
                type_name: "User".into(),
                field_name: "score".into(),
                target_field_type: FieldType::Scalar(ScalarType::F64),
                converter_name: "widen".into(),
                converter_version: 1,
                chunk_size: 4,
            })
            .unwrap_err();
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
                chunk_size: 0,
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
                chunk_size: 0,
            })
            .unwrap();
        assert_eq!(p1, 1);
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
                chunk_size: 0,
            })
            .unwrap();
        assert_eq!(p2, 2, "plan id must not reset or reuse across restart");
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 4,
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
                chunk_size: 4,
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
            &db.storage, &db.schema, type_name, field, target, "widen", 1, 4,
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
        // Reopen with the OLD (source) schema — open succeeds, quiesce armed.
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
        db.create_field_type_migration(MigrationPlanSpec {
            type_name: "User".into(),
            field_name: "score".into(),
            target_field_type: FieldType::Scalar(ScalarType::F64),
            converter_name: "widen".into(),
            converter_version: 1,
            chunk_size: 4,
        })
        .unwrap();
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
            let err = db
                .create_field_type_migration(MigrationPlanSpec {
                    type_name: "User".into(),
                    field_name: "score".into(),
                    target_field_type: FieldType::Scalar(ScalarType::F64),
                    converter_name: "widen".into(),
                    converter_version: 1,
                    chunk_size: 2,
                })
                .unwrap_err();
            assert!(matches!(
                err,
                EngineError::Catalog(crate::CatalogError::FieldTypeChangeConverterFailed { .. })
            ));
            plan_id = db.list_migrations().unwrap()[0].plan_id;
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
                chunk_size: 4,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::MigrationFieldHasActivePlan { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Card 2b: double-write producer hook + reader strip (isolation tests —
    // quiesce still blocks the live create/update path in 2b, so the hook is
    // exercised directly and the strip via a hand-written shadow blob)
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
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
        db.run_cutover(created.plan_id, created.type_id).unwrap();
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
            &db.storage, &db.schema, "User", "score", target, "widen", 1, 16,
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
    fn change_field_type_refuses_unrepresentable_target() {
        // DateTime/Json have no writable Value variant, so no converter could
        // ever produce a matching value — refuse the target up front (also
        // covers the chunked create path, which shares this validation).
        let dir = tempfile::tempdir().unwrap();
        let schema = parse_schema("type User { age: i64 }").unwrap();
        let db = Database::open(schema, dir.path()).unwrap();
        let err = db
            .change_field_type(
                "User",
                "age",
                rhypedb_schema::FieldType::Scalar(rhypedb_schema::ScalarType::DateTime),
                |_id, v| Ok(v.clone()),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EngineError::Catalog(crate::CatalogError::FieldTypeChangeUnrepresentableTarget { .. })
        ));
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
        let mut deleted_types = Vec::new();
        for _ in 0..2 {
            if let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(1)) {
                deleted_types.push(event.type_name.clone());
            }
        }
        deleted_types.sort();
        assert_eq!(deleted_types, vec!["Post", "User"]);
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
