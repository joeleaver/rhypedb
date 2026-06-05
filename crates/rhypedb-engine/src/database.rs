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
use crate::object::{FieldMap, Object, Value, deserialize_fields, serialize_fields};

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

    fn push_unique_index(&mut self, type_id: u64, field_hash: u64, value_bytes: &[u8]) {
        let r = KeyBuilder::unique_index_into(&mut self.buf, type_id, field_hash, value_bytes);
        self.ranges.push(r);
    }

    fn push_field_index(
        &mut self,
        type_id: u64,
        field_hash: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) {
        let r = KeyBuilder::field_index_into(
            &mut self.buf,
            type_id,
            field_hash,
            encoded_value,
            object_id,
        );
        self.ranges.push(r);
    }

    fn push_field_index_var(
        &mut self,
        type_id: u64,
        field_hash: u64,
        encoded_value: &[u8],
        object_id: u64,
    ) {
        let r = KeyBuilder::field_index_var_into(
            &mut self.buf,
            type_id,
            field_hash,
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
    next_object_id: AtomicU64,
    subscriptions: SubscriptionHub,
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
    /// (field_name, field_hash). Cached so the create/update/delete write
    /// paths don't re-traverse the schema per object.
    indexed_fields: HashMap<String, Vec<IndexedField>>,
    /// Per-object monotonic generation counter, bumped on every successful
    /// `update`. Lives in-memory for cheap reads (cover-write stamps the
    /// target's current generation into `<name>__cover_v`; executor fusion
    /// compares against the live generation to detect stale covers). Backed
    /// by `g:<type_id>:<object_id>` keys for restart durability — the map is
    /// repopulated by scanning that prefix in `open()`.
    version_counters: RwLock<HashMap<(u64, u64), u64>>,
    /// Cheap lockless "is `version_counters` non-empty?" check. Cascade
    /// delete uses it to skip the per-cascaded-object `version_counters.read()
    /// .contains_key()` when no object has ever been updated. For the bench
    /// (insert + delete, no updates) this saves 100 RwLock acquires per
    /// User-delete at K=100. Updated alongside every map mutation.
    version_counter_count: std::sync::atomic::AtomicUsize,
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
        config.zone_extractor = Some(Arc::new(extract_zone_fields));
        config.sync_on_commit = options.sync_on_commit;
        let storage = LsmTree::open(config)?;

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

        // Recover the max object ID by scanning existing objects.
        let mut max_object_id = 0u64;
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
                    }
                }
            }
        }
        drop(txn);

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

        // Repopulate the per-object generation counter from `g:` keys. Cost
        // is one prefix scan at startup proportional to the number of
        // never-yet-updated objects, then HashMap lookups for every cover-
        // write and every fusion check. Counters for objects that have
        // never been updated stay absent (read-side defaults to 0).
        let mut version_counters: HashMap<(u64, u64), u64> = HashMap::new();
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

        let db = Arc::new(Self {
            schema,
            storage,
            type_ids,
            rel_ids,
            field_ids,
            next_object_id: AtomicU64::new(max_object_id + 1),
            subscriptions: SubscriptionHub::new(),
            incoming_relations,
            cascade_meta_by_id,
            type_name_by_id,
            indexed_fields,
            version_counter_count: std::sync::atomic::AtomicUsize::new(version_counters.len()),
            version_counters: RwLock::new(version_counters),
            cover_refresh_tx: parking_lot::Mutex::new(None),
            cover_refresh_handle: parking_lot::Mutex::new(None),
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

    /// Create a new object of the given type.
    ///
    /// `fields` may include forward (non-inverse) relation fields whose
    /// value is an integer target id — the engine then writes the forward
    /// edge AND rev_edge as part of the same txn, with symmetric covers
    /// built from the in-memory FieldMap (no per-link `scan_prefix` for
    /// other targets, no extra commits). This collapses the historical
    /// `Type.create + link + link` 3-txn dance into one batched txn.
    pub fn create(&self, type_name: &str, fields: FieldMap) -> EngineResult<Object> {
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);

        let mut txn = self.storage.begin_txn();
        let mut puts: Vec<(Bytes, Bytes)> = Vec::new();

        let scalar_fields = self.stage_create_writes(
            &mut txn, type_name, type_def, type_id, object_id, &fields, &mut puts,
        )?;

        self.storage.put_batch(&mut txn, &puts)?;
        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
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
    /// stamps are read from the in-memory `version_counters` map (defaults
    /// to 0 for never-updated targets).
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

        let serialized = serialize_fields(&scalar_fields);

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

        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];

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
            let scalar_fields = self.stage_create_writes(
                &mut txn, type_name, type_def, type_id, object_id, fields, &mut puts,
            )?;
            scalar_rows.push(scalar_fields);
            object_ids.push(object_id);
        }

        self.storage.put_batch(&mut txn, &puts)?;

        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

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
        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

        let key = KeyBuilder::object(type_id, object_id);
        let snapshot = self.storage.read_snapshot();
        let data =
            self.storage
                .get_at(snapshot, &key)?
                .ok_or_else(|| EngineError::ObjectNotFound {
                    type_name: type_name.into(),
                    object_id,
                })?;

        let fields = deserialize_fields(&data);
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
        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

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
                out.push(Object {
                    type_name: type_name.into(),
                    id,
                    fields: deserialize_fields(&data),
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
        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

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

        let mut out = Vec::with_capacity(sorted.len());
        for (id, value) in sorted.into_iter().zip(values) {
            if let Some(data) = value {
                out.push(Object::from_raw(type_name.into(), id, data));
            }
        }
        Ok(out)
    }

    /// Scan all objects of a given type. Uses the LSM prefix scan on the
    /// object key prefix, so this is a real index scan — not a brute-force probe.
    pub fn scan_type(&self, type_name: &str) -> EngineResult<Vec<Object>> {
        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

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

            let fields = deserialize_fields(&data);
            objects.push(Object {
                type_name: type_name.into(),
                id: object_id,
                fields,
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
        use rhypedb_storage::zone::{FieldPredicate, hash_field_name};

        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def =
            type_def
                .get_field(field_name)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })?;

        // Cast the raw query target to the field's actual integer type so the
        // encoding matches what's on disk. Bail out to `scan_type` (no perf
        // gain, but correct) for non-integer scalar fields.
        let target_value = match &field_def.field_type {
            FieldType::Scalar(ScalarType::U32) => {
                if !(0..=u32::MAX as i64).contains(&target) {
                    // Predicate target out of u32 range — answer is trivially
                    // empty (no entry can equal/compare appropriately) for
                    // many ops; fall back to scan_type for the safety.
                    return self.scan_type(type_name);
                }
                Value::U32(target as u32)
            }
            FieldType::Scalar(ScalarType::U64) => {
                if target < 0 {
                    return self.scan_type(type_name);
                }
                Value::U64(target as u64)
            }
            FieldType::Scalar(ScalarType::I32) => Value::I32(target as i32),
            FieldType::Scalar(ScalarType::I64) => Value::I64(target),
            _ => return self.scan_type(type_name),
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
        let predicate = FieldPredicate {
            field_hash: hash_field_name(field_name.as_bytes()),
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
        let cap = limit.unwrap_or(usize::MAX);
        let mut objects = Vec::new();
        for (key, data) in entries {
            if objects.len() >= cap {
                break;
            }
            if key.len() < 8 {
                continue;
            }
            let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().unwrap();
            let object_id = u64::from_be_bytes(id_bytes);

            let fields = deserialize_fields(&data);
            if entry_passes_int_predicate(&fields, field_name, op, target_u64) {
                objects.push(Object {
                    type_name: type_name.into(),
                    id: object_id,
                    fields,
                    raw_fields: None,
                });
            }
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
                    out.push(Object {
                        type_name: type_name.into(),
                        id: object_id,
                        fields: deserialize_fields(&value),
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
                out.push(Object {
                    type_name: type_name.into(),
                    id: object_id,
                    fields: deserialize_fields(&value),
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
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def =
            type_def
                .get_field(field_name)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })?;
        if !matches!(field_def.field_type, FieldType::Scalar(ScalarType::Bool)) {
            return self.scan_type(type_name);
        }

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
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def =
            type_def
                .get_field(field_name)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })?;
        let is_float = matches!(
            field_def.field_type,
            FieldType::Scalar(ScalarType::F32 | ScalarType::F64)
        );
        if !is_float {
            return self.scan_type(type_name);
        }

        if let Some(idx_fields) = self.indexed_fields.get(type_name)
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
        self.filter_scan_fallback(type_name, field_name, limit, |v| match v {
            Value::F64(f) => Some(compare_partial(*f, op, target)),
            Value::F32(f) => Some(compare_partial(*f as f64, op, target)),
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
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def =
            type_def
                .get_field(field_name)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })?;
        if !matches!(field_def.field_type, FieldType::Scalar(ScalarType::Bytes)) {
            return self.scan_type(type_name);
        }

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
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def =
            type_def
                .get_field(field_name)
                .ok_or_else(|| EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.into(),
                })?;

        // Non-string scalar field — no point pretending the literal applies.
        if !matches!(field_def.field_type, FieldType::Scalar(ScalarType::String)) {
            return self.scan_type(type_name);
        }

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

    /// Per-row fallback when no index serves the predicate. Walks every
    /// object of the type via `scan_type` and applies `predicate` to the
    /// field's value, capping at `limit` matches. `predicate` returns
    /// `Some(bool)` (pass / fail) per value; `None` is treated as fail.
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
        let cap = limit.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        for obj in self.scan_type(type_name)? {
            if out.len() >= cap {
                break;
            }
            let pass = obj
                .fields
                .get(field_name)
                .and_then(&mut predicate)
                .unwrap_or(false);
            if pass {
                out.push(obj);
            }
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
                    out.push(Object {
                        type_name: type_name.into(),
                        id: object_id,
                        fields: deserialize_fields(&value),
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
                out.push(Object {
                    type_name: type_name.into(),
                    id: object_id,
                    fields: deserialize_fields(&value),
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
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];

        // Validate update fields.
        for (field_name, value) in &updates {
            let field_def =
                type_def
                    .get_field(field_name)
                    .ok_or_else(|| EngineError::FieldNotFound {
                        type_name: type_name.into(),
                        field: field_name.clone(),
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

        let serialized = serialize_fields(&fields);

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
                let new_value_opt = fields.get(&ifd.name).cloned();
                let value_changed = old_value_opt != &new_value_opt;
                if value_changed {
                    if let Some(old_v) = old_value_opt
                        && !matches!(old_v, Value::Null)
                    {
                        self.remove_field_index(
                            &mut txn,
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
                            &mut txn,
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
                        &mut txn,
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
        // source updates. Bounded cost: number of outbound relation
        // endpoints, which is at most the schema-declared field count for
        // forward-1:1 relations and total linked-target count for many
        // relations. (Phase 2 — refreshing this object's OWN cover in the
        // rev_edges that INCOMING references hold us under `__cover` — is
        // handled lazily via the per-object version + reader-side fall-
        // through, plus a background sweeper that opportunistically rewrites
        // stale embedded covers — see `cover_refresh_worker`.)
        self.refresh_outbound_rev_edges(&mut txn, type_name, object_id, Some(&serialized))?;

        // Phase 2: bump this object's generation. Every rev_edge that
        // embedded us as `<name>__cover` earlier now has a stale snapshot;
        // the executor's fusion path detects mismatch via a HashMap lookup
        // against this counter and falls through to a fresh LSM probe for
        // those specific targets. Bounded write cost (one in-memory bump +
        // one persisted `g:` put) regardless of how many incoming
        // references this object has — that fan-in could be millions.
        let new_version = self.bump_version(type_id, object_id);
        self.storage.put(
            &mut txn,
            &KeyBuilder::object_version(type_id, object_id),
            Bytes::copy_from_slice(&new_version.to_be_bytes()),
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

    /// Delete an object, enforcing @on_delete policies on all inbound relationships.
    /// Cascades are recursive — if deleting A cascades to B, and B has its own
    /// cascade relationships, those are followed too.
    pub fn delete(&self, type_name: &str, object_id: u64) -> EngineResult<()> {
        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

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

        // Drop the persisted generation counter. The `g:` key only exists
        // for objects that have been updated at least once; tombstoning a
        // non-existent key is pure WAL+memtable bloat. The atomic
        // `version_counter_count` lets us skip the RwLock acquire entirely
        // when nothing has ever been updated (the bench case).
        if self
            .version_counter_count
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0
        {
            let has_version = self
                .version_counters
                .read()
                .contains_key(&(type_id, object_id));
            if has_version {
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

        let rel = match &field.field_type {
            FieldType::Relation(r) => r,
            _ => {
                return Err(EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                });
            }
        };

        // Verify both objects exist.
        let source_type_id = self.type_ids[source_type];
        let target_type_id = self.type_ids[&rel.target_type];

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
        let rel_key = format!("{source_type}.{field_name}");
        let rel_id = *self
            .rel_ids
            .get(&rel_key)
            .ok_or_else(|| EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            })?;

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
        return Ok(target_data);
    };

    // Cheap pre-check: does this type even have any 1:1 forward outgoing
    // relations? If not, skip the deserialize+reserialize round trip and
    // return the original bytes (refcount-only, no copy).
    let has_any_forward_1to1 = type_def.fields.iter().any(|f| {
        matches!(&f.field_type, FieldType::Relation(rel) if !rel.is_many) && f.inverse().is_none()
    });
    if !has_any_forward_1to1 {
        return Ok(target_data);
    }

    let mut effective = deserialize_fields(&target_data);
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
        effective.insert(format!("{}__cover", field.name), Value::Bytes(next_data));
        let next_v = db.object_version(&rel.target_type, next_tid);
        effective.insert(format!("{}__cover_v", field.name), Value::U64(next_v));
    }

    if !wrote_any {
        // No outgoing 1:1 edges actually populated (type has the fields
        // but this instance hasn't been linked yet). Return original bytes
        // unchanged so we don't pay reserialization cost for no benefit.
        return Ok(target_data);
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

/// Zone-field extractor passed to `LsmConfig::zone_extractor`. Pulls integer
/// field values out of an object entry's serialized FieldMap so the SST
/// writer can record per-block min/max bounds.
///
/// Returns empty when the entry isn't an object key (edges, reverse edges,
/// unique-index entries, etc.) since their values aren't FieldMaps. Object
/// keys are `o:<type>:<id>` so the prefix check is a 2-byte compare.
pub(crate) fn extract_zone_fields(internal_key: &[u8], value: &[u8]) -> Vec<(u32, [u8; 8])> {
    use rhypedb_storage::zone::hash_field_name;

    if internal_key.len() < 2 || internal_key[0] != b'o' || internal_key[1] != b':' {
        return Vec::new();
    }
    let fields = deserialize_fields(value);
    let mut out = Vec::with_capacity(2);
    for (name, val) in &fields {
        if let Some(encoded) = encode_int_for_zone(val) {
            out.push((hash_field_name(name.as_bytes()), encoded));
        }
    }
    out
}

/// Per-entry re-check for `filter_scan`. The block-level zone filter is
/// coarse; entries that survived may still individually fail the predicate.
pub(crate) fn entry_passes_int_predicate(
    fields: &FieldMap,
    field_name: &str,
    op: rhypedb_storage::zone::CompareOp,
    target_u64: u64,
) -> bool {
    use rhypedb_storage::zone::CompareOp;

    let Some(value) = fields.get(field_name) else {
        return false;
    };
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
            0,
            "freshly-created object should have generation 0"
        );

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", uid, upd).unwrap();

        assert_eq!(
            db.object_version("User", uid),
            1,
            "successful update must bump the per-object generation"
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
            stamped_v, 0,
            "stamp records the target's generation as of cover-write time"
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
        //   1. Pre-update: cover_v stamp == 0 (user never bumped).
        //   2. Post-update + post-sweep: cover_v stamp matches the new
        //      live generation AND the embedded cover bytes reflect the new
        //      user.name.
        let (db, uid, mid, _rid, _dir) = build_one_rating(covering_schema());

        let stamp_before =
            read_movie_side_user_cover_v(&db, mid).expect("cover_v should be present pre-update");
        assert_eq!(stamp_before, 0);

        let mut upd = FieldMap::new();
        upd.insert("name".into(), Value::String("Renamed".into()));
        db.update("User", uid, upd).unwrap();
        let new_v = db.object_version("User", uid);
        assert_eq!(new_v, 1);

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
        assert_eq!(stamp, 0, "no sweeper means cover_v stamp must stay at 0");
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
        assert_eq!(stamp, 0, "fresh director has generation 0");
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
}
