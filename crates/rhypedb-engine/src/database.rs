use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;

use rhypedb_schema::{FieldType, OnDeletePolicy, Schema};
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::lsm::{LsmConfig, LsmTree};
use rhypedb_subscribe::{ChangeEvent, ChangeKind, SubscriptionHub};

use crate::error::{EngineError, EngineResult};
use crate::object::{deserialize_fields, serialize_fields, FieldMap, Object, Value};

/// One row of the precomputed reverse-relation index used by cascade delete.
#[derive(Debug, Clone)]
struct IncomingRelation {
    source_type: String,
    source_field: String,
    rel_id: u64,
    policy: OnDeletePolicy,
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
    /// target_type → list of relations that point at it. Built once at open()
    /// so cascade delete doesn't iterate the whole schema per recursive call.
    incoming_relations: HashMap<String, Vec<IncomingRelation>>,
    /// Set of type names that have at least one @unique field. Lets cascade
    /// delete skip the storage.get + unique-index walk for types that don't
    /// need it (e.g. Rating in the bench schema has no unique fields).
    types_with_unique: std::collections::HashSet<String>,
}

impl Database {
    /// Open a database with the given schema and data directory.
    pub fn open(schema: Schema, data_dir: impl AsRef<Path>) -> EngineResult<Self> {
        let mut config = LsmConfig::new(data_dir);
        // Wire a zone-field extractor that pulls integer field values out of
        // object entries' FieldMap blobs at SST flush/compaction time. Lets
        // `filter_scan` skip blocks whose min/max bounds rule out the
        // predicate without per-entry decode + compare.
        config.zone_extractor = Some(Arc::new(extract_zone_fields));
        let storage = Arc::new(LsmTree::open(config)?);

        // Assign stable numeric IDs to types and relationships.
        let mut type_ids = HashMap::new();
        let mut rel_ids = HashMap::new();
        let mut field_ids = HashMap::new();
        let mut next_rel_id = 1u64;
        let mut next_field_id = 1u64;

        let mut type_names: Vec<_> = schema.types.keys().cloned().collect();
        type_names.sort();
        for (type_id, name) in (1u64..).zip(type_names.iter()) {
            type_ids.insert(name.clone(), type_id);

            let type_def = &schema.types[name];
            for field in &type_def.fields {
                let field_key = format!("{}.{}", name, field.name);
                field_ids.insert(field_key.clone(), next_field_id);
                next_field_id += 1;

                if matches!(field.field_type, FieldType::Relation(_)) {
                    rel_ids.insert(field_key, next_rel_id);
                    next_rel_id += 1;
                }
            }
        }

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
        // looking for inbound edges.
        let mut incoming_relations: HashMap<String, Vec<IncomingRelation>> = HashMap::new();
        for (source_type, type_def) in &schema.types {
            for field in &type_def.fields {
                if let FieldType::Relation(rel) = &field.field_type {
                    let rel_key = format!("{source_type}.{}", field.name);
                    let rel_id = rel_ids[&rel_key];
                    let policy = field.on_delete().cloned().unwrap_or(if rel.is_many {
                        OnDeletePolicy::Remove
                    } else {
                        OnDeletePolicy::Deny
                    });
                    incoming_relations
                        .entry(rel.target_type.clone())
                        .or_default()
                        .push(IncomingRelation {
                            source_type: source_type.clone(),
                            source_field: field.name.clone(),
                            rel_id,
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

        Ok(Self {
            schema,
            storage,
            type_ids,
            rel_ids,
            field_ids,
            next_object_id: AtomicU64::new(max_object_id + 1),
            subscriptions: SubscriptionHub::new(),
            incoming_relations,
            types_with_unique,
        })
    }

    /// Create a new object of the given type.
    pub fn create(&self, type_name: &str, fields: FieldMap) -> EngineResult<Object> {
        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);

        // Validate fields against schema.
        for (field_name, value) in &fields {
            let field_def = type_def.get_field(field_name).ok_or_else(|| {
                EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.clone(),
                }
            })?;
            validate_value(field_def, value)?;
        }

        let key = KeyBuilder::object(type_id, object_id);
        let serialized = serialize_fields(&fields);

        let mut txn = self.storage.begin_txn();

        // Check and write unique index entries.
        for (field_name, value) in &fields {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() && !matches!(value, Value::Null) {
                self.check_unique_and_insert(
                    &mut txn, type_name, type_id, field_name, value, object_id,
                )?;
            }
        }

        self.storage.put(&mut txn, &key, serialized)?;
        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        self.subscriptions.publish(ChangeEvent {
            version,
            kind: ChangeKind::Create,
            type_name: type_name.into(),
            object_id,
            fields: Some(fields_to_json(&fields)),
        });

        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields,
        })
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
    pub fn create_batch(
        &self,
        type_name: &str,
        rows: Vec<FieldMap>,
    ) -> EngineResult<Vec<Object>> {
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

        for fields in &rows {
            // Validate fields against schema.
            for (field_name, value) in fields {
                let field_def = type_def.get_field(field_name).ok_or_else(|| {
                    EngineError::FieldNotFound {
                        type_name: type_name.into(),
                        field: field_name.clone(),
                    }
                })?;
                validate_value(field_def, value)?;
            }

            let object_id = self.next_object_id.fetch_add(1, Ordering::SeqCst);
            let key = KeyBuilder::object(type_id, object_id);
            let serialized = serialize_fields(fields);

            // Unique-index writes for this row. Within one txn, a second row
            // with the same unique value will see the first via MVCC and
            // fail with a unique violation — same semantics as N serial creates.
            for (field_name, value) in fields {
                let field_def = type_def.get_field(field_name).unwrap();
                if field_def.is_unique() && !matches!(value, Value::Null) {
                    self.check_unique_and_insert(
                        &mut txn, type_name, type_id, field_name, value, object_id,
                    )?;
                }
            }

            self.storage.put(&mut txn, &key, serialized)?;
            object_ids.push(object_id);
        }

        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        // Build the returned Objects + publish events after commit.
        let mut out = Vec::with_capacity(rows.len());
        for (id, fields) in object_ids.into_iter().zip(rows.into_iter()) {
            self.subscriptions.publish(ChangeEvent {
                version,
                kind: ChangeKind::Create,
                type_name: type_name.into(),
                object_id: id,
                fields: Some(fields_to_json(&fields)),
            });
            out.push(Object {
                type_name: type_name.into(),
                id,
                fields,
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
        let data = self.storage.get_at(snapshot, &key)?.ok_or_else(|| {
            EngineError::ObjectNotFound {
                type_name: type_name.into(),
                object_id,
            }
        })?;

        let fields = deserialize_fields(&data);
        Ok(Object {
            type_name: type_name.into(),
            id: object_id,
            fields,
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

        let mut out = Vec::with_capacity(sorted.len());
        for (id, value) in sorted.into_iter().zip(values.into_iter()) {
            if let Some(data) = value {
                out.push(Object {
                    type_name: type_name.into(),
                    id,
                    fields: deserialize_fields(&data),
                });
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
            });
        }

        Ok(objects)
    }

    /// Filtered scan: like `scan_type` but pushes a single-field integer
    /// comparison down to storage so SST blocks whose zone bounds rule out
    /// the predicate skip whole groups of entries without decode.
    ///
    /// `target` is the raw query-level integer; this method looks up the
    /// field's schema type (U32 / U64 / I32 / I64) and re-encodes the target
    /// to match the on-disk byte-order encoding. Out-of-range targets (e.g.,
    /// negative literal against a U32 field with `<` op) take a fast-path
    /// degenerate result rather than scanning.
    ///
    /// The caller still gets back only objects matching the predicate — the
    /// post-block decode loop re-evaluates the comparison since zone maps
    /// are a coarse-grained pre-filter.
    ///
    /// Returns `Err(FieldNotFound)` for unknown fields and falls back to
    /// `scan_type` for non-integer field types.
    pub fn filter_scan(
        &self,
        type_name: &str,
        field_name: &str,
        op: rhypedb_storage::zone::CompareOp,
        target: i64,
    ) -> EngineResult<Vec<Object>> {
        use rhypedb_schema::ScalarType;
        use rhypedb_storage::zone::{hash_field_name, FieldPredicate};

        let type_def = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;
        let type_id = self.type_ids[type_name];
        let field_def = type_def.get_field(field_name).ok_or_else(|| {
            EngineError::FieldNotFound {
                type_name: type_name.into(),
                field: field_name.into(),
            }
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

        // Safe to unwrap: just dispatched on int types above.
        let target_bytes = encode_int_for_zone(&target_value).unwrap();
        let target_u64 = u64::from_be_bytes(target_bytes);

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
        let mut objects = Vec::new();
        for (key, data) in entries {
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
                });
            }
        }

        Ok(objects)
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
            let field_def = type_def.get_field(field_name).ok_or_else(|| {
                EngineError::FieldNotFound {
                    type_name: type_name.into(),
                    field: field_name.clone(),
                }
            })?;
            validate_value(field_def, value)?;
        }

        let key = KeyBuilder::object(type_id, object_id);
        let mut txn = self.storage.begin_txn();

        let existing_data = self.storage.get(&txn, &key)?.ok_or_else(|| {
            EngineError::ObjectNotFound {
                type_name: type_name.into(),
                object_id,
            }
        })?;

        let mut fields = deserialize_fields(&existing_data);

        // Check unique constraints for updated fields.
        for (field_name, value) in &updates {
            let field_def = type_def.get_field(field_name).unwrap();
            if field_def.is_unique() && !matches!(value, Value::Null) {
                // Remove old unique index entry if the field had a value.
                if let Some(old_value) = fields.get(field_name)
                    && !matches!(old_value, Value::Null) {
                        self.remove_unique_index(&mut txn, type_name, type_id, field_name, old_value)?;
                    }
                self.check_unique_and_insert(
                    &mut txn, type_name, type_id, field_name, value, object_id,
                )?;
            }
        }

        for (k, v) in updates {
            fields.insert(k, v);
        }

        let serialized = serialize_fields(&fields);
        self.storage.put(&mut txn, &key, serialized)?;
        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

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
        })
    }

    /// Delete an object, enforcing @on_delete policies on all inbound relationships.
    /// Cascades are recursive — if deleting A cascades to B, and B has its own
    /// cascade relationships, those are followed too.
    pub fn delete(&self, type_name: &str, object_id: u64) -> EngineResult<()> {
        let mut txn = self.storage.begin_txn();
        let mut deleted = std::collections::HashSet::new();
        // Top-level delete: verify existence (per public API contract).
        self.delete_inner(&mut txn, type_name, object_id, true, &mut deleted)?;

        let version = self.storage.commit(&mut txn).map_err(|e| match e {
            rhypedb_storage::Error::WriteConflict => EngineError::WriteConflict,
            other => EngineError::Storage(other),
        })?;

        for (del_type, del_id) in &deleted {
            self.subscriptions.publish(ChangeEvent {
                version,
                kind: ChangeKind::Delete,
                type_name: del_type.clone(),
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
    fn delete_inner(
        &self,
        txn: &mut rhypedb_storage::mvcc::Transaction,
        type_name: &str,
        object_id: u64,
        verify_exists: bool,
        deleted: &mut std::collections::HashSet<(String, u64)>,
    ) -> EngineResult<()> {
        let delete_key = (type_name.to_string(), object_id);
        if !deleted.insert(delete_key) {
            return Ok(()); // already deleted in this cascade chain
        }

        let type_id = *self
            .type_ids
            .get(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?;

        let obj_key = KeyBuilder::object(type_id, object_id);

        // Unique-index cleanup. We can skip the storage.get + deserialize
        // entirely if this type has no @unique fields — saves one LSM probe
        // per cascading row (huge at scale where most cascaded rows are
        // edge-only types like Rating).
        let type_has_unique = self.types_with_unique.contains(type_name);
        if type_has_unique || verify_exists {
            let obj_data = self.storage.get(txn, &obj_key)?;
            if obj_data.is_none() {
                if verify_exists {
                    return Err(EngineError::ObjectNotFound {
                        type_name: type_name.into(),
                        object_id,
                    });
                }
                // Cascade-recursive call against an object that's already
                // gone (e.g. a circular cascade chain). Continue silently.
            }
            if type_has_unique && let Some(data) = &obj_data {
                let fields = deserialize_fields(data);
                if let Some(type_def) = self.schema.get_type(type_name) {
                    for field_def in &type_def.fields {
                        if field_def.is_unique()
                            && let Some(value) = fields.get(&field_def.name)
                                && !matches!(value, Value::Null) {
                                    let field_key = format!("{type_name}.{}", field_def.name);
                                    let field_id = self.field_ids[&field_key];
                                    let value_bytes = value_to_index_bytes(value);
                                    let unique_key =
                                        KeyBuilder::unique_index(type_id, field_id, &value_bytes);
                                    self.storage.delete(txn, &unique_key)?;
                                }
                    }
                }
            }
        }

        // Process inbound relationships (other objects pointing at this one).
        // Uses the precomputed reverse-relation index instead of walking the
        // whole schema per call.
        let mut deny_violations = Vec::new();
        let mut edges_to_remove = Vec::new();
        let mut objects_to_cascade = Vec::new();

        if let Some(incoming) = self.incoming_relations.get(type_name) {
            for inc in incoming {
                let rev_prefix = KeyBuilder::reverse_edge_prefix(object_id, inc.rel_id);
                let reverse_edges = self.scan_prefix(txn, &rev_prefix)?;
                if reverse_edges.is_empty() {
                    continue;
                }
                for (source_id, _) in reverse_edges {
                    match inc.policy {
                        OnDeletePolicy::Deny => {
                            deny_violations
                                .push((inc.source_type.clone(), inc.source_field.clone()));
                        }
                        OnDeletePolicy::Remove => {
                            edges_to_remove.push((source_id, inc.rel_id, object_id));
                        }
                        OnDeletePolicy::Cascade => {
                            edges_to_remove.push((source_id, inc.rel_id, object_id));
                            objects_to_cascade.push((inc.source_type.clone(), source_id));
                        }
                    }
                }
            }
        }

        // Check deny violations first (before any mutations).
        if let Some((ref_type, ref_field)) = deny_violations.into_iter().next() {
            return Err(EngineError::DeleteDenied {
                type_name: type_name.into(),
                object_id,
                referencing_type: ref_type,
                referencing_field: ref_field,
            });
        }

        // Remove inbound edges.
        for (source_id, rel_id, target_id) in &edges_to_remove {
            let edge_key = KeyBuilder::edge(*source_id, *rel_id, *target_id);
            let rev_key = KeyBuilder::reverse_edge(*target_id, *rel_id, *source_id);
            self.storage.delete(txn, &edge_key)?;
            self.storage.delete(txn, &rev_key)?;
        }

        // Recursively delete cascade targets. The IDs came from a reverse-
        // edge scan we just did inside this same txn, so they provably
        // exist — skip the existence check on the recursive call.
        for (cascade_type, cascade_id) in objects_to_cascade {
            self.delete_inner(txn, &cascade_type, cascade_id, false, deleted)?;
        }

        // Delete outbound edges from this object.
        if let Some(type_def) = self.schema.get_type(type_name) {
            for field in &type_def.fields {
                if let FieldType::Relation(_) = &field.field_type {
                    let rel_key_name = format!("{type_name}.{}", field.name);
                    let rel_id = self.rel_ids[&rel_key_name];
                    let edge_prefix = KeyBuilder::edge_prefix(object_id, rel_id);
                    let outbound = self.scan_prefix(txn, &edge_prefix)?;
                    for (target_id, _) in &outbound {
                        let edge_key = KeyBuilder::edge(object_id, rel_id, *target_id);
                        let rev_key = KeyBuilder::reverse_edge(*target_id, rel_id, object_id);
                        self.storage.delete(txn, &edge_key)?;
                        self.storage.delete(txn, &rev_key)?;
                    }
                }
            }
        }

        // Delete the object itself.
        self.storage.delete(txn, &obj_key)?;

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

        let field = type_def.get_field(field_name).ok_or_else(|| {
            EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            }
        })?;

        let rel = match &field.field_type {
            FieldType::Relation(r) => r,
            _ => {
                return Err(EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                })
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

        let field = type_def.get_field(field_name).ok_or_else(|| {
            EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            }
        })?;

        let snapshot = self.storage.read_snapshot();

        // If this field has @inverse, traverse via the reverse edge index
        // of the referenced relationship.
        if let Some(inv) = field.inverse() {
            let inv_rel_key = format!("{}.{}", inv.type_name, inv.field_name);
            let inv_rel_id = *self.rel_ids.get(&inv_rel_key).ok_or_else(|| {
                EngineError::FieldNotFound {
                    type_name: inv.type_name.clone(),
                    field: inv.field_name.clone(),
                }
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
    ) -> EngineResult<Vec<Vec<(u64, FieldMap)>>> {
        if source_ids.is_empty() {
            return Ok(Vec::new());
        }

        let type_def = self
            .schema
            .get_type(source_type)
            .ok_or_else(|| EngineError::TypeNotFound(source_type.into()))?;

        let field = type_def.get_field(field_name).ok_or_else(|| {
            EngineError::FieldNotFound {
                type_name: source_type.into(),
                field: field_name.into(),
            }
        })?;

        // Resolve the relation ID once (forward or inverse) — outside the
        // per-source loop the original get_links was paying.
        let (rel_id, use_inverse) = if let Some(inv) = field.inverse() {
            let inv_rel_key = format!("{}.{}", inv.type_name, inv.field_name);
            let id = *self.rel_ids.get(&inv_rel_key).ok_or_else(|| {
                EngineError::FieldNotFound {
                    type_name: inv.type_name.clone(),
                    field: inv.field_name.clone(),
                }
            })?;
            (id, true)
        } else {
            let rel_key = format!("{source_type}.{field_name}");
            let id = *self.rel_ids.get(&rel_key).ok_or_else(|| {
                EngineError::FieldNotFound {
                    type_name: source_type.into(),
                    field: field_name.into(),
                }
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

        Ok(raw.into_iter().map(Self::decode_edge_entries).collect())
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

    /// Scan for keys with a given prefix at a snapshot version (used by the
    /// read-only fast path).
    fn scan_prefix_at(
        &self,
        snapshot: u64,
        prefix: &[u8],
    ) -> EngineResult<Vec<(u64, FieldMap)>> {
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
    // each discovered other forward target.
    let mut effective = match source_data {
        Some(bytes) => deserialize_fields(bytes),
        None => FieldMap::new(),
    };
    effective.insert(field_name.to_string(), Value::U64(target_id));
    for (name, tid) in other_targets {
        effective.insert(name, Value::U64(tid));
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

fn validate_value(
    field_def: &rhypedb_schema::FieldDef,
    value: &Value,
) -> EngineResult<()> {
    use rhypedb_schema::ScalarType;

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
        FieldType::Relation(_) | FieldType::Vector(_) => {
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

        db.link("User", alice.id, "favorite_movies", alien.id, Some(edge_props))
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

        let event = rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
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

        let (_id, rx) = db
            .subscriptions()
            .subscribe(rhypedb_subscribe::SubscriptionFilter::for_object("User", user.id));

        let mut updates = FieldMap::new();
        updates.insert("name".into(), Value::String("Bob".into()));
        db.update("User", user.id, updates).unwrap();

        let event = rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
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

        let event = rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
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

        let gt = db.filter_scan("User", "age", CompareOp::Gt, 30).unwrap();
        assert_eq!(gt.len(), 20, "age > 30 should match users with age 31..=50");
        for u in &gt {
            match u.fields.get("age") {
                Some(Value::U32(v)) => assert!(*v > 30, "stray user with age {v}"),
                other => panic!("missing/bad age: {:?}", other),
            }
        }

        let eq = db.filter_scan("User", "age", CompareOp::Eq, 25).unwrap();
        assert_eq!(eq.len(), 1);
        assert!(matches!(eq[0].fields.get("age"), Some(Value::U32(25))));

        let lt = db.filter_scan("User", "age", CompareOp::Lt, 5).unwrap();
        assert_eq!(lt.len(), 4, "age < 5 should match users 1..=4");
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

        let gt = db.filter_scan("User", "age", CompareOp::Gt, 15).unwrap();
        assert_eq!(gt.len(), 25, "should include 16..=40");
    }
}
