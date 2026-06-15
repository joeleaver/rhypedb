use bytes::{BufMut, Bytes, BytesMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyPrefix {
    Object = b'o',
    Edge = b'e',
    ReverseEdge = b'r',
    Vector = b'v',
    Unique = b'u',
    Queue = b'q',
    VectorState = b's',
    /// Non-unique scalar secondary index: `i:<type_id>:<field_id>:<encoded_value>:<object_id>`.
    /// Empty value payload — the data is the key itself. Encoded value uses
    /// the same byte-order-preserving rules as zone maps so a prefix scan
    /// returns ids in ascending field-value order.
    FieldIndex = b'i',
    /// Per-object monotonic version counter: `g:<type_id>:<object_id>` →
    /// 8-byte big-endian u64. Bumped on every successful `Database::update`.
    /// The cover writer stamps each `<name>__cover` entry with the target's
    /// generation-at-write-time; the reader (executor fusion) compares
    /// against the current generation to detect stale covers and fall
    /// through to a fresh LSM probe for those targets.
    Generation = b'g',
    /// Persisted schema catalog: `c:<subtype>:<…>`. The catalog stores the
    /// mapping from schema name to stable numeric ID for every type, field,
    /// and relation in the schema. IDs are assigned exactly once (at first
    /// open) and persisted so that schema edits (adding a type, renaming a
    /// field) never silently renumber existing storage keys.
    /// See `rhypedb-engine/src/catalog.rs` for the subtype tag table
    /// (`F` format, `I` initialized, `M` metadata, `D` digest, `T` type,
    /// `E` field, `R` relation, `N` counter) and value-encoding format.
    Catalog = b'c',
}

pub const SEPARATOR: u8 = b':';

/// Encodes an internal key with a version suffix for MVCC.
///
/// Format: `<user_key><version_u64_be>`
///
/// The version is stored as big-endian u64 with bits inverted so that
/// higher versions sort first within the same user key. This lets a
/// forward prefix scan hit the newest version first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalKey {
    data: Bytes,
}

impl InternalKey {
    pub fn new(user_key: &[u8], version: u64) -> Self {
        let mut buf = BytesMut::with_capacity(user_key.len() + 8);
        buf.put_slice(user_key);
        buf.put_u64(!version);
        Self { data: buf.freeze() }
    }

    pub fn user_key(&self) -> &[u8] {
        &self.data[..self.data.len() - 8]
    }

    pub fn version(&self) -> u64 {
        let ver_bytes: [u8; 8] = self.data[self.data.len() - 8..].try_into().unwrap();
        !u64::from_be_bytes(ver_bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the wrapper and return the inner `Bytes` — lets callers
    /// hand off the owned allocation (e.g. memtable insert) without a
    /// `Bytes::copy_from_slice` second allocation.
    pub fn into_bytes(self) -> Bytes {
        self.data
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Builds user-visible keys for the different stores.
pub struct KeyBuilder;

impl KeyBuilder {
    /// Object key: `o:<type_id>:<object_id>`
    pub fn object(type_id: u64, object_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::Object as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.freeze()
    }

    /// Edge key: `e:<source_id>:<rel_id>:<target_id>`
    pub fn edge(source_id: u64, rel_id: u64, target_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::Edge as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(source_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(rel_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(target_id);
        buf.freeze()
    }

    /// Reverse edge key: `r:<target_id>:<rel_id>:<source_id>`
    pub fn reverse_edge(target_id: u64, rel_id: u64, source_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::ReverseEdge as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(target_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(rel_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(source_id);
        buf.freeze()
    }

    /// Vector key: `v:<type_id>:<object_id>:<field_id>`
    pub fn vector(type_id: u64, object_id: u64, field_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::Vector as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.freeze()
    }

    /// Object prefix for scanning all objects of a type: `o:<type_id>:`
    pub fn object_prefix(type_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::Object as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Edge prefix for scanning all edges from a source on a relationship: `e:<source_id>:<rel_id>:`
    pub fn edge_prefix(source_id: u64, rel_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::Edge as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(source_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(rel_id);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Reverse edge prefix: `r:<target_id>:<rel_id>:`
    pub fn reverse_edge_prefix(target_id: u64, rel_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::ReverseEdge as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(target_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(rel_id);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Vector prefix for scanning all vectors of a type: `v:<type_id>:`
    /// Returns all vectors across all fields for this type. The caller
    /// filters by field_id (encoded in the last 8 bytes of each key).
    pub fn vector_prefix(type_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::Vector as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Unique index key: `u:<type_id>:<field_id>:<value_bytes>`
    /// Maps to the object_id that holds this unique value.
    pub fn unique_index(type_id: u64, field_id: u64, value_bytes: &[u8]) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + value_bytes.len());
        buf.put_u8(KeyPrefix::Unique as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.put_slice(value_bytes);
        buf.freeze()
    }

    /// Secondary field index key: `i:<type_id>:<field_id>:<encoded_value>:<object_id>`.
    /// `encoded_value` is the 8-byte byte-order-preserving encoding from the
    /// engine; `object_id` is big-endian u64. Empty value payload.
    pub fn field_index(
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.freeze()
    }

    /// Prefix for scanning every entry of one type's indexed field, sorted
    /// ascending by encoded value: `i:<type_id>:<field_id>:`.
    pub fn field_index_prefix(type_id: u64, field_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Prefix for scanning every entry of one type's indexed field matching a
    /// specific value (equality lookup): `i:<type_id>:<field_id>:<encoded_value>:`.
    pub fn field_index_value_prefix(
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8; 8],
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Secondary field index key for variable-length encoded values:
    /// `i:<type_id>:<field_id>:<encoded_value><object_id>`.
    ///
    /// `encoded_value` is the caller-supplied sort-preserving encoding which
    /// MUST embed its own end-of-value marker (e.g. the engine's string
    /// encoder appends `\x00\x00`). No `:` separator sits between the
    /// encoded value and the object_id — the embedded terminator already
    /// disambiguates the value's end. Empty value payload, same as the
    /// fixed-width variant; sorts ascending by encoded value then object_id.
    pub fn field_index_var(
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8],
        object_id: u64,
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + encoded_value.len() + 8);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.put_u64(object_id);
        buf.freeze()
    }

    /// Equality-prefix for the variable-length secondary index:
    /// `i:<type_id>:<field_id>:<encoded_value>`. Used for `Eq` lookups —
    /// matches every key whose encoded value equals the supplied bytes,
    /// regardless of object_id. Because `encoded_value` carries its own
    /// terminator, this prefix can't be confused with a longer value that
    /// shares the same starting bytes.
    pub fn field_index_var_value_prefix(
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8],
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + encoded_value.len());
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.freeze()
    }

    /// Vectorization queue entry: `q:<job_id>`
    /// Value contains the serialized job (type, object_id, source field, vector field, model).
    pub fn queue_entry(job_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8);
        buf.put_u8(KeyPrefix::Queue as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(job_id);
        buf.freeze()
    }

    /// Queue prefix for scanning all pending jobs: `q:`
    pub fn queue_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(2);
        buf.put_u8(KeyPrefix::Queue as u8);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Vector state key: `s:<type_id>:<object_id>:<field_id>`
    /// Value: state byte (0=pending, 1=indexed, 2=failed)
    pub fn vector_state(type_id: u64, object_id: u64, field_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::VectorState as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_id);
        buf.freeze()
    }

    /// Per-object generation counter: `g:<type_id>:<object_id>` → u64 BE.
    pub fn object_version(type_id: u64, object_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::Generation as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.freeze()
    }

    // ---------------------------------------------------------------
    // Arena-style key encoders for the cascade-delete hot path.
    //
    // Hot loops that produce ~500 tombstone keys per User-delete used to
    // do one `BytesMut::with_capacity` malloc per key. The `*_into`
    // variants append to a single caller-owned `Vec<u8>` and return the
    // (start, end) range of the bytes they wrote — the caller wraps the
    // buffer in `Bytes::from` once and `.slice(range)` each key (refcount
    // only, no further allocations). Drops ~500 small mallocs per
    // User-delete at K=100.
    // ---------------------------------------------------------------

    pub fn object_into(buf: &mut Vec<u8>, type_id: u64, object_id: u64) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::Object as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&object_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    pub fn edge_into(buf: &mut Vec<u8>, source_id: u64, rel_id: u64, target_id: u64) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::Edge as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&source_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&rel_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&target_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    pub fn reverse_edge_into(
        buf: &mut Vec<u8>,
        target_id: u64,
        rel_id: u64,
        source_id: u64,
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::ReverseEdge as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&target_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&rel_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&source_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    pub fn object_version_into(buf: &mut Vec<u8>, type_id: u64, object_id: u64) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::Generation as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&object_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    pub fn unique_index_into(
        buf: &mut Vec<u8>,
        type_id: u64,
        field_id: u64,
        value_bytes: &[u8],
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::Unique as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&field_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(value_bytes);
        (start, buf.len() as u32)
    }

    pub fn field_index_into(
        buf: &mut Vec<u8>,
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::FieldIndex as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&field_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(encoded_value);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&object_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    /// Arena-style variant of `field_index_var`. Layout matches the owned
    /// constructor exactly: `i:<type>:<field_id>:<encoded_value><object_id>`
    /// — no separator between the value and id (the caller-supplied
    /// terminator inside `encoded_value` already disambiguates the
    /// boundary).
    pub fn field_index_var_into(
        buf: &mut Vec<u8>,
        type_id: u64,
        field_id: u64,
        encoded_value: &[u8],
        object_id: u64,
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::FieldIndex as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&field_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(encoded_value);
        buf.extend_from_slice(&object_id.to_be_bytes());
        (start, buf.len() as u32)
    }

    /// Prefix for scanning every per-object generation counter: `g:`.
    /// Used at `Database::open` to repopulate the in-memory counter map.
    pub fn object_version_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(2);
        buf.put_u8(KeyPrefix::Generation as u8);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    // -------------------------------------------------------------------
    // Persisted schema catalog keys. Subtype is a single ASCII byte to
    // keep the keyspace tight; per-entry fields use `\x00` as the inner
    // separator since the schema validator rejects `\x00` and `:` in
    // identifier names. See `rhypedb-engine/src/catalog.rs`.
    // -------------------------------------------------------------------

    /// Catalog format-version sentinel: `c:F:` → 8-byte u64 BE.
    /// Read first on every open before any other catalog decode; a
    /// future-version binary refuses an unknown format cleanly.
    pub fn catalog_format() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'F');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Catalog initialized marker: `c:I:` → 10-byte marker value
    /// (record header + unix-millis). Presence indicates the catalog is
    /// whole; absence (with other `c:` rows present) signals a torn write.
    pub fn catalog_initialized() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'I');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Catalog metadata header: `c:M:` → TLV body (reserved for future
    /// per-deploy state and capability bits; phase 1 leaves this absent).
    pub fn catalog_metadata() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'M');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Schema digest: `c:D:` → 32 bytes SHA-256 over canonical schema
    /// shape. Drives the fast-path reconcile skip on unchanged schemas.
    pub fn catalog_digest() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'D');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Type catalog entry: `c:T:<type_name>` → encoded `IdEntry`.
    pub fn catalog_type(type_name: &str) -> Bytes {
        debug_assert!(
            !type_name.as_bytes().contains(&0) && !type_name.as_bytes().contains(&SEPARATOR),
            "type name must not contain NUL or ':'"
        );
        let mut buf = BytesMut::with_capacity(3 + type_name.len());
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'T');
        buf.put_u8(SEPARATOR);
        buf.put_slice(type_name.as_bytes());
        buf.freeze()
    }

    /// Field catalog entry: `c:E:<type>\x00<field>` → encoded `IdEntry`.
    /// `E` is for "fiEld" — `F` is taken by the format-version sentinel.
    pub fn catalog_field(type_name: &str, field_name: &str) -> Bytes {
        debug_assert!(
            !type_name.as_bytes().contains(&0) && !type_name.as_bytes().contains(&SEPARATOR),
            "type name must not contain NUL or ':'"
        );
        debug_assert!(
            !field_name.as_bytes().contains(&0) && !field_name.as_bytes().contains(&SEPARATOR),
            "field name must not contain NUL or ':'"
        );
        let mut buf = BytesMut::with_capacity(4 + type_name.len() + field_name.len());
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'E');
        buf.put_u8(SEPARATOR);
        buf.put_slice(type_name.as_bytes());
        buf.put_u8(0);
        buf.put_slice(field_name.as_bytes());
        buf.freeze()
    }

    /// Relation catalog entry: `c:R:<type>\x00<field>` → encoded `IdEntry`.
    pub fn catalog_rel(type_name: &str, field_name: &str) -> Bytes {
        debug_assert!(
            !type_name.as_bytes().contains(&0) && !type_name.as_bytes().contains(&SEPARATOR),
            "type name must not contain NUL or ':'"
        );
        debug_assert!(
            !field_name.as_bytes().contains(&0) && !field_name.as_bytes().contains(&SEPARATOR),
            "field name must not contain NUL or ':'"
        );
        let mut buf = BytesMut::with_capacity(4 + type_name.len() + field_name.len());
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'R');
        buf.put_u8(SEPARATOR);
        buf.put_slice(type_name.as_bytes());
        buf.put_u8(0);
        buf.put_slice(field_name.as_bytes());
        buf.freeze()
    }

    /// Next-type-id counter: `c:N:T` → encoded counter value.
    pub fn catalog_next_type() -> Bytes {
        let mut buf = BytesMut::with_capacity(5);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'N');
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'T');
        buf.freeze()
    }

    /// Next-field-id counter: `c:N:E`.
    pub fn catalog_next_field() -> Bytes {
        let mut buf = BytesMut::with_capacity(5);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'N');
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'E');
        buf.freeze()
    }

    /// Next-relation-id counter: `c:N:R`.
    pub fn catalog_next_rel() -> Bytes {
        let mut buf = BytesMut::with_capacity(5);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'N');
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'R');
        buf.freeze()
    }

    /// Scan prefix for all type catalog entries: `c:T:`.
    pub fn catalog_prefix_type() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'T');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Scan prefix for all field catalog entries: `c:E:`.
    pub fn catalog_prefix_field() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'E');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Scan prefix for all relation catalog entries: `c:R:`.
    pub fn catalog_prefix_rel() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'R');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Scan prefix for the entire catalog keyspace: `c:`. Used by the
    /// torn-write recovery path to clear partial state before re-running
    /// deterministic backfill, and by debug dumps.
    pub fn catalog_prefix_all() -> Bytes {
        let mut buf = BytesMut::with_capacity(2);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Current applied migration count: `c:V:` → 8-byte u64 BE.
    /// Used by the migration-log driver (card 5/5): the next migration
    /// to apply is at ordinal `current_version`. Absent before any
    /// migration has been recorded; treated as 0 in that case.
    pub fn catalog_migration_version() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'V');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Migration log entry for ordinal `n` (0-indexed):
    /// `c:G:<u64 BE ordinal>` → migration record value (utf8 name +
    /// u64 BE applied_at_unix_ms; framing documented in
    /// `rhypedb-engine/src/catalog.rs`).
    pub fn catalog_migration_log(ordinal: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(3 + 8);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'G');
        buf.put_u8(SEPARATOR);
        buf.put_u64(ordinal);
        buf.freeze()
    }

    /// Scan prefix for all migration log entries: `c:G:`.
    pub fn catalog_migration_log_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(3);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'G');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Next-migration-id counter: `c:N:M` (parallel to `c:N:T/E/R`).
    /// Hands out monotonic `plan_id`s for chunked field-type migrations
    /// (shadow-field card 1/5). Absent before any migration is created
    /// (treated as 0 then) and self-healed at allocation time against the
    /// highest `c:P:` plan id, so a torn counter bump cannot reissue a
    /// live id. Distinct from the `c:M:` metadata header (3rd byte differs).
    pub fn catalog_next_migration() -> Bytes {
        let mut buf = BytesMut::with_capacity(5);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'N');
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'M');
        buf.freeze()
    }

    /// Migration plan record: `c:P:<u64 BE plan_id>` → encoded
    /// `MigrationPlan` (record header + TLV body; framing documented in
    /// `rhypedb-engine/src/catalog.rs`). One row per chunked field-type
    /// migration; survives restart to drive auto-resume.
    pub fn catalog_migration_plan(plan_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + 8);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'P');
        buf.put_u8(SEPARATOR);
        buf.put_u64(plan_id);
        buf.freeze()
    }

    /// Scan prefix for all migration plan records: `c:P:`. Used by
    /// auto-resume, the migration-id self-heal, and the recovery-clear
    /// exemption (these rows must NOT be wiped by torn-write recovery).
    pub fn catalog_migration_plan_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(4);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'P');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Per-partition migration cursor: `c:S:<u64 BE plan_id><u8 partition_idx>`
    /// (shadow-field card 3/5). Each parallel backfill worker advances ONLY its
    /// own partition's cursor, so workers never write the shared `c:P:` plan
    /// record on the hot path (which would WriteConflict every chunk). Like
    /// `c:P:`, these rows are NOT schema-derived and MUST be exempted from the
    /// torn-write recovery clear. Third byte `S` is distinct from every other
    /// `c:` subkey (P/T/E/R/F/I/M/D/N).
    pub fn catalog_partition_cursor(plan_id: u64, partition_idx: u8) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + 8 + 1);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'S');
        buf.put_u8(SEPARATOR);
        buf.put_u64(plan_id);
        buf.put_u8(partition_idx);
        buf.freeze()
    }

    /// Scan prefix for all partition cursors of ONE plan:
    /// `c:S:<u64 BE plan_id>`. Used to load a plan's partition state on resume
    /// and to delete them on a terminal cutover/cancel.
    pub fn catalog_partition_cursor_plan_prefix(plan_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + 8);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'S');
        buf.put_u8(SEPARATOR);
        buf.put_u64(plan_id);
        buf.freeze()
    }

    /// Broad scan prefix for EVERY partition cursor: `c:S:`. Used only by the
    /// torn-write recovery clear to exempt this keyspace (see
    /// `catalog_migration_plan_prefix`).
    pub fn catalog_partition_cursor_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(4);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'S');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Quarantine sidecar row: `c:Q:<u64 BE plan_id><u64 BE object_id>`
    /// (shadow-field card 4/5). Records a row whose converter failed under the
    /// Quarantine policy so the operator can triage + retry. Like `c:P:`/`c:S:`,
    /// NOT schema-derived → MUST be exempted from the torn-write recovery clear.
    /// Third byte `Q` is distinct from every other `c:` subkey.
    pub fn catalog_quarantine(plan_id: u64, object_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + 8 + 8);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'Q');
        buf.put_u8(SEPARATOR);
        buf.put_u64(plan_id);
        buf.put_u64(object_id);
        buf.freeze()
    }

    /// Scan prefix for all quarantine rows of ONE plan: `c:Q:<u64 BE plan_id>`.
    /// Used by `list_quarantined` / the cutover gate / `clear_quarantine`.
    pub fn catalog_quarantine_plan_prefix(plan_id: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + 8);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'Q');
        buf.put_u8(SEPARATOR);
        buf.put_u64(plan_id);
        buf.freeze()
    }

    /// Broad scan prefix for EVERY quarantine row: `c:Q:`. Used only by the
    /// torn-write recovery clear to exempt this keyspace.
    pub fn catalog_quarantine_prefix() -> Bytes {
        let mut buf = BytesMut::with_capacity(4);
        buf.put_u8(KeyPrefix::Catalog as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u8(b'Q');
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_key_roundtrip() {
        let user_key = KeyBuilder::object(1, 42);
        let ik = InternalKey::new(&user_key, 100);
        assert_eq!(ik.user_key(), &user_key[..]);
        assert_eq!(ik.version(), 100);
    }

    #[test]
    fn internal_key_ordering() {
        let user_key = KeyBuilder::object(1, 42);
        let v10 = InternalKey::new(&user_key, 10);
        let v20 = InternalKey::new(&user_key, 20);
        // Higher version should sort FIRST (lower bytes due to bit inversion)
        assert!(v20 < v10);
    }

    #[test]
    fn different_user_keys_sort_by_prefix() {
        let k1 = InternalKey::new(&KeyBuilder::object(1, 1), 5);
        let k2 = InternalKey::new(&KeyBuilder::object(1, 2), 5);
        assert!(k1 < k2);
    }

    #[test]
    fn edge_key_structure() {
        let key = KeyBuilder::edge(10, 20, 30);
        assert_eq!(key[0], b'e');
        assert_eq!(key[1], b':');
    }

    #[test]
    fn prefix_is_prefix_of_full_key() {
        let prefix = KeyBuilder::edge_prefix(10, 20);
        let full = KeyBuilder::edge(10, 20, 30);
        assert!(full.starts_with(&prefix));
    }

    #[test]
    fn field_index_key_structure() {
        let key = KeyBuilder::field_index(1, 0xdead_beef, &[0; 8], 42);
        assert_eq!(key[0], b'i');
        assert_eq!(key[1], b':');
    }

    #[test]
    fn field_index_prefix_is_prefix_of_full_key() {
        let prefix = KeyBuilder::field_index_prefix(1, 0xdead);
        let full = KeyBuilder::field_index(1, 0xdead, &[1, 2, 3, 4, 5, 6, 7, 8], 99);
        assert!(full.starts_with(&prefix));
    }

    #[test]
    fn field_index_value_prefix_is_prefix_of_full_key() {
        let value: [u8; 8] = [9, 8, 7, 6, 5, 4, 3, 2];
        let prefix = KeyBuilder::field_index_value_prefix(7, 0xbeef, &value);
        let full = KeyBuilder::field_index(7, 0xbeef, &value, 1234);
        assert!(full.starts_with(&prefix));
    }

    #[test]
    fn field_index_var_prefix_is_prefix_of_full_key() {
        let value = b"hello\x00\x00";
        let prefix = KeyBuilder::field_index_var_value_prefix(1, 0xdead, value);
        let full = KeyBuilder::field_index_var(1, 0xdead, value, 99);
        assert!(full.starts_with(&prefix));
        // Object id occupies the last 8 bytes — no separator before it.
        let id_bytes: [u8; 8] = full[full.len() - 8..].try_into().unwrap();
        assert_eq!(u64::from_be_bytes(id_bytes), 99);
    }

    #[test]
    fn field_index_var_keys_sort_by_value() {
        // Encoded "abc" < encoded "abd" by byte-wise comparison.
        let lo = KeyBuilder::field_index_var(1, 0xfeed, b"abc\x00\x00", 100);
        let hi = KeyBuilder::field_index_var(1, 0xfeed, b"abd\x00\x00", 5);
        assert!(lo < hi);
    }

    #[test]
    fn field_index_var_value_prefix_disambiguates_via_terminator() {
        // "abc\x00\x00" prefix must NOT match "abcd\x00\x00" — the embedded
        // terminator is the disambiguator (without it, "abc" would prefix
        // every value beginning with "abc").
        let prefix = KeyBuilder::field_index_var_value_prefix(1, 0xfeed, b"abc\x00\x00");
        let other = KeyBuilder::field_index_var(1, 0xfeed, b"abcd\x00\x00", 99);
        assert!(!other.starts_with(&prefix));
    }

    #[test]
    fn field_index_keys_sort_by_value_then_id() {
        // Same type+field, two different encoded values: the lower value's
        // key must sort before the higher value's, regardless of id.
        let lo = KeyBuilder::field_index(1, 0xfeed, &10u64.to_be_bytes(), 100);
        let hi = KeyBuilder::field_index(1, 0xfeed, &20u64.to_be_bytes(), 5);
        assert!(lo < hi);

        // Same encoded value, two different ids: ascending by id.
        let id1 = KeyBuilder::field_index(1, 0xfeed, &10u64.to_be_bytes(), 1);
        let id2 = KeyBuilder::field_index(1, 0xfeed, &10u64.to_be_bytes(), 2);
        assert!(id1 < id2);
    }

    #[test]
    fn catalog_header_keys_have_expected_bytes() {
        assert_eq!(&KeyBuilder::catalog_format()[..], b"c:F:");
        assert_eq!(&KeyBuilder::catalog_initialized()[..], b"c:I:");
        assert_eq!(&KeyBuilder::catalog_metadata()[..], b"c:M:");
        assert_eq!(&KeyBuilder::catalog_digest()[..], b"c:D:");
        assert_eq!(&KeyBuilder::catalog_next_type()[..], b"c:N:T");
        assert_eq!(&KeyBuilder::catalog_next_field()[..], b"c:N:E");
        assert_eq!(&KeyBuilder::catalog_next_rel()[..], b"c:N:R");
    }

    #[test]
    fn catalog_type_field_rel_keys_have_expected_bytes() {
        let t = KeyBuilder::catalog_type("User");
        assert_eq!(&t[..], b"c:T:User");

        let f = KeyBuilder::catalog_field("User", "name");
        assert_eq!(&f[..4], b"c:E:");
        assert_eq!(&f[4..8], b"User");
        assert_eq!(f[8], 0u8);
        assert_eq!(&f[9..], b"name");

        let r = KeyBuilder::catalog_rel("User", "friends");
        assert_eq!(&r[..4], b"c:R:");
        assert_eq!(&r[4..8], b"User");
        assert_eq!(r[8], 0u8);
        assert_eq!(&r[9..], b"friends");
    }

    #[test]
    fn migration_plan_and_counter_keys_have_expected_bytes() {
        // Counter `c:N:M` is distinct from the `c:M:` metadata header and
        // from the other `c:N:*` counters.
        assert_eq!(&KeyBuilder::catalog_next_migration()[..], b"c:N:M");
        assert_ne!(
            &KeyBuilder::catalog_next_migration()[..],
            &KeyBuilder::catalog_metadata()[..]
        );

        // Plan record `c:P:<u64 BE>`; prefix `c:P:` is a true prefix.
        let p = KeyBuilder::catalog_migration_plan(0x0102_0304_0506_0708);
        assert_eq!(&p[..4], b"c:P:");
        assert_eq!(&p[4..], &0x0102_0304_0506_0708u64.to_be_bytes());
        let prefix_p = KeyBuilder::catalog_migration_plan_prefix();
        assert_eq!(&prefix_p[..], b"c:P:");
        assert!(p.starts_with(&prefix_p));
        assert!(prefix_p.starts_with(&KeyBuilder::catalog_prefix_all()));
        // BE plan-id ordering = numeric ordering (scan returns ascending ids).
        assert!(KeyBuilder::catalog_migration_plan(1) < KeyBuilder::catalog_migration_plan(2));
    }

    #[test]
    fn catalog_prefixes_match_their_entries() {
        let prefix_t = KeyBuilder::catalog_prefix_type();
        let prefix_f = KeyBuilder::catalog_prefix_field();
        let prefix_r = KeyBuilder::catalog_prefix_rel();
        let prefix_all = KeyBuilder::catalog_prefix_all();

        assert!(KeyBuilder::catalog_type("User").starts_with(&prefix_t));
        assert!(KeyBuilder::catalog_field("User", "name").starts_with(&prefix_f));
        assert!(KeyBuilder::catalog_rel("User", "friends").starts_with(&prefix_r));

        assert!(prefix_t.starts_with(&prefix_all));
        assert!(prefix_f.starts_with(&prefix_all));
        assert!(prefix_r.starts_with(&prefix_all));
        assert!(KeyBuilder::catalog_format().starts_with(&prefix_all));
    }

    #[test]
    fn catalog_field_and_rel_disambiguate_via_nul() {
        // The `\x00` separator means `c:E:Use\x00rname` and `c:E:User\x00name`
        // are distinct keys even though both could exist if we used `.` as
        // the separator (since `.` is allowed in user identifier territory).
        let a = KeyBuilder::catalog_field("Use", "rname");
        let b = KeyBuilder::catalog_field("User", "name");
        assert_ne!(a, b);
    }
}
