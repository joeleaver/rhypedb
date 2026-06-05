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
    /// Non-unique scalar secondary index: `i:<type_id>:<field_hash>:<encoded_value>:<object_id>`.
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

    /// Unique index key: `u:<type_id>:<field_name_hash>:<value_bytes>`
    /// Maps to the object_id that holds this unique value.
    pub fn unique_index(type_id: u64, field_hash: u64, value_bytes: &[u8]) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + value_bytes.len());
        buf.put_u8(KeyPrefix::Unique as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_hash);
        buf.put_u8(SEPARATOR);
        buf.put_slice(value_bytes);
        buf.freeze()
    }

    /// Secondary field index key: `i:<type_id>:<field_hash>:<encoded_value>:<object_id>`.
    /// `encoded_value` is the 8-byte byte-order-preserving encoding from the
    /// engine; `object_id` is big-endian u64. Empty value payload.
    pub fn field_index(
        type_id: u64,
        field_hash: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8 + 1 + 8);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_hash);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.put_u8(SEPARATOR);
        buf.put_u64(object_id);
        buf.freeze()
    }

    /// Prefix for scanning every entry of one type's indexed field, sorted
    /// ascending by encoded value: `i:<type_id>:<field_hash>:`.
    pub fn field_index_prefix(type_id: u64, field_hash: u64) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_hash);
        buf.put_u8(SEPARATOR);
        buf.freeze()
    }

    /// Prefix for scanning every entry of one type's indexed field matching a
    /// specific value (equality lookup): `i:<type_id>:<field_hash>:<encoded_value>:`.
    pub fn field_index_value_prefix(
        type_id: u64,
        field_hash: u64,
        encoded_value: &[u8; 8],
    ) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 1 + 8 + 1 + 8 + 1 + 8 + 1);
        buf.put_u8(KeyPrefix::FieldIndex as u8);
        buf.put_u8(SEPARATOR);
        buf.put_u64(type_id);
        buf.put_u8(SEPARATOR);
        buf.put_u64(field_hash);
        buf.put_u8(SEPARATOR);
        buf.put_slice(encoded_value);
        buf.put_u8(SEPARATOR);
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

    pub fn edge_into(
        buf: &mut Vec<u8>,
        source_id: u64,
        rel_id: u64,
        target_id: u64,
    ) -> (u32, u32) {
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

    pub fn object_version_into(
        buf: &mut Vec<u8>,
        type_id: u64,
        object_id: u64,
    ) -> (u32, u32) {
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
        field_hash: u64,
        value_bytes: &[u8],
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::Unique as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&field_hash.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(value_bytes);
        (start, buf.len() as u32)
    }

    pub fn field_index_into(
        buf: &mut Vec<u8>,
        type_id: u64,
        field_hash: u64,
        encoded_value: &[u8; 8],
        object_id: u64,
    ) -> (u32, u32) {
        let start = buf.len() as u32;
        buf.push(KeyPrefix::FieldIndex as u8);
        buf.push(SEPARATOR);
        buf.extend_from_slice(&type_id.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(&field_hash.to_be_bytes());
        buf.push(SEPARATOR);
        buf.extend_from_slice(encoded_value);
        buf.push(SEPARATOR);
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
}
