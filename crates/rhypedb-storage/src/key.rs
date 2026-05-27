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
}
