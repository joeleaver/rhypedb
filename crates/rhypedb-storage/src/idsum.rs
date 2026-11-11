//! Per-SST summary of the object ids a file holds, so opening a database can
//! rebuild its live-object set without reading object data.
//!
//! The engine needs, at open, every live object id (the "born" bits behind
//! generation counters) and the highest one. Getting that by scanning the
//! object keyspace pages in the data around every key — in an object-heavy
//! store, about half of all object bytes, on every open. Instead each SST
//! records, per object type, which ids' newest entry in that file is a put
//! (`live`) and which a tombstone (`dead`). Merging those newest-source-first
//! — the same per-key precedence the read path applies across sources —
//! yields exactly the live set a full scan would.
//!
//! Only keys of the exact object-key shape `o:<type_id u64>:<object_id u64>`
//! are summarized ([`object_key_ids`]).

use std::collections::BTreeMap;

use roaring::RoaringTreemap;

use crate::error::{Error, Result};
use crate::key::{KeyPrefix, SEPARATOR};

/// Tag of the id-summary section appended to an SST meta block.
pub(crate) const META_TAG_OBJECT_IDS: u8 = 1;

/// Length of an object user key: `o` `:` type_id `:` object_id.
const OBJECT_KEY_LEN: usize = 1 + 1 + 8 + 1 + 8;

/// `(type_id, object_id)` if `user_key` is an object key, else `None`.
pub fn object_key_ids(user_key: &[u8]) -> Option<(u64, u64)> {
    if user_key.len() != OBJECT_KEY_LEN
        || user_key[0] != KeyPrefix::Object as u8
        || user_key[1] != SEPARATOR
        || user_key[10] != SEPARATOR
    {
        return None;
    }
    let type_id = u64::from_be_bytes(user_key[2..10].try_into().unwrap());
    let object_id = u64::from_be_bytes(user_key[11..19].try_into().unwrap());
    Some((type_id, object_id))
}

/// One type's ids in one source.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TypeIds {
    /// Ids whose newest entry in the source is a put.
    pub live: RoaringTreemap,
    /// Ids whose newest entry in the source is a tombstone.
    pub dead: RoaringTreemap,
}

/// Object ids of one source (an SST or a memtable), per type id.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ObjectIdSummary {
    pub types: BTreeMap<u64, TypeIds>,
}

impl ObjectIdSummary {
    /// Record the NEWEST entry of an object key in this source. A later call
    /// for an id already recorded is ignored (the first call is the newest:
    /// sources yield a key's versions newest first).
    pub fn record(&mut self, type_id: u64, object_id: u64, is_put: bool) {
        let t = self.types.entry(type_id).or_default();
        if t.live.contains(object_id) || t.dead.contains(object_id) {
            return;
        }
        if is_put {
            t.live.insert(object_id);
        } else {
            t.dead.insert(object_id);
        }
    }

    /// Serialized form: `[n_types u32]` then per type
    /// `[type_id u64][live_len u32][live][dead_len u32][dead]`, each bitmap in
    /// the portable `RoaringTreemap` format.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.types.len() as u32).to_be_bytes());
        for (type_id, ids) in &self.types {
            out.extend_from_slice(&type_id.to_be_bytes());
            for map in [&ids.live, &ids.dead] {
                let mut buf = Vec::with_capacity(map.serialized_size());
                map.serialize_into(&mut buf).expect("write to Vec");
                out.extend_from_slice(&(buf.len() as u32).to_be_bytes());
                out.extend_from_slice(&buf);
            }
        }
        out
    }

    pub(crate) fn decode(mut b: &[u8]) -> Result<Self> {
        let corrupt = || Error::SstCorrupted("object id summary malformed".into());
        let take = |n: usize, b: &mut &[u8]| -> Result<Vec<u8>> {
            if b.len() < n {
                return Err(corrupt());
            }
            let (head, rest) = b.split_at(n);
            *b = rest;
            Ok(head.to_vec())
        };
        let n = u32::from_be_bytes(take(4, &mut b)?.try_into().unwrap()) as usize;
        let mut types = BTreeMap::new();
        for _ in 0..n {
            let type_id = u64::from_be_bytes(take(8, &mut b)?.try_into().unwrap());
            let mut maps = [RoaringTreemap::new(), RoaringTreemap::new()];
            for map in &mut maps {
                let len = u32::from_be_bytes(take(4, &mut b)?.try_into().unwrap()) as usize;
                let bytes = take(len, &mut b)?;
                *map = RoaringTreemap::deserialize_from(&bytes[..]).map_err(|_| corrupt())?;
            }
            let [live, dead] = maps;
            if types.insert(type_id, TypeIds { live, dead }).is_some() {
                return Err(corrupt());
            }
        }
        if !b.is_empty() {
            return Err(corrupt());
        }
        Ok(Self { types })
    }
}

/// Folds source summaries, NEWEST source first, into the live set: an id is
/// live iff the newest source that mentions it holds a put for it.
#[derive(Default)]
pub struct LiveIdMerge {
    live: BTreeMap<u64, RoaringTreemap>,
    decided: BTreeMap<u64, RoaringTreemap>,
}

impl LiveIdMerge {
    /// Apply the next-older source.
    pub fn apply(&mut self, source: &ObjectIdSummary) {
        for (type_id, ids) in &source.types {
            let decided = self.decided.entry(*type_id).or_default();
            let fresh = &ids.live - &*decided;
            *decided |= &ids.live;
            *decided |= &ids.dead;
            *self.live.entry(*type_id).or_default() |= fresh;
        }
    }

    /// The live ids per type (types with none left out).
    pub fn finish(self) -> BTreeMap<u64, RoaringTreemap> {
        self.live.into_iter().filter(|(_, m)| !m.is_empty()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::KeyBuilder;

    #[test]
    fn object_key_shape() {
        assert_eq!(object_key_ids(&KeyBuilder::object(7, 42)), Some((7, 42)));
        assert_eq!(object_key_ids(&KeyBuilder::edge(1, 2, 3)), None);
        let mut long = KeyBuilder::object(7, 42).to_vec();
        long.push(0);
        assert_eq!(object_key_ids(&long), None);
    }

    #[test]
    fn encode_roundtrip_and_rejects_garbage() {
        let mut s = ObjectIdSummary::default();
        s.record(3, 10, true);
        s.record(3, 10, false); // older entry of the same id: ignored
        s.record(3, 11, false);
        s.record(9, u64::MAX - 1, true);
        let d = ObjectIdSummary::decode(&s.encode()).unwrap();
        assert_eq!(d, s);
        assert!(d.types[&3].live.contains(10) && !d.types[&3].dead.contains(10));
        assert!(ObjectIdSummary::decode(&s.encode()[..5]).is_err());
        let mut extra = s.encode();
        extra.push(0);
        assert!(ObjectIdSummary::decode(&extra).is_err());
    }

    #[test]
    fn newest_source_decides() {
        let mut newer = ObjectIdSummary::default();
        newer.record(1, 5, false); // deleted after the older put
        newer.record(1, 6, true);
        let mut older = ObjectIdSummary::default();
        older.record(1, 5, true);
        older.record(1, 6, false);
        older.record(1, 7, true);
        older.record(2, 8, false);
        let mut m = LiveIdMerge::default();
        m.apply(&newer);
        m.apply(&older);
        let live = m.finish();
        assert_eq!(live[&1].iter().collect::<Vec<_>>(), vec![6, 7]);
        assert!(!live.contains_key(&2));
    }
}
