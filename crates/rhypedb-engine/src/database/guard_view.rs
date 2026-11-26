//! What a write guard may read, inside the guarded write's own transaction.
//!
//! A guarded write (`update_if` and friends, see `guarded.rs`) decides on the
//! version of the object its transaction reads. A guard that also looks at
//! OTHER stored state — the security rules engine following `resource.owner.uid`
//! through a relation, say — has to read that state in the same transaction,
//! and every key the decision rested on has to join the commit's conflict
//! check, or a concurrent change to it would not stop the write (write skew).
//!
//! [`GuardView`] is that read surface. Every read through it is recorded, and
//! when the guard accepts, the write's transaction pins each recorded key:
//!
//! * an object read (the guarded object itself, or a relation target) pins the
//!   object's `o:` key, which every update and delete of it writes;
//! * a relation read pins every edge key it saw (an unlink or a cascade delete
//!   tombstones one of them) AND the relation's edge-set token (see
//!   [`edge_set_token`]), which every link, unlink and linking create claims.
//!   The token is what catches a phantom: an added edge is a key the reader
//!   could not have seen.
//!
//! A commit that changed any of it after the write's snapshot makes the write
//! fail with `WriteConflict`, and the guarded write retries: fresh snapshot,
//! fresh reads, fresh guard decision.

use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use rhypedb_schema::FieldType;
use rhypedb_storage::key::KeyBuilder;
use rhypedb_storage::mvcc::Transaction;

use super::DatabaseCore;
use crate::error::{EngineError, EngineResult};
use crate::object::{FieldMap, deserialize_fields};

/// Leading bytes of an edge-set token. A token is never stored (see
/// [`Transaction::record_conflict_token`]), so it only has to be distinct from
/// stored keys, whose first byte is a `KeyPrefix` letter.
const EDGE_SET_TOKEN_TAG: [u8; 2] = [0x00, b'E'];
/// Direction byte: the set of edges stored under the object as SOURCE
/// (`e:<source>:<rel>:*`, read by a forward relation field).
const FORWARD: u8 = b'f';
/// Direction byte: the set of edges stored under the object as TARGET
/// (`r:<target>:<rel>:*`, read by an `@inverse` field).
const REVERSE: u8 = b'r';

/// The conflict token for one object's set of edges of relation `rel_id`, in
/// one direction. Never stored.
fn edge_set_token(direction: u8, object_id: u64, rel_id: u64) -> Bytes {
    let mut buf = BytesMut::with_capacity(EDGE_SET_TOKEN_TAG.len() + 1 + 8 + 8);
    buf.put_slice(&EDGE_SET_TOKEN_TAG);
    buf.put_u8(direction);
    buf.put_u64(object_id);
    buf.put_u64(rel_id);
    buf.freeze()
}

/// Claim the edge-set tokens a write of edge `source -[rel]-> target` changes:
/// the source's forward set and the target's reverse set. Every transaction
/// that adds or removes an edge calls this, so a guard that read either set
/// conflicts with it.
pub(super) fn claim_edge_sets(txn: &mut Transaction, source_id: u64, rel_id: u64, target_id: u64) {
    txn.record_conflict_token(edge_set_token(FORWARD, source_id, rel_id));
    txn.record_conflict_token(edge_set_token(REVERSE, target_id, rel_id));
}

/// The guarded object, and reads of other stored state, at the guarded
/// write's snapshot. See the module docs.
pub struct GuardView<'a> {
    core: &'a DatabaseCore,
    txn: &'a Transaction,
    type_name: &'a str,
    object_id: u64,
    fields: &'a FieldMap,
    /// Keys read so far, pinned into the transaction if the guard accepts.
    pins: RefCell<Vec<Bytes>>,
}

impl<'a> GuardView<'a> {
    /// The guarded object's type.
    pub fn type_name(&self) -> &str {
        self.type_name
    }

    /// The guarded object's id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// The guarded object's fields at the write's snapshot.
    pub fn fields(&self) -> &FieldMap {
        self.fields
    }

    /// The ids linked from the guarded object through `field_name` (a forward
    /// relation or an `@inverse` field), at the write's snapshot. The edge set
    /// joins the conflict check: a link, unlink or delete changing it after the
    /// snapshot makes the write retry.
    pub fn link_ids(&self, field_name: &str) -> EngineResult<Vec<u64>> {
        let (rel_id, inverse) = self.core.relation_id_for(self.type_name, field_name)?;
        let (prefix, direction) = if inverse {
            (
                KeyBuilder::reverse_edge_prefix(self.object_id, rel_id),
                REVERSE,
            )
        } else {
            (KeyBuilder::edge_prefix(self.object_id, rel_id), FORWARD)
        };
        let ids: Vec<u64> = self
            .core
            .scan_prefix_raw(self.txn, &prefix)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let mut pins = self.pins.borrow_mut();
        pins.push(edge_set_token(direction, self.object_id, rel_id));
        for &id in &ids {
            pins.push(if inverse {
                KeyBuilder::reverse_edge(self.object_id, rel_id, id)
            } else {
                KeyBuilder::edge(self.object_id, rel_id, id)
            });
        }
        Ok(ids)
    }

    /// Another object's fields at the write's snapshot (`None` if it does not
    /// exist). The object joins the conflict check: an update or delete of it
    /// after the snapshot makes the write retry.
    pub fn object_fields(&self, type_name: &str, object_id: u64) -> EngineResult<Option<FieldMap>> {
        let type_id = self.core.resolve_type_id(type_name)?;
        let key = KeyBuilder::object(type_id, object_id);
        let blob = self.core.storage.get(self.txn, &key)?;
        self.pins.borrow_mut().push(key);
        Ok(blob.map(|b| deserialize_fields(&b)))
    }
}

impl DatabaseCore {
    /// Run `guard` over the object (`fields`, read by `txn`) and, if it
    /// accepts, pin everything it read through the view into `txn`.
    pub(super) fn run_guard(
        &self,
        txn: &mut Transaction,
        (type_name, object_id): (&str, u64),
        fields: &FieldMap,
        guard: super::ObjectGuard<'_>,
    ) -> bool {
        let (accepted, pins) = {
            let view = GuardView {
                core: self,
                txn,
                type_name,
                object_id,
                fields,
                pins: RefCell::new(Vec::new()),
            };
            let accepted = guard(&view);
            (accepted, view.pins.into_inner())
        };
        if accepted {
            for key in pins {
                txn.record_read_guard(key);
            }
        }
        accepted
    }

    /// The relation id `field_name` of `type_name` reads, and whether it reads
    /// it backwards (an `@inverse` field reads the referenced relation's
    /// reverse edges).
    fn relation_id_for(&self, type_name: &str, field_name: &str) -> EngineResult<(u64, bool)> {
        let not_found = |t: &str, f: &str| EngineError::FieldNotFound {
            type_name: t.into(),
            field: f.into(),
        };
        let field = self
            .schema
            .get_type(type_name)
            .ok_or_else(|| EngineError::TypeNotFound(type_name.into()))?
            .get_field(field_name)
            .ok_or_else(|| not_found(type_name, field_name))?;
        if !matches!(field.field_type, FieldType::Relation(_)) {
            return Err(not_found(type_name, field_name));
        }
        let (rel_key, inverse) = match field.inverse() {
            Some(inv) => (format!("{}.{}", inv.type_name, inv.field_name), true),
            None => (format!("{type_name}.{field_name}"), false),
        };
        let rel_id = *self
            .rel_ids
            .get(&rel_key)
            .ok_or_else(|| not_found(type_name, field_name))?;
        Ok((rel_id, inverse))
    }
}
