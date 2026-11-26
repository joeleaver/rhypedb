//! The authoritative write check: the security rules evaluated INSIDE each object's write
//! transaction.
//!
//! A write's rule (`update`, `delete`; `link`/`unlink` are gated as `update` on the source) reads
//! stored state: the object's own fields (`resource.ownerUid`), the edges of its relations
//! (`resource.editors`), and scalar fields of the objects those edges lead to
//! (`resource.editors.uid`). Checking that on a separate read before the write left a
//! time-of-check-to-time-of-use gap: an ownership change committed between the check and the
//! write let a write land that the rules no longer allowed.
//!
//! So every gated write goes through the engine's guarded API (`update_if`, `delete_if`,
//! `link_if`, `unlink_if`), and [`WriteGate::allows`] is (part of) its guard. The engine runs the
//! guard inside the object's write transaction, on the blob that write is based on, and every read
//! the rule makes goes through the engine's [`GuardView`], at the same snapshot:
//!
//! * the object itself — written by the update/delete, pinned for link/unlink;
//! * each relation edge set the rule follows — every edge key read, and the set's conflict token
//!   (an edge added concurrently is a key no reader saw);
//! * each linked object whose fields the rule reads.
//!
//! Any of them changed by a commit after the snapshot fails the write's commit, and the engine
//! retries from a fresh snapshot, re-evaluating the rule. A denial stages nothing: the guard runs
//! before the write stages anything, and a rejected attempt is never committed. The decision and
//! the write are therefore one atomic step per object.
//!
//! `create` needs none of this: its rule sees only the incoming fields (`request.*`; there is no
//! stored `resource` yet, so relation projections are absent), which no other writer can change.

use std::cell::Cell;

use rhypedb_authz::{Op, ResourceAccessor, RulesProgram};
use rhypedb_engine::database::GuardView;
use rhypedb_engine::object::{FieldMap, value_to_query_json};

use super::{denied, gate_write, relation_target, within_rule_budget};
use crate::error::QueryResult;
use crate::executor::ExecContext;

/// The rules check of one write step, run per object inside its write transaction. Built by
/// [`prepare`](Self::prepare) only when a rules program is configured.
pub(crate) struct WriteGate<'g> {
    ctx: &'g ExecContext<'g>,
    rules: &'g RulesProgram,
    op: Op,
    request: Option<&'g FieldMap>,
    /// Whether the LAST attempt's evaluation denied. Read by [`verdict`](Self::verdict) after the
    /// engine call returns; only a final, uncommitted attempt can leave it set.
    denied: Cell<bool>,
    /// Test hook: run once, on the first evaluation, after the attempt's snapshot was taken.
    #[cfg(test)]
    race: std::cell::RefCell<Option<Box<dyn FnOnce() + 'g>>>,
}

impl<'g> WriteGate<'g> {
    /// The gate for a write step `op` over `ids` of `type_name` (`request` = the incoming update
    /// fields), or `None` when rules are off. Runs the early, snapshot check first (see
    /// [`gate_write`]), so a write the rules refuse outright is refused before ANY row is written.
    pub(crate) fn prepare(
        ctx: &'g ExecContext<'g>,
        op: Op,
        type_name: &str,
        ids: &[u64],
        request: Option<&'g FieldMap>,
    ) -> QueryResult<Option<Self>> {
        let Some(rules) = ctx.rules.as_deref() else {
            return Ok(None);
        };
        gate_write(ctx, op, type_name, ids, request)?;
        Ok(Some(Self {
            ctx,
            rules,
            op,
            request,
            denied: Cell::new(false),
            #[cfg(test)]
            race: std::cell::RefCell::new(None),
        }))
    }

    /// Start an attempt: forget the previous attempt's decision.
    pub(crate) fn begin_attempt(&self) {
        self.denied.set(false);
    }

    /// The rule's decision on the object `view` shows, at the write's snapshot.
    pub(crate) fn allows(&self, view: &GuardView<'_>) -> bool {
        #[cfg(test)]
        if let Some(race) = self.race.borrow_mut().take() {
            race();
        }
        let res = TxnResource {
            view,
            ctx: self.ctx,
            request: self.request,
        };
        let allowed = self
            .rules
            .evaluate(self.op, view.type_name(), &self.ctx.principal, &res)
            .is_allow();
        self.denied.set(!allowed);
        allowed
    }

    /// After the engine call: `PermissionDenied` if its final attempt was denied.
    pub(crate) fn verdict(&self, type_name: &str) -> QueryResult<()> {
        if self.denied.get() {
            return Err(denied(self.op, type_name));
        }
        Ok(())
    }

    /// Test hook: run `race` once, on the first evaluation (inside the first attempt, after its
    /// snapshot was taken), to replay a concurrent change deterministically.
    #[cfg(test)]
    pub(crate) fn race_once(&self, race: impl FnOnce() + 'g) {
        *self.race.borrow_mut() = Some(Box::new(race));
    }
}

/// A [`ResourceAccessor`] reading through the write transaction's [`GuardView`]: the object at
/// the write's snapshot, relations and linked objects at the same snapshot, every read joining the
/// commit's conflict check. Relationship projections are bounded exactly as on the read path.
struct TxnResource<'v> {
    view: &'v GuardView<'v>,
    ctx: &'v ExecContext<'v>,
    request: Option<&'v FieldMap>,
}

impl TxnResource<'_> {
    /// Linked target ids for `rel`, or `None` = INDETERMINATE (read error, over budget, or too
    /// large), which the caller fails closed.
    fn linked_ids(&self, rel: &str) -> Option<Vec<u64>> {
        let ids = self.view.link_ids(rel).ok()?;
        within_rule_budget(&self.ctx.governor, ids.len() as u64).then_some(ids)
    }

    fn relation_target(&self, rel: &str) -> Option<String> {
        relation_target(self.ctx.db, self.view.type_name(), rel)
    }

    fn target_scalar(&self, target_type: &str, id: u64, field: &str) -> Option<serde_json::Value> {
        let fields = self.view.object_fields(target_type, id).ok()??;
        fields.get(field).map(value_to_query_json)
    }
}

impl ResourceAccessor for TxnResource<'_> {
    fn resource_scalar(&self, field: &str) -> Option<serde_json::Value> {
        self.view.fields().get(field).map(value_to_query_json)
    }

    fn resource_relation_one(&self, rel: &str, target_field: &str) -> Option<serde_json::Value> {
        let target_type = self.relation_target(rel)?;
        let id = *self.linked_ids(rel)?.first()?;
        self.target_scalar(&target_type, id, target_field)
    }

    fn resource_relation_many(
        &self,
        rel: &str,
        target_field: &str,
    ) -> Option<Vec<serde_json::Value>> {
        let target_type = self.relation_target(rel)?;
        let ids = self.linked_ids(rel)?;
        Some(
            ids.into_iter()
                .filter_map(|id| self.target_scalar(&target_type, id, target_field))
                .collect(),
        )
    }

    fn request_field(&self, field: &str) -> Option<serde_json::Value> {
        self.request?.get(field).map(value_to_query_json)
    }
}
