//! In-memory object generation counters ("cover_v" staleness stamps).
//!
//! Every live object has a generation: 1 when never updated ("born-at-1"),
//! higher after each update; a deleted or never-existing object reads 0. The
//! query executor trusts an embedded cover only when its stamped generation
//! equals the live one AND that generation is non-zero — the zero guard is
//! what keeps a deleted target's leftover cover from surfacing as a phantom.
//!
//! The old representation was one `HashMap<(type, id), generation>` entry for
//! EVERY live object, seeded at open — ~40 bytes per object, forever. Almost
//! every object sits at generation 1, so that information is just "this id is
//! live". It now lives in a compressed bitmap per type (`RoaringTreemap`:
//! about 1–2 bits per dense id), and only objects updated at least once keep
//! a map entry. Every read returns exactly what the old map returned.

use std::collections::HashMap;

use roaring::RoaringTreemap;

#[derive(Debug, Default, Clone)]
pub(crate) struct VersionCounters {
    /// Live objects per type: own count >= 1.
    born: HashMap<u64, RoaringTreemap>,
    /// Objects updated at least once: own count >= 2.
    updated: HashMap<(u64, u64), u64>,
    /// Per-type epoch (`t:` key), added to every live object's own count. A
    /// change that alters every object of a type at once (a field rename or
    /// type change) bumps it: one entry instead of one per object (GH #54).
    epochs: HashMap<u64, u64>,
}

impl VersionCounters {
    /// Current generation: the type's epoch plus the object's own count (its
    /// update count if updated, 1 if live and never updated); 0 if deleted /
    /// unknown, whatever the epoch.
    pub(crate) fn get(&self, type_id: u64, object_id: u64) -> u64 {
        match self.local(type_id, object_id) {
            0 => 0,
            own => self.epoch(type_id) + own,
        }
    }

    /// The object's own count, without the type epoch — what its `g:` key
    /// persists.
    pub(crate) fn local(&self, type_id: u64, object_id: u64) -> u64 {
        if let Some(&v) = self.updated.get(&(type_id, object_id)) {
            return v;
        }
        u64::from(self.is_born(type_id, object_id))
    }

    /// The type's epoch (0 until its first `t:` bump).
    pub(crate) fn epoch(&self, type_id: u64) -> u64 {
        self.epochs.get(&type_id).copied().unwrap_or(0)
    }

    /// Raise the type's epoch to at least `epoch` (open-time seeding from its
    /// `t:` key, or after a committed epoch bump).
    pub(crate) fn raise_epoch(&mut self, type_id: u64, epoch: u64) {
        let e = self.epochs.entry(type_id).or_insert(0);
        *e = (*e).max(epoch);
    }

    fn is_born(&self, type_id: u64, object_id: u64) -> bool {
        self.born.get(&type_id).is_some_and(|b| b.contains(object_id))
    }

    /// Counters with every object in `born` live at generation 1 (open-time
    /// seeding from the storage's live-id sets).
    pub(crate) fn from_born(born: HashMap<u64, RoaringTreemap>) -> Self {
        Self { born, updated: HashMap::new(), epochs: HashMap::new() }
    }

    /// Record a live object at generation 1 (open-time seeding).
    pub(crate) fn mark_born(&mut self, type_id: u64, object_id: u64) {
        self.born.entry(type_id).or_default().insert(object_id);
    }

    /// Set an object's own count from its persisted `g:` key. 0 forgets it.
    pub(crate) fn set(&mut self, type_id: u64, object_id: u64, generation: u64) {
        match generation {
            0 => self.forget(type_id, object_id),
            1 => {
                self.updated.remove(&(type_id, object_id));
                self.mark_born(type_id, object_id);
            }
            v => {
                self.mark_born(type_id, object_id);
                self.updated.insert((type_id, object_id), v);
            }
        }
    }

    /// Advance the object's own count by one (0 → 1 on create, n → n+1 on
    /// update); returns the new GENERATION (epoch included). Its own count —
    /// what `g:` persists — is [`local`](Self::local).
    pub(crate) fn bump(&mut self, type_id: u64, object_id: u64) -> u64 {
        let next = self.local(type_id, object_id) + 1;
        self.set(type_id, object_id, next);
        self.epoch(type_id) + next
    }

    /// Undo the [`bump`](Self::bump) that returned `bumped`, whose write
    /// didn't land: back to `bumped - 1` (1 → 0 forgets the object) — but only
    /// while the counter still reads `bumped`. Once another writer has bumped
    /// past it, the counter is left alone: that later generation may already
    /// be committed, and taking the counter below it would let the next bump
    /// hand the same generation out twice (GH #54). A skipped generation is
    /// harmless — cover stamps come from persisted generations, so it only
    /// makes copies read as stale until the next committed bump.
    pub(crate) fn rollback(&mut self, type_id: u64, object_id: u64, bumped: u64) {
        if bumped > 0 && self.get(type_id, object_id) == bumped {
            let own = self.local(type_id, object_id);
            self.set(type_id, object_id, own - 1);
        }
    }

    /// Drop the object entirely (it was deleted): reads 0 afterwards.
    pub(crate) fn forget(&mut self, type_id: u64, object_id: u64) {
        self.updated.remove(&(type_id, object_id));
        if let Some(b) = self.born.get_mut(&type_id) {
            b.remove(object_id);
            if b.is_empty() {
                self.born.remove(&type_id);
            }
        }
    }

    /// Whether the object has been updated at least once (generation >= 2).
    pub(crate) fn was_updated(&self, type_id: u64, object_id: u64) -> bool {
        self.updated.contains_key(&(type_id, object_id))
    }

    /// Number of objects with generation >= 2.
    pub(crate) fn updated_len(&self) -> usize {
        self.updated.len()
    }

    /// GH #54: an epoch moves every live object of the type at once, never
    /// resurrects a deleted one, and a rollback still undoes only its own bump.
    #[cfg(test)]
    pub(crate) fn epochs_len(&self) -> usize {
        self.epochs.len()
    }

    /// Heap bytes held, approximately (the memory regression test).
    #[cfg(test)]
    pub(crate) fn approx_heap_bytes(&self) -> usize {
        let bitmaps: usize = self.born.values().map(|b| b.serialized_size()).sum();
        let map = self.updated.capacity() * (std::mem::size_of::<((u64, u64), u64)>() + 1);
        bitmaps + map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference model: the old one-entry-per-live-object map, with the
    /// bump / rollback / forget rules.
    #[derive(Default)]
    struct OldMap(HashMap<(u64, u64), u64>);
    impl OldMap {
        fn get(&self, t: u64, i: u64) -> u64 {
            self.0.get(&(t, i)).copied().unwrap_or(0)
        }
        fn bump(&mut self, t: u64, i: u64) -> u64 {
            let e = self.0.entry((t, i)).or_insert(0);
            *e += 1;
            *e
        }
        fn rollback(&mut self, t: u64, i: u64, bumped: u64) {
            if bumped > 0 && self.get(t, i) == bumped {
                if bumped == 1 {
                    self.0.remove(&(t, i));
                } else {
                    self.0.insert((t, i), bumped - 1);
                }
            }
        }
        fn forget(&mut self, t: u64, i: u64) {
            self.0.remove(&(t, i));
        }
    }

    #[test]
    fn matches_the_old_map_under_random_operations() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut old = OldMap::default();
        let mut new = VersionCounters::default();
        for _ in 0..200_000 {
            let t = next() % 3;
            let i = next() % 500;
            match next() % 8 {
                0..=3 => assert_eq!(new.bump(t, i), old.bump(t, i)),
                4 => {
                    // Roll back the current bump, a stale one, or one never made.
                    let bumped = old.get(t, i).saturating_sub(next() % 2);
                    new.rollback(t, i, bumped);
                    old.rollback(t, i, bumped);
                }
                5 => {
                    new.forget(t, i);
                    old.forget(t, i);
                }
                _ => {}
            }
            assert_eq!(new.get(t, i), old.get(t, i));
            assert_eq!(new.was_updated(t, i), old.get(t, i) > 1);
        }
        for t in 0..3 {
            for i in 0..500 {
                assert_eq!(new.get(t, i), old.get(t, i), "({t}, {i})");
            }
        }
        assert_eq!(new.updated_len(), old.0.values().filter(|&&v| v > 1).count());
    }

    /// GH #54 race 2: a failed writer whose bump was overtaken leaves the
    /// newer generation alone; one whose bump is still newest restores.
    #[test]
    fn rollback_only_undoes_the_newest_bump() {
        let mut v = VersionCounters::default();
        v.mark_born(1, 10);
        let t1 = v.bump(1, 10); // 2
        let t3 = v.bump(1, 10); // 3
        v.rollback(1, 10, t1);
        assert_eq!(v.get(1, 10), t3, "overtaken: untouched");
        v.rollback(1, 10, t3);
        assert_eq!(v.get(1, 10), 2, "newest: restored");
        let born = v.bump(1, 11); // create: 0 -> 1
        v.rollback(1, 11, born);
        assert_eq!(v.get(1, 11), 0, "a rolled-back create is forgotten");
    }

    #[test]
    fn an_epoch_moves_every_live_object_of_its_type() {
        let mut v = VersionCounters::default();
        v.mark_born(1, 10);
        v.mark_born(1, 11);
        v.mark_born(2, 10);
        v.bump(1, 11); // own 2
        v.raise_epoch(1, 3);
        assert_eq!(v.get(1, 10), 4);
        assert_eq!(v.get(1, 11), 5);
        assert_eq!(v.get(2, 10), 1, "other type untouched");
        assert_eq!(v.get(1, 99), 0, "unknown stays 0");
        assert_eq!(v.local(1, 11), 2, "g: keeps the own count");
        let b = v.bump(1, 10);
        assert_eq!((b, v.local(1, 10)), (5, 2));
        v.rollback(1, 10, b);
        assert_eq!(v.get(1, 10), 4);
        v.raise_epoch(1, 2);
        assert_eq!(v.epoch(1), 3, "raise never lowers");
        v.forget(1, 10);
        assert_eq!(v.get(1, 10), 0, "deleted reads 0 whatever the epoch");
        assert_eq!(v.epochs_len(), 1);
        assert_eq!(v.updated_len(), 1, "an epoch costs no per-object entry");
    }

    #[test]
    fn seeding_and_persisted_generations() {
        let mut v = VersionCounters::default();
        v.mark_born(1, 10);
        assert_eq!(v.get(1, 10), 1);
        assert_eq!(v.get(1, 11), 0, "never seen");
        assert_eq!(v.get(2, 10), 0, "other type");
        v.set(1, 10, 7); // a g: key
        assert_eq!(v.get(1, 10), 7);
        v.forget(1, 10);
        assert_eq!(v.get(1, 10), 0, "deleted reads 0 (the phantom guard)");
    }

    /// The point of the change: dense ids cost bits, not a map entry each.
    #[test]
    fn a_million_never_updated_objects_fit_in_a_few_hundred_kb() {
        let mut v = VersionCounters::default();
        for id in 1..=1_000_000u64 {
            v.mark_born(7, id);
        }
        for id in (1..=1_000_000u64).step_by(1000) {
            v.bump(7, id); // 1,000 updated objects
        }
        assert_eq!(v.get(7, 500_000), 1);
        assert_eq!(v.get(7, 1), 2);
        let bytes = v.approx_heap_bytes();
        // The old map: ~40 MB for the same million objects.
        assert!(bytes < 400_000, "{bytes} bytes");
    }
}
