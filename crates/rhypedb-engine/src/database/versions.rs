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
    /// Live objects per type: generation >= 1.
    born: HashMap<u64, RoaringTreemap>,
    /// Objects updated at least once: generation >= 2.
    updated: HashMap<(u64, u64), u64>,
}

impl VersionCounters {
    /// Current generation: the update count if updated, 1 if live and never
    /// updated, 0 if deleted / unknown.
    pub(crate) fn get(&self, type_id: u64, object_id: u64) -> u64 {
        if let Some(&v) = self.updated.get(&(type_id, object_id)) {
            return v;
        }
        u64::from(self.is_born(type_id, object_id))
    }

    fn is_born(&self, type_id: u64, object_id: u64) -> bool {
        self.born.get(&type_id).is_some_and(|b| b.contains(object_id))
    }

    /// Record a live object at generation 1 (open-time seeding).
    pub(crate) fn mark_born(&mut self, type_id: u64, object_id: u64) {
        self.born.entry(type_id).or_default().insert(object_id);
    }

    /// Set an object's generation from its persisted `g:` key. 0 forgets it.
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

    /// Advance the generation by one (0 → 1 on create, n → n+1 on update);
    /// returns the new value.
    pub(crate) fn bump(&mut self, type_id: u64, object_id: u64) -> u64 {
        let next = self.get(type_id, object_id) + 1;
        self.set(type_id, object_id, next);
        next
    }

    /// Undo one [`bump`](Self::bump) whose write didn't land (n → n-1; 1 → 0
    /// forgets the object). A no-op at 0.
    pub(crate) fn rollback(&mut self, type_id: u64, object_id: u64) {
        let current = self.get(type_id, object_id);
        if current > 0 {
            self.set(type_id, object_id, current - 1);
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

    /// Reference model: the old one-entry-per-live-object map, with the old
    /// bump / rollback / forget rules verbatim.
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
        fn rollback(&mut self, t: u64, i: u64) {
            if let Some(v) = self.0.get_mut(&(t, i)) {
                if *v > 0 {
                    *v -= 1;
                }
                if *v == 0 {
                    self.0.remove(&(t, i));
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
                    new.rollback(t, i);
                    old.rollback(t, i);
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
