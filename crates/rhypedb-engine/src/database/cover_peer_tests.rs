//! Reverse-edge covers carry their source's to-one peer ids (`Rating.user` in
//! the `r:<movie>:Rating.movie:<rating>` entry, say), and the query executor's
//! fusion path and cascade delete read them as the truth. These tests pin the
//! invariant that they always match the edge index — whatever order the links
//! were made in, across relink / unlink / cascade removal — and the
//! `c:C:` soundness marker that tells readers whether a store (possibly
//! written by an older rhypedb) can be trusted.

use super::*;
use crate::object::find_u64_field_in_raw;
use rhypedb_schema::parser::parse_schema;

fn schema() -> Schema {
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
            user: User @on_delete(remove)
            movie: Movie
        }
        "#,
    )
    .unwrap()
}

fn no_sweeper() -> OpenOptions {
    OpenOptions {
        background_cover_refresh: false,
        ..Default::default()
    }
}

fn named(db: &Database, ty: &str, field: &str, v: &str) -> u64 {
    let mut f = FieldMap::new();
    f.insert(field.into(), Value::String(v.into()));
    db.create(ty, f).unwrap().id
}

fn rating(db: &Database) -> u64 {
    let mut f = FieldMap::new();
    f.insert("stars".into(), Value::U32(5));
    db.create("Rating", f).unwrap().id
}

/// The `user` peer id that the Movie-side reverse edge of `rating` carries.
fn movie_side_user_peer(db: &Database, movie: u64, rating: u64) -> Option<u64> {
    let groups = db.get_links_many("Movie", &[movie], "ratings").unwrap();
    let (_, cover) = groups[0].iter().find(|(r, _)| *r == rating).expect("rev edge");
    find_u64_field_in_raw(cover, "user")
}

#[test]
fn a_later_to_one_link_reaches_the_earlier_links_cover() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let (u, m, r) = (named(&db, "User", "name", "u"), named(&db, "Movie", "title", "m"), rating(&db));
    // Movie first: its reverse edge is written before the user peer exists.
    db.link("Rating", r, "movie", m, None).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), None);
    db.link("Rating", r, "user", u, None).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u));
}

#[test]
fn relink_and_unlink_replace_the_peer_in_other_covers() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let (u1, u2) = (named(&db, "User", "name", "a"), named(&db, "User", "name", "b"));
    let m = named(&db, "Movie", "title", "m");
    let r = rating(&db);
    db.link("Rating", r, "user", u1, None).unwrap();
    db.link("Rating", r, "movie", m, None).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u1));

    db.unlink("Rating", r, "user", u1).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), None, "unlink must drop the peer");
    db.link("Rating", r, "user", u2, None).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u2), "relink must replace the peer");

    // Unlinking an edge that doesn't exist changes nothing.
    db.unlink("Rating", r, "user", u1).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u2));
}

#[test]
fn cascade_remove_drops_the_deleted_peer_from_surviving_covers() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "u");
    let m = named(&db, "Movie", "title", "m");
    let r = rating(&db);
    db.link("Rating", r, "user", u, None).unwrap();
    db.link("Rating", r, "movie", m, None).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u));

    // Rating.user is @on_delete(remove): the rating survives, its user edge goes.
    db.delete("User", u).unwrap();
    assert!(db.get_link_ids("Rating", r, "user").unwrap().is_empty());
    assert_eq!(movie_side_user_peer(&db, m, r), None, "cover still lists the deleted user");
}

#[test]
fn a_to_one_change_invalidates_covers_that_nest_the_source() {
    // `A.c`'s reverse edge embeds A's other to-one target B as `b__cover`,
    // and that copy carries B's own to-one peer `d` nested inside it. A
    // relink of `B.d` must bump B's generation so readers stop trusting it.
    let schema = parse_schema(
        r#"
        type A { b: B  c: C }
        type B { name: String  d: D  as: [A] @inverse(A.b) }
        type C { name: String  as: [A] @inverse(A.c) }
        type D { name: String  bs: [B] @inverse(B.d) }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema, dir.path(), no_sweeper()).unwrap();
    let b = named(&db, "B", "name", "b");
    let c = named(&db, "C", "name", "c");
    let (d1, d2) = (named(&db, "D", "name", "d1"), named(&db, "D", "name", "d2"));
    let a = db.create("A", FieldMap::new()).unwrap().id;
    db.link("B", b, "d", d1, None).unwrap();
    db.link("A", a, "b", b, None).unwrap();
    db.link("A", a, "c", c, None).unwrap();

    let groups = db.get_links_many("C", &[c], "as").unwrap();
    let cover = &groups[0][0].1;
    let stamped = find_u64_field_in_raw(cover, "b__cover_v").expect("b is covered");
    assert_eq!(stamped, db.object_version("B", b), "fresh cover");

    db.unlink("B", b, "d", d1).unwrap();
    db.link("B", b, "d", d2, None).unwrap();
    assert!(
        db.object_version("B", b) > stamped,
        "B's nested peer changed, so covers embedding B must read as stale"
    );
}

#[test]
fn a_new_store_is_sound_and_stays_sound_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(schema(), dir.path()).unwrap();
        assert!(db.cover_peers_sound());
        let (u, m, r) = (named(&db, "User", "name", "u"), named(&db, "Movie", "title", "m"), rating(&db));
        db.link("Rating", r, "user", u, None).unwrap();
        db.link("Rating", r, "movie", m, None).unwrap();
    }
    let db = Database::open(schema(), dir.path()).unwrap();
    assert!(db.cover_peers_sound());
}

/// Make the store look like an older rhypedb wrote it: drop the marker and
/// put back `stale` as the Movie-side reverse edge of `r`.
fn age_store(db: &Database, m: u64, r: u64, stale: Bytes) {
    let rel = db.rel_ids["Rating.movie"];
    let mut txn = db.storage.begin_txn();
    db.storage.delete(&mut txn, &KeyBuilder::catalog_cover_peers()).unwrap();
    db.storage.put(&mut txn, &KeyBuilder::reverse_edge(m, rel, r), stale).unwrap();
    db.storage.commit(&mut txn).unwrap();
}

#[test]
fn an_old_store_is_distrusted_until_rewrite_covers_repairs_it() {
    let dir = tempfile::tempdir().unwrap();
    let (u1, u2, m, r);
    {
        let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
        u1 = named(&db, "User", "name", "a");
        u2 = named(&db, "User", "name", "b");
        m = named(&db, "Movie", "title", "m");
        r = rating(&db);
        db.link("Rating", r, "user", u1, None).unwrap();
        db.link("Rating", r, "movie", m, None).unwrap();
        let rel = db.rel_ids["Rating.movie"];
        let old = db.storage.get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(m, rel, r)).unwrap().unwrap();
        db.unlink("Rating", r, "user", u1).unwrap();
        db.link("Rating", r, "user", u2, None).unwrap();
        // What an older rhypedb left behind: the cover still names u1.
        age_store(&db, m, r, old);
    }
    {
        let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
        assert!(!db.cover_peers_sound(), "an unmarked store with edges is distrusted");
        assert_eq!(movie_side_user_peer(&db, m, r), Some(u1), "the planted stale peer");
        db.rewrite_covers().unwrap();
        assert!(db.cover_peers_sound());
        assert_eq!(movie_side_user_peer(&db, m, r), Some(u2), "repaired");
    }
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    assert!(db.cover_peers_sound(), "the repair is persisted");
}

#[test]
fn rewrite_covers_fills_an_empty_cover_whose_peer_arrived_later() {
    let dir = tempfile::tempdir().unwrap();
    let (u, m, r);
    {
        let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
        u = named(&db, "User", "name", "u");
        m = named(&db, "Movie", "title", "m");
        r = rating(&db);
        db.link("Rating", r, "movie", m, None).unwrap();
        db.link("Rating", r, "user", u, None).unwrap();
        // An older rhypedb left the first link's cover empty forever.
        age_store(&db, m, r, Bytes::new());
    }
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), None);
    db.rewrite_covers().unwrap();
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u));
}

#[test]
fn cascade_on_an_old_store_ignores_stale_peers_and_leaves_no_dangling_edge() {
    // Movie → Rating cascades here. The planted cover names u1, but the
    // rating really points at u2: trusting the cover would tombstone the
    // (absent) u1 edge and leave u2's reverse edge pointing at a deleted
    // rating.
    let schema = parse_schema(
        r#"
        type User { name: String  ratings: [Rating] @inverse(Rating.user) }
        type Movie { title: String  ratings: [Rating] @inverse(Rating.movie) }
        type Rating { stars: u32  user: User  movie: Movie @on_delete(cascade) }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (u2, m);
    {
        let db = Database::open_with_options(schema.clone(), dir.path(), no_sweeper()).unwrap();
        let u1 = named(&db, "User", "name", "a");
        u2 = named(&db, "User", "name", "b");
        m = named(&db, "Movie", "title", "m");
        let r = rating(&db);
        db.link("Rating", r, "user", u1, None).unwrap();
        db.link("Rating", r, "movie", m, None).unwrap();
        let rel = db.rel_ids["Rating.movie"];
        let old = db.storage.get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(m, rel, r)).unwrap().unwrap();
        db.unlink("Rating", r, "user", u1).unwrap();
        db.link("Rating", r, "user", u2, None).unwrap();
        age_store(&db, m, r, old);
    }
    let db = Database::open_with_options(schema, dir.path(), no_sweeper()).unwrap();
    assert!(!db.cover_peers_sound());
    db.delete("Movie", m).unwrap();
    assert!(
        db.get_link_ids("User", u2, "ratings").unwrap().is_empty(),
        "u2 still lists the cascade-deleted rating"
    );
}

#[test]
fn linking_a_to_one_relation_again_replaces_its_target() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let (u1, u2) = (named(&db, "User", "name", "a"), named(&db, "User", "name", "b"));
    let m = named(&db, "Movie", "title", "m");
    let r = rating(&db);
    db.link("Rating", r, "user", u1, None).unwrap();
    db.link("Rating", r, "movie", m, None).unwrap();
    // No unlink first: a to-one relation still ends up with one target.
    db.link("Rating", r, "user", u2, None).unwrap();
    assert_eq!(db.get_link_ids("Rating", r, "user").unwrap(), vec![u2]);
    assert!(db.get_link_ids("User", u1, "ratings").unwrap().is_empty(), "old reverse edge gone");
    assert_eq!(movie_side_user_peer(&db, m, r), Some(u2));
}

#[test]
fn an_old_store_whose_schema_carries_no_peers_opens_sound() {
    // Only to-many relations: no reverse edge can carry a to-one peer.
    let schema = parse_schema(
        r#"
        type Label { name: String  messages: [Message] @inverse(Message.labels) }
        type Message { subject: String  labels: [Label] }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(schema.clone(), dir.path()).unwrap();
        let l = named(&db, "Label", "name", "inbox");
        let m = named(&db, "Message", "subject", "hi");
        db.link("Message", m, "labels", l, None).unwrap();
        let mut txn = db.storage.begin_txn();
        db.storage.delete(&mut txn, &KeyBuilder::catalog_cover_peers()).unwrap();
        db.storage.commit(&mut txn).unwrap();
    }
    let db = Database::open(schema, dir.path()).unwrap();
    assert!(db.cover_peers_sound());
}

#[test]
fn a_store_whose_reverse_edges_are_all_deleted_opens_sound() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(schema(), dir.path()).unwrap();
        let (u, m, r) = (named(&db, "User", "name", "u"), named(&db, "Movie", "title", "m"), rating(&db));
        db.link("Rating", r, "user", u, None).unwrap();
        db.link("Rating", r, "movie", m, None).unwrap();
        db.unlink("Rating", r, "user", u).unwrap();
        db.unlink("Rating", r, "movie", m).unwrap();
        // Only tombstones remain under `r:`; drop the marker as an old store would lack it.
        let mut txn = db.storage.begin_txn();
        db.storage.delete(&mut txn, &KeyBuilder::catalog_cover_peers()).unwrap();
        db.storage.commit(&mut txn).unwrap();
    }
    let db = Database::open(schema(), dir.path()).unwrap();
    assert!(db.cover_peers_sound());
}
