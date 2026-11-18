//! GH #54: a cover's `<peer>__cover_v` stamp must name the generation its
//! `<peer>__cover` content was read at. Readers trust a copy when its stamp
//! equals the peer's live (in-memory) generation, and writers bump that
//! counter BEFORE they commit — so a stamp taken from the live counter by a
//! concurrent writer can name a generation whose content it never saw.
//!
//! These tests replay the interleavings deterministically: a writer's
//! in-flight bump is made by hand, the racing cover is built through the
//! public API, then the writer's commit is applied by hand.

use super::*;
use crate::object::{find_bytes_field_in_raw, find_u64_field_in_raw};
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
            user: User
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

/// The `user` copy in the Movie-side reverse edge of `rating`, with its stamp.
fn movie_side_user_copy(db: &Database, movie: u64, rating: u64) -> (Bytes, u64) {
    let rel = db.rel_ids["Rating.movie"];
    let value = db
        .storage
        .get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(movie, rel, rating))
        .unwrap()
        .expect("reverse edge");
    let copy = find_bytes_field_in_raw(&value, "user__cover").expect("user copy");
    let stamp = find_u64_field_in_raw(&value, "user__cover_v").expect("stamp");
    (copy, stamp)
}

/// The invariant readers rely on: a copy whose stamp matches the live
/// generation holds the object's current content.
fn assert_trusted_copy_is_current(db: &Database, movie: u64, rating: u64, user: u64) {
    let (copy, stamp) = movie_side_user_copy(db, movie, rating);
    if stamp == db.object_version("User", user) && stamp != 0 {
        let current = db.get("User", user).unwrap().fields;
        let copied = deserialize_fields(&copy);
        assert_eq!(
            copied.get("name"),
            current.get("name"),
            "a copy that reads as fresh (stamp {stamp}) must hold current content"
        );
    }
}

/// Apply an update of `user.name` as `update` would commit it, after its
/// in-memory bump already happened (`own`: the object's own count, which its
/// `g:` key persists).
fn commit_update_by_hand(db: &Database, user: u64, name: &str, own: u64) {
    let type_id = db.type_ids["User"];
    let mut fields = db.get("User", user).unwrap().fields;
    fields.insert("name".into(), Value::String(name.into()));
    let mut txn = db.storage.begin_txn();
    db.storage
        .put(&mut txn, &KeyBuilder::object(type_id, user), serialize_fields(&fields))
        .unwrap();
    db.storage
        .put(
            &mut txn,
            &KeyBuilder::object_version(type_id, user),
            Bytes::copy_from_slice(&own.to_be_bytes()),
        )
        .unwrap();
    db.storage.commit(&mut txn).unwrap();
}

/// Race 1 (the issue's scalar-update form): T1 updates the user and has
/// bumped its generation but not committed; T2 links a rating to the user,
/// copying the user as T2's snapshot sees it; then T1 commits.
#[test]
fn a_copy_built_during_an_uncommitted_bump_never_reads_as_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "old");
    let m = named(&db, "Movie", "title", "m");
    let r = db.create("Rating", FieldMap::new()).unwrap().id;
    db.link("Rating", r, "movie", m, None).unwrap();

    // T1: in-flight update — bumped in memory, not committed.
    let type_id = db.type_ids["User"];
    let (_, own) = db.bump_version(type_id, u);
    // T2: copies the user (still "old" in its snapshot) into the Movie side.
    db.link("Rating", r, "user", u, None).unwrap();
    // T1 commits "new".
    commit_update_by_hand(&db, u, "new", own);

    assert_trusted_copy_is_current(&db, m, r, u);
    let (_, stamp) = movie_side_user_copy(&db, m, r);
    assert_ne!(stamp, db.object_version("User", u), "the copy predates the update");
}

/// After the racing writer commits, a later cover build stamps the new
/// generation and is trusted again (the fix doesn't just distrust forever).
#[test]
fn a_copy_built_after_the_commit_is_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "old");
    let m = named(&db, "Movie", "title", "m");
    let r = db.create("Rating", FieldMap::new()).unwrap().id;
    db.link("Rating", r, "movie", m, None).unwrap();
    let mut f = FieldMap::new();
    f.insert("name".into(), Value::String("new".into()));
    db.update("User", u, f).unwrap();
    db.link("Rating", r, "user", u, None).unwrap();

    let (copy, stamp) = movie_side_user_copy(&db, m, r);
    assert_eq!(stamp, db.object_version("User", u));
    assert_eq!(deserialize_fields(&copy)["name"], Value::String("new".into()));
}

/// Race 2: T1 bumps v → v+1, T3 bumps v+1 → v+2 and commits, T1 fails. T1's
/// rollback must not take the counter below T3's committed v+2 — or the next
/// bump would hand out v+2 again, and copies stamped v+2 with T3's content
/// would read as fresh against newer content.
#[test]
fn a_failed_writer_never_rolls_back_a_later_committed_bump() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "v1");
    let m = named(&db, "Movie", "title", "m");
    let r = db.create("Rating", FieldMap::new()).unwrap().id;
    db.link("Rating", r, "movie", m, None).unwrap();
    let type_id = db.type_ids["User"];

    let (t1, _) = db.bump_version(type_id, u); // T1: 1 → 2, in flight
    let (t3, t3_own) = db.bump_version(type_id, u); // T3: 2 → 3
    commit_update_by_hand(&db, u, "t3", t3_own); // T3 commits generation 3
    db.rollback_version(type_id, u, t1); // T1's commit failed
    assert_eq!(db.object_version("User", u), t3, "T3's committed bump survives");

    // A copy made now carries T3's content, stamped with T3's generation.
    db.link("Rating", r, "user", u, None).unwrap();
    // The next real update must move past T3's generation.
    let mut f = FieldMap::new();
    f.insert("name".into(), Value::String("t4".into()));
    db.update("User", u, f).unwrap();
    assert!(db.object_version("User", u) > t3);
    assert_trusted_copy_is_current(&db, m, r, u);
}

/// A failed writer whose bump is still the newest restores the counter, so
/// the object's copies don't read as stale until its next update.
#[test]
fn a_failed_writer_restores_its_own_bump() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "v1");
    let type_id = db.type_ids["User"];
    let before = db.object_version("User", u);
    let (bumped, _) = db.bump_version(type_id, u);
    db.rollback_version(type_id, u, bumped);
    assert_eq!(db.object_version("User", u), before);
}

/// Every stamp a cover writer takes, across reopen: generations come back
/// from `g:` keys, and a copy stamped before the reopen stays trusted only
/// if nothing changed.
#[test]
fn stamps_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let (u, m, r);
    {
        let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
        u = named(&db, "User", "name", "a");
        m = named(&db, "Movie", "title", "m");
        r = db.create("Rating", FieldMap::new()).unwrap().id;
        let mut f = FieldMap::new();
        f.insert("name".into(), Value::String("b".into()));
        db.update("User", u, f).unwrap();
        db.link("Rating", r, "movie", m, None).unwrap();
        db.link("Rating", r, "user", u, None).unwrap();
    }
    let db = Database::open_with_options(schema(), dir.path(), no_sweeper()).unwrap();
    let (_, stamp) = movie_side_user_copy(&db, m, r);
    assert_eq!(stamp, db.object_version("User", u));
    assert_trusted_copy_is_current(&db, m, r, u);
}

/// Every copy a reader would trust (stamp == live generation) of an object of
/// `ty` holds only field names `ty` has now. A rename changes names without
/// changing content, so the copies made before it must stop reading as fresh.
fn assert_trusted_copies_use_current_names(db: &Database, holder: &str, rel: &str, peer: &str, ty: &str) -> usize {
    let td = db.schema.get_type(ty).unwrap();
    let names: Vec<&str> = td.fields.iter().map(|f| f.name.as_str()).collect();
    let rel_id = db.rel_ids[&format!("{holder}.{rel}")];
    let prefix = [rhypedb_storage::key::KeyPrefix::ReverseEdge as u8, rhypedb_storage::key::SEPARATOR];
    let snap = db.storage.read_snapshot();
    let mut checked = 0;
    for (key, value) in db.storage.scan_prefix_at(snap, &prefix).unwrap() {
        // r:<target 8>:<rel 8>:<source 8>
        if key.len() < 2 + 8 + 1 + 8 || u64::from_be_bytes(key[11..19].try_into().unwrap()) != rel_id {
            continue;
        }
        let (Some(copy), Some(id)) = (
            find_bytes_field_in_raw(&value, &format!("{peer}__cover")),
            find_u64_field_in_raw(&value, peer),
        ) else {
            continue;
        };
        let stamp = find_u64_field_in_raw(&value, &format!("{peer}__cover_v")).unwrap_or(0);
        if stamp == 0 || stamp != db.object_version(ty, id) {
            continue;
        }
        for name in deserialize_fields(&copy).keys() {
            let plain = name.trim_end_matches("__cover_v").trim_end_matches("__cover");
            assert!(names.contains(&plain), "trusted copy of {ty} {id} carries {name:?}; fields now {names:?}");
        }
        checked += 1;
    }
    checked
}

#[test]
fn a_scalar_rename_retires_the_copies_holding_the_old_name() {
    let before = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { title: String }
        "#,
    )
    .unwrap();
    let after = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { headline: String }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (u, m1);
    {
        let db = Database::open_with_options(before.clone(), dir.path(), no_sweeper()).unwrap();
        u = named(&db, "User", "name", "u");
        m1 = named(&db, "Movie", "title", "Inception");
        let m2 = named(&db, "Movie", "title", "Tenet");
        db.link("User", u, "favourite", m1, None).unwrap();
        db.link("User", u, "recommendation", m2, None).unwrap();
        assert!(assert_trusted_copies_use_current_names(&db, "User", "recommendation", "favourite", "Movie") > 0);
        db.rename_field("Movie", "title", "headline").unwrap();
    }
    let db = Database::open_with_options(after, dir.path(), no_sweeper()).unwrap();
    assert_eq!(db.get("Movie", m1).unwrap().fields["headline"], Value::String("Inception".into()));
    assert_trusted_copies_use_current_names(&db, "User", "recommendation", "favourite", "Movie");
    let _ = u;
}

#[test]
fn a_relation_rename_retires_the_copies_nesting_the_old_name() {
    let before = parse_schema(
        r#"
        type A { b: B  c: C }
        type B { name: String  d: D }
        type C { name: String }
        type D { name: String }
        "#,
    )
    .unwrap();
    let after = parse_schema(
        r#"
        type A { b: B  c: C }
        type B { name: String  e: D }
        type C { name: String }
        type D { name: String }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open_with_options(before, dir.path(), no_sweeper()).unwrap();
        let b = named(&db, "B", "name", "b");
        let c = named(&db, "C", "name", "c");
        let d = named(&db, "D", "name", "d");
        let a = db.create("A", FieldMap::new()).unwrap().id;
        db.link("B", b, "d", d, None).unwrap();
        db.link("A", a, "b", b, None).unwrap();
        db.link("A", a, "c", c, None).unwrap();
        assert!(assert_trusted_copies_use_current_names(&db, "A", "c", "b", "B") > 0);
        db.rename_field("B", "d", "e").unwrap();
    }
    let db = Database::open_with_options(after, dir.path(), no_sweeper()).unwrap();
    assert_trusted_copies_use_current_names(&db, "A", "c", "b", "B");
}

/// Review finding on #54: a consuming field-type change hands the new handle
/// the SAME in-memory counters. Its type-wide bump must reach them, or a copy
/// rebuilt on the new handle is stamped one generation ahead of memory and the
/// next update makes that stale copy read as fresh.
#[test]
fn a_consuming_type_change_keeps_the_shared_counters_in_step() {
    let before = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { score: u32 }
        "#,
    )
    .unwrap();
    let after = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { score: f64 }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(before, dir.path(), no_sweeper()).unwrap();
    let u = named(&db, "User", "name", "u");
    let mut f = FieldMap::new();
    f.insert("score".into(), Value::U32(3));
    let m1 = db.create("Movie", f).unwrap().id;
    let m2 = db.create("Movie", FieldMap::new()).unwrap().id;
    db.link("User", u, "favourite", m1, None).unwrap();
    db.link("User", u, "recommendation", m2, None).unwrap();
    assert!(assert_trusted_copies_use_current_names(&db, "User", "recommendation", "favourite", "Movie") > 0);

    let (_, db) = db
        .change_field_type_consuming(
            "Movie",
            "score",
            rhypedb_schema::FieldType::Scalar(rhypedb_schema::ScalarType::F64),
            |_, v| match v {
                Value::U32(x) => Ok(Value::F64(*x as f64)),
                other => Ok(other.clone()),
            },
            after,
        )
        .unwrap();
    // Old copies (score: U32) must no longer read as fresh.
    let copy_is_trusted = |db: &Database| {
        let rel = db.rel_ids["User.recommendation"];
        let value = db
            .storage
            .get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(m2, rel, u))
            .unwrap()
            .unwrap();
        let stamp = find_u64_field_in_raw(&value, "favourite__cover_v").unwrap();
        let copy = find_bytes_field_in_raw(&value, "favourite__cover").unwrap();
        (stamp == db.object_version("Movie", m1), deserialize_fields(&copy))
    };
    assert!(!copy_is_trusted(&db).0, "the pre-change copy is retired");

    // Rebuild the copy on the new handle, then update the movie.
    db.unlink("User", u, "recommendation", m2).unwrap();
    db.link("User", u, "recommendation", m2, None).unwrap();
    let (trusted, copy) = copy_is_trusted(&db);
    assert!(trusted, "a copy rebuilt after the change is current");
    assert_eq!(copy["score"], Value::F64(3.0));
    let mut f = FieldMap::new();
    f.insert("score".into(), Value::F64(9.0));
    db.update("Movie", m1, f).unwrap();
    let (trusted, copy) = copy_is_trusted(&db);
    assert!(!trusted, "after the update that copy is stale, and must read so: {copy:?}");
}

/// The epoch persists: a reopened store keeps every object of a renamed type
/// past the generations its old copies carry, and costs no per-object entry.
#[test]
fn a_rename_epoch_survives_reopen_without_per_object_entries() {
    let before = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { title: String }
        "#,
    )
    .unwrap();
    let after = parse_schema(
        r#"
        type User { name: String  favourite: Movie  recommendation: Movie }
        type Movie { headline: String }
        "#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let movies: Vec<u64>;
    {
        let db = Database::open_with_options(before, dir.path(), no_sweeper()).unwrap();
        movies = (0..50).map(|i| named(&db, "Movie", "title", &format!("m{i}"))).collect();
        let u = named(&db, "User", "name", "u");
        db.link("User", u, "favourite", movies[0], None).unwrap();
        db.link("User", u, "recommendation", movies[1], None).unwrap();
        let updated_before = db.version_counters.read().updated_len();
        db.rename_field("Movie", "title", "headline").unwrap();
        assert_eq!(db.version_counters.read().updated_len(), updated_before, "no per-object entries");
    }
    let db = Database::open_with_options(after, dir.path(), no_sweeper()).unwrap();
    assert!(movies.iter().all(|m| db.object_version("Movie", *m) == 2), "epoch 1 + own 1");
    assert_trusted_copies_use_current_names(&db, "User", "recommendation", "favourite", "Movie");
}

/// Second review on #54: a create rolled back on a type that has an epoch
/// (after a rename) must leave nothing behind — no born bit, no generation
/// that could make a deleted object's leftover copy read as fresh.
#[test]
fn a_failed_create_under_an_epoch_leaves_no_generation() {
    let before = parse_schema("type Movie { title: String }").unwrap();
    let after = parse_schema("type Movie { headline: String }").unwrap();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open_with_options(before, dir.path(), no_sweeper()).unwrap();
        named(&db, "Movie", "title", "m");
        db.rename_field("Movie", "title", "headline").unwrap();
    }
    let db = Database::open_with_options(after, dir.path(), no_sweeper()).unwrap();
    let existing = 1u64;
    assert!(db.object_version("Movie", existing) > 1, "the type has an epoch");
    // A restore that stages a fresh id, then refuses on an existing one.
    let fresh = 500u64;
    let rows = vec![(fresh, FieldMap::new()), (existing, FieldMap::new())];
    assert!(db.restore_objects("Movie", rows, true).is_err());
    assert_eq!(db.object_version("Movie", fresh), 0, "the refused create is rolled back");
}
