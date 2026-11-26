//! A pushed-down `.limit(N)` returns the first N LIVE matches, however many
//! matching rows were deleted and not yet compacted.
//!
//! `Type.filter(<indexed comparison>).limit(N)` pushes N into a bounded index
//! scan. That scan used to stop after N index keys counting tombstones, so a
//! deleted match cost a live one: 202 live channels and 2 deleted ones made
//! `Channel.filter(.group_id == g).limit(201)` return 199. Every case here is
//! checked against the same query without the limit (an unbounded scan), with
//! the deletes in the same memtable as the rows and in a newer layer than them.

use std::sync::Arc;

use rhypedb_engine::database::Database;
use rhypedb_query::executor::{ExecContext, QueryOutput, execute};
use rhypedb_query::parser::parse_query;
use rhypedb_schema::parser::parse_schema;

const SDL: &str = r#"
type Group {
    name: String @unique
    channels: [Channel]
}

type Channel {
    group_id: String @indexed
    rank: u64 @indexed
    handle: String @unique
}
"#;

/// Live channels in group `g`. Ids interleave with group `h`'s so the index
/// prefix for one value is surrounded by the other's keys.
const LIVE: u64 = 202;

fn open_db(dir: &std::path::Path) -> Arc<Database> {
    Database::open(parse_schema(SDL).unwrap(), dir).unwrap()
}

fn run(db: &Database, q: &str) -> QueryOutput {
    let ctx = ExecContext::new(db, None);
    execute(&ctx, &parse_query(q).unwrap()).unwrap_or_else(|e| panic!("{q}: {e}"))
}

fn ids(db: &Database, q: &str) -> Vec<u64> {
    match run(db, q) {
        QueryOutput::Objects(objs) => objs.iter().map(|o| o.id).collect(),
        QueryOutput::Single(o) => vec![o.id],
        other => panic!("{q}: unexpected output {other:?}"),
    }
}

fn create(db: &Database, q: &str) -> u64 {
    ids(db, q)[0]
}

/// Builds group `g` with `LIVE + deleted` channels (ranks 0.., interleaved
/// with group `h`), deletes the `deleted` lowest-ranked of `g`'s, and returns
/// the database. With `flush_first`, the rows reach an SST before the deletes,
/// so the tombstones sit in a newer layer than the rows they shadow.
fn seeded(dir: &std::path::Path, deleted: u64, flush_first: bool) -> Arc<Database> {
    let db = open_db(dir);
    let group = create(&db, r#"Group.create({ name: "g" })"#);
    let mut doomed = Vec::new();
    for rank in 0..LIVE + deleted {
        let id = create(
            &db,
            &format!(r#"Channel.create({{ group_id: "g", rank: {rank}, handle: "g{rank}" }})"#),
        );
        run(&db, &format!("Group.get({group}).link(Channel.get({id}))"));
        create(
            &db,
            &format!(r#"Channel.create({{ group_id: "h", rank: {rank}, handle: "h{rank}" }})"#),
        );
        if rank < deleted {
            doomed.push(id);
        }
    }
    if flush_first {
        db.storage().flush().unwrap();
    }
    for id in doomed {
        run(&db, &format!("Channel.get({id}).delete()"));
    }
    db
}

/// The first `n` of `q`'s unlimited result.
fn first(db: &Database, q: &str, n: usize) -> Vec<u64> {
    ids(db, q).into_iter().take(n).collect()
}

fn check_layouts(check: impl Fn(&Database)) {
    for flush_first in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        check(&seeded(dir.path(), 2, flush_first));
        let dir = tempfile::tempdir().unwrap();
        check(&seeded(dir.path(), 300, flush_first));
    }
}

/// The reported bug: a variable-width (`String`) indexed equality.
#[test]
fn indexed_string_equality_with_limit_returns_n_live_rows() {
    check_layouts(|db| {
        let all = ids(db, r#"Channel.filter(.group_id == "g")"#);
        assert_eq!(all.len() as u64, LIVE);
        for n in [1usize, 5, 199, 201, 202, 300] {
            let q = format!(r#"Channel.filter(.group_id == "g").limit({n})"#);
            assert_eq!(
                ids(db, &q),
                all.iter().copied().take(n).collect::<Vec<_>>(),
                "{q}"
            );
        }
    });
}

/// Fixed-width (`u64`) indexed comparisons push the limit down too: equality,
/// a seek for `>`/`>=`, and a bounded walk for `<`/`<=`.
#[test]
fn indexed_fixed_width_comparisons_with_limit_return_n_live_rows() {
    check_layouts(|db| {
        for pred in [
            ".rank >= 0",
            ".rank > 0",
            ".rank < 1000",
            ".rank <= 1000",
            ".rank == 400",
        ] {
            let unlimited = format!("Channel.filter({pred})");
            for n in [1usize, 3, 150, 400] {
                let q = format!("Channel.filter({pred}).limit({n})");
                assert_eq!(ids(db, &q), first(db, &unlimited, n), "{q}");
            }
        }
    });
}

/// `.offset(k).limit(n)` pages over live rows only.
#[test]
fn offset_then_limit_pages_over_live_rows() {
    check_layouts(|db| {
        for base in [
            r#"Channel.filter(.group_id == "g")"#,
            "Channel.filter(.rank >= 0)",
        ] {
            let all = ids(db, base);
            for (k, n) in [(0usize, 10usize), (1, 201), (195, 10), (500, 3)] {
                let q = format!("{base}.offset({k}).limit({n})");
                let want: Vec<u64> = all.iter().copied().skip(k).take(n).collect();
                assert_eq!(ids(db, &q), want, "{q}");
            }
        }
    });
}

/// A `@unique` equality finds the live holder of a value, and a value whose
/// holder was deleted finds nothing, limit or not.
#[test]
fn unique_equality_ignores_deleted_holders() {
    check_layouts(|db| {
        assert!(ids(db, r#"Channel.filter(.handle == "g0").limit(1)"#).is_empty());
        assert!(ids(db, r#"Channel.filter(.handle == "g1")"#).is_empty());
        let live = ids(db, r#"Channel.filter(.handle == "h0").limit(1)"#);
        assert_eq!(live.len(), 1);
        assert_eq!(ids(db, r#"Channel.filter(.handle == "h0")"#), live);
    });
}

/// A deleted object's value can be taken by a new one, and the limited lookup
/// returns the new holder, not nothing.
#[test]
fn unique_value_reused_after_delete_is_found_with_a_limit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_db(dir.path());
    let old = create(
        &db,
        r#"Channel.create({ group_id: "x", rank: 1, handle: "dup" })"#,
    );
    db.storage().flush().unwrap();
    run(&db, &format!("Channel.get({old}).delete()"));
    let new = create(
        &db,
        r#"Channel.create({ group_id: "x", rank: 1, handle: "dup" })"#,
    );
    assert_eq!(
        ids(&db, r#"Channel.filter(.handle == "dup").limit(1)"#),
        vec![new]
    );
    assert_eq!(
        ids(&db, r#"Channel.filter(.group_id == "x").limit(1)"#),
        vec![new]
    );
}

/// A traversal followed by a limit (and a filter on an indexed target field)
/// skips the deleted targets.
#[test]
fn traversal_then_limit_skips_deleted_targets() {
    check_layouts(|db| {
        let base = r#"Group.filter(.name == "g").channels"#;
        let all = ids(db, base);
        assert_eq!(all.len() as u64, LIVE);
        for n in [1usize, 201, 300] {
            let q = format!("{base}.limit({n})");
            assert_eq!(
                ids(db, &q),
                all.iter().copied().take(n).collect::<Vec<_>>(),
                "{q}"
            );
            let q = format!("{base}.offset(1).limit({n})");
            let want: Vec<u64> = all.iter().copied().skip(1).take(n).collect();
            assert_eq!(ids(db, &q), want, "{q}");
        }
        let filtered = format!(r#"{base}.filter(.rank >= 0).limit(201)"#);
        assert_eq!(ids(db, &filtered).len(), 201, "{filtered}");
    });
}
