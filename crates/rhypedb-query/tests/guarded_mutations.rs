//! Filter-guarded mutations are compare-and-set.
//!
//! `T.filter(<guard>).update({...})` (and `.delete()`, `.link()`, `.unlink()`)
//! evaluates the filter on a read snapshot, then mutates each matching object in
//! its own transaction. Each of those transactions re-checks the filter against
//! the object's current committed version and skips the object if it no longer
//! matches, so concurrent writers guarded on the same old value can never both
//! apply. See `docs/src/queries.md`, "Guarded writes".
//!
//! These are stress tests through the public executor (the path the server and
//! every embedder use): many threads race the same guarded statement on one row,
//! and exactly one may apply.

use std::sync::{Arc, Barrier};

use rhypedb_engine::database::Database;
use rhypedb_engine::object::Value;
use rhypedb_query::executor::{ExecContext, QueryOutput, execute};
use rhypedb_query::parser::parse_query;
use rhypedb_query::{Governor, QueryError};
use rhypedb_schema::parser::parse_schema;

/// Threads racing each guarded statement (the task floor is 16).
const RACERS: usize = 16;
/// Rounds per test: the unfixed race showed up in roughly one round in five.
const ROUNDS: usize = 40;

const SDL: &str = r#"
type Account {
    handle: String @indexed
    step: u64
    note: String
    peers: [Account]
}
"#;

fn open_db(dir: &std::path::Path) -> Arc<Database> {
    Database::open(parse_schema(SDL).unwrap(), dir).unwrap()
}

fn run(db: &Database, q: &str) -> Result<QueryOutput, QueryError> {
    let ctx = ExecContext {
        db,
        vectorizer: None,
        default_ef: None,
        default_rerank: None,
        governor: Governor::disabled(),
        principal: rhypedb_authz::Principal::anonymous(),
        rules: None,
    };
    execute(&ctx, &parse_query(q).unwrap())
}

/// Rows an update statement reports as written.
fn rows(out: &QueryOutput) -> usize {
    match out {
        QueryOutput::Single(_) => 1,
        QueryOutput::Objects(o) => o.len(),
        other => panic!("unexpected update output {other:?}"),
    }
}

fn create_account(db: &Database, handle: &str, step: u64) -> u64 {
    let out = run(
        db,
        &format!(r#"Account.create({{ handle: "{handle}", step: {step}, note: "" }})"#),
    )
    .unwrap();
    match out {
        QueryOutput::Single(o) => o.id,
        other => panic!("unexpected create output {other:?}"),
    }
}

fn read_step(db: &Database, id: u64) -> u64 {
    let mut obj = db.get("Account", id).unwrap();
    obj.ensure_fields_deserialized();
    match obj.fields.get("step") {
        Some(Value::U64(s)) => *s,
        other => panic!("unexpected step {other:?}"),
    }
}

/// Run `make(i)` on `RACERS` threads released together; collect each result.
fn race<T: Send + 'static>(
    db: &Arc<Database>,
    make: impl Fn(usize) -> String,
    each: impl Fn(&Database, &str) -> T + Send + Sync + 'static,
) -> Vec<T> {
    let barrier = Arc::new(Barrier::new(RACERS));
    let each = Arc::new(each);
    let handles: Vec<_> = (0..RACERS)
        .map(|i| {
            let (db, barrier, each, q) = (db.clone(), barrier.clone(), each.clone(), make(i));
            std::thread::spawn(move || {
                barrier.wait();
                each(&db, &q)
            })
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// The downstream shape (TOTP replay protection): every racer claims the same
/// step by guarding on the old value. Exactly one may apply; the others see an
/// empty result, never an error.
#[test]
fn concurrent_guarded_updates_on_one_row_apply_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_db(dir.path());
    let id = create_account(&db, "h", 0);
    for round in 0..ROUNDS as u64 {
        let results = race(
            &db,
            |i| {
                format!(
                    r#"Account.filter(.handle == "h" && .step == {round}).update({{ step: {next}, note: "r{i}" }})"#,
                    next = round + 1
                )
            },
            |db, q| run(db, q).map(|o| rows(&o)),
        );
        let applied: usize = results
            .iter()
            .map(|r| {
                *r.as_ref()
                    .expect("a guarded update must not fail under contention")
            })
            .sum();
        assert_eq!(
            applied, 1,
            "round {round}: {applied} of {RACERS} guarded updates applied"
        );
        assert_eq!(read_step(&db, id), round + 1);
    }
}

/// A high-water mark (`step < n` guard, as in a replay ledger): racers propose
/// different values; the mark only ever moves forward, and an update reports
/// the row only when its own write landed.
#[test]
fn concurrent_high_water_updates_never_move_backwards() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_db(dir.path());
    let id = create_account(&db, "hw", 0);
    for round in 0..ROUNDS as u64 {
        let base = round * 100;
        let results = race(
            &db,
            |i| {
                let n = base + 1 + i as u64;
                format!(r#"Account.filter(.handle == "hw" && .step < {n}).update({{ step: {n} }})"#)
            },
            |db, q| run(db, q).map(|o| rows(&o)),
        );
        for r in &results {
            r.as_ref()
                .expect("a guarded update must not fail under contention");
        }
        // The highest proposal always wins, whatever order the racers ran in.
        assert_eq!(read_step(&db, id), base + RACERS as u64, "round {round}");
    }
}

/// Delete-by-filter racing a guarded update: whichever commits first, the other
/// sees the row as no longer matching. Never both, never an error.
#[test]
fn guarded_delete_and_guarded_update_race_exactly_one_wins() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_db(dir.path());
    for round in 0..ROUNDS {
        let handle = format!("d{round}");
        let id = create_account(&db, &handle, 7);
        let results = race(
            &db,
            |i| {
                if i % 2 == 0 {
                    format!(r#"Account.filter(.handle == "{handle}" && .step == 7).delete()"#)
                } else {
                    format!(
                        r#"Account.filter(.handle == "{handle}" && .step == 7).update({{ step: 8 }})"#
                    )
                }
            },
            |db, q| run(db, q).map(|o| matches!(o, QueryOutput::Done) || rows(&o) == 1),
        );
        let updates_applied = results
            .iter()
            .enumerate()
            .filter(|(i, r)| i % 2 == 1 && *r.as_ref().expect("no errors under contention"))
            .count();
        let alive = db.get("Account", id).is_ok();
        if alive {
            // An update won; every delete then saw `step == 8` and skipped.
            assert_eq!(updates_applied, 1, "round {round}");
            assert_eq!(read_step(&db, id), 8);
        } else {
            // A delete won; no update may report a write to a deleted row.
            assert_eq!(updates_applied, 0, "round {round}");
        }
    }
}
