//! Engine-level tests for the full-text inverted index: the write paths
//! keep the `f:` / `l:` rows exactly in step with the objects, the corpus
//! stats follow every commit, and `Database::fulltext_search` ranks over
//! them. No server, no query language — those layers are tested where they
//! live.

use std::collections::HashSet;

use rhypedb_schema::parser::parse_schema;
use rhypedb_storage::key::KeyBuilder;
use tempfile::TempDir;

use crate::database::Database;
use crate::error::EngineError;
use crate::fulltext::CorpusStats;
use crate::object::{FieldMap, Value};

const SCHEMA: &str = r#"
    type Owner {
        name: String
    }
    type Note {
        title: String @fulltext
        body: String @fulltext(positions: false)
        tag: String @unique
        n: i64
        owner: Owner @on_delete(cascade)
    }
"#;

fn open(dir: &TempDir) -> std::sync::Arc<Database> {
    Database::open(parse_schema(SCHEMA).unwrap(), dir.path()).unwrap()
}

/// Open with the background builder DISABLED: markers are reconciled but no
/// backfill / sweep runs, so the pre-backfill state stays observable.
fn open_no_build(dir: &TempDir, sdl: &str) -> std::sync::Arc<Database> {
    Database::open_with_options(
        parse_schema(sdl).unwrap(),
        dir.path(),
        crate::database::OpenOptions {
            background_fulltext_build: false,
            background_cover_refresh: false,
            ..Default::default()
        },
    )
    .unwrap()
}

fn open_sdl(dir: &TempDir, sdl: &str) -> std::sync::Arc<Database> {
    Database::open(parse_schema(sdl).unwrap(), dir.path()).unwrap()
}

const BUILD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn state_of(db: &Database, field: &str) -> crate::fulltext::BuildState {
    db.fulltext_field("Note", field).unwrap().progress.state()
}

/// Every live full-text build marker: `(type_id, field_id) → marker`.
fn markers(db: &Database) -> Vec<((u64, u64), crate::fulltext::build::BuildMarker)> {
    let snap = db.storage().read_snapshot();
    db.storage()
        .scan_prefix_at(snap, &KeyBuilder::catalog_fulltext_marker_prefix())
        .unwrap()
        .into_iter()
        .map(|(k, v)| {
            (
                KeyBuilder::catalog_fulltext_marker_ids(&k).unwrap(),
                crate::fulltext::build::BuildMarker::decode(&v).unwrap(),
            )
        })
        .collect()
}

/// Search results as `(id, score)` for equality across databases.
fn scored(db: &Database, field: &str, q: &str) -> Vec<(u64, f32)> {
    db.fulltext_search("Note", field, q, 100, None, None)
        .unwrap()
        .hits
        .into_iter()
        .map(|h| (h.object_id, h.score))
        .collect()
}

fn fields(pairs: &[(&str, Value)]) -> FieldMap {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

fn note(db: &Database, title: &str, body: &str, tag: &str) -> u64 {
    db.create(
        "Note",
        fields(&[("title", s(title)), ("body", s(body)), ("tag", s(tag)), ("n", Value::I64(1))]),
    )
    .unwrap()
    .id
}

fn ids(db: &Database, field: &str, q: &str, k: usize) -> Vec<u64> {
    db.fulltext_search("Note", field, q, k, None, None)
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.object_id)
        .collect()
}

fn stats(db: &Database, field: &str) -> CorpusStats {
    db.fulltext_field("Note", field).unwrap().stats.snapshot()
}

/// Every live `f:` and `l:` row of `Note.<field>` (all generations).
fn raw_rows(db: &Database, field: &str) -> (usize, usize) {
    let type_id = db.type_ids()["Note"];
    let field_id = db.field_ids()[&format!("Note.{field}")];
    let snap = db.storage().read_snapshot();
    let f = db
        .storage()
        .scan_prefix_at(snap, &KeyBuilder::fulltext_field_all_generations_prefix(type_id, field_id))
        .unwrap()
        .len();
    let l = db
        .storage()
        .scan_prefix_at(snap, &KeyBuilder::fulltext_doc_all_generations_prefix(type_id, field_id))
        .unwrap()
        .len();
    (f, l)
}

#[test]
fn create_indexes_and_search_ranks() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let a = note(&db, "Invoice 4471 draft", "pay the invoice", "a");
    let b = note(&db, "Café receipts", "the invoice for the invoice run", "b");
    let c = note(&db, "groceries", "milk and eggs", "c");

    // Case + diacritic insensitive, analyzed on both sides.
    assert_eq!(ids(&db, "title", "INVOICE", 10), vec![a]);
    assert_eq!(ids(&db, "title", "cafe", 10), vec![b]);
    assert_eq!(ids(&db, "title", "Café", 10), vec![b]);
    // Body: tf 2 (b) outranks tf 1 (a); c never appears.
    assert_eq!(ids(&db, "body", "invoice", 10), vec![b, a]);
    assert_eq!(ids(&db, "body", "invoice", 1), vec![b]);
    // OR semantics + a term that is nowhere.
    let mut both = ids(&db, "body", "invoice eggs", 10);
    both.sort_unstable();
    assert_eq!(both, vec![a, b, c]);
    assert!(ids(&db, "title", "nowhere", 10).is_empty());
    // Scores are finite and descending.
    let hits = db.fulltext_search("Note", "body", "invoice", 10, None, None).unwrap().hits;
    assert!(hits[0].score > hits[1].score && hits[1].score > 0.0);

    // Stats: 3 docs per field; body has 3 + 6 + 3 tokens.
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 3, total_tokens: 3 + 2 + 1 });
    assert_eq!(stats(&db, "body"), CorpusStats { doc_count: 3, total_tokens: 3 + 6 + 3 });
    // Raw rows: one posting per distinct term per doc + one `l:` per doc.
    // body distinct terms: {pay,the,invoice}=3, {the,invoice,for,run}=4, {milk,and,eggs}=3.
    assert_eq!(raw_rows(&db, "body"), (10, 3));
}

#[test]
fn required_terms_phrases_and_field_errors() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let a = note(&db, "distributed consensus is hard", "x", "a");
    let b = note(&db, "consensus in distributed systems", "x", "b");
    let _c = note(&db, "hard systems", "x", "c");

    assert_eq!(ids(&db, "title", "+consensus +hard", 10), vec![a]);
    assert_eq!(ids(&db, "title", "\"distributed consensus\"", 10), vec![a]);
    assert!(ids(&db, "title", "\"consensus distributed\"", 10).is_empty());
    let mut both = ids(&db, "title", "consensus", 10);
    both.sort_unstable();
    assert_eq!(both, vec![a, b]);

    // Phrase on a positions:false field is a clear error, not a silent miss.
    let err = db.fulltext_search("Note", "body", "\"a b\"", 10, None, None).unwrap_err();
    assert!(matches!(err, EngineError::FulltextQuery(ref m) if m.contains("positions: false")), "{err}");
    // A malformed / empty query too.
    assert!(matches!(
        db.fulltext_search("Note", "title", "\"open", 10, None, None).unwrap_err(),
        EngineError::FulltextQuery(_)
    ));
    assert!(matches!(
        db.fulltext_search("Note", "title", "...", 10, None, None).unwrap_err(),
        EngineError::FulltextQuery(_)
    ));
    // Field without @fulltext / unknown field / unknown type.
    assert!(matches!(
        db.fulltext_search("Note", "tag", "a", 10, None, None).unwrap_err(),
        EngineError::FulltextNotEnabled { ref type_name, ref field } if type_name == "Note" && field == "tag"
    ));
    assert!(matches!(
        db.fulltext_search("Note", "nosuch", "a", 10, None, None).unwrap_err(),
        EngineError::FieldNotFound { .. }
    ));
    assert!(matches!(
        db.fulltext_search("Nope", "title", "a", 10, None, None).unwrap_err(),
        EngineError::TypeNotFound(_)
    ));
    // k = 0 → nothing, no error.
    assert!(ids(&db, "title", "consensus", 0).is_empty());
}

#[test]
fn null_or_absent_values_are_not_documents() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.create("Note", fields(&[("tag", s("a")), ("n", Value::I64(1))])).unwrap();
    db.create("Note", fields(&[("title", Value::Null), ("tag", s("b")), ("n", Value::I64(1))]))
        .unwrap();
    // A value that analyzes to zero tokens IS a (zero-length) document.
    let empty = note(&db, "...", "", "c");
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 0 });
    assert_eq!(raw_rows(&db, "title"), (0, 1));
    assert!(ids(&db, "title", "anything", 10).is_empty());
    // Updating the zero-length doc into a real one moves it to 1 token.
    db.update("Note", empty, fields(&[("title", s("hello"))])).unwrap();
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 1 });
    assert_eq!(ids(&db, "title", "hello", 10), vec![empty]);
}

#[test]
fn update_stages_only_the_difference_and_keeps_index_exact() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let id = note(&db, "alpha beta gamma", "unchanged body", "a");
    let before_body = raw_rows(&db, "body");

    // beta dropped, delta added, alpha/gamma positions shift.
    db.update("Note", id, fields(&[("title", s("delta alpha gamma"))])).unwrap();
    assert!(ids(&db, "title", "beta", 10).is_empty());
    assert_eq!(ids(&db, "title", "delta", 10), vec![id]);
    assert_eq!(ids(&db, "title", "\"alpha gamma\"", 10), vec![id]);
    assert!(ids(&db, "title", "\"alpha beta\"", 10).is_empty());
    assert_eq!(raw_rows(&db, "title"), (3, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 3 });
    // The body index is untouched by a title update.
    assert_eq!(raw_rows(&db, "body"), before_body);

    // Length change with the same term set rewrites postings (doc_len rides
    // on every posting) — search still finds it, rows stay exact.
    db.update("Note", id, fields(&[("title", s("delta alpha gamma gamma"))])).unwrap();
    assert_eq!(raw_rows(&db, "title"), (3, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 4 });
    assert_eq!(ids(&db, "title", "\"gamma gamma\"", 10), vec![id]);

    // Updating an unrelated field leaves everything alone.
    db.update("Note", id, fields(&[("n", Value::I64(2))])).unwrap();
    assert_eq!(raw_rows(&db, "title"), (3, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 4 });

    // Update to null removes the document entirely; back to a value re-adds.
    db.update("Note", id, fields(&[("title", Value::Null)])).unwrap();
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 0, total_tokens: 0 });
    assert!(ids(&db, "title", "delta", 10).is_empty());
    db.update("Note", id, fields(&[("title", s("back again"))])).unwrap();
    assert_eq!(raw_rows(&db, "title"), (2, 1));
    assert_eq!(ids(&db, "title", "again", 10), vec![id]);
}

#[test]
fn delete_and_cascade_delete_remove_every_row() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let owner = db.create("Owner", fields(&[("name", s("o"))])).unwrap().id;
    let a = note(&db, "alpha beta", "one two", "a");
    let b = db
        .create(
            "Note",
            fields(&[
                ("title", s("beta gamma")),
                ("body", s("three")),
                ("tag", s("b")),
                ("n", Value::I64(1)),
                ("owner", Value::U64(owner)),
            ]),
        )
        .unwrap()
        .id;
    assert_eq!(raw_rows(&db, "title"), (4, 2));

    db.delete("Note", a).unwrap();
    assert_eq!(ids(&db, "title", "beta", 10), vec![b]);
    assert!(ids(&db, "title", "alpha", 10).is_empty());
    assert_eq!(raw_rows(&db, "title"), (2, 1));
    assert_eq!(raw_rows(&db, "body"), (1, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 2 });

    // Cascade: deleting the owner deletes note b through @on_delete(cascade)
    // and its index rows with it.
    db.delete("Owner", owner).unwrap();
    assert!(db.get("Note", b).is_err());
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    assert_eq!(raw_rows(&db, "body"), (0, 0));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 0, total_tokens: 0 });
    assert_eq!(stats(&db, "body"), CorpusStats { doc_count: 0, total_tokens: 0 });
    assert!(ids(&db, "title", "beta gamma", 10).is_empty());
}

#[test]
fn batch_create_indexes_all_rows_and_a_failed_batch_indexes_none() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let rows = vec![
        fields(&[("title", s("one alpha")), ("tag", s("a")), ("n", Value::I64(1))]),
        fields(&[("title", s("two alpha")), ("tag", s("b")), ("n", Value::I64(1))]),
    ];
    let created = db.create_batch("Note", rows).unwrap();
    let mut found = ids(&db, "title", "alpha", 10);
    found.sort_unstable();
    assert_eq!(found, created.iter().map(|o| o.id).collect::<Vec<_>>());
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 2, total_tokens: 4 });

    // Second row violates @unique → the whole batch aborts → no index rows,
    // no stats movement, nothing searchable.
    let bad = vec![
        fields(&[("title", s("three zeta")), ("tag", s("c")), ("n", Value::I64(1))]),
        fields(&[("title", s("four zeta")), ("tag", s("a")), ("n", Value::I64(1))]),
    ];
    assert!(matches!(db.create_batch("Note", bad).unwrap_err(), EngineError::UniqueViolation { .. }));
    assert!(ids(&db, "title", "zeta", 10).is_empty());
    assert_eq!(raw_rows(&db, "title"), (4, 2));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 2, total_tokens: 4 });

    // Same for a single failed create and a failed update.
    assert!(db
        .create("Note", fields(&[("title", s("five zeta")), ("tag", s("a")), ("n", Value::I64(1))]))
        .is_err());
    assert!(ids(&db, "title", "zeta", 10).is_empty());
    let first = created[0].id;
    assert!(db
        .update("Note", first, fields(&[("title", s("six zeta")), ("tag", s("b"))]))
        .is_err());
    assert!(ids(&db, "title", "zeta", 10).is_empty());
    assert_eq!(ids(&db, "title", "one", 10), vec![first]);
    assert_eq!(raw_rows(&db, "title"), (4, 2));
}

#[test]
fn reopen_preserves_index_and_rebuilds_stats_exactly() {
    let dir = TempDir::new().unwrap();
    let (a, b, before) = {
        let db = open(&dir);
        let a = note(&db, "the invoice for the invoice run", "x", "a");
        let b = note(&db, "invoice 4471 is overdue", "y", "b");
        let _c = note(&db, "unrelated", "z", "c");
        db.update("Note", b, fields(&[("title", s("invoice 4471 overdue"))])).unwrap();
        let _d = note(&db, "gone soon", "w", "d");
        db.delete("Note", _d).unwrap();
        (a, b, db.fulltext_search("Note", "title", "invoice +4471", 10, None, None).unwrap())
    };
    let db = open(&dir);
    // Stats re-derived from the `l:` rows equal the live-maintained ones
    // (3 docs: 6 + 3 + 1 tokens), so every score is bit-identical.
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 3, total_tokens: 10 });
    let after = db.fulltext_search("Note", "title", "invoice +4471", 10, None, None).unwrap();
    assert_eq!(before, after);
    assert_eq!(after.hits[0].object_id, b);
    assert_eq!(ids(&db, "title", "invoice", 10), vec![a, b]);
    // And the index keeps working across the reopen.
    let e = note(&db, "invoice again", "v", "e");
    assert_eq!(ids(&db, "title", "again", 10), vec![e]);
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 4, total_tokens: 12 });
}

#[test]
fn restrict_set_applies_before_top_k_and_scan_count_is_reported() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let a = note(&db, "x x x", "b", "a");
    let b = note(&db, "x x", "b", "b");
    let c = note(&db, "x", "b", "c");
    let d = note(&db, "x", "b", "d");
    assert_eq!(ids(&db, "title", "x", 2), vec![a, b]);
    let restrict: HashSet<u64> = [c, d].into_iter().collect();
    let res = db.fulltext_search("Note", "title", "x", 2, Some(&restrict), None).unwrap();
    assert_eq!(res.hits.iter().map(|h| h.object_id).collect::<Vec<_>>(), vec![c, d]);
    // Four postings for `x` were examined regardless of the restriction.
    assert_eq!(res.postings_scanned, 4);
    let none: HashSet<u64> = HashSet::new();
    assert!(db.fulltext_search("Note", "title", "x", 2, Some(&none), None).unwrap().hits.is_empty());
}

#[test]
fn restore_objects_rebuilds_the_index_like_create() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.restore_objects(
        "Note",
        vec![
            (10, fields(&[("title", s("restored alpha")), ("tag", s("a")), ("n", Value::I64(1))])),
            (20, fields(&[("title", s("restored beta")), ("tag", s("b")), ("n", Value::I64(1))])),
        ],
        true,
    )
    .unwrap();
    let mut found = ids(&db, "title", "restored", 10);
    found.sort_unstable();
    assert_eq!(found, vec![10, 20]);
    assert_eq!(ids(&db, "title", "beta", 10), vec![20]);
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 2, total_tokens: 4 });
}

const SCHEMA_WITHOUT_TITLE_INDEX: &str = r#"
    type Owner {
        name: String
    }
    type Note {
        title: String
        body: String @fulltext(positions: false)
        tag: String @unique
        n: i64
        owner: Owner @on_delete(cascade)
    }
"#;

/// Objects written before `@fulltext` was added have a value but no index
/// rows. An update must index the NEW value in full (not diff against a
/// never-indexed old one) and a delete must not move stats it never joined.
#[test]
fn pre_directive_objects_are_unindexed_until_written_again() {
    let dir = TempDir::new().unwrap();
    let (a, b) = {
        let db = Database::open(parse_schema(SCHEMA_WITHOUT_TITLE_INDEX).unwrap(), dir.path())
            .unwrap();
        (note(&db, "alpha beta", "x", "a"), note(&db, "keep me", "y", "b"))
    };
    // Reopen with `title: String @fulltext`, builder OFF so nothing gets
    // backfilled behind the test's back.
    let db = open_no_build(&dir, SCHEMA);
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Building);
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 0, total_tokens: 0 });
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    // Searching a Building index refuses with progress.
    assert!(matches!(
        db.fulltext_search("Note", "title", "alpha", 10, None, None).unwrap_err(),
        EngineError::FulltextIndexBuilding { indexed: 0, .. }
    ));

    // Update A: the shared term `alpha` MUST be indexed too, and the l: row
    // written even though old and new lengths are equal.
    db.update("Note", a, fields(&[("title", s("alpha gamma"))])).unwrap();
    assert_eq!(raw_rows(&db, "title"), (2, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 2 });

    // Delete B (never indexed): no tombstones, and the stats — which never
    // counted it — stay put.
    db.delete("Note", b).unwrap();
    assert_eq!(raw_rows(&db, "title"), (2, 1));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 1, total_tokens: 2 });
    // Delete A (indexed): everything goes.
    db.delete("Note", a).unwrap();
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 0, total_tokens: 0 });
}

/// Ground truth for backfill tests: the same rows written into a database
/// that had the directive from the start.
fn clean_build(sdl: &str, rows: &[(u64, Option<&str>, &str)]) -> (TempDir, std::sync::Arc<Database>) {
    let dir = TempDir::new().unwrap();
    let db = open_sdl(&dir, sdl);
    let batch: Vec<(u64, FieldMap)> = rows
        .iter()
        .map(|(id, title, tag)| {
            let mut f = fields(&[("body", s("b")), ("tag", s(tag)), ("n", Value::I64(1))]);
            if let Some(t) = title {
                f.insert("title".into(), s(t));
            }
            (*id, f)
        })
        .collect();
    db.restore_objects("Note", batch, true).unwrap();
    (dir, db)
}

fn sample_rows(n: u64) -> Vec<(u64, Option<String>, String)> {
    (1..=n)
        .map(|i| {
            let title = match i % 7 {
                0 => None, // not a document
                1 => Some(String::new()), // zero-length document
                k => Some(format!("invoice {} draft {} common w{}", i, k, i % 13)),
            };
            (i, title, format!("t{i}"))
        })
        .collect()
}

#[test]
fn adding_the_directive_backfills_existing_objects_to_a_clean_build() {
    let rows = sample_rows(1300); // > 2 chunks of 512
    let dir = TempDir::new().unwrap();
    {
        let db = open_sdl(&dir, SCHEMA_WITHOUT_TITLE_INDEX);
        let batch: Vec<(u64, FieldMap)> = rows
            .iter()
            .map(|(id, title, tag)| {
                let mut f = fields(&[("body", s("b")), ("tag", s(tag)), ("n", Value::I64(1))]);
                if let Some(t) = title {
                    f.insert("title".into(), s(t));
                }
                (*id, f)
            })
            .collect();
        db.restore_objects("Note", batch, true).unwrap();
    }
    let db = open(&dir);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT), "build did not finish");
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);
    let st = db.fulltext_status();
    let title = st.iter().find(|s| s.name == "Note.title").unwrap();
    assert_eq!((title.indexed, title.total), (1300, 1300));
    assert_eq!(title.generation, 0);

    let borrowed: Vec<(u64, Option<&str>, &str)> =
        rows.iter().map(|(i, t, g)| (*i, t.as_deref(), g.as_str())).collect();
    let (_cdir, clean) = clean_build(SCHEMA, &borrowed);
    assert_eq!(raw_rows(&db, "title"), raw_rows(&clean, "title"));
    assert_eq!(stats(&db, "title"), stats(&clean, "title"));
    for q in ["invoice", "+draft +w3", "\"invoice 700\"", "w12 common", "nothing"] {
        assert_eq!(scored(&db, "title", q), scored(&clean, "title", q), "{q}");
    }
    // Marker persisted as Built; a reopen needs no build.
    let db2 = {
        drop(db);
        open(&dir)
    };
    assert_eq!(state_of(&db2, "title"), crate::fulltext::BuildState::Built);
    assert!(db2.wait_for_fulltext_builds(std::time::Duration::from_millis(10)));
    assert_eq!(scored(&db2, "title", "invoice"), scored(&clean, "title", "invoice"));
}

#[test]
fn writes_before_and_during_the_backfill_converge() {
    let rows = sample_rows(1100);
    let dir = TempDir::new().unwrap();
    {
        let db = open_sdl(&dir, SCHEMA_WITHOUT_TITLE_INDEX);
        let batch: Vec<(u64, FieldMap)> = rows
            .iter()
            .map(|(id, title, tag)| {
                let mut f = fields(&[("body", s("b")), ("tag", s(tag)), ("n", Value::I64(1))]);
                if let Some(t) = title {
                    f.insert("title".into(), s(t));
                }
                (*id, f)
            })
            .collect();
        db.restore_objects("Note", batch, true).unwrap();
    }
    // Phase 1: directive added, builder OFF — live writes hit a partially
    // indexed type (update of an unindexed object, delete, new create).
    {
        let db = open_no_build(&dir, SCHEMA);
        db.update("Note", 2, fields(&[("title", s("rewritten two"))])).unwrap();
        db.delete("Note", 3).unwrap();
        db.update("Note", 7, fields(&[("title", s("was null now set"))])).unwrap();
        note(&db, "brand new eleven", "new", "n1");
        assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Building);
    }
    // Phase 2: builder ON — the backfill covers everything else, and a
    // writer thread keeps mutating while it runs.
    let db = open(&dir);
    let writer = {
        let db = std::sync::Arc::clone(&db);
        std::thread::spawn(move || {
            for i in 0..60u64 {
                let id = 100 + (i * 17) % 900;
                // Pre-directive objects: some are being backfilled right now.
                let _ = db.update("Note", id, fields(&[("title", s(&format!("live edit {i}")))]));
            }
        })
    };
    writer.join().unwrap();
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);

    // Invariant check against the objects themselves: every object with a
    // String title is exactly one indexed document whose current terms are
    // all findable, and there are no orphan rows.
    let mut expect_docs = 0u64;
    let mut expect_postings = 0usize;
    let mut expect_tokens = 0u64;
    let snap = db.storage().read_snapshot();
    let mut cursor = 0;
    loop {
        let chunk = db.scan_chunk("Note", snap, cursor, 500).unwrap();
        for obj in &chunk.objects {
            if let Some(Value::String(t)) = obj.fields.get("title") {
                let doc = crate::fulltext::tokenize_for_index(db.fulltext_field("Note", "title").unwrap(), t);
                expect_docs += 1;
                expect_postings += doc.terms.len();
                expect_tokens += doc.doc_len as u64;
                // Findability: every term of a sample of objects (a full
                // sweep is O(docs × postings) in a debug build).
                if obj.id % 9 == 0 || obj.id < 12 || obj.id > 1090 {
                    for (term, _) in &doc.terms {
                        assert!(
                            ids(&db, "title", &format!("+{term}"), 5000).contains(&obj.id),
                            "object {} not findable by {term:?}",
                            obj.id
                        );
                    }
                }
            }
        }
        match chunk.next_cursor {
            Some(c) if chunk.more => cursor = c,
            _ => break,
        }
    }
    assert_eq!(raw_rows(&db, "title"), (expect_postings, expect_docs as usize));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: expect_docs, total_tokens: expect_tokens });
    assert!(ids(&db, "title", "rewritten", 10).contains(&2));
    assert!(ids(&db, "title", "eleven", 10).len() == 1);
}

#[test]
fn positions_change_bumps_the_generation_and_sweeps_the_old_rows() {
    const NO_POSITIONS: &str = r#"
        type Owner { name: String }
        type Note {
            title: String @fulltext(positions: false)
            body: String @fulltext(positions: false)
            tag: String @unique
            n: i64
            owner: Owner @on_delete(cascade)
        }
    "#;
    let dir = TempDir::new().unwrap();
    {
        let db = open_sdl(&dir, NO_POSITIONS);
        for i in 0..700 {
            note(&db, &format!("distributed consensus {i}"), "b", &format!("t{i}"));
        }
        assert!(db.fulltext_search("Note", "title", "\"distributed consensus\"", 5, None, None).is_err());
        assert_eq!(markers(&db).iter().find(|(k, _)| k.1 == db.field_ids()["Note.title"]).unwrap().1.generation, 0);
    }
    // Reopen with positions ON (= SCHEMA): new generation, rebuild, sweep.
    let db = open(&dir);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);
    assert_eq!(db.fulltext_field("Note", "title").unwrap().generation, 1);
    assert_eq!(ids(&db, "title", "\"distributed consensus\"", 1000).len(), 700);
    // Exactly one generation's rows remain: 3 distinct terms per doc.
    assert_eq!(raw_rows(&db, "title"), (700 * 3, 700));
    let m = markers(&db)
        .into_iter()
        .find(|(k, _)| k.1 == db.field_ids()["Note.title"])
        .unwrap()
        .1;
    assert_eq!((m.generation, m.positions, m.stale_generations), (1, true, false));
    assert_eq!(m.state, crate::fulltext::BuildState::Built);
}

#[test]
fn removing_the_directive_drops_every_row_and_the_marker() {
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        for i in 0..50 {
            note(&db, &format!("gone soon {i}"), "b", &format!("t{i}"));
        }
        assert_eq!(markers(&db).len(), 2);
    }
    let db = open_sdl(&dir, SCHEMA_WITHOUT_TITLE_INDEX);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    assert_eq!(raw_rows(&db, "body"), (50, 50));
    let ms = markers(&db);
    assert_eq!(ms.len(), 1, "only the body marker remains: {ms:?}");
    assert_eq!(ms[0].0.1, db.field_ids()["Note.body"]);
    // And re-adding it later rebuilds from scratch (new generation).
    drop(db);
    let db = open(&dir);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(ids(&db, "title", "gone", 100).len(), 50);
}

#[test]
fn a_build_resumes_from_the_persisted_cursor() {
    let dir = TempDir::new().unwrap();
    let field_id;
    {
        let db = open(&dir);
        for i in 0..900 {
            note(&db, &format!("resume {i}"), "b", &format!("t{i}"));
        }
        field_id = db.field_ids()["Note.title"];
        // Rewind the marker to Building at a mid-way cursor, as a crash
        // between a chunk's commits would leave it.
        let type_id = db.type_ids()["Note"];
        let m = crate::fulltext::build::BuildMarker {
            state: crate::fulltext::BuildState::Building,
            generation: 0,
            cursor: 450,
            positions: true,
            stale_generations: false,
            analyzer: "simple".into(),
        };
        let mut txn = db.storage().begin_txn();
        db.storage()
            .put(&mut txn, &KeyBuilder::catalog_fulltext_marker(type_id, field_id), m.encode())
            .unwrap();
        db.storage().commit(&mut txn).unwrap();
    }
    let db = open(&dir);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);
    // No duplicates, nothing lost: 2 distinct terms per doc, 900 docs.
    assert_eq!(raw_rows(&db, "title"), (1800, 900));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 900, total_tokens: 1800 });
    let st = db.fulltext_status();
    let t = st.iter().find(|s| s.name == "Note.title").unwrap();
    assert!(t.indexed <= 450 && t.total == 900, "{t:?}");
    assert_eq!(markers(&db).into_iter().find(|(k, _)| k.1 == field_id).unwrap().1.state, crate::fulltext::BuildState::Built);
}

#[test]
fn scan_budget_refuses_before_decoding() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    for t in ["a", "b", "c", "d"] {
        note(&db, "common word", "z", t);
    }
    let err = db
        .fulltext_search("Note", "title", "common", 10, None, Some(3))
        .unwrap_err();
    assert!(matches!(err, EngineError::FulltextScanBudgetExceeded { limit: 3, .. }), "{err}");
    // Exactly at the budget is fine; a second term adds to the same budget.
    assert_eq!(db.fulltext_search("Note", "title", "common", 10, None, Some(4)).unwrap().hits.len(), 4);
    assert!(db.fulltext_search("Note", "title", "common word", 10, None, Some(7)).is_err());
    assert_eq!(
        db.fulltext_search("Note", "title", "common word", 10, None, Some(8)).unwrap().postings_scanned,
        8
    );
}
