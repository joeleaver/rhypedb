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

/// A delete / set-to-null / rewrite racing the first backfill chunk from
/// another thread: the writer blocks on the maintenance lock until the chunk
/// has committed (proven by observing the chunk's rows before its own write
/// lands), then its write undoes the chunk's rows for that object — the final
/// index equals a clean build of the final state, and no user write fails.
#[test]
fn deletes_and_null_sets_racing_a_backfill_chunk_never_fail_or_resurrect() {
    let rows = sample_rows(700);
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
    let db = open_no_build(&dir, SCHEMA);
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Building);
    // Racing the first chunk (ids 1..=512) from another thread: delete 300,
    // null 301, rewrite 302. The hook fires while the chunk HOLDS the
    // maintenance lock, so the racer's first write parks until the chunk
    // commits; the hook only waits for the racer to have started.
    let racer_handle: std::sync::Arc<parking_lot::Mutex<Option<std::thread::JoinHandle<bool>>>> =
        std::sync::Arc::new(parking_lot::Mutex::new(None));
    {
        let racer = std::sync::Arc::clone(&db);
        let holder = std::sync::Arc::clone(&racer_handle);
        let mut fired = false;
        *db.fulltext_builder_test_hook() = Some(Box::new(move |cursor: u64| {
            if fired || cursor != 0 {
                return;
            }
            fired = true;
            let racer = std::sync::Arc::clone(&racer);
            let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
            *holder.lock() = Some(std::thread::spawn(move || {
                let type_id = racer.type_ids()["Note"];
                let field_id = racer.field_ids()["Note.title"];
                started_tx.send(()).unwrap();
                // Blocks on the maintenance lock until the chunk commits.
                racer.delete("Note", 300).unwrap();
                // Proof the delete ran AFTER the chunk: its neighbours' `l:`
                // rows (written by that chunk) are visible now.
                let snap = racer.storage().read_snapshot();
                let neighbours_indexed = racer
                    .storage()
                    .get_at(snap, &KeyBuilder::fulltext_doc(type_id, field_id, 0, 299))
                    .unwrap()
                    .is_some();
                racer.update("Note", 301, fields(&[("title", Value::Null)])).unwrap();
                racer.update("Note", 302, fields(&[("title", s("rewritten after chunk"))])).unwrap();
                neighbours_indexed
            }));
            started_rx.recv().unwrap();
            // Let the racer reach (and park on) the lock this chunk holds.
            std::thread::sleep(std::time::Duration::from_millis(50));
        }));
    }
    db.run_fulltext_tasks_inline();
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);
    assert_eq!(db.fulltext_tasks_failed(), 0);
    let waited_for_chunk = racer_handle.lock().take().unwrap().join().unwrap();
    assert!(waited_for_chunk, "the racing writer must have waited for the chunk to commit");

    // Expected final state = rows minus 300, 301 nulled, 302 rewritten.
    let mut expected: Vec<(u64, Option<String>, String)> =
        rows.iter().filter(|(id, _, _)| *id != 300).cloned().collect();
    for r in expected.iter_mut() {
        if r.0 == 301 {
            r.1 = None;
        }
        if r.0 == 302 {
            r.1 = Some("rewritten after chunk".into());
        }
    }
    let borrowed: Vec<(u64, Option<&str>, &str)> =
        expected.iter().map(|(i, t, g)| (*i, t.as_deref(), g.as_str())).collect();
    let (_cdir, clean) = clean_build(SCHEMA, &borrowed);
    assert_eq!(raw_rows(&db, "title"), raw_rows(&clean, "title"));
    assert_eq!(stats(&db, "title"), stats(&clean, "title"));
    // 300's and 301's old titles are gone; 302 is findable by its new text only.
    assert!(ids(&db, "title", "+invoice +300", 10).is_empty());
    assert!(ids(&db, "title", "+invoice +301", 10).is_empty());
    assert!(ids(&db, "title", "+invoice +302", 10).is_empty());
    assert_eq!(ids(&db, "title", "+rewritten +chunk", 10), vec![302]);
    assert_eq!(scored(&db, "title", "invoice"), scored(&clean, "title", "invoice"));
}

#[test]
fn status_lists_a_removed_field_as_dropping_until_swept() {
    let dir = TempDir::new().unwrap();
    let field_id = {
        let db = open(&dir);
        for i in 0..20 {
            note(&db, &format!("gone {i}"), "b", &format!("t{i}"));
        }
        db.field_ids()["Note.title"]
    };
    let db = open_no_build(&dir, SCHEMA_WITHOUT_TITLE_INDEX);
    let st = db.fulltext_status();
    let dropping = st
        .iter()
        .find(|s| s.state == crate::fulltext::BuildState::Dropping)
        .expect("removed field shows as dropping");
    assert_eq!(dropping.name, format!("Note.field#{field_id}"));
    assert_eq!(st.iter().filter(|s| s.name == "Note.body").count(), 1);
    drop(db);
    let db = open_sdl(&dir, SCHEMA_WITHOUT_TITLE_INDEX);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert!(db.fulltext_status().iter().all(|s| s.state == crate::fulltext::BuildState::Built));
    assert_eq!(raw_rows(&db, "title"), (0, 0));
}

// ---- `english` analyzer (issue #17) ----

const ENGLISH: &str = r#"
    type Owner { name: String }
    type Note {
        title: String @fulltext(analyzer: "english")
        body: String @fulltext(positions: false)
        tag: String @unique
        n: i64
        owner: Owner @on_delete(cascade)
    }
"#;

#[test]
fn english_analyzer_matches_inflections_on_both_sides() {
    let dir = TempDir::new().unwrap();
    let db = open_sdl(&dir, ENGLISH);
    let a = note(&db, "the security cameras were replaced", "x", "a");
    let b = note(&db, "Camera's battery", "x", "b");
    let c = note(&db, "a camera obscura", "x", "c");
    let d = note(&db, "running shoes", "x", "d");

    // Every inflection of the query finds every inflection in the corpus.
    for q in ["camera", "cameras", "Camera's", "CAMERAS", "camera\u{2019}s"] {
        let mut got = ids(&db, "title", q, 10);
        got.sort_unstable();
        assert_eq!(got, vec![a, b, c], "{q:?}");
    }
    for q in ["run", "runs", "running"] {
        assert_eq!(ids(&db, "title", q, 10), vec![d], "{q:?}");
    }
    // A phrase stems both sides: "security camera" matches "security cameras".
    assert_eq!(ids(&db, "title", "\"security camera\"", 10), vec![a]);
    assert_eq!(ids(&db, "title", "+\"security cameras\" +replace", 10), vec![a]);
    assert!(ids(&db, "title", "\"camera security\"", 10).is_empty());
    // The index holds ONE term per stem: a has 5 distinct stems (the, secur,
    // camera, were, replac), b 2 (camera, batteri), c 3 (a, camera, obscura), d 2.
    assert_eq!(raw_rows(&db, "title"), (5 + 2 + 3 + 2, 4));
    // Stats count surface tokens, not stems.
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 4, total_tokens: 5 + 2 + 3 + 2 });
    // Scoring is over the stem: the two "camera" mentions... c has tf 1 and
    // length 3, b has tf 1 and length 2 → b ranks above c on length norm.
    let hits = db.fulltext_search("Note", "title", "cameras", 10, None, None).unwrap().hits;
    assert_eq!(hits[0].object_id, b);
    // `body` keeps the simple analyzer: no stemming there.
    let e = note(&db, "x", "cameras", "e");
    assert_eq!(ids(&db, "body", "cameras", 10), vec![e]);
    assert!(ids(&db, "body", "camera", 10).is_empty());
    // Updates re-index through the same analyzer.
    db.update("Note", d, fields(&[("title", s("walking boots"))])).unwrap();
    assert!(ids(&db, "title", "running", 10).is_empty());
    assert_eq!(ids(&db, "title", "walk", 10), vec![d]);
    assert_eq!(db.fulltext_field("Note", "title").unwrap().analyzer, crate::fulltext::Analyzer::English);
}

#[test]
fn switching_the_analyzer_bumps_the_generation_and_rebuilds() {
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir); // title: simple
        for i in 0..300 {
            note(&db, &format!("security cameras {i}"), "b", &format!("t{i}"));
        }
        assert!(ids(&db, "title", "camera", 10).is_empty(), "simple: no stemming");
        assert_eq!(ids(&db, "title", "cameras", 1000).len(), 300);
        let m = markers(&db).into_iter().find(|(k, _)| k.1 == db.field_ids()["Note.title"]).unwrap().1;
        assert_eq!((m.generation, m.analyzer.as_str()), (0, "simple"));
    }
    // Reopen as english: new generation, backfill through the builder, sweep.
    let db = open_sdl(&dir, ENGLISH);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(state_of(&db, "title"), crate::fulltext::BuildState::Built);
    let ff = db.fulltext_field("Note", "title").unwrap();
    assert_eq!((ff.generation, ff.analyzer), (1, crate::fulltext::Analyzer::English));
    assert_eq!(ids(&db, "title", "camera", 1000).len(), 300);
    assert_eq!(ids(&db, "title", "\"security camera\"", 1000).len(), 300);
    // Exactly one generation's rows remain: 3 distinct stems per doc.
    assert_eq!(raw_rows(&db, "title"), (300 * 3, 300));
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 300, total_tokens: 900 });
    let m = markers(&db).into_iter().find(|(k, _)| k.1 == db.field_ids()["Note.title"]).unwrap().1;
    assert_eq!((m.generation, m.analyzer.as_str(), m.stale_generations), (1, "english", false));
    assert_eq!(m.state, crate::fulltext::BuildState::Built);
    // A write after the switch lands in the new generation only.
    let n = note(&db, "flying drones", "b", "new");
    assert_eq!(ids(&db, "title", "flies", 10), vec![n]);
    assert_eq!(raw_rows(&db, "title"), (300 * 3 + 2, 301));

    // And back to simple: another generation, stems gone.
    drop(db);
    let db = open(&dir);
    assert!(db.wait_for_fulltext_builds(BUILD_TIMEOUT));
    assert_eq!(db.fulltext_field("Note", "title").unwrap().generation, 2);
    assert!(ids(&db, "title", "camera", 10).is_empty());
    assert_eq!(ids(&db, "title", "cameras", 1000).len(), 300);
    assert_eq!(ids(&db, "title", "flying", 10), vec![n]);
}

// ---- prefix terms `cam*` (issue #17) ----

#[test]
fn prefix_terms_expand_over_the_index_and_score_as_one_term() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let a = note(&db, "camera", "x", "a");
    let b = note(&db, "cameras and a camera", "x", "b");
    let c = note(&db, "the campaign", "x", "c");
    let d = note(&db, "calm", "x", "d");
    let _e = note(&db, "nothing here", "x", "e");

    // `cam*` covers camera / cameras / campaign, not calm. Ranking is
    // BM25 over the merged expansion: a (tf 1, len 1) beats b (tf 2, len 4)
    // on length normalization, and both beat c (tf 1, len 2).
    assert_eq!(ids(&db, "title", "cam*", 10), vec![a, b, c]);
    assert_eq!(ids(&db, "title", "cam*", 1), vec![a]);
    // `ca*` widens to calm too; `camp*` narrows to the campaign; exact prefix
    // of a whole term still matches that term.
    assert_eq!(ids(&db, "title", "ca*", 10).len(), 4);
    assert_eq!(ids(&db, "title", "camp*", 10), vec![c]);
    assert_eq!(ids(&db, "title", "campaign*", 10), vec![c]);
    // Case/diacritics fold on the prefix too.
    assert_eq!(ids(&db, "title", "CAMP*", 10), vec![c]);
    // Required prefix intersects; optional prefix unions; both with terms.
    assert_eq!(ids(&db, "title", "+cam* +campaign", 10), vec![c]);
    let mut got = ids(&db, "title", "cam* calm", 10);
    got.sort_unstable();
    assert_eq!(got, vec![a, b, c, d]);
    // Prefix and exact term of the same text are different clauses.
    assert!(ids(&db, "title", "cam", 10).is_empty());
    assert_eq!(ids(&db, "title", "+cam +cam*", 10), Vec::<u64>::new());
    // Nothing under the prefix → empty, not an error.
    assert!(ids(&db, "title", "zz*", 10).is_empty());
    // Restrict applies before top-k, as for terms.
    let restrict: HashSet<u64> = [c, d].into_iter().collect();
    let hits = db.fulltext_search("Note", "title", "cam*", 10, Some(&restrict), None).unwrap();
    assert_eq!(hits.hits.iter().map(|h| h.object_id).collect::<Vec<_>>(), vec![c]);
    // Posting rows are charged to the budget: 4 rows under cam* (a, b×2, c).
    assert_eq!(hits.postings_scanned, 4);
    assert!(matches!(
        db.fulltext_search("Note", "title", "cam*", 10, None, Some(3)).unwrap_err(),
        EngineError::FulltextScanBudgetExceeded { limit: 3, .. }
    ));
    assert!(db.fulltext_search("Note", "title", "cam*", 10, None, Some(4)).is_ok());

    // ONE idf for the expansion: `cam*` scores document a exactly like an
    // index where every expansion were the same word would.
    let prefix_score = |q: &str, id: u64| {
        db.fulltext_search("Note", "title", q, 10, None, None)
            .unwrap()
            .hits
            .into_iter()
            .find(|h| h.object_id == id)
            .map(|h| h.score)
    };
    // df(cam*) = 3 docs vs df(camera) = 2: the exact term is rarer, so it
    // scores a HIGHER than the prefix does (a single-clause idf, not a sum
    // over expansions — a sum would make the prefix score exceed the term's).
    assert!(prefix_score("camera", a).unwrap() > prefix_score("cam*", a).unwrap());

    // Writes keep the expansion live: a delete and an update are reflected.
    db.delete("Note", c).unwrap();
    let mut got = ids(&db, "title", "cam*", 10);
    got.sort_unstable();
    assert_eq!(got, vec![a, b]);
    db.update("Note", d, fields(&[("title", s("camcorder"))])).unwrap();
    assert!(ids(&db, "title", "camc*", 10) == vec![d]);
}

#[test]
fn prefix_terms_under_english_are_stemmed_like_terms() {
    let dir = TempDir::new().unwrap();
    let db = open_sdl(&dir, ENGLISH);
    let a = note(&db, "security cameras", "x", "a");
    let b = note(&db, "tables and chairs", "x", "b");
    // The index holds stems (camera, tabl); the prefix text is stemmed the
    // same way, so a whole inflected word plus `*` still finds them —
    // where an unstemmed `cameras`/`tables` prefix would miss.
    assert_eq!(ids(&db, "title", "cameras*", 10), vec![a]);
    assert_eq!(ids(&db, "title", "tables*", 10), vec![b]);
    assert_eq!(ids(&db, "title", "table*", 10), vec![b]);
    assert_eq!(ids(&db, "title", "tab*", 10), vec![b]);
    assert_eq!(ids(&db, "title", "secur*", 10), vec![a]);
}

#[test]
fn prefix_expansion_cap_and_syntax_errors_are_clear() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    // 64 distinct terms under `tx*` is fine; the 65th tips it over.
    for i in 0..crate::fulltext::MAX_PREFIX_EXPANSION {
        note(&db, &format!("tx{i:03}"), "x", &format!("k{i}"));
    }
    assert_eq!(ids(&db, "title", "tx*", 1000).len(), crate::fulltext::MAX_PREFIX_EXPANSION);
    note(&db, "tx999", "x", "last");
    let err = db.fulltext_search("Note", "title", "tx*", 1000, None, None).unwrap_err();
    assert!(
        matches!(err, EngineError::FulltextQuery(ref m) if m.contains("\"tx*\"") && m.contains("more than 64") && m.contains("longer prefix")),
        "{err}"
    );
    // A longer prefix under the cap works again; the cap counts DISTINCT
    // terms, so many documents sharing a term are fine.
    assert_eq!(ids(&db, "title", "tx00*", 1000).len(), 10);
    for i in 0..100 {
        note(&db, "tx000", "x", &format!("dup{i}"));
    }
    assert_eq!(ids(&db, "title", "tx00*", 1000).len(), 110);
    assert_eq!(ids(&db, "title", "tx000*", 1000).len(), 101);
    // Syntax errors surface verbatim as FulltextQuery.
    for (q, needle) in [
        ("*", "too short"),
        ("c*", "too short"),
        ("tx000 x*", "too short"),
        ("\"tx000 tx*\"", "inside a phrase"),
    ] {
        let err = db.fulltext_search("Note", "title", q, 10, None, None).unwrap_err();
        assert!(matches!(err, EngineError::FulltextQuery(ref m) if m.contains(needle)), "{q:?}: {err}");
    }
    // A prefix on a positions:false field is fine (no positions needed).
    note(&db, "x", "camera", "pf");
    assert_eq!(ids(&db, "body", "cam*", 10).len(), 1);
}
