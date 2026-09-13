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
    // Reopen with `title: String @fulltext`.
    let db = open(&dir);
    assert_eq!(stats(&db, "title"), CorpusStats { doc_count: 0, total_tokens: 0 });
    assert_eq!(raw_rows(&db, "title"), (0, 0));
    assert!(ids(&db, "title", "alpha", 10).is_empty());

    // Update A: the shared term `alpha` MUST be indexed too, and the l: row
    // written even though old and new lengths are equal.
    db.update("Note", a, fields(&[("title", s("alpha gamma"))])).unwrap();
    assert_eq!(ids(&db, "title", "alpha", 10), vec![a]);
    assert_eq!(ids(&db, "title", "gamma", 10), vec![a]);
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
