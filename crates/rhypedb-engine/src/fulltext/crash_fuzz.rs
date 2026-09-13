//! Crash-recovery fuzz for the full-text index.
//!
//! Contract under test (issue #16 acceptance): "WAL replay after a crash
//! yields an index identical to a clean build." Two families:
//!
//! 1. **Write-path sites.** A workload of creates / updates / deletes on a
//!    `@fulltext` type is interrupted at each WAL durability boundary (the
//!    storage injector's `Wal*` sites) after `k` hits, torn down like a
//!    SIGKILL, cold-reopened, and checked with the *derived oracle*: for
//!    every live object, the `f:`/`l:` rows of its current value are exactly
//!    what tokenizing that value yields, there are no orphan rows, and the
//!    corpus stats equal the sum over live objects. Because index rows ride
//!    in the object's own transaction, whichever prefix of the workload
//!    survived, index and objects agree. A second reopen must be identical.
//! 2. **Backfill sites.** A populated type gains `@fulltext`; the backfill is
//!    driven on the test thread and crashed before / after a chunk commit;
//!    after the cold reopen the resumed build must produce a
//!    clean-build-identical index (same rows, stats and scores as a database
//!    that had the directive from the start).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use rhypedb_schema::parser::parse_schema;
use rhypedb_storage::crash_inject::{self, Caught, Mode, Site};
use rhypedb_storage::key::KeyBuilder;
use tempfile::TempDir;

use crate::database::{Database, OpenOptions};
use crate::fulltext::{BuildState, CorpusStats, tokenize_for_index};
use crate::object::{FieldMap, Value};

const WITH_INDEX: &str = r#"
    type Note {
        title: String @fulltext
        n: i64
    }
"#;
const WITHOUT_INDEX: &str = r#"
    type Note {
        title: String
        n: i64
    }
"#;

fn open(dir: &Path, sdl: &str) -> Arc<Database> {
    Database::open_with_options(
        parse_schema(sdl).unwrap(),
        dir,
        OpenOptions {
            background_fulltext_build: false,
            background_cover_refresh: false,
            ..Default::default()
        },
    )
    .unwrap()
}

fn fields(title: Option<&str>, n: i64) -> FieldMap {
    let mut f = FieldMap::new();
    if let Some(t) = title {
        f.insert("title".into(), Value::String(t.into()));
    }
    f.insert("n".into(), Value::I64(n));
    f
}

fn text(i: u64) -> String {
    format!("invoice {} draft {} common w{}", i, i % 5, i % 13)
}

/// Sum of index rows + stats derived from the LIVE objects; asserts every
/// object's current terms are findable. Returns `(postings, docs, tokens)`.
fn derived_expectation(db: &Database) -> (usize, usize, u64) {
    let ff = db.fulltext_field("Note", "title").unwrap();
    let snap = db.storage().read_snapshot();
    let mut cursor = 0;
    let (mut postings, mut docs, mut tokens) = (0usize, 0usize, 0u64);
    loop {
        let chunk = db.scan_chunk("Note", snap, cursor, 256).unwrap();
        for obj in &chunk.objects {
            if let Some(Value::String(t)) = obj.fields.get("title") {
                let doc = tokenize_for_index(ff, t);
                docs += 1;
                postings += doc.terms.len();
                tokens += doc.doc_len as u64;
                for (term, positions) in &doc.terms {
                    let hits = db
                        .fulltext_search("Note", "title", &format!("+{term}"), 10_000, None, None)
                        .unwrap()
                        .hits;
                    assert!(
                        hits.iter().any(|h| h.object_id == obj.id),
                        "object {} not findable by {term:?} after recovery",
                        obj.id
                    );
                    // The stored posting is the CURRENT value's posting (a
                    // stale one from an older value would carry other positions).
                    let key = KeyBuilder::fulltext_posting(
                        db.type_ids()["Note"],
                        ff.field_id,
                        ff.generation,
                        &crate::fulltext::encode_term(term),
                        obj.id,
                    );
                    let raw = db.storage().get_at(snap, &key).unwrap().expect("posting row present");
                    let p = crate::fulltext::posting::decode_posting(&raw).unwrap();
                    assert_eq!(&p.positions, positions, "object {} term {term:?}", obj.id);
                    assert_eq!(p.doc_len, doc.doc_len);
                }
            }
        }
        match chunk.next_cursor {
            Some(c) if chunk.more => cursor = c,
            _ => break,
        }
    }
    (postings, docs, tokens)
}

fn raw_rows(db: &Database) -> (usize, usize) {
    let type_id = db.type_ids()["Note"];
    let field_id = db.field_ids()["Note.title"];
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

fn stats(db: &Database) -> CorpusStats {
    db.fulltext_field("Note", "title").unwrap().stats.snapshot()
}

/// The derived oracle: rows and stats equal what the live objects imply.
fn assert_index_matches_objects(db: &Database) {
    let (postings, docs, tokens) = derived_expectation(db);
    assert_eq!(raw_rows(db), (postings, docs), "orphan or missing index rows");
    assert_eq!(
        stats(db),
        CorpusStats {
            doc_count: docs as u64,
            total_tokens: tokens
        }
    );
}

/// SIGKILL-faithful teardown: forget un-flushed WAL bytes, drop the handle
/// (releases the data-dir lock; the builder is off, so nothing else runs).
fn tear_down(db: Arc<Database>) {
    db.storage().discard_for_crash_recovery();
    drop(db);
}

/// The write workload: 40 creates, then updates (incl. to Null and back),
/// then deletes — every step touches the index inside its own transaction.
fn workload(db: &Database) {
    let mut ids = Vec::new();
    for i in 1..=40u64 {
        ids.push(db.create("Note", fields(Some(&text(i)), i as i64)).unwrap().id);
    }
    for (k, id) in ids.iter().enumerate() {
        match k % 4 {
            0 => {
                db.update("Note", *id, fields(Some(&format!("rewritten {k} common")), 0)).unwrap();
            }
            1 => {
                db.update("Note", *id, fields(Some(""), 0)).unwrap(); // zero-length doc
            }
            2 => {
                let mut f = FieldMap::new();
                f.insert("title".into(), Value::Null);
                db.update("Note", *id, f).unwrap();
            }
            _ => {}
        }
    }
    for id in ids.iter().step_by(5) {
        db.delete("Note", *id).unwrap();
    }
}

fn run_write_crash_case(site: Site, after_hits: u64) {
    let dir = TempDir::new().unwrap();
    {
        let db = open(dir.path(), WITH_INDEX);
        let outcome = crash_inject::catch_crash(|| {
            crash_inject::arm(site, after_hits, Mode::Crash);
            workload(&db);
        });
        crash_inject::disarm();
        assert!(
            matches!(outcome, Caught::Crashed(s) if s == site),
            "{site:?} after {after_hits} must crash, got {outcome:?}"
        );
        tear_down(db);
    }
    // Cold reopen: whatever prefix survived, index == objects.
    let survived = {
        let db = open(dir.path(), WITH_INDEX);
        assert_eq!(db.fulltext_field("Note", "title").unwrap().progress.state(), BuildState::Built);
        assert_index_matches_objects(&db);
        let r = raw_rows(&db);
        // The database keeps working (a fresh write + search).
        let id = db.create("Note", fields(Some("after recovery zzz"), 1)).unwrap().id;
        assert_eq!(
            db.fulltext_search("Note", "title", "zzz", 5, None, None)
                .unwrap()
                .hits
                .iter()
                .map(|h| h.object_id)
                .collect::<Vec<_>>(),
            vec![id]
        );
        assert_index_matches_objects(&db);
        drop(db);
        r
    };
    // Idempotence: a second reopen sees the same index (plus the one write).
    let db = open(dir.path(), WITH_INDEX);
    assert_index_matches_objects(&db);
    assert!(raw_rows(&db).1 == survived.1 + 1);
}

#[test]
fn fuzz_write_path_wal_boundaries_keep_index_and_objects_in_step() {
    // Each site × several hit counts, so the crash lands in creates, updates
    // (incl. Null flips) and deletes alike.
    for site in [
        Site::WalAfterWriteBeforeFlush,
        Site::WalAfterFlushBeforeFsync,
        Site::WalAfterFsync,
    ] {
        for after_hits in [1u64, 7, 23, 41, 55, 70] {
            run_write_crash_case(site, after_hits);
        }
    }
}

/// Ground truth: the same rows into a database that had the directive from
/// the start (builder off; the type is empty at open → Built immediately).
fn clean_build(rows: &[(u64, Option<String>)]) -> (TempDir, Arc<Database>) {
    let dir = TempDir::new().unwrap();
    let db = open(dir.path(), WITH_INDEX);
    let batch: Vec<(u64, FieldMap)> = rows
        .iter()
        .map(|(id, t)| (*id, fields(t.as_deref(), *id as i64)))
        .collect();
    db.restore_objects("Note", batch, true).unwrap();
    (dir, db)
}

fn scored(db: &Database, q: &str) -> Vec<(u64, f32)> {
    db.fulltext_search("Note", "title", q, 10_000, None, None)
        .unwrap()
        .hits
        .into_iter()
        .map(|h| (h.object_id, h.score))
        .collect()
}

fn run_build_crash_case(site: Site, after_hits: u64) {
    let rows: Vec<(u64, Option<String>)> = (1..=1100u64)
        .map(|i| (i, if i % 9 == 0 { None } else { Some(text(i)) }))
        .collect();
    let dir = TempDir::new().unwrap();
    {
        let db = open(dir.path(), WITHOUT_INDEX);
        let batch: Vec<(u64, FieldMap)> = rows
            .iter()
            .map(|(id, t)| (*id, fields(t.as_deref(), *id as i64)))
            .collect();
        db.restore_objects("Note", batch, true).unwrap();
    }
    // Add the directive; drive the backfill inline and crash mid-way.
    {
        let db = open(dir.path(), WITH_INDEX);
        assert_eq!(db.fulltext_field("Note", "title").unwrap().progress.state(), BuildState::Building);
        let outcome = crash_inject::catch_crash(|| {
            crash_inject::arm(site, after_hits, Mode::Crash);
            db.run_fulltext_tasks_inline();
        });
        crash_inject::disarm();
        assert!(
            matches!(outcome, Caught::Crashed(s) if s == site),
            "{site:?} after {after_hits} must crash, got {outcome:?}"
        );
        tear_down(db);
    }
    // Cold reopen → resume → finish; compare with a clean build. (A crash
    // right AFTER the final chunk's commit already left the marker Built.)
    let db = open(dir.path(), WITH_INDEX);
    let state = db.fulltext_field("Note", "title").unwrap().progress.state();
    assert!(matches!(state, BuildState::Building | BuildState::Built), "{state:?}");
    // A live write during the (resumed) build converges too.
    db.update("Note", 5, fields(Some("live edit during resume"), 5)).unwrap();
    db.run_fulltext_tasks_inline();
    assert_eq!(db.fulltext_field("Note", "title").unwrap().progress.state(), BuildState::Built);
    assert!(db.wait_for_fulltext_builds(std::time::Duration::from_millis(1)));

    let mut expected = rows.clone();
    expected[4].1 = Some("live edit during resume".into());
    let (_cdir, clean) = clean_build(&expected);
    assert_eq!(raw_rows(&db), raw_rows(&clean));
    assert_eq!(stats(&db), stats(&clean));
    for q in ["invoice", "+draft +w3", "\"invoice 700\"", "resume", "w12 common"] {
        assert_eq!(scored(&db, q), scored(&clean, q), "{q}");
    }
    assert_index_matches_objects(&db);
    // The resumed build never re-indexed an object twice: no object id
    // appears twice under any term prefix (would show as extra rows above),
    // and every object with a String title has exactly one `l:` row.
    let live_titled: HashSet<u64> = {
        let snap = db.storage().read_snapshot();
        let mut out = HashSet::new();
        let mut cursor = 0;
        loop {
            let chunk = db.scan_chunk("Note", snap, cursor, 500).unwrap();
            for o in &chunk.objects {
                if matches!(o.fields.get("title"), Some(Value::String(_))) {
                    out.insert(o.id);
                }
            }
            match chunk.next_cursor {
                Some(c) if chunk.more => cursor = c,
                _ => break,
            }
        }
        out
    };
    assert_eq!(raw_rows(&db).1, live_titled.len());
}

#[test]
fn fuzz_backfill_chunk_boundaries_resume_to_a_clean_build() {
    // 1100 objects = chunks of 512, 512, 76: hit 1 crashes mid-build, hit 3
    // crashes around the final chunk (before → Building; after → already Built).
    for site in [Site::FulltextBuildBeforeChunkCommit, Site::FulltextBuildAfterChunkCommit] {
        for after_hits in [1u64, 3] {
            run_build_crash_case(site, after_hits);
        }
    }
}
