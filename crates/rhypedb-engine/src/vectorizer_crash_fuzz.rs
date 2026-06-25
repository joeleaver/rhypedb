//! Crash-recovery fuzz harness for the vectorize queue + HNSW rebuild (Inc 4 of
//! the crash-fuzz epic, Overboard cmqshgpnx).
//!
//! Mirrors the storage harness (`rhypedb-storage/tests/crash_fuzz.rs`): a seeded,
//! deterministic workload is driven under [`crash_inject::catch_crash`] with one
//! of the four `Vectorize*` sites armed to fire after N hits, then the data dir
//! is faithfully torn down and cold-reopened, and a shadow-model oracle asserts
//! the recovered state is exactly correct.
//!
//! # Why in-process, single-threaded
//!
//! The vectorizer normally runs a background worker thread, but crash arming is
//! thread-local, so the harness NEVER starts the worker — it drives the pipeline
//! synchronously on the test thread via [`Vectorizer::process_pending`]. The four
//! sites bracket the two durability hand-offs the pipeline makes:
//!
//! * `claim_batch` deletes the queue entries and commits BEFORE any vector is
//!   written (`VectorizeAfterClaimCommit`) — the orphan window that
//!   `reconcile_pending_jobs` exists to close.
//! * `store_and_index` inserts into the in-RAM HNSW, then commits the `v:` vector
//!   + `Indexed` state (`VectorizeBeforeStoreCommit` / `VectorizeAfterStoreCommit`).
//! * the cold-reopen HNSW rebuild scans the durable `v:` keyspace
//!   (`VectorizeRebuildMidScan`).
//!
//! # Deterministic, model-free embedder
//!
//! [`DeterministicEmbedder`] maps the source text `"doc {i}"` to a distinct
//! near-one-hot unit vector `e_i` (1.0 at dimension `i-1`, a small constant
//! elsewhere). Distinct + near-orthogonal means each object is unambiguously its
//! own nearest neighbor, so the oracle can assert HNSW searchability exactly —
//! and no ONNX model is loaded, so the harness has no network dependency.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rhypedb_embed::EmbedResult;
use rhypedb_schema::parser::parse_schema;
use rhypedb_storage::lsm::LsmConfig;

// Brings the parent `vectorizer` module's scope into view: `Vectorizer`,
// `VectorizeJob`, `VectorState`, `Embedder`, `Schema`, `KeyBuilder`, `LsmTree`,
// the module-private `deserialize_f32_vec`, the `crash_inject` import, etc.
use super::*;

/// Number of `@vectorize` objects in each workload.
const N: u64 = 8;
/// Vector dimension. Strictly greater than `N + 1` so every object (including the
/// extra "queue resumes" object) gets a distinct near-one-hot vector.
const DIM: usize = 16;
/// `@vectorize` model name; the [`DeterministicEmbedder`] is injected under it.
const MODEL: &str = "deterministic";
const TYPE_ID: u64 = 1;
const EMBEDDING_FIELD_ID: u64 = 2;

/// Deterministic, model-free embedder. `"doc {i}"` -> the unit vector `e_i`.
struct DeterministicEmbedder;

impl Embedder for DeterministicEmbedder {
    fn embed(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| embed_text(t)).collect())
    }
    fn dimensions(&self) -> usize {
        DIM
    }
    fn model_name(&self) -> &str {
        MODEL
    }
}

/// The deterministic ground-truth vector for body text `"doc {i}"`: a near-one-hot
/// `e_i` whose hot dimension is `i - 1`. Distinct for every `i` in `1..=DIM`.
fn embed_text(text: &str) -> Vec<f32> {
    let idx: usize = text
        .rsplit(' ')
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .expect("deterministic embedder requires body text of the form `doc <n>`");
    assert!(
        (1..=DIM).contains(&idx),
        "object index {idx} out of vector range"
    );
    let hot = idx - 1;
    let mut v = vec![0.01f32; DIM];
    v[hot] = 1.0;
    v
}

fn doc_body(oid: u64) -> String {
    format!("doc {oid}")
}

/// Ground-truth vector for object `oid` (must match what the embedder produced).
fn gt(oid: u64) -> Vec<f32> {
    embed_text(&doc_body(oid))
}

fn schema_and_ids() -> (Schema, HashMap<String, u64>, HashMap<String, u64>) {
    let schema = parse_schema(
        r#"
        type Doc {
            body: String
            embedding: Vector<16> @vectorize(source: "body", model: "deterministic")
        }
        "#,
    )
    .unwrap();
    let mut type_ids = HashMap::new();
    type_ids.insert("Doc".into(), TYPE_ID);
    let mut field_ids = HashMap::new();
    field_ids.insert("Doc.body".into(), 1u64);
    field_ids.insert("Doc.embedding".into(), EMBEDDING_FIELD_ID);
    (schema, type_ids, field_ids)
}

fn open_storage(path: &Path) -> Arc<LsmTree> {
    LsmTree::open(LsmConfig::new(path)).unwrap()
}

/// Remove any persisted HNSW snapshot (`hnsw_*.bin`) from the data dir.
///
/// A real crash (SIGKILL) never runs `Vectorizer::Drop`, so it never writes a
/// fresh snapshot — but the harness drops its `Vectorizer` normally, and that Drop
/// calls `save_snapshots()`. Left in place, a complete snapshot would be reloaded
/// by `rebuild_indexes` on the next open and shadow the `v:`-keyspace rebuild this
/// harness exists to exercise (the search oracle would then validate the snapshot,
/// not a reconstruction). Clearing it before every cold reopen makes the reopen
/// faithfully reconstruct the HNSW from the durable `v:` vectors — exactly what a
/// crashed node with no saved snapshot does.
fn clear_hnsw_snapshots(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with("hnsw_") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Cold-reopen the data dir the way a crashed node would: clear any stale HNSW
/// snapshot (forcing a `v:`-derived rebuild), open the LSM, then open + inject a
/// fresh `Vectorizer` (which runs `rebuild_indexes` + `reconcile_pending_jobs`).
fn cold_reopen(
    dir: &Path,
    schema: &Schema,
    type_ids: &HashMap<String, u64>,
    field_ids: &HashMap<String, u64>,
) -> (Arc<LsmTree>, Vectorizer) {
    clear_hnsw_snapshots(dir);
    let storage = open_storage(dir);
    let vz = open_vectorizer(Arc::clone(&storage), schema, type_ids, field_ids);
    (storage, vz)
}

/// Cold-open a `Vectorizer` (rebuild + reconcile) and inject the deterministic
/// embedder. The embedder MUST be re-injected after every reopen — without it
/// `process_batch` fails on a missing model and the fixpoint never converges.
fn open_vectorizer(
    storage: Arc<LsmTree>,
    schema: &Schema,
    type_ids: &HashMap<String, u64>,
    field_ids: &HashMap<String, u64>,
) -> Vectorizer {
    let vz = Vectorizer::new(storage, schema.clone(), type_ids.clone(), field_ids.clone()).unwrap();
    vz.embedders
        .lock()
        .insert(MODEL.into(), Box::new(DeterministicEmbedder));
    vz
}

/// Write an object row carrying the source `body` field, the way `Database`
/// would, so `process_batch` can read the text to embed.
fn store_object(storage: &LsmTree, type_id: u64, oid: u64, body: &str) {
    let mut fields = crate::object::FieldMap::new();
    fields.insert("body".into(), Value::String(body.into()));
    let serialized = crate::object::serialize_fields(&fields);
    let key = KeyBuilder::object(type_id, oid);
    let mut txn = storage.begin_txn();
    storage.put(&mut txn, &key, serialized).unwrap();
    storage.commit(&mut txn).unwrap();
}

fn job(oid: u64) -> VectorizeJob {
    VectorizeJob {
        type_name: "Doc".into(),
        object_id: oid,
        source_field: "body".into(),
        vector_field: "embedding".into(),
        model: MODEL.into(),
    }
}

/// Read the durable full-precision `v:` vector for an object, or `None` if absent.
fn read_stored_vector(
    storage: &LsmTree,
    type_id: u64,
    oid: u64,
    field_id: u64,
) -> Option<Vec<f32>> {
    let key = KeyBuilder::vector(type_id, oid, field_id);
    let snap = storage.read_snapshot();
    let data = storage.get_at(snap, &key).ok()??;
    deserialize_f32_vec(&data)
}

/// Drive `process_pending` to a fixpoint (returns 0), asserting the recovery
/// reprocesses a bounded number of times — the harness's convergence guarantee.
fn drive_to_fixpoint(vz: &Vectorizer) {
    let mut passes = 0u64;
    loop {
        let processed = vz.process_pending().unwrap();
        if processed == 0 {
            return;
        }
        passes += 1;
        assert!(
            passes <= N + 2,
            "recovery must converge within {} passes (still processing after {passes})",
            N + 2,
        );
    }
}

/// The full oracle: every object ends `Indexed`, its `v:` vector is bit-exact,
/// it is its own nearest neighbor in the HNSW, and the queue is fully drained.
fn assert_fully_recovered(vz: &Vectorizer, storage: &LsmTree) {
    for oid in 1..=N {
        assert_indexed_and_searchable(vz, storage, oid);
    }
    assert_eq!(
        vz.status().pending,
        0,
        "queue must be fully drained after recovery"
    );
}

fn assert_indexed_and_searchable(vz: &Vectorizer, storage: &LsmTree, oid: u64) {
    assert_eq!(
        vz.get_state("Doc", oid, "embedding").unwrap(),
        VectorState::Indexed,
        "object {oid} must end Indexed",
    );
    let stored = read_stored_vector(storage, TYPE_ID, oid, EMBEDDING_FIELD_ID)
        .unwrap_or_else(|| panic!("object {oid} must have a durable v: vector"));
    assert_eq!(stored, gt(oid), "object {oid} v: vector must be bit-exact");
    let hits = vz
        .search_vector("Doc", "embedding", &gt(oid), 1, 64, false, None)
        .unwrap();
    assert_eq!(
        hits.first().map(|(id, _)| *id),
        Some(oid),
        "object {oid} must be its own nearest neighbor (HNSW searchable)",
    );
}

/// Phase 1 shared by the three "crash during `process_pending`" sites: store +
/// enqueue all N objects, then crash with `site` armed to fire after `after_hits`
/// hits. Returns once the (faithfully torn-down) data dir has been cold-reopened
/// and recovery has converged, with the full oracle + idempotence + queue-resume
/// all asserted.
fn run_process_crash_case(dir: &Path, site: crash_inject::Site, after_hits: u64) {
    let (schema, type_ids, field_ids) = schema_and_ids();

    // --- Phase 1: build the workload, then crash mid-processing. ---
    {
        let storage = open_storage(dir);
        let vz = open_vectorizer(Arc::clone(&storage), &schema, &type_ids, &field_ids);
        for oid in 1..=N {
            store_object(&storage, TYPE_ID, oid, &doc_body(oid));
            vz.enqueue(job(oid)).unwrap();
        }

        let outcome = crash_inject::catch_crash(|| {
            crash_inject::arm(site, after_hits, crash_inject::Mode::Crash);
            let _ = vz.process_pending();
        });
        // Disarm BEFORE any further work: the thread-local arm survives the
        // unwind, and an un-caught re-fire during recovery would abort the test.
        crash_inject::disarm();
        assert!(
            matches!(outcome, crash_inject::Caught::Crashed(s) if s == site),
            "site {site:?} after {after_hits} hit(s) must crash, got {outcome:?}",
        );

        // Teardown that models a SIGKILL. `discard_for_crash_recovery` forgets the
        // WAL's un-flushed buffer — a defensive no-op at the Vectorize* sites (the
        // last commit already fsynced, so the buffer is empty), kept uniform with
        // the storage harness. What actually matters here: dropping `vz` discards
        // the in-RAM HNSW a real crash would lose (its Drop persists a snapshot,
        // which `cold_reopen` then clears), and dropping both handles releases the
        // data-dir lock before the reopen.
        storage.discard_for_crash_recovery();
        drop(vz);
        drop(storage);
    }

    // --- Phase 2: cold reopen -> reconcile + `v:` rebuild; drive to fixpoint. ---
    {
        let (storage, vz) = cold_reopen(dir, &schema, &type_ids, &field_ids);
        drive_to_fixpoint(&vz);
        assert_fully_recovered(&vz, &storage);
    }

    // --- Phase 3: idempotence — a 2nd cold reopen with NO processing matches. ---
    {
        let (storage, vz) = cold_reopen(dir, &schema, &type_ids, &field_ids);
        assert_eq!(
            vz.status().pending,
            0,
            "reconcile must not re-enqueue already-Indexed objects",
        );
        assert_fully_recovered(&vz, &storage);

        // --- Phase 4: the queue still resumes — a brand-new object flows through. ---
        let extra = N + 1;
        store_object(&storage, TYPE_ID, extra, &doc_body(extra));
        vz.enqueue(job(extra)).unwrap();
        drive_to_fixpoint(&vz);
        assert_indexed_and_searchable(&vz, &storage, extra);
    }
}

#[test]
fn fuzz_claim_commit_orphans_whole_batch_then_recovers() {
    // `claim_batch` deletes + commits the WHOLE batch in one transaction, so this
    // site fires exactly once per `process_pending` (after_hits = 1): every job
    // is durably dequeued with no vector written. Reconcile must re-enqueue all N.
    let dir = tempfile::tempdir().unwrap();
    run_process_crash_case(dir.path(), crash_inject::Site::VectorizeAfterClaimCommit, 1);
}

#[test]
fn fuzz_before_store_commit_recovers() {
    // One hit per object: crash after the in-RAM HNSW insert but before the `v:`
    // commit for object k. Objects < k are durable, k..=N are still Pending.
    for after_hits in 1..=N {
        let dir = tempfile::tempdir().unwrap();
        run_process_crash_case(
            dir.path(),
            crash_inject::Site::VectorizeBeforeStoreCommit,
            after_hits,
        );
    }
}

#[test]
fn fuzz_after_store_commit_recovers() {
    // One hit per object: crash right after object k's `v:` + `Indexed` commit.
    // Object k is durable (reopen is a no-op for it); k+1..=N stay Pending.
    for after_hits in 1..=N {
        let dir = tempfile::tempdir().unwrap();
        run_process_crash_case(
            dir.path(),
            crash_inject::Site::VectorizeAfterStoreCommit,
            after_hits,
        );
    }
}

#[test]
fn fuzz_rebuild_mid_scan_converges() {
    // Index all N cleanly, then crash partway through the cold-reopen HNSW rebuild
    // scan (after_hits in 1..=N — one hit per `v:` entry). The rebuild writes
    // nothing, so the next reopen must reconstruct the identical, fully searchable
    // index from the durable `v:` keyspace. The snapshot Phase 1's clean drop
    // persists is cleared before each reopen, so the crash hits — and recovery
    // takes — the full `v:`-scan rebuild (`rebuild_index_from_lsm`), not the
    // snapshot-load delta path that would otherwise shadow it.
    for after_hits in 1..=N {
        let dir = tempfile::tempdir().unwrap();
        let (schema, type_ids, field_ids) = schema_and_ids();

        // Phase 1: clean index of all N (no crash) -> `v:` fully populated.
        {
            let storage = open_storage(dir.path());
            let vz = open_vectorizer(Arc::clone(&storage), &schema, &type_ids, &field_ids);
            for oid in 1..=N {
                store_object(&storage, TYPE_ID, oid, &doc_body(oid));
                vz.enqueue(job(oid)).unwrap();
            }
            assert_eq!(vz.process_pending().unwrap(), N as usize);
            assert_fully_recovered(&vz, &storage);
            drop(vz);
            drop(storage);
        }

        // Phase 2: cold reopen #1 — crash mid-rebuild-scan inside `Vectorizer::new`.
        // Clearing the snapshot first forces `rebuild_indexes` down the full
        // `v:`-scan path, where the site fires (and where a real no-snapshot crash
        // would land).
        {
            clear_hnsw_snapshots(dir.path());
            let storage = open_storage(dir.path());
            let outcome = crash_inject::catch_crash(|| {
                crash_inject::arm(
                    crash_inject::Site::VectorizeRebuildMidScan,
                    after_hits,
                    crash_inject::Mode::Crash,
                );
                let _ = Vectorizer::new(
                    Arc::clone(&storage),
                    schema.clone(),
                    type_ids.clone(),
                    field_ids.clone(),
                );
            });
            crash_inject::disarm();
            assert!(
                matches!(
                    outcome,
                    crash_inject::Caught::Crashed(crash_inject::Site::VectorizeRebuildMidScan)
                ),
                "rebuild scan after {after_hits} hit(s) must crash, got {outcome:?}",
            );
            // The rebuild only reads, so nothing durable changed (the crashed
            // partial index is empty -> its Drop writes no snapshot); release the
            // lock before the reopen.
            storage.discard_for_crash_recovery();
            drop(storage);
        }

        // Phase 3: cold reopen #2 — full `v:` rebuild converges; nothing to process.
        let (storage, vz) = cold_reopen(dir.path(), &schema, &type_ids, &field_ids);
        assert_eq!(
            vz.process_pending().unwrap(),
            0,
            "a crashed rebuild leaves no pending work (the `v:` keyspace is intact)",
        );
        assert_fully_recovered(&vz, &storage);
    }
}

#[test]
fn smoke_no_crash_indexes_all_in_one_pass() {
    // The control case: with nothing armed, the whole workload drains in a single
    // pass to a fully searchable, bit-exact index.
    let dir = tempfile::tempdir().unwrap();
    let (schema, type_ids, field_ids) = schema_and_ids();
    let storage = open_storage(dir.path());
    let vz = open_vectorizer(Arc::clone(&storage), &schema, &type_ids, &field_ids);
    for oid in 1..=N {
        store_object(&storage, TYPE_ID, oid, &doc_body(oid));
        vz.enqueue(job(oid)).unwrap();
    }
    assert_eq!(vz.process_pending().unwrap(), N as usize);
    assert_fully_recovered(&vz, &storage);
}

/// W1 (Overboard cmqsjn33n) — DEFERRED; not yet recoverable, so `#[ignore]`d.
///
/// The production enqueue is server-side and best-effort (`server/lib.rs`:
/// `let _ = vectorizer.enqueue(...)`), NOT transactional with the object commit.
/// A crash in the commit->enqueue gap leaves a committed `@vectorize` object with
/// NO vector AND NO `Pending` state marker, so `reconcile_pending_jobs` (which
/// scans the `vector_state` keyspace) cannot find it. This harness drives
/// `enqueue` directly, so it always has a durable `Pending` marker and cannot
/// reproduce W1 — but the gap is real. Recovering it needs an engine-driven
/// transactional enqueue (Option C) or a boot backfill scan (Option D), a pending
/// design decision. This test asserts the DESIRED end state (the object recovers)
/// and therefore fails until W1 is fixed; remove `#[ignore]` then.
#[test]
#[ignore = "W1 (cmqsjn33n): commit->enqueue gap leaves no Pending marker; needs Option C/D"]
fn w1_object_committed_without_enqueue_should_recover() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, type_ids, field_ids) = schema_and_ids();

    {
        let storage = open_storage(dir.path());
        let _vz = open_vectorizer(Arc::clone(&storage), &schema, &type_ids, &field_ids);
        // The object row commits, but the best-effort enqueue never ran: no queue
        // entry and (crucially) no `Pending` state marker.
        store_object(&storage, TYPE_ID, 1, &doc_body(1));
        drop(_vz);
        drop(storage);
    }

    let storage = open_storage(dir.path());
    let vz = open_vectorizer(Arc::clone(&storage), &schema, &type_ids, &field_ids);
    drive_to_fixpoint(&vz);
    // Desired (currently unmet): reconcile/backfill recovers the orphaned object.
    assert_indexed_and_searchable(&vz, &storage, 1);
}
