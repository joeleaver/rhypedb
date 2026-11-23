//! The in-memory, thread-free configuration (`Database::open_in_memory`): the
//! one a derived index in a browser Web Worker (`wasm32-unknown-unknown`) uses.
//! The same body runs natively here (`cargo test`) and in Node under
//! `wasm-bindgen-test` (`in_memory_wasm` below):
//! `cargo test --target wasm32-unknown-unknown -p rhypedb-engine --no-default-features --test in_memory`
//! with `wasm-bindgen-test-runner` as the target's runner.
//!
//! It writes enough to flush the memtable into in-memory SSTs and to trigger
//! inline compaction several times, updates and deletes rows (so version GC
//! and tombstones run through the merge), and then checks keyword search
//! (the `english` analyzer, a trailing prefix term) and vector search over
//! plain vectors (`Vectorizer::ingest_vectors` / `search_vector`).

use std::sync::Arc;

use rhypedb_engine::database::{Database, OpenOptions};
use rhypedb_engine::object::{FieldMap, Value};
use rhypedb_engine::vectorizer::{Vectorizer, VectorizerConfig};
use rhypedb_schema::parser::parse_schema;

const SCHEMA: &str = r#"
type Node {
    node_id: String @unique
    title: String @fulltext(analyzer: "english")
    text: String @fulltext(analyzer: "english")
    parent: Node @on_delete(remove)
    children: [Node] @inverse(Node.parent)
    chunks: [Chunk] @inverse(Chunk.node)
}

type Chunk {
    chunk_id: String @unique
    node: Node @on_delete(cascade)
    embedding: Vector<32> @index(hnsw, metric: cosine)
}
"#;

const WORDS: &[&str] = &[
    "garden", "kitchen", "invoice", "holiday", "bicycle", "recipe", "meeting", "doctor",
    "insurance", "library", "violin", "mountain", "harbor", "lantern", "orchard", "pebble",
];

/// A deterministic pseudo-random vector per row (an LCG, so the native and
/// wasm runs see the same data).
fn vector_for(i: usize) -> Vec<f32> {
    let mut state = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..32)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn node_fields(i: usize, text: &str) -> FieldMap {
    let mut f = FieldMap::new();
    f.insert("node_id".into(), Value::String(format!("n{i}")));
    f.insert("title".into(), Value::String(format!("Note {i} about {}", WORDS[i % WORDS.len()])));
    f.insert("text".into(), Value::String(text.to_string()));
    f
}

fn body(i: usize) -> String {
    // Long enough that a few thousand rows cross the 4 MB memtable several
    // times over (postings, covers and the object rows all count).
    let w = WORDS[i % WORDS.len()];
    format!(
        "The {w} entry number {i}. Gardeners were gardening in the gardens while \
         the {w} waited. {}",
        "Filler sentence about nothing in particular, repeated for size. ".repeat(12)
    )
}

pub fn exercise() {
    let schema = parse_schema(SCHEMA).expect("schema parses");
    let db = Database::open_in_memory(schema.clone(), OpenOptions::default()).expect("opens in memory");
    assert!(db.storage().is_in_memory());

    let vectorizer = Vectorizer::with_config(
        Arc::clone(db.storage()),
        schema,
        db.type_ids().clone(),
        db.field_ids().clone(),
        VectorizerConfig::default(),
    )
    .expect("vectorizer over in-memory storage");

    const N: usize = 3000;
    let mut node_ids = Vec::with_capacity(N);
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let node = db.create("Node", node_fields(i, &body(i))).expect("create node");
        let mut cf = FieldMap::new();
        cf.insert("chunk_id".into(), Value::String(format!("c{i}")));
        let chunk = db.create("Chunk", cf).expect("create chunk");
        db.link("Chunk", chunk.id, "node", node.id, None).expect("link chunk");
        node_ids.push(node.id);
        rows.push((chunk.id, vector_for(i)));
        if rows.len() == 256 {
            vectorizer.ingest_vectors("Chunk", "embedding", &rows).expect("ingest");
            rows.clear();
        }
    }
    vectorizer.ingest_vectors("Chunk", "embedding", &rows).expect("ingest");

    // Rewrite every third node (old versions to GC) and delete every tenth.
    for i in (0..N).step_by(3) {
        let mut f = FieldMap::new();
        f.insert("text".into(), Value::String(format!("Rewritten: a pebble in the orchard, {i}.")));
        db.update("Node", node_ids[i], f).expect("update");
    }
    for i in (0..N).step_by(10) {
        db.delete("Node", node_ids[i]).expect("delete");
    }
    db.run_deferred_work();
    // An explicit full compaction must also work without files.
    db.storage().flush().expect("flush");
    db.storage().compact().expect("compact");
    assert!(db.storage().sst_count() >= 1, "data reached in-memory SSTs");

    // Keyword: stemming ("gardening" finds "garden"), a trailing prefix term,
    // deleted rows gone, rewritten rows found by their new text only.
    let hits = db.fulltext_search("Node", "text", "gardening", 10, None, None).expect("search").hits;
    assert_eq!(hits.len(), 10);
    let deleted: std::collections::HashSet<u64> = (0..N).step_by(10).map(|i| node_ids[i]).collect();
    assert!(hits.iter().all(|h| !deleted.contains(&h.object_id)));

    let prefix = db.fulltext_search("Node", "title", "violi*", 1000, None, None).expect("prefix").hits;
    let expected = (0..N).filter(|i| WORDS[i % WORDS.len()] == "violin" && i % 10 != 0).count();
    assert_eq!(prefix.len(), expected, "prefix term matches every live violin note");

    let rewritten = db.fulltext_search("Node", "text", "pebble orchard", 2000, None, None).expect("search").hits;
    let expected_rewritten = (0..N).filter(|i| i % 3 == 0 && i % 10 != 0).count();
    // Rewritten rows match both words; untouched pebble/orchard rows match one.
    assert!(rewritten.len() >= expected_rewritten);
    let top = db.get("Node", rewritten[0].object_id).expect("top hit");
    let Some(Value::String(text)) = top.fields.get("text") else { panic!("text field") };
    assert!(text.starts_with("Rewritten"), "{text}");

    // Vector: a stored row's own vector finds that row among the nearest
    // (the index is approximate: quantized, HNSW), for nearly every row asked.
    let mut found_self = 0;
    let probes: Vec<usize> = (1..N).step_by(97).filter(|i| i % 10 != 0).collect();
    for &i in &probes {
        let hits = vectorizer
            .search_vector("Chunk", "embedding", &vector_for(i), 5, 64, false, None)
            .expect("vector search");
        assert!(!hits.is_empty());
        let ids: Vec<String> = hits
            .iter()
            // (A deleted row's chunk may still sit in the graph; it reads as gone.)
            .filter_map(|h| match db.get("Chunk", h.object_id).ok()?.fields.get("chunk_id") {
                Some(Value::String(c)) => Some(c.clone()),
                _ => None,
            })
            .collect();
        if ids.contains(&format!("c{i}")) {
            found_self += 1;
        }
    }
    assert!(found_self * 10 >= probes.len() * 9, "recall {found_self}/{}", probes.len());

    // Nothing was left on disk: an in-memory tree refuses a snapshot.
    assert!(db.storage().snapshot_to(std::path::Path::new("unused")).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn in_memory_ingest_and_search() {
    exercise();
    // Nothing was created in the working directory either.
    assert!(!std::path::Path::new("sst").exists());
    assert!(!std::path::Path::new("wal.log").exists());
}

#[cfg(target_arch = "wasm32")]
mod in_memory_wasm {
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn in_memory_ingest_and_search() {
        super::exercise();
    }
}
