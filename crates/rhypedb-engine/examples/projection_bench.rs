//! Column-projection micro-benchmark.
//!
//! Measures the win from selective deserialization on the two paths the
//! projection change actually touches:
//!
//!   1. Primitive level — `deserialize_fields` (full HashMap of every Value)
//!      vs `extract_field` (one field) vs `deserialize_fields_projected`
//!      (a 2-of-N subset), over a deliberately WIDE object.
//!
//!   2. Engine level — a selective non-indexed filter scan via the rewired
//!      `filter_scan_str` (extract only the predicate field, full-deserialize
//!      only matches) vs the OLD shape it replaced (`scan_type` + post-filter,
//!      which full-deserializes every row). Both still exist on this branch,
//!      so this is a true A/B in one binary on identical data.
//!
//! Run with: `cargo run --release --example projection_bench`
//! (release matters — the win is allocation/CPU, swamped by debug overhead).

use std::time::Instant;

use rhypedb_engine::database::Database;
use rhypedb_engine::object::{
    FieldMap, Value, deserialize_fields, deserialize_fields_projected, extract_field,
    serialize_fields,
};
use rhypedb_schema::parser::parse_schema;

const N_ROWS: usize = 50_000;
const WIDE_COLS: usize = 12;
const PRIM_ITERS: usize = 2_000_000;

fn wide_schema() -> rhypedb_schema::Schema {
    // One filter field (`tag`, NON-indexed → fallback path) plus many filler
    // columns so projection has something meaningful to skip.
    let mut sdl = String::from("type Wide {\n    tag: String\n");
    for i in 0..WIDE_COLS {
        sdl.push_str(&format!("    s{i}: String\n"));
    }
    sdl.push_str("    n0: u32\n    n1: u64\n}\n");
    parse_schema(&sdl).unwrap()
}

fn wide_fields(i: usize, matches: bool) -> FieldMap {
    let mut f = FieldMap::new();
    f.insert(
        "tag".into(),
        Value::String(if matches { "match".into() } else { format!("nomatch-{i}") }),
    );
    for c in 0..WIDE_COLS {
        f.insert(format!("s{c}"), Value::String(format!("filler-value-{i}-{c}")));
    }
    f.insert("n0".into(), Value::U32(i as u32));
    f.insert("n1".into(), Value::U64((i as u64) * 1000));
    f
}

fn main() {
    println!("== rhypedb column-projection micro-benchmark ==");
    println!("wide type: {} columns; {N_ROWS} rows\n", WIDE_COLS + 3);

    // ---- 1. Primitive level ----
    let blob = serialize_fields(&wide_fields(42, false));
    println!("serialized wide object: {} bytes", blob.len());

    let t = Instant::now();
    let mut sink = 0usize;
    for _ in 0..PRIM_ITERS {
        let m = deserialize_fields(&blob);
        sink ^= m.len();
    }
    let full = t.elapsed();

    let t = Instant::now();
    for _ in 0..PRIM_ITERS {
        let v = extract_field(&blob, "tag");
        sink ^= v.is_some() as usize;
    }
    let one = t.elapsed();

    let t = Instant::now();
    for _ in 0..PRIM_ITERS {
        let m = deserialize_fields_projected(&blob, &["tag", "n0"]);
        sink ^= m.len();
    }
    let proj = t.elapsed();

    let ns = |d: std::time::Duration| d.as_nanos() as f64 / PRIM_ITERS as f64;
    println!("  deserialize_fields (full):           {:8.1} ns/op", ns(full));
    println!(
        "  extract_field (1 field):             {:8.1} ns/op  ({:.1}x faster)",
        ns(one),
        ns(full) / ns(one)
    );
    println!(
        "  deserialize_fields_projected (2/{}):  {:8.1} ns/op  ({:.1}x faster)",
        WIDE_COLS + 3,
        ns(proj),
        ns(full) / ns(proj)
    );
    println!("  (sink={sink})\n");

    // ---- 2. Engine level ----
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(wide_schema(), dir.path()).unwrap();
    // ~1% selective.
    for i in 0..N_ROWS {
        db.create("Wide", wide_fields(i, i % 100 == 0)).unwrap();
    }

    // OLD shape: scan_type (full deserialize per row) + post-filter.
    let t = Instant::now();
    let old_hits = db
        .scan_type("Wide")
        .unwrap()
        .into_iter()
        .filter(|o| matches!(o.fields.get("tag"), Some(Value::String(s)) if s == "match"))
        .count();
    let old = t.elapsed();

    // NEW path: rewired filter_scan_str → filter_scan_fallback (projected predicate).
    let t = Instant::now();
    let new_hits = db
        .filter_scan_str("Wide", "tag", rhypedb_storage::zone::CompareOp::Eq, "match", None)
        .unwrap()
        .len();
    let new = t.elapsed();

    println!("selective non-indexed filter over {N_ROWS} rows (~1% match):");
    println!("  old (scan_type + full deserialize): {:8.2} ms  ({old_hits} hits)", old.as_secs_f64() * 1e3);
    println!("  new (projected predicate scan):     {:8.2} ms  ({new_hits} hits)", new.as_secs_f64() * 1e3);
    println!("  speedup: {:.2}x", old.as_secs_f64() / new.as_secs_f64());
    assert_eq!(old_hits, new_hits, "A/B result mismatch — projection changed the answer!");
}
