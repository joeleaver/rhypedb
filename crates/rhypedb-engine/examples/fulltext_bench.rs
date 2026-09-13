//! Full-text search benchmark (issue #16 acceptance: 100k short documents,
//! single-term query p50 under 5 ms).
//!
//! Builds a deterministic corpus of `N` short documents (Zipf-distributed
//! vocabulary, ~12 words each — the shape of note titles / short bodies),
//! ingests them through the normal write path (the index is maintained in
//! each object's transaction), then measures `Database::fulltext_search`
//! latency for: a rare term, a mid-frequency term, the most common term,
//! a two-term OR, a required-term pair, and a phrase. Also reports index row
//! counts and the ingest rate.
//!
//! Run with: `cargo run --release -p rhypedb-engine --example fulltext_bench [-- N]`
//! (release matters — debug tokenization/decoding is ~10× slower).

use std::time::{Duration, Instant};

use rhypedb_engine::database::{Database, OpenOptions};
use rhypedb_engine::object::{FieldMap, Value};
use rhypedb_schema::parser::parse_schema;
use rhypedb_storage::key::KeyBuilder;

const VOCAB: usize = 20_000;
const WORDS_PER_DOC: usize = 12;
const QUERY_ITERS: usize = 200;

/// splitmix64
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Word `i` of the vocabulary (`w0`, `w1`, … — analyzed as-is).
fn word(i: usize) -> String {
    format!("w{i}")
}

/// Zipf(s=1.0) rank sampler over VOCAB via inverse CDF on a precomputed table.
struct Zipf {
    cdf: Vec<f64>,
}
impl Zipf {
    fn new(n: usize) -> Self {
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for i in 1..=n {
            acc += 1.0 / i as f64;
            cdf.push(acc);
        }
        for v in &mut cdf {
            *v /= acc;
        }
        Self { cdf }
    }
    fn sample(&self, u: f64) -> usize {
        self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1)
    }
}

fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort_unstable();
    let idx = ((samples.len() as f64 - 1.0) * p).round() as usize;
    samples[idx]
}

fn bench_query(db: &Database, label: &str, q: &str, k: usize) -> Duration {
    // Warm once (page cache / allocator), then time.
    let first = db.fulltext_search("Doc", "body", q, k, None, None).unwrap();
    let mut samples = Vec::with_capacity(QUERY_ITERS);
    let mut hits = 0;
    for _ in 0..QUERY_ITERS {
        let t = Instant::now();
        let r = db.fulltext_search("Doc", "body", q, k, None, None).unwrap();
        samples.push(t.elapsed());
        hits = r.hits.len();
    }
    let p50 = percentile(&mut samples, 0.50);
    let p99 = percentile(&mut samples, 0.99);
    println!(
        "  {label:<28} {q:<24} postings={:<7} hits={:<4} p50={:>8.3} ms  p99={:>8.3} ms",
        first.postings_scanned,
        hits,
        p50.as_secs_f64() * 1e3,
        p99.as_secs_f64() * 1e3,
    );
    p50
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    println!("== rhypedb full-text benchmark: {n} docs × ~{WORDS_PER_DOC} words, vocab {VOCAB} (Zipf) ==");

    let dir = tempfile::tempdir().unwrap();
    let schema = parse_schema("type Doc { body: String @fulltext  n: u32 }").unwrap();
    let db = Database::open_with_options(
        schema,
        dir.path(),
        OpenOptions {
            background_cover_refresh: false,
            ..Default::default()
        },
    )
    .unwrap();

    // ---- Corpus + ingest (batches of 500 through create_batch) ----
    let zipf = Zipf::new(VOCAB);
    let mut rng = Rng(0x5EED_F00D);
    let mut docs: Vec<String> = Vec::with_capacity(n);
    for _ in 0..n {
        let words: Vec<String> = (0..WORDS_PER_DOC).map(|_| word(zipf.sample(rng.f64()))).collect();
        docs.push(words.join(" "));
    }
    // Plant one phrase + one rare word for the phrase / rare-term queries.
    docs[n / 2] = format!("{} distributed consensus protocol", docs[n / 2]);
    docs[n / 3] = format!("{} distributed consensus", docs[n / 3]);
    docs[n / 4] = format!("{} zebrafish", docs[n / 4]);

    let t = Instant::now();
    for chunk in docs.chunks(500) {
        let rows: Vec<FieldMap> = chunk
            .iter()
            .enumerate()
            .map(|(i, body)| {
                let mut f = FieldMap::new();
                f.insert("body".into(), Value::String(body.clone()));
                f.insert("n".into(), Value::U32(i as u32));
                f
            })
            .collect();
        db.create_batch("Doc", rows).unwrap();
    }
    let ingest = t.elapsed();
    println!(
        "ingest: {n} docs in {:.2} s ({:.0} docs/s, index maintained in-txn)",
        ingest.as_secs_f64(),
        n as f64 / ingest.as_secs_f64()
    );
    let type_id = db.type_ids()["Doc"];
    let field_id = db.field_ids()["Doc.body"];
    let snap = db.storage().read_snapshot();
    let postings = db
        .storage()
        .count_prefix_at(snap, &KeyBuilder::fulltext_field_prefix(type_id, field_id, 0))
        .unwrap();
    let stats = db.fulltext_field("Doc", "body").unwrap().stats.snapshot();
    println!(
        "index: {postings} posting rows, {} docs, avg {:.2} tokens/doc",
        stats.doc_count,
        stats.total_tokens as f64 / stats.doc_count.max(1) as f64
    );
    // Flush so the queries read a realistic SST-resident index (not just memtable).
    db.storage().flush().unwrap();

    // ---- Queries ----
    println!("queries (k=20, {QUERY_ITERS} iterations each, after flush):");
    let rare = bench_query(&db, "rare term (df≈1)", "zebrafish", 20);
    let mid = bench_query(&db, "mid term (rank 200)", &word(199), 20);
    let mid2 = bench_query(&db, "mid term (rank 50)", &word(49), 20);
    let common = bench_query(&db, "most common term (rank 0)", &word(0), 20);
    bench_query(&db, "two-term OR", &format!("{} {}", word(199), word(3000)), 20);
    bench_query(&db, "two required terms", &format!("+{} +{}", word(5), word(40)), 20);
    bench_query(&db, "phrase", "\"distributed consensus\"", 20);
    bench_query(&db, "phrase (3 words)", "\"distributed consensus protocol\"", 20);

    println!();
    let target = Duration::from_millis(5);
    for (label, p50) in [("rare", rare), ("mid-200", mid), ("mid-50", mid2), ("most-common", common)] {
        println!(
            "  single-term {label:<12} p50 {:>8.3} ms  → {}",
            p50.as_secs_f64() * 1e3,
            if p50 < target { "PASS (< 5 ms)" } else { "over 5 ms" }
        );
    }
}
