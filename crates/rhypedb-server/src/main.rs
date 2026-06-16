// Swap the global allocator. mimalloc consistently beats glibc malloc on
// small-object workloads (~20-30% faster on 24-32 byte allocs from the
// cascade hot path) without giving up determinism. `secure` feature is
// off — we're not on a hostile-input boundary inside the process.
//
// The allocator MUST live in the binary crate (not the library), so it stays
// here while the rest of the server lives in `lib.rs` — that lets the server
// also be embedded by another binary (e.g. an app built via a buildpack).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() {
    tune_mimalloc();
    rhypedb_server::run().await;
}

/// Tell mimalloc to return freed memory to the OS promptly.
///
/// By default mimalloc keeps freed memory in its per-thread heaps. On a vector
/// workload the freed HNSW build scratch (transient `Vec<f32>` during the
/// parallel index build) plus per-node allocation fragmentation never come back:
/// a 30k×384 load inflates steady serving RssAnon to ~3220 B/vec — 3.7× the
/// ~863 B/vec logical index. `purge_delay = 0` makes each thread purge its freed
/// pages immediately, cutting steady RssAnon to ~930 B/vec (≈ the index) at 30k
/// and 1958→675 B/vec at 100k, with no measurable cost: k-NN latency unchanged
/// (within run-to-run noise), and a write-heavy run (150k single-row `@unique`
/// inserts) was within ≲2% AND used ~10% less RSS. Small-object churn reuses
/// pages so there is little to decommit; the cost lands only on large transient
/// allocations like the build scratch — exactly what we want returned.
///
/// `purge_delay` is option index 15 in BOTH vendored mimalloc v2 and v3: the
/// enum keeps `deprecated_*` placeholder slots precisely to preserve option
/// numbering across versions, so the index is stable by design (the v2/v3 enums
/// both have 15 entries before it). The Rust binding doesn't expose a named
/// constant for it, hence the literal; the `debug_assert` read-back catches any
/// future enum drift. Setting it in-process (vs the `MIMALLOC_PURGE_DELAY` env
/// var) lets the tuning ship in the binary — jkbase wraps it and shouldn't have
/// to know the knob.
fn tune_mimalloc() {
    // mi_option_purge_delay; see the doc comment for why this literal is stable.
    const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
    // SAFETY: a plain FFI write of a global allocator option; no preconditions.
    unsafe { libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, 0) };
    debug_assert_eq!(
        unsafe { libmimalloc_sys::mi_option_get(MI_OPTION_PURGE_DELAY) },
        0,
        "mi_option_set(purge_delay) did not take — mimalloc option enum may have drifted"
    );
}
