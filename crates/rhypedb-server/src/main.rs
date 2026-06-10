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
    rhypedb_server::run().await;
}
