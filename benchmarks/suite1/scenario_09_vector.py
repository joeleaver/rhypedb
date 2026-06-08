"""Scenario 2.1 — Vector indexing + k-NN search (rhypedb vs Postgres+pgvector).

Both engines index the SAME precomputed base vectors (bring-your-own-embeddings)
and answer the SAME query vectors, so the comparison is apples-to-apples:

  * rhypedb — bare `Vector<D>` field, HNSW + TurboQuant. Vectors ingested via the
    binary REQ_VECTOR_BATCH op; searched via `Vec.similar(.embedding, [q], k)`.
  * pgvector — `vector(D)` column, HNSW index `vector_cosine_ops`; searched via
    `ORDER BY emb <=> q LIMIT k`.

HNSW build params are matched (m=16, ef_construction=100). The metric is cosine
on both sides (rhypedb's index is cosine; we default to an *angular*
ann-benchmarks dataset whose ground truth is computed under cosine).

Reports per engine: search latency (p50/p95/mean over the query set),
**recall@k vs the dataset's ground truth**, index-build time, and RSS.

Like Suite 1, this connects to an already-running rhypedb server — but that
server MUST start with a schema whose Vector dimension matches the dataset
(`type Vec { embedding: Vector<D> }`). The `run_vector_bench.sh` driver resolves
the dataset dimension, generates that schema, starts/stops the server, and runs
this scenario. Postgres comes from the Suite-1 `docker compose` (pgvector image).

Run via the driver: `benchmarks/run_vector_bench.sh glove-100-angular 100000`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.harness import common, vector_data
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100


def _fmt_vec(v) -> str:
    """A `[f1,f2,...]` literal for a query vector (rhypedb QL + pgvector text).
    Fixed-point (no scientific notation): rhypedb's query parser doesn't accept
    `1.2e-05`-style floats. 8 decimals preserves f32 precision for these data."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


# ----------------------------------------------------------------------------
# rhypedb
# ----------------------------------------------------------------------------

def run_rhypedb(
    ds: vector_data.VectorDataset,
    tcp_host: str,
    tcp_port: int,
    k: int,
    n_queries: int,
    insert_chunk: int,
    server_pattern: str,
    data_dir: str | None = None,
) -> common.ScenarioResult:
    """Connects to an ALREADY-RUNNING rhypedb server whose schema declares
    `type Vec { embedding: Vector<D> }` with D == ds.dim. The caller (a driver
    script) generates that schema + starts/stops the server, matching Suite 1."""
    result = common.ScenarioResult(
        scenario="2.1 vector_knn",
        implementation="rhypedb-tcp",
        iterations=1,
        metadata={
            "dataset": ds.name,
            "n_train": int(ds.train.shape[0]),
            "dim": ds.dim,
            "k": k,
            "n_queries": n_queries,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        },
    )

    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0.0

    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    try:
        train = ds.train
        n = train.shape[0]

        # 1) Create N empty Vec objects -> ids; map object_id -> train index.
        # 2) Ingest each object's vector via the binary batch op. The HNSW index
        # builds inline during ingest, so build time is the ingest wall-clock.
        id_to_idx: dict[int, int] = {}
        build_start = common.time_ns()
        for start in range(0, n, insert_chunk):
            end = min(start + insert_chunk, n)
            rows = ",".join("{}" for _ in range(end - start))
            resp = client.query(f"Vec.create_batch([{rows}])")
            if resp.get("error"):
                raise RuntimeError(f"create_batch: {resp['error']}")
            ids = [o["id"] for o in resp["objects"]]
            for off, oid in enumerate(ids):
                id_to_idx[oid] = start + off
            r = client.vector_batch("Vec", "embedding", ids, train[start:end])
            if r.get("error"):
                raise RuntimeError(f"vector_batch: {r['error']}")
        build_ns = common.time_ns() - build_start
        result.iteration_ms.append(build_ns / 1_000_000)
        result.metadata["build_ms"] = build_ns / 1_000_000
        if server_pid:
            result.rss_post_load_mb = common.rss_mb(server_pid)
            result.rss_peak_mb = result.rss_post_load_mb

        # Search + recall.
        test = ds.test[:n_queries]
        retrieved = np.full((len(test), k), -1, dtype=np.int64)
        for qi in range(len(test)):
            q = _fmt_vec(test[qi])
            op_start = common.time_ns()
            resp = client.query(f"Vec.similar(.embedding, {q}, k: {k})")
            result.operation_latency_ns.append(common.time_ns() - op_start)
            if resp.get("error"):
                raise RuntimeError(f"similar: {resp['error']}")
            for j, o in enumerate(resp.get("objects", [])[:k]):
                retrieved[qi, j] = id_to_idx.get(o["id"], -1)

        result.metadata["recall_at_k"] = vector_data.recall_at_k(
            retrieved, ds.neighbors[:n_queries], k
        )
        if server_pid:
            result.rss_steady_mb = common.rss_mb(server_pid)
    finally:
        client.close()

    # On-disk footprint (LSM: compressed v: keys + object keys). With --no-sync
    # much of the index lives in-memory (reflected in RSS); disk is supplementary.
    if data_dir:
        result.disk_mb = common.disk_usage_mb(Path(data_dir))
        result.metadata["bytes_per_vector_rss"] = (
            (result.rss_post_load_mb - result.rss_cold_mb) * 1024 * 1024 / max(1, ds.train.shape[0])
        )

    return result


# ----------------------------------------------------------------------------
# pgvector
# ----------------------------------------------------------------------------

def run_pgvector(
    ds: vector_data.VectorDataset,
    dsn: str,
    pg_container: str,
    k: int,
    n_queries: int,
    ef_search: int,
    insert_chunk: int,
) -> common.ScenarioResult:
    from benchmarks.harness.clients.postgres import find_pg_host_pid
    import psycopg

    result = common.ScenarioResult(
        scenario="2.1 vector_knn",
        implementation="pgvector",
        iterations=1,
        metadata={
            "dataset": ds.name,
            "n_train": int(ds.train.shape[0]),
            "dim": ds.dim,
            "k": k,
            "n_queries": n_queries,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": ef_search,
        },
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0.0

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        train = ds.train
        n = train.shape[0]
        dim = ds.dim
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS vec")
            cur.execute(f"CREATE TABLE vec (id int PRIMARY KEY, emb vector({dim}))")

            # Bulk-load base vectors via COPY (id = train index, 0-based).
            with cur.copy("COPY vec (id, emb) FROM STDIN") as cp:
                for i in range(n):
                    cp.write_row((i, _fmt_vec(train[i])))

            # Build the HNSW index — this is the build-time number.
            build_start = common.time_ns()
            cur.execute(
                f"CREATE INDEX ON vec USING hnsw (emb vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
            build_ns = common.time_ns() - build_start
            result.iteration_ms.append(build_ns / 1_000_000)
            result.metadata["build_ms"] = build_ns / 1_000_000
            cur.execute(f"SET hnsw.ef_search = {ef_search}")
            if pg_pid:
                result.rss_post_load_mb = common.rss_mb(pg_pid)
                result.rss_peak_mb = result.rss_post_load_mb

            # On-disk sizes — the reliable memory metric for pgvector (its HNSW
            # index + heap live in shared_buffers / OS page cache, not the
            # backend's RSS). pgvector HNSW stores the full f32 vectors INSIDE the
            # index, so pg_index_bytes is the resident-to-serve footprint; the
            # heap (pg_table_bytes) is a second f32 copy.
            cur.execute("SELECT pg_table_size('vec')")
            result.metadata["pg_table_bytes"] = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COALESCE(SUM(pg_relation_size(indexrelid)), 0) "
                "FROM pg_index WHERE indrelid = 'vec'::regclass"
            )
            result.metadata["pg_index_bytes"] = int(cur.fetchone()[0])
            cur.execute("SELECT pg_total_relation_size('vec')")
            result.metadata["pg_total_bytes"] = int(cur.fetchone()[0])
            result.metadata["bytes_per_vector_index"] = (
                result.metadata["pg_index_bytes"] / max(1, n)
            )
            result.disk_mb = result.metadata["pg_total_bytes"] / (1024 * 1024)

            test = ds.test[:n_queries]
            retrieved = np.full((len(test), k), -1, dtype=np.int64)
            for qi in range(len(test)):
                q = _fmt_vec(test[qi])
                op_start = common.time_ns()
                cur.execute(
                    "SELECT id FROM vec ORDER BY emb <=> %s::vector LIMIT %s", (q, k)
                )
                rows = cur.fetchall()
                result.operation_latency_ns.append(common.time_ns() - op_start)
                for j, (rid,) in enumerate(rows[:k]):
                    retrieved[qi, j] = rid

            result.metadata["recall_at_k"] = vector_data.recall_at_k(
                retrieved, ds.neighbors[:n_queries], k
            )
            if pg_pid:
                result.rss_steady_mb = common.rss_mb(pg_pid)
    finally:
        conn.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove-100-angular",
                        help=f"ann-benchmarks angular dataset: {sorted(vector_data.ANGULAR_DATASETS)}")
    parser.add_argument("--subset", type=int, default=100_000,
                        help="use the first N base vectors (0 = full dataset)")
    parser.add_argument("--queries", type=int, default=1000, help="number of query vectors")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=100, help="pgvector hnsw.ef_search")
    parser.add_argument("--insert-chunk", type=int, default=5000)
    parser.add_argument("--impl", default="tcp,pg", help="comma list: tcp,pg")
    parser.add_argument("--server-pattern", default="vecbench-rhypedb-data",
                        help="pgrep -f pattern to find the running rhypedb server PID (RSS sampling)")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4501)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--data-dir", default=None,
                        help="rhypedb data dir, for on-disk footprint measurement")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_09.json"))
    args = parser.parse_args()

    subset = None if args.subset == 0 else args.subset
    ds = vector_data.load(args.dataset, subset=subset, gt_k=max(100, args.k))
    print(ds.summary())
    n_queries = min(args.queries, ds.test.shape[0])
    impls = [s.strip() for s in args.impl.split(",") if s.strip()]
    print(f"impls={impls} k={args.k} queries={n_queries}\n")

    results = []
    if "tcp" in impls:
        print("Running rhypedb...")
        t0 = time.time()
        results.append(run_rhypedb(ds, args.tcp_host, args.tcp_port,
                                   args.k, n_queries, args.insert_chunk, args.server_pattern,
                                   data_dir=args.data_dir))
        print(f"  rhypedb done in {time.time() - t0:.1f}s")
    if "pg" in impls:
        print("Running pgvector...")
        t0 = time.time()
        results.append(run_pgvector(ds, args.pg_dsn, args.pg_container, args.k,
                                    n_queries, args.ef_search, args.insert_chunk))
        print(f"  pgvector done in {time.time() - t0:.1f}s")

    print()
    common.print_summary(results)
    print("\nVector metrics:")
    print(f"  {'impl':<14} {'recall@k':>10} {'build_ms':>12} {'search p50µs':>14} {'search mean µs':>16}")
    for r in results:
        op = r.operation_stats()
        recall = r.metadata.get("recall_at_k", 0.0)
        build = r.metadata.get("build_ms", 0.0)
        print(f"  {r.implementation:<14} {recall:>10.4f} {build:>12.1f} "
              f"{op.get('p50_us', 0):>14.2f} {op.get('mean_us', 0):>16.2f}")

    print("\nMemory (the comparison that counts):")
    n_train = ds.train.shape[0]
    print(f"  {'impl':<14} {'RSS post-load':>14} {'RSS delta':>11} {'disk/idx MB':>12} {'bytes/vec':>11}")
    for r in results:
        delta = r.rss_post_load_mb - r.rss_cold_mb
        if r.implementation == "pgvector":
            idx_mb = r.metadata.get("pg_index_bytes", 0) / (1024 * 1024)
            bpv = r.metadata.get("bytes_per_vector_index", 0.0)
            note = f"idx={idx_mb:.0f} (heap={r.metadata.get('pg_table_bytes',0)/(1024*1024):.0f})"
        else:
            idx_mb = r.disk_mb
            bpv = r.metadata.get("bytes_per_vector_rss", 0.0)
            note = f"disk={idx_mb:.0f}"
        print(f"  {r.implementation:<14} {r.rss_post_load_mb:>12.1f}MB {delta:>9.1f}MB "
              f"{note:>12} {bpv:>9.1f}B")
    print(f"  (n_train={n_train}; rhypedb bytes/vec = RSS delta/n; "
          f"pgvector bytes/vec = HNSW index size/n, +f32 heap copy)")

    common.write_results(results, args.out)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
