"""Iso-recall latency: rhypedb (ef + full-precision rerank) vs pgvector (ef_search).

The fair question for the rerank knob isn't "what does rerank cost" in isolation
— it's "at the SAME recall@10, which engine answers a query faster?". pgvector
stores full f32 and reaches recall by raising hnsw.ef_search (no quantization
loss); rhypedb stores 4-bit codes and reaches recall by raising ef and/or
re-scoring against the LSM f32. This builds each engine ONCE on the same 30k
data + query set, sweeps both knobs, and prints (recall, mean_us) for every
config sorted by recall so you can read off iso-recall latency.

rhypedb side connects to an already-running bench server (a wrapper starts it).
pgvector side uses the Suite-1 docker container.
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmarks.harness import common, vector_data
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient
from benchmarks.suite1.scenario_09_vector import _fmt_vec, HNSW_M, HNSW_EF_CONSTRUCTION


def _mean_us(lat_ns):
    return float(np.mean(lat_ns) / 1000.0)


# ---- rhypedb ----------------------------------------------------------------

def build_rhypedb(client, ds, insert_chunk):
    train = ds.train
    n = train.shape[0]
    id_to_idx = {}
    t0 = common.time_ns()
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
    return id_to_idx, (common.time_ns() - t0) / 1e6


def eval_rhypedb(client, test, k, id_to_idx, ef, rerank):
    params = ""
    if ef:
        params += f", ef: {ef}"
    if rerank:
        params += f", rerank: {rerank}"
    retrieved = np.full((len(test), k), -1, dtype=np.int64)
    lat = []
    for qi in range(len(test)):
        q = _fmt_vec(test[qi])
        t0 = common.time_ns()
        resp = client.query(f"Vec.similar(.embedding, {q}, k: {k}{params})")
        lat.append(common.time_ns() - t0)
        if resp.get("error"):
            raise RuntimeError(f"similar: {resp['error']}")
        for j, o in enumerate(resp.get("objects", [])[:k]):
            retrieved[qi, j] = id_to_idx.get(o["id"], -1)
    return retrieved, _mean_us(lat)


# ---- pgvector ---------------------------------------------------------------

def build_pg(conn, ds):
    train = ds.train
    n = train.shape[0]
    dim = ds.dim
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS vec")
        cur.execute(f"CREATE TABLE vec (id int PRIMARY KEY, emb vector({dim}))")
        with cur.copy("COPY vec (id, emb) FROM STDIN") as cp:
            for i in range(n):
                cp.write_row((i, _fmt_vec(train[i])))
        t0 = common.time_ns()
        cur.execute(
            f"CREATE INDEX ON vec USING hnsw (emb vector_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        return (common.time_ns() - t0) / 1e6


def eval_pg(conn, test, k, ef_search):
    retrieved = np.full((len(test), k), -1, dtype=np.int64)
    lat = []
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
        for qi in range(len(test)):
            q = _fmt_vec(test[qi])
            t0 = common.time_ns()
            cur.execute("SELECT id FROM vec ORDER BY emb <=> %s::vector LIMIT %s", (q, k))
            rows = cur.fetchall()
            lat.append(common.time_ns() - t0)
            for j, (rid,) in enumerate(rows[:k]):
                retrieved[qi, j] = rid
    return retrieved, _mean_us(lat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic-384-angular")
    p.add_argument("--subset", type=int, default=30000)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--insert-chunk", type=int, default=5000)
    p.add_argument("--tcp-host", default="127.0.0.1")
    p.add_argument("--tcp-port", type=int, default=4541)
    p.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    args = p.parse_args()

    ds = vector_data.load(args.dataset, subset=args.subset, gt_k=max(100, args.k))
    nq = min(args.queries, ds.test.shape[0])
    test = ds.test[:nq]
    gt = ds.neighbors[:nq]
    k = args.k
    rows = []  # (engine, label, recall, mean_us)

    # rhypedb: build once, sweep ef + rerank.
    client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
    client.connect()
    id_to_idx, rb_build = build_rhypedb(client, ds, args.insert_chunk)
    rhypedb_cfgs = [
        ("baseline ef=50", 0, 0), ("ef:200", 200, 0), ("ef:500", 500, 0),
        ("rerank:50", 0, 50), ("rerank:100", 0, 100), ("rerank:200", 0, 200),
        ("rerank:500", 0, 500), ("rerank:1000", 0, 1000),
    ]
    for label, ef, rr in rhypedb_cfgs:
        ret, us = eval_rhypedb(client, test, k, id_to_idx, ef, rr)
        rows.append(("rhypedb", label, vector_data.recall_at_k(ret, gt, k), us))

    # pgvector: build once, sweep ef_search on the same index (runtime param).
    import psycopg
    conn = psycopg.connect(args.pg_dsn, autocommit=True)
    pg_build = build_pg(conn, ds)
    for ef_search in [10, 40, 100, 200, 400, 800]:
        ret, us = eval_pg(conn, test, k, ef_search)
        rows.append(("pgvector", f"ef_search={ef_search}", vector_data.recall_at_k(ret, gt, k), us))

    print(f"\nbuild: rhypedb {rb_build:.0f} ms | pgvector {pg_build:.0f} ms "
          f"(n={ds.train.shape[0]}, {nq} queries, k={k})\n")
    print(f"{'engine':9s} {'config':18s} {'recall@'+str(k):>10s} {'mean_us':>9s}")
    print("-" * 50)
    for engine, label, rec, us in sorted(rows, key=lambda r: r[2]):
        print(f"{engine:9s} {label:18s} {rec:>10.4f} {us:>9.0f}")


if __name__ == "__main__":
    main()
