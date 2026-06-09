"""A/B the `.similar()` recall knobs (ef / full-precision rerank) on ONE index.

Each `run_vector_bench.sh` invocation rebuilds the HNSW graph with an unseeded
`random_level`, so recall varies run-to-run from topology alone — comparing a
rerank run against a *separate* baseline run conflates the knob with build
variance. This script builds the index ONCE and then evaluates several
(ef, rerank) configs against that single graph, so the only thing that changes
between configs is the query knob. Connects to an already-running rhypedb server
(the wrapper starts/stops it), mirroring scenario_09's build path.
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmarks.harness import common, vector_data
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient
from benchmarks.suite1.scenario_09_vector import _fmt_vec


def _eval_config(client, test, k, id_to_idx, ef, rerank):
    """Run the query set under one (ef, rerank) config; return (recall, mean_us)."""
    params = ""
    if ef:
        params += f", ef: {ef}"
    if rerank:
        params += f", rerank: {rerank}"
    retrieved = np.full((len(test), k), -1, dtype=np.int64)
    lat_ns = []
    for qi in range(len(test)):
        q = _fmt_vec(test[qi])
        t0 = common.time_ns()
        resp = client.query(f"Vec.similar(.embedding, {q}, k: {k}{params})")
        lat_ns.append(common.time_ns() - t0)
        if resp.get("error"):
            raise RuntimeError(f"similar: {resp['error']}")
        for j, o in enumerate(resp.get("objects", [])[:k]):
            retrieved[qi, j] = id_to_idx.get(o["id"], -1)
    return retrieved, float(np.mean(lat_ns) / 1000.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic-384-angular")
    p.add_argument("--subset", type=int, default=30000)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--insert-chunk", type=int, default=5000)
    p.add_argument("--tcp-host", default="127.0.0.1")
    p.add_argument("--tcp-port", type=int, default=4501)
    args = p.parse_args()

    ds = vector_data.load(args.dataset, subset=args.subset, gt_k=max(100, args.k))
    n_queries = min(args.queries, ds.test.shape[0])
    test = ds.test[:n_queries]
    gt = ds.neighbors[:n_queries]
    k = args.k

    client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
    client.connect()

    # Build the index ONCE.
    train = ds.train
    n = train.shape[0]
    id_to_idx: dict[int, int] = {}
    build_start = common.time_ns()
    for start in range(0, n, args.insert_chunk):
        end = min(start + args.insert_chunk, n)
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
    build_ms = (common.time_ns() - build_start) / 1_000_000
    print(f"built {n} vectors in {build_ms:.0f} ms; evaluating {n_queries} queries, k={k}\n")

    configs = [
        ("baseline (ef=50)", 0, 0),
        ("ef:200", 200, 0),
        ("ef:500", 500, 0),
        ("rerank:50", 0, 50),
        ("rerank:200", 0, 200),
        ("rerank:500", 0, 500),
        ("ef:500 + rerank:500", 500, 500),
    ]
    print(f"{'config':22s} {'recall@'+str(k):>10s} {'mean_us':>9s}")
    print("-" * 44)
    for label, ef, rerank in configs:
        retrieved, mean_us = _eval_config(client, test, k, id_to_idx, ef, rerank)
        recall = vector_data.recall_at_k(retrieved, gt, k)
        print(f"{label:22s} {recall:>10.4f} {mean_us:>9.0f}")


if __name__ == "__main__":
    main()
