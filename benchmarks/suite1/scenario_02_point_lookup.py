"""Scenario 1.2 — Point lookup by ID.

Setup: insert N users; collect their IDs.
Measure: K random `User.get(id)` (or `SELECT FROM users WHERE id=$1`) lookups
per iteration.

Run as `python -m benchmarks.suite1.scenario_02_point_lookup`.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from benchmarks.harness import common, data
from benchmarks.harness.clients.rhypedb_http import RhypedbHttpClient
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient


def escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def setup_rhypedb_users(users: list, tcp_host: str, tcp_port: int) -> list:
    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    try:
        ids = []
        for u in users:
            q = (
                f'User.create({{ name: "{escape_str(u.name)}", '
                f'email: "{escape_str(u.email + "_lookup_setup")}" }})'
            )
            resp = client.query(q)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"setup insert failed: {resp['error']}")
            ids.append(resp["object"]["id"])
        return ids
    finally:
        client.close()


def setup_pg_users(users: list, dsn: str) -> list:
    from benchmarks.harness.clients.postgres import PgClient

    client = PgClient(dsn, autocommit=True)
    try:
        client.reset()
        ids = []
        with client.conn.cursor() as cur:
            for u in users:
                cur.execute(
                    "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id",
                    (u.name, u.email),
                )
                ids.append(cur.fetchone()[0])
        return ids
    finally:
        client.close()


def _run_rhypedb(
    client,
    ids: list,
    lookups_per_iter: int,
    iterations: int,
    label: str,
    server_pattern: str,
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario="1.2 point_lookup_by_id",
        implementation=label,
        iterations=iterations,
        metadata={"n_users": len(ids), "lookups_per_iter": lookups_per_iter},
    )
    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0
    rng = random.Random(0)

    for _ in range(iterations):
        ids_to_lookup = [rng.choice(ids) for _ in range(lookups_per_iter)]
        op_latencies = []
        iter_start = common.time_ns()
        for uid in ids_to_lookup:
            op_start = common.time_ns()
            resp = client.query(f"User.get({uid})")
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"lookup failed: {resp['error']}")
            objs = resp.get("objects") or ([resp["object"]] if resp.get("object") else [])
            if not objs:
                raise RuntimeError(f"lookup returned no object for id {uid}")
        iter_end = common.time_ns()
        result.iteration_ms.append((iter_end - iter_start) / 1_000_000)
        result.operation_latency_ns.extend(op_latencies)
        if server_pid:
            current = common.rss_mb(server_pid)
            result.rss_peak_mb = max(result.rss_peak_mb, current)
            result.rss_steady_mb = current

    if server_pid:
        result.rss_post_load_mb = common.rss_mb(server_pid)
    return result


def _run_pg(
    ids: list,
    lookups_per_iter: int,
    iterations: int,
    dsn: str,
    pg_container: str,
) -> common.ScenarioResult:
    """Point lookup by primary key. The idiomatic and optimal forms are the
    same query — there's no faster way to fetch a row by its PK — so we report
    a single `pg-idiomatic` result and note the equivalence in metadata.
    """
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    result = common.ScenarioResult(
        scenario="1.2 point_lookup_by_id",
        implementation="pg-idiomatic",
        iterations=iterations,
        metadata={
            "n_users": len(ids),
            "lookups_per_iter": lookups_per_iter,
            "note": "optimal == idiomatic for PK lookup",
        },
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    rng = random.Random(0)
    try:
        # Warm the prepared statement cache so the first measured call is
        # representative of the steady state.
        with client.conn.cursor() as cur:
            cur.execute("SELECT id, name, email FROM users WHERE id=%s", (ids[0],))
            cur.fetchone()
            cur.execute("SELECT id, name, email FROM users WHERE id=%s", (ids[0],))
            cur.fetchone()

        for _ in range(iterations):
            ids_to_lookup = [rng.choice(ids) for _ in range(lookups_per_iter)]
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for uid in ids_to_lookup:
                    op_start = common.time_ns()
                    cur.execute("SELECT id, name, email FROM users WHERE id=%s", (uid,))
                    row = cur.fetchone()
                    op_end = common.time_ns()
                    op_latencies.append(op_end - op_start)
                    if row is None:
                        raise RuntimeError(f"lookup returned no row for id {uid}")
            iter_end = common.time_ns()
            result.iteration_ms.append((iter_end - iter_start) / 1_000_000)
            result.operation_latency_ns.extend(op_latencies)
            if pg_pid:
                current = common.rss_mb(pg_pid)
                result.rss_peak_mb = max(result.rss_peak_mb, current)
                result.rss_steady_mb = current
    finally:
        client.close()

    if pg_pid:
        result.rss_post_load_mb = common.rss_mb(pg_pid)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--lookups", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_02.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)
    ds = data.generate(
        n_users=args.users,
        n_movies=1,
        ratings_per_user=0,
        friends_per_user=0,
    )

    rhypedb_ids = []
    if "http" in impls or "tcp" in impls:
        print(f"Setup (rhypedb): inserting {args.users} users via TCP...")
        t0 = time.time()
        rhypedb_ids = setup_rhypedb_users(ds.users, args.tcp_host, args.tcp_port)
        print(f"  rhypedb setup done in {time.time() - t0:.1f}s")

    pg_ids = []
    if any(i.startswith("pg-") for i in impls):
        print(f"Setup (PG): inserting {args.users} users via psycopg...")
        t0 = time.time()
        pg_ids = setup_pg_users(ds.users, args.pg_dsn)
        print(f"  PG setup done in {time.time() - t0:.1f}s")

    print(f"Lookups: {args.lookups} random per iteration, {args.iterations} iterations")
    print(f"Impls: {impls}")
    print()

    results = []

    if "http" in impls:
        client = RhypedbHttpClient(args.http_url)
        t0 = time.time()
        results.append(_run_rhypedb(client, rhypedb_ids, args.lookups, args.iterations, "rhypedb-http", args.server_pattern))
        print(f"  HTTP done in {time.time() - t0:.1f}s")
        client.close()

    if "tcp" in impls:
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        try:
            t0 = time.time()
            results.append(_run_rhypedb(client, rhypedb_ids, args.lookups, args.iterations, "rhypedb-tcp", args.server_pattern))
            print(f"  TCP done in {time.time() - t0:.1f}s")
        finally:
            client.close()

    if "pg-idiomatic" in impls or "pg-optimal" in impls:
        # Idiomatic and optimal collapse for PK lookup; report one row.
        t0 = time.time()
        results.append(_run_pg(pg_ids, args.lookups, args.iterations, args.pg_dsn, args.pg_container))
        print(f"  pg done in {time.time() - t0:.1f}s")

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
