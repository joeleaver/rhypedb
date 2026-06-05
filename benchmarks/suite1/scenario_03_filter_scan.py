"""Scenario 1.3 — Filter scan.

Setup: insert N movies with varied years.
Measure: `Movie.filter(year > X).limit(50)` / `SELECT ... WHERE year > $1
LIMIT 50` returning up to 50 rows per query.

Run as `python -m benchmarks.suite1.scenario_03_filter_scan`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.harness import common, data
from benchmarks.harness.clients.rhypedb_http import RhypedbHttpClient
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient


def escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def setup_rhypedb_movies(movies: list, tcp_host: str, tcp_port: int) -> None:
    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    try:
        for m in movies:
            resp = client.query(
                f'Movie.create({{ title: "{escape_str(m.title)}", year: {m.year} }})'
            )
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"setup insert failed: {resp['error']}")
    finally:
        client.close()


def setup_pg_movies(movies: list, dsn: str) -> None:
    from benchmarks.harness.clients.postgres import PgClient

    client = PgClient(dsn, autocommit=True)
    try:
        client.reset()
        client.copy_rows(
            "COPY movies (title, year) FROM STDIN",
            ((m.title, m.year) for m in movies),
        )
    finally:
        client.close()


def _run_rhypedb(
    client, queries: list, iterations: int, label: str, server_pattern: str
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario="1.3 filter_scan_movies",
        implementation=label,
        iterations=iterations,
        metadata={"queries_per_iter": len(queries)},
    )
    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0

    for _ in range(iterations):
        op_latencies = []
        iter_start = common.time_ns()
        for q in queries:
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"filter failed: {resp['error']}")
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
    thresholds: list, iterations: int, dsn: str, pg_container: str
) -> common.ScenarioResult:
    """Idiomatic and optimal are the same query — single SELECT with WHERE +
    LIMIT. We report one `pg-idiomatic` row.
    """
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    result = common.ScenarioResult(
        scenario="1.3 filter_scan_movies",
        implementation="pg-idiomatic",
        iterations=iterations,
        metadata={
            "queries_per_iter": len(thresholds),
            "note": "optimal == idiomatic for single-table WHERE + LIMIT",
        },
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    sql = "SELECT id, title, year FROM movies WHERE year > %s LIMIT 50"
    client = PgClient(dsn, autocommit=True)
    try:
        # Warm prepared statement.
        with client.conn.cursor() as cur:
            cur.execute(sql, (thresholds[0],))
            cur.fetchall()
            cur.execute(sql, (thresholds[0],))
            cur.fetchall()

        for _ in range(iterations):
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for t in thresholds:
                    op_start = common.time_ns()
                    cur.execute(sql, (t,))
                    cur.fetchall()
                    op_end = common.time_ns()
                    op_latencies.append(op_end - op_start)
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
    parser.add_argument("--movies", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_03.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)
    ds = data.generate(
        n_users=1,
        n_movies=args.movies,
        ratings_per_user=0,
        friends_per_user=0,
    )

    # Same year thresholds for both engines so the rowset sizes match.
    thresholds = [1950 + (i * 75 // args.queries) for i in range(args.queries)]

    if "http" in impls or "tcp" in impls:
        print(f"Setup (rhypedb): inserting {args.movies} movies via TCP...")
        t0 = time.time()
        setup_rhypedb_movies(ds.movies, args.tcp_host, args.tcp_port)
        print(f"  rhypedb setup done in {time.time() - t0:.1f}s")

    if any(i.startswith("pg-") for i in impls):
        print(f"Setup (PG): inserting {args.movies} movies via COPY...")
        t0 = time.time()
        setup_pg_movies(ds.movies, args.pg_dsn)
        print(f"  PG setup done in {time.time() - t0:.1f}s")

    rhypedb_queries = [f"Movie.filter(.year > {t}).limit(50)" for t in thresholds]

    print(f"Filter: {len(thresholds)} queries × {args.iterations} iterations")
    print(f"Impls: {impls}")
    print()

    results = []

    if "http" in impls:
        client = RhypedbHttpClient(args.http_url)
        t0 = time.time()
        results.append(_run_rhypedb(client, rhypedb_queries, args.iterations, "rhypedb-http", args.server_pattern))
        print(f"  HTTP done in {time.time() - t0:.1f}s")
        client.close()

    if "tcp" in impls:
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        try:
            t0 = time.time()
            results.append(_run_rhypedb(client, rhypedb_queries, args.iterations, "rhypedb-tcp", args.server_pattern))
            print(f"  TCP done in {time.time() - t0:.1f}s")
        finally:
            client.close()

    if "pg-idiomatic" in impls or "pg-optimal" in impls:
        t0 = time.time()
        results.append(_run_pg(thresholds, args.iterations, args.pg_dsn, args.pg_container))
        print(f"  pg done in {time.time() - t0:.1f}s")

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
