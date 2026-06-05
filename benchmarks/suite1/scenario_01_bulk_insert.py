"""Scenario 1.1 — Bulk insert of users.

Measures total wall-clock for inserting N users one at a time, plus per-insert
latency distribution. Each insert is a separate HTTP/TCP/PG request (no batch
API yet on the rhypedb side — that's a separate backlog item).

Implementations:
  * `http`         — rhypedb HTTP+JSON
  * `tcp`          — rhypedb binary TCP, one User.create per row
  * `tcp-batch`    — rhypedb binary TCP, User.create_batch([...]) of size
                     `--batch-size` (default 1000) — the COPY-shape path
  * `pg-idiomatic` — one parameterized INSERT per row in autocommit (what
                     an ORM-style app gets out of the box)
  * `pg-optimal`   — single COPY ... FROM STDIN (the textbook fast path)

Run as `python -m benchmarks.suite1.scenario_01_bulk_insert`.
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


def _run_rhypedb(
    client,
    users: list,
    iterations: int,
    label: str,
    server_pattern: str,
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario="1.1 bulk_insert_users",
        implementation=label,
        iterations=iterations,
        metadata={"n_users": len(users)},
    )

    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0

    for it in range(iterations):
        suffix = f"_{label}_it{it}"
        op_latencies = []
        iter_start = common.time_ns()

        for u in users:
            q = (
                f'User.create({{ name: "{escape_str(u.name)}", '
                f'email: "{escape_str(u.email + suffix)}" }})'
            )
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"insert failed: {resp['error']}")

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


def _run_tcp_batch(
    users: list,
    iterations: int,
    host: str,
    port: int,
    batch_size: int,
    server_pattern: str,
) -> common.ScenarioResult:
    """Issue one User.create_batch per N rows. Per-row latency isn't a thing
    here — only per-batch wall-clock and the overall iteration time."""
    result = common.ScenarioResult(
        scenario="1.1 bulk_insert_users",
        implementation="rhypedb-tcp-batch",
        iterations=iterations,
        metadata={"n_users": len(users), "batch_size": batch_size},
    )

    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0

    client = RhypedbTcpClient(host, port)
    client.connect()
    try:
        for it in range(iterations):
            suffix = f"_tcpbatch_it{it}"
            batch_latencies = []
            iter_start = common.time_ns()
            for batch_start in range(0, len(users), batch_size):
                chunk = users[batch_start : batch_start + batch_size]
                # Build a `User.create_batch([{name,email}, ...])` query string.
                # We escape both fields the same way scenario_01 does for the
                # single-create path.
                row_strs = [
                    f'{{ name: "{escape_str(u.name)}", email: "{escape_str(u.email + suffix)}" }}'
                    for u in chunk
                ]
                q = f"User.create_batch([{','.join(row_strs)}])"
                op_start = common.time_ns()
                resp = client.query(q)
                op_end = common.time_ns()
                batch_latencies.append(op_end - op_start)
                if "error" in resp and resp.get("error"):
                    raise RuntimeError(f"batch insert failed: {resp['error']}")
            iter_end = common.time_ns()
            result.iteration_ms.append((iter_end - iter_start) / 1_000_000)
            # Per-op latency in this mode = per-batch latency; keep it so the
            # mean column has a meaningful number to report.
            result.operation_latency_ns.extend(batch_latencies)
            if server_pid:
                current = common.rss_mb(server_pid)
                result.rss_peak_mb = max(result.rss_peak_mb, current)
                result.rss_steady_mb = current
    finally:
        client.close()

    if server_pid:
        result.rss_post_load_mb = common.rss_mb(server_pid)
    return result


def _run_pg_idiomatic(
    users: list, iterations: int, dsn: str, pg_container: str
) -> common.ScenarioResult:
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    result = common.ScenarioResult(
        scenario="1.1 bulk_insert_users",
        implementation="pg-idiomatic",
        iterations=iterations,
        metadata={"n_users": len(users)},
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    try:
        for it in range(iterations):
            client.reset()
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for u in users:
                    op_start = common.time_ns()
                    cur.execute(
                        "INSERT INTO users (name, email) VALUES (%s, %s)",
                        (u.name, u.email),
                    )
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


def _run_pg_optimal(
    users: list, iterations: int, dsn: str, pg_container: str
) -> common.ScenarioResult:
    """COPY ... FROM STDIN. The whole insert is one streaming roundtrip, so
    there's no per-row latency to record — we only fill iteration_ms.
    """
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    result = common.ScenarioResult(
        scenario="1.1 bulk_insert_users",
        implementation="pg-optimal",
        iterations=iterations,
        metadata={"n_users": len(users), "note": "COPY: no per-row latency"},
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    try:
        for _ in range(iterations):
            client.reset()
            iter_start = common.time_ns()
            client.copy_rows(
                "COPY users (name, email) FROM STDIN",
                ((u.name, u.email) for u in users),
            )
            iter_end = common.time_ns()
            result.iteration_ms.append((iter_end - iter_start) / 1_000_000)

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
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="rows per create_batch call for the tcp-batch impl")
    parser.add_argument(
        "--impl",
        default="all",
        help="comma list: tcp,tcp-batch,http,pg-idiomatic,pg-optimal or aliases both/pg/all",
    )
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument(
        "--server-pattern",
        default="bench-rhypedb-data",
        help="pgrep -f pattern to find the rhypedb server PID for memory sampling",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_01.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)

    ds = data.generate(
        n_users=args.users,
        n_movies=1,
        ratings_per_user=0,
        friends_per_user=0,
    )
    print(f"Dataset: {ds.summary()}")
    print(f"Iterations: {args.iterations}")
    print(f"Impls: {impls}")
    print()

    results = []

    if "http" in impls:
        print(f"Running {len(ds.users) * args.iterations} HTTP inserts...")
        client = RhypedbHttpClient(args.http_url)
        t0 = time.time()
        results.append(_run_rhypedb(client, ds.users, args.iterations, "rhypedb-http", args.server_pattern))
        print(f"  HTTP done in {time.time() - t0:.1f}s")

    if "tcp" in impls:
        print(f"Running {len(ds.users) * args.iterations} TCP inserts...")
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        t0 = time.time()
        try:
            results.append(_run_rhypedb(client, ds.users, args.iterations, "rhypedb-tcp", args.server_pattern))
        finally:
            client.close()
        print(f"  TCP done in {time.time() - t0:.1f}s")

    if "tcp-batch" in impls:
        print(f"Running {args.iterations} iterations of {len(ds.users)} inserts via create_batch (batch_size={args.batch_size})...")
        t0 = time.time()
        results.append(_run_tcp_batch(
            ds.users, args.iterations, args.tcp_host, args.tcp_port,
            args.batch_size, args.server_pattern,
        ))
        print(f"  TCP batch done in {time.time() - t0:.1f}s")

    if "pg-idiomatic" in impls:
        print(f"Running {len(ds.users) * args.iterations} PG idiomatic inserts...")
        t0 = time.time()
        results.append(_run_pg_idiomatic(ds.users, args.iterations, args.pg_dsn, args.pg_container))
        print(f"  pg-idiomatic done in {time.time() - t0:.1f}s")

    if "pg-optimal" in impls:
        print(f"Running {args.iterations} PG COPY bulk inserts...")
        t0 = time.time()
        results.append(_run_pg_optimal(ds.users, args.iterations, args.pg_dsn, args.pg_container))
        print(f"  pg-optimal done in {time.time() - t0:.1f}s")

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
