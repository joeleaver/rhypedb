"""Scenario 1.7 — Unique constraint: violation + success paths.

For Postgres we report two implementations:
  * pg-idiomatic: plain INSERT; catch UniqueViolation (raises an exception)
  * pg-optimal:   INSERT ... ON CONFLICT (email) DO NOTHING (no exception)

The success path is identical for both (no conflict), but we run it under each
mode so the comparison rows stay symmetrical.

Run as `python -m benchmarks.suite1.scenario_07_unique_constraint`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.harness import common
from benchmarks.harness.clients.rhypedb_http import RhypedbHttpClient
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient


def _run_rhypedb(
    client, seed_email: str, n_attempts: int, iterations: int, label: str, server_pattern: str,
) -> list:
    server_pid = common.find_pid(server_pattern)
    base_rss = common.rss_mb(server_pid) if server_pid else 0

    violation = common.ScenarioResult(
        scenario="1.7a unique_violation", implementation=label,
        iterations=iterations, metadata={"attempts_per_iter": n_attempts},
        rss_cold_mb=base_rss,
    )
    success = common.ScenarioResult(
        scenario="1.7b unique_success", implementation=label,
        iterations=iterations, metadata={"attempts_per_iter": n_attempts},
        rss_cold_mb=base_rss,
    )

    for it in range(iterations):
        op_latencies = []
        iter_start = common.time_ns()
        for _ in range(n_attempts):
            q = f'User.create({{ name: "dup", email: "{seed_email}" }})'
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" not in resp or not resp.get("error"):
                raise RuntimeError("expected unique violation but insert succeeded")
        iter_end = common.time_ns()
        violation.iteration_ms.append((iter_end - iter_start) / 1_000_000)
        violation.operation_latency_ns.extend(op_latencies)

        op_latencies = []
        iter_start = common.time_ns()
        for i in range(n_attempts):
            email = f"u{label}_it{it}_i{i}@bench.test"
            q = f'User.create({{ name: "new", email: "{email}" }})'
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"success path failed: {resp['error']}")
        iter_end = common.time_ns()
        success.iteration_ms.append((iter_end - iter_start) / 1_000_000)
        success.operation_latency_ns.extend(op_latencies)

        if server_pid:
            current = common.rss_mb(server_pid)
            for r in (violation, success):
                r.rss_peak_mb = max(r.rss_peak_mb, current)
                r.rss_steady_mb = current

    if server_pid:
        for r in (violation, success):
            r.rss_post_load_mb = common.rss_mb(server_pid)
    return [violation, success]


def _run_pg(
    seed_email: str, n_attempts: int, iterations: int, mode: str, dsn: str, pg_container: str,
) -> list:
    """mode in {'idiomatic', 'optimal'}.

    idiomatic: try `INSERT ... VALUES`, catch `UniqueViolation`.
    optimal:   `INSERT ... ON CONFLICT (email) DO NOTHING RETURNING id`.
    """
    from psycopg.errors import UniqueViolation
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    pg_pid = find_pg_host_pid(pg_container)
    base_rss = common.rss_mb(pg_pid) if pg_pid else 0
    label = f"pg-{mode}"
    violation = common.ScenarioResult(
        scenario="1.7a unique_violation", implementation=label,
        iterations=iterations, metadata={"attempts_per_iter": n_attempts, "mode": mode},
        rss_cold_mb=base_rss,
    )
    success = common.ScenarioResult(
        scenario="1.7b unique_success", implementation=label,
        iterations=iterations, metadata={"attempts_per_iter": n_attempts, "mode": mode},
        rss_cold_mb=base_rss,
    )

    client = PgClient(dsn, autocommit=True)
    try:
        # Seed the duplicate-target row. Use ON CONFLICT so re-runs are idempotent.
        with client.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s) ON CONFLICT (email) DO NOTHING",
                ("seed", seed_email),
            )

        for it in range(iterations):
            # --- violation path -------------------------------------------
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                if mode == "idiomatic":
                    for _ in range(n_attempts):
                        op_start = common.time_ns()
                        try:
                            cur.execute(
                                "INSERT INTO users (name, email) VALUES (%s, %s)",
                                ("dup", seed_email),
                            )
                            op_end = common.time_ns()
                            raise RuntimeError("expected unique violation")
                        except UniqueViolation:
                            op_end = common.time_ns()
                        op_latencies.append(op_end - op_start)
                else:  # optimal
                    for _ in range(n_attempts):
                        op_start = common.time_ns()
                        cur.execute(
                            "INSERT INTO users (name, email) VALUES (%s, %s) "
                            "ON CONFLICT (email) DO NOTHING RETURNING id",
                            ("dup", seed_email),
                        )
                        cur.fetchall()
                        op_end = common.time_ns()
                        op_latencies.append(op_end - op_start)
            iter_end = common.time_ns()
            violation.iteration_ms.append((iter_end - iter_start) / 1_000_000)
            violation.operation_latency_ns.extend(op_latencies)

            # --- success path ---------------------------------------------
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                if mode == "idiomatic":
                    for i in range(n_attempts):
                        email = f"pg_{mode}_it{it}_i{i}@bench.test"
                        op_start = common.time_ns()
                        cur.execute(
                            "INSERT INTO users (name, email) VALUES (%s, %s)",
                            ("new", email),
                        )
                        op_end = common.time_ns()
                        op_latencies.append(op_end - op_start)
                else:
                    for i in range(n_attempts):
                        email = f"pg_{mode}_it{it}_i{i}@bench.test"
                        op_start = common.time_ns()
                        cur.execute(
                            "INSERT INTO users (name, email) VALUES (%s, %s) "
                            "ON CONFLICT (email) DO NOTHING RETURNING id",
                            ("new", email),
                        )
                        cur.fetchall()
                        op_end = common.time_ns()
                        op_latencies.append(op_end - op_start)
            iter_end = common.time_ns()
            success.iteration_ms.append((iter_end - iter_start) / 1_000_000)
            success.operation_latency_ns.extend(op_latencies)

            if pg_pid:
                current = common.rss_mb(pg_pid)
                for r in (violation, success):
                    r.rss_peak_mb = max(r.rss_peak_mb, current)
                    r.rss_steady_mb = current
    finally:
        client.close()

    if pg_pid:
        for r in (violation, success):
            r.rss_post_load_mb = common.rss_mb(pg_pid)
    return [violation, success]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_07.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)

    seed_email = "seed_unique_scenario@bench.test"

    if "http" in impls or "tcp" in impls:
        tcp = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        tcp.connect()
        try:
            resp = tcp.query(f'User.create({{ name: "seed", email: "{seed_email}" }})')
            if "error" in resp and resp.get("error") and "unique constraint" not in resp["error"]:
                raise RuntimeError(f"seed setup: {resp['error']}")
        finally:
            tcp.close()

    print(f"Unique constraint: {args.attempts} attempts × {args.iterations} iterations")
    print(f"Impls: {impls}")
    print()

    results = []
    if "http" in impls:
        client = RhypedbHttpClient(args.http_url)
        t0 = time.time()
        results.extend(_run_rhypedb(
            client, seed_email, args.attempts, args.iterations, "rhypedb-http", args.server_pattern,
        ))
        print(f"  HTTP done in {time.time() - t0:.1f}s")
        client.close()

    if "tcp" in impls:
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        try:
            t0 = time.time()
            results.extend(_run_rhypedb(
                client, seed_email, args.attempts, args.iterations, "rhypedb-tcp", args.server_pattern,
            ))
            print(f"  TCP done in {time.time() - t0:.1f}s")
        finally:
            client.close()

    if "pg-idiomatic" in impls:
        t0 = time.time()
        results.extend(_run_pg(
            seed_email, args.attempts, args.iterations, "idiomatic", args.pg_dsn, args.pg_container,
        ))
        print(f"  pg-idiomatic done in {time.time() - t0:.1f}s")

    if "pg-optimal" in impls:
        t0 = time.time()
        results.extend(_run_pg(
            seed_email, args.attempts, args.iterations, "optimal", args.pg_dsn, args.pg_container,
        ))
        print(f"  pg-optimal done in {time.time() - t0:.1f}s")

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
