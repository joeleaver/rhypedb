"""Scenario 1.6 — Cascade delete via on_delete policy / FK CASCADE.

Setup: insert N user graphs, each with K ratings linked.
Measure: deleting each user. rhypedb cascades via @on_delete(cascade);
Postgres cascades via ON DELETE CASCADE.

Idiomatic vs optimal collapse for the per-user delete — a `DELETE FROM users
WHERE id=$1` is both the natural and the optimal expression — so we report a
single `pg-idiomatic` PG result.

Run as `python -m benchmarks.suite1.scenario_06_cascade_delete`.
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


def setup_rhypedb_movies(tcp_host: str, tcp_port: int, n_movies: int) -> list:
    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    movie_ids = []
    try:
        for i in range(n_movies):
            resp = client.query(f'Movie.create({{ title: "Movie{i}_cascade", year: 2000 }})')
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"movie setup: {resp['error']}")
            movie_ids.append(resp["object"]["id"])
    finally:
        client.close()
    return movie_ids


def setup_rhypedb_user_batch(
    client, user_idx_offset: int, n_users: int, ratings_per_user: int, movie_ids: list, suffix: str,
) -> list:
    """Insert n_users users + their ratings; return user ids.

    Each rating is created with inline `user` + `movie` relations so the
    engine writes the object + both forward/rev edges in one txn — three
    times faster than the historical `Rating.create + link + link` shape.
    """
    user_ids = []
    for i in range(n_users):
        n = user_idx_offset + i
        resp = client.query(
            f'User.create({{ name: "User{n}", email: "del{n}{suffix}@bench.test" }})'
        )
        if "error" in resp and resp.get("error"):
            raise RuntimeError(f"user insert failed: {resp['error']}")
        uid = resp["object"]["id"]
        user_ids.append(uid)
        for k in range(ratings_per_user):
            mid = movie_ids[k % len(movie_ids)]
            resp = client.query(
                f"Rating.create({{ stars: 3.5, user: {uid}, movie: {mid} }})"
            )
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"rating insert failed: {resp['error']}")
    return user_ids


def _run_rhypedb(
    client, movie_ids, n_deletes, ratings_per_user, iterations, label, server_pattern,
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario="1.6 cascade_delete_user",
        implementation=label,
        iterations=iterations,
        metadata={"n_deletes": n_deletes, "ratings_per_user": ratings_per_user},
    )
    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0

    for it in range(iterations):
        suffix = f"_{label}_it{it}"
        user_ids = setup_rhypedb_user_batch(
            client, it * n_deletes, n_deletes, ratings_per_user, movie_ids, suffix,
        )
        op_latencies = []
        iter_start = common.time_ns()
        for uid in user_ids:
            op_start = common.time_ns()
            resp = client.query(f"User.get({uid}).delete()")
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"delete failed: {resp['error']}")
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
    n_deletes: int, ratings_per_user: int, iterations: int, dsn: str, pg_container: str,
) -> common.ScenarioResult:
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    result = common.ScenarioResult(
        scenario="1.6 cascade_delete_user",
        implementation="pg-idiomatic",
        iterations=iterations,
        metadata={
            "n_deletes": n_deletes,
            "ratings_per_user": ratings_per_user,
            "note": "optimal == idiomatic for DELETE+FK cascade",
        },
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    try:
        # Set up the movie pool once.
        client.reset()
        with client.conn.cursor() as cur:
            with cur.copy("COPY movies (title, year) FROM STDIN") as cp:
                for i in range(20):
                    cp.write_row((f"Movie{i}_cascade", 2000))
            cur.execute("SELECT id FROM movies ORDER BY id")
            movie_ids = [row[0] for row in cur.fetchall()]

        # Warm prepared DELETE statement.
        with client.conn.cursor() as cur:
            cur.execute("INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id", ("warm", "warm@bench.test"))
            warm_id = cur.fetchone()[0]
            cur.execute("DELETE FROM users WHERE id = %s", (warm_id,))
            cur.execute("INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id", ("warm2", "warm2@bench.test"))
            warm_id = cur.fetchone()[0]
            cur.execute("DELETE FROM users WHERE id = %s", (warm_id,))

        for it in range(iterations):
            # Seed n_deletes users, each with ratings_per_user ratings.
            with client.conn.cursor() as cur:
                rows = [
                    (f"User{it * n_deletes + i}", f"pgdel{it}_{i}@bench.test")
                    for i in range(n_deletes)
                ]
                with cur.copy("COPY users (name, email) FROM STDIN") as cp:
                    for row in rows:
                        cp.write_row(row)
                emails = tuple(r[1] for r in rows)
                cur.execute(
                    "SELECT id, email FROM users WHERE email = ANY(%s)",
                    (list(emails),),
                )
                id_by_email = {email: uid for uid, email in cur.fetchall()}
                user_ids = [id_by_email[r[1]] for r in rows]

                rating_rows = []
                for uid in user_ids:
                    for k in range(ratings_per_user):
                        rating_rows.append((uid, movie_ids[k % len(movie_ids)], 3.5))
                with cur.copy("COPY ratings (user_id, movie_id, stars) FROM STDIN") as cp:
                    for row in rating_rows:
                        cp.write_row(row)

            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for uid in user_ids:
                    op_start = common.time_ns()
                    cur.execute("DELETE FROM users WHERE id = %s", (uid,))
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
    parser.add_argument("--n-deletes", type=int, default=100)
    parser.add_argument("--ratings-per-user", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_06.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)
    movie_ids = []
    if "http" in impls or "tcp" in impls:
        movie_ids = setup_rhypedb_movies(args.tcp_host, args.tcp_port, n_movies=20)
        print(f"Set up {len(movie_ids)} shared rhypedb movies")

    print(f"Cascade delete: {args.n_deletes} deletes × {args.iterations} iterations")
    print(f"Each user has {args.ratings_per_user} cascading ratings")
    print(f"Impls: {impls}")
    print()

    results = []

    if "http" in impls:
        client = RhypedbHttpClient(args.http_url)
        t0 = time.time()
        results.append(_run_rhypedb(
            client, movie_ids, args.n_deletes, args.ratings_per_user, args.iterations,
            "rhypedb-http", args.server_pattern,
        ))
        print(f"  HTTP done in {time.time() - t0:.1f}s")
        client.close()

    if "tcp" in impls:
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        try:
            t0 = time.time()
            results.append(_run_rhypedb(
                client, movie_ids, args.n_deletes, args.ratings_per_user, args.iterations,
                "rhypedb-tcp", args.server_pattern,
            ))
            print(f"  TCP done in {time.time() - t0:.1f}s")
        finally:
            client.close()

    if "pg-idiomatic" in impls or "pg-optimal" in impls:
        t0 = time.time()
        results.append(_run_pg(
            args.n_deletes, args.ratings_per_user, args.iterations,
            args.pg_dsn, args.pg_container,
        ))
        print(f"  pg done in {time.time() - t0:.1f}s")

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
