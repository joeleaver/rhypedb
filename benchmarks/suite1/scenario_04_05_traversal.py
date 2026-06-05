"""Scenarios 1.4 (1-hop) and 1.5 (2-hop) — relationship traversal.

Setup: users + movies + ratings linking users to movies.

Queries:
  1.4: user → rated movies
  1.5: user → rated movies → users who also rated those movies

For Postgres we report two implementations per scenario:
  * pg-idiomatic: separate roundtripped queries (the ORM-style flow)
  * pg-optimal:   one JOIN (1.4) or 3-way JOIN (1.5)

Run as `python -m benchmarks.suite1.scenario_04_05_traversal`.
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


def setup_rhypedb_graph(ds: data.Dataset, tcp_host: str, tcp_port: int) -> list:
    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    try:
        user_ids = {}
        movie_ids = {}

        for u in ds.users:
            resp = client.query(
                f'User.create({{ name: "{escape_str(u.name)}", '
                f'email: "{escape_str(u.email + "_traversal")}" }})'
            )
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"user insert failed: {resp['error']}")
            user_ids[u.idx] = resp["object"]["id"]

        for m in ds.movies:
            resp = client.query(f'Movie.create({{ title: "{escape_str(m.title)}", year: {m.year} }})')
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"movie insert failed: {resp['error']}")
            movie_ids[m.idx] = resp["object"]["id"]

        for r in ds.ratings:
            uid = user_ids[r.user_idx]
            mid = movie_ids[r.movie_idx]
            resp = client.query(f"Rating.create({{ stars: {r.stars} }})")
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"rating insert failed: {resp['error']}")
            rating_id = resp["object"]["id"]
            for q in (
                f"Rating.get({rating_id}).link(User.get({uid}))",
                f"Rating.get({rating_id}).link(Movie.get({mid}))",
            ):
                resp = client.query(q)
                if "error" in resp and resp.get("error"):
                    raise RuntimeError(f"link failed: {resp['error']}")

        return list(user_ids.values())
    finally:
        client.close()


def setup_pg_graph(ds: data.Dataset, dsn: str) -> list:
    from benchmarks.harness.clients.postgres import PgClient

    client = PgClient(dsn, autocommit=True)
    try:
        client.reset()

        # Stream the dataset via COPY for fast setup. We need to reproduce the
        # idx→id mapping so we know which IDs to query against later.
        user_idx_to_id = {}
        with client.conn.cursor() as cur:
            with cur.copy("COPY users (name, email) FROM STDIN") as cp:
                for u in ds.users:
                    cp.write_row((u.name, u.email))
            # Reload to map back to data.User.idx order. Email is unique so
            # we can join on it.
            cur.execute("SELECT id, email FROM users")
            id_by_email = {email: uid for uid, email in cur.fetchall()}
            for u in ds.users:
                user_idx_to_id[u.idx] = id_by_email[u.email]

        movie_idx_to_id = {}
        with client.conn.cursor() as cur:
            with cur.copy("COPY movies (title, year) FROM STDIN") as cp:
                for m in ds.movies:
                    cp.write_row((m.title, m.year))
            cur.execute("SELECT id, title, year FROM movies ORDER BY id")
            rows = cur.fetchall()
            # COPY preserves insertion order, so row i corresponds to ds.movies[i].
            for i, (mid, _t, _y) in enumerate(rows):
                movie_idx_to_id[ds.movies[i].idx] = mid

        with client.conn.cursor() as cur:
            with cur.copy("COPY ratings (user_id, movie_id, stars) FROM STDIN") as cp:
                for r in ds.ratings:
                    cp.write_row(
                        (user_idx_to_id[r.user_idx], movie_idx_to_id[r.movie_idx], r.stars)
                    )

        return list(user_idx_to_id.values())
    finally:
        client.close()


def _run_rhypedb(
    client, user_ids, template, queries_per_iter, iterations, scenario, label, server_pattern
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario=scenario,
        implementation=label,
        iterations=iterations,
        metadata={"queries_per_iter": queries_per_iter, "template": template},
    )
    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0
    rng = random.Random(0)

    for _ in range(iterations):
        targets = [rng.choice(user_ids) for _ in range(queries_per_iter)]
        op_latencies = []
        iter_start = common.time_ns()
        for uid in targets:
            q = template.format(uid=uid)
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if "error" in resp and resp.get("error"):
                raise RuntimeError(f"traversal failed: {resp['error']}")
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


def _run_pg_traversal(
    user_ids, mode, hop, queries_per_iter, iterations, dsn, pg_container,
) -> common.ScenarioResult:
    """mode in {'idiomatic', 'optimal'}; hop in {1, 2}."""
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    scenario = {
        1: "1.4 traversal_1hop_user_to_movies",
        2: "1.5 traversal_2hop_user_to_users",
    }[hop]
    label = f"pg-{mode}"
    result = common.ScenarioResult(
        scenario=scenario,
        implementation=label,
        iterations=iterations,
        metadata={"queries_per_iter": queries_per_iter, "mode": mode, "hop": hop},
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    rng = random.Random(0)

    # Pre-build SQL strings so we don't pay python work inside the inner loop.
    if hop == 1 and mode == "optimal":
        sql_one = (
            "SELECT m.id, m.title, m.year FROM movies m "
            "JOIN ratings r ON r.movie_id = m.id WHERE r.user_id = %s"
        )
    elif hop == 2 and mode == "optimal":
        # Other users who rated any movie the given user rated. We exclude the
        # user themselves to match the rhypedb path's "other users" intent.
        sql_one = (
            "SELECT DISTINCT u2.id, u2.name, u2.email "
            "FROM ratings r1 "
            "JOIN ratings r2 ON r2.movie_id = r1.movie_id "
            "JOIN users u2 ON u2.id = r2.user_id "
            "WHERE r1.user_id = %s AND u2.id <> %s"
        )

    try:
        # Warm prepared statements with one sample id.
        sample = user_ids[0]
        with client.conn.cursor() as cur:
            if hop == 1 and mode == "idiomatic":
                for _ in range(2):
                    cur.execute("SELECT movie_id FROM ratings WHERE user_id=%s", (sample,))
                    movie_ids = [row[0] for row in cur.fetchall()]
                    if movie_ids:
                        cur.execute("SELECT id, title, year FROM movies WHERE id = ANY(%s)", (movie_ids,))
                        cur.fetchall()
            elif hop == 2 and mode == "idiomatic":
                for _ in range(2):
                    cur.execute("SELECT movie_id FROM ratings WHERE user_id=%s", (sample,))
                    movie_ids = [row[0] for row in cur.fetchall()]
                    if movie_ids:
                        cur.execute("SELECT DISTINCT user_id FROM ratings WHERE movie_id = ANY(%s)", (movie_ids,))
                        other_uids = [row[0] for row in cur.fetchall() if row[0] != sample]
                        if other_uids:
                            cur.execute("SELECT id, name, email FROM users WHERE id = ANY(%s)", (other_uids,))
                            cur.fetchall()
            else:
                for _ in range(2):
                    cur.execute(sql_one, (sample, sample) if hop == 2 else (sample,))
                    cur.fetchall()

        for _ in range(iterations):
            targets = [rng.choice(user_ids) for _ in range(queries_per_iter)]
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for uid in targets:
                    if mode == "optimal":
                        op_start = common.time_ns()
                        cur.execute(sql_one, (uid, uid) if hop == 2 else (uid,))
                        cur.fetchall()
                        op_end = common.time_ns()
                    elif hop == 1:
                        op_start = common.time_ns()
                        cur.execute("SELECT movie_id FROM ratings WHERE user_id=%s", (uid,))
                        mids = [row[0] for row in cur.fetchall()]
                        if mids:
                            cur.execute("SELECT id, title, year FROM movies WHERE id = ANY(%s)", (mids,))
                            cur.fetchall()
                        op_end = common.time_ns()
                    else:
                        op_start = common.time_ns()
                        cur.execute("SELECT movie_id FROM ratings WHERE user_id=%s", (uid,))
                        mids = [row[0] for row in cur.fetchall()]
                        if mids:
                            cur.execute("SELECT DISTINCT user_id FROM ratings WHERE movie_id = ANY(%s)", (mids,))
                            other = [row[0] for row in cur.fetchall() if row[0] != uid]
                            if other:
                                cur.execute("SELECT id, name, email FROM users WHERE id = ANY(%s)", (other,))
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
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--movies", type=int, default=100)
    parser.add_argument("--ratings-per-user", type=int, default=5)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--http-url", default="http://127.0.0.1:4300")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument("--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench")
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/scenario_04_05.json"))
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)
    ds = data.generate(
        n_users=args.users,
        n_movies=args.movies,
        ratings_per_user=args.ratings_per_user,
        friends_per_user=0,
    )
    print(f"Dataset: {ds.summary()}")

    rhypedb_uids = []
    pg_uids = []
    if "http" in impls or "tcp" in impls:
        print("Setup (rhypedb): inserting + linking via TCP...")
        t0 = time.time()
        rhypedb_uids = setup_rhypedb_graph(ds, args.tcp_host, args.tcp_port)
        print(f"  rhypedb setup done in {time.time() - t0:.1f}s")
    if any(i.startswith("pg-") for i in impls):
        print("Setup (PG): inserting via COPY...")
        t0 = time.time()
        pg_uids = setup_pg_graph(ds, args.pg_dsn)
        print(f"  PG setup done in {time.time() - t0:.1f}s")

    print(f"Traversal: {args.queries} queries × {args.iterations} iterations")
    print(f"Impls: {impls}")
    print()

    templates = [
        ("1.4 traversal_1hop_user_to_movies", "User.get({uid}).ratings.movie", 1),
        ("1.5 traversal_2hop_user_to_users",  "User.get({uid}).ratings.movie.ratings.user", 2),
    ]

    results = []
    for scenario, template, hop in templates:
        if "http" in impls:
            client = RhypedbHttpClient(args.http_url)
            results.append(_run_rhypedb(
                client, rhypedb_uids, template, args.queries, args.iterations,
                scenario, "rhypedb-http", args.server_pattern,
            ))
            client.close()
        if "tcp" in impls:
            client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
            client.connect()
            try:
                results.append(_run_rhypedb(
                    client, rhypedb_uids, template, args.queries, args.iterations,
                    scenario, "rhypedb-tcp", args.server_pattern,
                ))
            finally:
                client.close()
        if "pg-idiomatic" in impls:
            results.append(_run_pg_traversal(
                pg_uids, "idiomatic", hop, args.queries, args.iterations,
                args.pg_dsn, args.pg_container,
            ))
        if "pg-optimal" in impls:
            results.append(_run_pg_traversal(
                pg_uids, "optimal", hop, args.queries, args.iterations,
                args.pg_dsn, args.pg_container,
            ))

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
