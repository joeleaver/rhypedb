"""Scenario 1.8 — 3-hop traversal via chained 1:1 forward relations.

Setup: users + movies + directors + ratings. Each movie has one director.
Each rating links a user and a movie.

Query: `User.get(X).ratings.movie.director` (the rated movies' directors).

rhypedb uses 3-hop covering: the rating's reverse-edge embeds the movie
under `movie__cover`, which itself carries `director` (a u64) plus
`director__cover` (the director's serialized scalars) — so the final
hop satisfies from cover bytes without a per-movie LSM probe.

For Postgres we report two implementations:
  * pg-idiomatic: separate roundtripped queries
  * pg-optimal:   one 4-way JOIN

Run as `python -m benchmarks.suite1.scenario_08_3hop`.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness import common, data
from benchmarks.harness.clients.rhypedb_tcp import RhypedbTcpClient


@dataclass
class Director:
    idx: int
    name: str


def generate_directors(n: int, seed: int = 42) -> list:
    rng = random.Random(seed + 1)
    return [
        Director(
            idx=i,
            name=f"{rng.choice(data.FIRST_NAMES)} {rng.choice(data.LAST_NAMES)}",
        )
        for i in range(n)
    ]


def escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def setup_rhypedb_graph(
    ds: data.Dataset, directors: list, tcp_host: str, tcp_port: int
) -> list:
    """Insert users + directors + movies + ratings. Movies are linked to a
    round-robin director. Inline relations on Movie + Rating so symmetric
    covers (and the 3-hop director embed) land on the write path."""
    client = RhypedbTcpClient(tcp_host, tcp_port)
    client.connect()
    try:
        director_ids = {}
        for d in directors:
            resp = client.query(
                f'Director.create({{ name: "{escape_str(d.name)}" }})'
            )
            if resp.get("error"):
                raise RuntimeError(f"director insert failed: {resp['error']}")
            director_ids[d.idx] = resp["object"]["id"]

        user_ids = {}
        for u in ds.users:
            resp = client.query(
                f'User.create({{ name: "{escape_str(u.name)}", '
                f'email: "{escape_str(u.email + "_3hop")}" }})'
            )
            if resp.get("error"):
                raise RuntimeError(f"user insert failed: {resp['error']}")
            user_ids[u.idx] = resp["object"]["id"]

        movie_ids = {}
        for m in ds.movies:
            did = director_ids[m.idx % len(directors)]
            resp = client.query(
                f'Movie.create({{ title: "{escape_str(m.title)}", year: {m.year}, '
                f"director: {did} }})"
            )
            if resp.get("error"):
                raise RuntimeError(f"movie insert failed: {resp['error']}")
            movie_ids[m.idx] = resp["object"]["id"]

        for r in ds.ratings:
            uid = user_ids[r.user_idx]
            mid = movie_ids[r.movie_idx]
            resp = client.query(
                f"Rating.create({{ stars: {r.stars}, user: {uid}, movie: {mid} }})"
            )
            if resp.get("error"):
                raise RuntimeError(f"rating insert failed: {resp['error']}")

        return list(user_ids.values())
    finally:
        client.close()


def setup_pg_graph(
    ds: data.Dataset, directors: list, dsn: str
) -> list:
    from benchmarks.harness.clients.postgres import PgClient

    client = PgClient(dsn, autocommit=True)
    try:
        client.reset()

        user_idx_to_id = {}
        with client.conn.cursor() as cur:
            with cur.copy("COPY users (name, email) FROM STDIN") as cp:
                for u in ds.users:
                    cp.write_row((u.name, u.email))
            cur.execute("SELECT id, email FROM users")
            id_by_email = {email: uid for uid, email in cur.fetchall()}
            for u in ds.users:
                user_idx_to_id[u.idx] = id_by_email[u.email]

        director_idx_to_id = {}
        with client.conn.cursor() as cur:
            with cur.copy("COPY directors (name) FROM STDIN") as cp:
                for d in directors:
                    cp.write_row((d.name,))
            cur.execute("SELECT id, name FROM directors ORDER BY id")
            rows = cur.fetchall()
            for i, (did, _n) in enumerate(rows):
                director_idx_to_id[directors[i].idx] = did

        movie_idx_to_id = {}
        with client.conn.cursor() as cur:
            with cur.copy(
                "COPY movies (title, year, director_id) FROM STDIN"
            ) as cp:
                for m in ds.movies:
                    did = director_idx_to_id[m.idx % len(directors)]
                    cp.write_row((m.title, m.year, did))
            cur.execute("SELECT id, title, year FROM movies ORDER BY id")
            rows = cur.fetchall()
            for i, (mid, _t, _y) in enumerate(rows):
                movie_idx_to_id[ds.movies[i].idx] = mid

        with client.conn.cursor() as cur:
            with cur.copy("COPY ratings (user_id, movie_id, stars) FROM STDIN") as cp:
                for r in ds.ratings:
                    cp.write_row(
                        (
                            user_idx_to_id[r.user_idx],
                            movie_idx_to_id[r.movie_idx],
                            r.stars,
                        )
                    )

        return list(user_idx_to_id.values())
    finally:
        client.close()


def _run_rhypedb(
    client, user_ids, queries_per_iter, iterations, server_pattern
) -> common.ScenarioResult:
    result = common.ScenarioResult(
        scenario="1.8 traversal_3hop_user_to_directors",
        implementation="rhypedb-tcp",
        iterations=iterations,
        metadata={
            "queries_per_iter": queries_per_iter,
            "template": "User.get({uid}).ratings.movie.director",
        },
    )
    server_pid = common.find_pid(server_pattern)
    result.rss_cold_mb = common.rss_mb(server_pid) if server_pid else 0
    rng = random.Random(0)

    for _ in range(iterations):
        targets = [rng.choice(user_ids) for _ in range(queries_per_iter)]
        op_latencies = []
        iter_start = common.time_ns()
        for uid in targets:
            q = f"User.get({uid}).ratings.movie.director"
            op_start = common.time_ns()
            resp = client.query(q)
            op_end = common.time_ns()
            op_latencies.append(op_end - op_start)
            if resp.get("error"):
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


def _run_pg(
    user_ids, mode, queries_per_iter, iterations, dsn, pg_container,
) -> common.ScenarioResult:
    from benchmarks.harness.clients.postgres import PgClient, find_pg_host_pid

    label = f"pg-{mode}"
    result = common.ScenarioResult(
        scenario="1.8 traversal_3hop_user_to_directors",
        implementation=label,
        iterations=iterations,
        metadata={"queries_per_iter": queries_per_iter, "mode": mode},
    )
    pg_pid = find_pg_host_pid(pg_container)
    result.rss_cold_mb = common.rss_mb(pg_pid) if pg_pid else 0

    client = PgClient(dsn, autocommit=True)
    rng = random.Random(0)

    sql_one = (
        "SELECT DISTINCT d.id, d.name "
        "FROM ratings r "
        "JOIN movies m ON m.id = r.movie_id "
        "JOIN directors d ON d.id = m.director_id "
        "WHERE r.user_id = %s"
    )

    try:
        # Warm prepared statements.
        sample = user_ids[0]
        with client.conn.cursor() as cur:
            if mode == "optimal":
                for _ in range(2):
                    cur.execute(sql_one, (sample,))
                    cur.fetchall()
            else:
                for _ in range(2):
                    cur.execute(
                        "SELECT movie_id FROM ratings WHERE user_id=%s",
                        (sample,),
                    )
                    mids = [row[0] for row in cur.fetchall()]
                    if mids:
                        cur.execute(
                            "SELECT director_id FROM movies WHERE id = ANY(%s)",
                            (mids,),
                        )
                        dids = [row[0] for row in cur.fetchall() if row[0] is not None]
                        if dids:
                            cur.execute(
                                "SELECT id, name FROM directors WHERE id = ANY(%s)",
                                (dids,),
                            )
                            cur.fetchall()

        for _ in range(iterations):
            targets = [rng.choice(user_ids) for _ in range(queries_per_iter)]
            op_latencies = []
            iter_start = common.time_ns()
            with client.conn.cursor() as cur:
                for uid in targets:
                    if mode == "optimal":
                        op_start = common.time_ns()
                        cur.execute(sql_one, (uid,))
                        cur.fetchall()
                        op_end = common.time_ns()
                    else:
                        op_start = common.time_ns()
                        cur.execute(
                            "SELECT movie_id FROM ratings WHERE user_id=%s",
                            (uid,),
                        )
                        mids = [row[0] for row in cur.fetchall()]
                        if mids:
                            cur.execute(
                                "SELECT director_id FROM movies WHERE id = ANY(%s)",
                                (mids,),
                            )
                            dids = [row[0] for row in cur.fetchall() if row[0] is not None]
                            if dids:
                                cur.execute(
                                    "SELECT id, name FROM directors WHERE id = ANY(%s)",
                                    (dids,),
                                )
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
    parser.add_argument("--directors", type=int, default=20)
    parser.add_argument("--ratings-per-user", type=int, default=5)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=4301)
    parser.add_argument(
        "--pg-dsn", default="postgresql://bench:bench@127.0.0.1:5433/bench"
    )
    parser.add_argument("--pg-container", default="rhypedb-bench-postgres")
    parser.add_argument("--server-pattern", default="bench-rhypedb-data")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/scenario_08.json"),
    )
    args = parser.parse_args()

    impls = common.expand_impls(args.impl)
    ds = data.generate(
        n_users=args.users,
        n_movies=args.movies,
        ratings_per_user=args.ratings_per_user,
        friends_per_user=0,
    )
    directors = generate_directors(args.directors)
    print(f"Dataset: {ds.summary()}, {len(directors)} directors")

    rhypedb_uids = []
    pg_uids = []
    if "tcp" in impls or "http" in impls:
        print("Setup (rhypedb): inserting users + directors + movies + ratings...")
        t0 = time.time()
        rhypedb_uids = setup_rhypedb_graph(
            ds, directors, args.tcp_host, args.tcp_port
        )
        print(f"  rhypedb setup done in {time.time() - t0:.1f}s")
    if any(i.startswith("pg-") for i in impls):
        print("Setup (PG): inserting via COPY...")
        t0 = time.time()
        pg_uids = setup_pg_graph(ds, directors, args.pg_dsn)
        print(f"  PG setup done in {time.time() - t0:.1f}s")

    print(f"3-hop traversal: {args.queries} queries × {args.iterations} iterations")
    print(f"Impls: {impls}")
    print()

    results = []
    if "tcp" in impls:
        client = RhypedbTcpClient(args.tcp_host, args.tcp_port)
        client.connect()
        try:
            results.append(
                _run_rhypedb(
                    client,
                    rhypedb_uids,
                    args.queries,
                    args.iterations,
                    args.server_pattern,
                )
            )
        finally:
            client.close()
    if "pg-idiomatic" in impls:
        results.append(
            _run_pg(
                pg_uids,
                "idiomatic",
                args.queries,
                args.iterations,
                args.pg_dsn,
                args.pg_container,
            )
        )
    if "pg-optimal" in impls:
        results.append(
            _run_pg(
                pg_uids,
                "optimal",
                args.queries,
                args.iterations,
                args.pg_dsn,
                args.pg_container,
            )
        )

    common.print_summary(results)
    common.write_results(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
