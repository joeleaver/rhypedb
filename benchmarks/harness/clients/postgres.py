"""Postgres client wrapper used by every Suite 1 PG scenario.

Thin layer over psycopg v3. The point isn't to abstract psycopg away; it's to
give the scenarios a single place to:

  * manage the connection (autocommit on/off per call site)
  * register the simple-query and the prepared-statement code paths so the
    scenarios can opt in
  * sample the Postgres server PID for memory tracking (we run Postgres in a
    container, so we resolve the PID inside that container via `docker top`)

The harness identifies "the Postgres process" as PID 1 inside the
`rhypedb-bench-postgres` container's PID namespace; from the host we look up
the corresponding host PID via `docker top`.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import psycopg
from psycopg import sql


DEFAULT_DSN = "postgresql://bench:bench@127.0.0.1:5433/bench"


class PgClient:
    """One-connection helper. Not threadsafe; benchmarks are single-threaded."""

    def __init__(self, dsn: str = DEFAULT_DSN, autocommit: bool = True) -> None:
        self.dsn = dsn
        self.conn = psycopg.connect(dsn, autocommit=autocommit)
        # prepare_threshold=1 makes every parameterized query prepared on its
        # second call — this is what "idiomatic" psycopg apps get for free.
        self.conn.prepare_threshold = 1

    def close(self) -> None:
        if self.conn is not None and not self.conn.closed:
            self.conn.close()
            self.conn = None

    def reset(self) -> None:
        """Wipe all rows + reset sequences. Used between iterations."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT truncate_all()")

    def execute(self, query: str, params: Sequence | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(query, params)

    def fetchone(self, query: str, params: Sequence | None = None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def fetchall(self, query: str, params: Sequence | None = None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def executemany(self, query: str, rows: Iterable[Sequence]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(query, rows)

    def copy_rows(self, copy_sql: str, rows: Iterable[Sequence]) -> None:
        """COPY ... FROM STDIN, used by the 'optimal' bulk-insert path."""
        with self.conn.cursor() as cur, cur.copy(copy_sql) as cp:
            for row in rows:
                cp.write_row(row)

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Cursor]:
        """Run a block inside one transaction with a single cursor.

        Equivalent to BEGIN / ... / COMMIT, with rollback on exception. Used by
        the 'optimal' implementations that batch several statements into one
        roundtrip-bounded unit.
        """
        prior = self.conn.autocommit
        self.conn.autocommit = False
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()
            self.conn.autocommit = prior


def find_pg_host_pid(
    container: str = "rhypedb-bench-postgres",
) -> int:
    """Return the host-side PID of the Postgres `postmaster` process for the
    bench container. Returns 0 if docker isn't available or the container isn't
    running.

    `docker top` lists processes inside the container with their host-namespace
    PID in the first column.  We look for the `postgres` command without a
    parent that's also `postgres` (i.e. the postmaster, not a worker).
    """
    try:
        out = subprocess.run(
            ["docker", "top", container, "-o", "pid,ppid,cmd"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0

    lines = out.strip().splitlines()
    if len(lines) < 2:
        return 0
    # Skip header. Find the row whose ppid is not in the PID column (i.e. it's
    # the root of the postgres tree inside the container).
    pid_to_row = {}
    for line in lines[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        pid_to_row[pid] = (ppid, cmd)
    for pid, (ppid, cmd) in pid_to_row.items():
        if "postgres" in cmd and ppid not in pid_to_row:
            return int(pid)
    return 0


def reset_db(dsn: str = DEFAULT_DSN) -> None:
    """Stand-alone helper used by harness `__main__` blocks before setup."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT truncate_all()")


# Reusable SQL builders for scenarios. Putting them here keeps each scenario
# file focused on the workload shape, not the column list.

SQL_INSERT_USER = sql.SQL(
    "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id"
)
SQL_INSERT_MOVIE = sql.SQL(
    "INSERT INTO movies (title, year) VALUES (%s, %s) RETURNING id"
)
SQL_INSERT_RATING = sql.SQL(
    "INSERT INTO ratings (user_id, movie_id, stars) VALUES (%s, %s, %s) RETURNING id"
)
