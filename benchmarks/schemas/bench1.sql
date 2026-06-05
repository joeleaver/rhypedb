-- Suite 1 schema: mirrors benchmarks/schemas/bench1.sdl as closely as Postgres
-- allows. Mounted into the bench container at /docker-entrypoint-initdb.d/ so
-- it runs once on first container start.
--
-- Design notes:
--   * Surrogate BIGINT PKs match rhypedb's u64 object IDs. We use BIGSERIAL so
--     each scenario can insert without computing IDs client-side.
--   * email is UNIQUE — exercises Suite 1.7.
--   * ratings -> users / movies use ON DELETE CASCADE so 1.6 works.
--   * friendships is symmetric in the rhypedb side via @on_delete(remove);
--     here we model an undirected edge as a single row with a < b plus
--     ON DELETE CASCADE on both endpoints so deleting a user removes the
--     friendship rows that reference it.

CREATE TABLE IF NOT EXISTS users (
    id     BIGSERIAL PRIMARY KEY,
    name   TEXT NOT NULL,
    email  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS directors (
    id   BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS movies (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    year        INTEGER NOT NULL,
    director_id BIGINT REFERENCES directors(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS movies_director_idx ON movies(director_id);

CREATE TABLE IF NOT EXISTS ratings (
    id       BIGSERIAL PRIMARY KEY,
    user_id  BIGINT NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    movie_id BIGINT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    stars    REAL   NOT NULL
);

CREATE INDEX IF NOT EXISTS ratings_user_idx  ON ratings(user_id);
CREATE INDEX IF NOT EXISTS ratings_movie_idx ON ratings(movie_id);

CREATE INDEX IF NOT EXISTS movies_year_idx ON movies(year);

CREATE TABLE IF NOT EXISTS friendships (
    a_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    b_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (a_id, b_id),
    CHECK (a_id < b_id)
);

CREATE INDEX IF NOT EXISTS friendships_b_idx ON friendships(b_id);

-- truncate_all() is invoked by the harness between iterations to reset state
-- without paying the cost of recreating tables. RESTART IDENTITY keeps IDs
-- starting at 1 each iteration so test code can refer to them deterministically.
CREATE OR REPLACE FUNCTION truncate_all() RETURNS VOID AS $$
BEGIN
    TRUNCATE TABLE ratings, friendships, movies, directors, users RESTART IDENTITY CASCADE;
END;
$$ LANGUAGE plpgsql;
