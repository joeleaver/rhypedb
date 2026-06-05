"""Synthetic dataset generation for the relational/graph benchmark.

Reproducible (seeded). Generates Users, Movies, Ratings, and friend edges.
"""

import random
from dataclasses import dataclass


@dataclass
class User:
    idx: int           # 0-based index, used for ID stitching
    name: str
    email: str


@dataclass
class Movie:
    idx: int
    title: str
    year: int


@dataclass
class Rating:
    user_idx: int
    movie_idx: int
    stars: float


@dataclass
class Friendship:
    a_idx: int
    b_idx: int


@dataclass
class Dataset:
    users: list
    movies: list
    ratings: list
    friendships: list

    def summary(self) -> str:
        return (
            f"{len(self.users)} users, {len(self.movies)} movies, "
            f"{len(self.ratings)} ratings, {len(self.friendships)} friendships"
        )


FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Liam", "Mia", "Noah", "Olivia", "Peggy",
    "Quincy", "Rita", "Sam", "Trent", "Uma", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zach",
]

LAST_NAMES = [
    "Smith", "Jones", "Brown", "Davis", "Wilson", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia",
    "Martinez", "Robinson", "Clark", "Lewis", "Lee", "Walker", "Hall",
]

MOVIE_TITLE_WORDS = [
    "Shadow", "Edge", "Last", "First", "Silent", "Quiet", "Dark", "Bright",
    "Forgotten", "Lost", "Hidden", "Secret", "Final", "Eternal", "Broken",
    "Crystal", "Iron", "Gold", "Silver", "Wild", "Ancient", "Distant",
]

MOVIE_TITLE_NOUNS = [
    "Sky", "Forest", "City", "River", "Mountain", "Storm", "Dream", "Light",
    "Garden", "Path", "Bridge", "Tower", "Kingdom", "Echo", "Hour", "Truth",
    "Promise", "Memory", "Voyage", "Symphony", "Mirror", "Throne",
]


def generate(
    n_users: int,
    n_movies: int,
    ratings_per_user: int,
    friends_per_user: int,
    seed: int = 42,
) -> Dataset:
    """Generate a reproducible synthetic dataset."""
    rng = random.Random(seed)

    users = []
    for i in range(n_users):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        users.append(
            User(
                idx=i,
                name=f"{first} {last}",
                email=f"user{i}@bench.test",
            )
        )

    movies = []
    for i in range(n_movies):
        adj = rng.choice(MOVIE_TITLE_WORDS)
        noun = rng.choice(MOVIE_TITLE_NOUNS)
        suffix = rng.choice(["", " Returns", " Begins", f" {rng.randint(2, 9)}"])
        movies.append(
            Movie(
                idx=i,
                title=f"{adj} {noun}{suffix}",
                year=rng.randint(1950, 2024),
            )
        )

    # Each user rates approximately `ratings_per_user` movies.
    ratings = []
    for u in users:
        sampled = rng.sample(range(n_movies), min(ratings_per_user, n_movies))
        for movie_idx in sampled:
            ratings.append(
                Rating(
                    user_idx=u.idx,
                    movie_idx=movie_idx,
                    stars=round(rng.uniform(1.0, 5.0), 1),
                )
            )

    # Friendship edges (undirected; we store both directions when materializing).
    friendships = []
    seen_pairs = set()
    for u in users:
        targets = rng.sample(
            range(n_users), min(friends_per_user + 1, n_users)
        )
        for t in targets:
            if t == u.idx:
                continue
            pair = (min(u.idx, t), max(u.idx, t))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            friendships.append(Friendship(a_idx=u.idx, b_idx=t))

    return Dataset(
        users=users,
        movies=movies,
        ratings=ratings,
        friendships=friendships,
    )


if __name__ == "__main__":
    ds = generate(n_users=1000, n_movies=100, ratings_per_user=10, friends_per_user=5)
    print(ds.summary())
