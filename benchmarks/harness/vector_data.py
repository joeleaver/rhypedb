"""Vector dataset loader for Suite 2 (rhypedb vs Postgres+pgvector).

Uses the ann-benchmarks HDF5 datasets — the de-facto standard for ANN
benchmarking. Each file bundles everything we need:

  * ``train``     (N, D) float32 — the base vectors to index
  * ``test``      (Q, D) float32 — the query vectors
  * ``neighbors`` (Q, K) int     — ground-truth top-K base indices per query
  * ``distances`` (Q, K) float32 — their distances (unused here)
  * attr ``distance`` — "angular" (cosine) or "euclidean"

We default to an *angular* (cosine) dataset because rhypedb's HNSW index is
hardcoded to the cosine metric, so the provided ground truth lines up with what
both engines actually optimize. ``glove-100-angular`` is a good default;
``glove-25-angular`` is smaller for quick iteration.

Datasets are cached under ``benchmarks/.vector-data/``. With ``--subset N`` the
harness takes the first N train vectors and RECOMPUTES ground truth by brute
force (the bundled neighbors index the full train set, so they're invalid for a
subset).
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANN_BASE_URL = "https://ann-benchmarks.com"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".vector-data"

# Synthetic dataset defaults (for the "synthetic-<dim>-angular" datasets, which
# don't exist in ann-benchmarks at 384-dim — the production embedding size).
SYNTH_BASE_N = 1_000_000   # full base set; --subset slices the first N
SYNTH_QUERIES = 1_000
SYNTH_SEED = 1234

# Datasets whose ground truth is computed under angular (cosine) distance —
# the metric rhypedb's index uses. Dimensions noted for convenience.
ANGULAR_DATASETS = {
    "glove-25-angular": 25,
    "glove-50-angular": 50,
    "glove-100-angular": 100,
    "glove-200-angular": 200,
    "nytimes-256-angular": 256,
}


@dataclass
class VectorDataset:
    name: str
    train: np.ndarray  # (N, D) float32
    test: np.ndarray  # (Q, D) float32
    neighbors: np.ndarray  # (Q, K) int — ground-truth top-K train indices
    metric: str  # "angular" | "euclidean"

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    def summary(self) -> str:
        return (
            f"{self.name}: {self.train.shape[0]} train x {self.dim}d, "
            f"{self.test.shape[0]} queries, gt top-{self.neighbors.shape[1]}, "
            f"metric={self.metric}"
        )


def _download(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.hdf5"
    if path.exists():
        return path
    url = f"{ANN_BASE_URL}/{name}.hdf5"
    print(f"Downloading {url} -> {path} ...", file=sys.stderr)
    tmp = path.with_suffix(".hdf5.part")

    last_pct = [-1]

    def _progress(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, 100 * block * block_size // total)
            if pct != last_pct[0] and pct % 10 == 0:
                last_pct[0] = pct
                print(f"  {pct}%", file=sys.stderr, flush=True)

    # ann-benchmarks.com 403s the default "Python-urllib" User-Agent; a normal
    # browser-ish UA is served fine.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (rhypedb-bench)")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, tmp, _progress)
    print("", file=sys.stderr)
    tmp.rename(path)
    return path


def brute_force_neighbors(
    train: np.ndarray, test: np.ndarray, k: int, metric: str, base_chunk: int = 100_000
) -> np.ndarray:
    """Exact top-k ground truth, ORDERED nearest-first. Cosine = normalize then
    max dot product; euclidean = min squared L2. Returns (Q, k) int indices into
    ``train``.

    Chunked over the base set so the (Q, N) score matrix is never materialized
    whole — at 1M base × 1k queries that would be a 4 GB f32 allocation. We keep a
    running top-k per query, merging each base chunk's local top-k into it; the
    final sort within the surviving k gives a true ``[:j]`` = top-j slice."""
    if metric == "angular":
        tn = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-30)
        qn = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-30)
    else:
        t_sq = np.einsum("ij,ij->i", train, train).astype(np.float32)

    q = test.shape[0]
    n = train.shape[0]
    rows = np.arange(q)[:, None]
    best_idx: np.ndarray | None = None   # (q, <=k) global indices, smaller neg = closer
    best_neg: np.ndarray | None = None

    for s in range(0, n, base_chunk):
        e = min(s + base_chunk, n)
        if metric == "angular":
            neg = -(qn @ tn[s:e].T)  # (q, e-s)
        else:
            neg = t_sq[s:e][None, :] - 2.0 * (test @ train[s:e].T)
        kk = min(k, e - s)
        part = np.argpartition(neg, kk - 1, axis=1)[:, :kk]  # local top-kk set
        cand_idx = part + s
        cand_neg = neg[rows, part]
        if best_idx is None:
            best_idx, best_neg = cand_idx, cand_neg
        else:
            best_idx = np.concatenate([best_idx, cand_idx], axis=1)
            best_neg = np.concatenate([best_neg, cand_neg], axis=1)
            kk2 = min(k, best_idx.shape[1])
            part2 = np.argpartition(best_neg, kk2 - 1, axis=1)[:, :kk2]
            best_idx = best_idx[rows, part2]
            best_neg = best_neg[rows, part2]

    order = np.argsort(best_neg, axis=1)  # sort surviving k by true distance
    return best_idx[rows, order]


def _generate_synthetic(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clustered, unit-normalized (angular) synthetic base + query set.

    Real embedding spaces have local neighbor structure (not uniform-random, which
    in high dim is ~all-equidistant and makes recall meaningless). We model that as
    a mixture of Gaussians on the unit sphere: each vector is a random cluster
    centroid plus small isotropic noise, then L2-normalized. The noise scale is
    chosen so within-cluster angular spread gives a non-trivial but learnable
    top-k — comparable difficulty to real 384-d embeddings, and identical for both
    engines so the comparison stays fair regardless of realism."""
    rng = np.random.default_rng(SYNTH_SEED)
    n = SYNTH_BASE_N
    n_clusters = max(64, n // 200)  # ~200 vectors per cluster
    # noise_norm ≈ sigma·√dim ≈ 0.59 at 384d: distinct clusters (median cosine ~0)
    # with non-trivial intra-cluster ranking — meaningful, non-trivial top-k.
    sigma = np.float32(0.03)

    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-30

    def _sample(count: int) -> np.ndarray:
        out = np.empty((count, dim), dtype=np.float32)
        for s in range(0, count, 100_000):
            e = min(s + 100_000, count)
            a = rng.integers(0, n_clusters, size=e - s)
            chunk = centers[a] + rng.standard_normal((e - s, dim)).astype(np.float32) * sigma
            chunk /= np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-30
            out[s:e] = chunk
        return out

    print(f"generating synthetic {n}x{dim} base + {SYNTH_QUERIES} queries "
          f"({n_clusters} clusters, sigma={sigma})...", file=sys.stderr)
    train = _sample(n)
    test = _sample(SYNTH_QUERIES)
    return train, test


def _load_or_generate_synthetic(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (train_full, test, metric) for a 'synthetic-<dim>-angular' name,
    generating + caching the base set under .vector-data/ on first use."""
    parts = name.split("-")
    if len(parts) != 3 or parts[0] != "synthetic" or parts[2] != "angular":
        raise ValueError(f"synthetic dataset name must be 'synthetic-<dim>-angular', got {name!r}")
    dim = int(parts[1])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return np.ascontiguousarray(z["train"]), np.ascontiguousarray(z["test"]), "angular"
    train, test = _generate_synthetic(dim)
    tmp = cache.with_suffix(".npz.part")
    # Pass a file handle so np.savez doesn't append a second ".npz" to the name.
    with open(tmp, "wb") as fh:
        np.savez(fh, train=train, test=test)
    tmp.rename(cache)
    return train, test, "angular"


def load(name: str, subset: int | None = None, gt_k: int = 100) -> VectorDataset:
    if name.startswith("synthetic-"):
        train, test, metric = _load_or_generate_synthetic(name)
        bundled_neighbors = None
    else:
        import h5py

        path = _download(name)
        with h5py.File(path, "r") as f:
            train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
            test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
            bundled_neighbors = np.asarray(f["neighbors"][:], dtype=np.int64)
            metric = f.attrs.get("distance", "angular")
            if isinstance(metric, bytes):
                metric = metric.decode()

    if subset is not None and subset < train.shape[0]:
        train = np.ascontiguousarray(train[:subset])
        bundled_neighbors = None  # bundled GT indexes the full base — invalid now

    if bundled_neighbors is not None:
        neighbors = bundled_neighbors
    else:
        # Subset, or synthetic with no bundled GT: compute exact top-gt_k.
        print(
            f"computing brute-force top-{gt_k} ground truth over {train.shape[0]} base vectors...",
            file=sys.stderr,
        )
        neighbors = brute_force_neighbors(train, test, gt_k, metric)

    return VectorDataset(name=name, train=train, test=test, neighbors=neighbors, metric=metric)


def recall_at_k(retrieved_ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Mean recall@k: fraction of each query's true top-k that the index
    returned. ``retrieved_ids`` and ``ground_truth`` are (Q, >=k) arrays of
    train-vector ids; ids must be the same space (we use the train index as id).
    """
    q = retrieved_ids.shape[0]
    hits = 0
    for i in range(q):
        got = set(int(x) for x in retrieved_ids[i, :k])
        truth = set(int(x) for x in ground_truth[i, :k])
        hits += len(got & truth)
    return hits / (q * k)
