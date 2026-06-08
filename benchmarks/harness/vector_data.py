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
    train: np.ndarray, test: np.ndarray, k: int, metric: str
) -> np.ndarray:
    """Exact top-k ground truth. Cosine = normalize then argmax dot product;
    euclidean = argmin squared L2. Returns (Q, k) int indices into ``train``."""
    if metric == "angular":
        tn = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-30)
        qn = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-30)
        sims = qn @ tn.T  # (Q, N) cosine similarity; larger = closer
        return np.argpartition(-sims, k, axis=1)[:, :k]
    # euclidean
    # ||q - t||^2 = ||q||^2 - 2 q.t + ||t||^2; ||q||^2 is constant per row.
    t_sq = np.einsum("ij,ij->i", train, train)  # (N,)
    dots = test @ train.T  # (Q, N)
    d2 = t_sq[None, :] - 2.0 * dots  # drop the per-row ||q||^2 constant
    return np.argpartition(d2, k, axis=1)[:, :k]


def load(name: str, subset: int | None = None, gt_k: int = 100) -> VectorDataset:
    import h5py

    path = _download(name)
    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:], dtype=np.int64)
        metric = f.attrs.get("distance", "angular")
        if isinstance(metric, bytes):
            metric = metric.decode()

    if subset is not None and subset < train.shape[0]:
        train = np.ascontiguousarray(train[:subset])
        # Bundled neighbors index the full train set — invalid for a subset.
        # Recompute exact ground truth against the reduced base.
        print(
            f"subset={subset}: recomputing brute-force top-{gt_k} ground truth...",
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
