"""Shared utilities: latency sampling, memory sampling, result reporting."""

import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def time_ns() -> int:
    """Monotonic time in nanoseconds."""
    return time.perf_counter_ns()


@dataclass
class LatencyStats:
    """Latency distribution summary."""
    n: int
    min_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    mean_us: float
    total_ms: float


def latency_stats(samples_ns: list) -> LatencyStats:
    if not samples_ns:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)
    samples_us = sorted(s / 1000 for s in samples_ns)
    n = len(samples_us)
    return LatencyStats(
        n=n,
        min_us=samples_us[0],
        p50_us=samples_us[n // 2],
        p95_us=samples_us[min(int(n * 0.95), n - 1)],
        p99_us=samples_us[min(int(n * 0.99), n - 1)],
        max_us=samples_us[-1],
        mean_us=statistics.mean(samples_us),
        total_ms=sum(samples_us) / 1000,
    )


@dataclass
class MemorySample:
    rss_mb: float
    timestamp: float


def rss_mb(pid: int) -> float:
    """Return resident set size in MB for the given PID, via /proc."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except FileNotFoundError:
        return 0.0
    return 0.0


def find_pid(process_name: str) -> int:
    """Find the PID of a running process by name (first match)."""
    result = subprocess.run(
        ["pgrep", "-f", process_name],
        capture_output=True,
        text=True,
    )
    pids = result.stdout.strip().splitlines()
    return int(pids[0]) if pids else 0


def disk_usage_mb(path: Path) -> float:
    """Return total size of files under `path` in MB."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


@dataclass
class ScenarioResult:
    """Result of running one scenario against one implementation."""
    scenario: str
    implementation: str
    iterations: int
    # Per-iteration wall-clock for the whole task.
    iteration_ms: list = field(default_factory=list)
    # Optional per-operation latency samples (e.g., per-insert).
    operation_latency_ns: list = field(default_factory=list)
    # Memory snapshots.
    rss_cold_mb: float = 0.0
    rss_post_load_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_steady_mb: float = 0.0
    # Disk.
    disk_mb: float = 0.0
    # Free-form metadata (dataset size etc).
    metadata: dict = field(default_factory=dict)

    def iteration_stats(self) -> dict:
        if not self.iteration_ms:
            return {}
        sorted_ms = sorted(self.iteration_ms)
        n = len(sorted_ms)
        return {
            "min_ms": sorted_ms[0],
            "p50_ms": sorted_ms[n // 2],
            "p95_ms": sorted_ms[min(int(n * 0.95), n - 1)],
            "p99_ms": sorted_ms[min(int(n * 0.99), n - 1)],
            "max_ms": sorted_ms[-1],
            "mean_ms": statistics.mean(sorted_ms),
        }

    def operation_stats(self) -> dict:
        if not self.operation_latency_ns:
            return {}
        return asdict(latency_stats(self.operation_latency_ns))

    def to_json(self) -> dict:
        return {
            "scenario": self.scenario,
            "implementation": self.implementation,
            "iterations": self.iterations,
            "iteration_stats": self.iteration_stats(),
            "operation_stats": self.operation_stats(),
            "rss_cold_mb": self.rss_cold_mb,
            "rss_post_load_mb": self.rss_post_load_mb,
            "rss_peak_mb": self.rss_peak_mb,
            "rss_steady_mb": self.rss_steady_mb,
            "disk_mb": self.disk_mb,
            "metadata": self.metadata,
        }


def write_results(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_json() for r in results], f, indent=2)


IMPL_ALIASES = {
    "both":     ["http", "tcp"],
    "rhypedb":  ["http", "tcp"],
    "pg":       ["pg-idiomatic", "pg-optimal"],
    "all":      ["tcp", "tcp-batch", "pg-idiomatic", "pg-optimal"],
}

KNOWN_IMPLS = {"http", "tcp", "tcp-batch", "pg-idiomatic", "pg-optimal"}


def expand_impls(spec: str) -> list:
    """Parse a comma-separated impl string into a deduped, ordered list.

    Aliases: 'both'/'rhypedb' → http,tcp; 'pg' → pg-idiomatic,pg-optimal;
    'all' → tcp,pg-idiomatic,pg-optimal (the publish-ready comparison set).
    """
    seen = set()
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        names = IMPL_ALIASES.get(token, [token])
        for name in names:
            if name not in KNOWN_IMPLS:
                raise ValueError(
                    f"unknown impl '{name}' (expected one of {sorted(KNOWN_IMPLS)} "
                    f"or alias {sorted(IMPL_ALIASES)})"
                )
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def print_summary(results: list) -> None:
    """Print a compact comparison table to stdout."""
    print()
    print(f"{'Scenario':<28} {'Impl':<14} {'p50 ms':>10} {'p95 ms':>10} {'mean op µs':>14} {'RSS MB':>10}")
    print("-" * 90)
    for r in results:
        stats = r.iteration_stats()
        op = r.operation_stats()
        p50 = f"{stats.get('p50_ms', 0):.2f}" if stats else "-"
        p95 = f"{stats.get('p95_ms', 0):.2f}" if stats else "-"
        mean_op = f"{op.get('mean_us', 0):.1f}" if op else "-"
        rss = f"{r.rss_steady_mb:.1f}" if r.rss_steady_mb else "-"
        print(f"{r.scenario:<28} {r.implementation:<14} {p50:>10} {p95:>10} {mean_op:>14} {rss:>10}")
    print()
