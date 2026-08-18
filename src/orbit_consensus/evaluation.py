from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .schema import Grid, RankedCandidate, TaskOrbit, grid_key


def exact_match(left: Grid, right: Grid) -> bool:
    return grid_key(left) == grid_key(right)


def correct_rank(ranking: tuple[RankedCandidate, ...], target: Grid) -> int | None:
    target_key = grid_key(target)
    for rank, candidate in enumerate(ranking, start=1):
        if grid_key(candidate.grid) == target_key:
            return rank
    return None


def orbit_covers(orbit: TaskOrbit) -> bool:
    if orbit.target is None:
        raise ValueError(f"task {orbit.task_id} has no target")
    target_key = grid_key(orbit.target)
    return any(grid_key(candidate.grid) == target_key for candidate in orbit.candidates)


@dataclass(frozen=True)
class SelectionMetrics:
    rank1: float
    top2: float
    coverage: float
    ranked_coverage: float
    shape_selection_loss: float
    mean_reciprocal_rank_covered: float
    median_rank_covered: float
    covered_task_count: int


def selection_metrics(
    orbits: Iterable[TaskOrbit], rankings: Iterable[tuple[RankedCandidate, ...]]
) -> SelectionMetrics:
    """Evaluate every selector against the same orbit-level covered subset."""
    tasks = list(orbits)
    ranked = list(rankings)
    if len(tasks) != len(ranked) or not tasks:
        raise ValueError("orbits and rankings must be non-empty and have equal length")

    orbit_covered: list[bool] = []
    selector_ranks: list[int | None] = []
    for orbit, ranking in zip(tasks, ranked):
        if orbit.target is None:
            raise ValueError(f"task {orbit.task_id} has no target")
        orbit_covered.append(orbit_covers(orbit))
        selector_ranks.append(correct_rank(ranking, orbit.target))

    covered_indices = [index for index, covered in enumerate(orbit_covered) if covered]
    reciprocal_ranks = [
        0.0 if selector_ranks[index] is None else 1.0 / selector_ranks[index]
        for index in covered_indices
    ]
    covered_ranks = [
        float("inf") if selector_ranks[index] is None else float(selector_ranks[index])
        for index in covered_indices
    ]
    coverage = float(np.mean(orbit_covered))
    ranked_coverage = float(np.mean([rank is not None for rank in selector_ranks]))
    return SelectionMetrics(
        rank1=float(np.mean([rank == 1 for rank in selector_ranks])),
        top2=float(np.mean([rank is not None and rank <= 2 for rank in selector_ranks])),
        coverage=coverage,
        ranked_coverage=ranked_coverage,
        shape_selection_loss=coverage - ranked_coverage,
        mean_reciprocal_rank_covered=(
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
        ),
        median_rank_covered=(
            float(np.median(covered_ranks)) if covered_ranks else float("nan")
        ),
        covered_task_count=len(covered_indices),
    )


def paired_bootstrap_difference(
    method: NDArray[np.floating],
    baseline: NDArray[np.floating],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    method = np.asarray(method, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if method.shape != baseline.shape or method.ndim != 1 or not len(method):
        raise ValueError("paired inputs must be non-empty one-dimensional arrays of equal shape")
    differences = method - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        task_indices = rng.integers(0, len(differences), size=len(differences))
        samples[index] = np.mean(differences[task_indices])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(np.mean(differences)), float(low), float(high)
