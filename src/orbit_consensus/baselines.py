from __future__ import annotations

from collections import defaultdict

from .schema import Grid, GridKey, RankedCandidate, TaskOrbit, grid_key
from .similarity import unique_grids


def mean_q_by_grid(orbit: TaskOrbit, allowed_indices: set[int] | None = None) -> dict[GridKey, float]:
    q_values: dict[GridKey, list[float]] = defaultdict(list)
    for index, candidate in enumerate(orbit.candidates):
        if allowed_indices is None or index in allowed_indices:
            q_values[grid_key(candidate.grid)].append(candidate.q_value)
    return {key: sum(values) / len(values) for key, values in q_values.items()}


def rank_candidates(
    representatives: dict[GridKey, Grid],
    scores: dict[GridKey, float],
    counts: dict[GridKey, int],
    mean_q: dict[GridKey, float],
) -> tuple[RankedCandidate, ...]:
    """Match upstream: primary score, then mean Q, then a deterministic final key."""
    ordered = sorted(representatives, key=lambda key: (-scores[key], -mean_q[key], key))
    return tuple(
        RankedCandidate(
            representatives[key],
            float(scores[key]),
            int(counts[key]),
            float(mean_q[key]),
            is_invalid=key[:2] == (0, 0),
        )
        for key in ordered
    )


def canonical_prediction(orbit: TaskOrbit) -> Grid:
    identity = [candidate.grid for candidate in orbit.candidates if candidate.is_identity]
    if len(identity) != 1:
        raise ValueError(f"expected exactly one identity candidate, found {len(identity)}")
    return identity[0]


def majority_vote(orbit: TaskOrbit) -> tuple[RankedCandidate, ...]:
    """B1: exact-grid count with upstream's mean-Q tie break."""
    representatives, counts = unique_grids(candidate.grid for candidate in orbit.candidates)
    scores = {key: float(counts[key]) for key in representatives}
    return rank_candidates(representatives, scores, counts, mean_q_by_grid(orbit))


def contains_exact(
    ranked: tuple[RankedCandidate, ...], target: Grid, top_k: int | None = None
) -> bool:
    target_key = grid_key(target)
    selected = ranked if top_k is None else ranked[:top_k]
    return any(grid_key(candidate.grid) == target_key for candidate in selected)
