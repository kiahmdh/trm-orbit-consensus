from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .baselines import mean_q_by_grid, rank_candidates
from .schema import GridKey, RankedCandidate, TaskOrbit, grid_key
from .shape_screening import ShapeScreen, infer_shape_screen
from .similarity import (
    CentralityDefinition,
    SimilarityWorkspace,
    build_similarity_workspace,
    centrality_from_workspace,
    unique_grids,
)

MarginalSupport = Literal["emitted", "distinct_uniform"]
_CENTRALITY_DEFINITIONS: tuple[CentralityDefinition, ...] = (
    "distinct",
    "multiset",
    "non_identical_support",
)


@dataclass(frozen=True)
class M1Precomputation:
    task_id: str
    emission_keys: tuple[GridKey, ...]
    similarity: SimilarityWorkspace
    centrality: dict[CentralityDefinition, dict[GridKey, float]]


@dataclass(frozen=True)
class M1Result:
    ranking: tuple[RankedCandidate, ...]
    emission_weights: NDArray[np.float64]
    centrality: dict[GridKey, float]


@dataclass(frozen=True)
class M2Result:
    ranking: tuple[RankedCandidate, ...]
    modal_shape: tuple[int, int]
    shape_screen: ShapeScreen | None
    used_shape_fallback: bool


def prepare_m1(orbit: TaskOrbit) -> M1Precomputation:
    """Pay the pairwise-similarity cost exactly once for a task orbit."""
    emission_keys = tuple(grid_key(candidate.grid) for candidate in orbit.candidates)
    workspace = build_similarity_workspace(candidate.grid for candidate in orbit.candidates)
    centrality = {
        definition: centrality_from_workspace(workspace, definition)
        for definition in _CENTRALITY_DEFINITIONS
    }
    return M1Precomputation(orbit.task_id, emission_keys, workspace, centrality)


def _validate_precomputation(orbit: TaskOrbit, prepared: M1Precomputation) -> None:
    keys = tuple(grid_key(candidate.grid) for candidate in orbit.candidates)
    if prepared.task_id != orbit.task_id or prepared.emission_keys != keys:
        raise ValueError("M1 precomputation belongs to a different task orbit")


def m1_orbit_weighting(
    orbit: TaskOrbit,
    beta: float,
    definition: CentralityDefinition = "distinct",
    *,
    prepared: M1Precomputation | None = None,
) -> M1Result:
    """M1 using reusable task-level centralities; beta changes are O(k)."""
    prepared = prepare_m1(orbit) if prepared is None else prepared
    _validate_precomputation(orbit, prepared)
    centrality = prepared.centrality[definition]
    logits = np.asarray(
        [beta * centrality[key] for key in prepared.emission_keys], dtype=np.float64
    )
    logits -= np.max(logits)
    emission_weights = np.exp(logits)
    emission_weights /= np.sum(emission_weights)

    representatives = prepared.similarity.representatives
    counts = prepared.similarity.counts
    scores = {key: 0.0 for key in representatives}
    for key, weight in zip(prepared.emission_keys, emission_weights):
        scores[key] += float(weight)
    ranking = rank_candidates(
        representatives,
        scores,
        counts,
        mean_q_by_grid(orbit),
    )
    return M1Result(ranking, emission_weights, centrality)


def m1_beta_sweep(
    orbit: TaskOrbit,
    betas: tuple[float, ...],
    definition: CentralityDefinition,
    *,
    prepared: M1Precomputation | None = None,
) -> dict[float, M1Result]:
    prepared = prepare_m1(orbit) if prepared is None else prepared
    return {
        beta: m1_orbit_weighting(orbit, beta, definition, prepared=prepared)
        for beta in betas
    }


def _shape_filter(
    orbit: TaskOrbit, use_shape_screening: bool
) -> tuple[list[int], ShapeScreen | None, bool]:
    indices = list(range(len(orbit.candidates)))
    if not use_shape_screening or orbit.query_input is None:
        return indices, None, False
    screen = infer_shape_screen(orbit.support_pairs, orbit.query_input.shape)
    if not screen.allowed_shapes:
        return indices, screen, False
    filtered = [i for i in indices if orbit.candidates[i].grid.shape in screen.allowed_shapes]
    return (filtered, screen, False) if filtered else (indices, screen, True)


def m2_cell_marginal_score(
    orbit: TaskOrbit,
    interpolation: float,
    epsilon: float,
    *,
    emission_weights: NDArray[np.float64] | None = None,
    marginal_support: MarginalSupport = "emitted",
    use_shape_screening: bool = False,
) -> M2Result:
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be in [0, 1]")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")

    candidate_count = len(orbit.candidates)
    if emission_weights is None:
        weights = np.full(candidate_count, 1.0 / candidate_count, dtype=np.float64)
    else:
        weights = np.asarray(emission_weights, dtype=np.float64)
        if weights.shape != (candidate_count,) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError(
                "emission_weights must be non-negative with one value per candidate"
            )
        weights = weights / np.sum(weights)

    screened_indices, screen, used_fallback = _shape_filter(orbit, use_shape_screening)
    shape_mass: dict[tuple[int, int], float] = {}
    for index in screened_indices:
        shape = orbit.candidates[index].grid.shape
        shape_mass[shape] = shape_mass.get(shape, 0.0) + float(weights[index])
    modal_shape = min(shape_mass, key=lambda shape: (-shape_mass[shape], shape))
    active = [
        index
        for index in screened_indices
        if orbit.candidates[index].grid.shape == modal_shape
    ]

    height, width = modal_shape
    marginals = np.zeros((height, width, 10), dtype=np.float64)
    if marginal_support == "emitted":
        active_weights = weights[active] / np.sum(weights[active])
        marginal_items = [
            (orbit.candidates[index].grid, float(weight))
            for index, weight in zip(active, active_weights)
        ]
    elif marginal_support == "distinct_uniform":
        distinct = {
            grid_key(orbit.candidates[index].grid): orbit.candidates[index].grid
            for index in active
        }
        uniform = 1.0 / len(distinct)
        marginal_items = [(grid, uniform) for grid in distinct.values()]
    else:
        raise ValueError(f"unknown marginal_support: {marginal_support}")

    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    for grid, weight in marginal_items:
        marginals[rows, columns, grid] += weight

    active_grids = [orbit.candidates[index].grid for index in active]
    representatives, counts = unique_grids(active_grids)
    scores: dict[GridKey, float] = {}
    for key, grid in representatives.items():
        if modal_shape == (0, 0):
            cell_score = 0.0
        else:
            probabilities = marginals[rows, columns, grid]
            cell_score = float(
                np.log((1.0 - epsilon) * probabilities + epsilon / 10.0).sum()
            )
        scores[key] = (
            interpolation * cell_score
            + (1.0 - interpolation) * np.log(counts[key])
        )

    active_set = set(active)
    ranking = rank_candidates(
        representatives,
        scores,
        counts,
        mean_q_by_grid(orbit, active_set),
    )
    return M2Result(ranking, modal_shape, screen, used_fallback)


def two_attempt_policy(
    m2_ranking: tuple[RankedCandidate, ...],
    b1_ranking: tuple[RankedCandidate, ...],
) -> tuple[RankedCandidate, ...]:
    if not m2_ranking or not b1_ranking:
        raise ValueError("both rankings must be non-empty")
    first = m2_ranking[0]
    if grid_key(first.grid) != grid_key(b1_ranking[0].grid):
        return (first, b1_ranking[0])
    return (first,) if len(m2_ranking) == 1 else (first, m2_ranking[1])
