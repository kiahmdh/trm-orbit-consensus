from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .schema import Grid, GridKey, grid_key

CentralityDefinition = Literal["distinct", "multiset", "non_identical_support"]


def grid_similarity(left: Grid, right: Grid) -> float:
    """Cell agreement, with shape disagreement assigned zero similarity."""
    if left.shape != right.shape:
        return 0.0
    if left.shape == (0, 0):
        return 1.0
    return float(np.mean(left == right))


def unique_grids(grids: Iterable[Grid]) -> tuple[dict[GridKey, Grid], Counter[GridKey]]:
    representatives: dict[GridKey, Grid] = {}
    counts: Counter[GridKey] = Counter()
    for grid in grids:
        key = grid_key(grid)
        representatives.setdefault(key, grid)
        counts[key] += 1
    return representatives, counts


@dataclass(frozen=True)
class SimilarityWorkspace:
    """The O(m^2) task-level work shared by every definition and beta."""

    keys: tuple[GridKey, ...]
    representatives: dict[GridKey, Grid]
    counts: Counter[GridKey]
    matrix: NDArray[np.float32]


def build_similarity_workspace(grids: Iterable[Grid]) -> SimilarityWorkspace:
    representatives, counts = unique_grids(grids)
    keys = tuple(representatives)
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    by_shape: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(keys):
        by_shape.setdefault((key[0], key[1]), []).append(index)

    for indices in by_shape.values():
        if keys[indices[0]][:2] == (0, 0):
            matrix[np.ix_(indices, indices)] = 1.0
            continue
        stacked = np.stack(
            [representatives[keys[index]].reshape(-1) for index in indices]
        )
        block_size = 32
        for start in range(0, len(indices), block_size):
            stop = min(start + block_size, len(indices))
            similarities = np.mean(
                stacked[start:stop, None, :] == stacked[None, :, :],
                axis=-1,
                dtype=np.float32,
            )
            matrix[np.ix_(indices[start:stop], indices)] = similarities
    return SimilarityWorkspace(keys, representatives, counts, matrix)


def centrality_from_workspace(
    workspace: SimilarityWorkspace, definition: CentralityDefinition
) -> dict[GridKey, float]:
    keys = workspace.keys
    if not keys:
        return {}
    counts = np.asarray([workspace.counts[key] for key in keys], dtype=np.float64)
    matrix = workspace.matrix.astype(np.float64, copy=False)

    if definition == "distinct":
        if len(keys) == 1:
            values = np.zeros(1, dtype=np.float64)
        else:
            values = (np.sum(matrix, axis=1) - 1.0) / (len(keys) - 1)
    elif definition == "multiset":
        values = matrix @ counts / np.sum(counts)
    elif definition == "non_identical_support":
        denominators = np.sum(counts) - counts
        numerators = matrix @ counts - counts
        values = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators > 0,
        )
    else:
        raise ValueError(f"unknown centrality definition: {definition}")
    return {key: float(values[index]) for index, key in enumerate(keys)}


def candidate_centrality(
    grids: Iterable[Grid], definition: CentralityDefinition = "distinct"
) -> dict[GridKey, float]:
    return centrality_from_workspace(build_similarity_workspace(grids), definition)
