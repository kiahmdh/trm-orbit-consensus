from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import GridKey, TaskOrbit
from .similarity import (
    CentralityDefinition,
    SimilarityWorkspace,
    build_similarity_workspace,
    centrality_from_workspace,
)


@dataclass(frozen=True)
class VoteConcentration:
    unique_candidates: int
    modal_share: float
    cumulative_mass: tuple[float, ...]


def vote_concentration(
    orbit: TaskOrbit, *, workspace: SimilarityWorkspace | None = None
) -> VoteConcentration:
    workspace = workspace or build_similarity_workspace(
        candidate.grid for candidate in orbit.candidates
    )
    total = sum(workspace.counts.values())
    shares = sorted((count / total for count in workspace.counts.values()), reverse=True)
    return VoteConcentration(len(workspace.counts), shares[0], tuple(np.cumsum(shares).tolist()))


def equivariance_defect(
    orbit: TaskOrbit, *, workspace: SimilarityWorkspace | None = None
) -> float:
    workspace = workspace or build_similarity_workspace(
        candidate.grid for candidate in orbit.candidates
    )
    counts = np.asarray([workspace.counts[key] for key in workspace.keys], dtype=np.float64)
    total = np.sum(counts)
    mean_similarity = float(counts @ workspace.matrix.astype(np.float64) @ counts / (total * total))
    return 1.0 - mean_similarity


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_d_count(
    orbit: TaskOrbit,
    definition: CentralityDefinition,
    *,
    workspace: SimilarityWorkspace | None = None,
) -> float:
    workspace = workspace or build_similarity_workspace(
        candidate.grid for candidate in orbit.candidates
    )
    centrality = centrality_from_workspace(workspace, definition)
    keys: tuple[GridKey, ...] = workspace.keys
    if len(keys) < 2:
        return float("nan")
    d_ranks = _average_ranks(np.asarray([centrality[key] for key in keys]))
    count_ranks = _average_ranks(np.asarray([workspace.counts[key] for key in keys]))
    if np.std(d_ranks) == 0 or np.std(count_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(d_ranks, count_ranks)[0, 1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError("scores and labels must be non-empty one-dimensional arrays")
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ranks = _average_ranks(scores) + 1.0
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def defect_correctness_statistics(
    defects: np.ndarray, correctness: np.ndarray
) -> tuple[float, float]:
    defects = np.asarray(defects, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=bool)
    if defects.shape != correctness.shape:
        raise ValueError("defects and correctness must have equal shape")
    auc = auroc(-defects, correctness)
    correlation = (
        float("nan")
        if np.std(defects) == 0 or np.std(correctness.astype(np.float64)) == 0
        else float(np.corrcoef(defects, correctness.astype(np.float64))[0, 1])
    )
    return auc, correlation
