from __future__ import annotations

import numpy as np

from orbit_consensus.schema import Candidate, TaskOrbit, grid_key
from orbit_consensus.supporting_analysis import (
    upstream_evaluator_ranking,
    upstream_evaluator_scores,
)


def _orbit(task_id: str, target: int, candidates: list[tuple[int | None, float]]) -> TaskOrbit:
    emitted = []
    for index, (value, q) in enumerate(candidates):
        invalid = value is None
        grid = (
            np.empty((0, 0), dtype=np.uint8)
            if invalid
            else np.asarray([[value]], dtype=np.uint8)
        )
        emitted.append(Candidate(grid, index, q, is_invalid=invalid))
    return TaskOrbit(
        task_id,
        tuple(emitted),
        target=np.asarray([[target]], dtype=np.uint8),
    )


def test_upstream_ranking_uses_count_then_mean_q_and_retains_invalid() -> None:
    orbit = _orbit("a#0", 2, [(1, 0.1), (2, 0.9), (1, 0.1), (2, 0.9), (None, 1.0)])
    ranking = upstream_evaluator_ranking(orbit)
    assert ranking[0] == grid_key(np.asarray([[2]], dtype=np.uint8))
    assert (0, 0, b"") in ranking
    assert (0, 0, b"") not in upstream_evaluator_ranking(orbit, include_invalid=False)


def test_upstream_scores_are_puzzle_weighted_for_multiple_pairs() -> None:
    orbits = (
        _orbit("a#0", 1, [(1, 0.1)]),
        _orbit("a#1", 1, [(2, 0.1)]),
        _orbit("b#0", 1, [(1, 0.1)]),
    )
    scores = upstream_evaluator_scores(orbits, pass_ks=(1,))
    assert scores == {"pass@1": 0.75}
