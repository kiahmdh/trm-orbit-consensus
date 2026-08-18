import numpy as np
import pytest

from orbit_consensus.aggregators import (
    m1_orbit_weighting,
    m2_cell_marginal_score,
)
from orbit_consensus.baselines import majority_vote
from orbit_consensus.cache import load_task_orbit, save_task_orbit
from orbit_consensus.evaluation import exact_match
from orbit_consensus.schema import Candidate, SupportPair, TaskOrbit, grid_key
from orbit_consensus.similarity import build_similarity_workspace, grid_similarity


def _invalid(index: int, q_value: float = 0.5) -> Candidate:
    return Candidate(
        np.empty((0, 0), dtype=np.uint8),
        index,
        q_value,
        is_invalid=True,
    )


def test_empty_raw_prediction_candidate_npz_reload_is_lossless(tmp_path):
    raw_prediction = np.zeros(900, dtype=np.uint8)
    canonical_prediction = raw_prediction[raw_prediction != 0].reshape(0, 0)

    with pytest.raises(ValueError):
        Candidate(canonical_prediction, 0, 0.25)

    invalid = Candidate(canonical_prediction, 0, 0.25, is_invalid=True)
    normal_grid = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    normal = Candidate(normal_grid, 1, 0.75)
    orbit = TaskOrbit("empty-test#0", (invalid, normal), target=normal_grid)
    path = tmp_path / "empty-test#0.npz"

    save_task_orbit(path, orbit)
    loaded = load_task_orbit(path)

    assert loaded.candidates[0].is_invalid
    assert loaded.candidates[0].grid.shape == (0, 0)
    assert loaded.candidates[0].grid.dtype == np.uint8
    assert grid_key(loaded.candidates[0].grid) == (0, 0, b"")
    assert not loaded.candidates[1].is_invalid
    assert np.array_equal(loaded.candidates[1].grid, normal_grid)
    assert loaded.candidates[1].q_value == normal.q_value


def test_majority_vote_keeps_empty_predictions_in_vote_population():
    valid = Candidate(np.asarray([[7]], dtype=np.uint8), 2, 0.99)
    orbit = TaskOrbit("vote-test#0", (_invalid(0, 0.1), _invalid(1, 0.2), valid))

    ranking = majority_vote(orbit)

    assert ranking[0].is_invalid
    assert ranking[0].grid.shape == (0, 0)
    assert ranking[0].emission_count == 2
    assert sum(candidate.emission_count for candidate in ranking) == 3
    assert not exact_match(ranking[0].grid, valid.grid)


def test_similarity_centrality_and_marginals_accept_empty_predictions():
    empty = np.empty((0, 0), dtype=np.uint8)
    valid_grid = np.asarray([[3]], dtype=np.uint8)
    orbit = TaskOrbit(
        "similarity-test#0",
        (_invalid(0), _invalid(1), Candidate(valid_grid, 2, 0.9)),
    )

    assert grid_similarity(empty, empty) == 1.0
    assert grid_similarity(empty, valid_grid) == 0.0
    workspace = build_similarity_workspace(candidate.grid for candidate in orbit.candidates)
    assert np.isfinite(workspace.matrix).all()
    assert m1_orbit_weighting(orbit, beta=1.0).ranking

    marginal = m2_cell_marginal_score(orbit, interpolation=1.0, epsilon=0.001)
    assert marginal.modal_shape == (0, 0)
    assert marginal.ranking[0].is_invalid


def test_shape_screening_filters_empty_without_crashing():
    valid_grid = np.asarray([[4]], dtype=np.uint8)
    supports = (SupportPair(np.asarray([[1]], dtype=np.uint8), valid_grid),)
    orbit = TaskOrbit(
        "shape-test#0",
        (_invalid(0), Candidate(valid_grid, 1, 0.8)),
        query_input=np.asarray([[2]], dtype=np.uint8),
        support_pairs=supports,
    )

    result = m2_cell_marginal_score(
        orbit,
        interpolation=0.5,
        epsilon=0.001,
        use_shape_screening=True,
    )

    assert result.modal_shape == (1, 1)
    assert all(not candidate.is_invalid for candidate in result.ranking)
