import numpy as np

from orbit_consensus.aggregators import (
    m1_beta_sweep,
    m1_orbit_weighting,
    m2_cell_marginal_score,
    prepare_m1,
)
from orbit_consensus.baselines import majority_vote
from orbit_consensus.schema import Candidate, TaskOrbit, grid_key


def _candidate(grid, index, q=0.5, identity=False):
    return Candidate(np.asarray(grid, dtype=np.uint8), index, q, is_identity=identity)


def test_majority_vote_breaks_count_ties_by_mean_q():
    orbit = TaskOrbit(
        "task#0",
        (
            _candidate([[1]], 0, q=0.2, identity=True),
            _candidate([[1]], 1, q=0.4),
            _candidate([[2]], 2, q=0.8),
            _candidate([[2]], 3, q=0.6),
        ),
    )
    ranking = majority_vote(orbit)
    assert np.array_equal(ranking[0].grid, np.asarray([[2]], dtype=np.uint8))
    assert ranking[0].mean_q == 0.7


def test_m1_beta_zero_recovers_majority_order():
    orbit = TaskOrbit(
        "task#0",
        (
            _candidate([[1, 1]], 0, identity=True),
            _candidate([[1, 1]], 1),
            _candidate([[1, 0]], 2),
        ),
    )
    b1 = majority_vote(orbit)
    m1 = m1_orbit_weighting(orbit, beta=0.0)
    assert [grid_key(item.grid) for item in m1.ranking] == [grid_key(item.grid) for item in b1]


def test_m1_beta_sweep_reuses_precomputation():
    orbit = TaskOrbit(
        "task#0",
        tuple(_candidate([[index % 3, index % 2]], index) for index in range(12)),
    )
    prepared = prepare_m1(orbit)
    results = m1_beta_sweep(orbit, (-2.0, 0.0, 2.0), "distinct", prepared=prepared)
    assert set(results) == {-2.0, 0.0, 2.0}
    assert all(result.centrality is prepared.centrality["distinct"] for result in results.values())


def test_m2_lambda_zero_recovers_majority_on_modal_shape():
    orbit = TaskOrbit(
        "task#0",
        (
            _candidate([[1, 1]], 0, identity=True),
            _candidate([[1, 1]], 1),
            _candidate([[1, 0]], 2),
            _candidate([[2], [2]], 3),
        ),
    )
    result = m2_cell_marginal_score(orbit, interpolation=0.0, epsilon=0.001)
    assert result.modal_shape == (1, 2)
    assert np.array_equal(result.ranking[0].grid, np.asarray([[1, 1]], dtype=np.uint8))


def test_m2_never_synthesizes_a_grid():
    orbit = TaskOrbit(
        "task#0",
        (
            _candidate([[1, 0], [0, 1]], 0, identity=True),
            _candidate([[1, 1], [0, 0]], 1),
            _candidate([[0, 1], [1, 0]], 2),
        ),
    )
    emitted = {grid_key(candidate.grid) for candidate in orbit.candidates}
    result = m2_cell_marginal_score(orbit, interpolation=1.0, epsilon=0.001)
    assert all(grid_key(candidate.grid) in emitted for candidate in result.ranking)
