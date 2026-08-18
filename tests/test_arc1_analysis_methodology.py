import numpy as np

from orbit_consensus.arc1_analysis import _bootstrap, puzzle_mean, puzzle_values
from orbit_consensus.sampling import deterministic_dev_test_split
from orbit_consensus.schema import Candidate, TaskOrbit


def _orbit(task_id: str) -> TaskOrbit:
    return TaskOrbit(
        task_id,
        (Candidate(np.asarray([[1]], dtype=np.uint8), 0, 0.5, is_identity=True),),
        target=np.asarray([[1]], dtype=np.uint8),
    )


def test_split_is_over_unique_puzzle_ids_not_descriptors():
    puzzle_ids = ("a", "b", "c", "d")
    dev, test = deterministic_dev_test_split(puzzle_ids, seed=20260807, dev_fraction=0.5)
    assignment = {puzzle_id: "dev" if puzzle_id in dev else "test" for puzzle_id in puzzle_ids}
    task_ids = ("a#0", "a#1", "b#0", "c#0", "d#0", "d#1")
    assert all(assignment[task_id.split("#")[0]] in {"dev", "test"} for task_id in task_ids)
    assert set(dev).isdisjoint(test)


def test_puzzle_weighting_does_not_overweight_multi_pair_puzzles():
    orbits = (_orbit("a#0"), _orbit("a#1"), _orbit("b#0"))
    ids, values = puzzle_values((1.0, 0.0, 1.0), orbits)
    assert ids == ("a", "b")
    assert np.array_equal(values, np.asarray([0.5, 1.0]))
    assert puzzle_mean((1.0, 0.0, 1.0), orbits) == 0.75


def test_bootstrap_operates_on_puzzle_vectors_and_is_finite():
    method = np.asarray([1.0, 0.5, 0.0, 1.0])
    baseline = np.asarray([0.0, 0.5, 0.0, 1.0])
    result = _bootstrap(method, baseline, resamples=10_000, seed=20260807)
    assert result["difference"] == 0.25
    assert all(np.isfinite(value) for value in result.values())
