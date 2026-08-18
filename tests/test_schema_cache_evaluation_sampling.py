import numpy as np
import pytest

from orbit_consensus.cache import load_task_orbit, save_task_orbit
from orbit_consensus.evaluation import selection_metrics
from orbit_consensus.sampling import paired_subsamples
from orbit_consensus.schema import Candidate, RankedCandidate, TaskOrbit


def _candidate(grid, index, q=0.5, identity=False, **kwargs):
    return Candidate(
        np.asarray(grid, dtype=np.uint8),
        index,
        q,
        is_identity=identity,
        **kwargs,
    )


def test_task_ids_are_pair_scoped_and_orbit_size_is_recorded():
    orbit = TaskOrbit("00576224#1", (_candidate([[1]], 0),))
    assert orbit.puzzle_id == "00576224"
    assert orbit.test_pair_index == 1
    assert orbit.metadata["emitted_orbit_size"] == 1
    with pytest.raises(ValueError):
        TaskOrbit("00576224", (_candidate([[1]], 0),))


def test_cache_v2_round_trips_q_and_optional_cell_fields(tmp_path):
    entropy = np.asarray([[0.1, 0.2]], dtype=np.float32)
    top3 = np.asarray([[[1, 2, 3], [3, 2, 1]]], dtype=np.uint8)
    orbit = TaskOrbit(
        "task#0",
        (_candidate([[1, 3]], 0, q=0.73, entropy=entropy, top3_colors=top3),),
        target=np.asarray([[1, 3]], dtype=np.uint8),
    )
    path = tmp_path / "task#0.npz"
    save_task_orbit(path, orbit)
    loaded = load_task_orbit(path)
    assert loaded.candidates[0].q_value == 0.73
    assert np.array_equal(loaded.candidates[0].top3_colors, top3)
    assert np.allclose(loaded.candidates[0].entropy, entropy)


def test_coverage_is_orbit_level_and_shape_loss_is_separate():
    target = np.asarray([[9], [9]], dtype=np.uint8)
    orbit = TaskOrbit(
        "task#0",
        (
            _candidate([[1, 1]], 0, identity=True),
            _candidate([[1, 1]], 1),
            _candidate(target, 2),
        ),
        target=target,
    )
    modal_shape_only = (
        RankedCandidate(np.asarray([[1, 1]], dtype=np.uint8), 2.0, 2, 0.5),
    )
    metrics = selection_metrics((orbit,), (modal_shape_only,))
    assert metrics.coverage == 1.0
    assert metrics.ranked_coverage == 0.0
    assert metrics.shape_selection_loss == 1.0
    assert metrics.mean_reciprocal_rank_covered == 0.0


def test_small_orbit_budget_is_clamped_and_recorded():
    orbit = TaskOrbit(
        "task#0",
        tuple(_candidate([[index % 10]], index) for index in range(9)),
    )
    plan = paired_subsamples(orbit, budget=250, repeats=3, seed=7)
    assert plan.requested_budget == 250
    assert plan.effective_budget == 9
    assert plan.was_clamped
    assert all(len(sample) == 9 for sample in plan.repeats)
