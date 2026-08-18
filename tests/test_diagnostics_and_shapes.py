import numpy as np

from orbit_consensus.diagnostics import equivariance_defect
from orbit_consensus.schema import Candidate, SupportPair, TaskOrbit
from orbit_consensus.shape_screening import infer_shape_screen


def test_equivariance_defect_zero_for_degenerate_orbit():
    grid = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    orbit = TaskOrbit(
        "task#0",
        (Candidate(grid, 0, 0.5, is_identity=True), Candidate(grid, 1, 0.4)),
    )
    assert equivariance_defect(orbit) == 0.0


def test_integer_scale_shape_screen():
    supports = (
        SupportPair(np.zeros((2, 3), dtype=np.uint8), np.zeros((4, 9), dtype=np.uint8)),
        SupportPair(np.zeros((1, 2), dtype=np.uint8), np.zeros((2, 6), dtype=np.uint8)),
    )
    screen = infer_shape_screen(supports, (3, 4))
    assert (6, 12) in screen.allowed_shapes
    assert "integer_scale_2x3" in screen.relations
