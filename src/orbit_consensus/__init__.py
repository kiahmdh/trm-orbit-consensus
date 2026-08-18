"""CPU-side structured aggregation for cached TRM augmentation orbits."""

from .aggregators import (
    m1_beta_sweep,
    m1_orbit_weighting,
    m2_cell_marginal_score,
    prepare_m1,
)
from .baselines import canonical_prediction, majority_vote
from .schema import Candidate, RankedCandidate, SupportPair, TaskOrbit, make_task_id

__all__ = [
    "Candidate",
    "RankedCandidate",
    "SupportPair",
    "TaskOrbit",
    "canonical_prediction",
    "m1_beta_sweep",
    "m1_orbit_weighting",
    "m2_cell_marginal_score",
    "majority_vote",
    "make_task_id",
    "prepare_m1",
]
