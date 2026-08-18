from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .schema import TaskOrbit


@dataclass(frozen=True)
class SubsamplePlan:
    requested_budget: int
    effective_budget: int
    repeats: tuple[tuple[int, ...], ...]

    @property
    def was_clamped(self) -> bool:
        return self.effective_budget != self.requested_budget


def deterministic_dev_test_split(
    task_ids: Sequence[str], *, seed: int, dev_fraction: float = 0.5
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    unique = sorted(set(task_ids))
    if len(unique) != len(task_ids):
        raise ValueError("task IDs must be unique")
    rng = np.random.default_rng(seed)
    shuffled = [unique[index] for index in rng.permutation(len(unique))]
    cut = round(len(shuffled) * dev_fraction)
    return tuple(sorted(shuffled[:cut])), tuple(sorted(shuffled[cut:]))


def paired_subsamples(
    orbit: TaskOrbit, *, budget: int, repeats: int, seed: int
) -> SubsamplePlan:
    """Clamp small orbits and make the effective compute budget explicit."""
    if budget <= 0 or repeats <= 0:
        raise ValueError("budget and repeats must be positive")
    effective_budget = min(budget, len(orbit.candidates))
    task_seed_bytes = hashlib.sha256(f"{seed}:{orbit.task_id}".encode()).digest()[:8]
    task_seed = int.from_bytes(task_seed_bytes, "little")
    rng = np.random.default_rng(task_seed)
    samples = tuple(
        tuple(
            sorted(
                rng.choice(
                    len(orbit.candidates), size=effective_budget, replace=False
                ).tolist()
            )
        )
        for _ in range(repeats)
    )
    return SubsamplePlan(budget, effective_budget, samples)
