from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Grid = NDArray[np.uint8]
GridKey = tuple[int, int, bytes]


def is_empty_prediction(value: NDArray[Any]) -> bool:
    """Return whether *value* is the upstream invalid-prediction sentinel."""
    return np.asarray(value).shape == (0, 0)


def as_grid(value: NDArray[np.integer[Any]] | list[list[int]]) -> Grid:
    """Return a validated ARC grid as an owned uint8 array."""
    raw = np.asarray(value)
    if raw.ndim != 2 or not (1 <= raw.shape[0] <= 30 and 1 <= raw.shape[1] <= 30):
        raise ValueError(f"ARC grids must be 2-D and between 1x1 and 30x30; got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw > 9):
        raise ValueError("ARC cell values must be integers in [0, 9]")
    return np.array(raw, dtype=np.uint8, copy=True, order="C")


def grid_key(grid: Grid) -> GridKey:
    array = np.asarray(grid, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("grid keys require a two-dimensional array")
    return (int(array.shape[0]), int(array.shape[1]), array.tobytes(order="C"))


_TASK_ID_PATTERN = re.compile(r"^(?P<puzzle_id>[^#]+)#(?P<pair_index>0|[1-9][0-9]*)$")


def make_task_id(puzzle_id: str, test_pair_index: int) -> str:
    if not puzzle_id or "#" in puzzle_id:
        raise ValueError("puzzle_id must be non-empty and cannot contain '#'")
    if test_pair_index < 0:
        raise ValueError("test_pair_index must be non-negative")
    return f"{puzzle_id}#{test_pair_index}"


def split_task_id(task_id: str) -> tuple[str, int]:
    match = _TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        raise ValueError("task_id must be pair-scoped, for example '00576224#0'")
    return match.group("puzzle_id"), int(match.group("pair_index"))


@dataclass(frozen=True)
class Candidate:
    grid: Grid
    augmentation_index: int
    q_value: float
    transform: Mapping[str, Any] = field(default_factory=dict)
    is_identity: bool = False
    is_invalid: bool = False
    entropy: NDArray[np.floating[Any]] | None = None
    top3_colors: NDArray[np.integer[Any]] | None = None

    def __post_init__(self) -> None:
        if self.is_invalid:
            raw = np.asarray(self.grid)
            if raw.shape != (0, 0):
                raise ValueError("invalid predictions must use the exact (0, 0) sentinel")
            if not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("invalid prediction sentinels must have an integer dtype")
            object.__setattr__(self, "grid", np.empty((0, 0), dtype=np.uint8))
        else:
            object.__setattr__(self, "grid", as_grid(self.grid))
        if self.augmentation_index < 0:
            raise ValueError("augmentation_index must be non-negative")
        if not math.isfinite(self.q_value):
            raise ValueError("q_value must be finite")
        if self.entropy is not None and self.entropy.shape != self.grid.shape:
            raise ValueError("entropy must match the decoded grid shape")
        if self.top3_colors is not None and self.top3_colors.shape != (*self.grid.shape, 3):
            raise ValueError("top3_colors must have shape (height, width, 3)")


@dataclass(frozen=True)
class SupportPair:
    input_grid: Grid
    output_grid: Grid

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_grid", as_grid(self.input_grid))
        object.__setattr__(self, "output_grid", as_grid(self.output_grid))


@dataclass(frozen=True)
class TaskOrbit:
    task_id: str
    candidates: tuple[Candidate, ...]
    query_input: Grid | None = None
    support_pairs: tuple[SupportPair, ...] = ()
    target: Grid | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        split_task_id(self.task_id)
        if not self.candidates:
            raise ValueError("a task orbit must contain at least one candidate")
        indices = [candidate.augmentation_index for candidate in self.candidates]
        if len(indices) != len(set(indices)):
            raise ValueError("augmentation indices must be unique within a task orbit")
        if self.query_input is not None:
            object.__setattr__(self, "query_input", as_grid(self.query_input))
        if self.target is not None:
            object.__setattr__(self, "target", as_grid(self.target))
        metadata = dict(self.metadata)
        emitted_orbit_size = metadata.setdefault("emitted_orbit_size", len(self.candidates))
        if emitted_orbit_size != len(self.candidates):
            raise ValueError("metadata.emitted_orbit_size must equal the candidate count")
        object.__setattr__(self, "metadata", metadata)

    @property
    def puzzle_id(self) -> str:
        return split_task_id(self.task_id)[0]

    @property
    def test_pair_index(self) -> int:
        return split_task_id(self.task_id)[1]


@dataclass(frozen=True)
class RankedCandidate:
    grid: Grid
    score: float
    emission_count: int
    mean_q: float
    is_invalid: bool = False

    def __post_init__(self) -> None:
        if self.is_invalid:
            raw = np.asarray(self.grid)
            if raw.shape != (0, 0):
                raise ValueError("invalid ranked predictions must use the exact (0, 0) sentinel")
            object.__setattr__(self, "grid", np.empty((0, 0), dtype=np.uint8))
        else:
            object.__setattr__(self, "grid", as_grid(self.grid))
