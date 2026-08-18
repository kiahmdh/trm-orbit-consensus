from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Candidate, SupportPair, TaskOrbit, as_grid

FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = frozenset({2, FORMAT_VERSION})
MAX_GRID_SIZE = 30


def _pad(grid: np.ndarray, *, fill: float = 0, dtype=np.uint8) -> np.ndarray:
    padded = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), fill, dtype=dtype)
    padded[: grid.shape[0], : grid.shape[1]] = grid
    return padded


def _unpad(padded: np.ndarray, shape: np.ndarray) -> np.ndarray:
    return as_grid(padded[: int(shape[0]), : int(shape[1])])


def _unpad_prediction(
    padded: np.ndarray, shape: np.ndarray, *, is_invalid: bool
) -> np.ndarray:
    if is_invalid:
        if tuple(int(value) for value in shape) != (0, 0):
            raise ValueError("invalid cached predictions must have shape (0, 0)")
        return np.empty((0, 0), dtype=np.uint8)
    return _unpad(padded, shape)


def save_task_orbit(path: Path, orbit: TaskOrbit) -> None:
    """Persist one pair-scoped task immediately; no pickle is used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = np.stack([_pad(candidate.grid) for candidate in orbit.candidates])
    shapes = np.asarray([candidate.grid.shape for candidate in orbit.candidates], dtype=np.uint8)
    entropy = np.full((len(orbit.candidates), 30, 30), np.nan, dtype=np.float32)
    top3 = np.full((len(orbit.candidates), 30, 30, 3), 255, dtype=np.uint8)
    has_entropy = np.zeros(len(orbit.candidates), dtype=bool)
    has_top3 = np.zeros(len(orbit.candidates), dtype=bool)
    for index, candidate in enumerate(orbit.candidates):
        height, width = candidate.grid.shape
        if candidate.entropy is not None:
            entropy[index, :height, :width] = candidate.entropy
            has_entropy[index] = True
        if candidate.top3_colors is not None:
            top3[index, :height, :width] = candidate.top3_colors
            has_top3[index] = True

    payload: dict[str, Any] = {
        "format_version": np.asarray(FORMAT_VERSION),
        "task_id": np.asarray(orbit.task_id),
        "grids": grids,
        "shapes": shapes,
        "augmentation_indices": np.asarray(
            [candidate.augmentation_index for candidate in orbit.candidates], dtype=np.int32
        ),
        "q_values": np.asarray(
            [candidate.q_value for candidate in orbit.candidates], dtype=np.float64
        ),
        "is_identity": np.asarray([candidate.is_identity for candidate in orbit.candidates]),
        "is_invalid": np.asarray([candidate.is_invalid for candidate in orbit.candidates]),
        "transforms": np.asarray(
            [json.dumps(dict(candidate.transform), sort_keys=True) for candidate in orbit.candidates]
        ),
        "entropy": entropy,
        "top3_colors": top3,
        "has_entropy": has_entropy,
        "has_top3_colors": has_top3,
        "metadata": np.asarray(json.dumps(dict(orbit.metadata), sort_keys=True)),
    }
    if orbit.query_input is not None:
        payload["query_input"] = _pad(orbit.query_input)
        payload["query_shape"] = np.asarray(orbit.query_input.shape, dtype=np.uint8)
    if orbit.target is not None:
        payload["target"] = _pad(orbit.target)
        payload["target_shape"] = np.asarray(orbit.target.shape, dtype=np.uint8)
    payload["support_inputs"] = (
        np.stack([_pad(pair.input_grid) for pair in orbit.support_pairs])
        if orbit.support_pairs
        else np.empty((0, 30, 30), dtype=np.uint8)
    )
    payload["support_input_shapes"] = np.asarray(
        [pair.input_grid.shape for pair in orbit.support_pairs], dtype=np.uint8
    ).reshape(-1, 2)
    payload["support_outputs"] = (
        np.stack([_pad(pair.output_grid) for pair in orbit.support_pairs])
        if orbit.support_pairs
        else np.empty((0, 30, 30), dtype=np.uint8)
    )
    payload["support_output_shapes"] = np.asarray(
        [pair.output_grid.shape for pair in orbit.support_pairs], dtype=np.uint8
    ).reshape(-1, 2)
    np.savez_compressed(path, **payload)


def load_task_orbit(path: Path) -> TaskOrbit:
    with np.load(path, allow_pickle=False) as data:
        format_version = int(data["format_version"])
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"unsupported cache format in {path}")
        invalid_flags = (
            data["is_invalid"]
            if format_version >= 3
            else np.zeros(len(data["grids"]), dtype=bool)
        )
        candidates: list[Candidate] = []
        for position, (grid, shape, index, q_value, transform, identity, invalid) in enumerate(
            zip(
                data["grids"],
                data["shapes"],
                data["augmentation_indices"],
                data["q_values"],
                data["transforms"],
                data["is_identity"],
                invalid_flags,
            )
        ):
            height, width = int(shape[0]), int(shape[1])
            candidates.append(
                Candidate(
                    grid=_unpad_prediction(grid, shape, is_invalid=bool(invalid)),
                    augmentation_index=int(index),
                    q_value=float(q_value),
                    transform=json.loads(str(transform)),
                    is_identity=bool(identity),
                    is_invalid=bool(invalid),
                    entropy=(
                        np.array(data["entropy"][position, :height, :width], copy=True)
                        if bool(data["has_entropy"][position])
                        else None
                    ),
                    top3_colors=(
                        np.array(data["top3_colors"][position, :height, :width], copy=True)
                        if bool(data["has_top3_colors"][position])
                        else None
                    ),
                )
            )
        supports = tuple(
            SupportPair(_unpad(input_grid, input_shape), _unpad(output_grid, output_shape))
            for input_grid, input_shape, output_grid, output_shape in zip(
                data["support_inputs"],
                data["support_input_shapes"],
                data["support_outputs"],
                data["support_output_shapes"],
            )
        )
        query = _unpad(data["query_input"], data["query_shape"]) if "query_input" in data else None
        target = _unpad(data["target"], data["target_shape"]) if "target" in data else None
        return TaskOrbit(
            task_id=str(data["task_id"]),
            candidates=tuple(candidates),
            query_input=query,
            support_pairs=supports,
            target=target,
            metadata=json.loads(str(data["metadata"])),
        )
