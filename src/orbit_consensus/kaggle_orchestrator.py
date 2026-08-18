from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_pipeline import load_orbits
from .baselines import majority_vote
from .kaggle_cache import merge_partial_caches, task_descriptors
from .schema import grid_key


@dataclass(frozen=True)
class GPUCacheSpec:
    label: str
    upstream_root: Path
    dataset_path: Path
    checkpoint_path: Path
    output_dir: Path
    puzzle_id_mode: str
    code_revision: str
    checkpoint_revision: str
    dataset_revision: str
    batch_size: int = 8


def restore_cache_dataset(input_root: Path, working_root: Path) -> None:
    input_root = Path(input_root)
    working_root = Path(working_root)
    for name in ("arc1", "arc1_blank", "arc2"):
        candidates = [path for path in input_root.rglob(name) if path.is_dir()]
        if not candidates:
            continue
        source = min(candidates, key=lambda path: len(path.parts))
        shutil.copytree(source, working_root / name, dirs_exist_ok=True)
    for name in ("dataset-metadata.json", ".publish_state.json"):
        matches = list(input_root.rglob(name))
        if matches:
            shutil.copy2(matches[0], working_root / name)


def _worker_command(spec: GPUCacheSpec, output_dir: Path, task_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "orbit_consensus.kaggle_cache",
        "--upstream-root",
        str(spec.upstream_root),
        "--dataset-path",
        str(spec.dataset_path),
        "--checkpoint-path",
        str(spec.checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--task-ids-json",
        str(task_file),
        "--batch-size",
        str(spec.batch_size),
        "--device",
        "cuda:0",
        "--puzzle-id-mode",
        spec.puzzle_id_mode,
        "--code-revision",
        spec.code_revision,
        "--checkpoint-revision",
        spec.checkpoint_revision,
        "--dataset-revision",
        spec.dataset_revision,
    ]


def _run_round(
    spec: GPUCacheSpec,
    assignments: Sequence[Sequence[str]],
    *,
    partial_root: Path,
    log_root: Path,
) -> tuple[Path, ...]:
    processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
    partials: list[Path] = []
    log_root.mkdir(parents=True, exist_ok=True)
    for gpu_index, task_ids in enumerate(assignments):
        if not task_ids:
            continue
        partial = partial_root / spec.label / f"gpu{gpu_index}"
        partial.mkdir(parents=True, exist_ok=True)
        partials.append(partial)
        task_file = partial_root / f"{spec.label}-gpu{gpu_index}-{uuid.uuid4().hex}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(list(task_ids)), encoding="utf-8")
        log_path = log_root / f"{spec.label}-gpu{gpu_index}-{uuid.uuid4().hex}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            _worker_command(spec, partial, task_file),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        processes.append((process, log_handle, log_path))

    failures: list[str] = []
    for process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
            failures.append(f"{log_path} exited {return_code}:\n{tail}")
    if failures:
        raise RuntimeError("\n\n".join(failures))
    return tuple(partials)


def run_chunked_cache(
    spec: GPUCacheSpec,
    *,
    checkpoint_every: int,
    partial_root: Path,
    log_root: Path,
    visible_gpu_count: int,
    selected_task_ids: Sequence[str] | None = None,
    after_checkpoint: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    descriptors = task_descriptors(spec.dataset_path)
    expected = [descriptor.task_id for descriptor in descriptors]
    if selected_task_ids is not None:
        selected = set(selected_task_ids)
        expected = [task_id for task_id in expected if task_id in selected]
        missing = selected - set(expected)
        if missing:
            raise KeyError(f"selected task IDs are absent: {sorted(missing)}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {path.stem for path in spec.output_dir.glob("*.npz")}
    pending = [task_id for task_id in expected if task_id not in completed]

    worker_count = 2 if visible_gpu_count >= 2 and len(pending) > 1 else 1
    halves = [
        list(items)
        for items in np.array_split(np.asarray(pending, dtype=object), worker_count)
    ]
    pointers = [0] * worker_count
    rounds = 0
    while any(pointer < len(items) for pointer, items in zip(pointers, halves)):
        remaining_budget = checkpoint_every
        assignments: list[list[str]] = [[] for _ in range(worker_count)]
        while remaining_budget and any(
            pointer < len(items) for pointer, items in zip(pointers, halves)
        ):
            for worker_index in range(worker_count):
                if remaining_budget == 0:
                    break
                if pointers[worker_index] < len(halves[worker_index]):
                    assignments[worker_index].append(
                        str(halves[worker_index][pointers[worker_index]])
                    )
                    pointers[worker_index] += 1
                    remaining_budget -= 1

        partials = _run_round(
            spec,
            assignments,
            partial_root=Path(partial_root),
            log_root=Path(log_root),
        )
        manifest_path = merge_partial_caches(
            partials,
            spec.output_dir,
            expected_task_ids=expected,
            provenance={
                "label": spec.label,
                "code_revision": spec.code_revision,
                "checkpoint_revision": spec.checkpoint_revision,
                "dataset_revision": spec.dataset_revision,
                "puzzle_id_mode": spec.puzzle_id_mode,
            },
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rounds += 1
        if after_checkpoint is not None:
            after_checkpoint(manifest["completed_task_count"], len(expected))

    final_manifest = merge_partial_caches(
        (),
        spec.output_dir,
        expected_task_ids=expected,
        provenance={
            "label": spec.label,
            "code_revision": spec.code_revision,
            "checkpoint_revision": spec.checkpoint_revision,
            "dataset_revision": spec.dataset_revision,
            "puzzle_id_mode": spec.puzzle_id_mode,
        },
    )
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    if payload["completed_task_count"] != len(expected):
        raise AssertionError("cache run ended without every selected task")
    return {"rounds": rounds, "workers": worker_count, "manifest": payload}


def b1_reproduction_gate(cache_dir: Path, *, low: float = 0.38, high: float = 0.42) -> float:
    orbits = load_orbits(cache_dir)
    stream: list[float] = []
    reloaded: list[float] = []
    by_puzzle_stream: dict[str, list[float]] = defaultdict(list)
    by_puzzle_reload: dict[str, list[float]] = defaultdict(list)
    for orbit in orbits:
        if "stream_b1_correct" not in orbit.metadata:
            raise AssertionError(f"{orbit.task_id} lacks stream-time B1 metadata")
        stream_correct = float(bool(orbit.metadata["stream_b1_correct"]))
        target = grid_key(orbit.target) if orbit.target is not None else None
        ranking = majority_vote(orbit)
        reload_correct = float(grid_key(ranking[0].grid) == target)
        stream.append(stream_correct)
        reloaded.append(reload_correct)
        by_puzzle_stream[orbit.puzzle_id].append(stream_correct)
        by_puzzle_reload[orbit.puzzle_id].append(reload_correct)
    if stream != reloaded:
        mismatches = [
            orbit.task_id
            for orbit, left, right in zip(orbits, stream, reloaded)
            if left != right
        ]
        raise AssertionError(f"stream and reloaded B1 differ for {mismatches[:10]}")
    stream_score = float(np.mean([np.mean(values) for values in by_puzzle_stream.values()]))
    reload_score = float(np.mean([np.mean(values) for values in by_puzzle_reload.values()]))
    if stream_score != reload_score:
        raise AssertionError("stream and reloaded aggregate B1 differ")
    if not low <= reload_score <= high:
        raise AssertionError(
            f"ARC-AGI-1 B1={reload_score:.4f} is outside the hard [{low:.2f}, {high:.2f}] gate"
        )
    return reload_score


def blank_identifier_gate(cache_dir: Path) -> float:
    orbits = load_orbits(cache_dir)
    by_puzzle: dict[str, list[float]] = defaultdict(list)
    for orbit in orbits:
        target = grid_key(orbit.target) if orbit.target is not None else None
        correct = float(grid_key(majority_vote(orbit)[0].grid) == target)
        by_puzzle[orbit.puzzle_id].append(correct)
    score = float(np.mean([np.mean(values) for values in by_puzzle.values()]))
    if score != 0.0:
        raise AssertionError(
            f"blank puzzle IDs produced B1={score:.4f}; expected the reported collapse to 0"
        )
    return score


def publish_cache_dataset(
    cache_root: Path,
    *,
    owner: str,
    slug: str,
    message: str,
    dataset_exists: bool,
) -> None:
    cache_root = Path(cache_root)
    metadata = {
        "title": "TRM Structured Orbit Incremental Cache",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (cache_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    state_path = cache_root / ".publish_state.json"
    use_version = dataset_exists or state_path.exists()
    if use_version:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(cache_root),
            "-m",
            message,
            "-r",
            "zip",
        ]
    else:
        command = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(cache_root),
            "-r",
            "zip",
        ]
    state_path.write_text(
        json.dumps({"dataset": f"{owner}/{slug}", "last_message": message}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(command, check=True)
    except Exception:
        if not use_version:
            state_path.unlink(missing_ok=True)
        raise
