from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import load_task_orbit, save_task_orbit
from .schema import Candidate, SupportPair, TaskOrbit, grid_key, make_task_id


@dataclass(frozen=True)
class TaskDescriptor:
    index: int
    group_index: int
    puzzle_id: str
    pair_index: int
    task_id: str


@dataclass(frozen=True)
class CacheEntry:
    task_id: str
    relative_path: str
    sha256: str
    bytes: int
    candidates: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def task_descriptors(dataset_path: Path) -> tuple[TaskDescriptor, ...]:
    dataset_path = Path(dataset_path)
    with (dataset_path / "test_puzzles.json").open(encoding="utf-8") as handle:
        puzzles = json.load(handle)
    group_indices = np.load(dataset_path / "test" / "all__group_indices.npy", mmap_mode="r")
    if len(group_indices) - 1 != len(puzzles):
        raise AssertionError(
            f"test groups ({len(group_indices) - 1}) do not match puzzles ({len(puzzles)})"
        )

    descriptors: list[TaskDescriptor] = []
    for group_index, (puzzle_id, puzzle) in enumerate(puzzles.items()):
        for pair_index, _pair in enumerate(puzzle["test"]):
            descriptors.append(
                TaskDescriptor(
                    index=len(descriptors),
                    group_index=group_index,
                    puzzle_id=puzzle_id,
                    pair_index=pair_index,
                    task_id=make_task_id(puzzle_id, pair_index),
                )
            )
    return tuple(descriptors)


def _atomic_save(path: Path, orbit: TaskOrbit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        save_task_orbit(temporary, orbit)
        load_task_orbit(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_inventory(cache_dir: Path, *, verify: bool = True) -> tuple[CacheEntry, ...]:
    cache_dir = Path(cache_dir)
    entries: list[CacheEntry] = []
    seen: set[str] = set()
    for path in sorted(cache_dir.rglob("*.npz")):
        if path.name.startswith("."):
            continue
        orbit = load_task_orbit(path) if verify else None
        task_id = orbit.task_id if orbit is not None else path.stem
        if task_id in seen:
            raise AssertionError(f"duplicate task ID in cache: {task_id}")
        seen.add(task_id)
        entries.append(
            CacheEntry(
                task_id=task_id,
                relative_path=str(path.relative_to(cache_dir)),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                candidates=len(orbit.candidates) if orbit is not None else -1,
            )
        )
    return tuple(entries)


def write_cache_manifest(
    cache_dir: Path,
    *,
    expected_task_ids: Sequence[str] = (),
    provenance: dict[str, Any] | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    entries = cache_inventory(cache_dir, verify=True)
    actual_ids = {entry.task_id for entry in entries}
    expected_ids = set(expected_task_ids)
    unknown = actual_ids - expected_ids if expected_ids else set()
    if unknown:
        raise AssertionError(f"cache contains unknown task IDs: {sorted(unknown)[:5]}")
    payload = {
        "format_version": 1,
        "created_at_unix": time.time(),
        "completed_task_count": len(entries),
        "expected_task_count": len(expected_ids) if expected_ids else None,
        "complete": bool(expected_ids) and actual_ids == expected_ids,
        "aggregate_sha256": hashlib.sha256(
            "".join(f"{entry.task_id}:{entry.sha256}\n" for entry in entries).encode()
        ).hexdigest(),
        "entries": [asdict(entry) for entry in entries],
        "provenance": provenance or {},
    }
    path = cache_dir / "cache_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def merge_partial_caches(
    partial_dirs: Sequence[Path],
    merged_dir: Path,
    *,
    expected_task_ids: Sequence[str] = (),
    provenance: dict[str, Any] | None = None,
) -> Path:
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    for partial_dir in map(Path, partial_dirs):
        if not partial_dir.exists():
            continue
        for source in sorted(partial_dir.glob("*.npz")):
            orbit = load_task_orbit(source)
            destination = merged_dir / f"{orbit.task_id}.npz"
            if destination.exists():
                if sha256_file(source) != sha256_file(destination):
                    raise AssertionError(f"conflicting cache files for {orbit.task_id}")
                source.unlink()
            else:
                os.replace(source, destination)
    return write_cache_manifest(
        merged_dir,
        expected_task_ids=expected_task_ids,
        provenance=provenance,
    )


@contextmanager
def _import_upstream(root: Path) -> Iterator[None]:
    value = str(Path(root).resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path.remove(value)


def _crop(sequence: np.ndarray) -> np.ndarray:
    grid = np.asarray(sequence).reshape(30, 30)
    max_area = 0
    max_rows = 0
    max_columns = 0
    columns = 30
    for rows in range(1, 31):
        for column in range(1, columns + 1):
            value = grid[rows - 1, column - 1]
            if value < 2 or value > 11:
                columns = column - 1
                break
        area = rows * columns
        if area > max_area:
            max_area = area
            max_rows, max_columns = rows, columns
    return (grid[:max_rows, :max_columns] - 2).astype(np.uint8)


def _load_model(
    upstream_root: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device: str,
):
    import torch
    import yaml

    with (Path(dataset_path) / "test" / "dataset.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (Path(checkpoint_path).parent / "all_config.yaml").open(encoding="utf-8") as handle:
        checkpoint_config = yaml.safe_load(handle)
    architecture = dict(checkpoint_config["arch"])
    loss_config = architecture.pop("loss")

    capability = torch.cuda.get_device_capability(torch.device(device))
    requested_dtype = architecture.get("forward_dtype", "bfloat16")
    if requested_dtype == "bfloat16" and capability[0] < 8:
        architecture["forward_dtype"] = "float16"
    effective_dtype = architecture.get("forward_dtype", requested_dtype)
    architecture.update(
        batch_size=batch_size,
        vocab_size=metadata["vocab_size"],
        seq_len=metadata["seq_len"],
        num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
        causal=False,
    )

    with _import_upstream(upstream_root):
        from models.losses import ACTLossHead
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

        with torch.device(device):
            base = TinyRecursiveReasoningModel_ACTV1(architecture)
            wrapper = ACTLossHead(base, loss_type=loss_config["loss_type"])

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        state = torch.load(checkpoint_path, weights_only=True, mmap=True, **load_kwargs)
    except TypeError:
        state = torch.load(checkpoint_path, **load_kwargs)
    if any(key.startswith("_orig_mod.") for key in state):
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in state.items()
        }
    missing, unexpected = wrapper.load_state_dict(state, strict=False, assign=False)
    if missing or unexpected:
        raise AssertionError(
            f"checkpoint keys do not match model; missing={missing}, unexpected={unexpected}"
        )
    del state
    wrapper.eval()
    return wrapper.model, {
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "compute_capability": list(capability),
        "checkpoint_forward_dtype": requested_dtype,
        "effective_forward_dtype": effective_dtype,
        "dtype_fallback": requested_dtype != effective_dtype,
    }


def _load_selected_descriptors(
    dataset_path: Path, task_ids_path: Path
) -> tuple[TaskDescriptor, ...]:
    requested = set(json.loads(Path(task_ids_path).read_text(encoding="utf-8")))
    descriptors = task_descriptors(dataset_path)
    selected = tuple(descriptor for descriptor in descriptors if descriptor.task_id in requested)
    missing = requested - {descriptor.task_id for descriptor in selected}
    if missing:
        raise KeyError(f"requested task IDs are absent from the dataset: {sorted(missing)}")
    return selected


def run_worker(
    *,
    upstream_root: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    task_ids_path: Path,
    batch_size: int,
    device: str,
    puzzle_id_mode: str,
    code_revision: str,
    checkpoint_revision: str,
    dataset_revision: str,
) -> dict[str, Any]:
    import torch

    if puzzle_id_mode not in {"normal", "blank"}:
        raise ValueError("puzzle_id_mode must be normal or blank")
    selected = _load_selected_descriptors(dataset_path, task_ids_path)
    output_dir = Path(output_dir)
    pending = tuple(
        descriptor
        for descriptor in selected
        if not (output_dir / f"{descriptor.task_id}.npz").exists()
    )
    model, runtime = _load_model(
        upstream_root,
        dataset_path,
        checkpoint_path,
        batch_size=batch_size,
        device=device,
    )

    dataset_path = Path(dataset_path)
    split_path = dataset_path / "test"
    inputs = np.load(split_path / "all__inputs.npy", mmap_mode="r")
    puzzle_identifiers = np.load(split_path / "all__puzzle_identifiers.npy", mmap_mode="r")
    puzzle_indices = np.load(split_path / "all__puzzle_indices.npy", mmap_mode="r")
    group_indices = np.load(split_path / "all__group_indices.npy", mmap_mode="r")
    identifiers = json.loads((dataset_path / "identifiers.json").read_text(encoding="utf-8"))
    test_puzzles = json.loads((dataset_path / "test_puzzles.json").read_text(encoding="utf-8"))

    with _import_upstream(upstream_root):
        from dataset.build_arc_dataset import (
            PuzzleIdSeparator,
            arc_grid_to_np,
            grid_hash,
            inverse_aug,
        )

    descriptors_by_group: dict[int, list[TaskDescriptor]] = {}
    for descriptor in pending:
        descriptors_by_group.setdefault(descriptor.group_index, []).append(descriptor)

    worker_started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    for group_index in sorted(descriptors_by_group):
        group_started = time.perf_counter()
        first_puzzle = int(group_indices[group_index])
        last_puzzle = int(group_indices[group_index + 1])
        first_example = int(puzzle_indices[first_puzzle])
        last_example = int(puzzle_indices[last_puzzle])
        example_indices = np.arange(first_example, last_example, dtype=np.int64)
        example_puzzles = np.searchsorted(puzzle_indices, example_indices, side="right") - 1
        original_identifiers = np.asarray(puzzle_identifiers[example_puzzles], dtype=np.int64)
        group_inputs = np.asarray(inputs[first_example:last_example], dtype=np.int32)

        emissions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for start in range(0, len(group_inputs), batch_size):
            stop = min(start + batch_size, len(group_inputs))
            batch_inputs_cpu = group_inputs[start:stop]
            batch_ids_original = original_identifiers[start:stop]
            batch_ids_model = (
                np.zeros_like(batch_ids_original)
                if puzzle_id_mode == "blank"
                else batch_ids_original
            )
            batch = {
                "inputs": torch.as_tensor(batch_inputs_cpu, device=device),
                "puzzle_identifiers": torch.as_tensor(
                    batch_ids_model, dtype=torch.int32, device=device
                ),
            }
            with torch.inference_mode(), torch.device(device):
                carry = model.initial_carry(batch)
                while True:
                    carry, outputs = model(carry, batch)
                    if bool(carry.halted.all()):
                        break
                predictions = outputs["logits"].argmax(dim=-1).cpu().numpy()
                q_values = outputs["q_halt_logits"].sigmoid().cpu().numpy()

            for identifier, input_sequence, prediction, q_value in zip(
                batch_ids_original,
                batch_inputs_cpu,
                predictions,
                q_values,
            ):
                augmented_name = identifiers[int(identifier)]
                original_name, inverse = inverse_aug(augmented_name)
                canonical_input = inverse(_crop(input_sequence))
                canonical_prediction = inverse(_crop(prediction))
                if not np.all((canonical_prediction >= 0) & (canonical_prediction <= 9)):
                    raise AssertionError(f"prediction outside ARC palette for {augmented_name}")
                emissions.setdefault((original_name, grid_hash(canonical_input)), []).append(
                    {
                        "grid": canonical_prediction,
                        "q_value": float(q_value),
                        "upstream_identifier": augmented_name,
                        "is_identity": PuzzleIdSeparator not in augmented_name,
                    }
                )
            del batch, carry, outputs

        group_seconds = time.perf_counter() - group_started
        for descriptor in descriptors_by_group[group_index]:
            puzzle = test_puzzles[descriptor.puzzle_id]
            pair = puzzle["test"][descriptor.pair_index]
            query = arc_grid_to_np(pair["input"])
            target = arc_grid_to_np(pair["output"])
            records = emissions.get((descriptor.puzzle_id, grid_hash(query)), [])
            if not records:
                raise RuntimeError(f"no canonical emissions produced for {descriptor.task_id}")
            supports = tuple(
                SupportPair(arc_grid_to_np(item["input"]), arc_grid_to_np(item["output"]))
                for item in puzzle["train"]
            )
            candidates = tuple(
                Candidate(
                    grid=record["grid"],
                    augmentation_index=index,
                    q_value=record["q_value"],
                    transform={"upstream_identifier": record["upstream_identifier"]},
                    is_identity=record["is_identity"],
                    is_invalid=record["grid"].shape == (0, 0),
                )
                for index, record in enumerate(records)
            )
            counts: dict[Any, int] = {}
            q_sums: dict[Any, float] = {}
            representatives: dict[Any, np.ndarray] = {}
            for candidate in candidates:
                key = grid_key(candidate.grid)
                counts[key] = counts.get(key, 0) + 1
                q_sums[key] = q_sums.get(key, 0.0) + candidate.q_value
                representatives.setdefault(key, candidate.grid)
            stream_winner = min(
                counts,
                key=lambda key: (
                    -counts[key],
                    -(q_sums[key] / counts[key]),
                    key,
                ),
            )
            stream_b1_correct = bool(
                np.array_equal(representatives[stream_winner], target)
            )
            orbit = TaskOrbit(
                task_id=descriptor.task_id,
                candidates=candidates,
                query_input=query,
                support_pairs=supports,
                target=target,
                metadata={
                    "task_index": descriptor.index,
                    "group_index": descriptor.group_index,
                    "puzzle_id_mode": puzzle_id_mode,
                    "group_inference_seconds": group_seconds,
                    "stream_b1_correct": stream_b1_correct,
                    "code_revision": code_revision,
                    "checkpoint_revision": checkpoint_revision,
                    "dataset_revision": dataset_revision,
                    **runtime,
                },
            )
            path = output_dir / f"{descriptor.task_id}.npz"
            _atomic_save(path, orbit)
            completed.append(
                {
                    "task_id": descriptor.task_id,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "candidates": len(candidates),
                    "group_inference_seconds": group_seconds,
                }
            )

    elapsed = time.perf_counter() - worker_started
    summary = {
        "requested": len(selected),
        "already_complete": len(selected) - len(pending),
        "completed": completed,
        "elapsed_seconds": elapsed,
        "tasks_per_second": len(completed) / elapsed if elapsed else math.inf,
        "runtime": runtime,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"worker_summary_{os.getpid()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable task-scoped TRM cache worker")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--puzzle-id-mode", choices=("normal", "blank"), default="normal")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = run_worker(
        upstream_root=args.upstream_root,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        task_ids_path=args.task_ids_json,
        batch_size=args.batch_size,
        device=args.device,
        puzzle_id_mode=args.puzzle_id_mode,
        code_revision=args.code_revision,
        checkpoint_revision=args.checkpoint_revision,
        dataset_revision=args.dataset_revision,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
