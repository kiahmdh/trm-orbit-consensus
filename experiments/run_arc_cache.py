from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get("TRM_ROOT", PROJECT_ROOT / "external" / "TinyRecursiveModels")
).resolve()
for import_root in (UPSTREAM_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
import torch
from benchmark_arc_probe import GIB, CudaMemoryGuard, make_config
from dataset.build_arc_dataset import (
    PuzzleIdSeparator,
    arc_grid_to_np,
    grid_hash,
    inverse_aug,
)
from evaluators.arc import _crop
from pretrain import create_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

from orbit_consensus.baselines import majority_vote
from orbit_consensus.cache import load_task_orbit, save_task_orbit
from orbit_consensus.schema import Candidate, SupportPair, TaskOrbit, grid_key, make_task_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capped, resumable ARC TRM inference/cache runner"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allocator-cap-gib", type=float, default=4.5)
    parser.add_argument("--abort-gib", type=float, default=4.75)
    parser.add_argument("--protected-pid", type=int, default=0)
    parser.add_argument(
        "--puzzle-id-mode", choices=("normal", "blank"), default="normal"
    )
    parser.add_argument("--probe-puzzles", type=int)
    parser.add_argument("--probe-augmentations", type=int, default=2)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=UPSTREAM_ROOT / "data" / "arc1concept-aug-1000",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=UPSTREAM_ROOT
        / "checkpoints"
        / "hf_trm"
        / "arc_v1_public"
        / "step_518071",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=UPSTREAM_ROOT
        / "checkpoints"
        / "hf_trm"
        / "arc_v1_public"
        / "all_config.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-b1", type=float)
    parser.add_argument("--expected-b1-tolerance", type=float, default=0.02)
    return parser.parse_args()


def process_snapshot(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return {"pid": pid, "exists": False}
    stat = (proc / "stat").read_text(encoding="utf-8").split()
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
        errors="replace"
    ).strip()
    return {
        "pid": pid,
        "exists": True,
        "start_time_ticks": stat[21],
        "state": stat[2],
        "command": command,
    }


def atomic_save(path: Path, orbit: TaskOrbit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        save_task_orbit(temporary, orbit)
        load_task_orbit(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arc_score(orbits: list[TaskOrbit]) -> float:
    by_puzzle: dict[str, list[float]] = defaultdict(list)
    for orbit in orbits:
        if orbit.target is None:
            raise ValueError(f"{orbit.task_id} has no target")
        correct = grid_key(majority_vote(orbit)[0].grid) == grid_key(orbit.target)
        by_puzzle[orbit.puzzle_id].append(float(correct))
    return float(np.mean([np.mean(values) for values in by_puzzle.values()]))


def select_probe_groups(
    puzzles: dict[str, Any], group_indices: np.ndarray, count: int, augmentations: int
) -> list[int]:
    selected: list[int] = []
    seen_shapes: set[tuple[int, int]] = set()
    for group_index, (_puzzle_id, puzzle) in enumerate(puzzles.items()):
        available = int(group_indices[group_index + 1] - group_indices[group_index])
        shape = tuple(np.asarray(puzzle["test"][0]["input"]).shape)
        if available >= augmentations and shape not in seen_shapes:
            selected.append(group_index)
            seen_shapes.add(shape)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"only found {len(selected)} groups with distinct query shapes and "
        f"at least {augmentations} augmentations"
    )


def group_example_indices(
    *,
    group_index: int,
    group_indices: np.ndarray,
    puzzle_indices: np.ndarray,
    probe_augmentations: int | None,
) -> np.ndarray:
    first_puzzle = int(group_indices[group_index])
    last_puzzle = int(group_indices[group_index + 1])
    if probe_augmentations is not None:
        selected_puzzles = np.arange(
            first_puzzle,
            min(first_puzzle + probe_augmentations, last_puzzle),
            dtype=np.int64,
        )
        return np.asarray([int(puzzle_indices[index]) for index in selected_puzzles])
    return np.arange(
        int(puzzle_indices[first_puzzle]),
        int(puzzle_indices[last_puzzle]),
        dtype=np.int64,
    )


def make_orbits(
    *,
    puzzle_id: str,
    puzzle: dict[str, Any],
    emissions: dict[tuple[str, str], list[dict[str, Any]]],
    group_index: int,
    group_seconds: float,
    runtime: dict[str, Any],
    probe: bool,
    puzzle_id_mode: str,
) -> list[TaskOrbit]:
    supports = tuple(
        SupportPair(arc_grid_to_np(item["input"]), arc_grid_to_np(item["output"]))
        for item in puzzle["train"]
    )
    orbits: list[TaskOrbit] = []
    for pair_index, pair in enumerate(puzzle["test"]):
        query = arc_grid_to_np(pair["input"])
        target = arc_grid_to_np(pair["output"])
        records = emissions.get((puzzle_id, grid_hash(query)), [])
        if not records:
            if probe and pair_index > 0:
                continue
            raise RuntimeError(f"no canonical emissions produced for {puzzle_id}#{pair_index}")
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
        preliminary = TaskOrbit(
            task_id=make_task_id(puzzle_id, pair_index),
            candidates=candidates,
            query_input=query,
            support_pairs=supports,
            target=target,
            metadata={
                "group_index": group_index,
                "puzzle_id_mode": puzzle_id_mode,
                "group_inference_seconds": group_seconds,
                "probe": probe,
                **runtime,
            },
        )
        online_key = grid_key(majority_vote(preliminary)[0].grid)
        orbits.append(
            TaskOrbit(
                task_id=preliminary.task_id,
                candidates=preliminary.candidates,
                query_input=preliminary.query_input,
                support_pairs=preliminary.support_pairs,
                target=preliminary.target,
                metadata={
                    **preliminary.metadata,
                    "stream_b1_key": [
                        online_key[0],
                        online_key[1],
                        online_key[2].hex(),
                    ],
                    "stream_b1_correct": online_key == grid_key(target),
                },
            )
        )
    return orbits


def main() -> int:
    args = parse_args()
    if not 0 < args.allocator_cap_gib < args.abort_gib < 10:
        raise ValueError("require 0 < allocator cap < abort threshold < 10 GiB")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.expected_b1_tolerance < 0:
        raise ValueError("--expected-b1-tolerance must be non-negative")
    if args.probe_puzzles is not None and not 5 <= args.probe_puzzles <= 10:
        raise ValueError("--probe-puzzles must be between 5 and 10")
    if args.probe_augmentations <= 0:
        raise ValueError("--probe-augmentations must be positive")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "run_report.json"
    progress_path = args.output_dir / "progress.json"
    emergency_path = args.output_dir / "emergency_abort.json"
    model_log_path = args.output_dir / "model_setup.log"
    probe = args.probe_puzzles is not None
    if probe and any(args.output_dir.glob("*.npz")):
        raise FileExistsError("refusing to mix a probe with existing task caches")

    protected_before = (
        process_snapshot(args.protected_pid)
        if args.protected_pid > 0
        else {"pid": None, "exists": False, "monitoring_disabled": True}
    )
    if args.protected_pid > 0 and not protected_before["exists"]:
        raise RuntimeError(f"protected PID {args.protected_pid} is not running")

    torch.cuda.set_device(args.gpu)
    total_memory = int(torch.cuda.get_device_properties(args.gpu).total_memory)
    allocator_cap_bytes = int(args.allocator_cap_gib * GIB)
    abort_bytes = int(args.abort_gib * GIB)
    torch.cuda.set_per_process_memory_fraction(allocator_cap_bytes / total_memory, args.gpu)
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(args.gpu)

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[str(args.dataset_path)],
            global_batch_size=args.batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    dataset._lazy_load_dataset()
    arrays = dataset._data["all"]
    group_indices = arrays["group_indices"]
    puzzle_indices = arrays["puzzle_indices"]
    puzzle_identifiers = arrays["puzzle_identifiers"]
    identifiers = json.loads((args.dataset_path / "identifiers.json").read_text())
    puzzles = json.loads((args.dataset_path / "test_puzzles.json").read_text())
    all_groups = list(range(len(group_indices) - 1))
    groups = (
        select_probe_groups(
            puzzles,
            group_indices,
            args.probe_puzzles,
            args.probe_augmentations,
        )
        if probe
        else all_groups
    )
    puzzle_items = list(puzzles.items())

    config = make_config(
        args.config_path,
        args.dataset_path,
        args.checkpoint_path,
        batch_size=args.batch_size,
    )
    run_started = time.perf_counter()
    model = None
    group_times: list[float] = []
    prediction_times: list[float] = []
    processed_predictions = 0
    processed_invalid = 0
    try:
        with CudaMemoryGuard(
            abort_bytes=abort_bytes,
            emergency_path=emergency_path,
        ) as guard:
            with model_log_path.open("w", encoding="utf-8") as model_log:
                stdout = sys.stdout
                try:
                    sys.stdout = model_log
                    model, optimizers, optimizer_lrs = create_model(
                        config, dataset.metadata, rank=0, world_size=1
                    )
                finally:
                    sys.stdout = stdout
            del optimizers, optimizer_lrs
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            torch.cuda.synchronize()
            runtime = {
                "gpu_name": torch.cuda.get_device_name(args.gpu),
                "batch_size": args.batch_size,
                "allocator_cap_bytes": allocator_cap_bytes,
                "emergency_threshold_bytes": abort_bytes,
                "model_compiled": "DISABLE_COMPILE" not in os.environ,
            }
            guard.sample("model_loaded")

            for ordinal, group_index in enumerate(groups, start=1):
                puzzle_id, puzzle = puzzle_items[group_index]
                expected_paths = [
                    args.output_dir / f"{make_task_id(puzzle_id, pair_index)}.npz"
                    for pair_index in range(len(puzzle["test"]))
                ]
                if not probe and all(path.exists() for path in expected_paths):
                    continue
                example_indices = group_example_indices(
                    group_index=group_index,
                    group_indices=group_indices,
                    puzzle_indices=puzzle_indices,
                    probe_augmentations=args.probe_augmentations if probe else None,
                )
                emissions: dict[tuple[str, str], list[dict[str, Any]]] = {}
                group_started = time.perf_counter()
                for batch_start in range(0, len(example_indices), args.batch_size):
                    batch_indices = example_indices[
                        batch_start : batch_start + args.batch_size
                    ]
                    batch_puzzle_indices = np.searchsorted(
                        puzzle_indices, batch_indices, side="right"
                    ) - 1
                    batch_identifiers = np.asarray(
                        puzzle_identifiers[batch_puzzle_indices], dtype=np.int32
                    )
                    cpu_batch = dataset._collate_batch(
                        {
                            "inputs": np.asarray(arrays["inputs"][batch_indices]),
                            "labels": np.asarray(arrays["labels"][batch_indices]),
                            "puzzle_identifiers": batch_identifiers,
                        }
                    )
                    if args.puzzle_id_mode == "blank":
                        cpu_batch["puzzle_identifiers"].fill_(
                            dataset.metadata.blank_identifier_id
                        )
                    torch.cuda.synchronize()
                    prediction_started = time.perf_counter()
                    with torch.inference_mode():
                        batch = {key: value.cuda() for key, value in cpu_batch.items()}
                        with torch.device("cuda"):
                            carry = model.initial_carry(batch)
                        while True:
                            carry, loss, metrics, outputs, all_finish = model(
                                carry=carry,
                                batch=batch,
                                return_keys={"preds", "q_halt_logits"},
                            )
                            if all_finish:
                                break
                        predictions = outputs["preds"].cpu().numpy()
                        q_values = outputs["q_halt_logits"].sigmoid().cpu().numpy()
                    torch.cuda.synchronize()
                    batch_seconds = time.perf_counter() - prediction_started
                    per_prediction_seconds = batch_seconds / len(batch_indices)
                    prediction_times.extend(
                        [per_prediction_seconds] * len(batch_indices)
                    )
                    guard.sample(
                        f"prediction_{processed_predictions + len(batch_indices)}"
                    )

                    for example_index, identifier, prediction, q_value in zip(
                        batch_indices,
                        batch_identifiers,
                        predictions,
                        q_values,
                        strict=True,
                    ):
                        augmented_name = str(identifiers[int(identifier)])
                        original_name, inverse = inverse_aug(augmented_name)
                        input_sequence = np.asarray(arrays["inputs"][example_index])
                        canonical_input = inverse(_crop(input_sequence))
                        canonical_prediction = inverse(_crop(prediction))
                        if canonical_prediction.shape == (0, 0):
                            processed_invalid += 1
                        emissions.setdefault(
                            (original_name, grid_hash(canonical_input)), []
                        ).append(
                            {
                                "grid": canonical_prediction,
                                "q_value": float(q_value),
                                "upstream_identifier": augmented_name,
                                "is_identity": PuzzleIdSeparator not in augmented_name,
                            }
                        )
                    processed_predictions += len(batch_indices)
                    del batch, carry, loss, metrics, outputs, all_finish, cpu_batch

                torch.cuda.synchronize()
                group_seconds = time.perf_counter() - group_started
                group_times.append(group_seconds)
                for orbit in make_orbits(
                    puzzle_id=puzzle_id,
                    puzzle=puzzle,
                    emissions=emissions,
                    group_index=group_index,
                    group_seconds=group_seconds,
                    runtime=runtime,
                    probe=probe,
                    puzzle_id_mode=args.puzzle_id_mode,
                ):
                    path = args.output_dir / f"{orbit.task_id}.npz"
                    atomic_save(path, orbit)
                    reloaded = load_task_orbit(path)
                    online = orbit.metadata["stream_b1_key"]
                    cache_key = grid_key(majority_vote(reloaded)[0].grid)
                    if online != [cache_key[0], cache_key[1], cache_key[2].hex()]:
                        raise AssertionError(f"online/cache B1 mismatch for {orbit.task_id}")

                progress = {
                    "mode": "probe" if probe else "full",
                    "puzzle_id_mode": args.puzzle_id_mode,
                    "batch_size": args.batch_size,
                    "completed_groups_this_process": len(group_times),
                    "requested_groups": len(groups),
                    "last_group": group_index,
                    "last_puzzle": puzzle_id,
                    "predictions_this_process": processed_predictions,
                    "invalid_predictions_this_process": processed_invalid,
                    "peak_allocated_bytes": max(
                        guard.peak_allocated, int(torch.cuda.max_memory_allocated(args.gpu))
                    ),
                    "peak_reserved_bytes": max(
                        guard.peak_reserved, int(torch.cuda.max_memory_reserved(args.gpu))
                    ),
                }
                temporary_progress = progress_path.with_suffix(".json.tmp")
                temporary_progress.write_text(json.dumps(progress, indent=2) + "\n")
                os.replace(temporary_progress, progress_path)
                print(json.dumps({"event": "group_complete", **progress}), flush=True)

            cached_paths = sorted(args.output_dir.glob("*.npz"))
            reloaded_orbits = [load_task_orbit(path) for path in cached_paths]
            online_flags = [
                float(bool(orbit.metadata["stream_b1_correct"])) for orbit in reloaded_orbits
            ]
            cache_flags = [
                float(
                    grid_key(majority_vote(orbit)[0].grid)
                    == grid_key(orbit.target)
                )
                for orbit in reloaded_orbits
            ]
            if online_flags != cache_flags:
                raise AssertionError("online and cache-recomputed B1 differ")
            elapsed = time.perf_counter() - run_started
            protected_after = (
                process_snapshot(args.protected_pid)
                if args.protected_pid > 0
                else {"pid": None, "exists": False, "monitoring_disabled": True}
            )
            protected_unchanged = args.protected_pid <= 0 or (
                protected_after.get("exists")
                and protected_after.get("start_time_ticks")
                == protected_before.get("start_time_ticks")
                and protected_after.get("command") == protected_before.get("command")
            )
            report = {
                "status": "passed",
                "mode": "probe" if probe else "full",
                "puzzle_id_mode": args.puzzle_id_mode,
                "selected_gpu": args.gpu,
                "batch_size": args.batch_size,
                "protected_process_before": protected_before,
                "protected_process_after": protected_after,
                "protected_process_unchanged": bool(protected_unchanged),
                "completed_descriptors": len(reloaded_orbits),
                "completed_puzzles": len({orbit.puzzle_id for orbit in reloaded_orbits}),
                "candidate_count": sum(len(orbit.candidates) for orbit in reloaded_orbits),
                "empty_invalid_prediction_count": sum(
                    candidate.is_invalid
                    for orbit in reloaded_orbits
                    for candidate in orbit.candidates
                ),
                "predictions_this_process": processed_predictions,
                "wall_seconds": elapsed,
                "mean_prediction_seconds": (
                    statistics.mean(prediction_times) if prediction_times else None
                ),
                "median_prediction_seconds": (
                    statistics.median(prediction_times) if prediction_times else None
                ),
                "predictions_per_second": (
                    processed_predictions / sum(prediction_times)
                    if prediction_times
                    else None
                ),
                "peak_allocated_bytes": max(
                    guard.peak_allocated, int(torch.cuda.max_memory_allocated(args.gpu))
                ),
                "peak_reserved_bytes": max(
                    guard.peak_reserved, int(torch.cuda.max_memory_reserved(args.gpu))
                ),
                "cache_bytes": sum(path.stat().st_size for path in cached_paths),
                "online_b1": arc_score(reloaded_orbits),
                "cache_b1": arc_score(reloaded_orbits),
                "published_verification_b1": args.expected_b1,
                "reproduction_result": (
                    "pass"
                    if not probe
                    and args.puzzle_id_mode == "normal"
                    and args.expected_b1 is not None
                    and abs(arc_score(reloaded_orbits) - args.expected_b1)
                    <= args.expected_b1_tolerance
                    else "blank_id_ablation_pass"
                    if not probe
                    and args.puzzle_id_mode == "blank"
                    and arc_score(reloaded_orbits) == 0.0
                    else "not_applicable_probe"
                    if probe
                    else "not_configured"
                    if args.expected_b1 is None
                    else "fail"
                ),
            }
            if not protected_unchanged:
                raise RuntimeError("protected process identity changed during this run")
            temporary_report = report_path.with_suffix(".json.tmp")
            temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary_report, report_path)
            print(json.dumps({"event": "run_complete", **report}), flush=True)
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
