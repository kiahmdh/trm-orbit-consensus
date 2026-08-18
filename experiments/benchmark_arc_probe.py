from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get("TRM_ROOT", PROJECT_ROOT / "external" / "TinyRecursiveModels")
).resolve()
for import_root in (UPSTREAM_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
import yaml
from dataset.build_arc_dataset import (
    PuzzleIdSeparator,
    arc_grid_to_np,
    grid_hash,
    inverse_aug,
)
from evaluators.arc import _crop
from pretrain import PretrainConfig, create_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

from orbit_consensus.cache import load_task_orbit, save_task_orbit
from orbit_consensus.schema import Candidate, SupportPair, TaskOrbit, grid_key, make_task_id

GIB = 1024**3
OUTPUT_KEYS = ("inputs", "preds", "puzzle_identifiers", "q_halt_logits")


class MemoryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeExample:
    input_sequence: np.ndarray
    label_sequence: np.ndarray
    puzzle_identifier: int
    upstream_identifier: str


class CudaMemoryGuard:
    def __init__(
        self,
        *,
        abort_bytes: int,
        emergency_path: Path,
        poll_seconds: float = 0.05,
    ) -> None:
        self.abort_bytes = abort_bytes
        self.emergency_path = emergency_path
        self.poll_seconds = poll_seconds
        self.peak_allocated = 0
        self.peak_reserved = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.sample("shutdown")

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            self.peak_allocated = max(self.peak_allocated, allocated)
            self.peak_reserved = max(self.peak_reserved, reserved)
            if max(allocated, reserved) >= self.abort_bytes:
                payload = {
                    "status": "emergency_abort",
                    "reason": "new process approached its configured VRAM limit",
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "abort_bytes": self.abort_bytes,
                    "pid": os.getpid(),
                }
                self.emergency_path.parent.mkdir(parents=True, exist_ok=True)
                self.emergency_path.write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                os._exit(86)

    def sample(self, stage: str) -> dict[str, int | str]:
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        if max(allocated, reserved) >= self.abort_bytes:
            raise MemoryBudgetExceeded(
                f"VRAM budget approached at {stage}: allocated={allocated}, reserved={reserved}"
            )
        return {"stage": stage, "allocated_bytes": allocated, "reserved_bytes": reserved}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capped ARC-v1 production-path probe")
    parser.add_argument("--predictions", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allocator-cap-gib", type=float, default=4.5)
    parser.add_argument("--abort-gib", type=float, default=4.75)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmark" / "arc_probe_10",
    )
    return parser.parse_args()


def load_examples(dataset_path: Path, count: int) -> tuple[PuzzleDataset, int, list[ProbeExample]]:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[str(dataset_path)],
            global_batch_size=1,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    dataset._lazy_load_dataset()
    arrays = dataset._data["all"]  # type: ignore[index]
    group_indices = arrays["group_indices"]
    puzzle_indices = arrays["puzzle_indices"]
    puzzle_identifiers = arrays["puzzle_identifiers"]
    identifiers = json.loads((dataset_path / "identifiers.json").read_text(encoding="utf-8"))

    selected_group = next(
        group
        for group in range(len(group_indices) - 1)
        if int(group_indices[group + 1] - group_indices[group]) >= count
    )
    group_start = int(group_indices[selected_group])
    examples: list[ProbeExample] = []
    original_names: set[str] = set()
    for puzzle_index in range(group_start, group_start + count):
        example_index = int(puzzle_indices[puzzle_index])
        identifier = int(puzzle_identifiers[puzzle_index])
        upstream_identifier = str(identifiers[identifier])
        original_names.add(upstream_identifier.split(PuzzleIdSeparator)[0])
        examples.append(
            ProbeExample(
                input_sequence=np.array(arrays["inputs"][example_index], copy=True),
                label_sequence=np.array(arrays["labels"][example_index], copy=True),
                puzzle_identifier=identifier,
                upstream_identifier=upstream_identifier,
            )
        )
    if len(original_names) != 1:
        raise AssertionError(f"probe crossed original puzzle groups: {original_names}")
    return dataset, selected_group, examples


def make_config(
    config_path: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    batch_size: int = 1,
) -> PretrainConfig:
    with config_path.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    raw_config["data_paths"] = [str(dataset_path)]
    raw_config["data_paths_test"] = []
    raw_config["global_batch_size"] = batch_size
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["checkpoint_path"] = None
    raw_config["eval_save_outputs"] = list(OUTPUT_KEYS)
    return PretrainConfig(**raw_config)


def collate_one(dataset: PuzzleDataset, example: ProbeExample) -> dict[str, torch.Tensor]:
    return dataset._collate_batch(
        {
            "inputs": example.input_sequence[None, :],
            "labels": example.label_sequence[None, :],
            "puzzle_identifiers": np.asarray([example.puzzle_identifier], dtype=np.int32),
        }
    )


def tensor_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def build_probe_cache(
    *,
    dataset_path: Path,
    shard: dict[str, torch.Tensor],
    output_path: Path,
) -> dict[str, Any]:
    identifiers = json.loads((dataset_path / "identifiers.json").read_text(encoding="utf-8"))
    with (dataset_path / "test_puzzles.json").open(encoding="utf-8") as handle:
        test_puzzles = json.load(handle)

    q_values = shard["q_halt_logits"].to(torch.float64).sigmoid().numpy()
    candidates: list[Candidate] = []
    original_names: set[str] = set()
    input_hashes: set[str] = set()
    for index, (identifier, input_sequence, prediction, q_value) in enumerate(
        zip(
            shard["puzzle_identifiers"].numpy(),
            shard["inputs"].numpy(),
            shard["preds"].numpy(),
            q_values,
        )
    ):
        augmented_name = str(identifiers[int(identifier)])
        original_name, inverse = inverse_aug(augmented_name)
        original_names.add(original_name)
        input_hashes.add(grid_hash(inverse(_crop(input_sequence))))
        canonical_prediction = inverse(_crop(prediction))
        candidates.append(
            Candidate(
                grid=canonical_prediction,
                augmentation_index=index,
                q_value=float(q_value),
                transform={"upstream_identifier": augmented_name},
                is_identity=PuzzleIdSeparator not in augmented_name,
                is_invalid=canonical_prediction.shape == (0, 0),
            )
        )
    if len(original_names) != 1 or len(input_hashes) != 1:
        raise AssertionError(
            f"probe does not describe one task pair: names={original_names}, hashes={input_hashes}"
        )

    puzzle_id = next(iter(original_names))
    puzzle = test_puzzles[puzzle_id]
    supports = tuple(
        SupportPair(arc_grid_to_np(pair["input"]), arc_grid_to_np(pair["output"]))
        for pair in puzzle["train"]
    )
    pair = puzzle["test"][0]
    orbit = TaskOrbit(
        task_id=make_task_id(puzzle_id, 0),
        candidates=tuple(candidates),
        query_input=arc_grid_to_np(pair["input"]),
        support_pairs=supports,
        target=arc_grid_to_np(pair["output"]),
        metadata={"input_hash": next(iter(input_hashes)), "benchmark_probe": True},
    )
    save_task_orbit(output_path, orbit)
    reloaded = load_task_orbit(output_path)
    roundtrip_ok = (
        reloaded.task_id == orbit.task_id
        and len(reloaded.candidates) == len(orbit.candidates)
        and all(
            grid_key(left.grid) == grid_key(right.grid) and left.q_value == right.q_value
            for left, right in zip(orbit.candidates, reloaded.candidates)
        )
    )
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "roundtrip_ok": roundtrip_ok,
        "task_id": orbit.task_id,
        "candidate_count": len(orbit.candidates),
    }


def test_empty_prediction_roundtrip(
    *, output_path: Path, prediction_dtype: torch.dtype
) -> dict[str, Any]:
    empty_sequence = torch.zeros((1, 900), dtype=prediction_dtype)
    torch.save({"preds": empty_sequence}, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    raw_roundtrip_ok = torch.equal(empty_sequence, reloaded["preds"])
    canonical_shape = tuple(int(value) for value in _crop(reloaded["preds"][0].numpy()).shape)
    cache_accepts_empty = True
    cache_error = None
    try:
        Candidate(
            grid=np.empty((0, 0), dtype=np.uint8),
            augmentation_index=0,
            q_value=0.0,
            is_invalid=True,
        )
    except ValueError as exc:
        cache_accepts_empty = False
        cache_error = str(exc)
    return {
        "raw_shard_roundtrip_ok": raw_roundtrip_ok,
        "canonical_shape": canonical_shape,
        "orbit_cache_accepts_empty": cache_accepts_empty,
        "orbit_cache_error": cache_error,
    }


def main() -> int:
    args = parse_args()
    if not 10 <= args.predictions <= 20:
        raise ValueError("--predictions must be between 10 and 20")
    if not 0 < args.allocator_cap_gib < args.abort_gib < 5:
        raise ValueError("require 0 < allocator cap < abort threshold < 5 GiB")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU after CUDA_VISIBLE_DEVICES filtering; "
            f"found {torch.cuda.device_count()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / "step_0_all_preds.0"
    cache_path = args.output_dir / "probe_task_orbit.npz"
    report_path = args.output_dir / "benchmark_report.json"
    emergency_path = args.output_dir / "emergency_abort.json"
    model_log_path = args.output_dir / "model_setup.log"
    for path in (shard_path, cache_path, report_path, emergency_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite previous probe artifact: {path}")

    torch.cuda.set_device(args.gpu)
    total_memory = int(torch.cuda.get_device_properties(args.gpu).total_memory)
    allocator_cap_bytes = int(args.allocator_cap_gib * GIB)
    abort_bytes = int(args.abort_gib * GIB)
    torch.cuda.set_per_process_memory_fraction(allocator_cap_bytes / total_memory, args.gpu)
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    print(
        json.dumps(
            {
                "event": "probe_started",
                "pid": os.getpid(),
                "allocator_cap_bytes": allocator_cap_bytes,
                "abort_threshold_bytes": abort_bytes,
            }
        ),
        flush=True,
    )

    dataset, selected_group, examples = load_examples(args.dataset_path, args.predictions)
    config = make_config(args.config_path, args.dataset_path, args.checkpoint_path)
    memory_samples: list[dict[str, int | str]] = []
    model = None
    try:
        with CudaMemoryGuard(
            abort_bytes=abort_bytes,
            emergency_path=emergency_path,
        ) as guard:
            memory_samples.append(guard.sample("cuda_initialized"))
            model_start = time.perf_counter()
            with (
                model_log_path.open("w", encoding="utf-8") as model_log,
                contextlib.redirect_stdout(model_log),
            ):
                model, optimizers, optimizer_lrs = create_model(
                    config, dataset.metadata, rank=0, world_size=1
                )
            del optimizers, optimizer_lrs
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - model_start
            model_memory = guard.sample("model_loaded")
            memory_samples.append(model_memory)
            print(
                json.dumps({"event": "model_loaded", **model_memory}), flush=True
            )

            saved: dict[str, list[torch.Tensor]] = {key: [] for key in OUTPUT_KEYS}
            prediction_seconds: list[float] = []
            inference_start = time.perf_counter()
            for index, example in enumerate(examples):
                cpu_batch = collate_one(dataset, example)
                torch.cuda.synchronize()
                prediction_start = time.perf_counter()
                with torch.inference_mode():
                    batch = {key: value.cuda() for key, value in cpu_batch.items()}
                    with torch.device("cuda"):
                        carry = model.initial_carry(batch)  # type: ignore[union-attr]
                    while True:
                        carry, loss, metrics, preds, all_finish = model(
                            carry=carry,
                            batch=batch,
                            return_keys=set(OUTPUT_KEYS),
                        )
                        if all_finish:
                            break
                    for collection in (batch, preds):
                        for key, value in collection.items():
                            if key in saved:
                                saved[key].append(value.cpu())
                torch.cuda.synchronize()
                prediction_seconds.append(time.perf_counter() - prediction_start)
                prediction_memory = guard.sample(f"prediction_{index + 1}")
                memory_samples.append(prediction_memory)
                print(
                    json.dumps(
                        {
                            "event": "prediction_completed",
                            "prediction": index + 1,
                            "seconds": prediction_seconds[-1],
                            **prediction_memory,
                        }
                    ),
                    flush=True,
                )
                del batch, carry, loss, metrics, preds, all_finish, cpu_batch
            inference_seconds = time.perf_counter() - inference_start

            shard = {key: torch.cat(values, dim=0) for key, values in saved.items()}
            serialization_start = time.perf_counter()
            torch.save(shard, shard_path)
            serialization_seconds = time.perf_counter() - serialization_start
            reload_start = time.perf_counter()
            reloaded = torch.load(shard_path, map_location="cpu", weights_only=True)
            reload_seconds = time.perf_counter() - reload_start
            shard_roundtrip_ok = tensor_dict_equal(shard, reloaded)
            memory_samples.append(guard.sample("serialized_and_reloaded"))

            cache_result = build_probe_cache(
                dataset_path=args.dataset_path,
                shard=reloaded,
                output_path=cache_path,
            )
            empty_result = test_empty_prediction_roundtrip(
                output_path=args.output_dir / "empty_prediction_roundtrip.pt",
                prediction_dtype=reloaded["preds"].dtype,
            )

            report = {
                "status": "passed",
                "pid": os.getpid(),
                "selected_visible_gpu": args.gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(args.gpu),
                "batch_size": 1,
                "prediction_count": len(prediction_seconds),
                "selected_group": selected_group,
                "original_puzzle": examples[0].upstream_identifier.split(PuzzleIdSeparator)[0],
                "model_compiled": "DISABLE_COMPILE" not in os.environ,
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "mean_seconds_per_prediction": statistics.mean(prediction_seconds),
                "median_seconds_per_prediction": statistics.median(prediction_seconds),
                "prediction_seconds": prediction_seconds,
                "serialization_seconds": serialization_seconds,
                "reload_seconds": reload_seconds,
                "raw_shard": {
                    "path": str(shard_path),
                    "bytes": shard_path.stat().st_size,
                    "bytes_per_prediction": shard_path.stat().st_size / len(prediction_seconds),
                    "roundtrip_ok": shard_roundtrip_ok,
                    "keys": sorted(reloaded),
                },
                "orbit_cache": {
                    **cache_result,
                    "bytes_per_prediction": cache_result["bytes"] / len(prediction_seconds),
                },
                "empty_prediction": empty_result,
                "allocator_cap_bytes": allocator_cap_bytes,
                "abort_threshold_bytes": abort_bytes,
                "peak_allocated_bytes": max(
                    guard.peak_allocated, int(torch.cuda.max_memory_allocated(args.gpu))
                ),
                "peak_reserved_bytes": max(
                    guard.peak_reserved, int(torch.cuda.max_memory_reserved(args.gpu))
                ),
                "memory_samples": memory_samples,
                "config": {
                    "checkpoint_path": str(args.checkpoint_path),
                    "dataset_path": str(args.dataset_path),
                    "global_batch_size": config.global_batch_size,
                    "inference_mode": True,
                    "grad_enabled": torch.is_grad_enabled(),
                },
                "probe_examples": [asdict(example) | {"input_sequence": None, "label_sequence": None} for example in examples],
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 0
    except (torch.cuda.OutOfMemoryError, MemoryBudgetExceeded) as exc:
        aborted = {
            "status": "aborted",
            "reason": type(exc).__name__,
            "message": str(exc),
            "pid": os.getpid(),
            "allocator_cap_bytes": allocator_cap_bytes,
            "abort_threshold_bytes": abort_bytes,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.gpu)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(args.gpu)),
            "memory_samples": memory_samples,
        }
        report_path.write_text(json.dumps(aborted, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(aborted, indent=2), file=sys.stderr)
        return 3
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
