from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import load_task_orbit, save_task_orbit
from .schema import Candidate, SupportPair, TaskOrbit, make_task_id


@dataclass(frozen=True)
class DatasetBuildSpec:
    subsets: tuple[str, ...] = ("training", "evaluation", "concept")
    test_set_name: str = "evaluation"
    seed: int = 42
    num_aug: int = 1000
    expected_identifier_count: int = 876_406


@dataclass(frozen=True)
class IdentifierGate:
    identifier_count: int
    checkpoint_embedding_rows: int


@dataclass(frozen=True)
class TransformGate:
    trials: int
    dihedral_ids_seen: tuple[int, ...]
    translated_cases: int


class FrozenTRMAdapter:
    """Adapter over an exact, pinned upstream TRM checkout.

    Importing this module is CPU-only. Torch and upstream evaluator modules are loaded
    lazily inside gate/inference methods. `infer_orbits` consumes prediction shards
    produced by upstream evaluation and never launches a GPU process itself.
    """

    _PUZZLE_EMBEDDING_KEY = "_orig_mod.model.inner.puzzle_emb.weights"

    def __init__(
        self,
        *,
        upstream_root: Path,
        dataset_path: Path,
        checkpoint_path: Path,
        prediction_shards: Iterable[Path],
        cache_dir: Path,
        code_revision: str,
        checkpoint_revision: str,
        dataset_revision: str,
        input_file_prefix: str = "kaggle/combined/arc-agi",
        build_spec: DatasetBuildSpec | None = None,
    ) -> None:
        self.upstream_root = Path(upstream_root).resolve()
        self.dataset_path = Path(dataset_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.prediction_shards = tuple(Path(path).resolve() for path in prediction_shards)
        self.cache_dir = Path(cache_dir).resolve()
        self.code_revision = code_revision
        self.checkpoint_revision = checkpoint_revision
        self.dataset_revision = dataset_revision
        self.input_file_prefix = input_file_prefix
        self.build_spec = build_spec if build_spec is not None else DatasetBuildSpec()
        self._identifier_gate: IdentifierGate | None = None
        self._transform_gate: TransformGate | None = None
        self._assert_pinned_checkout()

    def _assert_pinned_checkout(self) -> None:
        result = subprocess.run(
            ["git", "-C", str(self.upstream_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = result.stdout.strip()
        if actual != self.code_revision:
            raise ValueError(
                f"upstream checkout is {actual}, expected pinned revision {self.code_revision}"
            )

    @contextmanager
    def _upstream_import_path(self) -> Iterator[None]:
        root = str(self.upstream_root)
        sys.path.insert(0, root)
        try:
            yield
        finally:
            if sys.path[0] == root:
                sys.path.pop(0)
            else:
                sys.path.remove(root)

    def dataset_build_command(self) -> tuple[str, ...]:
        spec = self.build_spec
        return (
            sys.executable,
            "-m",
            "dataset.build_arc_dataset",
            "--input-file-prefix",
            self.input_file_prefix,
            "--output-dir",
            str(self.dataset_path),
            "--subsets",
            *spec.subsets,
            "--test-set-name",
            spec.test_set_name,
            "--seed",
            str(spec.seed),
            "--num-aug",
            str(spec.num_aug),
        )

    def build_dataset(self) -> None:
        """Explicit CPU preprocessing entry point; never called implicitly by inference."""
        subprocess.run(
            self.dataset_build_command(),
            cwd=self.upstream_root,
            check=True,
        )

    def _checkpoint_embedding_rows(self) -> int:
        import torch  # Lazy: loading the adapter itself never initializes Torch/CUDA.

        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        try:
            state_dict = torch.load(
                self.checkpoint_path, weights_only=True, mmap=True, **load_kwargs
            )
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, **load_kwargs)
        key = self._PUZZLE_EMBEDDING_KEY
        if key not in state_dict:
            matches = [name for name in state_dict if name.endswith("puzzle_emb.weights")]
            if len(matches) != 1:
                raise KeyError(f"could not uniquely locate puzzle embeddings in checkpoint: {matches}")
            key = matches[0]
        return int(state_dict[key].shape[0])

    def verify_identifier_alignment(self, *, build_if_missing: bool = False) -> IdentifierGate:
        identifiers_path = self.dataset_path / "identifiers.json"
        if not identifiers_path.exists():
            if not build_if_missing:
                raise FileNotFoundError(
                    f"{identifiers_path} is missing; run build_dataset() explicitly first"
                )
            self.build_dataset()
        with identifiers_path.open(encoding="utf-8") as handle:
            identifier_count = len(json.load(handle))
        expected = self.build_spec.expected_identifier_count
        if identifier_count != expected:
            raise AssertionError(
                f"identifier count {identifier_count} does not match expected {expected}"
            )
        embedding_rows = self._checkpoint_embedding_rows()
        if embedding_rows != expected:
            raise AssertionError(
                f"checkpoint puzzle embeddings have {embedding_rows} rows; expected {expected}"
            )
        self._identifier_gate = IdentifierGate(identifier_count, embedding_rows)
        return self._identifier_gate

    def verify_transform_round_trips(
        self, *, trials: int = 4096, seed: int = 42
    ) -> TransformGate:
        """Exercise upstream augmentation, evaluation cropping, and translation.

        Evaluation grids are top-left aligned, so upstream ``_crop`` is tested on
        that exact path. Training translation is tested separately by locating the
        emitted non-padding rectangle before applying upstream ``inverse_aug``.
        """
        if trials < 8:
            raise ValueError("at least eight trials are required to cover D4")
        with self._upstream_import_path():
            from dataset.build_arc_dataset import (
                PuzzleIdSeparator,
                aug,
                inverse_aug,
                np_grid_to_seq_translational_augment,
            )
            from evaluators.arc import _crop

        previous_state = np.random.get_state()
        np.random.seed(seed)
        seen: set[int] = set()
        translated_cases = 0
        try:
            for trial in range(trials):
                height = 1 + trial % 11
                width = 1 + (trial * 7) % 13
                canonical = (
                    np.arange(height * width, dtype=np.uint8).reshape(height, width) % 10
                )
                augmented_name, forward = aug(f"roundtrip-{trial}")
                transformed = forward(canonical)
                _, eval_padded = np_grid_to_seq_translational_augment(
                    transformed, transformed, do_translation=False
                )
                original_name, inverse = inverse_aug(augmented_name)
                eval_round_tripped = inverse(_crop(eval_padded))
                if not np.array_equal(canonical, eval_round_tripped):
                    raise AssertionError(f"evaluation round trip failed at trial {trial}")

                _, translated_padded = np_grid_to_seq_translational_augment(
                    transformed, transformed, do_translation=True
                )
                translated_grid = translated_padded.reshape(30, 30)
                content = np.argwhere(translated_grid >= 2)
                if len(content) == 0:
                    raise AssertionError(f"translated grid is empty at trial {trial}")
                start = content.min(axis=0)
                stop = content.max(axis=0) + 1
                if tuple(start) != (0, 0):
                    translated_cases += 1
                translated_crop = (
                    translated_grid[start[0] : stop[0], start[1] : stop[1]] - 2
                ).astype(np.uint8)
                translated_round_tripped = inverse(translated_crop)
                if original_name != f"roundtrip-{trial}" or not np.array_equal(
                    canonical, translated_round_tripped
                ):
                    raise AssertionError(f"translated round trip failed at trial {trial}")

                transform_token = augmented_name.split(PuzzleIdSeparator)[-2]
                seen.add(int(transform_token[1:]))
        finally:
            np.random.set_state(previous_state)
        if seen != set(range(8)):
            raise AssertionError(f"round-trip sample did not cover all D4 elements: {seen}")
        if translated_cases == 0:
            raise AssertionError("round-trip sample did not exercise translated padding")
        self._transform_gate = TransformGate(trials, tuple(sorted(seen)), translated_cases)
        return self._transform_gate

    def _require_gates(self) -> None:
        if self._identifier_gate is None or self._transform_gate is None:
            raise RuntimeError(
                "identifier alignment and transform round-trip gates must pass before inference"
            )

    def infer_orbits(
        self, task_ids: Iterable[str], *, augmentations: int = 1000
    ) -> Iterator[TaskOrbit]:
        """Canonicalize saved upstream evaluator outputs and persist each pair immediately."""
        self._require_gates()
        if augmentations != self.build_spec.num_aug:
            raise ValueError(
                f"adapter is pinned to num_aug={self.build_spec.num_aug}; got {augmentations}"
            )
        requested = set(task_ids)
        cached: dict[str, TaskOrbit] = {}
        for task_id in tuple(requested):
            cache_path = self.cache_dir / f"{task_id}.npz"
            if cache_path.exists():
                cached[task_id] = load_task_orbit(cache_path)
        if requested and requested == set(cached):
            yield from (cached[task_id] for task_id in sorted(cached))
            return

        import torch  # Lazy and CPU-only here: shards are loaded with map_location='cpu'.

        with self._upstream_import_path():
            from dataset.build_arc_dataset import (
                PuzzleIdSeparator,
                arc_grid_to_np,
                grid_hash,
                inverse_aug,
            )
            from dataset.common import PuzzleDatasetMetadata
            from evaluators.arc import ARC, _crop

        with (self.dataset_path / "test" / "dataset.json").open(encoding="utf-8") as handle:
            eval_metadata = PuzzleDatasetMetadata(**json.load(handle))
        evaluator = ARC(str(self.dataset_path), eval_metadata, aggregated_voting=False)
        evaluator.begin_eval()
        emissions: dict[tuple[str, str], list[dict[str, Any]]] = {}

        required = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
        for shard_path in sorted(self.prediction_shards):
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            missing = required - set(shard)
            if missing:
                raise KeyError(f"prediction shard {shard_path} is missing {sorted(missing)}")
            evaluator.update_batch(shard, shard)
            q_values = shard["q_halt_logits"].to(torch.float64).sigmoid().cpu()
            mask = shard["puzzle_identifiers"] != eval_metadata.blank_identifier_id
            identifiers = shard["puzzle_identifiers"][mask].cpu().numpy()
            inputs = shard["inputs"][mask].cpu().numpy()
            predictions = shard["preds"][mask].cpu().numpy()
            q_array = q_values[mask].numpy()
            for identifier, input_sequence, prediction, q_value in zip(
                identifiers, inputs, predictions, q_array
            ):
                augmented_name = evaluator.identifier_map[int(identifier)]
                original_name, inverse = inverse_aug(augmented_name)
                input_hash = grid_hash(inverse(_crop(input_sequence)))
                canonical_prediction = inverse(_crop(prediction))
                prediction_hash = grid_hash(canonical_prediction)
                evaluator_grid = evaluator._local_hmap[prediction_hash]
                if not np.array_equal(evaluator_grid, canonical_prediction):
                    raise AssertionError("adapter/evaluator canonical grids disagree")
                emissions.setdefault((original_name, input_hash), []).append(
                    {
                        "grid": evaluator_grid,
                        "q_value": float(q_value),
                        "upstream_identifier": augmented_name,
                        "is_identity": PuzzleIdSeparator not in augmented_name,
                    }
                )

        with (self.dataset_path / "test_puzzles.json").open(encoding="utf-8") as handle:
            test_puzzles = json.load(handle)
        produced: set[str] = set(cached)
        for puzzle_id, puzzle in test_puzzles.items():
            supports = tuple(
                SupportPair(arc_grid_to_np(pair["input"]), arc_grid_to_np(pair["output"]))
                for pair in puzzle["train"]
            )
            for pair_index, pair in enumerate(puzzle["test"]):
                task_id = make_task_id(puzzle_id, pair_index)
                if requested and task_id not in requested:
                    continue
                if task_id in cached:
                    yield cached[task_id]
                    continue
                query = arc_grid_to_np(pair["input"])
                target = arc_grid_to_np(pair["output"])
                input_hash = grid_hash(query)
                records = emissions.get((puzzle_id, input_hash), [])
                if not records:
                    raise RuntimeError(f"no emissions found for {task_id} ({input_hash})")
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
                orbit = TaskOrbit(
                    task_id=task_id,
                    candidates=candidates,
                    query_input=query,
                    support_pairs=supports,
                    target=target,
                    metadata={
                        "input_hash": input_hash,
                        "code_revision": self.code_revision,
                        "checkpoint_revision": self.checkpoint_revision,
                        "dataset_revision": self.dataset_revision,
                    },
                )
                save_task_orbit(self.cache_dir / f"{task_id}.npz", orbit)
                produced.add(task_id)
                yield orbit
        missing_tasks = requested - produced
        if missing_tasks:
            raise KeyError(f"requested task IDs were not produced: {sorted(missing_tasks)}")

    def write_provenance(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "code_revision": self.code_revision,
            "checkpoint_revision": self.checkpoint_revision,
            "dataset_revision": self.dataset_revision,
            "build_spec": asdict(self.build_spec),
            "identifier_count": (
                self._identifier_gate.identifier_count if self._identifier_gate else None
            ),
            "checkpoint_embedding_rows": (
                self._identifier_gate.checkpoint_embedding_rows
                if self._identifier_gate
                else None
            ),
            "transform_gate": asdict(self._transform_gate) if self._transform_gate else None,
            "prediction_shards": [str(path) for path in self.prediction_shards],
        }
        destination = output_dir / "trm_provenance.json"
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination
