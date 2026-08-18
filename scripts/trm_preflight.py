from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# This command is intentionally CPU-only, even if the caller forgets to hide GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from orbit_consensus.upstream import DatasetBuildSpec, FrozenTRMAdapter

DEFAULT_UPSTREAM_REVISION = "010206d1f0c25ebac0865f69e39c09969e6b896b"
DEFAULT_CHECKPOINT_REVISION = (
    "sha256:53689643ad1606d7c22c758f8af0a71b3b66275dea074f214d2f1048d9a01fb0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CPU-only gates required before TRM orbit inference."
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=WORKSPACE_ROOT / "proj" / "TinyRecursiveModels",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=WORKSPACE_ROOT
        / "proj"
        / "TinyRecursiveModels"
        / "data"
        / "arc1concept-aug-1000",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=WORKSPACE_ROOT
        / "proj"
        / "TinyRecursiveModels"
        / "checkpoints"
        / "hf_trm"
        / "arc_v1_public"
        / "step_518071",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "cache" / "arc1",
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "preflight" / "arc1",
    )
    parser.add_argument("--code-revision", default=DEFAULT_UPSTREAM_REVISION)
    parser.add_argument("--checkpoint-revision", default=DEFAULT_CHECKPOINT_REVISION)
    parser.add_argument("--dataset-revision", default=DEFAULT_UPSTREAM_REVISION)
    parser.add_argument("--transform-trials", type=int, default=4096)
    return parser.parse_args()


def validate_build_manifest(dataset_path: Path, code_revision: str) -> dict[str, object]:
    manifest_path = dataset_path / "build_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected = {
        "subsets": ["training", "evaluation", "concept"],
        "test_set_name": "evaluation",
        "seed": 42,
        "num_aug": 1000,
        "code_revision": code_revision,
    }
    observed = {
        "subsets": manifest.get("recipe", {}).get("subsets"),
        "test_set_name": manifest.get("recipe", {}).get("test_set_name"),
        "seed": manifest.get("seed"),
        "num_aug": manifest.get("num_aug"),
        "code_revision": manifest.get("code_revision"),
    }
    if observed != expected:
        raise AssertionError(
            f"dataset build manifest does not match the strict ARC-v1 recipe: {observed}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    if args.transform_trials < 8:
        raise ValueError("--transform-trials must be at least 8")

    manifest = validate_build_manifest(args.dataset_path, args.code_revision)
    adapter = FrozenTRMAdapter(
        upstream_root=args.upstream_root,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        prediction_shards=(),
        cache_dir=args.cache_dir,
        code_revision=args.code_revision,
        checkpoint_revision=args.checkpoint_revision,
        dataset_revision=args.dataset_revision,
        build_spec=DatasetBuildSpec(),
    )
    identifier_gate = adapter.verify_identifier_alignment()
    transform_gate = adapter.verify_transform_round_trips(trials=args.transform_trials)
    provenance_path = adapter.write_provenance(args.provenance_dir)

    result = {
        "status": "passed",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dataset_manifest": manifest,
        "identifier_gate": asdict(identifier_gate),
        "transform_gate": asdict(transform_gate),
        "provenance_path": str(provenance_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
