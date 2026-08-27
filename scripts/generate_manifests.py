from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RESULT_DIRECTORIES = (
    "arc1_main",
    "arc1_blank_ablation",
    "arc2_supporting",
    "reproduction_audit",
    "paper_summary",
    "release_audit",
    "provenance",
    "top_mode_diagnostic",
    "top2_policy_audit",
    "compute_matched_q_audit",
    "m1_beta_diagnostic",
    "discriminative_cell_dev",
    "risk_coverage",
    "risk_coverage_bootstrap",
    "final_freeze",
)
LOCAL_ONLY_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "checkpoints",
    "data",
    "dist",
    "external",
}


def _is_publishable(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return not any(
        part in LOCAL_ONLY_PARTS or part.endswith(".egg-info") for part in relative.parts
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _provenance(relative: Path) -> tuple[str, str]:
    text = relative.as_posix()
    if text.startswith("results/arc1_main/"):
        return text.replace("results/", "artifacts/results/", 1), "raw result"
    if text.startswith("results/arc1_blank_ablation/"):
        return text.replace("results/", "artifacts/results/", 1), "raw result"
    if text.startswith("results/arc2_supporting/"):
        return text.replace("results/", "artifacts/results/", 1), "raw result"
    if text.startswith("results/reproduction_audit/"):
        return text.replace("results/", "artifacts/results/", 1), "derived summary"
    if text == "results/paper_summary/paper_summary.json":
        return "artifacts/results/paper_summary.json", "derived summary"
    if text.startswith("results/release_audit/"):
        return text.replace("results/", "artifacts/", 1), "derived summary"
    if text.startswith("results/provenance/"):
        return "release-audit derivation from pinned metadata", "derived summary"
    diagnostic_groups = (
        "top_mode_diagnostic",
        "top2_policy_audit",
        "compute_matched_q_audit",
        "m1_beta_diagnostic",
        "discriminative_cell_dev",
        "risk_coverage",
        "risk_coverage_bootstrap",
    )
    if text.startswith(tuple(f"results/{name}/" for name in diagnostic_groups)):
        return text.replace("results/", "artifacts/results/", 1), "derived summary"
    if text.startswith("results/final_freeze/"):
        return "Phase-4 reconciliation of compact release artifacts", "derived summary"
    if text == "experiments/run_arc_cache.py":
        return (
            "proj/TinyRecursiveModels/experiments/run_arc1_normal_cache.py",
            "source code",
        )
    if text == "experiments/benchmark_arc_probe.py":
        return (
            "proj/TinyRecursiveModels/experiments/benchmark_arc1_probe.py",
            "source code",
        )
    if text.startswith("experiments/"):
        return f"proj/TinyRecursiveModels/{text}", "source code"
    if text.startswith(("src/", "scripts/", "tests/", "configs/")):
        return text, "source code"
    return "release packaging", "documentation"


def _source_path(source_root: Path | None, provenance: str) -> Path | None:
    if source_root is None or provenance.startswith("release-"):
        return None
    candidate = source_root / provenance
    return candidate if candidate.is_file() else None


def _entry(root: Path, path: Path, source_root: Path | None) -> dict[str, Any]:
    relative = path.relative_to(root)
    provenance, category = _provenance(relative)
    source = _source_path(source_root, provenance)
    entry: dict[str, Any] = {
        "relative_path": relative.as_posix(),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "provenance": provenance,
        "category": category,
    }
    if source is not None:
        entry["source_sha256"] = _sha256(source)
    return entry


def _write_result_manifests(root: Path) -> None:
    for name in RESULT_DIRECTORIES:
        directory = root / "results" / name
        entries = [
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        _write_json(directory / "artifact_manifest.json", entries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic release manifests")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--release-only",
        action="store_true",
        help="refresh only the complete publication inventory; leave frozen scientific manifests unchanged",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve() if args.source_root else None

    if args.release_only:
        artifact_manifest = json.loads(
            (root / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        artifact_entries = artifact_manifest["artifacts"]
    else:
        _write_result_manifests(root)
        scientific_paths = sorted(
            path for path in (root / "results").rglob("*") if path.is_file()
        )
        artifact_entries = [_entry(root, path, source_root) for path in scientific_paths]
        artifact_manifest = {
            "format_version": 1,
            "release_date": "2026-08-19",
            "artifacts": artifact_entries,
        }
        _write_json(root / "manifests" / "artifact_manifest.json", artifact_manifest)

    excluded = {root / "manifests" / "release_manifest.json"}
    all_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and _is_publishable(root, path) and path not in excluded
    )
    release_manifest = {
        "format_version": 1,
        "release_date": "2026-08-19",
        "self_exclusion": "manifests/release_manifest.json is excluded to avoid a self-hash",
        "scientific_configuration": {
            "trm_commit": "010206d1f0c25ebac0865f69e39c09969e6b896b",
            "hugging_face_checkpoint_revision": "55ced5dd59de74c52f53d47aa2898232b5a15b7a",
            "dataset_seed": 42,
            "num_aug": 1000,
            "analysis_seed": 20260807,
            "frozen_m1": {"definition": "distinct", "beta": 1.0},
            "frozen_m2": {
                "lambda": 0.05,
                "epsilon": 1e-6,
                "marginal_support": "emitted",
            },
            "inventory": {
                "arc1_normal": {"puzzles": 400, "descriptors": 419},
                "arc1_blank": {"puzzles": 400, "descriptors": 419},
                "arc2_normal": {"puzzles": 120, "descriptors": 172},
            },
        },
        "files": [_entry(root, path, source_root) for path in all_files],
    }
    _write_json(root / "manifests" / "release_manifest.json", release_manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "scientific_artifacts": len(artifact_entries),
                "release_files_excluding_self": len(all_files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
