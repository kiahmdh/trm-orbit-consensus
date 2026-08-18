from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from orbit_consensus.analysis_pipeline import load_orbits
from orbit_consensus.schema import grid_key
from orbit_consensus.supporting_analysis import _cache_fingerprint

IMMUTABLE_RESULT_DIRS = (
    "arc1_main",
    "arc1_blank_ablation",
    "arc2_supporting",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifests(results_root: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name in IMMUTABLE_RESULT_DIRS:
        directory = results_root / name
        manifest_path = directory / "artifact_manifest.json"
        manifest = _read_json(manifest_path)
        checked = 0
        for entry in manifest:
            path = directory / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(entry["bytes"]):
                raise AssertionError(f"size mismatch: {path}")
            if _sha256(path) != entry["sha256"]:
                raise AssertionError(f"SHA256 mismatch: {path}")
            checked += 1
        reports[name] = {"status": "passed", "files_checked": checked}
    return reports


def _validate_serialized_files(paths: list[Path]) -> dict[str, int]:
    json_count = 0
    csv_count = 0
    numeric_cells = 0
    for path in paths:
        if path.suffix == ".json":
            _read_json(path)
            json_count += 1
        elif path.suffix == ".csv":
            for row in _read_csv(path):
                for field, value in row.items():
                    if field in {"task_id", "puzzle_id"}:
                        continue
                    if value is None or not value.strip():
                        continue
                    if value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
                        raise AssertionError(f"non-finite value in {path}: {value}")
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(number):
                        raise AssertionError(f"non-finite value in {path}: {value}")
                    numeric_cells += 1
            csv_count += 1
    return {
        "json_files_parsed": json_count,
        "csv_files_parsed": csv_count,
        "finite_numeric_csv_cells": numeric_cells,
    }


def _puzzle_weighted(rows: list[dict[str, str]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["puzzle_id"]].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _mean_by_budget(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in ("pass_at_k", "majority_at_k", "selection_gap"):
            grouped[row["budget"]][field].append(float(row[field]))
    return {
        budget: {field: float(np.mean(values)) for field, values in metrics.items()}
        for budget, metrics in grouped.items()
    }


def _method_table(path: Path) -> dict[str, dict[str, float]]:
    return {
        row["method"]: {
            "rank1": float(row["rank1_accuracy"]),
            "top2": float(row["top2_accuracy"]),
            "coverage": float(row["coverage"]),
        }
        for row in _read_csv(path)
    }


def _compare_probe_predictions(left: Path, right: Path) -> dict[str, Any]:
    left_orbits = {orbit.task_id: orbit for orbit in load_orbits(left)}
    right_orbits = {orbit.task_id: orbit for orbit in load_orbits(right)}
    if left_orbits.keys() != right_orbits.keys():
        raise AssertionError(f"probe task mismatch: {left} vs {right}")
    grid_differences = 0
    q_differences = 0
    max_abs_q_difference = 0.0
    candidates = 0
    for task_id in sorted(left_orbits):
        left_candidates = left_orbits[task_id].candidates
        right_candidates = right_orbits[task_id].candidates
        if len(left_candidates) != len(right_candidates):
            raise AssertionError(f"probe candidate mismatch: {task_id}")
        for a, b in zip(left_candidates, right_candidates, strict=True):
            candidates += 1
            grid_differences += int(grid_key(a.grid) != grid_key(b.grid))
            delta = abs(a.q_value - b.q_value)
            q_differences += int(delta != 0.0)
            max_abs_q_difference = max(max_abs_q_difference, delta)
    return {
        "candidates": candidates,
        "grid_differences": grid_differences,
        "q_value_differences": q_differences,
        "max_abs_q_difference": max_abs_q_difference,
    }


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {observed}")


def _result_checks(root: Path) -> dict[str, Any]:
    results = root / "artifacts" / "results"
    cache = root / "artifacts" / "cache"
    main_report = _read_json(results / "arc1_main" / "analysis_report.json")
    blank_report = _read_json(results / "arc1_blank_ablation" / "analysis_report.json")
    arc2_report = _read_json(results / "arc2_supporting" / "analysis_report.json")
    reproduction = _read_json(results / "reproduction_audit" / "diagnostic_report.json")
    normal_run = _read_json(cache / "arc1_normal" / "run_report.json")
    blank_run = _read_json(cache / "arc1_blank" / "run_report.json")
    arc2_run = _read_json(cache / "arc2_normal" / "run_report.json")

    expected_counts = {
        "arc1_normal": (400, 419, 385_815, 92),
        "arc1_blank": (400, 419, 385_815, 720),
        "arc2_normal": (120, 172, 170_393, 188),
    }
    for label, report in (
        ("arc1_normal", normal_run),
        ("arc1_blank", blank_run),
        ("arc2_normal", arc2_run),
    ):
        observed = (
            report["completed_puzzles"],
            report["completed_descriptors"],
            report["candidate_count"],
            report["empty_invalid_prediction_count"],
        )
        if observed != expected_counts[label]:
            raise AssertionError(f"{label} inventory mismatch: {observed}")

    normal_methods = _method_table(results / "arc1_main" / "main_test_results.csv")
    blank_methods = _method_table(
        results / "arc1_blank_ablation" / "frozen_method_results.csv"
    )
    arc2_methods = _method_table(results / "arc2_supporting" / "frozen_method_results.csv")
    expected_normal_rank1 = {
        "B0": 0.2925,
        "B1": 0.4025,
        "M1": 0.4025,
        "M1+M2": 0.4125,
        "M1+M2+M3": 0.4125,
    }
    for method, expected in expected_normal_rank1.items():
        _assert_close(normal_methods[method]["rank1"], expected, f"ARC1 {method}")
    expected_blank_rank1 = {
        "B1": 0.04,
        "M1": 0.04,
        "M1+M2": 0.045,
        "M1+M2+M3": 0.05,
    }
    for method, expected in expected_blank_rank1.items():
        _assert_close(blank_methods[method]["rank1"], expected, f"blank {method}")
    for method in ("B1", "M1", "M1+M2", "M1+M2+M3"):
        _assert_close(arc2_methods[method]["rank1"], 7 / 240, f"ARC2 {method}")

    selection = _mean_by_budget(_read_csv(results / "arc1_main" / "selection_gap.csv"))
    _assert_close(selection["1000"]["pass_at_k"], 0.6175, "pass@1000")
    _assert_close(selection["1000"]["majority_at_k"], 0.4, "majority@1000")
    bootstrap = _read_csv(results / "arc1_main" / "paired_bootstrap_cis.csv")
    m2_rank1 = next(
        row for row in bootstrap if row["method"] == "M1+M2" and row["metric"] == "rank1"
    )
    if int(m2_rank1["resamples"]) != 10_000 or int(m2_rank1["puzzle_count"]) != 200:
        raise AssertionError("ARC1 M2 bootstrap is not 10,000-resample / 200-puzzle")

    split = _read_json(results / "arc1_main" / "committed_split.json")
    dev_puzzles = set(split["dev_puzzle_ids"])
    test_puzzles = set(split["test_puzzle_ids"])
    if dev_puzzles & test_puzzles or len(dev_puzzles) != 200 or len(test_puzzles) != 200:
        raise AssertionError("committed puzzle split is invalid")
    for task_id in split["dev_task_ids"]:
        if task_id.split("#", 1)[0] not in dev_puzzles:
            raise AssertionError("dev descriptor assigned outside its puzzle split")
    for task_id in split["test_task_ids"]:
        if task_id.split("#", 1)[0] not in test_puzzles:
            raise AssertionError("test descriptor assigned outside its puzzle split")
    if len(split["dev_task_ids"]) != 208 or len(split["test_task_ids"]) != 211:
        raise AssertionError("committed descriptor split is not 208/211")

    shapes = _read_csv(results / "arc1_main" / "shape_screening_diagnostics.csv")
    m3_active = _puzzle_weighted(shapes, "filter_active")
    m3_filtered = _puzzle_weighted(shapes, "filter_fraction")
    m3_allowed = _puzzle_weighted(shapes, "target_shape_allowed")
    _assert_close(m3_active, 0.86, "M3 active rate")
    _assert_close(m3_filtered, 0.008416097791097792, "M3 filter fraction")

    if normal_run["online_b1"] != normal_run["cache_b1"] != 0.4:
        raise AssertionError("ARC1 online/cache score mismatch")
    _assert_close(normal_run["online_b1"], 0.4, "ARC1 online B1")
    _assert_close(blank_run["online_b1"], 0.035, "blank retained B1")
    _assert_close(arc2_run["online_b1"], 7 / 240, "ARC2 online B1")
    _assert_close(arc2_run["cache_b1"], 7 / 240, "ARC2 cache B1")

    return {
        "cache_inventory": {
            label: {
                "puzzles": values[0],
                "descriptors": values[1],
                "candidates": values[2],
                "invalid_predictions": values[3],
            }
            for label, values in expected_counts.items()
        },
        "arc1_full": {
            "online_b1": normal_run["online_b1"],
            "cache_b1": normal_run["cache_b1"],
            "published_b1": normal_run["published_verification_b1"],
        },
        "arc1_test_methods": normal_methods,
        "arc1_selection_gap": selection,
        "arc1_h1": main_report["h1"],
        "arc1_m3": {
            "active_rate": m3_active,
            "filter_fraction": m3_filtered,
            "target_shape_allowed": m3_allowed,
        },
        "arc1_m2_bootstrap": m2_rank1,
        "blank": {
            "full_retained_b1": blank_run["cache_b1"],
            "test_methods": blank_methods,
            "coverage": blank_methods["B1"]["coverage"],
            "classification": blank_report["classification"],
            "diagnosis": reproduction["arc1_blank"]["verdict"],
        },
        "arc2": {
            "online_b1": arc2_run["online_b1"],
            "cache_b1": arc2_run["cache_b1"],
            "upstream": reproduction["arc2"]["pinned_upstream_evaluator"],
            "methods": arc2_methods,
            "classification": arc2_report["classification"],
        },
        "split": {
            "dev_puzzles": len(dev_puzzles),
            "dev_descriptors": len(split["dev_task_ids"]),
            "test_puzzles": len(test_puzzles),
            "test_descriptors": len(split["test_task_ids"]),
            "overlap": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit immutable release inputs CPU-only")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    expected_files = [
        root / "artifacts/results/arc1_main/main_test_results.csv",
        root / "artifacts/results/arc1_main/marginal_support_ablation.csv",
        root / "artifacts/results/arc1_main/compute_matched_results.csv",
        root / "artifacts/results/arc1_blank_ablation/frozen_method_results.csv",
        root / "artifacts/results/arc2_supporting/frozen_method_results.csv",
        root / "artifacts/results/reproduction_audit/diagnostic_report.json",
        root / "artifacts/results/paper_summary.json",
        root / "artifacts/release_audit/shape_hedged_second_attempt.csv",
    ]
    missing = [str(path.relative_to(root)) for path in expected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing expected release inputs: {missing}")

    scientific_paths = sorted((root / "artifacts/results").rglob("*.json"))
    scientific_paths += sorted((root / "artifacts/results").rglob("*.csv"))
    scientific_paths += sorted((root / "artifacts/release_audit").rglob("*.json"))
    scientific_paths += sorted((root / "artifacts/release_audit").rglob("*.csv"))
    serialization = _validate_serialized_files(scientific_paths)
    manifests = _verify_manifests(root / "artifacts/results")
    cache_fingerprints = {
        "arc1_normal": _cache_fingerprint(root / "artifacts/cache/arc1_normal"),
        "arc1_blank": _cache_fingerprint(root / "artifacts/cache/arc1_blank"),
        "arc2_normal": _cache_fingerprint(root / "artifacts/cache/arc2_normal"),
    }
    expected_fingerprints = {
        "arc1_normal": "6cb320cb9b0e45c114d95722654663d2b7e98b01027a1d4a647c21ea4ce60fa1",
        "arc1_blank": "9982f59fd5b952f6e80430adadebf047b107d357a18c04abf3cc30b204f0e5bd",
        "arc2_normal": "81f5f077adcc35770053c162486ea98c19953b1a94912f4dccf3998021043d20",
    }
    for label, expected in expected_fingerprints.items():
        if cache_fingerprints[label]["aggregate_sha256"] != expected:
            raise AssertionError(f"immutable cache fingerprint changed: {label}")

    benchmark_root = root / "artifacts/benchmarks/arc2_gpu"
    batch_comparisons = {
        "batch1_repeat": _compare_probe_predictions(
            benchmark_root / "batch1", benchmark_root / "batch1_repeat"
        ),
        "batch1_vs_batch2": _compare_probe_predictions(
            benchmark_root / "batch1", benchmark_root / "batch2"
        ),
        "batch2_vs_batch4": _compare_probe_predictions(
            benchmark_root / "batch2", benchmark_root / "batch4"
        ),
        "batch2_vs_batch8": _compare_probe_predictions(
            benchmark_root / "batch2", benchmark_root / "batch8"
        ),
    }
    if batch_comparisons["batch1_repeat"]["grid_differences"] != 0:
        raise AssertionError("ARC2 batch-size-1 repeat is not deterministic")
    if batch_comparisons["batch1_vs_batch2"]["grid_differences"] != 31:
        raise AssertionError("ARC2 batch1/batch2 difference is not 31/80")
    if any(
        batch_comparisons[label]["grid_differences"] != 0
        for label in ("batch2_vs_batch4", "batch2_vs_batch8")
    ):
        raise AssertionError("ARC2 batches 2/4/8 are not mutually grid-consistent")

    unfinished = sorted(
        str(path.relative_to(root))
        for cache_dir in (
            root / "artifacts/cache/arc1_normal",
            root / "artifacts/cache/arc1_blank",
            root / "artifacts/cache/arc2_normal",
        )
        for path in cache_dir.iterdir()
        if ".tmp" in path.name
    )
    if unfinished:
        raise AssertionError(f"unfinished cache artifacts: {unfinished}")

    report = {
        "status": "passed",
        "cpu_only": True,
        "expected_files": {"checked": len(expected_files), "missing": missing},
        "serialization": serialization,
        "artifact_manifests": manifests,
        "cache_fingerprints": cache_fingerprints,
        "unfinished_cache_artifacts": unfinished,
        "batch_size_reproducibility": batch_comparisons,
        "verified_results": _result_checks(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
