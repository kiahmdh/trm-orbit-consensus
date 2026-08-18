from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from orbit_consensus.analysis_pipeline import (
    _shape_diagnostics,
    _shape_hedge_rows,
    load_orbits,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("shape-hedge audit produced no rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _puzzle_mean(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        puzzle_id = str(row["task_id"]).split("#", 1)[0]
        grouped[puzzle_id].append(float(row[field]))
    return float(sum(sum(values) / len(values) for values in grouped.values()) / len(grouped))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen, CPU-only proposal section 4.6 shape-hedge audit"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    split = json.loads((args.main_results / "committed_split.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (args.main_results / "frozen_hyperparameters.json").read_text(encoding="utf-8")
    )
    if frozen.get("selection_split") != "dev":
        raise AssertionError("hyperparameters were not selected on the committed dev split")
    parameters = frozen["parameters"]
    expected_parameters = {
        "definition": "distinct",
        "beta": 1.0,
        "interpolation": 0.05,
        "epsilon": 1e-6,
        "marginal_support": "emitted",
    }
    if parameters != expected_parameters:
        raise AssertionError(f"frozen parameter mismatch: {parameters}")

    orbits = load_orbits(args.cache)
    puzzle_ids = {orbit.puzzle_id for orbit in orbits}
    if len(orbits) != 419 or len(puzzle_ids) != 400:
        raise AssertionError("ARC1 cache must contain 419 descriptors and 400 puzzles")
    dev_ids = tuple(split["dev_puzzle_ids"])
    test_ids = tuple(split["test_puzzle_ids"])
    if set(dev_ids) & set(test_ids) or set(dev_ids) | set(test_ids) != puzzle_ids:
        raise AssertionError("committed split is not a disjoint puzzle-level partition")
    dev = tuple(orbit for orbit in orbits if orbit.puzzle_id in set(dev_ids))
    if len(dev) != 208 or len(dev_ids) != 200:
        raise AssertionError("committed dev inventory must be 200 puzzles / 208 descriptors")
    if set(split["dev_task_ids"]) != {orbit.task_id for orbit in dev}:
        raise AssertionError("dev task IDs do not match the committed split")

    rows = _shape_hedge_rows(dev, parameters, "dev")
    if {row["method"] for row in rows} != {"M1+M2", "M1+M2+M3"}:
        raise AssertionError("shape-hedge methods are incomplete")
    numeric_values = (
        value for row in rows for value in row.values() if isinstance(value, float)
    )
    if any(not math.isfinite(float(value)) for value in numeric_values):
        raise AssertionError("non-finite shape-hedge result")
    csv_path = args.output / "shape_hedged_second_attempt.csv"
    _write_csv(csv_path, rows)

    diagnostics = _shape_diagnostics(dev, parameters)
    report = {
        "status": "passed",
        "classification": "post-freeze CPU-only derivation of committed proposal section 4.6",
        "selection_split": "dev",
        "test_split_inspected_for_selection": False,
        "puzzles": len(dev_ids),
        "descriptors": len(dev),
        "frozen_parameters": parameters,
        "dev_puzzle_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
        "shape_selection_trigger": {
            "covered_descriptors": sum(int(row["covered"]) for row in diagnostics),
            "shape_selection_losses_on_covered": sum(
                int(row["shape_selection_loss_on_covered"]) for row in diagnostics
            ),
            "puzzle_weighted_loss_rate": _puzzle_mean(
                diagnostics, "shape_selection_loss_on_covered"
            ),
        },
        "results": rows,
        "output": {
            "path": csv_path.name,
            "bytes": csv_path.stat().st_size,
            "sha256": _sha256(csv_path),
        },
    }
    _write_json(args.output / "audit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
