from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {observed}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify compact committed result tables")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    args = parser.parse_args()

    main_rows = {
        row["method"]: row
        for row in _rows(args.results / "arc1_main" / "main_test_results.csv")
    }
    expected = {
        "B0": 0.2925,
        "B1": 0.4025,
        "M1": 0.4025,
        "M1+M2": 0.4125,
        "M1+M2+M3": 0.4125,
    }
    for method, value in expected.items():
        _close(float(main_rows[method]["rank1_accuracy"]), value, f"ARC1 {method}")

    gap_rows = _rows(args.results / "arc1_main" / "selection_gap.csv")
    full_rows = [row for row in gap_rows if int(row["budget"]) == 1000]
    pass_at_1000 = sum(float(row["pass_at_k"]) for row in full_rows) / len(full_rows)
    majority_at_1000 = sum(float(row["majority_at_k"]) for row in full_rows) / len(
        full_rows
    )
    _close(pass_at_1000, 0.6175, "ARC1 pass@1000")
    _close(majority_at_1000, 0.4, "ARC1 majority@1000")

    paper_summary = json.loads(
        (args.results / "paper_summary" / "paper_summary.json").read_text(encoding="utf-8")
    )
    _close(
        paper_summary["validated_ablations"]["arc1_blank_id"]["full_retained_prediction_b1"],
        0.035,
        "ARC1 blank retained B1",
    )
    _close(
        paper_summary["unresolved_supporting"]["arc2"]["pinned_upstream_pass_at_1"],
        7 / 240,
        "ARC2 pinned Pass@1",
    )

    shape_rows = _rows(
        args.results / "release_audit" / "shape_hedged_second_attempt.csv"
    )
    if len(shape_rows) != 2 or {row["split"] for row in shape_rows} != {"dev"}:
        raise AssertionError("shape-hedge artifact must contain two dev-only method rows")

    report = {
        "status": "passed",
        "arc1_rank1": {method: float(row["rank1_accuracy"]) for method, row in main_rows.items()},
        "selection": {
            "pass_at_1000": pass_at_1000,
            "majority_at_1000": majority_at_1000,
            "gap": pass_at_1000 - majority_at_1000,
        },
        "shape_hedge": shape_rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
