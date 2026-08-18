from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_entry(root: Path, entry: dict[str, object]) -> None:
    path = root / str(entry["relative_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["size"]):
        raise AssertionError(f"size mismatch: {path}")
    if _sha256(path) != entry["sha256"]:
        raise AssertionError(f"SHA256 mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a packaged release in place")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise AssertionError(f"release contains symlinks: {symlinks}")

    release_manifest = json.loads(
        (root / "manifests" / "release_manifest.json").read_text(encoding="utf-8")
    )
    for entry in release_manifest["files"]:
        _verify_entry(root, entry)
    expected = {entry["relative_path"] for entry in release_manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path != root / "manifests" / "release_manifest.json"
    }
    if expected != actual:
        raise AssertionError(
            f"manifest inventory mismatch: missing={sorted(expected - actual)}, "
            f"untracked={sorted(actual - expected)}"
        )

    artifact_manifest = json.loads(
        (root / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for entry in artifact_manifest["artifacts"]:
        _verify_entry(root, entry)

    json_count = 0
    csv_count = 0
    numeric_cells = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for field, value in row.items():
                    if field in {"task_id", "puzzle_id"} or not value:
                        continue
                    if value.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
                        raise AssertionError(f"non-finite value in {path}: {value}")
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(number):
                        raise AssertionError(f"non-finite value in {path}: {value}")
                    numeric_cells += 1
            csv_count += 1

    forbidden = str(Path.home()).encode()
    leaks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and forbidden in path.read_bytes()
    ]
    if leaks:
        raise AssertionError(f"machine-specific paths found: {leaks}")

    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_files": len(expected) + 1,
                "scientific_artifacts": len(artifact_manifest["artifacts"]),
                "json_files_parsed": json_count,
                "csv_files_parsed": csv_count,
                "finite_numeric_csv_cells": numeric_cells,
                "symlinks": 0,
                "machine_path_leaks": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
