from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

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

    symlinks = [
        path for path in root.rglob("*") if path.is_symlink() and _is_publishable(root, path)
    ]
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
        and _is_publishable(root, path)
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

    checksum_path = root / "results" / "final_freeze" / "final_artifact_checksums.csv"
    with checksum_path.open(newline="", encoding="utf-8") as handle:
        checksum_rows = list(csv.DictReader(handle))
    expected_checksums = {
        path.relative_to(root).as_posix()
        for path in (root / "results").rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and path != checksum_path
    }
    observed_checksums = {row["relative_path"] for row in checksum_rows}
    if expected_checksums != observed_checksums:
        raise AssertionError("final scientific checksum inventory mismatch")
    for row in checksum_rows:
        path = root / row["relative_path"]
        if path.stat().st_size != int(row["size_bytes"]):
            raise AssertionError(f"final checksum size mismatch: {path}")
        if _sha256(path) != row["sha256"]:
            raise AssertionError(f"final checksum mismatch: {path}")

    json_count = 0
    csv_count = 0
    numeric_cells = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_publishable(root, path):
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

    files = [
        path for path in root.rglob("*") if path.is_file() and _is_publishable(root, path)
    ]
    forbidden_names = {".DS_Store"}
    forbidden_suffixes = {".ckpt", ".log", ".npz", ".orig", ".pth", ".pt", ".rej", ".safetensors", ".tmp"}
    forbidden_parts = {".ruff_cache", ".venv", "__pycache__", "build", "env", "venv"}
    forbidden_files = [
        path.relative_to(root).as_posix()
        for path in files
        if path.name in forbidden_names
        or path.suffix.lower() in forbidden_suffixes
        or forbidden_parts.intersection(path.relative_to(root).parts)
    ]
    if forbidden_files:
        raise AssertionError(f"forbidden release files found: {forbidden_files}")
    oversized = [
        path.relative_to(root).as_posix() for path in files if path.stat().st_size > 25 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"unexpected files larger than 25 MiB: {oversized}")

    text_suffixes = {".cfg", ".cff", ".csv", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    machine_patterns = (
        bytes((47, 104, 111, 109, 101, 47)),
        bytes((47, 85, 115, 101, 114, 115, 47)),
        bytes((67, 58, 92, 85, 115, 101, 114, 115, 92)),
    )
    secret_patterns = (
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"\b(?:ghp|github_pat|hf)_[A-Za-z0-9_\-]{20,}\b"),
        re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    )
    path_leaks: list[str] = []
    secret_leaks: list[str] = []
    for path in files:
        if path.suffix.lower() not in text_suffixes and path.name not in {"LICENSE"}:
            continue
        payload = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        if any(pattern in payload for pattern in machine_patterns):
            path_leaks.append(relative)
        if any(pattern.search(payload) for pattern in secret_patterns):
            secret_leaks.append(relative)
    if path_leaks:
        raise AssertionError(f"machine-specific paths found: {path_leaks}")
    if secret_leaks:
        raise AssertionError(f"credential-like material found: {secret_leaks}")

    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_files": len(expected) + 1,
                "scientific_artifacts": len(artifact_manifest["artifacts"]),
                "json_files_parsed": json_count,
                "csv_files_parsed": csv_count,
                "finite_numeric_csv_cells": numeric_cells,
                "final_checksum_entries": len(checksum_rows),
                "symlinks": 0,
                "machine_path_leaks": 0,
                "secret_leaks": 0,
                "forbidden_files": 0,
                "oversized_files": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
