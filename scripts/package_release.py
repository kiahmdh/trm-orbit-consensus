from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, mode: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    archive.addfile(info, io.BytesIO(payload))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic upload archive from the verified release manifest"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    prefix = f"structured-orbit-consensus-{args.version}"
    output = (args.output or root.parent / f"{prefix}.tar.gz").resolve()
    checksum_output = output.with_suffix(output.suffix + ".sha256")
    manifest_path = root / "manifests" / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    members: list[tuple[str, Path, str | None]] = [
        (str(entry["relative_path"]), root / str(entry["relative_path"]), str(entry["sha256"]))
        for entry in manifest["files"]
    ]
    members.append(("manifests/release_manifest.json", manifest_path, None))
    members.sort(key=lambda item: item[0])

    for relative, path, expected_hash in members:
        if not path.is_file() or path.is_symlink():
            raise AssertionError(f"non-regular release member: {relative}")
        if expected_hash is not None and _sha256(path) != expected_hash:
            raise AssertionError(f"release manifest hash mismatch: {relative}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for relative, path, _ in members:
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            _add_bytes(archive, f"{prefix}/{relative}", path.read_bytes(), mode)

    archive_hash = _sha256(output)
    checksum_output.write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "archive": str(output),
                "archive_sha256": archive_hash,
                "bytes": output.stat().st_size,
                "files": len(members),
                "sha256_file": str(checksum_output),
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
