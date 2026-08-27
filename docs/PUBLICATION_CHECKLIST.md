# Publication and release checklist

This repository is the complete **software, compact-results, provenance, and
camera-ready asset bundle** for the frozen experiment. It intentionally does not
redistribute third-party ARC data, TRM code, model checkpoints, or prediction caches.
Their pinned revisions, download commands, expected sizes, hashes, and cache
fingerprints are recorded in `DATA_AND_CHECKPOINTS.md`.

## Included in the upload archive

- installable CPU analysis package and tests;
- deterministic dataset/preflight/cache-production entry points;
- frozen configuration and all compact scientific JSON/CSV records;
- final-freeze checksums, release audit, and provenance;
- camera-ready vector figures, previews, TeX tables/macros/captions, and the
  deterministic asset renderer;
- license, citation metadata, setup instructions, and caveat documentation.

## Deliberately excluded

- `.git`, `.venv`, Python/test/linter caches, and build products;
- raw ARC datasets and processed augmented datasets;
- TinyRecursiveModels checkout and Hugging Face checkpoints;
- NPZ prediction caches and logs.

These exclusions do not prevent compact verification or paper-asset regeneration.
Full GPU reproduction requires downloading/building the pinned external inputs.

## Required verification before upload

```bash
python -m pytest -q -p no:cacheprovider
ruff check --no-cache src scripts tests
python scripts/verify_compact_results.py
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python paper_assets/build_assets.py
python scripts/generate_manifests.py --release-only
python scripts/verify_release.py
python scripts/package_release.py
```

Upload the generated `structured-orbit-consensus-0.1.0.tar.gz` and its adjacent
`.sha256` file. A GitHub/GitLab source release additionally requires committing the
working tree, adding the intended remote, and tagging the release; the packager does
not perform those external actions.

## Manuscript boundary

The repository contains all generated figure/table/macro/caption assets, but it does
**not** contain a workshop manuscript source (`main.tex`, bibliography, style files)
or compiled submission PDF. The release scope is therefore the software, compact
results, provenance, and reusable paper assets, not the manuscript archive itself.
