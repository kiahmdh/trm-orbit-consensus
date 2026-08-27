# Beyond Majority Voting: Structured Orbit Consensus for Tiny Recursive Models

This repository studies a training-free question for Tiny Recursive Models (TRM):
when test-time augmentation produces hundreds of candidate ARC grids, can structure
inside that orbit select a better answer than whole-grid majority voting? It contains
the cache format, aggregation methods, frozen analyses, compact scientific results,
and reproducibility metadata. Model weights, processed datasets, and prediction
caches are intentionally not committed.

**EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.** The authoritative frozen record is
in `results/final_freeze/`; no result in that record is a prompt for further tuning
or held-out evaluation.

## Main findings

ARC-AGI-1 has a large candidate-selection gap. Across the full 400-puzzle cache,
the correct answer appears in **61.75%** of 1,000-augmentation orbits, while majority
voting solves **40.00%**, a **21.75 percentage-point gap**. Equivariance defect is a
useful structural signal: its held-out AUROC for majority-vote failure is **0.8419**.

Frozen methods were evaluated on the committed 200-puzzle held-out split:

| Method | Rank-1 | Top-2 | Orbit coverage |
|---|---:|---:|---:|
| B0: identity/canonical | 29.25% | 29.25% | 61.75% |
| B1: whole-grid majority | 40.25% | 45.50% | 61.75% |
| M1: orbit centrality | 40.25% | 45.50% | 61.75% |
| M1+M2: cell-marginal reranking | 41.25% | 44.00% | 61.75% |
| M1+M2+M3: shape screening | 41.25% | 44.00% | 61.75% |

The M2 gain over B1 is exploratory: +1.00 percentage point, with a 10,000-resample
paired puzzle bootstrap CI of [0.00, 2.50] points and two-sided p = 0.266. The
interval includes zero, so this is not statistically persuasive evidence of an
improvement. M1 and M3 do not improve rank-1; M2 also lowers top-2 by 1.5 points.

Structural defect also supports selective prediction relative to random ordering
(AURC 0.315862, 95% CI [0.246809, 0.394852], versus random 0.597500), but it is not
a superior confidence measure: vote margin, vote share, and vote entropy all have
lower AURC. Phase-1/2 diagnostics show that local structural signal exists but is
poorly aligned with the exact majority failures, and a DEV-only discriminative-cell
variant was therefore classified ambiguous/no-go without TEST evaluation.

## Ablations and supporting evaluation

Blanking puzzle identifiers causes a large degradation. Retained blank-embedding
predictions score 3.5% on all ARC1 puzzles; on the held-out split, B1 falls from
40.25% to 4.00% and coverage falls from 61.75% to 10.25%. This is a validated
ablation with a scorer caveat: the pinned upstream evaluator removes identifier-0
rows as padding before voting, so its historical 0% is produced by an empty
prediction map, not by scoring the retained model outputs. See
[ARC1_BLANK_SCORER_CAVEAT.md](docs/ARC1_BLANK_SCORER_CAVEAT.md).

ARC2 is supporting evidence only. The pinned evaluator, online orbit B1, and
cache-recomputed B1 agree exactly at 2.9167%, with Pass@2 = 5.0%. This does not
reconcile with the external model-card value of about 6.2%; therefore ARC2 is not
claimed as a benchmark reproduction. See
[ARC2_REPRODUCTION_CAVEAT.md](docs/ARC2_REPRODUCTION_CAVEAT.md).

## Repository layout

```text
configs/       Frozen sweep, split, sampling, and bootstrap settings
src/           CPU cache schema, aggregation, diagnostics, and analyses
scripts/       CPU analysis, audit, preflight, and compact-result verification
experiments/   Deterministic dataset builder, GPU probe, and resumable cache runner
results/       Compact immutable tables, reports, caveat audit, and provenance
paper_assets/  Camera-ready PDFs, PNG previews, tables, macros, captions, and renderer
docs/          Experiment status, coverage, results, and reproduction instructions
tests/         Unit tests, including invalid (0, 0) prediction round trips
manifests/     Scientific-artifact and complete release manifests
```

The canonical final CPU producers are `scripts/run_arc1_analysis.py` and
`scripts/run_supporting_analysis.py`. `experiments/run_arc_cache.py` is the
validated resumable production runner, renamed from the historical
`run_arc1_normal_cache.py` because the same path produced ARC1 normal, ARC1 blank,
and ARC2 normal caches. Debug notebooks and superseded orchestration scripts are
excluded from the release.

## Installation and compact verification

Python 3.10 or newer is supported for CPU analysis.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src scripts tests
python scripts/verify_compact_results.py
```

The compact verification command requires no GPU, checkpoint, processed dataset,
or prediction cache.

Camera-ready assets use only the frozen JSON/CSV record and can be rebuilt without
prediction caches or a GPU:

```bash
python -m pip install -e '.[figures]'
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python paper_assets/build_assets.py
```

Before uploading a software/artifact release, refresh only the publication inventory,
verify it, and build the deterministic archive:

```bash
python scripts/generate_manifests.py --release-only
python scripts/verify_release.py
python scripts/package_release.py
```

The archive is written next to this repository and excludes `.git`, virtual
environments, caches, checkpoints, datasets, and prediction artifacts. See
[PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md).

## Data and checkpoints

Full reproduction uses the pinned TRM fork at commit
`010206d1f0c25ebac0865f69e39c09969e6b896b` and Hugging Face repository
`arcprize/trm_arc_prize_verification` at revision
`55ced5dd59de74c52f53d47aa2898232b5a15b7a`. Download instructions, expected
paths, sizes, and checksums are in
[DATA_AND_CHECKPOINTS.md](docs/DATA_AND_CHECKPOINTS.md).

Use a separate TRM/CUDA environment for inference. Point the release scripts at the
checkout with a configurable environment variable rather than editing paths:

```bash
export TRM_ROOT=/path/to/TinyRecursiveModels
```

## Staged reproduction

The following commands are templates; GPU work should be scheduled only after the
CPU gates pass. Dataset augmentation uses seed 42 and `num_aug=1000`.

```bash
TRM_ROOT="$TRM_ROOT" python experiments/prepare_arc_datasets.py --dataset arc1
TRM_ROOT="$TRM_ROOT" python experiments/prepare_arc_datasets.py --dataset arc2

python scripts/trm_preflight.py \
  --upstream-root "$TRM_ROOT" \
  --dataset-path "$TRM_ROOT/data/arc1concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  --provenance-dir artifacts/preflight/arc1
```

Run a new production-path probe before inference. The validated production runs used
batch size 1, inference mode, atomic per-descriptor NPZ writes, and `(0, 0)` invalid
prediction preservation.

```bash
TRM_ROOT="$TRM_ROOT" CUDA_VISIBLE_DEVICES=0 python experiments/benchmark_arc_probe.py \
  --dataset-path "$TRM_ROOT/data/arc1concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  --config-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/all_config.yaml" \
  --output-dir artifacts/benchmark/arc1_probe

TRM_ROOT="$TRM_ROOT" CUDA_VISIBLE_DEVICES=0 python experiments/run_arc_cache.py \
  --batch-size 1 \
  --dataset-path "$TRM_ROOT/data/arc1concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  --config-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/all_config.yaml" \
  --output-dir artifacts/cache/arc1_normal \
  --expected-b1 0.4
```

The runner skips complete descriptor files and writes each cache atomically, so the
same command resumes safely. Commands for blank-ID and ARC2 are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Do not run downstream sweeps until
online B1 and cache-recomputed B1 agree.

Once external caches are present:

```bash
python scripts/run_arc1_analysis.py \
  --cache artifacts/cache/arc1_normal \
  --config configs/experiment.toml \
  --output artifacts/results/arc1_main

python scripts/run_supporting_analysis.py arc1-blank \
  --normal-cache artifacts/cache/arc1_normal \
  --blank-cache artifacts/cache/arc1_blank \
  --main-results artifacts/results/arc1_main \
  --output artifacts/results/arc1_blank_ablation

python scripts/run_supporting_analysis.py arc2 \
  --cache artifacts/cache/arc2_normal \
  --main-results artifacts/results/arc1_main \
  --output artifacts/results/arc2_supporting
```

Analysis producers refuse to overwrite non-empty output directories. Use a new output
path for an independent reproduction.

## Results and audit trail

- `results/arc1_main/`: frozen ARC1 sweeps, held-out tables, diagnostics, and bootstrap.
- `results/arc1_blank_ablation/`: retained-prediction blank-ID analysis.
- `results/arc2_supporting/`: pinned ARC2 supporting analysis.
- `results/reproduction_audit/`: scorer and external-reference diagnosis.
- `results/release_audit/`: definitive integrity report and the post-freeze §4.6 audit.
- `results/provenance/`: revisions, build recipes, cache inventories, and checksums.
- `results/paper_summary/`: concise paper-facing machine-readable summary.
- `results/top_mode_diagnostic/` through `results/discriminative_cell_dev/`: compact
  Phase-1/2 mechanism and policy audits.
- `results/risk_coverage/` and `results/risk_coverage_bootstrap/`: frozen Phase-3
  selective-prediction outputs.
- `results/final_freeze/`: authoritative headline manifest, claims, reconciliation,
  and scientific-artifact checksums.

See [RESULTS.md](docs/RESULTS.md) for the authoritative narrative and
[PROPOSAL_COVERAGE.md](docs/PROPOSAL_COVERAGE.md) for the line-by-line runbook audit.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Until a paper DOI or
archival URL is assigned, cite this software release by title and authors.

## License

The project code is released under the [MIT License](LICENSE). External TRM code,
ARC datasets, and model checkpoints retain their own licenses and are not redistributed.
