# Reproducibility

**EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.** Reproduction means verifying the
frozen record; it does not authorize new method development, tuning, or TEST use.

## Frozen contract

- TRM code commit: `010206d1f0c25ebac0865f69e39c09969e6b896b`.
- Hugging Face checkpoint revision:
  `55ced5dd59de74c52f53d47aa2898232b5a15b7a`.
- Dataset augmentation seed: 42; `num_aug`: 1000.
- Analysis split/subsample/bootstrap seed: 20260807.
- Split: puzzle-level 200 dev / 200 test; 208 / 211 pair descriptors.
- Budgets: 50, 250, 1000; ten paired subsamples per budget.
- Bootstrap: 10,000 paired puzzle resamples.
- Frozen M1: distinct centrality, beta 1.0.
- Frozen M2: lambda 0.05, epsilon 1e-6, emitted marginal support.
- M3: support-derived shape screening with empty-filter fallback.
- Empty upstream predictions: exact `(0, 0)` grids with `is_invalid=true`; they remain
  in the emitted population and B1 denominator/vote map.

Hyperparameters were selected on dev only. The stored dev-puzzle list hashes to
`d58a469503ff133c7f713bf43270a745398f49e09be51a432c0a5ffbf713e096`.

## CPU setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/verify_compact_results.py
```

CPU analysis depends only on NumPy. GPU inference additionally requires the pinned
TRM environment and its PyTorch/CUDA dependencies; keep that environment separate.

## Preflight gates

After acquiring/building external artifacts as described in
`DATA_AND_CHECKPOINTS.md`:

```bash
python scripts/trm_preflight.py \
  --upstream-root "$TRM_ROOT" \
  --dataset-path "$TRM_ROOT/data/arc1concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  --cache-dir artifacts/cache/arc1_normal \
  --provenance-dir artifacts/preflight/arc1 \
  --transform-trials 4096
```

Required ARC1 outcomes are 876,406 identifiers, 876,406 checkpoint embedding rows,
all eight D4 transform IDs, and bit-exact inverse-transform round trips. The recorded
gate passed all 4,096 trials.

## GPU probe and cache production

Always probe the exact model/dataset/serialization path before a full run. The
production runner uses `torch.inference_mode()`, disables gradients, preserves
ordering, writes each pair descriptor atomically, reload-checks every NPZ, and skips
valid completed descriptors on restart.

ARC1 blank is the same command as normal with `--puzzle-id-mode blank` and a different
output directory. ARC2 uses its own dataset/checkpoint/config and batch size 1:

```bash
TRM_ROOT="$TRM_ROOT" CUDA_VISIBLE_DEVICES=0 python experiments/run_arc_cache.py \
  --batch-size 1 --puzzle-id-mode blank \
  --dataset-path "$TRM_ROOT/data/arc1concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  --config-path "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/all_config.yaml" \
  --output-dir artifacts/cache/arc1_blank

TRM_ROOT="$TRM_ROOT" CUDA_VISIBLE_DEVICES=0 python experiments/run_arc_cache.py \
  --batch-size 1 --puzzle-id-mode normal \
  --dataset-path "$TRM_ROOT/data/arc2concept-aug-1000" \
  --checkpoint-path "$TRM_ROOT/checkpoints/hf_trm/arc_v2_public/step_723914" \
  --config-path "$TRM_ROOT/checkpoints/hf_trm/arc_v2_public/all_config.yaml" \
  --output-dir artifacts/cache/arc2_normal
```

For disconnect-safe execution, place each long command in its own GNU Screen session;
Screen supervision changes no scientific setting. Do not use the historical protected
PID from another machine. `--protected-pid 0` disables that optional identity check;
provide a live PID only when a specific unrelated process must be verified unchanged.

## ARC2 batch-size constraint

Batch size 1 is deterministic and was used for production. An 80-prediction repeat
matched grids and Q-values exactly. Batches 2/4/8 were mutually exact, but batch 2
differed from batch 1 on 31 grids and 59 Q-values, with maximum absolute Q difference
0.0638594. Batch 4 had the highest probe throughput, but throughput did not justify
changing numerical behavior. Production therefore remained at batch size 1.

## Analysis order

1. Complete each cache and verify inventory.
2. Require online B1 to equal cache-recomputed B1 exactly.
3. Require ARC1 normal B1 to equal 0.400.
4. Run ARC1 normal analysis and freeze dev-selected parameters.
5. Run blank-ID and ARC2 supporting analyses with those frozen parameters.
6. Run the dev-only shape-hedge audit without inspecting test to select a variant.
7. Run `scripts/audit_release_inputs.py` against the full external workspace.
8. Verify the Phase-1/2 compact diagnostics and Phase-3 selective-prediction outputs.
9. Run `scripts/build_final_freeze.py` to validate sources and regenerate the shared
   JSON/CSV freeze record.
10. Run `scripts/generate_manifests.py --source-root /path/to/workspace` followed by
    `scripts/verify_release.py` and `scripts/verify_compact_results.py`.

Every analysis producer refuses to overwrite a non-empty directory. Use a fresh
output location; compare its manifest to the committed compact results afterward.

## Metric and tie semantics

- Task IDs are pair-scoped, for example `00576224#0`.
- ARC metrics average pairs within puzzle and then average puzzles.
- B1 includes invalid `(0, 0)` emissions. It ranks distinct grids by count, then mean
  Q; the stable upstream insertion order is the final exact tie behavior.
- Coverage is an orbit property, independent of selector shape filtering.
- Small orbits clamp a requested subsample budget to the emitted orbit size and record
  the effective budget.
- Centrality similarity workspaces are independent of beta and reused across sweeps.

## What is and is not reproduced

- ARC1 40% is an exact pinned reproduction.
- Blank ID is a validated zero-embedding ablation with an upstream scorer caveat.
- ARC2 is internally consistent at 2.9167% under the pinned path, but it is **not**
  a reproduction of the external approximately 6.2% result; the discrepancy remains
  unresolved and ARC2 is supporting evidence only.

The Phase-4 builder writes `results/final_freeze/final_results_manifest.json` and
`final_results_summary.csv` from one in-memory record, preventing JSON/CSV drift.
Its checksum table covers compact scientific outputs, while the repository manifests
cover both the scientific inventory and the complete public release.

## Camera-ready assets and release archive

The paper assets are a deterministic rendering of the frozen release JSON/CSV files;
they do not read NPZ caches, raw ARC data, checkpoints, or model outputs.

```bash
python -m pip install -e '.[figures]'
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python paper_assets/build_assets.py
python scripts/generate_manifests.py --release-only
python scripts/verify_release.py
python scripts/package_release.py
```

`generate_manifests.py --release-only` leaves the frozen scientific manifests
unchanged and refreshes only the complete publication inventory. The packager reads
that verified inventory and emits a deterministic `.tar.gz` plus SHA256 file next to
the repository. Local environments and ignored runtime outputs are never archived.
