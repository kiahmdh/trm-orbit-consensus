# Experiment status

Audit date: 2026-08-19. **EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.** The
scientific definitions, split, hyperparameters, diagnostics, and selective-prediction
results are frozen. No GPU inference or retuning was performed during release
preparation.

## Cache inventory

| Cache | Status | Puzzles | Pair-scoped descriptors | Candidates | Invalid `(0, 0)` |
|---|---|---:|---:|---:|---:|
| ARC1 normal | Complete, exact B1 reproduction | 400 | 419 | 385,815 | 92 |
| ARC1 blank ID | Complete, scorer caveat | 400 | 419 | 385,815 | 720 |
| ARC2 normal | Complete, external discrepancy | 120 | 172 | 170,393 | 188 |

Completeness is pair-scoped: the correct descriptor counts are 419, 419, and 172,
not one descriptor per puzzle. The aggregate cache fingerprints are recorded in
`results/provenance/cache_completion.json`. Full caches are excluded from Git.

## Final scientific state

- ARC1 normal: pinned upstream online B1 = cache B1 = published reference = 0.400.
- ARC1 blank: retained-prediction B1 = 0.035; the upstream 0% scorer behavior is a
  padding-mask artifact.
- ARC2: pinned upstream Pass@1 = online B1 = cache B1 = 0.0291667; the external
  model-card value of approximately 0.062 remains unresolved.
- ARC1 dev/test split: 200/200 puzzles and 208/211 descriptors, with no puzzle overlap.
- Frozen method settings: distinct centrality, beta 1.0, lambda 0.05, epsilon 1e-6,
  emitted marginal support.
- The committed §4.6 shape hedge was absent from the original result directory. It
  was executed during release audit, CPU-only on the committed dev split, without
  retuning. It reduced dev top-2 from 43.75% to 42.75% for both M2 and M3 variants.
- Phase 1 found local top-mode structural signal but poor alignment with B1 failures;
  it also confirmed that the Top-2 regression is real, compute matching has no
  leakage, and the M1 beta response is mixed/non-monotonic.
- Phase 2's discriminative-cell feature was DEV-only and classified ambiguous/no-go;
  it was never evaluated on TEST.
- Phase 3 found that defect supports abstention relative to random ordering but is
  worse than vote margin/share/entropy for selective prediction.
- Phase 4 reconciles every headline metric in `results/final_freeze/` and closes the
  experiment layer.

## Canonical producers

| Purpose | Canonical release file | Historical note |
|---|---|---|
| Dataset preparation | `experiments/prepare_arc_datasets.py` | Strict ARC1/ARC2 recipes; does not mix the five subsets. |
| Production inference/cache | `experiments/run_arc_cache.py` | Renamed from `run_arc1_normal_cache.py`; same generalized validated path produced all three caches. |
| GPU feasibility probe | `experiments/benchmark_arc_probe.py` | Production serialization path and VRAM guard. |
| ARC1 main analysis | `scripts/run_arc1_analysis.py` | Produced the final ARC1 result directory. |
| Blank/ARC2 analyses | `scripts/run_supporting_analysis.py` | Produced both supporting result directories. |
| Shape-hedge audit | `scripts/run_shape_hedge_audit.py` | Frozen dev-only post-freeze derivation. |
| Definitive input audit | `scripts/audit_release_inputs.py` | Verifies external immutable caches and development artifacts. |
| Compact release check | `scripts/verify_compact_results.py` | Runs without external artifacts. |
| Final freeze builder | `scripts/build_final_freeze.py` | Validates compact sources and emits one shared JSON/CSV headline record plus claims and checksums. |

Superseded notebooks, exploratory logs, `.orig`/`.rej` files, large transition dumps,
raw caches, and legacy duplicate scripts were not included.

## Release validation

- Working-tree and clean-room test suites: **20 passed** each.
- Ruff: all release source, scripts, tests, and experiments passed.
- Python compilation: all release Python files compiled successfully with bytecode
  redirected outside the repository.
- Serialization/integrity: 46 JSON files and 70 CSV files parsed; 86,081 numeric
  CSV cells were finite; 128 scientific artifacts passed manifest verification.
- Final freeze: 162 headline records and 112 compact scientific files passed the
  dedicated checksum inventory; the complete release manifest contains 178 files
  including its self-excluded manifest.
- Clean-room package import resolved to the temporary environment, not the development
  tree; compact-result verification passed.
- Secret/machine scan: no credential pattern, private key, internal IP, or absolute
  development path found.
- Immutable-input before/after size/mtime inventories and all three aggregate cache
  fingerprints matched exactly.
