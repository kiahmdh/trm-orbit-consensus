# Step 1.3 — Mean-Q subsample / compute-matched audit

## Classification

**CASE A — IMPLEMENTATION CORRECT.** All ranking and tie-breaking statistics operate on the same deterministic k-emission suborbit. No correction is required.

## Exact production path

- `src/orbit_consensus/sampling.py:37-57 (paired_subsamples)` — effective k is min(requested k, orbit size); SHA256(seed:task_id) seeds NumPy; every repeat is sorted choice-without-replacement indices.
- `src/orbit_consensus/arc1_analysis.py:133-157 (_suborbit)` — selected indices are copied into a new TaskOrbit; each selected candidate's grid, q_value, invalid flag, transform, and diagnostics are preserved.
- `src/orbit_consensus/arc1_analysis.py:612-644 (compute_matched_rows)` — one paired plan per descriptor is materialized into reduced orbits, then evaluate_orbits receives only those reduced orbits.
- `src/orbit_consensus/baselines.py:9-14 (mean_q_by_grid)` — allowed_indices is accepted; the same conditional filters the candidate before both grid_key and q_value are appended. With no argument, all candidates in the supplied (already reduced) orbit are used.
- `src/orbit_consensus/baselines.py:17-34,44-48 (rank_candidates/majority_vote)` — B1 counts grids in the reduced orbit and ranks by count descending, reduced-orbit mean Q descending, then deterministic grid key ascending.
- `src/orbit_consensus/arc1_analysis.py:168-205 (evaluate_orbits)` — M1 preparation and B1/M1/M2/M3 evaluation all consume the same reduced orbit; no full-orbit prepared workspace is passed in compute-matched evaluation.
- `src/orbit_consensus/aggregators.py:51-97 (prepare_m1/m1_orbit_weighting)` — emission keys, distinct modes, counts, pairwise similarity, centrality, weights, and mean-Q tie-breaks are all built from the reduced orbit only.
- `src/orbit_consensus/aggregators.py:114-210 (_shape_filter/m2_cell_marginal_score)` — M2/M3 weights, screening indices, modal shape, marginals, active counts, and mean Q are restricted to the reduced orbit; mean_q_by_grid receives the within-suborbit active_set.
- `src/orbit_consensus/aggregators.py:213-222 (two_attempt_policy)` — the hedge combines already subset-scoped M2/M3 and B1 rankings and cannot access the original orbit.
- `configs/experiment.toml:1-6 and docs/PROPOSAL_COVERAGE.md:22` — the committed protocol fixes seed 20260807, k={50,250,1000}, ten paired repeats, and all four methods; reduced-orbit centrality matches the intended all-method compute-matched comparison.

The selected indices are materialized into a new `TaskOrbit`; selectors never receive the original full orbit during compute-matched evaluation. B1 and M1 therefore call `mean_q_by_grid` on the reduced candidate tuple. M2/M3 further pass active modal-shape indices belonging to that reduced tuple, so their grids, counts, evidence, and mean-Q values are filtered together.

M1 distinct-mode structural centrality is intentionally recomputed from only the k-subsample. No full-orbit precomputation is supplied by `compute_matched_rows`.

## Synthetic leakage regression

The selected set has a 2–2 vote tie: A has selected mean Q 0.80 versus B 0.45, so production selects A. Full-orbit Q reverses the comparison and the deliberately buggy hybrid selects B. The production result is deterministic and the allowed-index Q map exactly equals the reduced-orbit Q map.

## Independent DEV cross-check

Four deterministic DEV cases passed, including no top-count tie, a top-count tie, multiple duplicate modes, a naturally selected `(0,0)` invalid prediction, and k smaller than the full orbit. Maximum absolute numerical errors were: B1 mean Q `0.0`, M1 mean Q `0.0`, M1 score `0.0`, and M1 centrality `0.0`.

## Frozen compute-matched Rank-1 means

| k | Method | Mean Rank-1 | Status |
|---:|---|---:|---|
| 50 | B1 | 40.475% | remains valid |
| 50 | M1 | 40.375% | remains valid |
| 50 | M1+M2 | 40.375% | remains valid |
| 50 | M1+M2+M3 | 40.375% | remains valid |
| 250 | B1 | 40.500% | remains valid |
| 250 | M1 | 40.650% | remains valid |
| 250 | M1+M2 | 40.500% | remains valid |
| 250 | M1+M2+M3 | 40.500% | remains valid |
| 1000 | B1 | 40.350% | remains valid |
| 1000 | M1 | 40.350% | remains valid |
| 1000 | M1+M2 | 41.250% | remains valid |
| 1000 | M1+M2+M3 | 41.250% | remains valid |

## Validation

- Focused audit tests: 6 passed.
- Existing release tests: 20 passed.
- ARC1 cache unchanged: True (`6cb320cb9b0e45c114d95722654663d2b7e98b01027a1d4a647c21ea4ce60fa1`).
- Committed split unchanged: True (`0f6e16e36c3b37703928a294655905d4af659fcda5f9f8b12d5d336931a6106d`).
- Frozen ARC1 result tree unchanged: True (`3d47a841c13904ba4dbe10ac5f7e69adc80b48369e65d35d48e2577258f455ac`).
- CUDA was hidden, torch was not imported, no GPU work was launched, and no existing cache/result artifact was written.

No `compute_matched_corrected.csv` or `compute_matched_delta.csv` was created because no bug was found.
