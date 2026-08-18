# Proposal coverage

The table records the authoritative compact release artifact for every committed
runbook item. “Pass with caveat” means the experiment was executed and validated,
but its interpretation is constrained; it does not mean the artifact is missing.

| Proposal item | Status | Artifact | Notes |
|---|---|---|---|
| Step 1 — pinned code/checkpoint/data provenance; deterministic seed-42, 1,000-augmentation builds | PASS | `results/provenance/upstream_revisions.json`; `results/provenance/dataset_builds.json` | TRM commit, HF revision, checkpoint SHA256, strict ARC1/ARC2 recipes, and identifier checksums recorded. |
| Step 2 — ARC1 B1 reproduction gate | PASS | `results/provenance/cache_completion.json`; `results/arc1_main/analysis_report.json` | Online B1, cache B1, and published reference are exactly 0.400. |
| Step 3 — identifier/checkpoint alignment and blank-ID ablation | PASS WITH CAVEAT | `results/provenance/arc1_preflight.json`; `results/arc1_blank_ablation/analysis_report.json`; `results/reproduction_audit/diagnostic_report.json` | 876,406 identifiers equal checkpoint rows. Blank outputs are valid, but upstream ID-0 padding creates the historical 0% scorer artifact. |
| Step 4 — transform/inverse-transform round trips | PASS | `results/provenance/arc1_preflight.json` | 4,096 bit-exact trials, all D4 IDs seen, 4,090 translated cases. |
| Step 5 — ARC1/ARC2 caches and online/cache B1 agreement | PASS WITH CAVEAT | `results/provenance/cache_completion.json`; `results/reproduction_audit/diagnostic_report.json` | Inventories are 400/419, 400/419, and 120/172. ARC2 internal agreement is exact; external 6.2% remains unresolved. |
| Step 6 — orbit dispersion and vote-mass statistics | PASS | `results/arc1_main/orbit_statistics.csv`; `results/arc1_main/orbit_statistics_summary.csv` | 419 descriptor rows plus puzzle-weighted summary. |
| Step 7 — committed puzzle-level dev/test split | PASS | `results/arc1_main/committed_split.json`; `results/release_audit/integrity_report.json` | 200/200 puzzles, 208/211 descriptors, zero overlap, all pairs colocated. |
| Step 8 — H1/equivariance-defect AUROC | PASS | `results/arc1_main/h1_auroc_results.csv` | Held-out 0.8418618; overall 0.8215755. |
| Step 9 — d-count decoupling diagnostic | PASS | `results/arc1_main/d_count_diagnostic.csv` | All three centrality definitions executed. |
| Step 10 — pass@k versus majority@k for k=50/250/1000 | PASS | `results/arc1_main/selection_gap.csv` | Ten paired subsamples per budget with effective-budget/clamping metadata. |
| Steps 11–12 — M1 and M2 sweeps and frozen hyperparameters | PASS | `results/arc1_main/m1_dev_sweep.csv`; `results/arc1_main/m2_dev_sweep.csv`; `results/arc1_main/frozen_hyperparameters.json` | Selection is explicitly dev-only; frozen settings were used on test. |
| Step 13 — M3 shape-screening diagnostics | PASS | `results/arc1_main/shape_screening_diagnostics.csv`; `results/arc1_main/analysis_report.json` | Active on 86% of held-out puzzles; filters 0.8416% of emissions; no rank-1 gain. |
| Step 14 — held-out B0/B1/M1/M2/M3 and paired bootstrap | PASS | `results/arc1_main/main_test_results.csv`; `results/arc1_main/paired_bootstrap_cis.csv` | Puzzle-level 10,000-resample paired bootstrap. |
| Step 15 — compute-matched k=50/250/1000 | PASS | `results/arc1_main/compute_matched_results.csv` | All four aggregation methods, ten repeats, and three budgets executed. |
| Step 16 — coverage, rank/MRR, ID ablation, error correlation | PASS WITH CAVEAT | `results/arc1_main/coverage_conditioned_results.csv`; `results/arc1_main/rank_mrr_results.csv`; `results/arc1_main/error_correlation_results.csv`; `results/arc1_blank_ablation/analysis_report.json` | Blank-ID ablation has the documented scorer caveat. |
| Step 17 — ARC2 supporting evaluation | PASS WITH CAVEAT | `results/arc2_supporting/analysis_report.json`; `results/reproduction_audit/diagnostic_report.json` | Pinned paths agree at 2.9167%; classified supporting, not benchmark reproduction. |
| Marginal-support ablation | PASS | `results/arc1_main/marginal_support_ablation.csv` | Emitted and distinct-uniform definitions were compared on dev. |
| Shape-hedged second attempt (§4.6) | PASS WITH CAVEAT | `results/release_audit/shape_hedged_second_attempt.csv`; `results/release_audit/audit_report.json` | The file was not present before release audit. The committed frozen experiment was then executed CPU-only on dev; both variants changed top-2 by -1.00 point. No test-driven selection occurred. |

All items were actually parsed and their expected row counts, parameter grids,
inventories, or score contracts were checked. The definitive verifier is
`results/release_audit/integrity_report.json`; existing immutable artifacts were not
regenerated or overwritten.

FULL COMMITTED PROPOSAL COVERAGE CONFIRMED
