# Final experiment freeze

Freeze date: 2026-08-19.

**EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.**

The experiment layer is closed. No new method, tuning, held-out TEST evaluation, confidence statistic, selector, inference, cache generation, or ARC2 discrepancy experiment is authorized by this record.

## Final scientific status

- ARC1 baseline reproduction is exact: online B1 = cache B1 = published verification = 40.00% over 400 puzzles.
- Held-out ARC1 B1 is separately 40.25%; these values refer to different populations and must not be conflated.
- The strongest primary result is the k=1000 candidate-generation/selection gap: 61.75% coverage versus 40.00% majority accuracy, a 21.75-point gap.
- Held-out defect AUROC is 0.841862. Defect reliably improves selective prediction over random ordering, but is reliably worse than vote margin/share/entropy by AURC.
- M1 has no held-out Rank-1 gain; M2's +1 point is exploratory; M3 adds no gain. The committed M2/M3 Top-2 value is 44.00%.
- Discriminative-cell weighting is a DEV-only ambiguous/no-go result; no TEST evaluation was performed.
- Full retained-prediction blank-ID B1 is 3.5%; held-out blank-ID B1 is 4.00%. The upstream 0% is a padding-mask scorer artifact.
- Pinned ARC2 is 2.9167%, supporting only, with an unresolved discrepancy from the external approximately 6.2% report. It is not a successful benchmark reproduction.

## Frozen claim hierarchy

- **PRIMARY:** ARC1 has a 21.75 percentage-point candidate-generation/selection gap at k=1000.
- **PRIMARY:** Orbit-level structural defect predicts held-out ARC1 failure (AUROC 0.841862).
- **PRIMARY:** Candidate-level structured aggregation weakly exploits the signal: M1 has no gain, M2 has an exploratory +1 point, and M3 adds no gain.
- **SECONDARY:** Candidate-level structural signal exists locally but is poorly aligned with exact B1 failures.
- **SECONDARY:** Whole-grid similarity is dominated by agreement over mostly unchanged/background cells.
- **SECONDARY:** Discriminative-cell weighting improves failure alignment but not DEV selector accuracy and was not evaluated on TEST.
- **SECONDARY:** M1 weakness is not explained by beta merely being too small.
- **SECONDARY:** Structural defect supports selective prediction relative to random ordering.
- **NEGATIVE:** Structural defect is not better than vote margin, vote share, or vote entropy for selective prediction.
- **ABLATION:** Puzzle-ID blanking collapses candidate coverage from 61.75% to 10.25% on held-out ARC1.
- **SUPPORTING:** ARC2 qualitatively preserves the selection-gap/structural-signal pattern, but its absolute external-score discrepancy is unresolved.

## Metric reconciliation

| Value | Authoritative meaning |
|---:|---|
| 40.00% | Full 400-puzzle ARC1 baseline reproduction. |
| 40.25% | Committed 200-puzzle ARC1 held-out TEST B1. |
| 61.75% | Both held-out orbit coverage and full k=1000 pass@k; separate rows identify the context. |
| 44.00% | Committed M2/M3 held-out TEST Top-2. |
| 3.5% | Full retained-prediction blank-ID B1. |
| 4.00% | Held-out blank-ID B1. |
| 2.9167% | Internally consistent pinned ARC2 result, not external benchmark reproduction. |

## Artifact contract

`final_results_manifest.json` and `final_results_summary.csv` contain 162 frozen headline records from one shared data structure. `final_claims.json` records allowed and forbidden claims. `final_artifact_checksums.csv` covers all non-manifest scientific files under `results/` except itself, avoiding recursive hashes.

**EXPERIMENTAL RESULTS ARE FROZEN.**
