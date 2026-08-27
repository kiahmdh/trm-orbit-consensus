# Phase 3 Step 3.2 — Puzzle-Level Bootstrap Uncertainty

This analysis uses 10,000 percentile-bootstrap resamples with puzzle as the resampling unit. Every confidence statistic receives the exact same sampled puzzle-index vector within an ARC1 replicate. Duplicate sampled puzzles remain independent bootstrap instances. Confidence ties use the frozen score orientation, puzzle ID, then sampled-instance position—never correctness.

## ARC1 AURC uncertainty

| Statistic | Observed | Bootstrap mean | SE | 95% percentile CI |
| --- | --- | --- | --- | --- |
| structural_defect | 0.315862 | 0.317770 | 0.037917 | [0.246809, 0.394852] |
| vote_share | 0.262066 | 0.263937 | 0.034522 | [0.199826, 0.335120] |
| vote_margin | 0.260754 | 0.262607 | 0.034458 | [0.198744, 0.333966] |
| vote_entropy | 0.277689 | 0.279654 | 0.035665 | [0.212580, 0.353347] |
| winner_mean_q | 0.312777 | 0.314881 | 0.038399 | [0.243479, 0.394376] |

## Paired differences

Differences are `defect AURC - comparison AURC`; negative favors defect and positive favors the comparison.

| Comparison | Observed delta | Bootstrap mean | 95% CI | P(defect lower) | Two-sided p-style |
| --- | --- | --- | --- | --- | --- |
| vote_margin | 0.055108 | 0.055163 | [0.027036, 0.086492] | 0.000200 | 0.000600 |
| vote_share | 0.053796 | 0.053834 | [0.025594, 0.085073] | 0.000100 | 0.000400 |
| vote_entropy | 0.038173 | 0.038116 | [0.008446, 0.070410] | 0.005900 | 0.011999 |
| winner_mean_q | 0.003085 | 0.002889 | [-0.038495, 0.044443] | 0.443800 | 0.887711 |

Against analytically expected random ordering, the observed defect-minus-random difference is -0.281638, with 95% CI [-0.309608, -0.246320] and P(defect lower)=1.000000. Random-order AURC is recomputed analytically as each bootstrap population's overall error rate, so no random permutations are introduced.

## ARC1 structural-defect operating points

| Coverage | Retained | Observed accuracy | Accuracy 95% CI | Observed risk | Risk 95% CI |
| --- | --- | --- | --- | --- | --- |
| 0.100000 | 20 | 1.000000 | [1.000000, 1.000000] | 0.000000 | [0.000000, 0.000000] |
| 0.200000 | 40 | 0.925000 | [0.800000, 1.000000] | 0.075000 | [0.000000, 0.200000] |
| 0.250000 | 50 | 0.880000 | [0.780000, 0.980000] | 0.120000 | [0.020000, 0.220000] |
| 0.500000 | 100 | 0.630000 | [0.520000, 0.740000] | 0.370000 | [0.260000, 0.480000] |
| 0.750000 | 150 | 0.490000 | [0.406667, 0.573333] | 0.510000 | [0.426667, 0.593333] |
| 1.000000 | 200 | 0.402500 | [0.335000, 0.472500] | 0.597500 | [0.527500, 0.665000] |

## ARC1 blank-ID supporting result

Defect AURC is 0.922869, 95% CI [0.859425, 0.977352]. Defect-minus-random is -0.037131, 95% CI [-0.080175, -0.003121]. Only eight puzzle-equivalent correct outcomes exist at full coverage, so operating-point uncertainty is broad and operational claims are intentionally weak.

| Coverage | Retained | Observed accuracy | Accuracy 95% CI |
| --- | --- | --- | --- |
| 0.250000 | 50 | 0.100000 | [0.020000, 0.180000] |
| 0.500000 | 100 | 0.080000 | [0.030000, 0.140000] |
| 0.750000 | 150 | 0.053333 | [0.020000, 0.093333] |
| 1.000000 | 200 | 0.040000 | [0.015000, 0.070000] |

## ARC2 supporting result

Defect AURC is 0.918077, 95% CI [0.826144, 0.997883]. Defect-minus-random is -0.052756, 95% CI [-0.121275, 0.010857]. ARC2 contains only 3.5 puzzle-equivalent correct outcomes. Its pinned 2.9167% B1 result remains inconsistent with the externally reported approximately 6.2%, so this is supporting evidence only.

| Coverage | Retained | Observed accuracy | Accuracy 95% CI |
| --- | --- | --- | --- |
| 0.250000 | 30 | 0.066667 | [0.000000, 0.166667] |
| 0.500000 | 60 | 0.033333 | [0.000000, 0.083333] |
| 0.750000 | 90 | 0.038889 | [0.005556, 0.077778] |
| 1.000000 | 120 | 0.029167 | [0.004167, 0.058333] |

## Scientific verdict

- Defect reliably beats random abstention: **YES**.
- Defect reliably beats majority confidence: **NO**.
- Defect remains useful as a diagnostic confidence signal: **YES**.

Failure-AUROC CIs were not added: the frozen AUROC is descriptor-level with puzzle weights, whereas Step 3.1 contains already-aggregated, sometimes fractional puzzle correctness. Recasting that optional metric would require a new convention.

Protected caches, frozen results, Phase-1/Phase-2 diagnostics, and all Step 3.1 artifacts matched before/after fingerprints. No GPU, inference, cache generation, method tuning, or bootstrap threshold search was used.

STOP: await explicit approval before Phase 4 freeze.
