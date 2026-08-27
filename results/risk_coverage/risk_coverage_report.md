# Phase 3 Step 3.1 — Selective Prediction / Risk–Coverage

## Scope and frozen definition

This is a deterministic CPU-only analysis of frozen caches and frozen result statistics. It does not tune a threshold, train a calibration model, alter a selector, run inference, or modify a cache.

For descriptor/orbit τ with exact-grid emission modes g, multiplicities c_g, total emissions K, and the frozen grid similarity S, the structural defect is:

`Δ(τ) = 1 - (1/K²) Σ_g Σ_h c_g c_h S(g,h)`.

Here `S(g,h)` is exact cell agreement when shapes match, zero when shapes differ, and one for `(0,0)` versus `(0,0)`. Lower defect is the frozen higher-confidence orientation. The statistic is computed per pair-scoped descriptor. For a puzzle with multiple test outputs, descriptor defects and descriptor B1 correctness values are each averaged within the puzzle; every puzzle then has weight 1.

Invalid `(0,0)` emissions remain ordinary exact-grid vote modes in the emitted population. Their self-similarity is one, while their similarity to every non-empty grid is zero. Therefore invalid mass contributes cross-mode disagreement but does not create self-disagreement.

AURC uses the deterministic discrete right-Riemann convention `AURC = (1/N) Σ_{n=1}^N risk(n)`, where the n retained puzzles are ordered by frozen confidence and exact confidence ties are broken by puzzle ID. Under a uniformly random ordering, expected risk is the full-population risk at every cutoff, so expected random-order AURC is exactly the full-coverage risk.

## ARC1 normal held-out TEST

Defect AURC is **0.315862** and full-coverage B1 accuracy is **40.2500%**.

| Puzzles | Coverage | B1 correct | Accuracy | Risk | Δ accuracy vs full | Max defect |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.250000 | 44.000000 | 0.880000 | 0.120000 | 0.477500 | 0.019281 |
| 100 | 0.500000 | 63.000000 | 0.630000 | 0.370000 | 0.227500 | 0.068008 |
| 150 | 0.750000 | 73.500000 | 0.490000 | 0.510000 | 0.087500 | 0.207386 |
| 200 | 1.000000 | 80.500000 | 0.402500 | 0.597500 | 0.000000 | 0.961145 |

## Frozen confidence comparison

| Statistic | Orientation | AURC | Random expected | Improvement |
| --- | --- | --- | --- | --- |
| structural_defect | lower_is_more_confident | 0.315862 | 0.597500 | 0.281638 |
| vote_share | higher_is_more_confident | 0.262066 | 0.597500 | 0.335434 |
| vote_margin | higher_is_more_confident | 0.260754 | 0.597500 | 0.336746 |
| vote_entropy | lower_is_more_confident | 0.277689 | 0.597500 | 0.319811 |
| winner_mean_q | higher_is_more_confident | 0.312777 | 0.597500 | 0.284723 |

Structural defect improves on random ordering but is not uniformly better than every raw majority-confidence baseline; the table gives the exact comparison.

Winner mean Q is included because the frozen B1 implementation already defines it as the mean Q of emissions in the winning exact-grid mode and uses it for vote ties. Winner max Q is not added because it is not part of the frozen B1 definition.

## ARC1 blank-ID held-out TEST

Defect AURC is **0.922869**; full-coverage B1 accuracy is **4.0000%**. Absolute accuracy is extremely low, so this curve is evidence only about confidence ordering under capability collapse.

| Puzzles | Coverage | B1 correct | Accuracy | Risk |
| --- | --- | --- | --- | --- |
| 20 | 0.100000 | 1.000000 | 0.050000 | 0.950000 |
| 40 | 0.200000 | 4.000000 | 0.100000 | 0.900000 |
| 60 | 0.300000 | 5.000000 | 0.083333 | 0.916667 |
| 80 | 0.400000 | 7.000000 | 0.087500 | 0.912500 |
| 100 | 0.500000 | 8.000000 | 0.080000 | 0.920000 |
| 120 | 0.600000 | 8.000000 | 0.066667 | 0.933333 |
| 140 | 0.700000 | 8.000000 | 0.057143 | 0.942857 |
| 160 | 0.800000 | 8.000000 | 0.050000 | 0.950000 |
| 180 | 0.900000 | 8.000000 | 0.044444 | 0.955556 |
| 200 | 1.000000 | 8.000000 | 0.040000 | 0.960000 |

## ARC2 supporting set

Defect AURC is **0.918077**; full-coverage B1 accuracy is **2.9167%**. The table reports exact retained and correct counts because ARC2 has only 120 puzzles and very few successes.

| Puzzles | Coverage | B1 correct | Accuracy | Risk |
| --- | --- | --- | --- | --- |
| 12 | 0.100000 | 2.000000 | 0.166667 | 0.833333 |
| 24 | 0.200000 | 2.000000 | 0.083333 | 0.916667 |
| 36 | 0.300000 | 2.000000 | 0.055556 | 0.944444 |
| 48 | 0.400000 | 2.000000 | 0.041667 | 0.958333 |
| 60 | 0.500000 | 2.000000 | 0.033333 | 0.966667 |
| 72 | 0.600000 | 2.000000 | 0.027778 | 0.972222 |
| 84 | 0.700000 | 3.500000 | 0.041667 | 0.958333 |
| 96 | 0.800000 | 3.500000 | 0.036458 | 0.963542 |
| 108 | 0.900000 | 3.500000 | 0.032407 | 0.967593 |
| 120 | 1.000000 | 3.500000 | 0.029167 | 0.970833 |

ARC2 remains supporting evidence only: the pinned evaluator gives 2.9167%, while the external reported result is approximately 6.2%, so absolute benchmark reproduction is unresolved.

## Sanity checks and conclusion

Reversing the frozen ARC1 defect orientation changes AURC from 0.315862 to 0.810831. The frozen lower-defect-first orientation was not selected post hoc.

Across ARC1 normal, blank-ID, and ARC2, defect ordering beats expected random ordering. Together with the frozen candidate-reranking negatives, this supports the bounded claim: **orbit-level structural defect is useful as a failure-confidence signal even when it is insufficient as a candidate reranker.**

Focused unit tests cover puzzle weighting, confidence ordering, deterministic defect ties, AURC, 100% coverage equality, and multi-output aggregation. Release tests were run separately. Protected cache/result/Phase-1/Phase-2 fingerprints matched before and after analysis. No bootstrap CI was run.

STOP: no bootstrap analysis, method development, or GPU work was started.
