# Authoritative results

**EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.** The machine-readable authority is
`results/final_freeze/final_results_manifest.json`; this document summarizes it.

All metrics are puzzle-weighted: multiple test pairs are averaged within a puzzle
before averaging puzzles. Pair-scoped descriptor counts are reported separately.

## Robust / validated

- **ARC1 reproduction:** 400 puzzles, 419 descriptors, 385,815 candidates, 92
  preserved invalid predictions. Pinned online B1 = cache B1 = published result =
  **40.00%** exactly.
- **Candidate-selection gap:** at k=50, pass@k = 52.0875%, majority@k = 39.2875%,
  gap = 12.80 points; at k=250, 57.9125% versus 40.0875%, gap = 17.825 points;
  at k=1000, 61.75% versus 40.00%, gap = 21.75 points.
- **Structural signal:** equivariance-defect AUROC for B1 failure is 0.8418618 on
  held-out ARC1 and 0.8215755 over all ARC1 puzzles.
- **Blank-ID collapse:** held-out B1 falls from 40.25% to 4.00%, and coverage from
  61.75% to 10.25%. Blank-minus-normal B1 is -36.25 points, paired 95% bootstrap
  CI [-43.0, -29.5], p = 0.00019998.

## Held-out frozen methods

| Method | Rank-1 | Top-2 | Coverage | MRR |
|---|---:|---:|---:|---:|
| B0 | 29.25% | 29.25% | 61.75% | 0.2925 |
| B1 | 40.25% | 45.50% | 61.75% | 0.453263 |
| M1 | 40.25% | 45.50% | 61.75% | 0.453032 |
| M1+M2 | 41.25% | 44.00% | 61.75% | 0.447566 |
| M1+M2+M3 | 41.25% | 44.00% | 61.75% | 0.447566 |

Frozen parameters were selected on the 200-puzzle dev split only: M1 distinct
centrality with beta 1.0; M2 lambda 0.05, epsilon 1e-6, and emitted marginal support.
All 200 held-out puzzles and their 211 pair descriptors were evaluated afterward.

## Exploratory

M1+M2 improves held-out rank-1 over B1 by 1.00 point. The 10,000-resample paired
puzzle bootstrap interval is [0.00, 2.50] points with p = 0.26637. Because the
interval contains zero and the effect is small, this is exploratory rather than
statistically persuasive.

## Negative findings

- M1 alone does not improve rank-1 or top-2 over B1.
- M3 is active on 86% of held-out puzzles and filters about 0.84% of emissions, but
  it does not improve rank-1 beyond M2.
- M2 reduces held-out top-2 from 45.50% to 44.00%.
- The §4.6 dev-only shape-hedged second attempt reduces top-2 from 43.75% to 42.75%
  for both M2 and M3 variants.
- On ARC2, M1/M2/M3 do not improve rank-1; M1 reduces top-2 from 5.0% to 4.5833%,
  and M2/M3 reduce it to 3.75%.

## Caveated findings

### Blank-ID scorer behavior

The full retained-prediction blank cache contains 400 puzzles, 419 descriptors,
385,815 candidates, and 720 invalid predictions, with B1 = 3.5%. The pinned upstream
evaluator treats identifier 0 as padding and drops every blanked row before voting;
its 0% output is therefore not the score of the retained model predictions. This is
a validated ablation with a scorer caveat, not a reproduction of blank ID at 0%.

### ARC2 external discrepancy

ARC2 contains 120 puzzles, 172 descriptors, 170,393 candidates, and 188 invalid
predictions. Pinned upstream Pass@1, online B1, and cache B1 agree exactly at
2.9167%; pinned Pass@2 is 5.0%, coverage is 11.6667%, selection gap is 8.75 points,
and defect AUROC is 0.8279174. The external model-card value of about 6.2% cannot be
reconciled with this pinned evaluator/checkpoint/dataset combination. ARC2 remains
supporting evidence and is not labeled a benchmark reproduction.

## Final structural diagnostics

On covered DEV top-mode comparisons, structure has local signal: Top-2 pairwise
accuracy is 0.746914 (95% CI [0.654321, 0.833333]) and mean within-puzzle Top-5
AUROC is 0.729869 (95% CI [0.664326, 0.794007]). On the 46 puzzle-equivalent
covered B1 failures, however, structure prefers the correct alternative only 14
times and the wrong B1 winner 32 times. The signal is therefore poorly aligned with
the failures that selection must repair.

The M1 beta diagnostic is mixed and non-monotonic. Distinct-centrality DEV Rank-1
rises from 28.00% at beta=-16 to 40.25% at beta=1, 2, and 4, then falls to 39.00%
at beta=16. At beta=1 only 6/200 puzzle-equivalent decisions change (one fix, no
breaks). The empirical beta-flip quartiles are 63.2963, 248.374, and 993.969, with
only 8.33% at or below 16; increasing beta is not a simple remedy.

Whole-grid similarity is background dominated: covered top-mode candidates have
median ordinary similarity 0.9600, median disagreement over 4.00% of cells, 72.88%
at least 90% agreement, 60.26% median dominant-color occupancy, and 60.86% median
agreement explained by the shared dominant color. Covered B1 failures retain 0.9600
median similarity, 4.00% disagreement, and 66.81% dominant-color occupancy.

The DEV-only discriminative-cell diagnostic improves correct preference on B1
failures from 30.43% to 45.65%, but does not improve selector accuracy: B1 is
39.75%, frozen M1 beta=1 is 40.25%, discriminative beta=1 is 40.00%, and
discriminative beta=4 is 38.75%. Top-5 pairwise accuracy falls from 0.729869 to
0.599485. It is classified **AMBIGUOUS / NO-GO FOR TEST**; no TEST evaluation was
performed.

The Top-2 audit confirms the implementation exactly matches the committed policy.
The 44.00% M2/M3 TEST Top-2 result and -1.50-point effect are real, not a scoring
bug. A different-shape runner-up appeared in 0/402 eligible DEV rankings. The
compute-matched audit found no Q or full-orbit leakage; all frozen k=50/250/1000
results remain valid.

## Selective prediction

On held-out ARC1, structural-defect AURC is 0.315862 (95% CI
[0.246809, 0.394852]), versus random expected AURC 0.597500. The paired difference
is -0.281638 (95% CI [-0.309608, -0.246320]), so defect reliably beats random
ordering. Its predetermined retained accuracies are 88.00%, 63.00%, 49.00%, and
40.25% at 25%, 50%, 75%, and 100% coverage.

This is not superiority over simple confidence baselines. Vote-margin, vote-share,
and vote-entropy AURCs are 0.260754, 0.262066, and 0.277689, all significantly lower
than defect AURC; winner-mean-Q AURC is 0.312777 and is statistically indistinguishable.
Structural defect is an informative failure diagnostic, not the best confidence
measure for selective prediction. Blank-ID selective prediction beats random only
cautiously because there are eight puzzle-equivalent correct outcomes; ARC2's
defect-minus-random interval crosses zero.
