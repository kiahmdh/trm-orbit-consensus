# Phase 2 Step 2.1 — Discriminative-cell structural similarity

## Gate: AMBIGUOUS

The local and DEV evidence is mixed and does not support a defensible TEST evaluation.

Only committed ARC1 DEV descriptors were deserialized. TEST was not evaluated.

## Background-dominance mechanism

Among 46 equal-shaped covered B1-failure comparisons, median ordinary similarity was 0.9600, median disagreement occupied 4.00% of cells, and 67.39% agreed on at least 90% of cells. Median dominant-color occupancy was 66.81%.

## DEV method results

| Method | β | Rank-1 | Top-2 | Changed puzzle-equiv. | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0 | 39.75% | 43.75% | 0.0 | 0.0 | 0.0 | +0.0 |
| original_M1 | 1 | 40.25% | 43.75% | 6.0 | 1.0 | 0.0 | +1.0 |
| discriminative_M1 | 1 | 40.00% | 43.25% | 17.0 | 2.0 | 1.5 | +0.5 |
| discriminative_M1 | 4 | 38.75% | 43.00% | 49.0 | 2.5 | 4.5 | -2.0 |

## Exact covered B1-failure subset

| Method | β | Fixed | Same wrong | Different wrong |
|---|---:|---:|---:|---:|
| B1 | 0 | 0.0 | 46.0 | 0.0 |
| original_M1 | 1 | 1.0 | 45.0 | 0.0 |
| discriminative_M1 | 1 | 2.0 | 39.0 | 5.0 |
| discriminative_M1 | 4 | 3.0 | 29.0 | 14.0 |

## Local structural discrimination

| Feature | Top-2 pairwise | 95% CI | Top-2 pooled AUROC | Top-5 pairwise | 95% CI | Top-5 pooled AUROC |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.746914 | [0.654321, 0.833333] | 0.545118 | 0.729869 | [0.664326, 0.794007] | 0.549147 |
| discriminative | 0.746914 | [0.654321, 0.833333] | 0.766085 | 0.599485 | [0.536517, 0.659644] | 0.556704 |

## B1-failure preference alignment

Original: correct 14/46, wrong 32/46, ties 0/46.
Discriminative: correct 21/46, wrong 16/46, ties 9/46.

## Validation

Focused tests: 8 passed. Existing release tests: 20 passed. Cache, split, frozen results, and both Phase-1 diagnostic trees remained unchanged. CUDA was hidden and no TEST cache payload or TEST label was loaded.

TEST evaluation and further method development were not started.
