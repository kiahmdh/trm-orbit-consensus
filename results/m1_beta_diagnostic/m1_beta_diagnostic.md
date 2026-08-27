# Step 1.4 — M1 beta DEV-curve / score-scale analysis

## Result

**CASE C — MIXED / NON-MONOTONIC.** positive beta values 1/2/4 improve DEV slightly over B1, while larger positive beta values erase the gain at 8 and fall below B1 at 16

Only the committed 200-puzzle / 208-descriptor DEV split was evaluated. The historical sweep provided the frozen accuracy curve; the immutable DEV cache was used only for decision transitions and score thresholds.

## Original sweep and decision changes

Puzzle-equivalent counts average pair-scoped descriptors within each puzzle, matching the committed evaluation metric. `Changed puzzles` is the ordinary number of puzzles with at least one changed descriptor.

| Centrality | β | DEV Rank-1 | DEV Top-2 | Changed puzzles | Changed puzzle-equiv. | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distinct | -16 | 28.00% | 39.00% | 89 | 88.0 | 2.0 | 25.5 | -23.5 |
| distinct | -8 | 32.50% | 40.75% | 64 | 63.0 | 1.0 | 15.5 | -14.5 |
| distinct | -4 | 39.25% | 43.75% | 31 | 31.0 | 0.0 | 1.0 | -1.0 |
| distinct | -2 | 39.25% | 43.25% | 18 | 18.0 | 0.0 | 1.0 | -1.0 |
| distinct | -1 | 39.75% | 43.75% | 10 | 10.0 | 0.0 | 0.0 | +0.0 |
| distinct | 0 | 39.75% | 43.75% | 0 | 0.0 | 0.0 | 0.0 | +0.0 |
| distinct | 1 | 40.25% | 43.75% | 6 | 6.0 | 1.0 | 0.0 | +1.0 |
| distinct | 2 | 40.25% | 43.75% | 9 | 9.0 | 1.0 | 0.0 | +1.0 |
| distinct | 4 | 40.25% | 43.75% | 14 | 14.0 | 2.0 | 1.0 | +1.0 |
| distinct | 8 | 39.75% | 43.75% | 21 | 21.0 | 2.5 | 2.5 | +0.0 |
| distinct | 16 | 39.00% | 43.25% | 30 | 29.5 | 2.5 | 4.0 | -1.5 |
| multiset | -16 | 25.75% | 34.25% | 105 | 103.5 | 4.5 | 32.5 | -28.0 |
| multiset | -8 | 27.75% | 36.75% | 83 | 81.5 | 2.0 | 26.0 | -24.0 |
| multiset | -4 | 37.25% | 43.25% | 39 | 39.0 | 0.0 | 5.0 | -5.0 |
| multiset | -2 | 39.25% | 43.25% | 20 | 20.0 | 0.0 | 1.0 | -1.0 |
| multiset | -1 | 39.75% | 43.75% | 10 | 10.0 | 0.0 | 0.0 | +0.0 |
| multiset | 0 | 39.75% | 43.75% | 0 | 0.0 | 0.0 | 0.0 | +0.0 |
| multiset | 1 | 39.75% | 43.75% | 5 | 5.0 | 0.0 | 0.0 | +0.0 |
| multiset | 2 | 40.25% | 43.75% | 8 | 8.0 | 1.0 | 0.0 | +1.0 |
| multiset | 4 | 40.25% | 43.75% | 11 | 11.0 | 1.0 | 0.0 | +1.0 |
| multiset | 8 | 40.25% | 43.75% | 12 | 12.0 | 1.0 | 0.0 | +1.0 |
| multiset | 16 | 39.75% | 43.75% | 22 | 22.0 | 1.0 | 1.0 | +0.0 |
| non_identical_support | -16 | 28.00% | 37.00% | 92 | 91.0 | 2.0 | 25.5 | -23.5 |
| non_identical_support | -8 | 31.50% | 38.75% | 71 | 70.0 | 1.0 | 17.5 | -16.5 |
| non_identical_support | -4 | 38.25% | 43.75% | 37 | 37.0 | 0.0 | 3.0 | -3.0 |
| non_identical_support | -2 | 39.25% | 43.25% | 20 | 20.0 | 0.0 | 1.0 | -1.0 |
| non_identical_support | -1 | 39.75% | 43.75% | 10 | 10.0 | 0.0 | 0.0 | +0.0 |
| non_identical_support | 0 | 39.75% | 43.75% | 0 | 0.0 | 0.0 | 0.0 | +0.0 |
| non_identical_support | 1 | 39.75% | 43.75% | 5 | 5.0 | 0.0 | 0.0 | +0.0 |
| non_identical_support | 2 | 40.25% | 43.75% | 11 | 11.0 | 1.0 | 0.0 | +1.0 |
| non_identical_support | 4 | 40.25% | 43.75% | 12 | 12.0 | 1.0 | 0.0 | +1.0 |
| non_identical_support | 8 | 40.25% | 43.75% | 15 | 15.0 | 2.0 | 1.0 | +1.0 |
| non_identical_support | 16 | 38.25% | 42.25% | 37 | 36.0 | 3.0 | 6.0 | -3.0 |

## Frozen selection rule

The release maximized DEV Rank-1, then DEV Top-2, then smaller `|β|`, then earlier centrality order. It selected `distinct`, `β=1` with Rank-1 40.25% and Top-2 43.75%.

## Distinct-centrality flip scale

For the strongest-centrality alternative to each B1 winner, 144 descriptors (139 puzzles) could theoretically flip for positive β. The descriptor-level thresholds have minimum 0, Q25 63.2963, median 248.374, mean 5.51589e+07, and Q75 993.969. Fractions at original positive sweep thresholds are: β≤1 1.39%, β≤2 2.08%, β≤4 2.78%, β≤8 4.17%, β≤16 8.33%.

## Exact Step 1.1 B1-failure subset

| β | Fixed puzzle-equiv. | Same wrong | Different wrong |
|---:|---:|---:|---:|
| -16 | 2.0 | 30.0 | 14.0 |
| -8 | 1.0 | 37.0 | 8.0 |
| -4 | 0.0 | 42.0 | 4.0 |
| -2 | 0.0 | 45.0 | 1.0 |
| -1 | 0.0 | 46.0 | 0.0 |
| 0 | 0.0 | 46.0 | 0.0 |
| 1 | 1.0 | 45.0 | 0.0 |
| 2 | 1.0 | 44.0 | 1.0 |
| 4 | 2.0 | 41.0 | 3.0 |
| 8 | 3.0 | 38.0 | 5.0 |
| 16 | 3.0 | 36.0 | 7.0 |

## Interpretation

The first positive original beta meeting the descriptive materiality rule (>1% puzzle-equivalent decisions changed or several puzzles changed) is β=1. At β=1, 6.0/200 puzzle-equivalent decisions change and the net is +1.0. The best positive plateau is β=[1.0, 2.0, 4.0] at 40.25%; higher β eventually returns to or drops below B1. Thus scaling is mixed/non-monotonic, not evidence that ever-stronger structural weight repairs B1 failures.

## Validation

- Focused tests: 4 passed.
- Existing release tests: 20 passed.
- Historical sweep rows reproduced exactly: True.
- Cache unchanged: True (`6cb320cb9b0e45c114d95722654663d2b7e98b01027a1d4a647c21ea4ce60fa1`).
- Committed split unchanged: True (`0f6e16e36c3b37703928a294655905d4af659fcda5f9f8b12d5d336931a6106d`).
- Frozen result tree unchanged: True (`3d47a841c13904ba4dbe10ac5f7e69adc80b48369e65d35d48e2577258f455ac`).
- CUDA was hidden; torch was not imported; no TEST cache payload was deserialized; no GPU work was launched.

Phase 2, risk-coverage, discriminative-cell weighting, method changes, and new TEST evaluations were not started.
