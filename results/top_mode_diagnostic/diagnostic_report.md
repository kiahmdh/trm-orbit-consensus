# Step 1 — Top-mode structural discrimination

## Scope and frozen statistic

This analysis used only the committed ARC1 DEV split (200 puzzles,
208 pair-scoped descriptors). No held-out TEST cache payload or
label was loaded.

For the frozen M1 definition, each unique canonical mode `c` has raw structural centrality

`d(c) = (sum_j S(c,j) - 1) / (m - 1)` for `m > 1`, and `d(c)=0` for `m=1`,

where `m` is the number of distinct modes and `S` is exact cell-agreement for equal-shaped grids,
zero for different shapes, and one for `(0,0)` compared with itself. Larger `d(c)` means more
central/structurally consistent, so the oriented score equals raw `d(c)`.

With frozen `beta=1`, each emission receives `exp(d(c))/Z`; duplicate emissions are retained, so
the unique-mode M1 score is `count(c) * exp(d(c)) / Z`. Modes are ordered by that score, then mean
Q, then the deterministic grid key. B1 orders by vote count, then mean Q, then grid key. Invalid
`(0,0)` emissions remain in the population as an ordinary unique mode; under distinct centrality
their similarity to every differently shaped mode is zero.

## DEV coverage

- Covered: 124 puzzles (any descriptor covered), 131 descriptors.
- Uncovered: 76 puzzles (no descriptor covered), 77 descriptors.
- Multi-output detail: 123 fully covered puzzles and 1 partially covered puzzle.

## Discrimination results

- All modes: pooled AUROC 0.546792; puzzle-local pairwise accuracy 0.731687.
- Top 2: 81 eligible puzzles / 87 descriptors; pooled AUROC 0.545118; pairwise accuracy 0.746914 (95% puzzle-bootstrap CI 0.654321–0.833333).
- Top-2 puzzle preferences: 56 correct, 9 tied, 16 wrong.
- Top 5: 89 eligible puzzles / 95 descriptors; pooled AUROC 0.549147; mean/median within-puzzle AUROC 0.729869/0.833333; pairwise accuracy 0.729869 (95% puzzle-bootstrap CI 0.664326–0.794007).

Pooled-AUROC 95% puzzle-bootstrap CIs are 0.526852–0.570381 (top 2) and 0.532010–0.571042 (top 5). Multi-output descriptors were always resampled together at puzzle level.

## B1 failure subset

Among 46 covered puzzles (47 descriptors) where
B1 is wrong, structure prefers the highest-vote correct alternative in
14 puzzles, ties in
0, and prefers the wrong B1 winner in
32.

## Structural margins

Descriptor-level margin is `score(correct mode) - score(best structurally competing wrong mode)`.

- Top 2: mean 0.053570, median 0.003620, SD 0.171462, Q25/Q75 0.000000/0.031250; fractions >0/=0/<0 0.689655/0.068966/0.241379.
- Top 5: mean -0.007196, median 0.000000, SD 0.040788, Q25/Q75 -0.005265/0.003847; fractions >0/=0/<0 0.452632/0.084211/0.463158.

## Validation

- Cache fingerprint unchanged: True.
- Committed split unchanged: True.
- Existing ARC1 result artifacts unchanged: True.
- CUDA hidden for the analysis process: ''; torch was not imported.
- Bootstrap: 10000 resamples, puzzle as unit, seed 42.
- Focused diagnostic tests: 4 passed, 0 failed.
- Existing release CPU tests: 20 passed, 0 failed. Because pytest is not installed in the existing environment, the simple pytest-style functions were executed by the included dependency-free runner; its `pytest.raises` shim and `tmp_path` fixture cover the only pytest features used by this suite.

## Classification

**CASE B — MEASURABLE LOCAL SIGNAL**

At least one primary local metric is >=0.60 and its puzzle-bootstrap CI excludes chance in the useful direction.

This report stops after Step 1. No method, frozen parameter, cache, GPU inference, or TEST analysis was changed or started.
