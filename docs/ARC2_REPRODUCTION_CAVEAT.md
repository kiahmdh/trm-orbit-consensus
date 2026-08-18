# ARC2 reproduction caveat

The ARC2 cache and evaluation are internally consistent under the pinned setup:

- TRM commit `010206d1f0c25ebac0865f69e39c09969e6b896b`;
- checkpoint `arc_v2_public/step_723914` at Hugging Face revision
  `55ced5dd59de74c52f53d47aa2898232b5a15b7a`;
- seed 42, 1,000 augmentations, subsets `training2 evaluation2 concept`, test set
  `evaluation2`;
- 1,191,730 identifiers and the same number of checkpoint embedding rows;
- 120 puzzles, 172 pair-scoped descriptors, and 170,393 emissions;
- count then mean-Q ranking and puzzle-weighted scoring.

Pinned upstream Pass@1, online orbit B1, and cache-recomputed B1 agree exactly at
2.9167%, with zero task-level top-1 mismatches. Pinned Pass@2 is 5.0%. Removing all
188 invalid candidates changes none of Pass@1/2/5/10/100/1000.

The public model card reports approximately 6.2%, but does not provide a compatible
per-task output/scoring artifact. That number is neither local Pass@1 nor Pass@2,
and alternate descriptor or any-correct-puzzle weightings do not reconcile it.

Required classification: **SUPPORTING / EXTERNAL RESULT WITH UNRESOLVED
REPRODUCTION DISCREPANCY**. Do not label this ARC2 result as a benchmark reproduction.

Production used batch size 1. In the 80-prediction benchmark, an independent batch-1
repeat was exact. Batches 2, 4, and 8 agreed with one another, but batch 2 differed
from batch 1 on 31 grids and 59 Q-values (maximum absolute Q difference 0.06386).
This batch dependence is a reproducibility constraint, not a main scientific result.
