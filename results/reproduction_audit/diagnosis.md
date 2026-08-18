# CPU-only reproduction audit

Date: 2026-08-18

No GPU inference was run. The ARC1 normal, ARC1 blank, and ARC2 normal NPZ caches were treated as immutable and their aggregate fingerprints were unchanged across downstream analysis.

## ARC1 blank-ID diagnosis

The current cache is a valid retained-prediction ablation, but the published 0% number is not reproduced by a valid prediction-level comparison.

- Pinned builder metadata reserves identifier `0` as `<blank>`. No test puzzle uses it; ARC1 test identifiers begin at 362365.
- Checkpoint row 0 is exactly all-zero (L2 norm 0), while rows 1–10000 have median norm 3.1617. It is not a learned evaluation-puzzle identifier.
- The cache runner substitutes 0 only in the model-input tensor. It preserves the original augmentation identifier for `inverse_aug`, input hashing, and emission ordering.
- Normal and blank caches have identical 419 task IDs, candidate counts, augmentation indices, transform identifiers, queries, targets, and support data. Only 9,904/385,815 paired output grids (2.567%) are unchanged.
- The pinned paper experiment script also substitutes identifier 0, but then passes that tensor to the unmodified ARC evaluator. The evaluator's padding mask removes every row whose identifier equals 0 before canonicalization or voting. Its reported 0% therefore comes from an empty prediction map, not from scoring the blank-embedding model outputs.
- This also conflicts with the paper's prose description of a trained fixed embedding that does not correspond to an evaluation puzzle: the exact pinned row is zero and is explicitly reserved as a blank/padding sentinel.

When the actual blank-embedding predictions are retained and canonicalized, pinned upstream vote semantics give Pass@1 = 3.5% over all 400 puzzles. Fourteen descriptors remain correct: `070dd51e#0`, `1d0a4b61#0`, `256b0a75#0`, `40f6cd08#0`, `60c09cac#0`, `762cd429#0`, `7ee1c6ea#0`, `9def23fe#0`, `af24b4cc#0`, `c663677b#0`, `ca8f78db#0`, `d017b73f#0`, `e7b06bea#0`, and `e95e3d8e#0`.

All 14 were also correct under normal IDs and retain the same top-1 grid. One task (`c663677b#0`) is completely identifier-invariant, six have at least 50% emission overlap, five have a correct canonical/identity emission, and two are sparse pluralities below 2% vote mass. None is a query-copy or monochrome-target shortcut. Thus the successes are a mixture of identifier-insensitive behavior, partial emission overlap, and two fragile plurality coincidences—not an inventory mismatch.

Verdict: retain the 3.5% cache as a validated zero-embedding ablation, but do not call it an exact reproduction of the paper's 0% scorer output. No GPU rerun is warranted; a rerun with the same pinned checkpoint would not repair the evaluator-mask artifact.

## ARC2 diagnosis

The local inference/cache path is internally reproduced; the external 6.2% model-card value remains unresolved.

- Checkpoint: `arc_v2_public/step_723914`, Hugging Face revision `55ced5dd59de74c52f53d47aa2898232b5a15b7a`.
- Pinned code: `010206d1f0c25ebac0865f69e39c09969e6b896b`.
- Dataset manifest: seed 42, 1000 augmentations, subsets `training2 evaluation2 concept`, test set `evaluation2`.
- Identifier count and checkpoint embedding rows both equal 1,191,730. Row 0 is all-zero and reserved.
- Inventory is exactly 120 puzzles, 172 pair-scoped descriptors, and 170,393 emissions.
- The checkpoint-bundled TRM and loss implementations are byte-identical to the pinned checkout versions.
- Scoring uses the pinned evaluator's count then mean-Q ordering and averages test pairs within each puzzle before averaging 120 puzzles.
- The cache uses upstream `_crop` and `inverse_aug`; the previously validated round-trip gate was not rerun. Empty `(0,0)` outputs remain ordinary emitted invalid candidates under the evaluator's hash/vote behavior.
- Removing all 188 invalid candidates as a sensitivity check changes none of Pass@1, Pass@2, Pass@5, Pass@10, Pass@100, or Pass@1000.

Independent pinned-evaluator scores are: Pass@1 2.9167%, Pass@2 5.0%, Pass@5 6.9444%, Pass@10 7.2222%, Pass@100 10.8333%, Pass@1000 11.6667%. Pinned upstream Pass@1, online orbit B1, and cache-recomputed B1 agree exactly, with zero task-level top-1 mismatches.

The external model card reports 6.2%, but that value is neither local Pass@1 nor Pass@2. Descriptor weighting gives 3.49% top-1 and 5.23% top-2; counting any-correct puzzle gives 4.17% top-1 and 5.83% top-2. The model card provides no per-task output or scoring artifact to reconcile the residual, and its public mismatch discussion has no maintainer resolution.

Verdict: the ARC2 cache is valid under the pinned code, checkpoint, dataset, and evaluator, but the 6.2% external benchmark reproduction is unresolved. All ARC2 M1/M2/M3 results remain diagnostic/supporting and must not be promoted as a reproduced benchmark claim.
