# ARC1 blank-ID scorer caveat

The blank-ID experiment is a valid retained-prediction zero-embedding ablation, but
it does not reproduce the historical 0% output as a prediction-level score.

The pinned dataset reserves identifier 0 for `<blank>`. The checkpoint embedding at
row 0 is exactly zero. During the ablation, only the model-input puzzle-identifier
tensor is replaced with 0; original augmentation identifiers remain intact for
`inverse_aug`, input hashing, emission order, and serialization. Normal and blank
caches have identical task IDs, candidate counts, augmentation indices, inverse
transform identifiers, queries, targets, and support pairs.

The pinned upstream ARC evaluator applies a padding mask that removes rows whose
identifier is 0 before canonicalization and voting. When every evaluation row is
blanked, this produces an empty prediction map and consequently the reported 0%.
That value does not score the model outputs generated with the zero embedding.

When those outputs are retained and evaluated with the same count-then-mean-Q vote
semantics, full ARC1 B1 is 3.5%. On the committed held-out split, blank B1 is 4.0%,
M1 is 4.0%, M1+M2 is 4.5%, M1+M2+M3 is 5.0%, and orbit coverage is 10.25%.

Required classification: **VALIDATED ABLATION WITH SCORER CAVEAT**.

Do not write “blank ID at 0% was reproduced.” The technically accurate statement is
that the historical scorer returns 0% by masking every ID-0 row, while retained
blank-embedding predictions have a legitimate nonzero score.
