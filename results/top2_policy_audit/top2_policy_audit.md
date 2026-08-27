# Top-2 / §4.6 hedge definition audit

## Exact proposal definition

The authoritative Google Doc, §4.6 (tab `t.0`, paragraph indexes 15679–16031), defines the primary two-attempt policy as: M2 rank-1 in slot 1 and B1 rank-1 in slot 2; if those grids coincide, use M2 rank-2 in slot 2. §4.2 defines ordinary B1 Top-2 as the first two B1-ranked modes. M1 likewise uses its first two ranked modes.

The same §4.6 paragraph defines a distinct, additional DEV-only shape hedge: if modal-shape error is non-trivial, take slot 2 from the best candidate of the runner-up shape.

## Current implementation

`arc1_analysis.py:200–213` sends B1/M1 ranking prefixes to ordinary Top-2 and calls `two_attempt_policy` for M2/M3. `aggregators.py:213–222` implements exactly the primary proposal policy. `arc1_analysis.py:941–943` uses those outcomes to write `main_test_results.csv`.

Therefore the frozen 44.00% M2/M3 number is already the committed B1-diversity hedge, not the first two candidates from the structured ranking.

## Recomputed results

| Split | Method | Rank-1 | Frozen/current Top-2 | Committed hedge Top-2 | Difference | Structured ranking Top-2 |
|---|---|---:|---:|---:|---:|---:|
| DEV | B1 | 39.7500% | 43.7500% | 43.7500% | +0.0000% | 43.7500% |
| DEV | M1 | 40.2500% | 43.7500% | 43.7500% | +0.0000% | 43.7500% |
| DEV | M1+M2 | 39.7500% | 43.7500% | 43.7500% | +0.0000% | 42.7500% |
| DEV | M1+M2+M3 | 39.7500% | 43.7500% | 43.7500% | +0.0000% | 42.7500% |
| TEST | B1 | 40.2500% | 45.5000% | 45.5000% | +0.0000% | 45.5000% |
| TEST | M1 | 40.2500% | 45.5000% | 45.5000% | +0.0000% | 45.5000% |
| TEST | M1+M2 | 41.2500% | 44.0000% | 44.0000% | +0.0000% | 44.0000% |
| TEST | M1+M2+M3 | 41.2500% | 44.0000% | 44.0000% | +0.0000% | 44.0000% |

The committed policy changes no rank-1 result. On TEST, B1 Top-2 remains 45.50% and M2/M3 remain 44.00%, so the reported 1.50-point regression remains.

## Transition accounting

Current-versus-committed differing puzzles: 0 across every split/method comparison. All transition gains and losses are zero because the policies are identical.

## Separate shape-hedge audit caveat

`analysis_pipeline.py:636–669` searches the already-produced M2/M3 ranking for the first candidate whose shape differs from rank-1, then falls back to rank-2. But `aggregators.py:157–210` constructs that ranking only from candidates of one selected modal shape. Consequently no different-shape second slot is reachable; the released DEV `shape_hedged_second_attempt.csv` value of 42.75% is operationally structured-ranking Top-2, not a realized runner-up-shape policy. This audit records the issue but does not change or reimplement it.

## Scientific interpretation

1. The main frozen Top-2 number is technically correct relative to the primary committed §4.6 policy.
2. It is not merely a different metric; it is the exact proposed B1-diversity hedge.
3. The committed hedge does not remove the M2/M3 Top-2 regression because it is already what produced 44.00%.
4. No rank-1 value changes.
5. A paper should label the committed hedged second-attempt accuracy as the main pass@2 result and may report structured-ranking Top-2 separately, clearly named. The existing shape-hedge result should not be described as runner-up-shape performance without correcting and separately validating that implementation.

## Validation

- Focused hedge tests: 5 passed, 0 failed.
- Deterministic repeat: True; a second full 419-descriptor pass matched all results, transition accounting, definitions, and interpretation fields exactly.
- Cache unchanged: True.
- Committed split unchanged: True.
- Existing frozen results unchanged: True.
- CUDA was hidden and torch was not imported.

STOP: no Q audit, method change, reranking, risk-coverage analysis, cache mutation, or GPU inference was started.
