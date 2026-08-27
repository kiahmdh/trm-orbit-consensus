# PASS — paper asset validation

## Frozen headline checks

| Check | Observed | Expected | Status |
|---|---:|---:|---|
| Full ARC-1 B1 | 40 | 40 | PASS |
| ARC-1 TEST B1 Rank-1 | 40.25 | 40.25 | PASS |
| ARC-1 TEST B1 Top-2 | 45.5 | 45.5 | PASS |
| ARC-1 TEST M1+M2 Rank-1 | 41.25 | 41.25 | PASS |
| ARC-1 TEST M1+M2 Top-2 | 44 | 44 | PASS |
| ARC-1 k=1000 coverage | 61.75 | 61.75 | PASS |
| ARC-1 k=1000 gap | 21.75 | 21.75 | PASS |
| ARC-1 TEST defect AUROC | 0.8418617947451856 | 0.8418617947451856 | PASS |
| ARC-1 TEST defect AURC | 0.31586167853807995 | 0.31586167853807995 | PASS |
| ARC-1 TEST margin AURC | 0.2607539081534486 | 0.2607539081534486 | PASS |
| ARC-1 blank TEST B1 | 4 | 4 | PASS |
| ARC-2 B1 | 2.9166666666666667 | 2.9166666666666667 | PASS |
| ARC-1 identifier count | 876406 | 876406 | PASS |

## Other required checks

- Input allowlist: PASS — 46 JSON and 70 CSV frozen release inputs including the self-excluded release manifest.
- Macro provenance: PASS — all 151 macros have artifact and transformation traces in manifest.json.
- PDF fonts: PASS — pdf.fonttype=42 and ps.fonttype=42; binary inspection found 0 Type 3 subtype tokens.
- Determinism: PASS — a second in-process render matched every PDF, PNG, TeX, and CSV intermediate byte-for-byte.
- CSV intermediates: PASS — fixed field order, LF endings, and no timestamps.
- Beta histogram: PASS — all 144 frozen thresholds are present; positive thresholds use deterministic log bins and zero thresholds are separately annotated.
- ARC-2 wording: PASS — supporting-only and not external benchmark reproduction.
- DEV-only boundary: PASS — discriminative-cell assets state that no TEST evaluation occurred.
- Output boundary: PASS by construction — every write is guarded under paper_assets/. The invoking audit separately compares Git status.

## Frozen p-value semantic reconciliation

The requested margin value 0.0002 is the frozen one-sided tail fraction fraction_defect_lower_aurc. The frozen two-sided bootstrap p-value formats to 0.0006. Table A3 reports both under separate labels; neither was relabeled or adjusted.

## Missing inputs

None. MISSING.md is intentionally absent.
