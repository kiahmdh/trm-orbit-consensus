from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FREEZE = RESULTS / "final_freeze"
DATE = "2026-08-19"


def _read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected!r}, found {observed!r}")


def _validate_sources() -> None:
    cache = _read_json("results/provenance/cache_completion.json")
    inventory = cache
    expected_inventory = {
        "arc1_normal": (400, 419, 385815, 92),
        "arc1_blank": (400, 419, 385815, 720),
        "arc2_normal": (120, 172, 170393, 188),
    }
    for name, expected in expected_inventory.items():
        row = inventory[name]
        observed = (
            row["puzzles"],
            row["descriptors"],
            row["candidates"],
            row["invalid_predictions"],
        )
        if observed != expected:
            raise AssertionError(f"{name} inventory mismatch: {observed} != {expected}")

    main = {row["method"]: row for row in _read_csv("results/arc1_main/main_test_results.csv")}
    expected_main = {
        "B0": (0.2925, 0.2925),
        "B1": (0.4025, 0.455),
        "M1": (0.4025, 0.455),
        "M1+M2": (0.4125, 0.44),
        "M1+M2+M3": (0.4125, 0.44),
    }
    for method, expected in expected_main.items():
        _close(float(main[method]["rank1_accuracy"]), expected[0], f"{method} rank1")
        _close(float(main[method]["top2_accuracy"]), expected[1], f"{method} top2")

    risk = _read_json("results/risk_coverage_bootstrap/risk_coverage_bootstrap_summary.json")
    defect = next(row for row in risk["arc1"] if row["statistic"] == "structural_defect")
    _close(defect["observed_aurc"], 0.31586167853807995, "defect AURC")
    if risk["verdict"] != {
        "defect_vs_majority_confidence": "NO",
        "defect_vs_random": "YES",
        "diagnostic_confidence_signal": "YES",
    }:
        raise AssertionError("Step 3.2 verdict changed")

    discriminative = _read_json(
        "results/discriminative_cell_dev/discriminative_cell_summary.json"
    )
    if discriminative["scope"]["test_evaluated"] is not False:
        raise AssertionError("discriminative-cell experiment must remain DEV-only")
    if discriminative["classification"]["case"] != "AMBIGUOUS":
        raise AssertionError("discriminative-cell classification changed")


def _headline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        claim: str,
        dataset: str,
        split: str,
        metric: str,
        method: str,
        value: float | str,
        unit: str,
        source: str,
        classification: str,
        ci_low: float | str = "",
        ci_high: float | str = "",
        uncertainty: str = "",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "result_id": f"R{len(rows) + 1:03d}",
                "claim": claim,
                "dataset": dataset,
                "split": split,
                "metric": metric,
                "method": method,
                "value": value,
                "unit": unit,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "uncertainty": uncertainty,
                "artifact_source": source,
                "classification": classification,
                "note": note,
            }
        )

    provenance = "results/provenance/cache_completion.json"
    add("ARC1 baseline reproduction", "ARC1", "all", "puzzle_count", "B1", 400,
        "puzzles", provenance, "PRIMARY")
    add("ARC1 baseline reproduction", "ARC1", "all", "descriptor_count", "B1", 419,
        "descriptors", provenance, "PRIMARY")
    add("ARC1 baseline reproduction", "ARC1", "all", "candidate_count", "B1", 385815,
        "emissions", provenance, "PRIMARY")
    add("ARC1 baseline reproduction", "ARC1", "all", "invalid_prediction_count", "B1", 92,
        "emissions", provenance, "PRIMARY")
    for metric, method in (
        ("online_B1", "pinned upstream"),
        ("cache_B1", "cache recomputation"),
        ("published_verification", "published reference"),
    ):
        add("ARC1 baseline reproduction", "ARC1", "all", metric, method, 0.4,
            "fraction", provenance, "PRIMARY", note="EXACT BASELINE REPRODUCTION")

    split_source = "results/arc1_main/committed_split.json"
    for split, puzzles, descriptors in (("dev", 200, 208), ("test", 200, 211)):
        add("Committed puzzle split", "ARC1", split, "puzzle_count", "split", puzzles,
            "puzzles", split_source, "PRIMARY")
        add("Committed puzzle split", "ARC1", split, "descriptor_count", "split",
            descriptors, "descriptors", split_source, "PRIMARY")
    add("Committed puzzle split", "ARC1", "dev/test", "puzzle_overlap", "split", 0,
        "puzzles", split_source, "PRIMARY")

    main_source = "results/arc1_main/main_test_results.csv"
    methods = {
        "B0": (0.2925, 0.2925),
        "B1": (0.4025, 0.455),
        "M1": (0.4025, 0.455),
        "M1+M2": (0.4125, 0.44),
        "M1+M2+M3": (0.4125, 0.44),
    }
    for method, (rank1, top2) in methods.items():
        classification = "NEGATIVE" if method in {"M1", "M1+M2", "M1+M2+M3"} else "PRIMARY"
        add("Frozen held-out methods", "ARC1", "test", "rank1_accuracy", method, rank1,
            "fraction", main_source, classification)
        add("Frozen held-out methods", "ARC1", "test", "top2_accuracy", method, top2,
            "fraction", main_source, classification)
    hyper = "results/arc1_main/frozen_hyperparameters.json"
    for metric, value in (
        ("M1_centrality", "distinct"),
        ("M1_beta", 1.0),
        ("M2_lambda", 0.05),
        ("M2_epsilon", 1e-6),
        ("M2_marginal_support", "emitted"),
    ):
        add("Frozen hyperparameters", "ARC1", "dev", metric, "frozen selector", value,
            "setting", hyper, "PRIMARY", note="selected on DEV only")

    bootstrap_source = "results/arc1_main/paired_bootstrap_cis.csv"
    for method, metric, delta, low, high, p_value in (
        ("M1", "rank1", 0.0, 0.0, 0.0, 1.0),
        ("M1+M2", "rank1", 0.01, 0.0, 0.025, 0.2663733626637336),
        ("M1+M2+M3", "rank1", 0.01, 0.0, 0.025, 0.263973602639736),
        ("M1+M2", "top2", -0.015, -0.035, 0.0, 0.09919008099190081),
    ):
        add("Main paired bootstrap", "ARC1", "test", f"{metric}_delta_vs_B1", method,
            delta, "fraction", bootstrap_source, "NEGATIVE", low, high,
            "10,000 paired puzzle bootstrap", f"two-sided p-style={p_value}")

    gap_source = "results/arc1_main/selection_gap.csv"
    for budget, pass_k, majority_k, gap in (
        (50, 0.520875, 0.392875, 0.128),
        (250, 0.579125, 0.400875, 0.17825),
        (1000, 0.6175, 0.4, 0.2175),
    ):
        for metric, value in (("pass_at_k", pass_k), ("majority_at_k", majority_k),
                              ("selection_gap", gap)):
            add("Candidate-generation/selection gap", "ARC1", "all", metric,
                f"k={budget}", value, "fraction", gap_source, "PRIMARY")

    coverage_source = "results/arc1_main/coverage_conditioned_results.csv"
    add("Coverage-conditioned selection", "ARC1", "test", "candidate_coverage", "orbit",
        0.6175, "fraction", coverage_source, "PRIMARY")
    add("Coverage-conditioned selection", "ARC1", "test", "rank1_given_coverage", "B1",
        0.6532258064516129, "fraction", coverage_source, "PRIMARY")
    add("Coverage-conditioned selection", "ARC1", "test", "rank1_given_coverage", "M1+M2",
        0.6693548387096774, "fraction", coverage_source, "PRIMARY")

    defect_source = "results/arc1_main/h1_auroc_results.csv"
    for split, value in (("test", 0.8418617947451856), ("all", 0.8215755208333333)):
        add("Task-level structural defect", "ARC1", split, "failure_AUROC", "defect",
            value, "area", defect_source, "PRIMARY")
    add("Task-level structural defect", "ARC1", "all", "defect_correctness_correlation",
        "defect", -0.3561751889305399, "correlation", defect_source, "PRIMARY")
    add("Task-level structural defect", "ARC1 blank-ID", "test", "failure_AUROC",
        "defect", 0.7708333333333334, "area",
        "results/arc1_blank_ablation/defect_auroc.csv", "ABLATION")
    add("Task-level structural defect", "ARC2", "all", "failure_AUROC", "defect",
        0.8279174330676476, "area", "results/arc2_supporting/defect_auroc.csv",
        "SUPPORTING")

    top_mode = "results/top_mode_diagnostic/top_mode_summary.json"
    for metric, value, unit in (
        ("covered_puzzles", 124, "puzzles"),
        ("covered_descriptors", 131, "descriptors"),
        ("top2_pooled_AUROC", 0.5451182454749637, "area"),
        ("top2_pairwise_accuracy", 0.7469135802469136, "fraction"),
        ("top5_pooled_AUROC", 0.5491473096148192, "area"),
        ("top5_within_puzzle_AUROC", 0.729868913857678, "area"),
        ("covered_B1_failure_puzzles", 46, "puzzles"),
        ("correct_alternative_preferred", 14, "puzzle-equivalents"),
        ("wrong_B1_preferred", 32, "puzzle-equivalents"),
    ):
        ci = {
            "top2_pairwise_accuracy": (0.654320987654321, 0.8333333333333334),
            "top5_within_puzzle_AUROC": (0.6643258426966292, 0.7940074906367042),
        }.get(metric, ("", ""))
        add("Local top-mode structural signal", "ARC1", "dev", metric, "M1 centrality",
            value, unit, top_mode, "SECONDARY", ci[0], ci[1],
            "10,000 puzzle bootstrap" if ci[0] != "" else "")

    beta_source = "results/m1_beta_diagnostic/m1_beta_summary.csv"
    beta_values = {
        -16: 0.28, -8: 0.325, -4: 0.3925, -2: 0.3925, -1: 0.3975,
        0: 0.3975, 1: 0.4025, 2: 0.4025, 4: 0.4025, 8: 0.3975, 16: 0.39,
    }
    for beta, accuracy in beta_values.items():
        add("M1 beta diagnostic", "ARC1", "dev", "rank1_accuracy", f"distinct beta={beta}",
            accuracy, "fraction", beta_source, "SECONDARY")
    beta_json = "results/m1_beta_diagnostic/m1_beta_summary.json"
    for metric, value, unit in (
        ("beta1_changed_decisions", 6, "puzzle-equivalents"),
        ("beta1_fixes", 1, "puzzle-equivalents"),
        ("beta1_breaks", 0, "puzzle-equivalents"),
        ("beta_flip_q25", 63.29631604288519, "beta"),
        ("beta_flip_median", 248.37414325754463, "beta"),
        ("beta_flip_q75", 993.9686545057837, "beta"),
        ("beta_flip_fraction_le_16", 0.08333333333333333, "fraction"),
        ("beta16_net_gain", -1.5, "puzzle-equivalents"),
    ):
        add("M1 beta diagnostic", "ARC1", "dev", metric, "distinct", value, unit,
            beta_json, "SECONDARY", note="MIXED / NON-MONOTONIC")

    discriminative = "results/discriminative_cell_dev/discriminative_cell_summary.json"
    for metric, value in (
        ("covered_top_mode_similarity_median", 0.96),
        ("covered_top_mode_disagreement_median", 0.04),
        ("covered_top_mode_fraction_ge_90pct_agreement", 0.7288135593220338),
        ("covered_top_mode_dominant_color_occupancy_median", 0.6026315789473684),
        ("covered_top_mode_agreement_explained_by_dominant_color_median",
         0.6085810668353362),
        ("B1_failure_similarity_median", 0.96),
        ("B1_failure_disagreement_median", 0.04),
        ("B1_failure_dominant_color_occupancy_median", 0.6681072428971588),
    ):
        add("Background dominance", "ARC1", "dev", metric, "descriptive", value,
            "fraction", discriminative, "SECONDARY")
    for method, beta, accuracy in (
        ("B1", 0, 0.3975),
        ("original_M1", 1, 0.4025),
        ("discriminative_M1", 1, 0.4),
        ("discriminative_M1", 4, 0.3875),
    ):
        add("Discriminative-cell DEV experiment", "ARC1", "dev", "rank1_accuracy",
            f"{method} beta={beta}", accuracy, "fraction", discriminative, "NEGATIVE",
            note="DEV ONLY; AMBIGUOUS / NO-GO FOR TEST")
    for metric, value in (
        ("original_correct_preference", 0.30434782608695654),
        ("original_wrong_preference", 0.6956521739130435),
        ("discriminative_correct_preference", 0.45652173913043476),
        ("discriminative_wrong_preference", 0.34782608695652173),
        ("discriminative_tie_fraction", 0.1956521739130435),
        ("original_top5_pairwise_accuracy", 0.729868913857678),
        ("discriminative_top5_pairwise_accuracy", 0.5994850187265918),
    ):
        add("Discriminative-cell DEV experiment", "ARC1", "dev", metric,
            "discriminative diagnostic", value, "fraction", discriminative, "NEGATIVE",
            note="No TEST evaluation performed")

    top2 = "results/top2_policy_audit/top2_policy_summary.csv"
    for method, value in (("B1", 0.455), ("M1", 0.455), ("M1+M2", 0.44),
                          ("M1+M2+M3", 0.44)):
        add("Top-2 policy audit", "ARC1", "test", "committed_top2_accuracy", method,
            value, "fraction", top2, "NEGATIVE")
    add("Top-2 policy audit", "ARC1", "dev", "different_shape_runner_up_found",
        "shape hedge", 0, "rankings", top2, "NEGATIVE", note="0 of 402 eligible rankings")
    add("Top-2 policy audit", "ARC1", "dev", "shape_hedged_top2_before", "M2/M3",
        0.4375, "fraction", top2, "NEGATIVE")
    add("Top-2 policy audit", "ARC1", "dev", "shape_hedged_top2_after", "M2/M3",
        0.4275, "fraction", top2, "NEGATIVE")

    compute = "results/compute_matched_q_audit/compute_matched_q_audit.csv"
    compute_values = {
        50: {"B1": 0.40475, "M1": 0.40375, "M1+M2": 0.40375,
             "M1+M2+M3": 0.40375},
        250: {"B1": 0.405, "M1": 0.4065, "M1+M2": 0.405,
              "M1+M2+M3": 0.405},
        1000: {"B1": 0.4035, "M1": 0.4035, "M1+M2": 0.4125,
               "M1+M2+M3": 0.4125},
    }
    for budget, methods_at_budget in compute_values.items():
        for method, value in methods_at_budget.items():
            add("Compute-matched audit", "ARC1", "all", "mean_rank1_accuracy",
                f"{method} k={budget}", value, "fraction", compute, "SECONDARY",
                note="IMPLEMENTATION CORRECT; no Q/full-orbit leakage")

    risk = "results/risk_coverage_bootstrap/risk_coverage_bootstrap_summary.json"
    add("Selective prediction", "ARC1", "test", "AURC", "structural_defect",
        0.31586167853807995, "area", risk, "PRIMARY", 0.24680880147422518,
        0.3948521220258649, "10,000 puzzle bootstrap")
    add("Selective prediction", "ARC1", "test", "expected_random_AURC", "random",
        0.5975, "area", risk, "PRIMARY")
    add("Selective prediction", "ARC1", "test", "AURC_delta_vs_random",
        "structural_defect", -0.28163832146191997, "area", risk, "PRIMARY",
        -0.3096076300081239, -0.2463203774253707, "10,000 paired puzzle bootstrap")
    for coverage, accuracy, low, high in (
        (0.25, 0.88, 0.78, 0.98),
        (0.5, 0.63, 0.52, 0.74),
        (0.75, 0.49, 0.4066666666666667, 0.5733333333333334),
        (1.0, 0.4025, 0.335, 0.4725),
    ):
        add("Selective prediction", "ARC1", "test", "retained_accuracy",
            f"structural_defect coverage={coverage}", accuracy, "fraction", risk,
            "PRIMARY", low, high, "10,000 puzzle bootstrap")

    for method, value in (
        ("structural_defect", 0.31586167853807995),
        ("vote_margin", 0.2607539081534486),
        ("vote_share", 0.26206564698082846),
        ("vote_entropy", 0.2776887416829336),
        ("winner_mean_Q", 0.3127768181492204),
    ):
        add("Confidence baseline comparison", "ARC1", "test", "AURC", method, value,
            "area", risk, "NEGATIVE")
    for baseline, delta, low, high in (
        ("vote_margin", 0.05510777038463133, 0.02703551869187188, 0.08649194590784394),
        ("vote_share", 0.05379603155725149, 0.025594280662340593, 0.08507348759741762),
        ("vote_entropy", 0.03817293685514633, 0.008445785063642137, 0.0704103186743909),
        ("winner_mean_Q", 0.0030848603888595227, -0.03849454231179205,
         0.04444328644283999),
    ):
        add("Confidence baseline comparison", "ARC1", "test",
            "defect_minus_baseline_AURC", baseline, delta, "area", risk, "NEGATIVE",
            low, high, "10,000 paired puzzle bootstrap",
            "positive means structural defect is worse")

    blank = "results/arc1_blank_ablation/analysis_report.json"
    add("Puzzle-ID blanking ablation", "ARC1 blank-ID", "all", "B1_accuracy", "B1",
        0.035, "fraction", blank, "ABLATION", note="retained-prediction evaluation")
    for method, value in (("B1", 0.04), ("M1", 0.04), ("M1+M2", 0.045),
                          ("M1+M2+M3", 0.05)):
        add("Puzzle-ID blanking ablation", "ARC1 blank-ID", "test", "rank1_accuracy",
            method, value, "fraction",
            "results/arc1_blank_ablation/frozen_method_results.csv", "ABLATION")
    add("Puzzle-ID blanking ablation", "ARC1 normal", "test", "B1_accuracy", "B1",
        0.4025, "fraction", main_source, "ABLATION")
    add("Puzzle-ID blanking ablation", "ARC1 blank-ID", "test",
        "blank_minus_normal_B1", "B1", -0.3625, "fraction",
        "results/arc1_blank_ablation/paired_bootstrap_cis.csv", "ABLATION", -0.43,
        -0.295, "10,000 paired puzzle bootstrap", "two-sided p-style=0.00019998")
    add("Puzzle-ID blanking ablation", "ARC1 normal", "test", "candidate_coverage",
        "orbit", 0.6175, "fraction", coverage_source, "ABLATION")
    add("Puzzle-ID blanking ablation", "ARC1 blank-ID", "test", "candidate_coverage",
        "orbit", 0.1025, "fraction",
        "results/arc1_blank_ablation/frozen_method_results.csv", "ABLATION")
    add("Blank-ID selective prediction", "ARC1 blank-ID", "test", "defect_AURC",
        "structural_defect", 0.922868554959021, "area", risk, "CAVEATED",
        0.8594246481700603, 0.9773523396824799, "10,000 puzzle bootstrap")
    add("Blank-ID selective prediction", "ARC1 blank-ID", "test",
        "defect_minus_random_AURC", "structural_defect", -0.03713144504097898,
        "area", risk, "CAVEATED", -0.08017539343748889, -0.0031207315016965487,
        "10,000 paired puzzle bootstrap", "eight puzzle-equivalent correct outcomes")

    arc2 = "results/arc2_supporting/analysis_report.json"
    for metric, value, unit in (
        ("puzzle_count", 120, "puzzles"),
        ("descriptor_count", 172, "descriptors"),
        ("candidate_count", 170393, "emissions"),
        ("invalid_prediction_count", 188, "emissions"),
        ("pinned_upstream_pass_at_1", 0.029166666666666667, "fraction"),
        ("online_B1", 0.029166666666666667, "fraction"),
        ("cache_B1", 0.029166666666666667, "fraction"),
        ("pinned_upstream_pass_at_2", 0.05, "fraction"),
        ("external_reported_accuracy", 0.062, "fraction"),
        ("candidate_coverage", 0.11666666666666667, "fraction"),
        ("selection_gap", 0.0875, "fraction"),
    ):
        add("ARC2 supporting evaluation", "ARC2", "all", metric, "pinned path", value,
            unit, arc2, "SUPPORTING", note="unresolved external reproduction discrepancy")
    for method, rank1, top2_value in (
        ("B1", 0.029166666666666667, 0.05),
        ("M1", 0.029166666666666667, 0.04583333333333333),
        ("M1+M2", 0.029166666666666667, 0.0375),
        ("M1+M2+M3", 0.029166666666666667, 0.0375),
    ):
        add("ARC2 supporting evaluation", "ARC2", "all", "rank1_accuracy", method,
            rank1, "fraction", "results/arc2_supporting/frozen_method_results.csv",
            "SUPPORTING")
        add("ARC2 supporting evaluation", "ARC2", "all", "top2_accuracy", method,
            top2_value, "fraction", "results/arc2_supporting/frozen_method_results.csv",
            "SUPPORTING")
    add("ARC2 selective prediction", "ARC2", "all", "defect_AURC",
        "structural_defect", 0.9180768700795726, "area", risk, "SUPPORTING",
        0.8261435387233548, 0.9978830911092326, "10,000 puzzle bootstrap")
    add("ARC2 selective prediction", "ARC2", "all", "defect_minus_random_AURC",
        "structural_defect", -0.05275646325376071, "area", risk, "SUPPORTING",
        -0.12127500165351199, 0.010857276406621201,
        "10,000 paired puzzle bootstrap", "ambiguous; only 3.5 puzzle-equivalent correct")
    return rows


def _claims() -> dict[str, Any]:
    return {
        "freeze_date": DATE,
        "status": "EXPERIMENT LAYER CLOSED",
        "primary": [
            "ARC1 has a 21.75 percentage-point candidate-generation/selection gap at k=1000.",
            "Orbit-level structural defect predicts held-out ARC1 failure (AUROC 0.841862).",
            "Candidate-level structured aggregation weakly exploits the signal: M1 has no gain, M2 has an exploratory +1 point, and M3 adds no gain.",
        ],
        "secondary_mechanistic": [
            "Candidate-level structural signal exists locally but is poorly aligned with exact B1 failures.",
            "Whole-grid similarity is dominated by agreement over mostly unchanged/background cells.",
            "Discriminative-cell weighting improves failure alignment but not DEV selector accuracy and was not evaluated on TEST.",
            "M1 weakness is not explained by beta merely being too small.",
            "Structural defect supports selective prediction relative to random ordering.",
        ],
        "negative": [
            "Structural defect is not better than vote margin, vote share, or vote entropy for selective prediction.",
        ],
        "ablation": [
            "Puzzle-ID blanking collapses candidate coverage from 61.75% to 10.25% on held-out ARC1.",
        ],
        "supporting": [
            "ARC2 qualitatively preserves the selection-gap/structural-signal pattern, but its absolute external-score discrepancy is unresolved.",
        ],
        "forbidden": [
            "M1/M2/M3 significantly outperform majority voting.",
            "The +1 percentage-point M2 result is statistically significant.",
            "Structural defect is better than vote margin, vote share, or vote entropy.",
            "Discriminative-cell weighting was validated on TEST or improves the method.",
            "The Top-2 regression was caused by a scoring bug.",
            "Compute-matched results contained Q or full-orbit leakage.",
            "ARC2 was reproduced at 2.9167%.",
            "Blank-ID was reproduced at 0% under retained-prediction evaluation.",
            "Invalid outputs were dropped.",
            "TEST was used to tune hyperparameters.",
            "ARC2 was used for tuning.",
        ],
    }


def _freeze_markdown(rows: list[dict[str, Any]], claims: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final experiment freeze",
            "",
            f"Freeze date: {DATE}.",
            "",
            "**EXPERIMENTS FROZEN AS OF FINAL PHASE-4 AUDIT.**",
            "",
            "The experiment layer is closed. No new method, tuning, held-out TEST evaluation, confidence statistic, selector, inference, cache generation, or ARC2 discrepancy experiment is authorized by this record.",
            "",
            "## Final scientific status",
            "",
            "- ARC1 baseline reproduction is exact: online B1 = cache B1 = published verification = 40.00% over 400 puzzles.",
            "- Held-out ARC1 B1 is separately 40.25%; these values refer to different populations and must not be conflated.",
            "- The strongest primary result is the k=1000 candidate-generation/selection gap: 61.75% coverage versus 40.00% majority accuracy, a 21.75-point gap.",
            "- Held-out defect AUROC is 0.841862. Defect reliably improves selective prediction over random ordering, but is reliably worse than vote margin/share/entropy by AURC.",
            "- M1 has no held-out Rank-1 gain; M2's +1 point is exploratory; M3 adds no gain. The committed M2/M3 Top-2 value is 44.00%.",
            "- Discriminative-cell weighting is a DEV-only ambiguous/no-go result; no TEST evaluation was performed.",
            "- Full retained-prediction blank-ID B1 is 3.5%; held-out blank-ID B1 is 4.00%. The upstream 0% is a padding-mask scorer artifact.",
            "- Pinned ARC2 is 2.9167%, supporting only, with an unresolved discrepancy from the external approximately 6.2% report. It is not a successful benchmark reproduction.",
            "",
            "## Frozen claim hierarchy",
            "",
            *[f"- **PRIMARY:** {item}" for item in claims["primary"]],
            *[f"- **SECONDARY:** {item}" for item in claims["secondary_mechanistic"]],
            *[f"- **NEGATIVE:** {item}" for item in claims["negative"]],
            *[f"- **ABLATION:** {item}" for item in claims["ablation"]],
            *[f"- **SUPPORTING:** {item}" for item in claims["supporting"]],
            "",
            "## Metric reconciliation",
            "",
            "| Value | Authoritative meaning |",
            "|---:|---|",
            "| 40.00% | Full 400-puzzle ARC1 baseline reproduction. |",
            "| 40.25% | Committed 200-puzzle ARC1 held-out TEST B1. |",
            "| 61.75% | Both held-out orbit coverage and full k=1000 pass@k; separate rows identify the context. |",
            "| 44.00% | Committed M2/M3 held-out TEST Top-2. |",
            "| 3.5% | Full retained-prediction blank-ID B1. |",
            "| 4.00% | Held-out blank-ID B1. |",
            "| 2.9167% | Internally consistent pinned ARC2 result, not external benchmark reproduction. |",
            "",
            "## Artifact contract",
            "",
            f"`final_results_manifest.json` and `final_results_summary.csv` contain {len(rows)} frozen headline records from one shared data structure. `final_claims.json` records allowed and forbidden claims. `final_artifact_checksums.csv` covers all non-manifest scientific files under `results/` except itself, avoiding recursive hashes.",
            "",
            "**EXPERIMENTAL RESULTS ARE FROZEN.**",
            "",
        ]
    )


def _write_checksums() -> int:
    checksum_path = FREEZE / "final_artifact_checksums.csv"
    paths = sorted(
        path
        for path in RESULTS.rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and path != checksum_path
    )
    rows = [
        {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "result_group": path.relative_to(RESULTS).parts[0],
        }
        for path in paths
    ]
    _write_csv(checksum_path, rows)
    return len(rows)


def main() -> int:
    _validate_sources()
    rows = _headline_rows()
    claims = _claims()
    FREEZE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "freeze_date": DATE,
        "status": "EXPERIMENT LAYER CLOSED",
        "headline_result_count": len(rows),
        "results": rows,
        "metric_reconciliation": {
            "40.00%": "full ARC1 baseline reproduction",
            "40.25%": "held-out ARC1 TEST B1",
            "61.75%": "context-dependent: held-out coverage and full k=1000 pass@k",
            "44.00%": "committed M2/M3 held-out TEST Top-2",
            "3.5%": "full retained-prediction blank-ID B1",
            "4.00%": "held-out blank-ID B1",
            "2.9167%": "pinned ARC2 result; not external benchmark reproduction",
        },
        "checksum_scope": "all non-manifest scientific result files except final_artifact_checksums.csv itself",
        "exact_freeze_statement": "EXPERIMENTAL RESULTS ARE FROZEN.",
    }
    _write_json(FREEZE / "final_results_manifest.json", manifest)
    _write_csv(FREEZE / "final_results_summary.csv", rows)
    _write_json(FREEZE / "final_claims.json", claims)
    (FREEZE / "FINAL_FREEZE.md").write_text(
        _freeze_markdown(rows, claims), encoding="utf-8"
    )
    checksum_count = _write_checksums()
    print(
        json.dumps(
            {
                "status": "passed",
                "headline_results": len(rows),
                "checksummed_scientific_files": checksum_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
