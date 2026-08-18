from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

from .aggregators import (
    M1Precomputation,
    m1_orbit_weighting,
    m2_cell_marginal_score,
    prepare_m1,
    two_attempt_policy,
)
from .analysis_pipeline import load_orbits
from .baselines import canonical_prediction, majority_vote
from .diagnostics import equivariance_defect, spearman_d_count, vote_concentration
from .sampling import deterministic_dev_test_split, paired_subsamples
from .schema import Candidate, RankedCandidate, TaskOrbit, grid_key
from .shape_screening import infer_shape_screen

METHODS = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")
AGGREGATION_METHODS = ("B1", "M1", "M1+M2", "M1+M2+M3")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _correct(ranking: Sequence[RankedCandidate], orbit: TaskOrbit, top_k: int) -> float:
    if orbit.target is None:
        raise ValueError(f"{orbit.task_id} has no target")
    target = grid_key(orbit.target)
    return float(any(grid_key(candidate.grid) == target for candidate in ranking[:top_k]))


def _covered(orbit: TaskOrbit) -> float:
    if orbit.target is None:
        raise ValueError(f"{orbit.task_id} has no target")
    target = grid_key(orbit.target)
    return float(any(grid_key(candidate.grid) == target for candidate in orbit.candidates))


def _rank(ranking: Sequence[RankedCandidate], orbit: TaskOrbit) -> float:
    if orbit.target is None:
        raise ValueError(f"{orbit.task_id} has no target")
    target = grid_key(orbit.target)
    for index, candidate in enumerate(ranking, start=1):
        if grid_key(candidate.grid) == target:
            return float(index)
    return math.inf


def puzzle_values(
    values: Sequence[float], orbits: Sequence[TaskOrbit]
) -> tuple[tuple[str, ...], np.ndarray]:
    if len(values) != len(orbits):
        raise ValueError("values and orbits must have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, orbit in zip(values, orbits):
        grouped[orbit.puzzle_id].append(float(value))
    puzzle_ids = tuple(sorted(grouped))
    return puzzle_ids, np.asarray(
        [float(np.mean(grouped[puzzle_id])) for puzzle_id in puzzle_ids],
        dtype=np.float64,
    )


def puzzle_mean(values: Sequence[float], orbits: Sequence[TaskOrbit]) -> float:
    return float(np.mean(puzzle_values(values, orbits)[1]))


def _identity_ranking(orbit: TaskOrbit) -> tuple[RankedCandidate, ...]:
    identities = [candidate for candidate in orbit.candidates if candidate.is_identity]
    if len(identities) != 1:
        raise AssertionError(
            f"expected exactly one identity candidate for {orbit.task_id}; got {len(identities)}"
        )
    candidate = identities[0]
    grid = canonical_prediction(orbit)
    return (
        RankedCandidate(
            grid,
            1.0,
            1,
            candidate.q_value,
            is_invalid=candidate.is_invalid,
        ),
    )


def _suborbit(orbit: TaskOrbit, indices: Sequence[int]) -> TaskOrbit:
    candidates = tuple(
        Candidate(
            grid=orbit.candidates[index].grid,
            augmentation_index=position,
            q_value=orbit.candidates[index].q_value,
            transform=orbit.candidates[index].transform,
            is_identity=orbit.candidates[index].is_identity,
            is_invalid=orbit.candidates[index].is_invalid,
            entropy=orbit.candidates[index].entropy,
            top3_colors=orbit.candidates[index].top3_colors,
        )
        for position, index in enumerate(indices)
    )
    return TaskOrbit(
        task_id=orbit.task_id,
        candidates=candidates,
        query_input=orbit.query_input,
        support_pairs=orbit.support_pairs,
        target=orbit.target,
        metadata={
            **{key: value for key, value in orbit.metadata.items() if key != "emitted_orbit_size"},
            "is_budget_subsample": True,
        },
    )


def _assert_finite_ranking(ranking: Sequence[RankedCandidate], label: str) -> None:
    if not ranking:
        raise AssertionError(f"empty ranking for {label}")
    for candidate in ranking:
        if not math.isfinite(candidate.score) or not math.isfinite(candidate.mean_q):
            raise AssertionError(f"non-finite ranking score for {label}")


def evaluate_orbits(
    orbits: Sequence[TaskOrbit],
    parameters: Mapping[str, Any],
    *,
    prepared: Mapping[str, M1Precomputation] | None = None,
    include_b0: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for orbit in orbits:
        item = prepared[orbit.task_id] if prepared is not None else prepare_m1(orbit)
        b1 = majority_vote(orbit)
        m1 = m1_orbit_weighting(
            orbit,
            float(parameters["beta"]),
            str(parameters["definition"]),
            prepared=item,
        )
        m2 = m2_cell_marginal_score(
            orbit,
            float(parameters["interpolation"]),
            float(parameters["epsilon"]),
            emission_weights=m1.emission_weights,
            marginal_support=str(parameters["marginal_support"]),
        )
        m3 = m2_cell_marginal_score(
            orbit,
            float(parameters["interpolation"]),
            float(parameters["epsilon"]),
            emission_weights=m1.emission_weights,
            marginal_support=str(parameters["marginal_support"]),
            use_shape_screening=True,
        )
        outputs: dict[str, tuple[Sequence[RankedCandidate], Sequence[RankedCandidate]]] = {
            "B1": (b1, b1[:2]),
            "M1": (m1.ranking, m1.ranking[:2]),
            "M1+M2": (m2.ranking, two_attempt_policy(m2.ranking, b1)),
            "M1+M2+M3": (m3.ranking, two_attempt_policy(m3.ranking, b1)),
        }
        if include_b0:
            b0 = _identity_ranking(orbit)
            outputs = {"B0": (b0, b0), **outputs}
        covered = _covered(orbit)
        for method, (ranking, attempts) in outputs.items():
            _assert_finite_ranking(ranking, f"{orbit.task_id}:{method}")
            records[method]["rank1"].append(_correct(ranking, orbit, 1))
            records[method]["top2"].append(_correct(attempts, orbit, 2))
            records[method]["coverage"].append(covered)
            records[method]["rank"].append(_rank(ranking, orbit))
    return {
        method: {metric: np.asarray(values, dtype=np.float64) for metric, values in metrics.items()}
        for method, metrics in records.items()
    }


def summarize_methods(
    outcomes: Mapping[str, Mapping[str, np.ndarray]], orbits: Sequence[TaskOrbit]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        if method not in outcomes:
            continue
        arrays = outcomes[method]
        reciprocal = np.where(np.isfinite(arrays["rank"]), 1.0 / arrays["rank"], 0.0)
        rows.append(
            {
                "method": method,
                "puzzle_count": len({orbit.puzzle_id for orbit in orbits}),
                "descriptor_count": len(orbits),
                "rank1_accuracy": puzzle_mean(arrays["rank1"], orbits),
                "top2_accuracy": puzzle_mean(arrays["top2"], orbits),
                "coverage": puzzle_mean(arrays["coverage"], orbits),
                "selection_gap": puzzle_mean(arrays["coverage"] - arrays["rank1"], orbits),
                "mrr": puzzle_mean(reciprocal, orbits),
            }
        )
    return rows


def _weighted_auc(
    scores: Sequence[float], labels: Sequence[bool], weights: Sequence[float]
) -> float:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=bool)
    weight = np.asarray(weights, dtype=np.float64)
    positive = np.flatnonzero(label)
    negative = np.flatnonzero(~label)
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both classes")
    numerator = 0.0
    for index in positive:
        comparisons = (score[index] > score[negative]).astype(np.float64)
        comparisons += 0.5 * (score[index] == score[negative])
        numerator += weight[index] * float(np.sum(weight[negative] * comparisons))
    return numerator / (float(np.sum(weight[positive])) * float(np.sum(weight[negative])))


def _weighted_correlation(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float]
) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight = weight / np.sum(weight)
    left_centered = left - np.sum(weight * left)
    right_centered = right - np.sum(weight * right)
    denominator = math.sqrt(
        float(np.sum(weight * left_centered**2) * np.sum(weight * right_centered**2))
    )
    return (
        0.0
        if denominator == 0
        else float(np.sum(weight * left_centered * right_centered) / denominator)
    )


def _descriptor_weights(orbits: Sequence[TaskOrbit]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for orbit in orbits:
        counts[orbit.puzzle_id] += 1
    return np.asarray([1.0 / counts[orbit.puzzle_id] for orbit in orbits], dtype=np.float64)


def orbit_diagnostics(
    orbits: Sequence[TaskOrbit], prepared: Mapping[str, M1Precomputation]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for orbit in orbits:
        workspace = prepared[orbit.task_id].similarity
        concentration = vote_concentration(orbit, workspace=workspace)
        counts = np.asarray(list(workspace.counts.values()), dtype=np.float64)
        invalid = sum(candidate.is_invalid for candidate in orbit.candidates)
        defect = equivariance_defect(orbit, workspace=workspace)
        rows.append(
            {
                "task_id": orbit.task_id,
                "puzzle_id": orbit.puzzle_id,
                "pair_index": orbit.test_pair_index,
                "emitted_candidates": len(orbit.candidates),
                "distinct_candidate_grids": concentration.unique_candidates,
                "orbit_dispersion": defect,
                "majority_vote_mass": concentration.modal_share,
                "top2_vote_mass": concentration.cumulative_mass[min(1, len(counts) - 1)],
                "top5_vote_mass": concentration.cumulative_mass[min(4, len(counts) - 1)],
                "mean_candidate_multiplicity": float(np.mean(counts)),
                "max_candidate_multiplicity": int(np.max(counts)),
                "singleton_grid_fraction": float(np.mean(counts == 1)),
                "invalid_emissions": invalid,
                "invalid_emission_fraction": invalid / len(orbit.candidates),
                "b1_correct": _correct(majority_vote(orbit), orbit, 1),
                "d_count_distinct": spearman_d_count(orbit, "distinct", workspace=workspace),
                "d_count_multiset": spearman_d_count(orbit, "multiset", workspace=workspace),
                "d_count_non_identical_support": spearman_d_count(
                    orbit, "non_identical_support", workspace=workspace
                ),
            }
        )
    weights = _descriptor_weights(orbits)
    summary: list[dict[str, Any]] = []
    for metric in (
        "emitted_candidates",
        "distinct_candidate_grids",
        "orbit_dispersion",
        "majority_vote_mass",
        "top2_vote_mass",
        "mean_candidate_multiplicity",
        "max_candidate_multiplicity",
        "singleton_grid_fraction",
        "invalid_emissions",
        "invalid_emission_fraction",
    ):
        values = np.asarray([float(row[metric]) for row in rows])
        summary.append(
            {
                "metric": metric,
                "puzzle_weighted_mean": float(np.average(values, weights=weights)),
                "descriptor_median": float(np.median(values)),
                "descriptor_min": float(np.min(values)),
                "descriptor_max": float(np.max(values)),
            }
        )
    d_rows: list[dict[str, Any]] = []
    for definition in ("distinct", "multiset", "non_identical_support"):
        field = f"d_count_{definition}"
        values = np.asarray([float(row[field]) for row in rows])
        finite = np.isfinite(values)
        d_rows.append(
            {
                "definition": definition,
                "finite_descriptors": int(np.sum(finite)),
                "undefined_descriptors": int(np.sum(~finite)),
                "puzzle_weighted_mean_spearman": (
                    float(np.average(values[finite], weights=weights[finite]))
                    if np.any(finite)
                    else None
                ),
                "descriptor_median_spearman": (
                    float(np.median(values[finite])) if np.any(finite) else None
                ),
            }
        )
    return rows, summary, d_rows


def h1_rows(
    diagnostics: Sequence[Mapping[str, Any]], split_by_puzzle: Mapping[str, str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("all", "dev", "test"):
        selected = [
            row
            for row in diagnostics
            if split == "all" or split_by_puzzle[str(row["puzzle_id"])] == split
        ]
        counts: dict[str, int] = defaultdict(int)
        for row in selected:
            counts[str(row["puzzle_id"])] += 1
        weights = [1.0 / counts[str(row["puzzle_id"])] for row in selected]
        defects = [float(row["orbit_dispersion"]) for row in selected]
        correct = [bool(row["b1_correct"]) for row in selected]
        output.append(
            {
                "split": split,
                "puzzles": len(counts),
                "descriptors": len(selected),
                "auroc_defect_for_incorrectness": _weighted_auc(
                    defects, [not item for item in correct], weights
                ),
                "auroc_negative_defect_for_correctness": _weighted_auc(
                    [-item for item in defects], correct, weights
                ),
                "puzzle_weighted_point_biserial_defect_correctness": _weighted_correlation(
                    defects, [float(item) for item in correct], weights
                ),
            }
        )
    return output


def tune_m1(
    dev: Sequence[TaskOrbit],
    prepared: Mapping[str, M1Precomputation],
    definitions: Sequence[str],
    betas: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for definition_index, definition in enumerate(definitions):
        for beta in betas:
            rankings = [
                m1_orbit_weighting(
                    orbit, float(beta), str(definition), prepared=prepared[orbit.task_id]
                ).ranking
                for orbit in dev
            ]
            for orbit, ranking in zip(dev, rankings):
                _assert_finite_ranking(ranking, f"m1-dev:{orbit.task_id}")
            rows.append(
                {
                    "definition": str(definition),
                    "definition_order": definition_index,
                    "beta": float(beta),
                    "dev_rank1_accuracy": puzzle_mean(
                        [_correct(ranking, orbit, 1) for orbit, ranking in zip(dev, rankings)],
                        dev,
                    ),
                    "dev_top2_accuracy": puzzle_mean(
                        [_correct(ranking, orbit, 2) for orbit, ranking in zip(dev, rankings)],
                        dev,
                    ),
                }
            )
    best = max(
        rows,
        key=lambda row: (
            row["dev_rank1_accuracy"],
            row["dev_top2_accuracy"],
            -abs(row["beta"]),
            -row["definition_order"],
        ),
    )
    return dict(best), rows


def tune_m2(
    dev: Sequence[TaskOrbit],
    prepared: Mapping[str, M1Precomputation],
    *,
    m1: Mapping[str, Any],
    interpolations: Sequence[float],
    epsilons: Sequence[float],
    supports: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weights = {
        orbit.task_id: m1_orbit_weighting(
            orbit,
            float(m1["beta"]),
            str(m1["definition"]),
            prepared=prepared[orbit.task_id],
        ).emission_weights
        for orbit in dev
    }
    b1 = {orbit.task_id: majority_vote(orbit) for orbit in dev}
    rows: list[dict[str, Any]] = []
    for support_index, support in enumerate(supports):
        for interpolation in interpolations:
            for epsilon in epsilons:
                rankings = [
                    m2_cell_marginal_score(
                        orbit,
                        float(interpolation),
                        float(epsilon),
                        emission_weights=weights[orbit.task_id],
                        marginal_support=str(support),
                    ).ranking
                    for orbit in dev
                ]
                attempts = [
                    two_attempt_policy(ranking, b1[orbit.task_id])
                    for orbit, ranking in zip(dev, rankings)
                ]
                rows.append(
                    {
                        "marginal_support": str(support),
                        "support_order": support_index,
                        "interpolation": float(interpolation),
                        "epsilon": float(epsilon),
                        "dev_rank1_accuracy": puzzle_mean(
                            [_correct(ranking, orbit, 1) for orbit, ranking in zip(dev, rankings)],
                            dev,
                        ),
                        "dev_top2_accuracy": puzzle_mean(
                            [_correct(item, orbit, 2) for orbit, item in zip(dev, attempts)],
                            dev,
                        ),
                    }
                )
    best = max(
        rows,
        key=lambda row: (
            row["dev_rank1_accuracy"],
            row["dev_top2_accuracy"],
            -row["interpolation"],
            -row["epsilon"],
            -row["support_order"],
        ),
    )
    return dict(best), rows


def _bootstrap(
    method: np.ndarray,
    baseline: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if method.shape != baseline.shape or method.ndim != 1:
        raise ValueError("bootstrap inputs must be paired puzzle vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(method), size=(resamples, len(method)))
    method_samples = np.mean(method[indices], axis=1)
    difference_samples = np.mean((method - baseline)[indices], axis=1)
    low, high = np.quantile(difference_samples, [0.025, 0.975])
    accuracy_low, accuracy_high = np.quantile(method_samples, [0.025, 0.975])
    left = (np.sum(difference_samples <= 0) + 1) / (resamples + 1)
    right = (np.sum(difference_samples >= 0) + 1) / (resamples + 1)
    return {
        "accuracy": float(np.mean(method)),
        "accuracy_ci95_low": float(accuracy_low),
        "accuracy_ci95_high": float(accuracy_high),
        "baseline_accuracy": float(np.mean(baseline)),
        "difference": float(np.mean(method - baseline)),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(left, right))),
    }


def bootstrap_rows(
    outcomes: Mapping[str, Mapping[str, np.ndarray]],
    test: Sequence[TaskOrbit],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(("B1", "M1", "M1+M2", "M1+M2+M3")):
        for metric_index, metric in enumerate(("rank1", "top2")):
            puzzle_ids, values = puzzle_values(outcomes[method][metric], test)
            baseline_ids, baseline = puzzle_values(outcomes["B1"][metric], test)
            if puzzle_ids != baseline_ids:
                raise AssertionError("bootstrap puzzle alignment failure")
            stats = _bootstrap(
                values,
                baseline,
                resamples=resamples,
                seed=seed + method_index * 101 + metric_index,
            )
            rows.append(
                {
                    "method": method,
                    "baseline": "B1",
                    "metric": metric,
                    "puzzle_count": len(puzzle_ids),
                    "resamples": resamples,
                    **stats,
                }
            )
    return rows


def selection_gap_rows(
    orbits: Sequence[TaskOrbit], *, budgets: Sequence[int], repeats: int, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        plans = [
            paired_subsamples(orbit, budget=budget, repeats=repeats, seed=seed) for orbit in orbits
        ]
        for repeat in range(repeats):
            pass_values: list[float] = []
            majority_values: list[float] = []
            for orbit, plan in zip(orbits, plans):
                sampled = _suborbit(orbit, plan.repeats[repeat])
                pass_values.append(_covered(sampled))
                majority_values.append(_correct(majority_vote(sampled), sampled, 1))
            pass_at_k = puzzle_mean(pass_values, orbits)
            majority_at_k = puzzle_mean(majority_values, orbits)
            rows.append(
                {
                    "budget": int(budget),
                    "repeat": repeat,
                    "puzzle_count": len({orbit.puzzle_id for orbit in orbits}),
                    "descriptor_count": len(orbits),
                    "effective_budget_min": min(plan.effective_budget for plan in plans),
                    "effective_budget_max": max(plan.effective_budget for plan in plans),
                    "clamped_descriptors": sum(plan.was_clamped for plan in plans),
                    "pass_at_k": pass_at_k,
                    "majority_at_k": majority_at_k,
                    "selection_gap": pass_at_k - majority_at_k,
                }
            )
    return rows


def compute_matched_rows(
    test: Sequence[TaskOrbit],
    *,
    budgets: Sequence[int],
    repeats: int,
    seed: int,
    parameters: Mapping[str, Any],
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        plans = [
            paired_subsamples(orbit, budget=budget, repeats=repeats, seed=seed) for orbit in test
        ]
        for repeat in range(repeats):
            sampled = tuple(
                _suborbit(orbit, plan.repeats[repeat]) for orbit, plan in zip(test, plans)
            )
            outcomes = evaluate_orbits(sampled, parameters, include_b0=False)
            for summary in summarize_methods(outcomes, sampled):
                rows.append(
                    {
                        "budget": int(budget),
                        "repeat": repeat,
                        "effective_budget_min": min(plan.effective_budget for plan in plans),
                        "effective_budget_max": max(plan.effective_budget for plan in plans),
                        "clamped_descriptors": sum(plan.was_clamped for plan in plans),
                        **summary,
                    }
                )
            if checkpoint_path is not None:
                _write_csv(checkpoint_path, rows)
    return rows


def shape_diagnostics_rows(
    test: Sequence[TaskOrbit],
    prepared: Mapping[str, M1Precomputation],
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in test:
        m1 = m1_orbit_weighting(
            orbit,
            float(parameters["beta"]),
            str(parameters["definition"]),
            prepared=prepared[orbit.task_id],
        )
        plain = m2_cell_marginal_score(
            orbit,
            float(parameters["interpolation"]),
            float(parameters["epsilon"]),
            emission_weights=m1.emission_weights,
            marginal_support=str(parameters["marginal_support"]),
        )
        screened = m2_cell_marginal_score(
            orbit,
            float(parameters["interpolation"]),
            float(parameters["epsilon"]),
            emission_weights=m1.emission_weights,
            marginal_support=str(parameters["marginal_support"]),
            use_shape_screening=True,
        )
        screen = infer_shape_screen(orbit.support_pairs, orbit.query_input.shape)
        matching = [
            candidate
            for candidate in orbit.candidates
            if candidate.grid.shape in screen.allowed_shapes
        ]
        retained = (
            orbit.candidates
            if screened.used_shape_fallback or not screen.allowed_shapes
            else matching
        )
        rows.append(
            {
                "task_id": orbit.task_id,
                "puzzle_id": orbit.puzzle_id,
                "pair_index": orbit.test_pair_index,
                "emitted_candidates": len(orbit.candidates),
                "invalid_candidates_before": sum(
                    candidate.is_invalid for candidate in orbit.candidates
                ),
                "allowed_shapes": json.dumps(sorted(screen.allowed_shapes)),
                "shape_rule_count": len(screen.relations),
                "filter_active": int(bool(screen.allowed_shapes)),
                "retained_candidates": len(retained),
                "filtered_candidates": len(orbit.candidates) - len(retained),
                "filter_fraction": 1.0 - len(retained) / len(orbit.candidates),
                "invalid_candidates_retained": sum(candidate.is_invalid for candidate in retained),
                "used_empty_filter_fallback": int(screened.used_shape_fallback),
                "target_shape_allowed": int(orbit.target.shape in screen.allowed_shapes),
                "plain_modal_shape": json.dumps(plain.modal_shape),
                "screened_modal_shape": json.dumps(screened.modal_shape),
                "covered": _covered(orbit),
            }
        )
    return rows


def coverage_and_rank_rows(
    outcomes: Mapping[str, Mapping[str, np.ndarray]], test: Sequence[TaskOrbit]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    puzzle_ids = sorted({orbit.puzzle_id for orbit in test})
    indices = {puzzle_id: [] for puzzle_id in puzzle_ids}
    for index, orbit in enumerate(test):
        indices[orbit.puzzle_id].append(index)
    for method in METHODS:
        arrays = outcomes[method]
        conditioned_rank1: list[float] = []
        conditioned_top2: list[float] = []
        conditioned_mrr: list[float] = []
        covered_puzzles = 0
        for puzzle_id in puzzle_ids:
            selected = [index for index in indices[puzzle_id] if arrays["coverage"][index] > 0]
            if not selected:
                continue
            covered_puzzles += 1
            conditioned_rank1.append(float(np.mean(arrays["rank1"][selected])))
            conditioned_top2.append(float(np.mean(arrays["top2"][selected])))
            conditioned_mrr.append(
                float(
                    np.mean(
                        np.where(
                            np.isfinite(arrays["rank"][selected]),
                            1.0 / arrays["rank"][selected],
                            0.0,
                        )
                    )
                )
            )
        coverage_rows.append(
            {
                "method": method,
                "covered_puzzles": covered_puzzles,
                "rank1_given_covered": float(np.mean(conditioned_rank1)),
                "top2_given_covered": float(np.mean(conditioned_top2)),
                "mrr_given_covered": float(np.mean(conditioned_mrr)),
            }
        )
        reciprocal = np.where(np.isfinite(arrays["rank"]), 1.0 / arrays["rank"], 0.0)
        finite_ranks = arrays["rank"][np.isfinite(arrays["rank"])]
        rank_rows.append(
            {
                "method": method,
                "puzzle_weighted_mrr_all": puzzle_mean(reciprocal, test),
                "target_ranked_coverage": puzzle_mean(
                    np.isfinite(arrays["rank"]).astype(float), test
                ),
                "descriptor_median_finite_rank": float(np.median(finite_ranks))
                if len(finite_ranks)
                else None,
                "max_finite_rank": float(np.max(finite_ranks)) if len(finite_ranks) else None,
            }
        )
    return coverage_rows, rank_rows


def error_correlation_rows(
    test: Sequence[TaskOrbit], prepared: Mapping[str, M1Precomputation]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in test:
        workspace = prepared[orbit.task_id].similarity
        target = grid_key(orbit.target)
        incorrect_indices = [index for index, key in enumerate(workspace.keys) if key != target]
        block = workspace.matrix[np.ix_(incorrect_indices, incorrect_indices)]
        upper = block[np.triu_indices(len(incorrect_indices), k=1)]
        if target in workspace.keys:
            target_index = workspace.keys.index(target)
            target_similarity = workspace.matrix[target_index, incorrect_indices]
        else:
            target_similarity = np.asarray([], dtype=np.float32)
        rows.append(
            {
                "task_id": orbit.task_id,
                "puzzle_id": orbit.puzzle_id,
                "incorrect_distinct_candidates": len(incorrect_indices),
                "mean_incorrect_pair_similarity": float(np.mean(upper)) if len(upper) else None,
                "mean_correct_to_incorrect_similarity": (
                    float(np.mean(target_similarity)) if len(target_similarity) else None
                ),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def run_arc1_normal_analysis(
    *, cache_dir: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite analysis directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config["experiment"]
    if list(experiment["budgets"]) != [50, 250, 1000]:
        raise AssertionError("committed budget grid changed")
    if int(experiment["subsamples_per_budget"]) != 10:
        raise AssertionError("committed subsample count changed")
    if int(experiment["bootstrap_resamples"]) != 10_000:
        raise AssertionError("committed bootstrap count changed")

    orbits = load_orbits(cache_dir)
    puzzle_ids = tuple(sorted({orbit.puzzle_id for orbit in orbits}))
    invalid_count = sum(candidate.is_invalid for orbit in orbits for candidate in orbit.candidates)
    if len(orbits) != 419 or len(puzzle_ids) != 400:
        raise AssertionError("ARC-v1 cache must contain 419 descriptors and 400 puzzles")
    if invalid_count != 92:
        raise AssertionError(f"expected 92 invalid predictions; got {invalid_count}")
    b1_full = puzzle_mean([_correct(majority_vote(orbit), orbit, 1) for orbit in orbits], orbits)
    if b1_full != 0.4:
        raise AssertionError(f"full-cache B1 reproduction changed: {b1_full}")

    dev_ids, test_ids = deterministic_dev_test_split(
        puzzle_ids,
        seed=int(experiment["seed"]),
        dev_fraction=float(experiment["dev_fraction"]),
    )
    dev_set, test_set = set(dev_ids), set(test_ids)
    if dev_set & test_set or dev_set | test_set != set(puzzle_ids):
        raise AssertionError("invalid puzzle-level split")
    dev = tuple(orbit for orbit in orbits if orbit.puzzle_id in dev_set)
    test = tuple(orbit for orbit in orbits if orbit.puzzle_id in test_set)
    split_by_puzzle = {puzzle_id: "dev" for puzzle_id in dev_ids} | {
        puzzle_id: "test" for puzzle_id in test_ids
    }
    split_payload = {
        "seed": int(experiment["seed"]),
        "dev_fraction": float(experiment["dev_fraction"]),
        "dev_puzzle_ids": list(dev_ids),
        "test_puzzle_ids": list(test_ids),
        "dev_task_ids": [orbit.task_id for orbit in dev],
        "test_task_ids": [orbit.task_id for orbit in test],
        "dev_puzzle_count": len(dev_ids),
        "test_puzzle_count": len(test_ids),
        "dev_descriptor_count": len(dev),
        "test_descriptor_count": len(test),
    }
    _write_json(output_dir / "committed_split.json", split_payload)

    prepared = {orbit.task_id: prepare_m1(orbit) for orbit in orbits}
    diagnostic_rows, diagnostic_summary, d_count_rows = orbit_diagnostics(orbits, prepared)
    for row in diagnostic_rows:
        row["split"] = split_by_puzzle[str(row["puzzle_id"])]
    _write_csv(output_dir / "orbit_statistics.csv", diagnostic_rows)
    _write_csv(output_dir / "orbit_statistics_summary.csv", diagnostic_summary)
    _write_csv(output_dir / "h1_auroc_results.csv", h1_rows(diagnostic_rows, split_by_puzzle))
    _write_csv(output_dir / "d_count_diagnostic.csv", d_count_rows)

    selection_rows = selection_gap_rows(
        orbits,
        budgets=experiment["budgets"],
        repeats=int(experiment["subsamples_per_budget"]),
        seed=int(experiment["seed"]),
    )
    _write_csv(output_dir / "selection_gap.csv", selection_rows)

    m1_best, m1_rows = tune_m1(
        dev,
        prepared,
        config["m1"]["centrality_definitions"],
        config["m1"]["beta_grid"],
    )
    _write_csv(output_dir / "m1_dev_sweep.csv", m1_rows)
    _write_json(output_dir / "selected_m1.json", m1_best)
    m2_best, m2_rows = tune_m2(
        dev,
        prepared,
        m1=m1_best,
        interpolations=config["m2"]["lambda_grid"],
        epsilons=config["m2"]["epsilon_grid"],
        supports=config["m2"]["marginal_support"],
    )
    _write_csv(output_dir / "m2_dev_sweep.csv", m2_rows)
    _write_json(output_dir / "selected_m2.json", m2_best)
    support_ablation: list[dict[str, Any]] = []
    for support in config["m2"]["marginal_support"]:
        candidates = [row for row in m2_rows if row["marginal_support"] == support]
        support_ablation.append(
            max(
                candidates,
                key=lambda row: (
                    row["dev_rank1_accuracy"],
                    row["dev_top2_accuracy"],
                    -row["interpolation"],
                    -row["epsilon"],
                ),
            )
        )
    _write_csv(output_dir / "marginal_support_ablation.csv", support_ablation)

    parameters = {
        "definition": str(m1_best["definition"]),
        "beta": float(m1_best["beta"]),
        "interpolation": float(m2_best["interpolation"]),
        "epsilon": float(m2_best["epsilon"]),
        "marginal_support": str(m2_best["marginal_support"]),
    }
    frozen = {
        "selection_split": "dev",
        "dev_puzzle_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
        "m1": m1_best,
        "m2": m2_best,
        "parameters": parameters,
    }
    _write_json(output_dir / "frozen_hyperparameters.json", frozen)

    test_outcomes = evaluate_orbits(test, parameters, prepared=prepared)
    main_rows = summarize_methods(test_outcomes, test)
    _write_csv(output_dir / "main_test_results.csv", main_rows)
    bootstrap = bootstrap_rows(
        test_outcomes,
        test,
        resamples=int(experiment["bootstrap_resamples"]),
        seed=int(experiment["seed"]),
    )
    _write_csv(output_dir / "paired_bootstrap_cis.csv", bootstrap)
    shapes = shape_diagnostics_rows(test, prepared, parameters)
    _write_csv(output_dir / "shape_screening_diagnostics.csv", shapes)
    coverage_rows, rank_rows = coverage_and_rank_rows(test_outcomes, test)
    _write_csv(output_dir / "coverage_conditioned_results.csv", coverage_rows)
    _write_csv(output_dir / "rank_mrr_results.csv", rank_rows)
    _write_csv(output_dir / "error_correlation_results.csv", error_correlation_rows(test, prepared))

    compute_rows = compute_matched_rows(
        test,
        budgets=experiment["budgets"],
        repeats=int(experiment["subsamples_per_budget"]),
        seed=int(experiment["seed"]),
        parameters=parameters,
        checkpoint_path=output_dir / "compute_matched_results.csv",
    )
    _write_csv(output_dir / "compute_matched_results.csv", compute_rows)

    test_b1 = next(row["rank1_accuracy"] for row in main_rows if row["method"] == "B1")
    if any(not math.isfinite(float(row["rank1_accuracy"])) for row in main_rows):
        raise AssertionError("non-finite main result")
    if parameters != frozen["parameters"]:
        raise AssertionError("test parameters were not the dev-frozen parameters")
    shape_weights = _descriptor_weights(test)
    report = {
        "status": "passed",
        "scope": "ARC-v1 normal cache only",
        "cache_path": str(cache_dir.resolve()),
        "cache_treated_as_immutable": True,
        "counts": {
            "puzzles": len(puzzle_ids),
            "descriptors": len(orbits),
            "candidates": sum(len(orbit.candidates) for orbit in orbits),
            "invalid_predictions": invalid_count,
            "dev_puzzles": len(dev_ids),
            "test_puzzles": len(test_ids),
            "dev_descriptors": len(dev),
            "test_descriptors": len(test),
        },
        "validation": {
            "puzzle_level_split": True,
            "puzzle_weighted_metrics": True,
            "puzzle_level_bootstrap": True,
            "bootstrap_resamples": int(experiment["bootstrap_resamples"]),
            "full_cache_b1": b1_full,
            "invalid_predictions_preserved": invalid_count,
            "test_parameters_frozen_from_dev": True,
            "finite_main_metrics": True,
        },
        "frozen_hyperparameters": frozen,
        "main_test_results": main_rows,
        "h1": h1_rows(diagnostic_rows, split_by_puzzle),
        "m3_summary": {
            "puzzle_weighted_filter_fraction": float(
                np.average([row["filter_fraction"] for row in shapes], weights=shape_weights)
            ),
            "puzzle_weighted_fallback_rate": float(
                np.average(
                    [row["used_empty_filter_fallback"] for row in shapes], weights=shape_weights
                )
            ),
            "puzzle_weighted_target_shape_allowed": float(
                np.average([row["target_shape_allowed"] for row in shapes], weights=shape_weights)
            ),
        },
        "pending": [
            "puzzle-ID ablation (blank-ID cache unavailable)",
            "ARC-v2 supporting analysis (ARC-v2 cache unavailable)",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "output_files": sorted(path.name for path in output_dir.iterdir()),
        "test_b1": test_b1,
    }
    _write_json(output_dir / "analysis_report.json", report)
    _write_json(output_dir / "artifact_manifest.json", _manifest(output_dir))
    return report
