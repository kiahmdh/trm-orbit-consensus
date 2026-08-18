from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
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
from .baselines import canonical_prediction, majority_vote
from .cache import load_task_orbit
from .diagnostics import (
    defect_correctness_statistics,
    equivariance_defect,
    spearman_d_count,
    vote_concentration,
)
from .evaluation import paired_bootstrap_difference
from .sampling import deterministic_dev_test_split, paired_subsamples
from .schema import Candidate, RankedCandidate, TaskOrbit, grid_key
from .similarity import build_similarity_workspace, grid_similarity


def load_orbits(cache_dir: Path) -> tuple[TaskOrbit, ...]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no task caches found in {cache_dir}")
    orbits = tuple(load_task_orbit(path) for path in paths)
    ids = [orbit.task_id for orbit in orbits]
    if len(ids) != len(set(ids)):
        raise AssertionError("cache contains duplicate task IDs")
    return tuple(sorted(orbits, key=lambda orbit: orbit.task_id))


def _correct(ranking: Sequence[RankedCandidate], orbit: TaskOrbit, top_k: int) -> bool:
    if orbit.target is None:
        raise ValueError(f"task {orbit.task_id} has no target")
    target = grid_key(orbit.target)
    return any(grid_key(candidate.grid) == target for candidate in ranking[:top_k])


def _covered(orbit: TaskOrbit) -> bool:
    if orbit.target is None:
        raise ValueError(f"task {orbit.task_id} has no target")
    target = grid_key(orbit.target)
    return any(grid_key(candidate.grid) == target for candidate in orbit.candidates)


def _rank(ranking: Sequence[RankedCandidate], orbit: TaskOrbit) -> int | None:
    if orbit.target is None:
        raise ValueError(f"task {orbit.task_id} has no target")
    target = grid_key(orbit.target)
    for index, candidate in enumerate(ranking, start=1):
        if grid_key(candidate.grid) == target:
            return index
    return None


def _arc_average(values: Sequence[float], orbits: Sequence[TaskOrbit]) -> float:
    by_puzzle: dict[str, list[float]] = defaultdict(list)
    for value, orbit in zip(values, orbits):
        by_puzzle[orbit.puzzle_id].append(float(value))
    return float(np.mean([np.mean(items) for items in by_puzzle.values()]))


def _identity_ranking(orbit: TaskOrbit) -> tuple[RankedCandidate, ...]:
    grid = canonical_prediction(orbit)
    return (RankedCandidate(grid, 1.0, 1, 0.0, is_invalid=grid.shape == (0, 0)),)


def _suborbit(orbit: TaskOrbit, indices: Sequence[int]) -> TaskOrbit:
    selected = tuple(
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
        candidates=selected,
        query_input=orbit.query_input,
        support_pairs=orbit.support_pairs,
        target=orbit.target,
        metadata={
            **{key: value for key, value in orbit.metadata.items() if key != "emitted_orbit_size"},
            "is_budget_subsample": True,
        },
    )


def _method_rankings(
    orbit: TaskOrbit,
    *,
    beta: float,
    definition: str,
    interpolation: float,
    epsilon: float,
    marginal_support: str,
    prepared: M1Precomputation | None = None,
) -> dict[str, tuple[tuple[RankedCandidate, ...], tuple[RankedCandidate, ...]]]:
    identities = [candidate for candidate in orbit.candidates if candidate.is_identity]
    b0 = (
        _identity_ranking(orbit)
        if len(identities) == 1 and not orbit.metadata.get("is_budget_subsample")
        else None
    )
    b1 = majority_vote(orbit)
    prepared = prepared or prepare_m1(orbit)
    m1 = m1_orbit_weighting(orbit, beta, definition, prepared=prepared)
    m2 = m2_cell_marginal_score(
        orbit,
        interpolation,
        epsilon,
        emission_weights=m1.emission_weights,
        marginal_support=marginal_support,
    )
    m3 = m2_cell_marginal_score(
        orbit,
        interpolation,
        epsilon,
        emission_weights=m1.emission_weights,
        marginal_support=marginal_support,
        use_shape_screening=True,
    )
    outputs = {
        "B1": (b1, b1[:2]),
        "M1": (m1.ranking, m1.ranking[:2]),
        "M1+M2": (m2.ranking, two_attempt_policy(m2.ranking, b1)),
        "M1+M2+M3": (m3.ranking, two_attempt_policy(m3.ranking, b1)),
    }
    return ({"B0": (b0, b0), **outputs} if b0 is not None else outputs)


def _evaluate_methods(
    orbits: Sequence[TaskOrbit],
    *,
    beta: float,
    definition: str,
    interpolation: float,
    epsilon: float,
    marginal_support: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    records: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for orbit in orbits:
        outputs = _method_rankings(
            orbit,
            beta=beta,
            definition=definition,
            interpolation=interpolation,
            epsilon=epsilon,
            marginal_support=marginal_support,
        )
        covered = _covered(orbit)
        for method, (ranking, attempts) in outputs.items():
            rank = _rank(ranking, orbit)
            records[method]["rank1"].append(float(_correct(ranking, orbit, 1)))
            records[method]["top2"].append(float(_correct(attempts, orbit, 2)))
            records[method]["coverage"].append(float(covered))
            records[method]["rank"].append(float(rank) if rank is not None else np.inf)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for method, metrics in records.items():
        arrays[method] = {key: np.asarray(value) for key, value in metrics.items()}
        covered_mask = arrays[method]["coverage"].astype(bool)
        ranks = arrays[method]["rank"][covered_mask]
        reciprocal = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
        rows.append(
            {
                "method": method,
                "rank1": _arc_average(arrays[method]["rank1"], orbits),
                "top2": _arc_average(arrays[method]["top2"], orbits),
                "coverage": _arc_average(arrays[method]["coverage"], orbits),
                "selection_gap_rank1": _arc_average(
                    arrays[method]["coverage"] - arrays[method]["rank1"], orbits
                ),
                "covered_tasks": int(np.sum(covered_mask)),
                "mrr_covered": float(np.mean(reciprocal)) if len(reciprocal) else np.nan,
                "median_rank_covered": float(np.median(ranks)) if len(ranks) else np.nan,
            }
        )
    return rows, arrays


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tune_m1(
    orbits: Sequence[TaskOrbit], definitions: Sequence[str], betas: Sequence[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = {orbit.task_id: prepare_m1(orbit) for orbit in orbits}
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        for beta in betas:
            rankings = [
                m1_orbit_weighting(
                    orbit, beta, definition, prepared=prepared[orbit.task_id]
                ).ranking
                for orbit in orbits
            ]
            rank1 = _arc_average(
                [float(_correct(ranking, orbit, 1)) for orbit, ranking in zip(orbits, rankings)],
                orbits,
            )
            top2 = _arc_average(
                [float(_correct(ranking, orbit, 2)) for orbit, ranking in zip(orbits, rankings)],
                orbits,
            )
            rows.append(
                {"definition": definition, "beta": beta, "rank1": rank1, "top2": top2}
            )
    best = max(rows, key=lambda row: (row["rank1"], row["top2"], -abs(row["beta"])))
    return dict(best), rows


def _tune_m2(
    orbits: Sequence[TaskOrbit],
    *,
    beta: float,
    definition: str,
    interpolations: Sequence[float],
    epsilons: Sequence[float],
    supports: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = {orbit.task_id: prepare_m1(orbit) for orbit in orbits}
    weights = {
        orbit.task_id: m1_orbit_weighting(
            orbit, beta, definition, prepared=prepared[orbit.task_id]
        ).emission_weights
        for orbit in orbits
    }
    rows: list[dict[str, Any]] = []
    for marginal_support in supports:
        for interpolation in interpolations:
            for epsilon in epsilons:
                rankings = [
                    m2_cell_marginal_score(
                        orbit,
                        interpolation,
                        epsilon,
                        emission_weights=weights[orbit.task_id],
                        marginal_support=marginal_support,
                    ).ranking
                    for orbit in orbits
                ]
                rank1 = _arc_average(
                    [
                        float(_correct(ranking, orbit, 1))
                        for orbit, ranking in zip(orbits, rankings)
                    ],
                    orbits,
                )
                b1s = [majority_vote(orbit) for orbit in orbits]
                attempts = [
                    two_attempt_policy(ranking, b1)
                    for ranking, b1 in zip(rankings, b1s)
                ]
                top2 = _arc_average(
                    [
                        float(_correct(attempt, orbit, 2))
                        for orbit, attempt in zip(orbits, attempts)
                    ],
                    orbits,
                )
                rows.append(
                    {
                        "marginal_support": marginal_support,
                        "interpolation": interpolation,
                        "epsilon": epsilon,
                        "rank1": rank1,
                        "top2": top2,
                    }
                )
    best = max(
        rows,
        key=lambda row: (
            row["rank1"],
            row["top2"],
            -row["interpolation"],
            -row["epsilon"],
        ),
    )
    return dict(best), rows


def _diagnostics(orbits: Sequence[TaskOrbit]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    defects: list[float] = []
    correctness: list[bool] = []
    for orbit in orbits:
        workspace = build_similarity_workspace(candidate.grid for candidate in orbit.candidates)
        concentration = vote_concentration(orbit, workspace=workspace)
        defect = equivariance_defect(orbit, workspace=workspace)
        b1_correct = _correct(majority_vote(orbit), orbit, 1)
        defects.append(defect)
        correctness.append(b1_correct)
        rows.append(
            {
                "task_id": orbit.task_id,
                "unique_candidates": concentration.unique_candidates,
                "modal_share": concentration.modal_share,
                "top2_vote_mass": concentration.cumulative_mass[
                    min(1, len(concentration.cumulative_mass) - 1)
                ],
                "top5_vote_mass": concentration.cumulative_mass[
                    min(4, len(concentration.cumulative_mass) - 1)
                ],
                "top10_vote_mass": concentration.cumulative_mass[
                    min(9, len(concentration.cumulative_mass) - 1)
                ],
                "equivariance_defect": defect,
                "b1_correct": int(b1_correct),
                "d_count_distinct": spearman_d_count(
                    orbit, "distinct", workspace=workspace
                ),
                "d_count_multiset": spearman_d_count(
                    orbit, "multiset", workspace=workspace
                ),
                "d_count_non_identical_support": spearman_d_count(
                    orbit, "non_identical_support", workspace=workspace
                ),
            }
        )
    auc, correlation = defect_correctness_statistics(
        np.asarray(defects), np.asarray(correctness)
    )
    summary = {
        "equivariance_defect_auroc_for_correctness": auc,
        "equivariance_defect_point_biserial": correlation,
        "median_unique_candidates": float(np.median([row["unique_candidates"] for row in rows])),
        "median_modal_share": float(np.median([row["modal_share"] for row in rows])),
    }
    return rows, summary


def _compute_matched(
    orbits: Sequence[TaskOrbit],
    *,
    budgets: Sequence[int],
    repeats: int,
    seed: int,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        plans = [
            paired_subsamples(orbit, budget=budget, repeats=repeats, seed=seed)
            for orbit in orbits
        ]
        for repeat in range(repeats):
            sampled = [
                _suborbit(orbit, plan.repeats[repeat])
                for orbit, plan in zip(orbits, plans)
            ]
            result_rows, _arrays = _evaluate_methods(sampled, **parameters)
            for row in result_rows:
                rows.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "effective_budget_min": min(plan.effective_budget for plan in plans),
                        **row,
                    }
                )
    return rows


def _error_correlation(orbits: Sequence[TaskOrbit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in orbits:
        target_key = grid_key(orbit.target) if orbit.target is not None else None
        unique: dict[Any, np.ndarray] = {}
        for candidate in orbit.candidates:
            unique.setdefault(grid_key(candidate.grid), candidate.grid)
        incorrect = [grid for key, grid in unique.items() if key != target_key]
        similarities = [
            grid_similarity(incorrect[left], incorrect[right])
            for left in range(len(incorrect))
            for right in range(left + 1, len(incorrect))
        ]
        correct_grid = orbit.target
        correct_to_incorrect = (
            [grid_similarity(correct_grid, grid) for grid in incorrect]
            if correct_grid is not None and target_key in unique
            else []
        )
        rows.append(
            {
                "task_id": orbit.task_id,
                "incorrect_unique_candidates": len(incorrect),
                "mean_incorrect_pair_similarity": (
                    float(np.mean(similarities)) if similarities else np.nan
                ),
                "mean_correct_to_incorrect_similarity": (
                    float(np.mean(correct_to_incorrect)) if correct_to_incorrect else np.nan
                ),
            }
        )
    return rows


def run_analysis(
    *,
    arc1_cache: Path,
    arc2_cache: Path,
    arc1_blank_cache: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)

    arc1 = load_orbits(arc1_cache)
    arc2 = load_orbits(arc2_cache)
    arc1_blank = load_orbits(arc1_blank_cache)
    if {orbit.task_id for orbit in arc1_blank} != {orbit.task_id for orbit in arc1}:
        raise AssertionError("normal and blank-ID ARC-1 caches have different task IDs")

    experiment = config["experiment"]
    dev_ids, test_ids = deterministic_dev_test_split(
        [orbit.task_id for orbit in arc1],
        seed=int(experiment["seed"]),
        dev_fraction=float(experiment["dev_fraction"]),
    )
    split_payload = {"seed": experiment["seed"], "dev": dev_ids, "test": test_ids}
    (output_dir / "split.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )
    dev_set, test_set = set(dev_ids), set(test_ids)
    dev = tuple(orbit for orbit in arc1 if orbit.task_id in dev_set)
    test = tuple(orbit for orbit in arc1 if orbit.task_id in test_set)

    diagnostic_rows, diagnostic_summary = _diagnostics(arc1)
    for row in diagnostic_rows:
        row["split"] = "dev" if row["task_id"] in dev_set else "test"
    _write_csv(output_dir / "task_diagnostics.csv", diagnostic_rows)

    m1_best, m1_rows = _tune_m1(
        dev,
        config["m1"]["centrality_definitions"],
        config["m1"]["beta_grid"],
    )
    _write_csv(output_dir / "m1_dev_sweep.csv", m1_rows)
    m2_best, m2_rows = _tune_m2(
        dev,
        beta=float(m1_best["beta"]),
        definition=str(m1_best["definition"]),
        interpolations=config["m2"]["lambda_grid"],
        epsilons=config["m2"]["epsilon_grid"],
        supports=config["m2"]["marginal_support"],
    )
    _write_csv(output_dir / "m2_dev_sweep.csv", m2_rows)

    parameters = {
        "beta": float(m1_best["beta"]),
        "definition": str(m1_best["definition"]),
        "interpolation": float(m2_best["interpolation"]),
        "epsilon": float(m2_best["epsilon"]),
        "marginal_support": str(m2_best["marginal_support"]),
    }
    frozen = {"m1": m1_best, "m2": m2_best, "combined": parameters}
    (output_dir / "frozen_hyperparameters.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    test_rows, test_arrays = _evaluate_methods(test, **parameters)
    _write_csv(output_dir / "main_test_results.csv", test_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    for method in ("M1", "M1+M2", "M1+M2+M3"):
        for metric in ("rank1", "top2"):
            difference, low, high = paired_bootstrap_difference(
                test_arrays[method][metric],
                test_arrays["B1"][metric],
                resamples=int(experiment["bootstrap_resamples"]),
                seed=int(experiment["seed"]),
            )
            bootstrap_rows.append(
                {
                    "method": method,
                    "baseline": "B1",
                    "metric": metric,
                    "difference": difference,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    _write_csv(output_dir / "paired_bootstrap_cis.csv", bootstrap_rows)

    compute_rows = _compute_matched(
        test,
        budgets=experiment["budgets"],
        repeats=int(experiment["subsamples_per_budget"]),
        seed=int(experiment["seed"]),
        parameters=parameters,
    )
    _write_csv(output_dir / "compute_matched_results.csv", compute_rows)

    arc2_rows, _arc2_arrays = _evaluate_methods(arc2, **parameters)
    _write_csv(output_dir / "arc2_supporting_results.csv", arc2_rows)
    blank_rows, _blank_arrays = _evaluate_methods(arc1_blank, **parameters)
    _write_csv(output_dir / "puzzle_id_ablation_results.csv", blank_rows)
    _write_csv(output_dir / "error_correlation.csv", _error_correlation(arc1))
    _write_csv(
        output_dir / "shape_screening_diagnostics.csv",
        _shape_diagnostics(test, parameters),
    )
    shape_hedge_rows = _shape_hedge_rows(dev, parameters, "dev")
    shape_hedge_rows.extend(_shape_hedge_rows(test, parameters, "test"))
    _write_csv(
        output_dir / "shape_hedged_second_attempt.csv",
        shape_hedge_rows,
    )
    coverage_rows = []
    for method, arrays in test_arrays.items():
        covered = arrays["coverage"].astype(bool)
        coverage_rows.append(
            {
                "method": method,
                "covered_tasks": int(np.sum(covered)),
                "rank1_given_covered": float(np.mean(arrays["rank1"][covered])),
                "top2_given_covered": float(np.mean(arrays["top2"][covered])),
                "mrr_given_covered": float(
                    np.mean(
                        np.where(
                            np.isfinite(arrays["rank"][covered]),
                            1.0 / arrays["rank"][covered],
                            0.0,
                        )
                    )
                ),
            }
        )
    _write_csv(output_dir / "coverage_conditioned_results.csv", coverage_rows)

    summary = {
        "task_counts": {
            "arc1": len(arc1),
            "arc1_dev": len(dev),
            "arc1_test": len(test),
            "arc1_blank": len(arc1_blank),
            "arc2": len(arc2),
        },
        "diagnostics": diagnostic_summary,
        "frozen_hyperparameters": frozen,
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary

def _shape_diagnostics(
    orbits: Sequence[TaskOrbit], parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in orbits:
        prepared = prepare_m1(orbit)
        m1 = m1_orbit_weighting(
            orbit,
            parameters["beta"],
            parameters["definition"],
            prepared=prepared,
        )
        plain = m2_cell_marginal_score(
            orbit,
            parameters["interpolation"],
            parameters["epsilon"],
            emission_weights=m1.emission_weights,
            marginal_support=parameters["marginal_support"],
        )
        screened = m2_cell_marginal_score(
            orbit,
            parameters["interpolation"],
            parameters["epsilon"],
            emission_weights=m1.emission_weights,
            marginal_support=parameters["marginal_support"],
            use_shape_screening=True,
        )
        target_shape = orbit.target.shape if orbit.target is not None else None
        covered = _covered(orbit)
        allowed = (
            screened.shape_screen.allowed_shapes
            if screened.shape_screen
            else frozenset()
        )
        rows.append(
            {
                "task_id": orbit.task_id,
                "covered": int(covered),
                "modal_shape_matches_target": int(target_shape == plain.modal_shape),
                "shape_selection_loss_on_covered": int(
                    covered and target_shape != plain.modal_shape
                ),
                "m3_rule_count": (
                    len(screened.shape_screen.relations)
                    if screened.shape_screen
                    else 0
                ),
                "m3_filter_active": int(bool(allowed)),
                "m3_target_shape_allowed": int(target_shape in allowed) if allowed else -1,
                "m3_used_empty_filter_fallback": int(screened.used_shape_fallback),
            }
        )
    return rows

def _shape_hedge_rows(
    orbits: Sequence[TaskOrbit], parameters: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    standard: dict[str, list[float]] = defaultdict(list)
    shape_hedged: dict[str, list[float]] = defaultdict(list)
    methods = ("M1+M2", "M1+M2+M3")
    for orbit in orbits:
        outputs = _method_rankings(orbit, **parameters)
        for method in methods:
            ranking, attempts = outputs[method]
            standard[method].append(float(_correct(attempts, orbit, 2)))
            if len(ranking) < 2:
                hedged = ranking
            else:
                runner_up_shape = next(
                    (
                        candidate
                        for candidate in ranking[1:]
                        if candidate.grid.shape != ranking[0].grid.shape
                    ),
                    ranking[1],
                )
                hedged = (ranking[0], runner_up_shape)
            shape_hedged[method].append(float(_correct(hedged, orbit, 2)))
    return [
        {
            "split": split,
            "method": method,
            "standard_top2": _arc_average(standard[method], orbits),
            "shape_hedged_top2": _arc_average(shape_hedged[method], orbits),
            "difference": _arc_average(
                np.asarray(shape_hedged[method]) - np.asarray(standard[method]),
                orbits,
            ),
        }
        for method in methods
    ]
