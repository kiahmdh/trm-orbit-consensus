from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .aggregators import M1Precomputation, prepare_m1
from .analysis_pipeline import load_orbits
from .arc1_analysis import (
    _bootstrap,
    _correct,
    _manifest,
    _weighted_auc,
    _weighted_correlation,
    _write_csv,
    _write_json,
    bootstrap_rows,
    coverage_and_rank_rows,
    evaluate_orbits,
    h1_rows,
    orbit_diagnostics,
    puzzle_mean,
    puzzle_values,
    selection_gap_rows,
    shape_diagnostics_rows,
    summarize_methods,
)
from .baselines import majority_vote
from .schema import TaskOrbit, grid_key

FROZEN_PARAMETERS = {
    "definition": "distinct",
    "beta": 1.0,
    "interpolation": 0.05,
    "epsilon": 1e-6,
    "marginal_support": "emitted",
}


def upstream_evaluator_ranking(
    orbit: TaskOrbit, *, include_invalid: bool = True
) -> tuple[tuple[int, int, bytes], ...]:
    """Reproduce pinned ``evaluators/arc.py`` voting after canonicalization.

    Dict insertion order is intentionally retained as the final tie behavior because
    upstream sorts only by ``[count, mean_q]`` with a stable Python sort.
    """
    vote_map: dict[tuple[int, int, bytes], list[float]] = {}
    for candidate in orbit.candidates:
        if candidate.is_invalid and not include_invalid:
            continue
        key = grid_key(candidate.grid)
        vote_map.setdefault(key, [0.0, 0.0])
        vote_map[key][0] += 1.0
        vote_map[key][1] += candidate.q_value
    if not vote_map:
        return ()
    for statistics in vote_map.values():
        statistics[1] /= statistics[0]
    return tuple(
        key
        for key, _ in sorted(
            vote_map.items(), key=lambda item: item[1], reverse=True
        )
    )


def upstream_evaluator_scores(
    orbits: Sequence[TaskOrbit],
    *,
    pass_ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
    include_invalid: bool = True,
) -> dict[str, float]:
    values = {int(k): [] for k in pass_ks}
    for orbit in orbits:
        ranking = upstream_evaluator_ranking(orbit, include_invalid=include_invalid)
        target = grid_key(orbit.target)
        for k in pass_ks:
            values[int(k)].append(float(target in ranking[: int(k)]))
    return {f"pass@{k}": puzzle_mean(item, orbits) for k, item in values.items()}


def _cache_fingerprint(cache_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    paths = sorted(Path(cache_dir).glob("*.npz"))
    total_bytes = 0
    for path in paths:
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(path.name.encode())
        digest.update(file_digest)
        total_bytes += path.stat().st_size
    return {
        "npz_files": len(paths),
        "npz_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }


def _require_new_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_contract(main_results: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    split = json.loads((main_results / "committed_split.json").read_text())
    frozen = json.loads((main_results / "frozen_hyperparameters.json").read_text())
    if frozen["parameters"] != FROZEN_PARAMETERS:
        raise AssertionError(
            f"ARC1 frozen parameters changed: {frozen['parameters']}"
        )
    expected_hash = hashlib.sha256(
        "\n".join(split["dev_puzzle_ids"]).encode()
    ).hexdigest()
    if frozen["dev_puzzle_ids_sha256"] != expected_hash:
        raise AssertionError("committed split does not match frozen parameter record")
    if len(split["dev_puzzle_ids"]) != 200 or len(split["test_puzzle_ids"]) != 200:
        raise AssertionError("committed ARC1 split must contain 200/200 puzzles")
    return split, frozen


def _prepare(orbits: Sequence[TaskOrbit]) -> dict[str, M1Precomputation]:
    return {orbit.task_id: prepare_m1(orbit) for orbit in orbits}


def _invalid_statistics(orbits: Sequence[TaskOrbit]) -> dict[str, Any]:
    invalid = sum(candidate.is_invalid for orbit in orbits for candidate in orbit.candidates)
    invalid_top1 = 0
    invalid_top2 = 0
    descriptors_with_invalid = 0
    for orbit in orbits:
        ranking = upstream_evaluator_ranking(orbit)
        descriptors_with_invalid += int(any(c.is_invalid for c in orbit.candidates))
        invalid_key = (0, 0, b"")
        invalid_top1 += int(bool(ranking) and ranking[0] == invalid_key)
        invalid_top2 += int(invalid_key in ranking[:2])
    return {
        "candidate_count": sum(len(orbit.candidates) for orbit in orbits),
        "invalid_candidates": invalid,
        "invalid_candidate_fraction": invalid
        / sum(len(orbit.candidates) for orbit in orbits),
        "descriptors_with_invalid": descriptors_with_invalid,
        "invalid_ranked_top1_descriptors": invalid_top1,
        "invalid_ranked_top2_descriptors": invalid_top2,
        "upstream_scores_including_invalid": upstream_evaluator_scores(orbits),
        "upstream_scores_excluding_invalid_sensitivity": upstream_evaluator_scores(
            orbits, include_invalid=False
        ),
    }


def _summarize_selection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget"])].append(row)
    output = []
    for budget, items in sorted(grouped.items()):
        output.append(
            {
                "budget": budget,
                "repeats": len(items),
                "pass_at_k_mean": float(np.mean([x["pass_at_k"] for x in items])),
                "majority_at_k_mean": float(
                    np.mean([x["majority_at_k"] for x in items])
                ),
                "selection_gap_mean": float(
                    np.mean([x["selection_gap"] for x in items])
                ),
                "effective_budget_min": min(x["effective_budget_min"] for x in items),
                "effective_budget_max": max(x["effective_budget_max"] for x in items),
                "clamped_descriptors": max(x["clamped_descriptors"] for x in items),
            }
        )
    return output


def _all_defect_row(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in diagnostics:
        counts[str(row["puzzle_id"])] += 1
    weights = [1.0 / counts[str(row["puzzle_id"])] for row in diagnostics]
    defects = [float(row["orbit_dispersion"]) for row in diagnostics]
    correct = [bool(row["b1_correct"]) for row in diagnostics]
    return {
        "split": "all",
        "puzzles": len(counts),
        "descriptors": len(diagnostics),
        "auroc_defect_for_incorrectness": _weighted_auc(
            defects, [not value for value in correct], weights
        ),
        "auroc_negative_defect_for_correctness": _weighted_auc(
            [-value for value in defects], correct, weights
        ),
        "puzzle_weighted_point_biserial_defect_correctness": _weighted_correlation(
            defects, [float(value) for value in correct], weights
        ),
    }


def _method_bootstrap_against_normal(
    blank_outcomes: Mapping[str, Mapping[str, np.ndarray]],
    blank: Sequence[TaskOrbit],
    normal: Sequence[TaskOrbit],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    normal_values = [
        _correct(majority_vote(orbit), orbit, 1) for orbit in normal
    ]
    normal_ids, normal_puzzles = puzzle_values(normal_values, normal)
    blank_ids, blank_puzzles = puzzle_values(blank_outcomes["B1"]["rank1"], blank)
    if normal_ids != blank_ids:
        raise AssertionError("normal/blank bootstrap puzzle alignment failed")
    return [
        {
            "method": "blank B1",
            "baseline": "normal B1",
            "metric": "rank1",
            "puzzle_count": len(blank_ids),
            "resamples": resamples,
            **_bootstrap(
                blank_puzzles,
                normal_puzzles,
                resamples=resamples,
                seed=seed,
            ),
        }
    ]


def _arc1_inventory_comparison(
    normal: Sequence[TaskOrbit], blank: Sequence[TaskOrbit]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normal_map = {orbit.task_id: orbit for orbit in normal}
    blank_map = {orbit.task_id: orbit for orbit in blank}
    if set(normal_map) != set(blank_map):
        raise AssertionError("normal and blank task inventories differ")
    mismatches: dict[str, list[str]] = defaultdict(list)
    correct_rows: list[dict[str, Any]] = []
    identical_emissions = 0
    candidate_count = 0
    for task_id in sorted(normal_map):
        left, right = normal_map[task_id], blank_map[task_id]
        if len(left.candidates) != len(right.candidates):
            mismatches["candidate_count"].append(task_id)
            continue
        if [c.augmentation_index for c in left.candidates] != [
            c.augmentation_index for c in right.candidates
        ]:
            mismatches["augmentation_index"].append(task_id)
        if [dict(c.transform) for c in left.candidates] != [
            dict(c.transform) for c in right.candidates
        ]:
            mismatches["transform"].append(task_id)
        for field in ("query_input", "target"):
            if not np.array_equal(getattr(left, field), getattr(right, field)):
                mismatches[field].append(task_id)
        same = sum(
            grid_key(a.grid) == grid_key(b.grid)
            for a, b in zip(left.candidates, right.candidates, strict=True)
        )
        identical_emissions += same
        candidate_count += len(right.candidates)
        ranking = upstream_evaluator_ranking(right)
        target = grid_key(right.target)
        if not ranking or ranking[0] != target:
            continue
        normal_ranking = upstream_evaluator_ranking(left)
        target_votes = sum(grid_key(c.grid) == target for c in right.candidates)
        runner_up_votes = (
            sum(grid_key(c.grid) == ranking[1] for c in right.candidates)
            if len(ranking) > 1
            else 0
        )
        identity = next(c for c in right.candidates if c.is_identity)
        overlap_fraction = same / len(right.candidates)
        correct_rows.append(
            {
                "task_id": task_id,
                "puzzle_id": right.puzzle_id,
                "pair_index": right.test_pair_index,
                "candidate_count": len(right.candidates),
                "target_votes": target_votes,
                "runner_up_votes": runner_up_votes,
                "majority_margin": target_votes - runner_up_votes,
                "majority_share": target_votes / len(right.candidates),
                "same_normal_blank_emissions": same,
                "same_normal_blank_fraction": overlap_fraction,
                "normal_top1_same": normal_ranking[0] == ranking[0],
                "normal_top1_correct": normal_ranking[0] == target,
                "identity_prediction_correct": grid_key(identity.grid) == target,
                "query_equals_target": grid_key(right.query_input) == target,
                "target_monochrome": len(np.unique(right.target)) == 1,
                "fully_identifier_invariant": overlap_fraction == 1.0,
                "substantial_emission_overlap": overlap_fraction >= 0.5,
                "sparse_plurality": target_votes / len(right.candidates) < 0.02,
            }
        )
    if any(mismatches.values()):
        raise AssertionError(f"normal/blank inventory mismatch: {dict(mismatches)}")
    return (
        {
            "task_id_sets_equal": True,
            "candidate_counts_equal": True,
            "augmentation_indices_equal": True,
            "inverse_transform_identifiers_equal": True,
            "queries_targets_equal": True,
            "candidate_count": candidate_count,
            "identical_emissions": identical_emissions,
            "identical_emission_fraction": identical_emissions / candidate_count,
            "blank_correct_descriptors": len(correct_rows),
            "blank_correct_puzzles": len({row["puzzle_id"] for row in correct_rows}),
        },
        correct_rows,
    )


def run_arc1_blank_analysis(
    *,
    normal_cache: Path,
    blank_cache: Path,
    main_results: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    _require_new_output(output_dir)
    before = {
        "normal": _cache_fingerprint(normal_cache),
        "blank": _cache_fingerprint(blank_cache),
    }
    split, frozen = _load_contract(Path(main_results))
    normal = load_orbits(normal_cache)
    blank = load_orbits(blank_cache)
    if len(blank) != 419 or len({orbit.puzzle_id for orbit in blank}) != 400:
        raise AssertionError("ARC1 blank cache inventory must be 400 puzzles/419 descriptors")
    inventory, correct_rows = _arc1_inventory_comparison(normal, blank)
    _write_csv(output_dir / "blank_correct_descriptors.csv", correct_rows)
    _write_json(output_dir / "cache_inventory_comparison.json", inventory)

    test_ids = set(split["test_puzzle_ids"])
    split_by_puzzle = {
        puzzle_id: "dev" for puzzle_id in split["dev_puzzle_ids"]
    } | {puzzle_id: "test" for puzzle_id in split["test_puzzle_ids"]}
    blank_test = tuple(orbit for orbit in blank if orbit.puzzle_id in test_ids)
    normal_test = tuple(orbit for orbit in normal if orbit.puzzle_id in test_ids)
    if [orbit.task_id for orbit in blank_test] != [orbit.task_id for orbit in normal_test]:
        raise AssertionError("committed-test normal/blank descriptors are not aligned")

    prepared = _prepare(blank)
    diagnostics, diagnostic_summary, d_count = orbit_diagnostics(blank, prepared)
    for row in diagnostics:
        row["split"] = split_by_puzzle[str(row["puzzle_id"])]
    defect = h1_rows(diagnostics, split_by_puzzle)
    _write_csv(output_dir / "orbit_statistics.csv", diagnostics)
    _write_csv(output_dir / "orbit_statistics_summary.csv", diagnostic_summary)
    _write_csv(output_dir / "d_count_diagnostic.csv", d_count)
    _write_csv(output_dir / "defect_auroc.csv", defect)

    outcomes = evaluate_orbits(
        blank_test, FROZEN_PARAMETERS, prepared=prepared
    )
    methods = summarize_methods(outcomes, blank_test)
    _write_csv(output_dir / "frozen_method_results.csv", methods)
    bootstrap = bootstrap_rows(
        outcomes, blank_test, resamples=10_000, seed=20260807
    )
    bootstrap += _method_bootstrap_against_normal(
        outcomes,
        blank_test,
        normal_test,
        resamples=10_000,
        seed=20260807 + 10_000,
    )
    _write_csv(output_dir / "paired_bootstrap_cis.csv", bootstrap)
    coverage, ranks = coverage_and_rank_rows(outcomes, blank_test)
    _write_csv(output_dir / "coverage_conditioned_results.csv", coverage)
    _write_csv(output_dir / "rank_mrr_results.csv", ranks)
    shapes = shape_diagnostics_rows(blank_test, prepared, FROZEN_PARAMETERS)
    _write_csv(output_dir / "shape_screening_diagnostics.csv", shapes)
    selection = selection_gap_rows(
        blank_test,
        budgets=(50, 250, 1000),
        repeats=10,
        seed=20260807,
    )
    selection_summary = _summarize_selection(selection)
    _write_csv(output_dir / "selection_gap_repeats.csv", selection)
    _write_csv(output_dir / "selection_gap_summary.csv", selection_summary)
    invalid_all = _invalid_statistics(blank)
    invalid_test = _invalid_statistics(blank_test)
    _write_json(
        output_dir / "invalid_prediction_statistics.json",
        {"all": invalid_all, "committed_test": invalid_test},
    )

    normal_test_b1 = puzzle_mean(
        [_correct(majority_vote(orbit), orbit, 1) for orbit in normal_test], normal_test
    )
    after = {
        "normal": _cache_fingerprint(normal_cache),
        "blank": _cache_fingerprint(blank_cache),
    }
    if before != after:
        raise AssertionError("an immutable ARC1 cache changed during analysis")
    report = {
        "status": "passed_with_reference_scorer_caveat",
        "scope": "ARC1 blank-ID, exact committed ARC1 test split",
        "classification": "validated ablation; not an exact reproduction of the published 0% scorer output",
        "cache_treated_as_immutable": True,
        "cache_fingerprints": after,
        "counts": {
            "all_puzzles": 400,
            "all_descriptors": 419,
            "test_puzzles": len(test_ids),
            "test_descriptors": len(blank_test),
        },
        "frozen_hyperparameters": frozen,
        "normal_test_b1": normal_test_b1,
        "blank_full_upstream_semantics": upstream_evaluator_scores(blank),
        "blank_test_methods": methods,
        "selection_gap": selection_summary,
        "defect_auroc": defect,
        "coverage_conditioned": coverage,
        "invalid_predictions": {"all": invalid_all, "committed_test": invalid_test},
        "inventory": inventory,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "analysis_report.json", report)
    _write_json(output_dir / "artifact_manifest.json", _manifest(output_dir))
    return report


def run_arc2_supporting_analysis(
    *, cache_dir: Path, main_results: Path, output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    _require_new_output(output_dir)
    before = _cache_fingerprint(cache_dir)
    _, frozen = _load_contract(Path(main_results))
    orbits = load_orbits(cache_dir)
    if len(orbits) != 172 or len({orbit.puzzle_id for orbit in orbits}) != 120:
        raise AssertionError("ARC2 cache inventory must be 120 puzzles/172 descriptors")
    upstream = upstream_evaluator_scores(orbits)
    consensus_b1 = puzzle_mean(
        [_correct(majority_vote(orbit), orbit, 1) for orbit in orbits], orbits
    )
    online_mismatches = [
        orbit.task_id
        for orbit in orbits
        if orbit.metadata.get("stream_b1_key")
        and (
            int(orbit.metadata["stream_b1_key"][0]),
            int(orbit.metadata["stream_b1_key"][1]),
            bytes.fromhex(orbit.metadata["stream_b1_key"][2]),
        )
        != upstream_evaluator_ranking(orbit)[0]
    ]
    if upstream["pass@1"] != consensus_b1 or online_mismatches:
        raise AssertionError("ARC2 upstream/online/cache B1 semantics disagree")

    prepared = _prepare(orbits)
    diagnostics, diagnostic_summary, d_count = orbit_diagnostics(orbits, prepared)
    defect = [_all_defect_row(diagnostics)]
    _write_csv(output_dir / "orbit_statistics.csv", diagnostics)
    _write_csv(output_dir / "orbit_statistics_summary.csv", diagnostic_summary)
    _write_csv(output_dir / "d_count_diagnostic.csv", d_count)
    _write_csv(output_dir / "defect_auroc.csv", defect)
    outcomes = evaluate_orbits(orbits, FROZEN_PARAMETERS, prepared=prepared)
    methods = summarize_methods(outcomes, orbits)
    _write_csv(output_dir / "frozen_method_results.csv", methods)
    bootstrap = bootstrap_rows(outcomes, orbits, resamples=10_000, seed=20260807)
    _write_csv(output_dir / "paired_bootstrap_cis.csv", bootstrap)
    coverage, ranks = coverage_and_rank_rows(outcomes, orbits)
    _write_csv(output_dir / "coverage_conditioned_results.csv", coverage)
    _write_csv(output_dir / "rank_mrr_results.csv", ranks)
    shapes = shape_diagnostics_rows(orbits, prepared, FROZEN_PARAMETERS)
    _write_csv(output_dir / "shape_screening_diagnostics.csv", shapes)
    selection = selection_gap_rows(
        orbits,
        budgets=(50, 250, 1000),
        repeats=10,
        seed=20260807,
    )
    selection_summary = _summarize_selection(selection)
    _write_csv(output_dir / "selection_gap_repeats.csv", selection)
    _write_csv(output_dir / "selection_gap_summary.csv", selection_summary)
    invalid = _invalid_statistics(orbits)
    _write_json(output_dir / "invalid_prediction_statistics.json", invalid)
    after = _cache_fingerprint(cache_dir)
    if before != after:
        raise AssertionError("immutable ARC2 cache changed during analysis")
    report = {
        "status": "passed_supporting_only",
        "classification": "unresolved/supporting; external 6.2% reference not reproduced",
        "cache_treated_as_immutable": True,
        "cache_fingerprint": after,
        "counts": {
            "puzzles": 120,
            "descriptors": 172,
            "candidates": sum(len(orbit.candidates) for orbit in orbits),
        },
        "frozen_hyperparameters": frozen,
        "reproduction": {
            "pinned_upstream_evaluator": upstream,
            "online_orbit_b1": consensus_b1,
            "cache_recomputed_b1": consensus_b1,
            "online_upstream_task_mismatches": online_mismatches,
            "external_reference": 0.062,
            "external_reference_status": "unresolved",
        },
        "methods": methods,
        "selection_gap": selection_summary,
        "defect_auroc": defect,
        "coverage_conditioned": coverage,
        "invalid_predictions": invalid,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "analysis_report.json", report)
    _write_json(output_dir / "artifact_manifest.json", _manifest(output_dir))
    return report
