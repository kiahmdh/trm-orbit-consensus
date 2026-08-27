from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from orbit_consensus.cache import save_task_orbit
from orbit_consensus.schema import Candidate, TaskOrbit

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_figures.py"
SPEC = importlib.util.spec_from_file_location("make_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIGURES
SPEC.loader.exec_module(FIGURES)

FigureInputError = FIGURES.FigureInputError
make_figures = FIGURES.make_figures

METHODS = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_cache(cache_dir: Path) -> None:
    candidates = (
        Candidate(np.asarray([[1]], dtype=np.uint8), 0, 0.9, is_identity=True),
        Candidate(np.asarray([[1]], dtype=np.uint8), 1, 0.8),
        Candidate(np.asarray([[2]], dtype=np.uint8), 2, 0.7),
    )
    orbit = TaskOrbit(
        "fixture#0",
        candidates,
        query_input=np.asarray([[0]], dtype=np.uint8),
        target=np.asarray([[1]], dtype=np.uint8),
    )
    for name in ("arc1", "arc1_blank", "arc2"):
        save_task_orbit(cache_dir / name / "fixture#0.npz", orbit)


def _make_results(results: Path) -> None:
    main = results / "arc1_main"
    blank = results / "arc1_blank_ablation"
    diagnostic = results / "m1_beta_diagnostic"
    budgets = (50, 250, 1000)
    compute_rows = []
    compute_ci_rows = []
    for method_index, method in enumerate(METHODS[1:]):
        for budget_index, budget in enumerate(budgets):
            rank1 = 0.35 + 0.02 * budget_index + 0.004 * method_index
            top2 = rank1 + 0.05
            compute_rows.append(
                {
                    "budget": budget,
                    "repeat": 0,
                    "method": method,
                    "rank1_accuracy": rank1,
                    "top2_accuracy": top2,
                }
            )
            for metric, value in (("rank1", rank1), ("top2", top2)):
                compute_ci_rows.append(
                    {
                        "budget": budget,
                        "method": method,
                        "metric": metric,
                        "accuracy_ci95_low": value - 0.025,
                        "accuracy_ci95_high": value + 0.025,
                    }
                )
    _write_csv(main / "compute_matched_results.csv", compute_rows)
    _write_csv(main / "compute_matched_bootstrap_cis.csv", compute_ci_rows)
    _write_csv(
        main / "selection_gap.csv",
        [
            {"budget": 50, "pass_at_k": 0.50, "majority_at_k": 0.36},
            {"budget": 250, "pass_at_k": 0.58, "majority_at_k": 0.39},
            {"budget": 1000, "pass_at_k": 0.62, "majority_at_k": 0.40},
        ],
    )
    _write_csv(
        main / "selection_gap_top2.csv",
        [
            {"budget": 50, "majority_top2_at_k": 0.42},
            {"budget": 250, "majority_top2_at_k": 0.45},
            {"budget": 1000, "majority_top2_at_k": 0.47},
        ],
    )
    _write_csv(
        main / "orbit_statistics.csv",
        [
            {"puzzle_id": "a", "orbit_dispersion": 0.05, "b1_correct": 1, "split": "test"},
            {"puzzle_id": "b", "orbit_dispersion": 0.10, "b1_correct": 1, "split": "test"},
            {"puzzle_id": "c", "orbit_dispersion": 0.55, "b1_correct": 0, "split": "test"},
            {"puzzle_id": "d", "orbit_dispersion": 0.80, "b1_correct": 0, "split": "test"},
        ],
    )
    _write_csv(
        main / "h1_auroc_results.csv",
        [{"split": "test", "auroc_defect_for_incorrectness": 1.0}],
    )
    definitions = ("distinct", "multiset", "non_identical_support")
    _write_csv(
        main / "d_count_diagnostic.csv",
        [
            {"definition": value, "puzzle_weighted_mean_spearman": 0.10 + 0.08 * index}
            for index, value in enumerate(definitions)
        ],
    )
    _write_csv(
        main / "d_count_bootstrap_cis.csv",
        [
            {
                "definition": value,
                "correlation_ci95_low": 0.04 + 0.08 * index,
                "correlation_ci95_high": 0.16 + 0.08 * index,
            }
            for index, value in enumerate(definitions)
        ],
    )
    _write_csv(
        diagnostic / "m1_beta_summary.csv",
        [
            {
                "definition": "distinct",
                "beta": beta,
                "dev_rank1_accuracy": rank1,
                "dev_top2_accuracy": top2,
                "b1_dev_rank1_accuracy": 0.40,
            }
            for beta, rank1, top2 in ((-1, 0.39, 0.44), (0, 0.40, 0.45), (1, 0.41, 0.45))
        ],
    )
    (main / "selected_m1.json").write_text(
        json.dumps(
            {
                "definition": "distinct",
                "beta": 1.0,
                "dev_rank1_accuracy": 0.41,
                "dev_top2_accuracy": 0.45,
            }
        ),
        encoding="ascii",
    )
    _write_csv(
        main / "m2_dev_sweep.csv",
        [
            {
                "marginal_support": "emitted",
                "interpolation": interpolation,
                "epsilon": epsilon,
                "dev_rank1_accuracy": 0.40 + 0.01 * (interpolation > 0),
            }
            for epsilon in (0.000001, 0.001)
            for interpolation in (0.0, 0.05)
        ],
    )
    (main / "selected_m2.json").write_text(
        json.dumps(
            {
                "marginal_support": "emitted",
                "interpolation": 0.05,
                "epsilon": 0.000001,
            }
        ),
        encoding="ascii",
    )
    _write_csv(
        main / "marginal_support_ablation.csv",
        [
            {"marginal_support": "emitted", "dev_rank1_accuracy": 0.41},
            {"marginal_support": "distinct_uniform", "dev_rank1_accuracy": 0.39},
        ],
    )
    _write_csv(
        main / "marginal_support_bootstrap_cis.csv",
        [
            {
                "difference": 0.02,
                "difference_ci95_low": -0.01,
                "difference_ci95_high": 0.05,
            }
        ],
    )
    _write_csv(
        main / "rank_of_correct.csv",
        [
            {
                "method": method,
                "puzzle_id": puzzle,
                "correct_rank": rank,
                "covered": 1,
                "split": "test",
            }
            for method, ranks in (("B1", (1, 2, 5)), ("M1+M2", (1, 1, 3)))
            for puzzle, rank in zip(("a", "b", "c"), ranks, strict=True)
        ],
    )
    _write_csv(
        main / "rank_mrr_results.csv",
        [
            {"method": method, "puzzle_weighted_mrr_all": 0.30 + 0.03 * index}
            for index, method in enumerate(METHODS)
        ],
    )
    main_rows = []
    blank_rows = []
    main_ci_rows = []
    blank_ci_rows = []
    covered_rows = []
    covered_ci_rows = []
    for index, method in enumerate(METHODS):
        rank1 = 0.29 + 0.03 * index
        top2 = rank1 + 0.04
        blank_rank1 = 0.01 + 0.01 * index
        main_rows.append({"method": method, "rank1_accuracy": rank1, "top2_accuracy": top2})
        blank_rows.append({"method": method, "rank1_accuracy": blank_rank1})
        for metric, value in (("rank1", rank1), ("top2", top2)):
            main_ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "accuracy_ci95_low": value - 0.03,
                    "accuracy_ci95_high": value + 0.03,
                }
            )
        blank_ci_rows.append(
            {
                "method": method,
                "metric": "rank1",
                "accuracy_ci95_low": max(0, blank_rank1 - 0.01),
                "accuracy_ci95_high": blank_rank1 + 0.02,
            }
        )
        covered_rank1 = min(0.95, rank1 + 0.20)
        covered_top2 = min(0.98, top2 + 0.20)
        covered_rows.append(
            {
                "method": method,
                "rank1_given_covered": covered_rank1,
                "top2_given_covered": covered_top2,
            }
        )
        for metric, value in (
            ("rank1_given_covered", covered_rank1),
            ("top2_given_covered", covered_top2),
        ):
            covered_ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "accuracy_ci95_low": value - 0.04,
                    "accuracy_ci95_high": min(1, value + 0.04),
                }
            )
    _write_csv(main / "main_test_results.csv", main_rows)
    _write_csv(main / "paired_bootstrap_cis.csv", main_ci_rows)
    _write_csv(blank / "frozen_method_results.csv", blank_rows)
    _write_csv(blank / "paired_bootstrap_cis.csv", blank_ci_rows)
    _write_csv(main / "coverage_conditioned_results.csv", covered_rows)
    _write_csv(main / "covered_subset_bootstrap_cis.csv", covered_ci_rows)
    _write_csv(
        main / "error_correlation_results.csv",
        [
            {
                "mean_incorrect_pair_similarity": 0.20 + 0.10 * index,
                "mean_correct_to_incorrect_similarity": ("" if index == 0 else 0.25 + 0.08 * index),
            }
            for index in range(6)
        ],
    )


def _make_fixture(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    cache = root / "cache"
    _make_results(results)
    _make_cache(cache)
    return results, cache


def test_missing_compute_bootstrap_blocks_before_writing(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cache = tmp_path / "cache"
    output = tmp_path / "figures"
    _write_csv(
        results / "arc1_main" / "compute_matched_results.csv",
        [
            {
                "budget": 50,
                "method": "B1",
                "rank1_accuracy": 0.4,
                "top2_accuracy": 0.45,
            }
        ],
    )
    with pytest.raises(FigureInputError, match="fig_compute_matched blocked"):
        make_figures(results, cache, output, repo_root=tmp_path)
    assert not output.exists()


def test_all_figures_are_deterministic_and_inputs_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, cache = _make_fixture(tmp_path)
    output = tmp_path / "figures"
    monkeypatch.setenv("ORBIT_CONSENSUS_GIT_COMMIT", "a" * 40)
    before = _hash_tree(tmp_path)
    generated = make_figures(results, cache, output, repo_root=tmp_path)
    first_hashes = _hash_tree(output)
    make_figures(results, cache, output, repo_root=tmp_path)
    second_hashes = _hash_tree(output)

    assert len(list(output.glob("fig_*.pdf"))) == 11
    assert len(list(output.glob("fig_*.png"))) == 11
    assert {path.name for path in generated} >= {
        "MANIFEST.json",
        "captions.tex",
        "main_results_table.tex",
    }
    assert first_hashes == second_hashes
    assert {
        key: value for key, value in _hash_tree(tmp_path).items() if not key.startswith("figures/")
    } == {key: value for key, value in before.items() if not key.startswith("figures/")}
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="ascii"))
    assert manifest["git_commit"] == "a" * 40
    assert set(manifest["figures"]) == {
        "compute_matched",
        "selection_gap",
        "orbit_dispersion",
        "h1_defect",
        "d_count_decoupling",
        "m1_beta_sweep",
        "m2_sweep",
        "marginal_support_ablation",
        "rank_of_correct",
        "puzzle_id_ablation",
        "error_correlation",
    }
