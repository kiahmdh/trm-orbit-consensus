#!/usr/bin/env python3
"""Render deterministic, publication-ready figures from frozen analysis artifacts."""

from __future__ import annotations

# ruff: noqa: ISC004
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20260807
METHOD_ORDER = ("B0", "B1", "M1", "M2", "M1+M2", "M1+M2+M3")
METHOD_COLORS = {
    "B0": "#6B6B6B",
    "B1": "#0072B2",
    "M1": "#009E73",
    "M2": "#E69F00",
    "M1+M2": "#D55E00",
    "M1+M2+M3": "#CC79A7",
}
FULL_WIDTH = 5.5
HALF_WIDTH = 2.65
PDF_METADATA = {
    "Creator": "orbit-consensus make_figures.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "orbit-consensus make_figures.py"}

CAPTIONS = {
    "compute_matched": (
        "Compute-matched held-out ARC-AGI-1 accuracy. Lines show rank-1 (solid) "
        "and top-2 (dashed) accuracy; bands are paired-puzzle bootstrap 95\\% "
        "confidence intervals.",
        "fig:compute-matched",
    ),
    "selection_gap": (
        "Candidate-generation coverage and majority-vote selection accuracy as "
        "the augmentation budget grows. Shading isolates the top-2 selection gap.",
        "fig:selection-gap",
    ),
    "orbit_dispersion": (
        "ARC-AGI-1 orbit dispersion. Left: cumulative vote mass in the top "
        "distinct modes. Right: number of distinct emitted grids per descriptor.",
        "fig:orbit-dispersion",
    ),
    "h1_defect": (
        "Equivariance defect as a diagnostic of whole-grid majority-vote failure "
        "on the held-out split.",
        "fig:h1-defect",
    ),
    "d_count_decoupling": (
        "Rank correlation between candidate centrality and emission count under "
        "three definitions of equivariance defect on ARC-AGI-1.",
        "fig:d-count-decoupling",
    ),
    "m1_beta_sweep": (
        "Development-set accuracy across signed centrality weights for the "
        "committed defect definition.",
        "fig:m1-beta-sweep",
    ),
    "m2_sweep": (
        "Development-set rank-1 accuracy for the cell-marginal interpolation and "
        "smoothing sweep. The star marks the committed setting.",
        "fig:m2-sweep",
    ),
    "marginal_support_ablation": (
        "Development-set marginal-support ablation comparing count-weighted "
        "emissions with distinct-grid-uniform support.",
        "fig:marginal-support-ablation",
    ),
    "rank_of_correct": (
        "Empirical cumulative distribution of the correct candidate's rank on "
        "covered held-out puzzles.",
        "fig:rank-of-correct",
    ),
    "puzzle_id_ablation": (
        "Held-out rank-1 accuracy with normal puzzle identifiers and blank "
        "identifiers. Error bars are paired-puzzle bootstrap 95\\% intervals.",
        "fig:puzzle-id-ablation",
    ),
    "error_correlation": (
        "Pairwise grid similarity among incorrect candidates and between the "
        "correct candidate and incorrect candidates on covered held-out tasks.",
        "fig:error-correlation",
    ),
}


class FigureInputError(RuntimeError):
    """Raised when an immutable input cannot support a requested figure."""


@dataclass
class InputRegistry:
    results_dir: Path
    cache_dir: Path
    inputs: dict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))
    warnings: list[str] = field(default_factory=list)

    def _record(self, blocked: str, path: Path) -> Path:
        resolved = path.resolve()
        self.inputs[blocked].add(resolved)
        return resolved

    def require_file(
        self,
        blocked: str,
        candidates: Sequence[str],
        *,
        description: str,
    ) -> Path:
        checked = [self.results_dir / value for value in candidates]
        for path in checked:
            if path.is_file():
                return self._record(blocked, path)
        names = ", ".join(str(path) for path in checked)
        raise FigureInputError(
            f"{blocked} blocked: missing required {description}; checked {names}"
        )

    def require_csv(
        self,
        blocked: str,
        candidates: Sequence[str],
        columns: Sequence[str],
        *,
        description: str,
    ) -> tuple[Path, list[dict[str, str]]]:
        existing: list[tuple[Path, set[str]]] = []
        for relative in candidates:
            path = self.results_dir / relative
            if not path.is_file():
                continue
            rows, names = _read_csv_raw(path)
            existing.append((path, names))
            if set(columns).issubset(names):
                self._record(blocked, path)
                return path.resolve(), rows
        if existing:
            details = "; ".join(
                f"{path} lacks {sorted(set(columns) - names)}" for path, names in existing
            )
            raise FigureInputError(f"{blocked} blocked: {description}: {details}")
        checked = ", ".join(str(self.results_dir / value) for value in candidates)
        raise FigureInputError(
            f"{blocked} blocked: missing required {description}; checked {checked}"
        )

    def require_cache(self, blocked: str, aliases: Sequence[str]) -> tuple[Path, list[Path]]:
        checked = [self.cache_dir / alias for alias in aliases]
        for path in checked:
            files = sorted(path.glob("*.npz")) if path.is_dir() else []
            if files:
                for item in files:
                    self._record(blocked, item)
                return path.resolve(), [item.resolve() for item in files]
        names = ", ".join(str(path) for path in checked)
        raise FigureInputError(f"{blocked} blocked: missing NPZ cache directory; checked {names}")


@dataclass(frozen=True)
class FigureContext:
    registry: InputRegistry
    tables: Mapping[str, list[dict[str, str]]]
    paths: Mapping[str, Path]
    cache_paths: Mapping[str, tuple[Path, ...]]
    git_commit: str


def _read_csv_raw(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FigureInputError(f"empty CSV header: {path}")
        return list(reader), set(reader.fieldnames)


def _float(row: Mapping[str, str], field: str, *, source: str) -> float:
    value = row.get(field, "")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"{source}: invalid numeric field {field}={value!r}") from exc
    if not math.isfinite(result):
        raise FigureInputError(f"{source}: non-finite numeric field {field}={value!r}")
    return result


def _first(
    rows: Iterable[Mapping[str, str]],
    *,
    source: str,
    **matches: object,
) -> Mapping[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in matches.items()):
            return row
    criteria = ", ".join(f"{key}={value!r}" for key, value in matches.items())
    raise FigureInputError(f"{source}: missing required row ({criteria})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    override = os.environ.get("ORBIT_CONSENSUS_GIT_COMMIT")
    if override:
        if len(override) != 40 or any(c not in "0123456789abcdefABCDEF" for c in override):
            raise FigureInputError(
                "MANIFEST.json blocked: ORBIT_CONSENSUS_GIT_COMMIT must be a 40-digit hex hash"
            )
        return override.lower()
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise FigureInputError(
            "MANIFEST.json blocked: repository has no git commit; commit the release "
            "or set ORBIT_CONSENSUS_GIT_COMMIT to the exact source commit"
        )
    return commit


def _configure_matplotlib() -> Any:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    random.seed(SEED)
    np.random.seed(SEED)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.3,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "axes.unicode_minus": False,
        }
    )
    return plt


def _style_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linewidth=0.3, color="#D6D6D6")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _export(plt: Any, fig: Any, output_dir: Path, name: str) -> tuple[Path, Path]:
    pdf = output_dir / f"fig_{name}.pdf"
    png = output_dir / f"fig_{name}.png"
    fig.savefig(pdf, format="pdf", metadata=PDF_METADATA)
    fig.savefig(png, format="png", dpi=300, metadata=PNG_METADATA)
    plt.close(fig)
    return pdf, png


def _mean_by(rows: Sequence[Mapping[str, str]], key: str, value: str) -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_float(row, key, source=key)].append(_float(row, value, source=value))
    return {item: float(np.mean(values)) for item, values in grouped.items()}


def _resolve_ci_row(
    rows: Sequence[Mapping[str, str]],
    *,
    source: str,
    method: str,
    metric: str,
    budget: int | None = None,
) -> Mapping[str, str]:
    candidates = [
        row for row in rows if row.get("method") == method and row.get("metric") == metric
    ]
    if budget is not None:
        candidates = [row for row in candidates if int(float(row.get("budget", "-1"))) == budget]
    if len(candidates) != 1:
        qualifier = f", budget={budget}" if budget is not None else ""
        raise FigureInputError(
            f"{source}: expected one CI row for method={method}, metric={metric}{qualifier}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _preflight(results_dir: Path, cache_dir: Path, repo_root: Path) -> FigureContext:
    registry = InputRegistry(results_dir.resolve(), cache_dir.resolve())
    tables: dict[str, list[dict[str, str]]] = {}
    paths: dict[str, Path] = {}

    def table(
        key: str,
        blocked: str,
        candidates: Sequence[str],
        columns: Sequence[str],
        description: str,
    ) -> None:
        path, rows = registry.require_csv(blocked, candidates, columns, description=description)
        paths[key] = path
        tables[key] = rows

    table(
        "compute",
        "fig_compute_matched",
        ("arc1_main/compute_matched_results.csv", "compute_matched_results.csv"),
        ("budget", "method", "rank1_accuracy", "top2_accuracy"),
        "compute-matched estimates",
    )
    table(
        "compute_ci",
        "fig_compute_matched",
        (
            "arc1_main/compute_matched_bootstrap_cis.csv",
            "compute_matched_bootstrap_cis.csv",
            "arc1_main/paired_bootstrap_cis.csv",
        ),
        (
            "budget",
            "method",
            "metric",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
        ),
        "per-budget paired-bootstrap confidence intervals",
    )
    table(
        "selection",
        "fig_selection_gap",
        ("arc1_main/selection_gap.csv", "selection_gap.csv"),
        ("budget", "pass_at_k", "majority_at_k"),
        "selection-gap estimates",
    )
    table(
        "selection_top2",
        "fig_selection_gap",
        (
            "arc1_main/selection_gap_top2.csv",
            "selection_gap_top2.csv",
            "arc1_main/selection_gap.csv",
        ),
        ("budget", "majority_top2_at_k"),
        "full-set majority top-2 selection curve",
    )

    cache_paths: dict[str, tuple[Path, ...]] = {}
    _, arc1_files = registry.require_cache("fig_orbit_dispersion", ("arc1", "arc1_normal"))
    cache_paths["arc1"] = tuple(arc1_files)
    # Validate the advertised cache bundle even though only ARC1 is read by these plots.
    _, blank_files = registry.require_cache("cache bundle", ("arc1_blank",))
    _, arc2_files = registry.require_cache("cache bundle", ("arc2", "arc2_normal"))
    cache_paths["arc1_blank"] = tuple(blank_files)
    cache_paths["arc2"] = tuple(arc2_files)

    table(
        "orbit_stats",
        "fig_h1_defect",
        ("arc1_main/orbit_statistics.csv", "orbit_statistics.csv"),
        ("puzzle_id", "orbit_dispersion", "b1_correct", "split"),
        "per-descriptor defect diagnostics",
    )
    table(
        "h1",
        "fig_h1_defect",
        ("arc1_main/h1_auroc_results.csv", "h1_auroc_results.csv"),
        ("split", "auroc_defect_for_incorrectness"),
        "defect AUROC summary",
    )
    table(
        "d_count",
        "fig_d_count_decoupling",
        ("arc1_main/d_count_diagnostic.csv", "d_count_diagnostic.csv"),
        ("definition", "puzzle_weighted_mean_spearman"),
        "d-count correlation estimates",
    )
    table(
        "d_count_ci",
        "fig_d_count_decoupling",
        (
            "arc1_main/d_count_bootstrap_cis.csv",
            "d_count_bootstrap_cis.csv",
            "arc1_main/paired_bootstrap_cis.csv",
        ),
        ("definition", "correlation_ci95_low", "correlation_ci95_high"),
        "paired-bootstrap d-count confidence intervals",
    )
    table(
        "m1",
        "fig_m1_beta_sweep",
        (
            "m1_beta_diagnostic/m1_beta_summary.csv",
            "arc1_main/m1_dev_sweep.csv",
            "m1_dev_sweep.csv",
        ),
        (
            "definition",
            "beta",
            "dev_rank1_accuracy",
            "dev_top2_accuracy",
            "b1_dev_rank1_accuracy",
        ),
        "M1 development sweep",
    )
    paths["selected_m1"] = registry.require_file(
        "fig_m1_beta_sweep",
        ("arc1_main/selected_m1.json", "selected_m1.json"),
        description="committed M1 setting",
    )
    table(
        "m2",
        "fig_m2_sweep",
        ("arc1_main/m2_dev_sweep.csv", "m2_dev_sweep.csv"),
        (
            "marginal_support",
            "interpolation",
            "epsilon",
            "dev_rank1_accuracy",
        ),
        "M2 development sweep",
    )
    paths["selected_m2"] = registry.require_file(
        "fig_m2_sweep",
        ("arc1_main/selected_m2.json", "selected_m2.json"),
        description="committed M2 setting",
    )
    table(
        "marginal",
        "fig_marginal_support_ablation",
        ("arc1_main/marginal_support_ablation.csv", "marginal_support_ablation.csv"),
        ("marginal_support", "dev_rank1_accuracy"),
        "marginal-support ablation",
    )
    table(
        "marginal_ci",
        "fig_marginal_support_ablation",
        (
            "arc1_main/marginal_support_bootstrap_cis.csv",
            "marginal_support_bootstrap_cis.csv",
            "arc1_main/paired_bootstrap_cis.csv",
        ),
        ("difference", "difference_ci95_low", "difference_ci95_high"),
        "paired-bootstrap marginal-support difference",
    )
    table(
        "ranks",
        "fig_rank_of_correct",
        (
            "arc1_main/rank_of_correct.csv",
            "rank_of_correct.csv",
            "arc1_main/per_puzzle_correct_ranks.csv",
        ),
        ("method", "puzzle_id", "correct_rank", "covered", "split"),
        "per-puzzle correct-candidate ranks",
    )
    table(
        "rank_summary",
        "fig_rank_of_correct",
        ("arc1_main/rank_mrr_results.csv", "rank_mrr_results.csv"),
        ("method", "puzzle_weighted_mrr_all"),
        "rank MRR summary",
    )
    table(
        "main",
        "fig_puzzle_id_ablation",
        ("arc1_main/main_test_results.csv", "main_test_results.csv"),
        ("method", "rank1_accuracy", "top2_accuracy"),
        "normal-ID held-out results",
    )
    table(
        "main_ci",
        "fig_puzzle_id_ablation",
        ("arc1_main/paired_bootstrap_cis.csv", "paired_bootstrap_cis.csv"),
        ("method", "metric", "accuracy_ci95_low", "accuracy_ci95_high"),
        "normal-ID paired-bootstrap intervals",
    )
    table(
        "blank",
        "fig_puzzle_id_ablation",
        (
            "arc1_blank_ablation/frozen_method_results.csv",
            "arc1_blank_ablation/main_test_results.csv",
            "blank_id_results.csv",
        ),
        ("method", "rank1_accuracy"),
        "blank-ID held-out results",
    )
    table(
        "blank_ci",
        "fig_puzzle_id_ablation",
        (
            "arc1_blank_ablation/paired_bootstrap_cis.csv",
            "blank_id_paired_bootstrap_cis.csv",
        ),
        ("method", "metric", "accuracy_ci95_low", "accuracy_ci95_high"),
        "blank-ID paired-bootstrap intervals",
    )
    table(
        "errors",
        "fig_error_correlation",
        ("arc1_main/error_correlation_results.csv", "error_correlation_results.csv"),
        (
            "mean_incorrect_pair_similarity",
            "mean_correct_to_incorrect_similarity",
        ),
        "error-correlation diagnostics",
    )
    table(
        "covered",
        "main_results_table.tex",
        (
            "arc1_main/coverage_conditioned_results.csv",
            "coverage_conditioned_results.csv",
        ),
        ("method", "rank1_given_covered", "top2_given_covered"),
        "covered-subset estimates",
    )
    table(
        "covered_ci",
        "main_results_table.tex",
        (
            "arc1_main/covered_subset_bootstrap_cis.csv",
            "covered_subset_bootstrap_cis.csv",
            "arc1_main/paired_bootstrap_cis.csv",
        ),
        (
            "method",
            "metric",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
        ),
        "covered-subset paired-bootstrap confidence intervals",
    )

    expected_methods = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")
    for source in ("main_ci", "blank_ci"):
        for method in expected_methods:
            _resolve_ci_row(
                tables[source],
                source=str(paths[source]),
                method=method,
                metric="rank1",
            )
    for method in expected_methods:
        for metric in ("rank1_given_covered", "top2_given_covered"):
            _resolve_ci_row(
                tables["covered_ci"],
                source=str(paths["covered_ci"]),
                method=method,
                metric=metric,
            )
    for method in ("B1", "M1", "M1+M2", "M1+M2+M3"):
        for metric in ("rank1", "top2"):
            for budget in (50, 250, 1000):
                _resolve_ci_row(
                    tables["compute_ci"],
                    source=str(paths["compute_ci"]),
                    method=method,
                    metric=metric,
                    budget=budget,
                )

    return FigureContext(
        registry=registry,
        tables=tables,
        paths=paths,
        cache_paths=cache_paths,
        git_commit=_git_commit(repo_root),
    )


def _figure_compute_matched(plt: Any, context: FigureContext) -> Any:
    rows = context.tables["compute"]
    ci_rows = context.tables["compute_ci"]
    methods = ("B1", "M1", "M1+M2", "M1+M2+M3")
    budgets = np.asarray([50, 250, 1000], dtype=float)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.2), constrained_layout=True)
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for metric, linestyle, alpha in (
            ("rank1", "-", 0.13),
            ("top2", "--", 0.07),
        ):
            field = f"{metric}_accuracy"
            means = _mean_by(selected, "budget", field)
            values = np.asarray([means[float(k)] for k in budgets])
            lows: list[float] = []
            highs: list[float] = []
            for budget in budgets.astype(int):
                row = _resolve_ci_row(
                    ci_rows,
                    source=str(context.paths["compute_ci"]),
                    method=method,
                    metric=metric,
                    budget=int(budget),
                )
                lows.append(_float(row, "accuracy_ci95_low", source="compute CI"))
                highs.append(_float(row, "accuracy_ci95_high", source="compute CI"))
            color = METHOD_COLORS[method]
            ax.plot(
                budgets,
                values,
                color=color,
                linestyle=linestyle,
                marker="o",
                label=method if metric == "rank1" else None,
            )
            ax.fill_between(budgets, lows, highs, color=color, alpha=alpha, linewidth=0)
    b1_at_1000 = _mean_by(
        [row for row in rows if row["method"] == "B1"],
        "budget",
        "rank1_accuracy",
    )[1000.0]
    ax.axhline(
        b1_at_1000,
        color=METHOD_COLORS["B1"],
        linestyle=":",
        linewidth=0.8,
        label="B1 at k=1000",
    )
    ax.set_xscale("log")
    ax.set_xticks(budgets, ["50", "250", "1000"])
    ax.set_xlabel("Augmentation budget, k")
    ax.set_ylabel("Held-out puzzle accuracy")
    ax.set_ylim(0.25, 0.56)
    _style_axis(ax)
    from matplotlib.lines import Line2D

    metric_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", label="Rank-1"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Top-2"),
    ]
    first = ax.legend(loc="lower right", ncol=2, frameon=False)
    ax.add_artist(first)
    ax.legend(handles=metric_handles, loc="upper left", frameon=False)
    return fig


def _figure_selection_gap(plt: Any, context: FigureContext) -> Any:
    pass_values = _mean_by(context.tables["selection"], "budget", "pass_at_k")
    rank1 = _mean_by(context.tables["selection"], "budget", "majority_at_k")
    top2 = _mean_by(context.tables["selection_top2"], "budget", "majority_top2_at_k")
    budgets = np.asarray(sorted(pass_values), dtype=float)
    coverage = np.asarray([pass_values[k] for k in budgets])
    majority1 = np.asarray([rank1[k] for k in budgets])
    majority2 = np.asarray([top2[k] for k in budgets])
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.05), constrained_layout=True)
    ax.plot(
        budgets,
        coverage,
        color="#000000",
        marker="o",
        label=r"Coverage (pass@$k$)",
    )
    ax.plot(
        budgets,
        majority1,
        color=METHOD_COLORS["B1"],
        marker="s",
        label="B1 rank-1",
    )
    ax.plot(
        budgets,
        majority2,
        color=METHOD_COLORS["M1+M2"],
        marker="^",
        label="B1 top-2",
    )
    ax.fill_between(
        budgets,
        majority2,
        coverage,
        where=coverage >= majority2,
        color=METHOD_COLORS["M1+M2"],
        alpha=0.12,
        linewidth=0,
        label="Unclosed selection gap",
    )
    last = -1
    gap = coverage[last] - majority2[last]
    ax.annotate(
        f"{100 * gap:.1f} pp",
        xy=(budgets[last], (coverage[last] + majority2[last]) / 2),
        xytext=(-34, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "#555555"},
    )
    ax.set_xscale("log")
    ax.set_xticks(budgets, [str(int(k)) for k in budgets])
    ax.set_xlabel("Augmentation budget, k")
    ax.set_ylabel("Puzzle fraction")
    ax.set_ylim(0.3, 0.68)
    _style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return fig


def _grid_key(grid: np.ndarray[Any, Any]) -> tuple[int, int, bytes]:
    array = np.asarray(grid, dtype=np.uint8)
    return int(array.shape[0]), int(array.shape[1]), array.tobytes(order="C")


def _figure_orbit_dispersion(plt: Any, context: FigureContext) -> Any:
    from orbit_consensus.cache import load_task_orbit

    cumulative: list[np.ndarray[Any, Any]] = []
    distinct_counts: list[int] = []
    modal_shares: list[float] = []
    max_modes = 20
    for path in context.cache_paths["arc1"]:
        orbit = load_task_orbit(path)
        counts = np.asarray(
            sorted(
                Counter(_grid_key(candidate.grid) for candidate in orbit.candidates).values(),
                reverse=True,
            ),
            dtype=float,
        )
        shares = counts / counts.sum()
        curve = np.cumsum(shares)
        padded = np.pad(curve, (0, max(0, max_modes - len(curve))), constant_values=1.0)
        cumulative.append(padded[:max_modes])
        distinct_counts.append(len(counts))
        modal_shares.append(float(shares[0]))
    mean_curve = np.mean(np.stack(cumulative), axis=0)
    median_modal = float(np.median(modal_shares))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 2.05),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.1, 1.0)},
    )
    x = np.arange(1, max_modes + 1)
    axes[0].plot(x, mean_curve, color=METHOD_COLORS["B1"], marker="o", markevery=2)
    axes[0].axhline(1.0, color="#888888", linewidth=0.5)
    axes[0].annotate(
        f"Median modal share: {median_modal:.3f}",
        xy=(1, mean_curve[0]),
        xytext=(4, 0.24),
        arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "#555555"},
    )
    axes[0].set_xlabel("Top distinct modes, m")
    axes[0].set_ylabel("Mean cumulative vote mass")
    axes[0].set_xlim(1, max_modes)
    axes[0].set_ylim(0, 1.04)
    _style_axis(axes[0])
    positive = np.asarray(distinct_counts)
    bins = np.geomspace(1, max(2, positive.max()), 18)
    axes[1].hist(
        positive,
        bins=bins,
        color=METHOD_COLORS["M1"],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Distinct emitted grids, $|\mathcal{U}(C)|$")
    axes[1].set_ylabel("Descriptors")
    _style_axis(axes[1])
    return fig


def _weighted_roc(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    order = np.argsort(-scores, kind="mergesort")
    scores, labels, weights = scores[order], labels[order], weights[order]
    positives = float(weights[labels].sum())
    negatives = float(weights[~labels].sum())
    if positives <= 0 or negatives <= 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0.0
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[end] == scores[index]:
            end += 1
        tp += float(weights[index:end][labels[index:end]].sum())
        fp += float(weights[index:end][~labels[index:end]].sum())
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        index = end
    return np.asarray(fpr), np.asarray(tpr)


def _figure_h1_defect(plt: Any, context: FigureContext) -> Any:
    rows = [row for row in context.tables["orbit_stats"] if row["split"] == "test"]
    puzzle_counts = Counter(row["puzzle_id"] for row in rows)
    scores = np.asarray([_float(row, "orbit_dispersion", source="orbit stats") for row in rows])
    incorrect = np.asarray(
        [_float(row, "b1_correct", source="orbit stats") < 0.5 for row in rows],
        dtype=bool,
    )
    weights = np.asarray([1.0 / puzzle_counts[row["puzzle_id"]] for row in rows])
    fpr, tpr = _weighted_roc(scores, incorrect, weights)
    summary = _first(context.tables["h1"], source=str(context.paths["h1"]), split="test")
    auroc = _float(summary, "auroc_defect_for_incorrectness", source="h1 AUROC")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 2.05),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.15)},
    )
    axes[0].plot(fpr, tpr, color=METHOD_COLORS["M1+M2"], label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, loc="lower right")
    _style_axis(axes[0])
    groups = [scores[~incorrect], scores[incorrect]]
    group_weights = [weights[~incorrect], weights[incorrect]]
    max_score = max(float(scores.max()), 1e-6)
    bins = np.linspace(0, max_score, 20)
    axes[1].hist(
        groups,
        bins=bins,
        weights=group_weights,
        density=True,
        histtype="step",
        linewidth=1.2,
        color=[METHOD_COLORS["M1"], METHOD_COLORS["M1+M2"]],
        label=["B1 correct", "B1 incorrect"],
    )
    axes[1].set_xlabel("Equivariance defect")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    return fig


def _figure_d_count(plt: Any, context: FigureContext) -> Any:
    order = ("distinct", "multiset", "non_identical_support")
    labels = ("Distinct", "Multiset", "Non-identical\nsupport")
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for definition in order:
        row = _first(
            context.tables["d_count"],
            source=str(context.paths["d_count"]),
            definition=definition,
        )
        ci = _first(
            context.tables["d_count_ci"],
            source=str(context.paths["d_count_ci"]),
            definition=definition,
        )
        estimates.append(_float(row, "puzzle_weighted_mean_spearman", source="d-count"))
        lows.append(_float(ci, "correlation_ci95_low", source="d-count CI"))
        highs.append(_float(ci, "correlation_ci95_high", source="d-count CI"))
    values = np.asarray(estimates)
    errors = np.vstack((values - np.asarray(lows), np.asarray(highs) - values))
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.9), constrained_layout=True)
    ax.errorbar(
        np.arange(3),
        values,
        yerr=errors,
        fmt="o",
        color=METHOD_COLORS["M1"],
        capsize=2,
        linewidth=0.8,
    )
    ax.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel(r"Spearman $\rho(d,\mathrm{count})$")
    ax.set_ylim(min(-0.05, min(lows) - 0.03), max(highs) + 0.04)
    _style_axis(ax)
    return fig


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _figure_m1_sweep(plt: Any, context: FigureContext) -> Any:
    selected = _load_json(context.paths["selected_m1"])
    definition = str(selected["definition"])
    rows = [row for row in context.tables["m1"] if row["definition"] == definition]
    rows.sort(key=lambda row: _float(row, "beta", source="M1 sweep"))
    beta = np.asarray([_float(row, "beta", source="M1 sweep") for row in rows])
    rank1 = np.asarray([_float(row, "dev_rank1_accuracy", source="M1 sweep") for row in rows])
    top2 = np.asarray([_float(row, "dev_top2_accuracy", source="M1 sweep") for row in rows])
    zero = next(i for i, value in enumerate(beta) if value == 0)
    b1_dev = _float(rows[zero], "b1_dev_rank1_accuracy", source="M1 diagnostic")
    if not math.isclose(rank1[zero], b1_dev):
        raise FigureInputError("fig_m1_beta_sweep blocked: beta=0 does not reproduce B1")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 1.95), constrained_layout=True)
    ax.plot(beta, rank1, marker="o", color=METHOD_COLORS["M1"], label="Rank-1")
    ax.plot(beta, top2, marker="s", linestyle="--", color=METHOD_COLORS["M1+M2"], label="Top-2")
    ax.scatter(
        [float(selected["beta"])],
        [float(selected["dev_rank1_accuracy"])],
        marker="*",
        s=58,
        color="#000000",
        zorder=4,
        label="Committed",
    )
    ax.axvline(0, color=METHOD_COLORS["B1"], linestyle=":", linewidth=0.8)
    ax.annotate(
        "B1",
        xy=(0, rank1[zero]),
        xytext=(4, -11),
        textcoords="offset points",
        color=METHOD_COLORS["B1"],
    )
    ax.set_xlabel(r"Signed centrality weight, $\beta$")
    ax.set_ylabel("Development accuracy")
    ax.set_xticks(beta)
    ax.legend(frameon=False, ncol=3)
    _style_axis(ax)
    return fig


def _figure_m2_sweep(plt: Any, context: FigureContext) -> Any:
    selected = _load_json(context.paths["selected_m2"])
    support = str(selected["marginal_support"])
    rows = [row for row in context.tables["m2"] if row["marginal_support"] == support]
    lambdas = sorted({_float(row, "interpolation", source="M2 sweep") for row in rows})
    epsilons = sorted({_float(row, "epsilon", source="M2 sweep") for row in rows})
    matrix = np.full((len(epsilons), len(lambdas)), np.nan)
    for row in rows:
        y = epsilons.index(_float(row, "epsilon", source="M2 sweep"))
        x = lambdas.index(_float(row, "interpolation", source="M2 sweep"))
        matrix[y, x] = _float(row, "dev_rank1_accuracy", source="M2 sweep")
    if np.isnan(matrix).any():
        raise FigureInputError("fig_m2_sweep blocked: M2 lambda-epsilon grid is incomplete")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.15), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=matrix.min(), vmax=matrix.max())
    for y in range(len(epsilons)):
        for x in range(len(lambdas)):
            color = "white" if matrix[y, x] < np.median(matrix) else "black"
            ax.text(
                x, y, f"{100 * matrix[y, x]:.1f}", ha="center", va="center", color=color, fontsize=6
            )
    zero_x = lambdas.index(0.0)
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (zero_x - 0.5, -0.5),
            1,
            len(epsilons),
            fill=False,
            edgecolor="#FFFFFF",
            linewidth=1.1,
            linestyle="--",
        )
    )
    committed_x = lambdas.index(float(selected["interpolation"]))
    committed_y = epsilons.index(float(selected["epsilon"]))
    ax.scatter(committed_x, committed_y, marker="*", s=70, facecolor="white", edgecolor="black")
    ax.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    ax.set_yticks(
        range(len(epsilons)),
        [f"{value:.0e}".replace("e-0", "e-") for value in epsilons],
    )
    ax.set_xlabel(r"Cell-marginal interpolation, $\lambda$")
    ax.set_ylabel(r"Smoothing, $\epsilon$")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, aspect=16)
    colorbar.set_label("Development rank-1 accuracy")
    return fig


def _figure_marginal_ablation(plt: Any, context: FigureContext) -> Any:
    rows = context.tables["marginal"]
    emitted = _first(rows, source=str(context.paths["marginal"]), marginal_support="emitted")
    uniform = _first(
        rows,
        source=str(context.paths["marginal"]),
        marginal_support="distinct_uniform",
    )
    values = [
        _float(emitted, "dev_rank1_accuracy", source="marginal ablation"),
        _float(uniform, "dev_rank1_accuracy", source="marginal ablation"),
    ]
    ci_rows = context.tables["marginal_ci"]
    if len(ci_rows) != 1:
        raise FigureInputError(
            f"{context.paths['marginal_ci']}: expected exactly one marginal-support CI row"
        )
    ci = ci_rows[0]
    difference = _float(ci, "difference", source="marginal CI")
    low = _float(ci, "difference_ci95_low", source="marginal CI")
    high = _float(ci, "difference_ci95_high", source="marginal CI")
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.85), constrained_layout=True)
    ax.bar(
        [0, 1],
        values,
        color=[METHOD_COLORS["M2"], METHOD_COLORS["M1"]],
        width=0.62,
    )
    ax.set_xticks([0, 1], ["Count-\nweighted", "Distinct-\nuniform"])
    ax.set_ylabel("Development rank-1 accuracy")
    lower = max(0, min(values) - 0.025)
    ax.set_ylim(lower, max(values) + 0.045)
    ax.text(
        0.5,
        max(values) + 0.019,
        f"$\\Delta$={100 * difference:+.1f} pp\n95\\% CI [{100 * low:+.1f}, {100 * high:+.1f}]",
        ha="center",
        va="center",
        fontsize=7,
    )
    _style_axis(ax)
    return fig


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def _figure_rank_of_correct(plt: Any, context: FigureContext) -> Any:
    aliases = {"B1": "B1", "M2": "M1+M2", "M1+M2": "M1+M2"}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.0), constrained_layout=True)
    plotted = 0
    for source_method in ("B1", "M1+M2"):
        rows = [
            row
            for row in context.tables["ranks"]
            if aliases.get(row["method"], row["method"]) == source_method
            and row["split"] == "test"
            and _float(row, "covered", source="correct ranks") > 0.5
            and row["correct_rank"] != ""
        ]
        if not rows:
            raise FigureInputError(
                f"fig_rank_of_correct blocked: no covered test ranks for {source_method} "
                f"in {context.paths['ranks']}"
            )
        x, y = _ecdf([_float(row, "correct_rank", source="correct ranks") for row in rows])
        summary = _first(
            context.tables["rank_summary"],
            source=str(context.paths["rank_summary"]),
            method=source_method,
        )
        mrr = _float(summary, "puzzle_weighted_mrr_all", source="rank summary")
        ax.step(
            x,
            y,
            where="post",
            color=METHOD_COLORS[source_method],
            label=f"{source_method} (MRR {mrr:.3f})",
        )
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No covered tasks", transform=ax.transAxes, ha="center", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Correct candidate rank")
    ax.set_ylabel("Covered-task ECDF")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    return fig


def _figure_puzzle_id(plt: Any, context: FigureContext) -> Any:
    methods = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")
    normal = context.tables["main"]
    blank = context.tables["blank"]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.1), constrained_layout=True)
    for offset, label, rows, ci_rows, color in (
        (-width / 2, "Normal ID", normal, context.tables["main_ci"], METHOD_COLORS["B1"]),
        (width / 2, "Blank ID", blank, context.tables["blank_ci"], METHOD_COLORS["M1+M2"]),
    ):
        values: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for method in methods:
            row = _first(rows, source=label, method=method)
            ci = _resolve_ci_row(
                ci_rows,
                source=label,
                method=method,
                metric="rank1",
            )
            values.append(_float(row, "rank1_accuracy", source=label))
            lows.append(_float(ci, "accuracy_ci95_low", source=label))
            highs.append(_float(ci, "accuracy_ci95_high", source=label))
        values_array = np.asarray(values)
        errors = np.vstack((values_array - np.asarray(lows), np.asarray(highs) - values_array))
        ax.bar(
            x + offset,
            values_array,
            width,
            yerr=errors,
            capsize=1.8,
            color=color,
            alpha=0.82,
            label=label,
            error_kw={"linewidth": 0.6},
        )
    ax.set_xticks(x, methods)
    ax.set_ylabel("Held-out rank-1 accuracy")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, ncol=2)
    _style_axis(ax)
    return fig


def _figure_error_correlation(plt: Any, context: FigureContext) -> Any:
    fields = (
        "mean_incorrect_pair_similarity",
        "mean_correct_to_incorrect_similarity",
    )
    values: list[np.ndarray[Any, Any]] = []
    for column in fields:
        finite = []
        for row in context.tables["errors"]:
            raw = row.get(column, "")
            if raw == "":
                continue
            value = _float(row, column, source="error correlation")
            finite.append(value)
        values.append(np.asarray(finite))
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.9), constrained_layout=True)
    positions = [1, 2]
    nonempty_positions = [p for p, value in zip(positions, values) if len(value)]
    nonempty_values = [value for value in values if len(value)]
    if nonempty_values:
        violin = ax.violinplot(
            nonempty_values,
            positions=nonempty_positions,
            showmeans=False,
            showmedians=True,
            widths=0.7,
        )
        for index, body in enumerate(violin["bodies"]):
            body.set_facecolor(
                [METHOD_COLORS["M1+M2"], METHOD_COLORS["M1"]][nonempty_positions[index] - 1]
            )
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        violin["cmedians"].set_color("#222222")
        violin["cmedians"].set_linewidth(0.8)
    for position, value in zip(positions, values):
        if len(value) == 0:
            ax.text(position, 0.5, "No finite\nobservations", ha="center", va="center", fontsize=7)
    ax.set_xticks(
        positions,
        ["Incorrect-\nincorrect", "Correct-\nincorrect"],
    )
    ax.set_ylabel("Pairwise grid similarity")
    ax.set_ylim(0, 1.02)
    _style_axis(ax)
    return fig


def _format_ci(estimate: float, low: float, high: float) -> str:
    return f"{100 * estimate:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _write_main_table(context: FigureContext, output_dir: Path) -> Path:
    methods = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")
    labels = {
        "B0": "B0",
        "B1": "B1",
        "M1": "M1",
        "M1+M2": "M1+M2",
        "M1+M2+M3": "M1+M2+M3",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Rank-1 & Top-2 & Covered rank-1 \\\\",
        "\\midrule",
    ]
    for method in methods:
        main = _first(context.tables["main"], source="main results", method=method)
        covered = _first(context.tables["covered"], source="covered results", method=method)
        values: list[str] = []
        for metric, column in (("rank1", "rank1_accuracy"), ("top2", "top2_accuracy")):
            ci = _resolve_ci_row(
                context.tables["main_ci"],
                source=str(context.paths["main_ci"]),
                method=method,
                metric=metric,
            )
            values.append(
                _format_ci(
                    _float(main, column, source="main table"),
                    _float(ci, "accuracy_ci95_low", source="main table"),
                    _float(ci, "accuracy_ci95_high", source="main table"),
                )
            )
        ci = _resolve_ci_row(
            context.tables["covered_ci"],
            source=str(context.paths["covered_ci"]),
            method=method,
            metric="rank1_given_covered",
        )
        values.append(
            _format_ci(
                _float(covered, "rank1_given_covered", source="covered table"),
                _float(ci, "accuracy_ci95_low", source="covered table"),
                _float(ci, "accuracy_ci95_high", source="covered table"),
            )
        )
        method_tex = labels[method].replace("+", "{+}")
        lines.append(f"{method_tex} & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Held-out ARC-AGI-1 accuracy in percent with paired-puzzle "
            "bootstrap 95\\% confidence intervals.}",
            "\\label{tab:main-results}",
            "\\end{table}",
            "",
        ]
    )
    path = output_dir / "main_results_table.tex"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def _write_captions(output_dir: Path) -> Path:
    lines = []
    for name, (caption, label) in CAPTIONS.items():
        lines.extend(
            [
                f"% fig_{name}.pdf",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "",
            ]
        )
    path = output_dir / "captions.tex"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def _write_manifest(
    context: FigureContext,
    outputs: Mapping[str, tuple[Path, Path]],
    output_dir: Path,
    table_path: Path,
    captions_path: Path,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "git_commit": context.git_commit,
        "results_dir": str(context.registry.results_dir),
        "cache_dir": str(context.registry.cache_dir),
        "figures": {},
        "auxiliary_outputs": {
            "captions": captions_path.name,
            "main_results_table": table_path.name,
        },
        "warnings": context.registry.warnings,
    }
    for name in CAPTIONS:
        files = sorted(context.registry.inputs[f"fig_{name}"])
        payload["figures"][name] = {
            "pdf": outputs[name][0].name,
            "png": outputs[name][1].name,
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
        }
    table_inputs = set()
    for key in ("main", "main_ci", "covered", "covered_ci"):
        table_inputs.add(context.paths[key])
    payload["auxiliary_outputs"]["main_results_table_inputs"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(table_inputs)
    ]
    manifest = output_dir / "MANIFEST.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return manifest


def make_figures(
    results_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    repo_root: Path,
) -> tuple[Path, ...]:
    started = time.perf_counter()
    context = _preflight(results_dir, cache_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib()
    builders = {
        "compute_matched": _figure_compute_matched,
        "selection_gap": _figure_selection_gap,
        "orbit_dispersion": _figure_orbit_dispersion,
        "h1_defect": _figure_h1_defect,
        "d_count_decoupling": _figure_d_count,
        "m1_beta_sweep": _figure_m1_sweep,
        "m2_sweep": _figure_m2_sweep,
        "marginal_support_ablation": _figure_marginal_ablation,
        "rank_of_correct": _figure_rank_of_correct,
        "puzzle_id_ablation": _figure_puzzle_id,
        "error_correlation": _figure_error_correlation,
    }
    outputs: dict[str, tuple[Path, Path]] = {}
    for name, builder in builders.items():
        outputs[name] = _export(plt, builder(plt, context), output_dir, name)
    captions = _write_captions(output_dir)
    table = _write_main_table(context, output_dir)
    manifest = _write_manifest(context, outputs, output_dir, table, captions)
    generated = tuple(path for name in builders for path in outputs[name]) + (
        captions,
        table,
        manifest,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Generated {len(builders)} figures ({2 * len(builders)} image files) "
        f"in {elapsed:.2f}s at {output_dir.resolve()}"
    )
    if context.registry.warnings:
        print("Warnings:")
        for warning in context.registry.warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")
    return generated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render deterministic paper figures from frozen CPU analysis outputs"
    )
    parser.add_argument("RESULTS_DIR", type=Path)
    parser.add_argument("CACHE_DIR", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="output directory (default: figures/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        make_figures(
            args.RESULTS_DIR,
            args.CACHE_DIR,
            args.output_dir,
            repo_root=repo_root,
        )
    except FigureInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
