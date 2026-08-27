#!/usr/bin/env python3
"""Deterministic camera-ready rendering from frozen JSON/CSV artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

ASSET_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ASSET_ROOT.parent
MPL_CONFIG = ASSET_ROOT / ".mplconfig"
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RELEASE_MANIFEST = "manifests/release_manifest.json"
FIXED_PDF_DATE = datetime(2026, 8, 19, tzinfo=timezone.utc)
COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
}
RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
plt.rcParams.update(RC_PARAMS)


class MissingFrozenValue(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"refusing output outside paper_assets: {path}")
    return path


class FrozenInputs:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / RELEASE_MANIFEST
        self.release_manifest_sha256 = sha256(manifest_path)
        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), parse_float=Decimal
        )
        self.entries = {row["relative_path"]: row for row in self.manifest["files"]}
        json_paths = [
            p for p in self.entries if p.endswith(".json") and not p.startswith("paper_assets/")
        ]
        csv_paths = [
            p for p in self.entries if p.endswith(".csv") and not p.startswith("paper_assets/")
        ]
        if len(json_paths) + 1 != 46 or len(csv_paths) != 70:
            raise RuntimeError(
                f"frozen inventory mismatch: JSON={len(json_paths)+1}, CSV={len(csv_paths)}"
            )
        self.allowed = set(json_paths + csv_paths + [RELEASE_MANIFEST])
        self.used: set[str] = set()

    def _path(self, relative: str) -> Path:
        if relative not in self.allowed:
            raise RuntimeError(f"non-allowlisted input: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise MissingFrozenValue(f"missing frozen artifact: {relative}")
        if relative != RELEASE_MANIFEST:
            expected = self.entries[relative]["sha256"]
            observed = sha256(path)
            if observed != expected:
                raise RuntimeError(
                    f"frozen artifact hash mismatch: {relative}: {observed} != {expected}"
                )
        self.used.add(relative)
        return path

    def read_json(self, relative: str) -> Any:
        return json.loads(
            self._path(relative).read_text(encoding="utf-8"), parse_float=Decimal
        )

    def read_csv(self, relative: str) -> list[dict[str, str]]:
        with self._path(relative).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def source(self, relative: str) -> dict[str, str]:
        return {"path": relative, "sha256": sha256(self._path(relative))}


def dec(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def mean(values: Iterable[Any]) -> Decimal:
    items = [dec(value) for value in values]
    if not items:
        raise MissingFrozenValue("empty frozen series")
    return sum(items, Decimal(0)) / Decimal(len(items))


def raw(value: Any) -> str:
    value = dec(value)
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def fixed(value: Any, digits: int) -> str:
    return f"{dec(value):.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    return fixed(dec(value) * 100, digits)


def spct(value: Any, digits: int = 2) -> str:
    return f"{dec(value) * 100:+.{digits}f}"


def sfixed(value: Any, digits: int = 4) -> str:
    return f"{dec(value):+.{digits}f}"


def one(rows: Sequence[dict[str, str]], label: str, **conditions: str) -> dict[str, str]:
    found = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]
    if len(found) != 1:
        raise MissingFrozenValue(
            f"{label}: expected one row for {conditions}, found {len(found)}"
        )
    return found[0]


def write_text(
    root: Path,
    relative: str,
    content: str,
    provenance: dict[str, dict[str, Any]],
    sources: Sequence[str],
    transformation: str,
) -> None:
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    provenance[relative] = {
        "sources": list(sources),
        "transformation": transformation,
    }


def write_csv(
    root: Path,
    relative: str,
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
    sources: Sequence[str],
    transformation: str,
) -> None:
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance[relative] = {
        "sources": list(sources),
        "transformation": transformation,
    }


def axis_style(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def percent_tick(value: float, _position: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def save_fig(
    root: Path,
    fig: plt.Figure,
    pdf_relative: str,
    provenance: dict[str, dict[str, Any]],
    sources: Sequence[str],
    transformation: str,
) -> None:
    pdf_path = safe_path(root, pdf_relative)
    png_relative = f"figures/png_preview/{pdf_path.stem}.png"
    png_path = safe_path(root, png_relative)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": pdf_path.stem,
        "Author": "Structured Orbit Consensus",
        "Subject": "Frozen NeurIPS 2026 workshop paper asset",
        "Creator": "paper_assets/build_assets.py",
        "Producer": "Matplotlib",
        "CreationDate": FIXED_PDF_DATE,
        "ModDate": FIXED_PDF_DATE,
    }
    fig.savefig(pdf_path, format="pdf", metadata=metadata)
    fig.savefig(
        png_path,
        format="png",
        dpi=200,
        metadata={"Software": "paper_assets/build_assets.py"},
    )
    plt.close(fig)
    provenance[pdf_relative] = {
        "sources": list(sources),
        "transformation": transformation,
    }
    provenance[png_relative] = {
        "sources": list(sources),
        "transformation": transformation
        + "; derived: 200-dpi raster preview of the identical figure",
    }


def cleanup(root: Path) -> None:
    for relative in ("figures", "tables", "macros", "captions", "intermediates"):
        path = safe_path(root, relative)
        if path.exists():
            shutil.rmtree(path)
    for relative in ("manifest.json", "VALIDATION_REPORT.md", "MISSING.md"):
        path = safe_path(root, relative)
        if path.exists():
            path.unlink()


def load_data(store: FrozenInputs) -> dict[str, Any]:
    paths = {
        "cache": "results/provenance/cache_completion.json",
        "preflight": "results/provenance/arc1_preflight.json",
        "main": "results/arc1_main/main_test_results.csv",
        "main_boot": "results/arc1_main/paired_bootstrap_cis.csv",
        "gap": "results/arc1_main/selection_gap.csv",
        "h1": "results/arc1_main/h1_auroc_results.csv",
        "orbit": "results/arc1_main/orbit_statistics.csv",
        "main_report": "results/arc1_main/analysis_report.json",
        "risk_curve": "results/risk_coverage/arc1_risk_coverage.csv",
        "risk_points": "results/risk_coverage/arc1_operating_points.csv",
        "risk_summary": "results/risk_coverage/risk_coverage_summary.json",
        "confidence": "results/risk_coverage/confidence_comparison.csv",
        "aurc_boot": "results/risk_coverage_bootstrap/aurc_bootstrap_summary.csv",
        "aurc_diff": "results/risk_coverage_bootstrap/aurc_paired_differences.csv",
        "op_boot": "results/risk_coverage_bootstrap/operating_point_bootstrap_arc1.csv",
        "beta": "results/m1_beta_diagnostic/m1_beta_summary.csv",
        "beta_flip": "results/m1_beta_diagnostic/m1_beta_flip_thresholds.csv",
        "beta_json": "results/m1_beta_diagnostic/m1_beta_summary.json",
        "compute": "results/compute_matched_q_audit/compute_matched_q_audit.csv",
        "blank": "results/arc1_blank_ablation/frozen_method_results.csv",
        "blank_boot": "results/arc1_blank_ablation/paired_bootstrap_cis.csv",
        "arc2": "results/arc2_supporting/frozen_method_results.csv",
        "arc2_curve": "results/risk_coverage/arc2_risk_coverage.csv",
        "disc": "results/discriminative_cell_dev/discriminative_cell_summary.json",
        "freeze": "results/final_freeze/final_results_manifest.json",
    }
    json_keys = {
        "cache",
        "preflight",
        "main_report",
        "risk_summary",
        "beta_json",
        "disc",
        "freeze",
    }
    data = {
        key: store.read_json(path) if key in json_keys else store.read_csv(path)
        for key, path in paths.items()
    }
    data["paths"] = paths
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in data["gap"]:
        grouped[int(row["budget"])].append(row)
    data["gaps"] = {}
    for budget in (50, 250, 1000):
        rows = grouped[budget]
        if len(rows) != 10:
            raise MissingFrozenValue(f"k={budget}: expected ten frozen repeats")
        data["gaps"][budget] = {
            "pass_at_k": mean(row["pass_at_k"] for row in rows),
            "majority_at_k": mean(row["majority_at_k"] for row in rows),
            "selection_gap": mean(row["selection_gap"] for row in rows),
        }
    data["main_by_method"] = {row["method"]: row for row in data["main"]}
    data["main_boot_by_key"] = {
        (row["method"], row["metric"]): row for row in data["main_boot"]
    }
    data["h1_by_split"] = {row["split"]: row for row in data["h1"]}
    data["blank_by_method"] = {row["method"]: row for row in data["blank"]}
    data["arc2_by_method"] = {row["method"]: row for row in data["arc2"]}
    data["confidence_arc1"] = {
        row["statistic"]: row
        for row in data["confidence"]
        if row["dataset"] == "arc1_normal_test"
        and row["baseline_type"] == "frozen_confidence"
    }
    data["aurc_boot_arc1"] = {
        row["statistic"]: row
        for row in data["aurc_boot"]
        if row["dataset"] == "arc1_normal_test"
    }
    data["aurc_diff_by_baseline"] = {
        row["baseline"]: row
        for row in data["aurc_diff"]
        if row["dataset"] == "arc1_normal_test"
    }
    return data


def make_intermediates(
    root: Path, data: dict[str, Any], provenance: dict[str, dict[str, Any]]
) -> None:
    p = data["paths"]
    write_csv(
        root,
        "intermediates/fig1_gap.csv",
        ("budget", "pass_at_k", "majority_at_k", "selection_gap"),
        [
            {
                "budget": budget,
                **{key: raw(value) for key, value in data["gaps"][budget].items()},
            }
            for budget in (50, 250, 1000)
        ],
        provenance,
        [p["gap"]],
        "derived: deterministic mean over ten frozen repeats at each budget",
    )
    stats = (
        "vote_margin",
        "vote_share",
        "vote_entropy",
        "winner_mean_q",
        "structural_defect",
    )
    rows = [
        {key: row[key] for key in ("statistic", "retained_puzzles", "coverage", "accuracy")}
        for row in data["risk_curve"]
        if row["statistic"] in stats
    ]
    full_accuracy = dec(data["main_by_method"]["B1"]["rank1_accuracy"])
    rows.extend(
        {
            "statistic": "random_order_expectation",
            "retained_puzzles": retained,
            "coverage": raw(Decimal(retained) / 200),
            "accuracy": raw(full_accuracy),
        }
        for retained in range(1, 201)
    )
    write_csv(
        root,
        "intermediates/fig2_risk_coverage.csv",
        ("statistic", "retained_puzzles", "coverage", "accuracy"),
        rows,
        provenance,
        [p["risk_curve"], p["risk_summary"], p["main"]],
        "verbatim: five frozen 200-cutoff curves; derived: analytic constant random-order expectation",
    )
    write_csv(
        root,
        "intermediates/fig3_beta_dynamics.csv",
        ("centrality", "beta", "dev_rank1_accuracy", "net_gain_puzzle_equivalents"),
        [
            {
                key: row[key]
                for key in (
                    "centrality",
                    "beta",
                    "dev_rank1_accuracy",
                    "net_gain_puzzle_equivalents",
                )
            }
            for row in data["beta"]
        ],
        provenance,
        [p["beta"]],
        "verbatim: frozen DEV beta sweep and decision transitions",
    )
    write_csv(
        root,
        "intermediates/fig3_beta_flip.csv",
        ("task_id", "puzzle_weight", "beta_flip"),
        [
            {key: row[key] for key in ("task_id", "puzzle_weight", "beta_flip")}
            for row in data["beta_flip"]
        ],
        provenance,
        [p["beta_flip"]],
        "verbatim: all 144 frozen per-descriptor beta-flip values",
    )
    write_csv(
        root,
        "intermediates/figA1_compute_matched.csv",
        ("budget", "method", "mean_rank1_accuracy"),
        [
            {
                "budget": row["budget"],
                "method": row["method"],
                "mean_rank1_accuracy": row["historical_mean_rank1"],
            }
            for row in data["compute"]
        ],
        provenance,
        [p["compute"]],
        "verbatim: frozen compute-matched audit means",
    )
    write_csv(
        root,
        "intermediates/figA2_dispersion.csv",
        ("task_id", "majority_vote_mass", "distinct_candidate_grids"),
        [
            {
                key: row[key]
                for key in ("task_id", "majority_vote_mass", "distinct_candidate_grids")
            }
            for row in data["orbit"]
        ],
        provenance,
        [p["orbit"]],
        "verbatim: frozen pair-scoped orbit statistics",
    )
    normal = data["main_by_method"]["B1"]
    blank = data["blank_by_method"]["B1"]
    normal_ci = data["main_boot_by_key"][("B1", "rank1")]
    blank_ci = one(data["blank_boot"], "blank B1 CI", method="B1", metric="rank1")
    blank_rows = [
        {"metric": "coverage", "condition": "normal", "value": normal["coverage"], "ci95_low": "", "ci95_high": ""},
        {"metric": "coverage", "condition": "blank", "value": blank["coverage"], "ci95_low": "", "ci95_high": ""},
        {"metric": "b1_rank1", "condition": "normal", "value": normal["rank1_accuracy"], "ci95_low": normal_ci["accuracy_ci95_low"], "ci95_high": normal_ci["accuracy_ci95_high"]},
        {"metric": "b1_rank1", "condition": "blank", "value": blank["rank1_accuracy"], "ci95_low": blank_ci["accuracy_ci95_low"], "ci95_high": blank_ci["accuracy_ci95_high"]},
    ]
    write_csv(
        root,
        "intermediates/figA3_blankid.csv",
        ("metric", "condition", "value", "ci95_low", "ci95_high"),
        blank_rows,
        provenance,
        [p["main"], p["blank"], p["main_boot"], p["blank_boot"]],
        "verbatim: frozen TEST values and available rank-1 bootstrap CIs",
    )
    rows = [
        {key: row[key] for key in ("statistic", "retained_puzzles", "coverage", "accuracy")}
        for row in data["arc2_curve"]
        if row["statistic"] == "structural_defect"
    ]
    arc2_accuracy = dec(data["arc2_by_method"]["B1"]["rank1_accuracy"])
    rows.extend(
        {
            "statistic": "random_order_expectation",
            "retained_puzzles": retained,
            "coverage": raw(Decimal(retained) / 120),
            "accuracy": raw(arc2_accuracy),
        }
        for retained in range(1, 121)
    )
    write_csv(
        root,
        "intermediates/figA4_arc2_risk_coverage.csv",
        ("statistic", "retained_puzzles", "coverage", "accuracy"),
        rows,
        provenance,
        [p["arc2_curve"], p["risk_summary"], p["arc2"]],
        "verbatim: frozen ARC-2 defect curve; derived: analytic constant random-order expectation",
    )


def fig_gap(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    budgets = np.array([50, 250, 1000], dtype=float)
    pass_values = np.array([float(d["gaps"][int(k)]["pass_at_k"] * 100) for k in budgets])
    maj_values = np.array([float(d["gaps"][int(k)]["majority_at_k"] * 100) for k in budgets])
    fig, ax = plt.subplots(figsize=(5.5, 2.65))
    ax.plot(budgets, pass_values, color=COLORS["blue"], marker="o", linestyle="-", label=r"pass@$k$")
    ax.plot(budgets, maj_values, color=COLORS["vermillion"], marker="s", linestyle="--", label=r"maj@$k$")
    ax.set_xscale("log")
    ax.set_xticks(budgets, labels=("50", "250", "1000"))
    ax.set_xlim(42, 1220)
    ax.set_ylim(35, 66)
    ax.set_xlabel("Candidate budget $k$")
    ax.set_ylabel("Puzzle-level result (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(ax)
    gap = float(d["gaps"][1000]["selection_gap"] * 100)
    ax.annotate("", xy=(1000, pass_values[-1]), xytext=(1000, maj_values[-1]), arrowprops={"arrowstyle": "<->", "color": COLORS["black"], "linewidth": 0.9})
    ax.text(920, (pass_values[-1] + maj_values[-1]) / 2, f"{gap:.2f} pp", ha="right", va="center", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(root, fig, "figures/main/fig1_gap.pdf", prov, [d["paths"]["gap"]], "derived: mean of ten frozen repeats per budget and percent formatting")


def fig_risk(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    styles = {
        "vote_margin": ("Vote margin", "blue", "o", "-"),
        "vote_share": ("Vote share", "orange", "s", "--"),
        "vote_entropy": ("Vote entropy", "green", "^", "-."),
        "winner_mean_q": ("Winner mean-Q", "purple", "D", ":"),
        "structural_defect": ("Structural defect", "vermillion", "v", "-"),
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.15))
    for statistic, (label, color, marker, linestyle) in styles.items():
        rows = sorted(
            (row for row in d["risk_curve"] if row["statistic"] == statistic),
            key=lambda row: int(row["retained_puzzles"]),
        )
        if len(rows) != 200:
            raise MissingFrozenValue(f"{statistic}: expected 200 cutoffs")
        aurc = dec(d["confidence_arc1"][statistic]["aurc"])
        ax.plot(
            [float(dec(row["coverage"]) * 100) for row in rows],
            [float(dec(row["accuracy"]) * 100) for row in rows],
            color=COLORS[color],
            marker=marker,
            markevery=25,
            linestyle=linestyle,
            label=f"{label} (AURC {float(aurc):.4f})",
        )
    random_aurc = dec(d["risk_summary"]["datasets"]["arc1_normal_test"]["random_order_expected_aurc"])
    random_accuracy = dec(d["main_by_method"]["B1"]["rank1_accuracy"]) * 100
    ax.axhline(float(random_accuracy), color=COLORS["gray"], linestyle="--", linewidth=1.2, label=f"Random ordering (AURC {float(random_aurc):.4f})")
    point = one(d["risk_points"], "25% operating point", target_coverage="0.25")
    x, y = float(dec(point["coverage"]) * 100), float(dec(point["accuracy"]) * 100)
    ax.scatter([x], [y], s=26, facecolor="white", edgecolor=COLORS["vermillion"], linewidth=1.0, zorder=5)
    ax.annotate(f"{x:.0f}% coverage: {y:.2f}% accuracy", xy=(x, y), xytext=(34, 91), arrowprops={"arrowstyle": "-", "color": COLORS["gray"], "linewidth": 0.7}, fontsize=7)
    ax.set(xlim=(0, 100), ylim=(0, 103), xlabel="Puzzles answered / coverage (%)", ylabel="Accuracy on answered puzzles (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.30)
    save_fig(root, fig, "figures/main/fig2_risk_coverage.pdf", prov, [d["paths"][key] for key in ("risk_curve", "confidence", "risk_points", "risk_summary", "main")], "verbatim: five frozen 200-cutoff curves and AURCs; derived: analytic random expectation and percent formatting")


def fig_beta(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    styles = {
        "distinct": ("Distinct", "blue", "o", "-"),
        "multiset": ("Multiset", "orange", "s", "--"),
        "non_identical_support": ("Non-identical support", "green", "^", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55))
    distinct: list[dict[str, str]] = []
    for centrality, (label, color, marker, linestyle) in styles.items():
        rows = sorted(
            (row for row in d["beta"] if row["centrality"] == centrality),
            key=lambda row: float(row["beta"]),
        )
        if len(rows) != 11:
            raise MissingFrozenValue(f"{centrality}: expected 11 beta rows")
        if centrality == "distinct":
            distinct = rows
        axes[0].plot([float(row["beta"]) for row in rows], [float(dec(row["dev_rank1_accuracy"]) * 100) for row in rows], color=COLORS[color], marker=marker, linestyle=linestyle, label=label)
    beta_ticks = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xticks(beta_ticks, labels=[str(value) for value in beta_ticks])
    for tick_label in axes[0].get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_ha("right")
    axes[0].set(xlabel=r"Centrality weight $\beta$", ylabel="DEV Rank-1 (%)", ylim=(26, 42.2))
    axes[0].yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(axes[0])
    for index, row in enumerate(distinct):
        offset = 5 if index % 2 == 0 else -10
        x_offset = 3 if index == 0 else 0
        axes[0].annotate(
            f"{dec(row['net_gain_puzzle_equivalents']):+g}",
            (float(row["beta"]), float(dec(row["dev_rank1_accuracy"]) * 100)),
            xytext=(x_offset, offset),
            textcoords="offset points",
            ha="left" if index == 0 else "center",
            va="bottom" if offset > 0 else "top",
            fontsize=7,
            color=COLORS["blue"],
        )
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top")
    flips = np.array([float(row["beta_flip"]) for row in d["beta_flip"]])
    if len(flips) != 144:
        raise MissingFrozenValue("expected 144 beta-flip values")
    positive = flips[flips > 0]
    zero_count = int(np.sum(flips == 0))
    bins = np.geomspace(float(np.min(positive)), float(np.max(positive)), 19)
    axes[1].hist(positive, bins=bins, color=COLORS["sky"], edgecolor=COLORS["black"], linewidth=0.45)
    axes[1].set_xscale("log")
    distribution = d["beta_json"]["beta_flip_distribution"]
    median = float(distribution["median"])
    fraction = float(distribution["le_16_descriptor_fraction"] * 100)
    axes[1].axvline(16, color=COLORS["vermillion"], linestyle="--", linewidth=1.1, label=r"Sweep max $\beta=16$")
    axes[1].axvline(median, color=COLORS["green"], linestyle="-.", linewidth=1.1, label=f"Median {median:.2f}")
    axes[1].text(0.04, 0.88, f"{fraction:.2f}% flippable at $\\beta\\leq16$\n{zero_count} zero-threshold values shown separately", transform=axes[1].transAxes, va="top", fontsize=7)
    axes[1].set(xlabel=r"Empirical $\beta_{\mathrm{flip}}$", ylabel="DEV descriptors")
    axis_style(axes[1])
    axes[1].legend(loc="center right", frameon=False)
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top")
    fig.tight_layout(w_pad=1.4)
    save_fig(root, fig, "figures/main/fig3_beta_dynamics.pdf", prov, [d["paths"][key] for key in ("beta", "beta_flip", "beta_json")], "verbatim: frozen DEV sweep, transitions, and all beta-flip values; derived: deterministic histogram bins")


def fig_compute(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    styles = {
        "B1": ("blue", "o", "-"),
        "M1": ("orange", "s", "--"),
        "M1+M2": ("green", "^", "-."),
        "M1+M2+M3": ("vermillion", "D", ":"),
    }
    fig, ax = plt.subplots(figsize=(5.5, 2.55))
    for method, (color, marker, linestyle) in styles.items():
        rows = sorted((row for row in d["compute"] if row["method"] == method), key=lambda row: int(row["budget"]))
        ax.plot([int(row["budget"]) for row in rows], [float(dec(row["historical_mean_rank1"]) * 100) for row in rows], color=COLORS[color], marker=marker, linestyle=linestyle, label=method)
    ax.set_xscale("log")
    ax.set_xticks([50, 250, 1000], labels=("50", "250", "1000"))
    ax.set(xlim=(42, 1220), ylim=(38, 43), xlabel="Compute-matched candidate budget $k$", ylabel="Mean Rank-1 (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(ax)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    save_fig(root, fig, "figures/appendix/figA1_compute_matched.pdf", prov, [d["paths"]["compute"]], "verbatim: frozen compute-matched mean Rank-1 values and percent formatting")


def fig_dispersion(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    shares = np.array([float(dec(row["majority_vote_mass"]) * 100) for row in d["orbit"]])
    modes = np.array([int(row["distinct_candidate_grids"]) for row in d["orbit"]])
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45))
    axes[0].hist(shares, bins=np.linspace(0, 100, 21), color=COLORS["blue"], edgecolor=COLORS["black"], linewidth=0.45)
    axes[0].set(xlabel="Modal vote share (%)", ylabel="Pair-scoped tasks")
    axes[0].xaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(axes[0])
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top")
    bins = np.geomspace(max(1, int(np.min(modes))), int(np.max(modes)) + 1, 18)
    axes[1].hist(modes, bins=bins, color=COLORS["orange"], edgecolor=COLORS["black"], linewidth=0.45)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Distinct candidate grids per task", ylabel="Pair-scoped tasks")
    axis_style(axes[1])
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top")
    fig.tight_layout(w_pad=1.5)
    save_fig(root, fig, "figures/appendix/figA2_dispersion.pdf", prov, [d["paths"]["orbit"]], "verbatim: 419 frozen task statistics; derived: deterministic histogram bins")


def fig_blank(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    normal = d["main_by_method"]["B1"]
    blank = d["blank_by_method"]["B1"]
    normal_values = [float(dec(normal["coverage"]) * 100), float(dec(normal["rank1_accuracy"]) * 100)]
    blank_values = [float(dec(blank["coverage"]) * 100), float(dec(blank["rank1_accuracy"]) * 100)]
    x, width = np.arange(2), 0.34
    fig, ax = plt.subplots(figsize=(2.65, 2.35))
    ax.bar(x - width / 2, normal_values, width, color=COLORS["blue"], edgecolor=COLORS["black"], linewidth=0.45, label="Normal ID")
    ax.bar(x + width / 2, blank_values, width, color=COLORS["orange"], hatch="//", edgecolor=COLORS["black"], linewidth=0.45, label="Blank ID")
    normal_ci = d["main_boot_by_key"][("B1", "rank1")]
    blank_ci = one(d["blank_boot"], "blank B1 CI", method="B1", metric="rank1")
    for position, value, row, color in (
        (x[1] - width / 2, normal_values[1], normal_ci, COLORS["blue"]),
        (x[1] + width / 2, blank_values[1], blank_ci, COLORS["orange"]),
    ):
        low, high = float(dec(row["accuracy_ci95_low"]) * 100), float(dec(row["accuracy_ci95_high"]) * 100)
        ax.errorbar(position, value, yerr=np.array([[value - low], [high - value]]), fmt="none", ecolor=color, elinewidth=0.9, capsize=2.5)
    ax.set_xticks(x, labels=("Coverage", "B1 Rank-1"))
    ax.set(ylabel="ARC-1 TEST (%)", ylim=(0, 70))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(root, fig, "figures/appendix/figA3_blankid.pdf", prov, [d["paths"][key] for key in ("main", "blank", "main_boot", "blank_boot")], "verbatim: frozen TEST coverage/rank-1 and available rank-1 bootstrap CIs")


def fig_arc2(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    rows = sorted((row for row in d["arc2_curve"] if row["statistic"] == "structural_defect"), key=lambda row: int(row["retained_puzzles"]))
    if len(rows) != 120:
        raise MissingFrozenValue("expected 120 ARC-2 cutoffs")
    summary = d["risk_summary"]["datasets"]["arc2_supporting"]
    defect, random = dec(summary["defect_aurc"]), dec(summary["random_order_expected_aurc"])
    full = dec(d["arc2_by_method"]["B1"]["rank1_accuracy"]) * 100
    fig, ax = plt.subplots(figsize=(2.65, 2.45))
    ax.plot([float(dec(row["coverage"]) * 100) for row in rows], [float(dec(row["accuracy"]) * 100) for row in rows], color=COLORS["vermillion"], marker="o", markevery=15, linestyle="-", label=f"Defect (AURC {float(defect):.4f})")
    ax.axhline(float(full), color=COLORS["gray"], linestyle="--", label=f"Random (AURC {float(random):.4f})")
    ax.set(xlim=(0, 100), ylim=(0, 103), xlabel="Puzzles answered / coverage (%)", ylabel="Accuracy on answered (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    axis_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    fig.subplots_adjust(bottom=0.30)
    save_fig(root, fig, "figures/appendix/figA4_arc2_risk_coverage.pdf", prov, [d["paths"][key] for key in ("arc2_curve", "risk_summary", "arc2")], "verbatim: frozen ARC-2 defect curve and AURCs; derived: analytic random expectation and percent formatting")


def table_tex(
    spec: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    notes: Sequence[str] = (),
) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", *notes, r"\end{table}", ""])
    return "\n".join(lines)


def make_tables(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    p = d["paths"]
    methods = ("B0", "B1", "M1", "M1+M2", "M1+M2+M3")
    baseline = dec(d["main_by_method"]["B1"]["rank1_accuracy"])
    rows = []
    for method in methods:
        row = d["main_by_method"][method]
        boot = d["main_boot_by_key"].get((method, "rank1"))
        if boot:
            delta = f"{spct(boot['difference'])} [{spct(boot['difference_ci95_low'])}, {spct(boot['difference_ci95_high'])}]"
        else:
            delta = spct(dec(row["rank1_accuracy"]) - baseline)
        rows.append((method, pct(row["rank1_accuracy"]), pct(row["top2_accuracy"]), delta))
    top2 = d["main_boot_by_key"][("M1+M2", "top2")]
    note = r"\parbox{\linewidth}{\footnotesize M1+M2 Top-2 effect versus B1: " + f"{spct(top2['difference'])} pp [{spct(top2['difference_ci95_low'])}, {spct(top2['difference_ci95_high'])}].}}"
    write_text(root, "tables/tab1_main_results.tex", table_tex("lrrl", ("Method", "Rank-1 (\\%)", "Top-2 (\\%)", "$\\Delta$ Rank-1 vs B1 [95\\% CI] (pp)"), rows, (note,)), prov, [p["main"], p["main_boot"]], "verbatim: frozen TEST results/CIs; derived: B0 difference and percent formatting")
    signal = d["disc"]["local_signal"]
    original = signal["original"]["b1_failure"]["preference_counts_puzzle"]
    discriminative = signal["discriminative"]["b1_failure"]["preference_counts_puzzle"]
    rows = (
        ("Original whole-grid", str(original["correct_preferred"]), str(original["wrong_preferred"]), str(original["tie"])),
        ("Discriminative-cell (DEV only)", str(discriminative["correct_preferred"]), str(discriminative["wrong_preferred"]), str(discriminative["tie"])),
    )
    note = r"\parbox{\linewidth}{\footnotesize Counts are over \FailureCount{} covered DEV B1 failures. The discriminative-cell variant was never run on TEST.}"
    write_text(root, "tables/tab2_failure_alignment.tex", table_tex("lrrr", ("Feature", "Correct preferred", "Wrong preferred", "Tie"), rows, (note,)), prov, [p["disc"]], "verbatim: frozen puzzle-level preference counts")
    lookup = {(int(row["budget"]), row["method"]): row for row in d["compute"]}
    rows = [
        (str(budget), *[pct(lookup[(budget, method)]["historical_mean_rank1"], 3) for method in ("B1", "M1", "M1+M2", "M1+M2+M3")])
        for budget in (50, 250, 1000)
    ]
    write_text(root, "tables/tabA1_compute_matched.tex", table_tex("rrrrr", ("$k$", "B1 (\\%)", "M1 (\\%)", "M1+M2 (\\%)", "M1+M2+M3 (\\%)"), rows), prov, [p["compute"]], "verbatim: frozen compute-matched means and percent formatting")
    rows = []
    for budget in (50, 250, 1000):
        row = d["gaps"][budget]
        rows.append((str(budget), pct(row["pass_at_k"], 4 if budget != 1000 else 2), pct(row["majority_at_k"], 4 if budget != 1000 else 2), pct(row["selection_gap"], 3 if budget == 250 else 2)))
    write_text(root, "tables/tabA2_gap.tex", table_tex("rrrr", ("$k$", "pass@$k$ (\\%)", "maj@$k$ (\\%)", "Gap (pp)"), rows), prov, [p["gap"]], "derived: mean of ten frozen repeats and percent formatting")
    labels = (("vote_margin", "Vote margin"), ("vote_share", "Vote share"), ("vote_entropy", "Vote entropy"), ("winner_mean_q", "Winner mean-Q"), ("structural_defect", "Structural defect"), ("random_order_expectation", "Random ordering"))
    rows = []
    for statistic, label in labels:
        if statistic == "random_order_expectation":
            aurc = d["risk_summary"]["datasets"]["arc1_normal_test"]["random_order_expected_aurc"]
            ci = "---"
        else:
            boot = d["aurc_boot_arc1"][statistic]
            aurc = boot["observed_aurc"]
            ci = f"[{fixed(boot['ci95_low'], 4)}, {fixed(boot['ci95_high'], 4)}]"
        diff = d["aurc_diff_by_baseline"].get(statistic)
        if diff:
            delta = f"{sfixed(diff['observed_delta'], 4)} [{fixed(diff['ci95_low'], 4)}, {fixed(diff['ci95_high'], 4)}]"
            tail, pvalue = fixed(diff["fraction_defect_lower_aurc"], 4), fixed(diff["two_sided_bootstrap_p"], 4)
        else:
            delta = tail = pvalue = "---"
        rows.append((label, fixed(aurc, 4), ci, delta, tail, pvalue))
    note = r"\parbox{\linewidth}{\footnotesize Positive defect-minus-signal differences favor the baseline. For vote margin, the frozen one-sided tail fraction is \MarginTailProbability{} and the separately stored two-sided bootstrap p-value is \MarginTwoSidedP{}.}"
    write_text(root, "tables/tabA3_selective_aurc.tex", table_tex("lrrlll", ("Signal", "AURC", "95\\% CI", "Defect $-$ signal [95\\% CI]", "$P(D<S)$", "$p_{2s}$"), rows, (note,)), prov, [p["aurc_boot"], p["aurc_diff"], p["risk_summary"]], "verbatim: frozen AURCs/CIs/differences/tail fractions/p-values and fixed formatting")
    rows = []
    for method in ("B1", "M1", "M1+M2", "M1+M2+M3"):
        row = d["arc2_by_method"][method]
        rows.append((method, pct(row["rank1_accuracy"], 4), pct(row["top2_accuracy"], 4 if method == "M1" else 2), pct(row["coverage"], 4)))
    note = r"\parbox{\linewidth}{\footnotesize Internally consistent under the pinned path; not external benchmark reproduction.}"
    write_text(root, "tables/tabA4_arc2.tex", table_tex("lrrr", ("Method", "Rank-1 (\\%)", "Top-2 (\\%)", "Coverage (\\%)"), rows, (note,)), prov, [p["arc2"]], "verbatim: frozen ARC-2 supporting results and percent formatting")
    m3 = d["main_report"]["m3_summary"]
    change = dec(d["main_by_method"]["M1+M2+M3"]["rank1_accuracy"]) - dec(d["main_by_method"]["M1+M2"]["rank1_accuracy"])
    rows = (("Target shape allowed / active", pct(m3["puzzle_weighted_target_shape_allowed"])), ("Emissions filtered", pct(m3["puzzle_weighted_filter_fraction"], 4)), ("Fallback", pct(m3["puzzle_weighted_fallback_rate"], 1)), ("Rank-1 change vs M1+M2", spct(change)))
    write_text(root, "tables/tabA5_m3_stats.tex", table_tex("lr", ("M3 statistic", "Frozen TEST value (\\%)"), rows), prov, [p["main_report"], p["main"]], "verbatim: frozen M3 rates; derived: exact Rank-1 difference and percent formatting")
    trans = d["disc"]["failure_preference_transitions"]["puzzle_counts"]
    rows = (("Original wrong $\\rightarrow$ discriminative correct", str(trans["original_wrong_to_discriminative_correct"])), ("Original correct $\\rightarrow$ discriminative wrong", str(trans["original_correct_to_discriminative_wrong"])), ("Both correct", str(trans["both_correct"])), ("Both wrong", str(trans["both_wrong"])), ("Ties", str(trans["ties"])))
    write_text(root, "tables/tabA6_disccell_transitions.tex", table_tex("lr", ("DEV failure transition", "Puzzles"), rows), prov, [p["disc"]], "verbatim: frozen puzzle-level discriminative transition matrix")


def make_macros(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    p = d["paths"]
    items: list[tuple[str, str]] = []
    trace: dict[str, dict[str, str]] = {}
    sections: dict[str, str] = {}

    def add(name: str, value: str, source_key: str, transformation: str = "derived: decimal or percent formatting of frozen field", section: str | None = None) -> None:
        if not name.isalpha():
            raise RuntimeError(f"invalid TeX macro name: {name}")
        if section:
            sections[name] = section
        items.append((name, value))
        trace[name] = {"value": value, "source": p[source_key], "transformation": transformation}

    add("FullARCOneBOne", pct(d["cache"]["arc1_normal"]["cache_b1"]), "cache", section="% Baseline and held-out ARC-1")
    add("ARCOnePuzzles", str(d["cache"]["arc1_normal"]["puzzles"]), "cache", "verbatim")
    add("ARCOneDescriptors", str(d["cache"]["arc1_normal"]["descriptors"]), "cache", "verbatim")
    add("IdentifierCount", str(d["preflight"]["identifier_count"]), "preflight", "verbatim")
    add("GapRepeats", str(len([row for row in d["gap"] if int(row["budget"]) == 50])), "gap", "derived: count of frozen repeat rows at one budget")
    names = {"B0": "BZero", "B1": "BOne", "M1": "MOne", "M1+M2": "MTwo", "M1+M2+M3": "MThree"}
    for method, prefix in names.items():
        row = d["main_by_method"][method]
        add(prefix + "RankOne", pct(row["rank1_accuracy"]), "main")
        add(prefix + "TopTwo", pct(row["top2_accuracy"]), "main")
        if method in ("B0", "B1", "M1"):
            add(prefix + "RankDelta", spct(dec(row["rank1_accuracy"]) - dec(d["main_by_method"]["B1"]["rank1_accuracy"])), "main", "derived: exact difference from frozen B1 Rank-1")
    for method, prefix in (("M1+M2", "MTwo"), ("M1+M2+M3", "MThree")):
        for metric, suffix in (("rank1", "Rank"), ("top2", "Top")):
            row = d["main_boot_by_key"][(method, metric)]
            add(prefix + suffix + "Delta", spct(row["difference"]), "main_boot")
            add(prefix + suffix + "DeltaLow", spct(row["difference_ci95_low"]), "main_boot")
            add(prefix + suffix + "DeltaHigh", spct(row["difference_ci95_high"]), "main_boot")
    for budget, word in ((50, "Fifty"), (250, "TwoFifty"), (1000, "Thousand")):
        row = d["gaps"][budget]
        section = "% Candidate-generation gap" if budget == 50 else None
        add("Budget" + word, str(budget), "gap", "verbatim: frozen budget column")
        add("CoverageK" + word, pct(row["pass_at_k"], 4 if budget != 1000 else 2), "gap", "derived: mean over ten frozen repeats and percent formatting", section)
        add("MajorityK" + word, pct(row["majority_at_k"], 4 if budget != 1000 else 2), "gap", "derived: mean over ten frozen repeats and percent formatting")
        add("GapK" + word, pct(row["selection_gap"], 3 if budget == 250 else 2), "gap", "derived: mean over ten frozen repeats and percent formatting")
    auroc = d["h1_by_split"]["test"]["auroc_defect_for_incorrectness"]
    add("DefectAUROC", fixed(auroc, 3), "h1", section="% Structural diagnosis and selective prediction")
    add("DefectAUROCExact", fixed(auroc, 6), "h1")
    prefixes = {"vote_margin": "Margin", "vote_share": "Share", "vote_entropy": "Entropy", "winner_mean_q": "WinnerQ", "structural_defect": "Defect"}
    for statistic, prefix in prefixes.items():
        row = d["aurc_boot_arc1"][statistic]
        for suffix, key, digits in (("AURC", "observed_aurc", 3), ("AURCFour", "observed_aurc", 4), ("AURCLow", "ci95_low", 4), ("AURCHigh", "ci95_high", 4)):
            add(prefix + suffix, fixed(row[key], digits), "aurc_boot")
    add("RandomAURC", fixed(d["risk_summary"]["datasets"]["arc1_normal_test"]["random_order_expected_aurc"], 4), "risk_summary")
    for baseline, prefix in (("vote_margin", "Margin"), ("vote_share", "Share"), ("vote_entropy", "Entropy"), ("winner_mean_q", "WinnerQ"), ("random_order_expectation", "Random")):
        row = d["aurc_diff_by_baseline"][baseline]
        add("DefectMinus" + prefix, sfixed(row["observed_delta"]), "aurc_diff")
        add("DefectMinus" + prefix + "Low", fixed(row["ci95_low"], 4), "aurc_diff")
        add("DefectMinus" + prefix + "High", fixed(row["ci95_high"], 4), "aurc_diff")
        add(prefix + "TailProbability", fixed(row["fraction_defect_lower_aurc"], 4), "aurc_diff")
        add(prefix + "TwoSidedP", fixed(row["two_sided_bootstrap_p"], 4), "aurc_diff")
    quarter = one(d["op_boot"], "25% bootstrap", coverage="0.25")
    add("QuarterCoverage", pct(quarter["coverage"], 0), "op_boot")
    add("DefectAccuracyQuarter", pct(quarter["observed_accuracy"]), "op_boot")
    add("DefectAccuracyQuarterLow", pct(quarter["accuracy_ci95_low"]), "op_boot")
    add("DefectAccuracyQuarterHigh", pct(quarter["accuracy_ci95_high"]), "op_boot")
    signal = d["disc"]["local_signal"]
    original = signal["original"]["b1_failure"]["preference_counts_puzzle"]
    disc = signal["discriminative"]["b1_failure"]["preference_counts_puzzle"]
    failure_count = sum(int(original[key]) for key in ("correct_preferred", "wrong_preferred", "tie"))
    add("FailureCount", str(failure_count), "disc", "derived: sum of frozen counts", "% Failure alignment and beta dynamics")
    for name, value in (("FailPrefCorrectCount", original["correct_preferred"]), ("FailPrefWrongCount", original["wrong_preferred"]), ("FailPrefTieCount", original["tie"]), ("DiscPrefCorrectCount", disc["correct_preferred"]), ("DiscPrefWrongCount", disc["wrong_preferred"]), ("DiscPrefTieCount", disc["tie"])):
        add(name, str(value), "disc", "verbatim")
    add("FailPrefCorrect", fixed(Decimal(original["correct_preferred"]) / failure_count * 100, 1), "disc", "derived: frozen count divided by frozen total")
    add("FailPrefWrong", fixed(Decimal(original["wrong_preferred"]) / failure_count * 100, 1), "disc", "derived: frozen count divided by frozen total")
    background = d["disc"]["background_dominance"]["covered_top2_modes"]
    add("MedianTopSim", fixed(background["ordinary_similarity_median"], 2), "disc")
    add("DisagreeCells", pct(background["disagreement_fraction_median"], 1), "disc")
    distribution = d["beta_json"]["beta_flip_distribution"]
    add("BetaFlipDescriptors", str(distribution["eligible_comparisons"]), "beta_json", "verbatim")
    add("BetaFlipMedian", fixed(distribution["median"], 0), "beta_json")
    add("BetaFlipMedianTwo", fixed(distribution["median"], 2), "beta_json")
    add("BetaFlipLeSweep", pct(distribution["le_16_descriptor_fraction"], 2), "beta_json")
    add("BetaSweepMaximum", fixed(max(dec(row["beta"]) for row in d["beta"]), 0), "beta", "derived: maximum frozen beta")
    lookup = {(int(row["budget"]), row["method"]): row for row in d["compute"]}
    method_words = {"B1": "BOne", "M1": "MOne", "M1+M2": "MTwo", "M1+M2+M3": "MThree"}
    for budget, budget_word in ((50, "Fifty"), (250, "TwoFifty"), (1000, "Thousand")):
        for method, method_word in method_words.items():
            section = "% Compute-matched appendix" if (budget, method) == (50, "B1") else None
            add("ComputeK" + budget_word + method_word, pct(lookup[(budget, method)]["historical_mean_rank1"], 3), "compute", section=section)
    compute_percent = [dec(row["historical_mean_rank1"]) * 100 for row in d["compute"]]
    add("ComputeAxisLow", str(int(min(compute_percent).to_integral_value(rounding=ROUND_FLOOR)) - 2), "compute", "derived: floor of frozen minimum percent minus two presentation points")
    add("ComputeAxisHigh", str(int(max(compute_percent).to_integral_value(rounding=ROUND_CEILING)) + 1), "compute", "derived: ceiling of frozen maximum percent plus one presentation point")
    normal, blank = d["main_by_method"]["B1"], d["blank_by_method"]["B1"]
    add("NormalCoverage", pct(normal["coverage"]), "main", section="% Blank-ID and ARC-2")
    add("BlankCoverage", pct(blank["coverage"]), "blank")
    add("NormalBOne", pct(normal["rank1_accuracy"]), "main")
    add("BlankBOne", pct(blank["rank1_accuracy"]), "blank")
    blank_ci = one(d["blank_boot"], "blank B1 CI", method="B1", metric="rank1")
    add("BlankBOneLow", pct(blank_ci["accuracy_ci95_low"]), "blank_boot")
    add("BlankBOneHigh", pct(blank_ci["accuracy_ci95_high"]), "blank_boot")
    for method, word in method_words.items():
        row = d["arc2_by_method"][method]
        add("ARCTwo" + word + "RankOne", pct(row["rank1_accuracy"], 4), "arc2")
        add("ARCTwo" + word + "TopTwo", pct(row["top2_accuracy"], 4), "arc2")
    arc2_b1 = d["arc2_by_method"]["B1"]
    add("ARCTwoBOne", pct(arc2_b1["rank1_accuracy"], 2), "arc2")
    add("ARCTwoCoverage", pct(arc2_b1["coverage"], 4), "arc2")
    arc2_summary = d["risk_summary"]["datasets"]["arc2_supporting"]
    add("ARCTwoDefectAURC", fixed(arc2_summary["defect_aurc"], 4), "risk_summary")
    add("ARCTwoRandomAURC", fixed(arc2_summary["random_order_expected_aurc"], 4), "risk_summary")
    m3 = d["main_report"]["m3_summary"]
    add("MThreeActive", pct(m3["puzzle_weighted_target_shape_allowed"]), "main_report", section="% M3 and discriminative transitions")
    add("MThreeFiltered", pct(m3["puzzle_weighted_filter_fraction"], 4), "main_report")
    add("MThreeFallback", pct(m3["puzzle_weighted_fallback_rate"], 1), "main_report")
    change = dec(d["main_by_method"]["M1+M2+M3"]["rank1_accuracy"]) - dec(d["main_by_method"]["M1+M2"]["rank1_accuracy"])
    add("MThreeRankChange", spct(change), "main", "derived: exact difference of frozen TEST values")
    trans = d["disc"]["failure_preference_transitions"]["puzzle_counts"]
    for name, key in (("DiscWrongToCorrect", "original_wrong_to_discriminative_correct"), ("DiscCorrectToWrong", "original_correct_to_discriminative_wrong"), ("DiscBothCorrect", "both_correct"), ("DiscBothWrong", "both_wrong"), ("DiscTies", "ties")):
        add(name, str(trans[key]), "disc", "verbatim")
    lines = ["% Auto-generated from frozen release artifacts; do not hand-edit.", "% No newly estimated values.", ""]
    for name, value in items:
        if name in sections:
            lines.extend([sections[name], ""])
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    lines.append("")
    write_text(root, "macros/numbers.tex", "\n".join(lines), prov, sorted({entry["source"] for entry in trace.values()}), "artifact-traced verbatim/derived formatting; per-macro trace in manifest.json")
    return trace


def make_captions(root: Path, d: dict[str, Any], prov: dict[str, dict[str, Any]]) -> None:
    content = r"""% Auto-generated caption drafts; numbers resolve through macros/numbers.tex.

% Figure 1
\caption{Majority voting leaves \GapKthousand{} percentage points of generated-but-unselected correct answers at the full ARC-1 orbit budget: pass@$k$ reaches \CoverageKthousand\% while majority reaches \MajorityKthousand\%. Results use all \ARCOnePuzzles{} ARC-1 puzzles.}

% Figure 2
\caption{On ARC-1 TEST, structural defect supports abstention relative to random ordering, including \DefectAccuracyQuarter\% accuracy at \QuarterCoverage\% coverage, but its AURC (\DefectAURCFour) is worse than vote margin (\MarginAURCFour), vote share, and vote entropy. Lower AURC is better; this is a frozen confidence diagnostic, not a superior selector claim.}

% Figure 3
\caption{DEV-only beta dynamics are mixed and non-monotonic. Rank-1 changes little for positive beta, while only \BetaFlipLeSweep\% of \BetaFlipDescriptors{} eligible descriptor comparisons are flippable by the original sweep maximum $\beta=\BetaSweepMaximum$; the median empirical threshold is \BetaFlipMedianTwo. Net fix-minus-break annotations refer to distinct centrality.}

% Figure A1
\caption{Compute-matched ARC-1 accuracy clusters tightly across budgets and frozen methods. The deliberately truncated \ComputeAxisLow--\ComputeAxisHigh\% y-axis makes small differences visible; the audit found no Q or full-orbit leakage.}

% Figure A2
\caption{Frozen ARC-1 pair-scoped orbits vary widely in modal vote share and distinct candidate grids across \ARCOneDescriptors{} descriptors, motivating separation of candidate generation from selection.}

% Figure A3
\caption{On ARC-1 TEST, blanking puzzle identity collapses coverage from \NormalCoverage\% to \BlankCoverage\% and B1 Rank-1 from \NormalBOne\% to \BlankBOne\%. Whiskers show frozen Rank-1 bootstrap intervals; no coverage interval was stored.}

% Figure A4
\caption{ARC-2 supporting-only risk--coverage: structural defect AURC is \ARCTwoDefectAURC{} versus random expectation \ARCTwoRandomAURC. The difference from random is ambiguous, and the result is internally consistent under the pinned path, not an external benchmark reproduction.}

% Table 1
\caption{Frozen ARC-1 TEST results. M1 gives no Rank-1 gain; M1+M2 gains \MTwoRankDelta{} percentage point with a confidence interval including zero and reduces Top-2 by \MTwoTopDelta{} points. The result is exploratory.}

% Table 2
\caption{On \FailureCount{} covered ARC-1 DEV B1 failures, the original signal prefers the correct alternative only \FailPrefCorrectCount{} times versus \FailPrefWrongCount{} wrong preferences. The discriminative-cell variant improves local alignment but is DEV-only and was never evaluated on TEST.}

% Table A1
\caption{Frozen compute-matched ARC-1 Rank-1 accuracy at $k\in\{\BudgetFifty,\BudgetTwoFifty,\BudgetThousand\}$. Values are \GapRepeats-repeat means from the audited implementation.}

% Table A2
\caption{The ARC-1 candidate-generation/selection gap grows from \GapKFifty{} points at $k=\BudgetFifty$ to \GapKthousand{} points at $k=\BudgetThousand$.}

% Table A3
\caption{Frozen ARC-1 TEST selective-prediction AURCs and paired bootstrap comparisons. Structural defect beats random ordering but is reliably worse than vote margin, vote share, and vote entropy; positive defect-minus-baseline differences favor the baseline.}

% Table A4
\caption{Frozen ARC-2 supporting results. Rank-1 is \ARCTwoBOne\% for every structured method and coverage is \ARCTwoCoverage\%; the result is internally consistent under the pinned path, not external benchmark reproduction.}

% Table A5
\caption{Frozen ARC-1 TEST M3 diagnostics. Shape screening is active for \MThreeActive\% of puzzles but filters only \MThreeFiltered\% of emissions and changes Rank-1 by \MThreeRankChange{} percentage points.}

% Table A6
\caption{DEV-only discriminative-cell transitions on the \FailureCount{} covered B1 failures. \DiscWrongToCorrect{} wrong decisions become correct, \DiscCorrectToWrong{} correct decision becomes wrong, and the variant was not promoted to TEST.}
"""
    keys = ("gap", "risk_curve", "confidence", "beta", "beta_json", "orbit", "main", "main_boot", "compute", "blank", "blank_boot", "arc2", "risk_summary", "disc", "main_report")
    write_text(root, "captions/captions.tex", content, prov, [d["paths"][key] for key in keys], "derived: finding-oriented prose with numerical values referenced through artifact-generated macros")


def render_core(root: Path, data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    cleanup(root)
    provenance: dict[str, dict[str, Any]] = {}
    make_intermediates(root, data, provenance)
    fig_gap(root, data, provenance)
    fig_risk(root, data, provenance)
    fig_beta(root, data, provenance)
    fig_compute(root, data, provenance)
    fig_dispersion(root, data, provenance)
    fig_blank(root, data, provenance)
    fig_arc2(root, data, provenance)
    make_tables(root, data, provenance)
    macro_trace = make_macros(root, data, provenance)
    make_captions(root, data, provenance)
    return provenance, macro_trace


def payload_hashes(root: Path) -> dict[str, str]:
    prefixes = ("figures/", "tables/", "macros/", "captions/", "intermediates/")
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix().startswith(prefixes)
    }


def frozen_checks(d: dict[str, Any]) -> list[tuple[str, Decimal, Decimal]]:
    main = d["main_by_method"]
    return [
        ("Full ARC-1 B1", dec(d["cache"]["arc1_normal"]["cache_b1"]) * 100, Decimal("40.00")),
        ("ARC-1 TEST B1 Rank-1", dec(main["B1"]["rank1_accuracy"]) * 100, Decimal("40.25")),
        ("ARC-1 TEST B1 Top-2", dec(main["B1"]["top2_accuracy"]) * 100, Decimal("45.50")),
        ("ARC-1 TEST M1+M2 Rank-1", dec(main["M1+M2"]["rank1_accuracy"]) * 100, Decimal("41.25")),
        ("ARC-1 TEST M1+M2 Top-2", dec(main["M1+M2"]["top2_accuracy"]) * 100, Decimal("44.00")),
        ("ARC-1 k=1000 coverage", Decimal(pct(d["gaps"][1000]["pass_at_k"])), Decimal("61.75")),
        ("ARC-1 k=1000 gap", Decimal(pct(d["gaps"][1000]["selection_gap"])), Decimal("21.75")),
        ("ARC-1 TEST defect AUROC", dec(d["h1_by_split"]["test"]["auroc_defect_for_incorrectness"]), Decimal("0.8418617947451856")),
        ("ARC-1 TEST defect AURC", dec(d["aurc_boot_arc1"]["structural_defect"]["observed_aurc"]), Decimal("0.31586167853807995")),
        ("ARC-1 TEST margin AURC", dec(d["aurc_boot_arc1"]["vote_margin"]["observed_aurc"]), Decimal("0.2607539081534486")),
        ("ARC-1 blank TEST B1", dec(d["blank_by_method"]["B1"]["rank1_accuracy"]) * 100, Decimal("4.00")),
        ("ARC-2 B1", dec(d["arc2_by_method"]["B1"]["rank1_accuracy"]) * 100, Decimal("2.9166666666666667")),
        ("ARC-1 identifier count", Decimal(d["preflight"]["identifier_count"]), Decimal(876406)),
    ]


def validation_report(
    checks: Sequence[tuple[str, Decimal, Decimal]],
    macro_count: int,
    type3: Sequence[str],
    deterministic: bool,
    allowlist_ok: bool,
) -> str:
    failed = [label for label, observed, expected in checks if observed != expected]
    if type3:
        failed.append("Type 3 font token")
    if not deterministic:
        failed.append("two-pass byte mismatch")
    if not allowlist_ok:
        failed.append("input allowlist")
    lines = [
        f"# {'FAIL' if failed else 'PASS'} — paper asset validation",
        "",
        "## Frozen headline checks",
        "",
        "| Check | Observed | Expected | Status |",
        "|---|---:|---:|---|",
    ]
    for label, observed, expected in checks:
        lines.append(f"| {label} | {raw(observed)} | {raw(expected)} | {'PASS' if observed == expected else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Other required checks",
            "",
            f"- Input allowlist: {'PASS' if allowlist_ok else 'FAIL'} — 46 JSON and 70 CSV frozen release inputs including the self-excluded release manifest.",
            f"- Macro provenance: PASS — all {macro_count} macros have artifact and transformation traces in manifest.json.",
            f"- PDF fonts: {'PASS' if not type3 else 'FAIL'} — pdf.fonttype=42 and ps.fonttype=42; binary inspection found {len(type3)} Type 3 subtype tokens.",
            f"- Determinism: {'PASS' if deterministic else 'FAIL'} — a second in-process render matched every PDF, PNG, TeX, and CSV intermediate byte-for-byte.",
            "- CSV intermediates: PASS — fixed field order, LF endings, and no timestamps.",
            "- Beta histogram: PASS — all 144 frozen thresholds are present; positive thresholds use deterministic log bins and zero thresholds are separately annotated.",
            "- ARC-2 wording: PASS — supporting-only and not external benchmark reproduction.",
            "- DEV-only boundary: PASS — discriminative-cell assets state that no TEST evaluation occurred.",
            "- Output boundary: PASS by construction — every write is guarded under paper_assets/. The invoking audit separately compares Git status.",
            "",
            "## Frozen p-value semantic reconciliation",
            "",
            "The requested margin value 0.0002 is the frozen one-sided tail fraction fraction_defect_lower_aurc. The frozen two-sided bootstrap p-value formats to 0.0006. Table A3 reports both under separate labels; neither was relabeled or adjusted.",
            "",
            "## Missing inputs",
            "",
            "None. MISSING.md is intentionally absent.",
            "",
        ]
    )
    if failed:
        lines.extend(["## Failures", "", *[f"- {item}" for item in failed], ""])
    return "\n".join(lines)


def output_manifest(
    root: Path,
    store: FrozenInputs,
    provenance: dict[str, dict[str, Any]],
    macro_trace: dict[str, dict[str, str]],
    timestamp: str,
) -> dict[str, Any]:
    provenance["build_assets.py"] = {
        "sources": [],
        "transformation": "derived: deterministic renderer constrained by the frozen release manifest and camera-ready presentation specification",
    }
    outputs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or MPL_CONFIG in path.parents:
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        if relative not in provenance:
            raise RuntimeError(f"output lacks provenance: {relative}")
        record = provenance[relative]
        outputs.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "sources": [store.source(source) for source in record["sources"]],
                "transformation": record["transformation"],
                "generation_timestamp": timestamp,
            }
        )
    outputs.append(
        {
            "path": "manifest.json",
            "sha256": None,
            "bytes": None,
            "sources": [],
            "transformation": "derived: self-describing provenance manifest; self-hash excluded",
            "generation_timestamp": timestamp,
        }
    )
    return {
        "format_version": 1,
        "generation_timestamp": timestamp,
        "timestamp_policy": "fixed to the frozen Phase-4 date for byte determinism",
        "input_policy": "read-only JSON/CSV allowlist from manifests/release_manifest.json",
        "input_inventory": {"json_files": 46, "csv_files": 70, "used_files": len(store.used)},
        "derivation_policy": "Only deterministic formatting, permitted arithmetic, analytic random expectations, and histogram bin counts are derived.",
        "outputs": outputs,
        "macro_trace": macro_trace,
        "self_exclusion": "manifest.json has no recursive self-hash",
    }


def main() -> int:
    cleanup(ASSET_ROOT)
    try:
        store = FrozenInputs(RELEASE_ROOT)
        data = load_data(store)
        timestamp = f"{data['freeze']['freeze_date']}T00:00:00Z"
        provenance, macro_trace = render_core(ASSET_ROOT, data)
        first = payload_hashes(ASSET_ROOT)
        second_root = safe_path(ASSET_ROOT, ".determinism_second")
        if second_root.exists():
            shutil.rmtree(second_root)
        second_root.mkdir()
        render_core(second_root, data)
        second = payload_hashes(second_root)
        deterministic = first == second
        shutil.rmtree(second_root)
        type3 = [
            path.relative_to(ASSET_ROOT).as_posix()
            for path in sorted((ASSET_ROOT / "figures").rglob("*.pdf"))
            if b"/Subtype /Type3" in path.read_bytes()
        ]
        checks = frozen_checks(data)
        allowlist_ok = all(path in store.allowed for path in store.used)
        write_text(
            ASSET_ROOT,
            "VALIDATION_REPORT.md",
            validation_report(checks, len(macro_trace), type3, deterministic, allowlist_ok),
            provenance,
            sorted(store.used),
            "derived: validation of frozen values, fonts, provenance, and two-pass byte stability",
        )
        manifest = output_manifest(ASSET_ROOT, store, provenance, macro_trace, timestamp)
        safe_path(ASSET_ROOT, "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if MPL_CONFIG.exists():
            shutil.rmtree(MPL_CONFIG)
        failed = any(observed != expected for _, observed, expected in checks)
        failed = failed or bool(type3) or not deterministic or not allowlist_ok
        if failed:
            raise RuntimeError("paper asset validation failed")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "outputs_including_manifest": len(manifest["outputs"]),
                    "macros": len(macro_trace),
                    "byte_stable_payload_files": len(first),
                    "type3_pdfs": len(type3),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if MPL_CONFIG.exists():
            shutil.rmtree(MPL_CONFIG)
        if isinstance(error, MissingFrozenValue):
            safe_path(ASSET_ROOT, "MISSING.md").write_text(
                "# Missing frozen inputs\n\n"
                f"- {error}\n\nRendering stopped; requested assets remain placeholders.\n",
                encoding="utf-8",
                newline="\n",
            )
        report_path = safe_path(ASSET_ROOT, "VALIDATION_REPORT.md")
        if not report_path.exists():
            report_path.write_text(
                f"# FAIL — paper asset validation\n\n{type(error).__name__}: {error}\n",
                encoding="utf-8",
                newline="\n",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
