"""Generate all cartel paper figures exclusively from saved aggregate CSV data."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from common import read_csv

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.7,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: Any, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _nearest(values: Sequence[float], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _select_point(
    rows: Sequence[dict[str, str]], requests: dict[str, float]
) -> list[dict[str, str]]:
    selected = list(rows)
    for field, target in requests.items():
        available = sorted({_number(row, field) for row in selected})
        chosen = _nearest(available, target)
        selected = [
            row for row in selected if math.isclose(_number(row, field), chosen)
        ]
    return selected


def plot_disclosure(root: Path, alpha: float, alpha_t: float) -> list[Path]:
    source = root / "aggregate" / "summary.csv"
    if not source.exists():
        return []
    rows = read_csv(source)
    selected = sorted(
        _select_point(rows, {"alpha": alpha, "alpha_t": alpha_t}),
        key=lambda row: int(float(row["ell"])),
    )
    x = np.array([_number(row, "ell") for row in selected])
    y_sim = np.array([_number(row, "utility_difference_sim_mean") for row in selected])
    y_err = np.array([_number(row, "utility_difference_sim_ci95") for row in selected])
    y_theory = np.array([_number(row, "utility_difference_theory") for row in selected])
    threshold = int(float(selected[0]["ell_star_theory"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    ax.plot(x, y_theory, color=COLORS[0], lw=1.2, label="Theory")
    ax.errorbar(
        x,
        y_sim,
        yerr=y_err,
        fmt="o",
        ms=3,
        capsize=2,
        color=COLORS[1],
        label="Simulation",
    )
    ax.axhline(0.0, color="0.45", lw=0.7)
    ax.axvline(
        threshold, color=COLORS[2], lw=0.9, ls="--", label=rf"$\ell_\star={threshold}$"
    )
    ax.set(xlabel=r"Private lead $\ell$", ylabel=r"$U_{\rm obey}-U_{\rm betray}$")
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.legend(frameon=False)
    _save(fig, root / "figures", "figure_A1_disclosure_utility")

    unique: dict[tuple[float, float], float] = {}
    for row in rows:
        unique[(_number(row, "alpha_t"), _number(row, "alpha"))] = _number(
            row, "ell_star_theory"
        )
    xs = sorted({key[0] for key in unique})
    ys = sorted({key[1] for key in unique})
    matrix = np.full((len(ys), len(xs)), np.nan)
    for (x_value, y_value), value in unique.items():
        matrix[ys.index(y_value), xs.index(x_value)] = value
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(xs)), [f"{value:.2f}" for value in xs], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(ys)), [f"{value:.2f}" for value in ys])
    ax.set(
        xlabel=r"Traitor hashrate $\alpha_t$", ylabel=r"Total cartel hashrate $\alpha$"
    )
    fig.colorbar(image, ax=ax, label=r"$\ell_\star$")
    _save(fig, root / "figures", "figure_A2_disclosure_threshold_heatmap")
    return [
        root / "figures" / "figure_A1_disclosure_utility.pdf",
        root / "figures" / "figure_A2_disclosure_threshold_heatmap.pdf",
    ]


def _phase_matrix(
    rows: Sequence[dict[str, str]], w_value: float
) -> tuple[list[float], list[float], np.ndarray, list[tuple[float, float]]]:
    chosen = [row for row in rows if math.isclose(_number(row, "w_over_T"), w_value)]
    xs = sorted({_number(row, "alpha_t") for row in chosen})
    ys = sorted({_number(row, "alpha_l") for row in chosen})
    grouped: dict[tuple[float, float], list[dict[str, str]]] = {}
    for row in chosen:
        grouped.setdefault(
            (_number(row, "alpha_t"), _number(row, "alpha_l")), []
        ).append(row)
    order = {"HONEST": 0, "EXPEL_AND_SOLO_WITHHOLD": 1, "TOLERATE": 2}
    matrix = np.full((len(ys), len(xs)), np.nan)
    disagreements: list[tuple[float, float]] = []
    for (alpha_t, alpha_l), members in grouped.items():
        theory_best = max(members, key=lambda row: _number(row, "loyal_share_theory"))[
            "regime"
        ]
        simulation_best = max(
            members, key=lambda row: _number(row, "loyal_share_sim_mean")
        )["regime"]
        matrix[ys.index(alpha_l), xs.index(alpha_t)] = order[theory_best]
        if theory_best != simulation_best:
            disagreements.append((alpha_t, alpha_l))
    return xs, ys, matrix, disagreements


def plot_tolerance(root: Path) -> list[Path]:
    source = root / "aggregate" / "summary.csv"
    if not source.exists():
        return []
    rows = read_csv(source)
    w_values = sorted({_number(row, "w_over_T") for row in rows})
    columns = min(2, len(w_values))
    row_count = math.ceil(len(w_values) / columns)
    cmap = ListedColormap([COLORS[0], COLORS[1], COLORS[2]])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(7.0, 2.7 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, w_value in zip(axes.flat, w_values):
        xs, ys, matrix, disagreements = _phase_matrix(rows, w_value)
        ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(min(xs), max(xs), min(ys), max(ys)),
            interpolation="nearest",
        )
        theory_x = np.linspace(min(xs), max(xs), 200)
        ax.plot(theory_x, (1.0 - theory_x) / 2.0, color="black", lw=0.9)
        ax.plot(theory_x, (2.0 - theory_x) / 3.0, color="black", lw=0.9, ls="--")
        if disagreements:
            ax.scatter(
                [point[0] for point in disagreements],
                [point[1] for point in disagreements],
                marker="x",
                s=8,
                linewidths=0.5,
                color="black",
            )
        ax.set_title(rf"$w/T={w_value:g}$", fontsize=8)
        ax.set(xlabel=r"$\alpha_t$", ylabel=r"$\alpha_l$")
    for ax in axes.flat[len(w_values) :]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=cmap(index), label=label)
        for index, label in enumerate(("Honest", "Solo withhold", "Tolerate"))
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="x",
            ls="",
            color="black",
            markersize=4,
            label="Simulation disagrees",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, root / "figures", "figure_B1_tolerance_phase")

    selected_w = _nearest(w_values, 1.0)
    alpha_ts = sorted({_number(row, "alpha_t") for row in rows})
    requested = [_nearest(alpha_ts, value) for value in (0.10, 0.20, 0.30)]
    requested = list(dict.fromkeys(requested))
    fig, axes = plt.subplots(
        1, len(requested), figsize=(7.0, 2.35), squeeze=False, sharey=True
    )
    regime_style = {
        "TOLERATE": (COLORS[2], "o", "Tolerate"),
        "EXPEL_AND_SOLO_WITHHOLD": (COLORS[1], "s", "Solo withhold"),
        "HONEST": (COLORS[0], "^", "Honest"),
    }
    for ax, alpha_t in zip(axes.flat, requested):
        subset = [
            row
            for row in rows
            if math.isclose(_number(row, "w_over_T"), selected_w)
            and math.isclose(_number(row, "alpha_t"), alpha_t)
        ]
        for regime, (color, marker, label) in regime_style.items():
            series = sorted(
                (row for row in subset if row["regime"] == regime),
                key=lambda row: _number(row, "alpha_l"),
            )
            x = [_number(row, "alpha_l") for row in series]
            ax.plot(
                x,
                [_number(row, "loyal_share_theory") for row in series],
                color=color,
                lw=1.0,
            )
            ax.errorbar(
                x,
                [_number(row, "loyal_share_sim_mean") for row in series],
                yerr=[_number(row, "loyal_share_sim_ci95") for row in series],
                fmt=marker,
                ms=2.5,
                capsize=1.5,
                color=color,
                label=label,
            )
        ax.set_title(rf"$\alpha_t={alpha_t:.2f}$", fontsize=8)
        ax.set(xlabel=r"$\alpha_l$", ylabel=r"Loyal canonical share $\rho$")
        ax.grid(axis="y", color="0.92", lw=0.5)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, root / "figures", "figure_B2_tolerance_shares")
    return [
        root / "figures" / "figure_B1_tolerance_phase.pdf",
        root / "figures" / "figure_B2_tolerance_shares.pdf",
    ]


def plot_stubborn(root: Path, alpha: float, alpha_t: float, k: int) -> list[Path]:
    summary_path = root / "aggregate" / "summary.csv"
    threshold_path = root / "aggregate" / "thresholds.csv"
    if not summary_path.exists() or not threshold_path.exists():
        return []
    rows = read_csv(summary_path)
    thresholds = read_csv(threshold_path)
    selected = sorted(
        _select_point(rows, {"alpha": alpha, "alpha_t": alpha_t, "k": float(k)}),
        key=lambda row: int(float(row["d"])),
    )
    x = np.array([_number(row, "d") for row in selected])
    traitor_share = _number(selected[0], "alpha_t")
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    for prefix, color, marker, label in (
        ("u_cartel", COLORS[2], "o", "Follow cartel"),
        ("u_honest", COLORS[1], "s", "Mine honestly"),
    ):
        ax.plot(
            x,
            [_number(row, f"{prefix}_theory") / traitor_share for row in selected],
            color=color,
            lw=1.1,
        )
        ax.errorbar(
            x,
            [_number(row, f"{prefix}_sim_mean") / traitor_share for row in selected],
            yerr=[
                _number(row, f"{prefix}_sim_ci95") / traitor_share for row in selected
            ],
            fmt=marker,
            ms=3,
            capsize=2,
            color=color,
            label=label,
        )
    ax.set(
        xlabel=r"Cartel deficit $d$",
        ylabel=r"Conditional focal-block utility $U/\alpha_t$",
    )
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(frameon=False)
    _save(fig, root / "figures", "figure_C1_stubborn_utilities")

    k_values = sorted({int(float(row["k"])) for row in thresholds})
    chosen_k = min(k_values, key=lambda value: abs(value - k))
    fixed_alphas = [
        value
        for value in (0.60, 0.65, 0.70, 0.75)
        if any(math.isclose(_number(row, "alpha"), value) for row in thresholds)
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    for index, alpha_value in enumerate(fixed_alphas):
        series = sorted(
            (
                row
                for row in thresholds
                if math.isclose(_number(row, "alpha"), alpha_value)
                and int(float(row["k"])) == chosen_k
            ),
            key=lambda row: _number(row, "alpha_l"),
        )
        ax.plot(
            [_number(row, "alpha_l") for row in series],
            [_number(row, "B_lose_theory") for row in series],
            color=COLORS[index],
            lw=1.0,
        )
        ax.scatter(
            [_number(row, "alpha_l") for row in series],
            [_number(row, "B_lose_simulation") for row in series],
            color=COLORS[index],
            s=10,
            facecolors="none",
            label=rf"$\alpha={alpha_value:.2f}$",
        )
    ax.set(xlabel=r"Loyal hashrate $\alpha_l$", ylabel=r"$B_{\rm lose}$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(frameon=False)
    _save(fig, root / "figures", "figure_C2_Blose_vs_loyal")

    subset = _select_point(rows, {"alpha": alpha, "k": float(k)})
    xs = sorted({_number(row, "alpha_t") for row in subset})
    ys = sorted({int(float(row["d"])) for row in subset})
    matrix = np.full((len(ys), len(xs)), np.nan)
    for row in subset:
        matrix[ys.index(int(float(row["d"]))), xs.index(_number(row, "alpha_t"))] = (
            1.0 if row["best_response_simulation"] == "FOLLOW" else 0.0
        )
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    response_cmap = ListedColormap([COLORS[1], COLORS[2]])
    image = ax.imshow(
        matrix, origin="lower", aspect="auto", cmap=response_cmap, vmin=0, vmax=1
    )
    ax.set_xticks(
        range(len(xs)), [f"{value:.2f}" for value in xs], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(ys)), ys)
    ax.set(xlabel=r"$\alpha_t$", ylabel=r"Deficit $d$")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0.25, 0.75])
    colorbar.ax.set_yticklabels(["HONEST", "FOLLOW"])
    _save(fig, root / "figures", "figure_C3_follow_region")

    target_alpha = _nearest(
        sorted({_number(row, "alpha") for row in thresholds}), alpha
    )
    target_t = _nearest(
        sorted(
            {
                _number(row, "alpha_t")
                for row in thresholds
                if math.isclose(_number(row, "alpha"), target_alpha)
            }
        ),
        alpha_t,
    )
    series = sorted(
        (
            row
            for row in thresholds
            if math.isclose(_number(row, "alpha"), target_alpha)
            and math.isclose(_number(row, "alpha_t"), target_t)
        ),
        key=lambda row: int(float(row["k"])),
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    ax.plot(
        [_number(row, "k") for row in series],
        [_number(row, "B_lose_theory") for row in series],
        color=COLORS[0],
        lw=1.1,
        label="Theory",
    )
    ax.scatter(
        [_number(row, "k") for row in series],
        [_number(row, "B_lose_simulation") for row in series],
        color=COLORS[1],
        marker="o",
        s=14,
        facecolors="none",
        label="Simulation",
    )
    ax.set(xlabel=r"Stubborn threshold $k$", ylabel=r"$B_{\rm lose}$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(frameon=False)
    _save(fig, root / "figures", "figure_C4_Blose_vs_k")
    return [
        root / "figures" / f"figure_C{index}_{name}.pdf"
        for index, name in (
            (1, "stubborn_utilities"),
            (2, "Blose_vs_loyal"),
            (3, "follow_region"),
            (4, "Blose_vs_k"),
        )
    ]


def plot_exact(root: Path) -> list[Path]:
    source = root / "aggregate" / "thresholds.csv"
    if not source.exists():
        return []
    rows = read_csv(source)
    x = np.array([_number(row, "B_lose_theory_m2") for row in rows])
    y = np.array([_number(row, "B_lose_exact_simulation") for row in rows])
    limit = max(1.0, float(np.nanmax([*x, *y])))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(
        [0, limit], [0, limit], color="0.5", lw=0.8, ls="--", label="Exact agreement"
    )
    ax.scatter(x, y, s=12, color=COLORS[0], alpha=0.7)
    ax.set(
        xlabel=r"$B_{\rm lose}$, theory ($m=2$)",
        ylabel=r"$B_{\rm lose}$, exact continuation",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.93", lw=0.5)
    ax.legend(frameon=False)
    _save(fig, root / "figures", "figure_D1_exact_policy_robustness")
    return [root / "figures" / "figure_D1_exact_policy_robustness.pdf"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--only", choices=("all", "A", "B", "C", "D"), default="all")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--alpha-t", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args(argv)
    _style()
    jobs: list[tuple[str, Callable[[], list[Path]]]] = [
        (
            "A",
            lambda: plot_disclosure(
                args.results_root / "disclosure_threshold", args.alpha, args.alpha_t
            ),
        ),
        ("B", lambda: plot_tolerance(args.results_root / "tolerance_region")),
        (
            "C",
            lambda: plot_stubborn(
                args.results_root / "stubborn_follow_threshold",
                args.alpha,
                args.alpha_t,
                args.k,
            ),
        ),
        ("D", lambda: plot_exact(args.results_root / "exact_policy_robustness")),
    ]
    outputs: list[Path] = []
    for label, function in jobs:
        if args.only in ("all", label):
            outputs.extend(function())
    if not outputs:
        raise FileNotFoundError(
            "no matching aggregate CSV files; run experiments first"
        )
    print(f"generated {len(outputs)} PDF figures (and PNG previews)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
