"""Plot OCW and selfish-mining orphan-rate summaries together."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

DEFAULT_TBW_SUMMARY = Path("results/chain_withhold_summary.csv")
LEGACY_TBW_SUMMARY = Path("results/chain_withhold/summary.csv")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)


def theoretical_tbw_orphan_rate(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    return p * p * (1.0 - p) / (1.0 + p * p * (1.0 - p))


def theoretical_selfish_orphan_rate(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if math.isclose(p, 0.5, rel_tol=0.0, abs_tol=1e-12):
        return 0.5
    if p < 0.5:
        denominator = 1.0 - 4.0 * p * p + 2.0 * p * p * p
        return p * (1.0 - p) * (1.0 - p) / denominator
    # In the classical no-delay model, once the attacker has positive drift
    # (p > 0.5), the hidden lead grows and almost all honest blocks become
    # stale asymptotically, so the orphan rate tends to the honest hashrate.
    return 1.0 - p


def _read_first_float(
    row: Dict[str, str],
    field_names: Sequence[str],
    *,
    path: Path,
) -> float:
    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return float(value)
    joined = ", ".join(field_names)
    raise KeyError(f"Missing any of [{joined}] in {path}")


def load_tbw_summary(path: Path) -> List[Dict[str, float]]:
    rows = read_csv_rows(path)
    output: List[Dict[str, float]] = []
    for row in rows:
        p_value = float(row["p"])
        output.append(
            {
                "p": p_value,
                "sim_mean": _read_first_float(
                    row,
                    ("orphan_rate_mean", "orphan_rate_sim"),
                    path=path,
                ),
                "sim_std": _read_first_float(row, ("orphan_rate_std",), path=path),
                "theory": theoretical_tbw_orphan_rate(p_value),
            }
        )
    return sorted(output, key=lambda row: row["p"])


def load_selfish_summary(path: Path) -> List[Dict[str, float]]:
    rows = read_csv_rows(path)
    output: List[Dict[str, float]] = []
    for row in rows:
        p_value = float(row["p"])
        output.append(
            {
                "p": p_value,
                "sim_mean": float(row["mean_orphan_rate"]),
                "sim_std": float(row["std_orphan_rate"]),
                "theory": theoretical_selfish_orphan_rate(p_value),
            }
        )
    return sorted(output, key=lambda row: row["p"])


def build_plot_rows(
    tbw_rows: Sequence[Dict[str, float]],
    selfish_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    tbw_by_p = {row["p"]: row for row in tbw_rows}
    selfish_by_p = {row["p"]: row for row in selfish_rows}
    all_p = sorted(set(tbw_by_p) | set(selfish_by_p))

    rows: List[Dict[str, float]] = []
    for p in all_p:
        row: Dict[str, float] = {
            "p": p,
            "tbw_sim_mean": math.nan,
            "tbw_sim_std": math.nan,
            "tbw_theory": math.nan,
            "selfish_sim_mean": math.nan,
            "selfish_sim_std": math.nan,
            "selfish_theory": math.nan,
        }
        if p in tbw_by_p:
            row["tbw_sim_mean"] = tbw_by_p[p]["sim_mean"]
            row["tbw_sim_std"] = tbw_by_p[p]["sim_std"]
            row["tbw_theory"] = tbw_by_p[p]["theory"]
        if p in selfish_by_p:
            row["selfish_sim_mean"] = selfish_by_p[p]["sim_mean"]
            row["selfish_sim_std"] = selfish_by_p[p]["sim_std"]
            row["selfish_theory"] = selfish_by_p[p]["theory"]
        rows.append(row)
    return rows


def build_dense_curve(
    p_min: float,
    p_max: float,
    theory_fn: Callable[[float], float],
    num_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    p_grid = np.linspace(p_min, p_max, num_points, dtype=float)
    theory_values = np.array([theory_fn(float(p)) for p in p_grid], dtype=float)
    return p_grid, theory_values


def save_figure_bundle(base_path: Path, fig: plt.Figure) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight"
    )


def plot_comparison(
    tbw_rows: Sequence[Dict[str, float]],
    selfish_rows: Sequence[Dict[str, float]],
    output_path: Path,
    include_selfish_theory: bool,
) -> None:
    if not tbw_rows:
        raise ValueError("TBW summary is empty")
    if not selfish_rows:
        raise ValueError("selfish-mining summary is empty")

    try:
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use(["science", "ieee"])
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    marker = itertools.cycle(("+", "x", "s", "v", "o", "D", "^"))

    tbw_p = np.array([row["p"] for row in tbw_rows], dtype=float)
    tbw_sim = np.array([row["sim_mean"] for row in tbw_rows], dtype=float)
    tbw_std = np.array([row["sim_std"] for row in tbw_rows], dtype=float)

    selfish_p = np.array([row["p"] for row in selfish_rows], dtype=float)
    selfish_sim = np.array([row["sim_mean"] for row in selfish_rows], dtype=float)
    selfish_std = np.array([row["sim_std"] for row in selfish_rows], dtype=float)

    tbw_theory_p, tbw_theory_dense = build_dense_curve(
        float(np.min(tbw_p)),
        float(np.max(tbw_p)),
        theoretical_tbw_orphan_rate,
    )
    selfish_theory_p, selfish_theory_dense = build_dense_curve(
        float(np.min(selfish_p)),
        float(np.max(selfish_p)),
        theoretical_selfish_orphan_rate,
    )

    ax.errorbar(
        tbw_p,
        tbw_sim,
        yerr=tbw_std,
        marker=next(marker),
        markersize=3,
        markerfacecolor="none",
        capsize=4,
        linewidth=1.5,
        label="OCW sim",
    )
    ax.plot(
        tbw_theory_p,
        tbw_theory_dense,
        linestyle=":",
        linewidth=1.5,
        label="OCW theory",
    )
    ax.errorbar(
        selfish_p,
        selfish_sim,
        yerr=selfish_std,
        marker=next(marker),
        markersize=3,
        markerfacecolor="none",
        capsize=4,
        linewidth=1.5,
        label="SM sim",
    )
    if include_selfish_theory:
        ax.plot(
            selfish_theory_p,
            selfish_theory_dense,
            linestyle="--",
            linewidth=1.5,
            label="SM theory",
        )

    ax.set_xlabel(r"Attacker hashrate $\alpha$", fontsize=12)
    ax.set_ylabel("Orphan rate", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title("Orphan Rate Comparison")
    ax.legend(fontsize=10)
    ax.margins(x=0.02)

    all_values = [*tbw_sim, *tbw_theory_dense, *selfish_sim]
    all_errors = [*tbw_std, *selfish_std]
    if include_selfish_theory:
        all_values.extend(selfish_theory_dense.tolist())
    if all_values:
        lower = min(v - e for v, e in zip([*tbw_sim, *selfish_sim], all_errors))
        upper = max(all_values)
        ax.set_ylim(max(0.0, lower - 0.02), min(1.0, upper + 0.04))

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    save_figure_bundle(output_path, fig)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare orphan-rate summaries from TBW and selfish mining runs"
    )
    parser.add_argument(
        "--simulation-summary",
        default=str(DEFAULT_TBW_SUMMARY),
        help=(
            "chain-withhold summary CSV; default falls back to "
            "results/chain_withhold/summary.csv"
        ),
    )
    parser.add_argument(
        "--selfish-summary",
        default="results/selfish/summary.csv",
        help="summary.csv produced by pow_selfish.py",
    )
    parser.add_argument(
        "--output",
        default="figures/orphan_rate_comparison",
        help="output path without extension or with .png/.pdf extension",
    )
    parser.add_argument(
        "--plot-data",
        default="figures/orphan_rate_comparison_plot_data.csv",
        help="CSV path for the merged plot data",
    )
    parser.add_argument(
        "--skip-selfish-theory",
        action="store_true",
        help="omit the theoretical orphan-rate curve for selfish mining",
    )
    return parser.parse_args()


def normalize_output_path(raw: str) -> Path:
    path = Path(raw)
    if path.suffix.lower() in {".png", ".pdf"}:
        return path.with_suffix("")
    return path


def resolve_tbw_summary_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists() or path != DEFAULT_TBW_SUMMARY:
        return path
    if LEGACY_TBW_SUMMARY.exists():
        return LEGACY_TBW_SUMMARY
    return path


def main() -> None:
    args = parse_args()
    requested_simulation_summary = Path(args.simulation_summary)
    simulation_summary = resolve_tbw_summary_path(args.simulation_summary)
    selfish_summary = Path(args.selfish_summary)
    output_path = normalize_output_path(args.output)

    if not simulation_summary.exists():
        raise FileNotFoundError(f"Missing simulation summary: {simulation_summary}")
    if not selfish_summary.exists():
        raise FileNotFoundError(f"Missing selfish summary: {selfish_summary}")

    if simulation_summary != requested_simulation_summary:
        print(f"[info] using legacy chain-withhold summary: {simulation_summary}")

    tbw_rows = load_tbw_summary(simulation_summary)
    selfish_rows = load_selfish_summary(selfish_summary)
    plot_rows = build_plot_rows(tbw_rows, selfish_rows)

    plot_comparison(
        tbw_rows=tbw_rows,
        selfish_rows=selfish_rows,
        output_path=output_path,
        include_selfish_theory=not args.skip_selfish_theory,
    )
    write_csv_rows(Path(args.plot_data), plot_rows)

    print(f"saved figure to {output_path.with_suffix('.png')}")
    print(f"saved plot data to {args.plot_data}")


if __name__ == "__main__":
    main()
