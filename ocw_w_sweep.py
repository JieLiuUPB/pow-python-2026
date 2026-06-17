"""
Standalone OCW withholding-window sweep experiment.

This script reuses the repository's existing OCW attack simulator from
`pow_chain_withhold.py`, but replaces the fixed withholding window with a
configurable value `w`. It sweeps over several normalized windows `w/T` for two
fixed attacker hashrates (alpha = 0.65 and 0.75), saves raw and aggregated CSV
outputs, and generates one paper-style figure with two horizontal subplots:

1. attacker relative canonical share vs. w/T
2. orphan rate vs. w/T

The simulation points use 95% confidence intervals computed as
`1.96 * sample_std / sqrt(n)`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from pow_chain_withhold import ChainWithholdSimulation

DEFAULT_ALPHA_LIST = (0.65, 0.75)
DEFAULT_W_OVER_T_LIST = (0.0, 0.25,0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_T = 10.0
DEFAULT_REPEATS = 50
DEFAULT_TARGET_BLOCKS = 2016
DEFAULT_BASE_SEED = 2026
DEFAULT_JOBS = 30


class VariableWindowOCWSimulation(ChainWithholdSimulation):
    """Reuse the existing OCW simulator while overriding only the window length."""

    def __init__(self, *, withholding_window: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.withholding_window = float(withholding_window)

    def _w_star(self) -> float:
        return self.withholding_window


def ocw_profit_theory(alpha: float, w_over_t: float) -> float:
    """Attacker relative canonical share under the OCW event model.

    Derived from the same event model implemented in `pow_chain_withhold.py`.
    It interpolates between immediate release (w=0, share=alpha) and the
    original fixed-window OCW limit (w/T -> inf).
    """
    s = 1.0 - math.exp(-w_over_t)
    return alpha*(1-s) + alpha * (3 - 2*alpha) * alpha * s


def ocw_orphan_theory(alpha: float, w_over_t: float) -> float:
    """Orphan rate under the OCW event model.

    Uses the repository's orphan-rate definition:
    orphan_published / total_published.
    """
    s = 1.0 - math.exp(-w_over_t)
    numerator = alpha * alpha * (1.0 - alpha) * s
    return numerator / (1.0 + numerator)


def ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def parse_float_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def derive_seed(base_seed: int, alpha: float, w_over_t: float, run_id: int) -> int:
    token = f"{base_seed}|ocw_w_sweep|{alpha:.8f}|{w_over_t:.8f}|{run_id}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def run_one(
    *,
    T: float,
    alpha: float,
    w_over_t: float,
    run_id: int,
    seed: int,
    target_blocks: int,
    max_events: Optional[int],
) -> Dict[str, Any]:
    w_value = w_over_t * T
    sim = VariableWindowOCWSimulation(
        T=T,
        p=alpha,
        seed=seed,
        target_blocks=target_blocks,
        run_id=run_id,
        max_events=max_events,
        withholding_window=w_value,
    )
    row = sim.run().to_dict()
    row["alpha"] = alpha
    row["w_over_T"] = w_over_t
    row["w"] = w_value
    row["profit_theory"] = ocw_profit_theory(alpha, w_over_t)
    row["orphan_theory"] = ocw_orphan_theory(alpha, w_over_t)
    return row


def build_tasks(
    alpha_list: Sequence[float],
    w_over_t_list: Sequence[float],
    *,
    T: float,
    repeats: int,
    base_seed: int,
    target_blocks: int,
    max_events: Optional[int],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for alpha in alpha_list:
        for w_over_t in w_over_t_list:
            for run_id in range(repeats):
                tasks.append(
                    {
                        "T": T,
                        "alpha": alpha,
                        "w_over_t": w_over_t,
                        "run_id": run_id,
                        "seed": derive_seed(base_seed, alpha, w_over_t, run_id),
                        "target_blocks": target_blocks,
                        "max_events": max_events,
                    }
                )
    return tasks


def _task_star(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return run_one(**kwargs)


def run_experiments(
    alpha_list: Sequence[float],
    w_over_t_list: Sequence[float],
    *,
    T: float,
    repeats: int,
    base_seed: int,
    target_blocks: int,
    jobs: int,
    max_events: Optional[int],
) -> List[Dict[str, Any]]:
    tasks = build_tasks(
        alpha_list,
        w_over_t_list,
        T=T,
        repeats=repeats,
        base_seed=base_seed,
        target_blocks=target_blocks,
        max_events=max_events,
    )
    if not tasks:
        return []

    effective_jobs = max(1, min(jobs, len(tasks)))
    rows: List[Dict[str, Any]] = []

    if effective_jobs == 1:
        for task in tasks:
            rows.append(_task_star(task))
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=effective_jobs
            ) as ex:
                rows = list(ex.map(_task_star, tasks))
        except (OSError, PermissionError):
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_jobs) as ex:
                rows = list(ex.map(_task_star, tasks))

    rows.sort(key=lambda row: (float(row["alpha"]), float(row["w_over_T"]), int(row["run_id"])))
    return rows


def summarize(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[float, float], List[Dict[str, Any]]] = {}
    for row in raw_rows:
        key = (float(row["alpha"]), float(row["w_over_T"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (alpha, w_over_t) in sorted(grouped):
        rows = grouped[(alpha, w_over_t)]
        profit_values = [float(row["A_share"]) for row in rows]
        orphan_values = [float(row["orphan_rate"]) for row in rows]
        w_value = float(rows[0]["w"])
        summary_rows.append(
            {
                "alpha": alpha,
                "w_over_T": w_over_t,
                "w": w_value,
                "runs": len(rows),
                "profit_mean": mean(profit_values),
                "profit_std": stdev(profit_values) if len(rows) > 1 else 0.0,
                "profit_ci95": ci95(profit_values),
                "profit_theory": ocw_profit_theory(alpha, w_over_t),
                "orphan_rate_mean": mean(orphan_values),
                "orphan_rate_std": stdev(orphan_values) if len(rows) > 1 else 0.0,
                "orphan_rate_ci95": ci95(orphan_values),
                "orphan_rate_theory": ocw_orphan_theory(alpha, w_over_t),
            }
        )
    return summary_rows


def build_dense_theory(alpha: float, max_w_over_t: float = 10.0) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, max_w_over_t, 400)
    return {
        "w_over_T": x,
        "profit": np.array([ocw_profit_theory(alpha, float(v)) for v in x], dtype=float),
        "orphan": np.array([ocw_orphan_theory(alpha, float(v)) for v in x], dtype=float),
    }


def plot_results(summary_rows: Sequence[Dict[str, Any]], output_base: Path) -> None:
    if not summary_rows:
        return

    alpha_list = sorted({float(row["alpha"]) for row in summary_rows})
    curves = {alpha: build_dense_theory(alpha) for alpha in alpha_list}
    style_map = {
        0.65: {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        0.75: {"color": "#d62728", "linestyle": "-", "marker": "s"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    profit_ax, orphan_ax = axes

    for alpha in alpha_list:
        rows = sorted(
            [row for row in summary_rows if abs(float(row["alpha"]) - alpha) < 1e-12],
            key=lambda row: float(row["w_over_T"]),
        )
        x = np.array([float(row["w_over_T"]) for row in rows], dtype=float)
        profit_mean = np.array([float(row["profit_mean"]) for row in rows], dtype=float)
        profit_ci = np.array([float(row["profit_ci95"]) for row in rows], dtype=float)
        orphan_mean = np.array([float(row["orphan_rate_mean"]) for row in rows], dtype=float)
        orphan_ci = np.array([float(row["orphan_rate_ci95"]) for row in rows], dtype=float)

        style = style_map.get(
            alpha,
            {"color": None, "linestyle": "-", "marker": "o"},
        )
        theory = curves[alpha]
        alpha_label = rf"$\alpha={alpha:.2f}$"

        profit_ax.plot(
            theory["w_over_T"],
            theory["profit"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=rf"Theory, {alpha_label}",
        )
        profit_ax.errorbar(
            x,
            profit_mean,
            yerr=profit_ci,
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            capsize=3,
            label=rf"Sim, {alpha_label}",
        )

        orphan_ax.plot(
            theory["w_over_T"],
            theory["orphan"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=rf"Theory, {alpha_label}",
        )
        orphan_ax.errorbar(
            x,
            orphan_mean,
            yerr=orphan_ci,
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            capsize=3,
            label=rf"Sim, {alpha_label}",
        )

    profit_ax.set_title("(a) Attacker Relative Canonical Share")
    profit_ax.set_xlabel(r"$w/T$")
    profit_ax.set_ylabel("Attacker relative canonical share")
    profit_ax.grid(True, linestyle=":", linewidth=0.8)
    profit_ax.set_xlim(-0.1, 10.1)

    orphan_ax.set_title("(b) Orphan Rate")
    orphan_ax.set_xlabel(r"$w/T$")
    orphan_ax.set_ylabel("Orphan rate")
    orphan_ax.grid(True, linestyle=":", linewidth=0.8)
    orphan_ax.set_xlim(-0.1, 10.1)

    profit_ax.legend(loc="lower right", fontsize=9, frameon=True)
    orphan_ax.legend(loc="lower right", fontsize=9, frameon=True)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_rows: Iterable[Dict[str, Any]]) -> None:
    print(
        f"{'alpha':>6}  {'w/T':>6}  {'profit_mean':>12}  {'profit_ci95':>12}  "
        f"{'orphan_mean':>12}  {'orphan_ci95':>12}"
    )
    for row in summary_rows:
        print(
            f"{float(row['alpha']):>6.2f}  {float(row['w_over_T']):>6.2f}  "
            f"{float(row['profit_mean']):>12.6f}  {float(row['profit_ci95']):>12.6f}  "
            f"{float(row['orphan_rate_mean']):>12.6f}  {float(row['orphan_rate_ci95']):>12.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the OCW withholding window and compare simulation to theory."
    )
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--runs", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--alpha-list", default=",".join(str(v) for v in DEFAULT_ALPHA_LIST))
    parser.add_argument(
        "--w-over-T-list",
        default=",".join(str(v) for v in DEFAULT_W_OVER_T_LIST),
    )
    parser.add_argument("--target-blocks", type=int, default=DEFAULT_TARGET_BLOCKS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--output-dir", default="results/ocw_w_sweep")
    parser.add_argument("--figure-name", default="ocw_w_sweep")
    parser.add_argument("--raw-name", default="ocw_w_sweep_raw.csv")
    parser.add_argument("--summary-name", default="ocw_w_sweep_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alpha_list = parse_float_list(args.alpha_list)
    w_over_t_list = parse_float_list(args.w_over_T_list)
    output_dir = Path(args.output_dir)

    print(
        f"ocw_w_sweep: T={args.T}, runs={args.runs}, target_blocks={args.target_blocks}, jobs={args.jobs}"
    )
    print(f"alpha_list = {alpha_list}")
    print(f"w_over_T_list = {w_over_t_list}")

    raw_rows = run_experiments(
        alpha_list,
        w_over_t_list,
        T=args.T,
        repeats=args.runs,
        base_seed=args.base_seed,
        target_blocks=args.target_blocks,
        jobs=args.jobs,
        max_events=args.max_events,
    )
    summary_rows = summarize(raw_rows)

    write_csv(output_dir / args.raw_name, raw_rows)
    write_csv(output_dir / args.summary_name, summary_rows)
    plot_results(summary_rows, output_dir / args.figure_name)

    print_summary(summary_rows)
    print(f"raw CSV: {output_dir / args.raw_name}")
    print(f"summary CSV: {output_dir / args.summary_name}")
    print(f"figure: {output_dir / (args.figure_name + '.pdf')}")
    print(f"figure: {output_dir / (args.figure_name + '.png')}")


if __name__ == "__main__":
    main()
