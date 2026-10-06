"""Plot normalized canonical attacker/honest block counts for scenario 4."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pow-python-matplotlib-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional plotting style
    scienceplots = None


DEFAULT_INPUT = Path("results/scenario4_public_daa_by_time/checkpoints.csv")
DEFAULT_HONEST_OUTPUT = Path("figures/s4_honest_fixedtime_ratio")
DEFAULT_ATTACKER_OUTPUT = Path("figures/s4_attacker_fixedtime_ratio")
DEFAULT_COMBINED_OUTPUT = Path("figures/s4_attacker_honest_fixedtime_ratio")
P_VALUES = (0.65, 0.75)
PALETTE = ("#155eef", "#0f766e", "#b42318", "#7a5af8", "#dd6b20")
EPS = 1e-12


def load_ratio_rows(
    path: Path,
    p_values: Sequence[float] = P_VALUES,
    *,
    block_field: str = "H_blocks_canonical",
) -> List[Dict[str, float]]:
    """Read checkpoints and calculate the selected canonical-block ratio."""
    if block_field not in {"A_blocks_canonical", "H_blocks_canonical"}:
        raise ValueError(f"Unsupported canonical-block column: {block_field}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {"p", "n", "t_checkpoint", block_field}
        missing_fields = required_fields.difference(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required columns in {path}: {missing}")

        rows: List[Dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                p = float(row["p"])
                n = int(row["n"])
                t_checkpoint = float(row["t_checkpoint"])
                canonical_blocks = float(row[block_field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value in {path} at line {line_number}"
                ) from exc

            selected_p = next(
                (target for target in p_values if abs(p - target) < EPS), None
            )
            if selected_p is None:
                continue
            hash_share = (
                selected_p if block_field == "A_blocks_canonical" else 1.0 - selected_p
            )
            normalization = t_checkpoint * hash_share / 10.0
            if normalization <= 0.0:
                raise ValueError(
                    f"Non-positive normalization in {path} at line {line_number}"
                )
            rows.append(
                {
                    "p": selected_p,
                    "n": n,
                    "t_checkpoint": t_checkpoint,
                    block_field: canonical_blocks,
                    "ratio": canonical_blocks / normalization,
                }
            )

    found_p = {row["p"] for row in rows}
    missing_p = [p for p in p_values if p not in found_p]
    if missing_p:
        missing = ", ".join(f"{p:.2f}" for p in missing_p)
        raise ValueError(f"No checkpoint rows found for p={missing} in {path}")
    return rows


def summarize_ratio_rows(
    rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Return the mean and sample standard deviation for each (p, n)."""
    grouped: DefaultDict[tuple[float, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["p"]), int(row["n"]))].append(float(row["ratio"]))

    summary_rows: List[Dict[str, float]] = []
    for (p, n), ratios in sorted(grouped.items()):
        summary_rows.append(
            {
                "p": p,
                "n": n,
                "metric_mean": mean(ratios),
                "metric_std": stdev(ratios) if len(ratios) > 1 else 0.0,
                "runs": len(ratios),
            }
        )
    return summary_rows


def apply_plot_theme() -> None:
    if scienceplots is not None:
        try:
            plt.style.use(["science", "ieee", "no-latex"])
        except Exception:
            plt.style.use(["science", "ieee"])
    plt.rcParams["text.usetex"] = False


def draw_ratio_axes(
    ax: plt.Axes,
    summary_rows: Sequence[Dict[str, float]],
    *,
    title: str,
) -> None:
    """Draw both alpha curves on one set of scenario-4-style axes."""
    p_values = sorted({float(row["p"]) for row in summary_rows})
    n_values = sorted({int(row["n"]) for row in summary_rows})

    for index, p in enumerate(p_values):
        rows = sorted(
            (row for row in summary_rows if abs(float(row["p"]) - p) < EPS),
            key=lambda row: int(row["n"]),
        )
        x = np.array([int(row["n"]) for row in rows], dtype=int)
        y = np.array([float(row["metric_mean"]) for row in rows], dtype=float)
        yerr = np.array([float(row["metric_std"]) for row in rows], dtype=float)
        color = PALETTE[index % len(PALETTE)]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2.0,
            elinewidth=1.4,
            color=color,
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=1.4,
            label=rf"$\alpha$={p:.2f}",
        )
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.08)

    ax.axhline(
        1.0, linestyle="--", color="#344054", linewidth=1.6, label="baseline = 1"
    )
    ax.set_xticks(n_values)
    ax.set_xlabel("Round n", fontsize=18)
    ax.set_ylabel("Block Increase Ratio", fontsize=18)
    ax.set_ylim(bottom=0.30)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=19.5)
    ax.legend(loc="lower right", ncols=2, fontsize=15)
    ax.margins(x=0.03)
    ax.grid(True, linestyle="--", alpha=0.6)


def save_figure(output: Path, fig: plt.Figure) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")


def plot_ratio(
    summary_rows: Sequence[Dict[str, float]],
    output: Path,
    *,
    title: str,
) -> None:
    """Draw one canonical-block ratio figure."""
    apply_plot_theme()
    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    draw_ratio_axes(ax, summary_rows, title=title)
    fig.tight_layout()
    save_figure(output, fig)
    plt.close(fig)


def plot_combined_ratios(
    attacker_rows: Sequence[Dict[str, float]],
    honest_rows: Sequence[Dict[str, float]],
    output: Path,
) -> None:
    """Place the attacker and honest ratio plots side by side."""
    apply_plot_theme()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.125),
        squeeze=False,
        sharey=True,
    )
    draw_ratio_axes(axes[0, 0], attacker_rows, title="Attacker")
    draw_ratio_axes(axes[0, 1], honest_rows, title="Honest")
    fig.suptitle("Canonical-Block Change with Orphan-Block-Aware DAA", fontsize=22.5)
    fig.tight_layout()
    save_figure(output, fig)
    plt.close(fig)


def write_summary(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["p", "n", "metric_mean", "metric_std", "runs"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_summary(
    path: Path,
    *,
    attacker_rows: Sequence[Dict[str, float]],
    honest_rows: Sequence[Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "block_owner",
            "p",
            "n",
            "metric_mean",
            "metric_std",
            "runs",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for block_owner, rows in (
            ("attacker", attacker_rows),
            ("honest", honest_rows),
        ):
            for row in rows:
                writer.writerow({"block_owner": block_owner, **row})


def normalize_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.suffix.lower() in {".png", ".pdf"}:
        return path.with_suffix("")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized canonical attacker/honest block counts against n "
            "for scenario 4, separately and side by side"
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"checkpoint CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--honest-output",
        "--output",
        dest="honest_output",
        default=str(DEFAULT_HONEST_OUTPUT),
        help=(
            "honest-block figure path without an extension; --output is kept "
            f"as an alias (default: {DEFAULT_HONEST_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--attacker-output",
        default=str(DEFAULT_ATTACKER_OUTPUT),
        help=(
            "attacker-block figure path without an extension "
            f"(default: {DEFAULT_ATTACKER_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--combined-output",
        default=str(DEFAULT_COMBINED_OUTPUT),
        help=(
            "side-by-side figure path without an extension "
            f"(default: {DEFAULT_COMBINED_OUTPUT})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    honest_output = normalize_output_path(args.honest_output)
    attacker_output = normalize_output_path(args.attacker_output)
    combined_output = normalize_output_path(args.combined_output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint CSV: {input_path}")

    honest_ratio_rows = load_ratio_rows(input_path, block_field="H_blocks_canonical")
    attacker_ratio_rows = load_ratio_rows(input_path, block_field="A_blocks_canonical")
    honest_summary = summarize_ratio_rows(honest_ratio_rows)
    attacker_summary = summarize_ratio_rows(attacker_ratio_rows)

    plot_ratio(
        honest_summary,
        honest_output,
        title="Honest-Block Ratio with Orphan-Block-Aware DAA",
    )
    plot_ratio(
        attacker_summary,
        attacker_output,
        title="Attacker-Block Ratio with Orphan-Block-Aware DAA",
    )
    plot_combined_ratios(attacker_summary, honest_summary, combined_output)

    honest_summary_path = honest_output.parent / f"{honest_output.name}_plot_data.csv"
    attacker_summary_path = (
        attacker_output.parent / f"{attacker_output.name}_plot_data.csv"
    )
    combined_summary_path = (
        combined_output.parent / f"{combined_output.name}_plot_data.csv"
    )
    write_summary(honest_summary_path, honest_summary)
    write_summary(attacker_summary_path, attacker_summary)
    write_combined_summary(
        combined_summary_path,
        attacker_rows=attacker_summary,
        honest_rows=honest_summary,
    )

    for output in (honest_output, attacker_output, combined_output):
        print(f"saved figures to {output.with_suffix('.png')} and .pdf")
    for summary_path in (
        honest_summary_path,
        attacker_summary_path,
        combined_summary_path,
    ):
        print(f"saved plot data to {summary_path}")


if __name__ == "__main__":
    main()
