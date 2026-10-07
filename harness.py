"""Experiment harness shared by both projects: parallel Monte Carlo, result
files, and figure style.

An experiment is a module `<project>/<name>.py` with `run() -> DataFrame` and
`plot(df) -> {figure name: Figure}`, started from the repository root:

    python -m cartel.release         # simulate, save the CSV, plot
    python -m cartel.release plot    # replot from the saved CSV only

Results go to `<project>/results/`.
"""

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7, "legend.frameon": False, "pdf.fonttype": 42})


def _run(task):
    kernel, point, run = task
    return kernel(random.Random(f"{point}/{run}"), **point)


def monte_carlo(kernel, points, runs=30):
    """Call kernel(rng, **point) -> {metric: value} `runs` times per point, in
    parallel. Returns one row per point with each metric's mean and 95% CI
    half-width (column `<metric>_ci`)."""
    tasks = [(kernel, p, r) for p in points for r in range(runs)]
    with ProcessPoolExecutor() as pool:
        rows = pd.DataFrame(pool.map(_run, tasks, chunksize=runs))
    by_point = rows.groupby(np.arange(len(rows)) // runs)
    ci = 1.96 * by_point.std() / runs**0.5
    return pd.concat([pd.DataFrame(points), by_point.mean(), ci.add_suffix("_ci")], axis=1)


def main(file, run, plot):
    path = Path(file)
    out = path.parent / "results"
    csv = out / f"{path.stem}.csv"
    if sys.argv[1:] != ["plot"]:
        out.mkdir(exist_ok=True)
        run().to_csv(csv, index=False, float_format="%.6g")
    for name, fig in plot(pd.read_csv(csv)).items():
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def heatmap(ax, table, text=None):
    """Heatmap of a pivot table, each cell labelled with `text` (default: its value)."""
    text = table if text is None else text
    ax.imshow(table, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), [f"{v:.2f}" for v in table.columns], rotation=90)
    ax.set_yticks(range(table.shape[0]), [f"{v:.2f}" for v in table.index])
    for (i, j), v in np.ndenumerate(text.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
