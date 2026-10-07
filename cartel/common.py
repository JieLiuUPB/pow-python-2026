"""Monte Carlo runner, result files, and figure helpers shared by the experiments."""

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = 30  # independent runs per grid point; error bars are 95% CIs over runs
RESULTS = Path(__file__).parent / "results"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7, "legend.frameon": False, "pdf.fonttype": 42})


def cartel_grid(alphas):
    """(alpha, alpha_t) pairs with alpha_t = 0.05, 0.10, ... up to alpha - 0.05."""
    return [(a, round(t, 2)) for a in alphas for t in np.arange(0.05, a - 0.04, 0.05)]


def _run(task):
    kernel, point, run = task
    return kernel(random.Random(f"{point}/{run}"), **point)


def monte_carlo(kernel, points):
    """Call kernel(rng, **point) -> {metric: value} RUNS times per point, in
    parallel. Returns one row per point with each metric's mean and CI."""
    tasks = [(kernel, p, r) for p in points for r in range(RUNS)]
    with ProcessPoolExecutor() as pool:
        runs = pd.DataFrame(pool.map(_run, tasks, chunksize=RUNS))
    by_point = runs.groupby(np.arange(len(runs)) // RUNS)
    ci = 1.96 * by_point.std() / RUNS**0.5
    return pd.concat([pd.DataFrame(points), by_point.mean(), ci.add_suffix("_ci")], axis=1)


def main(name, run, plot):
    """`python <experiment>.py` simulates and plots; append `plot` to only replot."""
    csv = RESULTS / f"{name}.csv"
    if sys.argv[1:] != ["plot"]:
        RESULTS.mkdir(exist_ok=True)
        run().to_csv(csv, index=False, float_format="%.6g")
    plot(pd.read_csv(csv))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, table, text=None):
    """Integer heatmap of a pivot table, each cell labelled with `text` (default: its value)."""
    text = table if text is None else text
    ax.imshow(table, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), [f"{v:.2f}" for v in table.columns], rotation=90)
    ax.set_yticks(range(table.shape[0]), [f"{v:.2f}" for v in table.index])
    for (i, j), v in np.ndenumerate(text.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
