"""Plot a small analytical expression used for quick formula checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def analytical_term(p: float) -> float:
    """Return p^2(1-p)/(1+p)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return p**2 * (1.0 - p) / (1.0 + p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/analytical_term.png"),
        help="output PNG path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    p_values = np.linspace(0.51, 0.95, 200)
    values = np.array([analytical_term(float(p)) for p in p_values])

    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    ax.plot(p_values, values, linewidth=2)
    ax.set(xlabel="p", ylabel=r"$p^2(1-p)/(1+p)$", title="Analytical Check")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"saved figure to {args.output}")


if __name__ == "__main__":
    main()
