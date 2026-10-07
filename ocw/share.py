"""Attacker share of canonical blocks and orphan rate under OCW (paper Fig. 3, 4).

Each run mines until the canonical chain holds 2016 blocks, without
difficulty adjustment, and is compared with the closed forms.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.chain import EPOCH, Net

ALPHAS = np.arange(0.30, 0.81, 0.05).round(2).tolist()
WINDOWS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)  # withholding window w, in units of T


def rho(alpha, w):
    """Attacker share of canonical blocks (Eq. 2). w = 0 is honest mining."""
    z = math.exp(-w)
    return alpha * z + alpha**2 * (1 - z) * (3 - 2 * alpha)


def eta(alpha, w):
    """Orphan rate: orphaned / published blocks (Eq. 25)."""
    x = alpha**2 * (1 - alpha) * (1 - math.exp(-w))
    return x / (1 + x)


def simulate(rng, alpha, w):
    net = Net(rng, {"A": alpha, "H": 1 - alpha}, cartel={"A"}, w=w)
    net.grow(EPOCH)
    chain = net.chain()
    published = len(net.miner) - 1
    return {"share": sum(net.miner[b] == "A" for b in chain) / len(chain), "orphans": 1 - len(chain) / published}


def run():
    return monte_carlo(simulate, [dict(alpha=a, w=w) for a in ALPHAS for w in WINDOWS])


def plot(df):
    by_alpha, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.linspace(min(ALPHAS), max(ALPHAS), 200)
    for color, w in zip(COLORS, (0.5, 1.0, 10.0)):
        d = df[df.w == w]
        ax.plot(x, [rho(a, w) for a in x], color=color, lw=0.8)
        ax.errorbar(d.alpha, d.share, d.share_ci, fmt="o", ms=2.5, color=color, label=f"$w={w:g}T$")
    ax.plot(x, x, "k--", lw=0.8, label=r"Honest ($\alpha$)")
    ax.set(xlabel=r"Attacker hashrate $\alpha$", ylabel="Attacker share of canonical blocks",
           title="Lines: theory, dots: simulation")
    ax.legend()

    by_window, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    x = np.linspace(0, max(WINDOWS), 200)
    for ax, metric, theory in ((axes[0], "share", rho), (axes[1], "orphans", eta)):
        for color, a in zip(COLORS, (0.65, 0.75)):
            d = df[df.alpha == a]
            ax.plot(x, [theory(a, w) for w in x], color=color, lw=0.8)
            ax.errorbar(d.w, d[metric], d[f"{metric}_ci"], fmt="o", ms=2.5, color=color, label=rf"$\alpha={a}$")
        ax.set(xlabel=r"Withholding window $w/T$")
    axes[0].set(ylabel="Attacker share of canonical blocks", title="Lines: theory, dots: simulation")
    axes[1].set(ylabel="Orphan rate")
    axes[0].legend()
    by_window.tight_layout()
    return {"share_alpha": by_alpha, "share_window": by_window}


if __name__ == "__main__":
    main(__file__, run, plot)
