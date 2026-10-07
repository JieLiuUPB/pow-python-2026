"""Attacker share of canonical blocks and orphan rate: OCW vs selfish mining.

Each run mines until the canonical chain holds 2016 blocks, without
difficulty adjustment.
"""

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.chain import EPOCH, Net, selfish
from ocw.theory import ocw_orphans, ocw_share, sm_orphans, sm_share

ALPHAS = np.arange(0.30, 0.96, 0.05).round(2).tolist()
WINDOWS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)  # OCW release window w, in units of T


def simulate(rng, strategy, alpha, w=None):
    if strategy == "SM":
        a, h, orphans, _ = selfish(rng, alpha, EPOCH)
        return {"share": a / (a + h), "orphans": orphans / (a + h + orphans)}
    net = Net(rng, {"A": alpha, "H": 1 - alpha}, cartel={"A"}, w=w)
    while net.height[net.tip] < EPOCH:
        net.step()
    chain = net.chain()
    published = len(net.miner) - 1
    return {"share": sum(net.miner[b] == "A" for b in chain) / len(chain), "orphans": 1 - len(chain) / published}


def run():
    points = [dict(strategy="OCW", alpha=a, w=w) for a in ALPHAS for w in WINDOWS]
    points += [dict(strategy="SM", alpha=a) for a in ALPHAS]
    return monte_carlo(simulate, points)


def plot(df):
    ocw, sm = df[df.strategy == "OCW"], df[df.strategy == "SM"]
    x = np.linspace(min(ALPHAS), max(ALPHAS), 200)
    by_alpha, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, metric, ocw_theory, sm_theory in ((axes[0], "share", ocw_share, sm_share),
                                              (axes[1], "orphans", ocw_orphans, sm_orphans)):
        for color, w in zip(COLORS, (0.5, 1.0, 10.0)):
            d = ocw[ocw.w == w]
            ax.plot(x, [ocw_theory(a, w) for a in x], color=color, lw=0.8)
            ax.errorbar(d.alpha, d[metric], d[f"{metric}_ci"], fmt="o", ms=2.5, color=color, label=f"OCW, $w={w:g}T$")
        ax.plot(x[x < 0.5], [sm_theory(a) for a in x[x < 0.5]], color=COLORS[3], lw=0.8)
        ax.errorbar(sm.alpha, sm[metric], sm[f"{metric}_ci"], fmt="s", ms=2.5, color=COLORS[3], label="Selfish mining")
        ax.set(xlabel=r"Attacker hashrate $\alpha$")
    axes[0].plot(x, x, "k--", lw=0.8, label=r"Honest ($\alpha$)")
    axes[0].set(ylabel="Attacker share of canonical blocks", title="Lines: theory, dots: simulation")
    axes[1].set(ylabel="Orphan rate")
    axes[0].legend()
    by_alpha.tight_layout()

    by_window, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    x = np.linspace(0, max(WINDOWS), 200)
    for ax, metric, theory in ((axes[0], "share", ocw_share), (axes[1], "orphans", ocw_orphans)):
        for color, a in zip(COLORS, (0.65, 0.75)):
            d = ocw[ocw.alpha == a]
            ax.plot(x, [theory(a, w) for w in x], color=color, lw=0.8)
            ax.errorbar(d.w, d[metric], d[f"{metric}_ci"], fmt="o", ms=2.5, color=color, label=rf"$\alpha={a}$")
        ax.set(xlabel=r"Release window $w/T$")
    axes[0].set(ylabel="Attacker share of canonical blocks", title="OCW; lines: theory, dots: simulation")
    axes[1].set(ylabel="Orphan rate")
    axes[0].legend()
    by_window.tight_layout()
    return {"share_alpha": by_alpha, "share_window": by_window}


if __name__ == "__main__":
    main(__file__, run, plot)
