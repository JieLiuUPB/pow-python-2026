"""Canonical blocks over n difficulty rounds (paper Fig. 5, 6).

A round is 2016 T of wall-clock time, so with honest mining a miner with
hashrate a earns a * n * 2016 canonical blocks in n rounds. The block increase
ratio divides the canonical blocks actually earned by that baseline.
Difficulty adjusts every 2016 counted blocks:
- canonical DAA counts the canonical chain: OCW (w = 10T) vs selfish mining;
- public DAA counts every published block, orphans included: OCW attacker
  and honest miners.
Lines are the closed forms of the paper's Appendix B, dots the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.chain import EPOCH, Net
from ocw.theory import q, ratio, rho

ALPHAS = (0.65, 0.75)
ROUNDS = range(1, 11)
W = 10.0  # withholding window, in units of T


def selfish(rng, alpha, horizon):
    """Eyal–Sirer selfish mining with alpha > 1/2 under the canonical DAA.
    The private chain always stays ahead, so every attacker block and no
    honest block becomes canonical: the DAA sees attacker blocks only.
    Returns the attacker blocks mined by `horizon`."""
    t = start = 0.0
    D, a = 1.0, 0
    while (t := t + rng.expovariate(alpha / D)) <= horizon:
        a += 1
        if a % EPOCH == 0:
            D *= EPOCH / (t - start)
            start = t
    return a


def simulate(rng, strategy, daa, alpha, n):
    if strategy == "SM":
        return {"attacker": selfish(rng, alpha, n * EPOCH) / (alpha * n * EPOCH), "honest": 0.0}
    net = Net(rng, {"A": alpha, "H": 1 - alpha}, cartel={"A"}, w=W, daa=daa)
    while net.step(until=n * EPOCH):
        pass
    miners = [net.miner[b] for b in net.chain()]
    return {"attacker": miners.count("A") / (alpha * n * EPOCH), "honest": miners.count("H") / ((1 - alpha) * n * EPOCH)}


def run():
    runs = [("OCW", "canonical"), ("SM", "canonical"), ("OCW", "public")]
    return monte_carlo(simulate, [dict(strategy=s, daa=d, alpha=a, n=n) for s, d in runs for a in ALPHAS for n in ROUNDS])


def theory(strategy, daa, alpha, who, n):
    if strategy == "SM":  # all attacker blocks and nothing else is canonical
        return ratio(n, 1, alpha, alpha, alpha)
    share, hashrate = rho(alpha, W), alpha
    if who == "honest":
        share, hashrate = 1 - share, 1 - alpha
    return ratio(n, share, hashrate, *q(alpha, W, daa))


def curve(ax, df, strategy, daa, alpha, who, color, label):
    d = df[(df.strategy == strategy) & (df.daa == daa) & (df.alpha == alpha)]
    n = np.linspace(1, max(ROUNDS), 200)
    ax.plot(n, [theory(strategy, daa, alpha, who, k) for k in n], color=color, lw=0.8)
    ax.errorbar(d.n, d[who], d[f"{who}_ci"], fmt="o", ms=2.5, color=color, label=label)


def panels(titles):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, title in zip(axes, titles):
        ax.axhline(1, color="0.5", lw=0.7)
        ax.set(title=title, xlabel="Round $n$")
    axes[0].set_ylabel("Block increase ratio")
    return fig, axes


def plot(df):
    canonical, axes = panels([rf"$\alpha={a}$" for a in ALPHAS])
    for ax, alpha in zip(axes, ALPHAS):
        for color, strategy in zip(COLORS, ("OCW", "SM")):
            curve(ax, df, strategy, "canonical", alpha, "attacker", color, strategy)
    axes[0].legend()
    canonical.tight_layout()

    public, axes = panels(("Attacker (OCW)", "Honest miners"))
    for ax, who in zip(axes, ("attacker", "honest")):
        for color, alpha in zip(COLORS, ALPHAS):
            curve(ax, df, "OCW", "public", alpha, who, color, rf"$\alpha={alpha}$")
    axes[0].legend()
    public.tight_layout()
    return {"rounds_canonical": canonical, "rounds_public": public}


if __name__ == "__main__":
    main(__file__, run, plot)
