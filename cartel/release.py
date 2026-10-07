"""§4: should a traitor who just mined a private block publish it or withhold it?

Simulates U_withhold (Eq. 12) and compares it with publishing (1 + alpha_t,
Eq. 4) to check the closed-form threshold ell* of Theorem 1.
"""

import matplotlib.pyplot as plt
import numpy as np

from cartel.theory import ell_star, omega, withhold_utility
from harness import COLORS, heatmap, main, monte_carlo

ALPHAS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
LEADS = range(1, 11)
TRIALS = 5_000


def simulate(rng, alpha, alpha_t, ell):
    """The traitor's block is safe if it mines again before honest miners erase
    the lead (bonus block too); otherwise it survives the public race w.p. omega."""
    w, total = omega(alpha), 0
    for _ in range(TRIALS):
        lead = ell
        while lead > 0:
            u = rng.random()
            if u < alpha_t:
                break
            lead += 1 if u < alpha else -1
        total += 2 if lead > 0 else rng.random() < w
    return {"withhold": total / TRIALS}


def run():
    points = [dict(alpha=a, alpha_t=t, ell=ell)
              for a in ALPHAS for t in np.arange(0.05, a - 0.04, 0.05).round(2).tolist() for ell in LEADS]
    df = monte_carlo(simulate, points)
    df["withhold_theory"] = [withhold_utility(a, t, ell) for a, t, ell in zip(df.alpha, df.alpha_t, df.ell)]
    df["ell_star"] = [ell_star(a, t) for a, t in zip(df.alpha, df.alpha_t)]
    return df


def plot(df):
    utility, ax = plt.subplots(figsize=(3.4, 2.5))
    for color, t in zip(COLORS, (0.05, 0.15, 0.25)):
        d = df[(df.alpha == 0.35) & (df.alpha_t == t)]
        ax.plot(d.ell, d.withhold_theory - 1 - t, color=color, lw=1)
        ax.errorbar(d.ell, d.withhold - 1 - t, d.withhold_ci, fmt="o", ms=3, capsize=2, color=color,
                    label=rf"$\alpha_t={t}$, $\ell^\star={d.ell_star.iloc[0]}$")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set(xlabel=r"Private lead $\ell$", ylabel=r"$U_{\rm withhold}-U_{\rm publish}$",
           title=r"$\alpha=0.35$; lines: theory, dots: simulation")
    ax.legend()

    threshold, ax = plt.subplots(figsize=(3.4, 2.6))
    heatmap(ax, df.pivot_table("ell_star", "alpha", "alpha_t"))
    ax.set(xlabel=r"Traitor hashrate $\alpha_t$", ylabel=r"Cartel hashrate $\alpha$",
           title=r"Release threshold $\ell^\star$")
    return {"release_utility": utility, "release_threshold": threshold}


if __name__ == "__main__":
    main(__file__, run, plot)
