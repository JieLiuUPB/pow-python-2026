"""k-capped selfish mining (k = 2), without difficulty adjustment.

The attacker keeps at most k private blocks. Whenever honest miners take the
lead, it publishes its oldest private block to tie again; when a new block
would exceed the cap, it publishes the oldest one and wins the tie. With
nothing left to publish, the honest branch wins. Each run mines until 2016
blocks are final, then settles the open fork.
"""

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo

K = 2
ALPHAS = np.arange(0.55, 0.96, 0.05).round(2).tolist()
BLOCKS = 2016


def simulate(rng, alpha):
    a = h = orphans = 0
    tie = hidden = 0  # length of each of two tied public branches; private blocks
    while a + h < BLOCKS or tie:
        if rng.random() < alpha:
            hidden += 1
            if hidden > K:  # publish the oldest private block: the attacker branch wins
                hidden -= 1
                a, orphans, tie = a + tie + 1, orphans + tie, 0
        elif hidden:  # publish the oldest private block to tie again
            hidden -= 1
            tie += 1
        else:  # the honest branch wins
            h, orphans, tie = h + tie + 1, orphans + tie, 0
    a += hidden
    return {"attacker": a / (a + h), "honest": h / (a + h), "orphans": orphans / (a + h + orphans)}


def run():
    return monte_carlo(simulate, [dict(alpha=a) for a in ALPHAS])


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for color, who, fair in ((COLORS[0], "attacker", df.alpha), (COLORS[1], "honest", 1 - df.alpha)):
        axes[0].errorbar(df.alpha, df[who], df[f"{who}_ci"], fmt="o-", ms=3, lw=1, color=color, label=who.capitalize())
        axes[0].plot(df.alpha, fair, "--", color=color, lw=0.8)
    axes[0].set(xlabel=r"Attacker hashrate $\alpha$", ylabel="Share of canonical blocks",
                title=f"$k={K}$; dashed: hashrate")
    axes[0].legend()
    axes[1].errorbar(df.alpha, df.orphans, df.orphans_ci, fmt="o-", ms=3, lw=1, color=COLORS[2])
    axes[1].set(xlabel=r"Attacker hashrate $\alpha$", ylabel="Orphan rate")
    fig.tight_layout()
    return {"capped": fig}


if __name__ == "__main__":
    main(__file__, run, plot)
