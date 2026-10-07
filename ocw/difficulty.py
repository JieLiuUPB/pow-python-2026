"""Difficulty after the first adjustment: OCW (w = 10T) vs selfish mining.

Starting from D = 1, each run mines until 2016 blocks are on the canonical
chain and applies the canonical-chain rule once:
D_new = 2016 / (time in T until the 2016th canonical block).
"""

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.chain import EPOCH, Net, selfish

ALPHAS = np.arange(0.55, 0.96, 0.05).round(2).tolist()


def simulate(rng, strategy, alpha):
    if strategy == "SM":
        return {"difficulty": EPOCH / selfish(rng, alpha, EPOCH)[3]}
    net = Net(rng, {"A": alpha, "H": 1 - alpha}, cartel={"A"})
    while net.height[net.tip] < EPOCH:
        net.step()
    return {"difficulty": EPOCH / net.time[net.chain()[EPOCH - 1]]}


def run():
    return monte_carlo(simulate, [dict(strategy=s, alpha=a) for s in ("OCW", "SM") for a in ALPHAS])


def plot(df):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for color, (strategy, label) in zip(COLORS, (("OCW", "OCW, $w=10T$"), ("SM", "Selfish mining"))):
        d = df[df.strategy == strategy]
        ax.errorbar(d.alpha, d.difficulty, d.difficulty_ci, fmt="o-", ms=3, lw=1, color=color, label=label)
    ax.axhline(1, color="0.5", lw=0.7)
    ax.set(xlabel=r"Attacker hashrate $\alpha$", ylabel=r"New difficulty $D_{\rm new}$")
    ax.legend()
    return {"difficulty": fig}


if __name__ == "__main__":
    main(__file__, run, plot)
