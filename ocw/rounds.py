"""Canonical blocks gained over n difficulty rounds, OCW (w = 10T) vs selfish mining.

A round is 2016 T of wall-clock time. Difficulty adjusts every 2016 counted
blocks, counting either the canonical chain or every published block
(orphans included). The ratio compares a miner's canonical blocks after n
rounds with its fair share: hashrate * n * 2016.
"""

import matplotlib.pyplot as plt

from harness import COLORS, main, monte_carlo
from ocw.chain import EPOCH, Net

ALPHAS = (0.65, 0.75)
ROUNDS = range(1, 11)


def majority_selfish(rng, alpha, daa, horizon):
    """Selfish mining with alpha > 1/2: the attacker publishes its private chain
    in 2016-block batches, so all its blocks become canonical and all honest
    blocks are orphaned. Canonical DAA therefore counts attacker blocks only,
    public DAA every mined block. Returns attacker blocks mined by `horizon`."""
    t = start = 0.0
    D, a, counted = 1.0, 0, 0
    while (t := t + rng.expovariate(1 / D)) <= horizon:
        attacker = rng.random() < alpha
        a += attacker
        if attacker or daa == "public":
            counted += 1
            if counted % EPOCH == 0:
                D *= EPOCH / (t - start)
                start = t
    return a


def simulate(rng, strategy, daa, alpha, n):
    if strategy == "SM":
        return {"attacker": majority_selfish(rng, alpha, daa, n * EPOCH) / (alpha * n * EPOCH), "honest": 0.0}
    net = Net(rng, {"A": alpha, "H": 1 - alpha}, cartel={"A"}, daa=daa)
    while net.step(until=n * EPOCH):
        pass
    miners = [net.miner[b] for b in net.chain()]
    return {"attacker": miners.count("A") / (alpha * n * EPOCH), "honest": miners.count("H") / ((1 - alpha) * n * EPOCH)}


def run():
    points = [dict(strategy=s, daa=d, alpha=a, n=n) for s in ("OCW", "SM") for d in ("canonical", "public")
              for a in ALPHAS for n in ROUNDS]
    return monte_carlo(simulate, points)


def plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True, sharey=True)
    series = (("OCW", "attacker", "OCW attacker"), ("OCW", "honest", "Honest miners vs OCW"), ("SM", "attacker", "SM attacker"))
    for row, daa in zip(axes, ("canonical", "public")):
        for ax, alpha in zip(row, ALPHAS):
            for color, (strategy, who, label) in zip(COLORS, series):
                d = df[(df.daa == daa) & (df.alpha == alpha) & (df.strategy == strategy)]
                ax.errorbar(d.n, d[who], d[f"{who}_ci"], fmt="o-", ms=2.5, lw=1, color=color, label=label)
            ax.axhline(1, color="0.5", lw=0.7)
            ax.set_title(rf"{daa.capitalize()} DAA, $\alpha={alpha}$")
    for ax in axes[1]:
        ax.set_xlabel("Round $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Canonical blocks / fair share")
    axes[0, 0].legend()
    fig.tight_layout()
    return {"rounds": fig}


if __name__ == "__main__":
    main(__file__, run, plot)
