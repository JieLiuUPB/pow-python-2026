"""k-deficit stubborn mining vs honest and Eyal–Sirer selfish mining.

The stubborn attacker never publishes while ahead. Once honest miners draw
level, it publishes everything and races; when it loses a race, or falls
behind, it keeps mining its own branch until the honest lead reaches k. Each
run has a fixed number of block discoveries; the fork still open at the end
goes to the longer branch, and a tie goes to the attacker with probability alpha.
"""

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.theory import sm_share

ALPHAS = np.arange(0.05, 0.46, 0.05).round(2).tolist()
CAPS = (1, 2, 3, 4)
EVENTS = 100_000


def simulate(rng, alpha, k):
    a = h = 0  # canonical blocks
    private = public = 0  # branch lengths since the fork
    race = False
    for _ in range(EVENTS):
        if rng.random() < alpha:
            private += 1
            if race:  # attacker wins the race
                a, private, public, race = a + private, 0, 0, False
            elif private == public:
                race = True
            continue
        public += 1
        race = private == public and not race
        if public - private >= k:  # attacker gives up
            h, private, public = h + public, 0, 0
    if private > public or (private == public and rng.random() < alpha):
        a += private
    else:
        h += public
    return {"revenue": a / (a + h)}


def run():
    return monte_carlo(simulate, [dict(alpha=a, k=k) for a in ALPHAS for k in CAPS])


def plot(df):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for color, (k, d) in zip(COLORS, df.groupby("k")):
        ax.errorbar(d.alpha, d.revenue, d.revenue_ci, fmt="o-", ms=2.5, lw=1, color=color, label=f"Stubborn, $k={k}$")
    x = np.linspace(min(ALPHAS), max(ALPHAS), 100)
    ax.plot(x, [sm_share(a) for a in x], "k-", lw=0.8, label="Selfish (theory)")
    ax.plot(x, x, "k--", lw=0.8, label="Honest")
    ax.set(xlabel=r"Attacker hashrate $\alpha$", ylabel="Attacker revenue share")
    ax.legend()
    return {"stubborn": fig}


if __name__ == "__main__":
    main(__file__, run, plot)
