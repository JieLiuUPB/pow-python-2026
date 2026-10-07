"""§5: while the cartel branch trails by b blocks, should the traitor mine on it?

For each deficit b, simulates the chance that the traitor's next block becomes
canonical if mined on the cartel branch (P_C) or the honest branch (P_H). The
simulation plays the full fork out: k-deficit-capped stubborn mining until the
cartel catches up, then Eyal–Sirer selfish mining. The largest b with
P_C >= P_H is compared with the closed-form B_lose, whose sigma treats the
selfish phase as a race to lead M = 2.
"""

import matplotlib.pyplot as plt
import numpy as np

from cartel.theory import b_lose, branch_payoffs
from harness import COLORS, heatmap, main, monte_carlo

ALPHAS = (0.55, 0.60, 0.65, 0.70, 0.75)
CAPS = (2, 3, 4, 5, 6, 8)  # k: loyal miners abandon the cartel branch at deficit k
TRIALS = 10_000


def survives(rng, private, public, p, alpha, alpha_t, k):
    """Whether the focal block (the True entry in a branch list) ends up canonical.

    private/public list the cartel and honest blocks after the fork point, oldest
    first. Behind, the cartel branch grows at rate p. From hidden catch-up on, a
    traitor block is published at once, ending the fork in the cartel's favour.
    """
    while len(private) < len(public):
        if len(public) - len(private) >= k:
            return any(public)
        (private if rng.random() < p else public).append(False)
    race = False  # public tie after honest miners matched a one-block lead
    while True:
        u = rng.random()
        if race:
            return any(private) if u < alpha else any(public)
        if u < alpha:
            private.append(False)
            if u < alpha_t:
                return any(private)
            continue
        lead = len(private) - len(public)
        public.append(False)
        if lead == 0:
            return any(public)
        if lead == 1:
            race = True
        elif lead == 2:
            return any(private)
        else:  # publish the oldest private block, orphaning its honest rival
            if private.pop(0):
                return True
            if public.pop(0):
                return False


def simulate(rng, alpha, alpha_t, k, b):
    cartel = honest = 0
    for _ in range(TRIALS):
        cartel += survives(rng, [True], [False] * b, alpha, alpha, alpha_t, k)
        honest += survives(rng, [], [False] * b + [True], alpha - alpha_t, alpha, alpha_t, k)
    return {"cartel": cartel / TRIALS, "honest": honest / TRIALS}


def run():
    points = [dict(alpha=a, alpha_t=t, k=k, b=b) for a in ALPHAS
              for t in np.arange(0.05, a - 0.04, 0.05).round(2).tolist() for k in CAPS for b in range(1, k)]
    df = monte_carlo(simulate, points)
    theory = [branch_payoffs(a, t, b, k) for a, t, k, b in zip(df.alpha, df.alpha_t, df.k, df.b)]
    df["cartel_theory"], df["honest_theory"] = zip(*theory)
    df["b_lose"] = [b_lose(a, t, k) for a, t, k in zip(df.alpha, df.alpha_t, df.k)]
    return df


def plot(df):
    d = df[(df.alpha == 0.65) & (df.alpha_t == 0.15) & (df.k == 6)]
    payoff, ax = plt.subplots(figsize=(3.4, 2.5))
    for color, branch, label in ((COLORS[2], "cartel", "Cartel branch $P_C$"), (COLORS[1], "honest", "Honest branch $P_H$")):
        ax.plot(d.b, d[f"{branch}_theory"], color=color, lw=1)
        ax.errorbar(d.b, d[branch], d[f"{branch}_ci"], fmt="o", ms=3, capsize=2, color=color, label=label)
    ax.set(xlabel=r"Cartel deficit $b$", ylabel="P(traitor block canonical)",
           title=r"$\alpha=0.65,\ \alpha_t=0.15,\ k=6$; lines: theory")
    ax.legend()

    # Simulated threshold: the largest deficit where following still pays.
    df["b_lose_sim"] = df.b.where(df.cartel >= df.honest, 0)
    threshold, axes = plt.subplots(2, 3, figsize=(7, 4.6), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, CAPS):
        d = df[df.k == k]
        heatmap(ax, d.pivot_table("b_lose", "alpha", "alpha_t"), d.pivot_table("b_lose_sim", "alpha", "alpha_t", "max"))
        ax.set_title(f"$k={k}$")
    for ax in axes[1]:
        ax.set_xlabel(r"$\alpha_t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\alpha$")
    threshold.suptitle(r"$B_{\rm lose}$: colour = theory, number = simulation")
    threshold.tight_layout()
    return {"stubborn_payoff": payoff, "stubborn_threshold": threshold}


if __name__ == "__main__":
    main(__file__, run, plot)
