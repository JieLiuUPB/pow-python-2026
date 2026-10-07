"""§6: after a betrayal, should the loyal miner tolerate or expel the traitor?

Exact Markov reward chain of k_s-capped selfish mining (§6.1). A state is the
owner string of the unpublished cartel blocks, oldest first ("" = no fork,
"LT" = loyal block then traitor block, lead 2), or "r" + owner for a public
one-block race. The loyal miner follows the Eyal–Sirer release rule and never
holds more than k_s blocks. A tolerated traitor mines on the private tip and
publishes the whole private chain whenever its new block leaves a lead below
ell* (Theorem 1). An expelled traitor mines honestly, which is the same chain
with alpha_t = 0.

psi is the loyal miner's long-run share of canonical blocks (Eq. 41,
normalised by the canonical block rate). Tolerating pays iff
psi_tol >= max(alpha_l, psi_solo) (Theorem 3). gamma = 0 throughout.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from cartel.theory import ell_star
from harness import COLORS, main

CAPS = (1, 2, 4, 8)  # k_s
SHARES = np.arange(0.025, 1, 0.025).round(3).tolist()  # alpha_l and alpha_t grid
GRID = [(l, t) for l in SHARES for t in SHARES if l + t < 1]


def transitions(s, alpha_l, alpha_t, ks, ell):
    """Yield (probability, next state, loyal canonical blocks, canonical blocks)."""
    alpha_h = 1 - alpha_l - alpha_t
    if s.startswith("r"):  # race: the next cartel block decides it for the cartel
        tie = s[1] == "L"
        yield alpha_l, "", tie + 1, 2
        yield alpha_t, "", tie, 2
        yield alpha_h, "", 0, 2
        return
    for owner, p in (("L", alpha_l), ("T", alpha_t)):
        o = s + owner
        if owner == "T" and len(o) < ell:  # traitor publishes: the cartel chain wins
            yield p, "", o.count("L"), len(o)
        elif len(o) > ks:  # cap: publish the oldest block, honest miners switch to it
            yield p, o[1:], o[0] == "L", 1
        else:
            yield p, o, 0, 0
    if len(s) == 0:
        yield alpha_h, "", 0, 1
    elif len(s) == 1:
        yield alpha_h, "r" + s, 0, 0
    elif len(s) == 2:  # publish both blocks, orphaning the honest one
        yield alpha_h, "", s.count("L"), 2
    else:  # publish the oldest block; its rival will be orphaned
        yield alpha_h, s[1:], s[0] == "L", 1


def psi(alpha_l, alpha_t, ks):
    """Loyal share of canonical blocks under the stationary distribution (Eqs. 40–41)."""
    ell = ell_star(alpha_l + alpha_t, alpha_t) if alpha_t else 1
    states, index, edges = [""], {"": 0}, []
    for s in states:  # breadth-first: the list grows while we walk it
        for p, nxt, loyal, total in transitions(s, alpha_l, alpha_t, ks, ell):
            if p > 0:
                if nxt not in index:
                    index[nxt] = len(states)
                    states.append(nxt)
                edges.append((index[s], index[nxt], p, loyal, total))
    n = len(states)
    P, reward = np.zeros((n, n)), np.zeros((n, 2))
    for i, j, p, loyal, total in edges:
        P[i, j] += p
        reward[i] += p * loyal, p * total
    A = P.T - np.eye(n)
    A[-1] = 1  # replace one balance equation by sum(pi) = 1
    pi = np.linalg.solve(A, np.eye(n)[-1])
    loyal, total = pi @ reward
    return loyal / total


def run():
    rows = [(ks, l, t, psi(l, t, ks), psi(l, 0, ks)) for ks in CAPS for l, t in GRID]
    return pd.DataFrame(rows, columns=["ks", "alpha_l", "alpha_t", "psi_tol", "psi_solo"])


def plot(df):
    df["gain"] = df.psi_tol - np.maximum(df.alpha_l, df.psi_solo)
    region, axes = plt.subplots(1, len(CAPS), figsize=(7.2, 2.3), sharey=True)
    for ax, ks in zip(axes, CAPS):
        g = df[df.ks == ks].pivot_table("gain", "alpha_l", "alpha_t")
        image = ax.pcolormesh(g.columns, g.index, g, cmap="RdBu", norm=TwoSlopeNorm(0, -0.1, 0.1), shading="nearest")
        ax.contour(g.columns, g.index, g, levels=[0], colors="black", linewidths=0.7)
        ax.set(title=f"$k_s={ks}$", xlabel=r"$\alpha_t$", xticks=(0.2, 0.6), aspect="equal")
    axes[0].set_ylabel(r"$\alpha_l$")
    region.colorbar(image, ax=axes, shrink=0.8, label="Tolerance gain\n" r"$\psi_{\rm tol}-\max(\alpha_l,\psi_{\rm solo})$")

    share, axes = plt.subplots(1, 3, figsize=(7, 2.3), sharey=True)
    for ax, t in zip(axes, (0.10, 0.20, 0.30)):
        d = df[(df.ks == 2) & (df.alpha_t == t)]
        for color, y, label in zip(COLORS, ("alpha_l", "psi_solo", "psi_tol"), ("Honest", "Expel, withhold alone", "Tolerate")):
            ax.plot(d.alpha_l, d[y] / d.alpha_l, color=color, lw=1, label=label)
        ax.set(title=rf"$\alpha_t={t:.2f}$, $k_s=2$", xlabel=r"$\alpha_l$")
    axes[0].set_ylabel(r"Loyal share / $\alpha_l$")
    axes[0].legend()
    share.tight_layout()
    return {"tolerance_region": region, "tolerance_share": share}


if __name__ == "__main__":
    main(__file__, run, plot)
