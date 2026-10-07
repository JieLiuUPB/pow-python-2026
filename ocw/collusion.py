"""Three-pool OCW cartel with a traitor (w = 10T, no difficulty adjustment).

A loyal pool (0.3) and a traitor pool (0.3) run OCW together against honest
miners (0.4). The traitor publishes every block it mines at once. The loyal
pool tolerates a number of betrayals; the next one dissolves the cartel and
everyone mines honestly. Shares are taken over the first 20160 canonical blocks.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from harness import COLORS, main, monte_carlo
from ocw.chain import Net

HASHRATE = {"loyal": 0.3, "traitor": 0.3, "honest": 0.4}
BLOCKS = 20_160
SCENARIOS = {  # name: (traitor, betrayal that dissolves the cartel)
    "No betrayal": (None, math.inf),
    "Betrayal, no tolerance": ("traitor", 1),
    "Betrayal, 999 tolerated": ("traitor", 1000),
    "Betrayal, always tolerated": ("traitor", math.inf),
}


def simulate(rng, scenario):
    traitor, limit = SCENARIOS[scenario]
    net = Net(rng, HASHRATE, cartel={"loyal", "traitor"}, traitor=traitor, limit=limit)
    net.grow(BLOCKS)
    miners = [net.miner[b] for b in net.chain()[:BLOCKS]]
    return {pool: miners.count(pool) / BLOCKS for pool in HASHRATE}


def run():
    return monte_carlo(simulate, [dict(scenario=s) for s in SCENARIOS])


def plot(df):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    x = np.arange(len(HASHRATE))
    width = 0.16
    ax.bar(x - 2 * width, list(HASHRATE.values()), width, color="0.75", label="Hashrate")
    for i, (color, row) in enumerate(zip(COLORS, df.itertuples()), start=-1):
        ax.bar(x + i * width, [getattr(row, p) for p in HASHRATE], width, color=color, label=row.scenario,
               yerr=[getattr(row, f"{p}_ci") for p in HASHRATE])
    ax.set_xticks(x, [p.capitalize() for p in HASHRATE])
    ax.set(ylabel="Share of canonical blocks")
    ax.legend(ncol=2, loc="upper left")
    ax.set_ylim(0, 0.6)
    return {"collusion": fig}


if __name__ == "__main__":
    main(__file__, run, plot)
