"""Event-driven PoW network with an OCW attacker or cartel.

Time is measured in target block intervals T. Blocks arrive as a Poisson
process of rate 1/D (D = difficulty) and each block's miner is drawn by
hashrate. Publishing is instant and everyone sees the same block tree. The
highest block is the tip; a rival of equal height never replaces it, because
it was published later.

OCW (one-block-capped withholding): when a cartel member mines a block, the
cartel keeps it private for at most the withholding window w, then publishes it.
- Another cartel block comes first: publish the held block, hold the new one.
- An honest block comes first: race. The held block stays private; the cartel
  wins if it mines the next block, otherwise the honest block wins.
An optional traitor in the cartel publishes every block it mines at once,
together with the held block. After `limit` betrayals the cartel breaks up:
the traitor is expelled, or, if the rest is a minority, everyone mines honestly.
"""

import math

EPOCH = 2016  # blocks per difficulty adjustment


class Net:
    def __init__(self, rng, hashrate, cartel, w=10.0, daa=None, traitor=None, limit=math.inf):
        """hashrate: {pool: share}; daa: None, "canonical" (count blocks on
        the canonical chain), or "public" (count every published block)."""
        self.rng, self.hashrate, self.cartel, self.w, self.daa = rng, hashrate, set(cartel), w, daa
        self.traitor, self.limit, self.betrayals = traitor, limit, 0
        self.parent, self.height, self.miner, self.time = [None], [0], [None], [0.0]  # block 0: genesis
        self.tip = 0
        self.t = self.epoch_start = 0.0
        self.D, self.epochs = 1.0, 0
        self.reset()

    def reset(self):
        self.held = None  # (parent, miner) of the private block
        self.deadline = math.inf  # release time of the held block
        self.race = None  # honest block racing the held block

    def publish(self, parent, miner):
        self.parent.append(parent)
        self.height.append(self.height[parent] + 1)
        self.miner.append(miner)
        self.time.append(self.t)
        block = len(self.parent) - 1
        if self.height[block] > self.height[self.tip]:
            self.tip = block
        return block

    def chain(self):
        """Canonical block ids, oldest first, genesis excluded."""
        blocks, b = [], self.tip
        while b:
            blocks.append(b)
            b = self.parent[b]
        return blocks[::-1]

    def grow(self, n):
        """Mine until the canonical chain holds n blocks."""
        while self.height[self.tip] < n:
            self.step()

    def step(self, until=math.inf):
        """Process the next event. If it would happen after `until`, stop the
        clock at `until` instead and return False."""
        t = self.t + self.rng.expovariate(1 / self.D)
        if min(t, self.deadline) > until:
            self.t = until
            return False
        if self.deadline <= t:
            self.t = self.deadline
            self.publish(*self.held)
            self.reset()
        else:
            self.t = t
            miner = self.rng.choices(list(self.hashrate), list(self.hashrate.values()))[0]
            (self.cartel_block if miner in self.cartel else self.honest_block)(miner)
        if self.daa:
            self.adjust()
        return True

    def cartel_block(self, miner):
        parent = self.tip if self.held is None else self.publish(*self.held)
        if self.race is not None:  # the cartel wins the race
            self.publish(parent, miner)
            self.reset()
        elif miner == self.traitor:
            self.publish(parent, miner)
            self.reset()
            self.betrayed()
        else:
            self.held, self.deadline = (parent, miner), self.t + self.w

    def honest_block(self, miner):
        if self.race is not None:  # the honest side wins the race
            self.publish(self.race, miner)
            self.reset()
            return
        block = self.publish(self.tip, miner)
        if self.held is not None:
            self.race, self.deadline = block, math.inf

    def betrayed(self):
        self.betrayals += 1
        if self.betrayals >= self.limit:
            loyal = sum(self.hashrate[p] for p in self.cartel if p != self.traitor)
            self.cartel = self.cartel - {self.traitor} if loyal >= 0.5 else set()

    def adjust(self):
        """D_new = D_old * EPOCH / (time the last EPOCH counted blocks took)."""
        while True:
            n = (self.epochs + 1) * EPOCH
            if self.daa == "public":  # block ids follow publication order
                if len(self.time) <= n:
                    return
                b = n
            else:
                if self.height[self.tip] < n:
                    return
                b = self.tip
                while self.height[b] > n:
                    b = self.parent[b]
            self.D *= EPOCH / (self.time[b] - self.epoch_start)
            self.epoch_start, self.epochs = self.time[b], self.epochs + 1

