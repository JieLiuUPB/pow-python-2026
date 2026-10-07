"""The paper's closed forms for attacker hashrate alpha and withholding window w (in T)."""

import math


def rho(alpha, w):
    """Attacker share of canonical blocks (Eq. 2). w = 0 is honest mining."""
    z = math.exp(-w)
    return alpha * z + alpha**2 * (1 - z) * (3 - 2 * alpha)


def eta(alpha, w):
    """Orphan rate: orphaned / published blocks (Eq. 25)."""
    x = alpha**2 * (1 - alpha) * (1 - math.exp(-w))
    return x / (1 + x)


def q(alpha, w, daa):
    """(q_mc, q): canonical blocks and DAA-counted blocks per mined block (Eq. 4, 7)."""
    x = alpha * (1 - alpha) * (1 - math.exp(-w))
    mc = 1 / (1 + x)
    return mc, mc if daa == "canonical" else (1 + alpha * x) * mc


def ratio(n, share, hashrate, mc, counted):
    """Block increase ratio after n rounds of 2016 T, for a miner holding `share`
    of the canonical blocks. Before the first retarget it earns share * mc
    blocks per T (Eq. 3). That retarget comes after 1/counted rounds and speeds
    blocks up by 1/counted (Eq. 5, 8); later retargets change nothing."""
    t1 = 1 / counted
    return share * mc * (min(n, t1) + max(n - t1, 0) / counted) / (hashrate * n)
