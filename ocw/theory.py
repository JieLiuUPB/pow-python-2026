"""Closed forms for an attacker with hashrate alpha and gamma = 0.

OCW share and orphan rate hold for a release window w (in units of T); the
Eyal–Sirer formulas hold for alpha < 1/2.
"""

import math


def ocw_share(alpha, w):
    """Attacker share of canonical blocks under OCW. w = 0 is honest mining."""
    s = 1 - math.exp(-w)
    return alpha * (1 - s) + alpha**2 * (3 - 2 * alpha) * s


def ocw_orphans(alpha, w):
    """Orphan rate (orphaned / published blocks) under OCW."""
    x = alpha**2 * (1 - alpha) * (1 - math.exp(-w))
    return x / (1 + x)


def sm_share(alpha):
    """Eyal–Sirer selfish-mining revenue."""
    return (4 * alpha**2 * (1 - alpha) ** 2 - alpha**3) / (1 - alpha * (1 + (2 - alpha) * alpha))


def sm_orphans(alpha):
    """Selfish-mining orphan rate (orphaned / mined blocks). Above 1/2 the lead
    drifts upwards and every honest block is eventually orphaned."""
    if alpha >= 0.5:
        return 1 - alpha
    return alpha * (1 - alpha) ** 2 / (1 - 4 * alpha**2 + 2 * alpha**3)
