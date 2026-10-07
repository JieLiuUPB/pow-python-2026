"""Closed-form results of the paper. Equation numbers refer to Cartel.pdf.

Notation: alpha = alpha_l + alpha_t is the total cartel hashrate, alpha_l the
loyal and alpha_t the traitor share; honest miners outside hold 1 - alpha.
"""

import math
from itertools import count

M = 2  # The selfish phase is won once the cartel leads by M blocks (Eyal–Sirer override).


def omega(alpha, gamma=0.0):
    """Probability that the cartel branch wins a public tie (Eq. 1)."""
    return alpha + gamma * (1 - alpha)


def lam(alpha, alpha_t):
    """lambda^ell is the chance that honest miners erase a lead ell before
    the traitor mines again (Eq. 8, rationalized to avoid cancellation)."""
    alpha_l = alpha - alpha_t
    return 2 * (1 - alpha) / (1 + math.sqrt(1 - 4 * alpha_l * (1 - alpha)))


def withhold_utility(alpha, alpha_t, ell, gamma=0.0):
    """Traitor utility of obeying the cartel at private lead ell (Eq. 12).
    Publishing instead yields 1 + alpha_t (Eq. 4)."""
    w = omega(alpha, gamma)
    return w + (2 - w) * (1 - lam(alpha, alpha_t) ** ell)


def ell_star(alpha, alpha_t, gamma=0.0):
    """Smallest lead at which withholding beats publishing (Theorem 1, Eq. 14)."""
    bound = (1 - alpha_t) / (2 - omega(alpha, gamma))
    lam_ = lam(alpha, alpha_t)
    return next(ell for ell in count(1) if lam_**ell <= bound + 1e-12)


def h(p, b, k):
    """Probability that a branch growing at rate p climbs from deficit b back
    to 0 before falling to deficit k (gambler's ruin, Eq. 32)."""
    if math.isclose(p, 0.5):
        return 1 - b / k
    r = p / (1 - p)
    return (r**b - r**k) / (1 - r**k)


def sigma(alpha, k):
    """Value of hidden catch-up (b = 0): chance the full cartel then reaches
    lead M before falling to deficit k. A proxy for sigma_B of Eq. 17."""
    return h(alpha, M, k + M)


def branch_payoffs(alpha, alpha_t, b, k):
    """Chance that a traitor block found at deficit b becomes canonical when
    mined on the cartel branch (P_C, Eq. 33) or the honest branch (P_H, Eq. 34).
    sigma_in = sigma_out = sigma: the traitor rejoins the cartel after catch-up."""
    s = sigma(alpha, k)
    return s * h(alpha, b - 1, k), 1 - s * h(alpha - alpha_t, b + 1, k)


def b_lose(alpha, alpha_t, k):
    """Largest deficit at which the traitor still follows the cartel (Eqs. 31, 35);
    0 if it never does."""
    def follows(b):
        cartel, honest = branch_payoffs(alpha, alpha_t, b, k)
        return cartel >= honest - 1e-12

    return max((b for b in range(1, k) if follows(b)), default=0)

