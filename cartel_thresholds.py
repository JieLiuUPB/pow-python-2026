"""Closed-form benchmarks for the two-member cartel experiments.

The public API deliberately uses ``alpha`` for the total cartel hashrate.
Older simulators in this repository call the same quantity ``p``.
"""

from __future__ import annotations

import math


_EPS = 1e-12


def _validate_cartel(alpha: float, alpha_t: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 <= alpha_t < alpha:
        raise ValueError("alpha_t must be in [0, alpha)")
    return alpha - alpha_t


def lambda_value(alpha: float, alpha_t: float) -> float:
    """Return the disclosure random-walk root for loyal share ``alpha-alpha_t``.

    The rationalized form avoids cancellation when ``1-alpha`` is small.
    """

    alpha_l = _validate_cartel(alpha, alpha_t)
    discriminant = 1.0 - 4.0 * alpha_l * (1.0 - alpha)
    if discriminant < -_EPS:
        raise ValueError("lambda discriminant is negative")
    root = math.sqrt(max(0.0, discriminant))
    return 2.0 * (1.0 - alpha) / (1.0 + root)


def disclosure_utilities(
    alpha: float, alpha_t: float, ell: int, gamma: float = 0.0
) -> tuple[float, float]:
    """Return ``(U_betray, U_obey)`` at private lead ``ell``."""

    if ell < 1:
        raise ValueError("ell must be at least 1")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    lam = lambda_value(alpha, alpha_t)
    r_ell = 1.0 - lam**ell
    omega = alpha + gamma * (1.0 - alpha)
    u_betray = 1.0 + alpha_t
    u_obey = r_ell + (1.0 - r_ell) * omega + r_ell
    return u_betray, u_obey


def ell_star(alpha: float, alpha_t: float, gamma: float = 0.0) -> int:
    """Smallest positive integer lead at which obeying weakly dominates."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    lam = lambda_value(alpha, alpha_t)
    omega = alpha + gamma * (1.0 - alpha)
    rhs = (1.0 - alpha_t) / (2.0 - omega)
    if rhs >= 1.0 - _EPS or lam <= _EPS:
        return 1
    # Use the logarithm only as a starting point, then verify the defining
    # inequality so exact-boundary floating-point values are handled safely.
    candidate = max(1, int(math.ceil(math.log(rhs) / math.log(lam) - _EPS)))
    while candidate > 1 and lam ** (candidate - 1) <= rhs + _EPS:
        candidate -= 1
    while lam**candidate > rhs + _EPS:
        candidate += 1
    return candidate


def rho_tolerate(alpha_l: float, alpha_t: float, z: float) -> float:
    """Loyal canonical share in the one-block-capped tolerance model."""

    _validate_shares_and_z(alpha_l, alpha_t, z)
    alpha = alpha_l + alpha_t
    return alpha_l * (
        z
        + (1.0 - z)
        * (2.0 * alpha + alpha_l - alpha**2 - alpha * alpha_l)
    )


def rho_solo(alpha_l: float, z: float) -> float:
    """Loyal canonical share after expelling the traitor and withholding solo."""

    _validate_shares_and_z(alpha_l, 0.0, z)
    return alpha_l * z + alpha_l**2 * (1.0 - z) * (3.0 - 2.0 * alpha_l)


def tolerance_differences(
    alpha_l: float, alpha_t: float, z: float
) -> tuple[float, float]:
    """Return theory differences versus solo withholding and honest mining."""

    tolerate = rho_tolerate(alpha_l, alpha_t, z)
    return tolerate - rho_solo(alpha_l, z), tolerate - alpha_l


def tolerate_is_optimal(alpha_l: float, alpha_t: float) -> bool:
    """Whether TOLERATE weakly beats both alternatives in the theory."""

    _validate_shares_and_z(alpha_l, alpha_t, 1.0)
    return (
        2.0 * alpha_l + alpha_t >= 1.0 - _EPS
        and 3.0 * alpha_l + alpha_t <= 2.0 + _EPS
    )


def _validate_shares_and_z(alpha_l: float, alpha_t: float, z: float) -> None:
    if alpha_l < 0.0 or alpha_t < 0.0 or alpha_l + alpha_t >= 1.0:
        raise ValueError("hashrates must be nonnegative and sum to less than 1")
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must be in [0, 1]")


def F_private_success(p: float, d: int, m: int, k: int) -> float:
    """Probability of hitting deficit ``-m`` before abandonment at ``k``."""

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if m < 1 or k < 1:
        raise ValueError("m and k must be positive")
    if not -m <= d <= k:
        raise ValueError("d must lie between -m and k")
    if d == -m:
        return 1.0
    if d == k:
        return 0.0
    if p <= _EPS:
        return 0.0
    if p >= 1.0 - _EPS:
        return 1.0
    if math.isclose(p, 0.5, rel_tol=0.0, abs_tol=_EPS):
        return (k - d) / (k + m)
    r_p = p / (1.0 - p)
    return (r_p ** (d + m) - r_p ** (k + m)) / (
        1.0 - r_p ** (k + m)
    )


def G_loyal_catchup(alpha_l: float, d: int, k: int) -> float:
    """Probability loyal mining reaches a tie from deficit ``d`` before ``k``."""

    if not 0.0 <= alpha_l <= 1.0:
        raise ValueError("alpha_l must be in [0, 1]")
    if k < 1:
        raise ValueError("k must be positive")
    if not 0 <= d <= k:
        raise ValueError("d must lie between 0 and k")
    if d == 0:
        return 1.0
    if d == k:
        return 0.0
    if alpha_l <= _EPS:
        return 0.0
    if alpha_l >= 1.0 - _EPS:
        return 1.0
    if math.isclose(alpha_l, 0.5, rel_tol=0.0, abs_tol=_EPS):
        return 1.0 - d / k
    s = alpha_l / (1.0 - alpha_l)
    return (s**d - s**k) / (1.0 - s**k)


def stubborn_utilities(
    alpha: float, alpha_t: float, d: int, k: int, m: int = 2
) -> tuple[float, float]:
    """Return ``(U_cartel, U_honest)`` for the focal traitor decision."""

    alpha_l = _validate_cartel(alpha, alpha_t)
    if alpha_t <= 0.0:
        raise ValueError("alpha_t must be positive for a traitor decision")
    if k < 2 or not 1 <= d < k:
        raise ValueError("require k >= 2 and d in {1, ..., k-1}")
    follow_success = F_private_success(alpha, d - 1, m, k)
    s0 = F_private_success(alpha, 0, m, k)
    loyal_catchup = G_loyal_catchup(alpha_l, d + 1, k)
    return (
        alpha_t * follow_success,
        alpha_t * (1.0 - s0 * loyal_catchup),
    )


def stubborn_follow_margin(
    alpha: float, alpha_t: float, d: int, k: int, m: int = 2
) -> float:
    """Return FOLLOW utility minus HONEST utility, divided by ``alpha_t``.

    Comparing the two utilities in the task gives
    ``F(d-1) + F(0) G(d+1) - 1``.  This plus sign is important: the isolated
    minus-sign rendering of the requested inequality is inconsistent with the
    two utility definitions immediately above it.
    """

    u_cartel, u_honest = stubborn_utilities(alpha, alpha_t, d, k, m)
    return (u_cartel - u_honest) / alpha_t


def B_lose(alpha: float, alpha_t: float, k: int, m: int = 2) -> int:
    """Largest positive deficit at which FOLLOW weakly dominates.

    Zero means that the satisfying set in ``{1, ..., k-1}`` is empty.
    """

    satisfying = [
        d
        for d in range(1, k)
        if stubborn_follow_margin(alpha, alpha_t, d, k, m) >= -_EPS
    ]
    return max(satisfying, default=0)


def b_lose(alpha: float, alpha_t: float, k: int, m: int = 2) -> int:
    """PEP-8 alias for :func:`B_lose`."""

    return B_lose(alpha, alpha_t, k, m)
