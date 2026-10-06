"""Monte Carlo kernels for the two-member cartel incentive experiments.

The long-run tolerance experiment subclasses the repository's event-driven
cartel and chain-withhold engines.  The stubborn experiments reuse the exact
deficit/race transitions of :mod:`k_stubborn_sim` and the Eyal--Sirer release
rules of :mod:`pow_selfish`, while tracking the focal traitor block explicitly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


def _validate_trials(trials: int) -> None:
    if trials <= 0:
        raise ValueError("trials must be positive")


def _walk_disclosure_recovery(
    rng: random.Random, *, alpha_l: float, alpha_t: float, lead: int
) -> bool:
    """Simulate the killed random walk whose failure probability is lambda^ell."""

    while lead > 0:
        draw = rng.random()
        if draw < alpha_t:
            return True
        if draw < alpha_t + alpha_l:
            lead += 1
        else:
            lead -= 1
    return False


def simulate_disclosure_run(
    *,
    alpha: float,
    alpha_t: float,
    ell: int,
    gamma: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate both disclosure actions for one independent random seed."""

    alpha_l = alpha - alpha_t
    if not 0.0 < alpha_l < 1.0 or not 0.0 < alpha_t < alpha:
        raise ValueError("require 0 < alpha_t < alpha < 1")
    if ell < 1:
        raise ValueError("ell must be at least 1")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    _validate_trials(trials)

    rng = random.Random(seed)
    omega = alpha + gamma * (1.0 - alpha)
    obey_total = betray_total = 0.0
    recovery_count = current_obey_count = new_obey_count = 0
    for _ in range(trials):
        recovered = _walk_disclosure_recovery(
            rng, alpha_l=alpha_l, alpha_t=alpha_t, lead=ell
        )
        if recovered:
            recovery_count += 1
            current_obey_count += 1
            new_obey_count += 1
            obey_total += 2.0
        elif rng.random() < omega:
            current_obey_count += 1
            obey_total += 1.0

        # The published focal block is certain and the next opportunity is
        # owned by the traitor with probability alpha_t.
        betray_total += 1.0 + float(rng.random() < alpha_t)

    u_obey = obey_total / trials
    u_betray = betray_total / trials
    return {
        "alpha": alpha,
        "alpha_l": alpha_l,
        "alpha_t": alpha_t,
        "ell": ell,
        "gamma": gamma,
        "trials": trials,
        "seed": seed,
        "u_obey_sim": u_obey,
        "u_betray_sim": u_betray,
        "utility_difference_sim": u_obey - u_betray,
        "recovery_probability_sim": recovery_count / trials,
        "q_now_obey_sim": current_obey_count / trials,
        "q_new_obey_sim": new_obey_count / trials,
    }


def _simulate_tolerate(
    *, alpha_l: float, alpha_t: float, w_over_t: float, cycles: int, seed: int
) -> tuple[float, int, int, int]:
    """Compact regenerative form of ``pow_collusion.CollusionSimulation``.

    These are the same IDLE/WITHHOLD/RACE transitions, with block-tree storage
    replaced by canonical owner counters.  This makes the requested 200,000
    clean cycles practical without changing the policy.
    """

    rng = random.Random(seed)
    alpha = alpha_l + alpha_t
    loyal = traitor = honest = events = 0

    def discover() -> str:
        nonlocal events
        events += 1
        draw = rng.random()
        if draw < alpha_l:
            return "l"
        if draw < alpha:
            return "t"
        return "h"

    for _ in range(cycles):
        first = discover()
        if first == "h":
            honest += 1
            continue
        if first == "t":
            # A traitor who is first from IDLE publishes immediately.
            traitor += 1
            continue

        # Loyal owns the one hidden block.  Further loyal discoveries publish
        # the old block and renew this exact state with a fresh deadline.
        while True:
            if rng.expovariate(1.0) >= w_over_t:
                events += 1  # release deadline
                loyal += 1
                break
            second = discover()
            if second == "l":
                loyal += 1
                continue
            if second == "t":
                # Existing tolerated-disclosure rule: publish both blocks.
                loyal += 1
                traitor += 1
                break

            # Honest caught the hidden block: publish it into a zero-delay race.
            race_winner = discover()
            if race_winner == "l":
                loyal += 2
            elif race_winner == "t":
                loyal += 1
                traitor += 1
            else:
                honest += 2
            break

    canonical = loyal + traitor + honest
    return loyal / canonical, loyal, canonical, events


def _simulate_solo(
    *, alpha_l: float, w_over_t: float, cycles: int, seed: int
) -> tuple[float, int, int, int]:
    """Compact regenerative form of ``ChainWithholdSimulation``."""

    rng = random.Random(seed)
    loyal = honest = events = 0

    def attacker_discovers() -> bool:
        nonlocal events
        events += 1
        return rng.random() < alpha_l

    for _ in range(cycles):
        if not attacker_discovers():
            honest += 1
            continue
        while True:
            if rng.expovariate(1.0) >= w_over_t:
                events += 1
                loyal += 1
                break
            if attacker_discovers():
                loyal += 1
                continue
            if attacker_discovers():
                loyal += 2
            else:
                honest += 2
            break
    canonical = loyal + honest
    return loyal / canonical, loyal, canonical, events


def _simulate_honest(
    *, alpha_l: float, cycles: int, seed: int
) -> tuple[float, int, int, int]:
    rng = random.Random(seed)
    loyal = sum(rng.random() < alpha_l for _ in range(cycles))
    return loyal / cycles, loyal, cycles, cycles


def simulate_tolerance_run(
    *,
    regime: str,
    alpha_l: float,
    alpha_t: float,
    w_over_t: float,
    cycles: int,
    seed: int,
) -> dict[str, Any]:
    """Run one tolerance-policy trajectory using an exact repository engine."""

    if alpha_l <= 0.0 or alpha_t <= 0.0 or alpha_l + alpha_t >= 1.0:
        raise ValueError("require positive cartel shares summing to less than 1")
    if w_over_t < 0.0 or cycles <= 0:
        raise ValueError("w_over_t must be nonnegative and cycles positive")
    if regime == "TOLERATE":
        share, loyal, canonical, events = _simulate_tolerate(
            alpha_l=alpha_l,
            alpha_t=alpha_t,
            w_over_t=w_over_t,
            cycles=cycles,
            seed=seed,
        )
    elif regime == "EXPEL_AND_SOLO_WITHHOLD":
        share, loyal, canonical, events = _simulate_solo(
            alpha_l=alpha_l,
            w_over_t=w_over_t,
            cycles=cycles,
            seed=seed,
        )
    elif regime == "HONEST":
        share, loyal, canonical, events = _simulate_honest(
            alpha_l=alpha_l, cycles=cycles, seed=seed
        )
    else:
        raise ValueError(f"unknown regime: {regime}")
    return {
        "regime": regime,
        "alpha": alpha_l + alpha_t,
        "alpha_l": alpha_l,
        "alpha_t": alpha_t,
        "w_over_T": w_over_t,
        "cycles": cycles,
        "seed": seed,
        "loyal_blocks": loyal,
        "canonical_blocks": canonical,
        "events": events,
        "loyal_share_sim": share,
    }


def _hit_lower_before_upper(
    rng: random.Random, *, p: float, start: int, lower: int, upper: int
) -> bool:
    position = start
    while lower < position < upper:
        position += -1 if rng.random() < p else 1
    return position == lower


def simulate_stubborn_run(
    *,
    alpha: float,
    alpha_t: float,
    d: int,
    k: int,
    m: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Monte Carlo validation of the m-capped analytical random walks."""

    alpha_l = alpha - alpha_t
    if not 0.0 < alpha_t < alpha < 1.0:
        raise ValueError("require 0 < alpha_t < alpha < 1")
    if not 1 <= d < k or m < 1:
        raise ValueError("require 1 <= d < k and m >= 1")
    _validate_trials(trials)
    rng = random.Random(seed)
    cartel_payoff = honest_payoff = 0
    for _ in range(trials):
        if rng.random() < alpha_t and _hit_lower_before_upper(
            rng, p=alpha, start=d - 1, lower=-m, upper=k
        ):
            cartel_payoff += 1

        if rng.random() < alpha_t:
            loyal_catches = _hit_lower_before_upper(
                rng, p=alpha_l, start=d + 1, lower=0, upper=k
            )
            cartel_then_wins = loyal_catches and _hit_lower_before_upper(
                rng, p=alpha, start=0, lower=-m, upper=k
            )
            honest_payoff += int(not cartel_then_wins)

    u_cartel = cartel_payoff / trials
    u_honest = honest_payoff / trials
    return {
        "alpha": alpha,
        "alpha_l": alpha_l,
        "alpha_t": alpha_t,
        "d": d,
        "k": k,
        "m": m,
        "trials": trials,
        "seed": seed,
        "u_cartel_sim": u_cartel,
        "u_honest_sim": u_honest,
        "utility_difference_sim": u_cartel - u_honest,
    }


@dataclass(slots=True)
class _TrackedBlock:
    focal: bool = False
    owner: str = "other"


def _exact_selfish_focal_outcome(
    rng: random.Random,
    *,
    private: list[_TrackedBlock],
    public: list[_TrackedBlock],
    alpha: float,
    alpha_t: float,
    gamma: float,
    disclose: bool,
) -> bool:
    """Follow the focal block through exact Eyal--Sirer continuation.

    Entry is an unpublished equal-length cartel branch (``d=0``).  Unlike a
    normal public tie, it remains hidden until the exact selfish state machine
    calls for publication.
    """

    if len(private) != len(public):
        raise AssertionError("exact continuation must start at hidden equality")
    in_race = False
    while True:
        if in_race:
            if rng.random() < alpha:
                private.append(_TrackedBlock(owner="cartel"))
                return any(block.focal for block in private)
            if rng.random() < gamma:
                private.append(_TrackedBlock(owner="honest"))
                return any(block.focal for block in private)
            public.append(_TrackedBlock(owner="honest"))
            return any(block.focal for block in public)

        lead = len(private) - len(public)
        if lead < 0:  # Only possible after an honest discovery at hidden equality.
            return any(block.focal for block in public)

        draw = rng.random()
        if draw < alpha:
            owner = "traitor" if draw < alpha_t else "loyal"
            private.append(_TrackedBlock(owner=owner))
            # As in pow_collusion.py, a later traitor discovery discloses all
            # hidden blocks.  Publication is decisive only at a strict lead.
            if disclose and owner == "traitor" and len(private) > len(public):
                return any(block.focal for block in private)
            continue

        public.append(_TrackedBlock(owner="honest"))
        if lead == 0:
            return any(block.focal for block in public)
        if lead == 1:
            in_race = True
            continue
        if lead == 2:
            return any(block.focal for block in private)

        # lead >= 3: publish the oldest private block, orphan the matching
        # honest block, retain the remaining lead (classic k -> k-1 rule).
        private_finalized = private.pop(0)
        public_orphaned = public.pop(0)
        if private_finalized.focal:
            return True
        if public_orphaned.focal:
            return False


def _stubborn_then_exact(
    rng: random.Random,
    *,
    private: list[_TrackedBlock],
    public: list[_TrackedBlock],
    stubborn_private_rate: float,
    alpha: float,
    alpha_t: float,
    gamma: float,
    k: int,
    disclose: bool,
) -> bool:
    while len(public) - len(private) > 0:
        if len(public) - len(private) >= k:
            return any(block.focal for block in public)
        if rng.random() < stubborn_private_rate:
            private.append(_TrackedBlock(owner="loyal"))
        else:
            public.append(_TrackedBlock(owner="honest"))
    return _exact_selfish_focal_outcome(
        rng,
        private=private,
        public=public,
        alpha=alpha,
        alpha_t=alpha_t,
        gamma=gamma,
        disclose=disclose,
    )


def simulate_exact_policy_run(
    *,
    alpha: float,
    alpha_t: float,
    d: int,
    k: int,
    gamma: float,
    trials: int,
    seed: int,
    disclose: bool = True,
) -> dict[str, Any]:
    """Estimate focal-block payoffs with exact selfish continuation at ``d=0``."""

    alpha_l = alpha - alpha_t
    if not 0.0 < alpha_t < alpha < 1.0:
        raise ValueError("require 0 < alpha_t < alpha < 1")
    if not 1 <= d < k:
        raise ValueError("require 1 <= d < k")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    _validate_trials(trials)
    rng = random.Random(seed)
    follow_payoff = honest_payoff = 0
    for _ in range(trials):
        if rng.random() < alpha_t:
            follow_payoff += int(
                _stubborn_then_exact(
                    rng,
                    private=[_TrackedBlock(focal=True, owner="traitor")],
                    public=[_TrackedBlock(owner="honest") for _ in range(d)],
                    stubborn_private_rate=alpha,
                    alpha=alpha,
                    alpha_t=alpha_t,
                    gamma=gamma,
                    k=k,
                    disclose=disclose,
                )
            )

        if rng.random() < alpha_t:
            honest_payoff += int(
                _stubborn_then_exact(
                    rng,
                    private=[],
                    public=[
                        *[_TrackedBlock(owner="honest") for _ in range(d)],
                        _TrackedBlock(focal=True, owner="traitor"),
                    ],
                    stubborn_private_rate=alpha_l,
                    alpha=alpha,
                    alpha_t=alpha_t,
                    gamma=gamma,
                    k=k,
                    disclose=disclose,
                )
            )

    u_cartel = follow_payoff / trials
    u_honest = honest_payoff / trials
    return {
        "alpha": alpha,
        "alpha_l": alpha_l,
        "alpha_t": alpha_t,
        "d": d,
        "k": k,
        "gamma": gamma,
        "disclosure_rule": disclose,
        "trials": trials,
        "seed": seed,
        "u_cartel_exact_sim": u_cartel,
        "u_honest_exact_sim": u_honest,
        "utility_difference_exact_sim": u_cartel - u_honest,
    }
