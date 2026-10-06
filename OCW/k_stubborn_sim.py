"""Monte Carlo simulator for honest, selfish, and k-deficit mining.

The simulator uses a block-discovery event abstraction: one event chooses the
attacker with probability ``alpha`` and the honest miners otherwise. Rewards
are recorded only when competing branches are resolved. Any unresolved fork at
the finite horizon is settled by longest-chain length; an equal fork is settled
using ``alpha + (1 - alpha) * gamma``, the probability that the next miner would
extend the attacker's branch.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Protocol, Sequence

Strategy = Literal["honest", "selfish", "k_stubborn"]
Phase = Literal["private", "race"]


class RandomSource(Protocol):
    """Small random-number interface that also permits deterministic tests."""

    def random(self) -> float:
        """Return a sample in the half-open interval [0, 1)."""


@dataclass(slots=True)
class MiningState:
    """Mutable state since the most recently resolved common ancestor."""

    private_len: int = 0
    public_len: int = 0
    phase: Phase = "private"

    attacker_main_blocks: int = 0
    honest_main_blocks: int = 0
    attacker_orphan_blocks: int = 0
    honest_orphan_blocks: int = 0

    max_honest_orphan_run: int = 0
    race_events: int = 0
    abandon_events: int = 0
    attacker_win_events: int = 0
    honest_win_events: int = 0

    @property
    def total_main_blocks(self) -> int:
        """Number of blocks committed to the final main chain so far."""

        return self.attacker_main_blocks + self.honest_main_blocks

    @property
    def total_orphan_blocks(self) -> int:
        """Number of resolved stale blocks so far."""

        return self.attacker_orphan_blocks + self.honest_orphan_blocks

    @property
    def classified_blocks(self) -> int:
        """Number of discovered blocks already classified by a resolution."""

        return self.total_main_blocks + self.total_orphan_blocks


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """All requested metrics for one Monte Carlo trajectory."""

    strategy: Strategy
    alpha: float
    gamma: float
    k: int
    num_events: int
    seed: int
    total_main_blocks: int
    attacker_main_blocks: int
    honest_main_blocks: int
    attacker_revenue: float
    honest_revenue: float
    profitable: bool
    attacker_orphan_blocks: int
    honest_orphan_blocks: int
    total_orphan_blocks: int
    orphan_rate: float
    honest_orphan_rate: float
    attacker_orphan_rate: float
    max_honest_orphan_run: int
    race_events: int
    abandon_events: int
    attacker_win_events: int
    honest_win_events: int

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-ready representation of this result."""

        return asdict(self)

    def assert_sane(self) -> None:
        """Check accounting and metric bounds."""

        assert self.attacker_main_blocks + self.honest_main_blocks == (
            self.total_main_blocks
        )
        assert self.attacker_orphan_blocks >= 0
        assert self.honest_orphan_blocks >= 0
        assert self.total_orphan_blocks >= 0
        assert 0.0 <= self.attacker_revenue <= 1.0
        assert 0.0 <= self.honest_revenue <= 1.0
        assert self.total_main_blocks + self.total_orphan_blocks == self.num_events


class MiningSimulator:
    """Explicit state machine for a single strategy and random trajectory."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        alpha: float,
        gamma: float,
        k: int,
        seed: int,
    ) -> None:
        if strategy not in {"honest", "selfish", "k_stubborn"}:
            raise ValueError(f"unknown strategy: {strategy}")
        # The closed interval makes the exact boundary tests unambiguous.
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if k < 1:
            raise ValueError("k must be at least 1")

        self.strategy = strategy
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        self.rng: RandomSource = random.Random(seed)
        self.state = MiningState()
        self.events_processed = 0

    def _assert_state(self) -> None:
        state = self.state
        counts = (
            state.private_len,
            state.public_len,
            state.attacker_main_blocks,
            state.honest_main_blocks,
            state.attacker_orphan_blocks,
            state.honest_orphan_blocks,
        )
        if any(value < 0 for value in counts):
            raise AssertionError("block counts must be nonnegative")
        if state.phase == "race":
            if state.private_len <= 0 or state.private_len != state.public_len:
                raise AssertionError("race branches must be non-empty and equal")
        elif state.phase == "private":
            if state.private_len > 0 and state.private_len == state.public_len:
                raise AssertionError("a non-empty equal fork must be in race phase")
        else:  # pragma: no cover - protected by the Phase type and this class
            raise AssertionError(f"unknown phase: {state.phase}")

        unresolved = state.private_len + state.public_len
        if state.classified_blocks + unresolved != self.events_processed:
            raise AssertionError("every discovered block must be classified or pending")

    def _reset_fork(self) -> None:
        self.state.private_len = 0
        self.state.public_len = 0
        self.state.phase = "private"

    def _enter_race(self) -> None:
        state = self.state
        if state.private_len <= 0 or state.private_len != state.public_len:
            raise AssertionError("cannot publish an unequal fork as a race")
        state.phase = "race"
        state.race_events += 1

    def _attacker_branch_wins(self) -> None:
        """Commit the attacker's branch and orphan the honest branch."""

        state = self.state
        if state.private_len <= 0:
            raise AssertionError("the attacker has no branch to commit")
        state.attacker_main_blocks += state.private_len
        state.honest_orphan_blocks += state.public_len
        state.max_honest_orphan_run = max(state.max_honest_orphan_run, state.public_len)
        state.attacker_win_events += 1
        self._reset_fork()

    def _honest_branch_wins(self, *, abandoned: bool) -> None:
        """Commit the honest branch and orphan the attacker's branch."""

        state = self.state
        if state.public_len <= 0:
            raise AssertionError("the honest miners have no branch to commit")
        state.honest_main_blocks += state.public_len
        state.attacker_orphan_blocks += state.private_len
        state.honest_win_events += 1
        if abandoned:
            state.abandon_events += 1
        self._reset_fork()

    def _honest_strategy_step(self, attacker_found: bool) -> None:
        state = self.state
        if attacker_found:
            state.attacker_main_blocks += 1
            state.attacker_win_events += 1
        else:
            state.honest_main_blocks += 1
            state.honest_win_events += 1

    def _resolve_race(self, attacker_found: bool) -> None:
        """Apply the strategy-specific transition from an equal public race."""

        state = self.state
        if attacker_found:
            state.private_len += 1
            self._attacker_branch_wins()
            return

        if self.rng.random() < self.gamma:
            # An honest miner extends the attacker's published branch. The
            # attacker prefix and the new honest block both become canonical.
            state.attacker_main_blocks += state.private_len
            state.honest_main_blocks += 1
            state.honest_orphan_blocks += state.public_len
            state.max_honest_orphan_run = max(
                state.max_honest_orphan_run, state.public_len
            )
            state.attacker_win_events += 1
            self._reset_fork()
            return

        # The new honest block extends the competing honest race branch.
        state.public_len += 1
        if self.strategy == "selfish":
            self._honest_branch_wins(abandoned=True)
            return

        # A k-deficit attacker continues from the shorter attacker branch until
        # the honest advantage reaches k. For k=1 (the T1 threshold), losing
        # this race causes immediate abandonment at a one-block deficit.
        state.phase = "private"
        if state.public_len - state.private_len >= self.k:
            self._honest_branch_wins(abandoned=True)

    def _selfish_step(self, attacker_found: bool) -> None:
        state = self.state
        if state.phase == "race":
            self._resolve_race(attacker_found)
            return

        if attacker_found:
            state.private_len += 1
            return

        if state.private_len == 0 and state.public_len == 0:
            state.honest_main_blocks += 1
            state.honest_win_events += 1
            return

        state.public_len += 1
        if state.private_len == state.public_len:
            self._enter_race()
        elif state.private_len == state.public_len + 1 and state.private_len >= 2:
            self._attacker_branch_wins()
        elif state.public_len > state.private_len:
            self._honest_branch_wins(abandoned=True)

    def _k_stubborn_step(self, attacker_found: bool) -> None:
        state = self.state
        if state.phase == "race":
            self._resolve_race(attacker_found)
            return

        if attacker_found:
            was_trailing = state.private_len < state.public_len
            state.private_len += 1
            if state.private_len == state.public_len:
                self._enter_race()
            elif was_trailing and state.private_len > state.public_len:
                # Included for completeness if discoveries are later generalized
                # to add more than one block per event. A one-block event reaches
                # a tie before it can overtake from behind.
                self._attacker_branch_wins()
            return

        state.public_len += 1
        if state.private_len == state.public_len:
            self._enter_race()
        elif state.public_len > state.private_len:
            if state.public_len - state.private_len >= self.k:
                self._honest_branch_wins(abandoned=True)
        # If private_len > public_len, the attacker keeps all blocks private.

    def step(self) -> None:
        """Simulate one block discovery and all immediate protocol actions."""

        self._assert_state()
        self.events_processed += 1
        attacker_found = self.rng.random() < self.alpha

        if self.strategy == "honest":
            self._honest_strategy_step(attacker_found)
        elif self.strategy == "selfish":
            self._selfish_step(attacker_found)
        else:
            self._k_stubborn_step(attacker_found)
        self._assert_state()

    def settle(self) -> None:
        """Resolve the finite horizon's fork without mining extra blocks.

        A unique longest branch wins. At equal length, the attacker branch is
        selected with the probability that the next discovery would build on
        it: attacker mining plus gamma-following honest mining.
        """

        self._assert_state()
        state = self.state
        if state.private_len == 0 and state.public_len == 0:
            return

        if state.private_len > state.public_len:
            self._attacker_branch_wins()
        elif state.public_len > state.private_len:
            self._honest_branch_wins(abandoned=False)
        else:
            attacker_probability = self.alpha + (1.0 - self.alpha) * self.gamma
            if self.rng.random() < attacker_probability:
                self._attacker_branch_wins()
            else:
                self._honest_branch_wins(abandoned=False)
        self._assert_state()


def simulate(
    *,
    strategy: Strategy,
    alpha: float,
    gamma: float = 0.0,
    k: int = 1,
    num_events: int = 100_000,
    seed: int = 1,
) -> SimulationResult:
    """Run one simulation and return revenue, orphan, and resolution metrics."""

    if num_events <= 0:
        raise ValueError("num_events must be positive")

    simulator = MiningSimulator(
        strategy=strategy, alpha=alpha, gamma=gamma, k=k, seed=seed
    )
    for _ in range(num_events):
        simulator.step()
    simulator.settle()

    state = simulator.state
    total_main_blocks = state.total_main_blocks
    total_orphan_blocks = state.total_orphan_blocks
    total_blocks = total_main_blocks + total_orphan_blocks
    attacker_revenue = (
        state.attacker_main_blocks / total_main_blocks if total_main_blocks else 0.0
    )
    honest_revenue = (
        state.honest_main_blocks / total_main_blocks if total_main_blocks else 0.0
    )

    result = SimulationResult(
        strategy=strategy,
        alpha=alpha,
        gamma=gamma,
        k=k,
        num_events=num_events,
        seed=seed,
        total_main_blocks=total_main_blocks,
        attacker_main_blocks=state.attacker_main_blocks,
        honest_main_blocks=state.honest_main_blocks,
        attacker_revenue=attacker_revenue,
        honest_revenue=honest_revenue,
        profitable=attacker_revenue > alpha,
        attacker_orphan_blocks=state.attacker_orphan_blocks,
        honest_orphan_blocks=state.honest_orphan_blocks,
        total_orphan_blocks=total_orphan_blocks,
        orphan_rate=total_orphan_blocks / total_blocks,
        # Both per-miner rates use all discovered blocks as the denominator,
        # so their sum is the system orphan rate.
        honest_orphan_rate=state.honest_orphan_blocks / total_blocks,
        attacker_orphan_rate=state.attacker_orphan_blocks / total_blocks,
        max_honest_orphan_run=state.max_honest_orphan_run,
        race_events=state.race_events,
        abandon_events=state.abandon_events,
        attacker_win_events=state.attacker_win_events,
        honest_win_events=state.honest_win_events,
    )
    result.assert_sane()
    return result


def parse_alpha_sweep(specification: str) -> list[float]:
    """Parse comma-separated alphas or an inclusive start:stop:step range."""

    text = specification.strip()
    if not text:
        raise ValueError("alpha sweep must not be empty")

    if ":" not in text:
        try:
            values = [float(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as error:
            raise ValueError("alpha sweep contains a non-numeric value") from error
        if not values:
            raise ValueError("alpha sweep must contain at least one value")
        return values

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("alpha range must have start:stop:step format")
    try:
        start, stop, step = (Decimal(part.strip()) for part in parts)
    except InvalidOperation as error:
        raise ValueError("alpha range contains a non-numeric value") from error
    if step == 0:
        raise ValueError("alpha range step must be nonzero")
    if (stop - start) * step < 0:
        raise ValueError("alpha range step points away from the stop value")

    values: list[float] = []
    value = start
    if step > 0:
        while value <= stop:
            values.append(float(value))
            value += step
    else:
        while value >= stop:
            values.append(float(value))
            value += step
    return values


def parse_k_sweep(specification: str) -> list[int]:
    """Parse a comma-separated list of positive deficit thresholds."""

    try:
        values = [
            int(part.strip()) for part in specification.split(",") if part.strip()
        ]
    except ValueError as error:
        raise ValueError("k sweep contains a non-integer value") from error
    if not values:
        raise ValueError("k sweep must contain at least one value")
    if any(value < 1 for value in values):
        raise ValueError("all k values must be at least 1")
    return values


def write_results_csv(path: Path, results: Sequence[SimulationResult]) -> None:
    """Write one row per simulation run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    if not rows:
        raise ValueError("cannot write an empty result set")
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: Sequence[SimulationResult]) -> None:
    """Print a compact dependency-free summary table."""

    headers = (
        "strategy",
        "alpha",
        "gamma",
        "k",
        "A revenue",
        "H revenue",
        "orphan",
        "main",
        "races",
        "abandons",
        "profit",
    )
    rows = [
        (
            result.strategy,
            f"{result.alpha:.3f}",
            f"{result.gamma:.3f}",
            str(result.k),
            f"{result.attacker_revenue:.5f}",
            f"{result.honest_revenue:.5f}",
            f"{result.orphan_rate:.5f}",
            str(result.total_main_blocks),
            str(result.race_events),
            str(result.abandon_events),
            "yes" if result.profitable else "no",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Simulate honest, selfish, or k-deficit stubborn PoW mining."
    )
    parser.add_argument(
        "--strategy",
        choices=("honest", "selfish", "k_stubborn"),
        default="k_stubborn",
    )
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--num-events", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sweep-alpha",
        help="comma list or inclusive range, for example 0.05:0.45:0.01",
    )
    parser.add_argument(
        "--sweep-k", help="comma-separated positive integers, for example 1,2,3,4"
    )
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI simulations, print the table, and optionally save CSV."""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        alphas = (
            parse_alpha_sweep(arguments.sweep_alpha)
            if arguments.sweep_alpha is not None
            else [arguments.alpha]
        )
        k_values = (
            parse_k_sweep(arguments.sweep_k)
            if arguments.sweep_k is not None
            else [arguments.k]
        )
        results = [
            simulate(
                strategy=arguments.strategy,
                alpha=alpha,
                gamma=arguments.gamma,
                k=k,
                num_events=arguments.num_events,
                seed=arguments.seed,
            )
            for alpha in alphas
            for k in k_values
        ]
    except ValueError as error:
        parser.error(str(error))

    print_summary(results)
    if arguments.output_csv is not None:
        write_results_csv(arguments.output_csv, results)
        print(f"\nWrote {len(results)} row(s) to {arguments.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
