from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class SimConfig:
    T: float = 10.0
    gamma: float = 0.1
    target_blocks: int = 2016
    runs: int = 10
    base_seed: int = 20260223
    results_dir: Path = Path("results_section5")


@dataclass
class BetrayConfig:
    q: float = 0.3
    betray_start_height: int = 200
    betray_threshold: int = 5
    one_shot_betray: bool = True
    one_shot_height: Optional[int] = None


@dataclass
class Scenario:
    scenario_id: str
    experiment_id: str
    hashrates: Dict[str, float]
    cartel_initial_members: Sequence[str]
    traitor_id: Optional[str]
    betray_mode: str  # none | one_shot | probabilistic
    on_betray: str  # none | dissolve | kick
    threshold_action: str  # none | dissolve


@dataclass
class Block:
    block_id: int
    parent_id: Optional[int]
    height: int
    miner_id: str
    t_publish: float


@dataclass
class PrivateBlock:
    block_id: int
    parent_id: int
    height: int
    miner_id: str
    t_mine: float


@dataclass
class Process:
    pool_id: str
    target_tip_id: int
    lambda_rate: float
    is_private: bool = False


@dataclass
class CartelCounters:
    started: int = 0
    success_2blocks: int = 0
    abort: int = 0
    release_only: int = 0
    race_win: int = 0
    race_lose: int = 0


@dataclass
class RunResult:
    scenario_id: str
    experiment_id: str
    run_id: int
    seed: int
    T: float
    gamma: float
    target_blocks: int
    hashrates: Dict[str, float]
    canonical_len: int
    blocks_canonical_by_pool: Dict[str, int]
    share_by_pool: Dict[str, float]
    orphans_by_pool: Dict[str, int]
    betray_count: int
    cartel_end_height: Optional[int]
    num_attacks_started: int
    num_attacks_success_2blocks: int
    num_attacks_abort: int
    num_attacks_release_only: int
    num_attacks_race_win: int
    num_attacks_race_lose: int
    q: float
    betray_start_height: int
    betray_threshold: int

    def to_row(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "T": self.T,
            "gamma": self.gamma,
            "target_blocks": self.target_blocks,
            "hashrates": json.dumps(self.hashrates, sort_keys=True),
            "canonical_len": self.canonical_len,
            "blocks_canonical_by_pool": json.dumps(
                self.blocks_canonical_by_pool, sort_keys=True
            ),
            "share_by_pool": json.dumps(self.share_by_pool, sort_keys=True),
            "orphans_by_pool": json.dumps(self.orphans_by_pool, sort_keys=True),
            "betray_count": self.betray_count,
            "cartel_end_height": (
                "" if self.cartel_end_height is None else self.cartel_end_height
            ),
            "num_attacks_started": self.num_attacks_started,
            "num_attacks_success_2blocks": self.num_attacks_success_2blocks,
            "num_attacks_abort": self.num_attacks_abort,
            "num_attacks_release_only": self.num_attacks_release_only,
            "num_attacks_race_win": self.num_attacks_race_win,
            "num_attacks_race_lose": self.num_attacks_race_lose,
            "q": self.q,
            "betray_start_height": self.betray_start_height,
            "betray_threshold": self.betray_threshold,
        }


class ChainState:
    def __init__(self) -> None:
        genesis = Block(
            block_id=0,
            parent_id=None,
            height=0,
            miner_id="G",
            t_publish=0.0,
        )
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.children_map: Dict[int, List[int]] = {0: []}
        self.tips_set: set[int] = {0}
        self._next_block_id = 1

        self.canonical_tip_id = 0
        self.canonical_chain_ids: List[int] = []

    def new_block_id(self) -> int:
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id

    def _tip_sort_key(self, block_id: int) -> Tuple[int, float, int]:
        block = self.blocks_by_id[block_id]
        # Higher height is better; for ties earlier publish time is better.
        return (block.height, -block.t_publish, -block.block_id)

    def get_canonical_tip(self) -> int:
        return max(self.tips_set, key=self._tip_sort_key)

    def reconstruct_chain(self, tip_id: Optional[int] = None) -> List[int]:
        tip = self.canonical_tip_id if tip_id is None else tip_id
        chain: List[int] = []
        cur = tip
        while cur != 0:
            chain.append(cur)
            parent_id = self.blocks_by_id[cur].parent_id
            if parent_id is None:
                break
            cur = parent_id
        chain.reverse()
        return chain

    def refresh_canonical(self) -> None:
        self.canonical_tip_id = self.get_canonical_tip()
        self.canonical_chain_ids = self.reconstruct_chain(self.canonical_tip_id)

    def publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.block_id] = block
        self.children_map.setdefault(block.block_id, [])
        if block.parent_id is not None:
            self.children_map.setdefault(block.parent_id, []).append(block.block_id)

        self.tips_set.add(block.block_id)
        if block.parent_id is not None and block.parent_id in self.tips_set:
            self.tips_set.remove(block.parent_id)

        self.refresh_canonical()

    @property
    def canonical_len(self) -> int:
        return len(self.canonical_chain_ids)

    @property
    def canonical_height(self) -> int:
        return self.blocks_by_id[self.canonical_tip_id].height

    def is_descendant(self, child_id: int, ancestor_id: int) -> bool:
        cur: Optional[int] = child_id
        while cur is not None:
            if cur == ancestor_id:
                return True
            cur = self.blocks_by_id[cur].parent_id
        return False

    def count_canonical_blocks(self, pool_ids: Sequence[str]) -> Dict[str, int]:
        counts = {pool: 0 for pool in pool_ids}
        for block_id in self.canonical_chain_ids:
            miner_id = self.blocks_by_id[block_id].miner_id
            if miner_id in counts:
                counts[miner_id] += 1
        return counts

    def count_orphans(self, pool_ids: Sequence[str]) -> Dict[str, int]:
        canonical_ids = set(self.canonical_chain_ids)
        counts = {pool: 0 for pool in pool_ids}
        for block in self.blocks_by_id.values():
            if block.block_id == 0:
                continue
            if block.block_id not in canonical_ids and block.miner_id in counts:
                counts[block.miner_id] += 1
        return counts


class CartelController:
    def __init__(
        self, members: Sequence[str], hashrates: Dict[str, float], T: float
    ) -> None:
        self.members: set[str] = set(members)
        self.hashrates = hashrates
        self.T = T

        self.state = "IDLE"  # IDLE | WITHHOLD | RACE
        self.base_height: Optional[int] = None
        self.private_bn: Optional[PrivateBlock] = None
        self.private_bn1: Optional[PrivateBlock] = None
        self.deadline: Optional[float] = None
        self.race_tip_id: Optional[int] = None

        self.counters = CartelCounters()

    def is_member(self, pool_id: str) -> bool:
        return pool_id in self.members

    def cartel_power(self) -> float:
        return sum(self.hashrates[pool_id] for pool_id in self.members)

    def has_active_cartel(self) -> bool:
        return len(self.members) > 0

    def w_star(self) -> float:
        p = self.cartel_power()
        if p <= 0.5 + EPS:
            return 0.0
        return max(-(self.T / p) * math.log(2.0 * (1.0 - p)), 0.0)

    def can_start_tbw(self) -> bool:
        return self.has_active_cartel() and self.w_star() > 0.0

    def _reset_round_only(self) -> None:
        self.state = "IDLE"
        self.base_height = None
        self.private_bn = None
        self.private_bn1 = None
        self.deadline = None
        self.race_tip_id = None

    def dissolve(self) -> None:
        self.members.clear()
        self._reset_round_only()

    def kick_member(self, pool_id: str) -> None:
        self.members.discard(pool_id)
        self._reset_round_only()

    def enter_withhold(self, private_bn: PrivateBlock, t_now: float) -> None:
        self.state = "WITHHOLD"
        self.base_height = private_bn.height - 1
        self.private_bn = private_bn
        self.private_bn1 = None
        self.deadline = t_now + self.w_star()
        self.race_tip_id = None
        self.counters.started += 1

    def publish_double(self, chain: ChainState, t_now: float) -> None:
        if self.private_bn is None or self.private_bn1 is None:
            return

        bn = Block(
            block_id=self.private_bn.block_id,
            parent_id=self.private_bn.parent_id,
            height=self.private_bn.height,
            miner_id=self.private_bn.miner_id,
            t_publish=t_now,
        )
        bn1 = Block(
            block_id=self.private_bn1.block_id,
            parent_id=self.private_bn1.parent_id,
            height=self.private_bn1.height,
            miner_id=self.private_bn1.miner_id,
            t_publish=t_now,
        )

        chain.publish_block(bn)
        chain.publish_block(bn1)

        self.counters.success_2blocks += 1
        self._reset_round_only()

    def on_deadline(self, chain: ChainState, t_now: float) -> None:
        if self.state != "WITHHOLD" or self.deadline is None:
            return
        if self.private_bn is None or self.base_height is None:
            self._reset_round_only()
            return

        if chain.canonical_height >= self.base_height + 2:
            self.counters.abort += 1
            self._reset_round_only()
            return

        bn = Block(
            block_id=self.private_bn.block_id,
            parent_id=self.private_bn.parent_id,
            height=self.private_bn.height,
            miner_id=self.private_bn.miner_id,
            t_publish=t_now,
        )
        chain.publish_block(bn)
        self.counters.release_only += 1

        if chain.canonical_tip_id == bn.block_id:
            self._reset_round_only()
            return

        self.state = "RACE"
        self.private_bn = None
        self.private_bn1 = None
        self.deadline = None
        self.race_tip_id = bn.block_id

    def check_abort(self, chain: ChainState) -> None:
        if self.state != "WITHHOLD" or self.base_height is None:
            return
        if chain.canonical_height >= self.base_height + 2:
            self.counters.abort += 1
            self._reset_round_only()

    def resolve_race_if_finished(self, chain: ChainState) -> bool:
        if self.state != "RACE" or self.base_height is None:
            return False
        if chain.canonical_height < self.base_height + 2:
            return False

        if self.race_tip_id is not None and chain.is_descendant(
            chain.canonical_tip_id, self.race_tip_id
        ):
            self.counters.race_win += 1
        else:
            self.counters.race_lose += 1

        self._reset_round_only()
        return True


class Section5Simulation:
    def __init__(
        self,
        *,
        config: SimConfig,
        scenario: Scenario,
        betray_config: BetrayConfig,
        seed: int,
        run_id: int,
        log_events: bool = False,
        max_events: Optional[int] = None,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.betray_config = betray_config
        self.seed = seed
        self.run_id = run_id
        self.log_events = log_events
        self.max_events = max_events

        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.chain = ChainState()
        self.cartel = CartelController(
            members=scenario.cartel_initial_members,
            hashrates=scenario.hashrates,
            T=config.T,
        )

        self.betray_count = 0
        self.one_shot_done = False
        self.cartel_end_height: Optional[int] = None
        self.event_log: List[Dict[str, Any]] = []

    def _log(self, event: str, **kwargs: Any) -> None:
        if not self.log_events:
            return
        row: Dict[str, Any] = {
            "t": self.t,
            "event": event,
            "state": self.cartel.state,
            "canonical_tip": self.chain.canonical_tip_id,
            "canonical_height": self.chain.canonical_height,
            "betray_count": self.betray_count,
            "members": "|".join(sorted(self.cartel.members)),
        }
        row.update(kwargs)
        self.event_log.append(row)

    def _mine_private_block(self, parent_id: int, miner_id: str) -> PrivateBlock:
        parent_height = self.chain.blocks_by_id[parent_id].height
        return PrivateBlock(
            block_id=self.chain.new_block_id(),
            parent_id=parent_id,
            height=parent_height + 1,
            miner_id=miner_id,
            t_mine=self.t,
        )

    def _publish_public_block(
        self, parent_id: int, miner_id: str, t_publish: float
    ) -> int:
        parent = self.chain.blocks_by_id[parent_id]
        block = Block(
            block_id=self.chain.new_block_id(),
            parent_id=parent_id,
            height=parent.height + 1,
            miner_id=miner_id,
            t_publish=t_publish,
        )
        self.chain.publish_block(block)
        return block.block_id

    def _tip_sort_key(self, tip_id: int) -> Tuple[int, float, int]:
        block = self.chain.blocks_by_id[tip_id]
        return (block.height, -block.t_publish, -block.block_id)

    def _race_tie_tips(self) -> Optional[Tuple[int, int]]:
        if self.cartel.state != "RACE" or self.cartel.race_tip_id is None:
            return None

        cartel_tips = [
            tip
            for tip in self.chain.tips_set
            if self.chain.is_descendant(tip, self.cartel.race_tip_id)
        ]
        non_cartel_tips = [
            tip
            for tip in self.chain.tips_set
            if not self.chain.is_descendant(tip, self.cartel.race_tip_id)
        ]

        if not cartel_tips or not non_cartel_tips:
            return None

        cartel_tip = max(cartel_tips, key=self._tip_sort_key)
        non_cartel_tip = max(non_cartel_tips, key=self._tip_sort_key)

        if (
            self.chain.blocks_by_id[cartel_tip].height
            != self.chain.blocks_by_id[non_cartel_tip].height
        ):
            return None
        return cartel_tip, non_cartel_tip

    def build_mining_processes(self) -> List[Process]:
        processes: List[Process] = []
        tie_tips = self._race_tie_tips()

        for pool_id, hashrate in self.scenario.hashrates.items():
            rate = hashrate / self.config.T
            if rate <= EPS:
                continue

            if self.cartel.is_member(pool_id):
                if (
                    self.cartel.state == "WITHHOLD"
                    and self.cartel.private_bn is not None
                ):
                    processes.append(
                        Process(
                            pool_id=pool_id,
                            target_tip_id=self.cartel.private_bn.block_id,
                            lambda_rate=rate,
                            is_private=True,
                        )
                    )
                elif (
                    self.cartel.state == "RACE" and self.cartel.race_tip_id is not None
                ):
                    processes.append(
                        Process(
                            pool_id=pool_id,
                            target_tip_id=self.cartel.race_tip_id,
                            lambda_rate=rate,
                            is_private=False,
                        )
                    )
                else:
                    processes.append(
                        Process(
                            pool_id=pool_id,
                            target_tip_id=self.chain.canonical_tip_id,
                            lambda_rate=rate,
                            is_private=False,
                        )
                    )
                continue

            if tie_tips is not None:
                cartel_tip, honest_tip = tie_tips
                rate_cartel = self.config.gamma * rate
                rate_honest = (1.0 - self.config.gamma) * rate
                if rate_cartel > EPS:
                    processes.append(
                        Process(
                            pool_id=pool_id,
                            target_tip_id=cartel_tip,
                            lambda_rate=rate_cartel,
                            is_private=False,
                        )
                    )
                if rate_honest > EPS:
                    processes.append(
                        Process(
                            pool_id=pool_id,
                            target_tip_id=honest_tip,
                            lambda_rate=rate_honest,
                            is_private=False,
                        )
                    )
            else:
                processes.append(
                    Process(
                        pool_id=pool_id,
                        target_tip_id=self.chain.canonical_tip_id,
                        lambda_rate=rate,
                        is_private=False,
                    )
                )

        return processes

    def sample_next_event(self, processes: Sequence[Process]) -> Dict[str, Any]:
        best_time = float("inf")
        best_process: Optional[Process] = None

        for process in processes:
            if process.lambda_rate <= EPS:
                continue
            candidate = self.t + float(self.rng.exponential(1.0 / process.lambda_rate))
            if candidate < best_time:
                best_time = candidate
                best_process = process

        deadline = self.cartel.deadline
        if deadline is not None and deadline <= best_time:
            return {"type": "RELEASE_CARTEL", "time": deadline}

        if best_process is None:
            raise RuntimeError("No valid mining process available")

        return {"type": "MINE", "time": best_time, "process": best_process}

    def should_betray(self, pool_id: str) -> bool:
        if self.scenario.traitor_id is None or pool_id != self.scenario.traitor_id:
            return False

        public_height = self.chain.canonical_height

        if self.scenario.betray_mode == "none":
            return False

        if self.scenario.betray_mode == "one_shot":
            if not self.betray_config.one_shot_betray or self.one_shot_done:
                return False
            if (
                self.betray_config.one_shot_height is not None
                and public_height < self.betray_config.one_shot_height
            ):
                return False
            return True

        if self.scenario.betray_mode == "probabilistic":
            if public_height < self.betray_config.betray_start_height:
                return False
            return bool(self.rng.random() < self.betray_config.q)

        raise ValueError(f"Unknown betray mode: {self.scenario.betray_mode}")

    def apply_scenario_consequence(self, betrayer_pool_id: str) -> None:
        if self.scenario.on_betray == "dissolve":
            if self.cartel.has_active_cartel():
                self.cartel.dissolve()
                if self.cartel_end_height is None:
                    self.cartel_end_height = self.chain.canonical_height
        elif self.scenario.on_betray == "kick":
            if self.cartel.has_active_cartel():
                self.cartel.kick_member(betrayer_pool_id)
                if self.cartel_end_height is None:
                    self.cartel_end_height = self.chain.canonical_height

        if (
            self.scenario.threshold_action == "dissolve"
            and self.betray_count >= self.betray_config.betray_threshold
            and self.cartel.has_active_cartel()
        ):
            self.cartel.dissolve()
            if self.cartel_end_height is None:
                self.cartel_end_height = self.chain.canonical_height

    def is_first_block_opportunity(self, process: Process) -> bool:
        return (
            self.cartel.state == "IDLE"
            and process.target_tip_id == self.chain.canonical_tip_id
            and not process.is_private
        )

    def apply_mine_event(self, process: Process) -> None:
        pool_id = process.pool_id

        if process.is_private:
            if self.cartel.state != "WITHHOLD" or self.cartel.private_bn is None:
                return

            bn = self.cartel.private_bn
            bn1 = PrivateBlock(
                block_id=self.chain.new_block_id(),
                parent_id=bn.block_id,
                height=bn.height + 1,
                miner_id=pool_id,
                t_mine=self.t,
            )
            self.cartel.private_bn1 = bn1
            self._log(
                "withhold_second_block",
                pool_id=pool_id,
                bn_id=bn.block_id,
                bn1_id=bn1.block_id,
            )
            self.cartel.publish_double(self.chain, self.t)
            self._log("double_publish", bn_id=bn.block_id, bn1_id=bn1.block_id)
            return

        if self.cartel.is_member(pool_id):
            if self.is_first_block_opportunity(process):
                if self.should_betray(pool_id):
                    block_id = self._publish_public_block(
                        parent_id=process.target_tip_id,
                        miner_id=pool_id,
                        t_publish=self.t,
                    )
                    self.betray_count += 1
                    if self.scenario.betray_mode == "one_shot":
                        self.one_shot_done = True
                    self.apply_scenario_consequence(pool_id)
                    self._log(
                        "betray_publish",
                        pool_id=pool_id,
                        block_id=block_id,
                        betray_count=self.betray_count,
                    )
                    return

                if self.cartel.can_start_tbw():
                    private_bn = self._mine_private_block(
                        parent_id=process.target_tip_id,
                        miner_id=pool_id,
                    )
                    self.cartel.enter_withhold(private_bn, self.t)
                    self._log(
                        "enter_withhold",
                        pool_id=pool_id,
                        private_bn_id=private_bn.block_id,
                        base_height=self.cartel.base_height,
                        deadline=self.cartel.deadline,
                    )
                    return

            block_id = self._publish_public_block(
                parent_id=process.target_tip_id,
                miner_id=pool_id,
                t_publish=self.t,
            )
            self._log("cartel_public_mine", pool_id=pool_id, block_id=block_id)
            return

        block_id = self._publish_public_block(
            parent_id=process.target_tip_id,
            miner_id=pool_id,
            t_publish=self.t,
        )
        self._log(
            "honest_public_mine",
            pool_id=pool_id,
            block_id=block_id,
            target_tip_id=process.target_tip_id,
        )

    def apply_release_event(self) -> None:
        before_state = self.cartel.state
        before_abort = self.cartel.counters.abort
        before_release_only = self.cartel.counters.release_only

        self.cartel.on_deadline(self.chain, self.t)

        if self.cartel.counters.abort > before_abort:
            self._log("release_abort")
        elif self.cartel.counters.release_only > before_release_only:
            if self.cartel.state == "RACE":
                self._log("release_enter_race", race_tip_id=self.cartel.race_tip_id)
            else:
                self._log("release_no_race")
        elif before_state == "WITHHOLD":
            self._log("release_ignored")

    def collect_result(self) -> RunResult:
        pool_ids = list(self.scenario.hashrates.keys())
        canonical_counts = self.chain.count_canonical_blocks(pool_ids)
        orphans_counts = self.chain.count_orphans(pool_ids)
        canonical_len = self.chain.canonical_len

        share = {
            pool: (canonical_counts[pool] / canonical_len if canonical_len > 0 else 0.0)
            for pool in pool_ids
        }

        return RunResult(
            scenario_id=self.scenario.scenario_id,
            experiment_id=self.scenario.experiment_id,
            run_id=self.run_id,
            seed=self.seed,
            T=self.config.T,
            gamma=self.config.gamma,
            target_blocks=self.config.target_blocks,
            hashrates=dict(self.scenario.hashrates),
            canonical_len=canonical_len,
            blocks_canonical_by_pool=canonical_counts,
            share_by_pool=share,
            orphans_by_pool=orphans_counts,
            betray_count=self.betray_count,
            cartel_end_height=self.cartel_end_height,
            num_attacks_started=self.cartel.counters.started,
            num_attacks_success_2blocks=self.cartel.counters.success_2blocks,
            num_attacks_abort=self.cartel.counters.abort,
            num_attacks_release_only=self.cartel.counters.release_only,
            num_attacks_race_win=self.cartel.counters.race_win,
            num_attacks_race_lose=self.cartel.counters.race_lose,
            q=self.betray_config.q,
            betray_start_height=self.betray_config.betray_start_height,
            betray_threshold=self.betray_config.betray_threshold,
        )

    def run(self) -> Tuple[RunResult, List[Dict[str, Any]]]:
        events = 0

        while self.chain.canonical_len < self.config.target_blocks:
            if self.max_events is not None and events >= self.max_events:
                raise RuntimeError("Reached max_events before termination")

            processes = self.build_mining_processes()
            event = self.sample_next_event(processes)
            self.t = event["time"]

            if event["type"] == "RELEASE_CARTEL":
                self.apply_release_event()
            else:
                self.apply_mine_event(event["process"])

            self.cartel.check_abort(self.chain)
            self.cartel.resolve_race_if_finished(self.chain)
            events += 1

        return self.collect_result(), self.event_log


def validate_hashrates(hashrates: Dict[str, float]) -> None:
    if not hashrates:
        raise ValueError("hashrates must not be empty")
    total = 0.0
    for pool_id, value in hashrates.items():
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Invalid hashrate for {pool_id}: {value}")
        total += value
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Hashrates must sum to 1.0, got {total}")


def parse_hashrates(raw: str, expected_pools: Sequence[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid hashrate token: {item}")
        key, val = item.split("=", 1)
        mapping[key.strip()] = float(val.strip())

    expected_set = set(expected_pools)
    got_set = set(mapping.keys())
    if got_set != expected_set:
        raise ValueError(
            f"Expected pools {sorted(expected_set)}, got {sorted(got_set)}"
        )

    validate_hashrates(mapping)
    return mapping


def derive_seed(base_seed: int, scenario_id: str, run_id: int) -> int:
    token = f"{base_seed}|{scenario_id}|{run_id}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scenarios(
    three_hashrates: Dict[str, float],
    four_hashrates: Dict[str, float],
    traitor_id: str,
) -> List[Scenario]:
    scenarios: List[Scenario] = [
        Scenario(
            scenario_id="1.1",
            experiment_id="three_pools",
            hashrates=three_hashrates,
            cartel_initial_members=["b", "s"],
            traitor_id=None,
            betray_mode="none",
            on_betray="none",
            threshold_action="none",
        ),
        Scenario(
            scenario_id="1.2.1",
            experiment_id="three_pools",
            hashrates=three_hashrates,
            cartel_initial_members=["b", "s"],
            traitor_id="b",
            betray_mode="one_shot",
            on_betray="dissolve",
            threshold_action="none",
        ),
        Scenario(
            scenario_id="1.2.2",
            experiment_id="three_pools",
            hashrates=three_hashrates,
            cartel_initial_members=["b", "s"],
            traitor_id="s",
            betray_mode="one_shot",
            on_betray="dissolve",
            threshold_action="none",
        ),
        Scenario(
            scenario_id="1.3",
            experiment_id="three_pools",
            hashrates=three_hashrates,
            cartel_initial_members=["b", "s"],
            traitor_id="s",
            betray_mode="probabilistic",
            on_betray="none",
            threshold_action="dissolve",
        ),
        Scenario(
            scenario_id="2.1",
            experiment_id="four_pools",
            hashrates=four_hashrates,
            cartel_initial_members=["1", "2", "3"],
            traitor_id=None,
            betray_mode="none",
            on_betray="none",
            threshold_action="none",
        ),
        Scenario(
            scenario_id="2.2",
            experiment_id="four_pools",
            hashrates=four_hashrates,
            cartel_initial_members=["1", "2", "3"],
            traitor_id=traitor_id,
            betray_mode="one_shot",
            on_betray="kick",
            threshold_action="none",
        ),
        Scenario(
            scenario_id="2.3",
            experiment_id="four_pools",
            hashrates=four_hashrates,
            cartel_initial_members=["1", "2", "3"],
            traitor_id=traitor_id,
            betray_mode="probabilistic",
            on_betray="none",
            threshold_action="none",
        ),
    ]
    return scenarios


def summarize_runs(raw_results: Sequence[RunResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[RunResult]] = {}
    for result in raw_results:
        for pool_id in result.hashrates.keys():
            key = (result.experiment_id, result.scenario_id, pool_id)
            grouped.setdefault(key, []).append(result)

    summary_rows: List[Dict[str, Any]] = []
    for (_, _, pool_id), runs in sorted(grouped.items()):
        shares = [r.share_by_pool[pool_id] for r in runs]
        share_mean = mean(shares) if shares else 0.0
        summary_rows.append({"share_mean": share_mean})

    return summary_rows


def write_raw_csv(path: Path, raw_results: Sequence[RunResult]) -> None:
    rows = [result.to_row() for result in raw_results]
    if not rows:
        return
    write_csv(path, rows, list(rows[0].keys()))


def write_summary_csv(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(summary_rows)
    if not rows:
        return
    write_csv(path, rows, list(rows[0].keys()))


def save_params_json(
    path: Path,
    *,
    config: SimConfig,
    betray_config: BetrayConfig,
    three_hashrates: Dict[str, float],
    four_hashrates: Dict[str, float],
    traitor_id: str,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "config": {
            "T": config.T,
            "gamma": config.gamma,
            "target_blocks": config.target_blocks,
            "runs": config.runs,
            "base_seed": config.base_seed,
        },
        "betray_config": {
            "q": betray_config.q,
            "betray_start_height": betray_config.betray_start_height,
            "betray_threshold": betray_config.betray_threshold,
            "one_shot_betray": betray_config.one_shot_betray,
            "one_shot_height": betray_config.one_shot_height,
        },
        "three_hashrates": three_hashrates,
        "four_hashrates": four_hashrates,
        "traitor_id": traitor_id,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_event_log(path: Path, event_log: Sequence[Dict[str, Any]]) -> None:
    rows = list(event_log)
    if not rows:
        return
    fields = sorted({field for row in rows for field in row.keys()})
    write_csv(path, rows, fields)


def run_all_experiments(
    *,
    config: SimConfig,
    betray_config: BetrayConfig,
    three_hashrates: Dict[str, float],
    four_hashrates: Dict[str, float],
    traitor_id: str,
    save_sample_event_logs: bool,
) -> Tuple[List[RunResult], List[Dict[str, Any]]]:
    validate_hashrates(three_hashrates)
    validate_hashrates(four_hashrates)

    scenarios = build_scenarios(three_hashrates, four_hashrates, traitor_id)
    raw_results: List[RunResult] = []

    for scenario in scenarios:
        for run_id in range(config.runs):
            seed = derive_seed(config.base_seed, scenario.scenario_id, run_id)
            log_events = (
                save_sample_event_logs
                and run_id == 0
                and scenario.scenario_id
                in {
                    "1.3",
                    "2.3",
                }
            )
            sim = Section5Simulation(
                config=config,
                scenario=scenario,
                betray_config=betray_config,
                seed=seed,
                run_id=run_id,
                log_events=log_events,
            )
            run_result, event_log = sim.run()
            raw_results.append(run_result)

            if event_log:
                write_event_log(
                    config.results_dir
                    / f"event_log_{scenario.scenario_id}_run{run_id}.csv",
                    event_log,
                )

    summary_rows = summarize_runs(raw_results)
    write_raw_csv(config.results_dir / "raw_runs.csv", raw_results)
    write_summary_csv(config.results_dir / "summary.csv", summary_rows)
    save_params_json(
        config.results_dir / "scenario_params.json",
        config=config,
        betray_config=betray_config,
        three_hashrates=three_hashrates,
        four_hashrates=four_hashrates,
        traitor_id=traitor_id,
    )

    return raw_results, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 5 Cartel-TBW simulation")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--target-blocks", type=int, default=2016)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260223)
    parser.add_argument("--q", type=float, default=0.3)
    parser.add_argument("--betray-start-height", type=int, default=200)
    parser.add_argument("--betray-threshold", type=int, default=5)
    parser.add_argument("--one-shot-height", type=int, default=None)
    parser.add_argument("--traitor-id", choices=["1", "2", "3"], default="3")
    parser.add_argument(
        "--three-hashrates",
        default="b=0.4,s=0.25,h=0.35",
        help="comma-separated, e.g. b=0.4,s=0.25,h=0.35",
    )
    parser.add_argument(
        "--four-hashrates",
        default="1=0.27,2=0.27,3=0.27,h=0.19",
        help="comma-separated, e.g. 1=0.27,2=0.27,3=0.27,h=0.19",
    )
    parser.add_argument("--results-dir", default="results_section5")
    parser.add_argument("--save-sample-event-logs", action="store_true")

    args = parser.parse_args()

    config = SimConfig(
        T=args.T,
        gamma=args.gamma,
        target_blocks=args.target_blocks,
        runs=args.runs,
        base_seed=args.base_seed,
        results_dir=Path(args.results_dir),
    )
    betray_config = BetrayConfig(
        q=args.q,
        betray_start_height=args.betray_start_height,
        betray_threshold=args.betray_threshold,
        one_shot_betray=True,
        one_shot_height=args.one_shot_height,
    )

    three_hashrates = parse_hashrates(args.three_hashrates, ["b", "s", "h"])
    four_hashrates = parse_hashrates(args.four_hashrates, ["1", "2", "3", "h"])

    raw_results, _ = run_all_experiments(
        config=config,
        betray_config=betray_config,
        three_hashrates=three_hashrates,
        four_hashrates=four_hashrates,
        traitor_id=args.traitor_id,
        save_sample_event_logs=args.save_sample_event_logs,
    )

    print("Section 5 simulation completed")
    print(f"  runs_per_scenario = {config.runs}")
    print(f"  total_run_rows = {len(raw_results)}")
    print(f"  raw_csv = {config.results_dir / 'raw_runs.csv'}")
    print(f"  summary_csv = {config.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
