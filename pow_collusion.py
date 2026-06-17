from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    tqdm = None

SCENARIO_ORDER = [
    "AlwaysCartel",
    "BetrayBreakShort",
    "BetrayThenBreak",
    "BetrayTolerated",
]


@dataclass(frozen=True)
class PoolConfig:
    pool_id: str
    hashrate: float


@dataclass(frozen=True)
class BetrayConfig:
    mode: str  # none | short | prob
    traitor_id: Optional[str]
    betray_on_nth_opportunity: int = 3
    betray_start_height: int = 20
    q: float = 0.7
    betray_threshold: int = 10


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    mode: str  # long
    initial_members: tuple[str, ...]
    break_rule: str  # none | dissolve_immediate | kick_immediate | dissolve_threshold | kick_threshold
    betray: BetrayConfig


@dataclass(frozen=True)
class SimConfig:
    T: float
    gamma: float
    runs: int
    target_blocks_long: int
    betray_on_nth_opportunity: int
    betray_start_height: int
    q: float
    betray_threshold: int


@dataclass
class Block:
    id: int
    parent_id: Optional[int]
    height: int
    miner_id: str
    t_publish: float
    is_public: bool = True


@dataclass
class PrivateBlock:
    id: int
    parent_public_id: int
    height: int
    miner_id: str
    t_mine: float


@dataclass
class MiningProcess:
    pool_id: str
    target_tip_id: int
    lambda_rate: float
    target_kind: str  # public | private


@dataclass
class NextEvent:
    t_event: float
    event_type: str  # MINE | RELEASE_CARTEL
    mine_process: Optional[MiningProcess] = None


@dataclass
class RunResult:
    experiment: str
    scenario: str
    mode: str
    traitor_id: Optional[str]
    run_id: int
    seed: int
    T: float
    gamma: float
    q: float
    betray_start_height: int
    betray_threshold: int
    canonical_len: int
    opportunity_count_traitor: int
    betray_count: int
    canon_len_at_betray: Optional[int]
    opportunity_at_betray: Optional[int]
    blocks_by_pool: Dict[str, int]
    shares_by_pool: Dict[str, float]

    def to_row(self, pool_ids: Sequence[str]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "mode": self.mode,
            "traitor_id": "" if self.traitor_id is None else self.traitor_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "T": self.T,
            "gamma": self.gamma,
            "q": self.q,
            "betray_start_height": self.betray_start_height,
            "betray_threshold": self.betray_threshold,
            "canonical_len": self.canonical_len,
            "opportunity_count_traitor": self.opportunity_count_traitor,
            "betray_count": self.betray_count,
            "canon_len_at_betray": (
                "" if self.canon_len_at_betray is None else self.canon_len_at_betray
            ),
            "opportunity_at_betray": (
                "" if self.opportunity_at_betray is None else self.opportunity_at_betray
            ),
        }
        for pool_id in pool_ids:
            row[f"blocks_pool_{pool_id}"] = self.blocks_by_pool.get(pool_id, 0)
            row[f"share_pool_{pool_id}"] = self.shares_by_pool.get(pool_id, 0.0)
        return row


@dataclass
class CartelController:
    members: set[str]
    state: str = "IDLE"  # IDLE | WITHHOLD | RACE
    base_height: Optional[int] = None
    private_bn: Optional[PrivateBlock] = None
    deadline: Optional[float] = None
    race_cartel_tip_id: Optional[int] = None
    race_honest_tip_id: Optional[int] = None

    def cartel_power(self, rates: Dict[str, float]) -> float:
        return sum(rates[pool_id] for pool_id in self.members)

    def start_withhold(
        self, private_bn: PrivateBlock, base_height: int, deadline: float
    ) -> None:
        self.state = "WITHHOLD"
        self.base_height = base_height
        self.private_bn = private_bn
        self.deadline = deadline
        self.race_cartel_tip_id = None
        self.race_honest_tip_id = None

    def enter_race(self, honest_tip_id: int) -> None:
        self.state = "RACE"
        self.deadline = None
        self.race_cartel_tip_id = None
        self.race_honest_tip_id = honest_tip_id

    def reset_round(self) -> None:
        self.state = "IDLE"
        self.base_height = None
        self.private_bn = None
        self.deadline = None
        self.race_cartel_tip_id = None
        self.race_honest_tip_id = None


class CollusionSimulation:
    def __init__(
        self,
        *,
        experiment: str,
        sim_config: SimConfig,
        pools: Sequence[PoolConfig],
        scenario: ScenarioConfig,
        run_id: int,
        seed: int,
        max_events: int,
    ) -> None:
        self.experiment = experiment
        self.sim_config = sim_config
        self.pools = list(pools)
        self.pool_ids = [p.pool_id for p in self.pools]
        self.pool_rates = {p.pool_id: p.hashrate for p in self.pools}
        self.scenario = scenario
        self.run_id = run_id
        self.seed = seed
        self.max_events = max_events

        self.rng = np.random.default_rng(seed)

        self.t = 0.0
        genesis = Block(
            id=0, parent_id=None, height=0, miner_id="GENESIS", t_publish=0.0
        )
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.tips: set[int] = {0}
        self.canonical_tip_id = 0
        self._next_public_id = 1
        self._next_private_id = 1

        self.controller = CartelController(members=set(self.scenario.initial_members))

        self.opportunity_count_traitor = 0
        self.betray_count = 0
        self.betray_triggered = False
        self.break_triggered = False
        self.canon_len_at_betray: Optional[int] = None
        self.opportunity_at_betray: Optional[int] = None

    def canonical_height(self) -> int:
        return self.blocks_by_id[self.canonical_tip_id].height

    def _canonical_key(self, block_id: int) -> tuple[int, float, int]:
        blk = self.blocks_by_id[block_id]
        return (blk.height, -blk.t_publish, -blk.id)

    def get_canonical_tip(self) -> int:
        return max(self.tips, key=self._canonical_key)

    def reconstruct_chain(self, tip_id: Optional[int] = None) -> List[int]:
        node = self.canonical_tip_id if tip_id is None else tip_id
        chain: List[int] = []
        while node is not None and node != 0:
            chain.append(node)
            node = self.blocks_by_id[node].parent_id
        chain.reverse()
        return chain

    def count_blocks_on_chain(self, chain: Sequence[int]) -> Dict[str, int]:
        counts = {pool_id: 0 for pool_id in self.pool_ids}
        for block_id in chain:
            miner = self.blocks_by_id[block_id].miner_id
            if miner in counts:
                counts[miner] += 1
        return counts

    def get_new_canonical_blocks_since(
        self, old_tip_id: int, new_tip_id: int
    ) -> List[int]:
        if old_tip_id == new_tip_id:
            return []
        old_ancestors: set[int] = set()
        node: Optional[int] = old_tip_id
        while node is not None:
            old_ancestors.add(node)
            node = self.blocks_by_id[node].parent_id

        path: List[int] = []
        node = new_tip_id
        while node is not None and node not in old_ancestors:
            path.append(node)
            node = self.blocks_by_id[node].parent_id
        path.reverse()
        return path

    def is_descendant(self, child_id: int, ancestor_id: int) -> bool:
        node: Optional[int] = child_id
        while node is not None:
            if node == ancestor_id:
                return True
            node = self.blocks_by_id[node].parent_id
        return False

    @staticmethod
    def is_traitor_opportunity(
        cartel_state: str,
        mined_height: int,
        public_tip_height: int,
        miner_is_traitor_and_member: bool,
    ) -> bool:
        return (
            miner_is_traitor_and_member
            and cartel_state == "IDLE"
            and mined_height == public_tip_height + 1
        )

    def should_betray_short(self) -> bool:
        return (
            self.opportunity_count_traitor
            == self.scenario.betray.betray_on_nth_opportunity
        )

    def should_betray_prob(self) -> bool:
        if self.canonical_height() < self.scenario.betray.betray_start_height:
            return False
        return bool(self.rng.random() < self.scenario.betray.q)

    def _new_public_id(self) -> int:
        block_id = self._next_public_id
        self._next_public_id += 1
        return block_id

    def _new_private_id(self) -> int:
        block_id = self._next_private_id
        self._next_private_id += 1
        return block_id

    def _w_star(self, p: float) -> float:
        _ = p
        return 10.0 * self.sim_config.T

    def _publish_public_block(
        self, parent_id: int, miner_id: str, t_publish: Optional[float] = None
    ) -> Block:
        t_block = self.t if t_publish is None else t_publish
        block_id = self._new_public_id()
        height = self.blocks_by_id[parent_id].height + 1
        block = Block(
            id=block_id,
            parent_id=parent_id,
            height=height,
            miner_id=miner_id,
            t_publish=t_block,
            is_public=True,
        )
        old_tip = self.canonical_tip_id
        self.blocks_by_id[block_id] = block
        self.tips.add(block_id)
        self.tips.discard(parent_id)
        self.canonical_tip_id = self.get_canonical_tip()
        _ = self.get_new_canonical_blocks_since(old_tip, self.canonical_tip_id)
        return block

    def build_mining_processes(self) -> List[MiningProcess]:
        processes: List[MiningProcess] = []

        race_active = False
        race_honest_tip_id: Optional[int] = None
        if self.controller.state == "RACE":
            race_honest_tip_id = self.controller.race_honest_tip_id
            if (
                self.controller.private_bn is not None
                and race_honest_tip_id is not None
                and race_honest_tip_id in self.tips
            ):
                race_active = True
            else:
                self.controller.reset_round()

        for pool_id in self.pool_ids:
            rate = self.pool_rates[pool_id] / self.sim_config.T
            if rate <= 0.0:
                continue

            if pool_id in self.controller.members:
                if (
                    self.controller.state == "WITHHOLD"
                    and self.controller.private_bn is not None
                ):
                    processes.append(
                        MiningProcess(
                            pool_id=pool_id,
                            target_tip_id=self.controller.private_bn.id,
                            lambda_rate=rate,
                            target_kind="private",
                        )
                    )
                    continue
                if race_active and self.controller.private_bn is not None:
                    processes.append(
                        MiningProcess(
                            pool_id=pool_id,
                            target_tip_id=self.controller.private_bn.id,
                            lambda_rate=rate,
                            target_kind="private",
                        )
                    )
                    continue

            if race_active and race_honest_tip_id is not None:
                processes.append(
                    MiningProcess(
                        pool_id=pool_id,
                        target_tip_id=race_honest_tip_id,
                        lambda_rate=rate,
                        target_kind="public",
                    )
                )
            else:
                processes.append(
                    MiningProcess(
                        pool_id=pool_id,
                        target_tip_id=self.canonical_tip_id,
                        lambda_rate=rate,
                        target_kind="public",
                    )
                )

        return processes

    def sample_next_event(self, processes: Sequence[MiningProcess]) -> NextEvent:
        if not processes:
            raise RuntimeError("No mining processes available.")

        next_mine_t = math.inf
        next_process: Optional[MiningProcess] = None
        for process in processes:
            dt = float(self.rng.exponential(1.0 / process.lambda_rate))
            t_event = self.t + dt
            if t_event < next_mine_t:
                next_mine_t = t_event
                next_process = process

        deadline = self.controller.deadline
        if (
            self.controller.state == "WITHHOLD"
            and deadline is not None
            and deadline <= next_mine_t
        ):
            return NextEvent(t_event=deadline, event_type="RELEASE_CARTEL")

        if next_process is None:
            raise RuntimeError("Failed to sample mining event.")
        return NextEvent(
            t_event=next_mine_t, event_type="MINE", mine_process=next_process
        )

    def check_abort_condition(self) -> None:
        if self.controller.state == "WITHHOLD" and self.controller.private_bn is None:
            self.controller.reset_round()
            return
        if self.controller.state == "RACE" and (
            self.controller.private_bn is None
            or self.controller.race_honest_tip_id is None
        ):
            self.controller.reset_round()

    def break_cartel_dissolve(self) -> None:
        self.controller.members.clear()
        self.controller.reset_round()

    def break_cartel_kick(self, traitor_id: str) -> None:
        self.controller.members.discard(traitor_id)
        self.controller.reset_round()

    def _apply_break_if_needed(self) -> None:
        rule = self.scenario.break_rule
        traitor_id = self.scenario.betray.traitor_id

        if rule == "none":
            return

        should_break = False
        if rule.endswith("_immediate"):
            should_break = True
        elif rule.endswith("_threshold"):
            should_break = self.betray_count >= self.scenario.betray.betray_threshold

        if not should_break:
            return

        if rule.startswith("dissolve"):
            self.break_cartel_dissolve()
        elif rule.startswith("kick"):
            if traitor_id is None:
                raise RuntimeError("kick break rule requires traitor_id")
            self.break_cartel_kick(traitor_id)
        else:
            raise RuntimeError(f"Unknown break rule: {rule}")

        self.break_triggered = True
        if self.canon_len_at_betray is None:
            self.canon_len_at_betray = self.canonical_height()

    def _is_traitor_member(self, miner_id: str) -> bool:
        traitor_id = self.scenario.betray.traitor_id
        return (
            traitor_id is not None
            and miner_id == traitor_id
            and miner_id in self.controller.members
        )

    def _should_betray_after_cartel_mine(self, miner_id: str) -> bool:
        if not self._is_traitor_member(miner_id):
            return False
        if self.scenario.betray.mode == "none":
            return False

        self.opportunity_count_traitor += 1
        if self.scenario.betray.mode == "short":
            return self.should_betray_short()
        if self.scenario.betray.mode == "prob":
            return self.should_betray_prob()
        raise RuntimeError(f"Unknown betray mode: {self.scenario.betray.mode}")

    def _record_betrayal(self) -> None:
        self.betray_count += 1
        self.betray_triggered = True
        self.opportunity_at_betray = self.opportunity_count_traitor
        self.canon_len_at_betray = self.canonical_height()
        self._apply_break_if_needed()

    def _on_member_first_block(self, miner_id: str, parent_id: int) -> None:
        mined_height = self.blocks_by_id[parent_id].height + 1
        public_tip_height = self.canonical_height()

        if (
            mined_height == public_tip_height + 1
            and self._should_betray_after_cartel_mine(miner_id)
        ):
            self._publish_public_block(parent_id=parent_id, miner_id=miner_id)
            self._record_betrayal()
            return

        p_cartel = self.controller.cartel_power(self.pool_rates)
        deadline = self.t + self._w_star(p_cartel)
        private_bn = PrivateBlock(
            id=self._new_private_id(),
            parent_public_id=parent_id,
            height=mined_height,
            miner_id=miner_id,
            t_mine=self.t,
        )
        self.controller.start_withhold(
            private_bn=private_bn,
            base_height=public_tip_height,
            deadline=deadline,
        )

    def _on_private_mine(self, miner_id: str) -> None:
        state = self.controller.state
        if self.controller.private_bn is None:
            if state in {"WITHHOLD", "RACE"}:
                self.controller.reset_round()
            return

        private_bn = self.controller.private_bn
        if state == "WITHHOLD":
            if self._should_betray_after_cartel_mine(miner_id):
                public_bn = self._publish_public_block(
                    parent_id=private_bn.parent_public_id,
                    miner_id=private_bn.miner_id,
                    t_publish=self.t,
                )
                self._publish_public_block(
                    parent_id=public_bn.id,
                    miner_id=miner_id,
                    t_publish=self.t,
                )
                self._record_betrayal()
                self.controller.reset_round()
                return

            public_bn = self._publish_public_block(
                parent_id=private_bn.parent_public_id,
                miner_id=private_bn.miner_id,
                t_publish=self.t,
            )
            p_cartel = self.controller.cartel_power(self.pool_rates)
            new_private_bn = PrivateBlock(
                id=self._new_private_id(),
                parent_public_id=public_bn.id,
                height=public_bn.height + 1,
                miner_id=miner_id,
                t_mine=self.t,
            )
            self.controller.start_withhold(
                private_bn=new_private_bn,
                base_height=public_bn.height,
                deadline=self.t + self._w_star(p_cartel),
            )
            return

        if state == "RACE":
            public_bn = self._publish_public_block(
                parent_id=private_bn.parent_public_id,
                miner_id=private_bn.miner_id,
                t_publish=self.t,
            )
            self._publish_public_block(
                parent_id=public_bn.id,
                miner_id=miner_id,
                t_publish=self.t,
            )
            self.controller.reset_round()

    def _on_public_mine(self, miner_id: str, target_tip_id: int) -> None:
        if self.controller.state == "RACE":
            self._publish_public_block(parent_id=target_tip_id, miner_id=miner_id)
            self.controller.reset_round()
            return

        if (
            self.controller.state == "IDLE"
            and miner_id in self.controller.members
            and target_tip_id == self.canonical_tip_id
        ):
            self._on_member_first_block(miner_id=miner_id, parent_id=target_tip_id)
            return

        block = self._publish_public_block(parent_id=target_tip_id, miner_id=miner_id)
        if (
            self.controller.state == "WITHHOLD"
            and miner_id not in self.controller.members
        ):
            private_bn = self.controller.private_bn
            if private_bn is None:
                self.controller.reset_round()
                return
            if (
                block.height == private_bn.height
                and block.parent_id == private_bn.parent_public_id
            ):
                self.controller.enter_race(honest_tip_id=block.id)
                return
            self.check_abort_condition()

    def on_deadline(self) -> None:
        if self.controller.state != "WITHHOLD" or self.controller.private_bn is None:
            return

        private_bn = self.controller.private_bn
        self._publish_public_block(
            parent_id=private_bn.parent_public_id,
            miner_id=private_bn.miner_id,
            t_publish=self.t,
        )
        self.controller.reset_round()

    def _should_stop(self) -> bool:
        return self.canonical_height() >= self.sim_config.target_blocks_long

    def _summarize(self) -> RunResult:
        canonical_chain = self.reconstruct_chain()
        denominator = self.sim_config.target_blocks_long
        chain_slice = canonical_chain[:denominator]

        if denominator <= 0:
            denominator = len(chain_slice)

        blocks_by_pool = self.count_blocks_on_chain(chain_slice)
        shares_by_pool: Dict[str, float] = {}
        for pool_id in self.pool_ids:
            if denominator > 0:
                shares_by_pool[pool_id] = blocks_by_pool[pool_id] / denominator
            else:
                shares_by_pool[pool_id] = 0.0

        return RunResult(
            experiment=self.experiment,
            scenario=self.scenario.name,
            mode=self.scenario.mode,
            traitor_id=self.scenario.betray.traitor_id,
            run_id=self.run_id,
            seed=self.seed,
            T=self.sim_config.T,
            gamma=self.sim_config.gamma,
            q=self.scenario.betray.q,
            betray_start_height=self.scenario.betray.betray_start_height,
            betray_threshold=self.scenario.betray.betray_threshold,
            canonical_len=denominator,
            opportunity_count_traitor=self.opportunity_count_traitor,
            betray_count=self.betray_count,
            canon_len_at_betray=self.canon_len_at_betray,
            opportunity_at_betray=self.opportunity_at_betray,
            blocks_by_pool=blocks_by_pool,
            shares_by_pool=shares_by_pool,
        )

    def simulate_one_run(self) -> RunResult:
        num_events = 0
        while not self._should_stop():
            if num_events >= self.max_events:
                raise RuntimeError(
                    f"Exceeded max_events={self.max_events}, scenario={self.scenario.name}, run={self.run_id}"
                )
            num_events += 1

            processes = self.build_mining_processes()
            event = self.sample_next_event(processes)
            self.t = event.t_event

            if event.event_type == "RELEASE_CARTEL":
                self.on_deadline()
                continue
            if event.mine_process is None:
                raise RuntimeError("MINE event without process")

            process = event.mine_process
            if process.target_kind == "private":
                self._on_private_mine(miner_id=process.pool_id)
            else:
                self._on_public_mine(
                    miner_id=process.pool_id, target_tip_id=process.target_tip_id
                )

        return self._summarize()


class FastCollusionSimulation(CollusionSimulation):
    def _publish_public_block(
        self, parent_id: int, miner_id: str, t_publish: Optional[float] = None
    ) -> Block:
        t_block = self.t if t_publish is None else t_publish
        block_id = self._new_public_id()
        height = self.blocks_by_id[parent_id].height + 1
        block = Block(
            id=block_id,
            parent_id=parent_id,
            height=height,
            miner_id=miner_id,
            t_publish=t_block,
            is_public=True,
        )
        self.blocks_by_id[block_id] = block
        self.tips.add(block_id)
        self.tips.discard(parent_id)
        self.canonical_tip_id = self.get_canonical_tip()
        return block

    def simulate_one_run(
        self,
        progress_step_percent: int = 10,
        *,
        use_tqdm: bool = False,
    ) -> RunResult:
        num_events = 0
        total = self.sim_config.target_blocks_long
        refresh_blocks = max(1, math.ceil(total * progress_step_percent / 100))
        last_height = self.canonical_height()
        pending_update = 0

        progress_bar = None
        if use_tqdm and tqdm is not None:
            progress_bar = tqdm(
                total=total,
                desc=f"run {self.run_id + 1}",
                unit="blk",
                dynamic_ncols=True,
            )

        try:
            while not self._should_stop():
                if num_events >= self.max_events:
                    raise RuntimeError(
                        f"Exceeded max_events={self.max_events}, scenario={self.scenario.name}, run={self.run_id}"
                    )
                num_events += 1

                processes = self.build_mining_processes()
                event = self.sample_next_event(processes)
                self.t = event.t_event

                if event.event_type == "RELEASE_CARTEL":
                    self.on_deadline()
                else:
                    if event.mine_process is None:
                        raise RuntimeError("MINE event without process")

                    process = event.mine_process
                    if process.target_kind == "private":
                        self._on_private_mine(miner_id=process.pool_id)
                    else:
                        self._on_public_mine(
                            miner_id=process.pool_id,
                            target_tip_id=process.target_tip_id,
                        )

                current_height = self.canonical_height()
                height_delta = max(0, current_height - last_height)
                last_height = current_height
                pending_update += height_delta

                if progress_bar is not None and (
                    pending_update >= refresh_blocks or current_height >= total
                ):
                    progress_bar.update(pending_update)
                    pending_update = 0

            if progress_bar is not None and pending_update > 0:
                progress_bar.update(pending_update)

            return self._summarize()
        finally:
            if progress_bar is not None:
                progress_bar.close()


def parse_pool_spec(spec: str) -> List[PoolConfig]:
    pools: List[PoolConfig] = []
    seen: set[str] = set()
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid pool spec token: {token}")
        pool_id, value = token.split("=", 1)
        pool_id = pool_id.strip()
        if pool_id in seen:
            raise ValueError(f"Duplicated pool id in spec: {pool_id}")
        hashrate = float(value.strip())
        pools.append(PoolConfig(pool_id=pool_id, hashrate=hashrate))
        seen.add(pool_id)
    if not pools:
        raise ValueError("Empty pool spec")
    return pools


def validate_inputs(
    sim_config: SimConfig,
    three_pools: Sequence[PoolConfig],
    three_traitor: str,
) -> None:
    if sim_config.T <= 0:
        raise ValueError("T must be > 0")
    if not (0.0 <= sim_config.gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if sim_config.runs <= 0:
        raise ValueError("runs must be positive")
    if sim_config.target_blocks_long <= 0:
        raise ValueError("target_blocks_long must be positive")
    if sim_config.betray_on_nth_opportunity <= 0:
        raise ValueError("betray_on_nth_opportunity must be positive")
    if sim_config.betray_start_height < 0:
        raise ValueError("betray_start_height must be >= 0")
    if not (0.0 <= sim_config.q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    if sim_config.betray_threshold <= 0:
        raise ValueError("betray_threshold must be positive")

    total = sum(p.hashrate for p in three_pools)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"three hashrates must sum to 1.0, got {total}")
    for pool in three_pools:
        if pool.hashrate < 0.0:
            raise ValueError(
                f"three has negative hashrate: {pool.pool_id}={pool.hashrate}"
            )

    three_ids = {p.pool_id for p in three_pools}
    if {"b", "s", "h"} - three_ids:
        raise ValueError("three_pools must contain ids: b,s,h")
    if three_traitor not in three_ids:
        raise ValueError(f"three_traitor {three_traitor} not in pools")
    if three_traitor not in {"b", "s"}:
        raise ValueError("three_traitor must be one of b,s")


def validate_jobs(jobs: int) -> None:
    if jobs <= 0:
        raise ValueError("jobs must be positive")


def build_scenarios(
    *,
    sim_config: SimConfig,
    experiment: str,
    pools: Sequence[PoolConfig],
    initial_members: Sequence[str],
    traitor_id: str,
) -> List[ScenarioConfig]:
    if experiment != "three":
        raise ValueError(f"Unknown experiment: {experiment}")

    pool_rates = {p.pool_id: p.hashrate for p in pools}
    remaining_members = [pool_id for pool_id in initial_members if pool_id != traitor_id]
    remaining_power = sum(pool_rates[pool_id] for pool_id in remaining_members)

    if remaining_power < 0.5:
        short_break_rule = "dissolve_immediate"
        long_break_rule = "dissolve_threshold"
    else:
        short_break_rule = "kick_immediate"
        long_break_rule = "kick_threshold"

    return [
        ScenarioConfig(
            name="AlwaysCartel",
            mode="long",
            initial_members=tuple(initial_members),
            break_rule="none",
            betray=BetrayConfig(
                mode="none",
                traitor_id=traitor_id,
                betray_on_nth_opportunity=sim_config.betray_on_nth_opportunity,
                betray_start_height=sim_config.betray_start_height,
                q=sim_config.q,
                betray_threshold=sim_config.betray_threshold,
            ),
        ),
        ScenarioConfig(
            name="BetrayBreakShort",
            mode="long",
            initial_members=tuple(initial_members),
            break_rule=short_break_rule,
            betray=BetrayConfig(
                mode="short",
                traitor_id=traitor_id,
                betray_on_nth_opportunity=sim_config.betray_on_nth_opportunity,
                betray_start_height=sim_config.betray_start_height,
                q=sim_config.q,
                betray_threshold=sim_config.betray_threshold,
            ),
        ),
        ScenarioConfig(
            name="BetrayThenBreak",
            mode="long",
            initial_members=tuple(initial_members),
            break_rule=long_break_rule,
            betray=BetrayConfig(
                mode="prob",
                traitor_id=traitor_id,
                betray_on_nth_opportunity=sim_config.betray_on_nth_opportunity,
                betray_start_height=sim_config.betray_start_height,
                q=sim_config.q,
                betray_threshold=sim_config.betray_threshold,
            ),
        ),
        ScenarioConfig(
            name="BetrayTolerated",
            mode="long",
            initial_members=tuple(initial_members),
            break_rule="none",
            betray=BetrayConfig(
                mode="prob",
                traitor_id=traitor_id,
                betray_on_nth_opportunity=sim_config.betray_on_nth_opportunity,
                betray_start_height=sim_config.betray_start_height,
                q=sim_config.q,
                betray_threshold=sim_config.betray_threshold,
            ),
        ),
    ]


def make_seed(seed_base: int, experiment: str, scenario: str, run_id: int) -> int:
    payload = f"{seed_base}:{experiment}:{scenario}:{run_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def write_rows_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def summarize_results(
    rows: Sequence[Dict[str, Any]],
    pools: Sequence[PoolConfig],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    pool_ids = [p.pool_id for p in pools]
    pool_rates = {p.pool_id: p.hashrate for p in pools}

    for pool_id in pool_ids:
        row: Dict[str, Any] = {
            "pool_id": pool_id,
            "hashrate_p": pool_rates[pool_id],
        }
        for scenario_name in SCENARIO_ORDER:
            values = [
                float(item[f"share_pool_{pool_id}"])
                for item in rows
                if item["scenario"] == scenario_name
            ]
            row[f"{scenario_name}_share_mean"] = mean(values) if values else 0.0
            row[f"{scenario_name}_share_std"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(row)

    return summary


def build_display_pool_labels(
    pool_ids: Sequence[str],
    traitor_id: str,
) -> List[str]:
    labels: List[str] = []
    loyal_index = 1
    use_numbered_loyals = len(pool_ids) > 3

    for pool_id in pool_ids:
        if pool_id == "h":
            labels.append("honest")
        elif pool_id == traitor_id:
            labels.append("traitor")
        elif use_numbered_loyals:
            labels.append(f"loyal_{loyal_index}")
            loyal_index += 1
        else:
            labels.append("loyal")

    return labels


def plot_summary(
    summary_rows: Sequence[Dict[str, Any]],
    output_png: Path,
    output_pdf: Path,
    title: str,
    display_pool_labels: Optional[Sequence[str]] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import scienceplots  # noqa: F401
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib unavailable, skip plotting")
        return

    try:
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use(["science", "ieee"])
    plt.rcParams["text.usetex"] = False

    pool_ids = [str(row["pool_id"]) for row in summary_rows]
    x = np.arange(len(pool_ids))
    x_tick_labels = (
        list(display_pool_labels) if display_pool_labels is not None else pool_ids
    )

    series = [
        ("hashrate_p", "Baseline(Hashrate)", "#1f77b4", "", "black"),
        ("AlwaysCartel_share_mean", "NoBetray", "#2ca02c", "", "black"),
        (
            "BetrayBreakShort_share_mean",
            "Betray&NoTolerance",
            "#f28b82",
            "",
            "black",
        ),
        (
            "BetrayThenBreak_share_mean",
            "Betray&ShortTolerance",
            "#c62828",
            "",
            "black",
        ),
        (
            "BetrayTolerated_share_mean",
            "Betray&AlwaysTolerant",
            "#000000",
            "",
            "white",
        ),
    ]

    all_values = [float(row[key]) for row in summary_rows for key, _, _, _, _ in series]
    if all_values:
        y_min = min(all_values) - 0.01
        y_max = math.ceil((max(all_values) + 0.05) * 10.0) / 10.0
    else:
        y_min, y_max = 0.0, 1.0

    width = 0.12
    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    for idx, (key, label, color, hatch, edgecolor) in enumerate(series):
        values = [float(row[key]) for row in summary_rows]
        offset = (idx - 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor=edgecolor,
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("pool", fontsize=12)
    ax.set_ylabel("ratio", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(title)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_experiment(
    *,
    experiment: str,
    pools: Sequence[PoolConfig],
    sim_config: SimConfig,
    scenarios: Sequence[ScenarioConfig],
    seed_base: int,
    max_events: int,
    jobs: int,
    show_progress: bool,
) -> List[RunResult]:
    tasks = [
        (experiment, pools, sim_config, scenario, seed_base, max_events, run_id)
        for scenario in scenarios
        for run_id in range(sim_config.runs)
    ]
    effective_jobs = min(jobs, len(tasks))
    if jobs > effective_jobs:
        print(
            "[info] parallelism is per run, "
            f"so effective_jobs=min(jobs, scenario_count*runs)={effective_jobs}"
        )

    if effective_jobs <= 1:
        results = [_run_single_experiment_task(*task) for task in tasks]
    else:
        try:
            results = _collect_parallel_experiment_results(
                tasks=tasks,
                executor_cls=concurrent.futures.ProcessPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
                progress_desc=f"{experiment} runs",
            )
        except (PermissionError, OSError):
            print(
                "[warn] process-based parallelism is unavailable here; "
                "falling back to ThreadPoolExecutor"
            )
            results = _collect_parallel_experiment_results(
                tasks=tasks,
                executor_cls=concurrent.futures.ThreadPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
                progress_desc=f"{experiment} runs",
            )

    results.sort(
        key=lambda result: (SCENARIO_ORDER.index(result.scenario), int(result.run_id))
    )
    return results


def _run_single_experiment_task(
    experiment: str,
    pools: Sequence[PoolConfig],
    sim_config: SimConfig,
    scenario: ScenarioConfig,
    seed_base: int,
    max_events: int,
    run_id: int,
) -> RunResult:
    seed = make_seed(seed_base, experiment, scenario.name, run_id)
    sim_cls: type[CollusionSimulation]
    if scenario.name == "BetrayTolerated":
        sim_cls = FastCollusionSimulation
    else:
        sim_cls = CollusionSimulation

    sim = sim_cls(
        experiment=experiment,
        sim_config=sim_config,
        pools=pools,
        scenario=scenario,
        run_id=run_id,
        seed=seed,
        max_events=max_events,
    )
    result = sim.simulate_one_run()
    if scenario.name == "BetrayBreakShort":
        expected = sim_config.betray_on_nth_opportunity
        if result.opportunity_at_betray != expected:
            raise RuntimeError(
                "Nth-opportunity sanity check failed: "
                f"opportunity_at_betray={result.opportunity_at_betray}, expected={expected}"
            )
    return result


def _collect_parallel_experiment_results(
    *,
    tasks: Sequence[
        tuple[str, Sequence[PoolConfig], SimConfig, ScenarioConfig, int, int, int]
    ],
    executor_cls: type[concurrent.futures.Executor],
    max_workers: int,
    show_progress: bool,
    progress_desc: str,
) -> List[RunResult]:
    results: List[RunResult] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_single_experiment_task, *task) for task in tasks
        ]
        if tqdm is None or not show_progress:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        else:
            with tqdm(
                total=len(futures),
                desc=progress_desc,
                unit="run",
                dynamic_ncols=True,
            ) as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    progress_bar.update(1)
    return results


def rows_from_results(
    results: Sequence[RunResult], pool_ids: Sequence[str]
) -> List[Dict[str, Any]]:
    return [result.to_row(pool_ids) for result in results]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cartel-OCW collusion simulation")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--target-blocks-long", type=int, default=20160)

    parser.add_argument("--betray-on-nth-opportunity", type=int, default=1)
    parser.add_argument("--betray-start-height", type=int, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--betray-threshold", type=int, default=1000)

    parser.add_argument("--three-pools", type=str, default="b=0.3,s=0.3,h=0.4")
    parser.add_argument("--three-traitor", type=str, default="s")

    parser.add_argument("--seed-base", type=int, default=20260224)
    parser.add_argument("--max-events", type=int, default=2_000_000)
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/collusion"))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--plot-only-from-summary",
        action="store_true",
        help="rebuild plots from existing summary.csv files without rerunning simulations",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress display to reduce terminal overhead",
    )
    return parser


def run_option_a_unit_tests() -> None:
    assert CollusionSimulation.is_traitor_opportunity(
        cartel_state="IDLE",
        mined_height=11,
        public_tip_height=10,
        miner_is_traitor_and_member=True,
    )
    assert not CollusionSimulation.is_traitor_opportunity(
        cartel_state="WITHHOLD",
        mined_height=11,
        public_tip_height=10,
        miner_is_traitor_and_member=True,
    )
    assert not CollusionSimulation.is_traitor_opportunity(
        cartel_state="IDLE",
        mined_height=12,
        public_tip_height=10,
        miner_is_traitor_and_member=True,
    )
    assert not CollusionSimulation.is_traitor_opportunity(
        cartel_state="IDLE",
        mined_height=11,
        public_tip_height=10,
        miner_is_traitor_and_member=False,
    )

    sim_config = SimConfig(
        T=10.0,
        gamma=0.0,
        runs=1,
        target_blocks_long=10,
        betray_on_nth_opportunity=1,
        betray_start_height=0,
        q=1.0,
        betray_threshold=10,
    )
    pools = parse_pool_spec("b=0.33,s=0.33,h=0.34")

    compliant = CollusionSimulation(
        experiment="three",
        sim_config=sim_config,
        pools=pools,
        scenario=ScenarioConfig(
            name="AlwaysCartel",
            mode="long",
            initial_members=("b", "s"),
            break_rule="none",
            betray=BetrayConfig(mode="none", traitor_id="s"),
        ),
        run_id=0,
        seed=1,
        max_events=100,
    )
    compliant._on_member_first_block(miner_id="b", parent_id=0)
    assert compliant.controller.state == "WITHHOLD"
    assert compliant.controller.private_bn is not None
    compliant._on_private_mine(miner_id="s")
    assert compliant.controller.state == "WITHHOLD"
    assert compliant.controller.private_bn is not None
    assert compliant.controller.private_bn.miner_id == "s"
    assert compliant.canonical_height() == 1

    betraying = CollusionSimulation(
        experiment="three",
        sim_config=sim_config,
        pools=pools,
        scenario=ScenarioConfig(
            name="BetrayTolerated",
            mode="long",
            initial_members=("b", "s"),
            break_rule="none",
            betray=BetrayConfig(
                mode="prob",
                traitor_id="s",
                betray_on_nth_opportunity=1,
                betray_start_height=0,
                q=1.0,
                betray_threshold=10,
            ),
        ),
        run_id=0,
        seed=1,
        max_events=100,
    )
    betraying._on_member_first_block(miner_id="b", parent_id=0)
    assert betraying.controller.state == "WITHHOLD"
    betraying._on_private_mine(miner_id="s")
    assert betraying.controller.state == "IDLE"
    assert betraying.canonical_height() == 2
    assert betraying.betray_count == 1
    assert betraying.opportunity_count_traitor == 1
    chain = betraying.reconstruct_chain()
    assert [betraying.blocks_by_id[block_id].miner_id for block_id in chain] == [
        "b",
        "s",
    ]

    low_power_three_pools = parse_pool_spec("b=0.49,s=0.11,h=0.40")
    low_power_three_scenarios = build_scenarios(
        sim_config=sim_config,
        experiment="three",
        pools=low_power_three_pools,
        initial_members=("b", "s"),
        traitor_id="s",
    )
    assert low_power_three_scenarios[1].break_rule == "dissolve_immediate"
    assert low_power_three_scenarios[2].break_rule == "dissolve_threshold"

    low_power_break = CollusionSimulation(
        experiment="three",
        sim_config=sim_config,
        pools=low_power_three_pools,
        scenario=low_power_three_scenarios[1],
        run_id=0,
        seed=1,
        max_events=100,
    )
    low_power_break._on_member_first_block(miner_id="b", parent_id=0)
    low_power_break._on_private_mine(miner_id="s")
    assert low_power_break.controller.members == set()
    assert low_power_break.controller.state == "IDLE"
    assert all(
        process.target_kind == "public"
        and process.target_tip_id == low_power_break.canonical_tip_id
        for process in low_power_break.build_mining_processes()
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_option_a_unit_tests()
    validate_jobs(args.jobs)

    three_pools = parse_pool_spec(args.three_pools)

    sim_config = SimConfig(
        T=args.T,
        gamma=args.gamma,
        runs=args.runs,
        target_blocks_long=args.target_blocks_long,
        betray_on_nth_opportunity=args.betray_on_nth_opportunity,
        betray_start_height=args.betray_start_height,
        q=args.q,
        betray_threshold=args.betray_threshold,
    )

    validate_inputs(
        sim_config=sim_config,
        three_pools=three_pools,
        three_traitor=args.three_traitor,
    )

    scenarios_three = build_scenarios(
        sim_config=sim_config,
        experiment="three",
        pools=three_pools,
        initial_members=("b", "s"),
        traitor_id=args.three_traitor,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only_from_summary:
        summary_three = read_rows_csv(output_dir / "three" / "summary.csv")
        plot_summary(
            summary_rows=summary_three,
            output_png=output_dir / "cartel_three.png",
            output_pdf=output_dir / "cartel_three.pdf",
            title="Three-pool Cartel-OCW Simulation (Traitor above threshold)",
            display_pool_labels=build_display_pool_labels(
                [str(row["pool_id"]) for row in summary_three],
                args.three_traitor,
            ),
        )
        print(f"Plots rebuilt from summary CSVs in: {output_dir}")
        return

    results_three = run_experiment(
        experiment="three",
        pools=three_pools,
        sim_config=sim_config,
        scenarios=scenarios_three,
        seed_base=args.seed_base,
        max_events=args.max_events,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    print("[three] experiment completed")

    three_pool_ids = [p.pool_id for p in three_pools]
    three_rows = rows_from_results(results_three, three_pool_ids)

    base_fields = [
        "experiment",
        "scenario",
        "mode",
        "traitor_id",
        "run_id",
        "seed",
        "T",
        "gamma",
        "q",
        "betray_start_height",
        "betray_threshold",
        "canonical_len",
        "opportunity_count_traitor",
        "betray_count",
        "canon_len_at_betray",
        "opportunity_at_betray",
    ]

    three_fields = (
        base_fields
        + [f"blocks_pool_{pid}" for pid in three_pool_ids]
        + [f"share_pool_{pid}" for pid in three_pool_ids]
    )

    write_rows_csv(output_dir / "three" / "raw_runs.csv", three_rows, three_fields)

    summary_three = summarize_results(three_rows, three_pools)

    summary_fields = ["pool_id", "hashrate_p"]
    for scenario_name in SCENARIO_ORDER:
        summary_fields.append(f"{scenario_name}_share_mean")
        summary_fields.append(f"{scenario_name}_share_std")

    write_rows_csv(output_dir / "three" / "summary.csv", summary_three, summary_fields)

    if args.skip_plots:
        print("[info] --skip-plots enabled, skip png/pdf plotting")
    else:
        plot_summary(
            summary_rows=summary_three,
            output_png=output_dir / "cartel_three.png",
            output_pdf=output_dir / "cartel_three.pdf",
            title="Three-pool Cartel-OCW Simulation (Traitor above threshold)",
            display_pool_labels=build_display_pool_labels(
                three_pool_ids, args.three_traitor
            ),
        )

    config_text = "\n".join(
        [
            f"T={sim_config.T}",
            f"gamma={sim_config.gamma}",
            f"runs={sim_config.runs}",
            f"target_blocks_long={sim_config.target_blocks_long}",
            f"betray_on_nth_opportunity={sim_config.betray_on_nth_opportunity}",
            f"betray_start_height={sim_config.betray_start_height}",
            f"q={sim_config.q}",
            f"betray_threshold={sim_config.betray_threshold}",
            f"three_pools={args.three_pools}",
            f"three_traitor={args.three_traitor}",
        ]
    )
    (output_dir / "experiment_config.txt").write_text(
        config_text + "\n", encoding="utf-8"
    )

    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
