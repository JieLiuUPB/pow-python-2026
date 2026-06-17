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

plt = None
_PLOT_IMPORT_TRIED = False


def get_plt():  # pragma: no cover - plotting is optional at runtime
    global plt, _PLOT_IMPORT_TRIED
    if _PLOT_IMPORT_TRIED:
        return plt
    _PLOT_IMPORT_TRIED = True
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt
        import scienceplots  # noqa: F401
    except Exception:
        plt = None
    else:
        plt = _plt
    return plt


EPS = 1e-12
FIXED_TIME_SCENARIO_P_VALUES = [0.65, 0.75, 0.85]
SCENARIO4_P_VALUES = [0.55, 0.65, 0.75, 0.85]
FIXED_TIME_SCENARIO_N_VALUES = [1, 2, 3, 5]
SELFISH_GAMMA = 0.0


@dataclass
class Block:
    id: int
    parent_id: Optional[int]
    height: int
    miner: str
    t_publish: float


@dataclass
class PrivateBlock:
    id: int
    parent_id: int
    height: int
    miner: str
    t_mine: float


@dataclass
class AttackerState:
    state: str = "IDLE"  # IDLE | WITHHOLD | RACE
    base_height: Optional[int] = None
    private_bn: Optional[PrivateBlock] = None
    deadline: Optional[float] = None
    race_tip_id: Optional[int] = None


@dataclass
class AttackCounters:
    started: int = 0
    success_2blocks: int = 0
    success_race: int = 0
    abort: int = 0
    release_only: int = 0


@dataclass
class EpochStat:
    scenario: str
    p: float
    n: int
    run_id: int
    seed: int
    epoch_index: int
    t_start: float
    t_end: float
    t_total: float
    count_basis: str
    difficulty_old: float
    difficulty_new: float
    a_blocks_counted: int
    h_blocks_counted: int


@dataclass
class RunResult:
    scenario: str
    p: float
    run_id: int
    seed: int
    T: float
    t_end: float
    canonical_len: int
    A_blocks_canonical: int
    H_blocks_canonical: int
    A_share: float
    A_orphan_published: int
    H_orphan_published: int
    attacks_started: int
    attacks_success_2blocks: int
    attacks_success_race: int
    attacks_abort: int
    attacks_release_only: int
    final_difficulty: float
    num_epochs_completed: int
    n: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "p": self.p,
            "n": "" if self.n is None else self.n,
            "run_id": self.run_id,
            "seed": self.seed,
            "T": self.T,
            "t_end": self.t_end,
            "canonical_len": self.canonical_len,
            "A_blocks_canonical": self.A_blocks_canonical,
            "H_blocks_canonical": self.H_blocks_canonical,
            "A_share": self.A_share,
            "A_orphan_published": self.A_orphan_published,
            "H_orphan_published": self.H_orphan_published,
            "attacks_started": self.attacks_started,
            "attacks_success_2blocks": self.attacks_success_2blocks,
            "attacks_success_race": self.attacks_success_race,
            "attacks_abort": self.attacks_abort,
            "attacks_release_only": self.attacks_release_only,
            "final_difficulty": self.final_difficulty,
            "num_epochs_completed": self.num_epochs_completed,
        }


@dataclass
class SelfishRunResult:
    scenario: str
    p: float
    gamma: float
    run_id: int
    seed: int
    T: float
    t_end: float
    canonical_len: int
    A_blocks_canonical: int
    H_blocks_canonical: int
    A_share: float
    A_orphan_published: int
    H_orphan_published: int
    published_A_total: int
    published_H_total: int
    hidden_private_blocks_at_stop: int
    in_race_at_stop: bool
    final_difficulty: float
    num_epochs_completed: int
    n: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "p": self.p,
            "gamma": self.gamma,
            "n": "" if self.n is None else self.n,
            "run_id": self.run_id,
            "seed": self.seed,
            "T": self.T,
            "t_end": self.t_end,
            "canonical_len": self.canonical_len,
            "A_blocks_canonical": self.A_blocks_canonical,
            "H_blocks_canonical": self.H_blocks_canonical,
            "A_share": self.A_share,
            "A_orphan_published": self.A_orphan_published,
            "H_orphan_published": self.H_orphan_published,
            "published_A_total": self.published_A_total,
            "published_H_total": self.published_H_total,
            "hidden_private_blocks_at_stop": self.hidden_private_blocks_at_stop,
            "in_race_at_stop": self.in_race_at_stop,
            "final_difficulty": self.final_difficulty,
            "num_epochs_completed": self.num_epochs_completed,
        }


@dataclass
class TimeCheckpointResult:
    scenario: str
    p: float
    n: int
    run_id: int
    seed: int
    t_checkpoint: float
    A_blocks_canonical: int
    H_blocks_canonical: int
    canonical_len: int
    final_difficulty_so_far: float
    num_epochs_completed_so_far: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "p": self.p,
            "n": self.n,
            "run_id": self.run_id,
            "seed": self.seed,
            "t_checkpoint": self.t_checkpoint,
            "A_blocks_canonical": self.A_blocks_canonical,
            "H_blocks_canonical": self.H_blocks_canonical,
            "canonical_len": self.canonical_len,
            "final_difficulty_so_far": self.final_difficulty_so_far,
            "num_epochs_completed_so_far": self.num_epochs_completed_so_far,
        }


class TBWSimulation:
    def __init__(
        self,
        *,
        T: float,
        p: float,
        seed: int,
        mode: str,
        t_end: Optional[float],
        target_blocks: Optional[int],
        enable_daa: bool,
        epoch_len: int,
        scenario: str,
        run_id: int,
        n_value: Optional[int] = None,
        log_events: bool = False,
        max_events: Optional[int] = None,
        daa_count_basis: str = "canonical",
        checkpoint_times: Optional[Sequence[float]] = None,
    ) -> None:
        self.T = float(T)
        self.p = float(p)
        self.seed = int(seed)
        self.mode = mode
        self.t_end = t_end
        self.target_blocks = target_blocks
        self.enable_daa = enable_daa
        self.epoch_len = epoch_len
        self.scenario = scenario
        self.run_id = run_id
        self.n_value = n_value
        self.max_events = max_events
        if daa_count_basis not in {"canonical", "public"}:
            raise ValueError(
                f"daa_count_basis must be 'canonical' or 'public', got {daa_count_basis!r}"
            )
        self.daa_count_basis = daa_count_basis
        self.rng = np.random.default_rng(self.seed)

        self.t = 0.0
        self.difficulty = 1.0

        genesis = Block(id=0, parent_id=None, height=0, miner="G", t_publish=0.0)
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.tips: set[int] = {0}
        self.canonical_tip_id = 0
        self.canonical_chain_ids: List[int] = []
        self.published_block_ids: List[int] = []
        self._next_block_id = 1

        self.attacker = AttackerState()
        self.counters = AttackCounters()

        self.t_epoch_start = 0.0
        self.epochs_completed = 0
        self.epoch_stats: List[EpochStat] = []

        self.log_events = log_events
        self.event_log: List[Dict[str, Any]] = []
        self.checkpoint_times = (
            sorted(float(t) for t in checkpoint_times) if checkpoint_times else []
        )
        self._next_checkpoint_index = 0
        self.checkpoint_results: List[TimeCheckpointResult] = []

    def _checkpoint_n_value(self, t_checkpoint: float) -> int:
        base_horizon = self.epoch_len * self.T
        return int(round(t_checkpoint / base_horizon))

    def _build_checkpoint_result(self, t_checkpoint: float) -> TimeCheckpointResult:
        a_canonical = sum(
            1 for bid in self.canonical_chain_ids if self.blocks_by_id[bid].miner == "A"
        )
        h_canonical = sum(
            1 for bid in self.canonical_chain_ids if self.blocks_by_id[bid].miner == "H"
        )
        return TimeCheckpointResult(
            scenario=self.scenario,
            p=self.p,
            n=self._checkpoint_n_value(t_checkpoint),
            run_id=self.run_id,
            seed=self.seed,
            t_checkpoint=t_checkpoint,
            A_blocks_canonical=a_canonical,
            H_blocks_canonical=h_canonical,
            canonical_len=len(self.canonical_chain_ids),
            final_difficulty_so_far=self.difficulty,
            num_epochs_completed_so_far=self.epochs_completed,
        )

    def _capture_pending_checkpoints_before(self, next_event_time: float) -> None:
        while (
            self._next_checkpoint_index < len(self.checkpoint_times)
            and self.checkpoint_times[self._next_checkpoint_index]
            < next_event_time - EPS
        ):
            t_checkpoint = self.checkpoint_times[self._next_checkpoint_index]
            self.checkpoint_results.append(self._build_checkpoint_result(t_checkpoint))
            self._next_checkpoint_index += 1

    def _capture_due_checkpoints(self) -> None:
        while (
            self._next_checkpoint_index < len(self.checkpoint_times)
            and self.checkpoint_times[self._next_checkpoint_index] <= self.t + EPS
        ):
            t_checkpoint = self.checkpoint_times[self._next_checkpoint_index]
            self.checkpoint_results.append(self._build_checkpoint_result(t_checkpoint))
            self._next_checkpoint_index += 1

    def _log(self, event: str, **kwargs: Any) -> None:
        if not self.log_events:
            return
        row: Dict[str, Any] = {
            "t": self.t,
            "event": event,
            "state": self.attacker.state,
            "canonical_tip": self.canonical_tip_id,
            "canonical_height": self.blocks_by_id[self.canonical_tip_id].height,
        }
        row.update(kwargs)
        self.event_log.append(row)

    def _new_block_id(self) -> int:
        bid = self._next_block_id
        self._next_block_id += 1
        return bid

    def _w_star(self) -> float:
        return 10 * self.T

    def _get_chain_tip_id(self) -> int:
        def key_func(block_id: int) -> tuple[int, float, int]:
            blk = self.blocks_by_id[block_id]
            return (blk.height, -blk.t_publish, -blk.id)

        return max(self.tips, key=key_func)

    def _rebuild_canonical_chain(self, tip_id: Optional[int] = None) -> List[int]:
        tip = self.canonical_tip_id if tip_id is None else tip_id
        chain: List[int] = []
        cur = tip
        while cur != 0:
            chain.append(cur)
            cur = self.blocks_by_id[cur].parent_id  # type: ignore[assignment]
        chain.reverse()
        return chain

    def _refresh_canonical(self) -> None:
        old_tip = self.canonical_tip_id
        self.canonical_tip_id = self._get_chain_tip_id()
        self.canonical_chain_ids = self._rebuild_canonical_chain(self.canonical_tip_id)
        if old_tip != self.canonical_tip_id:
            self._log(
                "reorg_or_tip_change", old_tip=old_tip, new_tip=self.canonical_tip_id
            )

    def _publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.id] = block
        self.published_block_ids.append(block.id)
        self.tips.add(block.id)

        parent = block.parent_id
        if parent is not None and parent in self.tips:
            self.tips.remove(parent)

        self._refresh_canonical()
        self._log(
            "publish",
            block_id=block.id,
            miner=block.miner,
            parent_id=block.parent_id,
            height=block.height,
            t_publish=block.t_publish,
        )

    def _publish_new_block(self, parent_id: int, miner: str, t_publish: float) -> int:
        parent = self.blocks_by_id[parent_id]
        bid = self._new_block_id()
        block = Block(
            id=bid,
            parent_id=parent_id,
            height=parent.height + 1,
            miner=miner,
            t_publish=t_publish,
        )
        self._publish_block(block)
        return bid

    def _is_descendant(self, child_id: int, ancestor_id: int) -> bool:
        cur: Optional[int] = child_id
        while cur is not None:
            if cur == ancestor_id:
                return True
            cur = self.blocks_by_id[cur].parent_id
        return False

    def _reset_attacker(self) -> None:
        self.attacker = AttackerState()

    def _handle_mine_A(self) -> None:
        state = self.attacker.state

        if state == "IDLE":
            parent_tip = self.canonical_tip_id
            parent = self.blocks_by_id[parent_tip]
            private_bn = PrivateBlock(
                id=self._new_block_id(),
                parent_id=parent_tip,
                height=parent.height + 1,
                miner="A",
                t_mine=self.t,
            )
            self.attacker.state = "WITHHOLD"
            self.attacker.base_height = parent.height
            self.attacker.private_bn = private_bn
            self.attacker.deadline = self.t + self._w_star()
            self.attacker.race_tip_id = None
            self.counters.started += 1
            self._log(
                "mine_A_start_withhold",
                base_height=parent.height,
                private_bn_id=private_bn.id,
                deadline=self.attacker.deadline,
            )
            return

        if state == "WITHHOLD":
            bn = self.attacker.private_bn
            if bn is None:
                self._reset_attacker()
                return

            self._publish_block(
                Block(
                    id=bn.id,
                    parent_id=bn.parent_id,
                    height=bn.height,
                    miner="A",
                    t_publish=self.t,
                )
            )

            bn1 = PrivateBlock(
                id=self._new_block_id(),
                parent_id=bn.id,
                height=bn.height + 1,
                miner="A",
                t_mine=self.t,
            )
            self.attacker.base_height = bn.height
            self.attacker.private_bn = bn1
            self.attacker.deadline = self.t + self._w_star()
            self.counters.success_2blocks += 1
            self._log("mine_A_withhold_chain_extend", bn_id=bn.id, bn1_id=bn1.id)
            return

        if state == "RACE":
            bn = self.attacker.private_bn
            if bn is None:
                self.counters.abort += 1
                self._reset_attacker()
                return

            self._publish_block(
                Block(
                    id=bn.id,
                    parent_id=bn.parent_id,
                    height=bn.height,
                    miner="A",
                    t_publish=self.t,
                )
            )
            bnext_id = self._publish_new_block(
                parent_id=bn.id, miner="A", t_publish=self.t
            )
            self.counters.success_race += 1
            self._log("mine_A_race_success", block_id=bnext_id)
            self._reset_attacker()
            return

    def _handle_mine_H(self) -> None:
        state = self.attacker.state
        if state == "RACE":
            race_tip_id = self.attacker.race_tip_id
            if race_tip_id is None:
                self.counters.abort += 1
                self._reset_attacker()
                return
            hid = self._publish_new_block(
                parent_id=race_tip_id, miner="H", t_publish=self.t
            )
            self.attacker.race_tip_id = hid
            self.counters.abort += 1
            self._log("mine_H_race_win", block_id=hid)
            self._reset_attacker()
            return

        parent_tip = self.canonical_tip_id
        hid = self._publish_new_block(parent_id=parent_tip, miner="H", t_publish=self.t)
        self._log("mine_H_publish", block_id=hid)

        if state == "WITHHOLD":
            bn = self.attacker.private_bn
            if bn is None:
                self.counters.abort += 1
                self._reset_attacker()
                return
            hb = self.blocks_by_id[hid]
            if hb.height == bn.height and hb.parent_id == bn.parent_id:
                self.attacker.state = "RACE"
                self.attacker.deadline = None
                self.attacker.race_tip_id = hid
                self._log("mine_H_trigger_race", block_id=hid)

    def _handle_release_A(self) -> None:
        if self.attacker.state != "WITHHOLD" or self.attacker.deadline is None:
            return

        bn = self.attacker.private_bn
        if bn is None:
            self.counters.abort += 1
            self._reset_attacker()
            return

        self._publish_block(
            Block(
                id=bn.id,
                parent_id=bn.parent_id,
                height=bn.height,
                miner="A",
                t_publish=self.t,
            )
        )
        self.counters.release_only += 1
        self._log("release_bn", bn_id=bn.id)
        self._reset_attacker()

    def _check_abort_condition(self) -> None:
        if self.attacker.state == "WITHHOLD" and self.attacker.private_bn is None:
            self.counters.abort += 1
            self._reset_attacker()
            return
        if self.attacker.state == "RACE" and (
            self.attacker.private_bn is None or self.attacker.race_tip_id is None
        ):
            self.counters.abort += 1
            self._reset_attacker()

    def _maybe_adjust_difficulty(self) -> None:
        if not self.enable_daa:
            return

        if self.daa_count_basis == "canonical":
            daa_block_ids = self.canonical_chain_ids
        else:
            daa_block_ids = self.published_block_ids

        while len(daa_block_ids) >= (self.epochs_completed + 1) * self.epoch_len:
            epoch_index = self.epochs_completed + 1
            boundary_len = epoch_index * self.epoch_len
            boundary_block_id = daa_block_ids[boundary_len - 1]
            boundary_time = self.blocks_by_id[boundary_block_id].t_publish

            t_total = max(boundary_time - self.t_epoch_start, EPS)
            d_old = self.difficulty
            d_new = d_old * (self.epoch_len * self.T) / t_total
            self.difficulty = d_new

            segment = daa_block_ids[(epoch_index - 1) * self.epoch_len : boundary_len]
            a_blocks = sum(1 for bid in segment if self.blocks_by_id[bid].miner == "A")
            h_blocks = sum(1 for bid in segment if self.blocks_by_id[bid].miner == "H")

            self.epoch_stats.append(
                EpochStat(
                    scenario=self.scenario,
                    p=self.p,
                    n=self.n_value or 0,
                    run_id=self.run_id,
                    seed=self.seed,
                    epoch_index=epoch_index,
                    t_start=self.t_epoch_start,
                    t_end=boundary_time,
                    t_total=t_total,
                    count_basis=self.daa_count_basis,
                    difficulty_old=d_old,
                    difficulty_new=d_new,
                    a_blocks_counted=a_blocks,
                    h_blocks_counted=h_blocks,
                )
            )
            self._log(
                "daa_adjust",
                epoch_index=epoch_index,
                count_basis=self.daa_count_basis,
                t_total=t_total,
                d_old=d_old,
                d_new=d_new,
            )

            self.t_epoch_start = boundary_time
            self.epochs_completed = epoch_index

    def _terminated(self) -> bool:
        if self.mode == "by_blocks":
            if self.target_blocks is None:
                raise ValueError("target_blocks is required for by_blocks mode")
            return len(self.canonical_chain_ids) >= self.target_blocks

        if self.mode == "by_time":
            if self.t_end is None:
                raise ValueError("t_end is required for by_time mode")
            return self.t >= self.t_end - EPS

        raise ValueError(f"Unknown mode: {self.mode}")

    def _summarize(self) -> RunResult:
        canonical_ids = set(self.canonical_chain_ids)

        a_canonical = 0
        h_canonical = 0
        a_orphans = 0
        h_orphans = 0

        for block in self.blocks_by_id.values():
            if block.id == 0:
                continue
            if block.id in canonical_ids:
                if block.miner == "A":
                    a_canonical += 1
                elif block.miner == "H":
                    h_canonical += 1
            else:
                if block.miner == "A":
                    a_orphans += 1
                elif block.miner == "H":
                    h_orphans += 1

        canonical_len = len(self.canonical_chain_ids)
        a_share = (a_canonical / canonical_len) if canonical_len > 0 else 0.0

        return RunResult(
            scenario=self.scenario,
            p=self.p,
            run_id=self.run_id,
            seed=self.seed,
            T=self.T,
            t_end=self.t,
            canonical_len=canonical_len,
            A_blocks_canonical=a_canonical,
            H_blocks_canonical=h_canonical,
            A_share=a_share,
            A_orphan_published=a_orphans,
            H_orphan_published=h_orphans,
            attacks_started=self.counters.started,
            attacks_success_2blocks=self.counters.success_2blocks,
            attacks_success_race=self.counters.success_race,
            attacks_abort=self.counters.abort,
            attacks_release_only=self.counters.release_only,
            final_difficulty=self.difficulty,
            num_epochs_completed=self.epochs_completed,
            n=self.n_value,
        )

    def run(self) -> tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]:
        events = 0

        while not self._terminated():
            if self.max_events is not None and events >= self.max_events:
                raise RuntimeError("Reached max_events before termination")

            lam = 1.0 / (self.T * self.difficulty)
            lam_a = self.p * lam
            lam_h = (1.0 - self.p) * lam

            t_a = self.t + self.rng.exponential(1.0 / lam_a)
            t_h = self.t + self.rng.exponential(1.0 / lam_h)
            t_r = (
                self.attacker.deadline
                if self.attacker.deadline is not None
                else float("inf")
            )

            t_next = min(t_a, t_h, t_r)
            self._capture_pending_checkpoints_before(t_next)
            if (
                self.mode == "by_time"
                and self.t_end is not None
                and t_next > self.t_end
            ):
                self.t = self.t_end
                self._capture_due_checkpoints()
                break

            self.t = t_next
            if t_r <= t_a and t_r <= t_h:
                self._handle_release_A()
            elif t_a <= t_h:
                self._handle_mine_A()
            else:
                self._handle_mine_H()

            self._check_abort_condition()
            self._maybe_adjust_difficulty()
            self._capture_due_checkpoints()
            events += 1

        return self._summarize(), self.epoch_stats, self.event_log


class SelfishMiningDAASimulation:
    def __init__(
        self,
        *,
        T: float,
        p: float,
        gamma: float,
        seed: int,
        mode: str,
        t_end: Optional[float],
        enable_daa: bool,
        epoch_len: int,
        scenario: str,
        run_id: int,
        n_value: Optional[int] = None,
        daa_count_basis: str = "public",
        checkpoint_times: Optional[Sequence[float]] = None,
    ) -> None:
        self.T = float(T)
        self.p = float(p)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.mode = mode
        self.t_end = t_end
        self.enable_daa = enable_daa
        self.epoch_len = epoch_len
        self.scenario = scenario
        self.run_id = run_id
        self.n_value = n_value
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if daa_count_basis != "public":
            raise ValueError(
                "SelfishMiningDAASimulation currently supports only "
                f"daa_count_basis='public', got {daa_count_basis!r}"
            )
        self.daa_count_basis = daa_count_basis
        self.rng = np.random.default_rng(self.seed)

        self.t = 0.0
        self.difficulty = 1.0

        genesis = Block(id=0, parent_id=None, height=0, miner="G", t_publish=0.0)
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.tips: set[int] = {0}
        self.canonical_tip_id = 0
        self.canonical_chain_ids: List[int] = []
        self.published_block_ids: List[int] = []
        self._next_block_id = 1

        self.private_chain: List[PrivateBlock] = []
        self.in_race = False
        self.race_attacker_tip_id: Optional[int] = None
        self.race_honest_tip_id: Optional[int] = None

        self.t_epoch_start = 0.0
        self.epochs_completed = 0
        self.epoch_stats: List[EpochStat] = []

        self.checkpoint_times = (
            sorted(float(t) for t in checkpoint_times) if checkpoint_times else []
        )
        self._next_checkpoint_index = 0
        self.checkpoint_results: List[TimeCheckpointResult] = []

    def _checkpoint_n_value(self, t_checkpoint: float) -> int:
        base_horizon = self.epoch_len * self.T
        return int(round(t_checkpoint / base_horizon))

    def _build_checkpoint_result(self, t_checkpoint: float) -> TimeCheckpointResult:
        a_canonical = sum(
            1 for bid in self.canonical_chain_ids if self.blocks_by_id[bid].miner == "A"
        )
        h_canonical = sum(
            1 for bid in self.canonical_chain_ids if self.blocks_by_id[bid].miner == "H"
        )
        return TimeCheckpointResult(
            scenario=self.scenario,
            p=self.p,
            n=self._checkpoint_n_value(t_checkpoint),
            run_id=self.run_id,
            seed=self.seed,
            t_checkpoint=t_checkpoint,
            A_blocks_canonical=a_canonical,
            H_blocks_canonical=h_canonical,
            canonical_len=len(self.canonical_chain_ids),
            final_difficulty_so_far=self.difficulty,
            num_epochs_completed_so_far=self.epochs_completed,
        )

    def _capture_pending_checkpoints_before(self, next_event_time: float) -> None:
        while (
            self._next_checkpoint_index < len(self.checkpoint_times)
            and self.checkpoint_times[self._next_checkpoint_index]
            < next_event_time - EPS
        ):
            t_checkpoint = self.checkpoint_times[self._next_checkpoint_index]
            self.checkpoint_results.append(self._build_checkpoint_result(t_checkpoint))
            self._next_checkpoint_index += 1

    def _capture_due_checkpoints(self) -> None:
        while (
            self._next_checkpoint_index < len(self.checkpoint_times)
            and self.checkpoint_times[self._next_checkpoint_index] <= self.t + EPS
        ):
            t_checkpoint = self.checkpoint_times[self._next_checkpoint_index]
            self.checkpoint_results.append(self._build_checkpoint_result(t_checkpoint))
            self._next_checkpoint_index += 1

    def _new_block_id(self) -> int:
        bid = self._next_block_id
        self._next_block_id += 1
        return bid

    def _get_chain_tip_id(self) -> int:
        def key_func(block_id: int) -> tuple[int, float, int]:
            blk = self.blocks_by_id[block_id]
            return (blk.height, -blk.t_publish, -blk.id)

        return max(self.tips, key=key_func)

    def _rebuild_canonical_chain(self, tip_id: Optional[int] = None) -> List[int]:
        tip = self.canonical_tip_id if tip_id is None else tip_id
        chain: List[int] = []
        cur = tip
        while cur != 0:
            chain.append(cur)
            cur = self.blocks_by_id[cur].parent_id  # type: ignore[assignment]
        chain.reverse()
        return chain

    def _refresh_canonical(self) -> None:
        self.canonical_tip_id = self._get_chain_tip_id()
        self.canonical_chain_ids = self._rebuild_canonical_chain(self.canonical_tip_id)

    def _publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.id] = block
        self.published_block_ids.append(block.id)
        self.tips.add(block.id)

        parent = block.parent_id
        if parent is not None and parent in self.tips:
            self.tips.remove(parent)

        self._refresh_canonical()

    def _publish_new_block(self, parent_id: int, miner: str, t_publish: float) -> int:
        parent = self.blocks_by_id[parent_id]
        bid = self._new_block_id()
        block = Block(
            id=bid,
            parent_id=parent_id,
            height=parent.height + 1,
            miner=miner,
            t_publish=t_publish,
        )
        self._publish_block(block)
        return bid

    def _publish_private_prefix(self, count: int) -> List[int]:
        prefix = self.private_chain[:count]
        published_ids: List[int] = []
        for private_block in prefix:
            self._publish_block(
                Block(
                    id=private_block.id,
                    parent_id=private_block.parent_id,
                    height=private_block.height,
                    miner=private_block.miner,
                    t_publish=self.t,
                )
            )
            published_ids.append(private_block.id)
        self.private_chain = self.private_chain[count:]
        return published_ids

    def _mine_private_block(self) -> None:
        if self.private_chain:
            parent_id = self.private_chain[-1].id
            parent_height = self.private_chain[-1].height
        else:
            parent_id = self.canonical_tip_id
            parent_height = self.blocks_by_id[parent_id].height

        self.private_chain.append(
            PrivateBlock(
                id=self._new_block_id(),
                parent_id=parent_id,
                height=parent_height + 1,
                miner="A",
                t_mine=self.t,
            )
        )

    def _handle_mine_A(self) -> None:
        if self.in_race:
            if self.race_attacker_tip_id is None:
                raise RuntimeError("race attacker tip is missing")
            self._publish_new_block(
                parent_id=self.race_attacker_tip_id, miner="A", t_publish=self.t
            )
            self.in_race = False
            self.race_attacker_tip_id = None
            self.race_honest_tip_id = None
            return

        self._mine_private_block()

    def _handle_mine_H(self) -> None:
        if self.in_race:
            if self.race_attacker_tip_id is None or self.race_honest_tip_id is None:
                raise RuntimeError("race tips are missing")
            parent_id = (
                self.race_attacker_tip_id
                if self.rng.random() < self.gamma
                else self.race_honest_tip_id
            )
            self._publish_new_block(parent_id=parent_id, miner="H", t_publish=self.t)
            self.in_race = False
            self.race_attacker_tip_id = None
            self.race_honest_tip_id = None
            return

        lead = len(self.private_chain)
        if lead == 0:
            self._publish_new_block(
                parent_id=self.canonical_tip_id, miner="H", t_publish=self.t
            )
            return

        honest_tip_id = self._publish_new_block(
            parent_id=self.canonical_tip_id, miner="H", t_publish=self.t
        )
        if lead == 1:
            attacker_tip_id = self._publish_private_prefix(1)[0]
            self.in_race = True
            self.race_attacker_tip_id = attacker_tip_id
            self.race_honest_tip_id = honest_tip_id
            return

        if lead == 2:
            self._publish_private_prefix(2)
            return

        self._publish_private_prefix(1)

    def _maybe_adjust_difficulty(self) -> None:
        if not self.enable_daa:
            return

        daa_block_ids = self.published_block_ids
        while len(daa_block_ids) >= (self.epochs_completed + 1) * self.epoch_len:
            epoch_index = self.epochs_completed + 1
            boundary_len = epoch_index * self.epoch_len
            boundary_block_id = daa_block_ids[boundary_len - 1]
            boundary_time = self.blocks_by_id[boundary_block_id].t_publish

            t_total = max(boundary_time - self.t_epoch_start, EPS)
            d_old = self.difficulty
            d_new = d_old * (self.epoch_len * self.T) / t_total
            self.difficulty = d_new

            segment = daa_block_ids[(epoch_index - 1) * self.epoch_len : boundary_len]
            a_blocks = sum(1 for bid in segment if self.blocks_by_id[bid].miner == "A")
            h_blocks = sum(1 for bid in segment if self.blocks_by_id[bid].miner == "H")

            self.epoch_stats.append(
                EpochStat(
                    scenario=self.scenario,
                    p=self.p,
                    n=self.n_value or 0,
                    run_id=self.run_id,
                    seed=self.seed,
                    epoch_index=epoch_index,
                    t_start=self.t_epoch_start,
                    t_end=boundary_time,
                    t_total=t_total,
                    count_basis=self.daa_count_basis,
                    difficulty_old=d_old,
                    difficulty_new=d_new,
                    a_blocks_counted=a_blocks,
                    h_blocks_counted=h_blocks,
                )
            )

            self.t_epoch_start = boundary_time
            self.epochs_completed = epoch_index

    def _terminated(self) -> bool:
        if self.mode != "by_time":
            raise ValueError(
                f"Unsupported mode for selfish DAA simulation: {self.mode}"
            )
        if self.t_end is None:
            raise ValueError("t_end is required for by_time mode")
        return self.t >= self.t_end - EPS

    def _summarize(self) -> SelfishRunResult:
        canonical_ids = set(self.canonical_chain_ids)

        a_canonical = 0
        h_canonical = 0
        a_orphans = 0
        h_orphans = 0
        a_published = 0
        h_published = 0

        for block in self.blocks_by_id.values():
            if block.id == 0:
                continue
            if block.miner == "A":
                a_published += 1
            elif block.miner == "H":
                h_published += 1

            if block.id in canonical_ids:
                if block.miner == "A":
                    a_canonical += 1
                elif block.miner == "H":
                    h_canonical += 1
            else:
                if block.miner == "A":
                    a_orphans += 1
                elif block.miner == "H":
                    h_orphans += 1

        canonical_len = len(self.canonical_chain_ids)
        a_share = (a_canonical / canonical_len) if canonical_len > 0 else 0.0

        return SelfishRunResult(
            scenario=self.scenario,
            p=self.p,
            gamma=self.gamma,
            run_id=self.run_id,
            seed=self.seed,
            T=self.T,
            t_end=self.t,
            canonical_len=canonical_len,
            A_blocks_canonical=a_canonical,
            H_blocks_canonical=h_canonical,
            A_share=a_share,
            A_orphan_published=a_orphans,
            H_orphan_published=h_orphans,
            published_A_total=a_published,
            published_H_total=h_published,
            hidden_private_blocks_at_stop=len(self.private_chain),
            in_race_at_stop=self.in_race,
            final_difficulty=self.difficulty,
            num_epochs_completed=self.epochs_completed,
            n=self.n_value,
        )

    def run(
        self,
    ) -> tuple[SelfishRunResult, List[EpochStat], List[TimeCheckpointResult]]:
        while not self._terminated():
            lam = 1.0 / (self.T * self.difficulty)
            lam_a = self.p * lam
            lam_h = (1.0 - self.p) * lam

            t_a = self.t + self.rng.exponential(1.0 / lam_a)
            t_h = self.t + self.rng.exponential(1.0 / lam_h)
            t_next = min(t_a, t_h)
            self._capture_pending_checkpoints_before(t_next)
            if self.t_end is not None and t_next > self.t_end:
                self.t = self.t_end
                self._capture_due_checkpoints()
                break

            self.t = t_next
            if t_a <= t_h:
                self._handle_mine_A()
            else:
                self._handle_mine_H()

            self._maybe_adjust_difficulty()
            self._capture_due_checkpoints()

        return self._summarize(), self.epoch_stats, self.checkpoint_results


def derive_seed(
    base_seed: int, scenario: str, p: float, run_id: int, n_value: Optional[int]
) -> int:
    token = f"{base_seed}|{scenario}|{p:.8f}|{run_id}|{n_value if n_value is not None else 'NA'}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_metric(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def validate_jobs(jobs: int) -> None:
    if jobs <= 0:
        raise ValueError("jobs must be positive")


def _run_tbw_task(
    T: float,
    p: float,
    seed: int,
    mode: str,
    t_end: Optional[float],
    target_blocks: Optional[int],
    enable_daa: bool,
    epoch_len: int,
    scenario: str,
    run_id: int,
    n_value: Optional[int],
    log_events: bool,
    max_events: Optional[int],
    daa_count_basis: str,
    checkpoint_times: Optional[Sequence[float]],
) -> tuple[
    RunResult,
    List[EpochStat],
    List[Dict[str, Any]],
    List[TimeCheckpointResult],
]:
    sim = TBWSimulation(
        T=T,
        p=p,
        seed=seed,
        mode=mode,
        t_end=t_end,
        target_blocks=target_blocks,
        enable_daa=enable_daa,
        epoch_len=epoch_len,
        scenario=scenario,
        run_id=run_id,
        n_value=n_value,
        log_events=log_events,
        max_events=max_events,
        daa_count_basis=daa_count_basis,
        checkpoint_times=checkpoint_times,
    )
    run_result, epoch_stats, event_log = sim.run()
    return run_result, epoch_stats, event_log, sim.checkpoint_results


def _run_selfish_daa_task(
    T: float,
    p: float,
    gamma: float,
    seed: int,
    t_end: float,
    epoch_len: int,
    scenario: str,
    run_id: int,
    n_value: int,
    daa_count_basis: str,
    checkpoint_times: Optional[Sequence[float]],
) -> tuple[SelfishRunResult, List[EpochStat], List[TimeCheckpointResult]]:
    sim = SelfishMiningDAASimulation(
        T=T,
        p=p,
        gamma=gamma,
        seed=seed,
        mode="by_time",
        t_end=t_end,
        enable_daa=True,
        epoch_len=epoch_len,
        scenario=scenario,
        run_id=run_id,
        n_value=n_value,
        daa_count_basis=daa_count_basis,
        checkpoint_times=checkpoint_times,
    )
    return sim.run()


def _collect_parallel_tbw_results(
    tasks: Sequence[
        tuple[
            float,
            float,
            int,
            str,
            Optional[float],
            Optional[int],
            bool,
            int,
            str,
            int,
            Optional[int],
            bool,
            Optional[int],
            str,
            Optional[Sequence[float]],
        ]
    ],
    *,
    executor_cls: type[concurrent.futures.Executor],
    max_workers: int,
    show_progress: bool,
    progress_desc: str,
) -> List[
    tuple[
        RunResult,
        List[EpochStat],
        List[Dict[str, Any]],
        List[TimeCheckpointResult],
    ]
]:
    results: List[
        tuple[
            RunResult,
            List[EpochStat],
            List[Dict[str, Any]],
            List[TimeCheckpointResult],
        ]
    ] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_tbw_task, *task) for task in tasks]
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


def _collect_parallel_selfish_results(
    tasks: Sequence[
        tuple[
            float,
            float,
            float,
            int,
            float,
            int,
            str,
            int,
            int,
            str,
            Optional[Sequence[float]],
        ]
    ],
    *,
    executor_cls: type[concurrent.futures.Executor],
    max_workers: int,
    show_progress: bool,
    progress_desc: str,
) -> List[tuple[SelfishRunResult, List[EpochStat], List[TimeCheckpointResult]]]:
    results: List[
        tuple[SelfishRunResult, List[EpochStat], List[TimeCheckpointResult]]
    ] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_selfish_daa_task, *task) for task in tasks]
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


def _execute_tbw_tasks(
    tasks: Sequence[
        tuple[
            float,
            float,
            int,
            str,
            Optional[float],
            Optional[int],
            bool,
            int,
            str,
            int,
            Optional[int],
            bool,
            Optional[int],
            str,
            Optional[Sequence[float]],
        ]
    ],
    *,
    jobs: int,
    show_progress: bool,
    progress_desc: str,
) -> List[
    tuple[
        RunResult,
        List[EpochStat],
        List[Dict[str, Any]],
        List[TimeCheckpointResult],
    ]
]:
    effective_jobs = min(jobs, len(tasks))
    if jobs > effective_jobs:
        print(
            "[info] parallelism is per run, "
            f"so effective_jobs=min(jobs, task_count)={effective_jobs}"
        )

    if effective_jobs <= 1:
        return [_run_tbw_task(*task) for task in tasks]

    try:
        return _collect_parallel_tbw_results(
            tasks,
            executor_cls=concurrent.futures.ProcessPoolExecutor,
            max_workers=effective_jobs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    except (PermissionError, OSError):
        print(
            "[warn] process-based parallelism is unavailable here; "
            "falling back to ThreadPoolExecutor"
        )
        return _collect_parallel_tbw_results(
            tasks,
            executor_cls=concurrent.futures.ThreadPoolExecutor,
            max_workers=effective_jobs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )


def _execute_selfish_daa_tasks(
    tasks: Sequence[
        tuple[
            float,
            float,
            float,
            int,
            float,
            int,
            str,
            int,
            int,
            str,
            Optional[Sequence[float]],
        ]
    ],
    *,
    jobs: int,
    show_progress: bool,
    progress_desc: str,
) -> List[tuple[SelfishRunResult, List[EpochStat], List[TimeCheckpointResult]]]:
    effective_jobs = min(jobs, len(tasks))
    if jobs > effective_jobs:
        print(
            "[info] parallelism is per run, "
            f"so effective_jobs=min(jobs, task_count)={effective_jobs}"
        )

    if effective_jobs <= 1:
        return [_run_selfish_daa_task(*task) for task in tasks]

    try:
        return _collect_parallel_selfish_results(
            tasks,
            executor_cls=concurrent.futures.ProcessPoolExecutor,
            max_workers=effective_jobs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    except (PermissionError, OSError):
        print(
            "[warn] process-based parallelism is unavailable here; "
            "falling back to ThreadPoolExecutor"
        )
        return _collect_parallel_selfish_results(
            tasks,
            executor_cls=concurrent.futures.ThreadPoolExecutor,
            max_workers=effective_jobs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )


def scenario3(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    results_root: Path,
    jobs: int,
    show_progress: bool,
) -> tuple[List[RunResult], List[EpochStat], List[Dict[str, Any]]]:
    return run_fixed_time_daa_scenario(
        T=T,
        runs=runs,
        epoch_len=epoch_len,
        base_seed=base_seed,
        results_root=results_root,
        jobs=jobs,
        show_progress=show_progress,
        scenario_name="scenario3_daa_by_time",
        daa_count_basis="canonical",
        progress_desc="scenario3 runs",
    )


def build_fixed_time_ratio_summary_rows(
    *,
    checkpoint_results: Sequence[TimeCheckpointResult],
    p_values: Sequence[float],
    n_values: Sequence[int],
    metric_name: str,
    runs: int,
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for p in p_values:
        for n_value in n_values:
            baseline = p * n_value * 2016.0
            values = [
                checkpoint.A_blocks_canonical / baseline
                for checkpoint in checkpoint_results
                if abs(checkpoint.p - p) < 1e-12 and checkpoint.n == n_value
            ]
            m, s = aggregate_metric(values)
            summary_rows.append(
                {
                    "p": p,
                    "n": n_value,
                    "metric": metric_name,
                    "metric_mean": m,
                    "metric_std": s,
                    "runs": runs,
                }
            )
    return summary_rows


def scenario4(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    results_root: Path,
    jobs: int,
    show_progress: bool,
) -> tuple[List[RunResult], List[EpochStat], List[Dict[str, Any]]]:
    return run_fixed_time_daa_scenario(
        T=T,
        runs=runs,
        epoch_len=epoch_len,
        base_seed=base_seed,
        results_root=results_root,
        jobs=jobs,
        show_progress=show_progress,
        scenario_name="scenario4_public_daa_by_time",
        daa_count_basis="public",
        progress_desc="scenario4 runs",
        p_values=SCENARIO4_P_VALUES,
    )


def scenario4_selfish(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    gamma: float,
    results_root: Path,
    jobs: int,
    show_progress: bool,
) -> tuple[List[SelfishRunResult], List[EpochStat], List[Dict[str, Any]]]:
    p_values = SCENARIO4_P_VALUES
    n_values = FIXED_TIME_SCENARIO_N_VALUES
    scenario_name = "scenario4_selfish_public_daa_by_time"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)
    max_n = max(n_values)
    t_end = max_n * epoch_len * T
    checkpoint_times = [n_value * epoch_len * T for n_value in n_values]

    tasks = []
    for p in p_values:
        for run_id in range(runs):
            tasks.append(
                (
                    T,
                    p,
                    gamma,
                    derive_seed(base_seed, scenario_name, p, run_id, max_n),
                    t_end,
                    epoch_len,
                    scenario_name,
                    run_id,
                    max_n,
                    "public",
                    checkpoint_times,
                )
            )

    task_results = _execute_selfish_daa_tasks(
        tasks,
        jobs=jobs,
        show_progress=show_progress,
        progress_desc="scenario4 selfish runs",
    )

    raw_results: List[SelfishRunResult] = []
    epoch_rows: List[EpochStat] = []
    checkpoint_rows: List[TimeCheckpointResult] = []
    for run_result, epochs, checkpoints in task_results:
        raw_results.append(run_result)
        epoch_rows.extend(epochs)
        checkpoint_rows.extend(checkpoints)

    raw_results.sort(key=lambda r: (float(r.p), int(r.run_id)))
    epoch_rows.sort(
        key=lambda e: (float(e.p), int(e.n), int(e.run_id), int(e.epoch_index))
    )
    checkpoint_rows.sort(key=lambda c: (float(c.p), int(c.n), int(c.run_id)))

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    write_epoch_stats_csv(out_dir, epoch_rows)
    write_checkpoint_csv(out_dir, checkpoint_rows)

    summary_rows = build_fixed_time_ratio_summary_rows(
        checkpoint_results=checkpoint_rows,
        p_values=p_values,
        n_values=n_values,
        metric_name="A_blocks_canonical_fixedtime_n_round_over_pn2016",
        runs=runs,
    )
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["p", "n", "metric", "metric_mean", "metric_std", "runs"],
    )
    return raw_results, epoch_rows, summary_rows


def write_epoch_stats_csv(out_dir: Path, epoch_rows: Sequence[EpochStat]) -> None:
    epoch_dict_rows = [
        {
            "scenario": e.scenario,
            "p": e.p,
            "n": e.n,
            "run_id": e.run_id,
            "seed": e.seed,
            "epoch_index": e.epoch_index,
            "t_start": e.t_start,
            "t_end": e.t_end,
            "t_total": e.t_total,
            "count_basis": e.count_basis,
            "difficulty_old": e.difficulty_old,
            "difficulty_new": e.difficulty_new,
            "a_blocks_counted": e.a_blocks_counted,
            "h_blocks_counted": e.h_blocks_counted,
        }
        for e in epoch_rows
    ]
    epoch_fields = (
        list(epoch_dict_rows[0].keys())
        if epoch_dict_rows
        else [
            "scenario",
            "p",
            "n",
            "run_id",
            "seed",
            "epoch_index",
            "t_start",
            "t_end",
            "t_total",
            "count_basis",
            "difficulty_old",
            "difficulty_new",
            "a_blocks_counted",
            "h_blocks_counted",
        ]
    )
    write_csv(out_dir / "epoch_stats.csv", epoch_dict_rows, epoch_fields)


def write_checkpoint_csv(
    out_dir: Path, checkpoint_rows: Sequence[TimeCheckpointResult]
) -> None:
    checkpoint_dict_rows = [row.to_dict() for row in checkpoint_rows]
    checkpoint_fields = (
        list(checkpoint_dict_rows[0].keys())
        if checkpoint_dict_rows
        else [
            "scenario",
            "p",
            "n",
            "run_id",
            "seed",
            "t_checkpoint",
            "A_blocks_canonical",
            "H_blocks_canonical",
            "canonical_len",
            "final_difficulty_so_far",
            "num_epochs_completed_so_far",
        ]
    )
    write_csv(out_dir / "checkpoints.csv", checkpoint_dict_rows, checkpoint_fields)


def run_fixed_time_daa_scenario(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    results_root: Path,
    jobs: int,
    show_progress: bool,
    scenario_name: str,
    daa_count_basis: str,
    progress_desc: str,
    p_values: Optional[Sequence[float]] = None,
) -> tuple[List[RunResult], List[EpochStat], List[Dict[str, Any]]]:
    p_values = list(FIXED_TIME_SCENARIO_P_VALUES if p_values is None else p_values)
    n_values = FIXED_TIME_SCENARIO_N_VALUES
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)
    max_n = max(n_values)
    checkpoint_times = [n_value * epoch_len * T for n_value in n_values]

    tasks = []
    for p in p_values:
        t_end = max_n * epoch_len * T
        for run_id in range(runs):
            tasks.append(
                (
                    T,
                    p,
                    derive_seed(base_seed, scenario_name, p, run_id, max_n),
                    "by_time",
                    t_end,
                    None,
                    True,
                    epoch_len,
                    scenario_name,
                    run_id,
                    max_n,
                    False,
                    None,
                    daa_count_basis,
                    checkpoint_times,
                )
            )

    task_results = _execute_tbw_tasks(
        tasks,
        jobs=jobs,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    raw_results: List[RunResult] = []
    epoch_rows: List[EpochStat] = []
    checkpoint_rows: List[TimeCheckpointResult] = []
    for run_result, epochs, _, checkpoints in task_results:
        raw_results.append(run_result)
        epoch_rows.extend(epochs)
        checkpoint_rows.extend(checkpoints)

    raw_results.sort(key=lambda r: (float(r.p), int(r.run_id)))
    epoch_rows.sort(
        key=lambda e: (float(e.p), int(e.n), int(e.run_id), int(e.epoch_index))
    )
    checkpoint_rows.sort(key=lambda c: (float(c.p), int(c.n), int(c.run_id)))

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    write_epoch_stats_csv(out_dir, epoch_rows)
    write_checkpoint_csv(out_dir, checkpoint_rows)

    summary_rows = build_fixed_time_ratio_summary_rows(
        checkpoint_results=checkpoint_rows,
        p_values=p_values,
        n_values=n_values,
        metric_name="A_blocks_canonical_fixedtime_n_round_over_pn2016",
        runs=runs,
    )
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["p", "n", "metric", "metric_mean", "metric_std", "runs"],
    )
    return raw_results, epoch_rows, summary_rows


def apply_plot_theme(plt_mod: Any) -> None:
    try:
        plt_mod.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt_mod.style.use(["science", "ieee"])
    plt_mod.rcParams["text.usetex"] = False


def save_figure_bundle(fig_dir: Path, stem: str, fig: Any) -> None:
    ensure_dir(fig_dir)
    png_path = fig_dir / f"{stem}.png"
    pdf_path = fig_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")


def save_plot_data(fig_dir: Path, stem: str, rows: Sequence[Dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    write_csv(
        fig_dir / f"{stem}_plot_data.csv",
        row_list,
        list(row_list[0].keys()),
    )


def plot_fixed_time_ratio(
    summary_rows: List[Dict[str, Any]],
    fig_dir: Path,
    *,
    stem: str,
    title: str,
    param_symbol: str = r"\alpha",
) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    p_values = sorted({float(row["p"]) for row in summary_rows})
    x_ticks = sorted({int(row["n"]) for row in summary_rows})
    palette = ["#155eef", "#0f766e", "#b42318", "#7a5af8", "#dd6b20"]
    plot_data_rows: List[Dict[str, Any]] = []

    for idx, p in enumerate(p_values):
        rows = sorted(
            [r for r in summary_rows if abs(float(r["p"]) - p) < 1e-12],
            key=lambda x: int(x["n"]),
        )
        x = np.array([int(r["n"]) for r in rows], dtype=int)
        y = np.array([float(r["metric_mean"]) for r in rows], dtype=float)
        yerr = np.array([float(r["metric_std"]) for r in rows], dtype=float)
        color = palette[idx % len(palette)]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2.0,
            elinewidth=1.4,
            color=color,
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=1.4,
            label=rf"${param_symbol}$={p:.2f}",
        )
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.08)
        for i in range(len(x)):
            plot_data_rows.append(
                {
                    "p": p,
                    "n": int(x[i]),
                    "metric_mean": float(y[i]),
                    "metric_std": float(yerr[i]),
                    "baseline_ratio": 1.0,
                }
            )

    ax.axhline(
        1.0, linestyle="--", color="#344054", linewidth=1.6, label="baseline = 1"
    )
    ax.set_xlabel("Round n", fontsize=12)
    ax.set_ylabel("mean( A_share_sim / A_share_honest )", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xticks(x_ticks)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", ncols=2, fontsize=10)
    ax.margins(x=0.03)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    save_figure_bundle(fig_dir, stem, fig)
    plt_mod.show()
    plt_mod.close(fig)
    save_plot_data(fig_dir, stem, plot_data_rows)


def plot_scenario3(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plot_fixed_time_ratio(
        summary_rows,
        fig_dir,
        stem="s3_fixedtime_ratio",
        title="Fixed-Time Ratio with Canonical-Chain DAA",
        param_symbol=r"\alpha",
    )


def plot_scenario4(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plot_fixed_time_ratio(
        summary_rows,
        fig_dir,
        stem="s4_fixedtime_ratio",
        title="Fixed-Time Ratio with Orphan-Aware DAA",
        param_symbol=r"\alpha",
    )


def plot_scenario4_comparison(
    tbw_summary_rows: Sequence[Dict[str, Any]],
    selfish_summary_rows: Sequence[Dict[str, Any]],
    fig_dir: Path,
) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    all_rows = list(tbw_summary_rows) + list(selfish_summary_rows)
    if not all_rows:
        return

    p_values = sorted({float(row["p"]) for row in all_rows})
    x_ticks = sorted({int(row["n"]) for row in all_rows})
    nrows = 2
    ncols = 2
    fig, axes = plt_mod.subplots(
        nrows,
        ncols,
        figsize=(5.6 * ncols, 4.35 * nrows),
        squeeze=False,
        sharey=True,
    )
    palette = {"OCW": "#155eef", "SM": "#b42318"}
    markers = {"OCW": "o", "SM": "s"}
    strategy_order = ["OCW", "SM"]
    plot_data_rows: List[Dict[str, Any]] = []

    for idx, p in enumerate(p_values):
        ax = axes[idx // ncols][idx % ncols]
        for strategy in strategy_order:
            source_rows = (
                tbw_summary_rows if strategy == "OCW" else selfish_summary_rows
            )
            rows = sorted(
                [row for row in source_rows if abs(float(row["p"]) - p) < 1e-12],
                key=lambda row: int(row["n"]),
            )
            if not rows:
                continue

            x = np.array([int(row["n"]) for row in rows], dtype=int)
            y = np.array([float(row["metric_mean"]) for row in rows], dtype=float)
            yerr = np.array([float(row["metric_std"]) for row in rows], dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers[strategy],
                capsize=4,
                linewidth=2.0,
                elinewidth=1.4,
                color=palette[strategy],
                markersize=3.5,
                markerfacecolor="none",
                markeredgewidth=1.4,
                label=strategy,
            )
            ax.fill_between(x, y - yerr, y + yerr, color=palette[strategy], alpha=0.08)

            for i in range(len(x)):
                plot_data_rows.append(
                    {
                        "strategy": strategy,
                        "p": p,
                        "n": int(x[i]),
                        "metric_mean": float(y[i]),
                        "metric_std": float(yerr[i]),
                    }
                )

        ax.set_xticks(x_ticks)
        ax.set_title(rf"$\alpha$={p:.2f}", fontsize=14)
        ax.set_xlabel("Round n", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.margins(x=0.03)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.axhline(
            1.0, linestyle="--", color="#344054", linewidth=1.6, label="baseline = 1"
        )
        ax.legend(loc="best", fontsize=11)

    axes[0][0].set_ylabel("mean( A_share_sim / A_share_honest )", fontsize=13)
    axes[1][0].set_ylabel("mean( A_share_sim / A_share_honest )", fontsize=13)
    fig.suptitle("OCW vs SM under Orphan-Aware DAA", fontsize=17)
    fig.tight_layout()
    save_figure_bundle(fig_dir, "s4_fixedtime_ratio_ocw_vs_sm", fig)
    plt_mod.show()
    plt_mod.close(fig)
    save_plot_data(fig_dir, "s4_fixedtime_ratio_ocw_vs_sm", plot_data_rows)


def write_config_summary(
    *,
    path: Path,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    jobs: int,
    selfish_gamma: float,
) -> None:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    lines = [
        "TBW PoW Simulation Configuration",
        "================================",
        f"T = {T}",
        f"runs = {runs}",
        f"epoch_len = {epoch_len}",
        f"base_seed = {base_seed}",
        f"jobs = {jobs}",
        f"selfish_gamma = {selfish_gamma}",
        "gamma = 0",
        "DAA: D_new = D_old * (2016*T) / T_total",
        "w*(p, D) = 10*T",
        "w* values:",
    ]
    for p in p_values:
        w_star = 10.0 * T
        lines.append(f"  p={p:.2f}: w*={w_star:.6f}")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_scenarios(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ["3", "4"]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    for p in parts:
        if p not in {"3", "4"}:
            raise ValueError(f"Invalid scenario selector: {p}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="TBW PoW simulation runner")
    parser.add_argument(
        "--scenarios", default="all", help="all or comma-separated subset of 3,4"
    )
    parser.add_argument("--T", type=float, default=10.0, help="Target block interval")
    parser.add_argument(
        "--runs", type=int, default=10, help="Independent runs per configuration"
    )
    parser.add_argument(
        "--epoch-len", type=int, default=2016, help="Epoch length in canonical blocks"
    )
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed")
    parser.add_argument("--jobs", type=int, default=30, help="Parallel workers")
    parser.add_argument(
        "--results-dir", default="results", help="Results output directory"
    )
    parser.add_argument(
        "--figures-dir", default="figures", help="Figure output directory"
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress display to reduce terminal overhead",
    )
    parser.add_argument(
        "--selfish-gamma",
        type=float,
        default=SELFISH_GAMMA,
        help="tie-breaking gamma for selfish mining in scenario 4 comparison",
    )

    args = parser.parse_args()
    validate_jobs(args.jobs)

    scenarios = parse_scenarios(args.scenarios)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    write_config_summary(
        path=results_dir / "experiment_config.txt",
        T=args.T,
        runs=args.runs,
        epoch_len=args.epoch_len,
        base_seed=args.base_seed,
        jobs=args.jobs,
        selfish_gamma=args.selfish_gamma,
    )

    print("TBW simulation config summary")
    print(
        f"  T={args.T}, runs={args.runs}, epoch_len={args.epoch_len}, "
        f"base_seed={args.base_seed}, jobs={args.jobs}"
    )
    print("  gamma=0, private lead<=1, Poisson mining, tie by earlier t_publish")
    print(f"  selfish mining gamma={args.selfish_gamma}")
    print("  DAA formula: D_new = D_old * (2016*T) / T_total")

    s3_summary: List[Dict[str, Any]] = []
    s4_summary: List[Dict[str, Any]] = []
    s4_selfish_summary: List[Dict[str, Any]] = []
    if "3" in scenarios:
        _, _, s3_summary = scenario3(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            results_root=results_dir,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        print("Scenario 3 completed")

    if "4" in scenarios:
        _, _, s4_summary = scenario4(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            results_root=results_dir,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        _, _, s4_selfish_summary = scenario4_selfish(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            gamma=args.selfish_gamma,
            results_root=results_dir,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        print("Scenario 4 completed")
        print("Scenario 4 selfish comparison completed")

    if not args.skip_plots:
        if get_plt() is None:
            print("matplotlib unavailable, skipped plotting")
        else:
            if s3_summary:
                plot_scenario3(s3_summary, figures_dir)
            if s4_summary:
                plot_scenario4(s4_summary, figures_dir)
            if s4_summary and s4_selfish_summary:
                plot_scenario4_comparison(
                    s4_summary,
                    s4_selfish_summary,
                    figures_dir,
                )
            print("Figures saved")


if __name__ == "__main__":
    main()
