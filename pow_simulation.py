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
from tqdm import tqdm

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
    difficulty_old: float
    difficulty_new: float
    a_blocks_canonical: int
    h_blocks_canonical: int


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
        self.rng = np.random.default_rng(self.seed)

        self.t = 0.0
        self.difficulty = 1.0

        genesis = Block(id=0, parent_id=None, height=0, miner="G", t_publish=0.0)
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.tips: set[int] = {0}
        self.canonical_tip_id = 0
        self.canonical_chain_ids: List[int] = []
        self._next_block_id = 1

        self.attacker = AttackerState()
        self.counters = AttackCounters()

        self.t_epoch_start = 0.0
        self.epochs_completed = 0
        self.epoch_stats: List[EpochStat] = []

        self.log_events = log_events
        self.event_log: List[Dict[str, Any]] = []

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
        return 10.0 * self.T

    def _canonical_key(self, block_id: int) -> tuple[int, float, int]:
        blk = self.blocks_by_id[block_id]
        return (blk.height, -blk.t_publish, -blk.id)

    def _is_better_tip(self, candidate_id: int, current_id: int) -> bool:
        return self._canonical_key(candidate_id) > self._canonical_key(current_id)

    def _get_chain_tip_id(self) -> int:
        return max(self.tips, key=self._canonical_key)

    def _rebuild_canonical_chain(self, tip_id: Optional[int] = None) -> List[int]:
        tip = self.canonical_tip_id if tip_id is None else tip_id
        chain: List[int] = []
        cur = tip
        while cur != 0:
            chain.append(cur)
            cur = self.blocks_by_id[cur].parent_id  # type: ignore[assignment]
        chain.reverse()
        return chain

    def _refresh_canonical_chain(self, old_tip_id: int, new_tip_id: int) -> None:
        if old_tip_id == new_tip_id:
            return
        if old_tip_id == 0 and not self.canonical_chain_ids:
            self.canonical_chain_ids = self._rebuild_canonical_chain(new_tip_id)
            return

        old_ancestors: set[int] = set()
        node: Optional[int] = old_tip_id
        while node is not None:
            old_ancestors.add(node)
            node = self.blocks_by_id[node].parent_id

        appended_path: List[int] = []
        node = new_tip_id
        while node is not None and node not in old_ancestors:
            appended_path.append(node)
            node = self.blocks_by_id[node].parent_id

        common_ancestor = node
        if common_ancestor is None:
            self.canonical_chain_ids = self._rebuild_canonical_chain(new_tip_id)
            return

        while (
            self.canonical_chain_ids and self.canonical_chain_ids[-1] != common_ancestor
        ):
            self.canonical_chain_ids.pop()

        self.canonical_chain_ids.extend(reversed(appended_path))

    def _prune_inactive_tips(self) -> None:
        self.tips = {self.canonical_tip_id}

    def _publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.id] = block
        self.tips.add(block.id)

        parent = block.parent_id
        if parent is not None and parent in self.tips:
            self.tips.remove(parent)

        old_tip = self.canonical_tip_id
        if self._is_better_tip(block.id, self.canonical_tip_id):
            self.canonical_tip_id = block.id
            self._refresh_canonical_chain(old_tip, self.canonical_tip_id)
            self._log(
                "reorg_or_tip_change", old_tip=old_tip, new_tip=self.canonical_tip_id
            )
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
        self._prune_inactive_tips()

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

            bn_pub = Block(
                id=bn.id,
                parent_id=bn.parent_id,
                height=bn.height,
                miner="A",
                t_publish=self.t,
            )
            self._publish_block(bn_pub)

            bn1_id = self._publish_new_block(
                parent_id=bn.id, miner="A", t_publish=self.t
            )
            self.counters.success_2blocks += 1
            self._log("mine_A_withhold_success", bn_id=bn.id, bn1_id=bn1_id)
            self._reset_attacker()
            return

        if state == "RACE":
            race_tip_id = self.attacker.race_tip_id
            if race_tip_id is None:
                race_tip_id = self.canonical_tip_id

            bnext_id = self._publish_new_block(
                parent_id=race_tip_id, miner="A", t_publish=self.t
            )
            self.counters.success_race += 1
            self._log("mine_A_race_success", block_id=bnext_id)
            self._reset_attacker()
            return

    def _handle_mine_H(self) -> None:
        parent_tip = self.canonical_tip_id
        hid = self._publish_new_block(parent_id=parent_tip, miner="H", t_publish=self.t)
        self._log("mine_H_publish", block_id=hid)

        if self.attacker.state == "RACE" and self.attacker.base_height is not None:
            canonical_tip = self.blocks_by_id[self.canonical_tip_id]
            if canonical_tip.height >= self.attacker.base_height + 2:
                if self.attacker.race_tip_id is not None and not self._is_descendant(
                    self.canonical_tip_id, self.attacker.race_tip_id
                ):
                    self._log("race_lost")
                    self._reset_attacker()

    def _handle_release_A(self) -> None:
        if self.attacker.state != "WITHHOLD" or self.attacker.deadline is None:
            return

        if (
            self.blocks_by_id[self.canonical_tip_id].height
            >= (self.attacker.base_height or 0) + 2
        ):
            self.counters.abort += 1
            self._log("release_deadline_abort")
            self._reset_attacker()
            return

        bn = self.attacker.private_bn
        if bn is None:
            self._reset_attacker()
            return

        bn_pub = Block(
            id=bn.id,
            parent_id=bn.parent_id,
            height=bn.height,
            miner="A",
            t_publish=self.t,
        )
        self._publish_block(bn_pub)
        self.counters.release_only += 1
        self._log("release_bn", bn_id=bn.id)

        if self.canonical_tip_id == bn.id:
            self._log("release_no_race")
            self._reset_attacker()
            return

        self.attacker.state = "RACE"
        self.attacker.private_bn = None
        self.attacker.deadline = None
        self.attacker.race_tip_id = bn.id
        self._log("release_enter_race", race_tip_id=bn.id)

    def _check_abort_condition(self) -> None:
        if self.attacker.state != "WITHHOLD" or self.attacker.base_height is None:
            return

        canonical_height = self.blocks_by_id[self.canonical_tip_id].height
        if canonical_height >= self.attacker.base_height + 2:
            self.counters.abort += 1
            self._log("withhold_abort_honest_advanced")
            self._reset_attacker()

    def _maybe_adjust_difficulty(self) -> None:
        if not self.enable_daa:
            return

        while (
            len(self.canonical_chain_ids)
            >= (self.epochs_completed + 1) * self.epoch_len
        ):
            epoch_index = self.epochs_completed + 1
            boundary_len = epoch_index * self.epoch_len
            boundary_block_id = self.canonical_chain_ids[boundary_len - 1]
            boundary_time = self.blocks_by_id[boundary_block_id].t_publish

            t_total = max(boundary_time - self.t_epoch_start, EPS)
            d_old = self.difficulty
            d_new = d_old * (self.epoch_len * self.T) / t_total
            self.difficulty = d_new

            segment = self.canonical_chain_ids[
                (epoch_index - 1) * self.epoch_len : boundary_len
            ]
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
                    difficulty_old=d_old,
                    difficulty_new=d_new,
                    a_blocks_canonical=a_blocks,
                    h_blocks_canonical=h_blocks,
                )
            )
            self._log(
                "daa_adjust",
                epoch_index=epoch_index,
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

    def run(
        self,
        progress_step_percent: int = 10,
        *,
        use_tqdm: bool = False,
        progress_desc: Optional[str] = None,
    ) -> tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]:
        events = 0
        progress_bar = None
        last_progress_value = 0.0

        if use_tqdm:
            if self.mode == "by_blocks":
                if self.target_blocks is None:
                    raise ValueError("target_blocks is required for by_blocks mode")
                progress_total = float(self.target_blocks)
                progress_unit = "blk"
            elif self.mode == "by_time":
                if self.t_end is None:
                    raise ValueError("t_end is required for by_time mode")
                progress_total = float(self.t_end)
                progress_unit = "t"
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            progress_bar = tqdm(
                total=progress_total,
                desc=progress_desc or f"{self.scenario}: run {self.run_id + 1}",
                unit=progress_unit,
                dynamic_ncols=True,
                leave=False,
            )

        try:
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
                if (
                    self.mode == "by_time"
                    and self.t_end is not None
                    and t_next > self.t_end
                ):
                    self.t = self.t_end
                    if progress_bar is not None:
                        progress_bar.update(self.t_end - last_progress_value)
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
                events += 1

                if progress_bar is not None:
                    if self.mode == "by_blocks":
                        progress_value = float(len(self.canonical_chain_ids))
                        refresh_amount = max(
                            1.0,
                            math.ceil(
                                float(self.target_blocks or 1)
                                * progress_step_percent
                                / 100.0
                            ),
                        )
                    else:
                        progress_value = min(float(self.t), float(self.t_end or self.t))
                        refresh_amount = max(
                            1.0,
                            (float(self.t_end or 1.0) * progress_step_percent) / 100.0,
                        )

                    delta = progress_value - last_progress_value
                    if delta >= refresh_amount or self._terminated():
                        progress_bar.update(delta)
                        last_progress_value = progress_value

            if progress_bar is not None:
                final_value = (
                    float(len(self.canonical_chain_ids))
                    if self.mode == "by_blocks"
                    else float(self.t)
                )
                if final_value > last_progress_value:
                    progress_bar.update(final_value - last_progress_value)

            return self._summarize(), self.epoch_stats, self.event_log
        finally:
            if progress_bar is not None:
                progress_bar.close()


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


def theoretical_orphan_rate(p: float) -> float:
    a = (4 - 2 * p) * (1 - p) * p**3
    b = (1 + p) ** 2
    return a / b


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
) -> tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]:
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
        ]
    ],
    *,
    executor_cls: type[concurrent.futures.Executor],
    max_workers: int,
    show_progress: bool,
    progress_desc: str,
) -> List[tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]]:
    results: List[tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]] = []
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
        ]
    ],
    *,
    jobs: int,
    show_progress: bool,
    progress_desc: str,
) -> List[tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]]:
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


def theory_tbw(p: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    term_1_minus_p = 1.0 - p
    exponent = term_1_minus_p / p
    base = 2.0 * term_1_minus_p
    x = 2.0 * p * (term_1_minus_p * (base**exponent) + 1.0)
    return p * x / (1 + p)


def validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ScenarioTask:
    scenario_name: str
    p: float
    run_id: int
    n_value: Optional[int] = None
    t_end: Optional[float] = None
    target_blocks: Optional[int] = None
    enable_daa: bool = False
    log_events: bool = False


def run_scenario_task(
    *,
    task: ScenarioTask,
    T: float,
    epoch_len: int,
    base_seed: int,
) -> tuple[RunResult, List[EpochStat], List[Dict[str, Any]]]:
    seed = derive_seed(
        base_seed,
        task.scenario_name,
        task.p,
        task.run_id,
        task.n_value,
    )
    sim = TBWSimulation(
        T=T,
        p=task.p,
        seed=seed,
        mode="by_time" if task.t_end is not None else "by_blocks",
        t_end=task.t_end,
        target_blocks=task.target_blocks,
        enable_daa=task.enable_daa,
        epoch_len=epoch_len,
        scenario=task.scenario_name,
        run_id=task.run_id,
        n_value=task.n_value,
        log_events=task.log_events,
    )
    return sim.run()


def execute_scenario_tasks(
    *,
    tasks: List[ScenarioTask],
    T: float,
    epoch_len: int,
    base_seed: int,
    jobs: int,
    progress_desc: str,
    postfix_builder: Any,
) -> tuple[
    List[RunResult],
    List[EpochStat],
    Dict[tuple[float, int, Optional[int]], List[Dict[str, Any]]],
]:
    raw_results: List[RunResult] = []
    epoch_rows: List[EpochStat] = []
    event_logs: Dict[tuple[float, int, Optional[int]], List[Dict[str, Any]]] = {}
    effective_jobs = min(jobs, len(tasks))

    if jobs > effective_jobs:
        print(
            "[info] parallelism is per run, "
            f"so effective_jobs=min(jobs, tasks)={effective_jobs}"
        )

    if effective_jobs <= 1:
        if jobs > 1 and len(tasks) <= 1:
            print(
                "[info] only one run was requested; "
                "a single run is not split across multiple processes"
            )
        with tqdm(
            total=len(tasks), desc=progress_desc, unit="run", dynamic_ncols=True
        ) as progress_bar:
            for task in tasks:
                run_result, epochs, event_log = run_scenario_task(
                    task=task,
                    T=T,
                    epoch_len=epoch_len,
                    base_seed=base_seed,
                )
                raw_results.append(run_result)
                epoch_rows.extend(epochs)
                if event_log:
                    event_logs[(task.p, task.run_id, task.n_value)] = event_log
                progress_bar.update(1)
                progress_bar.set_postfix_str(postfix_builder(task))
    else:
        print(
            f"parallel mode enabled: requested_jobs={jobs}, "
            f"effective_jobs={effective_jobs}"
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs
        ) as executor:
            futures = {
                executor.submit(
                    run_scenario_task,
                    task=task,
                    T=T,
                    epoch_len=epoch_len,
                    base_seed=base_seed,
                ): task
                for task in tasks
            }
            with tqdm(
                total=len(tasks), desc=progress_desc, unit="run", dynamic_ncols=True
            ) as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    run_result, epochs, event_log = future.result()
                    raw_results.append(run_result)
                    epoch_rows.extend(epochs)
                    if event_log:
                        event_logs[(task.p, task.run_id, task.n_value)] = event_log
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(postfix_builder(task))

    raw_results.sort(
        key=lambda r: (
            float(r.p),
            -1 if r.n is None else int(r.n),
            int(r.run_id),
        )
    )
    epoch_rows.sort(
        key=lambda e: (
            float(e.p),
            int(e.n),
            int(e.run_id),
            int(e.epoch_index),
        )
    )
    return raw_results, epoch_rows, event_logs


def scenario1(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    jobs: int,
    results_root: Path,
    save_sample_event_log: bool,
    jobs: int,
    show_progress: bool,
) -> tuple[List[RunResult], List[Dict[str, Any]]]:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    scenario_name = "scenario1_no_daa_by_blocks"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

<<<<<<< HEAD
    tasks = [
        ScenarioTask(
            scenario_name=scenario_name,
            p=p,
            run_id=run_id,
            target_blocks=epoch_len,
            enable_daa=False,
            log_events=save_sample_event_log and (abs(p - 0.60) < 1e-9) and run_id == 0,
        )
        for p in p_values
        for run_id in range(runs)
    ]
    raw_results, _, event_logs = execute_scenario_tasks(
        tasks=tasks,
        T=T,
        epoch_len=epoch_len,
        base_seed=base_seed,
        jobs=jobs,
        progress_desc="scenario1",
        postfix_builder=lambda task: f"p={task.p:.2f}, run={task.run_id + 1}/{runs}",
    )

    for (p, run_id, _), event_log in sorted(event_logs.items()):
        event_path = out_dir / f"event_log_p{p:.2f}_run{run_id}.csv"
        fields = sorted({k for row in event_log for k in row.keys()})
        write_csv(event_path, event_log, fields)
=======
    tasks = []
    for p in p_values:
        for run_id in range(runs):
            seed = derive_seed(base_seed, scenario_name, p, run_id, None)
            log_events = (
                save_sample_event_log and (abs(p - 0.60) < 1e-9) and run_id == 0
            )
            tasks.append(
                (
                    T,
                    p,
                    seed,
                    "by_blocks",
                    None,
                    epoch_len,
                    False,
                    epoch_len,
                    scenario_name,
                    run_id,
                    None,
                    log_events,
                    None,
                )
            )

    task_results = _execute_tbw_tasks(
        tasks,
        jobs=jobs,
        show_progress=show_progress,
        progress_desc="scenario1 runs",
    )

    raw_results: List[RunResult] = []
    for run_result, _, event_log in task_results:
        raw_results.append(run_result)
        if event_log:
            event_path = (
                out_dir / f"event_log_p{run_result.p:.2f}_run{run_result.run_id}.csv"
            )
            fields = sorted({k for row in event_log for k in row.keys()})
            write_csv(event_path, event_log, fields)

    raw_results.sort(key=lambda r: (float(r.p), int(r.run_id)))
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    summary_rows: List[Dict[str, Any]] = []
    for p in p_values:
        p_rows = [r for r in raw_results if abs(r.p - p) < 1e-12]
        values = [r.A_share for r in p_rows]
        orphan_totals = [r.A_orphan_published + r.H_orphan_published for r in p_rows]
        orphan_rates = [
<<<<<<< HEAD
            float(total) / float(r.canonical_len if r.canonical_len > 0 else epoch_len)
            for r, total in zip(p_rows, orphan_totals)
=======
            (total / epoch_len) if epoch_len > 0 else 0.0 for total in orphan_totals
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
        ]
        m, s = aggregate_metric(values)
        orphan_rate_mean, orphan_rate_std = aggregate_metric(orphan_rates)
        orphan_sum_mean = mean(orphan_totals) if orphan_totals else 0.0
<<<<<<< HEAD
=======
        orphan_rate_mean, orphan_rate_std = aggregate_metric(orphan_rates)
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
        summary_rows.append(
            {
                "p": p,
                "metric": "A_share",
                "metric_mean": m,
                "metric_std": s,
                "runs": runs,
                "orphan_sum_mean": orphan_sum_mean,
                "orphan_rate_sim": orphan_rate_mean,
                "orphan_rate_std": orphan_rate_std,
                "orphan_rate_formula": theoretical_orphan_rate(p),
            }
        )

    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "p",
            "metric",
            "metric_mean",
            "metric_std",
            "runs",
            "orphan_sum_mean",
            "orphan_rate_sim",
            "orphan_rate_std",
            "orphan_rate_formula",
        ],
    )
    return raw_results, summary_rows


def scenario2(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    jobs: int,
    results_root: Path,
    jobs: int,
    show_progress: bool,
) -> tuple[List[RunResult], List[Dict[str, Any]]]:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    scenario_name = "scenario2_no_daa_by_time"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

    t_end = epoch_len * T
    tasks = [
<<<<<<< HEAD
        ScenarioTask(
            scenario_name=scenario_name,
            p=p,
            run_id=run_id,
            t_end=t_end,
            enable_daa=False,
=======
        (
            T,
            p,
            derive_seed(base_seed, scenario_name, p, run_id, None),
            "by_time",
            t_end,
            None,
            False,
            epoch_len,
            scenario_name,
            run_id,
            None,
            False,
            None,
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
        )
        for p in p_values
        for run_id in range(runs)
    ]
<<<<<<< HEAD
    raw_results, _, _ = execute_scenario_tasks(
        tasks=tasks,
        T=T,
        epoch_len=epoch_len,
        base_seed=base_seed,
        jobs=jobs,
        progress_desc="scenario2",
        postfix_builder=lambda task: f"p={task.p:.2f}, run={task.run_id + 1}/{runs}",
    )
=======
    task_results = _execute_tbw_tasks(
        tasks,
        jobs=jobs,
        show_progress=show_progress,
        progress_desc="scenario2 runs",
    )
    raw_results = [run_result for run_result, _, _ in task_results]
    raw_results.sort(key=lambda r: (float(r.p), int(r.run_id)))
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    summary_rows: List[Dict[str, Any]] = []
    baseline_blocks = float(epoch_len)
    for p in p_values:
        values = [
            r.A_blocks_canonical - p * baseline_blocks
            for r in raw_results
            if abs(r.p - p) < 1e-12
        ]
        m, s = aggregate_metric(values)
        summary_rows.append(
            {
                "p": p,
                "metric": "A_blocks_canonical_minus_p_times_2016",
                "metric_mean": m,
                "metric_std": s,
                "runs": runs,
            }
        )

    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["p", "metric", "metric_mean", "metric_std", "runs"],
    )
    return raw_results, summary_rows


def scenario3(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    jobs: int,
    results_root: Path,
    jobs: int,
    show_progress: bool,
) -> tuple[List[RunResult], List[EpochStat], List[Dict[str, Any]]]:
    p_values = [0.55, 0.65, 0.75, 0.85]
    n_values = [1, 2, 3, 5]
    scenario_name = "scenario3_daa_by_time"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

<<<<<<< HEAD
    tasks = [
        ScenarioTask(
            scenario_name=scenario_name,
            p=p,
            run_id=run_id,
            n_value=n_value,
            t_end=n_value * epoch_len * T,
            enable_daa=True,
        )
        for p in p_values
        for n_value in n_values
        for run_id in range(runs)
    ]
    raw_results, epoch_rows, _ = execute_scenario_tasks(
        tasks=tasks,
        T=T,
        epoch_len=epoch_len,
        base_seed=base_seed,
        jobs=jobs,
        progress_desc="scenario3",
        postfix_builder=lambda task: (
            f"p={task.p:.2f}, n={task.n_value}, run={task.run_id + 1}/{runs}"
        ),
=======
    tasks = []
    for p in p_values:
        for n_value in n_values:
            t_end = n_value * epoch_len * T
            for run_id in range(runs):
                tasks.append(
                    (
                        T,
                        p,
                        derive_seed(base_seed, scenario_name, p, run_id, n_value),
                        "by_time",
                        t_end,
                        None,
                        True,
                        epoch_len,
                        scenario_name,
                        run_id,
                        n_value,
                        False,
                        None,
                    )
                )

    task_results = _execute_tbw_tasks(
        tasks,
        jobs=jobs,
        show_progress=show_progress,
        progress_desc="scenario3 runs",
    )

    raw_results: List[RunResult] = []
    epoch_rows: List[EpochStat] = []
    for run_result, epochs, _ in task_results:
        raw_results.append(run_result)
        epoch_rows.extend(epochs)

    raw_results.sort(
        key=lambda r: (float(r.p), -1 if r.n is None else int(r.n), int(r.run_id))
    )
    epoch_rows.sort(
        key=lambda e: (float(e.p), int(e.n), int(e.run_id), int(e.epoch_index))
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
    )

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

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
            "difficulty_old": e.difficulty_old,
            "difficulty_new": e.difficulty_new,
            "a_blocks_canonical": e.a_blocks_canonical,
            "h_blocks_canonical": e.h_blocks_canonical,
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
            "difficulty_old",
            "difficulty_new",
            "a_blocks_canonical",
            "h_blocks_canonical",
        ]
    )
    write_csv(out_dir / "epoch_stats.csv", epoch_dict_rows, epoch_fields)

    summary_rows: List[Dict[str, Any]] = []
    for p in p_values:
        for n_value in n_values:
            baseline = p * n_value * epoch_len
            values = [
                r.A_blocks_canonical / baseline
                for r in raw_results
                if abs(r.p - p) < 1e-12 and r.n == n_value
            ]
            m, s = aggregate_metric(values)
            summary_rows.append(
                {
                    "p": p,
                    "n": n_value,
                    "metric": "A_blocks_canonical_over_pn2016",
                    "metric_mean": m,
                    "metric_std": s,
                    "runs": runs,
                }
            )

    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["p", "n", "metric", "metric_mean", "metric_std", "runs"],
    )
    return raw_results, epoch_rows, summary_rows


def apply_plot_theme(plt_mod: Any) -> None:
    plt_mod.style.use(["science", "ieee"])


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


def plot_scenario1(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    rows = sorted(summary_rows, key=lambda r: r["p"])
    p_vals = np.array([row["p"] for row in rows], dtype=float)
    means = np.array([row["metric_mean"] for row in rows], dtype=float)
    stds = np.array([row["metric_std"] for row in rows], dtype=float)
<<<<<<< HEAD
    theory_vals = np.array([theory_tbw(float(p)) for p in p_vals], dtype=float)
=======
    p_dense = np.linspace(
        float(np.min(p_vals)), float(np.max(p_vals)), 400, dtype=float
    )
    theory_dense = 2.0 * p_dense * p_dense * (2.0 - p_dense) / (1.0 + p_dense)
    theory_at_points = 2.0 * p_vals * p_vals * (2.0 - p_vals) / (1.0 + p_vals)
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f

    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    ax.errorbar(
        p_vals,
        means,
        yerr=stds,
        fmt="o-",
        color="#155eef",
        ecolor="#8eb4ff",
        linewidth=2.2,
        elinewidth=1.5,
        capsize=4,
        markersize=3,
        markerfacecolor="none",
        markeredgewidth=1.6,
        label="TBW simulation",
        zorder=3,
    )
    ax.fill_between(
        p_vals,
        means - stds,
        means + stds,
        color="#155eef",
        alpha=0.10,
        zorder=2,
    )
    ax.plot(
        p_dense,
        theory_dense,
        linestyle="--",
        color="#0f766e",
        linewidth=2.0,
        label="theory",
        zorder=2,
    )
    ax.plot(
        p_vals,
        p_vals,
        linestyle="--",
        color="#2f3a4f",
        linewidth=1.8,
        label="baseline y=x",
        zorder=1,
    )
<<<<<<< HEAD
    ax.plot(
        p_vals,
        theory_vals,
        linestyle="-.",
        color="#e04f16",
        linewidth=2.0,
        label="TBW theory",
        zorder=4,
    )
    ax.set_xlabel("Attacker hashrate p")
    ax.set_ylabel("mean(A_share)")
=======
    ax.set_xlabel("Attacker hashrate p", fontsize=12)
    ax.set_ylabel("mean(A_share)", fontsize=12)
    ax.tick_params(labelsize=10)
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
    ax.set_title("Scenario 1: Relative Share (No DAA, canonical length = 2016)")
    ax.legend(loc="upper left", fontsize=10)
    ax.margins(x=0.02)
<<<<<<< HEAD
    ymin = min(float(np.min(means - stds)), float(np.min(theory_vals)))
    ymax = max(float(np.max(means + stds)), float(np.max(theory_vals)))
    ax.set_ylim(
        bottom=max(0.0, ymin - 0.02),
        top=min(1.02, ymax + 0.02),
    )
=======
    ax.set_ylim(bottom=max(0.0, float(np.min(means - stds)) - 0.02))
    ax.grid(True, linestyle="--", alpha=0.6)
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f

    fig.tight_layout()
    save_figure_bundle(fig_dir, "s1_relative_share", fig)
    plt_mod.show()
    plt_mod.close(fig)

    data_rows = [
        {
            "p": float(p_vals[i]),
            "metric_mean": float(means[i]),
            "metric_std": float(stds[i]),
            "theory_relative_share": float(theory_at_points[i]),
            "baseline_y_equals_x": float(p_vals[i]),
            "theory_tbw": float(theory_vals[i]),
        }
        for i in range(len(p_vals))
    ]
    save_plot_data(fig_dir, "s1_relative_share", data_rows)


def plot_scenario1_orphan_rate(
    summary_rows: List[Dict[str, Any]], fig_dir: Path
) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    rows = sorted(summary_rows, key=lambda r: r["p"])
    p_vals = np.array([row["p"] for row in rows], dtype=float)
    orphan_rate_sim = np.array([row["orphan_rate_sim"] for row in rows], dtype=float)
    orphan_rate_std = np.array([row["orphan_rate_std"] for row in rows], dtype=float)
    orphan_rate_formula = np.array(
        [row["orphan_rate_formula"] for row in rows], dtype=float
    )
    p_dense = np.linspace(
        float(np.min(p_vals)), float(np.max(p_vals)), 400, dtype=float
    )
    orphan_rate_formula_dense = np.array(
        [theoretical_orphan_rate(float(p)) for p in p_dense],
        dtype=float,
    )

<<<<<<< HEAD
    fig, ax = plt_mod.subplots(figsize=(8.8, 5.2))
=======
    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
    ax.errorbar(
        p_vals,
        orphan_rate_sim,
        yerr=orphan_rate_std,
        fmt="o-",
<<<<<<< HEAD
=======
        capsize=4,
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
        color="#155eef",
        ecolor="#8eb4ff",
        linewidth=2.2,
<<<<<<< HEAD
        elinewidth=1.5,
        capsize=4,
        markerfacecolor="#ffffff",
=======
        markersize=3,
        markerfacecolor="none",
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
        markeredgewidth=1.6,
        label="simulation orphan rate",
        zorder=3,
    )
    ax.fill_between(
        p_vals,
        orphan_rate_sim - orphan_rate_std,
        orphan_rate_sim + orphan_rate_std,
        color="#155eef",
        alpha=0.10,
        zorder=2,
    )
    ax.plot(
        p_vals,
        orphan_rate_sim + orphan_rate_std,
        color="#155eef",
        linewidth=1.0,
        alpha=0.45,
        zorder=2,
    )
    ax.plot(
        p_vals,
        orphan_rate_sim - orphan_rate_std,
        color="#155eef",
        linewidth=1.0,
        alpha=0.45,
        zorder=2,
    )
    ax.plot(
        p_dense,
        orphan_rate_formula_dense,
        linestyle="--",
        color="#0f766e",
        linewidth=2.0,
        label="I(p)",
        zorder=2,
    )
    ax.set_xlabel("Attacker hashrate p", fontsize=12)
    ax.set_ylabel("orphan rate", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title("Scenario 1: Orphan Rate vs p")
    ax.legend(loc="upper left", fontsize=10)
    ax.margins(x=0.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    save_figure_bundle(fig_dir, "s1_orphan_rate_vs_p", fig)
    plt_mod.show()
    plt_mod.close(fig)

    data_rows = [
        {
            "p": float(p_vals[i]),
            "orphan_rate_sim": float(orphan_rate_sim[i]),
            "orphan_rate_std": float(orphan_rate_std[i]),
            "orphan_rate_formula": float(orphan_rate_formula[i]),
        }
        for i in range(len(p_vals))
    ]
    save_plot_data(fig_dir, "s1_orphan_rate_vs_p", data_rows)


def plot_scenario2(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    rows = sorted(summary_rows, key=lambda r: r["p"])
    p_vals = np.array([row["p"] for row in rows], dtype=float)
    means = np.array([row["metric_mean"] for row in rows], dtype=float)
    stds = np.array([row["metric_std"] for row in rows], dtype=float)

    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    ax.errorbar(
        p_vals,
        means,
        yerr=stds,
        fmt="o-",
        color="#0f766e",
        ecolor="#7bd4ce",
        linewidth=2.2,
        elinewidth=1.5,
        capsize=4,
        markersize=3,
        markerfacecolor="none",
        markeredgewidth=1.6,
    )
    ax.fill_between(
        p_vals,
        means,
        0.0,
        where=means <= 0.0,
        color="#f97316",
        alpha=0.12,
        interpolate=True,
        label="delta <= 0 region",
    )
    ax.axhline(
        0.0, linestyle="--", color="#344054", linewidth=1.6, label="baseline = 0"
    )
    ax.set_xlabel("Attacker hashrate p", fontsize=12)
    ax.set_ylabel("mean(A_blocks_canonical - p*2016)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title("Scenario 2: Absolute Gain Delta (No DAA, fixed time = 2016T)")
    ax.legend(loc="lower left", fontsize=10)
    ax.margins(x=0.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    save_figure_bundle(fig_dir, "s2_absolute_delta", fig)
    plt_mod.show()
    plt_mod.close(fig)

    data_rows = [
        {
            "p": float(p_vals[i]),
            "metric_mean": float(means[i]),
            "metric_std": float(stds[i]),
            "baseline_delta": 0.0,
        }
        for i in range(len(p_vals))
    ]
    save_plot_data(fig_dir, "s2_absolute_delta", data_rows)


def plot_scenario3(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)
    apply_plot_theme(plt_mod)

    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    p_values = sorted({float(row["p"]) for row in summary_rows})
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
            label=f"p={p:.2f}",
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
    ax.set_xlabel("Time horizon multiplier n", fontsize=12)
    ax.set_ylabel("mean( A_blocks_canonical / (p*n*2016) )", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title("Scenario 3: Long-Term Ratio with DAA")
    ax.legend(loc="best", ncols=2, fontsize=10)
    ax.margins(x=0.03)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    save_figure_bundle(fig_dir, "s3_longterm_ratio", fig)
    plt_mod.show()
    plt_mod.close(fig)
    save_plot_data(fig_dir, "s3_longterm_ratio", plot_data_rows)


def write_config_summary(
    *,
    path: Path,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    jobs: int,
) -> None:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    lines = [
        "TBW PoW Simulation Configuration",
        "================================",
        f"T = {T}",
        f"runs = {runs}",
        f"jobs = {jobs}",
        f"epoch_len = {epoch_len}",
        f"base_seed = {base_seed}",
        f"jobs = {jobs}",
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
        return ["1", "2", "3"]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    for p in parts:
        if p not in {"1", "2", "3"}:
            raise ValueError(f"Invalid scenario selector: {p}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="TBW PoW simulation runner")
    parser.add_argument(
        "--scenarios", default="all", help="all or comma-separated subset of 1,2,3"
    )
    parser.add_argument("--T", type=float, default=10.0, help="Target block interval")
    parser.add_argument(
        "--runs", type=int, default=100, help="Independent runs per configuration"
    )
    parser.add_argument(
        "--epoch-len", type=int, default=2016, help="Epoch length in canonical blocks"
    )
<<<<<<< HEAD
    parser.add_argument(
        "--jobs", type=int, default=25, help="Parallel worker processes across runs"
    )
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed")
=======
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed")
    parser.add_argument("--jobs", type=int, default=30, help="Parallel workers")
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
    parser.add_argument(
        "--results-dir", default="results", help="Results output directory"
    )
    parser.add_argument(
        "--figures-dir", default="figures", help="Figure output directory"
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--no-sample-event-log",
        action="store_true",
        help="Do not save sample event log for scenario1 p=0.60 run0",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress display to reduce terminal overhead",
    )

    args = parser.parse_args()
<<<<<<< HEAD
    validate_positive_int("runs", args.runs)
    validate_positive_int("jobs", args.jobs)
    validate_positive_int("epoch_len", args.epoch_len)
=======
    validate_jobs(args.jobs)
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f

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
    )

    print("TBW simulation config summary")
    print(
<<<<<<< HEAD
        f"  T={args.T}, runs={args.runs}, jobs={args.jobs}, epoch_len={args.epoch_len}, base_seed={args.base_seed}"
=======
        f"  T={args.T}, runs={args.runs}, epoch_len={args.epoch_len}, "
        f"base_seed={args.base_seed}, jobs={args.jobs}"
>>>>>>> 6aba88f0e551afe2d7d58038a646d2d5c460f83f
    )
    print("  gamma=0, private lead<=1, Poisson mining, tie by earlier t_publish")
    print("  DAA formula: D_new = D_old * (2016*T) / T_total")

    s1_summary: List[Dict[str, Any]] = []
    s2_summary: List[Dict[str, Any]] = []
    s3_summary: List[Dict[str, Any]] = []
    if "1" in scenarios:
        _, s1_summary = scenario1(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            jobs=args.jobs,
            results_root=results_dir,
            save_sample_event_log=not args.no_sample_event_log,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        print("Scenario 1 completed")

    if "2" in scenarios:
        _, s2_summary = scenario2(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            jobs=args.jobs,
            results_root=results_dir,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        print("Scenario 2 completed")

    if "3" in scenarios:
        _, _, s3_summary = scenario3(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            jobs=args.jobs,
            results_root=results_dir,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
        print("Scenario 3 completed")

    if not args.skip_plots:
        if get_plt() is None:
            print("matplotlib unavailable, skipped plotting")
        else:
            if s1_summary:
                plot_scenario1(s1_summary, figures_dir)
                plot_scenario1_orphan_rate(s1_summary, figures_dir)
            if s2_summary:
                plot_scenario2(s2_summary, figures_dir)
            if s3_summary:
                plot_scenario3(s3_summary, figures_dir)
            print("Figures saved")


if __name__ == "__main__":
    main()
