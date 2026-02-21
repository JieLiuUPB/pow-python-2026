from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import numpy as np

plt = None
_PLOT_IMPORT_TRIED = False


def get_plt():  # pragma: no cover - plotting is optional at runtime
    global plt, _PLOT_IMPORT_TRIED
    if _PLOT_IMPORT_TRIED:
        return plt
    _PLOT_IMPORT_TRIED = True
    try:
        import matplotlib.pyplot as _plt
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
        value = -(self.T * self.difficulty / self.p) * math.log(2.0 * (1.0 - self.p))
        return max(value, 0.0)

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
            self._log("reorg_or_tip_change", old_tip=old_tip, new_tip=self.canonical_tip_id)

    def _publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.id] = block
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

            bn_pub = Block(
                id=bn.id,
                parent_id=bn.parent_id,
                height=bn.height,
                miner="A",
                t_publish=self.t,
            )
            self._publish_block(bn_pub)

            bn1_id = self._publish_new_block(parent_id=bn.id, miner="A", t_publish=self.t)
            self.counters.success_2blocks += 1
            self._log("mine_A_withhold_success", bn_id=bn.id, bn1_id=bn1_id)
            self._reset_attacker()
            return

        if state == "RACE":
            race_tip_id = self.attacker.race_tip_id
            if race_tip_id is None:
                race_tip_id = self.canonical_tip_id

            bnext_id = self._publish_new_block(parent_id=race_tip_id, miner="A", t_publish=self.t)
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
                if (
                    self.attacker.race_tip_id is not None
                    and not self._is_descendant(self.canonical_tip_id, self.attacker.race_tip_id)
                ):
                    self._log("race_lost")
                    self._reset_attacker()

    def _handle_release_A(self) -> None:
        if self.attacker.state != "WITHHOLD" or self.attacker.deadline is None:
            return

        if self.blocks_by_id[self.canonical_tip_id].height >= (self.attacker.base_height or 0) + 2:
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

        while len(self.canonical_chain_ids) >= (self.epochs_completed + 1) * self.epoch_len:
            epoch_index = self.epochs_completed + 1
            boundary_len = epoch_index * self.epoch_len
            boundary_block_id = self.canonical_chain_ids[boundary_len - 1]
            boundary_time = self.blocks_by_id[boundary_block_id].t_publish

            t_total = max(boundary_time - self.t_epoch_start, EPS)
            d_old = self.difficulty
            d_new = d_old * (self.epoch_len * self.T) / t_total
            self.difficulty = d_new

            segment = self.canonical_chain_ids[(epoch_index - 1) * self.epoch_len : boundary_len]
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
            t_r = self.attacker.deadline if self.attacker.deadline is not None else float("inf")

            t_next = min(t_a, t_h, t_r)
            if self.mode == "by_time" and self.t_end is not None and t_next > self.t_end:
                self.t = self.t_end
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

        return self._summarize(), self.epoch_stats, self.event_log


def derive_seed(base_seed: int, scenario: str, p: float, run_id: int, n_value: Optional[int]) -> int:
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


def scenario1(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    results_root: Path,
    save_sample_event_log: bool,
) -> tuple[List[RunResult], List[Dict[str, Any]]]:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    scenario_name = "scenario1_no_daa_by_blocks"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

    raw_results: List[RunResult] = []

    for p in p_values:
        for run_id in range(runs):
            seed = derive_seed(base_seed, scenario_name, p, run_id, None)
            log_events = save_sample_event_log and (abs(p - 0.60) < 1e-9) and run_id == 0
            sim = TBWSimulation(
                T=T,
                p=p,
                seed=seed,
                mode="by_blocks",
                t_end=None,
                target_blocks=epoch_len,
                enable_daa=False,
                epoch_len=epoch_len,
                scenario=scenario_name,
                run_id=run_id,
                n_value=None,
                log_events=log_events,
            )
            run_result, _, event_log = sim.run()
            raw_results.append(run_result)

            if event_log:
                event_path = out_dir / f"event_log_p{p:.2f}_run{run_id}.csv"
                fields = sorted({k for row in event_log for k in row.keys()})
                write_csv(event_path, event_log, fields)

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    summary_rows: List[Dict[str, Any]] = []
    for p in p_values:
        values = [r.A_share for r in raw_results if abs(r.p - p) < 1e-12]
        m, s = aggregate_metric(values)
        summary_rows.append({
            "p": p,
            "metric": "A_share",
            "metric_mean": m,
            "metric_std": s,
            "runs": runs,
        })

    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["p", "metric", "metric_mean", "metric_std", "runs"],
    )
    return raw_results, summary_rows


def scenario2(
    *,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
    results_root: Path,
) -> tuple[List[RunResult], List[Dict[str, Any]]]:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    scenario_name = "scenario2_no_daa_by_time"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

    t_end = epoch_len * T
    raw_results: List[RunResult] = []

    for p in p_values:
        for run_id in range(runs):
            seed = derive_seed(base_seed, scenario_name, p, run_id, None)
            sim = TBWSimulation(
                T=T,
                p=p,
                seed=seed,
                mode="by_time",
                t_end=t_end,
                target_blocks=None,
                enable_daa=False,
                epoch_len=epoch_len,
                scenario=scenario_name,
                run_id=run_id,
                n_value=None,
                log_events=False,
            )
            run_result, _, _ = sim.run()
            raw_results.append(run_result)

    raw_rows = [r.to_dict() for r in raw_results]
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(out_dir / "raw_runs.csv", raw_rows, raw_fields)

    summary_rows: List[Dict[str, Any]] = []
    baseline_blocks = float(epoch_len)
    for p in p_values:
        values = [r.A_blocks_canonical - p * baseline_blocks for r in raw_results if abs(r.p - p) < 1e-12]
        m, s = aggregate_metric(values)
        summary_rows.append({
            "p": p,
            "metric": "A_blocks_canonical_minus_p_times_2016",
            "metric_mean": m,
            "metric_std": s,
            "runs": runs,
        })

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
    results_root: Path,
) -> tuple[List[RunResult], List[EpochStat], List[Dict[str, Any]]]:
    p_values = [0.65, 0.70, 0.75, 0.80]
    n_values = [2, 3, 5, 10]
    scenario_name = "scenario3_daa_by_time"
    out_dir = results_root / scenario_name
    ensure_dir(out_dir)

    raw_results: List[RunResult] = []
    epoch_rows: List[EpochStat] = []

    for p in p_values:
        for n_value in n_values:
            t_end = n_value * epoch_len * T
            for run_id in range(runs):
                seed = derive_seed(base_seed, scenario_name, p, run_id, n_value)
                sim = TBWSimulation(
                    T=T,
                    p=p,
                    seed=seed,
                    mode="by_time",
                    t_end=t_end,
                    target_blocks=None,
                    enable_daa=True,
                    epoch_len=epoch_len,
                    scenario=scenario_name,
                    run_id=run_id,
                    n_value=n_value,
                    log_events=False,
                )
                run_result, epochs, _ = sim.run()
                raw_results.append(run_result)
                epoch_rows.extend(epochs)

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
    epoch_fields = list(epoch_dict_rows[0].keys()) if epoch_dict_rows else [
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


def plot_scenario1(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)

    p_vals = [row["p"] for row in summary_rows]
    means = [row["metric_mean"] for row in summary_rows]
    stds = [row["metric_std"] for row in summary_rows]

    plt_mod.figure(figsize=(8, 5))
    plt_mod.errorbar(p_vals, means, yerr=stds, fmt="o-", capsize=3, label="TBW simulated")
    plt_mod.plot(p_vals, p_vals, "--", label="baseline y=x")
    plt_mod.xlabel("p")
    plt_mod.ylabel("A_share")
    plt_mod.title("Scenario 1: Relative Share (No DAA, stop at 2016 canonical blocks)")
    plt_mod.legend()
    plt_mod.grid(alpha=0.3)
    plt_mod.tight_layout()
    plt_mod.savefig(fig_dir / "s1_relative_share.png", dpi=160)
    plt_mod.close()


def plot_scenario2(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)

    p_vals = [row["p"] for row in summary_rows]
    means = [row["metric_mean"] for row in summary_rows]
    stds = [row["metric_std"] for row in summary_rows]

    plt_mod.figure(figsize=(8, 5))
    plt_mod.errorbar(p_vals, means, yerr=stds, fmt="o-", capsize=3)
    plt_mod.axhline(0.0, linestyle="--")
    plt_mod.xlabel("p")
    plt_mod.ylabel("A_blocks_canonical - p*2016")
    plt_mod.title("Scenario 2: Absolute Gain Delta (No DAA, stop at 2016T)")
    plt_mod.grid(alpha=0.3)
    plt_mod.tight_layout()
    plt_mod.savefig(fig_dir / "s2_absolute_delta.png", dpi=160)
    plt_mod.close()


def plot_scenario3(summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        return
    ensure_dir(fig_dir)

    plt_mod.figure(figsize=(8, 5))
    p_values = sorted({row["p"] for row in summary_rows})
    for p in p_values:
        rows = sorted([r for r in summary_rows if abs(r["p"] - p) < 1e-12], key=lambda x: x["n"])
        x = [r["n"] for r in rows]
        y = [r["metric_mean"] for r in rows]
        yerr = [r["metric_std"] for r in rows]
        plt_mod.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"p={p}")

    plt_mod.axhline(1.0, linestyle="--")
    plt_mod.xlabel("n (time horizon multiplier)")
    plt_mod.ylabel("A_blocks_canonical / (p*n*2016)")
    plt_mod.title("Scenario 3: Long-Term Ratio with DAA")
    plt_mod.grid(alpha=0.3)
    plt_mod.legend()
    plt_mod.tight_layout()
    plt_mod.savefig(fig_dir / "s3_longterm_ratio.png", dpi=160)
    plt_mod.close()


def write_config_summary(
    *,
    path: Path,
    T: float,
    runs: int,
    epoch_len: int,
    base_seed: int,
) -> None:
    p_values = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
    lines = [
        "TBW PoW Simulation Configuration",
        "================================",
        f"T = {T}",
        f"runs = {runs}",
        f"epoch_len = {epoch_len}",
        f"base_seed = {base_seed}",
        "gamma = 0",
        "DAA: D_new = D_old * (2016*T) / T_total",
        "w*(p, D) = -(T*D/p) * ln(2*(1-p))",
        "w* at D=1:",
    ]
    for p in p_values:
        w_star = -(T / p) * math.log(2.0 * (1.0 - p))
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
    parser.add_argument("--scenarios", default="all", help="all or comma-separated subset of 1,2,3")
    parser.add_argument("--T", type=float, default=10.0, help="Target block interval")
    parser.add_argument("--runs", type=int, default=10, help="Independent runs per configuration")
    parser.add_argument("--epoch-len", type=int, default=2016, help="Epoch length in canonical blocks")
    parser.add_argument("--base-seed", type=int, default=20260221, help="Base seed")
    parser.add_argument("--results-dir", default="results", help="Results output directory")
    parser.add_argument("--figures-dir", default="figures", help="Figure output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--no-sample-event-log",
        action="store_true",
        help="Do not save sample event log for scenario1 p=0.60 run0",
    )

    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    write_config_summary(
        path=results_dir / "experiment_config.txt",
        T=args.T,
        runs=args.runs,
        epoch_len=args.epoch_len,
        base_seed=args.base_seed,
    )

    print("TBW simulation config summary")
    print(f"  T={args.T}, runs={args.runs}, epoch_len={args.epoch_len}, base_seed={args.base_seed}")
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
            results_root=results_dir,
            save_sample_event_log=not args.no_sample_event_log,
        )
        print("Scenario 1 completed")

    if "2" in scenarios:
        _, s2_summary = scenario2(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            results_root=results_dir,
        )
        print("Scenario 2 completed")

    if "3" in scenarios:
        _, _, s3_summary = scenario3(
            T=args.T,
            runs=args.runs,
            epoch_len=args.epoch_len,
            base_seed=args.base_seed,
            results_root=results_dir,
        )
        print("Scenario 3 completed")

    if not args.skip_plots:
        if get_plt() is None:
            print("matplotlib unavailable, skipped plotting")
        else:
            if s1_summary:
                plot_scenario1(s1_summary, figures_dir)
            if s2_summary:
                plot_scenario2(s2_summary, figures_dir)
            if s3_summary:
                plot_scenario3(s3_summary, figures_dir)
            print("Figures saved")


if __name__ == "__main__":
    main()
