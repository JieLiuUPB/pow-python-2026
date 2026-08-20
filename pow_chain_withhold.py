"""
Chain-withhold simulation (modified TBW strategy).

Key difference from pow_simulation.py scenario 1:
  When A is in WITHHOLD and mines B_{n+1}, instead of publishing BOTH
  B_n and B_{n+1} immediately (original), A publishes ONLY B_n and starts
  a fresh w* withhold window for B_{n+1}.  This lets A chain multiple
  withhold cycles back-to-back.

The default experiment compares w = 0.5T, 1T, and 10T over attacker
hashrates from 0.30 through 0.95.

Statistics collected (no DAA, terminated at 2016 canonical blocks):
  - A_share  = A_blocks_canonical / canonical_len
  - orphan_rate = orphan_published / total_published
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Matplotlib (optional)
# ---------------------------------------------------------------------------
plt = None
_PLOT_IMPORT_TRIED = False


def get_plt():
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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


@dataclass
class AttackerState:
    state: str = "IDLE"  # IDLE | WITHHOLD | RACE
    private_bn: Optional[PrivateBlock] = None
    deadline: Optional[float] = None
    race_tip_id: Optional[int] = None


@dataclass
class AttackCounters:
    started: int = 0
    chain_extensions: int = 0  # B_n published, B_{n+1} withheld (new behaviour)
    success_race: int = 0
    abort: int = 0
    release_only: int = 0  # reached deadline, entered RACE


@dataclass
class RunResult:
    p: float
    w_over_T: float
    w: float
    run_id: int
    seed: int
    T: float
    canonical_len: int
    A_blocks_canonical: int
    H_blocks_canonical: int
    A_share: float
    published_total: int
    orphan_total: int
    orphan_rate: float
    A_orphan_published: int
    H_orphan_published: int
    attacks_started: int
    attacks_chain_extensions: int
    attacks_success_race: int
    attacks_abort: int
    attacks_release_only: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "w_over_T": self.w_over_T,
            "w": self.w,
            "run_id": self.run_id,
            "seed": self.seed,
            "T": self.T,
            "canonical_len": self.canonical_len,
            "A_blocks_canonical": self.A_blocks_canonical,
            "H_blocks_canonical": self.H_blocks_canonical,
            "A_share": self.A_share,
            "published_total": self.published_total,
            "orphan_total": self.orphan_total,
            "orphan_rate": self.orphan_rate,
            "A_orphan_published": self.A_orphan_published,
            "H_orphan_published": self.H_orphan_published,
            "attacks_started": self.attacks_started,
            "attacks_chain_extensions": self.attacks_chain_extensions,
            "attacks_success_race": self.attacks_success_race,
            "attacks_abort": self.attacks_abort,
            "attacks_release_only": self.attacks_release_only,
        }


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


class ChainWithholdSimulation:
    """
    Modified TBW: when A mines B_{n+1} during WITHHOLD, publish B_n
    immediately and start a new WITHHOLD for B_{n+1} with a fresh deadline.
    """

    def __init__(
        self,
        *,
        T: float,
        p: float,
        seed: int,
        target_blocks: int,
        run_id: int,
        max_events: Optional[int] = None,
        w_over_T: float = 10.0,
    ) -> None:
        self.T = float(T)
        self.p = float(p)
        self.seed = int(seed)
        self.target_blocks = target_blocks
        self.run_id = run_id
        self.max_events = max_events
        self.w_over_T = float(w_over_T)
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if not (0.0 < self.p < 1.0):
            raise ValueError("p must be in (0, 1)")
        if self.target_blocks <= 0:
            raise ValueError("target_blocks must be positive")
        if self.w_over_T < 0.0:
            raise ValueError("w_over_T must be non-negative")
        if self.max_events is not None and self.max_events <= 0:
            raise ValueError("max_events must be positive when provided")
        self.rng = np.random.default_rng(self.seed)

        self.t = 0.0
        self.difficulty = 1.0  # fixed (no DAA)

        genesis = Block(id=0, parent_id=None, height=0, miner="G", t_publish=0.0)
        self.blocks_by_id: Dict[int, Block] = {0: genesis}
        self.tips: set[int] = {0}
        self.canonical_tip_id: int = 0
        self.canonical_chain_ids: List[int] = []
        self._next_block_id: int = 1

        self.attacker = AttackerState()
        self.counters = AttackCounters()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _new_block_id(self) -> int:
        bid = self._next_block_id
        self._next_block_id += 1
        return bid

    def _w_star(self) -> float:
        """Return the configured withholding window w."""
        return self.w_over_T * self.T

    def _get_chain_tip_id(self) -> int:
        def key(bid: int) -> tuple[int, float, int]:
            b = self.blocks_by_id[bid]
            return (b.height, -b.t_publish, -b.id)

        return max(self.tips, key=key)

    def _rebuild_canonical_chain(self) -> List[int]:
        chain: List[int] = []
        cur: Optional[int] = self.canonical_tip_id
        while cur is not None and cur != 0:
            chain.append(cur)
            cur = self.blocks_by_id[cur].parent_id
        chain.reverse()
        return chain

    def _refresh_canonical(self) -> None:
        self.canonical_tip_id = self._get_chain_tip_id()
        self.canonical_chain_ids = self._rebuild_canonical_chain()

    def _publish_block(self, block: Block) -> None:
        self.blocks_by_id[block.id] = block
        self.tips.add(block.id)
        if block.parent_id is not None:
            self.tips.discard(block.parent_id)
        self._refresh_canonical()

    def _publish_new_block(self, parent_id: int, miner: str, t_publish: float) -> int:
        parent = self.blocks_by_id[parent_id]
        bid = self._new_block_id()
        self._publish_block(
            Block(
                id=bid,
                parent_id=parent_id,
                height=parent.height + 1,
                miner=miner,
                t_publish=t_publish,
            )
        )
        return bid

    def _reset_attacker(self) -> None:
        self.attacker = AttackerState()

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _handle_mine_A(self) -> None:
        state = self.attacker.state

        # ---- IDLE: start a new withhold cycle ----
        if state == "IDLE":
            parent = self.blocks_by_id[self.canonical_tip_id]
            private_bn = PrivateBlock(
                id=self._new_block_id(),
                parent_id=self.canonical_tip_id,
                height=parent.height + 1,
                miner="A",
            )
            self.attacker.state = "WITHHOLD"
            self.attacker.private_bn = private_bn
            self.attacker.deadline = self.t + self._w_star()
            self.attacker.race_tip_id = None
            self.counters.started += 1
            return

        # ---- WITHHOLD: A mined B_{n+1} ----
        if state == "WITHHOLD":
            bn = self.attacker.private_bn
            if bn is None:
                self._reset_attacker()
                return

            # Publish B_n now (t_publish = current time).
            self._publish_block(
                Block(
                    id=bn.id,
                    parent_id=bn.parent_id,
                    height=bn.height,
                    miner="A",
                    t_publish=self.t,
                )
            )

            # Create B_{n+1} as the new private block; start fresh w* window.
            # (This is the KEY DIFFERENCE from the original strategy.)
            bn1 = PrivateBlock(
                id=self._new_block_id(),
                parent_id=bn.id,
                height=bn.height + 1,
                miner="A",
            )
            self.attacker.private_bn = bn1
            self.attacker.deadline = self.t + self._w_star()
            # Stay in WITHHOLD – no state change.
            self.counters.chain_extensions += 1
            return

        # ---- RACE: A mines on hidden B first -> reveal 2-block private chain ----
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
            self._publish_new_block(parent_id=bn.id, miner="A", t_publish=self.t)
            self.counters.success_race += 1
            self._reset_attacker()
            return

    def _handle_mine_H(self) -> None:
        state = self.attacker.state

        if state == "RACE":
            race_tip = self.attacker.race_tip_id
            if race_tip is None:
                self.counters.abort += 1
                self._reset_attacker()
                return
            self._publish_new_block(parent_id=race_tip, miner="H", t_publish=self.t)
            self.counters.abort += 1
            self._reset_attacker()
            return

        new_h_id = self._publish_new_block(
            parent_id=self.canonical_tip_id, miner="H", t_publish=self.t
        )

        if state == "WITHHOLD":
            bn = self.attacker.private_bn
            if bn is None:
                self.counters.abort += 1
                self._reset_attacker()
                return
            hb = self.blocks_by_id[new_h_id]
            if hb.height == bn.height and hb.parent_id == bn.parent_id:
                self.attacker.state = "RACE"
                self.attacker.deadline = None
                self.attacker.race_tip_id = new_h_id

    def _handle_release_A(self) -> None:
        """WITHHOLD deadline reached before race trigger: publish hidden B and reset."""
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
        self._reset_attacker()

    def _check_abort_condition(self) -> None:
        """Lightweight sanity guard for impossible partial attacker state."""
        if self.attacker.state == "WITHHOLD" and self.attacker.private_bn is None:
            self.counters.abort += 1
            self._reset_attacker()
            return
        if self.attacker.state == "RACE" and (
            self.attacker.private_bn is None or self.attacker.race_tip_id is None
        ):
            self.counters.abort += 1
            self._reset_attacker()

    # ------------------------------------------------------------------
    # termination & summary
    # ------------------------------------------------------------------

    def _terminated(self) -> bool:
        return len(self.canonical_chain_ids) >= self.target_blocks

    def _summarize(self) -> RunResult:
        canonical_ids = set(self.canonical_chain_ids)
        a_can = h_can = a_orp = h_orp = 0
        for block in self.blocks_by_id.values():
            if block.id == 0:
                continue
            in_can = block.id in canonical_ids
            if block.miner == "A":
                if in_can:
                    a_can += 1
                else:
                    a_orp += 1
            elif block.miner == "H":
                if in_can:
                    h_can += 1
                else:
                    h_orp += 1

        canonical_len = len(self.canonical_chain_ids)
        published_total = a_can + h_can + a_orp + h_orp
        orphan_total = a_orp + h_orp
        a_share = (a_can / canonical_len) if canonical_len > 0 else 0.0
        orphan_rate = (orphan_total / published_total) if published_total > 0 else 0.0

        return RunResult(
            p=self.p,
            w_over_T=self._w_star() / self.T,
            w=self._w_star(),
            run_id=self.run_id,
            seed=self.seed,
            T=self.T,
            canonical_len=canonical_len,
            A_blocks_canonical=a_can,
            H_blocks_canonical=h_can,
            A_share=a_share,
            published_total=published_total,
            orphan_total=orphan_total,
            orphan_rate=orphan_rate,
            A_orphan_published=a_orp,
            H_orphan_published=h_orp,
            attacks_started=self.counters.started,
            attacks_chain_extensions=self.counters.chain_extensions,
            attacks_success_race=self.counters.success_race,
            attacks_abort=self.counters.abort,
            attacks_release_only=self.counters.release_only,
        )

    def run(self) -> RunResult:
        lam = 1.0 / (self.T * self.difficulty)
        lam_a = self.p * lam
        lam_h = (1.0 - self.p) * lam
        events = 0

        while not self._terminated():
            if self.max_events is not None and events >= self.max_events:
                raise RuntimeError("Reached max_events before termination")

            t_a = self.t + self.rng.exponential(1.0 / lam_a)
            t_h = self.t + self.rng.exponential(1.0 / lam_h)
            t_r = (
                self.attacker.deadline
                if self.attacker.deadline is not None
                else math.inf
            )

            self.t = min(t_a, t_h, t_r)

            if t_r <= t_a and t_r <= t_h:
                self._handle_release_A()
            elif t_a <= t_h:
                self._handle_mine_A()
            else:
                self._handle_mine_H()

            self._check_abort_condition()
            events += 1

        return self._summarize()


# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------


def _derive_seed(
    base_seed: int,
    p: float,
    run_id: int,
    w_over_T: float = 10.0,
) -> int:
    token = f"{base_seed}|chain_withhold|{p:.8f}|{w_over_T:.8f}|{run_id}"
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _run_task(
    T: float,
    p: float,
    seed: int,
    target_blocks: int,
    run_id: int,
    max_events: Optional[int],
    w_over_T: float = 10.0,
) -> Dict[str, Any]:
    sim = ChainWithholdSimulation(
        T=T,
        p=p,
        seed=seed,
        target_blocks=target_blocks,
        run_id=run_id,
        max_events=max_events,
        w_over_T=w_over_T,
    )
    return sim.run().to_dict()


def run_experiments(
    p_list: Sequence[float],
    *,
    w_over_T_list: Sequence[float] = (0.5, 1.0, 10.0),
    T: float = 10.0,
    n_repeats: int = 10,
    target_blocks: int = 2016,
    base_seed: int = 2026,
    jobs: int = 10,
    max_events: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    if T <= 0.0:
        raise ValueError("T must be positive")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if target_blocks <= 0:
        raise ValueError("target_blocks must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if not p_list:
        raise ValueError("p_list must not be empty")
    if not w_over_T_list:
        raise ValueError("w_over_T_list must not be empty")
    if any(not 0.0 < p < 1.0 for p in p_list):
        raise ValueError("all p values must be in (0, 1)")
    if any(w_over_T < 0.0 for w_over_T in w_over_T_list):
        raise ValueError("all w/T values must be non-negative")

    tasks = [
        (
            T,
            p,
            _derive_seed(base_seed, p, run_id, w_over_T),
            target_blocks,
            run_id,
            max_events,
            w_over_T,
        )
        for w_over_T in w_over_T_list
        for p in p_list
        for run_id in range(n_repeats)
    ]
    raw: List[Dict[str, Any]] = []
    effective_jobs = min(jobs, len(tasks))

    def _collect_with_executor(
        executor_cls: type[concurrent.futures.Executor],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with executor_cls(max_workers=effective_jobs) as ex:
            futures = [ex.submit(_run_task, *t) for t in tasks]
            if tqdm is not None and show_progress:
                with tqdm(
                    total=len(futures), desc="runs", unit="run", dynamic_ncols=True
                ) as bar:
                    for f in concurrent.futures.as_completed(futures):
                        rows.append(f.result())
                        bar.update(1)
            else:
                for f in concurrent.futures.as_completed(futures):
                    rows.append(f.result())
        return rows

    if effective_jobs <= 1:
        it = tasks
        if tqdm is not None and show_progress:
            it = tqdm(tasks, desc="runs", unit="run", dynamic_ncols=True)
        for task in it:
            raw.append(_run_task(*task))
    else:
        try:
            raw = _collect_with_executor(concurrent.futures.ProcessPoolExecutor)
        except (OSError, PermissionError) as exc:
            print(
                "[warn] ProcessPoolExecutor unavailable "
                f"({exc.__class__.__name__}: {exc}); falling back to threads."
            )
            raw = _collect_with_executor(concurrent.futures.ThreadPoolExecutor)

    raw.sort(
        key=lambda r: (
            float(r["w_over_T"]),
            float(r["p"]),
            int(r["run_id"]),
        )
    )
    return raw


# ---------------------------------------------------------------------------
# Summary / CSV
# ---------------------------------------------------------------------------


def summarize(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[float, float], List[Dict[str, Any]]] = {}
    for row in raw:
        key = (float(row.get("w_over_T", 10.0)), float(row["p"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for w_over_T, p in sorted(grouped):
        rows = grouped[(w_over_T, p)]
        shares = [float(r["A_share"]) for r in rows]
        orphans = [float(r["orphan_rate"]) for r in rows]
        ext = [float(r["attacks_chain_extensions"]) for r in rows]
        n = len(shares)
        summary.append(
            {
                "p": p,
                "w_over_T": w_over_T,
                "w": w_over_T * float(rows[0]["T"]),
                "runs": n,
                "A_share_mean": mean(shares),
                "A_share_std": stdev(shares) if n > 1 else 0.0,
                "orphan_rate_mean": mean(orphans),
                "orphan_rate_std": stdev(orphans) if n > 1 else 0.0,
                "chain_extensions_mean": mean(ext),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)


# ---------------------------------------------------------------------------
# Theoretical reference curves (original TBW, for comparison)
# ---------------------------------------------------------------------------


def _tbw_orphan_rate(p: float) -> float:
    """Theoretical orphan rate for TBW."""
    return p * p * (1.0 - p) / (1 + p * p * (1.0 - p))


def _tbw_a_share(p: float) -> float:
    """Theoretical attacker-block rate for TBW."""
    return (3.0 - 2 * p) * p**2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    summary: Sequence[Dict[str, Any]],
    figures_dir: Path,
) -> None:
    plt_mod = get_plt()
    if plt_mod is None:
        print("[warn] matplotlib unavailable, skip plotting")
        return

    try:
        plt_mod.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt_mod.style.use(["science", "ieee"])
    plt_mod.rcParams["text.usetex"] = False
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not summary:
        print("[warn] empty summary, skip plotting")
        return

    rows_by_window: Dict[float, List[Dict[str, Any]]] = {}
    for row in summary:
        rows_by_window.setdefault(float(row.get("w_over_T", 10.0)), []).append(row)
    for rows in rows_by_window.values():
        rows.sort(key=lambda row: float(row["p"]))

    all_p = np.array([row["p"] for row in summary], dtype=float)
    theory_start = max(0.5, float(all_p.min()))
    if float(all_p.max()) >= theory_start:
        p_theory = np.linspace(theory_start, float(all_p.max()), 400)
    else:
        p_theory = np.array([], dtype=float)
    theory_share = np.array([_tbw_a_share(float(p)) for p in p_theory])
    theory_orp = np.array([_tbw_orphan_rate(float(p)) for p in p_theory])

    # Keep the three simulated windows in one blue family: shorter windows are
    # darker and longer windows are lighter.  The lightest blue still has
    # enough contrast against the white background and grey grid.
    sim_colors = itertools.cycle(("#08306B", "#4292C6", "#9ECAE1"))
    sim_markers = itertools.cycle(("o", "s", "^"))
    window_styles = {
        w_over_T: (next(sim_colors), next(sim_markers))
        for w_over_T in sorted(rows_by_window)
    }
    theory_color = "#B42318"
    baseline_color = "#000000"

    # ---- Figure 1: A_share ----
    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    for zorder, w_over_T in enumerate(sorted(rows_by_window), start=3):
        rows = rows_by_window[w_over_T]
        color, marker = window_styles[w_over_T]
        ax.errorbar(
            [row["p"] for row in rows],
            [row["A_share_mean"] for row in rows],
            yerr=[row["A_share_std"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.5,
            markerfacecolor="none",
            capsize=3,
            linestyle="-",
            linewidth=1.5,
            label=rf"OCW sim, $w={w_over_T:g}T$",
            zorder=zorder,
        )
    ax.plot(
        [float(all_p.min()), float(all_p.max())],
        [float(all_p.min()), float(all_p.max())],
        color=baseline_color,
        linestyle="--",
        linewidth=1.5,
        label=r"Honest Mining ($y=\alpha$)",
        zorder=2,
    )
    ax.plot(
        p_theory,
        theory_share,
        color=theory_color,
        linestyle="--",
        linewidth=2.0,
        label="OCW theory",
        zorder=20,
    )
    ax.set_xlabel(r"Attacker hashrate $\alpha$", fontsize=12)
    ax.set_ylabel("Attacker Block Share", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(figures_dir / "cw_a_share.pdf", format="pdf", dpi=300)
    plt_mod.close(fig)

    # ---- Figure 2: orphan rate ----
    fig, ax = plt_mod.subplots(figsize=(5.5, 4.125))
    for zorder, w_over_T in enumerate(sorted(rows_by_window), start=3):
        rows = rows_by_window[w_over_T]
        color, marker = window_styles[w_over_T]
        ax.errorbar(
            [row["p"] for row in rows],
            [row["orphan_rate_mean"] for row in rows],
            yerr=[row["orphan_rate_std"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.5,
            markerfacecolor="none",
            capsize=3,
            linestyle="-",
            linewidth=1.5,
            label=rf"OCW sim, $w={w_over_T:g}T$",
            zorder=zorder,
        )
    ax.plot(
        p_theory,
        theory_orp,
        color=theory_color,
        linestyle="--",
        linewidth=2.0,
        label="OCW theory",
        zorder=20,
    )
    ax.set_xlabel(r"Attacker hashrate $\alpha$", fontsize=12)
    ax.set_ylabel("Orphan rate", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(figures_dir / "cw_orphan_rate.pdf", format="pdf", dpi=300)
    plt_mod.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_p_list(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("list must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chain-withhold modified TBW simulation"
    )
    parser.add_argument(
        "--p-list",
        default=("0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"),
    )
    parser.add_argument(
        "--w-over-T-list",
        default="0.5,1,10",
        help="comma-separated withholding-window multipliers w/T",
    )
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--target-blocks", type=int, default=2016)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--results-dir", default="results/chain_withhold")
    parser.add_argument("--figures-dir", default="figures/chain_withhold")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    p_list = _parse_p_list(args.p_list)
    w_over_T_list = _parse_p_list(args.w_over_T_list)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    print(
        f"chain-withhold simulation: T={args.T}, repeats={args.repeats}, "
        f"target_blocks={args.target_blocks}, jobs={args.jobs}"
    )
    print(f"p_list = {p_list}")
    print(f"w/T list = {w_over_T_list}")

    raw = run_experiments(
        p_list=p_list,
        w_over_T_list=w_over_T_list,
        T=args.T,
        n_repeats=args.repeats,
        target_blocks=args.target_blocks,
        base_seed=args.base_seed,
        jobs=args.jobs,
        max_events=args.max_events,
        show_progress=not args.no_progress,
    )
    summary = summarize(raw)

    write_csv(results_dir / "raw_runs.csv", raw)
    write_csv(results_dir / "summary.csv", summary)
    print(f"Results written to {results_dir}/")

    # Print summary table
    print(
        f"\n{'w/T':>6}  {'p':>6}  {'A_share_mean':>13}  {'A_share_std':>12}  "
        f"{'orp_mean':>10}  {'orp_std':>9}  {'chain_ext_mean':>14}"
    )
    for row in summary:
        print(
            f"{row['w_over_T']:>6.1f}  {row['p']:>6.2f}  "
            f"{row['A_share_mean']:>13.6f}  "
            f"{row['A_share_std']:>12.6f}  {row['orphan_rate_mean']:>10.6f}  "
            f"{row['orphan_rate_std']:>9.6f}  {row['chain_extensions_mean']:>14.2f}"
        )

    if not args.skip_plots:
        plot_results(summary, figures_dir)
        print(f"Figures written to {figures_dir}/")


if __name__ == "__main__":
    main()
