"""Simulate k-capped selfish mining without difficulty adjustment.

The attacker maintains a private tail on its own branch.  Let ``a_public`` and
``h_public`` be the lengths of the two published branches after their latest
common ancestor, and let ``hidden`` be the number of unpublished attacker
blocks extending ``a_public``.

The release policy is:

* after an attacker discovery, if ``hidden == k + 1``, publish the oldest
  hidden block immediately (so the post-action hidden count is at most ``k``);
* after an honest discovery makes ``h_public > a_public``, publish the oldest
  hidden block immediately, when one exists;
* if no hidden block is available for that response, the honest branch wins;
* at a public tie, a ``gamma`` fraction of honest hash power mines on the
  attacker branch, as in the Eyal--Sirer selfish-mining model.

The program sweeps attacker hashrate ``p`` and reports the attacker and honest
main-chain block shares plus the system-wide orphan rate.  Every mined block is
ultimately classified as either main-chain or orphan work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    pd = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    tqdm = None

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:  # pragma: no cover - visual style is optional
    import scienceplots  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    scienceplots = None


# Centralized defaults for easy modification.
P_LIST = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
K = 2
GAMMA = 0.0
N_REPEATS = 10
TARGET_MAIN_CHAIN_BLOCKS = 2016
BASE_SEED = 2026
JOBS = 10
RESULTS_DIR = Path("results/k_capped_selfish")
FIGURES_DIR = Path("figures/k_capped_selfish")


@dataclass(slots=True)
class RunResult:
    p: float
    k: int
    gamma: float
    run_id: int
    seed: int
    target_main_chain_blocks: int
    main_chain_blocks_attacker: int
    main_chain_blocks_honest: int
    attacker_orphan_blocks: int
    honest_orphan_blocks: int
    residual_hidden_before_finalize: int
    residual_public_tie_length_before_finalize: int
    cap_releases: int
    defensive_releases: int
    honest_extensions_on_attacker_branch: int
    max_hidden_after_release: int
    steps: int

    @property
    def main_chain_blocks_total(self) -> int:
        return self.main_chain_blocks_attacker + self.main_chain_blocks_honest

    @property
    def total_orphan_blocks(self) -> int:
        return self.attacker_orphan_blocks + self.honest_orphan_blocks

    @property
    def total_blocks_seen(self) -> int:
        return self.main_chain_blocks_total + self.total_orphan_blocks

    @property
    def attacker_main_chain_share(self) -> float:
        if self.main_chain_blocks_total == 0:
            return 0.0
        return self.main_chain_blocks_attacker / self.main_chain_blocks_total

    @property
    def honest_main_chain_share(self) -> float:
        if self.main_chain_blocks_total == 0:
            return 0.0
        return self.main_chain_blocks_honest / self.main_chain_blocks_total

    @property
    def orphan_rate(self) -> float:
        if self.total_blocks_seen == 0:
            return 0.0
        return self.total_orphan_blocks / self.total_blocks_seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "k": self.k,
            "gamma": self.gamma,
            "run_id": self.run_id,
            "seed": self.seed,
            "target_main_chain_blocks": self.target_main_chain_blocks,
            "main_chain_blocks_attacker": self.main_chain_blocks_attacker,
            "main_chain_blocks_honest": self.main_chain_blocks_honest,
            "main_chain_blocks_total": self.main_chain_blocks_total,
            "attacker_main_chain_share": self.attacker_main_chain_share,
            "honest_main_chain_share": self.honest_main_chain_share,
            "attacker_orphan_blocks": self.attacker_orphan_blocks,
            "honest_orphan_blocks": self.honest_orphan_blocks,
            "total_orphan_blocks": self.total_orphan_blocks,
            "total_blocks_seen": self.total_blocks_seen,
            "orphan_rate": self.orphan_rate,
            "residual_hidden_before_finalize": self.residual_hidden_before_finalize,
            "residual_public_tie_length_before_finalize": (
                self.residual_public_tie_length_before_finalize
            ),
            "cap_releases": self.cap_releases,
            "defensive_releases": self.defensive_releases,
            "honest_extensions_on_attacker_branch": (
                self.honest_extensions_on_attacker_branch
            ),
            "max_hidden_after_release": self.max_hidden_after_release,
            "steps": self.steps,
        }


def derive_seed(base_seed: int, p: float, k: int, gamma: float, run_id: int) -> int:
    token = f"{base_seed}|{p:.8f}|{k}|{gamma:.8f}|{run_id}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class KCappedSelfishSimulation:
    """One trajectory of the k-capped selfish-mining state machine."""

    def __init__(self, *, p: float, k: int, gamma: float, seed: int) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        if k < 0:
            raise ValueError("k must be non-negative")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")

        self.p = p
        self.k = k
        self.gamma = gamma
        self.rng = random.Random(seed)

        # The two unresolved public branches always have equal length between
        # events.  A one-block advantage is resolved immediately.
        self.a_public = 0
        self.h_public = 0
        self.hidden = 0

        self.main_attacker = 0
        self.main_honest = 0
        self.orphan_attacker = 0
        self.orphan_honest = 0

        self.cap_releases = 0
        self.defensive_releases = 0
        self.honest_extensions_on_attacker_branch = 0
        self.max_hidden_after_release = 0
        self.steps = 0

    @property
    def main_total(self) -> int:
        return self.main_attacker + self.main_honest

    def _assert_state(self) -> None:
        if self.a_public != self.h_public:
            raise AssertionError("unresolved public branches must be tied")
        if not (0 <= self.hidden <= self.k):
            raise AssertionError("post-action hidden count exceeds k")

    def _attacker_branch_wins(self) -> None:
        """Finalize a one-block public advantage for the attacker branch."""
        if self.a_public <= self.h_public:
            raise AssertionError("attacker branch is not ahead")
        self.main_attacker += self.a_public
        self.orphan_honest += self.h_public
        self.a_public = 0
        self.h_public = 0
        # Hidden descendants remain valid on top of the winning branch.

    def _honest_branch_wins(self) -> None:
        """Finalize the honest branch after the attacker runs out of blocks."""
        if self.h_public <= self.a_public or self.hidden != 0:
            raise AssertionError("honest branch cannot yet be finalized")
        self.main_honest += self.h_public
        self.orphan_attacker += self.a_public
        self.a_public = 0
        self.h_public = 0

    def _publish_oldest_hidden(self, *, reason: str) -> None:
        if self.hidden <= 0:
            raise AssertionError("no hidden block is available to publish")
        self.hidden -= 1
        self.a_public += 1
        if reason == "cap":
            self.cap_releases += 1
        elif reason == "defense":
            self.defensive_releases += 1
        else:  # pragma: no cover - internal programming error
            raise ValueError(f"unknown release reason: {reason}")

    def _honest_mines_on_attacker_branch(self) -> None:
        """Handle gamma-following at a non-empty public tie.

        The tied attacker prefix becomes common history and the old honest
        branch becomes stale.  If a private attacker block conflicts with the
        new honest extension, it is released immediately, creating a fresh
        one-block tie after the newly finalized prefix.
        """
        tied_length = self.a_public
        self.honest_extensions_on_attacker_branch += 1
        self.main_attacker += tied_length
        self.orphan_honest += self.h_public

        self.a_public = 0
        self.h_public = 1  # The new honest extension after the common prefix.
        if self.hidden > 0:
            self._publish_oldest_hidden(reason="defense")
        else:
            self.main_honest += 1
            self.h_public = 0

    def step(self) -> None:
        """Mine one block and apply all immediate publication decisions."""
        self._assert_state()
        self.steps += 1

        if self.rng.random() < self.p:
            self.hidden += 1
            if self.hidden > self.k:
                # The transient k+1-th hidden block forces publication of the
                # oldest hidden block; the private tail returns to size k.
                self._publish_oldest_hidden(reason="cap")
                self._attacker_branch_wins()
        elif (
            self.a_public > 0
            and self.a_public == self.h_public
            and self.rng.random() < self.gamma
        ):
            self._honest_mines_on_attacker_branch()
        else:
            # Honest miners extend the honest branch.  It is now one block
            # ahead of the attacker's published prefix.
            self.h_public += 1
            if self.hidden > 0:
                self._publish_oldest_hidden(reason="defense")
            else:
                self._honest_branch_wins()

        self.max_hidden_after_release = max(self.max_hidden_after_release, self.hidden)
        self._assert_state()

    def settle(self) -> None:
        """Classify all work after the target horizon has been crossed."""
        # A target can be crossed when gamma makes a tied prefix canonical but
        # leaves a fresh one-block race.  Resolve that race with normal events.
        while self.a_public > 0:
            self.step()

        # With no competing public branch, all residual hidden descendants are
        # guaranteed to be canonical when published.
        self.main_attacker += self.hidden
        self.hidden = 0
        self._assert_state()


def simulate_one_run(
    p: float,
    k: int,
    gamma: float,
    seed: int,
    target_main_chain_blocks: int = TARGET_MAIN_CHAIN_BLOCKS,
    run_id: int = 0,
) -> RunResult:
    if target_main_chain_blocks <= 0:
        raise ValueError("target_main_chain_blocks must be positive")

    simulation = KCappedSelfishSimulation(p=p, k=k, gamma=gamma, seed=seed)
    while simulation.main_total < target_main_chain_blocks:
        simulation.step()

    residual_hidden = simulation.hidden
    residual_public_tie_length = simulation.a_public
    simulation.settle()

    classified_blocks = (
        simulation.main_attacker
        + simulation.main_honest
        + simulation.orphan_attacker
        + simulation.orphan_honest
    )
    if classified_blocks != simulation.steps:
        raise AssertionError(
            "every mined block must be classified as main-chain or orphan work"
        )

    return RunResult(
        p=p,
        k=k,
        gamma=gamma,
        run_id=run_id,
        seed=seed,
        target_main_chain_blocks=target_main_chain_blocks,
        main_chain_blocks_attacker=simulation.main_attacker,
        main_chain_blocks_honest=simulation.main_honest,
        attacker_orphan_blocks=simulation.orphan_attacker,
        honest_orphan_blocks=simulation.orphan_honest,
        residual_hidden_before_finalize=residual_hidden,
        residual_public_tie_length_before_finalize=residual_public_tie_length,
        cap_releases=simulation.cap_releases,
        defensive_releases=simulation.defensive_releases,
        honest_extensions_on_attacker_branch=(
            simulation.honest_extensions_on_attacker_branch
        ),
        max_hidden_after_release=simulation.max_hidden_after_release,
        steps=simulation.steps,
    )


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _safe_se(values: Sequence[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def summarize_results(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_rows: Dict[float, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        grouped_rows.setdefault(float(row["p"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for p_value in sorted(grouped_rows):
        rows = grouped_rows[p_value]
        attacker_shares = [float(row["attacker_main_chain_share"]) for row in rows]
        honest_shares = [float(row["honest_main_chain_share"]) for row in rows]
        orphan_rates = [float(row["orphan_rate"]) for row in rows]
        summary_rows.append(
            {
                "p": p_value,
                "k": int(rows[0]["k"]),
                "gamma": float(rows[0]["gamma"]),
                "mean_attacker_main_chain_share": _safe_mean(attacker_shares),
                "std_attacker_main_chain_share": _safe_std(attacker_shares),
                "se_attacker_main_chain_share": _safe_se(attacker_shares),
                "mean_honest_main_chain_share": _safe_mean(honest_shares),
                "std_honest_main_chain_share": _safe_std(honest_shares),
                "se_honest_main_chain_share": _safe_se(honest_shares),
                "mean_orphan_rate": _safe_mean(orphan_rates),
                "std_orphan_rate": _safe_std(orphan_rates),
                "se_orphan_rate": _safe_se(orphan_rates),
                "mean_main_chain_blocks_attacker": _safe_mean(
                    [float(row["main_chain_blocks_attacker"]) for row in rows]
                ),
                "mean_main_chain_blocks_honest": _safe_mean(
                    [float(row["main_chain_blocks_honest"]) for row in rows]
                ),
                "mean_total_orphan_blocks": _safe_mean(
                    [float(row["total_orphan_blocks"]) for row in rows]
                ),
                "mean_cap_releases": _safe_mean(
                    [float(row["cap_releases"]) for row in rows]
                ),
                "mean_defensive_releases": _safe_mean(
                    [float(row["defensive_releases"]) for row in rows]
                ),
                "mean_steps": _safe_mean([float(row["steps"]) for row in rows]),
            }
        )
    return summary_rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)


def save_results(
    raw_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    results_dir: Path,
) -> None:
    ensure_dir(results_dir)
    if pd is not None:
        pd.DataFrame(raw_rows).to_csv(results_dir / "raw_runs.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
    else:
        write_csv_rows(results_dir / "raw_runs.csv", raw_rows)
        write_csv_rows(results_dir / "summary.csv", summary_rows)


def apply_plot_theme() -> None:
    if scienceplots is not None:
        try:
            plt.style.use(["science", "ieee", "no-latex"])
        except Exception:  # pragma: no cover - style-version compatibility
            plt.style.use("default")
    else:
        plt.style.use("default")
    plt.rcParams["text.usetex"] = False


def plot_results(summary_rows: Sequence[Dict[str, Any]], figures_dir: Path) -> None:
    if not summary_rows:
        print("[warn] empty summary, skip plotting")
        return

    ensure_dir(figures_dir)
    apply_plot_theme()

    p_values = [float(row["p"]) for row in summary_rows]
    attacker_shares = [
        float(row["mean_attacker_main_chain_share"]) for row in summary_rows
    ]
    attacker_errors = [
        float(row["se_attacker_main_chain_share"]) for row in summary_rows
    ]
    honest_shares = [float(row["mean_honest_main_chain_share"]) for row in summary_rows]
    honest_errors = [float(row["se_honest_main_chain_share"]) for row in summary_rows]
    orphan_rates = [float(row["mean_orphan_rate"]) for row in summary_rows]
    orphan_errors = [float(row["se_orphan_rate"]) for row in summary_rows]
    k = int(summary_rows[0]["k"])

    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    ax.errorbar(
        p_values,
        attacker_shares,
        yerr=attacker_errors,
        marker="o",
        capsize=4,
        linewidth=2.0,
        markersize=4,
        markerfacecolor="none",
        label="Attacker (main chain)",
    )
    ax.errorbar(
        p_values,
        honest_shares,
        yerr=honest_errors,
        marker="s",
        capsize=4,
        linewidth=2.0,
        markersize=4,
        markerfacecolor="none",
        label="Honest (main chain)",
    )
    ax.plot(p_values, p_values, "--", linewidth=1.2, label="Attacker hashrate p")
    ax.plot(
        p_values,
        [1.0 - p for p in p_values],
        ":",
        linewidth=1.2,
        label="Honest hashrate 1-p",
    )
    ax.set_xlabel("Attacker hashrate p", fontsize=14)
    ax.set_ylabel("Main-chain block share", fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"k-capped Selfish Mining (k={k})", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_chain_shares_vs_p.pdf", format="pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.125))
    ax.errorbar(
        p_values,
        orphan_rates,
        yerr=orphan_errors,
        marker="D",
        capsize=4,
        linewidth=2.0,
        markersize=4,
        markerfacecolor="none",
        label="Total orphan rate",
    )
    ax.set_xlabel("Attacker hashrate p", fontsize=14)
    ax.set_ylabel("Orphan blocks / all mined blocks", fontsize=14)
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"k-capped Selfish Mining Orphan Rate (k={k})", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "orphan_rate_vs_p.pdf", format="pdf", dpi=300)
    plt.close(fig)


def _parse_p_list(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("p_list must not be empty")
    return values


def validate_inputs(
    *,
    p_list: Sequence[float],
    k: int,
    gamma: float,
    repeats: int,
    target_main_chain_blocks: int,
    jobs: int,
) -> None:
    if k < 0:
        raise ValueError("k must be non-negative")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if target_main_chain_blocks <= 0:
        raise ValueError("target_main_chain_blocks must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    for p in p_list:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")


def _run_single_task(
    p: float,
    k: int,
    gamma: float,
    base_seed: int,
    target_main_chain_blocks: int,
    run_id: int,
) -> Dict[str, Any]:
    seed = derive_seed(base_seed, p, k, gamma, run_id)
    return simulate_one_run(
        p=p,
        k=k,
        gamma=gamma,
        seed=seed,
        target_main_chain_blocks=target_main_chain_blocks,
        run_id=run_id,
    ).to_dict()


def _collect_parallel_results(
    tasks: Sequence[tuple[float, int, float, int, int, int]],
    *,
    executor_cls: type[concurrent.futures.Executor],
    max_workers: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_task, *task) for task in tasks]
        if tqdm is None or not show_progress:
            for future in concurrent.futures.as_completed(futures):
                raw_rows.append(future.result())
        else:
            with tqdm(total=len(futures), desc="runs", unit="run") as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    raw_rows.append(future.result())
                    progress_bar.update(1)
    return raw_rows


def run_experiments(
    *,
    p_list: Sequence[float],
    k: int = K,
    gamma: float = GAMMA,
    n_repeats: int = N_REPEATS,
    target_main_chain_blocks: int = TARGET_MAIN_CHAIN_BLOCKS,
    base_seed: int = BASE_SEED,
    jobs: int = JOBS,
    show_progress: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_inputs(
        p_list=p_list,
        k=k,
        gamma=gamma,
        repeats=n_repeats,
        target_main_chain_blocks=target_main_chain_blocks,
        jobs=jobs,
    )
    tasks = [
        (p, k, gamma, base_seed, target_main_chain_blocks, run_id)
        for p in p_list
        for run_id in range(n_repeats)
    ]

    effective_jobs = min(jobs, len(tasks))
    if effective_jobs <= 1:
        raw_rows = [_run_single_task(*task) for task in tasks]
    else:
        try:
            raw_rows = _collect_parallel_results(
                tasks,
                executor_cls=concurrent.futures.ProcessPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
            )
        except (PermissionError, OSError):
            print(
                "[warn] process parallelism unavailable; "
                "falling back to ThreadPoolExecutor"
            )
            raw_rows = _collect_parallel_results(
                tasks,
                executor_cls=concurrent.futures.ThreadPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
            )

    raw_rows.sort(key=lambda row: (float(row["p"]), int(row["run_id"])))
    return raw_rows, summarize_results(raw_rows)


def print_report(summary_rows: Sequence[Dict[str, Any]]) -> None:
    display_columns = [
        "p",
        "k",
        "gamma",
        "mean_attacker_main_chain_share",
        "mean_honest_main_chain_share",
        "mean_orphan_rate",
        "se_attacker_main_chain_share",
        "se_honest_main_chain_share",
        "se_orphan_rate",
    ]
    if pd is not None:
        report = pd.DataFrame(summary_rows)[display_columns]
        print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        return

    print(",".join(display_columns))
    for row in summary_rows:
        print(
            ",".join(
                (
                    f"{row[column]:.6f}"
                    if isinstance(row[column], float)
                    else str(row[column])
                )
                for column in display_columns
            )
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="k-capped selfish-mining simulation")
    parser.add_argument(
        "--p-list",
        default=",".join(f"{p:.2f}" for p in P_LIST),
        help="comma-separated attacker hashrate list",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="maximum attacker blocks left hidden after an immediate release",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="fraction of honest hashrate following the attacker branch at a tie",
    )
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument(
        "--target-main-chain-blocks", type=int, default=TARGET_MAIN_CHAIN_BLOCKS
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_rows, summary_rows = run_experiments(
        p_list=_parse_p_list(args.p_list),
        k=args.k,
        gamma=args.gamma,
        n_repeats=args.repeats,
        target_main_chain_blocks=args.target_main_chain_blocks,
        base_seed=args.base_seed,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    save_results(raw_rows, summary_rows, Path(args.results_dir))
    if not args.skip_plots:
        plot_results(summary_rows, Path(args.figures_dir))
    print_report(summary_rows)


if __name__ == "__main__":
    main()
