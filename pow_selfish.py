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
import numpy as np

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

# Centralized defaults for easy modification.
P_LIST = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
GAMMA = 0.0
N_REPEATS = 10
TARGET_MAIN_CHAIN_BLOCKS = 2016
BASE_SEED = 20260312
JOBS = 10
RESULTS_DIR = Path("results/selfish")
FIGURES_DIR = Path("figures/selfish")


@dataclass(slots=True)
class RunResult:
    p: float
    gamma: float
    run_id: int
    seed: int
    target_main_chain_blocks: int
    main_chain_blocks_selfish: int
    main_chain_blocks_honest: int
    selfish_orphan_blocks: int
    honest_orphan_blocks: int
    lead_at_stop: int
    in_race_at_stop: bool
    steps: int

    @property
    def main_chain_blocks_total(self) -> int:
        return self.main_chain_blocks_selfish + self.main_chain_blocks_honest

    @property
    def total_orphan_blocks(self) -> int:
        return self.selfish_orphan_blocks + self.honest_orphan_blocks

    @property
    def total_blocks_seen(self) -> int:
        return self.main_chain_blocks_total + self.total_orphan_blocks

    @property
    def revenue_share(self) -> float:
        if self.main_chain_blocks_total == 0:
            return 0.0
        return self.main_chain_blocks_selfish / self.main_chain_blocks_total

    @property
    def orphan_rate(self) -> float:
        if self.total_blocks_seen == 0:
            return 0.0
        return self.total_orphan_blocks / self.total_blocks_seen

    @property
    def selfish_orphan_rate(self) -> float:
        if self.total_blocks_seen == 0:
            return 0.0
        return self.selfish_orphan_blocks / self.total_blocks_seen

    @property
    def honest_orphan_rate(self) -> float:
        if self.total_blocks_seen == 0:
            return 0.0
        return self.honest_orphan_blocks / self.total_blocks_seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "gamma": self.gamma,
            "run_id": self.run_id,
            "seed": self.seed,
            "target_main_chain_blocks": self.target_main_chain_blocks,
            "main_chain_blocks_selfish": self.main_chain_blocks_selfish,
            "main_chain_blocks_honest": self.main_chain_blocks_honest,
            "main_chain_blocks_total": self.main_chain_blocks_total,
            "selfish_orphan_blocks": self.selfish_orphan_blocks,
            "honest_orphan_blocks": self.honest_orphan_blocks,
            "total_orphan_blocks": self.total_orphan_blocks,
            "revenue_share": self.revenue_share,
            "orphan_rate": self.orphan_rate,
            "selfish_orphan_rate": self.selfish_orphan_rate,
            "honest_orphan_rate": self.honest_orphan_rate,
            "lead_at_stop": self.lead_at_stop,
            "in_race_at_stop": self.in_race_at_stop,
            "steps": self.steps,
        }


def derive_seed(base_seed: int, p: float, gamma: float, run_id: int) -> int:
    token = f"{base_seed}|{p:.8f}|{gamma:.8f}|{run_id}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def simulate_one_run(
    p: float,
    gamma: float,
    seed: int,
    target_main_chain_blocks: int = TARGET_MAIN_CHAIN_BLOCKS,
    run_id: int = 0,
) -> RunResult:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if target_main_chain_blocks <= 0:
        raise ValueError("target_main_chain_blocks must be positive")

    rng = random.Random(seed)
    rand = rng.random

    main_chain_blocks_selfish = 0
    main_chain_blocks_honest = 0
    selfish_orphan_blocks = 0
    honest_orphan_blocks = 0
    steps = 0

    # `lead` is the number of attacker private blocks ahead of the public chain.
    # `in_race` is the special Eyal-Sirer 0' state:
    # - the attacker had lead 1
    # - honest miners just caught up
    # - the attacker immediately published its private block
    # - the next block decides which branch survives
    lead = 0
    in_race = False

    while (
        main_chain_blocks_selfish + main_chain_blocks_honest < target_main_chain_blocks
    ):
        steps += 1

        if rand() < p:
            # Selfish miner finds the next block.
            if in_race:
                # 0' --A--> attacker wins the race.
                # The attacker's published block and the newly found block enter
                # the main chain; the competing honest block becomes orphan.
                main_chain_blocks_selfish += 2
                honest_orphan_blocks += 1
                lead = 0
                in_race = False
            else:
                # k --A--> k+1 in the classical selfish-mining state machine.
                lead += 1
            continue

        # Honest miners find the next block.
        if in_race:
            # 0' --H--> the honest side mines first after the tie.
            if rand() < gamma:
                # gamma fraction of honest miners extends the attacker's branch.
                # Finalized chain: attacker's published block + one honest block.
                # The competing honest block becomes orphan.
                main_chain_blocks_selfish += 1
                main_chain_blocks_honest += 1
                honest_orphan_blocks += 1
            else:
                # The honest branch wins outright.
                main_chain_blocks_honest += 2
                selfish_orphan_blocks += 1
            lead = 0
            in_race = False
            continue

        if lead == 0:
            # 0 --H--> honest block is directly accepted.
            main_chain_blocks_honest += 1
            continue

        if lead == 1:
            # 1 --H--> 0' : attacker publishes its hidden block immediately.
            in_race = True
            continue

        if lead == 2:
            # 2 --H--> attacker publishes the whole private chain and overrides
            # the new honest block.
            main_chain_blocks_selfish += 2
            honest_orphan_blocks += 1
            lead = 0
            continue

        # lead >= 3 and honest finds a block.
        # Classical transition: publish one hidden block to keep the lead.
        # That published selfish block is guaranteed onto the main chain, and
        # the just-found honest block is doomed to be orphaned.
        main_chain_blocks_selfish += 1
        honest_orphan_blocks += 1
        lead -= 1

    return RunResult(
        p=p,
        gamma=gamma,
        run_id=run_id,
        seed=seed,
        target_main_chain_blocks=target_main_chain_blocks,
        main_chain_blocks_selfish=main_chain_blocks_selfish,
        main_chain_blocks_honest=main_chain_blocks_honest,
        selfish_orphan_blocks=selfish_orphan_blocks,
        honest_orphan_blocks=honest_orphan_blocks,
        lead_at_stop=lead,
        in_race_at_stop=in_race,
        steps=steps,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def _safe_se(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def summarize_results(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    grouped_rows: Dict[float, List[Dict[str, Any]]] = {}

    for row in raw_rows:
        grouped_rows.setdefault(float(row["p"]), []).append(row)

    for p_value in sorted(grouped_rows):
        rows = grouped_rows[p_value]
        revenue_values = [float(row["revenue_share"]) for row in rows]
        orphan_values = [float(row["orphan_rate"]) for row in rows]
        selfish_orphan_values = [float(row["selfish_orphan_rate"]) for row in rows]
        honest_orphan_values = [float(row["honest_orphan_rate"]) for row in rows]

        mean_revenue_share = _safe_mean(revenue_values)
        summary_rows.append(
            {
                "p": float(p_value),
                "gamma": float(rows[0]["gamma"]),
                "mean_revenue_share": mean_revenue_share,
                "std_revenue_share": _safe_std(revenue_values),
                "se_revenue_share": _safe_se(revenue_values),
                "mean_orphan_rate": _safe_mean(orphan_values),
                "std_orphan_rate": _safe_std(orphan_values),
                "se_orphan_rate": _safe_se(orphan_values),
                "mean_selfish_orphan_rate": _safe_mean(selfish_orphan_values),
                "mean_honest_orphan_rate": _safe_mean(honest_orphan_values),
                "revenue_minus_p": mean_revenue_share - float(p_value),
                "beats_p": mean_revenue_share > float(p_value),
                "mean_main_chain_blocks_selfish": _safe_mean(
                    [float(row["main_chain_blocks_selfish"]) for row in rows]
                ),
                "mean_main_chain_blocks_honest": _safe_mean(
                    [float(row["main_chain_blocks_honest"]) for row in rows]
                ),
                "mean_total_orphans": _safe_mean(
                    [float(row["total_orphan_blocks"]) for row in rows]
                ),
                "mean_steps": _safe_mean([float(row["steps"]) for row in rows]),
            }
        )

    return summary_rows


def plot_results(summary_rows: Sequence[Dict[str, Any]], figures_dir: Path) -> None:
    ensure_dir(figures_dir)

    p_values = np.array([float(row["p"]) for row in summary_rows], dtype=float)
    mean_revenue_share = np.array(
        [float(row["mean_revenue_share"]) for row in summary_rows], dtype=float
    )
    se_revenue_share = np.array(
        [float(row["se_revenue_share"]) for row in summary_rows], dtype=float
    )
    mean_orphan_rate = np.array(
        [float(row["mean_orphan_rate"]) for row in summary_rows], dtype=float
    )
    se_orphan_rate = np.array(
        [float(row["se_orphan_rate"]) for row in summary_rows], dtype=float
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.errorbar(
        p_values,
        mean_revenue_share,
        yerr=se_revenue_share,
        fmt="o-",
        capsize=4,
        linewidth=2.0,
        color="#155eef",
        ecolor="#8eb4ff",
        label="mean revenue_share",
    )
    ax.plot(
        p_values,
        p_values,
        linestyle="--",
        linewidth=1.8,
        color="#344054",
        label="y = x",
    )
    ax.set_xlabel("Attacker hashrate p")
    ax.set_ylabel("Revenue share")
    ax.set_title("Selfish Mining: Revenue Share vs p")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "revenue_share_vs_p.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.errorbar(
        p_values,
        mean_orphan_rate,
        yerr=se_orphan_rate,
        fmt="o-",
        capsize=4,
        linewidth=2.0,
        color="#0f766e",
        ecolor="#7bd4ce",
    )
    ax.set_xlabel("Attacker hashrate p")
    ax.set_ylabel("Orphan rate")
    ax.set_title("Selfish Mining: Orphan Rate vs p")
    fig.tight_layout()
    fig.savefig(figures_dir / "orphan_rate_vs_p.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_p_list(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("p_list must not be empty")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classic Eyal-Sirer selfish-mining simulation"
    )
    parser.add_argument(
        "--p-list",
        default=",".join(f"{p:.2f}" for p in P_LIST),
        help="comma-separated attacker hashrate list",
    )
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument(
        "--target-main-chain-blocks", type=int, default=TARGET_MAIN_CHAIN_BLOCKS
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress display to reduce terminal overhead",
    )
    return parser


def validate_inputs(
    p_list: Sequence[float],
    gamma: float,
    repeats: int,
    target_main_chain_blocks: int,
    jobs: int,
) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if target_main_chain_blocks <= 0:
        raise ValueError("target_main_chain_blocks must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    for p in p_list:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")


def _run_single_task(
    p: float,
    gamma: float,
    base_seed: int,
    target_main_chain_blocks: int,
    run_id: int,
) -> Dict[str, Any]:
    seed = derive_seed(base_seed=base_seed, p=p, gamma=gamma, run_id=run_id)
    return simulate_one_run(
        p=p,
        gamma=gamma,
        seed=seed,
        target_main_chain_blocks=target_main_chain_blocks,
        run_id=run_id,
    ).to_dict()


def _collect_parallel_results(
    tasks: Sequence[tuple[float, float, int, int, int]],
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
            with tqdm(
                total=len(futures),
                desc="runs",
                unit="run",
                dynamic_ncols=True,
            ) as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    raw_rows.append(future.result())
                    progress_bar.update(1)
    return raw_rows


def run_experiments(
    p_list: Sequence[float],
    gamma: float,
    n_repeats: int = N_REPEATS,
    target_main_chain_blocks: int = TARGET_MAIN_CHAIN_BLOCKS,
    base_seed: int = BASE_SEED,
    jobs: int = JOBS,
    show_progress: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_inputs(
        p_list=p_list,
        gamma=gamma,
        repeats=n_repeats,
        target_main_chain_blocks=target_main_chain_blocks,
        jobs=jobs,
    )

    tasks = [
        (p, gamma, base_seed, target_main_chain_blocks, run_id)
        for p in p_list
        for run_id in range(n_repeats)
    ]
    raw_rows: List[Dict[str, Any]] = []

    effective_jobs = min(jobs, len(tasks))
    if effective_jobs <= 1:
        for task in tasks:
            raw_rows.append(_run_single_task(*task))
    else:
        # This workload is CPU-bound, so we prefer processes for real speedup.
        # Some restricted environments disallow the semaphore/sysconf checks
        # ProcessPoolExecutor needs; in that case we keep `jobs` usable by
        # falling back to threads.
        try:
            raw_rows = _collect_parallel_results(
                tasks,
                executor_cls=concurrent.futures.ProcessPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
            )
        except (PermissionError, OSError):
            print(
                "[warn] process-based parallelism is unavailable here; "
                "falling back to ThreadPoolExecutor"
            )
            raw_rows = _collect_parallel_results(
                tasks,
                executor_cls=concurrent.futures.ThreadPoolExecutor,
                max_workers=effective_jobs,
                show_progress=show_progress,
            )

    raw_rows.sort(key=lambda row: (float(row["p"]), int(row["run_id"])))
    summary_rows = summarize_results(raw_rows)
    return raw_rows, summary_rows


def save_results(
    raw_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    results_dir: Path,
) -> None:
    ensure_dir(results_dir)
    if pd is not None:
        pd.DataFrame(raw_rows).to_csv(results_dir / "raw_runs.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
        return

    write_csv_rows(results_dir / "raw_runs.csv", raw_rows)
    write_csv_rows(results_dir / "summary.csv", summary_rows)


def print_report(summary_rows: Sequence[Dict[str, Any]]) -> None:
    if pd is not None:
        summary_df = pd.DataFrame(summary_rows)
        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", None)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        headers = list(summary_rows[0].keys()) if summary_rows else []
        widths = {header: len(header) for header in headers}
        for row in summary_rows:
            for header in headers:
                value = row[header]
                text = f"{value:.6f}" if isinstance(value, float) else str(value)
                widths[header] = max(widths[header], len(text))

        print(" ".join(header.rjust(widths[header]) for header in headers))
        for row in summary_rows:
            print(
                " ".join(
                    (
                        f"{row[header]:.6f}"
                        if isinstance(row[header], float)
                        else str(row[header])
                    ).rjust(widths[header])
                    for header in headers
                )
            )

    winning_rows = [row for row in summary_rows if bool(row["beats_p"])]
    if not winning_rows:
        print("\nNo scanned p value achieved mean_revenue_share > p.")
    else:
        p_values = ", ".join(f"{float(row['p']):.2f}" for row in winning_rows)
        print(f"\nScanned p values with mean_revenue_share > p: {p_values}")


def main() -> None:
    args = build_arg_parser().parse_args()
    p_list = _parse_p_list(args.p_list)

    raw_rows, summary_rows = run_experiments(
        p_list=p_list,
        gamma=args.gamma,
        n_repeats=args.repeats,
        target_main_chain_blocks=args.target_main_chain_blocks,
        base_seed=args.base_seed,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    save_results(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        results_dir=Path(args.results_dir),
    )
    if not args.skip_plots:
        plot_results(summary_rows=summary_rows, figures_dir=Path(args.figures_dir))
    print_report(summary_rows)


if __name__ == "__main__":
    main()
