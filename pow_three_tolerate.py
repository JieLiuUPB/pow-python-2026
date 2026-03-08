from __future__ import annotations

import argparse
import concurrent.futures
import math
from statistics import mean

from tqdm import tqdm

from pow_collusion import (
    BetrayConfig,
    Block,
    CollusionSimulation,
    PoolConfig,
    ScenarioConfig,
    SimConfig,
    make_seed,
    parse_pool_spec,
)


class FastCollusionSimulation(CollusionSimulation):
    def _publish_public_block(
        self, parent_id: int, miner_id: str, t_publish: float | None = None
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
    ) -> "RunResult":
        num_events = 0
        total = self.sim_config.target_blocks_long
        refresh_blocks = max(1, math.ceil(total * progress_step_percent / 100))
        last_height = self.canonical_height()
        pending_update = 0

        progress_bar = None
        if use_tqdm:
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
                            miner_id=process.pool_id, target_tip_id=process.target_tip_id
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-pool BetrayTolerated simulation (minimal runner)"
    )
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--target-blocks-long", type=int, default=2016000)

    parser.add_argument("--betray-on-nth-opportunity", type=int, default=1)
    parser.add_argument("--betray-start-height", type=int, default=1)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--betray-threshold", type=int, default=100)

    parser.add_argument("--three-pools", type=str, default="b=0.37,s=0.33,h=0.3")
    parser.add_argument("--three-traitor", type=str, default="s")
    parser.add_argument("--seed-base", type=int, default=20260224)
    parser.add_argument("--max-events", type=int, default=20_000_000)
    parser.add_argument("--progress-step-percent", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=30)
    return parser


def validate_three_only(
    sim_config: SimConfig, pools: list[PoolConfig], traitor: str
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

    total = sum(p.hashrate for p in pools)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"three hashrates must sum to 1.0, got {total}")
    for pool in pools:
        if pool.hashrate < 0.0:
            raise ValueError(
                f"three has negative hashrate: {pool.pool_id}={pool.hashrate}"
            )

    ids = {p.pool_id for p in pools}
    if {"b", "s", "h"} - ids:
        raise ValueError("three_pools must contain ids: b,s,h")
    if traitor not in {"b", "s"}:
        raise ValueError("three_traitor must be one of b,s")
    if traitor not in ids:
        raise ValueError(f"three_traitor {traitor} not in pools")


def validate_progress_step_percent(progress_step_percent: int) -> None:
    if not (1 <= progress_step_percent <= 100):
        raise ValueError("progress_step_percent must be in [1, 100]")


def validate_jobs(jobs: int) -> None:
    if jobs <= 0:
        raise ValueError("jobs must be positive")


def run_single(
    run_id: int,
    args: argparse.Namespace,
    sim_config: SimConfig,
    three_pools: list[PoolConfig],
    scenario: ScenarioConfig,
    use_tqdm: bool = False,
) -> "RunResult":
    seed = make_seed(args.seed_base, "three", scenario.name, run_id)
    sim = FastCollusionSimulation(
        experiment="three",
        sim_config=sim_config,
        pools=three_pools,
        scenario=scenario,
        run_id=run_id,
        seed=seed,
        max_events=args.max_events,
    )
    return sim.simulate_one_run(
        progress_step_percent=args.progress_step_percent,
        use_tqdm=use_tqdm,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    three_pools = parse_pool_spec(args.three_pools)
    validate_progress_step_percent(args.progress_step_percent)
    validate_jobs(args.jobs)

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
    validate_three_only(sim_config, three_pools, args.three_traitor)

    scenario = ScenarioConfig(
        name="BetrayTolerated",
        mode="long",
        initial_members=("b", "s"),
        break_rule="none",
        betray=BetrayConfig(
            mode="prob",
            traitor_id=args.three_traitor,
            betray_on_nth_opportunity=sim_config.betray_on_nth_opportunity,
            betray_start_height=sim_config.betray_start_height,
            q=sim_config.q,
            betray_threshold=sim_config.betray_threshold,
        ),
    )

    pool_ids = [p.pool_id for p in three_pools]
    results = []
    effective_jobs = min(args.jobs, sim_config.runs)
    if args.jobs > effective_jobs:
        print(
            "[info] parallelism is per run, "
            f"so effective_jobs=min(jobs, runs)={effective_jobs}"
        )

    if effective_jobs <= 1:
        if args.jobs > 1 and sim_config.runs <= 1:
            print(
                "[info] only one run was requested; "
                "a single run is not split across multiple processes"
            )
        for run_id in range(sim_config.runs):
            result = run_single(
                run_id,
                args,
                sim_config,
                three_pools,
                scenario,
                use_tqdm=True,
            )
            results.append(result)
            print(
                f"[run {run_id + 1}/{sim_config.runs}] done, "
                f"canonical_len={result.canonical_len}"
            )
    else:
        print(
            f"parallel mode enabled: requested_jobs={args.jobs}, "
            f"effective_jobs={effective_jobs}"
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs
        ) as executor:
            futures = {
                executor.submit(
                    run_single,
                    run_id,
                    args,
                    sim_config,
                    three_pools,
                    scenario,
                    False,
                ): run_id
                for run_id in range(sim_config.runs)
            }
            with tqdm(
                total=sim_config.runs,
                desc="runs",
                unit="run",
                dynamic_ncols=True,
            ) as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    run_id = futures[future]
                    result = future.result()
                    results.append(result)
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"last_run={run_id + 1}, canonical_len={result.canonical_len}"
                    )

    print(
        "BetrayTolerated（三矿池）收益率均值 "
        f"(runs={sim_config.runs}, blocks_per_run={sim_config.target_blocks_long})"
    )
    for pool_id in pool_ids:
        revenue = mean(r.shares_by_pool[pool_id] for r in results)
        print(f"{pool_id}: {revenue:.6f}")


if __name__ == "__main__":
    main()
