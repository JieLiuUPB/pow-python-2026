"""Minimal runner for the three-pool BetrayTolerated scenario."""

from __future__ import annotations

import argparse
from statistics import mean

from pow_collusion import (
    SimConfig,
    build_scenarios,
    parse_pool_spec,
    run_experiment,
    validate_inputs,
    validate_jobs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-pool BetrayTolerated simulation"
    )
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--target-blocks-long", type=int, default=2016)

    parser.add_argument("--betray-on-nth-opportunity", type=int, default=1)
    parser.add_argument("--betray-start-height", type=int, default=1)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--betray-threshold", type=int, default=100)

    parser.add_argument("--three-pools", type=str, default="b=0.30,s=0.50,h=0.20")
    parser.add_argument("--three-traitor", type=str, default="s")
    parser.add_argument("--seed-base", type=int, default=20260224)
    parser.add_argument("--max-events", type=int, default=20_000_000)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--progress-step-percent",
        type=int,
        default=1,
        help="deprecated compatibility flag; ignored to match pow_collusion.py",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress display to reduce terminal overhead",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    validate_jobs(args.jobs)

    if args.progress_step_percent != 1:
        print(
            "[info] --progress-step-percent is ignored; "
            "progress handling now follows pow_collusion.py"
        )

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
    validate_inputs(sim_config, three_pools, args.three_traitor)

    scenarios = build_scenarios(
        sim_config=sim_config,
        experiment="three",
        pools=three_pools,
        initial_members=("b", "s"),
        traitor_id=args.three_traitor,
    )
    scenario = next(item for item in scenarios if item.name == "BetrayTolerated")

    results = run_experiment(
        experiment="three",
        pools=three_pools,
        sim_config=sim_config,
        scenarios=[scenario],
        seed_base=args.seed_base,
        max_events=args.max_events,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )

    print("[three] BetrayTolerated scenario completed")
    print(
        "BetrayTolerated (three-pool) mean revenue share "
        f"(runs={sim_config.runs}, blocks_per_run={sim_config.target_blocks_long})"
    )
    for pool in three_pools:
        revenue = mean(result.shares_by_pool[pool.pool_id] for result in results)
        print(f"{pool.pool_id}: {revenue:.6f}")


if __name__ == "__main__":
    main()
