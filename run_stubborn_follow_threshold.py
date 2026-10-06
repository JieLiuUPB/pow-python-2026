"""Experiment C: validate the k-stubborn follower threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cartel_incentives import simulate_stubborn_run
from cartel_thresholds import B_lose, stubborn_utilities
from common import (
    aggregate,
    alpha_t_grid,
    derive_seed,
    ensure_output_tree,
    parse_floats,
    parse_ints,
    run_parallel,
    seed_bases,
    write_config,
    write_csv,
)


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    row = simulate_stubborn_run(**task["simulation"])
    row["run_id"] = task["run_id"]
    row["seed_base"] = task["seed_base"]
    return row


def _threshold_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float, int, int], list[dict[str, Any]]] = {}
    for row in summary:
        key = (
            float(row["alpha"]),
            float(row["alpha_l"]),
            float(row["alpha_t"]),
            int(row["k"]),
            int(row["m"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (alpha, alpha_l, alpha_t, k, m), rows in sorted(grouped.items()):
        simulated = max(
            (
                int(row["d"])
                for row in rows
                if float(row["utility_difference_sim_mean"]) >= 0.0
            ),
            default=0,
        )
        output.append(
            {
                "alpha": alpha,
                "alpha_l": alpha_l,
                "alpha_t": alpha_t,
                "k": k,
                "m": m,
                "B_lose_theory": B_lose(alpha, alpha_t, k, m),
                "B_lose_simulation": simulated,
            }
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--alpha-ts", default="auto")
    parser.add_argument("--ks", default="2,3,4,5,6,8")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seeds", help="explicit comma-separated base seeds")
    parser.add_argument("--base-seed", type=int, default=20260909)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/stubborn_follow_threshold")
    )
    args = parser.parse_args(argv)
    bases = seed_bases(args.seeds, args.base_seed, args.runs)
    tasks: list[dict[str, Any]] = []
    for alpha in parse_floats(args.alphas):
        for alpha_t in alpha_t_grid(alpha, args.alpha_ts):
            for k in parse_ints(args.ks):
                for d in range(1, k):
                    for run_id, base in enumerate(bases):
                        seed = derive_seed(
                            base,
                            "stubborn_follow_threshold",
                            alpha,
                            alpha_t,
                            k,
                            d,
                            args.m,
                        )
                        tasks.append(
                            {
                                "run_id": run_id,
                                "seed_base": base,
                                "simulation": {
                                    "alpha": alpha,
                                    "alpha_t": alpha_t,
                                    "d": d,
                                    "k": k,
                                    "m": args.m,
                                    "trials": args.trials,
                                    "seed": seed,
                                },
                            }
                        )
    raw = run_parallel(_run_task, tasks, args.jobs)
    summary = aggregate(
        raw,
        group_fields=("alpha", "alpha_l", "alpha_t", "d", "k", "m"),
        metrics=("u_cartel_sim", "u_honest_sim", "utility_difference_sim"),
    )
    for row in summary:
        u_cartel, u_honest = stubborn_utilities(
            float(row["alpha"]),
            float(row["alpha_t"]),
            int(row["d"]),
            int(row["k"]),
            int(row["m"]),
        )
        row["u_cartel_theory"] = u_cartel
        row["u_honest_theory"] = u_honest
        row["utility_difference_theory"] = u_cartel - u_honest
        row["best_response_theory"] = "FOLLOW" if u_cartel >= u_honest else "HONEST"
        row["best_response_simulation"] = (
            "FOLLOW" if float(row["utility_difference_sim_mean"]) >= 0.0 else "HONEST"
        )
    thresholds = _threshold_rows(summary)
    ensure_output_tree(args.output_root)
    write_csv(args.output_root / "raw" / "runs.csv", raw)
    write_csv(args.output_root / "aggregate" / "summary.csv", summary)
    write_csv(args.output_root / "aggregate" / "thresholds.csv", thresholds)
    config = vars(args).copy()
    config.update(
        effective_runs=len(bases), effective_seed_bases=",".join(map(str, bases))
    )
    write_config(args.output_root, config)
    print(
        f"wrote {len(raw)} raw rows, {len(summary)} aggregate rows, and {len(thresholds)} thresholds to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
