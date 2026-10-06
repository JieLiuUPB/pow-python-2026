"""Experiment D: compare m=2 theory with exact selfish continuation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cartel_incentives import simulate_exact_policy_run
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
    row = simulate_exact_policy_run(**task["simulation"])
    row["run_id"] = task["run_id"]
    row["seed_base"] = task["seed_base"]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--alpha-ts", default="auto")
    parser.add_argument("--ks", default="2,3,4,5,6,8")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seeds", help="explicit comma-separated base seeds")
    parser.add_argument("--base-seed", type=int, default=20260909)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-disclosure-rule", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/exact_policy_robustness")
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
                            "exact_policy_robustness",
                            alpha,
                            alpha_t,
                            k,
                            d,
                            args.gamma,
                            not args.no_disclosure_rule,
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
                                    "gamma": args.gamma,
                                    "trials": args.trials,
                                    "seed": seed,
                                    "disclose": not args.no_disclosure_rule,
                                },
                            }
                        )
    raw = run_parallel(_run_task, tasks, args.jobs)
    summary = aggregate(
        raw,
        group_fields=(
            "alpha",
            "alpha_l",
            "alpha_t",
            "d",
            "k",
            "gamma",
            "disclosure_rule",
        ),
        metrics=(
            "u_cartel_exact_sim",
            "u_honest_exact_sim",
            "utility_difference_exact_sim",
        ),
    )
    grouped: dict[tuple[float, float, float, int], list[dict[str, Any]]] = {}
    for row in summary:
        alpha, alpha_t, d, k = (
            float(row["alpha"]),
            float(row["alpha_t"]),
            int(row["d"]),
            int(row["k"]),
        )
        u_cartel, u_honest = stubborn_utilities(alpha, alpha_t, d, k, 2)
        row["u_cartel_theory_m2"] = u_cartel
        row["u_honest_theory_m2"] = u_honest
        row["utility_difference_theory_m2"] = u_cartel - u_honest
        grouped.setdefault((alpha, float(row["alpha_l"]), alpha_t, k), []).append(row)
    thresholds = []
    for (alpha, alpha_l, alpha_t, k), rows in sorted(grouped.items()):
        exact = max(
            (
                int(row["d"])
                for row in rows
                if float(row["utility_difference_exact_sim_mean"]) >= 0.0
            ),
            default=0,
        )
        theory = B_lose(alpha, alpha_t, k, 2)
        thresholds.append(
            {
                "alpha": alpha,
                "alpha_l": alpha_l,
                "alpha_t": alpha_t,
                "k": k,
                "B_lose_theory_m2": theory,
                "B_lose_exact_simulation": exact,
                "exact_minus_theory": exact - theory,
                "classification": (
                    "exact"
                    if exact == theory
                    else ("m2_conservative" if exact > theory else "m2_optimistic")
                ),
            }
        )
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
        f"wrote {len(raw)} raw rows, {len(summary)} aggregate rows, and {len(thresholds)} comparisons to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
