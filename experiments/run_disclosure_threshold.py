"""Experiment A: validate the cartel disclosure threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from analysis.cartel_incentives import simulate_disclosure_run
from analysis.cartel_thresholds import disclosure_utilities, ell_star, lambda_value
from experiments.common import (
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


DEFAULT_ALPHAS = "0.55,0.60,0.65,0.70,0.75,0.80"


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    row = simulate_disclosure_run(**task["simulation"])
    row["run_id"] = task["run_id"]
    row["seed_base"] = task["seed_base"]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--alpha-ts", default="auto")
    parser.add_argument("--ells", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seeds", help="explicit comma-separated base seeds")
    parser.add_argument("--base-seed", type=int, default=20260909)
    parser.add_argument("--trials", type=int, default=5_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/disclosure_threshold")
    )
    args = parser.parse_args(argv)

    alphas = parse_floats(args.alphas)
    ells = parse_ints(args.ells)
    bases = seed_bases(args.seeds, args.base_seed, args.runs)
    tasks: list[dict[str, Any]] = []
    for alpha in alphas:
        for alpha_t in alpha_t_grid(alpha, args.alpha_ts):
            for ell in ells:
                for run_id, base in enumerate(bases):
                    seed = derive_seed(base, "disclosure_threshold", alpha, alpha_t, ell, args.gamma)
                    tasks.append(
                        {
                            "run_id": run_id,
                            "seed_base": base,
                            "simulation": {
                                "alpha": alpha,
                                "alpha_t": alpha_t,
                                "ell": ell,
                                "gamma": args.gamma,
                                "trials": args.trials,
                                "seed": seed,
                            },
                        }
                    )
    raw = run_parallel(_run_task, tasks, args.jobs)
    summary = aggregate(
        raw,
        group_fields=("alpha", "alpha_l", "alpha_t", "ell", "gamma"),
        metrics=("u_obey_sim", "u_betray_sim", "utility_difference_sim", "recovery_probability_sim"),
    )
    for row in summary:
        u_betray, u_obey = disclosure_utilities(
            float(row["alpha"]), float(row["alpha_t"]), int(row["ell"]), float(row["gamma"])
        )
        row.update(
            lambda_theory=lambda_value(float(row["alpha"]), float(row["alpha_t"])),
            u_obey_theory=u_obey,
            u_betray_theory=u_betray,
            utility_difference_theory=u_obey - u_betray,
            ell_star_theory=ell_star(float(row["alpha"]), float(row["alpha_t"]), float(row["gamma"])),
        )
    ensure_output_tree(args.output_root)
    write_csv(args.output_root / "raw" / "runs.csv", raw)
    write_csv(args.output_root / "aggregate" / "summary.csv", summary)
    config = vars(args).copy()
    config.update(effective_runs=len(bases), effective_seed_bases=",".join(map(str, bases)))
    write_config(args.output_root, config)
    print(f"wrote {len(raw)} raw rows and {len(summary)} aggregate rows to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
