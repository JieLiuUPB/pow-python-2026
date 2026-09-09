"""Experiment B: compare tolerate, solo-withhold, and honest regimes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from analysis.cartel_incentives import simulate_tolerance_run
from analysis.cartel_thresholds import rho_solo, rho_tolerate, tolerate_is_optimal
from experiments.common import (
    aggregate,
    derive_seed,
    ensure_output_tree,
    parse_floats,
    run_parallel,
    seed_bases,
    write_config,
    write_csv,
)


REGIMES = ("TOLERATE", "EXPEL_AND_SOLO_WITHHOLD", "HONEST")


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    row = simulate_tolerance_run(**task["simulation"])
    row["run_id"] = task["run_id"]
    row["seed_base"] = task["seed_base"]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-ts", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--alpha-ls", default="0.05:0.90:0.025")
    parser.add_argument("--w-over-ts", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seeds", help="explicit comma-separated base seeds")
    parser.add_argument("--base-seed", type=int, default=20260909)
    parser.add_argument("--cycles", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/tolerance_region")
    )
    args = parser.parse_args(argv)
    bases = seed_bases(args.seeds, args.base_seed, args.runs)
    tasks: list[dict[str, Any]] = []
    for alpha_t in parse_floats(args.alpha_ts):
        for alpha_l in parse_floats(args.alpha_ls):
            if alpha_l <= 0.0 or alpha_l + alpha_t >= 1.0:
                continue
            for w_over_t in parse_floats(args.w_over_ts):
                for regime in REGIMES:
                    for run_id, base in enumerate(bases):
                        seed = derive_seed(base, "tolerance_region", alpha_l, alpha_t, w_over_t, regime)
                        tasks.append(
                            {
                                "run_id": run_id,
                                "seed_base": base,
                                "simulation": {
                                    "regime": regime,
                                    "alpha_l": alpha_l,
                                    "alpha_t": alpha_t,
                                    "w_over_t": w_over_t,
                                    "cycles": args.cycles,
                                    "seed": seed,
                                },
                            }
                        )
    raw = run_parallel(_run_task, tasks, args.jobs)
    summary = aggregate(
        raw,
        group_fields=("regime", "alpha", "alpha_l", "alpha_t", "w_over_T"),
        metrics=("loyal_share_sim",),
    )
    for row in summary:
        alpha_l = float(row["alpha_l"])
        alpha_t = float(row["alpha_t"])
        z = math.exp(-float(row["w_over_T"]))
        theory = {
            "TOLERATE": rho_tolerate(alpha_l, alpha_t, z),
            "EXPEL_AND_SOLO_WITHHOLD": rho_solo(alpha_l, z),
            "HONEST": alpha_l,
        }[str(row["regime"])]
        row["loyal_share_theory"] = theory
        row["tolerate_optimal_theory"] = tolerate_is_optimal(alpha_l, alpha_t)

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
