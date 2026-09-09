"""Shared reproducibility, CSV, aggregation, and grid utilities."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Iterable, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def parse_floats(specification: str) -> list[float]:
    """Parse a comma list or inclusive ``start:stop:step`` decimal range."""

    text = specification.strip()
    if not text:
        raise ValueError("numeric grid must not be empty")
    if ":" not in text:
        return [float(token.strip()) for token in text.split(",") if token.strip()]
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("range must have start:stop:step format")
    try:
        start, stop, step = (Decimal(part.strip()) for part in parts)
    except InvalidOperation as error:
        raise ValueError("range contains a non-numeric value") from error
    if step == 0 or (stop - start) * step < 0:
        raise ValueError("range step is zero or points away from stop")
    result: list[float] = []
    current = start
    comparator = (lambda value: value <= stop) if step > 0 else (lambda value: value >= stop)
    while comparator(current):
        result.append(float(current))
        current += step
    return result


def parse_ints(specification: str) -> list[int]:
    values = [int(token.strip()) for token in specification.split(",") if token.strip()]
    if not values:
        raise ValueError("integer list must not be empty")
    return values


def alpha_t_grid(alpha: float, specification: str) -> list[float]:
    if specification.strip().lower() == "auto":
        return [value for value in parse_floats("0.05:0.95:0.05") if value <= alpha - 0.05 + 1e-12]
    return [value for value in parse_floats(specification) if 0.0 < value < alpha]


def derive_seed(base_seed: int, experiment: str, *parameters: object) -> int:
    token = "|".join([str(base_seed), experiment, *(str(value) for value in parameters)])
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def seed_bases(explicit: str | None, base_seed: int, runs: int) -> list[int]:
    if explicit:
        return parse_ints(explicit)
    if runs <= 0:
        raise ValueError("runs must be positive")
    return [base_seed + run_id for run_id in range(runs)]


def run_parallel(function: Callable[[T], R], tasks: Sequence[T], jobs: int) -> list[R]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if jobs == 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
        return list(executor.map(function, tasks))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        item = dict(zip(group_fields, key))
        item["runs"] = len(members)
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            metric_se = metric_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            half_width = 1.96 * metric_se
            item[f"{metric}_mean"] = metric_mean
            item[f"{metric}_std"] = metric_std
            item[f"{metric}_se"] = metric_se
            item[f"{metric}_ci95"] = half_width
            item[f"{metric}_ci95_low"] = metric_mean - half_width
            item[f"{metric}_ci95_high"] = metric_mean + half_width
        output.append(item)
    return output


def ensure_output_tree(root: Path) -> None:
    for child in ("raw", "aggregate", "figures"):
        (root / child).mkdir(parents=True, exist_ok=True)


def write_config(root: Path, values: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in values.items()]
    (root / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

