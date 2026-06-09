#!/usr/bin/env python3
"""Compute merged Habitat and MCoCoNav SR/SPL for shard CSVs.

By default this script reads shard outputs under:
  ./DSInC_EXP/multi_hm3d_2-robot/logs/

Specifically it looks for:
  - eval_s*/episode_metrics.csv
  - eval_s*_no_comm/episode_metrics.csv

This lets shard directories like ``eval_s0`` and ``eval_s1`` be merged into a
single ``comm`` result, and ``eval_s0_no_comm`` + ``eval_s1_no_comm`` into a
single ``no_comm`` result.

For each episode this script computes:
  - Habitat success rate from ``habitat_success``
  - Habitat SPL from ``habitat_spl``
  - MCoCoNav success rate from ``mcoconav_success``
  - MCoCoNav SPL from ``mcoconav_spl``

SPL is computed as ``min_path_length / max(path_length, min_path_length)`` when
that metric marks the episode as successful, otherwise 0.0.

You can also pass explicit CSV paths.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = SCRIPT_DIR / "DSInC_EXP" / "multi_hm3d_2-robot" / "logs"
COMM_SHARD_DIR_RE = re.compile(r"eval_s\d+$")
NO_COMM_SHARD_DIR_RE = re.compile(r"eval_s\d+_no_comm$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute merged Habitat and MCoCoNav SR/SPL for comm and no_comm CSVs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=(
            "Directory containing evaluation shard folders "
            f"(default: {DEFAULT_LOGS_DIR})"
        ),
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help=(
            "Optional CSV paths. If omitted, shard episode_metrics.csv files are "
            "discovered under --logs-dir."
        ),
    )
    return parser.parse_args()


def discover_paths(cli_paths: Iterable[str], logs_dir: Path) -> list[Path]:
    if cli_paths:
        return [Path(path) for path in cli_paths]

    patterns = [
        "eval_s*/episode_metrics.csv",
        "eval_s*_no_comm/episode_metrics.csv",
    ]
    found: set[Path] = set()
    for pattern in patterns:
        found.update(logs_dir.glob(pattern))
    return sorted(found)


def infer_group(path: Path) -> str:
    parent_name = path.parent.name
    if NO_COMM_SHARD_DIR_RE.fullmatch(parent_name):
        return "no_comm"
    if COMM_SHARD_DIR_RE.fullmatch(parent_name):
        return "comm"

    full_path = str(path)
    if "_no_comm" in full_path:
        return "no_comm"
    if "_comm" in full_path:
        return "comm"

    raise ValueError(
        f"Could not infer group from path: {path}. "
        "Expected a shard directory like 'eval_s0' or 'eval_s1_no_comm'."
    )


def parse_success_flag(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        return 1.0 if any(part not in {"0", "0.0"} for part in parts) else 0.0
    lowered = text.lower()
    if lowered in {"true", "yes", "success"}:
        return 1.0
    try:
        return 1.0 if float(text) > 0.0 else 0.0
    except ValueError:
        return 0.0


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def first_success_value(row: dict[str, str], keys: list[str]) -> float:
    for key in keys:
        if key in row:
            raw_value = row.get(key, "")
            if str(raw_value).strip():
                return parse_success_flag(raw_value)
    return 0.0


def first_float_value(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        if key in row:
            value = parse_float(row.get(key, ""))
            if value is not None:
                return value
    return None


def compute_episode_spl(success_value: float, min_path_length: float | None, path_length: float | None) -> float:
    if success_value <= 0.0:
        return 0.0
    if (
        min_path_length is None
        or path_length is None
        or min_path_length <= 0.0
        or path_length <= 0.0
    ):
        return 0.0
    return min_path_length / max(path_length, min_path_length)


def compute_metrics(paths: list[Path]) -> dict[str, dict[str, float]]:
    metrics = {
        "comm": {
            "files": 0,
            "episodes": 0,
            "habitat_successes": 0.0,
            "habitat_spl_sum": 0.0,
            "mcoconav_successes": 0.0,
            "mcoconav_spl_sum": 0.0,
        },
        "no_comm": {
            "files": 0,
            "episodes": 0,
            "habitat_successes": 0.0,
            "habitat_spl_sum": 0.0,
            "mcoconav_successes": 0.0,
            "mcoconav_spl_sum": 0.0,
        },
    }

    for path in paths:
        group = infer_group(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            row_count = 0
            habitat_success_sum = 0.0
            habitat_spl_sum = 0.0
            mcoconav_success_sum = 0.0
            mcoconav_spl_sum = 0.0

            for row in reader:
                row_count += 1
                mcoconav_success_value = first_success_value(
                    row,
                    ["mcoconav_success", "MCoCoNav SR", "success_metric"],
                )
                habitat_success_value = first_success_value(
                    row,
                    ["habitat_success", "habitat success", "habitat_success_metric"],
                )
                habitat_spl_value = first_float_value(row, ["habitat_spl"])
                mcoconav_spl_value = first_float_value(row, ["mcoconav_spl"])
                min_path_length = first_float_value(row, ["min_path_length"])
                path_length = first_float_value(row, ["path_length"])

                mcoconav_success_sum += mcoconav_success_value
                habitat_success_sum += habitat_success_value
                if mcoconav_spl_value is None:
                    mcoconav_spl_value = compute_episode_spl(
                        mcoconav_success_value, min_path_length, path_length
                    )
                if habitat_spl_value is None:
                    habitat_spl_value = compute_episode_spl(
                        habitat_success_value, min_path_length, path_length
                    )
                mcoconav_spl_sum += min(max(mcoconav_spl_value, 0.0), 1.0)
                habitat_spl_sum += min(max(habitat_spl_value, 0.0), 1.0)

        metrics[group]["files"] += 1
        metrics[group]["episodes"] += row_count
        metrics[group]["habitat_successes"] += habitat_success_sum
        metrics[group]["habitat_spl_sum"] += habitat_spl_sum
        metrics[group]["mcoconav_successes"] += mcoconav_success_sum
        metrics[group]["mcoconav_spl_sum"] += mcoconav_spl_sum

    return metrics


def format_metric(name: str, values: dict[str, float]) -> str:
    episodes = int(values["episodes"])
    habitat_sr = values["habitat_successes"] / episodes if episodes else 0.0
    habitat_spl = values["habitat_spl_sum"] / episodes if episodes else 0.0
    mcoconav_sr = values["mcoconav_successes"] / episodes if episodes else 0.0
    mcoconav_spl = values["mcoconav_spl_sum"] / episodes if episodes else 0.0
    return (
        f"{name}: files={int(values['files'])}, episodes={episodes}, "
        f"Habitat SR={habitat_sr:.4f} ({habitat_sr:.2%}), "
        f"Habitat SPL={habitat_spl:.4f}, "
        f"MCoCoNav SR={mcoconav_sr:.4f} ({mcoconav_sr:.2%}), "
        f"MCoCoNav SPL={mcoconav_spl:.4f}"
    )


def main() -> int:
    args = parse_args()
    paths = discover_paths(args.csv_paths, args.logs_dir)
    if not paths:
        raise SystemExit(
            f"No matching shard CSV files found under {args.logs_dir}."
        )

    metrics = compute_metrics(paths)
    print(format_metric("comm", metrics["comm"]))
    print(format_metric("no_comm", metrics["no_comm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
