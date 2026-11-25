"""Smoke run: the DE for 2 generations with the paper settings otherwise (same geometry, seed1), and the map on
one prior point and one training motion. Writes smoke_report.json to config.WORK_DIR.

Usage: python run_smoke.py [de | map]    (default: both; the two parts can run in two processes)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import config
import de
import map as observability_map
import records

SETTING, SEED, GENERATIONS = "same_geometry", "seed1", 2
POINT, MOTION = "development_1", "P-D01"


def run_de():
    started = time.perf_counter()
    records.make("teacher0", SETTING, sets=("training",))
    r = de.main(SETTING, SEED, GENERATIONS)
    costs = [e["cost"] for e in r["evaluations"]]
    return dict(setting=SETTING, seed=SEED, generations_logged=len(r["generation_log"]), evaluations=len(costs),
                all_costs_finite=bool(np.isfinite(costs).all()), best_training_cost=min(costs),
                generation_log=r["generation_log"], seconds=time.perf_counter() - started)


def run_map():
    started = time.perf_counter()
    r = observability_map.main(POINT, [MOTION])
    values = [x[k] for x in r["summary"] for k in ("marginal_snr", "schur_snr")] + [k for b in r["blocks"] for k in b["kappa"]]
    return dict(prior_point=POINT, motion=MOTION, all_values_finite=bool(np.isfinite(values).all()), summary=r["summary"],
                kappa=r["blocks"][0]["kappa"], seconds=time.perf_counter() - started)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["de", "map"]
    out = Path(config.WORK_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for part in parts:
        result = run_de() if part == "de" else run_map()
        (out / f"smoke_{part}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if all((out / f"smoke_{p}.json").exists() for p in ("de", "map")):
        d = json.loads((out / "smoke_de.json").read_text(encoding="utf-8"))
        m = json.loads((out / "smoke_map.json").read_text(encoding="utf-8"))
        passed = d["generations_logged"] == GENERATIONS and d["all_costs_finite"] and m["all_values_finite"]
        report = dict(passed=passed, rule="runs end without an error; every cost is finite; the DE log has 2 generations", de=d, map=m)
        (out / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("smoke", "passed" if passed else "not passed", flush=True)
