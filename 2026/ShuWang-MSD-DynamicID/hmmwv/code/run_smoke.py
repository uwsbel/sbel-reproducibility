"""Smoke run: two DE generations with the paper settings, and the map on one prior point and one motion.

Part de: training records of the main teacher (different geometry), then de.py for seed 1 with a
generation budget of 2. Part map: map.py for the first development point and the motion P-D01.
Writes <OUTPUT_DIR>/smoke_<part>.json per part and <OUTPUT_DIR>/smoke_report.json.
"""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

import config

PARTS = ("de", "map")


def smoke_de():
    import de
    import records
    records.generate("different_geometry", 0, sets=["training"])
    result = de.run("different_geometry", 1, budget=2)
    with (Path(config.OUTPUT_DIR) / "de" / "de_log_different_geometry_seed1.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    costs = [float(r["best_training_cost"]) for r in rows]
    return {"result": result, "log": rows, "generations_in_log": max(int(r["generation"]) for r in rows),
            "all_costs_finite": bool(np.isfinite(costs).all())}


def smoke_map():
    import map as observability
    out = observability.run(["development_1"], ["P-D01"])
    values = [v for b in out["blocks"] for k in ("kappa", "marginal_snr", "schur_snr") for v in b[k]]
    return {"motion_set": out["motion_set"], "all_values_finite": bool(np.isfinite(values).all())}


def main(parts):
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        started = time.time()
        try:
            d = dict(smoke_de() if part == "de" else smoke_map(), error=None)
        except Exception:
            d = {"error": traceback.format_exc()}
        d["wall_s"] = time.time() - started
        (out_dir / f"smoke_{part}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    if all((out_dir / f"smoke_{p}.json").exists() for p in PARTS):
        r = {p: json.loads((out_dir / f"smoke_{p}.json").read_text(encoding="utf-8")) for p in PARTS}
        r["passed"] = (r["de"]["error"] is None and r["map"]["error"] is None and r["de"]["all_costs_finite"]
                       and r["de"]["generations_in_log"] == 2 and r["map"]["all_values_finite"])
        (out_dir / "smoke_report.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("smoke passed:", r["passed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=PARTS, help="run one part only (default: both)")
    a = ap.parse_args()
    main([a.part] if a.part else list(PARTS))
