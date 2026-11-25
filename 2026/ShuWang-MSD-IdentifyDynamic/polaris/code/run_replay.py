"""Replay check: regenerate the main teacher's records, compare the test records with the stored signals,
and recompute J, V and T of the selected endpoints of both geometry settings.

Usage: python run_replay.py [setting ...]    (default: both settings; one setting per process is possible)
Each setting writes replay_<setting>.json to config.WORK_DIR; when both exist, replay_report.json is written.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import config
import model
import objective
import records

SETTINGS = ("different_geometry", "same_geometry")
TIER_1 = 1e-9      # relative difference: max abs difference / max abs value per array; |a - b| / |b| for a cost
TIER_2 = 1e-3      # costs only; the printed-digit rule of the paper is checked outside the package


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def stored(folder, setting, name):
    with np.load(model.CASE_DIR / "signals" / folder / setting / f"{name}.npz", allow_pickle=False) as f:
        return {k: np.array(f[k]) for k in f.files}


def replay(setting):
    started = time.perf_counter()
    records.make("teacher0", setting)
    rows = []
    for name in records.motions("test"):
        for kind, folder in ((False, "test_noisy"), (True, "test_noise_free")):
            new = records.load("teacher0", setting, "test", name, noise_free=kind)
            old = stored(folder, setting, name)
            for g in model.KEYS:
                rows.append(dict(record=folder, motion=name, group=g, relative_difference=rel(new[g][100::4], old[g])))
    run = json.loads((model.CASE_DIR / "results" / setting / "endpoints.json").read_text(encoding="utf-8"))
    selected = next(r for r in run["runs"] if r["selected"])
    params = np.array(selected["parameters"])
    sim = objective.Model()
    j = objective.training_cost(params, setting=setting, sim=sim)
    v = objective.validation_cost(params, setting=setting, sim=sim)
    test_rows = objective.motion_costs(params, "test", setting=setting, sim=sim)
    t = objective.test_cost(params, setting=setting, sim=sim)
    predictions = [dict(motion=r["motion"], group=g, relative_difference=rel(r["prediction"][g][100::4], stored("predictions_selected", setting, r["motion"])[g]))
                   for r in test_rows for g in model.KEYS]
    costs = {name: dict(recomputed=new, stored=selected[key], relative_difference=abs(new - selected[key]) / abs(selected[key]))
             for name, new, key in (("J", j, "training_cost"), ("V", v, "validation_cost"), ("T", t, "test_cost"))}
    report = dict(setting=setting, endpoint=f"seed{selected['seed']}", parameters=params.tolist(), costs=costs,
                  largest_record_difference=max(r["relative_difference"] for r in rows),
                  largest_prediction_difference=max(r["relative_difference"] for r in predictions),
                  records=rows, predictions=predictions, seconds=time.perf_counter() - started)
    Path(config.WORK_DIR).mkdir(parents=True, exist_ok=True)
    (Path(config.WORK_DIR) / f"replay_{setting}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(setting, {k: v["relative_difference"] for k, v in costs.items()}, "records", report["largest_record_difference"], flush=True)


def combine():
    parts = [json.loads((Path(config.WORK_DIR) / f"replay_{s}.json").read_text(encoding="utf-8")) for s in SETTINGS]
    cost_diff = max(c["relative_difference"] for p in parts for c in p["costs"].values())
    array_diff = max(max(p["largest_record_difference"], p["largest_prediction_difference"]) for p in parts)
    tier = 1 if max(cost_diff, array_diff) <= TIER_1 else (2 if cost_diff <= TIER_2 else 0)
    summary = dict(tier=tier, largest_cost_difference=cost_diff, largest_array_difference=array_diff,
                   note="Arrays: regenerated test records (noisy and noise-free) and test predictions of the selected endpoints, "
                        "compared at the scored samples with signals/. Tier 2 also needs the printed-digit check of the paper values.",
                   settings={p["setting"]: {k: p[k] for k in ("endpoint", "costs", "largest_record_difference", "largest_prediction_difference", "seconds")}
                             for p in parts})
    (Path(config.WORK_DIR) / "replay_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("tier", tier, "largest cost difference", cost_diff, "largest array difference", array_diff, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or SETTINGS:
        replay(s)
    if all((Path(config.WORK_DIR) / f"replay_{s}.json").exists() for s in SETTINGS):
        combine()
