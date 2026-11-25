"""Recompute the test costs and r values of the selected HMMWV endpoints from the stored signals.

The signals are the scored samples of the 18 test records and of the predictions.
check_signals(case_dir) compares each recomputed value with the value in results/.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SETTINGS = ("different_geometry", "same_geometry")
TOLERANCE = 1e-9


def _cost_module(case_dir):
    spec = importlib.util.spec_from_file_location("hmmwv_cost", Path(case_dir) / "cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entry(name, recomputed, stored, count=False):
    if count:
        ok = int(recomputed) == int(stored)
    else:
        ok = abs(recomputed - stored) <= TOLERANCE * max(abs(stored), 1e-300)
    return {"name": name, "recomputed": float(recomputed), "stored": float(stored), "ok": bool(ok)}


def recompute(case_dir, setting):
    """Costs and r values of one geometry setting, recomputed from the signals."""
    case_dir = Path(case_dir)
    cost = _cost_module(case_dir)
    sig = case_dir / "signals"
    motions = [m["id"] for m in _json(case_dir / "settings" / "motions.json")["test"]]
    out = {"motions": motions, "selected": {}, "selected_noise_free": {}, "noise_only": {}, "reference": {}}
    for m in motions:
        noisy = cost.load_record(sig / "test_noisy" / setting / f"{m}.npz")
        clean = cost.load_record(sig / "test_noise_free" / setting / f"{m}.npz")
        pred = cost.load_record(sig / "predictions_selected" / setting / f"{m}.npz")
        out["selected"][m] = {"group_costs": cost.group_costs(pred, noisy), "r": cost.r_values(pred, noisy)}
        out["selected_noise_free"][m] = {"group_costs": cost.group_costs(pred, clean), "r": cost.r_values(pred, clean)}
        out["noise_only"][m] = {"group_costs": cost.group_costs(clean, noisy)}
        ref = sig / "predictions_reference" / setting / f"{m}.npz"
        if ref.exists():
            out["reference"][m] = {"group_costs": cost.group_costs(cost.load_record(ref), noisy)}
    for key in ("selected", "noise_only", "reference"):
        rows = out[key]
        if not rows:
            continue
        per_motion = {m: float(np.mean(list(rows[m]["group_costs"].values()))) for m in motions}
        out[key + "_summary"] = {
            "motion_costs": per_motion,
            "test_cost": float(np.mean(list(per_motion.values()))),
            "group_costs": {g: float(np.mean([rows[m]["group_costs"][g] for m in motions])) for g in cost.GROUPS},
        }
    r_all = [out["selected"][m]["r"][g] for m in motions for g in cost.GROUPS]
    out["r_median"] = float(np.median(r_all))
    out["r_min"], out["r_max"] = min(r_all), max(r_all)
    out["outside_band"] = sum(abs(out["selected"][m]["r"][g] - 1) > cost.band(g) for m in motions for g in cost.GROUPS)
    t0 = out["noise_only_summary"]["test_cost"]
    ts = out["selected_summary"]["test_cost"]
    if out["reference"]:
        tt = out["reference_summary"]["test_cost"]
        out["share_removed"] = (tt - ts) / (tt - t0)
    return cost, out


def check_signals(case_dir):
    """Return a list of {"name", "recomputed", "stored", "ok"}; ok means a relative difference of at most 1e-9."""
    case_dir = Path(case_dir)
    checks = []
    for setting in SETTINGS:
        if not (case_dir / "signals" / "test_noisy" / setting).exists():
            continue
        cost, new = recompute(case_dir, setting)
        res = case_dir / "results" / setting
        seed = _json(res / "endpoints.json")["selected_seed"]
        stored = _json(res / "group_costs.json")
        end = next(e for e in stored["endpoints"] if e["seed"] == seed)
        levels = _json(res / "reference_levels.json")
        with (res / "test_measures.csv").open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["seed"]) == seed]
        tag = f"{setting}: "
        for r in rows:
            m, g = r["motion"], r["group"]
            checks.append(_entry(f"{tag}cost, {m}, {g}", new["selected"][m]["group_costs"][g], float(r["cost"])))
            checks.append(_entry(f"{tag}r, {m}, {g}", new["selected"][m]["r"][g], float(r["r"])))
            checks.append(_entry(f"{tag}cost against the noise-free record, {m}, {g}",
                                 new["selected_noise_free"][m]["group_costs"][g], float(r["cost_noise_free"])))
            checks.append(_entry(f"{tag}r against the noise-free record, {m}, {g}",
                                 new["selected_noise_free"][m]["r"][g], float(r["r_noise_free"])))
        checks.append(_entry(f"{tag}test cost of the selected endpoint", new["selected_summary"]["test_cost"],
                             end["test_cost"]))
        for g in cost.GROUPS:
            checks.append(_entry(f"{tag}group cost of the selected endpoint, {g}",
                                 new["selected_summary"]["group_costs"][g], end["group_costs"][g]))
        checks.append(_entry(f"{tag}median r", new["r_median"], end["r_median"]))
        checks.append(_entry(f"{tag}smallest r", new["r_min"], end["r_min"]))
        checks.append(_entry(f"{tag}largest r", new["r_max"], end["r_max"]))
        checks.append(_entry(f"{tag}number of r values outside the band", new["outside_band"], end["outside_band"],
                             count=True))
        checks.append(_entry(f"{tag}noise-only cost", new["noise_only_summary"]["test_cost"],
                             levels["noise_only"]["test_cost"]))
        for g in cost.GROUPS:
            checks.append(_entry(f"{tag}noise-only group cost, {g}", new["noise_only_summary"]["group_costs"][g],
                                 levels["noise_only"]["group_costs"][g]))
        if new["reference"]:
            ref = levels["reference_model"]
            checks.append(_entry(f"{tag}reference cost", new["reference_summary"]["test_cost"], ref["test_cost"]))
            for g in cost.GROUPS:
                checks.append(_entry(f"{tag}reference group cost, {g}", new["reference_summary"]["group_costs"][g],
                                     ref["group_costs"][g]))
            for row in ref["motions"]:
                checks.append(_entry(f"{tag}reference cost, {row['motion']}",
                                     new["reference_summary"]["motion_costs"][row["motion"]], row["cost"]))
            checks.append(_entry(f"{tag}share removed", new["share_removed"], levels["share_removed"]))
    return checks


if __name__ == "__main__":
    result = check_signals(Path(__file__).resolve().parent)
    bad = [r for r in result if not r["ok"]]
    print(f"values recomputed from the signals: {len(result)}, different: {len(bad)}")
    for r in bad:
        print(r)
