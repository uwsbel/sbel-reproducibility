"""Recompute the PUMA test measures from the stored test records and predictions.

check_signals compares the recomputed values with the values in results/.
It covers the selected endpoint of each geometry setting, the noise-only cost
and the reference cost. The module needs numpy only. It runs no simulation.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = ("different_geometry", "same_geometry")
TOLERANCE = 1e-9


def _load(case_dir, name):
    spec = importlib.util.spec_from_file_location("puma_" + name, Path(case_dir) / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entry(name, recomputed, stored, count=False):
    if count:
        ok = int(recomputed) == int(stored)
    else:
        ok = abs(recomputed - stored) <= TOLERANCE * abs(stored)
    return {"name": name, "recomputed": float(recomputed), "stored": float(stored), "ok": bool(ok)}


def records(case_dir, folder, setting, motions, cost):
    """{motion: arrays} of one folder of signals/."""
    return {m: cost.load(Path(case_dir) / "signals" / folder / setting / f"{m}.npz") for m in motions}


def check_signals(case_dir=HERE):
    """Return a list of {"name", "recomputed", "stored", "ok"}; ok means a relative
    difference of at most 1e-9 (counts: equal)."""
    case_dir = Path(case_dir)
    cost = _load(case_dir, "cost")
    tables = _load(case_dir, "tables")
    motions = [m["id"] for m in _json(case_dir / "settings" / "motions.json")["test"]]
    noise_free = {s: records(case_dir, "test_noise_free", s, motions, cost) for s in SETTINGS}
    out = []
    for s in SETTINGS:
        noisy = records(case_dir, "test_noisy", s, motions, cost)
        pred = records(case_dir, "predictions_selected", s, motions, cost)
        end = _json(case_dir / "results" / s / "endpoints.json")
        sel = end["selected_seed"]
        run = next(r for r in end["runs"] if r["seed"] == sel)
        stored = tables.test_measures(case_dir, s)[sel]
        stored_summary = tables.endpoint_summary(stored, cost)
        groups = _json(case_dir / "results" / s / "group_costs.json")
        ref = _json(case_dir / "results" / s / "reference_levels.json")

        # Selected endpoint: test cost, cost per output group, r, RMS difference to the noise-free record.
        test_cost = cost.cost([(pred[m], noisy[m]) for m in motions])
        out.append(_entry(f"{s}: test cost of the selected endpoint", test_cost, run["test_cost"]))
        out.append(_entry(f"{s}: test cost from the table of test measures", stored_summary["test_cost"], run["test_cost"]))
        r_all, n_outside = [], 0
        for g in cost.GROUPS:
            value = float(np.mean([cost.group_cost(pred[m], noisy[m], g) for m in motions]))
            out.append(_entry(f"{s}: cost of the group {g}, selected endpoint", value, groups["selected_endpoint"][g]))
        for m in motions:
            for g in cost.GROUPS:
                r = cost.ratio(pred[m], noisy[m], g)
                n = cost.values(noisy[m], g)
                r_all.append(r)
                n_outside += abs(r - 1.0) > cost.band_half_width(n)
                out.append(_entry(f"{s}: r, {m}, {g}", r, stored[m][g]["r"]))
                out.append(_entry(f"{s}: cost, {m}, {g}", cost.group_cost(pred[m], noisy[m], g), stored[m][g]["cost"]))
                out.append(_entry(f"{s}: scored values, {m}, {g}", n, stored[m][g]["values"], count=True))
                out.append(_entry(f"{s}: RMS difference to the noise-free record, {m}, {g}",
                                  cost.rmse(pred[m], noise_free[s][m], g), stored[m][g]["rmse"]))
        out.append(_entry(f"{s}: median r", float(np.median(r_all)), stored_summary["r_median"]))
        out.append(_entry(f"{s}: number of r values outside the band", n_outside, stored_summary["outside"], count=True))

        # Noise-only cost: noise-free record of the teacher against its noisy record.
        noise_only = cost.cost([(noise_free[s][m], noisy[m]) for m in motions])
        out.append(_entry(f"{s}: noise-only cost", noise_only, ref["noise_only_cost"]))
        # Reference model: teacher parameters with the calibrated geometry. Its output is the
        # noise-free record of the same-geometry setting.
        reference = cost.cost([(noise_free["same_geometry"][m], noisy[m]) for m in motions])
        out.append(_entry(f"{s}: reference cost", reference, ref["reference_cost"]))
        for g in cost.GROUPS:
            value = float(np.mean([cost.group_cost(noise_free["same_geometry"][m], noisy[m], g) for m in motions]))
            out.append(_entry(f"{s}: cost of the group {g}, reference model", value, groups["reference_model"][g]))
            value = float(np.mean([cost.group_cost(noise_free[s][m], noisy[m], g) for m in motions]))
            out.append(_entry(f"{s}: cost of the group {g}, noise only", value, groups["noise_only"][g]))
        if ref["share_of_excess_removed"] is not None:
            out.append(_entry(f"{s}: share of the excess removed",
                              (reference - test_cost) / (reference - noise_only), ref["share_of_excess_removed"]))
    return out
