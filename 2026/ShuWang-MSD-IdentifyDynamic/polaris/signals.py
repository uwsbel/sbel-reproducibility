"""Recompute the test costs and r of the selected Polaris endpoints from the stored signals.

check_signals scores the stored predictions against the noisy and the noise-free test records
and compares every value with results/. numpy only. No simulation.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SETTINGS = ("different_geometry", "same_geometry")


def load_module(name):
    """Load a module of this folder by its path (the three cases have modules with the same names)."""
    path = Path(__file__).resolve().parent / (name + ".py")
    spec = importlib.util.spec_from_file_location("polaris_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cost = load_module("cost")
GROUPS = cost.GROUPS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_signals(folder, motion):
    with np.load(Path(folder) / (motion + ".npz"), allow_pickle=False) as f:
        return {g: np.array(f[g]) for g in GROUPS}


def score(model, data):
    """Cost, group costs and r of one motion."""
    total, groups = cost.cost(model, data)
    return total, groups, {g: cost.r_value(g, model[g], data[g]) for g in GROUPS}


def check_signals(case_dir, tolerance=1e-9):
    """Return a list of {"name", "recomputed", "stored", "ok"}; ok = relative difference <= tolerance (counts: equal)."""
    case_dir = Path(case_dir)
    motions = [m["name"] for m in read_json(case_dir / "settings" / "motions.json")["test"]]
    out = []

    def add(name, recomputed, stored, count=False):
        ok = recomputed == stored if count else abs(recomputed - stored) <= tolerance * abs(stored)
        out.append({"name": name, "recomputed": float(recomputed), "stored": float(stored), "ok": bool(ok)})

    for s in SETTINGS:
        res = case_dir / "results" / s
        sig = case_dir / "signals"
        endpoints = read_json(res / "endpoints.json")
        run = next(x for x in endpoints["runs"] if x["selected"])
        sel = endpoints["selected"]
        levels = read_json(res / "reference_levels.json")
        stored_groups = read_json(res / "group_costs.json")["test_cost_by_group"][sel]
        with open(res / "test_measures.csv", newline="", encoding="utf-8") as f:
            stored = {(x["model"], x["motion"], x["group"]): x for x in csv.DictReader(f)}

        def compare_rows(model, motion, against_noisy, against_noise_free=None):
            for g in GROUPS:
                x = stored[(model, motion, g)]
                add(f"{s}: {model}, {motion}, {g}: cost", against_noisy[1][g], float(x["cost"]))
                add(f"{s}: {model}, {motion}, {g}: r", against_noisy[2][g], float(x["r"]))
                if against_noise_free is not None:
                    add(f"{s}: {model}, {motion}, {g}: cost against the noise-free record", against_noise_free[1][g], float(x["cost_noise_free"]))
                    add(f"{s}: {model}, {motion}, {g}: r against the noise-free record", against_noise_free[2][g], float(x["r_noise_free"]))

        selected, noise, reference = [], [], []
        for m in motions:
            noisy = load_signals(sig / "test_noisy" / s, m)
            clean = load_signals(sig / "test_noise_free" / s, m)
            pred = load_signals(sig / "predictions_selected" / s, m)
            a = score(pred, noisy)
            compare_rows(sel, m, a, score(pred, clean))
            selected.append(a)
            noise.append(score(clean, noisy))
            if s == "different_geometry":
                ref = load_signals(sig / "predictions_reference" / s, m)
                b = score(ref, noisy)
                compare_rows("reference", m, b, score(ref, clean))
            else:                                 # same geometry: the reference model is the teacher itself
                b = score(clean, noisy)
                compare_rows("reference", m, b)
            reference.append(b)

        mean_cost = lambda rows: float(np.mean([x[0] for x in rows]))
        mean_group = lambda rows, g: float(np.mean([x[1][g] for x in rows]))
        ts, t0, tt = mean_cost(selected), mean_cost(noise), mean_cost(reference)
        add(f"{s}: test cost of the selected endpoint", ts, run["test_cost"])
        add(f"{s}: noise-only cost", t0, levels["noise_only_cost"])
        add(f"{s}: cost of the reference model", tt, levels["reference_cost"])
        for g in GROUPS:
            add(f"{s}: selected endpoint, group cost {g}", mean_group(selected, g), stored_groups[g])
            add(f"{s}: noise only, group cost {g}", mean_group(noise, g), levels["noise_only_group_costs"][g])
            add(f"{s}: reference model, group cost {g}", mean_group(reference, g), levels["reference_group_costs"][g])
        r = [(g, x[2][g]) for x in selected for g in GROUPS]
        values = [v for _, v in r]
        add(f"{s}: median r of the selected endpoint", float(np.median(values)), run["test_r_median"])
        add(f"{s}: smallest r of the selected endpoint", min(values), run["test_r_min"])
        add(f"{s}: largest r of the selected endpoint", max(values), run["test_r_max"])
        add(f"{s}: r values outside the band", sum(1 for g, v in r if abs(v - 1.0) > cost.band(g)), run["test_r_outside_band"], count=True)
        if levels["share_removed"] is not None:
            add(f"{s}: share of the excess cost removed", (tt - ts) / (tt - t0), levels["share_removed"])
    return out


if __name__ == "__main__":
    results = check_signals(Path(__file__).resolve().parent)
    bad = [x for x in results if not x["ok"]]
    print("values recomputed from the signals:", len(results), "different:", len(bad))
    for x in bad:
        print(x)
