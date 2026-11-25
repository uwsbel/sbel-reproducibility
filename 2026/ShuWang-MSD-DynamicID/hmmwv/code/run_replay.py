"""Replay check: regenerate the records of the main teacher and recompute J, V and T of the selected endpoints.

For each geometry setting: simulate the teacher on the 41 motions (records.py), compare the test records
with signals/, then simulate the selected endpoint of results/<setting>/endpoints.json on the training,
validation and test motions and compare J, V, T and the test predictions with the stored values.
Writes <OUTPUT_DIR>/replay_<setting>.json per setting and <OUTPUT_DIR>/replay_report.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
import de
import model
import objective
import records

SETTINGS = ("different_geometry", "same_geometry")
SIGNALS = model.CASE_DIR / "signals"
TIER_1 = 1e-9


def rel_diff(a, b):
    """Largest absolute difference divided by the largest absolute value of the stored array."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(abs(a - b)) / max(float(np.max(abs(b))), 1e-300))


def scored(rec):
    return {g: v[100::4] for g, v in rec.items()}


def compare_with_signals(arrays, folder, ids):
    """Largest relative difference per output group between scored samples and the stored signals."""
    worst = {g: 0.0 for g in model.GROUPS}
    for a, i in zip(arrays, ids):
        with np.load(folder / f"{i}.npz", allow_pickle=False) as f:
            for g in model.GROUPS:
                worst[g] = max(worst[g], rel_diff(scored(a)[g], f[g]))
    return worst


def replay(setting):
    started = time.time()
    records.generate(setting, 0)
    test_ids = [e["id"] for e in records.motion_entries("test")]
    out = {"setting": setting, "test_records": {
        "noisy": compare_with_signals(records.load(setting, "test"), SIGNALS / "test_noisy" / setting, test_ids),
        "noise_free": compare_with_signals(records.load(setting, "test", kind="noise_free"), SIGNALS / "test_noise_free" / setting, test_ids)}}

    stored = json.loads((model.CASE_DIR / "results" / setting / "endpoints.json").read_text(encoding="utf-8"))
    run = next(r for r in stored["runs"] if r["selected"])
    params = de.CENTER + de.HALF * np.array(run["z"])
    geometry = model.load_geometry("calibrated")
    J, _ = objective.training_cost(params, geometry, records.motions("training"), records.load(setting, "training"))
    V, _ = objective.validation_cost(params, geometry, records.motions("validation"), records.load(setting, "validation"))
    predictions = []
    pred_dir = Path(config.OUTPUT_DIR) / "predictions" / setting
    pred_dir.mkdir(parents=True, exist_ok=True)
    for motion in records.motions("test"):
        r = model.simulate(params, geometry, motion)
        assert r["valid"], r["reason"]
        predictions.append(r["outputs"])
        np.savez_compressed(pred_dir / f"{motion['id']}.npz", **r["outputs"])
    T, _ = objective.test_cost(predictions, records.load(setting, "test"))
    recomputed = {"training_cost": J, "validation_cost": V, "test_cost": T}
    out.update(selected_seed=run["seed"], estimate_relative_difference=rel_diff(params, run["estimate"]),
               costs={k: {"recomputed": v, "stored": run[k], "relative_difference": abs(v - run[k]) / abs(run[k])}
                      for k, v in recomputed.items()},
               test_predictions=compare_with_signals(predictions, SIGNALS / "predictions_selected" / setting, test_ids),
               wall_s=time.time() - started)
    worst = max([d for part in ("test_records",) for kind in out[part].values() for d in kind.values()]
                + list(out["test_predictions"].values()) + [c["relative_difference"] for c in out["costs"].values()])
    out["largest_relative_difference"] = worst
    out["tier_1"] = worst <= TIER_1
    return out


def main(settings):
    out_dir = Path(config.OUTPUT_DIR)
    parts = {}
    for s in settings:
        path = out_dir / f"replay_{s}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(replay(s), indent=1), encoding="utf-8")
        parts[s] = json.loads(path.read_text(encoding="utf-8"))
    if all((out_dir / f"replay_{s}.json").exists() for s in SETTINGS):
        report = {"criterion_tier_1": "relative difference <= 1e-9 for the regenerated test records (scored samples against signals/), "
                                      "the test predictions of the selected endpoint, and J, V, T against results/",
                  "settings": {s: json.loads((out_dir / f"replay_{s}.json").read_text(encoding="utf-8")) for s in SETTINGS}}
        report["tier_1"] = all(p["tier_1"] for p in report["settings"].values())
        (out_dir / "replay_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("tier 1:", report["tier_1"])
    for s, p in parts.items():
        print(s, {k: (c["recomputed"], c["relative_difference"]) for k, c in p["costs"].items()}, "largest", p["largest_relative_difference"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", choices=SETTINGS, help="replay one setting only (default: both)")
    a = ap.parse_args()
    main([a.setting] if a.setting else list(SETTINGS))
