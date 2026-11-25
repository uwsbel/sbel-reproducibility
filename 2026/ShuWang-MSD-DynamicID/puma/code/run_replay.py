"""Replay check: the code in this folder reproduces the stored records and costs.

1. Regenerate the test records of the teacher (different geometry) and compare them with
   ../signals/test_noisy/ and ../signals/test_noise_free/.
2. Simulate the selected endpoint of each setting (../results/<setting>/endpoints.json) and
   recompute the training cost J, the validation cost V and the test cost T, the costs by
   motion and the test predictions (../signals/predictions_selected/).
Differences: max |new - stored| / max |stored| per array; |new - stored| / |stored| per cost.
Writes <OUTPUT_DIR>/replay/replay_report.json. Takes about 30 minutes.
"""
import json
import time

import numpy as np

import config
import cost
import model
import objective
import records

OUT = config.OUTPUT_DIR / "replay"
SIGNALS = config.CASE_DIR / "signals"
RECORD_SETTING = "different_geometry"


def array_difference(new, stored):
    """Largest relative difference over the arrays of two records."""
    assert set(new) == set(stored)
    return max(float(np.max(np.abs(new[k] - stored[k])) / max(np.max(np.abs(stored[k])), 1e-300)) for k in stored)


def relative(new, stored):
    return abs(new - stored) / abs(stored)


def main():
    started = time.time()
    report = {"records": {}, "endpoints": {}}

    # 1. Test records of the teacher.
    records.generate(OUT, RECORD_SETTING, "test")
    for noisy, name in ((True, "test_noisy"), (False, "test_noise_free")):
        folder = records.folder(OUT, RECORD_SETTING, "test", noisy)
        report["records"][name] = {p.stem: array_difference(cost.load(p), cost.load(SIGNALS / name / RECORD_SETTING / p.name))
                                   for p in sorted(folder.glob("*.npz"))}

    # 2. Selected endpoints.
    for setting in records.SETTINGS:
        stored = model.read_json(config.CASE_DIR / "results" / setting / "endpoints.json")
        run = next(r for r in stored["runs"] if r["seed"] == stored["selected_seed"])
        data = objective.Data(setting, OUT, ("training", "validation", "test"), {"test": SIGNALS / "test_noisy" / setting})
        x = model.x_from_z(run["z"])
        pred = {split: data.predictions(x, split) for split in ("training", "validation", "test")}
        f = data.residuals(x, pred["training"])
        by_motion = {split: data.motion_costs(x, split, pred[split]) for split in pred}
        new = {"training_cost": float(f @ f),
               "validation_cost": float(np.mean(list(by_motion["validation"].values()))),
               "test_cost": float(np.mean(list(by_motion["test"].values())))}
        stored_by_motion = {"training": run["training_cost_by_motion"], "validation": run["validation_cost_by_motion"]}
        report["endpoints"][setting] = {
            "seed": run["seed"],
            "costs": {k: {"recomputed": new[k], "stored": run[k], "relative_difference": relative(new[k], run[k])}
                      for k in new},
            "largest_relative_difference_of_the_costs_by_motion": {
                split: max(relative(by_motion[split][m], v) for m, v in stored_by_motion[split].items())
                for split in stored_by_motion},
            "test_predictions": {m: array_difference(p, cost.load(SIGNALS / "predictions_selected" / setting / f"{m}.npz"))
                                 for m, p in pred["test"].items()}}

    arrays = [v for group in report["records"].values() for v in group.values()]
    arrays += [v for e in report["endpoints"].values() for v in e["test_predictions"].values()]
    costs = [c["relative_difference"] for e in report["endpoints"].values() for c in e["costs"].values()]
    costs += [v for e in report["endpoints"].values()
              for v in e["largest_relative_difference_of_the_costs_by_motion"].values()]
    report["largest_relative_difference"] = {"arrays": max(arrays), "costs": max(costs)}
    report["tier_1"] = bool(max(arrays) <= 1e-9 and max(costs) <= 1e-9)
    report["costs_within_1e-3"] = bool(max(costs) <= 1e-3)
    report["elapsed_s"] = round(time.time() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("largest_relative_difference", "tier_1", "costs_within_1e-3")}, indent=1))


if __name__ == "__main__":
    main()
