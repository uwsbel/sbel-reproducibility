"""Smoke run: a short pass through DE, LM and the map with the settings of the paper.

DE of seed 1 in the different-geometry setting with a budget of 2 generations, then one
LM iteration from its endpoint, then the map at the centre of the box for the shortest
training motion. Writes <OUTPUT_DIR>/smoke/smoke_report.json. Takes about 60 minutes.
"""
import csv
import json
import time

import numpy as np

import config
import de
import lm
import map as prior_map
import motions
import objective

OUT = config.OUTPUT_DIR / "smoke"
SETTING = "different_geometry"


def main():
    started = time.time()
    data = objective.Data(SETTING, OUT)
    de_result = de.run(SETTING, 1, max_generations=2, out_dir=OUT, data=data)
    folder = OUT / "runs" / SETTING / "seed1"
    with open(folder / "de_history.csv", encoding="utf-8") as fh:
        history = list(csv.DictReader(fh))
    lm_result = lm.run(SETTING, de_result["endpoint_z"], max_iterations=1, data=data, folder=folder)
    shortest = min(motions.motions("training"), key=lambda m: m["duration_s"])["id"]
    map_result = prior_map.run("development_1", [shortest], out_dir=OUT)

    values = [float(r["best_training_cost"]) for r in history] + [float(r["validation_cost"]) for r in history if r["validation_cost"]]
    values += [c[k] for c in lm_result["checks"] for k in ("training_cost", "validation_cost")]
    values += map_result["marginal_snr"] + map_result["schur_snr"]
    generations = [int(r["generation"]) for r in history if int(r["generation"]) > 0]
    report = {"de": {k: de_result[k] for k in ("stop_generation", "stop_event", "endpoint_generation",
                                                "endpoint_training_cost", "endpoint_validation_cost")},
              "de_generations_logged": generations,
              "lm": {k: lm_result[k] for k in ("stop_iteration", "stop_event", "endpoint_iteration",
                                                "training_cost", "validation_cost")},
              "map": {"point": "development_1", "motion": shortest, "schur_snr": map_result["schur_snr"]},
              "all_costs_finite": bool(np.all(np.isfinite(values))),
              "de_log_has_2_generations": generations == [1, 2],
              "elapsed_s": round(time.time() - started)}
    report["pass"] = report["all_costs_finite"] and report["de_log_has_2_generations"]
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
