"""Differential evolution (DE) of one run, with the stopping rule on the validation cost V.

Settings: ../settings/optimizer.json ("de"), seeds.json, de_start_points.json. The search
is over z in [-1, 1]^17. A run is made of parts that end at the validation checks:
24 generations, then 4 at a time; part k calls SciPy with seed + k.
Usage: python de.py <different_geometry|same_geometry> <seed 1-4> [max generations]
"""
import csv
import json
import sys

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import config
import model
import objective

OPT = model.read_json(config.CASE_DIR / "settings" / "optimizer.json")["de"]
SEEDS = model.read_json(config.CASE_DIR / "settings" / "seeds.json")["de_seeds"]
START_Z = np.array(model.read_json(config.CASE_DIR / "settings" / "de_start_points.json")["z"], dtype=float)


def initial_population(seed):
    """The five development points, then Latin-hypercube points of the seed; infeasible ones are skipped."""
    need = OPT["population"] - len(START_Z)
    sampler = qmc.LatinHypercube(d=START_Z.shape[1], seed=seed)
    kept = []
    while len(kept) < need:
        for z in sampler.random(need) * 2.0 - 1.0:
            if model.feasible(model.x_from_z(z)):
                kept.append(z)
                if len(kept) == need:
                    break
    return np.vstack([START_Z, np.asarray(kept)])


class TrainingCost:
    """J(z) for SciPy, with a cache keyed by z rounded to 1e-12.

    Each part restarts SciPy from the population of the previous part. SciPy rescales
    the population, which changes the last bits of z; the cache returns the stored cost.
    """

    def __init__(self, data):
        self.data, self.cache, self.new = data, {}, []

    def __call__(self, z):
        key = np.round(np.asarray(z, dtype=float), 12).tobytes()
        if key not in self.cache:
            self.cache[key] = self.data.penalized_cost(model.x_from_z(z), "training")
            self.new.append(self.cache[key])
        return self.cache[key]


def stop_event(checks, max_generations):
    """checks: [(generation, V)]. Returns (event, generation) or (None, None).

    (i) V at or below the near-noise threshold; (ii) no decrease of V by more than 0.5 %
    below its reference value for 8 generations; (iii) the generation budget.
    """
    ref = g_ref = None
    for g, v in checks:
        if v <= OPT["near_noise_threshold"]:
            return "i", g
        if ref is None or v < (1.0 - OPT["improvement_fraction"]) * ref:
            ref, g_ref = v, g
        if g - g_ref >= OPT["generations_without_improvement"]:
            return "ii", g
        if g >= max_generations:
            return "iii", g
    return None, None


def select(histories):
    """histories: {seed: [(step, V, z), ...]} -> (seed, step, V, z) with the lowest V.

    Ties go to the earlier step, then to the lower seed.
    """
    best = None
    for seed in sorted(histories):
        for step, v, z in histories[seed]:
            if best is None or (v, step, seed) < best[0]:
                best = ((v, step, seed), seed, step, v, z)
    return best[1:]


def run(setting, seed_index, max_generations=None, out_dir=config.OUTPUT_DIR, data=None):
    """One DE run. Writes de_history.csv and de_result.json to <out_dir>/runs/<setting>/seed<i>/."""
    max_generations = max_generations or OPT["max_generations"]
    seed = SEEDS[f"seed{seed_index}"]
    data = data or objective.Data(setting, out_dir)
    cost = TrainingCost(data)
    pop = initial_population(seed)
    log, checks, generation, part = [], [], 0, 0

    def callback(intermediate_result):
        nonlocal generation
        feasible = [c < objective.INFEASIBLE_COST for c in cost.new]
        if generation == 0:  # the first new costs belong to the initial population
            first = cost.new[:OPT["population"]]
            log.append([0, min(first), "", sum(feasible[:OPT["population"]])])
            feasible = feasible[OPT["population"]:]
        generation += 1
        log.append([generation, float(intermediate_result.fun), "", sum(feasible)])
        cost.new.clear()
        print(f"generation {generation}: J {intermediate_result.fun:.6g}", flush=True)

    while True:
        first = OPT["first_check_generation"] if part == 0 else OPT["generations_between_checks"]
        requested = min(first, max_generations - generation)
        result = differential_evolution(
            cost, [(-1.0, 1.0)] * pop.shape[1], strategy=OPT["strategy"], init=pop.copy(), maxiter=requested,
            tol=OPT["tol"], atol=OPT["atol"], mutation=tuple(OPT["mutation"]), recombination=OPT["recombination"],
            seed=seed + part, polish=OPT["polish"], workers=OPT["workers"], updating=OPT["updating"], callback=callback)
        if int(result.nit) != requested:
            raise RuntimeError(f"SciPy ended after {result.nit} of {requested} generations")
        pop, part = np.asarray(result.population, dtype=float), part + 1
        z_best = np.asarray(result.x, dtype=float)
        v = data.penalized_cost(model.x_from_z(z_best), "validation")
        checks.append((generation, v, z_best.tolist(), float(result.fun)))
        log[-1][2] = v
        print(f"check at generation {generation}: V {v:.6g}", flush=True)
        event, g_stop = stop_event([(g, vv) for g, vv, _, _ in checks], max_generations)
        if event:
            break

    g_end, v_end, z_end, j_end = min(checks, key=lambda c: (c[1], c[0]))
    out = {"setting": setting, "seed": seed_index, "stop_generation": g_stop, "stop_event": event,
           "endpoint_generation": g_end, "endpoint_training_cost": j_end, "endpoint_validation_cost": v_end,
           "endpoint_z": z_end, "checks": [{"generation": g, "validation_cost": vv, "training_cost": jj}
                                           for g, vv, _, jj in checks]}
    folder = out_dir / "runs" / setting / f"seed{seed_index}"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "de_history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["generation", "best_training_cost", "validation_cost", "feasible_candidates"])
        w.writerows(log)
    (folder / "de_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
