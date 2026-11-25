"""Differential evolution of one run, with the stopping rule and the endpoint chosen by the validation cost.

Settings: settings/optimizer.json and settings/seeds.json. The search runs in calls of SciPy
differential_evolution: the first call covers 16 generations with the run seed, each later call 8 generations
with seed + k. V is checked at generation 24 and then every 4 generations (a check after 4 generations of a call
is the first half of that call). Usage:
    python de.py <setting> <seed1..seed4> [--generations N]     (N: one call of N generations, no checks)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import config
import model
import objective

SETTINGS = model.CASE_DIR / "settings"
OPT = json.loads((SETTINGS / "optimizer.json").read_text(encoding="utf-8"))
CENTER, HALF = model.bounds()


def initial_population(seed):
    """32 Latin-hypercube points of the seed in [-1, 1]^7; the first three rows are the development points."""
    pop = 2 * qmc.LatinHypercube(d=7, seed=seed).random(n=OPT["population"]) - 1
    points = json.loads((SETTINGS / "prior_points.json").read_text(encoding="utf-8"))["prior_points"]
    pop[:3] = np.array([p["z"] for p in points if p["role"] == "development"])
    return pop


def assess(checks):
    """Stopping rule on the checked generations [(generation, V)]. Returns (stop generation or None, event)."""
    rule = OPT["stop_rule"]
    first = OPT["validation_checks"]["first_generation"]
    best, last = checks[0][1], first
    for g, v in checks:
        if g > first and v < (1 - rule["event_ii_relative_improvement"]) * best:
            best, last = v, g
        if v <= rule["event_i_validation_cost_at_most"]:
            return g, "i"
        if g - last >= rule["event_ii_generations"]:
            return g, "ii"
        if g == rule["event_iii_generation_cap"]:
            return g, "iii"
    return None, None


class Run:
    """One DE run: training cost of z (remembered by exact z), the ordered log, and V at the checks."""

    def __init__(self, setting, seed_name):
        self.setting = setting
        self.seed = json.loads((SETTINGS / "seeds.json").read_text(encoding="utf-8"))["de_seeds"][seed_name]
        self.sim = objective.Model()
        self.costs, self.log, self.generations, self.checks = {}, [], [], {}

    def cost(self, z):
        key = np.asarray(z, dtype=np.float64).tobytes()
        if key not in self.costs:
            started = time.perf_counter()
            self.costs[key] = objective.training_cost(CENTER + HALF * z, setting=self.setting, sim=self.sim)
            self.log.append(dict(z=list(map(float, z)), cost=self.costs[key], seconds=time.perf_counter() - started))
            print(f"evaluation {len(self.log)}: {self.costs[key]:.6g}", flush=True)
        return self.costs[key]

    def call(self, population, generations, seed, stop_after=None):
        """One SciPy call from a population; stop_after ends it early after that many generations."""
        count = [0]

        def callback(xk, convergence):
            count[0] += 1
            best = self.costs.get(np.asarray(xk, dtype=np.float64).tobytes())
            self.generations.append(dict(call_seed=seed, generation_in_call=count[0], best_training_cost=best))
            return stop_after is not None and count[0] == stop_after

        r = differential_evolution(self.cost, [(-1.0, 1.0)] * 7, strategy=OPT["strategy"], init=np.array(population, dtype=float),
                                   maxiter=generations, tol=OPT["tolerance"], mutation=tuple(OPT["mutation"]),
                                   recombination=OPT["recombination"], seed=seed, polish=OPT["polishing"], workers=1,
                                   updating=OPT["updating"], callback=callback)
        assert r.nit == (stop_after or generations), r.message
        return np.array(r.population), np.array(r.x), float(r.fun)

    def check(self, g, x, j):
        """Validation cost of the best member at generation g (not simulated again for a repeated z)."""
        key = x.tobytes()
        v = next((c["V"] for c in self.checks.values() if c["key"] == key), None)
        if v is None:
            v = objective.validation_cost(CENTER + HALF * x, setting=self.setting, sim=self.sim)
        self.checks[g] = dict(key=key, generation=g, J=j, V=v, z=x.tolist(), parameters=(CENTER + HALF * x).tolist())
        print(f"check generation {g}: J {j:.6g}  V {v:.6g}", flush=True)
        return v

    def search(self):
        """Run to the stop; the endpoint is the checked generation with the lowest V (ties: earlier)."""
        pop, x, j = self.call(initial_population(self.seed), 16, self.seed)
        anchor, x, j = self.call(pop, 8, self.seed + 1)
        g = 24
        self.check(g, x, j)
        while True:
            stop, event = assess([(c["generation"], c["V"]) for c in self.checks.values()])
            if stop is not None:
                break
            k = (g - 16) // 8 + 1
            if max(self.checks) == g:            # first half of the next call, from the same population
                _, x, j = self.call(anchor, 8, self.seed + k, stop_after=4)
                self.check(g + 4, x, j)
            else:                                # the whole call; its first half repeats known evaluations
                anchor, x, j = self.call(anchor, 8, self.seed + k)
                g += 8
                self.check(g, x, j)
        endpoint = min((c for c in self.checks.values() if c["generation"] <= stop), key=lambda c: (c["V"], c["generation"]))
        return dict(seed=self.seed, stop_generation=stop, stop_event=event,
                    endpoint={k: v for k, v in endpoint.items() if k != "key"},
                    checks=[{k: v for k, v in c.items() if k != "key"} for c in self.checks.values()])


def select(runs):
    """Selected model of a setting: the run with the lowest endpoint V (ties: earlier generation, lower seed)."""
    return min(runs, key=lambda r: (r["endpoint"]["V"], r["endpoint"]["generation"], r["seed"]))


def main(setting, seed_name, generations=None):
    run = Run(setting, seed_name)
    out = Path(config.WORK_DIR) / "de"
    out.mkdir(parents=True, exist_ok=True)
    if generations is None:
        result = run.search()
    else:
        run.call(initial_population(run.seed), generations, run.seed)
        result = dict(seed=run.seed, generations=generations)
    result.update(setting=setting, generation_log=run.generations, evaluations=run.log)
    name = f"de_{setting}_{seed_name}" + ("" if generations is None else f"_{generations}_generations")
    (out / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--generations") + 1]) if "--generations" in args else None
    main(args[0], args[1], n)
