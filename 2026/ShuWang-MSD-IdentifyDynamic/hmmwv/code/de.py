"""Differential evolution with the stopping rule and the endpoint chosen by the validation cost.

The search follows settings/optimizer.json: scipy best1bin, 32 members, immediate updating. It runs in
chunks (16 generations, then 8) with the integer seed + chunk index. From generation 24 on, the best
member is checked every 4 generations on the validation motions. The run stops at event (i), (ii)
or (iii); the endpoint is the checked member with the lowest V. select() picks the endpoint over seeds.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import config
import model
import objective
import records

SETTINGS = model.CASE_DIR / "settings"
OPT = json.loads((SETTINGS / "optimizer.json").read_text(encoding="utf-8"))
BOUNDS = json.loads((SETTINGS / "bounds.json").read_text(encoding="utf-8"))["parameters"]
CENTER = np.array([p["center"] for p in BOUNDS])
HALF = np.array([p["half_range"] for p in BOUNDS])
DE_SEEDS = json.loads((SETTINGS / "seeds.json").read_text(encoding="utf-8"))["de_seeds"]
PRIOR = json.loads((SETTINGS / "prior_points.json").read_text(encoding="utf-8"))["points"]
EVENTS = {"i": "V at most the noise level", "ii": "no decrease of V", "iii": "generation budget"}


def initial_population(seed):
    """Three development prior points and 29 Latin-hypercube points in z (the hypercube is drawn for 32 rows)."""
    pop = 2 * qmc.LatinHypercube(d=7, seed=seed).random(n=OPT["population"]) - 1
    pop[:3] = np.array([p["z"] for p in PRIOR if p["role"] == "development"])
    return pop


def assess(checks, cap):
    """Stop rule over the checks [(generation, V), ...]. Returns (stop generation or None, event, endpoint check)."""
    rule = OPT["stop_rule"]
    every = OPT["validation_checks"]["every_generations"]
    first = checks[0]["generation"]
    best, last = checks[0]["V"], first
    stop = event = None
    for c in checks:
        g, v = c["generation"], c["V"]
        if g > first and v < (1 - rule["event_ii_relative_decrease"]) * best:
            best, last = v, g
        if v <= rule["event_i_validation_cost_at_most"]:
            event = "i"
        elif g - last >= rule["event_ii_generations_without_decrease"]:
            event = "ii"
        elif g == cap:
            event = "iii"
        if event:
            stop = g
            break
    assert all(b["generation"] - a["generation"] == every for a, b in zip(checks, checks[1:]))
    upto = stop if stop is not None else checks[-1]["generation"]
    endpoint = min((c for c in checks if c["generation"] <= upto), key=lambda c: (c["V"], c["generation"]))
    return stop, event, endpoint


class Search:
    """One DE run of one seed: training records, the model geometry, and the bookkeeping of the population costs."""

    def __init__(self, setting, label, teacher=0):
        self.setting, self.label, self.teacher = setting, label, teacher
        self.seed = DE_SEEDS[f"seed{label}"]
        self.geometry = model.load_geometry("calibrated")          # the model always uses the calibrated geometry
        self.train = (records.motions("training"), records.load(setting, "training", teacher))
        self.memo = {}           # z bytes -> (cost, stopped early); a repeated generation reuses its costs
        self.best = {}           # generation -> lowest training cost of the population
        self.offset = 0          # generations before the current chunk
        self.evaluations = 0

    def evaluate(self, z, limit):
        key = z.tobytes()
        if key in self.memo:
            cost, early = self.memo[key]
            if not early or (limit is not None and cost > limit * (1 + 1e-12)):
                return cost, early
        self.evaluations += 1
        cost, status = objective.training_cost(CENTER + HALF * z, self.geometry, *self.train, reject_above=limit)
        early = status == "stopped early"
        if not early or key not in self.memo:
            self.memo[key] = (cost, early)
        return cost, early

    def objective(self, z):
        """Cost of one scipy call. The first 32 calls evaluate the initial members; after that, call n is the
        trial of member (n - 32) mod 32, and its cost limit for early rejection is that member's cost."""
        n, size = self.n, OPT["population"]
        i = (n - size) % size if n >= size else None
        cost, early = self.evaluate(np.asarray(z, dtype=np.float64), None if i is None else float(self.E[i]))
        if n < size:
            self.E = np.append(self.E, cost)
            if n == size - 1:                             # scipy moves the best member to index 0
                j = int(np.argmin(self.E))
                self.E[[0, j]] = self.E[[j, 0]]
                self.best.setdefault(self.offset, float(self.E[0]))
        elif cost <= self.E[i]:
            assert not early
            self.E[i] = cost
            if cost <= self.E[0]:
                j = int(np.argmin(self.E))
                self.E[[0, j]] = self.E[[j, 0]]
        self.n = n + 1
        return cost

    def chunk(self, init, generations, rng_seed, stop_after=None):
        """One scipy call over `generations` generations (or the first `stop_after` of them)."""
        self.n, self.E, count = 0, np.zeros(0), [0]

        def callback(intermediate_result):
            count[0] += 1
            self.best[self.offset + count[0]] = float(intermediate_result.fun)
            if stop_after is not None and count[0] == stop_after:
                raise StopIteration

        r = differential_evolution(self.objective, [(-1.0, 1.0)] * 7, strategy=OPT["strategy"], init=np.array(init, dtype=float),
                                   maxiter=generations, tol=OPT["tolerance"], mutation=tuple(OPT["mutation"]),
                                   recombination=OPT["recombination"], seed=rng_seed, polish=OPT["polishing"], workers=1,
                                   updating=OPT["population_updating"], callback=callback)
        assert r.nit == count[0] == (stop_after or generations), "scipy ended the chunk early"
        assert np.array_equal(self.E, r.population_energies)
        return {"population": np.array(r.population), "x": np.array(r.x), "cost": float(r.fun),
                "generation": self.offset + count[0]}

    def check(self, state, done):
        """Validation cost of the best member; a member checked before keeps its V."""
        z = state["x"]
        prior = next((c for c in done if np.array_equal(c["z"], z)), None)
        if prior is not None:
            v = prior["V"]
        else:
            motions = records.motions("validation")
            v = objective.validation_cost(CENTER + HALF * z, self.geometry, motions,
                                          records.load(self.setting, "validation", self.teacher))[0]
        print(f"check generation {state['generation']}: J {state['cost']:.6f} V {v:.6f}", flush=True)
        return {"generation": state["generation"], "J": state["cost"], "V": v, "z": z}

    def run(self, budget=None):
        """The paper's schedule. budget = generation budget (event iii); default from optimizer.json."""
        rule, checks_at = OPT["stop_rule"], OPT["validation_checks"]
        cap = budget or rule["event_iii_generations"]
        first, nxt, every = OPT["generation_chunks"]["first"], OPT["generation_chunks"]["next"], checks_at["every_generations"]
        state = self.chunk(initial_population(self.seed), min(first, cap), self.seed)
        k = 1
        while state["generation"] < min(checks_at["first_generation"], cap):     # up to the first check
            self.offset = state["generation"]
            state = self.chunk(state["population"], min(nxt, cap - self.offset), self.seed + k)
            k += 1
        if cap < checks_at["first_generation"]:
            return {"stop_generation": state["generation"], "stop_event": "iii", "endpoint": None, "checks": []}
        anchor, checks = state, []
        checks.append(self.check(anchor, checks))
        while True:
            stop, event, endpoint = assess(checks, cap)
            if stop is not None:
                break
            self.offset = anchor["generation"]
            if checks[-1]["generation"] == anchor["generation"]:      # first half of the next chunk
                checks.append(self.check(self.chunk(anchor["population"], nxt, self.seed + k, stop_after=every), checks))
            else:                                                     # the whole chunk; its first half repeats from memory
                anchor = self.chunk(anchor["population"], nxt, self.seed + k)
                k += 1
                checks.append(self.check(anchor, checks))
        return {"stop_generation": stop, "stop_event": event, "endpoint": endpoint, "checks": checks}

    def write(self, result, out):
        out.mkdir(parents=True, exist_ok=True)
        name = f"{self.setting}_seed{self.label}"
        v = {c["generation"]: c["V"] for c in result["checks"]}
        with (out / f"de_log_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["generation", "best_training_cost", "validation_cost"])
            for g in sorted(self.best):
                if g <= result["stop_generation"]:
                    w.writerow([g, repr(self.best[g]), repr(v[g]) if g in v else ""])
        e = result["endpoint"]
        summary = {"setting": self.setting, "seed": self.label, "stop_generation": result["stop_generation"],
                   "stop_event": result["stop_event"], "stop_event_meaning": EVENTS[result["stop_event"]],
                   "new_candidates_evaluated": self.evaluations,
                   "endpoint": None if e is None else {"generation": e["generation"], "training_cost": e["J"],
                                                       "validation_cost": e["V"], "z": e["z"].tolist(),
                                                       "estimate": (CENTER + HALF * e["z"]).tolist()}}
        (out / f"de_result_{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return summary


def run(setting, seed_label, budget=None, teacher=0):
    """Run one seed (seed_label 1 to 4) and write its log and result to <OUTPUT_DIR>/de/."""
    s = Search(setting, seed_label, teacher)
    return s.write(s.run(budget), Path(config.OUTPUT_DIR) / "de")


def select(results):
    """The endpoint with the lowest V over the seeds; a tie goes to the earlier generation, then the lower seed."""
    done = [r for r in results if r["endpoint"]]
    return min(done, key=lambda r: (r["endpoint"]["validation_cost"], r["endpoint"]["generation"], r["seed"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", choices=["different_geometry", "same_geometry"], required=True)
    ap.add_argument("--seed", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument("--generations", type=int, default=None, help="generation budget (default: the paper's 120)")
    a = ap.parse_args()
    print(json.dumps(run(a.setting, a.seed, a.generations), indent=1))
