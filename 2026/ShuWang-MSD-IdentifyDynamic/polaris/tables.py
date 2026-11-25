"""Recompute the numbers of the Polaris tables of the paper from the result files of this folder.

build_tables and build_conclusions return unrounded numbers in the order of the printed cells.
Stored costs are read; means, medians, counts, ranges, shares and errors are computed here.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

PARAMETERS = ("mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz")
SETTINGS = ("different_geometry", "same_geometry")
TEACHERS = ("teacher0", "teacher1", "teacher2", "teacher3")
GEOMETRY_GROUPS = ("Chassis-side arm points", "Outer arm points", "Tie-rod outer point", "Wheel center",
                   "Spring and damper lower points", "Tie-rod inner, spring and damper upper points")


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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------- data
def load_results(case_dir):
    """Results of the four teachers: data[teacher][setting] with the keys
    runs (list of run records), measures (rows of the test measures), levels, teacher (values), half_range."""
    case_dir = Path(case_dir)
    bounds = read_json(case_dir / "settings" / "bounds.json")["parameters"]
    assert tuple(b["name"] for b in bounds) == PARAMETERS
    half = np.array([b["half_range"] for b in bounds])
    others = case_dir / "results" / "teachers"
    other_runs = read_json(others / "endpoints.json")["teachers"]
    other_levels = read_json(others / "reference_levels.json")["teachers"]
    other_rows = read_csv(others / "test_measures.csv")
    data = {}
    for t in TEACHERS:
        values = np.array(read_json(case_dir / "teacher" / (t + ".json"))["values"])
        data[t] = {}
        for s in SETTINGS:
            if t == "teacher0":
                runs = read_json(case_dir / "results" / s / "endpoints.json")["runs"]
                levels = read_json(case_dir / "results" / s / "reference_levels.json")
                rows = read_csv(case_dir / "results" / s / "test_measures.csv")
            else:
                runs = other_runs[t][s]["runs"]
                levels = other_levels[t][s]
                rows = [x for x in other_rows if x["teacher"] == t and x["setting"] == s]
            data[t][s] = dict(runs=runs, levels=levels, measures=rows, teacher=values, half_range=half)
    return data


# ---------------------------------------------------------------------------------------------- arithmetic
def group_costs(rows, model):
    """Test cost of each output group: mean over the test motions."""
    return {g: float(np.mean([float(x["cost"]) for x in rows if x["model"] == model and x["group"] == g])) for g in GROUPS}


def test_cost(rows, model):
    """Test cost: the cost of a motion is the mean of its eight group costs; the test cost is the mean over the motions."""
    motions = sorted({x["motion"] for x in rows if x["model"] == model})
    per_motion = []
    for m in motions:
        c = {x["group"]: float(x["cost"]) for x in rows if x["model"] == model and x["motion"] == m}
        per_motion.append(np.mean([c[g] for g in GROUPS]))
    return float(np.mean(per_motion))


def r_list(rows, model):
    """All r values of one model: (group, r) for every test motion and output group."""
    return [(x["group"], float(x["r"])) for x in rows if x["model"] == model]


def outside_band(r):
    return sum(1 for g, v in r if abs(v - 1.0) > cost.band(g))


def selected_run(runs):
    """Run with the lowest validation cost; ties go to the earlier endpoint generation, then to the lower seed."""
    return min(runs, key=lambda x: (x["validation_cost"], x["endpoint_generation"], x["seed"]))


def run_test_costs(d):
    return {x["seed"]: test_cost(d["measures"], "seed%d" % x["seed"]) for x in d["runs"]}


def without_upright(groups):
    return float(np.mean([groups[g] for g in GROUPS if g != "upright_rel"]))


def errors(d, run):
    """Relative error (percent) and signed error in half-ranges of one endpoint."""
    est = np.array(run["parameters"])
    rel = 100.0 * np.abs(est - d["teacher"]) / np.abs(d["teacher"])
    dz = (est - d["teacher"]) / d["half_range"]
    return rel, dz


def signed_percent(d, run):
    """Signed relative error (percent) of one endpoint."""
    est = np.array(run["parameters"])
    return 100.0 * (est - d["teacher"]) / np.abs(d["teacher"])


def share_removed(d):
    """(reference cost - selected cost) / (reference cost - noise-only cost), different-geometry setting."""
    t0 = d["levels"]["noise_only_cost"]
    tt = test_cost(d["measures"], "reference")
    ts = run_test_costs(d)[selected_run(d["runs"])["seed"]]
    return (tt - ts) / (tt - t0)


# ---------------------------------------------------------------------------------------------- tables
def seed_table(case_dir, data):
    noise = read_json(Path(case_dir) / "results" / "noise_only_cost.json")
    sd = noise["simulated"]["validation"]["sd"]
    out = {}
    for i, s in enumerate(SETTINGS):
        d = data["teacher0"][s]
        v0 = noise["of_the_data"]["teacher0"][s]["validation"]
        for run in d["runs"]:
            model = "seed%d" % run["seed"]
            r = r_list(d["measures"], model)
            out[4 * i + run["seed"] - 1] = [
                run["seed"], run["stop_generation"], run["endpoint_generation"], run["training_cost"],
                run["validation_cost"], test_cost(d["measures"], model), (run["validation_cost"] - v0) / sd,
                float(np.median([v for _, v in r])), outside_band(r), len(r)]
    return out


def truth_table(d):
    runs = sorted(d["runs"], key=lambda x: x["seed"])
    sel = selected_run(runs)
    rel, dz = errors(d, sel)
    est = np.array([x["parameters"] for x in runs])
    cv = 100.0 * est.std(axis=0, ddof=1) / np.abs(est.mean(axis=0))
    return {j: [float(d["teacher"][j]), sel["parameters"][j], float(rel[j]), float(abs(dz[j])), float(cv[j])]
            for j in range(len(PARAMETERS))}


def parameter_table(data):
    out = {}
    for i, s in enumerate(SETTINGS):
        runs = sorted(data["teacher0"][s]["runs"], key=lambda x: x["seed"])
        for j in range(len(PARAMETERS)):
            out[7 * i + j] = [x["parameters"][j] for x in runs]
    return out


def group_cost_table(d):
    lv = d["levels"]
    t0 = lv["noise_only_group_costs"]
    tt = group_costs(d["measures"], "reference")
    sel = "seed%d" % selected_run(d["runs"])["seed"]
    ts = group_costs(d["measures"], sel)
    out = {i: [t0[g], tt[g], ts[g]] for i, g in enumerate(GROUPS)}
    out[8] = [lv["noise_only_cost"], test_cost(d["measures"], "reference"), test_cost(d["measures"], sel)]
    out[9] = [without_upright(t0), without_upright(tt), without_upright(ts)]
    return out


def axle_table(case_dir, d):
    rows = read_csv(Path(case_dir) / "results" / "further" / "axle_split.csv")
    cells = {"front": ("calibrated", "true", "true"), "rear": ("true", "calibrated", "true"),
             "both": ("calibrated", "calibrated", "calibrated")}
    cols = {}
    for name, key in cells.items():
        part = [dict(x, model=name) for x in rows if (x["front_geometry"], x["rear_geometry"], x["settling_pose"]) == key]
        cols[name] = (group_costs(part, name), test_cost(part, name))
    t0 = d["levels"]["noise_only_group_costs"]
    out = {i: [t0[g]] + [cols[c][0][g] for c in ("front", "rear", "both")] for i, g in enumerate(GROUPS)}
    out[8] = [d["levels"]["noise_only_cost"]] + [cols[c][1] for c in ("front", "rear", "both")]
    out[9] = [without_upright(t0)] + [without_upright(cols[c][0]) for c in ("front", "rear", "both")]
    return out


def draw_tables(data):
    diff, same = {}, {}
    for k, t in enumerate(TEACHERS):
        d = data[t]["different_geometry"]
        sel = selected_run(d["runs"])
        costs = run_test_costs(d)
        rel, dz = errors(d, sel)
        signed = signed_percent(d, sel)
        diff[k] = [k, float(d["teacher"][0]), d["levels"]["noise_only_cost"], test_cost(d["measures"], "reference"),
                   costs[sel["seed"]], min(costs.values()), max(costs.values()), 100.0 * share_removed(d),
                   float(rel.mean()), float(signed[PARAMETERS.index("Iyy")])]
        d = data[t]["same_geometry"]
        sel = selected_run(d["runs"])
        costs = run_test_costs(d)
        rel, dz = errors(d, sel)
        signed = signed_percent(d, sel)
        same[k] = [k, d["levels"]["noise_only_cost"], costs[sel["seed"]], min(costs.values()), max(costs.values()),
                   float(rel.mean()), float(signed[PARAMETERS.index("Iyy")]), float(signed[PARAMETERS.index("Izz")])]
    return diff, same


def scope_row(data):
    row = [len(PARAMETERS), len(data["teacher0"]["different_geometry"]["runs"])]
    for s in SETTINGS:
        d = data["teacher0"][s]
        r = [v for _, v in r_list(d["measures"], "seed%d" % selected_run(d["runs"])["seed"])]
        row += [min(r), max(r)]
    return row


def level_row(data):
    same, diff = data["teacher0"]["same_geometry"], data["teacher0"]["different_geometry"]
    return [same["levels"]["noise_only_cost"],
            test_cost(diff["measures"], "reference"), run_test_costs(diff)[selected_run(diff["runs"])["seed"]],
            100.0 * share_removed(diff), run_test_costs(same)[selected_run(same["runs"])["seed"]]]


def all_sets_table(case_dir):
    rows = read_csv(Path(case_dir) / "results" / "map" / "all_sets.csv")
    out = {}
    for rank in sorted({int(x["rank"]) for x in rows}):
        part = [x for x in rows if int(x["rank"]) == rank and x["prior_point"].startswith("development")]
        snr = [float(x["schur_snr"]) for x in part]
        smallest = [min(float(x["schur_snr"]) for x in part if x["parameter"] == p) for p in PARAMETERS]
        out[rank - 1] = [rank, float(np.median(snr))] + smallest
    return out


def geometry_error_table(case_dir):
    rows = read_csv(Path(case_dir) / "geometry" / "geometry_error.csv")
    out = {}
    for i, group in enumerate(GEOMETRY_GROUPS):
        cells = []
        for axle in ("front", "rear"):
            part = [x for x in rows if x["group"] == group and x["axle"] == axle]
            if not part:
                continue
            e = np.array([[float(x["dx_mm"]), float(x["dy_mm"]), float(x["dz_mm"])] for x in part])
            norm = np.linalg.norm(e, axis=1)
            outward = np.array([e[n, 1] * (1.0 if x["side"] == "L" else -1.0) for n, x in enumerate(part)])
            cells += [float(np.sqrt(np.mean(norm ** 2))), float(norm.max()), float(outward.mean())]
        out[i] = cells
    return out


def build_tables(case_dir):
    """Return {label: {row index: [numbers]}} for the Polaris tables and the Polaris rows of the shared tables."""
    data = load_results(case_dir)
    main_diff, main_same = data["teacher0"]["different_geometry"], data["teacher0"]["same_geometry"]
    draws_diff, draws_same = draw_tables(data)
    return {
        "tab:polaris_seeds": seed_table(case_dir, data),
        "tab:polaris_draws": draws_diff,
        "tab:polaris_draws_same": draws_same,
        "tab:polaris_geometry_error": geometry_error_table(case_dir),
        "tab:polaris_all_sets": all_sets_table(case_dir),
        "tab:polaris_truth_diff": truth_table(main_diff),
        "tab:polaris_truth": truth_table(main_same),
        "tab:polaris_parameters": parameter_table(data),
        "tab:polaris_group_costs": group_cost_table(main_diff),
        "tab:polaris_axle": axle_table(case_dir, main_diff),
        "tab:case_scope": {2: scope_row(data)},
        "tab:geometry_levels": {2: level_row(data)},
    }


def build_conclusions(case_dir):
    """Share of the excess test cost removed (range over the four teachers, percent), largest relative mass
    error over all different-geometry endpoints (percent), and for each teacher and geometry setting the test
    cost of the selected run divided by the lowest test cost of its four runs."""
    data = load_results(case_dir)
    shares = [100.0 * share_removed(data[t]["different_geometry"]) for t in TEACHERS]
    mass = []
    for t in TEACHERS:
        d = data[t]["different_geometry"]
        mass += [float(errors(d, run)[0][PARAMETERS.index("M")]) for run in d["runs"]]
    ratios = []
    for t in TEACHERS:
        for s in SETTINGS:
            d = data[t][s]
            costs = run_test_costs(d)
            ratios.append(costs[selected_run(d["runs"])["seed"]] / min(costs.values()))
    return {"share_removed_pct_range": [min(shares), max(shares)], "mass_error_pct_max": max(mass),
            "selected_over_lowest_test_cost": ratios}
