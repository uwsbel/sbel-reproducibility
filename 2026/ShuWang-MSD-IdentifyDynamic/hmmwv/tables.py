"""Recompute the numbers of the HMMWV tables from the files of this folder.

build_tables(case_dir) returns the numbers of each table row, unrounded.
build_conclusions(case_dir) returns the HMMWV numbers of the conclusions.
Costs that need a simulation are read from the result files; all other values are arithmetic.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

PARAMETERS = ("mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz")
SETTINGS = ("different_geometry", "same_geometry")
TEACHERS = (0, 1, 2, 3)


def _cost_module(case_dir):
    spec = importlib.util.spec_from_file_location("hmmwv_cost", Path(case_dir) / "cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bounds(case_dir):
    rows = _json(Path(case_dir) / "settings" / "bounds.json")["parameters"]
    assert [r["name"] for r in rows] == list(PARAMETERS)
    return rows


def _select(runs):
    """Selected run: lowest validation cost, then the earlier generation, then the lower seed."""
    return min(runs, key=lambda r: (r["validation_cost"], r["endpoint_generation"], r["seed"]))


def _measures(rows, cost):
    """Per run: test cost, group costs, r values (motion order as stored)."""
    out = {}
    for row in rows:
        key = (int(row.get("teacher", 0)), row.get("setting", ""), int(row["seed"]))
        run = out.setdefault(key, {"cost": {}, "r": {}})
        run["cost"].setdefault(row["motion"], {})[row["group"]] = float(row["cost"])
        run["r"].setdefault(row["motion"], {})[row["group"]] = float(row["r"])
    for run in out.values():
        motions = list(run["cost"])
        assert len(motions) == 18 and all(len(run["cost"][m]) == 8 for m in motions)
        run["motion_costs"] = [float(np.mean([run["cost"][m][g] for g in cost.GROUPS])) for m in motions]
        run["test_cost"] = float(np.mean(run["motion_costs"]))
        run["group_costs"] = {g: float(np.mean([run["cost"][m][g] for m in motions])) for g in cost.GROUPS}
        r_all = [run["r"][m][g] for m in motions for g in cost.GROUPS]
        run["r_all"] = r_all
        run["r_median"] = float(np.median(r_all))
        run["outside"] = sum(abs(run["r"][m][g] - 1) > cost.band(g) for m in motions for g in cost.GROUPS)
    return out


def _levels(level, cost):
    """Noise-only cost T0 and reference cost Tt of one teacher and setting, by group and in total."""
    t0 = {g: level["noise_only"]["group_costs"][g] for g in cost.GROUPS}
    ref = level["reference_model"]
    if ref["equals_noise_free_record"]:
        tt = dict(t0)
    else:
        tt = {g: float(np.mean([m["group_costs"][g] for m in ref["motions"]])) for g in cost.GROUPS}
    return t0, tt


def _all_groups(group_costs):
    return float(np.mean(list(group_costs.values())))


def _without_upright(group_costs):
    return float(np.mean([v for g, v in group_costs.items() if g != "upright_relative_position"]))


def _load(case_dir):
    """All runs of the four teachers with their test measures, teacher values and levels."""
    case_dir = Path(case_dir)
    cost = _cost_module(case_dir)
    bounds = _bounds(case_dir)
    half = np.array([b["half_range"] for b in bounds])
    data = {}
    for setting in SETTINGS:
        end = _json(case_dir / "results" / setting / "endpoints.json")
        meas = _measures(_csv(case_dir / "results" / setting / "test_measures.csv"), cost)
        level = _json(case_dir / "results" / setting / "reference_levels.json")
        runs = [dict(r, **meas[(0, "", r["seed"])]) for r in end["runs"]]
        data[(0, setting)] = {"runs": runs, "levels": _levels(level, cost)}
    end = _json(case_dir / "results" / "teachers" / "endpoints.json")
    meas = _measures(_csv(case_dir / "results" / "teachers" / "test_measures.csv"), cost)
    levels = _json(case_dir / "results" / "teachers" / "reference_levels.json")["levels"]
    for level in levels:
        k, setting = level["teacher"], level["setting"]
        runs = [dict(r, **meas[(k, setting, r["seed"])]) for r in end["runs"]
                if r["teacher"] == k and r["setting"] == setting]
        data[(k, setting)] = {"runs": runs, "levels": _levels(level, cost)}
    for (k, setting), d in data.items():
        assert [r["seed"] for r in d["runs"]] == [1, 2, 3, 4]
        teacher = _json(case_dir / "teacher" / f"teacher{k}.json")
        d["teacher"] = np.array([p["value"] for p in teacher["parameters"]])
        d["selected"] = _select(d["runs"])
        for r in d["runs"]:
            est = np.array(r["estimate"])
            r["error_percent"] = 100 * np.abs(est - d["teacher"]) / np.abs(d["teacher"])
            r["dz"] = (est - d["teacher"]) / half
            r["signed_error_percent"] = 100 * (est - d["teacher"]) / np.abs(d["teacher"])
            r["error_mm"] = 1000 * (est - d["teacher"])        # meaningful for the center-of-mass positions (m)
        t0, tt = d["levels"]
        d["T0"], d["Tt"], d["Ts"] = _all_groups(t0), _all_groups(tt), d["selected"]["test_cost"]
    return case_dir, cost, bounds, data


def _four_seed_summary(case_dir, data):
    noise = _json(case_dir / "results" / "noise_only_cost.json")
    sd = noise["simulated"]["validation"]["standard_deviation"]
    rows = {}
    for i, setting in enumerate(SETTINGS):
        v0 = next(t["validation"] for t in noise["teachers"] if t["teacher"] == 0 and t["setting"] == setting)
        for j, r in enumerate(data[(0, setting)]["runs"]):
            rows[4 * i + j] = [r["seed"], r["stop_generation"], r["endpoint_generation"], r["training_cost"],
                               r["validation_cost"], r["test_cost"], (r["validation_cost"] - v0) / sd,
                               r["r_median"], r["outside"], len(r["r_all"])]
    return rows


def _draws(data):
    comz, iyy = PARAMETERS.index("COMz"), PARAMETERS.index("Iyy")
    diff, same = {}, {}
    for k in TEACHERS:
        d = data[(k, "different_geometry")]
        costs = [r["test_cost"] for r in d["runs"]]
        s = d["selected"]
        diff[k] = [k, float(d["teacher"][0]), d["T0"], d["Tt"], d["Ts"], min(costs), max(costs),
                   100 * (d["Tt"] - d["Ts"]) / (d["Tt"] - d["T0"]), float(np.mean(s["error_percent"])),
                   float(s["error_mm"][comz])]
        d = data[(k, "same_geometry")]
        costs = [r["test_cost"] for r in d["runs"]]
        s = d["selected"]
        same[k] = [k, d["T0"], d["Ts"], min(costs), max(costs), float(np.mean(s["error_percent"])),
                   float(s["error_mm"][comz]), float(s["signed_error_percent"][iyy])]
    return diff, same


def _geometry_error(case_dir):
    points = _csv(case_dir / "geometry" / "geometry_error.csv")
    groups = []
    for p in points:
        if p["group"] not in groups:
            groups.append(p["group"])
    rows = {}
    for i, group in enumerate(groups):
        row = []
        for axle in ("front", "rear"):
            sel = [p for p in points if p["group"] == group and p["axle"] == axle]
            norm = np.array([np.linalg.norm([float(p["dx_mm"]), float(p["dy_mm"]), float(p["dz_mm"])]) for p in sel])
            outward = np.array([float(p["outward_mm"]) for p in sel])
            row += [float(np.sqrt(np.mean(norm ** 2))), float(norm.max()), float(outward.mean())]
        rows[i] = row
    return rows


def _all_sets(case_dir):
    rows = {}
    for i, s in enumerate(_csv(case_dir / "results" / "map" / "all_sets.csv")):
        snr = {p: [float(s[f"{p}_development_{j}"]) for j in (1, 2, 3)] for p in PARAMETERS}
        values = [x for p in PARAMETERS for x in snr[p]]
        rows[i] = [int(s["rank"]), float(np.median(values))] + [min(snr[p]) for p in PARAMETERS]
    return rows


def _truth(data, setting):
    d = data[(0, setting)]
    est = np.array([r["estimate"] for r in d["runs"]])
    cv = 100 * est.std(axis=0, ddof=1) / np.abs(est.mean(axis=0))
    s = d["selected"]
    return {j: [float(d["teacher"][j]), s["estimate"][j], float(s["error_percent"][j]), float(abs(s["dz"][j])),
                float(cv[j])] for j in range(len(PARAMETERS))}


def _four_seed_parameters(data):
    rows = {}
    for i, setting in enumerate(SETTINGS):
        for j in range(len(PARAMETERS)):
            rows[7 * i + j] = [r["estimate"][j] for r in data[(0, setting)]["runs"]]
    return rows


def _group_costs(cost, data):
    d = data[(0, "different_geometry")]
    t0, tt = d["levels"]
    ts = d["selected"]["group_costs"]
    rows = {i: [t0[g], tt[g], ts[g]] for i, g in enumerate(cost.GROUPS)}
    rows[8] = [_all_groups(t0), _all_groups(tt), _all_groups(ts)]
    rows[9] = [_without_upright(t0), _without_upright(tt), _without_upright(ts)]
    return rows


def _axle(case_dir, cost):
    axle = _json(case_dir / "results" / "further" / "axle_split.json")["group_costs"]
    columns = [axle[c] for c in ("noise_only", "front_axle_calibrated", "rear_axle_calibrated", "both_axles_calibrated")]
    rows = {i: [c[g] for c in columns] for i, g in enumerate(cost.GROUPS)}
    rows[8] = [_all_groups(c) for c in columns]
    rows[9] = [_without_upright(c) for c in columns]
    return rows


def _noise(case_dir):
    groups = _json(case_dir / "settings" / "noise.json")["groups"]
    return {i: [g["components"], g["standard_deviation"]] for i, g in enumerate(groups)}


def _shared_with_polaris(case_dir, bounds):
    """Rows of the two tables that hold HMMWV and Polaris values; empty until the Polaris files exist."""
    out = {}
    other = case_dir.parent / "polaris"
    p_bounds = other / "settings" / "bounds.json"
    if p_bounds.exists():
        pb = _json(p_bounds)["parameters"]
        assert [b["name"] for b in pb] == list(PARAMETERS)
        out["tab:vehicle_bounds"] = {j: [bounds[j]["lower"], bounds[j]["upper"], pb[j]["lower"], pb[j]["upper"]]
                                     for j in range(len(PARAMETERS))}
    p_map = other / "results" / "map" / "schur_snr.json"
    if p_map.exists():
        both = []
        for path in (case_dir / "results" / "map" / "schur_snr.json", p_map):
            m = _json(path)
            assert list(m["parameters"]) == list(PARAMETERS) and len(m["prior_points"]) == 5
            both.append(np.array(m["schur_snr"], dtype=float))
        out["tab:vehicle_map_snr"] = {j: [float(both[0][:, j].min()), float(both[0][:, j].max()),
                                          float(both[1][:, j].min()), float(both[1][:, j].max())]
                                      for j in range(len(PARAMETERS))}
    return out


def build_tables(case_dir):
    """Return {table label: {row index: [numbers]}} for the HMMWV rows of the paper tables."""
    case_dir, cost, bounds, data = _load(case_dir)
    draws, draws_same = _draws(data)
    seeds = sorted(int(k[4:]) for k in _json(case_dir / "settings" / "seeds.json")["de_seeds"])
    optimizer = _json(case_dir / "settings" / "optimizer.json")
    sel = {s: data[(0, s)]["selected"] for s in SETTINGS}
    d, s = data[(0, "different_geometry")], data[(0, "same_geometry")]
    tables = {
        "tab:hmmwv_four_seed_summary": _four_seed_summary(case_dir, data),
        "tab:hmmwv_draws": draws,
        "tab:hmmwv_draws_same": draws_same,
        "tab:hmmwv_geometry_error": _geometry_error(case_dir),
        "tab:hmmwv_all_sets": _all_sets(case_dir),
        "tab:hmmwv_truth_diff": _truth(data, "different_geometry"),
        "tab:hmmwv_truth": _truth(data, "same_geometry"),
        "tab:hmmwv_four_seed_parameters": _four_seed_parameters(data),
        "tab:hmmwv_group_costs": _group_costs(cost, data),
        "tab:hmmwv_axle": _axle(case_dir, cost),
        "tab:vehicle_noise": _noise(case_dir),
        "tab:de_settings": {0: [seeds[0], seeds[-1]], 1: [optimizer["recombination"]]},
        "tab:case_scope": {1: [len(bounds), len(d["runs"]),
                               min(sel["different_geometry"]["r_all"]), max(sel["different_geometry"]["r_all"]),
                               min(sel["same_geometry"]["r_all"]), max(sel["same_geometry"]["r_all"])]},
        "tab:geometry_levels": {1: [s["T0"], d["Tt"], d["Ts"],
                                    100 * (d["Tt"] - d["Ts"]) / (d["Tt"] - d["T0"]), s["Ts"]]},
    }
    tables.update(_shared_with_polaris(case_dir, bounds))
    return tables


def build_conclusions(case_dir):
    """Return the HMMWV numbers of the conclusions."""
    case_dir, cost, bounds, data = _load(case_dir)
    mass = PARAMETERS.index("M")
    share, mass_error, ratios = [], [], []
    for k in TEACHERS:
        d = data[(k, "different_geometry")]
        share.append(100 * (d["Tt"] - d["Ts"]) / (d["Tt"] - d["T0"]))
        mass_error += [float(r["error_percent"][mass]) for r in d["runs"]]
        for setting in SETTINGS:
            d = data[(k, setting)]
            ratios.append(d["selected"]["test_cost"] / min(r["test_cost"] for r in d["runs"]))
    return {"share_removed_pct_range": [min(share), max(share)],
            "mass_error_pct_max": max(mass_error),
            "selected_over_lowest_test_cost": ratios}
