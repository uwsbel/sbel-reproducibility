"""Tables of the PUMA case, recomputed from the files of this folder.

build_tables returns the numbers of every PUMA table of the paper, unrounded.
build_conclusions returns the PUMA number of the conclusions.
The module needs numpy only. It runs no simulation.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = ("different_geometry", "same_geometry")
# Display units of the RMS columns: micro-rad, mrad/s, mm/s^2, mrad/s, mN m.
DISPLAY_UNIT = {"joint_angle": 1e-6, "joint_rate": 1e-3, "specific_force": 1e-3,
                "angular_rate": 1e-3, "torque": 1e-3}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cost_module(case_dir):
    spec = importlib.util.spec_from_file_location("puma_cost", Path(case_dir) / "cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bounds(case_dir):
    """Names, centres and half-ranges of the 17 parameter intervals."""
    par = _json(Path(case_dir) / "settings" / "bounds.json")["parameters"]
    lower = np.array([p["lower"] for p in par])
    upper = np.array([p["upper"] for p in par])
    return [p["name"] for p in par], lower, upper, 0.5 * (lower + upper), 0.5 * (upper - lower)


def test_measures(case_dir, setting):
    """{seed: {motion: {group: row}}} of results/<setting>/test_measures.csv, in file order."""
    out = {}
    for row in _csv(Path(case_dir) / "results" / setting / "test_measures.csv"):
        rec = {"r": float(row["r"]), "cost": float(row["cost"]), "values": int(row["scored_values"]),
               "rmse": float(row["rmse_to_noise_free"])}
        out.setdefault(int(row["seed"]), {}).setdefault(row["motion"], {})[row["group"]] = rec
    return out


def endpoint_summary(measures, cost):
    """Test cost, r values and count outside the band of one endpoint.
    measures: {motion: {group: row}}."""
    motion_cost = {m: float(np.mean([g["cost"] for g in groups.values()])) for m, groups in measures.items()}
    r = [g["r"] for groups in measures.values() for g in groups.values()]
    outside = sum(abs(g["r"] - 1.0) > cost.band_half_width(g["values"])
                  for groups in measures.values() for g in groups.values())
    return {"test_cost": float(np.mean(list(motion_cost.values()))), "motion_cost": motion_cost,
            "r_median": float(np.median(r)), "r_min": min(r), "r_max": max(r),
            "outside": int(outside), "n_r": len(r)}


def geometry_error(case_dir):
    """Axis error (deg) and lever error along and perpendicular to the true axis (mm),
    per joint and side, calibrated geometry minus true geometry."""
    true = _json(Path(case_dir) / "geometry" / "true.json")["joints"]
    cal = _json(Path(case_dir) / "geometry" / "calibrated.json")["joints"]
    rows = []
    for jt, jc in zip(true, cal):
        for side in ("parent", "child"):
            a, at = np.array(jc[f"axis_{side}"]), np.array(jt[f"axis_{side}"])
            cos = abs(a @ at) / np.linalg.norm(a) / np.linalg.norm(at)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            dl = 1e3 * (np.array(jc[f"lever_{side}_m"]) - np.array(jt[f"lever_{side}_m"]))
            along = float(dl @ at)
            perp = float(np.linalg.norm(dl - along * at))
            rows.append({"joint": jt["joint"], "side": side, "axis_deg": angle,
                         "lever_along_mm": along, "lever_perpendicular_mm": perp})
    return rows


def truth_rows(case_dir, setting):
    """Per parameter: teacher value, selected estimate, relative error (%), error in
    half-ranges, and coefficient of variation over the four endpoints (%)."""
    names, _, _, centre, half = bounds(case_dir)
    teacher = {p["name"]: p["value"] for p in _json(Path(case_dir) / "teacher" / "teacher0.json")["parameters"]}
    end = _json(Path(case_dir) / "results" / setting / "endpoints.json")
    runs = sorted(end["runs"], key=lambda r: r["seed"])
    sel = [r["seed"] for r in runs].index(end["selected_seed"])
    rows = []
    for k, name in enumerate(names):
        x = np.array([centre[k] + half[k] * r["z"][k] for r in runs])
        err = 100.0 * (x[sel] - teacher[name]) / teacher[name]
        dz = abs(x[sel] - teacher[name]) / half[k]
        cv = 100.0 * np.std(x, ddof=1) / abs(np.mean(x))
        rows.append([teacher[name], float(x[sel]), abs(float(err)), float(dz), float(cv)])
    return rows


def levels(case_dir):
    """Noise-only cost, reference cost and test cost of the selected endpoints."""
    cost = _cost_module(case_dir)
    out = {}
    for s in SETTINGS:
        ref = _json(Path(case_dir) / "results" / s / "reference_levels.json")
        sel = _json(Path(case_dir) / "results" / s / "endpoints.json")["selected_seed"]
        out[s] = {"noise_only": ref["noise_only_cost"], "reference": ref["reference_cost"],
                  "selected": endpoint_summary(test_measures(case_dir, s)[sel], cost)["test_cost"]}
    d = out["different_geometry"]
    out["share_removed_pct"] = 100.0 * (d["reference"] - d["selected"]) / (d["reference"] - d["noise_only"])
    return out


def build_tables(case_dir=HERE):
    """Return {label: {row_index: [numbers]}} for the PUMA tables of the paper."""
    case_dir = Path(case_dir)
    cost = _cost_module(case_dir)
    tabs = {}
    ends = {s: _json(case_dir / "results" / s / "endpoints.json") for s in SETTINGS}
    meas = {s: test_measures(case_dir, s) for s in SETTINGS}
    summ = {s: {seed: endpoint_summary(m, cost) for seed, m in meas[s].items()} for s in SETTINGS}

    # Four seeds, both settings.
    rows = {}
    for s in SETTINGS:
        for run in sorted(ends[s]["runs"], key=lambda r: r["seed"]):
            e = summ[s][run["seed"]]
            rows[len(rows)] = [run["seed"], run["de"]["stop_generation"], run["de"]["endpoint_validation_cost"],
                               run["lm"]["endpoint_iteration"], run["training_cost"], run["validation_cost"],
                               e["test_cost"], e["r_median"], e["outside"], e["n_r"]]
    tabs["tab:puma_seeds"] = rows

    # Parameter intervals.
    names, lower, upper, _, _ = bounds(case_dir)
    tabs["tab:pumabounds"] = {k: [float(lower[k]), float(upper[k])] for k in range(len(names))}

    # Schur SNR of the training motions: range over the development and the confirmation points.
    snr = _json(case_dir / "results" / "map" / "schur_snr.json")
    dev = [i for i, p in enumerate(snr["prior_points"]) if p.startswith("development")]
    conf = [i for i, p in enumerate(snr["prior_points"]) if p.startswith("confirmation")]
    table = np.array(snr["schur_snr"])
    rows = {}
    for k in range(len(snr["parameters"])):
        rows[k] = [float(table[dev, k].min()), float(table[dev, k].max()),
                   float(table[conf, k].min()), float(table[conf, k].max())]
    cond = np.array(snr["condition_number"])
    rows[len(rows)] = [float(cond[dev].min()), float(cond[dev].max()),
                       float(cond[conf].min()), float(cond[conf].max())]
    tabs["tab:puma_map_snr"] = rows

    # Test cost with the teacher parameters and single components of the geometry error.
    comp = _json(case_dir / "results" / "different_geometry" / "geometry_components.json")["geometries"]
    tabs["tab:puma_geometry_components"] = {k: [g["test_cost"], g["group_cost"]["joint_angle"]]
                                            for k, g in enumerate(comp)}

    # Errors of the calibrated geometry.
    tabs["tab:puma_geometry_error"] = {k: [g["joint"], g["axis_deg"], g["lever_along_mm"], g["lever_perpendicular_mm"]]
                                       for k, g in enumerate(geometry_error(case_dir))}

    # Test results of the selected endpoints for each motion.
    rows = {}
    for s in SETTINGS:
        sel = ends[s]["selected_seed"]
        for motion, groups in meas[s][sel].items():
            rows[len(rows)] = [summ[s][sel]["motion_cost"][motion]] + \
                [groups[g]["rmse"] / DISPLAY_UNIT[g] for g in cost.GROUPS]
    tabs["tab:puma_test_motions"] = rows

    # Teacher values and selected estimates.
    tabs["tab:puma_truth_diff"] = dict(enumerate(truth_rows(case_dir, "different_geometry")))
    tabs["tab:puma_truth"] = dict(enumerate(truth_rows(case_dir, "same_geometry")))

    # PUMA row of the two shared tables.
    sd = summ["different_geometry"][ends["different_geometry"]["selected_seed"]]
    ss = summ["same_geometry"][ends["same_geometry"]["selected_seed"]]
    tabs["tab:case_scope"] = {0: [len(names), len(ends["different_geometry"]["runs"]),
                                  sd["r_min"], sd["r_max"], ss["r_min"], ss["r_max"]]}
    lev = levels(case_dir)
    tabs["tab:geometry_levels"] = {0: [lev["same_geometry"]["noise_only"],
                                       lev["different_geometry"]["reference"], lev["different_geometry"]["selected"],
                                       lev["share_removed_pct"], lev["same_geometry"]["selected"]]}
    return tabs


def build_conclusions(case_dir=HERE):
    """Share of the excess test cost of the reference model that the selected endpoint removes (%)."""
    return {"share_removed_pct": levels(case_dir)["share_removed_pct"]}
