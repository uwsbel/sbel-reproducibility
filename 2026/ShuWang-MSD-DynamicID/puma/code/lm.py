"""Levenberg-Marquardt (LM) refinement of a DE endpoint, with central differences.

Settings: ../settings/optimizer.json ("lm"). The LM lowers the training cost J = f @ f;
the validation cost V is checked at iteration 0 and every 5 iterations and decides
the stop and the endpoint (the checked iterate with the lowest V).
Usage: python lm.py <setting> <seed 1-4> [max iterations]   (starts from de_result.json of the run)
       python lm.py <setting> select                         (selected endpoint over the four runs)
"""
import csv
import json
import sys

import numpy as np

import config
import de
import model
import objective

OPT = model.read_json(config.CASE_DIR / "settings" / "optimizer.json")["lm"]


def residuals_at(data, zs):
    """Residual vectors f at several points; None where a point is infeasible or its simulation stops with an error."""
    out = []
    for z in zs:
        f = None
        if np.all(np.abs(z) <= 1.0) and model.feasible(model.x_from_z(z)):
            try:
                f = data.residuals(model.x_from_z(z))
            except Exception:
                f = None
            if f is not None and not np.all(np.isfinite(f)):
                f = None
        out.append(f)
    return out


def lm_step(data, z, f, cost, lam):
    """One LM iteration. Returns (z, f, cost, lambda, accepted, info).

    Jacobian column j = (f(z + h e_j) - f(z - h e_j)) / 2h; one-sided if one point is
    missing, zero if both are. Three damping values lambda/10, lambda, 10 lambda with
    Marquardt scaling; the best trial is taken if it lowers J, else lambda x 100 and again.
    """
    n, h = z.size, OPT["jacobian_step_z"]
    E = np.eye(n) * h
    res = residuals_at(data, [z + E[j] for j in range(n)] + [z - E[j] for j in range(n)])
    fp, fm = res[:n], res[n:]
    Jm = np.zeros((f.size, n))
    one_sided = missing = 0
    for j in range(n):
        if fp[j] is not None and fm[j] is not None:
            Jm[:, j] = (fp[j] - fm[j]) / (2.0 * h)
        elif fp[j] is not None:
            Jm[:, j], one_sided = (fp[j] - f) / h, one_sided + 1
        elif fm[j] is not None:
            Jm[:, j], one_sided = (f - fm[j]) / h, one_sided + 1
        else:
            missing += 1
    A, g = Jm.T @ Jm, Jm.T @ f
    dA = np.diag(np.maximum(np.diag(A), 1e-12 * np.max(np.diag(A))))
    info = {"one_sided_columns": one_sided, "missing_columns": missing}
    for _ in range(OPT["max_tries_per_iteration"]):
        lams = [lam / 10.0, lam, lam * 10.0]
        trials = [np.clip(z + np.linalg.solve(A + l * dA, -g), -1.0, 1.0) for l in lams]
        fs = residuals_at(data, trials)
        costs = [float(ft @ ft) if ft is not None else OPT["infeasible_cost"] for ft in fs]
        k = int(np.argmin(costs))
        if costs[k] < cost:
            info.update(step_length_z=float(np.linalg.norm(trials[k] - z)), relative_decrease=(cost - costs[k]) / cost)
            return trials[k], fs[k], costs[k], lams[k], True, info
        lam *= OPT["damping_factor_without_descent"]
    return z, f, cost, lam, False, info


def check_event(checks, max_iterations):
    """checks: [(iteration, V)]. 'validation_rise', 'validation' (no improvement), 'budget' or None."""
    ref = k_ref = None
    best = np.inf
    for k, v in checks:
        if v > (1.0 + OPT["rise_fraction"]) * best:
            return "validation_rise"
        best = min(best, v)
        if ref is None or v < (1.0 - OPT["improvement_fraction"]) * ref:
            ref, k_ref = v, k
        if k - k_ref >= OPT["iterations_without_improvement"]:
            return "validation"
        if k >= max_iterations:
            return "budget"
    return None


def step_event(info):
    """'cost' if J fell by less than a relative 1e-4, 'step' if z moved by less than 1e-3, else None."""
    if info["relative_decrease"] < OPT["relative_cost_decrease_tolerance"]:
        return "cost"
    if info["step_length_z"] < OPT["step_tolerance_z"]:
        return "step"
    return None


def run(setting, z0, max_iterations=None, data=None, folder=None):
    """Refine z0. Returns the run summary; writes lm_history.csv and lm_result.json to folder."""
    max_iterations = max_iterations or OPT["max_iterations"]
    data = data or objective.Data(setting)
    z, lam, it = np.asarray(z0, dtype=float), OPT["damping_start"], 0
    f = residuals_at(data, [z])[0]
    cost = float(f @ f)
    v = data.cost(model.x_from_z(z), "validation")
    log = [[0, cost, v, lam, "", ""]]
    checks = [(0, v, z.tolist(), cost)]
    event = check_event([(k, vv) for k, vv, _, _ in checks], max_iterations)
    while event is None:
        z, f, cost, lam, accepted, info = lm_step(data, z, f, cost, lam)
        if not accepted:
            if checks[-1][0] != it:
                v = data.cost(model.x_from_z(z), "validation")
                checks.append((it, v, z.tolist(), cost))
                log.append([it, cost, v, lam, "", ""])
            event = "no_descent_step"
            break
        it += 1
        row = [it, cost, "", lam, info["step_length_z"], info["relative_decrease"]]
        print(f"iteration {it}: J {cost:.6g}", flush=True)
        if it % OPT["validation_check_every_iterations"] == 0 or step_event(info) or it >= max_iterations:
            v = data.cost(model.x_from_z(z), "validation")
            checks.append((it, v, z.tolist(), cost))
            row[2] = v
            event = check_event([(k, vv) for k, vv, _, _ in checks], max_iterations)
            print(f"check at iteration {it}: V {v:.6g}", flush=True)
        log.append(row)
        event = event or step_event(info)
    k_end, v_end, z_end, j_end = min(checks, key=lambda c: (c[1], c[0]))
    out = {"setting": setting, "stop_iteration": it, "stop_event": event, "endpoint_iteration": k_end,
           "training_cost": j_end, "validation_cost": v_end, "z": z_end,
           "checks": [{"iteration": k, "validation_cost": vv, "training_cost": jj} for k, vv, _, jj in checks]}
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "lm_history.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["iteration", "training_cost", "validation_cost", "damping", "step_length_z", "relative_decrease"])
            w.writerows(log)
        (folder / "lm_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def select_endpoint(setting, out_dir=config.OUTPUT_DIR):
    """Selected endpoint of a setting: lowest V over the LM checks of the four runs.

    The check with the lowest V of a run is its endpoint, so the endpoints are compared.
    """
    histories = {}
    for i in range(1, 5):
        res = json.loads((out_dir / "runs" / setting / f"seed{i}" / "lm_result.json").read_text(encoding="utf-8"))
        histories[i] = [(res["endpoint_iteration"], res["validation_cost"], res["z"])]
    seed, it, v, z = de.select(histories)
    return {"setting": setting, "selected_seed": seed, "iteration": it, "validation_cost": v, "z": z}


if __name__ == "__main__":
    if sys.argv[2] == "select":
        print(json.dumps(select_endpoint(sys.argv[1]), indent=1))
    else:
        run_dir = config.OUTPUT_DIR / "runs" / sys.argv[1] / f"seed{sys.argv[2]}"
        start = json.loads((run_dir / "de_result.json").read_text(encoding="utf-8"))["endpoint_z"]
        run(sys.argv[1], start, int(sys.argv[3]) if len(sys.argv) > 3 else None, folder=run_dir)
