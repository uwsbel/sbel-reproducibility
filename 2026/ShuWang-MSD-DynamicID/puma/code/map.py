"""Prior observability map: Schur SNR of the 17 parameters for a set of motions at one prior point.

Column j is the central difference (y(x + h e_j) - y(x - h e_j)) / 2h of the outputs after
0.2 s, divided by the noise standard deviations; h is in half-ranges. With G = sum of
D^T D over the motions, SNR_j = delta sqrt(G_jj) and Schur SNR_j = delta sqrt(G_jj -
G_j,rest pinv(G_rest,rest) G_rest,j). Settings: ../results/map/schur_snr.json ("settings").
Usage: python map.py <prior point> <motion> [motion ...] [--step-check]
"""
import json
import sys

import numpy as np

import config
import cost
import model
import motions

SET = model.read_json(config.CASE_DIR / "results" / "map" / "schur_snr.json")["settings"]
POINTS = {p["name"]: np.array(p["values"], dtype=float)
          for p in model.read_json(config.CASE_DIR / "settings" / "prior_points.json")["points"]}
LIBRARY = {m["id"]: m for m in motions.SETTINGS["library"]}


def response(out):
    """Outputs after the scoring start, divided by the noise standard deviations: samples x 27."""
    mask = out["time"] >= cost.SCORING_START_S - 1e-12
    return np.concatenate([out[cost.ARRAY[g]][mask].reshape(int(mask.sum()), -1) / cost.NOISE_SD[g]
                           for g in cost.GROUPS], axis=1)


def columns(x, motion, h):
    """Difference columns of one motion at x, flattened; None for a parameter whose step leaves the feasible set."""
    geometry, ref = model.load_geometry("calibrated"), motions.reference(motion)
    cols = []
    for j in range(len(model.NAMES)):
        ys = []
        for sign in (1.0, -1.0):
            xs = x.copy()
            xs[j] += sign * h * model.HALF[j]
            ys.append(response(model.simulate(xs, geometry, ref)) if model.feasible(xs) else None)
        if ys[1] is None or ys[0] is None:
            cols.append(None)
            continue
        diff = ys[0] - ys[1]
        sq = cost.NOISE_SD["joint_angle"]  # joint-angle differences are wrapped
        diff[:, :3] = np.arctan2(np.sin(diff[:, :3] * sq), np.cos(diff[:, :3] * sq)) / sq
        cols.append((diff / (2.0 * h)).astype(np.float32).astype(np.float64).reshape(-1))
        print(motion["id"], model.NAMES[j], flush=True)
    return cols


def snr(G):
    """Marginal and Schur SNR of each parameter for the Gram matrix G."""
    n, delta = len(G), SET["parameter_change_z"]
    marginal, schur = np.zeros(n), np.zeros(n)
    for j in range(n):
        rest = [k for k in range(n) if k != j]
        proj = G[j, rest] @ np.linalg.pinv(G[np.ix_(rest, rest)], rcond=1e-12) @ G[rest, j]
        marginal[j] = delta * np.sqrt(max(G[j, j], 0.0))
        schur[j] = delta * np.sqrt(max(G[j, j] - proj, 0.0))
    return marginal, schur


def run(point, motion_ids, step_check=False, out_dir=config.OUTPUT_DIR):
    """Map of one prior point for a motion set; writes map_<point>.json and returns it.

    With step_check, the step agreement kappa_j of each motion is computed with the
    second step size; a column counts only if kappa_j is at least the threshold.
    """
    x, n = POINTS[point], len(model.NAMES)
    G = np.zeros((n, n))
    reliable = np.zeros(n, dtype=bool)
    kappa = {}
    for mid in motion_ids:
        cols = columns(x, LIBRARY[mid], SET["difference_step_z"])
        D = np.stack([c if c is not None else np.zeros_like(next(c for c in cols if c is not None)) for c in cols], axis=1)
        G += D.T @ D
        ok = np.array([c is not None and np.all(np.isfinite(c)) and np.linalg.norm(c) > 0.0 for c in cols])
        if step_check:
            cols2 = columns(x, LIBRARY[mid], SET["second_step_z_first_development_point"])
            k = [float(np.linalg.norm(0.5 * (a + b)) / np.sqrt(0.5 * (a @ a + b @ b)))
                 if a is not None and b is not None and (a @ a + b @ b) > 0 else 0.0 for a, b in zip(cols, cols2)]
            kappa[mid] = k
            ok &= np.array(k) >= SET["smallest_step_agreement"]
        reliable |= ok
    marginal, schur = snr(G)
    ev = np.linalg.eigvalsh(G)
    out = {"point": point, "motions": motion_ids, "parameters": model.NAMES,
           "marginal_snr": marginal.tolist(), "schur_snr": schur.tolist(),
           "covered": [bool(m >= SET["snr_threshold"] and s >= SET["snr_threshold"] and r)
                       for m, s, r in zip(marginal, schur, reliable)],
           "numerical_rank": int(np.sum(np.sqrt(np.clip(ev, 0, None)) >= 1e-3 * np.sqrt(ev.max()))),
           "condition_number": float(np.sqrt(ev.max() / max(ev.min(), 1e-300))),
           "step_agreement": kappa or None}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"map_{point}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--step-check"]
    run(args[0], args[1:], "--step-check" in sys.argv)
