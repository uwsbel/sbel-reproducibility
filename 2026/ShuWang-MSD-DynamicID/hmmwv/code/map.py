"""Prior observability map: whitened sensitivities and Schur SNR of the seven parameters for a motion set.

For one prior point and one motion (a block), 44 rollouts of the model (calibrated geometry) give a
630-dimensional feature of the outputs at the point, twice, and at +/- 0.01, 0.02, 0.04 half-ranges of
each parameter. Central secants give the sensitivity J; 2000 noise realizations give the feature
covariance (Ledoit-Wolf), which whitens J. The Schur SNR of a parameter is 0.1 x the part of its
whitened column that the other six columns cannot explain. A motion set stacks its blocks per point.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import config
import model
import records

SETTINGS = model.CASE_DIR / "settings"
POINTS = {p["name"]: p for p in json.loads((SETTINGS / "prior_points.json").read_text(encoding="utf-8"))["points"]}
HALF = np.array([p["half_range"] for p in json.loads((SETTINGS / "bounds.json").read_text(encoding="utf-8"))["parameters"]])
MOTIONS = json.loads((SETTINGS / "motions.json").read_text(encoding="utf-8"))
NOISE_SEEDS = json.loads((SETTINGS / "map_noise_seeds.json").read_text(encoding="utf-8"))["seeds"]
RADII = (0.01, 0.02, 0.04)
DRAWS = 2000
RELIABLE_KAPPA = 0.70           # multiscale consistency needed for a reliable column
DT = 0.002


# ------------------------------------------------------------------ feature of one rollout
def _channels(o):
    """The eight output groups relative to their first sample, divided by the noise standard deviations."""
    n = records.NOISE
    rotation = Rotation.from_quat(np.roll(o["attitude"], -1, axis=1))
    out = [o["position"] - o["position"][0], (rotation[0].inv() * rotation).as_rotvec()]
    for g in model.GROUPS[2:]:
        x = o[g][:, None] if o[g].ndim == 1 else o[g]
        out.append(x - x[0])
    return [c / n[g] for c, g in zip(out, model.GROUPS)]


def _smooth(channel, window=51):
    flat = np.asarray(channel, dtype=float).reshape(len(channel), -1)
    size = min(int(window), len(flat) - (1 - len(flat) % 2))
    if size < 5:
        return flat
    if size % 2 == 0:
        size -= 1
    return savgol_filter(flat, size, 3, axis=0, mode="interp")


def _band_energy(flat):
    spectrum = np.fft.rfft(flat - np.mean(flat, axis=0, keepdims=True), axis=0)
    frequency = np.fft.rfftfreq(len(flat), d=DT)
    values = []
    for low, high in ((0.1, 0.7), (0.7, 1.6), (1.6, 4.0)):
        mask = (frequency >= low) & (frequency < high)
        values.append(np.sqrt(np.sum(np.abs(spectrum[mask]) ** 2, axis=0)) / len(flat) if np.any(mask) else np.zeros(flat.shape[1]))
    return np.concatenate(values)


def feature(outputs, control):
    """Per channel: mean, RMS and time integral in five windows, three band energies and three input projections."""
    edges = np.asarray([0, 100, 250, 500, 750, 1000], dtype=int)
    controls = np.asarray(control, dtype=float)
    centered = controls - np.mean(controls, axis=0, keepdims=True)
    parts = []
    for channel in _channels(outputs):
        flat = _smooth(channel)
        for start, stop in zip(edges[:-1], edges[1:]):
            block = flat[start:stop]
            parts.extend([np.mean(block, axis=0), np.sqrt(np.mean(block * block, axis=0)), np.trapz(block, dx=DT, axis=0)])
        parts.append(_band_energy(flat))
        for index in range(3):
            u = centered[:, index]
            parts.append((u[:, None] * flat).sum(axis=0) / max(float(np.dot(u, u)), 1e-12))
    return np.concatenate([np.asarray(p).reshape(-1) for p in parts])


# ------------------------------------------------------------------ statistics
def ledoit_wolf(samples):
    """Ledoit-Wolf shrinkage covariance of the rows of samples."""
    x = samples - samples.mean(0)
    n, p = x.shape
    x2 = x ** 2
    trace = np.sum(x2, axis=0) / n
    mu = np.sum(trace) / p
    delta_ = np.sum(np.dot(x.T, x) ** 2) / n ** 2
    beta_ = np.sum(np.dot(x2.T, x2))
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * trace.sum() + p * mu ** 2) / p
    shrinkage = 0 if min(beta, delta) == 0 else min(beta, delta) / delta
    emp = np.dot(x.T, x) / n
    cov = (1.0 - shrinkage) * emp
    cov.flat[::p + 1] += shrinkage * (np.sum(np.trace(emp)) / p)
    return cov


def snr(J):
    """Marginal SNR (0.1 x column norm) and Schur SNR (0.1 x norm of the column part not explained by the others)."""
    schur = np.zeros(7)
    for j in range(7):
        other = [k for k in range(7) if k != j]
        coef = np.linalg.lstsq(J[:, other], J[:, j], rcond=None)[0]
        schur[j] = np.linalg.norm(J[:, j] - J[:, other] @ coef)
    return 0.1 * np.linalg.norm(J, axis=0), 0.1 * schur


# ------------------------------------------------------------------ one block
def labels():
    return ["repeatA", "repeatB"] + [f"{p}_h{round(h * 1000):03d}{s}" for p in model.PARAMETERS for h in RADII for s in "pm"]


def parameters(point, label):
    """Parameters of one rollout of a block: the prior point, shifted in one parameter for the secant labels."""
    p = np.array(POINTS[point]["values"], dtype=float)
    if not label.startswith("repeat"):
        name, code = label.split("_h")
        p[model.PARAMETERS.index(name)] += HALF[model.PARAMETERS.index(name)] * (int(code[:-1]) / 1000 * (1 if code[-1] == "p" else -1))
    return p


def block(point, motion_id, geometry):
    """Whitened sensitivity of one prior point and one motion, with its per-parameter SNR."""
    entry = next(m for m in MOTIONS["library"] + MOTIONS["confirmation"] if m["id"] == motion_id)
    motion = model.load_motion(entry)
    features, base = {}, None
    for label in labels():
        r = model.simulate(parameters(point, label), geometry, motion)
        if not r["valid"]:
            raise RuntimeError(f"map rollout not valid: {point} {motion_id} {label} ({r['reason']})")
        features[label] = feature(r["outputs"], motion["control"])
        if label == "mu_h010p":
            base = r["outputs"]                                  # noise is added to this rollout
    repeat = float(np.linalg.norm(features["repeatA"] - features["repeatB"]) / np.sqrt(630))
    assert repeat <= 1e-9, "the two repeated rollouts differ"
    J, kappa = np.zeros((630, 7)), np.zeros(7)
    for j, p in enumerate(model.PARAMETERS):
        secants = np.array([(features[f"{p}_h{round(h * 1000):03d}p"] - features[f"{p}_h{round(h * 1000):03d}m"]) / (2 * h) for h in RADII])
        J[:, j] = secants.mean(axis=0)
        kappa[j] = np.linalg.norm(J[:, j]) / max(float(np.sqrt(np.mean(np.linalg.norm(secants, axis=1) ** 2))), 1e-15)
    f0 = features["mu_h010p"]
    rng = np.random.default_rng(NOISE_SEEDS[f"{point}/{motion_id}"])
    samples = np.stack([feature(records.add_noise(base, None, rng), motion["control"]) - f0 for _ in range(DRAWS)])
    eig, V = np.linalg.eigh(ledoit_wolf(samples))
    Jw = (V / np.sqrt(np.maximum(eig, 1e-12 * eig[-1]))) @ V.T @ J
    marginal, schur = snr(Jw)
    return {"Jw": Jw, "kappa": kappa, "reliable": kappa >= RELIABLE_KAPPA, "marginal_snr": marginal, "schur_snr": schur}


def stack(blocks, point, motion_ids):
    """SNR of a motion set at one prior point; a parameter is covered if reliable and both SNR >= 1."""
    marginal, schur = snr(np.vstack([blocks[(point, m)]["Jw"] for m in motion_ids]))
    reliable = np.logical_or.reduce([blocks[(point, m)]["reliable"] for m in motion_ids])
    return {"marginal_snr": marginal.tolist(), "schur_snr": schur.tolist(), "reliable": reliable.tolist(),
            "covered": (reliable & (marginal >= 1) & (schur >= 1)).tolist()}


def run(points, motion_ids):
    """Map of a motion set at the given prior points; writes <OUTPUT_DIR>/map/map_<points>.json."""
    geometry = model.load_geometry("calibrated")
    blocks = {(p, m): block(p, m, geometry) for p in points for m in motion_ids}
    out = {"parameters": list(model.PARAMETERS), "motions": list(motion_ids), "prior_points": list(points),
           "blocks": [{"prior_point": p, "motion": m, "kappa": b["kappa"].tolist(), "reliable": b["reliable"].tolist(),
                       "marginal_snr": b["marginal_snr"].tolist(), "schur_snr": b["schur_snr"].tolist()}
                      for (p, m), b in blocks.items()],
           "motion_set": {p: stack(blocks, p, motion_ids) for p in points}}
    path = Path(config.OUTPUT_DIR) / "map" / f"map_{'_'.join(points)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", nargs="+", default=[n for n, p in POINTS.items() if p["role"] == "development"])
    ap.add_argument("--motions", nargs="+", default=MOTIONS["training"])
    a = ap.parse_args()
    print(json.dumps(run(a.points, a.motions)["motion_set"], indent=1))
