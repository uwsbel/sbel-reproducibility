"""Prior observability map: Schur SNR of each parameter for a set of motions at the prior points.

A block is one prior point and one motion. The model output (calibrated geometry) is reduced to a feature
vector of 630 values. Its Jacobian in z is the mean of central differences at three radii. The feature noise
covariance comes from 2000 noisy copies of one output (Ledoit-Wolf shrinkage); the Jacobian is whitened with it.
For a motion set the whitened Jacobians are stacked. Marginal SNR = 0.1 |column|; Schur SNR = 0.1 |part of the
column that the other six columns cannot explain|. Usage: python map.py <prior point> <motion> [<motion> ...]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import config
import model
import records

RADII = (0.01, 0.02, 0.04)      # half-range units of z
REALIZATIONS = 2000
KAPPA_MIN = 0.70                # multi-radius consistency needed for a reliable column
DT = 0.002
NOISE = {g["name"]: g["noise_sd"] for g in json.loads((model.CASE_DIR / "settings" / "noise.json").read_text(encoding="utf-8"))["groups"]}
# Noise seed of each block, keyed by "prior point/motion".
NOISE_SEEDS = json.loads((model.CASE_DIR / "settings" / "map_noise_seeds.json").read_text(encoding="utf-8"))["seeds"]


# ----------------------------------------------------------------------------------------------- feature
def _channels(out):
    """Output groups relative to the first sample, in noise standard deviations."""
    rotation = Rotation.from_quat(np.roll(np.asarray(out["q"], dtype=float), -1, axis=1))
    speed = np.asarray(out["speed"], dtype=float)[:, None]
    delta = np.asarray(out["delta"], dtype=float)[:, None]
    p, v, wheel, gyro, upright = (np.asarray(out[k], dtype=float) for k in ("p", "v", "wheel_omega", "gyro", "upright_rel"))
    return [(p - p[0]) / NOISE["p"], (rotation[0].inv() * rotation).as_rotvec() / NOISE["q"], (v - v[0]) / NOISE["v"],
            (speed - speed[0]) / NOISE["speed"], (wheel - wheel[0]) / NOISE["wheel_omega"], (delta - delta[0]) / NOISE["delta"],
            (gyro - gyro[0]) / NOISE["gyro"], (upright - upright[0]) / NOISE["upright_rel"]]


def _smooth(channel, window=51):
    flat = np.asarray(channel, dtype=float).reshape(len(channel), -1)
    size = min(int(window), len(flat) - (1 - len(flat) % 2))
    if size < 5:
        return flat
    if size % 2 == 0:
        size -= 1
    return savgol_filter(flat, size, 3, axis=0, mode="interp")


def _band_energy(flat):
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequency = np.fft.rfftfreq(len(flat), d=DT)
    values = []
    for low, high in ((0.1, 0.7), (0.7, 1.6), (1.6, 4.0)):
        mask = (frequency >= low) & (frequency < high)
        values.append(np.sqrt(np.sum(np.abs(spectrum[mask]) ** 2, axis=0)) / len(flat) if np.any(mask) else np.zeros(flat.shape[1]))
    return np.concatenate(values)


def feature(out, inputs):
    """630 values: per smoothed channel, mean, RMS and integral over five time blocks, three band energies,
    and the regression on each centered input."""
    edges = np.asarray([0, 100, 250, 500, 750, 1000], dtype=int)
    controls = np.asarray(inputs, dtype=float)
    centered = controls - np.mean(controls, axis=0, keepdims=True)
    values = []
    for channel in _channels(out):
        flat = _smooth(channel)
        for start, stop in zip(edges[:-1], edges[1:]):
            block = flat[start:stop]
            values.extend([np.mean(block, axis=0), np.sqrt(np.mean(block * block, axis=0)), np.trapz(block, dx=DT, axis=0)])
        values.append(_band_energy(flat))
        for i in range(3):
            u = centered[:, i]
            values.append((u[:, None] * flat).sum(axis=0) / max(float(np.dot(u, u)), 1e-12))
    return np.concatenate([np.asarray(v).reshape(-1) for v in values])


# ----------------------------------------------------------------------------------------------- one block
def ledoit_wolf(samples):
    """Ledoit-Wolf shrunk covariance of the rows of samples (same arithmetic as scikit-learn LedoitWolf)."""
    X = samples - samples.mean(0)
    n, p = X.shape
    X2 = X ** 2
    trace = np.sum(X2, axis=0) / n
    mu = np.sum(trace) / p
    beta_ = 0.0 + np.sum(np.dot(X2.T, X2))
    delta_ = (0.0 + np.sum(np.dot(X.T, X) ** 2)) / n ** 2
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * trace.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    shrinkage = 0 if beta == 0 else beta / delta
    emp = np.dot(X.T, X) / n
    cov = (1.0 - shrinkage) * emp
    cov.flat[:: p + 1] += shrinkage * (np.sum(np.trace(emp)) / p)
    return cov


def metrics(J):
    """Marginal and Schur SNR (0.1 x column norms) and the largest absolute correlation of each column."""
    norms = np.linalg.norm(J, axis=0)
    schur = np.zeros(J.shape[1])
    for j in range(J.shape[1]):
        other = [k for k in range(J.shape[1]) if k != j]
        coef = np.linalg.lstsq(J[:, other], J[:, j], rcond=None)[0]
        schur[j] = np.linalg.norm(J[:, j] - J[:, other] @ coef)
    denom = np.outer(norms, norms)
    corr = np.divide(J.T @ J, denom, out=np.zeros(denom.shape), where=denom > 0)
    np.fill_diagonal(corr, 0)
    return dict(marginal=.1 * norms, schur=.1 * schur, largest_correlation=np.max(abs(corr), axis=1))


def prior_point(name):
    rows = json.loads((model.CASE_DIR / "settings" / "prior_points.json").read_text(encoding="utf-8"))["prior_points"]
    row = next(r for r in rows if r["name"] == name)
    return np.array(row["values"]), np.array(row["z"])


def block(point, motion):
    """Whitened Jacobian (630 x 7) and multi-radius consistency kappa of one prior point and motion."""
    p0, _ = prior_point(point)
    _, half = model.bounds()
    geometry = model.load_geometry("calibrated")
    inputs, road = model.motion_inputs(motion)
    features, base = {}, None
    for j, h, sign in itertools.product(range(7), RADII, (1, -1)):
        p = p0.copy()
        p[j] += half[j] * (sign * h)
        assert model.feasible(p)
        out, valid = model.simulate(p, geometry, inputs, road)
        assert valid, (point, motion, j, h, sign)
        features[j, h, sign] = feature(out, inputs)
        if (j, h, sign) == (0, RADII[0], 1):
            base = out
        print(f"{point} {motion} {model.NAMES[j]} {sign * h:+.2f}", flush=True)
    n = len(features[0, RADII[0], 1])
    secants, J, kappa = np.zeros((len(RADII), n, 7)), np.zeros((n, 7)), np.zeros(7)
    for j in range(7):
        for k, h in enumerate(RADII):
            secants[k, :, j] = (features[j, h, 1] - features[j, h, -1]) / (2 * h)
        J[:, j] = secants[:, :, j].mean(axis=0)
        rms = np.sqrt(np.mean(np.linalg.norm(secants[:, :, j], axis=1) ** 2))
        kappa[j] = np.linalg.norm(J[:, j]) / max(float(rms), 1e-15)
    # Noise covariance of the feature around the output with mu at +0.01.
    rng = np.random.default_rng(NOISE_SEEDS[f"{point}/{motion}"])
    f0 = feature(base, inputs)
    samples = np.stack([feature(records.add_noise(base, rng), inputs) - f0 for _ in range(REALIZATIONS)])
    eig, V = np.linalg.eigh(ledoit_wolf(samples))
    W = (V / np.sqrt(np.maximum(eig, 1e-12 * eig[-1]))) @ V.T
    return dict(Jw=W @ J, kappa=kappa, reliable=kappa >= KAPPA_MIN)


# ----------------------------------------------------------------------------------------------- motion sets
def summary(blocks, points, motions):
    """Per prior point and parameter: marginal and Schur SNR of the stacked set, and covered (reliable, both SNR >= 1)."""
    rows = []
    for point in points:
        m = metrics(np.vstack([blocks[point, x]["Jw"] for x in motions]))
        reliable = np.logical_or.reduce([blocks[point, x]["reliable"] for x in motions])
        for j, name in enumerate(model.NAMES):
            rows.append(dict(prior_point=point, parameter=name, marginal_snr=float(m["marginal"][j]), schur_snr=float(m["schur"][j]),
                             largest_correlation=float(m["largest_correlation"][j]), reliable=bool(reliable[j]),
                             covered=bool(reliable[j] and m["marginal"][j] >= 1 and m["schur"][j] >= 1)))
    return rows


def score(rows, motions):
    """Rank key of a motion set: more covered pairs, more parameters covered at most points, larger median Schur SNR."""
    points = sorted({r["prior_point"] for r in rows})
    counts = [sum(r["covered"] for r in rows if r["parameter"] == n) for n in model.NAMES]
    return (-sum(counts), -sum(c > len(points) / 2 for c in counts), -float(np.median([r["schur_snr"] for r in rows])), tuple(sorted(motions)))


def main(point, motions):
    blocks = {(point, m): block(point, m) for m in motions}
    rows = summary(blocks, [point], motions)
    out = Path(config.WORK_DIR) / "map"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(prior_point=point, motions=motions, summary=rows,
                  blocks=[dict(motion=m, kappa=b["kappa"].tolist(), reliable=b["reliable"].tolist()) for (p, m), b in blocks.items()])
    (out / f"map_{point}_{'_'.join(motions)}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
