"""Training, validation and test records of the teacher, with and without measurement noise.

The teacher has the parameters of ../teacher/teacher0.json. In the different-geometry
setting it has the true geometry, in the same-geometry setting the calibrated geometry.
Noise: white Gaussian, one seed per motion (../settings/seeds.json, noise.json).
"""
import sys

import numpy as np

import config
import model
import motions

SETTINGS = ("different_geometry", "same_geometry")
TEACHER_GEOMETRY = {"different_geometry": "true", "same_geometry": "calibrated"}
NOISE = model.read_json(config.CASE_DIR / "settings" / "noise.json")
SEEDS = model.read_json(config.CASE_DIR / "settings" / "seeds.json")["measurement_noise_seeds"]


def teacher():
    """Teacher parameter vector in the order of settings/bounds.json."""
    params = {p["name"]: p["value"] for p in model.read_json(config.CASE_DIR / "teacher" / "teacher0.json")["parameters"]}
    return np.array([params[n] for n in model.NAMES], dtype=float)


def add_noise(clean, seed):
    """Noisy record = noise-free record + normal draws, in the order of the output groups."""
    rng = np.random.default_rng(int(seed))
    noisy = {k: np.array(v, copy=True) for k, v in clean.items()}
    for group in NOISE["groups"]:
        a = group["array"]
        noisy[a] = clean[a] + rng.normal(0.0, group["noise_sd"], clean[a].shape)
    return noisy


def folder(out_dir, setting, split, noisy=True):
    return out_dir / "records" / setting / ("noisy" if noisy else "noise_free") / split


def generate(out_dir, setting, split, ids=None):
    """Simulate the teacher on the motions of one split and save both record files of each motion.

    Files: <out_dir>/records/<setting>/{noisy,noise_free}/<split>/<id>.npz. Existing files are kept.
    """
    x, geometry = teacher(), model.load_geometry(TEACHER_GEOMETRY[setting])
    for m in motions.motions(split):
        if ids is not None and m["id"] not in ids:
            continue
        paths = [folder(out_dir, setting, split, n) / f"{m['id']}.npz" for n in (True, False)]
        if all(p.exists() for p in paths):
            continue
        clean = model.simulate(x, geometry, motions.reference(m))
        for p, arrays in zip(paths, (add_noise(clean, SEEDS[split][m["id"]]), clean)):
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **arrays)
        print(setting, split, m["id"], flush=True)


if __name__ == "__main__":
    # python records.py <setting> [split ...]
    for split in sys.argv[2:] or ("training", "validation", "test"):
        generate(config.OUTPUT_DIR, sys.argv[1], split)
