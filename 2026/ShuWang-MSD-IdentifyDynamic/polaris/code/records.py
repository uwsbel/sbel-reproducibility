"""Training, validation and test records of a teacher.

The teacher is the model with the teacher parameters (teacher/) and the teacher geometry of the setting:
the true geometry in different_geometry, the calibrated geometry in same_geometry. Noise: settings/noise.json,
noise seeds: settings/seeds.json. Files: config.WORK_DIR/records/<teacher>/<setting>/<set>/<motion>.npz.
Usage: python records.py <teacher> <setting>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import config
import model

SETS = ("training", "validation", "test")
TEACHER_GEOMETRY = {"different_geometry": "true", "same_geometry": "calibrated"}
SETTINGS = model.CASE_DIR / "settings"


def motions(set_name):
    """Motion names of a set, in the order of settings/motions.json (this order fixes the noise seeds)."""
    m = json.loads((SETTINGS / "motions.json").read_text(encoding="utf-8"))
    return list(m["training"]) if set_name == "training" else [row["name"] for row in m[set_name]]


def noise_seed(teacher, set_name, index):
    """Seed of motion `index` of a set: first seed of the set + index, + 10000 k for teacher k."""
    s = json.loads((SETTINGS / "seeds.json").read_text(encoding="utf-8"))["noise_seeds"]
    k = int(teacher.replace("teacher", ""))
    return s[f"{set_name}_first"] + index + 10000 * k


def add_noise(clean, seed):
    """Noisy record: Gaussian noise on each group, drawn in the group order of settings/noise.json.

    The attitude noise is a rotation with Gaussian rotation-vector components, applied on the left.
    """
    groups = json.loads((SETTINGS / "noise.json").read_text(encoding="utf-8"))["groups"]
    rng = np.random.default_rng(seed)
    out = {g["name"]: np.array(clean[g["name"]], copy=True) for g in groups}
    for g in groups:
        key, sd = g["name"], g["noise_sd"]
        if key == "q":
            rot = Rotation.from_quat(np.roll(clean[key], -1, axis=1))
            noise = Rotation.from_rotvec(rng.normal(0, sd, (len(clean[key]), 3)))
            out[key] = np.roll((noise * rot).as_quat(), 1, axis=1)
        else:
            out[key] += rng.normal(0, sd, out[key].shape)
    return out


def path(teacher, setting, set_name, name, noise_free=False):
    folder = set_name + ("_noise_free" if noise_free else "")
    return Path(config.WORK_DIR) / "records" / teacher / setting / folder / f"{name}.npz"


def make(teacher="teacher0", setting="different_geometry", sets=SETS):
    """Simulate the teacher on every motion of the sets and write noisy and noise-free records (skips existing files)."""
    values = json.loads((model.CASE_DIR / "teacher" / f"{teacher}.json").read_text(encoding="utf-8"))["values"]
    geometry = model.load_geometry(TEACHER_GEOMETRY[setting])
    starts = {}
    for set_name in sets:
        for i, name in enumerate(motions(set_name)):
            target = path(teacher, setting, set_name, name)
            if target.exists():
                continue
            inputs, road = model.motion_inputs(name)
            if road not in starts:
                starts[road] = model.settle(values, geometry, road)
            clean, valid = model.simulate(values, geometry, inputs, road, starts[road])
            assert valid, (teacher, setting, name)
            for p, data in ((path(teacher, setting, set_name, name, True), clean),
                            (target, add_noise(clean, noise_seed(teacher, set_name, i)))):
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, input=inputs, **data)
            print(f"{teacher} {setting} {set_name} {name}", flush=True)


def load(teacher, setting, set_name, name, noise_free=False):
    """One record as a dict of arrays (input and the eight groups)."""
    with np.load(path(teacher, setting, set_name, name, noise_free), allow_pickle=False) as f:
        return {k: np.array(f[k]) for k in f.files}


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
