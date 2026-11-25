"""Records of a teacher: noise-free and noisy outputs of the training, validation and test motions.

The teacher is a parameter vector of teacher/ with the true geometry (setting different_geometry) or
the calibrated geometry (setting same_geometry). Noise follows settings/noise.json and the seeds of
settings/seeds.json. Records are written to <OUTPUT_DIR>/records/<setting>/teacher<k>/<set>/.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import config
import model

SETTINGS = model.CASE_DIR / "settings"
MOTIONS = json.loads((SETTINGS / "motions.json").read_text(encoding="utf-8"))
NOISE = {g["key"]: g["standard_deviation_si"] for g in json.loads((SETTINGS / "noise.json").read_text(encoding="utf-8"))["groups"]}
SEEDS = json.loads((SETTINGS / "seeds.json").read_text(encoding="utf-8"))
SETS = ("training", "validation", "test")
TEACHER_GEOMETRY = {"different_geometry": "true", "same_geometry": "calibrated"}


def motion_entries(set_name):
    """The entries of motions.json of one set, in the order of the noise seeds."""
    if set_name == "training":
        library = {m["id"]: m for m in MOTIONS["library"]}
        return [library[i] for i in MOTIONS["training"]]
    return MOTIONS[set_name]


def motions(set_name):
    return [model.load_motion(e) for e in motion_entries(set_name)]


def teacher_parameters(teacher=0):
    d = json.loads((model.CASE_DIR / "teacher" / f"teacher{teacher}.json").read_text(encoding="utf-8"))
    return np.array([p["value"] for p in d["parameters"]])


def noise_seed(set_name, index, teacher=0):
    return SEEDS["noise_seeds"][f"{set_name}_first"] + index + 10000 * teacher


def add_noise(clean, seed, rng=None):
    """Noisy copy of the outputs: Gaussian noise on every component; the attitude noise is a small rotation on the left.
    The generator is numpy.random.default_rng(seed), or rng if given."""
    rng = np.random.default_rng(seed) if rng is None else rng
    out = {g: np.array(v, copy=True) for g, v in clean.items()}
    for g in model.GROUPS:
        if g == "attitude":
            rot = Rotation.from_quat(np.roll(clean[g], -1, axis=1))
            noise = Rotation.from_rotvec(rng.normal(0, NOISE[g], (len(clean[g]), 3)))
            out[g] = np.roll((noise * rot).as_quat(), 1, axis=1)
        else:
            out[g] += rng.normal(0, NOISE[g], out[g].shape)
    return out


def folder(setting, teacher, set_name):
    return Path(config.OUTPUT_DIR) / "records" / setting / f"teacher{teacher}" / set_name


def generate(setting, teacher=0, sets=SETS):
    """Simulate the teacher on the motions of the given sets and write noise-free and noisy records."""
    params = teacher_parameters(teacher)
    geometry = model.load_geometry(TEACHER_GEOMETRY[setting])
    for set_name in sets:
        out = folder(setting, teacher, set_name)
        out.mkdir(parents=True, exist_ok=True)
        for i, motion in enumerate(motions(set_name)):
            if (out / f"{motion['id']}_noisy.npz").exists():
                continue
            r = model.simulate(params, geometry, motion)
            if not r["valid"]:
                raise RuntimeError(f"teacher rollout not valid: {setting} {set_name} {motion['id']} ({r['reason']})")
            np.savez_compressed(out / f"{motion['id']}_noise_free.npz", **r["outputs"])
            np.savez_compressed(out / f"{motion['id']}_noisy.npz", **add_noise(r["outputs"], noise_seed(set_name, i, teacher)))
            print(f"record {setting} teacher{teacher} {set_name} {motion['id']}", flush=True)


def load(setting, set_name, teacher=0, kind="noisy"):
    """Records of one set as a list of dicts of arrays, in the order of motions.json."""
    out = folder(setting, teacher, set_name)
    records = []
    for e in motion_entries(set_name):
        with np.load(out / f"{e['id']}_{kind}.npz", allow_pickle=False) as f:
            records.append({g: np.array(f[g]) for g in model.GROUPS})
    return records


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", choices=sorted(TEACHER_GEOMETRY), required=True)
    ap.add_argument("--teacher", type=int, default=0)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    a = ap.parse_args()
    generate(a.setting, a.teacher, a.sets)
