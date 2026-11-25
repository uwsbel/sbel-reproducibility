"""Training cost J, validation cost V and test cost T of a parameter vector.

The model uses the calibrated geometry. The cost of one motion is the mean of the eight group costs of
../cost.py over the scored samples. J is the mean over the 5 training motions, V over the 18 validation
motions, T over the 18 test motions. An infeasible vector or a run that fails the mechanical checks gets
the penalty cost of settings/optimizer.json.
"""
import json
import statistics
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import model
import records

sys.path.insert(0, str(model.CASE_DIR))
import cost  # noqa: E402  (../cost.py: groups, noise levels, scored samples, loss)

PENALTY = json.loads((model.CASE_DIR / "settings" / "optimizer.json").read_text(encoding="utf-8"))["penalty_cost"]


def motion_cost(prediction, record):
    """Cost of one motion and its eight group costs.

    Same as cost.cost() on the scored samples; the attitude error is the scipy rotation vector,
    which can differ from cost.rotation_vector() in the last bit.
    """
    groups = {}
    for g in cost.GROUPS:
        a, b = cost.scored_samples(prediction[g]), cost.scored_samples(record[g])
        if g == "q":
            e = (Rotation.from_quat(np.roll(b, -1, axis=1)).inv() * Rotation.from_quat(np.roll(a, -1, axis=1))).as_rotvec()
        else:
            e = a - b
        groups[g] = float(np.mean(cost.loss(e / cost.NOISE_SD[g])))
    return float(np.mean(list(groups.values()))), groups


class Model:
    """Model with the calibrated geometry; settles once per parameter vector and ground."""

    def __init__(self):
        self.geometry = model.load_geometry("calibrated")
        self.starts = {}

    def predict(self, params, name):
        inputs, road = model.motion_inputs(name)
        key = (np.asarray(params, dtype=np.float64).tobytes(), road)
        if key not in self.starts:
            self.starts = {k: v for k, v in self.starts.items() if k[0] == key[0]}
            self.starts[key] = model.settle(params, self.geometry, road)
        return model.simulate(params, self.geometry, inputs, road, self.starts[key])


def motion_costs(params, set_name, teacher="teacher0", setting="different_geometry", stop_on_invalid=False, sim=None):
    """Per-motion rows (name, cost, group costs, valid) of a set against the teacher's noisy records."""
    sim = sim or Model()
    rows = []
    for name in records.motions(set_name):
        prediction, valid = sim.predict(params, name)
        if valid:
            c, groups = motion_cost(prediction, records.load(teacher, setting, set_name, name))
        else:
            c, groups = PENALTY, None
        rows.append(dict(motion=name, cost=c, group_costs=groups, valid=valid, prediction=prediction))
        if stop_on_invalid and not valid:
            break
    return rows


def training_cost(params, teacher="teacher0", setting="different_geometry", sim=None):
    """J: mean cost over the training motions; the penalty if the vector is infeasible or a run is not valid."""
    if not model.feasible(params):
        return PENALTY
    rows = motion_costs(params, "training", teacher, setting, stop_on_invalid=True, sim=sim)
    return float(np.mean([r["cost"] for r in rows])) if all(r["valid"] for r in rows) else PENALTY


def validation_cost(params, teacher="teacher0", setting="different_geometry", sim=None):
    """V: mean cost over the validation motions (a motion that is not valid counts with the penalty)."""
    return float(np.mean([r["cost"] for r in motion_costs(params, "validation", teacher, setting, sim=sim)]))


def test_cost(params, teacher="teacher0", setting="different_geometry", sim=None):
    """T: mean cost over the test motions (statistics.mean, as in the stored results)."""
    return statistics.mean(r["cost"] for r in motion_costs(params, "test", teacher, setting, sim=sim))
