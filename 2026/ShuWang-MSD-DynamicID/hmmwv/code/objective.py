"""Training cost J, validation cost V and test cost T of a parameter vector.

The cost of one motion compares a prediction with a noisy record on the scored samples (skip 100,
then every fourth): soft-L1 loss of each component divided by its noise standard deviation, mean
within each of the eight output groups, then mean over the groups. J, V and T average the motion
costs of the training, validation and test motions. An infeasible parameter vector or an invalid
rollout gets the penalty cost.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import model

sys.path.insert(0, str(model.CASE_DIR))
from cost import GROUPS, NOISE_STD, SAMPLE_STRIDE, SKIPPED_SAMPLES, loss  # noqa: E402

OPTIMIZER = json.loads((model.CASE_DIR / "settings" / "optimizer.json").read_text(encoding="utf-8"))
PENALTY = OPTIMIZER["penalty_of_infeasible_or_invalid_candidates"]


def motion_cost(prediction, record):
    """Cost of one motion and the cost of each output group."""
    select = slice(SKIPPED_SAMPLES, len(record["position"]), SAMPLE_STRIDE)
    groups = {}
    for g in GROUPS:
        if g == "attitude":
            a = Rotation.from_quat(np.roll(prediction[g][select], -1, axis=1))
            b = Rotation.from_quat(np.roll(record[g][select], -1, axis=1))
            e = (b.inv() * a).as_rotvec() / NOISE_STD[g]
        else:
            e = (prediction[g][select] - record[g][select]) / NOISE_STD[g]
        groups[g] = float(np.mean(loss(e)))
    return float(np.mean(list(groups.values()))), groups


def motion_costs(parameters, geometry, motions, records, reject_above=None):
    """Simulate the motions in order and score each one against its record.

    With reject_above, the loop stops as soon as the sum of the motion costs exceeds
    reject_above x (number of motions): the trial cannot beat the member it is compared with.
    Returns (costs, status) with status 'valid', 'invalid' or 'stopped early'.
    """
    if not model.feasible(parameters):
        return [], "invalid"
    costs = []
    for motion, record in zip(motions, records):
        r = model.simulate(parameters, geometry, motion)
        if not r["valid"]:
            return costs, "invalid"
        costs.append(motion_cost(r["outputs"], record)[0])
        if reject_above is not None and len(costs) < len(motions) and sum(costs) > len(motions) * reject_above * (1 + 1e-12):
            return costs, "stopped early"
    return costs, "valid"


def training_cost(parameters, geometry, motions, records, reject_above=None):
    """J: mean motion cost over the training motions. Returns (J, status); a trial stopped early returns a lower bound."""
    costs, status = motion_costs(parameters, geometry, motions, records, reject_above)
    if status == "invalid":
        return PENALTY, status
    if status == "stopped early":
        return float(sum(costs)) / len(motions), status
    return float(np.mean(costs)), status


def validation_cost(parameters, geometry, motions, records):
    """V: mean motion cost over the 18 validation motions; an invalid motion counts with the penalty cost."""
    costs = []
    for motion, record in zip(motions, records):
        c, status = motion_costs(parameters, geometry, [motion], [record])
        costs.append(c[0] if status == "valid" else PENALTY)
    return float(np.mean(costs)), costs


def test_cost(predictions, records):
    """T: mean motion cost over the 18 test motions, from stored predictions (exact mean of the motion costs)."""
    costs = [motion_cost(p, r)[0] for p, r in zip(predictions, records)]
    return statistics.mean(costs), costs
