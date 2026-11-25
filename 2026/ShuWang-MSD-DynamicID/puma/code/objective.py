"""Training cost J, validation cost V and test cost T of a parameter vector.

The model always has the calibrated geometry. J, V and T are the same objective
(../cost.py) on the 4 training, 18 validation and 18 test motions. The LM uses
J as a sum of squares of the residual vector of residuals().
"""
import numpy as np

import config
import cost
import model
import motions
import records

INFEASIBLE_COST = model.read_json(config.CASE_DIR / "settings" / "optimizer.json")["de"]["infeasible_cost"]


class Data:
    """References and noisy teacher records of one geometry setting.

    Missing records are generated in <out_dir>/records/. record_dirs can point a
    split to other record files, for example the stored test records in ../signals/.
    """

    def __init__(self, setting, out_dir=config.OUTPUT_DIR, splits=("training", "validation"), record_dirs=None):
        self.setting = setting
        self.geometry = model.load_geometry("calibrated")
        self.sets = {}
        for split in splits:
            folder = (record_dirs or {}).get(split)
            if folder is None:
                records.generate(out_dir, setting, split)
                folder = records.folder(out_dir, setting, split)
            self.sets[split] = [(m["id"], motions.reference(m), cost.load(folder / f"{m['id']}.npz"))
                                for m in motions.motions(split)]

    def predictions(self, x, split):
        """Model outputs of every motion of a split: {id: record}."""
        return {mid: model.simulate(x, self.geometry, ref) for mid, ref, _ in self.sets[split]}

    def motion_costs(self, x, split, predictions=None):
        """{id: cost of the motion}; the predictions are simulated unless given."""
        predictions = predictions or self.predictions(x, split)
        return {mid: cost.motion_cost(predictions[mid], meas) for mid, _, meas in self.sets[split]}

    def cost(self, x, split):
        """Mean of the motion costs: J on 'training', V on 'validation', T on 'test'."""
        return float(np.mean(list(self.motion_costs(x, split).values())))

    def penalized_cost(self, x, split):
        """cost(), or INFEASIBLE_COST if x is infeasible or the simulation stops with an error."""
        if not model.feasible(x):
            return INFEASIBLE_COST
        try:
            value = self.cost(x, split)
        except Exception:
            return INFEASIBLE_COST
        return value if np.isfinite(value) else INFEASIBLE_COST

    def residuals(self, x, predictions=None):
        """Vector f with f @ f = J: sign(e) sqrt(loss(e) / n) per scored value, n = motions x groups x values of the group."""
        train = self.sets["training"]
        predictions = predictions or self.predictions(x, "training")
        parts = []
        for mid, _, meas in train:
            pred = predictions[mid]
            for group in cost.GROUPS:
                e = cost.residual(pred, meas, group).reshape(-1)
                parts.append(np.sign(e) * np.sqrt(cost.loss(e) / (len(train) * len(cost.GROUPS) * e.size)))
        return np.concatenate(parts)
