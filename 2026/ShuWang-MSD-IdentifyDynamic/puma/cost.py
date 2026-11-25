"""Loss, cost and test measure r of the PUMA case.

Each record is a set of arrays sampled at 100 Hz. Residuals are divided by
the noise standard deviation of their output group. All samples after the
first 0.2 s are scored. The module needs numpy only.
"""
import numpy as np

# Output group -> name of the array in the record files.
ARRAY = {"joint_angle": "q", "joint_rate": "qd", "specific_force": "imu_acc",
         "angular_rate": "imu_gyr", "torque": "tau"}
GROUPS = tuple(ARRAY)
# Noise standard deviations: rad, rad/s, m/s^2, rad/s, N m.
NOISE_SD = {"joint_angle": 1.0e-4, "joint_rate": 1.0e-3, "specific_force": 0.05,
            "angular_rate": 0.005, "torque": 0.05}
LOSS_SCALE = 3.0        # scale s of the soft-L1 loss, in noise standard deviations
SCORING_START_S = 0.2   # samples before this time are not scored
BAND_FACTOR = 3.0       # half-width of the band of r, in standard deviations of r under pure noise


def load(path):
    """Arrays of one record file."""
    with np.load(path, allow_pickle=False) as data:
        return {k: np.array(data[k]) for k in data.files}


def _scored(arr, time):
    return arr[time >= SCORING_START_S - 1e-12]


def residual(pred, meas, group):
    """(prediction - record) / noise standard deviation, samples x components.
    Joint-angle differences are wrapped to (-pi, pi]."""
    key = ARRAY[group]
    e = _scored(pred[key], meas["time"]) - _scored(meas[key], meas["time"])
    if group == "joint_angle":
        e = np.arctan2(np.sin(e), np.cos(e))
    return e.reshape(e.shape[0], -1) / NOISE_SD[group]


def loss(e):
    """Soft-L1 loss rho(e) = 2 s^2 (sqrt(1 + (e / s)^2) - 1)."""
    s = LOSS_SCALE
    return 2.0 * s * s * (np.sqrt(1.0 + (e / s) ** 2) - 1.0)


def group_cost(pred, meas, group):
    """Mean loss over the samples and components of one output group."""
    return float(np.mean(loss(residual(pred, meas, group))))


def motion_cost(pred, meas):
    """Cost of one motion: mean of the five group costs."""
    return float(np.mean([np.mean(loss(residual(pred, meas, g))) for g in GROUPS]))


def cost(pairs):
    """Cost of a set of motions: mean of the motion costs. pairs: [(prediction, record), ...]."""
    return float(np.mean([motion_cost(p, m) for p, m in pairs]))


def ratio(pred, meas, group):
    """Test measure r: RMSE / noise standard deviation per component, averaged over the components."""
    e = residual(pred, meas, group)
    return float(np.mean(np.sqrt(np.mean(e * e, axis=0))))


def values(meas, group):
    """Number of scored values of one group: scored samples x components."""
    return int(np.prod(residual(meas, meas, group).shape))


def band_half_width(n_values):
    """r lies outside the band if abs(r - 1) > 3 / sqrt(2 n), n = number of scored values."""
    return BAND_FACTOR / np.sqrt(2.0 * n_values)


def rmse(pred, ref, group):
    """RMS difference to a reference record per component, averaged over the components, in SI units."""
    return ratio(pred, ref, group) * NOISE_SD[group]
