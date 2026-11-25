"""Loss, cost and test measure r of the vehicle outputs (numpy only, no simulation).

The cost of one motion is the mean of eight group costs. A group cost is the mean of the loss
over the scored samples and the components of the group. r is the RMSE of a group divided by
the RMSE that noise alone gives.
"""
import numpy as np

GROUPS = ("position", "attitude", "velocity", "speed", "wheel_angular_speed",
          "steering_proxy", "angular_velocity", "upright_relative_position")

# Noise standard deviation of one component (m, rad, m/s, m/s, rad/s, rad, rad/s, m).
NOISE_STD = {
    "position": 0.003,
    "attitude": 0.002,
    "velocity": 0.020,
    "speed": 0.020,
    "wheel_angular_speed": 0.050,
    "steering_proxy": float(np.deg2rad(0.5)),
    "angular_velocity": 0.005,
    "upright_relative_position": 0.003,
}
COMPONENTS = {"position": 3, "attitude": 3, "velocity": 3, "speed": 1, "wheel_angular_speed": 4,
              "steering_proxy": 1, "angular_velocity": 3, "upright_relative_position": 12}
# These groups are scored by the norm of their three components in r.
VECTOR_GROUPS = ("position", "attitude", "velocity")

# A record has 1200 samples. Scoring skips the first 100 and uses every fourth of the rest.
RECORD_SAMPLES = 1200
SKIPPED_SAMPLES = 100
SAMPLE_STRIDE = 4
SCORED_SAMPLES = 275


def loss(e):
    """Loss of a residual e that is already divided by its noise standard deviation."""
    return 18 * (np.sqrt(1 + (e / 3) ** 2) - 1)


def _unit(q):
    """Quaternions (w, x, y, z) as unit quaternions in the order (x, y, z, w)."""
    x, y, z, w = q[:, 1], q[:, 2], q[:, 3], q[:, 0]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    return np.stack([x / n, y / n, z / n, w / n], axis=1)


def rotation_vector(q_candidate, q_target):
    """Rotation vector of target^-1 * candidate for quaternions (w, x, y, z), one row per sample."""
    a = _unit(np.asarray(q_candidate, dtype=float))
    b = _unit(np.asarray(q_target, dtype=float))
    b = b * np.array([-1.0, -1.0, -1.0, 1.0])            # inverse of the target rotation
    cx = b[:, 1] * a[:, 2] - b[:, 2] * a[:, 1]
    cy = b[:, 2] * a[:, 0] - b[:, 0] * a[:, 2]
    cz = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    r = np.stack([b[:, 3] * a[:, 0] + a[:, 3] * b[:, 0] + cx,
                  b[:, 3] * a[:, 1] + a[:, 3] * b[:, 1] + cy,
                  b[:, 3] * a[:, 2] + a[:, 3] * b[:, 2] + cz,
                  b[:, 3] * a[:, 3] - b[:, 0] * a[:, 0] - b[:, 1] * a[:, 1] - b[:, 2] * a[:, 2]], axis=1)
    r = r / np.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])[:, None]
    r = np.where(r[:, 3:4] < 0, -r, r)                   # rotation angle between 0 and pi
    angle = 2 * np.arctan2(np.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2]), r[:, 3])
    small = angle <= 1e-3
    angle2 = angle * angle
    safe = np.where(small, 1.0, angle)
    scale = np.where(small, 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880, safe / np.sin(safe / 2))
    return scale[:, None] * r[:, :3]


def errors(candidate, target):
    """Output error of each group in physical units; the attitude error is a rotation vector."""
    out = {}
    for g in GROUPS:
        if g == "attitude":
            out[g] = rotation_vector(candidate[g], target[g])
        else:
            out[g] = np.asarray(candidate[g], dtype=float) - np.asarray(target[g], dtype=float)
    return out


def group_costs(candidate, target):
    """Cost of each output group for one motion. The arrays hold the scored samples only."""
    e = errors(candidate, target)
    return {g: float(np.mean(loss(e[g] / NOISE_STD[g]))) for g in GROUPS}


def cost(candidate, target):
    """Cost of one motion: the mean of the eight group costs."""
    return float(np.mean(list(group_costs(candidate, target).values())))


def r_values(candidate, target):
    """Test measure r of each output group for one motion."""
    e = errors(candidate, target)
    out = {}
    for g in GROUPS:
        if g in VECTOR_GROUPS:
            rmse = float(np.sqrt(np.mean(np.sum(e[g] * e[g], axis=1))))
            floor = float(NOISE_STD[g] * np.sqrt(3))
        else:
            rmse = float(np.sqrt(np.mean(e[g] * e[g])))
            floor = float(NOISE_STD[g])
        out[g] = rmse / floor
    return out


def band(group, samples=SCORED_SAMPLES):
    """Three noise-only standard deviations of r for a group with the given number of scored samples."""
    if group in VECTOR_GROUPS:
        return 3 * (6 * samples) ** -0.5
    return 3 * (2 * samples * COMPONENTS[group]) ** -0.5


def load_record(path):
    """The eight output groups of one stored record or prediction (scored samples)."""
    with np.load(path, allow_pickle=False) as f:
        return {g: np.array(f[g]) for g in GROUPS}
