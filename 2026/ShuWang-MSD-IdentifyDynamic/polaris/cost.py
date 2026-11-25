"""Loss, cost and r of the vehicle outputs, as used for the training, validation and test costs.

The functions take the scored samples of a record: samples 100, 104, 108, ... (275 of 1200).
numpy only. No simulation.
"""
import numpy as np

# Output groups, in the order of the cost.
GROUPS = ("p", "q", "v", "speed", "wheel_omega", "delta", "gyro", "upright_rel")

# Noise standard deviation of one component of each group (SI units; q as a rotation vector in rad).
NOISE_SD = {
    "p": 0.003,                          # chassis position, m
    "q": 0.002,                          # chassis attitude, rad
    "v": 0.020,                          # chassis velocity, m/s
    "speed": 0.020,                      # forward speed, m/s
    "wheel_omega": 0.050,                # wheel angular speed, rad/s
    "delta": float(np.deg2rad(0.5)),     # steering proxy, rad (0.5 deg)
    "gyro": 0.005,                       # chassis angular velocity, rad/s
    "upright_rel": 0.003,                # upright position relative to the chassis, m
}

# Number of scalar components of each group.
COMPONENTS = {"p": 3, "q": 3, "v": 3, "speed": 1, "wheel_omega": 4, "delta": 1, "gyro": 3, "upright_rel": 12}

# Groups whose r uses the norm of the three-component error vector.
VECTOR_GROUPS = ("p", "q", "v")

SKIP = 100        # warm-up samples that are not scored
STRIDE = 4        # every fourth sample is scored
LOSS_SCALE = 3.0  # scale of the per-sample loss, in noise standard deviations


def scored_samples(x):
    """Scored samples of a full record (first axis = time)."""
    return np.asarray(x)[SKIP::STRIDE]


def rotation_vector(q_model, q_data):
    """Rotation vector of the rotation from the data attitude to the model attitude.

    Rows are unit quaternions (w, x, y, z). The result is the rotation vector of
    inverse(q_data) * q_model, with an angle in [0, pi].
    """
    a = np.asarray(q_model, dtype=float)
    b = np.asarray(q_data, dtype=float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    pw, pv = b[:, 0], -b[:, 1:]          # inverse of the data attitude
    qw, qv = a[:, 0], a[:, 1:]
    vec = pw[:, None] * qv + qw[:, None] * pv + np.cross(pv, qv)
    w = pw * qw - np.sum(pv * qv, axis=1)
    norm = np.sqrt(w * w + np.sum(vec * vec, axis=1))
    vec, w = vec / norm[:, None], w / norm
    flip = w < 0
    vec[flip] *= -1.0
    w = np.where(flip, -w, w)
    angle = 2.0 * np.arctan2(np.linalg.norm(vec, axis=1), w)
    small = angle <= 1e-3
    scale = np.empty_like(angle)
    scale[small] = 2.0 + angle[small] ** 2 / 12.0 + 7.0 * angle[small] ** 4 / 2880.0
    scale[~small] = angle[~small] / np.sin(angle[~small] / 2.0)
    return scale[:, None] * vec


def error(group, model, data):
    """Output error of one group: model minus data; for the attitude, the rotation vector."""
    if group == "q":
        return rotation_vector(model, data)
    return np.asarray(model, dtype=float) - np.asarray(data, dtype=float)


def loss(u):
    """Per-sample loss of a residual u in noise standard deviations: 18 (sqrt(1 + (u/3)^2) - 1)."""
    return 2.0 * LOSS_SCALE ** 2 * (np.sqrt(1.0 + (u / LOSS_SCALE) ** 2) - 1.0)


def group_cost(group, model, data):
    """Cost of one output group: mean loss over all scored samples and components."""
    return float(np.mean(loss(error(group, model, data) / NOISE_SD[group])))


def cost(model, data):
    """Cost of one motion: mean of the eight group costs. Returns (cost, {group: cost})."""
    groups = {g: group_cost(g, model[g], data[g]) for g in GROUPS}
    return float(np.mean(list(groups.values()))), groups


def r_value(group, model, data):
    """RMSE of one group divided by its noise level.

    Vector groups: RMS of the error norm, divided by sqrt(3) noise standard deviations.
    Other groups: RMS over all components, divided by one noise standard deviation.
    """
    e = error(group, model, data)
    if group in VECTOR_GROUPS:
        rmse = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
        return rmse / (NOISE_SD[group] * np.sqrt(3.0))
    return float(np.sqrt(np.mean(e * e))) / NOISE_SD[group]


def band(group, n_scored=275):
    """Three standard deviations of r for a model without error."""
    if group in VECTOR_GROUPS:
        return 3.0 * (6.0 * n_scored) ** -0.5
    return 3.0 * (2.0 * n_scored * COMPONENTS[group]) ** -0.5
