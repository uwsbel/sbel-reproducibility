"""Joint reference trajectories of the PUMA motions, from ../settings/motions.json.

Every reference starts at the home pose and is sampled on the 2-ms control grid.
Angles in the settings are in degrees and offsets from the home pose; rates are
second-order finite differences of the angles.
"""
import numpy as np
from scipy.interpolate import CubicSpline

import config
from model import DT, JOINT_LIMITS, Q0, read_json

LIMIT_MARGIN_RAD = 0.06
SETTINGS = read_json(config.CASE_DIR / "settings" / "motions.json")


def motions(split):
    """Motion entries of 'training', 'validation' or 'test', in the order of the settings file."""
    if split == "training":
        library = {m["id"]: m for m in SETTINGS["library"]}
        return [library[i] for i in SETTINGS["training"]]
    return list(SETTINGS[split])


def smoothstep5(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def _append_smooth(rows, start, stop, duration_s):
    count = max(3, int(round(float(duration_s) / DT)))
    blend = smoothstep5(np.arange(1, count + 1, dtype=float) / count)
    rows.append(start[None, :] + blend[:, None] * (stop - start)[None, :])
    return stop.copy()


def _append_hold(rows, pose, duration_s):
    count = max(1, int(round(float(duration_s) / DT)))
    rows.append(np.repeat(pose[None, :], count, axis=0))
    return pose.copy()


def _pose(offset_deg):
    return Q0 + np.deg2rad(np.asarray(offset_deg, dtype=float))


def gravity_pose_cycle(p):
    """Smooth moves between static poses, with a hold at each pose."""
    rows, current = [], Q0.copy()
    for offset in p["poses_deg"]:
        current = _append_smooth(rows, current, _pose(offset), p["transition_s"])
        current = _append_hold(rows, current, p["hold_s"])
    _append_smooth(rows, current, Q0, p["transition_s"])
    return np.vstack(rows)


def _sweep_fraction(travel, speed, accel_s):
    """Fraction of the travel at each step of one constant-speed sweep (S-curve start and stop if accel_s > 0)."""
    if accel_s <= 0.0:
        count = max(3, int(round(travel / max(speed, 1.0e-6) / DT)))
        return np.arange(1, count + 1, dtype=float) / count
    total = travel / speed + accel_s
    if travel / speed <= accel_s:
        raise ValueError("sweep too short for the requested acceleration time")
    t = np.arange(1, int(round(total / DT)) + 1, dtype=float) * DT
    t[-1] = total

    def ramp_distance(tau):  # integral of smoothstep5 on [0, tau]
        tau = np.clip(tau, 0.0, 1.0)
        return 2.5 * tau**4 - 3.0 * tau**5 + tau**6

    distance = speed * (t - 0.5 * accel_s)
    distance = np.where(t < accel_s, speed * accel_s * ramp_distance(t / accel_s), distance)
    distance = np.where(t > total - accel_s, travel - speed * accel_s * ramp_distance((total - t) / accel_s), distance)
    return distance / travel


def friction_velocity_sweep(p):
    """Back-and-forth sweeps of single joints at constant speeds."""
    rows, current = [], Q0.copy()
    transition = float(p.get("transition_s", 0.35))
    pause = float(p.get("pause_s", 0.16))
    travel = np.deg2rad(float(p["travel_deg"]))
    speeds = np.deg2rad(np.asarray(p["speed_levels_deg_s"], dtype=float))
    for posture_offset in p["postures_deg"]:
        posture = _pose(posture_offset)
        current = _append_smooth(rows, current, posture, transition)
        for joint in [int(v) for v in p["active_joints"]]:
            for speed in speeds:
                low, high = posture.copy(), posture.copy()
                low[joint] -= 0.5 * travel
                high[joint] += 0.5 * travel
                if not (p.get("skip_idle_transitions") and np.allclose(current, low, rtol=0.0, atol=1e-12)):
                    current = _append_smooth(rows, current, low, transition)
                alpha = _sweep_fraction(travel, speed, float(p.get("accel_s", 0.0)))
                rows.append(low[None, :] + alpha[:, None] * (high - low)[None, :])
                current = _append_hold(rows, high, pause)
                rows.append(high[None, :] + alpha[:, None] * (low - high)[None, :])
                current = _append_hold(rows, low, pause)
        current = _append_smooth(rows, current, posture, transition)
    _append_smooth(rows, current, Q0, transition)
    return np.vstack(rows)


def inertia_multisine(p):
    """Sum of sines on all joints about a base posture, with a smooth start."""
    duration = float(p["duration_s"])
    t = np.arange(int(round(duration / DT)), dtype=float) * DT
    base = _pose(p.get("base_offset_deg", [0.0, 0.0, 0.0]))
    q = np.repeat(base[None, :], len(t), axis=0)
    envelope = smoothstep5(t / float(p.get("ramp_s", 0.75)))
    amps = np.deg2rad(np.asarray(p["amplitudes_deg"], dtype=float))
    freqs = np.asarray(p["frequencies_hz"], dtype=float)
    phases = np.asarray(p["phases_rad"], dtype=float)
    for j in range(3):
        q[:, j] += envelope * np.sum(
            amps[j][None, :] * np.sin(2.0 * np.pi * t[:, None] * freqs[j][None, :] + phases[j][None, :]), axis=1)
    lead = float(p.get("start_transition_s", 0.0))
    if lead > 0.0:
        rows = []
        _append_smooth(rows, Q0.copy(), base, lead)
        q = np.vstack(rows + [q])
    return q


def coupling_modal_blocks(p):
    """Blocks of in-phase sine motion of several joints, each about its own posture."""
    rows, current = [], Q0.copy()
    transition = float(p.get("transition_s", 0.45))
    for block in p["blocks"]:
        base = _pose(block["base_offset_deg"])
        current = _append_smooth(rows, current, base, transition)
        duration = float(block["duration_s"])
        t = np.arange(int(round(duration / DT)), dtype=float) * DT
        envelope = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 2
        amp = np.deg2rad(np.asarray(block["amplitudes_deg"], dtype=float))
        phase = np.asarray(block["phases_rad"], dtype=float)
        q = np.repeat(base[None, :], len(t), axis=0)
        q += envelope[:, None] * amp[None, :] * np.sin(2.0 * np.pi * float(block["frequency_hz"]) * t[:, None] + phase[None, :])
        rows.append(q)
        current = q[-1].copy()
    _append_smooth(rows, current, Q0, transition)
    return np.vstack(rows)


def multi_posture_multisine_blocks(p):
    """Blocks of multisine motion, each about its own posture, with fade-in and fade-out."""
    rows, current = [], Q0.copy()
    transition = float(p.get("transition_s", 0.45))
    for block in p["blocks"]:
        target = _pose(block["base_offset_deg"])
        count = max(2, int(round(transition / DT)))
        blend = smoothstep5(np.arange(count, dtype=float) / count)
        rows.append(current[None, :] + blend[:, None] * (target - current)[None, :])
        duration = float(block["duration_s"])
        local_t = np.arange(max(2, int(round(duration / DT))), dtype=float) * DT
        envelope = smoothstep5(local_t / min(0.8, max(duration, DT)))
        fade_out = float(p.get("fade_out_s", 0.0))
        if fade_out > 0.0:
            envelope = envelope * smoothstep5((local_t[-1] - local_t) / fade_out)
        amps = np.deg2rad(np.asarray(block["amplitudes_deg"], dtype=float))
        freqs = np.asarray(block["frequencies_hz"], dtype=float)
        phases = np.asarray(block["phases_rad"], dtype=float)
        profile = np.repeat(target[None, :], len(local_t), axis=0)
        for j in range(3):
            profile[:, j] += envelope * np.sum(
                amps[j][None, :] * np.sin(2.0 * np.pi * local_t[:, None] * freqs[j][None, :] + phases[j][None, :]), axis=1)
        rows.append(profile)
        current = profile[-1].copy()
    count = max(2, int(round(transition / DT)))
    blend = smoothstep5(np.arange(count, dtype=float) / count)
    rows.append(current[None, :] + blend[:, None] * (Q0 - current)[None, :])
    return np.vstack(rows)


def spline_path(p):
    """Clamped cubic spline from the home pose through via points back to the home pose."""
    points = np.vstack([Q0, *[_pose(v) for v in p["via_offsets_deg"]], Q0])
    knots = np.arange(len(points), dtype=float) * float(p["knot_interval_s"])
    spline = CubicSpline(knots, points, bc_type="clamped")
    t = np.arange(int(round(knots[-1] / DT)) + 1, dtype=float) * DT
    return spline(t)


def chirp(p):
    """Linear frequency sweep of each joint about a base posture, with fade-in and fade-out."""
    base = _pose(p["base_offset_deg"])
    sweep = float(p["sweep_s"])
    t = np.arange(int(round(sweep / DT)), dtype=float) * DT
    fade = float(p["fade_s"])
    envelope = smoothstep5(t / fade) * smoothstep5((sweep - t) / fade)
    q = np.repeat(base[None, :], len(t), axis=0)
    for j in range(3):
        f0 = float(p["f0_hz"]) * float(p["joint_frequency_scale"][j])
        f1 = float(p["f1_hz"]) * float(p["joint_frequency_scale"][j])
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / sweep) + float(p["phases_rad"][j])
        q[:, j] += envelope * np.deg2rad(float(p["amplitudes_deg"][j])) * np.sin(phase)
    rows = []
    _append_smooth(rows, Q0.copy(), base, p["transition_s"])
    rows.append(q)
    _append_smooth(rows, base.copy(), Q0, p["transition_s"])
    return np.vstack(rows)


GENERATORS = {f.__name__: f for f in (gravity_pose_cycle, friction_velocity_sweep, inertia_multisine,
                                      coupling_modal_blocks, multi_posture_multisine_blocks, spline_path, chirp)}


def reference(motion):
    """(t, q_ref, qd_ref) of one motion entry on the 2-ms grid; the joint limits are checked."""
    q = GENERATORS[motion["reference_type"]](motion["reference_parameters"])
    if np.any(q <= JOINT_LIMITS[:, 0] + LIMIT_MARGIN_RAD) or np.any(q >= JOINT_LIMITS[:, 1] - LIMIT_MARGIN_RAD):
        raise ValueError(f"{motion['id']}: reference comes closer than {LIMIT_MARGIN_RAD} rad to a joint limit")
    qd = np.gradient(q, DT, axis=0, edge_order=2)
    t = np.arange(len(q), dtype=float) * DT
    return t, q, qd
