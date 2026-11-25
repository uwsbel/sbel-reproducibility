"""Forward model of the PUMA case: reduced PUMA-560 arm with its joint PD controller.

The arm has three revolute joints (Z-Y-Y) on a fixed base. The geometry comes from
../geometry/, the 17 parameters from ../settings/bounds.json. simulate() holds the
home pose for 2.5 s and then runs one motion; it returns the 100-Hz outputs.
"""
import contextlib
import json
import math
import os
import sys

import numpy as np
import pychrono as chrono
from scipy.spatial.transform import Rotation

import config

# Robot and controller. Angles in rad, torques in N m.
DT = 0.002                      # control interval and integration step, s
OUTPUT_STRIDE = 5               # one output sample every fifth step (100 Hz)
GRAVITY = np.array([0.0, 0.0, -9.81])
PEDESTAL_HEIGHT = 26.45 * 0.0254
Q0 = np.deg2rad(np.array([0.0, 30.0, -50.0]))          # home pose
JOINT_LIMITS = np.deg2rad(np.array([[-160.0, 160.0], [-110.0, 110.0], [-135.0, 135.0]]))
KP = np.array([120.0, 260.0, 150.0])
KD = np.array([14.0, 30.0, 16.0])
TORQUE_LIMITS = np.array([100.0, 220.0, 130.0])
FRICTION_SMOOTHING = 0.025      # rad/s, Coulomb term fc * tanh(rate / 0.025)
NOMINAL_AXES = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
HHT = {"alpha": -0.2, "max_iterations": 60, "abs_tol_state": 1.0e-8, "abs_tol_multiplier": 1.0e-6, "rel_tol": 1.0e-8}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


_BOUNDS = read_json(config.CASE_DIR / "settings" / "bounds.json")
NAMES = [p["name"] for p in _BOUNDS["parameters"]]
LOWER = np.array([p["lower"] for p in _BOUNDS["parameters"]])
UPPER = np.array([p["upper"] for p in _BOUNDS["parameters"]])
CENTRE = np.mean(np.stack([LOWER, UPPER], axis=1), axis=1)
HALF = 0.5 * (UPPER - LOWER)
HOLD_S = read_json(config.CASE_DIR / "settings" / "motions.json")["hold_before_each_motion_s"]


def x_from_z(z):
    """Parameter values from coordinates z (half-ranges from the centre of the box)."""
    return CENTRE + HALF * np.asarray(z, dtype=float)


def _triangle_ok(d, tol=1.0e-12):
    x, y, z = map(float, d)
    return x > 0.0 and y > 0.0 and z > 0.0 and x + y + tol >= z and x + z + tol >= y and y + z + tol >= x


def decode(x):
    """Parameter vector -> (dynamics of the three links, feasible).

    Feasible: inside the box, centres of mass inside their intervals, principal
    inertias positive and satisfying the triangle inequalities.
    """
    x = np.asarray(x, dtype=float)
    fixed, com = _BOUNDS["fixed"], _BOUNDS["centre_of_mass_intervals_m"]
    ok = not (np.any(x < LOWER) or np.any(x > UPPER))
    m2, m3 = x[0], x[1]
    if m2 <= 0.0 or m3 <= 0.0:
        ok, m2, m3 = False, max(m2, 1.0e-12), max(m3, 1.0e-12)
    c2 = np.array([x[2] / m2, 0.0, x[3] / m2])
    c3 = np.array([x[4] / m3, 0.0, x[5] / m3])
    for c, name in ((c2, "c2"), (c3, "c3")):
        for i, axis in ((0, "x"), (2, "z")):
            lo, hi = com[name + axis]
            if c[i] < lo - 1.0e-12 or c[i] > hi + 1.0e-12:
                ok = False
    inertia = np.array([[fixed["I1x_kg_m2"], fixed["I1y_kg_m2"], x[6]],
                        [fixed["I2x_kg_m2"], x[7], x[8]],
                        [fixed["I3x_kg_m2"], x[9], x[10]]], dtype=float)
    ok = ok and all(_triangle_ok(row) for row in inertia)
    dynamics = {"mass": np.array([fixed["m1_kg"], m2, m3], dtype=float),
                "com_ref": np.vstack([fixed["c1_m"], c2, c3]).astype(float),  # from the IMU origin of each link
                "inertia_xx": inertia,
                "viscous": np.array(x[11:14], dtype=float),
                "coulomb": np.array(x[14:17], dtype=float)}
    return dynamics, bool(ok)


def feasible(x):
    return decode(x)[1]


def load_geometry(name):
    """'true' or 'calibrated': joint axes and levers of ../geometry/<name>.json as arrays."""
    joints = read_json(config.CASE_DIR / "geometry" / f"{name}.json")["joints"]
    return {"axis_parent": np.array([j["axis_parent"] for j in joints], dtype=float),
            "lever_parent": np.array([j["lever_parent_m"] for j in joints], dtype=float),
            "lever_child": np.array([j["lever_child_m"] for j in joints], dtype=float)}


def _ch_quat(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return chrono.ChQuaternionD(float(w), float(x), float(y), float(z))


def _frame_from_z(axis):
    z = np.asarray(axis, dtype=float)
    z /= np.linalg.norm(z)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, z)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    x = trial - z * np.dot(trial, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _vec(v):
    return np.array([v.x, v.y, v.z], dtype=float)


def _rotation_matrix(q):
    w, x, y, z = q.e0, q.e1, q.e2, q.e3
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _dyn_vector(n):
    v = chrono.ChVectorDynamicD(n)
    for i in range(n):
        v[i] = 0.0
    return v


class _MotorTorque(chrono.ChFunction):
    """Motor torque = held command - viscous and Coulomb friction at the current joint rate."""

    def __init__(self, motor, viscous, coulomb):
        super().__init__()
        self.motor, self.b, self.fc, self.command = motor, float(viscous), float(coulomb), 0.0

    def Get_y(self, t):
        v = float(self.motor.GetMotorRot_dt())
        return float(self.command - self.b * v - self.fc * math.tanh(v / FRICTION_SMOOTHING))


class Arm:
    """One arm at the home pose, integrated with Chrono HHT at a fixed 2-ms step.

    Each link is a body whose reference frame sits at its IMU. The parameters move
    the centre of mass inside this frame; the joints follow from the geometry.
    """

    def __init__(self, x, geometry):
        self.dyn, ok = decode(x)
        if not ok:
            raise ValueError("infeasible parameter vector")
        s = self.system = chrono.ChSystemNSC()
        s.Set_G_acc(chrono.ChVectorD(*map(float, GRAVITY)))
        s.SetSolverMaxIterations(120)
        s.SetSolverTolerance(1e-10)
        base = chrono.ChBodyEasyBox(0.34, 0.34, PEDESTAL_HEIGHT, 2500.0, False, False)
        base.SetPos(chrono.ChVectorD(0.0, 0.0, PEDESTAL_HEIGHT / 2.0))
        base.SetBodyFixed(True)
        s.Add(base)

        # Link frames at the home pose; joint and IMU points from the levers.
        r1 = Rotation.from_rotvec([0.0, 0.0, Q0[0]]).as_matrix()
        r2 = r1 @ Rotation.from_rotvec([0.0, Q0[1], 0.0]).as_matrix()
        r3 = r2 @ Rotation.from_rotvec([0.0, Q0[2], 0.0]).as_matrix()
        rotations = (np.eye(3), r1, r2, r3)
        parent_sensor = np.array([0.0, 0.0, PEDESTAL_HEIGHT])
        self.bodies, joint_points = [], []
        for j in range(3):
            joint = parent_sensor + rotations[j] @ geometry["lever_parent"][j]
            sensor = joint - rotations[j + 1] @ geometry["lever_child"][j]
            body = chrono.ChBodyAuxRef()
            body.SetFrame_REF_to_abs(chrono.ChFrameD(chrono.ChVectorD(*map(float, sensor)), _ch_quat(rotations[j + 1])))
            body.SetFrame_COG_to_REF(chrono.ChFrameD(chrono.ChVectorD(*map(float, self.dyn["com_ref"][j])), chrono.QUNIT))
            body.SetMass(float(self.dyn["mass"][j]))
            body.SetInertiaXX(chrono.ChVectorD(*map(float, self.dyn["inertia_xx"][j])))
            body.SetInertiaXY(chrono.ChVectorD(0.0, 0.0, 0.0))
            s.Add(body)
            self.bodies.append(body)
            joint_points.append(joint)
            parent_sensor = sensor

        self.motors = []
        for j, (parent, child) in enumerate(zip([base] + self.bodies[:2], self.bodies)):
            axis = rotations[j] @ geometry["axis_parent"][j]
            axis /= np.linalg.norm(axis)
            if np.dot(axis, rotations[j] @ NOMINAL_AXES[j]) < 0.0:
                axis = -axis
            motor = chrono.ChLinkMotorRotationTorque()
            motor.Initialize(child, parent, chrono.ChFrameD(chrono.ChVectorD(*map(float, joint_points[j])),
                                                           _ch_quat(_frame_from_z(axis))))
            motor.SetTorqueFunction(chrono.ChFunction_Const(0.0))
            s.Add(motor)
            self.motors.append(motor)

        s.SetSolverType(chrono.ChSolver.Type_SPARSE_QR)
        s.SetNumThreads(1, 1, 1)
        tp = self.tp = chrono.ChTimestepperHHT(s)
        tp.SetAlpha(HHT["alpha"])
        tp.SetMode(chrono.ChTimestepperHHT.ACCELERATION)
        tp.SetScaling(False)
        tp.SetModifiedNewton(False)
        tp.SetStepControl(True)
        tp.SetMaxiters(HHT["max_iterations"])
        tp.SetAbsTolerances(HHT["abs_tol_state"], HHT["abs_tol_multiplier"])
        tp.SetRelTolerance(HHT["rel_tol"])
        s.SetTimestepper(tp)
        self.torques = []
        for j, motor in enumerate(self.motors):
            fn = _MotorTorque(motor, self.dyn["viscous"][j], self.dyn["coulomb"][j])
            motor.SetTorqueFunction(fn)
            self.torques.append(fn)

        # Consistent accelerations and reactions at rest.
        s.DoAssembly(0)
        x, v, a = chrono.ChState(s), chrono.ChStateDelta(s), chrono.ChStateDelta(s)
        s.StateSetup(x, v, a)
        x, v = self._gather()
        lam = _dyn_vector(s.GetNconstr())
        if not s.StateSolveA(a, lam, x, v, s.GetChTime(), DT, False, False):
            raise RuntimeError("initial acceleration solve did not converge")
        s.StateScatterAcceleration(a)
        s.StateScatterReactions(lam)

        # Work vectors of the command update.
        nv, nc = s.GetNcoords_v(), s.GetNconstr()
        self._jx, self._jv, spare = chrono.ChState(s), chrono.ChStateDelta(s), chrono.ChStateDelta(s)
        s.StateSetup(self._jx, self._jv, spare)
        self._da, self._acc = chrono.ChStateDelta(s), chrono.ChStateDelta(s)
        s.StateSetup(chrono.ChState(s), self._da, self._acc)
        self._r, self._dl, self._qc, self._lam = _dyn_vector(nv), _dyn_vector(nc), _dyn_vector(nc), _dyn_vector(nc)
        self._zeros_v, self._zeros_c = [0.0] * nv, [0.0] * nc

    def _gather(self):
        s = self.system
        x, v, a = chrono.ChState(s), chrono.ChStateDelta(s), chrono.ChStateDelta(s)
        s.StateSetup(x, v, a)
        for b in self.bodies:
            ox, ov, q = b.GetOffset_x(), b.GetOffset_w(), b.GetRot()
            for i, value in enumerate((*_vec(b.GetPos()), q.e0, q.e1, q.e2, q.e3)):
                x[ox + i] = float(value)
            for i, value in enumerate((*_vec(b.GetPos_dt()), *_vec(b.GetWvel_loc()))):
                v[ov + i] = float(value)
        return x, v

    def _set_command(self, u):
        """New held command: update accelerations and reactions at fixed state and time.

        Solves [M, Cq^T; Cq, 0] for the change of the applied forces caused by the
        command change, so that HHT starts each step from consistent accelerations.
        """
        s, t = self.system, self.system.GetChTime()
        r, da, dl = self._r, self._da, self._dl
        r.SetVect(self._zeros_v)
        s.LoadResidual_F(r, -1.0)
        for fn, cmd in zip(self.torques, u):
            fn.command = float(cmd)
        s.LoadResidual_F(r, 1.0)
        da.SetVect(self._zeros_v)
        dl.SetVect(self._zeros_c)
        if not s.StateSolveCorrection(da, dl, r, self._qc, 1.0, 0.0, 0.0, self._jx, self._jv, t, False, False, True):
            raise RuntimeError("command update did not converge")
        a, lam = self._acc, self._lam
        s.StateGatherAcceleration(a)
        s.StateGatherReactions(lam)
        a.SetVect((np.asarray(a.GetVect()).ravel() + np.asarray(da.GetVect()).ravel()).tolist())
        lam.SetVect((np.asarray(lam.GetVect()).ravel() + np.asarray(dl.GetVect()).ravel()).tolist())
        s.StateScatterAcceleration(a)
        s.StateScatterReactions(lam)

    def joint_state(self):
        q = Q0 + np.array([m.GetMotorRot() for m in self.motors], dtype=float)
        qd = np.array([m.GetMotorRot_dt() for m in self.motors], dtype=float)
        return q, qd

    def step(self, u):
        """Hold command u over one 2-ms step."""
        self._set_command(np.asarray(u, dtype=float))
        self.tp.SetMinStepSize(0.999 * DT)
        t0 = self.system.GetChTime()
        self.system.DoStepDynamics(DT)
        if abs(self.system.GetChTime() - t0 - DT) > 1e-11:
            raise RuntimeError("HHT changed its step size")
        if self.tp.GetNumIterations() >= self.tp.GetMaxiters():
            raise RuntimeError("HHT iteration limit reached")

    def sensors(self):
        """Per link: rotation to the world frame, world velocity of the IMU point, body angular rate."""
        out = []
        for b, com in zip(self.bodies, self.dyn["com_ref"]):
            rot = _rotation_matrix(b.GetRot())
            w = _vec(b.GetWvel_par())
            out.append((rot, _vec(b.GetPos_dt()) + _cross(w, rot @ (-com)), _vec(b.GetWvel_loc())))
        return out


def controller(q, qd, q_ref, qd_ref):
    """Joint PD law with torque limits; returns the applied command."""
    error = np.arctan2(np.sin(q_ref - q), np.cos(q_ref - q))
    return np.clip(KP * error + KD * (qd_ref - qd), -TORQUE_LIMITS, TORQUE_LIMITS)


@contextlib.contextmanager
def quiet():
    """Hide the console output of the Chrono library (some builds print a line at every step)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (devnull, *saved):
            os.close(fd)


def simulate(x, geometry, reference):
    """Hold the home pose for HOLD_S, then run one motion; return the 100-Hz outputs.

    reference = (t, q_ref, qd_ref) on the 2-ms grid. Row j is recorded after step 5 j.
    The accelerometer gives the mean specific force over the last step in the link frame.
    """
    with quiet():
        return _simulate(x, geometry, reference)


def _simulate(x, geometry, reference):
    arm = Arm(x, geometry)
    for _ in range(int(round(HOLD_S / DT))):
        q, qd = arm.joint_state()
        arm.step(controller(q, qd, Q0, np.zeros(3)))
    t, q_ref, qd_ref = (np.asarray(a, dtype=float) for a in reference)
    rows = {"q": [], "qd": [], "imu_acc": [], "imu_gyr": [], "tau": []}
    prev = arm.sensors()
    for k in range(len(t)):
        q, qd = arm.joint_state()
        u = controller(q, qd, q_ref[k], qd_ref[k])
        arm.step(u)
        if k % OUTPUT_STRIDE == OUTPUT_STRIDE - 1:
            prev = arm.sensors()
        if k % OUTPUT_STRIDE == 0:
            kin = arm.sensors()
            q, qd = arm.joint_state()
            rows["q"].append(np.asarray(q))
            rows["qd"].append(np.asarray(qd))
            rows["imu_acc"].append(np.asarray([r.T @ ((v - pv) / DT - GRAVITY) for (r, v, _), (_, pv, _) in zip(kin, prev)]))
            rows["imu_gyr"].append(np.asarray([w for _, _, w in kin]))
            rows["tau"].append(np.asarray(u))
    out = {key: np.asarray(value, dtype=float) for key, value in rows.items()}
    out["time"] = t[::OUTPUT_STRIDE].copy()
    return out
