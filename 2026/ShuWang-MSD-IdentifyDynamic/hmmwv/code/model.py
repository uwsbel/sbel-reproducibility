"""Forward model of the HMMWV: build the vehicle, settle it, drive one motion, return the outputs.

simulate() builds a Chrono wheeled vehicle from the seven chassis parameters and a suspension
geometry, settles it at rest for 2.0 s, drives one input sequence in the same simulation and
returns the eight output groups at the control interval, with the validity of the rollout.
"""
import contextlib
import copy
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import config

if config.PYCHRONO_DIR:
    sys.path.insert(0, str(config.PYCHRONO_DIR))
import pychrono as chrono  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent.parent
SIM = json.loads((CASE_DIR / "settings" / "simulation.json").read_text(encoding="utf-8"))
DT = SIM["integration_step_s"]
SUBSTEPS = round(SIM["control_interval_s"] / DT)
PARAMETERS = ("mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz")
GROUPS = ("position", "attitude", "velocity", "speed", "wheel_angular_speed",
          "steering_proxy", "angular_velocity", "upright_relative_position")
UPRIGHTS = ["HMMWV DoubleWishbone Front_upright_L", "HMMWV DoubleWishbone Front_upright_R",
            "HMMWV DoubleWishbone Rear_upright_L", "HMMWV DoubleWishbone Rear_upright_R"]
STEERING = {"Input File": "hmmwv/steering/HMMWV_PitmanArm.json", "Location": [1.24, 0, 0.4],
            "Orientation": [0.98699, 0, 0.16074, 0]}
_STATE = {}


class InvalidRollout(Exception):
    """A rollout that breaks a validity rule; the candidate gets the penalty cost."""


# ------------------------------------------------------------------ inputs
def load_geometry(name):
    """'true': the stock Chrono suspension files. 'calibrated': the package geometry and body inertias."""
    if name == "true":
        return None
    g = json.loads((CASE_DIR / "geometry" / "calibrated.json").read_text(encoding="utf-8"))
    inertia = json.loads((CASE_DIR / "settings" / "calibrated_suspension_inertia.json").read_text(encoding="utf-8"))
    return {"front": g["front"], "rear": g["rear"], "body_inertia": inertia["bodies"]}


def load_motion(entry):
    """Input sequence (throttle, steering, braking per control interval) and terrain of one motion of motions.json."""
    with np.load(CASE_DIR / entry["file"], allow_pickle=False) as f:
        control = np.array(f["control"])
    terrain = entry["terrain"]
    if terrain != "flat" and "variant_of" in entry:     # the confirmation variant uses its own rough road
        terrain += ", confirmation variant"
    return {"id": entry["id"], "control": control, "terrain": terrain}


def feasible(parameters):
    """Positive friction and mass; positive inertias that satisfy the triangle inequalities."""
    p = np.asarray(parameters, dtype=float)
    return bool(np.isfinite(p).all() and p[0] > 0 and p[1] > 0 and np.min(p[4:]) > 0
                and np.min(p[4:].sum() - 2 * p[4:]) >= 0)


# ------------------------------------------------------------------ vehicle files
def _data_root():
    """Private copy of the Chrono vehicle data of this process; the vehicle files of each rollout go next to it."""
    if "root" not in _STATE:
        root = Path(config.OUTPUT_DIR) / "vehicle_data" / f"process_{os.getpid()}"
        shutil.copytree(Path(config.CHRONO_DATA_DIR) / "hmmwv", root / "hmmwv", dirs_exist_ok=True)
        (root / "generated").mkdir(exist_ok=True)
        veh.SetDataPath(str(root) + os.sep)
        _STATE.update(root=root, count=0)
    return _STATE["root"]


def _write(root, rel, value):
    with (root / rel).open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _vehicle_files(params, geometry):
    """Write the chassis, suspension and vehicle files of one rollout; return the vehicle file name."""
    root = _data_root()
    _STATE["count"] += 1
    tag = f"generated/{_STATE['count']:06d}"
    mu, mass, com_x, com_z, ixx, iyy, izz = map(float, params)
    _write(root, f"{tag}_chassis.json", {
        "Name": "HMMWV Chassis", "Type": "Chassis", "Template": "RigidChassis",
        "Components": [{"Centroidal Frame": {"Location": [com_x, 0.0, com_z], "Orientation": [1, 0, 0, 0]},
                        "Mass": mass, "Moments of Inertia": [ixx, iyy, izz], "Products of Inertia": [0, 0, 0], "Void": False}],
        "Driver Position": {"Location": [0.87, 0.7, 1.05], "Orientation": [1, 0, 0, 0]}})
    for axle in ("front", "rear"):
        if geometry is None:
            # Stock file, written again with the spring forces and the damping coefficient as floats.
            data = json.loads((root / f"hmmwv/suspension/HMMWV_DoubleWishbone{axle.title()}.json").read_text(encoding="utf-8"))
            data["Spring"]["Spring Curve Data"] = [[float(x), float(force) * 1.0] for x, force in data["Spring"]["Spring Curve Data"]]
            data["Shock"]["Damping Coefficient"] = float(data["Shock"]["Damping Coefficient"]) * 1.0
        else:
            data = copy.deepcopy(geometry[axle])
        _write(root, f"{tag}_{axle}.json", data)
    axle = lambda suspension, x, steering, brake: {
        "Suspension Input File": suspension, "Suspension Location": [x, 0, 0], "Steering Index": steering,
        "Left Wheel Input File": "hmmwv/wheel/HMMWV_Wheel.json", "Right Wheel Input File": "hmmwv/wheel/HMMWV_Wheel.json",
        "Left Brake Input File": brake, "Right Brake Input File": brake}
    _write(root, f"{tag}_vehicle.json", {
        "Name": "HMMWV", "Type": "Vehicle", "Template": "WheeledVehicle",
        "Chassis": {"Input File": f"{tag}_chassis.json"},
        "Axles": [axle(f"{tag}_front.json", 1.688, 0, "hmmwv/brake/HMMWV_BrakeSimple_Front.json"),
                  axle(f"{tag}_rear.json", -1.688, -1, "hmmwv/brake/HMMWV_BrakeSimple_Rear.json")],
        "Steering Subsystems": [{"Input File": STEERING["Input File"], "Location": STEERING["Location"],
                                 "Orientation": STEERING["Orientation"]}],
        "Wheelbase": 3.378, "Maximum Steering Angle (deg)": 30,
        "Driveline": {"Input File": "hmmwv/driveline/HMMWV_Driveline2WD.json", "Suspension Indexes": [1]}})
    return f"{tag}_vehicle.json"


# ------------------------------------------------------------------ build
def build(params, geometry, terrain):
    """Vehicle, terrain and system for one rollout.

    Returns (vehicle, ground, system, wheels, rigid, road): ground is the terrain the tires query, rigid the
    Chrono terrain object, road the rough-road query object (None on flat ground).
    """
    vehicle = veh.WheeledVehicle(veh.GetDataFile(_vehicle_files(params, geometry)), chrono.ChContactMethod_SMC)
    vehicle.Initialize(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, SIM["initial_chassis_height_m"]), chrono.ChQuaternionD(1, 0, 0, 0)))
    for setter in (vehicle.SetChassisVisualizationType, vehicle.SetSuspensionVisualizationType,
                   vehicle.SetSteeringVisualizationType, vehicle.SetWheelVisualizationType, vehicle.SetTireVisualizationType):
        setter(veh.VisualizationType_NONE)
    vehicle.InitializePowertrain(veh.HMMWV_SimpleMapPowertrain("powertrain"))
    wheels = [wheel for axle in vehicle.GetAxles() for wheel in axle.GetWheels()]
    for wheel in wheels:
        vehicle.InitializeTire(veh.ReadTireJSON(veh.GetDataFile("hmmwv/tire/HMMWV_TMeasyTire.json")), wheel, veh.VisualizationType_NONE)
    system = vehicle.GetSystem()                   # SMC contact with the Chrono default settings
    rigid = veh.RigidTerrain(system)
    m = SIM["terrain_material"]
    material = chrono.ChMaterialSurfaceSMC()
    material.SetFriction(float(params[0]))
    material.SetYoungModulus(m["young_modulus_pa"])
    material.SetPoissonRatio(m["poisson_ratio"])
    material.SetRestitution(m["restitution"])
    material.SetAdhesion(m["adhesion_n"])
    if terrain == "flat":
        rigid.AddPatch(material, chrono.CSYSNORM, *SIM["flat_patch_size_m"])
    else:
        r = SIM["rough_roads"][terrain]
        rigid.AddPatch(material, chrono.CSYSNORM, str(CASE_DIR / r["height_map"]), r["length_m"], r["width_m"],
                       r["minimum_height_m"], r["maximum_height_m"])
    rigid.Initialize()
    if geometry is not None:                       # inertias of the calibrated suspension bodies
        bodies = {b.GetNameString(): b for b in system.Get_bodylist()}
        for name, inertia in geometry["body_inertia"].items():
            i = np.array(inertia)
            bodies[name].SetInertiaXX(chrono.ChVectorD(*np.diag(i)))
            bodies[name].SetInertiaXY(chrono.ChVectorD(i[0, 1], i[0, 2], i[1, 2]))
    # The rear-drive differential carries no torque in this model; its clutch is disabled. Sparse LU solver.
    clutches = [c for c in map(chrono.CastToChShaftsClutch, system.Get_otherphysicslist()) if c]
    assert len(clutches) == 1
    clutches[0].SetDisabled(True)
    system.SetSolver(chrono.ChSolverSparseLU())
    ground, road = rigid, None
    if terrain != "flat":                          # exact triangle queries on the rough road
        import rough_road
        road = rough_road.RoughRoad(rough_road.mesh(rigid), float(params[0]), rigid)
        ground = road.terrain
    return vehicle, ground, system, wheels, rigid, road


def driver(throttle=0.0, steering=0.0, braking=0.0):
    d = veh.DriverInputs()
    d.m_throttle, d.m_steering, d.m_braking = float(throttle), float(steering), float(braking)
    return d


def advance(vehicle, ground, system, inputs, sample=None, scale=None):
    """One control interval: two 1-ms steps. scale = (bodies, factor) multiplies the body velocities after each step."""
    for _ in range(SUBSTEPS):
        t = system.GetChTime()
        ground.Synchronize(t)
        vehicle.Synchronize(t, inputs, ground)
        ground.Advance(DT)
        vehicle.Advance(DT)
        if scale is not None:
            for b in scale[0]:
                b.SetPos_dt(b.GetPos_dt() * scale[1])
                b.SetWvel_loc(b.GetWvel_loc() * scale[1])
        if sample is not None:
            sample()


def settle(vehicle, system, rigid, road):
    """Bring the vehicle to rest: chassis held in x, y and yaw, neutral, zero inputs, low road friction,
    free for 0.4 s, then velocity scaling after every step, 2.0 s in total. The hold is removed at the end."""
    s = SIM["settle"]
    vehicle.GetPowertrain().SetDriveMode(veh.ChPowertrain.DriveMode_NEUTRAL)
    chassis = vehicle.GetChassisBody()
    ground_body = list(rigid.GetPatches())[0].GetGroundBody()
    pin = chrono.ChLinkMateGeneric(True, True, False, False, False, True)
    pin.Initialize(chassis, ground_body, chrono.ChFrameD(chassis.GetFrame_REF_to_abs().GetPos(), chrono.ChQuaternionD(1, 0, 0, 0)))
    system.AddLink(pin)
    if road is None:                               # the same terrain with the low friction
        twin = veh.FlatTerrain(0.0, s["road_friction"])
    else:
        import rough_road
        twin_road = rough_road.RoughRoad(road.mesh, s["road_friction"], rigid)
        twin = twin_road.terrain
    moving = [b for b in system.Get_bodylist() if not b.GetBodyFixed()]
    for k in range(s["control_steps"]):
        inputs = driver(s["throttle"], s["steering"], s["braking"])
        scale = None if k < s["free_control_steps"] else (moving, s["velocity_factor_after_each_step"])
        advance(vehicle, twin, system, inputs, scale=scale)
    system.RemoveLink(pin)


# ------------------------------------------------------------------ joint residual (validity)
def _vec(v):
    return np.array([v.x, v.y, v.z], float)


def _quat(q):
    return np.array([q.e0, q.e1, q.e2, q.e3], float)


def joint_anchors(system):
    """Local anchor points of every joint. The joint residual is evaluated from them and the body poses."""
    pitman = json.loads((_data_root() / STEERING["Input File"]).read_text(encoding="utf-8"))
    frame = chrono.ChFrameD(chrono.ChVectorD(*STEERING["Location"]), chrono.ChQuaternionD(*STEERING["Orientation"]))
    frame.ConcatenatePreTransformation(chrono.ChFrameD(chrono.ChVectorD(0, 0, SIM["initial_chassis_height_m"]), chrono.ChQuaternionD(1, 0, 0, 0)))
    specs = []
    for item in system.Get_linklist():
        link = chrono.CastToChLink(item)
        name, n = link.GetNameString(), link.GetConstraintViolation().Size()
        if not n:
            continue
        b1, b2 = chrono.CastToChBody(link.GetBody1()), chrono.CastToChBody(link.GetBody2())
        s = {"body1": b1.GetNameString(), "body2": b2.GetNameString()}
        if "distTierod" in name or n == 1:
            d = chrono.CastToChLinkDistance(link)
            s.update(kind="distance", p1=_vec(d.GetEndPoint1Rel()), p2=_vec(d.GetEndPoint2Rel()), length=d.GetImposedDistance())
        elif name.endswith("_universal"):
            joint = pitman["Universal Joint"]
            u = frame.TransformDirectionLocalToParent(chrono.ChVectorD(*joint["Direction Arm"]))
            v = frame.TransformDirectionLocalToParent(chrono.ChVectorD(*joint["Direction Link"]))
            mat = chrono.ChMatrix33D()
            mat.Set_A_axis(u, v, chrono.ChVectorD(*np.cross(_vec(u), _vec(v))))
            replica = chrono.ChLinkUniversal()
            replica.Initialize(b1, b2, chrono.ChFrameD(frame.TransformPointLocalToParent(chrono.ChVectorD(*joint["Location"])), mat.Get_A_quaternion()))
            s.update(kind="position", p1=_vec(replica.GetFrame1Rel().GetPos()), p2=_vec(replica.GetFrame2Rel().GetPos()))
        elif name.endswith("_revsph"):
            joint = pitman["Revolute-Spherical Joint"]
            a = frame.TransformPointLocalToParent(chrono.ChVectorD(*joint["Location Chassis"]))
            b = frame.TransformPointLocalToParent(chrono.ChVectorD(*joint["Location Link"]))
            direction = frame.TransformDirectionLocalToParent(chrono.ChVectorD(*joint["Direction"]))
            u = _vec(b) - _vec(a)
            v = np.cross(_vec(direction), u)
            v /= np.linalg.norm(v)
            u = np.cross(v, _vec(direction))
            mat = chrono.ChMatrix33D()
            mat.Set_A_axis(chrono.ChVectorD(*u), chrono.ChVectorD(*v), direction)
            replica = chrono.ChLinkRevoluteSpherical()
            replica.Initialize(b1, b2, chrono.ChCoordsysD(a, mat.Get_A_quaternion()), float(np.linalg.norm(_vec(b) - _vec(a))))
            s.update(kind="revsph", p1=_vec(replica.GetPoint1Rel()), p2=_vec(replica.GetPoint2Rel()),
                     axis1=_vec(replica.GetDir1Rel()), length=replica.GetImposedDistance())
        elif "Steering" in name and name.endswith("_revolute"):
            m = chrono.CastToChLinkMotorRotationAngle(link)
            s.update(kind="position", p1=_vec(m.GetFrame1().GetPos()), p2=_vec(m.GetFrame2().GetPos()))
        else:
            m = chrono.CastToChLinkLock(link)
            s.update(kind="position", p1=_vec(m.GetMarker1().GetPos()), p2=_vec(m.GetMarker2().GetPos()))
        specs.append(s)
    return specs


def _matrices(q):
    """Rotation matrices of quaternions (w, x, y, z) without normalization."""
    v, w = q[..., 1:], q[..., 0]
    k = np.zeros(q.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    k[..., 1, 2], k[..., 2, 0], k[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    return (2 * w * w - 1)[..., None, None] * np.eye(3) + 2 * v[..., :, None] * v[..., None, :] + 2 * w[..., None, None] * k


def joint_residual(body_p, body_q, names, specs):
    """Largest position error of all joints (m) over the given samples."""
    rot = _matrices(body_q)
    cols = []
    for s in specs:
        i, j = names.index(s["body1"]), names.index(s["body2"])
        d = (body_p[:, j] + rot[:, j] @ s["p2"]) - (body_p[:, i] + rot[:, i] @ s["p1"])
        if s["kind"] in ("distance", "revsph"):
            cols.append(np.linalg.norm(d, axis=1) - s["length"])
            if s["kind"] == "revsph":
                cols.append(np.sum(d * (rot[:, i] @ s["axis1"]), axis=1))
        else:
            cols.append(np.linalg.norm(d, axis=1))
    return float(np.max(abs(np.column_stack(cols))))


# ------------------------------------------------------------------ recorder
class Recorder:
    """Stores the poses and velocities of all bodies after every 1-ms step; outputs are derived at the end."""

    def __init__(self, vehicle, system, wheels):
        self.vehicle, self.system, self.wheels = vehicle, system, wheels
        self.bodies = list(system.Get_bodylist())
        self.names = [b.GetNameString() for b in self.bodies]
        self.chassis = vehicle.GetChassisBody()
        self.ichassis = self.names.index(self.chassis.GetNameString())
        self.iupright = [self.names.index(n) for n in UPRIGHTS]
        self.ispindle = [self.names.index(w.GetSpindle().GetNameString()) for w in wheels]
        self.rows = []

    def sample(self):
        f = self.chassis.GetFrame_REF_to_abs()
        self.rows.append(dict(
            p=_vec(f.GetPos()), q=_quat(f.GetRot()), speed=self.vehicle.GetSpeed(),
            body_p=np.array([_vec(b.GetPos()) for b in self.bodies]), body_q=np.array([_quat(b.GetRot()) for b in self.bodies]),
            body_v=np.array([_vec(b.GetPos_dt()) for b in self.bodies]), body_w=np.array([_vec(b.GetWvel_par()) for b in self.bodies]),
            wheel_omega=np.array([w.GetState().omega for w in self.wheels]), gyro=_vec(self.chassis.GetWvel_loc())))

    def arrays(self, first_only=False):
        """All samples (or the first) as arrays, with the derived outputs velocity, steering proxy and upright positions."""
        rows = self.rows[:1] if first_only else self.rows
        d = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        if not all(np.isfinite(a).all() for a in d.values()):
            raise InvalidRollout("nonfinite state")
        c = self.ichassis
        cog = Rotation.from_quat(np.roll(d["body_q"][:, c], -1, axis=1))
        d["v"] = d["body_v"][:, c] + np.cross(d["body_w"][:, c], d["p"] - d["body_p"][:, c])
        left, right = (Rotation.from_quat(np.roll(d["body_q"][:, i], -1, axis=1)) for i in self.iupright[:2])
        d["delta"] = .5 * ((cog.inv() * left).as_euler("ZYX")[:, 0] + (cog.inv() * right).as_euler("ZYX")[:, 0])
        d["upright_rel"] = np.stack([cog.inv().apply(d["body_p"][:, i] - d["p"]) for i in self.iupright], axis=1)
        if not all(np.isfinite(d[k]).all() for k in ("v", "delta", "upright_rel")):
            raise InvalidRollout("nonfinite output")
        return d

    def toe_camber(self, d):
        """Largest absolute toe and camber angle of the four wheels (deg) at the first sample."""
        rc = Rotation.from_quat(np.roll(d["q"][0], -1))
        axes = np.array([rc.inv().apply(Rotation.from_quat(np.roll(d["body_q"][0, i], -1)).apply([0., 1., 0.])) for i in self.ispindle])
        axes *= np.where(axes[:, 1:2] < 0, -1, 1)
        toe = np.degrees(np.arctan2(-axes[:, 0], axes[:, 1]))
        camber = np.degrees(np.arctan2(axes[:, 2], np.linalg.norm(axes[:, :2], axis=1)))
        return max(float(np.max(abs(toe))), float(np.max(abs(camber))))


OUTPUT_KEYS = dict(position="p", attitude="q", velocity="v", speed="speed", wheel_angular_speed="wheel_omega",
                   steering_proxy="delta", angular_velocity="gyro", upright_relative_position="upright_rel")


# ------------------------------------------------------------------ one rollout
@contextlib.contextmanager
def quiet():
    """Silence the console output of the Chrono library (file descriptors 1 and 2) during a rollout."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (null, *saved):
            os.close(fd)


def simulate(parameters, geometry, motion):
    """Settle the vehicle and drive one motion (console output of Chrono silenced)."""
    with quiet():
        return _simulate(parameters, geometry, motion)


def _simulate(parameters, geometry, motion):
    """Settle the vehicle and drive one motion.

    Returns {"valid": bool, "reason": str, "outputs": {group: array}} with one output sample per control
    interval (the state after it). Invalid rollouts have no outputs.
    """
    params = np.asarray(parameters, dtype=float)
    vehicle, ground, system, wheels, rigid, road = build(params, geometry, motion["terrain"])
    try:
        rec = Recorder(vehicle, system, wheels)
        anchors = joint_anchors(system)
        settle(vehicle, system, rigid, road)
        vehicle.GetPowertrain().SetDriveMode(veh.ChPowertrain.DriveMode_FORWARD)
        rec.sample()
        lim = SIM["validity"]
        first = rec.arrays(first_only=True)
        if not (joint_residual(first["body_p"], first["body_q"], rec.names, anchors) <= lim["maximum_joint_residual_m"]
                and rec.toe_camber(first) <= lim["maximum_initial_toe_camber_deg"]):
            raise InvalidRollout("initial joint residual or wheel angle")
        for row in motion["control"]:
            advance(vehicle, ground, system, driver(*row), rec.sample)
        d = rec.arrays()
        angles = Rotation.from_quat(np.roll(d["q"][1:], -1, axis=1)).as_euler("ZYX", degrees=True)
        checks = {
            "speed": float(np.max(abs(d["speed"][1:]))) <= lim["maximum_speed_m_per_s"],
            "roll and pitch": float(np.max(abs(angles[:, [1, 2]]))) <= lim["maximum_roll_pitch_deg"],
            "planar travel": float(np.linalg.norm(d["p"][-1, :2] - d["p"][2, :2])) >= lim["minimum_planar_travel_m"],
            "joint residual": joint_residual(d["body_p"], d["body_q"], rec.names, anchors) <= lim["maximum_joint_residual_m"],
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            return {"valid": False, "reason": ", ".join(bad), "outputs": None}
        out = {g: np.ascontiguousarray(d[k][SUBSTEPS::SUBSTEPS]) for g, k in OUTPUT_KEYS.items()}
        return {"valid": True, "reason": "", "outputs": out}
    except InvalidRollout as error:
        return {"valid": False, "reason": str(error), "outputs": None}
    finally:
        del vehicle, ground, system, wheels, rigid, road
        gc.collect()
