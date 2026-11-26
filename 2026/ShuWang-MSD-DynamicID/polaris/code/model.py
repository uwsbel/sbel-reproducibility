"""Forward model of the Polaris RZR in Project Chrono (old ChPowertrain interface of the Chrono version in ENVIRONMENT.md).

Sim() builds the vehicle from a geometry and the seven parameters. settle() gives the start state of one
parameter vector. simulate() drives one motion from that state and returns the eight output groups.
Paths come from config.py; all other inputs come from the package folders.
"""
import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pychrono as chrono
import pychrono.vehicle as veh
from scipy.spatial.transform import Rotation

import config

CASE_DIR = Path(__file__).resolve().parents[1]
NAMES = ("mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz")
KEYS = ("p", "q", "v", "speed", "wheel_omega", "delta", "gyro", "upright_rel")

STEP = 0.001             # integration step, s
SUBSTEPS = 2             # integration steps per input sample (input and output interval 0.002 s)
SETTLE_SAMPLES = 1000    # input samples of the settle (2 s)
GAMMA = 0.99             # factor on every body velocity after each integration step of the settle
TWIN_FRICTION = 0.001    # friction of the settle terrain; it has no effect, the in-plane tire force is removed
START_HEIGHT = 0.55      # chassis reference height at construction, m
BRAKE_TORQUE = 2000      # maximum torque of the simple brake, N m

# Height-map roads: generating wave, patch size and height range (m).
ROADS = {
    "asymmetric_roll_development": dict(fx=2.0, fy=1.0, phase=0.18, length=18.0, width=12.0, minimum=-0.08, maximum=0.08),
    "asymmetric_roll_confirmation": dict(fx=1.72, fy=0.92, phase=0.64, length=18.0, width=12.0, minimum=-0.072, maximum=0.072),
}
ROAD_OF_MOTION = {"P-R02": "asymmetric_roll_development", "P-R02_confirmation": "asymmetric_roll_confirmation"}
UPRIGHTS = ("Polaris DoubleWishbone_upright_L", "Polaris DoubleWishbone_upright_R",
            "Polaris ThreeLinkIRS_arm_L", "Polaris ThreeLinkIRS_arm_R")


def quiet(fn):
    """Run fn with the console output of the Chrono library silenced (some builds print file and solver messages)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        try:
            return fn(*args, **kwargs)
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in (null, *saved):
                os.close(fd)
    return wrapper


def read_json(path):
    """Read a JSON file; '//' comments (allowed in Chrono files) are removed first."""
    return json.loads(re.sub(r"//[^\n\r]*", "", Path(path).read_text(encoding="utf-8")))


def bounds():
    """Center and half range of the seven parameters (value = center + half_range * z)."""
    rows = read_json(CASE_DIR / "settings" / "bounds.json")["parameters"]
    return np.array([r["center"] for r in rows]), np.array([r["half_range"] for r in rows])


def feasible(params):
    """Positive friction, mass and inertias, and the triangle inequalities of the principal inertias."""
    p = np.asarray(params, dtype=float)
    return bool(np.isfinite(p).all() and p[0] > 0 and p[1] > 0 and np.min(p[4:]) > 0 and np.min(p[4:].sum() - 2 * p[4:]) >= 0)


def motion_inputs(name):
    """Input rows (throttle, steering, braking) and road of a motion of settings/motions.json."""
    motions = read_json(CASE_DIR / "settings" / "motions.json")
    rows = [m for key in ("library", "confirmation", "validation", "test") for m in motions[key]]
    row = next(m for m in rows if m["name"] == name)
    with np.load(CASE_DIR / row["input_file"], allow_pickle=False) as f:
        return np.array(f["input"]), ROAD_OF_MOTION.get(name)


# ----------------------------------------------------------------------------------------------- vehicle files
def load_geometry(name):
    """Suspension geometry: 'calibrated' (geometry/calibrated.json) or 'true' (the public Chrono files)."""
    stock = vehicle_data() / "Polaris"
    front = read_json(stock / "Polaris_Front_DoubleWishbone.json")
    rear = read_json(stock / "Polaris_Rear_TrailingArm.json")
    inertia = []
    if name == "calibrated":
        cal = read_json(CASE_DIR / "geometry" / "calibrated.json")
        front = cal["front"]
        for part, points in cal["rear"].items():
            rear[part].update(points)
        inertia = read_json(CASE_DIR / "settings" / "front_suspension_inertia.json")["bodies"]
    else:
        assert name == "true", name
        # Curves as lists of floats.
        front["Spring"]["Spring Curve Data"] = [[float(d), float(f)] for d, f in front["Spring"]["Spring Curve Data"]]
        front["Shock"]["Damping Curve Data"] = [[float(v), float(f)] for v, f in front["Shock"]["Damping Curve Data"]]
    rear["Spring"]["Spring Coefficient"] = float(rear["Spring"]["Spring Coefficient"])
    rear["Shock"]["Damping Coefficient"] = float(rear["Shock"]["Damping Coefficient"])
    return dict(name=name, front=front, rear=rear, inertia=inertia)


def vehicle_data():
    """Vehicle data path of Chrono: a copy of config.CHRONO_DATA plus the generated files of each vehicle.

    One folder per process, so that several processes can share WORK_DIR.
    """
    root = Path(config.WORK_DIR) / "vehicle_data" / f"process_{os.getpid()}"
    (root / "Polaris").mkdir(parents=True, exist_ok=True)
    (root / "generated").mkdir(exist_ok=True)
    for f in (Path(config.CHRONO_DATA) / "Polaris").glob("*.json"):
        _write_once(root / "Polaris" / f.name, f.read_bytes())
    return root


def _write_once(path, data):
    """Write a file if it is not there yet; a name always stands for the same content."""
    if not path.exists():
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


def vehicle_file(params, geometry):
    """Write the vehicle JSON files of one parameter vector and geometry; return the top file relative to the data path.

    The chassis center of mass goes into the chassis file (before the joints are made), the brakes are simple
    brakes, the suspensions are those of the geometry. Mass and inertia are set after construction (build()).
    """
    data = vehicle_data()
    tag = hashlib.sha256(np.asarray(params, dtype=np.float64).tobytes() + geometry["name"].encode()).hexdigest()[:20]
    gen = data / "generated"
    chassis = read_json(data / "Polaris" / "Polaris_Chassis.json")
    chassis["Components"][0]["Centroidal Frame"]["Location"] = [float(params[2]), 0.0, float(params[3])]
    brake = dict(Name="Polaris Simple Brake", Type="Brake", Template="BrakeSimple")
    brake["Maximum Torque"] = BRAKE_TORQUE
    top = read_json(data / "Polaris" / "Polaris.json")
    top["Chassis"]["Input File"] = f"generated/chassis_{tag}.json"
    top["Axles"][0]["Suspension Input File"] = f"generated/front_{tag}.json"
    top["Axles"][1]["Suspension Input File"] = f"generated/rear_{tag}.json"
    for axle in top["Axles"]:
        axle["Left Brake Input File"] = axle["Right Brake Input File"] = "generated/brake_simple.json"
    for name, value in (("chassis", chassis), ("front", geometry["front"]), ("rear", geometry["rear"]), ("vehicle", top)):
        _write_once(gen / f"{name}_{tag}.json", json.dumps(value, indent=2))
    _write_once(gen / "brake_simple.json", json.dumps(brake, indent=2))
    return f"generated/vehicle_{tag}.json"


# ----------------------------------------------------------------------------------------------- uneven road
def road_image(road):
    """128 x 128 gray height map of a road (0 = minimum, 255 = maximum height)."""
    r = ROADS[road]
    x = np.linspace(0.0, 1.0, 128, endpoint=False)[None, :]
    y = np.linspace(-1.0, 1.0, 128, endpoint=False)[:, None]
    field = 0.5 + 0.46 * np.sin(2.0 * np.pi * r["fx"] * x + r["phase"]) * np.sin(np.pi * r["fy"] * y)
    return np.clip(field * 255.0, 0.0, 255.0).astype(np.uint8)


def road_file(road):
    """Write the height map as an 8-bit gray PNG file (standard library only) and return its path."""
    path = vehicle_data() / f"road_{road}.png"
    image = road_image(road)
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, 0, 0, 0, 0)
    _write_once(path, b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


_LIBRARY = []


def _library():
    """Load the compiled terrain query helper (terrain/exact_query.cpp) once."""
    if not _LIBRARY:
        lib = ctypes.CDLL(str(config.TERRAIN_LIBRARY))
        lib.install_exact.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_double]
        lib.install_exact.restype = ctypes.c_void_p
        lib.exact_error.restype = ctypes.c_char_p
        _LIBRARY.append(lib)
    return _LIBRARY[0]


class RoadTerrain:
    """Terrain that answers height, normal and friction from the triangles of the road mesh, in double precision.

    The triangles are those of the Chrono height-map patch. Points outside the mesh use the patch itself.
    The query code is C++ (terrain/exact_query.cpp); this class only hands the mesh over.
    """

    def __init__(self, vertices, faces, friction, rigid):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.rigid = rigid
        self.terrain = veh.ChTerrain()
        handle = _library().install_exact(
            ctypes.c_void_p(int(self.terrain.this)), ctypes.c_void_p(int(rigid.this)),
            self.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(self.vertices),
            self.faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(self.faces), float(friction))
        if not handle:
            raise RuntimeError(_library().exact_error().decode())

    @classmethod
    def from_patch(cls, rigid, friction):
        """Take the triangle mesh of the single patch of a RigidTerrain, in world coordinates."""
        body = list(rigid.GetPatches())[0].GetGroundBody()
        visual = body.GetVisualModel()
        shapes = [(chrono.CastToChTriangleMeshShape(visual.GetShape(i)), visual.GetShapeFrame(i)) for i in range(visual.GetNumShapes())]
        (shape, frame), = [(s, f) for s, f in shapes if s]
        mesh = shape.GetMesh()
        vertices = np.array([_vec(v) for v in mesh.getCoordsVertices()])
        faces = np.array([[f.x, f.y, f.z] for f in mesh.getIndicesVertexes()], dtype=np.int32)
        rot = lambda q: Rotation.from_quat([q.e1, q.e2, q.e3, q.e0])
        vertices = rot(body.GetRot()).apply(rot(frame.GetRot()).apply(vertices) + _vec(frame.GetPos())) + _vec(body.GetPos())
        return cls(vertices, faces, friction, rigid)


# ----------------------------------------------------------------------------------------------- build
def _vec(v):
    return np.array([v.x, v.y, v.z], float)


def _quat(q):
    return np.array([q.e0, q.e1, q.e2, q.e3], float)


def driver_inputs(throttle=0.0, steering=0.0, braking=0.0):
    value = veh.DriverInputs()
    value.m_throttle, value.m_steering, value.m_braking = float(throttle), float(steering), float(braking)
    return value


class Sim:
    """One Chrono system with the vehicle, its wheels, the rigid terrain and the terrain the tires use."""

    def __init__(self, params, geometry, road):
        mu, mass, comx, comz, ixx, iyy, izz = map(float, params)
        rel = vehicle_file(params, geometry)
        veh.SetDataPath(vehicle_data().as_posix().rstrip("/") + "/")
        self.system = chrono.ChSystemSMC()
        self.system.Set_G_acc(chrono.ChVectorD(0.0, 0.0, -9.81))
        self.vehicle = veh.WheeledVehicle(self.system, veh.GetDataFile(rel))
        self.vehicle.Initialize(chrono.ChCoordsysD(chrono.ChVectorD(0.0, 0.0, START_HEIGHT), chrono.ChQuaternionD(1.0, 0.0, 0.0, 0.0)))
        for setter in (self.vehicle.SetChassisVisualizationType, self.vehicle.SetSuspensionVisualizationType,
                       self.vehicle.SetSteeringVisualizationType, self.vehicle.SetWheelVisualizationType,
                       self.vehicle.SetTireVisualizationType):
            setter(veh.VisualizationType_NONE)
        self.chassis = self.vehicle.GetChassisBody()
        self.chassis.SetMass(mass)
        self.chassis.SetInertiaXX(chrono.ChVectorD(ixx, iyy, izz))
        self.vehicle.InitializePowertrain(veh.Gator_SimpleMapPowertrain("Polaris powertrain"))
        self.wheels = [wheel for axle in self.vehicle.GetAxles() for wheel in axle.GetWheels()]
        tire_file = str(Path(config.CHRONO_DATA) / "Polaris" / "Polaris_TMeasyTire_fixed.json")
        for wheel in self.wheels:
            self.vehicle.InitializeTire(veh.ReadTireJSON(tire_file), wheel, veh.VisualizationType_NONE)
        self.system.UseMaterialProperties(True)
        self.system.SetSlipVelocityThreshold(0.05)
        self.rigid = veh.RigidTerrain(self.system)
        material = chrono.ChMaterialSurfaceSMC()
        material.SetFriction(mu)
        material.SetYoungModulus(200000.0)
        material.SetPoissonRatio(0.30)
        material.SetRestitution(0.05)
        material.SetAdhesion(0.0)
        if road is None:
            self.rigid.AddPatch(material, chrono.CSYSNORM, 300.0, 300.0)
        else:
            r = ROADS[road]
            self.rigid.AddPatch(material, chrono.CSYSNORM, str(road_file(road)), r["length"], r["width"], r["minimum"], r["maximum"])
        self.rigid.Initialize()
        # Calibrated geometry: inertia tensors of the six front suspension bodies in their own frames.
        bodies = {b.GetNameString(): b for b in self.system.Get_bodylist()}
        for row in geometry["inertia"]:
            I = np.array(row["inertia_kg_m2"])
            bodies[row["body"]].SetInertiaXX(chrono.ChVectorD(*np.diag(I)))
            bodies[row["body"]].SetInertiaXY(chrono.ChVectorD(I[0, 1], I[0, 2], I[1, 2]))
        self.road = RoadTerrain.from_patch(self.rigid, mu) if road is not None else None
        self.terrain = self.road.terrain if road is not None else self.rigid
        # The three differential clutches carry no torque (modulation 0); they are switched off. Direct sparse solver.
        for item in self.system.Get_otherphysicslist():
            clutch = chrono.CastToChShaftsClutch(item)
            if clutch:
                clutch.SetDisabled(True)
        self.solver = chrono.ChSolverSparseLU()
        self.system.SetSolver(self.solver)

    def advance(self, driver, terrain=None):
        """One integration step of STEP seconds."""
        terrain = terrain or self.terrain
        t = self.system.GetChTime()
        terrain.Synchronize(t)
        self.vehicle.Synchronize(t, driver, terrain)
        terrain.Advance(STEP)
        self.vehicle.Advance(STEP)
        self.system.DoStepDynamics(STEP)


# ----------------------------------------------------------------------------------------------- start state
def pose(system):
    """Reference-frame position and orientation of every moving body, sorted by body name."""
    rows = []
    for b in sorted(system.Get_bodylist(), key=lambda b: b.GetNameString()):
        if not b.GetBodyFixed():
            f = b.GetFrame_REF_to_abs()
            rows.append(dict(body=b.GetNameString(), position_m=_vec(f.GetPos()).tolist(), quaternion_wxyz=_quat(f.GetRot()).tolist()))
    return rows


def apply_pose(system, rows):
    """Move every moving body to the given reference-frame pose with zero velocity."""
    bodies = {b.GetNameString(): b for b in system.Get_bodylist() if not b.GetBodyFixed()}
    assert set(bodies) == {r["body"] for r in rows}
    zero = chrono.ChVectorD(0.0, 0.0, 0.0)
    for r in rows:
        body = bodies[r["body"]]
        ref = body.GetFrame_REF_to_abs()
        ref_rot = Rotation.from_quat(np.roll(_quat(ref.GetRot()), -1))
        ref_to_cog = ref_rot.inv().apply(_vec(body.GetPos()) - _vec(ref.GetPos()))
        cog_rel = ref_rot.inv() * Rotation.from_quat(np.roll(_quat(body.GetRot()), -1))
        target = Rotation.from_quat(np.roll(np.asarray(r["quaternion_wxyz"], dtype=float), -1))
        position = np.asarray(r["position_m"], dtype=float) + target.apply(ref_to_cog)
        body.SetPos(chrono.ChVectorD(*map(float, position)))
        body.SetRot(chrono.ChQuaternionD(*map(float, np.roll((target * cog_rel).as_quat(), 1))))
        body.SetPos_dt(zero)
        body.SetWvel_par(zero)
    system.Update()


def reference_pose(geometry, road):
    """Settled pose at the reference parameter point; the settle of every parameter vector starts from it."""
    poses = read_json(CASE_DIR / "settings" / "start_poses.json")["poses"]
    return poses[geometry["name"]]["flat" if road is None else "uneven"]


def _remove_inplane_tire_force(sim, terrain):
    """Cancel the in-plane part of the tire force that each wheel has just applied to its spindle.

    The force acts at the contact point: lowest point of the wheel disc, projected on the terrain plane.
    The normal force and the tire moments are kept.
    """
    for w in sim.wheels:
        tire = w.GetTire()
        reported_force = tire.ReportTireForce(terrain)  # keep the owner alive while the vector is read
        F = _vec(reported_force.force)
        st = w.GetState()
        c = _vec(st.pos)
        d = _vec(chrono.ChMatrix33D(st.rot).Get_A_Yaxis())
        nh = _vec(terrain.GetNormal(st.pos))
        dir1 = np.cross(d, nh)
        s2 = float(dir1 @ dir1)
        if s2 < 1e-3 or not np.any(F):
            continue
        ptd = c + tire.GetRadius() * np.cross(d, dir1 / np.sqrt(s2))
        n = _vec(terrain.GetNormal(chrono.ChVectorD(float(ptd[0]), float(ptd[1]), float(ptd[2]))))
        h = float(terrain.GetHeight(chrono.ChVectorD(float(ptd[0]), float(ptd[1]), float(ptd[2]) + 5.0)))
        pc = ptd + (h - ptd[2]) * n[2] * n
        inplane = F - float(F @ n) * n
        w.GetSpindle().Accumulate_force(chrono.ChVectorD(*(-inplane).tolist()), chrono.ChVectorD(*pc.tolist()), False)


@quiet
def settle(params, geometry, road=None):
    """Start state of one parameter vector: 2 s settle from the reference pose; returns the body poses.

    Chassis pinned in x, y and yaw; neutral gear and no inputs; in-plane tire forces removed;
    all body velocities multiplied by GAMMA after each step. The tires stand on a twin terrain.
    """
    sim = Sim(params, geometry, road)
    apply_pose(sim.system, reference_pose(geometry, road))
    sim.vehicle.GetPowertrain().SetDriveMode(veh.ChPowertrain.DriveMode_NEUTRAL)
    ground = list(sim.rigid.GetPatches())[0].GetGroundBody()
    frame = sim.chassis.GetFrame_REF_to_abs()
    pin = chrono.ChLinkMateGeneric(True, True, False, False, False, True)
    pin.Initialize(sim.chassis, ground, chrono.ChFrameD(frame.GetPos(), chrono.ChQuaternionD(1, 0, 0, 0)))
    sim.system.AddLink(pin)
    if road is None:
        twin = veh.FlatTerrain(0.0, TWIN_FRICTION)
    else:
        twin_road = RoadTerrain(sim.road.vertices, sim.road.faces, TWIN_FRICTION, sim.rigid)
        twin = twin_road.terrain
    moving = [b for b in sim.system.Get_bodylist() if not b.GetBodyFixed()]
    for _ in range(SUBSTEPS * SETTLE_SAMPLES):
        t = sim.system.GetChTime()
        twin.Synchronize(t)
        sim.vehicle.Synchronize(t, driver_inputs(), twin)
        _remove_inplane_tire_force(sim, twin)
        twin.Advance(STEP)
        sim.vehicle.Advance(STEP)
        sim.system.DoStepDynamics(STEP)
        for b in moving:
            b.SetPos_dt(b.GetPos_dt() * GAMMA)
            b.SetWvel_loc(b.GetWvel_loc() * GAMMA)
    return pose(sim.system)


# ----------------------------------------------------------------------------------------------- motion
@quiet
def simulate(params, geometry, inputs, road=None, start=None):
    """Drive one motion. Returns (outputs, valid).

    outputs holds the eight groups at every input sample (0.002 s, first sample after the first input).
    valid is False if the run breaks the mechanical checks: speed above 35 m/s, roll or pitch above 45 deg,
    planar travel below 0.05 m, a joint violation above 0.001 (checked at the start and at every input sample),
    or a start wheel angle above 5 deg.
    """
    if start is None:
        start = settle(params, geometry, road)
    sim = Sim(params, geometry, road)
    apply_pose(sim.system, start)
    sim.vehicle.GetPowertrain().SetDriveMode(veh.ChPowertrain.DriveMode_FORWARD)
    bodies = {b.GetNameString(): b for b in sim.system.Get_bodylist()}
    uprights = [bodies[n] for n in UPRIGHTS]
    links = [link for link in (chrono.CastToChLink(item) for item in sim.system.Get_linklist()) if link.GetConstraintViolation().Size()]
    rows, violation = [], [_largest_violation(links)]

    def sample():
        f = sim.chassis.GetFrame_REF_to_abs()
        rows.append((_vec(f.GetPos()), _quat(f.GetRot()), sim.vehicle.GetSpeed(), _vec(sim.chassis.GetPos()),
                     _vec(sim.chassis.GetPos_dt()), _vec(sim.chassis.GetWvel_par()), _vec(sim.chassis.GetWvel_loc()),
                     [w.GetState().omega for w in sim.wheels], [_vec(b.GetPos()) for b in uprights],
                     [_quat(b.GetRot()) for b in uprights[:2]]))

    sample()
    start_angle = _start_wheel_angle(sim)
    for throttle, steering, braking in inputs:
        driver = driver_inputs(throttle, steering, braking)
        for _ in range(SUBSTEPS):
            sim.advance(driver)
            sample()
        violation.append(_largest_violation(links))
    p, q, speed, cog, cog_v, cog_w, gyro, omega, body_p, body_q = (np.array(x) for x in zip(*rows))
    rc = Rotation.from_quat(np.roll(q, -1, axis=1))
    fronts = [Rotation.from_quat(np.roll(body_q[:, i], -1, axis=1)) for i in range(2)]
    full = dict(p=p, q=q, v=cog_v + np.cross(cog_w, p - cog), speed=speed, wheel_omega=omega,
                delta=.5 * sum((rc.inv() * r).as_euler("ZYX")[:, 0] for r in fronts), gyro=gyro,
                upright_rel=np.stack([rc.inv().apply(body_p[:, i] - p) for i in range(4)], axis=1))
    out = {k: np.ascontiguousarray(full[k][2::2]) for k in KEYS}
    angles = Rotation.from_quat(np.roll(q[1:], -1, axis=1)).as_euler("ZYX", degrees=True)
    valid = bool(all(np.isfinite(v).all() for v in out.values())
                 and np.max(np.abs(speed[1:])) <= 35 and np.max(np.abs(angles[:, [1, 2]])) <= 45
                 and np.linalg.norm(p[-1, :2] - p[2, :2]) >= .05 and max(violation) <= 1e-3 and start_angle <= 5)
    return out, valid


def _largest_violation(links):
    """Largest absolute constraint violation over all joints, as reported by Chrono (m or rad)."""
    return max(float(np.max(np.abs(link.GetConstraintViolation().GetVect()))) for link in links)


def _start_wheel_angle(sim):
    """Largest toe or camber angle (deg) of the four wheels in the chassis frame."""
    rc = Rotation.from_quat(np.roll(_quat(sim.chassis.GetFrame_REF_to_abs().GetRot()), -1))
    axes = np.array([rc.inv().apply(Rotation.from_quat(np.roll(_quat(w.GetSpindle().GetRot()), -1)).apply([0.0, 1.0, 0.0])) for w in sim.wheels])
    axes *= np.where(axes[:, 1:2] < 0, -1, 1)
    toe = np.degrees(np.arctan2(-axes[:, 0], axes[:, 1]))
    camber = np.degrees(np.arctan2(axes[:, 2], np.linalg.norm(axes[:, :2], axis=1)))
    return float(max(np.max(np.abs(toe)), np.max(np.abs(camber))))
