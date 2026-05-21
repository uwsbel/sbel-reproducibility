---
name: go2_quadruped
description: Create and control a Unitree Go2 quadruped robot from URDF with RL locomotion policy (model_3000.pt), including collision visualization, head collision, contact logging, and complete simulation loop.
compatibility: pychrono >= 8.0
metadata:
  domain: robot
---
# Skill: Unitree Go2 Quadruped (URDF + RL Policy)

## Purpose

Load a Unitree Go2 robot dog from its URDF description, configure collision and contact materials (feet, trunk, head), optionally overlay blue debug collision proxies, and drive it with a pre-trained reinforcement-learning locomotion policy (`model_3000.pt`).

## When to Use

When the user asks to simulate a quadruped robot, robot dog, Go2, or Unitree walking robot.

## Required Data Files

All files live under `data/robot/go2/` relative to the repository root:

| File                           | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| `urdf/go2_description.urdf`    | URDF model (references `../meshes/*.obj`)              |
| `meshes/*.obj`, `meshes/*.mtl` | Visual mesh geometry                                   |
| `policy/model_3000.pt`         | Pre-trained RL policy weights                          |
| `policy/cfgs_rigid.pkl`        | Training configuration (obs/action dims, hidden sizes) |

## System Requirements

Go2's foot contacts and PD-style motors are tuned for smooth contact — **always use `ChSystemSMC`**:

```python
system = chrono.ChSystemSMC()
system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
system.GetSolver().AsIterative().SetMaxIterations(60)
chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)
```

The scene-wide SMC contact material should match what the robot uses internally:

```python
mat = chrono.ChContactMaterialSMC()
mat.SetFriction(0.9)
mat.SetRestitution(0.01)
mat.SetGn(60.0)
mat.SetKn(2e5)
```

## Ground Plane

The RL policy was trained with a specific ground setup. **Always** use `ChBodyEasyBox` at z=0:

```python
ground = chrono.ChBodyEasyBox(10, 10, 0.1, 1000, True, True, mat)
ground.SetName("ground")
ground.SetPos(chrono.ChVector3d(0, 0, 0.0))
ground.SetFixed(True)
ground.GetVisualShape(0).SetTexture(
    chrono.GetChronoDataFile("textures/concrete.jpg"), 10, 10)
system.AddBody(ground)
```

Note: ground top surface is at z=+0.05. Any furniture/walls placed in the scene should account for this.

## Complete Go2Robot Class

```python
import pychrono.parsers as parsers

GO2_ROBOT_DIR = REPO_ROOT / "data" / "robot" / "go2"

class Go2Robot:
    """Wraps a Unitree Go2 URDF with helpers used by the RL policy."""

    def __init__(self, chsystem,
                 initial_state=chrono.ChFramed(chrono.ChVector3d(0, 0, 0.5), chrono.QuatFromAngleZ(0.0)),
                 actuation_type=parsers.ChParserURDF.ActuationType_POSITION,
                 vis_engine: str = 'vsg'):
        self.chsystem = chsystem
        # IMPORTANT: the dedicated go2_vsgvis URDF / meshes have an axis/export
        # mismatch — links render detached / mis-rotated under VSG. Always use
        # the go2 URDF (from data/robot/go2/) for all visualization backends.
        urdf_filename = str(GO2_ROBOT_DIR / "urdf" / "go2_description.urdf")

        self.foot_bodies_list = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.calf_bodies_list = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        self.thigh_bodies_list = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]

        self.robot = parsers.ChParserURDF(urdf_filename)
        self.robot.SetRootInitPose(initial_state)
        self.robot.SetAllJointsActuationType(actuation_type)
        for bodyname in self.foot_bodies_list:
            self.robot.SetBodyMeshCollisionType(
                bodyname, parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)
        # WARNING: do NOT call SetBodyMeshCollisionType("base", CONVEX_HULL).
        # The URDF already declares a tight box collision for "base"
        # (size 0.376 x 0.094 x 0.114). Computing a hull from trunk.obj
        # engulfs the legs and makes the dog T-pose / sprawl on spawn.
        self.robot.PopulateSystem(self.chsystem)
        self.robot.GetRootChBody().SetFixed(False)
        self._set_collision()
        self._add_collision_visualization()

        self.motor_name_list = [
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        ]
        self.motor_list = [self.robot.GetChMotor(n) for n in self.motor_name_list]

    def _set_collision(self):
        foot_mat = chrono.ChContactMaterialSMC()
        foot_mat.SetRestitution(0.01)
        foot_mat.SetFriction(0.9)
        foot_mat.SetGn(60.0)
        # Feet: collide with the world.
        for bodyname in self.foot_bodies_list:
            footbody = self.robot.GetChBody(bodyname)
            footbody.EnableCollision(True)
            footbody.GetCollisionModel().SetAllShapesMaterial(foot_mat)
        # Trunk: also collide so the body bumps into walls / furniture.
        trunk = self.robot.GetChBody("base")
        trunk.EnableCollision(True)
        if trunk.GetCollisionModel() is not None:
            trunk.GetCollisionModel().SetAllShapesMaterial(foot_mat)
        # Head_upper / Head_lower are fixed-jointed to base and declare their
        # own collision in the URDF (cylinder + sphere). Explicitly enable them
        # and assign the same contact material.
        for head_name in ("Head_upper", "Head_lower"):
            head_body = self.robot.GetChBody(head_name)
            if head_body is None:
                continue
            head_body.EnableCollision(True)
            if head_body.GetCollisionModel() is not None:
                head_body.GetCollisionModel().SetAllShapesMaterial(foot_mat)
        # Calf / thigh: disabled to avoid self-collision artifacts during gait.
        for bodyname in self.calf_bodies_list + self.thigh_bodies_list:
            body = self.robot.GetChBody(bodyname)
            body.EnableCollision(False)

    def _add_collision_visualization(self):
        """Overlay blue semi-transparent proxies that mirror the active
        collision geometry for visual debugging."""
        def make_blue_material():
            m = chrono.ChVisualMaterial()
            m.SetDiffuseColor(chrono.ChColor(0.0, 0.35, 1.0))
            m.SetOpacity(0.5)
            return m

        # Trunk: URDF box 0.3762 x 0.0935 x 0.114
        trunk_box = chrono.ChVisualShapeBox(0.3762, 0.0935, 0.114)
        trunk_box.AddMaterial(make_blue_material())
        self.robot.GetChBody("base").AddVisualShape(trunk_box, chrono.ChFramed())

        # Feet: ~22 mm radius sphere approximation of convex hull
        for foot_name in self.foot_bodies_list:
            foot_sphere = chrono.ChVisualShapeSphere(0.022)
            foot_sphere.AddMaterial(make_blue_material())
            self.robot.GetChBody(foot_name).AddVisualShape(foot_sphere, chrono.ChFramed())

        # Head_upper: URDF cylinder radius=0.05, length=0.09
        head_upper = self.robot.GetChBody("Head_upper")
        if head_upper is not None:
            cyl = chrono.ChVisualShapeCylinder(0.05, 0.09)
            cyl.AddMaterial(make_blue_material())
            head_upper.AddVisualShape(cyl, chrono.ChFramed())

        # Head_lower: URDF sphere radius=0.047
        head_lower = self.robot.GetChBody("Head_lower")
        if head_lower is not None:
            sph = chrono.ChVisualShapeSphere(0.047)
            sph.AddMaterial(make_blue_material())
            head_lower.AddVisualShape(sph, chrono.ChFramed())

    def get_base_body(self):
        return self.robot.GetChBody("base")

    def get_base_pos(self):
        return self.robot.GetChBody("base").GetPos()

    def get_base_quat(self):
        return self.robot.GetChBody("base").GetRot()

    def get_base_angvel_local(self):
        return self.robot.GetChBody("base").GetAngVelLocal()

    def get_joint_pos(self) -> np.ndarray:
        out = []
        for motor_name in self.motor_name_list:
            motor = self.robot.GetChMotor(motor_name)
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            out.append(rotation_motor.GetMotorAngle())
        return np.array(out, dtype=np.float32)

    def get_joint_speed(self) -> np.ndarray:
        out = []
        for motor_name in self.motor_name_list:
            motor = self.robot.GetChMotor(motor_name)
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            out.append(rotation_motor.GetMotorAngleDt())
        return np.array(out, dtype=np.float32)

    def actuate(self, motor_angles: np.ndarray):
        for i, motor in enumerate(self.motor_list):
            motor.SetMotorFunction(chrono.ChFunctionConst(float(motor_angles[i])))

    def print_motor_list(self):
        print(f"There are {len(self.motor_name_list)} motors: {self.motor_name_list}")
```

## RL Policy Controller

The policy needs PyTorch. It loads `model_3000.pt` and `cfgs_rigid.pkl`:

```python
import torch, torch.nn as nn, pickle

class PolicyActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hd in hidden_dims:
            layers += [nn.Linear(in_dim, hd), nn.ELU()]
            in_dim = hd
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class PolicyController:
    """Loads model_3000.pt and produces target joint angles for Go2Robot."""

    def __init__(self, data_dir: str):
        cfg_path = os.path.join(data_dir, "cfgs_rigid.pkl")
        with open(cfg_path, "rb") as f:
            self.env_cfg, self.train_cfg = pickle.load(f)

        checkpoint_path = os.path.join(data_dir, "model_3000.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        policy_cfg = self.train_cfg["policy"]
        self.actor = PolicyActor(45, 12, policy_cfg["actor_hidden_dims"])
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
        self.actor.actor.load_state_dict(actor_state)
        self.actor.eval()

        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05

        self.genesis_defaults = np.array([
            0.0, 0.8, -1.5,   # RR
            0.0, 0.8, -1.5,   # RL
            0.0, 1.0, -1.5,   # FR
            0.0, 1.0, -1.5,   # FL
        ], dtype=np.float32)
        self.chrono_to_genesis_mapping = np.array([6,7,8,9,10,11,0,1,2,3,4,5], dtype=np.int64)
        self.genesis_to_chrono_mapping = np.array([6,7,8,9,10,11,0,1,2,3,4,5], dtype=np.int64)
        self.command = np.array([0.5, 0.0, 0.0], dtype=np.float32) * self.lin_vel_scale
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.checkpoint_path = checkpoint_path

    def _projected_gravity(self, quat) -> np.ndarray:
        qw, qx, qy, qz = quat.e0, quat.e1, quat.e2, quat.e3
        r33 = 1 - 2 * (qx*qx + qy*qy)
        r13 = 2 * (qx*qz - qw*qy)
        r23 = 2 * (qy*qz + qw*qx)
        return np.array([-r13, -r23, -r33], dtype=np.float32)

    def build_observation(self, robot) -> torch.Tensor:
        base_ang_vel = robot.get_base_angvel_local()
        base_quat = robot.get_base_quat()
        joint_pos = robot.get_joint_pos()
        joint_vel = robot.get_joint_speed()

        dof_pos_genesis = -joint_pos[self.chrono_to_genesis_mapping]
        dof_vel_genesis = -joint_vel[self.chrono_to_genesis_mapping]

        obs = np.concatenate([
            np.array([base_ang_vel.x, base_ang_vel.y, base_ang_vel.z], dtype=np.float32) * self.ang_vel_scale,
            self._projected_gravity(base_quat),
            self.command,
            (dof_pos_genesis - self.genesis_defaults) * self.dof_pos_scale,
            dof_vel_genesis * self.dof_vel_scale,
            self.last_actions,
        ]).astype(np.float32)

        return torch.from_numpy(obs).unsqueeze(0)

    def compute_action(self, robot) -> np.ndarray:
        obs = self.build_observation(robot)
        with torch.no_grad():
            policy_action = self.actor(obs).squeeze(0).cpu().numpy().astype(np.float32)
        self.last_actions = policy_action

        target_angles_genesis = policy_action * 0.25 + self.genesis_defaults
        target_angles_chrono = -target_angles_genesis[self.genesis_to_chrono_mapping]
        return target_angles_chrono.astype(np.float64)
```

## Spawning and Initialization

```python
policy = PolicyController(str(GO2_ROBOT_DIR / "policy"))

go2_init = chrono.ChFramed(
    chrono.ChVector3d(0.0, -3.0, 0.7),
    chrono.QuatFromAngleZ(np.pi / 2)   # heading +Y
)
go2 = Go2Robot(system, initial_state=go2_init, vis_engine='vsg')
go2.print_motor_list()
```

Spawn height Z >= 0.5 to avoid ground interpenetration. Typical: Z=0.7.

### When a scene is present (`plan_type = mbs_in_scene`)

Do **not** hardcode the spawn XY in that case. Resolve it through the same `FootprintRegistry` the scene uses — otherwise the robot may spawn inside a chair/table/prop, SMC contact constraints will pin it, and the VLM will report "robot stationary / undisturbed" in what looks like an RL policy bug but is actually a placement collision.

```python
# After placing all scene assets through `placement`:
ROBOT_FOOTPRINT_X, ROBOT_FOOTPRINT_Y = 0.6, 0.4   # Go2 trunk+legs AABB, rounded up
spawn_x, spawn_y = placement.place(
    size_x=ROBOT_FOOTPRINT_X,
    size_y=ROBOT_FOOTPRINT_Y,
    preferred_x=0.0,       # plan-requested robot origin
    preferred_y=0.0,
)
go2_init = chrono.ChFramed(
    chrono.ChVector3d(spawn_x, spawn_y, 0.7),
    chrono.QuatFromAngleZ(np.pi / 2),
)
go2 = Go2Robot(system, initial_state=go2_init, vis_engine='vsg')
```

Use `placement.place(size_x, size_y, ...)` with explicit dimensions, not `placement.place_body(go2.get_base_body(), ...)`: Go2 is articulated, so only the `base` body would move — the legs would be left behind, tearing the kinematic chain. See `scene/custom_assets_scene_convex_decomp` §"Placing a robot/vehicle through the registry" for the full canonical pattern.

## Simulation Loop Integration

Prefer `run_recording_loop` (from `chrono_agent.utils`) for generated
simulations — it owns render throttling and recorder cleanup. When you
pass a `step_fn=`, the loop's default `sys.DoStepDynamics(time_step)`
call is REPLACED by your callback, so your step_fn **must end with
`system.DoStepDynamics(timestep)`** itself. Without it, `GetChTime()`
stays at 0, the duration exit never fires, and the robot looks stuck
in a frozen scene even though motors were actuated.

```python
from chrono_agent.utils import run_recording_loop

timestep = 1e-3
control_step_size = 1.0 / 50  # 50 Hz policy/control update rate
control_steps = int(round(control_step_size / timestep))
simulation_duration = 20.0
stand_action = np.array(
    [0.0, -1.0, 1.5, 0.0, -1.0, 1.5, 0.0, -0.8, 1.5, 0.0, -0.8, 1.5],
    dtype=np.float64,
)

def robot_step(step_index, sim_time):
    # Physics runs at 1 kHz, but the RL policy must be updated at 50 Hz.
    # Do not call policy.compute_action every physics step; that produces
    # high-frequency target jitter and can make Go2 shake in place.
    if step_index % control_steps == 0:
        if sim_time < 0.5:
            go2.actuate(stand_action)
        else:
            go2.actuate(policy.compute_action(go2))
    system.DoStepDynamics(timestep)   # REQUIRED — step_fn owns the advance

run_recording_loop(
    system, duration=simulation_duration, time_step=timestep,
    vis=vis, manager=manager, render_fps=50.0, step_fn=robot_step,
)
```

Hand-rolled loop (only when you need bespoke control — otherwise use
`run_recording_loop`):

```python
timestep = 1e-3               # physics step (stable for SMC + robot)
control_step_size = 1.0 / 50  # 50 Hz control
control_steps = int(round(control_step_size / timestep))
render_steps = control_steps
simulation_duration = 20.0

# Standing pose applied for first 0.5s so the robot settles on its feet
stand_action = np.array(
    [0.0, -1.0, 1.5, 0.0, -1.0, 1.5, 0.0, -0.8, 1.5, 0.0, -0.8, 1.5],
    dtype=np.float64,
)
go2.actuate(stand_action)

step_count = 0
start_time = time.time()

while vis.Run():
    sim_time = system.GetChTime()

    # 50 Hz control: stand for 0.5s, then RL policy
    if step_count % control_steps == 0:
        if sim_time < 0.5:
            go2.actuate(stand_action)
        else:
            go2.actuate(policy.compute_action(go2))

    if step_count % render_steps == 0:
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        manager.Update()  # sensor cameras

    system.DoStepDynamics(timestep)
    step_count += 1

    if sim_time >= simulation_duration:
        break
```

## Joint-Connectivity Logging (REQUIRED for review-mode runs)

The deterministic ``no_interpenetration`` predicate that the step
reviewer runs does AABB-overlap checks on every dynamic body pair —
which on a quadruped means EVERY adjacent link pair (trunk ↔ FL_hip,
FL_hip ↔ FL_thigh, FL_thigh ↔ FL_calf, FL_calf ↔ FL_foot, ...) would
falsely flag as "clipping" because revolute joints make their AABBs
share volume by design.

To suppress those false positives, write the joint connectivity to
``scene_links.csv`` at sim end:

```python
from chrono_agent.utils.scene_assets import write_links_csv

# After the sim loop exits — same place as scene_placement.csv
write_links_csv(system, output_dir=str(out_dir))
```

The util iterates ``system.GetLinks()``, calls ``GetBody1()`` /
``GetBody2()`` on each ``ChLink*``, and writes
``body1, body2, link_type``. The validator reads it and skips those
pairs from the AABB-overlap check, leaving only TRUE cross-asset
overlap (e.g. robot trunk clipping a wall, foot intruding into a
table) on the report.

This is REQUIRED any time codegen produces a ``simulation.py`` that
is reviewed by ``step_review_node`` — it costs effectively nothing
at runtime.

## Contact Logging (Optional)

Track robot-environment collisions over time for analysis:

```python
def _body_name_from_contact_obj(contact_obj):
    body = chrono.CastToChBody(contact_obj)
    if body:
        return body.GetName() or "<unnamed>"
    return "<non-body>"

class CollisionLogger(chrono.ReportContactCallback):
    def __init__(self):
        super().__init__()
        self.collision_log = []

    def OnReportContact(self, pA, pB, plane_coord, distance, eff_radius,
                        react_forces, react_torques, contactobjA, contactobjB, constraint_offset):
        # MUST swallow all exceptions and return True. Any Python exception
        # raised inside OnReportContact is re-raised by SWIG as a
        # RuntimeError("SWIG director method error"), which aborts
        # ReportAllContacts and crashes the whole simulation.
        try:
            bodyA_name = _body_name_from_contact_obj(contactobjA)
            bodyB_name = _body_name_from_contact_obj(contactobjB)
            force_mag = react_forces.Length() if hasattr(react_forces, 'Length') else 0.0
            self.collision_log.append({
                "body1": bodyA_name, "body2": bodyB_name,
                "force": force_mag, "time": system.GetChTime()
            })
        except Exception:
            pass
        return True

collision_logger = CollisionLogger()

# Inside simulation loop, after DoStepDynamics. ALWAYS wrap in try/except —
# SWIG director errors can still surface and must not abort the script.
try:
    system.GetContactContainer().ReportAllContacts(collision_logger)
except RuntimeError:
    pass
```

## Common Errors

- **Using ChSystemNSC**: Go2 SMC contact materials won't work — always use `ChSystemSMC`.
- **Calling `SetBodyMeshCollisionType("base", CONVEX_HULL)`**: Creates an oversized hull that engulfs legs; the URDF box collision for base is correct.
- **Wrong joint mapping**: The RL policy uses Genesis joint order (FR/FL first), Chrono URDF uses RR/RL first. Forgetting the mapping produces chaotic motion.
- **Step size too large**: Use <= 1e-3 s for stable SMC contacts with the robot.
- **Updating the RL policy every physics step**: The physics timestep is 1 ms, but the Go2 policy/control update must stay at 50 Hz (`control_steps = 20` for `timestep = 1e-3`). Calling `policy.compute_action(go2)` at 1000 Hz causes high-frequency target changes and commonly appears as the robot shaking in place instead of walking.
- **Spawning too low**: Z < 0.4 causes immediate ground penetration and instability.
- **Missing `torch`/`pickle` imports**: The RL policy requires PyTorch.
- **Using go2_vsgvis URDF**: Has axis/export mismatch causing detached/mis-rotated links under VSG. Always use the `data/robot/go2/` URDF.
- **Forgetting `SetDefaultSuggestedEnvelope/Margin`**: Omitting these can cause contact detection issues with the small foot geometry.
- **Not enabling Head collision**: `Head_upper` and `Head_lower` have URDF-declared collision shapes but may not be enabled by default; explicitly call `EnableCollision(True)`.
- **Wrong ground construction**: Using raw `ChBody()` without mass/inertia hurts SMC effective-mass calculations. Always use `ChBodyEasyBox`.
- **`step_fn` without `DoStepDynamics`**: When passing `step_fn=` to `run_recording_loop`, the default `sys.DoStepDynamics(time_step)` is REPLACED by your callback. A step_fn that only calls `go2.actuate(...)` leaves `GetChTime()` pinned at 0: the duration exit never fires, vis renders a frozen scene, and the robot appears stuck despite motors being actuated. Fix: `step_fn` must end with `system.DoStepDynamics(timestep)`.

## Skill Dependencies

- `../../mbs/system_create/` — ChSystemSMC creation, collision system
- `../../mbs/collision/` — Contact materials (SMC)
- `../../mbs/simulation_loop/` — Time stepping

## API Contract

allowed_classes:
- chrono.ChSystemSMC
- chrono.ChContactMaterialSMC
- chrono.ChBodyEasyBox
- chrono.ChVisualMaterial
- chrono.ChVisualShapeBox
- chrono.ChVisualShapeSphere
- chrono.ChVisualShapeCylinder
- chrono.ChFunctionConst
- chrono.ChFramed
- chrono.ChVector3d
- chrono.ChColor
- chrono.ReportContactCallback
- parsers.ChParserURDF

allowed_methods:
- chrono.ChSystemSMC()
- system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
- system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
- system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
- system.GetSolver().AsIterative().SetMaxIterations(60)
- system.AddBody(ground)
- system.DoStepDynamics(timestep)
- system.GetChTime()
- system.GetContactContainer().ReportAllContacts(collision_logger)
- chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
- chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)
- chrono.ChContactMaterialSMC()
- mat.SetFriction(0.9)
- mat.SetRestitution(0.01)
- mat.SetGn(60.0)
- mat.SetKn(2e5)
- chrono.ChBodyEasyBox(10, 10, 0.1, 1000, True, True, mat)
- ground.SetName("ground")
- ground.SetPos(chrono.ChVector3d(0, 0, 0.0))
- ground.SetFixed(True)
- ground.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/concrete.jpg"), 10, 10)
- parsers.ChParserURDF(urdf_filename)
- self.robot.SetRootInitPose(initial_state)
- self.robot.SetAllJointsActuationType(actuation_type)
- self.robot.SetBodyMeshCollisionType(bodyname, parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)
- self.robot.PopulateSystem(self.chsystem)
- self.robot.GetRootChBody().SetFixed(False)
- self.robot.GetChBody(bodyname)
- self.robot.GetChMotor(motor_name)
- footbody.EnableCollision(True)
- footbody.GetCollisionModel().SetAllShapesMaterial(foot_mat)
- body.EnableCollision(False)
- body.GetPos()
- body.GetRot()
- body.GetAngVelLocal()
- body.AddVisualShape(shape, chrono.ChFramed())
- chrono.CastToChLinkMotorRotation(motor)
- rotation_motor.GetMotorAngle()
- rotation_motor.GetMotorAngleDt()
- motor.SetMotorFunction(chrono.ChFunctionConst(float(value)))
- chrono.CastToChBody(contact_obj)
- body.GetName()
- react_forces.Length()
- chrono.ChVisualMaterial()
- m.SetDiffuseColor(chrono.ChColor(0.0, 0.35, 1.0))
- m.SetOpacity(0.5)
- shape.AddMaterial(material)
- chrono.ChFramed(chrono.ChVector3d(x, y, z), chrono.QuatFromAngleZ(angle))
- chrono.QuatFromAngleZ(angle)
- vis.Run()
- vis.BeginScene()
- vis.Render()
- vis.EndScene()

allowed_constants:
- chrono.ChCollisionSystem.Type_BULLET
- chrono.ChSolver.Type_BARZILAIBORWEIN
- parsers.ChParserURDF.ActuationType_POSITION
- parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL

allowed_utils:
- from chrono_agent.utils import setup_preview_camera
