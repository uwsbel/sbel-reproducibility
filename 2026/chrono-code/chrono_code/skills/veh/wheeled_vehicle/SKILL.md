---
name: wheeled_vehicle
description: Create and configure wheeled vehicles (HMMWV, CityBus, FEDA, custom JSON) with engine, transmission, drive, steering, and tire settings.
compatibility: pychrono >= 8.0
metadata:
  domain: veh
---

# Skill: Wheeled Vehicle Setup

## Purpose

Create PyChrono wheeled vehicles without losing system ownership, vehicle
subsystem updates, FSI coupling, or vehicle-specific VSG rendering.

Use this skill when a scene contains HMMWV, CityBus, FEDA, Polaris, or any
`veh.WheeledVehicle` JSON vehicle.

## Decision Tree

1. Wrapper vehicle on terrain (`HMMWV_Full`, `CityBus`, `FEDA`):
   create the wrapper with no args, initialize it, then get `system` from the
   wrapper.
2. `HMMWV_Reduced`:
   `veh.HMMWV_Reduced(sys)` is the only wrapper variant that takes a system.
3. Custom JSON vehicle in FSI or another shared-system scene:
   use `veh.WheeledVehicle(sysMBS, json_path)`.
4. Custom JSON vehicle as the whole standalone scene:
   use `veh.WheeledVehicle(json_path, chrono.ChContactMethod_SMC)`.

## Hard Rules

- Do not pass a pre-created `ChSystem` to `HMMWV_Full`, `CityBus`, or `FEDA`.
  These wrappers create and own their system internally.
- For wrapper vehicles, call `Initialize()` before `Set*VisualizationType(...)`.
- For wrapper vehicle scenes using `run_recording_loop`, pass a custom
  `step_fn`; the default step only advances the raw `ChSystem`, not driver,
  terrain, tire, and powertrain subsystems.
- For custom JSON vehicles in FSI/shared-system scenes, keep `sysMBS` as the
  first constructor argument. Do not "fix" overload errors by dropping `sysMBS`.
- Do not assert `vehicle.GetSystem() is sysMBS` in Python. Some PyChrono
  builds return a different wrapper/base pointer even when the 2-arg shared
  system constructor is correct. Verify by behavior/body registration instead.
- In FSI scenes, register every wheel spindle with collision geometry and FSI.
- In FSI scenes, drive with scripted inputs or `ChDataDriver`; do not use
  `ChInteractiveDriver` in batch/headless runs.
- In FSI scenes, use exactly one vehicle-aware VSG path: prefer
  `build_fsi_vehicle_visualizer(...)`. Do not also create a generic
  `chronovsg.ChVisualSystemVSG()`.

## Wrapper Vehicles

### HMMWV (Full Model)

```python
hmmwv = veh.HMMWV_Full()
hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
hmmwv.SetChassisCollisionType(veh.CollisionType_NONE)
hmmwv.SetChassisFixed(False)
hmmwv.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
hmmwv.SetEngineType(veh.EngineModelType_SHAFTS)
hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
hmmwv.SetTireType(veh.TireModelType_TMEASY)
hmmwv.SetTireStepSize(step_size)
hmmwv.Initialize()
system = hmmwv.GetSystem()
```

### City Bus

```python
bus = veh.CityBus()
bus.SetContactMethod(chrono.ChContactMethod_SMC)
bus.SetChassisCollisionType(veh.CollisionType_NONE)
bus.SetChassisFixed(False)
bus.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
bus.SetTireType(veh.TireModelType_TMEASY)
bus.SetTireStepSize(step_size)
bus.Initialize()
system = bus.GetSystem()
```

### FEDA

```python
feda = veh.FEDA()
feda.SetContactMethod(chrono.ChContactMethod_SMC)
feda.SetChassisCollisionType(veh.CollisionType_NONE)
feda.SetChassisFixed(False)
feda.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
feda.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
feda.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
feda.SetTireType(veh.TireModelType_PAC02)
feda.SetTireStepSize(step_size)
feda.Initialize()
system = feda.GetSystem()
```

### Reduced HMMWV (Simpler Suspension)

```python
hmmwv = veh.HMMWV_Reduced(sys)
hmmwv.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
hmmwv.SetEngineType(veh.EngineModelType_SIMPLE)
hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
hmmwv.SetDriveType(veh.DrivelineTypeWV_RWD)
hmmwv.SetTireType(veh.TireModelType_RIGID)
hmmwv.Initialize()
```

## Custom JSON Vehicle (Polaris and similar)

For JSON vehicles such as `Polaris/Polaris.json`, do not call wrapper-only
methods such as `SetEngineType`, `SetTransmissionType`, or `SetTireType`.
Construct `veh.WheeledVehicle`, then explicitly read and initialize
powertrain and tire JSON files.

### Constructor — pick the right overload (FSI vs standalone)

```python
# Standalone: vehicle owns a new system. Use only when the vehicle is the
# whole scene.
vehicle = veh.WheeledVehicle(
    veh.GetVehicleDataFile("Polaris/Polaris.json"),
    chrono.ChContactMethod_SMC,
)

# Shared system: required for FSI scenes and scenes with pre-existing sysMBS
# bodies such as tanks, platforms, floating plates, ramps, or props.
vehicle = veh.WheeledVehicle(
    sysMBS,
    veh.GetVehicleDataFile("Polaris/Polaris.json"),
)
```

Valid overloads:
- `WheeledVehicle(str)`
- `WheeledVehicle(str, ChContactMethod[, bool, bool])`
- `WheeledVehicle(ChSystem*, str[, bool, bool])`

There is no `WheeledVehicle(ChSystem*, str, ChContactMethod)` overload. If
SWIG rejects that call, drop the contact-method argument, not `sysMBS`.

### Full Polaris JSON setup (FSI form — the common case)

```python
vehicle = veh.WheeledVehicle(
    sysMBS,
    veh.GetVehicleDataFile("Polaris/Polaris.json"),
)
vehicle.Initialize(chrono.ChCoordsysd(init_pos, chrono.QUNIT))

engine = veh.ReadEngineJSON(
    veh.GetVehicleDataFile("Polaris/Polaris_EngineSimpleMap.json")
)
transmission = veh.ReadTransmissionJSON(
    veh.GetVehicleDataFile("Polaris/Polaris_AutomaticTransmissionSimpleMap.json")
)
powertrain = veh.ChPowertrainAssembly(engine, transmission)
vehicle.InitializePowertrain(powertrain)

tire_json = "Polaris/Polaris_RigidTire.json"  # FSI/liquid scenes
for axle in vehicle.GetAxles():
    for wheel in axle.GetWheels():
        tire = veh.ReadTireJSON(veh.GetVehicleDataFile(tire_json))
        vehicle.InitializeTire(tire, wheel, chrono.VisualizationType_MESH)
```

Polaris files normally used here:
- `Polaris/Polaris.json`
- `Polaris/Polaris_EngineSimpleMap.json`
- `Polaris/Polaris_AutomaticTransmissionSimpleMap.json`
- `Polaris/Polaris_RigidTire.json`
- `Polaris/Polaris_RigidMeshTire.json`
- `Polaris/Polaris_TMeasyTire.json`
- `Polaris/Polaris_Pac02Tire.json`

Do not invent `Polaris_Engine.json`.

## Chassis init height

`vehicle.Initialize(ChCoordsysd(init_pos, q))` takes the chassis-frame
origin in world coordinates. To rest wheel bottoms on a flat support:

```
chassis_init_z = support_top_z + tire_radius - front_spindle_z
```

`tire_radius` and `front_spindle_z` are vehicle-specific; read them
from the shipped JSONs via the project helper rather than hardcoding:

```python
from chrono_code.utils.vehicle_geometry import chassis_init_z
init_z = chassis_init_z(vehicle_json, support_top_z, tire_json=<tire you load>)
init_pos = chrono.ChVector3d(planner_x, planner_y, init_z)
```

X/Y come from the planner's `scene_predicates[]`; only Z is derived
here, because the planner doesn't know per-vehicle suspension geometry.

## Mandatory: assert footprint after Initialize()

The chassis-frame origin is **not** the geometric center for every
vehicle. Polaris puts the front axle at `x=0` and the rear axle at
`x=-2.7153` in chassis frame, so calling
`polaris.Initialize(ChCoordsysd(ChVector3d(-4, 0, z)))` lands the front
axle at world `x=-4` and the rear axle at `x=-6.72` — 0.72 m past the
left edge of a `[-6, -2]` platform. HMMWV / Sedan put the chassis origin
at the geometric center, so the same shift convention does not generalize.
Codegen cannot guess which convention applies without reading the JSON,
so the contract is: every wheeled-vehicle `simulation.py` MUST assert
the world-frame footprint immediately after `Initialize(...)`.

```python
from chrono_code.utils.vehicle_geometry import assert_vehicle_on_support

polaris.Initialize(chrono.ChCoordsysd(init_pos, chrono.QUNIT))
assert_vehicle_on_support(
    polaris,
    VEHICLE_JSON,                       # same path passed to chassis_init_z
    support_x_range=(LEFT_PLT_X - PLT_X / 2, LEFT_PLT_X + PLT_X / 2),
    support_y_range=(LEFT_PLT_Y - PLT_Y / 2, LEFT_PLT_Y + PLT_Y / 2),
    support_top_z=PLT_TOP_Z,
    tire_json=TIRE_JSON,                # same tire passed to chassis_init_z
    support_name="left_platform",
)
```

The helper reads the four axle spindle positions from the vehicle JSON
(via the same paths `chassis_init_z` already uses), transforms the
wheel envelopes through the chassis world pose, and checks four edge
clearances + the wheel-bottom z. It does NOT use `body.GetTotalAABB()`
— that returns ±DBL_MAX for any body without a registered collision
shape, which is the default state of the chassis and spindles
immediately after `Initialize(...)`. Failure messages include the
**suggested shift in meters** so the next codegen iteration can move
`VEH_INIT_X` directly instead of guessing.

If the support is the SCM terrain or RigidTerrain (no flat-platform
bounds), pass `support_x_range=(-1e9, 1e9)` and `support_y_range=(-1e9, 1e9)` —
the helper still validates the wheel-bottom z, which is the part that
generalizes. Skip the assert entirely only for vehicle-only test rigs
that have no support geometry.

## Tire Choice

| Context | Tire choice |
|---|---|
| RigidTerrain road | `RIGID` or `TMEASY` |
| SCM deformable terrain | `TMEASY`, `PAC02`, or `FIALA`; also call `SetTireStepSize(step_size)` |
| CRM/granular terrain | `TMEASY` or `PAC02` |
| FSI liquid/floating-body scenes | JSON `Polaris_RigidTire.json` or `Polaris_RigidMeshTire.json` |

Do not use wrapper default rigid tires on SCM if the vehicle must actually
drive; wheels spin with little chassis translation. For SCM with non-rigid
tire force models, see `veh/terrain` for explicit spindle/tire collision
cylinders.

## Visualization Settings

Wrapper visualization type calls go after `Initialize()`:

```python
veh_obj.Initialize()
veh_obj.SetChassisVisualizationType(chrono.VisualizationType_MESH)
veh_obj.SetSuspensionVisualizationType(chrono.VisualizationType_PRIMITIVES)
veh_obj.SetSteeringVisualizationType(chrono.VisualizationType_PRIMITIVES)
veh_obj.SetWheelVisualizationType(chrono.VisualizationType_MESH)
veh_obj.SetTireVisualizationType(chrono.VisualizationType_MESH)
```

## Vehicle-Specific VSG Visualization

For normal vehicle scenes, use `veh.ChWheeledVehicleVisualSystemVSG`, not
generic `chronovsg.ChVisualSystemVSG`.

```python
vis = veh.ChWheeledVehicleVisualSystemVSG()
vis.SetWindowTitle("Vehicle Scene")
vis.SetWindowSize(1280, 1024)
vis.EnableSkyTexture()
vis.SetLightIntensity(1.0)
vis.SetLightDirection(2.0, 0.75)
vis.EnableShadows()
vis.SetChaseCamera(chrono.ChVector3d(0, 0, 1.75), 9.0, 0.5)
vis.AttachVehicle(hmmwv.GetVehicle())
vis.AttachTerrain(terrain)
vis.AttachDriver(driver)
vis.Initialize()
```

Do not pass an empty sky texture path to `SetSkyDomeTexture`; it can throw
`vsg::Exception` in `Initialize()`.

## Simulation Loop — Synchronize + Advance Order

Wrapper vehicle scenes need the full vehicle subsystem loop:

```python
driver_inputs = driver.GetInputs()

driver.Synchronize(time)
terrain.Synchronize(time)
hmmwv.Synchronize(time, driver_inputs, terrain)
vis.Synchronize(time, driver_inputs)

driver.Advance(step_size)
terrain.Advance(step_size)
hmmwv.Advance(step_size)
vis.Advance(step_size)
```

With `run_recording_loop`, put that block in `step_fn=...`. Do not also call
`system.DoStepDynamics(step_size)` when `hmmwv.Advance(step_size)` is already
advancing the wrapper-owned system.

## FSI Coupling — Wheel Spindle Registration

When a vehicle interacts with SPH fluid or a floating FSI bridge, each wheel
spindle must be registered as an FSI body. Use the shared-system constructor,
initialize powertrain/tires, then attach tire collision mesh and FSI geometry
to every spindle.

### Vehicle spawn position (drive-across-tank scenarios)

Use the resolved planner `scene_predicates[].position` for the vehicle spawn.
Do not copy tutorial constants such as `-bxDim / 2 - bxDim * CH_1_3` unless
the scene uses exactly that tutorial tank/platform geometry. After initialize,
log the chassis AABB and check the vehicle footprint is supported by the
platform/ramp/bridge.

### Required pattern (every wheel spindle becomes an FSI body)

```python
mesh_filename = veh.GetVehicleDataFile("Polaris/meshes/Polaris_tire_collision.obj")

cmaterial = chrono.ChContactMaterialSMC()
cmaterial.SetYoungModulus(1e8)
cmaterial.SetFriction(0.9)
cmaterial.SetRestitution(0.4)

# Build trimesh / collision shape / ChBodyGeometry FRESH per spindle.
# DO NOT hoist any of these three out of the loop and share across wheels.
# `spindle.AddCollisionShape(shape)` registers the C++ shape's owning-body
# pointer; the second `AddCollisionShape` on the SAME shape object overwrites
# the first owner and the contact handler dereferences a stale pointer at
# the first DoStepDynamics call → SIGSEGV in ChContactContainerSMC::AddContact.
# Same hazard for `ChBodyGeometry`: AddFsiBody stores a pointer into it.
for axle in vehicle.GetAxles():
    for wheel in axle.GetWheels():
        spindle = wheel.GetSpindle()

        trimesh = chrono.ChTriangleMeshConnected()
        trimesh = trimesh.CreateFromWavefrontFile(mesh_filename, False, True)
        trimesh.RepairDuplicateVertices(1e-9)
        wheel_shape = chrono.ChCollisionShapeTriangleMesh(
            cmaterial, trimesh, False, False, 0.005
        )
        spindle.AddCollisionShape(wheel_shape)
        spindle.EnableCollision(True)

        geometry = chrono.ChBodyGeometry()
        geometry.coll_meshes.append(
            chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh_filename, chrono.VNULL)
        )
        sysFSI.AddFsiBody(spindle, geometry, False)
```

Do not use `sysSPH.CreatePointsBoxInterior(...)` for wheel spindles. Box BCE
is for non-rotating blocks/plates, not rotating wheels. The spindle needs all
three calls: `AddCollisionShape`, `EnableCollision(True)`, and `AddFsiBody`.

### FSI spindle geometry is NOT the visual tire (HARD)

The `ChCollisionShapeTriangleMesh` + `ChBodyGeometry` you attach to the
spindle above are **collision/BCE coupling** geometry. They participate in
contact + fluid pressure transfer. **They are NOT the visible tire mesh.**

Visual tires come from a separate channel — the vehicle's own tire JSON
loaded by `InitializeTire(...)`:

```python
# Visual tires — driven by the vehicle's own tire mesh assets.
for axle in vehicle.GetAxles():
    for wheel in axle.GetWheels():
        tire = veh.ReadTireJSON(veh.GetVehicleDataFile(tire_json))
        vehicle.InitializeTire(tire, wheel, chrono.VisualizationType_MESH)

# Plus the vehicle-aware visualizer that renders chassis + tires together:
vehicle.SetWheelVisualizationType(chrono.VisualizationType_MESH)
vehicle.SetTireVisualizationType(chrono.VisualizationType_MESH)
```

**Common confusion to avoid:**

- ❌ Treating the spindle's `wheel_shape` as the visible tire — it is a
  collision proxy used by chrono's contact solver and FSI marker
  generator. Hiding/showing it does not change what the user sees.
- ❌ Skipping `InitializeTire(..., VisualizationType_MESH)` because
  "spindles already have a mesh attached" — that mesh is the collision
  proxy, not the tire visualization. Without `InitializeTire(...)` the
  vehicle visualizer renders bare spindles and the user sees no tires.
- ❌ Using the spindle collision mesh asset as the tire visualization
  — the collision mesh is intentionally simplified for performance and
  looks crude as a render asset.

The two channels are independent:
- **FSI / collision** path: spindle.AddCollisionShape + spindle.AddFsiBody
  (this section's required pattern above)
- **Visual** path: InitializeTire(VisualizationType_MESH) +
  SetWheelVisualizationType + SetTireVisualizationType

Always wire BOTH for FSI vehicle scenes.

### API choice for FSI body BCE — pick by body kind

| Body kind | Pattern |
|---|---|
| Container/tank walls | Use `chrono_code.utils.fsi_assets.build_fsi_tank`; do not hand-roll wall BCE. |
| Floating non-rotating plate/block | `sysSPH.CreatePointsBoxInterior(size)` plus `sysFSI.AddFsiBody(body, bce, ChFramed(), False)`. |
| Vehicle wheel spindle | Trimesh `ChBodyGeometry` plus spindle collision shape; see required pattern above. |

### Synchronize signature in FSI scenes

FSI scenes advance with `sysFSI.DoStepDynamics(dT)`. Use two-argument vehicle
synchronization and do not separately step `vehicle.Advance`, `sysMBS`, or a
terrain object:

```python
vehicle.Synchronize(sim_time, driver_inputs)
if vis:
    vis.Synchronize(sim_time, driver_inputs)
sysFSI.DoStepDynamics(dT)
if vis:
    vis.Advance(dT)
```

### Driver — pre-programmed only, NO human-in-the-loop

For FSI batch/headless runs, use direct `veh.DriverInputs()` or
`veh.ChDataDriver`. Do not use `ChInteractiveDriver`.

```python
driver_inputs = veh.DriverInputs()

if sim_time < 0.4:
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_braking = 1.0
else:
    driver_inputs.m_throttle = 0.45
    driver_inputs.m_braking = 0.0
driver_inputs.m_steering = 0.0
```

### Visualization in FSI scenes

Prefer the helper from `chrono_code.utils.fsi_assets`; it wires the vehicle,
shared MBS system, and SPH plugin into one vehicle-aware VSG visualizer.

```python
from chrono_code.utils.fsi_assets import build_fsi_vehicle_visualizer

visFSI = fsi.ChSphVisualizationVSG(sysFSI)
visFSI.SetSPHColorCallback(fsi.ParticleVelocityColorCallback(0, 5.0))

vis = build_fsi_vehicle_visualizer(
    sysMBS,
    sysSPH,
    sysFSI,
    vehicle=vehicle,
    sph_visualization=visFSI,
    window_title="Vehicle on FSI",
)
```

Do not create a second generic `chronovsg.ChVisualSystemVSG()` in the same FSI
vehicle scene.

## Chassis collision

For wrapper chassis collision against scene props, prefer
`chrono_code.utils.add_collision_via_subbodies` with `CollisionType_NONE`.
Do not combine wrapper `CollisionType_MESH`/`PRIMITIVES` with sub-body weld
collision. After post-initialize collision shape edits, call
`system.GetCollisionSystem().BindAll()` once.

## Joint-Connectivity Logging (REQUIRED for review-mode runs)

For reviewed vehicle scenes, write joint connectivity so overlap validators do
not treat suspension and wheel joints as interpenetration bugs:

```python
from chrono_code.utils.scene_assets import write_links_csv

write_links_csv(system, output_dir=str(out_dir))
```

## Skill Dependencies

- `veh/terrain` for terrain creation and SCM tire collision cylinders.
- `veh/driver` for driver classes and schedules.
- `fsi/sph` for FSI system setup and complete SPH examples.
- `vsg` and `core/mbs_in_scene` for non-FSI visualization/recording.

## API Contract

Allowed constructors:
- `veh.HMMWV_Full()`
- `veh.CityBus()`
- `veh.FEDA()`
- `veh.HMMWV_Reduced(sys)`
- `veh.WheeledVehicle(json_path, chrono.ChContactMethod_SMC)` for standalone
- `veh.WheeledVehicle(sysMBS, json_path)` for shared-system/FSI

Forbidden constructs:
- `veh.HMMWV_Full(sys)`
- `veh.CityBus(sys)`
- `veh.FEDA(sys)`
- `veh.WheeledVehicle(sysMBS, json_path, chrono.ChContactMethod_SMC)`
- `assert vehicle.GetSystem() is sysMBS`
- generic `chronovsg.ChVisualSystemVSG()` alongside an FSI vehicle visualizer

Common constants and methods:
- `chrono.ChContactMethod_SMC`
- `chrono.VisualizationType_MESH`
- `chrono.VisualizationType_PRIMITIVES`
- `veh.EngineModelType_SHAFTS`
- `veh.EngineModelType_SIMPLE`
- `veh.EngineModelType_SIMPLE_MAP`
- `veh.TransmissionModelType_AUTOMATIC_SHAFTS`
- `veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP`
- `veh.DrivelineTypeWV_FWD`
- `veh.DrivelineTypeWV_RWD`
- `veh.DrivelineTypeWV_AWD`
- `veh.SteeringTypeWV_PITMAN_ARM`
- `veh.TireModelType_RIGID`
- `veh.TireModelType_RIGID_MESH`
- `veh.TireModelType_TMEASY`
- `veh.TireModelType_PAC02`
- `veh.TireModelType_FIALA`
- `veh.CollisionType_NONE`
- `veh.CollisionType_PRIMITIVES`
- `veh.CollisionType_MESH`
