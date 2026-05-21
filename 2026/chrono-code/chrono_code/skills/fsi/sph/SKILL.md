---
name: sph
description: >
  Set up SPH-based Fluid-Structure Interaction (FSI): create ChFsiFluidSystemSPH
  and ChFsiSystemSPH, configure fluid and SPH parameters, seed fluid particles with
  hydrostatic initialization, add container BCE boundary markers, register floating
  rigid bodies, optionally couple with a wheeled vehicle, and advance with
  sysFSI.DoStepDynamics(dT).
compatibility: pychrono >= 8.0
metadata:
  domain: fsi
---

## Numeric examples below are TEMPLATES, not defaults

Every concrete number in the canonical examples (e.g. `plate_size = 0.9 * fxDim`,
`4 * initial_spacing`, `bzDim + plate_size.z * 0.5`) is a **formula between
named variables** that show how the dimensions relate to one another. They are
NOT default sizes you may copy into a plan when the user did not specify a
size. Doing so produces silently-correct-looking values like 0.12 m thickness
or 2.0 m length that the user never asked for — this is the iter_001 bug
(`session_20260501_131758` had 4 plain-string "Confirm X (currently 0.12m)"
clarifications all derived from this skill's examples).

If a size or dimension is needed and the user did not specify it, raise a
structured clarification per `GEOMETRY_RELATION_RULES` and wait for the user's
choice. Substitute the user's value into the formula; never substitute the
example's incidental number.

## API Contract

```yaml
allowed_classes:
  - fsi.ChFsiFluidSystemSPH
  - fsi.ChFsiSystemSPH
  - fsi.FluidProperties
  - fsi.SPHParameters
  - fsi.ChSphVisualizationVSG
  - fsi.ParticleVelocityColorCallback
  - fsi.DepthPressurePropertiesCallback
  - fsi.IntegrationScheme_RK2
  - fsi.BoundaryMethod_ADAMI
  - fsi.ViscosityMethod_ARTIFICIAL_UNILATERAL
  - fsi.ShiftingMethod_PPST
  - fsi.ShiftingMethod_XSPH
  - fsi.EosType_TAIT
  - fsi.BC_Y_PERIODIC
  - chrono.ChGridSamplerd
  - chrono.ChAABB

allowed_methods:
  - sysFSI.SetStepSizeCFD(dt)
  - sysFSI.SetStepsizeMBD(dt)
  - sysFSI.SetGravitationalAcceleration(vec)
  - sysFSI.GetStepSizeCFD()
  - sysFSI.AddFsiBody(body, bce, frame, is_flex)
  - sysFSI.Initialize()
  - sysFSI.DoStepDynamics(dT)
  - sysSPH.SetCfdSPH(fluid_props)
  - sysSPH.SetSPHParameters(sph_params)
  - sysSPH.SetComputationalDomain(ChAABB, boundary_condition)
  - sysSPH.GetDensity()
  - sysSPH.GetViscosity()
  - sysSPH.GetSoundSpeed()
  - sysSPH.GetGravitationalAcceleration()
  - sysSPH.GetParticlePositions()
  - sysSPH.AddSPHParticle(pos, rho, p, viscosity)
  - sysSPH.CreatePointsBoxContainer(size_vec, bce_layers)
  - sysSPH.CreatePointsBoxInterior(size_vec)
  - sysSPH.SaveParticleData(path)
  - sysSPH.SaveSolidData(path, time)
  - sysSPH.GetSphIntegrationSchemeString()
  - sampler.SampleBox(center, half_dims)
  - visFSI.EnableFluidMarkers(bool)
  - visFSI.EnableBoundaryMarkers(bool)
  - visFSI.EnableRigidBodyMarkers(bool)
  - visFSI.EnableFlexBodyMarkers(bool)
  - visFSI.SetSPHColorCallback(callback)
  - visVSG.AttachPlugin(visFSI)

canonical_examples:
  - "System stack: sysSPH = fsi.ChFsiFluidSystemSPH(); sysFSI = fsi.ChFsiSystemSPH(sysMBS, sysSPH)"
  - "Container: build = chrono_code.utils.fsi_assets.build_fsi_tank(sysMBS, sysSPH, sysFSI, world_extent=ChAABB(...), contact_material=mat, bce_layers=ChVector3i(2,0,-1))"
  - "Fluid seeding: points = chrono.ChGridSamplerd(d0).SampleBox(build.sampler_box_center, build.sampler_box_halfdim)"
  - "Floating body: build = chrono_code.utils.fsi_assets.build_fsi_floating_box(sysMBS, sysSPH, sysFSI, world_center=ChVector3d(...), size=ChVector3d(...), density=400, contact_material=mat)"
  - "Pre-Initialize guard: chrono_code.utils.fsi_assets.assert_fsi_bodies_unique(sysMBS) — call right before sysFSI.Initialize() to catch double-AddBody on build_fsi_* bodies (silent BCE desync → fluid leaks through walls)"
  - "Step loop: def fsi_step(i, t): sysFSI.DoStepDynamics(step_size); run_recording_loop(sysMBS, ..., step_fn=fsi_step)"
```

## Purpose & When to Use

Build and run an SPH Fluid-Structure Interaction simulation using
`ChFsiFluidSystemSPH` (SPH solver) and `ChFsiSystemSPH` (FSI coupling).
Vehicle setup and `ChSystemSMC` creation are deferred to
`veh/wheeled_vehicle` and `mbs/system_create` respectively.

**Use this skill when** the plan involves a pool, channel, tank, dam-break,
sloshing, floating object, vehicle fording, or any open-surface fluid
exerting hydrodynamic force on rigid bodies.

**Do NOT use** for granular terrain under a vehicle chassis (use
`veh/terrain` / `CRMTerrain`) or for plans with no fluid component at all.

## Anatomy of an FSI Script

Every FSI script — vehicle or non-vehicle — follows the same three-section
shape with a fixed-string barrier comment between Section 2 and Section 3.
**Per-step codegen for FSI plans pattern-matches on this barrier** to know
where to insert new bodies; do not deviate from the section markers.

```python
# === SECTION 1: SYSTEMS ===
# sysMBS, sysSPH, sysFSI; step sizes; gravity; fluid props; SPH params;
# computational domain. See Pattern A.

# === SECTION 2: BODIES + FSI REGISTRATIONS ===
# Tank (build_fsi_tank), floating bodies (build_fsi_floating_box),
# SPH particles (sampler.SampleBox + AddSPHParticle), and — if vehicle
# present — vehicle.Initialize() + per-spindle sysFSI.AddFsiBody().
# Per-step codegen MUST insert new content above the barrier below.

# === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===
assert_fsi_bodies_unique(sysMBS)   # catches double-AddBody on build_fsi_* bodies
sysFSI.Initialize()

# === SECTION 3: VISUALIZATION + RUN LOOP ===
# visFSI plugin; vis (helper or hand-rolled); fsi_step;
# run_recording_loop(...). When the step declares motion_expectations,
# also write cam/motion_log.csv from the on_step callback (see codegen
# system prompt rule 6); otherwise nothing more is required.
# See Pattern F (vehicle scene) or Pattern G (non-vehicle scene).
```

`sysFSI.Initialize()` finalizes BCE neighbour lists, ghost-particle layers,
and FSI coupling state. Any `AddSPHParticle`, `AddFsiBody`, or SPH
parameter call AFTER it is silently ignored — the symptom is the
iteration_008 chassis-missing bug from `session_20260428_164422` (12
wasted iterations because vehicle code landed below the barrier).

## Pattern A — System Stack

The FSI stack always consists of three objects. Create + configure in this
order before any particle seeding or body registration.

```python
import pychrono.core as chrono
import pychrono.fsi as fsi

# 1. MBS — ChSystemSMC required (FSI coupling needs SMC contact model)
sysMBS = chrono.ChSystemSMC()
sysMBS.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
sysMBS.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

# 2. SPH fluid + 3. FSI coupling
sysSPH = fsi.ChFsiFluidSystemSPH()
sysFSI = fsi.ChFsiSystemSPH(sysMBS, sysSPH)

# 4. Step sizes + gravity (HR-3: same vector on both systems)
step_size = 1e-4
sysFSI.SetStepSizeCFD(step_size)
sysFSI.SetStepsizeMBD(step_size)
sysFSI.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

# 5. Fluid properties + SPH parameters (must come BEFORE AddSPHParticle —
# GetDensity/GetSoundSpeed/GetViscosity return values derived from these)
initial_spacing = 0.1
fluid_props = fsi.FluidProperties()
fluid_props.density   = 1000.0
fluid_props.viscosity = 5.0
sysSPH.SetCfdSPH(fluid_props)

sph_params = fsi.SPHParameters()
sph_params.integration_scheme         = fsi.IntegrationScheme_RK2
sph_params.initial_spacing            = initial_spacing
sph_params.d0_multiplier              = 1.2
sph_params.max_velocity               = 10.0
sph_params.shifting_method            = fsi.ShiftingMethod_PPST
sph_params.shifting_ppst_push         = 3.0
sph_params.shifting_ppst_pull         = 1.0
sph_params.artificial_viscosity       = 0.03
sph_params.boundary_method            = fsi.BoundaryMethod_ADAMI
sph_params.viscosity_method           = fsi.ViscosityMethod_ARTIFICIAL_UNILATERAL
sph_params.use_delta_sph              = True
sph_params.delta_sph_coefficient      = 0.1
sph_params.eos_type                   = fsi.EosType_TAIT
sph_params.num_proximity_search_steps = 1
sysSPH.SetSPHParameters(sph_params)

# 6. Computational domain (HR-5: must enclose every BCE marker including
# vehicle trajectory)
cMin = chrono.ChVector3d(-bxDim/2 - 5*initial_spacing,
                         -byDim/2 - initial_spacing/2,
                         -5*initial_spacing)
cMax = chrono.ChVector3d(+bxDim/2 + 5*initial_spacing,
                         +byDim/2 + initial_spacing/2,
                          bzDim   + 5*initial_spacing)
sysSPH.SetComputationalDomain(chrono.ChAABB(cMin, cMax), fsi.BC_Y_PERIODIC)
```

## Pattern B — Fluid Particle Seeding

Use `ChGridSamplerd` to generate a uniform lattice, then add each point as
an SPH particle with hydrostatically correct initial pressure and density.
Subtract `initial_spacing` from each fluid half-dim to avoid overlap with
BCE boundary layers (causes startup pressure spikes).

```python
boxCenter  = chrono.ChVector3d(cx, cy, fzDim / 2)
boxHalfDim = chrono.ChVector3d(fxDim/2 - initial_spacing,
                               fyDim/2,
                               fzDim/2 - initial_spacing)

sampler = chrono.ChGridSamplerd(initial_spacing)
points  = sampler.SampleBox(boxCenter, boxHalfDim)

gz = abs(sysSPH.GetGravitationalAcceleration().z)   # 9.81 m/s²
for pt in points:
    depth   = fzDim - pt.z
    pre_ini = sysSPH.GetDensity() * gz * depth
    rho_ini = sysSPH.GetDensity() + pre_ini / (sysSPH.GetSoundSpeed() ** 2)
    sysSPH.AddSPHParticle(pt, rho_ini, pre_ini, sysSPH.GetViscosity())
```

If you used `build_fsi_tank` (Pattern C) and the fluid fills to the tank rim,
you may use its returned `build.sampler_box_center` /
`build.sampler_box_halfdim` directly. If the free surface is below the rim
(the default), use the helper's X/Y center and inset but derive the sampler
Z from `WATER_SURFACE_Z`:

```python
fluid_center = chrono.ChVector3d(
    tank_build.sampler_box_center.x,
    tank_build.sampler_box_center.y,
    (TANK_FLOOR_Z + WATER_SURFACE_Z) / 2,
)
fluid_halfdim = chrono.ChVector3d(
    tank_build.sampler_box_halfdim.x,
    tank_build.sampler_box_halfdim.y,
    max(0.0, (WATER_SURFACE_Z - TANK_FLOOR_Z) / 2 - initial_spacing),
)
```

## Pattern C — Fluid Container (`build_fsi_tank`)

A fluid container has three layers — visual walls, BCE markers, fluid
sampler — that **must occupy the same world AABB**. Hand-rolling them is
the single most common FSI bug (mixed body-local frame conventions →
double-shifted BCE relative to walls).

Container geometry convention lives in `mbs/body_creation` under
"Generated Boundary Bodies": the tank/channel pose is xy-centered with
`pose.position.z` at the floor, not at the geometric center.

Default free surface rule for open SPH containers:

```text
CONTAINER_FLOOR_Z = pose.position.z
CONTAINER_RIM_Z   = pose.position.z + size.z
FREE_SURFACE_CLEARANCE = 0.2
WATER_SURFACE_Z = CONTAINER_RIM_Z - FREE_SURFACE_CLEARANCE
```

Do not make the fluid height equal to the container height by default. Use
`FREE_SURFACE_CLEARANCE = 0.0` only when the user explicitly asks for a
flush-to-rim fill. For a 1.0 m tall tank with floor at z=0, the default
free surface is z=0.8.

**Always use `chrono_code.utils.fsi_assets.build_fsi_tank`.** It takes
the tank's interior extent in WORLD coordinates and builds the body, the
per-wall visual + collision shapes, the BCE cluster, and a recommended
fluid sampler — all aligned by construction. There is no body-local
frame to get wrong.

```python
from chrono_code.utils.fsi_assets import build_fsi_tank

# WORLD coords of the tank's interior boundary. The SPH free surface is
# normally lower than this rim by FREE_SURFACE_CLEARANCE.
FREE_SURFACE_CLEARANCE = 0.2
WATER_SURFACE_Z = fzDim - FREE_SURFACE_CLEARANCE
TANK_WORLD = chrono.ChAABB(
    chrono.ChVector3d(-fxDim/2, -fyDim/2, 0.0),
    chrono.ChVector3d(+fxDim/2, +fyDim/2, fzDim),
)

build = build_fsi_tank(
    sysMBS, sysSPH, sysFSI,
    world_extent=TANK_WORLD,
    contact_material=cmaterial,
    bce_layers=chrono.ChVector3i(2, 0, -1),   # see "ChVector3i layers" below
    name="water_tank_boundary",
)
tank_body = build.tank_body
```

That's the entire container construction. Do **not** write
`tank_body = chrono.ChBody()` followed by per-wall `AddVisualShape /
AddCollisionShape / CreatePointsBoxContainer / AddFsiBody` — every
hand-rolled version we have shipped has had a body-vs-BCE
double-offset bug. Use the helper.

### What `build_fsi_tank` already did — DO NOT redo

The helper internally calls **all** of these on the body it returns:

  * `sysMBS.AddBody(tank)` — body is in the system
  * `tank.SetPos(center)` / `tank.SetFixed(True)` / `tank.EnableCollision(True)` — pose, anchor, collision configured
  * Visual + collision shapes for each enabled wall — already attached
  * `sysSPH.CreatePointsBoxContainer(...)` + `sysFSI.AddFsiBody(tank, bce_cluster, ChFramed(), False)` — BCE markers bound to **this exact** body reference

The `tank_body = build.tank_body` line above is purely a handle for
later code that needs to reference the body (e.g. plotting its AABB,
attaching extra FSI registrations to it). It is **not** a "now you
register it" cue.

**Forbidden — every line below silently corrupts the FSI registration**
(PyChrono does NOT deduplicate `AddBody`; the second call appends a
duplicate entry to `sysMBS.GetBodies()`, the body's `GetSystem()`
pointer becomes inconsistent, and the BCE markers no longer track the
tank → fluid leaks straight through the walls many seconds into the sim,
which is far away from the cause):

```python
sysMBS.AddBody(build.tank_body)         # FORBIDDEN — helper already added
build.tank_body.SetFixed(True)          # FORBIDDEN — helper already set
build.tank_body.EnableCollision(True)   # FORBIDDEN — helper already enabled
```

This is the iter_007 (session 20260506_193720) failure mode. The
defensive habit comes from generic MBS code where every body needs an
explicit `AddBody` — but `build_fsi_*` helpers are **not** plain shape
builders, they are full-registration helpers.

To make this trap impossible to ship, call the assertion below right
before `sysFSI.Initialize()`:

```python
from chrono_code.utils.fsi_assets import assert_fsi_bodies_unique

# === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===
assert_fsi_bodies_unique(sysMBS)   # raises if any body was double-added
sysFSI.Initialize()
```

It is O(N) over `sysMBS.GetBodies()` and raises a `RuntimeError` naming
the offending body and which `build_fsi_*` helper already registered
it — fail-early at simulation start instead of "fluid leaks at t=1.5s".

### `ChVector3i` layers — picking the right value

`bce_layers` is a 3-axis spec for which faces of the cuboid get BCE
markers. Sign and magnitude both matter:

| value | meaning | typical use |
|-------|---------|-------------|
| `N (≥ 1)` | `N` layers on BOTH ±axis faces | non-periodic walls of an open-top tank |
| `0` | skip axis entirely | periodic axes (`BC_*_PERIODIC`) |
| `-1` | bottom-face layer only; top open | gravity axis of an open-top tank (HR-6) |
| `-N (N≥2)` | bottom N layers; top open | multi-layer floor (rare) |

Canonical open-top tank with `BC_Y_PERIODIC`: `chrono.ChVector3i(2, 0, -1)`.

**Pairing rule (HARD)**: setting `bce_layers` to `0` on an axis means
"no BCE wall on this axis". That ONLY contains fluid if you also set
the matching periodic boundary on the SAME axis via
`sysSPH.SetComputationalDomain(ChAABB(...), fsi.BC_<AXIS>_PERIODIC)`.
Open walls on a non-periodic axis = particles leak straight out and
spray under the platforms. The plan's
`scene_objects[*].boundary_conditions: <axis>_periodic` is the
contract telling codegen which axis is periodic; codegen MUST pass
the matching `BC_*_PERIODIC` enum to `SetComputationalDomain`.

Anti-patterns (all → silent leak or SIGABRT):
- **`ChVector3i(2, 0, -1)` + `SetComputationalDomain(..., BC_NONE)`** →
  silent y-leak: `0` on Y meant "no Y wall, periodic handles it"; once
  you drop `BC_Y_PERIODIC` you MUST raise the Y component to `2`. Pair
  rule: **`BC_NONE` ⇒ `bce_layers=(2, 2, -1)`**, never `(2, 0, -1)`.
  This was the iter_006 second-crash-mode of session_20260506_204433
  after the agent correctly diagnosed "wheel BCE outside periodic
  period" and switched to `BC_NONE` but forgot to add Y walls.
- `ChVector3i(2, 0, 1)`: closed top → pressure-tight box → NaN in `DoStepDynamics`.
- `ChVector3i(2, 0, 0)`: zero floor → particles fall through → out of `cMin.z`.

## Pattern D — Floating Body (`build_fsi_floating_box`)

Same alignment trap as the container. Use
`chrono_code.utils.fsi_assets.build_fsi_floating_box`:

```python
from chrono_code.utils.fsi_assets import build_fsi_floating_box

# Spawn ABOVE the free surface (HR-4) — hydrostatic pressure settles the
# body into equilibrium rather than penetrating un-settled fluid.
plate_size   = chrono.ChVector3d(0.9 * fxDim, 0.7 * fyDim, 4 * initial_spacing)
plate_center = chrono.ChVector3d(0, 0, fzDim + plate_size.z * 0.5)

build = build_fsi_floating_box(
    sysMBS, sysSPH, sysFSI,
    world_center=plate_center,
    size=plate_size,
    density=400,                     # water=1000 → 60% above water
    contact_material=cmaterial,
    fluid_density=fluid_density,
    name="floating_plate",
)
floating_plate = build.body
```

The helper sets mass / inertia from `density × volume`, applies
`CreatePointsBoxInterior(size)` (full extents, not half — common bug),
and registers with identity `ChFramed`. Common mistakes the helper
prevents or makes obvious:

- `CreatePointsBoxContainer` instead of `CreatePointsBoxInterior` — the
  container variant generates a hollow shell, particles tunnel through.
- Half-extents passed to `CreatePointsBoxInterior` — BCE half the size of visual.
- `density ≥ fluid_density` when "floating" was the intent — anything ≥
  1000 sinks; target 0.3–0.5 of fluid density to get visible buoyancy.
- Spawning at `z = fzDim` instead of above it — body starts inside fluid
  before particles settle, gets punched downward by initial pressure spike.

## Pattern E — Vehicle FSI Coupling (deferred)

When the plan also includes a wheeled vehicle, vehicle construction +
spindle FSI registration lives in
[`veh/wheeled_vehicle`](../../veh/wheeled_vehicle/SKILL.md) section "FSI
Coupling — Wheel Spindle Registration". Pattern F below shows how the
vehicle code slots into Section 2 (above the barrier).

The only `AddFsiBody` overload codegen calls directly is the **wheel
spindle geometry overload** (HR-8) — three-arg form with a
`ChBodyGeometry` instead of a point list:

```python
geometry = chrono.ChBodyGeometry()
geometry.coll_meshes.append(
    chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh_filename, chrono.VNULL)
)
sysFSI.AddFsiBody(spindle, geometry, False)   # ← is_flexible always False
```

Container BCE (Pattern C) and floating body BCE (Pattern D) go through the
helpers, not direct `AddFsiBody`.

## Pattern F — Canonical FSI Vehicle Scene

End-to-end full-file shape for an FSI scene with a wheeled vehicle, SPH
water, and VSG-only mp4 recording. Codegen for FSI vehicle plans copies
this template; the section markers and barrier comment are the contract.

```python
import pychrono.core as chrono
import pychrono.fsi as fsi
import pychrono.vehicle as veh
import pychrono.vsg3d as chronovsg

from chrono_code.utils import run_recording_loop
from chrono_code.utils.fsi_assets import (
    build_fsi_tank, build_fsi_floating_box, build_fsi_vehicle_visualizer,
    assert_fsi_bodies_unique,
)
from chrono_code.utils.vsg_recording import (
    setup_vsg_recording, lock_side_camera, hide_vsg_gui,
)
from chrono_code.utils.scene_assets import write_links_csv

# === SECTION 1: SYSTEMS ===
sysMBS = chrono.ChSystemSMC()
sysSPH = fsi.ChFsiFluidSystemSPH()
sysFSI = fsi.ChFsiSystemSPH(sysMBS, sysSPH)
sysFSI.SetStepSizeCFD(step_size); sysFSI.SetStepsizeMBD(step_size)
sysFSI.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
sysSPH.SetCfdSPH(fluid_props)
sysSPH.SetSPHParameters(sph_params)
sysSPH.SetComputationalDomain(chrono.ChAABB(cMin, cMax), fsi.BC_Y_PERIODIC)

# === SECTION 2: BODIES + FSI REGISTRATIONS ===
FREE_SURFACE_CLEARANCE = 0.2
WATER_SURFACE_Z = tank_aabb.max.z - FREE_SURFACE_CLEARANCE
tank_build  = build_fsi_tank(sysMBS, sysSPH, sysFSI, world_extent=tank_aabb,
                             contact_material=cmaterial,
                             bce_layers=chrono.ChVector3i(2, 0, -1),
                             name="water_tank_boundary")
plate_build = build_fsi_floating_box(sysMBS, sysSPH, sysFSI,
                                     world_center=plate_center, size=plate_size,
                                     density=400, contact_material=cmaterial,
                                     name="floating_plate")

# Seed SPH particles up to WATER_SURFACE_Z, not necessarily to the tank rim.
fluid_center = chrono.ChVector3d(
    tank_build.sampler_box_center.x,
    tank_build.sampler_box_center.y,
    (tank_aabb.min.z + WATER_SURFACE_Z) / 2,
)
fluid_halfdim = chrono.ChVector3d(
    tank_build.sampler_box_halfdim.x,
    tank_build.sampler_box_halfdim.y,
    max(0.0, (WATER_SURFACE_Z - tank_aabb.min.z) / 2 - initial_spacing),
)
points = chrono.ChGridSamplerd(initial_spacing).SampleBox(
    fluid_center, fluid_halfdim,
)
for pt in points:
    depth   = WATER_SURFACE_Z - pt.z
    pre_ini = sysSPH.GetDensity() * 9.81 * depth
    rho_ini = sysSPH.GetDensity() + pre_ini / (sysSPH.GetSoundSpeed() ** 2)
    sysSPH.AddSPHParticle(pt, rho_ini, pre_ini, sysSPH.GetViscosity())

# Vehicle + spindle FSI registration. ALL of this MUST be above the barrier
# below — once sysFSI.Initialize() runs, new AddFsiBody calls are silently
# ignored (the iteration_008 chassis-missing bug).
polaris = veh.WheeledVehicle(sysMBS, veh.GetVehicleDataFile("Polaris/Polaris.json"))
polaris.Initialize(chrono.ChCoordsysd(veh_init_pos, chrono.QUNIT))
polaris.InitializePowertrain(powertrain)
for axle in polaris.GetAxles():
    for wheel in axle.GetWheels():
        polaris.InitializeTire(rigid_tire, wheel, chrono.VisualizationType_MESH)
# Spindle FSI registration — full pattern in
# ``veh/wheeled_vehicle`` § "FSI Coupling — Wheel Spindle Registration".
# Build trimesh / collision shape / ChBodyGeometry FRESH per spindle;
# sharing across wheels causes SIGSEGV in
# ChContactContainerSMC::AddContact at the first DoStepDynamics call.
for axle in polaris.GetAxles():
    for wheel in axle.GetWheels():
        spindle = wheel.GetSpindle()

        trimesh = chrono.ChTriangleMeshConnected()
        trimesh = trimesh.CreateFromWavefrontFile(spindle_mesh, False, True)
        trimesh.RepairDuplicateVertices(1e-9)
        wheel_shape = chrono.ChCollisionShapeTriangleMesh(
            cmaterial, trimesh, False, False, 0.005
        )
        spindle.AddCollisionShape(wheel_shape)
        spindle.EnableCollision(True)

        spindle_geometry = chrono.ChBodyGeometry()
        spindle_geometry.coll_meshes.append(
            chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, spindle_mesh, chrono.VNULL)
        )
        sysFSI.AddFsiBody(spindle, spindle_geometry, False)

# === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===
assert_fsi_bodies_unique(sysMBS)   # catches double-AddBody on build_fsi_* bodies
sysFSI.Initialize()

# === SECTION 3: VISUALIZATION + RUN LOOP ===
visFSI = fsi.ChSphVisualizationVSG(sysFSI)
visFSI.SetSPHColorCallback(fsi.ParticleVelocityColorCallback(0, 5.0))

vis = build_fsi_vehicle_visualizer(
    sysMBS, sysSPH, sysFSI,
    vehicle=polaris,
    sph_visualization=visFSI,
    window_title="FSI Polaris Scene",
)

# Camera + recording AFTER vis.Initialize() (already done by helper).
hide_vsg_gui(vis)
lock_side_camera(vis,
    chrono.ChVector3d(0, -7 * fyDim, 3 + bzDim / 2),
    chrono.ChVector3d(0, 0, bzDim / 2),
)
finalize_vsg_mp4 = setup_vsg_recording(vis, "cam/vsg.mp4", fps=50.0)

# step_fn OWNS physics advance — Hard Rule #1. run_recording_loop does
# NOT call DoStepDynamics on its own when step_fn is provided; this
# callback IS the physics step. For FSI it MUST call sysFSI.DoStepDynamics
# (NOT sysMBS — Hard Rule #2). The vehicle.Synchronize call goes inside
# step_fn so driver inputs are consumed every step.
def fsi_step(step_index, sim_time):
    polaris.Synchronize(sim_time, driver.GetInputs())
    sysFSI.DoStepDynamics(step_size)

try:
    run_recording_loop(
        sysMBS, duration=t_end, time_step=step_size,
        vis=vis, manager=None,            # VSG-only
        render_fps=50.0, step_fn=fsi_step,
        recorders=[],
    )
finally:
    finalize_vsg_mp4()                    # encodes BMP dump → mp4 (H.264)

# End-of-sim CSV dumps are NOT required. The only motion-related CSV
# output is `cam/motion_log.csv`, written ONLY when the current step's
# plan declares `motion_expectations` — see the codegen system prompt's
# rule 6 for the canonical on_step pattern.
```

For a fully worked vehicle/bridge/floating-plate version with concrete
numbers, see [`demo/scene/tutorial_VEH_FSI_FloatingBlock.py`](../../../demo/scene/tutorial_VEH_FSI_FloatingBlock.py).

## Pattern G — Canonical Non-Vehicle FSI Scene

Dam-break / sloshing / floating-only scenes use the same skeleton as
Pattern F but skip the vehicle code in Section 2 and use a hand-rolled
`ChVisualSystemVSG` in Section 3 (no `build_fsi_vehicle_visualizer` —
that helper is vehicle-specific).

```python
import pychrono.core as chrono
import pychrono.fsi as fsi
import pychrono.vsg3d as chronovsg

from chrono_code.utils import run_recording_loop
from chrono_code.utils.fsi_assets import build_fsi_tank, build_fsi_floating_box, assert_fsi_bodies_unique
from chrono_code.utils.vsg_recording import (
    setup_vsg_recording, lock_side_camera, hide_vsg_gui,
)

# === SECTION 1: SYSTEMS ===
# (identical to Pattern F — sysMBS / sysSPH / sysFSI + step_size + gravity +
# fluid_props + sph_params + computational domain)

# === SECTION 2: BODIES + FSI REGISTRATIONS ===
FREE_SURFACE_CLEARANCE = 0.2
WATER_SURFACE_Z = tank_aabb.max.z - FREE_SURFACE_CLEARANCE
tank_build  = build_fsi_tank(sysMBS, sysSPH, sysFSI, world_extent=tank_aabb,
                             contact_material=cmaterial,
                             bce_layers=chrono.ChVector3i(2, 0, -1),
                             name="water_tank_boundary")
# Optional: one or more floating bodies.
plate_build = build_fsi_floating_box(sysMBS, sysSPH, sysFSI,
                                     world_center=plate_center, size=plate_size,
                                     density=400, contact_material=cmaterial,
                                     name="floating_plate")

# Seed SPH particles up to WATER_SURFACE_Z, not necessarily to the tank rim.
fluid_center = chrono.ChVector3d(
    tank_build.sampler_box_center.x,
    tank_build.sampler_box_center.y,
    (tank_aabb.min.z + WATER_SURFACE_Z) / 2,
)
fluid_halfdim = chrono.ChVector3d(
    tank_build.sampler_box_halfdim.x,
    tank_build.sampler_box_halfdim.y,
    max(0.0, (WATER_SURFACE_Z - tank_aabb.min.z) / 2 - initial_spacing),
)
points = chrono.ChGridSamplerd(initial_spacing).SampleBox(
    fluid_center, fluid_halfdim,
)
for pt in points:
    depth   = WATER_SURFACE_Z - pt.z
    pre_ini = sysSPH.GetDensity() * 9.81 * depth
    rho_ini = sysSPH.GetDensity() + pre_ini / (sysSPH.GetSoundSpeed() ** 2)
    sysSPH.AddSPHParticle(pt, rho_ini, pre_ini, sysSPH.GetViscosity())

# === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===
assert_fsi_bodies_unique(sysMBS)   # catches double-AddBody on build_fsi_* bodies
sysFSI.Initialize()

# === SECTION 3: VISUALIZATION + RUN LOOP ===
# Generic VSG visualizer (no vehicle), with the same Attach order as the
# helper enforces internally for vehicle scenes:
#   1. visFSI plugin built from sysFSI
#   2. EnableFluidMarkers(True) so the SPH water cloud actually renders
#      (without it the mp4 has no blue particles); EnableBoundaryMarkers
#      and EnableRigidBodyMarkers default OFF — the dense green BCE dot
#      grid on tank walls / floating bodies was perceived by VLM review
#      as the only thing in the scene and caused false-negative reviews
#   3. ChVisualSystemVSG, AttachSystem(sysMBS), AttachPlugin(visFSI)
#   4. Window / camera config
#   5. Initialize() last
visFSI = fsi.ChSphVisualizationVSG(sysFSI)
visFSI.SetSPHColorCallback(fsi.ParticleVelocityColorCallback(0, 5.0))
visFSI.EnableFluidMarkers(True)
visFSI.EnableBoundaryMarkers(False)
visFSI.EnableRigidBodyMarkers(False)
visFSI.EnableFlexBodyMarkers(False)

vis = chronovsg.ChVisualSystemVSG()
vis.AttachSystem(sysMBS)
vis.AttachPlugin(visFSI)
vis.SetWindowSize(1280, 720)
vis.SetWindowTitle("FSI Sloshing Scene")
vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
vis.AddCamera(chrono.ChVector3d(0, -7 * fyDim, 3 + bzDim / 2),
              chrono.ChVector3d(0, 0, bzDim / 2))
vis.Initialize()

hide_vsg_gui(vis)
lock_side_camera(vis,
    chrono.ChVector3d(0, -7 * fyDim, 3 + bzDim / 2),
    chrono.ChVector3d(0, 0, bzDim / 2),
)
finalize_vsg_mp4 = setup_vsg_recording(vis, "cam/vsg.mp4", fps=50.0)

# step_fn OWNS physics advance — same rule as Pattern F. No vehicle.Synchronize
# call needed since there is no vehicle.
def fsi_step(step_index, sim_time):
    sysFSI.DoStepDynamics(step_size)

try:
    run_recording_loop(
        sysMBS, duration=t_end, time_step=step_size,
        vis=vis, manager=None,
        render_fps=50.0, step_fn=fsi_step,
        recorders=[],
    )
finally:
    finalize_vsg_mp4()

# No end-of-sim particles.csv / scene_links.csv requirement. The only
# motion-related CSV is `cam/motion_log.csv`, written ONLY when the
# current step's plan declares `motion_expectations` — see codegen
# system prompt rule 6.
```

### Logging SPH motion to motion_log.csv (use `SphMotionLogger` — do NOT hand-roll)

When the step declares ``motion_expectations`` for the SPH cloud (e.g.
``sph_water peak_velocity > 0.5 m/s`` to verify wave / splash dynamics),
the row written to ``cam/motion_log.csv`` MUST contain a real velocity.

**Do NOT hand-write the row.** PyChrono's Python FSI binding does NOT
expose ``GetParticleVelocities()`` — agents have repeatedly invented this
API or, equivalently, queried only ``GetParticlePositions()`` and padded
the velocity columns with ``0,0,0``. The reviewer then reads
"peak SPH velocity = 0" from the CSV and fails the step for "FSI coupling
broken" even when the VLM video clearly shows the water moving. This
exact failure happened in iter_006 of session_20260506_*.

Use the canonical helper, which finite-differences the SPH center-of-mass
internally:

```python
from chrono_code.utils.fsi_assets import SphMotionLogger

# Construct ONCE before the run loop (it carries internal state).
sph_log = SphMotionLogger(name="sph_water")

LOG_EVERY_N = 50  # ~50 ms at step_size=1e-3 — matches motion_log.csv cadence

step_idx = 0
def fsi_step(sim_time, dt):
    global step_idx
    sysFSI.DoStepDynamics(STEP_SIZE)
    if step_idx % LOG_EVERY_N == 0:
        sph_log.write(motion_csv, sim_time, sysSPH)
    step_idx += 1
```

Rules:

- One ``SphMotionLogger`` per fluid body (e.g. one for ``sph_water``).
- The ``name`` argument MUST match the body name in
  ``motion_expectations``. The reviewer fingerprint-matches on this name.
- Do NOT also write a separate ``sph_water`` row from a hand-rolled
  block — the helper is the only writer for this body name.
- Do NOT replace ``sph_log.write(...)`` with a tighter loop "for performance"
  — it's a single COM pass; the overhead is one ``GetParticlePositions()``
  call you'd be doing anyway.

## Performance Model & Parameter Tuning

SPH is fundamentally an N-body solver. Cost decisions made when picking
`initial_spacing` and `step_size` dominate everything else by orders of
magnitude — they matter more than render frequency, sensor count, vehicle
complexity, or any other knob in the scene.

### The cost equation

```
wall_time     ≈  N_steps × cost_per_step
N_steps       =  sim_duration / step_size
cost_per_step ≈  N_particles × avg_neighbors  +  FSI_coupling_overhead
N_particles   ≈  V_fluid / initial_spacing³   ← cubic in spacing
avg_neighbors ≈  30–60                        (3D, kernel radius ≈ 2*h)
```

The cubic relationship is the most important fact in this entire skill.
Halving `initial_spacing` multiplies particle count by **8**, and (because
the CFL bound on `step_size` also tightens proportionally) typically
multiplies wall time by **~16**.

### Particle count reference table

| `initial_spacing` | Particles per m³ | 8 m³ tank (4×2×1) | 1 m³ pool | 0.1 m³ glass |
| ----------------- | ---------------- | ----------------- | --------- | ------------ |
| 0.20 m            | 125              | 1,000             | 125       | 12           |
| 0.10 m (default)  | 1,000            | 8,000             | 1,000     | 100          |
| 0.05 m            | 8,000            | **64,000**        | 8,000     | 800          |
| 0.025 m           | 64,000           | 512,000           | 64,000    | 6,400        |
| 0.01 m            | 1,000,000        | 8,000,000         | 1,000,000 | 100,000      |

Anything past ~100k particles in an 8 m³ tank with vehicle coupling will be
painful (multi-minute wall time per simulated second on a 4070-class GPU).
1M+ is research-grade and needs a justification beyond visual fidelity.

### CFL constraint on `step_size`

PyChrono FSI uses **weakly compressible SPH** with an artificially low
sound speed (`sysSPH.GetSoundSpeed()` defaults to **10 m/s** for water —
intentionally far below the physical 1500 m/s to make the timestep
tractable). The CFL bound is approximately

```
step_size  <  C  ×  (1.2 × initial_spacing) / 10 m/s     (C ≈ 0.01–0.05)
```

The defaults `step_size = 1e-4 s` with `initial_spacing = 0.1 m` lie
comfortably inside this bound.

**Scaling rule**: if `initial_spacing` scales by factor `k`, scale
`step_size` by the same factor `k`. Otherwise the simulation either
becomes unstable (k<1, dt too large) or wastes compute (k>1, dt too small).

### `render_fps` is NOT a render bottleneck

Wall-time breakdown in a typical FSI scene:

| Component                                   | Share |
| ------------------------------------------- | ----- |
| SPH solver (CUDA: density, pressure, force) | 60–80% |
| BCE coupling onto FSI bodies                | 10–20% |
| MBS step (constraints, contacts)            | 5–10%  |
| Render (VSG)                                | <5%    |
| Driver / camera input                       | <0.1%  |

Each rendered frame requires `(1/render_fps)/step_size` physics steps —
with defaults `render_fps=50` and `step_size=1e-4`, that's **200 SPH
steps per visible frame**. The renderer is not the bottleneck.

Consequences for tuning:

- Lowering `render_fps` does almost nothing — saves a per-frame draw call,
  doesn't reduce step count.
- Disabling vis (`vis=None`) saves <5%.
- The only real lever is `initial_spacing`. See decision tree below.
- Driver type (interactive vs DataDriver vs PathFollower) is zero cost.

### Decision tree for choosing `initial_spacing`

```
Teaching demo or quick visual sanity check?
  → 0.10 m  (default; matches tutorial_VEH_FSI_FloatingBlock.py)

Research / publication run with explicit accuracy criteria?
  → 0.05 m  (8× cost, ~16× wall time vs default)

Need to resolve features < ~3 × default spacing
(thin sheets, narrow channels, splash droplet detail)?
  → 0.025 m  (64× cost; consider also shrinking V_fluid)

Default behavior — plan does NOT specify accuracy or feature size:
  → 0.10 m. Do NOT pre-emptively pick smaller "to be safe" — that's
    the most common cause of plans that look conservative on paper
    but burn 10–60 minutes of wall time per simulated second.
```

If a plan is currently too slow to iterate on, in priority order:

1. Increase `initial_spacing` (+ proportionally increase `step_size`).
2. Reduce `V_fluid` (half-fill the tank, shrink x/y dims).
3. Set `num_proximity_search_steps` to 3–5 (refresh neighbours less often
   — safe for slow-moving fluid).
4. Disable visualization (`vis=None`) only if running headlessly.

Do NOT lower `render_fps`, do NOT remove driver / camera setup, do NOT
remove sensor cameras for "speed" — negligible impact and only makes the
scene less useful.

## Output & Data Saving

End-of-sim ``particles.csv`` / ``scene_links.csv`` dumps are NOT
required by the review pipeline. The only motion-related CSV the
review pass reads is ``cam/motion_log.csv``, written ONLY when the
current step's plan declares ``motion_expectations``. The codegen
system prompt's rule 6 carries the canonical pattern (open the CSV
before the loop, append rows from the ``on_step`` callback, flush in
``finally``). Steps that declare no ``motion_expectations`` skip
this entirely and ship just the mp4.

Legacy writers ``write_links_csv`` / ``write_placement_csv`` /
``write_contacts_csv`` still exist in
``chrono_code.utils.scene_assets`` for ad-hoc diagnostic use, but
are no longer mandated.

### Per-step output (optional, for offline post-processing)

If the plan needs per-step particle / body data (publication plots, ML
training data, etc.):

```python
import os
particles_dir = os.path.join(out_dir, "particles")
fsi_dir       = os.path.join(out_dir, "fsi")
os.makedirs(particles_dir, exist_ok=True)
os.makedirs(fsi_dir,       exist_ok=True)

# Inside the loop — write at the desired output frequency
if output and time_val >= out_frame / output_fps:
    sysSPH.SaveParticleData(particles_dir)
    sysSPH.SaveSolidData(fsi_dir, time_val)
    out_frame += 1
```

`SaveParticleData` writes per-step particle CSV/VTK files;
`SaveSolidData` writes rigid body poses timestamped with the simulation
time. These are NOT a substitute for `particles.csv` (which is one frame
at end-of-sim, not the per-step trajectory).

## Hard Rules

Nine non-negotiable invariants. The first two are most often violated and
cause silent failures (no exception, wrong physics).

### HR-1: `step_fn` OWNS physics advance — must call `sysFSI.DoStepDynamics`

`run_recording_loop(sys, ..., step_fn=...)` **replaces** the default
`sys.DoStepDynamics(time_step)` call when `step_fn` is provided. The
caller's `step_fn` is the physics step, not a side-effect hook.

```python
# WRONG — only logs, never advances physics. Floating bodies don't move,
# sysSPH.GetParticlePositions() raises, particles.csv is never written.
def step_fn(i, t):
    if i % 1000 == 0:
        print(f"t={t:.3f}s")
run_recording_loop(sysMBS, ..., step_fn=step_fn)

# CORRECT — step_fn calls sysFSI.DoStepDynamics. Patterns F / G inline this.
def fsi_step(i, t):
    sysFSI.DoStepDynamics(step_size)
run_recording_loop(sysMBS, ..., step_fn=fsi_step)
```

This is the iteration_002 / session_20260428_164422 bug: a logging-only
`step_fn` froze the simulation, the floating plate never moved, and
`particles.csv` was never produced because the SPH state was never
finalized.

### HR-2: Single advance call — never `sysMBS.DoStepDynamics` or `vehicle.Advance` in the FSI loop

`sysFSI.DoStepDynamics(dT)` advances both SPH fluid and MBS bodies in one
coupled step. Adding a separate `sysMBS.DoStepDynamics(dT)` or
`vehicle.Advance(dT)` double-steps the MBS physics, producing kinematic
drift, unstable contacts, and wrong FSI coupling forces.

Vehicle scenes call only `vehicle.Synchronize(time, driver_inputs)`
before the FSI step (do this inside the `step_fn` body, see Pattern F).

### HR-3: Match gravity on both `sysMBS` and `sysFSI`

`sysMBS.SetGravitationalAcceleration(...)` and
`sysFSI.SetGravitationalAcceleration(...)` must receive the same vector.
Mismatch breaks hydrostatic equilibrium and causes floating bodies to
oscillate or drift.

### HR-4: Spawn floating body ABOVE the fluid free surface

Position the floating body so its bottom face is at or above `fzDim`.
Spawning inside the fluid causes initial BCE–particle overlap and an
explosive pressure impulse on the first few steps. `build_fsi_floating_box`
expects the caller to pass `world_center.z = fzDim + size.z/2` (or higher).

### HR-5: Computational domain MUST contain every FSI body's BCE markers (including trajectory)

`SetComputationalDomain(ChAABB(cMin, cMax), ...)` defines the spatial
hash range. Any BCE marker that lies outside `[cMin, cMax]` produces

```
[calcHashD] index N (x y z) out of min boundary (cMin.x cMin.y cMin.z)
```

followed almost immediately by `terminate called without an active
exception` and a coredump (no Python traceback).

The most common failure: cMin/cMax sized to hug the **fluid region only**,
while the vehicle / floating body spawns or moves *outside* that region.

**Wheel/spindle BCE markers extend to where the WHEEL is, not where the
chassis origin is.** For Polaris and similar catalog vehicles whose
chassis-frame origin sits at the *front axle* (not the geometric center —
see CLAUDE.md `project_chassis_frame_origin_conventions`), the rear wheel
BCE markers are at `chassis_x - wheelbase ± wheel_half_width`, which is
~2.7 m BEHIND the chassis origin. Sizing the domain to `vehicle_init_x ±
half_chassis_length` will leave the rear wheel BCE outside `cMin.x` and
trigger `[calcHashD] index N (...) out of min boundary` on the very first
`DoStepDynamics` call. **This was the failure mode of session
20260506_204433 iter 4-5**: front axle at x=−2.642, wheelbase 2.715 m,
wheel-collision-mesh half-width 0.34 m → rearmost BCE marker at
x ≈ −5.31, but `cMin.x` was −4.12. Particle at (−5.31, 0.63, 1.42) was
exactly the rear-wheel BCE.

```python
# Compute the union of {fluid region, all FSI body spawn footprints,
# vehicle WHEEL trajectory across full sim_duration} + ≥ 5 * initial_spacing
# margin on each face.
#
# For a wheeled vehicle with chassis-frame origin at the front axle
# (Polaris) or geometric center (HMMWV / Sedan), compute the four wheel
# centers explicitly; do NOT use chassis_x ± half_chassis_length:
#
#   front_left_x  = chassis_x  +  front_axle_offset_x       # 0 for Polaris
#   rear_left_x   = chassis_x  +  rear_axle_offset_x        # -wheelbase for Polaris
#   wheel_y       = chassis_y ± track_half + tire_half_width
#   wheel_z       = support_top_z + tire_radius
#   bce_extent    = wheel_collision_mesh.half_extent (~0.34 m for Polaris tire)
#
# Min/max over all wheels at all simulated times.
front_x_at_t = chassis_init_x + front_axle_offset_x + sim_duration * cruise_speed
rear_x_at_t0 = chassis_init_x + rear_axle_offset_x   # at t=0, before motion
veh_x_min = rear_x_at_t0 - bce_extent - 5 * d0
veh_x_max = front_x_at_t + bce_extent + 5 * d0
veh_y_max = abs(chassis_y) + track_half + tire_half_width + bce_extent + 5 * d0

union_min_x = min(-bxDim/2, veh_x_min, plate_min_x)
union_max_x = max(+bxDim/2, veh_x_max, plate_max_x)
cMin = chrono.ChVector3d(union_min_x, -byDim/2 - d0/2, -5*d0)
cMax = chrono.ChVector3d(union_max_x, +byDim/2 + d0/2, bzDim + 5*d0)
sysSPH.SetComputationalDomain(chrono.ChAABB(cMin, cMax), fsi.BC_Y_PERIODIC)
```

If `veh_y_max > byDim/2`, the wheel BCE extends BEYOND the periodic-Y
period. See HR-5b below — that combination is unsafe with `BC_Y_PERIODIC`.

**Anti-fix warning** — when the SPH solver throws `SIGABRT` / `NaN at
particle X` during the first few steps of a vehicle-FSI run, the
**immediate-looking fix is wrong**: deleting the
`sysFSI.AddFsiBody(spindle, geometry, False)` registration silences the
crash because the spindle BCE markers no longer exist to fall outside
`cMin/cMax`. But it silently disables wheel-fluid coupling — the
vehicle drives across the floating plate without sinking it, the VLM
review passes a physically broken sim because the visual still shows
water + vehicle + plate. Iter_009 of session_20260426_223737 did this.
The CORRECT fix is expanding the computational domain (above); keep the
spindle registration loop intact.

### HR-5b: FSI body BCE must NOT cross a periodic boundary

`BC_Y_PERIODIC` (and `BC_X_PERIODIC`) sets up a *fluid* periodicity: any
SPH particle that exits at `cMax.y` is re-injected at `cMin.y` with the
same x/z. The "period" of the boundary is `cMax.y - cMin.y`, which the
canonical pattern sets equal to `byDim` (= the tank's BCE container Y
width).

This is correct for the *fluid*. It is NOT correct for an FSI rigid
body whose BCE markers physically extend BEYOND the tank's Y footprint
(e.g. a vehicle on a platform that's wider than the tank — the wheels
sit at `|y| > byDim/2`). Those wheel BCE markers physically belong on
the platform, in air, NOT in periodically-wrapped fluid. With
`BC_Y_PERIODIC`, the solver still tries to apply periodic neighbour
search to those markers — a marker at `y = byDim/2 + 0.3` neighbours
markers at `y = -byDim/2 + 0.3`, which is on the OPPOSITE side of the
tank. Symptom: instant NaN density on first DoStepDynamics, or
`[calcHashD] out of boundary` if `cMax.y` doesn't match `byDim/2`
exactly.

**Decision rule** (apply BEFORE choosing `BC_*_PERIODIC`):

| Question | Answer A | Answer B |
|----------|----------|----------|
| Do any FSI body BCE markers (vehicle wheels, floating plate corners) extend beyond `±byDim/2` in y? | NO → `BC_Y_PERIODIC` is safe | YES → use `BC_NONE` + `bce_layers=ChVector3i(2, 2, -1)` (full Y walls) |

If you switch from `BC_Y_PERIODIC` to `BC_NONE`, you MUST also change
`bce_layers` from `(2, 0, -1)` to `(2, 2, -1)` — `0` on the y-axis was
"no Y wall, periodic handles it"; `BC_NONE` does NOT handle it, so
without explicit Y walls the tank is open on the long sides and the
fluid pours out. **This is the iter_006 contradiction of
session_20260506_204433**: agent flipped to `BC_NONE` because the
wheel-BCE-outside-period error told them to drop periodicity, but kept
`bce_layers=(2, 0, -1)` from the periodic recipe, leaving the tank Y
sides open → second crash mode.

### HR-6: BCE container z-axis MUST be `-1` for open-top tanks

For a vertical-gravity tank with the free surface on +z (the standard
FSI setup), the third component of `CreatePointsBoxContainer`'s layers
tuple MUST be `-1`:

```python
chrono.ChVector3i(2, 0, -1)   # ← z = -1, NOT 0, NOT 1
```

`-1` is "bottom-face only, top open." Other values for the z component
all break the tank in different ways:

| z value | What you get | Failure mode |
|---|---|---|
| `-1` | bottom only, open top | ✓ correct |
| `0` | NO floor, NO top | particles fall through → exceed cMin.z → SIGABRT in calcHashD ("Num nodes 2D: 0" — iter_006 first crash) |
| `1` | floor + ceiling (closed box) | pressure spike from no-air-escape → NaN → SIGABRT in DoStepDynamics (iter_006 second crash; codegen flipped 0→1 trying to fix the first crash and made it worse) |
| `2` | 2 layers each, top + bottom | same as `1` but worse |

The asymmetric convention (`+N` for X / Y closes both walls; `-1` for Z
closes the floor only) is the trip wire. `build_fsi_tank` exposes
`bce_layers` as a `ChVector3i` so the caller picks the right value once.

### HR-7: VSG-only recording — no sensor cameras for FSI

`ChCameraSensor` (OptiX) and `ChSphVisualizationVSG` (VSG) are two
independent render pipelines. SPH particles only register with VSG —
they never enter the OptiX scene tree. A sensor-camera mp4 of an FSI
scene therefore renders an **empty tank with no water at all**,
defeating the whole point.

```python
# WRONG for an FSI scene — water invisible
manager = sens.ChSensorManager(sysMBS)
setup_preview_camera(manager, ...)

# CORRECT — VSG-only, see Pattern F / G
finalize_vsg_mp4 = setup_vsg_recording(vis, "cam/vsg.mp4", fps=50.0)
```

If the scene also needs sensor data (depth / LiDAR for ML training),
add the sensor on top — but the **main video output of an FSI scene
must be VSG-only**.

### HR-8: `AddFsiBody` direct-call only for vehicle wheel spindles

For containers and floating bodies use the helpers (Patterns C / D);
they call `AddFsiBody` internally with the only correct frame. Direct
`AddFsiBody` calls survive only for the wheel-spindle geometry overload
(Pattern E). The trailing `is_flexible` argument is always `False` for
any `ChBody`-based registration — `True` triggers FEA coupling and
fails at runtime.

### HR-9: JSON `WheeledVehicle` requires the same `sysMBS` as the FSI stack

```python
vehicle = veh.WheeledVehicle(sysMBS, json_path)   # correct
```

Do NOT use no-arg convenience wrappers (`veh.HMMWV_Full()`,
`veh.CityBus()`) for FSI coupling — those wrappers create their own
internal `ChSystemSMC`, a different object from the one passed to
`ChFsiSystemSPH`. FSI coupling is then broken and spindle registration
silently does nothing.

### Diagnostic markers (BCE overlays default OFF; fluid markers default ON)

`build_fsi_vehicle_visualizer(...)` defaults `enable_fluid_markers` to
**True** (the SPH water cloud must render — without it the mp4 has no
blue particles), and defaults `enable_boundary_markers` and
`enable_rigid_body_markers` to **False**. Pattern G (non-vehicle) sets
the same flags identically in its manual setup. The BCE overlay
(boundary + rigid-body green dot grids) was previously default-on for
VLM review, but the dense green dot grid was routinely perceived by
the review LLM as the only thing in the scene and produced
false-negative verdicts (`session_20260429_060447` failed step 2 on
"tank invisible because only BCE markers visible" even though the tank
walls were correctly rendered behind the marker grid).

Physics correctness is now verified through `scene_placement.csv` body
end-states (pos / vel / ang_vel) inlined into the review LLM's prompt,
not through visual marker inspection. Pass `True` to
`enable_boundary_markers` or `enable_rigid_body_markers` only for
ad-hoc debugging when you suspect BCE alignment trouble on a specific
run.

### HR-10: Comments must not assert function contracts you have not verified

Codegen agents have repeatedly written comments above a utility call
that fabricate the utility's API, then written code matching the
fabricated comment. The most expensive variant is iter_007 of
`session_20260506_*`:

```python
# WRONG — comment is a hallucination; both sides of the call are wrong.
# build_fsi_vehicle_visualizer creates its own visFSI internally.
# then grab the returned visFSI to configure the color callback.
vis, visFSI = build_fsi_vehicle_visualizer(
    ...,
    sph_visualization=None,                       # ← false; required arg
)
visFSI.SetSPHColorCallback(...)                   # ← AttributeError on None
```

The function actually requires the caller to construct
`fsi.ChSphVisualizationVSG(sysFSI)` first and returns a single `vis`
(see Pattern F). Self-written prose explaining a function's behaviour
is unverifiable evidence; the next codegen iteration reads it back as
fact and amplifies the error.

The same pattern caused iter_006's `PLT_TOP_Z = LEFT_PLT_Z` bug —
`LEFT_PLT_Z` was named for the *top* but its formula stored the body
*center* z; the comment "platform top = tank rim" lied about what the
variable held, and the next iteration's vehicle init trusted the lie
and clipped the wheels into the platform.

Rules:

- Do NOT write comments that paraphrase or summarise a function's
  contract. If a contract clarification is genuinely useful, copy
  the exact docstring sentence verbatim and quote it.
- For utility helpers (`chrono_code.utils.*`), trust the function's
  docstring/signature, not your own restatement.
- For sanity-critical Z / TOP / BOTTOM / CENTER values, derive them
  from the actual body via
  `chrono_code.utils.vehicle_geometry.support_top_z_from_body(body)`
  rather than computing arithmetic on a constant whose name might
  drift from its meaning across iterations.
- When in doubt, replace the comment with a runtime `assert`:
  `assert plt_top_z == left_plt.GetTotalAABB().max.z`.

The Chrono API validator catches *invented method names*; it cannot
catch invented *call shapes* (returns, kwargs, side effects). HR-10
is the human-readable backstop for that gap.

## Skill Dependencies

### Required

| Skill               | Why                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `mbs/system_create` | `ChSystemSMC` creation, gravity, collision system type.                                    |
| `mbs/body_creation` | `ChBody` creation, mass/inertia, visual/collision shapes for container and floating plate. |

### Optional

| Skill                 | When to read                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `veh/wheeled_vehicle` | When the FSI scene includes a wheeled vehicle. Owns full vehicle construction, powertrain, tire setup.               |
| `veh/driver`          | When a driver (path-follower / data-driven) controls the vehicle. Owns `DriverInputs` struct and synchronize ordering. |
| `veh/terrain`         | When the vehicle also drives on SCM terrain adjacent to the FSI fluid domain.                                          |
| `mbs/collision`       | When non-trivial contact materials or collision families are needed.                                                   |
| `vsg`                 | Generic non-vehicle VSG window (Pattern G uses this directly).                                                         |
