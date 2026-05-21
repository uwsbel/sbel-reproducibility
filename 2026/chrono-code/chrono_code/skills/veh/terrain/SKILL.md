---
name: terrain
description: Create rigid terrain patches, SCM (Bekker-Wong soft-soil) terrain, and CRM (SPH) terrain for wheeled vehicle simulations.
compatibility: pychrono >= 8.0
metadata:
  domain: veh
---

# Skill: Vehicle Terrain

## Purpose

Create the terrain surface under a wheeled vehicle. Three flavors, from cheap to physically rich:

| Flavor | API | Physics | Cost | Use when |
|--------|-----|---------|------|----------|
| RigidTerrain | `veh.RigidTerrain` | Flat / heightmap rigid body, Bullet contacts | Low | Fast demos, paved roads, baseline sanity |
| SCMTerrain | `veh.SCMTerrain` | Bekker-Wong empirical soft soil, deformable grid | Medium | Off-road ruts, sinkage, deformation without SPH cost |
| CRMTerrain | `veh.CRMTerrain` | SPH continuum (elastic granular) | High | Splashing particles, large deformation, needs GPU |

## When to Use

When the user asks to simulate a vehicle driving on terrain — required for any wheeled vehicle simulation. Pick the flavor by the user's physics requirement; if unsure, SCM is the recommended default for off-road scenes.

## Rigid Terrain (Flat)

```python
terrain = veh.RigidTerrain(veh_obj.GetSystem())

# NSC material
patch_mat = chrono.ChContactMaterialNSC()
patch_mat.SetFriction(0.9)
patch_mat.SetRestitution(0.01)

# SMC material
patch_mat = chrono.ChContactMaterialSMC()
patch_mat.SetFriction(0.9)
patch_mat.SetRestitution(0.01)
patch_mat.SetYoungModulus(2e7)

# Add patch (flat ground)
patch = terrain.AddPatch(
    patch_mat,
    chrono.CSYSNORM,  # centered at origin, no rotation
    terrainLength,    # X direction size
    terrainWidth      # Y direction size
)

# Visual
patch.SetTexture(veh.GetVehicleDataFile("terrain/textures/tile4.jpg"), 200, 200)
patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))

terrain.Initialize()
```

## Rigid Terrain Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Friction | 0.8 - 0.9 | Friction coefficient |
| Restitution | 0.01 | Bounciness |
| YoungModulus | 2e7 | SMC only - stiffness |

## SCM Terrain (Bekker-Wong Soft Soil)

`SCMTerrain` is a deformable soft-soil model suitable for off-road driving — wheels sink into the surface and leave visible ruts. Cheaper than CRM (no SPH particles) but richer than RigidTerrain.

### Prerequisites

**CRITICAL: The ChSystem MUST have a collision system BEFORE constructing `SCMTerrain`.**
SCMTerrain's constructor checks for an associated collision system and raises `RuntimeError` if none is found.

If you are creating the system yourself (not via a vehicle wrapper):
```python
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
# MUST be called BEFORE constructing SCMTerrain
```

If using a vehicle wrapper (`hmmwv.Initialize()` etc.), the vehicle sets up the collision system automatically — no extra call needed. But **never call `SetCollisionSystemType()` AFTER `hmmwv.Initialize()`** (see Pitfalls below).

### Construction

```python
terrain = veh.SCMTerrain(system)   # takes the vehicle's ChSystem
```

### Soil Parameters (All 8 Are Required)

Signature: `SetSoilParameters(Bekker_Kphi, Bekker_Kc, Bekker_n, Mohr_cohesion, Mohr_friction, Janosi_shear, elastic_K, damping_R)`

```python
terrain.SetSoilParameters(
    2e6,    # Bekker_Kphi    — frictional modulus (Pa)
    0,      # Bekker_Kc      — cohesive modulus
    1.1,    # Bekker_n       — exponent (1.0 soft → 1.5 hard)
    0,      # Mohr_cohesion  — cohesive limit (Pa)
    30,     # Mohr_friction  — friction angle (deg)
    0.01,   # Janosi_shear   — shear coefficient (m)
    2e8,    # elastic_K      — elastic stiffness (Pa/m)
    3e4,    # damping_R      — vertical damping (Pa·s/m)
)
```

**Exactly 8 positional args are required** — no keyword form. Do not omit `damping_R` (8th arg) and do not add a 9th.

### Active Domain (MANDATORY for Large Patches)

SCM does a ray-cast per grid vertex per step. A 120×120 m patch at 5 cm resolution is ~6 million vertices and will grind the sim. Attach an **active domain** so only cells near the vehicle are updated.

**CRITICAL: Attach active domain to the CHASSIS body, NOT to wheel spindles.**

SCM's `UpdateActiveDomain()` projects the OOBB using the reference body's orientation (SCMTerrain.cpp:957-1001). Wheel spindles rotate at hundreds of RPM — the OOBB rotates with them, causing the projected terrain range to become empty at many rotation angles. Result: `rays=0`, zero deformation, no ruts.

```python
# CORRECT — chassis body stays level, stable OOBB projection
terrain.AddActiveDomain(
    hmmwv.GetChassisBody(),
    chrono.ChVector3d(0, 0, 0),     # local centre offset
    chrono.ChVector3d(5, 3, 1),     # half-extents (m)
)
```

```python
# WRONG — spindle rotates, OOBB becomes degenerate → rays=0
for axle in hmmwv.GetVehicle().GetAxles():
    terrain.AddActiveDomain(
        axle.m_wheels[0].GetSpindle(),   # DO NOT USE
        chrono.ChVector3d(0, 0, 0),
        chrono.ChVector3d(1.5, 1.0, 1.5),
    )
```

### Initialize

```python
terrain.Initialize(120.0, 120.0, 0.1)
# length (m), width (m), grid resolution (m).
# 0.1 m grid is a good balance for performance with visible ruts.
# 0.05 m gives finer ruts but 4× more vertices and ray-cast cost.
```

### Texture

```python
terrain.SetMeshWireframe(False)
terrain.SetTexture(
    chrono.GetChronoDataFile("vehicle/terrain/textures/dirt.jpg"),
    80, 80,  # UV tiling
)
```

### Tire Collision Cylinders (REQUIRED for Non-Rigid Tires)

SCM detects tire contact via `ChCollisionSystem::RayHit()` — it casts rays from each grid vertex and looks for collision shapes. **Rigid tires** add their own collision geometry automatically. **TMEASY / PAC02 / FIALA** tires do NOT — you must add explicit collision cylinders to each spindle.

```python
tire_rad = hmmwv.GetVehicle().GetAxles()[0].m_wheels[0].GetTire().GetRadius()
tire_w = hmmwv.GetVehicle().GetAxles()[0].m_wheels[0].GetTire().GetWidth()
tire_mat = chrono.ChContactMaterialSMC()
tire_mat.SetFriction(0.9)
tire_mat.SetRestitution(0.1)

TIRE_FAMILY = 1
SUPPORT_FAMILY = 4

for axle in hmmwv.GetVehicle().GetAxles():
    for iw in range(2):
        spindle = axle.m_wheels[iw].GetSpindle()
        spindle.AddCollisionShape(
            chrono.ChCollisionShapeCylinder(tire_mat, tire_rad + 0.04, tire_w),
            chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleX(math.pi / 2)),
        )
        spindle.EnableCollision(True)
        sp_cm = spindle.GetCollisionModel()
        sp_cm.SetFamily(TIRE_FAMILY)
        sp_cm.DisallowCollisionsWith(TIRE_FAMILY)
        sp_cm.DisallowCollisionsWith(SUPPORT_FAMILY)

# MANDATORY: rebuild all collision models after post-init shape changes
system.GetCollisionSystem().BindAll()
```

**Three critical rules for tire collision cylinders:**

**Rule 1 — Use `tire_rad + 0.04` not `tire_rad`.** The extra 4 cm ensures the collision cylinder extends slightly below the terrain surface at the TMEASY equilibrium spindle height, guaranteeing SCM detects sinkage.

**Rule 2 — NEVER call `DisallowCollisionsWith(0)` (family 0).** Bullet's `RayHit()` queries use family 0 by default. If you disallow family 0 on the tire cylinder, SCM's ray-casts are filtered out → `hits=0` → no ruts form. Chassis-tire non-collision should be handled on the chassis side (via `add_collision_via_subbodies(tire_family=TIRE_FAMILY)`), not on the tire side.

**Rule 3 — Call `collsys.BindAll()` after adding shapes.** The per-model `Remove()`/`BindItem()` path does NOT work — `BindItem()` checks `HasImplementation()` and silently skips models that are already bound. `BindAll()` forces Bullet to rebuild all collision models, making the new cylinders visible to ray-casts.

### Support Plane for Dynamic Props (Optional)

SCM is a depth-based deformable surface, not a general rigid ground. Dynamic props (rocks, cottage) placed on the terrain will free-fall through z=0. Add a hidden rigid support plane:

```python
support_mat = make_contact_material(
    friction=0.9, restitution=0.01, method="SMC", young_modulus=2e7,
)
support = chrono.ChBodyEasyBox(120.0, 120.0, 0.2, 1000, False, True, support_mat)
support.SetName("asset_support_ground")
support.SetPos(chrono.ChVector3d(0, 0, -0.1))  # top at z=0 (SCM rest plane)
support.SetFixed(True)
support.EnableCollision(True)

support_cm = support.GetCollisionModel()
support_cm.SetFamily(SUPPORT_FAMILY)
support_cm.DisallowCollisionsWith(TIRE_FAMILY)       # tires ride on SCM, not support
support_cm.DisallowCollisionsWith(CHASSIS_FAMILY)    # chassis rides on SCM
support_cm.DisallowCollisionsWith(3)                 # chassis sub-body family
system.AddBody(support)
```

### Sinkage Visualization

Not available in VSG — VSG shows geometric deformation directly. Do not call `SetPlotType`; it is an Irrlicht-only API and is not supported.

### Heightmap Variant (for pools / depressions)

```python
terrain.Initialize(heightmap_path, scm_size, scm_size,
                   -pool_depth, 0.0, scm_resolution)
# heightmap: 16-bit PNG, white=max height, black=min height
# min_z, max_z: height range mapping
```

### Simulation Loop

SCM uses the standard terrain synchronize/advance pattern:

```python
terrain.Synchronize(time)
terrain.Advance(step_size)
# Pass terrain into vehicle.Synchronize so the wheels sample it
hmmwv.Synchronize(time, driver_inputs, terrain)
```

### SCM vs CRM vs Rigid — Picking One

- **Off-road HMMWV / military / agricultural** → SCM (this section).
- **Flat paved road, fast demo** → RigidTerrain (above).
- **Splashing granular material, large free surface** → CRM (below). Requires SPH GPU cost.

## SCM Pitfalls Summary

| Symptom | Cause | Fix |
|---------|-------|-----|
| Chassis doesn't translate at any throttle, `GetSpeed()` oscillates around 0 | `TireModelType_RIGID` (the `HMMWV_Full` default) on SCM — no slip/grip curve, wheels break loose under engine torque | `SetTireType(TireModelType_TMEASY)` + `SetTireStepSize(step)` + add spindle cylinders (next row). Do not try to rescue with stiffer soil / higher cohesion — the failure is structural. See `veh/wheeled_vehicle/SKILL.md` "Picking a tire type by terrain". |
| `rays=0` | Active domain on rotating spindle | Use chassis body for `AddActiveDomain` |
| `hits=0` (rays > 0) | `DisallowCollisionsWith(0)` on tire | Remove that call; handle on chassis side |
| `hits=0` (rays > 0) | Missing `BindAll()` after shape changes | Call `collsys.BindAll()` after adding tire cylinders |
| `patches=0` (hits > 0) | Collision cylinder too small | Use `tire_rad + 0.04` not `tire_rad` |
| No ruts with TMEASY | No tire collision cylinders | Add explicit `ChCollisionShapeCylinder` to each spindle |
| Want "colored" C++-style ruts (sinkage heatmap) | `SetPlotType` is Irrlicht-only, silently a no-op in VSG | VSG only shows geometric mesh deformation. Accept it, or port to Irrlicht vis (big rewrite). Do not call `SetPlotType` under VSG. |
| Props fall through z=0 | No rigid support plane | Add hidden `ChBodyEasyBox` at z=-0.1 |
| Do NOT call `SetCollisionSystemType()` after `hmmwv.Initialize()` | Creates new collision system, leaves dangling `impl` pointers on all HMMWV bodies — `BindAll()` skips them | Trust the collision system HMMWV already created; only call `BindAll()` on it |

## CRM Terrain (Wheel-Soil Interaction)

For realistic wheel-soil interaction with deformable terrain:

```python
from pychrono.fsi import (
    ChFsiFluidSystemSPH, ChFsiSystemSPH,
    ElasticMaterialProperties, SPHParameters,
    IntegrationScheme_RK2, ViscosityMethod_ARTIFICIAL_BILATERAL,
    BoundaryMethod_ADAMI, OutputLevel_STATE
)

# Create CRM terrain
initial_spacing = 0.03
terrain = veh.CRMTerrain(system, initial_spacing)

# Configure FSI system
sysFSI = terrain.GetFsiSystemSPH()
sysSPH = terrain.GetFluidSystemSPH()
sysSPH.EnableCudaErrorCheck(False)
terrain.SetVerbose(verbose)
terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
terrain.SetStepSizeCFD(step_size)

# Elastic material properties
mat_props = ElasticMaterialProperties()
mat_props.density = 1700
mat_props.Young_modulus = 1e6
mat_props.Poisson_ratio = 0.3
mat_props.mu_I0 = 0.04
mat_props.mu_fric_s = 0.7
mat_props.mu_fric_2 = 0.7
mat_props.average_diam = 0.005
mat_props.cohesion_coeff = 5e3
terrain.SetElasticSPH(mat_props)

# SPH parameters
sph_params = SPHParameters()
sph_params.integration_scheme = IntegrationScheme_RK2
sph_params.initial_spacing = initial_spacing
sph_params.free_surface_threshold = 0.8
sph_params.artificial_viscosity = 0.5
sph_params.viscosity_method = ViscosityMethod_ARTIFICIAL_BILATERAL
sph_params.boundary_method = BoundaryMethod_ADAMI
sph_params.use_variable_time_step = True
terrain.SetSPHParameters(sph_params)

terrain.SetOutputLevel(OutputLevel_STATE)
terrain.Initialize()
```

## CRM Terrain Construction

### Rectangular Patch

```python
terrain.Construct(
    chrono.ChVector3d(length, width, 0.25),  # dimensions
    chrono.ChVector3d(length / 2, 0, 0),     # center
    BoxSide_ALL & ~BoxSide_Z_POS              # all boundaries except top
)
```

### Height Map Patch

```python
terrain.Construct(
    veh.GetVehicleDataFile("terrain/height_maps/bump64.bmp"),  # image file
    terrain_length,
    terrain_width,
    chrono.ChVector2d(0.25, 0.55),  # height range [min, max]
    0.25,                          # depth
    True,                           # uniform depth
    chrono.ChVector3d(terrain_length / 2, 0, 0),  # center
    BoxSide_Z_NEG                  # bottom wall
)
```

## Terrain Synchronization (Simulation Loop)

```python
terrain.Synchronize(time)
terrain.Advance(step_size)

# In vehicle sync, pass terrain as argument
veh_obj.Synchronize(time, driver_inputs, terrain)
```

## Getting Terrain Information

```python
aabb = terrain.GetSPHBoundingBox()
num_sph = terrain.GetNumSPHParticles()
num_bce = terrain.GetNumBoundaryBCEMarkers()
print(f"SPH particles: {num_sph}")
print(f"AABB: {aabb.min} to {aabb.max}")
```

## Skill Dependencies

For vehicle setup and system creation:
- `../../mbs/system_create/` — ChSystem creation
- `../../mbs/collision/` — Contact materials
- `../wheeled_vehicle/` — Vehicle creation
- `../driver/` — Driver systems

## API Contract

allowed_classes:
- veh.RigidTerrain(veh_obj.GetSystem())
- veh.SCMTerrain(system)
- veh.CRMTerrain(system, initial_spacing)
- chrono.ChContactMaterialNSC()
- chrono.ChContactMaterialSMC()
- chrono.ChCollisionShapeCylinder(tire_mat, tire_rad, tire_w)
- chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleX(math.pi / 2))
- chrono.ChBodyEasyBox(length, width, height, density, visualize, collide, material)
- ElasticMaterialProperties()
- SPHParameters()

allowed_methods:
- patch_mat.SetFriction(friction)
- patch_mat.SetRestitution(restitution)
- patch_mat.SetYoungModulus(young_modulus)
- terrain.AddPatch(patch_mat, chrono.CSYSNORM, terrainLength, terrainWidth)
- patch.SetTexture(veh.GetVehicleDataFile("terrain/textures/tile4.jpg"), u_tiles, v_tiles)
- patch.SetColor(chrono.ChColor(r, g, b))
- terrain.Initialize()
- terrain.Initialize(length, width, resolution)
- terrain.Initialize(heightmap_path, length, width, min_z, max_z, resolution)
- terrain.SetSoilParameters(Bekker_Kphi, Bekker_Kc, Bekker_n, Mohr_cohesion, Mohr_friction, Janosi_shear, elastic_K, damping_R)
- terrain.AddActiveDomain(hmmwv.GetChassisBody(), chrono.ChVector3d(cx, cy, cz), chrono.ChVector3d(hx, hy, hz))
- terrain.SetMeshWireframe(False)
- terrain.SetTexture(chrono.GetChronoDataFile("vehicle/terrain/textures/dirt.jpg"), u_tiles, v_tiles)
- terrain.Synchronize(time)
- terrain.Advance(step_size)
- terrain.GetFsiSystemSPH()
- terrain.GetFluidSystemSPH()
- terrain.SetVerbose(verbose)
- terrain.SetGravitationalAcceleration(chrono.ChVector3d(x, y, z))
- terrain.SetStepSizeCFD(step_size)
- terrain.SetElasticSPH(mat_props)
- terrain.SetSPHParameters(sph_params)
- terrain.SetOutputLevel(OutputLevel_STATE)
- terrain.Construct(chrono.ChVector3d(length, width, depth), chrono.ChVector3d(cx, cy, cz), BoxSide_ALL & ~BoxSide_Z_POS)
- terrain.Construct(heightmap_file, terrain_length, terrain_width, chrono.ChVector2d(min_h, max_h), depth, uniform_depth, chrono.ChVector3d(cx, cy, cz), BoxSide_Z_NEG)
- terrain.GetSPHBoundingBox()
- terrain.GetNumSPHParticles()
- terrain.GetNumBoundaryBCEMarkers()
- sysSPH.EnableCudaErrorCheck(False)
- spindle.AddCollisionShape(shape, frame)
- spindle.EnableCollision(True)
- spindle.GetCollisionModel()
- sp_cm.SetFamily(family)
- sp_cm.DisallowCollisionsWith(family)
- system.GetCollisionSystem().BindAll()
- support.SetName(name)
- support.SetPos(chrono.ChVector3d(x, y, z))
- support.SetFixed(True)
- support.EnableCollision(True)
- support.GetCollisionModel()
- system.AddBody(support)
- hmmwv.GetVehicle().GetAxles()
- axle.m_wheels[iw].GetSpindle()
- axle.m_wheels[iw].GetTire().GetRadius()
- axle.m_wheels[iw].GetTire().GetWidth()
- hmmwv.GetChassisBody()
- hmmwv.Synchronize(time, driver_inputs, terrain)
- veh.GetVehicleDataFile(path)
- chrono.GetChronoDataFile(path)

allowed_constants:
- chrono.CSYSNORM
- chrono.VNULL
- BoxSide_ALL
- BoxSide_Z_POS
- BoxSide_Z_NEG

allowed_utils:
- from pychrono.fsi import ChFsiFluidSystemSPH, ChFsiSystemSPH, ElasticMaterialProperties, SPHParameters, IntegrationScheme_RK2, ViscosityMethod_ARTIFICIAL_BILATERAL, BoundaryMethod_ADAMI, OutputLevel_STATE
