---
name: mbs_in_scene
description: Entry point for rigid-body hybrid plans (plan_type=mbs_in_scene) combining a robot or vehicle with scene assets — NO fluid coupling. Routes to the correct domain skills and defines only high-level invariants. Use core/fsi_in_scene instead when the plan involves SPH fluid or FSI body registration.
compatibility: pychrono >= 8.0
metadata:
  domain: core
---

# Core Skill: MBS in Scene

Use this skill when `plan_type = "mbs_in_scene"`.

This core skill is a **classifier, router, and constraint layer** for hybrid simulations that combine a moving rigid mechanical body with environmental assets. It does not own concrete environment layouts, terrain recipes, or exact utility signatures. Those belong to the relevant `veh/*`, `robot/*`, `scene/*`, and `sens/*` child skills.

**Use `fsi_in_scene` instead** when the plan contains SPH fluid (any
`scene_object` with `domain_type` starting `sph_`) or an FSI body
registration (`scene_objects[*].fsi_registration` set). `mbs_in_scene`
is for rigid-body hybrid scenes (vehicle / robot + terrain + props)
without fluid coupling — its required-skills tables intentionally omit
`fsi/sph` because including it on rigid-only plans wastes pre-injection
budget on irrelevant patterns.

## Responsibility

This core skill covers only:

- deciding whether the hybrid scene is vehicle-wrapper-driven, robot-driven, or another shared-system hybrid case
- which child skills must be read before writing code
- global invariants around system ownership, world convention, and rendering separation
- dispatch rules for vehicle-owned systems vs shared-system scenes

This core skill does **not** define:

- concrete offroad prop layouts
- concrete office/furniture layouts
- detailed terrain or robot recipes
- camera placement formulas tied to room sizes or vehicle dimensions
- exact utility signatures beyond short category reminders

## Classification Rules

Choose the child skill path based on the dominant moving-body family and ownership model:

- **Wrapper-managed vehicle scene**
  - indicators: wheeled vehicle wrappers, vehicle terrain integration, vehicle-specific visualization or driver stacks
- **Robot-centered hybrid scene**
  - indicators: robot, rover, quadruped, manipulator, robot policy/controller plus environment assets
- **Other shared-system hybrid scene**
  - any mixed moving-body + environment case not fully owned by the two categories above

## Required Skills for Wrapper-Managed Vehicle Scenes

Read these before writing code:

| Skill | Why it is required |
|-------|--------------------|
| `veh/wheeled_vehicle` | Vehicle wrapper ownership, visualization class, wrapper stepping rules, chassis collision via sub-bodies |
| `veh/terrain` | Terrain models, soil/contact setup, terrain-specific constraints |
| `veh/driver` | Driver setup and synchronization order |
| `scene/custom_assets_scene_convex_decomp` | **Required when the scene includes outdoor props (trees / rocks / cottage / foliage) OR indoor office assets.** Authoritative reference for `chrono_code.utils.scene_assets` — `AssetDescriptor` + `add_visual_assets` placement, per-asset-family collision rules, support-plane recipe for dynamic props on SCM, `FootprintRegistry` for deterministic non-overlap |
| `vsg` | VSG visualization window setup, camera, grid, sky, render loop |
| `sens/camera` | `setup_preview_camera`, recorder loop integration |
| `sens/sensor_manager` | `ChSensorManager` construction and scene-light API signatures — `SetAmbientLight` takes `ChVector3f` but `AddPointLight` takes `ChColor`; do not guess |

## Required Skills for Robot-Centered Hybrid Scenes

Read these before writing code:

| Skill | Why it is required |
|-------|--------------------|
| owning `scene/*` skill | Asset pipeline, placement rules, ground rules, scene-specific constraints |
| owning `robot/*` or vehicle skill | Robot or wrapper-managed body setup |
| `vsg` | VSG visualization window setup, camera, grid, sky, render loop |
| `sens/camera` | `setup_preview_camera`, recorder loop integration |
| `sens/sensor_manager` | `ChSensorManager` construction and scene-light API signatures — `SetAmbientLight` takes `ChVector3f` but `AddPointLight` takes `ChColor`; do not guess |

## Optional Skills — read if needed

| Skill | When to read |
|-------|--------------|
| `mbs/system_create` | Shared-system mechanics setup when the moving body does not own its own wrapper system |
| `mbs/collision` | Contact families, contact materials, collision policies |
| `mbs/quaternions` | Rotation rules |

## Global Invariants

- Default world convention is **`Z-up`** unless an owning child skill explicitly owns a different convention.
- `VSG` window viewpoint and sensor cameras are **separate concerns**:
  - `vis.AddCamera(...)` or `vis.SetChaseCamera(...)` sets the interactive VSG window view
  - `setup_preview_camera(...)` creates actual sensor-camera output
- Read the owning child skills before writing any concrete code.
- Do **not** call `chrono.SetChronoDataPath()` by default.
- All assets must be implemented through project utilities from `chrono_code.utils` / `chrono_code.utils.scene_assets`; the owning child skill decides which helper is valid.
- **Indoor robot/vehicle interaction assets are dynamic by default.** This rule is limited to repo-local indoor `data/scene/<name>/<name>.obj` furniture/props such as chairs, tables, desks, boxes, and tabletop props: set `fixed=false`, `is_dynamic=true`, and use a reasonable mass unless the user explicitly says the object is anchored/static. Do **not** apply this default to outdoor/offroad assets (trees, bushes, rocks, cottage, terrain props); keep those governed by the owning scene skill's outdoor family rules.
- **Robot/vehicle spawn XY must be resolved through the same `FootprintRegistry` that owns the scene assets.** Hardcoding `ChVector3d(x, y, z)` for the moving body while the scene uses `FootprintRegistry` bypasses the convex-hull AABB overlap check and routinely spawns the robot inside a prop (e.g. the robot's trunk clipping into a chair's seat). The robot then gets pinned by contact constraints and appears stationary to the VLM, masquerading as a control-loop bug. See `scene/custom_assets_scene_convex_decomp` §"Placing a robot/vehicle through the registry" for the canonical pattern.

## System Ownership Rules

- If the moving body is a wrapper-managed vehicle, initialize the wrapper first and then take:
  - `system = vehicle.GetSystem()`
- Do **not** create a second `ChSystem` for a wrapper-managed vehicle.
- Keep terrain, scene props, sensor manager, and visualization attached to that same owned system.
- Do not mix incompatible system/contact setups across one hybrid scene.

## Routing Rules

- Wrapper-managed vehicle + environment:
  route to `veh/wheeled_vehicle`, `veh/terrain`, `veh/driver`, `sens/camera`,
  and — if the scene contains outdoor props (trees/rocks/cottage) —
  `scene/custom_assets_scene_convex_decomp` (§5 "Outdoor Props")
- Robot-centered hybrid scene:
  route to the owning `scene/*` skill, the owning `robot/*` skill, and `sens/camera`
- Hybrid scene with environmental meshes but no wrapper-owned vehicle:
  route to the owning scene skill plus the relevant `mbs/*` or `robot/*` skill

## Minimal Category-Level Reminders

- Wrapper-managed vehicles own their system.
- Sensor recording requires `setup_preview_camera()`; VSG window camera APIs are not a replacement.
- All assets should be implemented through project utility functions; the owning child skill chooses the valid helper for the asset family.

## What Core Must Not Contain

If you are writing or updating a core hybrid skill, do not place these here:

- concrete asset-family layouts
- exact `AssetDescriptor(...)`, `add_visual_assets(...)`, `create_asset_body(...)`, or `SCMTerrain(...)` recipes
- terrain parameter tables
- room-size-specific or vehicle-dimension-specific camera formulas
- robot-specific or vehicle-specific code templates

Those belong in the owning child skills.
