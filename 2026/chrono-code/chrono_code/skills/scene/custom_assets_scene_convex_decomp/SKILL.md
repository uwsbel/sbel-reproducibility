---
name: custom_assets_scene_convex_decomp
description: "Authoritative reference for chrono_code.utils.scene_assets usage — AssetDescriptor + add_visual_assets + convex-hull collision + FootprintRegistry placement. Covers two canonical scenarios (indoor office workstation with repo-local data/scene/ assets, outdoor props for vehicle/robot scenes using Chrono built-in sensor/offroad/ assets)."
compatibility: pychrono >= 8.0
metadata:
  domain: scene
---
# Skill: Custom-Assets Scene Utilities

Use this skill any time a plan needs to place OBJ-mesh scene bodies via
`chrono_code.utils.scene_assets` utilities — `AssetDescriptor`,
`add_visual_assets`, `create_asset_body`, `make_contact_material`,
`ensure_convex_json`, `FootprintRegistry`, etc. It is the single
authoritative reference for how those utilities are wired together; the
same utilities cover both canonical scenarios below, and no other
`scene/*` skill owns them.

Canonical scenarios:

1. **Indoor office workstation** — repo-local assets under
   `data/scene/<name>/<name>.obj`, placed with `create_asset_body` +
   `FootprintRegistry`. Based on `demo/scene/custom_assets_scene_convex_decomp.py`.
2. **Outdoor props around a vehicle or robot** — Chrono built-in assets
   under `<ChronoDataPath>/sensor/offroad/` (trees, bushes, rocks,
   cottage), placed with `AssetDescriptor` + `add_visual_assets`. Based
   on `demo/scene/demo_SEN_HMMWV_offroad_vsg.py`.

Both scenarios share the same utility interface rules, reference asset
metadata conventions, mandatory imports, and banned inline redefinitions.
The indoor/outdoor split shows up only in §3 (asset discovery path) and
§5 (outdoor-specific collision recipes, support plane).

## API Contract

Any API call not listed below is presumed invalid for this scenario. Do not invent or guess APIs.

allowed_classes:

- chrono.ChSystemNSC
- chrono.ChSystemSMC
- chrono.ChVector3d
- chrono.ChVector3f
- chrono.ChQuaterniond
- chrono.ChFramed
- chrono.ChMatrix33d
- chrono.ChColor
- chrono.ChCollisionSystem
- chrono.ChCollisionModel
- chrono.ChTriangleMeshConnected
- chrono.ChBody
- chrono.ChBodyEasyBox
- chrono.ChContactMaterialNSC
- chrono.ChContactMaterialSMC
- chrono.ChCollisionShapeConvexHull
- chrono.ChCollisionShapeBox
- chrono.ChVisualShapeTriangleMesh
- chrono.ChVisualShapeBox
- chrono.ChVisualShapeLine
- chrono.ChLineSegment
- chrono.ReportContactCallback
- sens.ChSensorManager
- sens.Background

allowed_constants:

- chrono.ChCollisionSystem.Type_BULLET
- chrono.CameraVerticalDir_Z
- chrono.QUNIT
- sens.BackgroundMode_SOLID_COLOR

allowed_methods:

- system.SetGravitationalAcceleration(...)
- system.SetCollisionSystemType(...)
- system.AddBody(...)
- system.DoStepDynamics(dt)
- system.GetChTime()
- system.GetSolver()
- system.GetNumContacts()
- system.GetContactContainer().ReportAllContacts(callback)
- solver.AsIterative()
- solver.SetMaxIterations(...)
- chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(...)
- chrono.ChCollisionModel.SetDefaultSuggestedMargin(...)
- mat.SetFriction(...)
- mat.SetRestitution(...)
- mesh.LoadWavefrontMesh(path, True, True)
- mesh.GetBoundingBox()
- mesh.Transform(translation, rotation_matrix)
- mesh.GetCoordsVertices()
- mesh.GetNumTriangles()
- mesh.GetTriangle(i)
- mat33.SetMatr(matrix)
- mat33.SetFromDirectionAxes(forward, lateral, up)
- mat33.GetQuaternion()
- quat.Normalize()
- body.SetName(name)
- body.SetMass(mass)
- body.SetInertiaXX(inertia)
- body.SetFixed(flag)
- body.SetSleepingAllowed(False)
- body.AddCollisionShape(shape, frame)
- body.AddVisualShape(shape, frame)
- body.EnableCollision(True)
- body.SetPos(...)
- body.GetName()
- body.GetPos()
- body.GetRot()
- body.GetPosDt()
- body.GetTotalAABB()
- vis_shape.SetMesh(mesh)
- vis_shape.GetMesh()
- vis_shape.SetColor(color)
- vis_shape.SetOpacity(alpha)
- vis_line.SetColor(color)
- vis_line.SetThickness(val)
- vis.Run()
- vis.BeginScene()
- vis.Render()
- vis.EndScene()
- manager = sens.ChSensorManager(system)
- manager.scene.SetBackground(bg)
- manager.scene.SetAmbientLight(...)
- manager.scene.AddPointLight(...)
- manager.Update()

allowed_utils:

- chrono_code.utils.scene_assets.create_asset_body             # Scenario 1: repo-local data/scene/ assets
- chrono_code.utils.scene_assets.add_visual_assets             # Scenario 2: sensor/offroad/ outdoor assets
- chrono_code.utils.scene_assets.add_collision_from_decomposition
- chrono_code.utils.scene_assets.add_collision_via_subbodies   # vehicle chassis sub-body collision (see veh/wheeled_vehicle)
- chrono_code.utils.scene_assets.convex_decompose_asset
- chrono_code.utils.scene_assets.ensure_convex_json            # auto-runs inside add_visual_assets when collision=True; rarely called directly
- chrono_code.utils.scene_assets.load_convex_hulls
- chrono_code.utils.scene_assets.transform_vertices
- chrono_code.utils.scene_assets.make_contact_material
- chrono_code.utils.scene_assets.box_inertia
- chrono_code.utils.scene_assets.write_placement_csv
- chrono_code.utils.scene_assets.write_contacts_csv
- chrono_code.utils.scene_assets.write_links_csv
- chrono_code.utils.scene_assets.AssetDescriptor
- chrono_code.utils.scene_placement.FootprintRegistry
- chrono_code.utils.scene_placement.SurfaceStack
- chrono_code.utils.setup_preview_camera
- chrono_code.utils.run_recording_loop

## Scenario Contract

### Scenario 1 — indoor office workstation

- Scene type: indoor office workstation
- Assets: OBJ meshes in repo-local `data/scene/<name>/<name>.obj`, each with a pre-computed `*_convex.json` for collision
- Physics goal: stable resting contact for desk, chair, and small desktop props
- Collision goal: each concave mesh uses multiple `ChCollisionShapeConvexHull` parts loaded from JSON
- Placement goal: desk and chair on floor (no XY overlap), props on desk (no XY overlap with siblings, must stay inside desk's top footprint)
- Dynamic assets: all floor-level and stacked assets are **`fixed=False`** by default so gravity can settle them and a robot can push them. Only walls and ground are fixed.
- Placement API: `create_asset_body` + `FootprintRegistry` (floor) + `SurfaceStack` (per parent for stacked props)
- Visualization: VSG real-time view + 4 sensor cameras capturing screenshots on exit

### Scenario 2 — outdoor props around a vehicle or robot

- Scene type: outdoor prop scatter (forest, rocky field, cottage vista)
- Assets: Chrono built-in OBJs under `<ChronoDataPath>/sensor/offroad/` (tree1..3, bush1..2, rock1..5, cottage). No repo-local `_convex.json` needed — `add_visual_assets` generates & caches the decomposition JSON next to the OBJ on first run.
- Physics goal: vehicle/robot actually collides with rocks and the cottage; foliage stays visual-only for performance
- Collision goal: foliage has no collision; static rocks/cottage use VHACD convex hulls; dynamic / pushable rocks use a single outer convex hull (Bullet `cbtCompoundShape` broadphase-AABB bug workaround)
- Placement API: `AssetDescriptor` + `add_visual_assets`; optional `FootprintRegistry` when the plan cares about deterministic non-overlapping spots
- Dynamic props on `SCMTerrain`: **require** the support-plane recipe in §5
- Visualization: VSG via the owning vehicle/robot skill (e.g. `veh.ChWheeledVehicleVisualSystemVSG`) + `setup_preview_camera` sensor rig

## Mandatory Imports

Every generated `simulation.py` MUST use these utility imports. **Do NOT redefine these functions inline** — they are already implemented and tested in the project utilities.

```python
import json, os
import numpy as np
import pychrono.core as chrono
import pychrono.vsg3d as chronovsg
import pychrono.sensor as sens

from chrono_code.utils.scene_assets import (
    create_asset_body,
    add_visual_assets,
    AssetDescriptor,
    make_contact_material,
    box_inertia,
    write_placement_csv,
    write_contacts_csv,
    write_links_csv,
)
from chrono_code.utils.scene_placement import FootprintRegistry, SurfaceStack
from chrono_code.utils import setup_preview_camera   # MUST be called — see §7
```

**CRITICAL**: `setup_preview_camera` is NOT optional. You MUST actually call it (see §7 Sensor Cameras below). Simply importing it is not enough. `vis.AddCamera` is only for the VSG window viewpoint — it does NOT record video or produce camera output. Sensor cameras via `setup_preview_camera` are a **separate required step**.

## Utility Interface Rules (HARD)

- `make_contact_material(friction=..., restitution=..., method="NSC"|"SMC")`
  - First positional argument is `friction`, not `system`.
  - Wrong: `make_contact_material(system, friction=0.8, ...)`
- `create_asset_body(system, name, asset_dir, assets_base_dir, target_heights, position, ...)`
  - Only for repo-local assets laid out as `data/scene/<asset_dir>/<asset_dir>.obj`
  - Do NOT pass unsupported kwargs: `asset_file=`, `pos=`, `rot=`, `density=`, `visual_only=`
- `add_visual_assets(system, [AssetDescriptor(...)], data_dir=offroad_dir)`
  - Use this for Chrono built-in outdoor assets such as `sensor/offroad/tree1.obj`, `rock*.obj`, `cottage.obj`
- `setup_preview_camera(manager, attach_body, target_pos, cam_pos, up_direction, *, update_rate, ...)`
  - Passing a variable named `cam_body` positionally is fine, but the keyword name is `attach_body`, not `cam_body`
  - Do NOT pass legacy kwargs such as `system=`, `offset=`, `target=`, `resolution_width=`, or `horizontal_fov=`
  - Relative `output_root="cam"` resolves against the running script directory

**Banned inline definitions**: Do NOT define `load_convex_hulls`, `transform_vertices`, `add_collision_from_json`, `add_collision_from_decomposition`, `add_collision_debug_visuals`, `quat_from_angle_x`, `quat_from_angle_z`, `quat_mul`, `box_inertia`, `write_placement_csv`, `write_contacts_csv`, `write_links_csv`, or `ContactReporter` in the generated `simulation.py`. All of these are provided by `chrono_code.utils.scene_assets`. Camera setup is provided by `chrono_code.utils.setup_preview_camera` — do NOT construct `ChCameraSensor` or filters manually.

## Asset API Dispatch (MANDATORY — apply BEFORE writing any loader code)

For every entry in `plan.assets[]`, inspect its `filename` and pick the
loader API by the table below. **Do NOT** treat all `type=mesh` entries
uniformly — `create_asset_body` only works for the strict subdir layout,
and silently produces a missing-file path on any other layout (the
`LoadWavefrontMesh` warning is non-fatal; the empty mesh later SIGSEGVs
inside VSG `Render()`).

| filename pattern                                          | layout                | required loader                                                                  |
|-----------------------------------------------------------|-----------------------|----------------------------------------------------------------------------------|
| `data/scene/<n>/<n>.obj` + `<n>_convex.json`              | subdir + convex JSON  | `create_asset_body(asset_dir=<n>, assets_base_dir="data/scene", ...)` (Scenario 1) |
| `sensor/offroad/<n>.obj`                                  | flat chrono builtin   | `AssetDescriptor` + `add_visual_assets(data_dir=<chrono_data>/sensor/offroad)` (Scenario 2) |
| `models/<n>.obj` (chrono builtin, e.g. `coords`, `cube`, `sphere`, `cylinderY`, `cylinderZ`, `semicapsule`, `red_teapot`, `lime_bunny`) | flat, **no** convex JSON, **no** subdir | **NOT** `create_asset_body`. Either drop the entry (see Asset Role Rule below) OR load vis-only via raw `chrono.ChTriangleMeshConnected().LoadWavefrontMesh(filename, True, True)` + `chrono.ChVisualShapeTriangleMesh()` attached to a fixed `chrono.ChBody`. |
| `sensor/geometries/<n>.obj` (`box`, `cube`, `sphere`)     | flat geometric primitive | Replace with the procedural primitive: `chrono.ChVisualShapeBox/Sphere/Cylinder`. Loading a 1 KB unit-cube OBJ when `ChVisualShapeBox` exists is over-engineering. |
| `data/robot/<n>/...`                                      | (robot package)        | `read_skill('robot/<robot_name>')` for the wrapper recipe — do NOT load via `create_asset_body` |

**Hard test before calling `create_asset_body`**: the asset's `filename`
must contain a directory whose name equals the basename. I.e.
`<base>/<dir>/<dir>.obj` where `<dir>` matches the `.obj` stem. If
`os.path.basename(os.path.dirname(filename)) != os.path.splitext(os.path.basename(filename))[0]`,
`create_asset_body` is the wrong API. This check rules out
`data/models/coords.obj` (parent dir `models` ≠ stem `coords`),
`data/models/cube.obj`, all `sensor/geometries/*.obj`, etc.

**Asset Role Rule**: when an entry's purpose is an *orientation indicator*
(name matches `coords` / `axes` / `reference_frame` / contains "axis" or
"orientation" in its description), do **not** load it as a mesh body at
all. The correct realization is `vis.AddGrid(spacing_x, spacing_y, nx, ny,
ChCoordsys, ChColor)` in the VSG visualization setup — VSG ships a real
3-axis grid that conveys orientation by construction. Loading a 6.8 KB
`coords.obj` axis mesh as a `ChBody` is over-engineering even when the
file path is correct: it adds collision overhead, requires a contact
material, and clutters the physics scene with a non-physical reference.
Drop the entry from `assets[]` consumption (the plan-level entry can stay
— this rule only governs loader code emission) and add `vis.AddGrid(...)`
during VSG setup.

**Why this dispatch matters**: catalog rows under `<chrono_data>/models/`
are flat utility/primitive meshes (axis indicators, geometric primitives,
single-purpose helpers) lumped together with placeable assets like
`red_teapot.obj`. The catalog scanner indexes them all as `type=mesh`
without distinguishing layout. Following this dispatch table is the only
way to avoid the `data/models/<n>/<n>.obj` SIGSEGV class.

## Reference Asset Metadata (MANDATORY)

The single source of truth for asset heights, file paths, default face direction, and the global rotation convention is the JSON file:

```
data/scene/assets.json
```

Generated `simulation.py` MUST load this file directly with the standard library and derive `TARGET_HEIGHTS` from it. Do **NOT** hardcode the height table.

```python
with open(os.path.join(ASSETS_DIR, "assets.json")) as f:
    METADATA = json.load(f)

TARGET_HEIGHTS = {name: a["target_height"] for name, a in METADATA["assets"].items()}
```

Rules:

- If an asset in the plan matches a key under `METADATA["assets"]`, the generated code MUST use the exact `target_height` from `assets.json`.
- When computing uniform mesh scale, always use `scale_factor = TARGET_HEIGHTS[name] / raw_size[height_axis]` for these known assets.
- Only fall back to plan-provided `ideal_height` for assets not present in `assets.json`.
- If both the plan and `assets.json` provide a height for the same asset, `assets.json` wins.
- The `convention` block of `assets.json` is authoritative: all OBJ files on disk are baseline **+X facing, +Z up**. Any non-zero `(deg_x, deg_z)` in `ASSET_ROTATION` is a runtime rotation FROM that base.

**`target_height` semantics — read carefully.** The value is the **full
mesh bounding-box z-extent** that the asset should reach after scaling.
It is NOT a sub-feature like seat height, tabletop height, or "standing
height of a humanoid up to the eye." Concretely:

- Office chair: target_height = full chair height including back (e.g.
  ~0.85 m). The seat will land at ~0.45 m by mesh proportion. Setting
  `target_height = 0.45` to mean "seat height" instead crushes the
  whole chair to 45 cm tall and (because scaling is uniform) shrinks
  its footprint to ~30 cm × 30 cm — visibly tiny next to a 75 cm desk.
- Computer table: target_height = top of the tabletop (e.g. ~0.75 m).
- Cottage / building: target_height = roof ridge height.

If `assets.json` already provides a `target_height`, USE IT — it has been
verified against the mesh AABB. Do not invent a different "more
intuitive" number.

**`scale_factor` and `target_heights[asset_dir]` are mutually exclusive.**
`create_asset_body` raises `ValueError` if both are passed for the same
asset. Pass exactly one — prefer `target_heights` since the function then
reads the real mesh AABB and avoids hardcoded `NATIVE_H` guesses.

## Asset Discovery

**Do NOT hardcode asset paths or asset tables.** Use the file exploration tools to discover available assets at runtime:

1. Call `find_files("*.obj", "data/scene")` to locate all OBJ mesh files
2. Call `list_directory("data/scene")` to see the directory structure
3. Each asset lives in `data/scene/<asset_name>/<asset_name>.obj`
4. Each asset has a pre-computed convex decomposition: `data/scene/<asset_name>/<asset_name>_convex.json`

Include the discovered paths in your plan. If a `*_convex.json` is missing for an asset, skip collision for that asset and log a warning.

**Path resolution**: The generated script is written to `outputs/simulation.py`. Resolve asset paths from the repository root, then convert them to absolute paths immediately:

```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "data" / "scene").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start_dir}")

REPO_ROOT = find_repo_root(SCRIPT_DIR)
ASSETS_DIR = str((REPO_ROOT / "data" / "scene").resolve())
OUTPUT_DIR = str(SCRIPT_DIR.resolve())
```

Rules:

- `ASSETS_DIR` must point to the repo-global `data/scene` directory, not a path relative to the current shell.
- Do not hardcode a fixed number of `..` hops; the runtime file may live in `outputs/` or a history subdirectory.
- Prefer `pathlib.Path` for repo-root path assembly, or produce the equivalent absolute string path with `os.path.abspath(...)`.
- Never use `os.getcwd()` or bare relative paths.

## Workflow

### 1. System Setup

**Do NOT call `chrono.SetChronoDataPath()`** — the default path set by `import pychrono.core` is correct. Overriding it with a project-local path (e.g. `REPO_ROOT / "data"`) causes a segfault in `vis.Initialize()` because PyChrono does raw string concatenation to locate VSG shaders, and `pathlib.Path / ""` strips the trailing `/`, producing paths like `.../datavsg/shaders/...` instead of `.../data/vsg/shaders/...`. The C++ layer does not handle missing shaders gracefully — it dereferences a null pointer.

**Without robot (static scene)**:

```python
system = chrono.ChSystemNSC()
system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)
```

Use `ChContactMaterialNSC` with friction `0.95` and restitution `0.0`. Preferred timestep: `0.001`.

**With robot (e.g. Go2 quadruped)**: Use `ChSystemSMC` instead — see the `robot/go2_quadruped` skill for the full system setup including solver and contact parameters. The `create_asset_body` function and convex hull collision work identically with both system types; only the contact material class changes (`ChContactMaterialSMC` vs `ChContactMaterialNSC`).

### 1.5. Ground Plane (REQUIRED — `SetFixed(True)` AND explicit `ChVisualMaterial`)

Every scene MUST have a ground body that satisfies **two** independent requirements. Both are silent killers when missed: the simulation runs without error, but the result is broken.

**Requirement A — `SetFixed(True)` (otherwise the ground falls).** `ChBodyEasyBox` defaults to dynamic, so an 8x8x0.1 m / density-1000 ground weighs ~6400 kg and falls under gravity together with every asset standing on it.

**Requirement B — explicit `ChVisualMaterial` on `GetVisualShape(0)` (otherwise the sensor camera does not render it).** `ChBodyEasyBox` creates a `ChVisualShape` with **0 materials** by default. The Chrono::Sensor OptiX renderer requires >=1 material per shape and silently skips any shape that has none.

The full recipe — copy verbatim:

```python
mat = make_contact_material(friction=0.95, restitution=0.0)
ground = chrono.ChBodyEasyBox(8.0, 8.0, 0.1, 1000, True, True, mat)
ground.SetName("ground")
ground.SetPos(chrono.ChVector3d(0, 0, -0.05))   # top surface at z=0
ground.SetFixed(True)                           # Requirement A: do not omit

# Requirement B: explicit ChVisualMaterial so sensor camera renders the ground
ground_vis_mat = chrono.ChVisualMaterial()
ground_vis_mat.SetDiffuseColor(chrono.ChColor(0.6, 0.6, 0.6))
ground_vis_mat.SetSpecularColor(chrono.ChColor(0.2, 0.2, 0.2))
ground_vis_mat.SetRoughness(0.7)
ground.GetVisualShape(0).AddMaterial(ground_vis_mat)

system.AddBody(ground)
```

Constraints:

- `SetFixed(True)` MUST appear before `system.AddBody(ground)`.
- The visual material block MUST use `AddMaterial` (not `SetMaterial(0, ...)`).
- Ground name MUST be the literal string `"ground"`.
- Top surface of the ground sits at `z = +0.05` when `SetPos(z=-0.05)`.
- For SMC-based scenes, use the same recipe but pass an `SMC` material.

#### 1.5b — Indoor floor and walls (same support-surface rule)

When the scene is an indoor room (not the outdoor ground recipe above),
the floor and any wall furniture might lean against MUST also be a
**collidable, fixed body**. A common bug: agent constructs the floor
with `chrono.ChBody()` + `ChVisualShapeBox` only and forgets a
collision shape — the floor is then **invisible to the physics engine**
and every chair/table free-falls forever.

If you cannot use `ChBodyEasyBox` (e.g. you need named bodies or
non-default visual setup), the manual recipe MUST include both a
collision shape AND `EnableCollision(True)`:

```python
floor_mat = make_contact_material(friction=0.6, restitution=0.0, method="NSC")

room_floor = chrono.ChBody()
room_floor.SetName("room_floor")
room_floor.SetFixed(True)
room_floor.SetPos(chrono.ChVector3d(0.0, 0.0, 0.0))

# Visual
floor_vis = chrono.ChVisualShapeBox(FLOOR_X, FLOOR_Y, FLOOR_Z)
floor_vis.SetColor(chrono.ChColor(0.75, 0.75, 0.72))
room_floor.AddVisualShape(floor_vis)

# Collision — REQUIRED, not optional. Furniture cannot rest on a
# visual-only floor; gravity will pull it through.
floor_coll = chrono.ChCollisionShapeBox(floor_mat, FLOOR_X, FLOOR_Y, FLOOR_Z)
room_floor.AddCollisionShape(floor_coll)
room_floor.EnableCollision(True)

system.AddBody(room_floor)
```

The same rule applies to walls if the plan expects furniture to lean
against them or a robot to push furniture into them — pair every
`ChVisualShapeBox` with a `ChCollisionShapeBox` and `EnableCollision(True)`.

**Hard invariant**: any body listed as a support surface or boundary
in the plan MUST have `body.GetCollisionModel()` non-empty AND
`body.IsCollisionEnabled()` true at sim start. The deterministic review
checks this — it WILL fail the step if a fixed body has dynamic objects
nearby that never make contact with it.

### 2. Asset Rotation Convention (MANDATORY)

All scene assets under `data/scene/` are baseline **+X facing, +Z up**. The simulation also uses **Z-up** (`gravity_axis: -z`). The authoritative declaration lives in the `convention` block of `data/scene/assets.json`.

You **MUST** define an `ASSET_ROTATION` dictionary. The default for every asset is `(0.0, 0.0)` (no extra rotation). Only set non-zero values when the plan requires a different yaw.

```python
# Layer 1: deg_x — tilt around +X (almost always 0 for office props)
# Layer 2: deg_z — scene-specific yaw FROM the +X-facing baseline
#
# facing -> deg_z mapping (matches assets.json convention):
#   +X -> 0,  +Y -> 90,  -X -> 180,  -Y -> -90

ASSET_ROTATION = {
    # "asset_name": (deg_x, deg_z),
    # Example:
    # "asset1": (0.0, -90.0),  # rotate from +X baseline to face -Y
}
```

**Height axis**: For native Z-up assets, the raw Z axis is the height axis. Default: `height_axis = 2`.

### 3. Body Creation — Use `create_asset_body()` (MANDATORY)

All scene assets MUST be created through `create_asset_body()` from `chrono_code.utils.scene_assets`. **Do NOT scatter body creation logic across multiple ad-hoc blocks or reimplement the mesh loading/transform pipeline inline.**

> **Visual/collision sync invariant** — CRITICAL
>
> Both the visual mesh AND every convex hull share one body-local frame.
> `create_asset_body()` enforces this by passing the same `scale_rot` matrix
> to both the visual `mesh.Transform()` and `add_collision_from_decomposition()`.
> Do not break this invariant by calling `body.SetRot()` after creation — bake
> orientation into `ASSET_ROTATION` instead.

#### Pretransformed Convex Assets

Some convex JSONs have vertices already in the final local coordinate frame. Pass these as `pretransformed_assets`:

```python
PRETRANSFORMED_CONVEX_ASSETS = {
    # Add asset names whose *_convex.json has pre-transformed vertices
}
```

#### Example: Creating Assets

```python
mat = make_contact_material(friction=0.95, restitution=0.0, method="NSC")

body_a, size_a = create_asset_body(
    system,
    name="<asset_instance_name>",
    asset_dir="<asset_directory_name>",
    assets_base_dir=ASSETS_DIR,
    target_heights=TARGET_HEIGHTS,
    position=(0.0, 0.0, 0.0),
    asset_rotation=ASSET_ROTATION,
    pretransformed_assets=PRETRANSFORMED_CONVEX_ASSETS,
    contact_material=mat,
    fixed=False,
    mass=10.0,
)
```

`create_asset_body()` handles:
- Mesh loading via `ChTriangleMeshConnected.LoadWavefrontMesh()`
- Two-step transform: centre at origin, then scale + rotation
- Visual shape creation
- Convex hull collision from `*_convex.json`
- Support-surface Z placement (body bottom rests on `position[2]`)
- Adding the body to the system

Returns `(body, transformed_size)` where `transformed_size = [sx, sy, sz]`.

#### Material handling

Create materials separately, never retrieve from body. Use SMC when a robot is present, NSC otherwise:

```python
# With robot (SMC):
mat = chrono.ChContactMaterialSMC()
mat.SetFriction(0.9)
mat.SetRestitution(0.01)
mat.SetGn(60.0)
mat.SetKn(2e5)

# Without robot (NSC) — or use the helper:
mat = make_contact_material(friction=0.95, restitution=0.0)
```

### 4. Asset Placement

Every asset has exactly one **support surface** — the surface it physically rests on. Two cases:

| `placed_on` | Support surface          | Placement helper                     |
|-------------|--------------------------|--------------------------------------|
| `"ground"`  | the world ground plane   | `FootprintRegistry` (one per scene)  |
| `<parent_name>` | another asset's top    | `SurfaceStack` (one per parent)      |

`placed_on` comes from the plan / `step_context.json` (e.g. monitor and laptop both have `placed_on: "computer_table"`). When the field is absent, default to `"ground"`.

#### What `position[2]` means in `create_asset_body`

`create_asset_body` interprets `position[2]` as **the world Z where the body's bottom should rest** — *not* the body's centre. It then computes the body-centre Z internally so that the visual bottom lands on `position[2]`. Therefore:

- `placed_on == "ground"` → `position = (x, y, GROUND_TOP_Z)`
- `placed_on == <parent>`  → `position = (x, y, parent_top_z + SPAWN_GAP)` where `SPAWN_GAP ≈ 0.005–0.01` m so the prop drops a hair onto the surface (avoids initial-frame penetration impulses).

Never compute the body centre yourself. Never override `body.SetPos(..., cfg["position"][2])` after `create_asset_body` has run — that breaks the bottom-rests-on-surface invariant. If you need to rewrite XY (e.g., after the registry resolves an overlap), preserve the Z that `create_asset_body` computed: `body.SetPos(chrono.ChVector3d(nx, ny, body.GetPos().z))`.

#### Default dynamics

- **All floor and stacked assets default to `fixed=False`** so gravity settles them onto their support surface. Pin to `fixed=True` only when the plan explicitly requires it (walls, fixtures, the ground itself).
- Set mass based on realistic object weight.

#### MANDATORY overlap prevention (both surfaces)

Every XY placement **must** go through a registry. **Do not reimplement.**

| Symptom you'll see if you skip this | Underlying mistake |
|---|---|
| Two floor assets clip each other and explode at sim start | No `FootprintRegistry`, or floor asset bypassed it. |
| **Stacked prop teleports off the parent and floats / falls in mid-air** | **Stacked prop sent through the floor registry instead of a `SurfaceStack`.** The floor registry has no notion of "must stay inside this rectangle"; it shoves the prop sideways into clear floor space. |
| Stacked prop spawns clipping its sibling | Skipped `SurfaceStack` for one of the children. |

#### Single example covering both cases

```python
from chrono_code.utils.scene_placement import FootprintRegistry, SurfaceStack

GROUND_TOP_Z = 0.05    # top face of the 0.05 m thick ground slab
SPAWN_GAP    = 0.01    # how far above its support a prop spawns (avoids initial penetration)

# Floor registry — one per scene.
floor = FootprintRegistry(room_half=4.0, margin=0.03)

# Surface stacks — one per parent that has children. Built lazily, AFTER the
# parent has its final world XY (so the AABB read is meaningful).
surfaces: dict[str, SurfaceStack] = {}

mat = make_contact_material(friction=0.95, restitution=0.0)

ASSET_CONFIGS = [
    # Order: parents before children, then by decreasing footprint area within each.
    {"name": "computer_table",  "asset_dir": "computer_table",
     "placed_on": "ground",          "preferred_xy": (0.0, 0.0),
     "fixed": True,  "mass": 30.0},

    {"name": "office_chair",    "asset_dir": "office_chair",
     "placed_on": "ground",          "preferred_xy": (1.0, 0.0),
     "fixed": False, "mass": 15.0},

    {"name": "ultrawide_monitor", "asset_dir": "ultrawide_monitor",
     "placed_on": "computer_table",  "preferred_xy": (-0.10, 0.0),
     "fixed": False, "mass": 8.0},

    {"name": "macbook_pro_m3_16_inch_2024", "asset_dir": "macbook_pro_m3_16_inch_2024",
     "placed_on": "computer_table",  "preferred_xy": (0.30, 0.0),
     "fixed": False, "mass": 2.0},
]

bodies: dict[str, "chrono.ChBody"] = {}

for cfg in ASSET_CONFIGS:
    parent = cfg["placed_on"]
    px, py = cfg["preferred_xy"]

    # 1. Resolve the support-surface Z and choose the right registry.
    if parent == "ground":
        support_z = GROUND_TOP_Z
        registry  = floor
    else:
        if parent not in surfaces:
            surfaces[parent] = SurfaceStack.from_body(parent, bodies[parent])
        registry  = surfaces[parent]
        support_z = registry.surface_top_z + SPAWN_GAP

    # 2. Build the body. position[2] is the body-bottom Z — create_asset_body
    #    derives the centre internally.
    body, tsize = create_asset_body(
        system,
        name=cfg["name"],
        asset_dir=cfg["asset_dir"],
        assets_base_dir=ASSETS_DIR,
        target_heights=TARGET_HEIGHTS,
        position=(px, py, support_z),
        asset_rotation=ASSET_ROTATION,
        pretransformed_assets=PRETRANSFORMED_CONVEX_ASSETS,
        contact_material=mat,
        fixed=cfg.get("fixed", False),
        mass=cfg.get("mass", 5.0),
        scale_multiplier=cfg.get("scale_multiplier", 1.0),
    )

    # 3. Resolve XY through the chosen registry, then move the body — keeping
    #    the Z that create_asset_body computed.
    nx, ny = registry.place(
        size_x=tsize[0], size_y=tsize[1],
        preferred_x=px, preferred_y=py,
    )
    body.SetPos(chrono.ChVector3d(nx, ny, body.GetPos().z))
    bodies[cfg["name"]] = body
```

Rules:

1. **Imports are fixed**: `from chrono_code.utils.scene_placement import FootprintRegistry, SurfaceStack`.
2. **One `FootprintRegistry`** for ground-level assets; **one `SurfaceStack` per parent** that holds children (built via `SurfaceStack.from_body(parent_name, parent_body)` after the parent body has been moved to its final XY).
3. **Iterate `ASSET_CONFIGS` so parents come before their children.** A child may depend on its parent's AABB, which only exists after the parent has been built.
4. **Within each `placed_on` group, order by decreasing footprint area** so the largest reservation lands at the preferred XY.
5. **Never call `placement.place` before `create_asset_body`** — you need `transformed_size`.
6. **Z is `create_asset_body`'s job.** When you `SetPos` after the registry, copy `body.GetPos().z`; do not pass `cfg["position"][2]` raw (that puts the body *centre* at the bottom-Z and the prop sinks halfway into its support).
7. **Stacked props go through `SurfaceStack`, not `FootprintRegistry`.** The floor registry has no rectangular-bound check and will push them off the parent.
8. **If the scene contains a robot or vehicle, its spawn XY must go through the floor registry** — see the next subsection.

#### Placing a robot/vehicle through the registry

The registry offers two entry points:

- `registry.place(size_x, size_y, preferred_x, preferred_y)` — reserves a footprint from explicit dimensions. Use this **before** constructing an articulated robot whose limbs are separate bodies; `place_body()` would only move the base and tear the kinematic chain.
- `registry.place_body(body, preferred_x, preferred_y)` — reserves from the body's **collision AABB (convex hulls)** and moves the body there. Correct for single-rigid-body props and single-body vehicles, but **not** for multi-body robots.

Canonical pattern for a quadruped/manipulator robot (articulated, multi-body):

```python
from chrono_code.utils.scene_placement import FootprintRegistry

placement = FootprintRegistry(room_half=4.0, margin=0.03)

# 1. Place all scene assets first (largest footprint first).
for cfg in ASSET_CONFIGS:
    body, tsize = create_asset_body(system, ...)
    nx, ny = placement.place(tsize[0], tsize[1], cfg["position"][0], cfg["position"][1])
    body.SetPos(chrono.ChVector3d(nx, ny, body.GetPos().z))

# 2. Reserve the robot footprint BEFORE constructing it, using a conservative
#    bounding-box estimate for the robot + its legs/arms at spawn pose.
#    (Go2 trunk is ~0.38 x 0.09 m, but with legs spread the AABB is ~0.6 x 0.4 m.
#    Round up so contact margins don't nibble into neighbors.)
ROBOT_FOOTPRINT_X, ROBOT_FOOTPRINT_Y = 0.6, 0.4
spawn_x, spawn_y = placement.place(
    size_x=ROBOT_FOOTPRINT_X,
    size_y=ROBOT_FOOTPRINT_Y,
    preferred_x=0.0,     # plan-requested robot origin
    preferred_y=0.0,
)

# 3. Pass the resolved XY into the robot's initial_state — do NOT hardcode.
go2_init = chrono.ChFramed(
    chrono.ChVector3d(spawn_x, spawn_y, 0.55),
    chrono.QuatFromAngleZ(0.0),
)
go2 = Go2Robot(system, initial_state=go2_init)
```

Why this matters: without this step, a planner that deliberately places a prop "near center for the robot to approach" (and a codegen pass that hardcodes `ChVector3d(0, 0, 0.55)` for the robot) will silently spawn the robot **inside** that prop. SMC contact constraints then pin the robot; video shows it stationary; review fails for reasons that look like control-loop bugs. The real root cause is the skipped overlap check.

For single-rigid-body props and some wrapper-managed vehicles that expose a single chassis body, `placement.place_body(body, px, py)` is the shorter form and uses the actual convex-hull collision AABB for the check.

### 5. Outdoor Props (trees / rocks / cottage from `sensor/offroad/`)

Outdoor prop scatter for vehicle and robot scenes reuses the same
`chrono_code.utils.scene_assets` utilities as the indoor workflow
above — `AssetDescriptor`, `add_visual_assets`, `make_contact_material`,
`ensure_convex_json` — but with asset paths and collision decisions
tuned to the outdoor asset library. Do NOT use `create_asset_body` for
these; it targets repo-local `data/scene/<name>/<name>.obj` layouts,
not Chrono's built-in outdoor OBJs.

#### Asset directory

Chrono ships outdoor assets under `<ChronoDataPath>/sensor/offroad/`.
Resolve once:

```python
OFFROAD_DIR = os.path.join(chrono.GetChronoDataPath(), "sensor", "offroad")
```

Available: `tree1.obj` .. `tree3.obj` (native H ≈ 3.9 m), `bush1.obj` /
`bush2.obj` (H ≈ 1.0 m), `rock1.obj` .. `rock5.obj` (H 1.3 – 1.7 m),
`cottage.obj` (native H 14.7 m → typical scale `3.24 / 14.7 ≈ 0.22`
for a ~3.24 m tall model).

#### Collision decisions per asset family

Every outdoor prop falls into one of three families. Pick by asset type,
not by guess:

```python
descriptors = [
    # ── Family A: decorative foliage (trees, bushes) ──────────────────
    # Default fixed=True, default collision=False. Vehicle/robot passes
    # through them. Per-leaf collision would be extremely expensive and
    # not physically meaningful.
    AssetDescriptor(obj_path="tree1.obj", position=(8, 9, 0), yaw_deg=0,
                    scale=1.0, name="tree1"),
    AssetDescriptor(obj_path="bush1.obj", position=(7, 7, 0), yaw_deg=15,
                    scale=1.0, name="bush1"),

    # ── Family B: static obstacles (fixed rocks, cottage) ─────────────
    # Convex-decomposed collision via VHACD. add_visual_assets calls
    # ensure_convex_json internally on first run and caches the JSON
    # next to the OBJ — no manual decomp step.
    AssetDescriptor(obj_path="rock1.obj", position=(15, 16, 0), yaw_deg=10,
                    scale=1.0,
                    collision=True, collision_method="convex",
                    friction=0.8, name="rock1"),
    AssetDescriptor(obj_path="cottage.obj", position=(-18, -20, 0),
                    yaw_deg=90, scale=3.24 / 14.7,
                    collision=True, collision_method="convex",
                    friction=0.8, name="cottage"),

    # ── Family C: movable / dynamic obstacles (pushable boulders) ─────
    # Single outer convex hull, NOT a compound. Bullet's cbtCompoundShape
    # has a broadphase-AABB bug on dynamic bodies that silently makes
    # multi-hull compounds invisible to narrowphase.
    # collision_method="single_convex" attaches ONE
    # ChCollisionShapeConvexHull directly and avoids the compound wrapper.
    AssetDescriptor(obj_path="rock2.obj", position=(5, 10, 0.5),
                    yaw_deg=0, scale=1.0,
                    fixed=False, mass=500.0,
                    collision=True, collision_method="single_convex",
                    friction=0.8, name="rock2_movable"),
]

bodies = add_visual_assets(system, descriptors, data_dir=OFFROAD_DIR)
```

Hard rules (outdoor):

- Default to Family A (no collision) for ALL trees and bushes. Do NOT
  generate `collision=True` for foliage without a concrete reason.
- Default to Family B (`collision_method="convex"`) for rocks and the
  cottage when the plan does not say the props move.
- Only use Family C (`fixed=False, collision_method="single_convex"`)
  when the user explicitly wants pushable props. Dynamic props REQUIRE
  the support-plane recipe below when used on SCM terrain.
- Do NOT hand-write `chrono.ChBody() + ChVisualShapeTriangleMesh(...)`
  loops for outdoor props. The SWIG binding is fragile with inline
  mesh/shape lifetime, and you duplicate the scale / yaw / Z-up /
  material logic already inside `add_visual_assets`.

#### Collision family filtering (optional)

After `add_visual_assets` returns, optionally group collidable props
into a dedicated Bullet collision family so they don't jitter against
each other:

```python
PROP_FAMILY = 2   # any unused slot in [0, 15]

for body in bodies:
    if body is None:
        continue
    name = body.GetName() or ""
    if name.startswith("rock") or name == "cottage":
        cm = body.GetCollisionModel()
        if cm is not None:
            cm.SetFamily(PROP_FAMILY)
            cm.DisallowCollisionsWith(PROP_FAMILY)
```

#### Support plane (REQUIRED for dynamic props on SCM terrain)

`SCMTerrain` is NOT a general-purpose rigid ground — it only applies
forces under ray-casts from the active-domain body with an attached
chassis. A fixed prop sits where you place it, but a dynamic prop
(Family C) free-falls through `z=0` because SCM never generates a
contact force for it. Without a support plane, `fixed=False` props
disappear under the terrain within the first few steps.

Add a hidden rigid support box any time the scene contains dynamic
props on SCM. Keep it out of the tire and chassis families so the
vehicle still rides on SCM, not on the support box:

```python
from chrono_code.utils import make_contact_material

TIRE_FAMILY    = 1
CHASSIS_FAMILY = 0
SUPPORT_FAMILY = 4

support_mat = make_contact_material(
    friction=0.9, restitution=0.01, method="SMC", young_modulus=2e7,
)
support = chrono.ChBodyEasyBox(120.0, 120.0, 0.2, 1000, False, True, support_mat)
support.SetName("asset_support_ground")
support.SetPos(chrono.ChVector3d(0, 0, -0.1))    # just below z=0
support.SetFixed(True)
support.EnableCollision(True)
support_cm = support.GetCollisionModel()
support_cm.SetFamily(SUPPORT_FAMILY)
support_cm.DisallowCollisionsWith(TIRE_FAMILY)
support_cm.DisallowCollisionsWith(CHASSIS_FAMILY)
# also disallow the chassis sub-body family if add_collision_via_subbodies
# is used (veh/wheeled_vehicle's default self_family=3).
support_cm.DisallowCollisionsWith(3)
system.AddBody(support)
```

Omit the support plane when:
- There are no dynamic props (all descriptors have `fixed=True`).
- The terrain is `veh.RigidTerrain` (rigid ground already supports
  dynamic bodies; the support plane would double up).

#### Ordering requirements

Two independent ordering constraints apply — violate either and scene
props silently disappear from the VSG live window.

**(a) System ordering — wrapper vehicle first, then scene.**
For wrapper-managed vehicle scenes (`veh.HMMWV_Full` etc.), the system
already exists by the time this section's code runs — it came from
`system = hmmwv.GetSystem()` after `hmmwv.Initialize()`. Do NOT create
a new `ChSystem` here. See `veh/wheeled_vehicle`'s "Hard rules: system
ownership and construction order" for the full required sequence.

**(b) `add_visual_assets` MUST be called BEFORE `vis.Initialize()`.**
The VSG visualizer (both `chronovsg.ChVisualSystemVSG` and
`veh.ChWheeledVehicleVisualSystemVSG`) snapshots the attached system's
scene graph inside `Initialize()`. Bodies added to the system AFTER
`vis.Initialize()` do NOT appear in the live VSG window — even though
`system.GetNumBodies()` increases and `sensor/offroad/` meshes load
fine. The sensor-camera recorder still sees them (it re-reads the body
list each frame), so a common symptom is: **"trees show up in the
mp4 but not in the live VSG window."**

Correct skeleton for a vehicle-plus-scene simulation:

```python
# 1. Vehicle owns the system
hmmwv = veh.HMMWV_Full()
# ... configure ...
hmmwv.Initialize()
system = hmmwv.GetSystem()

# 2. Terrain on that system
terrain = veh.SCMTerrain(system)
# ... configure + Initialize ...

# 3. Scene bodies on that system — BEFORE vis.Initialize
add_visual_assets(system, descriptors, data_dir=OFFROAD_DIR)

# 4. NOW create + initialize the visualizer
vis = veh.ChWheeledVehicleVisualSystemVSG()
vis.AttachVehicle(hmmwv.GetVehicle())
vis.AttachTerrain(terrain)
vis.Initialize()   # ← snapshots scene graph; nothing added after this
                   #    renders in the live window.
```

If the plan schedules scene props as a step that runs *after* step 1
has already created+initialized `vis`, use `edit_file` to move the
`add_visual_assets` call BEFORE `vis.Initialize()`, not after. Do not
"just append" at the end of the file.

### 6. VSG Visualization

VSG setup is in the `core/scene` skill. Use `chronovsg.ChVisualSystemVSG` — do NOT use Irrlicht.

### 7. Sensor Cameras — REQUIRED (use `setup_preview_camera`)

**This step is MANDATORY. Do NOT skip it.** `vis.AddCamera` only sets the VSG window viewpoint — it does NOT produce any camera output, video, or frames. You MUST also create sensor cameras via `setup_preview_camera` to actually record the simulation. Every scene needs BOTH: `vis.AddCamera` (§6 VSG) AND `setup_preview_camera` calls (this section).

Set up 5 sensor cameras for static scenes (4 orthogonal + 1 top-down). This is the `static_5` camera layout. Do NOT construct `sens.ChCameraSensor`, `ChFilterVisualize`, `ChFilterSave`, or `ChFilterRGBA8Access` manually.

#### Sensor Manager + Lighting

```python
timestep = 0.001
sensor_target = chrono.ChVector3d(0.0, 0.0, 0.5)

manager = sens.ChSensorManager(system)

bg = sens.Background()
bg.mode = sens.BackgroundMode_SOLID_COLOR
bg.color_zenith = chrono.ChVector3f(0.0, 0.0, 0.0)
bg.color_horizon = chrono.ChVector3f(0.0, 0.0, 0.0)
manager.scene.SetBackground(bg)
manager.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))
manager.scene.AddPointLight(
    chrono.ChVector3f(2, 2.5, 100), chrono.ChColor(1.0, 1.0, 1.0), 500.0)
manager.scene.AddPointLight(
    chrono.ChVector3f(-2, -2, 3), chrono.ChColor(0.8, 0.8, 0.8), 200.0)
```

#### Creating the 5 Cameras (4 Cardinal + Top-Down)

```python
X, Y, Z = 2.5, 2.5, 2.0
Z_top = 3.0   # top-down camera height

camera_positions = {
    "camera+x": chrono.ChVector3d(+X, 0.0, +Z),
    "camera-x": chrono.ChVector3d(-X, 0.0, +Z),
    "camera+y": chrono.ChVector3d(0.0, +Y, +Z),
    "camera-y": chrono.ChVector3d(0.0, -Y, +Z),
    "top_down": chrono.ChVector3d(0.0, 0.0, +Z_top),
}

for name, eye_pos in camera_positions.items():
    cam_body = chrono.ChBody()
    cam_body.SetFixed(True)
    cam_body.SetPos(eye_pos)
    system.AddBody(cam_body)

    setup_preview_camera(
        manager, cam_body,
        target_pos=sensor_target,
        cam_pos=eye_pos,
        up_direction=chrono.ChVector3d(0, 0, 1),
        update_rate=1.0 / timestep,
        name=name,
    )
```

All 5 views must be created for static scenes. Do not drop cameras.

#### Simulation Loop

```python
print("Simulation running... (close the window or press Ctrl+C to exit)")
step_count = 0
while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    manager.Update()
    system.DoStepDynamics(timestep)
    step_count += 1

    if step_count % 100 == 0 or step_count == 1:
        t = system.GetChTime()
        print(f"Step {step_count}: t={t:.4f}s")

write_placement_csv(system, output_dir=OUTPUT_DIR)
write_contacts_csv(system, output_dir=OUTPUT_DIR)
write_links_csv(system, output_dir=OUTPUT_DIR)
print("Done.")
```

`setup_preview_camera` installs cleanup hooks, but explicit `recorder.close()` in `finally` is still recommended when you keep the returned recorders.

### 8. Placement Validation CSV (REQUIRED)

After the simulation loop, call the utility writers:

```python
write_placement_csv(system, output_dir=OUTPUT_DIR)
write_contacts_csv(system, output_dir=OUTPUT_DIR)
write_links_csv(system, output_dir=OUTPUT_DIR)
```

These are imported from `chrono_code.utils.scene_assets`. Do NOT redefine `ContactReporter`, `write_placement_csv`, `write_contacts_csv`, or `write_links_csv` inline.

`write_links_csv` records every body-pair joined by a `ChLink*` (revolute, prismatic, fixed-mate, motor, ...). The downstream `no_interpenetration` reviewer reads this to skip joint pairs whose AABB overlap is a kinematic-constraint artifact, not a real geometry bug — without it, articulated robots / vehicles flag every adjacent-link pair as "clipping".

## Pitfalls

- Always store `GetBoundingBox()` and `GetTotalAABB()` in a variable before reading `.min`/`.max` — they return C++ temporaries.
- Never resolve asset paths from `os.getcwd()`.
- Never apply non-uniform scaling to these assets.
- Never decompose an uncentered mesh — `create_asset_body` handles this correctly; do not bypass it.
- Never mix `ChSystemNSC` with `ChContactMaterialSMC`.
- Never start desk props with overlapping XY footprints — use `FootprintRegistry`.
- Do not use `coacd` or `trimesh` at runtime — convex hulls are pre-computed in JSON files.
- Do NOT use Irrlicht — it is not supported. Use VSG for visualization.
- Always call `manager.Update()` inside the simulation loop so sensor cameras render.
- Always call `vis.BindAll()` after `vis.Initialize()` for VSG.
- Do not call `body.SetRot()` to rotate an asset after creation; bake rotation into `ASSET_ROTATION` and pass it to `create_asset_body`.
- Do not redefine utility functions inline — import from `chrono_code.utils.scene_assets`.
- **NEVER call `chrono.SetChronoDataPath()`** — the default is correct. Overriding it with a project path missing a trailing `/` causes `vis.Initialize()` to segfault (null shader pointer). `pathlib.Path / ""` does NOT produce a trailing slash.
- **NEVER build a support surface (floor, wall, table-top fixture) as visual-only.** A `ChBody` with only `ChVisualShapeBox` and no `ChCollisionShapeBox` + `EnableCollision(True)` is invisible to physics. Every asset placed on it free-falls. If `placement.csv` shows furniture at `pos_z << 0` with `vel_z` still falling at sim end, the cause is almost always a missing collision shape on the support body — see §1.5b.
- **NEVER pass both `target_heights[asset_dir]` and `scale_factor` to `create_asset_body`.** It now raises `ValueError`. Pick one — prefer `target_heights` so the function reads the real mesh AABB instead of relying on a hardcoded `NATIVE_H` guess.

canonical_examples:

- `demo/scene/custom_assets_scene_cam.py`
- `data/scene/<asset>/<asset>.obj`
- `data/scene/<asset>/<asset>_convex.json`
