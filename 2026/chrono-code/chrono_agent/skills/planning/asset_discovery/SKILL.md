---
name: asset_discovery
description: Mandatory asset discovery protocol for project-local and Chrono built-in assets.
compatibility: pychrono >= 8.0
metadata:
  domain: planning
---
# Skill: Asset Discovery Protocol

## Purpose

Defines the mandatory steps for discovering and selecting external assets before
writing a scene, mbs_in_scene, or sensor-rich plan. Assets are discovered from
two sources: the project-local `data/` directory and the Chrono built-in data
directory. Do not confuse external assets with procedural primitives: simple
geometry, generated boundaries, terrain patches, and fluid domains belong in
`scene_objects[]`, not `assets[]`.

## When This Applies

- plan_type: scene, mbs_in_scene with external loadable resources → MANDATORY.
- Primitive-only scene / mbs_in_scene plans may have an empty `assets[]` as long
  as they have visible `scene_objects[]`.
- Pure-MBS plans with no external mesh, texture, or HDR reference → can be skipped.

## Asset Sources

### Project-local assets (`data/`)

Custom assets stored in the repository:
- `data/scene/<asset_name>/<asset_name>.obj` — scene furniture and props
- `data/robot/<robot_name>/` — robot URDF and mesh files

### Chrono built-in assets

Assets bundled with the PyChrono installation (accessed via
`chrono.GetChronoDataFile()`):
- `sensor/offroad/` — outdoor props (trees, bushes, rocks, cottage)
- `models/` — generic models (teapot, bunny, forklift, chess table, traffic cone, etc.)
- `sensor/geometries/` — basic shapes (box, sphere, cylinder, suzanne)
- `sensor/cones/` — cone markers
- `vehicle/` — vehicle chassis, tires, wheels (for reference scenes)

## Asset Discovery Protocol (in order)

1. **Search project assets**:
   - Call `find_assets("*.obj", source="project")`
   - Call `list_directory("data/scene/")` to see directory structure
2. **Search Chrono built-in assets** (for outdoor props, basic shapes, etc.):
   - Call `find_assets("*.obj", source="chrono")` for full listing, OR
   - Call `list_chrono_assets()` for a category overview, then
   - Call `list_chrono_assets("sensor/offroad")` to drill into a specific category
3. **Inspect metadata** (if present):
   - Call `read_file_content("data/scene/assets.json")` for project asset metadata

Before binding catalog rows, classify each user-mentioned object by construction
source:

- `wrapper_or_vehicle_json` / `imported_mesh_asset`: use this asset discovery
  protocol and put the result in `plan.assets[]`.
- `procedural_primitive`: simple rigid geometry that can be described by basic
  dimensions and material properties. Put it in `scene_objects[]` with
  dimensions and physical parameters.
- `fluid_domain`: SPH/FEA/fluid domains. Put it in `scene_objects[]`.
- `generated_boundary`: FSI containers, BCE walls/floors, terrain patches, and
  other generated support geometry. Put it in `scene_objects[]`.

If no matching catalog asset exists for a simple object, do not drop it and do
not ask the user by default. Use a procedural primitive fallback. Ask only if
the fallback would materially change the user's intent or visual requirements.

## Topology Closure Rule

Asset discovery is only one part of planning. Before submitting a plan, compare
the concrete physical objects in the user request against the union of
`plan.assets[]` and `plan.scene_objects[]`. Every requested object must be
represented in one of those two places.

Use construction-source categories rather than scenario-specific examples:

- simple rigid geometry → `scene_objects[]` as `procedural_primitive`
- generated supports, enclosures, boundaries, and terrain patches →
  `scene_objects[]` as `generated_boundary`
- solver domains such as fluids, deformable regions, or fields →
  `scene_objects[]` as `fluid_domain`
- named robots, vehicles, articulated systems, and complex props with available
  loaders or mesh files → `assets[]`

Do not submit a plan where only the actuated/moving entity is present while the
supporting topology named by the user is missing.

## Filename Convention

Asset filenames in `plan.assets[].filename` use **source-appropriate relative
paths**:

- **Project assets**: project-relative path under `data/scene/...`
- **Chrono built-in assets**: chrono-data-relative path under the Chrono data directory

The chrono-relative format matches `chrono.GetChronoDataFile()` convention.
Code generation uses `_resolve_asset_path()` from `chrono_agent.utils.scene_assets`
which automatically resolves both formats (project data_dir first, then
`chrono.GetChronoDataPath()` fallback).

## Validation Rules

- Skipping required tool calls → plan WILL BE REJECTED.
- Writing "No specific asset files found" or "assume generic assets" for an
  object that truly needs an external asset → REJECTED.
- Listing a path that was never returned by `find_assets`/`find_files`/`list_directory`/`list_chrono_assets` (i.e. invented) → REJECTED.
- Only include assets the user explicitly requested. Do not pad with optional props.
- An empty `assets` section for a scene plan is valid only when the visible
  objects are all represented in `scene_objects[]` as procedural primitives,
  generated boundaries, or domains.

## Implementation Steps Rules

For scene/mbs_in_scene plans, `implementation_steps` must:

- Follow visible-entity placement order derived from `physical_predicates` and
  `scene_predicates`.
- Each step introduces a reviewable visible entity or entity group. This may be
  a catalog asset in `assets[]`, a procedural entry in `scene_objects[]`, or both
  when they form one topology unit.
- Each step describes: (a) which asset or scene_object is created, (b) spatial
  relation to already-placed entities, (c) orientation or position rule.
- Wrapper-managed objects (Viper, HMMWV, R2D2 URDF, ...) get their own
  implementation step that names the wrapper class explicitly, NOT a list of
  internal `.obj` files.
- NEVER group multiple distinct asset entries into a single step.

### Ordering: static scene first, moving bodies LAST

Within `implementation_steps` (and the corresponding order of `plan.assets[]`):

1. Place all **static** scene assets first, in dependency order — supports
   before supported (table before monitor, floor before chair).
2. Place **moving / actuated** bodies (rovers, vehicles, manipulators, any
   body driven by a motor or controller, any wrapper-managed robot) **LAST**,
   after the static scene is fully laid out.

Reason: the moving body needs the final scene geometry to choose a valid
spawn pose, and downstream code paths (camera follow, collision setup, control
loop) attach themselves to the moving body once the scene exists. Putting the
robot first forces re-derivation later and breaks `camera_target_body` look-up.

A `mbs_in_scene` plan that lists `curiosity_rover` / `viper` / `hmmwv` /
URDF robot as the FIRST implementation step (or first asset) is INVALID and
will be rejected.
