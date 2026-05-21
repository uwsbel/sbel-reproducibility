---
name: scene
description: Entry point for static scene plans (plan_type=scene). Routes the agent to the correct scene, system, and camera skills and defines only high-level invariants.
compatibility: pychrono >= 8.0
metadata:
  domain: core
---

# Core Skill: Scene

Use this skill when `plan_type = "scene"`.

This core skill is a **router and constraint layer**, not a recipe catalog. It decides which child skills must be read and states scene-wide invariants that apply regardless of the specific asset family. Concrete asset-loading APIs, placement recipes, camera layouts, and scenario-specific code patterns belong to the owning child skills.

## Responsibility

This core skill covers only:

- static mesh scenes with no robot, vehicle, or actuated body
- which child skills must be read before writing code
- global invariants such as world convention, system ownership, and rendering separation
- high-level dispatch to the child skill that owns the asset pipeline

This core skill does **not** define:

- task-specific scene layouts
- named office / offroad / furniture recipes
- concrete utility signatures beyond short category reminders
- exact asset placement or camera placement formulas

## Required Skills — read these before writing code

| Skill | Why it is required |
|-------|--------------------|
| `mbs/system_create` | Choose and configure the physical system, gravity, solver |
| `vsg` | VSG visualization window setup, camera, grid, sky, render loop |
| `sens/camera` | Sensor manager, `setup_preview_camera`, recorder loop integration, camera output behavior |

## Choose the owning scene skill

After reading this core skill, read the child skill that actually owns the asset family:

- Read the owning `scene/*` skill for the asset family in the plan.
- All assets must be implemented through project utilities from `chrono_code.utils` / `chrono_code.utils.scene_assets`.
- The child skill is authoritative for **which** utility helper to call for a given asset family.

If a child skill owns a utility or asset pipeline, that child skill is authoritative for the concrete API and code pattern.

## Optional Skills — read if the plan requires them

| Skill | When to read |
|-------|--------------|
| `mbs/body_creation` | Manual body creation when not using a scene utility pipeline |
| `mbs/collision` | Contact material choice and collision shapes |
| `planning/scene_coordinate_system` | Scene predicate algebra and authored layout semantics |
| `planning/asset_discovery` | Discovering available assets and paths |

## Global Invariants

- Default world convention is **`Z-up`** unless a child skill explicitly owns a different convention.
- A static scene uses **one physical system** for all bodies, cameras, and visualization.
- `VSG` window viewpoint and sensor cameras are **separate concerns**:
  - `vis.AddCamera(...)` sets the interactive VSG window view
  - `setup_preview_camera(...)` creates actual sensor-camera output
- Do **not** call `chrono.SetChronoDataPath()` by default. Trust the default data path unless an owning child skill explicitly says otherwise.
- Choose one contact model and keep it consistent with the physical system and materials.
- `vis.AddCamera(...)` must be configured before `vis.Initialize()`.

## Routing Rules

- If the plan is a pure static scene with no driven body, stay in the `scene` family.
- Route to the owning `scene/*` child skill for the asset family in the plan.
- Do not invent a parallel asset-loading path in core; asset implementation must go through project utility functions owned by the child skill.
- If the request includes any robot, rover, vehicle, or actuated body, this is no longer a plain `scene` problem; route to `core/mbs_in_scene`.

## Minimal Category-Level Reminders

- All scene assets should be implemented through project utility functions, not ad-hoc inline loaders.
- The owning child skill decides which asset utility helper is valid for that asset family.
- Sensor recording requires `setup_preview_camera()` from `sens/camera`; `vis.AddCamera(...)` alone is not enough.

## What Core Must Not Contain

If you are writing or updating a core scene skill, do not place these here:

- concrete asset-family recipes
- room-size-specific camera formulas
- exact `AssetDescriptor(...)`, `create_asset_body(...)`, or `add_visual_assets(...)` recipes
- ground-construction recipes owned by a child scene skill

Those belong in the owning child skills.
