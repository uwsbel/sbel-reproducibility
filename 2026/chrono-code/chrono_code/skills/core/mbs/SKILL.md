---
name: mbs
description: Entry point for pure multi-body simulation plans (plan_type=mbs). Routes the agent to the correct mechanics, system, and camera skills and defines only high-level invariants.
compatibility: pychrono >= 8.0
metadata:
  domain: core
---

# Core Skill: MBS

Use this skill when `plan_type = "mbs"`.

This core skill is a **router and constraint layer** for pure multi-body problems. It does not own concrete body recipes, topology recipes, or scenario-specific camera/body layouts. Those belong to the relevant `mbs/*` and `sens/*` child skills.

## Responsibility

This core skill covers only:

- pure mechanics with no external mesh-scene pipeline
- which child skills must be read before writing code
- global invariants around system choice, rendering, and camera separation
- high-level dispatch to body, topology, collision, loop, and camera skills

This core skill does **not** define:

- named scenarios
- detailed body or joint construction recipes
- concrete camera placement formulas tied to a specific mechanism
- utility-specific signatures beyond category-level reminders

## Required Skills — read these before writing code

| Skill | Why it is required |
|-------|--------------------|
| `mbs/system_create` | System creation, gravity, solver, collision-system setup |
| `mbs/body_creation` | Bodies and visual/collision body patterns |
| `mbs/simulation_loop` | Time-stepping structure, stepping order, logging loop patterns |
| `vsg` | VSG visualization window setup, camera, grid, sky, render loop |
| `sens/camera` | Sensor manager, `setup_preview_camera`, recorder loop integration |

## Optional Skills — read if the plan requires them

| Skill | When to read |
|-------|--------------|
| `mbs/topology` | Joints, links, motors, springs, and constraints |
| `mbs/collision` | NSC vs SMC, contact materials, collision shapes |
| `mbs/quaternions` | Quaternion construction and rotation rules |
| `mbs/fea` | Finite-element modeling |

## Global Invariants

- Default world convention is **`Z-up`** unless a child skill explicitly owns a different convention.
- Use **one** physical system and keep its contact model consistent with all materials and collision setup.
- `VSG` window viewpoint and sensor cameras are **separate concerns**:
  - `vis.AddCamera(...)` sets the interactive VSG window view
  - `setup_preview_camera(...)` creates actual sensor-camera output
- `vis.AddCamera(...)` must be configured before `vis.Initialize()`.
- Do **not** call `chrono.SetChronoDataPath()` by default.
- Choose one stepping path and keep it consistent with the owning mechanics abstraction. Do not mix competing stepping APIs unless a child skill explicitly requires it.

## Routing Rules

- Need bodies only: read `mbs/body_creation`
- Need joints / actuators / springs: read `mbs/topology`
- Need contacts or friction: read `mbs/collision`
- Need camera output: read `sens/camera`
- Need pure mechanics only: stay in `mbs`
- If the request introduces external mesh-scene assets, route to `core/scene` or `core/mbs_in_scene` depending on whether there is a moving body

## Minimal Category-Level Reminders

- Sensor recording requires `setup_preview_camera()`; `vis.AddCamera(...)` alone is not enough.
- Light direction for VSG is category-level VSG configuration; use the owning skill for concrete usage.
- Read the owning child skill for exact APIs and code templates.

## What Core Must Not Contain

If you are writing or updating a core MBS skill, do not place these here:

- concrete pendulum, slider-crank, rover, or vehicle recipes
- body-geometry-specific code templates
- exact camera-placement formulas for a named mechanism
- detailed collision recipes owned by `mbs/collision`
- detailed recorder recipes owned by `sens/camera`

Those belong in the owning child skills.
