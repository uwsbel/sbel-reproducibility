---
name: plan_format
description: Output format specification for simulation plans — Markdown sections and current SimulationPlan schema conventions.
compatibility: pychrono >= 8.0
metadata:
  domain: planning
---
# Skill: Plan Format Specification

## Purpose

Defines the Markdown shape the PlanningAgent submits through `submit_plan`.
This document intentionally stays general: domain-specific construction
recipes belong in the domain skills and code-generation prompts.

## Required Markdown Sections

Plans use H2 headers exactly matching these section names:

- `## plan_type`
- `## simulation_parameters`
- `## objectives`
- `## implementation_steps`
- `## scene_objects`
- `## assets`
- `## topology`
- `## visualization`
- `## recording_mode`
- `## clarifications_needed`
- `## geometry_relations`

Structured sections use fenced `yaml` blocks. Flat string lists use dash
bullets. Empty sections may be omitted when they are truly empty, but keeping
the header is preferred for repairability.

## Plan Types

- `scene`: arranging 3D assets only; no mechanical bodies, joints, forces, or
  dynamics beyond static placement.
- `mbs`: pure multi-body mechanics with no external scene assets.
- `mbs_in_scene`: rigid-body dynamics, vehicles, robots, or mechanisms combined
  with procedural scene objects and/or external assets. No SPH fluid or FSI.
- `fsi_in_scene`: any plan combining SPH/FSI/CRM fluid or granular domains with
  multi-body dynamics or scene geometry.

## Implementation Steps

Pure `mbs` plans leave `implementation_steps: []`. Other plan types use a
3-5 item YAML list. Each item is a `SimulationStep`:

```yaml
- description: |
    <imperative build/review directive>
  assets: [<catalog asset names introduced this step>]
  scene_objects: [<procedural scene object names introduced this step>]
  cameras:
    - position: [<scene_edge_x>, <scene_edge_y>, <proper_height_z>]
      target: [<primary_action_or_visible_entities_center_x>, <..._y>, <..._z>]
      up: [<up_x>, <up_y>, <up_z>]
  constraints: []
  motion_expectations: []
```

Rules:

- The first non-MBS step must introduce at least one visible entity via
  `assets[]` or `scene_objects[]`; merge setup into that first visible step.
- Later steps may introduce no new visible entities when they animate, settle,
  or verify bodies already present.
- Every declared asset must be introduced by at least one step.
- `cameras` is always a list. The legacy singular `camera` shape is accepted by
  validators for backward compatibility but should not be emitted.
- Camera entries must use the format above; choose actual numeric coordinates
  from the planned scene geometry. Put the camera at the edge of the scene,
  slightly beyond the edge if needed, with enough height to frame the step.
  Aim the target at the center of the step's primary visible entities or action.
- Camera orientation is a geometry relation between camera and scene AABB
  (see `## geometry_relations` and `GEOMETRY_RELATION_RULES`). There is no
  silent FSI default — if the user did not specify side / chase / above,
  raise it as a structured clarification with options (side -Y, side +Y,
  perspective). When the relation is resolved, codegen reads the
  matching pattern in `geometry/scene_relations` to produce the camera pose.
- `motion_expectations` is required on every step. Use `[]` for static/setup
  steps. When non-empty, each name must refer to an asset with `is_dynamic: true`
  or a scene object with `fixed: false`.
- Treat `motion_expectations` as the CSV motion contract for the implementation
  step description. Before finalizing each step, identify every named dynamic
  body whose pose, velocity, orientation, or contact state is expected to change
  during THIS step, and list those body names. Use `[]` only when the step is
  strictly construction, placement, visualization, or setup with no intended body
  motion.

## Camera and Recording Mode

`recording_mode` selects the review video path:

- `vsg_only`: VSG is the sole renderer. Each step must have exactly one camera.
  Required for FSI, SPH, fluid, or particle-domain scenes that sensor cameras
  cannot render correctly.
- `sensor_cams`: `ChSensorManager` / `setup_preview_camera` renders one MP4 per
  camera. Each step must have 2-3 cameras from complementary viewing
  directions, not zoom variants of the same view.

`visualization.mode` should align with recording mode. FSI/SPH plans use
`mode: vsg` with `recording_mode: vsg_only`; non-FSI plans typically use
`vsg_with_sensor_camera` with `recording_mode: sensor_cams`.

## Assets and Scene Objects

Use `assets` only for external loadable catalog rows:

```yaml
- name: <catalog name>
  type: <mesh | urdf | vehicle_json | wrapper_vehicle | texture | heightmap>
  filename: <exact catalog path>   # file-backed rows
  factory: <exact catalog factory> # wrapper_vehicle rows
  count: 1
  fixed: true
  is_dynamic: false
  ideal_height: <float>            # meshes only, when scale matters
  description: <short description>
```

Use `scene_objects` for non-catalog procedural objects, generated boundaries,
terrain patches, solver domains, and fluid domains:

```yaml
- name: <object name>
  role: <short role>
  construction_source: procedural_primitive | generated_boundary | fluid_domain
  primitive: <box | sphere | cylinder | box_container | ...>
  domain_type: <sph_fluid_box | ...>
  size: [<x>, <y>, <z>]
  fixed: true
  dynamic: false
  description: <short description>
```

Every concrete object requested by the user must appear in either `assets[]` or
`scene_objects[]`. Do not drop supports, containers, platforms, walls, tanks,
ramps, terrain patches, bridges, fluid domains, or generated boundaries just
because they are procedural.

## Topology

For `scene`, `mbs_in_scene`, and `fsi_in_scene`, `topology.scene_predicates`
is the source of truth for placement. Every visible asset and scene object must
appear at least once as a predicate `subject`.

```yaml
gravity_axis: -z
working_plane: xy
orientation_convention: z_up_native
scene_size: null
reference_heights:
  - {name: ground_top, z: 0.0}
scene_predicates:
  - subject: <asset_or_scene_object_name>
    predicate: <predicate from planning/scene_coordinate_system>
    object: root
    params: {}
    position: {x: <float>, y: <float>, z: <float>}
    orientation: {deg_z: <float>}
body_positions: {}
joints: []
```

Use predicate names from `planning/scene_coordinate_system`. Pure `mbs` plans
may instead use `body_positions`, geometry, and `joints`.

## Geometry Relations

`geometry_relations` is a YAML list naming each spatial relation between a
pair of bodies (or between camera and scene AABB) that the codegen agent
must encode in coordinates. Each entry corresponds to a subsection in the
`geometry/scene_relations` skill, which the codegen agent reads at
generation time to produce the correct `SetPos` / camera pose.

```yaml
- relation_name: <pattern name from geometry/scene_relations skill,
                  e.g. platform_flush_wall_outer>
  body_a: <body name from assets[] or scene_objects[]>
  body_b: <body name, or "scene" / "camera" for non-body endpoints>
  parameters: {<pattern-specific kwargs, e.g. wall: -x>}
```

Rules:

- Every adjacent / attached / "on" / "next to" / "bridging" / "between" body
  pair the user describes must appear here.
- If the user did NOT specify the exact geometric relation (which face is
  flush, gap or no gap, which axis side for camera), leave
  `relation_name: TO_CLARIFY` and add a matching `clarifications_needed`
  entry per `GEOMETRY_RELATION_RULES`. Codegen will refuse to emit
  coordinates for `TO_CLARIFY` relations, forcing a clarify round.
- Once resolved, `relation_name` MUST exactly match a heading in
  `chrono_code/skills/geometry/scene_relations/SKILL.md`. Free-form names
  are not understood by codegen.

## Validation-Sensitive Rules

- `simulation_parameters` must include `time_step` and `simulation_duration`.
- Do not invent user-owned numeric values; add `clarifications_needed` when a
  required value is unspecified.
- `recording_mode` is either `vsg_only` or `sensor_cams`.
- `visualization.mode` is one of the supported renderer modes, normally `vsg`,
  `vsg_with_sensor_camera`, or `headless`.
- Numeric fields must be numeric literals, not prose.
- `orientation` uses `{deg_z: <float>}` for scene predicates.
- All asset names must come from the resolved catalog.
