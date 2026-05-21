---
name: fsi_in_scene
description: >
  Entry point for FSI-coupled hybrid plans (plan_type=fsi_in_scene) — any
  scene combining SPH fluid (water tanks, dam-break, wave channels) with
  multi-body dynamics, optionally with a wheeled vehicle. Pick this over
  mbs_in_scene whenever the plan involves a fluid domain (scene_objects
  with domain_type starting "sph_") or an FSI body registration
  (scene_objects with fsi_registration set). Routes to fsi/sph (always),
  veh/wheeled_vehicle (when vehicle present), and enforces FSI-specific
  invariants distinct from generic mbs_in_scene rigid scenes.
compatibility: pychrono >= 8.0
metadata:
  domain: core
---

# Core Skill: FSI-in-Scene

Use this skill when `plan_type = "fsi_in_scene"`.

This core skill is a **classifier, router, and constraint layer** for
hybrid simulations that mix SPH fluid with multi-body dynamics. The
canonical reference is
[`demo/scene/tutorial_VEH_FSI_FloatingBlock.py`](../../../../demo/scene/tutorial_VEH_FSI_FloatingBlock.py)
— a Polaris driving across a floating bridge over a water tank. It does
not own concrete fluid recipes, vehicle construction, or visualization
details; those belong to the relevant `fsi/*`, `veh/*`, and `vsg`
child skills.

## Responsibility

This core skill covers only:

- deciding when a plan should be `fsi_in_scene` rather than `mbs_in_scene`
- which child skills must be read before writing code
- FSI-specific invariants distinct from rigid-only hybrid scenes
- dispatch rules between vehicle-FSI and vehicle-less-FSI variants

This core skill does **not** define:

- concrete SPH parameter tables (see `fsi/sph` Pattern A/B)
- per-wall container layouts (see `fsi/sph` Pattern C)
- floating-body BCE patterns (see `fsi/sph` Pattern D)
- wheel-spindle FSI registration (see `veh/wheeled_vehicle` "FSI Coupling")
- driver schedules (see `veh/driver` and `veh/wheeled_vehicle` FSI Coupling)

## When to Use This Plan Type

**Pick `fsi_in_scene`** whenever any of these hold:

- A `scene_object` has `domain_type` starting with `sph_` (e.g.
  `sph_fluid_box`, `sph_granular_*`)
- A `scene_object` has `fsi_registration` set (e.g.
  `CreatePointsBoxInterior` for floating bodies,
  `CreatePointsBoxContainer` for tank walls)
- The user prompt mentions: SPH, FSI, fluid, water tank, dam-break,
  wave, splash, buoyancy, floating bridge, swim, submerge

**Pick `mbs_in_scene`** instead when the plan is rigid-body-only —
vehicle on terrain, robot in office, manipulator with props, etc. — even
if the prompt sounds "scene-y". `mbs_in_scene` doesn't pre-inject
`fsi/sph`, so it's the wrong bucket the moment any fluid is involved.

`scene` and `mbs` plan_types remain unchanged: pure asset placement
(no MBS) and pure mechanics (no scene), respectively.

## Required Skills for FSI Scenes (vehicle present)

Read these before writing code:

| Skill | Why it is required |
|-------|--------------------|
| `fsi/sph` | SPH stack creation, particle seeding, container per-wall layout, floating-body BCE registration, init/step-loop ordering, all FSI hard rules |
| `veh/wheeled_vehicle` | Vehicle construction, powertrain, RIGID tire choice for FSI, **wheel-spindle FSI registration** (the bug-prone bit), 2-arg `vehicle.Synchronize` in FSI scenes, `ChWheeledVehicleVisualSystemVSG` with FSI plugin |
| `veh/driver` | Pre-programmed driver schedules — **NEVER `ChInteractiveDriver`** in headless FSI runs |
| `vsg` | Generic VSG window, camera, light, render-loop |
| `mbs/body_creation` | Box/cylinder/sphere primitives for static infrastructure and generated-boundary coordinate conventions (`pose.position.z = floor` for tanks/channels). **Includes the FULL-extents hard rule** that prevents the half-size-box bug |

## Required Skills for FSI Scenes (no vehicle)

Read these before writing code:

| Skill | Why it is required |
|-------|--------------------|
| `fsi/sph` | Same as above — owns the entire FSI stack |
| `vsg` | Generic VSG window + FSI plugin attachment |
| `mbs/body_creation` | Static infrastructure primitives, generated-boundary coordinate conventions, and FULL-extents rule |

## Optional Skills — read if needed

| Skill | When to read |
|-------|--------------|
| `mbs/collision` | Custom contact materials beyond the default `ChContactMaterialSMC` |
| `mbs/quaternions` | Rotation rules for non-axis-aligned floating bodies |
| `scene/custom_assets_scene_convex_decomp` | When the FSI scene includes external mesh assets (rare; most FSI demos use procedural primitives) |

## Step Shape Rules

`fsi_in_scene` mirrors `mbs_in_scene`'s step contract:

- `implementation_steps` MUST be non-empty
- The **first step** must introduce at least one visible entity
  (`assets[]` or `scene_objects[]` non-empty); the FSI tank, fluid
  domain, or floating plate qualifies as a visible entity
- Every `plan.assets[]` entry must be introduced by some step
- Each step has `cameras: [...]` with 2-3 distinct viewing angles
- Later steps MAY have empty `assets`/`scene_objects` for pure-physics
  phases (settle, drive across, etc.)

## Hard Rules — FSI-Specific Invariants

These rules are non-negotiable. All defer to a single canonical statement
in a child skill.

1. **`ChSystemSMC` required (not NSC).** FSI coupling needs the smooth
   contact model. See `fsi/sph` Pattern A.

2. **Match gravity on both `sysMBS` and `sysFSI`.** Mismatch breaks
   hydrostatic equilibrium and makes floating bodies oscillate. See
   `fsi/sph` HR-3.

3. **Single step loop: `sysFSI.DoStepDynamics(dT)` only.** Never call
   `sysMBS.DoStepDynamics()` or `vehicle.Advance(dT)` separately —
   `sysFSI.DoStepDynamics` already advances the MBS bodies, and
   double-stepping desyncs the FSI coupling. See `fsi/sph` HR-2.

4. **`vehicle.Synchronize(t, driver_inputs)` — 2 args, no terrain.**
   In FSI scenes there is no `ChTerrain`; platforms / supports / ramps
   are plain `ChBody` instances on `sysMBS`. Calling the 3-arg overload
   raises a null-reference error. See `veh/wheeled_vehicle` "Synchronize
   signature in FSI scenes".

5. **Build the vehicle visualizer with
   `chrono_code.utils.fsi_assets.build_fsi_vehicle_visualizer(...)`.**
   The helper packages the 8+ ordered calls that must all be wired
   correctly to render the chassis + wheels + SPH particles + BCE markers
   together — `SetXVisualizationType` × 4, marker enables × 4,
   `AttachVehicle`, `AttachSystem(sysMBS)`, `AttachPlugin(visFSI)`,
   window/camera config, `Initialize()` last. See `fsi/sph` Pattern F for
   the full call site (Pattern G covers the non-vehicle case).

   ```python
   visFSI = fsi.ChSphVisualizationVSG(sysFSI)
   vis = build_fsi_vehicle_visualizer(
       sysMBS, sysSPH, sysFSI,
       vehicle=polaris,
       sph_visualization=visFSI,
       window_title="...",
   )
   ```

   Hand-rolling the sequence is the direct cause of
   `session_20260428_164422`'s "wheels visible but no chassis" failure
   (12 iterations / 98 turns). The helper exists to make that class of
   bug structurally impossible. Symptom → root cause table for any
   regression that bypasses the helper:

   | Visible failure | Missing call inside the helper |
   |---|---|
   | "only vehicle wheels visible, no platforms, no water" | `AttachSystem(sysMBS)` |
   | "only floating wheel BCE markers, no chassis" | `SetChassisVisualizationType(MESH)` |
   | "only chassis visible, no wheels" | `SetWheelVisualizationType(MESH)` |
   | "no SPH particles, just rigid bodies" | `AttachPlugin(visFSI)` |
   | "tank empty even though execution succeeded" | `vis.Initialize()` ran before `sysFSI.Initialize()` — see `fsi/sph` "Anatomy of an FSI Script" |

   The non-vehicle path (pure sloshing tank with no wheeled vehicle) uses
   a hand-rolled `chronovsg.ChVisualSystemVSG()` — the canonical example
   is `fsi/sph` Pattern G, which spells out the
   `AttachSystem` + `AttachPlugin` + `Initialize` ordering.

6. **NO `ChInteractiveDriver` in headless / batch runs.** Its keyboard
   loop never fires in CI; the vehicle stays at zero throttle forever
   and looks identical to a missing-physics bug. Use a plain
   `veh.DriverInputs()` struct (state-driven phased throttle) or
   `veh.ChDataDriver` (open-loop schedule). See `veh/driver` HARD RULE.

7. **Tank container layout — no top wall, skip walls on every periodic
   axis.** A closed top traps escaping fluid as a numerical pressure
   spike; a wall on a `BC_*_PERIODIC` axis fights the wrap-around. See
   `fsi/sph` Pattern C.

8. **Box constructors take FULL extents — never `/2`.** The
   `ChVisualShapeBox` / `ChCollisionShapeBox` / `ChBodyEasyBox`
   constructors all expect the full size (W, D, H), not half-extents.
   Passing `W/2, D/2, H/2` makes the body half-size on every axis and
   produces "platforms floating far from the tank" symptoms. See
   `mbs/body_creation` "Hard Rule: ChVisualShapeBox /
   ChCollisionShapeBox take FULL extents — never half".

9. **Wheel spindles MUST be registered as FSI bodies (when vehicle
   present).** Without this the vehicle-FSI coupling produces no force
   exchange — wheels never push back against floating bodies, fluid
   doesn't ripple, vehicle stays parked. See `veh/wheeled_vehicle`
   "FSI Coupling — Wheel Spindle Registration".

## Routing Rules

- FSI + wheeled vehicle: route to `fsi/sph`, `veh/wheeled_vehicle`,
  `veh/driver`, `vsg`, `mbs/body_creation`. The pre-injection should
  preserve `fsi/sph` Pattern A/B/C/D/F + relevant HRs and
  `veh/wheeled_vehicle` "FSI Coupling" section at FULL fidelity.
- FSI without vehicle (sloshing tank, dam-break, free-surface demo):
  route to `fsi/sph`, `vsg`, `mbs/body_creation`. Drop the vehicle
  skills.
- Skip `veh/terrain` entirely. No FSI scene uses `RigidTerrain` or `SCMTerrain`
  — the vehicle interacts with the SPH fluid via spindle FSI registration,
  not via a terrain abstraction.

## Reference Implementation

The canonical end-to-end example is
[`demo/scene/tutorial_VEH_FSI_FloatingBlock.py`](../../../../demo/scene/tutorial_VEH_FSI_FloatingBlock.py).
Every code template in the child skills (Pattern A through Pattern G in
`fsi/sph`, FSI Coupling in `veh/wheeled_vehicle`) is line-by-line
faithful to this tutorial. When in doubt, treat the tutorial as the
ground truth and the skill text as commentary.
