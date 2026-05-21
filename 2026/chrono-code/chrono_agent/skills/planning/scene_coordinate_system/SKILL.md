---
name: scene_coordinate_system
description: Coordinate system, predicate algebra, and orientation rules for PyChrono scene plans. Lets the planner derive concrete (x, y, z, deg_z) by composing predicates instead of guessing.
compatibility: pychrono >= 8.0
metadata:
  domain: planning
---
# Skill: Scene Coordinate System & Predicate Algebra

## Purpose

Use this skill to derive concrete scene coordinates from symbolic placement
relations. `scene_predicates` is the source of truth: every row must bind the
symbolic relation (`predicate` + `params`) to the resolved numeric state
(`position` + `orientation.deg_z`). Do not type coordinates from intuition;
compose predicates, apply their algebra in order, then serialize the result.

## Coordinate Frame

- gravity_axis: `-z`; working_plane: `xy`; height axis: `z`.
- camera_up is anti-parallel to gravity:
  - `gravity_axis="-z"` -> `camera_up=[0,0,1]`
  - `gravity_axis="-y"` -> `camera_up=[0,1,0]`
- Cardinal names used by predicates:
  - `+X = Front`, `-X = Back`, `+Y = Left`, `-Y = Right`, `+Z = Up`
- Asset OBJ baseline: `z_up_native`, `+Z` up, `+X` facing.
- Default rotation `(deg_x=0, deg_z=0)` keeps the asset facing `+X`.
- Per-asset metadata lives in `data/scene/assets.json`.

## Scene Invariants

### `topology.reference_heights`

Declare shared z-layers once and reference them from height predicates:

```yaml
topology:
  reference_heights:
    - {name: ground_top,    z: 0.0}
    - {name: tank_rim,      z: 1.0}
    - {name: water_surface, z: 0.8}
    - {name: walkway_top,   z: 1.2}
```

`FREE-SURFACE-AT` may use either a literal float or a reference name.
Serialized `position.z` is always the resolved float, never the string. For a
fluid row using `FREE-SURFACE-AT`, it is the free-surface marker that codegen
uses to derive the sampler volume.

## Orientation

- `deg_x`: tilt around X, almost always `0` for `z_up_native`.
- `deg_z`: yaw around Z, measured counterclockwise from world `+X`.
- Runtime composition: `R = Rz(deg_z) * Rx(deg_x)`; rotations are not baked
  into OBJ meshes.

Emit `FACING-*` only when the prompt or image expresses orientation intent:
"faces", "points at", "looks at", "heading toward", "aimed at", "rotated to
face". Spatial nouns such as "behind", "next to", "on top of", and "in front
of" are placement, not facing; keep the `+X` baseline unless orientation is
explicit.

If `FACING-TO` is emitted, compute `deg_z` from its formula. Pairing
`FACING-TO` with an uncomputed default `deg_z=0` is invalid unless the formula
really resolves to 0.

| Facing | `deg_z` |
|---|---:|
| `+X` / Front | 0 |
| `+Y` / Left | 90 |
| `-X` / Back | 180 |
| `-Y` / Right | -90 or 270 |

## Bounding-Box Notation

Each asset `A` has a fixed-size axis-aligned bbox derived from target height
and mesh aspect ratio unless a procedural predicate declares otherwise:

```text
A.min_x  A.max_x  A.center_x
A.min_y  A.max_y  A.center_y
A.min_z  A.max_z  A.center_z
A.size_x = A.max_x - A.min_x
A.size_y = A.max_y - A.min_y
A.size_z = A.max_z - A.min_z
A.bottom_z = A.min_z
A.top_z    = A.max_z
A.height   = A.size_z
A.yaw      = deg_z
```

When assigning one side or center, recompute the matching bbox values from the
fixed size. Apply predicates left-to-right; later predicates see the updated
state from earlier predicates.

## Predicate-to-Position Contract

The planner must derive coordinates by updating bbox anchors, not by inventing
literal centers. Unless a predicate explicitly says otherwise, serialized
`position` is the subject body's world-frame center.

General derivation loop:

1. Resolve every object's full extents first.
   - Procedural `size: [sx, sy, sz]` is full width/depth/height, never
     half-extents.
   - For generated boxes, tanks, platforms, floating plates, and ramps, bbox
     values are derived directly from `size`.
   - If a size is not specified and cannot be derived from an enclosing object
     or vehicle wheelbase, add `clarifications_needed`; do not guess a small
     placeholder.
2. Pick one coordinate convention for the enclosing scene object and keep it.
   - Regular box containers use the normal center convention.
   - `generated_boundary` containers/tanks/channels are special: center in
     `x/y`, floor at `position.z`, rim at `position.z + size.z`. A 4 x 2 x
     1 m generated-boundary tank with floor at z=0 therefore has
     `position=(0, 0, 0)`, rim z=1, x-range `[-2, +2]`, y-range `[-1, +1]`.
   - Do not interpret `position: {x: 0, y: 0, z: 0}` as "tank starts at the
     origin" unless the user explicitly asked for a corner-origin frame. In
     this schema, `position` is normally a center.
3. Apply predicates to anchors (`min_x`, `max_x`, `center_x`, `bottom_z`,
   `top_z`) and recompute `position` from the final bbox.
4. Serialize the final numeric center in `position` for rigid/procedural
   bodies. For SPH fluid rows with `FREE-SURFACE-AT`, the row's `position.z`
   may denote the free-surface marker; codegen derives the sampler center
   separately from the free-surface height.

### Common coordinate derivations

Container centered at the scene origin:

```text
B.center_x = 0
B.center_y = 0
B.bottom_z = 0
B.center_z = B.size_z / 2
B.min_x = -B.size_x / 2
B.max_x = +B.size_x / 2
B.min_y = -B.size_y / 2
B.max_y = +B.size_y / 2
```

Platform/support outside a container face:

```text
side = "-x": A.max_x = B.min_x - distance
side = "+x": A.min_x = B.max_x + distance
side = "-y": A.max_y = B.min_y - distance
side = "+y": A.min_y = B.max_y + distance
A.center_transverse = B.center_transverse
A.top_z or A.bottom_z comes from a height predicate
```

This is expressed with the existing predicates:

```text
side "-x" -> BACK-OF B {"distance": d}  + ALIGN-CENTER-LR B + resolved z
side "+x" -> FRONT-OF B {"distance": d} + ALIGN-CENTER-LR B + resolved z
side "-y" -> RIGHT-OF B {"distance": d} + ALIGN-CENTER-FB B + resolved z
side "+y" -> LEFT-OF B {"distance": d}  + ALIGN-CENTER-FB B + resolved z
```

Bridge/plate/beam spanning two flanks:

```text
axis = "x":
  inner_left  = left_support.max_x
  inner_right = right_support.min_x
  A.center_x  = (inner_left + inner_right) / 2
  A.center_y  = average(left_support.center_y, right_support.center_y)
  if A.size_x is unspecified:
      A.size_x = inner_right - inner_left + 2 * overlap
axis = "y": analogous with max_y/min_y and center_x as transverse center
```

Use this only when both flanks are already placed. If the user says a body
"bridges" two supports but does not specify whether it should float at the
water surface, sit flush with the platforms, overlap the supports, or leave a
gap, add a clarification or emit a `geometry_relations` entry with
`relation_name: TO_CLARIFY`; do not invent `bridges_between` and a center.

Floating body at a still-water surface:

```text
A.bottom_z = fluid.free_surface_z
A.center_z = fluid.free_surface_z + A.size_z / 2
```

If the user also says "flush with platform top", that is an additional
constraint. It is only satisfiable when:

```text
platform.top_z == fluid.free_surface_z + A.size_z
```

If this equality is not true, choose the user-explicit constraint if one was
clearly prioritized; otherwise ask for clarification. Do not silently set
`A.center_z = platform.top_z` or `A.center_z = fluid.free_surface_z`.

### Unknown or free-form predicates are invalid

Every emitted predicate must either be listed below or be decomposed into the
listed predicates. Free-form words may be placed in `description`, not in
`predicate`.

| Don't emit | Reason | Use instead |
|---|---|---|
| `at_origin` | Ambiguous: corner at origin vs center at origin vs bottom at origin. | `PLACE-ON-BASE root {"x": 0, "y": 0}` for bottom-on-ground objects, or resolved x/y/z coordinates. |
| `flush_against_wall` | Does not specify inner/outer face, axis, or distance. | `BACK-OF` / `FRONT-OF` / `LEFT-OF` / `RIGHT-OF` + alignment + resolved z. |
| `bridges_between` | Does not define axis, overlap, z support, or whether size is derived. | Bridge derivation above + `FLOATS-AT-SURFACE` or resolved z. |
| `on_top` | Ambiguous offsets and support face. | `PLACE-ON` with explicit x/y offsets. |
| `ON-TOP-OF` with z offset | Mixes support contact with arbitrary height offset. | `PLACE-ON` when contact is intended, otherwise resolve the final pose directly. |
| `BEHIND-AND-ON` | Combines two independent axes into one undefined operation. | `PLACE-ON` with explicit x/y offsets relative to the support. |
| `CENTERED-ON` | Ambiguous whether z support is included. | `ALIGN-CENTER-LR` + `ALIGN-CENTER-FB` + `PLACE-ON` when support contact is intended. |
| `INSIDE` | Ambiguous for container vs fluid. | `PLACE-IN` for containers or `SUBMERGED` for fluids. |

## Predicate Algebra

Each predicate updates the subject `A` using an object `B` and optional
`params`. A body is fully placed only when x, y, z, and yaw are determined.

### 1. 2-D Spatial

`distance` is meters and may be `0` for touching.

| Predicate | Effect on `A` |
|---|---|
| `LEFT-OF` | `A.min_y = B.max_y + params["distance"]` |
| `RIGHT-OF` | `A.max_y = B.min_y - params["distance"]` |
| `FRONT-OF` | `A.min_x = B.max_x + params["distance"]` |
| `BACK-OF` | `A.max_x = B.min_x - params["distance"]` |

`FRONT-OF` and `BACK-OF` are pure world-`X` bbox algebra. They do not encode
furniture semantics, camera-facing semantics, or "where a user usually sits".
When placement depends on a subject's actual facing direction, resolve yaw
with `FACING-TO` first, then choose the spatial predicate that realizes that
world-frame direction.

### 2. 2-D Alignment

| Predicate | Effect on `A` |
|---|---|
| `ALIGN-CENTER-LR` | `A.center_y = B.center_y` |
| `ALIGN-CENTER-FB` | `A.center_x = B.center_x` |
| `ALIGN-LEFT` | `A.max_y = B.max_y` |
| `ALIGN-RIGHT` | `A.min_y = B.min_y` |
| `ALIGN-FRONT` | `A.max_x = B.max_x` |
| `ALIGN-BACK` | `A.min_x = B.min_x` |

### 3. 2-D Symmetry

| Predicate | Effect on `A` |
|---|---|
| `SYMMETRY-ALONG` | Mirror A's center across object `C`: `A.center_x = 2*C.center_x - B.center_x`, `A.center_y = 2*C.center_y - B.center_y`; `C` is `params["C"]`. |

### 4. Rotation-Only

| Predicate | Effect on `A.yaw` |
|---|---|
| `FACING-TO` | `degrees(atan2(B.center_y - A.center_y, B.center_x - A.center_x))` |
| `FACING-SAME-AS` | `B.yaw` |
| `FACING-OPPOSITE-TO` | `(B.yaw + 180) mod 360` |
| `FACING-FRONT` | `0` |
| `FACING-BACK` | `180` |
| `FACING-LEFT` | `90` |
| `FACING-RIGHT` | `-90` |
| `RANDOM-ROT` | Pick one concrete `uniform(0, 360)` value at plan time. |
| `ORIENT-BY-RELATIVE-SIDE` | Choose `A.yaw` in `{default_yaw, default_yaw + 90}` to maximize bbox-overlap area with `B`. |

### 5. Height / Face-Level

Support predicates:

| Predicate | Effect on `A` |
|---|---|
| `PLACE-ON-BASE` | `A.center_x = params["x"]`, `A.center_y = params["y"]`, `A.bottom_z = root.top_z`; params may be empty if x/y are set by other predicates. |
| `PLACE-ON` | `A.center_x = B.center_x + params["x_offset"]`, `A.center_y = B.center_y + params["y_offset"]`, `A.bottom_z = B.top_z`; `B` must be a real flat support, not `root`. |

Use `PLACE-ON` when the support and offsets fully determine x/y.

### Vehicle exception (HARD)

The above support formulas (`PLACE-ON-BASE`, `PLACE-ON`) treat `A` as an
axis-aligned rigid box where `A.bottom_z = B.top_z` gives the rest position.
**This formula does NOT apply when `A` is a wheeled vehicle**
(`construction.kind=asset`,
`asset_type ∈ {wrapper_vehicle, vehicle_json}`). Reasons:

- `WheeledVehicle.Initialize(coordsys)` takes the **chassis frame
  origin** (which is offset from chassis bbox center by some
  vehicle-specific amount), not the bbox bottom or center.
- The vehicle's resting `init_z` depends on `tire_radius` and
  `front_spindle_z`, not on the chassis bbox z-extent.
- Using `B.top_z + chassis.size.z/2` puts the chassis somewhere near
  the suspension's nominal mid-stroke; the wheels then either dangle
  or penetrate the support.

**Plan-level rule for vehicles on supports:**

- Planner resolves `position.x`, `position.y`, and `orientation.deg_z`
  from the support's geometry (centered on the support, offset to leave
  clearance, etc.).
- Planner does **NOT** emit `position.z` — leave it `None`. The vehicle
  is a child object whose pose.position is derived later.
- Codegen at code-generation time computes `init_z` via
  `chrono_agent.utils.vehicle_geometry.chassis_init_z(vehicle_json,
  support_top_z=B.top_z, tire_json=<tire-you-load>)`. The planner only
  needs to identify which body is the support via `PLACE-ON` so codegen can
  resolve `B.top_z`.

**Vehicle footprint check** (still planner's job): when `A` is a wheeled
vehicle and `B` is a platform / ramp, size `B` so its x-extent is at
least `A.wheelbase + 0.4 m` (≈0.2 m clearance each side). The catalog
row for each wheeled vehicle exposes its `wheelbase` so the planner can
pick a feasible platform size up front rather than producing a plan
that geometrically cannot rest the vehicle (e.g. a 2.7 m-wheelbase
Polaris on a 1 m platform — codegen will then call `rebut_review` and
the step will loop).

Height declaration:

| Predicate | Params | Effect on `A` |
|---|---|---|
| `HEIGHT` | `{"value": <float>}` | Declares `A.size_z = value` for procedural bodies whose height is not known from mesh metadata. |

### 6. Grouping

| Predicate | Purpose |
|---|---|
| `GROUP` | Define a virtual object aggregating multiple assets. `params["anchor"]` is one member; group facing follows the anchor. |
| `COPY-GROUP` | Instantiate a new set of assets in the same relative configuration as an existing group, then apply spatial/rotation predicates to the new group. |

### 7. Special

| Predicate | Purpose |
|---|---|
| `PLACE-IN` | Place `A` inside container `B`; `A` may be a list like `[[name, count], ...]` for bulk insertion. |
| `PLACE-ANYWHERE` | Place `A` anywhere on root with no relational constraints. Emit after all other predicates for that asset. |

### 7a. Vocabulary discipline

Use only predicate names listed in the tables above. Names outside this
vocabulary have no solver and the resolved coordinate falls back to a naive
`B.position + params.get("offset", [0,0,0])` — silently wrong. If no single
predicate fits, compose predicates using the substitutions in
**Unknown or free-form predicates are invalid** above.

### 8. Fluid / Floating

| Predicate | Params | Effect on `A` |
|---|---|---|
| `FREE-SURFACE-AT` | `{"z": <float | ref_name>}` | Declares the fluid free-surface z. Must precede any `FLOATS-AT-SURFACE` or `SUBMERGED` reference to that fluid. |
| `FLOATS-AT-SURFACE` | `{"fluid": "<fluid_body>"}` | `A.bottom_z = fluid.free_surface` at t=0. |
| `SUBMERGED` | `{"fluid": "<F>", "depth": <float>, "anchor": "top" | "center"}` | If `anchor="top"`: `A.top_z = fluid.free_surface - depth`; if `center`: `A.center_z = fluid.free_surface - depth`. |
| `CONTAINS-FLUID` | `{"fluid": "<F>"}` | Marks `A` as a fluid container. Codegen's default `container_visual` becomes `none` so BCE/particles stay visible. Use on SPH / particle tanks, cups, pools, and channels. |

## Reasoning Workflow

Follow this order for every scene plan.

1. Set scene invariants.
   - Declare `topology.reference_heights` for shared z-layers.
2. Pick predicate families per body.
   - Fluid domain: `FREE-SURFACE-AT`.
   - Fluid container: `CONTAINS-FLUID` + resolved z + optional in-plane predicates.
   - Buoyant body: `FLOATS-AT-SURFACE` + height + in-plane predicates.
   - Flank along an axis: cardinal `FRONT-OF` / `BACK-OF` / `LEFT-OF` /
     `RIGHT-OF` + height + transverse alignment.
   - Bridge/beam spanning flanks: explicit `HEIGHT` + transverse
     alignment + height or fluid support; size the long dimension to span
     the gap between the two flanks.
   - Object on support: `PLACE-ON`.
   - Ground/root object: `PLACE-ON-BASE` or resolved z + in-plane predicates.
3. Emit predicates in dependency order.
   - Size declarations first (`HEIGHT` and known `size`), then z/support
     anchors, then in-plane placement, then orientation.
   - Keep all predicates for the same subject contiguous.
   - A referenced object must already be fully placed, except `"root"`.
4. Self-check the resolved state.
   - Every asset and scene object appears in `scene_predicates`.
   - Every row has concrete numeric `position.x/y/z` and `orientation.deg_z`.
   - Spanning bodies lie between their flanks (check via `position.x` /
     `position.y` matching the midpoint of the flank centres).
   - Flank pairs sit on opposite sides of their shared neighbour
     (e.g. one with `BACK-OF`, the other with `FRONT-OF`).
   - `FLOATS-AT-SURFACE` satisfies `position.z = fluid.free_surface + A.size_z/2`.
   - If a check fails, fix the predicate chain and recompute coordinates.

## Serializing `scene_predicates`

Each entry has this shape:

```jsonc
{
  "subject": "<A>",
  "predicate": "PLACE-ON",
  "object": "<B>",
  "params": {"x_offset": 0.0, "y_offset": 0.05},
  "position": {"x": 0.0, "y": 0.05, "z": 0.75},
  "orientation": {"deg_z": 0}
}
```

Rules:

- `subject`, `predicate`, `object`, `params`, `position`, and `orientation`
  must describe the same reasoning step.
- `position` is concrete meters. Resolve reference-height strings before
  serializing.
- `orientation` is a JSON object with numeric `deg_z`; free-form prose is
  invalid.
- `physical_predicates` (`supports`, `rests_on`, `contains`,
  `leans_against`, `attached_to`) is separate and declarative; it does not
  update bbox/yaw algebra.
- Predicate names are uppercase with hyphens.

## Worked Example: Vehicle + Bridge + Fluid

The numbers below are illustrative. Recompute every coordinate from the real
assets in the current scene.

Scene: Polaris drives `+x` across a 4 x 2 x 1 m water tank on a floating plate
bridging two walkway-level platforms.

```yaml
topology:
  reference_heights:
    - {name: tank_floor,    z: 0.0}
    - {name: tank_rim,      z: 1.0}
    - {name: water_surface, z: 0.8}
    - {name: walkway_top,   z: 1.2}
```

Predicate chain and resolved state:

```text
water_sph FREE-SURFACE-AT root {"z": "water_surface"}
  -> free_surface=0.8, position=(0,0,0.8), deg_z=0

water_tank HEIGHT root {"value": 1.0}
water_tank PLACE-ON-BASE root {"x": 0.0, "y": 0.0}
water_tank CONTAINS-FLUID root {"fluid": "water_sph"}
  -> floor z=0.0, rim z=1.0, free surface z=0.8, container_visual=none

left_platform BACK-OF water_tank {"distance": 0.05}
left_platform HEIGHT root {"value": 0.2}
left_platform ALIGN-CENTER-LR water_tank {}
  -> x center -4.05 for a 4 m platform, z=0.9, deg_z=0

right_platform FRONT-OF water_tank {"distance": 0.05}
right_platform HEIGHT root {"value": 0.2}
right_platform ALIGN-CENTER-LR water_tank {}
  -> position=(+4.05,0,0.9), deg_z=0

floating_plate ALIGN-CENTER-LR water_tank {}
floating_plate ALIGN-CENTER-FB water_tank {}
floating_plate FLOATS-AT-SURFACE root {"fluid":"water_sph"}
floating_plate HEIGHT root {"value": 0.4}
  -> long axis sized to span left_platform.max_x..right_platform.min_x,
     position=(0,0,1.0), deg_z=0

polaris PLACE-ON left_platform {"x_offset": 0.275, "y_offset": 0.0}
polaris FACING-TO floating_plate {}
  -> planner resolves: position.x=-3.775, position.y=0, deg_z=0
     position.z is NOT computed here — see "Vehicle exception" below.
     codegen will derive init_z via chassis_init_z(vehicle_json,
     support_top_z=left_platform.top_z, tire_json=...) at code-gen time.
```

Strict vehicle-on-platform rule:

- For any catalog vehicle on a finite support, quote the chassis bbox before
  placing it and prove the footprint fits the support.
- Both chassis ends must be inside the support footprint with at least 0.2 m
  rear clearance. The bridge-side end may overhang by at most 0.1 m only when
  the next action drives onto a bridge plate crossing that edge.
- Codegen must use the resolved `position` from `scene_predicates` for vehicle
  initialization. Do not copy tutorial constants such as `-bxDim/2 -
  bxDim*CH_1_3`; those caused the iter_009 backward-fall failure when the rear
  axle spawned off the platform.

Self-checks for this pattern:

- Floating plate lies between the platform centres
  (`(left_platform.center_x + right_platform.center_x) / 2`).
- Platforms sit on opposite `+x` / `-x` sides of the tank.
- Plate bottom equals the fluid free surface.
- Walkway tops are coplanar at `water_surface + platform_height`.
- Tank has `CONTAINS-FLUID`, so the fluid remains visible.

## Style Rules

- Use predicates to explain coordinates. A position without its predicate is a
  loose literal; a predicate without its resolved position is unfinished work.
- Do not emit symbolic placeholders in `position` or `orientation`.
- Use `FACING-TO` only for real orientation intent; otherwise default to
  `deg_z=0`.
- If an image is attached, geometric side choices must match the image, not
  domain convention.

---

## Relation Patterns (codegen formula reference)

Every entry in `objects[]` with `topology.role: child` carries a `relation`
naming one pattern from this table. The plan does NOT carry resolved
numeric `pose.position` for children — codegen reads this table and
emits Python expressions in `simulation.py` that compute each `obj`'s
position from its `ref`'s pose+size and `obj`'s own size.

### Conventions

Two endpoints in every pattern:

- `ref` — the **reference object**: the entry named in `topology.ref`.
  Its pose comes from upstream (either `topology.role: base` with absolute
  `pose.position`, or another resolved child).
- `obj` — the **object being placed**: the current entry. Its size is in
  `obj.construction.size`; its pose is what we are computing.

Frame:

- `gravity_axis = -z`. "Up" / "top" mean larger z.
- All bodies are axis-aligned with `size = {x, y, z}` extents centered on
  `pose.position` (DEFAULT convention; see "Anchor exception" below for
  `water_tank`). So:
  ```
  ref.top_z   = ref.pose.position.z + ref.size.z / 2
  ref.bot_z   = ref.pose.position.z - ref.size.z / 2
  ref.right_x = ref.pose.position.x + ref.size.x / 2     # +X face
  ref.left_x  = ref.pose.position.x - ref.size.x / 2     # -X face
  (similarly for Y, and for obj)
  ```
- `obj.pose.rotation_deg` defaults to `{x:0, y:0, z:0}` unless the pattern
  says otherwise.

### Anchor exception — `water_tank` (`primitive: generated_boundary`)

The water tank is the **only** primitive whose `pose.position.z` is NOT
the geometric center. Instead:

- `tank.pose.position.z` = the BOTTOM (floor) z-coordinate
- `tank.size.z` = full interior container height (from floor to rim)
- Rim / wall top z = `tank.z + tank.size.z` (full size, NOT `/ 2`)
- Default water surface z = `tank.z + tank.size.z - 0.2` (0.2 m below rim)
- Floor z = `tank.z`
- X and Y still use the center convention (`tank.x ± tank.size.x/2` etc.)

When applying any formula below where ref is a water_tank, **substitute
the Z parts** as follows:

| formula writes | for ref = water_tank, use |
|---|---|
| `ref.z + ref.size.z / 2` | `ref.z + ref.size.z`             (rim / wall top) |
| `ref.z - ref.size.z / 2` | `ref.z`                          (floor) |
| `ref.x ± ref.size.x / 2` | unchanged                        (XY centered) |
| `ref.y ± ref.size.y / 2` | unchanged                        (XY centered) |

The generated-boundary body convention lives in
[`mbs/body_creation`](../../mbs/body_creation/SKILL.md). SPH-specific
BCE, sampler, and free-surface rules live in
[`fsi/sph`](../../fsi/sph/SKILL.md) Pattern C.

Codegen output style: each formula below becomes one named Python constant
in the generated `simulation.py` (see `code_plan.md` §3.2 worked example).

### Relation Kinds — how the ref↔obj relationship dictates the formula

Before picking a specific pattern, identify which **kind** of geometric
relationship obj has with ref. The kind determines which `ref.size` /
`obj.size` axes the formula consumes and which axes inherit from ref.

| Kind | What it expresses | Axes that depend on ref+obj sizes | Axes that mirror ref | Patterns |
|---|---|---|---|---|
| **On-top** | obj rests on ref's top face. ref supports obj. | Z uses `ref.size.z + obj.size.z` (touching faces) | X, Y centered on ref | `spawned_on_top`, `placed_on_top`, `centered_on_ref` |
| **Adjacent-outside** | obj sits flush against one of ref's side faces; the two are siblings, not stacked. | The face-perpendicular axis (one of X/Y) uses `ref.size + obj.size`; Z uses `ref.size.z` and `obj.size.z` per the chosen Z-suffix | the face-parallel axis (the other of X/Y) shares ref's center | `adjacent_*` family with `_top_flush` / `_bottom_flush` / `_centers` Z-suffix |
| **At-water-surface** | obj's relationship is to a water surface derived from ref (ref is a fluid_domain or a water-filled tank's generated_boundary). The Z constraint says how submerged obj is. | Z uses `obj.size.z` plus the derived `water_surface_z`; for water tanks this is `ref.z + ref.size.z - 0.2` by default | X, Y centered on ref | `bottom_flush_water_surface`, `center_at_water_surface`, `top_flush_water_surface`, `floats_at_surface` |
| **Filling** | obj is the volume inside ref (e.g. SPH water inside a tank). obj's size is taken from ref's interior; obj's pose is centered or offset within ref. | obj.size is **derived from ref.size**; pose uses ref.size for offset within | X, Y, and Z all anchored on ref center / interior | `fills_container_to_top`, `fills_container_lower_half` |
| **Bridging** | obj spans between ref and a second body `ref_b` named in `params`. obj is centered between the two refs. | XY uses `(ref.x + ref_b.x)/2`, `(ref.y + ref_b.y)/2` — both refs' positions, no sizes; obj.size is consumed only by the separate Z height pattern that goes with the bridge. | none | `bridge_between_a_and_b`, `flush_with_platform_top` |
| **Camera framing** | obj is a camera/viewpoint, ref is the scene bounding box (or a target body). Position is on ref's edge looking inward. | obj's "size" is irrelevant; formula uses `ref.size` to pick the edge offset distance. | the look-at target = ref center | `side_minus_x` / `side_plus_x` / `side_minus_y` / `side_plus_y` / `top_down` / `perspective` |
| **Base (no ref)** | obj has no ref; user gives absolute coordinates. | n/a — pose is read from `obj.pose.position` directly | n/a | (no relation; `topology.role: base`) |

How to read the table:

- "Axes that depend on ref+obj sizes" — the formula MUST pull `size` from
  both endpoints on these axes. If your output expression for that axis
  doesn't include both, the formula is wrong.
- "Axes that mirror ref" — the formula reads ref's coordinate verbatim
  (no size math).
- The pattern's name tells you the kind by prefix (`spawned_on_top` →
  on-top; `adjacent_*` → adjacent-outside; `*_water_surface` → at-water-
  surface; `fills_*` → filling; `bridge_*` → bridging; `side_*` /
  `top_down` / `perspective` → camera framing).

Picking a pattern:

1. Read the user's intent ("vehicle starts on platform" / "platforms beside
   the tank with tops level" / "plate floats on water").
2. Map intent → kind from the table above.
3. Within that kind, pick the named variant that matches the geometric
   detail (which face? which Z alignment? which submersion depth?).
4. The formula for the chosen variant is in the corresponding subsection
   below.

### Stacking / on-top patterns

| `relation` | Formula |
|---|---|
| `spawned_on_top` | obj sits centered on top of ref. `obj.x = ref.x`, `obj.y = ref.y`, `obj.z = ref.z + ref.size.z/2 + obj.size.z/2` |
| `placed_on_top` | alias of `spawned_on_top` |
| `centered_on_ref` | obj center = ref center (XY only). `obj.x = ref.x`, `obj.y = ref.y`. z untouched. |

### Adjacency patterns

Each adjacency pattern places obj **outside** ref on one of ref's faces.
The XY offset always uses BOTH `ref.size` AND `obj.size` so the two bodies
just touch without overlap. The Z suffix names which alignment to use.

XY offset (shared by every variant):
```
+X face:  obj.x = ref.x + ref.size.x/2 + obj.size.x/2,  obj.y = ref.y
-X face:  obj.x = ref.x - ref.size.x/2 - obj.size.x/2,  obj.y = ref.y
+Y face:  obj.y = ref.y + ref.size.y/2 + obj.size.y/2,  obj.x = ref.x
-Y face:  obj.y = ref.y - ref.size.y/2 - obj.size.y/2,  obj.x = ref.x
```

Z component (pick one suffix):
```
_top_flush:    obj.z = ref.z + ref.size.z/2 - obj.size.z/2     (tops aligned)
_bottom_flush: obj.z = ref.z - ref.size.z/2 + obj.size.z/2     (bottoms on floor)
_centers:      obj.z = ref.z                                   (centers aligned)
```

| `relation` | When to pick |
|---|---|
| `adjacent_plus_x_top_flush` | obj outside ref's +x face, obj top = ref top. |
| `adjacent_minus_x_top_flush` | mirror on -x. |
| `adjacent_plus_y_top_flush` / `adjacent_minus_y_top_flush` | mirrors on Y. |
| `adjacent_plus_x_bottom_flush` | obj outside ref's +x face, obj and ref share a bottom plane. |
| `adjacent_minus_x_bottom_flush` / `adjacent_plus_y_bottom_flush` / `adjacent_minus_y_bottom_flush` | mirrors. |
| `adjacent_plus_x_centers` / `adjacent_minus_x_centers` / `adjacent_plus_y_centers` / `adjacent_minus_y_centers` | obj centered at the same Z as ref. |

### Water-surface / floating patterns

These assume `ref` is either a fluid_domain (use `ref.size.z` as fluid depth)
or a generated_boundary tank. Codegen inspects ref's primitive to pick the
right interpretation:

- For a regular centered `fluid_domain`, `water_surface_z = ref.z + ref.size.z / 2`.
- For a `water_tank` / generated_boundary tank, `water_surface_z = ref.z + ref.size.z - 0.2` by default.
- Use `0.0` clearance only when the user explicitly asks for fluid flush with the rim.

| `relation` | Formula |
|---|---|
| `bottom_flush_water_surface` | obj's bottom face = water surface. `obj.z = water_surface_z + obj.size.z/2`, `obj.x = ref.x`, `obj.y = ref.y`. |
| `center_at_water_surface` | obj center = water surface (half-submerged). `obj.z = water_surface_z`, `obj.x = ref.x`, `obj.y = ref.y`. |
| `top_flush_water_surface` | obj's top face = water surface (fully submerged). `obj.z = water_surface_z - obj.size.z/2`. |
| `floats_at_surface` | alias of `bottom_flush_water_surface` for buoyant bodies. |

### Container / fluid-fill patterns

| `relation` | Formula |
|---|---|
| `fills_container_to_top` | fluid_domain inside container ref; fluid top sits 0.2 m below the container top by default. For centered box containers: `obj.size.z = ref.size.z - 0.2`, `obj.z = ref.z - 0.1`; for water_tank generated boundaries, use `water_surface_z = ref.z + ref.size.z - 0.2`. |
| `fills_container_lower_half` | fluid fills bottom half of ref. `obj.size.z = ref.size.z/2`, `obj.z = ref.z - ref.size.z/4`. |

### Bridge / span patterns

| `relation` | Formula |
|---|---|
| `bridge_between_a_and_b` | requires `params: {ref_b: <name>}` naming a second ref. obj centered between ref and ref_b. `obj.x = (ref.x + ref_b.x)/2`, `obj.y = (ref.y + ref_b.y)/2`. `obj.z` comes from a separate height pattern (typically `bottom_flush_water_surface`). |
| `flush_with_platform_top` | obj's top face = ref's top face. `obj.z = ref.z + ref.size.z/2 - obj.size.z/2`. |

### Camera framing patterns

Used when `obj` is a camera and `ref` is the scene bounding box (or the
target body). Codegen treats `ref.size` as the scene extents.

For walled / enclosed scenes (any step whose `objects[]` includes
`room_wall_*` or other wall bodies forming a closed perimeter), use the
`inside_*_wall` variants below — the `side_*` patterns place the camera
*outside* the bbox, which would put it past an opaque wall and produce
a blank frame. For outdoor / open scenes the `side_*` patterns remain
correct.

| `relation` | Formula |
|---|---|
| `side_minus_y` | camera at -Y edge of scene looking +Y. `obj.position = [ref.x, ref.y - ref.size.y/2 - margin, ref.z + height_offset]`, `target = ref center`, `up = [0, 0, 1]`. |
| `side_plus_y` | mirror on +Y. |
| `side_minus_x` | at -X edge looking +X. |
| `side_plus_x` | mirror on +X. |
| `top_down` | above scene looking down. `obj.position = [ref.x, ref.y, ref.z + max(ref.size)*1.5]`, `target = ref center`, `up = [1, 0, 0]`. **Disallowed when the scene has a ceiling body — remove the ceiling first or pick a different framing.** |
| `perspective` | corner view. `obj.position = [ref.x + ref.size.x/2, ref.y - ref.size.y/2, ref.z + ref.size.z/2]`, `target = ref center`. |
| `inside_minus_x_wall` | camera AT the inner face of the -X wall, looking +X. `obj.position = [ref.x - ref.size.x/2 + wall_inset, ref.y, ref.z + height_offset]`, `target = [ref.x + ref.size.x/2, ref.y, ref.z + height_offset]`, `up = [0, 0, 1]`. `wall_inset` defaults to `0.2` m, `height_offset` defaults to `0.6 * ref.size.z`. |
| `inside_plus_x_wall` | mirror on +X (looking -X). |
| `inside_minus_y_wall` | inner face of -Y wall, looking +Y. `obj.position = [ref.x, ref.y - ref.size.y/2 + wall_inset, ref.z + height_offset]`, `target = [ref.x, ref.y + ref.size.y/2, ref.z + height_offset]`. |
| `inside_plus_y_wall` | mirror on +Y (looking -Y). |

### Base patterns (no ref needed)

When `topology.role = base`, the planner gives `obj.pose.position` directly
as absolute coordinates. There is no relation formula — codegen reads the
plan and emits the literal numbers as named constants.
