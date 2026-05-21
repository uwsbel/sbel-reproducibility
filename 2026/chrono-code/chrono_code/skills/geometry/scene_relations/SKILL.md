---
name: scene_relations
description: >
  Translate a `geometry_relations` entry from the plan into correct
  PyChrono coordinate code. Read this whenever the plan declares a
  relation_name you have not previously encoded, and especially BEFORE
  writing SetPos / camera placement for any body that participates in a
  multi-body geometric constraint. Each subsection below is one canonical
  pattern named exactly as it appears in `plan.geometry_relations[i].relation_name`.
compatibility: pychrono >= 8.0
metadata:
  domain: geometry
---

# Skill: scene_relations — geometry-relation patterns

## Numeric examples below are TEMPLATES, not defaults

Every concrete number in the worked examples (e.g. `scene_x_extent = 6.0`,
`platform_size`, `bxDim/2`) is illustrative and shows how to compose
formulas from named variables — they are NOT default sizes. If the user
did not specify a size, raise a structured clarification per
`GEOMETRY_RELATION_RULES` and substitute the user's chosen value into the
formula. Never silently copy a number from this skill into a plan.

## How to use this skill

The plan you receive contains a `geometry_relations` list. Each entry
names two bodies (`body_a`, `body_b`), a `relation_name`, and a
`parameters` dict.

For every entry whose `relation_name` is **not** `TO_CLARIFY`:

1. Locate the subsection in this file with the matching heading.
2. Read its **Intent**, **Coordinate derivation**, and **Worked example**.
3. Substitute the body names and sizes from the plan into the worked
   example. Do NOT re-derive coordinates; the derivation is already in
   the **Coordinate derivation** block.
4. Read the **Common mistakes** list and confirm your emitted code does
   not match any anti-pattern.

If `relation_name == TO_CLARIFY`, **do not emit coordinates** for that
relation — the planner has not resolved which pattern applies. The
workflow loops back to the planner to clarify before codegen continues.

If the plan names a `relation_name` that does NOT have a subsection
here, stop. Do not invent a coordinate scheme; ask for the skill to be
extended (write a `clarifications_needed` entry indicating the pattern
is undefined). Inventing a relation is precisely the failure this skill
is designed to prevent.

---

## platform_flush_wall_outer

**Parameters.** `wall: <"-x" | "+x" | "-y" | "+y">`,
`platform_top_aligned_with: <"wall_top" | "water_surface">` (optional;
defaults to `wall_top`).

**Intent.** A platform sits OUTSIDE the tank with its inner edge flush
against the OUTER face of a tank wall. Used so a vehicle starting on
the platform does not collide with the wall when it drives off.

**Coordinate derivation.**

Tank is centered at the origin. Along the `wall` axis the tank interior
half-extent is `bxDim/2` (or `byDim/2`); the wall thickness is
`wall_thickness`. The wall's outer face is at
`±(half_extent + wall_thickness)`. For platform full size
`platform_size.x` along that axis, the platform center sits half its
own length further out:

    sign = -1 if wall ∈ {"-x", "-y"} else +1
    center_along_axis = sign * (half_extent + wall_thickness + platform_size.x / 2)

For `platform_top_aligned_with: wall_top`: the wall's top is at
`bzDim + wall_thickness` (because the wall slab spans from
`-wall_thickness` floor to `bzDim + wall_thickness` top in tank-centered
z). With platform thickness `platform_size.z`,

    center_z = bzDim + wall_thickness - platform_size.z / 2

For `platform_top_aligned_with: water_surface`: rare; vehicle has zero
clearance to the water rim:

    center_z = bzDim - platform_size.z / 2

The transverse axis (perpendicular to `wall`) takes the tank's
transverse center for that axis (typically 0).

**Worked example (left platform, wall = "-x", wall_top aligned).**

```python
left_platform = chrono.ChBodyEasyBox(
    platform_size.x, platform_size.y, platform_size.z,
    1000.0, True, True, cmaterial,
)
left_platform.SetFixed(True)
left_platform.SetPos(chrono.ChVector3d(
    -1.0 * (bxDim / 2 + wall_thickness + platform_size.x / 2),
    0.0,
    bzDim + wall_thickness - platform_size.z / 2,
))
```

**Common mistakes.**
- Putting the platform's edge at the wall's INNER face (`bxDim/2`)
  produces a `wall_thickness`-wide overlap. iter_001 hit this bug.
- Using `bzDim - initial_spacing` (or `bzDim`) as `center_z` aligns the
  platform top with the water surface, not with the wall top. The
  vehicle then sees a `wall_thickness`-tall step climbing onto the
  wall edge.
- Setting the platform's transverse extent smaller than the tank's
  transverse extent leaves a side gap. For a periodic-y tank
  (`BC_Y_PERIODIC`), the platform y must equal `byDim`.

---

## floating_box_at_water_surface

**Parameters.** none required; pass the body's full `size`.

**Intent.** A floating rigid box's initial pose places its BOTTOM face
at the still-water free surface, so equilibrium is reached by the box
SINKING under load, not by being EJECTED out of an initially submerged
state.

**Coordinate derivation.** Let `free_surface_z` be the still-water
height (`bzDim` for tanks built with corner-origin bottom at z=0;
`bzDim/2` for tank-centered z=0). For a box of size `(sx, sy, sz)`:

    center_z = free_surface_z + sz / 2

`center_x` and `center_y` are taken from the plan; the relation only
constrains the vertical axis.

**Worked example (tank-centered z=0 case).**

```python
plate_size   = chrono.ChVector3d(0.9 * bxDim, 0.7 * byDim, 4 * initial_spacing)
plate_center = chrono.ChVector3d(0, 0, bzDim + plate_size.z / 2)
floating_plate.SetPos(plate_center)
```

**Common mistakes.**
- Setting `center_z = bzDim/2` (or any value below `free_surface_z`)
  makes the box start fully submerged; buoyancy ejects it on the first
  step and the simulation oscillates.
- Choosing a box density >= `fluid_density` while using this relation —
  the relation name implies the body floats. If the user wants a sinking
  weight, request a different pattern instead of overriding this one.

---

## camera_side_minus_y

**Parameters.** none required; the framing is taken from the scene's
overall AABB.

**Intent.** A pure −Y side view that frames the entire xz facade of
the scene, including any platforms or external bodies that extend
beyond the tank.

**Coordinate derivation.** Let `scene_x_extent` be the half-width of
the scene along x (i.e. the larger of `|x_min|` and `|x_max|` over all
visible bodies, including platforms). Let `scene_top_z` be the highest
visible z (typically `bzDim + wall_thickness` plus any platform
thickness).

    cam_pos    = (scene_center_x, -scene_x_extent * 1.7, scene_top_z * 1.3)
    target_pos = (scene_center_x, 0,                     scene_top_z / 2)

The `1.7` factor accommodates VSG's default ~45° horizontal FOV. If
the rendered frame still clips platforms, scale `cam_pos.y` by another
1.5x and re-run.

**Worked example.**

```python
scene_center_x = 0.0
scene_x_extent = 6.0   # tank half-width 2 + platform width 4 = 6
scene_top_z    = 1.4   # wall top 1.0 + wall_thickness 0.2 + platform thickness 0.2

cam_pos    = chrono.ChVector3d(scene_center_x,
                               -scene_x_extent * 1.7,
                                scene_top_z * 1.3)
target_pos = chrono.ChVector3d(scene_center_x, 0.0, scene_top_z / 2)
lock_side_camera(vis, cam_pos, target_pos)
```

**Common mistakes.**
- Pointing the camera at the tank center while platforms extend ±6m in
  x — only the tank is in frame, platforms clip out. `scene_x_extent`
  must include any external bodies, not just the tank.
- Using a margin factor < 1.5 — VSG's default FOV clips edges.
- Omitting `lock_side_camera` and relying on `SetCameraPosition` alone:
  vehicle visualizers re-compute the camera each render unless the
  chase camera state is set to Free; iter_001 lost the side view this
  way before the helper was added.
