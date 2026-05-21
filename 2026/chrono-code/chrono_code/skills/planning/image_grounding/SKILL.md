---
name: image_grounding
description: Procedure for grounding scene plans in attached reference images — enumerate, describe, orient, then derive predicates.
compatibility: pychrono >= 8.0
metadata:
  domain: planning
---
# Skill: Image Grounding for Scene Plans

## Purpose

When a user attaches one or more images as the scene spec, the image is the authoritative source for layout and orientation — not a decorative prop. This skill defines the required procedure for turning an image into scene_predicates so the downstream simulation matches what the user drew or photographed.

Use this skill whenever a planning prompt has `image_observation` as a required section and one or more images are attached.

## Procedure

Perform these steps in order. Do not shortcut.

1. **Enumerate visible objects.** For every object you recognize, write one line: `<object_name>: <what it is>`. Use names that match the `assets[]` entries you plan to emit.

2. **Relative positions.** For each ordered pair of main objects, describe the relative position in image-native terms (`left_of`, `right_of`, `in_front_of`, `behind`, `on_top_of`, `below`). Use the schema `<A>: <relation> <B>` with generic placeholder names (`<A>`, `<B>`), not a specific object pair.

3. **Orientations.** For every directional object (monitor, TV, chair, laptop, camera, lamp head, robot, vehicle, any actor), state what it is FACING by naming the target object. Schema: `<A>: facing <B>`. If a chair has a clearly visible back/seat, the seat direction tells you where the occupant would be looking.

4. **Viewpoint.** Describe the camera viewpoint of the image relative to the primary subject(s) using the schema `viewpoint: <relation_name> [— <one-line justification>]`. Pick the closest match from the closed list:
   - `top_down` — the image is shot from directly above, looking down.
   - `side_minus_y` — the image is shot from the −Y side of the subject (subject's left, world frame).
   - `side_plus_y` — from the +Y side (subject's right).
   - `side_minus_x` — from the −X side (subject's back).
   - `side_plus_x` — from the +X side (subject's front).
   - `perspective` — a corner / 3/4 view, neither pure side nor pure top-down. Use this when in doubt.

   Use ONLY these six names. The same enum drives `topology.relation` for camera objects in the plan, so the choice you write here will become the actual simulation viewpoint.

5. **Cross-check user prompt vs image.** If the prompt and the image disagree on placement or count, the image wins unless the user explicitly overrides the image in text.

## Deriving predicates from observations

Only after the `image_observation` block is written, derive `scene_predicates`. Apply these ordering rules:

- Use `FACING-TO` as the primary orientation driver. Whenever you observed a "facing" relation in step 3, emit a `FACING-TO` predicate for it first.
- Derive `FRONT-OF` / `BACK-OF` / `LEFT-OF` / `RIGHT-OF` FROM the facing direction — not from priors about "where things usually go". If `<A> FACING-TO <B>` and `<A>` sits on a support surface, `<B>` must lie on the side `<A>`'s forward axis actually points to in the scene's world frame.
- Do not reuse the numeric positions from the `scene_coordinate_system` worked example verbatim. Those are illustrative only. Recompute `(x, y, z, deg_z)` from YOUR predicates and the asset target heights.
- If an object in the image has no obvious front/back (a cup, a sphere), do not invent a `FACING-TO` for it.
- The `viewpoint` observation drives the **first camera of step 1**: at least one camera in the first implementation step MUST adopt the same `relation` as the observed `viewpoint`. Other cameras (when `recording_mode: sensor_cams` requires 2-3) remain free to pick complementary angles.

## Output contract

The `image_observation` block must appear verbatim in the plan output as a top-level string field named `image_observation`, so it can be audited.

During refinement or repair, re-describe the image in a fresh `image_observation` block; do not reuse the previous one unless it is still correct. Compare new observations against existing `scene_predicates` and fix any contradictions, even if the user's clarification did not explicitly mention them.
