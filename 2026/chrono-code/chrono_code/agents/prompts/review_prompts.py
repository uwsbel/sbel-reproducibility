"""
Prompts for the Review Agent (Agent 3) - per-image visual description only.
"""


from string import Template
VLM_SINGLE_IMAGE_PROMPT = Template("""Describe this PyChrono simulation camera image (${camera_label}) as objective visual evidence only.

Simulation objectives:
${objectives}

Plan summary:
${plan_summary}

Do not judge correctness, do not recommend fixes, and do not output approval decisions.
Only describe what is visibly present in the image.

Return JSON only:
{
  "description": "3-6 sentence factual description of what is visible in this image",
  "camera_view": "short description of the viewpoint/angle",
  "visible_objects": ["object or structure names that are clearly visible"],
  "observations": ["notable visual facts, spatial relationships, or anomalies"]
}
""")
STEP_REVIEW_PROMPT = Template("""You are reviewing ONE step of an incremental scene-building process.

Step ${step_number} of ${total_steps}: "${step_description}"

Previously completed steps:
${completed_steps_text}

Look at the camera image(s) and determine whether this step has been correctly implemented.
Specifically check:
1. Is the object or change described in the step visible in the scene?
2. Is it positioned and oriented correctly according to the step description?
3. Does it look reasonable and stable (not floating, not clipping through other objects)?

Return JSON only:
{
  "pass": true or false,
  "description": "2-4 sentence description of what you see relevant to this step",
  "issues": ["list of specific problems if pass is false, empty list if pass is true"]
}
""")
STEP_VLM_DESCRIBE_PROMPT = Template("""You are observing the rendered output of step ${step_number} of ${total_steps} in a multi-step physics simulation. Your job is to NARRATE the scene and report what you see — NOT to judge correctness.

## Step description (what the planner says this step builds)
${step_description}

Previously completed steps:
${completed_steps_text}

## Expected scene objects (canonical names + roles from the plan)
${scene_objects_manifest}

## Plan-level assets that must also be reported (vehicles, robots, external assets)
These are NOT in the procedural scene_objects manifest above but DO appear in the rendered scene and MUST get their own row in the ``objects`` array below. ``is_dynamic=True`` assets MUST be assessed for motion across the video — a stationary dynamic vehicle is the single most important failure mode the downstream judge needs to catch, and it can only catch it if you give it a per-asset ``motion_state``.
${plan_assets_manifest}

GUIDELINES

* Use canonical names from the manifest whenever you can identify a visible shape with one of them (role + primitive + size hints disambiguate e.g. ``left_platform`` vs ``right_platform``). For shapes that do NOT match any manifest entry, use ``unmapped: <short generic phrase>``.

* FSI scenes show two visual layers — solid surfaces (continuous shaded geometry — walls, slabs, plates) vs marker dots (BCE grid showing the FSI boundary, or SPH velocity-coloured particle cloud). Dot patterns alone are NOT solid geometry.

* A 2D screenshot is not a measurement instrument. Do not compute metric sizes from pixel ratios.

* Describe ONLY what you see. Do NOT speculate about what objects SHOULD do — that is the next stage's job. The agent that consumes your output has the step description and will do the consistency check itself.

MANIFEST ``size`` SEMANTICS — ``size=[X,Y,Z]`` in the manifest above is per role; misreading it is the main source of bogus claims. Before mentioning anything geometric, anchor on these definitions:

  * ``role=fluid_domain`` (e.g. ``sph_water``): ``size`` is the
    COMPUTATIONAL DOMAIN — the maximum AABB where SPH particles are
    allowed to exist. The actual fluid level at any moment is
    state-dependent and governed by gravity + initial particle count.
    Half-filled / partially-filled fluid is the NORMAL settled state,
    NOT an inconsistency. Do NOT claim "water should fill the tank
    because manifest size matches container size" — that is wrong.
  * ``role=*fsi_container*`` (e.g. ``water_tank_boundary``): ``size``
    is the container body's AABB. Open-top containers normally show
    BCE marker dots only on floor + side walls (no top markers); that
    is by design, not a missing wall.
  * ``role=*support_platform*`` / fixed bodies: ``size`` is the body's
    geometric extent. From a side / wide-angle camera view two
    platforms separated by a tank may visually overlap the gap because
    the camera looks ALONG the separation axis — that is foreshortening,
    NOT a "single continuous slab".
  * ``role=floating_*`` / dynamic bodies: ``size`` is the body's
    geometric extent at construction time. Resting position depends on
    buoyancy / contact / motion, not on the size field.
  * ``role=*chassis*`` / ``*spindle*`` / ``*wheel*``: same as floating —
    ``size`` describes the body, not where it ends up at the timestep
    you're looking at.

REPORT FORMAT

For each entry in the expected scene objects list AND each entry in the plan-level assets list, state whether you can identify it visually, what motion state it appears to be in, and a short relative location. Use canonical manifest names. Stick to factual observation — relations between bodies, what is moving vs at rest. No judgments about correctness, no derived sizes from pixels.

MOTION-STATE DISCIPLINE (critical for the downstream judge):

* You are watching a VIDEO, not a single frame. Compare frames across the clip — if a body's pixel position is the same at the start and at the end (allowing for camera motion), its ``motion_state`` is ``static``, regardless of what the surrounding scene's narrative would suggest.
* For chase / follow cameras, the camera moves with the subject. A vehicle that stays centered in a chase view while the world translates underneath it MAY still be moving in world frame — but if the world (water, platforms, ground) ALSO doesn't translate relative to the camera, the vehicle is NOT moving. Cross-check against background bodies before labeling a vehicle ``moving``.
* Do NOT infer motion from contextual cues (a vehicle near a ramp → "drives up the ramp", a plate near water → "the plate sinks"). Only label ``moving`` / ``falling`` / ``oscillating`` when you can point to a frame-to-frame pixel displacement of the body itself.
* When a dynamic-asset row in the plan-level list is reported as ``static`` and the step description says it should be moving, that is the EXACT signal the judge needs — do NOT soften it into ``unclear`` to avoid being wrong. Honest ``static`` on a body the plan said should drive is more useful than ``moving`` from confabulation.

``motion_state`` values:
  * ``static``       — visibly at rest across the entire clip (no frame-to-frame pixel displacement of THIS body relative to the background)
  * ``moving``       — actuated motion verified by frame-to-frame displacement of this body relative to the background (vehicle driving, robot walking, manipulator articulating)
  * ``falling``      — free-fall / settling under gravity, not yet at rest
  * ``oscillating``  — visible jitter / wobble around a rest pose
  * ``unclear``      — only one frame available, or body too small to tell

Return JSON only:
{
  "description": "3-6 sentence factual narration of the rendered scene",
  "visible_objects": ["<canonical name>", ...],
  "objects": [
    {
      "name": "<canonical name from manifest>",
      "present": true|false,
      "motion_state": "static|moving|falling|oscillating|unclear",
      "location": "<short relative description, or empty string when not present>"
    },
    ...
  ]
}
""")
STEP_REVIEW_DECISION_PROMPT = Template("""You are judging step ${step_number} of ${total_steps} in a multi-step physics simulation. Your role is to check whether the visible scene matches the step description and the declared object list. You are NOT enforcing physics rules — the deterministic CSV-backed validators upstream do that.

## Step description (the planner's statement of what this step builds)
${step_description}

## Step motion contract (authoritative — the planner's per-step decision)
This step's ``motion_expectations`` (body names that should move during
THIS step):
${step_motion_expectations_block}

Previously completed steps:
${completed_steps_text}

## CSV Validation Result (deterministic, CSV-backed; authoritative for the items it covers)
${csv_summary}

The "Body end-states" table at the end of the CSV section above (when
present) shows the final physics state of every body: position,
velocity magnitude |v|, angular velocity magnitude |ω|, and whether
the plan declared the body dynamic. Treat this table as PHYSICS
GROUND TRUTH:

  * For bodies marked is_dynamic, evidence of motion (nonzero |v| or
    |ω|, or position drifted from the plan-predicate position) confirms
    the simulation actually advanced. A dynamic body with |v|=0 AND
    |ω|=0 AND position == its plan-predicate position is a strong
    "physics-never-advanced" smell — flag it.
  * For static bodies, position should match the plan predicate within
    a few cm.
  * When the table conflicts with the visual observation report (e.g.
    table says floating_plate at z=0.92 with |v|=0.04, but VLM
    description says "floating_plate not visible"), the table wins —
    raw position/velocity numbers outweigh perception of the rendered
    mp4 image.

The "Per-step motion summary (declared-moving bodies — from
cam/motion_log.csv)" block, when present, lists Δp (start→end
position drift) and peak |v| during this step's run for every body
the planner explicitly named in ``step.motion_expectations``. Treat
this block as the AUTHORITATIVE motion check for those bodies:

  * Both Δp ≈ 0 (≤ 0.01 m) AND peak |v| ≈ 0 (≤ 0.05 m/s) for a
    declared-moving body = the body never moved → FAIL the step. The
    typical causes are (a) ``WheeledVehicle`` constructed with the
    standalone ``(filename, ChContactMethod_SMC)`` overload, putting
    the bodies in an orphan ``ChSystem`` that ``sysFSI.DoStepDynamics``
    never advances, (b) brake never released / driver inputs not
    wired, (c) FSI coupling not registered. Name a concrete cause in
    ``issues``.
  * "absent from CSV" for a declared-moving body = codegen forgot to
    include that body in its trajectory dump → FAIL with a directive
    asking codegen to extend the on_step callback.
  * A WARNING line about motion_log.csv being missing on a step that
    declared motion_expectations is the same signal — FAIL.
  * Non-trivial Δp OR non-trivial peak |v| = the body moved enough to
    pass the motion check, even if the visual narration is unsure.
    Be lenient on the magnitude — small drift, oscillation, settling
    all count as movement; you are looking for clearly-stuck bodies,
    not borderline motion. Empty motion_expectations for the step
    means there is no per-step motion contract to enforce; skip this
    paragraph.

WHEN THE CSV SECTION SAYS "CSV validation not available." (no Body
end-states table follows): the simulation did NOT write its
post-loop CSV outputs — almost always because the subprocess was
killed by the wall-clock timeout before reaching ``t_end``. Treat
this as a STRONG fail signal, not a lean-toward-PASS. In this case
you have NO physics ground truth, only the VLM video narration —
and a video that shows a stationary vehicle near a plate over water
is indistinguishable from a working "drives onto plate" scene to a
text describer that has been primed by the step description. The
correct verdict when CSV is unavailable AND the plan declares
dynamic motion that the VLM cannot positively verify (per-asset
``motion_state=moving``) is FAIL with reasoning like
"deterministic CSVs missing — likely simulation timed out before
finishing; need lower particle count / coarser dT / shorter t_end".

## Visual observation report (from a prior VLM pass; image attached too)
${vlm_description}
${rebuttal_block}
## Plan-Level Assets Declared (must EVENTUALLY appear in the rendered scene)
${plan_assets_manifest}

## Procedural Scene Objects (canonical names from the plan)
${scene_objects_manifest}

DECISION RULES

1. DETERMINISTIC FINDINGS are the physics ground truth. Any
   ``WHEEL_LANDING / INTERPENETRATION / FLUID_CONTAINMENT: FAIL``
   already failed the step upstream and you would not be reading this
   prompt. So in practice the CSV section here is either ``PASS``,
   ``SKIPPED``, or "not available". Treat ``PASS`` as a green light
   for everything it covers — do NOT re-litigate wheel contact,
   interpenetration, or fluid leakage from the image.

2. NAME ALIASING — the step description is free text and may refer to
   bodies abstractly (e.g. "concrete platform supports", "the water
   tank", "a floating bridge"). Before flagging anything missing,
   apply role-based mapping to canonical names:

     * "platform supports" / "concrete platforms"  ↔  role=*support_platform*
       (often appears as left_platform + right_platform — count both as
       one logical concept, not two missing entries)
     * "water tank" / "tank boundary"              ↔  role=*fsi_container*
     * "floating plate/block/bridge"               ↔  role=*floating_*
     * "water" / "fluid"                           ↔  role=*fluid_domain*

3. CONSISTENCY CHECK — read the visual observation against the step
   description. Pass when:

     a. Every scene_object the step description introduces is reported
        ``present=true`` in the visual observation's ``objects`` table
        (after name-aliasing).
     b. Each present object's ``motion_state`` is plausible for what
        the step description says is happening at this point — e.g. a
        body the step says "is parked on the platform" should be
        ``static``, a body the step says "is driving" should be
        ``moving``, a body the step says "begins to fall" can be
        ``falling`` or ``oscillating``. Use judgment; do not require an
        exact label match.
     c. The narration's account of relative layout does not contradict
        the step description in a structural way (e.g. "floating plate
        bridges the platforms" but the plate is rendered inside the
        tank instead of on top of it).

3a. PER-STEP MOTION CHECK — gated on ``step.motion_expectations``,
    NOT on plan-level ``is_dynamic`` or asset role.

      * EMPTY ``step_motion_expectations``: setup / braked / build-only
        step. Every body may report ``motion_state=static``. Do NOT FAIL.
      * NON-EMPTY ``step_motion_expectations``: every listed name MUST
        report ``motion_state ∈ {moving, falling, oscillating}``. Any
        listed name with ``motion_state=static`` is a FAIL — name a
        concrete cause (orphan ``ChSystem``, brake never released,
        driver inputs not wired, FSI coupling unregistered).
      * Bodies not on the list have no motion requirement — skip them.

    Do not infer expected motion from narrative verbs, asset role
    names, or plan-level ``is_dynamic``.

4. PLAN-LEVEL ASSET CARRY-OVER — every entry under ``## Plan-Level
   Assets Declared`` must EVENTUALLY appear. If this is the LAST step
   (${step_number} == ${total_steps}) and a declared asset has STILL
   not appeared in any prior step's review, this step FAILS.

5. FORWARD-LOOKING DYNAMICS CARVE-OUT — step descriptions sometimes
   mention phenomena that only emerge AFTER later steps add interacting
   objects (e.g. "configure for visible tire ruts" on a terrain-only
   step before the vehicle exists). The absence of forward-looking
   phenomena is NOT a failure of the current step.

6. RENDERER-WIRING HEURISTIC — when the visual observation reports a
   manifest body as ``present=false`` and the manifest clearly lists
   it (so codegen probably built it), the failure is most often at the
   RENDERER, not the geometry. When you fail under this rule, name a
   likely renderer cause in ``issues`` so codegen edits the wiring
   instead of rebuilding bodies that already exist. Common patterns:

     * SPH fluid markers default to ON, so the blue SPH water cloud
       SHOULD be visible in the mp4. If it is not, that's a real bug
       (likely `EnableFluidMarkers(False)` or visualizer attach order).
       The BCE diagnostic overlays (boundary / rigid-body green dot
       grids) are the ones that default OFF — their absence is normal
       and not a failure signal.
     * Container/tank not visible while ``ChWheeledVehicleVisualSystemVSG``
       is in use → ``vis.AttachSystem(sysMBS)`` missing
       (the vehicle-aware visualizer does NOT auto-attach MBS).
     * Only wheels visible, no chassis →
       ``vehicle.SetChassisVisualizationType(MESH)`` not called.
     * Scene appears 90° rotated → ``vis.SetCameraVertical(...)`` not
       set, or set to the wrong axis.

7. NO BCE MARKER DIAGNOSTICS — the BCE marker overlays (green dot
   grids on tank walls / floating bodies) are OFF by default. Do NOT
   fail a step on "BCE markers missing" or "BCE markers mis-aligned"
   — those visual checks no longer apply. (This does NOT cover the
   blue SPH fluid particles, which ARE rendered by default — those
   you should see.) Physics correctness comes from the Body
   end-states table and the deterministic CSV findings above.

7a. SPH CONTAINMENT — when the scene includes an SPH ``fluid_domain``
    plus an ``fsi_container`` tank, the blue particle cloud should
    stay INSIDE the tank's AABB throughout the clip. If the VLM
    narration mentions any of "particles spreading outside the tank",
    "water leaking under the platforms", "spray fanning out
    horizontally", or "particles disappearing below z=<floor>", that
    is a HARD FAIL — the simulation built the geometry but the fluid
    is escaping its container. Typical cause: a non-periodic axis
    where the BCE wall layer is 0 (planner declared
    ``boundary_conditions: <axis>_periodic`` on the fluid but codegen
    omitted ``sysSPH.SetComputationalDomain(..., BC_<AXIS>_PERIODIC)``).
    Do NOT treat this as "the tank is partially open by design";
    open-axis containment requires the periodic boundary on that
    same axis. Name "BC_*_PERIODIC missing" or "tank wall layer 0
    on non-periodic axis" in ``issues``.

8. WHEN UNCERTAIN — prefer PASS, BUT only when the CSV section above
   actually provided a Body end-states table or PASSed deterministic
   validators. When CSV said "not available" AND the plan declares
   dynamic motion you cannot verify from the per-asset
   ``motion_state`` column, prefer FAIL (see the missing-CSV section
   above). Your role is asset-presence, motion-state plausibility, and
   step-description consistency — NOT visual-ratio adjudication. Do
   NOT fail on:
     * fluid_domain looking partially full (size = max BCE extent, not
       fill level — fluid settles under gravity)
     * two named bodies appearing flush along the camera axis (depth
       compression, not merged geometry)
     * a body's visible thickness "looking thinner" than its declared
       size (a screenshot is not a measurement instrument)

Return JSON only:
{
  "pass": true|false,
  "reasoning": "<2-4 sentences>",
  "issues": ["<specific issue, empty list when pass=true>"]
}
""")
