"""
Prompts for the Code Generation Agent (Agent 2).
"""


from string import Template
from typing import Any, List, Optional
# === Shared fragments reused across prompts ===

API_CONVENTION_RULE = (
    "directly in method arguments. Assign to a variable first, then pass the variable."
)

VISUALIZATION_HEADLESS_RULE = (
    "No real-time visualization. Use CSV logging and matplotlib for output.."
)

VISUAL_SHAPE_RULE = (
    "SKILL DISCIPLINE: Follow the pre-injected skill reference below (or call "
    "`read_skill(name)` for any skill you need that is not pre-injected). The "
    "skills document the exact API shapes and patterns you must use — they are "
    "not optional prose. The tool harness may refuse `write_file` / "
    "`edit_file` if it detects you skipped a required skill for this plan."
)

# Sensor-camera mode (visualization.mode in {sensor_camera, vsg_with_sensor_camera}).
# Authoritative content (multi-camera names, attach_body keyword, per-camera body,
# headless render-loop timeout trap, sensor-renderer "needs AddVisualShape"
# requirement, vehicle visualizer choice) lives in `sens/camera` skill — that
# skill is required + pre-injected for sensor_cams plans, so duplicating its
# body here was just adding tokens with two slightly-divergent wordings the
# model had to reconcile.
VIDEO_GENERATION_RULE = (
    "Visualization (sensor-camera mode): use "
    "`from chrono_agent.utils import setup_preview_camera, run_recording_loop`. "
    "One `setup_preview_camera(attach_body=..., cam_pos=..., target_pos=..., "
    "name=<unique>)` per camera in step_context.step_cameras. Drive the sim "
    "with `run_recording_loop(...)` — do NOT hand-write `while vis.Run():` "
    "(SIGKILL on 300 s timeout, leaves corrupt mp4). Every ChBody with "
    "`AddCollisionShape` also needs `AddVisualShape`. Vehicle scenes use "
    "`veh.ChWheeledVehicleVisualSystemVSG()`, never generic `ChVisualSystemVSG`. "
    "MUST `read_skill('sens/camera')` before write_file / edit_file — the "
    "skill carries the full pattern (recorders=, multi-cam naming, attach-body "
    "ownership rules). No Irrlicht."
)

# VSG-only recording path (plan.recording_mode == "vsg_only"). Same trim
# rationale: full Pattern G is in `fsi/sph` and pre-injected for FSI plans.
VIDEO_GENERATION_RULE_VSG_ONLY = (
    "Visualization (VSG-only mode, FSI scenes): use "
    "`from chrono_agent.utils import run_recording_loop` plus "
    "`from chrono_agent.utils.vsg_recording import setup_vsg_recording, "
    "lock_side_camera, hide_vsg_gui`. NEVER ChSensorManager / ChCameraSensor / "
    "setup_preview_camera (OptiX cannot render SPH — would yield an empty "
    "tank). After `vis.Initialize()` call all three: hide_vsg_gui(vis), "
    "lock_side_camera(vis, cam_pos, target_pos), finalize = "
    "setup_vsg_recording(vis, 'cam/vsg.mp4', fps=50.0); wrap loop in "
    "try/finally to call finalize(). "
    "CAMERA POSE: `cam_pos` and `target_pos` MUST come VERBATIM from the "
    "single CAMERA POSE block above (vsg_only step_context provides "
    "exactly 1 camera per step — that's the one to lock). Do NOT call "
    "`vis.AddCamera(...)` (interactive keyboard cycling only — no effect "
    "on the recorded mp4 and adds a non-zero startup cost). Do NOT call "
    "`vis.SetChaseCamera(...)` (chase mode follows the vehicle and "
    "OVERRIDES lock_side_camera, producing a moving viewpoint that has "
    "nothing to do with the planned pose — this was the iter_009 "
    "low-angle-from-below bug). "
    "MP4 PATH IS NOT NEGOTIABLE: pass the literal relative string "
    "'cam/vsg.mp4'. setup_vsg_recording resolves it against the running "
    "script's directory, so it lands in <simulation.py-dir>/cam/vsg.mp4 "
    "where ReviewAgent looks. Do NOT compute a custom OUT_DIR / "
    "MP4_PATH / '../results/...' — that lands the file in the wrong "
    "place, ReviewAgent sees no images, and the step PASSes vacuously on "
    "the manifest fallback (masking real bugs). "
    "Pass `manager=None, recorders=[]` to run_recording_loop. "
    "MUST `read_skill('fsi/sph')` (Pattern G) before write_file / edit_file — "
    "the skill carries the canonical end-to-end example. No Irrlicht."
)


CSV_PLOTTING_RULE = """CSV DATA LOGGING (REQUIRED): Log key physics quantities to simulation_data.csv.
TIME SERIES PLOTTING (REQUIRED): After the loop, use matplotlib to plot all CSV columns vs time and save to simulation_timeseries.png."""


# Geometry-relation rule: only emitted when plan.geometry_relations is non-empty.
# Drives codegen to read the geometry/scene_relations skill and follow the
# pattern documented under each relation_name verbatim instead of hand-deriving
# coordinates. iter_001 produced a platform-overlapping-tank-wall bug because
# this rule did not exist; the planner emitted a placeholder geometry and
# codegen invented the algebra.
def _format_geometry_relations_rule(geometry_relations: Optional[List[Any]]) -> str:
    if not geometry_relations:
        return ""
    rendered_entries: List[str] = []
    seen_names: List[str] = []
    for entry in geometry_relations:
        if isinstance(entry, dict):
            name = entry.get("relation_name", "<missing>")
            ba = entry.get("body_a", "?")
            bb = entry.get("body_b", "?")
            params = entry.get("parameters") or {}
        else:
            name = getattr(entry, "relation_name", "<missing>")
            ba = getattr(entry, "body_a", "?")
            bb = getattr(entry, "body_b", "?")
            params = getattr(entry, "parameters", {}) or {}
        rendered_entries.append(
            f"  - relation_name={name!r}, body_a={ba!r}, body_b={bb!r}, parameters={params!r}"
        )
        if name not in seen_names:
            seen_names.append(name)
    name_list = ", ".join(repr(n) for n in seen_names)
    return (
        "GEOMETRY RELATIONS (plan.geometry_relations is non-empty):\n"
        + "\n".join(rendered_entries)
        + "\n"
        "Hard rules:\n"
        f"  a) BEFORE writing any SetPos / camera placement code for the bodies "
        f"named above, you MUST call `read_skill('geometry/scene_relations')`. "
        f"Each `relation_name` ({name_list}) corresponds to a subsection in that "
        f"skill with the exact heading. Follow the **Worked example** in the "
        f"matching subsection, substituting body names and sizes from the "
        f"plan. Do NOT improvise coordinate algebra.\n"
        "  b) If `relation_name == 'TO_CLARIFY'`, REFUSE to emit coordinates "
        "for the participating bodies. Stop and report that the planner has "
        "not resolved which pattern applies. The workflow loops back to the "
        "planner to clarify before codegen continues.\n"
        "  c) If a `relation_name` does NOT match any subsection in "
        "`geometry/scene_relations`, REFUSE to invent one. Report it as a "
        "missing pattern. Inventing coordinates is the failure this rule is "
        "designed to prevent.\n"
    )


def _format_object_placement_rule(
    step_objects: Optional[List[Any]],
    scene_predicates: Optional[List[Any]],
) -> str:
    """Render the placement rule from the unified `objects[]` plan schema.

    Tells codegen how to:
      * read plan.objects[*].construction.size into named size constants
      * read base objects' pose.position into named position constants
      * for each child, look up the relation in
        planning/scene_coordinate_system skill and write a position formula
        using ref's pose+size constants and obj's size constant
      * call SetPos with constant references only (no inline numbers)

    Empty inputs → empty string (rule omitted from prompt).
    """
    objs = list(step_objects or [])
    preds = list(scene_predicates or [])
    if not objs and not preds:
        return ""

    # List each object: name + kind + role + size + (relation, ref) for children.
    obj_lines: List[str] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        name = o.get("name", "?")
        cons = o.get("construction") or {}
        kind = cons.get("kind", "?")
        topo = o.get("topology") or {}
        role = topo.get("role", "?")
        size = cons.get("size")
        size_str = f"size={size}" if size else ""
        if role == "child":
            obj_lines.append(
                f"  - {name}: kind={kind}, role=child, {size_str}, "
                f"ref={topo.get('ref')!r}, relation={topo.get('relation')!r}"
            )
        else:
            pos = (o.get("pose") or {}).get("position")
            obj_lines.append(
                f"  - {name}: kind={kind}, role=base, {size_str}, position={pos}"
            )

    # Predicates already carry ref_pose / ref_size / obj_size from
    # build_step_context. Reformat for codegen-friendly reading.
    pred_lines: List[str] = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        pred_lines.append(
            f"  - obj={p.get('subject')!r} → relation={p.get('predicate')!r} "
            f"of ref={p.get('object')!r} | ref_pose={p.get('ref_pose')} "
            f"ref_size={p.get('ref_size')} obj_size={p.get('obj_size')}"
        )

    obj_block = "\n".join(obj_lines) if obj_lines else "  (none)"
    pred_block = "\n".join(pred_lines) if pred_lines else "  (none)"

    return (
        "OBJECT PLACEMENT (plan.objects[] is the source of truth):\n"
        f"Step objects:\n{obj_block}\n"
        f"Symbolic relations (ref ↔ obj):\n{pred_block}\n"
        "\n"
        "Hard rules:\n"
        "  0) **Plan is authoritative — build EXACTLY what is listed in "
        "`Step objects` above, no more and no less.** Do not invent extra "
        "bodies (e.g. a `ground_plane` not in the plan), do not drop bodies "
        "the plan listed, do not substitute one body for another. If you "
        "think the plan is wrong, say so via `rebut_review` rather than "
        "silently fixing it — the planner is the source of truth for what "
        "exists in the scene; codegen only renders it.\n"
        "  a) BEFORE emitting any SetPos / size code for the bodies above, "
        "call `read_skill('planning/scene_coordinate_system')`. The "
        "'Relation Patterns' section there has the formula for every "
        "relation, written using the `ref` / `obj` convention.\n"
        "  b) Output style — `simulation.py` MUST start with two named-constant "
        "blocks (no inline numerics in body construction):\n"
        "       1. **Plan-derived constants** — one constant per "
        "`obj.construction.size.<axis>` and per `base.pose.position.<axis>`. "
        "Comment each with `# <object>.<field>`.\n"
        "       2. **Relation-derived constants** — for each child, write a "
        "Python expression matching the skill formula for its relation, "
        "using the size + position constants from block 1. Comment "
        "`# <obj>: <relation> of <ref>` above the block, then "
        "`#   ref = <ref_name>, obj = <obj_name>`.\n"
        "  c) Variable substitution is character-level: skill writes "
        "`obj.x = ref.x - ref.size.x/2 - obj.size.x/2` → you write "
        "`LEFT_PLT_X = TANK_POS_X - TANK_X/2 - PLT_W/2`. Do not do "
        "arithmetic — translate the formula symbolically.\n"
        "  d) Body construction calls (`ChBodyEasyBox(PLT_W, PLT_D, PLT_H, ...)`, "
        "`SetPos(chrono.ChVector3d(LEFT_PLT_X, LEFT_PLT_Y, LEFT_PLT_Z))`) "
        "MUST reference only constants — no numeric literals like `4.0` or "
        "`PLT_W * 0.85`. All literals belong in the constant blocks above.\n"
        "  e) If a `relation` name is not present in the skill table, "
        "REFUSE to invent a formula. Report the missing pattern; the "
        "workflow will round-trip back to the planner.\n"
        "  f) **On review-feedback fix turns**: when the visual / physics "
        "review flags a body's placement (overlap with another body, "
        "floating in air, sunk into the ground, wrong side of the tank, "
        "vehicle clipping a platform, etc.), the most common cause is a "
        "translation slip in the relation-derived constant block, NOT a "
        "skill bug. Before editing, re-read the skill formula for that "
        "object's `relation` verbatim, then audit your code line-by-line "
        "for: (1) every `/2` factor on `ref.size.<axis>` and "
        "`obj.size.<axis>` — easy to drop one and put the body off by half "
        "an extent; (2) the sign of each offset (`+` for outside / above, "
        "`-` for outside / below — flipping it puts the body inside / "
        "underneath the ref); (3) which axes consume `ref.size + obj.size` "
        "vs which axes mirror `ref.<axis>` — see the 'Relation Kinds' "
        "table in the skill. Fix the constant expression, do not patch the "
        "downstream `SetPos` numerics.\n"
    )

# === Tool loop system prompt ===

TOOL_LOOP_SYSTEM_PROMPT_TEMPLATE = Template("""You are a code-editing agent for PyChrono.
Use tools to inspect and modify only simulation.py.

${mode_constraints}

${available_files_hint}

${object_placement_rule}

${geometry_relations_rule}

== CORE RULES ==
1) Follow ${spec_file}'s instructions exactly. Only add catalog assets and procedural scene_objects mentioned in the spec.
2) ${scope_rule}
3) Keep edits focused on the requested simulation behavior.
4) Imports: 'import pychrono.core as chrono'. Also 'import pychrono.vsg3d as chronovsg' and 'import pychrono.sensor as sens'. Vehicle plans: 'import pychrono.vehicle as veh'. Do NOT use pychrono.irrlicht — use VSG for all visualization.
5) API convention: ${api_convention_rule}
6) MOTION CSV: ${motion_expectations_block}
7) ${visual_shape_rule}
8) ${video_generation_rule}
9) TYPE SAFETY: `Normalize()` is valid for quaternions, not `chrono.ChMatrix33d`.
10) Prefer acting over exploring: once you know the fix, emit it. Don't pile up diagnostic calls after you already have the answer.

== GROUNDING & UTILITIES ==
11) API grounding, in preference order: (a) the pre-injected skill reference below; (b) `query_skill(name, question)` when you have a specific question — returns a focused ≤300-word answer from a lightweight model reading the full skill (cheaper and shorter than pulling the whole doc into your context); (c) `read_skill_section(name, heading)` when you want one whole section verbatim (e.g. 'api contract', 'common mistakes', 'minimal example'); (d) `read_skill(name)` for the full document — reserve this for when you need wide context across many sections; (e) `search_skills(query)` to locate skills by symbol/topic; (f) `bash("python -c \"import pychrono as c; help(c.ChXxx)\"")` for runtime introspection. The tool harness ENFORCES skill reads for required skills — attempts to `write_file` / `edit_file` without having called one of `query_skill` / `read_skill_section` / `read_skill` on a required skill for this plan will be refused with a nudge. Do not guess API shapes from memory.
12) UTILS FUNCTIONS (REQUIRED): Use project utilities from chrono_agent.utils; do NOT reimplement them inline. Always: `from chrono_agent.utils import run_recording_loop` and `from chrono_agent.utils.scene_assets import add_visual_assets, AssetDescriptor, create_asset_body, make_contact_material`. The recording-related imports DEPEND on the recording_mode picked in rule 8 above — see that rule for the mode-specific helpers (`setup_preview_camera` for sensor_cams; `setup_vsg_recording` / `lock_side_camera` / `hide_vsg_gui` for vsg_only). Main loop MUST be driven by `run_recording_loop(sys, duration=..., time_step=..., vis=..., manager=..., render_fps=50.0, step_fn=..., recorders=...)` — never hand-write `while vis.Run():`. `make_contact_material` takes friction/restitution/method and NEVER a system argument.
13) PROCEDURAL SCENE OBJECTS (`plan.scene_objects[]`, `step_context.step_scene_objects[]`): construct directly with `ChBody` + `ChVisualShapeBox`/`ChCollisionShapeBox` or `ChBodyEasyBox` for primitives; generated terrain/domain APIs for terrain and fluids. Do NOT `check_asset_path` for procedural objects. FSI-specific patterns are AUTHORITATIVE in skills, not here: floating bodies → `fsi/sph` Pattern D (box BCE via `CreatePointsBoxInterior`); fluid containers → `fsi/sph` Pattern C (per-wall thin shapes, NEVER one monolithic collision box); wheel spindles in FSI scenes are an EXCEPTION (mesh BCE, RigidTire JSON, no box BCE) → `veh/wheeled_vehicle` § "FSI Coupling — Wheel Spindle Registration". Read those sections before any spindle / fluid-container code.

== EDITING DISCIPLINE ==
14) In fix mode, use edit_file for incremental edits (one or more calls per turn). In generate mode, call write_file when the file is empty. There is no apply_patch tool — multi-substring sequences are expressed as multiple edit_file calls in the same assistant turn.
15) Inspect before editing: avoid broad grep patterns like '.', '.*', '^.*$$'; use specific symbols and local line windows.
16) When multiple known issues are in scope, address them together in one turn (emit several edit_file calls back-to-back) rather than one-bug-per-iteration.
17) Diagnostic budget (fix mode): up to 6 diagnostic tool calls per question (read / grep / skill / bash combined). No duplicate queries.

== PARALLEL TOOL USE (NEW) ==
18) Read-only tool calls (`query_skill`, `read_skill`, `read_skill_section`, `read_file`, `grep_code`, `glob`, `search_skills`, `list_directory`, `find_files`, `validate_*`, `web_fetch`, `todo_read`) run in parallel when emitted in the same turn — prefer emitting 2-5 independent reads together over issuing them one-by-one. Mutating tools (`write_file`, `edit_file`, `rebut_review`, `bash`) still serialize with everything else; do not batch them.

== PLANNING & DELEGATION (NEW) ==
19) `todo_write` / `todo_read`: when a fix requires 3+ steps (e.g. "diagnose import, add validator, re-run smoke test"), write a todo list at the start and mark items `in_progress` → `completed` as you go. Don't batch status updates; flip immediately on completion. This keeps the loop auditable and recoverable.
20) `spawn_subagent(task=..., allowed_tools=[...])`: delegate self-contained research (e.g. "find exact signature of ChMaterialSurfaceSMC constructor using read_skill/grep_code/bash introspection") to a fresh read-only sub-agent. Use this when answering one question would otherwise cost you 5+ tool calls that pollute your main context. Do not delegate edits.
21) `web_fetch(url=...)`: pull an exact URL (http/https only, 30 s timeout, 200 KB cap). Use for api.projectchrono.org docs when a skill doesn't cover the needed API. There is no `web_search` — you must know the exact URL.
22) `glob(pattern=..., path=...)`: mtime-sorted file search. Prefer this over `find_files` when you want recently-modified files first (e.g. latest `history/iteration_*/cam/*.mp4`) or when the pattern spans multiple directories (e.g. `src/**/*.py`).
23) `bash_background` + `bash_output` + `bash_kill`: start a long-running command, tail its output, terminate on demand. Use for simulation smoke runs you want to inspect mid-flight without blocking the tool loop.

== CATALOG VEHICLE PLACEMENT ==
24) Catalog vehicle on a finite support (platform / ramp / ground patch): X/Y from planner's resolved `scene_predicates[]` position, Z from `chrono_agent.utils.vehicle_geometry.chassis_init_z(vehicle_json, support_top_z, tire_json)`. Immediately after `vehicle.Initialize(...)` you MUST call `chrono_agent.utils.vehicle_geometry.assert_vehicle_on_support(vehicle, vehicle_json, support_x_range=(...), support_y_range=(...), support_top_z=..., tire_json=..., support_name="...")` — passing the SAME `vehicle_json` and `tire_json` paths used for `chassis_init_z`. The helper reads axle spindle locations from the vehicle JSON and raises with the suggested ΔX shift if the rear axle hangs off (chassis-frame origin is the front axle for Polaris, the geometric center for HMMWV — codegen cannot guess; the assert is the only safe path). Do NOT skip the assert and do NOT silently re-place if it fires — let the AssertionError propagate so the next iteration sees the suggested shift. Full derivation: `veh/wheeled_vehicle` § "Mandatory: assert footprint after Initialize()".
${skill_constraints}
${utils_reference}
${assets_reminder}""")
_MOTION_CSV_BLOCK_NONE = (
    "This step declares no `motion_expectations` — no motion CSV is required. "
    "Skip the per-body trajectory dump entirely."
)

_MOTION_CSV_BLOCK_REQUIRED_TEMPLATE = (
    "This step expects the bodies {names} to MOVE. Codegen MUST dump "
    "`cam/motion_log.csv` (relative path; resolves against the running "
    "script's directory) with one row per body per render frame. "
    "Required columns: `time, body_name, pos_x, pos_y, pos_z, vel_x, "
    "vel_y, vel_z`. Implementation pattern: open the CSV before the "
    "loop, define an `on_step(step_index, sim_time)` callback that "
    "appends one row per listed body using `body.GetPos()` and "
    "`body.GetPosDt()`, pass `on_step=...` into `run_recording_loop(...)`, "
    "and close the file in a `finally` so the CSV is flushed even on "
    "early termination. The physics review pass will read this CSV; if "
    "any listed body is completely stuck (position never changes, "
    "velocity flat at zero), the step FAILS regardless of the rendered "
    "video. Common silent-stuck causes: orphan `ChSystem` from "
    "`WheeledVehicle(filename, ChContactMethod_SMC)` (use "
    "`WheeledVehicle(sysMBS, filename)` form); brake never released; "
    "throttle wired but driveline unhooked; FSI coupling not advancing "
    "the body's system."
)


def _render_motion_expectations_block(motion_expectations: List[str]) -> str:
    """Render the rule-6 motion-CSV block conditionally on the step's list.

    Empty list → "no motion CSV required" notice.
    Non-empty   → directive instruction with the body list.
    """
    if not motion_expectations:
        return _MOTION_CSV_BLOCK_NONE
    pretty = ", ".join(f"`{name}`" for name in motion_expectations)
    return _MOTION_CSV_BLOCK_REQUIRED_TEMPLATE.format(names=pretty)


def build_tool_loop_system_prompt(
    *,
    skill_constraints: str,
    mode_constraints: str,
    scope_rule: str,
    available_files_hint: str,
    assets_reminder: str = "",
    spec_file: str,
    utils_reference: str = "",
    recording_mode: str = "sensor_cams",
    motion_expectations: Optional[List[str]] = None,
    geometry_relations: Optional[List[Any]] = None,
    step_objects: Optional[List[Any]] = None,
    scene_predicates: Optional[List[Any]] = None,
) -> str:
    """Build the tool loop system prompt from template and parameters.

    ``step_objects`` and ``scene_predicates`` come from the unified
    ``StepContext`` and drive the new object-placement rule (named-constants
    + relation-derived formulas). Empty / None → that rule is omitted.
    """
    if recording_mode == "vsg_only":
        video_rule = VIDEO_GENERATION_RULE_VSG_ONLY
    else:
        video_rule = VIDEO_GENERATION_RULE
    motion_block = _render_motion_expectations_block(list(motion_expectations or []))
    geometry_relations_rule = _format_geometry_relations_rule(geometry_relations)
    object_placement_rule = _format_object_placement_rule(step_objects, scene_predicates)
    return TOOL_LOOP_SYSTEM_PROMPT_TEMPLATE.substitute(
        skill_constraints=skill_constraints,
        mode_constraints=mode_constraints,
        scope_rule=scope_rule,
        available_files_hint=available_files_hint,
        assets_reminder=assets_reminder,
        api_convention_rule=API_CONVENTION_RULE,
        visual_shape_rule=VISUAL_SHAPE_RULE,
        video_generation_rule=video_rule,
        spec_file=spec_file,
        utils_reference=utils_reference,
        motion_expectations_block=motion_block,
        geometry_relations_rule=geometry_relations_rule,
        object_placement_rule=object_placement_rule,
    )


ERROR_CLASS_ANALYSIS_PROMPT = """Error: {error_info}

Code:
```python
{code_snippet}
```

Known classes: {existing_classes}

Return a JSON object with: error_analysis (brief cause), additional_classes (PyChrono class names like ChFrame, ChAxis), reasoning (why they help). Return ONLY valid JSON, no other text."""
