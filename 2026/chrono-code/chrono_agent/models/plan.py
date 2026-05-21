"""
Pydantic models for simulation planning (Agent 1 output).
"""

import os
from contextvars import ContextVar, Token
from typing import Annotated, List, Literal, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from chrono_agent.models.clarification import (
    ClarificationOption,
    StructuredClarification,
)


def _coerce_dict_entries_to_names(value: Any) -> Any:
    """Coerce ``[{name: X, ...}, ...]`` → ``[X, ...]`` for List[str] fields.

    Weak planner models (gpt-4o-mini and similar OpenAI-compat endpoints)
    frequently confuse ``plan.assets[]`` (List[Dict] of full asset
    descriptors) with ``step.assets[]`` / ``step.scene_objects[]``
    (List[str] of names referencing plan-level entries) and emit the full
    dict in the per-step list. The Pydantic error
    ("Input should be a valid string") then kicks the planner into a
    repair retry, which on weak models drops out of tool-use mode and
    fails again silently — turning a one-line shape mismatch into a
    full workflow abort.

    This validator runs BEFORE Pydantic's str check, extracts ``name``
    (or ``id``) from any dict entries, and leaves non-dict entries
    untouched so the regular validator still catches truly malformed
    input (e.g. ints, None) with a clear error.
    """
    if not isinstance(value, list):
        return value
    out: List[Any] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("id")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
                continue
            # No name → keep the dict so Pydantic raises with the
            # original error location and the LLM can see what it did.
        out.append(item)
    return out


_StringRefList = Annotated[List[str], BeforeValidator(_coerce_dict_entries_to_names)]


# ContextVar that carries the currently-active ``UserSpec`` for a plan
# construction. The PlanningAgent sets it around ``SimulationPlan(**plan_dict)``
# so the ``ensure_time_parameters`` validator can tell whether a missing
# ``time_step`` / ``simulation_duration`` is "user didn't specify, take the
# default" (OK to fill in) versus "user specified it and the LLM dropped it"
# (must raise so the repair loop fires with full context in the system prompt).
_user_spec_var: "ContextVar[Optional[Any]]" = ContextVar(
    "chrono_agent_user_spec", default=None
)


def set_user_spec_context(spec: Optional[Any], token: Optional[Token] = None) -> Token:
    """Set or clear the active ``UserSpec`` for plan construction.

    Returns a ``Token`` that the caller MUST pass back in a later
    ``set_user_spec_context(None, token=token)`` call to restore the prior
    value. Typical usage::

        token = set_user_spec_context(user_spec)
        try:
            plan = SimulationPlan(**plan_dict)
        finally:
            set_user_spec_context(None, token=token)
    """
    if token is not None:
        _user_spec_var.reset(token)
        return token
    return _user_spec_var.set(spec)


def _coerce_to_str(v: Any, default: str) -> str:
    """Coerce to string (LLM may return numbers or null for gravity_axis, working_plane)."""
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        # 0 or 0.0 for "no gravity" -> use default axis
        if v == 0:
            return default
        return str(v)
    return str(v) if v else default


def _asset_name_from_mapping(asset: Dict[str, Any]) -> str:
    """Infer a stable asset name from planner asset metadata."""
    explicit_name = str(asset.get("name") or "").strip()
    if explicit_name:
        return explicit_name

    apply_to = str(asset.get("apply_to") or "").strip()
    if apply_to and apply_to.lower() != "terrain":
        return apply_to

    filename = str(asset.get("filename") or "").strip()
    if filename:
        stem = os.path.splitext(os.path.basename(filename))[0].strip()
        if stem:
            return stem

    description = str(asset.get("description") or "").strip()
    return description or ""


# ---------------------------------------------------------------------------
# Predicate models for scene topology
# ---------------------------------------------------------------------------

class PhysicalPredicate(BaseModel):
    """A physical relationship between two assets."""
    subject: str = Field(description="Asset name (e.g. 'cup')")
    predicate: str = Field(
        description="Physical relation: supports, rests_on, contains, leans_against, attached_to"
    )
    object: str = Field(description="Other asset or 'floor'/'ground'")


class ScenePredicate(BaseModel):
    """A spatial arrangement relationship between two assets, expressed
    in the PhyScensis-style predicate algebra (see
    chrono_agent/skills/planning/scene_coordinate_system/SKILL.md).

    The planner emits BOTH the symbolic predicate (`predicate` + `params`)
    AND the resolved numerical state (`position` + `orientation.deg_z`)
    so codegen never has to re-derive coordinates.
    """

    model_config = {"populate_by_name": True, "extra": "allow"}

    subject: str = Field(description="Asset name")
    predicate: str = Field(
        default="",
        description=(
            "Predicate from the canonical algebra documented in "
            "chrono_agent/skills/planning/scene_coordinate_system. "
            "Free-form names like 'flush_against' / 'bridges_between' / "
            "'centered_in' are rejected by the geometry traceability "
            "validator — use the canonical decomposition "
            "(BACK-OF + ALIGN-CENTER-LR + BOTTOM-AT, etc.). See that "
            "skill for the full vocabulary and worked examples."
        ),
        # Accept legacy `relation` key as an alias so older plans still parse.
        validation_alias="predicate",
    )
    object: str = Field(description="Reference asset or 'root'")
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Predicate parameters (distance, x_offset, y_offset, x, y, ...)",
    )
    position: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resolved world coordinates {x, y, z} for the subject asset (numeric meters)",
    )
    orientation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resolved orientation, e.g. {'deg_z': 90}",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any) -> Any:
        """Accept legacy keys so older plans / repair attempts keep working.

        - `relation` is accepted as an alias for `predicate` (old schema).
        - `orientation` as a free-form string is converted to {"deg_z": <num>}
          when possible, otherwise wrapped as {"text": "..."}.
        - `position` values are coerced to floats when given as strings.
        """
        if not isinstance(data, dict):
            return data

        # relation -> predicate
        if "predicate" not in data and "relation" in data:
            data["predicate"] = data.pop("relation")

        # orientation: str -> dict
        ori = data.get("orientation")
        if isinstance(ori, str):
            txt = ori.strip()
            parsed: Optional[float] = None
            # Try common patterns: "deg_z=90", "90", "180.0"
            try:
                parsed = float(txt)
            except (TypeError, ValueError):
                if "=" in txt:
                    _, _, val = txt.partition("=")
                    try:
                        parsed = float(val.strip())
                    except (TypeError, ValueError):
                        parsed = None
            data["orientation"] = (
                {"deg_z": parsed} if parsed is not None else {"text": txt}
            )

        # position: stringified numbers -> floats
        pos = data.get("position")
        if isinstance(pos, dict):
            coerced: Dict[str, Any] = {}
            for k, v in pos.items():
                if isinstance(v, str):
                    try:
                        coerced[k] = float(v)
                    except (TypeError, ValueError):
                        coerced[k] = v
                else:
                    coerced[k] = v
            data["position"] = coerced

        return data


# ---------------------------------------------------------------------------
# Simulation topology (unified model with optional field groups)
# ---------------------------------------------------------------------------

class SimulationTopology(BaseModel):
    """
    Unified spatial and kinematic description of the simulation.

    Contains shared fields (coordinate system, camera) plus two optional
    field groups:
    - **Scene fields** (physical_predicates, scene_predicates) for scene /
      mbs_in_scene plans.
    - **MBS fields** (body_positions, body_geometry, joints) for mbs /
      mbs_in_scene plans.
    """

    # Coordinate system
    gravity_axis: str = Field(
        default="-z",
        description='Gravity direction, e.g. "-z" (PyChrono default), "-y", "+x".',
    )
    working_plane: str = Field(
        default="xz",
        description='Plane of the mechanism: "xy" | "xz" | "yz". Determines default camera direction.',
    )
    reference_heights: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            'Named z-level anchors used by height predicates (TOP-AT, '
            'BOTTOM-AT, FLOATS-AT-SURFACE, ...). Each entry: '
            '{"name": "water_surface", "z": 1.0}. Declaring common layers '
            'once (e.g. tank_floor, water_surface, walkway_level, ceiling) '
            'lets downstream height predicates reference names instead of '
            'repeating literals — and codifies intent like '
            '"walkway_level == water_surface" so codegen cannot drift. '
            'Leave None for scenes without layered vertical structure.'
        ),
    )

    @field_validator("gravity_axis", "working_plane", mode="before")
    @classmethod
    def coerce_topology_str_fields(cls, v: Any, info) -> str:
        """Coerce to string (LLM may return numbers, e.g. 0 for no gravity)."""
        default = "-z" if info.field_name == "gravity_axis" else "xz"
        return _coerce_to_str(v, default)

    # ---- Scene fields (scene / mbs_in_scene) ----

    orientation_convention: str = Field(
        default="y_up_to_z_up",
        description=(
            "Asset orientation convention. 'y_up_to_z_up' means assets are modelled "
            "with native up = +Y but the scene up = +Z (gravity_axis: -z), so code "
            "must rotate every asset from Y-up to Z-up (e.g. 90° around X axis)."
        ),
    )

    physical_predicates: Optional[List[PhysicalPredicate]] = Field(
        default=None,
        description="Physical relationships between assets (support, containment, attachment)",
    )

    scene_predicates: Optional[List[ScenePredicate]] = Field(
        default=None,
        description="Spatial arrangement relationships that define the 3D layout",
    )

    # ---- MBS fields (mbs / mbs_in_scene) ----

    body_positions: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description=(
            "Body name -> center-of-mass {x, y, z} formulas using simulation_parameters symbols. "
            "E.g. {'crank': {'x': 'L1/2', 'y': '0', 'z': '0'}}"
        ),
    )

    body_geometry: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description=(
            "Per-body visual/physical geometry: body_name -> {'length': float, 'radius': float, ...}. "
            "Use the body's link length as cylinder height in AddVisualShape."
        ),
    )

    joints: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of joints: [{'body1': '...', 'body2': '...', "
            "'type': 'revolute'|'prismatic', 'location': '...'}, ...]. "
            "Include all ground constraints."
        ),
    )

    # ---- Scene bounds (used to scale sensor cameras and floor extent) ----

    scene_size: Optional[List[float]] = Field(
        default=None,
        description=(
            "Overall scene bounding box as [extent_x, extent_y, extent_z] in "
            "meters — the smallest axis-aligned box that contains every body "
            "(walkway / platform / vehicle / camera buffer included), NOT "
            "just the central asset. Codegen uses it to scale sensor-camera "
            "eye distances and (for indoor scenes) wall extents. "
            "Set ONLY when the scene has a definable bound: indoor rooms, "
            "FSI tanks with surrounding walkways, fenced arenas. "
            "OMIT (leave None) when the scene is unbounded or has no "
            "meaningful spatial envelope: outdoor SCM terrain demos that "
            "the vehicle drives across freely, pure MBS mechanisms (pendulum, "
            "gear train), free-fall / projectile demos, infinite plane "
            "scenarios. For those cases the cameras derive their distance "
            "from the protagonist body's bbox at runtime."
        ),
    )

    # ---- Sensor cameras ----
    # Static scenes: 5 cameras (4 cardinal + top-down), world-frame.
    # Mobile scenes (robot/vehicle): 3 body-attached cameras (onboard + side + top-down).

    cameras: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description=(
            "Sensor camera EYE positions for VLM review (static_5 layout). "
            "Up to five entries keyed by name: 'camera+x', 'camera-x', "
            "'camera+y', 'camera-y', 'top_down'. Each value is [x,y,z]. "
            "All cameras look at `camera_target` with shared `camera_up`. "
            "For mobile_3 layout, this field can be omitted — cameras are "
            "derived from the chassis body at runtime."
        ),
    )
    camera_target: Optional[List[float]] = Field(
        default=None,
        description=(
            "Shared look-at target [x, y, z] for world-frame sensor cameras — "
            "typically the scene center [0, 0, 0]. Used as initial target and as "
            "fallback when `camera_target_body` is not set."
        ),
    )
    camera_target_body: Optional[str] = Field(
        default=None,
        description=(
            "Optional body name (e.g. 'curiosity_rover') that world-frame sensor "
            "cameras should follow during simulation. When set, codegen updates the "
            "camera look-at to that body's current world position every frame; "
            "otherwise the static `camera_target` is used."
        ),
    )
    camera_up: Optional[List[float]] = Field(
        default=None,
        description=(
            "Shared up vector [x, y, z] for sensor cameras — must be "
            "anti-parallel to gravity_axis. gravity_axis='-z' → [0,0,1]; "
            "gravity_axis='-y' → [0,1,0]."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_all_fields(cls, v: Any) -> Any:
        """Coerce all topology fields from potentially malformed LLM output."""
        if not isinstance(v, dict):
            return v
        # Clean body_positions: skip non-dict values, and coerce inner x/y/z to strings
        if "body_positions" in v:
            bp = v["body_positions"]
            if isinstance(bp, dict):
                cleaned = {}
                for k, vals in bp.items():
                    if isinstance(vals, dict):
                        cleaned[k] = {k2: str(v2) for k2, v2 in vals.items()}
                v["body_positions"] = cleaned if cleaned else None
            else:
                v["body_positions"] = None
        # Clean body_geometry: skip non-dict values
        if "body_geometry" in v:
            bg = v["body_geometry"]
            if isinstance(bg, dict):
                v["body_geometry"] = {
                    k: geom for k, geom in bg.items()
                    if isinstance(geom, dict)
                }
            else:
                v["body_geometry"] = None
        # Clean joints: skip non-dict items
        if "joints" in v:
            j = v["joints"]
            if isinstance(j, list):
                v["joints"] = [item for item in j if isinstance(item, dict)]
            else:
                v["joints"] = None
        # Clean physical_predicates: skip non-dict items
        if "physical_predicates" in v:
            pp = v["physical_predicates"]
            if isinstance(pp, list):
                v["physical_predicates"] = [item for item in pp if isinstance(item, dict)] or None
            else:
                v["physical_predicates"] = None
        # Clean scene_predicates: skip non-dict items
        if "scene_predicates" in v:
            sp = v["scene_predicates"]
            if isinstance(sp, list):
                v["scene_predicates"] = [item for item in sp if isinstance(item, dict)] or None
            else:
                v["scene_predicates"] = None
        # Backwards-compat: a legacy `camera_eye` collapses into the camera+x slot
        # of the new `cameras` dict. Legacy `camera_target` is kept as the shared
        # target since all 4 sensor cameras now look at the same point.
        legacy_eye = v.pop("camera_eye", None)
        if legacy_eye is not None and not v.get("cameras"):
            if isinstance(legacy_eye, list):
                v["cameras"] = {"camera+x": list(legacy_eye)}
        # Backwards-compat: legacy `room_size` → `scene_size`. Old plans (and
        # the LLM occasionally) still emit `room_size`; alias-coerce so we
        # accept either key without losing data.
        if "room_size" in v and "scene_size" not in v:
            v["scene_size"] = v.pop("room_size")
        elif "room_size" in v:
            # Both present — prefer the new name and drop the legacy one.
            v.pop("room_size", None)
        # Clean scene_size: must be a 3-element numeric list, else drop.
        # Sentinel strings ("not_applicable" / "n/a") collapse to None so
        # the planner can be explicit about scenes without a meaningful
        # bound.
        if "scene_size" in v:
            ss = v["scene_size"]
            if isinstance(ss, str) and ss.strip().lower() in {
                "not_applicable", "not applicable", "n/a", "na", "none", "unbounded",
            }:
                v["scene_size"] = None
            elif isinstance(ss, (list, tuple)) and len(ss) == 3:
                try:
                    v["scene_size"] = [float(x) for x in ss]
                except (TypeError, ValueError):
                    v["scene_size"] = None
            else:
                v["scene_size"] = None
        # Normalize camera_target_body: empty / non-string → None.
        if "camera_target_body" in v:
            ctb = v["camera_target_body"]
            if isinstance(ctb, str) and ctb.strip():
                v["camera_target_body"] = ctb.strip()
            else:
                v["camera_target_body"] = None
        # Clean cameras: keep only known camera keys with [x,y,z] lists.
        if "cameras" in v:
            cams = v["cameras"]
            if isinstance(cams, dict):
                allowed = {"camera+x", "camera-x", "camera+y", "camera-y", "top_down"}
                cleaned_cams: Dict[str, List[float]] = {}
                for name, entry in cams.items():
                    if name not in allowed:
                        continue
                    # Accept either bare [x,y,z] or {"eye": [x,y,z]} legacy form.
                    if isinstance(entry, list):
                        cleaned_cams[name] = list(entry)
                    elif isinstance(entry, dict) and isinstance(entry.get("eye"), list):
                        cleaned_cams[name] = list(entry["eye"])
                v["cameras"] = cleaned_cams or None
            else:
                v["cameras"] = None
        return v


# ---------------------------------------------------------------------------
# CameraPose + SimulationStep: structured step with per-step camera
# ---------------------------------------------------------------------------


class CameraPose(BaseModel):
    """Explicit camera pose for a single simulation step.

    Each ``SimulationStep`` owns one ``CameraPose``; the code-generation
    agent reads it and passes ``position`` / ``target`` / ``up`` to
    :func:`chrono_agent.utils.setup_preview_camera` so the VLM review gets
    a purposeful viewpoint for this phase of the build (e.g. looking at the
    terrain for the terrain-placement step, chasing the vehicle for a
    driving step).
    """

    position: List[float] = Field(
        description=(
            "Eye position [x, y, z] in world coordinates (meters). For "
            "chase-style views, the code agent may reinterpret this as "
            "body-relative offset — the plan-layer value is the authoritative "
            "intent."
        ),
    )
    target: List[float] = Field(
        description="Look-at target [x, y, z] in world coordinates (meters).",
    )
    up: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0],
        description="Up vector [x, y, z]. Default [0, 0, 1] for z-up worlds.",
    )

    @field_validator("position", "target", "up")
    @classmethod
    def _validate_length_3(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError(
                f"CameraPose vector must have 3 components [x, y, z]; got {len(v)}"
            )
        return [float(x) for x in v]


class SimulationStep(BaseModel):
    """One coarse-grained build phase for scene / mbs_in_scene plans.

    Each step is realized end-to-end by one codegen + execution + step_review
    pass. Carries the natural-language directive, the assets introduced this
    phase, the camera pose the VLM will review against, and any hard
    constraints the code agent must honor.

    Pure ``mbs`` plans leave ``implementation_steps`` empty and are built
    monolithically in a single codegen call (no step_loop).
    """

    description: str = Field(
        description="Natural-language directive the code agent will realize this step",
    )
    assets: _StringRefList = Field(
        default_factory=list,
        description=(
            "Catalog asset names introduced by this step. Procedural objects "
            "belong in scene_objects. The first step must introduce at least "
            "one visible asset or scene_object."
        ),
    )
    scene_objects: _StringRefList = Field(
        default_factory=list,
        description=(
            "Procedural scene object names introduced by this step, e.g. "
            "boxes, plates, platforms, tanks, walls, generated boundaries, "
            "or fluid domains described in SimulationPlan.scene_objects."
        ),
    )
    objects: _StringRefList = Field(
        default_factory=list,
        description=(
            "Names of plan.objects[] introduced by this step. Unified "
            "replacement for ``assets`` + ``scene_objects``; when populated, "
            "build_step_context derives step_assets / step_scene_objects "
            "from this list against plan.objects[]."
        ),
    )
    cameras: List[CameraPose] = Field(
        min_length=1,
        description=(
            "Viewing poses for this step. Each entry becomes one "
            "`setup_preview_camera(...)` call in the generated simulation. "
            "Use 2–3 angles that cover complementary VIEWING DIRECTIONS "
            "(e.g. wide-from-NE + wide-from-NW + top-down) so the VLM "
            "reviews the full environment, not a single slice. Not "
            "zoom levels of the same angle — different directions."
        ),
    )
    constraints: List[str] = Field(
        default_factory=list,
        description=(
            "'Do not violate' hard constraints surfaced verbatim in the "
            "codegen system prompt. E.g. 'no asset within 10m of origin', "
            "'HMMWV spawn clearance ≥ 5m'."
        ),
    )
    motion_expectations: List[str] = Field(
        default_factory=list,
        description=(
            "Names of bodies (plan_asset or scene_object) the step "
            "description says should move. Empty list = no motion expected "
            "this step (static / settle / build-only step). When non-empty, "
            "codegen MUST dump a cam/motion_log.csv with time / pos / vel "
            "rows for each listed body, and the physics review pass will "
            "read that CSV. Listed names should resolve to assets with "
            "is_dynamic=True or scene_objects with fixed=False."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_friendly_shapes(cls, v: Any) -> Any:
        """Tolerate single-string / singular-camera LLM shapes.

        LLMs often emit a single string where a list is expected, or the
        older ``camera: {...}`` singular shape after the schema migrated
        to ``cameras: [...]``. Repair loops catch it but every retry is
        waste; coerce here so the planner + historical plans load cleanly.
        """
        if not isinstance(v, dict):
            return v
        for key in ("assets", "scene_objects", "objects", "constraints", "motion_expectations"):
            val = v.get(key)
            if isinstance(val, str):
                v[key] = [val]
        # Back-compat: accept singular `camera: {...}` OR `cameras: {...}`
        # (bare dict) and promote to a 1-element list under `cameras`.
        if "cameras" not in v and isinstance(v.get("camera"), dict):
            v["cameras"] = [v.pop("camera")]
        elif isinstance(v.get("cameras"), dict):
            v["cameras"] = [v["cameras"]]
        return v

    @property
    def camera(self) -> CameraPose:
        """Backward-compatible alias for the first camera pose."""
        return self.cameras[0]


# ---------------------------------------------------------------------------
# Step context: per-step context bundle for code generation agent
# ---------------------------------------------------------------------------

class StepContext(BaseModel):
    """Per-step context bundle delivered to the code generation agent.

    Contains only the information relevant to the current step, so the LLM
    is not overwhelmed by the full plan.
    """
    step_index: int = Field(description="0-based index of the current step")
    total_steps: int = Field(description="Total number of steps in the plan")
    step_description: str = Field(description="Description of the current step")

    step_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Full asset records (name, type, filename, etc.) for every asset "
            "this step introduces. Empty for pure-physics steps."
        ),
    )
    step_scene_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Full procedural scene object records for every non-catalog object "
            "this step introduces."
        ),
    )
    step_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Unified Object records (new schema) for every object this step "
            "introduces. Each entry has name / construction / topology / pose "
            "/ fixed / is_dynamic / fsi_registration. Codegen reads these to "
            "emit named constants and relation-derived position formulas."
        ),
    )
    step_cameras: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of camera poses for this step, each with "
            "{position: [x,y,z], target: [x,y,z], up: [x,y,z]}. Codegen "
            "emits one setup_preview_camera call per entry. Guaranteed "
            "non-empty by SimulationStep.cameras' min_length=1 validator."
        ),
    )
    step_constraints: List[str] = Field(
        default_factory=list,
        description="Hard constraints this step must honor",
    )
    step_motion_expectations: List[str] = Field(
        default_factory=list,
        description=(
            "Body names the step expects to move. When non-empty, codegen "
            "must dump cam/motion_log.csv for these bodies; the physics "
            "review pass reads that CSV."
        ),
    )
    prior_constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Constraints from already-completed steps, "
            "{step_index, constraints: [...]}. Codegen keeps these active."
        ),
    )

    current_asset: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Quick reference to the FIRST asset of this step (when present). "
            "Full list is in ``step_assets``."
        ),
    )

    scene_predicates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scene predicates involving only current + completed assets",
    )
    physical_predicates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Physical predicates involving only current + completed assets",
    )

    simulation_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global simulation parameters (always needed)",
    )
    topology_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Topology metadata: gravity_axis, working_plane, orientation_convention, camera",
    )

    completed_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Already-placed assets (name + filename only)",
    )
    completed_scene_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Already-created procedural scene objects (name + construction source)",
    )
    completed_steps: List[str] = Field(
        default_factory=list,
        description="Descriptions of completed steps",
    )


class CameraConfig(BaseModel):
    """Sensor camera configuration for video generation for review."""
    enabled: bool = Field(
        default=True,
        description="Whether to enable sensor cameras for video generation"
    )
    layout: Optional[str] = Field(
        default=None,
        description=(
            "Camera layout policy. 'static_5': 4 cardinal + top-down world-frame cameras "
            "for static-only scenes (uses setup_preview_camera). 'mobile_3': 3 body-attached "
            "cameras (onboard + side + top-down) for scenes with a moving robot/vehicle "
            "(uses raw ChCameraSensor on chassis). null = auto-detect from plan context."
        ),
    )
    num_cameras: Optional[int] = Field(
        default=None,
        description=(
            "Number of camera angles. Auto-determined from layout when null: "
            "static_5 → 5, mobile_3 → 3. Override only if the user requests a specific count."
        ),
    )
    image_width: int = Field(
        default=1280,
        description="Image width in pixels"
    )
    image_height: int = Field(
        default=720,
        description="Image height in pixels"
    )
    fov: float = Field(
        default=1.408,
        description="Horizontal field of view in radians (~80 degrees)"
    )
    update_rate: Optional[int] = Field(
        default=None,
        description="Camera update rate in Hz for review/video export cameras; derive this from the simulation step size via update_rate = 1 / dt"
    )


class GeometryRelation(BaseModel):
    """One spatial relation between two named bodies (or body+scene/camera).

    Each entry references a pattern documented in the
    ``geometry/scene_relations`` skill. Codegen reads that skill at
    generation time and follows the canonical worked example to emit the
    correct ``SetPos`` / camera pose. ``relation_name == "TO_CLARIFY"`` is
    a sentinel meaning the planner could not resolve which pattern applies
    and a matching ``clarifications_needed`` entry must be present.
    """

    relation_name: str = Field(
        description=(
            "Either 'TO_CLARIFY' or the exact heading of a subsection in "
            "chrono_agent/skills/geometry/scene_relations/SKILL.md "
            "(e.g. 'platform_flush_wall_outer'). Codegen refuses to emit "
            "coordinates for 'TO_CLARIFY' relations."
        )
    )
    body_a: str = Field(description="First body name; from assets[] or scene_objects[].")
    body_b: str = Field(
        description=(
            "Second endpoint. Usually a body name; the literal strings "
            "'scene' and 'camera' are accepted for camera-framing relations."
        )
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pattern-specific kwargs (e.g. {wall: '-x'}, {axis_side: '-y'}).",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_geometry_relation_fields(cls, v: Any) -> Any:
        """Heal common LLM shape slips so the planner repair loop converges.

        Observed first-draft mistakes:
          * ``body_b: null`` — model omitted the second endpoint entirely
            (most common when the relation is between a body and the scene
            or camera). Coerce to the empty string so Pydantic surfaces a
            single clear error the LLM can fix in one repair round.
          * ``body_b: {"name": "left_platform"}`` — model emitted the full
            body record instead of just its name. Same dict-or-list shape
            as ``_coerce_dict_entries_to_names``; pull out ``name``/``id``.
          * ``body_b: ["left_platform"]`` — model wrapped the name in a
            list. Take the first element.
          * ``parameters: "wall=-x"`` — model wrote a string instead of a
            dict. Drop it back to ``{}`` and let the LLM re-emit; the
            relation is still routable from ``relation_name`` alone.

        Without this coercion the validation error is a deeply-nested
        ``geometry_relations.3.body_b: Input should be a valid string``
        which the repair loop tends to thrash on (LLM rewrites the whole
        block instead of fixing one field).
        """
        if not isinstance(v, dict):
            return v
        out = dict(v)
        for key in ("relation_name", "body_a", "body_b"):
            val = out.get(key)
            if val is None:
                out[key] = ""
                continue
            if isinstance(val, dict):
                name = val.get("name") or val.get("id")
                out[key] = str(name).strip() if name else ""
                continue
            if isinstance(val, list):
                first = next((str(x).strip() for x in val if x is not None), "")
                out[key] = first
                continue
            if not isinstance(val, str):
                out[key] = str(val).strip()
        params = out.get("parameters")
        if params is None:
            out["parameters"] = {}
        elif not isinstance(params, dict):
            out["parameters"] = {}
        return out


# ---------------------------------------------------------------------------
# Unified Object schema (new pipeline, see plan_agent.md §4)
# ---------------------------------------------------------------------------
#
# Each Object record collapses the legacy ``assets[]`` + ``scene_objects[]`` +
# ``topology.scene_predicates[]`` into ONE entry per body. Four required
# blocks per object:
#   1. ``name``                 — unique snake_case id
#   2. ``construction``         — asset (catalog-backed) or procedural (chrono body)
#   3. ``topology``             — base (absolute pose) or child (ref + relation)
#   4. ``pose``                 — final position + rotation_deg (numeric after Phase 5)
#
# This is added ALONGSIDE the legacy fields. Workflows opt into the new
# schema by populating ``SimulationPlan.objects[]``; legacy code paths
# continue reading ``assets[]`` / ``scene_objects[]`` / ``topology``.


class ObjectConstruction(BaseModel):
    """How a single object is built — catalog asset or procedural chrono body."""

    model_config = {"extra": "allow"}

    kind: Literal["asset", "procedural"] = Field(
        description=(
            "'asset' = catalog-backed (mesh / urdf / wrapper_vehicle / "
            "vehicle_json). 'procedural' = built from chrono primitives "
            "(box / sphere / cylinder / grid / fluid_domain / generated_boundary)."
        )
    )
    # asset-only fields
    catalog: Optional[str] = Field(
        default=None,
        description="Catalog row name (e.g. 'Polaris', 'hmmwv'). Required when kind='asset'.",
    )
    asset_type: Optional[
        Literal["mesh", "urdf", "wrapper_vehicle", "vehicle_json"]
    ] = Field(
        default=None,
        description="Asset loader type (copied from catalog). Required when kind='asset'.",
    )
    filename: Optional[str] = Field(
        default=None,
        description=(
            "File path (mesh / urdf / vehicle_json). Copy verbatim from "
            "catalog. None for kind='procedural' or kind='asset' with "
            "asset_type='wrapper_vehicle'."
        ),
    )
    factory: Optional[str] = Field(
        default=None,
        description=(
            "Python factory expression (e.g. 'veh.HMMWV_Full()') for "
            "asset_type='wrapper_vehicle'. None otherwise."
        ),
    )
    # procedural-only fields
    primitive: Optional[
        Literal[
            "box", "sphere", "cylinder", "grid",
            "fluid_domain", "generated_boundary",
        ]
    ] = Field(
        default=None,
        description="Primitive type. Required when kind='procedural'.",
    )
    size: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Procedural size {x, y, z} in meters. Required for "
            "kind='procedural'. Diameter for sphere; (radius, length) "
            "encoded as {x: radius, z: length} for cylinder."
        ),
    )
    density: Optional[float] = Field(
        default=None,
        description="Density in kg/m^3 for dynamic procedural bodies (e.g. floating plate).",
    )
    mass: Optional[float] = Field(
        default=None,
        description="Optional explicit mass in kg (overrides density-derived mass).",
    )
    viscosity: Optional[float] = Field(
        default=None,
        description="Dynamic viscosity (Pa·s) for fluid_domain primitives.",
    )

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> "ObjectConstruction":
        if self.kind == "asset":
            if not self.catalog:
                raise ValueError("construction.kind='asset' requires 'catalog'.")
            if not self.asset_type:
                raise ValueError("construction.kind='asset' requires 'asset_type'.")
            if self.asset_type == "wrapper_vehicle":
                if not self.factory:
                    raise ValueError(
                        "asset_type='wrapper_vehicle' requires 'factory' "
                        "(e.g. 'veh.HMMWV_Full()')."
                    )
            else:
                if not self.filename:
                    raise ValueError(
                        f"asset_type={self.asset_type!r} requires 'filename'."
                    )
        else:  # procedural
            if not self.primitive:
                raise ValueError(
                    "construction.kind='procedural' requires 'primitive'."
                )
            # size is loosely required (allow None for grid which is
            # purely visual axes), but the parser will warn.
        return self


class ObjectTopology(BaseModel):
    """An object's spatial relationship to the rest of the scene.

    ``role='base'`` ⇒ this object's ``pose`` is absolute world coordinates.
    ``role='child'`` ⇒ pose is derived from ``ref`` + ``relation`` via the
    scene_coordinate_system skill (Phase 5 resolves the relation pattern
    into numeric coordinates).
    """

    model_config = {"extra": "allow"}

    role: Literal["base", "child"] = Field(
        description=(
            "'base' = root anchor with absolute pose; 'child' = positioned "
            "relative to ``ref`` via the named ``relation`` pattern."
        )
    )
    ref: Optional[str] = Field(
        default=None,
        description=(
            "Name of the reference object (must be another object's "
            "``name`` in the same plan). Required when role='child'."
        ),
    )
    relation: Optional[str] = Field(
        default=None,
        description=(
            "Relation pattern from chrono_agent/skills/planning/"
            "scene_coordinate_system/SKILL.md (e.g. "
            "'bottom_flush_water_surface', 'spawned_on_top'). Required "
            "when role='child'. Phase 5 reads this to resolve the "
            "object's ``pose.position`` from the ref's pose."
        ),
    )

    @model_validator(mode="after")
    def _check_role_consistency(self) -> "ObjectTopology":
        if self.role == "child":
            if not self.ref:
                raise ValueError("topology.role='child' requires 'ref'.")
            if not self.relation:
                raise ValueError("topology.role='child' requires 'relation'.")
        else:  # base
            if self.ref or self.relation:
                # tolerate but warn — a base shouldn't carry ref/relation.
                # We don't raise because the planner sometimes writes
                # ref='world' + relation='base' for clarity.
                pass
        return self


class ObjectPose(BaseModel):
    """6-DoF pose for an object.

    For ``topology.role: base``: ``position`` is required (absolute coords).
    For ``topology.role: child``: ``position`` is omitted — codegen computes
    it from the relation pattern using ref's pose+size and obj's size.
    """

    model_config = {"extra": "allow"}

    position: Optional[Dict[str, float]] = Field(
        default=None,
        description="World-frame position {x, y, z} in meters. Required for base, omitted for child.",
    )
    rotation_deg: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        description="Euler rotation in degrees {x, y, z} (intrinsic XYZ).",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_pose_inputs(cls, v: Any) -> Any:
        """Tolerate list-shape inputs and stringified numbers."""
        if not isinstance(v, dict):
            return v
        for key in ("position", "rotation_deg"):
            val = v.get(key)
            if val is None:
                continue
            if isinstance(val, (list, tuple)) and len(val) == 3:
                try:
                    v[key] = {
                        "x": float(val[0]),
                        "y": float(val[1]),
                        "z": float(val[2]),
                    }
                except (TypeError, ValueError):
                    pass
            elif isinstance(val, dict):
                coerced: Dict[str, float] = {}
                for k, num in val.items():
                    try:
                        coerced[k] = float(num)
                    except (TypeError, ValueError):
                        coerced[k] = 0.0
                v[key] = coerced
        return v


class Object(BaseModel):
    """One physical entity in the simulation — unified asset / procedural body.

    Replaces the legacy split across ``assets[]`` / ``scene_objects[]`` /
    ``topology.scene_predicates[]``. See ``plan_agent.md`` §4 for the full
    schema rationale.
    """

    model_config = {"extra": "allow"}

    name: str = Field(description="Unique snake_case identifier (e.g. 'left_platform').")
    construction: ObjectConstruction = Field(
        description="How this object is built (asset vs procedural)."
    )
    topology: ObjectTopology = Field(
        description="Spatial role: base (absolute) or child (ref + relation)."
    )
    pose: ObjectPose = Field(
        default_factory=ObjectPose,
        description="Final position + rotation_deg (numeric after Phase 5).",
    )
    fixed: bool = Field(
        default=True,
        description=(
            "Whether the body is anchored. Procedural bodies must set this "
            "explicitly. Asset bodies default to True unless the catalog "
            "marks them dynamic, except indoor repo-local data/scene "
            "furniture/props in robot/vehicle scenes, which should be "
            "fixed=False so the robot can physically interact with them. "
            "This indoor exception does not apply to outdoor/offroad assets."
        ),
    )
    is_dynamic: bool = Field(
        default=False,
        description=(
            "Whether the body moves under simulation forces. For catalog "
            "assets, copy from the catalog row's is_dynamic flag when "
            "present. If absent, indoor repo-local data/scene furniture/props "
            "in robot/vehicle scenes should be dynamic; structural bodies "
            "and outdoor/offroad assets should remain static unless the user "
            "explicitly asks for a pushable/movable prop."
        ),
    )
    fsi_registration: Optional[str] = Field(
        default=None,
        description=(
            "Optional FSI-coupling hint, e.g. 'CreatePointsBoxInterior' "
            "(floating body) or 'CreatePointsBoxContainer' (tank wall). "
            "Only set for fsi_in_scene plans."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="One-line note about the body's role / physical params.",
    )

    @model_validator(mode="after")
    def _check_pose_role_consistency(self) -> "Object":
        """Base must have absolute pose.position; child must NOT (codegen computes)."""
        if self.topology.role == "base":
            if self.pose.position is None:
                raise ValueError(
                    f"Object {self.name!r} has topology.role='base' but no "
                    "pose.position. Base objects must give absolute world coords."
                )
        else:  # child
            # If the planner accidentally wrote a position for a child, drop it.
            # Codegen will compute the actual pose from ref + relation.
            self.pose.position = None
        return self


class SimulationPlan(BaseModel):
    """
    Represents the plan created by the Planning Agent (Agent 1).

    Attributes:
        plan_type: Category of simulation (scene, mbs, mbs_in_scene, or fsi_in_scene)
        simulation_parameters: All hyperparameters and system parameters with specific values
        objectives: High-level goals of the simulation
        implementation_steps: Ordered list of implementation steps; for asset-bearing plans
            these should follow the asset placement order from scene_predicates
        clarifications_needed: Questions that need user input
    """

    plan_type: Literal["scene", "mbs", "mbs_in_scene", "fsi_in_scene"] = Field(
        description=(
            "Category of simulation: "
            "'scene' (asset placement / layout only), "
            "'mbs' (multi-body system mechanics only), "
            "'mbs_in_scene' (rigid-body MBS within a scene; no fluid coupling), "
            "'fsi_in_scene' (SPH fluid + MBS hybrid: water tanks, dam-break, "
            "vehicle-on-floating-body, etc.)"
        )
    )

    simulation_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="All simulation hyperparameters with specific values (masses, dimensions, time_step, etc.)"
    )

    objectives: List[str] = Field(
        default_factory=list,
        exclude=True,
        description="High-level simulation objectives"
    )

    implementation_steps: List[SimulationStep] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Ordered list of structured build phases for scene / mbs_in_scene "
            "plans. Each step carries a description, the assets it introduces, "
            "a camera pose for VLM review, and optional hard constraints. "
            "Pure `mbs` plans leave this empty (the workflow skips step_loop "
            "and builds the sim monolithically)."
        ),
    )

    clarifications_needed: List[Union[str, StructuredClarification]] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Questions requiring user clarification. Each entry is either "
            "a free-text question (legacy plain string, accepted only for "
            "non-relational asks like duration) or a StructuredClarification "
            "with 2-3 ClarificationOptions. Geometry-relation clarifications "
            "MUST use the structured form so the UI can render labelled "
            "options + 'Other (text input)' per GEOMETRY_RELATION_RULES."
        ),
    )

    geometry_relations: List[GeometryRelation] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Named spatial relations between adjacent / attached / related "
            "body pairs. Each entry references a pattern in the "
            "geometry/scene_relations skill. Unresolved relations carry "
            "relation_name='TO_CLARIFY' and MUST have a matching "
            "clarifications_needed entry per GEOMETRY_RELATION_RULES. "
            "Codegen reads the referenced pattern and follows its canonical "
            "worked example to emit SetPos / camera pose code."
        ),
    )

    visualization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "mode": "headless",
            "apis": [],
        },
        exclude=True,
        description="Visualization setup.",
    )

    recording_mode: Literal["vsg_only", "sensor_cams"] = Field(
        default="sensor_cams",
        description=(
            "Which video pipeline records the primary mp4. "
            "'sensor_cams' (default): ChSensorManager + ChCameraSensor "
            "(OptiX) — fast, multi-angle, but CANNOT render SPH particles. "
            "'vsg_only': VSG window with chrono_agent.utils.vsg_recording "
            "(setup_vsg_recording / lock_side_camera / hide_vsg_gui) — required "
            "for FSI / SPH scenes because the SPH particle plugin is VSG-only "
            "and never reaches the OptiX scene tree. "
            "Source of truth is the planner: this field is taken verbatim "
            "from the plan markdown — no validator override. The PlanningAgent "
            "prompt teaches the FSI → vsg_only rule; if the planner picks "
            "wrong, the wrong mode propagates to codegen and the resulting "
            "mp4 will visibly demonstrate the mistake (empty tank for FSI)."
        ),
    )

    topology: Optional[SimulationTopology] = Field(
        default=None,
        exclude=True,
        description=(
            "Unified spatial+kinematic description: coordinate system, body positions, "
            "geometry, joints, scene predicates, and camera configuration."
        ),
    )

    camera: Optional[CameraConfig] = Field(
        default_factory=lambda: CameraConfig(enabled=True),
        description="Sensor camera configuration for video generation for review. "
            "When enabled, the generated code creates a ChSensorManager and calls "
            "chrono_agent.utils.setup_preview_camera(...) for every view to produce MP4 video.",
    )

    assets: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        exclude=True,
        description=(
            "Unified list of loadable external entities in the simulation. "
            "Catalog mesh, URDF robot, wrapper vehicle, and JSON-driven "
            "vehicle entries live here. Procedural primitive bodies, generated "
            "boundaries, and fluid domains live in scene_objects instead. "
            "Each asset dict may contain: "
            "{'name': 'catalog_name', "
            "'type': 'mesh'|'urdf'|'vehicle_json'|'wrapper_vehicle'|'texture'|'heightmap', "
            "'filename': 'path/to/file' (for file-backed types), "
            "'factory': 'veh.HMMWV_Full()' (for type=wrapper_vehicle; "
            "Python expression codegen evaluates to instantiate), "
            "'is_dynamic': bool (True = moves during simulation; "
            "downstream placement check skips it), "
            "'description': 'what this asset represents', "
            "'apply_to': 'body_name' | 'terrain', "
            "'ideal_height': float (target height in meters for uniform scaling), "
            "'fixed': bool (True = immovable/anchored, False = affected by forces; "
            "default True for terrain/ground/walls/platforms/containers and "
            "for outdoor/offroad props; default False only for indoor "
            "repo-local data/scene furniture/props in robot/vehicle scenes)}"
        ),
    )

    scene_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Non-catalog objects constructed procedurally by codegen. Each dict "
            "should include at minimum name, role, construction_source "
            "('procedural_primitive'|'fluid_domain'|'generated_boundary'), "
            "primitive or domain_type when applicable, size/dimensions, "
            "position_rule, fixed/dynamic, and key physics parameters such as "
            "density, mass, material, or FSI registration method."
        ),
    )

    plan_markdown: Optional[str] = Field(
        default=None,
        description="Canonical Markdown plan text produced by the planning agent",
    )

    # ---- New unified objects schema (see plan_agent.md §4) ----
    # Populated by the new 6-phase pipeline. When non-empty, downstream
    # consumers MAY read this in preference to the legacy ``assets[]`` /
    # ``scene_objects[]`` split. The legacy fields remain for back-compat
    # with stored plans and any code path not yet migrated.
    objects: List[Object] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Unified per-body records (new pipeline). Each entry collapses "
            "the legacy assets[] + scene_objects[] + topology.scene_predicates[] "
            "into one record with construction (asset|procedural), topology "
            "(base|child+ref+relation), pose (position+rotation_deg). "
            "Empty when the legacy split is in use."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_all_fields(cls, v: Any) -> Any:
        """Coerce all plan fields from potentially malformed LLM output."""
        if not isinstance(v, dict):
            return v

        # Re-hydrate from markdown/raw text when structured fields are absent
        # (i.e., deserializing from serialized state where excluded fields are missing)
        md = v.get("plan_markdown")
        if md and not v.get("objectives") and not v.get("implementation_steps"):
            from chrono_agent.models.plan_parser import (
                parse_markdown_to_fields,
                extract_plan_fields_from_text,
            )
            parsed = parse_markdown_to_fields(md, fallback_plan_type=v.get("plan_type", "mbs"))
            for key, value in parsed.items():
                if key not in v or v[key] is None or v[key] == [] or v[key] == {}:
                    v[key] = value
            # If markdown heading-based parsing still missed fields (e.g. the
            # stored plan_markdown is actually JSON), try per-field extraction.
            if not v.get("implementation_steps") or (
                not v.get("assets") and not v.get("scene_objects")
            ):
                extracted = extract_plan_fields_from_text(md)
                if extracted:
                    for key, value in extracted.items():
                        if key not in v or v[key] is None or v[key] == [] or v[key] == {}:
                            v[key] = value

        # Backward compatibility: map old plan_type values to new categories
        pt = v.get("plan_type", "")
        if pt in ("simple", "detailed"):
            v["plan_type"] = "mbs"

        # Drop removed fields silently (backward compat with old serialized plans)
        v.pop("estimated_complexity", None)
        v.pop("pychrono_notes", None)
        v.pop("asset_layout", None)

        # Clean simulation_parameters: ensure dict
        if "simulation_parameters" in v and not isinstance(v["simulation_parameters"], dict):
            v["simulation_parameters"] = {}
        # Drop legacy physics_concepts (replaced by step.description per the
        # motion-CSV review contract). Tolerated on incoming plans for
        # back-compat with old artifacts.
        v.pop("physics_concepts", None)
        # Clean list-of-string fields
        list_str_fields = ["objectives"]
        for field in list_str_fields:
            if field in v:
                val = v[field]
                if isinstance(val, list):
                    v[field] = [str(x) for x in val if x is not None]
                else:
                    v[field] = []

        # ``clarifications_needed`` is List[Union[str, StructuredClarification]].
        # Preserve dict entries (Pydantic builds the structured model) and
        # coerce non-dict, non-string entries to str only when they are not
        # already in a model-compatible shape.
        if "clarifications_needed" in v:
            val = v["clarifications_needed"]
            if isinstance(val, list):
                cleaned: List[Any] = []
                for x in val:
                    if x is None:
                        continue
                    if isinstance(x, (str, dict)) or isinstance(x, StructuredClarification):
                        cleaned.append(x)
                    else:
                        cleaned.append(str(x))
                v["clarifications_needed"] = cleaned
            else:
                v["clarifications_needed"] = []

        # implementation_steps is now List[SimulationStep]. Accept list-of-dict
        # as-is (Pydantic will construct SimulationStep); coerce list-of-string
        # to a repair-friendly error via empty list + downstream validator so
        # the LLM sees the real schema error instead of a silent type coerce.
        if "implementation_steps" in v:
            val = v["implementation_steps"]
            if isinstance(val, list):
                cleaned_steps: List[Any] = []
                for item in val:
                    if isinstance(item, dict):
                        cleaned_steps.append(item)
                    elif hasattr(item, "model_dump"):
                        cleaned_steps.append(item.model_dump())
                    # Strings are skipped silently: validate_step_shape_matches_plan_type
                    # will raise if the plan_type requires steps, giving the LLM
                    # a clear "empty implementation_steps" error to repair against.
                v["implementation_steps"] = cleaned_steps
            else:
                v["implementation_steps"] = []

        # Legacy `milestones` field — silently drop. Older plans may be loaded
        # from disk; we don't want them to error out, but we also don't port
        # their data into the new shape automatically (mapping verify →
        # step.constraints + asset grouping isn't reliable).
        v.pop("milestones", None)
        # Clean visualization: ensure dict
        if "visualization" in v and not isinstance(v["visualization"], dict):
            v["visualization"] = {"mode": "headless", "apis": []}
        # Drop legacy output_requirements (pre-motion-CSV-contract field;
        # the per-step ``motion_expectations`` field on each step now
        # owns CSV output expectations). Tolerated on incoming plans for
        # back-compat with old artifacts; not stored on the model.
        v.pop("output_requirements", None)
        # Clean assets: skip non-dict items, normalize `count` to a positive int
        if "assets" in v:
            assets = v["assets"]
            if isinstance(assets, list):
                cleaned_assets = []
                for item in assets:
                    if isinstance(item, dict):
                        item = dict(item)
                        inferred_name = _asset_name_from_mapping(item)
                        if inferred_name:
                            item.setdefault("name", inferred_name)
                        cnt = item.get("count", 1)
                        try:
                            cnt = int(cnt) if cnt is not None else 1
                        except (TypeError, ValueError):
                            cnt = 1
                        if cnt < 1:
                            cnt = 1
                        item["count"] = cnt
                        cleaned_assets.append(item)
                v["assets"] = cleaned_assets
            else:
                v["assets"] = None
        # Clean scene_objects: skip malformed entries but keep flexible payloads.
        if "scene_objects" in v:
            scene_objects = v["scene_objects"]
            if isinstance(scene_objects, list):
                cleaned_scene_objects: List[Dict[str, Any]] = []
                for item in scene_objects:
                    if isinstance(item, dict):
                        item = dict(item)
                        name = str(item.get("name") or "").strip()
                        if name:
                            cleaned_scene_objects.append(item)
                v["scene_objects"] = cleaned_scene_objects
            else:
                v["scene_objects"] = []
        return v

    @model_validator(mode="after")
    def validate_topology_plan_type_consistency(self) -> "SimulationPlan":
        """Validate that topology field groups match the plan_type category."""
        topo = self.topology
        if topo is None:
            return self

        has_scene = bool(topo.physical_predicates or topo.scene_predicates)
        has_mbs = bool(topo.body_positions or topo.body_geometry or topo.joints)

        if self.plan_type == "scene" and has_mbs:
            raise ValueError(
                "plan_type='scene' should not have MBS topology fields "
                "(body_positions, body_geometry, joints)."
            )
        if self.plan_type == "mbs" and has_scene:
            raise ValueError(
                "plan_type='mbs' should not have scene topology fields "
                "(physical_predicates, scene_predicates)."
            )
        # mbs_in_scene and fsi_in_scene: both groups are expected, no
        # restriction. fsi_in_scene mirrors mbs_in_scene's topology contract
        # because fluid scenes still describe rigid-body placement (tank,
        # platforms, floating bodies, vehicle) AND scene-level predicates
        # (FREE-SURFACE-AT, FLOATS-AT-SURFACE).

        return self


    @model_validator(mode="after")
    def ensure_time_parameters(self) -> "SimulationPlan":
        """Enforce plan fidelity for user-specified simulation parameters.

        Policy (see feedback_no_silent_sim_params.md). This validator is a
        **planner-time** invariant — it runs during initial plan construction
        and plan modification, when ``_user_spec_var`` is set by the
        PlanningAgent around ``SimulationPlan(**plan_dict)``:

        * If ``UserSpec`` names a value, the plan MUST carry the same value.
          Mismatch or omission raises ``ValueError`` so the repair loop fires
          with the full user prompt still in context.
        * If ``UserSpec`` does NOT name the value and the plan carries one,
          the plan MUST carry a matching ``clarifications_needed`` entry
          (fuzzy substring match). Without one we raise ``ValueError`` so the
          repair loop re-runs the LLM — this is what prevents the weak-model
          failure mode where ``time_step: 0.01`` and ``simulation_duration:
          10.0`` appear out of thin air with no clarification trail.
        * If ``UserSpec`` does NOT name the value and the plan omits it, a
          runtime fallback is applied (downstream code needs these fields)
          and a WARNING is logged.

        Outside planner context (``_user_spec_var`` unset), the plan is
        being reconstructed from an already-validated serialized state —
        workflow step routing, subprocess replay, main.py dialog handoff,
        etc. In that case the silent-default policy has been enforced once
        already, so we apply only the structural fallback (fill missing
        fields) and trust any present values. Otherwise every reconstruction
        of a perfectly valid approved plan would fail re-validation because
        no downstream node has the original UserSpec in scope.

        ``gravity`` is intentionally exempt from this cross-check — the
        template hardcodes ``-9.81`` and generating a clarification on
        every run would be pure noise for the 99% Earth-gravity case.
        """
        import logging as _logging

        log = _logging.getLogger(__name__)

        spec = _user_spec_var.get()
        params = self.simulation_parameters

        def _user_asserted(current: Any, expected: Optional[float]) -> bool:
            if expected is None:
                return False
            if current is None:
                return False
            try:
                return abs(float(current) - float(expected)) <= 1e-9
            except (TypeError, ValueError):
                return False

        def _apply_fallback(key: str, fallback: float) -> None:
            if key not in params:
                params[key] = fallback
                log.warning(
                    "[SimulationPlan] %s was missing and not named by UserSpec; "
                    "applying runtime fallback %s. The planner should have "
                    "either copied a user-specified value or added a "
                    "clarifications_needed entry — neither happened.",
                    key,
                    fallback,
                )

        def _has_clarification(keywords: Tuple[str, ...]) -> bool:
            """True iff any `clarifications_needed` entry mentions any keyword.

            Fuzzy, case-insensitive substring match — LLMs phrase questions
            variably ("What time step do you want?" / "simulation duration?")
            and we don't want to reject good plans on cosmetic wording.
            """
            for entry in self.clarifications_needed or []:
                if isinstance(entry, StructuredClarification):
                    # Match against the question text plus option labels /
                    # descriptions, since the user-visible "what is being
                    # asked" lives across all three fields.
                    parts = [entry.question]
                    # New shape: ``options`` is List[str] of labels;
                    # rich descriptions live on ``option_details``.
                    parts.extend(str(opt) for opt in (entry.options or []))
                    parts.extend(
                        det.description for det in (entry.option_details or [])
                        if det.description
                    )
                    if entry.unit:
                        parts.append(entry.unit)
                    entry_l = " ".join(parts).lower()
                else:
                    entry_l = str(entry).lower()
                if any(kw.lower() in entry_l for kw in keywords):
                    return True
            return False

        def _require_clarification_or_fallback(
            key: str,
            clarification_keywords: Tuple[str, ...],
            fallback: float,
        ) -> None:
            """Case D: user didn't name the value.

            If the plan carries a value (LLM filled a silent default) the plan
            MUST carry a matching `clarifications_needed` entry asking the user
            for it. Missing clarification → raise so the repair loop re-runs
            the LLM with the validation error appended. If the plan omits the
            value entirely, fall back (same as the old behavior).
            """
            if key in params:
                if not _has_clarification(clarification_keywords):
                    raise ValueError(
                        f"simulation_parameters.{key}={params.get(key)!r} was "
                        f"filled in without a matching `clarifications_needed` "
                        f"entry, and the user did not name this value in USER "
                        f"SPEC. Either (a) remove the value and add a "
                        f"clarifications_needed entry asking the user for it, "
                        f"or (b) keep the value AND add a clarifications_needed "
                        f"entry confirming the assumed default (e.g. "
                        f"'Assumed {key}={params.get(key)} — please confirm or "
                        f"override.'). A silent default is not acceptable per "
                        f"the no-silent-sim-params policy."
                    )
                # Clarification present → plan is honest about the assumption;
                # keep the value as-is.
                return
            _apply_fallback(key, fallback)

        # Reconstruction short-circuit: no planner UserSpec context means
        # we're rehydrating a previously-validated plan. Only structural
        # fallback applies — the silent-default policy already fired once
        # when the plan was first built.
        if spec is None:
            if "time_step" not in params:
                _apply_fallback("time_step", 0.001)
            if "simulation_duration" not in params:
                _apply_fallback("simulation_duration", 5.0)
            return self

        # time_step
        if getattr(spec, "time_step_s", None) is not None:
            if not _user_asserted(params.get("time_step"), spec.time_step_s):
                raise ValueError(
                    f"simulation_parameters.time_step={params.get('time_step')!r} "
                    f"does not match user-specified time_step_s={spec.time_step_s}. "
                    "The user named this value explicitly; the plan must honor it."
                )
        else:
            _require_clarification_or_fallback(
                key="time_step",
                clarification_keywords=("time_step", "time step", "timestep", "dt"),
                fallback=0.001,
            )

        # simulation_duration
        if getattr(spec, "duration_s", None) is not None:
            if not _user_asserted(params.get("simulation_duration"), spec.duration_s):
                raise ValueError(
                    f"simulation_parameters.simulation_duration="
                    f"{params.get('simulation_duration')!r} does not match "
                    f"user-specified duration_s={spec.duration_s}. "
                    "The user named this value explicitly; the plan must honor it."
                )
        else:
            _require_clarification_or_fallback(
                key="simulation_duration",
                clarification_keywords=(
                    "simulation_duration", "simulation duration",
                    "duration", "how long", "run time", "runtime", "sim_time",
                ),
                fallback=5.0,
            )

        # gravity is intentionally NOT cross-checked: the prompt template
        # hardcodes `gravity: -9.81` (Earth default). If the user ever asks
        # for Mars / Moon / zero-g, that's extracted via a separate channel;
        # treating every plan as "did you ask about gravity?" would generate
        # endless cosmetic clarifications on 99% of runs.
        return self

    @model_validator(mode="after")
    def validate_step_shape_matches_plan_type(self) -> "SimulationPlan":
        """Enforce the plan_type ↔ implementation_steps contract.

        - ``scene`` / ``mbs_in_scene`` / ``fsi_in_scene``:
          ``implementation_steps`` MUST be non-empty; each step carries its
          own camera for VLM review. fsi_in_scene is included because the
          step-loop / per-step VLM workflow applies identically once a
          plan is hybrid (catalog assets + procedural / fluid bodies).
        - ``mbs``: ``implementation_steps`` SHOULD be empty; the workflow
          builds the sim monolithically. A non-empty list here isn't fatal
          (legacy conversion paths may produce it) but is unusual.
        """
        if self.plan_type in ("scene", "mbs_in_scene", "fsi_in_scene"):
            if not self.implementation_steps:
                raise ValueError(
                    f"plan_type={self.plan_type!r} requires a non-empty "
                    "`implementation_steps`. Each step represents one build "
                    "phase reviewed against its own camera frames."
                )
        return self

    @model_validator(mode="after")
    def validate_camera_count_matches_recording_mode(self) -> "SimulationPlan":
        """Enforce per-step camera count appropriate to the recording mode.

        ``vsg_only`` mode (FSI / SPH scenes) renders ONE viewpoint at a
        time — VSG has a single active camera, the recorded mp4 captures
        only that view, and ``setup_vsg_recording`` is single-camera by
        construction. Plans that emit 2-3 cameras per step in this mode
        produce contradictory codegen instructions: the per-step camera
        block tells codegen to wire one ``setup_preview_camera`` per
        camera, while ``VIDEO_GENERATION_RULE_VSG_ONLY`` says
        ``setup_preview_camera`` is forbidden. With no resolution rule,
        codegen guesses which camera to lock to and the recorded mp4 has
        nothing to do with the planned poses.

        ``sensor_cams`` mode (the default for non-FSI scenes) keeps the
        legacy 2-3 cameras rule: each ``CameraPose`` becomes its own
        ``setup_preview_camera`` call → its own mp4 file, so multiple
        complementary angles per step is the right answer there.

        Validation runs after step-shape validation, so ``cameras``
        already exists on each step (min_length=1 from the field).
        """
        for idx, step in enumerate(self.implementation_steps or []):
            n = len(step.cameras)
            if self.recording_mode == "vsg_only":
                if n != 1:
                    raise ValueError(
                        f"implementation_steps[{idx}] has {n} cameras but "
                        f"recording_mode='vsg_only' renders exactly ONE "
                        f"viewpoint per mp4. Emit a single camera per step "
                        f"that frames the scene's primary action (e.g. a "
                        f"side view of the tank + vehicle). For multiple "
                        f"complementary angles, switch the plan to "
                        f"recording_mode='sensor_cams'."
                    )
            elif self.recording_mode == "sensor_cams":
                if not (2 <= n <= 3):
                    raise ValueError(
                        f"implementation_steps[{idx}] has {n} cameras but "
                        f"recording_mode='sensor_cams' wants 2-3 "
                        f"complementary viewing directions per step "
                        f"(e.g. wide-NE + wide-NW + top-down) so the VLM "
                        f"reviews the full environment, not a single slice."
                    )
        return self

    @model_validator(mode="after")
    def validate_motion_expectations_resolve(self) -> "SimulationPlan":
        """Reject motion_expectations that name a static / fixed body.

        Each name in ``step.motion_expectations`` should resolve to either
        an asset with ``is_dynamic=True`` or a scene_object with
        ``fixed=False``. Names that don't resolve at all only emit a
        warning (the planner may use abbreviated forms; codegen normalizes
        names downstream). Names that resolve to a body the plan declares
        static are a hard error — the planner contradicted itself.
        """
        # Build dynamic-name lookup. Asset is_dynamic defaults False;
        # scene_object fixed defaults True (treat absent fixed as static
        # so we don't accidentally green-light typos).
        dynamic_names: set[str] = set()
        static_names: set[str] = set()
        for a in self.assets or []:
            if not isinstance(a, dict):
                continue
            n = str(a.get("name") or "").strip().lower()
            if not n:
                continue
            if a.get("is_dynamic"):
                dynamic_names.add(n)
            else:
                static_names.add(n)
        for obj in self.scene_objects or []:
            if not isinstance(obj, dict):
                continue
            n = str(obj.get("name") or "").strip().lower()
            if not n:
                continue
            if obj.get("fixed", True):
                static_names.add(n)
            else:
                dynamic_names.add(n)

        for idx, step in enumerate(self.implementation_steps or []):
            for name in step.motion_expectations:
                low = str(name).strip().lower()
                if low in static_names and low not in dynamic_names:
                    raise ValueError(
                        f"implementation_steps[{idx}].motion_expectations "
                        f"includes {name!r}, but that body is declared "
                        f"static (asset is_dynamic=False or scene_object "
                        f"fixed=True). Either drop it from "
                        f"motion_expectations or mark the body dynamic."
                    )
        return self

    def derive_dynamic_bodies(self) -> List[str]:
        """Names of every body the plan declares dynamic.

        Pulls from ``assets[*].is_dynamic`` (asset rows where the flag is
        truthy) and ``scene_objects[*].fixed`` (rows where the flag is
        falsy). Used by the scene-placement validator and any other site
        that needs to know "which bodies should be allowed to move at all".
        Order: assets first, then scene_objects; names de-duplicated
        (lower-case keyed) while preserving first occurrence.
        """
        seen: set[str] = set()
        names: List[str] = []
        for a in self.assets or []:
            if not isinstance(a, dict) or not a.get("is_dynamic"):
                continue
            n = str(a.get("name") or "").strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                names.append(n)
        for obj in self.scene_objects or []:
            if not isinstance(obj, dict) or obj.get("fixed", True):
                continue
            n = str(obj.get("name") or "").strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                names.append(n)
        return names

    def dump_all(self) -> dict:
        """Dump all fields including those marked exclude=True.

        Pydantic's model_dump() respects field-level exclude=True and drops
        implementation_steps, assets, topology, etc. This method includes them
        so that downstream consumers (step_router, codegen) get the full plan.
        """
        d = self.model_dump()
        for field_name, field_info in type(self).model_fields.items():
            if field_info.exclude and field_name not in d:
                val = getattr(self, field_name, None)
                if val is None:
                    continue
                if hasattr(val, "model_dump"):
                    d[field_name] = val.model_dump()
                elif isinstance(val, list):
                    d[field_name] = [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in val
                    ]
                elif isinstance(val, dict):
                    d[field_name] = dict(val)
                else:
                    d[field_name] = val
        return d

    def build_step_context(
        self,
        step_index: int,
        completed_step_descriptions: Optional[List[str]] = None,
    ) -> "StepContext":
        """Build a per-step context bundle for the code generation agent.

        Derives everything from ``plan.objects[]`` (new unified schema). Each
        step lists names in ``step.objects`` (or legacy ``step.assets`` /
        ``step.scene_objects`` if a plan still uses the old shape); we look up
        the full Object record and split into asset / procedural for codegen.
        Children's pose.position is None — codegen computes it from
        relation + ref pose+size + obj size at codegen time.
        """
        completed_descs = list(completed_step_descriptions or [])
        steps = self.implementation_steps or []
        if step_index >= len(steps):
            raise IndexError(
                f"step_index {step_index} out of range for {len(steps)} steps"
            )

        current: "SimulationStep" = steps[step_index]
        completed_steps = list(steps[:step_index])

        # --- Object lookup by (lowercased) name ---
        objects_by_name: Dict[str, Object] = {
            o.name.lower(): o for o in (self.objects or [])
        }
        # Legacy fallback: a plan that came in via the older
        # ``## scene_objects`` / ``## assets`` markdown shape (and never got
        # rehydrated into ``self.objects``) still needs working step
        # lookups. Index those raw dicts by name so ``_split_step_objects``
        # can return them through the legacy-shape branch below.
        legacy_assets_by_name: Dict[str, Dict[str, Any]] = {
            str(a.get("name", "")).lower(): a
            for a in (self.assets or [])
            if isinstance(a, dict) and a.get("name")
        }
        legacy_scene_objects_by_name: Dict[str, Dict[str, Any]] = {
            str(s.get("name", "")).lower(): s
            for s in (self.scene_objects or [])
            if isinstance(s, dict) and s.get("name")
        }

        all_object_names = [o.name for o in (self.objects or [])] + list(
            legacy_assets_by_name.keys()
        ) + list(legacy_scene_objects_by_name.keys())
        is_single_step = len(steps) == 1

        def _step_object_names(step: "SimulationStep") -> List[str]:
            """Names introduced by a step — prefer new ``objects``, fall back to legacy.

            Final fallback for single-step plans: if the step lists no
            object names (a planner-side LLM oversight that strips every
            body's size / pose from ``step_context`` and forces codegen
            to invent dimensions from prose), default to every body in
            ``## objects``. Multi-step plans keep returning ``[]`` so we
            don't leak a later step's bodies into an earlier step.
            """
            names = list(step.objects or [])
            if not names:
                names = list(step.assets or []) + list(step.scene_objects or [])
            if not names and is_single_step and all_object_names:
                names = list(all_object_names)
            return names

        def _object_to_asset_dict(o: Object) -> Dict[str, Any]:
            """Project an Object to legacy asset-shape dict for codegen back-compat."""
            c = o.construction
            d: Dict[str, Any] = {
                "name": o.name,
                "type": c.asset_type,
                "is_dynamic": o.is_dynamic,
                "fixed": o.fixed,
                "description": o.description or "",
            }
            if c.filename:
                d["filename"] = c.filename
            if c.factory:
                d["factory"] = c.factory
            return d

        def _object_to_scene_object_dict(o: Object) -> Dict[str, Any]:
            """Project an Object to legacy scene_object-shape dict for codegen back-compat."""
            c = o.construction
            d: Dict[str, Any] = {
                "name": o.name,
                "construction_source": (
                    c.primitive if c.primitive in ("fluid_domain", "generated_boundary")
                    else "procedural_primitive"
                ),
                "primitive": c.primitive,
                "size": c.size,
                "fixed": o.fixed,
                "description": o.description or "",
            }
            if c.density is not None:
                d["density"] = c.density
            if c.viscosity is not None:
                d["viscosity"] = c.viscosity
            if c.mass is not None:
                d["mass"] = c.mass
            if o.fsi_registration:
                d["fsi_registration"] = o.fsi_registration
            return d

        def _object_to_predicate_dict(o: Object) -> Optional[Dict[str, Any]]:
            """Project a child Object's topology to a legacy scene_predicate dict.

            Position is NOT resolved here — codegen reads relation + ref to
            compute it. We carry the symbolic relation and the resolved ref
            pose/size so codegen has everything it needs.
            """
            t = o.topology
            if t.role != "child":
                return None
            ref = objects_by_name.get((t.ref or "").lower())
            if ref is None:
                return None
            return {
                "subject": o.name,
                "predicate": t.relation,
                "object": t.ref,
                "params": dict(getattr(t, "params", None) or {}),
                "ref_pose": (
                    ref.pose.position if ref.pose and ref.pose.position else None
                ),
                "ref_size": ref.construction.size,
                "obj_size": o.construction.size,
                "obj_role": "child",
            }

        def _split_step_objects(names: List[str]) -> tuple:
            """Split a step's object names into (full Object records, asset dicts, scene_obj dicts)."""
            step_objs: List[Dict[str, Any]] = []
            assets: List[Dict[str, Any]] = []
            scene_objs: List[Dict[str, Any]] = []
            for n in names:
                low = str(n).lower()
                o = objects_by_name.get(low)
                if o is not None:
                    step_objs.append(o.model_dump())
                    if o.construction.kind == "asset":
                        assets.append(_object_to_asset_dict(o))
                    else:
                        scene_objs.append(_object_to_scene_object_dict(o))
                    continue
                # Legacy plans (no unified ``self.objects[]``) — surface the
                # raw legacy dicts so codegen still has the per-step manifest.
                if low in legacy_assets_by_name:
                    a = legacy_assets_by_name[low]
                    step_objs.append(dict(a))
                    assets.append(dict(a))
                elif low in legacy_scene_objects_by_name:
                    s = legacy_scene_objects_by_name[low]
                    step_objs.append(dict(s))
                    scene_objs.append(dict(s))
            return step_objs, assets, scene_objs

        current_names = _step_object_names(current)
        step_objects, step_assets, step_scene_objects = _split_step_objects(current_names)

        current_asset: Optional[Dict[str, Any]] = step_assets[0] if step_assets else None

        # --- Completed objects (union across prior steps) ---
        completed_assets: List[Dict[str, Any]] = []
        completed_scene_objects: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for prev in completed_steps:
            for n in _step_object_names(prev):
                low = str(n).lower()
                if low in seen:
                    continue
                seen.add(low)
                o = objects_by_name.get(low)
                if o is None:
                    continue
                if o.construction.kind == "asset":
                    completed_assets.append({"name": o.name, "filename": o.construction.filename or ""})
                else:
                    completed_scene_objects.append({
                        "name": o.name,
                        "construction_source": (
                            o.construction.primitive
                            if o.construction.primitive in ("fluid_domain", "generated_boundary")
                            else "procedural_primitive"
                        ),
                    })

        # --- scene_predicates derived from children whose ref+subject are both relevant ---
        relevant = {str(n).lower() for n in current_names}
        relevant.update(seen)
        relevant.update({"floor", "ground", "terrain", "root"})
        scene_preds: List[Dict[str, Any]] = []
        for o in self.objects or []:
            if o.name.lower() not in relevant:
                continue
            pred = _object_to_predicate_dict(o)
            if pred and pred["object"] and pred["object"].lower() in relevant:
                scene_preds.append(pred)

        # --- Topology meta ---
        topo = self.topology
        topology_meta: Dict[str, Any] = {}
        if topo:
            topology_meta = {
                "gravity_axis": topo.gravity_axis,
                "working_plane": topo.working_plane,
                "orientation_convention": topo.orientation_convention,
            }
            if topo.cameras is not None:
                topology_meta["cameras"] = topo.cameras
            if topo.camera_target is not None:
                topology_meta["camera_target"] = topo.camera_target
            if topo.camera_up is not None:
                topology_meta["camera_up"] = topo.camera_up
            if topo.camera_target_body is not None:
                topology_meta["camera_target_body"] = topo.camera_target_body
            if topo.scene_size is not None:
                topology_meta["scene_size"] = topo.scene_size

        return StepContext(
            step_index=step_index,
            total_steps=len(steps),
            step_description=current.description,
            step_assets=step_assets,
            step_scene_objects=step_scene_objects,
            step_objects=step_objects,
            step_cameras=[c.model_dump() for c in current.cameras],
            step_constraints=list(current.constraints),
            step_motion_expectations=list(current.motion_expectations),
            prior_constraints=[
                {"step_index": i, "constraints": list(s.constraints)}
                for i, s in enumerate(completed_steps) if s.constraints
            ],
            current_asset=current_asset,
            scene_predicates=scene_preds,
            physical_predicates=[],
            simulation_parameters=dict(self.simulation_parameters),
            topology_meta=topology_meta,
            completed_assets=completed_assets,
            completed_scene_objects=completed_scene_objects,
            completed_steps=completed_descs,
        )

    class Config:
        json_schema_extra = {
            "example": {
                "plan_type": "mbs",
                "simulation_parameters": {
                    "time_step": 0.01,
                    "simulation_duration": 5.0,
                    "gravity": -9.81,
                    "ball_mass": 1.0,
                    "ball_radius": 0.5,
                    "ball_initial_height": 2.0,
                    "ground_friction": 0.5,
                    "ball_restitution": 0.8
                },
                "objectives": [
                    "Create a bouncing ball simulation",
                    "Demonstrate basic collision physics"
                ],
                "implementation_steps": [
                    "Initialize simulation environment",
                    "Create ground plane",
                    "Create sphere with mass and initial position",
                    "Set gravity and material properties",
                    "Add CSV logging and time series plot",
                    "Run simulation loop with visualization"
                ],
                "clarifications_needed": [],
                "visualization": {
                    "mode": "headless",
                    "apis": []
                },
                "topology": {
                    "gravity_axis": "-z",
                    "working_plane": "xz",
                    "body_positions": None,
                    "body_geometry": None,
                    "joints": None,
                    "cameras": None,
                    "camera_target": None,
                    "camera_up": None
                },
                "assets": None
            }
        }
