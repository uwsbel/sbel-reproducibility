"""
Deterministic scene-placement validator.

Reads ``scene_placement.csv`` and ``scene_contacts.csv`` produced by the
generated simulation code and checks them against the plan's
``physical_predicates`` and ``scene_predicates``.

Only used for **scene** plan types.  MBS plans keep the existing LLM-based
physics analysis path.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from chrono_agent.validators.body_classification import split_dynamic_static

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
SURFACE_CONTACT_TOL = 0.08   # metres – gap between bottom/top faces
VELOCITY_TOL = 0.05          # m/s – linear velocity magnitude threshold
ANG_VELOCITY_TOL = 0.1       # rad/s – angular velocity magnitude threshold
OVERLAP_TOL = 0.02           # metres – min AABB overlap for footprint check
CONTAINMENT_TOL = 0.05       # metres – slack for "contains" check
PROXIMITY_TOL = 0.15         # metres – max gap for "next_to" / "leans_against"
CENTERING_TOL = 0.10         # metres – max horizontal offset for "centered_on"
INTERPENETRATION_VOL_TOL = OVERLAP_TOL ** 3   # m³ slack for SMC contact penetration

# Body-name regex matching wheel spindles emitted by ``ChWheeledVehicle``
# variants (Polaris, HMMWV, ARTcar, ...). Matches ``spindle_FL``,
# ``Spindle_RR``, ``wheel_FL_spindle`` etc. Case-insensitive.
_WHEEL_NAME_PATTERN = re.compile(r"(?:^|_)spindle(?:$|_)|(?:^|_)wheel(?:$|_)", re.I)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class PredicateResult:
    predicate: Dict[str, Any]
    passed: bool
    reason: str
    measured_values: Dict[str, Any] = field(default_factory=dict)
    # ``kind`` identifies hard-override predicates whose FAIL outranks LLM
    # judgment in step_review_node. Empty string for legacy predicates.
    kind: str = ""


@dataclass
class SceneValidationResult:
    verdict: str                          # "physics_valid" | "physics_invalid"
    predicate_results: List[PredicateResult] = field(default_factory=list)
    stability_passed: bool = True
    stability_detail: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _load_placement_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Return ``{body_name: {col: float_value, ...}}``."""
    rows: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("body_name", "").strip()
            if not name:
                continue
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if k == "body_name":
                    continue
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = 0.0
            rows[name] = parsed
    return rows


def _load_contacts_csv(path: str) -> List[Dict[str, Any]]:
    """Return list of ``{"body1": str, "body2": str, "force_magnitude": float}``."""
    contacts: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contacts.append({
                "body1": row.get("body1", "").strip(),
                "body2": row.get("body2", "").strip(),
                "force_magnitude": float(row.get("force_magnitude", 0)),
            })
    return contacts


def _load_particles_csv(path: str) -> List[Tuple[float, float, float, float]]:
    """Return ``[(time, x, y, z), ...]`` from a particles CSV.

    Codegen writes one row per SPH particle for the final sim time.
    Columns: ``time, particle_id, x, y, z``. Missing or unparseable
    rows are skipped silently — this is best-effort diagnostic data.
    """
    particles: List[Tuple[float, float, float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get("time", 0))
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except (KeyError, ValueError, TypeError):
                continue
            particles.append((t, x, y, z))
    return particles


def _load_links_csv(path: str) -> Set[frozenset]:
    """Return ``{frozenset({body1, body2}), ...}`` from a scene_links CSV.

    Used to filter joint-connected body pairs out of the no_interpenetration
    discovery. Names lowercased so the lookup is case-insensitive (matches
    ``_has_contact``'s convention).
    """
    pairs: Set[frozenset] = set()
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = row.get("body1", "").strip().lower()
                b = row.get("body2", "").strip().lower()
                if a and b and a != b:
                    pairs.add(frozenset({a, b}))
    except OSError:
        pass
    return pairs


def _has_contact(contacts: List[Dict[str, Any]], a: str, b: str) -> bool:
    a_low, b_low = a.lower(), b.lower()
    return any(
        (c["body1"].lower() == a_low and c["body2"].lower() == b_low) or
        (c["body1"].lower() == b_low and c["body2"].lower() == a_low)
        for c in contacts
    )


# ---------------------------------------------------------------------------
# Gravity helpers
# ---------------------------------------------------------------------------

def _gravity_axis_index(gravity_axis: str) -> int:
    """Return (index, sign) for the gravity axis string like '-z'."""
    axis = gravity_axis.lstrip("+-").lower()
    return {"x": 0, "y": 1, "z": 2}.get(axis, 2)


def _up_axis(gravity_axis: str) -> str:
    return gravity_axis.lstrip("+-").lower()


def _get_pos(row: Dict[str, float], axis: str) -> float:
    return row.get(f"pos_{axis}", 0.0)


def _get_aabb_min(row: Dict[str, float], axis: str) -> float:
    return row.get(f"aabb_min_{axis}", 0.0)


def _get_aabb_max(row: Dict[str, float], axis: str) -> float:
    return row.get(f"aabb_max_{axis}", 0.0)


def _horizontal_axes(gravity_axis: str) -> list[str]:
    up = _up_axis(gravity_axis)
    return [a for a in ("x", "y", "z") if a != up]


# ---------------------------------------------------------------------------
# Predicate checkers
# ---------------------------------------------------------------------------

def _check_rests_on(
    subject: Dict[str, float],
    obj: Dict[str, float],
    contacts: List[Dict[str, Any]],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """subject rests_on object  →  subject bottom ≈ object top + footprint overlap."""
    up = _up_axis(gravity_axis)
    subj_bottom = _get_aabb_min(subject, up)
    obj_top = _get_aabb_max(obj, up)
    gap = abs(subj_bottom - obj_top)

    # Footprint overlap on horizontal axes
    h_axes = _horizontal_axes(gravity_axis)
    overlap = True
    for ax in h_axes:
        s_min, s_max = _get_aabb_min(subject, ax), _get_aabb_max(subject, ax)
        o_min, o_max = _get_aabb_min(obj, ax), _get_aabb_max(obj, ax)
        inter = min(s_max, o_max) - max(s_min, o_min)
        if inter < OVERLAP_TOL:
            overlap = False
            break

    passed = gap < SURFACE_CONTACT_TOL and overlap
    return PredicateResult(
        predicate={"subject": subject_name, "predicate": "rests_on", "object": obj_name},
        passed=passed,
        reason=(
            f"gap={gap:.3f}m (tol={SURFACE_CONTACT_TOL}), "
            f"footprint_overlap={'yes' if overlap else 'no'}"
        ),
        measured_values={"gap": gap, "footprint_overlap": overlap},
    )


def _check_supports(
    subject: Dict[str, float],
    obj: Dict[str, float],
    contacts: List[Dict[str, Any]],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """subject supports object  →  object rests_on subject."""
    result = _check_rests_on(obj, subject, contacts, obj_name, subject_name, gravity_axis)
    result.predicate = {"subject": subject_name, "predicate": "supports", "object": obj_name}
    return result


def _check_contains(
    container: Dict[str, float],
    contained: Dict[str, float],
    container_name: str,
    contained_name: str,
) -> PredicateResult:
    """contained AABB fully inside container AABB."""
    inside = True
    for ax in ("x", "y", "z"):
        if _get_aabb_min(contained, ax) < _get_aabb_min(container, ax) - CONTAINMENT_TOL:
            inside = False
            break
        if _get_aabb_max(contained, ax) > _get_aabb_max(container, ax) + CONTAINMENT_TOL:
            inside = False
            break
    return PredicateResult(
        predicate={"subject": container_name, "predicate": "contains", "object": contained_name},
        passed=inside,
        reason="contained AABB within container" if inside else "contained AABB exceeds container",
    )


def _check_leans_against(
    subject: Dict[str, float],
    obj: Dict[str, float],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """Subject AABB nearly touches object AABB on a lateral face."""
    h_axes = _horizontal_axes(gravity_axis)
    min_gap = float("inf")
    for ax in h_axes:
        gap1 = abs(_get_aabb_max(subject, ax) - _get_aabb_min(obj, ax))
        gap2 = abs(_get_aabb_max(obj, ax) - _get_aabb_min(subject, ax))
        min_gap = min(min_gap, gap1, gap2)
    passed = min_gap < PROXIMITY_TOL
    return PredicateResult(
        predicate={"subject": subject_name, "predicate": "leans_against", "object": obj_name},
        passed=passed,
        reason=f"min_lateral_gap={min_gap:.3f}m (tol={PROXIMITY_TOL})",
        measured_values={"min_lateral_gap": min_gap},
    )


def _check_attached_to(
    subject: Dict[str, float],
    obj: Dict[str, float],
    contacts: List[Dict[str, Any]],
    subject_name: str,
    obj_name: str,
) -> PredicateResult:
    """Contact exists between the two bodies."""
    has = _has_contact(contacts, subject_name, obj_name)
    return PredicateResult(
        predicate={"subject": subject_name, "predicate": "attached_to", "object": obj_name},
        passed=has,
        reason="contact detected" if has else "no contact found",
    )


# ---------------------------------------------------------------------------
# Hard-override deterministic predicates (added per session 103931)
# ---------------------------------------------------------------------------


def _aabb_overlap_volume(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Return ``(overlap_volume_m3, per_axis_overlap_metres)`` for two AABBs."""
    overlap: Dict[str, float] = {}
    for ax in ("x", "y", "z"):
        a_min, a_max = _get_aabb_min(a, ax), _get_aabb_max(a, ax)
        b_min, b_max = _get_aabb_min(b, ax), _get_aabb_max(b, ax)
        overlap[ax] = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    vol = overlap["x"] * overlap["y"] * overlap["z"]
    return vol, overlap


def _check_wheel_landing(
    wheel: Dict[str, float],
    platform: Dict[str, float],
    contacts: List[Dict[str, Any]],
    wheel_name: str,
    platform_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """Per-wheel deterministic check: wheel bottom must sit on platform top.

    Pass iff:
      1. ``|wheel.aabb_min_up - platform.aabb_max_up| <= SURFACE_CONTACT_TOL`` (touching)
      2. wheel and platform appear paired in scene_contacts.csv

    Reports ``hanging`` (positive gap) or ``clipping`` (negative gap, i.e.
    wheel bottom is BELOW platform top — wheel intrudes into platform body).
    """
    up = _up_axis(gravity_axis)
    wheel_bottom = _get_aabb_min(wheel, up)
    platform_top = _get_aabb_max(platform, up)
    signed_gap = wheel_bottom - platform_top
    abs_gap = abs(signed_gap)
    in_contact = _has_contact(contacts, wheel_name, platform_name)

    passed = abs_gap <= SURFACE_CONTACT_TOL and in_contact

    if signed_gap > SURFACE_CONTACT_TOL:
        verdict = "HANGING"
        narrative = (
            f"{wheel_name} bottom {up}={wheel_bottom:.3f}m is {signed_gap:.3f}m "
            f"ABOVE {platform_name} top {up}={platform_top:.3f}m "
            f"(threshold {SURFACE_CONTACT_TOL}m); "
            f"{'in contact list' if in_contact else 'not in contact list'} — "
            f"wheel is {verdict}"
        )
    elif signed_gap < -SURFACE_CONTACT_TOL:
        verdict = "CLIPPING"
        narrative = (
            f"{wheel_name} bottom {up}={wheel_bottom:.3f}m is {abs_gap:.3f}m "
            f"BELOW {platform_name} top {up}={platform_top:.3f}m — wheel "
            f"INTRUDES into platform body ({verdict})"
        )
    elif not in_contact:
        narrative = (
            f"{wheel_name} bottom within {SURFACE_CONTACT_TOL}m of "
            f"{platform_name} top but no contact recorded — likely barely "
            f"touching or platform-collision shape missing"
        )
    else:
        narrative = (
            f"{wheel_name} bottom {up}={wheel_bottom:.3f}m sits on "
            f"{platform_name} top {up}={platform_top:.3f}m (gap "
            f"{abs_gap:.3f}m, in contact) — landed"
        )

    return PredicateResult(
        predicate={
            "kind": "wheel_landing",
            "subject": wheel_name,
            "predicate": "wheels_landed_on",
            "object": platform_name,
        },
        passed=passed,
        reason=narrative,
        measured_values={
            "wheel_bottom": wheel_bottom,
            "platform_top": platform_top,
            "signed_gap": signed_gap,
            "in_contact": in_contact,
        },
        kind="wheel_landing",
    )


def _check_no_interpenetration(
    a: Dict[str, float],
    b: Dict[str, float],
    a_name: str,
    b_name: str,
) -> PredicateResult:
    """Pair of bodies must not have overlapping AABBs.

    Tolerance ``INTERPENETRATION_VOL_TOL`` (≈ 8 cm³) absorbs the small
    SMC contact penetration that ChCollisionShape negotiates at rest.
    """
    vol, per_axis = _aabb_overlap_volume(a, b)
    passed = vol <= INTERPENETRATION_VOL_TOL

    if passed:
        narrative = (
            f"{a_name} & {b_name} AABBs disjoint or within tolerance "
            f"(overlap_vol={vol:.6f}m³, tol={INTERPENETRATION_VOL_TOL:.6f}m³)"
        )
    else:
        worst_axis = max(("x", "y", "z"), key=lambda ax: per_axis[ax])
        narrative = (
            f"{a_name} & {b_name} AABBs overlap by {vol:.4f}m³ "
            f"(per-axis dx={per_axis['x']:.3f}m dy={per_axis['y']:.3f}m "
            f"dz={per_axis['z']:.3f}m, worst on {worst_axis}); "
            f"threshold {INTERPENETRATION_VOL_TOL:.6f}m³ — bodies are CLIPPING"
        )

    return PredicateResult(
        predicate={
            "kind": "no_interpenetration",
            "subject": a_name,
            "predicate": "no_overlap",
            "object": b_name,
        },
        passed=passed,
        reason=narrative,
        measured_values={"overlap_volume": vol, **{f"d{ax}": per_axis[ax] for ax in ("x", "y", "z")}},
        kind="no_interpenetration",
    )


def _check_fluid_containment(
    container: Dict[str, float],
    container_name: str,
    fluid_name: str,
    particles: List[Tuple[float, float, float, float]],
    gravity_axis: str,
) -> PredicateResult:
    """Every SPH particle must lie within the container's AABB.

    Reads particle positions from the last frame in ``particles.csv``
    (or the only frame, when codegen dumps just the final state).
    Counts particles that escape on any axis; reports min observed
    on the gravity axis (the most common leak direction).
    """
    if not particles:
        return PredicateResult(
            predicate={
                "kind": "fluid_containment",
                "subject": fluid_name,
                "predicate": "contained_in",
                "object": container_name,
            },
            passed=False,
            reason=(
                f"FLUID_CONTAINMENT: SKIPPED — no particles.csv data; "
                f"codegen must call sysSPH.GetParticlePositions() and write "
                f"particles.csv at end of sim (HR-16) so leaks can be detected"
            ),
            kind="fluid_containment",
        )

    # Use the last frame only — codegen typically dumps a single sim-end frame.
    last_t = max(p[0] for p in particles)
    last_frame = [p for p in particles if p[0] == last_t]

    cmin_x = _get_aabb_min(container, "x") - SURFACE_CONTACT_TOL
    cmax_x = _get_aabb_max(container, "x") + SURFACE_CONTACT_TOL
    cmin_y = _get_aabb_min(container, "y") - SURFACE_CONTACT_TOL
    cmax_y = _get_aabb_max(container, "y") + SURFACE_CONTACT_TOL
    cmin_z = _get_aabb_min(container, "z") - SURFACE_CONTACT_TOL
    cmax_z = _get_aabb_max(container, "z") + SURFACE_CONTACT_TOL

    escaped: List[Tuple[float, float, float]] = []
    min_obs_z = float("inf")
    for _, px, py, pz in last_frame:
        out = (
            px < cmin_x or px > cmax_x
            or py < cmin_y or py > cmax_y
            or pz < cmin_z or pz > cmax_z
        )
        if out:
            escaped.append((px, py, pz))
        if pz < min_obs_z:
            min_obs_z = pz

    n_escaped = len(escaped)
    passed = n_escaped == 0

    up = _up_axis(gravity_axis)
    if passed:
        narrative = (
            f"all {len(last_frame)} particles contained in {container_name} "
            f"AABB (tol {SURFACE_CONTACT_TOL}m); min observed {up}={min_obs_z:.3f}m, "
            f"floor {up}={cmin_z + SURFACE_CONTACT_TOL:.3f}m"
        )
    else:
        n_below = sum(1 for px, py, pz in escaped if pz < cmin_z)
        narrative = (
            f"{n_escaped} of {len(last_frame)} SPH particles escape "
            f"{container_name} AABB; {n_below} found BELOW tank floor "
            f"({up}<{cmin_z + SURFACE_CONTACT_TOL:.3f}m); min observed "
            f"{up}={min_obs_z:.3f}m — fluid is LEAKING"
        )

    return PredicateResult(
        predicate={
            "kind": "fluid_containment",
            "subject": fluid_name,
            "predicate": "contained_in",
            "object": container_name,
        },
        passed=passed,
        reason=narrative,
        measured_values={
            "n_particles": len(last_frame),
            "n_escaped": n_escaped,
            f"min_obs_{up}": min_obs_z,
            f"floor_{up}": cmin_z + SURFACE_CONTACT_TOL,
        },
        kind="fluid_containment",
    )


# ---------------------------------------------------------------------------
# Auto-discovery: turn plan + scene_objects into hard-override predicate runs
# ---------------------------------------------------------------------------


def _discover_wheel_landing_pairs(
    bodies: Dict[str, Dict[str, float]],
    physical_predicates: List[Dict[str, Any]],
    scene_objects: List[Dict[str, Any]],
    relevant_bodies: Optional[Set[str]],
) -> List[Tuple[str, str]]:
    """Return list of (wheel_name, platform_name) pairs to validate.

    Triggered when:
      - the plan has a ``rests_on(<vehicle_or_chassis>, <platform>)`` predicate, AND
      - the placement CSV contains body names matching ``_WHEEL_NAME_PATTERN``.

    Each wheel found in the placement CSV is paired with the platform
    referenced in the predicate. Falsely-named bodies (e.g. a body called
    ``wheel_decoration`` on the bridge) are filtered later by AABB
    plausibility — only wheels whose top is at most 1m above the platform
    top are checked.
    """
    pairs: List[Tuple[str, str]] = []
    wheel_names = [n for n in bodies if _WHEEL_NAME_PATTERN.search(n)]
    if not wheel_names:
        return pairs

    relevant_lower = {b.lower() for b in (relevant_bodies or [])}

    # Find platforms — bodies referenced as the "object" of a rests_on
    # predicate, OR scene_objects with role *_support_platform.
    platform_names: Set[str] = set()
    for pred in physical_predicates:
        if pred.get("predicate") == "rests_on":
            obj_name = str(pred.get("object", "")).strip()
            if obj_name and (not relevant_lower or obj_name.lower() in relevant_lower):
                platform_names.add(obj_name)
    for obj in scene_objects or []:
        role = str(obj.get("role") or "").lower()
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        if "support_platform" in role or "platform" == role:
            if not relevant_lower or name.lower() in relevant_lower:
                platform_names.add(name)

    if not platform_names:
        return pairs

    # Pair each wheel with the closest platform (smallest |wheel_bottom -
    # platform_top|) so the validator reports against the ONE platform the
    # wheel was supposed to land on, even when a plan has left+right.
    for wheel_name in wheel_names:
        wheel = _find_body(bodies, wheel_name)
        if wheel is None:
            continue
        wheel_bottom = _get_aabb_min(wheel, "z")
        best_platform: Optional[str] = None
        best_dist = float("inf")
        for plat_name in platform_names:
            plat = _find_body(bodies, plat_name)
            if plat is None:
                continue
            plat_top = _get_aabb_max(plat, "z")
            d = abs(wheel_bottom - plat_top)
            if d < best_dist:
                best_dist = d
                best_platform = plat_name
        if best_platform is not None and best_dist < 1.0:
            pairs.append((wheel_name, best_platform))
    return pairs


def _discover_interpenetration_pairs(
    bodies: Dict[str, Dict[str, float]],
    scene_objects: List[Dict[str, Any]],
    dynamic_bodies: Iterable[str],
    relevant_bodies: Optional[Set[str]],
    linked_pairs: Optional[Set[frozenset]] = None,
) -> List[Tuple[str, str]]:
    """Auto-enumerate body pairs that must not interpenetrate.

    Excluded by ground truth (so the predicate stays general for both
    vehicles and articulated robots):
      * ``linked_pairs`` — ChLink-connected pairs read from
        ``scene_links.csv``. Hip↔thigh, thigh↔calf, chassis↔suspension
        arm etc. all overlap at the joint by design; the ChLink
        constraint guarantees this. Without this exclusion every
        articulated robot reports false-positive clipping on every
        adjacent-link pair.
      * ``role=fluid_domain`` ↔ ``role=fsi_container`` — fluid sharing
        volume with its container is FSI design.
      * ``role=floating_*`` ↔ ``role=fluid_domain`` — bodies floating in
        the fluid share volume with it.

    Pair selection: at least one side must be dynamic (``dynamic_bodies``
    substring in name) — otherwise the enumeration explodes on fixed×fixed
    combinations with no diagnostic value.
    """
    role_by_name: Dict[str, str] = {}
    for obj in scene_objects or []:
        n = str(obj.get("name") or "").strip().lower()
        if n:
            role_by_name[n] = str(obj.get("role") or "").lower()

    linked_pairs = linked_pairs or set()

    def expected_overlap(a_name: str, b_name: str) -> bool:
        a_lower, b_lower = a_name.lower(), b_name.lower()

        # Joint-connected via ChLink → AABB overlap by design.
        if frozenset({a_lower, b_lower}) in linked_pairs:
            return True

        ra = role_by_name.get(a_lower, "")
        rb = role_by_name.get(b_lower, "")
        # fluid in container: expected
        if ("fluid_domain" in ra and "fsi_container" in rb) or (
            "fluid_domain" in rb and "fsi_container" in ra
        ):
            return True
        # floating object in fluid: expected
        if ("floating_" in ra and "fluid_domain" in rb) or (
            "floating_" in rb and "fluid_domain" in ra
        ):
            return True
        # wheel/spindle in contact with platform on a wheel-landing predicate
        # is handled by _check_wheel_landing; AABB overlap of <8 cm³ is
        # tolerated by INTERPENETRATION_VOL_TOL anyway.
        return False

    relevant_lower = {b.lower() for b in (relevant_bodies or [])}
    dyn_lower = {str(d).lower() for d in (dynamic_bodies or [])}

    pairs: List[Tuple[str, str]] = []
    body_names = list(bodies.keys())
    for i, a in enumerate(body_names):
        a_lower = a.lower()
        if relevant_lower and a_lower not in relevant_lower:
            continue
        # Only emit pairs where AT LEAST ONE side is dynamic — fixed × fixed
        # checks would explode combinatorially with no diagnostic value.
        a_is_dyn = any(d in a_lower for d in dyn_lower) if dyn_lower else False
        for b in body_names[i + 1:]:
            b_lower = b.lower()
            if relevant_lower and b_lower not in relevant_lower:
                continue
            b_is_dyn = any(d in b_lower for d in dyn_lower) if dyn_lower else False
            if not (a_is_dyn or b_is_dyn):
                continue
            if expected_overlap(a, b):
                continue
            pairs.append((a, b))
    return pairs


def _discover_fluid_containment(
    scene_objects: List[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    """Return ``(fluid_name, container_name)`` from scene_objects roles.

    Triggered when the plan declares both a ``role=fluid_domain`` and a
    body whose role contains ``fsi_container``.
    """
    fluid: Optional[str] = None
    container: Optional[str] = None
    for obj in scene_objects or []:
        role = str(obj.get("role") or "").lower()
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        if "fluid_domain" in role and fluid is None:
            fluid = name
        elif "fsi_container" in role and container is None:
            container = name
    if fluid and container:
        return (fluid, container)
    return None


# --- scene_predicates (spatial relations) ----------------------------------

def _check_on_top_of(
    subject: Dict[str, float],
    obj: Dict[str, float],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    result = _check_rests_on(subject, obj, [], subject_name, obj_name, gravity_axis)
    result.predicate["predicate"] = "on_top_of"
    return result


def _check_directional(
    subject: Dict[str, float],
    obj: Dict[str, float],
    subject_name: str,
    obj_name: str,
    relation: str,
    gravity_axis: str,
) -> PredicateResult:
    """Check left_of / right_of / in_front_of / behind.

    Convention (default -z gravity, xz working plane):
      +x = right, -x = left, +y = behind, -y = in_front_of
    Adjust if gravity axis differs.
    """
    h_axes = _horizontal_axes(gravity_axis)
    if len(h_axes) < 2:
        return PredicateResult(
            predicate={"subject": subject_name, "relation": relation, "object": obj_name},
            passed=False, reason="cannot determine horizontal axes",
        )
    ax_lr, ax_fb = h_axes[0], h_axes[1]  # first = left/right, second = front/back

    mapping = {
        "left_of":      (ax_lr, -1),
        "right_of":     (ax_lr, +1),
        "in_front_of":  (ax_fb, -1),
        "behind":       (ax_fb, +1),
    }
    ax, sign = mapping.get(relation, (ax_lr, 0))
    diff = _get_pos(subject, ax) - _get_pos(obj, ax)
    passed = (diff * sign) > 0
    return PredicateResult(
        predicate={"subject": subject_name, "relation": relation, "object": obj_name},
        passed=passed,
        reason=f"diff_{ax}={diff:.3f}m (expected sign={'+' if sign > 0 else '-'})",
        measured_values={"axis": ax, "diff": diff},
    )


def _check_next_to(
    subject: Dict[str, float],
    obj: Dict[str, float],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """Horizontal AABB edge distance is small."""
    h_axes = _horizontal_axes(gravity_axis)
    min_gap = float("inf")
    for ax in h_axes:
        gap = max(
            _get_aabb_min(subject, ax) - _get_aabb_max(obj, ax),
            _get_aabb_min(obj, ax) - _get_aabb_max(subject, ax),
            0.0,
        )
        min_gap = min(min_gap, gap)
    passed = min_gap < PROXIMITY_TOL
    return PredicateResult(
        predicate={"subject": subject_name, "relation": "next_to", "object": obj_name},
        passed=passed,
        reason=f"min_gap={min_gap:.3f}m (tol={PROXIMITY_TOL})",
        measured_values={"min_gap": min_gap},
    )


def _check_centered_on(
    subject: Dict[str, float],
    obj: Dict[str, float],
    subject_name: str,
    obj_name: str,
    gravity_axis: str,
) -> PredicateResult:
    """Subject centre ≈ object centre on horizontal plane."""
    h_axes = _horizontal_axes(gravity_axis)
    sq_sum = sum((_get_pos(subject, ax) - _get_pos(obj, ax)) ** 2 for ax in h_axes)
    dist = math.sqrt(sq_sum)
    passed = dist < CENTERING_TOL
    return PredicateResult(
        predicate={"subject": subject_name, "relation": "centered_on", "object": obj_name},
        passed=passed,
        reason=f"horizontal_offset={dist:.3f}m (tol={CENTERING_TOL})",
        measured_values={"horizontal_offset": dist},
    )


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

# ChBody.GetTotalAABB() returns +FLT_MAX / -FLT_MAX when the body has
# no collision shape registered. Any aabb_min coordinate above this
# threshold (or aabb_max below the negative threshold) means "no
# collision shape" — a strong, unambiguous signal that someone built
# a visual-only body that was meant to act as a support surface.
_AABB_SENTINEL_THRESHOLD = 1e30


def _has_no_collision_shape(row: Dict[str, float]) -> bool:
    """True iff the body's AABB is the sentinel pair returned for empty
    collision models (``±FLT_MAX``)."""
    return (
        row.get("aabb_min_x", 0.0) > _AABB_SENTINEL_THRESHOLD
        or row.get("aabb_min_y", 0.0) > _AABB_SENTINEL_THRESHOLD
        or row.get("aabb_min_z", 0.0) > _AABB_SENTINEL_THRESHOLD
    )


# Velocity threshold for "still falling" — much higher than VELOCITY_TOL
# (which catches normal jitter); this one is meant to catch free-fall
# cases like vz ≈ -10 m/s.
_FREEFALL_VEL_TOL = 1.0


def _check_support_surface_collision(
    bodies: Dict[str, Dict[str, float]],
    contacts: List[Dict[str, Any]],
    dynamic_exclusions: Iterable[str] = (),
) -> List[PredicateResult]:
    """Catch the "visual-only floor / falling furniture" failure class.

    Emits a hard-override predicate when:

    1. A *named* body whose name suggests it's a support surface (floor,
       wall, ground, platform, table, ...) has no collision shape
       registered — its AABB is the ``±FLT_MAX`` sentinel pair. This is
       a 100%-deterministic codegen bug: someone built it with
       ``ChVisualShapeBox`` but skipped ``AddCollisionShape`` /
       ``EnableCollision(True)``.

    2. A non-fixed, non-robot body has |vel| above the free-fall
       threshold AND zero contacts with any other body in the scene.
       Such a body is in free fall — there is no physical surface
       supporting it. Either the support surface lacks collision (case
       1 above) or the body's own collision shape is missing.

    Both findings produce a ``PredicateResult`` with
    ``kind="support_surface_collision"`` so the workflow's
    ``step_review_node`` can short-circuit them past the LLM verdict.
    """
    results: List[PredicateResult] = []

    _SUPPORT_NAME_TOKENS = (
        "floor", "ground", "wall", "platform", "support",
        "tabletop", "desk_top", "ceiling", "ramp", "terrain",
    )

    # Case 1: support-surface body with no collision shape.
    for name, row in bodies.items():
        nlow = name.lower()
        if not any(tok in nlow for tok in _SUPPORT_NAME_TOKENS):
            continue
        # Cameras and other helper bodies legitimately have no collision.
        if name.startswith("cam_") or "_camera" in nlow:
            continue
        if _has_no_collision_shape(row):
            results.append(PredicateResult(
                predicate={
                    "kind": "support_surface_collision",
                    "subject": name,
                    "predicate": "has_collision_shape",
                    "object": name,
                },
                passed=False,
                reason=(
                    f"SUPPORT_SURFACE: '{name}' has no collision shape "
                    f"(AABB = ±FLT_MAX). Likely built as ChBody() + "
                    f"ChVisualShapeBox without AddCollisionShape + "
                    f"EnableCollision(True). Furniture placed on this "
                    f"body will free-fall through it. Fix: add "
                    f"ChCollisionShapeBox(material, sx, sy, sz) and "
                    f"EnableCollision(True) — see scene SKILL §1.5b."
                ),
                kind="support_surface_collision",
            ))

    # Case 2: dynamic non-robot body free-falling with zero contacts.
    static_bodies, dynamic_link_names = split_dynamic_static(
        bodies, dynamic_exclusions
    )
    # Index contacts by body for quick lookup (case-insensitive).
    contact_partners: Dict[str, Set[str]] = {}
    for c in contacts:
        b1 = (c.get("body1") or "").strip().lower()
        b2 = (c.get("body2") or "").strip().lower()
        if not b1 or not b2:
            continue
        contact_partners.setdefault(b1, set()).add(b2)
        contact_partners.setdefault(b2, set()).add(b1)

    for name, row in static_bodies.items():
        # We only care about bodies that are visibly moving — the
        # standard stability check covers those that ARE moving but
        # don't tell us why. Free-fall has a very high downward velocity
        # signature; pure jitter does not.
        vz = row.get("vel_z", 0.0)
        vx = row.get("vel_x", 0.0)
        vy = row.get("vel_y", 0.0)
        speed_sq = vx * vx + vy * vy + vz * vz
        if speed_sq < _FREEFALL_VEL_TOL * _FREEFALL_VEL_TOL:
            continue
        # Bodies that ARE in contact with something are not free-falling
        # (they may be sliding / settling, but that's a different bug
        # class and the stability check already handles it).
        if contact_partners.get(name.lower()):
            continue
        results.append(PredicateResult(
            predicate={
                "kind": "support_surface_collision",
                "subject": name,
                "predicate": "rests_on",
                "object": "<any_fixed_support>",
            },
            passed=False,
            reason=(
                f"SUPPORT_SURFACE: '{name}' is free-falling at sim end "
                f"(vel=({vx:.2f}, {vy:.2f}, {vz:.2f})) with zero "
                f"contacts. The body has no support surface beneath it "
                f"— most often the floor was built visual-only "
                f"(missing AddCollisionShape + EnableCollision). "
                f"Check fixed bodies' AABBs in scene_placement.csv: "
                f"any AABB at ±FLT_MAX is the offender."
            ),
            kind="support_surface_collision",
        ))

    return results


def _check_stability(
    bodies: Dict[str, Dict[str, float]],
    dynamic_exclusions: Iterable[str] = (),
) -> tuple[bool, str]:
    """All *static* bodies must have near-zero velocity after settling.

    Bodies whose names match ``dynamic_exclusions`` (asset roots such as
    ``"go2"``) or the shared robot-link patterns are skipped — they are
    expected to move and would otherwise spam failures on navigation steps.
    """
    static_bodies, skipped = split_dynamic_static(bodies, dynamic_exclusions)
    unstable: list[str] = []
    for name, row in static_bodies.items():
        vx = row.get("vel_x", 0)
        vy = row.get("vel_y", 0)
        vz = row.get("vel_z", 0)
        lin_speed = math.sqrt(vx**2 + vy**2 + vz**2)
        ax = row.get("ang_vel_x", 0)
        ay = row.get("ang_vel_y", 0)
        az = row.get("ang_vel_z", 0)
        ang_speed = math.sqrt(ax**2 + ay**2 + az**2)
        if lin_speed > VELOCITY_TOL or ang_speed > ANG_VELOCITY_TOL:
            unstable.append(f"{name}(v={lin_speed:.3f}, w={ang_speed:.3f})")
    passed = len(unstable) == 0
    skipped_note = f" (skipped {len(skipped)} dynamic)" if skipped else ""
    detail = (
        f"all bodies settled{skipped_note}"
        if passed
        else f"unstable{skipped_note}: {', '.join(unstable)}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PHYSICAL_PRED_DISPATCH = {
    "rests_on":      lambda s, o, c, sn, on, ga: _check_rests_on(s, o, c, sn, on, ga),
    "supports":      lambda s, o, c, sn, on, ga: _check_supports(s, o, c, sn, on, ga),
    "contains":      lambda s, o, c, sn, on, ga: _check_contains(s, o, sn, on),
    "leans_against": lambda s, o, c, sn, on, ga: _check_leans_against(s, o, sn, on, ga),
    "attached_to":   lambda s, o, c, sn, on, ga: _check_attached_to(s, o, c, sn, on),
}

_SCENE_REL_DISPATCH = {
    "on_top_of":    lambda s, o, sn, on, ga: _check_on_top_of(s, o, sn, on, ga),
    "left_of":      lambda s, o, sn, on, ga: _check_directional(s, o, sn, on, "left_of", ga),
    "right_of":     lambda s, o, sn, on, ga: _check_directional(s, o, sn, on, "right_of", ga),
    "in_front_of":  lambda s, o, sn, on, ga: _check_directional(s, o, sn, on, "in_front_of", ga),
    "behind":       lambda s, o, sn, on, ga: _check_directional(s, o, sn, on, "behind", ga),
    "next_to":      lambda s, o, sn, on, ga: _check_next_to(s, o, sn, on, ga),
    "centered_on":  lambda s, o, sn, on, ga: _check_centered_on(s, o, sn, on, ga),
}


def _find_body(bodies: Dict[str, Dict[str, float]], name: str) -> Optional[Dict[str, float]]:
    """Case-insensitive body lookup."""
    low = name.lower()
    for k, v in bodies.items():
        if k.lower() == low:
            return v
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate_scene_placement(
    placement_csv_path: str,
    contacts_csv_path: Optional[str],
    physical_predicates: List[Dict[str, Any]],
    scene_predicates: List[Dict[str, Any]],
    gravity_axis: str = "-z",
    relevant_bodies: Optional[Set[str]] = None,
    dynamic_bodies: Optional[Iterable[str]] = None,
    scene_objects: Optional[List[Dict[str, Any]]] = None,
    particles_csv_path: Optional[str] = None,
    plan_type: Optional[str] = None,
    links_csv_path: Optional[str] = None,
) -> SceneValidationResult:
    """Run deterministic validation of scene placement.

    Args:
        placement_csv_path: Path to ``scene_placement.csv``.
        contacts_csv_path: Path to ``scene_contacts.csv`` (may be None).
        physical_predicates: From ``plan.topology.physical_predicates``.
        scene_predicates: From ``plan.topology.scene_predicates``.
        gravity_axis: e.g. ``"-z"`` (default).
        relevant_bodies: If set, only predicates involving these body names
            are checked (used for per-step validation).
        dynamic_bodies: Asset roots (e.g. ``["go2"]``) whose links should be
            skipped by the stability check. Robot-link name patterns are
            also always skipped via ``body_classification``.
        scene_objects: ``plan.scene_objects[]`` raw dicts. Used to drive
            auto-discovery of hard-override predicates: wheel-landing pairs
            (matched by name pattern + ``role=*support_platform``), no-
            interpenetration pairs (every dynamic × fixed body pair after
            FSI fluid/floating exclusions), and fluid_containment (paired
            ``fluid_domain`` ↔ ``fsi_container`` roles).
        particles_csv_path: Optional ``particles.csv`` produced by
            ``sysSPH.GetParticlePositions()`` at sim end (HR-16). Required
            for the fluid_containment check; missing file is reported as
            ``FLUID_CONTAINMENT: SKIPPED``.
        plan_type: ``plan.plan_type`` so FSI plans can flag missing
            ``particles.csv`` as a codegen bug rather than silently passing.
        links_csv_path: Optional ``scene_links.csv`` written by
            ``write_links_csv``. Each row records a ChLink-connected
            body pair. Used to suppress false-positive interpenetration
            on adjacent links of articulated robots / vehicles where
            the AABB overlap at the joint is a kinematic-constraint
            artifact, not a real geometry bug. Missing file → no
            exclusion (current behaviour); validator still works but
            articulated robots may report spurious clipping.

    Returns:
        SceneValidationResult with verdict and per-predicate results.
    """
    bodies = _load_placement_csv(placement_csv_path)
    contacts = _load_contacts_csv(contacts_csv_path) if contacts_csv_path else []
    particles: List[Tuple[float, float, float, float]] = []
    if particles_csv_path:
        try:
            particles = _load_particles_csv(particles_csv_path)
        except OSError as exc:
            logger.warning("particles.csv unreadable: %s", exc)
    linked_pairs: Set[frozenset] = set()
    if links_csv_path:
        linked_pairs = _load_links_csv(links_csv_path)
    if not linked_pairs:
        logger.info(
            "[validate_scene_placement] no scene_links.csv consumed — "
            "articulated systems may report spurious AABB overlap on "
            "adjacent joint pairs. Codegen should call "
            "write_links_csv(system, output_dir) at sim end."
        )

    if not bodies:
        return SceneValidationResult(
            verdict="physics_invalid",
            summary="scene_placement.csv is empty — no bodies found",
        )

    results: List[PredicateResult] = []

    def _is_relevant(name: str) -> bool:
        if relevant_bodies is None:
            return True
        return name.lower() in {b.lower() for b in relevant_bodies}

    # --- physical predicates ---
    for pred in physical_predicates:
        subj_name = pred.get("subject", "")
        obj_name = pred.get("object", "")
        pred_type = pred.get("predicate", "")

        if not (_is_relevant(subj_name) and _is_relevant(obj_name)):
            continue

        subj = _find_body(bodies, subj_name)
        obj = _find_body(bodies, obj_name)
        if subj is None or obj is None:
            results.append(PredicateResult(
                predicate=pred, passed=False,
                reason=f"body not found: {subj_name if subj is None else obj_name}",
            ))
            continue

        handler = _PHYSICAL_PRED_DISPATCH.get(pred_type)
        if handler:
            results.append(handler(subj, obj, contacts, subj_name, obj_name, gravity_axis))
        else:
            logger.warning("Unknown physical predicate type: %s", pred_type)

    # --- scene predicates (spatial relations) ---
    for pred in scene_predicates:
        subj_name = pred.get("subject", "")
        obj_name = pred.get("object", "")
        relation = pred.get("relation", "")

        if not (_is_relevant(subj_name) and _is_relevant(obj_name)):
            continue

        subj = _find_body(bodies, subj_name)
        obj = _find_body(bodies, obj_name)
        if subj is None or obj is None:
            results.append(PredicateResult(
                predicate=pred, passed=False,
                reason=f"body not found: {subj_name if subj is None else obj_name}",
            ))
            continue

        handler = _SCENE_REL_DISPATCH.get(relation)
        if handler:
            results.append(handler(subj, obj, subj_name, obj_name, gravity_axis))
        else:
            logger.warning("Unknown scene relation: %s", relation)

    # --- hard-override predicates (auto-discovered, no plan vocabulary
    # change required). Findings here outrank LLM judgment in the workflow's
    # step_review_node short-circuit (see ``predicate.kind`` field).
    scene_objects = scene_objects or []

    # (a) Per-wheel landing — ground-truth wheel z vs platform top.
    wheel_pairs = _discover_wheel_landing_pairs(
        bodies, physical_predicates, scene_objects, relevant_bodies
    )
    for wheel_name, plat_name in wheel_pairs:
        wheel = _find_body(bodies, wheel_name)
        plat = _find_body(bodies, plat_name)
        if wheel is None or plat is None:
            continue
        results.append(_check_wheel_landing(
            wheel, plat, contacts, wheel_name, plat_name, gravity_axis,
        ))

    # (b) Pairwise AABB no-interpenetration on dynamic × fixed pairs.
    # ``linked_pairs`` (read from scene_links.csv) suppresses joint-
    # connected adjacent links — without this, articulated robots flood
    # the report with hip↔thigh, thigh↔calf etc. false positives.
    overlap_pairs = _discover_interpenetration_pairs(
        bodies, scene_objects, dynamic_bodies or [], relevant_bodies,
        linked_pairs=linked_pairs,
    )
    # Cap pairwise enumeration to keep summaries short — 60 is plenty for
    # a vehicle scene (chassis + wheels + suspension links × a few static
    # bodies). If we exceed this, scene authoring is the bug, not validation.
    for a_name, b_name in overlap_pairs[:60]:
        a_body = _find_body(bodies, a_name)
        b_body = _find_body(bodies, b_name)
        if a_body is None or b_body is None:
            continue
        results.append(_check_no_interpenetration(a_body, b_body, a_name, b_name))

    # (c) Fluid containment — last-frame particle z vs container floor.
    fluid_pair = _discover_fluid_containment(scene_objects)
    is_fsi = (plan_type or "").lower().startswith("fsi") or (
        "fsi" in (plan_type or "").lower()
    )
    if fluid_pair is not None:
        fluid_name, container_name = fluid_pair
        container = _find_body(bodies, container_name)
        if container is None:
            results.append(PredicateResult(
                predicate={
                    "kind": "fluid_containment",
                    "subject": fluid_name,
                    "predicate": "contained_in",
                    "object": container_name,
                },
                passed=False,
                reason=(
                    f"FLUID_CONTAINMENT: SKIPPED — container body "
                    f"'{container_name}' not in scene_placement.csv"
                ),
                kind="fluid_containment",
            ))
        else:
            results.append(_check_fluid_containment(
                container, container_name, fluid_name, particles, gravity_axis,
            ))
    elif is_fsi and not particles:
        # FSI plan but no particle data and no obvious fluid/container pair —
        # surface as a SKIPPED finding so codegen sees the missing data.
        results.append(PredicateResult(
            predicate={
                "kind": "fluid_containment",
                "subject": "sph_water",
                "predicate": "contained_in",
                "object": "fsi_container",
            },
            passed=False,
            reason=(
                "FLUID_CONTAINMENT: SKIPPED — FSI plan ran but no "
                "particles.csv produced. Codegen must call "
                "sysSPH.GetParticlePositions() at end of sim and write "
                "particles.csv (HR-16) so leaks can be detected."
            ),
            kind="fluid_containment",
        ))

    # (d) Support-surface collision — catch visual-only floors/walls
    # and free-falling furniture before the stability check or LLM
    # review tries to interpret the symptom.
    results.extend(_check_support_surface_collision(
        bodies, contacts, dynamic_exclusions=tuple(dynamic_bodies or ()),
    ))

    # --- stability ---
    stability_passed, stability_detail = _check_stability(
        bodies, dynamic_exclusions=tuple(dynamic_bodies or ())
    )

    # --- verdict ---
    failed = [r for r in results if not r.passed]
    if failed or not stability_passed:
        fail_msgs = [f"{r.predicate}: {r.reason}" for r in failed]
        if not stability_passed:
            fail_msgs.append(f"stability: {stability_detail}")
        verdict = "physics_invalid"
        summary = f"{len(failed)} predicate(s) failed, stability={'pass' if stability_passed else 'fail'}. " + "; ".join(fail_msgs)
    else:
        verdict = "physics_valid"
        summary = f"All {len(results)} predicate(s) passed, stability OK."

    return SceneValidationResult(
        verdict=verdict,
        predicate_results=results,
        stability_passed=stability_passed,
        stability_detail=stability_detail,
        summary=summary,
    )
