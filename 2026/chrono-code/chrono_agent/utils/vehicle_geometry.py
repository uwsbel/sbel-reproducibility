"""Read PyChrono catalog vehicle geometry from the shipped JSON files.

The two values codegen and planner need at runtime:

* ``chassis_init_z(vehicle_json, support_top_z)`` — the world-z to pass
  into ``vehicle.Initialize(ChCoordsysd(...))`` so the wheel bottoms
  rest on a flat support of given top height. Default ``support_top_z=0``
  gives the world-frame init height for ground-level spawning.

* ``wheelbase(vehicle_json)`` — front-to-rear axle distance in chassis
  x. Used by the planner to gate ``SUPPORTED-BY`` against platform
  x-extent.

Both pull every number from the shipped JSON tree (``Polaris.json`` →
``Polaris_Front_DoubleWishbone.json`` → ``Polaris_RigidTire.json`` →
``Polaris_Wheel.json``). No per-vehicle constants live in code or skill
text — when ProjectChrono ships an updated vehicle JSON the helper just
re-reads it.

Layout fact (resolved empirically against Polaris + HMMWV + Sedan):

* The PyChrono main vehicle JSON has an ``Axles`` array; each entry has
  ``Suspension Location: [x, y, z]`` in chassis-frame coordinates and a
  ``Suspension Input File`` pointing to a separate suspension JSON.
* For some vehicles (HMMWV, Sedan) the axle x-offset lives directly in
  ``Suspension Location[0]`` (e.g. front=[+1.69, 0, 0], rear=[-1.69, 0, 0]).
* For others (Polaris) ``Suspension Location[0]`` is 0 and the actual
  axle x comes from ``Spindle.COM[0]`` inside the suspension JSON
  (e.g. rear ``Spindle.COM = [-2.71526, 0.616, 0.33504]``).
* Tire radius lives in the per-axle ``Left Tire Input File`` JSON's
  top-level ``Radius`` field (and is the OUTER radius — wheel-bottom
  contact distance from the spindle axis).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple


# PyChrono ships JSONs that include C++-style ``//`` comments and the
# occasional trailing comma — both legal in rapidjson (which the C++
# loader uses) but rejected by Python's strict ``json`` module. Strip
# them before parsing. Done as text-level regexes rather than a real
# JSON5 dep because the patterns we hit in the catalog are simple.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _load_lenient_json(path: Path) -> dict:
    text = path.read_text()
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON {path}: {exc}") from exc


def _resolve_under_chrono_data(rel_path: str) -> Path:
    """Resolve a relative JSON reference against the Chrono vehicle data root.

    PyChrono JSONs reference siblings via paths like
    ``"Polaris/Polaris_Front_DoubleWishbone.json"``; that path is
    relative to the ``share/chrono/data/vehicle/`` root. Use the helper
    that pychrono itself uses to find that root.
    """
    try:
        import pychrono.vehicle as veh  # type: ignore

        return Path(veh.GetVehicleDataFile(rel_path))
    except Exception:
        # Fallback: search common conda-installed location
        candidates = [
            Path.home() / "anaconda3/envs/chrono-agent/share/chrono/data/vehicle" / rel_path,
            Path.home() / "miniconda3/envs/chrono-agent/share/chrono/data/vehicle" / rel_path,
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Cannot locate vehicle JSON: {rel_path}")


def _load_json(path: Path) -> dict:
    return _load_lenient_json(path)


def _spindle_x_in_chassis_frame(axle: dict) -> float:
    """Return the front (or rear) axle's x-coordinate in chassis frame.

    First try ``Suspension Location[0]`` directly (HMMWV / Sedan layout).
    If that is ~0, drill into the suspension JSON's ``Spindle.COM[0]``
    (Polaris layout).
    """
    susp_loc = axle.get("Suspension Location") or [0, 0, 0]
    susp_x = float(susp_loc[0])
    if abs(susp_x) > 0.01:
        return susp_x

    susp_rel = axle.get("Suspension Input File")
    if not susp_rel:
        return susp_x
    susp_path = _resolve_under_chrono_data(susp_rel)
    susp = _load_json(susp_path)
    com = susp.get("Spindle", {}).get("COM") or [0, 0, 0]
    return float(com[0])


def _spindle_y_in_chassis_frame(axle: dict) -> float:
    """Half-track distance for this axle (positive). Both left and right
    spindles sit at ``±_spindle_y_in_chassis_frame(...)`` in chassis frame.

    First check ``Suspension Location[1]`` (HMMWV-style mount-point offset).
    Most vehicles encode the actual track in the suspension JSON's
    ``Spindle.COM[1]`` (Polaris: 0.616). Combine the two: the absolute
    sum is the world-frame y of the right spindle.
    """
    susp_loc = axle.get("Suspension Location") or [0, 0, 0]
    susp_y = float(susp_loc[1])
    susp_rel = axle.get("Suspension Input File")
    if not susp_rel:
        return abs(susp_y)
    susp_path = _resolve_under_chrono_data(susp_rel)
    susp = _load_json(susp_path)
    com = susp.get("Spindle", {}).get("COM") or [0, 0, 0]
    return abs(susp_y) + abs(float(com[1]))


def _spindle_z_in_chassis_frame(axle: dict) -> float:
    """Return the axle's spindle z relative to chassis-frame origin.

    Combines ``Suspension Location[2]`` (typical HMMWV-style spindle
    mount height) with the suspension's internal ``Spindle.COM[2]``
    offset (often 0 for revolute spindles, non-zero for trailing-arm
    designs).
    """
    susp_loc = axle.get("Suspension Location") or [0, 0, 0]
    susp_z = float(susp_loc[2])
    susp_rel = axle.get("Suspension Input File")
    if not susp_rel:
        return susp_z
    susp_path = _resolve_under_chrono_data(susp_rel)
    susp = _load_json(susp_path)
    com = susp.get("Spindle", {}).get("COM") or [0, 0, 0]
    return susp_z + float(com[2])


def _tire_radius_from_explicit_json(tire_json: str | Path) -> float:
    """Read top-level ``Radius`` from a tire JSON the caller specifies."""
    path = (
        Path(tire_json)
        if isinstance(tire_json, (str, Path))
        and "/" in str(tire_json)
        and Path(tire_json).exists()
        else _resolve_under_chrono_data(str(tire_json))
    )
    tire = _load_json(path)
    radius = tire.get("Radius")
    if radius is None:
        raise ValueError(f"tire JSON {tire_json} has no top-level 'Radius' field")
    return float(radius)


def _tire_radius_default_search(vehicle_json_path: Path) -> float:
    """Best-effort tire-radius lookup when caller did not specify a tire.

    PyChrono ships tire JSONs separately from the main vehicle JSON
    (the user picks ``Polaris_RigidTire.json`` vs ``Polaris_Pac02Tire.json``
    etc. at runtime). When the caller has not specified one, search for
    a ``*RigidTire.json`` first (matches the FSI / chassis-init use
    case) under both the vehicle JSON's directory (Polaris layout) and
    a sibling ``tire/`` directory at the chrono-data root (HMMWV
    layout). Fall back to the wheel JSON's ``Visualization.Radius``
    (rim, slightly smaller than tire outer) only if no tire JSON is
    findable — and emit no warning here so callers can choose to be
    strict via the explicit ``tire_json=`` argument instead.
    """
    veh_dir = vehicle_json_path.parent
    candidates: list[Path] = list(veh_dir.glob("*RigidTire.json"))
    # HMMWV layout: tire/ as sibling of vehicle/
    if veh_dir.name == "vehicle":
        candidates.extend((veh_dir.parent / "tire").glob("*RigidTire.json"))
    for c in candidates:
        try:
            tire = _load_json(c)
            radius = tire.get("Radius")
            if radius is not None:
                return float(radius)
        except Exception:
            continue
    # Final fallback: wheel viz rim radius (under-estimates by tire wall).
    vehicle = _load_json(vehicle_json_path)
    front = vehicle.get("Axles", [{}])[0]
    wheel_rel = front.get("Left Wheel Input File")
    if wheel_rel:
        wheel = _load_json(_resolve_under_chrono_data(wheel_rel))
        radius = (wheel.get("Visualization") or {}).get("Radius")
        if radius is not None:
            return float(radius)
    raise ValueError(
        f"Cannot determine tire radius for {vehicle_json_path.name}; "
        f"pass tire_json= explicitly"
    )


def _front_axle(vehicle: dict) -> dict:
    axles = vehicle.get("Axles") or []
    if not axles:
        raise ValueError("vehicle JSON has no Axles array")
    return axles[0]


def _rear_axle(vehicle: dict) -> dict:
    axles = vehicle.get("Axles") or []
    if len(axles) < 2:
        raise ValueError("vehicle JSON has fewer than 2 axles")
    return axles[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_AABB_SENTINEL = 1e30  # ChBody.GetTotalAABB returns ±FLT_MAX when no collision shape


def support_top_z_from_body(body, system=None) -> float:
    """Return the world Z of the top face of a fixed support body — the
    single source of truth for ``support_top_z`` arguments downstream.

    Reads ``body.GetTotalAABB().max.z`` directly. The point is that no
    caller-side arithmetic ever happens — this is the only sanctioned way
    to derive a "support top Z" once the body has been positioned in the
    world. The recurring iter_006 bug
    (``PLT_TOP_Z = LEFT_PLT_Z`` where ``LEFT_PLT_Z`` is the platform body
    *center* Z, not the top) was a pure naming-vs-semantics confusion that
    is impossible to reproduce when the value comes from
    ``GetTotalAABB().max.z`` directly.

    ``system``: optional ``ChSystemNSC`` / ``ChSystemSMC`` reference that
    owns ``body``. When provided, the helper calls
    ``system.GetCollisionSystem().Initialize()`` before reading the AABB.
    This is REQUIRED at codegen time when the helper is called between
    ``AddBody`` and the first ``DoStepDynamics`` — Chrono populates
    per-shape AABBs lazily at the first broadphase, so before any
    dynamics step, ``GetTotalAABB()`` returns the ±FLT_MAX sentinel
    pair. Pass ``system=sysMBS`` (the Python wrapper, not
    ``body.GetSystem()`` which returns an unwrapped SwigPyObject)
    so the helper can trigger the broadphase setup. ``Initialize()`` is
    idempotent — re-calling on an already-initialized system is a no-op.

    Raises ``RuntimeError`` when the body has no collision shape registered
    (the AABB stays at the sentinel even after Initialize). Build the body
    with ``ChBodyEasyBox(..., collide=True, ...)`` or attach a
    ``ChCollisionShapeBox`` + ``EnableCollision(True)`` before calling.
    """
    def _is_sentinel(a) -> bool:
        return a.min.z > _AABB_SENTINEL or a.max.z < -_AABB_SENTINEL

    aabb = body.GetTotalAABB()

    if _is_sentinel(aabb) and system is not None:
        # Trigger one broadphase setup pass — populates per-shape AABBs
        # without running dynamics. Idempotent on already-initialized
        # systems, so cheap to re-call.
        try:
            system.GetCollisionSystem().Initialize()
        except Exception:
            pass
        aabb = body.GetTotalAABB()

    if _is_sentinel(aabb):
        try:
            name = body.GetName() or "<unnamed>"
        except Exception:
            name = "<unnamed>"
        if system is None:
            hint = (
                " Pass system=<your ChSystem> so the helper can call "
                "system.GetCollisionSystem().Initialize() — at codegen time "
                "(before any DoStepDynamics) the per-shape AABBs are "
                "populated lazily, so the helper needs the system handle "
                "to trigger one broadphase pass."
            )
        else:
            hint = (
                " Build the body with ChBodyEasyBox(..., collide=True, ...) "
                "or attach ChCollisionShapeBox + EnableCollision(True) "
                "before calling."
            )
        raise RuntimeError(
            f"support_top_z_from_body({name!r}): body has no collision shape "
            "(GetTotalAABB returned the ±FLT_MAX sentinel)." + hint
        )
    return float(aabb.max.z)


def chassis_init_z(
    vehicle_json: str | Path,
    support_top_z: float = 0.0,
    tire_json: str | Path | None = None,
) -> float:
    """Z to pass into ``vehicle.Initialize(ChCoordsysd(ChVector3d(x,y,z), q))``
    so the wheel bottoms rest on a flat support of top height
    ``support_top_z`` (defaults to 0 for ground-level spawning).

    Formula (chassis frame coords, using the FRONT axle):

        wheel_bottom_z_in_chassis  = front_spindle_z - tire_radius
        chassis_init_z = support_top_z - wheel_bottom_z_in_chassis
                       = support_top_z - front_spindle_z + tire_radius

    Both signs come from the chassis-frame convention: chassis origin is
    above the wheel contact line, spindle z is positive (mount height),
    tire radius drops the contact patch back toward the chassis x-y
    plane. This matches the iter_006 manual derivation in
    session_20260429_195705 (z=1.2 - 0.397 + 0.33 ≈ 1.13).
    """
    path = Path(vehicle_json) if isinstance(vehicle_json, (str, Path)) and "/" in str(vehicle_json) and Path(vehicle_json).exists() else _resolve_under_chrono_data(str(vehicle_json))
    vehicle = _load_json(path)
    front = _front_axle(vehicle)
    spindle_z = _spindle_z_in_chassis_frame(front)
    if tire_json is not None:
        tire_r = _tire_radius_from_explicit_json(tire_json)
    else:
        tire_r = _tire_radius_default_search(path)
    return float(support_top_z) - spindle_z + tire_r


def wheelbase(vehicle_json: str | Path) -> float:
    """Distance between front and rear axle x-positions in chassis frame.

    Used by the planner to verify ``SUPPORTED-BY`` feasibility:
    a vehicle with wheelbase L cannot rest stably on a platform of
    x-extent < L (rear/front axle off the back/front edge — the
    iter_006 "rear suspended off platform" symptom in
    session_20260429_195705 where Polaris wheelbase 2.715 m was
    placed on a platform 1.0 m wide).
    """
    path = Path(vehicle_json) if isinstance(vehicle_json, (str, Path)) and "/" in str(vehicle_json) and Path(vehicle_json).exists() else _resolve_under_chrono_data(str(vehicle_json))
    vehicle = _load_json(path)
    front = _front_axle(vehicle)
    rear = _rear_axle(vehicle)
    fx = _spindle_x_in_chassis_frame(front)
    rx = _spindle_x_in_chassis_frame(rear)
    return abs(fx - rx)


def vehicle_geometry(
    vehicle_json: str | Path,
    tire_json: str | Path | None = None,
) -> Tuple[float, float, float]:
    """Convenience: return ``(wheelbase, front_spindle_z, tire_radius)``.

    Useful when codegen wants to log all three at once and run an AABB
    consistency check after ``vehicle.Initialize``. ``tire_json`` is the
    same explicit-tire override as ``chassis_init_z`` accepts.
    """
    path = Path(vehicle_json) if isinstance(vehicle_json, (str, Path)) and "/" in str(vehicle_json) and Path(vehicle_json).exists() else _resolve_under_chrono_data(str(vehicle_json))
    vehicle = _load_json(path)
    front = _front_axle(vehicle)
    rear = _rear_axle(vehicle)
    if tire_json is not None:
        tire_r = _tire_radius_from_explicit_json(tire_json)
    else:
        tire_r = _tire_radius_default_search(path)
    return (
        abs(_spindle_x_in_chassis_frame(front) - _spindle_x_in_chassis_frame(rear)),
        _spindle_z_in_chassis_frame(front),
        tire_r,
    )


# ---------------------------------------------------------------------------
# Post-Initialize footprint validation
# ---------------------------------------------------------------------------
#
# Background. The chassis-frame origin is *not* always the geometric center
# of the vehicle. For Polaris ``Spindle.COM[0]`` puts the front axle at
# x=0 and the rear axle at x=-2.7153 in chassis frame, so calling
# ``polaris.Initialize(ChCoordsysd(ChVector3d(-4, 0, z)))`` lands the front
# axle at world x=-4 and the rear axle at world x=-6.72 — 0.72 m past the
# left edge of a [-6, -2] platform. Treating ``placed_on_top of platform``
# as ``vehicle_x = platform_center_x`` is therefore wrong for Polaris and
# right for HMMWV / Sedan, and codegen has no way to know which without
# reading the JSON.
#
# These helpers compute the real post-``Initialize`` world AABB (chassis
# union spindles) and assert it fits the named support, with a message
# that tells the next codegen iteration *exactly* how many meters to shift
# ``VEH_INIT_X``. session_20260503_145628 spent 30+ codegen turns expanding
# the FSI computational domain to chase rear-wheel BCE markers that were
# off the back of the platform — an assert at Initialize time catches this
# in the first iteration.


def vehicle_world_aabb(
    vehicle,
    vehicle_json: str | Path,
    tire_json: str | Path | None = None,
) -> Tuple[float, float, float, float, float, float]:
    """Return the world-frame footprint AABB of a wheeled vehicle as
    ``(xmin, xmax, ymin, ymax, zmin, zmax)``, computed from the chassis
    world pose + JSON-derived suspension/tire geometry.

    Why JSON instead of ``GetTotalAABB()``: the chassis and spindle
    ``ChBody`` instances do not have collision shapes registered by
    ``vehicle.Initialize(...)`` alone. ``GetTotalAABB()`` therefore returns
    ``[+DBL_MAX, -DBL_MAX]`` (the "no collision shape" sentinel) and is
    useless as a footprint check. The JSON files always contain the axle
    spindle locations and tire radius, so we read them and transform the
    four wheel-envelope corners through the chassis world pose.

    Returns the union AABB of the four wheels' world-space envelopes.
    Chassis-only collision is intentionally not added — for every shipped
    PyChrono catalog vehicle the wheels bound the footprint along x and
    y, so wheel-only is a tight upper bound on what fits on a platform.

    Must be called after ``vehicle.Initialize(...)``. Handles arbitrary
    initial orientation via ``ChassisBody.TransformPointLocalToParent``.
    """
    path = (
        Path(vehicle_json)
        if isinstance(vehicle_json, (str, Path))
        and "/" in str(vehicle_json)
        and Path(vehicle_json).exists()
        else _resolve_under_chrono_data(str(vehicle_json))
    )
    veh_dict = _load_json(path)

    if tire_json is not None:
        tire_r = _tire_radius_from_explicit_json(tire_json)
    else:
        tire_r = _tire_radius_default_search(path)

    # Suspension Locations and Spindle.COM in the JSON are expressed in
    # the chassis REFERENCE frame, NOT in the chassis-body COM frame.
    # ``chassis_body.GetPos()`` returns the body COM (offset from the
    # reference frame by the chassis JSON's ``Body Reference Frame
    # Position`` field — Polaris's COM is roughly 1.6 m forward of the
    # reference origin). Use ``vehicle.GetTransform()``: that returns the
    # ChFramed of the chassis reference frame, which is exactly the pose
    # passed to ``vehicle.Initialize(ChCoordsysd(...))``.
    ref_frame = vehicle.GetTransform()

    axle_centers_local: List[Tuple[float, float, float]] = []
    for axle in veh_dict.get("Axles", []):
        ax_x = _spindle_x_in_chassis_frame(axle)
        ax_y_half = _spindle_y_in_chassis_frame(axle)
        ax_z = _spindle_z_in_chassis_frame(axle)
        axle_centers_local.append((ax_x, +ax_y_half, ax_z))
        axle_centers_local.append((ax_x, -ax_y_half, ax_z))

    # Build the world AABB of the four wheel envelopes. Each wheel is a
    # disk perpendicular to the spindle's local y-axis with outer radius
    # `tire_r`. Under chassis rotation the disk plane rotates with the
    # chassis. Approximating each wheel by an axis-aligned cube of side
    # 2*tire_r centred on the spindle is a tight enough upper bound for
    # the platform-fits check (tire_width is always smaller than tire_r).
    try:
        # PyChrono import is deferred so this module remains importable
        # without a chrono install (utils tests, planner-only flows).
        import pychrono.core as chrono  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "vehicle_world_aabb requires pychrono.core; install pychrono"
        ) from exc

    xmin = ymin = zmin = float("+inf")
    xmax = ymax = zmax = float("-inf")
    for cx, cy, cz in axle_centers_local:
        # Eight corners of the wheel-envelope cube in chassis frame.
        for sx in (-tire_r, +tire_r):
            for sy in (-tire_r, +tire_r):
                for sz in (-tire_r, +tire_r):
                    p_local = chrono.ChVector3d(cx + sx, cy + sy, cz + sz)
                    p_world = ref_frame.TransformPointLocalToParent(p_local)
                    xmin = min(xmin, p_world.x)
                    xmax = max(xmax, p_world.x)
                    ymin = min(ymin, p_world.y)
                    ymax = max(ymax, p_world.y)
                    zmin = min(zmin, p_world.z)
                    zmax = max(zmax, p_world.z)

    return (xmin, xmax, ymin, ymax, zmin, zmax)


def assert_vehicle_on_support(
    vehicle,
    vehicle_json: str | Path,
    support_x_range: Tuple[float, float] | None = None,
    support_y_range: Tuple[float, float] | None = None,
    support_top_z: float | None = None,
    tire_json: str | Path | None = None,
    clearance: float = 0.20,
    z_tolerance: float = 0.02,
    support_name: str = "support",
    support_body=None,
    aabb_tolerance: float = 1e-4,
) -> None:
    """Verify the post-``Initialize`` vehicle AABB fits on the named flat
    support and rests on its top face.

    Raises ``AssertionError`` with a message that includes the suggested
    ``VEH_INIT_X`` / ``VEH_INIT_Y`` shift so the next codegen iteration
    can move the vehicle without re-deriving the geometry.

    There are two calling conventions:

    1. **Body-driven (recommended)**: pass ``support_body=<the support's
       ChBody>`` and let this function derive ``support_x_range``,
       ``support_y_range``, ``support_top_z`` from
       ``body.GetTotalAABB()``. This is the single source of truth.
       If you ALSO pass any of ``support_x_range`` / ``support_y_range`` /
       ``support_top_z``, they are cross-checked against the body AABB and
       a mismatch outside ``aabb_tolerance`` raises ``AssertionError``
       (caller's accounting is wrong — historically this is how
       ``PLT_TOP_Z = LEFT_PLT_Z`` (center-vs-top confusion) sneaks past
       review without crashing in a way that points at the right line).

    2. **Manual ranges (legacy)**: pass all three ranges explicitly with
       no ``support_body``. This is the original signature and is preserved
       for callers that don't have a single ChBody (e.g., a multi-body
       support surface), but it provides no protection against unit /
       semantic errors in the ranges.

    Args:
        vehicle: a wheeled ``veh.ChVehicle`` whose ``Initialize(...)`` has
            already been called.
        support_x_range / support_y_range: ``(min, max)`` pairs giving the
            world-frame extents of the support's top face. Optional when
            ``support_body`` is provided; required otherwise.
        support_top_z: world z of the support's top face. Optional when
            ``support_body`` is provided; required otherwise.
        clearance: minimum required margin between every vehicle AABB face
            and the corresponding support edge along x and y. Defaults to
            0.20 m, matching the existing skill rule.
        z_tolerance: how far the lowest point of the vehicle AABB may
            differ from ``support_top_z`` before we consider the wheels
            either clipped through the support or floating above it.
        support_name: human-readable name used in the assertion message.
        support_body: optional ChBody whose ``GetTotalAABB()`` defines the
            ground-truth support geometry. Body must have a collision
            shape registered (use ``ChBodyEasyBox(..., collide=True)`` or
            attach a ``ChCollisionShapeBox`` + ``EnableCollision(True)``).
        aabb_tolerance: tolerance used when cross-checking caller-provided
            ranges / top-z against the body AABB.
    """
    # --- Resolve support geometry, with body-driven cross-check on top. ----
    if support_body is not None:
        body_top_z = support_top_z_from_body(support_body)
        body_aabb = support_body.GetTotalAABB()
        body_x_range = (float(body_aabb.min.x), float(body_aabb.max.x))
        body_y_range = (float(body_aabb.min.y), float(body_aabb.max.y))

        def _check(name: str, caller, truth):
            if caller is None:
                return truth
            if isinstance(caller, tuple):
                if (abs(caller[0] - truth[0]) > aabb_tolerance
                        or abs(caller[1] - truth[1]) > aabb_tolerance):
                    raise AssertionError(
                        f"caller-provided {name}={caller!r} disagrees with "
                        f"{support_name}.GetTotalAABB()={truth!r} (tol={aabb_tolerance}). "
                        "Either drop the explicit value (body is the source "
                        "of truth) or fix the caller-side arithmetic — this "
                        "mismatch is the iter_006 PLT_TOP_Z=LEFT_PLT_Z class "
                        "of bug (variable named for top, holds center)."
                    )
            else:
                if abs(float(caller) - float(truth)) > aabb_tolerance:
                    raise AssertionError(
                        f"caller-provided {name}={float(caller):.6f} disagrees with "
                        f"{support_name}.GetTotalAABB().max.z={float(truth):.6f} "
                        f"(tol={aabb_tolerance}). The body-derived value is "
                        "authoritative; fix or remove the caller-side value."
                    )
            return truth

        support_x_range = _check("support_x_range", support_x_range, body_x_range)
        support_y_range = _check("support_y_range", support_y_range, body_y_range)
        support_top_z = _check("support_top_z", support_top_z, body_top_z)
    else:
        if support_x_range is None or support_y_range is None or support_top_z is None:
            raise TypeError(
                "assert_vehicle_on_support: pass either support_body=<ChBody> "
                "(recommended) or all three of support_x_range, "
                "support_y_range, support_top_z."
            )

    xmin, xmax, ymin, ymax, zmin, zmax = vehicle_world_aabb(
        vehicle, vehicle_json, tire_json=tire_json
    )
    sx0, sx1 = support_x_range
    sy0, sy1 = support_y_range

    veh_cx = (xmin + xmax) / 2.0
    veh_cy = (ymin + ymax) / 2.0
    sup_cx = (sx0 + sx1) / 2.0
    sup_cy = (sy0 + sy1) / 2.0
    suggested_dx = sup_cx - veh_cx
    suggested_dy = sup_cy - veh_cy

    def _fmt_aabb() -> str:
        return (
            f"vehicle AABB x=[{xmin:+.3f}, {xmax:+.3f}], "
            f"y=[{ymin:+.3f}, {ymax:+.3f}], z=[{zmin:+.3f}, {zmax:+.3f}]; "
            f"{support_name} top x=[{sx0:+.3f}, {sx1:+.3f}], "
            f"y=[{sy0:+.3f}, {sy1:+.3f}], z={support_top_z:+.3f}"
        )

    veh_x_span = xmax - xmin
    veh_y_span = ymax - ymin
    sup_x_span = sx1 - sx0
    sup_y_span = sy1 - sy0
    needed_x_span = veh_x_span + 2 * clearance
    needed_y_span = veh_y_span + 2 * clearance

    if xmin < sx0 + clearance or xmax > sx1 - clearance:
        if needed_x_span > sup_x_span:
            raise AssertionError(
                f"vehicle does not fit on {support_name} along X: vehicle "
                f"x-span={veh_x_span:.3f} m, {support_name} x-span="
                f"{sup_x_span:.3f} m, need ≥ {needed_x_span:.3f} m for "
                f"{clearance} m clearance each side. Either widen "
                f"{support_name} or pick a different support. {_fmt_aabb()}"
            )
        edge = "−X edge (rear hangs off)" if xmin < sx0 + clearance else "+X edge (front hangs off)"
        raise AssertionError(
            f"vehicle off {support_name} {edge}: AABB x="
            f"[{xmin:+.3f}, {xmax:+.3f}], allowed x="
            f"[{sx0 + clearance:+.3f}, {sx1 - clearance:+.3f}]. "
            f"Shift VEH_INIT_X by {suggested_dx:+.3f} m to center wheelbase on "
            f"{support_name} (the chassis-frame origin is not the geometric "
            f"center for every vehicle — Polaris puts it at the front axle, "
            f"HMMWV at the geometric center). {_fmt_aabb()}"
        )
    if ymin < sy0 + clearance or ymax > sy1 - clearance:
        if needed_y_span > sup_y_span:
            raise AssertionError(
                f"vehicle does not fit on {support_name} along Y: vehicle "
                f"y-span={veh_y_span:.3f} m (track + tire), {support_name} "
                f"y-span={sup_y_span:.3f} m, need ≥ {needed_y_span:.3f} m for "
                f"{clearance} m clearance each side. Either widen "
                f"{support_name} or pick a different support. {_fmt_aabb()}"
            )
        edge = "−Y edge" if ymin < sy0 + clearance else "+Y edge"
        raise AssertionError(
            f"vehicle off {support_name} {edge}: AABB y="
            f"[{ymin:+.3f}, {ymax:+.3f}], allowed y="
            f"[{sy0 + clearance:+.3f}, {sy1 - clearance:+.3f}]. "
            f"Shift VEH_INIT_Y by {suggested_dy:+.3f} m. {_fmt_aabb()}"
        )
    if abs(zmin - support_top_z) > z_tolerance:
        raise AssertionError(
            f"vehicle wheels not resting on {support_name}: AABB zmin={zmin:+.3f} "
            f"vs support_top_z={support_top_z:+.3f} (tol={z_tolerance}). "
            f"Recompute VEH_INIT_Z via chassis_init_z(vehicle_json, "
            f"support_top_z={support_top_z}, tire_json=...). {_fmt_aabb()}"
        )
