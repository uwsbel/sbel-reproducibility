"""
Scene asset utilities for PyChrono simulations.

Provides reusable functions for:
- Batch visual asset placement from OBJ files
- Convex hull decomposition with JSON caching
- Loading pre-computed collision shapes onto bodies
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pychrono.core as chrono


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class AssetDescriptor:
    """Description of one visual/physics asset to place in the scene."""

    obj_path: str  # OBJ file path (absolute, or relative to data_dir)
    position: Tuple[float, float, float]  # (x, y, z)
    yaw_deg: float = 0.0  # rotation around Z axis (degrees)
    scale: float = 1.0  # uniform scale factor
    collision: bool = False  # enable collision on the body
    collision_method: str = "convex"  # "convex" (VHACD hulls) or "mesh" (triangle mesh)
    fixed: bool = True  # True → static body, False → dynamic
    mass: float = 1.0  # mass in kg (ignored when fixed=True)
    name: str = ""  # optional body name
    friction: float = 0.6  # contact friction coefficient
    restitution: float = 0.0  # contact restitution coefficient


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_asset_path(obj_path: str, data_dir: Optional[str] = None) -> str:
    """Resolve an asset path against *data_dir* or the Chrono data directory.

    Resolution order:
    1. *obj_path* is absolute and exists → return as-is.
    2. *data_dir* / *obj_path* exists → return that.
    3. ``chrono.GetChronoDataPath()`` / *obj_path* exists → return that.
    4. Return *obj_path* unchanged (caller will see FileNotFoundError later).
    """
    if os.path.isabs(obj_path) and os.path.isfile(obj_path):
        return obj_path

    if data_dir:
        candidate = os.path.join(data_dir, obj_path)
        if os.path.isfile(candidate):
            return candidate

    try:
        import pychrono.core as chrono

        candidate = os.path.join(chrono.GetChronoDataPath(), obj_path)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass

    return obj_path


def _quat_from_angle_x(deg: float):
    half = np.deg2rad(deg) * 0.5
    import pychrono.core as chrono

    return chrono.ChQuaterniond(float(np.cos(half)), float(np.sin(half)), 0.0, 0.0)


def _quat_from_angle_z(deg: float):
    half = np.deg2rad(deg) * 0.5
    import pychrono.core as chrono

    return chrono.ChQuaterniond(float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def _quat_mul(a, b):
    import pychrono.core as chrono

    return chrono.ChQuaterniond(
        a.e0 * b.e0 - a.e1 * b.e1 - a.e2 * b.e2 - a.e3 * b.e3,
        a.e0 * b.e1 + a.e1 * b.e0 + a.e2 * b.e3 - a.e3 * b.e2,
        a.e0 * b.e2 - a.e1 * b.e3 + a.e2 * b.e0 + a.e3 * b.e1,
        a.e0 * b.e3 + a.e1 * b.e2 - a.e2 * b.e1 + a.e3 * b.e0,
    )


# ── Public helpers ───────────────────────────────────────────────────────────


def make_contact_material(
    friction: float = 0.6,
    restitution: float = 0.0,
    *,
    method: str = "NSC",
    young_modulus: float = 2e7,
    gn: float = 60.0,
    kn: float = 2e5,
):
    """Create a contact material matching the system's contact method.

    Parameters
    ----------
    friction, restitution:
        Coulomb friction and restitution coefficients.
    method:
        ``"NSC"`` (default, non-smooth) or ``"SMC"`` (smooth / penalty).
        Must match the ``ChSystem`` type: NSC for ``ChSystemNSC``,
        SMC for ``ChSystemSMC``. Mixing the two corrupts the collision
        system and freezes the solver.
    young_modulus, gn, kn:
        SMC-only parameters. Ignored when method=``"NSC"``.
    """
    import pychrono.core as chrono

    m = method.upper()
    if m == "NSC":
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(friction)
        mat.SetRestitution(restitution)
    elif m == "SMC":
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(friction)
        mat.SetRestitution(restitution)
        mat.SetYoungModulus(young_modulus)
        mat.SetGn(gn)
        mat.SetKn(kn)
    else:
        raise ValueError(f"Unknown contact method: {method!r} (use 'NSC' or 'SMC')")
    return mat


def _detect_contact_method(system) -> str:
    """Infer ``'NSC'`` or ``'SMC'`` from a ChSystem instance."""
    name = type(system).__name__
    if "SMC" in name:
        return "SMC"
    return "NSC"


def box_inertia(mass: float, size: Sequence[float]):
    """Compute box-approximation rotational inertia for *(sx, sy, sz)*."""
    import pychrono.core as chrono

    sx, sy, sz = size
    return chrono.ChVector3d(
        mass * (sy * sy + sz * sz) / 12.0,
        mass * (sx * sx + sz * sz) / 12.0,
        mass * (sx * sx + sy * sy) / 12.0,
    )


def load_convex_hulls(
    json_path: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load pre-computed convex hull parts from a ``*_convex.json`` file.

    Returns a list of ``(vertices, faces)`` numpy array pairs.
    """
    with open(json_path) as f:
        data = json.load(f)
    return [
        (np.array(d["vertices"], dtype=np.float64), np.array(d["faces"], dtype=np.int32))
        for d in data
    ]


def transform_vertices(
    vertices: np.ndarray,
    center_orig: Sequence[float],
    scale_rot: np.ndarray,
    *,
    pretransformed: bool = False,
    extra_scale: float = 1.0,
) -> np.ndarray:
    """Apply the visual transform (center → scale+rotate) to hull vertices.

    When *pretransformed* is ``True`` the vertices are already centred and
    scaled; only the pure rotation extracted from *scale_rot* is applied,
    optionally multiplied by *extra_scale*.
    """
    sr = np.asarray(scale_rot, dtype=np.float64)
    if pretransformed:
        col_norms = np.linalg.norm(sr, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        rotation_only = sr / col_norms
        v = np.array(vertices, dtype=np.float64, copy=True) * float(extra_scale)
        return v @ rotation_only.T
    centered = vertices - np.asarray(center_orig, dtype=np.float64)
    return centered @ sr.T


# ── Function 1: add_visual_assets ────────────────────────────────────────────


def _normalize_descriptor(item: Union[AssetDescriptor, tuple]) -> AssetDescriptor:
    """Convert a plain tuple to an ``AssetDescriptor``."""
    if isinstance(item, AssetDescriptor):
        return item
    if isinstance(item, (tuple, list)):
        # Supported formats:
        #   (obj_path, position)
        #   (obj_path, position, yaw_deg)
        #   (obj_path, position, yaw_deg, scale)
        obj_path = item[0]
        position = item[1]
        yaw_deg = item[2] if len(item) > 2 else 0.0
        scale = item[3] if len(item) > 3 else 1.0
        return AssetDescriptor(obj_path=obj_path, position=position, yaw_deg=yaw_deg, scale=scale)
    raise TypeError(f"Expected AssetDescriptor or tuple, got {type(item)}")


def add_visual_assets(
    system,
    assets: Sequence[Union[AssetDescriptor, tuple]],
    *,
    data_dir: Optional[str] = None,
    debug_collision: bool = False,
) -> List:
    """Add multiple visual (and optionally collidable) assets to a Chrono system.

    Each entry in *assets* is either an :class:`AssetDescriptor` or a plain
    tuple ``(obj_path, position[, yaw_deg[, scale]])``.  Paths are resolved
    via :func:`_resolve_asset_path` (local *data_dir* first, then
    ``chrono.GetChronoDataPath()``).

    The contact material is auto-matched to the system's type (NSC/SMC).

    Parameters
    ----------
    debug_collision:
        If ``True``, render each VHACD convex hull as a translucent
        coloured overlay so the generated collision geometry is visible.

    Returns a list of ``chrono.ChBody`` objects (``None`` for missing files).
    """
    import pychrono.core as chrono

    contact_method = _detect_contact_method(system)
    bodies: List = []
    for item in assets:
        desc = _normalize_descriptor(item)
        resolved = _resolve_asset_path(desc.obj_path, data_dir)

        if not os.path.isfile(resolved):
            print(f"[warn] asset missing, skipping: {resolved}")
            bodies.append(None)
            continue

        # Load mesh
        mesh = chrono.ChTriangleMeshConnected()
        mesh.LoadWavefrontMesh(resolved, True, True)

        # Uniform scale
        if desc.scale != 1.0:
            mat33 = chrono.ChMatrix33d(desc.scale)
            mesh.Transform(chrono.ChVector3d(0, 0, 0), mat33)

        # Visual shape
        shape = chrono.ChVisualShapeTriangleMesh()
        shape.SetMesh(mesh)
        shape.SetName(desc.name or os.path.basename(resolved))
        shape.SetMutable(False)

        # Body
        body = chrono.ChBody()
        if desc.name:
            body.SetName(desc.name)
        body.SetFixed(desc.fixed)
        body.SetPos(chrono.ChVector3d(*desc.position))
        body.SetRot(chrono.QuatFromAngleZ(math.radians(desc.yaw_deg)))

        if not desc.fixed:
            body.SetMass(desc.mass)
            bb = mesh.GetBoundingBox()
            size = (
                bb.max.x - bb.min.x,
                bb.max.y - bb.min.y,
                bb.max.z - bb.min.z,
            )
            body.SetInertiaXX(box_inertia(desc.mass, size))

        # Collision
        body.EnableCollision(desc.collision)
        if desc.collision:
            mat = make_contact_material(
                desc.friction, desc.restitution, method=contact_method
            )
            if desc.collision_method == "convex":
                # VHACD convex-hull collision. The visual mesh was scaled in
                # place but NOT centered, so the hull vertices need the same
                # uniform scale applied and no centering offset.
                json_path = ensure_convex_json(resolved)
                scale_rot = np.eye(3, dtype=np.float64) * float(desc.scale)
                add_collision_from_decomposition(
                    body,
                    json_path,
                    mat,
                    center_orig=[0.0, 0.0, 0.0],
                    scale_rot=scale_rot,
                    pretransformed=False,
                    obj_path=resolved,
                    debug_visuals=debug_collision,
                )
            elif desc.collision_method == "mesh":
                collision_shape = chrono.ChCollisionShapeTriangleMesh(
                    mat, mesh, True, False, 0.0
                )
                body.AddCollisionShape(
                    collision_shape,
                    chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT),
                )
            elif desc.collision_method == "single_convex":
                # ONE outer convex hull of the entire mesh via scipy.
                # Use this for dynamic bodies: Bullet's cbtCompoundShape
                # on a dynamic body has a broadphase-AABB bug that makes
                # multi-hull compounds invisible to narrowphase. A single
                # ChCollisionShapeConvexHull is attached directly to the
                # bt_collision_object (no compound wrapper), which works.
                from scipy.spatial import ConvexHull as _SciHull
                verts_chrono = mesh.GetCoordsVertices()
                pts_np = np.array(
                    [[v.x, v.y, v.z] for v in verts_chrono], dtype=np.float64
                )
                hull_sci = _SciHull(pts_np)
                hull_pts_np = pts_np[hull_sci.vertices]
                hull_ch_pts = [
                    chrono.ChVector3d(float(p[0]), float(p[1]), float(p[2]))
                    for p in hull_pts_np
                ]
                collision_shape = chrono.ChCollisionShapeConvexHull(
                    mat, hull_ch_pts
                )
                body.AddCollisionShape(
                    collision_shape,
                    chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT),
                )
                print(
                    f"  [{desc.name or os.path.basename(resolved)}] "
                    f"single convex hull: {len(hull_pts_np)} verts "
                    f"(from {len(pts_np)} mesh verts)"
                )
            else:
                raise ValueError(
                    f"Unknown collision_method: {desc.collision_method!r} "
                    f"(use 'convex', 'mesh', or 'single_convex')"
                )

        body.AddVisualShape(
            shape, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
        )
        system.AddBody(body)
        bodies.append(body)

    return bodies


# ── Function 2: convex_decompose_asset ───────────────────────────────────────


def _cache_path_for(obj_path: str) -> str:
    """Return the JSON cache path that sits next to the OBJ file."""
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(os.path.dirname(obj_path), f"{stem}_convex.json")


def _save_cache(path: str, parts: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    data = [{"vertices": v.tolist(), "faces": f.tolist()} for v, f in parts]
    with open(path, "w") as fh:
        json.dump(data, fh)


def convex_decompose_asset(
    obj_path: str,
    *,
    method: str = "coacd",
    max_hulls: int = 8,
    max_verts_per_hull: int = 64,
    resolution: int = 100_000,
    threshold: float = 0.05,
    force: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Decompose an OBJ mesh into convex hulls, caching the result as JSON.

    The cache file is written next to the OBJ as ``{stem}_convex.json``.
    If the cache already exists and *force* is ``False``, the cached result
    is loaded and returned immediately.

    Parameters
    ----------
    obj_path:
        Path to the ``.obj`` file.
    method:
        ``"coacd"`` (default, uses the *coacd* package — preferred because
        CoACD produces tight, non-overlapping hulls that play better with
        Bullet narrowphase on compound shapes) or ``"vhacd"`` (uses
        ``vhacdx`` via ``trimesh.decomposition.convex_decomposition``).
    max_hulls:
        Maximum number of convex parts.
    max_verts_per_hull:
        Hulls with more vertices are down-sampled to this count.
        Also passed to VHACD as the per-hull vertex budget.
    resolution:
        VHACD voxel resolution (only used when *method* = ``"vhacd"``).
    threshold:
        CoACD concavity threshold (only used when *method* = ``"coacd"``).
    force:
        Re-compute even if a cache file exists.

    Returns
    -------
    list of (vertices, faces) ndarray pairs.
    """
    import trimesh

    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    cache_path = _cache_path_for(obj_path)

    # Try cache first
    if not force and os.path.isfile(cache_path):
        print(f"  [convex_decompose] Loading cache: {cache_path}")
        return load_convex_hulls(cache_path)

    # Load mesh via trimesh
    tm = trimesh.load(obj_path, force="mesh", process=False)

    if method == "vhacd":
        # trimesh wraps the vhacdx package; kwargs are VHACD's camelCase names.
        raw_parts = trimesh.decomposition.convex_decomposition(
            tm,
            maxConvexHulls=max_hulls,
            resolution=resolution,
            maxNumVerticesPerCH=max_verts_per_hull,
        )
        if not isinstance(raw_parts, list):
            raw_parts = [raw_parts]
        parts: List[Tuple[np.ndarray, np.ndarray]] = [
            (
                np.asarray(p["vertices"], dtype=np.float64),
                np.asarray(p["faces"], dtype=np.int32),
            )
            for p in raw_parts
        ]
    elif method == "coacd":
        import coacd

        mesh_coacd = coacd.Mesh(tm.vertices, tm.faces)
        raw_parts = coacd.run_coacd(
            mesh_coacd,
            threshold=threshold,
            max_convex_hull=max_hulls,
            # Cap per-hull vertex count inside CoACD itself. If we instead
            # let it emit 256-vert hulls and down-sample in the loop below,
            # the face index array would be discarded (see comment there),
            # which leaves the downstream _add_collision_debug_visuals
            # path trying to build a ChTriangleMesh from zero faces — VSG
            # then segfaults while computing normals on the empty mesh.
            max_ch_vertex=max_verts_per_hull,
        )
        parts = [
            (np.array(vs, dtype=np.float64), np.array(fs, dtype=np.int32))
            for vs, fs in raw_parts
        ]
    else:
        raise ValueError(f"Unknown decomposition method: {method!r} (use 'vhacd' or 'coacd')")

    # Down-sample oversized hulls (safety net; VHACD already respects
    # maxNumVerticesPerCH, but CoACD silently ignores max_ch_vertex and
    # routinely emits hulls with ~500 vertices).
    #
    # After down-sampling, the original face indices point into the old
    # vertex array and become invalid. If we just discarded the faces
    # (the previous behaviour), callers that rely on `faces` — notably
    # _add_collision_debug_visuals — build an empty triangle mesh, which
    # segfaults inside VSG while computing per-vertex normals. Instead,
    # rebuild a fresh convex triangulation from the sub-sampled vertices
    # using scipy.spatial.ConvexHull.
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for verts, faces in parts:
        if len(verts) > max_verts_per_hull:
            idx = np.linspace(0, len(verts) - 1, max_verts_per_hull, dtype=int)
            verts = verts[idx]
            try:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(verts)
                faces = np.asarray(hull.simplices, dtype=np.int32)
            except Exception:
                # If scipy is missing or the sub-sampled point set is
                # degenerate, fall back to empty faces — the collision
                # shape still works (ChCollisionShapeConvexHull only
                # needs the point cloud), we just lose the debug overlay
                # for that hull.
                faces = np.empty((0, 3), dtype=np.int32)
        result.append((verts, faces))

    # Save cache
    _save_cache(cache_path, result)
    print(f"  [convex_decompose] Saved {len(result)} hulls → {cache_path}")

    return result


def ensure_convex_json(
    obj_path: str,
    *,
    method: str = "coacd",
    max_hulls: int = 8,
    max_verts_per_hull: int = 64,
    resolution: int = 100_000,
    threshold: float = 0.05,
    force: bool = False,
) -> str:
    """Return the path to ``{stem}_convex.json`` next to *obj_path*.

    If the JSON cache already exists (and *force* is False), the path is
    returned immediately. Otherwise the asset is decomposed via CoACD (by
    default) and the cache is written first.

    Use this before :func:`add_collision_from_decomposition` when you want
    collision geometry to be generated on demand.

    Returns
    -------
    Absolute (or caller-provided) path to the ``*_convex.json`` file.
    """
    cache_path = _cache_path_for(obj_path)
    if force or not os.path.isfile(cache_path):
        convex_decompose_asset(
            obj_path,
            method=method,
            max_hulls=max_hulls,
            max_verts_per_hull=max_verts_per_hull,
            resolution=resolution,
            threshold=threshold,
            force=force,
        )
    return cache_path


# ── Function 3: add_collision_from_decomposition ─────────────────────────────


def add_collision_from_decomposition(
    body,
    json_path: str,
    contact_material,
    center_orig: Sequence[float],
    scale_rot: np.ndarray,
    *,
    pretransformed: bool = False,
    extra_scale: float = 1.0,
    max_verts_per_hull: int = 256,
    debug_visuals: bool = False,
    obj_path: Optional[str] = None,
) -> Tuple[int, Optional[float]]:
    """Load convex hulls from JSON and add them as collision shapes to *body*.

    Parameters
    ----------
    body:
        ``chrono.ChBody`` to receive the collision shapes.
    json_path:
        Path to a ``*_convex.json`` produced by :func:`convex_decompose_asset`.
    contact_material:
        ``ChContactMaterialNSC`` (or SMC) for the collision shapes.
    center_orig:
        Original mesh bounding-box centre ``(cx, cy, cz)`` before transforms.
    scale_rot:
        3×3 combined scale + rotation matrix (same one used to transform the
        visual mesh).
    pretransformed:
        ``True`` if the JSON vertices are already centred and scaled (only
        the rotation component of *scale_rot* is applied).
    extra_scale:
        Additional uniform scale factor applied on top of *scale_rot*.
    max_verts_per_hull:
        Down-sample hulls exceeding this vertex count.
    debug_visuals:
        Add translucent mesh overlays of each convex part for debugging.
    obj_path:
        Optional path to the source ``.obj`` file. When provided and
        *json_path* does not exist, the asset is decomposed via VHACD
        (``ensure_convex_json``) before loading the cache.

    Returns
    -------
    ``(num_hulls, collision_bottom_z)`` — the number of hulls added and the
    lowest Z coordinate across all hull vertices (useful for ground placement).
    """
    import pychrono.core as chrono

    # Auto-generate the JSON cache from the source OBJ if available.
    if not os.path.isfile(json_path):
        if obj_path and os.path.isfile(obj_path):
            print(
                f"  [auto-decomp] {os.path.basename(json_path)} missing — "
                f"running VHACD on {obj_path}"
            )
            ensure_convex_json(obj_path)
        else:
            print(f"[warn] convex JSON not found: {json_path}, skipping collision")
            return 0, None

    parts = load_convex_hulls(json_path)
    transformed_parts: List[Tuple[np.ndarray, np.ndarray]] = []
    total_verts = 0
    collision_bottom_z: Optional[float] = None

    for verts, faces in parts:
        verts = transform_vertices(
            verts,
            center_orig,
            scale_rot,
            pretransformed=pretransformed,
            extra_scale=extra_scale,
        )
        transformed_parts.append((verts, faces))

        # Track lowest Z for ground-plane placement
        if len(verts) > 0:
            part_min_z = float(np.min(verts[:, 2]))
            if collision_bottom_z is None:
                collision_bottom_z = part_min_z
            else:
                collision_bottom_z = min(collision_bottom_z, part_min_z)

        # Down-sample before creating collision shape
        if len(verts) > max_verts_per_hull:
            idx = np.linspace(0, len(verts) - 1, max_verts_per_hull, dtype=int)
            verts = verts[idx]

        total_verts += len(verts)
        hull_pts = [
            chrono.ChVector3d(float(v[0]), float(v[1]), float(v[2])) for v in verts
        ]
        shape = chrono.ChCollisionShapeConvexHull(contact_material, hull_pts)
        body.AddCollisionShape(
            shape, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
        )

    body.EnableCollision(True)

    # Optional debug overlays
    if debug_visuals:
        _add_collision_debug_visuals(body, transformed_parts)

    print(
        f"  [{body.GetName()}] Loaded {len(parts)} convex hulls, "
        f"{total_verts} total verts"
    )
    return len(parts), collision_bottom_z


def add_collision_via_subbodies(
    system,
    anchor_body,
    json_path: str,
    contact_material,
    center_orig: Sequence[float],
    scale_rot: np.ndarray,
    *,
    pretransformed: bool = False,
    extra_scale: float = 1.0,
    max_verts_per_hull: int = 256,
    obj_path: Optional[str] = None,
    sub_mass: float = 0.1,
    collision_family: int = 0,
    tire_family: int = 1,
    self_family: int = 3,
    debug_visuals: bool = False,
) -> List:
    """Attach a decomposed convex-hull collision to *anchor_body* by
    spawning one sub-body per hull and rigidly welding it to the anchor
    via ``ChLinkMateFix``.

    This sidesteps Bullet's ``cbtCompoundShape`` bug on dynamic bodies:
    every sub-body has a single ``ChCollisionShapeConvexHull`` attached
    with an identity frame, so Bullet stores it directly on the
    ``bt_collision_object`` (no compound wrapper). The welded sub-bodies
    move rigidly with the anchor, so narrowphase + physics behave as if
    all hulls were on the anchor.

    Parameters
    ----------
    system:
        The ``ChSystem`` that owns *anchor_body*.
    anchor_body:
        The main body (``ChBody`` or ``ChBodyAuxRef``) whose world pose
        the sub-bodies should follow. The sub-bodies are welded to it.
    json_path, contact_material, center_orig, scale_rot,
    pretransformed, extra_scale, max_verts_per_hull, obj_path:
        Same semantics as :func:`add_collision_from_decomposition`.
    sub_mass:
        Mass assigned to each sub-body. Default 0.1 kg — negligible
        compared to typical vehicle chassis mass, so the welded sum
        barely perturbs the anchor's dynamics. Pick smaller values if
        your simulation uses very light main bodies.
    collision_family, tire_family, self_family:
        Family slots for vehicle collision filtering. Each sub-body is
        placed in ``self_family``, disallows collisions with ``tire_family``
        (so tires don't bump into chassis hulls) and with ``self_family``
        itself (so sibling sub-bodies don't self-collide through their
        possibly-overlapping CoACD decomposition). Sub-bodies remain
        collideable with everything else (default family 0 rocks etc.).
    debug_visuals:
        Add translucent coloured overlays to each sub-body for visual
        inspection of where the collision hulls actually sit.

    Returns
    -------
    A list of the created sub-bodies (already welded and added to
    *system*).
    """
    import pychrono.core as chrono

    # Auto-generate the JSON cache from the source OBJ if available.
    if not os.path.isfile(json_path):
        if obj_path and os.path.isfile(obj_path):
            print(
                f"  [auto-decomp] {os.path.basename(json_path)} missing — "
                f"running decomposition on {obj_path}"
            )
            ensure_convex_json(obj_path)
        else:
            print(f"[warn] convex JSON not found: {json_path}, skipping collision")
            return []

    parts = load_convex_hulls(json_path)

    ref_pose = (
        anchor_body.GetFrameRefToAbs()
        if hasattr(anchor_body, "GetFrameRefToAbs")
        else anchor_body.GetFrame_REF_to_abs()
        if hasattr(anchor_body, "GetFrame_REF_to_abs")
        else anchor_body
    )
    anchor_pos = ref_pose.GetPos() if hasattr(ref_pose, "GetPos") else anchor_body.GetPos()
    anchor_rot = ref_pose.GetRot() if hasattr(ref_pose, "GetRot") else anchor_body.GetRot()

    anchor_name = anchor_body.GetName() or "anchor"
    sub_bodies: List = []

    for i, (verts, faces) in enumerate(parts):
        verts = transform_vertices(
            verts,
            center_orig,
            scale_rot,
            pretransformed=pretransformed,
            extra_scale=extra_scale,
        )
        if len(verts) > max_verts_per_hull:
            idx = np.linspace(0, len(verts) - 1, max_verts_per_hull, dtype=int)
            verts = verts[idx]

        hull_pts = [
            chrono.ChVector3d(float(v[0]), float(v[1]), float(v[2])) for v in verts
        ]
        shape = chrono.ChCollisionShapeConvexHull(contact_material, hull_pts)

        sub = chrono.ChBody()
        sub.SetName(f"{anchor_name}_hull{i}")
        sub.SetMass(sub_mass)
        # Tiny inertia — the weld locks all DOFs anyway.
        sub.SetInertiaXX(chrono.ChVector3d(sub_mass, sub_mass, sub_mass))
        sub.SetPos(anchor_pos)
        sub.SetRot(anchor_rot)
        sub.AddCollisionShape(
            shape, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
        )
        sub.EnableCollision(True)
        system.AddBody(sub)

        sub_cm = sub.GetCollisionModel()
        sub_cm.SetFamily(self_family)
        sub_cm.DisallowCollisionsWith(self_family)   # no sibling self-collisions
        sub_cm.DisallowCollisionsWith(tire_family)   # no chassis-tire contact

        # Weld sub rigidly to anchor: ChLinkMateFix locks all 6 DOFs.
        weld = chrono.ChLinkMateFix()
        weld.Initialize(sub, anchor_body)
        system.AddLink(weld)

        if debug_visuals and faces is not None and len(faces) > 0:
            debug_mesh = chrono.ChTriangleMeshConnected()
            for face in faces:
                if len(face) < 3:
                    continue
                v0 = verts[int(face[0])]
                for j in range(1, len(face) - 1):
                    v1 = verts[int(face[j])]
                    v2 = verts[int(face[j + 1])]
                    debug_mesh.AddTriangle(
                        chrono.ChVector3d(float(v0[0]), float(v0[1]), float(v0[2])),
                        chrono.ChVector3d(float(v1[0]), float(v1[1]), float(v1[2])),
                        chrono.ChVector3d(float(v2[0]), float(v2[1]), float(v2[2])),
                    )
            colors = [
                chrono.ChColor(1.0, 0.2, 0.2), chrono.ChColor(0.2, 1.0, 0.2),
                chrono.ChColor(0.2, 0.2, 1.0), chrono.ChColor(1.0, 1.0, 0.2),
                chrono.ChColor(1.0, 0.2, 1.0), chrono.ChColor(0.2, 1.0, 1.0),
                chrono.ChColor(1.0, 0.6, 0.2), chrono.ChColor(0.6, 0.2, 1.0),
            ]
            color = colors[i % len(colors)]
            dbg_shape = chrono.ChVisualShapeTriangleMesh()
            dbg_shape.SetMesh(debug_mesh)
            dbg_shape.SetColor(color)
            dbg_shape.SetOpacity(0.45)
            dbg_shape.SetName(f"{anchor_name}_hull{i}_dbg")
            dbg_vis_mat = chrono.ChVisualMaterial()
            dbg_vis_mat.SetDiffuseColor(color)
            dbg_vis_mat.SetOpacity(0.45)
            dbg_shape.AddMaterial(dbg_vis_mat)
            sub.AddVisualShape(
                dbg_shape, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
            )

        sub_bodies.append(sub)

    print(
        f"  [{anchor_name}] Attached {len(sub_bodies)} sub-body hulls "
        f"(single-shape each) welded via ChLinkMateFix"
    )
    return sub_bodies


def _add_collision_debug_visuals(
    body, parts: List[Tuple[np.ndarray, np.ndarray]]
) -> None:
    """Render each convex part as a translucent solid overlay on *body*.

    Both VSG and Chrono::Sensor (OptiX) render paths are covered:
    * VSG reads ``ChVisualShape.SetColor`` / ``SetOpacity`` directly.
    * Chrono::Sensor skips any shape with zero ``ChVisualMaterial`` entries,
      so an explicit material with the matching diffuse + transparency is
      attached as well.
    """
    import pychrono.core as chrono

    colors = [
        chrono.ChColor(1.0, 0.2, 0.2),
        chrono.ChColor(0.2, 1.0, 0.2),
        chrono.ChColor(0.2, 0.2, 1.0),
        chrono.ChColor(1.0, 1.0, 0.2),
        chrono.ChColor(1.0, 0.2, 1.0),
        chrono.ChColor(0.2, 1.0, 1.0),
        chrono.ChColor(1.0, 0.6, 0.2),
        chrono.ChColor(0.6, 0.2, 1.0),
    ]
    opacity = 0.35
    for i, (verts, faces) in enumerate(parts):
        # VSG segfaults while computing per-vertex normals on an empty
        # ChTriangleMeshConnected. Skip hulls without usable faces.
        if faces is None or len(faces) == 0:
            continue
        debug_mesh = chrono.ChTriangleMeshConnected()
        n_triangles_added = 0
        for face in faces:
            if len(face) < 3:
                continue
            v0 = verts[int(face[0])]
            for j in range(1, len(face) - 1):
                v1 = verts[int(face[j])]
                v2 = verts[int(face[j + 1])]
                debug_mesh.AddTriangle(
                    chrono.ChVector3d(float(v0[0]), float(v0[1]), float(v0[2])),
                    chrono.ChVector3d(float(v1[0]), float(v1[1]), float(v1[2])),
                    chrono.ChVector3d(float(v2[0]), float(v2[1]), float(v2[2])),
                )
                n_triangles_added += 1
        if n_triangles_added == 0:
            continue
        color = colors[i % len(colors)]
        debug_shape = chrono.ChVisualShapeTriangleMesh()
        debug_shape.SetMesh(debug_mesh)
        debug_shape.SetColor(color)          # VSG path
        debug_shape.SetOpacity(opacity)      # VSG path
        debug_shape.SetName(f"collision_debug_{i}")

        # Sensor (OptiX) path: needs an explicit ChVisualMaterial.
        debug_mat = chrono.ChVisualMaterial()
        debug_mat.SetDiffuseColor(color)
        debug_mat.SetOpacity(float(opacity))
        debug_shape.AddMaterial(debug_mat)

        body.AddVisualShape(
            debug_shape,
            chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT),
        )


# ── High-level: create_asset_body ────────────────────────────────────────────


def create_asset_body(
    system,
    name: str,
    asset_dir: str,
    assets_base_dir: str,
    target_heights: dict,
    position: Sequence[float],
    *,
    asset_rotation: Optional[dict] = None,
    pretransformed_assets: Optional[set] = None,
    contact_material=None,
    fixed: bool = False,
    mass: float = 5.0,
    scale_factor: Optional[float] = None,
    scale_multiplier: float = 1.0,
    height_axis: int = 2,
    debug_collision: bool = False,
) -> Tuple:
    """Create a body from a local scene asset with convex hull collision.

    This is the main high-level function that performs the complete pipeline:
    mesh loading → two-step transform (center + scale/rotate) → visual shape
    → convex hull collision from JSON → support-surface Z placement.

    Parameters
    ----------
    system:
        ``chrono.ChSystemNSC`` or ``ChSystemSMC``.
    name:
        Unique body name.
    asset_dir:
        Sub-directory under *assets_base_dir* (e.g. ``"computer_table"``).
    assets_base_dir:
        Absolute path to the ``data/scene/`` directory.
    target_heights:
        ``{asset_name: height_m}`` dict.  The value is the **full mesh
        z-extent** (asset bounding-box height), NOT a sub-feature like
        seat height or tabletop height.  Example: an office chair whose
        back reaches 0.85 m must be passed ``0.85``, not ``0.45`` (the
        seat height) — the seat will land at ~0.45 m by mesh proportion.
        Mutually exclusive with *scale_factor*.
    position:
        ``(x, y, z)`` world position. ``z`` is the **support surface Z**
        (e.g. floor top); the body is placed so its visual bottom rests
        on this Z. **Do NOT pass body-center Z** — the function applies
        the ``visual_bottom_z`` offset internally.
    asset_rotation:
        Optional ``{asset_name: (deg_x, deg_z)}`` dict.  Defaults to
        ``(0.0, 0.0)`` when the asset is absent.
    pretransformed_assets:
        Set of asset names whose ``*_convex.json`` is already centred and
        scaled at the base target height.
    contact_material:
        Pre-built ``ChContactMaterialNSC`` or ``ChContactMaterialSMC``.
        If ``None``, a default NSC material (friction=0.95) is created.
    fixed:
        ``False`` for dynamic (pushable), ``True`` for immovable.
    mass:
        Body mass in kg.
    scale_factor:
        Override the auto-computed scale.  When ``None`` (default), it is
        derived from ``target_heights[asset_dir] / raw_size[height_axis]``.
        **Cannot be passed together with a value in target_heights for
        the same asset_dir** — pass exactly one.
    scale_multiplier:
        Additional scale boost on top of *scale_factor*.
    height_axis:
        Index into the raw AABB size for the height dimension (default 2 = Z).
    debug_collision:
        Add translucent collision-hull debug overlays.

    Returns
    -------
    ``(body, transformed_size)`` — the body (already added to *system*) and
    its post-transform ``[sx, sy, sz]`` bounding-box dimensions.
    """
    # ── Argument validation (run BEFORE any I/O so callers see the
    # error even when the asset path is wrong).
    # Reject double-pass: if both scale_factor and target_heights[asset_dir]
    # are provided, scale_factor would silently override target_heights and
    # any inconsistency between them gets swallowed. Force the caller to
    # pick one.
    if scale_factor is not None and asset_dir in (target_heights or {}):
        raise ValueError(
            f"create_asset_body({name!r}): pass either scale_factor OR "
            f"target_heights[{asset_dir!r}], not both. "
            f"Got scale_factor={scale_factor} and "
            f"target_heights[{asset_dir!r}]={target_heights[asset_dir]}."
        )

    import pychrono.core as chrono

    obj_path = os.path.join(assets_base_dir, asset_dir, f"{asset_dir}.obj")
    json_path = os.path.join(assets_base_dir, asset_dir, f"{asset_dir}_convex.json")

    # ── Load mesh ────────────────────────────────────────────────────────
    mesh = chrono.ChTriangleMeshConnected()
    mesh.LoadWavefrontMesh(obj_path, True, True)

    bb = mesh.GetBoundingBox()
    raw_size = [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z]
    center_orig = [
        (bb.min.x + bb.max.x) / 2,
        (bb.min.y + bb.max.y) / 2,
        (bb.min.z + bb.max.z) / 2,
    ]

    # ── Compute scale ────────────────────────────────────────────────────
    if scale_factor is None:
        target_height = target_heights.get(asset_dir, 1.0)
        scale_factor = target_height / raw_size[height_axis]
    scale_factor *= scale_multiplier

    # ── Step 1: centre mesh at origin ────────────────────────────────────
    identity = np.eye(3, dtype=np.float64)
    mat33_identity = chrono.ChMatrix33d()
    mat33_identity.SetMatr(identity)
    mesh.Transform(
        chrono.ChVector3d(-center_orig[0], -center_orig[1], -center_orig[2]),
        mat33_identity,
    )

    # ── Step 2: scale + rotation ─────────────────────────────────────────
    rot_table = asset_rotation or {}
    deg_x, deg_z = rot_table.get(asset_dir, (0.0, 0.0))
    asset_quat = _quat_mul(_quat_from_angle_z(deg_z), _quat_from_angle_x(deg_x))
    qn = chrono.ChQuaterniond(asset_quat.e0, asset_quat.e1, asset_quat.e2, asset_quat.e3)
    qn.Normalize()
    w, x, y, z = qn.e0, qn.e1, qn.e2, qn.e3

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    scale_rot = R @ np.diag(np.full(3, scale_factor))
    mat33 = chrono.ChMatrix33d()
    mat33.SetMatr(scale_rot)
    mesh.Transform(chrono.ChVector3d(0, 0, 0), mat33)

    # ── Post-transform metrics ───────────────────────────────────────────
    bb = mesh.GetBoundingBox()
    transformed_size = [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z]
    visual_bottom_z = bb.min.z

    # ── Build body ───────────────────────────────────────────────────────
    body = chrono.ChBody()
    body.SetName(name)
    body.SetMass(mass)
    body.SetInertiaXX(box_inertia(mass, transformed_size))
    body.SetFixed(fixed)
    body.SetSleepingAllowed(False)

    vis_shape = chrono.ChVisualShapeTriangleMesh()
    vis_shape.SetMesh(mesh)
    vis_shape.SetColor(chrono.ChColor(0.8, 0.8, 0.8))
    body.AddVisualShape(
        vis_shape, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
    )

    # ── Convex-hull collision from JSON ──────────────────────────────────
    if contact_material is None:
        contact_material = make_contact_material(
            friction=0.95,
            restitution=0.0,
            method=_detect_contact_method(system),
        )

    pre_set = pretransformed_assets or set()
    # JSON is auto-generated inside add_collision_from_decomposition when
    # missing, by passing obj_path for the on-demand VHACD decomposition.
    add_collision_from_decomposition(
        body,
        json_path,
        contact_material,
        center_orig,
        scale_rot,
        pretransformed=asset_dir in pre_set,
        extra_scale=scale_multiplier,
        debug_visuals=debug_collision,
        obj_path=obj_path,
    )

    # ── Place body so its bottom rests on the support surface ────────────
    support_z = position[2] - visual_bottom_z
    body.SetPos(chrono.ChVector3d(position[0], position[1], support_z))

    system.AddBody(body)
    return body, transformed_size


# ── CSV writers ──────────────────────────────────────────────────────────────


def write_placement_csv(system, output_dir: str = ".") -> str:
    """Write body placement state for deterministic physics validation.

    OPTIONAL — not required by the review pipeline. Per-step motion
    review reads ``cam/motion_log.csv`` (written by codegen when the
    step declares ``motion_expectations``). This writer remains
    available for ad-hoc diagnostics on legacy plans.

    Returns the path to the written CSV file.
    """
    import csv

    import pychrono.core as chrono

    os.makedirs(output_dir, exist_ok=True)
    placement_path = os.path.join(output_dir, "scene_placement.csv")
    with open(placement_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "body_name",
                "pos_x", "pos_y", "pos_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "aabb_min_x", "aabb_min_y", "aabb_min_z",
                "aabb_max_x", "aabb_max_y", "aabb_max_z",
                "vel_x", "vel_y", "vel_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
            ]
        )
        for body in system.GetBodies():
            name = body.GetName()
            if not name or name.startswith("cam_") or name == "ground":
                continue
            pos = body.GetPos()
            rot = body.GetRot()
            vel = body.GetPosDt()
            aabb = body.GetTotalAABB()
            writer.writerow(
                [
                    name,
                    pos.x, pos.y, pos.z,
                    rot.e0, rot.e1, rot.e2, rot.e3,
                    aabb.min.x, aabb.min.y, aabb.min.z,
                    aabb.max.x, aabb.max.y, aabb.max.z,
                    vel.x, vel.y, vel.z,
                    0.0, 0.0, 0.0,
                ]
            )
    print(f"[placement] wrote {placement_path}")
    return placement_path


def write_contacts_csv(system, output_dir: str = ".") -> str:
    """Write active contact pairs after settling.

    OPTIONAL — not required by the review pipeline. Available for
    ad-hoc diagnostics on legacy plans.

    Returns the path to the written CSV file.
    """
    import csv

    import pychrono.core as chrono

    class _ContactReporter(chrono.ReportContactCallback):
        def __init__(self):
            super().__init__()
            self.contacts: list = []

        def OnReportContact(
            self, pA, pB, plane_coord, distance, eff_radius,
            react_forces, react_torques, contactobjA, contactobjB,
            constraint_offset,
        ):
            try:
                bodyA = contactobjA.GetBody() if contactobjA else None
                bodyB = contactobjB.GetBody() if contactobjB else None
                nameA = bodyA.GetName() if bodyA else "unknown"
                nameB = bodyB.GetName() if bodyB else "unknown"
                force_mag = (
                    react_forces.Length()
                    if hasattr(react_forces, "Length")
                    else abs(react_forces.x)
                )
                self.contacts.append((nameA, nameB, force_mag))
            except Exception:
                pass
            return True

    reporter = _ContactReporter()
    try:
        system.GetContactContainer().ReportAllContacts(reporter)
    except RuntimeError as e:
        print(f"[contacts] Warning: ReportAllContacts failed ({e}); writing empty CSV")

    os.makedirs(output_dir, exist_ok=True)
    contacts_path = os.path.join(output_dir, "scene_contacts.csv")
    with open(contacts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["body1", "body2", "force_magnitude"])
        for b1, b2, fm in reporter.contacts:
            writer.writerow([b1, b2, f"{fm:.4f}"])
    print(f"[contacts] wrote {contacts_path} ({len(reporter.contacts)} contacts)")
    return contacts_path


def write_links_csv(system, output_dir: str = ".") -> str:
    """Write every body-pair that is joint-connected via a ``ChLink*``.

    OPTIONAL — not required by the review pipeline. Originally used by
    the deterministic ``no_interpenetration`` predicate to skip pairs
    whose AABB overlap is GUARANTEED by a kinematic constraint. Still
    available for that diagnostic but no longer mandated by the
    motion-CSV review contract.

    The CSV is symmetric: (A, B) is recorded once. Validator side reads
    both orderings via ``frozenset({a, b})``.

    Returns the path to the written CSV file.
    """
    import csv

    rows: list = []
    try:
        links = list(system.GetLinks())
    except Exception as exc:
        print(f"[links] Warning: GetLinks() failed ({exc}); writing empty CSV")
        links = []

    for link in links:
        try:
            body1 = link.GetBody1() if hasattr(link, "GetBody1") else None
            body2 = link.GetBody2() if hasattr(link, "GetBody2") else None
            if body1 is None or body2 is None:
                continue
            name1 = body1.GetName() if hasattr(body1, "GetName") else ""
            name2 = body2.GetName() if hasattr(body2, "GetName") else ""
            if not name1 or not name2:
                continue
            link_type = type(link).__name__
            rows.append((name1, name2, link_type))
        except Exception:
            # ChLinkMotorRotation et al. occasionally return null bodies
            # during steady-state shutdown — skip silently.
            continue

    os.makedirs(output_dir, exist_ok=True)
    links_path = os.path.join(output_dir, "scene_links.csv")
    with open(links_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["body1", "body2", "link_type"])
        for b1, b2, lt in rows:
            writer.writerow([b1, b2, lt])
    print(f"[links] wrote {links_path} ({len(rows)} link pair(s))")
    return links_path


# ── Module exports ───────────────────────────────────────────────────────────

__all__ = [
    "AssetDescriptor",
    "add_visual_assets",
    "convex_decompose_asset",
    "ensure_convex_json",
    "add_collision_from_decomposition",
    "add_collision_via_subbodies",
    "create_asset_body",
    "load_convex_hulls",
    "transform_vertices",
    "make_contact_material",
    "box_inertia",
    "write_placement_csv",
    "write_contacts_csv",
    "write_links_csv",
]
