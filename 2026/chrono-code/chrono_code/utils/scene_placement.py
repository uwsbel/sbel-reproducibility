"""XY footprint overlap avoidance for scene asset placement.

Used by generated simulation scripts (see the
``scene/custom_assets_scene_convex_decomp`` skill) to guarantee that no
two placed assets share overlapping AABB footprints on the XY plane.
All overlap logic should go through :func:`find_non_overlapping_pos` and
:class:`FootprintRegistry` rather than being reimplemented inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

DEFAULT_PLACEMENT_MARGIN = 0.03  # metres of slack between neighboring AABBs


Footprint = Tuple[float, float, float, float]  # (cx, cy, half_x, half_y)


def footprints_overlap(
    ax: float, ay: float, ahx: float, ahy: float,
    bx: float, by: float, bhx: float, bhy: float,
    margin: float = 0.0,
) -> bool:
    """True iff two AABB footprints overlap (with optional extra margin)."""
    return (
        abs(ax - bx) < (ahx + bhx + margin)
        and abs(ay - by) < (ahy + bhy + margin)
    )


def has_any_overlap(
    occupied: List[Footprint],
    cx: float, cy: float, hx: float, hy: float,
    margin: float = 0.0,
) -> bool:
    return any(
        footprints_overlap(cx, cy, hx, hy, ocx, ocy, ohx, ohy, margin=margin)
        for (ocx, ocy, ohx, ohy) in occupied
    )


def find_non_overlapping_pos(
    occupied: List[Footprint],
    size_x: float,
    size_y: float,
    preferred_x: float = 0.0,
    preferred_y: float = 0.0,
    room_half: float = 4.0,
    margin: float = DEFAULT_PLACEMENT_MARGIN,
    max_ring: int = 24,
    bounds_xy: Tuple[float, float, float, float] | None = None,
) -> Tuple[float, float]:
    """Return a non-overlapping XY centre for a ``size_x * size_y`` footprint.

    The preferred position is tried first. If it collides, the search tries
    slots adjacent to every already-placed footprint, and finally falls back
    to a growing grid search around the preferred position. All candidates
    are clamped to either a symmetric square (``room_half``) or, when
    ``bounds_xy=(xmin, xmax, ymin, ymax)`` is supplied, that explicit
    rectangle. The rectangle path is used by :class:`SurfaceStack` to keep
    desktop props inside their parent's top-surface footprint.

    Raises ``RuntimeError`` if no slot can be found — the caller should
    either enlarge the bounds or reduce ``margin``.
    """
    new_hx = size_x * 0.5
    new_hy = size_y * 0.5

    if bounds_xy is not None:
        xmin, xmax, ymin, ymax = bounds_xy

        def _in_room(cx: float, cy: float) -> bool:
            return (
                xmin + new_hx <= cx <= xmax - new_hx
                and ymin + new_hy <= cy <= ymax - new_hy
            )
    else:
        def _in_room(cx: float, cy: float) -> bool:
            return abs(cx) + new_hx < room_half and abs(cy) + new_hy < room_half

    if _in_room(preferred_x, preferred_y) and not has_any_overlap(
        occupied, preferred_x, preferred_y, new_hx, new_hy, margin=margin
    ):
        return preferred_x, preferred_y

    candidates: List[Tuple[float, float]] = []
    for pcx, pcy, phw, phh in occupied:
        candidates.extend([
            (pcx + phw + new_hx + margin, pcy),
            (pcx - phw - new_hx - margin, pcy),
            (pcx, pcy + phh + new_hy + margin),
            (pcx, pcy - phh - new_hy - margin),
        ])
    for cx, cy in candidates:
        if _in_room(cx, cy) and not has_any_overlap(
            occupied, cx, cy, new_hx, new_hy, margin=margin
        ):
            return cx, cy

    step_x = max(size_x + margin, 0.05)
    step_y = max(size_y + margin, 0.05)
    for ring in range(1, max_ring + 1):
        for ix in range(-ring, ring + 1):
            for iy in range(-ring, ring + 1):
                if max(abs(ix), abs(iy)) != ring:
                    continue
                cx = preferred_x + ix * step_x
                cy = preferred_y + iy * step_y
                if _in_room(cx, cy) and not has_any_overlap(
                    occupied, cx, cy, new_hx, new_hy, margin=margin
                ):
                    return cx, cy

    bounds_desc = (
        f"bounds_xy={bounds_xy}" if bounds_xy is not None
        else f"room_half={room_half}"
    )
    raise RuntimeError(
        f"Failed to find non-overlapping position for footprint "
        f"({size_x:.3f}, {size_y:.3f}) within {bounds_desc} "
        f"after {max_ring} rings."
    )


@dataclass
class FootprintRegistry:
    """Track placed footprints and resolve new positions in one call.

    Typical usage inside a generated simulation script::

        from chrono_code.utils.scene_placement import FootprintRegistry

        registry = FootprintRegistry(room_half=4.0, margin=0.03)

        for cfg in ASSET_CONFIGS:
            px, py, pz = cfg["position"]
            body, size = create_asset_body(...)       # build mesh/hulls first
            nx, ny = registry.place(size[0], size[1], px, py)
            body.SetPos(chrono.ChVector3d(nx, ny, body.GetPos().z))

    Only the XY position is adjusted; Z (support height) is whatever the
    caller has already computed from visual/collision bbox bottoms.
    """

    room_half: float = 4.0
    margin: float = DEFAULT_PLACEMENT_MARGIN
    occupied: List[Footprint] = field(default_factory=list)

    def place(
        self,
        size_x: float,
        size_y: float,
        preferred_x: float = 0.0,
        preferred_y: float = 0.0,
    ) -> Tuple[float, float]:
        cx, cy = find_non_overlapping_pos(
            self.occupied,
            size_x,
            size_y,
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            room_half=self.room_half,
            margin=self.margin,
        )
        self.occupied.append((cx, cy, size_x * 0.5, size_y * 0.5))
        return cx, cy

    def place_body(
        self,
        body,
        preferred_x: float,
        preferred_y: float,
        *,
        use_collision_aabb: bool = True,
    ) -> Tuple[float, float]:
        """Resolve XY position for an already-built Chrono body.

        This is the recommended entry point: it derives the footprint from
        the body's actual collision geometry (convex hulls) rather than from
        the loose visual AABB, so props with big visual envelopes but small
        contact volumes (e.g. a monitor with a thin stand) don't get phantom
        collisions with nearby assets.

        The body's Z position is preserved untouched — this routine only
        rewrites X and Y.

        Parameters
        ----------
        body : pychrono.ChBody
            Body whose position should be resolved. Must already have its
            collision (and optionally visual) shapes attached.
        preferred_x, preferred_y : float
            Desired world XY centre. Used as-is if it clears every
            previously-registered footprint with the registry's margin.
        use_collision_aabb : bool
            If True (default) and the body exposes a collision AABB, use it.
            Otherwise fall back to the total (visual ∪ collision) AABB.

        Returns
        -------
        (cx, cy) : tuple of float
            The final world XY centre. The body has been moved there, and
            its footprint has been appended to ``self.occupied``.
        """
        import pychrono.core as chrono

        aabb = None
        if use_collision_aabb:
            coll_model = body.GetCollisionModel()
            if coll_model is not None:
                try:
                    aabb = coll_model.GetBoundingBox()
                except Exception:
                    aabb = None
        if aabb is None:
            aabb = body.GetTotalAABB()

        size_x = float(aabb.max.x - aabb.min.x)
        size_y = float(aabb.max.y - aabb.min.y)
        # Collision AABB is reported in world coordinates for a body already
        # at its spawn position. Subtract the current body centre to convert
        # into body-local offsets so the registry can reason about
        # "preferred_x" cleanly, then re-apply that offset when writing back.
        pos = body.GetPos()
        centre_offset_x = 0.5 * (aabb.max.x + aabb.min.x) - pos.x
        centre_offset_y = 0.5 * (aabb.max.y + aabb.min.y) - pos.y

        cx, cy = find_non_overlapping_pos(
            self.occupied,
            size_x,
            size_y,
            preferred_x=preferred_x + centre_offset_x,
            preferred_y=preferred_y + centre_offset_y,
            room_half=self.room_half,
            margin=self.margin,
        )
        self.occupied.append((cx, cy, size_x * 0.5, size_y * 0.5))

        new_body_x = cx - centre_offset_x
        new_body_y = cy - centre_offset_y
        body.SetPos(chrono.ChVector3d(new_body_x, new_body_y, pos.z))
        return new_body_x, new_body_y

    def register(self, cx: float, cy: float, size_x: float, size_y: float) -> None:
        """Register an already-placed footprint (no overlap search)."""
        self.occupied.append((cx, cy, size_x * 0.5, size_y * 0.5))

    def reset(self) -> None:
        self.occupied.clear()


@dataclass
class SurfaceStack:
    """Sub-registry for props stacked on top of a parent body's surface.

    Use for desktop / shelf / counter props whose XY *must* stay inside
    the parent's top-surface footprint while still avoiding overlap with
    sibling props on the same surface. Independent from the floor
    :class:`FootprintRegistry` — stacked props **must not** be sent
    through the floor registry, which would shove them off the parent
    laterally because it has no notion of "inside this rectangle only".

    Typical usage::

        from chrono_code.utils.scene_placement import (
            FootprintRegistry, SurfaceStack,
        )

        floor = FootprintRegistry(room_half=4.0)
        # ... place desk through floor.place(...) first, so desk has its
        # final world XY before we read its AABB ...

        desk_top = SurfaceStack.from_body("computer_table", desk_body)

        for cfg in DESKTOP_PROPS:                       # monitor, laptop, ...
            body, tsize = create_asset_body(
                system, ..., position=(cfg["x"], cfg["y"],
                                       desk_top.surface_top_z + 0.005), ...,
            )
            nx, ny = desk_top.place(
                size_x=tsize[0], size_y=tsize[1],
                preferred_x=cfg["x"], preferred_y=cfg["y"],
            )
            body.SetPos(chrono.ChVector3d(nx, ny, body.GetPos().z))

    On overflow (a child won't fit on the surface) :meth:`place` raises a
    descriptive ``RuntimeError`` so the failure is visible at codegen
    time instead of silently producing a fall-off-the-edge scene.
    """

    parent_name: str
    surface_top_z: float
    bounds_xy: Tuple[float, float, float, float]   # (xmin, xmax, ymin, ymax)
    margin: float = DEFAULT_PLACEMENT_MARGIN
    occupied: List[Footprint] = field(default_factory=list)

    @classmethod
    def from_body(
        cls,
        parent_name: str,
        parent_body,
        *,
        margin: float = DEFAULT_PLACEMENT_MARGIN,
        edge_inset: float = 0.02,
    ) -> "SurfaceStack":
        """Build a :class:`SurfaceStack` from a parent body's total AABB.

        ``edge_inset`` (metres) shrinks the usable XY rectangle by that
        amount on every side so children don't end up flush with the
        parent's edge — where collision envelope or contact margin would
        otherwise let them roll off as soon as the sim starts.
        """
        aabb = parent_body.GetTotalAABB()
        return cls(
            parent_name=parent_name,
            surface_top_z=float(aabb.max.z),
            bounds_xy=(
                float(aabb.min.x) + edge_inset,
                float(aabb.max.x) - edge_inset,
                float(aabb.min.y) + edge_inset,
                float(aabb.max.y) - edge_inset,
            ),
            margin=margin,
        )

    def place(
        self,
        size_x: float,
        size_y: float,
        preferred_x: float = 0.0,
        preferred_y: float = 0.0,
    ) -> Tuple[float, float]:
        """Reserve a non-overlapping XY slot inside the parent's surface.

        Raises ``RuntimeError`` (with a message naming ``parent_name``)
        if the child can't fit on the surface alongside its siblings.
        """
        try:
            cx, cy = find_non_overlapping_pos(
                self.occupied,
                size_x,
                size_y,
                preferred_x=preferred_x,
                preferred_y=preferred_y,
                margin=self.margin,
                bounds_xy=self.bounds_xy,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"SurfaceStack[{self.parent_name}] cannot fit footprint "
                f"({size_x:.3f} x {size_y:.3f}) on top surface "
                f"{self.bounds_xy}: {e}"
            ) from e
        self.occupied.append((cx, cy, size_x * 0.5, size_y * 0.5))
        return cx, cy


__all__ = [
    "DEFAULT_PLACEMENT_MARGIN",
    "Footprint",
    "footprints_overlap",
    "has_any_overlap",
    "find_non_overlapping_pos",
    "FootprintRegistry",
    "SurfaceStack",
]
