"""FSI scene-asset builders that keep visual / BCE / fluid sampler aligned.

Misalignment between the three layers is the most common FSI bug we see
(iter_003 of session 20260427_120708 was the canonical example: codegen
applied HR-15's ``ChFramed(0, 0, bzDim/2)`` lift while ALSO setting
``tank.SetPos(0, 0, bzDim/2)``, producing a double-shift that placed
the BCE cluster half a tank height above the visual walls).

This module collapses the three "patterns" the SKILL.md used to describe
into ONE helper function per FSI body type. The caller specifies the
target geometry in WORLD coordinates and never touches body-local
``ChFramed`` offsets — the helper picks a single internal convention and
makes the three layers coincide by construction.

Why do it this way:

  * ``CreatePointsBoxContainer((bx, by, bz), ...)`` returns a BCE point
    cluster whose nominal centre is the local origin, with ~``2 * d0``
    of wall-layer overhang outside the nominal half-extents.
  * If we always ``body.SetPos(world_centre)`` and pass identity
    ``ChFramed`` to ``AddFsiBody``, the cluster's nominal centre lands
    at the same world point as the body origin. The visual / collision
    walls, also placed in body-local coords with offsets symmetric
    around the origin, then occupy the same world AABB.
  * This single internal convention eliminates the "Pattern A vs
    Pattern B" choice and the iter_003 double-shift that came from
    mixing them.

The helpers also return the recommended fluid-sampler parameters in
WORLD coords so the caller doesn't have to re-derive offsets when
seeding particles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


_FSI_BUILT_BY: dict = {}  # body.GetIdentifier() -> helper name; set on AddBody, read by assert_fsi_bodies_unique


@dataclass
class FsiTankBuild:
    """Result of ``build_fsi_tank``.

    All AABBs are in WORLD coordinates so a downstream alignment check
    can directly compare them.
    """
    tank_body: object                       # chrono.ChBody
    sampler_box_center: object              # chrono.ChVector3d (world)
    sampler_box_halfdim: object             # chrono.ChVector3d
    visual_world_aabb: object               # chrono.ChAABB (target tank walls extent)
    bce_nominal_world_aabb: object          # chrono.ChAABB (BCE cluster centre)
    fluid_world_aabb: object                # chrono.ChAABB (sampler region)


@dataclass
class FsiFloatingBoxBuild:
    """Result of ``build_fsi_floating_box``."""
    body: object                            # chrono.ChBody
    visual_world_aabb: object               # chrono.ChAABB
    bce_nominal_world_aabb: object          # chrono.ChAABB
    expected_above_waterline_fraction: float  # for review consistency


def build_fsi_tank(
    sysMBS,
    sysSPH,
    sysFSI,
    *,
    world_extent,                            # chrono.ChAABB — the tank's interior extent in WORLD coords
    contact_material,
    wall_thickness: float = 0.05,
    bce_layers=(2, 0, -1),                   # x_walls, y_walls, z_walls (-1 = floor only, 0 = periodic)
    sampler_padding: Optional[float] = None, # default = initial_spacing
    name: str = "water_tank_boundary",
    visual_color=None,                       # chrono.ChColor or None for default
) -> FsiTankBuild:
    """Build a leak-proof FSI water tank with visual / BCE / sampler aligned.

    The caller specifies the tank's INTERIOR extent in WORLD coordinates
    (an AABB) and the helper picks an internal layout that guarantees:

      * Visual walls span exactly ``world_extent`` in world coords.
      * BCE cluster's nominal centre matches the visual centre.
        (BCE markers extend ``~2 * initial_spacing`` outward from each
        wall by design — that overhang is the wall layer thickness and
        is correct.)
      * Recommended fluid sampler box, returned via the result dataclass,
        sits inside ``world_extent`` shrunk by ``sampler_padding`` so
        particles do not initialise in contact with the walls.

    Args:
        sysMBS, sysSPH, sysFSI: the three FSI subsystems.
        world_extent: ``ChAABB`` giving the tank's INTERIOR extent in
            world coords (i.e. where you want the fluid to live).
        contact_material: ``ChContactMaterial*`` for the wall collision shapes.
        wall_thickness: thickness of the visual / collision wall slabs.
            Cosmetic only — does not affect SPH boundary resolution.
        bce_layers: ``(x_layers, y_layers, z_layers)`` passed verbatim
            to ``CreatePointsBoxContainer``. ``-1`` for floor-only (open-top
            tank, the standard FSI setup; HR-15a). ``0`` for an axis means
            no wall — used with periodic boundary conditions on that axis.
            ``2`` for both walls.
        sampler_padding: gap (in metres) between the sampler box and the
            tank walls. Defaults to ``sysSPH.GetInitialSpacing()`` so the
            first particle layer is one ``d0`` inside the wall.
        name: ``ChBody.SetName`` for the tank — used by the
            ``fluid_containment`` reviewer to find the container body.
        visual_color: optional ``ChColor`` applied to all visual shapes.

    Returns:
        ``FsiTankBuild`` — the body, the recommended sampler params, and
        the world AABBs of each layer for downstream alignment checks.
    """
    import pychrono.core as chrono

    cmin = world_extent.min
    cmax = world_extent.max
    bxDim = cmax.x - cmin.x
    byDim = cmax.y - cmin.y
    bzDim = cmax.z - cmin.z
    if bxDim <= 0 or byDim <= 0 or bzDim <= 0:
        raise ValueError(
            f"world_extent must have positive volume; got "
            f"({bxDim}, {byDim}, {bzDim})"
        )
    center = chrono.ChVector3d(
        (cmin.x + cmax.x) * 0.5,
        (cmin.y + cmax.y) * 0.5,
        (cmin.z + cmax.z) * 0.5,
    )

    if sampler_padding is None:
        try:
            sampler_padding = float(sysSPH.GetInitialSpacing())
        except Exception:
            sampler_padding = 0.05

    tank = chrono.ChBody()
    tank.SetName(name)
    tank.SetPos(center)
    tank.SetFixed(True)
    tank.EnableCollision(True)

    # Accept both ``ChVector3i`` (the form the fsi/sph skill instructs the
    # agent to pass at SKILL.md:233) and a plain tuple/list. PyChrono's
    # SWIG binding for ``ChVector3i`` does not implement ``__iter__``, so a
    # raw ``lx, ly, lz = bce_layers`` raises "cannot unpack non-iterable
    # ChVector3i object" -- a recurring iter_001 crash that has nothing to
    # do with the caller's code and everything to do with this helper
    # rejecting the exact value shape the skill prescribes.
    if hasattr(bce_layers, "x") and hasattr(bce_layers, "y") and hasattr(bce_layers, "z"):
        lx, ly, lz = bce_layers.x, bce_layers.y, bce_layers.z
    else:
        lx, ly, lz = bce_layers

    def _add_wall(half_offset: "chrono.ChVector3d",
                  size: "chrono.ChVector3d") -> None:
        frame = chrono.ChFramed(half_offset, chrono.QUNIT)
        tank.AddCollisionShape(
            chrono.ChCollisionShapeBox(contact_material, size.x, size.y, size.z),
            frame,
        )
        vshape = chrono.ChVisualShapeBox(size.x, size.y, size.z)
        if visual_color is not None:
            vshape.SetColor(visual_color)
        tank.AddVisualShape(vshape, frame)

    # Floor when z_layers != 0 (i.e. not "no floor")
    if lz != 0:
        _add_wall(
            chrono.ChVector3d(0, 0, -bzDim / 2.0),
            chrono.ChVector3d(bxDim, byDim, wall_thickness),
        )
        # Closed-top tank when z_layers > 0 — emit a top wall too.
        if lz > 0:
            _add_wall(
                chrono.ChVector3d(0, 0, +bzDim / 2.0),
                chrono.ChVector3d(bxDim, byDim, wall_thickness),
            )

    # x-walls when x_layers != 0
    if lx != 0:
        for sign in (-1, +1):
            _add_wall(
                chrono.ChVector3d(sign * bxDim / 2.0, 0, 0),
                chrono.ChVector3d(wall_thickness, byDim, bzDim),
            )

    # y-walls when y_layers != 0
    if ly != 0:
        for sign in (-1, +1):
            _add_wall(
                chrono.ChVector3d(0, sign * byDim / 2.0, 0),
                chrono.ChVector3d(bxDim, wall_thickness, bzDim),
            )

    sysMBS.AddBody(tank)
    _FSI_BUILT_BY[tank.GetIdentifier()] = "build_fsi_tank"

    # BCE cluster — centred at body origin which we set to ``center``.
    # Identity ``ChFramed`` is the ONE correct choice given our SetPos
    # convention. Mixing this with ``ChFramed(0, 0, bzDim/2)`` doubles the
    # lift — that is the iter_003 bug this helper is built to prevent.
    bce_cluster = sysSPH.CreatePointsBoxContainer(
        chrono.ChVector3d(bxDim, byDim, bzDim),
        chrono.ChVector3i(lx, ly, lz),
    )
    sysFSI.AddFsiBody(tank, bce_cluster, chrono.ChFramed(), False)

    # Recommended fluid sampler box in WORLD coords.
    sampler_center = chrono.ChVector3d(center.x, center.y, center.z)
    sampler_halfdim = chrono.ChVector3d(
        max(0.0, bxDim / 2.0 - sampler_padding),
        max(0.0, byDim / 2.0 - sampler_padding),
        max(0.0, bzDim / 2.0 - sampler_padding),
    )

    visual_world_aabb = chrono.ChAABB(
        chrono.ChVector3d(cmin.x, cmin.y, cmin.z),
        chrono.ChVector3d(cmax.x, cmax.y, cmax.z),
    )
    bce_nominal_world_aabb = chrono.ChAABB(
        chrono.ChVector3d(cmin.x, cmin.y, cmin.z),
        chrono.ChVector3d(cmax.x, cmax.y, cmax.z),
    )
    fluid_world_aabb = chrono.ChAABB(
        chrono.ChVector3d(
            sampler_center.x - sampler_halfdim.x,
            sampler_center.y - sampler_halfdim.y,
            sampler_center.z - sampler_halfdim.z,
        ),
        chrono.ChVector3d(
            sampler_center.x + sampler_halfdim.x,
            sampler_center.y + sampler_halfdim.y,
            sampler_center.z + sampler_halfdim.z,
        ),
    )

    return FsiTankBuild(
        tank_body=tank,
        sampler_box_center=sampler_center,
        sampler_box_halfdim=sampler_halfdim,
        visual_world_aabb=visual_world_aabb,
        bce_nominal_world_aabb=bce_nominal_world_aabb,
        fluid_world_aabb=fluid_world_aabb,
    )


def build_fsi_floating_box(
    sysMBS,
    sysSPH,
    sysFSI,
    *,
    world_center,                            # chrono.ChVector3d — body's centre in world coords
    size,                                    # chrono.ChVector3d — full extents (NOT half) on x / y / z
    density: float,
    contact_material,
    fluid_density: float = 1000.0,
    name: str = "floating_body",
    visual_color=None,
) -> FsiFloatingBoxBuild:
    """Build a floating box with visual + BCE aligned at the body origin.

    Unlike the tank container, a floating body's BCE is generated by
    ``CreatePointsBoxInterior(size)`` — its cluster is naturally centred
    on the body's local origin, so ``AddFsiBody(body, bce, ChFramed(), False)``
    with identity offset is correct (no lift needed). The common bug
    here is passing HALF-extents to ``CreatePointsBoxInterior`` (which
    expects full ``size``), making the BCE half the size of the visual.
    This helper takes ``size`` as full extents and uses the same value
    for both the visual shape and the BCE call, eliminating that mistake.

    Returns the body and the world AABB of the visual / BCE so a
    downstream alignment check can verify they coincide.
    """
    import pychrono.core as chrono

    body = chrono.ChBody()
    body.SetName(name)
    body.SetPos(world_center)
    body.SetFixed(False)
    body.EnableCollision(True)

    hx, hy, hz = size.x * 0.5, size.y * 0.5, size.z * 0.5
    volume = size.x * size.y * size.z
    mass = density * volume
    body.SetMass(mass)
    # Box inertia about centroid: (mass / 12) * (b² + c²) etc.
    Ixx = mass * (size.y ** 2 + size.z ** 2) / 12.0
    Iyy = mass * (size.x ** 2 + size.z ** 2) / 12.0
    Izz = mass * (size.x ** 2 + size.y ** 2) / 12.0
    body.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))

    body.AddCollisionShape(
        chrono.ChCollisionShapeBox(contact_material, size.x, size.y, size.z),
        chrono.ChFramed(),
    )
    vshape = chrono.ChVisualShapeBox(size.x, size.y, size.z)
    if visual_color is not None:
        vshape.SetColor(visual_color)
    body.AddVisualShape(vshape, chrono.ChFramed())

    sysMBS.AddBody(body)
    _FSI_BUILT_BY[body.GetIdentifier()] = "build_fsi_floating_box"

    # BCE — identity offset is correct because cluster is naturally
    # centred at the body's local origin.
    bce_cluster = sysSPH.CreatePointsBoxInterior(chrono.ChVector3d(size.x, size.y, size.z))
    sysFSI.AddFsiBody(body, bce_cluster, chrono.ChFramed(), False)

    visual_world_aabb = chrono.ChAABB(
        chrono.ChVector3d(world_center.x - hx, world_center.y - hy, world_center.z - hz),
        chrono.ChVector3d(world_center.x + hx, world_center.y + hy, world_center.z + hz),
    )
    bce_nominal_world_aabb = chrono.ChAABB(
        chrono.ChVector3d(world_center.x - hx, world_center.y - hy, world_center.z - hz),
        chrono.ChVector3d(world_center.x + hx, world_center.y + hy, world_center.z + hz),
    )

    expected_above = max(0.0, 1.0 - density / fluid_density)

    return FsiFloatingBoxBuild(
        body=body,
        visual_world_aabb=visual_world_aabb,
        bce_nominal_world_aabb=bce_nominal_world_aabb,
        expected_above_waterline_fraction=expected_above,
    )


def build_fsi_vehicle_visualizer(
    sysMBS,
    sysSPH,
    sysFSI,
    *,
    vehicle,                                  # veh.WheeledVehicle (already Initialized)
    sph_visualization,                        # fsi.ChSphVisualizationVSG instance
    window_title: str,
    window_size: Tuple[int, int] = (1280, 720),
    chassis_vis: str = "MESH",                # "MESH" | "PRIMITIVES" | "NONE"
    wheel_vis: str = "MESH",
    suspension_vis: str = "PRIMITIVES",
    steering_vis: str = "PRIMITIVES",
    enable_fluid_markers: bool = True,
    enable_boundary_markers: bool = False,
    enable_rigid_body_markers: bool = False,
):
    """Build a fully-wired VSG visualizer for an FSI wheeled-vehicle scene.

    **Caller MUST construct ``sph_visualization`` (a
    ``fsi.ChSphVisualizationVSG`` instance) and pass it in. This function
    does NOT create one for you, and does NOT return one to you — it
    returns a single ``vis`` value.** Passing ``sph_visualization=None``
    raises ``TypeError`` immediately (was iter_007 of
    session_20260506_*: agent wrote a comment claiming the function
    "creates its own visFSI internally" and then unpacked the return as
    a tuple, both of which were hallucinations).

    Encapsulates the 8+ ordered calls that must be assembled correctly to
    render the chassis + wheels + SPH particles + BCE markers in the same
    mp4. Hand-rolling this sequence is the single most common cause of
    "wheels visible but no chassis" / "no MBS bodies visible" /
    "vehicle invisible" failures in FSI scenes (session_20260428_164422
    spent 12 iterations on this), because four SKILL files describe the
    sequence in different orders and one of them
    (``veh/wheeled_vehicle``) omits ``AttachSystem(sysMBS)``, which is
    mandatory in FSI scenes.

    Returns a ready-to-use ``vis`` (already ``Initialize()``-d). Caller
    still applies camera lock + recording setup separately:

    .. code-block:: python

        import pychrono.fsi as fsi
        # 1. Caller constructs the SPH visualizer and configures color first.
        visFSI = fsi.ChSphVisualizationVSG(sysFSI)
        visFSI.SetSPHColorCallback(fsi.ParticleVelocityColorCallback(0, 5.0))
        # 2. Then hand it to this builder. Returns ONLY vis (single value).
        vis = build_fsi_vehicle_visualizer(
            sysMBS, sysSPH, sysFSI,
            vehicle=polaris,
            sph_visualization=visFSI,
            window_title="...",
        )
        lock_side_camera(vis, cam_pos, target_pos)
        finalize = setup_vsg_recording(vis, "cam/vsg.mp4", fps=50.0)

    Args:
        sysMBS, sysSPH, sysFSI: the three FSI subsystems.
        vehicle: a ``veh.WheeledVehicle`` (or ``HMMWV_Full`` etc.) whose
            ``Initialize()`` has already been called.
        sph_visualization: the ``fsi.ChSphVisualizationVSG`` instance built
            from ``sysFSI`` (caller constructs it; we only enable markers and
            attach as plugin).
        window_title: VSG window title string.
        window_size: ``(width, height)`` in pixels.
        chassis_vis / wheel_vis / suspension_vis / steering_vis: visualization
            type per vehicle component. Default of ``"NONE"`` for chassis/wheels
            is the PyChrono framework default — without explicit ``"MESH"`` the
            chassis is invisible (this is the iteration_008 bug).
        enable_fluid_markers: render the SPH fluid particles themselves
            (the blue water cloud). **Default True** — without these the
            water is invisible in the mp4. Only the BCE diagnostic
            overlays (`enable_boundary_markers`, `enable_rigid_body_markers`)
            are off by default, since the dense green BCE dot grid on
            tank walls and floating bodies was previously perceived by
            the VLM as the only thing in the scene
            (session_20260429_060447 "tank invisible because only BCE
            markers visible"). Pass `enable_boundary_markers=True` or
            `enable_rigid_body_markers=True` only when explicitly
            debugging BCE alignment for one run.

    Returns:
        ``veh.ChWheeledVehicleVisualSystemVSG`` with all attachments and
        ``Initialize()`` already called. Caller proceeds directly to camera
        lock + recording setup.
    """
    # Hard reject the iter_007 hallucination — the LLM has wrongly believed
    # this function constructs visFSI internally; assert otherwise upfront
    # so the failure is clearly attributed to the bad call site, not to a
    # downstream "AttributeError on None" stack 30 lines later.
    if sph_visualization is None:
        raise TypeError(
            "build_fsi_vehicle_visualizer: sph_visualization=None is not "
            "supported. Caller MUST construct a fsi.ChSphVisualizationVSG "
            "(typically `visFSI = fsi.ChSphVisualizationVSG(sysFSI)` plus "
            "`visFSI.SetSPHColorCallback(...)`) and pass it as "
            "sph_visualization. This function returns a single vis (NOT a "
            "(vis, visFSI) tuple) — see docstring example."
        )

    import pychrono.core as chrono
    import pychrono.vehicle as veh

    _vis_type_lookup = {
        "MESH": chrono.VisualizationType_MESH,
        "PRIMITIVES": chrono.VisualizationType_PRIMITIVES,
        "NONE": chrono.VisualizationType_NONE,
    }

    def _resolve(name: str):
        try:
            return _vis_type_lookup[name.upper()]
        except KeyError as exc:
            raise ValueError(
                f"unknown visualization type {name!r}; "
                f"choose one of {sorted(_vis_type_lookup)}"
            ) from exc

    # 1. Per-component visualization types ON THE VEHICLE.
    # The PyChrono default is VisualizationType_NONE for chassis and wheels,
    # so without these calls the chassis body is invisible and the VLM sees
    # only floating BCE wheel markers. This is precisely the iteration_008
    # symptom from session_20260428_164422.
    vehicle.SetChassisVisualizationType(_resolve(chassis_vis))
    vehicle.SetWheelVisualizationType(_resolve(wheel_vis))
    vehicle.SetSuspensionVisualizationType(_resolve(suspension_vis))
    vehicle.SetSteeringVisualizationType(_resolve(steering_vis))

    # 2. SPH plugin marker overlays (HR-14: required for VLM-reviewed runs).
    sph_visualization.EnableFluidMarkers(enable_fluid_markers)
    sph_visualization.EnableBoundaryMarkers(enable_boundary_markers)
    sph_visualization.EnableRigidBodyMarkers(enable_rigid_body_markers)
    sph_visualization.EnableFlexBodyMarkers(False)

    # 3. Vehicle-aware visualizer; attach in the order required by HR-5a:
    #    AttachVehicle, then AttachSystem(sysMBS), then AttachPlugin.
    # Without AttachSystem(sysMBS) every non-vehicle MBS body (tank walls,
    # platforms, floating plate) is invisible — the VLM sees only the
    # vehicle and SPH particles.
    vis = veh.ChWheeledVehicleVisualSystemVSG()
    vis.AttachVehicle(vehicle)
    vis.AttachSystem(sysMBS)
    vis.AttachPlugin(sph_visualization)

    # 4. Window + camera config (must be set before Initialize).
    vis.SetWindowSize(*window_size)
    vis.SetWindowTitle(window_title)
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)

    # 5. Initialize last (HR-1a: vis.Initialize() comes after sysFSI.Initialize()
    # and after every other AttachX/SetX call).
    vis.Initialize()

    return vis


class SphMotionLogger:
    """Per-step SPH motion logger that emits a real velocity, not 0,0,0.

    Why this exists:
        PyChrono's Python FSI binding does not expose a direct
        ``GetParticleVelocities()`` API on ``ChFsiFluidSystemSPH`` — only
        ``GetParticlePositions()`` and on-disk ``SaveParticleData(path)``.
        Codegen agents have repeatedly written motion_log.csv rows with
        velocity columns hardcoded to ``0,0,0`` (because the Get method
        they imagine doesn't exist), which then poisons reviewer logic
        ("FSI coupling broken — peak SPH velocity = 0") even when the
        VLM video clearly shows the water splashing.

    What this does:
        Tracks the SPH center-of-mass between successive ``write()`` calls
        and emits ``(com_x, com_y, com_z, com_vx, com_vy, com_vz)`` via
        finite differences. The COM-velocity row is sufficient evidence
        of fluid motion for the reviewer's "did SPH actually move?"
        check; it does NOT require stable per-particle ordering across
        frames (which the SPH solver does not guarantee).

    Usage:
        sph_log = SphMotionLogger(name="sph_water")
        ...
        def fsi_step(sim_time, dt):
            sysFSI.DoStepDynamics(STEP_SIZE)
            if step_idx % LOG_EVERY_N == 0:
                sph_log.write(motion_csv, sim_time, sysSPH)
    """

    __slots__ = ("name", "_prev_com", "_prev_t")

    def __init__(self, name: str = "sph_water"):
        self.name = name
        self._prev_com: Optional[Tuple[float, float, float]] = None
        self._prev_t: Optional[float] = None

    def write(self, motion_csv, sim_time: float, sysSPH) -> None:
        """Append one row to ``motion_csv`` and update internal state.

        ``motion_csv`` is any object with a ``write(str)`` method (file
        handle, StringIO, etc.). Emits the same column layout the rest
        of motion_log.csv uses:

            time,body_name,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
        """
        positions = sysSPH.GetParticlePositions()
        n = len(positions)
        if n == 0:
            # Empty SPH cloud: emit a zero-pos row so downstream parsers
            # don't drop the body from the log entirely. Velocity is also
            # 0 here, but that's a true 0 (no fluid), not a lie.
            motion_csv.write(
                f"{sim_time:.6f},{self.name},0.000000,0.000000,0.000000,0.000000,0.000000,0.000000\n"
            )
            self._prev_com = None
            self._prev_t = float(sim_time)
            return

        cx = cy = cz = 0.0
        for p in positions:
            cx += p.x
            cy += p.y
            cz += p.z
        cx /= n
        cy /= n
        cz /= n

        if (
            self._prev_com is None
            or self._prev_t is None
            or sim_time <= self._prev_t
        ):
            vx = vy = vz = 0.0
        else:
            dt = sim_time - self._prev_t
            vx = (cx - self._prev_com[0]) / dt
            vy = (cy - self._prev_com[1]) / dt
            vz = (cz - self._prev_com[2]) / dt

        motion_csv.write(
            f"{sim_time:.6f},{self.name},"
            f"{cx:.6f},{cy:.6f},{cz:.6f},"
            f"{vx:.6f},{vy:.6f},{vz:.6f}\n"
        )

        self._prev_com = (cx, cy, cz)
        self._prev_t = float(sim_time)


def write_fsi_alignment_csv(
    builds,                                  # iterable of FsiTankBuild / FsiFloatingBoxBuild
    output_dir: str = ".",
) -> str:
    """Write ``fsi_alignment.csv`` for downstream review.

    One row per layer per body: ``body_name, layer, world_min_x/y/z,
    world_max_x/y/z``. The ``fluid_containment`` reviewer (or a future
    ``alignment`` predicate) can read this to verify the visual and BCE
    AABBs match within tolerance — catching any future regression of the
    iter_003 double-shift bug at validation time, not just at codegen
    time.
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fsi_alignment.csv")
    rows = []
    for b in builds:
        body_name = (
            b.tank_body.GetName() if hasattr(b, "tank_body") else b.body.GetName()
        )
        for layer_name, aabb in (
            ("visual", b.visual_world_aabb),
            ("bce_nominal", b.bce_nominal_world_aabb),
            *(((("fluid_sampler", b.fluid_world_aabb)),) if hasattr(b, "fluid_world_aabb") else ()),
        ):
            rows.append((
                body_name, layer_name,
                aabb.min.x, aabb.min.y, aabb.min.z,
                aabb.max.x, aabb.max.y, aabb.max.z,
            ))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "body_name", "layer",
            "world_min_x", "world_min_y", "world_min_z",
            "world_max_x", "world_max_y", "world_max_z",
        ])
        for row in rows:
            w.writerow([row[0], row[1]] + [f"{v:.6f}" for v in row[2:]])
    print(f"[fsi_alignment] wrote {path} ({len(rows)} rows)")
    return path


def assert_fsi_bodies_unique(sysMBS) -> None:
    """Detect double-AddBody on bodies built by ``build_fsi_*`` helpers.

    The ``build_fsi_tank`` / ``build_fsi_floating_box`` helpers register
    their body in ``sysMBS`` internally and pair that single registration
    with an ``AddFsiBody`` call binding the BCE cluster to that exact
    reference. If the caller then *also* calls ``sysMBS.AddBody(build.tank_body)``
    — a defensive habit carried over from MBS code where every body needs
    an explicit ``AddBody`` — PyChrono does not deduplicate: the body
    appears twice in ``sysMBS.GetBodies()`` and ``body.GetSystem()`` is
    silently corrupted, which desyncs the FSI BCE registration. The first
    visible symptom is fluid leaking through tank walls (the BCE markers
    no longer track the body) — many iterations downstream from the cause.

    Call this immediately before ``sysFSI.Initialize()``. The check is
    O(N) over ``sysMBS.GetBodies()``; cheap enough to leave in production.

    Raises ``RuntimeError`` listing every offender plus the canonical fix.
    """
    # PyChrono SWIG returns a fresh Python proxy each call, so id() differs
    # for the same C++ body. Use ChObj.GetIdentifier() — the C++-side counter
    # assigned at body construction — to detect true duplicates. Python
    # attributes set on the SWIG proxy don't survive the GetBodies()
    # roundtrip, so the build-helper attribution lives in module-level
    # _FSI_BUILT_BY keyed by the same identifier.
    seen: dict = {}
    duplicates = []
    for b in sysMBS.GetBodies():
        key = b.GetIdentifier()
        if key in seen:
            built_by = _FSI_BUILT_BY.get(key)
            try:
                name = b.GetName()
            except Exception:
                name = "<unknown>"
            duplicates.append((name, built_by))
        else:
            seen[key] = b
    if not duplicates:
        return
    lines = ["FSI body registered twice in sysMBS — this desyncs BCE markers and causes fluid leak through walls."]
    for name, built_by in duplicates:
        if built_by:
            lines.append(
                f"  - body {name!r} was created by {built_by}() (which already called sysMBS.AddBody internally); "
                f"remove your second sysMBS.AddBody({name}) plus any redundant EnableCollision / SetFixed on it."
            )
        else:
            lines.append(f"  - body {name!r} appears twice in sysMBS.GetBodies(); remove the duplicate AddBody call.")
    raise RuntimeError("\n".join(lines))


__all__ = [
    "FsiTankBuild",
    "FsiFloatingBoxBuild",
    "build_fsi_tank",
    "build_fsi_floating_box",
    "build_fsi_vehicle_visualizer",
    "write_fsi_alignment_csv",
    "assert_fsi_bodies_unique",
]
