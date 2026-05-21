---
name: topology
description: Joint topology, kinematic constraints, and elastic links for MBS. Covers revolute, prismatic, motors, springs, axis-aware joint frames, general revolute hinge alignment (local +Z to physical axis) plus special planar cases, and correct topology for mechanisms with ground pivots and fixed guides.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

# Skill: MBS Topology (Joints, Motors, Springs)

## Purpose

Connect bodies with kinematic constraints, motors, and springs without mixing up topology or frame semantics. The core rule is: identify the physical degree-of-freedom axis first, then choose joint type and frame orientation so the joint axis matches that physical intent.

For quaternion and rotation construction details, see `../quaternions/SKILL.md`. For planar XY body-orientation conventions, see `../body_creation/SKILL.md`.

## When to Use

- Two bodies must move relative to each other in a constrained way
- A body rotates about a fixed pivot
- A body slides along a fixed guide
- A linkage combines pivots, guides, motors, or springs

---

# Part 1: Topology First

### Rule 1: Decide the physical DOF before choosing the joint

For every connection, answer these questions in order:

1. What relative motion is physically allowed: rotation, translation, or neither?
2. About or along which axis does that motion occur?
3. Is the constraint attached to ground or to another moving body?

Then choose the joint:

- **Revolute**: one relative rotation about a hinge axis
- **Prismatic**: one relative translation along a guide axis
- **Motor**: actuation only; it does not replace the geometric constraint

Do not infer joint type from a copied code pattern. Infer it from the mechanism.

### Rule 2: Pivot + Motor needs BOTH links

When a body rotates about a fixed ground pivot and is actuated:

- Add a **revolute** for the geometric hinge
- Add a **motor** for the actuation

```python
joint_pivot = chrono.ChLinkLockRevolute()
frame_origin = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
joint_pivot.Initialize(rotating_body, ground, frame_origin)
sys.AddLink(joint_pivot)

motor = chrono.ChLinkMotorRotationTorque()
motor.Initialize(rotating_body, ground, frame_origin)
motor.SetTorqueFunction(chrono.ChFunctionSine(amplitude, frequency))
sys.AddLink(motor)
```

```python
# WRONG: motor only
motor.Initialize(rotating_body, ground, chrono.ChFramed(chrono.ChVector3d(0, 0, 0)))
sys.AddLink(motor)
```

### Rule 3: Fixed guide means prismatic to GROUND

If a body slides along a fixed rail, the prismatic belongs between the **slider** and **ground**, not between the slider and another moving body.

```python
guide_axis = chrono.ChVector3d(1, 0, 0)  # example only
guide_pos = chrono.ChVector3d(guide_x, guide_y, guide_z)

frame_prismatic = chrono.ChFramed()
frame_prismatic.SetPos(guide_pos)
frame_prismatic.SetRot(guide_rotation_that_maps_frame_Z_to_guide_axis)

joint_slider_ground = chrono.ChLinkLockPrismatic()
joint_slider_ground.Initialize(slider, ground, frame_prismatic)
sys.AddLink(joint_slider_ground)
```

### Rule 4: Fixed-guide rod-slider linkage uses revolute at rod-slider

When a connecting rod attaches to a slider that already moves on a fixed guide:

- **Rod-Slider** = Revolute
- **Slider-Ground** = Prismatic

Do not replace the rod-slider pin with a prismatic. That changes the mechanism.

### Rule 5: Team convention for revolute links

Use `chrono.ChLinkLockRevolute()` as the default revolute implementation for these MBS skills and demos.

```python
# CORRECT
joint = chrono.ChLinkLockRevolute()
```

```python
# WRONG for this skill family
joint = chrono.ChLinkRevolute()
```

### Rule 6: Fixed-guide slider linkage checklist

Before finalizing a crank-slider or similar fixed-guide mechanism, confirm all of these are present:

- rotating body-ground revolute at the pivot
- rotating body-ground motor if actuated
- rod-rotating body revolute
- rod-slider revolute
- slider-ground prismatic

---

# Part 2: Frame Semantics

### Rule 7: Revolute and prismatic frames are NOT interchangeable

Do not reuse a frame rotation from one joint type just because the point is the same.

- **Revolute** frame rotation defines the hinge axis
- **Prismatic** frame rotation defines the sliding axis

If you change a joint from prismatic to revolute, recompute the frame orientation from the hinge axis requirement. Do not keep the old guide-axis rotation by default.

### Rule 8: Revolute frame must align to the intended hinge axis

**Engine convention:** `ChLinkLockRevolute` uses the joint `ChFramed` **local +Z** as the hinge axis (relative rotation is about that axis). The **general rule** is independent of gravity or world axes: choose a quaternion `q` for `ChFramed(pivot_pos, q)` such that, in **world** coordinates, **local +Z** after the frame rotation coincides with the **physical** hinge direction (a unit vector `axis_hat`).

**General case (always start here):**

1. Define the physical hinge as a unit vector `axis_hat` in the world frame (from the mechanism geometry, not from a copied code snippet).
2. Find a rotation `q_align` that maps joint **local +Z** onto `axis_hat` (same geometric problem as aligning a prismatic’s sliding axis—only the physical meaning differs). Use `chrono.QuatFromAngleAxis` when the axis is a simple rotation from +Z, combine `chrono.Q_ROTATE_*` helpers when they apply, or build a rotation from two vectors / axis-angle as in `../quaternions/SKILL.md`.
3. `joint.Initialize(body_a, body_b, chrono.ChFramed(pivot_pos, q_align))`.

```python
# General pattern: q_align maps local +Z onto the physical hinge direction in world frame
q_align = rotation_z_to_axis_hat  # construct from axis_hat; see ../quaternions/SKILL.md
joint = chrono.ChLinkLockRevolute()
joint.Initialize(body_a, body_b, chrono.ChFramed(pivot_pos, q_align))
sys.AddLink(joint)
```

**Coordinate consistency checklist:**

1. Fix the **motion plane** (e.g. XY or XZ) and **gravity** (e.g. −Y or −Z).
2. For **planar** swing, the hinge axis must be **perpendicular to that plane** (in-plane motion ⟺ hinge along the plane normal).
3. Map **local +Z** to that normal using the general rule above. Do not reuse a prismatic frame or copy `Q_ROTATE_*` without verifying the hinge.

**Special cases (reminders):**

| Situation | Physical hinge axis (typical) | Frame rotation (typical) |
| --- | --- | --- |
| Identity / default frame | World **+Z** | `chrono.QUNIT` (local +Z = world +Z) |
| Planar **XY** in-world, gravity **−Y**, in-plane motion | World **+Z** (normal to XY) | `chrono.QUNIT` is common in demos |
| Planar **XZ** in-world, gravity **−Z**, in-plane swing | World **±Y** (normal to XZ) | **Not** `QUNIT`; e.g. `QuatFromAngleAxis(chrono.CH_PI_2, chrono.VECT_X)` so local +Z aligns with **Y** |
| Hinge along world **X** or **Y** explicitly | **±X** or **±Y** | `QuatFromAngleAxis` or `Q_ROTATE_*` from `../quaternions/SKILL.md`—never assume `QUNIT` |
| Tilted or non-cardinal hinge | Arbitrary `axis_hat` | Build `q_align` from `axis_hat` (general case); do not assume `QUNIT` |

**Examples (special cases as code):**

```python
# Special: planar XY, hinge about world Z (local +Z already world +Z)
joint = chrono.ChLinkLockRevolute()
joint.Initialize(body_a, body_b, chrono.ChFramed(pivot_pos, chrono.QUNIT))
sys.AddLink(joint)
```

```python
# Special: XZ plane swing, gravity along -Z — hinge must be world ±Y, not +Z
q_hinge_y = chrono.QuatFromAngleAxis(chrono.CH_PI_2, chrono.VECT_X)
joint = chrono.ChLinkLockRevolute()
joint.Initialize(body_a, body_b, chrono.ChFramed(pivot_pos, q_hinge_y))
sys.AddLink(joint)
```

Do **not** default to `chrono.QUNIT` for an XZ pendulum with gravity along −Z: the torque from gravity in the swing plane will not match a hinge about +Z.

```python
# WRONG: copied from a prismatic guide example without checking hinge axis
joint = chrono.ChLinkLockRevolute()
joint.Initialize(body_a, body_b, chrono.ChFramed(pivot_pos, chrono.Q_ROTATE_Z_TO_X))
sys.AddLink(joint)
```

### Rule 9: Prismatic frame must map frame +Z onto the guide axis

`ChLinkLockPrismatic` uses the frame's local +Z as the sliding axis. Therefore:

- first identify the desired guide axis
- then rotate the frame so local +Z aligns with that guide axis

`chrono.Q_ROTATE_Z_TO_X` is only the common X-guide example, not a universal default.

```python
# Example: guide along global X
joint = chrono.ChLinkLockPrismatic()
joint.Initialize(slider, ground, chrono.ChFramed(pos, chrono.Q_ROTATE_Z_TO_X))
sys.AddLink(joint)
```

```python
# Example: guide along arbitrary axis_hat
joint = chrono.ChLinkLockPrismatic()
joint.Initialize(slider, ground, chrono.ChFramed(pos, rotation_z_to_axis_hat))
sys.AddLink(joint)
```

### Rule 10: Use body-local frames when the attachment point belongs to each body

For linkages such as crank-rod or rod-slider, the body-local `Initialize(body1, body2, True, frame1, frame2)` form is preferred because each marker is defined in the body it belongs to.

```python
joint = chrono.ChLinkLockRevolute()
joint.Initialize(body1, body2, True, frame1_local_to_body1, frame2_local_to_body2)
sys.AddLink(joint)
```

---

### Rule 11: Lock Joint (`ChLinkLockLock`) for Fixed Connections

Use `ChLinkLockLock` when two bodies must move together as a single unit — they share the same position and rotation. Common use: structural axle or connector that visually links two bodies but has no independent motion.

**3-arg form:**
```python
joint = chrono.ChLinkLockLock()
joint.Initialize(body1, body2, chrono.ChFramed(connection_point, q_alignment))
sys.AddLink(joint)
```
The `connection_point` is interpreted as a **body-local coordinate** on `body1`.

**Pattern — Axle connecting two bodies:**
```python
# body_a: one end of axle (e.g., at pivot)
# body_b: other end (e.g., disc at far end of axle)
# axle_pos: midpoint between body_a and body_b attachment points

axle.SetPos(axle_pos)

# Lock joint at body-local origin on axle, connecting to body_b
axle_to_bodyb = chrono.ChLinkLockLock()
axle_to_bodyb.Initialize(axle, body_b, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
sys.AddLink(axle_to_bodyb)
```

**Rule:** When using `ChLinkLockLock`, both bodies must be created and positioned first. The joint enforces that `body1.frame_origin = body2.frame_origin` in world coordinates. If the declared frame positions don't match the actual body positions, the constraint will be impossible and the simulation will diverge.

---

# Part 3: Springs and Supporting Rules

### ChLinkTSDA

```python
spring = chrono.ChLinkTSDA()
spring.Initialize(body1, body2, True,
                  chrono.ChVector3d(0, 0, 0),
                  chrono.ChVector3d(-1, 0, 0))
spring.SetRestLength(rest_length)
spring.SetSpringCoefficient(spring_k)
spring.SetDampingCoefficient(damping_c)
sys.AddLink(spring)
```

### Link / Spring / Rod Visualization

**Spring (ChLinkTSDA):**

```python
spring = chrono.ChLinkTSDA()
spring.Initialize(body1, body2, True,
                  chrono.ChVector3d(0, 0, 0),
                  chrono.ChVector3d(-1, 0, 0))
spring.SetRestLength(rest_length)
spring.SetSpringCoefficient(spring_k)
spring.SetDampingCoefficient(damping_c)
sys.AddLink(spring)

# MUST add visual shape to the LINK (not a body)
spring.AddVisualShape(chrono.ChVisualShapeSpring(coil_radius, resolution, turns))
```

**Rule:** Visual shapes on links (springs, dampers) must be added to the link object itself, not to any body.

**Rod / Axle between two bodies:**

```python
# Method A: ChBody with cylinder visual (if rod has mass/inertia)
rod = chrono.ChBody()
rod.SetMass(mass)
rod.SetInertiaXX(chrono.ChVector3d(1e-6, 1e-6, 1e-6))
rod.SetFixed(True)
sys.AddBody(rod)

cyl = chrono.ChVisualShapeCylinder(radius, length)
rod.AddVisualShape(cyl, chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleY(chrono.CH_PI_2)))

# Method B: ChVisualShapeLine for thin rod/cable (no mass)
rod_vis = chrono.ChVisualShapeLine()
rod_vis.SetLineGeometry(chrono.ChLineSegment(pos1, pos2))
rod_vis.SetColor(chrono.ChColor(0.3, 0.3, 0.3))
body1.AddVisualShape(rod_vis)  # attach to either body
```

**Visualization Checklist — All Structural Bodies:**

Before finishing body creation, confirm ALL of these are visualized:
- [ ] Ground (if visible)
- [ ] Fixed pivot / post / column
- [ ] Axle / shaft / connecting rod (even if fixed or visual-only)
- [ ] Main moving bodies (disc, arm, slider, etc.)
- [ ] Any spring / damper links

**Common mistake:** Creating bodies and joints but forgetting to add visual shapes for axle shafts, support posts, or other "minor" structural elements that are not the main moving bodies.

### Solver Configuration for Spring-Damper Systems

Spring-damper chains are stiff systems that require proper solver settings to remain stable:

```python
# Configure solver for spring stability
sys.SetSolverType(chrono.ChSolver.Type_PSOR)
solver = sys.GetSolver().AsIterative()
solver.SetMaxIterations(100)
solver.SetTolerance(1e-10)
solver.EnableWarmStart(True)  # Critical for spring convergence
```

**Why warm start matters:** Without it, the solver starts each timestep from scratch. With warm start (`EnableWarmStart(True)`), the solver uses the previous timestep's solution as an initial guess, which dramatically improves convergence for spring systems.

### Spherical Joint Initialization: 3-arg vs 5-arg

`ChLinkLockSpherical` has two main initialization forms:

**3-arg form (used in demos):**
```python
joint = chrono.ChLinkLockSpherical()
joint.Initialize(body1, body2, chrono.ChFramed(connection_point))
# connection_point is in body1's body-local coordinates
# The frame on body2 is automatically aligned to the same world point
```

**5-arg form (explicit relative frames):**
```python
joint.Initialize(body1, body2, True,  # rel_frames=True
                chrono.ChFramed(local_point_on_body1),   # point in body1's local frame
                chrono.ChFramed(local_point_on_body2))   # point in body2's local frame
# Explicitly specifies where on each body the joint connects
```

**Use 5-arg when:** The connection points matter for geometry (e.g., anchor points not at body center).

**Common mistake:** Using world-space coordinates instead of body-local coordinates in the frame, causing the joint to connect at wrong positions.

### Pure MBS: omit collision system

If the mechanism has no contact and is fully described by joints, springs, and motors, do not call `sys.SetCollisionSystemType()`.

### Constraint logging

If `sys.GetConstraintViolation()` is not directly usable in your logging path, log a derived scalar or `0.0`; do not block the whole simulation on debug logging.

---

# Pitfalls to Avoid

- Motor without revolute at a ground pivot
- Rod-slider prismatic in a fixed-guide linkage
- Missing slider-ground prismatic when a slider is supposed to move on a fixed guide
- Using `ChLinkRevolute` instead of the team's `ChLinkLockRevolute` convention
- Copying `Q_ROTATE_Z_TO_X` from a prismatic example into a revolute without checking the hinge axis
- Treating `Q_ROTATE_Z_TO_X` as the only valid prismatic rotation instead of an X-axis example
- Defaulting `chrono.QUNIT` for a revolute when the mechanism swings in the **XZ** plane with gravity along **−Z** (hinge should be **±Y**, not **+Z**; rotate the joint frame per Rule 8)
- Using `chrono.QUNIT` for a revolute **without** first defining the physical hinge `axis_hat` and aligning joint local +Z to it (Rule 8 general case—`QUNIT` is only correct when `axis_hat` is **+Z**)
- Running spring-damper systems without PSOR solver + warm start (causes numerical instability/explosion)
- Passing mass to `ChBodyEasySphere` instead of density (results in wrong mass and physics explosion)
- Using `(1, 1, 1)` for sphere inertia instead of correct formula `(2/5)*m*r²`
- **Geometry inconsistency**: Before creating a joint, verify that body positions and declared joint frame positions are consistent. If body A is at `P_A` and body B is at `P_B`, and a joint declares a frame at `F_A` in A's local coords and `F_B` in B's local coords, then `P_A + Rotate(F_A)` must equal `P_B + Rotate(F_B)` in world space. A mismatch causes the physics solver to impose impossible constraints, leading to explosion or divergence.
- **Body-local connection points must be within the body's geometry**: If a joint declares a connection point on a body, that point should be on or within the body's visual/collision geometry, not at an arbitrary distance that places the joint outside the physical structure.

## API Contract

allowed_classes:
- chrono.ChLinkLockRevolute
- chrono.ChLinkLockPrismatic
- chrono.ChLinkLockLock
- chrono.ChLinkLockSpherical
- chrono.ChLinkMotorRotationTorque
- chrono.ChLinkTSDA
- chrono.ChFramed
- chrono.ChVector3d
- chrono.ChBody
- chrono.ChVisualShapeSpring
- chrono.ChVisualShapeCylinder
- chrono.ChVisualShapeLine
- chrono.ChLineSegment
- chrono.ChColor
- chrono.ChFunctionSine

allowed_methods:
- joint.Initialize(body1, body2, chrono.ChFramed(pos, quat))
- joint.Initialize(body1, body2, True, frame1_local_to_body1, frame2_local_to_body2)
- joint.Initialize(body1, body2, chrono.ChFramed(connection_point))
- sys.AddLink(joint)
- motor.Initialize(body_slave, body_master, chrono.ChFramed(pos))
- motor.SetTorqueFunction(chrono.ChFunctionSine(amplitude, frequency))
- spring.Initialize(body1, body2, True, chrono.ChVector3d(x, y, z), chrono.ChVector3d(x, y, z))
- spring.SetRestLength(rest_length)
- spring.SetSpringCoefficient(spring_k)
- spring.SetDampingCoefficient(damping_c)
- spring.AddVisualShape(chrono.ChVisualShapeSpring(coil_radius, resolution, turns))
- frame.SetPos(pos)
- frame.SetRot(quat)
- rod.SetMass(mass)
- rod.SetInertiaXX(chrono.ChVector3d(x, y, z))
- rod.SetFixed(True)
- sys.AddBody(rod)
- rod.AddVisualShape(cyl, chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleY(chrono.CH_PI_2)))
- rod_vis.SetLineGeometry(chrono.ChLineSegment(pos1, pos2))
- rod_vis.SetColor(chrono.ChColor(r, g, b))
- body.AddVisualShape(rod_vis)
- axle.SetPos(axle_pos)
- sys.SetSolverType(chrono.ChSolver.Type_PSOR)
- sys.GetSolver().AsIterative()
- solver.SetMaxIterations(100)
- solver.SetTolerance(1e-10)
- solver.EnableWarmStart(True)
- chrono.QuatFromAngleAxis(angle, axis)
- chrono.QuatFromAngleY(angle)

allowed_constants:
- chrono.QUNIT
- chrono.Q_ROTATE_Z_TO_X
- chrono.VECT_X
- chrono.CH_PI_2
- chrono.CH_PI
- chrono.VNULL
- chrono.ChSolver.Type_PSOR

allowed_utils:
