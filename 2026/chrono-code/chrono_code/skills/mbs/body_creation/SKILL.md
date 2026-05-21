---
name: body_creation
description: Create rigid bodies with mass, geometry, collision shapes, and visual assets.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

## API Contract

allowed_classes:
- chrono.ChBody
- chrono.ChBodyEasyBox
- chrono.ChBodyEasySphere
- chrono.ChBodyEasyCylinder
- chrono.ChVisualShapeBox
- chrono.ChVisualShapeSphere
- chrono.ChVisualShapeCylinder

allowed_methods:
- body.SetMass(mass)
- body.SetInertiaXX(vec)  # moment of inertia
- body.SetFixed(True)  # fix body in place (ground, wall, etc.)
- body.SetPos(vec)  # set position
- body.SetRot(quat)  # set rotation
- body.SetAngVelParent(vec)  # set angular velocity in parent/world frame

canonical_examples:
- Fixed ground body: body.SetFixed(True); sys.AddBody(body)
- Body with initial velocity: body.SetAngVelParent(chrono.ChVector3d(10, 0, 0))

# Skill: MBS Body Creation

## Purpose

Create rigid bodies — using easy factory functions or manually — with mass, geometry, collision shapes, and visual assets.

## When to Use

When populating a simulation with floors, objects, cranks, pistons, pendulum arms, or any physical part.

---

## PyChrono Defaults — Know Before Customizing

| Shape / Factory                          | Cylinder axis (body-local) | Visual auto-added?                          |
|------------------------------------------|----------------------------|---------------------------------------------|
| `ChVisualShapeCylinder` (manual)         | Z                          | No — must call `AddVisualShape` explicitly  |
| `ChBodyEasyCylinder(ChAxis_Y, ...)`      | Y                          | Yes — do NOT call `AddVisualShape` again    |
| `ChBodyEasyCylinder(ChAxis_X, ...)`      | X                          | Yes — do NOT call `AddVisualShape` again    |
| `ChBodyEasyCylinder(ChAxis_Z, ...)`      | Z                          | Yes — do NOT call `AddVisualShape` again    |

**Rule:** Easy factory bodies already have a visual shape built in. Never call `AddVisualShape`
on an easy factory body for the same shape — that creates a duplicate visual.

---

## Hard Rule: ChVisualShapeBox / ChCollisionShapeBox take FULL extents — never half

PyChrono's box constructors all expect the **full size** along each axis (not half-extents
as in Bullet's older C++ API). This applies to every overload — scalar and `ChVector3d`
forms — for both visual and collision boxes. Passing `W/2, D/2, H/2` makes the box come
out at half its intended dimensions on every axis, which silently looks correct in
isolation but breaks geometry alignment in any scene that depends on adjacent bodies
touching (platforms next to a tank, plate bridging two supports, wheels on a deck, etc.).

```python
# ✓ RIGHT — pass FULL dimensions (W, D, H), in metres
chrono.ChVisualShapeBox(chrono.ChVector3d(W, D, H))
chrono.ChVisualShapeBox(W, D, H)                            # scalar overload — also full
chrono.ChCollisionShapeBox(material, chrono.ChVector3d(W, D, H))
chrono.ChCollisionShapeBox(material, W, D, H)               # scalar overload — also full
chrono.ChBodyEasyBox(W, D, H, density, vis, collide, mat)   # easy factory — also full
```

```python
# ✗ WRONG — passes half-extents (Bullet's older convention).
# The visual / collision box is rendered at HALF its intended size on every axis.
chrono.ChVisualShapeBox(chrono.ChVector3d(W/2, D/2, H/2))
chrono.ChCollisionShapeBox(material, W/2, D/2, H/2)
```

**Common symptom**: bodies positioned with `SetPos` at the geometrically-correct center
appear to be floating in space far from the other bodies they should touch (the visible
gap looks like ½·W instead of the intended small clearance), and collision behaves as
if the body were half its expected size. The placement formula is right; the box is
just shrunk.

This rule is canonical for box primitives. Skill files that build container walls,
floating plates, platforms, ramps, or any other box geometry — `fsi/sph` Patterns C / D,
`scene/*` props, `core/*` examples — all defer to this rule. Cross-referencing skills
should NOT re-document the half-extent antipattern; just point here.

---

## Generated Boundary Bodies

Use `construction.primitive: generated_boundary` for open containers such as
water tanks, channels, bins, and other boundary-only domains. A generated
boundary is a special body: it represents walls/floor/boundary surfaces, not a
solid centered box.

Plan-level convention for a generated-boundary tank/channel:

```text
pose.position.x = XY center of the interior domain
pose.position.y = XY center of the interior domain
pose.position.z = floor / bottom z-coordinate, NOT geometric center
size.x          = full interior width along X
size.y          = full interior width along Y
size.z          = full interior container height from floor to rim
```

Derived geometry:

```text
BOTTOM_Z = pose.position.z
RIM_Z    = pose.position.z + size.z
LEFT_X   = pose.position.x - size.x / 2
RIGHT_X  = pose.position.x + size.x / 2
MIN_Y    = pose.position.y - size.y / 2
MAX_Y    = pose.position.y + size.y / 2
```

Do not apply the ordinary centered-box z formulas to generated boundaries:

| Ordinary centered box formula | Generated-boundary replacement |
|---|---|
| `top_z = z + size.z / 2` | `rim_z = z + size.z` |
| `bottom_z = z - size.z / 2` | `bottom_z = z` |
| `x ± size.x / 2` | unchanged |
| `y ± size.y / 2` | unchanged |

Implementation rule: do not create one monolithic collision box for an open
container. Build it as floor and wall slabs, or use the domain-specific helper
that constructs those slabs. For SPH/FSI tanks, the helper and BCE rules live
in `fsi/sph` Pattern C.

---

## Easy Factory vs Manual — Choose Before Writing Code

**Use `ChBodyEasyBox` / `ChBodyEasySphere` freely** for any body shape.

**Use `ChBodyEasyCylinder` only when:**
- The body is a fixed or structural element (post, column, ground pin) whose axis-aligned
  orientation (ChAxis_X/Y/Z) directly matches the world orientation without further `SetRot`
- OR the simulation doesn't require re-orienting the body during the run

**Use `ChBody` + `ChVisualShapeCylinder` for:**
- Any link body whose orientation is non-trivial or changes during simulation
  (crank, connecting rod, pendulum arm, any rotating or translating link)
- Reason: manual setup gives full control; the two-step visual formula below applies only
  to manual `ChBody` + `ChVisualShapeCylinder` — not to easy factory bodies

---

## Easy Factory Bodies (simplest approach)

These create the ChBody, add a visual shape, optionally add a collision shape, and set the mass from density — all in one call.

PyChrono easy-body constructors are positional-only here; do not use keyword args such as `material=`, `collision=`, or `visualization=`.

**Hard rule: easy-body boolean arguments are NOT fixed flags.** In
`ChBodyEasyBox(..., density, visualize, collide, material)`, the two booleans
mean `visualize` and `collide`. A body created this way is still dynamic and
will fall under gravity unless you explicitly call `body.SetFixed(True)`.
Never write comments such as `True  # fixed` on these constructor arguments.

```python
# Box: x, y, z extents (full sizes), density, visualize, collide, material
box_sx = float       # box width [m]
box_sy = float       # box height [m]
box_sz = float       # box depth [m]
box_density = float  # material density [kg/m³]
body = chrono.ChBodyEasyBox(box_sx, box_sy, box_sz, box_density, True, True, mat)  # visualize, collide

# Cylinder: axis enum, radius, height, density, visualize, collide, material
# Axis enum sets which body-local axis the cylinder points along.
# Visual shape is created automatically — do NOT call AddVisualShape after.
cyl_radius = float   # cylinder radius [m]
cyl_height = float   # cylinder height [m]
cyl_density = float  # material density [kg/m³]
body = chrono.ChBodyEasyCylinder(chrono.ChAxis_Y, cyl_radius, cyl_height, cyl_density, True, True, mat)  # visualize, collide

# Sphere: radius, DENSITY (not mass!), visualize, collide, material
sph_radius = float   # sphere radius [m]
sph_density = float  # material density [kg/m³] — THIS IS DENSITY, NOT MASS
body = chrono.ChBodyEasySphere(sph_radius, sph_density, True, True, mat)  # visualize, collide
```

**⚠️ Common Mistake:** The second parameter to `ChBodyEasySphere` is **density** [kg/m³], not mass.
With `sph_radius=0.1` and `sph_density=1000` (water), the actual mass is only ~4.2e-3 kg.
If you need a specific mass, use manual `ChBody()` setup (see below).

After factory creation:
```python
body.SetPos(chrono.ChVector3d(x, y, z))
body.SetFixed(True)          # immovable (floor/wall)
sys.Add(body)
# Do NOT call AddVisualShape — the factory already added one.
```

---

## Manual Body Setup (full control)

Use this path for any link body (crank, rod, pendulum arm, etc.):

```python
mass = float         # total body mass [kg]
inertia_xx = float   # moment of inertia about X [kg·m²]
pos_x = float        # x position [m]
pos_y = float        # y position [m]
pos_z = float        # z position [m]
body = chrono.ChBody()
body.SetMass(mass)
body.SetInertiaXX(chrono.ChVector3d(inertia_xx, inertia_xx, inertia_xx))
body.SetPos(chrono.ChVector3d(pos_x, pos_y, pos_z))
body.EnableCollision(False)  # or True
sys.AddBody(body)
```

**Inertia formulas for common shapes:**
- Solid sphere: `I = (2/5) * m * r²` (same about all axes)
- Solid cylinder (about axis): `I = (1/2) * m * r²`
- Solid box: `I_x = (1/12) * m * (y² + z²)` etc.

---

## Cylinder Orientation — General Rule (manual ChBody only)

**Goal:** make the cylinder visual point along the link's physical axis in the world.

**Convention:** define body-local X as the link direction. Then apply two steps:

```
Step 1 — Body rotation:
    body.SetRot(rotation_that_points_body_local_X_along_link_direction)
    → encodes the link direction in world space

Step 2 — Visual offset (always the same formula):
    cyl = chrono.ChVisualShapeCylinder(radius, height)
    body.AddVisualShape(cyl, chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleY(chrono.CH_PI_2)))
    → rotates cylinder axis from body-local Z to body-local X
```

**Why it works:**
- `ChVisualShapeCylinder` default axis is body-local **Z**
- `QuatFromAngleY(CH_PI_2)` rotates Z → X
- So the cylinder ends up pointing along body-local X, which Step 1 aligned with the link
- `ChFramed` position = `VNULL` when `body.SetPos()` is at the link's midpoint (body COM)

**Never invent per-body magic ChFramed rotations.** Use `SetRot` to encode orientation,
and always use the same `ChFramed(VNULL, QuatFromAngleY(CH_PI_2))` offset for the visual.

---

## Step 1 Rotation by Case

```python
# Planar mechanism — link at angle θ from world X, rotating in XY plane:
body.SetRot(chrono.QuatFromAngleZ(theta))
# → body-local X points at angle θ in the XY plane
# Step 2 formula unchanged

# Link initially along world X (angle 0):
body.SetRot(chrono.QUNIT)   # body-local X = world X already
# Step 2 formula unchanged

# Vertical link along world Z:
# Option A — use ChBodyEasyCylinder(ChAxis_Z, ...) for a fixed vertical rod (no manual visual needed)
# Option B — manual body: skip the QuatFromAngleY rotation and attach with QUNIT
#   body.AddVisualShape(cyl, chrono.ChFramed(chrono.VNULL, chrono.QUNIT))
#   (cylinder default Z is already vertical)

# 3D link along arbitrary unit vector — align body-local X with that vector:
body.SetRot(chrono.QuatFromAngleAxis(angle, axis_vector))
# Step 2 formula unchanged
```

### Rotation helpers (quaternion constants and functions)

```python
chrono.QUNIT                            # identity rotation
chrono.QuatFromAngleZ(angle_rad)        # rotate about world Z by angle (use for planar links)
chrono.QuatFromAngleY(angle_rad)        # rotate about Y (used in Step 2 visual offset)
chrono.QuatFromAngleX(angle_rad)        # rotate about X
chrono.QuatFromAngleAxis(angle_rad, chrono.VECT_X)  # arbitrary rotation about X
chrono.Q_ROTATE_Y_TO_Z                 # cylinder axis Y → Z direction (easy factory only)
chrono.Q_ROTATE_Y_TO_X                 # cylinder axis Y → X direction (easy factory only)
chrono.Q_ROTATE_Z_TO_X                 # frame Z → X (used for prismatic joints)
```

---

## Visual Shapes (attach to manual ChBody)

```python
vs_sx = float  # visual box width [m]
vs_sy = float  # visual box height [m]
vs_sz = float  # visual box depth [m]
box_shape = chrono.ChVisualShapeBox(vs_sx, vs_sy, vs_sz)
box_shape.SetTexture(chrono.GetChronoDataFile('textures/concrete.jpg'))
body.AddVisualShape(box_shape)

vs_radius = float  # visual sphere radius [m]
sph_shape = chrono.ChVisualShapeSphere(vs_radius)
body.AddVisualShape(sph_shape, chrono.ChFramed(chrono.ChVector3d(offset_x, 0, 0)))

# Cylinder — default axis is body-local Z.
# For any link body, use the two-step formula above (Step 1 SetRot + Step 2 AddVisualShape):
cyl_shape = chrono.ChVisualShapeCylinder(radius, height)
body.AddVisualShape(cyl_shape, chrono.ChFramed(chrono.VNULL, chrono.QuatFromAngleY(chrono.CH_PI_2)))
```

---

## Support Structures (Axles, Pins, Posts)

Every mechanism has structural elements that are not main moving bodies but are still essential to visualize:

- **Axle / Shaft**: connects rotating parts to fixed pivots
- **Pin / Pivot**: small cylindrical joint at the attachment point
- **Post / Column**: vertical or angled structural support

**Rule:** If the mechanism has an axle, post, or pin visible in a diagram or description, it must be created as a body with a visual shape — even if it is fixed and has no physics function beyond structural connection.

**Example — Axle from disc to pivot:**
```python
axle = chrono.ChBody()
axle.SetMass(1.0)  # arbitrary small mass
axle.SetInertiaXX(chrono.ChVector3d(1e-6, 1e-6, 1e-6))
axle.SetFixed(True)
sys.AddBody(axle)

axle_visual = chrono.ChVisualShapeCylinder(axle_radius, axle_length)
axle_visual.SetColor(chrono.ChColor(0.3, 0.3, 0.3))
axle.AddVisualShape(axle_visual, chrono.ChFramed(
    chrono.ChVector3d(offset_x, 0.0, 0.0),
    chrono.QuatFromAngleY(chrono.CH_PI_2)
))
```

**Do NOT skip the axle** just because it is "only visual" — it is a structural body that the code generator must not omit.

---

## Assembly Positioning (Connected Bodies)

When creating an assembly of multiple bodies connected by joints, body positions must be set such that joint connection points are consistent with body positions in world space.

**Rule:** Before setting body positions, determine the joint connection geometry first:

1. Identify where each joint connects (which body, which point on that body)
2. Compute world position of each connection point
3. Position each body so that its declared joint frame aligns with that world position

**General pattern — Two bodies connected by an intermediate structural element:**
```
World anchor point: (Ax, Ay, Az)
Body B center: (Bx, By, Bz)
Intermediate body (e.g., axle): centered at midpoint between anchor and B

When joint declares frame at (Fx, Fy, Fz) on body A's local coords,
that point in world must equal: A.GetPos() + Rotate(F, A.GetRot())
```

**Common mistake:** Setting body positions without regard to declared joint frames, resulting in impossible constraint distances that cause the physics solver to diverge.

---

## Texture / Visual Material

```python
# Simple texture on easy body:
body.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))

# Full visual material:
vis_mat = chrono.ChVisualMaterial()
vis_mat.SetKdTexture(chrono.GetChronoDataFile("textures/concrete.jpg"))
body.GetVisualShape(0).SetMaterial(0, vis_mat)
```
