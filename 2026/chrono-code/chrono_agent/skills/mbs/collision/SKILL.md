---
name: collision
description: Enable contact detection using the correct contact material for NSC or SMC systems.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

# Skill: MBS Collision

## Purpose

Enable contact detection between bodies using the correct contact material for the system type (NSC or SMC), and attach collision shapes to bodies.

## When to Use

When bodies should interact physically through contacts (floors, walls, falling objects, mixers, etc.).

## Key Concepts

### API Contract

allowed_classes:
- chrono.ChContactMaterialNSC
- chrono.ChContactMaterialSMC
- chrono.ChCollisionShapeSphere
- chrono.ChCollisionShapeBox
- chrono.ChBodyEasySphere
- chrono.ChBodyEasyBox
- chrono.ChBodyEasyCylinder

allowed_methods:
- sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
- body.EnableCollision(True)
- body.AddCollisionShape(shape)
- mat.SetFriction(...)
- mat.SetRestitution(...)

canonical_examples:
- ChSystemNSC -> chrono.ChContactMaterialNSC()
- ChSystemSMC -> chrono.ChContactMaterialSMC()
- Call sys.SetCollisionSystemType(...) before stepping collision bodies

### Rule: Material must match system type

- `ChSystemNSC` → use `ChContactMaterialNSC`
- `ChSystemSMC` → use `ChContactMaterialSMC`
Mixing them causes silent errors or crashes.

### Required: enable the collision system

```python
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
```

This must be called once on the system before any bodies with collision are stepped.

## NSC (Non-Smooth Contact)

### Material

```python
friction = float     # friction coefficient
restitution = float  # restitution coefficient
mat = chrono.ChContactMaterialNSC()
mat.SetFriction(friction)
mat.SetRestitution(restitution)
```

### Easy bodies (material passed to factory)

```python
# Sphere
sph_radius = float   # sphere radius [m]
sph_density = float  # sphere density [kg/m³]
sphere = chrono.ChBodyEasySphere(sph_radius, sph_density, True, True, mat)
sphere.SetPos(chrono.ChVector3d(x, y, z))
sys.Add(sphere)

# Box
box_sx = float      # box width [m]
box_sy = float      # box height [m]
box_sz = float      # box depth [m]
box_density = float # box density [kg/m³]
box = chrono.ChBodyEasyBox(box_sx, box_sy, box_sz, box_density, True, True, mat)  # visualize, collide
sys.Add(box)

# Cylinder
cyl_radius = float   # cylinder radius [m]
cyl_height = float   # cylinder height [m]
cyl_density = float  # cylinder density [kg/m³]
cyl = chrono.ChBodyEasyCylinder(chrono.ChAxis_Y, cyl_radius, cyl_height, cyl_density, True, True, mat)  # visualize, collide
sys.Add(cyl)
```

The two booleans in `ChBodyEasy*` constructors are `visualize` and `collide`,
not `fixed`. For immovable collision bodies such as ground, walls, platforms,
or supports, call `body.SetFixed(True)` explicitly before adding the body to the
system.

### Visual material (texture on contact body)

```python
vis_mat = chrono.ChVisualMaterial()
vis_mat.SetKdTexture(chrono.GetChronoDataFile("textures/concrete.jpg"))
body.GetVisualShape(0).SetMaterial(0, vis_mat)
```

## SMC (Smooth/Penalty Contact)

### Material

```python
friction = float  # friction coefficient
mat = chrono.ChContactMaterialSMC()
mat.SetFriction(friction)
# mat.SetYoungModulus(2e7)   # optional
# mat.SetPoissonRatio(0.3)   # optional
```

### Manual body with explicit collision shapes

```python
mass = float    # body mass [kg]
radius = float  # collision sphere radius [m]
body = chrono.ChBody()
body.SetMass(mass)
body.SetPos(chrono.ChVector3d(x, y, z))

# Add collision shape (SMC material embedded in shape)
shape = chrono.ChCollisionShapeSphere(mat, radius)
body.AddCollisionShape(shape)
body.EnableCollision(True)

# Add visual shape separately
sphere_vs = chrono.ChVisualShapeSphere(radius)
sphere_vs.SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
body.AddVisualShape(sphere_vs)

sys.AddBody(body)
```

### SMC collision shapes

```python
chrono.ChCollisionShapeSphere(mat, radius)
chrono.ChCollisionShapeBox(mat, sx, sy, sz)
# Attach at offset: body.AddCollisionShape(shape, chrono.ChFramed(pos, rot))
```

## When to Use NSC vs SMC


| Scenario                              | Recommended |
| ------------------------------------- | ----------- |
| Rigid/hard impacts (balls, blocks)    | NSC         |
| Many simultaneous contacts (granular) | NSC         |
| Soft/compliant contacts               | SMC         |
| Need continuous force (no impulsive)  | SMC         |
