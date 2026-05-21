---
name: quaternions
description: Quaternion creation, component access, and Euler angle conversion in PyChrono
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

# Skill: ChQuaterniond — Quaternion API in PyChrono

## Purpose

Access quaternion components, create quaternions from angles/axes, and convert to Euler angles in PyChrono.

## When to Use

When reading body orientation, converting to Euler angles, creating a rotation, or passing a quaternion to Initialize/SetRot.

## Component Attributes

PyChrono quaternions use **scalar-first** storage: `e0` is the scalar (w) part.

```python
q = body.GetRot()   # ChQuaterniond

# Correct attribute access:
w = q.e0   # scalar (w) component
i = q.e1   # x component
j = q.e2   # y component
k = q.e3   # z component
```

**Do NOT use** any of these — they do not exist in PyChrono:

```python
# WRONG — AttributeError:
q.w       # does not exist
q.x       # does not exist
q.y       # does not exist
q.z       # does not exist
q.GetW()  # does not exist
q.GetX()  # does not exist
q.GetY()  # does not exist
q.GetZ()  # does not exist
```

## Euler Angle Conversion

```python
rot = body.GetRot()               # ChQuaterniond
euler = rot.GetCardanAnglesXYZ()  # returns ChVector3d

angle_x = euler.x  # rotation about X-axis [rad]
angle_y = euler.y  # rotation about Y-axis [rad]
angle_z = euler.z  # rotation about Z-axis [rad]
```

## Creating Quaternions

```python
# From angle and axis (free function — returns a new quaternion):
q = chrono.QuatFromAngleAxis(angle_rad, chrono.VECT_Z)
q = chrono.QuatFromAngleAxis(angle_rad, chrono.ChVector3d(0, 0, 1))

# From angle and axis (instance method — mutates existing quaternion in-place):
q = chrono.ChQuaterniond()
q.SetFromAngleAxis(angle_rad, chrono.ChVector3d(0, 1, 0))

# Identity (no rotation):
q = chrono.QUNIT

# From Euler angles XYZ:
q = chrono.QuatFromAngleX(ax) * chrono.QuatFromAngleY(ay) * chrono.QuatFromAngleZ(az)
```

## Useful Constants

```python
chrono.QUNIT              # identity quaternion (no rotation)
chrono.Q_ROTATE_Y_TO_Z   # rotate so Y-axis maps to Z-axis
chrono.Q_ROTATE_Y_TO_X   # rotate so Y-axis maps to X-axis
chrono.Q_ROTATE_Z_TO_X   # rotate so Z-axis maps to X-axis
```

## Rotating Vectors (Coordinate Frame Transforms)

```python
rot = body.GetRot()   # ChQuaterniond

# Body-local → world frame (e.g., attachment point on body to world position)
world_vec = rot.RotateBack(local_vec)

# World → body-local frame
local_vec = rot.Rotate(world_vec)
```

Typical usage — compute world position of a body-local attachment point:

```python
rotor_pos = rotor.GetPos()
rotor_rot = rotor.GetRot()
attach_local = chrono.ChVector3d(0.25, 0, 0)  # point in body frame
attach_world = rotor_pos + rotor_rot.RotateBack(attach_local)
```

## Vector Operations

```python
v = chrono.ChVector3d(x, y, z)
v.Length()              # magnitude (scalar)
v.Cross(other_vec)      # cross product → ChVector3d
v.Dot(other_vec)        # dot product → scalar
# Vector arithmetic: +, -, * (scalar) work as expected
delta = pos2 - pos1     # difference vector
```

## Common Patterns

### Get rotation angle of a body about the Z-axis

```python
rot = body.GetRot()
euler = rot.GetCardanAnglesXYZ()  # ChVector3d
angle_about_z = euler.z           # [rad]
```

### Set body orientation

```python
q = chrono.QuatFromAngleAxis(chrono.CH_PI / 4, chrono.VECT_Z)
body.SetRot(q)
```

### Check if quaternion is normalized

```python
norm = (q.e0**2 + q.e1**2 + q.e2**2 + q.e3**2) ** 0.5
# Should be ~1.0 for a valid rotation quaternion
```

### Initialize joint with a rotated frame

```python
rot = chrono.QuatFromAngleAxis(-chrono.CH_PI / 2, chrono.ChVector3d(0, 1, 0))
frame = chrono.ChFramed(chrono.ChVector3d(x, y, z), rot)
joint.Initialize(body1, body2, frame)
```

## API Contract

allowed_classes:
- chrono.ChQuaterniond
- chrono.ChVector3d
- chrono.ChFramed

allowed_methods:
- body.GetRot()
- q.e0
- q.e1
- q.e2
- q.e3
- rot.GetCardanAnglesXYZ()
- chrono.QuatFromAngleAxis(angle_rad, axis)
- chrono.QuatFromAngleX(ax)
- chrono.QuatFromAngleY(ay)
- chrono.QuatFromAngleZ(az)
- q.SetFromAngleAxis(angle_rad, chrono.ChVector3d(x, y, z))
- rot.RotateBack(local_vec)
- rot.Rotate(world_vec)
- body.GetPos()
- body.SetRot(q)
- v.Length()
- v.Cross(other_vec)
- v.Dot(other_vec)
- joint.Initialize(body1, body2, frame)

allowed_constants:
- chrono.QUNIT
- chrono.Q_ROTATE_Y_TO_Z
- chrono.Q_ROTATE_Y_TO_X
- chrono.Q_ROTATE_Z_TO_X
- chrono.VECT_Z
- chrono.CH_PI
- chrono.CH_PI_2

allowed_utils:

