---
name: system_create
description: Create and configure a PyChrono ChSystem, gravity, contact method, and solver.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

## API Contract

allowed_classes:
- chrono.ChSystemNSC
- chrono.ChSystemSMC
- chrono.ChSolver.Type_PSOR
- chrono.ChCollisionSystem.Type_BULLET

allowed_methods:
- sys.SetGravityY()  # shorthand: gravity = (0, -9.81, 0)
- sys.SetGravitationalAcceleration(vec)  # explicit vector form
- sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
- sys.SetSolverType(chrono.ChSolver.Type_PSOR)
- sys.GetSolver().AsIterative().SetMaxIterations(n)

allowed_utils:
- from chrono_code.utils import setup_preview_camera

canonical_examples:
- ChSystemNSC with gravity: sys = chrono.ChSystemNSC(); sys.SetGravityY()
- ChSystemNSC with explicit gravity: sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))
- ChSystemSMC (no gravity): sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))

# Skill: MBS System Creation

## Purpose

Create and configure a PyChrono multi-body simulation system (ChSystem), set gravity, choose contact method, and configure the solver.

## When to Use

Use at the top of every MBS simulation script, before creating any bodies or links.

## When NOT to Use

**DO NOT use this skill for wrapper-managed vehicle scenes.** The wrapper
vehicles `veh.HMMWV_Full`, `veh.CityBus`, and `veh.FEDA` create their
own `ChSystemSMC` internally — calling `chrono.ChSystemSMC()` (or
`ChSystemNSC()`) before the vehicle produces an orphan system that the
vehicle's VSG visualizer will never render, and passing that pre-built
system to the wrapper (`HMMWV_Full(sys)`) segfaults during
`Initialize()` when the collision system is rebuilt. For those scenes:

1. Construct the wrapper first: `hmmwv = veh.HMMWV_Full()` (no-arg).
2. `hmmwv.Initialize()`.
3. Then take `system = hmmwv.GetSystem()` and use that for everything
   else (terrain, scene bodies, sensor manager).

See the `veh/wheeled_vehicle` skill — specifically its "Hard rules:
system ownership and construction order" section — for the complete
required ordering. `HMMWV_Reduced` is the only wheeled-vehicle wrapper
that takes `sys` as a constructor argument; do not generalize that
signature.

## `chrono.SetChronoDataPath(...)` — DO NOT CALL BY DEFAULT

**Default behavior**: `import pychrono.core as chrono` already sets
`GetChronoDataPath()` to the installed data root (e.g.
`$CONDA_PREFIX/share/chrono/data/`). **In 99% of generated simulations
you should NOT call `SetChronoDataPath` at all.** Just import and use
the default.

### Hard rules

1. **Do not call `chrono.SetChronoDataPath()`.** The default path set by
   `import pychrono.core` is correct. Hand-computed paths from `__file__`
   produce wrong locations (the generated script lives under
   `history/iteration_NNN/`, not the chrono data root) and cause VSG
   segfaults from missing shaders.

2. **Do not hard-code absolute paths** like `/home/...`, `/opt/...`,
   `/usr/local/...`. Scripts must be environment-independent.

3. **NEVER strip the trailing `/data/`**. PyChrono resolves
   sub-resources (`vsg/fonts/...`, `shaders/...`) relative to this
   root; omitting the final segment also produces the font/shader
   segfault.

### When explicit override IS actually needed

Only override when the default cannot be trusted (e.g. tests, CI with
a non-standard install). Use the active interpreter's prefix:

```python
import sys, os
import pychrono.core as chrono
from pathlib import Path

_default = chrono.GetChronoDataPath()  # whatever pychrono picked up at import
_derived = str(Path(sys.prefix) / "share" / "chrono" / "data") + "/"

# Only override if pychrono's default is empty or doesn't actually exist.
if not _default or not os.path.isdir(_default):
    if os.path.isdir(_derived):
        chrono.SetChronoDataPath(_derived)
```

The trailing `/` after `data` is **required**.

See the [scene/asset](../../scene/asset/SKILL.md) skill for the full
asset-loading conventions.

## Key Concepts

### ChSystemNSC vs ChSystemSMC

- **ChSystemNSC** (Non-Smooth Contact): rigid/impulsive impacts, fast, best for rigid-body scenes with discrete collisions
- **ChSystemSMC** (Smooth Contact Method): penalty-based, soft contacts, best for granular or compliant-contact scenarios

### Gravity

```python
sys.SetGravityY()                                          # shorthand: (0, -9.81, 0)
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))  # explicit form
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))       # disable gravity
```

**Gravity shorthand helpers (all take ZERO arguments):**
- `SetGravityY()` — sets gravity to (0, -9.81, 0) in one call
- `SetGravityX()` — sets gravity to (-9.81, 0, 0)
- `SetGravityZ()` — sets gravity to (0, 0, -9.81)
- `SetGravitationalAcceleration(vec)` — arbitrary vector form

**Common Mistakes:**

| Wrong | Correct | Why |
|-------|---------|-----|
| `sys.SetGravityX(0)` | `sys.SetGravityX()` | Shorthand takes NO arguments |
| `sys.SetGravityZ(-9.81)` | `sys.SetGravityZ()` | Shorthand takes NO arguments |
| `sys.Set_G_acc(vec)` | `sys.SetGravitationalAcceleration(vec)` | `Set_G_acc` does not exist |
| `sys.SetMaxIterations(n)` | `sys.GetSolver().AsIterative().SetMaxIterations(n)` | No direct `SetMaxIterations` on system |

### Collision System

Call only when bodies have collision shapes and contact is needed. For pure MBS (joints and motors only, no contact), omit — it adds overhead and can cause instability.

```python
# When contact/collision is needed:
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

# Pure MBS (no contact): do NOT call SetCollisionSystemType
```

### NSC Solver (tune for collision-heavy NSC scenes)

```python
solver_max_iter = int  # max solver iterations
sys.SetSolverType(chrono.ChSolver.Type_PSOR)
sys.GetSolver().AsIterative().SetMaxIterations(solver_max_iter)
# Alternative:
solver = chrono.ChSolverPSOR()
solver.SetMaxIterations(solver_max_iter)
sys.SetSolver(solver)
```

## Minimal Example

```python
import pychrono.core as chrono

solver_max_iter = int  # max solver iterations

# NSC system with gravity and collision
sys = chrono.ChSystemNSC()
sys.SetGravityY()
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
sys.SetSolverType(chrono.ChSolver.Type_PSOR)
sys.GetSolver().AsIterative().SetMaxIterations(solver_max_iter)
```

```python
# SMC system (no gravity, for spring demo)
sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))
```

## Required API Patterns

Use these exact patterns. Do not deviate.

```python
# Solver iteration limit — requires .AsIterative()
sys.GetSolver().AsIterative().SetMaxIterations(100)
```

```python
# Bounding box access — .min / .max attributes, not methods
bb = mesh.GetBoundingBox()
low = bb.min
high = bb.max
```

```python
# Sensor background mode — module-level constant
bg = sens.Background()
bg.mode = sens.BackgroundMode_SOLID_COLOR
```

```python
# Sensor ambient light — ChVector3f, NOT ChColor
manager.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))
```

VSG visualization setup is in the core skill for your plan type (`core/mbs`, `core/scene`, or `core/mbs_in_scene`).
