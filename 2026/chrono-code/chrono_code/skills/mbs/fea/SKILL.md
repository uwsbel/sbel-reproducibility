---
name: fea
description: Finite Element Analysis — beam elements, tetrahedral solid elements, MKL solver, HHT timestepper, breakable constraints, FEA contact surfaces.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

# Skill: Finite Element Analysis (FEA)

## Purpose

Model deformable structures using Chrono's FEA module: Euler-Bernoulli beam elements (e.g., breakable trees), tetrahedral solid elements (e.g., soft rubber obstacles), and breakable constraints that snap under load.

## When to Use

Use when the plan involves deformable bodies, flexible structures, breakable joints, or soft-body physics. FEA elements interact with rigid bodies through contact surfaces.

## System Setup for FEA

FEA requires `ChSystemSMC` and a direct solver (MKL). Iterative solvers diverge on FEA stiffness matrices.

```python
import pychrono as chrono
import pychrono.fea as fea
import pychrono.pardisomkl as mkl

sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.01)

# MKL direct solver (required for FEA)
sys.SetSolver(mkl.ChSolverPardisoMKL())
```

### Timestepper Selection

Choose based on element type:

**Beam elements** — HHT (Hilber-Hughes-Taylor) timestepper:
```python
hht = chrono.ChTimestepperHHT(sys)
hht.SetAlpha(-0.3)
hht.SetMaxIters(200)
hht.SetAbsTolerances(1e-4)
hht.SetStepControl(False)
sys.SetTimestepper(hht)
```

**Tetrahedral solid elements** — Euler implicit:
```python
sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
```

Typical timestep for FEA stability: `step_size = 5e-4`.

## Beam Elements (Euler-Bernoulli)

Create flexible beams using `ChBeamSectionEulerAdvanced` and `ChBuilderBeamEuler`.

### Section Properties

```python
sec = fea.ChBeamSectionEulerAdvanced()
sec.SetAsCircularSection(diameter)   # e.g., 0.10 for thin trunk, 0.25 for thick
sec.SetDensity(density)              # e.g., 500-700 kg/m³ for wood
sec.SetYoungModulus(E)               # e.g., 8e9-10e9 Pa for wood
sec.SetShearModulus(E * 0.35)        # typically ~35% of Young's modulus
sec.SetRayleighDamping(0.05)
```

### Building Beams

```python
mesh = fea.ChMesh()
mesh.SetAutomaticGravity(True)

up = chrono.ChVector3d(1, 0, 0)  # lateral reference direction

builder = fea.ChBuilderBeamEuler()
builder.BuildBeam(
    mesh, sec, n_elements,
    chrono.ChVector3d(x, y, z_start),   # start point
    chrono.ChVector3d(x, y, z_end),     # end point
    up,
)

# Fix the root node
builder.GetLastBeamNodes().front().SetFixed(True)

# Access tip node
tip = builder.GetLastBeamNodes().back()

sys.Add(mesh)
```

### SWIG GC Pitfall (CRITICAL)

**Must keep a reference to `GetLastBeamNodes()` before indexing.** Without this, the SWIG temporary container is garbage-collected and node `shared_ptr`s become dangling — causing a segfault when `GetPos()` is called later.

```python
# CORRECT: store the container first
beam_nodes = builder.GetLastBeamNodes()
nodes = [beam_nodes[i] for i in range(beam_nodes.size())]

# WRONG: indexing into a temporary
node = builder.GetLastBeamNodes()[0]  # may segfault later
```

Also keep strong references to meshes, builders, sections, and contact surfaces in a dict or list to prevent premature GC.

## Tetrahedral Solid Elements

Create soft deformable bodies from TetGen mesh files.

### Material

```python
material = fea.ChContinuumElastic()
material.SetYoungModulus(0.5e6)        # soft rubber
material.SetPoissonRatio(0.4)
material.SetRayleighDampingBeta(0.01)
material.SetDensity(1200)             # kg/m³
```

### Loading from TetGen Files

```python
mesh = fea.ChMesh()
mesh.SetAutomaticGravity(True)

# Non-uniform scale via ChMatrix33d diagonal
scale = chrono.ChMatrix33d(1)
scale.setitem(0, 0, 5.0)    # X scale
scale.setitem(1, 1, 0.6)    # Y scale
scale.setitem(2, 2, 15.0)   # Z scale

fea.ChMeshFileLoader.FromTetGenFile(
    mesh,
    chrono.GetChronoDataFile("fea/beam.node"),
    chrono.GetChronoDataFile("fea/beam.ele"),
    material,
    chrono.ChVector3d(0, 0, 0.3),   # center position
    scale,
)

sys.Add(mesh)
```

## FEA Contact Surfaces

FEA meshes need explicit contact surfaces to interact with rigid bodies.

### For Beam Elements — Node Cloud

```python
contact_mat = chrono.ChContactMaterialSMC()
contact_mat.SetFriction(0.5)
contact_mat.SetYoungModulus(1e7)

ct = fea.ChContactSurfaceNodeCloud(contact_mat)
ct.AddAllNodes(mesh, contact_radius)   # radius around each node
mesh.AddContactSurface(ct)
```

### For Tetrahedral Elements — Surface Mesh

```python
contact_surf = fea.ChContactSurfaceMesh(contact_mat)
contact_surf.AddFacesFromBoundary(mesh, 0.002)  # sphere-swept thickness
mesh.AddContactSurface(contact_surf)
```

## Breakable Constraints

Use `ChLinkMateGeneric` to join two FEA meshes (or an FEA mesh and a rigid body) with a constraint that can be broken when reaction forces exceed a threshold.

```python
# Join top of lower trunk to bottom of upper trunk
lnk = chrono.ChLinkMateGeneric()
lnk.Initialize(node_top, node_bot, False,
               node_top.Frame(), node_bot.Frame())
lnk.SetConstrainedCoords(True, True, True, True, True, True)  # all 6 DOF
lnk.SetName("breakable_joint")
sys.Add(lnk)
```

### Break Check in Simulation Loop

```python
if not lnk.IsBroken():
    M = lnk.GetReaction2().torque.Length()
    if M > break_threshold:
        lnk.SetBroken(True)
        print(f"Joint snapped! M={M:.0f} > {break_threshold}")
```

## FEA Visualization

### VSG Limitation

`ChVisualShapeFEA` causes a segfault with VSG on frame update. For beam elements with VSG, use **rigid-body cylinder trackers** that follow FEA node positions each frame:

```python
# Create one cylinder body per beam segment
for i in range(len(nodes) - 1):
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    cyl = chrono.ChVisualShapeCylinder(radius, segment_length)
    cyl.SetColor(color)
    body.AddVisualShape(cyl, chrono.ChFramed())
    sys.Add(body)

# Each step: update cylinder pos/rot to match FEA node positions
```

### Irrlicht (Tetrahedral Visualization)

`ChVisualShapeFEA` works with Irrlicht for tetrahedral meshes:

```python
vis_surface = chrono.ChVisualShapeFEA()
vis_surface.SetFEMdataType(chrono.ChVisualShapeFEA.DataType_NODE_SPEED_NORM)
vis_surface.SetColormapRange(chrono.ChVector2d(0.0, 2.0))
vis_surface.SetSmoothFaces(True)
mesh.AddVisualShapeFEA(vis_surface)

# Wireframe overlay (undeformed reference)
vis_wire = chrono.ChVisualShapeFEA()
vis_wire.SetFEMdataType(chrono.ChVisualShapeFEA.DataType_SURFACE)
vis_wire.SetWireframe(True)
vis_wire.SetDrawInUndeformedReference(True)
mesh.AddVisualShapeFEA(vis_wire)
```

## API Contract

allowed_classes:
- fea.ChMesh
- fea.ChBeamSectionEulerAdvanced
- fea.ChBuilderBeamEuler
- fea.ChContinuumElastic
- fea.ChMeshFileLoader
- fea.ChContactSurfaceNodeCloud
- fea.ChContactSurfaceMesh
- chrono.ChLinkMateGeneric
- chrono.ChTimestepperHHT
- chrono.ChVisualShapeFEA
- chrono.ChVisualShapeCylinder
- mkl.ChSolverPardisoMKL

allowed_methods:
- sec.SetAsCircularSection(diameter)
- sec.SetDensity(density)
- sec.SetYoungModulus(E)
- sec.SetShearModulus(G)
- sec.SetRayleighDamping(beta)
- mesh.SetAutomaticGravity(True)
- mesh.AddContactSurface(surface)
- mesh.AddVisualShapeFEA(vis)
- mesh.GetNumNodes()
- mesh.GetNumElements()
- builder.BuildBeam(mesh, sec, n, start, end, up)
- builder.GetLastBeamNodes()
- fea.ChMeshFileLoader.FromTetGenFile(mesh, node_file, ele_file, material, center, scale)
- material.SetYoungModulus(E)
- material.SetPoissonRatio(nu)
- material.SetRayleighDampingBeta(beta)
- material.SetDensity(rho)
- ct.AddAllNodes(mesh, radius)
- contact_surf.AddFacesFromBoundary(mesh, thickness)
- lnk.Initialize(body1, body2, False, frame1, frame2)
- lnk.SetConstrainedCoords(tx, ty, tz, rx, ry, rz)
- lnk.SetBroken(True)
- lnk.IsBroken()
- lnk.GetReaction2().torque.Length()
- hht.SetAlpha(alpha)
- hht.SetMaxIters(n)
- hht.SetAbsTolerances(tol)
- hht.SetStepControl(bool)
- sys.SetSolver(mkl.ChSolverPardisoMKL())
- sys.SetTimestepper(hht)
- sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
- scale.setitem(row, col, value)

allowed_constants:
- chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED

## Pitfalls

- **SWIG GC**: Always keep strong references to `GetLastBeamNodes()` containers, meshes, builders, sections, and contact surfaces. Failing to do so causes dangling pointers and segfaults.
- **Solver**: FEA requires `ChSystemSMC` + `ChSolverPardisoMKL`. Iterative solvers (PSOR, etc.) will diverge.
- **VSG + FEA**: `ChVisualShapeFEA` segfaults with VSG. Use rigid-body cylinder trackers for beam visualization, or Irrlicht for tetrahedral visualization.
- **Timestep**: Use `5e-4` or smaller for FEA stability. Larger steps cause divergence.
- **Contact**: FEA meshes need explicit contact surfaces (`ChContactSurfaceNodeCloud` or `ChContactSurfaceMesh`) — they do not participate in collision by default.
