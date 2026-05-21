---
name: vsg
description: Set up and run ChVisualSystemVSG for 3D rendering — window, camera, grid, sky, lights, and render loop.
compatibility: pychrono >= 8.0
metadata:
  domain: vis
---

## API Contract

allowed_classes:
- chronovsg.ChVisualSystemVSG

allowed_methods:
- vis.AttachSystem(sys)
- vis.SetWindowSize(width, height)
- vis.SetWindowTitle(title)
- vis.SetCameraAngleDeg(angleDeg)
- vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
- vis.SetLightIntensity(intensity)
- vis.SetLightDirection(azimuth, elevation)
- vis.EnableSkyTexture()
- vis.SetSkyDomeTexture(filename, sun_azimuth)
- vis.SetSkyBoxTexture(filename, sun_azimuth)
- vis.SetBackgroundColor(chrono.ChColor(r, g, b))
- vis.EnableShadows(True)
- vis.HideLogo()
- vis.AddGrid(x_step, y_step, nx, ny, chrono.ChCoordsysd(), chrono.ChColor(r, g, b))
- vis.AddCamera(chrono.ChVector3d(px, py, pz), chrono.ChVector3d(tx, ty, tz))
- vis.Initialize()
- vis.Run()
- vis.BeginScene()
- vis.Render()
- vis.EndScene()
- vis.SetCameraPosition(id, chrono.ChVector3d(x, y, z))
- vis.SetCameraTarget(id, chrono.ChVector3d(x, y, z))
- vis.UpdateCamera(id, chrono.ChVector3d(pos), chrono.ChVector3d(target))
- vis.SetTargetRenderFPS(fps)
- vis.SetImageOutput(True)
- vis.SetImageOutputDirectory(path)
- vis.WriteImageToFile()

canonical_examples:
- Basic VSG window: vis = chronovsg.ChVisualSystemVSG(); vis.AttachSystem(sys); vis.SetWindowSize(1280, 720); vis.AddCamera(chrono.ChVector3d(8,-8,4)); vis.Initialize()

# Skill: VSG Visualization

## Purpose

Create and configure a `ChVisualSystemVSG` window for real-time 3D rendering of PyChrono simulations. Covers window setup, camera placement, sky/grid/lighting, and the render loop.

## When to Use

Whenever the simulation needs a 3D visualization window (non-headless). This skill covers the **generic VSG system**. For wheeled vehicle scenes, prefer `veh.ChWheeledVehicleVisualSystemVSG` (see `veh/wheeled_vehicle` skill).

## Import

```python
import pychrono.core as chrono
import pychrono.vsg3d as chronovsg
```

## CRITICAL: Call Order

`ChVisualSystemVSG` is **order-sensitive**. The following sequence MUST be respected:

```
1. Create          vis = chronovsg.ChVisualSystemVSG()
2. Attach          vis.AttachSystem(sys)
3. Configure       SetWindowSize, SetWindowTitle, SetCameraAngleDeg, ...
4. AddCamera       vis.AddCamera(pos, target)        <-- BEFORE Initialize
5. AddGrid         vis.AddGrid(...)                   <-- BEFORE Initialize
6. Initialize      vis.Initialize()
7. Render loop     while vis.Run(): ...
```

**Hard rules:**

1. **`AddCamera()` MUST be called BEFORE `Initialize()`.**
   Calling it after `Initialize()` triggers:
   `Function ChVisualSystemVSG::AddCamera can only be called before initialization!`

2. **Do NOT call `vis.BindAll()` after `vis.Initialize()`.** This has been
   observed to **SIGSEGV at `vsg::BindGraphicsPipeline::record`** on the
   first render when combined with `vis.EnableShadows(True)` and any body
   carrying a mesh visual (ChBodyEasyMesh, ChVisualShapeTriangleMesh,
   ChVisualShapeModelFile). Either feature alone is fine; the
   `EnableShadows + BindAll` combination is not. The project's own
   reference demo `demo/scene/demo_SEN_HMMWV_offroad_vsg.py` calls
   `vis.Initialize()` and nothing else — match that pattern. `BindAll()`
   is intended for the Chrono collision system (`chrono.ChSystem::Initialize`
   already triggers it) and for the `ChCollisionSystem.BindAll()` shown in
   the demo. It is NOT needed on `ChVisualSystemVSG`.

3. **All `Set*` configuration calls go between `AttachSystem` and `Initialize`.** Setting properties after `Initialize` may silently have no effect.

## Window Setup

```python
vis = chronovsg.ChVisualSystemVSG()
vis.AttachSystem(sys)
vis.SetWindowSize(1280, 720)              # width, height in pixels
vis.SetWindowTitle("My Simulation")
vis.SetCameraAngleDeg(45)                 # field-of-view angle in degrees
vis.SetCameraVertical(chrono.CameraVerticalDir_Z)  # Z-up (default for PyChrono)
```

### Common Mistakes

| Wrong | Correct | Why |
|-------|---------|-----|
| `vis.SetCameraAngle(45)` | `vis.SetCameraAngleDeg(45)` | `SetCameraAngle` does not exist |
| `vis.UseSkydome(True)` | `vis.EnableSkyTexture()` | `UseSkydome` does not exist |
| `vis.Update()` | `vis.Run()` + render loop | `Update` does not exist on VSG |
| `vis.AddCOI(...)` | (remove the call) | `AddCOI` does not exist |

## Camera

```python
# Add camera BEFORE Initialize — returns a camera id (int)
cam_id = vis.AddCamera(
    chrono.ChVector3d(8, -8, 4),       # eye position
    chrono.ChVector3d(0, 0, 0),        # look-at target (default: origin)
)
```

Move the camera at runtime (after Initialize):
```python
vis.SetCameraPosition(cam_id, chrono.ChVector3d(new_x, new_y, new_z))
vis.SetCameraTarget(cam_id, chrono.ChVector3d(tx, ty, tz))
# or update both at once:
vis.UpdateCamera(cam_id, new_pos, new_target)
```

## Sky and Background

```python
# Option 1: procedural sky dome (default mode)
vis.EnableSkyTexture()                              # SkyMode::DOME

# Option 2: custom sky dome texture
vis.SetSkyDomeTexture("sky_texture.hdr", 0.0)      # filename, sun azimuth (rad)

# Option 3: sky box
vis.SetSkyBoxTexture("skybox.hdr", 0.0)

# Option 4: solid background color (no sky)
vis.SetBackgroundColor(chrono.ChColor(0.2, 0.3, 0.4))
```

**`UseSkydome(True)` does NOT exist.** Use `EnableSkyTexture()` instead.

## Lighting

```python
vis.SetLightIntensity(1.0)                          # global intensity [0..∞]
vis.SetLightDirection(1.5, 0.8)                     # azimuth (rad), elevation (rad)
vis.EnableShadows(True)                             # optional shadow mapping
```

## Grid (Ground Reference)

```python
vis.AddGrid(
    2.0, 2.0,                    # x_step, y_step (meters between lines)
    20, 20,                      # nx, ny (number of cells in each direction)
    chrono.ChCoordsysd(),        # position/orientation — ChCoordsysd, NOT ChVector3d
    chrono.ChColor(0.4, 0.4, 0.4),  # grid line color
)
```

**Correct signature:** `AddGrid(x_step, y_step, nx, ny, ChCoordsysd, ChColor)`

| Wrong | Correct | Why |
|-------|---------|-----|
| `AddGrid(10.0, 100, ChColor(...), True, True, True)` | `AddGrid(10.0, 10.0, 20, 20, ChCoordsysd(), ChColor(...))` | Wrong param count and types |
| 5th arg: `ChVector3d(0,0,0)` | 5th arg: `chrono.ChCoordsysd()` | Must be `ChCoordsysd` (pos+rot), not `ChVector3d` |

To place the grid at a custom position/orientation:
```python
grid_frame = chrono.ChCoordsysd(
    chrono.ChVector3d(0, 0, 0),       # position
    chrono.QUNIT,                      # rotation (identity)
)
vis.AddGrid(2.0, 2.0, 20, 20, grid_frame, chrono.ChColor(0.4, 0.4, 0.4))
```

## Render Loop — Use `run_recording_loop`, do NOT write `while vis.Run():` by hand

The low-level VSG render primitives are `vis.Run()` + `vis.BeginScene()`
+ `vis.Render()` + `vis.EndScene()`. **Do not compose your own main loop
out of them.** The project provides a single helper that owns the loop,
throttles rendering, pumps the sensor manager, and runs mp4 recorder
cleanup in a `finally` block:

```python
from chrono_code.utils import run_recording_loop

run_recording_loop(
    sys,
    duration=10.0,
    time_step=0.001,
    vis=vis,                 # any ChVisualSystemVSG subclass (generic, vehicle, etc.)
    manager=manager,         # optional: ChSensorManager
    recorders=[rec_a, rec_b],# optional: list of CameraRecorder from setup_preview_camera
    render_fps=50.0,         # 50 Hz VSG rendering regardless of physics dt
)
```

What `run_recording_loop` does for you:

1. `while sys.GetChTime() < duration AND (vis is None OR vis.Run()):`
   exit on either condition.
2. Calls `vis.BeginScene()/Render()/EndScene()` every `N = round(1/(dt*render_fps))`
   physics steps — **not** every step.
3. Calls `sys.DoStepDynamics(time_step)` every iteration **only when
   `step_fn=` is not passed**. If you pass `step_fn=`, the default
   `DoStepDynamics` call is REPLACED by your callback — the callback now
   owns the physics advance. See the "step_fn owns the advance" rule
   below.
4. Calls `manager.Update()` + `recorder.tick()` for each recorder AFTER
   the physics step, so sensors see post-step body poses.
5. Guarantees `recorder.close()` runs in a `finally` — essential for
   mp4 `moov` atom finalization under timeout.

**Why this matters:** the generated `simulation.py` runs under a 300 s
`ExecutionAgent` wall-clock timeout. On timeout the subprocess gets
`SIGKILL`, which bypasses `finally` blocks. A naive inline loop that
renders every physics step takes **> 300 s wall-clock** for a 30 s sim
at 0.001 dt (VSG pegged to display refresh), triggers the timeout, loses
any in-flight mp4 frames, and the VLM review step then rejects the
48-byte stub with HTTP 400. `run_recording_loop(render_fps=50)` renders
1 frame per 20 physics steps, cutting the cost 20× so the sim finishes
in ~30 s wall-clock and `finally` has a chance to write `moov`.

### `step_fn` owns the physics advance — READ BEFORE PASSING `step_fn=`

When you pass `step_fn=...`, `run_recording_loop` no longer calls
`sys.DoStepDynamics(time_step)` for you — the default step_fn (which
does exactly that one call) is **replaced wholesale** by your callback.
The symptom of a `step_fn` that forgets to advance physics is that
`sys.GetChTime()` stays at 0.0 forever, the loop never hits the
`sim_time >= duration` exit, and `vis` renders a frozen scene. The
robot / motors look wired but nothing moves.

Two valid shapes, and they are not interchangeable:

**Shape A — URDF robot / raw ChBody scenes (no vehicle wrapper).** Your
step_fn must end with an explicit `sys.DoStepDynamics(time_step)`:

```python
from chrono_code.utils import run_recording_loop

def robot_step(step_index, sim_time):
    go2.actuate(stand_action)       # motor command / policy output
    sys.DoStepDynamics(time_step)   # REQUIRED — no one else advances physics

run_recording_loop(
    sys, duration=10.0, time_step=1e-3,
    vis=vis, manager=manager, step_fn=robot_step, render_fps=50.0,
)
```

**Shape B — vehicle / terrain scenes with Synchronize/Advance.** The
`veh.*` wrapper's `Advance()` internally calls `DoStepDynamics` on the
wrapper-owned `ChSystem`. Do NOT also call `sys.DoStepDynamics()` here —
that double-steps the physics with stale driver inputs.

```python
from chrono_code.utils import run_recording_loop

def vehicle_step(step_index, sim_time):
    driver_inputs = driver.GetInputs()
    driver.Synchronize(sim_time)
    terrain.Synchronize(sim_time)
    hmmwv.Synchronize(sim_time, driver_inputs, terrain)
    vis.Synchronize(sim_time, driver_inputs)
    driver.Advance(time_step)
    terrain.Advance(time_step)
    hmmwv.Advance(time_step)        # internally calls DoStepDynamics
    vis.Advance(time_step)

run_recording_loop(
    hmmwv.GetSystem(),
    duration=30.0,
    time_step=1e-3,
    vis=vis,
    manager=manager,
    recorders=recorders,
    step_fn=vehicle_step,
    render_fps=50.0,
)
```

Rule of thumb: if your scene has no `veh.*` wrapper whose `Advance()`
internally ticks the system, your `step_fn` **must** call
`sys.DoStepDynamics(time_step)` itself. If unsure, you almost certainly
want Shape A.

### Image output (per-frame PNG)

If the simulation needs per-frame PNG output (not mp4), pass an
`on_step` callback that calls `vis.WriteImageToFile()`:

```python
vis.SetImageOutput(True)
vis.SetImageOutputDirectory("output/frames")

run_recording_loop(
    sys, duration=5.0, time_step=0.001,
    vis=vis,
    on_step=lambda i, t: vis.WriteImageToFile(),
)
```

### VSG low-level primitives (reference only)

For completeness, the primitives `run_recording_loop` composes under
the hood are `vis.Run()`, `vis.BeginScene()`, `vis.Render()`,
`vis.EndScene()`. `vis.Update()` does NOT exist on VSG. Unless you have
a documented reason to inline the loop, **use `run_recording_loop`**.

## Optional Features

```python
vis.HideLogo()                        # remove Chrono logo watermark
vis.EnableShadows(True)               # shadow mapping
vis.SetTargetRenderFPS(60.0)          # hint for render pacing
vis.EnableFullscreen(True)            # fullscreen mode
vis.SetGuiFontSize(14)                # GUI text size
vis.SetModelScale(1.0)                # global model scale
```

## Minimal Example

```python
import pychrono.core as chrono
import pychrono.vsg3d as chronovsg
from chrono_code.utils import run_recording_loop

sys = chrono.ChSystemNSC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

# ... add bodies, joints, etc. ...

vis = chronovsg.ChVisualSystemVSG()
vis.AttachSystem(sys)
vis.SetWindowSize(1280, 720)
vis.SetWindowTitle("Simulation")
vis.SetCameraAngleDeg(45)
vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
vis.SetLightIntensity(1.0)
vis.EnableSkyTexture()
vis.AddGrid(2.0, 2.0, 20, 20, chrono.ChCoordsysd(), chrono.ChColor(0.4, 0.4, 0.4))
vis.AddCamera(chrono.ChVector3d(8, -8, 4), chrono.ChVector3d(0, 0, 0))
vis.Initialize()

run_recording_loop(sys, duration=10.0, time_step=0.001, vis=vis, render_fps=50.0)
```

## Skill Dependencies

- `mbs/system_create` — ChSystem creation (must exist before AttachSystem)
- `mbs/simulation_loop` — data logging and post-processing patterns
- `sens/sensor_manager` — if combining VSG window with sensor cameras (separate concerns)
- `sens/camera` — sensor camera recording (VSG window is NOT a sensor camera)
