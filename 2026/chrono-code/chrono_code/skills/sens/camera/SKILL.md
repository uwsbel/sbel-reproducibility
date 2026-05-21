---
name: camera
description: Attach an RGB preview+recording camera to a body using the project helper setup_preview_camera (returns a CameraRecorder). NON-FSI scenes only — for FSI/SPH scenes use fsi/sph Pattern G + chrono_code.utils.vsg_recording instead, since sensor cameras (OptiX) cannot render SPH particles.
compatibility: pychrono >= 8.0
metadata:
  domain: sens
---

# Skill: Camera Sensor

## Purpose

Attach an RGB camera (live preview window + recorded `.mp4`) to a simulation body.

## When to Use

Any time the user asks for a camera, a camera view, a recording of the scene, a top-down view, or any vision-based sensing.

## When NOT to Use — FSI / SPH scenes

If the scene contains SPH fluid (the `fsi/sph` skill is also relevant —
plan mentions water tank, pool, splash, floating body, etc.), do NOT use
`setup_preview_camera` for the primary mp4 of the scene. `ChCameraSensor`
renders via OptiX, but `ChSphVisualizationVSG` is a VSG-only plugin —
SPH particles never reach the OptiX scene tree, so the recorded mp4
shows an empty tank with no water.

For FSI scenes use the VSG-side recording helpers in
`chrono_code.utils.vsg_recording` (`hide_vsg_gui`, `lock_side_camera`,
`setup_vsg_recording`); see `fsi/sph` Pattern G. A sensor camera may
still be added on top when the scene needs sensor data for ML training
where missing fluid is acceptable, but the **primary video output of an
FSI scene is always VSG-recorded**, never sensor-recorded.

## Hard Rule

**Do not construct `ChCameraSensor`, `ChFilterVisualize`, `ChFilterSave`, `ChFilterRGBA8Access`, or a look-at rotation by hand.** The project provides a single helper that does all of this, including direct mp4 recording via `cv2.VideoWriter` and cleanup when the user closes the VSG window. Call it.

```python
from chrono_code.utils import setup_preview_camera
```

Any generated `simulation.py` that creates a camera MUST import and call `setup_preview_camera`. If the code you are writing contains `sens.ChCameraSensor(`, `ChFilterVisualize`, `ChFilterSave`, or `ChFilterRGBA8Access` directly, you are doing it wrong — delete that code and call the helper instead.

## Exact Helper Signature

```python
setup_preview_camera(
    manager,
    attach_body,
    target_pos,
    cam_pos,
    up_direction,
    *,
    update_rate,
    name="camera",
    width=1280,
    height=720,
    fov=1.408,
    fps=None,
    output_root="cam",
    preview=True,
    background_recording=False,
)
```

Hard rules:
- Do NOT pass legacy kwargs such as `system=`, `offset=`, `target=`, `resolution_width=`, `resolution_height=`, or `horizontal_fov=`.
- `output_root="cam"` is resolved relative to the running script directory, not `os.getcwd()`.
- Keep `background_recording=False` unless the caller truly cannot invoke `recorder.tick()` in the loop. The default path is intentionally single-threaded for stability.

## How to Call It

```python
from chrono_code.utils import setup_preview_camera

recorder = setup_preview_camera(
    manager,                                  # existing sens.ChSensorManager
    cam_body,                                 # body the camera rides on
    target_pos=main_body.GetPos(),            # geometric center of the main simulation body
    cam_pos=chrono.ChVector3d(0, -5, 1),      # where the camera eye sits in world coords
    up_direction=chrono.ChVector3d(0, 0, 1),  # world up (usually +Z)
    update_rate=1.0 / step_size,              # derive from the active simulation dt
    name="chase_cam",                         # becomes cam/chase_cam.mp4
)
```

That single call:
- computes the look-at rotation from `cam_pos` / `target_pos` / `up_direction`
- builds the `ChCameraSensor` and registers it with `manager`
- attaches a live preview window (`ChFilterVisualize`) when `preview=True`, plus a `ChFilterRGBA8Access` for direct frame access
- returns a `CameraRecorder` — an opaque object exposing only `tick()` and `close()`
- registers a process-wide atexit + SIGINT/SIGTERM hook that will call `close()` on any recorder that is still open when the process exits (e.g., crash, or the user closes the VSG window)

Output always lands in `cam/<name>.mp4` relative to the running script directory unless `output_root` is absolute. You do not need to create the directory yourself. The mp4 is written **directly** by `cv2.VideoWriter` — there are no intermediate PNG files and no external `ffmpeg` invocation.

## Required Loop Integration — Use `run_recording_loop`

**Hard rule: do NOT write your own `while vis.Run():` loop.** The project
provides a second helper, `run_recording_loop`, that owns the entire main
loop, throttles VSG rendering, pumps the sensor manager + every recorder,
and guarantees `recorder.close()` runs in a `finally` block. Use it.

```python
from chrono_code.utils import setup_preview_camera, run_recording_loop

recorder = setup_preview_camera(manager, cam_body, ...)

run_recording_loop(
    sys,
    duration=simulation_duration,
    time_step=step_size,
    vis=vis,                     # optional: ChVisualSystemVSG or subclass, or None
    manager=manager,             # optional: ChSensorManager
    # recorders=...              # OMIT to auto-collect ALL setup_preview_camera()
                                  # results; see below. Pass an explicit list only
                                  # to override (e.g. tests).
    render_fps=50.0,             # throttles VSG to ~50 Hz regardless of physics rate
)
```

## Multi-camera per step (plan-driven)

Every `SimulationStep` in the plan carries a non-empty `cameras` list
(see `chrono_code/models/plan.py` — `SimulationStep.cameras:
List[CameraPose]` with `min_length=1`). That list is serialized into
`step_context.json` as `step_cameras`, and the codegen MUST emit one
`setup_preview_camera(...)` call per entry. One-camera-per-step is the
old (broken) default and produces cam/*.mp4 files from a single angle
only, which hides occluded regions from the VLM reviewer.

### Canonical pattern — iterate `step_context.step_cameras`

Always resolve `step_context.json` against the script's own directory via
`Path(__file__).parent`, **not** via `open("step_context.json")`. The
file lives next to `simulation.py` inside the iteration dir, but users
(and the execution harness for debugging) may run the script from any
cwd — a bare `open("step_context.json")` only works when cwd happens to
be the iteration dir and silently `FileNotFoundError`s everywhere else.

```python
import json
from pathlib import Path
from chrono_code.utils import setup_preview_camera, run_recording_loop

_STEP_CONTEXT_PATH = Path(__file__).resolve().parent / "step_context.json"
with _STEP_CONTEXT_PATH.open() as f:
    step_cameras = json.load(f)["step_cameras"]  # list of {position, target, up}

# One fixed world-frame body per camera — reuse or create per entry.
for i, cam in enumerate(step_cameras):
    cam_body = chrono.ChBody()
    cam_body.SetFixed(True)
    cam_body.SetPos(chrono.ChVector3d(*cam["position"]))
    sys.AddBody(cam_body)

    setup_preview_camera(
        manager,
        attach_body=cam_body,
        cam_pos=chrono.ChVector3d(*cam["position"]),
        target_pos=chrono.ChVector3d(*cam["target"]),
        up_direction=chrono.ChVector3d(*cam.get("up", [0, 0, 1])),
        update_rate=1.0 / time_step,
        name=f"cam_{i}",          # -> cam/cam_0.mp4, cam/cam_1.mp4, ...
    )

# No recorders=[...] — registry auto-collects all N recorders above.
run_recording_loop(sys, duration=simulation_duration, time_step=time_step,
                   vis=vis, manager=manager, render_fps=50.0, step_fn=step_fn)
```

`step_fn=step_fn` above refers to a user-defined callback (robot
actuation, vehicle Synchronize/Advance, etc.). The callback **owns the
physics advance**: passing `step_fn=` replaces the default
`sys.DoStepDynamics(time_step)` call inside `run_recording_loop`, so
unless your step_fn delegates to a `veh.*` wrapper's `Advance()` (which
internally steps the system), your step_fn must end with
`sys.DoStepDynamics(time_step)` explicitly. See the `vsg` skill's
"`step_fn` owns the physics advance" section for the URDF-robot vs.
vehicle shapes. If you have no per-step orchestration, omit `step_fn=`
entirely — the default ticks `DoStepDynamics` for you.

Every recorder returned by `setup_preview_camera` is registered in a
process-wide list. `run_recording_loop(..., recorders=None)` (the
default) ticks every registered recorder automatically, so you cannot
"forget to wire in" one. If you pass an explicit `recorders=[r1]` list
while more recorders exist, `run_recording_loop` logs a WARNING and the
omitted recorders write zero frames — that is almost always a bug. The
usual fix is to drop the argument.

### Picking camera angles

Plan-level cameras are ALL world-frame — each one attaches to a fixed
`ChBody()` you create just for it. Pick 2–3 entries per step covering
complementary VIEWING DIRECTIONS (e.g. wide-from-NE + wide-from-NW +
top-down), NOT zoom levels of the same angle. Duplicate coverage adds
mp4 files without adding VLM signal.

Chassis-attached onboard cameras (moving with the vehicle) are a
different mechanism: in `demo/scene/demo_SEN_HMMWV_offroad_vsg.py:502-546`
the demo creates onboard + left-surround + top-down rigs by passing
`hmmwv.GetChassisBody()` as `attach_body` so cameras follow the
vehicle in its local frame. Those are codegen-time decisions (when a
step's `description` says "onboard POV" or "chase-style view") and do
NOT appear in the plan's `step_cameras` list.

### File lifecycle — don't play the mp4 until the loop finishes

`setup_preview_camera` writes frames to `cam/<name>.inprogress.mp4` while
the sim is running; only after `recorder.close()` finalizes the moov atom
does the helper atomically rename to `cam/<name>.mp4`. Reading the file
mid-run will either show `.inprogress.mp4` (clearly not done) or the final
`.mp4` (guaranteed playable) — never a half-written file under the final
name. A lingering `*.inprogress.mp4` after a run means the process was
`SIGKILL`-ed before `finally`; that file is corrupt and should be deleted.

### Why you MUST use `run_recording_loop` and not inline the loop

The generated `simulation.py` is run under `ExecutionAgent` with a hard
**300 s wall-clock timeout**. On timeout the subprocess is **SIGKILL**ed,
which bypasses `finally` blocks and any atexit/SIGTERM handlers. If the
mp4 writer has not been explicitly `release()`-d by the time SIGKILL
fires, `cv2` leaves a 48-byte `ftyp`+`free`+`mdat` stub on disk — no
`moov` atom, nothing decodable. The downstream VLM review step then
receives this stub, rejects it with HTTP 400 "invalid argument", and
the whole step_review fails.

The naive inline loop that every previous codegen copied —

```python
while vis.Run() and sys.GetChTime() < sim_duration:
    manager.Update()
    recorder.tick()
    vis.BeginScene(); vis.Render(); vis.EndScene()   # every physics step!
    sys.DoStepDynamics(step_size)
```

— renders VSG **every physics step**. With `step_size=0.001` and a 30 s
sim that is 30,000 VSG frames, which pins the loop to the display's 60 Hz
refresh and makes a 30 s sim take **500+ s wall-clock** — well past the
300 s timeout. SIGKILL → corrupt mp4 → failed review. Observed in the
wild, repeatedly. `run_recording_loop(render_fps=50)` renders 1 frame per
20 physics steps instead, cutting the VSG cost by 20× and letting the sim
finish in 30-60 s wall time.

### If you really must inline the loop (DISCOURAGED)

Only for special cases where `run_recording_loop`'s API does not fit
(e.g., you need to switch VSG attachments mid-run). You are still
responsible for:

1. **Render throttling**: `if step_number % max(1, round(1/(dt*render_fps))) == 0: vis.Render()`.
2. **try/finally cleanup**: `recorder.close()` in a `finally`.
3. **Exit bound**: both `vis.Run()` AND `sys.GetChTime() < duration` checked every step.

If any of these is missing, your mp4 will be corrupt under timeout. The
helper exists precisely so you do not have to get all three right
manually.

### Encoding detail

`setup_preview_camera` defaults `fps = int(round(update_rate))` so video
time equals wall-clock time. Do not override `fps=30` — it causes visible
jitter when `update_rate` is much higher than 30 Hz.

## Stability Rules For Preview + VSG

If the script already uses a VSG main window (for example `veh.ChWheeledVehicleVisualSystemVSG` or `chronovsg.ChVisualSystemVSG`), the sensor camera preview is still allowed, but you must keep the integration conservative:

- Use the helper exactly as provided; do not hand-roll sensor filters or extra worker threads.
- Keep the default `background_recording=False` and call `recorder.tick()` explicitly after `manager.Update()`.
- Always call `recorder.close()` in `finally:` before clearing or dropping `manager`, `vis`, the vehicle wrapper, or the Chrono system.
- If a native crash occurs only at shutdown, first verify the order: `recorder.close()` -> drop sensor-side objects -> drop VSG/system objects.
- Do not treat the sensor preview window as a substitute for the main VSG camera. It is a separate rendering path with its own native resources.

## Required Inputs From The User's Prompt

Map the user's request to the helper's arguments:

- `cam_pos` ← user's `camera_position` / "eye" / "viewpoint"
- `target_pos` ← **always** `main_body.GetPos()` (the geometric center of the main simulation body, not a hardcoded world point, not the ground). This guarantees the sensor camera and any VSG camera render from the same viewpoint.
- `up_direction` ← `chrono.ChVector3d(0, 0, 1)` by default; only change if the user explicitly asks for a tilted horizon.
- `update_rate` ← `1.0 / step_size` (the simulation dt), not an arbitrary 30.
- `name` ← a short slug describing the shot: `chase_cam`, `top_down`, `front_left`, etc. It becomes the mp4 filename.

If the user describes multiple viewpoints ("a chase cam and a top-down"), call `setup_preview_camera` once per viewpoint with a different `name=` — each call produces its own mp4 and its own cleanup.

### Mapping Pose Descriptions

Treat `pose_description` as a modifier on `cam_pos`, not on the rotation — the helper handles rotation internally from the three vectors you pass in:

- `keep horizon level` → leave `up_direction` as world `+Z`
- `top-down` → put `cam_pos` above the target (larger `z`); the helper's look-at still resolves the orientation
- `front-left-above` → choose `cam_pos` accordingly; no rotation override needed

## Required Surrounding Setup

The helper needs a `ChSensorManager` (see the `sens/sensor_manager` skill) and a body to attach the camera to. A common pattern:

```python
import pychrono.sensor as sens

manager = sens.ChSensorManager(sys)
# ... lighting etc. per sensor_manager skill ...

cam_body = chrono.ChBody()
cam_body.SetFixed(True)
sys.AddBody(cam_body)

recorder = setup_preview_camera(
    manager, cam_body,
    target_pos=main_body.GetPos(),
    cam_pos=chrono.ChVector3d(0, -5, 1),
    up_direction=chrono.ChVector3d(0, 0, 1),
    update_rate=1.0 / step_size,
    name="main_cam",
)
```

Inside the simulation loop you still call `manager.Update()` each step, same as for any sensor, and then call `recorder.tick()` to record a frame.

## Important: Bodies Must Be Independent for Sensor Rendering

The sensor camera renders **independently from VSG**:

- **VSG** renders any `AddVisualShape` attached to any body.
- **Sensor renderer** only renders bodies that have **collision geometry** — visual shapes alone are not sufficient.

If a structural element (axle, rod, post) must appear in the sensor camera, create it as an **independent body** with both visual and collision shapes, connected via a joint (e.g., `ChLinkLockLock`). Do not attach it as a visual shape on another body — it will be invisible in the sensor view.

**Correct:**
```python
body_struct = chrono.ChBodyEasyCylinder(
    chrono.ChAxis_X, radius, length,
    small_density, True, False, mat
)
body_struct.SetPos(midpoint_pos)
sys.AddBody(body_struct)

joint = chrono.ChLinkLockLock()
joint.Initialize(body_struct, main_body,
                 chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
sys.AddLink(joint)
```

**Wrong (invisible in sensor view):**
```python
disc.AddVisualShape(chrono.ChVisualShapeCylinder(...))
```

## Static Scene 5-Camera Layout (layout = "static_5")

For scenes with only static objects, use 5 world-frame cameras: 4 cardinal directions + 1 top-down. All use the `setup_preview_camera` helper.

```python
from chrono_code.utils import setup_preview_camera

# Scale camera positions with scene/room size
X, Y, Z = 2.5, 2.5, 2.0   # adjust based on room_size
Z_top = 3.0                 # top-down height

sensor_target = chrono.ChVector3d(0.0, 0.0, 0.5)  # scene center

camera_positions = {
    "camera+x": chrono.ChVector3d(+X, 0.0, +Z),
    "camera-x": chrono.ChVector3d(-X, 0.0, +Z),
    "camera+y": chrono.ChVector3d(0.0, +Y, +Z),
    "camera-y": chrono.ChVector3d(0.0, -Y, +Z),
    "top_down": chrono.ChVector3d(0.0, 0.0, +Z_top),
}

for name, eye_pos in camera_positions.items():
    cam_body = chrono.ChBody()
    cam_body.SetFixed(True)
    cam_body.SetPos(eye_pos)
    system.AddBody(cam_body)

    setup_preview_camera(
        manager, cam_body,
        target_pos=sensor_target,
        cam_pos=eye_pos,
        up_direction=chrono.ChVector3d(0, 0, 1),
        update_rate=1.0 / timestep,
        name=name,
    )
```

When `room_size` is set in the plan, scale positions: `X = 0.6 × room_size[0]/2`, `Y = 0.6 × room_size[1]/2`, `Z = 0.4 × room_size[2]`, `Z_top = 0.8 × room_size[2]`.

All 5 views must be created for static scenes. Do not drop cameras.

### Walled / enclosed scenes (override the layout above)

When the scene includes wall bodies (e.g. `room_wall_*`, `wall_north`,
`wall_east`), DO NOT use the four cardinal cameras at `±X / ±Y` — those
positions sit *outside* the room and the opaque walls will produce blank
frames. Instead emit **2-3 cameras at the inner face of perpendicular
walls** so the review agent has lateral coverage from inside the room.

```python
# room half-extents from plan: Lx, Ly, Lz = room_size[0]/2, room_size[1]/2, room_size[2]
wall_inset = 0.2   # m, distance from inner wall surface
h          = 0.6 * room_size[2]   # eye height

cam_positions = {
    "inside_minus_x_wall": chrono.ChVector3d(-Lx + wall_inset, 0.0, h),
    "inside_minus_y_wall": chrono.ChVector3d(0.0, -Ly + wall_inset, h),
    # add a third perpendicular wall view if you need 3 cameras:
    # "inside_plus_x_wall":  chrono.ChVector3d(+Lx - wall_inset, 0.0, h),
}
cam_targets = {
    "inside_minus_x_wall": chrono.ChVector3d(+Lx, 0.0, h),
    "inside_minus_y_wall": chrono.ChVector3d(0.0, +Ly, h),
    # "inside_plus_x_wall":  chrono.ChVector3d(-Lx, 0.0, h),
}
```

Add a `top_down` camera ONLY if the plan has no ceiling body
(`room_ceiling` / `roof` / etc.). An opaque ceiling occludes top-down
views the same way walls occlude side views — if a ceiling exists, omit
`top_down` and rely on the inside-wall pair.

The `setup_preview_camera()` call is otherwise identical (it accepts any
`cam_pos` / `target_pos`); only the eye / target geometry changes.

## See Also

- `../sensor_manager/` — `ChSensorManager` setup and lighting

## API Contract

allowed_imports:
- `from chrono_code.utils import setup_preview_camera, run_recording_loop`

required_in_generated_code:
- `run_recording_loop(sys, duration=..., time_step=..., vis=..., manager=..., recorders=[...])` drives the main loop.
- Do NOT write `while vis.Run():` directly. The helper does.
- Do NOT call `recorder.tick()` or `recorder.close()` manually unless you have a documented reason to inline the loop. The helper owns that cadence.
