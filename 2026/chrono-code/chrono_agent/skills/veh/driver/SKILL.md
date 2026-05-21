---
name: driver
description: Configure interactive, path-following, and data-based driver systems for wheeled and tracked vehicles.
compatibility: pychrono >= 8.0
metadata:
  domain: veh
---

# Skill: Vehicle Driver Systems

## Purpose

Configure driver systems to control wheeled or tracked vehicles — interactive (keyboard), path-following (autonomous), or pre-recorded data. Driver systems work the same for both vehicle types via `veh_obj.GetVehicle()`.

## When to Use

When the user asks to drive a vehicle, follow a path, or record vehicle inputs.

## HARD RULE — No human-in-the-loop in batch / headless runs

This codegen pipeline runs simulations **headless** (Xvfb / no
keyboard / no GUI input loop). `ChInteractiveDriver` is a
keyboard-input driver — its `GetInputs()` returns whatever the user
last typed at the SDL window, and **in a headless run that's always
zero**. The vehicle never accelerates, never steers, and the run
looks identical to a "missing physics" bug.

**Pick one of the autonomous drivers below for every generated
simulation:**

| Need | Use |
|------|-----|
| Open-loop schedule (fixed throttle/brake/steering at known times) | `veh.ChDataDriver` with `veh.DataDriverEntry(t, s, th, br, g)` entries |
| Closed-loop on vehicle state (e.g. throttle depends on x-position, brake when speed > X) | Plain `veh.DriverInputs()` struct, write fields each step |
| Path following with target speed | `veh.ChPathFollowerDriver` |

**Do NOT** instantiate `ChInteractiveDriver` in generated code. The
Interactive Driver section below is documented for reference only —
it is the wrong tool for this pipeline.

## Interactive Driver (ChInteractiveDriver) — REFERENCE ONLY, NOT FOR GENERATED CODE

For human-in-the-loop or scripted driving:

```python
driver = veh.ChInteractiveDriver(veh_obj.GetVehicle())

# Time to reach max steering/throttle/braking
steering_time = 1.0   # seconds to go 0 -> +1 steering
throttle_time = 1.0   # seconds to go 0 -> +1 throttle
braking_time = 0.3    # seconds to go 0 -> +1 brake

driver.SetSteeringDelta(render_step_size / steering_time)
driver.SetThrottleDelta(render_step_size / throttle_time)
driver.SetBrakingDelta(render_step_size / braking_time)
driver.Initialize()

# In simulation loop
driver_inputs = driver.GetInputs()
driver.Synchronize(time)
driver.Advance(step_size)
```

## Path-Follower Driver (ChPathFollowerDriver)

For autonomous path following with cruise control:

```python
# Create path (ISO double lane change to left)
path = veh.DoubleLaneChangePath(
    start,      # initial position ChVector3d
    13.5,       # length
    4.0,        # width
    11.0,       # offset
    50.0,       # total length
    True        # to left
)

# Create path-following driver
target_speed = 12  # m/s
driver = veh.ChPathFollowerDriver(
    veh_obj.GetVehicle(),
    path,
    "my_path",
    target_speed
)

# Configure controllers
driver.GetSteeringController().SetLookAheadDistance(5.0)
driver.GetSteeringController().SetGains(0.8, 0, 0)   # KP, KI, KD
driver.GetSpeedController().SetGains(0.4, 0, 0)

driver.Initialize()

# In simulation loop
driver_inputs = driver.GetInputs()
driver.Synchronize(time)
driver.Advance(step_size)
```

### Available Paths

```python
veh.DoubleLaneChangePath(start, length, width, offset, total_length, to_left)
# Creates an ISO double lane change maneuver path
```

## Data Driver (ChDataDriver)

For pre-recorded input sequences:

```python
# Create data entries: (time, steering, throttle, braking, gear)
driver_data = veh.vector_Entry([
    veh.DataDriverEntry(0.0, 0.0, 0.0, 0.0, 0.0),
    veh.DataDriverEntry(0.5, 0.0, 0.0, 0.0, 0.0),
    veh.DataDriverEntry(0.7, 0.3, 0.7, 0.0, 0.0),
    veh.DataDriverEntry(1.0, 0.3, 0.7, 0.0, 0.0),
    veh.DataDriverEntry(3.0, 0.5, 0.1, 0.0, 0.0)
])

driver = veh.ChDataDriver(veh_obj.GetVehicle(), driver_data)
driver.Initialize()

# In simulation loop
driver_inputs = driver.GetInputs()
driver.Synchronize(time)
driver.Advance(step_size)
```

## Driver Inputs Structure

The `driver_inputs` object has these fields:
```python
driver_inputs.m_steering   # -1 to +1
driver_inputs.m_throttle   # 0 to +1
driver_inputs.m_braking    # 0 to +1
driver_inputs.m_gear       # gear index
```

## Synchronization Order

In the simulation loop, synchronize in this order:
```python
driver.Synchronize(time)
terrain.Synchronize(time)
veh_obj.Synchronize(time, driver_inputs, terrain)
vis.Synchronize(time, driver_inputs)
```

## Visualizing Path Controller Points

```python
# Get sentinel and target locations for visualization
pS = driver.GetSteeringController().GetSentinelLocation()
pT = driver.GetSteeringController().GetTargetLocation()

# Visualize with sphere markers
ballS = vis.GetSceneManager().addSphereSceneNode(0.1)
ballT = vis.GetSceneManager().addSphereSceneNode(0.1)
ballS.setPosition(irr.vector3df(pS.x, pS.y, pS.z))
ballT.setPosition(irr.vector3df(pT.x, pT.y, pT.z))
```

## Chase Camera

```python
vis.SetChaseCamera(trackPoint, distance, offset)
# trackPoint: ChVector3d — point on chassis to follow
# distance: float — camera distance behind vehicle
# offset: float — camera height offset
```

## Attaching a Driver to Vehicle Visualization

When visualizing with `veh.ChWheeledVehicleVisualSystemVSG`, attach the driver to the vis system so the HUD bars (steering / throttle / brake) render and reflect the live inputs:

```python
vis.AttachVehicle(veh_obj.GetVehicle())
vis.AttachDriver(driver)              # enables input-bar HUD
vis.Initialize()

# In the simulation loop, pass the same inputs through:
driver_inputs = driver.GetInputs()
driver.Synchronize(time)
vis.Synchronize(time, driver_inputs)  # HUD picks up from here
```

`AttachDriver` only affects visualization — the driver still drives the vehicle via the `vehicle.Synchronize(time, driver_inputs, terrain)` path regardless.

## Skill Dependencies

For vehicle setup and system creation:
- `../../mbs/system_create/` — ChSystem creation
- `../wheeled_vehicle/` — Vehicle creation + `ChWheeledVehicleVisualSystemVSG` pattern (also carries the authoritative Synchronize/Advance order for SCM-backed setups)
- `../terrain/` — Terrain creation (Rigid / SCM / CRM)

## API Contract

allowed_classes:
- veh.ChInteractiveDriver(veh_obj.GetVehicle())
- veh.ChPathFollowerDriver(veh_obj.GetVehicle(), path, "my_path", target_speed)
- veh.ChDataDriver(veh_obj.GetVehicle(), driver_data)
- veh.vector_Entry([...])
- veh.DataDriverEntry(time, steering, throttle, braking, gear)

allowed_methods:
- driver.SetSteeringDelta(render_step_size / steering_time)
- driver.SetThrottleDelta(render_step_size / throttle_time)
- driver.SetBrakingDelta(render_step_size / braking_time)
- driver.Initialize()
- driver.GetInputs()
- driver.Synchronize(time)
- driver.Advance(step_size)
- driver.GetSteeringController().SetLookAheadDistance(5.0)
- driver.GetSteeringController().SetGains(KP, KI, KD)
- driver.GetSteeringController().GetSentinelLocation()
- driver.GetSteeringController().GetTargetLocation()
- driver.GetSpeedController().SetGains(KP, KI, KD)
- veh.DoubleLaneChangePath(start, length, width, offset, total_length, to_left)
- vis.SetChaseCamera(trackPoint, distance, offset)
- vis.AttachVehicle(veh_obj.GetVehicle())
- vis.AttachDriver(driver)
- vis.Initialize()
- vis.Synchronize(time, driver_inputs)
- terrain.Synchronize(time)
- veh_obj.Synchronize(time, driver_inputs, terrain)

allowed_constants:
- driver_inputs.m_steering
- driver_inputs.m_throttle
- driver_inputs.m_braking
- driver_inputs.m_gear

allowed_utils:
- from chrono_agent.utils import setup_preview_camera
