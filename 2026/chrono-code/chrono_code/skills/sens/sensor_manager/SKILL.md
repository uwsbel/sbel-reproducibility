---
name: sensor_manager
description: Create and configure a ChSensorManager, add scene lighting, and run sensor update loops. NON-FSI scenes only — FSI/SPH scenes must use fsi/sph Pattern G (VSG-only recording) because OptiX cannot render SPH particles.
compatibility: pychrono >= 8.0
metadata:
  domain: sens
---

# Skill: Sensor Manager

## Purpose

Create and configure a `ChSensorManager` to oversee all sensors in a simulation, add scene lighting for camera-based sensors, and run the sensor update loop.

## When to Use

Use when building any simulation that includes sensors (camera, lidar, radar, IMU, GPS, etc.). The sensor manager must be created before adding any sensors.

## When NOT to Use — FSI / SPH scenes

If the scene includes **SPH fluid** (the `fsi/sph` skill is loaded, plan
mentions water / pool / fluid / floating body), do NOT use this skill for
the primary mp4 recording. `ChCameraSensor` (OptiX) cannot render SPH
particles — `ChSphVisualizationVSG` is a VSG-pipeline plugin and never
reaches the OptiX scene tree, so any sensor-camera mp4 of an FSI scene
shows an empty tank with no water.

For FSI scenes, follow `fsi/sph` **Pattern G — VSG-Only mp4 Recording**
instead, using the helpers in `chrono_code.utils.vsg_recording`
(`hide_vsg_gui`, `lock_side_camera`, `setup_vsg_recording`).

A sensor camera MAY still be added on top of the VSG recording when the
scene also needs sensor data for ML training (e.g. depth maps, where
missing fluid is acceptable for the downstream task) — but the **main
video** of an FSI scene is always VSG-only.

## Key Concepts

### ChSensorManager Creation

```python
import pychrono.sensor as sens

manager = sens.ChSensorManager(physical_system)
```

### Scene Lighting

Camera-based sensors require scene lighting to function properly:

```python
# AddPointLight: (ChVector3f position, ChColor color, float range)
manager.scene.AddPointLight(
    chrono.ChVector3f(2, 2.5, 100),
    chrono.ChColor(1.0, 1.0, 1.0),
    500.0
)

# SetAmbientLight: takes ChVector3f, NOT ChColor
manager.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))

# AddDirectionalLight: (ChColor color, float elevation, float azimuth)
# Note the order: color first, then two scalar angles in radians.
manager.scene.AddDirectionalLight(
    chrono.ChColor(0.8, 0.8, 0.7),
    0.45,
    -1.10,
)
```

Directional light on the sensor scene is **not** the same API style as VSG lighting:

- `manager.scene.AddDirectionalLight(...)` expects `ChColor` + 2 scalar angles
- `vis.SetLightDirection(...)` configures VSG lighting and also uses scalar angles
- Do **not** pass a direction vector or an up vector to the sensor-scene directional light API

### Common Mistakes

| Wrong | Correct | Why |
|-------|---------|-----|
| `SetAmbientLight(chrono.ChColor(0.5, 0.5, 0.5))` | `SetAmbientLight(chrono.ChVector3f(0.5, 0.5, 0.5))` | Takes `ChVector3f`, NOT `ChColor` |
| `SetAmbientLight(chrono.ChVector3f(0.5, 0.5, 0.5), 1.0)` | `SetAmbientLight(chrono.ChVector3f(0.5, 0.5, 0.5))` | Only 1 argument (the vector), no intensity param |
| `AddPointLight(chrono.ChVector3d(...), ...)` | `AddPointLight(chrono.ChVector3f(...), ...)` | Position must be `ChVector3f`, NOT `ChVector3d` |
| `AddDirectionalLight(chrono.ChVector3f(...), chrono.ChColor(...), chrono.ChVector3f(...))` | `AddDirectionalLight(chrono.ChColor(...), elevation, azimuth)` | Sensor-scene directional light takes color first, then two scalar angles |
| `AddDirectionalLight(direction, color, up)` | `AddDirectionalLight(color, elevation, azimuth)` | This binding does not accept a direction vector or an up vector |

### Sensor Update Loop

In the simulation loop, update all sensors before stepping the system:

```python
# In your while loop:
manager.Update()  # Update all sensors
sys.DoStepDynamics(dt)
```

## Minimal Example

```python
import pychrono.core as chrono
import pychrono.sensor as sens

# Create the physical system
sys = chrono.ChSystemNSC()
sys.SetGravityY()

# Create sensor manager
manager = sens.ChSensorManager(sys)

# Add scene lighting for camera sensors
intensity = 1.0
manager.scene.AddPointLight(
    chrono.ChVector3f(2, 2.5, 100),
    chrono.ChColor(intensity, intensity, intensity),
    500.0
)
manager.scene.AddDirectionalLight(
    chrono.ChColor(0.8, 0.8, 0.7),
    0.45,
    -1.10,
)

# Now add sensors to manager:
# manager.AddSensor(sensor)

# In simulation loop:
# manager.Update()
# sys.DoStepDynamics(dt)
```

## Background Settings (Optional)

```python
bg = sens.Background()
bg.mode = sens.BackgroundMode_SOLID_COLOR      # module-level constant
bg.color_zenith = chrono.ChVector3f(0, 0, 0)
bg.color_horizon = chrono.ChVector3f(0, 0, 0)
manager.scene.SetBackground(bg)
manager.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))
```

## See Also

- `../camera/` - Camera sensor setup via `setup_preview_camera`

## API Contract

allowed_classes:
- sens.ChSensorManager
- sens.Background
- chrono.ChSystemNSC
- chrono.ChVector3f
- chrono.ChColor

allowed_methods:
- sens.ChSensorManager(physical_system)
- manager.scene.AddPointLight(chrono.ChVector3f, chrono.ChColor, float)
- manager.scene.AddDirectionalLight(chrono.ChColor, float, float)
- manager.scene.SetAmbientLight(chrono.ChVector3f)
- manager.scene.SetBackground(sens.Background)
- manager.Update()
- manager.AddSensor(sensor)
- sys.SetGravityY()
- sys.DoStepDynamics(dt)

allowed_constants:
- sens.BackgroundMode_SOLID_COLOR
