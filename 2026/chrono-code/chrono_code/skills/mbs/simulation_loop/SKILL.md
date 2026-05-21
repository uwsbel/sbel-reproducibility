---
name: simulation_loop
description: Run the time-stepping loop, collect data, and post-process results with matplotlib or CSV.
compatibility: pychrono >= 8.0
metadata:
  domain: mbs
---

## API Contract

allowed_methods:
- sys.DoStepDynamics(dt)  # timestep passed as argument, NOT on system object

canonical_examples:
- ChSystemNSC -> sys.DoStepDynamics(dt) with dt=0.001 for high-precision MBS

# Skill: MBS Simulation Loop

## Purpose

Run the time-stepping loop, collect data, and post-process results with matplotlib or CSV output.

## When to Use

After setting up the system, bodies, joints, and visualization — to advance the simulation in time and optionally record outputs.

## Key Concepts

### Basic Loop

```python
dt = float  # time step [s]
while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    sys.DoStepDynamics(dt)
```

### System Timestep Configuration

**Timestep is set per-call, NOT on the system object:**
```python
dt = 0.001  # time step [s]
while vis.Run():
    sys.DoStepDynamics(dt)  # dt passed as argument to the stepping call
```

**Common solver/timestepper settings (if needed):**
```python
sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER)  # default
# Other types: Type_BDF, Type_RK45, etc.
```

### Timestep Guidelines


| Scenario                  | Typical dt      |
| ------------------------- | --------------- |
| High-precision mechanisms | `1e-3` (1 ms)   |
| General MBS               | `5e-3` (5 ms)   |
| Collision-heavy scenes    | `0.02` (20 ms)  |
| SMC soft contacts         | `1e-4` (0.1 ms) |


### Timed Stop

```python
duration = float  # simulation duration [s]
if sys.GetChTime() > duration:
    vis.GetDevice().closeDevice()
```

### Setting Initial Body State

```python
body.SetPos(chrono.ChVector3d(x, y, z))              # position
body.SetRot(chrono.QuatFromAngleZ(angle))             # orientation
body.SetPosDt(chrono.ChVector3d(vx, vy, vz))         # linear velocity
body.SetLinVel(chrono.ChVector3d(vx, vy, vz))         # linear velocity (alias for SetPosDt)
body.SetAngVelParent(chrono.ChVector3d(wx, wy, wz))   # angular velocity in world frame
```

### Reading Body State

```python
body.GetPos()              # ChVector3d position
body.GetPosDt()            # ChVector3d velocity
body.GetLinVel()           # ChVector3d velocity (alias for GetPosDt)
body.GetRot()              # ChQuaterniond orientation
body.GetAngVelParent()     # ChVector3d angular velocity in world frame
```

### Reading Motor State

```python
# Rotational motors (ChLinkMotorRotation*)
motor.GetMotorAngle()     # integrated angle [rad]
motor.GetMotorAngleDt()   # angular speed [rad/s]
motor.GetMotorAngleDt2()  # angular acceleration [rad/s²]

# Linear motors (ChLinkMotorLinear*)
motor.GetMotorPos()       # linear position [m]
motor.GetMotorPosDt()     # linear speed [m/s]
motor.GetMotorPosDt2()    # linear acceleration [m/s²]
```

### Reading Spring State

```python
spring.GetLength()    # current length [m]
spring.GetVelocity()  # extension rate [m/s]
spring.GetForce()     # current force [N]
```

### Data Logging (for matplotlib)

Initialize lists **before** the loop, append **inside**:

```python
dt = float       # time step [s]
duration = float # simulation duration [s]
array_time  = []
array_angle = []
array_pos   = []
array_speed = []

while vis.Run():
    array_time.append(sys.GetChTime())
    array_angle.append(motor.GetMotorAngle())
    array_pos.append(piston.GetPos().x)
    array_speed.append(piston.GetPosDt().x)

    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    sys.DoStepDynamics(dt)

    if sys.GetChTime() > duration:
        vis.GetDevice().closeDevice()
```

### Post-Simulation Matplotlib Plots

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(array_angle, array_pos)
ax1.set(ylabel='position [m]')
ax1.grid()

ax2.plot(array_angle, array_speed, 'r--')
ax2.set(ylabel='speed [m/s]', xlabel='angle [rad]')
ax2.grid()

# Format x-axis in multiples of π
plt.xticks(np.linspace(0, 2 * np.pi, 5),
           ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.show()
```

### CSV Output

```python
import csv

dt = float  # time step [s]
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'pos_x', 'vel_x'])  # header

    while vis.Run():
        t = sys.GetChTime()
        writer.writerow([t, body.GetPos().x, body.GetPosDt().x])

        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        sys.DoStepDynamics(dt)
```

### Periodic Console Output

```python
dt = float  # time step [s]
frame = 0
while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    sys.DoStepDynamics(dt)

    if frame % 50 == 0:
        print(f"t={sys.GetChTime():.4f}  L={spring.GetLength():.4f}  F={spring.GetForce():.4f}")
    frame += 1
```

### Render-only at fixed intervals (SMC pattern)

```python
dt = float   # physics time step [s]
fps = int    # render frames per second
out_step = 1.0 / fps
out_time = 0.0

while vis.Run():
    sys.DoStepDynamics(dt)
    if sys.GetChTime() >= out_time:
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        out_time += out_step
```

