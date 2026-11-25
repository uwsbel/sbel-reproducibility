# Polaris simulation and estimation code

This folder shows how the Polaris case is computed: the vehicle model, the start state, the records, the costs,
the differential evolution with its stopping rule, and the prior observability map.
The code is short and serial. It is written for reading; a full run of the paper takes several days on one computer.
The numbers of the paper were made with a local build of Project Chrono; another Chrono build can give other last digits or more.

## Files

| File | What it does |
|---|---|
| `config.py` | All machine paths: vehicle data, work folder, terrain helper. Edit this file only. |
| `model.py` | Forward model. Writes the vehicle JSON files for a geometry and seven parameters, builds the Chrono system, settles the vehicle, drives one motion and returns the eight output groups and the mechanical checks. |
| `records.py` | Noise-free and noisy training, validation and test records of a teacher (`python records.py teacher0 different_geometry`). |
| `objective.py` | Cost of one motion, training cost J, validation cost V and test cost T. Uses `../cost.py`. |
| `de.py` | One DE run: calls of SciPy `differential_evolution`, checks of V, stopping rule, endpoint; `select()` picks the selected model of a setting (`python de.py different_geometry seed1`). |
| `map.py` | Prior observability map of one or more motions at one prior point: feature, multi-radius Jacobian, Ledoit-Wolf noise covariance, whitening, marginal and Schur SNR (`python map.py development_1 P-D01`). The noise seed of each block is in `../settings/map_noise_seeds.json`. |
| `run_replay.py` | Replay check: regenerates the main teacher's records, compares the test records with `../signals/`, and recomputes J, V and T of the two selected endpoints. Writes `replay_report.json`. |
| `run_smoke.py` | Short run: DE for 2 generations and the map for one prior point and one motion. Writes `smoke_report.json`. |
| `chrono_data/Polaris/` | The Project Chrono Polaris files that the model reads (BSD 3-Clause, see the NOTICE of the package). No meshes. `Polaris_TMeasyTire_fixed.json` is `Polaris_TMeasyTire.json` with the rolling resistance written as a list and the roundness entry added, for the tire reader of this Chrono version. |
| `terrain/` | C++ source and build note of the terrain query helper for the uneven road. No binary. |

All other inputs come from the package: `../settings/`, `../geometry/`, `../teacher/`.
Two settings files serve only this code: `settings/start_poses.json` (settled poses at the reference point, start of every settle)
and `settings/front_suspension_inertia.json` (inertia of the six front suspension bodies of the calibrated geometry).

## Setup

1. Python 3.9 with numpy, scipy and pychrono. pychrono must be a Chrono 8.0 build with the old powertrain interface
   (`WheeledVehicle.InitializePowertrain`, `ChPowertrain.DriveMode_*`, `Gator_SimpleMapPowertrain`).
2. Build the terrain helper (`terrain/BUILD.md`) if you run the uneven road.
3. Edit `config.py`: `WORK_DIR` (output), `TERRAIN_LIBRARY`, and `CHRONO_DATA` if you keep the vehicle files elsewhere.
4. Run from this folder, for example `python run_replay.py` or `python run_smoke.py`.

## Model in short

- Vehicle: Chrono `WheeledVehicle` from JSON. The chassis center of mass is written into the chassis file; mass and
  principal inertias are set on the chassis body after construction. Simple brakes (2000 N m). Gator simple-map powertrain.
  TMeasy tires. The three differential clutches carry no torque and are switched off.
- Geometry: `true` = the public Chrono files; `calibrated` = `../geometry/calibrated.json` (complete front suspension,
  five rear points) and the six front body inertias.
- Contact: SMC system, friction = parameter mu, Young's modulus 2e5, Poisson ratio 0.3, restitution 0.05.
  Flat ground 300 m x 300 m, or the height-map road with the terrain helper.
- Solver: direct sparse LU. Integration step 0.001 s; inputs and outputs every 0.002 s.
- Start: each parameter vector is settled for 2 s from `start_poses.json`: chassis pinned in x, y and yaw, no inputs,
  in-plane tire forces removed, all velocities multiplied by 0.99 after each step. The motion starts from the settled poses
  with zero velocities in a new build.
- Mechanical checks (a run that breaks one gets the penalty cost): speed at most 35 m/s, roll and pitch at most 45 deg,
  planar travel at least 0.05 m, largest joint violation reported by Chrono at most 0.001, start toe and camber at most 5 deg.

## Chrono interface used

`ChSystemSMC`, `ChSolverSparseLU`, `ChMaterialSurfaceSMC`, `ChLinkMateGeneric`, `ChVectorD`, `ChQuaternionD`, `ChFrameD`,
`ChCoordsysD`, `ChMatrix33D`, `CastToChShaftsClutch`, `CastToChLink`, `CastToChTriangleMeshShape`; vehicle module:
`SetDataPath`, `GetDataFile`, `WheeledVehicle`, `Gator_SimpleMapPowertrain`, `ReadTireJSON`, `RigidTerrain`, `FlatTerrain`,
`ChTerrain`, `DriverInputs`, `ChPowertrain.DriveMode_FORWARD` and `DriveMode_NEUTRAL`, `VisualizationType_NONE`.
