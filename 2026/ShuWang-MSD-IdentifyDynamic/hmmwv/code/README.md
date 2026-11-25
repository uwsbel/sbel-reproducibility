# HMMWV simulation and estimation code

This folder shows how the HMMWV example is computed: the vehicle model, the settle and start, the records of a teacher, the costs, the differential evolution with its stopping rule, and the prior observability map. The code reads its inputs from the case folder (`../settings`, `../geometry`, `../teacher`, `../results`, `../signals`). It writes only to `OUTPUT_DIR`.

The code is meant to be read. It runs with the Chrono build named below. With another Chrono build the numbers are not expected to agree bit for bit.

## Files

- `config.py`: all machine paths. Edit this file first.
- `model.py`: the forward model. `simulate(parameters, geometry, motion)` builds the vehicle, settles it, drives one motion and returns the eight output groups with the validity of the rollout.
- `rough_road.py`: exact height, normal and friction queries on the rough road, through the compiled helper of `terrain/`.
- `records.py`: noise-free and noisy records of a teacher for the training, validation and test motions.
- `objective.py`: the cost of one motion, the training cost J (with early rejection), the validation cost V and the test cost T. It uses `../cost.py`.
- `de.py`: differential evolution with the stopping rule; the endpoint of a run and the selected endpoint over the seeds.
- `map.py`: the prior observability map (whitened sensitivities, Schur SNR) of a motion set.
- `run_replay.py`: regenerates the records of the main teacher and recomputes J, V and T of the two selected endpoints.
- `run_smoke.py`: two DE generations and the map of one prior point and one motion.
- `chrono_data/`: the Chrono vehicle files the model reads, with the Chrono license.
- `terrain/`: C++ source and build note of the terrain query helper.

## Model

- Chrono::Vehicle `WheeledVehicle` built from JSON files: a rigid chassis with the seven parameters (mass, center of mass x and z, principal inertias), two double-wishbone suspensions, Pitman-arm steering on the front axle, rear-wheel drive, simple brakes, TMeasy tires, the HMMWV simple-map powertrain. SMC contact with the Chrono default settings.
- Suspension geometry: `true` uses the stock Chrono files (`chrono_data/hmmwv/suspension/`). `calibrated` uses `../geometry/calibrated.json` and the body inertias of `../settings/calibrated_suspension_inertia.json`.
- The differential clutch carries no torque in this model and is disabled. The solver is sparse LU.
- Terrain: a flat patch of 300 m x 300 m, or the rough road from a height map (`../settings/terrain/`). The terrain friction is the parameter mu. Young modulus, Poisson ratio, restitution and adhesion are in `../settings/simulation.json`.
- Integration step 1 ms. The inputs are held for 2 ms; one output sample is taken after each 2-ms interval.
- Settle and start: the chassis is held in x, y and yaw. The powertrain is in neutral, all inputs are zero, and the tires see a terrain with a low friction. The vehicle moves freely for 0.4 s; then the velocities of all moving bodies are multiplied by 0.99 after every step, up to 2.0 s. The hold is removed, the drive mode is set to forward, and the motion starts in the same simulation.
- A rollout is valid if all outputs are finite, the speed stays at most 35 m/s, roll and pitch at most 45 deg, the planar travel is at least 0.05 m, every joint stays within 1 mm of its constraint, and toe and camber after the settle are at most 5 deg.

## Estimation

- Parameters are searched in z, with value = center + half_range x z (`../settings/bounds.json`).
- J, V and T are means of motion costs. An infeasible vector or an invalid rollout costs 1e9.
- Early rejection: a trial stops as soon as its partial training cost exceeds the cost of the member it is compared with. The search path is unchanged.
- DE (`../settings/optimizer.json`): best1bin, 32 members (three development prior points and Latin-hypercube points), mutation 0.5 to 1, recombination 0.7, immediate updating, no polishing. Chunks of 16 and then 8 generations, seeded with seed + chunk index (`../settings/seeds.json`). The best member is checked on the validation motions at generation 24 and every fourth generation. The run stops at event (i), (ii) or (iii); the endpoint is the checked member with the lowest V. Over the four seeds the endpoint with the lowest V is selected.
- Map: for each prior point and motion, 44 rollouts at the point, at the point again, and at +/- 0.01, 0.02 and 0.04 half-ranges of each parameter give the sensitivity of a 630-dimensional output feature. The feature covariance comes from 2000 noise realizations (seeds in `../settings/map_noise_seeds.json`) with Ledoit-Wolf shrinkage. The motion set of the paper was ranked by the rule described with `../results/map/all_sets.csv`.

## How to run

Python 3.9 with numpy, scipy and pychrono. Edit `config.py`, then run from this folder:

```
python records.py --setting different_geometry          # records of the main teacher (41 rollouts)
python de.py --setting different_geometry --seed 1      # one DE run (several hours)
python map.py --points development_1 --motions P-D01    # map of one block (44 rollouts)
python run_replay.py                                    # replay of the selected endpoints (164 rollouts)
python run_smoke.py                                     # smoke run
```

One rollout takes about 4 s on one idle core; set the number of BLAS threads to 1.

## Chrono interface

The code uses the Chrono 8.0 Python interface with the single powertrain class: `veh.WheeledVehicle`, `vehicle.InitializePowertrain(veh.HMMWV_SimpleMapPowertrain(...))`, `veh.ChPowertrain.DriveMode_NEUTRAL` and `DriveMode_FORWARD`, `veh.ReadTireJSON`, `vehicle.InitializeTire`, `veh.RigidTerrain`, `veh.FlatTerrain`, `veh.ChTerrain`, `chrono.ChMaterialSurfaceSMC`, `chrono.ChVectorD`, `ChQuaternionD`, `ChFrameD`, `ChLinkMateGeneric`, `ChSolverSparseLU`, `CastToChShaftsClutch`, `CastToChTriangleMeshShape`. Later Chrono versions renamed several of these. The results of the paper were computed with a Chrono build compiled from the 8.0 source tree.

## What to edit in config.py

- `CHRONO_DATA_DIR`: a folder that holds `hmmwv/` with the Chrono vehicle files (default: `chrono_data/`).
- `TERRAIN_LIBRARY`: the compiled helper of `terrain/` (only for the rough road).
- `OUTPUT_DIR`: where records, logs and reports are written.
- `PYCHRONO_DIR`: the folder that holds the pychrono package, if it is not on the Python path.

## Third-party files

`chrono_data/hmmwv/` holds eight files of the Chrono HMMWV model (`data/vehicle/hmmwv`), unchanged, under the BSD 3-Clause license of Project Chrono (`chrono_data/LICENSE`). Meshes are not included; the model does not load them.
