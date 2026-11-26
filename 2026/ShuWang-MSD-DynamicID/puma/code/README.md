# PUMA-560 case: simulation and estimation code

This folder holds the code that simulates the PUMA-560 case and estimates its 17 parameters.
It reads its inputs from the case folder (`../settings`, `../geometry`, `../teacher`) and
uses `../cost.py` for the loss and the costs. It shows how the method is implemented.
With another Chrono build the numbers can differ in the last digits or more.

## Files

- `config.py`: the machine settings. All paths are set here.
- `model.py`: the forward model. It builds the reduced PUMA-560 arm in Chrono from a geometry
  file and a parameter vector, holds the home pose for 2.5 s under the joint PD controller,
  runs one motion and returns the outputs at 100 Hz (`q`, `qd`, `imu_acc`, `imu_gyr`, `tau`,
  `time`). It also holds the parameter box, the coordinates z and the feasibility test.
- `motions.py`: the joint reference trajectories of the motions in `../settings/motions.json`.
- `records.py`: simulates the teacher (`../teacher/teacher0.json`) on the training, validation
  and test motions and adds the measurement noise of `../settings/noise.json` with the seeds
  of `../settings/seeds.json`. The teacher has the true geometry in the different-geometry
  setting and the calibrated geometry in the same-geometry setting.
- `objective.py`: training cost J, validation cost V and test cost T. The model always has
  the calibrated geometry. Infeasible points get the cost 1e9.
- `de.py`: differential evolution with the settings of `../settings/optimizer.json`, the
  validation checks, the stop events (i), (ii), (iii) and the endpoint with the lowest V.
  The run is made of parts that end at the checks; part k uses the seed + k.
- `lm.py`: Levenberg-Marquardt refinement of a DE endpoint with a central-difference Jacobian,
  its stop events, the endpoint of the run, and the selected endpoint over the four runs.
- `map.py`: the prior observability map (marginal and Schur SNR) of a motion set at one prior point.
- `run_replay.py`: regenerates the test records of the teacher and simulates the two selected
  endpoints; compares records, predictions and the costs J, V and T with the stored files.
- `run_smoke.py`: a short run of DE (2 generations), LM (1 iteration) and the map (one prior
  point, one motion).

New settings file used only by this code: `../settings/de_start_points.json` (the z of the
five development points as given to the optimizer).

## Setup

Python 3.9 with numpy, scipy and pychrono. Edit `config.py`:

- `PYCHRONO_DIR`: the folder that holds the `pychrono` package, if it is not on the Python path.
- `OUTPUT_DIR`: where records, run logs and reports are written (default `code/output`).

The PUMA model uses no Chrono data files.

## Use

Run from this folder. Times were measured on a 6-core laptop processor with other programs running.

    python records.py different_geometry          # 40 teacher records, about 3 minutes
    python de.py different_geometry 1             # one DE run of the paper: about 1 to 4 hours
    python lm.py different_geometry 1             # its LM refinement: about 2 to 5 hours
    python lm.py different_geometry select        # selected endpoint after the four runs
    python map.py development_1 CP03 GR03 J1R02 TZR03
    python run_replay.py                          # about 30 minutes
    python run_smoke.py                           # about 60 minutes

One motion simulation takes 4 to 12 s, depending on the load of the machine; J needs 4 motions, V and T need 18 each.
One LM iteration needs at least 37 evaluations of J. The code runs in one process.

## Chrono interface

The code uses the Python interface of Chrono commit f022f9b (see `../../ENVIRONMENT.md`):

- bodies and joints: `ChSystemNSC`, `ChBodyEasyBox`, `ChBodyAuxRef` (`SetFrame_REF_to_abs`,
  `SetFrame_COG_to_REF`), `ChLinkMotorRotationTorque` with a Python subclass of `ChFunction`
  (`Get_y`) for the motor torque, `ChVectorD`, `ChQuaternionD`, `ChFrameD`, `ChFunction_Const`;
- integration: `ChTimestepperHHT` (alpha -0.2, acceleration form, fixed 2-ms step), the sparse
  QR solver, `DoStepDynamics`;
- the update of accelerations and reactions when the held command changes: `StateSetup`,
  `StateSolveA`, `StateSolveCorrection`, `LoadResidual_F`, `StateGatherAcceleration`,
  `StateGatherReactions`, `StateScatterAcceleration`, `StateScatterReactions`,
  `GetOffset_x`, `GetOffset_w`, `ChVectorDynamicD`.

Later Chrono versions rename several of these calls (for example `ChVectorD` to `ChVector3d`).
