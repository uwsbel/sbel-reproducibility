# PUMA-560 case

This folder holds the stored results of the PUMA-560 case and the code that
recomputes the tables and the two result figures of the paper from them.
These scripts need no simulator. Python with numpy is enough; `figures.py` also needs matplotlib. The simulation and estimation code is in `code/` (see `code/README.md`).

The model is a reduced PUMA-560 with three revolute joints (Z-Y-Y) on a fixed base.
The nominal dimensions follow the PUMA-560 model of the Robotics Toolbox by P. Corke.
The teacher and the model are simulated together with a joint PD controller.
The input of each simulation is the reference trajectory of the motion.

## Names

- **Geometry settings.** `different_geometry`: the teacher uses the true geometry and the
  model uses the calibrated geometry. `same_geometry`: teacher and model use the calibrated geometry.
- **Teacher.** `teacher0` is the one teacher of this case.
- **Seeds.** The four runs of each setting are `seed 1` to `seed 4`. The integers passed to the
  optimizer are in `settings/seeds.json`. Both settings use the same seeds.
- **Endpoint.** Each run is a differential-evolution (DE) search followed by a
  Levenberg-Marquardt (LM) refinement. The endpoint of a run is its checked iterate with the
  lowest validation cost. The selected endpoint of a setting is the endpoint with the lowest
  validation cost over the four runs: seed 1 in `different_geometry`, seed 4 in `same_geometry`.
- **Motions.** Test motions: two letters for the family and the variant number 4 to 6.
  `HG` gravity-dominated, `HF` friction-dominated, `HI` inertia-rich, `HC` coupling,
  `CS` spline path, `CH` chirp. Validation motions use the same letters with the variants 1 to 3.
  The 21 library motions have their own identifiers (for example `CP03`); four of them form the training set.
- **Parameters.** `m2`, `m3` (kg); `m2c2x`, `m2c2z`, `m3c3x`, `m3c3z` (first moments, kg m);
  `I1z`, `I2y`, `I2z`, `I3y`, `I3z` (kg m^2); `b1`, `b2`, `b3` (viscous friction, N m s/rad);
  `fc1`, `fc2`, `fc3` (Coulomb friction, N m). This order is used in every list.
- **Coordinate z.** `z = (value - centre) / half_range` of the parameter interval. One unit of z is one half-range.
- **Output groups.** `joint_angle` (array `q`, rad), `joint_rate` (`qd`, rad/s),
  `specific_force` (`imu_acc`, m/s^2), `angular_rate` (`imu_gyr`, rad/s),
  `torque` (`tau`, command torque of the controller, N m).
- **Costs.** Training cost J, validation cost V and test cost T are the same objective on the
  4 training, 18 validation and 18 test motions.
- **Prior points.** `development_1` to `development_5` and `confirmation_1`, `confirmation_2`.
  `development_1` is the centre of the box.

## Python files

- `cost.py`: residual, soft-L1 loss, cost of a group, of a motion and of a set of motions,
  the test measure r, the band of r, and the RMS difference to a reference record.
  The group names and the noise standard deviations are written out in the file.
- `tables.py`: `build_tables(case_dir)` returns the numbers of the PUMA tables of the paper,
  unrounded. `build_conclusions(case_dir)` returns the share of the excess test cost removed.
- `signals.py`: `check_signals(case_dir)` recomputes the test measures of the two selected
  endpoints, the noise-only cost and the reference cost from `signals/` and compares them with `results/`.
- `figures.py`: `make_figures(case_dir, out_dir)` draws `PUMA_test_predictions.pdf` and
  `PUMA_parameter_recovery.pdf`.

## settings/

- `bounds.json`: the 17 parameter intervals (`name`, `unit`, `lower`, `upper`), the fixed
  inertial quantities, the centre-of-mass intervals and the feasibility conditions.
- `prior_points.json`: the seven prior points of the map, as values and as z.
- `motions.json`: the library (identifier, category, duration), the training set, and the
  validation and test motions (identifier, family, number of samples, time of the last
  sample). `duration_s` is the length of the reference trajectory; the records are sampled
  every 0.01 s from time 0. `reference_type` and `reference_parameters` describe the
  reference trajectory of a motion; angles in degrees, times in seconds, frequencies in
  hertz. `code/motions.py` generates the references.
- `noise.json`: the five output groups with array name, unit, number of components and
  noise standard deviation; the scoring start (0.2 s); the loss; the band of r.
- `optimizer.json`: the DE and LM settings and the stop events.
- `seeds.json`: the DE seeds of seed 1 to 4, the rule for the seed of each part of a DE run,
  and the measurement-noise seeds of the training, validation and test motions.

## geometry/

- `true.json`, `calibrated.json`: for each joint, the unit axis and the lever (m) from the
  frame origin to the joint, in the parent frame and in the child frame. The frame of a
  moving link has its origin at the IMU of the link.
- `calibration_selection.json`: the six calibrations with their check values, the four that
  pass, and the selected one (number 5).
- `geometry_error.csv`: calibrated minus true geometry. Axis error in degrees; lever error in
  mm, along the true axis, perpendicular to it, and by component. `tables.py` recomputes
  these values from the two JSON files.

## teacher/

- `teacher0.json`: the 17 teacher values, with unit and z.

## results/

`different_geometry/` and `same_geometry/` hold the same files:

- `endpoints.json`: the four runs. `de`: stop generation, stop event (i, ii, iii), and the
  generation, training cost, validation cost and z of the DE endpoint. `lm`: stop iteration,
  stop event (`cost`, `step`, `validation`) and the iteration of the endpoint. Then the
  training, validation and test cost of the endpoint, its z and its parameter values.
  `selected_seed` names the selected endpoint. For the selected endpoint, the training and
  validation costs are also given by motion.
- `test_measures.csv`: one row per seed, test motion and output group (4 x 18 x 5 rows).
  `scored_values` is the number of scored samples times components. `r` is the RMSE against
  the noisy test record divided by the noise standard deviation. `cost` is the cost of the
  group on the motion. `rmse_to_noise_free` is the RMS difference to the noise-free teacher
  output, in SI units. The cost of a motion is the mean of its five group costs. The test
  cost is the mean of the 18 motion costs.
- `group_costs.json`: cost of each output group on the test set for the selected endpoint,
  the reference model and noise only.
- `reference_levels.json`: noise-only cost, reference cost, cost of the selected endpoint and,
  in `different_geometry`, the share of the excess removed. The reference model has the
  teacher parameters and the calibrated geometry.
- `truth_audit.json`: errors of the four endpoints against the teacher values, in half-ranges
  and in percent.
- `geometry_components.json` (`different_geometry` only): test cost with the teacher
  parameters and single components of the calibration error.

Other folders:

- `map/schur_snr.json`: Schur SNR of the training motions at the seven prior points, numerical
  rank and condition number of their stack, coverage, and the first ten sets of the ranking.
- `map/all_sets.csv`: all 450 sets with one motion per category, in rank order. The columns
  `smallest_schur_snr_<parameter>` give the smallest value over the five development points.
- `history/de_history_<setting>_seed<i>.csv`: generation, best training cost, validation cost
  at the checks (empty elsewhere), and the number of feasible candidates of the generation.
  Generation 0 is the initial population.
- `history/lm_history_<setting>_seed<i>.csv`: iteration, training cost, validation cost at the
  checks, damping, step length in z and relative decrease of the training cost. Iteration 0 is
  the DE endpoint.
- `further/noise_only_cost.json`: mean and standard deviation of the cost of pure noise on the
  training, validation and test sets.

Training and validation costs, DE and LM histories, map values and the costs of
`geometry_components.json` come from simulations. They are stored values and are not
recomputed here. Test costs, r, the RMS differences, the noise-only cost and the reference
cost are recomputed from `signals/`.

## signals/

Each file is a compressed `.npz` with float64 arrays sampled at 100 Hz, named by motion:
`time` (n), `q`, `qd`, `tau` (n x 3, joints 1 to 3), `imu_acc`, `imu_gyr` (n x 3 x 3, IMU of
link 1 to 3 times the three axes of the link frame). Units are SI.

- `test_noisy/<setting>/`: test records with measurement noise. These are the data of the test cost.
- `test_noise_free/<setting>/`: the same teacher outputs without noise.
- `predictions_selected/<setting>/`: predictions of the selected endpoint of the setting.

The output of the reference model is `test_noise_free/same_geometry/`: teacher parameters
with the calibrated geometry. The figures are drawn from these files; there are no separate
curve files. For the six other endpoints, `test_measures.csv` holds the test measures.

## Use

From the parent folder:

    python check_tables.py --case puma --signals

In Python, with this folder as the working directory:

    import tables, signals, figures
    tables.build_tables()
    signals.check_signals()
    figures.make_figures(out_dir="figures_out")
