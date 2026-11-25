# Polaris case

This folder holds the settings, the stored results and a part of the signals of the Polaris RZR case.
The Python files recompute the numbers of the Polaris tables and redraw the two Polaris result figures.
These scripts need no simulator. Python with numpy is enough; the figures also need matplotlib. The simulation and estimation code is in `code/` (see `code/README.md`).

## Use

```
python ../check_tables.py --case polaris --signals
python signals.py
python figures.py
```

- `tables.py`: `build_tables(case_dir)` returns the numbers of every Polaris table. `build_conclusions(case_dir)` returns the Polaris part of the numbers of the conclusions.
- `signals.py`: `check_signals(case_dir)` scores the stored predictions against the stored test records and compares the costs and r values with `results/`.
- `figures.py`: `make_figures(case_dir, out_dir)` writes `Polaris_test_predictions.pdf` and `Polaris_parameter_recovery.pdf`.
- `cost.py`: per-sample loss, group cost, cost of a motion, r, and the band of r. The group names and the noise standard deviations are written out there.

## Names

- Parameters: `mu`, `M`, `COMx`, `COMz`, `Ixx`, `Iyy`, `Izz`. Units: -, kg, m, m, kg m^2, kg m^2, kg m^2. This order is used in every list of seven values.
- `z`: normalized parameter. value = center + half_range * z, with z in [-1, 1] (`settings/bounds.json`).
- Geometry settings: `different_geometry` (the teacher uses the true geometry, the model the calibrated geometry) and `same_geometry` (both use the calibrated geometry).
- Teachers: `teacher0` is the main teacher (draw 0). `teacher1` to `teacher3` are the further teachers (draws 1 to 3).
- Runs: `seed1` to `seed4`. The integer seeds are in `settings/seeds.json` only.
- Output groups: `p` position, `q` attitude (unit quaternion w, x, y, z), `v` velocity, `speed` forward speed, `wheel_omega` wheel angular speed, `delta` steering proxy, `gyro` angular velocity, `upright_rel` upright position relative to the chassis.
- Costs: training cost J, validation cost V, test cost T. T0 is the cost of the noise alone. The reference model has the teacher parameters and the calibrated geometry; its test cost is Tt. Ts is the test cost of the selected endpoint.
- Stop events: `i` validation cost at or below the near-noise threshold, `ii` no improvement of more than 0.5 % in 8 generations, `iii` generation cap.
- r: RMSE of an output group divided by its noise level. The band is 1 plus or minus three standard deviations of r for a model without error.
- Test motions: six families with three variants each, for example `lane_change_brake_1`. Library motions carry the labels of the paper, for example `P-R02`.

## settings/

- `bounds.json`: `parameters` is a list with `name`, `unit`, `lower`, `upper`, `center`, `half_range` and `reference_value` (value at the reference point) of each parameter.
- `prior_points.json`: the three development points and the two confirmation points of the prior map, with `z` and `values`.
- `motions.json`: `library` (11 motions with `name`, `category`, `description`, `samples`, `terrain`, `training`, `input_file`), `training` (the five training motions), `confirmation` (one separate variant of each training motion), `validation` (18 motions) and `test` (18 motions, with `family`, `family_index` and `variant`).
- `inputs/*.npz`: one file per motion. Array `input`: one row per 0.002 s; columns throttle, steering, braking.
- `noise.json`: output groups with `components`, `noise_sd` and `unit`; the scoring rule (skip 100 samples, then every fourth sample); the rule for the band of r.
- `optimizer.json`: settings of the differential evolution, the validation checks, the stop rule and the rules for the endpoint and the selected model.
- `seeds.json`: integer seeds of the four runs, the rule for the seeds of the later calls of a run, the noise seeds, the seeds of the teacher draws 1 to 3, and the seed of the noise-only simulation.

## geometry/

- `true.json`: the true geometry is that of the public Chrono Polaris RZR model. The file gives the names and sha256 of the Chrono files. The files are not copied.
- `calibrated.json`: calibrated geometry of the model. `front` is a complete double-wishbone description; `rear` gives the five calibrated points of the three-link rear suspension. Metres, suspension frame, left side.
- `calibration_selection.json`: the nine front calibrations with their road (`height_max_m`, `periods_x`, `periods_y`), the quantities of the selection rule, the three mechanical checks, the order after the screen, and the selected calibration.
- `geometry_error.csv`: one row per hardpoint and side. `dx_mm`, `dy_mm`, `dz_mm`: calibrated minus true position, each geometry in its own settled reference configuration, chassis frame. `group` and `source` are those of the geometry-error table. Outward is +dy on the left side and -dy on the right side.

## teacher/

- `teacher0.json` to `teacher3.json`: `values` (seven parameter values) and `z` of each teacher.

## results/

Main teacher, one folder per geometry setting (`different_geometry/`, `same_geometry/`):

- `endpoints.json`: one record per run: `seed`, `selected`, `stop_generation`, `stop_event`, `endpoint_generation`, `training_cost`, `validation_cost`, `test_cost`, `parameters`, `z`, `all_test_motions_valid`, and the stored summaries of r (`test_r_min`, `test_r_max`, `test_r_median`, `test_r_outside_band`).
- `test_measures.csv`: one row per model, test motion and output group. `model` is `seed1` to `seed4` or `reference`. `cost` and `r` are against the noisy test record; `cost_noise_free` and `r_noise_free` are against the noise-free teacher output. In the same-geometry setting the reference model is the teacher itself.
- `group_costs.json`: stored test cost of each output group and largest r of each group, per run.
- `truth_audit.json`: stored comparison of each endpoint with the teacher: `estimate`, `teacher`, `relative_error_percent`, `signed_error_half_ranges`, `mean_relative_error_percent`, `largest_error_half_ranges`.
- `reference_levels.json`: `noise_only_cost`, `reference_cost`, `selected_cost`, each with its group costs; `share_removed`; `cost_without_upright_group`.

Further files:

- `teachers/endpoints.json`, `teachers/reference_levels.json`, `teachers/truth_audit.json`, `teachers/test_measures.csv`: the same content for teachers 1 to 3 in both settings. The CSV has the extra columns `teacher` and `setting`.
- `noise_only_cost.json`: `simulated` gives mean and standard deviation of the cost of the noise alone over 2000 realizations, for the training, validation and test sets. `of_the_data` gives the cost of the noise in the records of each teacher and setting. The Delta V column of the seed table is (validation cost - `of_the_data` validation) / `simulated` validation `sd`.
- `map/schur_snr.json`: Schur SNR of the selected training set. `schur_snr[i][j]` belongs to prior point i and parameter j. `marginal_snr` and `largest_correlation` have the same layout.
- `map/all_sets.csv`: all 20 sets with one motion per category. One row per set, development point and parameter: `rank`, `motions`, `prior_point`, `parameter`, `covered` (1 or 0), `schur_snr`.
- `further/axle_split.csv`: test costs of models with the teacher parameters and a mixed geometry. `front_geometry`, `rear_geometry` and `settling_pose` are `true` or `calibrated`; one row per motion and output group with `cost` and `r`. Different-geometry test data of the main teacher.
- `further/continued_search.json`: the selected same-geometry run continued with stop event i switched off. `reported` and `continued` give the two endpoints with their costs, group costs, parameters and errors; `checks` lists every checked generation.
- `further/parameter_exchange.json`: test cost of the reference model for teacher 0, teacher 1 and four cases with the friction or the pitch inertia exchanged between them, with the mean speed, the largest lateral acceleration and the mean number of spinning wheels of each case.
- `further/friction_above_tire_limit.json`: the selected different-geometry endpoint of teacher 3 has a friction estimate above the limit of the tire model. The file gives its costs with the estimate and with friction set to the limit of 1.0.
- `history/de_history_<setting>_seed<i>.csv` (main teacher) and `history/de_history_teacher<k>_<setting>_seed<i>.csv`: one row per generation up to the stop: `generation`, `best_training_cost`, and `validation_cost` at the checked generations (generation 24 and then every fourth).

## signals/

Main teacher only. Every file holds the eight output groups at the 275 scored samples of one test motion (samples 100, 104, ..., 1196 of 1200; 0.002 s per sample), as float64 arrays. Shapes: `p` (275, 3), `q` (275, 4), `v` (275, 3), `speed` (275), `wheel_omega` (275, 4), `delta` (275), `gyro` (275, 3), `upright_rel` (275, 4, 3). SI units; angles in rad.

- `test_noisy/<setting>/<motion>.npz`: noisy test records.
- `test_noise_free/<setting>/<motion>.npz`: noise-free teacher output.
- `predictions_selected/<setting>/<motion>.npz`: predictions of the selected endpoint.
- `predictions_reference/different_geometry/<motion>.npz`: predictions of the reference model. In the same-geometry setting the reference prediction equals the noise-free teacher output.
- `figure_curves/<motion>.npz`: full-rate curves (1200 samples) of the four plotted channels for the first variant of each test family, different-geometry setting. Arrays `<channel>_teacher` (noise-free teacher output) and `<channel>_model` (selected endpoint), with the channels `position_y_m`, `wheel_speed_front_left_rad_s`, `upright_y_front_left_m`, `yaw_rate_rad_s`.

## What the package does not hold

- The simulation code and the Chrono vehicle files are in `code/`. The training, validation and test costs of the endpoints and the map values are stored results.
- No validation records, no training records, and no signals of teachers 1 to 3.
- No populations and no logs of single evaluations of the differential evolution.
