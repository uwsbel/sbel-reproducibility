# HMMWV case

This folder holds the stored results of the HMMWV example. The tables and the two result figures of the paper can be recomputed from these files. These scripts need no simulator. The simulation and estimation code is in `code/` (see `code/README.md`).

## How to use

- `python check_tables.py --case hmmwv` (from the folder above) compares the recomputed table numbers with the printed ones.
- `python check_tables.py --case hmmwv --signals` also recomputes the test costs from the stored signals.
- `python figures.py` draws the two figures into the folder `figures_out` of the working directory.
- The scripts need Python 3 with numpy. `figures.py` also needs matplotlib.

## Names

- Parameters: `mu` (tire-terrain friction, no unit), `M` (chassis mass, kg), `COMx`, `COMz` (center of mass, m), `Ixx`, `Iyy`, `Izz` (principal chassis inertias, kg m^2). All lists of seven values use this order.
- `z` is the normalized parameter: value = center + half_range * z, with z between -1 and 1 (`settings/bounds.json`).
- Geometry settings: `different_geometry` (the teacher uses the true geometry, the model the calibrated geometry) and `same_geometry` (both use the calibrated geometry).
- Teachers: `teacher0` is the main teacher (draw 0). `teacher1` to `teacher3` are the further teachers (draws 1 to 3).
- Seeds: `seed1` to `seed4`, or the numbers 1 to 4. The integers are in `settings/seeds.json`.
- Costs: J is the training cost, V the validation cost, T the test cost. T0 is the cost of the noise alone, Tt the test cost of the reference model, Ts the test cost of the selected endpoint.
- Reference model: the teacher parameters with the calibrated geometry.
- Stop events: `i` (V at most 1.1196), `ii` (two consecutive checks without a decrease of V of more than 0.5 %), `iii` (120 generations).
- Output groups: `position`, `attitude`, `velocity`, `speed`, `wheel_angular_speed`, `steering_proxy`, `angular_velocity`, `upright_relative_position`.
- Motions: the library motions have the labels of the paper (`H-D01` to `P-W04`). The validation and test motions are named `validation_<family>_<variant>` and `test_<family>_<variant>`, with variants 1 to 3.

## Scripts

- `cost.py`: the loss, the cost of one motion, the measure r and its noise band. It lists the group names, the noise standard deviations and the number of components.
- `tables.py`: `build_tables(case_dir)` returns the unrounded numbers of every HMMWV table row. `build_conclusions(case_dir)` returns the HMMWV numbers of the conclusions. The two tables shared with Polaris (`tab:vehicle_bounds`, `tab:vehicle_map_snr`) also read `../polaris/settings/bounds.json` and `../polaris/results/map/schur_snr.json`.
- `signals.py`: `check_signals(case_dir)` recomputes costs and r from `signals/` and compares them with `results/`.
- `figures.py`: `make_figures(case_dir, out_dir)` draws `HMMWV_test_predictions.pdf` and `HMMWV_parameter_recovery.pdf`.

## settings/

- `bounds.json`: `parameters` lists name, unit, lower and upper bound, center and half_range of each parameter. `tire_model_friction_limits` is the range of friction that the tire model accepts. `scale` gives the wheelbase and the center-of-mass height that set the ranges of COMx and COMz.
- `prior_points.json`: the five prior points of the map. `points` gives name, role (development or confirmation), `z` and the physical `values`.
- `motions.json`: `input_columns` and `input_interval_s` describe the input files. `training` lists the five training motions. `library` lists the 12 library motions with category, name and a flag `training`. `confirmation` lists the variants of the library motions used at the confirmation points (`variant_of`). `validation` and `test` list the 18 motions of each set with family and variant. Each entry gives `file`, `samples`, `duration_s` and `terrain`.
- `inputs/*.npz`: one file per motion. The array `control` has one row per control interval of 0.002 s and the columns throttle, steering, braking.
- `noise.json`: `groups` lists the eight output groups with the number of components and the noise standard deviation in the unit of the paper table (`standard_deviation`, `unit`) and in SI units (`standard_deviation_si`; rad for the steering proxy). `scoring` gives the skipped samples, the stride and the number of scored samples.
- `optimizer.json`: the settings of the differential evolution runs, the validation checks, the stop rule, the endpoint rule and the selection rule.
- `seeds.json`: `de_seeds` maps seed1 to seed4 to the integers. `de_chunk_rule` says how the generator is seeded in each part of a run. `noise_seeds` gives the first noise seed of each record set and the rule for the records and teachers. `test_input_seeds` are the seeds of the test input variants. `teacher_draw_seeds` are the seeds of teachers 1 to 3. `noise_only_cost_simulation_seed` is the seed of the noise realizations in `results/noise_only_cost.json`.

## geometry/

- `true.json`: the true geometry is the stock geometry of the Project Chrono HMMWV model. The Chrono files are not copied. The file gives their names, their sha256 and the hardpoint locations they contain (left side, axle reference frame, m).
- `calibrated.json`: the calibrated geometry. `front` and `rear` hold the two double-wishbone suspensions in the format of a Chrono suspension file. `hardpoints_m` repeats the hardpoint locations with the point names used in `geometry_error.csv`.
- `calibration_selection.json`: the six calibrations with their excitation, the metrics of the selection rule, the result of the quality screen, the rank by window dispersion, the mechanical checks and the selected calibration.
- `geometry_error.csv`: error of the calibrated geometry relative to the true geometry, one row per hardpoint and side. Each geometry is taken in its own settled reference configuration, in the chassis frame. Columns: `axle`, `side`, `point`, `group` and `source` (as in the paper table), `dx_mm`, `dy_mm`, `dz_mm`, and `outward_mm` (lateral error, positive outward). The values have three decimals.

## teacher/

- `teacher0.json` to `teacher3.json`: `parameters` gives name, unit, `value` and `z` of the seven teacher parameters. `draw` gives the rule of the draw and the number of rejected draws.

## results/

- `noise_only_cost.json`: `simulated` gives mean and standard deviation of the noise-only cost over 2000 noise realizations, for sets with the size of the validation, test and training sets, and the analytic values. `teachers` gives the noise-only cost of the stored training, validation and test records of each teacher and geometry setting. The column Delta V of the seed table is (V - V0) / sd, with V0 the `validation` value of teacher 0 and sd the simulated standard deviation of the validation set.

### results/different_geometry/ and results/same_geometry/ (main teacher)

- `endpoints.json`: one entry per seed in `runs`: `stop_generation`, `stop_event`, `endpoint_generation`, `training_cost`, `validation_cost`, `test_cost`, `estimate` (seven parameters), `z`, `selected`. `selected_seed` is the seed with the lowest validation cost.
- `test_measures.csv`: one row per seed, test motion and output group. `cost` and `r` compare the prediction with the noisy test record. `cost_noise_free` and `r_noise_free` compare it with the noise-free test record.
- `group_costs.json`: stored summary of each endpoint: mean cost of each group over the 18 test motions, `test_cost`, median, smallest and largest r, the number of r values outside the noise band (`outside_band`) and the largest r of each group.
- `reference_levels.json`: `noise_only` (T0) and `reference_model` (Tt) with the cost of each group. `reference_model.motions` gives cost and r of the reference model for every test motion. `selected_endpoint` repeats Ts. `share_removed` is (Tt - Ts) / (Tt - T0). `without_upright_group` gives the three costs as the mean of the seven other groups. In the same-geometry setting the reference model reproduces the noise-free records exactly (`equals_noise_free_record`), so Tt equals T0 and no share is defined.
- `truth_audit.json`: teacher values and, per seed, the estimate, the relative error in percent and the signed error in half-ranges of each parameter, with their mean and largest value.

### results/teachers/ (teachers 1 to 3, both settings)

- `endpoints.json`: the 24 runs, with the fields of the main teacher and the fields `teacher` and `setting`. Each run also has its stored `group_costs`, `r_median`, `r_min`, `r_max` and `outside_band`.
- `test_measures.csv`: as for the main teacher, with the columns `teacher` and `setting`.
- `reference_levels.json`: one entry per teacher and setting in `levels`, with the fields of the main teacher.
- `truth_audit.json`: one row per run, with the fields of the main teacher.

### results/map/

- `schur_snr.json`: Schur SNR of the five training motions. `schur_snr` has one row per prior point (three development points, then two confirmation points) and one column per parameter. `marginal_snr`, `crlb_standard_deviation_half_range`, `covered` and `reliable_column` have the same layout. `whole_library` gives the Schur SNR of all 12 library motions at the development points.
- `all_sets.csv`: the 30 sets with one motion per category, in rank order. Columns: `rank`, `selected`, `motions`, `covered_pairs`, and the Schur SNR of each parameter at each development point (`<parameter>_development_<i>`). The median of the 21 values ranks the sets.

### results/history/

- `de_history_<setting>_seed<i>.csv` (main teacher) and `de_history_teacher<k>_<setting>_seed<i>.csv`: one row per generation up to the stop. `best_training_cost` is the lowest training cost of the population. `validation_cost` is filled at the checks (generation 24 and every fourth generation after it).

### results/further/

- `axle_split.json`: test cost of each output group for the teacher parameters with the calibrated geometry in the front axle only, in the rear axle only, and in both axles; `motion_costs` gives the cost of every test motion.
- `continued_search.json`: the two same-geometry runs that stopped by event (i), continued without this event: stop, endpoint, costs and parameter errors with and without event (i), and the validation cost at every check.
- `friction_exchange.json`: reference cost of teacher 0 with a friction of 0.69 and of teacher 2 with a friction of 0.942, with the reference cost of each teacher with its own friction.
- `evaluation_counts.json`: number of evaluated candidates, of infeasible candidates and of candidates stopped early in the eight runs of the main teacher.

## signals/

All arrays are float64. The files of the first four folders hold the 275 scored samples of each record (samples 100, 104, ..., 1196 of the 1200 samples; the sample interval is 0.002 s). Each file has eight arrays named by output group: `position` (275, 3), `attitude` (275, 4; unit quaternion w, x, y, z), `velocity` (275, 3), `speed` (275), `wheel_angular_speed` (275, 4), `steering_proxy` (275; rad), `angular_velocity` (275, 3), `upright_relative_position` (275, 4, 3). Wheel 0 and upright 0 are front left; the first two are the front axle.

- `test_noisy/<setting>/<motion>.npz`: noisy test records of the main teacher.
- `test_noise_free/<setting>/<motion>.npz`: the same records without noise.
- `predictions_selected/<setting>/<motion>.npz`: predictions of the selected endpoint.
- `predictions_reference/different_geometry/<motion>.npz`: predictions of the reference model. In the same-geometry setting they equal the noise-free records and are not stored.
- `figure_curves/<motion>.npz`: the curves of the prediction figure at all 1200 samples, for the first variant of each test family in the different-geometry setting: `noise_free_<channel>` and `prediction_<channel>` for the channels `position_y` (m), `front_left_wheel_speed` (rad/s), `front_left_upright_y` (m) and `yaw_rate` (rad/s).

## Not included

The validation records, the training records, the test records of teachers 1 to 3, the populations and evaluation logs of the DE runs, and the sensitivity matrices of the map are not part of this folder. The code in `code/` regenerates the records.

## Third-party values

`geometry/true.json` and the mass, spring and damper values in `geometry/calibrated.json` come from the HMMWV model of Project Chrono (BSD 3-Clause license, Project Chrono Development Team).
