# Dynamic parameter identification with calibrated geometry: data and scripts

This repository holds the data, the simulation and estimation code, and the scripts behind the tables and figures of the paper

> S. Wang and D. Negrut, *Dynamic Parameter Identification of Multibody Models with Calibrated Geometry*, submitted to Multibody System Dynamics.

The paper identifies the dynamic parameters of three simulated systems whose geometry was calibrated first and then fixed: a reduced PUMA-560 manipulator, an HMMWV, and a Polaris RZR. Each case is identified in two settings. In the different-geometry setting the data come from the true geometry and the model uses the calibrated geometry. In the same-geometry setting both use the calibrated geometry.

## What is here

| Folder | Content |
|---|---|
| `puma/`, `hmmwv/`, `polaris/` | one folder per case; each has its own `README.md` |
| `<case>/settings/` | parameter bounds, prior points, motions, noise levels, optimizer settings, seeds |
| `<case>/geometry/` | true and calibrated geometry, the selection among the calibrations, geometry errors |
| `<case>/teacher/` | teacher parameters |
| `<case>/results/` | endpoints of all runs with their costs, test measures for every test motion and output group, truth audit, reference costs, observability-map tables, optimizer histories |
| `<case>/signals/` | test records and predictions of the selected models |
| `<case>/code/` | simulation and estimation code of the case: model, records, costs, differential evolution with the stopping rule and the selection by validation cost, the observability map, and for the PUMA case the Levenberg-Marquardt step; see `<case>/code/README.md` |
| `<case>/tables.py`, `signals.py`, `figures.py`, `cost.py` | scripts that recompute the tables, the test costs and the figures |
| `check_tables.py` | compares the recomputed values with the printed values of the paper |
| `paper_values.json` | the printed values of the tables and of the conclusions |
| `common/paper_values.py` | reads the printed values from the LaTeX sources of the paper |

The code in `<case>/code/` shows how the method is implemented. It needs PyChrono (see `ENVIRONMENT.md`). Set the local paths in `<case>/code/config.py` before a run. The checks below need no simulator.

## Install

Use Python 3.9 or later with numpy and matplotlib:

```text
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

## Check the tables and figures

From the repository root:

```text
python check_tables.py
python check_tables.py --signals
python make_figures.py
```

- `check_tables.py` recomputes every number of the 36 tables and the numbers of the conclusions from the files in `results/`, `settings/`, `geometry/` and `teacher/`. A recomputed number matches a printed number with d decimals if they differ by at most half a unit of the last printed digit. The two tables of the observability map print three significant digits; there the unit is that of the third significant digit. The script prints the numbers that differ and ends with `PASS` or `FAIL`.
- With `--signals` it also recomputes, from the test records and predictions in `signals/`, the test cost, the cost of each output group, the measure r of each test motion and output group, the noise-only cost, the reference cost and the share removed, for the selected models of the main teacher. These must agree with the stored values to a relative difference of 1e-9.
- `make_figures.py` redraws the six result figures into `figures_out/`.

All three steps take a few minutes. They run no simulation.

## Simulation code

Each `<case>/code/` folder reads its inputs from the case folder (`settings/`, `geometry/`, `teacher/`). It has two entry scripts:

- `run_replay.py` regenerates the test records of the main teacher and simulates the two selected endpoints. It compares the records, the predictions and the costs J, V and T with the stored values.
- `run_smoke.py` runs two generations of the optimizer and one block of the observability map.

On the build described in `ENVIRONMENT.md`, the replay reproduces the stored records, predictions and costs: the arrays and V and T are bit for bit equal, and J agrees to a relative difference below 1e-15. A replay took 15 to 30 minutes per case on that machine. With another Chrono version the code may need small changes to the interface, and the numbers will differ.

## Names

- **Settings:** `different_geometry` and `same_geometry`.
- **Teachers:** `teacher0` is the main teacher of a case. The vehicles have three further teachers, `teacher1` to `teacher3`.
- **Runs:** `seed1` to `seed4`. `settings/seeds.json` gives the integer seeds of the optimizer.
- **Motions and parameters:** as in the tables of the paper.
- **Costs:** training cost J, validation cost V, test cost T. T0 is the noise-only test cost, Tt the test cost of the reference model (teacher parameters with the calibrated geometry), Ts the test cost of the selected model.

## What is stored and what is not

- Stored: every number that the tables need, and the test records and predictions that the test costs of the selected models need.
- Not stored: the training and validation records, the records of the further teachers, the predictions of the runs that were not selected, and the working files of the optimizer. The code in `<case>/code/` regenerates the records. Costs that need a simulation, such as the training and validation costs of the endpoints and the values of the observability map, are stored as results and are not recomputed by the check scripts.
- Not included: the code of the geometry calibration. The calibrated geometries are given as data in `<case>/geometry/`.

## License

Code and scripts: BSD 3-Clause License (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA.md`). See also `NOTICE.md`.

## Citation

If you use these data or scripts, please cite the paper above.
