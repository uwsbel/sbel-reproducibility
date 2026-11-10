# STEP: code and data for the paper

This repository contains the code and data behind the tables of the paper

> S. Wang and D. Negrut, *Kinematic Tracking of Multibody Systems via a Decoupled Sequential Two-Stage Estimation Procedure*, submitted to Multibody System Dynamics.

STEP (Sequential Two-stage Estimation Procedure) estimates the joint geometry relative to the IMUs (Stage I) and then reconstructs the motion and sensor biases with an extended Kalman filter (Stage II), one body at a time. The repository also contains JT, our implementation of a filter based on JointTracker, and the JT variants of the paper.

## Install

Use Python 3.9 (tested with 3.9.13, 64 bit, on Windows):

```text
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

## Reproduce the tables

Each folder reproduces one part of the paper. Every run writes to `runs/<tag>/` inside its folder and never overwrites an existing run; use a new tag each time. `check_replay.py` compares a run with the paper: each table value is recomputed and rounded to the printed number of decimals and compared with the printed value. `expected/` holds the saved outputs of the runs behind the paper (JSON and CSV; their state histories are not stored here, to keep the repository small).

| Folder | Paper | Command (run inside the folder) | Time |
|---|---|---|---|
| `section3_spatial` | Section 3.1, Table 2 | `set STEP_CASE_CONFIG=spatial_eiv.json` then `python run_case.py spatial --tag t1` | about 8 min (most of it generates the measurements) |
| `section3_slider` | Section 3.2, Table 3 | `set STEP_SLIDER_CONFIG=slider_eiv_p.json` then `python run_slider.py --tag t1` | under 1 min |
| `section3_experiment` | Section 3.3, Table 4 | `python run_experiment.py --tag t1` | under 1 min |
| `section4_*` | Section 4, Tables 5-12 | first `python prepare_data.py` in the repository root, then see `section4_README.md` | about 5 min, then see there |

On Linux/macOS, write `STEP_CASE_CONFIG=spatial_eiv.json python run_case.py spatial --tag t1` instead of `set ...`.

On Windows without long-path support, keep the repository in a short folder (for example `D:\STEPv2-repro`). Numba's cache files otherwise exceed the 260-character path limit; the Section 4 scripts stop with a message if a folder path is too long.

Then, from the repository root:

```text
python check_replay.py section3_spatial t1
python check_replay.py section3_slider t1
python check_replay.py section3_experiment t1
```

Each prints the table values next to the paper values and ends with `PASS` or `FAIL`.

## Data

- **Spatial double pendulum (Section 3.1):** regenerated on every run from the analytical model with fixed seeds. The record used in the paper is also in `section3_spatial/expected/data/`; a run checks that the regenerated measurements have the same SHA-256 hash.
- **Slider-crank (Section 3.2):** regenerated on every run with fixed seeds.
- **Experimental double pendulum (Section 3.3):** `section3_experiment/data/`.
  - `experimental_measurements.npz`: accelerometer and gyroscope records of the two WitMotion WT9011DCL IMUs, with the time axis at the measured rate of 198.6 Hz (`TIME_AXIS_CHANGE.json`).
  - `experimental_optical_angles.csv`: link angles tracked from ArUco markers at 240 frames per second.
  - `section3_experiment/configs/experimental_reference.json`: frame rate, time offset and measured dimensions used for the optical reference.
- **Three-link and longer chains (Section 4):** the bias- and noise-free three-link record is stored in `section4_three_link/step/data/clean`. `prepare_data.py` generates the other Section 4 records: the noisy three-link records and the chain records for N = 3, 5 and 10. On the machine of the paper runs, all of them were byte-identical to the records used in the paper; the chain records are integrated numerically, so on other machines they may differ in the last bits. See `section4_README.md`.

## Notes

- Each simulated case uses fixed random draws. A replay checks that the code reproduces the paper; it does not test other draws.
- Some scripts and files use the internal names `JT-BHS` (N-body scripts) and `A' + OC` (`Aprime_oc`, three-link scripts) for the variant called `JT-BHSF` in the paper. They are the same filter.
- Each folder has a `SOURCES.json` that lists where each file came from in the authors' research workspace (`<workspace>` stands for its root). After copying, comments, docstrings and descriptive text were edited, and in the Section 4 folders code paths and saved outputs of configurations that are not used in the paper were removed; these edits do not change the computations behind the paper (see the `CHANGES.md` files of the Section 4 folders). To keep the repository small, the records that `prepare_data.py` generates and the state histories of the paper runs are not stored; `SOURCES.json` lists only the files that are stored. `section3_experiment/evaluate_experiment.py` records in `PROTOCOL.json` the camera frame rate of `configs/experimental_reference.json`, the rate that its calculation uses.

## License

Code: BSD 3-Clause License (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA.md`). See also `NOTICE.md`.

## Citation

If you use this code or data, please cite the paper above. [DOI to be added]
