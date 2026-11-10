# Section 4: three-link benchmark, longer chains and computational cost

Section 4 of the paper is reproduced by three folders:

| Folder | Paper | Rerun for the release check? |
|---|---|---|
| `section4_three_link` | Tables 5, 6, 7, 9 and 10 (single draw) | yes |
| `section4_chains` | Table 8 (20 draws), Table 11 (N = 3, 5, 10), the Section 4.2.4 sentence on the second motion | no: packaged with scripts and the saved outputs of the paper runs; one draw per producer was rerun |
| `section4_timing` | Table 12 and the Stage I times of Section 4.5 | yes (machine-dependent) |

Before any Section 4 replay, run `python prepare_data.py` once in the repository root (about 5 min). Only the bias- and
noise-free three-link record is stored in this repository (`section4_three_link/step/data/clean`); `prepare_data.py`
copies it to the other two folders and generates the other input records (see "Data" at the end of this file).

Each folder is self-contained, except that the spatial double-pendulum rows of Table 9 use `section3_spatial`. Inside
each folder:

- `step/`: STEP (Stage I `src/calibration.py`, Stage II `src/core.py`, compiled Stage II `src/compiled_step.py`), the JT core
  `src/jt_checked.py`, configs, and the three-link records.
- `review/`, `nbody/`, `parity/`: the producer scripts of the paper, unchanged except for the files listed in
  `CHANGES.md` (import paths changed, checks against runs that are not part of this repository removed, code paths and
  runs of configurations that are not used in the paper removed, comments and descriptive text edited; the computations
  behind the paper are unchanged).
- `expected/`: the saved outputs of the runs behind the paper (JSON and CSV; no state histories).
- `make_table*.py`: recompute the table from the outputs and compare every cell with `../paper_values_ch4.json`.
- `SOURCES.json`: source path (`<workspace>` = root of the authors' workspace) and SHA-256 of every file as copied.

Some scripts and result files use the internal names `JT-BHS` (N-body scripts) and `A' + OC` (`Aprime_oc`, three-link
scripts) for the variant called `JT-BHSF` in the paper, `JT-bias`
for JT-B, `JT-bias-OC` for JT-BH, `W1` for JT, `W1-RW` for JT-W, `W1-OC` for JT-H and `mid02` for JT-N. Results quoted
only in the text: `P0m` (JT with the matched lever prior, Section 4.2.2), `JT-2p` (two-pass JT, Section 4.2.2),
`R005` (JT-BHSF with constraint-residual SDs of 0.005 m and 0.005 m/s, Section 4.2.3) and `JT-BS-OC` (JT-BS with the
heading-consistent linearization, Section 4.4).

## What "match" means

A table script takes `--source replay` (default: the outputs of a replay in the folder) or `--source expected` (the saved
outputs of the paper runs). It prints the table in the layout of the paper, then compares every paper cell:

- **Accuracy tables (5-11, Section 4.2.4):** the recomputed value, rounded to the printed number of decimals, must equal
  the printed value (the rule of `check_replay.py`). For Table 5, the JT row is a range over the nine perturbations c:
  every c must round to a single printed value; for a printed range `a--b`, the smallest c rounds to `a` and the largest
  to `b`. For Section 4.2.4, the statements "the intervals include one" and "exclude one" are checked as true/false.
- **Timing (Table 12, Stage I times):** times depend on the CPU, the BLAS library and the machine load. Absolute times
  (ms per sample, Stage I and parents' Stage II seconds) are printed but not judged. The ratios to STEP Stage II and the
  total-time ratios match if they are within 20 % of the paper value, and the exponents if within 0.15 of it. On the
  machine of the paper, repeated timing runs differed by up to 13 % in these ratios. A deviation larger than the tolerance
  on another machine does not by itself mean that the code differs; compare the ordering of the filters and the growth
  with N.

Each script writes `tables/<label>_<source>.json` and exits with status 0 only if every judged cell matches. The saved
outputs reproduce every paper cell: `python make_table5.py --source expected` and so on all end with `PASS`.

## section4_three_link (Tables 5, 6, 7, 9, 10)

```text
cd section4_three_link
python run_section4_three_link.py --spatial-tag t9
python make_table5.py
python make_table6.py
python make_table7.py
python make_table9.py --spatial-tag t9
python make_table10.py
```

`run_section4_three_link.py` runs the producers in the order of the paper runs (logs in `logs/`):

| Step | Script | Produces | Used in | Time (4 processes) |
|---|---|---|---|---|
| 1 | `review/c2_step10.py` | STEP, Stage I on [0, 10] s, both scenarios: `review/out_d/C2_STEP10.json` | Tables 5, 6, 7, 9, 10 | about 2.5 min |
| 2 | `review/c2bc_sens_attrib.py` | Table 9 three-link rows, Table 10: `review/out_d/C2bc_SENS_ATTRIB.json` | Tables 9, 10 | about 5 min |
| 3 | `review/run_Aprime_oc_stepR.py` | JT-BHSF: `review/out_e/APRIME_OC_STEPR.json` | Table 7 | about 3 min |
| 4 | `review/c1f_jt_variants_stepR.py` | JT, JT-W, JT-H, JT-B, JT-BH and JT with the matched lever prior, nine c: `review/out_f/C1F_JT_VARIANTS_STEPR.json` | Tables 5, 6, 7 and the text of Section 4.2.2 | about 7 min |
| 5 | `review/c1g_jt_prior_mid.py` | JT-N (lever SD 0.2 m): `review/out_g/C1G_JT_PRIOR_MID.json` | Table 7 | under 1 min |
| 6 | `nbody/run_jtbs_single.py` | JT-BS, N = 3: `nbody/out/single/N03_*.json` | Table 7 | about 2 min |
| 7 | `run_table9_spatial.py --tag t9` | spatial double pendulum, 7 rows + 1 diagnostic: `../section3_spatial/runs/t9_*` | Table 9 | about 4 min |

The whole replay took 24 minutes with 4 processes on the machine of the paper. Step 4 writes about 1.4 GB of state
histories to `review/out_f/jt/`; 2 GB of free disk space is enough. The producers write to fixed folders,
so the runner refuses to start if their outputs exist; use a fresh copy of the folder for a second replay, or
`--from-step k` to continue after a failure.

Inputs: `step/data/{clean,noise_only,noise_bias}/` (measurements 4.2 MB and reference 9.6-9.7 MB each; the
reference holds the ground truth used for the JT initial states, the Table 10 replacements and the scoring). `clean` is
stored; `noise_only` and `noise_bias` are written by `../prepare_data.py`: `clean` plus the noise and bias drawn by
`step/generate_measurement_errors.py` with the body seeds 20261009-20261011 (`step/configs/measurement_errors.json`),
the same arrays as in the paper runs (`review/c3_mc.py` in `section4_chains` checks this bit for bit). The clean record
was generated by the sympy model `generate_three_checked.py` of the authors' workspace (not included); the N-body
generator `section4_chains/nbody/nbody_sim.py` reproduces it only to about 5e-6, so the record is stored.

Table 9, spatial double-pendulum rows: `run_table9_spatial.py` runs `../section3_spatial/run_case.py` once per row, with the
Stage I configuration of the row (`STEP_CASE_CONFIG`) on the record of the paper (`STEP_COPY_DATA_FROM =
../section3_spatial/expected/data`). The paper run of the "yes, 0" row used a `calibration.py` without the
noise-compensation code path. With the package code and `spatial_eiv_sd0.json` the row is reproduced exactly: e_l
0.72 %, e_p 0.0155 m, e_R 0.23 deg, with bit-identical state histories and levers. The same holds for
`spatial_rule.json` (the diagnostic row `yes0rule`).

Check without rerunning: `python make_table5.py --source expected` (and 6, 7, 9, 10).

Partial check of the package code (copy of this folder): STEP on the biased record (Stage I on the first 10 s,
Stage II), and JT and JT-B at c = 0, reproduced the saved state RMSEs and lever errors of the paper runs exactly
(largest difference 0).

## section4_chains (Table 8, Table 11, Section 4.2.4): packaged, not rerun

```text
cd section4_chains
python run_section4_chains.py table8      (about 1.5 h with 4 processes)
python run_section4_chains.py table11     (about 5 h with 4 processes)
python run_section4_chains.py s424        (about 45 min; part of table11)
python make_table8.py
python make_table11.py
python make_s424.py
```

`python run_section4_chains.py all --dry-run` lists the steps. The producers write one JSON file per job and skip files
that exist, so an interrupted run can be restarted.

- **Table 8** (20 draws, seeds 20262000 + 10 r + b): `review/c3_mc.py` (STEP), `review/c3c_mc_aprime_oc.py` (JT-BHSF, with
  the STEP residual SDs and, for the text, with 0.005 m and 0.005 m/s; run it after `c3_mc.py`, whose levers it checks),
  `review/c3f_mc_stepR.py` (JT and JT-B, and the two-pass JT of the text), `nbody/run_mc.py 3` (JT-BS).
- **Table 11**: Motion 1 at 1 deg `parity/run_mc.py`; Motion 2 at 1 deg `parity/run_mc_c_base.py` (seeds 20263000 + 10 r + b);
  JT-B `nbody/run_mc.py 3 5 10` (Motion 1) and `parity/run_mc_d.py` (Motion 2). The 5 deg counts of the text use
  `nbody/out/mc` and `parity/out/paper/mc/t2_p5`. JT-BS with the heading-consistent linearization (text, N = 5 and 10) is
  in the 1 deg outputs of `parity/run_mc.py` and `parity/run_mc_c_base.py` (`JT-BS-OC`).
- **Section 4.2.4**: `parity/run_mc_d.py` (its N = 3, 5 deg jobs); `make_s424.py` computes the ratios and the paired
  bootstrap intervals (2000 resamples, `default_rng(20261104)`), the algorithm of the paper.

## Data

| Record | Files (MB) | Source |
|---|---|---|
| three-link, Motion 1 (N = 3) | `step/data/clean`: measurements 4.2, reference 9.6 | stored in `section4_three_link`; copied to the other two folders by `prepare_data.py` |
| three-link with noise / with noise and bias | `step/data/noise_only`, `noise_bias`: 4.2 + 9.7 each | `prepare_data.py`, from `clean` (see above); same arrays as in the paper runs |
| Motion 1, N = 5 | `nbody/data/N05/clean`: measurements 6.9, reference 16.0 | `prepare_data.py` runs `nbody/gen_data.py 5` (about 35 s); copied to `section4_timing` |
| Motion 1, N = 10 | `nbody/data/N10/clean`: measurements 13.8, reference 31.8 | `nbody/gen_data.py 10`; copied to `section4_timing` |
| Motion 2, N = 3, 5, 10 | `parity/data_c/t2/N{03,05,10}/clean`: 4.2 + 9.6, 6.9 + 16.0, 13.8 + 31.8 | `parity/gen_data_c.py` |

The generators integrate the equations of motion with `solve_ivp`. On the machine of the paper runs, every record
written by `prepare_data.py` was byte-identical to the record used in the paper runs (checked 2026-11-10). On another
machine or with other library versions the chain records may differ in the last bits, and table values computed from
them may then differ in the last digit. (`gen_data_c.py` also checks its Motion 2 code path by integrating Motion 1 for
N = 5 and comparing with `nbody/data/N05/clean`; the two paths agree to 2.3e-9.) The solver settings and run information of the paper records (`SOLVER.json`) are in
`section4_chains/expected/nbody/data/` and `section4_chains/expected/parity/data_c/`. Both generators refuse to
overwrite existing records. They also write `generalized.npz` (generalized coordinates, not read by any script).

Check without rerunning: `python make_table8.py --source expected` (and 11, `make_s424.py`).

## section4_timing (Table 12, Stage I times)

```text
cd section4_timing
python run_timing.py --smoke      (code check: N = 3, one repetition, about 1.5 min; writes parity/out/paper/timing/SMOKE_P.json)
python run_timing.py              (about 30 min; writes parity/out/paper/timing/TIMING_P.json, refuses to overwrite)
python make_table12.py
```

`run_timing.py` only checks the path length (below) and then runs `parity/timing_paper.py`, the unchanged script of the
paper. A smoke run of the package copy on the machine of the paper gave 0.033, 0.070, 0.083 and 0.082 ms per sample for
STEP, JT-BHSF, JT-BS and JT-B at N = 3 (paper: 0.032, 0.072, 0.081, 0.082) and the same largest difference from the
reference STEP implementation as the paper run (2.0e-12).

Run the timing on an otherwise idle machine, in one process (the script sets one BLAS thread). Each filter is timed in
ten full-record runs, alternated, after a warm-up run, so Numba compilation is excluded. Stage I (not compiled) is timed
in three runs. `timing_paper.py` counts running Python processes with the Windows command `tasklist`, so as shipped it
runs on Windows only. On another system, the call `python_count()` must be replaced, for example by a function that
returns 0; this does not affect the timing.

## Windows path length

`nbody/` and `parity/` scripts keep the Numba cache in `<folder>/.numba_cache/`. The cache file names are long (for
example `src_<40 hex digits>/jt_checked.quat_to_rotmat_numba-42.py39.1.nbc.tmp.<16 hex digits>`), so the full paths are
about 135 characters longer than the section folder. Without long-path support, Windows rejects paths of 260 characters or
more, and Numba then stops with `FileNotFoundError` while writing its cache. Keep each `section4_*` folder at most 115
characters long, for example by cloning the repository to `C:\src\STEPv2-repro`, or enable long paths in Windows
(`LongPathsEnabled = 1`). `run_section4_three_link.py`, `run_section4_chains.py` and `run_timing.py` check this before
they start. Linux and macOS are not affected.

## Requirements and environment

The paper runs used Python 3.9.13, numpy 1.26.4, scipy 1.13.1, numba 0.60.0 and pandas 2.2.3 on Windows 11
(`requirements.txt` in the repository root). The JT core imports pandas and matplotlib at import time.
