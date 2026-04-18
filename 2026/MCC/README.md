# Introduction

This document links to the source files, run scripts, and plotting utilities used
to generate every figure and table in the **Results** section of the paper
*"Modified Cam-Clay vs. μ(I): A Continuum Terramechanics Comparison within
Chrono::CRM"* (Journal submission, 2026).

The numerical experiments fall into five families that map one-to-one to the
subsections in `results.tex`:

| Paper section | Experiment | Demo executable | Run script | Plot script |
| --- | --- | --- | --- | --- |
| 4.1 Cone Penetrometer | CPT on GRC-1 | `demo_FSI_ConePenetrometer` | `shell_scripts/cone_nrel.sh` | `python_scripts/penetrometer_plot_single.py` |
| 4.2 Normal Bevameter | 19 cm plate on GRC-1 | `demo_FSI_NormalBevameter` | `shell_scripts/bev_nrel.sh` | `python_scripts/plot_plate_penetration.py` |
| 4.3 MGRU3 Wheel (constant depth) | Single-wheel slope sweep | `demo_FSI_SlopedSingleWheelTest` | `shell_scripts/viper_wheel_paperRuns.sh` | `python_scripts/viper_singleWheel_plotting.py` |
| 4.3 MGRU3 Wheel (depth sensitivity) | Slope × bin-depth sweep | `demo_FSI_SlopedSingleWheelTest` | `shell_scripts/viper_wheel_paperRuns_DepthVar.sh` | `python_scripts/viper_singleWheel_DepthVar_plotting.py` |
| 4.4 Lunar Lander | MCC parameter sweep + μ(I) baseline | `demo_ROBOT_Lander_CRM` | `shell_scripts/lander_runs_sbatch.sh` | `python_scripts/plot_lander_crm_sweep.py` |
| 4.5 Efficiency (RTF) | Wheel/lander wall-clock | `demo_FSI_SlopedSingleWheelRTFBenchmark`, `demo_ROBOT_Lander_CRM_RTFBenchmark` | Run by hand (see §5) | n/a (Table 1 in paper) |

All scripts are copied from `build_new/bin/` and therefore assume a build-tree
working directory: they expect to be launched from `<build>/bin/` and read/write
relative paths such as `./DEMO_OUTPUT`, `./paper_plots`, and (for the lander
sweep) `./DEMO_OUTPUT_NREL_LANDER`.

---

# 1. Source Repository

| Item | Value |
| --- | --- |
| Repository | `git@github.com:uwsbel/chrono-wisc.git` |
| Branch | `feature/crm_models` |

```bash
git clone git@github.com:uwsbel/chrono-wisc.git
cd chrono-wisc
git checkout feature/crm_models
```

All demos, shell scripts (`shell_scripts/`), and plotting scripts
(`python_scripts/`) used in the paper are tracked on this branch. If you do not
have access to `uwsbel/chrono-wisc`, the same code base is mirrored on the
public Project Chrono repository (`github.com/projectchrono/chrono`); however,
the curated `shell_scripts/` and `python_scripts/` folders and the MCC changes
in the CRM solver are only on the branch above.

If you only want to **use** Chrono::CRM with MCC (not reproduce the paper),
build the `main` branch of Project Chrono once the MCC work has been merged.

---

# 2. Dependencies

## 2.1 Build-time

| Dependency | Version | Notes |
| --- | --- | --- |
| CMake | ≥ 3.26.5 | |
| GCC / Clang | ≥ 11.0 (GCC 11+) | C++17 |
| CUDA Toolkit | ≥ 12.0 (12.3 recommended) | Required by Chrono::FSI SPH / CRM |
| Eigen3 | ≥ 3.4.0 | |
| OpenMP | | |
| (optional) Intel MKL | 2024.0 | Only if `CH_ENABLE_MODULE_PARDISO_MKL=ON` |
| (optional) VulkanSceneGraph (VSG) | latest | Runtime visualization |

On the NREL / NCSA clusters used for the paper runs, the compilation
environment is loaded with:

```bash
module load intel-oneapi-mkl
module load cuda/12.3
```

## 2.2 Python runtime (plotting)

The plotting scripts require a Python ≥ 3.10 interpreter with the following
packages:

```text
numpy
pandas
matplotlib
seaborn
```

A minimal environment can be created with:

```bash
python3 -m venv mcc_venv
source mcc_venv/bin/activate
pip install numpy pandas matplotlib seaborn
```

---

# 3. Building Chrono

Chrono forbids in-source builds. Create a dedicated build directory at the top
of the repository (the scripts assume the name `build_new/`, but any name
works as long as the shell scripts are launched from `<build>/bin/`).

```bash
cd chrono-wisc
mkdir build_new
cd build_new
```

Configure with the module set actually used for the paper (verified from the
`build_new/CMakeCache.txt` of the authors' environment):

```bash
cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_DEMOS=ON \
  -DBUILD_DEMOS_FSI=ON \
  -DBUILD_DEMOS_VEHICLE=ON \
  -DBUILD_DEMOS_VSG=ON \
  -DCH_ENABLE_MODULE_FSI=ON \
  -DCH_ENABLE_MODULE_FSI_SPH=ON \
  -DCH_ENABLE_MODULE_VEHICLE=ON \
  -DCH_ENABLE_MODULE_VEHICLE_MODELS=ON \
  -DCH_ENABLE_MODULE_POSTPROCESS=ON \
  -DCH_ENABLE_MODULE_VSG=ON \
  -DCH_USE_FSI_DOUBLE=OFF
```

Notes:
- `CH_USE_FSI_DOUBLE=OFF` (single precision) matches the numerics in the paper.
- VSG is optional but recommended; if it is not available, drop
  `-DBUILD_DEMOS_VSG=ON -DCH_ENABLE_MODULE_VSG=ON` and pass `--no_vis` when
  running the demos (all paper runs already do).
- If you need the PARDISO sparse solver, add
  `-DCH_ENABLE_MODULE_PARDISO_MKL=ON` (requires Intel oneAPI MKL 2024.0).
- CUDA architecture is normally auto-detected; if not, set
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>` for your GPU (e.g. `80` for A100,
  `89` for RTX 4080, `90` for H100).

Compile:

```bash
cmake --build . -j
```

After the build completes, the executables used by the paper are installed in
`build_new/bin/`:

- `demo_FSI_ConePenetrometer`
- `demo_FSI_NormalBevameter`
- `demo_FSI_SlopedSingleWheelTest`
- `demo_FSI_SlopedSingleWheelRTFBenchmark`
- `demo_ROBOT_Lander_CRM`
- `demo_ROBOT_Lander_CRM_RTFBenchmark`

Please open an issue on `uwsbel/chrono-wisc` (and tag `@Huzaifg`) for any
build problems.

---

# 4. Reproducing the Results

All shell scripts in `shell_scripts/` are copies of the exact scripts used to
generate the paper data. Copy the one you want to run into `build_new/bin/`
(which is where the executables live) and launch it from that directory.

```bash
cp shell_scripts/<script>.sh build_new/bin/
cd build_new/bin
chmod +x <script>.sh
./<script>.sh
```

On SLURM clusters, the scripts with `#SBATCH` headers can be submitted
directly via `sbatch <script>.sh`; adjust the partition / account lines to
match your allocation before doing so.

Each demo writes into a sub-folder of `./DEMO_OUTPUT/` in the current working
directory (the default returned by `GetChronoOutputPath()`), keyed by the run
parameters. The plot scripts in `python_scripts/` read back from that folder.

Sections 4.1–4.5 below correspond one-to-one to the paper sub-sections.

## 4.1 Cone Penetrometer Test (§4.1, Fig. 2)

Setup: 60° cone, base area 323 mm², penetrating at 0.3 cm/s to 18 cm, in a
0.3 × 0.3 × 0.24 m bin of GRC-1 at ρ = 1600 kg/m³. The script runs one MCC
case (`OCR = 10`, κ = 0.00625, λ = 0.025 in the demo's defaults) and a μ(I)
cohesion sweep `c ∈ {0, 100, 1000, 5000} Pa`.

```bash
# From <repo>/build_new/bin/
cp ../../shell_scripts/cone_nrel.sh .
./cone_nrel.sh
```

Output is written into `DEMO_OUTPUT/FSI_ConePenetrometer_GRC1_*`.

The CPT demo accepts the following CLI options (see
`src/demos/fsi/demo_FSI_ConePenetrometer.cpp`):
`--rheology_model_crm {MU_OF_I|MCC}`, `--pre_pressure_scale <OCR>`,
`--cohesion <Pa>`, `--container_height <m>`, `--initial_spacing <m>`,
`--mu_s`, `--mu_2`, `--density`, `--y_modulus`, `--penetration_depth`.

## 4.2 Normal Bevameter Test (§4.2, Fig. 3)

Setup: 19 cm plate, pressure ramp 0 → 30 kPa over 3 s, soil bin
0.584 × 0.584 × 0.24 m, GRC-1 at ρ = 1670 kg/m³. The script sweeps cohesion
`c ∈ {0, 100, 1000, 5000} Pa` × bin heights `{0.024, 0.12, 0.24} m`; the
paper figure uses only the 0.24 m cases. MCC uses `OCR = 20`, κ = 0.00625,
λ = 0.025.

```bash
# From <repo>/build_new/bin/
cp ../../shell_scripts/bev_nrel.sh .
./bev_nrel.sh
```

Output: `DEMO_OUTPUT/FSI_NormalBevameter_GRC1_<heightCm>_<model>_58.4cm*`.

Relevant CLI flags (`src/demos/fsi/demo_FSI_NormalBevameter.cpp`):
`--rheology_model_crm`, `--pre_pressure_scale`, `--cohesion`,
`--container_height`, `--plate_diameter`, `--max_pressure`.

## 4.3 MGRU3 Single-Wheel Test (§4.3, Figs. 4–5)

### 4.3.1 Constant-depth slope sweep (Fig. 4)

Single MGRU3 wheel, 24 grousers (height 0.03 m), ω = 0.8 rad/s,
sprung mass 17.5 kg (one quarter of the rover), soil bin
5 × 0.8 × 0.2 m, GRC-1 at ρ = 1760 kg/m³. Slope angles
`{0, 2.5, 5, 10, 15, 20, 25}°` for MCC (`OCR = 2`, κ = 0.2, λ = 0.8)
only; rerun with `RHEOLOGY_MODEL_CRM_VALUES=("MU_OF_I")` at the top of the
script for the μ(I) curve.

```bash
# From <repo>/build_new/bin/
cp ../../shell_scripts/viper_wheel_paperRuns.sh .
./viper_wheel_paperRuns.sh
```

Output: `DEMO_OUTPUT/FSI_SlopedSingleWheelTest/...`.

### 4.3.2 Depth-sensitivity sweep (Fig. 5)

Same wheel and constitutive parameters as above, but with an additional
`--container_depth` sweep over `{0.1, 0.5, 1.0} m`, run for both MCC and
μ(I):

```bash
cp ../../shell_scripts/viper_wheel_paperRuns_DepthVar.sh .
./viper_wheel_paperRuns_DepthVar.sh
```

## 4.4 Lunar Lander Drop Test (§4.4, Figs. 6–7)

Simplified four-legged rigid lander (body 4 m × 1 m Ø × 2000 kg; four 0.5 m Ø
footpads; total ~2048 kg) dropped at 1 m/s under lunar gravity onto a
6 × 6 × 0.3 m CRM bed (GRC-1 at ρ = 1700 kg/m³, E = 1 MPa, Δx = 0.02 m).
The paper reports a 168-run MCC sweep:

- `--pre_pressure_scale (OCR) ∈ {1.1, 2, 5, 10, 15, 20}`
- `--kappa ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}`
- `--lambda = λ_mult · κ`, with `λ_mult ∈ {4, 6, 10, 20}`

plus a single μ(I) baseline (μs = μd = 0.6, c = 0 Pa).

On a SLURM cluster:

```bash
cp shell_scripts/lander_runs_sbatch.sh build_new/bin/
cd build_new/bin
sbatch lander_runs_sbatch.sh
```

The array job submits tasks `0–167` with up to 8 concurrent GPUs
(`--array=0-167%8`). Task 0 additionally runs the μ(I) baseline once.

On a workstation (serial), you can replace the SLURM dispatch with a nested
loop (the parameter arrays in the script can be reused verbatim):

```bash
for pps in 1.1 2 5 10 15 20; do
  for k in 0.01 0.02 0.05 0.1 0.2 0.5 1.0; do
    for m in 4 6 10 20; do
      l=$(echo "$m * $k" | bc -l)
      ./demo_ROBOT_Lander_CRM \
          --rheology_model_crm MCC \
          --pre_pressure_scale "$pps" --kappa "$k" --lambda "$l" \
          --no_vis
    done
  done
done
./demo_ROBOT_Lander_CRM --rheology_model_crm MU_OF_I --no_vis
```

Output folders follow the pattern
`DEMO_OUTPUT/ROBOT_Lander_CRM_<model>_gravity_planet_moon_*_pre_pressure_scale_<OCR>_kappa_<κ>_lambda_<λ>/`.

> **Post-run rename.** `plot_lander_crm_sweep.py` defaults to
> `build_new/bin/DEMO_OUTPUT_NREL_LANDER/`. Either rename the sweep folder to
> that name, or pass the actual path via `--input-dir`. The latter is
> recommended.

Relevant CLI flags (`src/demos/robot/lander/demo_ROBOT_Lander_CRM.cpp`):
`--rheology_model_crm`, `--pre_pressure_scale`, `--kappa`, `--lambda`,
`--flat_terrain`, `--no_vis`, `--particle_output`, `--blender_output`,
`--gravity_planet {earth|mars|moon}`, `--gravity_polar_deg`,
`--gravity_azimuth_deg`.

## 4.5 Efficiency Comparison (§4.5, Table 1)

Runs one wheel case and one lander case under matched numerical settings on a
single GPU, with visualization, particle output, and body-trajectory output
disabled. These two benchmarks generate the wall-clock numbers in Table 1.

```bash
# MGRU3 single-wheel RTF, slope 10°, 0.1 m bin depth, 0.5 s simulated
./demo_FSI_SlopedSingleWheelRTFBenchmark \
    --slope_angle=10 --container_depth=0.1 --total_time=0.5 \
    --initial_spacing=0.01 --d0_multiplier=1.3 --time_step=2.5e-4 \
    --rheology_model_crm=MCC --pre_pressure_scale=2.0 --kappa=0.2 --lambda=0.8 \
    --no_vis
# Repeat with --rheology_model_crm=MU_OF_I for the μ(I) row.

# Lunar lander RTF, flat terrain, 0.5 s simulated
./demo_ROBOT_Lander_CRM_RTFBenchmark \
    --flat_terrain true --rheology_model_crm MCC \
    --pre_pressure_scale 2.0 --kappa 0.01 --lambda 0.04 --no_vis
./demo_ROBOT_Lander_CRM_RTFBenchmark \
    --flat_terrain true --rheology_model_crm MU_OF_I --no_vis
```

The wall-clock time printed at the end of each run is divided by the
simulated duration (0.5 s) to obtain the RTF values in Table 1.

---

# 5. Reproducing the Figures

Copy the plotting scripts into `build_new/bin/` (same working directory as the
`DEMO_OUTPUT` tree) and run them with the Python environment from §2.2. Each
script creates or uses a sibling `./paper_plots/` directory for output.

```bash
cp python_scripts/*.py build_new/bin/
cd build_new/bin
mkdir -p paper_plots
```

| Figure in paper | Command | Output |
| --- | --- | --- |
| Fig. 2b (CPT) | `python penetrometer_plot_single.py` | `paper_plots/penetrometer_comparison.png` |
| Fig. 3b (Bevameter) | `python plot_plate_penetration.py` | `paper_plots/bevameter_comp.png` |
| Fig. 4 (Wheel, const. depth) | `python viper_singleWheel_plotting.py` | `paper_plots/Viper_slip_vs_slope_rheology_comparison_ps*.png` |
| Fig. 5a (Wheel, MCC depth sweep) | `python viper_singleWheel_DepthVar_plotting.py --rheology mcc` | `paper_plots/Viper_slip_vs_slope_depth_comparison_mcc_only.png` |
| Fig. 5b (Wheel, μ(I) depth sweep) | `python viper_singleWheel_DepthVar_plotting.py --rheology mu_i` | `paper_plots/Viper_slip_vs_slope_depth_comparison_mu_i_only.png` |
| Figs. 6–7 (Lander) | `python plot_lander_crm_sweep.py --input-dir DEMO_OUTPUT --output-dir paper_plots` | `paper_plots/lander_grouped_profiles.png`, `paper_plots/lander_main_effects.png` |

Notes:

- `penetrometer_plot_single.py` imports `penetrometer_folderParser.py` and
  `penetrometer_plotting.py`; copy all three into the working directory.
- The cone and bevameter plotters use a fixed query dictionary matching the
  `shell_scripts/*.sh` defaults. If you change SPH spacing, cohesion, or
  bin height in the run scripts, edit the `query_dict` / `RUN_CONFIGS` at the
  top of the corresponding plot script to match.
- `plot_lander_crm_sweep.py --input-dir` must point at the directory that
  contains the `ROBOT_Lander_CRM_*` sub-folders. By default it looks at
  `build_new/bin/DEMO_OUTPUT_NREL_LANDER`; pass `--input-dir DEMO_OUTPUT` if
  you kept the default output location from §4.4.

---

# 6. Expected Wall-Clock

All paper runs were performed on NVIDIA H100 GPUs (NREL Kestrel) except the
Table 1 RTF benchmarks, which were run on a single NVIDIA GeForce RTX 4080.
Order-of-magnitude costs on the RTX 4080:

| Case | Sim. time | Wall-clock |
| --- | --- | --- |
| MGRU3 wheel (single slope, Δx = 1 cm, 0.2 m bin) | ~4 s | ~10 min |
| Lander (single MCC run, flat terrain, 1 s) | ~1 s | ~4 min |
| CPT (Δx = 1 mm, 0.24 m bin) | ~60 s | ~4 h |
| Bevameter (Δx = 2 mm, 0.24 m bin) | ~3 s | ~30 min |

Full sweeps (lander 168-case, wheel 7-slope × 3-depth × 2-rheology) should be
distributed across multiple GPUs.

---

# 7. Troubleshooting & Contact

- Build issues: open an issue on
  [`uwsbel/chrono-wisc`](https://github.com/uwsbel/chrono-wisc/issues) and
  tag `@Huzaifg`.
- Missing `DEMO_OUTPUT/...` folders when plotting: confirm that the run
  scripts finished successfully (check the per-run logs under
  `viper_wheel_logs/` or the SLURM `*.out`/`*.err` files) and that you launched
  them from `build_new/bin/`, not from the repository root.
- Plot scripts that raise `FileNotFoundError`: the `query_dict` /
  `path_components` at the top of each plot script hard-codes the exact run
  configuration used for the paper. If you intentionally changed a run
  parameter, update those dictionaries to match.
