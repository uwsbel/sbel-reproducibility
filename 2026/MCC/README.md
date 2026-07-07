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
| 4.5 Efficiency (RTF) | Wheel/lander wall-clock | `demo_FSI_SlopedSingleWheelRTFBenchmark`, `demo_ROBOT_Lander_CRM_RTFBenchmark` | Run by hand (see §4.5) | n/a (Table 1 in paper) |

The intended workflow now keeps **Chrono itself** and the **paper/demo project**
separate. Chrono is configured and built once, and the custom demos are built in
a separate build directory that links against that Chrono build.

---

# 1. Intended Directory Layout

Use one top-level workspace directory with sibling folders for Chrono, external
dependencies, and this demo/reproducibility project:

```text
project_top_folder/
├── chrono/                              # Chrono source tree
│   └── build/                           # Chrono build tree, generated locally
├── dependencies/
│   └── vsg/                             # VSG install tree, if using Chrono::VSG
└── MCC/                                 # the contents of this project (sbel-reproducibility/2026/MCC)
    ├── README.md
    ├── CMakeLists.txt                   # demo-specific CMake project
    ├── build_chrono_and_project.sh      # full build script (chrono source code and demos)
    ├── rebuild_project.sh               # quick rebuild script (just demos)
    ├── demo_FSI_ConePenetrometer.cpp
    ├── demo_FSI_NormalBevameter.cpp
    ├── demo_FSI_SlopedSingleWheelTest.cpp
    ├── demo_FSI_SlopedSingleWheelRTFBenchmark.cpp
    ├── demo_ROBOT_Lander_CRM.cpp
    ├── demo_ROBOT_Lander_CRM_RTFBenchmark.cpp
    ├── generate_heightmap.py            # heightmap generator for the lander non-flat terrain (see note below)
    ├── shell_scripts/
    └── build/                           # demo build tree, generated locally
```

Important points:

- `chrono/` is the only Chrono source checkout.
- `chrono/build/` is the Chrono build used by this project.
- `MCC/build/` is only for the custom demo executables.
- Do not build inside the Chrono source tree except for the dedicated
  `chrono/build/` out-of-source build directory.
- The custom project finds Chrono through `Chrono_DIR=<workspace>/chrono/build/cmake`.

This README is intended to live inside `MCC/`, next to
the demo-specific `CMakeLists.txt`, build scripts, demo sources, and run scripts.

> **Note on `generate_heightmap.py`.** This helper script generates the randomized
> crater/bump heightmap used by the lander demos when they run with non-flat
> terrain (the default for `demo_ROBOT_Lander_CRM`). It lives in `MCC/` alongside
> the demo sources rather than in the Chrono data tree. The project `CMakeLists.txt`
> passes the demo source directory to the build via the `DEMO_SOURCE_DIR` compile
> definition so the lander demos can locate this script regardless of the working
> directory they are launched from. It requires Python 3 with `numpy` and `Pillow`
> (and, optionally, the `noise` package for faster Perlin noise). It is only needed
> for non-flat terrain runs; runs with `--flat_terrain true` do not invoke it.

---

# 2. Source Repositories

## 2.1 Chrono source

| Item | Value |
| --- | --- |
| Repository | `git@github.com:projectchrono/chrono.git` |
| Branch | `main` |

From the desired root:

```bash
mkdir -p ~/project_top_folder 
cd ~/project_top_folder

git clone git@github.com:projectchrono/chrono.git
```

## 2.2 Demo/reproducibility project

This project should be placed as a sibling of `chrono/`:

```text
project_top_folder/
├── chrono/
└── MCC/
```

The MCC directory owns the demo `.cpp` files, run scripts,
plotting scripts, and its own `CMakeLists.txt`. It does not modify the Chrono
source tree during normal use.

---

# 3. Dependencies

## 3.1 Build-time

| Dependency | Version | Notes |
| --- | --- | --- |
| CMake | ≥ 3.26.5 | |
| GCC / Clang | ≥ 11.0 (GCC 11+) | C++17 |
| CUDA Toolkit | ≥ 12.0 (12.3 recommended) | Required by Chrono::FSI SPH / CRM |
| Eigen3 | ≥ 3.4.0 | |
| OpenMP | | |
| Ninja or Make | | Used by CMake build backend |
| (optional) Intel MKL | 2024.0 | Only if `CH_ENABLE_MODULE_PARDISO_MKL=ON` |
| (optional) VulkanSceneGraph (VSG) | Chrono-compatible version | Runtime visualization |

On the NREL / NCSA clusters used for the paper runs, the compilation
environment is loaded with:

```bash
module load intel-oneapi-mkl
module load cuda/12.3
```

For local Linux workstations, verify the NVIDIA driver and CUDA Toolkit
separately:

```bash
nvidia-smi        # driver/GPU visibility
nvcc --version    # CUDA Toolkit compiler
```

## 3.2 VSG dependency

If building with `CH_ENABLE_MODULE_VSG=ON`, install/build VSG first. The Chrono
VSG build script should install VSG under:

```text
<workspace>/dependencies/vsg/
```

The Chrono configure step then needs either:

```bash
-DCMAKE_PREFIX_PATH=<workspace>/dependencies/vsg
```

or the exact VSG CMake package directory:

```bash
-Dvsg_DIR=<workspace>/dependencies/vsg/lib/cmake/vsg
```

## 3.3 Python runtime for plotting

The plotting scripts require Python ≥ 3.10 with:

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

The lander heightmap generator (`generate_heightmap.py`, see §1 and §5.4)
additionally requires `numpy` and `Pillow`, plus the optional `noise` package for
faster Perlin noise. These are only needed for non-flat lander terrain runs:

```bash
pip install numpy Pillow
pip install noise   # optional, faster Perlin noise
```

---

# 4. Building and Rebuilding

There are two build layers:

1. **Chrono build**: configures and compiles Chrono with FSI/CRM, Vehicle,
   Postprocess, and optional VSG support.
2. **Custom demo build**: compiles the paper demo executables and links them
   against the existing Chrono build.

## 4.1 Full build: Chrono + custom demos

Use the full build script when setting up the workspace for the first time, when
Chrono has not been built yet, or when changing Chrono module options.

From `MCC`:

```bash
chmod +x build_chrono_and_project.sh
./build_chrono_and_project.sh
```

The script should follow this pattern:

```bash
#!/usr/bin/env bash
set -e

ROOT="$(realpath ..)"

CHRONO_SRC="$ROOT/chrono"
CHRONO_BUILD="$ROOT/chrono/build"

PROJECT_SRC="$ROOT/MCC"
PROJECT_BUILD="$PROJECT_SRC/build"

VSG_PREFIX="$ROOT/dependencies/vsg"
VSG_DIR="$VSG_PREFIX/lib/cmake/vsg"

mkdir -p "$CHRONO_BUILD"

cmake -S "$CHRONO_SRC" -B "$CHRONO_BUILD" \
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
  -DCH_USE_FSI_DOUBLE=OFF \
  -DCMAKE_PREFIX_PATH="$VSG_PREFIX" \
  -Dvsg_DIR="$VSG_DIR"

cmake --build "$CHRONO_BUILD" -j"$(nproc)"

mkdir -p "$PROJECT_BUILD"

cmake -S "$PROJECT_SRC" -B "$PROJECT_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DChrono_DIR="$CHRONO_BUILD/cmake"

cmake --build "$PROJECT_BUILD" -j"$(nproc)"
```

Notes:

- `CH_USE_FSI_DOUBLE=OFF` uses single precision, matching the paper runs.
- `BUILD_DEMOS_*` builds Chrono's own demos. The custom paper demos are built
  by this project's `CMakeLists.txt` in the second configure/build step.
- If VSG is not available, remove the VSG flags and use `--no_vis` when running
  demos.
- If you need PARDISO MKL, add `-DCH_ENABLE_MODULE_PARDISO_MKL=ON` and make
  sure Intel oneAPI MKL is loaded/found.
- If CUDA architecture is not auto-detected, set
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>` for your GPU.

## 4.2 Custom project CMake pattern

The demo-specific `CMakeLists.txt` should find the already-built Chrono package:

```cmake
find_package(Chrono
             COMPONENTS FSI
             OPTIONAL_COMPONENTS VSG PardisoMKL Postprocess
             CONFIG REQUIRED)
```

Each demo executable is a separate CMake target. A compact pattern is:

```cmake
set(DEMOS
    demo_FSI_ConePenetrometer
    demo_FSI_NormalBevameter
    demo_FSI_SlopedSingleWheelTest
    demo_FSI_SlopedSingleWheelRTFBenchmark
    demo_ROBOT_Lander_CRM
    demo_ROBOT_Lander_CRM_RTFBenchmark
)

foreach(demo ${DEMOS})
    add_executable(${demo} ${demo}.cpp)
    target_include_directories(${demo} PRIVATE ${CHRONO_INCLUDE_DIRS})
    set_target_properties(${demo} PROPERTIES
        COMPILE_FLAGS "${CHRONO_CXX_FLAGS}"
        LINK_FLAGS    "${CHRONO_LINKER_FLAGS}"
    )
    target_link_libraries(${demo} ${CHRONO_LIBRARIES})

    # Make the demo source directory available at compile time so the lander demos
    # can locate generate_heightmap.py regardless of the working directory.
    target_compile_definitions(${demo} PRIVATE
        DEMO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
```

The `DEMO_SOURCE_DIR` compile definition is what lets `demo_ROBOT_Lander_CRM` and
`demo_ROBOT_Lander_CRM_RTFBenchmark` find `generate_heightmap.py` in the `MCC/`
source directory rather than in the Chrono data tree.

## 4.3 Quick rebuild after editing a demo

After the project has been configured once, ordinary edits to a demo `.cpp` file
do not require rebuilding Chrono and usually do not require rerunning CMake.
Only rebuild the custom project:

```bash
cmake --build build -j"$(nproc)"
```

A minimal `rebuild_project.sh` can be:

```bash
#!/usr/bin/env bash
set -e

ROOT="$(realpath ..)"
PROJECT_BUILD="$ROOT/MCC/build"

cmake --build "$PROJECT_BUILD" -j"$(nproc)"
```

To rebuild only one executable:

```bash
cmake --build build --target demo_FSI_ConePenetrometer -j"$(nproc)"
```

or through a target-aware script:

```bash
#!/usr/bin/env bash
set -e

ROOT="$(realpath ..)"
PROJECT_BUILD="$ROOT/MCC/build"
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    cmake --build "$PROJECT_BUILD" -j"$(nproc)"
else
    cmake --build "$PROJECT_BUILD" --target "$TARGET" -j"$(nproc)"
fi
```

Usage:

```bash
./rebuild_project.sh
./rebuild_project.sh demo_ROBOT_Lander_CRM
```

## 4.4 When to rerun configure

Rerun the custom project configure step if you:

- edit `CMakeLists.txt`,
- add or remove a demo executable,
- delete `MCC/build/`,
- switch to a different Chrono build, or
- change the `Chrono_DIR` path.

Use:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DChrono_DIR="$(realpath ../chrono/build/cmake)"
```

Then rebuild:

```bash
cmake --build build -j"$(nproc)"
```

---

# 5. Running the Demos

The custom demo executables are built in this project's build directory. Unless
your `CMakeLists.txt` sets a different `RUNTIME_OUTPUT_DIRECTORY`, run from:

```bash
cd MCC/build
```

The run scripts expect to be launched from the directory containing the demo
executables. They read and write relative paths such as:

```text
./DEMO_OUTPUT
./paper_plots
./DEMO_OUTPUT_NREL_LANDER
```

To run a script:

```bash
cd build
cp ../shell_scripts/<script>.sh .
chmod +x <script>.sh
./<script>.sh
```

On SLURM clusters, scripts with `#SBATCH` headers can be submitted directly:

```bash
sbatch <script>.sh
```

Adjust partition/account lines to match your allocation.

Each demo writes into a sub-folder of `./DEMO_OUTPUT/` in the current working
directory, keyed by the run parameters. The plot scripts in `python_scripts/`
read back from that folder.

Sections 5.1 through 5.5 below correspond one-to-one to the paper subsections.

## 5.1 Cone Penetrometer Test (§4.1, Fig. 2)

Setup: 60° cone, base area 323 mm², penetrating at 0.3 cm/s to 18 cm, in a
0.3 × 0.3 × 0.24 m bin of GRC-1 at ρ = 1600 kg/m³. The script runs one MCC
case (`OCR = 10`, κ = 0.00625, λ = 0.025 in the demo's defaults) and a μ(I)
cohesion sweep `c ∈ {0, 100, 1000, 5000} Pa`.

```bash
# From MCC/build/
cp ../shell_scripts/cone_nrel.sh .
./cone_nrel.sh
```

Output is written into `DEMO_OUTPUT/FSI_ConePenetrometer_GRC1_*`.

The CPT demo accepts the following CLI options:
`--rheology_model_crm {MU_OF_I|MCC}`, `--pre_pressure_scale <OCR>`,
`--cohesion <Pa>`, `--container_height <m>`, `--initial_spacing <m>`,
`--mu_s`, `--mu_2`, `--density`, `--y_modulus`, `--penetration_depth`.

## 5.2 Normal Bevameter Test (§4.2, Fig. 3)

Setup: 19 cm plate, pressure ramp 0 to 30 kPa over 3 s, soil bin
0.584 × 0.584 × 0.24 m, GRC-1 at ρ = 1670 kg/m³. The script sweeps cohesion
`c ∈ {0, 100, 1000, 5000} Pa` × bin heights `{0.024, 0.12, 0.24} m`; the
paper figure uses only the 0.24 m cases. MCC uses `OCR = 20`, κ = 0.00625,
λ = 0.025.

```bash
# From MCC/build/
cp ../shell_scripts/bev_nrel.sh .
./bev_nrel.sh
```

Output: `DEMO_OUTPUT/FSI_NormalBevameter_GRC1_<heightCm>_<model>_58.4cm*`.

Relevant CLI flags:
`--rheology_model_crm`, `--pre_pressure_scale`, `--cohesion`,
`--container_height`, `--plate_diameter`, `--max_pressure`.

## 5.3 MGRU3 Single-Wheel Test (§4.3, Figs. 4 to 5)

### 5.3.1 Constant-depth slope sweep (Fig. 4)

Single MGRU3 wheel, 24 grousers (height 0.03 m), ω = 0.8 rad/s,
sprung mass 17.5 kg, soil bin 5 × 0.8 × 0.2 m, GRC-1 at ρ = 1760 kg/m³.
Slope angles `{0, 2.5, 5, 10, 15, 20, 25}°` for MCC (`OCR = 2`, κ = 0.2,
λ = 0.8) only; rerun with `RHEOLOGY_MODEL_CRM_VALUES=("MU_OF_I")` at the top
of the script for the μ(I) curve.

```bash
# From MCC/build/
cp ../shell_scripts/viper_wheel_paperRuns.sh .
./viper_wheel_paperRuns.sh
```

Output: `DEMO_OUTPUT/FSI_SlopedSingleWheelTest/...`.

### 5.3.2 Depth-sensitivity sweep (Fig. 5)

Same wheel and constitutive parameters as above, but with an additional
`--container_depth` sweep over `{0.1, 0.5, 1.0} m`, run for both MCC and μ(I):

```bash
cp ../shell_scripts/viper_wheel_paperRuns_DepthVar.sh .
./viper_wheel_paperRuns_DepthVar.sh
```

## 5.4 Lunar Lander Drop Test (§4.4, Figs. 6 to 7)

Simplified four-legged rigid lander (body 4 m × 1 m Ø × 2000 kg; four 0.5 m Ø
footpads; total ~2048 kg) dropped at 1 m/s under lunar gravity onto a
6 × 6 × 0.3 m CRM bed (GRC-1 at ρ = 1700 kg/m³, E = 1 MPa, Δx = 0.02 m).
The paper reports a 168-run MCC sweep:

- `--pre_pressure_scale (OCR) ∈ {1.1, 2, 5, 10, 15, 20}`
- `--kappa ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}`
- `--lambda = λ_mult · κ`, with `λ_mult ∈ {4, 6, 10, 20}`

plus a single μ(I) baseline (μs = μd = 0.6, c = 0 Pa).

> **Terrain and `generate_heightmap.py`.** `demo_ROBOT_Lander_CRM` defaults to
> non-flat terrain, which is generated at run time by `generate_heightmap.py`
> (see §1 and §3.3). The demo locates the script through the `DEMO_SOURCE_DIR`
> compile definition set by the project `CMakeLists.txt`, so the script must be
> present in the `MCC/` source directory and the project must have been
> (re)configured after that definition was added. The script needs Python 3 with
> `numpy` and `Pillow`. To skip heightmap generation entirely (flat 6 × 6 × 0.3 m
> bed, no rocks), pass `--flat_terrain true`; the paper sweep below uses the
> default non-flat terrain.

On a SLURM cluster:

```bash
# From MCC/build/
cp ../shell_scripts/lander_runs_sbatch.sh .
sbatch lander_runs_sbatch.sh
```

The array job submits tasks `0-167` with up to 8 concurrent GPUs
(`--array=0-167%8`). Task 0 additionally runs the μ(I) baseline once.

On a workstation (serial), you can replace the SLURM dispatch with a nested
loop:

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
> `build/DEMO_OUTPUT_NREL_LANDER/`. Either rename the sweep folder to that name,
> or pass the actual path via `--input-dir`. The latter is recommended.

Relevant CLI flags:
`--rheology_model_crm`, `--pre_pressure_scale`, `--kappa`, `--lambda`,
`--flat_terrain`, `--no_vis`, `--particle_output`, `--blender_output`,
`--gravity_planet {earth|mars|moon}`, `--gravity_polar_deg`,
`--gravity_azimuth_deg`.

## 5.5 Efficiency Comparison (§4.5, Table 1)

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

The Table 1 lander benchmarks use `--flat_terrain true`, so they do not invoke
`generate_heightmap.py`. The wall-clock time printed at the end of each run is
divided by the simulated duration (0.5 s) to obtain the RTF values in Table 1.

---

# 6. Reproducing the Figures

Copy the plotting scripts into the custom project build directory, which should
also contain the `DEMO_OUTPUT` tree:

```bash
cd MCC/build
cp ../python_scripts/*.py .
mkdir -p paper_plots
```

| Figure in paper | Command | Output |
| --- | --- | --- |
| Fig. 2b (CPT) | `python penetrometer_plot_single.py` | `paper_plots/penetrometer_comparison.png` |
| Fig. 3b (Bevameter) | `python plot_plate_penetration.py` | `paper_plots/bevameter_comp.png` |
| Fig. 4 (Wheel, const. depth) | `python viper_singleWheel_plotting.py` | `paper_plots/Viper_slip_vs_slope_rheology_comparison_ps*.png` |
| Fig. 5a (Wheel, MCC depth sweep) | `python viper_singleWheel_DepthVar_plotting.py --rheology mcc` | `paper_plots/Viper_slip_vs_slope_depth_comparison_mcc_only.png` |
| Fig. 5b (Wheel, μ(I) depth sweep) | `python viper_singleWheel_DepthVar_plotting.py --rheology mu_i` | `paper_plots/Viper_slip_vs_slope_depth_comparison_mu_i_only.png` |
| Figs. 6 to 7 (Lander) | `python plot_lander_crm_sweep.py --input-dir DEMO_OUTPUT --output-dir paper_plots` | `paper_plots/lander_grouped_profiles.png`, `paper_plots/lander_main_effects.png` |

Notes:

- `penetrometer_plot_single.py` imports `penetrometer_folderParser.py` and
  `penetrometer_plotting.py`; copy all three into the working directory.
- The cone and bevameter plotters use a fixed query dictionary matching the
  `shell_scripts/*.sh` defaults. If you change SPH spacing, cohesion, or bin
  height in the run scripts, edit the `query_dict` / `RUN_CONFIGS` at the top
  of the corresponding plot script.
- `plot_lander_crm_sweep.py --input-dir` must point at the directory that
  contains the `ROBOT_Lander_CRM_*` sub-folders. Pass `--input-dir DEMO_OUTPUT`
  if you kept the default output location from §5.4.

---

# 7. Expected Wall-Clock

All paper runs were performed on NVIDIA H100 GPUs (NREL Kestrel) except the
Table 1 RTF benchmarks, which were run on a single NVIDIA GeForce RTX 4080.
Order-of-magnitude costs on the RTX 4080:

| Case | Sim. time | Wall-clock |
| --- | --- | --- |
| MGRU3 wheel (single slope, Δx = 1 cm, 0.2 m bin) | ~4 s | ~10 min |
| Lander (single MCC run, flat terrain, 1 s) | ~1 s | ~4 min |
| CPT (Δx = 1 mm, 0.24 m bin) | ~60 s | ~4 h |
| Bevameter (Δx = 2 mm, 0.24 m bin) | ~3 s | ~30 min |

Full sweeps should be distributed across multiple GPUs.

---

# 8. Troubleshooting

- **Chrono cannot find VSG:** verify that
  `<workspace>/dependencies/vsg/lib/cmake/vsg/vsgConfig.cmake` exists and pass either
  `-Dvsg_DIR=<workspace>/dependencies/vsg/lib/cmake/vsg` or
  `-DCMAKE_PREFIX_PATH=<workspace>/dependencies/vsg` during the Chrono configure step.
- **`nvcc` is not found:** the NVIDIA driver may be installed even if the CUDA
  Toolkit is not on `PATH`. Check `nvidia-smi`, `nvcc --version`, and the CUDA
  Toolkit install path.
- **Custom project cannot find Chrono:** verify that
  `<workspace>/chrono/build/cmake/ChronoConfig.cmake` exists and configure the
  custom project with `-DChrono_DIR=<workspace>/chrono/build/cmake`.
- **Edits to a demo do not appear:** rebuild the custom project with
  `cmake --build build -j"$(nproc)"` from `MCC/`.
- **Lander demo reports "Heightmap generator script not found":** the lander
  demos expect `generate_heightmap.py` in the `MCC/` source directory and locate
  it through the `DEMO_SOURCE_DIR` compile definition. Confirm the script is
  present next to the demo sources, that `CMakeLists.txt` sets `DEMO_SOURCE_DIR`
  (see §4.2), and that you re-ran configure after adding it. As a quick
  workaround, run with `--flat_terrain true` to skip heightmap generation.
- **Lander heightmap generation fails (Python error):** ensure Python 3 with
  `numpy` and `Pillow` is available on `PATH` (see §3.3), or run with
  `--flat_terrain true`.
- **Missing `DEMO_OUTPUT/...` folders when plotting:** confirm that the run
  scripts finished successfully and that they were launched from the directory
  containing the demo executables.
- **Plot scripts raise `FileNotFoundError`:** the query dictionaries at the top
  of the plotting scripts hard-code the exact run configuration used for the
  paper. If you intentionally changed a run parameter, update those dictionaries
  to match.