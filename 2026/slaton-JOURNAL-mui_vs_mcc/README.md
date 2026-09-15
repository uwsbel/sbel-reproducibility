# Modified Cam-Clay vs. μ(I): Chrono::CRM Reproducibility Project

This repository contains the custom Project Chrono demos, experiment launch scripts,
plotting utilities, and build helpers used for the paper:

> *Modified Cam-Clay vs. μ(I): A Continuum Terramechanics Comparison within Chrono::CRM*

The repository is intentionally kept separate from the Project Chrono source tree.
Chrono is built in a sibling directory, while the custom paper demos are compiled
into this project's own `build/` directory and link against that Chrono build.

---

# 1. Repository and Workspace Layout

The current project directory is `PAPER_mui_vs_mcc`.

The expected workspace layout is:

```text
workspace/
├── chrono/                              # Project Chrono source checkout
│   └── build/                           # Chrono build tree
│
├── _deps/
│   └── vsg/                             # VulkanSceneGraph install/build prefix
│
└── PAPER_mui_vs_mcc/
    ├── build/                           # Generated build tree for the custom demos
    ├── data/                            # Symlink to ../chrono/data created by CMake
    ├── plotting_scripts/                # Manuscript plotting/post-processing scripts
    ├── shell_scripts/                   # Experiment launch scripts
    │
    ├── build_chrono_and_demos.sh        # Configure/build Chrono and this project
    ├── rebuild_demos.sh                 # Rebuild only the already-configured demos
    ├── CMakeLists.txt                   # Custom demo CMake project
    │
    ├── demo_FSI_ConePenetrometer.cpp
    ├── demo_FSI_NormalBevameter.cpp
    ├── demo_FSI_SlopedSingleWheel_RTFBenchmark.cpp
    ├── demo_FSI_SlopedSingleWheel_Test.cpp
    ├── demo_ROBOT_Lander_CRM_RTFBenchmark.cpp
    ├── demo_ROBOT_Lander_CRM.cpp
    │
    ├── generate_heightmap.py            # Runtime terrain generator for lander demos
    └── README.md
```

Important path assumptions:

- `PAPER_mui_vs_mcc/`, `chrono/`, and `_deps/` are siblings.
- Chrono is expected at `../chrono`.
- The Chrono CMake package is expected at `../chrono/build/cmake`.
- VSG is expected under `../_deps/vsg`.
- `build/` belongs only to this custom project.
- `data/` is created as a symbolic link to `../chrono/data` by the project CMake configuration.

Because several scripts still use relative paths, the working directory matters.
The sections below specify where each command should be launched.

---

# 2. Experiment Map

The paper results are organized around the following experiment families.

| Paper result | Demo executable | Run script | Plot script |
| --- | --- | --- | --- |
| Cone Penetrometer Test | `demo_FSI_ConePenetrometer` | `shell_scripts/cone_pentrometer.sh` | `plotting_scripts/cone_pentrometer_single_plot.py` |
| Normal Bevameter Test | `demo_FSI_NormalBevameter` | `shell_scripts/normal_bevelmeter.sh` | `plotting_scripts/normal_bevelmeter_plot.py` |
| MGRU3 wheel, constant depth | `demo_FSI_SlopedSingleWheel_Test` | `shell_scripts/single_wheel_test.sh` | `plotting_scripts/single_wheel_test_plot.py` |
| MGRU3 wheel, depth sensitivity | `demo_FSI_SlopedSingleWheel_Test` | `shell_scripts/single_wheel_test_depth_var.sh` | `plotting_scripts/single_wheel_test_depth_var_plot.py` |
| Lunar lander touchdown study | `demo_ROBOT_Lander_CRM` | `shell_scripts/lander.sh` | `plotting_scripts/lander_plot.py` |
| Efficiency / real-time-factor benchmark | `demo_FSI_SlopedSingleWheel_RTFBenchmark`, `demo_ROBOT_Lander_CRM_RTFBenchmark` | run directly | no dedicated plotting script |

The filenames `pentrometer` and `bevelmeter` are retained because they are the
current filenames in this repository; the experiments themselves are a cone
**penetrometer** test and a normal **bevameter** test.

---

# 3. Top-Level Source Files

## 3.1 `demo_FSI_ConePenetrometer.cpp`

Cone penetrometer validation problem using Chrono::FSI SPH / CRM.

The demo drives a 60-degree cone into granular material at a prescribed velocity
and records the force on the cone tip. The terrain model can be selected between
`MU_OF_I` and MCC, with parameters exposed through the command line.

Simulation output is organized under a directory beginning with:

```text
DEMO_OUTPUT/FSI_ConePenetrometer_GRC1_...
```

---

## 3.2 `demo_FSI_NormalBevameter.cpp`

Normal bevameter validation problem using a circular plate pressed into CRM
terrain under prescribed force.

The demo measures pressure versus sinkage and supports both `MU_OF_I` and MCC
terrain configurations.

Simulation output is organized under a directory beginning with:

```text
DEMO_OUTPUT/FSI_NormalBevameter_GRC1_...
```

---

## 3.3 `demo_FSI_SlopedSingleWheel_Test.cpp`

Main MGRU3 / VIPER-style single-wheel experiment.

The wheel is driven through CRM terrain while the effective slope is imposed by
tilting gravity. The demo records wheel motion and interaction quantities used to
compute slip-versus-slope behavior.

The source supports:

- `MU_OF_I` and MCC rheology;
- configurable terrain slope;
- configurable container depth;
- configurable proximity-search frequency;
- MCC `pre_pressure_scale`, `kappa`, and `lambda`;
- optional visualization and snapshot/output controls.

Output is organized under:

```text
DEMO_OUTPUT/FSI_SlopedSingleWheelTest/...
```

---

## 3.4 `demo_FSI_SlopedSingleWheel_RTFBenchmark.cpp`

Benchmark version of the single-wheel experiment.

It retains the single-wheel CRM physics while additionally recording
wall-clock / real-time-factor information for the efficiency comparison.

Output is organized separately from the main wheel test under:

```text
DEMO_OUTPUT/FSI_SlopedSingleWheelRTFBenchmark/...
```

---

## 3.5 `demo_ROBOT_Lander_CRM.cpp`

Lunar lander touchdown simulation on CRM terrain.

The current demo contains:

- a cylindrical rigid lander body;
- four rigidly attached legs and footpads;
- configurable Earth, Mars, or lunar gravity;
- `MU_OF_I` and MCC CRM terrain;
- optional randomized heightmap terrain;
- optional embedded rocks;
- VSG visualization;
- CSV/snapshot output.

The default terrain geometry is a `6 m x 6 m x 0.3 m` CRM bed.

Non-flat terrain is generated at runtime through `generate_heightmap.py`.

---

## 3.6 `demo_ROBOT_Lander_CRM_RTFBenchmark.cpp`

Benchmark version of the lander simulation.

It uses the same general CRM lander setup while recording timing information for
the paper's real-time-factor comparison.

---

## 3.7 `generate_heightmap.py`

Generates the cratered / uneven heightmap used by the lander demos for non-flat
terrain.

The script supports configurable:

- resolution;
- terrain length and width;
- minimum and maximum elevation;
- maximum crater count;
- crater size and depth;
- random seed.

The built-in implementation generates Perlin-style noise without requiring the
external `noise` package. An optional `--use-noise-lib` mode can use that package
when installed.

Required Python packages:

```text
numpy
Pillow
```

Optional:

```text
noise
```

The project CMake configuration defines `DEMO_SOURCE_DIR` for every demo so the
lander executables can locate `generate_heightmap.py` from the source directory
rather than depending on the runtime working directory.

---

# 4. CMake Project

`CMakeLists.txt` defines six executable targets:

```text
demo_FSI_ConePenetrometer
demo_FSI_NormalBevameter
demo_FSI_SlopedSingleWheel_RTFBenchmark
demo_FSI_SlopedSingleWheel_Test
demo_ROBOT_Lander_CRM_RTFBenchmark
demo_ROBOT_Lander_CRM
```

The project requests the following Chrono components:

```text
FSI
VSG
Vehicle
Postprocess
```

The two lander targets additionally compile Chrono's lander model source:

```text
../chrono/src/demos/robot/lander/model/Lander.cpp
```

The CMake project also:

1. adds Chrono and lander source directories to the include path;
2. applies Chrono's compile and linker flags;
3. links each executable against `${CHRONO_TARGETS}`;
4. defines `DEMO_SOURCE_DIR` for runtime helper-file lookup;
5. creates the project-level `data/` symlink pointing to `../chrono/data`.

---

# 5. Building

## 5.1 Full build: Chrono + demos

From the root of `PAPER_mui_vs_mcc`:

```bash
chmod +x build_chrono_and_demos.sh
./build_chrono_and_demos.sh
```

The script configures Chrono in:

```text
../chrono/build
```

with the following major modules enabled:

```text
FSI
FSI_SPH
Vehicle
Vehicle_Models
Postprocess
VSG
```

It then configures and builds this project in:

```text
PAPER_mui_vs_mcc/build
```

Both builds currently use four parallel jobs.

The VSG CMake prefix used by the script is:

```text
../_deps/vsg/lib/cmake
```

## 5.2 Skip rebuilding Chrono

If Chrono is already configured and built:

```bash
./build_chrono_and_demos.sh --chrono
```

Despite the flag name, the current script interprets `--chrono` as:

> **skip the Chrono build and build only the custom project**

The output will begin with:

```text
Skipping Chrono build.
Building custom project...
```

## 5.3 Quick demo rebuild

After the project has already been configured, source-only changes can normally
be rebuilt with:

```bash
./rebuild_demos.sh
```

This executes:

```bash
cmake --build ./build -j4
```

without reconfiguring Chrono or the custom project.

To rebuild one target manually:

```bash
cmake --build build --target demo_FSI_ConePenetrometer -j4
```

or, for example:

```bash
cmake --build build --target demo_FSI_SlopedSingleWheel_Test -j4
```

## 5.4 When to reconfigure

Rerun the CMake configuration when:

- `CMakeLists.txt` changes;
- a demo target is added or removed;
- `build/` is deleted;
- the Chrono build location changes;
- Chrono module configuration changes.

---

# 6. Build and Runtime Dependencies

Typical build dependencies are:

```text
CMake
C++17-capable GCC/Clang
CUDA Toolkit
Eigen3
OpenMP
Vulkan / Vulkan development headers
VulkanSceneGraph
```

Chrono::FSI SPH / CRM requires CUDA.

For the cluster scripts used here, the scripts load:

```bash
module load intel-oneapi-mkl
module load cuda/12.3
```

For local GPU/CUDA verification:

```bash
nvidia-smi
nvcc --version
```

When using VSG, both the Vulkan library **and Vulkan headers** must be available
to the compiler. In particular, the build must be able to resolve:

```text
vulkan/vulkan.h
```

---

# 7. Experiment Shell Scripts

Experiment launchers live in:

```text
shell_scripts/
```

The current files are:

```text
shell_scripts/
├── cone_pentrometer.sh
├── normal_bevelmeter.sh
├── single_wheel_test.sh
├── single_wheel_test_depth_var.sh
└── lander.sh
```

These scripts use relative executable paths such as:

```text
./../build/demo_FSI_*
./../build/demo_ROBOT_*
```

Therefore, with the scripts in their current form, launch them **from inside
`shell_scripts/`**:

```bash
cd shell_scripts
```

For the local Bash wheel runs:

```bash
./single_wheel_test.sh
./single_wheel_test_depth_var.sh
```

For the SLURM jobs:

```bash
sbatch cone_pentrometer.sh
sbatch normal_bevelmeter.sh
sbatch lander.sh
```

## 7.1 Cone penetrometer

`cone_pentrometer.sh` runs:

- one MCC case with `pre_pressure_scale=10`;
- four `MU_OF_I` cases with cohesion values
  `{0, 100, 1000, 5000}` Pa;
- `container_height=0.24 m`;
- `initial_spacing=0.001 m`.

## 7.2 Normal bevameter

`normal_bevelmeter.sh` sweeps:

```text
cohesion:
0, 100, 1000, 5000 Pa

container height:
0.024, 0.12, 0.24 m
```

for both MCC and `MU_OF_I`.

The MCC command explicitly passes:

```text
pre_pressure_scale = 20
```

The current shell script does not explicitly pass `kappa` or `lambda`; those
therefore come from the executable's internal/default configuration.

## 7.3 Constant-depth wheel test

`single_wheel_test.sh` currently sweeps:

```text
slope angle:
0, 2.5, 5, 10, 15, 20, 25 deg
```

with the active settings:

```text
rheology_model_crm = MCC
ps_freq            = 1
pre_pressure_scale = 2.0
kappa              = 0.2
lambda             = 0.8
```

Logs are written under:

```text
shell_scripts/viper_wheel_logs/
```

when launched from `shell_scripts/`.

## 7.4 Depth-variation wheel test

`single_wheel_test_depth_var.sh` runs both MCC and `MU_OF_I` for:

```text
depth:
0.1, 0.5, 1.0 m

slope:
0, 2.5, 5, 10, 15, 20, 25 deg
```

The current active sweep contains:

```text
2 rheologies x 3 depths x 7 slopes = 42 simulations
```

Logs are written under:

```text
shell_scripts/viper_wheel_depthVar_logs/
```

when launched from `shell_scripts/`.

## 7.5 Lander sweep

`lander.sh` is a SLURM array job with:

```text
pre_pressure_scale:
1.1, 2, 5, 10, 15, 20

kappa:
0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0

lambda / kappa:
4, 6, 10, 20
```

This produces:

```text
6 x 7 x 4 = 168 MCC parameter combinations
```

with up to eight array tasks running concurrently:

```text
#SBATCH --array=0-167%8
```

Array task 0 also runs one `MU_OF_I` baseline.

---


# 8. Plotting Scripts

Manuscript plotting utilities live in:

```text
plotting_scripts/
```

The curated plotting files are:

```text
plotting_scripts/
├── cone_pentrometer_single_plot.py
├── normal_bevelmeter_plot.py
├── single_wheel_test_plot.py
├── single_wheel_test_depth_var_plot.py
└── lander_plot.py
```

Most of these scripts were originally written against a build-tree context and
still use relative paths such as:

```text
./DEMO_OUTPUT
./DEMO_OUTPUT_NREL_LANDER
./paper_plots
```

For that reason, either:

1. run the plotting script with the working directory expected by the script; or
2. update/pass the input/output paths before running it.

## 9.1 Cone plot

`cone_pentrometer_single_plot.py` selects the desired cone-penetration runs and
generates the cone pressure-versus-depth comparison.

The current script imports:

```python
from penetrometer_folderParser import collect_experiments, extract_force_data
from penetrometer_plotting import plot_depth_vs_cone_pressure
```

Those helper modules must therefore also be available on `PYTHONPATH` or placed
beside the script. They are not represented by the five curated plotting files
listed above.

## 9.2 Normal bevameter plot

`normal_bevelmeter_plot.py` loads selected `MU_OF_I` and MCC cases and produces
the pressure-versus-sinkage comparison.

Its current output directory is:

```text
plate_penetration_plots/
```

## 9.3 Constant-depth wheel plot

`single_wheel_test_plot.py` computes slip from the simulated translational and
angular wheel motion and compares CRM results with experimental and DEM data.

Output is written under:

```text
paper_plots/
```

## 9.4 Depth-variation wheel plot

`single_wheel_test_depth_var_plot.py` supports:

```bash
--rheology both
--rheology mcc
--rheology mu_i
```

and can additionally enable the proximity-search sweep with:

```bash
--vary_ps
```

The currently selected depths are:

```text
0.1, 0.5, 1.0 m
```

## 9.5 Lander plot

`lander_plot.py` analyzes the MCC parameter sweep relative to the `MU_OF_I`
baseline.

Example:

```bash
python plotting_scripts/lander_plot.py \
    --input-dir <lander-output-directory> \
    --output-dir <analysis-output-directory>
```

The analysis generates:

```text
lander_grouped_profiles.png
lander_penetration_delta_heatmaps.png
lander_main_effects.png
lander_sweep_summary.csv
lander_sweep_trajectories.csv
analysis_notes.md
```

The lander sweep currently regenerates randomized terrain for individual runs
without recording the local initial surface height beneath each footpad. The
analysis therefore retains both nominal penetration and body settlement metrics;
body settlement is the cleaner cross-run metric when comparing the existing
randomized-terrain sweep.

---

# 9. Real-Time-Factor Benchmarks

The two benchmark sources are:

```text
demo_FSI_SlopedSingleWheel_RTFBenchmark.cpp
demo_ROBOT_Lander_CRM_RTFBenchmark.cpp
```

After building, the corresponding executables are:

```text
build/demo_FSI_SlopedSingleWheel_RTFBenchmark
build/demo_ROBOT_Lander_CRM_RTFBenchmark
```

These are intended for the paper's efficiency / wall-clock comparison rather
than the primary validation figures.

Example wheel benchmark:

```bash
./build/demo_FSI_SlopedSingleWheel_RTFBenchmark \
    --slope_angle=10 \
    --container_depth=0.1 \
    --total_time=0.5 \
    --initial_spacing=0.01 \
    --d0_multiplier=1.3 \
    --time_step=2.5e-4 \
    --rheology_model_crm=MCC \
    --pre_pressure_scale=2.0 \
    --kappa=0.2 \
    --lambda=0.8 \
    --no_vis
```

Example lander benchmark:

```bash
./build/demo_ROBOT_Lander_CRM_RTFBenchmark \
    --flat_terrain true \
    --rheology_model_crm MCC \
    --pre_pressure_scale 2.0 \
    --kappa 0.01 \
    --lambda 0.04 \
    --no_vis
```

---

# 10. Generated Data and Build Products

The following directories are generated or runtime-dependent and should not be
treated as source code:

```text
build/
DEMO_OUTPUT/
paper_plots/
*_logs/
```

The project-level `data/` entry is a symlink to the Chrono data directory and is
created by CMake.

Large CRM outputs, marker files, VTK data, snapshots, and rendered frames should
normally remain outside version control unless they are intentionally being
archived as paper artifacts.

---

# 11. Typical Workflow

From the project root:

```bash
# First build or after changing the Chrono configuration
./build_chrono_and_demos.sh

# Later, when Chrono is already built and only this project needs configuring/building
./build_chrono_and_demos.sh --chrono

# After editing only a demo source file
./rebuild_demos.sh
```

Run paper experiments from the shell-script directory:

```bash
cd shell_scripts

# Local/bash examples
./single_wheel_test.sh
./single_wheel_test_depth_var.sh

# SLURM examples
sbatch cone_pentrometer.sh
sbatch normal_bevelmeter.sh
sbatch lander.sh
```

Then run the relevant post-processing script using the generated output tree.

---

# 12. Troubleshooting

## Chrono package not found

Verify:

```text
../chrono/build/cmake/ChronoConfig.cmake
```

exists.

The current project is designed around a sibling Chrono checkout.

## VSG / Vulkan compile errors

If compilation fails with:

```text
fatal error: vulkan/vulkan.h: No such file or directory
```

CMake may have found the Vulkan library without propagating or locating the
Vulkan development headers. Verify that `vulkan/vulkan.h` is installed and that
the Vulkan include directory is visible to the targets using VSG.

## `nvcc` not found

Check:

```bash
nvidia-smi
nvcc --version
```

The NVIDIA driver and CUDA Toolkit are separate installations.


## Plotting script cannot find `DEMO_OUTPUT`

The plotting scripts retain build-tree-oriented relative paths. Check the current
working directory or update the script/input path accordingly.

## Cone plotting imports fail

Ensure the helper modules:

```text
penetrometer_folderParser.py
penetrometer_plotting.py
```

are available to Python.

## Lander cannot locate `generate_heightmap.py`

Confirm that:

```text
PAPER_mui_vs_mcc/generate_heightmap.py
```

exists and that the project has been configured with the current `CMakeLists.txt`,
which defines `DEMO_SOURCE_DIR`.

Flat-terrain runs can bypass heightmap generation using:

```text
--flat_terrain true
```
