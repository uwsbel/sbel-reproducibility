# Introduction

This document links to the source files, run scripts, and other resources for the paper "Co-Design of Rover Wheels and Control using Bayesian Optimization and Rover-Terrain Simulations".

---

# Dependencies

## Build-time

| Dependency | Version |
|------------|---------|
| CMake | ≥3.26.5 |
| GCC toolchain | ≥11.0 |
| CUDA Toolkit | ≥12.3 (with `nvcc`) |
| SWIG | ≥4.0.2 |
| Python (Anaconda) | ≥3.11.4 |
| Eigen3 | ≥3.4.0 |
| Vulkan | loader + headers for VSG |

## Python Runtime

| Package |
|---------|
| `numpy` |
| `six` |

---

# Build PyChrono with CRM

To run and reproduce all the data in this paper, you will need to build Project Chrono from an internal repository with Python bindings enabled.

### 1. Clone the repository

```bash
git clone https://github.com/uwsbel/chrono-wisc.git
```

### 2. Check out the wheel optimization branch

```bash
git checkout feature/wheel-optimization
```

### 3. Create a build folder

```bash
cd chrono-wisc
mkdir build
cd build
```

### 4. Configure the build with CMake

Use the following recommended CMake options:

| Option | Value |
|--------|-------|
| Build type | Release |
| Shared libs | ON |
| Demos | ON |
| Tests/benchmarks | OFF |
| Modules | FSI, Postprocess, Python, Vehicle, VSG |
| OpenMP | ON (Bullet and Eigen OpenMP ON) |
| SIMD | ON |
| FSI double precision | OFF |
| Bullet double precision | OFF |
| CUDA | Set your [CUDA architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) based on available hardware |
| Python | Your Python interpreter |
| SWIG | Your SWIG executable |

> **Note:** Enabling the VSG module requires installing VSG dependencies. Follow the [Project Chrono installation guide](https://api.projectchrono.org/module_vsg_installation.html), but use the `feature/wheel-optimization` branch's build script.

### 5. Build the project

```bash
make -j 8
```

### 6. Set Python environment variables

```bash
export PYTHONPATH=$PYTHONPATH:<path_to_chrono-wisc>/build/bin/
```

### 7. Create Bayesian Optimization Python environment

Use the provided `requirements.txt` file to create a Python environment. First, create a venv:

```bash
python -m venv bo_venv
source bo_venv/bin/activate
pip install -r requirements.txt
```

---

# What Do the Scripts Do?

## Launch Scripts

Launch shell scripts used to run Bayesian optimization on the NCSA cluster (via NSF ACCESS) are in the [reproduce_scripts](https://github.com/uwsbel/chrono-wisc/tree/feature/wheel_optimization/reproduce_scripts) folder. These scripts also provide the arguments with which the Python scripts were run. See the README in that folder for usage instructions.

| Script | Description |
|--------|-------------|
| `launch_pull.sh` | Launches wheel optimization for pulling a fixed load along a straight path |
| `launch_ssl_bo_join.sh` | Launches joint optimization of wheel and steering controller parameters over the sine curved trajectory while pulling a fixed load |
| `launch_ssl_bo_wheelOnly.sh` | Launches wheel-only optimization (steering controller parameters fixed) over the sine curved trajectory while pulling a fixed load |
| `launch_ssl_bo_stControllerOnly.sh` | Launches steering controller optimization (wheel parameters fixed from previous step) over the sine curved trajectory while pulling a fixed load |

## Python Scripts

All Python scripts used for the process (including experimentation and trials) can be found in the [python vehicle demos folder](https://github.com/uwsbel/chrono-wisc/tree/feature/wheel_optimization/src/demos/python/vehicle). The scripts that run Bayesian Optimization for each case in the paper are referenced by the launch scripts above. Additional scripts used for plotting and analysis are listed below.

Before running the Bayesian optimization scripts, please download the marker files from the [SimulationData](https://uwmadison.box.com/s/2wfm5fblz1ntdc54swtdkm1vl1s4w1i9) folder and place them in the `python vehicle demos` folder.

### Plotting / Analysis Reports

Before any of the plotting/analysis scripts can be run, make sure to run the Bayesian Optimization or download the data from [Box](https://uwmadison.box.com/s/vsi8hw3e3jicrb79mg8evk71fnvatuzn).

| Script | Description |
|--------|-------------|
| `analyze_bo_run.py` | Analyzes BO trials (`trials.csv` + `failed_trials.jsonl`) with convergence, correlations, trends, identifiability, ICE/PDP, and GP-like CV plots |
| `extended_analysis_bo.py` | Extended BO post-analysis: data cleaning, metric boxplots/histograms, corner plots, clustering, and PCA |
| `global_sens.py` | Builds a surrogate model and computes Sobol (Saltelli/Jansen) sensitivity indices |
| `global_sens_single.py` | Sobol sensitivity for single-wheel slalom BO trials using a surrogate model |
| `global_sens_single_newParams.py` | Builds a surrogate model and computes Sobol (Saltelli/Jansen) sensitivity indices |
| `global_sens_stControllerOnly.py` | Sobol sensitivity for steering-controller-only BO runs; includes optional ICE/PDP plots |

### Testing Optimized Controller and Wheels

After Bayesian Optimization, the best controller and wheels can be tested on the same sine maneuver using the `run_best_wheel_and_controller.py` script. For instance, to test the controller and wheels obtained using the joint optimization approach, run the following command (data folders from [Box](https://uwmadison.box.com/s/vsi8hw3e3jicrb79mg8evk71fnvatuzn)):

```bash
python run_best_wheel_and_controller.py --joint-csv SineSideSlip_wControl_Dec22/trials_wControl.csv --snapshots
```

### Testing Optimized Controller and Wheels on a Race Track

To test the performance of the controller and the wheels, we have some additional scripts.

First, generate the race track centerline:

```bash
python racetrack_gen.py --length 10 --max-waypoints 10 --seed 10 --min-turn-radius 0.6 --output track_ctrl.csv --output-centerline convex_racetrack.csv --output-centerline-ds 0.1
```

Then, use the centerline to generate SPH and BCE markers for the simulation. The SPH and BCE marker files can be found in the [SimulationData](https://uwmadison.box.com/s/2wfm5fblz1ntdc54swtdkm1vl1s4w1i9) folder within `out`.

```bash
python racetrack_gen.py --input-waypoints convex_racetrack.csv --markers --width 1.2 --delta 0.005 --output-dir out --centerline-resolution 0.02 --no-plot
```

These SPH and BCE marker files must be placed in `race-track-gen/out/`. Then use the `race_best_wheel_and_controller.py` script to run the simulation. For instance, to run the optimization results for the best wheel and controller obtained through the looped optimization approach, run the following command (data folders from [Box](https://uwmadison.box.com/s/vsi8hw3e3jicrb79mg8evk71fnvatuzn)):

```bash
python race_best_wheel_and_controller.py --separate SineSideSlip_global_1700_0_0.6_25_3_5_24Dec --controller-csv SineSideSlip_global_1700_0_0.6_25_3_5_24Dec/trials_stPIDControllerOnly_0.0_0.0_0.0_1.0.csv --snapshots
```
