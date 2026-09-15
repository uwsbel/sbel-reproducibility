# Experiment Shell Scripts

This folder contains the shell scripts used to run the simulation families associated with the paper results.

The scripts invoke executables from the project build directory using relative paths of the form:

```text
./../build/demo_FSI_*
./../build/demo_ROBOT_*
```

The cone, bevameter, and lander scripts are SLURM batch scripts configured for GPU execution. The two single-wheel scripts are conventional Bash scripts intended to be launched directly.

## Files

| Script                           | Paper section / figure family             | What it runs                                                                                                                                  |
| -------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `cone_pentrometer.sh`            | Cone Penetrometer Test                    | Runs `demo_FSI_ConePenetrometer` for one MCC case with `pre_pressure_scale=10` and a `MU_OF_I` cohesion sweep at a `0.24 m` container height. |
| `normal_bevelmeter.sh`           | Normal Bevameter Test                     | Runs `demo_FSI_NormalBevameter` for MCC and `MU_OF_I` across four cohesion values and three soil-bin heights.                                 |
| `single_wheel_test.sh`           | MGRU3 single-wheel constant-depth results | Runs `demo_FSI_SlopedSingleWheelTest` across seven terrain slopes for the current constant-depth MCC configuration.                           |
| `single_wheel_test_depth_var.sh` | MGRU3 depth-sensitivity results           | Runs `demo_FSI_SlopedSingleWheelTest` for both `MU_OF_I` and MCC across seven slopes and three container depths.                              |
| `lander.sh`                      | Lunar lander touchdown sensitivity study  | Runs the 168-case MCC SLURM parameter sweep and a single `MU_OF_I` baseline using `demo_ROBOT_Lander_CRM`.                                    |

## Script Details

### `cone_pentrometer.sh`

This is a SLURM batch script configured for one GPU on the `gpu-h100` partition.

The script loads:

```text
intel-oneapi-mkl
cuda/12.3
```

and runs:

```text
./../build/demo_FSI_ConePenetrometer
```

The MCC case uses:

```text
rheology_model_crm = MCC
pre_pressure_scale = 10
container_height   = 0.24 m
initial_spacing    = 0.001 m
```

It then runs a `MU_OF_I` cohesion sweep over:

```text
0
100
1000
5000
```

with the same:

```text
container_height = 0.24 m
initial_spacing  = 0.001 m
```

The script therefore performs:

```text
1 MCC run
4 MU_OF_I runs
-----------
5 total simulations
```

SLURM output is written to:

```text
cone10.out
cone10.err
```

---

### `normal_bevelmeter.sh`

This is a SLURM batch script configured for one GPU on the `gpu-h100` partition.

The script runs:

```text
./../build/demo_FSI_NormalBevameter
```

for both:

```text
MCC
MU_OF_I
```

The cohesion sweep is:

```text
0
100
1000
5000
```

and the soil-bin height sweep is:

```text
0.024 m
0.12 m
0.24 m
```

For MCC, the script additionally specifies:

```text
pre_pressure_scale = 20
```

For every combination of bin height and cohesion, the script launches one MCC case and one `MU_OF_I` case.

This gives:

```text
3 bin heights
× 4 cohesion values
× 2 rheology models
-------------------
24 total simulations
```

SLURM output is written to:

```text
bev20.out
bev20.err
```

Note that the current script does **not** explicitly pass `kappa` or `lambda` to `demo_FSI_NormalBevameter`; those values therefore come from the executable defaults or other internal configuration.

---

### `single_wheel_test.sh`

This script runs the constant-depth MGRU3 single-wheel simulation using:

```text
./../build/demo_FSI_SlopedSingleWheelTest
```

The current active slope sweep is:

```text
0°
2.5°
5°
10°
15°
20°
25°
```

The current active configuration is:

```text
rheology_model_crm = MCC
ps_freq            = 1

pre_pressure_scale = 2.0
kappa              = 0.2
lambda             = 0.8
```

Other fixed simulation parameters include:

```text
initial_spacing       = 0.01 m
d0_multiplier         = 1.3
time_step             = 2.5e-4 s
boundary_type         = adami
viscosity_type        = artificial_bilateral
kernel_type           = wendland
artificial_viscosity  = 0.1
total_mass            = 17.5
wheel_AngVel          = 0.8
gravity_G             = 9.81
grouser_height        = 0.03 m
```

Visualization is disabled with:

```text
--no_vis
```

The currently active arrays therefore produce:

```text
1 rheology
× 1 kappa
× 1 proximity-search frequency
× 7 slopes
-------------------------------
7 total simulations
```

Logs are written to:

```text
viper_wheel_logs/
```

with filenames containing the rheology, `kappa`, proximity-search frequency, slope, and simulation number.

There are commented-out alternatives in the script for larger parameter sweeps, but they are not part of the current active configuration.

---

### `single_wheel_test_depth_var.sh`

This script extends the single-wheel experiment by varying the soil-container depth.

It runs:

```text
./../build/demo_FSI_SlopedSingleWheelTest
```

for both rheology models:

```text
MU_OF_I
MCC
```

across slopes:

```text
0°
2.5°
5°
10°
15°
20°
25°
```

and depths:

```text
0.1 m
0.5 m
1.0 m
```

The currently active proximity-search configuration is:

```text
ps_freq = 1
```

The MCC parameters are:

```text
pre_pressure_scale = 2.0
kappa              = 0.2
lambda             = 0.8
```

The remaining wheel and CRM parameters are shared with the constant-depth script:

```text
initial_spacing       = 0.01 m
d0_multiplier         = 1.3
time_step             = 2.5e-4 s
boundary_type         = adami
viscosity_type        = artificial_bilateral
kernel_type           = wendland
artificial_viscosity  = 0.1
total_mass            = 17.5
wheel_AngVel          = 0.8
gravity_G             = 9.81
grouser_height        = 0.03 m
```

The active configuration produces:

```text
2 rheology models
× 1 kappa
× 1 proximity-search frequency
× 3 depths
× 7 slopes
-------------------------------
42 total simulations
```

Logs are written to:

```text
viper_wheel_depthVar_logs/
```

The simulation number resets for each depth so that the slope-to-simulation-number mapping remains consistent within each depth set.

---

### `lander.sh`

This is a SLURM array job for the lunar lander CRM parameter study.

Resource configuration:

```text
CPUs per task : 8
GPUs          : 1
memory        : 32 GB
time limit    : 48 hours
partition     : gpu-h100
```

The array is configured as:

```text
#SBATCH --array=0-167%8
```

so there are 168 MCC parameter combinations with at most eight array tasks running concurrently.

The parameter sweep is:

```text
pre_pressure_scale:
    1.1, 2, 5, 10, 15, 20

kappa:
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0

lambda / kappa:
    4, 6, 10, 20
```

For each run:

```text
lambda = lambda_multiplier × kappa
```

giving:

```text
6 pre-pressure scales
× 7 kappa values
× 4 lambda/kappa ratios
------------------------
168 MCC simulations
```

Array task `0` additionally runs one `MU_OF_I` baseline before its MCC simulation.

The executable is:

```text
./../build/demo_ROBOT_Lander_CRM
```

The baseline command is equivalent to:

```bash
./../build/demo_ROBOT_Lander_CRM \
    --rheology_model_crm MU_OF_I \
    --no_vis
```

Each MCC array task runs:

```bash
./../build/demo_ROBOT_Lander_CRM \
    --rheology_model_crm MCC \
    --kappa <kappa> \
    --lambda <lambda> \
    --pre_pressure_scale <pre_pressure_scale> \
    --no_vis
```

SLURM output follows:

```text
lander_<job-id>_<array-index>.out
lander_<job-id>_<array-index>.err
```

## Running the Scripts

The single-wheel scripts can be launched directly:

```bash
bash single_wheel_test.sh
```

or:

```bash
bash single_wheel_test_depth_var.sh
```

The SLURM scripts should be submitted with:

```bash
sbatch cone_pentrometer.sh
sbatch normal_bevelmeter.sh
sbatch lander.sh
```

All scripts assume that the corresponding simulation executable has already been built and is available under:

```text
../build/
```

relative to the directory from which the script is executed.

## Experiment Summary

| Experiment                   | Rheology models          | Primary sweep                        | Active simulation count |
| ---------------------------- | ------------------------ | ------------------------------------ | ----------------------: |
| Cone Penetrometer            | MCC, `MU_OF_I`           | Cohesion for `MU_OF_I`               |                       5 |
| Normal Bevameter             | MCC, `MU_OF_I`           | Cohesion × bin height                |                      24 |
| Single Wheel                 | MCC                      | Slope                                |                       7 |
| Single Wheel Depth Variation | MCC, `MU_OF_I`           | Depth × slope                        |                      42 |
| Lander                       | MCC + `MU_OF_I` baseline | OCR proxy × `kappa` × `lambda/kappa` |    168 MCC + 1 baseline |

## Notes

* These scripts correspond to the experiment families retained for the manuscript results.
* The cone, bevameter, and lander scripts are configured for the `gpu-h100` SLURM partition and load CUDA 12.3.
* `single_wheel_test.sh` currently runs **only MCC**, despite containing comments and commented-out arrays from earlier, broader sweeps.
* `single_wheel_test_depth_var.sh` currently runs both MCC and `MU_OF_I` for depths `0.1`, `0.5`, and `1.0 m`.
* `normal_bevelmeter.sh` currently supplies `pre_pressure_scale=20` for MCC but does not explicitly pass `kappa` or `lambda`.
* `lander.sh` still invokes `demo_ROBOT_Lander_CRM`. If the intended manuscript experiment should instead use a different restored lander executable, update the `MU_OF_I` and MCC commands together so that the baseline and sweep remain consistent.
* The filenames `cone_pentrometer.sh` and `normal_bevelmeter.sh` are retained as they currently appear in this directory.
