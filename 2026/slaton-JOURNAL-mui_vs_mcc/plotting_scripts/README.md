# Plotting Scripts

This folder contains the plotting and analysis scripts used to generate figures for the paper results section.

The scripts operate directly on simulation output directories and, in most cases, still assume a build-tree-style directory layout. The expected input and output locations differ slightly between scripts and are summarized below.

## Files

| Script                                | Paper section / figure family             | Description                                                                                                                                                                                                                            |
| ------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cone_pentrometer_single_plot.py`     | Cone Penetrometer Test                    | Selects a cone-penetration simulation configuration from `./DEMO_OUTPUT/FSI_ConePenetrometer_GRC1` and generates the cone pressure-versus-depth comparison plot.                                                                       |
| `normal_bevelmeter_plot.py`           | Normal Bevameter Test                     | Loads selected `MU_OF_I` and MCC normal-bevameter runs and generates the pressure-sinkage comparison plot.                                                                                                                             |
| `single_wheel_test_plot.py`           | MGRU3 single-wheel constant-depth results | Computes wheel slip from the single-wheel simulations and plots slip ratio versus terrain slope for `MU_OF_I`, MCC, experimental data, and DEM reference data.                                                                         |
| `single_wheel_test_depth_var_plot.py` | MGRU3 depth-sensitivity results           | Computes and compares slip-versus-slope behavior at multiple wheel penetration depths. Can generate separate MCC and `MU_OF_I` figures as well as a combined comparison.                                                               |
| `lander_plot.py`                      | Lunar lander touchdown sensitivity study  | Analyzes the MCC lander parameter sweep against a `MU_OF_I` baseline and generates grouped penetration histories, penetration-difference heat maps, and main-effect plots. It also writes processed summary tables and analysis notes. |

## Script Details

### `cone_pentrometer_single_plot.py`

Expected input directory:

```text
./DEMO_OUTPUT/FSI_ConePenetrometer_GRC1
```

The script defines a parameter query for the desired simulation runs, extracts the associated force data, and generates the cone pressure-versus-penetration-depth plot.

This script is not fully standalone. It imports:

```text
penetrometer_folderParser.py
penetrometer_plotting.py
```

These helper modules must therefore be available on the Python import path when the script is executed.

---

### `normal_bevelmeter_plot.py`

Expected input directory:

```text
<directory containing script>/DEMO_OUTPUT
```

Specific `MU_OF_I` and MCC simulation configurations are selected through the `RUN_CONFIGS` list near the top of the script.

The script reads `force_vs_time.txt` from each selected run and converts the prescribed pressure ramp into a pressure-versus-penetration curve.

Output:

```text
plate_penetration_plots/
└── penetration_vs_time.png
```

Although the output filename retains `penetration_vs_time`, the plotted axes are pressure versus penetration.

---

### `single_wheel_test_plot.py`

Expected input directory:

```text
./DEMO_OUTPUT/FSI_SlopedSingleWheelTest
```

The script compares constant-depth single-wheel simulations using:

* `MU_OF_I`
* MCC
* experimental MGRU3 data
* DEM reference data

Wheel slip is computed from the average translational and angular wheel velocities:

```text
slip = 1 - v / (omega * R)
```

By default, the script uses proximity-search frequency `PS=1`.

To compare multiple proximity-search frequencies:

```bash
python single_wheel_test_plot.py --vary_ps
```

Figures are written to:

```text
./paper_plots/
```

with filenames beginning with:

```text
Viper_slip_vs_slope_rheology_comparison
```

---

### `single_wheel_test_depth_var_plot.py`

Expected input directory:

```text
./DEMO_OUTPUT/FSI_SlopedSingleWheelTest
```

This script extends the single-wheel comparison to multiple penetration depths.

The currently selected depths are:

```text
0.1 m
0.5 m
1.0 m
```

These can be changed through the `PLOT_DEPTHS` list near the top of the script.

The rheology plotted can be selected with:

```bash
python single_wheel_test_depth_var_plot.py --rheology both
python single_wheel_test_depth_var_plot.py --rheology mcc
python single_wheel_test_depth_var_plot.py --rheology mu_i
```

Multiple proximity-search frequencies can additionally be enabled with:

```bash
python single_wheel_test_depth_var_plot.py --vary_ps
```

Depending on the `--rheology` option, the script can generate:

```text
./paper_plots/Viper_slip_vs_slope_depth_comparison_mcc_only.png
./paper_plots/Viper_slip_vs_slope_depth_comparison_mu_i_only.png
./paper_plots/Viper_slip_vs_slope_depth_comparison.png
```

When `--vary_ps` is enabled, `_ps_vary` is appended to the corresponding filename.

---

### `lander_plot.py`

Default input directory:

```text
build_new/bin/DEMO_OUTPUT_NREL_LANDER
```

A different sweep directory can be specified with:

```bash
python lander_plot.py --input-dir <path>
```

The script compares the MCC parameter sweep against the `MU_OF_I` baseline. The MCC sweep is analyzed with respect to:

* pre-pressure scale / OCR proxy
* `kappa`
* `lambda / kappa`

The default plotted quantity is nominal penetration depth. Body settlement can instead be selected with:

```bash
python lander_plot.py --metric settlement_m
```

The script generates:

```text
lander_grouped_profiles.png
lander_penetration_delta_heatmaps.png
lander_main_effects.png
```

It also writes the processed analysis data:

```text
lander_sweep_summary.csv
lander_sweep_trajectories.csv
analysis_notes.md
```

By default, these files are written to:

```text
<input-dir>/analysis
```

A different output directory can be selected with:

```bash
python lander_plot.py \
    --input-dir <input-path> \
    --output-dir <output-path>
```

Runs can optionally be filtered by maximum peak nominal penetration:

```bash
python lander_plot.py --max-peak-penetration <meters>
```

### Lander analysis caveat

The existing sweep regenerates a random terrain height map and rock field for each run, but the terrain seed and local initial surface height beneath each footpad are not recorded.

As a result, the script reports absolute penetration relative to the nominal reference terrain surface rather than the exact local terrain height. Body settlement is also retained as a cleaner cross-run metric.

## Directory Expectations

The scripts are not yet fully location-independent.

Current path conventions are:

```text
cone_pentrometer_single_plot.py
    -> ./DEMO_OUTPUT/FSI_ConePenetrometer_GRC1

normal_bevelmeter_plot.py
    -> <script directory>/DEMO_OUTPUT

single_wheel_test_plot.py
    -> ./DEMO_OUTPUT/FSI_SlopedSingleWheelTest
    -> ./paper_plots

single_wheel_test_depth_var_plot.py
    -> ./DEMO_OUTPUT/FSI_SlopedSingleWheelTest
    -> ./paper_plots

lander_plot.py
    -> build_new/bin/DEMO_OUTPUT_NREL_LANDER
       unless overridden with --input-dir
```

Because several of these are relative paths, the wheel and cone scripts should currently be launched from the directory layout for which they were written.

## Dependencies

The plotting scripts use combinations of:

```text
numpy
pandas
matplotlib
seaborn
```

The cone-penetrometer script additionally requires the local helper modules:

```text
penetrometer_folderParser
penetrometer_plotting
```

## Notes

* This directory is intended to contain only plotting and post-processing code associated with manuscript results.
* Simulation executables and general diagnostic plotting utilities are not included here.
* The wheel plotting scripts compute slip directly from the recorded wheel trajectory rather than relying on a precomputed slip value.
* The depth-variation wheel script can generate MCC-only, `MU_OF_I`-only, and combined figures from the same simulation dataset.
* The lander script performs both plotting and sweep post-processing, including generation of summary CSV files and analysis notes.
