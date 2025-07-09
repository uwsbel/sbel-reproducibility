# Introduction
This document links to the source files, run scripts and other resources for the paper "A Physics-Based Continuum Model for Versatile, Scalable, and
Fast Terramechanics Simulation" submitted and under review at the Journal of Terramechanics.

If you are interested in just using Project Chrono and the Chrono::CRM solver, please directly clone and build the main branch of the official [Project Chrono repository](https://github.com/projectchrono/chrono).

# Prerequisites
To run and reproduce all the data in this paper, you will need to build Project Chrono from a internal repository on three different branches:

1) [testing/baseline](https://github.com/uwsbel/chrono-wisc/tree/testing/baseline) - To run speed analysis tests of Chrono's baseline SPH solver (Chrono version 9.0.1).
2) [testing/benchmark3](https://github.com/uwsbel/chrono-wisc/tree/testing/benchmark3) - To run speed analysis tests of Chrono::CRM.
3) [testing/fsi-CRMAcc](https://github.com/uwsbel/chrono-wisc/tree/testing/fsi-CRMAcc) - To run the validation tests discussed in the paper.

The build instructions are the same as those for [Project Chrono](https://api.projectchrono.org/tutorial_install_chrono.html). Make sure you build each of these in a separate directory for ease.  

Please build with FSI, PARDISO_MKL, VEHICLE and VSG enabled. For FSI you will require CUDA 12.0 or higher. For PARDISO_MKL you will require MKL 2024.0 which can be installed from Intel's [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit/download.html). VSG is not necessary but recommended for cool viz. You can build with VSG by following instructions from Project Chrono's [VSG install instructions](https://api.projectchrono.org/module_vsg_installation.html).  

Please create an issue in this repository and tag @Huzaifg for any build related issues.

# Baseline Speed Benchmark
To run the baseline benchmark, run [benchmarkSmall.sh](https://github.com/uwsbel/chrono-wisc/blob/testing/baseline/benchmarkShellScripts/benchmarkSmall.sh) by placing it in the `<build_dir>/bin` directory. The results will be saved in the `<build_dir>/bin/BENCHMARK_BASELINE_RTF` directory. Keep this directory around if you want to plot the results.

# Chrono::CRM Speed Benchmark
For running the speed benchmarks comparing the latest code with SCM and the baseline, run [benchmarkSmall.sh](https://github.com/uwsbel/chrono-wisc/blob/testing/benchmark3/shell_scripts/benchmarkSmall.sh) by placing it in the `<build_dir>/bin` directory. The results with Active Domains enabled will be saved in `<build_dir>/bin/BENCHMARK3_RTF_Active` directory and the ones without Active Domains will be saved in `<build_dir>/bin/BENCHMARK3_RTF_noActive` directory. Keep these directories around if you want to plot the results.

To run the rigid and flexible tires with the rigid and deformable terrain, run [benchmark_demo_runs.sh](https://github.com/uwsbel/chrono-wisc/blob/testing/benchmark3/shell_scripts/benchmark_demo_runs.sh) by placing it in the `<build_dir>/bin` directory. The results will be saved in the `<build_dir>/bin/DEMO_OUTPUT` in different folders. 

To run the terrain scaling results, run [run_terrain_benchmark.sh](https://github.com/uwsbel/chrono-wisc/blob/testing/benchmark3/shell_scripts/run_terrain_benchmark.sh) by placing it in the `<build_dir>/bin` directory. The results will be saved in the `<build_dir>/bin/benchmark_results_CRM` within a csv file.

# Chrono::CRM Validation Tests
To run the validation tests, run for each demo case the scripts within [shell_scripts](https://github.com/uwsbel/chrono-wisc/tree/testing/fsi-CRMAcc/shell_scripts) from the `<build_dir>/bin` directory. 

# Plotting
For plotting, each of the branches have folders called `python_scripts` which contain the plotting scripts for various different plots from the paper.
