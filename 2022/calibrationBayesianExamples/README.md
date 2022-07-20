# Metadata for Bayesian inference demonstration 

This metadata is tied to how Bayesian inference can be used to calibrate an 8DOF vehicle model to make it behave like a Chrono::Vehicle HMMWV model as shown in the paper titled _Using a Bayesian-inference approach to calibrating models for simulation in robotics_.

## Table of contents -
- [Setup Guide](#setup)
  - [Project Chrono](#project-chrono)
  - [pyMC (4.0.0)](#pymc)
  - [arviz (0.12.0)](#arviz)
  - [jupyter](#jupyter)
- [Running](#running)
- [Support](#support)

## Setup

### Project Chrono
The data presented in the paper above utilizes the Chrono::Vehicle module as part of Project Chrono.  
__Note__ - Project Chrono only needs to be installed to generate the data used in the calibration effort. If you would like to only run the sampling and visulization scripts, you can skip this section as the data generated in the paper is already provided [here](https://github.com/uwsbel/public-metadata/tree/master/2022/calibrationBayesianExamples/data).  
Project Chrono can be built from source following the installation instructions given on the [Project Chrono website](https://api.projectchrono.org/development/tutorial_install_chrono_linux.html). Ensure to enable the Vehicle module in step 6.  

### pyMC
The sampling scripts use the Bayesian statistics python package _pyMC_ (4.0.0). Follow instructions on the pyMC [git repository](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)) to install. _Recommendation_ :  Use an [Anaconda](https://anaconda.org/) environment.  

### arviz
The visulizatiion notebooks leverage the Bayesian models exploratory analysis python package _arviz_. Instructions provided in the  _arviz_ [git repository](https://github.com/arviz-devs/arviz) can be used for installtion. _Recommendation_: Use an [Anaconda](https://anaconda.org/) environment.

### jupyter
The vizulization requires _jupyter notebook_ which can be installed using this [link](https://jupyter.org/install).

## Running
### Chrono::Vehicle model
In order to run the Chrono::Vehicle model, first place the folder [_calib\_mod_](https://github.com/uwsbel/public-metadata/tree/master/2022/calibrationBayesianExamples/chrono_model/calib_mod)- which contains the JSON files used to describe the vehicle and the txt files that define the driver inputs - in the subdirectory, _<path_to_chrono_build_directory>\data\vehicle_ on your local machine.
Then, overwrite the [_demo\_VEH\_WheeledJSON.cpp_](https://github.com/uwsbel/public-metadata/blob/master/2022/calibrationBayesianExamples/chrono_model/demo_VEH_WheeledJSON.cpp) in the subdirectory _chrono/src/demos/vehicle/wheeled_models_ in your local machine.  
Finally, move to the chrono build directory and build once again using -
```console
make
```
Then, run the calibration model using (from the chrono build directory)-
```console
cd bin
./demo_VEH_WheeledJSON 
```
### Sampling scripts
The sampling scripts made available are -  
**vd_8dof_st.py** - Sampling script for lateral dynamics.  
**vd_8dof_acc.py** - Sampling script for longitudinal dynamics.  
**vd_8dof_rr.py** - Sampling scrip for rolling resistance. 

These scripts can be run from the command line using -

```
python3 vd_8dof_st.py no_of_draws sampler_choice

```

Currently, there are three sampler choices - 'smc' : Sequential Monte Carlo, 'met' - Metropolis-Hastings and 'nuts' - NO U-Turn Sampler. 

Once the bayesian inference is complete, a netcdf file (.nc) will be saved in the results subdirectory along with a few log files. The naming convention is a date-time convention. This file contains information about the posterior distribution and additional statistics that can be used with the notebooks provided for analysing the posterior.

### Visulization notebooks
These jupyter notebooks can be run from the jupyer notebook app that can be launched after installation from the command line using -
```console
jupter notebook
```
The visulization notebooks present are -  
**neat_plots.ipynb** - Various chain plots - Posterior plot, Trace plot, Pairwise plot, ESS plot.  
**posteriors_acc.ipynb** - Plots the 100 responses of the 8 DOF model by drawing from the posterior distribution of the paramters calibrated during the longitudinal dynamics.  
**posteriors_st.ipynb** - Plots the 100 responses of the 8 DOF model by drawing from the posterior distribution of the paramters calibrated during the lateral dynamics.  
**posteriors_rr.ipynb** - Plots the 100 responses of the 8 DOF model by drawing from the posterior distribution of the paramters calibrated during the rolling resistance calibration.  
**posteriors_test.ipynb** - Plots the 100 responses of the 8 DOF model by drawing from the posterior distribution of the paramters calibrated during all the different stages of the calibration effort. These responses are compared to unseen "test" data.  

## Support
Contact [Huzaifa Mustafa Unjhawala](unjhawala@wisc.edu) for any questions or concerns regarding the contents of this folder.
