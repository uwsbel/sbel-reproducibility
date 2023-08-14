## Metadata for paper titled _"An Expeditious and Expressive Vehicle Model for Machine Learning, Control, and Estimation"_
This folder contains the Vehicle Model (VM), the calibration scripts and data and the plotting scripts (which use the data in the calibration folder).

### Building the vehicle model on Linux systems 
#### C++
For building the vehicle model cd into the VM folder and then build using cmake
```bash
cd VM
cmake .
make
```
This will generate an executable called `model` which can be run as `./model`
#### Python
For the python version of the VM, we use a swig wrapper. To build this follow the below instructions
```bash
cd VM/interfaces
cmake .
make
```
This will generate the shared library `_rom.so`. The path to this library then needs to be added into the `.bashrc`/`.zshrc` file as  
`export PYTHONPATH=$PYTHONPATH:<path_to_Unjhawala-IEEE-ExressiveVM>/VM/interface`
The library can then be called from any python script anywhere on your computer

### Running the calibration scripts
The calibration scripts used to calibrate the VM to ART and to the Chrono HMMWV simulation can be found in the `calibration` folder. Before these can be run, you must first install [pymc using conda](https://www.pymc.io/projects/docs/en/stable/installation.html) and build the python wrapped version of the VM using the instructions from above. To run the calibration scripts, shell scripts are provided which can be run as follows
#### ART Longitudinal Dynamics calibration
```bash
cd calibration/ART
sh ART_acc_calib_run.sh
```
#### ART Lateral Dynamics calibration
```bash
cd calibration/ART
sh ART_st_calib_run.sh
```
#### HMMWV calibration
```bash
cd calibration/HMMMWV
sh HMMWV_calib_run.sh
```

### Running plotting scripts
The `plotting` folder consists of scripts that can be run to generate the plots in the paper.

