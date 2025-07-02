# Code and Reproducibility for the CCTA Highway Control Problem
To conduct a reproducibility study and facilitate future research, we open-sourced our simulation experiment results and code. Here are the instructions to run the experiments.

## Install PyChrono as Simulation Engine 
You can install the pychrono engine on your local computer using two methods:

(1) Use the `environment.yml` file in this directory and follow these steps:
Create a conda environment and install all necessary packages and libraries as listed in the `environment.yml`:
```bash
conda env create -f environment.yml
```
Then activate the new environment, named chrono:
```bash
conda activate chrono
```
Download pychrono using this link: https://anaconda.org/projectchrono/pychrono/9.0.0/download/linux-64/pychrono-9.0.0-py310_4853.tar.bz2 and install the downloaded package in the chrono environment:
```bash
conda install <path_to_the_downloaded_file>.tar.bz2
```

(2) Official pychrono download links are available at: https://api.projectchrono.org/pychrono_installation.html. You will need to manually install the supporting packages such as: torch, casadi, scipy, pandas, etc. as suggested in the `environment.yml`.

## Run the Simulation Experiments
Two experiments described in the paper are presented as `exp1_straight_driving.py` and `exp2_complex_driving.py`. During our testing, a driving wheel was used as hardware. Considering that driving simulator setups might be limited for most users, we decided to set `enable_joystick=False` as the default. The following are the commands to run the experiments:
```bash
python exp1_straight_driving.py 
python exp2_complex_driving.py
```
The runtime visualization aims to achieve real-time and fast rendering for the simulation scene. We can post-process the experiments to obtain high-quality rendering for simulation experiments.

<!-- Display the two images side by side -->
<p align="center">
  <img src="data/render/3rd.png" alt="Straight-driving sample" width="66%"/>
  <img src="data/render/top_view.png" alt="Complex-driving sample" width="29%"/>
</p>

## Recorded Data for Experiments
For the experiment results presented in the paper, we stored and published them in the folder `expData_CCTA25`. There is also data plotting and analyzing code in the folder.