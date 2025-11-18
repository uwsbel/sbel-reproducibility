# FNODE: Functional Neural ODE Framework for Multibody Dynamics

## Overview

This repository provides the implementation of the **FNODE** framework for data-driven modeling of constrained multibody systems, which directly learns **acceleration vector fields** from trajectory data, avoiding the computational bottleneck of backpropagating through an ODE solver.

## Environment Setup

```bash
git clone https://github.com/Hongyu0329/FNODE.git  
cd FNODE 
conda create -n fnode python=3.11 && conda activate fnode
pip install -r requirements.txt  
```

## Training Models

Each model has its own driver script:

```bash
python main_fnode.py  
python main_pnode.py  
python main_fcnn.py  
python main_lstm.py  
```

You can configure dataset paths, hyperparameters, and solver settings directly in each file.

Or you can use shell script for reproduce the experiments

```bash
python run_experiments.sh
```

## Visualization (in /figures folder)

All visualization scripts that generates comparison plots are in the `figures/` directory.

## Track the Test MSE in Training

Two scripts support test-trajectory tracking and visualization:

- **train_all_models_with_test_tracking.py**
  Runs all models and logs test-set MSE curve during training.
- **plot_testtracking.py**
  Plots prediction-vs-ground-truth trajectories, MSE curves, and other evaluation metrics.

```bash
python train_all_models_with_test_tracking.py  
python plot_testtracking.py  
```

Results are saved automatically into `figures/`.

## Slider-Crank Full DOF Recover

```bash
python slider_crank_all_dof.py
```

Recover the Degree of Freedom (DOF) reduction and produce full DOF figures for slider crank

## Model Predictive Control (MPC)

All MPC-related logic is consolidated under /mpc_cartpole:
First, train the fnode model for cart pole mpc

```bash
python main_con.py 
```

Second, run the baseline code and test code

```bash
python mpc_cartpole/mpc_baseline.py
python mpc_cartpole/mpc_fnode.py
```

Remember that when run `mpc_fnode.py`, make sure the path for the `.pkl` model is set properly

Finally, visualize the results:

```bash
python mpc_cartpole/polt_results.py
```

## Citation

```bash
@misc{wang2025fnode,
      title={FNODE: Flow-Matching for data-driven simulation of constrained multibody systems}, 
      author={Hongyu Wang and Jingquan Wang and Dan Negrut},
      year={2025},
      eprint={2509.00183},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.00183}, 
}
```
