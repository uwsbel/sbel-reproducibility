# FNODE

This repository contains the code used to train and evaluate neural surrogates for multibody dynamics, centered on **FNODE** (Functional Neural ODE) and several benchmark models. Tthe codebase includes both the main benchmark pipelines and a number of analysis scripts for noise, symplectic rollouts, acceleration targets, convergence studies, control, and vehicle dynamics.

## What Is In This Repo

The current repository includes:

- Benchmark training for `FNODE`, `MBDNODE`, `FCNN`, and `LSTM`
- Reference energy-based models for simple systems: `HNN` and `LNN`
- Symplectic training and rollout variants for FNODE and MBDNODE
- Noisy-data training, acceleration-target comparison, and error-analysis scripts
- Slider-crank utilities, including friction-aware training and full-DOF reconstruction
- Controlled cart-pole training and MPC evaluation
- 4-DOF vehicle training, including a parameterized vehicle workflow

## Installation

Create a clean Python environment and install the dependencies:

```bash
git clone <your-fork-or-this-repo>
cd FNODE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda instead, the workflow is the same after activating the environment:

```bash
conda create -n fnode python=3.12
conda activate fnode
pip install -r requirements.txt
```

Core dependencies are listed in `[requirements.txt](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/requirements.txt)`, including `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `torchdiffeq`, `cvxpy`, `tqdm`, and `h5py`.

## Repository Layout

```text
FNODE/
├── Model/                    # core models, integrators, data generation, utilities
├── Model/veh_4dof/           # parameterized vehicle ROM and sampling tools
├── mpc_cartpole/             # MPC scripts for baseline and FNODE-controlled cart-pole
├── main_fnode.py             # main FNODE benchmark entry point
├── main_mbdnode.py           # MBDNODE baseline
├── main_fcnn.py              # FCNN baseline
├── main_lstm.py              # LSTM baseline
├── main_HNN.py               # Hamiltonian NN reference
├── main_LNN.py               # Lagrangian NN reference
├── main_fnode_symplectic.py  # FNODE-style symplectic training
├── main_mbdnode_symplectic.py
├── main_noise.py             # FNODE with noisy training data
├── main_accel.py             # FFT / FD / hybrid acceleration comparison
├── main_con.py               # controlled cart-pole FNODE_CON training
├── main_fnode_veh.py         # 4-DOF vehicle with control
├── main_fnode_veh_param.py   # parameterized 4-DOF vehicle
├── train_with_test_tracking.py
├── plot_testtracking.py
└── run_experiments.sh        # example experiment commands
```

Generated outputs are usually written to:

- `dataset/<test_case>/`
- `saved_model/<test_case>/`
- `results/<test_case>/<model_type>/`
- `figures/<test_case>/<model_type>/`
- `log/<test_case>/`

Some analysis scripts use custom output folders such as `results/dt_convergence`, `results/dt_accel_convergence`, or `results/error`.

## Main Benchmark Workflows

The main benchmark systems used by the primary scripts are:

- `Single_Mass_Spring`
- `Single_Mass_Spring_Damper`
- `Double_Pendulum`
- `Triple_Mass_Spring_Damper`
- `Slider_Crank`
- `Cart_Pole`

Most scripts will generate the dataset automatically if the expected CSV files are missing.

### FNODE

`[main_fnode.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode.py)` is the main entry point for the benchmark pipeline.

Examples:

```bash
python main_fnode.py --test_case Double_Pendulum
python main_fnode.py --test_case Slider_Crank --data_total_steps 4500 --train_ratio 0.3
python main_fnode.py --test_case Single_Mass_Spring_Damper --fnode_accel_mtd analytical
```

### Baselines

Use these scripts for model comparisons on the same benchmark systems:

- `[main_mbdnode.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_mbdnode.py)`
- `[main_fcnn.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fcnn.py)`
- `[main_lstm.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_lstm.py)`

Examples:

```bash
python main_mbdnode.py --test_case Double_Pendulum
python main_fcnn.py --test_case Cart_Pole
python main_lstm.py --test_case Single_Mass_Spring_Damper
```

### Reference Models

For the single-mass-spring family, the repository also includes:

- `[main_HNN.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_HNN.py)`
- `[main_LNN.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_LNN.py)`

## Extended Workflows

### Symplectic Training

These scripts train models with symplectic rollout logic:

- `[main_fnode_symplectic.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_symplectic.py)`
- `[main_mbdnode_symplectic.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_mbdnode_symplectic.py)`
- `[main_fnode_sms.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_sms.py)`

Example:

```bash
python main_fnode_symplectic.py --test_case Single_Mass_Spring --integrator yoshida4
```

### Noisy Data and Acceleration Targets

These scripts are useful when studying target generation rather than only end-to-end rollouts:

- `[main_noise.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_noise.py)`: FNODE training with AWGN or band-limited noise
- `[main_accel.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_accel.py)`: compare FD, FFT, and hybrid acceleration targets on saved training trajectories
- `[main_sc_accel.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_sc_accel.py)`: slider-crank acceleration learning study
- `[main_fnode_error.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_error.py)`: compare analytical, FD, and FFT/FD target pipelines on `Single_Mass_Spring_Damper`
- `[main_fnode_dt.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_dt.py)`: RK4 trajectory `dt` convergence study
- `[main_fnode_accel_dt.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_accel_dt.py)`: acceleration-only `dt` convergence study

### Test Tracking

To compare the training-time/test-MSE tradeoff across models:

- `[train_with_test_tracking.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/train_with_test_tracking.py)`
- `[plot_testtracking.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/plot_testtracking.py)`

Example:

```bash
python train_with_test_tracking.py --test_case Triple_Mass_Spring_Damper
python plot_testtracking.py --test_case Triple_Mass_Spring_Damper
```

### Slider-Crank Specific Utilities

- `[main_fnode_fric.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_fric.py)`: FNODE with a slider-crank friction parameter
- `[slider_crank_all_dof.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/slider_crank_all_dof.py)`: reconstruct and visualize full-DOF slider-crank motion

### Controlled Cart-Pole and MPC

Controlled-system workflows live in:

- `[main_con.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_con.py)`: train `FNODE_CON` on `Cart_Pole_Controlled` or `Cart_Pole_D_Controlled`
- `[mpc_cartpole/mpc_baseline.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/mpc_cartpole/mpc_baseline.py)`
- `[mpc_cartpole/mpc_fnode.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/mpc_cartpole/mpc_fnode.py)`
- `[mpc_cartpole/plot_results.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/mpc_cartpole/plot_results.py)`

Typical workflow:

```bash
python main_con.py --test_case Cart_Pole_Controlled
python mpc_cartpole/mpc_baseline.py
python mpc_cartpole/mpc_fnode.py
python mpc_cartpole/plot_results.py
```

### Vehicle Workflows

Vehicle-related scripts include:

- `[main_fnode_veh.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_veh.py)`: 4-DOF vehicle with control inputs
- `[main_fnode_veh_param.py](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/main_fnode_veh_param.py)`: parameterized vehicle training with sampled physical parameters
- `[Model/veh_4dof](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/Model/veh_4dof)`: ROM model, parameter samplers, and normalization utilities

Examples:

```bash
python main_fnode_veh.py --test_case veh_4dof
python main_fnode_veh_param.py
```

## Practical Notes

- There is no single global config file. Most options are defined directly in each script.
- Many scripts default to `cuda` when available, but can also be run with `--device cpu`.
- The repository already contains some generated `dataset`, `results`, `figures`, `saved_model`, and `log` folders in this working copy. They are experiment artifacts, not required source files.
- `[run_experiments.sh](/home/hongyu/Documents/sbel-reproducibility/2025/FNODE/run_experiments.sh)` is best treated as a collection of example commands, not the canonical description of the project.
- Some scripts are clearly research utilities rather than polished end-user entry points. Their arguments, defaults, and output locations may differ from the main benchmark scripts.

## Citation

If you use this repository, cite:

```bibtex
@article{wang2025fnode,
  title={FNODE: Flow-Matching for data-driven simulation of constrained multibody systems},
  author={Hongyu Wang and Jingquan Wang and Dan Negrut},
  year={2025},
  eprint={2509.00183},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.00183}
}
```

