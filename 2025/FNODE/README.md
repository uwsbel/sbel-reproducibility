# FNODE

FNODE is a framework for learning and evaluating neural surrogates for multibody dynamics. The repository centers on **Functional Neural ODEs (FNODE)**, with comparison models and analysis tools for constrained mechanical systems, noisy training data, symplectic rollouts, controlled systems, and parameterized vehicle dynamics.

The code currently includes:

- Core benchmark training for `FNODE`, `MBDNODE`, `FCNN`, and `LSTM`
- Reference energy-based models for the single-mass-spring family (`HNN`, `LNN`)
- Symplectic training and rollout variants
- Noisy-data experiments and acceleration-target studies
- Slider-crank friction and full-DOF reconstruction utilities
- Controlled cart-pole training and MPC evaluation
- 4-DOF vehicle and parameterized vehicle experiments

## Installation

```bash
git clone https://github.com/Hongyu0329/FNODE.git
cd FNODE
conda create -n fnode python=3.12
conda activate fnode
pip install -r requirements.txt
```

If you prefer Conda, create any Python environment that can install the packages in `[requirements.txt](requirements.txt)`.

## Core Benchmark Systems

The main benchmark scripts work on the following systems:

- `Single_Mass_Spring`
- `Single_Mass_Spring_Damper`
- `Double_Pendulum`
- `Triple_Mass_Spring_Damper`
- `Slider_Crank`
- `Cart_Pole`
- `4dof vehicle`

Most training scripts generate datasets automatically on first run and write them under `dataset/<test_case>/`.

## Main Training Scripts

### FNODE

`[main_fnode.py](main_fnode.py)` is the primary entry point for FNODE training and evaluation. It learns state-to-acceleration mappings, saves checkpoints, rolls out trajectories, and writes comparison plots.

Example:

```bash
python main_fnode.py --test_case Double_Pendulum
python main_fnode.py --test_case Slider_Crank --data_total_steps 4500 --train_ratio 0.3
```

### Baselines

Use the following scripts for model comparisons on the same benchmark systems:

- `[main_mbdnode.py](main_mbdnode.py)`
- `[main_fcnn.py](main_fcnn.py)`
- `[main_lstm.py](main_lstm.py)`

Examples:

```bash
python main_mbdnode.py --test_case Double_Pendulum
python main_fcnn.py --test_case Cart_Pole
python main_lstm.py --test_case Single_Mass_Spring_Damper
```

### Reference Models

The repo also includes reference implementations for the single-mass-spring family:

- `[main_HNN.py](main_HNN.py)`
- `[main_LNN.py](main_LNN.py)`

## Specialized Workflows

### Symplectic Training

- `[main_fnode_symplectic.py](main_fnode_symplectic.py)`: FNODE-style Hamiltonian-gradient training with symplectic rollout loss
- `[main_mbdnode_symplectic.py](main_mbdnode_symplectic.py)`: MBDNODE variant with symplectic integrators
- `[main_fnode_sms.py](main_fnode_sms.py)`: single-mass-spring focused FNODE symplectic workflow

Example:

```bash
python main_fnode_symplectic.py --test_case Single_Mass_Spring --integrator yoshida4
```

### Noisy Data and Acceleration Studies

- `[main_noise.py](main_noise.py)`: train FNODE with AWGN or band-limited noise
- `[main_accel.py](main_accel.py)`: compare FD and FFT-based acceleration targets
- `[main_sc_accel.py](main_sc_accel.py)`: slider-crank acceleration analysis

### Test-Tracking Comparisons

- `[train_with_test_tracking.py](train_with_test_tracking.py)`: train `MBDNODE`, `FNODE`, `LSTM`, and `FCNN` while logging test MSE during training
- `[plot_testtracking.py](plot_testtracking.py)`: plot the saved tracking CSV files

Example:

```bash
python train_with_test_tracking.py --test_case Triple_Mass_Spring_Damper
python plot_testtracking.py --test_case Triple_Mass_Spring_Damper
```

### Slider-Crank Utilities

- `[slider_crank_all_dof.py](slider_crank_all_dof.py)`: reconstruct and visualize the full-DOF slider-crank motion
- `[main_fnode_fric.py](main_fnode_fric.py)`: train FNODE on slider-crank data with friction as an extra parameter

### Controlled Systems and MPC

- `[main_con.py](main_con.py)`: train `FNODE_CON` for controlled cart-pole systems
- `[mpc_cartpole/mpc_baseline.py](mpc_cartpole/mpc_baseline.py)`: baseline MPC evaluation
- `[mpc_cartpole/mpc_fnode.py](mpc_cartpole/mpc_fnode.py)`: MPC using a trained FNODE surrogate
- `[mpc_cartpole/plot_results.py](mpc_cartpole/plot_results.py)`: plot MPC trajectories and controls

Typical workflow:

```bash
python main_con.py --test_case Cart_Pole_Controlled
python mpc_cartpole/mpc_baseline.py
python mpc_cartpole/mpc_fnode.py
python mpc_cartpole/plot_results.py
```

### Vehicle Experiments

- `[main_fnode_veh.py](main_fnode_veh.py)`: 4-DOF vehicle dynamics with control inputs
- `[main_fnode_veh_param.py](main_fnode_veh_param.py)`: parameterized 4-DOF vehicle training with sampled physical parameters
- `[Model/veh_4dof/](Model/veh_4dof)`: parameter sampler, ROM model, and normalization utilities used by the vehicle workflows

## Output Layout

Most scripts write to a consistent directory structure:

- `dataset/<test_case>/`: generated datasets
- `saved_model/<test_case>/`: model checkpoints
- `results/<test_case>/<model_type>/`: rollouts, CSV metrics, and saved arrays
- `figures/<test_case>/<model_type>/`: trajectory, phase-space, and comparison plots
- `log/<test_case>/`: run logs

The helper `Model/utils.py` builds these paths for most benchmark scripts.

## Repository Layout

```text
FNODE/
├── Model/                  # data generation, force functions, models, integrators, utilities
├── Model/veh_4dof/         # parameterized vehicle tools
├── mpc_cartpole/           # MPC scripts using baseline and FNODE dynamics
├── figures/                # standalone plotting helpers
├── main_fnode.py           # primary FNODE benchmark training script
├── main_mbdnode.py         # MBDNODE benchmark training
├── main_fcnn.py            # FCNN baseline
├── main_lstm.py            # LSTM baseline
├── main_noise.py           # noisy-data FNODE experiments
├── main_con.py             # controlled cart-pole training
└── train_with_test_tracking.py
```

## Notes

- The command-line flags are defined directly in each script; there is no single shared config file.
- Several scripts are research utilities for specific studies, so defaults and output formats can differ slightly between workflows.
- `[run_experiments.sh](run_experiments.sh)` contains a collection of experiment commands, but it should be treated as a convenience script rather than the authoritative description of the repository.

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

