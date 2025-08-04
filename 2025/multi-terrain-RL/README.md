# Multi-terrains Reinforcement Learning (RL) using PyChrono Engine

This folder contains code for training / finetune RL policies with pychrono support. We only made one demostration case, a locomotion policy training with unitree GO2 model. The policy is firstly trained in rigid terrain environment (95% done with env: `chrono_env.py`, policy checkpoint:`data/rl_models/rslrl/model_2000.pt`), then finetuned the policy in granular terrain environment (5% done with env: `chrono_crmenv.py`, policy checkpoint:`data/rl_models/rslrl/model_2999.pt`). Rigid terrain suppose to be faster than CRM granular terrain, but also physics with lower fidelity and less diverse. 

## Installation
### Install PyChrono via Conda

```bash
conda create -n chrono "python<3.13" -c conda-forge
conda activate chrono
conda install bochengzou::pychrono -c bochengzou -c nvidia -c dlr-sc -c conda-forge
```
### Install RL Library
Inside the conda environment, supposingly named as `chrono`:
```bash
pip install rsl-rl-lib==2.2.4
```
### Verify Installation
Inside the conda environment:
```bash
cd <path_to_current_folder>/multi-terrain-RL
python rl_examples/rslrl/eval.py --ckpt 2999
```