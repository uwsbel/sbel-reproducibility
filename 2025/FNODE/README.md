<<<<<<< HEAD
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
=======
# The "sbel-reproducibility" repo
The repo contains assets, obj files, json files, scripts, Chrono models, etc., needed to reproduce results reported in papers, tech reports, presentations, etc.

**This repo is organized by year of when the metadata was first added to the repo. Each year shows as a folder in this repo.**

## Info for SBEL members
- Subfolders are meant to contain information associated with a certain paper, tech report, thesis, etc. Feel free to use subfolders in subfolders.
- The naming convention is as follows:
  - If it's a technical report you are dealing with, then call the sub-folder TR-2023-04
  - If it's a paper, then call the subfolder, for instance, Unjhawala-IEEE-LibraryOfROM-vehicles. In other words, last name of first author, the journal (or conference where published), and a couple of words tied to the title or topic of the paper
  - If it's a thesis, then call the subfolder, for instance, Elmquist-PhD. In other words, last name of the person who defended the thesis, and the type of thesis. It can be PhD, MS, or IndepThesis
- Please include a readme.md file in the high level subfolder to describe the content of the directroy along with where the material was used (which paper/tech report/etc.).
- **Important:** If the metadata is meant to be used in conjunction with Chrono, please provide a commit ID/SHA1 - hash value that the results reported in your paper/tech report/etc. were generated with
- Please avoid dropping large files here, particularly so if they're non-ascii. For large files that are available elsewhere, provide a link (in your readme.md) that can be used to download the file. This applies, for instance, to large pics, movies, etc.
- When adding data, scripts, assets, models, etc., please take the long view. The metadata that you provide will likely be used for years to come. 
- For SBEL members, a style issue: When referencing in your manuscript this metadata repo, say for TR-2020-02, please define in SBEL's **BibFiles** repo, under **refsSBELspecific.bib**, an entry like this:

  *@misc{TR-2020-02metadata, \
  author = {Hu, Wei and Serban, Radu and Negrut, Dan}, \
  title = {{TR-2020-02 Public Metadata}}, \
  note              = {{Simulation-Based Engineering Laboratory, University of Wisconsin-Madison}}, \
  year              = {2020}, \
  howpublished      = {\url{https://github.com/uwsbel/public-metadata/tree/master/2020/TR-2020-02}} \
  }*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To maintain consistency, please copy/paste/edit the sample above to fit your needs when dropping in **refsSBELspecific.bib**.
>>>>>>> 7c11e5200424b8ef3ee0fc1529b9dceefd3ed278
