#!/usr/bin/env bash
#SBATCH --gres=gpu:p100:1
#SBATCH --time=10-0:0:0
##SBATCH --partition=sbel
##SBATCH --account=sbel
##SBATCH --qos=sbel_owner

module load gcc/7.3.0
module load cmake/3.15.4
module load cuda/10.1

../../chrono-dev-build-gauss/bin/demo_FSI_Rover_granular_NSC demo_FSI_Rover_granular.json


