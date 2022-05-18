#!/usr/bin/env bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10-0:0:0
#SBATCH --partition=sbel
#SBATCH --account=sbel
#SBATCH --qos=sbel_owner
#SBATCH --cpus-per-task=4

module load gcc/9.2.0
module load cmake/3.18.1
module load cuda/11.1

./demo_FSI_Curiosity_granular_NSC demo_FSI_Curiosity_granular.json 1 0.5 1.0
