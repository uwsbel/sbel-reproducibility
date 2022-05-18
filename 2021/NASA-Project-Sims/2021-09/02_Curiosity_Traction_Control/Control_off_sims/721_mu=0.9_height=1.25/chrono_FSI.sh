#!/usr/bin/env bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=5-0:0:0
#SBATCH --partition=sbel
##SBATCH --account=sbel
##SBATCH --qos=priority

#module load gcc/9.2.0
#module load cmake/3.18.1
#module load cuda/11.1

module load nvidia/cuda/11.3.1

../../chrono-dev-build/bin/demo_FSI_Curiosity_granular_NSC_control demo_FSI_Curiosity_granular.json 1 0.5 0 0.5 1.5 0.025 0.0
