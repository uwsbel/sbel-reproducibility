#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-0:0:0
#SBATCH --partition=research
##SBATCH --account=sbel
##SBATCH --qos=priority

#module load gcc/9.2.0
#module load cmake/3.18.1
#module load cuda/11.1

module load nvidia/cuda/11.3.1

../../chrono-dev-build/bin/demo_FSI_RoverSingleWheelGrouser demo_FSI_RoverSingleWheelGrouser_granular.json 0.0
