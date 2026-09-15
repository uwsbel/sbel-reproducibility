#!/bin/bash
#SBATCH --job-name=bev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -o bev20.out
#SBATCH -e bev20.err
#SBATCH --partition=gpu-h100
module load intel-oneapi-mkl
module load cuda/12.3

cohesions=(0 100 1000 5000)
# Bin heights in meters: 2.4 cm, 12 cm, 24 cm
bin_heights=(0.024 0.12 0.24)

for bin_height in "${bin_heights[@]}"; do
    for cohesion in "${cohesions[@]}"; do
        ./../build/demo_FSI_NormalBevameter --rheology_model_crm MCC --pre_pressure_scale 20 --cohesion "$cohesion" --container_height "$bin_height"
        ./../build/demo_FSI_NormalBevameter --rheology_model_crm MU_OF_I --cohesion "$cohesion" --container_height "$bin_height"
    done
done