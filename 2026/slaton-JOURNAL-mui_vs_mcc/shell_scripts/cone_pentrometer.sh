#!/bin/bash
#SBATCH --job-name=cone
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -o cone10.out
#SBATCH -e cone10.err
#SBATCH --partition=gpu-h100
module load intel-oneapi-mkl
module load cuda/12.3

# Heights in meters for 2.4 cm, 5.8 cm, 11.6 cm, 23.2 cm, 46.4 cm 
cohesions=(0 100 1000 5000)
./../build/demo_FSI_ConePenetrometer --rheology_model_crm MCC --pre_pressure_scale 10 --container_height 0.24 --initial_spacing 0.001

for cohesion in "${cohesions[@]}"; do
	./../build/demo_FSI_ConePenetrometer --rheology_model_crm MU_OF_I --cohesion $cohesion --container_height 0.24 --initial_spacing 0.001
done