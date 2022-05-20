#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-0:0:0
#SBATCH --partition=sbel
##SBATCH --account=sbel
##SBATCH --qos=priority
##SBATCH --cpus-per-task=1
#SBATCH --output=./Slurm_Out/%j.out

#module load gcc/9.2.0
#module load cmake/3.18.1
#module load cuda/11.1

module load nvidia/cuda/11.3.1

mkdir ./DEMO_OUTPUT/FSI_M113/M113_05/script

cp demo_FSI_m113_granular.cpp \
demo_FSI_m113_granular_NSC.json \
M113_Simulation.txt \
chrono_FSI.sh \
CMakeLists.txt \
./DEMO_OUTPUT/FSI_M113/M113_05/script/

./demo_FSI_m113_granular ./demo_FSI_m113_granular_NSC.json
