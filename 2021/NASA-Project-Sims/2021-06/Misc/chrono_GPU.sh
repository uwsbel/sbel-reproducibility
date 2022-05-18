#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=3-0:0:0
#SBATCH --partition=sbel
#SBATCH --account=sbel
#SBATCH --qos=sbel_owner
##SBATCH --qos=priority

##SBATCH --nodes=1
##SBATCH --tasks-per-node=2
##SBATCH --cpus-per-task=2

##SBATCH --array=0-9
##SBATCH -w euler04

##declare -r id=$SLURM_ARRAY_TASK_ID

module load gcc/9.2.0
module load cmake/3.18.1
module load cuda/11.1
module load glfw/3.3.2
module load intel/mkl/2019_U2
module load openmpi/4.0.2

./demo_GPU_ballcosim demo_GPU_ballcosim.json
