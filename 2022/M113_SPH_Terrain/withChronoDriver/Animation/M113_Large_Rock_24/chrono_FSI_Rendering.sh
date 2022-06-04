#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:0:0
#SBATCH --partition=research
##SBATCH --account=sbel
##SBATCH --qos=sbel_owner
#SBATCH --cpus-per-task=12

#SBATCH --output=./Slurm_Out/%j.out

#SBATCH --array=401-600%30

##SBATCH -w euler07

##module load gcc/7.3.0
##module load cmake/3.15.4
##module load cuda/10.1

declare -r id=$SLURM_ARRAY_TASK_ID

/srv/home/whu59/chrono_related_package/blender-2.91.0-linux64/blender --background --python ./bld_m113.py $id
