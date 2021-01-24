#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=10-0:0:0
#SBATCH --partition=sbel
#SBATCH --account=sbel
#SBATCH --qos=sbel_owner

##module load gcc/7.3.0
##module load cmake/3.15.4
##module load cuda/10.1

/srv/home/whu59/research/chrono_related_package/blender-2.91.0-linux64/blender --background --python ./bld_test.py
