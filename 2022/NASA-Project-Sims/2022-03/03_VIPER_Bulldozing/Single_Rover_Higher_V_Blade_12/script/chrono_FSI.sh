#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-0:0:0
#SBATCH --partition=sbel
##SBATCH --account=sbel
#SBATCH --qos=priority
#SBATCH --output=/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1008/viper_sim/Slurm_Out/%j.out

#module load gcc/9.2.0
#module load cmake/3.18.1
#module load cuda/11.1

module load nvidia/cuda/11.3.1

mkdir ./DEMO_OUTPUT/FSI_VIPER/Rover_simple_wheel_12/script
cp demo_ROBOT_Viper_SPH.cpp \
demo_FSI_Viper_granular_NSC.json \
VIPER_Crater_SImulation.txt \
chrono_FSI.sh \
CMakeLists.txt \
blade.obj \
rock.obj \
./DEMO_OUTPUT/FSI_VIPER/Rover_simple_wheel_12/script/

./demo_ROBOT_Viper_SPH ./demo_FSI_Viper_granular_NSC.json