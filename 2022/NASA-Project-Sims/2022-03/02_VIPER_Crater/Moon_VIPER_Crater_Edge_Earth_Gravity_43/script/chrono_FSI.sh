#!/usr/bin/env bash
#SBATCH --gres=gpu:1
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

mkdir ./DEMO_OUTPUT/FSI_VIPER/Rover_rock_43/script
cp demo_ROBOT_Viper_SPH.cpp \
demo_FSI_Viper_granular_NSC.json \
VIPER_Rock_Simulation.txt \
chrono_FSI.sh \
CMakeLists.txt \
blade.obj \
plate.obj \
rock.obj \
rock1.obj \
rock2.obj \
rock3.obj \
./DEMO_OUTPUT/FSI_VIPER/Rover_rock_43/script/

./demo_ROBOT_Viper_SPH ./demo_FSI_Viper_granular_NSC.json