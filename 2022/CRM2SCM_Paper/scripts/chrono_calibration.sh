#!/usr/bin/env bash
##SBATCH --gres=gpu:a100:1
#SBATCH --time=5-0:0:0
#SBATCH --partition=sbel
##SBATCH --account=sbel
##SBATCH --qos=sbel_owner
##SBATCH --qos=priority
#SBATCH --output=./slurm_out/%j.out

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5

##SBATCH --array=0-9
##SBATCH -w euler129

##declare -r id=$SLURM_ARRAY_TASK_ID



# export MKL_NUM_THREADS=16
# export OMP_NUM_THREADS=16
module load mamba
bootstrap_conda
conda activate pymc3_env
# python3 wheelCalAllParaNoCoheSepSave.py
# python3 wheelCalibrationNormal.py
# python3 wheelCalibrationFrictional.py

# python3 BevameterCalibrationNormal_3para.py
python3 BevameterCalibrationFrictional_2para.py
# python3 BevameterCalibrationFrictional_1para.py