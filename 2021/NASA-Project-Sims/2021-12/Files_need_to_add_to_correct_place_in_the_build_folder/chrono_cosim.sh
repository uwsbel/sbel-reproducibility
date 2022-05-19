#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-0:0:0
#SBATCH --partition=research
##SBATCH --account=sbel
##SBATCH --qos=sbel_owner
##SBATCH --qos=priority

#SBATCH --nodes=1
#SBATCH --tasks-per-node=3
#SBATCH --cpus-per-task=2

##SBATCH --array=0-9
##SBATCH -w euler44

##declare -r id=$SLURM_ARRAY_TASK_ID

#module load gcc/9.2.0
#module load cmake/3.18.1
module load nvidia/cuda/11.3.1
#module load glfw/3.3.2
#module load intel/mkl/2019_U2
module load mpi/openmpi/4.1.1
#module load blaze/3.8

mpirun -np 3 ./demo_VEH_Cosim_WheelRig \
--terrain_specfile="../data/vehicle/cosim/terrain/granular_gpu_10mm_grouser.json" \
--tire_specfile="../data/vehicle/hmmwv/tire/Curiosity_wheel_grouser.json" \
--sim_time=40.0 \
--settling_time=0.01 \
--step_size=0.000025 \
--base_vel=0.192 \
--total_mass=18.5 \
--actuation_type=SET_ANG_VEL \
--output_fps=20 \
--render_fps=20 \
--threads_tire=1 \
--threads_terrain=1 \
--slip=0.0 \
--suffix="_coh=2.0_slip=0.0"
##mpirun -np 2 ../../chrono-cosim-build/bin/demo_VEH_Cosim_WheelRig $id
