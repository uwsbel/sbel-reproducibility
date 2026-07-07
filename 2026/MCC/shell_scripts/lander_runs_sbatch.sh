#!/bin/bash

#SBATCH --job-name=lander_crm_sweep
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-h100
#SBATCH --array=0-167%8
#SBATCH -o lander_%A_%a.out
#SBATCH -e lander_%A_%a.err

module load intel-oneapi-mkl
module load cuda/12.3

# Parameter arrays (must match original script)
pre_pressure_scales=(1.1 2 5 10 15 20)
kappas=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
lambda_multipliers=(4 6 10 20)

# Calculate array sizes
num_pre_pressure=${#pre_pressure_scales[@]}
num_kappas=${#kappas[@]}
num_lambda_mults=${#lambda_multipliers[@]}

# Calculate total number of simulations
total_sims=$((num_pre_pressure * num_kappas * num_lambda_mults))

# Calculate parameter indices from SLURM_ARRAY_TASK_ID
# Order: pre_pressure_scale (outer), kappa (middle), lambda_mult (inner)
lambda_mult_idx=$((SLURM_ARRAY_TASK_ID % num_lambda_mults))
kappa_idx=$(((SLURM_ARRAY_TASK_ID / num_lambda_mults) % num_kappas))
pre_pressure_idx=$((SLURM_ARRAY_TASK_ID / (num_lambda_mults * num_kappas)))

# Get parameter values
pre_pressure_scale=${pre_pressure_scales[$pre_pressure_idx]}
kappa=${kappas[$kappa_idx]}
lambda_mult=${lambda_multipliers[$lambda_mult_idx]}

# Calculate lambda = multiplier * kappa
lambda=$(echo "$lambda_mult * $kappa" | bc -l)

# Display job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Simulation $((SLURM_ARRAY_TASK_ID + 1))/$total_sims"
echo "pre_pressure_scale: $pre_pressure_scale"
echo "kappa: $kappa"
echo "lambda: $lambda (multiplier: $lambda_mult)"
echo "=========================================="

# Run the MU_OF_I simulation first (only for task 0)
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Running MU_OF_I baseline simulation..."
    ./../build/demo_ROBOT_Lander_CRM --rheology_model_crm MU_OF_I --no_vis
    if [ $? -ne 0 ]; then
        echo "ERROR: MU_OF_I simulation failed"
    fi
    echo ""
fi

# Run the MCC simulation with current parameters
echo "Running MCC simulation with parameters:"
echo "  kappa=$kappa, lambda=$lambda, pre_pressure_scale=$pre_pressure_scale"
./../build/demo_ROBOT_Lander_CRM --rheology_model_crm MCC --kappa "$kappa" --lambda "$lambda" --pre_pressure_scale "$pre_pressure_scale" --no_vis

# Check if simulation failed
if [ $? -ne 0 ]; then
    echo "ERROR: Simulation failed for parameters:"
    echo "  kappa=$kappa, lambda=$lambda, pre_pressure_scale=$pre_pressure_scale"
    exit 1
fi

echo "Simulation completed successfully!"
echo "=========================================="