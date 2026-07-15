#!/bin/bash

# Script to run the FSI Viper Wheel benchmark with depth variation
# Keeps the same parameter variations as viper_wheel_paperRuns.sh and also varies container depth.

# Default parameters
INITIAL_SPACING=0.01
D0_MULTIPLIER=1.3
TIME_STEP=2.5e-4
BOUNDARY_TYPE="adami"
VISCOSITY_TYPE="artificial_bilateral"
KERNEL_TYPE="wendland"
ARTIFICIAL_VISCOSITY=0.1
TOTAL_MASS=17.5
WHEEL_ANGVEL=0.8
GRAVITY_G=9.81
GROUSER_HEIGHT=0.03

# Define arrays for the parameters we're testing
SLOPE_ANGLES=(0 2.5 5 10 15 20 25)
PS_FREQS=(1)
RHEOLOGY_MODEL_CRM_VALUES=("MU_OF_I" "MCC")
# DEPTH_VALUES=(0.1 0.2 0.4 0.6 0.8 1.0)
DEPTH_VALUES=(0.1 0.5 1.0)

# MCC parameters (only used when rheology_model_crm="MCC")
PRE_PRESSURE_SCALE=2.0
KAPPA_VALUES=(0.2)
LAMBDA=0.8

# Directory containing the executable
EXEC="./../build/demo_FSI_SlopedSingleWheelTest"

# Check if executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable not found at $EXEC"
    echo "Please navigate to the directory containing the executable or update the EXEC variable."
    exit 1
fi

# Create a log directory
LOG_DIR="viper_wheel_depthVar_logs"
mkdir -p "$LOG_DIR"

# Counter for total simulations
total_sims=0

# Loop through all parameter combinations
for rheology_model_crm in "${RHEOLOGY_MODEL_CRM_VALUES[@]}"; do
    for kappa in "${KAPPA_VALUES[@]}"; do
        for ps_freq in "${PS_FREQS[@]}"; do
            for depth in "${DEPTH_VALUES[@]}"; do
                # Keep slope index mapping consistent inside each depth set
                sim_number=1

                for slope_angle in "${SLOPE_ANGLES[@]}"; do
                    echo "==================================================="
                    echo "Running simulation with:"
                    echo "  rheology_model_crm = $rheology_model_crm"
                    echo "  kappa = $kappa"
                    echo "  ps_freq = $ps_freq"
                    echo "  depth = $depth"
                    echo "  slope_angle = $slope_angle"
                    echo "  sim_number = $sim_number"
                    echo "==================================================="

                    CMD="$EXEC --ps_freq=$ps_freq --initial_spacing=$INITIAL_SPACING --d0_multiplier=$D0_MULTIPLIER --time_step=$TIME_STEP --boundary_type=$BOUNDARY_TYPE --viscosity_type=$VISCOSITY_TYPE --kernel_type=$KERNEL_TYPE --artificial_viscosity=$ARTIFICIAL_VISCOSITY --total_mass=$TOTAL_MASS --slope_angle=$slope_angle --wheel_AngVel=$WHEEL_ANGVEL --gravity_G=$GRAVITY_G --grouser_height=$GROUSER_HEIGHT --container_depth=$depth --sim_number=$sim_number --rheology_model_crm=$rheology_model_crm --no_vis"

                    if [[ "$rheology_model_crm" == "MCC" ]]; then
                        CMD="$CMD --pre_pressure_scale=$PRE_PRESSURE_SCALE --kappa=$kappa --lambda=$LAMBDA"
                    fi

                    LOG_FILE="$LOG_DIR/sim_${rheology_model_crm}_kappa${kappa}_ps${ps_freq}_depth${depth}_slope${slope_angle}_sim${sim_number}.log"
                    echo "Executing: $CMD"
                    echo "Logging to: $LOG_FILE"

                    eval "$CMD" > "$LOG_FILE" 2>&1

                    if [ $? -eq 0 ]; then
                        echo "Simulation completed successfully."
                    else
                        echo "Simulation failed! Check log file: $LOG_FILE"
                    fi

                    ((sim_number++))
                    ((total_sims++))

                    echo ""
                done
            done
        done
    done
done

echo "All simulations completed"
echo "Total simulations run: $total_sims"