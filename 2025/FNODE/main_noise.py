#!/usr/bin/env python3
"""
Refactored main_noise.py - FNODE training with noisy data
Consistent with main_fnode.py structure while preserving noise-specific functionality

Author: FNODE Noise Training Pipeline (Refactored)
Date: 2025-01-12
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import logging
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

# --- Simplified Logging Configuration (from main_fnode.py) ---
def setup_logging(log_file_path=None):
    """Setup logging with optional file output, avoiding duplicates"""
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file_path:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )


# Initial basic logging setup
setup_logging()
logger = logging.getLogger("FNODE_Noise_Main")

# --- Ensure Model directory is in the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    sys.path.insert(0, model_package_dir)

# --- Imports from custom Model package ---
try:
    from Model.Data_generator import generate_dataset, generate_analytical_sms
    from Model.utils import (
        set_seed, get_output_paths, save_data_pd, save_data, save_model_state, load_model_state,
        plot_acceleration_comparison, plot_trajectory_comparison, calculate_model_parameters,
        generate_target_accelerations,
        add_high_frequency_noise, add_awgn, calculate_fft_target_derivative, trim_method_2_robust_residual
    )
    from Model.model import FNODE, test_fnode, train_fnode_with_csv_targets
    from Model.force_fun import (
        force_sms, calculate_analytical_accelerations_sms,
        calculate_analytical_accelerations_smsd, calculate_analytical_accelerations_tmsd,
        calculate_analytical_accelerations
    )
    from Model.integrator import (
        sep_stormer_verlet_multiple_body, yoshida4_multiple_body,
        fukushima6_multiple_body, runge_kutta_four_multiple_body
    )

except ImportError as e:
    logger.error(f"Failed to import from Model package/directory: {e}. Check PYTHONPATH.", exc_info=True)
    sys.exit(1)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="FNODE Training with Noisy Data (Refactored)")

    # Test case and data generation
    parser.add_argument('--test_case', type=str, default='Single_Mass_Spring_Damper',
                       choices=['Single_Mass_Spring', 'Single_Mass_Spring_Damper', 'Double_Pendulum',
                               'Triple_Mass_Spring_Damper', 'Slider_Crank', 'Cart_Pole'],
                       help="Dynamical system to simulate.")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Computation device.")
    parser.add_argument('--generate_new_data', action='store_true', default=True,
                       help="Flag to generate new dataset.")
    parser.add_argument('--data_dt', type=float, default=0.01, help="Time step for data generation.")
    parser.add_argument('--data_total_steps', type=int, default=3000,
                       help="Total steps for trajectory (train+test).")

    # Noise-specific parameters
    parser.add_argument('--add_noise', action='store_true', default=True,
                       help="Enable noise addition to training data")
    parser.add_argument('--noise_type', type=str, default='band_limited',
                       choices=['awgn', 'band_limited'],
                       help="Type of noise to add: 'awgn' or 'band_limited'")
    parser.add_argument('--noise_level', type=float, default=0.03,
                       help="Noise level relative to signal std (0.05 = 5%)")
    parser.add_argument('--freq_low', type=float, default=0.6,
                       help="Lower bound of noise frequency band (fraction of Nyquist, for band_limited noise)")
    parser.add_argument('--freq_high', type=float, default=0.95,
                       help="Upper bound of noise frequency band (fraction of Nyquist, for band_limited noise)")

    # FFT parameters
    parser.add_argument('--prob', type=int, default=50,
                       help="Probability factor for FFT truncation. Used to calculate trunc = train_time_step//prob")
    parser.add_argument('--fft_smooth_factor', type=int, default=48,
                       help="FFT smoothing factor - set to 24 as validated (train_steps // this)")
    parser.add_argument(
        '--fnode_accel_mtd',
        type=str,
        default='fd',
        choices=['fft', 'fd', 'analytical'],
        help="How to generate FNODE acceleration targets when using standard (non-noisy) targets.",
    )

    # Gibbs trimming parameters
    parser.add_argument('--use_gibbs_trim', action='store_true', default=True,
                       help="Apply Method2 Gibbs trimming")
    parser.add_argument('--trim_smooth_window', type=int, default=21,
                       help="Smoothing window for Method2")
    parser.add_argument('--trim_mad_k', type=float, default=6.0,
                       help="MAD multiplier for threshold")
    parser.add_argument('--trim_stable_run', type=int, default=12,
                       help="Required consecutive stable samples")
    parser.add_argument('--trim_max_frac', type=float, default=0.25,
                       help="Maximum fraction to trim from each side")
    parser.add_argument('--trim_min_keep_frac', type=float, default=0.5,
                       help="Minimum fraction of signal to keep")

    # Model architecture
    parser.add_argument('--layers', type=int, default=2, help="Number of layers for FNODE.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden layer width for FNODE.")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'],
                       help="Activation function.")
    parser.add_argument('--initializer', type=str, default='xavier', choices=['xavier', 'kaiming'],
                       help="Weight initializer.")
    parser.add_argument('--d_interest', type=int, default=0,
                       help="Extra input features for FNODE. MUST BE 0 for state -> acceleration mapping.")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=450, help="Number of training epochs.")
    parser.add_argument('--early_stop', type=str, default='False',
                       help="Enable early stopping ('True' or 'False').")
    parser.add_argument('--patience', type=int, default=50,
                       help="Patience for early stopping (number of epochs without improvement).")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                       help="Optimizer.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="Weight decay for regularization.")
    parser.add_argument('--lr_scheduler', type=str, default='ExponentialLR',
                       choices=['StepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None'],
                       help="Learning rate scheduler.")
    parser.add_argument('--grad_clip_value', type=float, default=1.0,
                       help="Gradient clipping value for handling stability.")
    parser.add_argument('--save_ckpt_freq', type=int, default=500,
                       help="Frequency (in epochs) to save model checkpoints.")

    # Training control
    parser.add_argument('--skip_train', action='store_true', default=False,
                       help="Skip training and only perform evaluation.")
    parser.add_argument('--train_ratio', type=float, default=0.15,
                       help="Ratio of data to use for training (0.1 means 10%)")
    parser.add_argument('--out_time_log', type=str, default='False',
                       help="Output detailed timing logs during training.")
    parser.add_argument('--num_workers', type=int, default=0,
                       help="Number of workers for data loading.")

    # Loss and target parameters
    parser.add_argument('--fnode_loss_type', type=str, default='L2',
                       choices=['L1', 'L2', 'Smooth_L1', 'Huber'],
                       help="Loss function type for FNODE.")

    # Evaluation parameters
    parser.add_argument('--ode_method', type=str, default='stormer_verlet',
                       choices=['euler', 'midpoint', 'rk4', 'dopri5',
                               'stormer_verlet', 'yoshida4', 'fukushima6'],
                       help="ODE solver method for testing. Symplectic methods: stormer_verlet, yoshida4, fukushima6")
    parser.add_argument('--test_interval', type=int, default=10,
                       help="Test evaluation interval during training")

    # Symplectic integrator parameters
    parser.add_argument('--symplectic_use_analytic_dh', action='store_true', default=False,
                       help="Use analytical Hamiltonian gradients for symplectic integrators (Single_Mass_Spring only)")
    parser.add_argument('--mass_value', type=float, default=10.0,
                       help="Mass value for symplectic integrators (default: 10.0)")

    # Model type for compatibility
    parser.add_argument('--model_type', type=str, default='FNODE',
                       help="Model type (fixed to FNODE)")

    # Training batch size
    parser.add_argument('--batch_size', type=int, default=64,
                       help="Batch size for training. Larger values (32-128) provide more stable gradients. "
                            "Small values (1-8) add noise but may cause energy drift in physics simulations.")

    # Gradient accumulation for small batch sizes
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help="Number of steps to accumulate gradients before optimizer step. "
                            "Effective batch size = batch_size * gradient_accumulation_steps. "
                            "Use this when batch_size=1 to stabilize training.")

    # Enhanced gradient clipping for small batch training
    parser.add_argument('--use_adaptive_grad_clip', action='store_true', default=False,
                       help="Use adaptive gradient clipping based on gradient statistics (recommended for batch_size=1)")

    # Momentum adjustment for small batch
    parser.add_argument('--momentum_beta', type=float, default=0.9,
                       help="Momentum coefficient for Adam optimizer (beta1). "
                            "Increase to 0.95-0.99 for batch_size=1 to smooth gradients.")

    return parser.parse_args()


# --- Noise-specific data preparation ---
def prepare_noisy_data(s_train, t_train, args, dataset_dir):
    """
    Add noise to training data and compute acceleration targets.
    For Single_Mass_Spring_Damper and Triple_Mass_Spring_Damper, uses analytical accelerations.
    For Single_Mass_Spring, uses FFT-based acceleration calculation.

    Args:
        s_train: Clean training trajectory tensor [steps, 2*num_bodies]
        t_train: Time tensor [steps]
        args: Command line arguments
        dataset_dir: Dataset directory to save CSV files

    Returns:
        s_train_noisy: Noisy trajectory (possibly trimmed at both ends)
        t_train_noisy: Corresponding time (possibly trimmed at both ends)
        target_csv_path: Path to saved acceleration targets
        trim_info: Dictionary with trimming information
    """
    # Ensure we're working with the right device
    device = torch.device(args.device)

    # Extract velocities (assuming [x1, v1, x2, v2, ...] format)
    num_states = s_train.shape[1]
    if num_states % 2 != 0:
        raise ValueError(f"State dimension {num_states} is not even (expected [pos, vel] pairs)")

    num_bodies = num_states // 2
    trim_info = {'applied': False}

    # For Single Mass Spring, state is [position, velocity]
    positions = s_train[:, 0::2]  # Extract all positions
    velocities = s_train[:, 1::2]  # Extract all velocities

    # Add noise to positions and velocities
    if args.add_noise:
        logger.info(f"Adding {args.noise_type.upper()} noise with level={args.noise_level}")
        positions_noisy = []
        velocities_noisy = []

        for body_idx in range(num_bodies):
            pos_clean = positions[:, body_idx]
            vel_clean = velocities[:, body_idx]

            # Add noise to each body's state based on noise type
            if args.noise_type == 'awgn':
                # Add AWGN noise
                pos_noisy = add_awgn(
                    pos_clean,
                    noise_level=args.noise_level,
                    seed=args.seed + body_idx * 2
                )
                vel_noisy = add_awgn(
                    vel_clean,
                    noise_level=args.noise_level,
                    seed=args.seed + body_idx * 2 + 1
                )
            else:  # band_limited
                # Add band-limited high-frequency noise
                pos_noisy = add_high_frequency_noise(
                    pos_clean, args.data_dt,
                    noise_level=args.noise_level,
                    freq_band=(args.freq_low, args.freq_high),
                    seed=args.seed + body_idx * 2
                )
                vel_noisy = add_high_frequency_noise(
                    vel_clean, args.data_dt,
                    noise_level=args.noise_level,
                    freq_band=(args.freq_low, args.freq_high),
                    seed=args.seed + body_idx * 2 + 1
                )

            positions_noisy.append(pos_noisy)
            velocities_noisy.append(vel_noisy)

        # Stack back into state format
        s_train_noisy = torch.zeros_like(s_train)
        for body_idx in range(num_bodies):
            s_train_noisy[:, body_idx*2] = positions_noisy[body_idx]
            s_train_noisy[:, body_idx*2 + 1] = velocities_noisy[body_idx]

        # Calculate SNR
        signal_power = torch.mean(s_train ** 2)
        noise_power = torch.mean((s_train_noisy - s_train) ** 2)
        snr_db = 10 * torch.log10(signal_power / noise_power)
        logger.info(f"{args.noise_type.upper()} SNR: {snr_db:.2f} dB")
    else:
        s_train_noisy = s_train

    # Compute acceleration targets based on test case
    # For Single_Mass_Spring, Single_Mass_Spring_Damper and Triple_Mass_Spring_Damper: use analytical accelerations
    # All three test cases must use analytical results as requested by user

    use_analytical = args.test_case in ['Single_Mass_Spring', 'Single_Mass_Spring_Damper', 'Triple_Mass_Spring_Damper']

    if use_analytical:
        logger.info(f"Using analytical accelerations for {args.test_case}")

        # Calculate analytical accelerations based on noisy trajectory
        if args.test_case == 'Single_Mass_Spring':
            analytical_accels = calculate_analytical_accelerations_sms(
                s_train_noisy.cpu().numpy(),
                t_train.cpu().numpy(),
                args.test_case,
                dataset_dir
            )
        elif args.test_case == 'Single_Mass_Spring_Damper':
            analytical_accels = calculate_analytical_accelerations_smsd(
                s_train_noisy.cpu().numpy(),
                t_train.cpu().numpy(),
                args.test_case,
                dataset_dir
            )
        elif args.test_case == 'Triple_Mass_Spring_Damper':
            analytical_accels = calculate_analytical_accelerations_tmsd(
                s_train_noisy.cpu().numpy(),
                t_train.cpu().numpy(),
                args.test_case,
                dataset_dir
            )

        # Convert to tensor
        accel_tensor = torch.tensor(analytical_accels, dtype=torch.float32)

    else:
        # Use FFT-based calculation for other test cases (Double_Pendulum, Slider_Crank, Cart_Pole)
        logger.info(f"Using FFT-based accelerations for {args.test_case}")
        accelerations = []

        if args.fft_smooth_factor is None or int(args.fft_smooth_factor) <= 0:
            raise ValueError(f"fft_smooth_factor must be a positive int, got: {args.fft_smooth_factor}")
        fft_smooth_factor = int(args.fft_smooth_factor)

        for body_idx in range(num_bodies):
            vel_noisy = s_train_noisy[:, body_idx*2 + 1]

            # Use pure FFT derivative calculation
            accel_fft = calculate_fft_target_derivative(
                vel_noisy, t_train,
                deriv_smoothing_gaussian_width=max(1, len(vel_noisy) // fft_smooth_factor)
            )
            accelerations.append(accel_fft)

        # Stack accelerations
        accel_tensor = torch.stack(accelerations, dim=1)  # [steps, num_bodies]

    # Save full noisy data before trimming for plotting
    s_train_noisy_full = s_train_noisy.clone()
    accel_tensor_full = accel_tensor.clone()

    # Apply Gibbs trimming if enabled
    # This removes boundary artifacts from both ends of the pure FFT result,
    # keeping only the clean middle portion for training
    if args.use_gibbs_trim and args.add_noise:
        # For multiple bodies, we need to find a common trim region
        # Use the first body's acceleration to determine trim points
        accel_first = accel_tensor[:, 0].cpu().numpy()

        trim_result = trim_method_2_robust_residual(
            accel_first,
            smooth_window=args.trim_smooth_window,
            mad_k=args.trim_mad_k,
            stable_run=args.trim_stable_run,
            max_trim_frac=args.trim_max_frac,
            min_keep_frac=args.trim_min_keep_frac
        )

        logger.info(f"Trimmed: [{trim_result.start}:{trim_result.end}] ({trim_result.length}/{len(s_train)} samples)")

        # Apply trimming
        s_train_noisy = s_train_noisy[trim_result.start:trim_result.end]
        t_train_noisy = t_train[trim_result.start:trim_result.end]
        accel_tensor = accel_tensor[trim_result.start:trim_result.end]

        trim_info = {
            'applied': True,
            'start': trim_result.start,
            'end': trim_result.end,
            'length': trim_result.length,
            'original_length': len(s_train),
            'meta': trim_result.meta,
            's_noisy_full': s_train_noisy_full,  # Store full noisy data for plotting
            'accel_full': accel_tensor_full      # Store full acceleration data for plotting
        }
    else:
        t_train_noisy = t_train
        trim_info['s_noisy_full'] = s_train_noisy_full
        trim_info['accel_full'] = accel_tensor_full

    # Save filtered trajectory data to CSV files in dataset folder
    # These will be used by train_fnode_with_csv_targets

    # Save filtered/noisy trajectory states (s_train_noisy)
    s_train_csv_path = os.path.join(dataset_dir, "s_train_noisy.csv")
    s_train_df = pd.DataFrame(s_train_noisy.cpu().numpy())
    # Add column names for clarity
    col_names = []
    for body_idx in range(num_bodies):
        col_names.extend([f'pos_body_{body_idx}', f'vel_body_{body_idx}'])
    s_train_df.columns = col_names
    s_train_df.to_csv(s_train_csv_path, index=False)
    logger.info(f"Saved filtered trajectory to: {s_train_csv_path}")

    # Save filtered time vector
    t_train_csv_path = os.path.join(dataset_dir, "t_train_noisy.csv")
    t_train_df = pd.DataFrame({'time': t_train_noisy.cpu().numpy()})
    t_train_df.to_csv(t_train_csv_path, index=False)
    logger.info(f"Saved filtered time vector to: {t_train_csv_path}")

    # Save acceleration targets to CSV (for train_fnode_with_csv_targets)
    target_csv_path = os.path.join(dataset_dir, "noisy_target_accelerations.csv")

    # Create DataFrame with proper column names
    accel_df = pd.DataFrame()
    accel_df['time'] = t_train_noisy.cpu().numpy()
    for body_idx in range(num_bodies):
        accel_df[f'target_accel_body_{body_idx}'] = accel_tensor[:, body_idx].cpu().numpy()

    accel_df.to_csv(target_csv_path, index=False)
    logger.info(f"Saved acceleration targets to: {target_csv_path}")

    return s_train_noisy, t_train_noisy, target_csv_path, trim_info


def create_acceleration_comparison_plot(model, s_full, t_full, target_csv_path,
                                       train_end_idx, args, output_dir):
    """
    Create a plot comparing predicted accelerations vs ground truth.

    Args:
        model: Trained FNODE model
        s_full: Full trajectory states [steps, state_dim]
        t_full: Full time vector
        target_csv_path: Path to CSV with target accelerations
        train_end_idx: Index separating train and test data
        args: Command line arguments
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    logger = logging.getLogger("FNODE_Noise_Main")

    try:
        # Load target accelerations from CSV
        if target_csv_path and os.path.exists(target_csv_path):
            target_df = pd.read_csv(target_csv_path)
            target_cols = [col for col in target_df.columns if 'target_accel' in col or 'accel_body' in col]
            if target_cols:
                target_accels = torch.tensor(target_df[target_cols].values, dtype=torch.float32)
            else:
                logger.warning("No acceleration columns found in target CSV")
                return
        else:
            logger.warning("Target CSV not found for acceleration comparison")
            return

        # Get model predictions for accelerations
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            # Prepare states for model input
            s_full_device = s_full.to(device)

            # Add time feature if needed
            if model.d_interest == 1:
                time_feature = torch.zeros((s_full_device.shape[0], 1), device=device, dtype=s_full_device.dtype)
                model_input = torch.cat([s_full_device, time_feature], dim=-1)
            else:
                model_input = s_full_device

            # Get acceleration predictions from model
            predicted_accels = model(model_input)

        # Convert to CPU for plotting
        predicted_accels = predicted_accels.cpu().numpy()
        s_full_np = s_full.cpu().numpy()
        t_full_np = t_full.cpu().numpy()

        # Handle dimension mismatch
        min_len = min(len(target_accels), len(predicted_accels))
        if len(target_accels) > min_len:
            logger.info(f"Trimming target accelerations from {len(target_accels)} to {min_len}")
            target_accels = target_accels[:min_len]
        if len(predicted_accels) > min_len:
            predicted_accels = predicted_accels[:min_len]

        t_plot = t_full_np[:min_len]

        # Create figure with subplots
        num_bodies = predicted_accels.shape[1] if len(predicted_accels.shape) > 1 else 1
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Acceleration comparison (first body or all if single)
        ax = axes[0, 0]
        if num_bodies == 1:
            target_plot = target_accels.numpy() if torch.is_tensor(target_accels) else target_accels
            if len(target_plot.shape) > 1:
                target_plot = target_plot[:, 0]
            pred_plot = predicted_accels if len(predicted_accels.shape) == 1 else predicted_accels[:, 0]

            ax.plot(t_plot[:train_end_idx], target_plot[:train_end_idx], 'b-',
                   label='Target (Train)', linewidth=1.5, alpha=0.8)
            ax.plot(t_plot[train_end_idx:], target_plot[train_end_idx:], 'b--',
                   label='Target (Test)', linewidth=1.5, alpha=0.6)
            ax.plot(t_plot[:train_end_idx], pred_plot[:train_end_idx], 'r-',
                   label='Predicted (Train)', linewidth=1, alpha=0.8)
            ax.plot(t_plot[train_end_idx:], pred_plot[train_end_idx:], 'r--',
                   label='Predicted (Test)', linewidth=1, alpha=0.6)
        else:
            # For multiple bodies, plot first body
            target_plot = target_accels[:, 0].numpy() if torch.is_tensor(target_accels) else target_accels[:, 0]
            pred_plot = predicted_accels[:, 0]

            ax.plot(t_plot[:train_end_idx], target_plot[:train_end_idx], 'b-',
                   label='Target (Train)', linewidth=1.5, alpha=0.8)
            ax.plot(t_plot[train_end_idx:], target_plot[train_end_idx:], 'b--',
                   label='Target (Test)', linewidth=1.5, alpha=0.6)
            ax.plot(t_plot[:train_end_idx], pred_plot[:train_end_idx], 'r-',
                   label='Predicted (Train)', linewidth=1, alpha=0.8)
            ax.plot(t_plot[train_end_idx:], pred_plot[train_end_idx:], 'r--',
                   label='Predicted (Test)', linewidth=1, alpha=0.6)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Acceleration: Predicted vs Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=t_plot[train_end_idx] if train_end_idx < len(t_plot) else t_plot[-1],
                  color='gray', linestyle=':', alpha=0.5, label='Train/Test Split')

        # Plot 2: Acceleration error over time
        ax = axes[0, 1]
        if num_bodies == 1:
            error = pred_plot - target_plot
        else:
            error = predicted_accels[:, 0] - target_plot

        ax.plot(t_plot[:train_end_idx], error[:train_end_idx], 'g-',
               label='Error (Train)', linewidth=1, alpha=0.8)
        if train_end_idx < len(error):
            ax.plot(t_plot[train_end_idx:], error[train_end_idx:], 'g--',
                   label='Error (Test)', linewidth=1, alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prediction Error (m/s²)')
        ax.set_title('Acceleration Prediction Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Scatter plot of predicted vs target
        ax = axes[1, 0]
        if num_bodies == 1:
            ax.scatter(target_plot[:train_end_idx], pred_plot[:train_end_idx],
                      alpha=0.5, s=2, label='Train', color='blue')
            if train_end_idx < len(target_plot):
                ax.scatter(target_plot[train_end_idx:], pred_plot[train_end_idx:],
                          alpha=0.5, s=2, label='Test', color='red')
        else:
            ax.scatter(target_plot[:train_end_idx], predicted_accels[:train_end_idx, 0],
                      alpha=0.5, s=2, label='Train', color='blue')
            if train_end_idx < len(target_plot):
                ax.scatter(target_plot[train_end_idx:], predicted_accels[train_end_idx:, 0],
                          alpha=0.5, s=2, label='Test', color='red')

        # Add diagonal line for perfect prediction
        min_val = min(target_plot.min(), pred_plot.min())
        max_val = max(target_plot.max(), pred_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Target Acceleration (m/s²)')
        ax.set_ylabel('Predicted Acceleration (m/s²)')
        ax.set_title('Predicted vs Target Acceleration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Plot 4: Error statistics
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate error statistics
        train_error = error[:train_end_idx]
        test_error = error[train_end_idx:] if train_end_idx < len(error) else np.array([0])

        stats_text = "Acceleration Prediction Statistics:\n\n"
        stats_text += "Training Set:\n"
        stats_text += f"  RMSE: {np.sqrt(np.mean(train_error**2)):.6f} m/s²\n"
        stats_text += f"  MAE:  {np.mean(np.abs(train_error)):.6f} m/s²\n"
        stats_text += f"  Max:  {np.max(np.abs(train_error)):.6f} m/s²\n"
        stats_text += f"  Std:  {np.std(train_error):.6f} m/s²\n\n"

        if len(test_error) > 0 and train_end_idx < len(error):
            stats_text += "Test Set:\n"
            stats_text += f"  RMSE: {np.sqrt(np.mean(test_error**2)):.6f} m/s²\n"
            stats_text += f"  MAE:  {np.mean(np.abs(test_error)):.6f} m/s²\n"
            stats_text += f"  Max:  {np.max(np.abs(test_error)):.6f} m/s²\n"
            stats_text += f"  Std:  {np.std(test_error):.6f} m/s²\n\n"

        stats_text += "Model Configuration:\n"
        stats_text += f"  Noise Type: {args.noise_type.upper()}\n"
        stats_text += f"  Noise Level: {args.noise_level*100:.1f}%\n"
        stats_text += f"  FFT Smooth Factor: {args.fft_smooth_factor}\n"
        stats_text += f"  Train Ratio: {args.train_ratio*100:.1f}%\n"

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'Acceleration Prediction Comparison - {args.test_case}\n'
                    f'Epochs: {args.epochs}, {args.noise_type.upper()} Noise: {args.noise_level*100:.1f}%',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(output_dir, f'acceleration_prediction_comparison_{args.noise_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved acceleration prediction comparison to {save_path}")

    except Exception as e:
        logger.error(f"Failed to create acceleration comparison plot: {e}", exc_info=True)


def save_noise_summary_plot(s_train_clean, s_train_noisy, accel_targets,
                           t_train, trim_info, args, output_dir):
    """
    Create and save noise-specific summary plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convert to numpy for plotting
    s_clean_np = s_train_clean.cpu().numpy()
    s_noisy_np = s_train_noisy.cpu().numpy()
    accel_np = accel_targets.cpu().numpy() if torch.is_tensor(accel_targets) else accel_targets
    t_np = t_train.cpu().numpy()

    # Determine actual lengths (may be trimmed)
    actual_len = len(s_noisy_np)

    # Plot 1: Trajectory comparison (position)
    ax = axes[0, 0]

    # Always plot the full clean data
    ax.plot(t_np[:len(s_clean_np)], s_clean_np[:, 0], 'b-', label='Clean', linewidth=2, alpha=0.7)

    if trim_info['applied']:
        trim_start = trim_info['start']
        trim_end = trim_info['end']

        # Get full noisy data from trim_info
        s_noisy_full_np = trim_info['s_noisy_full'].cpu().numpy()

        # Plot full noisy data with lighter color (context only; avoid separate legend entry)
        ax.plot(t_np[:len(s_noisy_full_np)], s_noisy_full_np[:, 0], 'r-',
            label='_nolegend_', linewidth=1, alpha=0.25)

        # Plot the used (kept) portion
        ax.plot(t_np[trim_start:trim_end], s_noisy_np[:, 0], 'r-',
            label='Noisy', linewidth=2, alpha=0.9)

        # Add vertical lines to mark trim boundaries
        ax.axvline(x=t_np[trim_start], color='orange', linestyle='--', linewidth=1.5,
                   label=f'Trim Start (idx={trim_start})', alpha=0.8)
        ax.axvline(x=t_np[trim_end-1], color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Trim End (idx={trim_end})', alpha=0.8)

        # Add shaded regions for trimmed parts
        ax.axvspan(t_np[0], t_np[trim_start], alpha=0.15, color='gray', label='Trimmed Region')
        if trim_end < len(t_np):
            ax.axvspan(t_np[trim_end-1], t_np[-1], alpha=0.15, color='gray')
    else:
        # No trimming, plot normally
        ax.plot(t_np[:actual_len], s_noisy_np[:, 0], 'r-', label='Noisy', linewidth=1, alpha=0.7)
    ax.set_title('Position Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity comparison
    ax = axes[0, 1]

    # Always plot the full clean data
    ax.plot(t_np[:len(s_clean_np)], s_clean_np[:, 1], 'b-', label='Clean', linewidth=2, alpha=0.7)

    if trim_info['applied']:
        trim_start = trim_info['start']
        trim_end = trim_info['end']

        # Get full noisy data from trim_info
        s_noisy_full_np = trim_info['s_noisy_full'].cpu().numpy()

        # Plot full noisy data with lighter color (context only; avoid separate legend entry)
        ax.plot(t_np[:len(s_noisy_full_np)], s_noisy_full_np[:, 1], 'r-',
            label='_nolegend_', linewidth=1, alpha=0.25)

        # Plot the used (kept) portion
        ax.plot(t_np[trim_start:trim_end], s_noisy_np[:, 1], 'r-',
            label='Noisy', linewidth=2, alpha=0.9)

        # Add vertical lines to mark trim boundaries
        ax.axvline(x=t_np[trim_start], color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t_np[trim_end-1], color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)

        # Add shaded regions for trimmed parts
        ax.axvspan(t_np[0], t_np[trim_start], alpha=0.15, color='gray')
        if trim_end < len(t_np):
            ax.axvspan(t_np[trim_end-1], t_np[-1], alpha=0.15, color='gray')
    else:
        # No trimming, plot normally
        ax.plot(t_np[:actual_len], s_noisy_np[:, 1], 'r-', label='Noisy', linewidth=1, alpha=0.7)
    ax.set_title('Velocity Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Acceleration targets
    ax = axes[1, 0]

    # Determine label based on test case (all three specified test cases use analytical)
    accel_label = 'Analytical Accel' if args.test_case in ['Single_Mass_Spring', 'Single_Mass_Spring_Damper', 'Triple_Mass_Spring_Damper'] else 'FFT Accel'

    if trim_info['applied']:
        trim_start = trim_info['start']
        trim_end = trim_info['end']

        # Get full acceleration data from trim_info
        accel_full_np = trim_info['accel_full'].cpu().numpy()

        # Plot full acceleration with lighter color (context only; avoid separate legend entry)
        if accel_full_np.ndim == 1:
            ax.plot(t_np[:len(accel_full_np)], accel_full_np, 'g-',
                   label='_nolegend_', linewidth=1, alpha=0.25)
        else:
            ax.plot(t_np[:len(accel_full_np)], accel_full_np[:, 0], 'g-',
                   label='_nolegend_', linewidth=1, alpha=0.25)

        # Plot the used (kept) portion
        if accel_np.ndim == 1:
            ax.plot(t_np[trim_start:trim_end], accel_np, 'g-',
                   label=accel_label, linewidth=2, alpha=0.9)
        else:
            ax.plot(t_np[trim_start:trim_end], accel_np[:, 0], 'g-',
                   label=accel_label, linewidth=2, alpha=0.9)

        # Add vertical lines to mark trim boundaries
        ax.axvline(x=t_np[trim_start], color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t_np[trim_end-1], color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)

        # Add shaded regions for trimmed parts
        ax.axvspan(t_np[0], t_np[trim_start], alpha=0.15, color='gray')
        if trim_end < len(t_np):
            ax.axvspan(t_np[trim_end-1], t_np[-1], alpha=0.15, color='gray')
    else:
        if accel_np.ndim == 1:
            ax.plot(t_np[:actual_len], accel_np, 'g-', label=accel_label, linewidth=1.5)
        else:
            ax.plot(t_np[:actual_len], accel_np[:, 0], 'g-', label=accel_label, linewidth=1.5)
    ax.set_title('Computed Acceleration Targets' + (' (Trimmed)' if trim_info['applied'] else ''))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Noise information
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"Noise Configuration:\n\n"
    info_text += f"Noise Type: {args.noise_type.upper()}\n"
    info_text += f"Noise Level: {args.noise_level*100:.1f}%\n"
    if args.noise_type == 'band_limited':
        info_text += f"Frequency Band: [{args.freq_low:.2f}, {args.freq_high:.2f}] × Nyquist\n"
    info_text += f"FFT Smooth Factor: {args.fft_smooth_factor}\n\n"

    if trim_info['applied']:
        info_text += f"Gibbs Trimming Applied:\n"
        info_text += f"  Original: {trim_info['original_length']} samples\n"
        info_text += f"  Used range: [{trim_info['start']}:{trim_info['end']}]\n"
        info_text += f"  Final: {trim_info['length']} samples\n"
        info_text += f"  Reduction: {100*(1-trim_info['length']/trim_info['original_length']):.1f}%\n"
    else:
        info_text += "No Gibbs trimming applied\n"

    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f'Noise Training Data Preparation - {args.test_case}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'noise_data_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# --- Main function ---
def main():
    # Parse arguments
    args = parse_arguments()

    # Set global random seed
    set_seed(args.seed)

    # Device configuration
    current_device = torch.device(args.device)

    # Get noise-specific output paths (separate from FNODE)
    output_paths = {
        "figures": os.path.join(".", "figures", args.test_case, "Noise"),
        "results": os.path.join(".", "results", args.test_case, "Noise"),
        "model": os.path.join(".", "saved_model", args.test_case, "Noise"),
        "log": os.path.join(".", "log", args.test_case, "Noise")
    }

    # Setup logging with file output
    os.makedirs(output_paths["log"], exist_ok=True)
    log_file_path = os.path.join(output_paths["log"], f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file_path)

    # Log configuration with noise type
    noise_info = f"{args.noise_type.upper()} noise={args.noise_level}" if args.add_noise else "no noise"
    logger.info(f"Training: {args.test_case}, train_ratio={args.train_ratio}, {noise_info}")

    # Create output directories
    try:
        for path_key, path_value in output_paths.items():
            os.makedirs(path_value, exist_ok=True)
    except Exception as err:
        logger.error(f"Failed to create output directories: {err}")
        return

    # --- Data Generation/Loading ---
    dataset_dir = os.path.join("dataset", args.test_case)
    os.makedirs(dataset_dir, exist_ok=True)

    required_s_train_file = os.path.join(dataset_dir, "s_train.csv")
    required_t_train_file = os.path.join(dataset_dir, "t_train.csv")
    required_s_test_file = os.path.join(dataset_dir, "s_test.csv")
    required_t_test_file = os.path.join(dataset_dir, "t_test.csv")

    # Check if we need to generate data
    data_gen_needed = args.generate_new_data or \
                     not os.path.exists(required_s_train_file) or \
                     not os.path.exists(required_t_train_file)

    if data_gen_needed:
        try:
            # Generate clean dataset with default 75/25 train/test split
            # train_ratio will be applied later to use only a subset for training
            generate_dataset(
                test_case=args.test_case,
                numerical_methods="rk4",
                dt=args.data_dt,
                num_steps=args.data_total_steps
                # Don't pass train_split - use default 75/25 split
            )
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}", exc_info=True)
            return

    # Load data
    try:
        # Load training data
        s_train_df = pd.read_csv(required_s_train_file)
        t_train_df = pd.read_csv(required_t_train_file)  # Let pandas auto-detect header

        # Handle column name variations
        if s_train_df.columns[0] in ["idx", "Unnamed: 0", "index"]:
            s_train_np = s_train_df.values[:, 1:]
        else:
            s_train_np = s_train_df.values

        # Convert to tensors
        s_train_clean = torch.tensor(s_train_np, dtype=torch.float32, device='cpu')
        # Extract time values correctly (handle both with/without header cases)
        if 'time' in t_train_df.columns:
            t_train = torch.tensor(t_train_df['time'].values, dtype=torch.float32, device='cpu')
        else:
            t_train = torch.tensor(t_train_df.values.flatten(), dtype=torch.float32, device='cpu')

        # Load test data (should exist from 75/25 split)
        if os.path.exists(required_s_test_file):
            s_test_df = pd.read_csv(required_s_test_file)
            t_test_df = pd.read_csv(required_t_test_file)  # Let pandas auto-detect header

            if s_test_df.columns[0] in ["idx", "Unnamed: 0", "index"]:
                s_test_np = s_test_df.values[:, 1:]
            else:
                s_test_np = s_test_df.values

            s_test = torch.tensor(s_test_np, dtype=torch.float32, device='cpu')
            # Extract time values correctly (handle both with/without header cases)
            if 'time' in t_test_df.columns:
                t_test = torch.tensor(t_test_df['time'].values, dtype=torch.float32, device='cpu')
            else:
                t_test = torch.tensor(t_test_df.values.flatten(), dtype=torch.float32, device='cpu')
        else:
            logger.error("Test files not found! They should have been generated with 75/25 split.")
            return

        # Apply train_ratio to select subset of training data for actual training
        # This simulates having limited labeled data
        total_data_points = len(s_train_clean) + len(s_test)  # Total dataset size
        num_train_to_use = int(total_data_points * args.train_ratio)  # How many points to use for training

        # Ensure we don't try to use more training data than available
        num_train_to_use = min(num_train_to_use, len(s_train_clean))

        # Select subset of training data
        s_train_subset = s_train_clean[:num_train_to_use]
        t_train_subset = t_train[:num_train_to_use]

        # Combine for full trajectory (using all data for visualization)
        s_full = torch.cat([s_train_clean, s_test], dim=0)
        t_full = torch.cat([t_train, t_test], dim=0)

        logger.info(f"Dataset: Total={total_data_points} (train_available={len(s_train_clean)}, test={len(s_test)})")
        logger.info(f"Training subset: Using {num_train_to_use}/{len(s_train_clean)} available training samples ({args.train_ratio*100:.1f}% of total)")

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return

    # --- Add noise and compute targets (if enabled) ---
    if args.add_noise:
        s_train_noisy, t_train_noisy, target_csv_path, trim_info = prepare_noisy_data(
            s_train_subset, t_train_subset, args, dataset_dir
        )

        # Save noise summary plot
        # Load targets for plotting
        target_df = pd.read_csv(target_csv_path)
        target_cols = [col for col in target_df.columns if 'target_accel' in col]
        accel_targets = torch.tensor(target_df[target_cols].values, dtype=torch.float32)

        save_noise_summary_plot(
            s_train_subset, s_train_noisy, accel_targets,
            t_train_subset, trim_info, args, output_paths["figures"]
        )

        # Save noise configuration
        config = vars(args).copy()
        config['trim_info'] = trim_info
        config_path = os.path.join(output_paths["results"], "noise_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        # Use noisy data for training
        s_train = s_train_noisy
        t_train = t_train_noisy
    else:
        # Generate standard targets
        target_csv_path = generate_target_accelerations(
            s_full, t_full, args.test_case,
            s_train_subset.shape[1] // 2,  # num_bodies
            args, dataset_dir
        )
        s_train = s_train_subset
        t_train = t_train_subset
        trim_info = {'applied': False}

    if target_csv_path is None:
        logger.error("Failed to generate target accelerations")
        return

    # --- Model Initialization ---
    num_bodies = s_train.shape[1] // 2

    try:
        fnode_model = FNODE(
            num_bodys=num_bodies,
            layers=args.layers,
            width=args.hidden_size,
            d_interest=args.d_interest,
            activation=args.activation,
            initializer=args.initializer
        ).to(current_device)

        calculate_model_parameters(fnode_model)

    except Exception as model_init_err:
        logger.error(f"Failed to initialize FNODE model: {model_init_err}", exc_info=True)
        return

    # --- Training ---
    if not args.skip_train:
        # Setup optimizer
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(fnode_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(fnode_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Setup scheduler
        scheduler = None
        if args.lr_scheduler == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        elif args.lr_scheduler == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Training parameters
        resolved_accel_mtd = args.fnode_accel_mtd
        try:
            base_target_name = os.path.basename(target_csv_path or "")
            if "analytical" in base_target_name:
                resolved_accel_mtd = "analytical"
            elif "fft" in base_target_name:
                resolved_accel_mtd = "fft"
            elif "fd" in base_target_name:
                resolved_accel_mtd = "fd"
        except Exception:
            pass

        train_params = {
            'epochs': args.epochs,
            'grad_clip': args.grad_clip_value,
            'outime_log': 10 if args.out_time_log.lower() == 'true' else 1,  # Use 1 instead of 0 to avoid modulo by zero
            'save_ckpt_freq': args.save_ckpt_freq,
            'fnode_loss_type': args.fnode_loss_type,
            'fnode_accel_mtd': resolved_accel_mtd,
            'prob': args.prob,
            'num_workers': args.num_workers,
            'train_ratio': args.train_ratio,  # Use the actual train_ratio parameter
            'val_ratio': 0,
            'early_stop': args.early_stop.lower() == 'true',
            'patience': args.patience,
            'batch_size': args.batch_size  # Batch size for training
        }

        training_start_time = time.time()

        try:
            # Use the standard train_fnode_with_csv_targets function with pre-computed FFT targets
            # The targets are already saved to CSV during prepare_noisy_data
            trained_model, loss_history = train_fnode_with_csv_targets(
                model=fnode_model,
                s_train=s_train,
                t_train=t_train,
                train_params=train_params,
                optimizer=optimizer,
                scheduler=scheduler,
                output_paths=output_paths,
                target_csv_path=target_csv_path
            )

            training_time = time.time() - training_start_time
            logger.info(f"Training completed in {training_time:.2f}s")

            # Save final model
            save_model_state(trained_model, output_paths["model"], model_filename="FNODE_final.pkl")

            # Load best model if available
            best_model_path = os.path.join(output_paths["model"], "FNODE_best.pkl")
            if os.path.exists(best_model_path):
                load_model_state(fnode_model, output_paths["model"],
                               model_filename="FNODE_best.pkl", current_device=current_device)

        except Exception as train_err:
            logger.error(f"Training failed: {train_err}", exc_info=True)
            return

    # --- Evaluation ---
    # Test on full trajectory
    test_ode_params = {
        'ode_method': args.ode_method,
        'symplectic_use_analytic_dh': args.symplectic_use_analytic_dh,
        'test_case': args.test_case,
        'mass_value': args.mass_value
    }
    s0_for_test = s_full[0:1, :]  # Initial condition

    try:
        predictions_full = test_fnode(fnode_model, s0_for_test, t_full, test_ode_params, output_paths)

        # Calculate metrics
        train_end_idx = len(s_train)

        # Ensure same device for comparison
        pred_device = predictions_full.device
        gt_on_pred_device = s_full.to(pred_device)

        # Overall MSE
        full_mse = torch.mean((predictions_full - gt_on_pred_device) ** 2).item()

        # Train/Test MSE
        train_pred = predictions_full[:train_end_idx]
        train_gt = gt_on_pred_device[:train_end_idx]
        train_mse = torch.mean((train_pred - train_gt) ** 2).item()

        test_pred = predictions_full[train_end_idx:]
        test_gt = gt_on_pred_device[train_end_idx:]
        test_mse = torch.mean((test_pred - test_gt) ** 2).item()

        # Additional metrics for noise experiments
        train_rmse = torch.sqrt(torch.mean((train_pred - train_gt) ** 2)).item()
        test_rmse = torch.sqrt(torch.mean((test_pred - test_gt) ** 2)).item()
        train_mae = torch.mean(torch.abs(train_pred - train_gt)).item()
        test_mae = torch.mean(torch.abs(test_pred - test_gt)).item()

        # Log metrics
        logger.info(f"Results: Train RMSE={train_rmse:.6f}, Test RMSE={test_rmse:.6f}, Test/Train={test_rmse/train_rmse:.2f}")

        # Save metrics
        metrics = {
            'full_mse': [full_mse],
            'train_mse': [train_mse],
            'train_rmse': [train_rmse],
            'train_mae': [train_mae],
            'test_mse': [test_mse],
            'test_rmse': [test_rmse],
            'test_mae': [test_mae],
            'noise_level': [args.noise_level if args.add_noise else 0],
            'samples_used': [trim_info['length'] if trim_info['applied'] else len(s_train)]
        }

        save_data_pd(metrics, output_paths["results"], "test_metrics.csv")

        # Save predictions to Noise folder
        save_data(predictions_full, args.test_case, "Noise",
                 len(s_train), predictions_full.shape[0] - len(s_train), args.data_dt)

    except Exception as eval_err:
        logger.error(f"Evaluation failed: {eval_err}", exc_info=True)
        return

    # --- Plotting ---
    try:
        # Standard trajectory comparison
        plot_trajectory_comparison(
            test_case_name=args.test_case,
            model_predictions={"Noise": predictions_full.reshape(-1, num_bodies, 2)},  # Keep as tensor
            ground_truth_trajectory=s_full.reshape(-1, num_bodies, 2),  # Keep as tensor
            time_vector=t_full,  # Keep as tensor
            num_bodies_to_plot=min(num_bodies, 3),
            num_steps_train=train_end_idx,
            output_dir=output_paths["figures"],
            base_filename=f"{args.test_case}_Noise_comparison",
            num_epochs=args.epochs
        )

        # Create acceleration prediction comparison plot
        create_acceleration_comparison_plot(
            fnode_model, s_full, t_full, target_csv_path,
            train_end_idx, args, output_paths["figures"]
        )

        # Acceleration comparison
        # Load analytical accelerations for comparison
        analytical_accel_path = os.path.join(output_paths["results"], "analytical_accelerations.csv")
        if not os.path.exists(analytical_accel_path):
            # Generate analytical accelerations based on test case
            analytical_accels = calculate_analytical_accelerations(
                s_full.cpu().numpy(), t_full.cpu().numpy(),
                args.test_case, output_paths["results"]
            )

    except Exception as plot_err:
        logger.error(f"Plotting failed: {plot_err}", exc_info=True)

    logger.info("Completed successfully!")


if __name__ == "__main__":
    main()
