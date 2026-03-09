# FNODE/main_accel.py
"""
Script for comparing FFT vs FD acceleration computation methods.
Uses the EXACT SAME functions as main_fnode.py for consistency.
"""
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AccelComparison")

# Add Model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# Import required modules - USE SAME FUNCTIONS AS main_fnode.py
from Model.utils import estimate_temporal_gradient_finite_diff, calculate_fft_target_derivative


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare FFT vs FD acceleration methods")

    parser.add_argument('--test_case', type=str, default='Slider_Crank',
                        choices=['Single_Mass_Spring_Damper', 'Double_Pendulum',
                                 'Triple_Mass_Spring_Damper', 'Slider_Crank', 'Cart_Pole'],
                        help="Dynamical system to analyze")
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help="Path to dataset directory")
    parser.add_argument('--output_dir', type=str, default='./accel_comparison',
                        help="Output directory for results")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device")
    parser.add_argument('--save_csv', action='store_true', default=True,
                        help="Save computed accelerations to CSV")
    parser.add_argument('--prob', type=int, default=100,
                        help="prob parameter for hybrid method boundary calculation (same as main_fnode.py)")

    return parser.parse_args()


def load_trajectory_data(test_case, data_path, device):
    """Load trajectory data from dataset files (TRAINING DATA ONLY)"""
    dataset_dir = os.path.join(data_path, test_case)

    # Load training data only
    s_train_df = pd.read_csv(os.path.join(dataset_dir, "s_train.csv"))
    t_train_df = pd.read_csv(os.path.join(dataset_dir, "t_train.csv"))

    # Process data based on test case
    if test_case == "Slider_Crank":
        if 'theta_0_2pi' in s_train_df.columns and 'omega' in s_train_df.columns:
            s_train_np = s_train_df[['theta_0_2pi', 'omega']].values
        else:
            s_train_np = s_train_df.iloc[:, 1:3].values if 'Unnamed: 0' in s_train_df.columns else s_train_df.iloc[:, 0:2].values
    else:
        s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values

    # Get time data
    t_train_np = t_train_df['time'].values if 'time' in t_train_df.columns else t_train_df.values.flatten()

    # Use only training data (no test data)
    s_tensor = torch.tensor(s_train_np, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_train_np, dtype=torch.float32, device=device)

    train_size = len(s_train_np)

    logger.info(f"Loaded TRAINING data only: {s_tensor.shape}, train_size={train_size}")

    return s_tensor, t_tensor, train_size


def compute_accelerations(s_tensor, t_tensor, test_case, train_size, prob, output_dir):
    """
    Compute accelerations using EXACT SAME methods as main_fnode.py (TRAINING DATA ONLY):
    1. FD method (4th order finite difference) - training trajectory only
    2. FFT method (paper methodology) - training trajectory only
    3. Hybrid method (FFT + FD boundaries) - training data only, matching main_fnode.py exactly
    """

    # Determine number of bodies
    if test_case == "Slider_Crank":
        num_bodies = 1
    else:
        num_bodies = s_tensor.shape[-1] // 2

    results = {}
    device = s_tensor.device

    # Calculate truncation for hybrid method boundaries (SAME as main_fnode.py)
    num_total_points = train_size
    trunc = num_total_points // prob

    # Ensure reasonable truncation
    if trunc >= num_total_points // 3:
        logger.warning(f"Boundary size too large with prob={prob}, adjusting")
        trunc = num_total_points // 4
        prob = num_total_points // trunc

    logger.info(f"Hybrid method: boundary size={trunc}, training points={num_total_points}, prob={prob}")

    # 1. Finite Difference (SAME as main_fnode.py uses) - training trajectory only
    logger.info("Computing FD accelerations (4th order) for training trajectory...")
    fd_accels = torch.zeros((len(s_tensor), num_bodies), device=device)

    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < s_tensor.shape[-1]:
            velocity = s_tensor[:, velocity_idx]
            fd_deriv = estimate_temporal_gradient_finite_diff(velocity, t_tensor, order=4)
            if fd_deriv is not None:
                fd_accels[:, body_idx] = fd_deriv

    results['FD (4th order)'] = fd_accels

    # 2. FFT method (SAME as main_fnode.py uses) - training trajectory only
    logger.info("Computing FFT accelerations (paper methodology) for training trajectory...")
    fft_accels = torch.zeros((len(s_tensor), num_bodies), device=device)

    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < s_tensor.shape[-1]:
            velocity = s_tensor[:, velocity_idx]
            fft_deriv = calculate_fft_target_derivative(velocity, t_tensor)
            if fft_deriv is not None:
                fft_accels[:, body_idx] = fft_deriv

    results['FFT (full)'] = fft_accels

    # 3. Hybrid method (SAME as main_fnode.py) - training data only
    logger.info("Computing Hybrid accelerations (FFT+FD boundaries) for training data...")
    hybrid_accels_train = torch.zeros((train_size, num_bodies), device=device)

    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < s_tensor.shape[-1]:
            velocity_train = s_tensor[:, velocity_idx]

            # Step 1: Calculate FFT for entire training set
            fft_full = calculate_fft_target_derivative(velocity_train, t_tensor)
            if fft_full is None:
                logger.warning(f"FFT failed for body {body_idx}, using FD")
                fd_full = estimate_temporal_gradient_finite_diff(velocity_train, t_tensor, order=4)
                hybrid_accels_train[:, body_idx] = fd_full if fd_full is not None else 0
                continue

            # Start with FFT results
            hybrid_accels_train[:, body_idx] = fft_full

            # Step 2: Calculate FD for boundary replacement
            fd_partial = estimate_temporal_gradient_finite_diff(velocity_train, t_tensor, order=4)
            if fd_partial is None:
                logger.warning(f"FD failed for body {body_idx}, keeping FFT")
                continue

            # Step 3: Replace BOTH beginning and end with FD
            # Beginning
            hybrid_accels_train[:trunc, body_idx] = fd_partial[:trunc]

            # End
            end_start_idx = num_total_points - trunc
            hybrid_accels_train[end_start_idx:, body_idx] = fd_partial[end_start_idx:]

            # Step 4: Smooth blending at transition points
            blend_length = min(10, trunc // 4)

            if blend_length > 0:
                # Blend at beginning (around index trunc)
                for i in range(blend_length):
                    idx = trunc - blend_length // 2 + i
                    if 0 <= idx < num_total_points:
                        alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                        hybrid_accels_train[idx, body_idx] = (
                            (1 - alpha) * fd_partial[idx] + alpha * fft_full[idx]
                        )

                # Blend at end (around index end_start_idx)
                for i in range(blend_length):
                    idx = end_start_idx - blend_length // 2 + i
                    if 0 <= idx < num_total_points:
                        alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                        hybrid_accels_train[idx, body_idx] = (
                            alpha * fd_partial[idx] + (1 - alpha) * fft_full[idx]
                        )

            logger.info(f"Body {body_idx}: Hybrid = FD[0:{trunc}] + FFT[{trunc}:{end_start_idx}] + FD[{end_start_idx}:{num_total_points}]")

    results['Hybrid (train only)'] = hybrid_accels_train
    results['_hybrid_metadata'] = {
        'trunc': trunc,
        'end_start_idx': num_total_points - trunc,
        'blend_length': min(10, trunc // 4),
        'train_size': train_size
    }

    return results


def plot_comparison(results, t_tensor, train_size, test_case, output_dir):
    """Create comparison plots showing FD, FFT, and Hybrid methods (training data only)"""

    time_np = t_tensor.cpu().numpy()

    # Get metadata
    metadata = results.get('_hybrid_metadata', {})
    trunc = metadata.get('trunc', 0)
    end_start_idx = metadata.get('end_start_idx', train_size)
    blend_length = metadata.get('blend_length', 0)

    # Get first non-metadata result to determine num_bodies
    num_bodies = None
    for key, val in results.items():
        if not key.startswith('_') and val is not None:
            num_bodies = val.shape[1]
            break

    if num_bodies is None:
        logger.error("No valid acceleration data found")
        return

    # Color scheme
    colors = {
        'FD (4th order)': 'blue',
        'FFT (full)': 'red',
        'Hybrid (train only)': 'green'
    }

    # Figure: Full comparison
    fig, axes = plt.subplots(num_bodies, 1, figsize=(16, 6 * num_bodies), sharex=True)
    if num_bodies == 1:
        axes = [axes]

    for body_idx in range(num_bodies):
        ax = axes[body_idx]

        # Plot FD and FFT for training trajectory
        for method_name in ['FD (4th order)', 'FFT (full)']:
            if method_name in results and results[method_name] is not None:
                accel_data = results[method_name]
                accel_np = accel_data.cpu().numpy()
                ax.plot(time_np, accel_np[:, body_idx],
                        color=colors.get(method_name, 'gray'),
                        label=method_name, linewidth=1.5, alpha=0.7)

        # Plot Hybrid
        if 'Hybrid (train only)' in results and results['Hybrid (train only)'] is not None:
            hybrid_data = results['Hybrid (train only)']
            hybrid_np = hybrid_data.cpu().numpy()
            ax.plot(time_np, hybrid_np[:, body_idx],
                    color=colors['Hybrid (train only)'],
                    label='Hybrid (FD+FFT)', linewidth=2.0, alpha=0.9)

            # Mark hybrid method boundaries
            if trunc > 0:
                # Mark FD start boundary
                ax.axvline(x=time_np[trunc], color='purple', linestyle=':',
                          alpha=0.6, linewidth=1.5, label=f'FD→FFT transition (idx={trunc})')

                # Mark FD end boundary
                if end_start_idx < train_size:
                    ax.axvline(x=time_np[end_start_idx], color='orange', linestyle=':',
                              alpha=0.6, linewidth=1.5, label=f'FFT→FD transition (idx={end_start_idx})')

                # Shade FD regions
                ax.axvspan(time_np[0], time_np[trunc], alpha=0.1, color='blue',
                          label=f'FD start region [0:{trunc}]')
                if end_start_idx < train_size:
                    ax.axvspan(time_np[end_start_idx], time_np[-1], alpha=0.1, color='blue',
                              label=f'FD end region [{end_start_idx}:{train_size}]')

                # Shade FFT region
                if end_start_idx > trunc:
                    ax.axvspan(time_np[trunc], time_np[end_start_idx], alpha=0.1, color='red',
                              label=f'FFT region [{trunc}:{end_start_idx}]')

        ax.set_ylabel(f'Acceleration (Body {body_idx + 1})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, ncol=2)

        if test_case == "Slider_Crank":
            ax.set_title(f'Angular Acceleration Comparison: FD vs FFT vs Hybrid (Training Only) - {test_case}', fontsize=16)
        else:
            ax.set_title(f'Body {body_idx + 1} Acceleration Comparison (Training Only) - {test_case}', fontsize=16)

    axes[-1].set_xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'acceleration_comparison_with_hybrid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparison plot saved to: {save_path}")


def compute_statistics(results, train_size):
    """Compute statistics for each method (training data only)"""
    stats = {}

    for method_name, accel_data in results.items():
        if method_name.startswith('_'):  # Skip metadata
            continue
        if accel_data is not None:
            accel_np = accel_data.cpu().numpy()

            method_stats = {
                'mean': np.mean(accel_np),
                'std': np.std(accel_np),
                'min': np.min(accel_np),
                'max': np.max(accel_np),
            }

            stats[method_name] = method_stats

    return stats


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging to file
    log_file = os.path.join(args.output_dir, 'accel_comparison.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Starting acceleration comparison for {args.test_case}")
    logger.info(f"Using SAME methods as main_fnode.py")
    logger.info(f"Arguments: {args}")

    # Load data
    s_tensor, t_tensor, train_size = load_trajectory_data(args.test_case, args.data_path, args.device)

    # Compute accelerations using SAME functions as main_fnode.py
    results = compute_accelerations(s_tensor, t_tensor, args.test_case, train_size, args.prob, args.output_dir)

    # Compute statistics
    stats = compute_statistics(results, train_size)

    # Save statistics
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(args.output_dir, 'method_statistics.csv'))
    logger.info("Statistics:")
    logger.info("\n" + str(stats_df))

    # Create plots
    plot_comparison(results, t_tensor, train_size, args.test_case, args.output_dir)

    # Save combined results (training data only)
    if args.save_csv:
        time_np = t_tensor.cpu().numpy()

        # Get metadata for method labeling
        metadata = results.get('_hybrid_metadata', {})
        trunc = metadata.get('trunc', 0)
        end_start_idx = metadata.get('end_start_idx', train_size)
        blend_length = metadata.get('blend_length', 0)

        # Create method indicator
        method_indicator = []
        for i in range(train_size):
            if i < trunc - blend_length // 2:
                method_indicator.append('FD_start')
            elif trunc - blend_length // 2 <= i < trunc + blend_length // 2:
                method_indicator.append('Blend_start')
            elif i >= end_start_idx + blend_length // 2:
                method_indicator.append('FD_end')
            elif end_start_idx - blend_length // 2 <= i < end_start_idx + blend_length // 2:
                method_indicator.append('Blend_end')
            else:
                method_indicator.append('FFT')

        # Save all methods in one CSV
        combined_data = {'time': time_np, 'method': method_indicator}

        for method_name in ['FD (4th order)', 'FFT (full)', 'Hybrid (train only)']:
            if method_name in results and results[method_name] is not None:
                accel_data = results[method_name]
                accel_np = accel_data.cpu().numpy()
                for body_idx in range(accel_np.shape[1]):
                    col_name = f'{method_name.replace(" ", "_").replace("(", "").replace(")", "")}_body_{body_idx}'
                    combined_data[col_name] = accel_np[:, body_idx]

        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(os.path.join(args.output_dir, 'all_methods_train_accelerations.csv'),
                           index=False, float_format='%.8g')
        logger.info(f"All methods (training data) saved to all_methods_train_accelerations.csv")

    logger.info("=" * 70)
    logger.info("Acceleration comparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Generated files:")
    logger.info("  - acceleration_comparison_with_hybrid.png: Visual comparison plot with hybrid boundaries")
    logger.info("  - all_methods_train_accelerations.csv: All methods (FD, FFT, Hybrid) for training data")
    logger.info("  - method_statistics.csv: Statistical summary")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
