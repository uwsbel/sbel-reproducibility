# FNODE/main_fnode_fric.py
# Main script for training FNODE with friction parameter support
import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt


# --- Simplified Logging Configuration ---
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
logger = logging.getLogger("FNODE_Friction_Main")

# --- Ensure Model directory is in the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    logger.info(f"Adding Model package directory to sys.path: {model_package_dir}")
    sys.path.insert(0, model_package_dir)

# --- Imports from custom Model package ---
try:
    from Model.Data_generator import generate_slider_crank_dataset_with_friction
    from Model.utils import *
    from Model.model import *
    from Model.force_fun import *

    logger.info("Successfully imported components from Model package/directory.")
except ImportError as e:
    logger.error(f"Failed to import from Model package/directory: {e}. Check PYTHONPATH.", exc_info=True)
    sys.exit(1)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="FNODE with Friction Parameter Support")

    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")

    # Friction parameter settings
    parser.add_argument('--c_max', type=float, default=0.6,
                        help='Maximum friction coefficient')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')

    # Data generation parameters
    parser.add_argument('--generate_new_data', action='store_true', default=False,
                        help='Generate new dataset (default: use existing if available)')
    parser.add_argument('--time_span', type=float, default=20.0,
                        help='Time span for each trajectory in seconds')
    parser.add_argument('--dt_sample', type=float, default=0.01,
                        help='Sampling timestep for training data (default: 0.01)')

    # Model architecture
    parser.add_argument('--layers', type=int, default=6,
                        help="Number of layers for FNODE (input + hidden + output).")
    parser.add_argument('--hidden_size', type=int, default=256,
                        help="Hidden layer width for FNODE.")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'],
                        help="Activation function.")
    parser.add_argument('--initializer', type=str, default='xavier', choices=['xavier', 'kaiming'],
                        help="Weight initializer.")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate.")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help="Optimizer.")
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                        choices=['none', 'exponential', 'step', 'cosine'],
                        help="LR scheduler.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help="Decay rate for schedulers.")
    parser.add_argument('--lr_decay_steps', type=int, default=1,
                        help="Step size or T_max for schedulers.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay for optimizer.")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Gradient clipping value.")

    # Other parameters
    parser.add_argument('--fd_order', type=int, default=4,
                        help="Finite difference order for acceleration computation (2, 4, or 6).")

    # Plotting parameters
    parser.add_argument('--plot_train_indices', type=str, default='0,-1',
                        help="Comma-separated indices of training trajectories to plot (e.g., '0,-1' for first and last)")
    parser.add_argument('--plot_test_indices', type=str, default='1,3',
                        help="Comma-separated indices of test trajectories to plot (e.g., '0,2' for first and third)")
    parser.add_argument('--out_time_log', type=int, default=1,
                        help="Log training progress every N epochs.")
    parser.add_argument('--prob', type=int, default=100,
                        help="Probability factor for FFT truncation.")
    parser.add_argument('--early_stop', type=str, default='False',
                        help="Enable early stopping ('True' or 'False').")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience for early stopping.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of parallel workers for data loading.")
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training, load model and test.")
    parser.add_argument('--model_load_filename', type=str, default="FNODE_fric.pkl",
                        help="Model filename to load when skip_train is True.")

    args = parser.parse_args()
    args.model_type = 'FNODE'
    args.test_case = 'Slider_Crank'
    args.d_interest = 1  # We use d_interest=1 for friction parameter
    args.fnode_use_hybrid_target = False  # Use finite difference for friction case

    logger.info(f"Arguments parsed: {args}")
    return args


def generate_multi_friction_datasets(args):
    """Generate training and test datasets with different friction values"""
    logger.info("=== Multi-Friction Datasets ===")

    # Calculate friction values for training and testing
    c_train_values = np.arange(0.0, args.c_max + 0.01, 0.1)
    c_test_values = np.arange(0.05, args.c_max, 0.1)

    logger.info(f"Training friction values: {c_train_values}")
    logger.info(f"Test friction values (interpolation): {c_test_values}")

    # Create cache directory for datasets
    cache_dir = os.path.join('dataset', 'Slider_Crank_Friction')
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file names based on parameters
    cache_file_train = os.path.join(cache_dir, f'train_c{args.c_max}_t{args.time_span}_dt{args.dt_sample}_seed{args.seed}.pkl')
    cache_file_test = os.path.join(cache_dir, f'test_c{args.c_max}_t{args.time_span}_dt{args.dt_sample}_seed{args.seed}.pkl')

    # Check if we should use cached data
    use_cache = not args.generate_new_data and os.path.exists(cache_file_train) and os.path.exists(cache_file_test)

    if use_cache:
        logger.info("Cache files found. Loading cached trajectories...")
        logger.info(f"  Training cache: {cache_file_train}")
        logger.info(f"  Test cache: {cache_file_test}")
        try:
            with open(cache_file_train, 'rb') as f:
                import pickle
                train_data = pickle.load(f)
                train_trajectories = train_data['trajectories']
                cached_c_train = train_data['c_values']

            with open(cache_file_test, 'rb') as f:
                test_data = pickle.load(f)
                test_trajectories = test_data['trajectories']
                cached_c_test = test_data['c_values']

            # Verify the c values match
            if np.allclose(cached_c_train, c_train_values) and np.allclose(cached_c_test, c_test_values):
                logger.info("Successfully loaded cached trajectories")
                return train_trajectories, test_trajectories, c_train_values, c_test_values
            else:
                logger.warning("Cached c values don't match current settings, regenerating...")
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}, regenerating...")

    # Generate new trajectories
    if args.generate_new_data:
        logger.info("Force generating new trajectories (--generate_new_data flag set)...")
    elif not os.path.exists(cache_file_train):
        logger.info("Training cache file not found. Generating trajectories for first run...")
        logger.info(f"  Expected cache: {cache_file_train}")
    elif not os.path.exists(cache_file_test):
        logger.info("Test cache file not found. Generating trajectories for first run...")
        logger.info(f"  Expected cache: {cache_file_test}")
    else:
        logger.info("Generating new trajectories...")

    # Generate training trajectories
    train_trajectories = []
    logger.info("Generating training trajectories...")
    for c_val in c_train_values:
        logger.info(f"  Generating trajectory with c={c_val:.2f}")
        s_data, t_data = generate_slider_crank_dataset_with_friction(
            c_slide=c_val,
            time_span=args.time_span,
            seed=args.seed + int(c_val * 100)  # Different seed for each trajectory
        )
        train_trajectories.append((s_data, t_data))

    # Generate test trajectories
    test_trajectories = []
    logger.info("Generating test trajectories...")
    for c_val in c_test_values:
        logger.info(f"  Generating test trajectory with c={c_val:.2f}")
        s_data, t_data = generate_slider_crank_dataset_with_friction(
            c_slide=c_val,
            time_span=args.time_span,
            seed=args.seed + int(c_val * 100) + 1000  # Different seed
        )
        test_trajectories.append((s_data, t_data))

    # Save to cache
    logger.info("Saving trajectories to cache...")
    try:
        import pickle
        with open(cache_file_train, 'wb') as f:
            pickle.dump({'trajectories': train_trajectories, 'c_values': c_train_values}, f)
        with open(cache_file_test, 'wb') as f:
            pickle.dump({'trajectories': test_trajectories, 'c_values': c_test_values}, f)
        logger.info(f"Cached trajectories saved to {cache_dir}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return train_trajectories, test_trajectories, c_train_values, c_test_values


def compute_normalization_stats(trajectories, dt_sample, fd_order=4):
    """Compute normalization statistics for theta, omega, and accelerations

    Args:
        trajectories: List of trajectory data
        dt_sample: Time step between samples
        fd_order: Finite difference order for acceleration computation (2, 4, or 6)
    """
    all_theta = []
    all_omega = []
    all_accel = []

    for s_data, _ in trajectories:
        all_theta.append(s_data[:, 0])
        all_omega.append(s_data[:, 1])

        # Compute accelerations using higher-order finite differences
        # Create time vector for this trajectory
        time_vector = torch.arange(len(s_data)) * dt_sample

        # Use estimate_temporal_gradient_finite_diff for acceleration computation
        omega_data = s_data[:, 1]
        accel = estimate_temporal_gradient_finite_diff(
            omega_data,
            time_vector,
            order=fd_order,
            smooth_boundaries=False
        )

        if accel is not None:
            # Remove the last element to match the original length (N-1)
            # since we're computing differences
            all_accel.append(accel[:-1] if len(accel) == len(s_data) else accel)

    all_theta = torch.cat(all_theta)
    all_omega = torch.cat(all_omega)
    all_accel = torch.cat(all_accel)

    # Compute mean and std for normalization
    theta_mean = all_theta.mean().item()
    theta_std = all_theta.std().item()
    omega_mean = all_omega.mean().item()
    omega_std = all_omega.std().item()
    accel_mean = all_accel.mean().item()
    accel_std = all_accel.std().item()

    # Avoid division by zero
    if theta_std < 1e-6:
        theta_std = 1.0
    if omega_std < 1e-6:
        omega_std = 1.0
    if accel_std < 1e-6:
        accel_std = 1.0

    stats = {
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'omega_mean': omega_mean,
        'omega_std': omega_std,
        'accel_mean': accel_mean,
        'accel_std': accel_std
    }

    logger.info(f"Normalization statistics computed:")
    logger.info(f"  - theta: mean={theta_mean:.4f}, std={theta_std:.4f}")
    logger.info(f"  - omega: mean={omega_mean:.4f}, std={omega_std:.4f}")
    logger.info(f"  - accel: mean={accel_mean:.4f}, std={accel_std:.4f}")

    return stats


def normalize_trajectories(trajectories, stats):
    """Normalize trajectories using computed statistics"""
    normalized_trajectories = []

    for s_data, t_data in trajectories:
        s_normalized = s_data.clone()
        s_normalized[:, 0] = (s_data[:, 0] - stats['theta_mean']) / stats['theta_std']
        s_normalized[:, 1] = (s_data[:, 1] - stats['omega_mean']) / stats['omega_std']
        normalized_trajectories.append((s_normalized, t_data))

    return normalized_trajectories


def combine_friction_trajectories(trajectories, c_values, normalize_c_max):
    """Combine multiple trajectories with different friction values into single tensors"""
    s_combined_list = []
    t_combined_list = []

    current_time_offset = 0.0

    for (s_data, t_data), c_val in zip(trajectories, c_values):
        # Add normalized friction parameter as third column
        # Note: s_data is already normalized for theta and omega
        c_normalized = c_val / normalize_c_max
        c_column = torch.full((len(s_data), 1), c_normalized, dtype=torch.float32)
        s_with_c = torch.cat([s_data, c_column], dim=-1)
        s_combined_list.append(s_with_c)

        # Adjust time vector to be continuous
        t_adjusted = t_data + current_time_offset
        t_combined_list.append(t_adjusted)

        # Update time offset for next trajectory
        current_time_offset = t_adjusted[-1].item() + 0.01  # Add small gap between trajectories

    # Combine all trajectories
    s_combined = torch.cat(s_combined_list, dim=0)
    t_combined = torch.cat(t_combined_list, dim=0)

    return s_combined, t_combined


def plot_results(model, train_trajectories, test_trajectories,
                 train_trajectories_original, test_trajectories_original,
                 c_train_values, c_test_values, norm_stats,
                 args, output_dir,
                 train_indices_to_show=None, test_indices_to_show=None):
    """Generate visualization plots with improved clarity

    Args:
        train_indices_to_show: List of indices for training trajectories to display.
                              If None, defaults to [0, -1] (first and last)
        test_indices_to_show: List of indices for test trajectories to display.
                             If None, defaults to [0, -1] (first and last)
    """
    logger.info("=== Generating Main Results Visualization ===")

    # Create figure with 2x2 subplots for better layout
    fig = plt.figure(figsize=(14, 11))

    # Select specific c values to display:
    # Default: Training min/max, Test furthest points
    if train_indices_to_show is None:
        train_indices_to_show = [0, -1]  # First and last training values
    if test_indices_to_show is None:
        test_indices_to_show = [0, -1]   # c=0.05 and c=0.55

    # Create colors for selected trajectories
    # Use color maps that scale with the number of selected trajectories
    import matplotlib.cm as cm
    train_cmap = cm.get_cmap('Blues')
    test_cmap = cm.get_cmap('Reds')

    # Generate colors based on number of trajectories to show
    n_train = len(train_indices_to_show)
    n_test = len(test_indices_to_show)

    # Create colors for train trajectories (blue shades)
    if n_train == 1:
        train_colors_selected = ['#1f77b4']
    elif n_train == 2:
        train_colors_selected = ['#1f77b4', '#2ca02c']  # Blue and green
    else:
        train_colors_selected = [train_cmap(0.4 + 0.5 * i / max(n_train - 1, 1)) for i in range(n_train)]

    # Create colors for test trajectories (red/orange shades)
    if n_test == 1:
        test_colors_selected = ['#ff7f0e']
    elif n_test == 2:
        test_colors_selected = ['#ff7f0e', '#d62728']  # Orange and red
    else:
        test_colors_selected = [test_cmap(0.4 + 0.5 * i / max(n_test - 1, 1)) for i in range(n_test)]

    model.eval()
    device = next(model.parameters()).device

    # Generate predictions only for selected trajectories
    selected_train_predictions = []
    selected_test_predictions = []

    # Generate predictions for selected training trajectories
    for idx in train_indices_to_show:
        c_val = c_train_values[idx]
        s_data_norm, t_data = train_trajectories[idx]
        s_data_orig, _ = train_trajectories_original[idx]
        with torch.no_grad():
            s0_norm = s_data_norm[0:1].to(device)
            t_eval = t_data.to(device)
            c_normalized = torch.tensor([c_val / args.c_max], dtype=torch.float32).to(device)

            pred_states_norm = [s0_norm]
            current_state_norm = s0_norm
            dt = args.dt_sample

            for t_idx in range(len(t_eval) - 1):
                model_input = torch.cat([current_state_norm, c_normalized.unsqueeze(0)], dim=-1)
                pred_accel_norm = model(model_input)
                # Denormalize the acceleration for integration
                pred_accel = pred_accel_norm * norm_stats['accel_std'] + norm_stats['accel_mean']
                # For normalized omega update: we need to add the change in normalized space
                # delta_omega_original = pred_accel * dt
                # delta_omega_normalized = delta_omega_original / omega_std
                new_omega_norm = current_state_norm[:, 1:2] + (pred_accel * dt) / norm_stats['omega_std']
                # For theta update: denormalize omega, integrate, then normalize the result
                omega_denorm = current_state_norm[:, 1:2] * norm_stats['omega_std'] + norm_stats['omega_mean']
                new_theta_norm = current_state_norm[:, 0:1] + (omega_denorm * dt) / norm_stats['theta_std']
                current_state_norm = torch.cat([new_theta_norm, new_omega_norm], dim=-1)
                pred_states_norm.append(current_state_norm)

            pred_states_norm = torch.cat(pred_states_norm, dim=0).cpu().numpy()
            # Denormalize predictions for visualization
            pred_states = pred_states_norm.copy()
            pred_states[:, 0] = pred_states_norm[:, 0] * norm_stats['theta_std'] + norm_stats['theta_mean']
            pred_states[:, 1] = pred_states_norm[:, 1] * norm_stats['omega_std'] + norm_stats['omega_mean']
            selected_train_predictions.append(pred_states)

    # Generate predictions for selected test trajectories
    for idx in test_indices_to_show:
        c_val = c_test_values[idx]
        s_data_norm, t_data = test_trajectories[idx]
        s_data_orig, _ = test_trajectories_original[idx]
        with torch.no_grad():
            s0_norm = s_data_norm[0:1].to(device)
            t_eval = t_data.to(device)
            c_normalized = torch.tensor([c_val / args.c_max], dtype=torch.float32).to(device)

            pred_states_norm = [s0_norm]
            current_state_norm = s0_norm
            dt = args.dt_sample

            for t_idx in range(len(t_eval) - 1):
                model_input = torch.cat([current_state_norm, c_normalized.unsqueeze(0)], dim=-1)
                pred_accel_norm = model(model_input)
                # Denormalize the acceleration for integration
                pred_accel = pred_accel_norm * norm_stats['accel_std'] + norm_stats['accel_mean']
                # For normalized omega update: we need to add the change in normalized space
                # delta_omega_original = pred_accel * dt
                # delta_omega_normalized = delta_omega_original / omega_std
                new_omega_norm = current_state_norm[:, 1:2] + (pred_accel * dt) / norm_stats['omega_std']
                # For theta update: denormalize omega, integrate, then normalize the result
                omega_denorm = current_state_norm[:, 1:2] * norm_stats['omega_std'] + norm_stats['omega_mean']
                new_theta_norm = current_state_norm[:, 0:1] + (omega_denorm * dt) / norm_stats['theta_std']
                current_state_norm = torch.cat([new_theta_norm, new_omega_norm], dim=-1)
                pred_states_norm.append(current_state_norm)

            pred_states_norm = torch.cat(pred_states_norm, dim=0).cpu().numpy()
            # Denormalize predictions for visualization
            pred_states = pred_states_norm.copy()
            pred_states[:, 0] = pred_states_norm[:, 0] * norm_stats['theta_std'] + norm_stats['theta_mean']
            pred_states[:, 1] = pred_states_norm[:, 1] * norm_stats['omega_std'] + norm_stats['omega_mean']
            selected_test_predictions.append(pred_states)

    # Subplot 1: Theta Time Series (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Angular Position (θ) Time Series', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Time [s]', fontsize=20)
    ax1.set_ylabel('θ [rad]', fontsize=20)
    ax1.grid(True, alpha=0.3)

    # Plot selected training trajectories
    for i, idx in enumerate(train_indices_to_show):
        c_val = c_train_values[idx]
        s_data_orig, t_data = train_trajectories_original[idx]
        # Ground truth
        ax1.plot(t_data.numpy(), s_data_orig[:, 0].numpy(),
                color=train_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5,
                label=f'Train c={c_val:.1f}')
        # Prediction
        ax1.plot(t_data.numpy(), selected_train_predictions[i][:, 0],
                color=train_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    # Plot selected test trajectories
    for i, idx in enumerate(test_indices_to_show):
        c_val = c_test_values[idx]
        s_data_orig, t_data = test_trajectories_original[idx]
        # Ground truth
        ax1.plot(t_data.numpy(), s_data_orig[:, 0].numpy(),
                color=test_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5,
                label=f'Test c={c_val:.2f}')
        # Prediction
        ax1.plot(t_data.numpy(), selected_test_predictions[i][:, 0],
                color=test_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    ax1.set_xlim([0, 5])  # Show first 5 seconds for clarity
    # No individual legend - will be combined at top

    # Subplot 2: Omega Time Series (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Angular Velocity (ω) Time Series', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Time [s]', fontsize=20)
    ax2.set_ylabel('ω [rad/s]', fontsize=20)
    ax2.grid(True, alpha=0.3)

    # Plot selected training trajectories
    for i, idx in enumerate(train_indices_to_show):
        c_val = c_train_values[idx]
        s_data_orig, t_data = train_trajectories_original[idx]
        # Ground truth
        ax2.plot(t_data.numpy(), s_data_orig[:, 1].numpy(),
                color=train_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)
        # Prediction
        ax2.plot(t_data.numpy(), selected_train_predictions[i][:, 1],
                color=train_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    # Plot selected test trajectories
    for i, idx in enumerate(test_indices_to_show):
        c_val = c_test_values[idx]
        s_data_orig, t_data = test_trajectories_original[idx]
        # Ground truth
        ax2.plot(t_data.numpy(), s_data_orig[:, 1].numpy(),
                color=test_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)
        # Prediction
        ax2.plot(t_data.numpy(), selected_test_predictions[i][:, 1],
                color=test_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    ax2.set_xlim([0, 5])  # Show first 5 seconds for clarity
    # No individual legend - will be combined at top

    # Subplot 3: Phase Space Portrait (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Phase Space Portrait', fontsize=20, fontweight='bold')
    ax3.set_xlabel('θ [rad]', fontsize=20)
    ax3.set_ylabel('ω [rad/s]', fontsize=20)
    ax3.grid(True, alpha=0.3)

    # Plot phase space for selected trajectories
    for i, idx in enumerate(train_indices_to_show):
        c_val = c_train_values[idx]
        s_data_orig, t_data = train_trajectories_original[idx]
        # Ground truth
        ax3.plot(s_data_orig[:, 0].numpy(), s_data_orig[:, 1].numpy(),
                color=train_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)
        # Prediction
        ax3.plot(selected_train_predictions[i][:, 0], selected_train_predictions[i][:, 1],
                color=train_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    for i, idx in enumerate(test_indices_to_show):
        c_val = c_test_values[idx]
        s_data_orig, t_data = test_trajectories_original[idx]
        # Ground truth
        ax3.plot(s_data_orig[:, 0].numpy(), s_data_orig[:, 1].numpy(),
                color=test_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)
        # Prediction
        ax3.plot(selected_test_predictions[i][:, 0], selected_test_predictions[i][:, 1],
                color=test_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    # No individual legend - will be combined at top

    # Subplot 4: Angular Acceleration Comparison (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Angular Acceleration Comparison', fontsize=20, fontweight='bold')
    ax4.set_xlabel('Time [s]', fontsize=20)
    ax4.set_ylabel('α [rad/s²]', fontsize=20)
    ax4.grid(True, alpha=0.3)

    # Plot acceleration for selected training trajectories
    for i, idx in enumerate(train_indices_to_show):
        c_val = c_train_values[idx]
        s_data_orig, t_data = train_trajectories_original[idx]
        s_data_norm, _ = train_trajectories[idx]

        # Calculate true accelerations from original data
        dt = args.dt_sample
        true_accel = np.diff(s_data_orig[:, 1].numpy()) / dt

        # Calculate predicted accelerations using normalized inputs
        with torch.no_grad():
            c_normalized = torch.tensor([c_val / args.c_max], dtype=torch.float32).to(device)
            pred_accels = []
            for t_idx in range(len(s_data_norm)):
                state_norm = s_data_norm[t_idx:t_idx+1].to(device)
                model_input = torch.cat([state_norm, c_normalized.unsqueeze(0)], dim=-1)
                pred_accel_norm = model(model_input).cpu().numpy()
                # Denormalize the predicted acceleration
                pred_accel = pred_accel_norm[0, 0] * norm_stats['accel_std'] + norm_stats['accel_mean']
                pred_accels.append(pred_accel)

        # Only plot for first 5 seconds for clarity
        time_mask = t_data[:-1].numpy() <= 5
        ax4.plot(t_data[:-1].numpy()[time_mask], true_accel[time_mask],
                color=train_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)

        time_mask_pred = t_data.numpy() <= 5
        ax4.plot(t_data.numpy()[time_mask_pred], np.array(pred_accels)[time_mask_pred],
                color=train_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    # Plot acceleration for selected test trajectories
    for i, idx in enumerate(test_indices_to_show):
        c_val = c_test_values[idx]
        s_data_orig, t_data = test_trajectories_original[idx]
        s_data_norm, _ = test_trajectories[idx]

        # Calculate true accelerations from original data
        dt = args.dt_sample
        true_accel = np.diff(s_data_orig[:, 1].numpy()) / dt

        # Calculate predicted accelerations using normalized inputs
        with torch.no_grad():
            c_normalized = torch.tensor([c_val / args.c_max], dtype=torch.float32).to(device)
            pred_accels = []
            for t_idx in range(len(s_data_norm)):
                state_norm = s_data_norm[t_idx:t_idx+1].to(device)
                model_input = torch.cat([state_norm, c_normalized.unsqueeze(0)], dim=-1)
                pred_accel_norm = model(model_input).cpu().numpy()
                # Denormalize the predicted acceleration
                pred_accel = pred_accel_norm[0, 0] * norm_stats['accel_std'] + norm_stats['accel_mean']
                pred_accels.append(pred_accel)

        time_mask = t_data[:-1].numpy() <= 5
        ax4.plot(t_data[:-1].numpy()[time_mask], true_accel[time_mask],
                color=test_colors_selected[i], linestyle='-', alpha=0.9, linewidth=2.5)

        time_mask_pred = t_data.numpy() <= 5
        ax4.plot(t_data.numpy()[time_mask_pred], np.array(pred_accels)[time_mask_pred],
                color=test_colors_selected[i], linestyle='--', alpha=0.7, linewidth=2.0)

    ax4.set_xlim([0, 5])
    # No individual legend - will be combined at top

    # Add a combined legend at the top with both trajectory labels and line styles
    from matplotlib.lines import Line2D

    # Create legend handles for trajectories and line styles
    legend_handles = []
    legend_labels = []

    # Add ground truth and prediction pairs for each trajectory
    for i, idx in enumerate(train_indices_to_show):
        c_val = c_train_values[idx]
        # Ground truth
        legend_handles.append(Line2D([0], [0], color=train_colors_selected[i], linestyle='-', linewidth=2))
        legend_labels.append(f'Ground Truth (c={c_val:.1f})')
        # Prediction
        legend_handles.append(Line2D([0], [0], color=train_colors_selected[i], linestyle='--', linewidth=2))
        legend_labels.append(f'Prediction (c={c_val:.1f})')

    for i, idx in enumerate(test_indices_to_show):
        c_val = c_test_values[idx]
        # Ground truth
        legend_handles.append(Line2D([0], [0], color=test_colors_selected[i], linestyle='-', linewidth=2))
        legend_labels.append(f'Ground Truth (c={c_val:.2f})')
        # Prediction
        legend_handles.append(Line2D([0], [0], color=test_colors_selected[i], linestyle='--', linewidth=2))
        legend_labels.append(f'Prediction (c={c_val:.2f})')

    fig.legend(legend_handles, legend_labels,
              loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=len(legend_handles)//2, fontsize=16, frameon=True, fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave more space for legend at top
    plt.savefig(os.path.join(output_dir, 'main_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved main results plot to {output_dir}/main_results.png")


# --- Main Execution ---
def main():
    args = parse_arguments()

    # Setup directories using Model package utility
    output_paths = get_output_paths(args.test_case, args.model_type)

    # Add custom figure directory for friction experiment
    figures_dir = os.path.join('figures', 'fnode_fric')
    os.makedirs(figures_dir, exist_ok=True)

    # Ensure all directories exist
    os.makedirs(output_paths["model"], exist_ok=True)
    os.makedirs(output_paths["results"], exist_ok=True)
    os.makedirs(output_paths["figures"], exist_ok=True)

    # Setup logging with file
    log_dir = os.path.join('log', args.test_case)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'fnode_fric.log')
    setup_logging(log_file)
    logger = logging.getLogger("FNODE_Friction_Main")

    logger.info("=== Starting FNODE with Friction Parameter Training ===")
    logger.info(f"Configuration: c_max={args.c_max}, batch_size={args.batch_size}")

    # Set random seed
    set_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Generate datasets
    train_trajectories, test_trajectories, c_train_values, c_test_values = \
        generate_multi_friction_datasets(args)

    # Compute normalization statistics using training data only
    logger.info("=== Computing Normalization Statistics ===")
    norm_stats = compute_normalization_stats(train_trajectories, args.dt_sample, fd_order=args.fd_order)

    # Apply normalization to both train and test trajectories
    logger.info("=== Normalizing Trajectories ===")
    train_trajectories_normalized = normalize_trajectories(train_trajectories, norm_stats)
    test_trajectories_normalized = normalize_trajectories(test_trajectories, norm_stats)

    # Store both normalized and original for visualization
    train_trajectories_original = train_trajectories
    test_trajectories_original = test_trajectories
    train_trajectories = train_trajectories_normalized
    test_trajectories = test_trajectories_normalized

    # Combine training trajectories into single tensor with friction parameter
    logger.info("=== Combining Training Trajectories ===")
    s_train_combined, t_train_combined = combine_friction_trajectories(
        train_trajectories, c_train_values, args.c_max
    )

    # Also create full trajectory for visualization
    all_trajectories = train_trajectories + test_trajectories
    all_c_values = np.concatenate([c_train_values, c_test_values])
    s_full_combined, t_full_combined = combine_friction_trajectories(
        all_trajectories, all_c_values, args.c_max
    )

    logger.info(f"Combined training data shape: {s_train_combined.shape}")
    logger.info(f"Combined training time shape: {t_train_combined.shape}")
    logger.info(f"State dimension: 2 (theta, omega) + 1 (friction) = 3")

    # Verify sampling is working
    expected_points_per_traj = int(args.time_span / args.dt_sample)
    actual_points_per_traj = len(train_trajectories[0][0])
    logger.info(f"Data sampling verification:")
    logger.info(f"  - Expected points per trajectory (dt={args.dt_sample}): {expected_points_per_traj}")
    logger.info(f"  - Actual points per trajectory: {actual_points_per_traj}")
    logger.info(f"  - Total training points: {len(s_train_combined)}")

    if actual_points_per_traj != expected_points_per_traj:
        logger.warning(f"WARNING: Data may not be sampled correctly! Expected {expected_points_per_traj} but got {actual_points_per_traj}")

    # Initialize model
    logger.info("=== Model Initialization ===")
    model = FNODE(
        num_bodys=1,  # Single body (slider-crank has 1 angular acceleration)
        layers=args.layers,
        width=args.hidden_size,
        d_interest=args.d_interest,  # 1 for friction parameter
        activation=args.activation,
        initializer=args.initializer
    ).to(device)

    logger.info(f"Model initialized: InputDim={model.dim_input}, OutputDim={model.output_dim}")
    calculate_model_parameters(model)

    # Generate target accelerations using Model package function
    logger.info("=== Generating Target Accelerations ===")

    # For the combined trajectory, we need to handle the third column (friction parameter)
    # The generate_target_accelerations function expects only state data [theta, omega]
    # So we'll compute targets from the original trajectories and combine them

    # Create a temporary combined state tensor without friction for target generation
    s_for_target_list = []
    t_for_target_list = []
    current_time_offset = 0.0

    for (s_data, t_data) in train_trajectories:
        s_for_target_list.append(s_data)
        t_adjusted = t_data + current_time_offset
        t_for_target_list.append(t_adjusted)
        current_time_offset = t_adjusted[-1].item() + 0.01

    s_for_target = torch.cat(s_for_target_list, dim=0)
    t_for_target = torch.cat(t_for_target_list, dim=0)

    # Generate target accelerations
    target_csv_path = generate_target_accelerations(
        s_for_target,  # State without friction parameter
        t_for_target,
        args.test_case,
        num_bodies=1,
        args=args,
        results_dir=output_paths["results"]
    )

    if target_csv_path is None:
        logger.error("Failed to generate target accelerations. Exiting.")
        return

    logger.info(f"Target accelerations saved to: {target_csv_path}")

    # Skip acceleration normalization - use original targets
    logger.info("Using original target accelerations (no normalization)")

    # Set norm_stats for accelerations to identity transform
    norm_stats['accel_mean'] = 0.0
    norm_stats['accel_std'] = 1.0

    # Use original target file directly
    target_csv_normalized_path = target_csv_path
    logger.info(f"Using original targets from: {target_csv_normalized_path}")

    # Store the path for potential use in skip_train mode
    args.target_csv_normalized_path = target_csv_normalized_path

    # Setup optimizer and scheduler
    logger.info("=== Optimizer and Scheduler Setup ===")
    optimizer_class = optim.AdamW if args.optimizer == 'adamw' else optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr * 0.0001)

    logger.info(f"Optimizer '{args.optimizer}' and Scheduler '{args.lr_scheduler}' configured.")

    # Training or loading model
    if not args.skip_train:
        # Training using Model package function
        logger.info("=== FNODE Training ===")
        start_time = time.time()

        # Prepare training parameters
        train_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'grad_clip': args.grad_clip,
            'outime_log': args.out_time_log,
            'save_ckpt_freq': 0,  # Disable intermediate checkpoints
            'fnode_loss_type': 'derivative',
            'fnode_use_hybrid_target': False,  # Use finite difference for friction case
            'prob': args.prob,
            'num_workers': args.num_workers,
            'train_ratio': 1.0,  # Use all data for training
            'val_ratio': 0.0,  # No validation set
            'early_stop': args.early_stop.lower() == 'true',
            'patience': args.patience
        }

        logger.info(f"Training parameters: {train_params}")

        # Train the model with normalized targets
        trained_model, loss_history = train_fnode_with_csv_targets(
            model=model,
            s_train=s_train_combined,  # Combined state with friction
            t_train=t_train_combined,
            train_params=train_params,
            optimizer=optimizer,
            scheduler=scheduler,
            output_paths=output_paths,
            target_csv_path=target_csv_normalized_path  # Use normalized targets
        )

        if trained_model is None:
            logger.error("Training failed - function returned None.")
            return

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save the trained model with friction support
        model_save_path = os.path.join(output_paths["model"], 'FNODE_friction_best.pkl')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Saved friction model to {model_save_path}")

        # Also save a copy with a simpler name for easy loading
        model_simple_path = os.path.join(output_paths["model"], 'FNODE_fric.pkl')
        torch.save(model.state_dict(), model_simple_path)
        logger.info(f"Saved friction model copy to {model_simple_path}")

        # Save normalization statistics
        norm_stats_path = os.path.join(output_paths["model"], "normalization_stats.pkl")
        import pickle
        with open(norm_stats_path, 'wb') as f:
            pickle.dump(norm_stats, f)
        logger.info(f"Normalization statistics saved to {norm_stats_path}")

        # Load best model for evaluation
        best_model_path = os.path.join(output_paths["model"], "FNODE_best.pkl")
        if os.path.exists(best_model_path):
            load_model_state(model, output_paths["model"], model_filename="FNODE_best.pkl", current_device=device)
            logger.info("Best model loaded for evaluation")
        else:
            logger.warning("Best model not found, using final trained model")
            model = trained_model
    else:
        # Skip training and load pre-trained model
        logger.info("=== Skipping Training - Loading Pre-trained Model ===")
        model_path = os.path.join(output_paths["model"], args.model_load_filename)

        if not os.path.exists(model_path):
            # Try alternate paths
            alt_path = os.path.join('saved_models', args.model_load_filename)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                alt_path = os.path.join('saved_models', 'fnode_fric_best.pth')
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    logger.error(f"Model file not found at {model_path} or alternate paths. Cannot proceed.")
                    return

        # Load the model
        logger.info(f"Loading model from {model_path}")
        try:
            if model_path.endswith('.pth'):
                # Load state dict directly
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                # Use the utility function for .pkl files
                success = load_model_state(model, os.path.dirname(model_path),
                                          model_filename=os.path.basename(model_path),
                                          current_device=device)
                if not success:
                    logger.error(f"Failed to load model from {model_path}")
                    return

            logger.info(f"Successfully loaded model from {model_path}")

            # Also load normalization statistics
            norm_stats_path = os.path.join(output_paths["model"], "normalization_stats.pkl")
            if os.path.exists(norm_stats_path):
                import pickle
                with open(norm_stats_path, 'rb') as f:
                    loaded_norm_stats = pickle.load(f)
                    # Update norm_stats with loaded values
                    norm_stats.update(loaded_norm_stats)
                logger.info(f"Loaded normalization statistics from {norm_stats_path}")
            else:
                logger.warning("Normalization statistics not found - using computed values from current data")

            training_time = 0.0  # No training time when skipping

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

    # Parse plotting indices if specified
    train_plot_indices = None
    test_plot_indices = None

    if args.plot_train_indices:
        try:
            train_plot_indices = [int(idx.strip()) for idx in args.plot_train_indices.split(',')]
            logger.info(f"Using custom training plot indices: {train_plot_indices}")
        except ValueError:
            logger.warning(f"Invalid plot_train_indices format: {args.plot_train_indices}. Using defaults.")

    if args.plot_test_indices:
        try:
            test_plot_indices = [int(idx.strip()) for idx in args.plot_test_indices.split(',')]
            logger.info(f"Using custom test plot indices: {test_plot_indices}")
        except ValueError:
            logger.warning(f"Invalid plot_test_indices format: {args.plot_test_indices}. Using defaults.")

    # Generate visualizations
    plot_results(model, train_trajectories, test_trajectories,
                train_trajectories_original, test_trajectories_original,
                c_train_values, c_test_values, norm_stats,
                args, figures_dir,
                train_indices_to_show=train_plot_indices,
                test_indices_to_show=test_plot_indices)

    # Save numerical results and log MSE for all c values
    results_file = os.path.join(output_paths["results"], 'fnode_fric_metrics.txt')

    # Store MSE values for logging
    train_mse_dict = {}
    test_mse_dict = {}

    with open(results_file, 'w') as f:
        f.write("Training Configuration:\n")
        f.write(f"- c_max: {args.c_max}\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- learning_rate: {args.lr}\n")
        f.write(f"- dt_sample: {args.dt_sample}\n")
        f.write(f"- time_span: {args.time_span}\n")
        f.write("\n")

        f.write("Training Set Performance:\n")
        for i, c_val in enumerate(c_train_values):
            # Calculate MSE for this trajectory
            s_data_norm, t_data = train_trajectories[i]
            s_data_orig, _ = train_trajectories_original[i]
            with torch.no_grad():
                mse = 0.0
                count = 0
                for t_idx in range(len(s_data_norm) - 1):
                    state_norm = s_data_norm[t_idx:t_idx+1].to(device)
                    c_normalized = torch.tensor([[c_val / args.c_max]], dtype=torch.float32).to(device)
                    model_input = torch.cat([state_norm, c_normalized], dim=-1)
                    pred_accel_norm = model(model_input)
                    # Denormalize the predicted acceleration for MSE calculation
                    pred_accel = pred_accel_norm.cpu().item() * norm_stats['accel_std'] + norm_stats['accel_mean']

                    dt = args.dt_sample
                    true_accel = (s_data_orig[t_idx+1, 1] - s_data_orig[t_idx, 1]) / dt
                    mse += (pred_accel - true_accel.item()) ** 2
                    count += 1
                mse = mse / count if count > 0 else 0.0
            train_mse_dict[c_val] = mse
            f.write(f"- c={c_val:.2f}: MSE={mse:.6e}\n")

        f.write("\n")
        f.write("Test Set Performance (Interpolation):\n")
        for i, c_val in enumerate(c_test_values):
            s_data_norm, t_data = test_trajectories[i]
            s_data_orig, _ = test_trajectories_original[i]
            with torch.no_grad():
                mse = 0.0
                count = 0
                for t_idx in range(len(s_data_norm) - 1):
                    state_norm = s_data_norm[t_idx:t_idx+1].to(device)
                    c_normalized = torch.tensor([[c_val / args.c_max]], dtype=torch.float32).to(device)
                    model_input = torch.cat([state_norm, c_normalized], dim=-1)
                    pred_accel_norm = model(model_input)
                    # Denormalize the predicted acceleration for MSE calculation
                    pred_accel = pred_accel_norm.cpu().item() * norm_stats['accel_std'] + norm_stats['accel_mean']

                    dt = args.dt_sample
                    true_accel = (s_data_orig[t_idx+1, 1] - s_data_orig[t_idx, 1]) / dt
                    mse += (pred_accel - true_accel.item()) ** 2
                    count += 1
                mse = mse / count if count > 0 else 0.0
            test_mse_dict[c_val] = mse
            f.write(f"- c={c_val:.2f}: MSE={mse:.6e}\n")

        f.write("\n")
        f.write("Overall Metrics:\n")
        if not args.skip_train:
            if 'loss_history' in locals() and loss_history is not None:
                # Handle both dict and list formats
                if isinstance(loss_history, dict):
                    if 'train_loss' in loss_history and len(loss_history['train_loss']) > 0:
                        f.write(f"- Final Training Loss: {loss_history['train_loss'][-1]:.6e}\n")
                elif isinstance(loss_history, list) and len(loss_history) > 0:
                    f.write(f"- Final Training Loss: {loss_history[-1]:.6e}\n")
            if 'training_time' in locals():
                f.write(f"- Training Time: {training_time:.2f} seconds\n")
        else:
            f.write(f"- Model loaded from: {args.model_load_filename}\n")

    logger.info(f"Saved numerical results to {results_file}")

    # Log MSE for all c values
    logger.info("=== MSE Results for All Friction Values ===")
    logger.info("Training Set MSE:")
    for c_val, mse in train_mse_dict.items():
        logger.info(f"  c={c_val:.2f}: MSE={mse:.6e}")

    logger.info("Test Set MSE (Interpolation):")
    for c_val, mse in test_mse_dict.items():
        logger.info(f"  c={c_val:.2f}: MSE={mse:.6e}")

    # Calculate and log average MSE
    avg_train_mse = np.mean(list(train_mse_dict.values()))
    avg_test_mse = np.mean(list(test_mse_dict.values()))
    logger.info(f"Average Training MSE: {avg_train_mse:.6e}")
    logger.info(f"Average Test MSE: {avg_test_mse:.6e}")
    logger.info("=== FNODE Friction Training Complete ===")


if __name__ == '__main__':
    main()