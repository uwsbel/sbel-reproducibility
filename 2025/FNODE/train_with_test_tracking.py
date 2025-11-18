#!/usr/bin/env python3
"""
Comprehensive training script for all 4 models with epoch-wise test MSE tracking.

This script:
1. Trains MBDNODE, FNODE, LSTM, FCNN on Double Pendulum
2. Performs test inference after EACH epoch
3. Tracks training time and test MSE per epoch
4. Generates comparison plot: Test MSE vs Training Time
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Set matplotlib backend to Agg before importing pyplot to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, prevents ICE errors
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

# Add Model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# Imports
from Model.Data_generator import generate_dataset
from Model.utils import set_seed, get_output_paths
from Model.model import (
    MBDNODE, FNODE, LSTMModel, FCNN,
    test_MBDNODE, test_fnode, infer_lstm,
    neural_network_force_function_MBDNODE
)
from Model.integrator import runge_kutta_four_multiple_body

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainAllModels")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train all 4 models with epoch-wise test tracking")

    parser.add_argument('--test_case', type=str, default='Triple_Mass_Spring_Damper',
                        choices=['Cart_Pole', 'Double_Pendulum', 'Single_Mass_Spring_Damper',
                                'Slider_Crank', 'Triple_Mass_Spring_Damper'],
                        help='Test case to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (must match main scripts for alignment)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_total_steps', type=int, default=400)
    parser.add_argument('--train_ratio', type=float, default=0.75)
    parser.add_argument('--data_dt', type=float, default=0.01)
    parser.add_argument('--force_regen_data', action='store_true', default=True,
                        help='Force regenerate dataset even if CSV files exist (use when changing train_ratio, seed, or data_total_steps)')

    # Training epochs for each model (reduced to reasonable testing values)
    parser.add_argument('--mbdnode_epochs', type=int, default=300)
    parser.add_argument('--fnode_epochs', type=int, default=450)
    parser.add_argument('--lstm_epochs', type=int, default=300)
    parser.add_argument('--fcnn_epochs', type=int, default=450)

    # Model execution control (skip flags to disable models, all enabled by default)
    parser.add_argument('--skip_mbdnode', action='store_true',
                        help='Skip MBDNODE training')
    parser.add_argument('--skip_fnode', action='store_true',
                        help='Skip FNODE training')
    parser.add_argument('--skip_lstm', action='store_true',
                        help='Skip LSTM training')
    parser.add_argument('--skip_fcnn', action='store_true',
                        help='Skip FCNN training')

    # Test inference control
    parser.add_argument('--test_freq', type=int, default=10, help='Test every N epochs (0 to disable testing)')
    parser.add_argument('--use_full_traj_mse', action='store_true', default=True,
                        help='Use full trajectory MSE (train+test) instead of test-only MSE')
    parser.add_argument('--skip_test_inference', action='store_true',
                        help='Disable test inference during training')

    # FNODE specific parameters
    parser.add_argument('--prob', type=int, default=10, help='FNODE probability parameter')
    parser.add_argument('--fnode_use_hybrid_target', action='store_true', default=False,
                        help='Use hybrid FFT-FD targets for FNODE')

    return parser.parse_args()


def generate_data(args):
    """Generate dataset for specified test case - matching main_fnode.py approach"""
    logger.info("="*80)
    logger.info(f"GENERATING DATASET FOR {args.test_case}")
    logger.info("="*80)

    num_steps_train = int(args.data_total_steps * args.train_ratio)

    # Generate dataset if files don't exist OR if force_regen_data flag is set
    dataset_path = os.path.join(os.getcwd(), 'dataset', args.test_case)
    s_train_file = os.path.join(dataset_path, 's_train.csv')

    if not os.path.exists(s_train_file) or args.force_regen_data:
        if args.force_regen_data:
            logger.info(f"Force regenerating dataset (--force_regen_data flag set)")
        logger.info(f"Generating dataset (seed={args.seed}, steps={args.data_total_steps}, train_ratio={args.train_ratio})")
        ground_trajectory = generate_dataset(
            test_case=args.test_case,
            numerical_methods='rk4',
            dt=args.data_dt,
            num_steps=args.data_total_steps,
            seed=args.seed,
            gen_train_num_steps=num_steps_train,
            output_root_dir='.',
            if_noise=False,
            save_to_file=True
        )
        logger.info(f"Dataset generated: train_steps={num_steps_train}, test_steps={args.data_total_steps - num_steps_train}")
    else:
        logger.info(f"Dataset files already exist, skipping generation (use --force_regen_data to regenerate)")

    # ALWAYS load from CSV files (matching main_*.py behavior EXACTLY)
    # This ensures EXACT numerical alignment with main scripts
    logger.info("Loading dataset from CSV files (matching main_*.py approach)")

    # Load full continuous trajectory (s_full.csv and t_full.csv)
    # This avoids discontinuity issues when concatenating s_train and s_test
    s_full_path = os.path.join(dataset_path, 's_full.csv')
    t_full_path = os.path.join(dataset_path, 't_full.csv')

    if os.path.exists(s_full_path) and os.path.exists(t_full_path):
        logger.info("Using s_full.csv and t_full.csv for continuous trajectory")
        s_full_df = pd.read_csv(s_full_path)
        t_full_df = pd.read_csv(t_full_path)

        # Remove index column if present
        s_full_np = s_full_df.values[:, 1:] if s_full_df.columns[0] in ["idx", "Unnamed: 0"] else s_full_df.values
        t_full_np = t_full_df['time'].values if 'time' in t_full_df.columns else t_full_df.values.flatten()

        # Calculate actual split point based on the data we have
        # Use train_ratio if the actual data size matches expected total_steps
        actual_total = len(s_full_np)
        if actual_total == args.data_total_steps:
            split_idx = num_steps_train
        else:
            # Use existing train file size as split point
            train_csv_path = os.path.join(dataset_path, 's_train.csv')
            if os.path.exists(train_csv_path):
                train_df_temp = pd.read_csv(train_csv_path)
                split_idx = len(train_df_temp)
                logger.info(f"Using existing s_train.csv size ({split_idx}) as split point")
            else:
                split_idx = int(actual_total * args.train_ratio)
                logger.info(f"Using train_ratio ({args.train_ratio}) to split at {split_idx}")

        # Split into train and test
        s_train_np = s_full_np[:split_idx]
        s_test_np = s_full_np[split_idx:]
        t_train_np = t_full_np[:split_idx]
        t_test_np = t_full_np[split_idx:]

        logger.info(f"Split s_full ({len(s_full_np)}) into train ({len(s_train_np)}) and test ({len(s_test_np)})")
    else:
        # Fallback to separate train/test files (legacy behavior)
        logger.warning(f"s_full.csv not found, using separate s_train.csv and s_test.csv (may have discontinuity!)")
        s_train_df = pd.read_csv(os.path.join(dataset_path, 's_train.csv'))
        s_test_df = pd.read_csv(os.path.join(dataset_path, 's_test.csv'))
        t_train_df = pd.read_csv(os.path.join(dataset_path, 't_train.csv'))
        t_test_df = pd.read_csv(os.path.join(dataset_path, 't_test.csv'))

        # Remove index column if present
        s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values
        s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values

        # For time columns, explicitly get 'time' column
        t_train_np = t_train_df['time'].values if 'time' in t_train_df.columns else t_train_df.values.flatten()
        t_test_np = t_test_df['time'].values if 'time' in t_test_df.columns else t_test_df.values.flatten()

    # Convert to tensors on CPU (matching main_fnode.py line 282-284 EXACTLY)
    # Keep flat shape [n, 4] - do NOT reshape to [n, 2, 2] to match main scripts exactly
    s_train = torch.tensor(s_train_np, dtype=torch.float32, device='cpu')
    s_test = torch.tensor(s_test_np, dtype=torch.float32, device='cpu')
    t_train = torch.tensor(t_train_np, dtype=torch.float32, device='cpu')
    t_test = torch.tensor(t_test_np, dtype=torch.float32, device='cpu')

    logger.info(f"Data loaded from CSV: train={s_train.shape}, test={s_test.shape}")
    logger.info(f"Initial state: {s_train[0].tolist()}")
    logger.info(f"CSV path: {dataset_path}")
    logger.info(f"Train steps: {len(s_train)}, Test steps: {len(s_test)}")

    return s_train, s_test, t_train, t_test


def train_mbdnode_with_tracking(args, s_train, s_test, t_train, t_test, device):
    """Train MBDNODE with epoch-wise test MSE tracking"""
    logger.info("="*80)
    logger.info(f"TRAINING MBDNODE ON {args.test_case}")
    logger.info("="*80)

    # Setup
    output_paths = get_output_paths(args.test_case, 'MBDNODE')
    os.makedirs(output_paths['results'], exist_ok=True)
    os.makedirs(output_paths['model'], exist_ok=True)

    # Determine number of bodies from data shape
    num_bodies = s_train.shape[1] // 2

    # Model
    model = MBDNODE(num_bodys=num_bodies, layers=3, width=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # Prepare training data (matching Model/model.py EXACTLY - keep on CPU for DataLoader!)
    step_delay = 2

    # Reshape to 3D [n, num_bodies, 2] for MBDNODE (matching main_mbdnode.py)
    s_train_3d = s_train.cpu().reshape(-1, num_bodies, 2)
    s_test_3d = s_test.cpu().reshape(-1, num_bodies, 2)

    training_size = s_train_3d.shape[0]
    inputs_list = []
    targets_list = []
    for i in range(training_size - step_delay):
        inputs_list.append(s_train_3d[i, :, :].clone())
        targets_list.append(s_train_3d[i + step_delay - 1, :, :].clone())

    train_inputs = torch.stack(inputs_list)
    train_targets = torch.stack(targets_list)

    # Create DataLoader (matching Model/model.py line 1058-1061)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_inputs, train_targets)
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=use_cuda, persistent_workers=False)

    # Prepare test data - use 3D for MBDNODE
    s_full_3d = torch.cat([s_train_3d, s_test_3d], dim=0)
    t_full = torch.cat([t_train, t_test], dim=0)

    # Training loop with test tracking
    tracking_data = []
    cumulative_time = 0.0
    best_test_mse = float('inf')

    for epoch in range(args.mbdnode_epochs):
        epoch_start = time.time()

        # Training (using DataLoader like main_mbdnode.py)
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            # Move batch to device (matching Model/model.py line 1112-1113)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Zero gradients at start of batch (matching Model/model.py line 1116)
            optimizer.zero_grad()
            batch_loss = 0

            # Process each sample in the batch (matching Model/model.py line 1121-1145)
            for idx in range(batch_inputs.shape[0]):
                # Get individual sample (matching Model/model.py line 1123-1124)
                inputs = batch_inputs[idx].clone().detach().requires_grad_(True)
                target = batch_targets[idx].clone().detach()

                # Forward through RK4 integration (matching Model/model.py line 1127-1129)
                integrated_pred = runge_kutta_four_multiple_body(
                    inputs, neural_network_force_function_MBDNODE,
                    step_delay, args.data_dt, if_final_state=True, model=model
                )

                # Calculate loss (matching Model/model.py line 1132-1135)
                loss = criterion(integrated_pred, target)

                # Normalize loss by batch size for gradient accumulation
                loss_normalized = loss / batch_inputs.shape[0]

                # Accumulate gradients
                loss_normalized.backward(retain_graph=True)

                # Track loss for reporting
                batch_loss += loss.item()

                optimizer.step()

            epoch_loss += batch_loss
            num_batches = 1

        avg_train_loss = epoch_loss / len(train_loader)

        # Get current LR before stepping (matching Model/model.py)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        epoch_time = time.time() - epoch_start
        cumulative_time += epoch_time

        # Test inference (every test_freq epochs, if enabled)
        test_mse = None
        if not args.skip_test_inference and args.test_freq > 0 and (epoch + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                test_trajectory = test_MBDNODE(
                    numerical_methods='rk4',
                    model=model,
                    body=s_train_3d[0].cpu().numpy(),
                    num_steps=len(s_full_3d),
                    dt=args.data_dt,
                    device=device
                )
                # test_MBDNODE returns a Tensor, not numpy array
                if isinstance(test_trajectory, torch.Tensor):
                    test_pred = test_trajectory.cpu()
                else:
                    test_pred = torch.from_numpy(test_trajectory).float()

                # Ensure length alignment
                min_len = min(test_pred.shape[0], s_full_3d.shape[0])
                test_pred_aligned = test_pred[:min_len]
                s_full_3d_aligned = s_full_3d[:min_len].cpu()

                # Calculate MSE based on user preference
                if args.use_full_traj_mse:
                    # Full trajectory MSE (train + test)
                    test_mse = nn.MSELoss()(test_pred_aligned, s_full_3d_aligned).item()
                else:
                    # Test-only MSE (extrapolation region only)
                    train_len = len(s_train_3d)
                    if min_len > train_len:
                        test_mse = nn.MSELoss()(test_pred_aligned[train_len:], s_full_3d_aligned[train_len:]).item()
                    else:
                        test_mse = float('nan')

                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    torch.save(model.state_dict(), os.path.join(output_paths['model'], 'MBDNODE_best.pth'))

        tracking_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_mse': test_mse,
            'epoch_time': epoch_time,
            'cumulative_time': cumulative_time,
            'learning_rate': current_lr  # LR before stepping
        })

        if (epoch + 1) % 10 == 0:
            test_mse_str = f"{test_mse:.6e}" if test_mse is not None else "N/A"
            logger.info(f"MBDNODE Epoch {epoch+1}/{args.mbdnode_epochs}: "
                       f"Train Loss={avg_train_loss:.6e}, Test MSE={test_mse_str}, "
                       f"Time={cumulative_time/60:.2f}min")

    # Save tracking data
    df = pd.DataFrame(tracking_data)
    df.to_csv(os.path.join(output_paths['results'], 'MBDNODE_test_tracking.csv'), index=False)
    logger.info(f"MBDNODE training complete. Best test MSE: {best_test_mse:.6e}")

    return df


def train_fnode_with_tracking(args, s_train, s_test, t_train, t_test, device, model, optimizer, scheduler):
    """
    Train FNODE with EXACT training logic from train_fnode_with_csv_targets() in Model/model.py.
    Adds test inference after each epoch WITHOUT affecting training process.
    Guarantees perfect alignment with main_fnode.py (same training code).
    """
    from Model.model import test_fnode
    from Model.utils import (estimate_temporal_gradient_finite_diff, calculate_fft_target_derivative,
                             save_model_state)
    import time
    import numpy as np

    logger.info("="*80)
    logger.info(f"TRAINING FNODE ON {args.test_case} WITH EXACT Model/model.py LOGIC + TEST TRACKING")
    logger.info("="*80)

    # Setup
    output_paths = get_output_paths(args.test_case, 'FNODE')
    os.makedirs(output_paths['results'], exist_ok=True)
    os.makedirs(output_paths['model'], exist_ok=True)

    # ===== EXACT COPY from train_fnode_with_csv_targets() lines 315-587 =====

    current_device = next(model.parameters()).device
    use_hybrid = args.fnode_use_hybrid_target
    method_name = "Hybrid FFT-FD" if use_hybrid else "Pure Finite Difference"
    logger.info(f"--- Starting FNODE Training with {method_name} Method ---")

    prob = args.prob
    logger.info(f"Using prob parameter: {prob}")

    # Ensure tensors are on CPU for DataLoader compatibility
    s_train = s_train.cpu() if s_train.is_cuda else s_train
    t_train = t_train.cpu() if t_train.is_cuda else t_train

    # s_train is already flat [n, 4] from generate_data, use directly (matching Model/model.py)
    # Do NOT clone/detach to preserve tensor identity for DataLoader shuffle determinism
    s_train_flat = s_train

    # Get number of bodies
    num_bodies = s_train.shape[1] if s_train.dim() == 3 else s_train_flat.shape[-1] // 2

    # Calculate truncation
    num_total_points = s_train_flat.shape[0]
    trunc = num_total_points // prob

    if trunc >= num_total_points // 3:
        logger.warning(f"Boundary size too large with prob={prob}, adjusting")
        trunc = num_total_points // 4
        prob = num_total_points // trunc

    logger.info(f"Boundary size for hybrid method: {trunc}, Total points: {num_total_points}")

    # Initialize target accelerations on CPU
    target_accelerations = torch.zeros((num_total_points, num_bodies), device='cpu')

    # Calculate target accelerations (EXACT COPY from lines 360-458)
    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < s_train_flat.shape[-1]:
            if s_train.dim() == 3:
                velocity_train = s_train[:, body_idx, 1]
            else:
                velocity_train = s_train_flat[:, velocity_idx]

            if use_hybrid:
                logger.info(f"Body {body_idx}: Calculating FFT for entire training set")
                fft_output_path = os.path.join(output_paths["results"], f"train_fft_body_{body_idx}_full.csv")
                fft_full = calculate_fft_target_derivative(velocity_train, t_train, output_csv_path=fft_output_path)

                if fft_full is None:
                    logger.warning(f"FFT calculation failed for body {body_idx}, falling back to pure FD")
                    # Use order=4 to match Model/model.py (line 376, 449)
                    fd_full = estimate_temporal_gradient_finite_diff(velocity_train, t_train, order=4)
                    if fd_full is None:
                        logger.error(f"FD calculation also failed for body {body_idx}")
                        return None
                    target_accelerations[:, body_idx] = fd_full
                    continue

                target_accelerations[:, body_idx] = fft_full

                logger.info(f"Body {body_idx}: Calculating FD for boundary replacement")
                # Use order=4 to match Model/model.py (line 388)
                fd_partial = estimate_temporal_gradient_finite_diff(velocity_train, t_train, order=4)

                if fd_partial is not None:
                    end_start_idx = num_total_points - trunc
                    target_accelerations[:trunc, body_idx] = fd_partial[:trunc]
                    target_accelerations[end_start_idx:, body_idx] = fd_partial[end_start_idx:]

                    blend_length = min(10, trunc // 4)
                    for i in range(blend_length):
                        idx = trunc - blend_length // 2 + i
                        if 0 <= idx < num_total_points:
                            alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                            target_accelerations[idx, body_idx] = (
                                (1 - alpha) * fd_partial[idx] + alpha * fft_full[idx]
                            )

                    for i in range(blend_length):
                        idx = end_start_idx - blend_length // 2 + i
                        if 0 <= idx < num_total_points:
                            alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                            target_accelerations[idx, body_idx] = (
                                alpha * fd_partial[idx] + (1 - alpha) * fft_full[idx]
                            )

                    logger.info(f"Body {body_idx}: Hybrid acceleration computed")
            else:
                logger.info(f"Body {body_idx}: Calculating pure FD for entire training set")
                # Use order=4 to match main_fnode.py default (fnode_target_fd_order=4)
                fd_full = estimate_temporal_gradient_finite_diff(velocity_train, t_train, order=4)
                if fd_full is None:
                    logger.error(f"FD calculation failed for body {body_idx}")
                    return None
                target_accelerations[:, body_idx] = fd_full

    # Model input preparation
    if model.d_interest == 0:
        model_input = s_train_flat
    else:
        time_feature = t_train.unsqueeze(-1)
        model_input = torch.cat([s_train_flat, time_feature], dim=-1)

    logger.info(f"Model input shape: {model_input.shape}, Expected input dim: {model.dim_input}")

    # Training parameters
    num_epochs = args.fnode_epochs
    criterion = torch.nn.MSELoss()

    best_loss_train = float('inf')
    best_epoch_train = -1
    patience_counter = 0
    early_stopped = False

    # Create DataLoader (EXACT COPY from lines 558-567)
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(model_input, target_accelerations)
    use_cuda = current_device.type == 'cuda'
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=use_cuda, persistent_workers=False)

    logger.info(f"Training with {len(dataset)} samples")
    logger.info(f"Starting training loop with {num_epochs} epochs")

    # ===== TRAINING LOOP - EXACT COPY from lines 589-664 =====
    tracking_data = []
    cumulative_time = 0.0
    start_time_train = time.time()

    # Prepare test data for inference (ensure same device)
    s_test_for_cat = s_test.cpu() if s_test.is_cuda else s_test
    t_test_for_cat = t_test.cpu() if t_test.is_cuda else t_test
    s_full = torch.cat([s_train, s_test_for_cat], dim=0)
    t_full = torch.cat([t_train, t_test_for_cat], dim=0)
    s_full_flat = s_full.reshape(s_full.shape[0], -1)
    best_test_mse = float('inf')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase (EXACT COPY)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(current_device)
            batch_target = batch_target.to(current_device)

            optimizer.zero_grad()

            predicted_accelerations = model(batch_input)
            loss = criterion(predicted_accelerations, batch_target)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            else:
                logger.warning(f"Non-finite loss at epoch {epoch + 1}, batch {num_batches}")

            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch + 1:>5}/{num_epochs} | "
                       f"Train Loss: {avg_train_loss:.6e} | "
                       f"Time: {(time.time() - epoch_start_time):.2f}s | LR: {current_lr:.2e}")

        if scheduler:
            scheduler.step()

        # Save best model
        if avg_train_loss < best_loss_train:
            best_loss_train = avg_train_loss
            best_epoch_train = epoch
            save_model_state(model, output_paths["model"], model_filename="FNODE_best.pkl")
            patience_counter = 0

        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time

        # ===== ADD TEST INFERENCE (does not affect training) =====
        test_mse = None
        if not args.skip_test_inference and args.test_freq > 0 and (epoch + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                test_params = {
                    'ode_solver_params': {
                        'method': 'rk4',
                        'rtol': 1e-7,
                        'atol': 1e-9
                    }
                }

                # For test inference, start from FIRST training point (to predict full trajectory)
                # NOT last training point, to match how other models are tested
                s0 = s_train_flat[0:1].to(device)
                test_trajectory = test_fnode(
                    model=model,
                    s0_test_core_state=s0,
                    t_test_eval_times=t_full.to(device),
                    test_params=test_params,
                    output_paths=output_paths
                )

                if test_trajectory is not None:
                    # Ensure length alignment between prediction and ground truth
                    min_len = min(test_trajectory.shape[0], s_full_flat.shape[0])
                    test_trajectory_aligned = test_trajectory[:min_len].cpu()
                    s_full_flat_aligned = s_full_flat[:min_len].cpu()

                    # Calculate MSE based on user preference
                    if args.use_full_traj_mse:
                        # Full trajectory MSE (train + test)
                        test_mse = torch.nn.MSELoss()(test_trajectory_aligned, s_full_flat_aligned).item()
                    else:
                        # Test-only MSE (extrapolation region only)
                        train_len = len(s_train)
                        if min_len > train_len:
                            test_pred = test_trajectory_aligned[train_len:]
                            test_gt = s_full_flat_aligned[train_len:]
                            test_mse = torch.nn.MSELoss()(test_pred, test_gt).item()
                        else:
                            test_mse = float('nan')

                    if test_mse < best_test_mse:
                        best_test_mse = test_mse

        tracking_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_mse': test_mse,
            'epoch_time': epoch_time,
            'cumulative_time': cumulative_time,
            'learning_rate': current_lr
        })

    training_duration = time.time() - start_time_train
    logger.info(f"--- FNODE Training Finished: Total Time {training_duration:.2f}s ---")
    logger.info(f"Method used: {method_name}")

    # Save tracking data
    df = pd.DataFrame(tracking_data)
    df.to_csv(os.path.join(output_paths['results'], 'FNODE_test_tracking.csv'), index=False)
    logger.info(f"FNODE training complete. Best test MSE: {best_test_mse:.6e}")

    return df

def train_lstm_with_tracking(args, s_train, s_test, t_train, t_test, device):
    """Train LSTM with epoch-wise test MSE tracking"""
    logger.info("="*80)
    logger.info(f"TRAINING LSTM ON {args.test_case}")
    logger.info("="*80)

    # Setup
    output_paths = get_output_paths(args.test_case, 'LSTM')
    os.makedirs(output_paths['results'], exist_ok=True)
    os.makedirs(output_paths['model'], exist_ok=True)

    # Determine number of bodies from data shape
    num_bodies = s_train.shape[1] // 2

    # Model
    seq_len = 16
    model = LSTMModel(num_body=num_bodies, hidden_size=256, num_layers=3, dropout_rate=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # Prepare training sequences - ensure on CPU for DataLoader
    s_train_flat = s_train.cpu().reshape(len(s_train), -1)  # [time, 4]
    sequences = []
    targets = []

    for i in range(len(s_train_flat) - seq_len):
        sequences.append(s_train_flat[i:i+seq_len])
        targets.append(s_train_flat[i+seq_len])

    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    # Create DataLoader (matching Model/model.py lines 685-692)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(sequences, targets)
    num_workers = 0
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=num_workers, pin_memory=use_cuda)

    # Prepare test data
    s_full = torch.cat([s_train, s_test], dim=0)
    t_full = torch.cat([t_train, t_test], dim=0)

    # Training loop - matching Model/model.py lines 712-752
    tracking_data = []
    cumulative_time = 0.0
    best_test_mse = float('inf')

    test_params = {}

    for epoch in range(args.lstm_epochs):
        epoch_start_time = time.time()

        # Training phase - matching Model/model.py lines 716-731
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_seq, batch_target in train_loader:
            optimizer.zero_grad()
            batch_seq = batch_seq.to(device)
            batch_target = batch_target.to(device)

            # Forward pass
            prediction = model(batch_seq)
            loss = criterion(prediction, batch_target)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            else:
                logger.warning(f"Non-finite loss at epoch {epoch + 1}, batch {num_batches}")

            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        # Get current LR before stepping (matching Model/model.py)
        current_lr = optimizer.param_groups[0]['lr']

        # Scheduler step
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time

        # Test inference (if enabled) - matching main_lstm.py's rollout approach
        test_mse = None
        if not args.skip_test_inference and args.test_freq > 0 and (epoch + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                # Roll out predictions for entire trajectory (matching main_lstm.py lines 191-199)
                s_test_flat = s_test.cpu().reshape(len(s_test), -1)
                s_full_flat = torch.cat([s_train_flat, s_test_flat], dim=0)
                total_steps = len(s_full_flat)

                # Seed with first seq_len steps from s_train (actual ground truth)
                predictions_list = [s_train_flat[i].unsqueeze(0) for i in range(seq_len)]

                # Roll out predictions using predicted states as input (accumulates error)
                for step in range(seq_len, total_steps):
                    input_seq = torch.stack(predictions_list[-seq_len:], dim=1).to(device)
                    prediction = model(input_seq)
                    predictions_list.append(prediction.cpu())

                predictions = torch.cat(predictions_list, dim=0)

                # Ensure length alignment
                min_len = min(predictions.shape[0], s_full_flat.shape[0])
                predictions_aligned = predictions[:min_len]
                s_full_flat_aligned = s_full_flat[:min_len]

                # Calculate MSE based on user preference
                train_region_len = len(s_train_flat)
                if min_len > train_region_len:
                    if args.use_full_traj_mse:
                        # Full trajectory MSE (train + test)
                        test_mse = nn.MSELoss()(predictions_aligned, s_full_flat_aligned).item()
                    else:
                        # Test-only MSE (extrapolation region only)
                        test_mse = nn.MSELoss()(
                            predictions_aligned[train_region_len:],
                            s_full_flat_aligned[train_region_len:]
                        ).item()

                    if test_mse < best_test_mse:
                        best_test_mse = test_mse
                        torch.save(model.state_dict(), os.path.join(output_paths['model'], 'LSTM_best.pkl'))

        tracking_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_mse': test_mse,
            'epoch_time': epoch_time,
            'cumulative_time': cumulative_time,
            'learning_rate': current_lr  # LR before stepping
        })

        if (epoch + 1) % 10 == 0:
            test_mse_str = f"{test_mse:.6e}" if test_mse is not None else "N/A"
            logger.info(f"LSTM Epoch {epoch+1}/{args.lstm_epochs}: "
                       f"Train Loss={avg_train_loss:.6e}, Test MSE={test_mse_str}, "
                       f"Time={cumulative_time/60:.2f}min")

    # Save tracking data
    df = pd.DataFrame(tracking_data)
    df.to_csv(os.path.join(output_paths['results'], 'LSTM_test_tracking.csv'), index=False)
    logger.info(f"LSTM training complete. Best test MSE: {best_test_mse:.6e}")

    return df


def train_fcnn_with_tracking(args, s_train, s_test, t_train, t_test, device):
    """Train FCNN with epoch-wise test MSE tracking"""
    logger.info("="*80)
    logger.info(f"TRAINING FCNN ON {args.test_case}")
    logger.info("="*80)

    # Setup
    output_paths = get_output_paths(args.test_case, 'FCNN')
    os.makedirs(output_paths['results'], exist_ok=True)
    os.makedirs(output_paths['model'], exist_ok=True)

    # Determine number of bodies from data shape
    num_bodies = s_train.shape[1] // 2

    # Model
    model = FCNN(num_bodys=num_bodies, layers=3, width=256, d_input_interest=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # FCNN uses ExponentialLR (matching main_fcnn.py line 83)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # Prepare training data (time -> state)
    train_inputs = t_train.unsqueeze(1)  # [time, 1]
    train_targets = s_train.reshape(len(s_train), -1)  # [time, 4]

    # Create DataLoader (matching Model/model.py line 922-923)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_inputs, train_targets)
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=use_cuda, persistent_workers=False)

    # Prepare test data
    s_full = torch.cat([s_train, s_test], dim=0)
    t_full = torch.cat([t_train, t_test], dim=0)

    # Training loop
    tracking_data = []
    cumulative_time = 0.0
    best_test_mse = float('inf')

    for epoch in range(args.fcnn_epochs):
        epoch_start = time.time()

        # Training (using DataLoader like main_fcnn.py)
        model.train()
        epoch_loss = 0.0

        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            prediction = model(batch_input)
            loss = criterion(prediction, batch_target)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Get current LR before stepping (matching Model/model.py)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        epoch_time = time.time() - epoch_start
        cumulative_time += epoch_time

        # Test inference (if enabled) - matching main_fcnn.py's full trajectory approach
        test_mse = None
        if not args.skip_test_inference and args.test_freq > 0 and (epoch + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                # Predict on full trajectory (matching main_fcnn.py)
                s_train_flat_fcnn = s_train.reshape(len(s_train), -1)
                s_test_flat_fcnn = s_test.reshape(len(s_test), -1)
                s_full_flat = torch.cat([s_train_flat_fcnn, s_test_flat_fcnn], dim=0)
                predictions = model(t_full.unsqueeze(1).to(device)).cpu()

                # Ensure length alignment
                min_len = min(predictions.shape[0], s_full_flat.shape[0])
                predictions_aligned = predictions[:min_len]
                s_full_flat_aligned = s_full_flat[:min_len]

                # Calculate MSE based on user preference
                train_region_len = len(s_train)
                if min_len > train_region_len:
                    if args.use_full_traj_mse:
                        # Full trajectory MSE (train + test)
                        test_mse = nn.MSELoss()(predictions_aligned, s_full_flat_aligned).item()
                    else:
                        # Test-only MSE (extrapolation region only)
                        test_mse = nn.MSELoss()(
                            predictions_aligned[train_region_len:],
                            s_full_flat_aligned[train_region_len:]
                        ).item()

                    if test_mse < best_test_mse:
                        best_test_mse = test_mse
                        torch.save(model.state_dict(), os.path.join(output_paths['model'], 'FCNN_best.pkl'))

        tracking_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_mse': test_mse,
            'epoch_time': epoch_time,
            'cumulative_time': cumulative_time,
            'learning_rate': current_lr  # LR before stepping
        })

        if (epoch + 1) % 10 == 0:
            test_mse_str = f"{test_mse:.6e}" if test_mse is not None else "N/A"
            logger.info(f"FCNN Epoch {epoch+1}/{args.fcnn_epochs}: "
                       f"Train Loss={avg_train_loss:.6e}, Test MSE={test_mse_str}, "
                       f"Time={cumulative_time/60:.2f}min")

    # Save tracking data
    df = pd.DataFrame(tracking_data)
    df.to_csv(os.path.join(output_paths['results'], 'FCNN_test_tracking.csv'), index=False)
    logger.info(f"FCNN training complete. Best test MSE: {best_test_mse:.6e}")

    return df


def plot_comparison(mbdnode_df, fnode_df, lstm_df, fcnn_df, test_case):
    """Generate Test MSE vs Training Time comparison plots (both minutes and seconds)"""
    logger.info("="*80)
    logger.info("GENERATING COMPARISON PLOTS")
    logger.info("="*80)

    # Create output directory
    comparison_dir = os.path.join(os.getcwd(), 'figures', test_case, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Plot each model
    models_data = [
        ('MBDNODE', mbdnode_df, 'blue', 'o'),
        ('FNODE', fnode_df, 'green', 's'),
        ('LSTM', lstm_df, 'red', '^'),
        ('FCNN', fcnn_df, 'orange', 'D')
    ]

    # Generate both plots: minutes and seconds
    for time_unit in ['minutes', 'seconds']:
        plt.figure(figsize=(12, 8))

        for model_name, df, color, marker in models_data:
            # Skip if DataFrame is None (model was skipped)
            if df is None:
                continue
            # Filter out None values
            valid_data = df[df['test_mse'].notna()]

            # Convert time based on unit
            if time_unit == 'minutes':
                time_data = valid_data['cumulative_time'] / 60
                xlabel = 'Cumulative Training Time (minutes)'
                filename = 'test_mse_vs_time_all_models.png'
            else:  # seconds
                time_data = valid_data['cumulative_time']
                xlabel = 'Cumulative Training Time (seconds)'
                filename = 'test_mse_vs_time_all_models_seconds.png'

            test_mse = valid_data['test_mse']

            # Get final test MSE
            final_mse = test_mse.iloc[-1] if len(test_mse) > 0 else float('nan')

            plt.plot(time_data, test_mse,
                    label=f'{model_name} (final: {final_mse:.4f})',
                    color=color, marker=marker, markevery=max(1, len(time_data)//20),
                    linewidth=2, markersize=6, alpha=0.8)

        plt.xlabel(xlabel, fontsize=14, fontweight='bold')
        plt.ylabel('Test MSE', fontsize=14, fontweight='bold')
        plt.title(f'Test MSE vs Training Time - All Models ({test_case})',
                  fontsize=16, fontweight='bold', pad=20)
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.tight_layout()

        # Save
        output_path = os.path.join(comparison_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot ({time_unit}) saved to: {output_path}")
        plt.close()


def main():
    args = parse_arguments()

    logger.info("="*80)
    logger.info("EPOCH-WISE TEST TRACKING FOR ALL MODELS")
    logger.info("="*80)
    # Determine which models to run (all enabled by default unless skipped)
    run_mbdnode = not args.skip_mbdnode
    run_fnode = not args.skip_fnode
    run_lstm = not args.skip_lstm
    run_fcnn = not args.skip_fcnn
    enable_test = not args.skip_test_inference

    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Train ratio: {args.train_ratio}")
    logger.info(f"Test inference: {'Enabled' if enable_test else 'Disabled'} (freq={args.test_freq})")
    logger.info(f"Models to run: MBDNODE={run_mbdnode}, FNODE={run_fnode}, "
               f"LSTM={run_lstm}, FCNN={run_fcnn}")
    logger.info(f"Epochs: MBDNODE={args.mbdnode_epochs}, FNODE={args.fnode_epochs}, "
               f"LSTM={args.lstm_epochs}, FCNN={args.fcnn_epochs}")

    # Set seed
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[TRACKING] Device: {device}, CUDA available: {torch.cuda.is_available()}", flush=True)

    # Generate/load data
    print("[DEBUG] About to call generate_data", flush=True)
    s_train, s_test, t_train, t_test = generate_data(args)
    print(f"[DEBUG] Data loaded: s_train.shape={s_train.shape}", flush=True)

    # Determine number of bodies from data shape
    num_bodies = s_train.shape[1] // 2  # Each body has 2 states (position, velocity)
    logger.info(f"Detected {num_bodies} bodies from data shape {s_train.shape}")

    # Train models based on control flags
    mbdnode_df = None
    fnode_df = None
    lstm_df = None
    fcnn_df = None

    if run_mbdnode:
        # Reset seed before MBDNODE to match main_mbdnode.py
        set_seed(args.seed)
        mbdnode_df = train_mbdnode_with_tracking(args, s_train, s_test, t_train, t_test, device)
    else:
        logger.info("Skipping MBDNODE (disabled)")

    if run_fnode:
        # Reset seed before FNODE to match main_fnode.py (CRITICAL for alignment!)
        set_seed(args.seed)
        # Create FNODE model and optimizer in main (matching main_fnode.py defaults EXACTLY)
        print("[DEBUG] Creating FNODE model", flush=True)
        fnode_model = FNODE(num_bodys=num_bodies, layers=3, width=256, activation='tanh', initializer='xavier').to(device)
        first_param = next(fnode_model.parameters())
        print(f"[TRACKING] Initial model weight[0]={first_param.data.flatten()[0].item():.15f}", flush=True)
        print("[DEBUG] Creating optimizer", flush=True)
        fnode_optimizer = optim.Adam(fnode_model.parameters(), lr=0.001, weight_decay=5e-5)
        print("[DEBUG] Creating scheduler", flush=True)
        #fnode_scheduler = optim.lr_scheduler.StepLR(fnode_optimizer, step_size=10, gamma=0.98)
        fnode_scheduler = optim.lr_scheduler.ExponentialLR(fnode_optimizer, gamma=0.98)

        print("[DEBUG] Calling train_fnode_with_tracking", flush=True)
        fnode_df = train_fnode_with_tracking(args, s_train, s_test, t_train, t_test, device,
                                             fnode_model, fnode_optimizer, fnode_scheduler)
        print("[DEBUG] FNODE training completed", flush=True)
    else:
        logger.info("Skipping FNODE (disabled)")

    if run_lstm:
        # Reset seed before LSTM to match main_lstm.py
        set_seed(args.seed)
        lstm_df = train_lstm_with_tracking(args, s_train, s_test, t_train, t_test, device)
    else:
        logger.info("Skipping LSTM (disabled)")

    if run_fcnn:
        # Reset seed before FCNN to match main_fcnn.py
        set_seed(args.seed)
        fcnn_df = train_fcnn_with_tracking(args, s_train, s_test, t_train, t_test, device)
    else:
        logger.info("Skipping FCNN (disabled)")

    # Generate comparison plot only if at least 2 models were run
    run_models = [df for df in [mbdnode_df, fnode_df, lstm_df, fcnn_df] if df is not None]
    if len(run_models) >= 2:
        plot_comparison(mbdnode_df, fnode_df, lstm_df, fcnn_df, args.test_case)
    else:
        logger.info("Skipping comparison plot (need at least 2 models)")

    logger.info("="*80)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\nResults saved to:")
    logger.info(f"  - results/{args.test_case}/{{MODEL}}/{{MODEL}}_test_tracking.csv")
    logger.info(f"  - figures/{args.test_case}/comparison/test_mse_vs_time_all_models.png (minutes)")
    logger.info(f"  - figures/{args.test_case}/comparison/test_mse_vs_time_all_models_seconds.png (seconds)")


if __name__ == '__main__':
    main()
