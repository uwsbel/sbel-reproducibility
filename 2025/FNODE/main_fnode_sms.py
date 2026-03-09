#!/usr/bin/env python3
"""
FNODE_SMS - Position-only input variant for Single Mass Spring system
This script trains FNODE that takes only position x as input (not velocity)
and outputs acceleration a, specifically for Single Mass Spring system.

Key differences from main_fnode.py:
1. FNODE_SMS class uses only position x as input
2. Only supports Single_Mass_Spring test case
3. Uses FFT + trimming for acceleration computation
4. Only uses symplectic integrators

Author: FNODE_SMS Implementation
Date: 2024
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
from torch.utils.data import DataLoader, TensorDataset

# --- Logging Configuration ---
def setup_logging(log_file_path=None):
    """Setup logging with optional file output"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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

setup_logging()
logger = logging.getLogger("FNODE_SMS_Main")

# --- Add Model directory to path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    sys.path.insert(0, model_package_dir)

# --- Imports from Model package ---
try:
    from Model.Data_generator import generate_dataset
    from Model.utils import (
        set_seed, get_output_paths, save_data_pd, save_data,
        save_model_state, load_model_state,
        plot_trajectory_comparison, plot_acceleration_comparison,
        calculate_model_parameters
    )
    from Model.integrator import (
        sep_stormer_verlet_multiple_body, yoshida4_multiple_body,
        fukushima6_multiple_body
    )
    from Model.force_fun import force_sms

except ImportError as e:
    logger.error(f"Failed to import from Model package: {e}")
    sys.exit(1)


# ============================================================================
# FNODE_SMS Class - Position-only input variant
# ============================================================================
class FNODE_SMS(nn.Module):
    """
    FNODE for Single Mass Spring that uses only position x as input.

    Architecture:
    - Input: position x (dimension 1)
    - Output: acceleration a (dimension 1)
    - Hidden layers: Fully connected neural network

    This assumes the force/acceleration depends only on position (like Hooke's law)
    which is valid for conservative systems like mass-spring.
    """

    def __init__(self, layers=3, width=256, activation='tanh', initializer='xavier'):
        super(FNODE_SMS, self).__init__()

        self.layers = layers
        self.width = width
        self.activation_name = activation
        self.initializer = initializer

        # Only 1 input (position x) instead of 2 (position + velocity)
        self.input_dim = 1
        self.output_dim = 1  # acceleration

        # Build network layers
        self.net = self._build_network()

        # Apply weight initialization
        self._initialize_weights()

    def _build_network(self):
        """Build the neural network architecture"""
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, self.width))
        layers.append(self._get_activation())

        # Hidden layers
        for _ in range(self.layers - 1):
            layers.append(nn.Linear(self.width, self.width))
            layers.append(self._get_activation())

        # Output layer (no activation for regression)
        layers.append(nn.Linear(self.width, self.output_dim))

        return nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function"""
        if self.activation_name == 'relu':
            return nn.ReLU()
        elif self.activation_name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initializer == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.initializer == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=self.activation_name)
                nn.init.zeros_(m.bias)

    def forward(self, states):
        """
        Forward pass - compute acceleration from position only.

        Args:
            states: tensor [batch_size, 2] containing [position, velocity]
                    We only use position (first column)

        Returns:
            accelerations: tensor [batch_size, 1]
        """
        # Extract only position (ignore velocity)
        if states.dim() == 1:
            # Single state vector
            position = states[0:1].unsqueeze(0)  # [1, 1]
        else:
            # Batch of states
            position = states[:, 0:1]  # [batch_size, 1]

        # Pass through network
        acceleration = self.net(position)

        # Ensure output shape matches input batch dimension
        if states.dim() == 1:
            acceleration = acceleration.squeeze(0)

        return acceleration


# ============================================================================
# Training function for FNODE_SMS
# ============================================================================
def train_fnode_sms(model, s_train, t_train, target_accelerations,
                    train_params, optimizer, scheduler, output_paths):
    """
    Train FNODE_SMS model using position-only input.

    Similar to train_fnode_with_csv_targets but adapted for FNODE_SMS.
    """
    logger.info("Starting FNODE_SMS training...")

    device = next(model.parameters()).device
    model.train()

    # Prepare data
    s_train_tensor = s_train.to(device)
    target_tensor = target_accelerations.to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(s_train_tensor, target_tensor)
    batch_size = train_params.get('batch_size', 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function
    loss_type = train_params.get('fnode_loss_type', 'L2')
    if loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'L2':
        criterion = nn.MSELoss()
    elif loss_type == 'Smooth_L1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    # Training loop
    epochs = train_params.get('epochs', 500)
    grad_clip = train_params.get('grad_clip', 1.0)
    best_loss = float('inf')
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_states, batch_targets in dataloader:
            optimizer.zero_grad()

            # Forward pass - FNODE_SMS uses position only
            predictions = model(batch_states)

            # Compute loss
            loss = criterion(predictions, batch_targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Average epoch loss
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_state(model, output_paths["model"], "FNODE_SMS_best.pkl")

        # Logging
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

    # Save final model
    save_model_state(model, output_paths["model"], "FNODE_SMS_final.pkl")

    return model, loss_history


# ============================================================================
# Testing function using symplectic integrators
# ============================================================================
def test_fnode_sms_symplectic(model, s0, t_span, integrator='stormer_verlet',
                              mass=10.0):
    """
    Test FNODE_SMS using symplectic integrators.

    For symplectic integration, we need to provide dH/dq and dH/dp functions.
    Since FNODE_SMS only uses position, we have:
    - dH/dq = -model(q) * mass  (force from position)
    - dH/dp = p/mass = v         (kinetic energy derivative)
    """
    device = next(model.parameters()).device
    model.eval()

    # Initial state
    q0 = s0[0].item()  # Initial position
    p0 = s0[1].item() * mass  # Initial momentum = mass * velocity
    dt = t_span[1].item() - t_span[0].item()
    num_steps = len(t_span)

    # Trajectory storage
    trajectory = torch.zeros((num_steps, 2))
    trajectory[0, 0] = q0
    trajectory[0, 1] = p0 / mass  # Store velocity, not momentum

    # Current state
    q = torch.tensor(q0, dtype=torch.float32, device=device)
    p = torch.tensor(p0, dtype=torch.float32, device=device)

    # Integrate using selected method
    for i in range(1, num_steps):
        if integrator == 'stormer_verlet':
            # Störmer-Verlet method
            # p_{n+1/2} = p_n - dt/2 * dH/dq(q_n)
            # q_{n+1} = q_n + dt * dH/dp(p_{n+1/2})
            # p_{n+1} = p_{n+1/2} - dt/2 * dH/dq(q_{n+1})

            # Calculate acceleration at current position
            with torch.no_grad():
                state_q = torch.tensor([[q.item(), 0.0]], dtype=torch.float32, device=device)
                accel_q = model(state_q)
                # Model predicts a = F/m = -kx/m
                # For Hamiltonian: dp/dt = -dH/dq = F = ma
                force_q = mass * accel_q[0, 0]  # F = ma

            # Half step for momentum (dp/dt = F)
            p_half = p + 0.5 * dt * force_q

            # Full step for position
            q_new = q + dt * p_half / mass

            # Calculate acceleration at new position
            with torch.no_grad():
                state_q_new = torch.tensor([[q_new.item(), 0.0]], dtype=torch.float32, device=device)
                accel_q_new = model(state_q_new)
                force_q_new = mass * accel_q_new[0, 0]  # F = ma

            # Half step for momentum
            p_new = p_half + 0.5 * dt * force_q_new

            q = q_new
            p = p_new

        elif integrator == 'yoshida4':
            # Yoshida 4th order symplectic integrator
            # Coefficients for 4th order
            c1 = 1.0 / (2.0 * (2.0 - 2.0**(1.0/3.0)))
            c2 = (1.0 - 2.0**(1.0/3.0)) / (2.0 * (2.0 - 2.0**(1.0/3.0)))
            c3 = c2
            c4 = c1

            d1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
            d2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
            d3 = d1
            d4 = 0.0

            # Apply Yoshida steps
            coeffs_c = [c1, c2, c3, c4]
            coeffs_d = [d1, d2, d3, d4]

            q_temp = q.clone()
            p_temp = p.clone()

            for c, d in zip(coeffs_c, coeffs_d):
                # Update position
                q_temp = q_temp + c * dt * p_temp / mass

                # Calculate acceleration at current position
                with torch.no_grad():
                    state_temp = torch.tensor([[q_temp.item(), 0.0]], dtype=torch.float32, device=device)
                    accel_temp = model(state_temp)
                    force_temp = mass * accel_temp[0, 0]  # F = ma

                # Update momentum
                p_temp = p_temp + d * dt * force_temp

            q = q_temp
            p = p_temp

        elif integrator == 'fukushima6':
            # Fukushima 6th order symplectic integrator
            # Coefficients from Fukushima (2003)
            w1 = 0.78451361047755726382
            w2 = 0.23557321335935813368
            w3 = -1.17767998417887100695
            w4 = 1.31518632068391121888
            w5 = w3
            w6 = w2
            w7 = w1
            w0 = 1.0 - 2.0 * (w1 + w2 + w3 + w4)

            coeffs_c = [w1/2, w1/2, (w0+w2)/2, w2/2, (w2+w3)/2, w3/2,
                       (w3+w4)/2, w4/2, (w4+w5)/2, w5/2, (w5+w6)/2,
                       w6/2, (w6+w7)/2, w7/2, w7/2]
            coeffs_d = [0, w1, w2, w0, w3, w2, w4, w3, w5, w4,
                       w6, w5, w7, w6, w7]

            q_temp = q.clone()
            p_temp = p.clone()

            for c, d in zip(coeffs_c, coeffs_d):
                # Update position
                if c != 0:
                    q_temp = q_temp + c * dt * p_temp / mass

                # Calculate force at current position
                if d != 0:
                    with torch.no_grad():
                        state_temp = torch.tensor([[q_temp.item(), 0.0]], dtype=torch.float32, device=device)
                        accel_temp = model(state_temp)
                        force_temp = mass * accel_temp[0, 0]  # F = ma

                    # Update momentum
                    p_temp = p_temp + d * dt * force_temp

            q = q_temp
            p = p_temp

        else:
            raise ValueError(f"Unknown symplectic integrator: {integrator}")

        # Store trajectory (position and velocity)
        trajectory[i, 0] = q.cpu()
        trajectory[i, 1] = (p / mass).cpu()

    return trajectory


# ============================================================================
# Argument parsing
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FNODE_SMS Training - Position-only input for Single Mass Spring")

    # Fixed parameters for SMS
    parser.add_argument('--test_case', type=str, default='Single_Mass_Spring',
                       help="Fixed to Single_Mass_Spring")

    # Data generation
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Computation device")
    parser.add_argument('--generate_new_data', action='store_true', default=True,
                       help="Generate new dataset")
    parser.add_argument('--data_dt', type=float, default=0.01,
                       help="Time step for data generation")
    parser.add_argument('--data_total_steps', type=int, default=3000,
                       help="Total steps for trajectory")

    # FFT parameters for acceleration computation
    parser.add_argument('--fft_smooth_factor', type=int, default=9,
                       help="FFT smoothing factor (steps // this)")
    parser.add_argument('--use_gibbs_trim', action='store_true', default=True,
                       help="Apply Gibbs trimming to FFT results")
    parser.add_argument('--trim_smooth_window', type=int, default=21,
                       help="Smoothing window for trimming")
    parser.add_argument('--trim_mad_k', type=float, default=6.0,
                       help="MAD multiplier for threshold")
    parser.add_argument('--trim_stable_run', type=int, default=12,
                       help="Required consecutive stable samples")
    parser.add_argument('--trim_max_frac', type=float, default=0.25,
                       help="Maximum fraction to trim from each side")
    parser.add_argument('--trim_min_keep_frac', type=float, default=0.5,
                       help="Minimum fraction of signal to keep")

    # Model architecture
    parser.add_argument('--layers', type=int, default=3,
                       help="Number of hidden layers")
    parser.add_argument('--hidden_size', type=int, default=256,
                       help="Hidden layer width")
    parser.add_argument('--activation', type=str, default='tanh',
                       choices=['relu', 'tanh'],
                       help="Activation function")
    parser.add_argument('--initializer', type=str, default='xavier',
                       choices=['xavier', 'kaiming'],
                       help="Weight initializer")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                       help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                       help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help="Optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument('--lr_scheduler', type=str, default='ExponentialLR',
                       choices=['StepLR', 'ExponentialLR', 'ReduceLROnPlateau',
                               'CosineAnnealingLR', 'None'],
                       help="Learning rate scheduler")
    parser.add_argument('--grad_clip_value', type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument('--fnode_loss_type', type=str, default='L2',
                       choices=['L1', 'L2', 'Smooth_L1'],
                       help="Loss function type")
    parser.add_argument('--train_ratio', type=float, default=0.1,
                       help="Ratio of data for training")

    # Testing parameters - Only symplectic integrators
    parser.add_argument('--integrator', type=str, default='stormer_verlet',
                       choices=['stormer_verlet', 'yoshida4', 'fukushima6'],
                       help="Symplectic integrator for testing")
    parser.add_argument('--mass_value', type=float, default=10.0,
                       help="Mass value for Single Mass Spring")

    # Control flags
    parser.add_argument('--skip_train', action='store_true', default=False,
                       help="Skip training and only evaluate")

    return parser.parse_args()


# ============================================================================
# Main function
# ============================================================================
def main():
    args = parse_arguments()

    # Force Single_Mass_Spring
    args.test_case = 'Single_Mass_Spring'

    # Set random seed
    set_seed(args.seed)

    # Device configuration
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Output paths
    output_paths = {
        "figures": os.path.join(".", "figures", args.test_case, "FNODE_SMS"),
        "results": os.path.join(".", "results", args.test_case, "FNODE_SMS"),
        "model": os.path.join(".", "saved_model", args.test_case, "FNODE_SMS"),
        "log": os.path.join(".", "log", args.test_case, "FNODE_SMS")
    }

    # Create output directories
    for path_key, path_value in output_paths.items():
        os.makedirs(path_value, exist_ok=True)

    # Setup logging with file
    log_file_path = os.path.join(output_paths["log"],
                                 f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file_path)

    logger.info("="*60)
    logger.info("FNODE_SMS Training - Position-only input variant")
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")

    # ========== Data Generation/Loading ==========
    dataset_dir = os.path.join("dataset", args.test_case)
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if we need to generate data
    s_train_file = os.path.join(dataset_dir, "s_train.csv")
    t_train_file = os.path.join(dataset_dir, "t_train.csv")
    s_test_file = os.path.join(dataset_dir, "s_test.csv")
    t_test_file = os.path.join(dataset_dir, "t_test.csv")

    if args.generate_new_data or not os.path.exists(s_train_file):
        logger.info("Generating new dataset...")
        generate_dataset(
            test_case=args.test_case,
            numerical_methods="rk4",
            dt=args.data_dt,
            num_steps=args.data_total_steps
        )

    # Load data
    s_train_df = pd.read_csv(s_train_file)
    t_train_df = pd.read_csv(t_train_file)
    s_test_df = pd.read_csv(s_test_file)
    t_test_df = pd.read_csv(t_test_file)

    # Handle column names
    for df in [s_train_df, s_test_df]:
        if df.columns[0] in ["idx", "Unnamed: 0", "index"]:
            df.drop(df.columns[0], axis=1, inplace=True)

    # Convert to tensors - keep original full data
    s_train_orig = torch.tensor(s_train_df.values, dtype=torch.float32)
    t_train_orig = torch.tensor(t_train_df.values.flatten(), dtype=torch.float32)
    s_test = torch.tensor(s_test_df.values, dtype=torch.float32)
    t_test = torch.tensor(t_test_df.values.flatten(), dtype=torch.float32)

    # Fix dimension mismatch: time vectors are 1 element shorter than state vectors
    # Generate proper time vectors based on dt
    dt = args.data_dt
    if len(t_train_orig) < len(s_train_orig):
        t_train_orig = torch.arange(len(s_train_orig), dtype=torch.float32) * dt
    if len(t_test) < len(s_test):
        t_test_start = t_train_orig[-1] + dt
        t_test = torch.arange(len(s_test), dtype=torch.float32) * dt + t_test_start

    # Apply train_ratio for training subset
    num_train = int(len(s_train_orig) * args.train_ratio)
    s_train = s_train_orig[:num_train].clone()
    t_train = t_train_orig[:num_train].clone()

    logger.info(f"Data loaded: train={len(s_train)}, test={len(s_test)}")

    # ========== Compute Acceleration Targets ==========
    # For Single Mass Spring, we should use analytical accelerations
    # since FNODE_SMS only uses position as input (assumes conservative force)
    logger.info("Computing acceleration targets...")

    # For SMS, the analytical acceleration is a = -kx/m
    # where k=50, m=10 (from force_sms)
    k = 50.0
    m = args.mass_value  # Use the mass value from args

    # Compute analytical accelerations directly
    position_train = s_train[:, 0]  # First column is position
    accel_analytical = -k * position_train / m

    # Use analytical accelerations as targets
    accel_fft = accel_analytical  # Keep variable name for compatibility

    # Skip Gibbs trimming for analytical accelerations
    # (it was only needed for noisy FFT derivatives)
    if args.use_gibbs_trim:
        logger.info("Skipping Gibbs trimming (using analytical accelerations)")

    # Ensure acceleration is 2D [num_samples, 1]
    if accel_fft.dim() == 1:
        accel_fft = accel_fft.unsqueeze(-1)

    # Save acceleration targets
    accel_df = pd.DataFrame({
        'time': t_train.cpu().numpy(),
        'fft_acceleration': accel_fft.squeeze().cpu().numpy()
    })
    accel_csv_path = os.path.join(dataset_dir, "fft_accelerations_sms.csv")
    accel_df.to_csv(accel_csv_path, index=False)
    logger.info(f"Saved FFT accelerations to: {accel_csv_path}")

    # ========== Model Initialization ==========
    model = FNODE_SMS(
        layers=args.layers,
        width=args.hidden_size,
        activation=args.activation,
        initializer=args.initializer
    ).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params}, Trainable={trainable_params}")

    # ========== Training ==========
    if not args.skip_train:
        # Setup optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
        else:  # adamw
            optimizer = optim.AdamW(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)

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
        train_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'grad_clip': args.grad_clip_value,
            'fnode_loss_type': args.fnode_loss_type
        }

        # Train model
        training_start = time.time()
        model, loss_history = train_fnode_sms(
            model, s_train, t_train, accel_fft,
            train_params, optimizer, scheduler, output_paths
        )
        training_time = time.time() - training_start

        logger.info(f"Training completed in {training_time:.2f}s")

        # Save loss history
        loss_df = pd.DataFrame({'epoch': range(len(loss_history)), 'loss': loss_history})
        loss_df.to_csv(os.path.join(output_paths["results"], "loss_history.csv"), index=False)

        # Load best model for testing
        load_model_state(model, output_paths["model"], "FNODE_SMS_best.pkl", device)

    # ========== Testing with Symplectic Integrator ==========
    logger.info(f"Testing with {args.integrator} integrator...")

    # For testing, use the original full trajectory (before trimming)
    # Combine full train and test data for trajectory prediction
    s_full = torch.cat([s_train_orig, s_test], dim=0)
    t_full = torch.cat([t_train_orig, t_test], dim=0)

    # Initial condition
    s0 = s_full[0]

    # Test model
    with torch.no_grad():
        predictions = test_fnode_sms_symplectic(
            model, s0, t_full,
            integrator=args.integrator,
            mass=args.mass_value
        )

    # Calculate metrics
    # Use original train length for splitting
    train_end_idx = len(s_train_orig)

    # MSE metrics
    full_mse = torch.mean((predictions - s_full) ** 2).item()
    train_mse = torch.mean((predictions[:train_end_idx] - s_train_orig) ** 2).item()
    test_mse = torch.mean((predictions[train_end_idx:] - s_test) ** 2).item()

    logger.info(f"Results: Full MSE={full_mse:.6f}, Train MSE={train_mse:.6f}, Test MSE={test_mse:.6f}")

    # Save metrics
    metrics = {
        'full_mse': [full_mse],
        'train_mse': [train_mse],
        'test_mse': [test_mse],
        'integrator': [args.integrator],
        'mass': [args.mass_value]
    }
    save_data_pd(metrics, output_paths["results"], "test_metrics.csv")

    # Save predictions
    pred_df = pd.DataFrame({
        'time': t_full.cpu().numpy(),
        'position_pred': predictions[:, 0].cpu().numpy(),
        'velocity_pred': predictions[:, 1].cpu().numpy(),
        'position_true': s_full[:, 0].cpu().numpy(),
        'velocity_true': s_full[:, 1].cpu().numpy()
    })
    pred_df.to_csv(os.path.join(output_paths["results"], "predictions.csv"), index=False)

    # ========== Plotting ==========
    logger.info("Creating plots...")

    # Trajectory comparison plot
    plot_trajectory_comparison(
        test_case_name=args.test_case,
        model_predictions={"FNODE_SMS": predictions.unsqueeze(1)},  # Add body dimension
        ground_truth_trajectory=s_full.unsqueeze(1),
        time_vector=t_full,
        num_bodies_to_plot=1,
        num_steps_train=train_end_idx,
        output_dir=output_paths["figures"],
        base_filename="FNODE_SMS_trajectory",
        num_epochs=args.epochs
    )

    # Acceleration comparison plot
    model.eval()
    with torch.no_grad():
        # Get acceleration predictions from model using the PREDICTED trajectory
        # This shows what the model actually predicts during integration
        predicted_accels = model(predictions.to(device))

        # For plotting, we need a dictionary of model predictions
        model_accel_dict = {"FNODE_SMS": predicted_accels.cpu()}

        plot_acceleration_comparison(
            test_case_name=args.test_case,
            model_predictions_accel=model_accel_dict,
            ground_truth_trajectory=s_full,
            time_vector=t_full,
            num_bodies_to_plot=1,
            num_steps_train=train_end_idx,
            output_dir=output_paths["figures"],
            results_dir=output_paths["results"],
            selected_target_csv_path=accel_csv_path,
            num_epochs=args.epochs,
            base_filename="FNODE_SMS_acceleration"
        )

    # Energy plot for Single Mass Spring
    positions = predictions[:, 0].cpu().numpy()
    velocities = predictions[:, 1].cpu().numpy()

    # Calculate energy (assuming k=50, m=10 from force_sms)
    k = 50.0
    m = args.mass_value
    kinetic_energy = 0.5 * m * velocities**2
    potential_energy = 0.5 * k * positions**2
    total_energy = kinetic_energy + potential_energy

    # Energy plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Energy components
    time_np = t_full.cpu().numpy()
    ax1.plot(time_np[:train_end_idx], kinetic_energy[:train_end_idx],
            'b-', label='Kinetic (Train)', linewidth=1.5)
    ax1.plot(time_np[train_end_idx:], kinetic_energy[train_end_idx:],
            'b--', label='Kinetic (Test)', linewidth=1.5, alpha=0.7)
    ax1.plot(time_np[:train_end_idx], potential_energy[:train_end_idx],
            'r-', label='Potential (Train)', linewidth=1.5)
    ax1.plot(time_np[train_end_idx:], potential_energy[train_end_idx:],
            'r--', label='Potential (Test)', linewidth=1.5, alpha=0.7)
    ax1.plot(time_np[:train_end_idx], total_energy[:train_end_idx],
            'g-', label='Total (Train)', linewidth=2)
    ax1.plot(time_np[train_end_idx:], total_energy[train_end_idx:],
            'g--', label='Total (Test)', linewidth=2, alpha=0.7)

    ax1.axvline(x=time_np[train_end_idx], color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy Conservation - FNODE_SMS ({args.integrator})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy drift
    energy_drift = (total_energy - total_energy[0]) / total_energy[0] * 100
    ax2.plot(time_np[:train_end_idx], energy_drift[:train_end_idx],
            'k-', label='Train', linewidth=1.5)
    ax2.plot(time_np[train_end_idx:], energy_drift[train_end_idx:],
            'k--', label='Test', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=time_np[train_end_idx], color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy Drift (%)')
    ax2.set_title('Relative Energy Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    energy_plot_path = os.path.join(output_paths["figures"], "FNODE_SMS_energy.png")
    plt.savefig(energy_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Energy plot saved to: {energy_plot_path}")

    # Phase space plot
    plt.figure(figsize=(10, 8))
    plt.plot(s_full[:train_end_idx, 0].cpu().numpy(),
            s_full[:train_end_idx, 1].cpu().numpy(),
            'b-', label='Ground Truth (Train)', linewidth=2, alpha=0.7)
    plt.plot(s_full[train_end_idx:, 0].cpu().numpy(),
            s_full[train_end_idx:, 1].cpu().numpy(),
            'b--', label='Ground Truth (Test)', linewidth=2, alpha=0.5)
    plt.plot(positions[:train_end_idx], velocities[:train_end_idx],
            'r-', label='FNODE_SMS (Train)', linewidth=1.5)
    plt.plot(positions[train_end_idx:], velocities[train_end_idx:],
            'r--', label='FNODE_SMS (Test)', linewidth=1.5, alpha=0.7)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'Phase Space - FNODE_SMS (Position-only input)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    phase_plot_path = os.path.join(output_paths["figures"], "FNODE_SMS_phase_space.png")
    plt.savefig(phase_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Phase space plot saved to: {phase_plot_path}")

    logger.info("="*60)
    logger.info("FNODE_SMS training and evaluation completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()