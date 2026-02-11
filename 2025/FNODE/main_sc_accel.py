"""
Main script for training FNODE models with different number of layers
to learn slider-crank accelerations directly (without integration).
Uses c=0.6 for friction coefficient.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
from typing import List, Tuple, Dict

# Import custom modules
from Model.model import FNODE
from Model.Data_generator import generate_slider_crank_dataset_with_friction
from Model.utils import estimate_temporal_gradient_finite_diff

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def generate_data_with_friction(c_friction: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate slider-crank data with specified friction coefficient.

    Args:
        c_friction: Friction coefficient (default 0.6)

    Returns:
        time: Time vector
        state_data: State data [theta, omega]
        fd_accelerations: FD computed accelerations (higher order, noisier)
        true_accelerations: TRUE ground truth accelerations from simulation
    """
    print(f"\nGenerating slider-crank data with c={c_friction}...")

    # Hyperparameters from main_fnode_fric.py
    T_span = 5.0  # Total simulation time
    dt_sample = 0.01  # Sampling timestep
    fd_order = 6  # Finite difference order for noisy FD acceleration

    # Generate data using the existing function - NOW returns true accelerations too!
    state_data, time, true_accelerations = generate_slider_crank_dataset_with_friction(
        c_slide=c_friction,
        time_span=T_span,
        dt=1e-3  # Internal timestep, function returns data at dt=0.01
    )

    # state_data shape: (num_timesteps, 2) where columns are [theta, omega]
    print(f"Generated data shape: {state_data.shape}")
    print(f"Time span: {time[0]:.3f} to {time[-1]:.3f} seconds")
    print(f"TRUE acceleration data shape: {true_accelerations.shape}")

    # Extract accelerations using finite differences for FD comparison
    velocities = state_data[:, 1]  # omega is the second column

    # Use higher order finite difference for FD acceleration (noisier)
    fd_accelerations = estimate_temporal_gradient_finite_diff(
        velocities, time, order=fd_order, smooth_boundaries=False
    )

    # Convert true accelerations to numpy if it's a tensor
    if torch.is_tensor(true_accelerations):
        true_accelerations = true_accelerations.numpy()

    print(f"FD Acceleration shape (order {fd_order}): {fd_accelerations.shape}")
    print(f"Ground Truth Acceleration shape (from simulation): {true_accelerations.shape}")

    return time, state_data, fd_accelerations.reshape(-1, 1), true_accelerations.reshape(-1, 1)


def create_dataloader(state_data: np.ndarray, accelerations: np.ndarray,
                      batch_size: int = 64) -> DataLoader:
    """
    Create PyTorch DataLoader for training.

    Args:
        state_data: Input states [theta, omega]
        accelerations: Target accelerations
        batch_size: Batch size for training

    Returns:
        DataLoader object
    """
    # Convert to PyTorch tensors
    X = torch.FloatTensor(state_data)
    y = torch.FloatTensor(accelerations)

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


def train_fnode_model(model: FNODE, dataloader: DataLoader,
                      num_epochs: int = 500, lr: float = 5e-3,
                      device: str = 'cuda', num_layers: int = None) -> Dict:
    """
    Train a single FNODE model to predict accelerations.

    Args:
        model: FNODE model instance
        dataloader: Training data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to use for training
        num_layers: Number of layers in the model (for logging)

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    # Optimizer settings from main_fnode_fric.py
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Loss function
    criterion = nn.MSELoss()

    # Training history
    history = {'loss': [], 'epoch_times': []}

    # Early stopping parameters (disabled by default as in main_fnode_fric.py)
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    use_early_stopping = False

    layers_str = f" with {num_layers} layers" if num_layers else ""
    print(f"\nTraining model{layers_str}...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            pred_accel = model(batch_X)
            loss = criterion(pred_accel, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        # Update learning rate
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        history['loss'].append(avg_loss)
        history['epoch_times'].append(epoch_time)

        # Print progress every epoch (as in main_fnode_fric.py with output_time_log=1)
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f}s")

        # Early stopping check (disabled by default)
        if use_early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    return history


def evaluate_models(models_dict: Dict, state_data: np.ndarray,
                   true_accelerations: np.ndarray, device: str = 'cuda') -> Dict:
    """
    Evaluate all trained models on the full dataset.

    Args:
        models_dict: Dictionary of trained models {num_layers: model}
        state_data: Input states
        true_accelerations: True accelerations (can be numpy array or torch tensor)
        device: Device to use

    Returns:
        Dictionary with predictions for each model
    """
    predictions = {}
    errors = {}

    # Convert inputs to proper format
    if isinstance(state_data, torch.Tensor):
        state_tensor = state_data.to(device)
    else:
        state_tensor = torch.FloatTensor(state_data).to(device)

    # Convert true_accelerations to numpy if it's a tensor
    if isinstance(true_accelerations, torch.Tensor):
        true_accelerations_np = true_accelerations.cpu().numpy()
    else:
        true_accelerations_np = true_accelerations

    for num_layers, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            pred_accel = model(state_tensor).cpu().numpy()
            predictions[num_layers] = pred_accel

            # Calculate MSE
            mse = np.mean((pred_accel - true_accelerations_np) ** 2)
            errors[num_layers] = mse
            print(f"Model with {num_layers} layers - MSE: {mse:.6e}")

    return predictions, errors


def plot_acceleration_comparison(time: np.ndarray, true_accel: np.ndarray,
                                fd_accel: np.ndarray, predictions: Dict,
                                save_path: str = None):
    """
    Plot acceleration comparison for all models.

    Args:
        time: Time vector
        true_accel: True accelerations from simulation
        fd_accel: FD computed accelerations
        predictions: Dictionary of predictions {num_layers: pred_accel}
        save_path: Path to save figure
    """
    # Increase default font size
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(16, 12))

    # Convert accelerations to numpy if they're tensors
    if isinstance(true_accel, torch.Tensor):
        true_accel = true_accel.cpu().numpy()
    if isinstance(fd_accel, torch.Tensor):
        fd_accel = fd_accel.cpu().numpy()

    # Ensure time is numpy array
    if isinstance(time, torch.Tensor):
        time = time.cpu().numpy()

    # Use distinct colors for predictions - custom color scheme
    # Define specific colors for each layer count
    color_map = {
        2: 'orange',
        3: 'green',
        4: 'purple',
        5: 'blue',
        6: 'red',  # Changed from default to red
        7: 'cyan',
    }

    # Plot Ground Truth (from simulation) first
    plt.plot(time, true_accel.flatten(), 'k-', linewidth=3.0, label='Ground Truth', alpha=1.0, zorder=1)

    # Plot FD acceleration
    plt.plot(time, fd_accel.flatten(), 'cyan', linestyle='-', linewidth=2.5, label='FD Acceleration', alpha=0.8, zorder=2)

    # Plot predictions on top with higher zorder and dashed lines
    for i, (num_layers, pred) in enumerate(predictions.items()):
        color = color_map.get(num_layers, 'blue')  # Default to blue if not in map
        plt.plot(time, pred.flatten(), linestyle='--', color=color, linewidth=3.5,
                label=f'FNODE ({num_layers} Hidden Layers)', alpha=0.9, zorder=i+3)

    plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
    plt.ylabel('Angular Acceleration (rad/s²)', fontsize=20, fontweight='bold')
    plt.title('FNODE Acceleration Learning Comparison (c=0.6)', fontsize=20, fontweight='bold', pad=40)

    # Move legend above title with 3 columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=20,
              frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linewidth=0.8)

    # Increase tick label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def plot_error_comparison(errors: Dict, save_path: str = None):
    """
    Plot MSE errors for different model architectures.

    Args:
        errors: Dictionary of MSE errors {num_layers: mse}
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    layers = list(errors.keys())
    mse_values = list(errors.values())

    plt.bar(layers, mse_values, color=plt.cm.tab10(np.linspace(0, 0.9, len(layers))),
            edgecolor='black', linewidth=1.5)

    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('FNODE Architecture Comparison - MSE on Acceleration Prediction', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for layer, mse in errors.items():
        plt.text(layer, mse, f'{mse:.2e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error plot saved to: {save_path}")

    plt.show()


def main():
    """Main function to orchestrate the training and evaluation."""

    # Configuration
    c_friction = 0.6  # Friction coefficient
    num_layers_list = [2, 3, 4, 5, 6]  # Different architectures to test
    skip_train = True  # Skip training and load existing models

    # Hyperparameters from main_fnode_fric.py
    batch_size = 16
    hidden_units = 256
    learning_rate = 1e-3
    num_epochs = 500
    activation = 'tanh'
    weight_init = 'xavier'

    print("=" * 80)
    print("FNODE Acceleration Learning - Slider Crank System")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Friction coefficient: {c_friction}")
    print(f"  Layer configurations: {num_layers_list}")
    print(f"  Hidden units: {hidden_units}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print("=" * 80)

    # Step 1: Generate data
    time, state_data, fd_accelerations, true_accelerations = generate_data_with_friction(c_friction)

    # Step 2: Create dataloader (using all data for training with FD accelerations)
    dataloader = create_dataloader(state_data, fd_accelerations, batch_size)

    # Step 3: Train models with different architectures
    trained_models = {}
    training_histories = {}

    # Create save directory
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    if skip_train:
        print("\n" + "="*80)
        print("Skip training mode - Loading existing models...")
        print("="*80)

        for num_layers in num_layers_list:
            model_path = save_dir / f"sc_accel_layers_{num_layers}.pth"

            if model_path.exists():
                # Create model
                model = FNODE(
                    num_bodys=1,  # Single body (crank)
                    layers=num_layers,
                    width=hidden_units,
                    activation=activation,
                    initializer=weight_init
                )

                # Load model weights
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)

                trained_models[num_layers] = model
                if 'training_history' in checkpoint:
                    training_histories[num_layers] = checkpoint['training_history']

                print(f"Loaded model with {num_layers} layers from {model_path}")
            else:
                print(f"Warning: Model file not found: {model_path}")
    else:
        for num_layers in num_layers_list:
            print(f"\n{'='*60}")
            print(f"Training FNODE with {num_layers} layers")
            print(f"{'='*60}")

            # Create model
            model = FNODE(
                num_bodys=1,  # Single body (crank)
                layers=num_layers,
                width=hidden_units,
                activation=activation,
                initializer=weight_init
            )

            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")

            # Train model
            history = train_fnode_model(
                model, dataloader,
                num_epochs=num_epochs,
                lr=learning_rate,
                device=device,
                num_layers=num_layers
            )

            # Save model
            model_path = save_dir / f"sc_accel_layers_{num_layers}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_layers': num_layers,
                'hidden_units': hidden_units,
                'training_history': history
            }, model_path)
            print(f"Model saved to: {model_path}")

            trained_models[num_layers] = model
            training_histories[num_layers] = history

    # Step 4: Evaluate all models
    print("\n" + "="*80)
    print("Evaluating all models...")
    print("="*80)

    predictions, errors = evaluate_models(trained_models, state_data, fd_accelerations, device)

    # Step 5: Plot results
    print("\nGenerating comparison plots...")

    # Create figures directory
    fig_dir = Path("figures/sc_accel_comparison")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot acceleration comparison
    plot_acceleration_comparison(
        time, true_accelerations, fd_accelerations,
        predictions,
        save_path=fig_dir / "acceleration_comparison.png"
    )

    # Plot error comparison
    plot_error_comparison(
        errors,
        save_path=fig_dir / "mse_comparison.png"
    )

    # Print summary
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"Best model: {min(errors, key=errors.get)} layers")
    print(f"Best MSE: {min(errors.values()):.6e}")
    print("\nAll MSE values:")
    for num_layers in sorted(errors.keys()):
        print(f"  {num_layers} layers: {errors[num_layers]:.6e}")

    # Plot training loss curves (only if we have training histories)
    if training_histories:
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14, 8))

        colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']

        for (num_layers, history), color in zip(training_histories.items(), colors[:len(training_histories)]):
            plt.plot(history['loss'], label=f'L={num_layers}', color=color, linewidth=2.5, alpha=0.8)

        plt.xlabel('Epoch', fontsize=18, fontweight='bold')
        plt.ylabel('Training Loss', fontsize=18, fontweight='bold')
        plt.title('Training Loss Curves for Different Architectures', fontsize=20, fontweight='bold')
        plt.legend(loc='upper right', ncol=3, fontsize=14, frameon=True, fancybox=True, shadow=True)
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linewidth=0.8)

        # Increase tick label size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.show()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()