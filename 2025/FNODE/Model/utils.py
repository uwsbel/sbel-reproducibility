# FNODE/Model/utils.py
# Enhanced version with pure FFT differentiation based on fft-deriv.pdf
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import pandas as pd
import logging

# Try importing SciPy components, needed for advanced signal processing
try:
    from scipy.signal import find_peaks, savgol_filter, windows
    from scipy.interpolate import interp1d

    SCIPY_AVAILABLE = True
    logging.info("SciPy library found and imported successfully.")
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy library not found. Peak finding and window functionality will be disabled.")
    find_peaks = None
    savgol_filter = None
    windows = None
    interp1d = None

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- General Utilities ---

def save_data(data, test_case, model_type, training_size, test_size, dt):
    """
    Saves numpy data to a standardized location with proper reshaping.
    """
    save_path = os.path.join("results", test_case)
    os.makedirs(save_path, exist_ok=True)

    # Convert data to numpy if needed
    if isinstance(data, torch.Tensor):
        data_np = data.clone().cpu().detach().numpy()
    else:
        data_np = np.array(data)

    # Create a standardized filename (MNODE format)
    filename = f"{model_type} for {test_case} with training_size={training_size}_num_steps_test={test_size}_dt={dt}.npy"
    full_path = os.path.join(save_path, filename)

    # Save the data
    try:
        np.save(full_path, data_np)
    except Exception as e:
        logger.error(f"Error saving data to {full_path}: {e}")


def set_seed(seed_value):
    """Sets the random seed for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to: {seed_value}")


def calculate_model_parameters(model):
    """Calculates and logs the total number of trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters in the model: {total_params:,}")
    return total_params


def get_output_paths(test_case, model_type, config_suffix=""):
    """
    Creates and returns a dictionary of standardized output paths for models, results, and figures.
    """
    # Sanitize parts to be file-system friendly
    test_case_safe = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(test_case))
    model_type_safe = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(model_type))

    # Create the paths dictionary
    paths = {
        "model": os.path.join('.', 'saved_model', test_case_safe),
        "results": os.path.join('.', 'results', test_case_safe, model_type_safe),
        "figures": os.path.join('.', 'figures', test_case_safe, model_type_safe),
    }

    # Create directories if they don't exist
    for path_val in paths.values():
        try:
            os.makedirs(path_val, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directory {path_val}: {e}")
            raise

    logging.info(f"Output paths configured for {test_case_safe}/{model_type_safe}")
    return paths


def save_model_state(model, model_path, model_filename="best_model.pkl"):
    """Saves the state dictionary of a PyTorch model."""
    os.makedirs(model_path, exist_ok=True)
    full_path = os.path.join(model_path, model_filename)
    try:
        torch.save(model.state_dict(), full_path)
    except Exception as e:
        logging.error(f"Error saving model to {full_path}: {e}")


def load_model_state(model, model_path, model_filename="best_model.pkl", current_device=None):
    """Loads a saved state dictionary into a PyTorch model."""
    full_path = os.path.join(model_path, model_filename)
    if not current_device:
        current_device = device
    try:
        if os.path.exists(full_path):
            state_dict = torch.load(full_path, map_location=current_device)
            model.load_state_dict(state_dict)
            model.to(current_device)
            logging.info(f"Model state loaded from {full_path} to device {current_device}")
            return True
        else:
            logging.warning(f"Model file not found at {full_path}")
            return False
    except Exception as e:
        logging.error(f"Error loading model from {full_path}: {e}")
        return False


def save_data_pd(data_dict, results_path, filename="results.csv"):
    """Saves data (provided as a dictionary) to a CSV file using pandas."""
    os.makedirs(results_path, exist_ok=True)
    full_path = os.path.join(results_path, filename)
    try:
        df = pd.DataFrame(data_dict)
        df.to_csv(full_path, index=False, float_format='%.8g')
        logging.info(f"Results data saved to {full_path}")
    except ValueError as ve:
        logging.error(f"Error creating DataFrame (check array lengths): {ve} in {full_path}")
    except Exception as e:
        logging.error(f"Error saving data to {full_path}: {e}")


# --- Loss Tracking Utilities ---

def save_loss_history(loss_history, output_path, filename_prefix="loss_history"):
    """
    Save loss history to CSV and NPZ formats.

    Args:
        loss_history: Dictionary containing loss history with keys like 'epochs', 'train_loss', 'val_loss'
        output_path: Directory to save the files
        filename_prefix: Prefix for the output files
    """
    os.makedirs(output_path, exist_ok=True)

    # Save to CSV for easy viewing
    csv_path = os.path.join(output_path, f"{filename_prefix}.csv")
    try:
        df = pd.DataFrame(loss_history)
        df.to_csv(csv_path, index=False, float_format='%.8g')
        logging.info(f"Loss history saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error saving loss history to CSV: {e}")

    # Save to NPZ for easy loading in Python
    npz_path = os.path.join(output_path, f"{filename_prefix}.npz")
    try:
        np.savez(npz_path, **loss_history)
        logging.info(f"Loss history saved to {npz_path}")
    except Exception as e:
        logging.error(f"Error saving loss history to NPZ: {e}")


def append_loss_to_csv(epoch_data, output_path, filename="loss_history.csv"):
    """
    Append a single epoch's loss data to CSV file.

    Args:
        epoch_data: Dictionary with keys like 'epoch', 'train_loss', 'val_loss', etc.
        output_path: Directory containing the CSV file
        filename: Name of the CSV file
    """
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, filename)

    try:
        df_new = pd.DataFrame([epoch_data])

        if os.path.exists(csv_path):
            # Append to existing file
            df_new.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.8g')
        else:
            # Create new file with header
            df_new.to_csv(csv_path, mode='w', header=True, index=False, float_format='%.8g')
    except Exception as e:
        logging.error(f"Error appending loss to CSV: {e}")


def calculate_flops_per_epoch(model, input_shape, batch_size=1):
    """
    Calculate FLOPs (Floating Point Operations) per epoch for a neural network model.

    Args:
        model: PyTorch model
        input_shape: Shape of single input sample (excluding batch dimension)
        batch_size: Batch size for training

    Returns:
        Total FLOPs per forward pass
    """
    total_flops = 0

    # Get model layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # For Linear layers: FLOPs = 2 * input_features * output_features
            # Factor of 2 accounts for multiply-accumulate operations
            in_features = module.in_features
            out_features = module.out_features
            # Each output needs in_features multiplications and (in_features-1) additions
            # Simplified to 2 * in_features * out_features
            flops = 2 * in_features * out_features
            if module.bias is not None:
                flops += out_features  # Adding bias
            total_flops += flops * batch_size

        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
            # Activation functions - approximate as 1 FLOP per element
            # Need to track the output size from previous layer
            # For simplicity, we'll estimate based on typical layer sizes
            pass  # Activation FLOPs are relatively small compared to linear layers

    return total_flops


def calculate_flops_per_epoch_enhanced(model, input_shape, batch_size=1, model_type='FCNN',
                                      sequence_length=None, num_integration_steps=1,
                                      numerical_method='rk4', include_backward=False):
    """
    Enhanced FLOPs calculation that accounts for model-specific computational costs.

    Args:
        model: PyTorch model
        input_shape: Shape of single input sample (excluding batch dimension)
        batch_size: Batch size for training
        model_type: Type of model ('FNODE', 'MBDNODE', 'LSTM', 'FCNN')
        sequence_length: For LSTM models, the sequence length
        num_integration_steps: For ODE-based models, number of integration steps
        numerical_method: Integration method ('rk4', 'fe', 'midpoint')
        include_backward: Whether to include backward pass FLOPs (approx 2-3x forward)

    Returns:
        Dictionary with:
            - base_flops: Basic network FLOPs
            - activation_flops: FLOPs from activation functions
            - model_specific_flops: Additional FLOPs from model-specific operations
            - total_flops: Total computational FLOPs
    """
    # Track different types of FLOPs
    linear_flops = 0
    activation_flops = 0
    layer_outputs = []  # Track output sizes for activation FLOPs

    # Activation function FLOPs per element
    activation_costs = {
        nn.ReLU: 1,      # max(0, x)
        nn.Tanh: 10,     # exp operations for tanh
        nn.Sigmoid: 4,   # exp operations for sigmoid
        nn.GELU: 8,      # Gaussian error linear unit
    }

    # Track layer dimensions for activation calculations
    current_size = np.prod(input_shape) if hasattr(input_shape, '__len__') else input_shape

    # Calculate base network FLOPs
    # For Sequential models, we need to traverse in order
    if hasattr(model, 'network') and isinstance(model.network, nn.Sequential):
        prev_linear_out = 0
        for module in model.network:
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features

                # MAC operations: 2 * in * out
                flops = 2 * in_features * out_features
                if module.bias is not None:
                    flops += out_features
                linear_flops += flops * batch_size

                # Store output size for next activation
                prev_linear_out = out_features

            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
                # Calculate activation FLOPs based on previous linear layer output
                for activation_type, cost in activation_costs.items():
                    if isinstance(module, activation_type):
                        if prev_linear_out > 0:
                            activation_flops += prev_linear_out * cost * batch_size
                        break
    else:
        # Fallback for non-sequential models
        modules_list = list(model.named_modules())
        prev_linear_out = 0

        for i, (name, module) in enumerate(modules_list):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features

                # MAC operations: 2 * in * out
                flops = 2 * in_features * out_features
                if module.bias is not None:
                    flops += out_features
                linear_flops += flops * batch_size

                # Store output size for next activation
                prev_linear_out = out_features
                layer_outputs.append(out_features)

            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
                # Calculate activation FLOPs based on previous linear layer output
                for activation_type, cost in activation_costs.items():
                    if isinstance(module, activation_type):
                        # Use the previous linear layer's output size
                        if prev_linear_out > 0:
                            activation_flops += prev_linear_out * cost * batch_size
                        break

    # LSTM-specific FLOPs
    lstm_flops = 0
    if model_type == 'LSTM':
        if hasattr(model, 'lstm'):
            lstm_module = model.lstm
            input_size = lstm_module.input_size
            hidden_size = lstm_module.hidden_size
            num_layers = lstm_module.num_layers

            # LSTM has 4 gates (forget, input, cell, output)
            # Each gate: (input_size + hidden_size) * hidden_size
            if sequence_length is None:
                sequence_length = 10  # Default sequence length

            lstm_flops_per_step = 4 * hidden_size * (input_size + hidden_size + 1)  # +1 for bias
            lstm_flops = lstm_flops_per_step * sequence_length * num_layers * batch_size

    # ODE solver multiplier for MBDNODE
    ode_multiplier = 1
    if model_type == 'MBDNODE':
        # Different numerical methods require different numbers of function evaluations
        if numerical_method == 'rk4':
            ode_multiplier = 4  # RK4 requires 4 function evaluations per step
        elif numerical_method == 'midpoint':
            ode_multiplier = 2  # Midpoint method requires 2 evaluations
        elif numerical_method == 'fe':
            ode_multiplier = 1  # Forward Euler requires 1 evaluation

        # Additional arithmetic operations for ODE solver
        # RK4: y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4)/6
        if numerical_method == 'rk4':
            # Per step: 6 additions, 3 multiplications, 1 division per state variable
            state_dim = np.prod(input_shape) if hasattr(input_shape, '__len__') else input_shape
            ode_arithmetic = 10 * state_dim * num_integration_steps * batch_size
        else:
            ode_arithmetic = 0
    else:
        ode_arithmetic = 0

    # Calculate total FLOPs
    base_flops = linear_flops + activation_flops

    if model_type == 'LSTM':
        model_specific_flops = lstm_flops
        total_forward_flops = base_flops + lstm_flops
    elif model_type == 'MBDNODE':
        # MBDNODE uses the network multiple times per integration step
        model_specific_flops = base_flops * (ode_multiplier - 1) * num_integration_steps + ode_arithmetic
        total_forward_flops = base_flops * ode_multiplier * num_integration_steps + ode_arithmetic
    elif model_type == 'FNODE':
        # FNODE during training just computes accelerations directly
        model_specific_flops = 0
        total_forward_flops = base_flops
    else:  # FCNN
        model_specific_flops = 0
        total_forward_flops = base_flops

    # Include backward pass if requested
    # Backward pass FLOPs breakdown:
    #   - Gradient computation: ~2x forward (∂L/∂W and ∂L/∂input for each layer)
    #   - Optimizer (Adam): ~0.5x forward (momentum, second moment, parameter update)
    #   - Total: forward + 2x + 0.5x = 3.5x forward
    if include_backward:
        backward_multiplier = 2.0  # Gradient computation
        optimizer_multiplier = 0.5  # Adam optimizer updates
        total_flops = total_forward_flops * (1 + backward_multiplier + optimizer_multiplier)
    else:
        total_flops = total_forward_flops

    return {
        'base_flops': base_flops,
        'activation_flops': activation_flops,
        'model_specific_flops': model_specific_flops,
        'forward_flops': total_forward_flops,
        'backward_flops': total_forward_flops * backward_multiplier if include_backward else 0,
        'optimizer_flops': total_forward_flops * optimizer_multiplier if include_backward else 0,
        'total_flops': total_flops,
        'ode_multiplier': ode_multiplier if model_type == 'MBDNODE' else 1
    }


def plot_loss_vs_flops(loss_history, flops_history, output_path, test_case, model_name,
                       show_plot=False, num_epochs=None):
    """
    Generate and save plot of training loss versus FLOPs.

    Args:
        loss_history: Dictionary or list containing loss values
        flops_history: Dictionary or list containing cumulative FLOPs
        output_path: Directory to save the plot
        test_case: Name of the test case (e.g., 'Double_Pendulum')
        model_name: Name of the model for the plot title
        show_plot: Whether to display the plot (default: False)
        num_epochs: Total number of epochs trained (optional)
    """
    os.makedirs(output_path, exist_ok=True)

    # Extract data
    if isinstance(loss_history, dict):
        losses = loss_history.get('train_loss', [])
    else:
        losses = loss_history

    if isinstance(flops_history, dict):
        flops = flops_history.get('cumulative_flops', [])
    else:
        flops = flops_history

    # Ensure same length
    min_len = min(len(losses), len(flops))
    losses = losses[:min_len]
    flops = flops[:min_len]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Convert FLOPs to GFLOPs for better readability
    gflops = [f / 1e9 for f in flops]

    plt.plot(gflops, losses, 'b-', linewidth=1.5, alpha=0.8)
    plt.xlabel('GFLOPs (Billion Floating Point Operations)', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)

    if num_epochs:
        plt.title(f'{test_case} - {model_name}: Training Loss vs Computational Cost\n({num_epochs} epochs)',
                 fontsize=14)
    else:
        plt.title(f'{test_case} - {model_name}: Training Loss vs Computational Cost', fontsize=14)

    plt.grid(True, alpha=0.3, linestyle=':')
    plt.yscale('log')

    # Add markers at specific checkpoints (e.g., every 10% of training)
    if len(gflops) > 10:
        checkpoint_indices = [int(len(gflops) * i / 10) for i in range(1, 11)]
        checkpoint_gflops = [gflops[i] for i in checkpoint_indices if i < len(gflops)]
        checkpoint_losses = [losses[i] for i in checkpoint_indices if i < len(losses)]
        plt.scatter(checkpoint_gflops, checkpoint_losses, c='red', s=30, alpha=0.5, zorder=5)

    # Add annotations for min loss point
    if losses:
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]
        min_loss_gflops = gflops[min_loss_idx]
        plt.annotate(f'Min Loss: {min_loss:.2e}\n@ {min_loss_gflops:.1f} GFLOPs',
                    xy=(min_loss_gflops, min_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save the plot
    plot_filename = f"{test_case}_{model_name}_loss_vs_flops"
    if num_epochs:
        plot_filename += f"_epochs_{num_epochs}"
    plot_filename += ".png"

    plot_path = os.path.join(output_path, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Loss vs FLOPs plot saved to {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_loss_curves(loss_history, output_path, model_name, show_plot=False, num_epochs=None, early_stopped=False, patience=None):
    """
    Generate and save loss curve plots.

    Args:
        loss_history: Dictionary or CSV path containing loss history
        output_path: Directory to save the plot
        model_name: Name of the model for the plot title
        show_plot: Whether to display the plot (default: False)
        num_epochs: Total number of epochs trained (optional, used for title)
        early_stopped: Whether early stopping was triggered (default: False)
        patience: Patience value used for early stopping (optional)
    """
    os.makedirs(output_path, exist_ok=True)

    # Load data if path is provided
    if isinstance(loss_history, str):
        try:
            df = pd.read_csv(loss_history)
            loss_history = df.to_dict('list')
        except Exception as e:
            logging.error(f"Error loading loss history from {loss_history}: {e}")
            return

    # Create the plot
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    epochs = loss_history.get('epoch', loss_history.get('epochs', []))
    train_loss = loss_history.get('train_loss', [])
    val_loss = loss_history.get('val_loss', [])

    if train_loss:
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1.5)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=1.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if num_epochs:
        plt.title(f'{model_name} - Training Progress ({num_epochs} epochs)')
    else:
        plt.title(f'{model_name} - Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Learning rate (if available)
    if 'learning_rate' in loss_history:
        plt.subplot(1, 2, 2)
        lr = loss_history['learning_rate']
        plt.plot(epochs, lr, 'g-', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        if num_epochs:
            plt.title(f'{model_name} - Learning Rate Schedule ({num_epochs} epochs)')
        else:
            plt.title(f'{model_name} - Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    plt.tight_layout()

    # Save the plot
    if early_stopped and patience:
        plot_path = os.path.join(output_path, f"{model_name}_loss_curves_early_stop_pat_{patience}.png")
    elif num_epochs:
        plot_path = os.path.join(output_path, f"{model_name}_loss_curves_epochs_{num_epochs}.png")
    else:
        plot_path = os.path.join(output_path, f"{model_name}_loss_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Loss curves saved to {plot_path}")

    # Always close the figure to prevent memory leaks
    plt.close()


def create_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split data into train/validation/test sets.

    Args:
        data: Input data (can be tensor, array, or tuple of tensors/arrays)
        train_ratio: Fraction of data to use for training (default 0.7)
        val_ratio: Fraction of data to use for validation (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        train_data, val_data, test_data: Split data in the same format as input
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio + val_ratio must be <= 1.0, got {train_ratio + val_ratio}")

    # Handle different data types
    if isinstance(data, (torch.Tensor, np.ndarray)):
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        # Create random indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        if isinstance(data, torch.Tensor):
            return data[train_indices], data[val_indices], data[test_indices]
        else:
            return data[train_indices], data[val_indices], data[test_indices]

    elif isinstance(data, (tuple, list)):
        # Handle multiple arrays/tensors
        train_data = []
        val_data = []
        test_data = []

        n_samples = len(data[0])
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        for item in data:
            if isinstance(item, torch.Tensor):
                train_data.append(item[train_indices])
                val_data.append(item[val_indices])
                test_data.append(item[test_indices])
            else:
                train_data.append(item[train_indices])
                val_data.append(item[val_indices])
                test_data.append(item[test_indices])

        return tuple(train_data), tuple(val_data), tuple(test_data)

    elif hasattr(data, '__len__') and hasattr(data, '__getitem__'):
        # Handle dataset-like objects (e.g., TensorDataset)
        from torch.utils.data import Subset

        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        indices = list(range(n_samples))
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_data = Subset(data, train_indices)
        val_data = Subset(data, val_indices)
        test_data = Subset(data, test_indices)

        return train_data, val_data, test_data

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def create_validation_split(data, val_ratio=0.2, seed=42):
    """
    Split data into train/validation sets (legacy function for compatibility).

    Args:
        data: Input data (can be tensor, array, or tuple of tensors/arrays)
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        train_data, val_data: Split data in the same format as input
    """
    train_ratio = 1.0 - val_ratio
    train_data, val_data, _ = create_train_val_test_split(data, train_ratio, val_ratio, seed)
    return train_data, val_data


def wrap_to_pi_torch(theta):
    """Wraps angles in a PyTorch tensor to the interval [-pi, pi)."""
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32, device=device)
    return torch.remainder(theta + torch.pi, 2 * torch.pi) - torch.pi


def wrap_to_2pi_torch(theta):
    """Wraps angles in a PyTorch tensor to the interval [0, 2*pi)."""
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32, device=device)
    return torch.remainder(theta, 2 * torch.pi)


def wrap_to_2pi_np(theta):
    """Wraps angles in a NumPy array to the interval [0, 2*pi)."""
    if not isinstance(theta, np.ndarray):
        theta = np.array(theta)
    return np.mod(theta, 2 * np.pi)


def trig_angle_loss(pred_angles, true_angles):
    """Calculates a loss based on the difference between sin/cos of angles."""
    cos_pred = torch.cos(pred_angles)
    sin_pred = torch.sin(pred_angles)
    cos_true = torch.cos(true_angles)
    sin_true = torch.sin(true_angles)
    # Mean squared error for both cosine and sine components
    loss = torch.mean((cos_pred - cos_true) ** 2 + (sin_pred - sin_true) ** 2) / 2.0
    return loss


# --- Signal Processing Utilities ---

def estimate_temporal_gradient_finite_diff(
    trajectory_component,
    time_vector,
    order=4,
    smooth_boundaries=False,
    boundary_order=None,
    strict_boundary=False,
):
    """
    Estimates the gradient of a trajectory component using finite differences.
    Enhanced version with better error handling and support for higher-order differences.

    Args:
        trajectory_component: The trajectory data
        time_vector: Time points
        order: Interior finite-difference order (1, 2, or 4)
        smooth_boundaries: Deprecated and ignored (boundary smoothing removed)
        boundary_order: Optional endpoint stencil order (1, 2, 4, 6). If None, keep legacy behavior.
        strict_boundary: If True, return None when selected boundary_order needs more samples than available.
    """
    if trajectory_component is None or time_vector is None or len(trajectory_component) < 2:
        logger.error("Invalid input for finite difference gradient estimation.")
        return None

    current_device = trajectory_component.device if isinstance(trajectory_component, torch.Tensor) else torch.device(
        'cpu')

    # Ensure tensors
    if not isinstance(trajectory_component, torch.Tensor):
        trajectory_component = torch.tensor(trajectory_component, dtype=torch.float64, device=current_device)
    if not isinstance(time_vector, torch.Tensor):
        time_vector = torch.tensor(time_vector, dtype=torch.float64, device=current_device)

    # Ensure 1D
    if trajectory_component.dim() == 2 and trajectory_component.shape[1] == 1:
        trajectory_component = trajectory_component.squeeze(-1)
    elif trajectory_component.dim() != 1:
        logger.error(f"Finite difference input must be 1D, got shape {trajectory_component.shape}")
        return None

    time_steps = len(trajectory_component)
    min_len_req = {1: 2, 2: 3, 4: 5}.get(order, 3)

    if time_steps < min_len_req:
        logger.warning(f"Finite difference Order {order} needs N>={min_len_req}, got N={time_steps}. Returning None.")
        return None

    # Check for constant time step
    dt_vals = time_vector[1:] - time_vector[:-1]
    dt_mean = dt_vals.mean().item()
    dt_std = dt_vals.std().item()

    if dt_std > 0.001 * dt_mean:
        logger.warning(f"Time vector has non-uniform spacing. Using mean dt={dt_mean:.8f}")
    dt_val = dt_mean

    if abs(dt_val) < 1e-9:
        logger.error(f"Finite difference dt is too small ({dt_val:.2e}).")
        return None

    gradient = torch.zeros_like(trajectory_component)

    # Optional endpoint-only stencil override.
    boundary_req = {1: 2, 2: 3, 4: 5, 6: 7}

    def _apply_endpoint_stencils(grad_tensor, bo, dt_local):
        if bo == 1:
            grad_tensor[0] = (trajectory_component[1] - trajectory_component[0]) / dt_local
            grad_tensor[-1] = (trajectory_component[-1] - trajectory_component[-2]) / dt_local
            return
        if bo == 2:
            grad_tensor[0] = (-3.0 * trajectory_component[0] + 4.0 * trajectory_component[1] - trajectory_component[2]) / (2.0 * dt_local)
            grad_tensor[-1] = (3.0 * trajectory_component[-1] - 4.0 * trajectory_component[-2] + trajectory_component[-3]) / (2.0 * dt_local)
            return
        if bo == 4:
            grad_tensor[0] = (
                -25.0 * trajectory_component[0] + 48.0 * trajectory_component[1] - 36.0 * trajectory_component[2]
                + 16.0 * trajectory_component[3] - 3.0 * trajectory_component[4]
            ) / (12.0 * dt_local)
            grad_tensor[-1] = (
                25.0 * trajectory_component[-1] - 48.0 * trajectory_component[-2] + 36.0 * trajectory_component[-3]
                - 16.0 * trajectory_component[-4] + 3.0 * trajectory_component[-5]
            ) / (12.0 * dt_local)
            return
        # bo == 6
        grad_tensor[0] = (
            -147.0 * trajectory_component[0] + 360.0 * trajectory_component[1] - 450.0 * trajectory_component[2]
            + 400.0 * trajectory_component[3] - 225.0 * trajectory_component[4] + 72.0 * trajectory_component[5]
            - 10.0 * trajectory_component[6]
        ) / (60.0 * dt_local)
        grad_tensor[-1] = (
            147.0 * trajectory_component[-1] - 360.0 * trajectory_component[-2] + 450.0 * trajectory_component[-3]
            - 400.0 * trajectory_component[-4] + 225.0 * trajectory_component[-5] - 72.0 * trajectory_component[-6]
            + 10.0 * trajectory_component[-7]
        ) / (60.0 * dt_local)

    try:
        if order == 2:
            # Central difference for interior points
            gradient[1:-1] = (trajectory_component[2:] - trajectory_component[:-2]) / (2 * dt_val)

            # Progressive order increase at boundaries to avoid spikes
            # At t=0: Use 1st order forward difference (most stable)
            gradient[0] = (trajectory_component[1] - trajectory_component[0]) / dt_val

            # At t=end: Use 1st order backward difference for symmetry
            gradient[-1] = (trajectory_component[-1] - trajectory_component[-2]) / dt_val
        elif order == 4:
            if time_steps >= 5:
                # Central difference (4th order) for interior points
                gradient[2:-2] = (-trajectory_component[4:] + 8 * trajectory_component[3:-1] - 8 * trajectory_component[
                                                                                                   1:-3] + trajectory_component[
                                                                                                           :-4]) / (
                                             12 * dt_val)

                # Progressive order increase at boundaries
                # At t=0: Use 1st order forward difference (most stable)
                gradient[0] = (trajectory_component[1] - trajectory_component[0]) / dt_val

                # At t=dt: Use 2nd order central difference
                gradient[1] = (trajectory_component[2] - trajectory_component[0]) / (2 * dt_val)

                # At t=end-dt: Use 2nd order central difference
                gradient[-2] = (trajectory_component[-1] - trajectory_component[-3]) / (2 * dt_val)

                # At t=end: Use 1st order backward difference
                gradient[-1] = (trajectory_component[-1] - trajectory_component[-2]) / dt_val
            else:
                logger.warning(f"Not enough points ({time_steps}) for 4th order FD, falling back to 2nd order.")
                return estimate_temporal_gradient_finite_diff(trajectory_component, time_vector, order=2)
        else:  # order == 1
            logger.warning("Using 1st order finite difference (forward/backward).")
            gradient[:-1] = (trajectory_component[1:] - trajectory_component[:-1]) / dt_val
            gradient[-1] = gradient[-2]

        # Override only endpoint formulas if boundary_order is explicitly requested.
        if boundary_order is not None:
            if boundary_order not in boundary_req:
                logger.error(f"Unsupported boundary_order={boundary_order}. Choose from 1, 2, 4, 6.")
                return None
            req_n = boundary_req[boundary_order]
            if time_steps < req_n:
                msg = (f"Boundary order {boundary_order} requires N>={req_n}, got N={time_steps}.")
                if strict_boundary:
                    logger.error(msg)
                    return None
                logger.warning(msg + " Falling back to boundary order 1.")
                _apply_endpoint_stencils(gradient, 1, dt_val)
            else:
                _apply_endpoint_stencils(gradient, boundary_order, dt_val)

    except Exception as e:
        logger.error(f"Finite difference calculation error (Order {order}): {e}")
        return None

    if smooth_boundaries:
        logger.warning("smooth_boundaries is deprecated and ignored.")

    return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


def add_high_frequency_noise(signal, dt, noise_level=0.01, freq_band=(0.3, 0.8), seed=None):
    """
    Add band-limited high-frequency noise to a signal.

    Args:
        signal: Input signal (velocity data), shape [num_steps] or [num_steps, num_bodies]
        dt: Time step
        noise_level: Amplitude of noise (relative to signal std)
        freq_band: Frequency band as fraction of Nyquist frequency, e.g. (0.3, 0.8)
        seed: Random seed for reproducibility

    Returns:
        Noisy signal with same shape as input
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Ensure tensor
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    original_shape = signal.shape
    device = signal.device

    # Work with 1D signal, reshape if needed
    if signal.dim() > 1:
        signal_flat = signal.reshape(-1)
    else:
        signal_flat = signal

    N = len(signal_flat)
    fs = 1.0 / dt  # Sampling frequency
    fN = fs / 2.0  # Nyquist frequency

    # Generate white noise in time domain
    white_noise = torch.randn_like(signal_flat)

    # FFT to frequency domain
    noise_fft = torch.fft.fft(white_noise)

    # Create frequency array
    freqs = torch.fft.fftfreq(N, dt).to(device)

    # Create band-pass filter for high frequencies
    freq_min = freq_band[0] * fN
    freq_max = freq_band[1] * fN

    # Create smooth band-pass filter using Gaussian transitions
    filter_mask = torch.zeros_like(freqs)
    transition_width = 0.05 * fN  # 5% of Nyquist for smooth transition

    for i, f in enumerate(torch.abs(freqs)):
        if freq_min <= f <= freq_max:
            filter_mask[i] = 1.0
        elif freq_min - transition_width <= f < freq_min:
            # Smooth rise
            filter_mask[i] = 0.5 * (1 + torch.cos(torch.pi * (freq_min - f) / transition_width))
        elif freq_max < f <= freq_max + transition_width:
            # Smooth fall
            filter_mask[i] = 0.5 * (1 + torch.cos(torch.pi * (f - freq_max) / transition_width))

    # Apply band-pass filter in frequency domain
    noise_fft_filtered = noise_fft * filter_mask

    # Back to time domain
    band_limited_noise = torch.fft.ifft(noise_fft_filtered).real

    # Scale noise relative to signal standard deviation
    signal_std = signal_flat.std()
    noise_std = band_limited_noise.std()
    if noise_std > 0:
        band_limited_noise = band_limited_noise * (noise_level * signal_std / noise_std)

    # Add noise to signal
    noisy_signal = signal_flat + band_limited_noise

    # Reshape back to original shape
    noisy_signal = noisy_signal.reshape(original_shape)

    return noisy_signal


def add_awgn(signal, noise_level=0.05, seed=None):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    Args:
        signal: Input signal (numpy array or torch tensor)
        noise_level: Standard deviation of the noise relative to signal std
        seed: Random seed for reproducibility

    Returns:
        noisy_signal: Signal with AWGN added
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Convert to tensor if needed
    if isinstance(signal, np.ndarray):
        signal = torch.tensor(signal, dtype=torch.float32)
        return_numpy = True
    else:
        return_numpy = False

    # Calculate noise standard deviation
    signal_std = torch.std(signal)
    noise_std = noise_level * signal_std

    # Generate AWGN
    awgn = torch.randn_like(signal) * noise_std

    # Add noise to signal
    noisy_signal = signal + awgn

    # Convert back to numpy if needed
    if return_numpy:
        return noisy_signal.numpy()
    else:
        return noisy_signal


def design_lowpass_filter(N, dt, cutoff_hz, filter_type='gaussian'):
    """
    Design a lowpass filter in frequency domain.

    Args:
        N: Number of frequency points
        dt: Time step
        cutoff_hz: Cutoff frequency in Hz
        filter_type: 'gaussian', 'butterworth', or 'hard'

    Returns:
        Filter coefficients in frequency domain
    """
    fs = 1.0 / dt
    freqs = torch.fft.fftfreq(N, dt)

    if filter_type == 'gaussian':
        # Gaussian filter: H(f) = exp(-0.5 * (f/fc)^2)
        H = torch.exp(-0.5 * (freqs / cutoff_hz) ** 2)

    elif filter_type == 'butterworth':
        # 2nd order Butterworth approximation
        order = 2
        H = 1.0 / (1.0 + (torch.abs(freqs) / cutoff_hz) ** (2 * order))

    elif filter_type == 'hard':
        # Hard cutoff (not recommended due to Gibbs phenomenon)
        H = (torch.abs(freqs) <= cutoff_hz).float()

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return H


def fft_derivative_with_lowpass(signal, dt, cutoff_hz=10.0, filter_type='gaussian', derivative_order=1):
    """
    Calculate derivative using FFT with simultaneous lowpass filtering.
    This avoids noise amplification by filtering before/during differentiation.

    Args:
        signal: Input signal (e.g., velocity), shape [num_steps]
        dt: Time step
        cutoff_hz: Lowpass filter cutoff frequency in Hz
        filter_type: 'gaussian', 'butterworth', or 'hard'
        derivative_order: 1 for first derivative, 2 for second derivative

    Returns:
        Filtered derivative (e.g., acceleration)
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    device = signal.device
    N = len(signal)

    # FFT of signal
    signal_fft = torch.fft.fft(signal)

    # Get frequencies for derivative operator
    freqs = torch.fft.fftfreq(N, dt).to(device)

    # Derivative operator in frequency domain: (j*2*pi*f)^n
    if derivative_order == 1:
        derivative_operator = 2j * np.pi * freqs
    elif derivative_order == 2:
        derivative_operator = -(2 * np.pi * freqs) ** 2
    else:
        raise ValueError(f"Derivative order {derivative_order} not supported")

    # Design lowpass filter
    H_lowpass = design_lowpass_filter(N, dt, cutoff_hz, filter_type).to(device)

    # Apply both derivative and lowpass filter in frequency domain
    # Key: Do both operations together to avoid amplifying noise
    filtered_derivative_fft = signal_fft * derivative_operator * H_lowpass

    # Back to time domain
    filtered_derivative = torch.fft.ifft(filtered_derivative_fft).real

    return filtered_derivative


def spectral_derivative_1d(y_data, dt, derivative_order=1):
    """
    Calculate derivative using spectral method (FFT) following Algorithm 1 or 2 from fft-deriv.pdf.

    Args:
        y_data: 1D tensor of sampled function values
        dt: Time step (assumed uniform)
        derivative_order: 1 for first derivative, 2 for second derivative

    Returns:
        Derivative at sample points
    """
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data, dtype=torch.float32)

    device = y_data.device
    N = len(y_data)
    L = N * dt  # Period length

    # Step 1: Compute DFT using FFT
    Y_k = torch.fft.fft(y_data) / N  # Normalize by N as in the PDF

    # Step 2: Multiply by appropriate factors based on derivative order
    k = torch.arange(N, device=device)

    if derivative_order == 1:
        # Algorithm 1: First derivative
        # Create multiplier array
        multiplier = torch.zeros(N, dtype=torch.complex64, device=device)

        # For k < N/2
        multiplier[:N // 2] = 2j * torch.pi * k[:N // 2] / L

        # For k > N/2
        multiplier[N // 2 + 1:] = 2j * torch.pi * (k[N // 2 + 1:] - N) / L

        # For k = N/2 (if N is even), set to zero
        if N % 2 == 0:
            multiplier[N // 2] = 0

    elif derivative_order == 2:
        # Algorithm 2: Second derivative
        # Create multiplier array
        multiplier = torch.zeros(N, dtype=torch.complex64, device=device)

        # For k <= N/2
        multiplier[:N // 2 + 1] = -(2 * torch.pi * k[:N // 2 + 1] / L) ** 2

        # For k > N/2
        multiplier[N // 2 + 1:] = -(2 * torch.pi * (k[N // 2 + 1:] - N) / L) ** 2

    else:
        raise ValueError(f"Unsupported derivative order: {derivative_order}")

    # Apply multiplier
    Y_prime_k = Y_k * multiplier

    # Step 3: Inverse FFT to get derivative in physical space
    y_prime = N * torch.fft.ifft(Y_prime_k).real  # Multiply by N because we normalized earlier

    return y_prime


def minimal_oscillation_interpolation(y_samples, t_samples, t_eval):
    """
    Perform minimal-oscillation trigonometric interpolation as described in fft-deriv.pdf.
    This creates a smooth periodic interpolation that minimizes oscillations between samples.

    Args:
        y_samples: Sample values at discrete points
        t_samples: Time points where samples are taken
        t_eval: Time points where interpolation is desired

    Returns:
        Interpolated values at t_eval
    """
    if not isinstance(y_samples, torch.Tensor):
        y_samples = torch.tensor(y_samples, dtype=torch.float32)
    if not isinstance(t_samples, torch.Tensor):
        t_samples = torch.tensor(t_samples, dtype=torch.float32)
    if not isinstance(t_eval, torch.Tensor):
        t_eval = torch.tensor(t_eval, dtype=torch.float32)

    device = y_samples.device
    N = len(y_samples)
    L = t_samples[-1] - t_samples[0] + (t_samples[1] - t_samples[0])  # Assume periodic

    # Compute FFT coefficients
    Y_k = torch.fft.fft(y_samples) / N

    # Initialize interpolated values
    y_interp = torch.zeros_like(t_eval, dtype=torch.float32)

    # Add DC component
    y_interp += Y_k[0].real

    # Add frequency components up to Nyquist
    for k in range(1, N // 2):
        freq = 2 * torch.pi * k / L
        y_interp += 2 * (Y_k[k].real * torch.cos(freq * (t_eval - t_samples[0])) -
                         Y_k[k].imag * torch.sin(freq * (t_eval - t_samples[0])))

    # Handle Nyquist frequency (if N is even) - split equally between positive and negative
    if N % 2 == 0:
        k = N // 2
        freq = torch.pi * N / L
        y_interp += Y_k[k].real * torch.cos(freq * (t_eval - t_samples[0]))

    return y_interp


def calculate_fft_target_derivative(
                velocity_trajectory,
                time_vector,
                output_csv_path=None,
                *,
                deriv_smoothing_gaussian_width=None

):
    """
    Calculate derivative using FFT following paper methodology (Equations 16-22).
    Implements: Detrending → Mirror Reflection → Tukey Window → FFT → Gaussian Filter → IFFT

    Note: This version strictly follows the paper - NO zero-padding, NO Savitzky-Golay smoothing.

    Noisy-target options:
    - deriv_smoothing_gaussian_width: Gaussian smoothing bandwidth on derivative spectrum.
      If not provided, defaults to the paper-recommended sigma ~= N/20 (Eq. 21).
    """

    # Ensure tensors
    if not isinstance(velocity_trajectory, torch.Tensor):
        velocity_trajectory = torch.tensor(velocity_trajectory, dtype=torch.float32)
    if not isinstance(time_vector, torch.Tensor):
        time_vector = torch.tensor(time_vector, dtype=torch.float32)

    # Ensure both tensors are on the same device and have the same dtype
    device = velocity_trajectory.device
    time_vector = time_vector.to(device).to(velocity_trajectory.dtype)
    N = len(velocity_trajectory)
    if N < 3:
        logger.warning(f"FFT derivative needs at least 3 samples, got N={N}. Falling back to FD order=2.")
        return estimate_temporal_gradient_finite_diff(velocity_trajectory, time_vector, order=2)
    dt = (time_vector[1] - time_vector[0]).item()

    # Step 1: Remove linear trend to improve periodicity
    t_normalized = (time_vector - time_vector[0]) / (time_vector[-1] - time_vector[0])

    # Fit linear trend using least squares
    A = torch.stack([torch.ones_like(t_normalized), t_normalized], dim=1)
    coeffs = torch.linalg.lstsq(A, velocity_trajectory.unsqueeze(-1))[0].squeeze()
    linear_trend = coeffs[0] + coeffs[1] * t_normalized

    # Detrended signal
    detrended_velocity = velocity_trajectory - linear_trend

    # Step 2: Mirror reflection with cosine weights (Eq. 19)
    extension_length = max(1, min(N // 4, N - 1))
    mirror_idx = torch.arange(1, extension_length + 1, device=device, dtype=velocity_trajectory.dtype)
    tau = 0.5 * (1.0 - torch.cos(torch.pi * mirror_idx / extension_length))

    left_reflect = torch.flip(detrended_velocity[1:extension_length + 1], dims=[0]) * tau
    right_reflect = torch.flip(detrended_velocity[-extension_length - 1:-1], dims=[0]) * torch.flip(tau, dims=[0])

    # Step 3: Make the extended signal periodic using cosine tapering
    # This ensures smooth transition at the boundaries
    extended_signal = torch.cat([left_reflect, detrended_velocity, right_reflect])
    extended_N = len(extended_signal)

    # Apply Tukey (tapered cosine) window with alpha=0.2 (Eq. 20)
    alpha = 0.2
    tukey_window = torch.ones(extended_N, device=device, dtype=velocity_trajectory.dtype)
    if extended_N > 1 and alpha > 0:
        n = torch.arange(extended_N, device=device, dtype=velocity_trajectory.dtype)
        x = n / (extended_N - 1)
        left_mask = x < (alpha / 2.0)
        right_mask = x > (1.0 - alpha / 2.0)
        tukey_window[left_mask] = 0.5 * (
            1.0 + torch.cos(torch.pi * (2.0 * x[left_mask] / alpha - 1.0))
        )
        tukey_window[right_mask] = 0.5 * (
            1.0 + torch.cos(torch.pi * (2.0 * x[right_mask] / alpha - 2.0 / alpha + 1.0))
        )

    windowed_extended = extended_signal * tukey_window

    # Step 4: Apply spectral derivative (following paper exactly - NO zero-padding)
    fft_size = extended_N  # Use extended signal size directly, no padding

    # Compute FFT
    Y_k = torch.fft.fft(windowed_extended)

    # Create frequency array
    k = torch.arange(fft_size, device=device)
    L = fft_size * dt

    # Derivative multiplier (Algorithm 1 from fft-deriv.pdf)
    multiplier = torch.zeros(fft_size, dtype=torch.complex64, device=device)

    # Positive frequencies
    multiplier[:fft_size // 2] = 2j * torch.pi * k[:fft_size // 2] / L

    # Negative frequencies
    multiplier[fft_size // 2 + 1:] = 2j * torch.pi * (k[fft_size // 2 + 1:] - fft_size) / L

    # Nyquist frequency set to zero
    if fft_size % 2 == 0:
        multiplier[fft_size // 2] = 0

    # Apply derivative in frequency domain
    Y_prime_k = Y_k * multiplier

    # Apply spectral smoothing to reduce high-frequency noise (on derivative)
    # Use Gaussian filter in frequency domain.
    if deriv_smoothing_gaussian_width is None:
        freq_gaussian_width = max(1.0, N / 20.0)  # Paper Eq. (21): sigma ~= N/20
    else:
        freq_gaussian_width = float(deriv_smoothing_gaussian_width)

    if freq_gaussian_width is not None and freq_gaussian_width > 0:
        freq_indices = torch.arange(fft_size, device=device)
        gauss_filter = torch.zeros(fft_size, device=device)
        gauss_filter[:fft_size // 2] = torch.exp(-0.5 * (freq_indices[:fft_size // 2] / freq_gaussian_width) ** 2)
        gauss_filter[fft_size // 2:] = torch.exp(
            -0.5 * ((fft_size - freq_indices[fft_size // 2:]) / freq_gaussian_width) ** 2)
        Y_prime_k_filtered = Y_prime_k * gauss_filter
    else:
        Y_prime_k_filtered = Y_prime_k

    # Inverse FFT (Paper Equation 22c)
    derivative_extended = torch.fft.ifft(Y_prime_k_filtered).real

    # Step 5: Extract the original region and restore the trend derivative
    derivative_detrended = derivative_extended[extension_length:-extension_length]

    # Add back the derivative of the linear trend
    trend_derivative = coeffs[1] / (time_vector[-1] - time_vector[0])
    final_derivative = derivative_detrended + trend_derivative

    # Remove any NaN or Inf values (numerical safety)
    final_derivative = torch.nan_to_num(final_derivative, nan=0.0, posinf=0.0, neginf=0.0)

    # Save results if requested
    if output_csv_path:
        try:
            save_data = {
                'time': time_vector.cpu().numpy(),
                'velocity': velocity_trajectory.cpu().numpy(),
                'fft_derivative': final_derivative.cpu().numpy(),
                'detrended_velocity': detrended_velocity.cpu().numpy(),
                'linear_trend': linear_trend.cpu().numpy(),
                'tukey_window_sample': tukey_window[::max(1, extended_N // 1000)].cpu().numpy()
                # Subsample for visualization
            }

            df = pd.DataFrame({k: v for k, v in save_data.items() if len(v) == N})
            df.to_csv(output_csv_path, index=False, float_format='%.8g')
            logger.info(f"Saved FFT derivative (paper methodology - no zero-padding, no Savitzky-Golay) to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save FFT derivative: {e}")

    return final_derivative


def find_periodic_segments(data, time_vector, min_segment_length=30):
    """
    Find segments of data that appear to be periodic.

    Args:
        data: 1D tensor of data values
        time_vector: 1D tensor of time points
        min_segment_length: Minimum length for a valid segment

    Returns:
        List of tuples (start_idx, end_idx) for periodic segments
    """
    if not SCIPY_AVAILABLE or find_peaks is None:
        logger.warning("SciPy not available for peak finding")
        return []

    # Convert to numpy for peak finding
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)

    # Find peaks and troughs
    data_range = np.max(data_np) - np.min(data_np)
    prominence = 0.05 * data_range if data_range > 0 else 0.1

    peaks, _ = find_peaks(data_np, prominence=prominence)
    troughs, _ = find_peaks(-data_np, prominence=prominence)

    # Combine and sort all extrema
    all_extrema = np.sort(np.concatenate([peaks, troughs]))

    if len(all_extrema) < 4:  # Need at least 4 extrema for one full period
        return []

    segments = []

    # Look for segments with consistent period
    for i in range(len(all_extrema) - 3):
        # Check if we have at least 2 full periods
        segment_start = all_extrema[i]

        # Look for end points that would give us periodic behavior
        for j in range(i + 3, len(all_extrema)):
            segment_end = all_extrema[j]
            segment_length = segment_end - segment_start

            if segment_length >= min_segment_length:
                # Check if values at start and end are similar (periodicity)
                start_val = data_np[segment_start]
                end_val = data_np[segment_end]
                val_diff = abs(end_val - start_val)

                # Also check derivatives at endpoints
                if segment_start > 0 and segment_end < len(data_np) - 1:
                    dt = time_vector[1] - time_vector[0] if isinstance(time_vector, torch.Tensor) else time_vector[1] - \
                                                                                                       time_vector[0]
                    start_deriv = (data_np[segment_start + 1] - data_np[segment_start - 1]) / (2 * dt)
                    end_deriv = (data_np[segment_end + 1] - data_np[segment_end - 1]) / (2 * dt)
                    deriv_diff = abs(end_deriv - start_deriv)

                    # If both value and derivative are close, consider it periodic
                    if val_diff < 0.01 * data_range and deriv_diff < 0.1 * data_range / dt:
                        segments.append((segment_start, segment_end))
                        break

    # Remove overlapping segments (keep longer ones)
    filtered_segments = []
    for seg in segments:
        overlaps = False
        for existing in filtered_segments:
            if (seg[0] >= existing[0] and seg[0] <= existing[1]) or \
                    (seg[1] >= existing[0] and seg[1] <= existing[1]):
                overlaps = True
                break
        if not overlaps:
            filtered_segments.append(seg)

    return filtered_segments


# --- Analytical Acceleration Functions (for plotting only) ---

def calculate_analytical_accelerations_for_plotting(s_full, t_full, test_case, output_path):
    """
    Calculate analytical accelerations for plotting as ground truth.
    This is NOT used for training targets, only for visualization.
    """
    logger.info(f"Calculating analytical accelerations for plotting (test case: {test_case})")

    # Import force functions to get the analytical acceleration formulas
    try:
        from Model.force_fun import calculate_analytical_accelerations as force_calc_analytical
        # Delegate to force_fun.py's implementation with specific filename for plotting
        return force_calc_analytical(s_full, t_full, test_case, output_path, filename="analytical_accelerations_for_plotting.csv")
    except ImportError:
        logger.error("Cannot import calculate_analytical_accelerations from force_fun.py")
        return None


# --- Plotting Functions ---

def plot_trajectory_comparison(
        test_case_name,
        model_predictions,
        ground_truth_trajectory,
        time_vector,
        num_bodies_to_plot,
        num_steps_train,
        output_dir,
        base_filename="trajectory_comparison",
        plot_phase_space=True,
        plot_time_series=True,
        label_fontsize=15,
        legend_fontsize=15,
        title_fontsize=16,
        num_epochs=None,
):
    """
    Generates plots comparing predicted trajectories against ground truth.

    Args:
        num_epochs: Total number of training epochs (optional, for display in title)
    """
    os.makedirs(output_dir, exist_ok=True)
    gt_np = ground_truth_trajectory.detach().cpu().numpy()
    pred_np = {name: pred.detach().cpu().numpy() for name, pred in model_predictions.items()}
    time_np = time_vector.detach().cpu().numpy() if isinstance(time_vector, torch.Tensor) else np.array(time_vector)

    num_steps_total = gt_np.shape[0]
    num_bodies_total = gt_np.shape[1]
    num_bodies_actual_plot = min(num_bodies_to_plot, num_bodies_total)

    # Ensure compatible lengths
    for model_name, traj_pred in pred_np.items():
        if traj_pred.shape[0] != time_np.shape[0]:
            min_length = min(traj_pred.shape[0], time_np.shape[0])
            pred_np[model_name] = traj_pred[:min_length]
            time_np = time_np[:min_length]
            if gt_np.shape[0] > min_length:
                gt_np = gt_np[:min_length]
            num_steps_total = min_length

    if num_steps_train > num_steps_total:
        num_steps_train = num_steps_total

    # Time Series Plots
    if plot_time_series:
        fig_ts, axs_ts = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        ax_x = axs_ts[0]
        ax_v = axs_ts[1]

        if num_epochs:
            ax_x.set_title(f"{test_case_name}: Position vs Time ({num_epochs} epochs)", fontsize=title_fontsize)
            ax_v.set_title(f"{test_case_name}: Velocity vs Time ({num_epochs} epochs)", fontsize=title_fontsize)
        else:
            ax_x.set_title(f"{test_case_name}: Position vs Time", fontsize=title_fontsize)
            ax_v.set_title(f"{test_case_name}: Velocity vs Time", fontsize=title_fontsize)

        for i in range(num_bodies_actual_plot):
            # Ground Truth
            ax_x.plot(time_np, gt_np[:, i, 0], 'b-', label=f'Body {i + 1} (GT)', lw=1.5)
            ax_v.plot(time_np, gt_np[:, i, 1], 'b-', label=f'Body {i + 1} (GT)', lw=1.5)

            # Predictions
            for model_name, traj_pred in pred_np.items():
                # Training part
                if num_steps_train > 0:
                    ax_x.plot(time_np[:num_steps_train], traj_pred[:num_steps_train, i, 0],
                              'r--', label=f'Body {i + 1} - {model_name} (Train)' if i == 0 else "", lw=1.2, alpha=0.7)
                    ax_v.plot(time_np[:num_steps_train], traj_pred[:num_steps_train, i, 1],
                              'r--', label=f'Body {i + 1} - {model_name} (Train)' if i == 0 else "", lw=1.2, alpha=0.7)

                # Testing part
                if num_steps_train < num_steps_total:
                    ax_x.plot(time_np[num_steps_train:], traj_pred[num_steps_train:, i, 0],
                              'r:', label=f'Body {i + 1} - {model_name} (Test)' if i == 0 else "", lw=1.2, alpha=0.7)
                    ax_v.plot(time_np[num_steps_train:], traj_pred[num_steps_train:, i, 1],
                              'r:', label=f'Body {i + 1} - {model_name} (Test)' if i == 0 else "", lw=1.2, alpha=0.7)

        ax_x.set_ylabel('Position (x)', fontsize=label_fontsize)
        ax_v.set_ylabel('Velocity (v)', fontsize=label_fontsize)
        ax_x.set_xlabel('Time (t)', fontsize=label_fontsize)
        ax_v.set_xlabel('Time (t)', fontsize=label_fontsize)
        ax_x.grid(True, linestyle=':')
        ax_v.grid(True, linestyle=':')

        # Legend
        # If multiple models are plotted, keep their names in the legend.
        # Otherwise fall back to the original compact (GT/Train/Test) legend.
        if len(pred_np) <= 1:
            handles = [
                plt.Line2D([], [], color='blue', linestyle='-', label='Ground Truth'),
                plt.Line2D([], [], color='red', linestyle='--', label='Train'),
                plt.Line2D([], [], color='red', linestyle=':', label='Test')
            ]
            ax_x.legend(handles=handles, loc='best', fontsize=legend_fontsize - 2)
            ax_v.legend(handles=handles, loc='best', fontsize=legend_fontsize - 2)
        else:
            ax_x.legend(loc='best', fontsize=legend_fontsize - 4)
            ax_v.legend(loc='best', fontsize=legend_fontsize - 4)

        fig_ts.tight_layout()
        if num_epochs:
            save_path_ts = os.path.join(output_dir, f"{base_filename}_timeseries_epochs_{num_epochs}.png")
        else:
            save_path_ts = os.path.join(output_dir, f"{base_filename}_timeseries.png")
        plt.savefig(save_path_ts, dpi=200)
        logger.info(f"Time series comparison plot saved to: {save_path_ts}")
        plt.close(fig_ts)

    # Phase Space Plots
    if plot_phase_space:
        fig_ps, axs_ps = plt.subplots(1, num_bodies_actual_plot, figsize=(6 * num_bodies_actual_plot, 5), squeeze=False)

        for i in range(num_bodies_actual_plot):
            ax = axs_ps[0, i]

            # Ground Truth
            ax.plot(gt_np[:, i, 0], gt_np[:, i, 1], 'b-', label='Ground Truth', lw=1.5)

            # Predictions
            for model_name, traj_pred in pred_np.items():
                if num_steps_train > 0:
                    ax.plot(traj_pred[:num_steps_train, i, 0], traj_pred[:num_steps_train, i, 1],
                            'r--', label=f'{model_name} (Train)', lw=1.2)
                if num_steps_train < num_steps_total:
                    ax.plot(traj_pred[num_steps_train:, i, 0], traj_pred[num_steps_train:, i, 1],
                            'r:', label=f'{model_name} (Test)', lw=1.2, alpha=0.8)

            ax.set_xlabel(rf'$x_{i + 1}$', fontsize=label_fontsize)
            ax.set_ylabel(rf'$v_{i + 1}$', fontsize=label_fontsize)
            if num_epochs:
                ax.set_title(f'Body {i + 1} Phase Space ({num_epochs} epochs)', fontsize=title_fontsize)
            else:
                ax.set_title(f'Body {i + 1} Phase Space', fontsize=title_fontsize)
            ax.grid(True, linestyle=':')
            ax.legend(loc='best', fontsize=legend_fontsize - 2)

        fig_ps.tight_layout()
        if num_epochs:
            save_path_ps = os.path.join(output_dir, f"{base_filename}_phasespace_epochs_{num_epochs}.png")
        else:
            save_path_ps = os.path.join(output_dir, f"{base_filename}_phasespace.png")
        plt.savefig(save_path_ps, dpi=200)
        logger.info(f"Phase space comparison plot saved to: {save_path_ps}")
        plt.close(fig_ps)


def plot_acceleration_comparison(
        test_case_name,
        model_predictions_accel,
        ground_truth_trajectory,
        time_vector,
        num_bodies_to_plot,
        num_steps_train,
        output_dir,
        results_dir,
        selected_target_csv_path,
        base_filename="acceleration_comparison",
        label_fontsize=15,
        legend_fontsize=15,
        title_fontsize=16,
        num_epochs=None
):
    """
    Enhanced acceleration comparison plot that handles dimension mismatches properly.

    Args:
        num_epochs: Total number of training epochs (optional, for display in title)
    """
    logger.info("Generating acceleration comparison plots...")

    # Convert to numpy and get dimensions
    if isinstance(time_vector, torch.Tensor):
        time_np = time_vector.detach().cpu().numpy()
    else:
        time_np = np.array(time_vector)

    full_length = len(time_np)

    # Get ground truth trajectory shape
    if isinstance(ground_truth_trajectory, torch.Tensor):
        gt_np = ground_truth_trajectory.detach().cpu().numpy()
    else:
        gt_np = np.array(ground_truth_trajectory)

    # Initialize acceleration data dictionary
    acceleration_data = {}

    # 1. Process model predictions
    for model_name, accel_tensor in model_predictions_accel.items():
        if isinstance(accel_tensor, torch.Tensor):
            accel_np = accel_tensor.detach().cpu().numpy()
        else:
            accel_np = np.array(accel_tensor)

        if accel_np.ndim == 1:
            accel_np = accel_np[:, np.newaxis]

        logger.info(f"Model {model_name} predictions shape: {accel_np.shape}")

        # Ensure it matches full length
        if len(accel_np) != full_length:
            logger.warning(f"Model {model_name} length mismatch: {len(accel_np)} vs {full_length}")
            # If model predictions are shorter, pad with NaN
            if len(accel_np) < full_length:
                padded_accel = np.full((full_length, accel_np.shape[1]), np.nan)
                padded_accel[:len(accel_np)] = accel_np
                accel_np = padded_accel
            else:
                # If longer, truncate
                accel_np = accel_np[:full_length]

        acceleration_data[model_name] = accel_np

    # 2. Calculate or load analytical accelerations
    analytical_csv_path = os.path.join(results_dir, "analytical_accelerations_for_plotting.csv")

    # Always recalculate to ensure we have the right dimensions
    logger.info("Calculating analytical accelerations for full trajectory...")
    try:
        from Model.force_fun import calculate_analytical_accelerations

        # Make sure we're using the full trajectory
        analytical_accels = calculate_analytical_accelerations(
            gt_np[:full_length],  # Use full ground truth
            time_np[:full_length],  # Use full time vector
            test_case_name,
            results_dir
        )

        if analytical_accels is not None:
            if isinstance(analytical_accels, np.ndarray):
                if analytical_accels.ndim == 1:
                    analytical_accels = analytical_accels[:, np.newaxis]

                logger.info(f"Calculated analytical accelerations shape: {analytical_accels.shape}")

                # Ensure correct length
                if len(analytical_accels) != full_length:
                    logger.warning(f"Analytical accel length mismatch: {len(analytical_accels)} vs {full_length}")
                    if len(analytical_accels) < full_length:
                        padded_analytical = np.full((full_length, analytical_accels.shape[1]), np.nan)
                        padded_analytical[:len(analytical_accels)] = analytical_accels
                        analytical_accels = padded_analytical
                    else:
                        analytical_accels = analytical_accels[:full_length]

                acceleration_data['Analytical (GT)'] = analytical_accels
            else:
                logger.warning("Analytical accelerations returned non-array type")
        else:
            logger.warning("Failed to calculate analytical accelerations")

    except Exception as e:
        logger.error(f"Error calculating analytical accelerations: {e}")

    # 3. Load target accelerations (training portion only)
    if selected_target_csv_path and os.path.exists(selected_target_csv_path):
        try:
            target_df = pd.read_csv(selected_target_csv_path)

            # Determine target type and columns
            target_method_name = "Target"
            if "fft_target.csv" in selected_target_csv_path:
                target_method_name = "Target (FFT)"
            elif "fd_target.csv" in selected_target_csv_path:
                target_method_name = "Target (FD)"

            # Find acceleration columns
            target_cols = [col for col in target_df.columns if 'accel' in col.lower() and col != 'time']

            if target_cols:
                target_accel_np = target_df[target_cols].values
                logger.info(f"Loaded {target_method_name} shape: {target_accel_np.shape}")

                # Create full array with NaN for non-training portion
                target_full = np.full((full_length, len(target_cols)), np.nan)
                # Only fill up to the length of target data (should be training portion)
                target_len = min(len(target_accel_np), full_length)
                target_full[:target_len] = target_accel_np[:target_len]

                acceleration_data[target_method_name] = target_full
        except Exception as e:
            logger.warning(f"Failed to load target accelerations: {e}")

    if not acceleration_data:
        logger.error("No acceleration data available for plotting")
        return

    # Determine number of bodies
    first_accel_data = list(acceleration_data.values())[0]
    num_bodies_total = first_accel_data.shape[1] if len(first_accel_data.shape) > 1 else 1
    num_bodies_actual_plot = min(num_bodies_to_plot, num_bodies_total) if num_bodies_to_plot > 0 else num_bodies_total

    # Log dimension information for debugging
    logger.info(f"Plotting acceleration comparison with {len(acceleration_data)} data sources")
    logger.info(f"Time vector length: {len(time_np)}")
    for name, data in acceleration_data.items():
        logger.info(f"  {name}: shape {data.shape}")

    # Color mapping
    color_map = {
        'FNODE': 'red',
        'Analytical (GT)': 'blue',
        'Target (FFT)': 'green',
        'Target (FD)': 'orange',
        'Target': 'purple'
    }

    def _plot_priority(data_name):
        # Draw analytical first, then target, then model on top
        lower = data_name.lower()
        if 'analytical' in lower or 'gt' in lower:
            return 0
        if 'target' in lower:
            return 1
        return 2

    # --- PLOT 1: Full trajectory comparison ---
    fig, axs = plt.subplots(1, num_bodies_actual_plot, figsize=(8 * num_bodies_actual_plot, 6), squeeze=False)

    for body_idx in range(num_bodies_actual_plot):
        ax = axs[0, body_idx]

        for data_name, accel_data in sorted(acceleration_data.items(), key=lambda kv: _plot_priority(kv[0])):
            if body_idx >= accel_data.shape[1]:
                continue

            accel_body_data = accel_data[:, body_idx]
            data_color = color_map.get(data_name, 'gray')

            if 'analytical' in data_name.lower() or 'gt' in data_name.lower():
                # Plot analytical as continuous line
                mask = ~np.isnan(accel_body_data)
                if np.any(mask):
                    # Ensure time and data have compatible lengths
                    min_length = min(len(time_np), len(accel_body_data))
                    time_subset = time_np[:min_length]
                    accel_subset = accel_body_data[:min_length]
                    mask_subset = mask[:min_length]
                    ax.plot(time_subset[mask_subset], accel_subset[mask_subset], color=data_color, linestyle='-',
                            label=data_name, lw=2.0, alpha=0.85, zorder=1)

            elif 'target' in data_name.lower():
                # Target only exists for training
                mask = ~np.isnan(accel_body_data)
                if np.any(mask):
                    # Ensure time and data have compatible lengths
                    min_length = min(len(time_np), len(accel_body_data))
                    time_subset = time_np[:min_length]
                    accel_subset = accel_body_data[:min_length]
                    mask_subset = mask[:min_length]
                    ax.plot(time_subset[mask_subset], accel_subset[mask_subset], color=data_color, linestyle='--',
                            label=f'{data_name} (Train)', lw=1.5, alpha=0.8, zorder=2)

            else:
                # Model predictions - show train/test split
                mask = ~np.isnan(accel_body_data)
                if np.any(mask):
                    # Ensure time and data have compatible lengths
                    min_length = min(len(time_np), len(accel_body_data))
                    time_subset = time_np[:min_length]
                    accel_subset = accel_body_data[:min_length]
                    mask_subset = mask[:min_length]

                    train_mask = mask_subset & (np.arange(len(mask_subset)) < num_steps_train)
                    test_mask = mask_subset & (np.arange(len(mask_subset)) >= num_steps_train)

                    if np.any(train_mask):
                        ax.plot(time_subset[train_mask], accel_subset[train_mask],
                                color=data_color, linestyle='--', label=f'{data_name} (Train)', lw=1.5, zorder=3)
                    if np.any(test_mask):
                        ax.plot(time_subset[test_mask], accel_subset[test_mask],
                                color=data_color, linestyle=':', label=f'{data_name} (Test)', lw=1.5, alpha=0.9, zorder=3)

        # Add vertical line for train/test split
        if num_steps_train > 0 and num_steps_train < len(time_np):
            ax.axvline(x=time_np[num_steps_train - 1], color='gray', linestyle='--',
                       alpha=0.5, label='Train/Test Split')

        if num_epochs:
            ax.set_title(f'Body {body_idx + 1} Acceleration - Full Trajectory ({num_epochs} epochs)', fontsize=title_fontsize)
        else:
            ax.set_title(f'Body {body_idx + 1} Acceleration - Full Trajectory', fontsize=title_fontsize)
        ax.set_xlabel('Time (t)', fontsize=label_fontsize)
        ax.set_ylabel('Acceleration', fontsize=label_fontsize)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=legend_fontsize - 2)

    fig.tight_layout()
    if num_epochs:
        save_path = os.path.join(output_dir, f"{base_filename}_full_epochs_{num_epochs}.png")
    else:
        save_path = os.path.join(output_dir, f"{base_filename}_full.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Full acceleration comparison plot saved to {save_path}")

    # --- PLOT 2: Training-only detailed comparison ---
    # Extract training portion for detailed comparison
    train_time_np = time_np[:num_steps_train]

    # Create training-only data dictionary
    train_comparison_data = {}

    # Add all available data for training portion
    for data_name, accel_data in acceleration_data.items():
        train_data = accel_data[:num_steps_train]
        if not np.all(np.isnan(train_data)):
            train_comparison_data[data_name] = train_data

    # Also try to load specific training files if they exist
    train_analytical_path = os.path.join(results_dir, "train_analytical_accelerations.csv")
    train_target_path = os.path.join(results_dir, "train_target_accelerations.csv")

    if os.path.exists(train_analytical_path):
        try:
            train_analytical_df = pd.read_csv(train_analytical_path)
            analytical_cols = [col for col in train_analytical_df.columns if 'analytical_accel_body_' in col]
            if analytical_cols:
                train_analytical_np = train_analytical_df[analytical_cols].values
                if len(train_analytical_np) >= num_steps_train:
                    train_comparison_data['Analytical (Train File)'] = train_analytical_np[:num_steps_train]
        except Exception as e:
            logger.warning(f"Could not load train analytical file: {e}")

    if os.path.exists(train_target_path):
        try:
            train_target_df = pd.read_csv(train_target_path)
            target_cols = [col for col in train_target_df.columns if 'target_accel_body_' in col]
            if target_cols:
                train_target_np = train_target_df[target_cols].values
                if len(train_target_np) >= num_steps_train:
                    train_comparison_data['Target (Train File)'] = train_target_np[:num_steps_train]
        except Exception as e:
            logger.warning(f"Could not load train target file: {e}")

    if train_comparison_data:
        fig_train, axs_train = plt.subplots(1, num_bodies_actual_plot,
                                            figsize=(8 * num_bodies_actual_plot, 6), squeeze=False)

        for body_idx in range(num_bodies_actual_plot):
            ax = axs_train[0, body_idx]

            for data_name, train_data in sorted(train_comparison_data.items(), key=lambda kv: _plot_priority(kv[0])):
                if body_idx >= train_data.shape[1]:
                    continue

                train_body_data = train_data[:, body_idx]

                # Choose color and style
                if 'analytical' in data_name.lower():
                    color = 'blue'
                    style = '-'
                    lw = 2.0
                elif 'target' in data_name.lower():
                    color = 'green'
                    style = '--'
                    lw = 1.5
                else:  # Model
                    color = 'red'
                    style = ':'
                    lw = 1.5

                # Plot only non-NaN values
                mask = ~np.isnan(train_body_data)
                if np.any(mask):
                    plot_time = train_time_np[:len(train_body_data)]
                    ax.plot(plot_time[mask], train_body_data[mask],
                            color=color, linestyle=style, label=data_name, lw=lw, alpha=0.8,
                            zorder=1 if 'analytical' in data_name.lower() else (2 if 'target' in data_name.lower() else 3))

            if num_epochs:
                ax.set_title(f'Body {body_idx + 1} Training Acceleration Detail ({num_epochs} epochs)', fontsize=title_fontsize)
            else:
                ax.set_title(f'Body {body_idx + 1} Training Acceleration Detail', fontsize=title_fontsize)
            ax.set_xlabel('Time (t)', fontsize=label_fontsize)
            ax.set_ylabel('Acceleration', fontsize=label_fontsize)
            ax.grid(True, linestyle=':', alpha=0.3)
            ax.legend(loc='best', fontsize=legend_fontsize - 2)

        fig_train.tight_layout()
        if num_epochs:
            save_path_train = os.path.join(output_dir, f"{base_filename}_train_detail_epochs_{num_epochs}.png")
        else:
            save_path_train = os.path.join(output_dir, f"{base_filename}_train_detail.png")
        plt.savefig(save_path_train, dpi=200, bbox_inches='tight')
        plt.close(fig_train)
        logger.info(f"Training detail acceleration comparison plot saved to {save_path_train}")


def load_fnode_target_accelerations(target_csv_path, num_steps_train, current_device):
    """
    Loads target accelerations from CSV file for FNODE training.
    """
    try:
        if not os.path.exists(target_csv_path):
            logger.error(f"Target CSV file not found: {target_csv_path}")
            return None

        df = pd.read_csv(target_csv_path)

        # Determine column type
        target_cols = []
        for pattern in ['target_accel_body_', 'fft_derivative', 'accel_body_']:
            target_cols = [col for col in df.columns if col.startswith(pattern)]
            if target_cols:
                break

        if not target_cols:
            target_cols = [col for col in df.columns if 'accel' in col.lower() and col != 'time']

        if not target_cols:
            logger.error(f"No target acceleration columns found in {target_csv_path}")
            return None

        # Extract data
        if len(df) < num_steps_train:
            logger.warning(f"CSV has fewer rows ({len(df)}) than requested training steps ({num_steps_train})")
            num_steps_train = len(df)

        target_data = df[target_cols].values[:num_steps_train]
        target_tensor = torch.tensor(target_data, dtype=torch.float64, device=current_device)

        logger.info(f"Loaded target accelerations from {target_csv_path}, shape: {target_tensor.shape}")
        return target_tensor

    except Exception as e:
        logger.error(f"Error loading target accelerations: {e}")
        return None


def generate_target_accelerations(full_trajectory, time_vector, test_case, num_bodies, args, target_dir):
    """
    Generate target accelerations for FNODE training.
    Supports three methods via args.fnode_accel_mtd:
      - 'fft': FFT spectral differentiation on velocity -> acceleration
      - 'fd': finite difference on velocity -> acceleration
      - 'analytical': closed-form acceleration from state (when available)
    """
    logger.info("Generating target accelerations for FNODE training...")
    os.makedirs(target_dir, exist_ok=True)

    accel_mtd = getattr(args, "fnode_accel_mtd", "fd")
    fd_order = int(getattr(args, "fnode_target_fd_order", 2) or 2)

    if accel_mtd not in {"fft", "fd", "analytical"}:
        logger.error(f"Invalid fnode_accel_mtd: {accel_mtd}. Expected one of: fft, fd, analytical.")
        return None

    # Systems where FFT is not suitable
    fft_unsuitable_systems = {
        'Double_Pendulum': 'Chaotic/quasi-periodic motion not suitable for FFT-based method',
    }

    if accel_mtd == "fft" and test_case in fft_unsuitable_systems:
        logger.warning(f"FFT method not suitable for {test_case}: {fft_unsuitable_systems[test_case]}")
        logger.warning("Falling back to finite difference targets.")
        accel_mtd = "fd"

    # 0. Analytical method (when supported)
    if accel_mtd == "analytical":
        logger.info("Using analytical acceleration targets...")
        analytical_csv_path = os.path.join(target_dir, "analytical_target.csv")

        try:
            from Model.force_fun import force_sms, force_smsd, force_tmsd, force_dp, force_cp

            force_func_map = {
                'Single_Mass_Spring': force_sms,
                'Single_Mass_Spring_Damper': force_smsd,
                'Triple_Mass_Spring_Damper': force_tmsd,
                'Double_Pendulum': force_dp,
                'Cart_Pole': force_cp,
            }

            force_func = force_func_map.get(test_case)
            if force_func is None:
                logger.error(
                    f"Analytical acceleration targets are not implemented for test_case='{test_case}'. "
                    f"Please use --fnode_accel_mtd fd or fft."
                )
                return None

            if not torch.is_tensor(full_trajectory):
                full_trajectory = torch.tensor(full_trajectory, dtype=torch.float64)
            if not torch.is_tensor(time_vector):
                time_vector = torch.tensor(time_vector, dtype=torch.float64)

            device = full_trajectory.device
            N = full_trajectory.shape[0]
            analytical_targets = torch.zeros((N, num_bodies), device=device, dtype=torch.float64)

            for i in range(N):
                state_i = full_trajectory[i]
                bodys = state_i.view(num_bodies, 2)
                accel_i = force_func(bodys)
                if accel_i.dim() == 0:
                    accel_i = accel_i.unsqueeze(0)
                analytical_targets[i, : accel_i.numel()] = accel_i.to(device=device, dtype=torch.float64).flatten()

            analytical_data = {'time': time_vector.squeeze().cpu().numpy()}
            for j in range(num_bodies):
                analytical_data[f'target_accel_body_{j}'] = analytical_targets[:, j].cpu().numpy()
            pd.DataFrame(analytical_data).to_csv(analytical_csv_path, index=False, float_format='%.8g')
            logger.info(f"Saved analytical target accelerations to {analytical_csv_path}")
            return analytical_csv_path

        except Exception as e:
            logger.error(f"Failed to generate analytical targets: {e}", exc_info=True)
            return None

    # 2. Try FFT-FD hybrid method if requested and suitable
    if accel_mtd == "fft":
        logger.info("Attempting FFT-FD hybrid differentiation method...")

        fft_csv_path = os.path.join(target_dir, "fft_target.csv")
        fft_targets = torch.zeros((len(full_trajectory), num_bodies), device=full_trajectory.device)
        prob = int(getattr(args, "prob", 50) or 50)
        prob = max(1, prob)

        success_count = 0
        for body_idx in range(num_bodies):
            velocity_idx = body_idx * 2 + 1
            if velocity_idx < full_trajectory.shape[1]:
                velocity = full_trajectory[:, velocity_idx]

                # FFT for full range
                body_output_path = os.path.join(target_dir, f"fft_derivative_body_{body_idx}.csv")
                fft_deriv = calculate_fft_target_derivative(
                    velocity, time_vector,
                    output_csv_path=body_output_path
                )
                # FD for full range (used by hybrid boundaries)
                fd_deriv = estimate_temporal_gradient_finite_diff(velocity, time_vector, order=fd_order)

                if fft_deriv is not None and fd_deriv is not None:
                    n_pts = len(velocity)
                    trunc = max(1, n_pts // prob)
                    if trunc >= n_pts // 3:
                        trunc = max(1, n_pts // 4)
                    end_start_idx = n_pts - trunc

                    hybrid = fft_deriv.clone()
                    hybrid[:trunc] = fd_deriv[:trunc]
                    hybrid[end_start_idx:] = fd_deriv[end_start_idx:]

                    blend_length = min(10, trunc // 4)
                    if blend_length > 0:
                        # Smooth FD->FFT transition near left boundary
                        for i in range(blend_length):
                            idx = trunc - blend_length // 2 + i
                            if 0 <= idx < n_pts:
                                alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                                hybrid[idx] = (1 - alpha) * fd_deriv[idx] + alpha * fft_deriv[idx]
                        # Smooth FFT->FD transition near right boundary
                        for i in range(blend_length):
                            idx = end_start_idx - blend_length // 2 + i
                            if 0 <= idx < n_pts:
                                alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                                hybrid[idx] = alpha * fd_deriv[idx] + (1 - alpha) * fft_deriv[idx]

                    fft_targets[:, body_idx] = hybrid
                    success_count += 1
                    logger.info(
                        f"FFT-FD hybrid successful for body {body_idx}: "
                        f"FD boundaries [0:{trunc}] and [{end_start_idx}:{n_pts}], FFT in middle"
                    )
                elif fft_deriv is not None:
                    fft_targets[:, body_idx] = fft_deriv
                    success_count += 1
                    logger.info(f"FD unavailable for body {body_idx}, using pure FFT")
                elif fd_deriv is not None:
                    fft_targets[:, body_idx] = fd_deriv
                    success_count += 1
                    logger.info(f"FFT failed for body {body_idx}, using pure FD")

        # Save hybrid targets if at least one body succeeded
        if success_count > 0:
            try:
                fft_data = {'time': time_vector.cpu().numpy()}
                for j in range(num_bodies):
                    fft_data[f'target_accel_body_{j}'] = fft_targets[:, j].cpu().numpy()
                pd.DataFrame(fft_data).to_csv(fft_csv_path, index=False, float_format='%.8g')
                logger.info(f"Saved FFT-FD hybrid target accelerations to {fft_csv_path}")
                return fft_csv_path
            except Exception as e:
                logger.error(f"Failed to save FFT targets: {e}")

    # 3. Fall back to pure finite difference
    logger.info("Using finite difference method for all bodies")
    fd_csv_path = os.path.join(target_dir, "fd_target.csv")
    fd_targets = torch.zeros((len(full_trajectory), num_bodies), device=full_trajectory.device)

    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < full_trajectory.shape[1]:
            velocity = full_trajectory[:, velocity_idx]
            fd_deriv = estimate_temporal_gradient_finite_diff(velocity, time_vector, order=fd_order)
            if fd_deriv is not None:
                fd_targets[:, body_idx] = fd_deriv

    # Save FD targets
    try:
        # Debug: Log shapes
        logger.debug(f"time_vector shape: {time_vector.shape}")
        logger.debug(f"fd_targets shape: {fd_targets.shape}")

        # Ensure time_vector is 1D
        if time_vector.dim() > 1:
            time_vector = time_vector.squeeze()

        fd_data = {'time': time_vector.cpu().numpy()}
        for j in range(num_bodies):
            fd_data[f'target_accel_body_{j}'] = fd_targets[:, j].cpu().numpy()
        pd.DataFrame(fd_data).to_csv(fd_csv_path, index=False, float_format='%.8g')
        logger.info(f"Saved FD target accelerations to {fd_csv_path}")
        return fd_csv_path
    except Exception as e:
        logger.error(f"Failed to save FD targets: {e}")
        logger.error(f"time_vector shape: {time_vector.shape if 'time_vector' in locals() else 'undefined'}")
        logger.error(f"fd_targets shape: {fd_targets.shape if 'fd_targets' in locals() else 'undefined'}")
        return None


def save_data_np(data, test_case, model_string, training_size, num_steps_test, dt):
    """
    Save prediction data to numpy file.

    Args:
        data: Data to save (numpy array or torch tensor)
        test_case: Name of test case
        model_string: Model type string
        training_size: Number of training samples
        num_steps_test: Number of test steps
        dt: Time step
    """
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    save_path = os.path.join("results", test_case)
    os.makedirs(save_path, exist_ok=True)

    filename = f"{model_string}_prediction_train{training_size}_test{num_steps_test}_dt{dt}.npy"
    full_path = os.path.join(save_path, filename)

    np.save(full_path, data)
    logger.info(f"Saved prediction data to {full_path}")


def plot_sms_results(pred_traj, model_type, training_size, dt, output_dir,
                     num_steps_test=None, ground_truth_data=None, start_time=None):
    """
    Unified plotting function for Single Mass Spring results (LNN and HNN).
    Generates 2 plots: time series and phase space with energy.

    Args:
        pred_traj: Predicted trajectory [num_steps, 2] with columns [position, velocity]
        model_type: 'LNN' or 'HNN'
        training_size: Number of training steps
        dt: Time step
        output_dir: Directory to save figures
        num_steps_test: Total number of test steps (optional)
        ground_truth_data: Optional ground truth trajectory for comparison
        start_time: Start time for timing (optional)
    """
    import matplotlib.pyplot as plt
    import time

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy if needed
    if torch.is_tensor(pred_traj):
        pred_traj = pred_traj.detach().cpu().numpy()
    else:
        pred_traj = np.asarray(pred_traj)

    # Ensure 2D shape
    if pred_traj.ndim == 3:
        pred_traj = pred_traj[:, 0, :]  # Take first body if multiple

    # Special handling for HNN: convert (q,p) to (x,v) if needed
    # HNN outputs (q, p) where p = m*v, need to convert to (x, v)
    m = 10.0  # mass
    if model_type == 'HNN' and np.max(np.abs(pred_traj[:, 1])) > 5:
        # Likely in (q, p) format, convert to (x, v)
        pred_traj = pred_traj.copy()
        pred_traj[:, 1] = pred_traj[:, 1] / m  # Convert momentum to velocity

    num_steps = len(pred_traj)
    if num_steps_test is None:
        num_steps_test = num_steps

    # Generate time array
    t = np.arange(num_steps) * dt

    # Generate analytical ground truth for Single_Mass_Spring
    m = 10.0  # mass
    k = 50.0  # spring constant
    omega = np.sqrt(k / m)  # angular frequency

    x_true = np.cos(omega * t)
    v_true = -omega * np.sin(omega * t)

    # Calculate energies
    E_true = 0.5 * k * x_true[0]**2  # Constant total energy (initial)
    E_pred = 0.5 * m * pred_traj[:, 1]**2 + 0.5 * k * pred_traj[:, 0]**2

    # Set up the figure style
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    # ========== Plot 1: Time Series ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Position subplot
    ax1.plot(t, x_true, 'b-', linewidth=2, label='Ground truth')

    # Split prediction into training and test regions
    if training_size < num_steps:
        # Training region (ID generalization)
        ax1.plot(t[:training_size], pred_traj[:training_size, 0], 'r--',
                linewidth=2, label='ID generalization')
        # Test region (OOD generalization)
        ax1.plot(t[training_size-1:], pred_traj[training_size-1:, 0], 'r:',
                linewidth=2.5, label='OOD generalization')
    else:
        ax1.plot(t, pred_traj[:, 0], 'r--', linewidth=2, label=f'{model_type} prediction')

    ax1.set_ylabel('Position x (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_title(f'{model_type} Time Series Prediction', fontsize=14)

    # Velocity subplot
    ax2.plot(t, v_true, 'b-', linewidth=2, label='Ground truth')

    if training_size < num_steps:
        # Training region
        ax2.plot(t[:training_size], pred_traj[:training_size, 1], 'r--',
                linewidth=2, label='ID generalization')
        # Test region
        ax2.plot(t[training_size-1:], pred_traj[training_size-1:, 1], 'r:',
                linewidth=2.5, label='OOD generalization')
    else:
        ax2.plot(t, pred_traj[:, 1], 'r--', linewidth=2, label=f'{model_type} prediction')

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity v (m/s)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_time_series.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ========== Plot 2: Phase Space and Energy ==========
    # Use GridSpec for equal subplot frames with 1:1 aspect ratio
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Phase space subplot
    ax1.plot(x_true, v_true, 'b-', linewidth=2, label='Ground truth', alpha=0.8)
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', linewidth=2,
            label=f'{model_type} prediction', alpha=0.8)

    # Mark initial condition
    ax1.plot(x_true[0], v_true[0], 'go', markersize=8, label='Initial condition')

    ax1.set_xlabel('Position x (m)', fontsize=12)
    ax1.set_ylabel('Velocity v (m/s)', fontsize=12)
    ax1.set_title('Phase Space', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)

    # Set equal aspect ratio with proper limits
    ax1.set_aspect('equal', adjustable='datalim')
    x_margin = 0.1
    y_margin = 0.1
    x_range = max(abs(x_true.max()), abs(x_true.min()), abs(pred_traj[:, 0].max()), abs(pred_traj[:, 0].min()))
    y_range = max(abs(v_true.max()), abs(v_true.min()), abs(pred_traj[:, 1].max()), abs(pred_traj[:, 1].min()))
    max_range = max(x_range, y_range)
    ax1.set_xlim(-max_range - x_margin, max_range + x_margin)
    ax1.set_ylim(-max_range - y_margin, max_range + y_margin)

    # Energy subplot
    ax2.axhline(y=E_true, color='b', linestyle='-', linewidth=2,
               label=f'Ground truth (E={E_true:.2f} J)', alpha=0.8)
    ax2.plot(t, E_pred, 'r-', linewidth=2, label=f'{model_type} prediction', alpha=0.8)

    # Show absolute energy values (disable Matplotlib offset like "+2.499e1")
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Total Energy (J)', fontsize=12)
    ax2.set_title('Energy Conservation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)

    # Set y-axis limits centered around E_true (e.g., ~25 J) with tight margin
    # based on observed variation. This avoids unreadable offset/scaled ticks.
    E_min, E_max = np.min(E_pred), np.max(E_pred)
    E_range = E_max - E_min
    energy_margin = max(1e-6, 1.2 * E_range)
    ax2.set_ylim([E_true - energy_margin, E_true + energy_margin])

    # Don't use tight_layout with GridSpec and aspect='equal' - it causes warnings
    # bbox_inches='tight' in savefig handles the layout properly
    plt.savefig(os.path.join(output_dir, f'{model_type}_phase_energy.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Log energy conservation metrics
    energy_error = np.abs(E_pred - E_true) / E_true * 100
    logger.info(f"{model_type} Energy Conservation:")
    logger.info(f"  Mean relative error: {np.mean(energy_error):.2f}%")
    logger.info(f"  Max relative error: {np.max(energy_error):.2f}%")
    if training_size < num_steps:
        logger.info(f"  Train region error: {np.mean(energy_error[:training_size]):.2f}%")
        logger.info(f"  Test region error: {np.mean(energy_error[training_size:]):.2f}%")

    # Save time cost if provided
    if start_time is not None:
        end_time = time.time()
        logger.info(f"Total time: {end_time - start_time:.2f}s")


# ---------------------------------------------------------------------------
# Gibbs Phenomenon Trimming Functions (from gibbs_trim_benchmark.py)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class TrimResult:
    """Result from automatic trimming methods."""
    start: int
    end: int
    method: str
    meta: Dict[str, float]

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def compute_mad(x):
    """Compute Median Absolute Deviation (MAD) - robust measure of variability."""
    x = np.asarray(x)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def moving_average_filter(x, window):
    """Apply moving average filter to signal."""
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def trim_method_2_robust_residual(
    accel,
    smooth_window=21,
    mad_k=6.0,
    stable_run=12,
    max_trim_frac=0.25,
    min_keep_frac=0.5,
):
    """
    Auto-detect boundary artifact zones using robust residual threshold.

    This method detects and removes Gibbs phenomenon artifacts at signal boundaries
    by comparing the signal to a smoothed version and finding stable regions.

    Steps:
    1. Smooth signal with moving average
    2. Compute residual = |signal - smooth|
    3. Set threshold = median(residual) + mad_k * MAD(residual)
    4. Find stable regions where residual stays below threshold
    5. Trim boundaries outside stable regions

    Args:
        accel: Input acceleration signal (numpy array)
        smooth_window: Window size for moving average (default: 21)
        mad_k: MAD multiplier for threshold (default: 6.0)
        stable_run: Required consecutive stable samples (default: 12)
        max_trim_frac: Maximum fraction to trim from each side (default: 0.25)
        min_keep_frac: Minimum fraction of signal to keep (default: 0.5)

    Returns:
        TrimResult: Object containing start/end indices and metadata
    """
    a = np.asarray(accel, dtype=float)
    n = len(a)
    if n < 32:
        return TrimResult(0, n, "method2_residual", {"reason": "too_short"})

    # Ensure odd window size
    smooth_window = int(max(3, smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    # Compute smoothed signal and residual
    smooth = moving_average_filter(a, smooth_window)
    resid = np.abs(a - smooth)

    # Compute robust threshold
    med = float(np.median(resid))
    mad = compute_mad(resid) + 1e-12
    thr = med + float(mad_k) * mad

    # Compute maximum trim limits
    max_trim = int(max_trim_frac * n)
    max_trim = min(max_trim, int((1.0 - min_keep_frac) * n / 2))
    max_trim = max(0, max_trim)

    # Find left boundary: first stable run
    left = 0
    for i in range(0, max_trim + 1):
        j = i + stable_run
        if j >= n:
            break
        if np.all(resid[i:j] <= thr):
            left = i
            break
    else:
        left = max_trim

    # Find right boundary: last stable run
    right = n
    for i in range(0, max_trim + 1):
        j = n - i - stable_run
        if j <= 0:
            break
        if np.all(resid[j : n - i] <= thr):
            right = n - i
            break
    else:
        right = n - max_trim

    # Fallback if boundaries overlap
    if right <= left:
        left = max_trim
        right = n - max_trim

    return TrimResult(
        left,
        right,
        "method2_residual",
        {
            "smooth_window": float(smooth_window),
            "mad_k": float(mad_k),
            "stable_run": float(stable_run),
            "threshold": thr,
            "median": med,
            "mad": mad,
            "max_trim": float(max_trim),
        },
    )
