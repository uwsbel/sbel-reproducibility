# FNODE/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
import logging
import time

# Attempt to import torchdiffeq for advanced ODE solving
try:
    import torchdiffeq
    # Use adjoint method if available for potentially better memory efficiency during training
    ODEINT_FN = torchdiffeq.odeint_adjoint if hasattr(torchdiffeq, 'odeint_adjoint') else torchdiffeq.odeint
    TORCHDIFFEQ_AVAILABLE = True
    logging.info("torchdiffeq library loaded successfully.")
except ImportError:
    logging.warning("torchdiffeq library not found. Advanced ODE solving (like for FNODE test) will not be available.")
    ODEINT_FN = None
    TORCHDIFFEQ_AVAILABLE = False

# Import utilities and integrators from the FNODE/Model package
# (Ensure these imports work based on your project structure)
# Using specific imports within model.py is generally better practice than import *
try:
    # Assuming utils.py is in the same directory or accessible via Model.* path
    from Model.utils import *
    # Assuming integrator.py is in the same directory or accessible via Model.* path
    from Model.integrator import * # General integrators
except ImportError as e:
    # Fallback for local execution if FNODE/Model isn't in the Python path
    logging.warning(f"Could not import from utils/integrator - trying direct imports (may fail): {e}")
    # If run directly, these might fail unless utils.py/integrator.py are in the same dir
    # Attempt direct import (might still fail depending on execution context)
    try:
        from Model.utils import *
        from Model.integrator import *
    except ImportError as direct_e:
         logging.error(f"Direct import failed as well: {direct_e}. Check PYTHONPATH or execution location.")
         # Exit or raise might be appropriate here depending on requirements
         raise direct_e # Re-raise the error


# Import find_peaks specifically for FNODE_SliderCrank training
if SCIPY_AVAILABLE:
    try:
        from scipy.signal import find_peaks
    except ImportError:
        find_peaks = None # Ensure it's None if scipy.signal exists but find_peaks doesn't
else:
    find_peaks = None # Will be checked in train_fnode

logger = logging.getLogger(__name__)
# Configure root logger if no handlers are configured (useful for direct script execution)
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FNODE(nn.Module):
    """
    FNODE: Maps current state to accelerations of each body.
    Input: Core state [x1, v1, x2, v2, ...]. (dim_input = core_state_dim if d_interest=0)
           Or Core state + features if d_interest > 0.
    Output: Accelerations [a1, a2, ...]. (output_dim = num_bodys)
    """

    def __init__(self, num_bodys, layers=2, width=256, d_interest=0, activation='relu', initializer='kaiming'):
        super(FNODE, self).__init__()
        if layers < 2: raise ValueError("FNODE must have at least 2 layers (e.g., input layer and output layer).")
        self.num_bodys = num_bodys
        self.d_interest = d_interest  # Number of additional features (e.g., 0 for pure state->accel)
        self.core_state_dim = 2 * self.num_bodys  # e.g., [pos, vel] for each body

        # self.dim_input is the total number of features fed into the first linear layer of the FNODE
        self.dim_input = self.core_state_dim + self.d_interest

        # self.output_dim is the dimension of the FNODE's direct prediction, which are the accelerations
        self.output_dim = self.num_bodys  # Predicts accelerations [a1, a2, ...]

        self.activation_name = activation.lower()
        self.act = {'tanh': nn.Tanh(), 'relu': nn.ReLU()}.get(self.activation_name)
        if self.act is None: raise ValueError(f"Unsupported activation: {activation}")

        self.initializer = initializer.lower()
        if self.initializer not in ['xavier', 'kaiming']:
            logging.warning(f"Unsupported initializer: {initializer}. Using Kaiming as default.")
            self.initializer = 'kaiming'

        module_list = [nn.Linear(self.dim_input, width, bias=True), self.act]
        for _ in range(layers - 1):  # Number of hidden layers
            module_list.extend([nn.Linear(width, width, bias=True), self.act])
        module_list.append(nn.Linear(width, self.output_dim, bias=True))  # Output layer predicts accelerations
        self.network = nn.Sequential(*module_list)
        self._init_weights()  # Apply weight initialization
        logger.info(f"Initialized FNODE (State-to-Acceleration): "
                    f"CoreStateDim={self.core_state_dim}, d_interest={self.d_interest}, "
                    f"ModelInputDim={self.dim_input}, ModelOutputDim (Accelerations)={self.output_dim}, "
                    f"Bodies={self.num_bodys}, Layers={layers}, Width={width}, "
                    f"Activation='{self.activation_name}', Initializer='{self.initializer}'")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initializer == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif self.initializer == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=self.activation_name)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_input_to_network):
        # x_input_to_network is expected to be [batch_size, self.dim_input] or [self.dim_input]
        # It should already be (core_state + d_interest_features) if d_interest > 0, or just core_state if d_interest = 0.
        if x_input_to_network.shape[-1] != self.dim_input:
            raise ValueError(
                f"Input tensor last dim ({x_input_to_network.shape[-1]}) != FNODE model's expected input dim ({self.dim_input})")
        return self.network(x_input_to_network)  # Outputs predicted accelerations [a1_pred, a2_pred, ...]


class FNODE_CON(nn.Module):
    """
    Controlled FNODE - Neural network for controlled dynamical systems.
    Maps combined state+control input to accelerations.
    """
    def __init__(self, num_bodies, dim_input, dim_output, layers=2, width=256, 
                 activation='tanh', initializer='xavier'):
        super(FNODE_CON, self).__init__()
        
        self.num_bodies = num_bodies
        self.dim_input = dim_input    # Total input dimension (state + control)
        self.dim_output = dim_output  # Output dimension (accelerations)
        self.layers = layers
        self.width = width
        self.activation_name = activation
        self.initializer = initializer
        
        # Determine activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers_list = []
        
        # Input layer
        layers_list.extend([nn.Linear(dim_input, width), self.activation])
        
        # Hidden layers
        for _ in range(layers - 1):
            layers_list.extend([nn.Linear(width, width), self.activation])
        
        # Output layer
        layers_list.append(nn.Linear(width, dim_output))
        
        self.network = nn.Sequential(*layers_list)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"FNODE_CON initialized: Input={dim_input}, Output={dim_output}, "
                    f"Bodies={num_bodies}, Layers={layers}, Width={width}, "
                    f"Activation='{activation}', Initializer='{initializer}'")
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initializer == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif self.initializer == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', 
                                            nonlinearity=self.activation_name)
                else:
                    # Default initialization
                    nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, dim_input] or [dim_input]
               Expected to contain state + control variables
        
        Returns:
            Predicted accelerations [batch_size, dim_output] or [dim_output]
        """
        if x.shape[-1] != self.dim_input:
            raise ValueError(f"Input dimension mismatch: expected {self.dim_input}, "
                             f"got {x.shape[-1]}")
        
        return self.network(x)


class MBDNODE(nn.Module):
    def __init__(self, num_bodys, layers=2, width=256, d_interest=0,activation='tanh', initializer='xavier'):
        super(MBDNODE, self).__init__()

        # Determine activation
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[activation]

        # Construct the neural network layers
        self.d_interest=d_interest
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys+self.d_interest
        self.width = width
        #we want to use layer normalization here
        module_list = [nn.Linear(self.dim, self.width), self.act]
        for _ in range(layers - 1):
            module_list.extend([nn.Linear(self.width, self.width), self.act])

        module_list.append(nn.Linear(self.width, self.num_bodys))

        self.network = nn.Sequential(*module_list)

        # Apply initializer if specified
        if initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        if initializer == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.network(x)

def forward(self, x):
    if x.shape[-1] != self.dim_input:
         raise ValueError(f"Input tensor last dim ({x.shape[-1]}) != MBDNODE expected input dim ({self.dim_input})")
    return self.network(x)

class LSTMModel(nn.Module):
    """Standard LSTM baseline for sequence prediction."""
    def __init__(self, num_body=1, hidden_size=256, num_layers=2, dropout_rate=0.0):
        super(LSTMModel, self).__init__()
        self.num_body = num_body
        self.input_dim = 2 * num_body
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = 2 * num_body

        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_rate, bias=True)
        self.fc = nn.Linear(self.hidden_size, self.output_dim)
        logger.info(f"Initialized LSTMModel: Input Dim={self.input_dim}, Output Dim={self.output_dim}, Hidden={hidden_size}, Layers={num_layers}, Dropout={dropout_rate}")

    def forward(self, x_sequence):
        lstm_out, _ = self.lstm(x_sequence)
        last_time_step_output = lstm_out[:, -1, :]
        out = self.fc(last_time_step_output)
        return out

class FCNN(nn.Module):
    """Standard Feedforward NN (MLP) baseline for regression (e.g., state vs time)."""
    def __init__(self, num_bodys=1, layers=2, width=256, d_input_interest=0, activation='tanh', initializer='xavier'):
        super(FCNN, self).__init__()
        self.act = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}.get(activation)
        if self.act is None: raise ValueError("Unsupported activation")
        self.activation_name = activation

        self.num_bodys = num_bodys
        self.d_input_interest = d_input_interest
        self.dim_input = 1 + self.d_input_interest # Example: time + optional features
        self.width = width
        self.output_dim = 2 * self.num_bodys # Predicts full state

        module_list = [nn.Linear(self.dim_input, self.width), self.act]
        for _ in range(layers - 1):
            module_list.extend([nn.Linear(self.width, self.width), self.act])
        module_list.append(nn.Linear(self.width, self.output_dim))
        self.network = nn.Sequential(*module_list)

        self.initializer = initializer
        self._init_weights()
        logger.info(f"Initialized FCNN: Input Dim={self.dim_input}, Output Dim={self.output_dim}, Bodies={self.num_bodys}, Layers={layers}, Width={width}, Activation='{self.activation_name}', Initializer='{self.initializer}'")

    def _init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 if self.initializer == 'xavier': nn.init.xavier_normal_(m.weight)
                 elif self.initializer == 'kaiming': nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=self.activation_name)
                 if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_features):
        if x_features.shape[-1] != self.dim_input:
             raise ValueError(f"Input tensor last dim ({x_features.shape[-1]}) != FCNN expected input dim ({self.dim_input})")
        return self.network(x_features)


def train_fnode_with_csv_targets(model, s_train, t_train, train_params, optimizer, scheduler, output_paths,
                                 target_csv_path):
    """
    Modified FNODE training function that supports both hybrid FFT-FD and pure FD methods.
    Method selection based on fnode_use_hybrid_target parameter.

    FNODE maps state -> acceleration, so input is full state (positions and velocities).
    """
    import numpy as np
    import time
    import pandas as pd
    from utils import estimate_temporal_gradient_finite_diff, calculate_fft_target_derivative
    from utils import save_model_state, load_model_state

    current_device = next(model.parameters()).device

    # Get method selection parameter
    use_hybrid = train_params.get('fnode_use_hybrid_target', True)
    method_name = "Hybrid FFT-FD" if use_hybrid else "Pure Finite Difference"
    logger.info(f"--- Starting FNODE Training with {method_name} Method ---")

    # Get prob parameter
    prob = train_params.get('prob', 40)
    logger.info(f"Using prob parameter: {prob}")

    # Get test case name
    test_case = target_csv_path.split('/')[-3] if '/' in target_csv_path else "Unknown"

    # Keep tensors on device for efficient training (avoid CPU-GPU transfers)
    s_train = s_train.to(current_device)
    t_train = t_train.to(current_device)

    # Get number of bodies
    if s_train.dim() == 3:
        num_bodies = s_train.shape[1]
    else:
        num_bodies = s_train.shape[-1] // 2

    # Calculate truncation for hybrid method boundaries (not for discarding data)
    num_total_points = s_train.shape[0]
    trunc = num_total_points // prob  # Keep for hybrid method boundary calculations

    # Ensure reasonable truncation for boundaries
    if trunc >= num_total_points // 3:
        logger.warning(f"Boundary size too large with prob={prob}, adjusting")
        trunc = num_total_points // 4
        prob = num_total_points // trunc

    logger.info(f"Boundary size for hybrid method: {trunc}, Total points: {num_total_points}")

    # Initialize target accelerations on device for efficient training
    target_accelerations = torch.zeros((num_total_points, num_bodies), device=current_device)

    for body_idx in range(num_bodies):
        velocity_idx = body_idx * 2 + 1
        if velocity_idx < s_train.shape[-1]:
            # Extract velocity for this body
            if s_train.dim() == 3:
                velocity_train = s_train[:, body_idx, 1]
            else:
                velocity_train = s_train[:, velocity_idx]

            if use_hybrid:
                # Hybrid FFT-FD Method
                # Step 1: Calculate FFT for the entire training set [0:end]
                logger.info(f"Body {body_idx}: Calculating FFT for entire training set")

                fft_output_path = os.path.join(output_paths["results"], f"train_fft_body_{body_idx}_full.csv")
                fft_full = calculate_fft_target_derivative(
                    velocity_train, t_train,
                    output_csv_path=fft_output_path
                )

                if fft_full is None:
                    logger.warning(f"FFT calculation failed for body {body_idx}, falling back to pure FD")
                    # Fallback to pure FD
                    fd_full = estimate_temporal_gradient_finite_diff(
                        velocity_train, t_train, order=4
                    )
                    if fd_full is None:
                        logger.error(f"FD calculation also failed for body {body_idx}")
                        return None
                    target_accelerations[:, body_idx] = fd_full
                    continue

                # Start with FFT results for entire dataset
                target_accelerations[:, body_idx] = fft_full

                # Step 2: Calculate FD for the entire dataset
                logger.info(f"Body {body_idx}: Calculating FD for boundary replacement")

                fd_partial = estimate_temporal_gradient_finite_diff(
                    velocity_train, t_train, order=4
                )

                if fd_partial is None:
                    logger.warning(f"FD calculation failed for body {body_idx}, keeping FFT for all points")
                    continue

                # Step 3: Replace BOTH beginning [0:trunc] AND end portions with FD results
                # Beginning portion
                target_accelerations[:trunc, body_idx] = fd_partial[:trunc]

                # End portion - replace last trunc points
                end_start_idx = num_total_points - trunc
                target_accelerations[end_start_idx:, body_idx] = fd_partial[end_start_idx:]

                # Step 4: Add smooth blending at BOTH transition points
                blend_length = min(10, trunc // 4)  # Smooth transition length

                if blend_length > 0:
                    # Smooth transition from FD to FFT at the beginning (around index trunc)
                    for i in range(blend_length):
                        idx = trunc - blend_length // 2 + i
                        if 0 <= idx < num_total_points:
                            # Cosine interpolation for smooth transition
                            alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                            target_accelerations[idx, body_idx] = (
                                    (1 - alpha) * fd_partial[idx] +
                                    alpha * fft_full[idx]
                            )

                    # Smooth transition from FFT to FD at the end (around index end_start_idx)
                    for i in range(blend_length):
                        idx = end_start_idx - blend_length // 2 + i
                        if 0 <= idx < num_total_points:
                            # Cosine interpolation for smooth transition
                            alpha = 0.5 * (1 - np.cos(np.pi * i / blend_length))
                            target_accelerations[idx, body_idx] = (
                                    alpha * fd_partial[idx] +
                                    (1 - alpha) * fft_full[idx]
                            )

                logger.info(
                    f"Body {body_idx}: Hybrid acceleration computed - FD at both ends [0:{trunc}] and [{end_start_idx}:{num_total_points}], FFT in middle")

            else:
                # Pure Finite Difference Method
                logger.info(f"Body {body_idx}: Calculating pure FD for entire training set")

                fd_order = train_params.get('fnode_target_fd_order', 2)
                fd_full = estimate_temporal_gradient_finite_diff(
                    velocity_train, t_train, order=fd_order
                )

                if fd_full is None:
                    logger.error(f"FD calculation failed for body {body_idx}")
                    return None

                target_accelerations[:, body_idx] = fd_full
                logger.info(f"Body {body_idx}: Pure FD acceleration computed (order={fd_order})")

    # Use full training set - NO DISCARDING
    train_end_idx = num_total_points
    effective_points = train_end_idx

    logger.info(f"Using full training data: [0:{train_end_idx}] ({effective_points} points)")

    # Use full data (no truncation)
    s_train_effective = s_train
    t_train_effective = t_train
    target_accelerations_effective = target_accelerations
    # Save the target accelerations for training portion
    try:
        if use_hybrid:
            train_target_path = os.path.join(output_paths["results"], "train_hybrid_accelerations.csv")
        else:
            train_target_path = os.path.join(output_paths["results"], "train_fd_accelerations.csv")

        target_np = target_accelerations_effective.cpu().numpy()

        # Create method indicator for each point in training data
        method_indicator = []

        if use_hybrid:
            blend_length = min(10, trunc // 4)
            end_start_idx = num_total_points - trunc

            for i in range(len(target_accelerations_effective)):
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
        else:
            method_indicator = ['FD'] * len(target_accelerations_effective)

        df_target = pd.DataFrame({
            'time': t_train_effective.cpu().numpy(),
            **{f'target_accel_body_{j}': target_np[:, j] for j in range(num_bodies)},
            'method': method_indicator
        })
        df_target.to_csv(train_target_path, index=False, float_format='%.8g')
        logger.info(f"Saved training target accelerations to {train_target_path}")

    except Exception as e:
        logger.error(f"Failed to save training targets: {e}")

    # Prepare training data
    # FNODE expects full state as input (positions and velocities)
    model_input = s_train_effective

    logger.info(f"Model input shape: {model_input.shape}, Expected input dim: {model.dim_input}")

    # Training parameters
    num_epochs = train_params.get('epochs', 5000)
    outime_log_freq = train_params.get('outime_log', 100)
    criterion = torch.nn.MSELoss()

    # Training metrics
    best_loss_train = float('inf')
    best_epoch_train = -1
    start_time_train = time.time()

    # Early stopping parameters
    early_stop = train_params.get('early_stop', False)
    patience = train_params.get('patience', 50)
    patience_counter = 0
    early_stopped = False
    final_epoch = num_epochs - 1

    logger.info(f"Starting training loop with {num_epochs} epochs")

    if use_hybrid:
        logger.info(f"Method breakdown:")
        logger.info(f"  - FD at start: [0:{trunc}]")
        logger.info(f"  - FFT in middle: [{trunc}:{num_total_points - trunc}]")
        logger.info(f"  - FD at end: [{num_total_points - trunc}:{num_total_points}]")
        logger.info(f"  - Training uses: [0:{train_end_idx}] ({effective_points} points)")
    else:
        logger.info(f"Method breakdown:")
        logger.info(f"  - Pure FD: [0:{num_total_points}]")
        logger.info(f"  - Training uses: [0:{train_end_idx}] ({effective_points} points)")

    # Create DataLoader for batch processing
    from torch.utils.data import TensorDataset, DataLoader

    batch_size = train_params.get('batch_size', 1)

    # Create dataset
    dataset = TensorDataset(model_input, target_accelerations_effective)

    # Use all data for training (no validation split)
    train_data = dataset

    # Create DataLoader with proper batch size for efficient GPU utilization
    num_workers = train_params.get('num_workers', 0)
    # Data already on device, no need for pin_memory
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=False)

    logger.info(f"Training with {len(train_data)} samples (no validation set)")
    logger.info(f"Batch size: {batch_size}, Train batches: {len(train_loader)}, num_workers={num_workers}")

    # Initialize loss history tracking
    from utils import save_loss_history, plot_loss_curves
    loss_history = {'epoch': [], 'train_loss': [], 'learning_rate': []}


    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_input, batch_target in train_loader:
            # Data already on device from DataLoader
            optimizer.zero_grad()

            # Forward pass
            predicted_accelerations = model(batch_input)
            loss = criterion(predicted_accelerations, batch_target)

            if torch.isfinite(loss):
                loss.backward()

                if train_params.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['grad_clip'])

                optimizer.step()
                epoch_loss += loss.item()
            else:
                logger.warning(f"Non-finite loss at epoch {epoch + 1}, batch {num_batches}")

            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        # Record loss history
        current_lr = optimizer.param_groups[0]['lr']
        loss_history['epoch'].append(epoch + 1)
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['learning_rate'].append(current_lr)

        if epoch % outime_log_freq == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch + 1:>5}/{num_epochs} | "
                       f"Train Loss: {avg_train_loss:.6e} | "
                       f"Time: {(time.time() - epoch_start_time):.2f}s | LR: {current_lr:.2e}")

        if scheduler:
            scheduler.step()

        # Save best model based on train loss and handle early stopping
        comparison_loss = avg_train_loss
        if comparison_loss < best_loss_train:
            best_loss_train = comparison_loss
            best_epoch_train = epoch
            if output_paths.get("model"):
                save_model_state(model, output_paths["model"], model_filename="FNODE_best.pkl")
            patience_counter = 0  # Reset patience counter when improvement found
        else:
            if early_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1} (patience={patience})")
                    early_stopped = True
                    final_epoch = epoch
                    break

    training_duration = time.time() - start_time_train
    logger.info(f"--- FNODE Training Finished: Total Time {training_duration:.2f}s ---")
    logger.info(f"Method used: {method_name}")

    if best_epoch_train != -1:
        logger.info(f"Best model at epoch {best_epoch_train + 1} with train loss {best_loss_train:.6f}")
        if output_paths.get("model"):
            load_model_state(model, output_paths["model"], model_filename="FNODE_best.pkl",
                             current_device=current_device)

    # Save loss history
    if output_paths.get("results"):
        save_loss_history(loss_history, output_paths["results"], filename_prefix="FNODE")

    if output_paths.get("figures"):
        plot_loss_curves(loss_history, output_paths["figures"], model_name="FNODE", show_plot=False,
                        num_epochs=final_epoch + 1,
                        early_stopped=early_stopped,
                        patience=patience if early_stopped else None)
        logger.info("Loss history saved and plotted")

    return model, loss_history

def train_lstm(model, s_train, train_params, optimizer, scheduler, output_paths):
    """Trains the LSTMModel using train loss for early stopping."""
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    current_device = next(model.parameters()).device
    logger.info(f"--- Starting LSTM Training with pre-split data ---")
    logger.info(f"Sequence Length: {train_params['lstm_seq_len']}")
    seq_len = train_params['lstm_seq_len']

    # Normalize the training data
    num_steps_train = len(s_train)
    if s_train.dim() > 2: s_train = s_train.view(num_steps_train, -1)

    logger.info(f"Normalizing training data with MinMaxScaler(0, 1)")
    logger.info(f"  Before normalization - mean: {s_train.mean(dim=0)}, std: {s_train.std(dim=0)}")
    logger.info(f"  Before normalization - min: {s_train.min(dim=0)[0]}, max: {s_train.max(dim=0)[0]}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    s_train_np = s_train.cpu().numpy()
    s_train_normalized = scaler.fit_transform(s_train_np)
    s_train = torch.tensor(s_train_normalized, dtype=torch.float32)

    logger.info(f"  After normalization - mean: {s_train.mean(dim=0)}, std: {s_train.std(dim=0)}")
    logger.info(f"  After normalization - min: {s_train.min(dim=0)[0]}, max: {s_train.max(dim=0)[0]}")

    # Save scaler for inverse transform during testing
    scaler_path = os.path.join(output_paths["model"], "lstm_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    # Prepare training sequences
    train_sequences = []; train_targets = []
    for i in range(num_steps_train - seq_len):
        train_sequences.append(s_train[i : i + seq_len]); train_targets.append(s_train[i + seq_len])
    if not train_sequences: logger.error("Not enough training data for LSTM sequences."); return None, None
    train_sequences = torch.stack(train_sequences); train_targets = torch.stack(train_targets)


    # Move data to device before creating DataLoader for efficient training
    train_sequences = train_sequences.to(current_device)
    train_targets = train_targets.to(current_device)

    # Create data loaders
    train_dataset = TensorDataset(train_sequences, train_targets)
    num_workers = train_params.get('num_workers', 0)
    # Data already on device, no need for pin_memory
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers,
                            pin_memory=False)

    logger.info(f"Training with {len(train_sequences)} sequences (no validation set)")

    criterion = nn.MSELoss()
    best_loss_train = float('inf'); best_epoch_train = -1
    start_time_train = time.time()

    # Early stopping parameters
    early_stop = train_params.get('early_stop', False)
    patience = train_params.get('patience', 50)
    patience_counter = 0
    early_stopped = False
    final_epoch = train_params['epochs'] - 1

    # Initialize loss history
    loss_history = {'epoch': [], 'train_loss': [], 'learning_rate': []}

    test_case = train_params.get('test_case', 'Unknown')

    for epoch in range(train_params['epochs']):
        epoch_start_time = time.time()

        # Training phase
        model.train(); epoch_loss = 0.0; num_batches = 0
        for batch_seq, batch_target in train_loader:
            # Data already on device from DataLoader
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_seq)
            loss = criterion(predictions, batch_target)

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step(); epoch_loss += loss.item()
            else: logger.warning(f"Skipping step due to non-finite loss (Epoch {epoch})")
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        # Record loss history
        current_lr = optimizer.param_groups[0]['lr']
        loss_history['epoch'].append(epoch)
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['learning_rate'].append(current_lr)

        # Save loss history incrementally
        from utils import append_loss_to_csv
        epoch_data = {
            'epoch': epoch, 'train_loss': avg_train_loss,
            'learning_rate': current_lr
        }
        append_loss_to_csv(epoch_data, output_paths["results"], "LSTM_loss_history.csv")

        outime_log_freq = train_params.get('outime_log', 1)
        if epoch % outime_log_freq == 0 or epoch == train_params['epochs'] - 1:
            logger.info(f"Epoch {epoch:>5}/{train_params['epochs']} | Train Loss: {avg_train_loss:.2e} | Time: {(time.time() - epoch_start_time):.2f}s | LR: {current_lr:.2e}")

        if scheduler: scheduler.step()

        # Save best model based on validation loss and handle early stopping
        # Use train loss for early stopping when no validation
        comparison_loss = avg_train_loss
        if comparison_loss < best_loss_train:
            best_loss_train = comparison_loss; best_epoch_train = epoch
            save_model_state(model, output_paths["model"], model_filename="LSTM_best.pkl")
            logger.info(f"*** New best LSTM model saved at epoch {epoch} with train loss {best_loss_train:.2e} ***")
            patience_counter = 0  # Reset patience counter when improvement found
        else:
            if early_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    early_stopped = True
                    final_epoch = epoch
                    break

        save_ckpt_freq = train_params.get('save_ckpt_freq', 0)
        if save_ckpt_freq > 0 and (epoch + 1) % save_ckpt_freq == 0:
            save_model_state(model, output_paths["model"], model_filename=f"LSTM_ckpt_e{epoch + 1}.pkl")

    training_time = time.time() - start_time_train
    logger.info(f"--- LSTM Training Finished: {training_time:.2f}s ---")

    # Save final loss history
    from utils import save_loss_history, plot_loss_curves
    save_loss_history(loss_history, output_paths["results"], "LSTM_loss_history")
    plot_loss_curves(loss_history, output_paths["figures"], "LSTM",
                    num_epochs=final_epoch + 1,
                    early_stopped=early_stopped,
                    patience=patience if early_stopped else None)

    if best_epoch_train != -1:
        logger.info(f"Best LSTM model occurred at epoch {best_epoch_train} with train loss {best_loss_train:.2e}")
        load_model_state(model, output_paths["model"], model_filename="LSTM_best.pkl", current_device=current_device)
    return model, loss_history


def train_fcnn(model, s_train, t_train, train_params, optimizer, scheduler, output_paths):
    """Trains the FCNN model using train loss for early stopping."""
    current_device = next(model.parameters()).device
    logger.info(f"--- Starting FCNN Training with pre-split data ---")

    # Prepare training data
    num_steps_train = len(t_train)
    if model.dim_input == 1:
        train_inputs = t_train.unsqueeze(-1)
    elif model.dim_input > 1:
        logger.warning(f"FCNN d>0 currently uses time only.")
        train_inputs = t_train.unsqueeze(-1)
    else:
        logger.error("FCNN input dim cannot be zero.")
        return None, None

    # Prepare targets
    if s_train.dim() == 3:
        train_targets = s_train.view(num_steps_train, -1)
    else:
        train_targets = s_train

    if train_targets.shape[-1] != model.output_dim:
        logger.error(f"FCNN target dim mismatch")
        return None, None

    # Move data to device before creating DataLoader for efficient training
    train_inputs = train_inputs.to(current_device)
    train_targets = train_targets.to(current_device)

    criterion = nn.MSELoss()
    best_loss_train = float('inf'); best_epoch_train = -1
    start_time_train = time.time()

    # Early stopping parameters
    early_stop = train_params.get('early_stop', False)
    patience = train_params.get('patience', 50)
    patience_counter = 0
    early_stopped = False
    final_epoch = train_params['epochs'] - 1

    # Initialize loss history
    loss_history = {'epoch': [], 'train_loss': [], 'learning_rate': []}

    # Create data loader
    train_dataset = TensorDataset(train_inputs, train_targets)
    num_workers = train_params.get('num_workers', 0)
    # Data already on device, no need for pin_memory
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers,
                            pin_memory=False)

    logger.info(f"Training with {len(train_inputs)} samples (no validation set)")

    for epoch in range(train_params['epochs']):
        epoch_start_time = time.time()

        # Training phase
        model.train(); epoch_loss = 0.0; num_batches = 0
        for batch_input, batch_target in train_loader:
            # Data already on device from DataLoader
            optimizer.zero_grad()
            # Forward pass
            predictions = model(batch_input)
            loss = criterion(predictions, batch_target)
            if torch.isfinite(loss):
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); epoch_loss += loss.item()
            else: logger.warning(f"Skipping step due to non-finite loss (Epoch {epoch})")
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        # Record loss history
        current_lr = optimizer.param_groups[0]['lr']
        loss_history['epoch'].append(epoch)
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['learning_rate'].append(current_lr)

        # Save loss history incrementally
        from utils import append_loss_to_csv
        epoch_data = {
            'epoch': epoch, 'train_loss': avg_train_loss,
            'learning_rate': current_lr
        }
        append_loss_to_csv(epoch_data, output_paths["results"], "FCNN_loss_history.csv")

        logger.info(f"Epoch {epoch:>5}/{train_params['epochs']} | Train Loss: {avg_train_loss:.6f} | Time: {(time.time() - epoch_start_time):.2f}s | LR: {current_lr:.2e}")

        if scheduler: scheduler.step()

        # Save best model based on train loss and handle early stopping
        comparison_loss = avg_train_loss
        if comparison_loss < best_loss_train:
            best_loss_train = comparison_loss; best_epoch_train = epoch
            save_model_state(model, output_paths["model"], model_filename="FCNN_best.pkl")
            logger.info(f"*** New best FCNN model saved at epoch {epoch} with train loss {best_loss_train:.6f} ***")
            patience_counter = 0  # Reset patience counter when improvement found
        else:
            if early_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    early_stopped = True
                    final_epoch = epoch
                    break

        
        save_model_state(model, output_paths["model"], model_filename=f"FCNN_ckpt_e{epoch + 1}.pkl")

    training_time = time.time() - start_time_train
    logger.info(f"--- FCNN Training Finished: {training_time:.2f}s ---")

    # Save final loss history and plot
    from utils import save_loss_history, plot_loss_curves
    save_loss_history(loss_history, output_paths["results"], "FCNN_loss_history")
    plot_loss_curves(loss_history, output_paths["figures"], "FCNN",
                    num_epochs=final_epoch + 1,
                    early_stopped=early_stopped,
                    patience=patience if early_stopped else None)

    if best_epoch_train != -1:
        logger.info(f"Best FCNN model occurred at epoch {best_epoch_train} with train loss {best_loss_train:.6f}")
        load_model_state(model, output_paths["model"], model_filename="FCNN_best.pkl", current_device=current_device)
    return model, loss_history



def train_trajectory_MBDNODE(test_case, numerical_methods, model, body_tensor, training_size,
                           step_delay=2, num_epochs=450, dt=0.01, device='cuda', verbose=True,
                           batch_size=32, num_workers=0, output_paths=None, train_ratio=0.7,
                           lr=0.001, lr_scheduler='exponential', lr_decay_rate=0.98, lr_decay_steps=100,
                           early_stop=False, patience=30):
    # Setup logging
    def log(message):
        if verbose:
            print(message)

    log("Starting training of the MBDNODE model with training loss tracking")
    log(f"body_tensor shape: {body_tensor.shape}")
    log(f"Model: {model}")
    log(f"Batch size: {batch_size}")

    # Prepare training data for DataLoader
    inputs_list = []
    targets_list = []
    for i in range(training_size - step_delay):
        inputs_list.append(body_tensor[i, :, :].clone())
        targets_list.append(body_tensor[i + step_delay - 1, :, :].clone())

    # Create tensors from lists and move to device for efficient training
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    # Use all data for training (no validation split)
    train_inputs = inputs_tensor
    train_targets = targets_tensor

    # Create data loader
    train_dataset = TensorDataset(train_inputs, train_targets)
    # Data already on device, no need for pin_memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=False)

    log(f"Training with {len(train_inputs)} samples (no validation set)")

    criterion = nn.MSELoss()
    # Initialize optimizer with the provided learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize scheduler based on the lr_scheduler parameter
    if lr_scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_decay_steps, eta_min=lr * 0.0001)
    else:  # lr_scheduler == 'none'
        scheduler = None

    print(f"Optimizer: Adam (lr={lr}), Scheduler: {lr_scheduler}")
    best_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    early_stopped = False
    final_epoch = num_epochs - 1

    # Numerical methods mapping
    methods = {
        "fe": forward_euler_multiple_body,
        "rk4": runge_kutta_four_multiple_body,
        "midpoint": midpoint_method_multiple_body,
    }
    if numerical_methods not in methods:
        raise ValueError("The numerical method is not specified correctly")
    integration_method = methods[numerical_methods]

    # Create directory if it doesn't exist
    model_dir = f"saved_model/{test_case}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize loss history
    loss_history = {'epoch': [], 'train_loss': [], 'learning_rate': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        num_batches = 0

        model.train()
        for batch_inputs, batch_targets in train_loader:
            # Data already on device from DataLoader
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            batch_loss = 0
            accumulated_loss = 0

            # Process each sample in the batch
            for idx in range(batch_inputs.shape[0]):
                # Get individual sample from batch
                inputs = batch_inputs[idx].clone().detach().requires_grad_(True)
                target = batch_targets[idx].clone().detach()

                # Forward pass through integration
                Integrated_pred = integration_method(
                    inputs, neural_network_force_function_MBDNODE,
                    step_delay, dt, if_final_state=True, model=model)

                # Calculate loss for this sample
                loss = criterion(Integrated_pred, target)

                # Normalize loss by batch size for gradient accumulation
                loss_normalized = loss / batch_inputs.shape[0]

                # Accumulate gradients
                loss_normalized.backward(retain_graph=True)

                # Track loss for reporting
                batch_loss += loss.item()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Optional gradient clipping
                optimizer.step()

            # Add batch loss directly without extra averaging
            epoch_loss += batch_loss
            num_batches += 1

        # Average epoch loss
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        # Record loss history
        current_lr = optimizer.param_groups[0]['lr']
        loss_history['epoch'].append(epoch)
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['learning_rate'].append(current_lr)

        # Save loss history incrementally if output_paths provided
        if output_paths:
            from utils import append_loss_to_csv
            epoch_data = {
                'epoch': epoch, 'train_loss': avg_train_loss,
                'learning_rate': current_lr
            }
            append_loss_to_csv(epoch_data, output_paths.get("results", "."), "MBDNODE_loss_history.csv")

        # Calculate epoch time and get current learning rate
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Print in standardized format matching FNODE
        print(f"Epoch {epoch:>5}/{num_epochs} | Train Loss: {avg_train_loss:.2e} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")

        # Save best model based on train loss and handle early stopping
        comparison_loss = avg_train_loss
        if comparison_loss < best_loss:
            best_loss = comparison_loss
            best_epoch = epoch
            log(f"Best train loss: {best_loss}")
            model_path = f'{model_dir}/MBDNODE_best.pth'
            torch.save(model.state_dict(), model_path)
            patience_counter = 0  # Reset patience counter when improvement found
        else:
            if early_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    log(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    early_stopped = True
                    final_epoch = epoch
                    break

        if scheduler:
            scheduler.step()

    # Save final loss history and plot if output_paths provided
    if output_paths:
        from utils import save_loss_history, plot_loss_curves
        save_loss_history(loss_history, output_paths.get("results", "."), "MBDNODE_loss_history")
        plot_loss_curves(loss_history, output_paths.get("figures", "."), "MBDNODE",
                        num_epochs=final_epoch + 1,
                        early_stopped=early_stopped,
                        patience=patience if early_stopped else None)

    return model, loss_history

# --- Testing/Inference Functions ---

def neural_network_force_function_MBDNODE(body_tensor, model):
    """
    Calculate accelerations for the bodies using the MBDNODE model.

    Args:
        body_tensor: Tensor with shape [num_bodys, 2] (position, velocity)
        model: MBDNODE model instance

    Returns:
        Acceleration tensor with shape [num_bodys]
    """
    # Determine the number of bodies from input tensor
    num_bodys = body_tensor.shape[0]
    device = body_tensor.device

    # Use appropriate mass values based on the number of bodies
    if num_bodys == 3:  # Triple_Mass_Spring_Damper
        masses = torch.tensor([100.0, 10.0, 1.0], dtype=torch.float32, device=device)
    elif num_bodys == 2:  # Double_Pendulum or similar
        masses = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    else:  # Single mass system
        masses = torch.tensor([10.0], dtype=torch.float32, device=device)

    # Prepare input for the model (flattened state vector)
    inputs = body_tensor.reshape(-1).clone().to(device)

    # Get forces from the model
    forces = model(inputs)

    # Convert forces to accelerations using F = ma
    accelerations = forces / masses

    return accelerations


def neural_network_force_function_FNODE(body_tensor, model):
    """
    Calculate accelerations for the bodies using the FNODE model.

    Args:
        body_tensor: Tensor with shape [num_bodys, 2] (position, velocity)
        model: FNODE model instance

    Returns:
        Acceleration tensor with shape [num_bodys]
    """
    device = body_tensor.device
    num_bodies = body_tensor.shape[0]

    # Convert body tensor to flat state format expected by FNODE
    # body_tensor is [num_bodies, 2] -> flat_state is [2*num_bodies]
    flat_state = torch.zeros(2 * num_bodies, device=device)
    for i in range(num_bodies):
        flat_state[2*i] = body_tensor[i, 0]      # position
        flat_state[2*i + 1] = body_tensor[i, 1]  # velocity

    # Add time feature if model requires it (d_interest > 0)
    if model.d_interest == 1:
        # For simplicity, use zero time feature (could be enhanced to track actual time)
        time_feature = torch.zeros(1, device=device)
        model_input = torch.cat([flat_state, time_feature], dim=-1)
    elif model.d_interest > 1:
        # For d_interest > 1, additional feature handling would be needed
        raise NotImplementedError(f"d_interest={model.d_interest} > 1 not implemented for RK4 integration")
    else:
        model_input = flat_state

    # Get accelerations from FNODE model
    # Add batch dimension for model forward pass
    model_input = model_input.unsqueeze(0)
    accelerations = model(model_input).squeeze(0)

    return accelerations

# Assume necessary imports (torch, integrators) and definition of neural_network_force_function_MBDNODE

def test_fnode(model, s0_test_core_state, t_test_eval_times, test_params, output_paths):
    """
    Generates trajectory predictions using a trained FNODE model (that predicts accelerations)
    and an ODE solver.
    Args:
        model (FNODE): The trained FNODE model instance.
        s0_test_core_state (torch.Tensor): Initial CORE state for testing,
                                           shape [1, core_state_dim] (e.g., [1, x1, v1, x2, v2]).
        t_test_eval_times (torch.Tensor): Time vector for prediction [steps].
        test_params (dict): Dictionary containing parameters for testing, including
                            'ode_solver_params' (rtol, atol, method).
    Returns:
        torch.Tensor or None: Predicted trajectory [steps, core_state_dim] or None on failure.
    """
    current_device = next(model.parameters()).device
    model.eval()
    logger.info(f"--- Starting FNODE Testing (State-to-Acceleration Model) ---")
    logger.info(f"Model config for test: d_interest={model.d_interest}, output_dim (accelerations)={model.output_dim}")

    if not TORCHDIFFEQ_AVAILABLE:
        logger.error("torchdiffeq is required for test_fnode (ODE solving). Aborting.")
        return None
    if model.output_dim != model.num_bodys:
        logger.error(f"Testing FNODE that should predict accelerations, but its output_dim ({model.output_dim}) "
                     f"does not match num_bodys ({model.num_bodys}). Aborting.")
        return None

    # Validate t_test_eval_times for monotonicity
    if len(t_test_eval_times) > 1:
        diff_t = t_test_eval_times[1:] - t_test_eval_times[:-1]
        is_increasing = (diff_t > 0).all().item()
        is_decreasing = (diff_t < 0).all().item()
        if not (is_increasing or is_decreasing):
            logger.warning("Time vector for ODE solver is not strictly monotonic. Attempting to create a new one.")
            t_start_val, t_end_val = t_test_eval_times[0].item(), t_test_eval_times[-1].item()
            if t_start_val > t_end_val and not is_decreasing: t_end_val = t_start_val + (
                        t_test_eval_times[-1] - t_test_eval_times[0]).abs().item()
            t_test_eval_times = torch.linspace(min(t_start_val, t_end_val), max(t_start_val, t_end_val),
                                               len(t_test_eval_times), device=current_device)
            logger.info(
                f"New monotonic time vector for ODE solver: [{t_test_eval_times[0]:.2f}, ..., {t_test_eval_times[-1]:.2f}] length {len(t_test_eval_times)}")

    if s0_test_core_state.dim() == 1:
        s0_test_core_state = s0_test_core_state.unsqueeze(0)
    if s0_test_core_state.shape[0] != 1:
        logger.warning(
            f"s0_test_core_state batch size is {s0_test_core_state.shape[0]} > 1. ODEINT will run for multiple initial conditions.")
    if s0_test_core_state.shape[-1] != model.core_state_dim:
        logger.error(f"Initial state s0_test_core_state dim ({s0_test_core_state.shape[-1]}) "
                     f"does not match FNODE model's core_state_dim ({model.core_state_dim}). Aborting.")
        return None

    class AccelerationBasedODEFunc(nn.Module):
        def __init__(self, fnode_accel_model_instance):
            super().__init__()
            self.fnode_model = fnode_accel_model_instance
            self.num_bodys = self.fnode_model.num_bodys

        def forward(self, t_scalar, y_current_core_state_batch):
            fnode_input_batch = y_current_core_state_batch
            if self.fnode_model.d_interest == 1:
                time_feature_batch = t_scalar.repeat(y_current_core_state_batch.shape[0]).unsqueeze(-1)
                fnode_input_batch = torch.cat([y_current_core_state_batch, time_feature_batch], dim=-1)
            elif self.fnode_model.d_interest > 1:
                logger.error(
                    f"ODE Func: FNODE model expects d_interest={self.fnode_model.d_interest} features. Feature construction not implemented here.")

            predicted_accelerations_batch = self.fnode_model(fnode_input_batch)
            dy_dt_batch = torch.zeros_like(y_current_core_state_batch)
            for i in range(self.num_bodys):
                pos_idx = 2 * i;
                vel_idx = 2 * i + 1
                dy_dt_batch[..., pos_idx] = y_current_core_state_batch[..., vel_idx]
                dy_dt_batch[..., vel_idx] = predicted_accelerations_batch[..., i]

            return dy_dt_batch

    ode_solver_params = test_params.get('ode_solver_params',
                                        {'method': 'dopri5', 'rtol': 1e-7, 'atol': 1e-9})  # Adjusted defaults

    # Check if we should use runge_kutta_four_multiple_body for 'rk4' method
    if ode_solver_params.get('method') == 'rk4':
        logger.info("Using runge_kutta_four_multiple_body for RK4 integration")

        # Prepare initial body tensor from s0_test_core_state
        # s0_test_core_state is [1, core_state_dim] where core_state_dim = 2 * num_bodies
        s0_flat = s0_test_core_state.squeeze(0)  # Remove batch dimension
        num_bodies = model.num_bodys

        # Convert flat state to body tensor format [num_bodies, 2]
        body_tensor = torch.zeros(num_bodies, 2, device=current_device)
        for i in range(num_bodies):
            body_tensor[i, 0] = s0_flat[2*i]      # position
            body_tensor[i, 1] = s0_flat[2*i + 1]  # velocity

        # Calculate number of steps and dt from time vector
        num_steps = len(t_test_eval_times) - 1
        if num_steps <= 0:
            logger.error("Need at least 2 time points for RK4 integration")
            return None
        dt = (t_test_eval_times[-1] - t_test_eval_times[0]).item() / num_steps

        # Run RK4 integration using the same pattern as test_MBDNODE
        try:
            with torch.no_grad():
                testing_result = runge_kutta_four_multiple_body(
                    body_tensor,
                    neural_network_force_function_FNODE,
                    num_steps,
                    dt,
                    if_final_state=False,
                    model=model
                )

            if testing_result is None:
                logger.error("runge_kutta_four_multiple_body returned None")
                return None

            # Convert result from [num_steps+1, num_bodies, 2] to [num_steps+1, core_state_dim]
            num_time_points = testing_result.shape[0]
            pred_test_traj = torch.zeros(num_time_points, 2 * num_bodies, device=current_device)

            for t in range(num_time_points):
                for i in range(num_bodies):
                    pred_test_traj[t, 2*i] = testing_result[t, i, 0]      # position
                    pred_test_traj[t, 2*i + 1] = testing_result[t, i, 1]  # velocity

            logger.info(f"RK4 solver finished. Predicted trajectory shape: {pred_test_traj.shape}")

            if torch.isnan(pred_test_traj).any() or torch.isinf(pred_test_traj).any():
                logger.warning("Prediction contains NaN or inf values. Clamping them to zero.")
                pred_test_traj = torch.nan_to_num(pred_test_traj, nan=0.0, posinf=0.0, neginf=0.0)

            return pred_test_traj

        except Exception as rk4_err:
            logger.error(f"RK4 integration failed during testing: {rk4_err}", exc_info=True)
            return None

    else:
        # Use existing torchdiffeq implementation for other methods
        ode_func_for_solver = AccelerationBasedODEFunc(model).to(current_device)

        logger.info(
            f"Running ODE solver ({ode_solver_params.get('method', 'dopri5')}) with rtol={ode_solver_params.get('rtol')} atol={ode_solver_params.get('atol')}")
        try:
            with torch.no_grad():
                pred_test_traj = ODEINT_FN(
                    ode_func_for_solver,
                    s0_test_core_state.to(current_device),
                    t_test_eval_times.to(current_device),
                    rtol=ode_solver_params.get('rtol'),
                    atol=ode_solver_params.get('atol'),
                    method=ode_solver_params.get('method')
                )
            if pred_test_traj.shape[1] == 1: pred_test_traj = pred_test_traj.squeeze(1)
            logger.info(f"ODE solver finished. Predicted trajectory shape: {pred_test_traj.shape}")
            if torch.isnan(pred_test_traj).any() or torch.isinf(pred_test_traj).any():
                logger.warning("Prediction contains NaN or inf values. Clamping them to zero.")
                pred_test_traj = torch.nan_to_num(pred_test_traj, nan=0.0, posinf=0.0, neginf=0.0)
            return pred_test_traj
        except Exception as ode_err:
            logger.error(f"ODE integration failed during testing: {ode_err}", exc_info=True)
            return None



def test_MBDNODE(numerical_methods,model,body,num_steps,dt,device='cuda'):
    print("start testing the MBDNODE model")
    print("body shape is "+str(body.shape))
    print("model is ",model)
    print("numerical_methods is "+numerical_methods)
    #reshape the body_tensor to (num_data,num_bodys*2)
    num_body=body.shape[0]
    body_tensor = torch.tensor(body, dtype=torch.float32, device=device)
    if numerical_methods=="fe":
        testing_result = forward_euler_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    if numerical_methods=="rk4":
        testing_result = runge_kutta_four_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    if numerical_methods=="midpoint":
        testing_result = midpoint_method_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    #If the numerical methods is not specified correctly, raise an error
    #if numerical_methods!="fe" and numerical_methods!="rk4":
    #raise ValueError("The numerical methods is not specified correctly")
    return testing_result




def infer_lstm(model, s_history, num_steps_pred, test_params):
    """
    Generates future predictions using a trained LSTMModel iteratively.

    Args:
        model (LSTMModel): The trained LSTM model instance.
        s_history (torch.Tensor): Initial sequence of states, shape [seq_len, features] or [seq_len, bodies, 2].
        num_steps_pred (int): Number of future steps to predict.
        test_params (dict): Dictionary for potential future testing options.

    Returns:
        torch.Tensor or None: Predicted trajectory [num_steps_pred, features] or None on failure.
    """
    current_device = next(model.parameters()).device
    model.eval()
    logger.info(f"--- Starting LSTM Inference ---")
    logger.info(f"Predicting {num_steps_pred} steps.")

    # --- Data Preparation ---
    # Ensure input history is on the correct device and has shape [seq_len, features]
    if not isinstance(s_history, torch.Tensor): s_history = torch.tensor(s_history, dtype=torch.float32)
    s_history = s_history.to(current_device)
    if s_history.dim() == 3: # [seq_len, bodies, 2]
        s_history = s_history.view(s_history.shape[0], -1) # Flatten to [seq_len, features]

    seq_len = s_history.shape[0]
    features_dim = s_history.shape[1]
    expected_input_dim = model.input_dim

    if features_dim != expected_input_dim:
        logger.error(f"LSTM input history feature dim ({features_dim}) != model input dim ({expected_input_dim})")
        return None
    if seq_len < 1:
         logger.error("Input history sequence length must be at least 1.")
         return None

    # Initialize tensor to store predictions
    predictions = torch.zeros((num_steps_pred, features_dim), dtype=torch.float32, device=current_device)
    current_sequence = s_history.clone().unsqueeze(0) # Add batch dimension -> [1, seq_len, features]

    # --- Iterative Prediction Loop ---
    try:
        with torch.no_grad():
            for i in range(num_steps_pred):
                # Predict the next step using the current sequence
                next_state_pred = model(current_sequence) # Shape [1, features]

                # Store the prediction
                predictions[i] = next_state_pred.squeeze(0) # Shape [features]

                # Update the sequence: remove oldest step, append prediction
                # Shape of next_state_pred.unsqueeze(1) is [1, 1, features]
                current_sequence = torch.cat((current_sequence[:, 1:, :], next_state_pred.unsqueeze(1)), dim=1)
                # Shape remains [1, seq_len, features]

        logger.info(f"LSTM inference finished. Predicted trajectory shape: {predictions.shape}")
        return predictions
    except Exception as e:
        logger.error(f"Error during LSTM inference loop: {e}", exc_info=True)
        return None


def test_fcnn(model, t_test, test_params):
    """
    Generates state predictions using a trained FCNN model evaluated at given time points.

    Args:
        model (FCNN): The trained FCNN model instance.
        t_test (torch.Tensor): Time vector for prediction points [steps].
        test_params (dict): Dictionary for potential future testing options
                            (e.g., providing external features if d_interest > 0).

    Returns:
        torch.Tensor or None: Predicted states [steps, state_dim] or None on failure.
    """
    current_device = next(model.parameters()).device
    model.eval()
    logger.info(f"--- Starting FCNN Testing ---")

    # --- Data Preparation ---
    if not isinstance(t_test, torch.Tensor): t_test = torch.tensor(t_test, dtype=torch.float32)
    t_test = t_test.to(current_device)
    if t_test.dim() == 1: t_test = t_test.unsqueeze(-1) # Ensure shape [steps, 1]

    # Prepare input features
    fcnn_input_test = t_test
    if model.d_input_interest > 0:
        # Need to get external features corresponding to t_test
        # This data should ideally be passed via test_params
        logger.warning(f"FCNN testing with d_interest={model.d_input_interest} > 0 needs feature handling (not implemented). Using time only.")
        # Example: if features were passed in test_params['external_features_test']
        # ext_feat_test = test_params['external_features_test'].to(current_device)
        # fcnn_input_test = torch.cat([t_test, ext_feat_test], dim=-1)

    if fcnn_input_test.shape[-1] != model.dim_input:
         logger.error(f"FCNN test input dim ({fcnn_input_test.shape[-1]}) != model input dim ({model.dim_input})")
         return None

    # --- Prediction ---
    try:
        with torch.no_grad():
            predictions = model(fcnn_input_test) # Shape [steps, output_dim]
        logger.info(f"FCNN testing finished. Predicted states shape: {predictions.shape}")
        # Check shape dimensions individually for robustness
        if predictions.shape[0] != len(t_test) or predictions.shape[1] != model.output_dim:
             logger.error(f"Unexpected output shape from FCNN: {predictions.shape}, expected ({len(t_test)}, {model.output_dim})")
             return None
        return predictions
    except Exception as e:
        logger.error(f"Error during FCNN testing evaluation: {e}", exc_info=True)
        return None


class ControlledCartPoleDataset(torch.utils.data.Dataset):
    """Dataset class for controlled cart-pole batch training"""
    def __init__(self, body_tensor, force_tensor, accel_tensor):
        self.body_tensor = body_tensor
        self.force_tensor = force_tensor  
        self.accel_tensor = accel_tensor

    def __len__(self):
        return self.body_tensor.size(0)

    def __getitem__(self, idx):
        # Handle different tensor shapes based on data format
        if self.body_tensor.dim() == 3:
            # Cart_Pole_Controlled: [num_steps, 2, 2] -> flatten to [4]
            state = self.body_tensor[idx, :, :].view(-1)
        else:
            # Cart_Pole_D_Controlled: [num_steps, 6] -> already flat
            state = self.body_tensor[idx, :]
        
        action = self.force_tensor[idx]  # [1] - control force
        acceleration = self.accel_tensor[idx, :]  # [num_bodies] - accelerations
        return state, action, acceleration


def train_fnode_con_batch(model, body_tensor, force_tensor, accel_tensor, train_params,
                         optimizer, scheduler, output_paths):
    """
    Batch-based training function for FNODE_CON using MBDNODE-for-MBD2 approach.
    Direct mapping from [state, control] -> acceleration.
    
    Args:
        model: FNODE_CON model
        body_tensor: States tensor
            - Cart_Pole_Controlled: [num_steps, 2, 2] - states for cart and pole
            - Cart_Pole_D_Controlled: [num_steps, 6] - flat states for cart and two pendulums
        force_tensor: [num_steps, 1] - control forces
        accel_tensor: [num_steps, num_bodies] - target accelerations
        train_params: Training parameters dictionary (must include 'test_case')
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        output_paths: Output directory paths
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting FNODE_CON Batch Training ---")
    
    device = next(model.parameters()).device
    
    # Training setup
    num_epochs = train_params.get('epochs', 450)
    criterion = torch.nn.MSELoss()
    
    # Determine state dimensions based on test case
    test_case = train_params.get('test_case', 'Cart_Pole_Controlled')
    if test_case == 'Cart_Pole_Controlled':
        state_dim = 4  # [x, x_dot, theta, theta_dot]
        # Flatten body tensor from [num_steps, 2, 2] to [num_steps, 4]
        all_states = body_tensor.view(-1, state_dim)
    elif test_case == 'Cart_Pole_D_Controlled':
        state_dim = 6  # [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        # Body tensor is already in shape [num_steps, 6]
        all_states = body_tensor.view(-1, state_dim)
    else:
        raise ValueError(f"Unknown test case: {test_case}")

    # No input normalization - train on raw inputs to match inference
    all_forces = force_tensor.view(-1, 1)  # [num_steps, 1]
    all_inputs = torch.cat([all_states, all_forces], dim=1)  # [num_steps, state_dim+1]

    # Log input statistics for debugging
    logger.info(f"Raw input statistics (no normalization applied):")
    logger.info(f"  Mean: {all_inputs.mean(dim=0).cpu().numpy()}")
    logger.info(f"  Std:  {all_inputs.std(dim=0).cpu().numpy()}")
    logger.info(f"  Min:  {all_inputs.min(dim=0)[0].cpu().numpy()}")
    logger.info(f"  Max:  {all_inputs.max(dim=0)[0].cpu().numpy()}")

    # Move data to device before creating DataLoader for efficient training
    body_tensor = body_tensor.to(device)
    force_tensor = force_tensor.to(device)
    accel_tensor = accel_tensor.to(device)

    # Create dataset and dataloader
    dataset = ControlledCartPoleDataset(body_tensor, force_tensor, accel_tensor)
    # Force num_workers=0 when data is on CUDA to avoid process forking issues
    num_workers = 0  # Cannot use multiprocessing with CUDA tensors
    # Data already on device, no need for pin_memory
    batch_size = train_params.get('batch_size', 512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    
    # Training metrics
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()

    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(dataloader)}")

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for state_batch, action_batch, accel_batch in dataloader:
            # Data already on device from DataLoader
            optimizer.zero_grad()

            # Concatenate state and control as network input
            # state_batch: [batch, 4], action_batch: [batch, 1]
            inputs = torch.cat([state_batch, action_batch], dim=1)  # [batch, 5]

            # Forward pass: predict accelerations (no normalization)
            pred_accel = model(inputs)  # [batch, 2]

            # Loss: MSE between predicted and analytical accelerations
            loss = criterion(pred_accel, accel_batch)

            # Backpropagation
            if torch.isfinite(loss):
                loss.backward()

                if train_params.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['grad_clip'])

                optimizer.step()
                epoch_loss += loss.item()
            else:
                logger.warning(f"Non-finite loss at epoch {epoch + 1}, batch {num_batches}")

            num_batches += 1
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        # Logging with more details
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1:>5}/{num_epochs} | Loss: {avg_epoch_loss:.6e} | "
                   f"Total Loss: {epoch_loss:.6e} | Batches: {num_batches} | "
                   f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
        
        # Log gradient statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            if grad_norms:
                logger.info(f"  Gradient norms - Mean: {np.mean(grad_norms):.3e}, "
                           f"Max: {np.max(grad_norms):.3e}, Min: {np.min(grad_norms):.3e}")
        
        # Check for convergence issues
        if np.isnan(avg_epoch_loss) or np.isinf(avg_epoch_loss):
            logger.warning(f"NaN or Inf loss detected at epoch {epoch + 1}!")
            break


        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            if output_paths.get("model"):
                save_model_state(model, output_paths["model"], model_filename="FNODE_con_best.pkl")
        
        # Periodic checkpoint saves
        save_ckpt_freq = train_params.get('save_ckpt_freq', 0)
        if save_ckpt_freq > 0 and (epoch + 1) % save_ckpt_freq == 0:
            save_model_state(model, output_paths["model"], model_filename=f"FNODE_con_ckpt_e{epoch + 1}.pkl")
    
    training_duration = time.time() - start_time
    logger.info(f"--- FNODE_CON Batch Training Finished: Total Time {training_duration:.2f}s ---")
    
    # Save final model state
    if output_paths.get("model"):
        save_model_state(model, output_paths["model"], model_filename="FNODE_con_final.pkl")
        logger.info("Saved final model state as FNODE_con_final.pkl")
    
    if best_epoch != -1:
        logger.info(f"Best model at epoch {best_epoch + 1} with loss {best_loss:.6f}")
        if output_paths.get("model"):
            load_model_state(model, output_paths["model"], model_filename="FNODE_con_best.pkl",
                           current_device=device)
    
    return model
