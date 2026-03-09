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
import scipy.integrate
from functools import partial
from torch.autograd.functional import jacobian, hessian

# Attempt to import torchdiffeq for advanced ODE solving
try:
    import torchdiffeq
    from torchdiffeq import odeint  # Import odeint directly
    # Use adjoint method if available for potentially better memory efficiency during training
    ODEINT_FN = torchdiffeq.odeint_adjoint if hasattr(torchdiffeq, 'odeint_adjoint') else torchdiffeq.odeint
    TORCHDIFFEQ_AVAILABLE = True
    logging.info("torchdiffeq library loaded successfully.")
except ImportError:
    logging.warning("torchdiffeq library not found. Advanced ODE solving (like for FNODE test) will not be available.")
    ODEINT_FN = None
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None  # Define as None if not available

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
        self.output_type = "accel"  # model outputs accelerations

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


def _masses_for_test_case(test_case: str, num_bodys: int, device: torch.device) -> torch.Tensor:
    """Return per-body masses consistent with `Model/force_fun.py` for common benchmarks."""
    if test_case in {"Single_Mass_Spring", "Single_Mass_Spring_Damper", "Slider_Crank"}:
        # Slider_Crank state is [theta, omega] in this repo; treat canonical momentum as p = m * omega.
        return torch.full((num_bodys,), 10.0 if test_case != "Slider_Crank" else 1.0,
                          dtype=torch.float32, device=device)
    if test_case == "Triple_Mass_Spring_Damper":
        return torch.tensor([100.0, 10.0, 1.0], dtype=torch.float32, device=device)
    if test_case == "Double_Pendulum":
        return torch.ones((num_bodys,), dtype=torch.float32, device=device)
    if test_case == "Cart_Pole":
        return torch.full((num_bodys,), 10.0, dtype=torch.float32, device=device)
    # Fallback: assume unit masses (better than silently forcing 10.0)
    return torch.ones((num_bodys,), dtype=torch.float32, device=device)


class MBDNODE_Symplectic(nn.Module):
    """
    Direct copy of PNODE_Symplectic from PNODE-for-MBD2.
    Outputs Hamiltonian gradients [dH/dq, dH/dp] for symplectic integration.
    """
    def __init__(self, num_bodys=1, layers=3, width=256, d_interest=0, activation='relu', initializer='xavier'):
        super(MBDNODE_Symplectic, self).__init__()

        # Determine activation (exactly as PNODE_Symplectic)
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[activation]

        # Construct the neural network layers (exactly as PNODE_Symplectic)
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys  # [q, p] concatenated, no d_interest
        self.width = width
        self.output_type = "hamiltonian_grad"  # model outputs [dH/dq, dH/dp]

        # Build network exactly as PNODE_Symplectic (layers-2 middle layers)
        module_list = [nn.Linear(self.dim, self.width), self.act]
        for _ in range(self.layers - 2):  # CRITICAL: layers - 2, not layers - 1
            module_list.extend([nn.Linear(self.width, self.width), self.act])
        module_list.append(nn.Linear(self.width, self.dim))  # Output: same dimension as input
        self.network = nn.Sequential(*module_list)

        # Apply initializer if specified
        if initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
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


class MLP(nn.Module):
    """Simple MLP used as differentiable model inside HNN."""
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)


class HNN(nn.Module):
    """Hamiltonian Neural Network."""
    def __init__(self, input_dim, differentiable_model):
        super().__init__()
        self.input_dim = input_dim
        self.differentiable_model = differentiable_model

    def forward(self, x):
        return self.differentiable_model(x)

    def time_derivative(self, x, t=None):
        x_req = x.detach().requires_grad_(True)
        y = self.forward(x_req)
        dy = torch.autograd.grad(y.sum(), x_req, create_graph=True, allow_unused=True)[0]
        if dy is None:
            dy = torch.zeros_like(x_req)
        perm = self.permutation_tensor(self.input_dim).to(x_req.device)
        return dy @ perm

    @staticmethod
    def permutation_tensor(n):
        m = torch.eye(n)
        return torch.cat([-m[n // 2:], m[:n // 2]])


class LNN(nn.Module):
    """Lagrangian Neural Network for second-order systems."""
    def __init__(self, num_bodys=1, layers=3, width=256, d_interest=0,
                 activation='softmax', initializer='xavier'):
        super(LNN, self).__init__()
        self.act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1)
        }[activation]

        self.d_interest = d_interest
        self.layers = layers
        self.num_bodys = num_bodys
        self.dim = 2 * num_bodys + self.d_interest
        self.width = width

        self.fc1 = nn.Linear(2 * self.num_bodys, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.num_bodys)

    def lagrangian(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    def forward(self, x):
        n = x.shape[1] // 2
        xv = torch.autograd.Variable(x, requires_grad=True)
        xv_tup = tuple([xi for xi in xv])
        tqt = xv[:, n:]
        jacpar = partial(jacobian, self.lagrangian, create_graph=True)
        hesspar = partial(hessian, self.lagrangian, create_graph=True)
        A = tuple(map(hesspar, xv_tup))
        B = tuple(map(jacpar, xv_tup))
        multi = lambda Ai, Bi, tqti, n_val: torch.inverse(Ai[n_val:, n_val:]) @ (Bi[:n_val, 0] - Ai[n_val:, :n_val] @ tqti)
        multi_par = partial(multi, n_val=n)
        tqtt_tup = tuple(map(multi_par, A, B, tqt))
        tqtt = torch.cat([tqtti[None] for tqtti in tqtt_tup])
        xt = torch.cat([tqt, tqtt], axis=1)
        xt.retain_grad()
        return xt

    def t_forward(self, t, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Training / evaluation helpers for LNN and HNN
# ---------------------------------------------------------------------------
def lnn_solve_ode(model, x0, t_eval):
    """
    Integrate dynamics predicted by an LNN using scipy.odeint.
    """
    x0_np = x0.clone().detach().cpu().numpy()
    state_shape = x0_np.shape
    x0_vec = x0_np.reshape(-1)
    device_local = next(model.parameters()).device

    def f(x, t):
        state = np.array(x, dtype=np.float32).reshape(state_shape)
        x_tor = torch.tensor(state, requires_grad=True, dtype=torch.float32, device=device_local)
        pred = model(x_tor)
        return pred.clone().cpu().detach().numpy().reshape(-1)

    result = scipy.integrate.odeint(f, x0_vec, t_eval, atol=1e-10, rtol=1e-10)
    return result.reshape(len(t_eval), *state_shape)


def train_lnn(model, trajectory, derivatives, train_params, optimizer, scheduler, output_paths):
    """
    Train an LNN on provided trajectory and analytical derivatives.
    """
    logger = logging.getLogger(__name__)
    current_device = next(model.parameters()).device
    criterion = nn.MSELoss()
    num_epochs = train_params.get('epochs', 400)
    training_size = min(train_params.get('training_size', trajectory.shape[0]), trajectory.shape[0])
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(1, training_size):
            optimizer.zero_grad()
            inputs = trajectory[i, :, :].to(current_device)
            target = derivatives[i, :, :].to(current_device)
            pred_derivative = model(inputs)
            loss = criterion(pred_derivative, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / max(1, training_size - 1)
        logger.info(f"LNN Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_state(model, output_paths.get("model", "."), model_filename="LNN_best.pkl")

    return model


def test_lnn(model, initial_state, t_eval):
    """Roll out an LNN using scipy integration."""
    pred = lnn_solve_ode(model, initial_state, t_eval)
    return torch.tensor(pred, dtype=torch.float32)


def hnn_integrate(model, x0, t_eval, step_size=None):
    """
    Integrate Hamiltonian dynamics using RK4 and the model's time_derivative.
    """
    model.eval()
    device_local = next(model.parameters()).device
    if not isinstance(t_eval, np.ndarray):
        t_eval = np.asarray(t_eval)
    if len(t_eval) < 2:
        raise ValueError("t_eval must contain at least two time points for integration.")
    h = step_size if step_size is not None else float(t_eval[1] - t_eval[0])

    def deriv_fn(state, t):
        state_req = state.clone().detach().requires_grad_(True)
        return model.time_derivative(state_req.unsqueeze(0)).squeeze(0)

    state = torch.tensor(x0, dtype=torch.float32, device=device_local)
    traj = []
    for t in t_eval:
        traj.append(state.clone().detach().cpu().numpy())
        state = rk4_step(deriv_fn, state, t, h)
    return np.stack(traj)


def train_hnn(model, data_dict, train_params, optimizer, scheduler, output_paths):
    """
    Train an HNN using pre-generated datasets (x, dx) and their test splits.
    """
    logger = logging.getLogger(__name__)
    device_local = next(model.parameters()).device
    criterion = nn.MSELoss()
    num_steps = train_params.get('epochs', 30000)
    log_interval = max(1, train_params.get('log_interval', 1000))

    x = torch.tensor(data_dict['x'], dtype=torch.float32, device=device_local)
    dxdt = torch.tensor(data_dict['dx'], dtype=torch.float32, device=device_local)
    test_x = torch.tensor(data_dict['test_x'], dtype=torch.float32, device=device_local)
    test_dxdt = torch.tensor(data_dict['test_dx'], dtype=torch.float32, device=device_local)

    stats = {'train_loss': [], 'test_loss': []}
    best_loss = float("inf")

    for step in range(num_steps):
        optimizer.zero_grad()
        dxdt_hat = model.time_derivative(x)
        loss = criterion(dxdt, dxdt_hat)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Need gradients through test_x for time_derivative; keep graph off for params
        with torch.set_grad_enabled(True):
            test_dxdt_hat = model.time_derivative(test_x)
            test_loss = criterion(test_dxdt, test_dxdt_hat).detach()

        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())

        if (step + 1) % log_interval == 0 or step == num_steps - 1:
            logger.info(f"HNN Step {step + 1}/{num_steps} | Train: {loss.item():.6e} | Test: {test_loss.item():.6e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_model_state(model, output_paths.get("model", "."), model_filename="HNN_best.pkl")

    return model, stats


def train_fnode_with_csv_targets(model, s_train, t_train, train_params, optimizer, scheduler, output_paths,
                                 target_csv_path):
    """
    FNODE training with precomputed acceleration targets loaded from CSV.
    Target generation method is controlled by train_params['fnode_accel_mtd'] and performed upstream.
    """
    import time
    import pandas as pd
    from Model.utils import save_model_state, load_model_state
    from Model.utils import load_fnode_target_accelerations

    current_device = next(model.parameters()).device

    accel_mtd = train_params.get('fnode_accel_mtd', 'fd')
    if accel_mtd not in {'fft', 'fd', 'analytical'}:
        logger.warning(f"Unknown fnode_accel_mtd='{accel_mtd}', falling back to 'fd'")
        accel_mtd = 'fd'
    logger.info(f"--- Starting FNODE Training (targets: {accel_mtd}) ---")

    # Get test case name
    test_case = target_csv_path.split('/')[-3] if '/' in target_csv_path else "Unknown"

    # Get zero_v parameter for Single_Mass_Spring
    zero_v = train_params.get('zero_v', False)
    test_case_from_params = train_params.get('test_case', test_case)

    # Keep tensors on device for efficient training (avoid CPU-GPU transfers)
    s_train = s_train.to(current_device)
    t_train = t_train.to(current_device)

    # Get number of bodies
    if s_train.dim() == 3:
        num_bodies = s_train.shape[1]
    else:
        num_bodies = s_train.shape[-1] // 2

    # Load target accelerations from CSV
    num_total_points = s_train.shape[0]
    target_accelerations = load_fnode_target_accelerations(
        target_csv_path=target_csv_path,
        num_steps_train=num_total_points,
        current_device=current_device,
    )
    if target_accelerations is None:
        logger.error("Failed to load target accelerations from CSV; aborting training.")
        return None
    if target_accelerations.dim() != 2 or target_accelerations.shape[1] != num_bodies:
        logger.error(
            f"Target accelerations shape mismatch. Expected (N,{num_bodies}), got {tuple(target_accelerations.shape)} "
            f"from {target_csv_path}"
        )
        return None

    # Optionally drop tail points for FD targets to avoid boundary FD artifacts
    fd_drop_tail = int(train_params.get('fnode_fd_drop_tail', 0) or 0)
    if accel_mtd == 'fd' and fd_drop_tail > 0:
        train_end_idx = num_total_points - fd_drop_tail
        if train_end_idx <= 1:
            logger.warning(
                f"Requested fnode_fd_drop_tail={fd_drop_tail} leaves too few points (N={num_total_points}); disabling drop."
            )
            train_end_idx = num_total_points
            fd_drop_tail = 0
    else:
        train_end_idx = num_total_points

    effective_points = train_end_idx
    logger.info(f"Training data range used: [0:{train_end_idx}] ({effective_points} points)")

    # Slice effective training tensors
    s_train_effective = s_train[:train_end_idx]
    t_train_effective = t_train[:train_end_idx]
    target_accelerations_effective = target_accelerations[:train_end_idx]
    # Save the target accelerations for training portion
    try:
        target_base_dir = os.path.dirname(target_csv_path) if target_csv_path else output_paths["results"]
        os.makedirs(target_base_dir, exist_ok=True)
        train_target_path = os.path.join(target_base_dir, f"train_{accel_mtd}_accelerations.csv")

        target_np = target_accelerations_effective.cpu().numpy()

        method_indicator = [accel_mtd] * len(target_accelerations_effective)

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
    if model.d_interest == 0:
        # Pure state input
        model_input = s_train_effective
    else:
        # State + additional features (e.g., time)
        time_feature = t_train_effective.unsqueeze(-1)
        model_input = torch.cat([s_train_effective, time_feature], dim=-1)

    # Align data dtype with model parameters to avoid backward dtype mismatch
    model_dtype = next(model.parameters()).dtype
    model_input = model_input.to(dtype=model_dtype)
    target_accelerations_effective = target_accelerations_effective.to(dtype=model_dtype)

    logger.info(f"Model input shape: {model_input.shape}, dtype: {model_input.dtype}, Expected input dim: {model.dim_input}")

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

    logger.info("Method breakdown:")
    logger.info(f"  - Targets loaded from: {os.path.basename(target_csv_path)}")
    logger.info(f"  - Target method: {accel_mtd}")
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

    # Log zero_v status
    if zero_v and test_case_from_params == 'Single_Mass_Spring':
        logger.info("*** ZERO_V MODE ENABLED: Velocity components will be zeroed in model input during training ***")
        logger.info("*** Note: Acceleration targets are computed from original data with actual velocities ***")

    # Initialize loss history tracking
    from Model.utils import save_loss_history, plot_loss_curves
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

            # Apply zero_v if enabled for Single_Mass_Spring
            if zero_v and test_case_from_params == 'Single_Mass_Spring':
                # Create a copy of batch_input and zero out velocity components
                batch_input_modified = batch_input.clone()
                # For Single_Mass_Spring: column 1 is velocity
                batch_input_modified[:, 1] = 0.0
                batch_input = batch_input_modified

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
    logger.info(f"Method used: {accel_mtd}")

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
    outime_log_freq = train_params.get('outime_log', 1)
    if outime_log_freq == False or outime_log_freq == 0:
        outime_log_freq = 1  # Prevent division by zero

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
        from Model.utils import append_loss_to_csv
        epoch_data = {
            'epoch': epoch, 'train_loss': avg_train_loss,
            'learning_rate': current_lr
        }
        append_loss_to_csv(epoch_data, output_paths["results"], "LSTM_loss_history.csv")

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
    from Model.utils import save_loss_history, plot_loss_curves
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
        from Model.utils import append_loss_to_csv
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
    from Model.utils import save_loss_history, plot_loss_curves
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
                           lr=0.001, lr_scheduler='exponential', lr_decay_rate=0.98):
    # Setup logging
    def log(message):
        if verbose:
            print(message)
    log("Starting training of the PNODE model")
    log(f"body_tensor shape: {body_tensor.shape}")
    log(f"Model: {model}")
    criterion = nn.MSELoss()

    # Use provided learning rate parameter
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize scheduler based on the lr_scheduler parameter
    if lr_scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=lr_decay_rate)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr * 0.0001)
    else:  # lr_scheduler == 'none'
        scheduler = None

    print(f"Optimizer: Adam (lr={lr}), Scheduler: {lr_scheduler}")
    best_loss = float("inf")
    # Numerical methods mapping
    methods = {
        "fe": forward_euler_multiple_body,
        "rk4": runge_kutta_four_multiple_body,
        "midpoint": midpoint_method_multiple_body,
    }
    if numerical_methods not in methods:
        raise ValueError("The numerical method is not specified correctly")
    integration_method = methods[numerical_methods]
    # Create directory if it doesn't exist (matching sbel path)
    model_dir = f"saved_model/{test_case}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        #log(f"Epoch: {epoch}")
        for i in range(0, training_size - step_delay):
            optimizer.zero_grad()
            inputs = body_tensor[i, :, :].clone().detach().to(device).requires_grad_(True)
            target = body_tensor[i + step_delay - 1, :, :].clone().detach().to(device)
            #target = body_tensor[i:i + step_delay, :, :].clone().to(device)
            Integrated_pred = integration_method(inputs, neural_network_force_function_MBDNODE, step_delay, dt,if_final_state=True, model=model)
            #print("target shape is "+str(target.shape))
            #Integrated_pred = integration_method(inputs, neural_network_force_function_PNODE, step_delay, dt,if_final_state=False, model=model)
            loss1 = criterion(Integrated_pred, target)
            #energy_loss=criterion(total_energy_sms(Integrated_pred),total_energy_sms(inputs))
            #loss = criterion((Integrated_pred-inputs)/dt, (target-inputs)/dt)#+criterion(Integrated_pred, target)
            loss=loss1
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("Epoch "+str(epoch)+" loss is "+str(loss.item()))
        #losses.append(epoch_loss / training_size)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            log(f"Best loss: {best_loss}")
            # Use consistent naming like sbel version
            model_path = f'{model_dir}/MBDNODE_best.pth'
            torch.save(model.state_dict(), model_path)
        if scheduler:
            scheduler.step()
        log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')

    # Load the best model before returning (instead of returning the final epoch model)
    best_model_path = f'{model_dir}/MBDNODE_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        log(f"Loaded best model from {best_model_path} for testing")
    else:
        log(f"Warning: Best model not found at {best_model_path}, using final epoch model")

    return model




def train_trajectory_MBDNODE_symplectic(test_case, numerical_methods, model, body_tensor, training_size,
                                        step_delay=2, num_epochs=450, dt=0.01, device='cuda', verbose=True):
    """
    Direct copy of train_trajectory_PNODE_symplectic2 from PNODE-for-MBD2.
    Trains model to output Hamiltonian gradients [dH/dq, dH/dp].

    CRITICAL: Expects body_tensor in [q, p] format, not [q, v]!
    """
    def log(message):
        if verbose:
            print(message)

    log("Starting training of the MBDNODE_symplectic model")
    log(f"body_tensor shape: {body_tensor.shape}")
    log(f"Model: {model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # PNODE uses 0.0001
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_loss = float("inf")

    # Numerical methods mapping
    methods = {
        "sep_sv": sep_stormer_verlet_multiple_body,
        "yoshida4": yoshida4_multiple_body,
        "fukushima6": fukushima6_multiple_body
    }
    if numerical_methods not in methods:
        raise ValueError("The numerical method is not specified correctly")
    integration_method = methods[numerical_methods]

    # Create directory if it doesn't exist (matching sbel path)
    model_dir = f"saved_model/{test_case}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Always show epoch progress every 10 epochs
        if epoch % 1 == 0:
            print(f"Training Progress: Epoch {epoch}/{num_epochs}")
        log(f"Epoch: {epoch}")

        for i in range(training_size - step_delay + 1):
            optimizer.zero_grad()

            # CRITICAL: Use data directly as [q, p] without conversion
            inputs = body_tensor[i, :, :].clone().to(device).requires_grad_(True)
            target = body_tensor[i:i + step_delay, :, :].clone().to(device)

            # Integrate using symplectic method
            Integrated_pred = integration_method(
                inputs,
                neural_network_force_function_MBDNODE_symplectic_dH_dq,
                neural_network_force_function_MBDNODE_symplectic_dH_dp,
                step_delay,
                dt,
                if_final_state=False,
                model=model
            )

            loss = criterion(Integrated_pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / training_size)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            log(f"Best loss: {best_loss}")
            # Use consistent naming like sbel version
            model_path = f'{model_dir}/MBDNODE_best.pth'
            torch.save(model.state_dict(), model_path)

        if scheduler:
            scheduler.step()
        log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / training_size:.10f}')

    # Load the best model before returning (instead of returning the final epoch model)
    best_model_path = f'{model_dir}/MBDNODE_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        log(f"Loaded best model from {best_model_path} for testing")
    else:
        log(f"Warning: Best model not found at {best_model_path}, using final epoch model")

    return model


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
        masses = torch.tensor([100.0, 10.0, 1.0], dtype=torch.float32, device=device)  # FIXED: Correct mass values
    elif num_bodys == 2:  # Double_Pendulum or similar
        masses = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    elif num_bodys == 1:  # Single mass system (SMS/SMSD in this repo use m=10)
        masses = torch.tensor([10.0], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported number of bodies: {num_bodys}")

    # Prepare input for the model (flattened state vector)
    inputs = body_tensor.reshape(-1).clone().to(device)

    # Get forces from the model
    forces = model(inputs)

    # Convert forces to accelerations using F = ma
    accelerations = forces / masses

    return accelerations


def _mbdnode_masses(num_bodys: int, device: torch.device) -> torch.Tensor:
    """Mass vector consistent with `neural_network_force_function_MBDNODE*` helpers."""
    if num_bodys == 3:
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    if num_bodys == 2:
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    if num_bodys == 1:
        # Single_Mass_Spring in this repo uses m=10.0 in data/force definitions.
        return torch.tensor([10.0], dtype=torch.float32, device=device)
    raise ValueError(f"Unsupported number of bodies: {num_bodys}")

# Note: Old MBDNODE_dH_dq and MBDNODE_dH_dp functions removed - they incorrectly tried to
# convert force predictions to Hamiltonian gradients. Use MBDNODE_Symplectic model instead
# for symplectic integration, which directly learns Hamiltonian gradients.


def neural_network_force_function_MBDNODE_symplectic_dH_dq(body_tensor, model):
    """
    Direct copy of neural_network_force_function_PNODE_symplectic_dH_dq.
    body_tensor is a tensor with shape (num_bodys,2), [q, p] format.
    """
    body_num = body_tensor.shape[0]
    inputs = body_tensor.view(-1).clone().to(body_tensor.device).requires_grad_(True)
    f_pred = model(inputs)
    # Extract the gradient of the Hamiltonian with respect to the generalized coordinate
    return f_pred[0:body_num]


def neural_network_force_function_MBDNODE_symplectic_dH_dp(body_tensor, model):
    """
    Direct copy of neural_network_force_function_PNODE_symplectic_dH_dp.
    body_tensor is a tensor with shape (num_bodys,2), [q, p] format.
    """
    body_num = body_tensor.shape[0]
    inputs = body_tensor.view(-1).clone().to(body_tensor.device).requires_grad_(True)
    f_pred = model(inputs)
    # Extract the gradient of the Hamiltonian with respect to the generalized momentum
    return f_pred[body_num:2*body_num]


def train_fnode_symplectic_grad(model, s_train, t_train, train_params, optimizer, scheduler, output_paths,
                                test_case: str, symplectic_integrator: str = "yoshida4"):
    """Train a grad-output model using direct supervision on (dH/dq, dH/dp) targets.

    IMPORTANT (per user requirement): this function does NOT use a symplectic integrator during
    training. Instead it constructs training targets from the dataset:

    - Input dataset state is [q, v] (per body).
    - Convert to canonical input [q, p] with p = m*v.
    - Supervise dH/dp with v (since dH/dp = p/m for standard kinetic energy).
    - Supervise dH/dq with -m*a where a = dv/dt estimated from the trajectory.

    The model is expected to map a flattened [q, p] state (length 2*num_bodys) to a vector
    interpreted as [dH/dq (num_bodys), dH/dp (num_bodys)].

    Args:
        symplectic_integrator: kept for backward compatibility (unused).
    """
    logger = logging.getLogger(__name__)
    device_local = next(model.parameters()).device
    criterion = nn.MSELoss()

    if s_train.dim() != 2:
        raise ValueError(f"Expected s_train shape [T, 2*num_bodys], got {tuple(s_train.shape)}")
    if s_train.shape[1] % 2 != 0:
        raise ValueError(f"State dimension must be even (q,v pairs). Got {s_train.shape[1]}")

    num_bodys = s_train.shape[1] // 2
    expected_out = 2 * num_bodys
    if getattr(model, "dim", None) is not None and model.dim != expected_out:
        logger.warning(f"Model dim={model.dim} but expected {expected_out} for [dH/dq,dH/dp]")

    # Determine dt from t_train (assume fixed step)
    if len(t_train) < 2:
        raise ValueError("Need at least 2 time points to infer dt")
    dt = float((t_train[1] - t_train[0]).item())

    # Convert [q, v] -> [q, p] for model input
    masses = _masses_for_test_case(test_case, num_bodys, device_local)
    s_train_dev = s_train.to(device_local)
    body_qv = s_train_dev.view(-1, num_bodys, 2)
    body_qp = body_qv.clone()
    body_qp[:, :, 1] = body_qv[:, :, 1] * masses.view(1, -1)  # p = m*v

    # Build direct targets for Hamiltonian gradients
    # dH/dp = v (since p = m*v and dH/dp = p/m)
    target_dH_dp = body_qv[:, :, 1]

    # dH/dq = -m * a, where a = dv/dt from finite differences
    from Model.utils import estimate_temporal_gradient_finite_diff
    fd_order = int(train_params.get("grad_fd_order", 4))
    smooth_boundaries = bool(train_params.get("grad_smooth_boundaries", False))

    t_train_dev = t_train.to(device_local)
    target_accel = torch.zeros((body_qv.shape[0], num_bodys), dtype=body_qv.dtype, device=device_local)
    for body_idx in range(num_bodys):
        v_series = body_qv[:, body_idx, 1]
        a_series = estimate_temporal_gradient_finite_diff(
            v_series,
            t_train_dev,
            order=fd_order,
            smooth_boundaries=smooth_boundaries,
        )
        if a_series is None:
            raise RuntimeError(
                f"Failed to build acceleration targets for body={body_idx} using finite differences (order={fd_order})."
            )
        target_accel[:, body_idx] = a_series

    target_dH_dq = -(target_accel * masses.view(1, -1))

    # Final target: [dH/dq, dH/dp] flattened
    targets = torch.cat([target_dH_dq, target_dH_dp], dim=1)  # [T, 2*num_bodys]

    num_epochs = int(train_params.get("epochs", 450))
    outime_log = int(train_params.get("outime_log", 1))
    grad_clip = float(train_params.get("grad_clip", 0.0) or 0.0)
    batch_size = int(train_params.get("batch_size", 1))

    # Loss history
    loss_history = {"epoch": [], "train_loss": [], "learning_rate": []}
    best_loss_train = float("inf")
    best_epoch_train = -1

    logger.info(
        f"--- Supervised Hamiltonian-grad Training --- dt={dt}, bodies={num_bodys}, fd_order={fd_order}, batch_size={batch_size}"
    )
    logger.info(f"Using masses: {masses.detach().cpu().numpy()}")

    # Prepare dataloader
    model_inputs = body_qp.view(body_qp.shape[0], -1)
    dataset = TensorDataset(model_inputs, targets)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_params.get("num_workers", 0)),
        pin_memory=False,
    )

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        num_batches = 0
        for batch_input, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)

            pred = model(batch_input)
            loss = criterion(pred, batch_target)

            if torch.isfinite(loss):
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        loss_history["epoch"].append(epoch + 1)
        loss_history["train_loss"].append(avg_loss)
        loss_history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % outime_log == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch + 1:>5}/{num_epochs} | Train Loss: {avg_loss:.6e}")

        if scheduler:
            scheduler.step()

        if avg_loss < best_loss_train:
            best_loss_train = avg_loss
            best_epoch_train = epoch
            if output_paths.get("model"):
                from Model.utils import save_model_state
                save_model_state(model, output_paths["model"], model_filename="FNODE_best.pkl")

    logger.info(f"Best epoch: {best_epoch_train + 1 if best_epoch_train >= 0 else 'n/a'} loss={best_loss_train:.6e}")
    return model, loss_history



# def test_fnode(model, s0_test_core_state, t_test_eval_times, test_params, output_paths):
#     """
#     Generates trajectory predictions using a trained FNODE model (that predicts accelerations)
#     and an ODE solver.
#     Args:
#         model (FNODE): The trained FNODE model instance.
#         s0_test_core_state (torch.Tensor): Initial CORE state for testing,
#                                            shape [1, core_state_dim] (e.g., [1, x1, v1, x2, v2]).
#         t_test_eval_times (torch.Tensor): Time vector for prediction [steps].
#         test_params (dict): Dictionary containing parameters for testing, including
#                             'ode_solver_params' (rtol, atol, method).
#     Returns:
#         torch.Tensor or None: Predicted trajectory [steps, core_state_dim] or None on failure.
#     """
#     current_device = next(model.parameters()).device
#     model.eval()
#     logger.info(f"--- Starting FNODE Testing (State-to-Acceleration Model) ---")
#     logger.info(f"Model config for test: d_interest={model.d_interest}, output_dim (accelerations)={model.output_dim}")

#     if not TORCHDIFFEQ_AVAILABLE:
#         logger.error("torchdiffeq is required for test_fnode (ODE solving). Aborting.")
#         return None
#     if model.output_dim != model.num_bodys:
#         logger.error(f"Testing FNODE that should predict accelerations, but its output_dim ({model.output_dim}) "
#                      f"does not match num_bodys ({model.num_bodys}). Aborting.")
#         return None

#     # Validate t_test_eval_times for monotonicity
#     if len(t_test_eval_times) > 1:
#         diff_t = t_test_eval_times[1:] - t_test_eval_times[:-1]
#         is_increasing = (diff_t > 0).all().item()
#         is_decreasing = (diff_t < 0).all().item()
#         if not (is_increasing or is_decreasing):
#             logger.warning("Time vector for ODE solver is not strictly monotonic. Attempting to create a new one.")
#             t_start_val, t_end_val = t_test_eval_times[0].item(), t_test_eval_times[-1].item()
#             if t_start_val > t_end_val and not is_decreasing: t_end_val = t_start_val + (
#                         t_test_eval_times[-1] - t_test_eval_times[0]).abs().item()
#             t_test_eval_times = torch.linspace(min(t_start_val, t_end_val), max(t_start_val, t_end_val),
#                                                len(t_test_eval_times), device=current_device)
#             logger.info(
#                 f"New monotonic time vector for ODE solver: [{t_test_eval_times[0]:.2f}, ..., {t_test_eval_times[-1]:.2f}] length {len(t_test_eval_times)}")

#     if s0_test_core_state.dim() == 1:
#         s0_test_core_state = s0_test_core_state.unsqueeze(0)
#     if s0_test_core_state.shape[0] != 1:
#         logger.warning(
#             f"s0_test_core_state batch size is {s0_test_core_state.shape[0]} > 1. ODEINT will run for multiple initial conditions.")
#     if s0_test_core_state.shape[-1] != model.core_state_dim:
#         logger.error(f"Initial state s0_test_core_state dim ({s0_test_core_state.shape[-1]}) "
#                      f"does not match FNODE model's core_state_dim ({model.core_state_dim}). Aborting.")
#         return None

#     class AccelerationBasedODEFunc(nn.Module):
#         def __init__(self, fnode_accel_model_instance):
#             super().__init__()
#             self.fnode_model = fnode_accel_model_instance
#             self.num_bodys = self.fnode_model.num_bodys

#         def forward(self, t_scalar, y_current_core_state_batch):
#             fnode_input_batch = y_current_core_state_batch
#             if self.fnode_model.d_interest == 1:
#                 time_feature_batch = t_scalar.repeat(y_current_core_state_batch.shape[0]).unsqueeze(-1)
#                 fnode_input_batch = torch.cat([y_current_core_state_batch, time_feature_batch], dim=-1)
#             elif self.fnode_model.d_interest > 1:
#                 logger.error(
#                     f"ODE Func: FNODE model expects d_interest={self.fnode_model.d_interest} features. Feature construction not implemented here.")

#             predicted_accelerations_batch = self.fnode_model(fnode_input_batch)
#             dy_dt_batch = torch.zeros_like(y_current_core_state_batch)
#             for i in range(self.num_bodys):
#                 pos_idx = 2 * i;
#                 vel_idx = 2 * i + 1
#                 dy_dt_batch[..., pos_idx] = y_current_core_state_batch[..., vel_idx]
#                 dy_dt_batch[..., vel_idx] = predicted_accelerations_batch[..., i]

#             return dy_dt_batch

#     ode_func_for_solver = AccelerationBasedODEFunc(model).to(current_device)
#     ode_solver_params = test_params.get('ode_solver_params',
#                                         {'method': 'dopri5', 'rtol': 1e-7, 'atol': 1e-9})  # Adjusted defaults

#     logger.info(
#         f"Running ODE solver ({ode_solver_params.get('method', 'dopri5')}) with rtol={ode_solver_params.get('rtol')} atol={ode_solver_params.get('atol')}")
#     try:
#         with torch.no_grad():
#             pred_test_traj = ODEINT_FN(
#                 ode_func_for_solver,
#                 s0_test_core_state.to(current_device),
#                 t_test_eval_times.to(current_device),
#                 rtol=ode_solver_params.get('rtol'),
#                 atol=ode_solver_params.get('atol'),
#                 method=ode_solver_params.get('method')
#             )
#         if pred_test_traj.shape[1] == 1: pred_test_traj = pred_test_traj.squeeze(1)
#         logger.info(f"ODE solver finished. Predicted trajectory shape: {pred_test_traj.shape}")
#         if torch.isnan(pred_test_traj).any() or torch.isinf(pred_test_traj).any():
#             logger.warning("Prediction contains NaN or inf values. Clamping them to zero.")
#             pred_test_traj = torch.nan_to_num(pred_test_traj, nan=0.0, posinf=0.0, neginf=0.0)
#         return pred_test_traj
#     except Exception as ode_err:
#         logger.error(f"ODE integration failed during testing: {ode_err}", exc_info=True)
#         return None

# ---------------------------------------------------------------------------
# FNODE Hamiltonian Wrapper for Symplectic Integrators
# ---------------------------------------------------------------------------
# Simple functions for symplectic integrators with FNODE
def create_fnode_hamiltonian_functions(fnode_model, masses, device='cuda'):
    """
    Create simple dH_dq and dH_dp functions for symplectic integrators.

    Simplified design:
    - dH/dq = acceleration from FNODE
    - dH/dp = velocity = p/m

    Args:
        fnode_model: Trained FNODE model that outputs accelerations
        masses: Mass values for each body (scalar or tensor)
        device: Device for computations

    Returns:
        dH_dq_fn: Function that returns acceleration
        dH_dp_fn: Function that returns velocity
    """
    # Ensure masses is a tensor on the correct device
    if not isinstance(masses, torch.Tensor):
        masses = torch.tensor(masses, dtype=torch.float32)
    masses = masses.to(device)
    num_bodies = len(masses)

    def dH_dq(state_qp, model=None):
        """
        Return acceleration from FNODE.

        Args:
            state_qp: State in [q, p] format, shape [num_bodies, 2]
        Returns:
            Acceleration [num_bodies]
        """
        # Convert [q, p] to [q, v] for FNODE
        state_qv = state_qp.clone()
        state_qv[:, 1] = state_qp[:, 1] / masses  # p/m = v

        # Flatten for FNODE input
        state_flat = state_qv.flatten().unsqueeze(0)  # [1, num_bodies*2]

        # Get accelerations from FNODE
        with torch.no_grad():
            accelerations = fnode_model(state_flat).squeeze(0)  # [num_bodies]

        return -masses * accelerations

    def dH_dp(state_qp, model=None):
        """
        Return velocity = p/m.

        Args:
            state_qp: State in [q, p] format, shape [num_bodies, 2]
        Returns:
            Velocity [num_bodies]
        """
        return state_qp[:, 1] / masses

    return dH_dq, dH_dp


class FNODE_Hamiltonian_Wrapper:
    """Bridge an acceleration FNODE into (q,p) callbacks for separable Hamiltonian symplectic integrators.

    For a separable Hamiltonian H(q,p) = T(p) + V(q):
    - T(p) = (1/2) p^T M^{-1} p  (kinetic energy)
    - V(q) = potential energy

    The required gradients are:
    - ∂H/∂p = ∂T/∂p = M^{-1}p = v  (velocity from momentum)
    - ∂H/∂q = ∂V/∂q = -Ma(q)      (potential gradient, ideally only depends on q)

    For true Hamiltonian structure with energy conservation:
    - Acceleration a should ideally depend ONLY on position q: a = a(q)
    - If a = a(q,v) includes velocity dependence, it introduces non-conservative forces
    - This breaks the separable Hamiltonian assumption and degrades symplectic properties

    The FNODE model receives state [q, v] and outputs acceleration a.
    For best results with symplectic integrators, train the model to minimize velocity dependence.
    """

    def __init__(self, fnode_model, masses, device=None, mass_fn=None, position_only=False):
        """
        Initialize the Hamiltonian wrapper for FNODE.

        Args:
            fnode_model: The trained FNODE model that predicts accelerations
            masses: Array of masses for each body
            device: Computation device
            mass_fn: Optional function to compute position-dependent masses
            position_only: If True, pass zero velocity to model to enforce a(q) dependence only
                          This improves Hamiltonian structure but may reduce accuracy if model
                          was trained with velocity dependence
        """
        self.model = fnode_model
        model_device = next(self.model.parameters()).device
        self.device = model_device if device is None else torch.device(device)
        if self.device != model_device:
            self.device = model_device

        self.mass_fn = mass_fn
        self.position_only = position_only

        if not isinstance(masses, torch.Tensor):
            masses = torch.tensor(masses, dtype=torch.float32)
        self.masses = masses.to(self.device)
        self.num_bodies = int(self.masses.numel())

    def _masses_for_q(self, q):
        """Return per-body masses as shape [batch, num_bodies] on wrapper device."""
        if self.mass_fn is None:
            return self.masses.view(1, -1).expand(q.shape[0], -1)

        masses_dyn = self.mass_fn(q)
        if not isinstance(masses_dyn, torch.Tensor):
            masses_dyn = torch.tensor(masses_dyn, dtype=torch.float32)
        masses_dyn = masses_dyn.to(self.device)

        if masses_dyn.dim() == 1:
            masses_dyn = masses_dyn.view(1, -1).expand(q.shape[0], -1)
        if masses_dyn.shape != (q.shape[0], self.num_bodies):
            raise ValueError(
                f"mass_fn(q) must return shape {(q.shape[0], self.num_bodies)}, got {tuple(masses_dyn.shape)}"
            )
        return masses_dyn

    def dH_dq(self, state_qp, model=None):
        """
        Compute dH/dq = -Ma where a is the acceleration predicted by FNODE.

        Args:
            state_qp: State in (q, p) format with shape [num_bodies, 2] or [batch, num_bodies, 2]
            model: Optional model override (not used, kept for compatibility)

        Returns:
            dH/dq = -Ma (negative mass times acceleration)
        """
        if not isinstance(state_qp, torch.Tensor):
            state_qp = torch.tensor(state_qp, dtype=torch.float32, device=self.device)
        state_qp = state_qp.to(self.device)

        squeeze_batch = False
        if state_qp.dim() == 2:
            state_qp = state_qp.unsqueeze(0)
            squeeze_batch = True
        if state_qp.dim() != 3 or state_qp.shape[-1] != 2:
            raise ValueError(
                f"Expected state_qp shape [num_bodies,2] or [batch,num_bodies,2], got {tuple(state_qp.shape)}"
            )
        if state_qp.shape[1] != self.num_bodies:
            raise ValueError(
                f"Wrapper num_bodies={self.num_bodies} but got state with {state_qp.shape[1]} bodies"
            )

        # Extract positions and momenta
        q = state_qp[:, :, 0]  # positions
        p = state_qp[:, :, 1]  # momenta

        # Get masses for each body
        masses_eff = self._masses_for_q(q)

        # Convert momentum to velocity: v = p/m
        v = p / masses_eff

        # For better Hamiltonian structure, optionally use zero velocity
        # This enforces a(q) dependence only, improving energy conservation
        if self.position_only:
            v_for_model = torch.zeros_like(v)
        else:
            v_for_model = v

        # Prepare state in (q, v) format for FNODE
        state_qv = torch.stack([q, v_for_model], dim=-1)

        # Reshape to [batch, num_bodies * 2] for model input
        model_input = state_qv.reshape(state_qv.shape[0], -1)

        # Add time feature if needed
        if getattr(self.model, "d_interest", 0) == 1:
            time_feature = torch.zeros((model_input.shape[0], 1), device=self.device, dtype=model_input.dtype)
            model_input = torch.cat([model_input, time_feature], dim=-1)
        elif getattr(self.model, "d_interest", 0) not in (0, 1):
            raise NotImplementedError(
                f"FNODE_Hamiltonian_Wrapper only supports d_interest=0/1, got {self.model.d_interest}"
            )

        # Get acceleration prediction from FNODE
        with torch.no_grad():
            accelerations = self.model(model_input)

        if accelerations.dim() != 2 or accelerations.shape[1] != self.num_bodies:
            raise ValueError(
                f"Expected accelerations shape [batch,{self.num_bodies}], got {tuple(accelerations.shape)}"
            )

        # Return dH/dq = -Ma
        dH_dq = -(masses_eff * accelerations)
        return dH_dq.squeeze(0) if squeeze_batch else dH_dq

    def dH_dp(self, state_qp, model=None):
        """
        Compute dH/dp = v = p/m (velocity from momentum).

        Args:
            state_qp: State in (q, p) format with shape [num_bodies, 2] or [batch, num_bodies, 2]
            model: Optional model override (not used, kept for compatibility)

        Returns:
            dH/dp = v = p/m (velocity)
        """
        if not isinstance(state_qp, torch.Tensor):
            state_qp = torch.tensor(state_qp, dtype=torch.float32, device=self.device)
        state_qp = state_qp.to(self.device)

        squeeze_batch = False
        if state_qp.dim() == 2:
            state_qp = state_qp.unsqueeze(0)
            squeeze_batch = True
        if state_qp.dim() != 3 or state_qp.shape[-1] != 2:
            raise ValueError(
                f"Expected state_qp shape [num_bodies,2] or [batch,num_bodies,2], got {tuple(state_qp.shape)}"
            )
        if state_qp.shape[1] != self.num_bodies:
            raise ValueError(
                f"Wrapper num_bodies={self.num_bodies} but got state with {state_qp.shape[1]} bodies"
            )

        # Extract positions and momenta
        q = state_qp[:, :, 0]  # positions
        p = state_qp[:, :, 1]  # momenta

        # Get masses for each body
        masses_eff = self._masses_for_q(q)

        # Return dH/dp = v = p/m
        dH_dp = p / masses_eff
        return dH_dp.squeeze(0) if squeeze_batch else dH_dp

    def compute_energy(self, state_qp, k=None):
        """
        Compute total energy E = T + V for verification.
        For single mass-spring: E = (1/2)mv^2 + (1/2)kq^2

        Args:
            state_qp: State in (q, p) format
            k: Spring constant (for single mass-spring system)

        Returns:
            Total energy (scalar or tensor)
        """
        if not isinstance(state_qp, torch.Tensor):
            state_qp = torch.tensor(state_qp, dtype=torch.float32, device=self.device)

        if state_qp.dim() == 2:
            state_qp = state_qp.unsqueeze(0)

        q = state_qp[:, :, 0]  # positions
        p = state_qp[:, :, 1]  # momenta

        masses_eff = self._masses_for_q(q)
        v = p / masses_eff

        # Kinetic energy: T = (1/2)mv^2
        T = 0.5 * torch.sum(masses_eff * v**2, dim=-1)

        # Potential energy (for single mass-spring)
        if k is not None:
            V = 0.5 * k * torch.sum(q**2, dim=-1)
        else:
            # If k not provided, only return kinetic energy
            V = torch.zeros_like(T)

        return T + V


def test_fnode(model, s0_test_core_state, t_test_eval_times, test_params, output_paths):
    """
    Generates trajectory predictions using a trained FNODE model (that predicts accelerations)
    and an ODE solver (RK4 or dopri5).

    Args:
        model (FNODE): The trained FNODE model instance.
        s0_test_core_state (torch.Tensor): Initial CORE state for testing,
                                           shape [1, core_state_dim] (e.g., [1, x1, v1, x2, v2]).
        t_test_eval_times (torch.Tensor): Time vector for prediction [steps].
        test_params (dict): Dictionary containing parameters for testing.
                           Can include 'ode_method': 'rk4' or 'dopri5' (default: 'rk4')
                           For dopri5: 'rtol' and 'atol' (default: 1e-7, 1e-9)
        output_paths (dict): Dictionary of output paths (not used in this version).

    Returns:
        torch.Tensor or None: Predicted trajectory [steps, core_state_dim] or None on failure.
    """
    current_device = next(model.parameters()).device
    model.eval()

    # Get ODE solver method from test_params
    ode_method = test_params.get('ode_method', 'rk4')
    symplectic_methods = ['stormer_verlet', 'yoshida4', 'fukushima6']
    is_symplectic = ode_method in symplectic_methods

    logger.info(f"--- Starting FNODE Testing with {ode_method.upper()} ---")
    logger.info(f"Model config for test: d_interest={model.d_interest}, output_dim (accelerations)={model.output_dim}")
    if is_symplectic:
        logger.info("Using symplectic integrator - will convert state space [q,v] <-> [q,p]")

    if model.output_dim != model.num_bodys:
        logger.error(f"Testing FNODE that should predict accelerations, but its output_dim ({model.output_dim}) "
                     f"does not match num_bodys ({model.num_bodys}). Aborting.")
        return None

    # Prepare initial state
    if s0_test_core_state.dim() == 1:
        s0_test_core_state = s0_test_core_state.unsqueeze(0)
    if s0_test_core_state.shape[0] != 1:
        logger.warning(f"s0_test_core_state batch size is {s0_test_core_state.shape[0]} > 1. Using first initial condition only.")
        s0_test_core_state = s0_test_core_state[0:1]
    if s0_test_core_state.shape[-1] != model.core_state_dim:
        logger.error(f"Initial state s0_test_core_state dim ({s0_test_core_state.shape[-1]}) "
                     f"does not match FNODE model's core_state_dim ({model.core_state_dim}). Aborting.")
        return None

    # Calculate time step
    num_steps = len(t_test_eval_times)
    if num_steps < 2:
        logger.error("Need at least 2 time points to determine time step.")
        return None

    # Assume uniform time spacing (fixed step)
    dt = (t_test_eval_times[-1] - t_test_eval_times[0]).item() / (num_steps - 1)
    logger.info(f"Using fixed time step dt={dt:.6f} for {num_steps} steps")

    # Define force function wrapper for RK4
    def fnode_force_function(state, model=None):
        """
        Wrapper function to make FNODE model compatible with RK4 integrator.

        Args:
            state: Current state tensor [num_bodys, 2] or [batch, num_bodys, 2]
            model: The FNODE model instance

        Returns:
            Accelerations [num_bodys] or [batch, num_bodys]
        """
        if model is None:
            logger.error("Model is None in force function")
            return torch.zeros(state.shape[0], device=state.device)

        # Handle different input dimensions
        squeeze_batch = False
        if state.dim() == 2:  # [num_bodys, 2]
            state = state.unsqueeze(0)  # Add batch dimension
            squeeze_batch = True

        # Flatten state for model input: [batch, num_bodys * 2]
        batch_size = state.shape[0]
        num_bodys = state.shape[1]
        state_flat = state.view(batch_size, -1)

        # Add time feature if needed
        if model.d_interest == 1:
            # Get current time from context (we'll pass it through the state)
            # For simplicity, we'll use zero time or could track it externally
            time_feature = torch.zeros(batch_size, 1, device=state.device)
            model_input = torch.cat([state_flat, time_feature], dim=-1)
        else:
            model_input = state_flat

        # Get accelerations from model
        with torch.no_grad():
            accelerations = model(model_input)  # [batch, num_bodys]

        if squeeze_batch:
            accelerations = accelerations.squeeze(0)  # Remove batch dimension

        return accelerations

    # Reshape initial state for RK4 integrator
    num_bodys = model.num_bodys
    # For standard models: from [1, core_state_dim] to [num_bodys, 2]
    initial_state = s0_test_core_state.squeeze(0).view(num_bodys, 2).to(current_device)  # [num_bodys, 2]

    logger.info(f"Initial state shape for RK4: {initial_state.shape}")

    # Run integration based on selected method
    try:
        with torch.no_grad():
            if ode_method == 'dopri5':
                # Use torchdiffeq for dopri5
                if not TORCHDIFFEQ_AVAILABLE:
                    logger.error("torchdiffeq is required for dopri5 method. Falling back to RK4.")
                    ode_method = 'rk4'
                else:
                    # Define ODE function for torchdiffeq
                    def ode_func(t, state):
                        """
                        ODE function for torchdiffeq.
                        Args:
                            t: Current time (scalar)
                            state: Current state [core_state_dim]
                        Returns:
                            State derivatives [core_state_dim]
                        """
                        # Ensure everything runs on the same device as the model.
                        # `torchdiffeq` may call this with `state` on a different device
                        # if inputs aren't carefully aligned.
                        model_device = next(model.parameters()).device
                        if state.device != model_device:
                            state = state.to(model_device)

                        # Reshape state to [num_bodys, 2]
                        state_reshaped = state.view(num_bodys, 2)

                        # Get accelerations from model
                        accels = fnode_force_function(state_reshaped, model=model)

                        # Construct state derivative [velocities, accelerations]
                        velocities = state_reshaped[:, 1]  # Extract velocities
                        state_deriv = torch.zeros_like(state_reshaped)
                        state_deriv[:, 0] = velocities  # dx/dt = v
                        state_deriv[:, 1] = accels      # dv/dt = a

                        return state_deriv.view(-1)  # Flatten back

                    # Get tolerances from test_params
                    rtol = test_params.get('rtol', 1e-7)
                    atol = test_params.get('atol', 1e-9)

                    # Prepare initial state for dopri5 [core_state_dim]
                    model_device = next(model.parameters()).device
                    initial_state_flat = s0_test_core_state.squeeze(0).to(model_device)  # Remove batch dim
                    t_eval = t_test_eval_times.to(model_device)

                    # Integrate using dopri5
                    logger.info(f"Running dopri5 with rtol={rtol}, atol={atol}")
                    pred_test_traj = odeint(
                        ode_func,
                        initial_state_flat,
                        t_eval,
                        method='dopri5',
                        rtol=rtol,
                        atol=atol
                    )

                    logger.info(f"Dopri5 integration finished. Predicted trajectory shape: {pred_test_traj.shape}")

            if ode_method == 'rk4' or (ode_method == 'dopri5' and not TORCHDIFFEQ_AVAILABLE):
                # Use the standard runge_kutta_four_multiple_body function
                pred_trajectory = runge_kutta_four_multiple_body(
                    bodys=initial_state,
                    force_function=fnode_force_function,
                    num_steps=num_steps,
                    time_step=dt,
                    if_final_state=False,
                    model=model
                )

                # pred_trajectory shape: [num_steps, num_bodys, 2]
                # Reshape to match expected output: [num_steps, core_state_dim]
                pred_test_traj = pred_trajectory.view(num_steps, -1)

                logger.info(f"RK4 integration finished. Predicted trajectory shape: {pred_test_traj.shape}")

            # Handle symplectic integrators
            if is_symplectic:
                # Get masses for the system (use custom mass_value if provided)
                mass_value = test_params.get('mass_value', 10.0)
                masses = mass_value * torch.ones(num_bodys, dtype=torch.float32, device=current_device)
                logger.info(f"Using masses: {masses.cpu().numpy()}")

                # Convert initial state from [q, v] to [q, p]
                initial_state_qp = initial_state.clone()
                initial_state_qp[:, 1] = initial_state[:, 1] * masses  # p = m * v


                # use_analytic_dh = bool(test_params.get('symplectic_use_analytic_dh', False))
                # if use_analytic_dh:
                #     logger.info("Using analytical dH_dq_smp/dH_dp_smp (bypassing FNODE wrapper/model) for symplectic rollout")
                #     from functools import partial
                #     from Model.force_fun import dH_dq_smp, dH_dp_smp
                #     k1 = float(test_params.get('k1', 50.0))
                #     dH_dq_fn = partial(dH_dq_smp, mass=float(mass_value), k1=k1)
                #     dH_dp_fn = partial(dH_dp_smp, mass=float(mass_value))
                # else:
                logger.info("Using simplified FNODE Hamiltonian functions for symplectic rollout")
                dH_dq_fn, dH_dp_fn = create_fnode_hamiltonian_functions(model, masses, current_device)
                
                # Run the appropriate symplectic integrator
                if ode_method == 'stormer_verlet':
                    logger.info("Running Störmer-Verlet integration")
                    pred_trajectory_qp = sep_stormer_verlet_multiple_body(
                        bodys=initial_state_qp,
                        dH_dq=dH_dq_fn,
                        dH_dp=dH_dp_fn,
                        num_steps=num_steps,
                        time_step=dt,
                        if_final_state=False,
                        current_device=current_device
                    )
                elif ode_method == 'yoshida4':
                    logger.info("Running Yoshida4 integration")
                    pred_trajectory_qp = yoshida4_multiple_body(
                        bodys=initial_state_qp,
                        dH_dq=dH_dq_fn,
                        dH_dp=dH_dp_fn,
                        num_steps=num_steps,
                        time_step=dt,
                        if_final_state=False,
                        current_device=current_device
                    )
                elif ode_method == 'fukushima6':
                    logger.info("Running Fukushima6 integration")
                    pred_trajectory_qp = fukushima6_multiple_body(
                        bodys=initial_state_qp,
                        dH_dq=dH_dq_fn,
                        dH_dp=dH_dp_fn,
                        num_steps=num_steps,
                        time_step=dt,
                        if_final_state=False,
                        device=current_device
                    )

                # Convert back from [q, p] to [q, v]
                # pred_trajectory_qp shape: [num_steps, num_bodys, 2]
                pred_trajectory_qv = pred_trajectory_qp.clone()
                pred_trajectory_qv[:, :, 1] = pred_trajectory_qp[:, :, 1] / masses.unsqueeze(0)  # v = p / m

                # Reshape to match expected output: [num_steps, core_state_dim]
                pred_test_traj = pred_trajectory_qv.view(num_steps, -1)

                logger.info(f"{ode_method.upper()} integration finished. Predicted trajectory shape: {pred_test_traj.shape}")

        # Check for NaN or Inf values
        if torch.isnan(pred_test_traj).any() or torch.isinf(pred_test_traj).any():
            nan_count = torch.isnan(pred_test_traj).sum().item()
            inf_count = torch.isinf(pred_test_traj).sum().item()
            logger.warning(f"Prediction contains {nan_count} NaN and {inf_count} Inf values. Clamping them.")
            pred_test_traj = torch.nan_to_num(pred_test_traj, nan=0.0, posinf=1e6, neginf=-1e6)

        # Verify output shape
        expected_shape = (num_steps, model.core_state_dim)
        if pred_test_traj.shape != expected_shape:
            logger.error(f"Output shape {pred_test_traj.shape} doesn't match expected {expected_shape}")
            return None

        return pred_test_traj

    except Exception as integration_err:
        logger.error(f"{ode_method.upper()} integration failed during testing: {integration_err}", exc_info=True)
        return None


def test_MBDNODE(numerical_methods,model,body,num_steps,dt,device='cuda'):
    print("start testing the MBDNODE model")
    print("body shape is "+str(body.shape))
    print("model is ",model)
    print("numerical_methods is "+numerical_methods)
    #reshape the body_tensor to (num_data,num_bodys*2)
    num_body = body.shape[0]
    body_tensor = torch.tensor(body, dtype=torch.float32, device=device)

    # Data in this repo is generally stored as [q, v].
    # Symplectic integrators in `Model/integrator.py` expect canonical [q, p].
    if numerical_methods in ["sep_sv", "yoshida4", "fukushima6"]:
        masses = _mbdnode_masses(num_body, body_tensor.device)
        body_tensor = body_tensor.clone()
        body_tensor[:, 1] = body_tensor[:, 1] * masses

    # Handle symplectic methods
    if numerical_methods in ["sep_sv", "yoshida4", "fukushima6"]:
        if numerical_methods == "sep_sv":
            testing_result = sep_stormer_verlet_multiple_body(
                body_tensor, neural_network_force_function_MBDNODE_symplectic_dH_dq,
                neural_network_force_function_MBDNODE_symplectic_dH_dp,
                num_steps, dt, if_final_state=False, model=model)
        elif numerical_methods == "yoshida4":
            testing_result = yoshida4_multiple_body(
                body_tensor, neural_network_force_function_MBDNODE_symplectic_dH_dq,
                neural_network_force_function_MBDNODE_symplectic_dH_dp,
                num_steps, dt, if_final_state=False, model=model)
        elif numerical_methods == "fukushima6":
            testing_result = fukushima6_multiple_body(
                body_tensor, neural_network_force_function_MBDNODE_symplectic_dH_dq,
                neural_network_force_function_MBDNODE_symplectic_dH_dp,
                num_steps, dt, if_final_state=False, model=model)
    # Handle standard methods
    elif numerical_methods=="fe":
        testing_result = forward_euler_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    elif numerical_methods=="rk4":
        testing_result = runge_kutta_four_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    elif numerical_methods=="midpoint":
        testing_result = midpoint_method_multiple_body(body_tensor,neural_network_force_function_MBDNODE,num_steps,dt,if_final_state=False,model=model)
    else:
        raise ValueError(f"The numerical method '{numerical_methods}' is not supported. Supported methods: fe, rk4, midpoint, sep_sv, yoshida4, fukushima6")

    # Convert back to [q, v] for downstream plotting/metrics consistency.
    if numerical_methods in ["sep_sv", "yoshida4", "fukushima6"]:
        masses = _mbdnode_masses(num_body, testing_result.device)
        # testing_result: [num_steps, num_bodys, 2] with [q, p]
        testing_result = testing_result.clone()
        testing_result[:, :, 1] = testing_result[:, :, 1] / masses

    return testing_result


def test_MBDNODE_symplectic(numerical_methods, model, body, num_steps, dt, device='cuda'):
    """
    Test MBDNODE_Symplectic model with symplectic integration.

    Args:
        numerical_methods: Symplectic integration method ('sep_sv', 'yoshida4', 'fukushima6')
        model: MBDNODE_Symplectic model instance
        body: Initial state with shape [num_bodys, 2] containing [position, velocity]
        num_steps: Number of integration steps
        dt: Time step
        device: Device to run on

    Returns:
        Testing result with shape [num_steps, num_bodys, 2] in [q, v] format
    """
    print("Testing MBDNODE_Symplectic model")
    print(f"body shape: {body.shape}")
    print(f"model: {model}")
    print(f"numerical_methods: {numerical_methods}")

    # Prepare initial state
    num_bodys = body.shape[0]
    body_tensor = torch.tensor(body, dtype=torch.float32, device=device)

    # Get masses for v->p conversion
    masses = _mbdnode_masses(num_bodys, device)

    # Convert [q, v] to [q, p] for symplectic integration
    body_tensor_qp = body_tensor.clone()
    body_tensor_qp[:, 1] = body_tensor[:, 1] * masses  # v -> p

    # Numerical methods mapping for symplectic integrators
    methods = {
        "sep_sv": sep_stormer_verlet_multiple_body,
        "yoshida4": yoshida4_multiple_body,
        "fukushima6": fukushima6_multiple_body
    }

    if numerical_methods not in methods:
        raise ValueError(f"Unknown symplectic method: {numerical_methods}. Supported: {list(methods.keys())}")

    integration_method = methods[numerical_methods]

    # Integrate using the symplectic method
    testing_result = integration_method(
        body_tensor_qp,
        neural_network_force_function_MBDNODE_symplectic_dH_dq,
        neural_network_force_function_MBDNODE_symplectic_dH_dp,
        num_steps,
        dt,
        if_final_state=False,  # Get full trajectory
        model=model
    )

    # Convert back from [q, p] to [q, v] for consistency
    testing_result = testing_result.clone()
    testing_result[:, :, 1] = testing_result[:, :, 1] / masses  # p -> v

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
