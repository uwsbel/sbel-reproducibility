"""
Parameterized vehicle model based on original rom_vehicle.py
This version supports variable vehicle parameters for training
Includes GPU batch processing for fast data generation
"""

import numpy as np

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, GPU acceleration disabled")


class ParameterizedVehModel():
    """
    Parameterized version of simplifiedVehModel that accepts variable parameters
    No visualization support - pure dynamics only
    """

    def __init__(self, system, state, control, dt, Visualize=False, params=None):
        # Note: system and Visualize parameters are kept for compatibility but ignored
        # state = [x, y, theta, v]
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]
        self.accel = 0

        # control = [alpha, beta]
        self.alpha = control[0]
        self.beta = control[1]
        self.dt = dt  # time step

        # Vehicle parameters - either from params dict or default values
        if params is not None:
            self.l = params.get('l', 2.5)
            self.r_wheel = params.get('r_wheel', 0.3)
            self.i_wheel = params.get('i_wheel', 0.6)
            self.tau0 = params.get('tau0', 100)
            self.omega0 = params.get('omega0', 1200)
            self.c0 = params.get('c0', 0.01)
            self.c1 = params.get('c1', 0.02)
            self.delta = params.get('delta', 0.667)
            self.gamma = params.get('gamma', 1/3)
        else:
            # Default parameters (original values from rom_vehicle.py)
            self.l = 2.5
            self.r_wheel = 0.3
            self.i_wheel = 0.6
            self.tau0 = 100
            self.omega0 = 1200
            self.c0 = 0.01
            self.c1 = 0.02
            self.delta = 0.667
            self.gamma = 1/3

    def update(self, control):
        # Update control inputs
        self.alpha = control[0]
        self.beta = control[1]

        # Use instance parameters for dynamics calculation
        omega_m = self.v / (self.r_wheel * self.gamma)
        helpfun1 = -self.tau0 * omega_m / self.omega0 + self.tau0
        helpfunT = self.alpha * helpfun1 - self.c1 * omega_m - self.c0

        # Update state
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta += (self.v * np.tan(self.beta * self.delta) / self.l) * self.dt
        self.accel = helpfunT * self.gamma / self.i_wheel * self.r_wheel
        self.v += self.accel * self.dt

    # NOTE: Keep dynamics consistent with the original ROM (rom_vehicle.py):
    # do not clamp velocity to be non-negative.

    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]

    def get_state(self):
        return [self.x, self.y, self.theta, self.v, self.accel]

    def get_parameters(self):
        """Return current vehicle parameters as a dictionary"""
        return {
            'l': self.l,
            'r_wheel': self.r_wheel,
            'i_wheel': self.i_wheel,
            'tau0': self.tau0,
            'omega0': self.omega0,
            'c0': self.c0,
            'c1': self.c1,
            'delta': self.delta,
            'gamma': self.gamma
        }

    def set_parameters(self, params):
        """Update vehicle parameters"""
        if params is not None:
            self.l = params.get('l', self.l)
            self.r_wheel = params.get('r_wheel', self.r_wheel)
            self.i_wheel = params.get('i_wheel', self.i_wheel)
            self.tau0 = params.get('tau0', self.tau0)
            self.omega0 = params.get('omega0', self.omega0)
            self.c0 = params.get('c0', self.c0)
            self.c1 = params.get('c1', self.c1)
            self.delta = params.get('delta', self.delta)
            self.gamma = params.get('gamma', self.gamma)


# For backward compatibility, create an alias
simplifiedVehModel = ParameterizedVehModel


class GPUBatchVehicleModel:
    """
    GPU batch processing version for fast parallel simulation
    Implements exactly the same dynamics as ParameterizedVehModel
    """

    def __init__(self, params_batch, dt, device=None):
        """
        Initialize GPU batch vehicle model

        Args:
            params_batch: numpy array of shape (N, 9) with N parameter combinations
                         [l, r_wheel, i_wheel, tau0, omega0, c0, c1, delta, gamma]
            dt: time step
            device: torch device ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU batch processing")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.dt = dt
        self.batch_size = len(params_batch)

        # Convert parameters to GPU tensors
        params_tensor = torch.tensor(params_batch, dtype=torch.float32, device=device)

        # Extract individual parameters (matching PARAM_ORDER from parameter_sampler.py)
        self.l = params_tensor[:, 0]
        self.r_wheel = params_tensor[:, 1]
        self.i_wheel = params_tensor[:, 2]
        self.tau0 = params_tensor[:, 3]
        self.omega0 = params_tensor[:, 4]
        self.c0 = params_tensor[:, 5]
        self.c1 = params_tensor[:, 6]
        self.delta = params_tensor[:, 7]
        self.gamma = params_tensor[:, 8]

    def update_batch(self, states, controls):
        """
        Update all vehicle states in parallel using GPU

        Args:
            states: torch tensor of shape (N, 4) [x, y, theta, v]
            controls: torch tensor of shape (N, 2) [alpha, beta]

        Returns:
            new_states: torch tensor of shape (N, 4) with updated states
            accelerations: torch tensor of shape (N,) with accelerations
        """
        # Extract current states
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        v = states[:, 3]

        # Extract controls
        alpha = controls[:, 0]
        beta = controls[:, 1]

        # Implement EXACT same dynamics as ParameterizedVehModel.update()
        # This ensures complete consistency with the original implementation

        # omega_m = self.v / (self.r_wheel * self.gamma)
        omega_m = v / (self.r_wheel * self.gamma)

        # helpfun1 = -self.tau0 * omega_m / self.omega0 + self.tau0
        helpfun1 = -self.tau0 * omega_m / self.omega0 + self.tau0

        # helpfunT = self.alpha * helpfun1 - self.c1 * omega_m - self.c0
        helpfunT = alpha * helpfun1 - self.c1 * omega_m - self.c0

        # self.accel = helpfunT * self.gamma / self.i_wheel * self.r_wheel
        accel = helpfunT * self.gamma / self.i_wheel * self.r_wheel

        # Update states (same as original)
        # self.v += self.accel * self.dt
        v_new = v + accel * self.dt

    # NOTE: Keep dynamics consistent with the original ROM (rom_vehicle.py):
    # do not clamp velocity to be non-negative.

        # self.theta += (self.v * np.tan(self.beta * self.delta) / self.l) * self.dt
        theta_new = theta + (v * torch.tan(beta * self.delta) / self.l) * self.dt

        # self.x += self.v * np.cos(self.theta) * self.dt
        x_new = x + v * torch.cos(theta) * self.dt

        # self.y += self.v * np.sin(self.theta) * self.dt
        y_new = y + v * torch.sin(theta) * self.dt

        # Stack into new state tensor
        new_states = torch.stack([x_new, y_new, theta_new, v_new], dim=1)

        return new_states, accel

    def generate_controls_gpu(self, t, v):
        """
        Generate control inputs directly on GPU to match the original control function.
        This reimplements the control logic from generate_control_inputs() on GPU.

        Args:
            t: time (scalar)
            v: velocities tensor of shape (N,) on GPU

        Returns:
            controls: tensor of shape (N, 2) [alpha, beta] on GPU
        """
        accel_time = 5.0

        if t <= accel_time:
            # Phase 1: Straight-line acceleration
            ramp_time = 4.0
            s = min(t / ramp_time, 1.0)
            alpha = 0.25 * (1.0 - np.cos(np.pi * s))
            beta = 0.0

            # Create tensors for all vehicles (same control)
            alpha_tensor = torch.full((self.batch_size,), alpha, device=self.device)
            beta_tensor = torch.full((self.batch_size,), beta, device=self.device)
        else:
            # Phase 2: S-curve with deceleration
            t_s = t - accel_time
            alpha_base = 0.5 * np.exp(-t_s / 10.0)
            amplitude = 0.4
            frequency = 0.3
            beta_base = amplitude * np.sin(2 * np.pi * frequency * t_s)

            # Create tensors
            alpha_tensor = torch.full((self.batch_size,), alpha_base, device=self.device)
            beta_tensor = torch.full((self.batch_size,), beta_base, device=self.device)

        # Clip values
        alpha_tensor = torch.clamp(alpha_tensor, 0.0, 1.0)
        beta_tensor = torch.clamp(beta_tensor, -1.0, 1.0)

        return torch.stack([alpha_tensor, beta_tensor], dim=1)

    def simulate_batch(self, num_steps, control_func=None):
        """
        Simulate all vehicles for num_steps (optimized GPU version)

        Args:
            num_steps: number of simulation steps
            control_func: function(t, v) -> (alpha, beta) for generating controls
                         If 'gpu', uses built-in GPU control generation

        Returns:
            trajectories: numpy array of shape (N, num_steps, 6) [x, y, v, theta, alpha, beta]
            accelerations: numpy array of shape (N, num_steps)
        """
        # Initialize states [x, y, theta, v] all at origin with zero velocity
        states = torch.zeros((self.batch_size, 4), device=self.device)

        # Pre-allocate all storage to avoid dynamic memory allocation
        trajectories = torch.zeros((self.batch_size, num_steps, 6), device=self.device)
        accelerations = torch.zeros((self.batch_size, num_steps), device=self.device)

        # Disable gradient computation for faster inference
        with torch.no_grad():
            # Main simulation loop - must be sequential due to state dependencies
            for step in range(num_steps):
                t = step * self.dt

                # Generate controls
                if control_func == 'gpu' or control_func is not None:
                    v = states[:, 3]
                    controls = self.generate_controls_gpu(t, v)
                else:
                    controls = torch.zeros((self.batch_size, 2), device=self.device)

                # Store current state and controls directly (avoid intermediate tensors)
                trajectories[:, step, 0] = states[:, 0]  # x
                trajectories[:, step, 1] = states[:, 1]  # y
                trajectories[:, step, 2] = states[:, 3]  # v
                trajectories[:, step, 3] = states[:, 2]  # theta
                trajectories[:, step, 4:6] = controls

                # Update states
                states, accel = self.update_batch(states, controls)
                accelerations[:, step] = accel

        # Single GPU to CPU transfer
        return trajectories.cpu().numpy(), accelerations.cpu().numpy()


def verify_gpu_cpu_consistency(params_dict, dt=0.01, num_steps=100):
    """
    Verify that GPU and CPU implementations produce identical results

    Args:
        params_dict: dictionary of vehicle parameters
        dt: time step
        num_steps: number of steps to simulate

    Returns:
        bool: True if implementations are consistent
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping GPU verification")
        return True

    # Simple test control function
    def test_control(t, v):
        alpha = 0.3 * np.sin(t)
        beta = 0.1 * np.cos(t * 0.5)
        return alpha, beta

    # CPU version
    cpu_vehicle = ParameterizedVehModel(None, [0, 0, 0, 0], [0, 0], dt,
                                        Visualize=False, params=params_dict)
    cpu_states = []
    cpu_accels = []

    for step in range(num_steps):
        t = step * dt
        state = cpu_vehicle.get_state()
        cpu_states.append(state[:4])
        cpu_accels.append(state[4])

        control = test_control(t, state[3])
        cpu_vehicle.update(control)

    cpu_states = np.array(cpu_states)
    cpu_accels = np.array(cpu_accels)

    # GPU version
    params_array = np.array([[
        params_dict['l'], params_dict['r_wheel'], params_dict['i_wheel'],
        params_dict['tau0'], params_dict['omega0'], params_dict['c0'],
        params_dict['c1'], params_dict['delta'], params_dict['gamma']
    ]])

    gpu_model = GPUBatchVehicleModel(params_array, dt)
    gpu_traj, gpu_accels = gpu_model.simulate_batch(num_steps, test_control)

    # Extract single trajectory (we only have one vehicle)
    # GPU format is [x, y, v, theta], need to reorder to [x, y, theta, v]
    gpu_traj_single = gpu_traj[0]  # Shape: (num_steps, 6)
    gpu_states = gpu_traj_single[:, [0, 1, 3, 2]]  # Reorder to [x, y, theta, v]
    gpu_accels = gpu_accels[0]

    # Debug: print shapes
    print(f"  CPU states shape: {cpu_states.shape}")
    print(f"  GPU states shape: {gpu_states.shape}")

    # Compare results
    state_error = np.max(np.abs(cpu_states - gpu_states))
    accel_error = np.max(np.abs(cpu_accels - gpu_accels))

    print(f"Max state error: {state_error:.2e}")
    print(f"Max accel error: {accel_error:.2e}")

    # Check if errors are within tolerance
    tolerance = 1e-5
    if state_error < tolerance and accel_error < tolerance:
        print("✓ GPU and CPU implementations are consistent!")
        return True
    else:
        print("✗ GPU and CPU implementations differ!")
        return False