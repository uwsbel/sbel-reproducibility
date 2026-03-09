import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import scipy.integrate
from Model.utils import *
from Model.force_fun import *
from Model.model import *

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Legacy PNODE-style generators for Single Mass Spring family (used by LNN/HNN)
# ---------------------------------------------------------------------------
def generate_training_data(test_case, numerical_methods, dt, num_steps,
                           if_external_force=False, external_force_function=None,
                           aug=False, model=None):
    """
    Generate trajectories for energy-conserving test cases (Single_Mass_Spring variants).
    Mirrors PNODE behavior to keep LNN/HNN scripts compatible.
    """
    logger.info("Generating training data for test case: %s", test_case)
    bodys = None
    training_set = None

    if test_case == "Single_Mass_Spring":
        initial_state = torch.tensor([[1.0, 0.0]], device=device)  # [[position, velocity]]
        bodys = []

        if numerical_methods == "rk4":
            # Generate a trajectory
            trajectory = generate_trajectory_rk4_sms(
                initial_state, force_sms, dt, num_steps,
                if_external_force=if_external_force,
                external_force_function=external_force_function
            )
            bodys.append(trajectory)

        elif numerical_methods == "analytical":
            # Generate analytically
            trajectory = generate_analytical_sms(initial_state, dt, num_steps)
            bodys.append(trajectory)

        elif numerical_methods == "fe":
            # Forward Euler
            trajectory = generate_trajectory_forward_euler_sms(
                initial_state, force_sms, dt, num_steps
            )
            bodys.append(trajectory)

        elif numerical_methods == "midpoint":
            # Midpoint (Leapfrog/Stormer-Verlet)
            trajectory = generate_trajectory_midpoint_sms(
                initial_state, force_sms, dt, num_steps
            )
            bodys.append(trajectory)

        else:
            raise ValueError(f"Unknown numerical method: {numerical_methods}")

        # Convert to tensor
        training_set = torch.stack(bodys).to(device)  # Shape: [1, num_steps, 2]
        logger.info("Training set shape: %s", training_set.shape)

    elif test_case == "Single_Mass_Spring_Damper":
        initial_state = torch.tensor([[1.0, 0.0]], device=device)  # [[position, velocity]]
        bodys = []

        if numerical_methods == "rk4":
            # Generate a trajectory
            trajectory = generate_trajectory_rk4_smsd(
                initial_state, force_smsd, dt, num_steps
            )
            bodys.append(trajectory)

        else:
            raise ValueError(f"Unknown numerical method for SMSD: {numerical_methods}")

        # Convert to tensor
        training_set = torch.stack(bodys).to(device)  # Shape: [1, num_steps, 2]
        logger.info("Training set shape: %s", training_set.shape)

    else:
        raise ValueError(f"Test case not recognized: {test_case}")

    return training_set

# ---------------------------------------------------------------------------
# Updated trajectory generation functions using specific force functions
# ---------------------------------------------------------------------------

def generate_trajectory_rk4_sms(initial_state, force_func, dt, num_steps,
                                 if_external_force=False,
                                 external_force_function=None):
    """Generate trajectory for single mass spring using RK4 with explicit force function"""
    logger.debug(f"Generating SMS trajectory: RK4, dt={dt}, steps={num_steps}")
    trajectory = []
    state = initial_state.clone()

    for step in range(num_steps):
        trajectory.append(state.clone())

        # RK4 integration
        def dynamics(s):
            # s shape: [1, 2] where s[0,0] is position, s[0,1] is velocity
            acc = force_func(s, if_external_force=if_external_force,
                            external_force_function=external_force_function, t=step*dt)
            # Ensure acc has the right shape for concatenation
            if acc.ndim == 1:
                acc = acc.unsqueeze(1)  # Convert from [batch] to [batch, 1]
            # For SMS: dx/dt = v, dv/dt = a
            return torch.cat([s[:, 1:2], acc], dim=1)

        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)

        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    return torch.stack(trajectory).squeeze(1)  # Shape: [num_steps, 2]


def generate_analytical_sms(initial_state, dt, num_steps):
    """Generate analytical solution for single mass spring (no damping)"""
    logger.debug(f"Generating SMS trajectory: Analytical, dt={dt}, steps={num_steps}")

    x0 = initial_state[0, 0].item()
    v0 = initial_state[0, 1].item()
    k = 50.0  # Spring constant
    m = 10.0  # Mass
    omega = np.sqrt(k/m)

    trajectory = []
    for step in range(num_steps):
        t = step * dt
        x = x0 * np.cos(omega * t) + v0/omega * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        trajectory.append(torch.tensor([x, v], device=device))

    return torch.stack(trajectory)  # Shape: [num_steps, 2]

def generate_analytical_smsd(initial_state, dt, num_steps):
    """Generate analytical solution for single mass spring damper (underdamped case)
    
    System: m*x'' + b*x' + k*x = 0
    Parameters must match force_smsd(): m=10, k=50, b=2 (no cubic term for analytical)
    """
    logger.debug(f"Generating SMSD trajectory: Analytical, dt={dt}, steps={num_steps}")

    x0 = initial_state[0, 0].item()
    v0 = initial_state[0, 1].item()
    
    # Parameters matching force_smsd (without cubic term for analytical solution)
    k = 50.0   # Spring constant
    m = 10.0   # Mass
    b = 2.0    # Damping coefficient
    
    # Underdamped oscillator parameters
    omega_0 = np.sqrt(k / m)                      # Natural frequency = sqrt(5) ≈ 2.236
    gamma = b / (2 * m)                           # Decay rate = 0.1
    zeta = b / (2 * np.sqrt(k * m))              # Damping ratio ≈ 0.045 (underdamped)
    omega_d = omega_0 * np.sqrt(1 - zeta**2)     # Damped frequency ≈ 2.234
    
    # Coefficients for x(t) = exp(-gamma*t) * (A*cos(omega_d*t) + B*sin(omega_d*t))
    A = x0
    B = (v0 + gamma * x0) / omega_d
    
    trajectory = []
    for step in range(num_steps):
        t = step * dt
        exp_decay = np.exp(-gamma * t)
        cos_term = np.cos(omega_d * t)
        sin_term = np.sin(omega_d * t)
        
        # Position: x(t) = exp(-gamma*t) * (A*cos(omega_d*t) + B*sin(omega_d*t))
        x = exp_decay * (A * cos_term + B * sin_term)
        
        # Velocity: v(t) = dx/dt
        # v = exp(-gamma*t) * [(-gamma*A + omega_d*B)*cos + (-gamma*B - omega_d*A)*sin]
        v = exp_decay * ((-gamma * A + omega_d * B) * cos_term + 
                         (-gamma * B - omega_d * A) * sin_term)
        
        trajectory.append(torch.tensor([x, v], device=device))

    return torch.stack(trajectory)  # Shape: [num_steps, 2]


def generate_analytical_tmsd(initial_state, dt, num_steps):
    """Generate analytical solution for two mass spring damper system.
    Note: This is a placeholder function and needs to be implemented based on the specific system dynamics.
    """
    logger.debug(f"Generating TMSD trajectory: Analytical, dt={dt}, steps={num_steps}")

    # NOTE:
    # This implements an *exact* one-step discrete mapping for a linear,
    # time-invariant (LTI) triple mass–spring–damper chain:
    #   z_dot = A z  =>  z_{k+1} = expm(A*dt) z_k
    # where z = [x1,x2,x3,v1,v2,v3].
    #
    # IMPORTANT: This must match your force_tmsd() topology; if your force
    # function uses different connectivity/parameters, update the constants
    # below accordingly.

    # ---- Parameters (match Model/force_fun.py::force_tmsd by default) ----
    # force_tmsd uses masses [100, 10, 1], k=50, c=2 with a serial chain:
    # ground --(k,c)-- m1 --(k,c)-- m2 --(k,c)-- m3
    m1, m2, m3 = 100.0, 10.0, 1.0
    k1 = k2 = k3 = 50.0
    d1 = d2 = d3 = 2.0

    # ---- Build continuous-time A matrix (6x6) in numpy ----
    # x_dot = v
    # v_dot = B x + D v
    B = np.array(
        [
            [-(k1 + k2) / m1, k2 / m1, 0.0],
            [k2 / m2, -(k2 + k3) / m2, k3 / m2],
            [0.0, k3 / m3, -k3 / m3],
        ],
        dtype=np.float64,
    )

    D = np.array(
        [
            [-(d1 + d2) / m1, d2 / m1, 0.0],
            [d2 / m2, -(d2 + d3) / m2, d3 / m2],
            [0.0, d3 / m3, -d3 / m3],
        ],
        dtype=np.float64,
    )

    Z = np.zeros((3, 3), dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    A = np.block([[Z, I], [B, D]])  # shape (6, 6)

    # ---- expm(A*dt) (prefer scipy if available; fall back to eig) ----
    def _expm_fallback(A_dt: np.ndarray) -> np.ndarray:
        # Fallback: eigen-decomposition expm for diagonalizable matrices.
        # scipy.linalg.expm is preferred when installed.
        w, V = np.linalg.eig(A_dt)
        Vinv = np.linalg.inv(V)
        M = V @ np.diag(np.exp(w)) @ Vinv
        if np.max(np.abs(np.imag(M))) < 1e-10:
            M = np.real(M)
        return M

    try:
        from scipy.linalg import expm as _scipy_expm  # type: ignore

        M = _scipy_expm(A * float(dt))
    except Exception:
        M = _expm_fallback(A * float(dt))

    # ---- Iterate using exact one-step mapping ----
    # initial_state expected shape is typically [3,2] for this repo.
    # Convert it to z=[x1,x2,x3,v1,v2,v3] for the mapping.
    if initial_state.dim() == 2 and initial_state.shape[-1] == 2 and initial_state.shape[0] == 3:
        # [3,2] -> [1,6]
        z0 = torch.cat([initial_state[:, 0], initial_state[:, 1]], dim=0).unsqueeze(0)
    elif initial_state.dim() == 3 and initial_state.shape[-1] == 2 and initial_state.shape[1] == 3:
        # [B,3,2] -> [B,6]
        z0 = torch.cat([initial_state[:, :, 0], initial_state[:, :, 1]], dim=1)
    elif initial_state.dim() == 2 and initial_state.shape[-1] == 6:
        # already [B,6]
        z0 = initial_state
    elif initial_state.dim() == 1 and initial_state.shape[0] == 6:
        z0 = initial_state.unsqueeze(0)
    else:
        raise ValueError(
            "Unsupported initial_state shape for TMSD analytical generator: "
            f"{tuple(initial_state.shape)}; expected [3,2], [B,3,2], [B,6], or [6]."
        )

    device_local = initial_state.device
    M_torch = torch.tensor(M, dtype=initial_state.dtype, device=device_local)

    trajectory = []
    state = z0.clone().to(device_local)
    for _ in range(num_steps):
        trajectory.append(state.clone())
        # batch row-vector form: z_{k+1} = z_k @ M^T
        state = state @ M_torch.T

    traj = torch.stack(trajectory, dim=0)  # [T, B, 6]
    if traj.shape[1] == 1:
        traj = traj[:, 0, :]  # [T, 6]

    # Convert back to repo's common format [T, 3, 2] when possible.
    if traj.dim() == 2 and traj.shape[1] == 6:
        x = traj[:, 0:3]
        v = traj[:, 3:6]
        traj = torch.stack([x, v], dim=-1)  # [T, 3, 2]
        return traj
    if traj.dim() == 3 and traj.shape[2] == 6:
        x = traj[:, :, 0:3]
        v = traj[:, :, 3:6]
        traj = torch.stack([x, v], dim=-1)  # [T, B, 3, 2]
        return traj

    return traj


def generate_trajectory_forward_euler_sms(initial_state, force_func, dt, num_steps):
    """Generate trajectory using forward Euler method"""
    logger.debug(f"Generating SMS trajectory: Forward Euler, dt={dt}, steps={num_steps}")
    trajectory = []
    state = initial_state.clone()

    for step in range(num_steps):
        trajectory.append(state.clone())

        # Forward Euler: x_{n+1} = x_n + dt * f(x_n)
        acc = force_func(state, if_external_force=False, t=step*dt)
        velocity = state[:, 1:2]

        # Update position and velocity
        state = state + dt * torch.cat([velocity, acc], dim=1)

    return torch.stack(trajectory).squeeze(1)  # Shape: [num_steps, 2]


def generate_trajectory_midpoint_sms(initial_state, force_func, dt, num_steps):
    """Generate trajectory using midpoint method (leapfrog/Stormer-Verlet)"""
    logger.debug(f"Generating SMS trajectory: Midpoint, dt={dt}, steps={num_steps}")
    trajectory = []

    # Initialize position and velocity
    x = initial_state[:, 0:1].clone()
    v = initial_state[:, 1:2].clone()

    for step in range(num_steps):
        trajectory.append(torch.cat([x, v], dim=1).clone())

        # Leapfrog integration
        # v_{n+1/2} = v_n + dt/2 * a(x_n)
        acc = force_func(torch.cat([x, v], dim=1), if_external_force=False, t=step*dt)
        v_half = v + 0.5 * dt * acc

        # x_{n+1} = x_n + dt * v_{n+1/2}
        x = x + dt * v_half

        # v_{n+1} = v_{n+1/2} + dt/2 * a(x_{n+1})
        acc_next = force_func(torch.cat([x, v_half], dim=1), if_external_force=False, t=(step+1)*dt)
        v = v_half + 0.5 * dt * acc_next

    return torch.stack(trajectory).squeeze(1)  # Shape: [num_steps, 2]


def generate_trajectory_rk4_smsd(initial_state, force_func, dt, num_steps):
    """Generate trajectory for single mass spring damper using RK4"""
    logger.debug(f"Generating SMSD trajectory: RK4, dt={dt}, steps={num_steps}")
    trajectory = []
    state = initial_state.clone()

    for step in range(num_steps):
        trajectory.append(state.clone())

        # RK4 integration
        def dynamics(s):
            # s shape: [1, 2] where s[0,0] is position, s[0,1] is velocity
            acc = force_func(s)  # force_smsd doesn't take external force args
            # Ensure acc has the right shape for concatenation
            if acc.ndim == 1:
                acc = acc.unsqueeze(-1)  # Convert [1] to [1, 1]
            # For SMSD: dx/dt = v, dv/dt = a
            return torch.cat([s[:, 1:2], acc], dim=1)

        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)

        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    return torch.stack(trajectory).squeeze(1)  # Shape: [num_steps, 2]


# ---------------------------------------------------------------------------
# Slider-Crank Data Generator
# ---------------------------------------------------------------------------
def generate_slider_crank_dataset(total_num_steps, train_num_steps, dt, root_dir, seed=42):
    """
    Generate Slider-Crank mechanism dataset.

    Note: This function already saves CSV files directly, matching MNODE-code behavior.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Generating Slider-Crank dataset with {total_num_steps} total steps")

    # -------------------------------------------------------------------
    # System parameters (aligned with the 3-body description + slider_crank_bb.py)
    # -------------------------------------------------------------------
    # 3 bodies: crank (1), rod (2), slider (3)
    m1 = 1.0
    m2 = 1.0
    m3 = 1.0
    J1 = 0.10
    J2 = 0.10
    J3 = 0.10

    # Geometry NOTE:
    # The kinematics used below requires l_rod >= r_crank to keep the square-root real.
    # In slider_crank_bb.py, lengths are defined via half-lengths: L1 (crank), L2 (rod).
    # There, crank length is 2*L1 and rod length is 2*L2.
    L1 = 1.0  # crank half-length -> crank "radius" r_crank
    L2 = 2.0  # rod   half-length -> rod length is 2*L2
    r_crank = L1
    l_rod = 2.0 * L2

    # Slider spring to wall (linear spring, N/m) as per the requirement
    k_spring = 1.0

    # Non-inertial loads (taken from slider_crank_bb.py)
    # No external motor torque for this dataset (unforced system).
    # Keep this as a named function for clarity/possible future extension.
    def motor_torque(t):
        return 0.0

    # Rotational damper between crank and rod
    c12 = 0.05
    gamma = 1.0

    # Slider friction (Coulomb/viscous mix modeled as |v|^psi sgn(v) )
    c_slide = 0.20
    psi = 1.0

    # Initial conditions: crank at 45 degrees, zero velocity
    theta_0 = np.pi / 4  # 45 degrees
    omega_0 = 0.0

    def slider_position(theta):
        """Calculate slider position given crank angle."""
        return r_crank * np.cos(theta) + np.sqrt(l_rod**2 - r_crank**2 * np.sin(theta)**2)

    def dx_dtheta(theta):
        """Derivative of slider position with respect to crank angle."""
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        denominator = np.sqrt(l_rod**2 - r_crank**2 * sin_theta**2)
        return -r_crank * sin_theta - (r_crank**2 * sin_theta * cos_theta) / denominator

    def slider_velocity(theta, omega):
        """Calculate slider velocity given crank angle and angular velocity."""
        return dx_dtheta(theta) * omega

    def dynamics(t, state):
        """Slider-crank dynamics."""
        theta, omega = state

        # --- Kinematics: rod angle beta and its derivative ---
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        temp = (r_crank * sin_theta) / l_rod
        temp_clipped = np.clip(temp, -1.0, 1.0)
        beta = np.arcsin(temp_clipped)

        # d beta / d theta
        dbeta_dtheta = (r_crank * cos_theta / l_rod) / np.sqrt(1.0 - temp_clipped**2 + 1e-12)
        omega2 = dbeta_dtheta * omega

        # --- Slider position/velocity ---
        x = slider_position(theta)
        x_eq = slider_position(theta_0)  # reference at initial configuration
        x_dot = dx_dtheta(theta) * omega

        # --- Map forces/torques from slider_crank_bb.py style terms ---
        # Motor torque (disabled: unforced)
        tau_motor = motor_torque(t)

        # Rotational damper between crank and rod
        rel_omega = omega - omega2
        tau_damper = c12 * (np.abs(rel_omega) ** gamma) * np.sign(rel_omega)

        # Slider friction force (sign-power law)
        f_slide = c_slide * (np.abs(x_dot) ** psi) * np.sign(x_dot)
        tau_slide = f_slide * dx_dtheta(theta)

        # Slider spring force to wall
        f_spring = -k_spring * (x - x_eq)
        tau_spring = f_spring * dx_dtheta(theta)

        # --- Effective inertia about crank axis (1-DOF approximation) ---
        # - crank inertia J1 directly
        # - slider translating mass m3 via m3*(dx/dtheta)^2
        # - rod rotational inertia approximated via J2*(d beta / d theta)^2
        # (Rod translational kinetic energy is ignored in this 1-DOF model.)
        J_eff = J1 + (m3 * (dx_dtheta(theta) ** 2)) + (J2 * (dbeta_dtheta ** 2))

        # Net torque: motor - damper - friction + spring
        tau_net = tau_motor - tau_damper - tau_slide + tau_spring

        # Angular acceleration
        alpha = tau_net / (J_eff + 1e-12)

        return [omega, alpha]

    # Generate trajectory using scipy's RK45
    # NOTE: Use an explicit step of `dt` so downstream sees exactly dt=0.01 (no linspace rounding).
    t_span = [0, (total_num_steps - 1) * dt]
    t_eval = np.arange(total_num_steps, dtype=float) * dt
    initial_state = [theta_0, omega_0]

    logger.info(f"Integrating Slider-Crank dynamics with RK45...")
    solution = scipy.integrate.solve_ivp(
        dynamics, t_span, initial_state,
        method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    # Extract states
    theta_trajectory = solution.y[0]  # Crank angle
    omega_trajectory = solution.y[1]  # Crank angular velocity

    # Calculate derived quantities
    x_slider = np.array([slider_position(theta) for theta in theta_trajectory])
    v_slider = np.array([slider_velocity(theta, omega)
                        for theta, omega in zip(theta_trajectory, omega_trajectory)])

    # Calculate accelerations (finite differences)
    alpha_trajectory = np.zeros_like(omega_trajectory)
    alpha_trajectory[1:] = (omega_trajectory[1:] - omega_trajectory[:-1]) / dt
    alpha_trajectory[0] = alpha_trajectory[1]  # Duplicate first value

    # Wrap theta to [0, 2π]
    theta_0_2pi = theta_trajectory % (2 * np.pi)

    # Create full dataset with all quantities
    full_data = np.column_stack([
        theta_trajectory,      # Original theta (can exceed 2π)
        theta_0_2pi,           # Wrapped theta [0, 2π]
        omega_trajectory,      # Angular velocity
        alpha_trajectory,      # Angular acceleration
        x_slider,              # Slider position
        v_slider               # Slider velocity
    ])

    # Split into train and test
    train_data = full_data[:train_num_steps]
    test_data = full_data[train_num_steps:train_num_steps + (total_num_steps - train_num_steps)]

    # Time arrays
    t_train = t_eval[:train_num_steps]
    t_test = t_eval[train_num_steps:train_num_steps + (total_num_steps - train_num_steps)]

    # Save to CSV with headers
    dataset_dir = os.path.join(root_dir, 'dataset', 'Slider_Crank')
    os.makedirs(dataset_dir, exist_ok=True)

    # Save with headers for clarity
    train_df = pd.DataFrame(train_data, columns=[
        'theta', 'theta_0_2pi', 'omega', 'alpha', 'x_slider', 'v_slider'
    ])
    test_df = pd.DataFrame(test_data, columns=[
        'theta', 'theta_0_2pi', 'omega', 'alpha', 'x_slider', 'v_slider'
    ])

    train_df.to_csv(os.path.join(dataset_dir, 's_train.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_dir, 's_test.csv'), index=False)

    # Also save full trajectory
    full_df = pd.DataFrame(full_data, columns=[
        'theta', 'theta_0_2pi', 'omega', 'alpha', 'x_slider', 'v_slider'
    ])
    full_df.to_csv(os.path.join(dataset_dir, 's_full.csv'), index=False)

    # Save time arrays (with header for consistent downstream loading)
    pd.DataFrame({'time': t_train}).to_csv(os.path.join(dataset_dir, 't_train.csv'), index=False)
    pd.DataFrame({'time': t_test}).to_csv(os.path.join(dataset_dir, 't_test.csv'), index=False)
    pd.DataFrame({'time': t_eval}).to_csv(os.path.join(dataset_dir, 't_full.csv'), index=False)

    logger.info(f"Saved Slider-Crank dataset to {dataset_dir}")
    logger.info(f"  Train: {train_num_steps} steps, Test: {total_num_steps - train_num_steps} steps")

    # Create tensors for return (only theta_0_2pi and omega for FNODE)
    # Shape: [num_steps, 2]
    train_tensor = torch.tensor(train_data[:, [1, 2]], dtype=torch.float16)  # theta_0_2pi, omega
    test_tensor = torch.tensor(test_data[:, [1, 2]], dtype=torch.float16)
    full_tensor = torch.tensor(full_data[:, [1, 2]], dtype=torch.float16)

    time_tensor = torch.tensor(t_eval, dtype=torch.float16)

    logger.info(f"Slider-Crank tensors created - Train: {train_tensor.shape}, Test: {test_tensor.shape}")

    # Return full tensor for compatibility
    return full_tensor, time_tensor


# ---------------------------------------------------------------------------
# Updated main dataset generator with consistent saving
# ---------------------------------------------------------------------------


def generate_dataset(test_case, numerical_methods, dt, num_steps,
                     output_root_dir='.', seed=42, **kwargs):
    """
    Generates datasets for various test cases and saves them to disk with consistent organization.

    Args:
        test_case (str): Name of the system (e.g., 'Single_Mass_Spring', 'Cartpole').
        numerical_methods (str): Method for generating data ('rk4', 'fe', 'analytical', 'midpoint').
        dt (float): Time step for simulation.
        num_steps (int): Total number of time steps to generate.
        output_root_dir (str): Root directory for saving 'dataset' and 'figures'.
        seed (int): Random seed for reproducibility.
        **kwargs: Additional keyword arguments specific to test cases or saving options.

    Returns:
        torch.Tensor or None: The generated trajectory data as a tensor, or None if generation fails.
    """
    # Note: Do NOT call set_seed here - main script already called it
    # Calling it again resets random state and causes model init mismatch with MNODE
    # set_seed(seed)  # Commented out to match MNODE-code behavior
    logger.info(f"--- Generating Dataset ---")
    logger.info(f"Test Case: {test_case}")
    logger.info(f"Method: {numerical_methods}, dt: {dt}, Steps: {num_steps}")

    # Create standard dataset directory structure
    dataset_dir = os.path.join(output_root_dir, 'dataset', test_case)
    figures_dir = os.path.join(output_root_dir, 'figures', test_case)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Note: Do NOT reset seed here - main script already called set_seed()
    # Resetting only np.random.seed would desync torch random state
    # np.random.seed(seed)  # Commented out to match MNODE-code behavior

    # Initialize variables
    training_set = None
    initial_state_np = None
    force_func = None
    num_bodies = 1
    use_scipy = False

    # Resolve device once for this generation call.
    # Some branches may override it (e.g., GPU batch generation), but it must
    # always be defined before any torch.tensor(..., device=device) calls.
    device_arg = kwargs.get('device', None)
    if device_arg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_arg, torch.device):
        device = device_arg
    else:
        device = torch.device(str(device_arg))

    # =========================================================================
    # Define initial conditions and force functions based on test case
    # =========================================================================

    if test_case == "Single_Mass_Spring":
        logger.info("Setting up Single Mass Spring system")
        initial_state_np = np.array([[1.0, 0.0]])  # [[position, velocity]]
        force_func = force_sms
        num_bodies = 1

    elif test_case == "Single_Mass_Spring_Damper":
        logger.info("Setting up Single Mass Spring Damper system")
        initial_state_np = np.array([[1.0, 0.0]])  # [[position, velocity]]
        force_func = force_smsd
        num_bodies = 1

    elif test_case == "Triple_Mass_Spring_Damper":
        logger.info("Setting up Triple Mass Spring Damper system")
        initial_state_np = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])  # [[x1,v1], [x2,v2], [x3,v3]]
        force_func = force_tmsd
        num_bodies = 3

    elif test_case == "Single_Pendulum":
        logger.info("Setting up Single Pendulum system")
        initial_state_np = np.array([[np.pi / 2.0, 0.0]])  # [[theta, omega]]
        use_scipy = True

        def pendulum_dynamics(t, state):
            theta, omega = state.reshape(-1, 2).T
            g = 9.81
            L = 1.0
            alpha = -g/L * np.sin(theta)
            return np.column_stack([omega, alpha]).flatten()

        force_func = pendulum_dynamics
        num_bodies = 1

    elif test_case == "Double_Pendulum":
        logger.info("Setting up Double Pendulum system")
        initial_state_np = np.array([[np.pi / 2, 0.0], [np.pi / 2, 0.0]])  # [[theta1, omega1], [theta2, omega2]]
        force_func = force_dp
        num_bodies = 2

    elif test_case == "Cart_Pole":
        logger.info("Setting up Cart-Pole system")
        # Initial conditions matching original code: cart at x=1, pole at angle pi/6
        initial_state_np = np.array([[1.0, 0.0], [np.pi / 6, 0.0]])  # [[cart_pos, cart_vel], [pole_angle, pole_omega]]
        force_func = force_cp
        num_bodies = 2

    elif test_case == "Slider_Crank":
        logger.info("Setting up Slider Crank system")

        # Call specialized Slider_Crank generator with only supported parameters
        slider_crank_args = {
            'total_num_steps': num_steps,
            'train_num_steps': kwargs.get('gen_train_num_steps', int(0.75 * num_steps)),
            'dt': dt,
            'root_dir': output_root_dir,
            'seed': seed
        }

        logger.info(f"Calling generate_slider_crank_dataset with {num_steps} steps, dt={dt}")
        training_set, time_tensor = generate_slider_crank_dataset(**slider_crank_args)

        # Note: generate_slider_crank_dataset() already saves CSV files internally
        logger.info(f"Slider_Crank dataset generated (CSV files already saved by generate_slider_crank_dataset)")

        return training_set

    elif test_case == "veh_4dof":
        logger.info("Setting up 4DOF Vehicle system with control")

        # Import parameterized vehicle model
        from Model.veh_4dof.rom_vehicle_param import ParameterizedVehModel

        # Generate trajectory with correct control strategy
        initial_state = [0.0, 0.0, 0.0, 0.0]  # [x, y, theta, v]
        duration = num_steps * dt

        logger.info(f"Generating vehicle trajectory: duration={duration}s, dt={dt}s, steps={num_steps}")

        # Initialize vehicle with pychrono system
        try:
            import pychrono as chrono
            system = chrono.ChSystemNSC()
            system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        except ImportError:
            system = None  # Will work without visualization

        # Use default parameters (no params dict means using defaults)
        vehicle = ParameterizedVehModel(system, initial_state, [0.0, 0.0], dt, Visualize=False)

        # Initialize storage arrays
        time_array = np.zeros(num_steps)
        states = np.zeros((num_steps, 4))  # [x, y, theta, v]
        accelerations = np.zeros(num_steps)
        controls = np.zeros((num_steps, 2))  # [alpha, beta]

        # Control generation function (from original test_vehicle_trajectory_control.py)
        def generate_control_inputs(t, v):
            accel_time = 5.0

            if t <= accel_time:
                # Phase 1: Straight-line acceleration
                ramp_time = 4.0
                s = np.clip(t / ramp_time, 0.0, 1.0)
                alpha = 0.25 * (1.0 - np.cos(np.pi * s))  # Smooth ramp to 0.5
                beta = 0.0  # No steering
            else:
                # Phase 2: S-curve with deceleration
                t_s = t - accel_time

                # Throttle: exponential decay
                alpha = 0.5 * np.exp(-t_s / 10.0)

                # Steering: sinusoidal S-curve with 0.3 Hz frequency (important!)
                amplitude = 0.4
                frequency = 0.3  # This creates multiple oscillations
                beta = amplitude * np.sin(2 * np.pi * frequency * t_s)

            # Apply saturation
            alpha = np.clip(alpha, 0.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

            return alpha, beta

        # Generate trajectory
        print(f"Generating {duration}s trajectory with dt={dt}s ({num_steps} steps)...")
        print("Phase 1 (0-5s): Straight acceleration")
        print("Phase 2 (5s+): S-curve with 0.3Hz steering oscillation")

        for i in range(num_steps):
            t = i * dt
            time_array[i] = t

            # Get current state
            current_state = vehicle.get_state()  # [x, y, theta, v, accel]
            states[i, :] = current_state[:4]
            accelerations[i] = current_state[4]

            # Generate control inputs
            alpha, beta = generate_control_inputs(t, current_state[3])
            controls[i] = [alpha, beta]

            # Update vehicle
            vehicle.update([alpha, beta])

            # Progress indicator
            if (i+1) % 1000 == 0:
                print(f"  Step {i+1}/{num_steps}: t={t:.2f}s, x={current_state[0]:.2f}m, "
                      f"v={current_state[3]:.2f}m/s, α={alpha:.3f}, β={beta:+.3f}")

        print(f"Trajectory generation complete!")
        print(f"Final position: x={states[-1, 0]:.2f}m, y={states[-1, 1]:.2f}m")
        print(f"Max velocity: {np.max(states[:, 3]):.2f}m/s")

        # Count steering oscillations
        beta_sign_changes = np.sum(np.abs(np.diff(np.sign(controls[controls[:, 1] != 0, 1]))) > 0)
        print(f"Steering oscillations: {beta_sign_changes} direction changes")

        # states shape: (num_steps, 4) with columns [x, y, theta, v]
        # controls shape: (num_steps, 2) with columns [alpha, beta]
        # Reorder to [x, y, v, theta] to match the requirement
        states_reordered = states[:, [0, 1, 3, 2]]  # [x, y, v, theta]

        # Combine states and controls into 6D data
        full_data = np.concatenate([states_reordered, controls], axis=1)  # [x, y, v, theta, alpha, beta]

        logger.info(f"Generated data shape: {full_data.shape} (expecting {num_steps} x 6)")

        # Split into train and test
        train_steps = kwargs.get('gen_train_num_steps', int(0.75 * num_steps))
        test_steps = kwargs.get('gen_test_num_steps', num_steps - train_steps)

        s_train = full_data[:train_steps]
        s_test = full_data[train_steps:train_steps + test_steps]

        # Time arrays
        t_train = time_array[:train_steps]
        t_test = time_array[train_steps:train_steps + test_steps]

        # Save to CSV files
        dataset_dir = os.path.join(output_root_dir, 'dataset', test_case)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save with headers for clarity
        train_df = pd.DataFrame(s_train, columns=['x', 'y', 'v', 'theta', 'alpha', 'beta'])
        test_df = pd.DataFrame(s_test, columns=['x', 'y', 'v', 'theta', 'alpha', 'beta'])

        train_df.to_csv(os.path.join(dataset_dir, 's_train.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 's_test.csv'), index=False)

        # Also save time arrays
        pd.DataFrame({'time': t_train}).to_csv(os.path.join(dataset_dir, 't_train.csv'), index=False)
        pd.DataFrame({'time': t_test}).to_csv(os.path.join(dataset_dir, 't_test.csv'), index=False)

        logger.info(f"Saved veh_4dof dataset to {dataset_dir}")
        logger.info(f"  Train: {train_steps} steps, Test: {test_steps} steps")

        # Also save the acceleration data for potential use in training targets
        accel_train = accelerations[:train_steps]
        accel_test = accelerations[train_steps:train_steps + test_steps]
        np.save(os.path.join(dataset_dir, 'accelerations_train.npy'), accel_train)
        np.save(os.path.join(dataset_dir, 'accelerations_test.npy'), accel_test)

        # Convert to torch tensor and return
        training_set = torch.tensor(full_data, dtype=torch.float16, device=device)
        return training_set

    elif test_case == "veh_4dof_param":
        logger.info("Setting up 4DOF Vehicle with parameterized data generation")

        # Import required modules
        from Model.veh_4dof.parameter_sampler import VehicleParameterSampler
        from Model.veh_4dof.rom_vehicle_param import (
            ParameterizedVehModel,
            GPUBatchVehicleModel,
            verify_gpu_cpu_consistency
        )
        import h5py
        from tqdm import tqdm

        # Check for GPU availability
        # NOTE: Do not `import torch` here; doing so makes `torch` a function-local
        # name and can break other branches that use the module.
        try:
            use_gpu = torch.cuda.is_available() and kwargs.get('use_gpu', True)
        except Exception:
            use_gpu = False
            logger.warning("PyTorch not available, using CPU-only generation")

        # Generate parameter combinations (4^9 = 262,144)
        sampler = VehicleParameterSampler()
        all_params = sampler.generate_all_combinations()
        n_params = len(all_params)
        logger.info(f"Generated {n_params:,} parameter combinations (4^9)")

        # Parameters
        train_ratio = kwargs.get('gen_train_num_steps', 3000) / num_steps
        train_steps = int(num_steps * train_ratio)
        test_steps = num_steps - train_steps

        # Control generation function (same as original veh_4dof)
        def generate_control_inputs(t, v):
            accel_time = 5.0

            if t <= accel_time:
                # Phase 1: Straight-line acceleration
                ramp_time = 4.0
                s = np.clip(t / ramp_time, 0.0, 1.0)
                alpha = 0.25 * (1.0 - np.cos(np.pi * s))
                beta = 0.0
            else:
                # Phase 2: S-curve with deceleration
                t_s = t - accel_time
                alpha = 0.5 * np.exp(-t_s / 10.0)
                amplitude = 0.4
                frequency = 0.3
                beta = amplitude * np.sin(2 * np.pi * frequency * t_s)

            alpha = np.clip(alpha, 0.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

            return alpha, beta

        # Create output directory
        dataset_dir = os.path.join(output_root_dir, 'dataset', test_case)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save parameter information
        sampler.save_parameter_info(dataset_dir)

        # Create HDF5 file for storage
        h5_path = os.path.join(dataset_dir, 'vehicle_data.h5')

        with h5py.File(h5_path, 'w') as h5f:
            # Create datasets
            train_data = h5f.create_dataset(
                'train_data',
                shape=(n_params * train_steps, 15),
                dtype='float16'
            )
            train_accel = h5f.create_dataset(
                'train_accel',
                shape=(n_params * train_steps,),
                dtype='float16'
            )

            test_data = h5f.create_dataset(
                'test_data',
                shape=(n_params * test_steps, 15),
                dtype='float16'
            )
            test_accel = h5f.create_dataset(
                'test_accel',
                shape=(n_params * test_steps,),
                dtype='float16'
            )

            # Time arrays
            time_train = np.arange(train_steps) * dt
            time_test = np.arange(test_steps) * dt
            h5f.create_dataset('time_train', data=time_train)
            h5f.create_dataset('time_test', data=time_test)

            # Store parameter combinations
            h5f.create_dataset('parameter_combinations', data=all_params)

            # Add metadata attributes
            h5f.attrs['n_parameters'] = n_params
            h5f.attrs['train_steps'] = train_steps
            h5f.attrs['test_steps'] = test_steps
            h5f.attrs['total_steps'] = num_steps
            h5f.attrs['dt'] = dt
            # NOTE: This dataset is intentionally stored in PHYSICAL UNITS (no normalization).
            # main_fnode_veh_param.py will train & test directly in physical space.
            h5f.attrs['normalized'] = False

            if use_gpu:
                logger.info("Using GPU acceleration for data generation")

                # GPU batch processing
                # Adjust batch_size based on your GPU memory:
                # - RTX 3090 (24GB): 10000-20000
                # - RTX 3080 (10GB): 5000-10000
                # - RTX 3070 (8GB): 3000-5000
                # Can be overridden by kwargs
                default_batch_size = 10000
                batch_size = min(kwargs.get('gpu_batch_size', default_batch_size), n_params)
                device = torch.device('cuda')

                logger.info(f"GPU batch size: {batch_size}")

                for batch_start in tqdm(range(0, n_params, batch_size), desc="GPU Batches"):
                    batch_end = min(batch_start + batch_size, n_params)
                    batch_params = all_params[batch_start:batch_end]

                    # Create GPU batch model
                    gpu_model = GPUBatchVehicleModel(batch_params, dt, device)

                    # Simulate batch - use 'gpu' for fully GPU-based control generation
                    trajectories, accelerations = gpu_model.simulate_batch(
                        num_steps, 'gpu'  # Use GPU-optimized control generation
                    )

                    for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                        traj = trajectories[local_idx]
                        accel = accelerations[local_idx]
                        param_vec = batch_params[local_idx]

                        # Store PHYSICAL data before writing to HDF5.
                        # traj shape: (num_steps, 6) = [x, y, v, theta, alpha, beta]
                        # Expand parameters to match trajectory length (still physical units)
                        param_expanded = np.tile(param_vec.reshape(1, -1), (num_steps, 1))
                        full_data = np.concatenate([traj, param_expanded], axis=1)

                        # Store normalized data in HDF5
                        train_idx_start = global_idx * train_steps
                        train_idx_end = (global_idx + 1) * train_steps
                        test_idx_start = global_idx * test_steps
                        test_idx_end = (global_idx + 1) * test_steps

                        train_data[train_idx_start:train_idx_end] = full_data[:train_steps]
                        train_accel[train_idx_start:train_idx_end] = accel[:train_steps]

                        test_data[test_idx_start:test_idx_end] = full_data[train_steps:num_steps]
                        test_accel[test_idx_start:test_idx_end] = accel[train_steps:num_steps]

            else:
                logger.info("Using CPU multiprocessing for data generation")

                # CPU parallel processing using multiprocessing
                from multiprocessing import Pool, cpu_count

                def generate_single_trajectory(params_with_idx):
                    idx, param_vec = params_with_idx
                    param_dict = sampler.get_parameter_dict(param_vec)

                    # Create PyChrono system if available
                    try:
                        import pychrono as chrono
                        system = chrono.ChSystemNSC()
                        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
                    except ImportError:
                        system = None

                    # Create vehicle with parameters
                    vehicle = ParameterizedVehModel(
                        system, [0, 0, 0, 0], [0, 0], dt,
                        Visualize=False, params=param_dict
                    )

                    # Generate trajectory
                    trajectory = np.zeros((num_steps, 6))
                    accelerations_local = np.zeros(num_steps)

                    for step in range(num_steps):
                        t = step * dt
                        state = vehicle.get_state()

                        # Store state [x, y, v, theta] and controls
                        trajectory[step, :4] = [state[0], state[1], state[3], state[2]]
                        accelerations_local[step] = state[4]

                        # Generate and apply control
                        alpha, beta = generate_control_inputs(t, state[3])
                        trajectory[step, 4:6] = [alpha, beta]
                        vehicle.update([alpha, beta])

                    return idx, trajectory, accelerations_local

                # Process in parallel
                with Pool(cpu_count()) as pool:
                    params_with_idx = list(enumerate(all_params))

                    for idx, traj, accel in tqdm(
                        pool.imap(generate_single_trajectory, params_with_idx),
                        total=n_params,
                        desc="CPU Parallel"
                    ):
                        param_vec = all_params[idx]

                        # Store PHYSICAL data (no normalization):
                        # traj: (num_steps, 6) = [x, y, v, theta, alpha, beta]
                        # params: (9,) expanded to (num_steps, 9)
                        param_expanded = np.tile(param_vec.reshape(1, -1), (num_steps, 1))
                        full_data = np.concatenate([traj, param_expanded], axis=1)

                        # Store physical data in HDF5
                        train_idx_start = idx * train_steps
                        train_idx_end = (idx + 1) * train_steps
                        test_idx_start = idx * test_steps
                        test_idx_end = (idx + 1) * test_steps

                        train_data[train_idx_start:train_idx_end] = full_data[:train_steps]
                        train_accel[train_idx_start:train_idx_end] = accel[:train_steps]

                        test_data[test_idx_start:test_idx_end] = full_data[train_steps:num_steps]
                        test_accel[test_idx_start:test_idx_end] = accel[train_steps:num_steps]

            # Add metadata
            h5f.attrs['n_parameters'] = n_params
            h5f.attrs['train_steps'] = train_steps
            h5f.attrs['test_steps'] = test_steps
            h5f.attrs['dt'] = dt
            h5f.attrs['total_train_samples'] = n_params * train_steps
            h5f.attrs['total_test_samples'] = n_params * test_steps
            h5f.attrs['generation_method'] = 'GPU' if use_gpu else 'CPU'

        logger.info(f"Saved parameterized vehicle data to {h5_path}")
        logger.info(f"Total training samples: {n_params * train_steps:,}")
        logger.info(f"Total test samples: {n_params * test_steps:,}")

        # Verify GPU/CPU consistency if requested
        if kwargs.get('verify_consistency', False) and use_gpu:
            logger.info("Verifying GPU/CPU consistency...")
            test_params = sampler.get_parameter_dict(all_params[0])
            verify_gpu_cpu_consistency(test_params, dt, 100)

        # Return path to the HDF5 file instead of tensor
        return h5_path

    elif test_case == "veh_11dof":
        logger.info("Setting up 11DOF Vehicle system with GPU acceleration")

        # Get generation mode from kwargs
        generate_mode = kwargs.get('generate_mode', 'trajectory')  # 'trajectory' or 'random_points'
        logger.info(f"Data generation mode: {generate_mode}")

        # Import the GPU simulator wrapper from Model/veh_11dof
        import sys
        model_dir = os.path.dirname(os.path.abspath(__file__))
        veh_11dof_dir = os.path.join(model_dir, 'veh_11dof')
        if veh_11dof_dir not in sys.path:
            sys.path.insert(0, veh_11dof_dir)
        from dof11_gpu_wrapper import (
            DOF11GPUSimulator,
            VehicleState,
            TireState,
            DriverInput,
            TireType,
            create_driver_inputs_from_arrays
        )
        from scipy.ndimage import gaussian_filter1d

        # Setup simulator - use the compiled library in veh_11dof directory
        lib_path = os.path.join(veh_11dof_dir, 'libdof11_gpu_python.so')

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"GPU library not found at {lib_path}")

        simulator = DOF11GPUSimulator(lib_path=lib_path)

        # Initialize with vehicle parameters (HMMWV)
        vehicle_params = os.path.join(veh_11dof_dir, 'data', 'json', 'HMMWV', 'vehicle.json')
        tire_params = os.path.join(veh_11dof_dir, 'data', 'json', 'HMMWV', 'tmeasy.json')

        # Initialize with enough vehicles for batch processing
        if generate_mode == 'random_points':
            # For random points, we need more vehicles for batch processing
            max_batch_size = min(1024, num_steps)
        else:
            # For trajectory, we process sequentially but still need batch for derivative computation
            max_batch_size = min(1024, num_steps)

        simulator.initialize(
            vehicle_params,
            tire_params,
            num_vehicles=max_batch_size,
            tire_type=TireType.TMEASY
        )

        # Generate control trajectory from run_s_curve_simulation.py
        # Keep exactly `num_steps` samples with step `dt`.
        total_time = (num_steps - 1) * dt
        t_array = np.arange(num_steps, dtype=float) * dt

        # Initialize control arrays
        steering = np.zeros_like(t_array)
        throttle = np.zeros_like(t_array)
        braking = np.zeros_like(t_array)

        # New 6-Phase Control Strategy (matching user requirements)
        # Phase 1: 0-5s Initial acceleration
        # Phase 2: 5-20s S-curve (serpentine) trajectory
        # Phase 3: 20-25s First braking
        # Phase 4: 25-30s Re-acceleration
        # Phase 5: 30-45s Circular trajectory
        # Phase 6: 45-50s Final braking to stop

        # THROTTLE PROFILE - Simple 3-phase for S-curve (reference implementation)
        # Can extend for circular path if needed
        for i, time_val in enumerate(t_array):
            if total_time <= 20.0:  # S-curve only
                # Phase 1: Acceleration (0-3s)
                if time_val < 3.0:
                    progress = time_val / 3.0
                    # Smooth ramp to 0.4
                    throttle[i] = 0.4 * np.sin(np.pi/2 * progress)**2

                # Phase 2: Maintain speed during S-curve (3-17s)
                elif time_val < 17.0:
                    # Cruise at moderate throttle
                    throttle[i] = 0.3

                # Phase 3: Deceleration (17-20s)
                elif time_val < 20.0:
                    progress = (time_val - 17.0) / 3.0
                    # Smooth reduction to 0
                    throttle[i] = 0.3 * (1 - np.sin(np.pi/2 * progress)**2)
                else:
                    throttle[i] = 0.0

            else:  # Extended trajectory with circle
                # Phase 1: Initial acceleration (0-5s)
                if time_val < 5.0:
                    progress = time_val / 5.0
                    throttle[i] = 0.4 * np.sin(np.pi/2 * progress)**2

                # Phase 2: S-curve (5-20s)
                elif time_val < 20.0:
                    throttle[i] = 0.35

                # Phase 3: Transition (20-25s)
                elif time_val < 25.0:
                    throttle[i] = 0.3

                # Phase 4: Re-acceleration for circle (25-30s)
                elif time_val < 30.0:
                    progress = (time_val - 25.0) / 5.0
                    throttle[i] = 0.3 + 0.1 * np.sin(np.pi/2 * progress)**2

                # Phase 5: Circle (30-45s)
                elif time_val < 45.0:
                    throttle[i] = 0.35

                # Phase 6: Final stop (45-50s)
                elif time_val < 50.0:
                    progress = (time_val - 45.0) / 5.0
                    throttle[i] = 0.35 * (1 - np.sin(np.pi/2 * progress)**2)
                else:
                    throttle[i] = 0.0

        # STEERING PROFILE (updated for new timeline)
        for i, time_val in enumerate(t_array):
            # Phase 1: Straight during acceleration (0-5s)
            if time_val < 5.0:
                steering[i] = 0.0

            # Phase 2: S-curve segment (5-20s)
            elif 5.0 <= time_val < 20.0:
                t_norm = (time_val - 5.0) / 15.0  # Normalize to [0, 1]
                # Create serpentine motion with 2 complete S-curves
                steering[i] = 0.25 * np.sin(4 * np.pi * t_norm)

                # Smooth envelope for entry and exit
                if t_norm < 0.1:
                    entry_factor = t_norm / 0.1
                    steering[i] *= (3*entry_factor**2 - 2*entry_factor**3)
                elif t_norm > 0.9:
                    exit_factor = (1.0 - t_norm) / 0.1
                    steering[i] *= (3*exit_factor**2 - 2*exit_factor**3)

            # Phase 3-4: Transition to circle during braking and re-acceleration (20-30s)
            elif 20.0 <= time_val < 30.0:
                progress = (time_val - 20.0) / 10.0
                # Gradual transition to circular steering
                steering[i] = 0.4 * (3*progress**2 - 2*progress**3)

            # Phase 5: Circular path (30-45s)
            elif 30.0 <= time_val < 45.0:
                steering[i] = 0.4  # Constant steering for circle

            # Phase 6: Exit circle during final braking (45-50s)
            elif 45.0 <= time_val < 50.0:
                progress = (time_val - 45.0) / 5.0
                steering[i] = 0.4 * (1 - (3*progress**2 - 2*progress**3))

        # BRAKING PROFILE (updated for new timeline with re-acceleration gap)
        for i, time_val in enumerate(t_array):
            # Phase 3: First braking phase (20-25s)
            if 20.0 <= time_val < 22.0:
                progress = (time_val - 20.0) / 2.0
                # Smooth application of brakes
                braking[i] = 0.15 * (3*progress**2 - 2*progress**3)
            elif 22.0 <= time_val < 25.0:
                progress = (time_val - 22.0) / 3.0
                # Gradual release of brakes
                braking[i] = 0.15 * (1 - (3*progress**2 - 2*progress**3))

            # Phase 4: NO braking during re-acceleration (25-30s)
            elif 25.0 <= time_val < 30.0:
                braking[i] = 0.0

            # Phase 5: NO braking during circle (30-45s)
            elif 30.0 <= time_val < 45.0:
                braking[i] = 0.0

            # Phase 6: Final braking phase to stop (45-50s)
            elif 45.0 <= time_val < 47.0:
                progress = (time_val - 45.0) / 2.0
                # Gradual application of brakes
                braking[i] = 0.2 * (3*progress**2 - 2*progress**3)
            elif 47.0 <= time_val <= 50.0:
                progress = (time_val - 47.0) / 3.0
                # Increase braking to full stop
                braking[i] = 0.2 + 0.05 * (3*progress**2 - 2*progress**3)
            else:
                braking[i] = 0.0

        # Apply moderate smoothing for smooth transitions
        # First pass - medium smoothing
        steering = gaussian_filter1d(steering, sigma=10.0)
        throttle = gaussian_filter1d(throttle, sigma=12.0)  # Slightly smoother throttle
        braking = gaussian_filter1d(braking, sigma=10.0)

        # Second pass - light smoothing
        steering = gaussian_filter1d(steering, sigma=5.0)
        throttle = gaussian_filter1d(throttle, sigma=5.0)
        braking = gaussian_filter1d(braking, sigma=5.0)

        # Third pass - very light smoothing
        steering = gaussian_filter1d(steering, sigma=4.0)
        throttle = gaussian_filter1d(throttle, sigma=4.0)
        braking = gaussian_filter1d(braking, sigma=4.0)

        # Clip to valid ranges
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        braking = np.clip(braking, 0.0, 1.0)

        # Branch based on generation mode
        if generate_mode == 'random_points':
            # ========== MODE 1: Generate Random Training Points ==========
            logger.info(f"Generating {num_steps} random training points...")

            # Define sampling ranges for states and controls
            state_ranges = {
                'x': (0.0, 0.0),  # We don't vary position for training
                'y': (0.0, 0.0),
                'u': (0.0, 30.0),  # Longitudinal velocity [m/s]
                'v': (-5.0, 5.0),  # Lateral velocity [m/s]
                'psi': (-np.pi, np.pi),  # Yaw angle [rad]
                'wz': (-2.0, 2.0),  # Yaw rate [rad/s]
                'w_crank': (100.0, 600.0),  # Engine speed [rad/s]
                'wf': (0.0, 100.0),  # Front wheel speed [rad/s]
                'wr': (0.0, 100.0),  # Rear wheel speed [rad/s]
                'xe': (-0.1, 0.1),  # Front tire deflection [m]
                'ye': (-0.1, 0.1),  # Front tire deflection [m]
            }

            control_ranges = {
                'steering': (-0.5, 0.5),  # Steering angle [rad]
                'throttle': (0.0, 1.0),  # Throttle [0-1]
                'braking': (0.0, 0.5),  # Braking limited for stability [0-0.5]
            }

            # Generate random states
            states = np.zeros((num_steps, 11))
            for i, (key, (low, high)) in enumerate(state_ranges.items()):
                if key in ['x', 'y']:
                    states[:, i] = 0.0  # Keep position at origin
                else:
                    states[:, i] = np.random.uniform(low, high, num_steps)

            # Generate random controls
            controls = np.zeros((num_steps, 3))
            controls[:, 0] = np.random.uniform(control_ranges['steering'][0],
                                              control_ranges['steering'][1], num_steps)
            controls[:, 1] = np.random.uniform(control_ranges['throttle'][0],
                                              control_ranges['throttle'][1], num_steps)
            controls[:, 2] = np.random.uniform(control_ranges['braking'][0],
                                              control_ranges['braking'][1], num_steps)

            # Store states for later use
            x = states[:, 0]
            y = states[:, 1]
            u = states[:, 2]
            v = states[:, 3]
            psi = states[:, 4]
            wz = states[:, 5]
            w_crank = states[:, 6]
            wf = states[:, 7]
            wr = states[:, 8]
            xe = states[:, 9]
            ye = states[:, 10]

            steering = controls[:, 0]
            throttle = controls[:, 1]
            braking = controls[:, 2]

        else:
            # ========== MODE 2: Generate Trajectory via Simulation ==========
            logger.info(f"Generating trajectory via simulation for {total_time}s...")

            # Create driver inputs
            driver_inputs = create_driver_inputs_from_arrays(t_array, steering, throttle, braking)
            simulator.set_driver_inputs(driver_inputs)

            # Set initial state (reduced to avoid excessive speeds)
            initial_vehicle = VehicleState(
                x=0.0, y=0.0,
                u=0.1, v=0.0,  # Reduced initial forward velocity
                psi=0.0, wz=0.0,
                crank_omega=50.0  # Reduced crankshaft speed to avoid huge accelerations
            )
            # Match wheel angular velocity to vehicle speed (omega = v/r, assuming r=0.3m)
            initial_front_tire = TireState(omega=0.1/0.3, xe=0.0, ye=0.0)
            initial_rear_tire = TireState(omega=0.1/0.3, xe=0.0, ye=0.0)

            simulator.set_initial_state(
                initial_vehicle,
                initial_front_tire,
                initial_rear_tire,
                vehicle_index=0
            )

            # Run simulation
            logger.info(f"Running 11DOF simulation for {total_time}s with dt={dt}")
            results = simulator.simulate(
                end_time=total_time,
                dt=dt,
                output_freq=1  # Save every step
            )

            # Extract ALL 11 states for trajectory mode
            x = results['x'][:, 0][:num_steps]
            y = results['y'][:, 0][:num_steps]
            u = results['u'][:, 0][:num_steps]  # longitudinal velocity
            v = results['v'][:, 0][:num_steps]  # lateral velocity
            psi = results['psi'][:, 0][:num_steps]  # yaw angle
            wz = results['wz'][:, 0][:num_steps]  # yaw rate
            w_crank = results['w_engine'][:, 0][:num_steps] if 'w_engine' in results else results['crank_omega'][:, 0][:num_steps] if 'crank_omega' in results else np.zeros(num_steps)
            wf = results['wf'][:, 0][:num_steps] if 'wf' in results else results['omega_f'][:, 0][:num_steps] if 'omega_f' in results else np.zeros(num_steps)
            wr = results['wr'][:, 0][:num_steps] if 'wr' in results else results['omega_r'][:, 0][:num_steps] if 'omega_r' in results else np.zeros(num_steps)
            xe = results['xe'][:, 0][:num_steps] if 'xe' in results else results['xe_f'][:, 0][:num_steps] if 'xe_f' in results else np.zeros(num_steps)
            ye = results['ye'][:, 0][:num_steps] if 'ye' in results else results['ye_f'][:, 0][:num_steps] if 'ye_f' in results else np.zeros(num_steps)

        # Full 11-dimensional state vector
        states_full = np.column_stack([x, y, u, v, psi, wz, w_crank, wf, wr, xe, ye])

        # 3-dimensional control vector [steering, throttle, brake]
        controls = np.column_stack([steering[:num_steps], throttle[:num_steps], braking[:num_steps]])

        # Combine states and controls for full 14-dimensional input
        full_data = np.concatenate([states_full, controls], axis=1)

        # Calculate all 8 derivatives using EXACT physics kernel
        # This is CRITICAL - use the same kernel as simulation to avoid train/test mismatch
        logger.info("Computing exact derivatives using GPU physics kernel...")

        # Initialize arrays for derivatives
        accelerations = np.zeros((num_steps, 8))

        # Process in batches for efficiency
        batch_size = min(1024, num_steps)  # Process up to 1024 at a time

        for i in range(0, num_steps, batch_size):
            end_idx = min(i + batch_size, num_steps)
            batch_len = end_idx - i

            # Set states for this batch
            for j in range(batch_len):
                idx = i + j
                vehicle_state = VehicleState(
                    x=x[idx], y=y[idx],
                    u=u[idx], v=v[idx],
                    psi=psi[idx], wz=wz[idx],
                    crank_omega=w_crank[idx]
                )
                front_tire = TireState(omega=wf[idx], xe=xe[idx], ye=ye[idx])
                # Note: Rear tire relaxation set to 0 as per reference
                rear_tire = TireState(omega=wr[idx], xe=0.0, ye=0.0)

                simulator.set_initial_state(
                    vehicle_state, front_tire, rear_tire,
                    vehicle_index=j
                )

            # Compute derivatives for this batch
            batch_steering = steering[i:end_idx]
            batch_throttle = throttle[i:end_idx]
            batch_braking = braking[i:end_idx]

            # Get exact derivatives from physics kernel
            rhs = simulator.compute_derivatives_exact(
                batch_steering, batch_throttle, batch_braking,
                num_vehicles=batch_len
            )

            # Extract the 8 derivatives we need to learn
            # Note: We don't learn x_dot, y_dot, psi_dot (kinematic)
            # and we don't use xe_r_dot, ye_r_dot (rear tire simplified)
            accelerations[i:end_idx, 0] = rhs[:batch_len, 0]  # u_dot
            accelerations[i:end_idx, 1] = rhs[:batch_len, 1]  # v_dot
            accelerations[i:end_idx, 2] = rhs[:batch_len, 2]  # wz_dot
            accelerations[i:end_idx, 3] = rhs[:batch_len, 3]  # w_crank_dot
            accelerations[i:end_idx, 4] = rhs[:batch_len, 4]  # wf_dot
            accelerations[i:end_idx, 5] = rhs[:batch_len, 5]  # wr_dot
            accelerations[i:end_idx, 6] = rhs[:batch_len, 6]  # xe_dot
            accelerations[i:end_idx, 7] = rhs[:batch_len, 7]  # ye_dot

        logger.info(f"Computed exact derivatives for {num_steps} states")

        # Split data with 1:1 train:test ratio (50/50)
        train_ratio = kwargs.get('train_ratio', 0.5)  # Default to 0.5 for 1:1
        train_steps = int(num_steps * train_ratio)
        test_steps = num_steps - train_steps

        logger.info(f"Splitting data: {train_steps} train, {test_steps} test (ratio {train_ratio:.2f})")

        # Save to CSV
        dataset_dir = os.path.join(output_root_dir, 'dataset', test_case)
        os.makedirs(dataset_dir, exist_ok=True)

        s_train = full_data[:train_steps]
        s_test = full_data[train_steps:train_steps + test_steps]

        # Save with full 11DOF state + 3 controls = 14 columns
        column_names = ['x', 'y', 'u', 'v', 'psi', 'wz', 'w_crank', 'wf', 'wr', 'xe', 'ye',
                       'steering', 'throttle', 'brake']
        pd.DataFrame(s_train, columns=column_names).to_csv(
            os.path.join(dataset_dir, 's_train.csv'), index=False
        )
        pd.DataFrame(s_test, columns=column_names).to_csv(
            os.path.join(dataset_dir, 's_test.csv'), index=False
        )

        # Save time arrays
        t_train = t_array[:train_steps]
        t_test = t_array[train_steps:train_steps + test_steps]
        pd.DataFrame({'time': t_train}).to_csv(
            os.path.join(dataset_dir, 't_train.csv'), index=False
        )
        pd.DataFrame({'time': t_test}).to_csv(
            os.path.join(dataset_dir, 't_test.csv'), index=False
        )

        # Save accelerations for target computation
        accel_train = accelerations[:train_steps]
        accel_test = accelerations[train_steps:train_steps + test_steps]
        np.save(os.path.join(dataset_dir, 'accelerations_train.npy'), accel_train)
        np.save(os.path.join(dataset_dir, 'accelerations_test.npy'), accel_test)

        # Convert to torch tensor and return
        training_set = torch.tensor(full_data, dtype=torch.float16, device=device)
        return training_set

    else:
        raise ValueError(f"Test case '{test_case}' not recognized.")

    # =========================================================================
    # Generate trajectory using selected method
    # =========================================================================

    # Convert initial state to tensor
    initial_state = torch.tensor(initial_state_np, dtype=torch.float32, device=device)

    if use_scipy and test_case == "Single_Pendulum":
        # Use scipy for Single Pendulum
        logger.info(f"Using scipy.integrate for {test_case}")

        # Use an explicit step grid so time is fully determined by (dt, num_steps).
        t_eval = np.arange(num_steps, dtype=float) * dt
        t_span = [0, (num_steps - 1) * dt]
        initial_state_flat = initial_state_np.flatten()

        solution = scipy.integrate.solve_ivp(
            force_func, t_span, initial_state_flat,
            method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
        )

        if solution.success:
            trajectory_np = solution.y.T.reshape(num_steps, num_bodies, 2)
            training_set = torch.tensor(trajectory_np, dtype=torch.float32, device=device)
            logger.info(f"Generated trajectory shape: {training_set.shape}")
        else:
            logger.error(f"Integration failed: {solution.message}")
            return None

    elif test_case in ["Double_Pendulum", "Cart_Pole"]:
        if numerical_methods == "rk45":
            # Use scipy's RK45 for multi-body systems
            logger.info(f"Using scipy RK45 integration for {test_case}")

            # Define dynamics function for scipy
            def scipy_dynamics(t, state_flat):
                # Reshape flat state to multi-body format
                state = state_flat.reshape(num_bodies, 2)
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get accelerations from force function
                acc = force_func(state_tensor)
                if isinstance(acc, torch.Tensor):
                    acc = acc.numpy()

                # Construct state derivative
                state_dot = np.zeros_like(state)
                state_dot[:, 0] = state[:, 1]  # Position derivatives are velocities
                state_dot[:, 1] = acc  # Velocity derivatives are accelerations

                return state_dot.flatten()

            # Set up integration
            t_eval = np.arange(num_steps, dtype=float) * dt
            t_span = [0, (num_steps - 1) * dt]
            initial_state_flat = initial_state_np.flatten()

            # Integrate using RK45
            solution = scipy.integrate.solve_ivp(
                scipy_dynamics, t_span, initial_state_flat,
                method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
            )

            if solution.success:
                trajectory_np = solution.y.T.reshape(num_steps, num_bodies, 2)
                training_set = torch.tensor(trajectory_np, dtype=torch.float32, device=device)
                logger.info(f"Generated trajectory shape: {training_set.shape}")
            else:
                logger.error(f"RK45 integration failed: {solution.message}")
                return None
        else:
            # Use custom RK4 for multi-body systems (default)
            logger.info(f"Using custom RK4 integration for {test_case}")

            trajectory = []
            state = initial_state.clone()

            for step in range(num_steps):
                trajectory.append(state.clone())

                # RK4 integration
                def dynamics(s):
                    # Get accelerations from force function
                    acc = force_func(s)
                    # Construct state derivative [velocities, accelerations]
                    velocities = s[:, 1::2]  # Extract all velocities
                    state_dot = torch.zeros_like(s)
                    state_dot[:, 0::2] = velocities  # Position derivatives are velocities
                    state_dot[:, 1::2] = acc  # Velocity derivatives are accelerations
                    return state_dot

                k1 = dynamics(state)
                k2 = dynamics(state + 0.5 * dt * k1)
                k3 = dynamics(state + 0.5 * dt * k2)
                k4 = dynamics(state + dt * k3)

                state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

            training_set = torch.stack(trajectory)
            logger.info(f"Generated trajectory shape: {training_set.shape}")

    elif test_case == "Triple_Mass_Spring_Damper":
        # Support both RK4 and analytical methods for TMSD
        if numerical_methods == "rk4":
            logger.info(f"Using custom RK4 integration for {test_case}")

            trajectory = []
            state = initial_state.clone()

            for step in range(num_steps):
                trajectory.append(state.clone())

                # RK4 integration
                def dynamics(s):
                    # Reshape to match force function expectations
                    # force_tmsd expects [3, 2] not [1, 6]
                    s_reshaped = s.view(3, 2)  # Reshape to [3, 2] for 3 bodies
                    acc = force_func(s_reshaped)  # Returns [3] accelerations
                    # Construct state derivative
                    velocities = s[1::2]  # Extract velocities [v1, v2, v3]
                    state_dot = torch.zeros_like(s)
                    state_dot[0::2] = velocities  # Position derivatives
                    state_dot[1::2] = acc  # Velocity derivatives (already correct shape)
                    return state_dot

                state_flat = state.flatten()
                k1 = dynamics(state_flat)
                k2 = dynamics(state_flat + 0.5 * dt * k1)
                k3 = dynamics(state_flat + 0.5 * dt * k2)
                k4 = dynamics(state_flat + dt * k3)

                state_flat = state_flat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                state = state_flat.view(num_bodies, 2)

            training_set = torch.stack(trajectory)
            logger.info(f"Generated trajectory shape: {training_set.shape}")

        elif numerical_methods == "analytical":
            logger.info(f"Using analytical solution for {test_case}")
            trajectory = generate_analytical_tmsd(initial_state, dt, num_steps)
            training_set = trajectory
            logger.info(f"Generated trajectory shape: {training_set.shape}")

        elif numerical_methods == "rk45":
            logger.info(f"Using scipy RK45 integration for {test_case}")

            # Define dynamics function for scipy
            def scipy_dynamics(t, state_flat):
                # Reshape flat state to multi-body format [3, 2]
                state = state_flat.reshape(3, 2)
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get accelerations from force function
                acc = force_func(state_tensor)  # Returns [3] accelerations
                if isinstance(acc, torch.Tensor):
                    acc = acc.numpy()

                # Construct state derivative
                state_dot = np.zeros_like(state)
                state_dot[:, 0] = state[:, 1]  # Position derivatives are velocities
                state_dot[:, 1] = acc  # Velocity derivatives are accelerations

                return state_dot.flatten()

            # Set up integration
            t_eval = np.arange(num_steps, dtype=float) * dt
            t_span = [0, (num_steps - 1) * dt]
            initial_state_flat = initial_state_np.flatten()

            # Integrate using RK45
            solution = scipy.integrate.solve_ivp(
                scipy_dynamics, t_span, initial_state_flat,
                method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
            )

            if solution.success:
                trajectory_np = solution.y.T.reshape(num_steps, 3, 2)
                training_set = torch.tensor(trajectory_np, dtype=torch.float32, device=device)
                logger.info(f"Generated trajectory shape: {training_set.shape}")
            else:
                logger.error(f"RK45 integration failed: {solution.message}")
                return None

        else:
            raise ValueError(f"Unsupported numerical method '{numerical_methods}' for {test_case}. "
                           f"Supported methods: 'rk4', 'analytical', 'rk45'")

    else:
        # For simple single-body systems
        logger.info(f"Using standard integration for {test_case}")

        if numerical_methods == "rk4":
            if test_case == "Single_Mass_Spring":
                trajectory = generate_trajectory_rk4_sms(
                    initial_state, force_func, dt, num_steps
                )
            elif test_case == "Single_Mass_Spring_Damper":
                trajectory = generate_trajectory_rk4_smsd(
                    initial_state, force_func, dt, num_steps
                )
            else:
                raise ValueError(f"RK4 not implemented for {test_case}")

        elif numerical_methods == "analytical":
            if test_case == "Single_Mass_Spring":
                trajectory = generate_analytical_sms(initial_state, dt, num_steps)
            elif test_case == "Single_Mass_Spring_Damper":
                trajectory = generate_analytical_smsd(initial_state, dt, num_steps)
            elif test_case == "Triple_Mass_Spring_Damper":
                trajectory = generate_analytical_tmsd(initial_state, dt, num_steps)
            else:
                raise ValueError(f"Analytical solution not available for {test_case}")

        elif numerical_methods == "fe":
            if test_case == "Single_Mass_Spring":
                trajectory = generate_trajectory_forward_euler_sms(
                    initial_state, force_func, dt, num_steps
                )
            else:
                raise ValueError(f"Forward Euler not implemented for {test_case}")

        elif numerical_methods == "midpoint":
            if test_case == "Single_Mass_Spring":
                trajectory = generate_trajectory_midpoint_sms(
                    initial_state, force_func, dt, num_steps
                )
            else:
                raise ValueError(f"Midpoint method not implemented for {test_case}")

        elif numerical_methods == "rk45":
            logger.info(f"Using scipy RK45 integration for {test_case}")

            # Define dynamics function for scipy
            def scipy_dynamics(t, state_flat):
                state = state_flat.reshape(-1, 2)  # Reshape to [num_bodies, 2]
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get accelerations from force function
                acc = force_func(state_tensor)
                if isinstance(acc, torch.Tensor):
                    acc = acc.numpy()

                # Construct state derivative
                state_dot = np.zeros_like(state)
                state_dot[:, 0] = state[:, 1]  # Position derivatives are velocities
                state_dot[:, 1] = acc.reshape(-1) if acc.ndim > 1 else acc  # Velocity derivatives are accelerations

                return state_dot.flatten()

            # Set up integration
            t_eval = np.arange(num_steps, dtype=float) * dt
            t_span = [0, (num_steps - 1) * dt]
            initial_state_flat = initial_state_np.flatten()

            # Integrate using RK45
            solution = scipy.integrate.solve_ivp(
                scipy_dynamics, t_span, initial_state_flat,
                method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
            )

            if solution.success:
                trajectory_np = solution.y.T.reshape(num_steps, -1, 2)
                trajectory = torch.tensor(trajectory_np, dtype=torch.float32, device=device).squeeze(1)
                logger.info(f"RK45 integration successful for {test_case}")
            else:
                logger.error(f"RK45 integration failed: {solution.message}")
                return None

        else:
            raise ValueError(f"Unknown numerical method: {numerical_methods}")

        # Reshape trajectory to standard format [num_steps, num_bodies, 2]
        training_set = trajectory.unsqueeze(1) if len(trajectory.shape) == 2 else trajectory
        logger.info(f"Generated trajectory shape: {training_set.shape}")

    # =========================================================================
    # Save dataset with consistent structure
    # =========================================================================

    if training_set is not None:
        # Split data into train and test sets.
        # Prefer explicit step counts (used by main_mbdnode.py style callers).
        gen_train_num_steps = kwargs.get('gen_train_num_steps', None)
        gen_test_num_steps = kwargs.get('gen_test_num_steps', None)

        if gen_train_num_steps is not None:
            train_steps = int(gen_train_num_steps)
        else:
            train_split_ratio = kwargs.get('train_split_ratio', 0.75)
            train_steps = int(num_steps * float(train_split_ratio))

        # Clamp to valid range
        train_steps = max(1, min(train_steps, num_steps - 1))

        if gen_test_num_steps is not None:
            test_steps = int(gen_test_num_steps)
            test_steps = max(1, min(test_steps, num_steps - train_steps))
        else:
            test_steps = num_steps - train_steps

        # Reshape to 2D for saving: [num_steps, num_features]
        if len(training_set.shape) == 3:
            # Multi-body system: flatten bodies into features
            save_data = training_set.view(num_steps, -1).cpu().numpy()
        else:
            # Single trajectory
            save_data = training_set.cpu().numpy()

        # Split train/test
        train_data = save_data[:train_steps]
        test_data = save_data[train_steps:train_steps + test_steps]

        # Generate time arrays
        t_train = np.arange(train_steps) * dt
        t_test = np.arange(test_steps) * dt + t_train[-1] + dt
        t_full = np.arange(num_steps) * dt

        # Save as CSV files
        np.savetxt(os.path.join(dataset_dir, 's_train.csv'), train_data, delimiter=',',
                   header=','.join([f'state_{i}' for i in range(train_data.shape[1])]), comments='')
        np.savetxt(os.path.join(dataset_dir, 's_test.csv'), test_data, delimiter=',',
                   header=','.join([f'state_{i}' for i in range(test_data.shape[1])]), comments='')
        np.savetxt(os.path.join(dataset_dir, 's_full.csv'), save_data, delimiter=',',
                   header=','.join([f'state_{i}' for i in range(save_data.shape[1])]), comments='')
        # Save time arrays with proper header for pandas compatibility
        pd.DataFrame({'time': t_train}).to_csv(os.path.join(dataset_dir, 't_train.csv'), index=False)
        pd.DataFrame({'time': t_test}).to_csv(os.path.join(dataset_dir, 't_test.csv'), index=False)
        pd.DataFrame({'time': t_full}).to_csv(os.path.join(dataset_dir, 't_full.csv'), index=False)

        logger.info(f"Saved dataset to {dataset_dir}/")
        logger.info(f"  Train: {train_steps} steps, shape {train_data.shape}")
        logger.info(f"  Test: {test_steps} steps, shape {test_data.shape}")

    return training_set


def save_npz_dataset(dataset_path, bodys, forces, accelerations, **metadata):
    """
    Save dataset in NPZ format with metadata.

    Args:
        dataset_path (str): Path to save the NPZ file
        bodys (torch.Tensor): Body states
        forces (torch.Tensor): Forces
        accelerations (torch.Tensor): Accelerations
        **metadata: Additional metadata to save
    """
    # Convert tensors to numpy
    bodys_np = bodys.cpu().numpy() if isinstance(bodys, torch.Tensor) else bodys
    forces_np = forces.cpu().numpy() if isinstance(forces, torch.Tensor) else forces
    accelerations_np = accelerations.cpu().numpy() if isinstance(accelerations, torch.Tensor) else accelerations

    # Save with metadata
    np.savez(dataset_path,
             bodys=bodys_np,
             forces=forces_np,
             accelerations=accelerations_np,
             **metadata)

    logger.info(f"Saved NPZ dataset to {dataset_path}")


def load_npz_dataset(dataset_path):
    """
    Load dataset from NPZ format.

    Args:
        dataset_path (str): Path to the NPZ file

    Returns:
        tuple: (bodys_tensor, force_tensor, accel_tensor, metadata_dict)
    """
    logger.info(f"Loading dataset from {dataset_path}")

    # Load the NPZ file
    data = np.load(dataset_path, allow_pickle=True)

    # Convert numpy arrays back to tensors
    bodys_tensor = torch.tensor(data['bodys'], dtype=torch.float32)
    force_tensor = torch.tensor(data['forces'], dtype=torch.float32)
    accel_tensor = torch.tensor(data['accelerations'], dtype=torch.float32)

    # Log metadata
    if 'seed' in data:
        logger.info(f"Dataset seed: {data['seed']}")
    if 'test_case' in data:
        logger.info(f"Test case: {data['test_case']}")
    if 'num_steps' in data:
        logger.info(f"Number of steps: {data['num_steps']}")

    logger.info(f"Loaded shapes - Bodys: {bodys_tensor.shape}, Forces: {force_tensor.shape}, Accels: {accel_tensor.shape}")

    return bodys_tensor, force_tensor, accel_tensor


# ---------------------------------------------------------------------------
# PNODE / MNODE Legacy Support Functions
# ---------------------------------------------------------------------------

def generate_detailed_training_data(test_case, numerical_methods, dt, num_steps,
                                   seed=42):
    """
    PNODE-compatible detailed data generator.
    Generates (bodys, forces, accelerations) triplet for physics-informed training.
    """
    logger.info(f"[generate_detailed_training_data] Starting for {test_case}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize lists
    bodys = []
    forces = []
    accels = []

    if test_case == "Single_Mass_Spring":
        initial_state = torch.tensor([[1.0, 0.0]], device=device)

        # Generate trajectory and collect forces
        state = initial_state.clone()
        for step in range(num_steps):
            bodys.append(state.clone())

            # Calculate force and acceleration at current state
            force = force_sms(state, if_external_force=False, t=step*dt)
            forces.append(force.clone())
            accels.append(force.clone())  # For unit mass, F = ma = a

            # RK4 step
            def dynamics(s):
                acc = force_sms(s, if_external_force=False, t=step*dt)
                return torch.cat([s[:, 1:2], acc], dim=1)

            k1 = dynamics(state)
            k2 = dynamics(state + 0.5 * dt * k1)
            k3 = dynamics(state + 0.5 * dt * k2)
            k4 = dynamics(state + dt * k3)

            state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        # Stack into tensors
        bodys_tensor = torch.stack(bodys).squeeze(1)  # [num_steps, 2]
        force_tensor = torch.stack(forces).squeeze(1)  # [num_steps, 1]
        accel_tensor = torch.stack(accels).squeeze(1)  # [num_steps, 1]

    else:
        raise ValueError(f"Detailed data generation not implemented for {test_case}")

    logger.info(f"[generate_detailed_training_data] Generated shapes:")
    logger.info(f"  Bodys: {bodys_tensor.shape}")
    logger.info(f"  Forces: {force_tensor.shape}")
    logger.info(f"  Accels: {accel_tensor.shape}")

    return bodys_tensor, force_tensor, accel_tensor


def generate_trajectory_based_dataset(test_case, num_states, num_steps,
                                     state_bounds=(-5, 5), force_func=None,
                                     dt=0.01, seed=42):
    """
    Generate dataset by sampling states and computing forces/accelerations.
    Used for force network pre-training in PNODE/MNODE.
    """
    logger.info(f"[generate_trajectory_based_dataset] Generating for {test_case}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if force_func is None:
        if test_case == "Single_Mass_Spring":
            force_func = force_sms
        else:
            raise ValueError(f"No force function provided for {test_case}")

    # Generate random states
    states = []
    forces = []
    accels = []

    for _ in range(num_states):
        # Random initial state
        if test_case == "Single_Mass_Spring":
            pos = np.random.uniform(-2, 2)
            vel = np.random.uniform(-2, 2)
            initial_state = torch.tensor([[pos, vel]], device=device)
        else:
            raise ValueError(f"State generation not implemented for {test_case}")

        # Generate short trajectory
        state = initial_state.clone()
        for step in range(num_steps):
            states.append(state.clone())

            # Calculate force
            force = force_func(state, if_external_force=False, t=step*dt)
            forces.append(force.clone())
            accels.append(force.clone())  # Unit mass assumption

            # Simple Euler step
            vel = state[:, 1:2]
            acc = force
            state = state + dt * torch.cat([vel, acc], dim=1)

    # Stack all samples
    states_tensor = torch.cat(states, dim=0)  # [num_states * num_steps, 2]
    force_tensor = torch.cat(forces, dim=0)   # [num_states * num_steps, 1]
    accel_tensor = torch.cat(accels, dim=0)   # [num_states * num_steps, 1]

    logger.info(f"[generate_trajectory_based_dataset] Generated {len(states_tensor)} samples")

    return states_tensor, force_tensor, accel_tensor


def generate_batch_dataset_from_dynamics(test_case, num_states, num_steps,
                                        dt=0.01,
                                        save_to_disk=False, dataset_path=None,
                                        seed=42):
    """
    Batch generation of multiple trajectories from different initial conditions.
    Returns states, forces, and accelerations for supervised learning.
    """
    logger.info(f"[generate_batch_dataset_from_dynamics] Generating batch dataset for {test_case}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_states = []
    all_forces = []
    all_accels = []

    for i in range(num_states):
        if test_case == "Single_Mass_Spring":
            # Random initial conditions
            pos = np.random.uniform(-2, 2)
            vel = np.random.uniform(-2, 2)
            initial_state = torch.tensor([[pos, vel]], device=device)

            # Generate trajectory
            trajectory = generate_trajectory_rk4_sms(
                initial_state, force_sms, dt, num_steps
            )

            # Calculate forces and accelerations along trajectory
            forces = []
            for state in trajectory:
                force = force_sms(state.unsqueeze(0), if_external_force=False)
                forces.append(force)

            forces = torch.cat(forces, dim=0)
            accels = forces  # Unit mass

            all_states.append(trajectory)
            all_forces.append(forces)
            all_accels.append(accels)

        else:
            raise ValueError(f"Batch generation not implemented for {test_case}")

    # Stack all trajectories
    states_tensor = torch.stack(all_states)  # [num_states, num_steps, 2]
    force_tensor = torch.stack(all_forces)   # [num_states, num_steps, 1]
    accel_tensor = torch.stack(all_accels)   # [num_states, num_steps, 1]

    logger.info(f"[generate_batch_dataset_from_dynamics] Generated batch:")
    logger.info(f"  States: {states_tensor.shape}")
    logger.info(f"  Forces: {force_tensor.shape}")
    logger.info(f"  Accels: {accel_tensor.shape}")

    if save_to_disk and dataset_path:
        # Save as NPZ
        np.savez(dataset_path,
                 bodys=states_tensor.cpu().numpy(),
                 forces=force_tensor.cpu().numpy(),
                 accelerations=accel_tensor.cpu().numpy(),
                 test_case=test_case,
                 num_states=num_states,
                 num_steps=num_steps,
                 generation_method='trajectory_based')

        logger.info(f"Saved trajectory-based dataset to {dataset_path}")

    return states_tensor, force_tensor, accel_tensor


