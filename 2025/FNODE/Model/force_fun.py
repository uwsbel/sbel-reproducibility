# FNODE/Model/force_fun.py
import numpy as np
import torch
import scipy.integrate
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Standard Force Functions (Return Acceleration) ---

def force_sms(bodys, model=None):
    """Acceleration for single mass-spring system."""
    # bodys: tensor shape [1, 2] -> [[position, velocity]]
    mass = 10.0;
    k = 50.0
    # Returns acceleration -k/m * x
    accel = (-k * bodys[:, 0]) / mass
    return accel


def force_smsd(bodys, model=None):
    """Acceleration for single mass-spring-damper system."""
    # bodys: tensor shape [1, 2] -> [[position, velocity]]
    mass = 10.0;
    k_linear = 50.0;
    k_cubic = 0.1;
    c_damping = 2.0
    # Returns acceleration: (-k_linear*x - k_cubic*x^3 - c_damping*v) / m
    accel = (-k_linear * bodys[:, 0] - k_cubic * bodys[:, 0] ** 3 - c_damping * bodys[:, 1]) / mass
    return accel


def force_tmsd(bodys, model=None):
    """Accelerations for triple mass-spring-damper system."""
    # bodys: tensor shape [3, 2] -> [[x1,v1], [x2,v2], [x3,v3]]
    masses = torch.tensor([100.0, 10.0, 1.0], dtype=torch.float32, device=bodys.device)
    k = 50.0;
    c = 2.0
    forces = torch.zeros(3, dtype=torch.float32, device=bodys.device)
    # Forces on body 3 (top mass)
    forces[2] = -k * (bodys[2, 0] - bodys[1, 0]) - c * (bodys[2, 1] - bodys[1, 1])
    # Forces on body 2 (middle mass)
    forces[1] = k * (bodys[2, 0] - bodys[1, 0]) + c * (bodys[2, 1] - bodys[1, 1]) - \
                k * (bodys[1, 0] - bodys[0, 0]) - c * (bodys[1, 1] - bodys[0, 1])
    # Forces on body 1 (bottom mass, connected to ground)
    forces[0] = k * (bodys[1, 0] - bodys[0, 0]) + c * (bodys[1, 1] - bodys[0, 1]) - \
                k * bodys[0, 0] - c * bodys[0, 1]
    accel = forces / masses
    return accel  # Return accelerations directly


# --- System Derivatives (dy/dt = f(t,y) for scipy.integrate.solve_ivp) ---

def single_pendulum_derivs(t, state):
    """Derivative function for a single pendulum."""
    # state: numpy array [theta, omega] (radians)
    G = 9.81;
    L = 1.0
    dydt = np.zeros_like(state)
    dydt[0] = state[1]  # d(theta)/dt = omega
    dydt[1] = -(G / L) * np.sin(state[0])  # d(omega)/dt = -g/L * sin(theta)
    return dydt


def double_pendulum_derivs(t, state):
    """Derivative function for a double pendulum."""
    # state: numpy array [theta1, omega1, theta2, omega2] (radians)
    G = 9.8;
    L1 = 1.0;
    M1 = 1.0;  # Changed back to mass=1.0
    L2 = 1.0;
    M2 = 1.0;  # Changed back to mass=1.0
    th1, w1, th2, w2 = state[0], state[1], state[2], state[3]
    dydt = np.zeros_like(state)
    dydt[0] = w1;
    dydt[2] = w2
    delta = th2 - th1
    cos_delta = np.cos(delta);
    sin_delta = np.sin(delta)
    den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta ** 2
    if abs(den1) < 1e-9: den1 = 1e-9  # Avoid division by zero
    dydt[1] = (M2 * L1 * w1 ** 2 * sin_delta * cos_delta +
               M2 * G * np.sin(th2) * cos_delta +
               M2 * L2 * w2 ** 2 * sin_delta -
               (M1 + M2) * G * np.sin(th1)) / den1
    den2 = (L2 / L1) * den1
    if abs(den2) < 1e-9: den2 = 1e-9  # Avoid division by zero
    dydt[3] = (-M2 * L2 * w2 ** 2 * sin_delta * cos_delta +
               (M1 + M2) * G * np.sin(th1) * cos_delta -
               (M1 + M2) * L1 * w1 ** 2 * sin_delta -
               (M1 + M2) * G * np.sin(th2)) / den2
    return dydt


# --- Analytical Solutions (if applicable) ---

def analytic_sms(bodys, dt, model=None):
    """Analytic solution for the next step of a simple single mass-spring system."""
    # bodys: tensor shape [1, 2] -> [[position, velocity]]
    k = torch.tensor(50.0, device=bodys.device);
    m = torch.tensor(10.0, device=bodys.device)
    dt_tensor = torch.tensor(dt, device=bodys.device, dtype=bodys.dtype) if not isinstance(dt, torch.Tensor) else dt.to(
        bodys.device, bodys.dtype)
    x0 = bodys[:, 0];
    v0 = bodys[:, 1];
    omega_freq = torch.sqrt(k / m)
    if omega_freq.item() < 1e-9:  # Handle case k=0 or m=inf
        x1 = x0 + v0 * dt_tensor
        v1 = v0
    else:
        A_amp = torch.sqrt(x0 ** 2 + (v0 / omega_freq) ** 2)
        phi_phase = torch.atan2(-v0, omega_freq * x0)  # Note: atan2(y,x)
        x1 = A_amp * torch.cos(omega_freq * dt_tensor + phi_phase)
        v1 = -A_amp * omega_freq * torch.sin(omega_freq * dt_tensor + phi_phase)
    return torch.stack([x1, v1], dim=-1)


def force_cp(bodys, model=None):
    """
    Calculate accelerations for cart-pole system.

    Args:
        bodys: tensor shape [2, 2] -> [[x_cart, v_cart], [theta_pole, omega_pole]]
               where:
               - x_cart: cart position
               - v_cart: cart velocity
               - theta_pole: pole angle (radians)
               - omega_pole: pole angular velocity
        model: optional model parameter

    Returns:
        tensor: [x_acceleration, theta_acceleration]
    """
    # System parameters - matching MBDNODE mass assumption
    m1 = 10.0  # cart mass (kg) - updated to match MBDNODE
    m2 = 10.0  # pole mass (kg) - updated to match MBDNODE
    l = 0.5  # pole length (m)
    g = 9.81  # gravity (m/s^2)

    # Extract state variables
    x = bodys[0, 0]  # cart position
    x_dot = bodys[0, 1]  # cart velocity
    theta = bodys[1, 0]  # pole angle
    theta_dot = bodys[1, 1]  # pole angular velocity

    # Calculate trigonometric functions
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Calculate cart acceleration (without external force)
    x_ddot_numerator = (-m2 * g * sin_theta * cos_theta -
                        m2 * l * theta_dot ** 2 * sin_theta)
    x_ddot_denominator = m1 + m2 * (1 - cos_theta ** 2)
    x_ddot = x_ddot_numerator / x_ddot_denominator

    # Calculate pole angular acceleration
    theta_ddot_numerator = (g * sin_theta * (m1 + m2) +
                            m2 * l * theta_dot ** 2 * sin_theta * cos_theta)
    theta_ddot_denominator = l * (m1 + m2 * (1 - cos_theta ** 2))
    theta_ddot = theta_ddot_numerator / theta_ddot_denominator

    return torch.tensor([x_ddot, theta_ddot], dtype=torch.float32, device=bodys.device)


def force_cp_controlled(state_and_control, model=None):
    """
    Calculate accelerations for controlled cart-pole system.
    
    Args:
        state_and_control: array [theta, x, theta_dot, x_dot, u]
        
    Returns:
        array: [theta_ddot, x_ddot]
    """
    # Extract state and control
    theta = state_and_control[0]
    x = state_and_control[1] 
    theta_dot = state_and_control[2]
    x_dot = state_and_control[3]
    u = state_and_control[4]  # Control force
    
    # System parameters - matching MBDNODE mass assumption
    m1 = 10.0  # cart mass (kg) - updated to match MBDNODE
    m2 = 10.0  # pole mass (kg) - updated to match MBDNODE
    l = 0.5   # pole half-length (m)
    g = 9.81  # gravity (m/s^2)
    
    # Calculate trigonometric functions
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Controlled cart-pole dynamics equations
    # From Lagrangian mechanics with external force u
    
    # Denominator (common term)
    denominator = m1 + m2 * sin_theta**2
    
    # Cart acceleration with control force u
    x_ddot_numerator = (u + m2 * l * sin_theta * theta_dot**2 - 
                        m2 * g * sin_theta * cos_theta)
    x_ddot = x_ddot_numerator / denominator
    
    # Pole angular acceleration  
    theta_ddot_numerator = (-u * cos_theta - m2 * l * theta_dot**2 * sin_theta * cos_theta +
                            (m1 + m2) * g * sin_theta)
    theta_ddot = theta_ddot_numerator / (l * denominator)
    
    return np.array([theta_ddot, x_ddot])

def get_slider_crank_angular_acceleration(theta, omega, t, T_ext, r_crank=1.0, l_rod=4.0):
    """
    Calculates angular acceleration for slider crank mechanism based on slider_crank_bb.py physics.

    This improved function better approximates the full dynamics by:
    1. Using the same motor torque function as in slider_crank_bb.py
    2. Estimating the rod angular velocity based on slider-crank kinematics
    3. Calculating damping based on relative angular velocity
    4. Approximating the slider friction effect on the crank

    Args:
        theta: Current crank angle (rad)
        omega: Current crank angular velocity (rad/s)
        t: Current time (s)
        T_ext: External torque applied to the crank (N·m) - can be overridden by motor_torque
        r_crank: Crank length (radius) in meters
        l_rod: Rod length in meters

    Returns:
        Angular acceleration of the crank (rad/s²)
    """
    import numpy as np

    # Physical parameters from slider_crank_bb.py
    J1 = 0.10  # Crank moment of inertia (kg·m²)
    c12 = 0.05  # Rotational damping coefficient
    gamma = 1.0  # Exponent for nonlinear damping

    # Slider parameters
    c_slide = 0.20  # Slider friction coefficient
    psi = 1.0  # Friction exponent

    # Use the motor torque function directly from slider_crank_bb.py
    motor_torque = -0.002 * np.sin(2 * np.pi * t)

    # Use T_ext if provided, otherwise use the standard motor torque function
    applied_torque = T_ext if T_ext != 0 else motor_torque

    # Estimate rod angular velocity (ω₂) based on slider-crank kinematics
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Calculate rod angle (β) from the crank angle
    temp = r_crank * sin_theta / l_rod
    # Clip to valid range for arcsin
    temp_clipped = np.clip(temp, -1.0, 1.0)
    beta = np.arcsin(temp_clipped)  # Rod angle

    # Calculate derivative of rod angle w.r.t. crank angle
    # Using the chain rule: dβ/dt = (dβ/dθ) * (dθ/dt) = (dβ/dθ) * omega
    # First calculate dβ/dθ
    dbeta_dtheta = (r_crank * cos_theta / l_rod) / np.sqrt(1 - temp_clipped ** 2 + 1e-10)
    # Rod angular velocity
    omega2 = dbeta_dtheta * omega

    # Calculate rotational damping torque based on relative angular velocity
    rel_omega = omega - omega2
    damping_torque = c12 * np.abs(rel_omega) ** gamma * np.sign(rel_omega)

    # Calculate slider velocity
    slider_x = r_crank * cos_theta + l_rod * np.cos(beta)
    slider_vx = -r_crank * sin_theta * omega - l_rod * np.sin(beta) * omega2

    # Calculate slider friction force
    slider_friction = c_slide * np.abs(slider_vx) ** psi * np.sign(slider_vx)

    # Convert slider friction to equivalent torque on crank
    dx_dtheta = -r_crank * sin_theta - l_rod * np.sin(beta) * dbeta_dtheta
    friction_torque = slider_friction * dx_dtheta

    # Net torque acting on the crank
    net_torque = applied_torque - damping_torque - friction_torque

    # Angular acceleration = Net torque / Moment of inertia
    alpha = net_torque / J1

    return alpha

# --- NEW: Analytical Acceleration Calculation Functions ---

def calculate_analytical_accelerations_sms(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Single Mass Spring system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters (same as in force_sms)
    mass = 10.0
    k = 50.0

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]
    num_bodies = 1  # Single mass system

    # Calculate analytical accelerations
    analytical_accelerations = np.zeros((num_steps, num_bodies))

    for i in range(num_steps):
        x = s_train_np[i, 0]  # Position
        # Analytical acceleration: a = -k/m * x
        analytical_accelerations[i, 0] = (-k * x) / mass

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations_smsd(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Single Mass Spring Damper system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters (same as in force_smsd)
    mass = 10.0
    k_linear = 50.0
    k_cubic = 0.1
    c_damping = 2.0

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]
    num_bodies = 1  # Single mass system

    # Calculate analytical accelerations
    analytical_accelerations = np.zeros((num_steps, num_bodies))

    for i in range(num_steps):
        x = s_train_np[i, 0]  # Position
        v = s_train_np[i, 1]  # Velocity
        # Analytical acceleration: a = (-k_linear*x - k_cubic*x^3 - c_damping*v) / m
        analytical_accelerations[i, 0] = (-k_linear * x - k_cubic * x ** 3 - c_damping * v) / mass

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations_tmsd(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Triple Mass Spring Damper system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters (same as in force_tmsd)
    masses = np.array([100.0, 10.0, 1.0])
    k = 50.0
    c = 2.0

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]
    num_bodies = 3  # Triple mass system

    # Calculate analytical accelerations
    analytical_accelerations = np.zeros((num_steps, num_bodies))

    for i in range(num_steps):
        # Extract positions and velocities for all bodies
        x1, v1 = s_train_np[i, 0], s_train_np[i, 1]  # Body 1
        x2, v2 = s_train_np[i, 2], s_train_np[i, 3]  # Body 2
        x3, v3 = s_train_np[i, 4], s_train_np[i, 5]  # Body 3

        # Calculate forces (same as in force_tmsd)
        forces = np.zeros(3)

        # Forces on body 3 (top mass)
        forces[2] = -k * (x3 - x2) - c * (v3 - v2)

        # Forces on body 2 (middle mass)
        forces[1] = k * (x3 - x2) + c * (v3 - v2) - k * (x2 - x1) - c * (v2 - v1)

        # Forces on body 1 (bottom mass, connected to ground)
        forces[0] = k * (x2 - x1) + c * (v2 - v1) - k * x1 - c * v1

        # Calculate accelerations: a = F/m
        analytical_accelerations[i, :] = forces / masses

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations_double_pendulum(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Double Pendulum system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters (same as in double_pendulum_derivs)
    G = 9.8
    L1 = 1.0
    M1 = 1.0
    L2 = 1.0
    M2 = 1.0

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]
    num_bodies = 2  # Double pendulum system

    # Calculate analytical accelerations
    analytical_accelerations = np.zeros((num_steps, num_bodies))

    for i in range(num_steps):
        # Extract angles and angular velocities
        th1 = s_train_np[i, 0]  # theta1
        w1 = s_train_np[i, 1]  # omega1
        th2 = s_train_np[i, 2]  # theta2
        w2 = s_train_np[i, 3]  # omega2

        # Calculate angular accelerations using double pendulum equations
        delta = th2 - th1
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        # Denominators (with safety checks)
        den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta ** 2
        if abs(den1) < 1e-9:
            den1 = 1e-9

        den2 = (L2 / L1) * den1
        if abs(den2) < 1e-9:
            den2 = 1e-9

        # Angular accelerations
        alpha1 = (M2 * L1 * w1 ** 2 * sin_delta * cos_delta +
                  M2 * G * np.sin(th2) * cos_delta +
                  M2 * L2 * w2 ** 2 * sin_delta -
                  (M1 + M2) * G * np.sin(th1)) / den1

        alpha2 = (-M2 * L2 * w2 ** 2 * sin_delta * cos_delta +
                  (M1 + M2) * G * np.sin(th1) * cos_delta -
                  (M1 + M2) * L1 * w1 ** 2 * sin_delta -
                  (M1 + M2) * G * np.sin(th2)) / den2

        analytical_accelerations[i, 0] = alpha1
        analytical_accelerations[i, 1] = alpha2

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations_slider_crank(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Slider Crank system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters for slider crank
    r_crank = 1.0
    l_rod = 4.0

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]
    num_bodies = 1  # Slider crank system (only angular acceleration of crank)

    # Calculate analytical accelerations
    analytical_accelerations = np.zeros((num_steps, num_bodies))

    for i in range(num_steps):
        # For Slider Crank, we typically have theta and omega
        theta = s_train_np[i, 0]  # Crank angle
        omega = s_train_np[i, 1]  # Crank angular velocity
        t = t_train_np[i]  # Current time

        # Calculate analytical angular acceleration using slider crank dynamics
        T_ext = 0.0  # Use default motor torque
        alpha = get_slider_crank_angular_acceleration(theta, omega, t, T_ext, r_crank, l_rod)

        analytical_accelerations[i, 0] = alpha

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations_cart_pole(s_train, t_train, test_case, output_path,
                                                 filename="analytical_accelerations_for_plotting.csv"):
    """Calculate analytical accelerations for Cart Pole system."""
    logger.info(f"Calculating analytical accelerations for {test_case}")

    # Physical parameters (same as in force_cp)
    mc = 1.0  # mass of cart
    mp = 0.1  # mass of pole
    l = 0.5  # half-length of pole
    g = 9.8  # gravity

    # Convert tensors to numpy for processing
    if isinstance(s_train, torch.Tensor):
        s_train_np = s_train.detach().cpu().numpy()
    else:
        s_train_np = np.array(s_train)

    if isinstance(t_train, torch.Tensor):
        t_train_np = t_train.detach().cpu().numpy()
    else:
        t_train_np = np.array(t_train)

    # Reshape if necessary
    if s_train_np.ndim == 3:  # [steps, bodies, 2]
        s_train_np = s_train_np.reshape(s_train_np.shape[0], -1)

    num_steps = s_train_np.shape[0]

    # Determine if we have 2-state or 4-state system
    if s_train_np.shape[1] == 2:
        # 2-state system: [theta, theta_dot] - simple pendulum
        num_bodies = 1
        analytical_accelerations = np.zeros((num_steps, num_bodies))

        for i in range(num_steps):
            theta = s_train_np[i, 0]  # pole angle
            # Simple pendulum: theta_ddot = -(g/l) * sin(theta)
            analytical_accelerations[i, 0] = -(g / l) * np.sin(theta)

    elif s_train_np.shape[1] == 4:
        # 4-state system: [x, x_dot, theta, theta_dot] - full cart-pole
        num_bodies = 2
        analytical_accelerations = np.zeros((num_steps, num_bodies))

        for i in range(num_steps):
            x = s_train_np[i, 0]  # cart position
            x_dot = s_train_np[i, 1]  # cart velocity
            theta = s_train_np[i, 2]  # pole angle
            theta_dot = s_train_np[i, 3]  # pole angular velocity

            # External force (assumed zero for free dynamics)
            F = 0.0

            # Calculate accelerations using correct cart-pole equations
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            # Avoid division by zero
            denominator = mc + mp * sin_theta ** 2
            if abs(denominator) < 1e-8:
                denominator = 1e-8

            # Cart acceleration
            numerator_cart = F + mp * l * sin_theta * (theta_dot ** 2 + g * cos_theta / l)
            x_ddot = numerator_cart / denominator

            # Pole angular acceleration
            numerator_pole = (-F * cos_theta - mp * l * theta_dot ** 2 * sin_theta * cos_theta -
                              (mc + mp) * g * sin_theta)
            theta_ddot = numerator_pole / (l * denominator)

            analytical_accelerations[i, 0] = x_ddot
            analytical_accelerations[i, 1] = theta_ddot
    else:
        logger.error(f"Unsupported cart pole state dimension: {s_train_np.shape[1]}")
        return None

    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    # Create DataFrame
    columns = [f'analytical_accel_body_{j}' for j in range(num_bodies)]
    df = pd.DataFrame(analytical_accelerations, columns=columns)
    df['time'] = t_train_np

    # Reorder columns to have time first
    df = df[['time'] + columns]

    df.to_csv(filepath, index=False, float_format='%.8g')
    logger.info(f"Analytical accelerations for {test_case} saved to {filepath}")

    return analytical_accelerations


def calculate_analytical_accelerations(s_train, t_train, test_case, output_path, filename="analytical_accelerations_for_plotting.csv"):
    """Main function to calculate analytical accelerations for any test case."""
    logger.info(f"Starting analytical acceleration calculation for {test_case}")

    # Dispatch to appropriate function based on test case
    if test_case == "Single_Mass_Spring":
        return calculate_analytical_accelerations_sms(s_train, t_train, test_case, output_path, filename)
    elif test_case == "Single_Mass_Spring_Damper":
        return calculate_analytical_accelerations_smsd(s_train, t_train, test_case, output_path, filename)
    elif test_case == "Triple_Mass_Spring_Damper":
        return calculate_analytical_accelerations_tmsd(s_train, t_train, test_case, output_path, filename)
    elif test_case == "Double_Pendulum":
        return calculate_analytical_accelerations_double_pendulum(s_train, t_train, test_case, output_path, filename)
    elif test_case == "Slider_Crank":
        return calculate_analytical_accelerations_slider_crank(s_train, t_train, test_case, output_path, filename)
    elif test_case == "Cart_Pole":  # Add cart pole case
        return calculate_analytical_accelerations_cart_pole(s_train, t_train, test_case, output_path, filename)
    else:
        logger.warning(f"Analytical acceleration calculation not implemented for {test_case}")
        return None


def force_cp_ext_sequence_fnode(bodys_tensor, u_tensor):
    """
    Compute analytical accelerations for controlled cart-pole system (batch version).
    Follows MBDNODE-for-MBD2 approach for FNODE integration.
    
    Args:
        bodys_tensor: [num_steps, 2, 2] - states for cart and pole
                     bodys_tensor[i, 0, :] = [x, x_dot] (cart)
                     bodys_tensor[i, 1, :] = [theta, theta_dot] (pole)
        u_tensor: [num_steps, 1] - control forces
        
    Returns:
        accel_tensor: [num_steps, 2] - accelerations [x_ddot, theta_ddot]
    """
    device = bodys_tensor.device
    num_steps = bodys_tensor.shape[0]
    
    # System parameters - matching MBDNODE mass assumption
    m1 = 1.0    # cart mass (kg) - updated to match MBDNODE
    m2 = 1.0    # pole mass (kg) - updated to match MBDNODE
    l = 0.5     # pole half-length (m)
    g = 9.81    # gravity (m/s^2)
    
    # Initialize output tensor
    accel_tensor = torch.zeros((num_steps, 2), dtype=torch.float32, device=device)
    
    for i in range(num_steps):
        # Extract states for this sample
        x = bodys_tensor[i, 0, 0]           # cart position
        x_dot = bodys_tensor[i, 0, 1]       # cart velocity
        theta = bodys_tensor[i, 1, 0]       # pole angle
        theta_dot = bodys_tensor[i, 1, 1]   # pole angular velocity
        u = u_tensor[i, 0]                  # control force
        
        # Controlled cart-pole dynamics (same equations as MBDNODE-for-MBD2)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Common denominator
        denominator = m1 + m2 * (1 - cos_theta**2)
        
        # Cart acceleration with control
        x_ddot_num = -m2 * g * sin_theta * cos_theta - m2 * l * theta_dot**2 * sin_theta + u
        x_ddot = x_ddot_num / denominator
        
        # Pole angular acceleration with control
        theta_ddot_num = (g * sin_theta * (m1 + m2) + 
                         m2 * l * theta_dot**2 * sin_theta * cos_theta) - u * cos_theta
        theta_ddot_den = l * denominator
        theta_ddot = theta_ddot_num / theta_ddot_den
        
        # Store accelerations
        accel_tensor[i, 0] = x_ddot      # cart acceleration
        accel_tensor[i, 1] = theta_ddot  # pole angular acceleration
    
    return accel_tensor


