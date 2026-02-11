import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from Model.utils import *
from Model.force_fun import *
from Model.model import *

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Update the existing generate_slider_crank_dataset function in Data_generator.py
# to avoid numerical instability problems

def generate_slider_crank_dataset_with_friction(
        c_slide, time_span=20.0, dt=1e-3,
        root_dir='.', seed=42
):
    """
    Generate slider-crank dataset with variable friction parameter.

    Args:
        c_slide: Friction coefficient parameter [N·(m/s)^-ψ]
        time_span: Simulation time length in seconds (default 20.0s)
        dt: Internal timestep (1e-3), returns data sampled at 1e-2
        root_dir: Root directory for saving data
        seed: Random seed

    Returns:
        s_data: State data [theta, omega] sampled at dt=0.01
        t_data: Time vector sampled at dt=0.01
    """
    # Calculate total steps for internal dt=0.001
    total_num_steps = int(time_span / dt)

    # Same physics parameters as original
    m1, m2, m3 = 1.0, 1.0, 1.0
    J1, J2, J3 = 0.10, 0.10, 0.10
    L1, L2 = 1.0, 2.0
    g = 9.81
    tau1 = lambda t: -0.002 * np.sin(2 * np.pi * t)
    c12 = 0.05
    gamma = 1.0
    # Use the passed friction parameter
    psi = 1.0
    K_spring = 1.0
    delta = 1.0
    x3_max = 2.5 * L1 + 2.0 * L2

    # Rest of the implementation remains the same
    h = dt
    N_steps = total_num_steps
    n_gen = 9
    n_constr = 8
    M = np.diag([m1, m1, J1, m2, m2, J2, m3, m3, J3])

    def constraint(q: np.ndarray) -> np.ndarray:
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        return np.array([
            x1 - L1 * np.cos(th1),
            y1 - L1 * np.sin(th1),
            x1 + L1 * np.cos(th1) - x2 + L2 * np.cos(th2),
            y1 + L1 * np.sin(th1) - y2 + L2 * np.sin(th2),
            x2 + L2 * np.cos(th2) - x3,
            y2 + L2 * np.sin(th2) - y3,
            y3,
            th3
        ])

    def constraint_jacobian(q: np.ndarray) -> np.ndarray:
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        J = np.zeros((8, 9))
        J[0, 0] = 1.0
        J[0, 2] = L1 * np.sin(th1)
        J[1, 1] = 1.0
        J[1, 2] = -L1 * np.cos(th1)
        J[2, 0] = 1.0
        J[2, 3] = -1.0
        J[2, 2] = -L1 * np.sin(th1)
        J[2, 5] = -L2 * np.sin(th2)
        J[3, 1] = 1.0
        J[3, 4] = -1.0
        J[3, 2] = L1 * np.cos(th1)
        J[3, 5] = L2 * np.cos(th2)
        J[4, 3] = 1.0
        J[4, 6] = -1.0
        J[4, 5] = -L2 * np.sin(th2)
        J[5, 4] = 1.0
        J[5, 7] = -1.0
        J[5, 5] = L2 * np.cos(th2)
        J[6, 7] = 1.0
        J[7, 8] = 1.0
        return J

    def external_forces(q: np.ndarray, v: np.ndarray, lambda_val: float, t: float) -> np.ndarray:
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        vx1, vy1, w1, vx2, vy2, w2, vx3, vy3, w3 = v
        f = np.zeros(n_gen)
        f[1] = -m1 * g
        f[4] = -m2 * g
        f[7] = -m3 * g
        rel_w = w1 - w2
        damper = c12 * np.abs(rel_w) ** gamma * np.sign(rel_w)
        f[2] = tau1(t) - damper
        f[5] = -damper
        # Using passed friction parameter
        friction = c_slide * abs(lambda_val) * np.sign(vx3)
        spring = 0.0
        comp = x3_max - x3
        if comp > 0.0:
            spring = K_spring * comp ** delta
        f[6] = -(friction + spring)
        return f

    def bb_alm_step(v_guess: np.ndarray, lam_guess: np.ndarray,
                    v_prev: np.ndarray, q_prev: np.ndarray, t_next: float):
        v = v_guess.copy()
        lam = lam_guess.copy()
        rho = 1.0e10
        max_inner = 12
        max_outer = 25
        tol = 1.0e-6
        local_tol = 1.0e-1
        alpha = 1.0e-3
        use_bb1 = True
        normal_force = lam[6]

        def grad_L(v_loc: np.ndarray, lambda_val: float) -> np.ndarray:
            g_term = (M @ (v_loc - v_prev)) / h
            qA = q_prev + h * v_loc
            ext = external_forces(qA, v_loc, lambda_val, t_next)
            c_val = constraint(qA)
            J_val = constraint_jacobian(qA)
            return g_term - ext + J_val.T @ (lam + rho * h * c_val)

        for outer_iter in range(max_outer):
            local_tol = max(local_tol * 0.5, tol)
            vk, gk = v.copy(), grad_L(v, normal_force)
            for inner_iter in range(max_inner):
                vk1 = vk - alpha * gk
                gk1 = grad_L(vk1, lam[7])
                norm_gk1 = np.linalg.norm(gk1)
                if norm_gk1 < local_tol:
                    vk, gk = vk1, gk1
                    break
                s, y = vk1 - vk, gk1 - gk
                if use_bb1:
                    alpha = np.dot(s, s) / (np.dot(s, y) + 1e-12)
                else:
                    alpha = np.dot(s, y) / (np.dot(y, y) + 1e-12)
                use_bb1 = not use_bb1
                vk, gk = vk1, gk1
            v = vk
            qA = q_prev + h * v
            c_val = constraint(qA)
            lam += rho * h * c_val
            normal_force = lam[6]
            if np.linalg.norm(c_val) < tol:
                break
        return v, lam

    # Initial configuration
    theta1_0 = theta2_0 = theta3_0 = 0.0
    x1_0, y1_0 = L1, 0.0
    x2_0, y2_0 = 2 * L1 + L2, 0.0
    x3_0, y3_0 = 2 * L1 + 2 * L2, 0.0
    q = np.array([x1_0, y1_0, theta1_0, x2_0, y2_0, theta2_0, x3_0, y3_0, theta3_0])
    v = np.zeros(n_gen)
    lam = np.zeros(n_constr)
    v_guess = v.copy()

    # History arrays
    q_hist = np.zeros((N_steps + 1, n_gen))
    v_hist = np.zeros_like(q_hist)
    a_hist = np.zeros_like(q_hist)  # Add acceleration history
    q_hist[0] = q
    v_hist[0] = v
    a_hist[0] = np.zeros(n_gen)  # Initial acceleration

    # Time integration loop (silent)
    logger.info(f"Generating slider-crank data with c_slide={c_slide:.2f}, time_span={time_span}s")
    for k in range(N_steps):
        t_next = (k + 1) * h
        v, lam = bb_alm_step(v_guess, lam, v_prev=v, q_prev=q, t_next=t_next)
        q += h * v
        a = (v - v_hist[k]) / h
        q_hist[k + 1] = q
        v_hist[k + 1] = v
        a_hist[k + 1] = a  # Store acceleration
        v_guess = v + h * a

    # Extract theta1, omega1, and alpha1 (angular acceleration)
    theta1_data = q_hist[:-1, 2]
    omega1_data = v_hist[:-1, 2]
    alpha1_data = a_hist[:-1, 2]  # Extract theta1 acceleration

    # Sample every 10 points to get dt=0.01 from dt=0.001
    sampling_interval = 10
    s_data_np = np.column_stack((theta1_data, omega1_data))
    t_data_np = np.arange(len(theta1_data)) * dt
    a_data_np = alpha1_data  # Keep acceleration data

    # Sample data
    s_data_sampled = s_data_np[::sampling_interval]
    t_data_sampled = t_data_np[::sampling_interval]
    a_data_sampled = a_data_np[::sampling_interval]

    # Convert to torch tensors
    s_tensor = torch.tensor(s_data_sampled, dtype=torch.float32)
    t_tensor = torch.tensor(t_data_sampled, dtype=torch.float32)
    a_tensor = torch.tensor(a_data_sampled, dtype=torch.float32)

    return s_tensor, t_tensor, a_tensor


def generate_slider_crank_dataset(
        total_num_steps, train_num_steps, dt=1e-3,
        root_dir='.', seed=42
):

    """
    Double-Pendulum  -  Augmented-Lagrangian + Barzilai-Borwein
    ==========================================================
    * Planar mechanism: 3 bodies (crank, rod, slider)
    * DOF ordering  q = [x1, y1, th1,  x2, y2, th2,  x3, y3, th3]
    * 8 scalar holonomic constraints
        C1 (2) crank-ground pin
        C2 (2) crank-rod pin
        C3 (2) rod-slider pin
        C4 (2) slider guide  (y₃ = 0, θ₃ = 0)
    * Implicit **Backward Euler** for both velocities and positions.
    * Inner solver: Barzilai-Borwein; outer: Augmented Lagrangian.

    The script now includes **all non-inertial loads** found in the LaTeX write-up:
        ▸ driving motor torque   τ₁(t)
        ▸ non-smooth rotational damper  c₁₂‖ω₁−ω₂‖^γ sgn(ω₁−ω₂)
        ▸ slider Coulomb/viscous mix    c|ẋ₃|^ψ sgn(ẋ₃)
        ▸ nonlinear slider spring       K(ℓᵐᵃˣ − x₃)^δ  (active in compression)
    """
    # -------------------------------------------------------------------
    #  Physical parameters
    # -------------------------------------------------------------------
    # Masses [kg] and out‑of-plane inertias [kg·m²]
    m1, m2, m3 = 1.0, 1.0, 1.0
    J1, J2, J3 = 0.10, 0.10, 0.10

    # Half‑lengths  (full lengths are 2·L₁, 2·L₂) [m]
    L1, L2 = 1.0, 2.0

    # Gravity
    g = 9.81  # [m/s²]

    # Motor torque (user function)
    tau1 = lambda t: -0.002 * np.sin(2 * np.pi * t)  # [N·m]

    # Rotational damper between crank & rod
    c12 = 0.05  # [N·m·(rad/s)^‑γ]
    gamma = 1.0

    # Slider friction + nonlinear spring
    c_slide = 0.00  # [N·(m/s)^‑ψ]
    psi = 1.0
    K_spring = 1.0  # [N/m^δ]
    delta = 1.0
    x3_max = 2.5 * L1 + 2.0 * L2  # reference slider travel limit

    # -------------------------------------------------------------------
    #  Time‑stepping parameters
    # -------------------------------------------------------------------
    h = 1.0e-3  # step size [s]
    N_steps = total_num_steps

    n_gen = 9  # generalized coordinates
    n_constr = 8  # constraints

    M = np.diag([m1, m1, J1, m2, m2, J2, m3, m3, J3])

    # -------------------------------------------------------------------
    #  Constraint functions
    # -------------------------------------------------------------------

    def constraint(q: np.ndarray) -> np.ndarray:
        """
        c(q) = 0  (size 8).
        Here each value represents the violation of a specific
        holomonic constraint. The goal is to keep these values to be zero
        """
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        return np.array([
            x1 - L1 * np.cos(th1),  # C1‑x
            y1 - L1 * np.sin(th1),  # C1‑y
            x1 + L1 * np.cos(th1) - x2 + L2 * np.cos(th2),  # C2‑x
            y1 + L1 * np.sin(th1) - y2 + L2 * np.sin(th2),  # C2‑y
            x2 + L2 * np.cos(th2) - x3,  # C3‑x
            y2 + L2 * np.sin(th2) - y3,  # C3‑y
            y3,  # C4‑y
            th3  # C4‑θ
        ])

    def constraint_jacobian(q: np.ndarray) -> np.ndarray:
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        J = np.zeros((8, 9))

        # C1
        J[0, 0] = 1.0
        J[0, 2] = L1 * np.sin(th1)
        J[1, 1] = 1.0
        J[1, 2] = -L1 * np.cos(th1)

        # C2
        J[2, 0] = 1.0
        J[2, 3] = -1.0
        J[2, 2] = -L1 * np.sin(th1)
        J[2, 5] = -L2 * np.sin(th2)
        J[3, 1] = 1.0
        J[3, 4] = -1.0
        J[3, 2] = L1 * np.cos(th1)
        J[3, 5] = L2 * np.cos(th2)

        # C3
        J[4, 3] = 1.0
        J[4, 6] = -1.0
        J[4, 5] = -L2 * np.sin(th2)
        J[5, 4] = 1.0
        J[5, 7] = -1.0
        J[5, 5] = L2 * np.cos(th2)

        # C4
        J[6, 7] = 1.0
        J[7, 8] = 1.0

        return J

    # -------------------------------------------------------------------
    #  Non‑inertial loads  f_ext(q, v, t) - MODIFIED FOR COULOMB FRICTION
    # -------------------------------------------------------------------

    def external_forces(q: np.ndarray, v: np.ndarray, lambda_val: float, t: float) -> np.ndarray:
        """Generalised loads (size 9) including gravity, drive, damper, spring."""
        x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
        vx1, vy1, w1, vx2, vy2, w2, vx3, vy3, w3 = v

        f = np.zeros(n_gen)
        # gravity
        f[1] = -m1 * g
        f[4] = -m2 * g
        f[7] = -m3 * g
        # motor torque + crank-rod damper (body-1 DOF θ₁)
        rel_w = w1 - w2
        damper = c12 * np.abs(rel_w) ** gamma * np.sign(rel_w)
        f[2] = tau1(t) - damper  # τ₁ - c₁₂ϕ̇

        # damper reaction on body-2 DOF θ₂
        f[5] = -damper  # ‑c₁₂ϕ̇  (opposite sign)

        # slider friction + spring (x‑direction of body‑3)
        # NOW USING COULOMB FRICTION MODEL
        friction = c_slide * abs(lambda_val) * np.sign(vx3)
        spring = 0.0
        comp = x3_max - x3  # only active when positive (compression)
        if comp > 0.0:
            spring = K_spring * comp ** delta

        f[6] = -(friction + spring)

        return f

    # -------------------------------------------------------------------
    #  ALM + BB velocity solve - MODIFIED FOR COULOMB FRICTION
    # -------------------------------------------------------------------

    def bb_alm_step(v_guess: np.ndarray, lam_guess: np.ndarray,
                    v_prev: np.ndarray, q_prev: np.ndarray, t_next: float):
        v = v_guess.copy()
        lam = lam_guess.copy()

        rho = 1.0e10  # slightly milder than 1e10 for stability
        max_inner = 12
        max_outer = 25
        tol = 1.0e-6
        local_tol = 1.0e-1
        alpha = 1.0e-3
        use_bb1 = True
        normal_force = lam[6]  # Get normal force from constraint

        def grad_L(v_loc: np.ndarray, lambda_val: float) -> np.ndarray:
            g_term = (M @ (v_loc - v_prev)) / h
            qA = q_prev + h * v_loc
            ext = external_forces(qA, v_loc, lambda_val, t_next)  # Pass lambda_val
            c_val = constraint(qA)
            J_val = constraint_jacobian(qA)
            return g_term - ext + J_val.T @ (lam + rho * h * c_val)

        for outer_iter in range(max_outer):
            local_tol = max(local_tol * 0.5, tol)
            vk, gk = v.copy(), grad_L(v, normal_force)  # Pass normal_force

            for inner_iter in range(max_inner):
                vk1 = vk - alpha * gk
                gk1 = grad_L(vk1, lam[7])  # Use lam[7] as in original code
                norm_gk1 = np.linalg.norm(gk1)
                # print inner loop statistics
                # print(f"inner {inner_iter}, norm(gk1)={norm_gk1:.2e}")
                if norm_gk1 < local_tol:
                    vk, gk = vk1, gk1;
                    break
                s, y = vk1 - vk, gk1 - gk
                if use_bb1:
                    alpha = np.dot(s, s) / (np.dot(s, y) + 1e-12)
                else:
                    alpha = np.dot(s, y) / (np.dot(y, y) + 1e-12)
                use_bb1 = not use_bb1
                vk, gk = vk1, gk1

            v = vk
            qA = q_prev + h * v
            c_val = constraint(qA)
            lam += rho * h * c_val
            normal_force = lam[6]  # Update normal force
            # print outer loop statistics
            # print(f">>>>> End of  OUTER STEP #{outer_iter}; norm(constr_violation)={np.linalg.norm(c_val):.2e}")

            if np.linalg.norm(c_val) < tol:
                break

        return v, lam

    # -------------------------------------------------------------------
    #  Initial configuration: straight‑line pose
    # -------------------------------------------------------------------
    theta1_0 = theta2_0 = theta3_0 = 0.0
    x1_0, y1_0 = L1, 0.0
    x2_0, y2_0 = 2 * L1 + L2, 0.0
    x3_0, y3_0 = 2 * L1 + 2 * L2, 0.0

    q = np.array([x1_0, y1_0, theta1_0,
                  x2_0, y2_0, theta2_0,
                  x3_0, y3_0, theta3_0])

    v = np.zeros(n_gen)
    lam = np.zeros(n_constr)
    v_guess = v.copy()

    # -------------------------------------------------------------------
    #  History arrays
    # -------------------------------------------------------------------
    q_hist = np.zeros((N_steps + 1, n_gen))
    v_hist = np.zeros_like(q_hist)
    q_hist[0] = q;
    v_hist[0] = v

    # -------------------------------------------------------------------
    #  Time integration loop
    # -------------------------------------------------------------------
    print("\n=== Slider-Crank ALM + BB (with damper & spring) ===")
    for k in range(N_steps):
        t_next = (k + 1) * h
        v, lam = bb_alm_step(v_guess, lam, v_prev=v, q_prev=q, t_next=t_next)

        q += h * v  # position update
        a = (v - v_hist[k]) / h  # accel estimate

        q_hist[k + 1] = q
        v_hist[k + 1] = v

        v_guess = v + h * a  # Gustafson predictor

        # if k % 1 == 0:
        #     print(f"step {k:5d}  t={t_next: .3f}  |c|={np.linalg.norm(constraint(q)):.2e}")

    # -------------------------------------------------------------------
    #  Diagnostics plots
    # -------------------------------------------------------------------
    plt.figure()
    plt.title("Crank time history")
    plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 2], label="θ₁ [rad]")
    plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 5], label="θ₂ [rad]")
    plt.xlabel("time [s]")
    plt.ylabel("θ₁ [rad], θ₂ [rad]")
    plt.legend();
    plt.grid(True)

    plt.figure()
    plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 6], label="x₃(t)")
    plt.xlabel("time [s]")
    plt.ylabel("slider position x₃ [m]")
    plt.legend();
    plt.grid(True)
    plt.close()

    # -------------------------------------------------------------------
    #  Split and save data (using logic similar to user's original file)
    # -------------------------------------------------------------------
    test_case = "Slider_Crank"
    output_dir = os.path.join(root_dir, 'dataset', test_case)
    figure_dir = os.path.join(root_dir, 'figures', test_case)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    import logging

    local_logger = logging.getLogger(f"{__name__}.generate_slider_crank_dataset_repro")

    if not (0 < train_num_steps < total_num_steps):
        local_logger.warning(
            f"train_num_steps ({train_num_steps}) is not valid for total_num_steps ({total_num_steps}). Adjusting."
        )
        train_num_steps = max(1, min(total_num_steps - 1, int(0.75 * total_num_steps)))
        if train_num_steps == total_num_steps: train_num_steps = total_num_steps - 1  # ensure test set has at least one point
        if train_num_steps <= 0: train_num_steps = 1

    theta1_data = q_hist[:-1, 2]  # theta1 is the 3rd element (index 2), exclude last point to get exactly N_steps points
    omega1_data = v_hist[:-1, 2]  # omega1 is the 3rd element (index 2), exclude last point to get exactly N_steps points

    sampling_interval = 10

    #theta1_wrapped = np.mod(theta1_data, 2 * np.pi)
    s_data_np = np.column_stack((theta1_data, omega1_data))
    t_data_np = np.arange(len(theta1_data)) * dt  # Time vector matching the length of theta1_data

    #sampling dataset
    s_data_np = s_data_np[::sampling_interval]
    t_data_np = t_data_np[::sampling_interval]
    train_num_steps = int(train_num_steps/sampling_interval)

    s_train_np = s_data_np[:train_num_steps]
    s_test_np = s_data_np[train_num_steps:]
    t_train_np = t_data_np[:train_num_steps]
    t_test_np = t_data_np[train_num_steps:]
    test_num_steps_actual = len(s_test_np)

    file_float_fmt = '%.8g'
    seed_suffix = f"_seed{seed}"
    s_columns = ['theta_0_2pi', 'omega']

    try:
        pd.DataFrame(s_data_np, columns=s_columns).to_csv(
            os.path.join(output_dir, f"s_full.csv"), index=False, float_format=file_float_fmt)
        pd.DataFrame(t_data_np, columns=['time']).to_csv(
            os.path.join(output_dir, f"t_full.csv"), index=False, float_format=file_float_fmt)
        pd.DataFrame(s_train_np, columns=s_columns).to_csv(
            os.path.join(output_dir, f"s_train.csv"), index=False, float_format=file_float_fmt)
        if test_num_steps_actual > 0:
            pd.DataFrame(s_test_np, columns=s_columns).to_csv(
                os.path.join(output_dir, f"s_test.csv"), index=False, float_format=file_float_fmt)
        pd.DataFrame(t_train_np, columns=['time']).to_csv(
            os.path.join(output_dir, f"t_train.csv"), index=False, float_format=file_float_fmt)
        if test_num_steps_actual > 0:
            pd.DataFrame(t_test_np, columns=['time']).to_csv(
                os.path.join(output_dir, f"t_test.csv"), index=False, float_format=file_float_fmt)

        # Dummy 'u' files
        pd.DataFrame(np.zeros((total_num_steps, 1)), columns=['u']).to_csv(
            os.path.join(output_dir, f"u_full.csv"), index=False)
        pd.DataFrame(np.zeros((train_num_steps, 1)), columns=['u']).to_csv(
            os.path.join(output_dir, f"u_train.csv"), index=False)
        # Ensure u_test is created correctly based on test_num_steps_actual
        pd.DataFrame(np.zeros((test_num_steps_actual, 1)), columns=['u']).to_csv(
            os.path.join(output_dir, f"u_test.csv"), index=False)

        pd.DataFrame(s_train_np, columns=s_columns).to_csv(
            os.path.join(output_dir, f"s_valid.csv"), index=False, float_format=file_float_fmt)
        pd.DataFrame(np.zeros((train_num_steps, 1)), columns=['u']).to_csv(
            os.path.join(output_dir, f"u_valid.csv"), index=False)
        local_logger.info(f"Saved Slider-Crank dataset to {output_dir}")
    except Exception as e:
        local_logger.error(f"Error saving dataset files: {e}")
        raise

    # Convert to torch tensor and return (matching other test case formats)
    # Reshape s_data_np to [num_steps, num_bodies, 2] format: [timesteps, 1 body, [theta, omega]]
    s_tensor = torch.tensor(s_data_np, dtype=torch.float32).unsqueeze(1)  # [timesteps, 1, 2]
    t_tensor = torch.tensor(t_data_np, dtype=torch.float32)

    return s_tensor, t_tensor


def generate_dataset_multi(test_case, numerical_methods, dt, num_steps,
                           output_root_dir='.', if_noise=False, seed=42, return_splits=False, **kwargs):
    """
    Extension of generate_dataset that returns both train and test splits.
    Used for generating multiple trajectories with different initial conditions.

    Args:
        test_case (str): Name of the system (e.g., 'Double_Pendulum', 'Slider_Crank').
        numerical_methods (str): Method for generating data ('rk4', 'fe', 'analytical').
        dt (float): Time step for simulation.
        num_steps (int): Total number of time steps to generate.
        output_root_dir (str): Root directory for saving 'dataset' and 'figures'.
        if_noise (bool): If True, adds Gaussian noise to the generated data.
        seed (int): Random seed for reproducibility.
        return_splits (bool): If True, returns train and test splits separately.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: (s_train_np, s_test_np, t_train_np, t_test_np) if return_splits=True
        or torch.Tensor: The generated trajectory data
    """
    # Note: Do NOT call set_seed here - main script already called it
    # Calling it again resets random state and causes model init mismatch with MNODE
    # set_seed(seed)  # Commented out to match MNODE-code behavior
    logger.info(f"--- Generating Dataset (Multi-Split Mode) ---")
    logger.info(f"Test Case: {test_case}")
    logger.info(f"Method: {numerical_methods}, dt: {dt}, Steps: {num_steps}")

    # Create standard dataset directory structure
    dataset_dir = os.path.join(output_root_dir, 'dataset', test_case)
    figures_dir = os.path.join(output_root_dir, 'figures', test_case)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Special handling for Double_Pendulum to vary initial conditions but keep zero velocities
    if test_case == "Double_Pendulum":
        # Set the seed for this specific generation
        np.random.seed(seed)

        # Use specific variation ranges for theta1 and theta2
        # For physical constraints, both angles should typically be within [-pi, pi]
        theta1_range = 0.8  # Limit variation range for theta1
        theta2_range = 0.8  # Limit variation range for theta2

        # Set theta1 to vary around pi/2 (standard pendulum initial position)
        theta1_init = np.pi / 2 + np.random.uniform(-theta1_range, theta1_range)
        theta1_init = np.clip(theta1_init, -np.pi, np.pi)  # Ensure within physical limits

        # Set theta2 to vary around 0 (hanging straight initially)
        theta2_init = np.random.uniform(-theta2_range, theta2_range)
        theta2_init = np.clip(theta2_init, -np.pi, np.pi)  # Ensure within physical limits

        # Set both angular velocities to zero as required
        omega1_init = 0.0
        omega2_init = 0.0

        initial_state_np = np.array([[theta1_init, omega1_init], [theta2_init, omega2_init]])
        logger.info(f"Generated initial state for Double_Pendulum with seed {seed}: {initial_state_np}")
        logger.info(f"Initial angular velocities set to zero: omega1={omega1_init}, omega2={omega2_init}")

        # Use scipy solver
        if not SCIPY_AVAILABLE:
            logger.error("SciPy is required to generate data for Double_Pendulum using solve_ivp.")
            return None

        logger.info("Using scipy.integrate.solve_ivp for Double_Pendulum (method='RK45')")
        t_eval = np.linspace(0, (num_steps - 1) * dt, num_steps)
        sol = scipy.integrate.solve_ivp(double_pendulum_derivs,
                                        [0, t_eval[-1]],
                                        initial_state_np.flatten(),
                                        t_eval=t_eval, rtol=1e-9, atol=1e-9, method='RK45')

        if not sol.success:
            logger.error(f"SciPy solver failed for Double_Pendulum: {sol.message}")
            return None

        y_result = sol.y.T  # Shape [steps, state_dim]
        # Reshape to [steps, bodies, 2]
        num_bodies = 2  # Double pendulum has 2 bodies
        training_set = torch.tensor(y_result, dtype=torch.float32, device=device).view(num_steps, num_bodies, 2)

    # Special handling for Slider_Crank to vary initial conditions
    elif test_case == "Slider_Crank":
        # Set the seed for this specific generation
        np.random.seed(seed)

        # Get generation parameters from kwargs or use defaults
        r_crank = kwargs.get('gen_r_crank', 1.0)
        l_rod = kwargs.get('gen_l_rod', 4.0)

        # Define variation ranges for initial conditions
        theta_range = 0.6 * np.pi  # Allow significant variation in initial angle
        omega_base = kwargs.get('gen_omega0', 1.0 * np.pi)  # Base angular velocity
        omega_variation = 0.5  # Fractional variation

        # Set initial theta to vary (can be anywhere in the full rotation)
        theta_init = np.random.uniform(0, 2 * np.pi)

        # Set initial omega to vary around base value
        omega_init = omega_base * (1.0 + np.random.uniform(-omega_variation, omega_variation))

        # Use specialized generator for Slider_Crank
        slider_crank_args = {
            'total_num_steps': num_steps,
            'train_num_steps': kwargs.get('gen_train_num_steps', int(0.75 * num_steps)),
            'dt': dt,
            'root_dir': output_root_dir,
            'seed': seed,
            # Pass specific generation args with varied initial conditions
            'r_crank': r_crank,
            'l_rod': l_rod,
            'theta_0_sim': theta_init,  # Use varied initial theta
            'omega_initial_sim': omega_init,  # Use varied initial omega
            'generate_forces': kwargs.get('generate_forces', True),
            'force_amplitude': kwargs.get('force_amp', 0.5),
            'torque_amplitude': kwargs.get('torque_amp', 0.5),
            'force_frequency': kwargs.get('force_freq', 0.25),
            'torque_frequency': kwargs.get('torque_freq', 0.5),
        }

        logger.info(
            f"Generated initial state for Slider_Crank with seed {seed}: theta={theta_init:.4f}, omega={omega_init:.4f}")

        # Call the specialized generator
        generated_data_tensor, time_tensor = generate_slider_crank_dataset(**slider_crank_args)
        training_set = generated_data_tensor

    else:
        # For other test cases, just call the original function
        training_set = generate_dataset(test_case, numerical_methods, dt, num_steps,
                                        output_root_dir, if_noise, seed, **kwargs)

    # If dataset generation failed
    if training_set is None:
        return None if not return_splits else (None, None, None, None)

    # If we need to return splits separately
    if return_splits:
        # Reshape state to [steps, features] for splitting
        s_full_np = training_set.detach().cpu().numpy()
        if s_full_np.ndim == 3:
            s_full_np = s_full_np.reshape(num_steps, -1)

        t_full_np = np.linspace(0, (num_steps - 1) * dt, num_steps)

        # Determine split point
        train_split_idx = kwargs.get('gen_train_num_steps', int(0.75 * num_steps))
        if train_split_idx >= num_steps:
            train_split_idx = num_steps - 1

        # Create splits
        s_train_np = s_full_np[:train_split_idx]
        t_train_np = t_full_np[:train_split_idx]
        s_test_np = s_full_np[train_split_idx:]
        t_test_np = t_full_np[train_split_idx:]

        # Also save to files for consistency
        file_float_fmt = '%.8g'
        output_dir_std = os.path.join(dataset_dir)
        os.makedirs(output_dir_std, exist_ok=True)

        try:
            # Save with unique prefix based on seed
            seed_prefix = f"seed{seed}_"
            pd.DataFrame(s_full_np).to_csv(os.path.join(output_dir_std, f"{seed_prefix}s_full.csv"),
                                           index=False, float_format=file_float_fmt)
            pd.DataFrame(t_full_np, columns=['time']).to_csv(os.path.join(output_dir_std, f"{seed_prefix}t_full.csv"),
                                                             index=False, float_format=file_float_fmt)
            pd.DataFrame(s_train_np).to_csv(os.path.join(output_dir_std, f"{seed_prefix}s_train.csv"),
                                            index=False, float_format=file_float_fmt)
            pd.DataFrame(t_train_np, columns=['time']).to_csv(os.path.join(output_dir_std, f"{seed_prefix}t_train.csv"),
                                                              index=False, float_format=file_float_fmt)
            pd.DataFrame(s_test_np).to_csv(os.path.join(output_dir_std, f"{seed_prefix}s_test.csv"),
                                           index=False, float_format=file_float_fmt)
            pd.DataFrame(t_test_np, columns=['time']).to_csv(os.path.join(output_dir_std, f"{seed_prefix}t_test.csv"),
                                                             index=False, float_format=file_float_fmt)
        except Exception as e:
            logger.error(f"Error saving dataset CSV files: {e}")

        logger.info(f"Dataset for seed {seed} generated and split into train/test.")
        return s_train_np, s_test_np, t_train_np, t_test_np

    return training_set


def generate_dataset(test_case, numerical_methods, dt, num_steps,
                     output_root_dir='.', if_noise=False, seed=42, **kwargs):
    """
    Generates datasets for various test cases and saves them to disk with consistent organization.

    Args:
        test_case (str): Name of the system (e.g., 'Single_Mass_Spring', 'Cartpole').
        numerical_methods (str): Method for generating data ('rk4', 'fe', 'analytical', 'midpoint').
        dt (float): Time step for simulation.
        num_steps (int): Total number of time steps to generate.
        output_root_dir (str): Root directory for saving 'dataset' and 'figures'.
        if_noise (bool): If True, adds Gaussian noise to the generated data.
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
        num_bodies = 1

    elif test_case == "Double_Pendulum":
        logger.info("Setting up Double Pendulum system")
        # Fixed initial conditions matching MNODE-code
        theta1_init = np.pi / 2  # 90 degrees (2/4 * 180)
        theta2_init = np.pi / 4  # 45 degrees (1/4 * 180)
        omega1_init = 0.0
        omega2_init = 0.0

        initial_state_np = np.array([[theta1_init, omega1_init], [theta2_init, omega2_init]])
        logger.info(f"Using fixed initial conditions: theta1={np.degrees(theta1_init):.1f}°, theta2={np.degrees(theta2_init):.1f}°")
        use_scipy = True
        num_bodies = 2
        # Create dummy bodys_tensor to match MNODE-code GPU tensor operations sequence
        bodys_dummy = np.array([[np.pi/2, 0, 0, 0]])
        bodys_tensor_dummy = torch.tensor(bodys_dummy, dtype=torch.float32, device=device)

    elif test_case == "Cart_Pole":
        logger.info("Setting up Cartpole system")
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

    else:
        raise ValueError(f"Test case '{test_case}' not recognized.")

    # =========================================================================
    # Generate trajectory using selected method
    # =========================================================================

    try:
        if use_scipy and test_case in ["Single_Pendulum", "Double_Pendulum"]:
            # Use scipy.integrate.solve_ivp for pendulum systems
            if not SCIPY_AVAILABLE:
                logger.error(f"SciPy is required to generate data for {test_case} using solve_ivp.")
                return None

            logger.info(f"Using scipy.integrate.solve_ivp for {test_case} (method='RK45')")

            # Select appropriate derivative function
            if test_case == "Single_Pendulum":
                deriv_func = single_pendulum_derivs
            elif test_case == "Double_Pendulum":
                deriv_func = double_pendulum_derivs
            else:
                raise ValueError(f"No derivative function defined for {test_case}")

            # Generate time vector (matching MNODE-code)
            t_stop = dt * num_steps
            t_eval = np.linspace(0, t_stop, num_steps)

            # Solve using scipy
            sol = scipy.integrate.solve_ivp(
                deriv_func,
                [0, t_eval[-1]],
                initial_state_np.flatten(),
                t_eval=t_eval,
                rtol=1e-10,
                atol=1e-10,
                method='RK45'
            )

            if not sol.success:
                logger.error(f"SciPy solver failed for {test_case}: {sol.message}")
                return None

            # Reshape result to [steps, bodies, 2]
            y_result = sol.y.T  # Shape [steps, state_dim]
            # Match MNODE-code's tensor creation pattern for consistent CUDA random state
            training_set = torch.zeros((num_steps, num_bodies, 2), dtype=torch.float32, device=device)
            training_set[:,0,:] = torch.tensor(y_result[:,0:2], dtype=torch.float32, device=device)
            training_set[:,1,:] = torch.tensor(y_result[:,2:4], dtype=torch.float32, device=device)

        elif numerical_methods == "analytical" and test_case == "Single_Mass_Spring":
            # Use analytical solution for single mass spring
            logger.info(f"Using analytical solution for {test_case}")
            bodys_tensor = torch.tensor(initial_state_np, dtype=torch.float32, device=device)
            training_set = torch.zeros((num_steps, num_bodies, 2), dtype=torch.float32, device=device)
            training_set[0, :, :] = bodys_tensor

            for i in range(num_steps - 1):
                training_set[i + 1, :, :] = analytic_sms(training_set[i, :, :], dt)

        elif numerical_methods in ["fe", "rk4", "midpoint"]:
            # Use numerical integrators
            logger.info(f"Using numerical integrator '{numerical_methods}' for {test_case}")
            bodys_tensor = torch.tensor(initial_state_np, dtype=torch.float32, device=device)

            # Select appropriate integrator
            if numerical_methods == "fe":
                integrator_func = forward_euler_multiple_body
            elif numerical_methods == "rk4":
                integrator_func = runge_kutta_four_multiple_body
            elif numerical_methods == "midpoint":
                integrator_func = midpoint_method_multiple_body
            else:
                raise ValueError(f"Unknown numerical method: {numerical_methods}")

            # Generate trajectory
            training_set = integrator_func(
                bodys_tensor,
                force_func,
                num_steps,
                dt,
                if_final_state=False,
                model=kwargs.get('model')
            )

        else:
            raise ValueError(f"Unsupported numerical_method '{numerical_methods}' for {test_case}")

    except Exception as e:
        logger.error(f"Error during trajectory generation for {test_case}: {e}")
        return None

    # =========================================================================
    # Add noise if requested
    # =========================================================================

    if if_noise and training_set is not None:
        noise_levels = {
            "Single_Mass_Spring": 0.01,
            "Single_Mass_Spring_Damper": 0.003,
            "Triple_Mass_Spring_Damper": 0.003,
            "Single_Pendulum": 0.001,
            "Double_Pendulum": 0.001,
            "Cartpole": 0.001
        }

        noise_level = noise_levels.get(test_case, 0.01)
        training_set += torch.randn_like(training_set) * noise_level
        logger.info(f"Added Gaussian noise with std dev {noise_level}")

    # =========================================================================
    # Save generated dataset to files
    # =========================================================================

    if training_set is not None:
        try:
            # Reshape state to [steps, features] for saving
            s_full_np = training_set.detach().cpu().numpy().reshape(num_steps, -1)
            t_full_np = np.linspace(0, dt * num_steps, num_steps)

            # Determine train/test split
            train_split_idx = kwargs.get('gen_train_num_steps', int(0.75 * num_steps))
            if train_split_idx >= num_steps:
                train_split_idx = num_steps - 1
            test_split_idx = train_split_idx

            # Create data splits
            s_train_np = s_full_np[:train_split_idx]
            t_train_np = t_full_np[:train_split_idx]
            s_test_np = s_full_np[test_split_idx:]
            t_test_np = t_full_np[test_split_idx:]

            # Define output directory
            output_dir_std = os.path.join(dataset_dir)
            os.makedirs(output_dir_std, exist_ok=True)

            # Save format settings
            file_float_fmt = '%.8g'

            # Save all data files
            data_files = {
                "s_full.csv": s_full_np,
                "s_train.csv": s_train_np,
                "s_test.csv": s_test_np,
                "s_valid.csv": s_train_np,  # Use train data for validation
            }

            time_files = {
                "t_full.csv": t_full_np,
                "t_train.csv": t_train_np,
                "t_test.csv": t_test_np,
            }

            # Save state data
            for filename, data in data_files.items():
                pd.DataFrame(data).to_csv(
                    os.path.join(output_dir_std, filename),
                    index=False,
                    float_format=file_float_fmt
                )

            # Save time data
            for filename, data in time_files.items():
                pd.DataFrame(data, columns=['time']).to_csv(
                    os.path.join(output_dir_std, filename),
                    index=False,
                    float_format=file_float_fmt
                )

            # Save dummy control input files (u) for compatibility
            control_files = {
                "u_train.csv": len(s_train_np),
                "u_test.csv": len(s_test_np),
                "u_valid.csv": len(s_train_np),
            }

            for filename, length in control_files.items():
                pd.DataFrame(np.zeros((length, 1)), columns=['u']).to_csv(
                    os.path.join(output_dir_std, filename),
                    index=False
                )

            logger.info(f"Dataset for {test_case} saved to {output_dir_std}")
            logger.info(f"Data shape: {s_full_np.shape}, Time range: {t_full_np[0]:.3f} to {t_full_np[-1]:.3f}")
            logger.info(f"Train steps: {len(s_train_np)}, Test steps: {len(s_test_np)}")

        except Exception as e:
            logger.error(f"Error saving dataset files for {test_case}: {e}")
            return None

    logger.info(f"--- Dataset Generation Finished for {test_case} ---")
    return training_set


# === CONTROLLED SYSTEMS DATA GENERATION ===

def generate_cartpole_data(num_steps=100000, seed=42, save_to_dataset=True, dataset_dir='dataset/Cart_Pole_Controlled'):
    """
    Generate cart-pole data using MBDNODE-for-MBD2 approach.
    
    Args:
        num_steps: Number of data samples to generate
        seed: Random seed for reproducibility
        save_to_dataset: Whether to save the generated data to disk
        dataset_dir: Directory to save the dataset
        
    Returns:
        body_tensor: [num_steps, 2, 2] tensor with states [[x, x_dot], [theta, theta_dot]]
        force_tensor: [num_steps, 1] tensor with control forces
        accel_tensor: [num_steps, 2] tensor with accelerations [x_ddot, theta_ddot]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Generating {num_steps} cart-pole samples")

    # Bounds from MBDNODE-for-MBD2 project
    x_min, x_max = -1.5, 1.5
    x_dot_min, x_dot_max = -4, 4
    theta_min, theta_max = 0, 2 * np.pi
    theta_dot_min, theta_dot_max = -8, 8
    F_min, F_max = -15, 30

    # Generate random samples - MBDNODE format: [[x, x_dot], [theta, theta_dot]]
    bodys = np.random.uniform([[x_min, x_dot_min], [theta_min, theta_dot_min]],
                              [[x_max, x_dot_max], [theta_max, theta_dot_max]],
                              (num_steps, 2, 2))
    u = np.random.uniform(F_min, F_max, (num_steps, 1))

    # Convert to tensors
    bodys_tensor = torch.tensor(bodys, dtype=torch.float32)
    u_tensor = torch.tensor(u, dtype=torch.float32)

    # Import force function
    from Model.force_fun import force_cp_ext_sequence_fnode
    
    # Compute accelerations using force function
    accel_tensor = force_cp_ext_sequence_fnode(bodys_tensor, u_tensor)

    logger.info(f"Data shapes: bodys={bodys_tensor.shape}, u={u_tensor.shape}, accel={accel_tensor.shape}")
    logger.info(f"Bounds: x=[{x_min},{x_max}], theta=[{theta_min:.1f},{theta_max:.1f}], F=[{F_min},{F_max}]")

    # Save to dataset directory if requested
    if save_to_dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f'controlled_dataset_seed{seed}_n{num_steps}.npz')
        
        # Convert tensors to numpy for saving
        np.savez(dataset_path,
                 bodys=bodys_tensor.cpu().numpy(),
                 forces=u_tensor.cpu().numpy(),
                 accelerations=accel_tensor.cpu().numpy(),
                 seed=seed,
                 num_steps=num_steps,
                 bounds={'x_min': x_min, 'x_max': x_max,
                        'x_dot_min': x_dot_min, 'x_dot_max': x_dot_max,
                        'theta_min': theta_min, 'theta_max': theta_max,
                        'theta_dot_min': theta_dot_min, 'theta_dot_max': theta_dot_max,
                        'F_min': F_min, 'F_max': F_max})
        
        logger.info(f"Saved dataset to {dataset_path}")

    return bodys_tensor, u_tensor, accel_tensor


def load_cartpole_data(dataset_path):
    """
    Load cart-pole dataset from NPZ file.
    
    Args:
        dataset_path: Path to the NPZ file
        
    Returns:
        body_tensor: [num_steps, 2, 2] tensor with states [[x, x_dot], [theta, theta_dot]]
        force_tensor: [num_steps, 1] tensor with control forces
        accel_tensor: [num_steps, 2] tensor with accelerations [x_ddot, theta_ddot]
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
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
    if 'num_steps' in data:
        logger.info(f"Dataset size: {data['num_steps']} samples")
    if 'bounds' in data:
        bounds = data['bounds'].item()  # Convert 0-d array to dict
        logger.info(f"Dataset bounds: {bounds}")
    
    logger.info(f"Loaded data shapes: bodys={bodys_tensor.shape}, forces={force_tensor.shape}, accel={accel_tensor.shape}")
    
    return bodys_tensor, force_tensor, accel_tensor


def data_generator_cp_d(num_steps=100000, seed=42, save_to_dataset=True, dataset_dir='dataset/Cart_Pole_D_Controlled'):
    """
    Generate cart double pendulum data using uniform random sampling.
    
    Args:
        num_steps: Number of data samples to generate
        seed: Random seed for reproducibility
        save_to_dataset: Whether to save the generated data to disk
        dataset_dir: Directory to save the dataset
        
    Returns:
        states_tensor: [num_steps, 6] tensor with states [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        force_tensor: [num_steps, 1] tensor with control forces
        accel_tensor: [num_steps, 3] tensor with accelerations [x_ddot, theta1_ddot, theta2_ddot]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Generating {num_steps} cart double pendulum samples with uniform random sampling")
    
    # State bounds - using physical constraints for inverted pendulum
    x_min, x_max = -2.0, 2.0
    x_dot_min, x_dot_max = -4.0, 4.0
    theta1_min, theta1_max = -np.pi/2, np.pi/2  # ±90 degrees
    theta1_dot_min, theta1_dot_max = -6.0, 6.0
    theta2_min, theta2_max = -np.pi/2, np.pi/2  # ±90 degrees
    theta2_dot_min, theta2_dot_max = -6.0, 6.0
    F_min, F_max = -50.0, 50.0  # Symmetric control range
    
    # Generate uniform random samples across all state bounds
    x = np.random.uniform(x_min, x_max, (num_steps,))
    x_dot = np.random.uniform(x_dot_min, x_dot_max, (num_steps,))
    theta1 = np.random.uniform(theta1_min, theta1_max, (num_steps,))
    theta1_dot = np.random.uniform(theta1_dot_min, theta1_dot_max, (num_steps,))
    theta2 = np.random.uniform(theta2_min, theta2_max, (num_steps,))
    theta2_dot = np.random.uniform(theta2_dot_min, theta2_dot_max, (num_steps,))
    u = np.random.uniform(F_min, F_max, (num_steps, 1))
    
    # Stack states
    states = np.stack([x, theta1, theta2, x_dot, theta1_dot, theta2_dot], axis=1)
    
    # Import force function
    from Model.force_fun import force_cp_d_con
    
    # Compute accelerations for each sample
    accelerations = np.zeros((num_steps, 3))
    
    for i in range(num_steps):
        # Create state+control array
        state_and_control = np.concatenate([states[i], [u[i, 0]]])
        # Compute acceleration
        accelerations[i] = force_cp_d_con(state_and_control)
    
    # Convert to tensors
    states_tensor = torch.tensor(states, dtype=torch.float32)
    u_tensor = torch.tensor(u, dtype=torch.float32)
    accel_tensor = torch.tensor(accelerations, dtype=torch.float32)
    
    logger.info(f"Data shapes: states={states_tensor.shape}, u={u_tensor.shape}, accel={accel_tensor.shape}")
    logger.info(f"Bounds: x=[{x_min},{x_max}], theta1=[{theta1_min:.2f},{theta1_max:.2f}], theta2=[{theta2_min:.2f},{theta2_max:.2f}], F=[{F_min},{F_max}]")
    logger.info(f"Sampling: Uniform random sampling across all state bounds")
    
    # Save to dataset directory if requested
    if save_to_dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f'controlled_dataset_seed{seed}_n{num_steps}.npz')
        
        # Convert tensors to numpy for saving
        np.savez(dataset_path,
                 states=states_tensor.cpu().numpy(),
                 forces=u_tensor.cpu().numpy(),
                 accelerations=accel_tensor.cpu().numpy(),
                 seed=seed,
                 num_steps=num_steps,
                 bounds={'x_min': x_min, 'x_max': x_max,
                        'x_dot_min': x_dot_min, 'x_dot_max': x_dot_max,
                        'theta1_min': theta1_min, 'theta1_max': theta1_max,
                        'theta1_dot_min': theta1_dot_min, 'theta1_dot_max': theta1_dot_max,
                        'theta2_min': theta2_min, 'theta2_max': theta2_max,
                        'theta2_dot_min': theta2_dot_min, 'theta2_dot_max': theta2_dot_max,
                        'F_min': F_min, 'F_max': F_max})
        
        logger.info(f"Saved dataset to {dataset_path}")
    
    return states_tensor, u_tensor, accel_tensor


def load_cartdoublependulum_data(dataset_path):
    """
    Load cart double pendulum dataset from NPZ file.
    
    Args:
        dataset_path: Path to the NPZ file
        
    Returns:
        states_tensor: [num_steps, 6] tensor with states [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        force_tensor: [num_steps, 1] tensor with control forces
        accel_tensor: [num_steps, 3] tensor with accelerations [x_ddot, theta1_ddot, theta2_ddot]
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load the NPZ file
    data = np.load(dataset_path, allow_pickle=True)
    
    # Convert numpy arrays back to tensors
    states_tensor = torch.tensor(data['states'], dtype=torch.float32)
    force_tensor = torch.tensor(data['forces'], dtype=torch.float32)
    accel_tensor = torch.tensor(data['accelerations'], dtype=torch.float32)
    
    # Log metadata
    if 'seed' in data:
        logger.info(f"Dataset seed: {data['seed']}")
    if 'num_steps' in data:
        logger.info(f"Dataset size: {data['num_steps']} samples")
    if 'bounds' in data:
        bounds = data['bounds'].item()  # Convert 0-d array to dict
        logger.info(f"Dataset bounds: {bounds}")
    
    logger.info(f"Loaded data shapes: states={states_tensor.shape}, forces={force_tensor.shape}, accel={accel_tensor.shape}")
    
    return states_tensor, force_tensor, accel_tensor


def cartdoublependulum_dynamics_controlled(state, u):
    """
    Calculate accelerations for cart double pendulum given state and control.
    
    Args:
        state: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        u: control force applied to cart
        
    Returns:
        accelerations: [x_ddot, theta1_ddot, theta2_ddot]
    """
    # Import the force function
    from Model.force_fun import force_cp_d_con
    
    # Create state+control array
    state_and_control = np.concatenate([state, [u]])
    
    # Calculate accelerations
    return force_cp_d_con(state_and_control)


def data_generator_cp_d_trajectory(num_steps=1000000, seed=42, save_to_dataset=True, dataset_dir="dataset"):
    """
    Generate cart double pendulum data using trajectory-based sampling.
    This approach generates physically coherent trajectories instead of random isolated points.
    
    Args:
        num_steps: Total number of data samples to generate
        seed: Random seed for reproducibility
        save_to_dataset: Whether to save the generated data to disk
        dataset_dir: Directory to save the dataset
        
    Returns:
        states_tensor: [num_steps, 6] tensor with states [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        force_tensor: [num_steps, 1] tensor with control forces
        accel_tensor: [num_steps, 3] tensor with accelerations [x_ddot, theta1_ddot, theta2_ddot]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Generating {num_steps} cart double pendulum samples using trajectory-based approach")

    # System parameters - matching MBDNODE mass assumption
    m_cart = 10.0  # Updated to match MBDNODE
    m1 = 10.0      # Updated to match MBDNODE
    m2 = 10.0      # Updated to match MBDNODE
    l1 = 1.0
    l2 = 1.0
    g = 9.81
    
    # Integration parameters
    dt = 0.01  # Time step for integration
    samples_per_trajectory = 100  # Number of samples per trajectory
    num_trajectories = num_steps // samples_per_trajectory + 1
    
    # Storage for all data
    all_states = []
    all_forces = []
    all_accels = []
    
    # Define trajectory categories with different initial conditions and control strategies
    trajectory_types = {
        'exact_equilibrium': {
            'weight': 0.15,  # 15% of trajectories start from exact equilibrium
            'x_range': (0.0, 0.0),
            'theta_range': (0.0, 0.0),
            'vel_range': (0.0, 0.0),
            'control_type': 'equilibrium_perturbation'
        },
        'equilibrium_dense': {
            'weight': 0.25,  # 25% of trajectories
            'x_range': (-0.05, 0.05),
            'theta_range': (-0.05, 0.05),  # Very close to equilibrium
            'vel_range': (-0.05, 0.05),
            'control_type': 'stabilizing'
        },
        'near_equilibrium': {
            'weight': 0.20,  # 20% of trajectories
            'x_range': (-0.2, 0.2),
            'theta_range': (-0.15, 0.15),  # Small perturbations
            'vel_range': (-0.3, 0.3),
            'control_type': 'stabilizing'
        },
        'medium_motion': {
            'weight': 0.25,  # 25% of trajectories
            'x_range': (-0.5, 0.5),
            'theta_range': (-0.6, 0.6),  # Medium angles
            'vel_range': (-1.5, 1.5),
            'control_type': 'mixed'
        },
        'large_motion': {
            'weight': 0.15,  # 15% of trajectories
            'x_range': (-1.0, 1.0),
            'theta_range': (-1.0, 1.0),  # Large angles
            'vel_range': (-3.0, 3.0),
            'control_type': 'swing_up'
        }
    }
    
    # Calculate number of trajectories for each type
    traj_counts = {}
    total_weight = sum(t['weight'] for t in trajectory_types.values())
    for ttype, config in trajectory_types.items():
        traj_counts[ttype] = int(num_trajectories * config['weight'] / total_weight)
    
    # Generate trajectories for each type
    for ttype, config in trajectory_types.items():
        n_traj = traj_counts[ttype]
        logger.info(f"Generating {n_traj} trajectories of type '{ttype}'")
        
        for _ in range(n_traj):
            # Random initial conditions within specified ranges
            if ttype == 'exact_equilibrium':
                # Start from exact equilibrium
                x0 = 0.0
                theta1_0 = 0.0
                theta2_0 = 0.0
                x_dot0 = 0.0
                theta1_dot0 = 0.0
                theta2_dot0 = 0.0
            else:
                x0 = np.random.uniform(*config['x_range'])
                theta1_0 = np.random.uniform(*config['theta_range'])
                theta2_0 = np.random.uniform(*config['theta_range'])
                x_dot0 = np.random.uniform(*config['vel_range'])
                theta1_dot0 = np.random.uniform(*config['vel_range'])
                theta2_dot0 = np.random.uniform(*config['vel_range'])
            
            # Initial state
            state = np.array([x0, theta1_0, theta2_0, x_dot0, theta1_dot0, theta2_dot0])
            
            # Generate control strategy based on type
            if config['control_type'] == 'equilibrium_perturbation':
                # Small random perturbations from equilibrium
                K = np.array([15.0, 40.0, 30.0, 8.0, 15.0, 12.0])  # Strong feedback gains
                # Add small sinusoidal perturbations
                time_counter = 0
                def equilibrium_control(s):
                    nonlocal time_counter
                    time_counter += dt
                    perturbation = 2.0 * np.sin(2.0 * np.pi * 0.5 * time_counter)  # 0.5 Hz perturbation
                    return -K @ s + perturbation
                base_control = equilibrium_control
                noise_level = 0.5  # Very small noise
            elif config['control_type'] == 'stabilizing':
                # LQR-like control for stabilization
                K = np.array([10.0, 30.0, 20.0, 5.0, 10.0, 8.0])  # Feedback gains
                base_control = lambda s: -K @ s
                noise_level = 2.0
            elif config['control_type'] == 'mixed':
                # Mixed control with some exploration
                K = np.array([5.0, 15.0, 10.0, 2.5, 5.0, 4.0])
                base_control = lambda s: -K @ s + 5.0 * np.sin(0.5 * np.sum(s[:3]))
                noise_level = 5.0
            else:  # swing_up
                # Swing-up control with energy pumping
                def swing_control(s):
                    energy = 0.5 * (s[3]**2 + s[4]**2 + s[5]**2) - g * (l1 * np.cos(s[1]) + l2 * np.cos(s[2]))
                    return 10.0 * np.sign(energy) * s[3] + 5.0 * np.sin(2.0 * s[1])
                base_control = swing_control
                noise_level = 10.0
            
            # Simulate trajectory
            trajectory_states = []
            trajectory_forces = []
            
            for step in range(samples_per_trajectory):
                # Store current state
                trajectory_states.append(state.copy())
                
                # Calculate control force with noise
                u_base = base_control(state)
                u_noise = noise_level * np.random.randn()
                u = np.clip(u_base + u_noise, -50.0, 50.0)
                trajectory_forces.append([u])
                
                # Calculate accelerations using dynamics
                accel = cartdoublependulum_dynamics_controlled(state, u)
                
                # Integrate to next state (RK4)
                k1 = np.concatenate([state[3:], accel])
                k2_state = state + 0.5 * dt * k1
                k2_accel = cartdoublependulum_dynamics_controlled(k2_state, u)
                k2 = np.concatenate([k2_state[3:], k2_accel])
                k3_state = state + 0.5 * dt * k2
                k3_accel = cartdoublependulum_dynamics_controlled(k3_state, u)
                k3 = np.concatenate([k3_state[3:], k3_accel])
                k4_state = state + dt * k3
                k4_accel = cartdoublependulum_dynamics_controlled(k4_state, u)
                k4 = np.concatenate([k4_state[3:], k4_accel])
                
                state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
                # Apply state constraints
                state[0] = np.clip(state[0], -2.0, 2.0)  # Cart position
                state[3] = np.clip(state[3], -4.0, 4.0)  # Cart velocity
                state[4] = np.clip(state[4], -6.0, 6.0)  # Angular velocities
                state[5] = np.clip(state[5], -6.0, 6.0)
            
            # Add trajectory data to collection
            all_states.extend(trajectory_states)
            all_forces.extend(trajectory_forces)
    
    # Trim to exact number of requested samples
    all_states = all_states[:num_steps]
    all_forces = all_forces[:num_steps]
    
    # Convert to tensors
    states_array = np.array(all_states)
    forces_array = np.array(all_forces)
    
    # Calculate accelerations for all states
    logger.info("Computing accelerations for all states...")
    accels_list = []
    for i in range(len(states_array)):
        accel = cartdoublependulum_dynamics_controlled(states_array[i], forces_array[i, 0])
        accels_list.append(accel)
    accels_array = np.array(accels_list)
    
    # Convert to tensors
    states_tensor = torch.tensor(states_array, dtype=torch.float32)
    force_tensor = torch.tensor(forces_array, dtype=torch.float32)
    accel_tensor = torch.tensor(accels_array, dtype=torch.float32)
    
    # Log statistics
    logger.info(f"Generated data statistics:")
    logger.info(f"  States shape: {states_tensor.shape}")
    logger.info(f"  Forces shape: {force_tensor.shape}")
    logger.info(f"  Accelerations shape: {accel_tensor.shape}")
    logger.info(f"  State ranges:")
    for i, name in enumerate(['x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot']):
        logger.info(f"    {name}: [{states_tensor[:, i].min():.3f}, {states_tensor[:, i].max():.3f}]")
    
    # Save dataset if requested
    if save_to_dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f"cart_double_pendulum_trajectory_{num_steps}_{seed}.npz")
        
        np.savez_compressed(dataset_path,
                 states=states_tensor.cpu().numpy(),
                 forces=force_tensor.cpu().numpy(),
                 accelerations=accel_tensor.cpu().numpy(),
                 seed=seed,
                 num_steps=num_steps,
                 generation_method='trajectory_based')
        
        logger.info(f"Saved trajectory-based dataset to {dataset_path}")
    
    return states_tensor, force_tensor, accel_tensor


