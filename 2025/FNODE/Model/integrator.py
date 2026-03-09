# FNODE/Model/Integrator.py
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_euler_multiple_body(bodys, force_function, num_steps, time_step,
                                if_final_state=False, current_device=None, model=None):
    """
    Integrates system dynamics using the forward Euler method.
    'bodys': Initial state [num_bodys, 2] (position, velocity/momentum).
    'force_function': Computes accelerations given current state [num_bodys, 2] and optional model.
                      Output should be shape [num_bodys].
    """
    if current_device is None: current_device = device
    if not isinstance(bodys, torch.Tensor): bodys = torch.tensor(bodys, dtype=torch.float32, device=current_device)

    num_bodys = bodys.shape[0]
    # Initialize trajectory list instead of pre-allocating tensor to avoid potential in-place issues if list grows
    body_trajectory_list = [bodys.clone()]

    current_state_i = bodys.clone()

    for i in range(num_steps - 1):
        # Detach previous state for gradient calculation if needed by force_function
        current_state_for_grad = current_state_i.detach().requires_grad_(True)
        try:
            acceleration = force_function(current_state_for_grad, model=model)
        except Exception as e:
             logger.error(f"Error calling force_function in FE step {i}: {e}")
             acceleration = torch.zeros(num_bodys, device=current_device) # Fallback

        # Ensure acceleration has the correct shape [num_bodys]
        if acceleration.ndim > 1:
             if acceleration.shape[-1] == 1 and acceleration.ndim == 2: acceleration = acceleration.squeeze(-1)
             elif acceleration.numel() == num_bodys: acceleration = acceleration.view(num_bodys)
             else:
                logger.error(f"FE: Acceleration shape {acceleration.shape} unexpected for {num_bodys} bodies. Using zeros.")
                acceleration = torch.zeros(num_bodys, device=current_device)

        # Calculate next state without in-place modification
        new_positions = current_state_i[:, 0] + time_step * current_state_i[:, 1]
        new_velocities = current_state_i[:, 1] + time_step * acceleration
        next_state = torch.stack([new_positions, new_velocities], dim=-1)

        body_trajectory_list.append(next_state)
        current_state_i = next_state # Update for the next iteration

    # Stack the list into a tensor at the end
    body_tensor_trajectory = torch.stack(body_trajectory_list, dim=0)

    # Return final state with grad enabled if requested
    if if_final_state:
        final_state = body_tensor_trajectory[-1].clone().requires_grad_()
        return final_state
    else:
        return body_tensor_trajectory


def rk4_step(f, x, t, h):
    """Single Runge-Kutta 4 integration step for generic functions."""
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(x + k3, t + h)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def sep_stormer_verlet_multiple_body(bodys, dH_dq, dH_dp, num_steps, time_step,
                                     if_final_state=False, current_device=None, model=None):
    """
    Separable Störmer-Verlet integrator for Hamiltonian systems.
    Expects bodys shape [num_bodys, 2] with [position, momentum].
    """
    if current_device is None:
        current_device = device
    if not isinstance(bodys, torch.Tensor):
        bodys = torch.tensor(bodys, dtype=torch.float32, device=current_device)

    num_bodys = bodys.shape[0]
    q = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=current_device)
    p = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=current_device)
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]

    for i in range(num_steps - 1):
        temp_state = torch.stack((q[i, :], p[i, :]), dim=1)
        grad_q = dH_dq(temp_state, model=model)
        p_half = p[i, :] - 0.5 * time_step * grad_q

        temp_state_half = torch.stack((q[i, :], p_half), dim=1)
        grad_p = dH_dp(temp_state_half, model=model)
        q[i + 1, :] = q[i, :] + time_step * grad_p

        temp_state_next = torch.stack((q[i + 1, :], p_half), dim=1)
        grad_q_next = dH_dq(temp_state_next, model=model)
        p[i + 1, :] = p_half - 0.5 * time_step * grad_q_next

    if if_final_state:
        return torch.stack((q[-1, :], p[-1, :]), dim=1).requires_grad_()
    return torch.stack((q, p), dim=2)


def yoshida4_multiple_body(bodys, dH_dq, dH_dp, num_steps, time_step,
                           if_final_state=False, current_device=None, model=None):
    """
    Fourth-order Yoshida symplectic integrator for separable Hamiltonians.
    Fixed to match PNODE-for-MBD2 implementation (momentum update first, then position).
    """
    if current_device is None:
        current_device = device
    if not isinstance(bodys, torch.Tensor):
        bodys = torch.tensor(bodys, dtype=torch.float32, device=current_device)

    # Coefficients matching PNODE-for-MBD2 implementation
    c_list = [
        1.0 / (2 * (2 - 2 ** (1.0 / 3))),
        (1 - 2 ** (1.0 / 3)) / (2 * (2 - 2 ** (1.0 / 3))),
        (1 - 2 ** (1.0 / 3)) / (2 * (2 - 2 ** (1.0 / 3))),
        1.0 / (2 * (2 - 2 ** (1.0 / 3)))
    ]
    d_list = [
        1.0 / (2 - 2 ** (1.0 / 3)),
        -2 ** (1.0 / 3) / (2 - 2 ** (1.0 / 3)),
        1.0 / (2 - 2 ** (1.0 / 3)),
        0
    ]

    q = torch.zeros((num_steps, bodys.shape[0]), dtype=torch.float32, device=current_device)
    p = torch.zeros_like(q)
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]

    # NOTE: These (c_list, d_list) are the merged-coefficient form that results
    # from composing 2nd-order Störmer-Verlet steps. The correct update order is:
    #   q <- q + c*dt*dH/dp
    #   p <- p - d*dt*dH/dq
    # with the final d=0 stage meaning "no p update".
    for i in range(num_steps - 1):
        q_i = q[i, :].clone()
        p_i = p[i, :].clone()
        for c, d in zip(c_list, d_list):
            # Update position first
            temp_state = torch.stack((q_i, p_i), dim=1)
            q_i = q_i + c * time_step * dH_dp(temp_state, model=model)
            # Then update momentum (skip when d == 0)
            if d != 0:
                temp_state = torch.stack((q_i, p_i), dim=1)
                p_i = p_i - d * time_step * dH_dq(temp_state, model=model)
        q[i + 1, :] = q_i
        p[i + 1, :] = p_i

    if if_final_state:
        return torch.stack((q[-1, :], p[-1, :]), dim=1).requires_grad_()
    return torch.stack((q, p), dim=2)


def fukushima6_multiple_body(bodys, dH_dq, dH_dp, num_steps, time_step, if_final_state=False, device=device, model=None):
    """6th-order symplectic integrator via symmetric composition of Störmer-Verlet.

    Uses the standard 6th-order symmetric composition coefficients:
        [w1, w2, w3, w0, w3, w2, w1]

    Each stage applies one full Störmer-Verlet step of size h = w * time_step:
        p <- p - (h/2) * dH/dq(q, p)
        q <- q + h     * dH/dp(q, p)
        p <- p - (h/2) * dH/dq(q, p)

    Expects bodys shape [num_bodys, 2] with [q, p].
    """

    # Stage coefficients (commonly attributed to Yoshida/Suzuki; also used in literature around Fukushima methods)
    w1 = 0.784513610477560
    w2 = 0.235573213359357
    w3 = -1.177679984178870
    w0 = 1.315186320683906
    stages = (w1, w2, w3, w0, w3, w2, w1)

    current_device = device
    if not isinstance(bodys, torch.Tensor):
        bodys = torch.tensor(bodys, dtype=torch.float32, device=current_device)
    else:
        bodys = bodys.to(current_device)

    num_bodys = bodys.shape[0]
    q = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=current_device)
    p = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=current_device)
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]

    for i in range(num_steps - 1):
        q_i = q[i, :].clone()
        p_i = p[i, :].clone()

        for w in stages:
            h = float(w) * time_step

            temp_state = torch.stack((q_i, p_i), dim=1)
            grad_q = dH_dq(temp_state, model=model)
            p_i = p_i - 0.5 * h * grad_q

            temp_state = torch.stack((q_i, p_i), dim=1)
            grad_p = dH_dp(temp_state, model=model)
            q_i = q_i + h * grad_p

            temp_state = torch.stack((q_i, p_i), dim=1)
            grad_q = dH_dq(temp_state, model=model)
            p_i = p_i - 0.5 * h * grad_q

        q[i + 1, :] = q_i
        p[i + 1, :] = p_i

    if if_final_state:
        return torch.stack((q[-1, :], p[-1, :]), dim=1).requires_grad_()
    return torch.stack((q, p), dim=2)

def runge_kutta_four_multiple_body(bodys, force_function, num_steps, time_step, if_final_state=False,
                                             device=device, model=None):
    num_bodys = bodys.shape[0]
    positions = torch.zeros((num_steps, num_bodys), dtype=torch.float64, device=device)
    velocities = torch.zeros((num_steps, num_bodys), dtype=torch.float64, device=device)
    body_tensor = torch.zeros((num_steps, num_bodys, 2), dtype=torch.float64, device=device)
    positions[0, :] = bodys[:, 0]
    velocities[0, :] = bodys[:, 1]
    body_tensor[0, :, :] = bodys
    # Preallocate temporary tensors outside the loop
    body_temp = torch.zeros((num_bodys, 2), dtype=torch.float64, device=device)
    k1 = torch.zeros_like(body_temp)
    k2 = torch.zeros_like(body_temp)
    k3 = torch.zeros_like(body_temp)
    k4 = torch.zeros_like(body_temp)
    for i in range(num_steps - 1):
        k1[:, 0] = velocities[i, :]
        k1[:, 1] = force_function(body_tensor[i, :, :], model=model)
        body_temp[:, 0] = positions[i, :] + 0.5 * time_step * k1[:, 0]
        body_temp[:, 1] = velocities[i, :] + 0.5 * time_step * k1[:, 1]
        k2[:, 0] = body_temp[:, 1]
        k2[:, 1] = force_function(body_temp, model=model)
        body_temp[:, 0] = positions[i, :] + 0.5 * time_step * k2[:, 0]
        body_temp[:, 1] = velocities[i, :] + 0.5 * time_step * k2[:, 1]
        k3[:, 0] = body_temp[:, 1]
        k3[:, 1] = force_function(body_temp, model=model)
        body_temp[:, 0] = positions[i, :] + time_step * k3[:, 0]
        body_temp[:, 1] = velocities[i, :] + time_step * k3[:, 1]
        k4[:, 0] = body_temp[:, 1]
        k4[:, 1] = force_function(body_temp, model=model)
        # Combining the RK4 update steps
        positions[i + 1, :] = positions[i, :] + time_step * (k1[:, 0] + 2 * k2[:, 0] + 2 * k3[:, 0] + k4[:, 0]) / 6
        velocities[i + 1, :] = velocities[i, :] + time_step * (k1[:, 1] + 2 * k2[:, 1] + 2 * k3[:, 1] + k4[:, 1]) / 6
        body_tensor[i + 1, :, 0] = positions[i + 1, :]
        body_tensor[i + 1, :, 1] = velocities[i + 1, :]
    if if_final_state:
        return body_tensor[-1, :, :]
    else:
        return body_tensor



def midpoint_method_multiple_body(bodys, force_function, num_steps, time_step,
                                  if_final_state=False, current_device=None, model=None):
    """
    Integrates system dynamics using the explicit midpoint method.
    Avoids in-place operations that could interfere with autograd.
    'bodys': Initial state [num_bodys, 2] (position, velocity/momentum).
    'force_function': Computes accelerations given current state [num_bodys, 2] and optional model.
                      Output should be shape [num_bodys].
    """
    if current_device is None: current_device = device
    if not isinstance(bodys, torch.Tensor): bodys = torch.tensor(bodys, dtype=torch.float32, device=current_device)

    num_bodys = bodys.shape[0]
    # Initialize trajectory list
    body_trajectory_list = [bodys.clone()]
    current_state_i = bodys.clone()

    for i in range(num_steps - 1):
        # Detach previous state for gradient calculation if needed by force_function
        current_state_for_grad = current_state_i.detach().requires_grad_(True)

        def get_accel(state_tensor, model_instance):
            # Ensure requires_grad is set correctly for the force function call
            state_input = state_tensor.detach().requires_grad_(True)
            try:
                acc = force_function(state_input, model=model_instance)
                # Ensure acceleration has the correct shape [num_bodys]
                if acc.ndim > 1:
                    if acc.shape[-1] == 1 and acc.ndim == 2: acc = acc.squeeze(-1)
                    elif acc.numel() == num_bodys: acc = acc.view(num_bodys)
                    else:
                         logger.error(f"Midpoint GetAccel: Shape {acc.shape} unexpected for {num_bodys} bodies. Zeros."); acc = torch.zeros(num_bodys, device=current_device)
                return acc
            except Exception as e:
                logger.error(f"Error calling force_function in Midpoint GetAccel: {e}")
                return torch.zeros(num_bodys, device=current_device) # Fallback

        # Calculate midpoint state estimate - Create new tensors
        half_time_step = 0.5 * time_step
        accel_i = get_accel(current_state_i, model)
        mid_vel = current_state_i[:, 1] + half_time_step * accel_i
        mid_pos = current_state_i[:, 0] + half_time_step * current_state_i[:, 1]
        mid_state = torch.stack([mid_pos, mid_vel], dim=-1)

        # Evaluate acceleration at the midpoint state
        mid_accel = get_accel(mid_state, model)

        # Update positions and velocities using midpoint velocity and acceleration - Create new tensor
        next_pos = current_state_i[:, 0] + time_step * mid_vel # Use midpoint velocity
        next_vel = current_state_i[:, 1] + time_step * mid_accel # Use midpoint acceleration
        next_state = torch.stack([next_pos, next_vel], dim=-1)

        body_trajectory_list.append(next_state)
        current_state_i = next_state # Update for the next iteration

    # Stack the list into a tensor at the end
    body_tensor_trajectory = torch.stack(body_trajectory_list, dim=0)

    # Return final state with grad enabled if requested
    if if_final_state:
        final_state = body_tensor_trajectory[-1].clone().requires_grad_()
        return final_state
    else:
        return body_tensor_trajectory
