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

def runge_kutta_four_multiple_body(bodys, force_function, num_steps, time_step,
                                   if_final_state=False, current_device=None, model=None):
    """
    Integrates system dynamics using the Runge-Kutta 4th order method.
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
                         logger.error(f"RK4 GetAccel: Shape {acc.shape} unexpected for {num_bodys} bodies. Zeros."); acc = torch.zeros(num_bodys, device=current_device)
                return acc
            except Exception as e:
                logger.error(f"Error calling force_function in RK4 GetAccel: {e}")
                return torch.zeros(num_bodys, device=current_device) # Fallback

        # k1
        if current_state_i.dim() == 2:
            k1_vel = current_state_i[:, 1]
            k1_accel = get_accel(current_state_i, model)


            # k2 - Create new tensor for intermediate state
            state_k2_input = torch.zeros_like(current_state_i)
            state_k2_input[:, 0] = current_state_i[:, 0] + 0.5 * time_step * k1_vel
            state_k2_input[:, 1] = current_state_i[:, 1] + 0.5 * time_step * k1_accel
            k2_vel = state_k2_input[:, 1]
            k2_accel = get_accel(state_k2_input, model)

            # k3 - Create new tensor for intermediate state
            state_k3_input = torch.zeros_like(current_state_i)
            state_k3_input[:, 0] = current_state_i[:, 0] + 0.5 * time_step * k2_vel
            state_k3_input[:, 1] = current_state_i[:, 1] + 0.5 * time_step * k2_accel
            k3_vel = state_k3_input[:, 1]
            k3_accel = get_accel(state_k3_input, model)

            # k4 - Create new tensor for intermediate state
            state_k4_input = torch.zeros_like(current_state_i)
            state_k4_input[:, 0] = current_state_i[:, 0] + time_step * k3_vel
            state_k4_input[:, 1] = current_state_i[:, 1] + time_step * k3_accel
            k4_vel = state_k4_input[:, 1]
            k4_accel = get_accel(state_k4_input, model)

            # Update positions and velocities - Create new tensor for the next state
            next_pos = current_state_i[:, 0] + (time_step / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
            next_vel = current_state_i[:, 1] + (time_step / 6.0) * (k1_accel + 2 * k2_accel + 2 * k3_accel + k4_accel)
            next_state = torch.stack([next_pos, next_vel], dim=-1)

            body_trajectory_list.append(next_state)
            current_state_i = next_state # Update for the next iteration
        elif current_state_i.dim() == 3:
            k1_vel = current_state_i[:,:, 1]
            k1_accel = get_accel(current_state_i, model)

            # k2 - Create new tensor for intermediate state
            state_k2_input = torch.zeros_like(current_state_i)
            state_k2_input[:,:, 0] = current_state_i[:,:, 0] + 0.5 * time_step * k1_vel
            state_k2_input[:,:, 1] = current_state_i[:,:, 1] + 0.5 * time_step * k1_accel
            k2_vel = state_k2_input[:,:, 1]
            k2_accel = get_accel(state_k2_input, model)

            # k3 - Create new tensor for intermediate state
            state_k3_input = torch.zeros_like(current_state_i)
            state_k3_input[:,:, 0] = current_state_i[:,:, 0] + 0.5 * time_step * k2_vel
            state_k3_input[:,:, 1] = current_state_i[:,:, 1] + 0.5 * time_step * k2_accel
            k3_vel = state_k3_input[:,:, 1]
            k3_accel = get_accel(state_k3_input, model)

            # k4 - Create new tensor for intermediate state
            state_k4_input = torch.zeros_like(current_state_i)
            state_k4_input[:,:, 0] = current_state_i[:,:, 0] + time_step * k3_vel
            state_k4_input[:,:, 1] = current_state_i[:,:, 1] + time_step * k3_accel
            k4_vel = state_k4_input[:,:, 1]
            k4_accel = get_accel(state_k4_input, model)

            # Update positions and velocities - Create new tensor for the next state
            next_pos = current_state_i[:,:, 0] + (time_step / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
            next_vel = current_state_i[:,:, 1] + (time_step / 6.0) * (k1_accel + 2 * k2_accel + 2 * k3_accel + k4_accel)
            next_state = torch.stack([next_pos, next_vel], dim=-1)

            body_trajectory_list.append(next_state)

    # Stack the list into a tensor at the end
    body_tensor_trajectory = torch.stack(body_trajectory_list, dim=0)

    # Return final state with grad enabled if requested
    if if_final_state:
        final_state = body_tensor_trajectory[-1].clone().requires_grad_()
        return final_state
    else:
        return body_tensor_trajectory


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