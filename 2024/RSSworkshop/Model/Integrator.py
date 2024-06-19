import numpy as np
from numpy import cos, sin
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import fsolve
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import scipy
from utils import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_euler_multiple_body(bodys, force_function, num_steps, time_step, if_final_state=False, device=device,
                                model=None):
    #print("Using forward Euler method to integrate the system,bodys shape is " + str(bodys.shape))
    num_bodys = bodys.shape[0]
    positions = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    velocities = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    body_tensor = torch.zeros((num_steps, num_bodys, 2), dtype=torch.float32, device=device)
    #last_tensor = torch.tensor(bodys, dtype=torch.float32, device=device).detach().requires_grad_(True)
    positions[0, :] = bodys[:, 0]
    velocities[0, :] = bodys[:, 1]
    body_tensor[0] = bodys.clone()
    for i in range(num_steps - 1):
        state=body_tensor[i].detach().requires_grad_(True)
        acceleration = force_function(state, model=model)
        new_positions = positions[i] + time_step * velocities[i]
        new_velocities = velocities[i] + time_step * acceleration
        positions[i + 1] = new_positions
        velocities[i + 1] = new_velocities
        body_tensor[i + 1, :, 0] = new_positions
        body_tensor[i + 1, :, 1] = new_velocities
        last_tensor = torch.stack([positions[i+1], velocities[i+1]], dim=-1).requires_grad_()
    if if_final_state:
        return last_tensor
    else:
        return body_tensor
def forward_euler_multiple_body_single_step_external(bodys, force_function,dt, device=device,model=None):
    #input bodys is a tensor with shape (num_bodys,3), the first column is position, the second column is velocity, the third column is external input like force, torque, etc.
    #do a single step forward euler integration
    #output bodys is a tensor with shape (num_bodys,3), the first column is position, the second column is velocity, the third column is external input like force, torque, etc.
    num_bodys = bodys.shape[0]
    positions = torch.zeros((num_bodys), dtype=torch.float32, device=device)
    velocities = torch.zeros((num_bodys), dtype=torch.float32, device=device)
    body_tensor = torch.zeros((num_bodys, 3), dtype=torch.float32, device=device)
    positions[:] = bodys[:, 0]
    velocities[:] = bodys[:, 1]
    body_tensor[:] = bodys.clone()
    state = body_tensor.detach().requires_grad_(True)
    acceleration = force_function(state, model=model)
    new_positions = positions[:] + velocities[:] * dt
    new_velocities = velocities[:] + acceleration[:] * dt
    positions[:] = new_positions[:]
    velocities[:] = new_velocities[:]
    body_tensor[:, 0] = new_positions[:]
    body_tensor[:, 1] = new_velocities[:]
    body_tensor[:, 2] = bodys[:, 2]+dt
    return body_tensor



#the midpoint method for multiple body system
def midpoint_method_multiple_body(bodys, force_function, num_steps, time_step, if_final_state=False, device=device, model=None):
    num_bodys = bodys.shape[0]
    positions = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    velocities = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    body_tensor = torch.zeros((num_steps, num_bodys, 2), dtype=torch.float32, device=device)
    #print(bodys)

    positions[0, :] = bodys[:, 0]
    velocities[0, :] = bodys[:, 1]
    body_tensor[0] = bodys.clone()

    for i in range(num_steps - 1):
        state = body_tensor[i].detach().requires_grad_(True)

        # Calculate midpoint
        half_time_step = time_step / 2
        mid_positions = positions[i] + half_time_step * velocities[i]
        mid_state = torch.stack([mid_positions, velocities[i]], dim=-1).requires_grad_(True)
        mid_acceleration = force_function(mid_state, model=model)
        #print("mid_acceleration is ",mid_acceleration)
        mid_velocities = velocities[i] + half_time_step * mid_acceleration

        # Update positions and velocities using midpoint
        new_positions = positions[i] + time_step * mid_velocities
        new_velocities = velocities[i] + time_step * mid_acceleration
        #print("new_positions is ",new_positions)
        #print("new_velocities is ",new_velocities)

        positions[i + 1] = new_positions
        velocities[i + 1] = new_velocities
        body_tensor[i + 1, :, 0] = new_positions
        body_tensor[i + 1, :, 1] = new_velocities

    if if_final_state:
        return torch.stack([positions[-1], velocities[-1]], dim=-1).requires_grad_(True)
    else:
        return body_tensor
def backward_euler_multiple_body(body_tensor, force_function, num_steps, time_step,
                                 if_final_state=False, Max_iteration=50, Tolerance=1e-6, device=device):
    num_bodys = body_tensor.shape[1]
    body_tensor_result = torch.zeros((num_steps, num_bodys, 2), dtype=torch.float32, device=device)
    body_tensor_result[0, :, :] = body_tensor[0, :, :]
    # Time-stepping loop
    for i in range(num_steps - 1):
        # Calculate acceleration
        acceleration = force_function(body_tensor_result[i, :, :])
        # Initial guess using forward Euler for positions and velocities
        body_tensor_result[i + 1, :, 0] = body_tensor_result[i, :, 0] + time_step * body_tensor_result[i, :, 1]
        body_tensor_result[i + 1, :, 1] = body_tensor_result[i, :, 1] + time_step * acceleration
        # Fixed-point iteration for backward Euler
        for j in range(Max_iteration):
            new_positions = body_tensor_result[i, :, 0] + time_step * body_tensor_result[i + 1, :, 1]
            new_velocities = body_tensor_result[i, :, 1] + time_step * force_function(
                torch.stack([new_positions, body_tensor_result[i + 1, :, 1]], dim=-1))
            error = torch.norm(new_positions - body_tensor_result[i + 1, :, 0]) + torch.norm(
                new_velocities - body_tensor_result[i + 1, :, 1])
            body_tensor_result[i + 1, :, 0] = new_positions
            body_tensor_result[i + 1, :, 1] = new_velocities
            if error < Tolerance:
                break
    # Output based on if_final_state flag
    if if_final_state:
        return body_tensor_result[-1, :, :]
    else:
        return body_tensor_result
def rk4_step(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + k1 / 2, t + h / 2)
    k3 = h * f(x + k2 / 2, t + h / 2)
    k4 = h * f(x + k3, t + h)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def runge_kutta_four_multiple_body(bodys, force_function, num_steps, time_step, if_final_state=False,
                                             device=device, model=None):
    num_bodys = bodys.shape[0]
    positions = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    velocities = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    body_tensor = torch.zeros((num_steps, num_bodys, 2), dtype=torch.float32, device=device)
    positions[0, :] = bodys[:, 0]
    velocities[0, :] = bodys[:, 1]
    body_tensor[0, :, :] = bodys
    # Preallocate temporary tensors outside the loop
    body_temp = torch.zeros((num_bodys, 2), dtype=torch.float32, device=device)
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
        k3[:, 0] = body_temp[:, 1]
        k3[:, 1] = force_function(body_temp, model=model)
        body_temp[:, 0] = positions[i, :] + time_step * k3[:, 0]
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
#Symplectic integrator for multiple body system with energy conservation property, based on Hamiltonian dynamics
#The numerical methods used here are the Störmer–Verlet methods for Hamiltonian systems
def sep_stormer_verlet_multiple_body(bodys,dH_dq,dH_dp,num_steps,time_step,if_final_state=False,device=device,model=None):
    #body_tensor is a tensor with shape (num_steps,num_bodys,2), the first column is generalized coordinate, the second column is generalized momentum
    #dH_dq is a function that takes in generalized coordinate and momentum and returns the derivative of the potential energy with respect to the generalized coordinate, it can be a neural network
    #dT_dp is a function that takes in generalized coordinate and momentum and returns the derivative of the kinetic energy with respect to the generalized momentum, it can be a neural network
    #num_steps is the number of steps we want to integrate
    #time_step is the time step we want to use
    #if_final_state is a boolean variable that indicates whether we only want to return the final state
    #The following code is the Störmer–Verlet methods for Hamiltonian systems
    #print("Using Störmer-Verlet method to integrate the system, bodys shape is " + str(bodys.shape))
    num_bodys = bodys.shape[0]
    # Create storage for positions (q) and momenta (p) over time
    q = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    p = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    # Initialize the positions and momenta from the bodys tensor
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]
    # Störmer-Verlet integration loop
    for i in range(num_steps - 1):
        # Update the momentum using half of the time step
        temp_body_tensor = torch.stack((q[i, :], p[i, :]), dim=1)
        grad_q_V = dH_dq(temp_body_tensor,model=model)
        p_half = p[i, :] - (time_step / 2) * grad_q_V
        # Update the position using the full time step
        temp_body_tensor2 = torch.stack((q[i, :], p_half), dim=1)
        grad_p_T = dH_dp(temp_body_tensor2,model=model)
        q[i + 1, :] = q[i, :] + time_step * grad_p_T
        # Update the momentum again using the remaining half of the time step
        temp_body_tensor3=torch.stack((q[i+1,:],p_half),dim=1)
        grad_q_V_next = dH_dq(temp_body_tensor3,model=model)
        p[i + 1, :] = p_half - (time_step / 2) * grad_q_V_next
    # Construct the body tensor from q and p
    body_tensor = torch.stack((q, p), dim=2)
    if if_final_state:
        return body_tensor[-1, :, :]
    else:
        return body_tensor
def implicit_stormer_verlet_step(p, q, h, grad_p, grad_q):
    """The implicit Störmer-Verlet integrator for a single step."""
    # Define the function for the first half update of p
    func1 = lambda x: x - p + h / 2 * grad_q(x, q).numpy()
    # Use fsolve to solve for the half step update of p
    p_half_new = fsolve(func1, p.numpy())
    # Define the function for the full update of q
    func2 = lambda x: x - q - h / 2 * (grad_p(p_half_new, q).numpy() + grad_p(p_half_new, x).numpy())
    # Use fsolve to solve for the full step update of q
    q_new = fsolve(func2, q.numpy())
    # Calculate the second half update for p
    p_new = p_half_new - h / 2 * grad_q(p_half_new, q_new).numpy()
    return torch.tensor(p_new, dtype=torch.float32, device=device), torch.tensor(q_new, dtype=torch.float32,device=device)
def yoshida4_multiple_body(bodys, dH_dq, dH_dp, num_steps, time_step, if_final_state=False,
                                           device=device, model=None):
    # Coefficients for the 4th-order symplectic integrator
    c_list = [1. / (2 * (2 - 2 ** (1. / 3))), (1 - 2 ** (1. / 3)) / (2 * (2 - 2 ** (1. / 3))),
              (1 - 2 ** (1. / 3)) / (2 * (2 - 2 ** (1. / 3))), 1. / (2 * (2 - 2 ** (1. / 3)))]
    d_list = [1. / (2 - 2 ** (1. / 3)), -2 ** (1. / 3) / (2 - 2 ** (1. / 3)), 1. / (2 - 2 ** (1. / 3)), 0]

    num_bodys = bodys.shape[0]
    q = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    p = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]

    for i in range(num_steps - 1):
        q_new = q[i, :]
        p_new = p[i, :]
        for c, d in zip(c_list, d_list):
            # Update the momentum using coefficient d
            temp_body_tensor = torch.stack((q_new, p_new), dim=1)
            grad_q_V = dH_dq(temp_body_tensor, model=model)
            p_new = p_new - d * time_step * grad_q_V

            # Update the position using coefficient c
            temp_body_tensor2 = torch.stack((q_new, p_new), dim=1)
            grad_p_T = dH_dp(temp_body_tensor2, model=model)
            q_new = q_new + c * time_step * grad_p_T

        q[i + 1, :] = q_new
        p[i + 1, :] = p_new

    body_tensor = torch.stack((q, p), dim=2)
    if if_final_state:
        return body_tensor[-1, :, :]
    else:
        return body_tensor
def fukushima6_multiple_body(bodys, dH_dq, dH_dp, num_steps, time_step, if_final_state=False, device=device, model=None):
    # Coefficients for the 6th-order symplectic Fukushima method
    b1 = 0.784513610477560
    b2 = 0.235573213359357
    b3 = -1.177679984178870
    b4 = 1.315186320683906
    d1 = 0.3922568052387800
    d2 = 0.5100434119184585
    d3 = -0.4710533854097565
    d4 = 0.0687531682525180
    b_coeff = [b1, b2, b3, b4, b4, b3, b2, b1]
    d_coeff = [d1, d2, d3, d4, d4, d3, d2, d1]
    num_bodys = bodys.shape[0]
    q = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    p = torch.zeros((num_steps, num_bodys), dtype=torch.float32, device=device)
    q[0, :] = bodys[:, 0]
    p[0, :] = bodys[:, 1]
    for i in range(num_steps - 1):
        q_new = q[i, :]
        p_new = p[i, :]
        for d, b in zip(d_coeff, b_coeff):
            # Update the momentum using coefficient b
            temp_body_tensor = torch.stack((q_new, p_new), dim=1)
            grad_q_H = dH_dq(temp_body_tensor, model=model)
            p_new = p_new - b * time_step * grad_q_H
            # Update the position using coefficient d
            temp_body_tensor2 = torch.stack((q_new, p_new), dim=1)
            grad_p_H = dH_dp(temp_body_tensor2, model=model)
            q_new = q_new + d * time_step * grad_p_H
        q[i + 1, :] = q_new
        p[i + 1, :] = p_new
    body_tensor = torch.stack((q, p), dim=2)
    if if_final_state:
        return body_tensor[-1, :, :]
    else:
        return body_tensor
def midpoint_control_single_step(bodys, external,force_function, time_step, device=device, model=None):
    num_bodys = 2
    #print(f"Device: {device}, type: {type(device)}")  # Debugging line
    #body_tensor = torch.zeros((2 * num_bodys + 1,), dtype=torch.float32, device=device).requires_grad_(True)
    positions = bodys[:, 0]
    velocities = bodys[:, 1]
    #contatenate the position, velocity and external force to form the input
    bodys_portion = bodys.view(-1)  # Reshape 'bodys' to a flat tensor
    #print(external.shape)
    external_portion = external.view(-1)  # Add a dimension to 'external' to make it compatible
    #print("shape of bodys_portion is ",bodys_portion.shape)
    #print("shape of external_portion is ",external_portion.shape)
    # Concatenate along the appropriate dimension to combine the tensors without in-place operations
    body_tensor = torch.cat([bodys_portion, external_portion], dim=0).requires_grad_(True)

    #print("in the integration function, the input to the control system is ",body_tensor)
    #check shape
    #print("The shape of the input to the control system is ",body_tensor.shape)
    #print("The input to the control system is ",body_tensor)
    #s midpoint integration
    state = body_tensor.detach().requires_grad_(True)
    half_time_step = time_step / 2
    mid_positions = positions + half_time_step * velocities
    #make a 5 dimensional tensor and the first two colums are the first body, the second two columns are the second body, the last column is the external force
    #print("mid_positions shape is ",mid_positions.shape,"velocities shape is ",velocities.shape,"external shape is ",external.shape)
    mid_states1 = torch.cat([
    mid_positions[0].view(-1) ,  # pos1
    velocities[0].view(-1) ,     # vel1
    mid_positions[1].view(-1) ,  # pos2
    velocities[1].view(-1) ,     # vel2
    external.view(-1)           # ext, assuming external is a single value or already a 1D tensor
], dim=0).requires_grad_(True)
    mid_states2= torch.cat([mid_positions[0].view(-1) ,  # pos1
    velocities[0].view(-1) ,     # vel1
    mid_positions[1].view(-1) ,  # pos2
    velocities[1].view(-1)]      # vel2          # ext, assuming external is a single value or already a 1D tensor
, dim=0).requires_grad_(True)
    mid_states2=mid_states2.reshape(2,2)
    mid_acceleration = force_function(mid_states2,external_portion, model=model)
    #print("The acceleration is ",mid_acceleration)
    mid_velocities = velocities + half_time_step * mid_acceleration
    # Update positions and velocities using midpoint
    new_positions = positions + time_step * mid_velocities
    new_velocities = velocities + time_step * mid_acceleration
    #form the output using the new position and velocity without external stored in a new tensor
    # Assuming new_positions and new_velocities are 1D tensors of the same length
    # Stack them along a new dimension to avoid in-place operations
    new_tensor = torch.stack((new_positions, new_velocities), dim=1).requires_grad_(True)

    #print(new_tensor- bodys)
    return new_tensor