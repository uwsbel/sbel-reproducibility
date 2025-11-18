"""
MPC Cart-Pole Control - Analytical Baseline
Exact implementation following PNODE-for-MBD2/simulations/mpc_cartpole.py
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def force_cp_ext(bodys, u, model=None):
    """
    Cart-pole dynamics exactly as in PNODE-for-MBD2/Model/force_fun.py
    """
    m1 = 1  # cart mass
    m2 = 1  # pole mass  
    l = 0.5
    g = 9.81
    
    x = bodys[0, 0]
    x_dot = bodys[0, 1]
    theta = bodys[1, 0]
    theta_dot = bodys[1, 1]
    
    x_ddot_num = -m2 * g * torch.sin(theta) * torch.cos(theta) - m2 * l * theta_dot ** 2 * torch.sin(theta) + u
    x_ddot_den = m1 + m2 * (1 - torch.cos(theta) ** 2)
    x_ddot = x_ddot_num / x_ddot_den
    
    theta_ddot_num = (g * torch.sin(theta) * (m1 + m2) + m2 * l * theta_dot ** 2 * torch.sin(theta) * torch.cos(theta)) - u * torch.cos(theta)
    theta_ddot_den = l * (m1 + m2 * (1 - torch.cos(theta) ** 2))
    theta_ddot = theta_ddot_num / theta_ddot_den
    
    return_value = torch.tensor([x_ddot, theta_ddot], dtype=torch.float32, device=device)
    return return_value

def midpoint_control_single_step(bodys, external, force_function, time_step, device=device, model=None):
    """
    Midpoint integration exactly as in PNODE-for-MBD2/Model/Integrator.py
    """
    num_bodys = 2
    positions = bodys[:, 0]
    velocities = bodys[:, 1]
    
    # Concatenate position, velocity and external force to form the input
    bodys_portion = bodys.view(-1)  # Reshape 'bodys' to a flat tensor
    external_portion = external.view(-1)  # Add a dimension to 'external' to make it compatible
    
    # Concatenate along the appropriate dimension to combine the tensors
    body_tensor = torch.cat([bodys_portion, external_portion], dim=0).requires_grad_(True)
    
    # Midpoint integration
    state = body_tensor.detach().requires_grad_(True)
    half_time_step = time_step / 2
    mid_positions = positions + half_time_step * velocities
    
    # Make a 5 dimensional tensor
    mid_states1 = torch.cat([
        mid_positions[0].view(-1),  # pos1
        velocities[0].view(-1),      # vel1
        mid_positions[1].view(-1),  # pos2
        velocities[1].view(-1),      # vel2
        external.view(-1)            # ext
    ], dim=0).requires_grad_(True)
    
    mid_states2 = torch.cat([
        mid_positions[0].view(-1),  # pos1
        velocities[0].view(-1),      # vel1
        mid_positions[1].view(-1),  # pos2
        velocities[1].view(-1)       # vel2
    ], dim=0).requires_grad_(True)
    
    mid_states2 = mid_states2.reshape(2, 2)
    mid_acceleration = force_function(mid_states2, external_portion, model=model)
    
    mid_velocities = velocities + half_time_step * mid_acceleration
    
    # Update positions and velocities using midpoint
    new_positions = positions + time_step * mid_velocities
    new_velocities = velocities + time_step * mid_acceleration
    
    # Form the output using the new position and velocity
    new_tensor = torch.stack((new_positions, new_velocities), dim=1).requires_grad_(True)
    
    return new_tensor

# Constants
m, l, M, g = 1, 0.5, 1, 9.81
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [(m+M)*g/l/m, 0, 0, 0], [-m*g/M, 0, 0, 0]])
B = np.array([[0], [0], [-1/l/M], [1/M]])

# Objectives and constraints
IC = np.array([np.pi/6, 1, 0, 0])
zmax, zmin = 10, -10
umax, umin = 100, -100
thetamax, thetamin = np.pi/2, -np.pi/2
dt = 0.05
h = 0.05
t = np.arange(0, 10+h, h)
Q = np.eye(4)
R = np.eye(1)
Qhalf = sqrtm(Q)
Rhalf = sqrtm(R)

# Model Predictive Control
N = 50
x = np.zeros((4, len(t)))
u = np.zeros(len(t))
init = IC
umpc = 0
x[:, 0] = init

def dx(t, x, u):
    return A.dot(x) + B.flatten()*u

for k in range(len(t)-1):
    if k % 1 == 0:
        X = cp.Variable((4, N+1))
        U = cp.Variable((1, N))
        constraints = [X[:, 1:N+1] == dt*(A@X[:, 0:N] + B@U) + X[:, 0:N], X[:, 0] == init, X[:, N] == 0,
                       cp.max(X[1, :]) <= zmax, cp.min(X[1, :]) >= zmin,
                       cp.max(X[0, :]) <= thetamax, cp.min(X[0, :]) >= thetamin,
                       cp.max(U) <= umax, cp.min(U) >= umin]
        objective = cp.Minimize(cp.norm(cp.vstack([Qhalf@X[:, 0:N], Rhalf@U]), 'fro'))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)  # Using SCS as the solver

        u[k] = U.value[0, 0]
        umpc = U.value[0, 0]

    print(k)
    print("x[:, k]",x[:, k])
    print("u[k]",u[k])
    # x order: theta, x, theta_dot, x_dot
    # tensor order: x, x_dot, theta, theta_dot
    bodys_tensor = torch.tensor([[x[1, k],x[3,k]],[x[0,k],x[2,k]]], dtype=torch.float32).to(device)
    external_force = torch.tensor([u[k]], dtype=torch.float32).to(device)
    state = midpoint_control_single_step(bodys_tensor, external_force,force_cp_ext ,dt)
    cpu = state.cpu().detach().numpy().reshape(4, 1)

    x[:, k + 1] = np.array([cpu[2, 0], cpu[0, 0], cpu[3, 0], cpu[1, 0]])
    init = x[:, k + 1]

# Save full 10-second results
t_full = t
x_full = x
u_full = u[:-1]  # Exclude last u since it's not used

# Calculate the loss along the trajectory (as in reference)
loss_trajectory = np.zeros(len(t))
for i in range(len(t)):
    loss_trajectory[i] = np.sum(x[:,i]**2) + u[i]**2 if i < len(t)-1 else np.sum(x[:,i]**2)

print("loss_trajectory", loss_trajectory)

# Save results to mpc_cartpole directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results_baseline")
os.makedirs(results_dir, exist_ok=True)

# Save full 10-second data for plotting
np.savetxt(os.path.join(results_dir, "times.txt"), t_full, fmt="%.4f")
np.savetxt(os.path.join(results_dir, "states.txt"), x_full.T, fmt="%.6f",
           header="theta x theta_dot x_dot")
np.savetxt(os.path.join(results_dir, "controls.txt"), u_full, fmt="%.6f")

# Save summary
with open(os.path.join(results_dir, "summary.txt"), "w") as f:
    f.write("MPC Cart-Pole Control - Analytical\n")
    f.write(f"Simulation time: {t_full[-1]} seconds\n")
    f.write(f"Time step: {dt} seconds\n")
    f.write(f"Number of steps: {len(t_full)-1}\n")
    f.write(f"Initial state: {IC}\n")
    f.write(f"Final state (10s): {x_full[:, -1]}\n")
    f.write(f"Final error norm (10s): {np.linalg.norm(x_full[:, -1]):.6f}\n")
    f.write(f"Control effort (10s): {np.sum(u_full**2) * dt:.2f}\n")
    f.write(f"Max |control| (10s): {np.max(np.abs(u_full)):.2f} N\n")

print(f"\nFinal state at 10s: θ={x[0,-1]:.6f}, x={x[1,-1]:.6f}, θ̇={x[2,-1]:.6f}, ẋ={x[3,-1]:.6f}")
print(f"Final error norm at 10s: {np.linalg.norm(x[:, -1]):.6f}")