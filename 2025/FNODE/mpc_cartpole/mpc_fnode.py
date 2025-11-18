import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import os
import sys


# Add parent directory to path for FNODE model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import FNODE model
from Model.model import FNODE_CON

# Constants
m, l, M, g = 1, 0.5, 1, 9.8

# Load trained FNODE model
# trained_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                                  "saved_model", "Cart_Pole_Controlled", "trained_model_CP4.pt")
# if os.path.exists(trained_model_path):
#     trained_model = torch.load(trained_model_path, weights_only=False)
#     trained_model.eval()
#     print("Loaded trained_model_CP4.pt")
# else:
# Use FNODE_con_best.pkl from backup directory
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "saved_model", "Cart_Pole_Controlled", "run_20251113_134317", "FNODE_con_best.pkl")

# Function to convert old state dict keys to new format
def convert_state_dict_keys(old_state_dict):
    """Convert old fc1, fc2, ... keys to network.0, network.2, ... format

    Mapping:
    fc1 -> network.0 (Linear) + network.1 (Activation)
    fc2 -> network.2 (Linear) + network.3 (Activation)
    fc3 -> network.4 (Linear) + network.5 (Activation)
    fc4 -> network.6 (Linear) + network.7 (Activation)
    fc5 -> network.8 (Linear) + network.9 (Activation)
    fc6 -> network.10 (Linear, output layer - no activation after)
    """
    new_state_dict = {}

    for old_key, value in old_state_dict.items():
        if old_key.startswith('fc'):
            # Extract layer number (handles both single and multi-digit)
            layer_str = old_key[2:].split('.')[0]  # fc1.weight -> '1', fc10.bias -> '10'
            layer_num = int(layer_str)
            # Calculate new index (each layer has linear + activation, except last)
            new_idx = (layer_num - 1) * 2
            # Replace fcN with network.idx
            suffix = old_key[2+len(layer_str):]  # Get .weight or .bias part
            new_key = f'network.{new_idx}{suffix}'
            new_state_dict[new_key] = value
        else:
            # Keep other keys as is
            new_state_dict[old_key] = value

    return new_state_dict

# Create model instance (must match the saved model architecture)
trained_model = FNODE_CON(
    num_bodies=2,      # Cart and pole
    dim_input=5,       # 4 states + 1 control (theta, x, theta_dot, x_dot, u)
    dim_output=2,      # 2 accelerations (theta_ddot, x_ddot)
    layers=4,          # 6 layers (fc1 through fc6 in saved model)
    width=256,         # 256 hidden units
    activation='tanh'  # tanh activation
).to(device)

# Try loading the state dict
try:
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's already a state dict or the full model
    if isinstance(state_dict, dict):
        # Check the key format
        sample_key = list(state_dict.keys())[0]

        # Handle torch.compile() prefix
        if sample_key.startswith('_orig_mod.'):
            print("Removing _orig_mod. prefix from torch.compile()...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            sample_key = list(state_dict.keys())[0]

        if sample_key.startswith('fc'):
            # Old format - need to convert
            print("Converting old state dict format to new format...")
            state_dict = convert_state_dict_keys(state_dict)

        trained_model.load_state_dict(state_dict)
        trained_model.eval()
        print("Loaded FNODE_con_best.pkl successfully")
    else:
        # If it's not a dict, it might be a full model - try to extract state dict
        if hasattr(state_dict, 'state_dict'):
            actual_state_dict = state_dict.state_dict()

            # Handle torch.compile() prefix
            sample_key = list(actual_state_dict.keys())[0]
            if sample_key.startswith('_orig_mod.'):
                actual_state_dict = {k.replace('_orig_mod.', ''): v for k, v in actual_state_dict.items()}
                sample_key = list(actual_state_dict.keys())[0]

            # Check and convert if needed
            if sample_key.startswith('fc'):
                actual_state_dict = convert_state_dict_keys(actual_state_dict)
            trained_model.load_state_dict(actual_state_dict)
        else:
            raise ValueError("Unable to extract state dict from loaded object")
        trained_model.eval()
        print("Loaded FNODE_con_best.pkl as full model")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

# MPC time step (needed for discretization)
dt = 0.05

# Define equilibrium point for linearization
x_eq = 0.0      # cart position
dx_eq = 0.0     # cart velocity
theta_eq = 0.0  # pole angle (upright)
dtheta_eq = 0.0 # pole angular velocity
u_eq = 0.0      # control force

# Create input tensor for FNODE model at equilibrium
# Based on training data format, FNODE expects: [x, x_dot, theta, theta_dot, u]
# This matches the PNODE-style body_tensor format flattened with control
input_eq = torch.tensor([[x_eq, dx_eq, theta_eq, dtheta_eq, u_eq]], 
                        requires_grad=True, dtype=torch.float32).to(device)

# Get Jacobian of FNODE model
output = trained_model(input_eq)
print(f"FNODE output at equilibrium: {output}")

jacobian = torch.autograd.functional.jacobian(trained_model, input_eq)

# Extract Jacobian matrix and convert to numpy
# jacobian shape: (1, 2, 1, 5) -> (2, 5)
jac_np = jacobian.squeeze().cpu().numpy()

print("\nFNODE Jacobian at equilibrium:")
print(jac_np)
print("Shape:", jac_np.shape)

# Verify the Jacobian makes sense
# Input format is [x, x_dot, theta, theta_dot, u]
# Output format is [x_ddot, theta_ddot]
print(f"\nJacobian interpretation (input: [x, x_dot, theta, theta_dot, u]):")
print(f"∂(x_ddot)/∂x = {jac_np[0, 0]:.4f} (should be ~0)")
print(f"∂(x_ddot)/∂theta = {jac_np[0, 2]:.4f} (should be negative, ~-9.8)")
print(f"∂(x_ddot)/∂u = {jac_np[0, 4]:.4f} (should be positive, ~1)")
print(f"∂(theta_ddot)/∂x = {jac_np[1, 0]:.4f} (should be ~0)")
print(f"∂(theta_ddot)/∂theta = {jac_np[1, 2]:.4f} (should be positive, ~39.2)")
print(f"∂(theta_ddot)/∂u = {jac_np[1, 4]:.4f} (should be negative, ~-2)")

# The FNODE outputs [x_ddot, theta_ddot] based on PNODE format
# The state for MPC is [theta, x, theta_dot, x_dot] and control is [u]
# So we need to rearrange the Jacobian to match MPC state ordering

# First, set up the continuous-time state space form
# State: [theta, x, theta_dot, x_dot]
# The dynamics are:
# d/dt [theta]     = [theta_dot]
# d/dt [x]         = [x_dot]
# d/dt [theta_dot] = f1(theta, x, theta_dot, x_dot, u)  <- from FNODE
# d/dt [x_dot]     = f2(theta, x, theta_dot, x_dot, u)  <- from FNODE

# Extract partial derivatives from Jacobian
# FNODE input: [x, x_dot, theta, theta_dot, u]
# FNODE output: [x_ddot, theta_ddot]
# MPC state: [theta, x, theta_dot, x_dot]

# Need to rearrange Jacobian columns to match MPC state ordering
# From [x, x_dot, theta, theta_dot] to [theta, x, theta_dot, x_dot]
# Reorder indices: [2, 0, 3, 1]

# Construct continuous-time A matrix
Ac = np.zeros((4, 4))
Ac[0, 2] = 1.0  # d(theta)/dt = theta_dot
Ac[1, 3] = 1.0  # d(x)/dt = x_dot

# For theta_ddot (from jac_np[1,:] which is second output)
# Rearrange: ∂(theta_ddot)/∂[theta, x, theta_dot, x_dot]
Ac[2, 0] = jac_np[1, 2]  # ∂(theta_ddot)/∂theta
Ac[2, 1] = jac_np[1, 0]  # ∂(theta_ddot)/∂x
Ac[2, 2] = jac_np[1, 3]  # ∂(theta_ddot)/∂theta_dot
Ac[2, 3] = jac_np[1, 1]  # ∂(theta_ddot)/∂x_dot

# For x_ddot (from jac_np[0,:] which is first output)
# Rearrange: ∂(x_ddot)/∂[theta, x, theta_dot, x_dot]
Ac[3, 0] = jac_np[0, 2]  # ∂(x_ddot)/∂theta
Ac[3, 1] = jac_np[0, 0]  # ∂(x_ddot)/∂x
Ac[3, 2] = jac_np[0, 3]  # ∂(x_ddot)/∂theta_dot
Ac[3, 3] = jac_np[0, 1]  # ∂(x_ddot)/∂x_dot

# Construct continuous-time B matrix
Bc = np.zeros((4, 1))
Bc[2, 0] = jac_np[1, 4]  # effect of u on theta_ddot
Bc[3, 0] = jac_np[0, 4]  # effect of u on x_ddot

print("\nContinuous-time linearization:")
print("Ac =")
print(Ac)
print("\nBc =")
print(Bc)

# Convert to discrete-time using forward Euler approximation
# x[k+1] = x[k] + dt * (Ac * x[k] + Bc * u[k])
# x[k+1] = (I + dt * Ac) * x[k] + (dt * Bc) * u[k]
I = np.eye(4)

# Also compute analytical linearization for comparison
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [(m+M)*g/l/m, 0, 0, 0], [-m*g/M, 0, 0, 0]])
B = np.array([[0], [0], [-1/l/M], [1/M]])

print("Analytical A:")
print(A)
print("Calculated A (Ac):")
print(Ac)
print("Analytical B:")
print(B)
print("Calculated B (Bc):")
print(Bc)

# Debug: Check if linearization is reasonable
print("\nLinearization check:")
print(f"Ac eigenvalues: {np.linalg.eigvals(Ac)}")
print(f"Is Ac stable? {np.all(np.abs(np.linalg.eigvals(Ac)) < 1.1)}")  # Allow small instability

# MPC Parameters
IC = np.array([np.pi/6, 1, 0, 0])  # Initial condition: [theta, x, theta_dot, x_dot]
zmax, zmin = 10, -10
umax, umin = 50, -50
thetamax, thetamin = np.pi/2, -np.pi/2
h = dt  # Use same time step
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
    """Analytical dynamics for comparison"""
    return A.dot(x) + B.flatten()*u

def force_cp_ext(bodys, u, model=None):
    """
    Cart-pole dynamics with external force
    bodys: tensor with shape (2,2) - [[x, x_dot], [theta, theta_dot]]
    u: external force (tensor)
    Returns: [x_ddot, theta_ddot]
    """
    m1 = 1  # Cart mass
    m2 = 1  # Pole mass
    l = 0.5  # Pole length
    g = 9.8  # Gravity
    
    # Extract states
    x_pos = bodys[0, 0]
    x_dot = bodys[0, 1]
    theta = bodys[1, 0]
    theta_dot = bodys[1, 1]
    
    # Convert u to scalar if needed
    u_scalar = u.item() if hasattr(u, 'item') else u
    
    # Euler-Lagrange equations
    x_ddot_num = -m2 * g * torch.sin(theta) * torch.cos(theta) - m2 * l * theta_dot ** 2 * torch.sin(theta) + u_scalar
    x_ddot_den = m1 + m2 * (1 - torch.cos(theta) ** 2)
    x_ddot = x_ddot_num / x_ddot_den
    
    theta_ddot_num = (g * torch.sin(theta) * (m1 + m2) + m2 * l * theta_dot ** 2 * torch.sin(theta) * torch.cos(theta)) - u_scalar * torch.cos(theta)
    theta_ddot_den = l * (m1 + m2 * (1 - torch.cos(theta) ** 2))
    theta_ddot = theta_ddot_num / theta_ddot_den
    
    return torch.tensor([x_ddot, theta_ddot], dtype=torch.float32, device=device)

def midpoint_control_single_step(bodys, external, force_function, time_step, device=device, model=None):
    """
    Midpoint integration following PNODE approach
    bodys: tensor with shape (2,2) - [[x, x_dot], [theta, theta_dot]]
    external: control input tensor
    force_function: function to compute accelerations
    time_step: dt
    """
    positions = bodys[:, 0]
    velocities = bodys[:, 1]
    
    # Midpoint integration
    half_time_step = time_step / 2
    mid_positions = positions + half_time_step * velocities
    
    # Create midpoint state
    mid_states = torch.zeros_like(bodys)
    mid_states[:, 0] = mid_positions
    mid_states[:, 1] = velocities
    
    # Get acceleration at midpoint
    mid_acceleration = force_function(mid_states, external, model=model)
    
    # Update velocities using midpoint acceleration
    mid_velocities = velocities + half_time_step * mid_acceleration
    
    # Update positions and velocities
    new_positions = positions + time_step * mid_velocities
    new_velocities = velocities + time_step * mid_acceleration
    
    # Form the output tensor
    new_bodys = torch.stack((new_positions, new_velocities), dim=1)
    
    return new_bodys

# Main MPC loop
for k in range(len(t)-1):
    if k % 1 == 0:
        X = cp.Variable((4, N+1))
        U = cp.Variable((1, N))
        constraints = [X[:, 1:N+1] == dt*(Ac@X[:, 0:N] + Bc@U) + X[:, 0:N], X[:, 0] == init, X[:, N] == 0,
                       cp.max(X[1, :]) <= zmax, cp.min(X[1, :]) >= zmin,
                       cp.max(X[0, :]) <= thetamax, cp.min(X[0, :]) >= thetamin,
                       cp.max(U) <= umax, cp.min(U) >= umin]
        objective = cp.Minimize(cp.norm(cp.vstack([Qhalf@X[:, 0:N], Rhalf@U]), 'fro'))
        prob = cp.Problem(objective, constraints)
        
        # Solve with error handling and increased tolerance
        try:
            # Try with default settings first
            prob.solve(solver=cp.SCS, verbose=False)
            
            # If that fails, try with relaxed tolerances
            if prob.status not in ["optimal", "optimal_inaccurate"] or U.value is None:
                print(f"Initial solve failed at k={k}, status: {prob.status}")
                # Try with more relaxed settings
                prob.solve(solver=cp.SCS, verbose=False, eps=1e-3, max_iters=10000)
                
            # Check if solution is valid
            if prob.status not in ["optimal", "optimal_inaccurate"] or U.value is None:
                print(f"Warning at k={k}: Solver status = {prob.status}")
                print(f"Current state: {init}")
                # Use previous control or zero if failed
                if k > 0:
                    u[k] = u[k-1]
                else:
                    u[k] = 0.0
            else:
                u[k] = float(U.value[0, 0])
                umpc = float(U.value[0, 0])
        except Exception as e:
            print(f"Solver error at k={k}: {e}")
            print(f"Current state: {init}")
            # Use previous control or zero if error
            if k > 0:
                u[k] = u[k-1]
            else:
                u[k] = 0.0

    # Integrate using midpoint method with cart-pole dynamics
    # State order: [theta, x, theta_dot, x_dot]
    # Convert to bodys tensor format: [[x, x_dot], [theta, theta_dot]]
    bodys_tensor = torch.tensor([[x[1, k], x[3, k]], [x[0, k], x[2, k]]], dtype=torch.float32).to(device)
    external_force = torch.tensor([u[k]], dtype=torch.float32).to(device)
    
    # Use analytical cart-pole dynamics with midpoint integration
    state = midpoint_control_single_step(bodys_tensor, external_force, force_cp_ext, dt)
    cpu = state.cpu().detach().numpy().reshape(4, 1)
    
    # Extract new state: cpu format is [[x, x_dot], [theta, theta_dot]]
    x[:, k + 1] = np.array([cpu[2, 0], cpu[0, 0], cpu[3, 0], cpu[1, 0]])
    init = x[:, k + 1]

# Save full 10-second results
t_full = t
x_full = x
u_full = u[:-1]  # Exclude last u since it's not used

# Calculate the loss along the trajectory
loss_trajectory = np.zeros(len(t))
for i in range(len(t)):
    loss_trajectory[i] = np.sum(x[:, i]**2) + u[i]**2 if i < len(t)-1 else np.sum(x[:, i]**2)

print("loss_trajectory", loss_trajectory)

# Save results to mpc_cartpole directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results_fnode")
os.makedirs(results_dir, exist_ok=True)

# Save full 10-second data for plotting
np.savetxt(os.path.join(results_dir, "times.txt"), t_full, fmt="%.4f")
np.savetxt(os.path.join(results_dir, "states.txt"), x_full.T, fmt="%.6f",
           header="theta x theta_dot x_dot")
np.savetxt(os.path.join(results_dir, "controls.txt"), u_full, fmt="%.6f")

# Save summary
with open(os.path.join(results_dir, "summary.txt"), "w") as f:
    f.write("MPC Cart-Pole Control - FNODE\n")
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