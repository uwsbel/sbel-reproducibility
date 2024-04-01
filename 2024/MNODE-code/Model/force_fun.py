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
from functools import partial
from torch.autograd.functional import jacobian, hessian
import autograd
from utils import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is "+str(device))
#
def force_sms(bodys,model=None):
    mass=10
    #print("bodys shape is "+str(bodys.shape))
    return_value=torch.tensor([-50 * bodys[:,0]], dtype=torch.float32, device=device) / mass
    #print("return value shape is "+str(return_value.shape))
    #print("return value is "+str(return_value))
    #print("return value shape is "+str(return_value.shape))
    return return_value

def analytic_sms(bodys, dt, model=None):
    """
        Calculate the analytic solution of the single mass spring system.

        Given the initial condition, calculate the analytic solution at next time step.
        Here bodys is a tensor with shape (1, 2), the first column is the position,
        the second column is the velocity.

        Args:
        - bodys (torch.Tensor): A tensor of shape (1, 2) where the first column is position and
          the second column is velocity.
        - dt (float): Time step to compute the next state.
        - model (optional): Not used in this function, but can be added for future extensions.

        Returns:
        - torch.Tensor: A tensor of shape (1, 2) representing the new position and velocity.
        """

    # Constants for the mass-spring system (can be adapted as needed)
    k = torch.tensor(50.0, device=device)  # spring constant
    m = torch.tensor(10.0, device=device)  # mass

    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, device=device)

    # Extracting the initial position and velocity from the tensor
    x0 = bodys[0, 0]
    v0 = bodys[0, 1]

    # Calculate angular frequency
    omega = torch.sqrt(torch.tensor([k / m])).to(device)

    # Calculate amplitude and phase angle based on initial conditions
    A = torch.sqrt(x0 ** 2 + (v0 / omega) ** 2)
    phi = torch.atan2(-v0, omega * x0)

    # Compute position and velocity at time dt
    x1 = A * torch.cos(omega * dt + phi)
    v1 = -A * omega * torch.sin(omega * dt + phi)

    # Returning the new position and velocity as a tensor of shape (1, 2)
    return torch.tensor([[x1, v1]])
def force_smsd(bodys,model=None):
    #print("enter force_smsd function")
    mass=10
    return_value=torch.tensor(-50 * bodys[:,0] - 0.1 * bodys[:,0] ** 3 - 2 * bodys[:,1], dtype=torch.float32, device=device) / mass
    #print("return value shape is "+str(return_value.shape))
    return return_value
def lagrangian_sms(bodys):
    #print("enter lagrangian_sms function")
    mass=10
    k=50
    return_value=torch.tensor([0.5*mass*bodys[:,1]**2-0.5*k*bodys[:,0]**2],dtype=torch.float32,device=device)
    #print("return value shape is "+str(return_value.shape))
    return return_value
def get_xt(lagrangian, t, x):
    n = x.shape[0]//2
    #print("x shape",x.shape,"n is",n)
    xv = torch.autograd.Variable(x, requires_grad=True)
    #print("xv shape",xv.shape)
    tq, tqt = torch.split(xv, n, dim=0)
    A = torch.inverse(hessian(lagrangian, xv, create_graph=True)[n:, n:])
    B = jacobian(lagrangian, xv, create_graph=True)[:n]
    C = hessian(lagrangian, xv, create_graph=True)[n:, :n]
    tqtt = A @ (B - C @ tqt)
    #print("tqtt shape",tqtt.shape,"tq shape",tq.shape,"A shape",A.shape,"B shape",B.shape,"C shape",C.shape)
    xt = torch.cat([tqt, tqtt])
    return xt
def get_qdtt(q, qt, m=10, k=50):

    #print("q shape",q.shape)
    qdtt = np.zeros_like(q)
    qdtt[:, 0] = -k*q[:, 0]/m
    return qdtt
def get_xt_anal(x, t):
    print("x shape",x.shape)
    d = np.zeros_like(x)
    d[:,:, :1] = x[:,:, 1:]
    d[:,:, 1:] = get_qdtt(x[:,:, :1], x[:,:, 1:])

    # print(x, d)
    return d
def hamiltonian_fn(coords):
    q, p = np.split(coords, 2)
    H = 0.05*p ** 2 + 25*q ** 2  #  corrsponds to m = 10 and k = 50
    return H
def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S
def get_trajectory(t_span=[0, 3], timescale=100, y0=None, noise_std=0, **kwargs):

    t_span = [0, 30]
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    if y0 is None:
        y0 = np.array([1,0])

    spring_ivp = scipy.integrate.solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt, 2)

    # add noise
    #q += np.random.randn(*q.shape) * noise_std
    #p += np.random.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t_eval
def get_dataset(seed=0, samples=1, test_split=0.1, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs, ts = [], [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(noise_std=0.0,  **kwargs)
        xs.append(np.stack([x, y]).T)
        dxs.append(np.stack([dx, dy]).T)
        ts.append(t)
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['t'] = np.concatenate(ts).squeeze()
    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
def lnn_xt_anal(x, t):
    #print("x shape",x.shape)
    d = np.zeros_like(x)
    d[:, :1] = x[:, 1:]
    d[:, 1:] = get_qdtt(x[:, :1], x[:, 1:])
    return d
def total_energy_sms(bodys):
    #the energy of the single mass spring system, it returns a tensor
    mass=10
    k=50
    return_value=torch.tensor([0.5*mass*bodys[:,1]**2+0.5*k*bodys[:,0]**2],dtype=torch.float32,device=device)
    return return_value
def energy_sms(bodys):
    #the energy of the single mass spring system, it returns a tensor
    mass=10
    k=50
    return_value=torch.tensor([0.5*mass*bodys[:,:,1]**2+0.5*k*bodys[:,:,0]**2],dtype=torch.float32,device=device)
    #reshape the return value to a (num_step ) shape numpy array
    return_value=return_value.reshape((return_value.shape[1]))
    #convert the return value to a numpy array
    return_value=return_value.cpu().numpy()


    return return_value
def force_tmsd(bodys,model=None,mass=None):
    mass=torch.tensor([100,10,1],dtype=torch.float32,device=device)
    force = torch.zeros((3),dtype=torch.float32,device=device)
    force[2] = -50*(bodys[2,0]-bodys[1,0])-2*(bodys[2,1]-bodys[1,1])
    force[1] = -50*(bodys[1,0]-bodys[0,0])-2*(bodys[1,1]-bodys[0,1])-force[2]
    force[0] = -force[1]-50*bodys[0,0]-2*bodys[0,1]
    #print("force shape is "+str(force.shape))
    return force/mass
def force_crank(bodys,model=None):
    r_constant=2
    #convert the r_constant to a tensor
    r_constant=torch.tensor([r_constant],dtype=torch.float32,device=device)
    #print("bodys shape is "+str(bodys.shape))
    t= bodys[:, 2]
    #print("bodys shape is "+str(bodys.shape))
    alpha=r_constant * torch.sin(r_constant * t)
    #print("return value shape is "+str(return_value.shape))
    return torch.tensor([alpha], dtype=torch.float32, device=device)
def simulate_slider_crank_dynamic2(theta_0, omega_initial, t_values, r_constant):
    r=1
    l=4
    dt=0.01
    theta_values = np.zeros_like(t_values)
    omega_values = np.zeros_like(t_values)
    x_values = np.zeros_like(t_values)
    dxdt_values = np.zeros_like(t_values)
    theta_values[0] = theta_0
    omega_values[0] = omega_initial
    x_values[0] = r + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_0) ** 2)
    dxdt_values[0] = omega_initial * r * np.sin(theta_0)
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        # Dynamic angular acceleration
        alpha = r_constant * np.sin(r_constant * t)
        # Update angular velocity and angle using the dynamic angular acceleration
        omega_values[i] = omega_values[i - 1] + alpha * dt
        theta_values[i] = theta_values[i - 1] + omega_values[i] * dt
        # Compute the slider's position and velocity
        x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
        dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
    return theta_values, omega_values, x_values, dxdt_values
def dH_dq_smp(bodys, m=10, k1=50, model=None):
    """
    Compute the derivative of the potential energy with respect to the generalized position.
    For the single-mass-spring system: dV/dq = k1*q + k2*q^3
    """
    # Extracting generalized coordinates (position) from bodys tensor
    q = bodys[:, 0]
    #print("q shape is "+str(q.shape))
    return k1 * q
def dH_dp_smp(bodys, m=10,model=None):
    """
    Compute the derivative of the kinetic energy with respect to the generalized momentum.
    For the single-mass-spring system: dT/dp = p/m
    """
    # Extracting generalized momentum from bodys tensor
    p = bodys[:, 1]
    #print("p shape is "+str(p.shape))
    return p / m
def double_pendulum_derivs(t, state):
    G = 9.8  # acceleration due to gravity, in m/s^2
    L1 = 1.0  # length of pendulum 1 in m
    L2 = 1.0  # length of pendulum 2 in m
    L = L1 + L2  # maximal length of the combined pendulum
    M1 = 1.0  # mass of pendulum 1 in kg
    M2 = 1.0
    # unpack the state vector
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)
    dydx[2] = state[3]
    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)
    return dydx
def double_pendulum_energy(bodys, M1=1.0, M2=1.0, L1=1.0, L2=1.0, G=9.8):
    """
    Calculate the energy of the current state of the double pendulum.
    """
    # bodys is a tensor with shape ( num_bodys, 2)
    # For the last dimension, the first column is the angular, the second column is the angular velocity
    # read the thetas and omegas from bodys tensor
    theta1 = bodys[0, 0]
    theta2 = bodys[1, 0]
    theta1_dot = bodys[0, 1]
    theta2_dot = bodys[1, 1]
    PE1 = M1 * G * (L1 - L1 * np.cos(theta1))
    PE2 = M2 * G * (L1 + L2 - L1 * np.cos(theta1) - L2 * np.cos(theta2))

    # Kinetic energy
    KE1 = 0.5 * M1 * (L1 * theta1_dot) ** 2
    KE2 = 0.5 * M2 * ((L1 * theta1_dot) ** 2 + (L2 * theta2_dot) ** 2 + 2 * L1 * L2 * theta1_dot * theta2_dot * np.cos(
        theta1 - theta2))
    return PE1 + PE2 + KE1 + KE2

def double_pendulum_thetatoxyv(bodys,L1=1.0, L2=1.0):
    # bodys is a tensor with shape (num_step, num_bodys, 2)
    # For the last dimension, the first column is the angular, the second column is the angular velocity
    # read the thetas and omegas from bodys tensor
    theta1 = bodys[:, 0, 0]
    theta2 = bodys[:, 1, 0]
    theta1_dot = bodys[:, 0, 1]
    theta2_dot = bodys[:, 1, 1]
    x1 = L1 * torch.sin(theta1)
    y1 = -L1 * torch.cos(theta1)
    x2 = x1 + L2 * torch.sin(theta2)
    y2 = y1 - L2 * torch.cos(theta2)
    x1_dot = L1 * theta1_dot * torch.cos(theta1)
    y1_dot = L1 * theta1_dot * torch.sin(theta1)
    x2_dot = x1_dot + L2 * theta2_dot * torch.cos(theta2)
    y2_dot = y1_dot + L2 * theta2_dot * torch.sin(theta2)
    #make the output tensor with shape (num_step, num_bodys*2, 2),2 for x and y, 2 for velocity and acceleration,
    #the first num_bodys*2 is for x and y, the second num_bodys*2 is for velocity and acceleration
    output = torch.zeros((bodys.shape[0],4,2),dtype=torch.float32,device=device)
    output[:,0,0] = x1
    output[:,1,0] = y1
    output[:,2,0] = x2
    output[:,3,0] = y2
    output[:,0,1] = x1_dot
    output[:,1,1] = y1_dot
    output[:,2,1] = x2_dot
    output[:,3,1] = y2_dot
    #print("output shape is "+str(output.shape))
    return output

def total_energy_tmsd(bodys):
    # Given parameters
    mass = torch.tensor([100, 10, 1], dtype=torch.float32)
    k = 50  # spring constant
    # Kinetic energy for each body
    KE = 0.5 * mass * bodys[:, 1] ** 2
    # Potential energy due to springs
    # Spring between third and second body
    PE_32 = 0.5 * k * (bodys[2, 0] - bodys[1, 0]) ** 2
    # Spring between second and first body
    PE_21 = 0.5 * k * (bodys[1, 0] - bodys[0, 0]) ** 2
    # Spring for the first body (assuming its other end is fixed)
    PE_1 = 0.5 * k * bodys[0, 0] ** 2
    # Total potential energy
    PE = PE_32 + PE_21 + PE_1
    # Total energy
    total_E = torch.sum(KE) + PE
    return total_E
def total_energy_sms_symplectic(bodys, m=10, k1=50, k2=0.1):
    """
    Compute the total energy of the single-mass-spring system. Generalized coordinates (position) and momenta (momentum)
    """
    # Extracting generalized coordinates (position) and momenta (momentum) from bodys tensor
    q = bodys[:, 0]
    p = bodys[:, 1]

    # Kinetic energy
    KE = 0.5 * p ** 2/m

    # Potential energy
    PE = 0.025 * q ** 4 + 25 * q ** 2

    # Total energy
    total_E = torch.sum(KE + PE)
    #print("total_E is "+str(total_E))
    #print("KE is "+str(KE),"PE is "+str(PE))
    #print("displacement is "+str(q),"momentum is "+str(p))

    return total_E