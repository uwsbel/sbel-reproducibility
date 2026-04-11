from casadi import *
from scipy.optimize import minimize
import cvxpy as cp
import numpy as np
import time 

def mpc_solver(model_type = 'M1'):
    # parameters related to the vehicle dynamics
    r_wheel = 0.08451952624
    i_wheel = 1e-3
    gamma = 1/3
    tau_0 = 0.09
    omega_0 = 161.185
    c1 = 1e-4
    l_car = 0.5
    delta_t = 0.1 # Sampling time
    beta = 0.6
    c0 = 0.02

    h1 = -0.1
    h2 = -0.1
    # h1 = -0.50469456
    # h2 =  -0.45387264

    # xr = np.array([0, 0, 0.,0])
    nx = 5
    nu = 3

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')
    x4 = MX.sym('x4')
    x5 = MX.sym('x5')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2
    eta = MX.sym('eta') # Input #3

    # dx1 = v*tan(delta*beta) / l_car + vel_ref * cos(e3) - v
    # dx2 = -v*tan(delta*beta) * e1 /l_car + vel_ref * sin(e3)
    # dx3 = -v*tan(delta*beta) / l_car
    # dx4 = tau_0 * r_wheel * gamma / i_wheel * (0 - alpha) - e4 * (c1*omega_0+tau_0)/(i_wheel*omega_0)
    dx1 = x4 * cos(x3)
    dx2 = x4 * sin(x3)
    dx3 = x4 * tan(delta*beta) / l_car

    wm = x4 / (r_wheel*gamma)
    f1 = -tau_0 * wm /omega_0 + tau_0
    Tav = alpha * f1 - c1*wm -c0
    dx4 = Tav * gamma / i_wheel * r_wheel

    if model_type == 'M1':
        print("Selected Model: M1")
        dx5 = x4 * (h1* sin(eta) + h2 * beta) # M1
    elif model_type == 'M2':
        print("Selected Model: M2")
        #dx5 =h1* eta + h2 * beta # M2
        dx5 =-0.1* eta - 0.1 * beta # M2
    elif model_type == 'M3':
        print("Selected Model: M3")
        dx5 =h1* eta  # M3
    else:
        print("Invalid model type")
        return


    x = vertcat(x1, x2, x3, x4, x5)
    u = vertcat(alpha, delta, eta)
    ode = vertcat(dx1, dx2, dx3, dx4, dx5)
    f = Function('f', [x,u], [ode])

    # Prediction horizon
    N = 10
    dae = {'x':x, 'p':u, 'ode':f(x,u)}
    intg_opt = {'tf':delta_t, 'simplify': True, 'number_of_finite_elements':5}
    intg = integrator('intg', 'rk', dae, intg_opt)
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x,u], [x_next], ['x', 'u'], ['x_next'])
    opti = Opti()
    x = opti.variable(nx, N+1)
    u = opti.variable(nu,N)
    p = opti.parameter(nx,1)
    xref = opti.parameter(nx,N+1)
    uref = opti.parameter(nu,N)

    # Q = np.diag([3000.0, 5500., 3500., 300.])
    Q = np.diag([3000.0, 5500., 0., 300., 80])
    QN = 10 * Q
    R = np.diag([10., 1000., 10])
    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
    # opti.subject_to([u<=1,u>=-1])
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=1,u[0,:]>=0, u[1,:]<=1,u[1,:]>=-1, u[2,:]<=0.5,u[2,:]>=-0.5])
    opti.subject_to(x[:,0]==p)

    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)

    opti.solver('ipopt', p_opts, s_opts)
    NMPC = opti.to_function('M', [p,xref,uref], [u[:,0]],['p','xref','uref'],['u0'])

    # umpc = NMPC(e0, x_ref, u_ref)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    return NMPC


if __name__ == '__main__':
    current_state = [0.0, 0.0, 0.0, 0.0, -1.0]
    ref_state = [0.6, 0.3, 0.0, 0.0, 0.0]
    ref_control = [0.0, 0.0, 0.0]
    NMPC = mpc_solver('M3')
    alpha, beta, eta = np.array(NMPC(current_state,ref_state,ref_control)).squeeze() 
    print("Computed Control:", alpha, beta, eta)

