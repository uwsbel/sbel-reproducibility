#
# BSD 3-Clause License
#
# Copyright (c) 2022 University of Wisconsin - Madison
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#

#// =============================================================================
#// Author: Harry Zhang
#// =============================================================================

import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import time
from casadi import *

def mpc_wpts_solver(e,u,vel,vel_ref,h1=0.5,h2=0.8):
    """This function is the mpc solver for error based tracking problem
        Use: [throttle, steering, eta] = mpc_wpts_solver(error state, current control input, current velocity of the vehicle, reference velocity)
    """
    x0 = np.array(e)
    u0 = np.array(u)

    # parameters related to the vehicle dynamics
    r_wheel = 0.08451952624
    i_wheel = 1e-3
    gamma = 1/3
    tau_0 = 0.09
    omega_0 = 161.185
    c1 = 1e-4
    l_car = 0.5
    nsim = 1
    delta_t = 0.1
    start_time = time.time()
    for i in range(nsim):

        #time dependent variables in Ad and Bd matrices
        alpha = u0[0]
        delta = u0[1] * 0.6
        eta = u0[2]

        e1 = x0[0]
        e2 = x0[1]
        e3 = x0[2]
        #reference/target point
        xr = np.array([0, 0, 0.,0, 0])

            # Discrete time model of a vehicle
        Ad = sparse.csc_matrix([
            [1.0,   vel*np.tan(delta)/l_car*delta_t,  -vel_ref*np.sin(e3)*delta_t,  0.0,  0.0],
            [-vel*np.tan(delta)/l_car*delta_t,   1.0,  vel_ref*np.cos(e3)*delta_t,  0.0,  0.0 ],
            [0.,   0.,  1.0,  0,  0],
            [0.,   0.,  0.,  1.0-(c1*omega_0+tau_0/(i_wheel*omega_0))*delta_t,  0.0],
            [0.,   0.,  0.,  0.0,  1.0]
        ])
        #Ad = Ad * delta_t + sparse.eye(4)

        Bd = sparse.csc_matrix([
                [0.,  vel*e2/(l_car*np.cos(delta)*np.cos(delta))*delta_t,  0],
                [0.,  -vel*e1/(l_car*np.cos(delta)*np.cos(delta))*delta_t,  0],
                [0.,  -vel/(l_car*np.cos(delta)*np.cos(delta))*delta_t,  0],
                [-gamma*r_wheel*tau_0/i_wheel*delta_t,  0,  0],
                [0.,  -vel*h2*delta_t, -vel *h1 * cos(eta)*delta_t]
                ])

        [nx, nu] = Bd.shape

        # Constraints
        #u0 = 10.5916
        umin = np.array([0.0,-0.6, -0.5]) 
        umax = np.array([1, 0.6,  0.5]) 
        xmin = np.array([-np.inf,-np.inf,-np.inf,-np.inf, -np.inf])
        xmax = np.array([np.inf, np.inf, np.inf,np.inf, np.inf])
        # xmin = np.array([-0.75,-0.75,-np.inf,-np.inf])
        # xmax = np.array([0.75, 0.75, np.inf,np.inf])
        # Objective function
        Q = sparse.diags([0.0, 5500., 1500., 300., 500])
        QN = Q
        R = sparse.diags([10., 1000., 5])



        # Prediction horizon
        N = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                    np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True,max_iter=10000)

        # Simulate in closed loop


            # Solve
        res = prob.solve()

            # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

            # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        u0 = ctrl
        x0 = Ad.dot(x0) + Bd.dot(ctrl)

            # Update initial state
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)
    # scale back the steering command
    ctrl[1] = ctrl[1] * 1.6667
    # ctrl[2] = ctrl[2] * 2.0
    print('controls:',ctrl)
    return ctrl 

def mpc_wpts_solver_hang2():
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

    # xr = np.array([0, 0, 0.,0])
    nx = 4
    nu = 2

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')
    x4 = MX.sym('x4')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2

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

    x = vertcat(x1, x2, x3, x4)
    u = vertcat(alpha, delta)
    ode = vertcat(dx1, dx2, dx3, dx4)
    f = Function('f', [x,u], [ode])

    # Prediction horizon
    N = 10
    dae = {'x':x, 'p':u, 'ode':f(x,u)}
    intg_opt = {'tf':delta_t, 'simplify': True, 'number_of_finite_elements':4}
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
    Q = np.diag([3000.0, 5500., 0., 300.])
    QN = 10 * Q
    R = np.diag([10., 1000.])
    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
    # opti.subject_to([u<=1,u>=-1])
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=1,u[0,:]>=0, u[1,:]<=1,u[1,:]>=-1])
    opti.subject_to(x[:,0]==p)

    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)

    opti.solver('ipopt', p_opts, s_opts)
    NMPC = opti.to_function('M', [p,xref,uref], [u[:,0]],['p','xref','uref'],['u0'])

    # umpc = NMPC(e0, x_ref, u_ref)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    return NMPC