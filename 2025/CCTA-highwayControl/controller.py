import numpy as np
import scipy as sp
from scipy import sparse
import time
from casadi import *

def error_state(veh_state,ref_traj,lookahead=3.0):
        
        x_current = veh_state[0]
        y_current = veh_state[1]
        theta_current = veh_state[2]
        v_current = 1.0
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(ref_traj[:,1])))
        for i in range(len(ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*lookahead-ref_traj[i][0])**2+(y_current+np.sin(theta_current)*lookahead-ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, 0]

        return error_state


def vir_veh_controller(veh_state,ref_traj,lookahead=3.0):
        
        x_current = veh_state[0]
        y_current = veh_state[1]
        theta_current = veh_state[2]
        v_current = veh_state[3]
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(ref_traj[:,1])))
        for i in range(len(ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*lookahead-ref_traj[i][0])**2+(y_current+np.sin(theta_current)*lookahead-ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, ref_state_current[3]-v_current]
        
        # pid for steering
        steering = sum([x * y for x, y in zip(error_state, [0.02176878 , 0.72672704 , 0.78409284 ,-0.0105355 ])])
        # pid for throttle
        throttle = sum([x * y for x, y in zip(error_state, [0.37013526 ,0.00507144, 0.15476554 ,1.0235402 ])])
        
        # clip throttle and steering
        throttle = np.clip(throttle, 0, 1)
        steering = np.clip(steering, -1, 1)
        
        return [throttle,steering]
def mpc_wpts_solver_sedan3(SV_pos, p,xref,uref,delta_t = 0.1, N = 10):
    # parameters related to the vehicle dynamics
    l_car = 4.52
    beta = 0.626671  # 35.9 degree
    # beta = 1.117 # 64 degree


    # xr = np.array([0, 0, 0.,0])
    nx = 3
    nu = 2

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2

    dx1 = alpha * cos(x3)
    dx2 = alpha * sin(x3)
    dx3 = alpha * tan(delta*beta) / l_car

    x = vertcat(x1, x2, x3)
    u = vertcat(alpha, delta)
    ode = vertcat(dx1, dx2, dx3)
    f = Function('f', [x,u], [ode])

    # Prediction horizon
    dae = {'x':x, 'p':u, 'ode':f(x,u)}
    intg_opt = {'tf':delta_t, 'simplify': True, 'number_of_finite_elements':4}
    intg = integrator('intg', 'rk', dae, intg_opt)
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x,u], [x_next], ['x', 'u'], ['x_next'])
    opti = Opti()
    x = opti.variable(nx, N+1)
    u = opti.variable(nu,N)
    # p = opti.parameter(nx,1)

    # SV_pos = opti.parameter(2,5)
    # xref = opti.parameter(nx,N+1)
    # uref = opti.parameter(nu,N)

    # Q = np.diag([3000.0, 5500., 0.1])
    # QN = 5 * Q
    # R = np.diag([30., 100.])
    # r = 0.3
    # lbd = 300

    Q = np.diag([3000.0, 5500., 0.1])
    QN = 5 * Q
    R = np.diag([30., 100.])
    r = 0.3
    # lbd = 3000 # large for static penalty
    # lbd = 500 # good for static penalty
    # lbd = 10 # small for static penalty


    # lbd = 10 # small for state-dependent penalty
    # lbd = 50 # good for state-dependent penalty

    # lbd = 300 # good for varying weights

    # lbd = 100 # used for warning
    # lbd = 50
    r_cons = 20

    ind_sv = []
    d_list = []
    for i in range(SV_pos.shape[1]):
        dd = (p[0] - SV_pos[0,i])**2 + (p[1] - SV_pos[1,i])**2
        if dd <= r_cons*r_cons:
            ind_sv.append(i)
            d_list.append(dd)
    # print(f'd_list = {d_list}')

    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
        # for i in range(len(ind_sv)):
        # for i in range(SV_pos.shape[1]):
            # opti.subject_to((x[0,k] - SV_pos[0,i])**2 + (x[1,k] - SV_pos[1,i])**2 >=r**2)
            # cost += -lbd * np.exp(-np.sqrt(d_list[i])/r_cons) * ((x[0,k] - SV_pos[0,ind_sv[i]])**2 + (x[1,k] - SV_pos[1,ind_sv[i]])**2)
    for k in range(N-1):
        opti.subject_to(u[1,k+1] - u[1,k]<=0.05) # good
        opti.subject_to(u[1,k+1] - u[1,k]>=-0.05) # good 
        # opti.subject_to(u[1,k+1] - u[1,k]<= 0.2) 
        # opti.subject_to(u[1,k+1] - u[1,k]>=-0.2) 
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=35,u[0,:]>=0, u[1,:]<= 0.2,u[1,:]>=-0.2])
    opti.subject_to(x[:,0]==p)


    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)

    opti.solver('ipopt', p_opts, s_opts)
    # NMPC = opti.to_function('M', [SV_pos, p,xref,uref], [u[:,0]],['SV_pos','p','xref','uref'],['u0'])
    
    sol = opti.solve()
    ctrl = sol.value(u)
    x_mpc = sol.value(x)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    print(sol.stats()['return_status'])
    if sol.stats()['return_status'] == 'Infeasible_Problem_Detected':
        print(opti.debug.value)
        print(opti.debug.show_infeasibilities)

    return ctrl[:,0], x_mpc[:,1]
    
def mpc_wpts_solver_sedan():
    # parameters related to the vehicle dynamics
    l_car = 4.52
    delta_t = 0.1 # Sampling time
    beta = 0.6


    # xr = np.array([0, 0, 0.,0])
    nx = 3
    nu = 2

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2

    dx1 = alpha * cos(x3)
    dx2 = alpha * sin(x3)
    dx3 = alpha * tan(delta*beta) / l_car

    x = vertcat(x1, x2, x3)
    u = vertcat(alpha, delta)
    ode = vertcat(dx1, dx2, dx3)
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

    SV_pos = opti.parameter(2,5)
    xref = opti.parameter(nx,N+1)
    uref = opti.parameter(nu,N)

    Q = np.diag([3000.0, 5500., 0.1])
    QN = 10 * Q
    R = np.diag([10., 1000.])
    r = 0.3

    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
        for i in range(SV_pos.shape[1]):
            opti.subject_to((x[0,k] - SV_pos[0,i])**2 + (x[1,k] - SV_pos[1,i])**2 >=r**2)
    # opti.subject_to([u<=1,u>=-1])
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=35,u[0,:]>=0, u[1,:]<=1,u[1,:]>=-1])
    opti.subject_to(x[:,0]==p)


    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0, max_iter=10)

    opti.solver('ipopt', p_opts, s_opts)
    NMPC = opti.to_function('M', [SV_pos, p,xref,uref], [u[:,0]],['SV_pos','p','xref','uref'],['u0'])

    # umpc = NMPC(e0, x_ref, u_ref)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    return NMPC
def mpc_wpts_solver_sedan2(SV_pos, p,xref,uref,delta_t = 0.1, N = 10, lbd=50):
    # parameters related to the vehicle dynamics
    l_car = 4.52
    beta = 0.626671  # 35.9 degree
    # beta = 1.117 # 64 degree


    # xr = np.array([0, 0, 0.,0])
    nx = 3
    nu = 2

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2

    dx1 = alpha * cos(x3)
    dx2 = alpha * sin(x3)
    dx3 = alpha * tan(delta*beta) / l_car

    x = vertcat(x1, x2, x3)
    u = vertcat(alpha, delta)
    ode = vertcat(dx1, dx2, dx3)
    f = Function('f', [x,u], [ode])

    # Prediction horizon
    dae = {'x':x, 'p':u, 'ode':f(x,u)}
    intg_opt = {'tf':delta_t, 'simplify': True, 'number_of_finite_elements':4}
    intg = integrator('intg', 'rk', dae, intg_opt)
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x,u], [x_next], ['x', 'u'], ['x_next'])
    opti = Opti()
    x = opti.variable(nx, N+1)
    u = opti.variable(nu,N)
    # p = opti.parameter(nx,1)

    # SV_pos = opti.parameter(2,5)
    # xref = opti.parameter(nx,N+1)
    # uref = opti.parameter(nu,N)

    # Q = np.diag([3000.0, 5500., 0.1])
    # QN = 5 * Q
    # R = np.diag([30., 100.])
    # r = 0.3
    # lbd = 300

    Q = np.diag([3000.0, 5500., 0.1])
    QN = 5 * Q
    R = np.diag([30., 100.])
    r = 0.3
    # lbd = 3000 # large for static penalty
    # lbd = 500 # good for static penalty
    # lbd = 10 # small for static penalty


    # lbd = 10 # small for state-dependent penalty
    # lbd = 50 # good for state-dependent penalty

    # lbd = 300 # good for varying weights

    # lbd = 100 # used for warning
    # lbd = 50
    r_cons = 20

    ind_sv = []
    d_list = []
    for i in range(SV_pos.shape[1]):
        dd = (p[0] - SV_pos[0,i])**2 + (p[1] - SV_pos[1,i])**2
        if dd <= r_cons*r_cons:
            ind_sv.append(i)
            d_list.append(dd)
    # print(f'd_list = {d_list}')

    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
        for i in range(len(ind_sv)):
        # for i in range(SV_pos.shape[1]):
            # opti.subject_to((x[0,k] - SV_pos[0,i])**2 + (x[1,k] - SV_pos[1,i])**2 >=r**2)
            cost += -lbd * np.exp(-np.sqrt(d_list[i])/r_cons) * ((x[0,k] - SV_pos[0,ind_sv[i]])**2 + (x[1,k] - SV_pos[1,ind_sv[i]])**2)
    for k in range(N-1):
        opti.subject_to(u[1,k+1] - u[1,k]<=0.05) # good
        opti.subject_to(u[1,k+1] - u[1,k]>=-0.05) # good 
        # opti.subject_to(u[1,k+1] - u[1,k]<= 0.2) 
        # opti.subject_to(u[1,k+1] - u[1,k]>=-0.2) 
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=35,u[0,:]>=0, u[1,:]<= 0.2,u[1,:]>=-0.2])
    opti.subject_to(x[:,0]==p)


    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)

    opti.solver('ipopt', p_opts, s_opts)
    # NMPC = opti.to_function('M', [SV_pos, p,xref,uref], [u[:,0]],['SV_pos','p','xref','uref'],['u0'])
    
    sol = opti.solve()
    ctrl = sol.value(u)
    x_mpc = sol.value(x)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    print(sol.stats()['return_status'])
    if sol.stats()['return_status'] == 'Infeasible_Problem_Detected':
        print(opti.debug.value)
        print(opti.debug.show_infeasibilities)

    return ctrl[:,0], x_mpc[:,1]
    
def FindMinDistInd(veh_state, lookahead, ref_traj):
    while veh_state[2]<-np.pi:
        veh_state[2] = veh_state[2]+2*np.pi
    while veh_state[2]>np.pi:
        veh_state[2] = veh_state[2] - 2*np.pi
    N_traj = ref_traj.shape[0]
    x_current = veh_state[0]
    y_current = veh_state[1]
    theta_current = veh_state[2]
    dist = np.zeros((1,len(ref_traj[:,1])))
    for i in range(len(ref_traj[:,1])):
        dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*lookahead-ref_traj[i][0])**2+(y_current+np.sin(theta_current)*lookahead-ref_traj[i][1])**2
    index = dist.argmin()
    return index
def weighted_nearest_quantile(values, quantiles, sample_weights=None):
    """
    Compute weighted quantiles and return the nearest sample values.

    Parameters:
    - values: array-like, the data values.
    - quantiles: array-like, quantiles to compute, e.g., [0.25, 0.5, 0.75].
    - sample_weights: array-like, weights of the same length as `values`. If None, all weights are considered equal.

    Returns:
    - array-like, the nearest sample values corresponding to the quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weights is None:
        sample_weights = np.ones_like(values, dtype=np.float64)  # Ensure float
    else:
        sample_weights = np.array(sample_weights, dtype=np.float64)  # Ensure float

    # Sort values and weights together
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    sample_weights = sample_weights[sorted_indices]

    # Compute the cumulative sum of weights
    cumulative_weights = np.cumsum(sample_weights, dtype=np.float64)
    cumulative_weights /= cumulative_weights[-1]  # Normalize to [0, 1]

    # Find the nearest value for each quantile
    nearest_samples = []
    index = np.searchsorted(cumulative_weights, quantiles, side="left")
    return values[index]

def CPWarningSystem(sc, Q, Rmat, CP_data_f0, pred, SV_pos, eps):
    # tuple is (sc_gt, sc_pred)
    ct = 0
    gY = 10000.
    for i in range(SV_pos.shape[1]):
        for j in range(pred.shape[0]):
            gY = np.min((gY, sc(pred[j],SV_pos[:,i],Q,Rmat)))
    
    for dcp in CP_data_f0:
        if dcp[1] < gY:
            ct += 1
    q = (ct + 1) / (len(CP_data_f0) + 1)
    if q <= 1 - eps:
        return 1, gY
    else:
        return 0, gY

def mpc_wpts_solver_sedan_dob(SV_pos, p,xref,uref,hdxyvta,delta_t, N = 10):
    # parameters related to the vehicle dynamics
    l_car = 4.52
    # delta_t = 0.1 # Sampling time
    beta = 0.626671  # 35.9 degree
    # beta = 1.117 # 64 degree


    # xr = np.array([0, 0, 0.,0])
    nx = 3
    nu = 2

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')

    alpha = MX.sym('alpha') # Input #1
    delta = MX.sym('delta') # Input #2

    dx1 = alpha * cos(x3) + hdxyvta[0]
    dx2 = alpha * sin(x3) + hdxyvta[1]
    dx3 = alpha * tan(delta*beta) / l_car + hdxyvta[3]

    x = vertcat(x1, x2, x3)
    u = vertcat(alpha, delta)
    ode = vertcat(dx1, dx2, dx3)
    f = Function('f', [x,u], [ode])

    # Prediction horizon
    dae = {'x':x, 'p':u, 'ode':f(x,u)}
    intg_opt = {'tf':delta_t, 'simplify': True, 'number_of_finite_elements':4}
    intg = integrator('intg', 'rk', dae, intg_opt)
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x,u], [x_next], ['x', 'u'], ['x_next'])
    opti = Opti()
    x = opti.variable(nx, N+1)
    u = opti.variable(nu,N)
    # p = opti.parameter(nx,1)

    # SV_pos = opti.parameter(2,5)
    # xref = opti.parameter(nx,N+1)
    # uref = opti.parameter(nu,N)

    # Q = np.diag([3000.0, 5500., 0.1])
    # QN = 5 * Q
    # R = np.diag([30., 100.])
    # r = 0.3
    # lbd = 300

    Q = np.diag([3000.0, 5500., 0.1])
    QN = 5 * Q
    R = np.diag([30., 100.])
    r = 0.3
    # lbd = 3000 # large for static penalty
    # lbd = 500 # good for static penalty
    # lbd = 10 # small for static penalty


    # lbd = 10 # small for state-dependent penalty
    # lbd = 50 # good for state-dependent penalty

    # lbd = 300 # good for varying weights

    # lbd = 100 # used for warning
    lbd = 50
    r_cons = 20

    ind_sv = []
    d_list = []
    for i in range(SV_pos.shape[1]):
        dd = (p[0] - SV_pos[0,i])**2 + (p[1] - SV_pos[1,i])**2
        if dd <= r_cons*r_cons:
            ind_sv.append(i)
            d_list.append(dd)
    print(f'd_list = {d_list}')

    cost = 0
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))
        
        cost += (x[:,k] - xref[:,k]).T @ Q @ (x[:,k] - xref[:,k])
        cost += (u[:,k] - uref[:,k]).T @ R @ (u[:,k] - uref[:,k])
        for i in range(len(ind_sv)):
        # for i in range(SV_pos.shape[1]):
            # opti.subject_to((x[0,k] - SV_pos[0,i])**2 + (x[1,k] - SV_pos[1,i])**2 >=r**2)
            cost += -lbd * np.exp(-np.sqrt(d_list[i])/r_cons) * ((x[0,k] - SV_pos[0,ind_sv[i]])**2 + (x[1,k] - SV_pos[1,ind_sv[i]])**2)
    for k in range(N-1):
        opti.subject_to(u[1,k+1] - u[1,k]<=0.05) # good
        opti.subject_to(u[1,k+1] - u[1,k]>=-0.05) # good 
        # opti.subject_to(u[1,k+1] - u[1,k]<= 0.2) 
        # opti.subject_to(u[1,k+1] - u[1,k]>=-0.2) 
    cost += (x[:,N] - xref[:,N]).T @ QN @ (x[:,N] - xref[:,N])
    opti.minimize(cost)

    opti.subject_to([u[0,:]<=35,u[0,:]>=0, u[1,:]<=0.5,u[1,:]>=-0.5])
    opti.subject_to(x[:,0]==p)


    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)

    opti.solver('ipopt', p_opts, s_opts)
    # NMPC = opti.to_function('M', [SV_pos, p,xref,uref], [u[:,0]],['SV_pos','p','xref','uref'],['u0'])
    
    sol = opti.solve()
    ctrl = sol.value(u)
    x_mpc = sol.value(x)
    # ctrl = np.array(umpc).squeeze() @ np.diag([1,beta])
    print(sol.stats()['return_status'])
    if sol.stats()['return_status'] == 'Infeasible_Problem_Detected':
        print(opti.debug.value)
        print(opti.debug.show_infeasibilities)

    return ctrl[:,0], x_mpc[:,1]