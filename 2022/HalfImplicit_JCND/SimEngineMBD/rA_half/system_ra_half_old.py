# had some issues with self parameters 

import logging

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .gcons_ra_half import Constraints, DP1, DP2, CD, D, Body, ConGroup
from ..utils.physics import Z_AXIS, block_mat, R, skew, exp, SolverType
from ..utils.systems import read_model_file


class SystemRA_half:

    def __init__(self, bodies, constraints, solv_ord=1):
        self.bodies = bodies
        self.g_cons = constraints
        self.solver_type = SolverType.KINEMATICS
        # set to 2 to use 2nd order Martin Arnold's Lie for rotation and 2nd order BDF for translation. Has no effect for kinematics
        self.solver_order = solv_ord

        self.nc = self.g_cons.nc
        self.nb = self.g_cons.nb

        assert self.nb == len(self.bodies), "Mismatch on number of bodies"

        self.g_acc = np.zeros((3, 1))

        self.is_initialized = False

        # Set solver parameters
        self.h = 1e-5
        # set to None before, make tighter tolerance
        self.tol = 1e-8
        self.max_iters = 6
        self.k = 0

        # Set physical quantities
        self.M = np.zeros((3*self.nb, 3*self.nb))
        self.J = np.zeros((3*self.nb, 3*self.nb))
        
        self.M_inv = np.zeros((3*self.nb, 3*self.nb))
        self.J_inv = np.zeros((3*self.nb, 3*self.nb))
        
        self.F_ext = np.zeros((3*self.nb, 1))
        self.τ = np.zeros((3*self.nb, 1))

        # consraints and their jacobians, make sure dimension matches ....
        self.Φ = np.zeros((self.nc, 1))
        self.Φ_r = np.zeros((self.nc, 3*self.nb))
        self.Π = np.zeros((self.nc, 3*self.nb))
        self.test = np.zeros((self.nc, 3*self.nb))

        self.Φq = np.zeros((self.nc, 6*self.nb))

        # lagrange multipliers
        self.λ = np.zeros((self.nc, 1))
        # λ_hat = λ * step_size^2
        self.λ_hat = np.zeros((self.nc, 1))
        # theta_bar = omega_bar * step_size
        self.theta_bar = np.zeros((3*self.nb, 1))
        

        # Storage arrays for dynamics
        # acceleration level, evaluated after position level data converges
        self.ddr = np.zeros((3*self.nb, 1))
        self.dω = np.zeros((3*self.nb, 1))
        
        # J term is 3*nb by 1?
        self.J_term = np.zeros((3*self.nb, 1))
        
        self.r = np.zeros((3*self.nb, 1))
        self.ω = np.zeros((3*self.nb, 1))
        
        # Storage arrays for kinematics in previous step, and bn and cn used in half implicit
        self.dr = np.zeros((3*self.nb, 1))
        self.ω = np.zeros((3*self.nb, 1))
        self.bn = np.zeros((3*self.nb, 1))
        self.cn = np.zeros((3*self.nb, 1))
        

    def set_dynamics(self):
        if self.is_initialized:
            logging.warning(
                'Cannot change solver type on an initialized system')
        else:
            self.solver_type = SolverType.DYNAMICS

    def set_kinematics(self):
        if self.is_initialized:
            logging.warning(
                'Cannot change solver type on an initialized system')
        else:
            self.solver_type = SolverType.KINEMATICS

    @classmethod
    def init_from_file(cls, filename):
        file_info = read_model_file(filename)
        return cls(*process_system(*file_info))



    def set_g_acc(self, g=-9.81*Z_AXIS):
        self.g_acc = g

    def initialize(self):

        for body in self.bodies:
            body.F += body.m * self.g_acc

        self.M = np.diagflat([[body.m] * 3 for body in self.bodies])
        self.J = block_mat([body.J for body in self.bodies])
        
        self.M_inv = np.diagflat([[1/body.m] * 3 for body in self.bodies])
        self.J_inv = block_mat([body.J_inv for body in self.bodies])
        
        self.F_ext = np.vstack([body.F for body in self.bodies])

        if self.solver_type == SolverType.KINEMATICS:
            if self.nc == 6*self.nb:
                # Set tighter tolerance for kinematics
                self.tol = 1e-6 if self.tol is None else self.tol

                logging.info('Initializing system for kinematics')
            else:
                logging.warning('Kinematic system has nc ({}) < 6⋅nb ({}), running dynamics instead'.format(
                    self.nc, 6*self.nb))
                self.solver_type = SolverType.DYNAMICS

        if self.solver_type == SolverType.DYNAMICS:
            if self.nc > 6*self.nb:
                logging.warning('System is overconstrained')
            self.tol = 1e-3 if self.tol is None else self.tol
            
            # comment this out, does not seem like rA half needs it?
            # self.initialize_dynamics()

        # self.is_initialized = True

    def initialize_dynamics(self):

        logging.info('Initializing system for dynamics')

        t_start = 0

        # Compute initial values
        self.Φ_r = self.g_cons.get_phi_r(t_start)
        # for deubg
        self.Π = self.g_cons.get_pi(t_start)

        # Quantities for the right-hand side:
        # Fg is constant, defined above
        self.τ = np.vstack([body.get_tau() for body in self.bodies])
        γ = self.g_cons.get_gamma(t_start)

        G_ωω = block_mat([body.get_J_term(self.h) for body in self.bodies])
        G = np.block([[self.M, np.zeros((3*self.nb, 3*self.nb)), self.Φ_r.T], [np.zeros((3*self.nb, 3*self.nb)), G_ωω, self.Π.T],
                      [self.Φ_r, self.Π, np.zeros((self.nc, self.nc))]])

        g = np.block([[self.F_ext], [self.τ], [γ]])

        z = np.linalg.solve(G, g)

        for i, body in enumerate(self.bodies):
            body.ddr = z[3*i:3*(i+1)]
            body.dω = z[3*self.nb + 3*i:3*self.nb + 3*(i+1)]

            body.cache_rA_values()

        if self.solver_order == 2:
            self.dr_prevprev = np.zeros([len(self.bodies), 3])
            self.r_prevprev = np.zeros([len(self.bodies), 3])
            self.ω_prevprev = np.zeros([len(self.bodies), 3])
            self.ω0_prev = np.zeros([len(self.bodies), 3])

        self.λ = z[6*self.nb:]

    def do_step(self, i, t):
        if self.solver_type == SolverType.KINEMATICS:
            self.do_kinematics_step(t)
        else:
            self.do_dynamics_step(i, t)

    def do_dynamics_step(self, i, t):

        # assert self.is_initialized, "Cannot dyn_step before system initialization"

        # if i == 0:
        #     return

        self.g_cons.maybe_swap_gcons(t)

        Phi_r_old = self.g_cons.get_phi_r(t)
        Pi_old = self.g_cons.get_pi(t)


        # define quasi newton iteration matrix
        # G = np.block([[self.M, np.zeros((3*self.nb, 3*self.nb)), Phi_r_old.T],
        #               [np.zeros((3*self.nb, 3*self.nb)), self.J, Pi_old.T],
        #               [self.Φ_r, self.Π, np.zeros((self.nc, self.nc))]])

        # G_lu = lu_factor(G)

        for body in self.bodies:
            body.cache_rA_values() #r_prev, dr_prev, A_prev and ω_prev are assigned
            
        # populate r_prev and dr_prev array for the system    
        for j, body in enumerate(self.bodies):
            self.r[3*j:3*(j+1)] = body.r_prev
            self.dr[3*j:3*(j+1)] = body.dr_prev            
            J_term = skew(body.ω_prev) @ body.J @ body.ω_prev
            self.cn[3*j:3*(j+1)] = self.h**2 * (J_term - body.n_ω) - self.h * body.J @ body.ω_prev        
        # assemble bn values
        self.bn = -(self.M @ self.r + self.h * self.M @ self.dr + self.h ** 2 * self.F_ext)
        
        # initial guess of the iterative values
        for j, body in enumerate(self.bodies):
            body.r = body.r_prev + self.h * body.dr_prev + self.h**2 * body.ddr
            body.ω = body.ω_prev + self.h * body.dω
            body.A = body.A_prev @ exp(self.h * skew(body.ω))            
            self.r[3*j:3*(j+1)] = body.r
            self.theta_bar[3*j:3*(j+1)] = body.ω * self.h
 
        
        # # Compute values needed for the residual
        # self.Φ = self.g_cons.get_phi(t)
        # self.Φ_r = self.g_cons.get_phi_r(t)
        # self.Π = self.g_cons.get_pi(t)

        
        # todo: what is J???? in e2 figure out

        # Setup and do Newton-Raphson Iteration
        self.k = 0
        while True:
            # for j, body in enumerate(self.bodies):
                # TODO: updates specific to half-implicit
                # body.ω = body.ω_prev + self.h*body.dω
                # body.r = body.r_prev + self.h * body.dr
                # body.A = body.A_prev @ exp(self.h * skew(body.ω))

                # Conceptually these go lower, but this way we just loop over bodies once
                # self.r[3*j:3*(j + 1)] = body.r
                # self.ω[3*j:3*(j + 1)] = body.ω

            # solving for current r theta_bar and previous lambda_hat
            # Form right hand side e matrix g = [e0, e1, e2]
            e0 = self.M @ self.r + Phi_r_old.T @ self.λ_hat + self.bn
            e1 = self.J @ self.theta_bar + Pi_old.T @ self.λ_hat + self.cn
            e2 = self.g_cons.get_phi(t)
            e = np.block([[e0], [e1], [e2]])
            # print("e0 = {} {} {}".format(e0[0], e0[1], e0[2]))
            # print("e1 = {} {} {}".format(e1[0], e1[1], e1[2]))
            # print("e2 = {} {} {} {} {}".format(e2[0], e2[1], e2[2], e2[3], e2[4]))
            
            # update Newton matrix (use exact ones and see what happens)
            # pi_updated = self.g_cons.get_pi(t)
            # self.Π = self.g_cons.get_pi(t)
            # self.Φ_r = self.g_cons.get_phi_r(t)            
            # G = np.block([[self.M, np.zeros((3*self.nb, 3*self.nb)), Phi_r_old.T],
            #               [np.zeros((3*self.nb, 3*self.nb)), self.J, Pi_old.T],
            #               [self.Φ_r, self.Π, np.zeros((self.nc, self.nc))]])


            G = np.block([[self.M, np.zeros((3*self.nb, 3*self.nb)), Phi_r_old.T],
                          [np.zeros((3*self.nb, 3*self.nb)), self.J, Pi_old.T],
                          [Phi_r_old, Pi_old, np.zeros((self.nc, self.nc))]])


            # print("is pi being changed? {} {}".format(pi_updated[3][1], pi_updated[3][2]))
            ### end of wacky update

            G_lu = lu_factor(G)
            δ = lu_solve(G_lu, -e)
            
            # print('phi difference')
            # diff = self.g_cons.Φ - self.Φ
            # print(diff)
            
            
            # for j, body in enumerate(self.bodies):
            #     body.r += δ[3*j:3*(j+1)]
            #     body.ω += δ[3*(self.nb + j):3*(self.nb + j+1)]/self.h

            self.r += δ[0:3*self.nb]
            self.theta_bar += δ[3*self.nb:6*self.nb]                
            self.λ_hat += δ[6*self.nb: len(δ)]
            # print("λ_hat, {} {}".format(self.λ_hat[3]/self.h**2, self.λ_hat[4]/self.h**2))

            # logging.debug('t: {:.3f}, k: {:>2d}, norm: {:6.6e}'.format(
            #     t, self.k, np.linalg.norm(δ)))

            if np.linalg.norm(δ) < self.tol:            
                print("rA-half, itr %d, correction: δr %E, %E, δω %E, λ %.13E" % (self.k+1, δ[1], δ[2], δ[5]/self.h, δ[-1]/(self.h**2)))
                break
            
            # todo: populate self.r and self.theta_bar into each body to 
            # evaluate Phi at new iteration
            # need to be very careful with the update of A, because it is updated iteratively
            # from previous one.
            # figure out when A_prev is updated, make sure it is updated every time step
            for j, body in enumerate(self.bodies):
                body.r = self.r[3*j:3*(j+1), :]
                body.ω = self.theta_bar[3*j: 3*(j+1), :]/self.h
                body.A = body.A_prev @ exp(self.h * skew(body.ω))


            self.k += 1

            print("rA-half, itr %d, correction: δr %E, %E, δω %E, λ %.13E" % (self.k, δ[1], δ[2], δ[5]/self.h, δ[-1]/(self.h**2)))
            
            if self.k >= self.max_iters:
            # temporarily disable this for the purpose of debugging
                raise RuntimeError(
                    'Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(t, self.max_iters))
            
        # logging.debug('t: {:.3f}, iterations: {:>2d}'.format(t, self.k))
                    # logging position
        # print("tol %E, itr %d, residual e0 %E, e1 %E e2 %E" % (self.tol, self.k, np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2)))

        self.update_ddr(Phi_r_old)
        self.update_dω(Pi_old)
        
    def update_ddr(self, Phi_r_old):
        self.λ = self.λ_hat / (self.h * self.h)
        self.ddr = self.M_inv @ (self.F_ext - Phi_r_old.T @ self.λ)
        for j, body in enumerate(self.bodies):
            body.ddr = self.ddr[3*j:3*(j+1), :]
            body.dr = body.dr + self.h*body.ddr
            body.r = self.r[3*j:3*(j+1), :]

    def update_dω(self, Pi_old):
        sum_torque = np.zeros((3*self.nb, 1))
        for j, body in enumerate(self.bodies):
            body.ω = self.theta_bar[3*j: 3*(j+1), :]/self.h
            body.A = body.A_prev @ exp(self.h * skew(body.ω))

            J_term = skew(body.ω) @ body.J @ body.ω
            sum_torque[3*j: 3*(j+1), :] = J_term

        self.dω = self.J_inv @ ( -Pi_old.T @ self.λ + sum_torque)
        for j, body in enumerate(self.bodies):
            body.dω = self.dω[3*j:3*(j+1), :]
            print("r = {} {}, omic = {}".format(body.r[1], body.r[2], body.ω[2]))



    def do_kinematics_step(self, t):

        self.g_cons.maybe_swap_gcons(t)

        # Refresh the inverse matrix with our new positions
        self.Φq = self.g_cons.get_phi_q(t)
        Φq_lu = lu_factor(self.Φq)

        self.k = 0
        while True:
            self.Φ = self.g_cons.get_phi(t)

            Δq = lu_solve(Φq_lu, -self.Φ)

            for j, body in enumerate(self.bodies):
                Δr = Δq[3*j:3*(j+1)]
                Δθ = Δq[3*(self.nb + j):3*(self.nb + j+1)]
                Δθ_mag = np.linalg.norm(Δθ)

                body.r = body.r + Δr
                if Δθ_mag != 0:
                    body.A = body.A @ R(Δθ/Δθ_mag, Δθ_mag)

            self.k += 1

            # logging.debug('t: {:.3f}, k: {:>2d}, norm: {:6.6e}'.format(t, self.k, np.linalg.norm(Δq)))

            if np.linalg.norm(Δq) < self.tol:
                break

            if self.k >= self.max_iters:
                raise RuntimeError(
                    'Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(t, self.max_iters))

        self.Φq = self.g_cons.get_phi_q(t)
        Φq_lu = lu_factor(self.Φq)

        dq = lu_solve(Φq_lu, self.g_cons.get_nu(t))
        for j, body in enumerate(self.bodies):
            body.dr = dq[3*j:3*(j+1), :]
            body.ω = dq[3*(self.nb + j):3*(self.nb + j+1), :]

        ddq = lu_solve(Φq_lu, self.g_cons.get_gamma(t))
        for j, body in enumerate(self.bodies):
            body.ddr = ddq[3*j:3*(j+1), :]
            body.dω = ddq[3*(self.nb + j):3*(self.nb + j+1), :]
    


def create_constraint_from_bodies(json_con, all_bodies):
    """
    Reads from all_bodies to call create_constraint with appropriate args
    """

    body_i = all_bodies[json_con["body_i"]]
    body_j = all_bodies[json_con["body_j"]]

    return create_constraint(json_con, body_i, body_j)


def create_constraint(json_con, body_i, body_j):
    """
    Branches on a json constraints type to call the appropriate constraint constructor
    """

    con_type = Constraints[json_con["type"]]
    if con_type == Constraints.DP1:
        con = DP1.init_from_dict(json_con, body_i, body_j)
    elif con_type == Constraints.CD:
        con = CD.init_from_dict(json_con, body_i, body_j)
    elif con_type == Constraints.DP2:
        con = DP2.init_from_dict(json_con, body_i, body_j)
    elif con_type == Constraints.D:
        con = D.init_from_dict(json_con, body_i, body_j)
    else:
        raise ValueError('Unmapped enum value')

    return con


def process_system(file_bodies, file_constraints):
    """
    Takes the output of read_model_file and returns a list of constraint and body objects
    """

    all_bodies = {}
    bodies = []
    j = 0

    # Keys user uses for bodies must correspond to constraints
    for file_body in file_bodies:
        body = Body.init_from_dict(file_body)
        all_bodies[file_body['id']] = body

        # Give non-ground bodies an ID and save them separately
        if not body.is_ground:
            body.id = j
            bodies.append(body)
            j += 1

    cons = [create_constraint_from_bodies(f_con, all_bodies)
            for f_con in file_constraints]

    con_group = ConGroup(cons, len(bodies))

    return (bodies, con_group)
