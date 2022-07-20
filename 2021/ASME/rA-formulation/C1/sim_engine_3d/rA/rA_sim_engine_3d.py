#!/usr/bin/env python3

import rA.rA_gcons as gcons

import json as js
import numpy as np
import time
import logging
from scipy.linalg import lu_factor, lu_solve

I3 = np.eye(3)


# ================================== System utility functions =====================================
def R(u_bar, chi):
    """Return the value of the rotation matrix calculated based on Rodrigues's formula."""
    return np.cos(chi) * I3 + (1 - np.cos(chi)) * (u_bar @ u_bar.T) + np.sin(chi) * gcons.skew(u_bar)


# ==================================== 3D Simulation Engine =======================================
class rASimEngine3D:
    def __init__(self, filename):
        self.bodies_list = []
        self.bodies_full = []
        self.nb = 0  # number of bodies that don't include the ground!
        self.constraint_list = []
        self.nc = 0

        # simulation parameters
        self.h = 0.001
        self.t_start = 0
        self.t_end = 3
        self.tspan = None
        self.N = None
        self.t_grid = None
        self.tol = None
        self.max_iters = 20
        self.duration = 0
        self.avg_iterations = 0
        self.full_jacobian = False

        self.init_system(filename)
        self.dynamics_initialized = False
        self.plotting_initialized = False

        self.alternative_driver = None

        self.r_sol = None
        self.r_dot_sol = None
        self.r_ddot_sol = None

        # values needed for dynamics analysis
        self.lam = 0
        self.g = -9.81

    def init_system(self, filename):
        """Setup initial system based on model parameters in .mdl file."""
        with open(filename) as f:
            model = js.load(f)
            bodies = model['bodies']
            constraints = model['constraints']

        for body in bodies:
            self.bodies_full.append(RigidBody(body))
        # list of non-ground bodies
        self.bodies_list = [body for body in self.bodies_full if not body.is_ground]
        self.nb = len(self.bodies_list)

        for con in constraints:
            for body in self.bodies_full:
                if body.body_id == con['body_i']:
                    body_i = body
                    #logging.info("body_i found")
                if body.body_id == con['body_j']:
                    body_j = body
                    #logging.info("body_j found")
            if con['type'] == 'DP1':
                self.constraint_list.append(gcons.GConDP1(con, body_i, body_j))
            elif con['type'] == 'DP2':
                self.constraint_list.append(gcons.GConDP2(con, body_i, body_j))
            elif con['type'] == 'D':
                self.constraint_list.append(gcons.GConD(con, body_i, body_j))
            elif con['type'] == 'CD':
                self.constraint_list.append(gcons.GConCD(con, body_i, body_j))
            else:
                logging.warning("Incorrect geometric constraint type given.")
        self.nc = len(self.constraint_list)
        return

    def kinematics_solver(self):
        self.initialize_plotting()
        
        iterations = np.zeros((self.N, 1))

        start = time.process_time()
        for i, t in enumerate(self.t_grid):
            iterations[i] = self.kinematics_step(i, t)
        self.duration = time.process_time() - start
        self.avg_iterations = np.mean(iterations)

        logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        logging.info('Simulation time: {}'.format(self.duration))
    
    def kinematics_step(self, i, t):
        if not self.plotting_initialized:
            self.initialize_plotting()
        # check for driving constraint singularity
        if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
            logging.info("Switching to alternative constraint. Time = ", t)
            if self.alternative_driver is None:
                logging.warning("Alternative driving constraint not defined.")
                return
            self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

        Phi_q = self.get_sensitivities()
        Phi_q_lu = lu_factor(Phi_q)
        iterations = 0
        while True:
            Phi = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
            delta_q = lu_solve(Phi_q_lu, -Phi)

            for idx, body in enumerate(self.bodies_list):
                body.r = body.r + delta_q[idx * 3:(idx * 3) + 3, :]
                theta = delta_q[3 * self.nb + idx * self.nb:3 * self.nb + idx * self.nb + 3, :]
                theta_norm = np.linalg.norm(theta)
                if theta_norm != 0:
                    body.A = body.A @ R(theta / theta_norm, theta_norm)

            iterations += 1
            if iterations >= self.max_iters:
                logging.warning("Newton-Raphson has not converged after", str(self.max_iters),
                                "iterations. Stopping.")
                break
            if np.linalg.norm(delta_q) < self.tol:
                break

        Phi_q = self.get_sensitivities()
        Phi_q_lu = lu_factor(Phi_q)
        
        # calculate velocity
        nu = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
        q_dot = lu_solve(Phi_q_lu, nu)
        for idx, body in enumerate(self.bodies_list):
            body.r_dot = q_dot[idx * 3:idx * 3 + 3, :]
            body.omega = q_dot[3 * self.nb + idx * self.nb:3 * self.nb + idx * self.nb + 3, :]

        # calculate acceleration
        gamma = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
        q_ddot = lu_solve(Phi_q_lu, gamma)
        for idx, body in enumerate(self.bodies_list):
            body.r_ddot = q_ddot[idx * 3:idx * 3 + 3, :]
            body.omega_dot = q_ddot[3 * self.nb + idx * self.nb:3 * self.nb + idx * self.nb + 3, :]
        
        # store solution for plotting
        for body in self.bodies_list:
            self.store_data(i, body)

        return iterations

    def dynamics_solver(self):
        self.initialize_dynamics()
        
        iterations = np.zeros((self.N, 1))
        
        start = time.process_time()
        for i, t in enumerate(self.t_grid):
            if i == 0:
                continue
            iterations[i] = self.dynamics_step(i, t)
        self.duration = time.process_time() - start
        self.avg_iterations = np.mean(iterations)
        
        logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        logging.info('Simulation time: {}'.format(self.duration))

    def initialize_dynamics(self):
        self.initialize_plotting()

        # build right hand side of equations of motion
        F = self.get_F_g()
        tau = self.get_tau()
        gamma = np.concatenate([con.gamma(self.t_start) for con in self.constraint_list], axis=0)
        eom_rhs = np.block([[F],
                            [tau],
                            [gamma]])
            
        # solve to find initial accelerations and lagrange multipliers
        z = np.linalg.solve(self.get_psi(), eom_rhs)

        # set the initial conditions
        self.lam = z[6 * self.nb:]
        for idx, body in enumerate(self.bodies_list):
            body.r_ddot = z[idx * 3:idx * 3 + 3]
            body.omega_dot = z[3 * self.nb + idx * 3:3 * self.nb + idx * 3 + 3]
            
            # store solution in array for plotting
            self.store_data(0, body)
            
            # cache previous data for integration
            self.cache_prev(body)  

        self.dynamics_initialized = True
            
    def dynamics_step(self, i, t, order=1):
        if not self.dynamics_initialized:
            self.initialize_dynamics()

        # check for driving constraint singularity
        if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
            logging.info("Switching to alternative constraint. Time = {}".format(t))
            if self.alternative_driver is None:
                logging.warning("Alternative driving constraint not defined.")
                return
            self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

        for body in self.bodies_list:
            self.cache_prev(body)

        psi_lu = lu_factor(self.get_psi())

        # Begin Newton Iteration
        iterations = 0
        delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
        while delta_norm > self.tol:
            
            r_ddot_all = np.zeros((3 * self.nb, 1))
            
            for idx, body in enumerate(self.bodies_list):
                body.r_dot = body.r_dot_prev + self.h * body.r_ddot
                body.omega = body.omega_prev + self.h * body.omega_dot
                body.r = body.r_prev + self.h * body.r_dot
                theta_mat = self.h * gcons.skew(body.omega)
                theta = np.array([[theta_mat[2,1]], [theta_mat[0,2]], [theta_mat[1,0]]])
                theta_norm = np.linalg.norm(theta)
                if theta_norm != 0:
                    body.A = body.A_prev @ R(theta / theta_norm, theta_norm)

                r_ddot_all[idx * 3:idx * 3 + 3] = body.r_ddot

            g = self.get_residual(t, r_ddot_all)
            delta = lu_solve(psi_lu, -g)

            for idx, body in enumerate(self.bodies_list):
                body.r_ddot = body.r_ddot + delta[idx * 3:idx * 3 + 3]
                body.omega_dot = body.omega_dot + delta[3 * self.nb + idx * 3:3 * self.nb + idx * 3 + 3]

            self.lam += delta[6 * self.nb:]

            delta_norm = np.linalg.norm(delta)
            iterations += 1
            if iterations >= self.max_iters:
                logging.warning("Solution has not converged after", str(self.max_iters), "iterations. Stopping.")
                break

        for body in self.bodies_list:
            # store solution in array for plotting
            self.store_data(i, body)

        return iterations

    def initialize_plotting(self):
        self.tspan = self.t_end - self.t_start
        self.N = int(self.tspan / self.h)
        self.t_grid = np.linspace(self.t_start, self.t_end, self.N, endpoint=True)
        self.r_sol = np.zeros((self.N, 3 * self.nb))
        self.r_dot_sol = np.zeros((self.N, 3 * self.nb))
        self.r_ddot_sol = np.zeros((self.N, 3 * self.nb))
        self.plotting_initialized = True

    def store_data(self, time, body):
        self.r_sol[time, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
        self.r_dot_sol[time, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
        self.r_ddot_sol[time, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

    def cache_prev(self, body):
        body.r_prev = body.r
        body.A_prev = body.A
        body.r_dot_prev = body.r_dot
        body.omega_prev = body.omega

    def get_psi(self):
        M = self.get_M()
        J = self.get_J()
        Phi_q = self.get_sensitivities()
        Phi_r = Phi_q[:, :3 * self.nb]
        Pi = Phi_q[:, 3 * self.nb:]

        G_rr = M 
        G_rw = np.zeros((3 * self.nb, 3 * self.nb))
        G_rlam = Phi_r.T
        G_wr = np.zeros((3 * self.nb, 3 * self.nb))
        G_ww = J
        G_wlam = Pi.T
        G_lamr = Phi_r
        G_lamw = Pi
        zero_block_33 = np.zeros((self.nc, self.nc))


        return np.block([[G_rr, G_rw, G_rlam],
                         [G_wr, G_ww, G_wlam],
                         [G_lamr, G_lamw, zero_block_33]])

    def get_sensitivities(self):
        jacobian = np.zeros((self.nc, 6 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            idi = con.body_i.body_id - 1
            idj = con.body_j.body_id - 1
            if con.body_i.is_ground:
                # fill row of jacobian with only body j
                jacobian[row, 3 * idj:3 * idj + 3] = con.r_sensitivity()
                jacobian[row, offset + 3 * idj:offset + 3 * idj + 3] = con.theta_sensitivity()
            elif con.body_j.is_ground:
                # fill row of jacobian with only body i
                jacobian[row, 3 * idi:3 * idi + 3] = con.r_sensitivity()
                jacobian[row, offset + 3 * idi:offset + 3 * idi + 3] = con.theta_sensitivity()
            else:
                # fill row of jacobian with both body i and body j
                jacobian[row, 3 * idi:3 * idi + 3] = con.r_sensitivity()[0]
                jacobian[row, offset + 3 * idi:offset + 3 * idi + 3] = con.theta_sensitivity()[0]
                jacobian[row, 3 * idj:3 * idj + 3] = con.r_sensitivity()[1]
                jacobian[row, offset + 3 * idj:offset + 3 * idj + 3] = con.theta_sensitivity()[1]

        return jacobian

    def get_residual(self, t, r_ddot_all):
        lam = self.lam
        Phi = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
        Phi_q = self.get_sensitivities()
        Phi_r = Phi_q[:, :3 * self.nb]
        Pi = Phi_q[:, 3 * self.nb:]
        
        g_row1 = self.get_M() @ r_ddot_all + Phi_r.T @ lam - self.get_F_g()
        g_row2 = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            g_row2[idx * 3:idx * 3 + 3] = body.J @ body.omega_dot + (
                    gcons.skew(body.omega) @ body.J @ body.omega)
        g_row2 += Pi.T @ lam
        g_row3 = 1 / self.h ** 2 * Phi
        
        return np.block([[g_row1],
                         [g_row2],
                         [g_row3]])

    def get_F_g(self):
        # returns F when gravity is the only force
        f_g_mat = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
        return f_g_mat

    def get_M(self):
        m_mat = np.zeros((3 * self.nb, 3 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            m_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.m * I3
        return m_mat

    def get_J(self):
        j_mat = np.zeros((3 * self.nb, 3 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            j_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.J - self.h*(gcons.skew(body.J @ body.omega) -
                                                                               gcons.skew(body.omega) @ body.J) #+ body.n_omega
        return j_mat

    def get_tau(self):
        tau = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            n_bar = np.zeros((3, 1))
            omega_bar = body.omega
            tau[idx * 3:idx * 3 + 3] = n_bar - gcons.skew(omega_bar) @ body.J @ omega_bar
        return tau


class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']
            self.A = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
            self.omega = np.array([[0],
                                   [0],
                                   [0]])
            self.omega_dot = np.array([[0],
                                       [0],
                                       [0]])
            self.r = np.array([[0],
                               [0],
                               [0]])
            self.r_dot = np.array([[0],
                                   [0],
                                   [0]])
            self.r_ddot = np.array([[0],
                                    [0],
                                    [0]])

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.A_prev = self.A
            self.omega_prev = self.omega
            self.m = None
            self.J = None

        else:
            self.is_ground = False
            self.body_id = body_dict['id']
            self.A = np.array(body_dict['A'])
            self.omega = np.array([body_dict['omega']]).T
            self.omega_dot = np.array([[0],
                                       [0],
                                       [0]])
            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.r_ddot = np.array([[0],
                                    [0],
                                    [0]])

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.A_prev = self.A
            self.omega_prev = self.omega
            self.m = 0
            self.J = np.zeros((3, 3))
