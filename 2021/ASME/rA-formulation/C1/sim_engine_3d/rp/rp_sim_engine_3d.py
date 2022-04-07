#!/usr/bin/env python3

import json as js
import numpy as np
import time
import logging
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as Rot

import rp.rp_gcons as gcons


class rpSimEngine3D:
    def __init__(self, filename):
        self.bodies_list = []
        self.bodies_full = []
        self.nb = 0  # number of bodies that don't include the ground
        self.constraint_list = []
        self.nc = 0

        # solver parameters
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

        self.init_system(filename)
        self.dynamics_initialized = False
        self.plotting_initialized = False

        self.alternative_driver = None

        self.r_sol = None
        self.r_dot_sol = None
        self.r_ddot_sol = None

        self.g = -9.81
        self.lam = 0
        self.lambda_p = 0

    def init_system(self, filename):
        # setup initial system based on model parameters
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
                    # logging.info("body_i found")
                if body.body_id == con['body_j']:
                    body_j = body
                    # logging.info("body_j found")
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
            logging.info("Switching to alternative constraint. Time = {}".format(t))
            if self.alternative_driver is None:
                logging.warning("Alternative driving constraint not defined.")
                return
            self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

        Phi_q = self.get_phi_q()
        Phi_q_lu = lu_factor(Phi_q)
        iterations = 0
        while True:
            Phi_K = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
            Phi_euler = np.concatenate([0.5 * body.p.T @ body.p - 0.5 for body in self.bodies_list], axis=0)
            Phi = np.concatenate((Phi_K, Phi_euler), axis=0)
            delta_q = lu_solve(Phi_q_lu, -Phi)

            for idx, body in enumerate(self.bodies_list):
                body.r = body.r + delta_q[idx * 3:(idx * 3) + 3, :]
                body.p = body.p + delta_q[3 * self.nb + idx * 4:3 * self.nb + idx * 4 + 4, :]

            iterations += 1
            if iterations >= self.max_iters:
                logging.warning("Newton-Raphson self.has not converged after {} iterations. "
                                "Stopping at time {}".format(self.max_iters, t))
                break
            
            if np.linalg.norm(delta_q) < self.tol:
                break

        Phi_q = self.get_phi_q()
        Phi_q_lu = lu_factor(Phi_q)
        
        # calculate velocity
        nu_G = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
        nu_euler = np.zeros((self.nb, 1))
        nu = np.concatenate((nu_G, nu_euler), axis=0)
        q_dot = lu_solve(Phi_q_lu, nu)
        for idx, body in enumerate(self.bodies_list):
            body.r_dot = q_dot[idx * 3:idx * 3 + 3, :]
            body.p_dot = q_dot[3 * self.nb + idx * 4:3 * self.nb + idx * 4 + 4, :]

        # calculate acceleration
        gamma_G = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
        gamma_euler = np.concatenate([-body.p_dot.T @ body.p_dot for body in self.bodies_list], axis=0)
        gamma = np.concatenate((gamma_G, gamma_euler), axis=0)
        q_ddot = lu_solve(Phi_q_lu, gamma)
        for idx, body in enumerate(self.bodies_list):
            body.r_ddot = q_ddot[idx * 3:idx * 3 + 3, :]
            body.p_ddot = q_ddot[3 * self.nb + idx * 4:3 * self.nb + idx * 4 + 4, :]

        # store solution for plotting
        for body in self.bodies_list:
            self.store_data(i, body)

        return iterations

    def dynamics_solver(self, order=1):
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
        gamma_euler = np.concatenate([-body.p_dot.T @ body.p_dot for body in self.bodies_list], axis=0)
        gamma = np.concatenate([con.gamma(self.t_start) for con in self.constraint_list], axis=0)
        eom_rhs = np.block([[F],
                            [tau],
                            [gamma_euler],
                            [gamma]])

        # solve to find initial accelerations and lagrange multipliers
        z = np.linalg.solve(self.get_psi(), eom_rhs)

        # set the initial conditions
        self.lam = z[8 * self.nb:]
        self.lambda_p = z[7 * self.nb:8 * self.nb]
        for idx, body in enumerate(self.bodies_list):
            body.r_ddot = z[idx * 3:idx * 3 + 3, :]
            body.p_ddot = z[3 * self.nb + idx * 4:3 * self.nb + idx * 4 + 4, :]

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

        if i == 1 or order == 1:
            beta = 1
            alphas = np.array([1, 0])
        else:
            beta = 2 / 3
            alphas = np.array([4 / 3, -1 / 3])

        for body in self.bodies_list:
            body.c_r_dot = alphas[0] * body.r_dot + alphas[1] * body.r_dot_prev
            body.c_p_dot = alphas[0] * body.p_dot + alphas[1] * body.p_dot_prev

            body.c_r = alphas[0] * body.r + alphas[1] * body.r_prev + beta * self.h * body.c_r_dot
            body.c_p = alphas[0] * body.p + alphas[1] * body.p_prev + beta * self.h * body.c_p_dot

            # cache previous data for integration
            self.cache_prev(body)

        psi_lu = lu_factor(self.get_psi())
        
        # Begin Newton Iteration
        iterations = 0
        delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
        while delta_norm > self.tol:
            
            r_ddot_all = np.zeros((3*self.nb, 1))
            p_ddot_all = np.zeros((4*self.nb, 1))
            
            for idx, body in enumerate(self.bodies_list):
                body.r = body.c_r + beta ** 2 * self.h ** 2 * body.r_ddot
                body.r_dot = body.c_r_dot + beta * self.h * body.r_ddot
                body.p = body.c_p + beta ** 2 * self.h ** 2 * body.p_ddot
                body.p_dot = body.c_p_dot + beta * self.h * body.p_ddot

                r_ddot_all[3*idx:3*(idx+1)] = body.r_ddot
                p_ddot_all[4*idx:4*(idx+1)] = body.p_ddot

            g = self.get_residual(t, beta, r_ddot_all, p_ddot_all)
            delta = lu_solve(psi_lu, -g)

            for idx, body in enumerate(self.bodies_list):
                body.r_ddot = body.r_ddot + delta[idx * 3:(idx * 3) + 3, :]
                body.p_ddot = body.p_ddot + delta[3 * self.nb + idx * 4:3 * self.nb + idx * 4 + 4, :]

            self.lam += delta[8 * self.nb:]
            self.lambda_p += delta[7 * self.nb:8 * self.nb]

            delta_norm = np.linalg.norm(delta)
            iterations += 1
            if iterations >= self.max_iters:
                logging.info("Solution has not converged after {} iterations. "
                                "Stopping. Time = {}".format(self.max_iters, t))
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
        body.p_prev = body.p
        body.r_dot_prev = body.r_dot
        body.p_dot_prev = body.p_dot    

    def get_psi(self):
        # build Psi, our quasi-newton iteration matrix
        P = self.get_P()
        Phi_q = self.get_phi_q()
        Phi_r = Phi_q[0:self.nc, 0:3 * self.nb]
        Phi_p = Phi_q[0:self.nc, 3 * self.nb:]
        return np.block([[self.get_M(), np.zeros((3 * self.nb, 4 * self.nb)), np.zeros((3 * self.nb, self.nb)), Phi_r.T],
                        [np.zeros((4 * self.nb, 3 * self.nb)), self.get_J_P(), P.T, Phi_p.T],
                        [np.zeros((self.nb, 3 * self.nb)), P, np.zeros((self.nb, self.nb)), np.zeros((self.nb, self.nc))],
                        [Phi_r, Phi_p, np.zeros((self.nc, self.nb)), np.zeros((self.nc, self.nc))]])

    def get_phi_q(self):
        jacobian = np.zeros((self.nc + self.nb, 7 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            idi = con.body_i.body_id - 1
            idj = con.body_j.body_id - 1
            if con.body_i.is_ground:
                # fill row of jacobian with only body j
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()
                jacobian[row, offset + 4 * idj:offset + 4 * idj + 4] = con.partial_p()
            elif con.body_j.is_ground:
                # fill row of jacobian with only body i
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()
                jacobian[row, offset + 4 * idi:offset + 4 * idi + 4] = con.partial_p()
            else:
                # fill row of jacobian with both body i and body j
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()[0]
                jacobian[row, offset + 4 * idi:offset + 4 * idi + 4] = con.partial_p()[0]
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()[1]
                jacobian[row, offset + 4 * idj:offset + 4 * idj + 4] = con.partial_p()[1]

        # Euler parameter rows for each body
        row_euler = self.nc
        for idx, body in enumerate(self.bodies_list):
            jacobian[row_euler, offset + 4 * idx:offset + 4 * idx + 4] = body.p.T
            row_euler += 1
        return jacobian

    def get_residual(self, t, beta, r_ddot, p_ddot):
        Phi = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
        Phi_euler = np.concatenate([0.5 * body.p.T @ body.p - 0.5 for body in self.bodies_list], axis=0)
        Phi_q = self.get_phi_q()
        Phi_r = Phi_q[0:self.nc, 0:3 * self.nb]
        Phi_p = Phi_q[0:self.nc, 3 * self.nb:]
        
        g_row1 = self.get_M() @ r_ddot + Phi_r.T @ self.lam - self.get_F_g()
        g_row2 = self.get_J_P() @ p_ddot + Phi_p.T @ self.lam \
                    + self.get_P().T @ self.lambda_p - self.get_tau()
        g_row3 = 1 / (beta ** 2 *self.h** 2) * Phi_euler
        g_row4 = 1 / (beta ** 2 *self.h** 2) * Phi
        
        return np.block([[g_row1],
                         [g_row2],
                         [g_row3],
                         [g_row4]])

    def get_M(self):
        m_mat = np.zeros((3 * self.nb, 3 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            m_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.m * np.eye(3)
        return m_mat

    def get_J_P(self):
        j_p_mat = np.zeros((4 * self.nb, 4 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            G = np.concatenate((-body.p[1:], -gcons.skew(body.p[1:]) + body.p[0] * np.eye(3)), axis=1)
            j_p_mat[idx * 4:idx * 4 + 4, idx * 4:idx * 4 + 4] = 4 * G.T @ body.J @ G
        return j_p_mat

    def get_P(self):
        p_mat = np.zeros((self.nb, 4 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            p_mat[idx, 4 * idx:4 * idx + 4] = body.p.T
        return p_mat

    def get_F_g(self):
        # return F when gravity is the only force
        f_g_mat = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
        return f_g_mat

    def get_tau(self):
        tau = np.zeros((4 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            G_dot = np.concatenate((-body.p_dot[1:], -gcons.skew(body.p_dot[1:]) + body.p_dot[0] * np.eye(3)), axis=1)
            tau[idx * 4:idx * 4 + 4] = 8 * G_dot.T @ body.J @ G_dot @ body.p
        return tau

    def get_reaction_torque(self):
        Phi_p = self.get_phi_q()[0:self.nc, 3:]
        for body in self.bodies_list:
            e_mat = np.concatenate((-body.p[1:], gcons.skew(body.p[1:]) + body.p[0] * np.eye(3)), axis=1)
            pi = 1 / 2 * Phi_p @ e_mat.T
            torque = -pi.T @ self.lam
        return torque


class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']
            self.r = np.zeros((3, 1))
            self.r_dot = np.zeros((3, 1))
            self.r_ddot = np.zeros((3, 1))
            self.p = np.array([[1],
                               [0],
                               [0],
                               [0]])
            self.p_dot = np.zeros((4, 1))
            self.p_ddot = np.zeros((4, 1))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.p_prev = self.p
            self.p_dot_prev = self.p_dot
            self.c_r = None
            self.c_p = None
            self.c_r_dot = None
            self.c_p_dot = None
            self.m = None
            self.J = None
        else:
            self.is_ground = False
            self.body_id = body_dict['id']
            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.r_ddot = np.zeros((3, 1))

            quat = np.array(
                [Rot.from_matrix(np.array(body_dict['A'])).as_quat()]).T
            self.p = np.array([[quat[3, 0], quat[0, 0], quat[1, 0], quat[2, 0]]]).T

            omega = np.array([body_dict['omega']]).T
            e_mat = np.concatenate((-self.p[1:], gcons.skew(self.p[1:]) + self.p[0] * np.eye(3)), axis=1)
            self.p_dot = 1 / 2 * e_mat.T @ omega
            self.p_ddot = np.zeros((4, 1))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.p_prev = self.p
            self.p_dot_prev = self.p_dot
            self.c_r = None
            self.c_p = None
            self.c_r_dot = None
            self.c_p_dot = None
            self.m = 0
            self.J = np.zeros((3, 3))
