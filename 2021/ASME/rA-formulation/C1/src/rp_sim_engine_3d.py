#!/usr/bin/env python3

import json as js
import numpy as np
import time
import logging
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as Rot

import rp_gcons as gcons


class rpSimEngine3D:
    def __init__(self, filename):
        self.bodies_list = []
        self.nb = 0  # number of bodies that don't include the ground!
        self.constraint_list = []
        self.nc = 0

        self.h = 0.001
        self.t_start = 0
        self.t_end = 1
        self.tspan = None
        self.N = None
        self.t_grid = None
        self.tol = None
        self.max_iters = 20

        self.init_system(filename)

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
            self.bodies_list.append(RigidBody(body))

        for con in constraints:
            for body in self.bodies_list:
                if body.body_id == con['body_i']:
                    body_i = body
                    logging.info("body_i found")
                if body.body_id == con['body_j']:
                    body_j = body
                    logging.info("body_j found")
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

    def initialize_plotting(self):
        self.tspan = self.t_end - self.t_start
        self.N = int(self.tspan / self.h)
        self.t_grid = np.linspace(self.t_start, self.t_end, self.N, endpoint=True)
        self.r_sol = np.zeros((self.N, 3 * self.nb))
        self.r_dot_sol = np.zeros((self.N, 3 * self.nb))
        self.r_ddot_sol = np.zeros((self.N, 3 * self.nb))
        return

    def kinematics_solver(self):
        logging.info("Number of bodies counted:", self.nb)
        self.initialize_plotting()
        iterations = np.zeros((self.N, 1))

        start = time.perf_counter()
        for i, t in enumerate(self.t_grid):
            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                logging.info("Switching to alternative constraint. Time = ", t)
                if self.alternative_driver is None:
                    logging.warning("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

            Phi_q = self.get_phi_q()
            Phi_q_lu = lu_factor(Phi_q)

            iteration = 0
            while True:

                Phi = self.get_phi(t)

                delta_q = lu_solve(Phi_q_lu, -Phi)

                for body in self.bodies_list:
                    if body.is_ground:
                        pass
                    else:
                        body.r = body.r + delta_q[(body.body_id - 1) * 3:((body.body_id - 1) * 3) + 3, :]
                        body.p = body.p + delta_q[3 * self.nb + (body.body_id - 1) * 4:3 * self.nb + (
                                body.body_id - 1) * 4 + 4, :]

                iteration += 1
                if iteration >= self.max_iters:
                    logging.warning("Newton-Raphson self.has not converged after", str(self.max_iters), "iterations. Stopping at time ", str(t))
                    break

                if np.linalg.norm(delta_q) < self.tol:
                    break

            #logging.info("Newton-Raphson took", str(iteration), "iterations to converge.")
            iterations[i] = iteration

            Phi_q = self.get_phi_q()
            Phi_q_lu = lu_factor(Phi_q)
            # calculate velocity
            q_dot = lu_solve(Phi_q_lu, self.get_nu(t))
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_dot = q_dot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                    body.p_dot = q_dot[3 * self.nb + (body.body_id - 1) * 4:3 * self.nb + (
                                body.body_id - 1) * 4 + 4, :]

            # calculate acceleration
            q_ddot = lu_solve(Phi_q_lu, self.get_gamma(t))
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_ddot = q_ddot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                    body.p_ddot = q_ddot[3 * self.nb + (body.body_id - 1) * 4:3 * self.nb + (
                                body.body_id - 1) * 4 + 4, :]

                    # store solution in array for plotting
                    self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                    self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                    self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

        duration = time.perf_counter() - start
        print('Avg. iterations: {}'.format(np.mean(iterations)))
        print('Simulation time: {}'.format(duration))

    def dynamics_solver(self, order=1):
        logging.info("Number of bodies counted:", self.nb)
        self.initialize_plotting()

        # build full RHS matrix
        RHS = np.zeros((self.nc + 8 * self.nb, 1))
        RHS[0:3 * self.nb] = self.get_F_g()
        RHS[3 * self.nb:7 * self.nb] = self.get_tau()
        RHS[7 * self.nb:8 * self.nb] = self.get_gamma(self.t_start)[self.nc:, :]
        RHS[8 * self.nb:] = self.get_gamma(self.t_start)[0:self.nc, :]

        # solve to find initial accelerations and lagrange multipliers, vector z
        z = np.linalg.solve(self.psi(), RHS)
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                body.r_ddot = z[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                body.p_ddot = z[3 * self.nb + (body.body_id - 1) * 4:3 * self.nb + (
                        body.body_id - 1) * 4 + 4, :]

                # store solution in array for plotting
                self.r_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                self.r_dot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                self.r_ddot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

                body.r_prev = body.r
                body.p_prev = body.p
                body.r_dot_prev = body.r_dot
                body.p_dot_prev = body.p_dot

        self.lam += z[8 * self.nb:]
        self.lambda_p += z[7 * self.nb:8 * self.nb]

        iterations = np.zeros((self.N, 1))
        start = time.perf_counter()
        for i, t in enumerate(self.t_grid):
            if i == 0:
                continue

            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                logging.info("Switching to alternative constraint. Time = ", t)
                if self.alternative_driver is None:
                    logging.warning("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

            if i == 1 or order == 1:
                beta = 1
                alphas = np.array([1, 0])
            else:
                beta = 2/3
                alphas = np.array([4/3, -1/3])

            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.c_r_dot = alphas[0]*body.r_dot + alphas[1]*body.r_dot_prev
                    body.c_p_dot = alphas[0]*body.p_dot + alphas[1]*body.p_dot_prev

                    body.c_r = alphas[0]*body.r + alphas[1]*body.r_prev + beta*self.h*body.c_r_dot
                    body.c_p = alphas[0] * body.p + alphas[1] * body.p_prev + beta * self.h * body.c_p_dot

            psi = self.psi()
            psi_lu = lu_factor(psi)
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_prev = body.r
                    body.p_prev = body.p
                    body.r_dot_prev = body.r_dot
                    body.p_dot_prev = body.p_dot

            # Begin Newton Iteration
            iteration = 0
            delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
            while delta_norm > self.tol:
                for body in self.bodies_list:
                    if body.is_ground:
                        pass
                    else:
                        body.r = body.c_r + beta**2 * self.h**2 * body.r_ddot
                        body.r_dot = body.c_r_dot + beta * self.h *body.r_ddot
                        body.p = body.c_p + beta ** 2 * self.h ** 2 * body.p_ddot
                        body.p_dot = body.c_p_dot + beta * self.h * body.p_ddot

                if order == 2 and i == 1:
                    g = self.residual(1, t)
                else:
                    g = self.residual(order, t)

                delta = lu_solve(psi_lu, -g)

                for body in self.bodies_list:
                    if body.is_ground:
                        pass
                    else:
                        body.r_ddot = body.r_ddot + delta[(body.body_id - 1) * 3:((body.body_id - 1) * 3) + 3, :]
                        body.p_ddot = body.p_ddot + delta[3 * self.nb + (body.body_id - 1) * 4:3 * self.nb + (
                                body.body_id - 1) * 4 + 4, :]

                self.lam += delta[8 * self.nb:]
                self.lambda_p += delta[7 * self.nb:8 * self.nb]

                delta_norm = np.linalg.norm(delta)
                iteration += 1
                if iteration >= self.max_iters:
                    logging.warning("Solution self.has not converged after", str(self.max_iters), "iterations. Stopping.")
                    break

            #logging.info("Iteration: ", iteration)
            iterations[i] = iteration

            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    # store solution in array for plotting
                    self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                    self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                    self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T


        duration = time.perf_counter() - start
        print('Avg. iterations: {}'.format(np.mean(iterations)))
        print('Simulation time: {}'.format(duration))

    def get_phi(self, t):
        # includes all kinematic constraints
        Phi_K = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
        # create euler parameter constraints to add to full Phi
        Phi_p = np.zeros((self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                Phi_p[idx] = 0.5*body.p.T @ body.p - 0.5
                idx += 1
        return np.concatenate((Phi_K, Phi_p), axis=0)

    def get_phi_q(self):
        jacobian = np.zeros((self.nc + self.nb, 7 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            # subtract 1 since ground body does not show up in jacobian
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
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                jacobian[row_euler, offset + 4 * (body.body_id - 1):offset + 4 * (body.body_id - 1) + 4] = body.p.T
                row_euler += 1

        return jacobian

    def get_nu(self, t):
        nu_G = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
        nu_euler = np.zeros((self.nb, 1))
        return np.concatenate((nu_G, nu_euler), axis=0)

    def get_gamma(self, t):
        gamma_G = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
        gamma_euler = np.zeros((self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                gamma_euler[idx, :] = -body.p_dot.T @ body.p_dot
                idx += 1
        return np.concatenate((gamma_G, gamma_euler), axis=0)

    def get_M(self):
        m_mat = np.zeros((3 * self.nb, 3 * self.nb))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                m_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.m * np.eye(3)
                idx += 1
        return m_mat

    def get_J_P(self):
        j_p_mat = np.zeros((4 * self.nb, 4 * self.nb))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                G = gcons.g_mat(body.p)
                j_p_mat[idx * 4:idx * 4 + 4, idx * 4:idx * 4 + 4] = 4 * G.T @ body.J @ G
                idx += 1
        return j_p_mat

    def get_P(self):
        p_mat = np.zeros((self.nb, 4 * self.nb))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                p_mat[idx, 4 * idx:4 * idx + 4] = body.p.T
                idx += 1
        return p_mat

    def get_F_g(self):
        # return F when gravity is the only force
        f_g_mat = np.zeros((3 * self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
                idx += 1
        return f_g_mat

    def get_tau(self):
        tau = np.zeros((4 * self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                G_dot = gcons.g_dot_mat(body.p_dot)
                tau[idx * 4:idx * 4 + 4] = 8 * G_dot.T @ body.J @ G_dot @ body.p
                idx += 1
        return tau

    def residual(self, order, t):
        if order == 1:
            beta_0 = 1
        elif order == 2:
            beta_0 = 2 / 3
        else:
            logging.warning("BDF of order greater than 2 not implemented yet.")

        r_ddot = np.vstack([body.r_ddot for body in self.bodies_list if body.is_ground == False])
        p_ddot = np.vstack([body.p_ddot for body in self.bodies_list if body.is_ground == False])

        Phi = self.get_phi(t)[:self.nc, :]
        Phi_euler = self.get_phi(t)[self.nc:self.nc + self.nb, :]
        Phi_r = self.get_phi_q()[0:self.nc, 0:3 * self.nb]
        Phi_p = self.get_phi_q()[0:self.nc, 3 * self.nb:]

        g_row1 = self.get_M() @ r_ddot + Phi_r.T @ self.lam - self.get_F_g()
        g_row2 = self.get_J_P() @ p_ddot + Phi_p.T @ self.lam \
                 + self.get_P().T @ self.lambda_p - self.get_tau()
        g_row3 = 1 / (beta_0 ** 2 * self.h ** 2) * Phi_euler
        g_row4 = 1 / (beta_0 ** 2 * self.h ** 2) * Phi
        g = np.block([[g_row1],
                      [g_row2],
                      [g_row3],
                      [g_row4]])
        return g

    def psi(self):
        M = self.get_M()
        J_P = self.get_J_P()
        P = self.get_P()
        Phi_r = self.get_phi_q()[0:self.nc, 0:3 * self.nb]
        Phi_p = self.get_phi_q()[0:self.nc, 3 * self.nb:]

        # build Psi, our quasi-newton iteration matrix
        zero_block_12 = np.zeros((3 * self.nb, 4 * self.nb))
        zero_block_13 = np.zeros((3 * self.nb, self.nb))
        zero_block_21 = np.zeros((4 * self.nb, 3 * self.nb))
        zero_block_31 = np.zeros((self.nb, 3 * self.nb))
        zero_block_33 = np.zeros((self.nb, self.nb))
        zero_block_34 = np.zeros((self.nb, self.nc))
        zero_block_43 = np.zeros((self.nc, self.nb))
        zero_block_44 = np.zeros((self.nc, self.nc))
        psi = np.block([[M, zero_block_12, zero_block_13, Phi_r.T],
                        [zero_block_21, J_P, P.T, Phi_p.T],
                        [zero_block_31, P, zero_block_33, zero_block_34],
                        [Phi_r, Phi_p, zero_block_43, zero_block_44]])

        return psi

    def reaction_torque(self):
        Phi_p = self.get_phi_q()[0:self.nc, 3:]  # this isn't going to be right with more than one body
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                pi = 1 / 2 * Phi_p @ gcons.e_mat(body.p).T
                torque = -pi.T @ self.lam
                idx += 1

        return torque


class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']
            self.r = np.zeros((3,1))
            self.r_dot = np.zeros((3,1))
            self.r_ddot = np.zeros((3,1))
            self.p = np.array([[1],
                               [0],
                               [0],
                               [0]])
            self.p_dot = np.zeros((4,1))
            self.p_ddot = np.zeros((4,1))

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
            self.r_ddot = np.zeros((3,1))

            quat = np.array(
                    [Rot.from_matrix(np.array(body_dict['A'])).as_quat()]).T
            self.p = gcons.to_scalar_first(quat)

            omega = np.array([body_dict['omega']]).T
            self.p_dot = 1 / 2 * gcons.e_mat(self.p).T @ omega
            self.p_ddot = np.zeros((4,1))

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
