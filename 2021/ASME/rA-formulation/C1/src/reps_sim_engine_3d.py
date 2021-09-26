#!/usr/bin/env python3

import reps_gcons as gcons

import json as js
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

import time
import logging

from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as Rot

class repsSimEngine3D:
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
        self.avg_iterations = 0

        self.init_system(filename)

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

    def initialize_plotting(self):
        self.tspan = self.t_end - self.t_start
        self.N = int(self.tspan / self.h)
        self.t_grid = np.linspace(self.t_start, self.t_end, self.N, endpoint=True)
        self.r_sol = np.zeros((self.N, 3 * self.nb))
        self.r_dot_sol = np.zeros((self.N, 3 * self.nb))
        self.r_ddot_sol = np.zeros((self.N, 3 * self.nb))

    def kinematics_solver(self):
        #logging.info("Number of bodies counted:", self.nb)
        self.initialize_plotting()
        nb = self.nb
        iterations = np.zeros((self.N, 1))

        start = time.process_time()
        for i, t in enumerate(self.t_grid):
            # check for configuration singularity
            for body in self.bodies_list:
                if body.near_singular:
                    value, flip_mat = body.compute_new_frame()
                    body.eps = value
                    for con in self.constraint_list:
                        con.flip_gcons(body.body_id, flip_mat)
            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                #logging.info("Switching to alternative constraint. Time = ", t)
                if self.alternative_driver is None:
                    #logging.warning("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

            Phi_q = self.get_phi_q()
            Phi_q_lu = lu_factor(Phi_q)

            iteration = 0
            while True:
                Phi = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
                delta_q = lu_solve(Phi_q_lu, -Phi)

                for idx, body in enumerate(self.bodies_list):
                    body.r = body.r + delta_q[idx * 3:(idx * 3) + 3, :]
                    body.eps = body.eps + delta_q[3 * nb + idx * 3:3 * nb + idx * 3 + 3, :]

                iteration += 1
                if iteration >= self.max_iters:
                    #logging.warning("Newton-Raphson self.has not converged after", str(self.max_iters), "iterations. Stopping at time ", str(t))
                    break
                if np.linalg.norm(delta_q) < self.tol:
                    break
            #logging.info("Newton-Raphson took", str(iteration), "iterations to converge.")
            iterations[i] = iteration

            Phi_q = self.get_phi_q()
            Phi_q_lu = lu_factor(Phi_q)
            # calculate velocity
            nu = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
            q_dot = lu_solve(Phi_q_lu, nu)
            for idx, body in enumerate(self.bodies_list):
                body.r_dot = q_dot[idx* 3:idx * 3 + 3, :]
                body.eps_dot = q_dot[3 * nb + idx * 3:3 * nb + idx * 3 + 3, :]

            # calculate acceleration
            gamma = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
            q_ddot = lu_solve(Phi_q_lu, gamma)
            for idx, body in enumerate(self.bodies_list):
                body.r_ddot = q_ddot[idx * 3:idx * 3 + 3, :]
                body.eps_ddot = q_ddot[3 * nb + idx * 3:3 * nb + idx * 3 + 3, :]

                # store solution in array for plotting
                self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

        duration = time.process_time() - start
        self.avg_iterations = np.mean(iterations)
        # logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        # print('Simulation time: {}'.format(duration))

    def dynamics_solver(self, order=1):
        # logging.info("Number of bodies counted:", self.nb)
        self.initialize_plotting()
        nb = self.nb
        nc = self.nc
        h = self.h

        # build full RHS matrix
        F = self.get_F_g()
        tau = self.get_tau()
        gamma = np.concatenate([con.gamma(self.t_start) for con in self.constraint_list], axis=0)
        eom_rhs = np.block([[F],
                            [tau],
                            [gamma]])
        M = self.get_M()
        J = self.get_J()
        Phi_q = self.get_phi_q()
        Phi_r = Phi_q[0:nc, 0:3 * nb]
        Phi_eps = Phi_q[0:nc, 3 * nb:]

        # build Psi, our quasi-newton iteration matrix
        zero_block_12 = np.zeros((3 * nb, 3 * nb))
        zero_block_21 = np.zeros((3 * nb, 3 * nb))
        zero_block_33 = np.zeros((nc, nc))
        psi = np.block([[M, zero_block_12, Phi_r.T],
                        [zero_block_21, J, Phi_eps.T],
                        [Phi_r, Phi_eps, zero_block_33]])

        # solve to find initial accelerations and lagrange multipliers, vector z
        z = np.linalg.solve(psi, eom_rhs)
        for idx, body in enumerate(self.bodies_list):
            body.r_ddot = z[idx * 3:idx * 3 + 3, :]
            body.eps_ddot = z[3 * nb + idx * 3:3 * nb + idx * 3 + 3, :]

            # store solution in array for plotting
            self.r_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
            self.r_dot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
            self.r_ddot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

            body.r_prev = body.r
            body.eps_prev = body.eps
            body.r_dot_prev = body.r_dot
            body.eps_dot_prev = body.eps_dot

        self.lam = z[6 * self.nb:]

        iterations = np.zeros((self.N, 1))
        start = time.process_time()
        for i, t in enumerate(self.t_grid):
            if i == 0:
                continue

            # check for configuration singularity
            for body in self.bodies_list:
                if body.near_singular:
                    value, flip_mat = body.compute_new_frame()
                    body.eps = value
                    for con in self.constraint_list:
                        con.flip_gcons(body.body_id, flip_mat)
            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                # logging.info("Switching to alternative constraint. Time = ", t)
                if self.alternative_driver is None:
                    # logging.warning("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[
                    -1]

            if i == 1 or order == 1:
                beta = 1
                alphas = np.array([1, 0])
            else:
                beta = 2 / 3
                alphas = np.array([4 / 3, -1 / 3])

            for body in self.bodies_list:
                body.c_r_dot = alphas[0] * body.r_dot + alphas[1] * body.r_dot_prev
                body.c_eps_dot = alphas[0] * body.eps_dot + alphas[1] * body.eps_dot_prev

                body.c_r = alphas[0] * body.r + alphas[1] * body.r_prev + beta * self.h * body.c_r_dot
                body.c_eps = alphas[0] * body.eps + alphas[1] * body.eps_prev + beta * self.h * body.c_eps_dot

                body.r_prev = body.r
                body.eps_prev = body.eps
                body.r_dot_prev = body.r_dot
                body.eps_dot_prev = body.eps_dot

            # build Psi, our quasi-newton iteration matrix
            J = self.get_J()
            Phi_q = self.get_phi_q()
            Phi_r = Phi_q[0:nc, 0:3 * nb]
            Phi_eps = Phi_q[0:nc, 3 * nb:]
            psi = np.block([[M, zero_block_12, Phi_r.T],
                            [zero_block_21, J, Phi_eps.T],
                            [Phi_r, Phi_eps, zero_block_33]])
            psi_lu = lu_factor(psi)

            # Begin Newton Iteration
            iteration = 0
            delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
            while delta_norm > self.tol:
                r_ddot_all = np.zeros((3 * nb, 1))
                eps_ddot_all = np.zeros((3 * nb, 1))
                for idx, body in enumerate(self.bodies_list):
                    r_ddot = body.r_ddot
                    eps_ddot = body.eps_ddot
                    body.r = body.c_r + beta ** 2 * h ** 2 * r_ddot
                    body.r_dot = body.c_r_dot + beta * h * r_ddot
                    body.eps = body.c_eps + beta ** 2 * h ** 2 * eps_ddot
                    body.eps_dot = body.c_eps_dot + beta * h * eps_ddot

                    r_ddot_all[3 * idx:3 * (idx + 1)] = r_ddot
                    eps_ddot_all[3 * idx:3 * (idx + 1)] = eps_ddot

                Phi = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
                Phi_q = self.get_phi_q()
                Phi_r = Phi_q[0:nc, 0:3 * nb]
                Phi_eps = Phi_q[0:nc, 3 * nb:]

                g_row1 = M @ r_ddot_all + Phi_r.T @ self.lam - F
                g_row2 = self.get_J() @ eps_ddot_all + Phi_eps.T @ self.lam - self.get_tau()
                g_row3 = 1 / (beta ** 2 * self.h ** 2) * Phi
                g = np.block([[g_row1],
                              [g_row2],
                              [g_row3]])

                delta = lu_solve(psi_lu, -g)

                for idx, body in enumerate(self.bodies_list):
                    body.r_ddot = body.r_ddot + delta[idx * 3:(idx * 3) + 3, :]
                    body.eps_ddot = body.eps_ddot + delta[3 * nb + idx * 3:3 * nb + idx * 3 + 3, :]

                self.lam += delta[6 * nb:]

                delta_norm = np.linalg.norm(delta)
                iteration += 1
                if iteration >= self.max_iters:
                    # logging.info("Solution self.has not converged after", str(self.max_iters), "iterations. Stopping. Time = ", t)
                    break

            iterations[i] = iteration

            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    # store solution in array for plotting
                    self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                    self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                    self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

        self.duration = time.process_time() - start
        self.avg_iterations = np.mean(iterations)
        # logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        print('Simulation time: {}'.format(self.duration))

    def get_phi_q(self):
        jacobian = np.zeros((self.nc, 6 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            idi = con.body_i.body_id - 1
            idj = con.body_j.body_id - 1
            if con.body_i.is_ground:
                # fill row of jacobian with only body j
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()
                jacobian[row, offset + 3 * idj:offset + 3 * idj + 3] = con.partial_eps()
            elif con.body_j.is_ground:
                # fill row of jacobian with only body i
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()
                jacobian[row, offset + 3 * idi:offset + 3 * idi + 3] = con.partial_eps()
            else:
                # fill row of jacobian with both body i and body j
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()[0]
                jacobian[row, offset + 3 * idi:offset + 3 * idi + 3] = con.partial_eps()[0]
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()[1]
                jacobian[row, offset + 3 * idj:offset + 3 * idj + 3] = con.partial_eps()[1]

        return jacobian

    def get_M(self):
        m_mat = np.zeros((3 * self.nb, 3 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            m_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.m * np.eye(3)
        return m_mat

    def get_J(self):
        j_mat = np.zeros((3 * self.nb, 3 * self.nb))
        for idx, body in enumerate(self.bodies_list):
            B_bar = body.A.T @ body.B
            j_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = B_bar.T @ body.J @ B_bar
        return j_mat

    def get_F_g(self):
        # return F when gravity is the only force
        f_g_mat = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
        return f_g_mat

    def get_tau(self):
        tau = np.zeros((3 * self.nb, 1))
        for idx, body in enumerate(self.bodies_list):
            B_bar = body.A.T @ body.B
            term_1 = body.B.T @ gcons.skew(B_bar @ body.eps_dot) @ body.J @ body.B @ body.eps_dot
            term_2 = B_bar.T @ body.J @ body.B_dot @ body.eps_dot
            tau[idx * 3:idx * 3 + 3] = term_1 - term_2
        return tau

def from_eps(eps):
    """
    Deconstruct eps into our three angles
    """
    return eps[0, 0], eps[1, 0], eps[2, 0]

class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']

            self.r = np.zeros((3,1))
            self.r_dot = np.zeros((3,1))
            self.r_ddot = np.zeros((3,1))

            eps = np.zeros((3,1))
            self.eps = eps
            self.eps_dot = np.zeros((3,1))
            self.eps_ddot = np.zeros((3,1))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.eps_prev = eps
            self.eps_dot_prev = self.eps_dot

            self.c_r = None
            self.c_eps = None
            self.c_r_dot = None
            self.c_eps_dot = None

            self.m = None
            self.J = None

            self.A = np.eye(3,3)
            self.A_dot = np.zeros((3, 3))
            self.A_ddot = np.zeros((3, 3))
        else:
            self.is_ground = False
            self.body_id = body_dict['id']

            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.r_ddot = np.zeros((3,1))

            self.A = Rot.from_matrix(np.array(body_dict['A']))
            eps = self.A.as_euler('ZXZ', degrees=False)
            eps = np.asmatrix(eps).T
            self.eps = eps
            self.eps_dot = np.zeros((3, 1))
            self.eps_ddot = np.zeros((3, 1))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.eps_prev = eps
            self.eps_dot_prev = self.eps_dot

            self.c_r = None
            self.c_eps = None
            self.c_r_dot = None
            self.c_eps_dot = None

            self.m = 0
            self.J = np.zeros((3, 3))

            self.A_dot = np.zeros((3, 3))
            self.A_ddot = np.zeros((3, 3))

    def compute_new_frame(self):
        flip_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        new_A = flip_mat @ self.A
        rot = Rot.from_matrix(new_A)
        value = np.array([rot.as_euler('ZXZ', degrees=False)]).T

        return value, flip_mat

    def get_partials(self):
        return (self.A_phi, self.A_theta, self.A_psi)

    def get_As(self):
        A1 = np.array([[self._cos_phi, -self._sin_phi, 0],
                       [self._sin_phi, self._cos_phi, 0], [0, 0, 1]])
        A2 = np.array([[1, 0, 0], [0, self._cos_theta, -self._sin_theta],
                       [0, self._sin_theta, self._cos_theta]])
        A3 = np.array([[self._cos_psi, -self._sin_psi, 0],
                       [self._sin_psi, self._cos_psi, 0], [0, 0, 1]])

        return A1, A2, A3

    def get_A_dots(self):
        A_dot1 = np.array([[-self._sin_phi, -self._cos_phi, 0],
                        [self._cos_phi, -self._sin_phi, 0], [0, 0, 0]])
        A_dot2 = np.array([[0, 0, 0], [0, -self._sin_theta, -self._cos_theta],
                        [0, self._cos_theta, -self._sin_theta]])
        A_dot3 = np.array([[-self._sin_psi, -self._cos_psi, 0],
                        [self._cos_psi, -self._sin_psi, 0], [0, 0, 0]])

        return A_dot1, A_dot2, A_dot3

    def get_A_ddots(self):
        A_ddot1 = np.array([[-self._cos_phi, self._sin_phi, 0],
                         [-self._sin_phi, -self._cos_phi, 0], [0, 0, 0]])
        A_ddot2 = np.array([[0, 0, 0], [0, -self._cos_theta, self._sin_theta],
                         [0, -self._sin_theta, -self._cos_theta]])
        A_ddot3 = np.array([[-self._cos_psi, self._sin_psi, 0],
                         [-self._sin_psi, -self._cos_psi, 0], [0, 0, 0]])

        return A_ddot1, A_ddot2, A_ddot3

    def cache_sin_cos(self, phi, theta, psi):
        """
        From a given set of Euler angles, computes and caches their sine and cosine in local properties
        """
        self._cos_phi = np.cos(phi)
        self._sin_phi = np.sin(phi)

        self._cos_theta = np.cos(theta)
        self._sin_theta = np.sin(theta)

        self._cos_psi = np.cos(psi)
        self._sin_psi = np.sin(psi)

    def cache_A_partials(self):
        """
        Based on the cached sine/cosine values, computes and caches the partial derivative of the rotation matrix A with
        respect to the three euler angles
        """

        self.A_phi = np.array([[-self._sin_psi * self._cos_theta * self._cos_phi - self._sin_phi * self._cos_psi,
                               self._sin_psi * self._sin_phi - self._cos_theta * self._cos_psi * self._cos_phi, self._sin_theta * self._cos_phi],
                              [-self._sin_psi * self._sin_phi * self._cos_theta + self._cos_psi * self._cos_phi,
                               -self._sin_psi * self._cos_phi - self._sin_phi * self._cos_theta * self._cos_psi, self._sin_theta * self._sin_phi], [0, 0, 0]])
        self.A_theta = np.array([[self._sin_theta * self._sin_psi * self._sin_phi, self._sin_theta * self._sin_phi * self._cos_psi, self._sin_phi * self._cos_theta],
                              [-self._sin_theta * self._sin_psi * self._cos_phi, -self._sin_theta * self._cos_psi * self._cos_phi, -self._cos_theta * self._cos_phi],
                              [self._sin_psi * self._cos_theta, self._cos_theta * self._cos_psi, -self._sin_theta]])
        self.A_psi = np.array([[-self._sin_psi * self._cos_phi - self._sin_phi * self._cos_theta * self._cos_psi,
                               self._sin_psi * self._sin_phi * self._cos_theta - self._cos_psi * self._cos_phi, 0],
                              [-self._sin_psi * self._sin_phi + self._cos_theta * self._cos_psi * self._cos_phi,
                               -self._sin_psi * self._cos_theta * self._cos_phi - self._sin_phi * self._cos_psi, 0],
                              [self._sin_theta * self._cos_psi, -self._sin_theta * self._sin_psi, 0]])

    def cache_time_derivs(self):
        """
        Computes time derivative terms needed by g-cons in the computation of γ and caches these so that g-cons don't
        have to re-compute them
        """
        phi_dot, theta_dot, psi_dot = from_eps(self.eps_dot)

        A1, A2, A3 = self.get_As()
        A_dot1, A_dot2, A_dot3 = self.get_A_dots()
        A_ddot1, A_ddot2, A_ddot3 = self.get_A_ddots()

        # Terms related to Ȧ
        A_dot_phi = phi_dot * A_dot1 @ A2 @ A3
        A_dot_theta = theta_dot * A1 @ A_dot2 @ A3
        A_dot_psi = psi_dot * A1 @ A2 @ A_dot3

        # Ȧ
        self.A_dot = A_dot_phi + A_dot_theta + A_dot_psi

        # Terms related to Ä
        phi_theta = 2 * phi_dot * theta_dot * A_dot1 @ A_dot2 @ A3
        theta_psi = 2 * theta_dot * psi_dot * A1 @ A_dot2 @ A_dot3
        phi_psi = 2 * phi_dot * psi_dot * A_dot1 @ A2 @ A_dot3
        A_ddot_phi = phi_dot ** 2 * A_ddot1 @ A2 @ A3
        A_ddot_theta = theta_dot ** 2 * A1 @ A_ddot2 @ A3
        A_ddot_psi = psi_dot ** 2 * A1 @ A2 @ A_ddot3

        # What we denote as Ä_γ - Ä with all second derivative terms removed
        self.A_ddot = A_ddot_phi + A_ddot_theta + A_ddot_psi + phi_theta + theta_psi + phi_psi

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        """
        Whenever we set eps, cache A for future use
        """

        rot = Rot.from_euler('ZXZ', value.T, degrees=False)
        # If we keep value as a 1x3 array it will give A an extra dimension, so we squeeze away the extra dim
        self.A = np.squeeze(rot.as_matrix())

        phi, theta, psi = from_eps(value)

        self.near_singular = np.abs(np.fmod(theta, np.pi)) < 0.1 and (not self.is_ground)

        self.cache_sin_cos(phi, theta, psi)
        self.cache_A_partials()

        self._eps = value

    @property
    def B(self):
        B = np.array([[0, self._cos_phi, self._sin_theta * self._sin_phi],
                      [0, self._sin_phi, -self._sin_theta * self._cos_phi], [1, 0, self._cos_theta]])
        return B

    @property
    def B_dot(self):
        phi_dot, theta_dot, psi_dot = from_eps(self.eps_dot)

        B_dot = np.array([[0, -phi_dot * self._sin_phi, theta_dot * self._cos_theta * self._sin_phi + phi_dot * self._sin_theta * self._cos_phi], [0,
                                                                                                  phi_dot * self._sin_phi,
                                                                                                  -theta_dot * self._cos_theta * self._cos_phi + phi_dot * self._sin_theta * self._sin_phi],
                       [0, 0, -theta_dot * self._sin_theta]])
        B_dot_bar = np.array([[psi_dot * self._cos_psi * self._sin_theta + theta_dot * self._sin_psi * self._cos_theta, -psi_dot * self._sin_psi, 0],
                           [-psi_dot * self._sin_psi * self._sin_theta + theta_dot * self._cos_psi * self._cos_theta, -psi_dot * self._cos_psi, 0],
                           [-theta_dot * self._sin_theta, 0, 0]])
        return B_dot_bar

    @property
    def omega(self):
        return self.A.T @ self.B @ self.eps_dot
