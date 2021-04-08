import sys
import pathlib as pl

src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

import rA_gcons as gcons

import json as js
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

import time
import logging
from scipy.linalg import lu_factor, lu_solve


# ================================== System utility functions =====================================
def R(u_bar, chi):
    """Return the value of the rotation matrix calculated based on Rodrigues's formula."""
    return np.cos(chi) * np.eye(3) + (1 - np.cos(chi)) * (u_bar @ u_bar.T) + np.sin(chi) * gcons.skew(u_bar)


# ==================================== 3D Simulation Engine =======================================
class rASimEngine3D:
    def __init__(self, filename):
        self.bodies_list = []
        self.nb = 0  # number of bodies that don't include the ground!
        self.constraint_list = []
        self.nc = 0

        # simulation parameters
        self.h = 0.001
        self.t_start = 0
        self.t_end = 1
        self.tspan = None
        self.N = None
        self.t_grid = None
        self.tol = None
        self.max_iters = 20
        self.duration = 0
        self.iterations = 0

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

            Phi_q = self.get_sensitivities()
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
                        theta = delta_q[3 * self.nb + (body.body_id - 1) * self.nb:3 * self.nb + (
                                    body.body_id - 1) * self.nb + 3, :]
                        theta_norm = np.linalg.norm(theta)
                        if theta_norm != 0:
                            body.A = body.A @ R(theta / theta_norm, theta_norm)

                iteration += 1
                if iteration >= self.max_iters:
                    logging.warning("Newton-Raphson has not converged after", str(self.max_iters),
                                    "iterations. Stopping.")
                    break
                if np.linalg.norm(delta_q) < self.tol:
                    break
            logging.info("Newton-Raphson took", str(iteration), "iterations to converge.")
            iterations[i] = iteration

            Phi_q = self.get_sensitivities()
            Phi_q_lu = lu_factor(Phi_q)
            # calculate velocity
            q_dot = lu_solve(Phi_q_lu, self.get_nu(t))
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_dot = q_dot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                    body.omega = q_dot[3 * self.nb + (body.body_id - 1) * self.nb:3 * self.nb + (
                                body.body_id - 1) * self.nb + 3, :]

            # calculate acceleration
            Phi_q = self.get_sensitivities()
            Phi_q_lu = lu_factor(Phi_q)
            q_ddot = lu_solve(Phi_q_lu, self.get_gamma(t))
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_ddot = q_ddot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                    body.omega_dot = q_ddot[3 * self.nb + (body.body_id - 1) * self.nb:3 * self.nb + (
                                body.body_id - 1) * self.nb + 3, :]
                    # store solution in array for plotting
                    self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                    self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                    self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

        self.duration = time.perf_counter() - start
        self.avg_iterations = np.mean(iterations)
        logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        print('Simulation time: {}'.format(self.duration))

    def dynamics_solver(self):
        logging.info("Number of bodies counted: ", self.nb)
        self.initialize_plotting()

        # Need to solve for initial accelerations using our EOM and constraints
        F = self.get_F_g()
        tau = self.get_tau()
        gamma = self.get_gamma(self.t_start)
        eom_rhs = np.block([[F],
                            [tau],
                            [gamma]])

        # solve to find initial accelerations and lagrange multipliers, vector z
        z = np.linalg.solve(self.get_psi(), eom_rhs)
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                body.r_ddot = z[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3]
                body.omega_dot = z[3 * self.nb + (body.body_id - 1) * 3:3 * self.nb + (body.body_id - 1) * 3 + 3]
                # for plotting
                self.r_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                self.r_dot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                self.r_ddot_sol[0, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

                body.r_prev = body.r
                body.A_prev = body.A
                body.r_dot_prev = body.r_dot
                body.omega_prev = body.omega

        self.lam = z[6 * self.nb:]

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

            psi = self.get_psi()
            psi_lu = lu_factor(psi)
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_prev = body.r
                    body.A_prev = body.A
                    body.r_dot_prev = body.r_dot
                    body.omega_prev = body.omega

            # Begin Newton Iteration
            iteration = 0
            delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
            while delta_norm > self.tol:
                for body in self.bodies_list:
                    if body.is_ground:
                        pass
                    else:
                        body.r_dot = body.r_dot_prev + self.h * body.r_ddot
                        body.omega = body.omega_prev + self.h * body.omega_dot
                        body.r = body.r_prev + self.h * body.r_dot
                        theta_mat = self.h * gcons.skew(body.omega)
                        theta = np.array([[theta_mat[2,1]], [theta_mat[0,2]], [theta_mat[1,0]]])
                        theta_norm = np.linalg.norm(theta)
                        if theta_norm != 0:
                            body.A = body.A_prev @ R(theta / theta_norm, theta_norm)

                g = self.get_residual(t)
                delta = lu_solve(psi_lu, -g)

                for body in self.bodies_list:
                    if body.is_ground:
                        pass
                    else:
                        body.r_ddot = body.r_ddot + delta[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3]
                        body.omega_dot = body.omega_dot + delta[3 * self.nb + (body.body_id - 1) * 3:3 * self.nb + (body.body_id - 1) * 3 + 3]

                self.lam += delta[6 * self.nb:]

                delta_norm = np.linalg.norm(delta)
                iteration += 1
                if iteration >= self.max_iters:
                    logging.warning("Solution has not converged after", str(self.max_iters), "iterations. Stopping.")
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

        self.duration = time.perf_counter() - start
        self.avg_iterations = np.mean(iterations)
        logging.info('Avg. iterations: {}'.format(self.avg_iterations))
        print('Simulation time: {}'.format(self.duration))

    # ============================ System variable getter functions ===============================
    def get_phi(self, t):
        # includes all kinematic constraints
        Phi_K = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
        return Phi_K

    def get_sensitivities(self):
        jacobian = np.zeros((self.nc, 6 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            # subtract 1 since ground body does not show up in jacobian
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

    def get_nu(self, t):
        nu_G = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
        return nu_G

    def get_gamma(self, t):
        gamma_G = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
        return gamma_G

    def get_F_g(self):
        # returns F when gravity is the only force
        f_g_mat = np.zeros((3 * self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
                idx += 1
        return f_g_mat

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

    def get_J(self):
        j_mat = np.zeros((3 * self.nb, 3 * self.nb))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                j_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.J - self.h*(gcons.skew(body.J @ body.omega) -
                                                                                   gcons.skew(body.omega) @ body.J) #+ body.n_omega
                idx += 1
        return j_mat

    def get_tau(self):
        tau = np.zeros((3 * self.nb, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                n_bar = np.zeros((3, 1))
                omega_bar = body.omega
                tau[idx * 3:idx * 3 + 3] = n_bar - gcons.skew(omega_bar) @ body.J @ omega_bar
                idx += 1
        return tau

    def get_residual(self, t):
        r_ddot = np.zeros((3 * self.nb, 1))
        omega_bar = np.zeros((3 * self.nb, 1))
        omega_bar_dot = np.zeros((3 * self.nb, 1))
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                r_ddot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot
                omega_bar[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.omega
                omega_bar_dot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.omega_dot

        M = self.get_M()
        lam = self.lam
        F_g = self.get_F_g()
        Phi = self.get_phi(t)
        Phi_r = self.get_sensitivities()[:, :3 * self.nb]
        Pi = self.get_sensitivities()[:, 3 * self.nb:]

        g_row1 = M @ r_ddot + Phi_r.T @ lam - F_g
        g_row2 = np.zeros((3 * self.nb, 1))
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                g_row2[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.J @ body.omega_dot + (
                            gcons.skew(body.omega) @ body.J @ body.omega)
        g_row2 += Pi.T @ lam
        g_row3 = 1 / self.h ** 2 * Phi
        g = np.block([[g_row1],
                      [g_row2],
                      [g_row3]])

        return g

    def get_psi(self):
        M = self.get_M()
        J = self.get_J()
        Phi_r = self.get_sensitivities()[:, :3 * self.nb]
        Pi = self.get_sensitivities()[:, 3 * self.nb:]

        G_rr = M  # -self.h * F_r_dot -h**2 * (F_r + F^c_r) (zero for now)
        G_rw = np.zeros((3 * self.nb, 3 * self.nb))  # -h * F_omega_bar -h ** 2 * (Pi(F) + Pi(F^c))
        G_rlam = Phi_r.T
        G_wr = np.zeros((3 * self.nb, 3 * self.nb))  # -h * n_bar_r_dot -h**2(n_bar_r + n_bar^c_r_)
        G_ww = J
        G_wlam = Pi.T
        G_lamr = Phi_r
        G_lamw = Pi
        zero_block_33 = np.zeros((self.nc, self.nc))

        return np.block([[G_rr, G_rw, G_rlam],
                         [G_wr, G_ww, G_wlam],
                         [G_lamr, G_lamw, zero_block_33]])


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
            # self.omega_dot = np.array([body_dict['omega_dot']]).T
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
