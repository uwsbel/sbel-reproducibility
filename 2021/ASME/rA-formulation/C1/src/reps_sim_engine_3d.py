import sys
import pathlib as pl

src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

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
            # check for configuration singularity
            for body in self.bodies:
                if body.near_singular:
                    value, flip_mat = body.compute_new_frame()
                    body.eps = value
                    gcons.flip_gcons(body.body_id, flip_mat)
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
                        body.eps = body.eps + delta_q[3 * self.nb + (body.body_id - 1) * 3:3 * self.nb + (
                                body.body_id - 1) * 3 + 3, :]

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
                    body.eps_dot = q_dot[3 * self.nb + (body.body_id - 1) * 3:3 * self.nb + (
                                body.body_id - 1) * 3 + 3, :]

            # calculate acceleration
            q_ddot = lu_solve(Phi_q_lu, self.get_gamma(t))
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    body.r_ddot = q_ddot[(body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3, :]
                    body.eps_ddot = q_ddot[3 * self.nb + (body.body_id - 1) * 3:3 * self.nb + (
                                body.body_id - 1) * 3 + 3, :]

                    # store solution in array for plotting
                    self.r_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r.T
                    self.r_dot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_dot.T
                    self.r_ddot_sol[i, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3] = body.r_ddot.T

        duration = time.perf_counter() - start
        print('Avg. iterations: {}'.format(np.mean(iterations)))
        print('Simulation time: {}'.format(duration))

    def get_phi(self, t):
        # includes all kinematic constraints
        return np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)

    def get_phi_q(self):
        jacobian = np.zeros((self.nc + self.nb, 6 * self.nb))
        offset = 3 * self.nb

        for row, con in enumerate(self.constraint_list):
            # subtract 1 since ground body does not show up in jacobian
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

    def get_nu(self, t):
        return np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)

    def get_gamma(self, t):
        return np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)

class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']
            self.r = np.zeros((3,1))
            self.r_dot = np.zeros((3,1))
            self.r_ddot = np.zeros((3,1))
            self.eps = np.zeros((3,1))
            self.eps_dot = np.zeros((3,1))
            self.eps_ddot = np.zeros((3,1))

            self.A_dot = np.zeros((3, 3))
            self.A_ddot = np.zeros((3, 3))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.eps_prev = self.eps
            self.eps_dot_prev = self.eps_dot
            self.c_r = None
            self.c_eps = None
            self.c_r_dot = None
            self.c_eps_dot = None
            self.m = None
            self.J = None
        else:
            self.is_ground = False
            self.body_id = body_dict['id']
            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.r_ddot = np.zeros((3,1))

            self.A = Rot.from_matrix(np.array(dict['A']))
            eps = self.A.as_euler('ZXZ', degrees=False)
            self.eps = np.asmatrix(eps).T

            self.eps_dot = np.zeros((3, 1)) #@todo should this be nonzero?
            self.eps_ddot = np.zeros((3, 1))

            self.A_dot = np.zeros((3, 3))
            self.A_ddot = np.zeros((3, 3))

            self.r_prev = self.r
            self.r_dot_prev = self.r_dot
            self.eps_prev = self.eps
            self.eps_dot_prev = self.eps_dot
            self.c_r = None
            self.c_eps = None
            self.c_r_dot = None
            self.c_eps_dot = None
            self.m = 0
            self.J = np.zeros((3, 3))

    def compute_new_frame(self):
        flip_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        new_A = flip_mat @ self.A
        rot = Rot.from_matrix(new_A)
        value = np.array([rot.as_euler('ZXZ', degrees=False)]).T

        return value, flip_mat
