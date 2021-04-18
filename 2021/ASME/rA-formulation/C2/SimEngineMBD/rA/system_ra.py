import logging
import json as js

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .gcons_ra import Constraints, DP1, DP2, CD, D, Body, ConGroup
from ..utils.physics import Z_AXIS, block_mat, R, skew, exp, SolverType
from ..utils.systems import read_model_file


class SystemRA:

    def __init__(self, bodies, constraints):
        self.bodies = bodies
        self.g_cons = constraints
        self.solver_type = SolverType.KINEMATICS
        self.solver_order = 1

        self.nc = self.g_cons.nc
        self.nb = self.g_cons.nb

        assert self.nb == len(self.bodies), "Mismatch on number of bodies"

        self.g_acc = np.zeros((3, 1))

        self.is_initialized = False

        # Set solver parameters
        self.h = 1e-3
        self.tol = None
        self.max_iters = 50
        self.k = 0

        # Set physical quantities
        self.M = np.zeros((3*self.nb, 3*self.nb))
        self.J = np.zeros((3*self.nb, 3*self.nb))
        self.F_ext = np.zeros((3*self.nb, 1))

        self.Φ = np.zeros((self.nc, 1))
        self.Φ_r = np.zeros((self.nc, 3*self.nb))
        self.Π = np.zeros((self.nc, 3*self.nb))
        self.λ = np.zeros((self.nc, 1))

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

        # TODO print info about system setup here

        for body in self.bodies:
            body.F += body.m * self.g_acc

        self.M = np.diagflat([[body.m] * 3 for body in self.bodies])
        self.J = block_mat([body.J for body in self.bodies])
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

            self.initialize_dynamics()

        self.is_initialized = True

    def initialize_dynamics(self):

        logging.info('Initializing system for dynamics')

        t_start = 0

        # Compute initial values
        self.Φ_r = self.g_cons.get_phi_r(t_start)
        self.Π = self.g_cons.get_pi(t_start)

        # Quantities for the right-hand side:
        # Fg is constant, defined above
        τ = np.vstack([body.get_tau() for body in self.bodies])
        γ = self.g_cons.get_gamma(t_start)

        G_ωω = block_mat([body.get_J_term(self.h) for body in self.bodies])
        G = np.block([[self.M, np.zeros((3*self.nb, 3*self.nb)), self.Φ_r.T], [np.zeros((3*self.nb, 3*self.nb)), G_ωω, self.Π.T],
                      [self.Φ_r, self.Π, np.zeros((self.nc, self.nc))]])

        g = np.block([[self.F_ext], [τ], [γ]])

        z = np.linalg.solve(G, g)

        for i, body in enumerate(self.bodies):
            body.ddr = z[3*i:3*(i+1)]
            body.dω = z[3*self.nb + 3*i:3*self.nb + 3*(i+1)]

            body.cache_rA_values()

        self.λ = z[6*self.nb:]

        self.do_step = self.do_dynamics_step

    def do_step(self, i, t):
        if self.solver_type == SolverType.KINEMATICS:
            self.do_kinematics_step(t)
        else:
            self.do_dynamics_step(i, t)

    def do_dynamics_step(self, i, t):

        assert self.is_initialized, "Cannot dyn_step before system initialization"

        if i == 0:
            return

        self.g_cons.maybe_swap_gcons(t)

        self.Φ_r = self.g_cons.get_phi_r(t)
        self.Π = self.g_cons.get_pi(t)

        G_rω = np.zeros((3*self.nb, 3*self.nb))
        G_ωr = np.zeros((3*self.nb, 3*self.nb))
        G_ωω = block_mat([body.get_J_term(self.h) for body in self.bodies])
        G = np.block([[self.M, G_rω, self.Φ_r.T], [G_ωr, G_ωω, self.Π.T],
                      [self.Φ_r, self.Π, np.zeros((self.nc, self.nc))]])

        G_lu = lu_factor(G)

        for body in self.bodies:
            body.cache_rA_values()

        # Setup and do Newton-Raphson Iteration
        self.k = 0
        while True:
            for body in self.bodies:
                body.dr = body.dr_prev + self.h*body.ddr
                body.ω = body.ω_prev + self.h*body.dω

                body.r = body.r_prev + self.h*body.dr
                body.A = body.A_prev @ exp(self.h*skew(body.ω))

            # Compute values needed for the g matrix
            # We can't move this outside the loop since the g_cons
            #   use e.g. body.A in their computations and body.A gets updated as we iterate
            self.Φ = self.g_cons.get_phi(t)
            self.Φ_r = self.g_cons.get_phi_r(t)
            self.Π = self.g_cons.get_pi(t)

            ddr = np.vstack([body.ddr for body in self.bodies])

            # Form g matrix
            g0 = self.M @ ddr + self.Φ_r.T @ self.λ - self.F_ext
            g1 = np.vstack([body.J @ body.dω + (skew(body.ω) @ body.J @ body.ω)
                            for body in self.bodies]) + self.Π.T @ self.λ
            g2 = 1/self.h**2 * self.Φ
            g = np.block([[g0], [g1], [g2]])

            δ = lu_solve(G_lu, -g)

            for j, body in enumerate(self.bodies):
                body.ddr += δ[3*j:3*(j+1)]
                body.dω += δ[3*(self.nb + j):3*(self.nb + j+1)]

            self.λ += δ[6*self.nb:]

            # logging.debug('t: {:.3f}, k: {:>2d}, norm: {:6.6e}'.format(
            #     t, self.k, np.linalg.norm(δ)))

            if np.linalg.norm(δ) < self.tol:
                break

            self.k += 1
            if self.k >= self.max_iters:
                raise RuntimeError(
                    'Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(t, self.max_iters))

        # logging.debug('t: {:.3f}, iterations: {:>2d}'.format(t, self.k))

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
