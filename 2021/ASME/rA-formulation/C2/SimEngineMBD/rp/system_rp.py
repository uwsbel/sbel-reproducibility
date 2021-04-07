import numpy as np
import json as js
import logging
from scipy.linalg import lu_factor, lu_solve

from .gcons_rp import Constraints, DP1, DP2, CD, D, Body, ConGroup, EulerCon
from ..utils.physics import Z_AXIS, block_mat, R, skew, exp, SolverType, bdf1, bdf2
from ..utils.systems import read_model_file

class SystemRP:

    def __init__(self, bodies, constraints):
        self.bodies = bodies
        self.g_cons = constraints
        self.e_cons = [EulerCon(body) for body in self.bodies]
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
        self.Jp = np.zeros((4*self.nb, 4*self.nb))
        self.P = np.zeros((4*self.nb, self.nb))
        self.F_ext = np.zeros((3*self.nb, 1))

        self.Φ = np.zeros((self.nc, 1))
        self.Φ_r = np.zeros((self.nc, 3*self.nb))
        self.Φ_p = np.zeros((self.nc, 4*self.nb))

        self.λp = np.zeros((self.nb, 1))
        self.λ = np.zeros((self.nc, 1))

    def set_dynamics(self):
        if self.is_initialized:
            logging.warning('Cannot change solver type on an initialized system')
        else:
            self.solver_type = SolverType.DYNAMICS

    def set_kinematics(self):
        if self.is_initialized:
            logging.warning('Cannot change solver type on an initialized system')
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
        self.Jp = block_mat([body.get_j() for body in self.bodies])
        self.P = block_mat([body.p for body in self.bodies])
        self.F_ext = np.vstack([body.F for body in self.bodies])

        if self.solver_type == SolverType.KINEMATICS:
            if self.nc == 6*self.nb:
                # Set tighter tolerance for kinematics
                self.tol = 1e-6 if self.tol == None else self.tol
                for e_con in self.e_cons:
                    self.g_cons.cons.append(e_con)
                    self.g_cons = ConGroup(self.g_cons.cons, self.nb)

                logging.info('Initializing system for kinematics')
            else:
                logging.warning('Kinematic system has nc ({}) < 6⋅nb ({}), running dynamics instead'.format(
                    self.nc, 6*self.nb))

                self.solver_type = SolverType.DYNAMICS
                self.tol = 1e-3 if self.tol == None else self.tol
        if self.solver_type == SolverType.DYNAMICS:
            if self.nc > 6*self.nb:
                logging.warning('System is overconstrained')

            self.tol = 1e-3 if self.tol == None else self.tol

            self.initialize_dynamics()

        self.is_initialized = True

    def initialize_dynamics(self):

        logging.info('Initializing system for dynamics')

        t_start = 0

        if False:
            # Preliminary kinematics step to settle the initial conditions
            kin_Φq = self.g_cons.get_phi_q(t_start)

            # kin_Φq has the euler constraints whereas self. versions don't
            for e_con in self.e_cons:
                row = np.zeros((1, 7*self.nb))

                for b_id, phiR in e_con.get_phi_r(t_start):
                    row[0, 3*b_id:3*(b_id + 1)] = phiR

                for b_id, phiP in e_con.get_phi_p(t_start):
                    row[0, 3*self.nb + 4*b_id:3*self.nb + 4*(b_id + 1)] = phiP

                kin_Φq = np.append(kin_Φq, row, axis=0)

            Φq_lu = lu_factor(kin_Φq)

            self.k = 0
            while True:
                kin_Φ = self.g_cons.get_phi(t_start)
                for e_con in self.e_cons:
                    kin_Φ = np.append(kin_Φ, e_con.get_phi(t_start), axis=0)

                Δq = lu_solve(Φq_lu, -kin_Φ)

                for j, body in enumerate(self.bodies):
                    Δr = Δq[3*j:3*(j+1)]
                    Δp = Δq[3*self.nb + 4*j:3*self.nb + 4*(j+1)]

                    body.r = body.r + Δr
                    body.p = body.p + Δp

                self.k += 1

                if np.linalg.norm(Δq) < 1e-9:
                    break

                if self.k >= self.max_iters:
                    raise RuntimeError('Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(t, self.max_iters))

            kin_Φq = self.g_cons.get_phi_q(t_start)

            # kin_Φq has the euler constraints whereas self. versions don't
            for e_con in self.e_cons:
                row = np.zeros((1, 7*self.nb))

                for b_id, phiR in e_con.get_phi_r(t_start):
                    row[0, 3*b_id:3*(b_id + 1)] = phiR

                for b_id, phiP in e_con.get_phi_p(t_start):
                    row[0, 3*self.nb + 4*b_id:3*self.nb + 4*(b_id + 1)] = phiP

                kin_Φq = np.append(kin_Φq, row, axis=0)

            Φq_lu = lu_factor(kin_Φq)
            kin_ν = self.g_cons.get_nu(t_start)
            for e_con in self.e_cons:
                kin_ν = np.append(kin_ν, [e_con.get_nu(t_start)], axis=0)

            kin_dq = np.linalg.solve(kin_Φq, kin_ν)
            # kin_dq = lu_solve(Φq_lu, kin_ν)
            for j, body in enumerate(self.bodies):
                body.dr = kin_dq[3*j:3*(j+1), :]
                body.dp = kin_dq[3*self.nb + 4*j:3*self.nb + 4*(j+1), :]

            kin_γ = self.g_cons.get_gamma(t_start)
            for e_con in self.e_cons:
                kin_γ = np.append(kin_γ, e_con.get_gamma(t_start), axis=0)

            kin_ddq = np.linalg.solve(kin_Φq, kin_γ)
            # kin_ddq = lu_solve(Φq_lu, kin_γ)
            for j, body in enumerate(self.bodies):
                body.ddr = kin_ddq[3*j:3*(j+1), :]
                body.ddp = kin_ddq[3*self.nb + 4*j:3*self.nb + 4*(j+1), :]

        # Compute initial values
        self.Φ = self.g_cons.get_phi(t_start)
        self.Φ_r = self.g_cons.get_phi_r(t_start)
        self.Φ_p = self.g_cons.get_phi_p(t_start)

        self.Jp = block_mat([body.get_j() for body in self.bodies])
        self.P = block_mat([body.p for body in self.bodies])

        # Quantities for the right-hand side
        γ = self.g_cons.get_gamma(t_start)
        γp = np.vstack([-body.dp.T @ body.dp for body in self.bodies])
        τ = np.vstack([body.get_tau() for body in self.bodies])
        # Fg is constant, defined above

        # Here we solve the larger system and redundantly retrieve r̈ and p̈
        Ψ = np.block([[self.M, np.zeros((3*self.nb, 4*self.nb)), np.zeros((3*self.nb, self.nb)), self.Φ_r.T], [np.zeros((4*self.nb, 3*self.nb)), self.Jp, self.P, self.Φ_p.T], [
            np.zeros((self.nb, 3*self.nb)), self.P.T, np.zeros((self.nb, self.nb)), np.zeros((self.nb, self.nc))], [self.Φ_r, self.Φ_p, np.zeros((self.nc, self.nb)), np.zeros((self.nc, self.nc))]])
        g = np.block([[self.F_ext], [τ], [γp], [γ]])

        z = np.linalg.solve(Ψ, g)

        for i, body in enumerate(self.bodies):
            body.ddr = z[3*i:3*(i+1)]
            body.ddp = z[3*self.nb + 4*i:3*self.nb + 4*(i+1)]

            body.cache_rp_values()

            self.λp[i] = z[7*self.nb + i]

        self.λ = z[8*self.nb:8*self.nb + self.nc]

    def do_step(self, i, t):
        if self.solver_type == SolverType.KINEMATICS:
            self.do_kinematics_step(t)
        else:
            self.do_dynamics_step(i, t)

    def do_dynamics_step(self, i, t):

        assert self.is_initialized, "Cannot dyn_step before system initialization"

        if i == 0:
            return

        self.bdf = bdf1 if (i == 1 or self.solver_order == 1) else bdf2
        for body in self.bodies:
            body.update_bdf_coeffs(self.bdf, self.h)

        self.P = block_mat([body.p for body in self.bodies])
        self.Jp = block_mat([body.get_j() for body in self.bodies])
        self.Φ_r = self.g_cons.get_phi_r(t)
        self.Φ_p = self.g_cons.get_phi_p(t)

        Ψ = np.block([[self.M, np.zeros((3*self.nb, 4*self.nb)), np.zeros((3*self.nb, self.nb)), self.Φ_r.T], [np.zeros((4*self.nb, 3*self.nb)), self.Jp, self.P, self.Φ_p.T], [np.zeros(
            (self.nb, 3*self.nb)), self.P.T, np.zeros((self.nb, self.nb)), np.zeros((self.nb, self.nc))], [self.Φ_r, self.Φ_p, np.zeros((self.nc, self.nb)), np.zeros((self.nc, self.nc))]])

        Ψ_lu = lu_factor(Ψ)

        for body in self.bodies:
            body.cache_rp_values()

        # Setup and do Newton-Raphson Iteration
        self.k = 0
        while True:
            for body in self.bodies:
                body.r = body.C_r + self.bdf.β**2 * self.h**2 * body.ddr
                body.p = body.C_p + self.bdf.β**2 * self.h**2 * body.ddp
                body.dr = body.C_dr + self.bdf.β*self.h*body.ddr
                body.dp = body.C_dp + self.bdf.β*self.h*body.ddp

            # Compute values needed for the g matrix
            # We can't move this outside the loop since the g_cons
            #   use e.g. body.A in their computations and body.A gets updated as we iterate
            self.Φ = self.g_cons.get_phi(t)
            self.Φ_r = self.g_cons.get_phi_r(t)
            self.Φ_p = self.g_cons.get_phi_p(t)

            self.Jp = block_mat([body.get_j() for body in self.bodies])
            self.P = block_mat([body.p for body in self.bodies])

            τ = np.vstack([body.get_tau() for body in self.bodies])

            ddr = np.vstack([body.ddr for body in self.bodies])
            ddp = np.vstack([body.ddp for body in self.bodies])

            # Form g matrix
            g0 = self.M @ ddr + self.Φ_r.T @ self.λ - self.F_ext
            g1 = self.Jp @ ddp + self.Φ_p.T @ self.λ + self.P @ self.λp - τ
            g2 = 1/(self.bdf.β**2 * self.h**2) * \
                np.vstack([e_con.get_phi(t) for e_con in self.e_cons])
            g3 = 1/(self.bdf.β**2 * self.h**2) * self.Φ
            g = np.block([[g0], [g1], [g2], [g3]])

            Δz = lu_solve(Ψ_lu, -g)

            for j, body in enumerate(self.bodies):
                body.ddr += Δz[3*j:3*(j+1)]
                body.ddp += Δz[3*self.nb + 4*j:3*self.nb + 4*(j+1)]

                self.λp[j] += Δz[7*self.nb + j]

            self.λ += Δz[8*self.nb:8*self.nb + self.nc]

            # logging.debug('t: {:.3f}, k: {:>2d}, norm: {:6.6e}'.format(
            #     t, self.k, np.linalg.norm(Δz)))

            if np.linalg.norm(Δz) < self.tol:
                break

            self.k += 1
            if self.k >= self.max_iters:
                raise RuntimeError('Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(
                    t, self.max_iters))

        # logging.debug('t: {:.3f}, iterations: {:>2d}'.format(t, self.k))

    def do_kinematics_step(self, t):

        # Refresh the inverse matrix with our new positions
        self.Φq = self.g_cons.get_phi_q(t)
        Φq_lu = lu_factor(self.Φq)

        self.k = 0
        while True:
            self.Φ = self.g_cons.get_phi(t)

            Δq = lu_solve(Φq_lu, -self.Φ)

            for j, body in enumerate(self.bodies):
                Δr = Δq[3*j:3*(j+1)]
                Δp = Δq[3*self.nb + 4*j:3*self.nb + 4*(j+1)]

                body.r = body.r + Δr
                body.p = body.p + Δp

            self.k += 1

            # logging.debug('t: {:.3f}, k: {:>2d}, norm: {:6.6e}'.format(t, self.k, np.linalg.norm(Δq)))

            if np.linalg.norm(Δq) < self.tol:
                break

            if self.k >= self.max_iters:
                raise RuntimeError('Newton-Raphson not converging at t: {:.3f}, k: {:>2d}'.format(t, self.max_iters))

        self.Φq = self.g_cons.get_phi_q(t)
        Φq_lu = lu_factor(self.Φq)

        dq = lu_solve(Φq_lu, self.g_cons.get_nu(t))
        for j, body in enumerate(self.bodies):
            body.dr = dq[3*j:3*(j+1), :]
            body.dp = dq[3*self.nb + 4*j:3*self.nb + 4*(j+1), :]

        ddq = lu_solve(Φq_lu, self.g_cons.get_gamma(t))
        for j, body in enumerate(self.bodies):
            body.ddr = ddq[3*j:3*(j+1), :]
            body.ddp = ddq[3*self.nb + 4*j:3*self.nb + 4*(j+1), :]


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
