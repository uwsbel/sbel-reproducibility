import numpy as np
from scipy.spatial.transform import Rotation as Rot
from physics import Constraints, check_vector, skew, I3, A, B, G, dG, E, to_scalar_first, generate_sympy_constraint
from collections import namedtuple
import json as js
from enum import Enum

AI = 'a_bar_i'
AJ = 'a_bar_j'
SI = 's_bar_p_i'
SJ = 's_bar_q_j'
C = 'c'
F = 'f'
DF = 'f_dot'
DDF = 'f_ddot'
JS_r = "r"
JS_rdot = "r_dot"
JS_p = "p"
JS_pdot = "p_dot"
JS_A = "A"
JS_omega = "omega"


def distance_fn(body_i, body_j, si, sj):
    """
    d_ij in Haug and Negrut's notation
    Distance between point s_i on body i and point s_j on body j. Used by DP2, D and CD constraints
    """
    return body_j.r + body_j.A @ sj - body_i.r - body_i.A @ si


class Quaternion:
    """
    Weird, couldn't find a NumPy/SciPy package for this...
    """

    def __init__(self, v):
        v = check_vector(v, 4)
        self.r = v[0, 0]
        self.i = v[1, 0]
        self.j = v[2, 0]
        self.k = v[3, 0]

        self.arr = v

    def __mul__(self, other):
        """
        Maybe the matrix definition of multiplication is cleaner? idk
        """
        r = self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k
        i = self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j
        j = self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i
        k = self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r

        return Quaternion(np.array([[r], [i], [j], [k]]))


def rot_axis(v, θ):
    """
    Gets the quaternion representing a rotation of θ radians about the v axis
    """
    v = check_vector(v, 3)

    e0 = np.array([[np.cos(θ/2)]])
    e = v * np.sin(θ/2)

    return Quaternion(np.concatenate((e0, e), axis=0))


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


def read_model_file(file_name):
    with open(file_name) as model_file:
        model_data = js.load(model_file)

        model_bodies = model_data['bodies']
        model_constraints = model_data['constraints']

    return (model_bodies, model_constraints)


class Body:
    def __init__(self, r, dr, p, dp, is_ground):
        self.is_ground = is_ground
        self.id = None  # Assigned later for non-ground bodies

        self.r = r
        self.dr = dr
        self.ddr = np.zeros((3, 1))

        self.p = p
        self.dp = dp
        self.ddp = np.zeros((4, 1))

        # Give ourselves some properties to use later
        self.m = 0
        self.V = 0
        self.J = np.zeros((3, 3))
        self.F = np.zeros((3, 1))

        self.r_prev = self.r
        self.p_prev = self.p
        self.dr_prev = self.dr
        self.dp_prev = self.dp

        self.C_r = 0
        self.C_p = 0
        self.C_dr = 0
        self.C_dp = 0

    @classmethod
    def init_from_dict(cls, dict, is_ground=False):
        is_ground = is_ground or dict['name'] == 'ground'

        if is_ground:
            r = np.zeros((3, 1))
            dr = np.zeros((3, 1))

            p = np.array([[1, 0, 0, 0]]).T
            dp = np.zeros((4, 1))
        else:
            r = np.array([dict[JS_r]]).T
            dr = np.array([dict[JS_rdot]]).T

            if JS_p in dict:
                p = np.array([dict[JS_p]]).T
            else:
                quat = np.array(
                    [Rot.from_matrix(np.array(dict[JS_A])).as_quat()]).T
                p = to_scalar_first(quat)

            if JS_pdot in dict:
                dp = np.array([dict[JS_pdot]]).T
            else:
                ω = np.array([dict[JS_omega]]).T
                dp = 1/2 * E(p).T @ ω

        return cls(r, dr, p, dp, is_ground)

    def cache_rp_values(self):
        self.r_prev = self.r
        self.p_prev = self.p
        self.dr_prev = self.dr
        self.dp_prev = self.dp

    def update_bdf_coeffs(self, bdf, h):
        self.C_dr = bdf.α[1]*self.dr + bdf.α[2]*self.dr_prev
        self.C_r = bdf.α[1]*self.r + bdf.α[2]*self.r_prev + bdf.β*h*self.C_dr

        self.C_dp = bdf.α[1]*self.dp + bdf.α[2]*self.dp_prev
        self.C_p = bdf.α[1]*self.p + bdf.α[2]*self.p_prev + bdf.β*h*self.C_dp

    def get_j(self):
        return 4*self.G.T @ self.J @ self.G

    def get_tau(self):
        return 8*self.dG.T @ self.J @ self.G @ self.dp

    @property
    def dG(self):
        return dG(self.dp)

    @property
    def G(self):
        return self._G

    @property
    def E(self):
        return E(self.p)

    @property
    def dω(self):
        return 2*self.G @ self.ddp

    @property
    def ω(self):
        return 2*self.G @ self.dp

    @property
    def A(self):
        return self._A

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._A = A(value)
        self._G = G(value)
        self._p = value


class CD:
    cons_type = Constraints.CD

    def __init__(self, body_i, body_j, si, sj, c, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.si = si
        self.sj = sj

        self.c = c

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        si = np.array([dict[SI]]).T
        sj = np.array([dict[SJ]]).T
        c = np.array([dict[C]]).T

        return cls(body_i, body_j, si, sj, c, dict[F], dict[DF], dict[DDF])

    def d_ij(self):
        """
        Compact function call for distance between two points
        """
        return distance_fn(self.body_i, self.body_j, self.si, self.sj)

    def get_phi(self, t):
        return self.c.T @ self.d_ij() - self.f(t)

    def get_gamma(self, t):
        term1 = B(self.body_i.dp, self.si) @ self.body_i.dp
        term2 = B(self.body_j.dp, self.sj) @ self.body_j.dp

        return self.c.T @ (term1 - term2) + self.ddf(t)

    def get_nu(self, t):
        return [self.df(t)]

    def get_phi_r(self, t):
        if self.body_i.is_ground:
            return [(self.body_j.id, self.c.T)]
        if self.body_j.is_ground:
            return [(self.body_i.id, -self.c.T)]
        return [(self.body_i.id, -self.c.T), (self.body_j.id, self.c.T)]

    def get_phi_p(self, t):
        Bpj = (self.body_j.id, self.c.T @ B(self.body_j.p, self.sj))
        Bpi = (self.body_i.id, -self.c.T @ B(self.body_i.p, self.si))

        if self.body_i.is_ground:
            return [Bpj]
        if self.body_j.is_ground:
            return [Bpi]

        return [Bpi, Bpj]


class DP1:
    cons_type = Constraints.DP1

    def __init__(self, body_i, body_j, ai, aj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.ai = ai
        self.aj = aj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        ai = np.array([dict[AI]]).T
        aj = np.array([dict[AJ]]).T

        return cls(body_i, body_j, ai, aj, dict[F], dict[DF], dict[DDF])

    def get_phi(self, t):
        Ai = self.body_i.A
        Aj = self.body_j.A

        return self.ai.T @ Ai.T @ Aj @ self.aj - self.f(t)

    def get_gamma(self, t):
        B_dpi = B(self.body_i.dp, self.ai)
        B_dpj = B(self.body_j.dp, self.aj)

        aiT = self.ai.T @ self.body_i.A.T
        ajT = self.aj.T @ self.body_j.A.T

        ai_dot = B(self.body_i.p, self.ai) @ self.body_i.dp
        aj_dot = B(self.body_j.p, self.aj) @ self.body_j.dp

        γ_rp = -aiT @ B_dpj @ self.body_j.dp - \
            ajT @ B_dpi @ self.body_i.dp - 2*ai_dot.T @ aj_dot + self.ddf(t)

        return γ_rp

    def get_nu(self, t):
        return [self.df(t)]

    def get_phi_r(self, t):
        return []

    def get_phi_p(self, t):
        Ai = self.body_i.A
        Aj = self.body_j.A

        term_i = (self.body_i.id, (B(self.body_i.p, self.ai).T @ Aj @ self.aj).T)
        term_j = (self.body_j.id, self.ai.T @ Ai.T @ B(self.body_j.p, self.aj))

        if self.body_i.is_ground:
            return [term_j]
        if self.body_j.is_ground:
            return [term_i]

        return [term_i, term_j]

    def set_constraint_fn(self, f_sym, var):
        f, df, ddf = generate_sympy_constraint(f_sym, var)

        self.f = f
        self.df = df
        self.ddf = ddf


class DP2:
    cons_type = Constraints.DP2

    def __init__(self, body_i, body_j, ai, si, sj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.ai = ai

        self.si = si
        self.sj = sj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        ai = np.array([dict[AI]]).T

        si = np.array([dict[SI]]).T
        sj = np.array([dict[SJ]]).T

        return cls(body_i, body_j, ai, si, sj, dict[F], dict[DF], dict[DDF])

    def d_ij(self):
        """
        Compact function call for distance between two points
        """
        return distance_fn(self.body_i, self.body_j, self.si, self.sj)

    def get_phi(self, t):
        Ai = self.body_i.A

        return self.ai.T @ Ai.T @ self.d_ij() - self.f(t)

    def get_gamma(self, t):
        B_dpi = B(self.body_i.dp, self.si)
        B_dpj = B(self.body_j.dp, self.sj)
        B_dpi_ai = B(self.body_i.dp, self.ai)

        Ai = self.body_i.A

        aiT = self.ai.T @ Ai.T

        ai_dot = B(self.body_i.p, self.ai) @ self.body_i.dp

        dij = self.d_ij()
        dij_dot = self.body_j.dr + B(self.body_j.p, self.sj) @ self.body_j.dp - \
            self.body_i.dr - B(self.body_i.p, self.si) @ self.body_i.dp

        γ_rp = -aiT @ B_dpj @ self.body_j.dp + \
            aiT @ B_dpi @ self.body_i.dp - \
            dij.T @ B_dpi_ai @ self.body_i.dp - \
            2 * ai_dot.T @ dij_dot + self.ddf(t)

        return γ_rp

    def get_nu(self, t):
        return [self.df(t)]

    def get_phi_r(self, t):
        Φr = []

        ai = self.ai.T @ self.body_i.A.T
        if not self.body_i.is_ground:
            Φr.append((self.body_i.id, -ai))
        if not self.body_j.is_ground:
            Φr.append((self.body_j.id, ai))

        return Φr

    def get_phi_p(self, t):
        Φp = []

        Ai = self.body_i.A

        term_i = self.d_ij().T @ B(self.body_i.p, self.ai) - \
            self.ai.T @ Ai.T @ B(self.body_i.p, self.si)
        term_j = self.ai.T @ Ai.T @ B(self.body_j.p, self.sj)

        if not self.body_i.is_ground:
            Φp.append((self.body_i.id, term_i))
        if not self.body_j.is_ground:
            Φp.append((self.body_j.id, term_j))

        return Φp


class D:
    cons_type = Constraints.D

    def __init__(self, body_i, body_j, si, sj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.si = si
        self.sj = sj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        si = np.array([dict[SI]]).T
        sj = np.array([dict[SJ]]).T

        return cls(body_i, body_j, si, sj, dict[F], dict[DF], dict[DDF])

    def d_ij(self):
        """
        Compact function call for distance between two points
        """
        return distance_fn(self.body_i, self.body_j, self.si, self.sj)

    def get_phi(self, t):
        dij = self.d_ij()

        return dij.T @ dij - self.f(t)

    def get_gamma(self, t):
        B_dpi = B(self.body_i.dp, self.si)
        B_dpj = B(self.body_j.dp, self.sj)

        dij = self.d_ij()
        dij_dot = self.body_j.dr + B(self.body_j.p, self.sj) @ self.body_j.dp - \
            self.body_i.dr - B(self.body_i.p, self.si) @ self.body_i.dp

        γ = -2 * dij.T @ B_dpj @ self.body_j.dp + 2 * \
            dij.T @ B_dpi @ self.body_i.dp - 2 * \
            dij_dot.T @ dij_dot + self.ddf(t)

        return γ

    def get_nu(self, t):
        return [self.df(t)]

    def get_phi_r(self, t):
        Φr = []

        dij = self.d_ij()

        if not self.body_i.is_ground:
            Φr.append((self.body_i.id, -2*dij.T))
        if not self.body_j.is_ground:
            Φr.append((self.body_j.id, 2*dij.T))

        return Φr

    def get_phi_p(self, t):
        Φp = []

        dij = self.d_ij()

        term_i = -2 * dij.T @ B(self.body_i.p, self.si)
        term_j = 2 * dij.T @ B(self.body_j.p, self.sj)

        if not self.body_i.is_ground:
            Φp.append((self.body_i.id, term_i))
        if not self.body_j.is_ground:
            Φp.append((self.body_j.id, term_j))

        return Φp

    def set_constraint_fn(self, f_sym, var):
        f, df, ddf = generate_sympy_constraint(f_sym, var)

        self.f = f
        self.df = df
        self.ddf = ddf


class EulerCon:
    cons_type = Constraints.EULER

    def __init__(self, body):
        self.body = body

    def get_phi(self, t):
        return 0.5*self.body.p.T @ self.body.p - 0.5

    def get_gamma(self, t):
        return -(self.body.dp.T @ self.body.dp)

    def get_nu(self, t):
        return [0]

    def get_phi_r(self, t):
        return [(self.body.id, np.zeros((1, 3)))]

    def get_phi_p(self, t):
        return [(self.body.id, self.body.p.T)]

    def get_phi_q(self, t):
        return np.concatenate((self.get_phi_r(t), self.get_phi_p(t)), axis=1)


class ConGroup:
    def __init__(self, con_list, nb):
        self.cons = con_list
        self.nc = len(self.cons)
        self.nb = nb

        self.init_storage()

    def init_storage(self):
        self.Φ = np.zeros((self.nc, 1))
        self.Φr = np.zeros((self.nc, 3*self.nb))
        self.Φp = np.zeros((self.nc, 4*self.nb))
        self.γ = np.zeros((self.nc, 1))
        self.nu = np.zeros((self.nc, 1))

    def add_constraint(self, con):
        self.cons.append(con)
        self.nc = len(self.cons)

        self.init_storage()

    def get_phi(self, t):
        for i, con in enumerate(self.cons):
            self.Φ[i, 0] = con.get_phi(t)
        return self.Φ

    def get_gamma(self, t):
        for i, con in enumerate(self.cons):
            self.γ[i] = con.get_gamma(t)
        return self.γ

    def get_nu(self, t):
        for i, con in enumerate(self.cons):
            self.nu[i] = con.get_nu(t)
        return self.nu

    def get_phi_r(self, t):
        for i, con in enumerate(self.cons):
            for b_id, phiR in con.get_phi_r(t):
                self.Φr[i, 3*b_id:3*(b_id + 1)] = phiR
        return self.Φr

    def get_phi_p(self, t):
        for i, con in enumerate(self.cons):
            for b_id, phiP in con.get_phi_p(t):
                self.Φp[i, 4*b_id:4*(b_id + 1)] = phiP
        return self.Φp

    def get_phi_q(self, t):
        self.init_storage()
        return np.concatenate((self.get_phi_r(t), self.get_phi_p(t)), axis=1)
