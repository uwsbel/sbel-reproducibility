import numpy as np
import warnings as warn
from physics import Constraints, skew, I3, check_SO3, generate_sympy_constraint, euler_to_rot
from scipy.spatial.transform import Rotation as Rot

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
JS_A = "A"
JS_omega = "omega"

ZXZ = 'ZXZ'             # Euler angle sequence that we use

THETA_CRITERIA = 0.1    # When to consider an euler angle dangerously close to singular

# Causes us to error on Euler angle problems
warn.filterwarnings(action="error", category=UserWarning)

def from_eps(ε):
    """
    Deconstruct ε into our three angles
    """
    return ε[0, 0], ε[1, 0], ε[2, 0]


def distance_fn(body_i, body_j, si, sj):
    """
    d_ij in Haug and Negrut's notation

    Distance between point s_i on body i and point s_j on body j. Used by DP2, D and CD constraints
    """
    return body_j.r + body_j.A @ sj - body_i.r - body_i.A @ si


class Body:
    def __init__(self, r, dr, ε, dε, is_ground):
        self.is_ground = is_ground
        self.id = None  # Assigned later for non-ground bodies

        self.r = r
        self.dr = dr
        self.ε = ε
        self.dε = dε

        self.m = 0
        self.V = 0
        self.J = np.zeros((3, 3))
        self.F = np.zeros((3, 1))

        self.r_prev = self.r
        self.ε_prev = self.ε
        self.dr_prev = self.dr
        self.dε_prev = self.dε

        self.C_r = 0
        self.C_ε = 0
        self.C_dr = 0
        self.C_dε = 0

        self._dA = np.zeros((3, 3))
        self._ddA = np.zeros((3, 3))

    @classmethod
    def init_from_dict(cls, dict, is_ground=False):
        is_ground = is_ground or dict['name'] == 'ground'

        if is_ground:
            r = np.zeros((3, 1))
            dr = np.zeros((3, 1))
            ε = np.zeros((3, 1))
            dε = np.zeros((3, 1))
        else:
            r = np.array([dict[JS_r]]).T
            dr = np.array([dict[JS_rdot]]).T

            A = Rot.from_matrix(np.array(dict[JS_A]))
            ε = A.as_euler(ZXZ, degrees=False)
            ε = np.asmatrix(ε).T

            assert np.linalg.norm(A.as_matrix() - euler_to_rot(ε)) < 1e-12

            dε = np.zeros((3, 1))

        return cls(r, dr, ε, dε, is_ground)

    def cache_repsilon_values(self):
        self.r_prev = self.r
        self.ε_prev = self.ε
        self.dr_prev = self.dr
        self.dε_prev = self.dε

    def update_bdf_coeffs(self, bdf, h):
        self.C_dr = bdf.α[1]*self.dr + bdf.α[2]*self.dr_prev
        self.C_r = bdf.α[1]*self.r + bdf.α[2]*self.r_prev + bdf.β*h*self.C_dr

        self.C_dε = bdf.α[1]*self.dε + bdf.α[2]*self.dε_prev
        self.C_ε = bdf.α[1]*self.ε + bdf.α[2]*self.ε_prev + bdf.β*h*self.C_dε

    def get_tau(self):
        B_bar = self.A.T @ self.B
        term_1 = self.B.T @ skew(B_bar @ self.dε) @ self.J @ self.B @ self.dε
        term_2 = B_bar.T @ self.J @ self.dB @ self.dε
        
        return term_1 + term_2

    def get_J_term(self):
        B_bar = self.A.T @ self.B
        return B_bar.T @ self.J @ B_bar

    def get_partials(self):
        return (self.A_ϕ, self.A_θ, self.A_ψ)

    def get_As(self):
        A1 = np.array([[self._cϕ, -self._sϕ, 0],
                       [self._sϕ, self._cϕ, 0], [0, 0, 1]])
        A2 = np.array([[1, 0, 0], [0, self._cθ, -self._sθ],
                       [0, self._sθ, self._cθ]])
        A3 = np.array([[self._cψ, -self._sψ, 0],
                       [self._sψ, self._cψ, 0], [0, 0, 1]])

        return A1, A2, A3

    def get_dAs(self):
        dA1 = np.array([[-self._sϕ, -self._cϕ, 0],
                        [self._cϕ, -self._sϕ, 0], [0, 0, 0]])
        dA2 = np.array([[0, 0, 0], [0, -self._sθ, -self._cθ],
                        [0, self._cθ, -self._sθ]])
        dA3 = np.array([[-self._sψ, -self._cψ, 0],
                        [self._cψ, -self._sψ, 0], [0, 0, 0]])

        return dA1, dA2, dA3

    def get_ddAs(self):
        ddA1 = np.array([[-self._cϕ, self._sϕ, 0],
                         [-self._sϕ, -self._cϕ, 0], [0, 0, 0]])
        ddA2 = np.array([[0, 0, 0], [0, -self._cθ, self._sθ],
                         [0, -self._sθ, -self._cθ]])
        ddA3 = np.array([[-self._cψ, self._sψ, 0],
                         [-self._sψ, -self._cψ, 0], [0, 0, 0]])

        return ddA1, ddA2, ddA3

    def cache_sin_cos(self, ϕ, θ, ψ):
        """
        From a given set of Euler angles, computes and caches their sine and cosine in local properties
        """
        self._cϕ = np.cos(ϕ)
        self._sϕ = np.sin(ϕ)

        self._cθ = np.cos(θ)
        self._sθ = np.sin(θ)

        self._cψ = np.cos(ψ)
        self._sψ = np.sin(ψ)

    def cache_A_partials(self):
        """
        Based on the cached sine/cosine values, computes and caches the partial derivative of the rotation matrix A with
        respect to the three euler angles
        """

        self._A_ϕ = np.array([[-self._sψ*self._cθ*self._cϕ - self._sϕ*self._cψ, self._sψ*self._sϕ - self._cθ*self._cψ*self._cϕ, self._sθ*self._cϕ], [-self._sψ*self._sϕ*self._cθ + self._cψ*self._cϕ, -self._sψ*self._cϕ - self._sϕ*self._cθ*self._cψ, self._sθ*self._sϕ], [0, 0, 0]])
        self._A_θ = np.array([[self._sθ*self._sψ*self._sϕ, self._sθ*self._sϕ*self._cψ, self._sϕ*self._cθ], [-self._sθ*self._sψ * self._cϕ, -self._sθ*self._cψ*self._cϕ, -self._cθ*self._cϕ], [self._sψ*self._cθ, self._cθ*self._cψ, -self._sθ]])
        self._A_ψ = np.array([[-self._sψ*self._cϕ - self._sϕ*self._cθ*self._cψ, self._sψ*self._sϕ*self._cθ - self._cψ*self._cϕ, 0], [-self._sψ * self._sϕ + self._cθ*self._cψ*self._cϕ, -self._sψ*self._cθ*self._cϕ - self._sϕ*self._cψ, 0], [self._sθ*self._cψ, -self._sθ*self._sψ, 0]])

    def cache_time_derivs(self):
        """
        Computes time derivative terms needed by g-cons in the computation of γ and caches these so that g-cons don't
        have to re-compute them
        """
        ϕ_dot, θ_dot, ψ_dot = from_eps(self.dε)

        A1, A2, A3 = self.get_As()
        dA1, dA2, dA3 = self.get_dAs()
        ddA1, ddA2, ddA3 = self.get_ddAs()

        # Terms related to Ȧ
        dAϕ = ϕ_dot * dA1 @ A2 @ A3
        dAθ = θ_dot * A1 @ dA2 @ A3
        dAψ = ψ_dot * A1 @ A2 @ dA3
        
        # Ȧ
        self._dA = dAϕ + dAθ + dAψ

        # Terms related to Ä
        ϕθ = 2*ϕ_dot*θ_dot * dA1 @ dA2 @ A3
        θψ = 2*θ_dot*ψ_dot * A1 @ dA2 @ dA3
        ϕψ = 2*ϕ_dot*ψ_dot * dA1 @ A2 @ dA3
        ddAϕ = ϕ_dot**2 * ddA1 @ A2 @ A3
        ddAθ = θ_dot**2 * A1 @ ddA2 @ A3
        ddAψ = ψ_dot**2 * A1 @ A2 @ ddA3
        
        # What we denote as Ä_γ - Ä with all second derivative terms removed
        self._ddA = ddAϕ + ddAθ + ddAψ + ϕθ + θψ + ϕψ

    def compute_new_frame(self):
        """
        Rotates a body's reference frame by the hard-coded 'flip_mat', updating stored ε values along with it
        """

        flip_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        new_A = flip_mat @ self.A
        rot = Rot.from_matrix(new_A)
        value = np.array([rot.as_euler(ZXZ, degrees=False)]).T

        return value, flip_mat

    @property
    def ε(self):
        return self._ε

    @ε.setter
    def ε(self, value):
        """
        Whenever we set ε, cache A for future use
        """

        rot = Rot.from_euler(ZXZ, value.T, degrees=False)
        # If we keep value as a 1x3 array it will give A an extra dimension, so we squeeze away the extra dim
        self._A = np.squeeze(rot.as_matrix())

        ϕ, θ, ψ = from_eps(value)

        self.near_singular = np.abs(np.fmod(θ, np.pi)) < THETA_CRITERIA and (not self.is_ground)

        self.cache_sin_cos(ϕ, θ, ψ)
        self.cache_A_partials()

        self._ε = value

    @property
    def A(self):
        return self._A

    @property
    def A_ϕ(self):
        return self._A_ϕ

    @property
    def A_θ(self):
        return self._A_θ

    @property
    def A_ψ(self):
        return self._A_ψ

    @property
    def dA(self):
        return self._dA

    @property
    def ddA(self):
        return self._ddA

    @property
    def B(self):
        B = np.array([[0, self._cϕ, self._sθ*self._sϕ],
                      [0, self._sϕ, -self._sθ*self._cϕ], [1, 0, self._cθ]])
        return B

    @property
    def dB(self):
        dϕ, dθ, dψ = from_eps(self.dε)

        dB = np.array([[0, -dϕ*self._sϕ, dθ*self._cθ*self._sϕ + dϕ*self._sθ*self._cϕ], [0,
                                                                                        dϕ*self._sϕ, -dθ*self._cθ*self._cϕ + dϕ*self._sθ*self._sϕ], [0, 0, -dθ*self._sθ]])
        dB_bar = np.array([[dψ*self._cψ*self._sθ + dθ*self._sψ*self._cθ, -dψ*self._sψ, 0],
                           [-dψ*self._sψ*self._sθ+dθ*self._cψ*self._cθ, -dψ*self._cψ, 0], [-dθ*self._sθ, 0, 0]])
        return dB_bar

    @property
    def ω(self):
        return self.A.T @ self.B @ self.dε


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
        dAi = self.body_i.dA
        dAj = self.body_j.dA

        ddAi = self.body_i.ddA
        ddAj = self.body_j.ddA

        i_term = self.ai.T @ ddAi.T @ self.body_j.A @ self.aj
        j_term = self.ai.T @ self.body_i.A.T @ ddAj @ self.aj
        x_term = 2*self.ai.T @ dAi.T @ dAj @ self.aj

        γ_rε = -i_term -x_term -j_term + self.ddf(t)

        return γ_rε

    def get_nu(self, t):
        return self.df(t)

    def get_phi_r(self, t):
        return []

    def get_phi_eps(self, t):
        ai = self.body_i.A @ self.ai
        aj = self.body_j.A @ self.aj

        Ai_ϕ, Ai_θ, Ai_ψ = self.body_i.get_partials()
        Aj_ϕ, Aj_θ, Aj_ψ = self.body_j.get_partials()

        Φε = []
        if not self.body_i.is_ground:
            i_term = np.block(
                [[aj.T @ Ai_ϕ @ self.ai, aj.T @ Ai_θ @ self.ai, aj.T @ Ai_ψ @ self.ai]])
            Φε.append((self.body_i.id, i_term))
        if not self.body_j.is_ground:
            j_term = np.block(
                [[ai.T @ Aj_ϕ @ self.aj, ai.T @ Aj_θ @ self.aj, ai.T @ Aj_ψ @ self.aj]])
            Φε.append((self.body_j.id, j_term))
        return Φε

    def set_constraint_fn(self, f_sym, var):
        f, df, ddf = generate_sympy_constraint(f_sym, var)

        self.f = f
        self.df = df
        self.ddf = ddf

    def flip_gcon_body(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.id == body_id:
            self.ai = flip_mat @ self.ai

        if self.body_j.id == body_id:
            self.aj = flip_mat @ self.aj


class CD:
    cons_type = Constraints.CD

    def __init__(self, body_i, body_j, si, sj, c, f, df, ddf, name=None):
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

        self.name = name

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        si = np.array([dict[SI]]).T
        sj = np.array([dict[SJ]]).T
        c = np.array([dict[C]]).T

        return cls(body_i, body_j, si, sj, c, dict[F], dict[DF], dict[DDF], dict["name"])

    def d_ij(self):
        """
        Compact function call for distance between two points
        """
        return distance_fn(self.body_i, self.body_j, self.si, self.sj)

    def get_phi(self, t):
        return self.c.T @ self.d_ij() - self.f(t)

    def get_gamma(self, t):
        ddAi = self.body_i.ddA
        ddAj = self.body_j.ddA

        return -self.c.T @ (ddAj @ self.sj - ddAi @ self.si) + self.ddf(t)

    def get_nu(self, t):
        return self.df(t)

    def get_phi_r(self, t):
        Φr = []
        if not self.body_i.is_ground:
            Φr.append((self.body_i.id, -self.c.T))
        if not self.body_j.is_ground:
            Φr.append((self.body_j.id, self.c.T))
        return Φr

    def get_phi_eps(self, t):
        Φε = []

        Ai_ϕ, Ai_θ, Ai_ψ = self.body_i.get_partials()
        Aj_ϕ, Aj_θ, Aj_ψ = self.body_j.get_partials()

        if not self.body_i.is_ground:
            i_term = self.c.T @ np.block(
                [[-Ai_ϕ @ self.si, -Ai_θ @ self.si, -Ai_ψ @ self.si]])
            Φε.append((self.body_i.id, i_term))
        if not self.body_j.is_ground:
            j_term = self.c.T @ np.block(
                [[Aj_ϕ @ self.sj, Aj_θ @ self.sj, Aj_ψ @ self.sj]])
            Φε.append((self.body_j.id, j_term))
        return Φε

    def flip_gcon_body(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.id == body_id:
            self.si = flip_mat @ self.si
            
        if self.body_j.id == body_id:
            self.sj = flip_mat @ self.sj


class DP2:
    cons_type = Constraints.DP2

    def __init__(self, body_i, body_j, ai, si, sj, f, df, ddf, name=None):
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

        self.name = name

    @classmethod
    def init_from_dict(cls, dict, body_i, body_j):
        ai = np.array([dict[AI]]).T

        si = np.array([dict[SI]]).T
        sj = np.array([dict[SJ]]).T

        return cls(body_i, body_j, ai, si, sj, dict[F], dict[DF], dict[DDF], dict["name"])

    def d_ij(self):
        """
        Compact function call for distance between two points
        """
        return distance_fn(self.body_i, self.body_j, self.si, self.sj)

    def get_phi(self, t):
        return self.ai.T @ self.body_i.A.T @ self.d_ij() - self.f(t)

    def get_gamma(self, t):
        dAi = self.body_i.dA
        dAj = self.body_j.dA

        ddAi = self.body_i.ddA
        ddAj = self.body_j.ddA

        # Shorter expressions
        ai = self.ai
        dri = self.body_i.dr
        drj = self.body_j.dr

        # Three terms, we took the second derivative of the product A d_ij, each term is one of the terms in the expansion
        term_1 = -ai.T @ ddAi @ self.d_ij()
        term_2 = -ai.T @ self.body_i.A.T @ (ddAj @ self.sj - ddAi @ self.si)
        term_3 = -2*ai.T @ dAi @ (drj - dri + dAj @ self.sj - dAi @ self.si)

        γ_rε = term_1 + term_2 + term_3 + self.ddf(t)

        return γ_rε

    def get_nu(self, t):
        return self.df(t)

    def get_phi_r(self, t):
        Φr = []

        ai = self.ai.T @ self.body_i.A.T
        if not self.body_i.is_ground:
            Φr.append((self.body_i.id, -ai))
        if not self.body_j.is_ground:
            Φr.append((self.body_j.id, ai))

        return Φr

    def get_phi_eps(self, t):
        Φε = []

        Ai_ϕ, Ai_θ, Ai_ψ = self.body_i.get_partials()
        Aj_ϕ, Aj_θ, Aj_ψ = self.body_j.get_partials()

        aiT = self.ai.T @ self.body_i.A.T
        dijT = self.d_ij().T

        if not self.body_i.is_ground:
            i_term = np.block([[dijT @ Ai_ϕ @ self.ai, dijT @ Ai_θ @ self.ai, dijT @ Ai_ψ @ self.ai]]) - np.block([[aiT @ Ai_ϕ @ self.si, aiT @ Ai_θ @ self.si, aiT @ Ai_ψ @ self.si]])
            Φε.append((self.body_i.id, i_term))
        if not self.body_j.is_ground:
            j_term = np.block([[aiT @ Aj_ϕ @ self.sj, aiT @ Aj_θ @ self.sj, aiT @ Aj_ψ @ self.sj]])
            Φε.append((self.body_j.id, j_term))

        return Φε

    def flip_gcon_body(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.id == body_id:
            self.ai = flip_mat @ self.ai
            self.si = flip_mat @ self.si
            
        if self.body_j.id == body_id:
            self.sj = flip_mat @ self.sj


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
        dAi = self.body_i.dA
        dAj = self.body_j.dA

        ddAi = self.body_i.ddA
        ddAj = self.body_j.ddA

        # First time derivative of d_{ij}
        d_dot = (self.body_j.dr - self.body_i.dr) + (dAj @ self.sj - dAi @ self.si)

        # Three terms, we took the second derivative of the product A d_ij, each term is one of the terms in the expansion
        term_1 = -2*d_dot.T @ d_dot
        term_2 = -2*self.d_ij().T @ (ddAj @ self.sj - ddAi @ self.si)

        γ_rε = term_1 + term_2 + self.ddf(t)

        return γ_rε

    def get_nu(self, t):
        return self.df(t)

    def get_phi_r(self, t):
        Φr = []

        dijT = self.d_ij().T

        if not self.body_i.is_ground:
            Φr.append((self.body_i.id, -2*dijT))
        if not self.body_j.is_ground:
            Φr.append((self.body_j.id, 2*dijT))

        return Φr

    def get_phi_eps(self, t):
        Φε = []

        Ai_ϕ, Ai_θ, Ai_ψ = self.body_i.get_partials()
        Aj_ϕ, Aj_θ, Aj_ψ = self.body_j.get_partials()

        dij = self.d_ij()
        dijT = dij.T

        if not self.body_i.is_ground:
            term_i = -np.block([[dijT @ Ai_ϕ @ self.si, dijT @ Ai_θ @ self.si, dijT @ Ai_ψ @ self.si]])- np.block([[self.si.T @ Ai_ϕ.T @ dij, self.si.T @ Ai_θ.T @ dij, self.si.T @ Ai_ψ.T @ dij]])
            Φε.append((self.body_i.id, term_i)) 
        if not self.body_j.is_ground:
            term_j = np.block([[dijT @ Aj_ϕ @ self.sj, dijT @ Aj_θ @ self.sj, dijT @ Aj_ψ @ self.sj]]) + np.block([[self.sj.T @ Aj_ϕ.T @ dij, self.sj.T @ Aj_θ.T @ dij, self.sj.T @ Aj_ψ.T @ dij]])
            Φε.append((self.body_j.id, term_j))

        return Φε

    def set_constraint_fn(self, f_sym, var):
        f, df, ddf = generate_sympy_constraint(f_sym, var)

        self.f = f
        self.df = df
        self.ddf = ddf

    def flip_gcon_body(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.id == body_id:
            self.si = flip_mat @ self.si
            
        if self.body_j.id == body_id:
            self.sj = flip_mat @ self.sj


class ConGroup:
    def __init__(self, con_list, nb):
        self.cons = con_list
        self.nc = len(self.cons)
        self.nb = nb

        self.init_storage()

    def init_storage(self):
        self.Φ = np.zeros((self.nc, 1))
        self.Φr = np.zeros((self.nc, 3*self.nb))
        self.Φε = np.zeros((self.nc, 3*self.nb))
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

    def get_phi_eps(self, t):
        for i, con in enumerate(self.cons):
            for b_id, phiEps in con.get_phi_eps(t):
                self.Φε[i, 3*b_id:3*(b_id + 1)] = phiEps
        return self.Φε

    def get_phi_q(self, t):
        return np.concatenate((self.get_phi_r(t), self.get_phi_eps(t)), axis=1)

    def flip_gcons(self, body_id, flip_mat):
        """
        Iterates over all g-cons in this group and flips any vectors local to (body.id == body_id) that those g-cons use
        by a rotation of flip_mat

        TODO what about the alternate g-con or any g-con with a constraint function?
        """
        for con in self.cons:
            con.flip_gcon_body(body_id, flip_mat.T)
