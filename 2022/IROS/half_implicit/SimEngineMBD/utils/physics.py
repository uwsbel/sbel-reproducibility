import logging

import numpy as np
import sympy as sp
from enum import Enum, auto
from collections import namedtuple

I3 = np.identity(3)

# Helpful constants for global values
X_AXIS = np.array([[1], [0], [0]])  # x-out of page
Y_AXIS = np.array([[0], [1], [0]])  # y-right
Z_AXIS = np.array([[0], [0], [1]])  # z-up

# Setup BDF values
BDFVals = namedtuple('BDFVals', ['β', 'α'])
bdf1 = BDFVals(β=1, α=[-1, 1, 0])
bdf2 = BDFVals(β=2/3, α=[-1, 4/3, -1/3])


class Constraints(Enum):
    DP1 = auto()
    DP2 = auto()
    D = auto()
    CD = auto()
    EULER = auto()


class SolverType(Enum):
    KINEMATICS = auto()
    DYNAMICS = auto()


def euler_to_rot(ε):
    ϕ = ε[0]
    θ = ε[1]
    ψ = ε[2]

    A = np.zeros((3, 3))

    A[0, 0] = -np.sin(ψ)*np.sin(ϕ)*np.cos(θ) + np.cos(ψ)*np.cos(ϕ)
    A[0, 1] = -np.sin(ψ)*np.cos(ϕ) - np.sin(ϕ)*np.cos(θ)*np.cos(ψ)
    A[0, 2] = np.sin(θ)*np.sin(ϕ)

    A[1, 0] = np.sin(ψ)*np.cos(θ)*np.cos(ϕ) + np.sin(ϕ)*np.cos(ψ)
    A[1, 1] = -np.sin(ψ)*np.sin(ϕ) + np.cos(θ)*np.cos(ψ)*np.cos(ϕ)
    A[1, 2] = -np.sin(θ)*np.cos(ϕ)

    A[2, 0] = np.sin(θ)*np.sin(ψ)
    A[2, 1] = np.sin(θ)*np.cos(ψ)
    A[2, 2] = np.cos(θ)

    return A


def check_SO3(mat):
    """
    Checks if the passed matrix is in SO(3) (determinant is 1 and is 3x3)
    """
    tol = 1e-1

    if mat.shape != (3, 3):
        raise ValueError('Non 3x3 matrix passed to check_SO3')

    det = np.linalg.det(mat)
    det_diff = np.abs(det - 1)
    if det_diff > tol**2:
        logging.warning('|det(mat) -1| = ' + str(det_diff))
    if det_diff > tol:
        raise ValueError(
            'Matrix was non-orthogonal: |det(mat) -1| = {}'.format(det_diff))


def check_vector(v, n):
    """
    Checks if input v was a column vector of length n, noisily returns a column vector if it was a row vector.
    """
    if v.shape == (1, n):
        logging.warning('Input was a row vector, automatically transposed it')
        v = v.T
    elif v.shape != (n, 1):
        raise ValueError('Input vector v did not have dimension ' + str(n))

    return v


def check_unit_vector(k, n):
    """
    Checks if input k was a unit column vector of length n, noisily returns a column vector if it was a row vector.
    """
    k = check_vector(k, n)

    tol = 1e-6

    norm = np.linalg.norm(k)
    norm_diff = np.abs(norm - 1)
    if norm_diff > tol**2:
        logging.warning('|norm(k) -1| = ' + str(norm_diff))
    if norm_diff > tol:
        raise ValueError('k was not a unit vector')

    return k


def skew(v):
    """
    Computes the n by n cross product matrix ṽ for a given vector of dimensions n. Expects a column vector but will
    noisily transpose a row vector

    ṽ satisfies ṽa = v x a - where x is the cross-product operator
    """
    if __debug__:
        v = check_vector(v, 3)

    # NOTE: Using np.cross is significantly slower than this version, despite the aesthetic appeal
    return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])


def A(p):
    """
    Computes a rotation matrix (A) from a given orientation vector (unit quaternion) p. Expects a column vector but will
    noisily transpose a row vector
    """
    if __debug__:
        p = check_vector(p, 4)

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = skew(e)

    return (e0**2 - e.T @ e) * I3 + 2*(e @ e.T + e0 * ẽ)


def B(p, a):
    """
    Computes the B matrix from a given orientation vector (unit quaternion) and position vector. Expects column vectors
    but will noisily transpose row vectors.
    """
    if __debug__:
        p = check_vector(p, 4)
        a = check_vector(a, 3)

    e = p[1:, ...]
    e0 = np.diag(3*[p[0, 0]])
    
    c = e0 + skew(e)

    return 2 * np.concatenate((c @ a, e @ a.T - c @ skew(a)), axis=1)


def dG(dp):
    """
    Computes the Ġ matrix Ġ(p) = d/dt[-e, -ẽ + e_0 I]
    """

    e = dp[1:, ...]
    e0 = dp[0, 0]
    ẽ = skew(e)

    return np.concatenate((-e, -ẽ + e0 * I3), axis=1)


def G(p):
    """
    Computes the G matrix G(p) = [-e, -ẽ + e_0 I]
    """

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = skew(e)

    return np.concatenate((-e, -ẽ + e0 * I3), axis=1)


def E(p):
    """
    Computes the E matrix E(p) = [-e, ẽ + e_0 I]
    """

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = skew(e)

    return np.concatenate((-e, ẽ + e0 * I3), axis=1)


def cross_vec(R):
    """
    Moves from a skew-symmetric matrix to the corresponding vector. cross_vec(skew(A)) = A
    """
    return np.array([[R[2, 1]], [R[0, 2]], [R[1, 0]]])


def R(u, Chi):
    """
    Get the rotation matrix associated with a rotation Chi around unit axis u. Rodrigues' formula
    """
    if __debug__:
        u = check_unit_vector(u, 3)

    return np.cos(Chi)*I3 + (1 - np.cos(Chi)) * (u @ u.T) + np.sin(Chi)*skew(u)


def exp(mat):
    """
    un-does the work of skew to apply the Rodrigues formula to a matrix
    """

    tol = 1

    v = cross_vec(mat)
    v_norm = np.linalg.norm(v)

    # If length 0 then we don't rotate at all
    if v_norm == 0:
        return I3

    return R(v/v_norm, v_norm)


def rodrigues_rot(v, k, θ):
    """
    Uses Rodrigues' rotation formula to rotate a vector v by θ about the unit vector axis k
    """
    if __debug__:
        v = check_vector(v, 3)

    return v @ R(k, θ)


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


def to_scalar_first(q):
    """
    Swaps from scalar last to scalar first storage of a quaternion
    """
    return np.array([[q[3, 0], q[0, 0], q[1, 0], q[2, 0]]]).T


def rot_axis(v, θ):
    """
    Gets the quaternion representing a rotation of θ radians about the v axis
    """
    if __debug__:
        v = check_vector(v, 3)

    e0 = np.array([[np.cos(θ/2)]])
    e = v * np.sin(θ/2)

    return Quaternion(np.concatenate((e0, e), axis=0))


def block_mat(lst):
    """
    Takes a list (of len k) of (m x n) matrices and assembles a (km x kn) block matrix with the matrices on the diagonal
    """

    m, n = np.shape(lst[0])
    k = len(lst)

    ret = np.zeros((m*k, n*k))
    for i, mat in enumerate(lst):
        ret[m*i:m*(i+1), n*i:n*(i+1)] = mat

    return ret


def generate_sympy_constraint(f_sym, var):
    """
    Takes in a sympy expression for a constraint along with a variable and differentiates three times to return lambda 
    function derivatives of the constraint 
    """
    df_sym = sp.diff(f_sym, var)
    ddf_sym = sp.diff(df_sym, var)

    f = sp.lambdify(var, f_sym)
    df = sp.lambdify(var, df_sym)
    ddf = sp.lambdify(var, ddf_sym)

    return (f, df, ddf)

def create_col_slice(i_id, j_id, dim):
    """
    Creates a NumPy multi-slice object representing the columns in the global Φ_r, Π_* arrays at which a particular
    gcon's data should be placed. Combined with the gcon's ID (which only ConGroup knows), this tells us where to put
    data in the global arrays
    """
    if i_id is None:
        return dim*j_id + np.arange(0, dim)

    if j_id is None:
        return dim*i_id + np.arange(0, dim)
    
    return np.concatenate((dim*i_id + np.arange(0, dim), dim*j_id + np.arange(0, dim)))
