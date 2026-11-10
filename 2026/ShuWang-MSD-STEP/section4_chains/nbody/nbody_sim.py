"""N-link chain of the three-link benchmark, numeric instead of sympy.
Links: massless rod of length L, point mass m at the distal end, IMU at the midpoint, R_i = Rz(phi_i) Rx(theta_i)."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

L, MASS, GRAVITY, EPS, DT, DURATION = 2.0, 1.0, 9.81, 1e-4, 0.001, 30.0


@njit(cache=False)
def link_terms(th, ph, thd, phd):
    """z = R e_z; J = dz/d(theta, phi) (3 x 2); c = z'' at zero generalized acceleration."""
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    z = np.array([sp * st, -cp * st, ct])
    J = np.array([[sp * ct, cp * st], [-cp * ct, sp * st], [-st, 0.0]])
    c = np.array([-sp * st * thd * thd + 2 * cp * ct * thd * phd - sp * st * phd * phd,
                  cp * st * thd * thd + 2 * sp * ct * thd * phd + cp * st * phd * phd,
                  -ct * thd * thd])
    return z, J, c


@njit(cache=False)
def mass_forcing(t, y, N):
    """Mass matrix M and forcing F of the 2N generalized coordinates (theta_1, phi_1, ...); no regularization."""
    n = 2 * N; M = np.zeros((n, n)); F = np.zeros(n)
    Js = np.zeros((N, 3, 2)); cs = np.zeros((N, 3))
    for i in range(N):
        _, Js[i], cs[i] = link_terms(y[2 * i], y[2 * i + 1], y[n + 2 * i], y[n + 2 * i + 1])
    for i in range(N):
        for l in range(N):
            w = N - max(i, l)                          # bodies distal to both links, in 0-based indices
            Mil = MASS * L * L * w * (Js[i].T @ Js[l])
            M[2 * i:2 * i + 2, 2 * l:2 * l + 2] = Mil
            F[2 * i:2 * i + 2] -= MASS * L * L * w * (Js[i].T @ cs[l])
        F[2 * i:2 * i + 2] += MASS * L * GRAVITY * (N - i) * Js[i][2, :]
    F[0] += 2.0 * np.sin(t)
    return M, F


@njit(cache=False)
def rhs(t, y, N):
    n = 2 * N; M, F = mass_forcing(t, y, N)
    out = np.zeros(2 * n); out[:n] = y[n:]
    out[n:] = np.linalg.solve(M + np.eye(n) * EPS, F)
    return out


@njit(cache=False)
def sample_terms(t, y, N, root):
    """Reference motion and exact IMU signals of all bodies at one state."""
    n = 2 * N; qdd = rhs(t, y, N)[n:]
    pos = np.zeros((N, 3)); vel = np.zeros((N, 3)); acc = np.zeros((N, 3)); gyr = np.zeros((N, 3)); rot = np.zeros((N, 3, 3))
    pe = root.copy(); ve = np.zeros(3); ae = np.zeros(3); gv = np.array([0.0, 0.0, -GRAVITY])
    for i in range(N):
        th, ph, thd, phd = y[2 * i], y[2 * i + 1], y[n + 2 * i], y[n + 2 * i + 1]
        z, J, c = link_terms(th, ph, thd, phd)
        qd = y[n + 2 * i:n + 2 * i + 2]
        zd = J @ qd; zdd = J @ qdd[2 * i:2 * i + 2] + c
        pos[i] = pe - .5 * L * z; vel[i] = ve - .5 * L * zd; a = ae - .5 * L * zdd
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        rot[i] = np.array([[cp, -sp * ct, sp * st], [sp, cp * ct, -cp * st], [0.0, st, ct]])
        acc[i] = rot[i].T @ (a - gv)
        gyr[i] = np.array([thd, phd * st, phd * ct])
        pe = pe - L * z; ve = ve - L * zd; ae = ae - L * zdd
    return pos, vel, rot, acc, gyr


@njit(cache=False)
def all_samples(ts, ys, N, root):
    T = len(ts)
    pos = np.zeros((T, N, 3)); vel = np.zeros((T, N, 3)); rot = np.zeros((T, N, 3, 3)); acc = np.zeros((T, N, 3)); gyr = np.zeros((T, N, 3))
    for k in range(T):
        pos[k], vel[k], rot[k], acc[k], gyr[k] = sample_terms(ts[k], ys[k], N, root)
    return pos, vel, rot, acc, gyr


def initial_state(N):
    """N = 3: the benchmark. N = 5, 10: a fixed draw (N = 5 uses the first five bodies)."""
    if N == 3:
        return np.array([np.pi / 4, 0.1, np.pi / 3, 0.2, np.pi / 6, 0.3, 0.5, 0.2, 0.4, 0.3, 0.2, 0.2])
    rng = np.random.default_rng(20261103)
    th = np.deg2rad(rng.uniform(30, 60, 10)); ph = rng.uniform(.1, .3, 10); thd = rng.uniform(.2, .5, 10); phd = rng.uniform(.2, .5, 10)
    y = np.zeros(4 * N)
    y[0::2][:N] = th[:N]; y[1::2][:N] = ph[:N]
    y[2 * N::2] = thd[:N]; y[2 * N + 1::2] = phd[:N]
    return y


def simulate(N):
    """Returns (measurements, reference, solver info) in the layout of step/data/clean."""
    y0 = initial_state(N); root = np.array([0.0, 0.0, 2.0 * N])
    t_eval = np.arange(0, DURATION, DT)
    sol = solve_ivp(lambda t, y: rhs(t, y, N), [0, DURATION], y0, method='Radau', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol.success and len(sol.t) == len(t_eval), sol.message
    pos, vel, rot, acc, gyr = all_samples(sol.t, sol.y.T, N, root)
    m = {'time': sol.t}; ref = {'time': sol.t}
    for b in range(1, N + 1):
        m[f'acc{b}'] = acc[:, b - 1]; m[f'gyr{b}'] = gyr[:, b - 1]
        ref[f'p{b}'] = pos[:, b - 1]; ref[f'v{b}'] = vel[:, b - 1]; ref[f'R{b}'] = rot[:, b - 1]
        ref[f'lever{b}'] = np.array([0., 0., 1.])
        if b > 1: ref[f'parent_lever{b}'] = np.array([0., 0., -1.])
    info = dict(nfev=int(sol.nfev), njev=int(sol.njev), nlu=int(sol.nlu), status=int(sol.status))
    return m, ref, sol.y.T, info


def lagrangian(q, qd, N):
    """T - V by direct evaluation; independent of mass_forcing, for the finite-difference check."""
    T = 0.0; V = 0.0; vend = np.zeros(3); pend = np.zeros(3)
    for i in range(N):
        z, J, _ = link_terms(q[2 * i], q[2 * i + 1], qd[2 * i], qd[2 * i + 1])
        vend = vend - L * (J @ qd[2 * i:2 * i + 2]); pend = pend - L * z
        T += .5 * MASS * vend @ vend; V += MASS * GRAVITY * pend[2]
    return T - V
