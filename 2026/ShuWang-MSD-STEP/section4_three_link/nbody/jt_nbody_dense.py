"""Dense N-body coupled filters: the three-body coupled filters (review/jt_bias.py, JT+bias, and the heading-consistent
coupled filter) generalized to N bodies. Validation baseline only. State: [r v th] x N | levers (if estimated) | [ba bg] x N."""
import numpy as np
from numba import njit
import nbody_common as nc
from jt_checked import _skew_numba, quat_mult_numba, quat_to_rotmat_numba, _exp_quat_numba, right_jacobian
from jt_bias import prop_bias


@njit(cache=False)
def heading_dir(r, v, q, pivot, g, n):
    """Common rotation of the whole chain about the vertical through the support."""
    N = r.shape[0]; u = -g / np.sqrt(np.sum(g * g)); z = np.zeros(n); Rm = np.zeros((3, 3))
    for i in range(N):
        quat_to_rotmat_numba(q[i], Rm); o = 9 * i
        z[o:o + 3] = np.cross(u, r[i] - pivot); z[o + 3:o + 6] = np.cross(u, v[i]); z[o + 6:o + 9] = Rm.T @ u
    return z


@njit(cache=False)
def constraint_dense(N, n, lo, bo, r, v, q, L, w, pivot):
    """Residuals y (6N) and Jacobian H (6N x n). w: bias-corrected gyro; lo: first lever column, -1 if levers are fixed."""
    y = np.zeros(6 * N); H = np.zeros((6 * N, n)); I3 = np.eye(3); s = np.zeros((3, 3)); Rs = np.zeros((N, 3, 3))
    for i in range(N): quat_to_rotmat_numba(q[i], Rs[i])
    for j in range(N):
        row = 6 * j; rj = 9 * j; Rj = Rs[j]; Lp = L[2 * j]; wxL = np.cross(w[j], Lp)
        y[row:row + 3] = r[j] + Rj @ Lp; y[row + 3:row + 6] = v[j] + Rj @ wxL
        H[row:row + 3, rj:rj + 3] = I3
        _skew_numba(Lp, s); H[row:row + 3, rj + 6:rj + 9] = -Rj @ s
        g = bo + 6 * j + 3; H[row + 3:row + 6, g:g + 3] = Rj @ s
        H[row + 3:row + 6, rj + 3:rj + 6] = I3
        _skew_numba(wxL, s); H[row + 3:row + 6, rj + 6:rj + 9] = -Rj @ s
        if lo >= 0:
            c = lo + 3 * (2 * j); H[row:row + 3, c:c + 3] = Rj
            _skew_numba(w[j], s); H[row + 3:row + 6, c:c + 3] = Rj @ s
        if j == 0:
            y[row:row + 3] -= pivot
        else:
            Rp = Rs[j - 1]; Ld = L[2 * j - 1]; wxLd = np.cross(w[j - 1], Ld); rp = 9 * (j - 1)
            y[row:row + 3] -= r[j - 1] + Rp @ Ld; y[row + 3:row + 6] -= v[j - 1] + Rp @ wxLd
            H[row:row + 3, rp:rp + 3] = -I3
            _skew_numba(Ld, s); H[row:row + 3, rp + 6:rp + 9] = Rp @ s
            gp = bo + 6 * (j - 1) + 3; H[row + 3:row + 6, gp:gp + 3] = -(Rp @ s)
            H[row + 3:row + 6, rp + 3:rp + 6] = -I3
            _skew_numba(wxLd, s); H[row + 3:row + 6, rp + 6:rp + 9] = Rp @ s
            if lo >= 0:
                c = lo + 3 * (2 * j - 1); H[row:row + 3, c:c + 3] = -Rp
                _skew_numba(w[j - 1], s); H[row + 3:row + 6, c:c + 3] = -(Rp @ s)
    return y, H


@njit(cache=False)
def dense_core(acc, gyr, dt, g_vec, pivot, r, v, q, L, b, P, Qd, Rm, est, oc):
    T = acc.shape[0]; N = acc.shape[1]; nl = 2 * N - 1; n = P.shape[0]; nm = 9 * N
    lo = nm if est else -1; bo = nm + (3 * nl if est else 0)
    hr = np.zeros((T, N, 3)); hv = np.zeros((T, N, 3)); hq = np.zeros((T, N, 4)); hb = np.zeros((T, 6 * N))
    hL = np.zeros((T if est else 1, nl, 3)); w = np.zeros((N, 3))
    In = np.eye(n); Phi = np.eye(n)
    z_prev = heading_dir(r, v, q, pivot, g_vec, n); z_new = z_prev.copy()
    for k in range(T):
        if k > 0:
            p = k - 1
            Phi[:, :] = In
            for i in range(N):
                o = 9 * i; bi = bo + 6 * i
                r2, v2, q2, f, a, g = prop_bias(r[i], v[i], q[i], acc[p, i], gyr[p, i], acc[k, i], gyr[k, i], b[6 * i:6 * i + 3], b[6 * i + 3:6 * i + 6], dt, g_vec)
                r[i] = r2; v[i] = v2; q[i] = q2
                Phi[o:o + 9, o:o + 9] = f; Phi[o:o + 9, bi:bi + 3] = a; Phi[o:o + 9, bi + 3:bi + 6] = g
            if oc:
                z_new = heading_dir(r, v, q, pivot, g_vec, n)
                Phi = Phi - np.outer(Phi @ z_prev - z_new, z_prev) / (z_prev @ z_prev)
            P = Phi @ P @ Phi.T + Qd
            for i in range(N): w[i] = gyr[k, i] - b[6 * i + 3:6 * i + 6]
            y, H = constraint_dense(N, n, lo, bo, r, v, q, L, w, pivot)
            if oc:
                H = H - np.outer(H @ z_new, z_new) / (z_new @ z_new)
            PHt = P @ H.T; S = H @ PHt + Rm
            K = np.linalg.solve(S.T, PHt.T).T
            dx = K @ (-y)
            for i in range(N):
                o = 9 * i
                r[i] += dx[o:o + 3]; v[i] += dx[o + 3:o + 6]
                qn = quat_mult_numba(q[i], _exp_quat_numba(dx[o + 6:o + 9])); qn /= np.sqrt(np.sum(qn ** 2)); q[i] = qn
            if est:
                for i in range(nl): L[i] += dx[lo + 3 * i:lo + 3 * i + 3]
            b += dx[bo:bo + 6 * N]
            I_KH = In - K @ H
            P = I_KH @ P @ I_KH.T + K @ Rm @ K.T
            reset = In.copy()
            for i in range(N):
                o = 9 * i; reset[o + 6:o + 9, o + 6:o + 9] = right_jacobian(dx[o + 6:o + 9])
            P = reset @ P @ reset.T
            if k % 1000 == 0 and (not np.isfinite(P).all() or not np.isfinite(dx).all()):
                raise FloatingPointError('non-finite coupled update')
            P = 0.5 * (P + P.T)
            if oc:
                z_prev = z_new
        hr[k] = r; hv[k] = v; hq[k] = q; hb[k] = b
        if est: hL[k] = L
    return hr, hv, hq, hL, hb


def dense_Q(N, est, dt):
    nm, nl, n = nc.layout(N, est); Q = np.zeros((n, n))
    for o in range(0, nm, 9): Q[o:o + 9, o:o + 9] = nc.QK
    for o in range(nm + nl, n, 6):
        Q[o:o + 3, o:o + 3] = np.eye(3) * nc.SBA ** 2 * dt; Q[o + 3:o + 6, o + 3:o + 6] = np.eye(3) * nc.SBG ** 2 * dt
    return Q


def run(m, N, r0, v0, q0, L0, est, oc, lever_sd, motion_sd):
    dt = float(np.median(np.diff(m['time']))); acc, gyr = nc.pack(m, N)
    P = nc.initial_cov(N, est, lever_sd, motion_sd); Q = dense_Q(N, est, dt); Rm = np.diag(nc.rdiag(N))
    hr, hv, hq, hL, hb = dense_core(acc, gyr, dt, nc.G_VEC, nc.pivot(N), r0.copy(), v0.copy(), q0.copy(), np.array(L0, float).copy(),
                                    np.zeros(6 * N), P, Q, Rm, bool(est), bool(oc))
    return dict(r=hr, v=hv, q=hq, L=hL, b=hb)
