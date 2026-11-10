"""JT+bias: the JT-Qn0 filter of step/src/jt_checked.py with per-body accelerometer and
gyroscope bias states (60 states). jt_checked.py is imported, not modified.
State order: [r1 v1 th1 | r2 v2 th2 | r3 v3 th3 | L1p L1d L2p L2d L3p | ba1 bg1 | ba2 bg2 | ba3 bg3]."""
import sys
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0, str(Path(__file__).parent.parent / 'step' / 'src'))  # RELEASE: path changed for this repository
import jt_checked as jtc
from jt_checked import (_skew_numba, quat_mult_numba, quat_to_rotmat_numba, _exp_quat_numba, rot_exp,
                        right_jacobian, propagate_body, constraint_terms)

NX = 60


@njit(cache=False)
def prop_bias(r, v, q, al, gl, ar, gr, ba, bg, dt, g):
    """Bias-corrected propagation of one body; returns the 9x9 motion block and the 9x3 bias blocks."""
    r2, v2, q2, phi = propagate_body(r, v, q, al - ba, gl - bg, ar - ba, gr - bg, dt, g, True)
    rot = np.zeros((3, 3)); quat_to_rotmat_numba(q, rot)
    w = .5 * ((gl - bg) + (gr - bg)); rmid = rot @ rot_exp(w * dt * .5)
    Ba = np.zeros((9, 3)); Bg = np.zeros((9, 3))
    Ba[0:3, :] = -.5 * rmid * dt * dt; Ba[3:6, :] = -rmid * dt
    Bg[6:9, :] = -right_jacobian(w * dt) * dt
    return r2, v2, q2, phi, Ba, Bg


@njit(cache=False)
def bias_H(q1, q2, q3, L1p, L1d, L2p, L2d, L3p):
    """d(joint-velocity residuals)/d(gyro biases) = R [l]_x per lever term (18 x 60, bias columns only)."""
    Hb = np.zeros((18, NX)); R1 = np.zeros((3, 3)); R2 = np.zeros((3, 3)); R3 = np.zeros((3, 3)); s = np.zeros((3, 3))
    quat_to_rotmat_numba(q1, R1); quat_to_rotmat_numba(q2, R2); quat_to_rotmat_numba(q3, R3)
    g1, g2, g3 = 45, 51, 57
    _skew_numba(L1p, s); Hb[3:6, g1:g1 + 3] = R1 @ s
    _skew_numba(L2p, s); Hb[9:12, g2:g2 + 3] = R2 @ s
    _skew_numba(L1d, s); Hb[9:12, g1:g1 + 3] = -(R1 @ s)
    _skew_numba(L3p, s); Hb[15:18, g3:g3 + 3] = R3 @ s
    _skew_numba(L2d, s); Hb[15:18, g2:g2 + 3] = -(R2 @ s)
    return Hb


@njit(cache=False)
def jt_bias_core(acc1, gyr1, acc2, gyr2, acc3, gyr3, dt, g_vec, pos_fixed_SJ,
                 r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p, b, P, Qd, R_meas):
    N = len(acc1)
    hist = np.zeros((N, 9, 3)); hq = np.zeros((N, 3, 4)); hL = np.zeros((N, 5, 3)); hb = np.zeros((N, 18))
    I60 = np.eye(NX); Phi = np.eye(NX)
    for k in range(N):
        if k > 0:
            p = k - 1
            Phi[:, :] = I60
            r1, v1, q1, f1, a1, w1 = prop_bias(r1, v1, q1, acc1[p], gyr1[p], acc1[k], gyr1[k], b[0:3], b[3:6], dt, g_vec)
            r2, v2, q2, f2, a2, w2 = prop_bias(r2, v2, q2, acc2[p], gyr2[p], acc2[k], gyr2[k], b[6:9], b[9:12], dt, g_vec)
            r3, v3, q3, f3, a3, w3 = prop_bias(r3, v3, q3, acc3[p], gyr3[p], acc3[k], gyr3[k], b[12:15], b[15:18], dt, g_vec)
            Phi[0:9, 0:9] = f1; Phi[0:9, 42:45] = a1; Phi[0:9, 45:48] = w1
            Phi[9:18, 9:18] = f2; Phi[9:18, 48:51] = a2; Phi[9:18, 51:54] = w2
            Phi[18:27, 18:27] = f3; Phi[18:27, 54:57] = a3; Phi[18:27, 57:60] = w3
            P = Phi @ P @ Phi.T + Qd
            y, H42 = constraint_terms(r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p,
                                      gyr1[k] - b[3:6], gyr2[k] - b[9:12], gyr3[k] - b[15:18], pos_fixed_SJ, True)
            H = bias_H(q1, q2, q3, L1p, L1d, L2p, L2d, L3p); H[:, 0:42] = H42
            PHt = P @ H.T; S = H @ PHt + R_meas
            K = np.linalg.solve(S.T, PHt.T).T
            dx = K @ (-y)
            r1 += dx[0:3]; v1 += dx[3:6]; q1 = quat_mult_numba(q1, _exp_quat_numba(dx[6:9])); q1 /= np.sqrt(np.sum(q1 ** 2))
            r2 += dx[9:12]; v2 += dx[12:15]; q2 = quat_mult_numba(q2, _exp_quat_numba(dx[15:18])); q2 /= np.sqrt(np.sum(q2 ** 2))
            r3 += dx[18:21]; v3 += dx[21:24]; q3 = quat_mult_numba(q3, _exp_quat_numba(dx[24:27])); q3 /= np.sqrt(np.sum(q3 ** 2))
            L1p += dx[27:30]; L1d += dx[30:33]; L2p += dx[33:36]; L2d += dx[36:39]; L3p += dx[39:42]
            b += dx[42:60]
            I_KH = I60 - K @ H
            P = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
            reset = I60.copy()
            reset[6:9, 6:9] = right_jacobian(dx[6:9]); reset[15:18, 15:18] = right_jacobian(dx[15:18]); reset[24:27, 24:27] = right_jacobian(dx[24:27])
            P = reset @ P @ reset.T
            if k % 1000 == 0 and (not np.isfinite(P).all() or not np.isfinite(dx).all()):
                raise FloatingPointError('non-finite JT+bias update')
            P = 0.5 * (P + P.T)
        hist[k, 0] = r1; hist[k, 1] = v1; hist[k, 2] = r2; hist[k, 3] = v2; hist[k, 4] = r3; hist[k, 5] = v3
        hq[k, 0] = q1; hq[k, 1] = q2; hq[k, 2] = q3
        hL[k, 0] = L1p; hL[k, 1] = L1d; hL[k, 2] = L2p; hL[k, 3] = L2d; hL[k, 4] = L3p
        hb[k] = b
    return hist, hq, hL, hb, P


def build_PQ(kin_Q, lever_sd, dt, bias=True, P_motion=1e-3,
             ba0_sd=0.02, bg0_sd=0.002, sba=1e-4, sbg=1e-5, motion_sd=None):
    """motion_sd = (position, velocity, attitude) standard deviations; if None, P_motion * I is used."""
    P = np.zeros((NX, NX)); Q = np.zeros((NX, NX))
    if motion_sd is None:
        P[:27, :27] = np.eye(27) * P_motion
    else:
        for o in (0, 9, 18):
            for j, sdv in enumerate(motion_sd): P[o + 3 * j:o + 3 * j + 3, o + 3 * j:o + 3 * j + 3] = np.eye(3) * sdv ** 2
    P[27:42, 27:42] = np.eye(15) * lever_sd ** 2
    for i in (0, 9, 18): Q[i:i + 9, i:i + 9] = kin_Q
    if bias:
        for o in (42, 48, 54):
            P[o:o + 3, o:o + 3] = np.eye(3) * ba0_sd ** 2; P[o + 3:o + 6, o + 3:o + 6] = np.eye(3) * bg0_sd ** 2
            Q[o:o + 3, o:o + 3] = np.eye(3) * sba ** 2 * dt; Q[o + 3:o + 6, o + 3:o + 6] = np.eye(3) * sbg ** 2 * dt
    return P, Q


def run_jt_bias(m, states, init_levers, lever_sd, kin_Q, residual_sd, g_vec, pivot, bias=True, motion_sd=None):
    dt = float(np.median(np.diff(m['time']))); P, Q = build_PQ(kin_Q, lever_sd, dt, bias, motion_sd=motion_sd)
    Rm = np.eye(18) * residual_sd ** 2; c = np.ascontiguousarray
    lab = ['L1p', 'L1d', 'L2p', 'L2d', 'L3p']
    hist, hq, hL, hb, Pend = jt_bias_core(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, g_vec, pivot,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(), states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in lab], np.zeros(18), P, Q, Rm)
    out = {'r1': hist[:, 0], 'v1': hist[:, 1], 'r2': hist[:, 2], 'v2': hist[:, 3], 'r3': hist[:, 4], 'v3': hist[:, 5],
           'q1': hq[:, 0], 'q2': hq[:, 1], 'q3': hq[:, 2], 'b': hb, 'P_end': Pend}
    out.update({k: hL[:, i] for i, k in enumerate(lab)})
    return out
