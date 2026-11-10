"""Instrumented copy of jt_bias.jt_bias_core: records the heading SD along the
common-heading direction z (after propagation and after update) and the heading correction of each update."""
import numpy as np
from numba import njit
import jt_bias as jb
from jt_bias import (quat_mult_numba, quat_to_rotmat_numba, _exp_quat_numba, right_jacobian, constraint_terms,
                     prop_bias, bias_H, NX)


@njit(cache=False)
def heading_dir(r1, v1, q1, r2, v2, q2, r3, v3, q3, pivot, g):
    u = -g / np.sqrt(np.sum(g * g)); z = np.zeros(NX); Rm = np.zeros((3, 3))
    rs = (r1, r2, r3); vs = (v1, v2, v3); qs = (q1, q2, q3)
    for i in range(3):
        quat_to_rotmat_numba(qs[i], Rm); o = 9 * i
        z[o:o + 3] = np.cross(u, rs[i] - pivot); z[o + 3:o + 6] = np.cross(u, vs[i]); z[o + 6:o + 9] = Rm.T @ u
    return z


@njit(cache=False)
def jt_bias_core_diag(acc1, gyr1, acc2, gyr2, acc3, gyr3, dt, g_vec, pos_fixed_SJ,
                      r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p, b, P, Qd, R_meas):
    N = len(acc1)
    hist = np.zeros((N, 9, 3)); hq = np.zeros((N, 3, 4)); hb = np.zeros((N, 18))
    sd_pred = np.zeros(N); sd_upd = np.zeros(N); alpha = np.zeros(N); att_head = np.zeros((N, 3))
    Rt = np.zeros((3, 3)); uv = -g_vec / np.sqrt(np.sum(g_vec * g_vec))
    I60 = np.eye(NX); Phi = np.eye(NX)
    z = heading_dir(r1, v1, q1, r2, v2, q2, r3, v3, q3, pos_fixed_SJ, g_vec)
    sd_pred[0] = np.sqrt(z @ P @ z) / (z @ z); sd_upd[0] = sd_pred[0]
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
            z = heading_dir(r1, v1, q1, r2, v2, q2, r3, v3, q3, pos_fixed_SJ, g_vec); zz = z @ z
            sd_pred[k] = np.sqrt(max(z @ P @ z, 0.0)) / zz
            y, H42 = constraint_terms(r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p,
                                      gyr1[k] - b[3:6], gyr2[k] - b[9:12], gyr3[k] - b[15:18], pos_fixed_SJ, True)
            H = bias_H(q1, q2, q3, L1p, L1d, L2p, L2d, L3p); H[:, 0:42] = H42
            PHt = P @ H.T; S = H @ PHt + R_meas
            K = np.linalg.solve(S.T, PHt.T).T
            dx = K @ (-y)
            alpha[k] = (z @ dx) / zz
            quat_to_rotmat_numba(q1, Rt); att_head[k, 0] = uv @ (Rt @ dx[6:9])
            quat_to_rotmat_numba(q2, Rt); att_head[k, 1] = uv @ (Rt @ dx[15:18])
            quat_to_rotmat_numba(q3, Rt); att_head[k, 2] = uv @ (Rt @ dx[24:27])
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
            sd_upd[k] = np.sqrt(max(z @ P @ z, 0.0)) / zz
        hist[k, 0] = r1; hist[k, 1] = v1; hist[k, 2] = r2; hist[k, 3] = v2; hist[k, 4] = r3; hist[k, 5] = v3
        hq[k, 0] = q1; hq[k, 1] = q2; hq[k, 2] = q3
        hb[k] = b
    return hist, hq, hb, sd_pred, sd_upd, alpha, att_head


def run(m, states, init_levers, lever_sd, kin_Q, residual_sd, g_vec, pivot):
    dt = float(np.median(np.diff(m['time']))); P, Q = jb.build_PQ(kin_Q, lever_sd, dt, True)
    Rm = np.eye(18) * residual_sd ** 2; c = np.ascontiguousarray
    lab = ['L1p', 'L1d', 'L2p', 'L2d', 'L3p']
    hist, hq, hb, sp, su, al, ah = jt_bias_core_diag(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, g_vec, pivot,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(), states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in lab], np.zeros(18), P, Q, Rm)
    return {'r1': hist[:, 0], 'v1': hist[:, 1], 'r2': hist[:, 2], 'v2': hist[:, 3], 'r3': hist[:, 4], 'v3': hist[:, 5],
            'q1': hq[:, 0], 'q2': hq[:, 1], 'q3': hq[:, 2], 'b': hb, 'sd_pred': sp, 'sd_upd': su, 'alpha': al, 'att_head': ah}
