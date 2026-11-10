"""JT+bias+OC: copy of jt_bias.jt_bias_core with the heading-consistent linearization of
jt_checked.run_jointtracker_3link_numba_core (oc branch) extended to 60 states.
jt_bias.py and jt_checked.py are imported, not modified."""
import numpy as np
from numba import njit
import jt_bias as jb
from jt_bias import (quat_mult_numba, _exp_quat_numba, right_jacobian, constraint_terms, prop_bias, bias_H, NX)
from jt_bias_diag import heading_dir


@njit(cache=False)
def jt_bias_oc_core(acc1, gyr1, acc2, gyr2, acc3, gyr3, dt, g_vec, pos_fixed_SJ,
                    r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p, b, P, Qd, R_meas, oc, check_every):
    N = len(acc1)
    hist = np.zeros((N, 9, 3)); hq = np.zeros((N, 3, 4)); hL = np.zeros((N, 5, 3)); hb = np.zeros((N, 18))
    proj = np.zeros((N // check_every + 1, 2)); npj = 0
    I60 = np.eye(NX); Phi = np.eye(NX)
    z_prev = heading_dir(r1, v1, q1, r2, v2, q2, r3, v3, q3, pos_fixed_SJ, g_vec); z_new = z_prev.copy()
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
            if oc:
                z_new = heading_dir(r1, v1, q1, r2, v2, q2, r3, v3, q3, pos_fixed_SJ, g_vec)
                Phi = Phi - np.outer(Phi @ z_prev - z_new, z_prev) / (z_prev @ z_prev)
            P = Phi @ P @ Phi.T + Qd
            y, H42 = constraint_terms(r1, v1, q1, r2, v2, q2, r3, v3, q3, L1p, L1d, L2p, L2d, L3p,
                                      gyr1[k] - b[3:6], gyr2[k] - b[9:12], gyr3[k] - b[15:18], pos_fixed_SJ, True)
            H = bias_H(q1, q2, q3, L1p, L1d, L2p, L2d, L3p); H[:, 0:42] = H42
            if oc:
                H = H - np.outer(H @ z_new, z_new) / (z_new @ z_new)
                if k % check_every == 0:
                    proj[npj, 0] = np.sqrt(np.sum((Phi @ z_prev - z_new) ** 2)) / np.sqrt(np.sum(z_new ** 2))
                    proj[npj, 1] = np.sqrt(np.sum((H @ z_new) ** 2)) / (np.sqrt(np.sum(H ** 2)) * np.sqrt(np.sum(z_new ** 2)))
                    npj += 1
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
                raise FloatingPointError('non-finite JT+bias+OC update')
            P = 0.5 * (P + P.T)
            if oc:
                z_prev = z_new
        hist[k, 0] = r1; hist[k, 1] = v1; hist[k, 2] = r2; hist[k, 3] = v2; hist[k, 4] = r3; hist[k, 5] = v3
        hq[k, 0] = q1; hq[k, 1] = q2; hq[k, 2] = q3
        hL[k, 0] = L1p; hL[k, 1] = L1d; hL[k, 2] = L2p; hL[k, 3] = L2d; hL[k, 4] = L3p
        hb[k] = b
    return hist, hq, hL, hb, proj[:npj]


def run(m, states, init_levers, lever_sd, kin_Q, residual_sd, g_vec, pivot, oc=True, check_every=500, motion_sd=None):
    dt = float(np.median(np.diff(m['time']))); P, Q = jb.build_PQ(kin_Q, lever_sd, dt, True, motion_sd=motion_sd)
    if np.ndim(residual_sd) == 0:
        Rm = np.eye(18) * residual_sd ** 2
    else:   # (position SD, velocity SD); residual rows alternate [p, v] per joint
        Rm = np.diag(np.tile(np.r_[np.full(3, residual_sd[0] ** 2), np.full(3, residual_sd[1] ** 2)], 3))
    c = np.ascontiguousarray
    lab = ['L1p', 'L1d', 'L2p', 'L2d', 'L3p']
    hist, hq, hL, hb, proj = jt_bias_oc_core(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, g_vec, pivot,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(), states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in lab], np.zeros(18), P, Q, Rm, bool(oc), int(check_every))
    out = {'r1': hist[:, 0], 'v1': hist[:, 1], 'r2': hist[:, 2], 'v2': hist[:, 3], 'r3': hist[:, 4], 'v3': hist[:, 5],
           'q1': hq[:, 0], 'q2': hq[:, 1], 'q3': hq[:, 2], 'b': hb, 'proj_check': proj}
    out.update({k: hL[:, i] for i, k in enumerate(lab)})
    return out
