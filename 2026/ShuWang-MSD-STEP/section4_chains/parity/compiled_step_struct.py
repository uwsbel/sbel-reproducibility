"""Structured version of compiled_step.filter_spherical, used for timing only.
Same equations; the state propagation and residual lines are unchanged. What is exploited, matching jt_nbody_struct.py:
- Phi: its known blocks are applied to the rows and columns of P.
- Q: constant and precomputed (the rotation in G cancels, as in QK and qb of the coupled filters).
- H: sparse, so PH^T and HPH^T use only the rows and columns it touches.
- Joseph form in expanded algebra, P - KC - (KC)^T + KSK^T. joseph=False gives P - KC, for a timing sensitivity.
- Heading-consistent projection (Body 1): rank-one corrections of Phi and H.
The solve is LU, as in compiled_step."""
import numpy as np
from numba import njit
from compiled_step import skew3, exp3


@njit(cache=False)
def _phi_rows(P, dt, Ap, Bp, Av, Bv, Th, t3):
    """P <- Phi P."""
    for j in range(15):
        for i in range(3):
            P[i, j] += (dt * P[3 + i, j] + Ap[i, 0] * P[6, j] + Ap[i, 1] * P[7, j] + Ap[i, 2] * P[8, j]
                        + Bp[i, 0] * P[9, j] + Bp[i, 1] * P[10, j] + Bp[i, 2] * P[11, j])
        for i in range(3):
            P[3 + i, j] += (Av[i, 0] * P[6, j] + Av[i, 1] * P[7, j] + Av[i, 2] * P[8, j]
                            + Bv[i, 0] * P[9, j] + Bv[i, 1] * P[10, j] + Bv[i, 2] * P[11, j])
        for i in range(3):
            t3[i] = Th[i, 0] * P[6, j] + Th[i, 1] * P[7, j] + Th[i, 2] * P[8, j] - dt * P[12 + i, j]
        P[6, j] = t3[0]; P[7, j] = t3[1]; P[8, j] = t3[2]


@njit(cache=False)
def _phi_cols(P, dt, Ap, Bp, Av, Bv, Th, t3):
    """P <- P Phi^T."""
    for j in range(15):
        x = P[j]
        for i in range(3):
            x[i] += (dt * x[3 + i] + Ap[i, 0] * x[6] + Ap[i, 1] * x[7] + Ap[i, 2] * x[8]
                     + Bp[i, 0] * x[9] + Bp[i, 1] * x[10] + Bp[i, 2] * x[11])
        for i in range(3):
            x[3 + i] += (Av[i, 0] * x[6] + Av[i, 1] * x[7] + Av[i, 2] * x[8]
                         + Bv[i, 0] * x[9] + Bv[i, 1] * x[10] + Bv[i, 2] * x[11])
        for i in range(3):
            t3[i] = Th[i, 0] * x[6] + Th[i, 1] * x[7] + Th[i, 2] * x[8] - dt * x[12 + i]
        x[6] = t3[0]; x[7] = t3[1]; x[8] = t3[2]


@njit(cache=False)
def _phi_vec(x, dt, Ap, Bp, Av, Bv, Th):
    """Phi x."""
    o = x.copy()
    for i in range(3):
        o[i] = (x[i] + dt * x[3 + i] + Ap[i, 0] * x[6] + Ap[i, 1] * x[7] + Ap[i, 2] * x[8]
                + Bp[i, 0] * x[9] + Bp[i, 1] * x[10] + Bp[i, 2] * x[11])
        o[3 + i] = (x[3 + i] + Av[i, 0] * x[6] + Av[i, 1] * x[7] + Av[i, 2] * x[8]
                    + Bv[i, 0] * x[9] + Bv[i, 1] * x[10] + Bv[i, 2] * x[11])
        o[6 + i] = Th[i, 0] * x[6] + Th[i, 1] * x[7] + Th[i, 2] * x[8] - dt * x[12 + i]
    return o


@njit(cache=False)
def filter_spherical_struct(acc, gyr, dt, p0, v0, R0, ell, gravity, sd, noise, sigma_p, sigma_v, jp, jv, midpoint, oc, support, joseph):
    n = len(acc); p = p0.copy(); v = v0.copy(); r = R0.copy(); ba = np.zeros(3); bg = np.zeros(3)
    P = np.diag(sd**2); e3 = np.eye(3)
    op = np.empty((n, 3)); ov = np.empty((n, 3)); orr = np.empty((n, 3, 3)); oba = np.empty((n, 3)); obg = np.empty((n, 3))
    sa2 = noise[0]**2; sg2 = noise[3]**2; sb2 = noise[6]**2; sw2 = noise[9]**2
    qpp = .25 * sa2 * dt**4; qpv = .5 * sa2 * dt**3; qvv = sa2 * dt**2; qth = sg2 * dt**2; qba = sb2 * dt; qbg = sw2 * dt
    vp = sigma_p**2; vv = sigma_v**2
    C = np.zeros((6, 15)); S = np.zeros((6, 6)); K = np.zeros((15, 6)); y = np.zeros(6); dx = np.zeros(15); t3 = np.zeros(3)
    up = -gravity / np.sqrt(np.dot(gravity, gravity)); z6 = np.zeros(6)
    n_prev = np.concatenate((np.cross(up, p - support), np.cross(up, v), r.T @ up, z6)); n_new = n_prev.copy(); hz = np.zeros(6)
    for k in range(n):
        if k:
            w = .5 * (gyr[k - 1] + gyr[k]) - bg; f = .5 * (acc[k - 1] + acc[k]) - ba
            rf = r @ exp3(.5 * w * dt) if midpoint else r
            r = r @ exp3(w * dt)
            if not midpoint: rf = r
            aw = rf @ f + gravity; p = p + v * dt + .5 * aw * dt**2; v = v + aw * dt
            B = rf @ skew3(f)
            Ap = -.5 * dt**2 * B; Bp = -.5 * dt**2 * rf; Av = -dt * B; Bv = -dt * rf; Th = e3 - skew3(w) * dt
            if oc:                       # Phi_oc = Phi - a z^T: P <- Phi P Phi^T - h a^T - a h^T
                n_new = np.concatenate((np.cross(up, p - support), np.cross(up, v), r.T @ up, z6))
                a = (_phi_vec(n_prev, dt, Ap, Bp, Av, Bv, Th) - n_new) / np.dot(n_prev, n_prev)
                wv = np.zeros(15)
                for i in range(9):
                    wv += n_prev[i] * P[i]
                h = _phi_vec(wv, dt, Ap, Bp, Av, Bv, Th) - .5 * np.dot(n_prev, wv) * a
            _phi_rows(P, dt, Ap, Bp, Av, Bv, Th, t3)
            _phi_cols(P, dt, Ap, Bp, Av, Bv, Th, t3)
            if oc:
                for i in range(15):
                    for j in range(9):
                        t = h[i] * a[j]
                        P[i, j] -= t; P[j, i] -= t
            for i in range(3):
                P[i, i] += qpp; P[i, 3 + i] += qpv; P[3 + i, i] += qpv; P[3 + i, 3 + i] += qvv
                P[6 + i, 6 + i] += qth; P[9 + i, 9 + i] += qba; P[12 + i, 12 + i] += qbg
            u = np.cross(gyr[k] - bg, ell); hp = p + r @ ell - jp[k]; hv = v + r @ u - jv[k]
            y[:3] = hp; y[3:] = hv
            E1 = -(r @ skew3(ell)); E2 = -(r @ skew3(u))
            for i in range(3):           # C = H P
                for j in range(15):
                    C[i, j] = P[i, j] + E1[i, 0] * P[6, j] + E1[i, 1] * P[7, j] + E1[i, 2] * P[8, j]
                    C[3 + i, j] = (P[3 + i, j] + E2[i, 0] * P[6, j] + E2[i, 1] * P[7, j] + E2[i, 2] * P[8, j]
                                   - E1[i, 0] * P[12, j] - E1[i, 1] * P[13, j] - E1[i, 2] * P[14, j])
            if oc:                       # H_oc = H - hz e^T
                for i in range(3):
                    hz[i] = n_new[i] + E1[i, 0] * n_new[6] + E1[i, 1] * n_new[7] + E1[i, 2] * n_new[8]
                    hz[3 + i] = n_new[3 + i] + E2[i, 0] * n_new[6] + E2[i, 1] * n_new[7] + E2[i, 2] * n_new[8]
                e = n_new / np.dot(n_new, n_new)
                pe = np.zeros(15)
                for i in range(9):
                    pe += e[i] * P[i]
                for row in range(6):
                    C[row] -= hz[row] * pe
            for row in range(6):         # S = C H^T + V
                x = C[row]
                for i in range(3):
                    S[row, i] = x[i] + x[6] * E1[i, 0] + x[7] * E1[i, 1] + x[8] * E1[i, 2]
                    S[row, 3 + i] = (x[3 + i] + x[6] * E2[i, 0] + x[7] * E2[i, 1] + x[8] * E2[i, 2]
                                     - x[12] * E1[i, 0] - x[13] * E1[i, 1] - x[14] * E1[i, 2])
                if oc:
                    t = np.dot(x, e)
                    for i in range(6):
                        S[row, i] -= t * hz[i]
            for i in range(3):
                S[i, i] += vp; S[3 + i, 3 + i] += vv
            Kt = np.linalg.solve(S, C)
            for i in range(15):
                t = 0.0
                for row in range(6):
                    K[i, row] = Kt[row, i]; t -= Kt[row, i] * y[row]
                dx[i] = t
            KC = K @ C
            if joseph:
                KSK = (K @ S) @ Kt
                for i in range(15):
                    for j in range(i, 15):
                        t = .5 * (P[i, j] + P[j, i]) + .5 * (KSK[i, j] + KSK[j, i]) - KC[i, j] - KC[j, i]
                        P[i, j] = t; P[j, i] = t
            else:
                for i in range(15):
                    for j in range(i, 15):
                        t = .5 * (P[i, j] + P[j, i] - KC[i, j] - KC[j, i])
                        P[i, j] = t; P[j, i] = t
            p += dx[:3]; v += dx[3:6]; r = r @ exp3(dx[6:9]); ba += dx[9:12]; bg += dx[12:15]
            if oc: n_prev = n_new
        op[k] = p; ov[k] = v; orr[k] = r; oba[k] = ba; obg[k] = bg
    return op, ov, orr, oba, obg


def run_chain(m, cals, cfg, joseph=True):
    """As compiled_step.run_chain, with the structured filter."""
    dt = float(np.median(np.diff(m['time']))); n = len(m['time']); hist = {}
    sd = np.repeat(cfg['initial_state_sd'], 3); noise = np.repeat([cfg['acc_noise_sd'], cfg['gyro_noise_sd'], cfg['ba_rw_sd'], cfg['bg_rw_sd']], 3)
    fix = cfg.get('stage2_fix', {}); midpoint = fix.get('propagation') == 'midpoint'
    for b in range(1, cfg['n_bodies'] + 1):
        c = cals[b]
        oc = bool(fix.get('oc_heading', False)) and b == 1 and cfg['joint_types'][0] == 'spherical'
        if b == 1: jp = np.tile(np.asarray(cfg['fixed_support'], float), (n, 1)); jv = np.zeros((n, 3))
        else:
            h = hist[b - 1]; lp = c['parent_lever']; rp = h['R']
            jp = h['p'] + np.einsum('nij,j->ni', rp, lp)
            jv = h['v'] + np.einsum('nij,nj->ni', rp, np.cross(m[f'gyr{b - 1}'] - h['bg'], lp))
        values = filter_spherical_struct(m[f'acc{b}'], m[f'gyr{b}'], dt, c['p0'], c['v0'], c['R0'], c['lever'], np.asarray(cfg['gravity'], float), sd, noise,
                                         cfg['constraint_sd']['point_p'], cfg['constraint_sd']['point_v'], jp, jv, midpoint, oc, np.asarray(cfg['fixed_support'], float), bool(joseph))
        hist[b] = dict(zip(['p', 'v', 'R', 'ba', 'bg'], values))
    return hist
