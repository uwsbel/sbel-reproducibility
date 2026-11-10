"""Wrappers for the JT-type filters with a (position SD, velocity SD) constraint-residual covariance.
The filter cores are the existing ones; only the residual covariance is built here."""
import numpy as np
import d_common as d
import jt_bias as jb
import jt_bias_oc as jo
import jt_bias_diag as jd

RSD_STEP = (0.01, 0.05)
LEVER_KEYS = ['L1p', 'L1d', 'L2p', 'L2d', 'L3p']
STATE_KEYS = ['r1', 'v1', 'q1', 'r2', 'v2', 'q2', 'r3', 'v3', 'q3']


def Rdiag(rsd):
    """Residual rows alternate [position (3), velocity (3)] per joint, three joints."""
    return np.diag(np.tile(np.r_[np.full(3, rsd[0] ** 2), np.full(3, rsd[1] ** 2)], 3))


def run_jt(m, states, init_levers, P, rsd, lever_q=0.0, oc=False):
    """d_common.run_jt with the residual covariance Rdiag(rsd)."""
    dt = float(np.median(np.diff(m['time'])))
    Qd = np.zeros((42, 42))
    for i in (0, 9, 18): Qd[i:i + 9, i:i + 9] = d.jtd.QK
    for i in range(27, 42, 3): Qd[i:i + 3, i:i + 3] = np.eye(3) * lever_q
    c = np.ascontiguousarray
    res = d.jtd.jt.run_jointtracker_3link_numba_core(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, d.G_VEC, d.PIVOT,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(),
        states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in LEVER_KEYS],
        P, Qd, Rdiag(rsd), 1, True, True, True, True, True, bool(oc))
    return {k: res[i] for i, k in enumerate(STATE_KEYS + LEVER_KEYS)}


def run_jt_bias(m, states, init_levers, lever_sd, rsd):
    """jt_bias.run_jt_bias (bias states, truth-type P0) with the residual covariance Rdiag(rsd)."""
    dt = float(np.median(np.diff(m['time']))); P, Q = jb.build_PQ(d.jtd.QK, lever_sd, dt, True)
    c = np.ascontiguousarray
    hist, hq, hL, hb, Pend = jb.jt_bias_core(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, d.G_VEC, d.PIVOT,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(), states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in LEVER_KEYS], np.zeros(18), P, Q, Rdiag(rsd))
    out = {'r1': hist[:, 0], 'v1': hist[:, 1], 'r2': hist[:, 2], 'v2': hist[:, 3], 'r3': hist[:, 4], 'v3': hist[:, 5],
           'q1': hq[:, 0], 'q2': hq[:, 1], 'q3': hq[:, 2], 'b': hb}
    out.update({k: hL[:, i] for i, k in enumerate(LEVER_KEYS)})
    return out


def run_jt_bias_oc(m, states, init_levers, lever_sd, rsd):
    """jt_bias_oc.run (JT+bias with the heading-consistent linearization); it already accepts a (p, v) tuple."""
    return jo.run(m, states, init_levers, lever_sd, d.jtd.QK, tuple(rsd), d.G_VEC, d.PIVOT, oc=True)


def run_jt_bias_diag(m, states, init_levers, lever_sd, rsd):
    """jt_bias_diag.run (instrumented JT+bias core) with the residual covariance Rdiag(rsd)."""
    dt = float(np.median(np.diff(m['time']))); P, Q = jb.build_PQ(d.jtd.QK, lever_sd, dt, True)
    c = np.ascontiguousarray
    hist, hq, hb, sp, su, al, ah = jd.jt_bias_core_diag(
        c(m['acc1']), c(m['gyr1']), c(m['acc2']), c(m['gyr2']), c(m['acc3']), c(m['gyr3']), dt, d.G_VEC, d.PIVOT,
        states['r1'].copy(), states['v1'].copy(), states['q1'].copy(), states['r2'].copy(), states['v2'].copy(), states['q2'].copy(),
        states['r3'].copy(), states['v3'].copy(), states['q3'].copy(),
        *[np.array(init_levers[k], dtype=float).copy() for k in LEVER_KEYS], np.zeros(18), P, Q, Rdiag(rsd))
    return {'r1': hist[:, 0], 'v1': hist[:, 1], 'r2': hist[:, 2], 'v2': hist[:, 3], 'r3': hist[:, 4], 'v3': hist[:, 5],
            'q1': hq[:, 0], 'q2': hq[:, 1], 'q3': hq[:, 2], 'b': hb, 'sd_pred': sp, 'sd_upd': su, 'alpha': al, 'att_head': ah}


def matched_sd(c):
    return max(abs(c) / 100 * 1.0, 0.01)


def variants(m, st, c, rsd):
    """The single-realization JT variants of Table 7 (JT-N excepted, see c1g_jt_prior_mid.py) and the JT with the matched
    lever prior of the text, for one (record, c)."""
    out = {}
    out['W1'] = run_jt(m, st, d.perturbed(c), d.P0(1.0), rsd)
    out['W1-RW'] = run_jt(m, st, d.perturbed(c), d.P0(1.0), rsd, lever_q=1e-9)
    out['W1-OC'] = run_jt(m, st, d.perturbed(c), d.P0(1.0), rsd, oc=True)
    out['P0m'] = run_jt(m, st, d.perturbed(c), d.P0(matched_sd(c)), rsd)
    out['JT-bias'] = run_jt_bias(m, st, d.perturbed(c), 1.0, rsd)
    out['JT-bias-OC'] = run_jt_bias_oc(m, st, d.perturbed(c), 1.0, rsd)
    return out
