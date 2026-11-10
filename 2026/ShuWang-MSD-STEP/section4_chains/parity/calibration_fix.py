"""Stage I of step/src/calibration.py (calibration.calibrate) for the chain runner parity_fix.py; it reproduces
calibration.calibrate exactly. Unknowns: [rotation, (parent lever), lever, (axis)]. Stage I does not estimate the
sensor biases."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import calibration as cal
from core import tangent, increments, apply, fit_best, initial_guesses, unit


def calibrate(body, acc, gyr, dt, cfg, parent=None):
    g = np.array(cfg['gravity']); nc = len(acc); rj = cfg['joint_types'][body - 1] == 'revolute'
    f, w, A = cal.inputs(acc, gyr, dt, cfg['sg_windows'][body - 1]); M = increments(gyr, dt)
    il = 3 if parent is None else 6
    if parent is not None:
        Rp = parent['R'][:nc]
        fp, wp, Ap = cal.inputs(parent['acc'][:nc] - parent['ba'][:nc], parent['gyr'][:nc] - parent['bg'][:nc], dt, cfg['sg_windows'][body - 1])
    if rj:
        _, _, V = np.linalg.svd(w, full_matrices=False); nbase = V[0]
        if nbase[np.argmax(np.abs(nbase))] < 0: nbase = -nbase
        T = tangent(nbase)

        def axis(x): return R.from_rotvec(T @ x).apply(nbase)

    def fun(x):
        Rc = R.from_rotvec(x[:3]).as_matrix() @ M; lc = x[il:il + 3]; v = f + np.einsum('nij,j->ni', A, lc); wc = w
        if parent is None:
            ra = apply(Rc, v) + g; wr = wc
        else:
            ra = apply(Rc, v) - apply(Rp, fp + np.einsum('nij,j->ni', Ap, x[3:6]))
            if rj:
                u = apply(np.transpose(Rc, (0, 2, 1)), apply(Rp, wp))
                wr = wc - u
            else: wr = None
        if cal.STAGE1_KEEP is not None: ra = ra[cal.STAGE1_KEEP]
        if not rj: return ra.ravel()
        n = axis(x[-2:]); rw = wr - (wr @ n)[:, None] * n
        if cal.STAGE1_KEEP is not None: rw = rw[cal.STAGE1_KEEP]
        return np.r_[ra.ravel(), cfg['axis_residual_multipliers'][body - 1] * rw.ravel()]

    rng = np.random.default_rng(cfg['initialization_seed'] + body)
    if cfg.get('random_starts', False):
        starts = [np.r_[R.random(random_state=rng).as_rotvec(), rng.uniform(-.5, .5, 3 if parent is None else 6), np.zeros(2) if rj else []] for _ in range(cfg['n_starts'])]
    else:
        bases = initial_guesses(acc, g)
        bases += [(R.from_rotvec(unit(g) * a) * R.from_rotvec(bases[1])).as_rotvec() for a in [np.pi / 2, np.pi, 3 * np.pi / 2]]
        starts = [np.r_[r, rng.uniform(-.1, .1, 3 if parent is None else 6), np.zeros(2) if rj else []] for r in bases]
    x, info = fit_best(fun, starts, cfg['max_nfev'][body - 1], cfg['solver_tolerance'])
    _sd = cfg.get('stage1_eiv_gyro_sd', cfg['gyro_noise_sd'])
    if cfg.get('stage1_eiv', False) and _sd > 0:
        seed = cfg['stage1_eiv_seed'] + 10 * body; win = cfg['sg_windows'][body - 1]; nmc = cfg['stage1_eiv_mc']
        Qc = cal.eiv_Q(gyr, dt, win, cal.STAGE1_KEEP, _sd, seed, nmc)
        Qp = cal.eiv_Q(parent['gyr'][:nc] - parent['bg'][:nc], dt, win, cal.STAGE1_KEEP, _sd, seed + 1, nmc) if parent is not None else None

        def Jc(z):
            r = fun(z); return .5 * (r @ r - (z[6:9] @ Qc @ z[6:9] + z[3:6] @ Qp @ z[3:6] if parent is not None else z[3:6] @ Qc @ z[3:6]))
        sb = minimize(Jc, x, method='BFGS', options=dict(gtol=cfg['stage1_eiv_gtol'], maxiter=2000))
        info['eiv'] = dict(x_uncompensated=x.tolist(), Jc=float(sb.fun), message=str(sb.message), nit=int(sb.nit), sd=_sd, Qc=Qc.tolist(), Qp=None if Qp is None else Qp.tolist())
        x = sb.x
    bgc = np.zeros(3)                            # no Stage I gyroscope bias
    R0 = R.from_rotvec(x[:3]).as_matrix(); G = np.eye(3)
    if parent is None:
        R0, G = cal.gauge(R0, g, axis(x[-2:]) if rj else None, cfg.get('heading_axis_target') if rj else None)
        ell = x[3:6]; p0 = np.array(cfg['fixed_support']) - R0 @ ell; v0 = -R0 @ np.cross(gyr[0] - bgc, ell)
        result = dict(R0=R0, lever=ell, p0=p0, v0=v0, gauge_rotation=G, fit=info)
    else:
        lp = x[3:6]; ell = x[6:9]
        p0 = parent['p'][0] + Rp[0] @ lp - R0 @ ell
        v0 = parent['v'][0] + Rp[0] @ np.cross(parent['gyr'][0] - parent['bg'][0], lp) - R0 @ np.cross(gyr[0] - bgc, ell)
        result = dict(R0=R0, lever=ell, parent_lever=lp, p0=p0, v0=v0, fit=info)
    if rj: result['axis'] = axis(x[-2:])
    return result
