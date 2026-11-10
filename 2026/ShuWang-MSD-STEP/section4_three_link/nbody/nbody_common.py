"""Shared helpers of the N-body chain runs. STEP is the code in ../step."""
import os, sys
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ[k] = '1'
HERE = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.dont_write_bytecode = True
import json, time
import numpy as np
from scipy.spatial.transform import Rotation as R
CODE = HERE.parent / 'step'; REVIEW = HERE.parent / 'review'  # RELEASE: paths changed for this repository
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'src')); sys.path.append(str(REVIEW))
import run_error_comparison as rec
import calibration, core, compiled_step

OUT = HERE / 'out'
SCEN = ('noise_only', 'noise_bias')
G_VEC = np.array([0., 0., -9.81]); T_EVAL = 10.0
COLS = ['position_rmse_m', 'velocity_rmse_m_s', 'attitude_rmse_deg']
ECFG = json.loads((CODE / 'configs' / 'measurement_errors.json').read_text())
STEP_P0 = (0.05, 0.1, 0.0872664626)          # as the JT-BHS runs of review/
RSD = (0.01, 0.05)
BA0, BG0, SBA, SBG = 0.02, 0.002, 1e-4, 1e-5
_sa, _sg, _dtq = 0.01, 0.001, 0.001; _I3 = np.eye(3)
QK = np.zeros((9, 9))                         # same expressions as review/jt_t1_diag.py
QK[:3, :3] = .25 * _sa**2 * _dtq**4 * _I3; QK[:3, 3:6] = .5 * _sa**2 * _dtq**3 * _I3; QK[3:6, :3] = .5 * _sa**2 * _dtq**3 * _I3
QK[3:6, 3:6] = _sa**2 * _dtq**2 * _I3; QK[6:9, 6:9] = _sg**2 * _dtq**2 * _I3


def pivot(N):
    return np.array([0., 0., 2. * N])


def step_cfg(N):
    c = json.loads((CODE / 'configs' / 'step_eiv.json').read_text(encoding='utf-8'))
    c.update(calibration_s=10.0, n_bodies=N, joint_types=['spherical'] * N, sg_windows=[11] * N, max_nfev=[2000] * N,
             fixed_support=[0, 0, 2 * N])
    return c


def clean_dir(N):
    return CODE / 'data' / 'clean' if N == 3 else HERE / 'data' / f'N{N:02d}' / 'clean'


def load_clean(N):
    with np.load(clean_dir(N) / 'measurements.npz') as z: m = {k: z[k] for k in z.files}
    with np.load(clean_dir(N) / 'reference.npz') as z: ref = {k: z[k] for k in z.files}
    return m, ref


def paper_seeds(N):
    return {b: 20261008 + b for b in range(1, N + 1)}


def mc_seeds(N, r):
    return {b: 20262000 + 10 * r + b for b in range(1, N + 1)}


def draw(clean, N, seeds):
    """generate_measurement_errors.generate_arrays for N bodies; same draw order per body."""
    out = {s: {k: v.copy() for k, v in clean.items()} for s in SCEN}; bias = {}
    for b in range(1, N + 1):
        rng = np.random.default_rng(seeds[b])
        ba = rng.uniform(*ECFG['acc_bias_component_bounds_m_s2'], 3); bg = rng.uniform(*ECFG['gyro_bias_component_bounds_rad_s'], 3)
        na = rng.normal(0, ECFG['acc_white_noise_sd_m_s2'], clean[f'acc{b}'].shape)
        ng = rng.normal(0, ECFG['gyro_white_noise_sd_rad_s'], clean[f'gyr{b}'].shape)
        bias.update({f'ba{b}': ba, f'bg{b}': bg})
        for prefix, bb, noise in (('acc', ba, na), ('gyr', bg, ng)):
            key = f'{prefix}{b}'
            out['noise_only'][key] = clean[key] + noise; out['noise_bias'][key] = clean[key] + noise + bb
    return out, bias


def target(N, ref):
    _, G = calibration.gauge(ref['R1'][0], G_VEC); t = {}; piv = pivot(N)
    for b in range(1, N + 1):
        h = rec.chart(dict(p=ref[f'p{b}'], v=ref[f'v{b}'], R=ref[f'R{b}']), G, piv)
        t.update({f'p{b}': h['p'], f'v{b}': h['v'], f'R{b}': h['R']})
    return t, G


def truth_levers(N):
    return np.array([[0., 0., 1.] if i % 2 == 0 else [0., 0., -1.] for i in range(2 * N - 1)])


def true_states(N, ref, G):
    piv = pivot(N)
    r = np.array([G @ (ref[f'p{b}'][0] - piv) + piv for b in range(1, N + 1)])
    v = np.array([G @ ref[f'v{b}'][0] for b in range(1, N + 1)])
    q = np.array([R.from_matrix(G @ ref[f'R{b}'][0]).as_quat()[[3, 0, 1, 2]] for b in range(1, N + 1)])
    return r, v, q


def run_step(m, cfg, N, stage2=True):
    """d_common.run_step for N bodies: Stage I on the first calibration_s seconds, Stage II on all samples."""
    dt = float(np.median(np.diff(m['time']))); nc = int(round(cfg['calibration_s'] / dt))
    calibration.STAGE1_KEEP = calibration.range_keep([m[f'gyr{b}'][:nc] for b in range(1, N + 1)],
                                                     cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])
    excluded = int((~calibration.STAGE1_KEEP).sum())
    cals, step, t1 = {}, {}, []
    for b in range(1, N + 1):
        tic = time.perf_counter(); cal = calibration.calibrate(b, m[f'acc{b}'][:nc], m[f'gyr{b}'][:nc], dt, cfg, step.get(b - 1))
        t1.append(time.perf_counter() - tic); cals[b] = cal
        if stage2 or b < N:
            step[b] = core.reconstruct(b, m[f'acc{b}'], m[f'gyr{b}'], dt, cal, cfg, step.get(b - 1))
    calibration.STAGE1_KEEP = None
    levers = [cals[1]['lever']]
    for b in range(2, N + 1): levers += [cals[b]['parent_lever'], cals[b]['lever']]
    levers = np.array([np.asarray(x, float) for x in levers]); gt = truth_levers(N)
    lever_pct = [float(100 * np.linalg.norm(x - g) / np.linalg.norm(g)) for x, g in zip(levers, gt)]
    return dict(cals=cals, step=step, lever_pct=lever_pct, levers=levers, stage1_s=t1, excluded=excluded, samples=nc)


def stage1_states(cals, N):
    r = np.array([np.asarray(cals[b]['p0'], float) for b in range(1, N + 1)])
    v = np.array([np.asarray(cals[b]['v0'], float) for b in range(1, N + 1)])
    q = np.array([R.from_matrix(np.asarray(cals[b]['R0'])).as_quat()[[3, 0, 1, 2]] for b in range(1, N + 1)])
    return r, v, q


def windows(time_):
    return {'t>=10': time_ >= T_EVAL, 'full': np.ones(len(time_), bool)}


def rows_step(step, t, time_, N):
    out = {}
    for w, mask in windows(time_).items():
        rows = []
        for b in range(1, N + 1):
            r = rec.metrics(step[b], t, b, mask, '', '', ''); rows.append([r[c] for c in COLS])
        out[w] = rows
    return out


def jt_body(h, b):
    return dict(p=h['r'][:, b - 1], v=h['v'][:, b - 1], R=R.from_quat(h['q'][:, b - 1][:, [1, 2, 3, 0]]).as_matrix())


def rows_jt(h, t, time_, N):
    out = {}
    for w, mask in windows(time_).items():
        rows = []
        for b in range(1, N + 1):
            r = rec.metrics(jt_body(h, b), t, b, mask, '', '', ''); rows.append([r[c] for c in COLS])
        out[w] = rows
    return out


def jt_lever_pct(h, N):
    gt = truth_levers(N)
    return [float(100 * np.linalg.norm(h['L'][-1, i] - gt[i]) / np.linalg.norm(gt[i])) for i in range(2 * N - 1)]


def heading_tilt(Rh, Rt, mask):
    """RMS heading (deg, signed series also returned) and tilt of the attitude error."""
    rv = R.from_matrix(Rh @ np.transpose(Rt, (0, 2, 1))).as_rotvec()
    h = np.rad2deg(rv[:, 2]); tl = np.rad2deg(np.linalg.norm(rv[:, :2], axis=1))
    return float(np.sqrt(np.mean(h[mask] ** 2))), float(np.sqrt(np.mean(tl[mask] ** 2))), h


def layout(N, est):
    nm = 9 * N; nl = 3 * (2 * N - 1) if est else 0
    return nm, nl, nm + nl + 6 * N


def initial_cov(N, est, lever_sd, motion_sd):
    """motion_sd = (position, velocity, attitude) SDs, or None for 1e-3 I, as jt_bias.build_PQ."""
    nm, nl, n = layout(N, est); P = np.zeros((n, n))
    if motion_sd is None:
        P[:nm, :nm] = np.eye(nm) * 1e-3
    else:
        for o in range(0, nm, 9):
            for j, sdv in enumerate(motion_sd): P[o + 3 * j:o + 3 * j + 3, o + 3 * j:o + 3 * j + 3] = np.eye(3) * sdv ** 2
    if est: P[nm:nm + nl, nm:nm + nl] = np.eye(nl) * lever_sd ** 2
    for o in range(nm + nl, n, 6):
        P[o:o + 3, o:o + 3] = np.eye(3) * BA0 ** 2; P[o + 3:o + 6, o + 3:o + 6] = np.eye(3) * BG0 ** 2
    return P


def pack(m, N):
    acc = np.stack([m[f'acc{b}'] for b in range(1, N + 1)], axis=1); gyr = np.stack([m[f'gyr{b}'] for b in range(1, N + 1)], axis=1)
    return np.ascontiguousarray(acc), np.ascontiguousarray(gyr)


def rdiag(N):
    return np.tile(np.r_[np.full(3, RSD[0] ** 2), np.full(3, RSD[1] ** 2)], N)
