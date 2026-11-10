"""Shared helpers of the review/ scripts: STEP from ../step; the JT core and JT settings via jt_t1_diag."""
import os
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).parent
CODE = HERE.parent / 'step'  # RELEASE: path changed for this repository
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'src'))
import run_error_comparison as rec
import calibration, core
import jt_t1_diag as jtd
OUT = HERE / 'out_d'; OUT.mkdir(exist_ok=True)
LABELS = ['L1p', 'L1d', 'L2p', 'L2d', 'L3p']
TRUTH = np.array([[0., 0., 1.], [0., 0., -1.], [0., 0., 1.], [0., 0., -1.], [0., 0., 1.]])
COLS = ['position_rmse_m', 'velocity_rmse_m_s', 'attitude_rmse_deg']
PIVOT = np.array([0., 0., 6.]); G_VEC = np.array([0., 0., -9.81])
T_EVAL = 10.0


def scfg(name, calibration_s=10.0):
    c = json.loads((CODE / 'configs' / name).read_text(encoding='utf-8'))
    c['calibration_s'] = calibration_s
    return c


def data(s):
    with np.load(CODE / 'data' / s / 'measurements.npz') as z: m = {k: z[k] for k in z.files}
    with np.load(CODE / 'data' / s / 'reference.npz') as z: ref = {k: z[k] for k in z.files}
    return m, ref


def target(ref):
    _, G = calibration.gauge(ref['R1'][0], G_VEC); t = {}
    for b in (1, 2, 3):
        h = rec.chart(dict(p=ref[f'p{b}'], v=ref[f'v{b}'], R=ref[f'R{b}']), G, PIVOT)
        t.update({f'p{b}': h['p'], f'v{b}': h['v'], f'R{b}': h['R']})
    return t, G


def jt_states(ref, G):
    st = {}
    for b in (1, 2, 3):
        st.update({f'r{b}': G @ (ref[f'p{b}'][0] - PIVOT) + PIVOT, f'v{b}': G @ ref[f'v{b}'][0],
                   f'q{b}': R.from_matrix(G @ ref[f'R{b}'][0]).as_quat()[[3, 0, 1, 2]]})
    return st


def run_step(m, cfg, stage2=True):
    """STEP as in run_error_comparison.run (Stage I on the first calibration_s seconds, Stage II on all samples)."""
    dt = float(np.median(np.diff(m['time']))); nc = int(round(cfg['calibration_s'] / dt))
    calibration.STAGE1_KEEP = None; excluded = 0
    if 'stage1_gyro_range_rad_s' in cfg:
        calibration.STAGE1_KEEP = calibration.range_keep([m[f'gyr{b}'][:nc] for b in (1, 2, 3)],
                                                         cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])
        excluded = int((~calibration.STAGE1_KEEP).sum())
    cals, step, t1 = {}, {}, 0.0
    for b in (1, 2, 3):
        tic = time.perf_counter(); cal = calibration.calibrate(b, m[f'acc{b}'][:nc], m[f'gyr{b}'][:nc], dt, cfg, step.get(b - 1))
        t1 += time.perf_counter() - tic; cals[b] = cal
        if stage2 or b < 3:
            step[b] = core.reconstruct(b, m[f'acc{b}'], m[f'gyr{b}'], dt, cal, cfg, step.get(b - 1))
    calibration.STAGE1_KEEP = None
    levers = [cals[1]['lever'], cals[2]['parent_lever'], cals[2]['lever'], cals[3]['parent_lever'], cals[3]['lever']]
    lever_pct = [float(100 * np.linalg.norm(np.asarray(x) - gt) / np.linalg.norm(gt)) for x, gt in zip(levers, TRUTH)]
    return dict(cals=cals, step=step, lever_pct=lever_pct, levers=[np.asarray(x).tolist() for x in levers],
                stage1_s=t1, excluded=excluded, samples=nc)


def windows(time_):
    return {'t>=10': time_ >= T_EVAL, 'full': np.ones(len(time_), bool)}


def state_rows(hist, t, time_, jt=False):
    out = {}
    for w, mask in windows(time_).items():
        rows = []
        for b in (1, 2, 3):
            if jt:
                hb = dict(p=hist[f'r{b}'], v=hist[f'v{b}'], R=R.from_quat(hist[f'q{b}'][:, [1, 2, 3, 0]]).as_matrix())
            else:
                hb = hist[b]
            r = rec.metrics(hb, t, b, mask, '', '', ''); rows.append([r[c] for c in COLS])
        out[w] = rows
    return out


def jt_lever_pct(h):
    return [float(100 * np.linalg.norm(h[k][-1] - gt) / np.linalg.norm(gt)) for k, gt in zip(LABELS, TRUTH)]


def P0(lever_sd):
    P = np.eye(42) * 1e-3
    P[27:42, 27:42] = np.eye(15) * lever_sd ** 2
    return P


def perturbed(c):
    signs = np.array(jtd.cfg['perturbation_signs'])
    return dict(zip(LABELS, TRUTH + c / 100 * signs))
