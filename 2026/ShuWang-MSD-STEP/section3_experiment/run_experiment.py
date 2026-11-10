"""Experimental double pendulum (Section 3.3, Table 4).

Stage I uses the first 10 s of the record, Stage II the whole record. The results are scored
against the optical reference. This is the procedure of the run behind Table 4, with paths made
relative to this folder.

Usage: python run_experiment.py --tag <new tag> [--config experimental_eiv_p.json]
Writes runs/<tag>/ (never overwrites) and runs/<tag>/TABLE4.json.
"""
import os
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[k] = '1'
import sys
import io
import json
import argparse
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'src'))
import calibration
import core
import evaluate_experiment as ev
ev.ROOT = HERE


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else bool(v) if isinstance(v, np.bool_) else str(v)))


def run(tag, config, calibration_s=10.0):
    cfg = json.loads((HERE / 'configs' / config).read_text())
    cfg['calibration_s'] = calibration_s
    out = HERE / 'runs' / tag
    out.mkdir(parents=True, exist_ok=False)
    with np.load(HERE / 'data' / 'experimental_measurements.npz') as z:
        m = {k: z[k] for k in z.files}
    dt = float(np.median(np.diff(m['time'])))
    nc = int(round(cfg['calibration_s'] / dt))
    calibration.STAGE1_KEEP = calibration.range_keep([m[f'gyr{b}'][:nc] for b in (1, 2)], cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])
    excluded = int((~calibration.STAGE1_KEEP).sum())
    cals, hists = {}, {}
    for b in (1, 2):
        parent = hists.get(b - 1)
        cal = calibration.calibrate(b, m[f'acc{b}'][:nc], m[f'gyr{b}'][:nc], dt, cfg, parent, cals[b - 1].get('axis') if b > 1 else None)
        save(out / f'body{b}_stageI.json', cal)
        hist = core.reconstruct(b, m[f'acc{b}'], m[f'gyr{b}'], dt, cal, cfg, parent, cals[1].get('axis') if b > 1 else None)
        np.savez_compressed(out / f'body{b}_states.npz', time=m['time'], **{k: v for k, v in hist.items() if isinstance(v, np.ndarray)})
        cals[b] = cal
        hists[b] = hist
    calibration.STAGE1_KEEP = None
    with contextlib.redirect_stdout(io.StringIO()):
        ev.main(tag, 'notebook')
    met = pd.read_csv(out / 'reference_notebook' / 'metrics.csv')
    geo = json.loads((out / 'reference_notebook' / 'geometry.json').read_text())
    st = json.loads((HERE / 'configs' / 'experimental_reference.json').read_text())
    df = pd.read_csv(HERE / 'data' / 'experimental_optical_angles.csv')
    tv = df.Frame.to_numpy() / st['fps']
    ang = np.unwrap(np.deg2rad(df[['AngleBar1', 'AngleBar2']].to_numpy()), axis=0)
    ang -= ang[0]
    q = m['time'] + st['offset_video_minus_imu_s']
    mask = (q >= tv[0] - 1e-12) & (q <= tv[-1] + 1e-12)
    k0 = int(np.argmax(mask))
    th = {}
    for b in (1, 2):
        Rref = R.from_euler('y', CubicSpline(tv, ang)(q[k0])[b - 1]).as_matrix()
        th[b] = dict(theta0=float(np.rad2deg(R.from_matrix(Rref.T @ np.asarray(cals[b]['R0'])).magnitude())),
                     dn=float(np.rad2deg(np.arccos(min(1.0, abs(np.asarray(cals[b]['axis']) @ np.array([0., 1., 0.])))))))
    l1a, l1b, l2b = (np.asarray(cals[1]['lever']), np.asarray(cals[2]['parent_lever']), np.asarray(cals[2]['lever']))
    jd = float(1000 * np.hypot(l1a[0] - l1b[0], l1a[2] - l1b[2]))
    lev_err = {g['vector'] + str(g['body_stage']): g['in_plane_error_percent'] for g in geo}
    res = dict(config=config, calibration_s=calibration_s, samples=nc, excluded=excluded,
               levers_mm={'l1A': (1000 * l1a).tolist(), 'l1B': (1000 * l1b).tolist(), 'l2B': (1000 * l2b).tolist()},
               lever_error_pct=lev_err, joint_distance_mm=jd, theta0_dn=th,
               states=met[['body', 'position_rmse_m', 'velocity_rmse_m_s', 'attitude_rmse_deg']].to_dict('records'),
               converged=bool(all(c['fit']['converged'] for c in cals.values())))
    save(out / 'TABLE4.json', res)
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--config', default='experimental_eiv_p.json')
    a = p.parse_args()
    run(a.tag, a.config)
