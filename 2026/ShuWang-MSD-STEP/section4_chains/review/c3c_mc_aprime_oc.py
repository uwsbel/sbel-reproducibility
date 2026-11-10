"""Monte Carlo of A' + OC with two residual settings.
Each job writes out_e/mc_aprime_oc/r{r:02d}_{scenario}.json; finished jobs are skipped on restart."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
import jt_bias_oc as jo
import c3_mc

MCO = d.HERE / 'out_e' / 'mc_aprime_oc'; MCO.mkdir(parents=True, exist_ok=True)
STEP_P0 = (0.05, 0.1, 0.0872664626)
SETTINGS = {'R005': (0.005, 0.005), 'Rstep': (0.01, 0.05)}


def heading(h, t, mask):
    hs, ts = [], []
    for b in (1, 2, 3):
        rv = R.from_matrix(R.from_quat(h[f'q{b}'][:, [1, 2, 3, 0]]).as_matrix() @ np.transpose(t[f'R{b}'], (0, 2, 1))).as_rotvec()
        hs.append(float(np.rad2deg(np.sqrt(np.mean(rv[mask, 2] ** 2))))); ts.append(float(np.rad2deg(np.sqrt(np.mean(np.sum(rv[mask, :2] ** 2, axis=1))))))
    return hs, ts


def job(args):
    r, s = args
    path = MCO / f'r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped'
    m = c3_mc.realization(r)[s]
    _, ref = d.data('clean'); t, G = d.target(ref); w = m['time'] >= d.T_EVAL
    step = d.run_step(m, d.scfg('step_eiv.json', 10.0), stage2=False)
    saved = json.loads((c3_mc.MC / f'r{r:02d}_{s}.json').read_text())['STEP']['lever_pct']
    if not np.array_equal(np.array(step['lever_pct']), np.array(saved)):
        return path.name, 'STEP Stage I does not reproduce the levers of c3_mc.py'
    st = {}
    for b in (1, 2, 3):
        cal = step['cals'][b]
        st.update({f'r{b}': np.asarray(cal['p0'], float), f'v{b}': np.asarray(cal['v0'], float),
                   f'q{b}': R.from_matrix(np.asarray(cal['R0'])).as_quat()[[3, 0, 1, 2]]})
    lev = dict(zip(d.LABELS, [np.array(x) for x in step['levers']]))
    res = dict(realization=r, scenario=s)
    for name, rs in SETTINGS.items():
        h = jo.run(m, st, lev, 0.0, d.jtd.QK, rs, d.G_VEC, d.PIVOT, oc=True, motion_sd=STEP_P0)
        hs, ts = heading(h, t, w)
        res[name] = dict(state=d.state_rows(h, t, m['time'], jt=True), heading=hs, tilt=ts,
                         finite=bool(all(np.isfinite(h[k]).all() for k in h)))
    path.write_text(json.dumps(res, indent=1))
    return path.name, 'done'


if __name__ == '__main__':
    jobs = [(r, s) for r in range(1, 21) for s in ('noise_only', 'noise_bias')]
    with Pool(4) as pool:
        for name, status in pool.imap_unordered(job, jobs):
            print(name, status, flush=True)
