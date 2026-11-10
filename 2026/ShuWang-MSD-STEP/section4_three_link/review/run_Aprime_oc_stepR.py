"""A' + OC with the STEP residual SDs. Writes out_e/APRIME_OC_STEPR.{json,txt}."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
import jt_bias_oc as jo

OUTE = d.HERE / 'out_e'
STEP_P0 = (0.05, 0.1, 0.0872664626)
cs = d.scfg('step_eiv.json', 10.0)['constraint_sd']; STEP_R = (cs['point_p'], cs['point_v'])
c2 = json.loads((d.OUT / 'C2_STEP10.json').read_text())
# RELEASE: guard removed (out_e/APRIME_OC.json, a run with residual SD 0.005/0.005 that is not part of this repository)
OUTE.mkdir(exist_ok=True)  # RELEASE: creates out_e
res, L = {'STEP_residual_sd': STEP_R}, []


def heading(h, t, mask):
    hs, ts = [], []
    for b in (1, 2, 3):
        rv = R.from_matrix(R.from_quat(h[f'q{b}'][:, [1, 2, 3, 0]]).as_matrix() @ np.transpose(t[f'R{b}'], (0, 2, 1))).as_rotvec()
        hs.append(float(np.rad2deg(np.sqrt(np.mean(rv[mask, 2] ** 2))))); ts.append(float(np.rad2deg(np.sqrt(np.mean(np.sum(rv[mask, :2] ** 2, axis=1))))))
    return hs, ts


for s in ('noise_only', 'noise_bias'):
    m, ref = d.data(s); t, G = d.target(ref); w = m['time'] >= d.T_EVAL
    r = d.run_step(m, d.scfg('step_eiv.json', 10.0), stage2=False)
    st = {}
    for b in (1, 2, 3):
        cal = r['cals'][b]
        st.update({f'r{b}': np.asarray(cal['p0'], float), f'v{b}': np.asarray(cal['v0'], float),
                   f'q{b}': R.from_matrix(np.asarray(cal['R0'])).as_quat()[[3, 0, 1, 2]]})
    lev = dict(zip(d.LABELS, [np.array(x) for x in r['levers']]))
    h = jo.run(m, st, lev, 0.0, d.jtd.QK, STEP_R, d.G_VEC, d.PIVOT, oc=True, motion_sd=STEP_P0)
    sr = d.state_rows(h, t, m['time'], jt=True); hs, ts = heading(h, t, w)
    a = np.array(sr['t>=10']); stp = np.array(c2[s]['state']['t>=10'])  # RELEASE: guard removed ('old' came from out_e/APRIME_OC.json)
    res[s] = dict(finite=bool(all(np.isfinite(h[k]).all() for k in h)),
                  t10=a.tolist(), full=np.array(sr['full']).tolist(), heading=hs, tilt=ts,
                  step_smaller=int((stp < a).sum()), ratio_min=float((a / stp).min()), ratio_max=float((a / stp).max()))
    L.append(f'== {s} (finite {res[s]["finite"]})')
    for name, x in (('STEP', stp), (f"A'+OC, STEP residual SD {STEP_R[0]}/{STEP_R[1]}", a)):
        L.append(f'  {name:40s} e_p ' + '/'.join(f'{v:.4f}' for v in x[:, 0]) + '  e_v ' + '/'.join(f'{v:.4f}' for v in x[:, 1])
                 + '  e_R ' + '/'.join(f'{v:.3f}' for v in x[:, 2])
                 + ('' if name == 'STEP' else f'  | STEP smaller {int((stp < x).sum())}/9, JT/STEP {np.min(x / stp):.2f}-{np.max(x / stp):.2f}'))
    L.append('  new run heading ' + '/'.join(f'{v:.3f}' for v in hs) + ' tilt ' + '/'.join(f'{v:.3f}' for v in ts))
(OUTE / 'APRIME_OC_STEPR.json').write_text(json.dumps(res, indent=1))
(OUTE / 'APRIME_OC_STEPR.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
