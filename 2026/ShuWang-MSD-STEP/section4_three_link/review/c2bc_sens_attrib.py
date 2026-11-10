"""Stage I on [0, 10] s: sample exclusion and sigma_g sensitivity (Table 9, three-link columns) and the error sources of
Bodies 2 and 3 (Table 10). Writes out_d/C2bc_SENS_ATTRIB.json."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d

SENS = {'none': 'step.json', '0': 'step_rule.json', '0.5': 'step_eiv_x0p5.json', '0.8': 'step_eiv_x0p8.json',
        '1.2': 'step_eiv_x1p2.json', '2': 'step_eiv_x2.json'}
LEV = {1: ([0, 0, 1.], None), 2: ([0, 0, 1.], [0, 0, -1.]), 3: ([0, 0, 1.], [0, 0, -1.])}


def sens(args):
    s, key = args
    m, ref = d.data(s); t, _ = d.target(ref)
    r = d.run_step(m, d.scfg(SENS[key], 10.0))
    return f'sens|{s}|{key}', dict(lever_pct=r['lever_pct'], state=d.state_rows(r['step'], t, m['time']), excluded=r['excluded'])


def attrib(args):
    b, mode = args
    m, ref = d.data('noise_bias'); cfg = d.scfg('step_eiv.json', 10.0)
    dt = float(np.median(np.diff(m['time']))); nc = int(round(cfg['calibration_s'] / dt)); N = len(m['time'])
    d.calibration.STAGE1_KEEP = d.calibration.range_keep([m[f'gyr{i}'][:nc] for i in (1, 2, 3)],
                                                         cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])
    t, _ = d.target(ref)
    tb = {i: dict(p=t[f'p{i}'], v=t[f'v{i}'], R=t[f'R{i}']) for i in (1, 2, 3)}
    parent = dict(R=tb[b - 1]['R'], p=tb[b - 1]['p'], v=tb[b - 1]['v'], ba=np.tile(ref[f'ba{b-1}'], (N, 1)),
                  bg=np.tile(ref[f'bg{b-1}'], (N, 1)), acc=m[f'acc{b-1}'], gyr=m[f'gyr{b-1}'])
    if mode == 'truth_parent':
        cal = d.calibration.calibrate(b, m[f'acc{b}'][:nc], m[f'gyr{b}'][:nc], dt, cfg, parent)
    else:
        lv, lp = LEV[b]; cal = dict(R0=tb[b]['R'][0], p0=tb[b]['p'][0], v0=tb[b]['v'][0], lever=np.array(lv), parent_lever=np.array(lp))
    h = d.core.reconstruct(b, m[f'acc{b}'], m[f'gyr{b}'], dt, cal, cfg, parent)
    d.calibration.STAGE1_KEEP = None
    rows = {}
    for w, mask in d.windows(m['time']).items():
        r = d.rec.metrics(h, t, b, mask, mode, '', ''); rows[w] = [r[c] for c in d.COLS]
    return f'attrib|body{b}|{mode}', dict(state=rows,
                                          e_lever_child_pct=float(100 * np.linalg.norm(np.asarray(cal['lever']) - LEV[b][0])),
                                          e_lever_parent_pct=float(100 * np.linalg.norm(np.asarray(cal['parent_lever']) - LEV[b][1])))


def job(a):
    return attrib(a[1:]) if a[0] == 'attrib' else sens(a[1:])


if __name__ == '__main__':
    jobs = [('sens', s, k) for s in ('noise_only', 'noise_bias') for k in SENS] + \
           [('attrib', b, mode) for b in (2, 3) for mode in ('truth_parent', 'truth_all')]
    out = {}
    with Pool(4) as pool:
        for key, val in pool.imap_unordered(job, jobs):
            out[key] = val; print(key, 'done', flush=True)
    (d.OUT / 'C2bc_SENS_ATTRIB.json').write_text(json.dumps(out, indent=1))
