"""Monte Carlo over 20 noise/bias draws (seeds 20262000 + 10 r + b), both scenarios: STEP, Stage I on [0, 10] s.
Each (realization, scenario) job writes out_d/mc/r{r:02d}_{scenario}.json; finished jobs are skipped on restart."""
import sys, json, copy
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
sys.path.insert(0, str(d.CODE))
import generate_measurement_errors as gme

MC = d.OUT / 'mc'; MC.mkdir(exist_ok=True)
ECFG = json.loads((d.CODE / 'configs' / 'measurement_errors.json').read_text())


def clean():
    with np.load(d.CODE / 'data' / 'clean' / 'measurements.npz') as z: return {k: z[k] for k in z.files}


def realization(r):
    cfg = copy.deepcopy(ECFG)
    cfg['body_seeds'] = {str(b): 20262000 + 10 * r + b for b in (1, 2, 3)}
    return gme.generate_arrays(clean(), cfg)[0]


def job(args):
    r, s = args
    path = MC / f'r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped'
    m = realization(r)[s]
    _, ref = d.data('clean'); t, G = d.target(ref)
    step = d.run_step(m, d.scfg('step_eiv.json', 10.0))
    res = dict(realization=r, scenario=s, seeds={str(b): 20262000 + 10 * r + b for b in (1, 2, 3)},
               STEP=dict(lever_pct=step['lever_pct'], state=d.state_rows(step['step'], t, m['time']), excluded=step['excluded']))
    path.write_text(json.dumps(res, indent=1))
    return path.name, 'done'


if __name__ == '__main__':
    # check: the paper seeds regenerate the saved measurement files exactly
    sc = gme.generate_arrays(clean(), ECFG)[0]
    ok = True
    for s in ('noise_only', 'noise_bias'):
        m0, _ = d.data(s)
        ok &= all(np.array_equal(sc[s][k], m0[k]) for k in m0)
    _, rc = d.data('clean'); _, rn = d.data('noise_bias')
    ok &= all(np.array_equal(rc[k], rn[k]) for k in ('p1', 'v1', 'R1', 'p2', 'v2', 'R2', 'p3', 'v3', 'R3'))
    print('paper seeds reproduce saved measurements, reference motion shared:', ok, flush=True)
    if not ok: raise SystemExit('generation check failed')
    jobs = [(r, s) for r in range(1, 21) for s in ('noise_only', 'noise_bias')]
    with Pool(4) as pool:
        for name, status in pool.imap_unordered(job, jobs):
            print(name, status, flush=True)
