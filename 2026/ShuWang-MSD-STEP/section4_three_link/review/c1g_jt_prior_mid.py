"""JT with an intermediate lever prior SD of 0.2 m per component, otherwise the
JT of Table 7 (STEP residual SDs, constant levers, no bias states, no OC, true initial states). Paper draw, both
scenarios, nine c. Writes out_g/; refuses to overwrite."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
import jt_stepR as jr

G_OUT = d.HERE / 'out_g'; JOUT = G_OUT / 'jt'
LEVELS = [0, 10, -10, 20, -20, 50, -50, 100, -100]
SD_MID = 0.2


def job(args):
    s, c = args
    m, ref = d.data(s); t, G = d.target(ref); st = d.jt_states(ref, G)
    h = jr.run_jt(m, st, d.perturbed(c), d.P0(SD_MID), jr.RSD_STEP)
    fin = bool(all(np.isfinite(np.asarray(h[k])).all() for k in jr.STATE_KEYS + jr.LEVER_KEYS))
    np.savez_compressed(JOUT / f'{s}_mid02_{c:+d}.npz', **{k: np.asarray(h[k]) for k in jr.STATE_KEYS + jr.LEVER_KEYS})
    return {f'{s}|mid02|{c:+d}': dict(state=d.state_rows(h, t, m['time'], jt=True), lever_pct=d.jt_lever_pct(h), finite=fin)}


if __name__ == '__main__':
    target = G_OUT / 'C1G_JT_PRIOR_MID.json'
    if G_OUT.exists(): raise SystemExit('out_g exists; refusing to overwrite')
    G_OUT.mkdir(); JOUT.mkdir()
    res = dict(lever_sd=SD_MID, residual_sd=jr.RSD_STEP, results={})
    jobs = [(s, c) for s in ('noise_only', 'noise_bias') for c in LEVELS]
    with Pool(4) as pool:
        for out in pool.imap_unordered(job, jobs):
            res['results'].update(out); print(sorted(out)[0], 'done', flush=True)
    res['all_finite'] = bool(all(v['finite'] for v in res['results'].values()))
    target.write_text(json.dumps(res, indent=1))
    print('all finite:', res['all_finite'])
