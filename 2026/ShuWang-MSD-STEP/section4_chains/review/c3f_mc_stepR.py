"""The 20 draws of c3_mc.py, both scenarios, c in {0, +-10, +-50}: main JT,
two-pass JT and JT+bias with the STEP constraint-residual standard deviations. STEP and the coupled filter are not rerun.
Each (realization, scenario) job writes out_f/mc/r{rr}_{scenario}.json; an existing file is skipped, never overwritten."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
import jt_stepR as jr
import c3_mc

MCF = d.HERE / 'out_f' / 'mc'
LEVELS = [0, 10, -10, 50, -50]


def job(args):
    r, s = args
    path = MCF / f'r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped'
    m = c3_mc.realization(r)[s]
    _, ref = d.data('clean'); t, G = d.target(ref); st = d.jt_states(ref, G)
    res = dict(realization=r, scenario=s, residual_sd=jr.RSD_STEP, seeds={str(b): 20262000 + 10 * r + b for b in (1, 2, 3)})

    def rec(name, h):
        res[name] = dict(lever_pct=d.jt_lever_pct(h), state=d.state_rows(h, t, m['time'], jt=True),
                         finite=bool(all(np.isfinite(np.asarray(h[k])).all() for k in jr.STATE_KEYS + jr.LEVER_KEYS)))
    for c in LEVELS:
        h = jr.run_jt(m, st, d.perturbed(c), d.P0(1.0), jr.RSD_STEP); rec(f'JT|{c:+d}', h)
        rec(f'JT-2p|{c:+d}', jr.run_jt(m, st, {k: h[k][-1].copy() for k in jr.LEVER_KEYS}, d.P0(0.01), jr.RSD_STEP))
        rec(f'JT-bias|{c:+d}', jr.run_jt_bias(m, st, d.perturbed(c), 1.0, jr.RSD_STEP))
    path.write_text(json.dumps(res, indent=1))
    return path.name, 'done'


if __name__ == '__main__':
    # RELEASE: guard removed (out_f/REPRO_CHECK.json, a check against saved state histories that are not part of this repository)
    MCF.mkdir(parents=True, exist_ok=True)  # RELEASE: parents=True also creates out_f
    jobs = [(r, s) for r in range(1, 21) for s in ('noise_only', 'noise_bias')]
    with Pool(4) as pool:
        for name, status in pool.imap_unordered(job, jobs):
            print(name, status, flush=True)
