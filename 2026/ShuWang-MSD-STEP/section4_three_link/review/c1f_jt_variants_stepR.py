"""Paper draw, both scenarios, nine c, all JT variants with the STEP constraint-residual standard deviations
(0.01 m, 0.05 m/s). Writes out_f/C1F_JT_VARIANTS_STEPR.json and out_f/jt/*.npz; refuses to overwrite."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d
import jt_stepR as jr

F = d.HERE / 'out_f'; JOUT = F / 'jt'
LEVELS = [0, 10, -10, 20, -20, 50, -50, 100, -100]


def job(args):
    s, c = args
    m, ref = d.data(s); t, G = d.target(ref); st = d.jt_states(ref, G); mask = m['time'] >= d.T_EVAL
    v = jr.variants(m, st, c, jr.RSD_STEP)
    out = {}
    for name, h in v.items():
        fin = bool(all(np.isfinite(np.asarray(h[k])).all() for k in jr.STATE_KEYS + jr.LEVER_KEYS))
        rec = dict(state=d.state_rows(h, t, m['time'], jt=True), lever_pct=d.jt_lever_pct(h), finite=fin)
        if 'b' in h:
            bias = {}
            for bi in (1, 2, 3):
                o = 6 * (bi - 1)
                for key, sl in (('ba', slice(o, o + 3)), ('bg', slice(o + 3, o + 6))):
                    e = h['b'][mask, sl] - ref[f'{key}{bi}']
                    bias[f'body{bi}|{key}'] = dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                                                   truth_norm=float(np.linalg.norm(ref[f'{key}{bi}'])))
            rec['bias'] = bias
        out[f'{s}|{name}|{c:+d}'] = rec
        keep = jr.STATE_KEYS + jr.LEVER_KEYS + (['b'] if 'b' in h else [])
        np.savez_compressed(JOUT / f'{s}_{name}_{c:+d}.npz', **{k: np.asarray(h[k]) for k in keep})
    return out


if __name__ == '__main__':
    # RELEASE: guard removed (out_f/REPRO_CHECK.json, a check against saved state histories that are not part of this repository)
    F.mkdir(exist_ok=True)  # RELEASE: creates out_f
    target = F / 'C1F_JT_VARIANTS_STEPR.json'
    if target.exists(): raise SystemExit('output exists; refusing to overwrite')
    JOUT.mkdir(exist_ok=False)
    res = dict(residual_sd=jr.RSD_STEP, results={})
    jobs = [(s, c) for s in ('noise_only', 'noise_bias') for c in LEVELS]
    with Pool(4) as pool:
        for out in pool.imap_unordered(job, jobs):
            res['results'].update(out); print(sorted(out)[0], 'done', flush=True)
    res['all_finite'] = bool(all(v['finite'] for v in res['results'].values()))
    target.write_text(json.dumps(res, indent=1))
    print('all finite:', res['all_finite'])
