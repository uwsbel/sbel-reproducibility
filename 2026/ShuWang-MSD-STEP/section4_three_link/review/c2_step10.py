"""STEP with Stage I on [0, 10] s, both scenarios. Writes out_d/C2_STEP10.json."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import d_common as d

out = {}
# RELEASE: guard removed (a Stage I check on [0, 30] s against saved levers that are not part of this repository;
# it ran before, and independently of, the 10 s runs below, so its removal does not change their results)

for s in ('noise_only', 'noise_bias'):
    m, ref = d.data(s); t, G = d.target(ref)
    r = d.run_step(m, d.scfg('step_eiv.json', 10.0))
    np.savez_compressed(d.OUT / f'step10_{s}.npz', **{f'b{b}_{k}': v for b in (1, 2, 3) for k, v in r['step'][b].items()
                                                       if isinstance(v, np.ndarray)})
    res = dict(lever_pct=r['lever_pct'], levers=r['levers'], stage1_s=r['stage1_s'], excluded=r['excluded'],
               samples=r['samples'], state=d.state_rows(r['step'], t, m['time']))
    if s == 'noise_bias':
        bias = {}
        for w, mask in d.windows(m['time']).items():
            for b in (1, 2, 3):
                for key in ('ba', 'bg'):
                    e = r['step'][b][key][mask] - ref[f'{key}{b}']
                    bias[f'{w}|body{b}|{key}'] = dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                                                      truth_norm=float(np.linalg.norm(ref[f'{key}{b}'])))
        res['bias'] = bias
    out[s] = res
    print(s, 'levers %', ' '.join(f'{x:.3f}' for x in r['lever_pct']), 'Stage I s', round(r['stage1_s'], 1),
          'excluded', r['excluded'], flush=True)
    for w in ('t>=10', 'full'):
        a = np.array(res['state'][w])
        print('  ', w, 'e_p', ' '.join(f'{x:.4f}' for x in a[:, 0]), 'e_v', ' '.join(f'{x:.4f}' for x in a[:, 1]),
              'e_R', ' '.join(f'{x:.3f}' for x in a[:, 2]), flush=True)
(d.OUT / 'C2_STEP10.json').write_text(json.dumps(out, indent=1))
