"""Table 5 (tab:30s_param_results): relative lever-vector errors, single draw.

JT rows: the range over the nine initial perturbations c of the lever errors at the end of the record
(review/out_f/C1F_JT_VARIANTS_STEPR.json, key W1 = JT with lever SD 1 m and the STEP residual SDs).
STEP rows: Stage I on the first 10 s (review/out_d/C2_STEP10.json).

Usage: python make_table5.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

LEV = [0, 10, -10, 20, -20, 50, -50, 100, -100]
VEC = ['l1A', 'l1B', 'l2B', 'l2C', 'l3C']
SC = [('noise_only', 'bias-free'), ('noise_bias', 'biased')]


def cells(b):
    jt = tc.jload(b / 'review' / 'out_f' / 'C1F_JT_VARIANTS_STEPR.json')['results']
    c2 = tc.jload(b / 'review' / 'out_d' / 'C2_STEP10.json')
    v = {}
    for s, name in SC:
        lj = np.array([jt[f'{s}|W1|{c:+d}']['lever_pct'] for c in LEV])
        for i, vec in enumerate(VEC):
            v[f'{name}|JT|{vec}|min'] = float(lj[:, i].min())
            v[f'{name}|JT|{vec}|max'] = float(lj[:, i].max())
            v[f'{name}|STEP|{vec}'] = float(c2[s]['lever_pct'][i])
        v[f'{name}|STEP excluded samples'] = c2[s]['excluded']
    return v


def show(v):
    print('Table 5: relative lever-vector error e_l (%)')
    print(f"{'':22s}" + ''.join(f'{x:>14s}' for x in VEC))
    for _, name in SC:
        print(f'({name})')
        d = 3 if name == 'bias-free' else 2

        def rng(vec):
            lo, hi = round(v[f'{name}|JT|{vec}|min'], d), round(v[f'{name}|JT|{vec}|max'], d)
            return f'{lo:.{d}f}' if lo == hi else f'{lo:.{d}f}--{hi:.{d}f}'
        print(f"{'JT, all nine c':22s}" + ''.join(f'{rng(x):>14s}' for x in VEC))
        print(f"{'STEP':22s}" + ''.join(f"{v[f'{name}|STEP|{x}']:>14.{d}f}" for x in VEC))
        print(f"  STEP Stage I excluded {v[f'{name}|STEP excluded samples']} of 10000 samples")


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v = cells(tc.base(a.source))
    show(v)
    ok = tc.compare('tab:30s_param_results', v, a)
    raise SystemExit(0 if ok is not False else 1)
