"""Table 10 (tab:chain_attribution): sources of the STEP errors of Bodies 2 and 3 with biased measurements.
Inputs of the child calculation are replaced by ground-truth values, and Stages I and II of the child are repeated.
RMSEs over t in [10, 30] s. Also the lever errors of the child calculation quoted in the text.

Sources: review/out_d/C2bc_SENS_ATTRIB.json (keys attrib|body<b>|truth_parent and truth_all) and
review/out_d/C2_STEP10.json (the rows without replacement).

Usage: python make_table10.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

Q = ['e_p', 'e_v', 'e_R']
MODES = [('no', 'no', None), ('yes', 'no', 'truth_parent'), ('yes', 'yes', 'truth_all')]


def cells(b):
    x = tc.jload(b / 'review' / 'out_d' / 'C2bc_SENS_ATTRIB.json')
    c2 = tc.jload(b / 'review' / 'out_d' / 'C2_STEP10.json')
    S = np.array(c2['noise_bias']['state']['t>=10'])
    v = {}
    for body in (2, 3):
        for gp, gg, mode in MODES:
            a = S[body - 1] if mode is None else np.array(x[f'attrib|body{body}|{mode}']['state']['t>=10'])
            for j, q in enumerate(Q):
                v[f'B{body}|ground-truth parent={gp}|ground-truth geometry and initial state={gg}|{q}'] = float(a[j])
        r = x[f'attrib|body{body}|truth_parent']
        v[f'text|B{body}|ground-truth parent: child lever error (%)'] = r['e_lever_child_pct']
        v[f'text|B{body}|ground-truth parent: parent-joint lever error (%)'] = r['e_lever_parent_pct']
    lp = c2['noise_bias']['lever_pct']
    v['text|STEP child-calculation lever errors (%) min'] = float(min(lp[1:]))
    v['text|STEP child-calculation lever errors (%) max'] = float(max(lp[1:]))
    return v


def show(v):
    print('Table 10: biased measurements, RMSEs over t in [10, 30] s')
    for body in (2, 3):
        for gp, gg, _ in MODES:
            k = f'B{body}|ground-truth parent={gp}|ground-truth geometry and initial state={gg}|'
            print(f"  Body {body}  parent {gp:3s} geometry {gg:3s}  e_p {v[k + 'e_p']:.4f}  e_v {v[k + 'e_v']:.4f}  e_R {v[k + 'e_R']:.3f}")
    for k, x in v.items():
        if k.startswith('text|'):
            print(f'  {k[5:]}: {x:.2f}')


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v = cells(tc.base(a.source))
    show(v)
    ok = tc.compare('tab:chain_attribution', v, a)
    raise SystemExit(0 if ok is not False else 1)
