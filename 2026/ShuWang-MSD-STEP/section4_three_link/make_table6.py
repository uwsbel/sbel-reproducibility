"""Table 6 (tab:30s_tracking_results): state RMSEs over t in [10, 30] s, single draw; the note gives the STEP RMSEs
over the complete record. Also the Body 3 bias-estimate RMSEs quoted in the text after Table 5.

Sources: review/out_f/C1F_JT_VARIANTS_STEPR.json (JT = key W1, nine c) and review/out_d/C2_STEP10.json (STEP).

Usage: python make_table6.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

LEV = [0, 10, -10, 20, -20, 50, -50, 100, -100]
SC = [('noise_only', 'bias-free'), ('noise_bias', 'biased')]
Q = ['e_p', 'e_v', 'e_R']


def cs(c):
    return '0' if c == 0 else f'{c:+d}'


def cells(b):
    jt = tc.jload(b / 'review' / 'out_f' / 'C1F_JT_VARIANTS_STEPR.json')['results']
    c2 = tc.jload(b / 'review' / 'out_d' / 'C2_STEP10.json')
    v = {}
    for s, name in SC:
        for c in LEV:
            a = np.array(jt[f'{s}|W1|{c:+d}']['state']['t>=10'])
            for j, q in enumerate(Q):
                for bb in range(3):
                    v[f'{name}|JT|c={cs(c)}|{q}|B{bb + 1}'] = float(a[bb, j])
        for w, tag in (('t>=10', 'STEP'), ('full', 'STEP full record')):
            a = np.array(c2[s]['state'][w])
            for j, q in enumerate(Q):
                for bb in range(3):
                    v[f'{name}|{tag}|{q}|B{bb + 1}'] = float(a[bb, j])
    for key in ('ba', 'bg'):
        x = c2['noise_bias']['bias'][f't>=10|body3|{key}']
        v[f'text|Body 3 {key} RMSE'] = x['rmse']
        v[f'text|Body 3 {key} true magnitude'] = x['truth_norm']
    return v


def show(v):
    print('Table 6: RMSEs over t in [10, 30] s (e_p m, e_v m/s, e_R deg; Bodies 1/2/3)')
    for _, name in SC:
        print(f'({name})')
        rows = [(f'JT  c={cs(c)}', f'JT|c={cs(c)}') for c in LEV] + [('STEP', 'STEP'), ('STEP, full record (note)', 'STEP full record')]
        for lab, key in rows:
            p = ' '.join(f"{v[f'{name}|{key}|e_p|B{i}']:.4f}" for i in (1, 2, 3))
            q = ' '.join(f"{v[f'{name}|{key}|e_v|B{i}']:.4f}" for i in (1, 2, 3))
            r = ' '.join(f"{v[f'{name}|{key}|e_R|B{i}']:.2f}" for i in (1, 2, 3))
            print(f'  {lab:26s} {p} | {q} | {r}')
    print(f"Text: Body 3 bias RMSE {v['text|Body 3 ba RMSE']:.4f} m/s^2 and {v['text|Body 3 bg RMSE']:.4f} rad/s; "
          f"true magnitudes {v['text|Body 3 ba true magnitude']:.3f} m/s^2 and {v['text|Body 3 bg true magnitude']:.4f} rad/s")


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v = cells(tc.base(a.source))
    show(v)
    ok = tc.compare('tab:30s_tracking_results', v, a)
    raise SystemExit(0 if ok is not False else 1)
