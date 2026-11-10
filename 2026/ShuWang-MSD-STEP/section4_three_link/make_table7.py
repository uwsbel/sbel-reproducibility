"""Table 7 (tab:jt_variants): largest state RMSEs of the JT variants and STEP, single draw, t in [10, 30] s.

Rows and sources (internal names in brackets):
  JT [W1], JT-W [W1-RW], JT-H [W1-OC], JT-B [JT-bias], JT-BH [JT-bias-OC]  review/out_f/C1F_JT_VARIANTS_STEPR.json
  JT-N, lever SD 0.2 m [mid02]                                            review/out_g/C1G_JT_PRIOR_MID.json
  JT-BS                                                                   nbody/out/single/N03_<scenario>.json
  JT-BHSF [A'+OC with the STEP residual SDs]                              review/out_e/APRIME_OC_STEPR.json
  STEP                                                                    review/out_d/C2_STEP10.json
"c=0": largest over the three bodies at c = 0; "all c": largest over the bodies and the nine c.

Usage: python make_table7.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

LEV = [0, 10, -10, 20, -20, 50, -50, 100, -100]
SC = [('noise_only', 'bias-free'), ('noise_bias', 'biased')]
Q = ['e_p', 'e_v', 'e_R']
ROWS = [('JT', 'f', 'W1'), ('JT-N', 'g', 'mid02'), ('JT-W', 'f', 'W1-RW'), ('JT-H', 'f', 'W1-OC'), ('JT-B', 'f', 'JT-bias'),
        ('JT-BH', 'f', 'JT-bias-OC')]


def cells(b):
    f = tc.jload(b / 'review' / 'out_f' / 'C1F_JT_VARIANTS_STEPR.json')['results']
    g = tc.jload(b / 'review' / 'out_g' / 'C1G_JT_PRIOR_MID.json')['results']
    apr = tc.jload(b / 'review' / 'out_e' / 'APRIME_OC_STEPR.json')
    c2 = tc.jload(b / 'review' / 'out_d' / 'C2_STEP10.json')
    v = {}
    for s, name in SC:
        S = np.array(c2[s]['state']['t>=10'])
        for lab, src, key in ROWS:
            res = f if src == 'f' else g
            A = np.array([res[f'{s}|{key}|{c:+d}']['state']['t>=10'] for c in LEV])
            for j, q in enumerate(Q):
                v[f'{name}|{lab}|c=0|{q}'] = float(A[0, :, j].max())
                v[f'{name}|{lab}|all c|{q}'] = float(A[:, :, j].max())
        nb = tc.jload(b / 'nbody' / 'out' / 'single' / f'N03_{s}.json')
        for lab, a in (('JT-BS', np.array(nb['JT-BS']['state']['t>=10'])), ('JT-BHSF', np.array(apr[s]['t10'])), ('STEP', S)):
            for j, q in enumerate(Q):
                v[f'{name}|{lab}|c=0|{q}'] = float(a[:, j].max())
        # text: in how many of the nine measures STEP is smaller than JT-BS
        v[f'{name}|text|STEP smaller than JT-BS (of 9)'] = int((S < np.array(nb['JT-BS']['state']['t>=10'])).sum())
    return v


def show(v):
    print('Table 7: largest RMSEs, t in [10, 30] s (e_p m, e_v m/s, e_R deg)')
    for _, name in SC:
        print(f'({name})')
        for lab in [r[0] for r in ROWS] + ['JT-BS', 'JT-BHSF', 'STEP']:
            c0 = ' '.join(tc.g3(v[f'{name}|{lab}|c=0|{q}']) if q != 'e_R' else f"{v[f'{name}|{lab}|c=0|{q}']:.2f}" for q in Q)
            al = (' '.join(tc.g3(v[f'{name}|{lab}|all c|{q}']) if q != 'e_R' else f"{v[f'{name}|{lab}|all c|{q}']:.2f}" for q in Q)
                  if f'{name}|{lab}|all c|e_p' in v else '---')
            print(f'  {lab:8s} c=0: {c0:26s} all c: {al}')
        print(f"  text: STEP smaller than JT-BS in {v[f'{name}|text|STEP smaller than JT-BS (of 9)']} of 9 measures")


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v = cells(tc.base(a.source))
    show(v)
    ok = tc.compare('tab:jt_variants', v, a)
    raise SystemExit(0 if ok is not False else 1)
