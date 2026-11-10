"""Table 8 (tab:mc_realizations): three-link benchmark over 20 further draws (seeds 20262000 + 10 r + b), t in [10, 30] s.
Each entry: mean +- SD over the draws of the largest RMSE of the three bodies; e_l of the largest of the five lever errors;
last column: percentage of draws in which STEP is smaller than the filter in all nine state measures.

Sources (internal names in brackets):
  STEP                           review/out_d/mc/r<rr>_<scenario>.json      [STEP]        (c3_mc.py)
  JT, JT-B, c in {0, +-10, +-50}  review/out_f/mc/r<rr>_<scenario>.json      [JT, JT-bias] (c3f_mc_stepR.py)
  JT-BHSF                        review/out_e/mc_aprime_oc/r<rr>_<s>.json   [Rstep]       (c3c_mc_aprime_oc.py)
  JT-BS                          nbody/out/mc/N03_r<rr>_<scenario>.json     [JT-BS]       (nbody/run_mc.py 3)
Also prints the percentages and ranges quoted in the text of Section 4.2.3.

Usage: python make_table8.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

SC = [('noise_only', 'bias-free'), ('noise_bias', 'biased')]
MC_L = [0, 10, -10, 50, -50]
Q = ['e_p', 'e_v', 'e_R']


def cs(c):
    return '0' if c == 0 else f'{c:+d}'


def load(folder, pattern, n=20):
    fs = sorted(folder.glob(pattern))
    if len(fs) != n:
        raise SystemExit(f'{folder}/{pattern}: {len(fs)} files, expected {n}')
    return [tc.jload(f) for f in fs]


def cells(b):
    v, text = {}, {}
    for s, name in SC:
        St = load(b / 'review' / 'out_d' / 'mc', f'r*_{s}.json')
        F = load(b / 'review' / 'out_f' / 'mc', f'r*_{s}.json')
        A = load(b / 'review' / 'out_e' / 'mc_aprime_oc', f'r*_{s}.json')
        NBm = load(b / 'nbody' / 'out' / 'mc', f'N03_r*_{s}.json')
        assert [x['realization'] for x in St] == [x['realization'] for x in F] == [x['realization'] for x in A] == list(range(1, 21))
        assert [x['seeds'] for x in St] == [x['seeds'] for x in NBm]
        S = np.array([x['STEP']['state']['t>=10'] for x in St])
        SL = np.array([x['STEP']['lever_pct'] for x in St])

        def put(method, c, X, XL, step_row=False):
            mx = X.max(axis=1)
            for j, q in enumerate(Q):
                v[f'{name}|{method}|c={c}|{q}|mean'] = float(mx[:, j].mean())
                v[f'{name}|{method}|c={c}|{q}|sd'] = float(mx[:, j].std(ddof=1))
            if XL is not None:
                v[f'{name}|{method}|c={c}|e_l|mean'] = float(XL.max(axis=1).mean())
                v[f'{name}|{method}|c={c}|e_l|sd'] = float(XL.max(axis=1).std(ddof=1))
            if not step_row:
                v[f'{name}|{method}|c={c}|STEP smaller in all nine (%)'] = float((S < X).all(axis=(1, 2)).mean() * 100)
                text[f'{name}|{method}|c={c}|filter smaller in all nine (%)'] = float((S >= X).all(axis=(1, 2)).mean() * 100)
                text[f'{name}|{method}|c={c}|mean ratio to STEP, min-max over p v R'] = (
                    float((mx.mean(axis=0) / S.max(axis=1).mean(axis=0)).min()), float((mx.mean(axis=0) / S.max(axis=1).mean(axis=0)).max()))
        for key, method in (('JT', 'JT'), ('JT-bias', 'JT-B')):
            for c in MC_L:
                X = np.array([x[f'{key}|{c:+d}']['state']['t>=10'] for x in F])
                XL = np.array([x[f'{key}|{c:+d}']['lever_pct'] for x in F])
                put(method, cs(c), X, XL)
        put('JT-BS', '---', np.array([x['JT-BS']['state']['t>=10'] for x in NBm]), np.array([x['JT-BS']['lever_pct'] for x in NBm]))
        put('JT-BHSF', '---', np.array([x['Rstep']['state']['t>=10'] for x in A]), None)
        put('STEP', '---', S, SL, step_row=True)
        X5 = np.array([x['R005']['state']['t>=10'] for x in A]).max(axis=1)
        text[f'{name}|JT-BHSF with residual SD 0.005/0.005: mean ratio to STEP, min-max'] = (
            float((X5.mean(axis=0) / S.max(axis=1).mean(axis=0)).min()), float((X5.mean(axis=0) / S.max(axis=1).mean(axis=0)).max()))
        two = np.array([x['JT-2p|+0']['state']['t>=10'] for x in F]).max(axis=1).mean(axis=0) / \
            np.array([x['JT|+0']['state']['t>=10'] for x in F]).max(axis=1).mean(axis=0)
        text[f'{name}|two-pass JT / JT at c = 0, mean largest errors (p, v, R)'] = two.round(4).tolist()
        if s == 'noise_bias':
            j0 = np.array([x['JT|+0']['state']['t>=10'] for x in F]).max(axis=1)[:, 2]
            text['biased|JT c=0 largest e_R over draws: min, max, median (deg)'] = (float(j0.min()), float(j0.max()), float(np.median(j0)))
            text['biased|STEP largest e_R over draws: min, max (deg)'] = (float(S.max(axis=1)[:, 2].min()), float(S.max(axis=1)[:, 2].max()))
    return v, text


def show(v, text):
    print('Table 8: mean +- SD over 20 draws (e_p m, e_v m/s, e_R deg, e_l %), STEP smaller in all nine (%)')
    for _, name in SC:
        print(f'({name})')
        rows = [('JT', cs(c)) for c in MC_L] + [('JT-B', cs(c)) for c in MC_L] + [('JT-BS', '---'), ('JT-BHSF', '---'), ('STEP', '---')]
        for m, c in rows:
            k = f'{name}|{m}|c={c}|'
            t = [f"{v[k + q + '|mean']:.{d}f}+-{v[k + q + '|sd']:.{d}f}" for q, d in (('e_p', 4), ('e_v', 4), ('e_R', 3))]
            el = f"{v[k + 'e_l|mean']:.3f}+-{v[k + 'e_l|sd']:.3f}" if k + 'e_l|mean' in v else '---'
            pc = f"{v[k + 'STEP smaller in all nine (%)']:.0f}" if k + 'STEP smaller in all nine (%)' in v else '---'
            print(f'  {m:8s} {c:4s} ' + '  '.join(t) + f'  {el:14s} {pc}')
    print('Text of Section 4.2.3:')
    for k, x in text.items():
        print(f'  {k}: {x}')


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v, text = cells(tc.base(a.source))
    show(v, text)
    ok = tc.compare('tab:mc_realizations', v, a)
    raise SystemExit(0 if ok is not False else 1)
