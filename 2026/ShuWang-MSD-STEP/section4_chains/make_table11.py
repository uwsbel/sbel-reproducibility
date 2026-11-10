"""Table 11 (tab:longer_chains): chains of 3, 5 and 10 bodies, two motions, 20 draws each, initial attitude SD 1 deg.
Each entry: mean over the 20 draws of the largest RMSE over the bodies, t in [10, 30] s. Also prints the numbers quoted
in the text of Section 4.4.

Sources (internal name JT-BHS = JT-BHSF):
  Motion 1, STEP / JT-BS / JT-BHSF at 1 deg   parity/out/mc/                    (parity/run_mc.py, seeds 20262000 + 10 r + b)
  Motion 2, STEP / JT-BS / JT-BHSF at 1 deg   parity/out/fixc/mc/t2/base/       (parity/run_mc_c_base.py, seeds 20263000 + 10 r + b)
  JT-B, Motion 1                              nbody/out/mc/                     (nbody/run_mc.py 3 5 10)
  JT-B, Motion 2                              parity/out/paper/mc/t2_JT-B/      (parity/run_mc_d.py)
  text, 5 deg runs: Motion 1 nbody/out/mc/, Motion 2 parity/out/paper/mc/t2_p5/

Usage: python make_table11.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

SC = [('noise_only', 'bias-free'), ('noise_bias', 'biased')]
Ns = (3, 5, 10)
Q = ['e_p', 'e_v', 'e_R']
FILT = {'JT-BS': 'JT-BS', 'JT-BHSF': 'JT-BHS', 'STEP': 'STEP'}


def roots(b):
    return {(1, 1): b / 'parity' / 'out' / 'mc', (2, 1): b / 'parity' / 'out' / 'fixc' / 'mc' / 't2' / 'base',
            (1, 5): b / 'nbody' / 'out' / 'mc', (2, 5): b / 'parity' / 'out' / 'paper' / 'mc' / 't2_p5'}, \
           {1: b / 'nbody' / 'out' / 'mc', 2: b / 'parity' / 'out' / 'paper' / 'mc' / 't2_JT-B'}


def load(folder, N, s):
    fs = sorted(folder.glob(f'N{N:02d}_r*_{s}.json'))
    if len(fs) != 20:
        raise SystemExit(f'{folder} N={N} {s}: {len(fs)} files, expected 20')
    return [tc.jload(f) for f in fs]


def largest(J, F):
    return np.array([np.array(j[F]['state']['t>=10']).max(axis=0) if j[F].get('finite', True) and 'state' in j[F] else [np.nan] * 3
                     for j in J])


def cells(b):
    R, JTB = roots(b)
    v, text = {}, {}
    D = {}
    for s, name in SC:
        for m in (1, 2):
            for N in Ns:
                J = D[(m, 1, N, s)] = load(R[(m, 1)], N, s)
                for lab, F in FILT.items():
                    for j, q in enumerate(Q):
                        v[f'{name}|Motion {m}|{lab}|N={N}|{q}'] = float(largest(J, F)[:, j].mean())
                B = load(JTB[m], N, s)
                for j, q in enumerate(Q):
                    v[f'{name}|Motion {m}|JT-B|N={N}|{q}'] = float(largest(B, 'JT-B')[:, j].mean())
    # text of Section 4.4
    for m in (1, 2):
        J = D[(m, 1, 10, 'noise_bias')]
        st = largest(J, 'STEP')[:, 2]
        text[f'biased, N=10, Motion {m}: STEP draws with largest e_R > 1 / > 5 deg, median'] = (int((st > 1).sum()), int((st > 5).sum()), float(np.median(st)))
        text[f'biased, N=10, Motion {m}: JT-BHSF / JT-BS draws with largest e_R > 1 deg'] = (
            int((largest(J, 'JT-BHS')[:, 2] > 1).sum()), int((largest(J, 'JT-BS')[:, 2] > 1).sum()))
    for lab, F in FILT.items():
        cnt = [int((largest(load(R[(m, 5)], 10, s), F)[:, 2] > 5).sum()) for m in (1, 2) for s, _ in SC]
        text[f'5 deg, N=10: {lab} draws with largest e_R > 5 deg (M1 bias-free, M1 biased, M2 bias-free, M2 biased)'] = cnt
    for N in (5, 10):
        for s, name in SC:
            for lab in ('JT-BS', 'JT-BHSF'):
                r = [v[f'{name}|Motion {m}|STEP|N={N}|{q}'] / v[f'{name}|Motion {m}|{lab}|N={N}|{q}'] for m in (1, 2) for q in Q]
                text[f'{name}, N={N}: STEP / {lab}, min-max over both motions and p, v, R'] = (round(min(r), 2), round(max(r), 2))
    r = [v[f'{name}|Motion {m}|JT-BHSF|N=10|{q}'] / v[f'{name}|Motion {m}|JT-BS|N=10|{q}'] for m in (1, 2) for _, name in SC for q in Q]
    text['N=10: JT-BHSF / JT-BS, min-max'] = (round(min(r), 2), round(max(r), 2))
    for s, name in SC:
        for m in (1, 2):
            for lab, F in FILT.items():
                ch = largest(load(R[(m, 1)], 3, s), F).mean(axis=0) / largest(load(R[(m, 5)], 3, s), F).mean(axis=0)
                text[f'N=3, {name}, Motion {m}: {lab} at 1 deg / at 5 deg (p, v, R)'] = ch.round(2).tolist()
    return v, text


def show(v, text):
    print('Table 11: mean over 20 draws of the largest RMSE over the bodies (e_p m, e_v m/s, e_R deg), N = 3 | 5 | 10')
    for _, name in SC:
        print(f'({name})')
        for m in (1, 2):
            print(f'  Motion {m}')
            for lab in ('JT-B', 'JT-BS', 'JT-BHSF', 'STEP'):
                t = ' | '.join(' '.join(f"{v[f'{name}|Motion {m}|{lab}|N={N}|{q}']:.{3 if q == 'e_R' else 4}f}" for q in Q) for N in Ns)
                print(f'    {lab:8s} {t}')
    print('Text of Section 4.4:')
    for k, x in text.items():
        print(f'  {k}: {x}')


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v, text = cells(tc.base(a.source))
    show(v, text)
    ok = tc.compare('tab:longer_chains', v, a)
    raise SystemExit(0 if ok is not False else 1)
