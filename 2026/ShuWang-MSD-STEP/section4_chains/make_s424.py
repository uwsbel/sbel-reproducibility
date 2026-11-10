"""Section 4.2.4 sentence: on a second motion (Motion 2 of Section 4.4, N = 3, initial attitude SD 5 deg, biased
measurements, 20 draws with seeds 20263000 + 10 r + b), the mean STEP errors relative to JT-BHSF and JT-BS, with paired
bootstrap 95 % intervals of the ratio of means.

Definitions: per draw the largest RMSE over the bodies (t in [10, 30] s), then the mean over the draws; 2000 paired
resamples with numpy default_rng(20261104), separately for each filter.

Source: parity/out/paper/mc/t2_p5/N03_r<rr>_noise_bias.json (parity/run_mc_d.py). Internal name JT-BHS = JT-BHSF.

Usage: python make_s424.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc


def boot(a, b, n=2000):
    """Paired bootstrap of the ratio of means, 95 %, by column."""
    rng = np.random.default_rng(20261104)
    i = rng.integers(0, len(a), (n, len(a)))
    r = a[i].mean(axis=1) / b[i].mean(axis=1)
    return np.percentile(r, [2.5, 97.5], axis=0)


def cells(b):
    folder = b / 'parity' / 'out' / 'paper' / 'mc' / 't2_p5'
    fs = sorted(folder.glob('N03_r*_noise_bias.json'))
    if len(fs) != 20:
        raise SystemExit(f'{folder}: {len(fs)} files, expected 20')
    J = [tc.jload(f) for f in fs]
    assert all(j['p0_deg'] == 5.0 and j['trajectory'] == 2 for j in J)
    a = np.array([np.array(j['STEP']['state']['t>=10']).max(axis=0) for j in J])
    v, info = {}, {}
    for lab, F, word in (('JT-BHSF', 'JT-BHS', 'includes'), ('JT-BS', 'JT-BS', 'excludes')):
        x = np.array([np.array(j[F]['state']['t>=10']).max(axis=0) for j in J])
        r = a.mean(axis=0) / x.mean(axis=0)
        lo, hi = boot(a, x)
        v[f'STEP/{lab}|min'] = float(r.min())
        v[f'STEP/{lab}|max'] = float(r.max())
        inc = (lo <= 1) & (hi >= 1)
        v[f'STEP/{lab}|every interval {word} one'] = bool(inc.all()) if word == 'includes' else bool((~inc).all())
        info[lab] = dict(ratio_p_v_R=r.round(3).tolist(), ci_low=lo.round(3).tolist(), ci_high=hi.round(3).tolist(),
                         filter_smaller_in_all_nine_pct=float(np.mean([(np.array(j[F]['state']['t>=10']) < np.array(j['STEP']['state']['t>=10'])).all() for j in J]) * 100),
                         STEP_smaller_in_all_nine_pct=float(np.mean([(np.array(j['STEP']['state']['t>=10']) < np.array(j[F]['state']['t>=10'])).all() for j in J]) * 100))
    return v, info


if __name__ == '__main__':
    args = tc.parse_args(__doc__)
    v, info = cells(tc.base(args.source))
    for lab, d in info.items():
        print(f"STEP / {lab} (p, v, R): {d['ratio_p_v_R']}, 95% intervals "
              + ', '.join(f'{lo:.2f}-{hi:.2f}' for lo, hi in zip(d['ci_low'], d['ci_high']))
              + f"; {lab} smaller in all nine: {d['filter_smaller_in_all_nine_pct']:.0f}%, STEP: {d['STEP_smaller_in_all_nine_pct']:.0f}%")
    ok = tc.compare('sec4.2.4 second motion', v, args)
    raise SystemExit(0 if ok is not False else 1)
