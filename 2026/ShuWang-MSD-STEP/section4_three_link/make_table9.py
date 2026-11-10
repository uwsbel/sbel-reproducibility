"""Table 9 (tab:sigma_sensitivity): Stage I sample exclusion and the gyroscope noise level sigma_g assumed in the noise
compensation. Each entry is the largest value of the system: e_l over its lever vectors, e_p and e_R over its bodies.

Three-link columns (t in [10, 30] s): review/out_d/C2bc_SENS_ATTRIB.json (keys sens|<scenario>|<row>) and, for
sigma_g = sigma_sim, review/out_d/C2_STEP10.json.
Spatial double-pendulum columns (complete record): the runs written by run_table9_spatial.py into
../section3_spatial/runs/<tag>_<row>/ (replay), or expected/spatial/<row>/ (paper runs).

Usage: python make_table9.py [--source replay|expected] [--spatial-tag t9] [--paper ../paper_values_ch4.json]
"""
import csv
import numpy as np
import tables_common as tc

# (paper row: exclusion, sigma_g / sigma_sim) -> (three-link key in C2bc_SENS_ATTRIB.json, spatial row name)
ROWS = [(('no', '0'), 'none', 'no0'), (('yes', '0'), '0', 'yes0'), (('yes', '0.5'), '0.5', 'x0p5'), (('yes', '0.8'), '0.8', 'x0p8'),
        (('yes', '1'), None, 'x1'), (('yes', '1.2'), '1.2', 'x1p2'), (('yes', '2'), '2', 'x2')]
EXTRA = [('yes0rule', 'yes, 0 (spatial_rule.json, diagnostic, not a paper row)')]


def spatial_run(run):
    geo = tc.jload(run / 'geometry_metrics.json')
    with open(run / 'state_metrics.csv', newline='') as f:
        st = [r for r in csv.DictReader(f) if r.get('frame', 'declared_heading_chart') == 'declared_heading_chart']
    return dict(e_l=max(g['error_percent'] for g in geo), e_p=max(float(r['position_rmse_m']) for r in st),
                e_R=max(float(r['attitude_rmse_deg']) for r in st))


def cells(b, source, tag):
    x = tc.jload(b / 'review' / 'out_d' / 'C2bc_SENS_ATTRIB.json')
    c2 = tc.jload(b / 'review' / 'out_d' / 'C2_STEP10.json')
    v, extra = {}, {}
    for (ex, sg), key, row in ROWS:
        for s, name in (('noise_only', 'three-link bias-free'), ('noise_bias', 'three-link biased')):
            r = c2[s] if key is None else x[f'sens|{s}|{key}']
            st = np.array(r['state']['t>=10'])
            v[f'{name}|exclusion={ex}|sigma_g={sg}|e_l'] = float(max(r['lever_pct']))
            v[f'{name}|exclusion={ex}|sigma_g={sg}|e_p'] = float(st[:, 0].max())
            v[f'{name}|exclusion={ex}|sigma_g={sg}|e_R'] = float(st[:, 2].max())
        run = (tc.HERE.parent / 'section3_spatial' / 'runs' / f'{tag}_{row}') if source == 'replay' else (b / 'spatial' / row)
        if run.exists():
            for q, val in spatial_run(run).items():
                v[f'spatial double pendulum|exclusion={ex}|sigma_g={sg}|{q}'] = val
        else:
            print(f'missing spatial run {run}')
    if source == 'replay':
        for row, lab in EXTRA:
            run = tc.HERE.parent / 'section3_spatial' / 'runs' / f'{tag}_{row}'
            if run.exists():
                extra[lab] = spatial_run(run)
    return v, extra


def show(v, extra):
    print('Table 9: e_l (%) / e_p (m) / e_R (deg), largest over levers or bodies')
    print(f"{'exclusion, sigma_g':22s}{'spatial double pendulum':>28s}{'three-link, bias-free':>28s}{'three-link, biased':>28s}")
    for (ex, sg), _, _ in ROWS:
        line = f'{ex + ", " + sg:22s}'
        for name in ('spatial double pendulum', 'three-link bias-free', 'three-link biased'):
            k = f'{name}|exclusion={ex}|sigma_g={sg}|'
            line += (f"{v[k + 'e_l']:>10.2f}{v[k + 'e_p']:>9.4f}{v[k + 'e_R']:>9.2f}" if k + 'e_l' in v else f"{'---':>28s}")
        print(line)
    for lab, r in extra.items():
        print(f"{lab}: {r['e_l']:.2f} / {r['e_p']:.4f} / {r['e_R']:.2f}")


if __name__ == '__main__':
    a = tc.parse_args(__doc__, lambda p: p.add_argument('--spatial-tag', default='t9'))
    v, extra = cells(tc.base(a.source), a.source, a.spatial_tag)
    show(v, extra)
    ok = tc.compare('tab:sigma_sensitivity', v, a)
    raise SystemExit(0 if ok is not False else 1)
