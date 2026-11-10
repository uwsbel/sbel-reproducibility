"""Compare a replay with the paper.

For each section: (1) every table value in paper_values.json is recomputed from the replay and
rounded to the printed number of decimals; (2) if the state histories of the paper run are present in
expected/ (body<b>_states.npz; not stored in the public repository to keep it small), they are compared
with those of the replay.

Usage: python check_replay.py <section> <tag>
Exit status 0 only if every table value matches.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent


def spatial(run):
    m = [r for r in json.loads((run / 'metrics.json').read_text()) if r['frame'] == 'declared_heading_chart']
    g = {(x['body_stage'], x['vector']): x['error_percent'] for x in json.loads((run / 'geometry_metrics.json').read_text())}
    v = {'e_l l1A': g[(1, 'lever')], 'e_l l1B': g[(2, 'parent_lever')], 'e_l l2B': g[(2, 'lever')]}
    for r in m:
        b = r['body']
        v.update({f'theta0 B{b}': r['initial_attitude_deg'], f'e_p B{b}': r['position_rmse_m'],
                  f'e_v B{b}': r['velocity_rmse_m_s'], f'e_R B{b}': r['attitude_rmse_deg']})
    return v


def slider(run):
    res = run / 'results'
    g = {x['vector']: x['in_plane_relative_error_percent'] for x in json.loads((res / 'geometry_metrics.json').read_text())}
    v = {f'e_l l{k}': g[k] for k in ('1A', '1B', '2B', '2C')}
    for r in json.loads((res / 'state_metrics.json').read_text()):
        b = r['body']
        v.update({f'theta0 B{b}': r['initial_attitude_error_deg'], f'dir B{b}': r['direction_error_deg'],
                  f'e_p B{b}': r['position_rmse_m'], f'e_v B{b}': r['velocity_rmse_m_s'], f'e_R B{b}': r['attitude_rmse_deg']})
    return v


def experiment(run):
    t = json.loads((run / 'TABLE4.json').read_text())
    e = t['lever_error_pct']
    v = {'e_l l1A': e['lever1'], 'e_l l1B': e['parent_lever2'], 'e_l l2B': e['lever2']}
    for b in ('1', '2'):
        v[f'theta0 B{b}'] = t['theta0_dn'][b]['theta0']
        v[f'dir B{b}'] = t['theta0_dn'][b]['dn']
    for r in t['states']:
        b = r['body']
        v.update({f'e_p B{b}': r['position_rmse_m'], f'e_v B{b}': r['velocity_rmse_m_s'], f'e_R B{b}': r['attitude_rmse_deg']})
    return v


SECTIONS = {'section3_spatial': (spatial, '', 2), 'section3_slider': (slider, 'results/', 3), 'section3_experiment': (experiment, '', 2)}


def states(run_dir, exp_dir, n):
    if not all((exp_dir / f'body{b}_states.npz').exists() for b in range(1, n + 1)):
        return 'state histories of the paper run not available; table values compared only'
    out = {}
    for b in range(1, n + 1):
        a = np.load(run_dir / f'body{b}_states.npz')
        e = np.load(exp_dir / f'body{b}_states.npz')
        ang = R.from_matrix(np.einsum('nji,njk->nik', a['R'], e['R'])).magnitude()
        out[f'body{b}'] = dict(p_m=float(np.abs(a['p'] - e['p']).max()), v_m_s=float(np.abs(a['v'] - e['v']).max()),
                               R_deg=float(np.rad2deg(ang.max())), ba=float(np.abs(a['ba'] - e['ba']).max()),
                               bg=float(np.abs(a['bg'] - e['bg']).max()))
    return out


def main(section, tag):
    fn, sub, n = SECTIONS[section]
    base = ROOT / section
    run = base / 'runs' / tag
    paper = json.loads((ROOT / 'paper_values.json').read_text())[section]
    got = fn(run)
    rows, ok = [], True
    for key, printed in paper['values'].items():
        d = len(printed.split('.')[1]) if '.' in printed else 0
        val = got[key]
        match = round(val, d) == float(printed)
        ok &= match
        rows.append(dict(quantity=key, paper=printed, replay=f'{val:.{d + 2}f}', match=match))
    hist = states(run / sub if sub else run, base / 'expected' / sub if sub else base / 'expected', n)
    inputs = None
    if (base / 'expected' / 'INPUT_HASHES.json').exists():
        inputs = json.loads((run / 'INPUT_HASHES.json').read_text()) == json.loads((base / 'expected' / 'INPUT_HASHES.json').read_text())
        ok &= inputs
        print(f'input measurements identical to the paper run (SHA-256): {inputs}')
    report = dict(section=section, tag=tag, table=paper['table'], all_table_values_match=ok, values=rows,
                  inputs_identical_to_paper_run=inputs, max_abs_difference_to_paper_run=hist)
    (run / 'CHECK_AGAINST_PAPER.json').write_text(json.dumps(report, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(hist, indent=1))
    print(f"{section}: {'PASS' if ok else 'FAIL'} ({paper['table']})")
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
