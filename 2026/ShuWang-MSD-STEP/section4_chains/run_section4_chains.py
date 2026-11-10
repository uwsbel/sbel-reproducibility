"""Producers of Table 8, Table 11 and the Section 4.2.4 sentence (20 draws each). These runs take hours; they were not
rerun for the release check. Each producer writes one JSON file per job and skips files that exist, so a run can be
restarted. Run times are those of the paper runs (4 processes, Intel Xeon E-2176M); some of those runs also computed
results that are not used in the paper, so the producers here take at most these times.

  table8   review/c3_mc.py               STEP                                                review/out_d/mc/          ~30 min
           review/c3c_mc_aprime_oc.py    JT-BHSF, two residual settings; reads review/out_d/mc/ review/out_e/mc_aprime_oc ~17 min
           review/c3f_mc_stepR.py        JT, two-pass JT, JT-B                               review/out_f/mc/          ~25 min
           nbody/run_mc.py 3             JT-BS; STEP, JT-BHS, JT-B for Table 11 and its text nbody/out/mc/N03_*        ~15 min
  table11  nbody/run_mc.py 3 5 10        JT-B Motion 1; 5 deg runs of the text               nbody/out/mc/             ~1 h 45 min
           parity/run_mc.py              Motion 1 at 1 deg                                   parity/out/mc/            ~1 h 40 min
           parity/run_mc_c_base.py       Motion 2 at 1 deg                                   parity/out/fixc/mc/t2/base ~40 min
           parity/run_mc_d.py            Motion 2: JT-B, and 5 deg runs (text and s424)      parity/out/paper/mc/      ~45 min
  s424     parity/run_mc_d.py            (as above; only its N = 3, 5 deg jobs are used)                               ~45 min

Usage: python run_section4_chains.py table8|table11|s424|all [--dry-run]
Then:  python make_table8.py, python make_table11.py, python make_s424.py
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = {'table8': [('review', ['c3_mc.py']), ('review', ['c3c_mc_aprime_oc.py']), ('review', ['c3f_mc_stepR.py']), ('nbody', ['run_mc.py', '3'])],
        'table11': [('nbody', ['run_mc.py', '3', '5', '10']), ('parity', ['run_mc.py']), ('parity', ['run_mc_c_base.py']), ('parity', ['run_mc_d.py'])],
        's424': [('parity', ['run_mc_d.py'])]}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('target', choices=list(PLAN) + ['all'])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    import pathcheck
    if not a.dry_run:
        pathcheck.check(HERE)                   # Windows MAX_PATH and the Numba caches of nbody/ and parity/
    steps = []
    for t in (list(PLAN) if a.target == 'all' else [a.target]):
        for st in PLAN[t]:
            if st not in steps:
                steps.append(st)
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    for i, (folder, cmd) in enumerate(steps, 1):
        name = f'{folder}_{Path(cmd[0]).stem}' + ''.join(f'_{x}' for x in cmd[1:])
        print(f'step {i}: cd {folder} && python {" ".join(cmd)}   (log logs/{name}.log)', flush=True)
        if a.dry_run:
            continue
        tic = time.time()
        with open(logs / f'{name}.log', 'a') as log:
            r = subprocess.run([sys.executable] + cmd, cwd=HERE / folder, stdout=log, stderr=subprocess.STDOUT)
        print(f'  exit {r.returncode}, {(time.time() - tic) / 60:.1f} min', flush=True)
        if r.returncode:
            raise SystemExit(f'step {i} failed; see logs/{name}.log')
