"""Replay of Tables 5, 6, 7, 9 and 10 (three-link benchmark, single draw), then the table scripts.

Runs the producer scripts of the paper in their original order, each in its own folder, with logs in logs/:
  1  review/c2_step10.py                STEP, Stage I on [0, 10] s, both scenarios            -> review/out_d/C2_STEP10.json
  2  review/c2bc_sens_attrib.py         Table 9 three-link rows, Table 10 (4 processes)       -> review/out_d/C2bc_SENS_ATTRIB.json
  3  review/run_Aprime_oc_stepR.py      JT-BHSF (STEP start, fixed levers)                     -> review/out_e/APRIME_OC_STEPR.json
  4  review/c1f_jt_variants_stepR.py    JT, JT-W, JT-H, JT-B, JT-BH, matched prior, nine c     -> review/out_f/C1F_JT_VARIANTS_STEPR.json
  5  review/c1g_jt_prior_mid.py         JT-N with lever SD 0.2 m (4 processes)                 -> review/out_g/C1G_JT_PRIOR_MID.json
  6  nbody/run_jtbs_single.py           JT-BS, N = 3, paper draw                               -> nbody/out/single/N03_*.json
  7  run_table9_spatial.py --tag <tag>  Table 9 spatial double-pendulum rows                   -> ../section3_spatial/runs/<tag>_*
The producers write to fixed folders (review/out_*, nbody/out), so this script refuses to start if any of them exists.
Step 4 writes about 1.4 GB of state histories (review/out_f/jt/).

Usage: python run_section4_three_link.py [--spatial-tag t9] [--from-step 1]
Then:  python make_table5.py   (and make_table6.py, make_table7.py, make_table9.py --spatial-tag <tag>, make_table10.py)
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [('review', ['c2_step10.py']), ('review', ['c2bc_sens_attrib.py']), ('review', ['run_Aprime_oc_stepR.py']),
         ('review', ['c1f_jt_variants_stepR.py']), ('review', ['c1g_jt_prior_mid.py']), ('nbody', ['run_jtbs_single.py']),
         ('.', ['run_table9_spatial.py', '--tag', None])]
OUTPUTS = [HERE / 'review' / 'out_d' / 'C2_STEP10.json', HERE / 'review' / 'out_d' / 'C2bc_SENS_ATTRIB.json',
           HERE / 'review' / 'out_e' / 'APRIME_OC_STEPR.json', HERE / 'review' / 'out_f', HERE / 'review' / 'out_g',
           HERE / 'nbody' / 'out' / 'single']

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spatial-tag', default='t9')
    p.add_argument('--from-step', type=int, default=1, help='restart after a failure (earlier outputs are kept)')
    a = p.parse_args()
    import pathcheck
    pathcheck.check(HERE)                       # Windows MAX_PATH and the Numba cache of nbody/ (step 6)
    if a.from_step == 1:
        old = [str(x) for x in OUTPUTS if x.exists()]
        if old:
            raise SystemExit('replay outputs already exist (delete them, or use a fresh copy of the package):\n  ' + '\n  '.join(old))
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    for i, (folder, cmd) in enumerate(STEPS, 1):
        if i < a.from_step:
            continue
        cmd = [a.spatial_tag if c is None else c for c in cmd]
        tic = time.time()
        with open(logs / f'step{i}_{Path(cmd[0]).stem}.log', 'w') as log:
            r = subprocess.run([sys.executable] + cmd, cwd=HERE / folder, stdout=log, stderr=subprocess.STDOUT)
        print(f'step {i} {folder}/{" ".join(cmd)}: exit {r.returncode}, {(time.time() - tic) / 60:.1f} min', flush=True)
        if r.returncode:
            raise SystemExit(f'step {i} failed; see logs/step{i}_{Path(cmd[0]).stem}.log')
    print('done; now run the make_table*.py scripts')
