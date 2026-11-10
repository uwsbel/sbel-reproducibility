"""Table 9, spatial double-pendulum columns: rerun the seven rows with the package code of section3_spatial.

Each row is one run of ../section3_spatial/run_case.py with its own Stage I configuration (STEP_CASE_CONFIG) on the
record of the paper (STEP_COPY_DATA_FROM = ../section3_spatial/expected/data), exactly as the paper runs were made.
Runs are written to ../section3_spatial/runs/<tag>_<row>/; run_case.py refuses to
overwrite an existing run.

Rows (paper: sample exclusion, assumed sigma_g):
  no0   no,  0          spatial.json
  yes0  yes, 0          spatial_eiv_sd0.json    (the paper run used a calibration.py without the noise-compensation
                                                 code path; see section4_README.md)
  x0p5  yes, 0.5 sim    spatial_eiv_x0p5.json
  x0p8  yes, 0.8 sim    spatial_eiv_x0p8.json
  x1    yes, sim        spatial_eiv.json        (the Table 2 configuration)
  x1p2  yes, 1.2 sim    spatial_eiv_x1p2.json
  x2    yes, 2 sim      spatial_eiv_x2.json
  yes0rule  diagnostic: the same row with spatial_rule.json (exclusion only, no compensation code path); not a paper row

Usage: python run_table9_spatial.py [--tag t9] [--rows no0 yes0 ...]
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = HERE.parent / 'section3_spatial'
ROWS = {'no0': 'spatial.json', 'yes0': 'spatial_eiv_sd0.json', 'x0p5': 'spatial_eiv_x0p5.json', 'x0p8': 'spatial_eiv_x0p8.json',
        'x1': 'spatial_eiv.json', 'x1p2': 'spatial_eiv_x1p2.json', 'x2': 'spatial_eiv_x2.json', 'yes0rule': 'spatial_rule.json'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='t9')
    p.add_argument('--rows', nargs='*', default=list(ROWS))
    a = p.parse_args()
    data = SP / 'expected' / 'data'
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    for row in a.rows:
        tag = f'{a.tag}_{row}'
        if (SP / 'runs' / tag).exists():
            raise SystemExit(f'{SP / "runs" / tag} exists; use a new --tag')
        env = dict(os.environ, STEP_CASE_CONFIG=ROWS[row], STEP_COPY_DATA_FROM=str(data))
        with open(logs / f'spatial_{tag}.log', 'w') as log:
            r = subprocess.run([sys.executable, 'run_case.py', 'spatial', '--tag', tag], cwd=SP, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'{row}: {ROWS[row]} -> section3_spatial/runs/{tag} (exit {r.returncode})', flush=True)
        if r.returncode:
            raise SystemExit(f'run failed, see {logs / f"spatial_{tag}.log"}')
