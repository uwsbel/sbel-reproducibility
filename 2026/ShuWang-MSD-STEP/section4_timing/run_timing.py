"""Table 12 and the Stage I times: run parity/timing_paper.py (unchanged), after the Windows path-length check.

timing_paper.py: paper draw, Motion 1, biased measurements, initial attitude SD 1 deg, N = 3, 5, 10; STEP Stage II
(compiled), JT-BHSF, JT-BS and JT-B timed in ten alternated full-record runs after warm-up, one process, one BLAS thread;
Stage I three runs; the parents' Stage II on [0, 10] s three runs. About 30 min. Run it on an otherwise idle machine.
It writes parity/out/paper/timing/TIMING_P.json and refuses to overwrite it; --smoke runs the code check
(N = 3, one repetition, SMOKE_P.json, about 1.5 min). It uses the Windows command tasklist, so it runs on Windows only.

Usage: python run_timing.py [--smoke]     then: python make_table12.py
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path
import pathcheck

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    pathcheck.check(HERE)
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    name = 'timing_paper_smoke' if a.smoke else 'timing_paper'
    tic = time.time()
    with open(logs / f'{name}.log', 'w') as log:
        r = subprocess.run([sys.executable, 'timing_paper.py'] + (['smoke'] if a.smoke else []), cwd=HERE / 'parity', stdout=log, stderr=subprocess.STDOUT)
    print(f'timing_paper.py{" smoke" if a.smoke else ""}: exit {r.returncode}, {(time.time() - tic) / 60:.1f} min (log logs/{name}.log)')
    raise SystemExit(r.returncode)
