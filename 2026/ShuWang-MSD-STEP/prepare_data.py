"""Generate the simulated input records of Section 4 that are not stored in this repository.

Run once from the repository root before the Section 4 replays (about 5 minutes):

    python prepare_data.py

Records that already exist are left unchanged.

1. Three-link clean record: stored in section4_three_link/step/data/clean; copied to section4_chains/step/data/clean
   and section4_timing/step/data/clean.
2. Three-link noisy records (noise_only, noise_bias) in section4_three_link/step/data and section4_chains/step/data:
   the clean record plus the sensor noise and the constant biases that step/generate_measurement_errors.py draws with
   the seeds of step/configs/measurement_errors.json. The arrays are identical to those of the paper runs.
3. Chain records, Motion 1, N = 5 and 10: section4_chains/nbody/gen_data.py; copied to section4_timing/nbody/data.
4. Chain records, Motion 2, N = 3, 5 and 10: section4_chains/parity/gen_data_c.py, which also checks the generator
   against the Motion 1 record for N = 5.
On the machine of the paper runs (Windows 11, Python 3.9.13 and the package versions of requirements.txt), every
record written by this script was byte-identical to the record used in the paper runs (checked 2026-11-10). Records 3
and 4 are integrated with solve_ivp, so on another machine or with other library versions they may differ in the last
bits, and table values computed from them may then differ in the last digit.
The SOLVER.json files of the paper records are in section4_chains/expected/.
"""
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TL, CH, TI = ROOT / 'section4_three_link', ROOT / 'section4_chains', ROOT / 'section4_timing'

NOISE = r'''
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import generate_measurement_errors as gme
from workflow_io import load_npz
here = Path('.').resolve()
cfg = json.loads((here / 'configs' / 'measurement_errors.json').read_text())
clean = load_npz(here / 'data' / 'clean' / 'measurements.npz')
reference = load_npz(here / 'data' / 'clean' / 'reference.npz')
scenarios, draws = gme.generate_arrays(clean, cfg)
for name in ('noise_only', 'noise_bias'):
    dest = here / 'data' / name
    if dest.exists():
        print('exists, kept:', dest); continue
    ref = {k: v.copy() for k, v in reference.items()}
    for b in (1, 2, 3):
        ref[f'ba{b}'] = draws[f'ba{b}'].copy() if name != 'noise_only' else np.zeros(3)
        ref[f'bg{b}'] = draws[f'bg{b}'].copy() if name != 'noise_only' else np.zeros(3)
    dest.mkdir()
    np.savez_compressed(dest / 'measurements.npz', **scenarios[name]); np.savez_compressed(dest / 'reference.npz', **ref)
    print('generated:', dest)
'''


def run(cmd, cwd):
    print(f'> {" ".join(cmd)}   (in {cwd.relative_to(ROOT)})', flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def copy(src, dst):
    if dst.exists():
        print('exists, kept:', dst.relative_to(ROOT)); return
    shutil.copytree(src, dst)
    print('copied:', dst.relative_to(ROOT))


def main():
    clean = TL / 'step' / 'data' / 'clean'
    copy(clean, CH / 'step' / 'data' / 'clean')
    copy(clean, TI / 'step' / 'data' / 'clean')
    for sec in (TL, CH):
        run([sys.executable, '-c', NOISE], sec / 'step')
    for N in (5, 10):
        if (CH / 'nbody' / 'data' / f'N{N:02d}' / 'clean').exists():
            print('exists, kept:', f'section4_chains/nbody/data/N{N:02d}/clean')
        else:
            run([sys.executable, 'gen_data.py', str(N)], CH / 'nbody')
        copy(CH / 'nbody' / 'data' / f'N{N:02d}', TI / 'nbody' / 'data' / f'N{N:02d}')
    t2 = CH / 'parity' / 'data_c' / 't2'
    if all((t2 / f'N{N:02d}' / 'clean').exists() for N in (3, 5, 10)):
        print('exists, kept: section4_chains/parity/data_c/t2')
    elif any((t2 / f'N{N:02d}' / 'clean').exists() for N in (3, 5, 10)):
        raise SystemExit(f'{t2} is incomplete; delete it and run again')
    else:
        run([sys.executable, 'gen_data_c.py'], CH / 'parity')
    print('done')


if __name__ == '__main__':
    main()
