"""Table 7, JT-BS row: the N = 3 jobs of run_single.py (paper draw, seeds 20261008 + b, both scenarios).

run_single.py runs N = 3, 5 and 10; this release driver calls its work() function, unchanged, for N = 3 only, because the
N = 5 and 10 records are not part of this section. Writes out/single/N03_<scenario>.json (and state histories in
out/single/hist/); an existing file is skipped, as in run_single.py.

Usage (inside nbody/): python run_jtbs_single.py
"""
from multiprocessing import Pool
import run_single as rs
import nbody_common as nc

if __name__ == '__main__':
    rs.HIST.mkdir(parents=True, exist_ok=True)
    with Pool(2) as pool:
        for name, status, an in pool.imap_unordered(rs.work, [(3, s) for s in nc.SCEN]):
            print(name, status, an, flush=True)
