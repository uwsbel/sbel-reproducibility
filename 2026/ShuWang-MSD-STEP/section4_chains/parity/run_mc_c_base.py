"""Table 11, Motion 2 at 1 deg: STEP (parity_fix.chain), JT-BHS, JT-BS and JT-BS-OC on the records of parity_c.py,
trajectory 2, N = 3, 5, 10, 20 draws with seeds 20263000 + 10 r + b, both scenarios, attitude P0 = 1 deg; run_mc_b.job.
Writes out/fixc/mc/t2/base/N<NN>_r<rr>_<scenario>.json; existing files are skipped, so a restart continues; stops on an
anomaly, finished jobs stay on disk.

Usage (inside parity/): python run_mc_c_base.py
"""
import parity_c as pcx
import parity_common as pc
from multiprocessing import Pool
import nbody_jobs as nj
import run_mc_b as rb
nc = pc.nc
MC = pc.OUT / 'fixc' / 'mc' / 't2' / 'base'
FILTERS = ('JT-BHS', 'JT-BS', 'JT-BS-OC')


def work(a):
    N, r, s = a; path = MC / f'N{N:02d}_r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped', []
    nc.load_clean = lambda n: pcx.load_clean(2, n)
    res = rb.job(N, r, pcx.seeds(2, N, r), s, pc.P0_DEG, FILTERS); res['trajectory'] = 2
    nj.write(path, res)
    an = nj.anomaly(res)
    if not res['STEP']['finite']: an.append('STEP non-finite')
    return path.name, 'done', an


if __name__ == '__main__':
    MC.mkdir(parents=True, exist_ok=True)
    jobs = [(N, r, s) for N in (3, 5, 10) for r in range(1, 21) for s in nc.SCEN]
    with Pool(4) as pool:
        for name, status, an in pool.imap(work, jobs):
            print(name, status, an, flush=True)
            if an:
                pool.terminate()
                raise SystemExit(f'anomaly in {name}: {an}')
