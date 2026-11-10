"""The 20 draws of Motion 1 (seeds 20262000 + 10 r + b), both scenarios, N = 3, 5, 10,
attitude P0 = 1 deg for all filters. One file per job in out/mc/; existing files are skipped, so a restart continues. argv: optional list of N."""
import parity_common as pc
import sys, json
from multiprocessing import Pool
import nbody_jobs as nj
import parity_jobs as pj
nc = pc.nc
MC = pc.OUT / 'mc'


def work(a):
    N, r, s = a; path = MC / f'N{N:02d}_r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped', []
    res = pj.job(N, r, nc.mc_seeds(N, r), s, pc.P0_DEG)
    nj.write(path, res)
    return path.name, 'done', nj.anomaly(res)


if __name__ == '__main__':
    MC.mkdir(parents=True, exist_ok=True)
    Ns = [int(x) for x in sys.argv[1:]] or [3, 5, 10]
    jobs = [(N, r, s) for N in Ns for r in range(1, 21) for s in nc.SCEN]
    with Pool(4) as pool:
        for name, status, an in pool.imap_unordered(work, jobs):
            print(name, status, an, flush=True)
            if an:                                 # stop and report; finished jobs stay on disk
                pool.terminate(); raise SystemExit(f'anomaly in {name}: {an}')
