"""The 20 draws (seeds 20262000 + 10 r + b), both scenarios, N = 3, 5, 10. One file per job in out/mc/; existing
files are skipped, so a restart continues. argv: optional list of N."""
import nbody_common as nc
import sys
from multiprocessing import Pool
import nbody_jobs as nj

MC = nc.OUT / 'mc'


def work(a):
    N, r, s = a; path = MC / f'N{N:02d}_r{r:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped', []
    res = nj.job(N, r, nc.mc_seeds(N, r), s)
    nj.write(path, res)
    return path.name, 'done', nj.anomaly(res)


if __name__ == '__main__':
    MC.mkdir(parents=True, exist_ok=True)
    Ns = [int(x) for x in sys.argv[1:]] or [3, 5, 10]
    jobs = [(N, r, s) for N in Ns for r in range(1, 21) for s in nc.SCEN]
    with Pool(4) as pool:
        for name, status, an in pool.imap_unordered(work, jobs):
            print(name, status, an, flush=True)
