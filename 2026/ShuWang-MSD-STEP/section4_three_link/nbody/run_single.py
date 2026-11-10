"""Paper draw (seeds 20261008 + b), both scenarios, N = 3, 5, 10. Writes out/single/; existing files are skipped."""
import nbody_common as nc
from multiprocessing import Pool
import nbody_jobs as nj

SINGLE = nc.OUT / 'single'; HIST = SINGLE / 'hist'


def work(a):
    N, s = a; path = SINGLE / f'N{N:02d}_{s}.json'
    if path.exists(): return path.name, 'skipped', []
    res = nj.job(N, 'paper', nc.paper_seeds(N), s, HIST)
    nj.write(path, res)
    return path.name, 'done', nj.anomaly(res)


if __name__ == '__main__':
    HIST.mkdir(parents=True, exist_ok=True)
    jobs = [(N, s) for N in (3, 5, 10) for s in nc.SCEN]
    with Pool(4) as pool:
        for name, status, an in pool.imap_unordered(work, jobs):
            print(name, status, an, flush=True)
