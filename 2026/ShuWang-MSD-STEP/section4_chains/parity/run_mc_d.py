"""Second trajectory (Motion 2), 20 draws, both scenarios, seeds 20263000 + 10 r + b. Jobs: N = 3, attitude P0 = 5 deg; truth-start
JT-B, N = 3, 5, 10; N = 5, 10, attitude P0 = 5 deg. The 5 deg jobs use run_mc_b.job with STEP, JT-BHS and JT-BS. One file per job in out/paper/mc/; existing files are skipped.
At 5 deg, non-finite filters and Stage I that does not converge are results: recorded, counted, not a stop. A non-finite JT-B or a code error stops the run.
argv: gate, the same code on the first trajectory (N = 3, r01, noise_bias) against nbody/out/mc; writes out/paper/GATE_D.json, exit 1 above 1e-5."""
import parity_c as pcx
import parity_common as pc
import json, sys, time
from multiprocessing import Pool
import numpy as np
import nbody_jobs as nj
import jt_nbody_struct as js
import run_mc_b as rb
nc = pc.nc
ROOT = pc.OUT / 'paper' / 'mc'; ORDER = [('p5', 3), ('JT-B', 3), ('JT-B', 5), ('JT-B', 10), ('p5', 5), ('p5', 10)]
FILTERS_P5 = ('JT-BHS', 'JT-BS')
FIRST_LOAD = nc.load_clean


def jtb_job(N, tag, seeds, s):
    """The JT-B part of nbody_jobs.job."""
    clean, ref = nc.load_clean(N); sc, bias = nc.draw(clean, N, seeds); m = sc[s]
    t, G = nc.target(N, ref); tm = m['time']; mask = tm >= nc.T_EVAL

    def truth(b, key):
        return bias[f'{key}{b}'] if s == 'noise_bias' else np.zeros(3)
    res = dict(N=N, draw=tag, scenario=s, seeds={str(b): int(v) for b, v in seeds.items()})
    rt, vt, qt = nc.true_states(N, ref, G); gt = nc.truth_levers(N); tic = time.perf_counter()
    try:
        h = js.run(m, N, rt, vt, qt, gt, True, False, 1.0, None, pattern=js.h_pattern(N, True))
    except FloatingPointError as err:
        res['JT-B'] = dict(finite=False, error=str(err)); return res
    ok = bool(all(np.isfinite(h[k]).all() for k in 'rvqbL')); rec = dict(finite=ok, wall_s=time.perf_counter() - tic)
    if ok:
        rec['state'] = nc.rows_jt(h, t, tm, N); rec['lever_pct'] = nc.jt_lever_pct(h, N)

        def estb(b, key):
            o = 6 * (b - 1) + (0 if key == 'ba' else 3)
            return h['b'][:, o:o + 3]
        hs, ts, series = [], [], []
        for b in range(1, N + 1):
            a, c, x = nc.heading_tilt(nc.jt_body(h, b)['R'], t[f'R{b}'], mask); hs.append(a); ts.append(c); series.append(x)
        rec.update(bias=nj.bias_rows(estb, truth, N, mask), heading=hs, tilt=ts, common_heading=float(np.sqrt(np.mean(np.mean(series, axis=0)[mask] ** 2))))
    res['JT-B'] = rec
    return res


def work(a):
    kind, N, r, s = a; path = ROOT / f't2_{kind}' / f'N{N:02d}_r{r:02d}_{s}.json'
    if path.exists(): return f'{kind}/{path.name}', 'skipped', [], False
    nc.load_clean = lambda n: pcx.load_clean(2, n); seeds = pcx.seeds(2, N, r)
    if kind == 'JT-B':
        res = jtb_job(N, r, seeds, s); res['trajectory'] = 2; nj.write(path, res); bad = not res['JT-B']['finite']
        return f'{kind}/{path.name}', 'done', ['JT-B non-finite'] if bad else [], bad
    res = rb.job(N, r, seeds, s, 5.0, FILTERS_P5); res['trajectory'] = 2; nj.write(path, res)
    notes = nj.anomaly(res)
    if not res['STEP']['finite']: notes.append('STEP non-finite')
    return f'{kind}/{path.name}', 'done', notes, False


def gate():
    nc.load_clean = FIRST_LOAD; N, r, s = 3, 1, 'noise_bias'; seeds = nc.mc_seeds(N, r); old = json.loads((pc.OLD_MC / f'N{N:02d}_r{r:02d}_{s}.json').read_text()); out = {}
    a = rb.job(N, r, seeds, s, 5.0, FILTERS_P5); b = jtb_job(N, r, seeds, s)
    for name, res in (('STEP', a), ('JT-BHS', a), ('JT-BS', a), ('JT-B', b)):
        x = np.array(res[name]['state']['t>=10']); y = np.array(old[name]['state']['t>=10']); out[name] = float((np.abs(x - y) / y).max())
    out['passed'] = bool(max(v for v in out.values()) <= 1e-5); (pc.OUT / 'paper' / 'GATE_D.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out), flush=True)
    if not out['passed']: raise SystemExit('gate failed')


if __name__ == '__main__':
    (pc.OUT / 'paper').mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ['gate']: gate(); raise SystemExit
    for kind in ('p5', 'JT-B'): (ROOT / f't2_{kind}').mkdir(parents=True, exist_ok=True)
    jobs = [(kind, N, r, s) for kind, N in ORDER for r in range(1, 21) for s in nc.SCEN]
    with Pool(4) as pool:
        for name, status, notes, stop in pool.imap(work, jobs):
            print(name, status, notes, flush=True)
            if stop:
                pool.terminate(); raise SystemExit(f'stop at {name}: {notes}')
