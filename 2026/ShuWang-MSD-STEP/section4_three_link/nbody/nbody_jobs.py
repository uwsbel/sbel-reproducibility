"""One (N, draw, scenario) job: STEP, JT-BHS, JT-BS, truth-start JT-B. Non-finite filters are recorded,
not removed."""
import nbody_common as nc
import json, time
import numpy as np
import jt_nbody_struct as js


def bias_rows(est, truth, N, mask):
    out = {}
    for b in range(1, N + 1):
        for key in ('ba', 'bg'):
            e = est(b, key)[mask] - truth(b, key)
            out[f'body{b}|{key}'] = dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))), truth_norm=float(np.linalg.norm(truth(b, key))))
    return out


def job(N, tag, seeds, s, hist_dir=None):
    clean, ref = nc.load_clean(N); sc, bias = nc.draw(clean, N, seeds); m = sc[s]
    t, G = nc.target(N, ref); tm = m['time']; mask = tm >= nc.T_EVAL; cfg = nc.step_cfg(N)

    def truth(b, key):
        return bias[f'{key}{b}'] if s == 'noise_bias' else np.zeros(3)
    res = dict(N=N, draw=tag, scenario=s, seeds={str(b): int(v) for b, v in seeds.items()})

    def split(hb):
        hs, ts, series = [], [], []
        for b in range(1, N + 1):
            a, c, h = nc.heading_tilt(hb(b)['R'], t[f'R{b}'], mask); hs.append(a); ts.append(c); series.append(h)
        return dict(heading=hs, tilt=ts, common_heading=float(np.sqrt(np.mean(np.mean(series, axis=0)[mask] ** 2))))

    tic = time.perf_counter(); st = nc.run_step(m, cfg, N); wall = time.perf_counter() - tic
    fits = [st['cals'][b]['fit'] for b in range(1, N + 1)]
    res['STEP'] = dict(lever_pct=st['lever_pct'], state=nc.rows_step(st['step'], t, tm, N), excluded=st['excluded'], stage1_s=st['stage1_s'], wall_s=wall,
                       converged=[bool(f['converged']) for f in fits], selected_start=[int(f['selected_start']) for f in fits],
                       bias=bias_rows(lambda b, key: st['step'][b][key], truth, N, mask), **split(lambda b: st['step'][b]))
    if hist_dir is not None:
        np.savez_compressed(hist_dir / f'N{N:02d}_{s}_STEP.npz', **{f'{k}{b}': st['step'][b][k] for b in range(1, N + 1) for k in ('p', 'v', 'R', 'ba', 'bg')})
    r0, v0, q0 = nc.stage1_states(st['cals'], N); lev = st['levers']
    rt, vt, qt = nc.true_states(N, ref, G); gt = nc.truth_levers(N)
    runs = [('JT-BHS', (r0, v0, q0, lev, False, True, 0.0, nc.STEP_P0)), ('JT-BS', (r0, v0, q0, lev, True, False, 1.0, nc.STEP_P0)),
            ('JT-B', (rt, vt, qt, gt, True, False, 1.0, None))]
    pattern = {True: js.h_pattern(N, True), False: js.h_pattern(N, False)}
    for name, a in runs:
        tic = time.perf_counter()
        try:
            h = js.run(m, N, *a, pattern=pattern[a[4]])
        except FloatingPointError as err:
            res[name] = dict(finite=False, error=str(err)); continue
        fin = bool(all(np.isfinite(h[k]).all() for k in 'rvqbL'))
        rec = dict(finite=fin, wall_s=time.perf_counter() - tic)
        if fin:
            rec['state'] = nc.rows_jt(h, t, tm, N)
            if a[4]: rec['lever_pct'] = nc.jt_lever_pct(h, N)

            def est(b, key, h=h):
                o = 6 * (b - 1) + (0 if key == 'ba' else 3)
                return h['b'][:, o:o + 3]
            rec['bias'] = bias_rows(est, truth, N, mask); rec.update(split(lambda b: nc.jt_body(h, b)))
        res[name] = rec
        if hist_dir is not None and name in ('JT-BHS', 'JT-BS', 'JT-B'):
            np.savez_compressed(hist_dir / f'N{N:02d}_{s}_{name}.npz', **{k: h[k] for k in 'rvqbL'})
    return res


def anomaly(res):
    """Anomalies that a job can show by itself: Stage I not converged, or a non-finite filter."""
    out = []
    if not all(res['STEP']['converged']): out.append('Stage I not converged')
    for k, v in res.items():
        if isinstance(v, dict) and 'finite' in v and not v['finite']: out.append(f'{k} non-finite')
    return out


def write(path, res):
    path.write_text(json.dumps(res, indent=1))
