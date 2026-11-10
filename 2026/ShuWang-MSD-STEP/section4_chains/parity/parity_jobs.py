"""One (N, draw, scenario, attitude P0) job: STEP and the filters started from its Stage I.
Same measures as nbody_jobs.job. Truth-start JT-B is not run; it comes from nbody/run_mc.py. Non-finite filters are recorded."""
import parity_common as pc
import time
import numpy as np
import nbody_jobs as nj
import jt_nbody_struct as js
nc = pc.nc

SPEC = {'JT-BHS': (False, True, 0.0), 'JT-BS': (True, False, 1.0), 'JT-BS-OC': (True, True, 1.0)}     # est, oc, lever_sd


def job(N, tag, seeds, s, deg, filters=('JT-BHS', 'JT-BS', 'JT-BS-OC')):
    clean, ref = nc.load_clean(N); sc, bias = nc.draw(clean, N, seeds); m = sc[s]
    t, G = nc.target(N, ref); tm = m['time']; mask = tm >= nc.T_EVAL; cfg = pc.step_cfg(N, deg); sd = pc.motion_sd(deg)

    def truth(b, key):
        return bias[f'{key}{b}'] if s == 'noise_bias' else np.zeros(3)
    res = dict(N=N, draw=tag, scenario=s, seeds={str(b): int(v) for b, v in seeds.items()}, p0_deg=deg)

    def split(hb):
        hs, ts, series = [], [], []
        for b in range(1, N + 1):
            a, c, h = nc.heading_tilt(hb(b)['R'], t[f'R{b}'], mask); hs.append(a); ts.append(c); series.append(h)
        return dict(heading=hs, tilt=ts, common_heading=float(np.sqrt(np.mean(np.mean(series, axis=0)[mask] ** 2))))

    tic = time.perf_counter(); st = nc.run_step(m, cfg, N); wall = time.perf_counter() - tic
    fits = [st['cals'][b]['fit'] for b in range(1, N + 1)]
    res['STEP'] = dict(lever_pct=st['lever_pct'], state=nc.rows_step(st['step'], t, tm, N), excluded=st['excluded'], stage1_s=st['stage1_s'], wall_s=wall,
                       converged=[bool(f['converged']) for f in fits], selected_start=[int(f['selected_start']) for f in fits],
                       bias=nj.bias_rows(lambda b, key: st['step'][b][key], truth, N, mask), **split(lambda b: st['step'][b]))
    r0, v0, q0 = nc.stage1_states(st['cals'], N); lev = st['levers']
    pattern = {True: js.h_pattern(N, True), False: js.h_pattern(N, False)}
    for name in filters:
        est, oc, lsd = SPEC[name]; tic = time.perf_counter()
        try:
            h = js.run(m, N, r0, v0, q0, lev, est, oc, lsd, sd, pattern=pattern[est])
        except FloatingPointError as err:
            res[name] = dict(finite=False, error=str(err)); continue
        fin = bool(all(np.isfinite(h[k]).all() for k in 'rvqbL'))
        rec = dict(finite=fin, wall_s=time.perf_counter() - tic)
        if fin:
            rec['state'] = nc.rows_jt(h, t, tm, N)
            if est: rec['lever_pct'] = nc.jt_lever_pct(h, N)

            def estb(b, key, h=h):
                o = 6 * (b - 1) + (0 if key == 'ba' else 3)
                return h['b'][:, o:o + 3]
            rec['bias'] = nj.bias_rows(estb, truth, N, mask); rec.update(split(lambda b: nc.jt_body(h, b)))
        res[name] = rec
    return res
