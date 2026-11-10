"""One (N, draw, scenario, attitude P0) job: STEP through parity_fix.chain (Stage I as calibration.calibrate, compiled
Stage II), and the coupled filters started from that Stage I. Same draws and measures as run_mc.py. Used by run_mc_c_base.py
(Motion 2, 1 deg) and run_mc_d.py (Motion 2, 5 deg)."""
import parity_common as pc
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import nbody_jobs as nj
import jt_nbody_struct as js
import parity_jobs as pj
import parity_fix as pf
nc = pc.nc


def job(N, tag, seeds, s, deg, filters):
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

    tic = time.perf_counter(); st = pf.chain(m, cfg, N); wall = time.perf_counter() - tic
    fits = [st['cals'][b]['fit'] for b in range(1, N + 1)]
    fin = bool(all(np.isfinite(st['step'][b][k]).all() for b in st['step'] for k in ('p', 'v', 'R', 'ba', 'bg')))
    res['STEP'] = dict(finite=fin, lever_pct=st['lever_pct'], excluded=st['excluded'], stage1_s=st['stage1_s'], wall_s=wall,
                       converged=[bool(f['converged']) for f in fits], selected_start=[int(f['selected_start']) for f in fits],
                       R0_deg=[float(np.degrees(np.linalg.norm(R.from_matrix(st['cals'][b]['R0'] @ t[f'R{b}'][0].T).as_rotvec()))) for b in range(1, N + 1)])
    if fin:
        res['STEP'].update(state=nc.rows_step(st['step'], t, tm, N), bias=nj.bias_rows(lambda b, key: st['step'][b][key], truth, N, mask), **split(lambda b: st['step'][b]))
    r0, v0, q0 = nc.stage1_states(st['cals'], N); lev = st['levers']
    pattern = {True: js.h_pattern(N, True), False: js.h_pattern(N, False)}
    for name in filters:
        est_l, oc, lsd = pj.SPEC[name]; tic = time.perf_counter()
        try:
            h = js.run(m, N, r0, v0, q0, lev, est_l, oc, lsd, sd, pattern=pattern[est_l])
        except FloatingPointError as err:
            res[name] = dict(finite=False, error=str(err)); continue
        ok = bool(all(np.isfinite(h[k]).all() for k in 'rvqbL'))
        rec = dict(finite=ok, wall_s=time.perf_counter() - tic)
        if ok:
            rec['state'] = nc.rows_jt(h, t, tm, N)
            if est_l: rec['lever_pct'] = nc.jt_lever_pct(h, N)

            def estb(b, key, h=h):
                o = 6 * (b - 1) + (0 if key == 'ba' else 3)
                return h['b'][:, o:o + 3]
            rec['bias'] = nj.bias_rows(estb, truth, N, mask); rec.update(split(lambda b: nc.jt_body(h, b)))
        res[name] = rec
    return res
