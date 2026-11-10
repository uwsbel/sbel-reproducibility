"""One timing set for the paper, N = 3, 5, 10. Protocol: numba, one BLAS thread, one process, ten full-record runs alternated,
median and IQR, excluding compilation, warm-up, I/O and scoring. Paper draw, noise_bias, first trajectory, attitude P0 = 1 deg. Stage II: STEP-S, JT-BHS, JT-BS, truth-start JT-B.
Stage I: reference calibrate, three runs, parents' Stage II outside the timed intervals. Parents' Stage II on [0, 10] s: STEP-S, three runs. Also the largest
difference of STEP-S from the reference implementation. Writes out/paper/timing/TIMING_P.json; refuses to overwrite. argv: smoke (code check only, N = 3, one repetition, SMOKE_P.json; not used for the paper)."""
import parity_common as pc
import json, time, platform, subprocess, sys
import numpy as np
import numba
import jt_nbody_struct as js
import compiled_step_struct as cs
nc = pc.nc

SMOKE = 'smoke' in sys.argv[1:]
OUT = pc.OUT / 'paper' / 'timing'; OUT.mkdir(parents=True, exist_ok=True); FILE = OUT / ('SMOKE_P.json' if SMOKE else 'TIMING_P.json')
if FILE.exists() and not SMOKE: raise SystemExit(f'{FILE} exists; refusing to overwrite')
REPS = 1 if SMOKE else 10; REPS_S1 = 1 if SMOKE else 3
out = dict(software=dict(python=sys.version, numba=numba.__version__, numpy=np.__version__, platform=platform.platform(), processor=platform.processor()), p0_deg=pc.P0_DEG, by_N={})


def python_count():
    return subprocess.run('tasklist', capture_output=True, text=True).stdout.lower().count('python')


out['python_processes_at_start'] = python_count(); out['start'] = time.strftime('%Y-%m-%d %H:%M:%S')
for N in ((3,) if SMOKE else (3, 5, 10)):
    clean, ref = nc.load_clean(N); sc, _ = nc.draw(clean, N, nc.paper_seeds(N)); m = sc['noise_bias']; cfg = pc.step_cfg(N, pc.P0_DEG); sd = pc.motion_sd(pc.P0_DEG)
    n = len(m['time']); dt = float(np.median(np.diff(m['time']))); n1 = int(round(cfg['calibration_s'] / dt)); res = dict(samples=n); stage1 = []
    for rep in range(REPS_S1):
        nc.calibration.STAGE1_KEEP = nc.calibration.range_keep([m[f'gyr{b}'][:n1] for b in range(1, N + 1)], cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])
        cals, step, tb = {}, {}, []
        for b in range(1, N + 1):
            tic = time.perf_counter(); cal = nc.calibration.calibrate(b, m[f'acc{b}'][:n1], m[f'gyr{b}'][:n1], dt, cfg, step.get(b - 1)); tb.append(time.perf_counter() - tic)
            cals[b] = cal; step[b] = nc.core.reconstruct(b, m[f'acc{b}'], m[f'gyr{b}'], dt, cal, cfg, step.get(b - 1))
        nc.calibration.STAGE1_KEEP = None; stage1.append(sum(tb)); print(f'N={N} Stage I rep {rep + 1}: {sum(tb):.2f} s', flush=True)
    res['stage1_s'] = dict(times=stage1, median=float(np.median(stage1)), by_body_last=tb)
    hs = cs.run_chain(m, cals, cfg); diffs = [float(np.max(np.abs(x - step[b][k]))) for b, h in hs.items() for k, x in h.items() if k in step[b]]
    res['STEP_S_vs_reference_max'] = max(diffs); print(f'N={N} STEP-S vs reference: {max(diffs):.2e}', flush=True)
    mt = {k: v[:n1] for k, v in m.items()}; cfgp = dict(cfg, n_bodies=N - 1); tp = []
    cs.run_chain(mt, cals, cfgp)
    for rep in range(3):
        tic = time.perf_counter(); cs.run_chain(mt, cals, cfgp); tp.append(time.perf_counter() - tic)
    res['parents_stage2_10s_s'] = dict(times=tp, median=float(np.median(tp)))
    r0, v0, q0 = nc.stage1_states(cals, N); levers = [cals[1]['lever']]
    for b in range(2, N + 1): levers += [cals[b]['parent_lever'], cals[b]['lever']]
    lev = np.array([np.asarray(x, float) for x in levers])
    _, G = nc.target(N, ref); rt, vt, qt = nc.true_states(N, ref, G); gt = nc.truth_levers(N); pt, pf = js.h_pattern(N, True), js.h_pattern(N, False)
    calls = {'STEP-S': lambda: cs.run_chain(m, cals, cfg),
             'JT-BHS': lambda: js.run(m, N, r0, v0, q0, lev, False, True, 0.0, sd, pattern=pf),
             'JT-BS': lambda: js.run(m, N, r0, v0, q0, lev, True, False, 1.0, sd, pattern=pt),
             'JT-B': lambda: js.run(m, N, rt, vt, qt, gt, True, False, 1.0, None, pattern=pt)}
    for f in calls.values(): f()
    times = {k: [] for k in calls}
    for i in range(REPS):
        for k, f in calls.items():
            tic = time.perf_counter(); f(); times[k].append(time.perf_counter() - tic)
        print(f'N={N} repetition {i + 1}', {k: round(v[-1], 2) for k, v in times.items()}, flush=True)
    base = np.median(times['STEP-S'])
    for k, x in times.items():
        x = np.array(x); q1, q3 = np.percentile(x, [25, 75])
        res[k] = dict(times_s=x.tolist(), median_ms=float(np.median(x) / n * 1e3), q1_ms=float(q1 / n * 1e3), q3_ms=float(q3 / n * 1e3), min_ms=float(x.min() / n * 1e3), max_ms=float(x.max() / n * 1e3), ratio_median_to_STEP_S=float(np.median(x) / base))
    res['python_processes_after'] = python_count(); out['by_N'][str(N)] = res
    FILE.write_text(json.dumps(out, indent=1))
out['end'] = time.strftime('%Y-%m-%d %H:%M:%S'); FILE.write_text(json.dumps(out, indent=1))
print('done')
