"""Chain runner: nbody_common.run_step with calibration_fix for Stage I and the compiled
filter (compiled_step_struct) for Stage II. Much faster than nc.run_step; run_mc_d.py gate compares the state RMSEs of the
two (relative tolerance 1e-5)."""
import parity_common as pc
import time
import numpy as np
import compiled_step_struct as cs
import calibration_fix as cf
nc = pc.nc
cal = nc.calibration


def stage2_body(m, b, c, cfg, parent):
    dt = float(np.median(np.diff(m['time']))); n = len(m['time'])
    sd = np.repeat(cfg['initial_state_sd'], 3); noise = np.repeat([cfg['acc_noise_sd'], cfg['gyro_noise_sd'], cfg['ba_rw_sd'], cfg['bg_rw_sd']], 3)
    fix = cfg.get('stage2_fix', {}); midpoint = fix.get('propagation') == 'midpoint'; oc = bool(fix.get('oc_heading', False)) and b == 1
    if b == 1: jp = np.tile(np.asarray(cfg['fixed_support'], float), (n, 1)); jv = np.zeros((n, 3))
    else:
        lp = c['parent_lever']; rp = parent['R']
        jp = parent['p'] + np.einsum('nij,j->ni', rp, lp)
        jv = parent['v'] + np.einsum('nij,nj->ni', rp, np.cross(parent['gyr'] - parent['bg'], lp))
    values = cs.filter_spherical_struct(m[f'acc{b}'], m[f'gyr{b}'], dt, c['p0'], c['v0'], c['R0'], c['lever'], np.asarray(cfg['gravity'], float), sd, noise,
                                        cfg['constraint_sd']['point_p'], cfg['constraint_sd']['point_v'], jp, jv, midpoint, oc, np.asarray(cfg['fixed_support'], float), True)
    h = dict(zip(['p', 'v', 'R', 'ba', 'bg'], values))
    h['acc'] = m[f'acc{b}']; h['gyr'] = m[f'gyr{b}']
    return h


def keep_mask(m, cfg, N, nc_):
    return cal.range_keep([m[f'gyr{b}'][:nc_] for b in range(1, N + 1)], cfg['stage1_gyro_range_rad_s'], cfg['stage1_exclusion_half_width'])


def chain(m, cfg, N):
    dt = float(np.median(np.diff(m['time']))); nc_ = int(round(cfg['calibration_s'] / dt))
    cal.STAGE1_KEEP = keep_mask(m, cfg, N, nc_); excluded = int((~cal.STAGE1_KEEP).sum())
    cals, step, t1 = {}, {}, []
    for b in range(1, N + 1):
        tic = time.perf_counter(); c = cf.calibrate(b, m[f'acc{b}'][:nc_], m[f'gyr{b}'][:nc_], dt, cfg, step.get(b - 1))
        t1.append(time.perf_counter() - tic); cals[b] = c
        step[b] = stage2_body(m, b, c, cfg, step.get(b - 1))
    cal.STAGE1_KEEP = None
    levers = [cals[1]['lever']]
    for b in range(2, N + 1): levers += [cals[b]['parent_lever'], cals[b]['lever']]
    levers = np.array([np.asarray(x, float) for x in levers]); gt = nc.truth_levers(N)
    lever_pct = [float(100 * np.linalg.norm(x - g) / np.linalg.norm(g)) for x, g in zip(levers, gt)]
    return dict(cals=cals, step=step, lever_pct=lever_pct, levers=levers, stage1_s=t1, excluded=excluded, samples=nc_)
