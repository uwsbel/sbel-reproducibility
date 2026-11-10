"""Chain records of trajectory 2 (Motion 2), N = 3, 5, 10. Refuses to overwrite.
Check: the same path with the initial state of nbody/ (Motion 1) reproduces nbody/data/N05/clean. Writes out/fixc/GEN_GATE.json."""
import parity_c as pcx
import parity_common as pc
import json, time
from multiprocessing import Pool
import numpy as np
import nbody_sim as ns
nc = pc.nc
ORIG = ns.initial_state


def gen(a):
    traj, N = a; dest = pcx.clean_dir(traj, N)
    if dest.exists(): raise SystemExit(f'{dest} exists; refusing to overwrite')
    ns.initial_state = ORIG if traj == 1 else pcx.initial_state_t2
    tic = time.perf_counter(); m, ref, ys, info = ns.simulate(N)
    info.update(seconds=time.perf_counter() - tic, N=N, trajectory=traj, y0=ns.initial_state(N).tolist())
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'measurements.npz', **m); np.savez_compressed(dest / 'reference.npz', **ref)
    np.savez_compressed(dest / 'generalized.npz', time=m['time'], y=ys)
    (dest / 'SOLVER.json').write_text(json.dumps(info, indent=1))
    return traj, N, round(info['seconds'], 1)


def gate():
    ns.initial_state = ORIG; m, ref, _, _ = ns.simulate(5); old_m, old_ref = nc.load_clean(5)
    d = max(float(np.abs(m[k] - old_m[k]).max()) for k in old_m); e = max(float(np.abs(ref[k] - old_ref[k]).max()) for k in old_ref)
    return dict(measurements_max=d, reference_max=e, passed=bool(max(d, e) <= 1e-6))


def run(a):
    return gate() if a == 'gate' else gen(a)


if __name__ == '__main__':
    (pc.OUT / 'fixc').mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool: res = pool.map(run, ['gate', (2, 3), (2, 5), (2, 10)], chunksize=1)
    (pc.OUT / 'fixc' / 'GEN_GATE.json').write_text(json.dumps(res[0], indent=1)); print('gate', res[0], flush=True); print('generated', res[1:], flush=True)
    if not res[0]['passed']: raise SystemExit('generator gate failed')
