"""Generate the clean N-body record once; refuses to overwrite. argv: N. Writes data/N{N:02d}/clean and SOLVER.json."""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import nbody_sim as ns

N = int(sys.argv[1])
dest = HERE / 'data' / f'N{N:02d}' / 'clean'
if dest.exists(): raise SystemExit(f'{dest} exists; refusing to overwrite')
tic = time.perf_counter()
m, ref, ys, info = ns.simulate(N)
info.update(seconds=time.perf_counter() - tic, N=N, y0=ns.initial_state(N).tolist())
dest.mkdir(parents=True)
np.savez_compressed(dest / 'measurements.npz', **m); np.savez_compressed(dest / 'reference.npz', **ref)
np.savez_compressed(dest / 'generalized.npz', time=m['time'], y=ys)
(dest / 'SOLVER.json').write_text(json.dumps(info, indent=1))
print('generated', N, info, flush=True)
