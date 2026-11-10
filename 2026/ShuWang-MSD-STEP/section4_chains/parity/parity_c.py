"""Where the chain records of parity/ live and which seeds they use.
Trajectory 1 is the motion of nbody/ (Motion 1; its records are in nbody/data). Trajectory 2 (Motion 2, records in
data_c/t2) comes from default_rng(20261104)."""
import parity_common as pc
import numpy as np
nc = pc.nc
DATA = pc.HERE / 'data_c'


def clean_dir(traj, N):
    return DATA / f't{traj}' / f'N{N:02d}' / 'clean'


def load_clean(traj, N):
    with np.load(clean_dir(traj, N) / 'measurements.npz') as z: m = {k: z[k] for k in z.files}
    with np.load(clean_dir(traj, N) / 'reference.npz') as z: ref = {k: z[k] for k in z.files}
    return m, ref


def seeds(traj, N, r):
    return nc.mc_seeds(N, r) if traj == 1 else {b: 20263000 + 10 * r + b for b in range(1, N + 1)}


def initial_state_t2(N):
    """Same ranges and draw order as nbody_sim.initial_state; the first N bodies of the draw. N = 3 follows this rule too."""
    rng = np.random.default_rng(20261104)
    th = np.deg2rad(rng.uniform(30, 60, 10)); ph = rng.uniform(.1, .3, 10); thd = rng.uniform(.2, .5, 10); phd = rng.uniform(.2, .5, 10)
    y = np.zeros(4 * N)
    y[0::2][:N] = th[:N]; y[1::2][:N] = ph[:N]
    y[2 * N::2] = thd[:N]; y[2 * N + 1::2] = phd[:N]
    return y
