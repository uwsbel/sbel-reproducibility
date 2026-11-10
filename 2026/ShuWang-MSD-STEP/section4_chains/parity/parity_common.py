"""Shared setup of the parity/ scripts. Import this module first in every script:
nbody_common sets the BLAS thread variables before numpy loads. nbody/ is read only; its .numba_cache is not used."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
NB = HERE.parent / 'nbody'  # RELEASE: path changed for this repository
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(NB))
import nbody_common as nc
import numpy as np

OUT = HERE / 'out'
OLD_MC = NB / 'out' / 'mc'
P0_DEG = 1.0                                     # initial attitude SD (deg); the nbody/ runs use 5.0


def step_cfg(N, deg):
    c = nc.step_cfg(N)
    if deg != 5.0: c['initial_state_sd'][2] = float(np.radians(deg))
    return c


def motion_sd(deg):
    return nc.STEP_P0 if deg == 5.0 else (nc.STEP_P0[0], nc.STEP_P0[1], float(np.radians(deg)))
