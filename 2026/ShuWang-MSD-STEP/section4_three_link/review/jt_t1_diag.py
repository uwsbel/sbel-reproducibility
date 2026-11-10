"""JT settings shared by d_common.py and jt_stepR.py: the per-body process-noise block QK, the comparison configuration
(cfg, from step/configs/comparison_noise_bias.json) and the JT core module jt_checked (jt)."""
import os
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; DG = HERE.parent
ROOT = DG / 'step'  # RELEASE: path changed for this repository
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
import jt_checked as jt
sa, sg, dtq = 0.01, 0.001, 0.001
QK = np.zeros((9, 9)); I3 = np.eye(3)
QK[:3, :3] = .25 * sa**2 * dtq**4 * I3; QK[:3, 3:6] = .5 * sa**2 * dtq**3 * I3; QK[3:6, :3] = .5 * sa**2 * dtq**3 * I3
QK[3:6, 3:6] = sa**2 * dtq**2 * I3; QK[6:9, 6:9] = sg**2 * dtq**2 * I3
cfg = json.loads((ROOT / 'configs/comparison_noise_bias.json').read_text(encoding='utf-8'))
