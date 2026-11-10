"""Shared utilities for the standalone nonzero-offset slider benchmark."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,shutil,platform
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.bool_):return bool(x)
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.floating):return float(x)
        return str(x)
    Path(path).write_text(json.dumps(obj,indent=2,default=encode),encoding='utf-8')

def freeze(out,cfg):
    if out.parent.resolve()!=(ROOT/'runs').resolve() or not out.name.replace('_','').replace('-','').isalnum():
        raise ValueError('Use a fresh alphanumeric run tag')
    out.parent.mkdir(exist_ok=True);out.mkdir(exist_ok=False)
    hashes={}
    for path in sorted(ROOT.glob('*.py'))+sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'configs').glob('*.json')):
        rel=path.relative_to(ROOT);hashes[rel.as_posix()]=sha(path)
        dest=out/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    save(out/'FROZEN_PROTOCOL.json',dict(configuration=cfg,source_hashes=hashes,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        version='Slider IMU mounting offset: [-0.15, 0.10, 0.05] m.',
        truth_access='Estimator accepts measurements and prescribed mounting/boundary inputs. Estimates are saved and hashed before scoring. No truth-based initialization or selection.',
        guide='Known point at world origin; estimated direction. The guided point is joint C, not the offset IMU center.',
        selection='Minimum finite measurement residual across predefined starts (the same starts as for a zero IMU offset). Convergence checked separately.',
        settings='Estimator equations, noise seeds, tolerances and weights are the same as for a zero IMU offset.'))
