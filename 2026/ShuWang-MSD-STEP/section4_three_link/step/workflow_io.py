import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
