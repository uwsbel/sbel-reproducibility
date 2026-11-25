"""Machine settings of the PUMA code. Edit this file before the first run.

All other files take their paths from here. The PUMA model uses only the
multibody core of Chrono, so no Chrono data folder is needed.
"""
import os
import sys
from pathlib import Path

# Folder that holds the pychrono package. None: pychrono is already on the Python path.
PYCHRONO_DIR = None

# Folder for generated records, run logs and reports.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# The case folder with settings/, geometry/, teacher/, results/ and signals/.
CASE_DIR = Path(__file__).resolve().parent.parent

# One thread for the numerical libraries; set before numpy is imported.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

if PYCHRONO_DIR is not None and str(PYCHRONO_DIR) not in sys.path:
    sys.path.insert(0, str(PYCHRONO_DIR))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))  # for ../cost.py
