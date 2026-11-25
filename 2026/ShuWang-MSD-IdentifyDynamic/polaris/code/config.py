"""Machine paths of the Polaris code. This is the only file a reader needs to edit.

CHRONO_DATA holds the Polaris vehicle files that the model reads (shipped in chrono_data/).
WORK_DIR receives generated vehicle files, records, logs and reports.
TERRAIN_LIBRARY is the compiled terrain query helper (see terrain/BUILD.md); only the uneven road needs it.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHRONO_DATA = HERE / "chrono_data"
WORK_DIR = HERE.parent / "work"
TERRAIN_LIBRARY = HERE / "terrain" / "build" / "exact_query.dll"
