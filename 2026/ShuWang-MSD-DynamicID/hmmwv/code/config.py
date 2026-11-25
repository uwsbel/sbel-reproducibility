"""Machine paths of the HMMWV scripts. Edit this file before a run.

CHRONO_DATA_DIR holds the Chrono vehicle files the model reads (a copy is in chrono_data/).
TERRAIN_LIBRARY is the compiled terrain query helper of terrain/ (needed for the rough road only).
OUTPUT_DIR receives records, logs and reports. PYCHRONO_DIR is only needed if pychrono is not importable.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHRONO_DATA_DIR = HERE / "chrono_data"
TERRAIN_LIBRARY = HERE / "terrain" / "build" / "terrain_query.dll"
OUTPUT_DIR = HERE.parent / "output"
PYCHRONO_DIR = None
