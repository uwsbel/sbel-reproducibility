# Environment

## Check scripts

`check_tables.py`, `make_figures.py` and the scripts in the case folders need Python with numpy and matplotlib. No simulator is needed.

- Checked with Python 3.9.13 (64 bit, Windows), numpy 1.26.4, matplotlib 3.5.2.
- `requirements.txt` lists these two packages.

## Simulation code

The code in `<case>/code/` needs:

- PyChrono (the Python interface of Project Chrono) with the Chrono::Vehicle module for the HMMWV and the Polaris.
- numpy and scipy.
- For the rough-road motions of the vehicles: the terrain helper of `<case>/code/terrain/`, compiled against the same Chrono build. Its path is set in `config.py`.

All machine paths are set in `<case>/code/config.py`.

## Software behind the stored results

The stored results were computed with this software:

- PyChrono compiled by the authors from the Chrono 8.0 sources. The version header of the build reports 8.0.0. The Python modules of the build date from February 2023. The source revision of the build was not recorded.
- Python 3.9.13, numpy 1.26.4, scipy 1.13.1.
- Operating system: Windows 11, 64 bit. Processor: Intel Xeon E-2176M (6 cores).

On this build the code in `<case>/code/` reproduces the stored records, predictions and costs (see `README.md`).

## Other Chrono versions

The code was run only on the build above. It uses these parts of the Chrono 8.0 interface:

- the powertrain interface `ChPowertrain` of Chrono::Vehicle. Later Chrono versions replaced it with separate engine and transmission classes;
- the vector and quaternion classes `ChVectorD` and `ChQuaternionD`, which later versions renamed;
- the JSON vehicle and tire readers of Chrono::Vehicle, and the TMEasy tire.

With another version these calls may need small changes, and the simulated numbers will differ from the stored ones.
