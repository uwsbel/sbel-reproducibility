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

- Project Chrono, development version: commit `f022f9b38a92bf479a504e447547568e42985cd4` of https://github.com/projectchrono/chrono (main branch, 26 January 2023). Its version header reads 8.0.0.
- Python 3.9.13, numpy 1.26.4, scipy 1.13.1.
- Operating system: Windows 11, 64 bit. Processor: Intel Xeon E-2176M (6 cores).

On this version the code in `<case>/code/` reproduces the stored records, predictions and costs (see `README.md`).

## Building this Chrono version

1. Clone https://github.com/projectchrono/chrono and check out commit `f022f9b38a92bf479a504e447547568e42985cd4`.
2. Configure with CMake: `ENABLE_MODULE_VEHICLE=ON`, `ENABLE_MODULE_PYTHON=ON`, `ENABLE_OPENMP=ON`. Other modules and the demos are not needed.
3. Build the Release targets `_core` and `_vehicle`. Put the folder of the built `pychrono` package on the Python path.

We built it on Windows with Visual Studio 2022 (x64, Release), CMake 3.25, Eigen 3.4.0, SWIG 4.1.1 and Python 3.9. A fresh build of this commit reproduces the stored replays of the three cases within a relative difference of 1e-9. A different compiler can change the last digits.

## Other Chrono versions

The code uses these parts of the interface of this version:

- the powertrain interface `ChPowertrain` of Chrono::Vehicle. Later Chrono versions replaced it with separate engine and transmission classes;
- the vector and quaternion classes `ChVectorD` and `ChQuaternionD`, which later versions renamed;
- the JSON vehicle and tire readers of Chrono::Vehicle, and the TMeasy tire.

Other versions also differ in behavior. The official 8.0.0 release has an older TMeasy tire. Later versions have a newer TMeasy tire, which computes other forces at low speed, and an automatic transmission with another shift rule. With these versions the vehicle simulations differ from the stored ones. To use the method with another version, generate the records with the teacher parameters on that version first.
