# International Conference on Intelligent Robots and Systems (IROS)

This folder holds code, models, assets, etc. used for the IROS 2020 paper titled _SynChrono: A Scalable, Physics-Based Simulation Platform For Testing Groups of Autonomous Vehicles and/or Robots_.

## Table of Contents
- [Setup Guide](#setup-guide)
  - [ProjectChrono](#projectchrono-and-submodules)
- [Running](#running)
- [See Also](#see-also)
- [Support](#support)

## Setup Guide

The results presented in this paper used [ProjectChrono](http://www.projectchrono.org/), specifically its Chrono::Vehicle, Chrono::Sensor and SynChrono submodules. The scaling results can be reproduced using just the Chrono::Vehicle and SynChrono modules, but all visuals of the environment presented in the paper used Chrono::Sensor which introduces some additional dependences. Similar, but much lower quality visuals can be produced with the Chrono::Irrlicht module instead of Chrono::Sensor.

#### ProjectChrono and Submodules

To clone:
```bash
$ git clone https://github.com/projectchrono/chrono.git && cd chrono
```

The installation guide for Chrono can be found [here](http://api.projectchrono.org/tutorial_install_chrono.html). At the time of submission, SynChrono is not included as a part of Chrono, but it will be incorporated into Chrono as a module by the October publication date.

When using CMake to build Chrono, ensure that the Chrono::Vehicle and SynChrono modules are selected, and optionally the Chrono::Sensor or Chrono::Irrlicht modules. Chrono::Sensor requires an Nvidia GPU Maxwell or newer along with CUDA and the OptiX ray-tracing engine. Chrono::Irrlicht requires Irrlicht installed. Information for both of these dependencies can be found in the installation guides on the Project Chrono website.

## Running

_This section assumes the previous installation steps have been successfully completed_.

The Park St. demo (the intersection show in the paper) unfortunately requires a 3rd party mesh that is not able to be publicly distributed. The scaling results however can be generated using the demo_SYN_platoon scenario and the demo_chrono_platoon scenario.

To produce outputs of the fraction of real time for a particular number of vehicles `n`, run the two scenarios like:

```bash
$ mpirun -n <n + 1> /path/to/demo_SYN_platoon --verbose
$ ./path/to/demo_chrono_platoon <n>
```

A word of caution: while the general scaling trends should always hold, the specific numbers are very dependent on the version of the code, the hardware it is running on, the MPI implementation and other factors. Do not expect to reproduce the exact numbers shown in the paper, only the general scaling trend. In addition, the demo_SYN_platoon should be run across several computing nodes, connected via MPI to achieve proper scaling. While good performance for a limited number of ranks may be achieved on a single computing node, eventually that computing node will become saturated.

## See Also

Videos: [link](https://uwmadison.box.com/s/b1afaj0usjn4nvkanbjk23xqa14m5ouh).

## Support

Contact [Jay Taves](jtaves@wisc.edu) for any questions or concerns regarding the contents of this folder.
