# Autonomous Off-Road Gator Policy

_Last Update_: Saturday, November 27th, 2021

## Table of Contents
- [Setup Guide](#setup-guide)
  - [Publications Data](#publications-data)
  - [ProjectChrono](#projectchrono-and-pychrono)
  - [GymChrono](#gymchrono)
  - [Additional Requirements](#additional-requirements)
- [Running](#running)
- [See Also](#see-also)
- [Support](#support)

## Setup Guide
The results presented in the aforementioned paper utilized [ProjectChrono](http://www.projectchrono.org/) and it's python wrapped bindings [PyChrono](http://www.projectchrono.org/pychrono/). Furthermore, the demonstration environment is provided in [GymChrono](https://github.com/projectchrono/gym-chrono).

Python3.6 is required, as it is the only version compatible with the version of PyChrono, PyTorch and Stable Baselines we will be using. _Recommendation_: Use an [Anaconda](https://anaconda.org/) environment.

#### Publications Data

To clone:
```bash
$ git clone https://github.com/uwsbel/publications-data.git
```

#### ProjectChrono and PyChrono
Commit: [cb6d623](https://github.com/projectchrono/chrono/tree/cb6d623dfcee6078b1c3c6dcb7a37a1bd7411cea)

To clone:
```bash
$ git clone https://github.com/projectchrono/chrono.git && cd chrono
$ git checkout cb6d623dfcee6078b1c3c6dcb7a37a1bd7411cea
```

The installation guide for PyChrono can be found [here](http://api.projectchrono.org/development/pychrono_installation.html). At the time of submission, the anaconda installation does not contain the sensor module required for the demonstration environment. As a result, please follow the installation sequence titled _Build Python modules from the C++ API_ as seen in the previous link.

Additionally, to test/train on a height map with deformable SCM terrain, you have to complete a method marked as TODO in the SCMDeformableTerrain.cpp file. Please add the following lines to the function as seen on [GitHub](https://github.com/projectchrono/chrono/blob/cb6d623dfcee6078b1c3c6dcb7a37a1bd7411cea/src/chrono_vehicle/terrain/SCMDeformableTerrain.cpp#L535):
```cpp
double height = std::min_element(m_trimesh_shape->GetMesh()->getCoordsVertices().begin(),  //
                                     m_trimesh_shape->GetMesh()->getCoordsVertices().end(),    //
                                     [loc](const ChVector<>& a, const ChVector<>& b)           //
                                     { return (a - loc).Length() < (b - loc).Length(); })      //
                        ->z();
```

#### GymChrono
Commit: [8d636c6](https://github.com/Benatti1991/gym-chrono/tree/8d636c62d0a5d4b529162891ea783e8eb34c77cf)

To clone:
```bash
$ git clone https://github.com/Benatti1991/gym-chrono.git && cd gym-chrono
$ git checkout 8d636c62d0a5d4b529162891ea783e8eb34c77cf
```

To install GymChrono, please follow the installation guide found at the bottom of the README ([link](https://github.com/Benatti1991/gym-chrono/tree/8d636c62d0a5d4b529162891ea783e8eb34c77cf#installation)).

#### Additional Requirements

- __PyTorch__: [Installation Guide](https://pytorch.org/)
  - _Version_: 1.3.1
  - _Example Command_: `conda install pytorch=1.3.1 torchvision -c pytorch`
- __Tensorflow__
  - _Version_: 1.14
  - _Example Command_: `conda install tensorflow-gpu==1.14`
- __Stable Baselines__: [Installation Guide](https://github.com/hill-a/stable-baselines#installation)

## Running

_This section assumes the previous installation steps have been successfully completed_.

In this folder, the saved model weights used in the paper are provided in the file `gator_2021.pth`. As a result, there is no need to retrain the environment. To test the policy and visualize the simulation, please run the following command.
```bash
python runner.py gym_chrono.envs:robot_learning-v0 gator_2021.pth
```

If it is desired to train policy using the demonstration environment, please run the following command.
```bash
python runner.py gym_chrono.envs:robot_learning-v0 <model file path>.pth --train
```
__Note__: To use the provided weights in the training processes, replace `<model file path>` with `gator_2021.pth`.

## See Also

Videos: [link](https://uwmadison.box.com/s/0vin8yddy5q2zhj9wpwgmqvw4g07ll9j).

## Support

Contact [Aaron Young](aryoung5@wisc.edu) for any questions or concerns regarding the contents of this folder.
