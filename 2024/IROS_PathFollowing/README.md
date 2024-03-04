# Instruction for Reproducing Simulation Experiment Results for paper titled "A Study on the Transferability and Effectiveness of Path Following Control Policies Synthesized in Simulation"

## Requirement

- NVIDIA GPU
- Docker (24.0.6 or above)
    - for Linux user, you will need to download [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Docker Container Setup Instruction

1. Pull our docker image

    - In terminal run: 
    
        ```docker pull uwsbel/ros:iros2024```
    
    - To check if docker image got pulled correctly, run ```docker images```, there should be an entry named and labeled ```uwsbel/ros:iros2024```

2. Spin a docker container using pulled image

    - In terminal run:

        ```docker run -d --gpus=all -p 5901:5901 -p 6901:6901 uwsbel/ros:iros2024```

    - To check if docker container is working fine, run ```docker ps```, there should be a container entry based on the previous image you pulled

    - If you did previous step, open your browser and go to the link: http://localhost:6901/. We added VNC support for better running and visualizing the simulation.

## Running Simulation Experiments
In your website browser (http://localhost:6901/), you should be able to access a GUI-like desktop, which is the container we prepare for you to run simulation (**Note, the following command should be running inside this desktop in the above website**). Simulation demos needs two parts in our design: (1) the CHRONO simulation with ROS2 communication support; (2) a ROS2 node that subscribing information from CHRONO simulation, running different control policies, and then publishing control command back to CHRONO simulation to form feedback loop.

1. Run CHRONO Simulation Demo

    - Click Terminal to open a terminal, go to directory: ```cd /sbel/Desktop/chrono/build/bin/```, and run:

        ```./demo_ROS_dART <reference_path_file>```
        
        Here, ```<reference_path_file> ```could be ```path_iros_1.csv``` or ```path_iros_2.csv``` or ```lot17_sinsquare.csv```, which the csv files storing information of different reference trajectory. 

2. Open another terminal or if you are familiar with tmux you can only use one terminal window. Go to ROS2 workspace directory: ```cd /sbel/Desktop/ros_ws```, source the ROS2 executable by running: ```source install/setup.bash```.  

    - Then running ROS2 node by:

        ```ros2 run path_follower path_follower --ros-args -p controller:=<controller_type> -p ref_path:=<reference_path>```

        - ```<controller_type>``` could be ```mpc``` or ```nnmpc``` or ```pid``` or ```nnhd```, which stands for the four types of controllers mentioned in the paper
        - ```<reference_path> ```could be ```path_iros_1``` or ```path_iros_2``` or ```lot17_sinsquare```. **(*Important*: this need to be same as the reference path you select to run simulation in previous step)**
