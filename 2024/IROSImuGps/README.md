# Quantifying the Sim2real Gap for GPS and IMU Sensors
This repository holds the source code to reproduce results obtained in the IROS submission "Quantifying the Sim2real Gap for GPS and IMU Sensors".

### Using the Chrono Demo 

We set up a Chrono demo based on the three trajectories we describe in the paper. The demo sets up the dART vehicle equipped with 3 GPS sensors with zero noise, normal noise, and random walk based noise. Each GPS publishes to 3 different ROS2 topics to account for covariance - zero dynamics, default AirSim parameters, and calibrated AirSim parameters. 

To use the demo, follow the instructions below - 

### Requirement

- NVIDIA GPU
- Docker (24.0.6 or above)
    - for Linux user, you will need to download [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Docker Container Setup Instruction

1. Pull our docker image

    - In terminal run: 
    
        ```docker pull iros2024:latest```
    
    - To check if docker image got pulled correctly, run ```docker images```, there should be an entry named and labeled ```iros2024:latest```

2. Spin a docker container using pulled image

    - In terminal run:

        ```docker run -d --gpus=all -p 5901:5901 -p 6901:6901 iros2024:latest```

    - To check if docker container is working fine, run ```docker ps```, there should be a container entry based on the previous image you pulled

    - If you did previous step, open your browser and go to the link: http://localhost:6901/. **Password is: "sbel"** We added VNC support for better running and visualizing the simulation.
  
### Running the demo
In your website browser (http://localhost:6901/), you should be able to access a GUI-like desktop, which is the container we prepare for you to run simulation (**Note, the following command should be running inside this desktop in the above website**). Simulation demos needs two parts in our design: (1) the CHRONO simulation with ROS2 communication support; (2) a ROS2 node that subscribing information from CHRONO simulation, running different control policies, and then publishing control command back to CHRONO simulation to form feedback loop.

1. Run CHRONO Simulation Demo

    - Click Terminal to open a terminal, go to directory: ```cd /sbel/Desktop/chrono/build/bin/```, and run:

        ```./demo_ROS_dART <trajectory>```
        
        Here, ```<trajectory> ```could be ```straight``` or ```half_sine``` or ```circle```.  

   The 3 trajectories used to generate the results for this paper are stored in ```/sbel/Desktop/chrono/build/data/trajectories```. 

   The demonstration is configured to receive control inputs (steering and throttle), and is designed to accommodate any controller. 

### Reproducing Plots and Data Analysis
All the scripts used to generate the plots and tables from the paper are present in the `scripts` directory. To run the scripts, first download the `data2` folder from the following link: [Data](https://drive.google.com/drive/folders/1t3GVovJ4nNwaoIZhAmNQ-YUVguKNMHZP?usp=sharing) and place it in the root directory of the repository.  
The `data2` folder consists of the processed data from the experiments conducted in the paper. The data is stored in the form of `.txt` files.   
Once the data is downloaded, the following commands can be run to generate the plots and tables:
1) To generate the data showing the agnostic nature of the metrics to other sim2real gaps run the following command:
```bash
cd scripts
python agnostic.py
```
This generates the VEPD values for the comparison between tests carried out in sim and grass and between sim and concrete. These two values being close indicates that the the metric is agnostic to changes in environment and is only sensitive to sensor sim2real differences. The corresponding plots comparing the velocity profiles can also be generated using:
```bash
python agnostic_plot.py
```
2) To generate the VEPD metrics and the corresponding histogram plots for different variants of the GPS and IMU sensors run the following command:
```bash
python computeEPD.py
```
3) To compare different velocity profiles for each of the tests between the sim variants of the sensor and reality run the following command:
```bash
bash plot_all_velCompare.sh
```
This will populate the `plots_3` directory with the plots comparing the velocity profiles for each of the tests.

