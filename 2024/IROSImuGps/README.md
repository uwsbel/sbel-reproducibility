### Quantifying the Sim2real Gap for GPS and IMU Sensors
This repository holds the source code to reproduce results obtained in the IROS submission "Quantifying the Sim2real Gap for GPS and IMU Sensors".


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

