import sys
import matplotlib.pyplot as mpl
# import arviz as az
import rom
import pandas as pd
import numpy as np


"""
Generates five different plots for each of the training inputs . Each plot has throttle, braking and steering as seperate subplots
Command line inputs

1) flag to specify if the plots need to be saved
"""

mpl.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino', 'serif'],
    # "font.serif" : ["Computer Modern Serif"],
})


# Input file to be used
vehicle = "ART"
test_nums = ['4','3', '2', '0','2']
exp_nums = ['02', '04', '06', '08', '1']
cols = ['k', 'b', 'g', 'r', 'y']
exp_test_map = {exp_nums[i]: test_nums[i] for i in range(len(exp_nums))}
fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (10,6), sharey = True)

i = 0
for exp_num,test_num in exp_test_map.items():

    full_file = "test" + test_num
    test_type = exp_num + "s-rampt-10s"
    ART_path = "/ART1_021923/"
    fileName_con = "../calibration/" + vehicle+ "/inputs" + ART_path + test_type + "/" + full_file + ".txt"



    # json files
    fileName_veh = "../calibration/"  + vehicle + "/jsons/HMMWV.json"
    fileName_tire = "../calibration/" + vehicle + "/jsons/TMeasy.json"

    # fill up our driver data from the file
    driverData = rom.vector_entry()
    rom.driverInput(driverData,fileName_con)


    print(fileName_con)
    ## Just get the controls using get controls and save it into a list
    steering_controls = []
    acc_controls = []
    braking_controls = []
    time = []
    controls = rom.vector_double(4,0)

    # time stepping
    step = 0.001

    # simulation end time 
    endTime = 10.
    t = 0
    timeStepNo = 0
    while(t<endTime):
        rom.getControls(controls, driverData, t)

        steering_controls.append(controls[1])
        acc_controls.append(controls[2])
        braking_controls.append(controls[3])
        time.append(t)
        t = t + step


    

    axes[0].plot(time,acc_controls, cols[i], label = f'Experiment {i+1}')
    axes[1].plot(time, steering_controls, cols[i], label =f'Experiment {i+1}')



    i = i +1



mpl.xlabel("Time (s)")
axes[0].set_ylabel("Normalized Throttle")
axes[1].set_ylabel("Normalized Steering")
axes[1].legend(fontsize = 12)
for i, ax in enumerate(axes):
    ax.set_xlabel("Time (s)")
    ax.set_ylim([0,1.01])
    ax.set_xlim([0,max(time)+0.5])
fig.tight_layout()
save = int(sys.argv[1])
if(save):
    mpl.savefig(f"./images/art_train_lat_input.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_train_lat_input", facecolor = 'w', dpi = 600) 
mpl.show()