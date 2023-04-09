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
test_nums = np.arange(1,10,1)

fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (8,4), sharey = True)

i = 0
actual_test = 0
for test_num in test_nums:
    exp_num = sys.argv[1]
    test_type = "r" + exp_num
    full_file = test_type + "_r" +  str(test_num)
    ART_path = "/ART1_040623_mocap_reapTest/"
    fileName_con = "../calibration/" + vehicle+ "/inputs" + ART_path + test_type + "/" + full_file + ".txt"

    if((test_type == "r1") & (test_num == 7)):
        continue
    if((test_type == "r1") & (test_num == 3)):
        continue
    if((test_type == "r2") & (test_num == 8)):
        continue
    if((test_type == "r2") & (test_num == 3)):
        continue



    # json files
    fileName_veh = "../calibration/"  + vehicle + "/jsons/HMMWV.json"
    fileName_tire = "../calibrtion/" + vehicle + "/jsons/TMeasy.json"

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
    endTime = 12.
    t = 0.
    timeStepNo = 0
    while(t<endTime):
        rom.getControls(controls, driverData, t)

        steering_controls.append(controls[1])
        acc_controls.append(controls[2])
        braking_controls.append(controls[3])
        time.append(t)
        t = t + step


    


    axes[1].plot(time, steering_controls)
    axes[0].plot(time, acc_controls, label = f"Test {actual_test+1}")



    i = i +1
    actual_test = actual_test + 1



mpl.xlabel("Time (s)")
axes[0].set_ylabel("Normalized Throttle")
axes[1].set_ylabel("Normalized Steering")
axes[0].set_xlabel("Time (s)")
axes[1].set_xlabel("Time (s)")
axes[0].legend()
# axes.set_ylim([-1.01,1.01])
# axes.set_xticks(np.arange(3.5,endTime,0.2))
# axes.set_xlim([3.5,max(time)])
fig.tight_layout()
save = int(sys.argv[2])
if(save):
    mpl.savefig(f"./images/art_test_input_showSame" + f"r{exp_num}" ".eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_test_input_showSame" + f"r{exp_num}" , facecolor = 'w', dpi = 600) 
mpl.show()