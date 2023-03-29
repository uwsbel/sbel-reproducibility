import sys
sys.path.append('/home/unjhawala/projectlets/model-repo/simple-vehicles/lang-c/interfaces')
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
test_nums = [sys.argv[1]]

fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (8,4), sharey = True)

i = 0
for test_num in test_nums:
    exp_num = sys.argv[2]
    full_file = "r" + test_num
    test_type = "Sin_" + exp_num
    ART_path = "/ART1_032523_mocapV2/"
    fileName_con = "../calibration/" + vehicle+ "/inputs" + ART_path + test_type + "/" + full_file + ".txt"



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
    endTime = 6.
    t = 3.5
    timeStepNo = 0
    while(t<endTime):
        rom.getControls(controls, driverData, t)

        steering_controls.append(controls[1])
        acc_controls.append(controls[2])
        braking_controls.append(controls[3])
        time.append(t)
        t = t + step


    


    axes.plot(time, steering_controls, 'k')



    i = i +1



mpl.xlabel("Time (s)")
axes.set_ylabel("Normalized Steering")
axes.set_xlabel("Time (s)")
axes.set_ylim([-1.01,1.01])
axes.set_xticks(np.arange(3.5,endTime,0.2))
axes.set_xlim([3.5,max(time)])
fig.tight_layout()
save = int(sys.argv[3])
if(save):
    mpl.savefig(f"./images/art_test_input_comp2.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_test_input_comp2", facecolor = 'w', dpi = 600) 
mpl.show()