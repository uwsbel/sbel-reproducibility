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


vehicle = "ART"

fileName_con_st = "../calibration/" + vehicle + "/inputs/multi_run_acc/full_throttle/test1.txt"
fileName_con = "../calibration/" + vehicle + "/inputs/multi_run_acc/ramp/test0.txt"


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



# fill up our driver data from the file
driverData = rom.vector_entry()
rom.driverInput(driverData,fileName_con_st)

## Just get the controls using get controls and save it into a list
steering_controls = []
acc_controls_st = []
braking_controls = []
time = []
controls = rom.vector_double(4,0)

t = 0
timeStepNo = 0
while(t<endTime):
    rom.getControls(controls, driverData, t)

    steering_controls.append(controls[1])
    acc_controls_st.append(controls[2])
    braking_controls.append(controls[3])
    time.append(t)
    t = t + step


fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (10,6), sharey = True)

axes[0].plot(time,acc_controls,'k')
axes[1].plot(time,acc_controls_st, 'y')
axes[0].set_ylabel("Normalized Throttle")
titles = ["Ramp Throttle", "Step Throttle"]
for i, ax in enumerate(axes):
    ax.set_xlabel("Time (s)")
    ax.set_ylim([0,1.01])
    ax.set_xlim([0,max(time)+0.5])
    ax.set_title(titles[i])
    # ax.grid()

mpl.xlabel("Time (s)")

fig.tight_layout()
save = int(sys.argv[1])
if(save):
    mpl.savefig(f"./images/art_train_long_inputs.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/hmmwv_train_long_inputs", facecolor = 'w', dpi = 600) 
mpl.show()