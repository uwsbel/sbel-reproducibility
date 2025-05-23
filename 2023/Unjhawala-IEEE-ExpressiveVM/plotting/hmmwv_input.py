import sys
import matplotlib.pyplot as mpl
# import arviz as az
import rom
import pandas as pd
import numpy as np

"""
Generates five different plots for each of the training inputs . Each plot has throttle, braking and steering as seperate subplots
Command line inputs
1) input txt file name which is supplied to the ROM and Chrono model
2) End time of the simulation -> for now this needs to be added manually
3) flag to specify if the plots need to be saved
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

vehicle = "HMMWV"
file = sys.argv[1]

fig, axes = mpl.subplots(nrows = 1, ncols = 3, sharey = True)
fileName_con = "../calibration/" + vehicle + "/inputs/" + file  + ".txt"


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

endTime = float(sys.argv[2]) + 0.0001
t = 0
timeStepNo = 0
while(t<endTime):
    rom.getControls(controls, driverData, t)

    steering_controls.append(controls[1])
    acc_controls.append(controls[2])
    braking_controls.append(controls[3])
    time.append(t)
    t = t + step


axes[0].plot(time,acc_controls,'k')
axes[1].plot(time,steering_controls, 'k')
axes[2].plot(time,braking_controls, 'k')
axes[0].set_ylabel("Normalized")
titles = ["Throttle", "Steering", "Braking"]

for i, ax in enumerate(axes):
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1,1.01])
    ax.set_xlim([0,max(time)+0.5])
    ax.set_title(titles[i])
mpl.xlabel("Time (s)")

fig.tight_layout()

save = sys.argv[3]
if(save):
    mpl.savefig(f"./images/hmmwv_train_inputs_{file}.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/hmmwv_train_inputs_{file}", facecolor = 'w', dpi = 600) 
mpl.show()