import sys
import matplotlib.pyplot as mpl
import arviz as az
import pandas as pd
import numpy as np

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


vehicles = [10, 3200, 51200, 102400, 204800, 260000, 272000, 296000, 512000]
times = np.array([1184, 1703, 3477, 7048, 14404, 18271, 18827, 20452, 35655])

rtf = times / 20000

fig = mpl.figure(figsize=(6,6))
mpl.plot(vehicles, rtf)
# mpl.grid()
mpl.xlabel("Number of Vehicles")
mpl.ylabel("Real Time Factor (RTF)")
mpl.xlim([0,max(vehicles) + 50000])
mpl.ylim([0,max(rtf) + 0.05])
mpl.axhline(y=1, xmin = 0, xmax = 0.51, color='r', linestyle='-')
mpl.axvline(x=290000, ymin = 0, ymax = 0.545, color='r', linestyle='-')
mpl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig.tight_layout()
save = int(sys.argv[1])

if(save):   
    mpl.savefig(f"./images/gpu_scaling.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/gpu_scaling", facecolor = 'w', dpi = 600) 

mpl.show()