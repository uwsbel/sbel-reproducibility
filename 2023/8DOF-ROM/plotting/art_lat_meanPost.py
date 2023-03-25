import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import seaborn as sns
from scipy import stats


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


files = [
            "20230302_183045", # 0.2
            "20230302_181306", # 0.4
            "20230302_115955", # 0.6
            "20230302_115455",
            "20230302_180819"
        ]


# Parameter names for plots
names = ["$\delta_{0.2}$", "$\delta_{0.4}$", "$\delta_{0.6}$", "$\delta_{0.8}$", "$\delta_{1.}$"]

# Parameter names in nc file
names_nc = ["f_02", "f_04", "f_06", "f_08", "f_10"]


means = [0]
vehicle = "ART"
fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,4))
for i,filename in enumerate(files):
    idata = az.from_netcdf('../calibration/' +   vehicle +'/results/' + filename + ".nc")
    means.append(np.mean(idata['posterior'][names_nc[i]].values))



norm_steer = [0,0.2, 0.4, 0.6, 0.8, 1.]

axes.plot(norm_steer, means, 'ko-')
axes.set_xlabel("Normalized steering input [-]")
axes.set_ylabel("Wheel angle [$\delta$ rad]")

fig.tight_layout()
save = int(sys.argv[1])

if(save):
    mpl.savefig(f"./images/art_train_lat_meanPost.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_train_lat_meanPost", facecolor = 'w', dpi = 600) 
mpl.show()


