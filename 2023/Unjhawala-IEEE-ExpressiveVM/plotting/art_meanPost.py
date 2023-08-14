import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import seaborn as sns
from scipy import stats
"""

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

### Get the nc file
vehicle = "ART"
filename = "20221214_080100_test4"
idata_acc = az.from_netcdf("../calibration/" + vehicle + "/results/" + filename + ".nc")


### Generate the mean plot
rpms = []
rpms_loss = []
tors = []
losses = []

rpms_m = [-100, 400, 900, 1600]
rpms_loss_m = [-100, 400, 1600]
tors_m = [np.mean(idata_acc['posterior']['p0tor'].values), np.mean(idata_acc['posterior']['p1tor'].values),
                np.mean(idata_acc['posterior']['p2tor'].values), np.mean(idata_acc['posterior']['p3tor'].values)]
losses_m = [np.mean(idata_acc['posterior']['p0loss'].values), np.mean(idata_acc['posterior']['p1loss'].values), 
                    np.mean(idata_acc['posterior']['p2loss'].values)]


 # Used to Label the plot   
names_tor = ["$Tor_0$", "$Tor_1$", "$Tor_2$", "$Tor_3$"]
names_loss = ["$Loss_0$", "$Loss_1$", "$Loss_2$"] 
    

    

     


### Maps with throttle
fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (10,5))

## Ignoring throttle dependence for now
# ths = np.arange(0.1,1,0.1)

# for t in ths:
#   new_rpms = np.array(rpms_m) * t
#   new_tors = np.array(tors_m) * t
#   axes[0].plot(new_rpms, new_tors, label =f"{int(t*100)}% throttle")

axes[0].plot(rpms_m, tors_m, 'ko-')
axes[1].plot(rpms_loss_m, losses_m, 'ko-')

yLabel = ["Motor Torque (N-m)", "Motor Losses (N-m)"]
titles = ["Motor Torque Map", "Motor Losses Map"]

# fig.suptitle('Combined ramp and step calibration', fontsize=14)
for i, ax in enumerate(axes):
    # ax.grid(True)
    ax.set_xlabel("RPM")
    ax.set_ylabel(yLabel[i])
    ax.set_title(titles[i])
    if(i == 0):
        for k, txt in enumerate(range(len(rpms_m))):
            if(k == len(rpms_m) - 1):
                ax.annotate(names_tor[k], (rpms_m[k] - 125, tors_m[k] + 0.013))
            else:
                ax.annotate(names_tor[k], (rpms_m[k] + 100, tors_m[k]))
    else:
        for l, txt in enumerate(range(len(rpms_loss_m))):
            if(l == len(rpms_loss_m) - 1):
                ax.annotate(names_loss[l], (rpms_loss_m[l] - 200, losses_m[l] + 0.0005))
            else:
                ax.annotate(names_loss[l], (rpms_loss_m[l] + 100, losses_m[l]))


# axes[0].legend()


fig.tight_layout()
save = int(sys.argv[1])
if(save):
    mpl.savefig(f"./images/art_train_long_meanPost.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_train_long_meanPost", facecolor = 'w', dpi = 600) 
mpl.show()