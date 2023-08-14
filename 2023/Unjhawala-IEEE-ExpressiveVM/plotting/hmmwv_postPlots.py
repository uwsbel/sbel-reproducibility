import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import seaborn as sns
from scipy import stats
# mpl.style.use('ieee.mplstyle')
def normalize(x):
    return (x - x.min(0)) / x.ptp(0)

"""
Generates the posterior plots for the HMMWV vehcile in the paper. Requires from the command line a flag to save the plot
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

# The chain file
filename = "20230210_093107"
vehicle = "HMMWV"
idata = az.from_netcdf('../calibration/' +   vehicle +'/results/' + filename + ".nc")


# Parameter names for plots
names = ["$f_{df^0_y}$", "$f_{F^M_y}$", "$f_{maxSteer}$", "$f_{df^0_x}$", "$f_{F^M_x}$", "$f_{loss}$"]

# Parameter names in nc file

names_nc = ["f_dfy","f_fym", "f_maxSteer", "f_dfx", "f_fxm", "f_loss"]


# Define the plot
fig, axes = mpl.subplots(nrows = 2, ncols = 3, sharey = True, figsize = (8,4))

for i,name in enumerate(names_nc):
    posterior_para = idata['posterior'][name]
    shape_nc = idata['posterior'][name].shape
    tot_nc_length = shape_nc[0]*shape_nc[1]  
    
    # Flatten the array
    posterior_para = np.ravel(posterior_para)
    
    # Print mean as sanity check
    para_mean = float(np.mean(posterior_para))
    print("Mean value is: ", para_mean)

    ## Normalizing the y axis
    if(i < 3):
        g = sns.kdeplot(posterior_para , bw_adjust = 0.5, ax = axes[0,i])
    else:
        g = sns.kdeplot(posterior_para , bw_adjust = 0.5, ax = axes[1,i-3])

    line = g.get_lines()[0]
    xd = line.get_xdata()
    yd = line.get_ydata()

    norm_y = normalize(yd)
    if(i < 3):
        axes[0,i].clear()
        axes[0,i].plot(xd,norm_y,alpha=0.6, ls='-', linewidth=2.0)
        axes[0,i].set_title(names[i])
        axes[0,i].text(0.5, 0.5, f'Mean = {para_mean:.3}', horizontalalignment='center', verticalalignment='center',transform=axes[0,i].transAxes, fontsize = 14)
    else:
        axes[1,i-3].clear()
        axes[1,i-3].plot(xd,norm_y,alpha=0.6, ls='-', linewidth=2.0)
        axes[1,i-3].set_title(names[i])
        axes[1,i-3].text(0.5, 0.5, f'Mean = {para_mean:.3}', horizontalalignment='center', verticalalignment='center',transform=axes[1,i-3].transAxes, fontsize = 14)



fig.tight_layout()
save = sys.argv[1]
if(save):
    mpl.savefig(f"./images/hmmwv_post.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/hmmwv_post.png", facecolor = 'w', dpi = 600) 
mpl.show()








