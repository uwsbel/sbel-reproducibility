import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import seaborn as sns
from scipy import stats
def normalize(x):
    return (x - x.min(0)) / x.ptp(0)

"""
Generates the posterior plots for the ART vehcile Lateral calibration in the paper. Requires from the command line a flag to save the plot
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


for i,filename in enumerate(files):
    fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (4,2))
    idata = az.from_netcdf('../calibration/' +   vehicle +'/results/' + filename + ".nc")
    posterior_para = idata['posterior'][names_nc[i]]
    shape_nc = idata['posterior'][names_nc[i]].shape
    tot_nc_length = shape_nc[0]*shape_nc[1]

    # Flatten the array
    posterior_para = np.ravel(posterior_para)
    
    # Print mean as sanity check
    para_mean = float(np.mean(posterior_para))
    print("Mean value is: ", para_mean)

    g = sns.kdeplot(posterior_para , bw_adjust = 3, ax = axes)

    line = g.get_lines()[0]
    xd = line.get_xdata()
    yd = line.get_ydata()

    norm_y = normalize(yd)

    axes.clear()
    axes.plot(xd,norm_y,alpha=0.6, ls='-', linewidth=2.0)
    axes.set_title(names[i])
    axes.text(0.5, 0.5, f'Mean = {para_mean:.3}', horizontalalignment='center', verticalalignment='center',transform=axes.transAxes, fontsize = 13)

    fig.tight_layout()
    save = int(sys.argv[1])
    if(save):
        mpl.savefig(f"./images/art_lat_post_{names_nc[i]}.eps", format='eps', dpi=3000) 
        mpl.savefig(f"./images/art_lat_post_{names_nc[i]}.png", facecolor = 'w', dpi = 600)
    mpl.show()
