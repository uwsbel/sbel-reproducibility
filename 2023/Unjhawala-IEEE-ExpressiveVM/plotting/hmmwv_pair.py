import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import seaborn as sns
from scipy import stats
import pandas as pd
def normalize(x):
    return (x - x.min(0)) / x.ptp(0)

"""
Generates the pairwise plots for the HMMWV vehcile in the paper. Requires from the command line a flag to save the plot
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
idata = az.from_netcdf('../' +   vehicle +'/results/' + filename + ".nc")


# Parameter names for plots
names = ["$f_{df^0_y}$", "$f_{F^M_y}$", "$f_{maxSteer}$", "$f_{df^0_x}$", "$f_{F^M_x}$", "$f_{loss}$"]

# Parameter names in nc file

names_nc = ["f_dfy","f_fym", "f_maxSteer", "f_dfx", "f_fxm", "f_loss"]
idata_arr = np.asarray(idata['posterior'].to_array())
idata_arr = idata_arr.reshape(idata_arr.shape[0],-1).T

means = abs(np.mean(idata_arr,axis=0))
idata_arr = idata_arr/means
print(idata_arr.shape)
idata_df = pd.DataFrame(idata_arr,columns = names)

g = sns.pairplot(idata_df, kind = 'kde', corner = True,plot_kws=dict(s = 10))
mpl.tight_layout()

save = sys.argv[1]
if(save):
    mpl.savefig(f"./images/hmmwv_pair.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/hmmwv_pair.png", facecolor = 'w', dpi = 600) 
mpl.show()










