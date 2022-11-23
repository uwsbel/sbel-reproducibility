# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
# 
# Author: Huzaifa Mustafa Unjhawala, Wei Hu
# 
# =============================================================================
# Calibration of the SCM parameters using Bayesian Optimization
# Plotting the pairwise KDE plots
# =============================================================================

import numpy as np
import matplotlib.pyplot as mpl
import arviz as az
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import pandas as pd

# Only number of steps can be changed
nsteps = 500000
sigma = 0.01
nchains = 4


# Get our file names
nc_dir = "./nc_file/"
direction_normal = "new_3_para_Normal_"
direction_fricti = "new_2_para_Frictional_"
direction_janosi = "new_1_para_Frictional_"
num_chains = str(nchains) + "_chains_"
num_steps = str(nsteps) + "_steps_"
sigma_val = "sigma_" + str(sigma) 
NC_File_Normal = nc_dir + direction_normal + num_chains + num_steps + sigma_val + ".nc"
NC_File_Fricti = nc_dir + direction_fricti + num_chains + num_steps + sigma_val + ".nc"
NC_File_Janosi = nc_dir + direction_janosi + num_chains + num_steps + sigma_val + ".nc"


# Where to save the png files
out_dir = "plot_pairwise/"
pairWise = out_dir + "Pair_Wise_"   + str(nsteps) + "_samples_"


# extract the data from the nc files
nc_data_normal = az.from_netcdf(NC_File_Normal)
nc_data_fricti = az.from_netcdf(NC_File_Fricti)
nc_data_janosi = az.from_netcdf(NC_File_Janosi)



# conver to a shape where columns are parameters and rows are all the points
# from all the chains combined
posterior_para_normal = np.asarray(nc_data_normal['posterior'].to_array())
posterior_para_normal = posterior_para_normal.reshape(posterior_para_normal.shape[0],-1).T

posterior_para_fricti = np.asarray(nc_data_fricti['posterior'].to_array())
posterior_para_fricti = posterior_para_fricti.reshape(posterior_para_fricti.shape[0],-1).T


# Normalize all the data
means_normal = abs(np.mean(posterior_para_normal,axis=0))
means_fricti = abs(np.mean(posterior_para_fricti,axis=0))

posterior_para_normal = posterior_para_normal / means_normal
posterior_para_fricti  =  posterior_para_fricti / means_fricti

# rename the columns to latex for paper
normal_cols = [r'$K_c$',r'$K_{\phi}$',r'$n$']
normal = pd.DataFrame(posterior_para_normal,columns = normal_cols)


fricti_cols = [r'$c$',r'$\phi$']
fricti = pd.DataFrame(posterior_para_fricti,columns = fricti_cols)

# Size of the figure (unit:in)
fig_size = [10, 5]


# DPI of the image file
dpi_png = 300

# Font size in the plot
text_size = 16
font = {'weight': 'bold', 'size': text_size}


# mpl.figure(figsize = fig_size)
mpl.rc('font', **font)
sns.set(rc={'figure.figsize':fig_size})
sns.set_style(style='white')

# plot the pairwise plot for normal
g = sns.pairplot(normal,kind='kde')
# for ax in g.axes.flatten():
#     ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

# save the plot
png_name_normal = pairWise + "normal.png" 
mpl.savefig(png_name_normal, facecolor = 'w', dpi=dpi_png)


# plot the pairwise plot for frictional 
g = sns.pairplot(fricti,kind='kde')


# # save the plot
png_name_fricti = pairWise + "fricti.png" 
mpl.savefig(png_name_fricti, facecolor = 'w', dpi=dpi_png)





