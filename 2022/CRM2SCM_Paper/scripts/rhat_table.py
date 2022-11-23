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
# Generate the R hat, MCSE mean and MCSE std tables
# =============================================================================


import numpy as np
import arviz as az
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



# # extract the stats using the arviz function
stats_normal = az.summary(nc_data_normal,kind='diagnostics',round_to = 5)
stats_fricti = az.summary(nc_data_fricti,kind='diagnostics',round_to = 5)
stats_janosi = az.summary(nc_data_janosi,kind='diagnostics',round_to = 5)


# Normalize the statistics with posterior mean
means_normal = np.array([abs(np.mean(nc_data_normal['posterior']['K_c'].values)), \
                    abs(np.mean(nc_data_normal['posterior']['K_phi'].values)), \
                        abs(np.mean(nc_data_normal['posterior']['n'].values))])


stats_normal.loc[:,'mcse_mean'] = stats_normal.loc[:,'mcse_mean'] / means_normal
stats_normal.loc[:,'mcse_sd'] = stats_normal.loc[:,'mcse_sd'] / means_normal


means_fricti = np.array([abs(np.mean(nc_data_fricti['posterior']['cohesion'].values)), \
                    abs(np.mean(nc_data_fricti['posterior']['phi'].values))])

stats_fricti.loc[:,'mcse_mean'] = stats_fricti.loc[:,'mcse_mean'] / means_fricti
stats_fricti.loc[:,'mcse_sd'] = stats_fricti.loc[:,'mcse_sd'] / means_fricti


means_janosi = np.array([abs(np.mean(nc_data_janosi['posterior']['K_s'].values))])

stats_janosi.loc[:,'mcse_mean'] = stats_janosi.loc[:,'mcse_mean'] / means_janosi
stats_janosi.loc[:,'mcse_sd'] = stats_janosi.loc[:,'mcse_sd'] / means_janosi


# # drop the columns that are not required
stats_normal = stats_normal.drop(['ess_bulk','ess_tail'], axis = 1)
stats_fricti = stats_fricti.drop(['ess_bulk','ess_tail'], axis = 1)
stats_janosi = stats_janosi.drop(['ess_bulk','ess_tail'], axis = 1)

# # print as latex code
print(stats_normal)
print(stats_fricti)
print(stats_janosi)