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
# Author: Wei Hu
# 
# =============================================================================
# Calibration of the SCM parameters using Bayesian Optimization
# =============================================================================

import numpy as np
import matplotlib.pyplot as mpl
import arviz as az
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Name of the parameters that need to be plotted
names = ["K_c", "K_phi", "n", "phi", "cohesion", "K_s"]

# Distribution limits of each parameter in the axes
axes_lim = [[ -150000,  -70000], # K_c
            [ 2000000, 2450000], # K_phi
            [    1.16,    1.25], # n
            [      22,      26], # phi
            [    1500,    3500], # cohesion
            [  0.0025,  0.0034]] # K_s

# If needs to print name of the parameter as a title of the plot, set to 1
print_name_parameter = 0

# Size of the figure (unit:in)
fig_size = [10, 5]

# DPI of the image file
dpi_png = 300

# Font size in the plot
text_size = 16
font = {'weight': 'bold', 'size': text_size}

# Number of bins in the probabilities plot
NumBin = 40

# ========================================================================
# ========================== Loop cases with different number of samplings
# ========================================================================
for nsamples in range(5):
    # Five different cases with different number of samples
    nsteps = 10000
    if nsamples == 1:
        nsteps = 50000
    elif nsamples == 2:
        nsteps = 100000
    elif nsamples == 3:
        nsteps = 500000
    elif nsamples == 4:
        nsteps = 1000000
    
    # ========================================================================
    # ============================= Some general information of the simulation
    # ========================================================================
    sigma = 0.01
    nchains = 4
    
    # Where to load the .nc files
    nc_dir = "nc_file/"
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
    out_dir = "03_plot_distribution/"
    Trace = out_dir  + "Trace_"  + str(nsteps) + "_samples_" 
    Post  = out_dir  + "Post_"   + str(nsteps) + "_samples_"
    PostMean  = out_dir + "Post_Mean_"   + str(nsteps) + "_samples_"

    # Load the data from a .nc file
    nc_data_normal = az.from_netcdf(NC_File_Normal)
    nc_data_fricti = az.from_netcdf(NC_File_Fricti)
    nc_data_janosi = az.from_netcdf(NC_File_Janosi)

    # Get the dimension of the data
    shape_nc = nc_data_normal['posterior']['K_c'].shape
    tot_nc_length = shape_nc[0]*shape_nc[1]
    print("===================================================================")
    print(tot_nc_length, shape_nc[0], shape_nc[1])

    # ========================================================================
    # ================================= Start the plot the trace and posterior
    # ========================================================================
    for nfig in range(6):
        # Get the sampling data for each parameter
        posterior_para = nc_data_normal['posterior'][names[0]]
        if nfig < 3:
            # The first 3 parameters are from the Bekker Wong model
            posterior_para = nc_data_normal['posterior'][names[nfig]]
        elif nfig < 5:
            # The next 2 parameters are from the Janosi Hanamoto model
            posterior_para = nc_data_fricti['posterior'][names[nfig]]
        else:
            # The last 1 parameter is from the Janosi Hanamoto model
            posterior_para = nc_data_janosi['posterior'][names[nfig]]
        
        # Transpose the matrix of the sampling data
        posterior_para_new = posterior_para.transpose()

        # Reshape the matrix to one dimentional array
        posterior_para_mean = np.reshape(posterior_para, (tot_nc_length, 1))

        # Calculate the mean of the one dimentional array
        para_mean = float(np.mean(posterior_para_mean))
        print("Mean value is: ", para_mean)
        
        # ====================================================================
        # =================================== Plot the trace of the parameters
        mpl.figure(figsize = fig_size)
        mpl.rc('font', **font)
        # mpl.xlabel('No. of samples')
        # mpl.ylabel('Value of the parameter')

        # Print name of the parameters in the title
        if print_name_parameter == 1:
            if names[nfig]=="K_c":
                mpl.title("$K_c$")
            if names[nfig]=="K_phi":
                mpl.title("$K_{\phi}$")
            if names[nfig]=="n":
                mpl.title("$n$")
            if names[nfig]=="phi":
                mpl.title("${\phi}$")
            if names[nfig]=="cohesion":
                mpl.title("$c$")
            if names[nfig]=="K_s":
                mpl.title("$K_s$")

        # mpl.plot(posterior_para_new, alpha=.7, linewidth = 1)
        mpl.plot(posterior_para_new[:,0], color='red',    alpha=.7, linestyle='solid',   linewidth = 2, label='Chain 1')
        mpl.plot(posterior_para_new[:,1], color='green',  alpha=.7, linestyle='dotted',  linewidth = 2, label='Chain 2')
        mpl.plot(posterior_para_new[:,2], color='orange', alpha=.7, linestyle='dashed',  linewidth = 2, label='Chain 3')
        mpl.plot(posterior_para_new[:,3], color='blue',   alpha=.7, linestyle='dashdot', linewidth = 2, label='Chain 4')
        
        # Only put legend for first parameter
        # if names[nfig]=="K_c":
        mpl.legend(loc='upper right', ncol=1)

        # frame0 = mpl.gca()
        # frame0.axes.set_ylim(axes_lim[nfig][0], axes_lim[nfig][1])

        # Use scientific notation for K_c and K_phi
        mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if names[nfig]=="K_c":
            mpl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        if names[nfig]=="K_phi":
            mpl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        if names[nfig]=="K_s":
            mpl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        # Print the plot into a png file
        png_name = Trace + names[nfig] + '.png'
        mpl.savefig(png_name, facecolor = 'w', dpi=dpi_png)   
        print(png_name, " was saved into a file")
        mpl.close()
        # ====================================================================

        # ====================================================================
        # ================== Plot the posterior distribution of the parameters
        mpl.figure(figsize = fig_size)
        mpl.rc('font', **font)
        # mpl.xlabel('Value of the parameter')
        # mpl.ylabel('Density')

        # Print name of the parameters in the title
        if print_name_parameter == 1:
            if names[nfig]=="K_c":
                mpl.title("$K_c$")
            if names[nfig]=="K_phi":
                mpl.title("$K_{\phi}$")
            if names[nfig]=="n":
                mpl.title("$n$")
            if names[nfig]=="phi":
                mpl.title("${\phi}$")
            if names[nfig]=="cohesion":
                mpl.title("$c$")
            if names[nfig]=="K_s":
                mpl.title("$K_s$")
        
        # sns.kdeplot(posterior_para_new, fill=False, alpha=.7, linewidth=2.0, bw_adjust=0.5, legend=True)
        sns.kdeplot(posterior_para_new[:,0], color='red',    fill=False, alpha=1, ls='-', linewidth=2.0, bw_adjust=0.5)
        sns.kdeplot(posterior_para_new[:,1], color='green',  fill=False, alpha=1, ls=':', linewidth=2.0, bw_adjust=0.5)
        sns.kdeplot(posterior_para_new[:,2], color='orange', fill=False, alpha=1, ls='--',linewidth=2.0, bw_adjust=0.5)
        sns.kdeplot(posterior_para_new[:,3], color='blue',   fill=False, alpha=1, ls='-.',linewidth=2.0, bw_adjust=0.5)

        # Use scientific notation for K_c and K_phi
        if names[nfig]=="K_c":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if names[nfig]=="K_phi":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if names[nfig]=="K_s":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        # Only put legend for first parameter
        # if names[nfig]=="K_c":
        mpl.legend(['Chain 1', 'Chain 2', 'Chain 3', 'Chain 4'])

        frame1 = mpl.gca()
        # frame1.axes.xaxis.set_ticklabels([])
        # frame1.axes.yaxis.set_ticklabels([])
        # frame1.axes.get_xaxis().set_ticks([])
        # frame1.axes.get_yaxis().set_ticks([])
        frame1.axes.get_yaxis().set_visible(False)

        frame1.axes.set_xlim(axes_lim[nfig][0], axes_lim[nfig][1])

        # Print the plot into a png file
        png_name = Post + names[nfig] + '.png'
        mpl.savefig(png_name, facecolor = 'w', dpi=dpi_png)   
        print(png_name, " was saved into a file")
        mpl.close()
        # ====================================================================

        # ====================================================================
        # ================ Plot the mean of the distribution of the parameters
        # mpl.figure(figsize = fig_size)
        fig, ax1 = mpl.subplots(figsize = fig_size)
        mpl.rc('font', **font)
        # mpl.xlabel(' ')
        # mpl.ylabel(' ')

        # Print name of the parameters in the title
        if print_name_parameter == 1:
            if names[nfig]=="K_c":
                mpl.title("$K_c$")
            if names[nfig]=="K_phi":
                mpl.title("$K_{\phi}$")
            if names[nfig]=="n":
                mpl.title("$n$")
            if names[nfig]=="phi":
                mpl.title("${\phi}$")
            if names[nfig]=="cohesion":
                mpl.title("$c$")
            if names[nfig]=="K_s":
                mpl.title("$K_s$")
        
        # Divide the x axis into a given number of bins, and get the width of the bin
        binwidth = (axes_lim[nfig][1] - axes_lim[nfig][0]) / NumBin
        
        # Plot the probability using bins with given width
        sns.histplot(data=posterior_para_mean, color='red', kde=False, stat="probability", label="Probability",
                     binwidth=binwidth, ax=ax1)
        ax1.set_ylabel(" ")
        ax1.legend(loc='upper left')
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))

        # Plot the KDE (Kernel Density Estimation) density
        ax2 = ax1.twinx()
        sns.kdeplot(data=posterior_para_mean, color='red', label="KDE", ls='-', lw=3, ax=ax2)
        ax2.set_ylim(0, ax1.get_ylim()[1] / binwidth)  # similir limits on the y-axis to align the plots
        ax2.yaxis.set_major_formatter(PercentFormatter(1 / binwidth))  # show axis such that 1/binwidth corresponds to 100%
        ax2.yaxis.set_ticklabels([])
        ax2.get_yaxis().set_visible(False)
        ax2.set_ylabel(" ")
        ax2.legend(loc='upper right')
        mpl.show()

        # Use scientific notation for K_c and K_phi
        if names[nfig]=="K_c":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if names[nfig]=="K_phi":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if names[nfig]=="K_s":
            mpl.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        frame2 = mpl.gca()
        # frame2.axes.xaxis.set_ticklabels([])
        # frame2.axes.yaxis.set_ticklabels([])
        # frame2.axes.get_xaxis().set_ticks([])
        # frame2.axes.get_yaxis().set_ticks([])

        frame2.axes.set_xlim(axes_lim[nfig][0], axes_lim[nfig][1])

        frame2.set_frame_on(True)
        # frame2.get_xaxis().tick_bottom()
        # frame2.get_yaxis().set_visible(False)

        # frame2.spines.right.set_visible(False)
        # frame2.spines.left.set_visible(False)
        # frame2.spines.top.set_visible(False)

        # Print the mean value of the parameter into the plot at a given postion
        small_dis = (axes_lim[nfig][1] - axes_lim[nfig][0]) / 80.0
        text_pos = [frame2.axes.get_xlim()[0] + small_dis, 
                    0.75 * (frame2.axes.get_ylim()[0] + frame2.axes.get_ylim()[1])]
        textMean = 'Mean = %6.4e' %para_mean
        if names[nfig]=="n":
            textMean = 'Mean = %6.4f' %para_mean
        mpl.text(text_pos[0], text_pos[1], textMean)

        # Print the bin width value into the plot at a given postion
        text_pos = [frame2.axes.get_xlim()[0] + small_dis, 
                    0.6 * (frame2.axes.get_ylim()[0] + frame2.axes.get_ylim()[1])]
        textBinwidth = 'Bin Width = %d' %binwidth
        if names[nfig]=="n" or names[nfig]=="K_s":
            textBinwidth = 'Bin Width = %.2e' %binwidth
        if names[nfig]=="phi":
            textBinwidth = 'Bin Width = %.1f' %binwidth
        mpl.text(text_pos[0], text_pos[1], textBinwidth)

        # Print the plot into a png file
        png_name = PostMean + names[nfig] + '.png'
        mpl.savefig(png_name, facecolor = 'w', dpi=dpi_png)   
        print(png_name, " was saved into a file")
        mpl.close()
        # ====================================================================