import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import polynomial as P

# ==================================================
print("Processing data for VIPER rover simulation")

# General information of the data and plots
lw = 2
fs = 14
ms = 14
DPI = 300
Weight = 'bold'
fig_size = [8, 6]
xlim = [[ 0,   10 ], 
        [-0.1, 0.9]]
ylim = [[-500, 3000],
        [ 0,   1200],
        [-5,   35  ]]
legend_pos = 'upper right'
legend_ncol = 4
FaceColor = 'w'

# Weight of the VIPER rover
tot_load = 430 * 9.81

# Plot fitted curve?
plot_fit = False

# Plot vs time curve every N_p point
N_p_crm = 80
N_p_scm = 40

# Range of time to get the mean of dbp or torque
t_range = [[2, 10.0], [2, 10.0]]

# Data directory for DBP and Torque
dir0 = "/srv/home/whu59"
# dir0 = "/home/whu59/research/server/euler"
dir1 = "/research/sbel/d_chrono_fsi_granular/chrono_3001/chrono_build/bin/DEMO_OUTPUT/"
dir2 = [["/FSI_Viper/final_real_wheel_dx1cm/slip"],
        ["/SCM_Viper/final_real_wheel_new_Janosi_Ks_15s/slip"]]
dir3 = "/DBP.txt"
slip = [["00", "003", "006", "01", "015", "02", "03", "04", "05", "06", "07", "08"],
        ["0.0", "0.03", "0.06", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
        [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
num_slip = len(slip[0])

out_dir = "05_plot_full_rover/"

DBP_Torque_Slope_Mean = np.zeros((7, num_slip))
DBP_Torque_Slope_Mean[0][:num_slip] = slip[2][:num_slip]

# Name of the plot and txt file
name_png_dbp = "final_viper_rover_dbp_vs_slip_crm_vs_scm.png"
name_png_slope = "final_viper_rover_slope_vs_slip_crm_vs_scm.png"
name_png_torque = "final_viper_rover_torque_vs_slip_crm_vs_scm.png"
name_txt_dbp_torque_slope_vs_slip = "DBP_Torque_Slope_vs_Slip_VIPER_Rover_CRM_And_SCM.txt"

# ============= Plot DBP VS time ===============
for n in range(2):
    if n == 0:
        print("Load data for the CRM simulation of VIPER rover")
    if n == 1:
        print("Load data for the SCM simulation of VIPER rover")
    # Plot DBP Torque vs time
    for k in range(2):
        if k == 0:
            print(" Plot DBP VS time")
        if k == 1:
            print(" Plot Torque VS time")
        plt.figure(figsize = fig_size)
        font = {'weight': Weight, 'size': fs}
        plt.rc('font', **font)
        for i in range(num_slip):
            # Find the directory of dbp data
            tot_dir = dir0 + dir1 + dir2[n][0] + slip[0][i] + dir3

            Time = []
            FT = []
            
            ni = 0
            val_tot = 0.0
            file = open(tot_dir,"r")
            for line in file:
                result = list(map(float, line.split("\t")))
                if len(result) < 4:
                    break
                Time.append(result[0])
                FT.append(result[5+k])
                if result[0] > t_range[n][0] and result[0] < t_range[n][1]:
                    val_tot = val_tot + result[5+k]
                    ni = ni + 1
            file.close()

            DBP_Torque_Slope_Mean[3 * n + k + 1][i] = val_tot / ni

            x = np.array(Time)
            y = np.array(FT)

            if plot_fit == True:
                p = P.polynomial.Polynomial.fit(x, y, deg = 15)
                y = p(x)

            N_p = N_p_crm
            if n == 1:
                N_p = N_p_scm

            plt.plot(x[::N_p], y[::N_p], linestyle = "--", lw = lw, label = slip[1][i])

        plt.grid(linestyle = '--')
        plt.legend(loc = legend_pos, ncol = legend_ncol)
        ax = plt.gca()
        ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
        if k == 0:
            ax.set_ylabel('DrawBar-Pull (N)', fontsize = fs, weight = Weight)
        if k == 1:
            ax.set_ylabel('Wheel Torque (Nm)', fontsize = fs, weight = Weight)
        ax.set_xlim([xlim[0][0], xlim[0][1]])
        ax.set_ylim([ylim[k][0], ylim[k][1]])

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(lw)
        ax.tick_params(width = lw)

        name_ = "final_viper_rover_"
        if k == 0:
            name_ = name_ + "dbp_vs_time"
        if k == 1:
            name_ = name_ + "torque_vs_time"
        if n == 0:
            name_ = name_ + "_crm.png"
        if n == 1:
            name_ = name_ + "_scm.png"
        plt.savefig(out_dir + name_, facecolor = FaceColor, dpi = DPI)
        # plt.show()

# ==============================================
# =============== CRM VS SCM ===================
# ==============================================
# ==============================================
# ========= DBP Torque Slope vs Slip ===========

# Loop for DBP, Torque and Slope
print("Plot comparison: CRM and SCM")
for k in range(3):
    if k == 0:
        print(" CRM VS SCM: Plot DBP VS Slip")
    if k == 1:
        print(" CRM VS SCM: Plot Torque VS Slip")
    if k == 2:
        print(" CRM VS SCM: Plot Slope VS Slip")
    plt.figure(figsize = fig_size)
    font = {'weight': Weight, 'size': fs}
    plt.rc('font', **font)

    x = DBP_Torque_Slope_Mean[0][:]
    y_crm = DBP_Torque_Slope_Mean[1][:]
    y_scm = DBP_Torque_Slope_Mean[4][:]
    if k == 1:
        y_crm = DBP_Torque_Slope_Mean[2][:]
        y_scm = DBP_Torque_Slope_Mean[5][:]
    if k == 2: 
        DBP_Torque_Slope_Mean[3][:] = 180 / math.pi * np.arctan(1.0 / tot_load * DBP_Torque_Slope_Mean[1][:])
        DBP_Torque_Slope_Mean[6][:] = 180 / math.pi * np.arctan(1.0 / tot_load * DBP_Torque_Slope_Mean[4][:])
        y_crm = DBP_Torque_Slope_Mean[3][:]
        y_scm = DBP_Torque_Slope_Mean[6][:]
    plt.plot(x, y_crm, 'rs--', lw = lw, markersize = ms, label = "CRM")
    plt.plot(x, y_scm, 'b*-.', lw = lw, markersize = ms, label = "SCM")

    plt.grid(linestyle = '--')
    plt.legend(loc='upper left')
    ax = plt.gca()

    # ax.set_title("No title")

    ax.set_xlabel('Slip', fontsize = fs, weight = Weight)
    if k == 0:
        ax.set_ylabel('DrawBar-Pull (N)', fontsize = fs, weight = Weight)
    if k == 1:
        ax.set_ylabel('Wheel Torque (Nm)', fontsize = fs, weight = Weight)
    if k == 2:
        ax.set_ylabel('Slope (deg)', fontsize = fs, weight = Weight)
    
    # Set xlim and ylim
    ax.set_xlim([-0.1, 0.9])
    if k == 0:
        ax.set_ylim([-500, 2500])
    if k == 1:
        ax.set_ylim([0, 1000])
    if k == 2:
        ax.set_ylim([-5, 35])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.tick_params(width = lw)

    # Set the output png name

    if k == 0:
        name = name_png_dbp
    if k == 1:
        name = name_png_torque
    if k == 2:
        name = name_png_slope
    plt.savefig(out_dir + name, facecolor = FaceColor, dpi = DPI)
    # plt.show()


# Save DBP Torque Slope vs Slip data into a text file
np.savetxt(out_dir + name_txt_dbp_torque_slope_vs_slip, DBP_Torque_Slope_Mean.transpose(), delimiter = " ")