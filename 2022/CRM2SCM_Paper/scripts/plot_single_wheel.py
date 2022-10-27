import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import polynomial as P

# ==============================================
# Loop for two different wheels: HMMWV and VIPER
# ==============================================
for n_type in range(2):
    # Choose a wheel type
    wheel_type = "HMMWV_Wheel"
    if n_type == 1:
        wheel_type = "VIPER_Wheel"
    print("Processing data for " + wheel_type)

    # General information of the data and plots
    lw = 2
    fs = 14
    ms = 14
    DPI = 300
    Weight = 'bold'
    fig_size = [8, 6]
    xlim = [0, 15]
    ylim_dbp = [0, 900]
    ylim_torque = [0, 600]
    if wheel_type == "VIPER_Wheel":
        xlim = [0, 10]
        ylim_dbp = [-100, 700]
        ylim_torque = [0, 280]
    legend_pos = 'upper right'
    legend_ncol = 4
    FaceColor = 'w'

    tot_load = 108.22 * 9.81

    # Plot fitted curve?
    plot_fit = False

    # Plot vs time curve every N_p point
    N_p = 10

    # Data directory for DBP and Torque
    dir0 = "/srv/home/whu59/"
    # dir0 = "/home/whu59/research/server/euler"
    dir1 = "/research/sbel/d_chrono_fsi_granular/chrono_3001/chrono_build/bin/DEMO_OUTPUT/FSI_Single_Wheel_Test"
    dir2_hmmwv = "/Regular_HMMWV_Tire/dx1cm/31-108kg-slip"
    dir2_viper = "/Real_VIPER_Wheel/new_dx10mm_h10mm_height15cm/01-108kg-slip"
    dir2 = dir2_hmmwv
    if wheel_type == "VIPER_Wheel":
        dir2 = dir2_viper

    slip_hmmwv = [["00", "003", "006", "01", "02", "03", "04", "05", "06", "07", "08"],
                  ["0.0", "0.03", "0.06", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                  [0.0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    slip_viper = [["00", "003", "006", "01", "015", "02", "03", "04", "05", "06", "07", "08"],
                  ["0.0", "0.03", "0.06", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                  [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    slip = slip_hmmwv
    if wheel_type == "VIPER_Wheel":
        slip = slip_viper

    dir3_dbp = "/DBP.txt"
    dir3_torque = "/Torque.txt"
    num_slip = len(slip[0])

    out_dir = "04_plot_single_wheel/"

    DBP_Torque_Mean = np.zeros((3, num_slip))
    DBP_Torque_Mean[0][:num_slip] = slip[2][:num_slip]
    DBP_Torque_Mean_SCM = np.zeros((3, num_slip))
    DBP_Torque_Mean_SCM[0][:num_slip] = slip[2][:num_slip]
    t_start = 5

    # Name of the plot and txt file
    name_png_dbp_crm = "final_hmmwv_wheel_dbp_vs_time_crm.png"
    name_png_torque_crm = "final_hmmwv_wheel_torque_vs_time_crm.png"
    name_txt_dbp_torque_vs_slip_crm = "DBP_Torque_vs_Slip_HMMWV_Wheel_CRM.txt"
    if wheel_type == "VIPER_Wheel":
        name_png_dbp_crm = "final_viper_wheel_dbp_vs_time_crm.png"
        name_png_torque_crm = "final_viper_wheel_torque_vs_time_crm.png"
        name_txt_dbp_torque_vs_slip_crm = "DBP_Torque_vs_Slip_VIPER_Wheel_CRM.txt"

    # Name of the plot and txt file
    name_png_dbp_scm = "final_hmmwv_wheel_dbp_vs_time_scm.png"
    name_png_torque_scm = "final_hmmwv_wheel_torque_vs_time_scm.png"
    name_txt_dbp_torque_vs_time_scm = "DBP_Torque_vs_Time_HMMWV_Wheel_SCM.txt"
    name_txt_dbp_torque_vs_slip_scm = "DBP_Torque_vs_Slip_HMMWV_Wheel_SCM.txt"
    if wheel_type == "VIPER_Wheel":
        name_png_dbp_scm = "final_viper_wheel_dbp_vs_time_scm.png"
        name_png_torque_scm = "final_viper_wheel_torque_vs_time_scm.png"
        name_txt_dbp_torque_vs_time_scm = "DBP_Torque_vs_Time_VIPER_Wheel_SCM.txt"
        name_txt_dbp_torque_vs_slip_scm = "DBP_Torque_vs_Slip_VIPER_Wheel_SCM.txt"

    # ==============================================
    # ==================== CRM =====================
    # ==============================================
    # ==============================================
    # ============= Plot DBP VS time ===============
    print(" CRM: Plot DBP VS time")
    plt.figure(figsize = fig_size)
    font = {'weight': Weight, 'size': fs}
    plt.rc('font', **font)
    for i in range(num_slip):
        # Find the directory of dbp data
        tot_dir = dir0 + dir1 + dir2 + slip[0][i] + dir3_dbp

        file = open(tot_dir,"r")
        Time = []
        DBP = []

        val_tot = 0.0
        ni = 0
        for line in file:
            result = list(map(float, line.split("\t")))
            if len(result) < 4:
                break
            Time.append(result[0])
            DBP.append(result[3])
            if result[0] > t_start:
                val_tot = val_tot + result[3]
                ni = ni + 1
        file.close()

        DBP_Torque_Mean[1][i] = val_tot / ni

        x = np.array(Time)
        y = np.array(DBP)

        if plot_fit == True:
            p = P.polynomial.Polynomial.fit(x, y, deg = 15)
            y = p(x)

        plt.plot(x[::N_p], y[::N_p], linestyle = "--", lw = lw, label = slip[1][i])

    plt.grid(linestyle = '--')
    plt.legend(loc = legend_pos, ncol = legend_ncol)
    ax = plt.gca()
    # ax.set_title("No title")
    ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
    ax.set_ylabel('DrawBar-Pull (N)', fontsize = fs, weight = Weight)
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim_dbp[0], ylim_dbp[1]])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.tick_params(width = lw)

    plt.savefig(out_dir + name_png_dbp_crm, facecolor = FaceColor, dpi = DPI)
    # plt.show()

    # ==============================================
    # ============ Plot Torque VS time =============
    print(" CRM: Plot Torque VS time")
    plt.figure(figsize = fig_size)
    font = {'weight': Weight, 'size': fs}
    plt.rc('font', **font)
    for i in range(num_slip):
        # Find the directory of torque data
        tot_dir = dir0 + dir1 + dir2 + slip[0][i] + dir3_torque

        file = open(tot_dir,"r")
        Time = []
        Torque = []

        val_tot = 0.0
        ni = 0
        for line in file:
            result = list(map(float, line.split("\t")))
            if len(result) < 4:
                break
            Time.append(result[0])
            Torque.append(-result[5])
            if result[0] > t_start:
                val_tot = val_tot -result[5]
                ni = ni + 1
        file.close()

        DBP_Torque_Mean[2][i] = val_tot / ni

        x = np.array(Time)
        y = np.array(Torque)

        if plot_fit == True:
            p = P.polynomial.Polynomial.fit(x, y, deg = 15)
            y = p(x)

        plt.plot(x[::N_p], y[::N_p], linestyle = "--", lw = lw, label = slip[1][i])

    plt.grid(linestyle = '--')
    plt.legend(loc = legend_pos, ncol = legend_ncol)
    ax = plt.gca()
    # ax.set_title("No title")
    ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
    ax.set_ylabel('Wheel Torque (Nm)', fontsize = fs, weight = Weight)
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim_torque[0], ylim_torque[1]])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.tick_params(width = lw)

    plt.savefig(out_dir + name_png_torque_crm, facecolor = FaceColor, dpi = DPI)
    # plt.show()

    # Save DBP Torque vs Slip data into a text file
    np.savetxt(out_dir + name_txt_dbp_torque_vs_slip_crm, DBP_Torque_Mean.transpose(), delimiter = " ")

    # ==============================================
    # ==================== SCM =====================
    # ==============================================
    # ==============================================
    # ============= Plot DBP VS time ===============
    print(" SCM: Plot DBP VS time")
    plt.figure(figsize = fig_size)
    font = {'weight': Weight, 'size': fs}
    plt.rc('font', **font)
    for i in range(num_slip):
        # Find the directory of dbp data
        tot_dir = out_dir + name_txt_dbp_torque_vs_time_scm

        Time = []
        DBP = []
        file = open(tot_dir,"r")
        for line in file:
            result = list(map(float, line.split(" ")))
            if len(result) < 4:
                break
            Time.append(result[0])
            DBP.append(result[1 + i])
        file.close()

        x = np.array(Time)
        y = np.array(DBP)

        if plot_fit == True:
            p = P.polynomial.Polynomial.fit(x, y, deg = 15)
            y = p(x)

        plt.plot(x[::N_p], y[::N_p], linestyle = "--", lw = lw, label = slip[1][i])

    plt.grid(linestyle = '--')
    plt.legend(loc = legend_pos, ncol = legend_ncol)
    ax = plt.gca()
    # ax.set_title("No title")
    ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
    ax.set_ylabel('DrawBar-Pull (N)', fontsize = fs, weight = Weight)
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim_dbp[0], ylim_dbp[1]])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.tick_params(width = lw)

    plt.savefig(out_dir + name_png_dbp_scm, facecolor = FaceColor, dpi = DPI)
    # plt.show()

    # ==============================================
    # ============= Plot Torque VS time ============
    print(" SCM: Plot Torque VS time")
    plt.figure(figsize = fig_size)
    font = {'weight': Weight, 'size': fs}
    plt.rc('font', **font)
    for i in range(num_slip):
        # Find the directory of dbp data
        tot_dir = out_dir + name_txt_dbp_torque_vs_time_scm

        Time = []
        Torque = []
        file = open(tot_dir,"r")
        for line in file:
            result = list(map(float, line.split(" ")))
            if len(result) < 4:
                break
            Time.append(result[0])
            Torque.append(result[i + num_slip + 1])
        file.close()

        x = np.array(Time)
        y = np.array(Torque)

        if plot_fit == True:
            p = P.polynomial.Polynomial.fit(x, y, deg = 15)
            y = p(x)

        plt.plot(x[::N_p], y[::N_p], linestyle = "--", lw = lw, label = slip[1][i])

    plt.grid(linestyle = '--')
    plt.legend(loc = legend_pos, ncol = legend_ncol)
    ax = plt.gca()
    # ax.set_title("No title")
    ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
    ax.set_ylabel('Wheel Torque (Nm)', fontsize = fs, weight = Weight)
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim_torque[0], ylim_torque[1]])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.tick_params(width = lw)

    plt.savefig(out_dir + name_png_torque_scm, facecolor = FaceColor, dpi = DPI)
    # plt.show()


    # ==============================================
    # ================= CRM VS SCM =================
    # ==============================================
    # ==============================================
    # ============== DBP Torque vs Slip ============
    tot_dir = out_dir + name_txt_dbp_torque_vs_slip_scm
    nl = 0
    file = open(tot_dir,"r")
    for line in file:
        result = list(map(float, line.split(" ")))
        DBP_Torque_Mean_SCM[0][nl] = result[0]
        DBP_Torque_Mean_SCM[1][nl] = result[1]
        DBP_Torque_Mean_SCM[2][nl] = result[2]
        nl = nl + 1
    file.close()

    # Loop for DBP, Torque and Slope
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

        x_crm = DBP_Torque_Mean[0][:]
        x_scm = DBP_Torque_Mean_SCM[0][:]

        y_crm = DBP_Torque_Mean[1][:]
        y_scm = DBP_Torque_Mean_SCM[1][:]
        if k == 1:
            y_crm = DBP_Torque_Mean[2][:]
            y_scm = DBP_Torque_Mean_SCM[2][:]
        if k == 2: 
            y_crm = 180 / math.pi * np.arctan(1.0 / tot_load * DBP_Torque_Mean[1][:])
            y_scm = 180 / math.pi * np.arctan(1.0 / tot_load * DBP_Torque_Mean_SCM[1][:])
        plt.plot(x_crm, y_crm, 'rs--', lw = lw, markersize = ms, label = "CRM")
        plt.plot(x_scm, y_scm, 'b*-.', lw = lw, markersize = ms, label = "SCM")

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
        if n_type == 0 and k == 0:
            ax.set_ylim([-100, 800])
        if n_type == 0 and k == 1:
            ax.set_ylim([0, 500])
        if n_type == 1 and k == 0:
            ax.set_ylim([-100, 700])
        if n_type == 1 and k == 1:
            ax.set_ylim([0, 250])

        if k == 2:
            ax.set_ylim([-5, 35])

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(lw)
        ax.tick_params(width = lw)

        # Set the output png name
        name_base = "final_hmmwv_wheel"
        if n_type == 1:
            name_base = "final_viper_wheel"

        name = name_base + "_dbp_vs_slip_crm_vs_scm.png"
        if k == 1:
            name = name_base + "_torque_vs_slip_crm_vs_scm.png"
        if k == 2:
            name = name_base + "_slope_vs_slip_crm_vs_scm.png"
        plt.savefig(out_dir + name, facecolor = FaceColor, dpi = DPI)
        # plt.show()