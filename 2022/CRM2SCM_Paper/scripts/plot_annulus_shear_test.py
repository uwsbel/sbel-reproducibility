import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import polynomial as P
import math

# General information of the data and plots
lw = 2
fs = 14
ms = 14
DPI = 300
Ncol = 4
Weight = 'bold'

# This is the start time and end time that used to calculate 
# the average value of the torque at steady state
t_start = 1.25
t_end = 12.5

# dir0 = "/srv/home/whu59/"
dir0 = "/home/whu59/research/server/euler"
dir1 = "/research/sbel/d_chrono_fsi_granular/chrono_3001"
dir2 = "/chrono_build/bin/DEMO_OUTPUT/FSI_Bevameter/21_r20cm_r15cm_"
dir3 = "kg_1deg/Torque.txt"

out_dir = "02_plot_annulus_shear/"

# Max torque at each load: CRM and SCM (Janosi Hanamoto)
Torque_Max = np.zeros((3, 8))

# Selected torque at each load and at three different time (1s, 2s, 3s)
# This is used to calibrate Ks in Janosi Hanamoto model
Torque_Selected_Points = np.zeros((4, 8))

# ===================================================================
plt.figure(figsize = (9, 5))
font = {'weight': Weight, 'size': fs}
plt.rc('font', **font)
for i in range(8):
    # Get the mass of the annulus and find the data directory
    # Mass changes from 25kg to 200kg
    mass = 25 * (i + 1)
    stuff_in_string = "{}".format(mass)
    dir_tot = dir0 + dir1 + dir2 + stuff_in_string + dir3

    ni = 0
    val_tot = 0.0
    Time = []
    Torque = []

    file = open(dir_tot,"r")
    for line in file:
        result = list(map(float, line.split("\t")))
        if len(result) < 4:
            break
        Time.append(result[0])
        Torque.append(-result[4])
        if result[0] > t_start and result[0] < t_end:
                val_tot = val_tot - result[4]
                ni = ni + 1
    file.close()

    # Calculate the max torque from CRM, compare with SCM (Janosi Hanamoto) later
    Torque_Max[0][i] = mass
    Torque_Max[1][i] = val_tot / ni

    x = np.array(Time)
    y = np.array(Torque)

    # Fit the curve
    p = P.polynomial.Polynomial.fit(x, y, deg = 15)
    yvals = p(x)

    # Selected points at t = 1s, 2s, and 3s
    Torque_Selected_Points[0][i] = mass
    Torque_Selected_Points[1][i] = p(1.0)
    Torque_Selected_Points[2][i] = p(2.0)
    Torque_Selected_Points[3][i] = p(3.0)

    plt.plot(x, yvals, linestyle = "-", lw = lw, label = stuff_in_string + " kg")

plt.grid(linestyle = '--')
plt.legend(loc='upper left', ncol = Ncol)
ax = plt.gca()
ax.set_xlabel('Time (s)', fontsize = fs, weight = Weight)
ax.set_ylabel('Torque (Nm)', fontsize = fs, weight = Weight)
ax.set_xlim([0, 12])
ax.set_ylim([0, 250])

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(lw)
ax.tick_params(width = lw)

name_png = out_dir + "final_annulus_torque_vs_time_diff_load_fitted.png"
print("Plot " + name_png)
plt.savefig(name_png, facecolor = 'w', dpi = DPI)

# Save the selected torque at different time for each load
selected_torque = out_dir + "annulus_selected_torque_points_for_each_load.txt"
print("Save " + selected_torque)
np.savetxt(selected_torque, Torque_Selected_Points.transpose(), delimiter = " ")

# ===================================================================
# Phi and coh are obtained from calibration
# Here, compare against SCM (Janosi Hanamoto) 
Phi = 24
coh = 2495
A = math.pi * ( 0.2 * 0.2 - 0.15 * 0.15)
L = (0.2 + 0.15) / 2.0
for i in range(8):
    mass = 25 * (i + 1)
    T_max = (coh * A + mass * 9.81 * math.tan(math.pi * Phi / 180 )) * L
    Torque_Max[2][i] = T_max

# Save the torque vs load into a txt file, crm vs scm (Janosi Hanamoto)
annulus_txt_name = out_dir + "annulus_max_torque_vs_load_crm_vs_scm.txt"
print("Save " + annulus_txt_name)
np.savetxt(annulus_txt_name, Torque_Max.transpose(), delimiter = " ")

# Plot a comparision between CRM and SCM (Janosi Hanamoto)
plt.figure(figsize = (8, 6))
font = {'weight': Weight, 'size': fs}
plt.rc('font', **font)

plt.plot(Torque_Max[0][:], Torque_Max[1][:], 'rs--', lw = lw, 
    markersize = ms, fillstyle='none', markeredgewidth = lw, label = "CRM")
plt.plot(Torque_Max[0][:], Torque_Max[2][:], 'b*-.', lw = lw, 
    markersize = ms+3, fillstyle='none', markeredgewidth = lw, label = "SCM (Janosi Hanamoto)")

plt.grid(linestyle = '--')
plt.legend(loc='upper left')
ax = plt.gca()
ax.set_xlabel('Load (kg)', fontsize = fs, weight = Weight)
ax.set_ylabel('Torque (Nm)', fontsize = fs, weight = Weight)
ax.set_xlim([0, 225])
ax.set_ylim([0, 225])

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(lw)
ax.tick_params(width = lw)
name_png = out_dir + "final_annulus_max_torque_vs_load_crm_vs_scm.png"
print("Plot " + name_png)
plt.savefig(name_png, facecolor = 'w', dpi = DPI)