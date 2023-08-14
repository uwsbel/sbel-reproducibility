import sys
import matplotlib.pyplot as mpl
import pandas as pd


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


##################### Torque Map ####################################
rpms = [-100, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2700]
tors = [300, 382, 490, 579, 650, 706, 746, 774, 789, 793, 788, 774, 754, 728, 697, 664, 628, 593, 558, -400]

fig,ax = mpl.subplots()
ax.plot(rpms, tors,color = 'k')
ax.scatter(rpms,tors, color = 'k',marker = 'o')
# mpl.title("Engine Torque Map", fontsize = 16)
ax.set_xlabel("RPM")
ax.set_ylabel("Torque [Nm]")
ax.set_title("Torque Map")
ax.set_xlim([min(rpms), max(rpms)+50])
ax.set_ylim([min(tors)-20, max(tors)+20])
fig.tight_layout()
mpl.savefig('./images/tor_map.eps', format='eps', dpi=3000)
mpl.show()



####################### Losses Map ################################

rpms = [-50, 0, 50, 1000, 2000, 3000]
tors = [30, 0, -30, -50, -70, -90]

fig,ax = mpl.subplots()
ax.plot(rpms, tors, color = 'k')
ax.scatter(rpms, tors, color = 'k', marker = 'o')
ax.set_title("Loss Map")
ax.set_xlabel("RPM")
ax.set_ylabel("Losses [Nm]")
mpl.xlim([min(rpms)-50, max(rpms)+50])
mpl.ylim([min(tors)-20, max(tors)+20])
fig.tight_layout()
mpl.savefig('./images/loss_map.eps', format='eps', dpi=3000)
mpl.show()


###################### Capacity factor Map #####################

rpms = [0, 0.25, 0.5, 0.75, 0.9, 1]
tors = [15, 15, 15, 16, 18, 35]

fig, ax = mpl.subplots()
ax.plot(rpms, tors, color = 'k')
ax.scatter(rpms, tors, color = 'k', marker = 'o')
ax.set_title("Capacity Factor Map")
ax.set_xlabel(r'Speed Ratio $\left(\dfrac{\omega_{out}}{\omega_{in}}\right)$')
ax.set_ylabel(r'Capacity Factor ($K$) $\left[\dfrac{rad/s}{\sqrt{Nm}}\right]$')
ax.set_xlim([min(rpms), max(rpms) + 0.02])
ax.set_ylim([min(tors)-3, max(tors)+10])
fig.tight_layout()
mpl.savefig('./images/CF_map.eps', format='eps', dpi=3000)
mpl.show()


###################### Torque Ratio map #########################


rpms = [0, 0.25, 0.5, 0.75, 1]
tors = [2, 1.8, 1.5, 1.15, 1]


fig, ax = mpl.subplots()
ax.plot(rpms, tors, color = 'k')
ax.scatter(rpms, tors, color = 'k', marker = 'o')
ax.set_title("Torque Ratio Map" )
ax.set_xlabel(r'Speed Ratio $\left(\dfrac{\omega_{out}}{\omega_{in}}\right)$')
ax.set_ylabel(r'Torque Ratio ($T$)')
mpl.xlim([min(rpms), max(rpms) + 0.02])
mpl.ylim([min(tors)-0.02, max(tors)+0.25])
fig.tight_layout()
mpl.savefig('./images/TR_map.eps', format='eps', dpi=3000)
mpl.show()