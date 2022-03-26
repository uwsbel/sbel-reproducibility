#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# number of data points = 500
df_pos_coarse = pd.read_pickle("./position_coarse.pkl")
df_vel_coarse = pd.read_pickle("./velocity_coarse.pkl")
df_acc_coarse = pd.read_pickle("./acceleration_coarse.pkl")

# number of data points = 5,000
df_pos_med = pd.read_pickle("./position_med.pkl")
df_vel_med = pd.read_pickle("./velocity_med.pkl")
df_acc_med = pd.read_pickle("./acceleration_med.pkl")

# number of data points = 50,000
df_pos_fine = pd.read_pickle("./position_fine.pkl")
df_vel_fine = pd.read_pickle("./velocity_fine.pkl")
df_acc_fine = pd.read_pickle("./acceleration_fine.pkl")

# number of data points = 50,000,000
df_pos_y_ref = pd.read_pickle("./position_y_reference.pkl")
df_pos_z_ref = pd.read_pickle("./position_z_reference.pkl")

N_ref = 50000000
Ns = [500, 5000, 50000]

fig, axes = plt.subplots(3, 1, sharex=True)
fig.suptitle(r'Double pendulum: Comparison of the rA, rp, and r$\epsilon$ solutions to a reference solution')
fig.tight_layout()

sns.set(rc={'figure.figsize': (5, 3), "axes.labelsize": 10,
    "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8})

sns.set_style("ticks")
sns.set_context("paper")
sns.color_palette()

###### rA #######
# coarse
jump_coarse_z = int(N_ref/Ns[0])
delta_coarse_z = df_pos_z_ref['pos_b2'].tolist()[::jump_coarse_z] - df_pos_coarse['rA_Body2_z_pos']
# medium
jump_med_z = int(N_ref/Ns[1])
delta_med_z = df_pos_z_ref['pos_b2'].tolist()[::jump_med_z] - df_pos_med['rA_Body2_z_pos']
# fine
jump_fine_z = int(N_ref/Ns[2])
delta_fine_z = df_pos_z_ref['pos_b2'].tolist()[::jump_fine_z] - df_pos_fine['rA_Body2_z_pos']

sns.lineplot(ax=axes[0], x='Time', y=delta_coarse_z, data=df_pos_coarse, label='h=1e-2')
sns.lineplot(ax=axes[0], x='Time', y=delta_med_z, data=df_pos_med, label='h=1e-3')
sns.lineplot(ax=axes[0], x='Time', y=delta_fine_z, data=df_pos_fine, label='h=1e-4')
axes[0].set_ylabel(r'$\Delta z$ [m]')
axes[0].set_title("rA formulation")
#plt.show()

###### rp #######
# coarse
jump_coarse_z = int(N_ref/Ns[0])
delta_coarse_z = df_pos_z_ref['pos_b2'].tolist()[::jump_coarse_z] - df_pos_coarse['rp_Body2_z_pos']
# medium
jump_med_z = int(N_ref/Ns[1])
delta_med_z = df_pos_z_ref['pos_b2'].tolist()[::jump_med_z] - df_pos_med['rp_Body2_z_pos']
# fine
jump_fine_z = int(N_ref/Ns[2])
delta_fine_z = df_pos_z_ref['pos_b2'].tolist()[::jump_fine_z] - df_pos_fine['rp_Body2_z_pos']


sns.lineplot(ax=axes[1], x='Time', y=delta_coarse_z, data=df_pos_coarse, label='h=1e-2')
sns.lineplot(ax=axes[1], x='Time', y=delta_med_z, data=df_pos_med, label='h=1e-3')
sns.lineplot(ax=axes[1], x='Time', y=delta_fine_z, data=df_pos_fine, label='h=1e-4')
axes[1].set_ylabel(r'$\Delta z$ [m]')
axes[1].set_title("rp formulation")
#plt.show()

###### reps #######
# coarse
jump_coarse_z = int(N_ref/Ns[0])
delta_coarse_z = df_pos_z_ref['pos_b2'].tolist()[::jump_coarse_z] - df_pos_coarse['reps_Body2_z_pos']
# medium
jump_med_z = int(N_ref/Ns[1])
delta_med_z = df_pos_z_ref['pos_b2'].tolist()[::jump_med_z] - df_pos_med['reps_Body2_z_pos']
# fine
jump_fine_z = int(N_ref/Ns[2])
delta_fine_z = df_pos_z_ref['pos_b2'].tolist()[::jump_fine_z] - df_pos_fine['reps_Body2_z_pos']

sns.lineplot(ax=axes[2], x='Time', y=delta_coarse_z, data=df_pos_coarse, label='h=1e-2')
sns.lineplot(ax=axes[2], x='Time', y=delta_med_z, data=df_pos_med, label='h=1e-3')
sns.lineplot(ax=axes[2], x='Time', y=delta_fine_z, data=df_pos_fine, label='h=1e-4')
axes[2].set_ylabel(r'$\Delta z$ [m]')
axes[2].set_title("rε formulation")
axes[2].set_xlabel('Time [s]')


plt.legend()
plt.savefig('Solution_Deltas.png', dpi=800, bbox_inches='tight')
plt.show()