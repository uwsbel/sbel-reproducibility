#!/usr/bin/env python3

import pickle
import os
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

dir_path = './output/oa_single_pendulum_kinematics_c2/'

plt.rcParams.update({'font.size': 30})

files = []
for f in os.listdir(dir_path):
    if f == 'steps.pickle':
        with open(dir_path + f, 'rb') as handle:
            step_sizes = pickle.load(handle)

        continue

    if f.endswith('.pickle') and f.startswith('Single_Pendulum'):
        files.append(f)

# store results in dataframe for plotting later
df_pos = pd.DataFrame()
df_vel = pd.DataFrame()
df_acc = pd.DataFrame()
df_pos['h'] = step_sizes
df_vel['h'] = step_sizes
df_acc['h'] = step_sizes
df_pos['trend'] = [step for step in step_sizes]
df_vel['trend'] = [step for step in step_sizes]
df_acc['trend'] = [step for step in step_sizes]

for file_name in files:
    with open(dir_path + file_name, 'rb') as handle:
        info, pos_diff, vel_diff, acc_diff = pickle.load(handle)

    name, form, body, component = info
    # save data in dataframe
    header = '{}_Body{}_{}'.format(form, body, component)
    df_pos[header+'_pos'] = pos_diff
    df_vel[header+'_vel'] = vel_diff
    df_acc[header+'_acc'] = acc_diff


sns.set(rc={'figure.figsize': (5, 3), "axes.labelsize": 10,
    "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8})

sns.set_style("ticks")
sns.set_context("paper")
#sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})

vel_title = 'Single Pendulum Mechanism Order Analysis, z-Coordinate of Pendulum '
# only grab link 2 z-velocity data
body_0_z = df_pos[['h', 'trend', 'rA_Body0_z_pos']]
body_0_z = body_0_z.append(df_vel[['h', 'rA_Body0_z_vel']])
body_0_z = body_0_z.append(df_acc[['h','rA_Body0_z_acc']])
body_0_z = body_0_z.set_index(['h'])
# reorganize df to classic table
body_0_z_new = body_0_z.stack().reset_index()
print(body_0_z_new)
body_0_z_new.columns = ['Step size', 'Level', 'Absolute difference']
vel_plot = sns.lineplot(x='Step size', y='Absolute difference', hue='Level', style='Level', data=body_0_z_new,
                        markers=[".", "v", "o", "s"], legend='full')
new_title = 'Formulation'
new_labels = ['Order 1 trendline', 'Position', 'Velocity', 'Acceleration']
vel_plot.legend(title=new_title, labels=new_labels)
vel_plot.set(xscale="log")
vel_plot.set(yscale="log")
vel_plot.set_title(vel_title)
plt.savefig('single_pendulum_order_analysis_rA_body0_z.png', dpi=800, bbox_inches='tight')
plt.show()