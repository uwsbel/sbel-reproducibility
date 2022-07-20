#!/usr/bin/env python3

import pickle
import os
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

dir_path = './output/oa1_four_link_kinematics/'

plt.rcParams.update({'font.size': 30})

files = []
for f in os.listdir(dir_path):
    if f == 'steps.pickle':
        with open(dir_path + f, 'rb') as handle:
            step_sizes = pickle.load(handle)

        continue

    if f.endswith('.pickle') and f.startswith('Four Link'):
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
    df_pos[header] = pos_diff
    df_vel[header] = vel_diff
    df_acc[header] = acc_diff

    # title = '{} {} Order Analysis: Body {}, {}'.format(name, form, body, component)
    #
    # _, ax = plt.subplots()
    # ax.loglog(step_sizes, [step for step in step_sizes], label='Order 1 trendline', markersize=15, linewidth=7)
    # #ax.loglog(step_sizes, [step*step for step in step_sizes], label='Order 2 trendline', markersize=15, linewidth=7)
    # ax.loglog(step_sizes, pos_diff, label='Position', marker='o', linestyle='-', markersize=15, linewidth=7)
    # ax.loglog(step_sizes, vel_diff, label='Velocity', marker='x', linestyle='-', markersize=15, linewidth=7)
    # ax.loglog(step_sizes, acc_diff, label='Acceleration', marker='+', linestyle='-', markersize=15, linewidth=7)
    # ax.set(xlabel='Step Size', ylabel='Absolute Difference', title=title)
    #
    # ax.legend(loc='lower right', prop={'size': 24})
    #
    # plt.gcf().set_size_inches(20, 12)
    # plt.savefig('./plots/oa1_four_link/Four_Link_OrderAnalysis_Body_{}_{}.png'.format(form, body, component))

# plt.show()
sns.set(rc={'figure.figsize': (5, 3), "axes.labelsize": 10,
    "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8})

sns.set_style("ticks")
sns.set_context("paper")
#sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})

vel_title = 'Four Link Mechanism Order Analysis, z-Velocity of Link 3'
# only grab link 2 z-velocity data
link_2_vel_z = df_vel[['h', 'trend', 'rA_Body2_z', 'rε_Body2_z', 'rp_Body2_z']]
link_2_vel_z = link_2_vel_z.set_index(['h'])
# reorganize df to classic table
link_2_vel_z_new = link_2_vel_z.stack().reset_index()
link_2_vel_z_new.columns = ['Step size', 'Form', 'Absolute difference']
vel_plot = sns.lineplot(x='Step size', y='Absolute difference', hue='Form', style='Form', data=link_2_vel_z_new,
                        markers=[".", "v", "o", "s"], legend='full')
new_title = 'Formulation'
new_labels = ['Order 1 trendline', 'rA', 'rε', 'rp']
vel_plot.legend(title=new_title, labels=new_labels)
vel_plot.set(xscale="log")
vel_plot.set(yscale="log")
vel_plot.set_title(vel_title)
plt.savefig('Four_Link_Order_Analysis_Body_2_z_Velocity.png', dpi=800, bbox_inches='tight')
plt.show()

sns.set(rc={'figure.figsize': (5, 3), "axes.labelsize": 10,
    "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8})
sns.set_style("ticks")
sns.set_context("paper")
acc_title = 'Four Link Mechanism Order Analysis, z-Acceleration of Link 3'
# only grab link 2 z-velocity data
link_2_acc_z = df_acc[['h', 'trend', 'rA_Body2_z', 'rε_Body2_z', 'rp_Body2_z']]
link_2_acc_z['trend'] = 100 * link_2_acc_z['trend']  # shift trendline up
link_2_acc_z = link_2_acc_z.set_index(['h'])
# reorganize df to classic table
link_2_acc_z_new = link_2_acc_z.stack().reset_index()
link_2_acc_z_new.columns = ['Step size', 'Form', 'Absolute difference']
acc_plot = sns.lineplot(x='Step size', y='Absolute difference', hue='Form', style='Form', data=link_2_acc_z_new,
                        markers=[".", "v", "o", "s"], legend='full')
new_title = 'Formulation'
new_labels = ['Order 1 trendline', 'rA', 'rε', 'rp']
acc_plot.legend(title=new_title, labels=new_labels)
acc_plot.set(xscale="log")
acc_plot.set(yscale="log")
acc_plot.set_title(acc_title)
plt.savefig('Four_Link_Order_Analysis_Body_2_z_Acc.png', dpi=800, bbox_inches='tight')
plt.show()