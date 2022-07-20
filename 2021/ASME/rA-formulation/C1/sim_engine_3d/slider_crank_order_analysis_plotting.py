#!/usr/bin/env python3

import pickle
import os
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

dir_path = './output/oa1_slider_crank_kinematics/'

plt.rcParams.update({'font.size': 30})

files = []
for f in os.listdir(dir_path):
    if f == 'steps.pickle':
        with open(dir_path + f, 'rb') as handle:
            step_sizes = pickle.load(handle)

        continue

    if f.endswith('.pickle') and f.startswith('Slider Crank'):
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

sns.set(rc={'figure.figsize': (5, 3), "axes.labelsize": 10,
    "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8})

sns.set_style("ticks")
sns.set_context("paper")

def plot(data, title, filename):
    data = data.set_index(['h'])
    # reorganize df to classic table
    data_new = data.stack().reset_index()
    data_new.columns = ['Step size', 'Form', 'Absolute difference']
    lp_obj = sns.lineplot(x='Step size', y='Absolute difference', hue='Form', style='Form', data=data_new,
                            markers=[".", "v", "o", "s"], legend='full')
    new_title = 'Formulation'
    new_labels = ['Order 1 trendline', 'rA', 'rε', 'rp']
    lp_obj.legend(title=new_title, labels=new_labels)
    lp_obj.set(xscale="log")
    lp_obj.set(yscale="log")
    lp_obj.set_title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##################################################################################################################
#################################### x coordinate, Slider (body 2) ###############################################
##################################################################################################################

slider_pos_x_title = 'Slider Crank Mechanism Order Analysis, x-Position of Slider'
slider_pos_x_filename = 'Slider_Crank_Order_Analysis_Body_2_x_Position.png'
slider_pos_x = df_pos[['h', 'trend', 'rA_Body2_x', 'rε_Body2_x', 'rp_Body2_x']]
plot(slider_pos_x, slider_pos_x_title, slider_pos_x_filename)


slider_vel_x_title = 'Slider Crank Mechanism Order Analysis, x-Velocity of Slider'
slider_vel_x_filename = 'Slider_Crank_Order_Analysis_Body_2_x_Velocity.png'
slider_vel_x = df_vel[['h', 'trend', 'rA_Body2_x', 'rε_Body2_x', 'rp_Body2_x']]
plot(slider_vel_x, slider_vel_x_title, slider_vel_x_filename)

slider_acc_x_title = 'Slider Crank Mechanism Order Analysis, x-Acceleration of Slider'
slider_acc_x_filename = 'Slider_Crank_Order_Analysis_Body_2_x_Acceleration.png'
slider_acc_x = df_acc[['h', 'trend', 'rA_Body2_x', 'rε_Body2_x', 'rp_Body2_x']]
plot(slider_acc_x, slider_acc_x_title, slider_acc_x_filename)

##################################################################################################################
#################################### y coordinate, Slider (body 2) ###############################################
##################################################################################################################

slider_pos_y_title = 'Slider Crank Mechanism Order Analysis, y-Position of Slider'
slider_pos_y_filename = 'Slider_Crank_Order_Analysis_Body_2_y_Position.png'
slider_pos_y = df_pos[['h', 'trend', 'rA_Body2_y', 'rε_Body2_y', 'rp_Body2_y']]
plot(slider_pos_y, slider_pos_y_title, slider_pos_y_filename)


slider_vel_y_title = 'Slider Crank Mechanism Order Analysis, y-Velocity of Slider'
slider_vel_y_filename = 'Slider_Crank_Order_Analysis_Body_2_y_Velocity.png'
slider_vel_y = df_vel[['h', 'trend', 'rA_Body2_y', 'rε_Body2_y', 'rp_Body2_y']]
plot(slider_vel_y, slider_vel_y_title, slider_vel_y_filename)

slider_acc_y_title = 'Slider Crank Mechanism Order Analysis, y-Acceleration of Slider'
slider_acc_y_filename = 'Slider_Crank_Order_Analysis_Body_2_y_Acceleration.png'
slider_acc_y = df_acc[['h', 'trend', 'rA_Body2_y', 'rε_Body2_y', 'rp_Body2_y']]
plot(slider_acc_y, slider_acc_y_title, slider_acc_y_filename)

##################################################################################################################
#################################### z coordinate, Slider (body 2) ###############################################
##################################################################################################################

slider_pos_z_title = 'Slider Crank Mechanism Order Analysis, z-Position of Slider'
slider_pos_z_filename = 'Slider_Crank_Order_Analysis_Body_2_z_Position.png'
slider_pos_z = df_pos[['h', 'trend', 'rA_Body2_z', 'rε_Body2_z', 'rp_Body2_z']]
plot(slider_pos_z, slider_pos_z_title, slider_pos_z_filename)


slider_vel_z_title = 'Slider Crank Mechanism Order Analysis, z-Velocity of Slider'
slider_vel_z_filename = 'Slider_Crank_Order_Analysis_Body_2_z_Velocity.png'
slider_vel_z = df_vel[['h', 'trend', 'rA_Body2_z', 'rε_Body2_z', 'rp_Body2_z']]
plot(slider_vel_z, slider_vel_z_title, slider_vel_z_filename)

slider_acc_z_title = 'Slider Crank Mechanism Order Analysis, z-Acceleration of Slider'
slider_acc_z_filename = 'Slider_Crank_Order_Analysis_Body_2_z_Acceleration.png'
slider_acc_z = df_acc[['h', 'trend', 'rA_Body2_z', 'rε_Body2_z', 'rp_Body2_z']]
plot(slider_acc_z, slider_acc_z_title, slider_acc_z_filename)

