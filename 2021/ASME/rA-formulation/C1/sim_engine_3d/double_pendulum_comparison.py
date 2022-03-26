#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mechanisms.double_pendulum import double_pendulum
import double_pend_ODE

# store results in dataframe for plotting later
df_pos = pd.DataFrame()
df_vel = pd.DataFrame()
df_acc = pd.DataFrame()


def run_model(args, store_grid=True):
    form, model_fn, num_bodies = args
    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])

    pos_data, vel_data, acc_data, _, grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', '0.1',
                                             '--step_size', '0.00001', '-t', '5'])

    if store_grid:
        df_pos['Time'] = grid
        df_vel['Time'] = grid
        df_acc['Time'] = grid

    # save data in dataframes
    pos_header = '{}_Body1_z_pos'.format(form)
    vel_header = '{}_Body1_z_vel'.format(form)
    acc_header = '{}_Body1_z_acc'.format(form)
    df_pos[pos_header] = pos_data[0, 2, :]
    df_vel[vel_header] = vel_data[0, 2, :]
    df_acc[acc_header] = acc_data[0, 2, :]

    pos_header = '{}_Body1_y_pos'.format(form)
    vel_header = '{}_Body1_y_vel'.format(form)
    acc_header = '{}_Body1_y_acc'.format(form)
    df_pos[pos_header] = pos_data[0, 1, :]
    df_vel[vel_header] = vel_data[0, 1, :]
    df_acc[acc_header] = acc_data[0, 1, :]

    pos_header = '{}_Body2_z_pos'.format(form)
    vel_header = '{}_Body2_z_vel'.format(form)
    acc_header = '{}_Body2_z_acc'.format(form)
    df_pos[pos_header] = pos_data[1, 2, :]
    df_vel[vel_header] = vel_data[1, 2, :]
    df_acc[acc_header] = acc_data[1, 2, :]

    pos_header = '{}_Body2_y_pos'.format(form)
    vel_header = '{}_Body2_y_vel'.format(form)
    acc_header = '{}_Body2_y_acc'.format(form)
    df_pos[pos_header] = pos_data[1, 1, :]
    df_vel[vel_header] = vel_data[1, 1, :]
    df_acc[acc_header] = acc_data[1, 1, :]

    #plot_many_kinematics(grid, pos_data, vel_data, acc_data, ["test", "test"])


tasks = []

for model_fn in [double_pendulum]:
    num_bodies = 2 if model_fn.__name__ == 'double_pendulum' else 3

    for form in ['rp', 'reps', 'rA']:
        tasks.append((form, model_fn, num_bodies))

# store_grid = True
# for task in tasks:
#     run_model(task, store_grid)
#     store_grid = False

# plot differences in rp vs rA
# position
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1y = sns.lineplot(x='Time', y=df_pos['rA_Body1_y_pos'] - df_pos['rp_Body1_y_pos'], data=df_pos)
# pos_plot_1y.set_title("Difference in position of double pendulum rp and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1z = sns.lineplot(x='Time', y=df_pos['rA_Body1_z_pos'] - df_pos['rp_Body1_z_pos'], data=df_pos)
# pos_plot_1z.set_title("Difference in position of double pendulum rp and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2y = sns.lineplot(x='Time', y=df_pos['rA_Body2_y_pos'] - df_pos['rp_Body2_y_pos'], data=df_pos)
# pos_plot_2y.set_title("Difference in position of double pendulum rp and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2z = sns.lineplot(x='Time', y=df_pos['rA_Body2_z_pos'] - df_pos['rp_Body2_z_pos'], data=df_pos)
# pos_plot_2z.set_title("Difference in position of double pendulum rp and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # velocity
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1y = sns.lineplot(x='Time', y=df_vel['rA_Body1_y_vel'] - df_vel['rp_Body1_y_vel'], data=df_vel)
# vel_plot_1y.set_title("Difference in velocity of double pendulum rp and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1z = sns.lineplot(x='Time', y=df_vel['rA_Body1_z_vel'] - df_vel['rp_Body1_z_vel'], data=df_vel)
# vel_plot_1z.set_title("Difference in velocity of double pendulum rp and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2y = sns.lineplot(x='Time', y=df_vel['rA_Body2_y_vel'] - df_vel['rp_Body2_y_vel'], data=df_vel)
# vel_plot_2y.set_title("Difference in velocity of double pendulum rp and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2z = sns.lineplot(x='Time', y=df_vel['rA_Body2_z_vel'] - df_vel['rp_Body2_z_vel'], data=df_vel)
# vel_plot_2z.set_title("Difference in velocity of double pendulum rp and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # acceleration
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1y = sns.lineplot(x='Time', y=df_acc['rA_Body1_y_acc'] - df_acc['rp_Body1_y_acc'], data=df_acc)
# acc_plot_1y.set_title("Difference in acceleration of double pendulum rp and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1z = sns.lineplot(x='Time', y=df_acc['rA_Body1_z_acc'] - df_acc['rp_Body1_z_acc'], data=df_acc)
# acc_plot_1z.set_title("Difference in acceleration of double pendulum rp and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2y = sns.lineplot(x='Time', y=df_acc['rA_Body2_y_acc'] - df_acc['rp_Body2_y_acc'], data=df_acc)
# acc_plot_2y.set_title("Difference in acceleration of double pendulum rp and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2z = sns.lineplot(x='Time', y=df_acc['rA_Body2_z_acc'] - df_acc['rp_Body2_z_acc'], data=df_acc)
# acc_plot_2z.set_title("Difference in acceleration of double pendulum rp and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # plot differences in reps vs rA
# # position
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1y = sns.lineplot(x='Time', y=df_pos['rA_Body1_y_pos'] - df_pos['reps_Body1_y_pos'], data=df_pos)
# pos_plot_1y.set_title("Difference in position of double pendulum reps and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1z = sns.lineplot(x='Time', y=df_pos['rA_Body1_z_pos'] - df_pos['reps_Body1_z_pos'], data=df_pos)
# pos_plot_1z.set_title("Difference in position of double pendulum reps and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2y = sns.lineplot(x='Time', y=df_pos['rA_Body2_y_pos'] - df_pos['reps_Body2_y_pos'], data=df_pos)
# pos_plot_2y.set_title("Difference in position of double pendulum reps and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2z = sns.lineplot(x='Time', y=df_pos['rA_Body2_z_pos'] - df_pos['reps_Body2_z_pos'], data=df_pos)
# pos_plot_2z.set_title("Difference in position of double pendulum reps and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # velocity
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1y = sns.lineplot(x='Time', y=df_vel['rA_Body1_y_vel'] - df_vel['reps_Body1_y_vel'], data=df_vel)
# vel_plot_1y.set_title("Difference in velocity of double pendulum reps and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1z = sns.lineplot(x='Time', y=df_vel['rA_Body1_z_vel'] - df_vel['reps_Body1_z_vel'], data=df_vel)
# vel_plot_1z.set_title("Difference in velocity of double pendulum reps and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2y = sns.lineplot(x='Time', y=df_vel['rA_Body2_y_vel'] - df_vel['reps_Body2_y_vel'], data=df_vel)
# vel_plot_2y.set_title("Difference in velocity of double pendulum reps and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2z = sns.lineplot(x='Time', y=df_vel['rA_Body2_z_vel'] - df_vel['reps_Body2_z_vel'], data=df_vel)
# vel_plot_2z.set_title("Difference in velocity of double pendulum reps and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # acceleration
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1y = sns.lineplot(x='Time', y=df_acc['rA_Body1_y_acc'] - df_acc['reps_Body1_y_acc'], data=df_acc)
# acc_plot_1y.set_title("Difference in acceleration of double pendulum reps and rA solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1z = sns.lineplot(x='Time', y=df_acc['rA_Body1_z_acc'] - df_acc['reps_Body1_z_acc'], data=df_acc)
# acc_plot_1z.set_title("Difference in acceleration of double pendulum reps and rA solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2y = sns.lineplot(x='Time', y=df_acc['rA_Body2_y_acc'] - df_acc['reps_Body2_y_acc'], data=df_acc)
# acc_plot_2y.set_title("Difference in acceleration of double pendulum reps and rA solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2z = sns.lineplot(x='Time', y=df_acc['rA_Body2_z_acc'] - df_acc['reps_Body2_z_acc'], data=df_acc)
# acc_plot_2z.set_title("Difference in acceleration of double pendulum reps and rA solution of Body 2 in the z-direction")
# plt.show()
#
# # plot differences in reps vs rp
# # position
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1y = sns.lineplot(x='Time', y=df_pos['rp_Body1_y_pos'] - df_pos['reps_Body1_y_pos'], data=df_pos)
# pos_plot_1y.set_title("Difference in position of double pendulum reps and rp solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_1z = sns.lineplot(x='Time', y=df_pos['rp_Body1_z_pos'] - df_pos['reps_Body1_z_pos'], data=df_pos)
# pos_plot_1z.set_title("Difference in position of double pendulum reps and rp solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2y = sns.lineplot(x='Time', y=df_pos['rp_Body2_y_pos'] - df_pos['reps_Body2_y_pos'], data=df_pos)
# pos_plot_2y.set_title("Difference in position of double pendulum reps and rp solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# pos_plot_2z = sns.lineplot(x='Time', y=df_pos['rp_Body2_z_pos'] - df_pos['reps_Body2_z_pos'], data=df_pos)
# pos_plot_2z.set_title("Difference in position of double pendulum reps and rp solution of Body 2 in the z-direction")
# plt.show()
#
# # velocity
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1y = sns.lineplot(x='Time', y=df_vel['rp_Body1_y_vel'] - df_vel['reps_Body1_y_vel'], data=df_vel)
# vel_plot_1y.set_title("Difference in velocity of double pendulum reps and rp solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_1z = sns.lineplot(x='Time', y=df_vel['rp_Body1_z_vel'] - df_vel['reps_Body1_z_vel'], data=df_vel)
# vel_plot_1z.set_title("Difference in velocity of double pendulum reps and rp solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2y = sns.lineplot(x='Time', y=df_vel['rp_Body2_y_vel'] - df_vel['reps_Body2_y_vel'], data=df_vel)
# vel_plot_2y.set_title("Difference in velocity of double pendulum reps and rp solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# vel_plot_2z = sns.lineplot(x='Time', y=df_vel['rp_Body2_z_vel'] - df_vel['reps_Body2_z_vel'], data=df_vel)
# vel_plot_2z.set_title("Difference in velocity of double pendulum reps and rp solution of Body 2 in the z-direction")
# plt.show()
#
# # acceleration
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1y = sns.lineplot(x='Time', y=df_acc['rp_Body1_y_acc'] - df_acc['reps_Body1_y_acc'], data=df_acc)
# acc_plot_1y.set_title("Difference in acceleration of double pendulum reps and rp solution of Body 1 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_1z = sns.lineplot(x='Time', y=df_acc['rp_Body1_z_acc'] - df_acc['reps_Body1_z_acc'], data=df_acc)
# acc_plot_1z.set_title("Difference in acceleration of double pendulum reps and rp solution of Body 1 in the z-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2y = sns.lineplot(x='Time', y=df_acc['rp_Body2_y_acc'] - df_acc['reps_Body2_y_acc'], data=df_acc)
# acc_plot_2y.set_title("Difference in acceleration of double pendulum reps and rp solution of Body 2 in the y-direction")
# plt.show()
#
# sns.set()
# sns.set_style("ticks")
# sns.set_context("paper")
# acc_plot_2z = sns.lineplot(x='Time', y=df_acc['rp_Body2_z_acc'] - df_acc['reps_Body2_z_acc'], data=df_acc)
# acc_plot_2z.set_title("Difference in acceleration of double pendulum reps and rp solution of Body 2 in the z-direction")
# plt.show()

'''
sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_1y = sns.lineplot(x='Time', y=df_pos['rp_Body1_y_pos'], data=df_pos)
pos_plot_1y.set_title("Position of double pendulum, Body 1 in the y-direction")

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_1y = sns.lineplot(x=grid, y=y1)
plt.show()

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_1z = sns.lineplot(x='Time', y=df_pos['rp_Body1_z_pos'], data=df_pos)
pos_plot_1z.set_title("Position of double pendulum, Body 1 in the z-direction")

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_1z = sns.lineplot(x=grid, y=z1)
plt.show()

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_2y = sns.lineplot(x='Time', y=df_pos['rp_Body2_y_pos'], data=df_pos)
pos_plot_2y.set_title("Position of double pendulum, Body 2 in the y-direction")

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_2y = sns.lineplot(x=grid, y=y2)
plt.show()

sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_2z = sns.lineplot(x='Time', y=df_pos['rp_Body2_z_pos'], data=df_pos)
pos_plot_2z.set_title("Position of double pendulum, Body 2 in the z-direction")


sns.set()
sns.set_style("ticks")
sns.set_context("paper")
pos_plot_2z = sns.lineplot(x=grid, y=z2)
plt.show()
'''