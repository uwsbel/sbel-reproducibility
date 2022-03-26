#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mechanisms.double_pendulum import double_pendulum

# store results in dataframe for plotting later
df_pos_coarse = pd.DataFrame()
df_vel_coarse = pd.DataFrame()
df_acc_coarse = pd.DataFrame()

df_pos_med = pd.DataFrame()
df_vel_med = pd.DataFrame()
df_acc_med = pd.DataFrame()

df_pos_fine = pd.DataFrame()
df_vel_fine = pd.DataFrame()
df_acc_fine = pd.DataFrame()

def run_coarse(args, step_size, store_grid=True):
    form, model_fn, num_bodies = args
    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])

    pos_data, vel_data, acc_data, _, grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(1e-11/step_size**2),
                                             '--step_size', str(step_size), '-t', '5'])

    if store_grid:
        df_pos_coarse['Time'] = grid
        df_vel_coarse['Time'] = grid
        df_acc_coarse['Time'] = grid

    pos_header = '{}_Body1_z_pos'.format(form)
    vel_header = '{}_Body1_z_vel'.format(form)
    acc_header = '{}_Body1_z_acc'.format(form)
    df_pos_coarse[pos_header] = pos_data[0, 2, :]
    df_vel_coarse[vel_header] = vel_data[0, 2, :]
    df_acc_coarse[acc_header] = acc_data[0, 2, :]

    pos_header = '{}_Body1_y_pos'.format(form)
    vel_header = '{}_Body1_y_vel'.format(form)
    acc_header = '{}_Body1_y_acc'.format(form)
    df_pos_coarse[pos_header] = pos_data[0, 1, :]
    df_vel_coarse[vel_header] = vel_data[0, 1, :]
    df_acc_coarse[acc_header] = acc_data[0, 1, :]

    pos_header = '{}_Body2_z_pos'.format(form)
    vel_header = '{}_Body2_z_vel'.format(form)
    acc_header = '{}_Body2_z_acc'.format(form)
    df_pos_coarse[pos_header] = pos_data[1, 2, :]
    df_vel_coarse[vel_header] = vel_data[1, 2, :]
    df_acc_coarse[acc_header] = acc_data[1, 2, :]

    pos_header = '{}_Body2_y_pos'.format(form)
    vel_header = '{}_Body2_y_vel'.format(form)
    acc_header = '{}_Body2_y_acc'.format(form)
    df_pos_coarse[pos_header] = pos_data[1, 1, :]
    df_vel_coarse[vel_header] = vel_data[1, 1, :]
    df_acc_coarse[acc_header] = acc_data[1, 1, :]


def run_med(args, step_size, store_grid=True):
    form, model_fn, num_bodies = args
    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])

    pos_data, vel_data, acc_data, _, grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(1e-11/step_size**2),
                                             '--step_size', str(step_size), '-t', '5'])

    if store_grid:
        df_pos_med['Time'] = grid
        df_vel_med['Time'] = grid
        df_acc_med['Time'] = grid

    pos_header = '{}_Body1_z_pos'.format(form)
    vel_header = '{}_Body1_z_vel'.format(form)
    acc_header = '{}_Body1_z_acc'.format(form)
    df_pos_med[pos_header] = pos_data[0, 2, :]
    df_vel_med[vel_header] = vel_data[0, 2, :]
    df_acc_med[acc_header] = acc_data[0, 2, :]

    pos_header = '{}_Body1_y_pos'.format(form)
    vel_header = '{}_Body1_y_vel'.format(form)
    acc_header = '{}_Body1_y_acc'.format(form)
    df_pos_med[pos_header] = pos_data[0, 1, :]
    df_vel_med[vel_header] = vel_data[0, 1, :]
    df_acc_med[acc_header] = acc_data[0, 1, :]

    pos_header = '{}_Body2_z_pos'.format(form)
    vel_header = '{}_Body2_z_vel'.format(form)
    acc_header = '{}_Body2_z_acc'.format(form)
    df_pos_med[pos_header] = pos_data[1, 2, :]
    df_vel_med[vel_header] = vel_data[1, 2, :]
    df_acc_med[acc_header] = acc_data[1, 2, :]

    pos_header = '{}_Body2_y_pos'.format(form)
    vel_header = '{}_Body2_y_vel'.format(form)
    acc_header = '{}_Body2_y_acc'.format(form)
    df_pos_med[pos_header] = pos_data[1, 1, :]
    df_vel_med[vel_header] = vel_data[1, 1, :]
    df_acc_med[acc_header] = acc_data[1, 1, :]


def run_fine(args, step_size, store_grid=True):
    form, model_fn, num_bodies = args
    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])

    pos_data, vel_data, acc_data, _, grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(1e-11/step_size**2),
                                             '--step_size', str(step_size), '-t', '5'])

    if store_grid:
        df_pos_fine['Time'] = grid
        df_vel_fine['Time'] = grid
        df_acc_fine['Time'] = grid

    pos_header = '{}_Body1_z_pos'.format(form)
    vel_header = '{}_Body1_z_vel'.format(form)
    acc_header = '{}_Body1_z_acc'.format(form)
    df_pos_fine[pos_header] = pos_data[0, 2, :]
    df_vel_fine[vel_header] = vel_data[0, 2, :]
    df_acc_fine[acc_header] = acc_data[0, 2, :]

    pos_header = '{}_Body1_y_pos'.format(form)
    vel_header = '{}_Body1_y_vel'.format(form)
    acc_header = '{}_Body1_y_acc'.format(form)
    df_pos_fine[pos_header] = pos_data[0, 1, :]
    df_vel_fine[vel_header] = vel_data[0, 1, :]
    df_acc_fine[acc_header] = acc_data[0, 1, :]

    pos_header = '{}_Body2_z_pos'.format(form)
    vel_header = '{}_Body2_z_vel'.format(form)
    acc_header = '{}_Body2_z_acc'.format(form)
    df_pos_fine[pos_header] = pos_data[1, 2, :]
    df_vel_fine[vel_header] = vel_data[1, 2, :]
    df_acc_fine[acc_header] = acc_data[1, 2, :]

    pos_header = '{}_Body2_y_pos'.format(form)
    vel_header = '{}_Body2_y_vel'.format(form)
    acc_header = '{}_Body2_y_acc'.format(form)
    df_pos_fine[pos_header] = pos_data[1, 1, :]
    df_vel_fine[vel_header] = vel_data[1, 1, :]
    df_acc_fine[acc_header] = acc_data[1, 1, :]


tasks_coarse = []
tasks_med = []
tasks_fine = []

for model_fn in [double_pendulum]:
    num_bodies = 2 if model_fn.__name__ == 'double_pendulum' else 3

    for form in ['reps', 'rp', 'rA']:
        tasks_coarse.append((form, model_fn, num_bodies))
        tasks_med.append((form, model_fn, num_bodies))
        tasks_fine.append((form, model_fn, num_bodies))

store_grid = True
for task in tasks_coarse:
    run_coarse(task, 1e-2, store_grid)
    store_grid = False
print("coarse done.")
df_pos_coarse.to_pickle("./position_coarse.pkl")
df_vel_coarse.to_pickle("./velocity_coarse.pkl")
df_acc_coarse.to_pickle("./acceleration_coarse.pkl")

store_grid = True
for task in tasks_med:
    run_med(task, 1e-3, store_grid)
    store_grid = False
print("med done.")
df_pos_med.to_pickle("./position_med.pkl")
df_vel_med.to_pickle("./velocity_med.pkl")
df_acc_med.to_pickle("./acceleration_med.pkl")

store_grid = True
for task in tasks_fine:
    run_fine(task, 1e-4, store_grid)
    store_grid = False
df_pos_fine.to_pickle("./position_fine.pkl")
df_vel_fine.to_pickle("./velocity_fine.pkl")
