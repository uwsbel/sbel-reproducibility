import sys
import pathlib as pl
src_folder = pl.Path('tests')
sys.path.append(str(src_folder))

import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

from four_link import four_link
from slider_crank import slider_crank
from single_pendulum import single_pendulum

step_sizes = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2])
# testing
# step_sizes = np.array([1e-2, 2e-2, 1e-2, 2e-2, 1e-2, 2e-2])
# step_sizes = np.array([1e-2, 2e-2])

pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε'}
# We leave the NaNs in if it fails to converge and they don't get plotted
conv_iters_kinematics = np.full([3, len(step_sizes)], np.nan)
conv_iters_dynamics = np.full([3, len(step_sizes)], np.nan)

def run_model(args):
    form, model_fn, num_bodies = args
    if form == 'rA':
        form_index = 0
    if form == 'rp':
        form_index = 1
    if form == 'reps':
        form_index = 2

    pretty_name = '_'.join([word.capitalize() for word in model_fn.__name__.split('_')])

    for j, step in enumerate(step_sizes):
        try:
            _, _, _, iters = model_fn(['--form', form, '--mode', 'kinematics', '--tol', '1e-10', '--step_size', str(step)])

            conv_iters_kinematics[form_index, j] = iters
        except RuntimeError:
            print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(step), '1e-10'))
        except ValueError:
            print('{}-{}, step: {}, tol: {} raised value error'.format(form, pretty_name, str(step), '1e-10'))
            raise

    for j, step in enumerate(step_sizes):
        try:
            tol = 1e-11 / step ** 2
            # tol = 1e-3

            _, _, _, iters = model_fn(
                ['--form', form, '--mode', 'dynamics', '--tol', str(tol), '--step_size', str(step)])

            conv_iters_dynamics[form_index, j] = iters
        except RuntimeError:
            print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(step), str(tol)))
        except ValueError:
            print('{}-{}, step: {}, tol: {} raised value error'.format(form, pretty_name, str(step), str(tol)))
            raise

    print('Completed {} {} Analysis'.format(pretty_name, pretty_form[form]))

tasks = []
for model_fn in [single_pendulum, four_link, slider_crank]:
    num_bodies = 1 if model_fn.__name__ == 'single_pendulum' else 3

    for form in ['rA', 'rp', 'reps']:
        tasks.append((form, model_fn, num_bodies))

for task in tasks:
    run_model(task)
    # plotting kinematics
    title = '{} Kinematics: Iterations to Convergence'.format(task[1].__name__)
    ind = np.arange(len(step_sizes))
    width = 0.15
    fig = plt.subplots(figsize=(9, 6))
    plt.bar(ind - width, conv_iters_kinematics[0], width, label='rA')
    plt.bar(ind, conv_iters_kinematics[1], width, label='rp')
    plt.bar(ind + width, conv_iters_kinematics[2], width, label='rε')
    plt.title(title, fontsize=16)
    plt.ylabel('Average Iterations to Convergence', fontsize=12)
    plt.xlabel('Step Size', fontsize=12)
    plt.xticks(ind, (
        r"$1\times 10^{-3}$", r"$2\times 10^{-3}$", r"$4\times 10^{-3}$", r"$8\times 10^{-3}$", r"$1\times 10^{-2}$",
        r"$2\times 10^{-2}$"))
    plt.legend(loc='best', prop={'size': 12})
    plt.gcf().set_size_inches(20, 12)
    plt.savefig('./output/convergence/{}_Convergence.png'.format(task[1].__name__))

    # plotting dynamics
    title = '{} Dynamics: Iterations to Convergence'.format(task[1].__name__)
    fig = plt.subplots(figsize=(9, 6))
    plt.bar(ind - width, conv_iters_dynamics[0], width, label='rA')
    plt.bar(ind, conv_iters_dynamics[1], width, label='rp')
    plt.bar(ind + width, conv_iters_dynamics[2], width, label='rε')
    plt.title(title, fontsize=16)
    plt.ylabel('Average Iterations to Convergence', fontsize=12)
    plt.xlabel('Step Size', fontsize=12)
    plt.xticks(ind, (
        r"$1\times 10^{-3}$", r"$2\times 10^{-3}$", r"$4\times 10^{-3}$", r"$8\times 10^{-3}$", r"$1\times 10^{-2}$",
        r"$2\times 10^{-2}$"))
    plt.legend(loc='best', prop={'size': 12})
    plt.gcf().set_size_inches(20, 12)
    plt.savefig('./output/convergence/{}_Convergence.png'.format(task[1].__name__))

