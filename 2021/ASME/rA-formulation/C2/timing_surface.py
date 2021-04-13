import pickle
from multiprocessing import Pool
import os

import numpy as np
import matplotlib.pyplot as plt

from SimEngineMBD.example_models.single_pendulum import time_single_pendulum
from SimEngineMBD.example_models.four_link import time_four_link
from SimEngineMBD.example_models.slider_crank import time_slider_crank

dir_path = './output/timing/'

def save_data():

    # For 'production'
    # step_sizes = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1e-1])
    # M_vals = np.array([1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13])

    # # For testing
    step_sizes = np.array([2e-2, 4e-2, 8e-2])
    M_vals = np.array([1e-8, 1e-9])

    end_time = 3
    timing_runs = 5

    pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε'}

    ss, MM = np.meshgrid(step_sizes, M_vals)
    with open(dir_path + 'mesh_params.pickle', 'wb') as handle:
        pickle.dump((ss, MM), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_model(args):
        form, model_fn = args

        # time_some_model_name -> Some_Model_Name
        pretty_name = '_'.join([word.capitalize() for word in model_fn.__name__[5:].split('_')])

        timing = np.full((len(M_vals), len(step_sizes)), np.nan)

        for i, M in enumerate(M_vals):
            for j, step in enumerate(step_sizes):
                try:
                    times = np.zeros(timing_runs)
                    tol = M / step**2

                    for k in range(0, timing_runs):
                        Δt = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tol), '--step_size', str(step), '--end_time', str(end_time)])
                        times[k] = Δt

                    timing[i, j] = np.mean(times) / end_time

                except RuntimeError:
                    print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(step), str(tol)))

        save_name = '{}_{}_timing.pickle'.format(pretty_name, form)
        info = (pretty_name, pretty_form[form])

        with open(dir_path + save_name, 'wb') as handle:
            pickle.dump((info, timing), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Completed {} {} Analysis'.format(pretty_name, pretty_form[form]))

    tasks = []

    for model_fn in [time_single_pendulum, time_four_link, time_slider_crank]:
        for form in ['rp', 'reps', 'rA']:
            tasks.append((form, model_fn))

    # for task in tasks:
    #     run_model(task)

    pool = Pool()
    pool.map(run_model, tasks)
    pool.close()

def generate_plots():

    files = []
    for f in os.listdir(dir_path):
        if f == 'mesh_params.pickle':
            with open(dir_path + f, 'rb') as handle:
                ss, MM = pickle.load(handle)
            
            continue

        if f.endswith('.pickle') and f.startswith('Four_Link'):
            files.append(f)

    for file_name in files:
        with open(dir_path + file_name, 'rb') as handle:
            info, timing = pickle.load(handle)

        title = '{} {} Timing Analysis'.format(*info)

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.gca(projection='3d')
        _ = ax.plot_surface(np.log10(ss), np.log10(MM), timing)
        ax.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

    plt.show()