import itertools
from multiprocessing import Pool
import pickle

import numpy as np

from SimEngineMBD.example_models.single_pendulum import run_single_pendulum
from SimEngineMBD.example_models.four_link import run_four_link
from SimEngineMBD.example_models.slider_crank import run_slider_crank

# For 'production'
# step_sizes = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1e-1])
# M_vals = np.array([1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13])

# # For testing
step_sizes = np.array([2e-2, 4e-2, 8e-2])
M_vals = np.array([1e-8, 1e-9])

dir_path = './output/surf/'

to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε'}

ss, MM = np.meshgrid(step_sizes, M_vals)
with open(dir_path + 'mesh_params.pickle', 'wb') as handle:
    pickle.dump((ss, MM), handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_model(args):
    form, model_fn, num_bodies = args

    pos_exact, vel_exact, acc_exact, _, _ = model_fn(['--form', form, '--mode', 'kinematics', '--tol', '1e-12'])

    # run_some_model_name -> Some_Model_Name
    pretty_name = '_'.join([word.capitalize() for word in model_fn.__name__[4:].split('_')])

    # We leave the NaNs in if it fails to converge and they don't get plotted
    pos_diff = np.full((len(M_vals), len(step_sizes), num_bodies, 3), np.nan)
    vel_diff = np.full((len(M_vals), len(step_sizes), num_bodies, 3), np.nan)
    acc_diff = np.full((len(M_vals), len(step_sizes), num_bodies, 3), np.nan)
    conv_iters = np.full((len(M_vals), len(step_sizes)), np.nan)

    for i, M in enumerate(M_vals):
        for j, step in enumerate(step_sizes):
            try:
                tol = M / step**2

                pos, vel, acc, iters, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tol), '--step_size', str(step)])

                pos_diff[i, j, :, :] = np.abs(pos_exact[:, :, -1] - pos[:, :, -1])
                vel_diff[i, j, :, :] = np.abs(vel_exact[:, :, -1] - vel[:, :, -1])
                acc_diff[i, j, :, :] = np.abs(acc_exact[:, :, -1] - acc[:, :, -1])

                conv_iters[i, j] = 1 / np.mean(iters)
            except RuntimeError:
                print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(step), str(tol)))
            except ValueError:
                print('{}-{}, step: {}, tol: {} raised value error'.format(form, pretty_name, str(step), str(tol)))
                raise

    with open(dir_path + '{}_{}_iterations.pickle'.format(pretty_name, form), 'wb') as handle:
        pickle.dump(((pretty_name, pretty_form[form]), conv_iters), handle, protocol=pickle.HIGHEST_PROTOCOL)

    for body in range(0, num_bodies):
        for component in range(0, 3):
            save_name = '{}_{}_OA_Body_{}_{}.pickle'.format(pretty_name, form, body, to_xyz[component])
            info = (pretty_name, pretty_form[form], body, to_xyz[component])

            with open(dir_path + save_name, 'wb') as handle:
                pickle.dump((info, pos_diff[:, :, body, component], vel_diff[:, :, body, component], acc_diff[:, :, body, component]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Completed {} {} Analysis'.format(pretty_name, pretty_form[form]))

tasks = []

for model_fn in [run_single_pendulum, run_four_link, run_slider_crank]:
    num_bodies = 1 if model_fn.__name__ == 'run_single_pendulum' else 3
    
    for form in ['rA']:
        tasks.append((form, model_fn, num_bodies))

# for task in tasks:
#     run_model(task)

pool = Pool()
pool.map(run_model, tasks)
pool.close()
