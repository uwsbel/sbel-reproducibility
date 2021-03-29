import numpy as np
import pickle

import itertools
from multiprocessing import Pool
from four_link_oa import four_link
from slider_crank_oa import slider_crank
from single_pendulum_oa import single_pendulum

step_sizes = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1e-1])
M_vals = np.array([1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13])

# step_sizes = np.array([2e-2, 4e-2, 8e-2])
# M_vals = np.array([1e-8, 1e-9])

to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε'}

ss, MM = np.meshgrid(step_sizes, M_vals)
with open('mesh_params.pickle', 'wb') as handle:
    pickle.dump((ss, MM), handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_model(args):
    form, model_fn, num_bodies = args

    pos_exact, vel_exact, acc_exact, _, _ = model_fn(['--form', form, '--mode', 'kinematics', '--tol', '1e-12'])

    pretty_name = '_'.join([word.capitalize() for word in model_fn.__name__.split('_')])

    # Start empty, we won't add the ones that fail to converge
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

    with open('{}_{}_iterations.pickle'.format(pretty_name, form), 'wb') as handle:
        pickle.dump(((pretty_name, pretty_form[form]), conv_iters), handle, protocol=pickle.HIGHEST_PROTOCOL)

    for body in range(0, num_bodies):
        for component in range(0, 3):
            save_name = '{}_{}_OA_Body_{}_{}'.format(pretty_name, form, body, to_xyz[component])
            info = (pretty_name, pretty_form[form], body, to_xyz[component])

            with open(save_name + '.pickle', 'wb') as handle:
                pickle.dump((info, pos_diff[:, :, body, component], vel_diff[:, :, body, component], acc_diff[:, :, body, component]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Completed {} {} Analysis'.format(pretty_name, pretty_form[form]))

tasks = []

for model_fn in [single_pendulum, four_link, slider_crank]:
    num_bodies = 1 if model_fn.__name__ == 'single_pendulum' else 3
    
    for form in ['rA']:
        tasks.append((form, model_fn, num_bodies))
        # run_model((form, model_fn, num_bodies))

pool = Pool()
pool.map(run_model, tasks)
pool.close()