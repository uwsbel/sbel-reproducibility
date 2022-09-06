# run fully and half-implicit solvers using various step sizes
# double pendulum problem
# pickle position-level results for convergence analysis

import numpy as np
import pickle
from SimEngineMBD.example_models.double_pendulum import run_double_pendulum



to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

num_bodies = 2

legends = []


model_fn = run_double_pendulum
# same step size
# check slides/tech report for tolerances
tolerance = 1e-10
step_sizes = [1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3, 1e-2, 2e-2, 4e-2]

dir_path = 'ground_truth/'


t_end = 8




for form in ['rA', 'rA_half']:
    diff_list = []

    for ii, step_size in enumerate(step_sizes):

        t = np.arange(step_size, t_end+step_size, step_size)


        print('===============')
        print('Run {}, step_size {}, tolerance {}'.format(form, step_size, tolerance))    
        print('===============')
        
        if form == 'rA':
        
            pos, _, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance/step_size**2), '--step_size', str(step_size), '-t', str(t_end)])

        else:
            pos, _, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
        
        
        pickle_name = '{}_{}_dynamics_dt_{}.pickle'.format(model_fn.__name__[4:], form, step_size)
        with open(dir_path + pickle_name, 'wb') as handle:
            pickle.dump((t, pos, numItr), handle, protocol=pickle.HIGHEST_PROTOCOL)
