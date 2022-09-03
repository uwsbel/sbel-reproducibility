# accuracy tests
# step size 1e-3
# look at how step sizes influece half implicit and fully implicit 

import matplotlib.pyplot as plt
import numpy as np

from SimEngineMBD.example_models.N_pendulum import time_N_pendulum
# from SimEngineMBD.example_models.N_pendulum import run_N_pendulum

# from matplotlib import ticker


to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

t_end = 3

# step_sizes = [5e-5, 1e-4, 5e-4, 1e-3]
forms = ['rA_half', 'rA']
model_fn = time_N_pendulum
# model_fn = run_N_pendulum

step_size = 1e-3
tolerances = [1e-7, 1e-5]

# tolerances = [1e-6, 1e-5]

bodies = [2, 4, 6, 8, 16, 32]

for num_bodies in bodies:
    
    t = np.arange(step_size, t_end+step_size, step_size)
    time_half = model_fn(num_bodies, ['--form', forms[0], '--mode', 'dynamics', '--tol', str(tolerances[0]), '--step_size', str(step_size), '-t', str(t_end)])
    time_full = model_fn(num_bodies, ['--form', forms[1], '--mode', 'dynamics', '--tol', str(tolerances[1]), '--step_size', str(step_size), '-t', str(t_end)])

    print("{}, {}, {}".format(num_bodies, time_half, time_full))

    