# compute timing and average number of iterations 
# for double pendulum, slider crank, and four link problems
# step sizes are 1e-4, 1e-3 and 1e-2
# tolerance is handpicked to ensure same quality of solution

import matplotlib.pyplot as plt
import numpy as np
import pickle
from SimEngineMBD.example_models.slider_crank import run_slider_crank
from SimEngineMBD.example_models.slider_crank import time_slider_crank
from SimEngineMBD.example_models.four_link import run_four_link
from SimEngineMBD.example_models.four_link import time_four_link
from SimEngineMBD.example_models.double_pendulum import run_double_pendulum
from SimEngineMBD.example_models.double_pendulum import time_double_pendulum
from matplotlib import ticker



forms = ['rA_half', 'rA']
model_fn = time_slider_crank
# same step size
# check slides/tech report for tolerances
tolerances = [1e-9, 1e-3, 1e-10, 1e-7, 1e-10, 1e-7]
step_sizes = [1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2]

t_end = 8

cpu_time_list = []


for ii, step_size in enumerate(step_sizes):
    tolerance = tolerances[ii]
    
    if ii%2 == 0:
        form = 'rA_half'
    else:
        form = 'rA'

    # _, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
    cpu_time= model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
        
    cpu_time_list.append(cpu_time)
    print("{} step size {} tolerance {} cpu time {}".format(form, step_size, tolerance, cpu_time))


forms = ['rA_half', 'rA']
model_fn = time_four_link
# same step size
# check slides/tech report for tolerances
tolerances = [1e-7, 1e-2, 1e-9, 1e-5, 1e-10, 1e-7]
step_sizes = [1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2]



for ii, step_size in enumerate(step_sizes):
    tolerance = tolerances[ii]
    
    if ii%2 == 0:
        form = 'rA_half'
    else:
        form = 'rA'

    # _, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
    cpu_time = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
    cpu_time_list.append(cpu_time)

        
    print("{} step size {} tolerance {} cpu time {}".format(form, step_size, tolerance, cpu_time))


model_fn = time_double_pendulum
# same step size
# check slides/tech report for tolerances
tolerances = [1e-7, 1e-5, 1e-7, 1e-6, 1e-7, 1e-7]
step_sizes = [1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2]


for ii, step_size in enumerate(step_sizes):
    tolerance = tolerances[ii]
    
    if ii%2 == 0:
        form = 'rA_half'
    else:
        form = 'rA'

    # _, _, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
    cpu_time = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
    cpu_time_list.append(cpu_time)
        
        
    print("{} step size {} tolerance {} cpu time {}".format(form, step_size, tolerance, cpu_time))
    
cpu_time_array = np.array(cpu_time_list)