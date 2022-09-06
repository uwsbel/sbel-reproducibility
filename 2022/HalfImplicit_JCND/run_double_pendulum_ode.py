# generate ground truth for double pendulum using ode solution

import pickle
import time
from multiprocessing import Pool
import warnings
import os
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from SimEngineMBD.example_models.double_pendulum import run_double_pendulum
from double_pendulum_ode import refSol

dir_path = './ground_truth/'
ground_truth_file_name = "double_pendulum_ode_soln.pickle"
to_xyz = 'xyz'

if not os.path.isdir(dir_path):
    os.makedirs(dir_path)
    
# check if ground truth solution exist
path_to_file = dir_path + ground_truth_file_name
dt_exact = 1e-6  # step size for ode solver
t_end = 8

#run ode solver and pickle solution for future use
pos_exact = refSol(t_end+dt_exact, dt_exact)
with open(path_to_file, 'wb') as handle:
    pickle.dump((pos_exact, dt_exact, t_end), handle, protocol=pickle.HIGHEST_PROTOCOL)
        