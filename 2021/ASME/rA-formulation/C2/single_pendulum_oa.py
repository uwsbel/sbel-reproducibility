import logging
import argparse as arg
from collections import defaultdict
from time import perf_counter, process_time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from system_ra import SystemRA
from system_rp import SystemRP
from system_reps import SystemREps
from physics import Z_AXIS
from tools import profiler, plot_many_kinematics, print_profiling, standard_setup

def single_pendulum(args):
    π = np.pi

    # Set up command-line options
    parser = arg.ArgumentParser(description='Simulation of a single link pendulum')
    parser.add_argument('-t', '--end_time', type=float, default=3, dest='t_end')

    parser.add_argument('--tracked_body', type=int, choices=[0], default=0)
    parser.add_argument('--tracked_component', type=int, choices=[0, 1, 2], default=2)

    model_files = defaultdict(lambda: 'models/single_pendulum.mdl')

    # Call utility function to setup our system and set the running mode
    sys, params = standard_setup(parser, model_files, args)

    # Get system and change some settings
    sys.set_g_acc(-9.81 * Z_AXIS)
    sys.h = params.h
    sys.tol = params.tol
    sys.solver_order = 1

    # Physical constants
    L = 2                                   # [m] - length of the bar
    w = 0.05                                # [m] - side length of bar
    ρ = 7800                                # [kg/m^3] - density of the bar

    b_len = [2*L]
    for j, body in enumerate(sys.bodies):
        body.V = b_len[j] * w**2                    # [m^3] - bar volume
        body.m = ρ * body.V                         # [kg] - bar mass

        J_xx = 1/6 * body.m * w**2
        J_yz = 1/12 * body.m * (w**2 + b_len[j]**2)
        body.J = np.diag([J_xx, J_yz, J_yz])        # [??] - Inertia tensor of bar

    # Functions for driving constraint
    t = sp.symbols('t')
    ang_sym = π/2 + π/4 * sp.cos(2*t)

    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    pos_data = np.zeros(t_steps)
    vel_data = np.zeros(t_steps)
    acc_data = np.zeros(t_steps)

    for i, t in enumerate(t_grid):
        sys.do_step(i, t)

        pos_data = sys.bodies[params.tracked_body].r[params.tracked_component].T
        vel_data = sys.bodies[params.tracked_body].dr[params.tracked_component].T
        acc_data = sys.bodies[params.tracked_body].ddr[params.tracked_component].T

    return pos_data, vel_data, acc_data


