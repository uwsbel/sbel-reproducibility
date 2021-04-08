import logging
import argparse as arg
from collections import defaultdict
from time import process_time
from copy import copy

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from system_ra import SystemRA
from system_rp import SystemRP
from system_reps import SystemREps
from physics import R, Y_AXIS, Z_AXIS
from tools import profiler, plot_many_kinematics, plot_kinematics_analysis, print_profiling, standard_setup

def slider_crank(args):

    π = np.pi

    # Set up command-line options
    parser = arg.ArgumentParser(description='Simulation of a Haug\'s slider-crank model')
    parser.add_argument('-t', '--end_time', type=float, default=3, dest='t_end')

    model_files = defaultdict(lambda: 'models/slider_crank.json')

    # Get system and change some settings
    sys, params = standard_setup(parser, model_files, args)
    sys.set_g_acc(-9.81 * Z_AXIS)
    sys.h = params.h
    sys.tol = params.tol
    sys.solver_order = 2

    # See Haug p. 456 for properties
    # Crank
    sys.bodies[0].m = 0.12
    sys.bodies[0].J = np.diag([1e-4, 1e-5, 1e-4])

    # Connecting Rod
    sys.bodies[1].m = 0.5
    sys.bodies[1].J = np.diag([4e-3, 4e-4, 4e-3])

    # Slider
    sys.bodies[2].m = 2
    sys.bodies[2].J = np.diag([1e-4, 1e-4, 1e-4])

    # Functions for initial driving constraint
    t = sp.symbols('t')
    ang_sym = -2*π*t + π/2
    ang_alt = ang_sym - π/2

    con_num = len(sys.g_cons.cons) - 1

    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)
    sys.g_cons.cons[-2].set_constraint_fn(1, t)

    alt_dp1 = copy(sys.g_cons.cons[-1])
    alt_dp1.set_constraint_fn(sp.cos(ang_alt), t)
    alt_dp1.ai = Z_AXIS

    sys.initialize()

    t_grid = np.arange(0, params.t_end, params.h)

    start = process_time()
    for i, t in enumerate(t_grid):
        # (Hack) swap g-cons to avoid driving constraint singularity
        if np.abs(np.abs(sys.g_cons.cons[con_num].f(t)) - 1) < 0.1:
            logging.info('Swapped g-con at time {}'.format(t))
            sys.g_cons.cons[con_num], alt_dp1 = alt_dp1, sys.g_cons.cons[con_num]
            logging.debug(sys.g_cons.cons[con_num].f(t))
            logging.debug(alt_dp1.f(t))

        sys.do_step(i, t)
    Δt = process_time() - start

    return Δt


