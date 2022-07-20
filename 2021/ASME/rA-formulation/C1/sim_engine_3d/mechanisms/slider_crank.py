#!/usr/bin/env python3

""" Run simulation of the slider crank mechanism.

This module runs solvers using the slider crank model and solver parameters specified by tools.standard_setup().

Functions:

    slider_crank(args)

"""

import numpy as np
from copy import copy
import argparse as arg

from rA.rA_sim_engine_3d import rASimEngine3D
from rp.rp_sim_engine_3d import rpSimEngine3D
from reps.reps_sim_engine_3d import repsSimEngine3D
import reps.reps_gcons as gcons
from utils.tools import standard_setup

def slider_crank(args):
    parser = arg.ArgumentParser(description='Simulation of Haug\'s four-link mechanism')

    model_files = "./sim_engine_3d/models/slider_crank_rotated.mdl"

    sys, params = standard_setup(parser, model_files, args)
    sys.h = params.h
    sys.tol = params.tol
    sys.t_start = 0
    sys.t_end = params.t_end

    # body 1 properties
    sys.bodies_full[0].m = 0.12
    sys.bodies_full[0].J = np.diag([0.0001, 0.00001, 0.0001])

    # body 2 properties
    sys.bodies_full[1].m = 0.5
    sys.bodies_full[1].J = np.diag([0.004, 0.0004, 0.004])

    # body 3 properties
    sys.bodies_full[2].m = 2
    sys.bodies_full[2].J = np.diag([0.0001, 0.0001, 0.0001])

    # Alternative driving constraint for singularity encounter
    sys.alternative_driver = copy(sys.constraint_list[-1])
    sys.alternative_driver.a_bar_i = np.array([[0], [0], [1]])
    sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-2*pi*t)",
                                                                    "2*pi*cos(-2*pi*t-pi/2)",
                                                                    "-4*pi**2*sin(-2*pi*t-pi/2)")
    if args[3] == 'dynamics':
        sys.dynamics_solver()
    else:
        sys.kinematics_solver()

    return sys.r_sol, sys.r_dot_sol, sys.r_ddot_sol, sys.avg_iterations, sys.t_grid