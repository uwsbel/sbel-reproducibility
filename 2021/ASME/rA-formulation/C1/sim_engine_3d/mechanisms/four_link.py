#!/usr/bin/env python3

""" Run simulation of the four link mechanism.

This module runs solvers using the four link model and solver parameters specified by tools.standard_setup().

Functions:

    four_link(args)

"""

import numpy as np
from copy import copy
import argparse as arg

from rA.rA_sim_engine_3d import rASimEngine3D
from rp.rp_sim_engine_3d import rpSimEngine3D
from reps.reps_sim_engine_3d import repsSimEngine3D
import reps.reps_gcons as gcons
from utils.tools import standard_setup

def four_link(args):
    parser = arg.ArgumentParser(description='Simulation of Haug\'s four-link mechanism')

    model_files = "./models/four_link_rotated.mdl"

    sys, params = standard_setup(parser, model_files, args)
    sys.h = params.h
    sys.tol = params.tol
    sys.t_start = 0
    sys.t_end = params.t_end

    # body 1 properties
    sys.bodies_full[1].m = 2.0
    sys.bodies_full[1].J = np.diag([4, 2, 0])

    # body 2 properties
    sys.bodies_full[2].m = 1
    sys.bodies_full[2].J = np.diag([12.4, 0.01, 0])

    # body 3 properties
    sys.bodies_full[3].m = 1
    sys.bodies_full[3].J = np.diag([4.54, 0.01, 0])

    # Alternative driving constraint for singularity encounter
    sys.alternative_driver = copy(sys.constraint_list[-1])
    sys.alternative_driver.a_bar_j = np.array([[0], [0], [1]])
    sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-pi * t - pi/2 + pi/2)",
                                                                "-pi*sin(pi*t)",
                                                                "-pi**2*cos(pi*t)")
    if args[3] == 'dynamics':
        sys.dynamics_solver()
    else:
        sys.kinematics_solver()

    iterations = sys.avg_iterations
    pos = np.zeros((sys.nb, 3, sys.N))
    vel = np.zeros((sys.nb, 3, sys.N))
    acc = np.zeros((sys.nb, 3, sys.N))
    
    for t in range(sys.N):
        for body in sys.bodies_list:
            if body.is_ground:
                pass
            else:
                pos[(body.body_id - 1), :, t] = sys.r_sol[t, (body.body_id - 1) * 3:((body.body_id - 1) * 3) + 3]
                vel[(body.body_id - 1), :, t] = sys.r_dot_sol[t, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3].T
                acc[(body.body_id - 1), :, t] = sys.r_ddot_sol[t, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3].T

    return pos, vel, acc, iterations
