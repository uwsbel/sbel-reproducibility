#!/usr/bin/env python3

""" Run simulation of the single pendulum mechanism.

This module runs solvers using the single pendulum model and solver parameters specified by tools.standard_setup().

Functions:

    single_pendulum(args)

"""

import numpy as np
import argparse as arg

from rA.rA_sim_engine_3d import rASimEngine3D
from rp.rp_sim_engine_3d import rpSimEngine3D
from reps.reps_sim_engine_3d import repsSimEngine3D
from utils.tools import standard_setup

def single_pendulum(args):
    parser = arg.ArgumentParser(description='Simulation of a single pendulum mechanism')

    model_files = "./sim_engine_3d/models/revJoint.mdl"

    sys, params = standard_setup(parser, model_files, args)
    sys.h = params.h
    sys.tol = params.tol
    sys.t_start = 0
    sys.t_end = params.t_end

    L = 2
    w = 0.05
    rho = 7800
    b_len = [2 * L]

    V = b_len[0] * w ** 2
    sys.bodies_full[1].m = rho * V
    J_xx = 1 / 6 * sys.bodies_full[1].m * w ** 2
    J_yz = 1 / 12 * sys.bodies_full[1].m * (w ** 2 + b_len[0] ** 2)
    sys.bodies_full[1].J = np.diag([J_xx, J_yz, J_yz])

    if args[3] == 'dynamics':
        sys.dynamics_solver()
    else:
        sys.kinematics_solver()

    return sys.r_sol, sys.r_dot_sol, sys.r_ddot_sol, sys.avg_iterations, sys.t_grid
