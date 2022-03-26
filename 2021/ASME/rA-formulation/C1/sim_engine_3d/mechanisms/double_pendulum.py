#!/usr/bin/env python3

""" Run simulation of the double pendulum mechanism.

This module runs solvers using the double pendulum model and solver parameters specified by tools.standard_setup().

Functions:

    double_pendulum(args)

"""

import numpy as np
import argparse as arg

from rA.rA_sim_engine_3d import rASimEngine3D
from rp.rp_sim_engine_3d import rpSimEngine3D
from reps.reps_sim_engine_3d import repsSimEngine3D
from utils.tools import standard_setup

def double_pendulum(args):
    
    parser = arg.ArgumentParser(description='Simulation of a double pendulum mechanism')

    model_files = "./models/double_pendulum.mdl"

    sys, params = standard_setup(parser, model_files, args)
    sys.h = params.h
    sys.tol = params.tol
    sys.t_start = 0
    sys.t_end = params.t_end

    L = 2
    w = 0.05
    rho = 7800
    b_len = [2 * L, L]
    for j, body in enumerate(sys.bodies_list):
        V = b_len[j] * w ** 2
        body.m = rho * V
        J_xx = 1 / 6 * body.m * w ** 2
        J_yz = 1 / 12 * body.m * (w ** 2 + b_len[j] ** 2)
        body.J = np.diag([J_xx, J_yz, J_yz])

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

    return pos, vel, acc, iterations, sys.t_grid
