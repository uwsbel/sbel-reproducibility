#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

import logging
import argparse as arg

from rA_sim_engine_3d import rASimEngine3D
from rp_sim_engine_3d import rpSimEngine3D
from reps_sim_engine_3d import repsSimEngine3D

import reps_gcons as gcons
from tools import standard_setup

def single_pendulum(args):
    parser = arg.ArgumentParser(description='Simulation of a single pendulum mechanism')

    model_files = "./models/revJoint.mdl"

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
    iterations = sys.avg_iterations
    pos = np.zeros((sys.nb, 3, sys.N))
    vel = np.zeros((sys.nb, 3, sys.N))
    acc = np.zeros((sys.nb, 3, sys.N))
    for t in range(sys.N):
        for body in sys.bodies_list:
            if body.is_ground:
                pass
            else:
                pos[(body.body_id - 1), :, t] = sys.r_sol[t, (body.body_id - 1) * 3:((body.body_id - 1) * 3) + 3].T
                vel[(body.body_id - 1), :, t] = sys.r_dot_sol[t, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3].T
                acc[(body.body_id - 1), :, t] = sys.r_ddot_sol[t, (body.body_id - 1) * 3:(body.body_id - 1) * 3 + 3].T

    t_grid = sys.t_grid

    _, ax1 = plt.subplots()
    ax1.plot(t_grid, pos[0, :, :].T)
    ax1.set(xlabel='time [s]', ylabel='position [m]',
            title='Position')

    _, ax2 = plt.subplots()
    ax2.plot(t_grid, acc[0, :, :].T)
    ax2.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration')

    _, ax3 = plt.subplots()
    ax3.plot(t_grid, vel[0, :, :].T)
    ax3.set(xlabel='time [s]', ylabel='velocity [m/s]',
            title='Velocity')

    plt.show()

    return pos, vel, acc, iterations, sys.t_grid
