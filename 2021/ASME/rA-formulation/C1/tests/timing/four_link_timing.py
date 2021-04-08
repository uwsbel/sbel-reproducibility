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

import rA_gcons as gcons
from tools import standard_setup

def four_link(args):
    parser = arg.ArgumentParser(description='Simulation of Haug\'s four-link mechanism')
    parser.add_argument('-t', '--end_time', type=float, default=3, dest='t_end')

    model_files = "./models/four_link.mdl"

    sys, params = standard_setup(parser, model_files, args)
    sys.h = params.h
    sys.tol = params.tol

    # body 1 properties
    sys.bodies_list[1].m = 2.0
    sys.bodies_list[1].J = np.diag([4, 2, 0])
    sys.nb += 1

    # body 2 properties
    sys.bodies_list[2].m = 1
    sys.bodies_list[2].J = np.diag([12.4, 0.01, 0])
    sys.nb += 1

    # body 3 properties
    sys.bodies_list[3].m = 1
    sys.bodies_list[3].J = np.diag([4.54, 0.01, 0])
    sys.nb += 1

    # Alternative driving constraint for singularity encounter
    sys.alternative_driver = copy(sys.constraint_list[-1])
    sys.alternative_driver.a_bar_j = np.array([[0], [0], [1]])
    sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-pi * t - pi/2 + pi/2)",
                                                                "-pi*sin(pi*t)",
                                                                "-pi**2*cos(pi*t)")

    # sys.t_start = 0
    # sys.t_end = 1
    # sys.h = 1e-3
    #
    # sys.max_iters = 200
    # sys.tol = 1e-3

    sys.dynamics_solver()
    delta_t = sys.duration

    return delta_t