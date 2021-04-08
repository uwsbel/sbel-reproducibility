#!/usr/bin/env python3
import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from rp_sim_engine_3d import rpSimEngine3D
import rp_gcons as gcons

sys = rpSimEngine3D("../models/slider_crank.mdl")

sys.nb = 3

# Alternative driving constraint for singularity encounter
sys.alternative_driver = copy(sys.constraint_list[-1])
sys.alternative_driver.a_bar_i = np.array([[0], [0], [1]])
sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-2*pi*t + pi/2 - pi/2)",
                                                            "2*pi*sin(-2*pi*t + pi/2 - pi/2)",
                                                            "-4*pi**2*cos(2*pi*t + pi/2 - pi/2)")


sys.t_start = 0
sys.t_end = 3
sys.h = 1e-3
sys.max_iters = 20
sys.tol = 1e-10

sys.kinematics_solver()