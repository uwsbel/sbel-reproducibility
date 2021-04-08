#!/usr/bin/env python3
import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from rA_sim_engine_3d import rASimEngine3D
import rA_gcons as gcons

sys = rASimEngine3D("../models/slider_crank.mdl")

# body 1 properties
sys.bodies_list[0].m = 0.12
sys.bodies_list[0].J = np.diag([0.0001, 0.00001, 0.0001])
sys.nb += 1

# body 2 properties
sys.bodies_list[1].m = 0.5
sys.bodies_list[1].J = np.diag([0.004, 0.0004, 0.004])
sys.nb += 1

# body 3 properties
sys.bodies_list[2].m = 2
sys.bodies_list[2].J = np.diag([0.0001, 0.0001, 0.0001])
sys.nb += 1

# Alternative driving constraint for singularity encounter
sys.alternative_driver = copy(sys.constraint_list[-1])
sys.alternative_driver.a_bar_i = np.array([[0], [0], [1]])
sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-2*pi*t)",
                                                            "2*pi*cos(-2*pi*t-pi/2)",
                                                            "-4*pi**2*sin(-2*pi*t-pi/2)")

sys.t_start = 0
sys.t_end = 3
sys.h = 1e-3
sys.max_iters = 20
sys.tol = 1e-3

sys.dynamics_solver()