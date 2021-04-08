#!/usr/bin/env python3
import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
from rA_sim_engine_3d import rASimEngine3D
import rA_gcons as gcons

sys = rASimEngine3D("../models/four_link.mdl")

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

sys.t_start = 0
sys.t_end = 3
sys.h = 1e-3
sys.max_iters = 20
sys.tol = 1e-3

sys.dynamics_solver()