#!/usr/bin/env python3
import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from reps_sim_engine_3d import repsSimEngine3D
import reps_gcons as gcons

sys = repsSimEngine3D("../models/four_link_rotated.mdl")

sys.nb = 3

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
sys.tol = 1e-10

sys.kinematics_solver()