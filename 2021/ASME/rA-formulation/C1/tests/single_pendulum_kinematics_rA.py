#!/usr/bin/env python3
import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from rA_sim_engine_3d import rASimEngine3D

sys = rASimEngine3D("../models/revJoint.mdl")

L = 2
w = 0.05
rho = 7800
b_len = [L]
for j, body in enumerate(sys.bodies_list[1:2]):
    V = b_len[j] * w**2
    body.m = rho * V
    J_xx = 1/6 * body.m * w**2
    J_yz = 1/12 * body.m * (w**2 + b_len[j]**2)
    body.J = np.diag([J_xx, J_yz, J_yz])
    sys.nb += 1

sys.t_start = 0
sys.t_end = 3
sys.h = 1e-3
sys.max_iters = 20
sys.tol = 1e-10

sys.kinematics_solver()