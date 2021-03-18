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

sys = rASimEngine3D("../models/revJoint.mdl")

L = 2
w = 0.05
rho = 7800
b_len = [2*L]
for j, body in enumerate(sys.bodies_list[1:2]):
    V = b_len[j] * w**2
    body.m = rho * V
    J_xx = 1/6 * body.m * w**2
    J_yz = 1/12 * body.m * (w**2 + b_len[j]**2)
    body.J = np.diag([J_xx, J_yz, J_yz])
    sys.nb += 1


sys.t_start = 0
sys.t_end = 1
sys.h = 1e-3

sys.max_iters = 20
sys.tol = 1e-3

sys.dynamics_solver()

# print positions of bodies
link1 = sys.r_ddot_sol[:, 0:3]

# _, ax1 = plt.subplots()
# ax1.plot(sys.t_grid, link1[:, 0])
# ax1.set_title('Link 1, x')
#
# _, ax2 = plt.subplots()
# ax2.plot(sys.t_grid, link1[:, 1])
# ax2.set_title('Link 1, y')
#
# _, ax3 = plt.subplots()
# ax3.plot(sys.t_grid, link1[:, 2])
# ax3.set_title('Link 1, z')
#
# plt.show()