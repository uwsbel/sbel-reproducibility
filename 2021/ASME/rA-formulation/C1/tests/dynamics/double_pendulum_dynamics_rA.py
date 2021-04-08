#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from rA_sim_engine_3d import rASimEngine3D
import rA_gcons as gcons

sys = rASimEngine3D("../models/double_pendulum.mdl")

L = 2
w = 0.05
rho = 7800
b_len = [2*L, L]
for j, body in enumerate(sys.bodies_list[1:3]):
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
link1p = sys.r_sol[:, 0:3]
link2p = sys.r_sol[:, 3:6]

link1v = sys.r_dot_sol[:, 0:3]
link2v = sys.r_dot_sol[:, 3:6]

link1a = sys.r_ddot_sol[:, 0:3]
link2a = sys.r_ddot_sol[:, 3:6]

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link1p[:, 0])
ax1.set_title('Link 1 pos, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link1p[:, 1])
ax2.set_title('Link 1 pos, y')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link1p[:, 2])
ax3.set_title('Link 1 pos, z')

_, ax4 = plt.subplots()
ax4.plot(sys.t_grid, link2p[:, 0])
ax4.set_title('Link 2 pos, x')

_, ax5 = plt.subplots()
ax5.plot(sys.t_grid, link2p[:, 1])
ax5.set_title('Link 2 pos, y')

_, ax6 = plt.subplots()
ax6.plot(sys.t_grid, link2p[:, 2])
ax6.set_title('Link 2 pos, z')

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link1v[:, 0])
ax1.set_title('Link 1 vel, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link1v[:, 1])
ax2.set_title('Link 1 vel, y')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link1v[:, 2])
ax3.set_title('Link 1 vel, z')

_, ax4 = plt.subplots()
ax4.plot(sys.t_grid, link2v[:, 0])
ax4.set_title('Link 2 vel, x')

_, ax5 = plt.subplots()
ax5.plot(sys.t_grid, link2v[:, 1])
ax5.set_title('Link 2 vel, y')

_, ax6 = plt.subplots()
ax6.plot(sys.t_grid, link2v[:, 2])
ax6.set_title('Link 2 vel, z')

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link1a[:, 0])
ax1.set_title('Link 1 acc, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link1a[:, 1])
ax2.set_title('Link 1 acc, y')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link1a[:, 2])
ax3.set_title('Link 1 acc, z')

_, ax4 = plt.subplots()
ax4.plot(sys.t_grid, link2a[:, 0])
ax4.set_title('Link 2 acc, x')

_, ax5 = plt.subplots()
ax5.plot(sys.t_grid, link2a[:, 1])
ax5.set_title('Link 2 acc, y')

_, ax6 = plt.subplots()
ax6.plot(sys.t_grid, link2a[:, 2])
ax6.set_title('Link 2 acc, z')

plt.show()