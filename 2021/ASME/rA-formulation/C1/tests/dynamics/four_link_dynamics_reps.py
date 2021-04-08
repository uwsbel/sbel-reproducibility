#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from reps_sim_engine_3d import repsSimEngine3D
import reps_gcons as gcons
sys = repsSimEngine3D("../../models/four_link_rotated.mdl")

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
sys.t_end = 1
sys.h = 1e-3

sys.max_iters = 200
sys.tol = 1e-3

sys.dynamics_solver()

# print positions of bodies
link1 = sys.r_ddot_sol[:, 0:3]
link2 = sys.r_ddot_sol[:, 3:6]
link3 = sys.r_ddot_sol[:, 6:9]

link1v = sys.r_dot_sol[:, 0:3]
link2v = sys.r_dot_sol[:, 3:6]
link3v = sys.r_dot_sol[:, 6:9]

link1p = sys.r_sol[:, 0:3]
link2p = sys.r_sol[:, 3:6]
link3p = sys.r_sol[:, 6:9]


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
# _, ax4 = plt.subplots()
# ax4.plot(sys.t_grid, link2[:, 0])
# ax4.set_title('Link 2, x')
#
# _, ax5 = plt.subplots()
# ax5.plot(sys.t_grid, link2[:, 1])
# ax5.set_title('Link 2, y')
#
# _, ax6 = plt.subplots()
# ax6.plot(sys.t_grid, link2[:, 2])
# ax6.set_title('Link 2, z')
#
_, ax7 = plt.subplots()
ax7.plot(sys.t_grid, link3[:, 0])
ax7.set_title('Link 3 acc, x')

# _, ax8 = plt.subplots()
# ax8.plot(sys.t_grid, link3[:, 1])
# ax8.set_title('Link 3, y')
#
# _, ax9 = plt.subplots()
# ax9.plot(sys.t_grid, link3[:, 2])
# ax9.set_title('Link 3, z')

_, ax7 = plt.subplots()
ax7.plot(sys.t_grid, link3v[:, 0])
ax7.set_title('Link 3 vel, x')

_, ax7 = plt.subplots()
ax7.plot(sys.t_grid, link3p[:, 0])
ax7.set_title('Link 3 pos, x')
#
plt.show()