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

sys = repsSimEngine3D("../../models/slider_crank_rotated.mdl")

sys.nb = 3

# Alternative driving constraint for singularity encounter
sys.alternative_driver = copy(sys.constraint_list[-1])
sys.alternative_driver.a_bar_i = np.array([[0], [0], [1]])
sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-2*pi*t + pi/2 - pi/2)",
                                                            "2*pi*sin(-2*pi*t + pi/2 - pi/2)",
                                                            "-4*pi**2*cos(2*pi*t + pi/2 - pi/2)")


sys.t_start = 0
sys.t_end = 1
sys.h = 1e-3

sys.max_iters = 200
sys.tol = 1e-6

sys.kinematics_solver()

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

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link3[:, 0])
ax1.set_title('Link 3 acc, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link3v[:, 0])
ax2.set_title('Link 3 vel, x')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link3p[:, 0])
ax3.set_title('Link 3 pos, x')

# accelerations
# _, ax1 = plt.subplots()
# ax1.plot(sys.t_grid, link3[:, 0])
# ax1.set_title('Slider acc, x')
#
# _, ax2 = plt.subplots()
# ax2.plot(sys.t_grid, link3[:, 1])
# ax2.set_title('Slider acc, y')
#
# _, ax3 = plt.subplots()
# ax3.plot(sys.t_grid, link3[:, 2])
# ax3.set_title('Slider acc, z')
#
# _, ax4 = plt.subplots()
# ax4.plot(sys.t_grid, link2[:, 0])
# ax4.set_title('Rod acc, x')
#
# _, ax5 = plt.subplots()
# ax5.plot(sys.t_grid, link2[:, 1])
# ax5.set_title('Rod acc, y')
#
# _, ax6 = plt.subplots()
# ax6.plot(sys.t_grid, link2[:, 2])
# ax6.set_title('Rod acc, z')
#
# _, ax7 = plt.subplots()
# ax7.plot(sys.t_grid, link1[:, 0])
# ax7.set_title('Crank acc, x')
#
# _, ax8 = plt.subplots()
# ax8.plot(sys.t_grid, link1[:, 1])
# ax8.set_title('Crank acc, y')
#
# _, ax9 = plt.subplots()
# ax9.plot(sys.t_grid, link1[:, 2])
# ax9.set_title('Crank acc, z')
#
# # velocities
# _, ax1 = plt.subplots()
# ax1.plot(sys.t_grid, link3v[:, 0])
# ax1.set_title('Slider vel, x')

# _, ax2 = plt.subplots()
# ax2.plot(sys.t_grid, link3v[:, 1])
# ax2.set_title('Link 3 vel, y')
#
# _, ax3 = plt.subplots()
# ax3.plot(sys.t_grid, link3v[:, 2])
# ax3.set_title('Link 3 vel, z')

# _, ax4 = plt.subplots()
# ax4.plot(sys.t_grid, link2v[:, 0])
# ax4.set_title('Link 2 vel, x')
#
# _, ax5 = plt.subplots()
# ax5.plot(sys.t_grid, link2v[:, 1])
# ax5.set_title('Link 2 vel, y')
#
# _, ax6 = plt.subplots()
# ax6.plot(sys.t_grid, link2v[:, 2])
# ax6.set_title('Link 2 vel, z')
#
# _, ax7 = plt.subplots()
# ax7.plot(sys.t_grid, link1v[:, 0])
# ax7.set_title('Link 1 vel, x')
#
# _, ax8 = plt.subplots()
# ax8.plot(sys.t_grid, link1v[:, 1])
# ax8.set_title('Link 1 vel, y')
#
# _, ax9 = plt.subplots()
# ax9.plot(sys.t_grid, link1v[:, 2])
# ax9.set_title('Link 1 vel, z')

# position
# _, ax1 = plt.subplots()
# ax1.plot(sys.t_grid, link3p[:, 0])
# ax1.set_title('Link 3 pos, x')
#
# _, ax2 = plt.subplots()
# ax2.plot(sys.t_grid, link3p[:, 1])
# ax2.set_title('Link 3 pos, y')
#
# _, ax3 = plt.subplots()
# ax3.plot(sys.t_grid, link3p[:, 2])
# ax3.set_title('Link 3 pos, z')
#
# _, ax4 = plt.subplots()
# ax4.plot(sys.t_grid, link2p[:, 0])
# ax4.set_title('Link 2 pos, x')
#
# _, ax5 = plt.subplots()
# ax5.plot(sys.t_grid, link2p[:, 1])
# ax5.set_title('Link 2 pos, y')
#
# _, ax6 = plt.subplots()
# ax6.plot(sys.t_grid, link2p[:, 2])
# ax6.set_title('Link 2 pos, z')
#
# _, ax7 = plt.subplots()
# ax7.plot(sys.t_grid, link1p[:, 0])
# ax7.set_title('Link 1 pos, x')
#
# _, ax8 = plt.subplots()
# ax8.plot(sys.t_grid, link1p[:, 1])
# ax8.set_title('Link 1 pos, y')
#
# _, ax9 = plt.subplots()
# ax9.plot(sys.t_grid, link1p[:, 2])
# ax9.set_title('Link 1 pos, z')


plt.show()