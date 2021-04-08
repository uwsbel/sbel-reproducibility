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

# step_sizes = [1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1e-1]

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

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link1[:, 0])
ax1.set_title('Body 1 acc, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link1[:, 0])
ax2.set_title('Body 1 acc, x')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link1[:, 0])
ax3.set_title('Body 1 acc, x')

_, ax4 = plt.subplots()
ax4.plot(sys.t_grid, link2[:, 0])
ax4.set_title('Link 2 acc, x')

_, ax5 = plt.subplots()
ax5.plot(sys.t_grid, link2[:, 1])
ax5.set_title('Link 2 acc, y')

_, ax6 = plt.subplots()
ax6.plot(sys.t_grid, link2[:, 2])
ax6.set_title('Link 2 acc, z')

_, ax7 = plt.subplots()
ax7.plot(sys.t_grid, link3[:, 0])
ax7.set_title('Link 3 acc, x')

_, ax8 = plt.subplots()
ax8.plot(sys.t_grid, link3[:, 1])
ax8.set_title('Link 3 acc, y')

_, ax9 = plt.subplots()
ax9.plot(sys.t_grid, link3[:, 2])
ax9.set_title('Link 3 acc, z')
#
plt.show()

# print("position reps {}, velocity reps P{}, acceleration reps {}".format(link2p[-1], link2v[-1], link2[-1]))