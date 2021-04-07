import logging
import argparse as arg
from collections import defaultdict
from time import process_time
from copy import copy

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from ..rEps.system_reps import SystemREps
from ..rp.system_rp import SystemRP
from ..rA.system_ra import SystemRA
from ..utils.physics import Y_AXIS, Z_AXIS
from ..utils.tools import profiler, plot_many_kinematics, print_profiling, standard_setup

π = np.pi

# Set up command-line options
parser = arg.ArgumentParser(description='Simulation of Haug\'s four-link mechanism')
parser.add_argument('-t', '--end_time', type=float, default=2.5, dest='t_end')

model_files = defaultdict(lambda: 'models/four_link.mdl')

# Get system and change some settings
sys, params = standard_setup(parser, model_files)
sys.set_g_acc(-9.81 * Z_AXIS)
sys.h = params.h
sys.tol = params.tol
sys.solver_order = 2

t_grid = np.arange(0, params.t_end, params.h)
t_steps = len(t_grid)

# Physical constants
L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 10                                  # [kg/m^3] - density of the bar
r = 2                                   # [m] - radius of the rotor
height = 0.1                            # [m] - the height of the rotor

# See Haug p. 459 for properties
# Link 1
sys.bodies[0].m = 2
sys.bodies[0].J = np.diag([4, 2, 0])

# Link 2
sys.bodies[1].m = 1
sys.bodies[1].J = np.diag([12.4, 0.01, 0])

# Link 3
sys.bodies[2].m = 1
sys.bodies[2].J = np.diag([4.54, 0.01, 0])

# Create driving constraint function and alternate function to swap to
t = sp.symbols('t')
ang_sym = π * t + π/2
ang_alt = ang_sym - π/2

con_num = len(sys.g_cons.cons) - 1

# Set driving constraint properties, create alternate constraint function
sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)
alt_dp1 = copy(sys.g_cons.cons[-1])
alt_dp1.set_constraint_fn(sp.cos(ang_alt), t)
alt_dp1.aj = Z_AXIS

sys.initialize()

pt_Bz = np.zeros(t_steps)
body_3z = np.zeros(t_steps)
rotor_rot = np.zeros(t_steps)
rotor_acc = np.zeros((t_steps, 3))
l2_acc = np.zeros((t_steps, 3))
l3_acc = np.zeros((t_steps, 3))

rotor_vel = np.zeros((t_steps, 3))
l2_vel = np.zeros((t_steps, 3))
l3_vel = np.zeros((t_steps, 3))

l2_pos = np.zeros((t_steps, 3))

num_iters = [0] * t_steps

start = process_time()
for i, t in enumerate(t_grid):
    # (Hack) swap g-cons to avoid driving constraint singularity
    if np.abs(np.abs(sys.g_cons.cons[con_num].f(t)) - 1) < 0.1:
        logging.info('Swapped g-con at time {:.3f}'.format(t))
        sys.g_cons.cons[con_num], alt_dp1 = alt_dp1, sys.g_cons.cons[con_num]

    sys.do_step(i, t)

    # Save stuff for this timestep
    num_iters[i] = sys.k

    pt_Bz[i] = sys.bodies[1].r[2] + (sys.bodies[1].A @ (6.1*Y_AXIS))[2]
    body_3z[i] = sys.bodies[2].r[2]
    rotor_rot[i] = sys.bodies[0].ω[0]

    rotor_acc[i] = sys.bodies[0].ddr.T
    l2_acc[i] = sys.bodies[1].ddr.T
    l3_acc[i] = sys.bodies[2].ddr.T

    rotor_vel[i] = sys.bodies[0].dr.T
    l2_vel[i] = sys.bodies[1].dr.T
    l3_vel[i] = sys.bodies[2].dr.T

    l2_pos[i] = sys.bodies[1].r.T
Δt = process_time() - start

logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
logging.info('Simulation time: {}'.format(Δt))

if params.save_data:
    np.savetxt("pos.csv", l2_pos, delimiter=",")
    np.savetxt("vel.csv", l2_vel, delimiter=",")
    np.savetxt("acc.csv", l2_acc, delimiter=",")

if params.read_data:
    pos_exact = np.genfromtxt("pos.csv", delimiter=",")
    vel_exact = np.genfromtxt("vel.csv", delimiter=",")
    acc_exact = np.genfromtxt("acc.csv", delimiter=",")
    
    pos_diff = pos_exact[-1] - l2_pos[-1]
    vel_diff = vel_exact[-1] - l2_vel[-1]
    acc_diff = acc_exact[-1] - l2_acc[-1]

    logging.info('Pos. diff: {}'.format(pos_diff))
    logging.info('Vel. diff: {}'.format(vel_diff))
    logging.info('Acc. diff: {}'.format(acc_diff))

if params.plot:
    _, ax1 = plt.subplots()
    ax1.plot(t_grid, pt_Bz)
    ax1.set(xlabel='time [s]', ylabel='position [m]', title='Position of point B')
    plt.xticks(np.arange(0, 3, 0.5))
    plt.yticks(np.arange(-3, 4, 1))

    _, ax2 = plt.subplots()
    ax2.plot(t_grid, body_3z)
    ax2.set(xlabel='time [s]', ylabel='position [m]', title='Position of link 3')
    plt.xticks(np.arange(0, 3, 0.5))
    plt.yticks(np.arange(2.5, 4.25, 0.25))

    _, ax3 = plt.subplots()
    ax3.plot(t_grid, rotor_rot)
    ax3.set(xlabel='time [s]', ylabel='ω [rad/s]',
            title='Angular velocity of rotor')
    plt.yticks(np.arange(-2*π, 2.5*π, π/4))
    # plt.ylim([-2.5*π, 2.5*π])

    _, ax4 = plt.subplots()
    ax4.plot(t_grid, rotor_acc[:, :])
    ax4.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of rotor')

    _, ax5 = plt.subplots()
    ax5.plot(t_grid, l2_acc[:, :])
    ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of link 2')

    _, ax6 = plt.subplots()
    ax6.plot(t_grid, l3_acc[:, :])
    ax6.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of link 3')

    _, ax7 = plt.subplots()
    ax7.plot(t_grid, rotor_vel[:, :])
    ax7.set(xlabel='time [s]', ylabel='velocity [m/s]',
            title='Velocity of rotor')

    _, ax8 = plt.subplots()
    ax8.plot(t_grid, l2_vel[:, :])
    ax8.set(xlabel='time [s]', ylabel='velocity [m/s]',
            title='Velocity of link 2')

    _, ax9 = plt.subplots()
    ax9.plot(t_grid, l3_vel[:, :])
    ax9.set(xlabel='time [s]', ylabel='velocity [m/s]',
            title='Velocity of link 3')

    plt.show()
