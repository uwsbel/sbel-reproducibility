import logging
import argparse as arg
from collections import defaultdict
from time import perf_counter, process_time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from system_ra import SystemRA
from system_rp import SystemRP
from system_reps import SystemREps
from physics import Z_AXIS
from tools import profiler, plot_many_kinematics, print_profiling, standard_setup

π = np.pi

# Set up command-line options
parser = arg.ArgumentParser(description='Simulation of a single link pendulum')
parser.add_argument('-t', '--end_time', type=float, default=10, dest='t_end')

model_files = defaultdict(lambda: 'models/single_pendulum.mdl')

# Call utility function to setup our system and set the running mode
sys, params = standard_setup(parser, model_files)

# Get system and change some settings
sys.set_g_acc(-9.81 * Z_AXIS)
sys.h = params.h
sys.tol = params.tol
sys.solver_order = 1

# Physical constants
L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar

b_len = [2*L]
for j, body in enumerate(sys.bodies):
    body.V = b_len[j] * w**2                    # [m^3] - bar volume
    body.m = ρ * body.V                         # [kg] - bar mass

    J_xx = 1/6 * body.m * w**2
    J_yz = 1/12 * body.m * (w**2 + b_len[j]**2)
    body.J = np.diag([J_xx, J_yz, J_yz])        # [??] - Inertia tensor of bar

# Functions for driving constraint
t = sp.symbols('t')
ang_sym = π/2 + π/4 * sp.cos(2*t)

sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)

sys.initialize()

t_steps = int(params.t_end/params.h)
t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

# Create arrays to hold kinematic data
O_poses = np.zeros((t_steps, 3))
O_vels = np.zeros((t_steps, 3))
O_accs = np.zeros((t_steps, 3))

num_iters = [0] * t_steps

logging.info('Simulation started')
start = process_time()
for i, t in enumerate(t_grid):
    sys.do_step(i, t)

    # Save stuff for this timestep
    num_iters[i] = sys.k

    O_poses[i, :] = sys.bodies[0].r.T
    O_vels[i, :] = sys.bodies[0].dr.T
    O_accs[i, :] = sys.bodies[0].ddr.T
Δt = process_time() - start

logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
logging.info('Simulation time: {}'.format(Δt))

if params.save_data:
    np.savetxt("pos.csv", O_poses, delimiter=",")
    np.savetxt("vel.csv", O_vels, delimiter=",")
    np.savetxt("acc.csv", O_accs, delimiter=",")

if params.read_data:
    pos_exact = np.genfromtxt("pos.csv", delimiter=",")
    vel_exact = np.genfromtxt("vel.csv", delimiter=",")
    acc_exact = np.genfromtxt("acc.csv", delimiter=",")
    
    pos_diff = pos_exact[-1] - O_poses[-1]
    vel_diff = vel_exact[-1] - O_vels[-1]
    acc_diff = acc_exact[-1] - O_accs[-1]

    logging.info('Pos. diff: {}'.format(pos_diff))
    logging.info('Vel. diff: {}'.format(vel_diff))
    logging.info('Acc. diff: {}'.format(acc_diff))

# print_profiling(profiler)
# plot_many_kinematics(t_grid, O_poses, O_vels, O_accs, ['Pendulum'])

if params.plot:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.suptitle('Pendulum', fontsize=16)
    # O′ - position
    ax1.plot(t_grid, O_poses[:, 0], '.')
    ax1.plot(t_grid, O_poses[:, 1], '.')
    ax1.plot(t_grid, O_poses[:, 2], '.')
    ax1.set_title('Position of body')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('Position [m]')

    # O′ - velocity
    ax2.plot(t_grid, O_vels[:, 0], '.')
    ax2.plot(t_grid, O_vels[:, 1], '.')
    ax2.plot(t_grid, O_vels[:, 2], '.')
    ax2.set_title('Velocity of body')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Velocity [m/s]')

    # O′ - acceleration
    ax3.plot(t_grid, O_accs[:, 0], '.', label='x')
    ax3.plot(t_grid, O_accs[:, 1], '.', label='y')
    ax3.plot(t_grid, O_accs[:, 2], '.', label='z')
    ax3.set_title('Acceleration of body')
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('Acceleration [m/s²]')

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')

    plt.show()
