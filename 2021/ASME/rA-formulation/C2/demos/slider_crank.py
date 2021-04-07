import logging
import os
import argparse as arg
from collections import defaultdict
from time import process_time
from copy import copy

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from SimEngineMBD.rEps.system_reps import SystemREps
from SimEngineMBD.rp.system_rp import SystemRP
from SimEngineMBD.rA.system_ra import SystemRA
from SimEngineMBD.utils.physics import Z_AXIS
from SimEngineMBD.utils.tools import profiler, plot_many_kinematics, plot_kinematics_analysis, print_profiling, standard_setup

π = np.pi

# Set up command-line options
parser = arg.ArgumentParser(description='Simulation of a Haug\'s slider-crank model')
parser.add_argument('-t', '--end_time', type=float, default=3, dest='t_end')

model_file = os.path.join(os.path.dirname(__file__), '..models/slider_crank.mdl')

# Get system and change some settings
sys, params = standard_setup(parser, model_file)
sys.set_g_acc(-9.81 * Z_AXIS)
sys.h = params.h
sys.tol = params.tol
sys.solver_order = 2

# See Haug p. 456 for properties
# Crank
sys.bodies[0].m = 0.12
sys.bodies[0].J = np.diag([1e-4, 1e-5, 1e-4])

# Connecting Rod
sys.bodies[1].m = 0.5
sys.bodies[1].J = np.diag([4e-3, 4e-4, 4e-3])

# Slider
sys.bodies[2].m = 2
sys.bodies[2].J = np.diag([1e-4, 1e-4, 1e-4])

# Functions for initial driving constraint
t = sp.symbols('t')
ang_sym = -2*π*t + π/2
ang_alt = ang_sym - π/2

con_num = len(sys.g_cons.cons) - 1

sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)
sys.g_cons.cons[-2].set_constraint_fn(1, t)

alt_dp1 = copy(sys.g_cons.cons[-1])
alt_dp1.set_constraint_fn(sp.cos(ang_alt), t)
alt_dp1.ai = Z_AXIS

sys.initialize()

t_grid = np.arange(0, params.t_end, params.h)
t_steps = len(t_grid)

# Create arrays to hold kinematic data
slider_pos = np.zeros((t_steps, 3))
slider_vel = np.zeros((t_steps, 3))
slider_acc = np.zeros((t_steps, 3))
crank_rot = np.zeros((t_steps, 3))

rod_acc = np.zeros((t_steps, 3))
crank_acc = np.zeros((t_steps, 3))

Φ_normal = np.zeros(t_steps)
γ_normal = np.zeros(t_steps)
Φ_alt = np.zeros(t_steps)
γ_alt = np.zeros(t_steps)

num_iters = [0] * t_steps

start = process_time()
profiler.enable()
for i, t in enumerate(t_grid):
    # (Hack) swap g-cons to avoid driving constraint singularity
    if np.abs(np.abs(sys.g_cons.cons[con_num].f(t)) - 1) < 0.1:
        logging.info('Swapped g-con at time {:.3f}'.format(t))
        sys.g_cons.cons[con_num], alt_dp1 = alt_dp1, sys.g_cons.cons[con_num]
        logging.debug(sys.g_cons.cons[con_num].f(t))
        logging.debug(alt_dp1.f(t))

    sys.do_step(i, t)

    # Save stuff for this timestep
    num_iters[i] = sys.k

    slider_pos[i] = sys.bodies[2].r.T
    slider_vel[i] = sys.bodies[2].dr.T
    slider_acc[i] = sys.bodies[2].ddr.T

    crank_acc[i] = sys.bodies[0].ddr.T
    rod_acc[i] = sys.bodies[1].ddr.T

    crank_rot[i] = sys.bodies[0].ω.T

    Φ_normal[i] = sys.g_cons.cons[con_num].get_phi(t)
    Φ_alt[i] = alt_dp1.get_phi(t)
    γ_normal[i] = sys.g_cons.cons[con_num].get_gamma(t)
    γ_alt[i] = alt_dp1.get_gamma(t)
profiler.disable()
Δt = process_time() - start

logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
logging.info('Simulation time: {}'.format(Δt))

print_profiling(profiler)

if params.save_data:
    np.savetxt("pos.csv", slider_pos, delimiter=",")
    np.savetxt("vel.csv", slider_vel, delimiter=",")
    np.savetxt("acc.csv", slider_acc, delimiter=",")

if params.read_data:
    pos_exact = np.genfromtxt("pos.csv", delimiter=",")
    vel_exact = np.genfromtxt("vel.csv", delimiter=",")
    acc_exact = np.genfromtxt("acc.csv", delimiter=",")
    
    pos_diff = pos_exact[-1] - slider_pos[-1]
    vel_diff = vel_exact[-1] - slider_vel[-1]
    acc_diff = acc_exact[-1] - slider_acc[-1]

    logging.info('Pos. diff: {}'.format(pos_diff))
    logging.info('Vel. diff: {}'.format(vel_diff))
    logging.info('Acc. diff: {}'.format(acc_diff))

if params.plot:
    _, ax1 = plt.subplots()
    ax1.plot(t_grid, slider_pos[:, 0], 'k')
    ax1.set(xlabel='time [s]', ylabel='position [m]', title='Position of slider')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 0.35, 0.05))

    _, ax2 = plt.subplots()
    ax2.plot(t_grid, slider_vel[:, 0], 'k')
    ax2.set(xlabel='time [s]', ylabel='velocity [m/s]', title='Velocity of slider')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(-0.6, 0.7, 0.1))

    _, ax3 = plt.subplots()
    ax3.plot(t_grid, slider_acc[:, 0], 'k')
    ax3.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of slider')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(-3, 16.2, 3.2))

    _, ax4 = plt.subplots()
    ax4.plot(t_grid, crank_rot[:, :])
    ax4.set(xlabel='time [s]', ylabel='ω [rad/s]',
            title='Angular velocity of crank')
    plt.yticks(np.arange(-2*π, 2.5*π, π/4))
    # plt.ylim([-2.5*π, 2.5*π])

    _, ax5 = plt.subplots()
    ax5.plot(t_grid, rod_acc[:, :])
    ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of rod')

    _, ax5 = plt.subplots()
    ax5.plot(t_grid, crank_acc[:, :])
    ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
            title='Acceleration of crank')

    _, ax10 = plt.subplots()
    ax10.plot(t_grid, Φ_normal)
    ax10.plot(t_grid, Φ_alt)
    ax10.set(xlabel='time [s]', ylabel='Φ',
            title='Normal Constraint Satisfication')

    _, ax11 = plt.subplots()
    ax11.plot(t_grid, γ_normal)
    ax11.plot(t_grid, γ_alt)
    ax11.set(xlabel='time [s]', ylabel='γ',
            title='Normal γ values')

    plt.show()
