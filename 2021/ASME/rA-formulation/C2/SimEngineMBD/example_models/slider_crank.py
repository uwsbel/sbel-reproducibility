"""
Contains several functions related to a four link model

See models/four_link.json for gcon details and body initial conditions
"""

import argparse as arg
import logging
import os
from time import process_time
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ..utils.physics import Z_AXIS
from ..utils.tools import standard_setup, profiler, print_profiling

π = np.pi


def setup_slider_crank(args=None):
    """
    Sets up a system containing bodies and geometric constraints for a slider crank model
    """

    # Set up command-line options
    parser = arg.ArgumentParser(
        description='Simulation of a Haug\'s slider-crank model')

    model_file = os.path.join(os.path.dirname(
        __file__), 'models/slider_crank.json')

    # Get system and change some settings
    sys, params = standard_setup(parser, model_file, args)
    sys.set_g_acc(-9.81 * Z_AXIS)
    sys.h = params.h
    sys.tol = params.tol
    sys.solver_order = 1

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

    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)
    sys.g_cons.cons[-2].set_constraint_fn(1, t)

    sys.g_cons.alt_gcon = copy(sys.g_cons.cons[-1])
    sys.g_cons.alt_gcon.set_constraint_fn(sp.cos(ang_alt), t)
    sys.g_cons.alt_gcon.ai = Z_AXIS
    sys.g_cons.alt_index = len(sys.g_cons.cons) - 1

    return sys, params


def run_slider_crank(args=None):
    """
    Runs the slider crank model, saving off position-, velocity-, and acceleration-level
    data for immediate plotting or further processing
    """

    sys, params = setup_slider_crank(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    # (num bodies) x (x, y, z) x (time steps)
    pos_data = np.zeros((sys.nb, 3, t_steps))
    vel_data = np.zeros((sys.nb, 3, t_steps))
    acc_data = np.zeros((sys.nb, 3, t_steps))

    crank_rot = np.zeros((t_steps, 3))

    num_iters = np.zeros(t_steps)

    logging.info('Simulation started')
    start = process_time()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)

        # Save stuff for this timestep
        num_iters[i] = sys.k

        for j, body in enumerate(sys.bodies):
            pos_data[j, :, i] = body.r.T
            vel_data[j, :, i] = body.dr.T
            acc_data[j, :, i] = body.ddr.T

        crank_rot[i] = sys.bodies[0].ω.T
    Δt = process_time() - start
    logging.info('Simulation Ended')

    logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
    logging.info('Simulation time: {}'.format(Δt))

    plt.rcParams.update({'font.size': 30})

    if params.plot:
        _, ax1 = plt.subplots()
        ax1.plot(t_grid, pos_data[2, 0, :].T, 'k')
        ax1.set(xlabel='time [s]', ylabel='position [m]',
                title='x Position of slider')
        # plt.xticks(np.arange(0, 1.2, 0.2))
        # plt.yticks(np.arange(0, 0.35, 0.05))

        _, ax2 = plt.subplots()
        ax2.plot(t_grid, vel_data[2, 0, :].T, 'k')
        ax2.set(xlabel='time [s]', ylabel='velocity [m/s]',
                title='x Velocity of slider')
        # plt.xticks(np.arange(0, 1.2, 0.2))
        # plt.yticks(np.arange(-0.6, 0.7, 0.1))

        _, ax3 = plt.subplots()
        ax3.plot(t_grid, acc_data[2, 0, :].T, 'k')
        ax3.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
                title='x Acceleration of slider')
        # plt.xticks(np.arange(0, 1.2, 0.2))
        # plt.yticks(np.arange(-3, 16.2, 3.2))

        # _, ax4 = plt.subplots()
        # ax4.plot(t_grid, crank_rot[:, :])
        # ax4.set(xlabel='time [s]', ylabel='ω [rad/s]',
        #         title='Angular velocity of crank')
        # plt.yticks(np.arange(-2*π, 2.5*π, π/4))
        # # plt.ylim([-2.5*π, 2.5*π])

        # _, ax5 = plt.subplots()
        # ax5.plot(t_grid, acc_data[1, :, :].T)
        # ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
        #         title='Acceleration of rod')
        #
        # _, ax5 = plt.subplots()
        # ax5.plot(t_grid, acc_data[0, :, :].T)
        # ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
        #         title='Acceleration of crank')

        plt.show()

    return pos_data, vel_data, acc_data, num_iters, t_grid


def time_slider_crank(args=None):
    """
    Runs and times the slider crank simulation, saving off no data
    """

    sys, params = setup_slider_crank(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    start = process_time()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)
    Δt = process_time() - start

    return Δt


def profile_slider_crank(args=None):
    """
    Profiles the slider crank simulation, saving off no data
    """

    sys, params = setup_slider_crank(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    profiler.enable()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)
    profiler.disable()

    print_profiling(profiler)


if __name__ == '__main__':
    run_slider_crank()
