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

from ..utils.physics import Y_AXIS, Z_AXIS
from ..utils.tools import standard_setup

π = np.pi

def setup_four_link(args=None):
    """
    Sets up a system containing bodies and geometric constraints for a four link model
    """

    parser = arg.ArgumentParser(description='Simulation of Haug\'s four-link mechanism')

    model_file = os.path.join(os.path.dirname(__file__), 'models/four_link.json')

    # Get system and change some settings
    sys, params = standard_setup(parser, model_file, args)
    sys.set_g_acc(-9.81 * Z_AXIS)
    sys.h = params.h
    sys.tol = params.tol
    sys.solver_order = 2

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

    # Set driving constraint properties
    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)

    # Create alternate constraint function
    sys.g_cons.alt_gcon = copy(sys.g_cons.cons[-1])
    sys.g_cons.alt_gcon.set_constraint_fn(sp.cos(ang_alt), t)
    sys.g_cons.alt_gcon.aj = Z_AXIS
    sys.g_cons.alt_index = len(sys.g_cons.cons) - 1

    return sys, params

def run_four_link(args=None):
    """
    Runs the four link model, saving off position-, velocity-, and acceleration-level
    data for immediate plotting or further processing
    """

    sys, params = setup_four_link(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    # (num bodies) x (x, y, z) x (time steps)
    pos_data = np.zeros((sys.nb, 3, t_steps))
    vel_data = np.zeros((sys.nb, 3, t_steps))
    acc_data = np.zeros((sys.nb, 3, t_steps))

    pt_Bz = np.zeros(t_steps)
    rotor_rot = np.zeros(t_steps)

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

        pt_Bz[i] = sys.bodies[1].r[2] + (sys.bodies[1].A @ (6.1*Y_AXIS))[2]
        rotor_rot[i] = sys.bodies[0].ω[0]
    Δt = process_time() - start
    logging.info('Simulation Ended')

    logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
    logging.info('Simulation time: {}'.format(Δt))

    plt.rcParams.update({'font.size': 30})

    if params.plot:
        _, ax1 = plt.subplots()
        ax1.plot(t_grid, pt_Bz)
        ax1.set(xlabel='time [s]', ylabel='position [m]', title='Position of point B')
        plt.xticks(np.arange(0, 3, 0.5))
        plt.yticks(np.arange(-3, 4, 1))

        _, ax2 = plt.subplots()
        ax2.plot(t_grid, pos_data[2, 2, :])
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
        ax4.plot(t_grid, acc_data[0, :, :].T)
        ax4.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
                title='Acceleration of rotor')

        _, ax5 = plt.subplots()
        ax5.plot(t_grid, acc_data[1, :, :].T)
        ax5.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
                title='Acceleration of link 2')

        _, ax6 = plt.subplots()
        ax6.plot(t_grid, acc_data[2, :, :].T)
        ax6.set(xlabel='time [s]', ylabel='acceleration [m/s²]',
                title='Acceleration of link 3')

        _, ax7 = plt.subplots()
        ax7.plot(t_grid, vel_data[0, :, :].T)
        ax7.set(xlabel='time [s]', ylabel='velocity [m/s]',
                title='Velocity of rotor')

        _, ax8 = plt.subplots()
        ax8.plot(t_grid, vel_data[1, :, :].T)
        ax8.set(xlabel='time [s]', ylabel='velocity [m/s]',
                title='Velocity of link 2')

        _, ax9 = plt.subplots()
        ax9.plot(t_grid, vel_data[2, :, :].T)
        ax9.set(xlabel='time [s]', ylabel='velocity [m/s]',
                title='Velocity of link 3')

        plt.show()

    return pos_data, vel_data, acc_data, num_iters, t_grid

def time_four_link(args=None):
    """
    Runs and times the four link simulation, saving off no data
    """

    sys, params = setup_four_link(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    start = process_time()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)
    Δt = process_time() - start

    return Δt

if __name__ == '__main__':
    run_four_link()
