"""
Contains several functions related to a double pendulum model

See models/double_pendulum.json for gcon details and body initial conditions
"""

import argparse as arg
import logging
import os
from time import process_time

import matplotlib.pyplot as plt
import numpy as np

from ..utils.physics import Z_AXIS
from ..utils.tools import standard_setup, plot_many_kinematics

# Pendulum Physical constants
L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar

π = np.pi

def setup_double_pendulum(args=None):
    """
    Sets up a system containing bodies and geometric constraints for a double pendulum model
    """

    # Set up command-line options
    parser = arg.ArgumentParser(description='Simulation of a two-link pendulum')
    model_file = os.path.join(os.path.dirname(__file__), 'models/double_pendulum.json')

    # Get system and change some settings
    sys, params = standard_setup(parser, model_file, args)
    sys.set_g_acc(-9.81 * Z_AXIS)
    sys.h = params.h
    sys.tol = params.tol

    if params.mode.startswith('kin'):
        raise ValueError('Cannot run double-pendulum in kinematics mode')

    pend_len = [2*L, L]
    for j, body in enumerate(sys.bodies):
        body.V = pend_len[j] * w**2                 # [m^3] - bar volume
        body.m = ρ * body.V                         # [kg] - bar mass

        J_xx = 1/6 * body.m * w**2
        J_yz = 1/12 * body.m * (w**2 + pend_len[j]**2)
        body.J = np.diag([J_xx, J_yz, J_yz])        # [kg*m^2] - Inertia tensor of bar

    return sys, params

def run_double_pendulum(args=None):
    """
    Runs the Double Pendulum model, saving off position-, velocity-, and acceleration-level
    data for immediate plotting or further processing
    """

    sys, params = setup_double_pendulum(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    # (num bodies) x (x, y, z) x (time steps)
    pos_data = np.zeros((sys.nb, 3, t_steps))
    vel_data = np.zeros((sys.nb, 3, t_steps))
    acc_data = np.zeros((sys.nb, 3, t_steps))

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
    Δt = process_time() - start
    logging.info('Simulation Ended')

    logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
    logging.info('Simulation time: {}'.format(Δt))

    plt.rcParams.update({'font.size': 30})

    if params.plot:
        plot_many_kinematics(t_grid, pos_data, vel_data, acc_data, ['Pendulum 1', 'Pendulum 2'])

    plt.show()

    return pos_data, vel_data, acc_data, num_iters, t_grid

def time_double_pendulum(args=None):
    """
    Runs and times the double pendulum simulation, saving off no data
    """

    sys, params = setup_double_pendulum(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    start = process_time()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)
    Δt = process_time() - start

    return Δt

if __name__ == '__main__':
    run_double_pendulum()
