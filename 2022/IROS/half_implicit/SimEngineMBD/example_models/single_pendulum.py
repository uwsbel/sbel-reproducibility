import argparse as arg
import logging
import os
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ..utils.physics import Z_AXIS
from ..utils.tools import standard_setup

# Pendulum Physical constants
L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar

π = np.pi

def setup_single_pendulum(args=None):
    """
    Sets up a system containing bodies and geometric constraints for a single pendulum model
    """

    # Set up command-line options
    parser = arg.ArgumentParser(description='Simulation of a single link pendulum')
    model_file = os.path.join(os.path.dirname(__file__), 'models/single_pendulum.json')

    # Call utility function to setup our system and set the running mode
    sys, params = standard_setup(parser, model_file, args)
    sys.h = params.h
    sys.tol = params.tol

    sys.set_g_acc(-9.81 * Z_AXIS)

    # Set derived values
    pend_length = 2*L                       # [m] - full length of the bar
    sys.bodies[0].V = pend_length * w**2             # [m^3] - bar volume
    sys.bodies[0].m = ρ * sys.bodies[0].V                     # [kg] - bar mass

    # Set Moment of Inertia
    J_xx = 1/6 * sys.bodies[0].m * w**2
    J_yz = 1/12 * sys.bodies[0].m * (w**2 + pend_length**2)
    sys.bodies[0].J = np.diag([J_xx, J_yz, J_yz])    # [kg*m^2] - Inertia tensor of bar
    sys.bodies[0].J_inv = np.diag([1/J_xx, 1/J_yz, 1/J_yz])

    # Functions for driving constraint
    t = sp.symbols('t')
    ang_sym = π/2 + π/4 * sp.cos(2*t)

    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)

    return sys, params

def run_single_pendulum(args=None):
    """
    Runs the Single Pendulum model, saving off position-, velocity-, and acceleration-level
    data for immediate plotting or further processing
    """

    sys, params = setup_single_pendulum(args)

    sys.initialize()
    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    # (num bodies) x (x, y, z) x (time steps)
    pos_data = np.zeros((sys.nb, 3, t_steps))
    vel_data = np.zeros((sys.nb, 3, t_steps))
    acc_data = np.zeros((sys.nb, 3, t_steps))

    num_iters = np.zeros(t_steps)

    logging.info('single pendulum simulation started')
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
    logging.info('single pendulum simulation Ended')

    logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
    logging.info('Simulation time: {}'.format(Δt))

    if params.plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        fig.suptitle('Pendulum', fontsize=16)
        # O′ - position
        ax1.plot(t_grid, pos_data[0, 0, :])
        ax1.plot(t_grid, pos_data[0, 1, :])
        ax1.plot(t_grid, pos_data[0, 2, :])
        ax1.set_title('Position of body')
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('Position [m]')

        # O′ - velocity
        ax2.plot(t_grid, vel_data[0, 0, :])
        ax2.plot(t_grid, vel_data[0, 1, :])
        ax2.plot(t_grid, vel_data[0, 2, :])
        ax2.set_title('Velocity of body')
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Velocity [m/s]')

        # O′ - acceleration
        ax3.plot(t_grid, acc_data[0, 0, :], label='x')
        ax3.plot(t_grid, acc_data[0, 1, :], label='y')
        ax3.plot(t_grid, acc_data[0, 2, :], label='z')
        ax3.set_title('Acceleration of body')
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('Acceleration [m/s²]')

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

        plt.show()

    return pos_data, vel_data, acc_data, num_iters, t_grid

def time_single_pendulum(args=None):
    """
    Runs and times the single pendulum simulation, saving off no data
    """

    sys, params = setup_single_pendulum(args)

    sys.initialize()

    t_steps = int(params.t_end/params.h)
    t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

    start = process_time()
    for i, t in enumerate(t_grid):
        sys.do_step(i, t)
    Δt = process_time() - start

    return Δt

if __name__ == '__main__':
    run_single_pendulum()
