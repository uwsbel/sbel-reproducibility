"""
Provides helper functions that setup a system containing four common models
"""

import os
import argparse as arg
from copy import copy

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

    sys.set_g_acc(-9.81 * Z_AXIS)

    # Set derived values
    pend_length = 2*L                       # [m] - full length of the bar
    sys.bodies[0].V = pend_length * w**2             # [m^3] - bar volume
    sys.bodies[0].m = ρ * sys.bodies[0].V                     # [kg] - bar mass

    # Set Moment of Inertia
    J_xx = 1/6 * sys.bodies[0].m * w**2
    J_yz = 1/12 * sys.bodies[0].m * (w**2 + pend_length**2)
    sys.bodies[0].J = np.diag([J_xx, J_yz, J_yz])    # [kg*m^2] - Inertia tensor of bar

    # Functions for driving constraint
    t = sp.symbols('t')
    ang_sym = π/2 + π/4 * sp.cos(2*t)

    sys.g_cons.cons[-1].set_constraint_fn(sp.cos(ang_sym), t)

    return sys, params

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

def setup_slider_crank(args=None):
    """
    Sets up a system containing bodies and geometric constraints for a four link model
    """

    # Set up command-line options
    parser = arg.ArgumentParser(description='Simulation of a Haug\'s slider-crank model')

    model_file = os.path.join(os.path.dirname(__file__), 'models/slider_crank.json')

    # Get system and change some settings
    sys, params = standard_setup(parser, model_file, args)
    sys.set_g_acc(-9.81 * Z_AXIS)

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
