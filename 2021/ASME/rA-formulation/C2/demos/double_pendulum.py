import logging
import os
import argparse as arg
from collections import defaultdict
from time import process_time

import numpy as np
import matplotlib.pyplot as plt

from SimEngineMBD.rEps.system_reps import SystemREps
from SimEngineMBD.rp.system_rp import SystemRP
from SimEngineMBD.rA.system_ra import SystemRA
from SimEngineMBD.utils.physics import Z_AXIS
from SimEngineMBD.utils.tools import profiler, plot_many_kinematics, print_profiling, standard_setup

# Set up command-line options
parser = arg.ArgumentParser(description='Simulation of a two-link pendulum')
parser.add_argument('-t', '--end_time', type=float, default=10, dest='t_end')

model_file = os.path.join(os.path.dirname(__file__), '..models/double_pendulum.mdl')

# Get system and change some settings
sys, params = standard_setup(parser, model_file)
sys.set_g_acc(-9.81 * Z_AXIS)
sys.h = params.h
sys.tol = params.tol

if params.mode.startswith('kin'):
    raise ValueError('Cannot run double-pendulum in kinematics mode')

# Physical constants
L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar

b_len = [2*L, L]
for j, body in enumerate(sys.bodies):
    body.V = b_len[j] * w**2                    # [m^3] - bar volume
    body.m = ρ * body.V                         # [kg] - bar mass

    J_xx = 1/6 * body.m * w**2
    J_yz = 1/12 * body.m * (w**2 + b_len[j]**2)
    body.J = np.diag([J_xx, J_yz, J_yz])        # [??] - Inertia tensor of bar

for body in sys.bodies:
    print(body.m)
    print(body.J)

sys.initialize()

t_steps = int(params.t_end/params.h)
t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

# Create arrays to hold kinematic data
O_poses = [np.zeros((t_steps, 3)) for _ in sys.bodies]
O_vels = [np.zeros((t_steps, 3)) for _ in sys.bodies]
O_accs = [np.zeros((t_steps, 3)) for _ in sys.bodies]

frobs = [np.zeros(t_steps) for _ in sys.bodies]

num_iters = [0] * t_steps

start = process_time()
for i, t in enumerate(t_grid):
    sys.do_step(i, t)

    # Save stuff for this timestep
    num_iters[i] = sys.k

    for j, body in enumerate(sys.bodies):
        O_poses[j][i, :] = body.r.T
        O_vels[j][i, :] = body.dr.T
        O_accs[j][i, :] = body.ddr.T

        frobs[j][i] = np.linalg.norm(body.A.T @ body.A - np.identity(3), ord='fro')
Δt = process_time() - start

logging.info('Avg. iterations: {}'.format(np.mean(num_iters)))
logging.info('Simulation time: {}'.format(Δt))

plt.rcParams.update({'font.size': 30})

if params.plot:
    plot_many_kinematics(t_grid, O_poses, O_vels, O_accs, ['Pendulum 1', 'Pendulum 2'])


_, ax = plt.subplots()
ax.plot(t_grid, num_iters, 'k.')
ax.set(xlabel='time [s]', ylabel='Iterations to Convergence', title='Convergence with the Double Pendulum')
plt.xlim((0, params.t_end))
plt.yticks(np.arange(0, 10))

_, ax2 = plt.subplots()
ax2.plot(t_grid, frobs[0], label='Body 1')
ax2.plot(t_grid, frobs[1], label='Body 2')
ax2.set(xlabel='time [s]', ylabel=r'Frobenius Norm of $A^T A - I_3$', title='Rotation Matrix Non-Orthogonality')
ax2.set_yscale('log')
ax2.legend(loc='best', prop={'size': 24})

plt.show()
