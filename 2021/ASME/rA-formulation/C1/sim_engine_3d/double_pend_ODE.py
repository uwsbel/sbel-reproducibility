#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# The gravitational acceleration (m.s-2).
g = 9.81
# Check that the integration conserves total energy to within this absolute
# tolerance.
EDRIFT = 0.05


def deriv(t, y, L1, L2, m1, m2):
    """Return the derivatives of y = theta1, p_theta1, theta2, p_theta2.

    These are the generalized coordinates (here, angles) and generalized
    momenta for the two rigid rods.

    """

    theta1, p_theta1, theta2, p_theta2 = y

    M = m1 + 3 * m2
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    Lr = L1 / L2
    den = 4 * M - 9 * m2 * c ** 2

    theta1dot = 6 / L1 ** 2 * (2 * p_theta1 - 3 * Lr * c * p_theta2) / den
    theta2dot = 6 / m2 / L2 ** 2 * (
            (2 * p_theta2 * M - 3 * m2 / Lr * c * p_theta1) / den)
    term = m2 * L1 * L2 / 2 * theta1dot * theta2dot * s
    p_theta1dot = -term - (m1 / 2 + m2) * g * L1 * np.sin(theta1)
    p_theta2dot = term - m2 / 2 * g * L2 * np.sin(theta2)

    return theta1dot, p_theta1dot, theta2dot, p_theta2dot


def calc_H(y, L1, L2, m1, m2):
    """Calculate the Hamiltonian at y = theta1, p_theta1, theta2, p_theta2."""
    theta1, p_theta1, theta2, p_theta2 = y

    theta1dot, p_theta1dot, theta2dot, p_theta2dot = deriv(None, y, L1, L2,
                                                           m1, m2)
    # The Lagrangian
    c = np.cos(theta1 - theta2)
    L = (m1 * (L1 * theta1dot) ** 2 / 6 + m2 * (L2 * theta2dot) ** 2 / 6
         + m2 / 2 * ((L1 * theta1dot) ** 2 + L1 * L2 * theta1dot * theta2dot * c)
         + g * L1 * np.cos(theta1) * (m1 / 2 + m2)
         + g * L2 * np.cos(theta2) * m2 / 2
         )

    # The Hamiltonian
    H = p_theta1 * theta1dot + p_theta2 * theta2dot - L
    return H


def calcPend(tmax, dt, L1, L2, m1, m2, theta1_0, theta2_0, p_theta1_0, p_theta2_0):
    y0 = np.array([theta1_0, p_theta1_0, theta2_0, p_theta2_0])
    # Could call calc_H, but since the initial p_thetai are zero, and the
    # Hamiltonian is conserved (since the Langrangian has no explicit time-
    # dependence, H0 is just the potential energy of the initial configuration:
    H0 = -g * (L1 * np.cos(theta1_0) * (m1 / 2 + m2) +
               L2 * np.cos(theta2_0) * m2 / 2)

    # Do the numerical integration of the equations of motion.
    y = solve_ivp(deriv, (0, tmax), y0, method='Radau', dense_output=True,
                  args=(L1, L2, m1, m2))
    # theta1dot, p_theta1dot, theta2dot, p_theta2dot = deriv(0, y, L1, L2, m1, m2)
    # theta1ddot = p_theta1dot/m1
    # theta2ddot = p_theta2dot/m2

    # Check that the Hamiltonian didn't drift too much.
    H = calc_H(y.y, L1, L2, m1, m2)
    if any(abs(H - H0) > EDRIFT):
        print('Maximum energy drift exceeded')

    # Unpack dynamical variables as a function of time.
    theta1, p_theta1, theta2, p_theta2 = y.sol(t)
    #
    # theta1dot = p_theta1/m1
    # theta2dot = p_theta2/m2

    # Convert to Cartesian coordinates of the two rods.
    x1 = 1/2 * L1 * np.sin(theta1)
    y1 = -1/2 * L1 * np.cos(theta1)
    x2 = L1 * np.sin(theta1) + 1/2 * L2 * np.sin(theta2)
    y2 = -L1 * np.cos(theta1) - 1/2 * L2 * np.cos(theta2)

    # x1dot = 1/2 * L1 * theta1dot * np.cos(theta1)
    # y1dot = 1/2 * L1 * theta1dot * np.sin(theta1)
    # x2dot = L1 * theta1dot * np.cos(theta1) + 1/2 * L2 * theta2dot * np.cos(theta2)
    # y2dot = L1 * theta1dot * np.sin(theta1) + 1/2 * L2 * theta2dot * np.sin(theta2)

    # x1ddot = 1/2 * L1 * (-theta1dot**2 * np.sin(theta1) + theta1ddot * np.cos(theta1))
    # y1ddot = 1/2 * L1 * (theta1dot**2 * np.cos(theta1) + theta1ddot * np.sin(theta1))
    # x2ddot = L1 * (-theta1dot**2 * np.sin(theta1) + theta1ddot * np.cos(theta1)) \
    #          + 1/2 * L2 * (-theta2dot**2 * np.sin(theta2) + theta2ddot * np.cos(theta2))
    # y2ddot = L1 * (theta1dot**2 * np.cos(theta1) + theta1ddot * np.sin(theta1)) \
    #          + 1/2 * L2 * (theta2dot**2 * np.cos(theta2) + theta2ddot * np.sin(theta2))

    return x1, y1, x2, y2, theta1, theta2


def make_plot(i, x1, y1, x2, y2, max_trail):
    """
    Plot and save an image of the double pendulum configuration for time
    point i.
    """

    # The pendulum rods (thick, black).
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=8, c='k')

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns - j) * s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j / ns) ** 2
        # Two trails, initiating at the centres of mass of the two rods.
        ax.plot(x1[imin:imax] / 2, y1[imin:imax] / 2, c='b', solid_capstyle='butt',
                lw=2, alpha=alpha)
        ax.plot((x1[imin:imax] + x2[imin:imax]) / 2,
                (y1[imin:imax] + y2[imin:imax]) / 2,
                c='r', solid_capstyle='butt', lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1 - L2, L1 + L2)
    ax.set_ylim(-L1 - L2, L1 + L2)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i // di), dpi=72)
    plt.cla()


if __name__ == '__main__':
    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 5, 1e-7
    t = np.arange(0, tmax + dt, dt)
    N = int(tmax / dt)
    t = np.linspace(0, tmax, N, endpoint=True)
    # Pendulum rod lengths (m) and masses (kg).
    L1, L2 = 4, 2
    m1, m2 = 78, 39

    # Initial conditions:
    # angles: theta1, theta2 and generalized momenta: p_theta1, p_theta2
    theta1_0, theta2_0 = np.pi / 2, 0
    p_theta1_0, p_theta2_0 = 0, 0

    y1, z1, y2, z2, theta1, theta2 = calcPend(tmax, dt, L1, L2, m1, m2, theta1_0, theta2_0, p_theta1_0, p_theta2_0)

    df_pos_y_reference = pd.DataFrame()
    df_pos_y_reference['Time'] = t
    df_pos_y_reference.to_pickle("./position_y_reference.pkl")
    #df_pos_reference = pd.read_pickle("./position_y_reference.pkl")
    df_pos_z_reference = pd.DataFrame()
    df_pos_z_reference['Time'] = t
    df_pos_z_reference['pos_b2'] = z2
    df_pos_z_reference['theta1'] = theta1
    df_pos_z_reference['theta2'] = theta2
    print(df_pos_z_reference)
    df_pos_z_reference.to_pickle("./position_z_reference.pkl")
    #df_pos_reference = pd.read_pickle("./position_z_reference.pkl")

    # sampled_grid = t.tolist()[::10000]
    # sampled_z1 = z1.tolist()[::10000]

    # sns.set()
    # sns.set_style("ticks")
    # sns.set_context("paper")
    # pos_plot_2z = sns.lineplot(x=sampled_grid, y=sampled_z1)
    # plt.show()

    # sns.set()
    # sns.set_style("ticks")
    # sns.set_context("paper")
    # pos_plot_2z = sns.lineplot(x=grid, y=y1)
    # plt.show()
    #
    # sns.set()
    # sns.set_style("ticks")
    # sns.set_context("paper")
    # pos_plot_2z = sns.lineplot(x=grid, y=z2)
    # plt.show()
    #
    # sns.set()
    # sns.set_style("ticks")
    # sns.set_context("paper")
    # pos_plot_2z = sns.lineplot(x=grid, y=y2)
    # plt.show()


    # Make an image every di time points, corresponding to a frame rate of fps
    # frames per second.
    # Frame rate, s-1
    fps = 10
    di = int(1 / fps / dt)
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    # Plot a trail of the m2 bob's position for the last trail_secs seconds.
    trail_secs = 1
    # This corresponds to max_trail time points.
    max_trail = int(trail_secs / dt)

    # for i in range(0, t.size, di):
    #     print(i // di, '/', t.size // di)
    #     make_plot(i, x1, y1, x2, y2, max_trail)
