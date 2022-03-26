#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:58:07 2021

@author: simone

ODE to solve recursive formulation of a double pendulum

"""
import numpy as np


def recursivePendulum(tmax, h):
    # for graphics. Leave out for now
    # from panda3d.core import LPoint3, LVector3, BitMask32, Quat, Filename
    render = True
    try:
        import render
    except:
        render = False
    if render:
        render = render.OnscreenRender()
        render.setupLights()
        rendModels = []

    """ DATA """
    L = 2  # [m] - length of the bar
    w = 0.05  # [m] - side length of bar
    ρ = 7800  # [kg/m^3] - density of the bar
    inertias = [0, 0]  # [1.0,1.0]
    masses = [0, 0]  # [1.0,1.0]
    link_lengths = [2 * L, L]
    Tmax = tmax
    g = 9.81
    π = np.pi

    for j in range(2):
        V = link_lengths[j] * w ** 2  # [m^3] - bar volume
        masses[j] = ρ * V  # [kg] - bar mass

        # J = 1/6 * masses[j] * w**2
        J = 1 / 12 * masses[j] * (w ** 2 + link_lengths[j] ** 2)
        inertias[j] = J  # [kg*m^2] - Inertia tensor of bar

    if render:
        render.addRenderedBody('./cylinder.obj')  # , scale, pos, rot, color)

    """ INIT CONDITIONS """
    thetas = np.array([0.0, π / 2])
    omegas = np.array([0.0, 0.0])
    omegadots = np.array([0.0, 0.0])

    thetas_save = np.zeros([2, int(Tmax / h) + 2])
    omegas_save = np.zeros([2, int(Tmax / h) + 2])
    omegadots_save = np.zeros([2, int(Tmax / h) + 2])

    t = 0
    step = 0

    while t <= Tmax:
        t += h
        step += 1

        omegadots[1] = 2*np.sin(thetas[0]-thetas[1])*(omegas[0]**2*link_lengths[0]*(masses[0] + masses[1])
                        + g*(masses[0] + masses[1])*np.cos(thetas[0])
                        + omegas[1]**2*link_lengths[1]*masses[1]*np.cos(thetas[0] - thetas[1]))/(link_lengths[1]*(2*masses[0]
                        + masses[1] - masses[1]*np.cos(2*thetas[0] - 2*thetas[1])))
        omegadots[0] = - g * (2*masses[0] + masses[1])*np.sin(thetas[0]) - masses[1]*g*np.sin(thetas[0] - 2*thetas[1]) \
                       - 2*np.sin(thetas[0] - thetas[1])*masses[1]*(omegas[1]**2*link_lengths[1]
                       + omegas[0]**2*link_lengths[0]*np.cos(thetas[0]-thetas[1])) / (link_lengths[0]*(2*masses[0] + masses[1]
                       - masses[1]*np.cos(2*thetas[0] - 2*thetas[1])))

        #omegadots[1] = (np.cos(thetas.sum()) * masses[1] * g * (link_lengths[1] / 2)) / inertias[1]
        #omegadots[0] = (np.cos(thetas[0]) * g * link_lengths[0] * (masses[1] + masses[0] / 2)) / inertias[0]

        omegas[1] = omegas_save[1, step - 1] + omegadots[1] * h
        omegas[0] = omegas_save[0, step - 1] + omegadots[0] * h

        thetas[1] = thetas_save[1, step - 1] + omegas[1] * h
        thetas[0] = thetas_save[0, step - 1] + omegas[0] * h

        thetas_save[:, step] = thetas
        omegas_save[:, step] = omegas
        omegadots_save[:, step] = omegadots

    if render:
        render.destroy()

    return (thetas_save, omegas_save, omegadots_save)


if __name__ == '__main__':
    recursivePendulum(2.5, 1E-3)