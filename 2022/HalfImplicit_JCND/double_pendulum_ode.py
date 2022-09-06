# double pendulum ode solution: https://scipython.com/blog/the-double-pendulum/

import numpy as np
import scipy.integrate as ode


def deriv(t, y):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 4, 2
    m1, m2 = 78,39
    # The gravitational acceleration (m.s-2).
    g = 9.81
    
    theta1, p_theta1, theta2, p_theta2 = y

    M = m1 + 3*m2
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    Lr = L1 / L2
    den = 4 * M - 9 * m2 * c**2

    theta1dot = 6 / L1**2 * (2*p_theta1 - 3 * Lr * c * p_theta2) / den
    theta2dot = 6 / m2 / L2**2 * (
                    (2 * p_theta2 * M - 3 * m2 / Lr * c * p_theta1) / den)
    term = m2 * L1 * L2 / 2 * theta1dot * theta2dot * s
    p_theta1dot = -term - (m1/2 + m2) * g * L1 * np.sin(theta1)
    p_theta2dot = term - m2/2 * g * L2 * np.sin(theta2)
    
    return theta1dot, p_theta1dot, theta2dot, p_theta2dot


def refSol(tmax, dt):
    
    # Pendulum rod lengths (m) and masses (kg).
    L1, L2 = 4, 2

    t = np.arange(dt, tmax, dt)

    theta1=[]
    theta2=[]

    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = np.array([np.pi/2, 0, 0, 0])
    theta1.append(y0[0])
    theta2.append(y0[2])

    # Do the numerical integration of the equations of motion
    myODE = ode.DOP853(deriv, 0, y0, tmax, dt, dt)

    # theta1.append(y0[0])
    # theta2.append(y0[2])

    for ii, t_curr in enumerate(t):
        if ii%10000 == 0:
            print(t_curr)

        myODE.step()
        theta1.append(myODE.y[0])
        theta2.append(myODE.y[2])

    # Convert to Cartesian coordinates of the two rods.
    x1 = (L1 / 2) * np.sin(theta1)
    y1 = -(L1 / 2) * np.cos(theta1)
    x2 = 2*x1 + (L2/2) * np.sin(theta2)
    y2 = 2*y1 - (L2/2) * np.cos(theta2)
    mstack1 = np.stack([np.zeros(x1.shape),x1, y1])
    mstack2 = np.stack([np.zeros(x2.shape),x2, y2])
    mstack = np.stack([mstack1, mstack2])
    return(mstack)

