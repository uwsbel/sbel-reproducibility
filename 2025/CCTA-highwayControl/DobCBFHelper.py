import numpy as np
from scipy.optimize import minimize

def CBF_psi01_edob_weighted_vo(Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, 
                               l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x, xo, y, yo):
    """
    Compute the weighted Control Barrier Function (CBF) components psi0 and psi1 with obstacle velocity.

    Parameters:
        Iwheel, L, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1: System parameters.
        P: Weight matrix (2x2).
        l1cbf, l2cbf: CBF gains.
        omega0, ro: Angular and obstacle parameters.
        ta, tau0, v: System state variables.
        vxo, vyo: Obstacle velocity components.
        x, xo, y, yo: Position variables.
    
    Returns:
        psi0: First component of the CBF.
        psi1: Second component of the CBF.
    """
    P1_1 = P[0, 0]
    P1_2 = P[0, 1]
    P2_1 = P[1, 0]
    P2_2 = P[1, 1]

    # Trigonometric terms
    t2 = np.cos(ta)
    t3 = np.sin(ta)

    # Weighted terms
    t4 = P1_2 + P2_1
    t5 = 2.0 * P1_1 * vxo
    t6 = 2.0 * P2_2 * vyo

    # Inverse terms
    t7 = 1.0 / Iwheel
    t9 = 1.0 / Rwheel
    t11 = 1.0 / gamma

    # Position differences
    t12 = -xo
    t13 = -yo
    t20 = t12 + x
    t21 = t13 + y

    # Velocity components
    t8 = t2 * v
    t10 = t3 * v
    t18 = hdx0 + t8
    t19 = hdy0 + t10

    # Negative terms
    t14 = -t5
    t15 = -t6
    t16 = t4 * vxo
    t17 = t4 * vyo
    t22 = -t16
    t23 = -t17

    # Weighted positional terms
    t24 = P1_2 * t20
    t25 = P2_1 * t20
    t26 = P1_2 * t21
    t27 = P2_1 * t21
    t30 = 2.0 * P1_1 * t20
    t31 = 2.0 * P2_2 * t21
    t28 = 2.0 * P1_1 * t18
    t29 = 2.0 * P2_2 * t19

    # Combine terms
    t32 = t4 * t18
    t33 = t4 * t19
    t34 = t24 + t25 + t31
    t35 = t26 + t27 + t30
    t36 = l1cbf * t34
    t37 = l1cbf * t35

    # Angular and velocity terms
    t38 = t2 * t35
    t39 = t3 * t34
    t40 = t8 * t34
    t41 = t10 * t35
    t42 = -t41

    # Final combinations
    t43 = t38 + t39
    t44 = t15 + t22 + t29 + t32 + t36
    t45 = t14 + t23 + t28 + t33 + t37
    t46 = t40 + t42

    # Psi0 computation
    psi0 = (
        t43 * (hdv0 - Rwheel * gamma * t7 * (c0 + c1 * t9 * t11 * v))
        + hdta0 * t46
        + hdx1 * t35
        + hdy1 * t34
        + l2cbf * (
            t18 * t35
            + t19 * t34
            - t35 * vxo
            - t34 * vyo
            + l1cbf * (t20 * (t27 + P1_1 * t20) + t21 * (t24 + P2_2 * t21) - ro**2)
        )
        + t18 * t45
        + t19 * t44
        - t45 * vxo
        - t44 * vyo
    )

    # Psi1 computation
    psi1 = [
        (t46 * v) / L,
        Rwheel * gamma * t7 * t43 * (tau0 - (t9 * t11 * tau0 * v) / omega0),
    ]

    return psi0, psi1


def CBF_psi01_edob_weighted(Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, x, xo, y, yo):
    """
    Compute the weighted Control Barrier Function (CBF) components psi0 and psi1.

    Parameters:
        Iwheel, L, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1: System parameters.
        P: 2x2 weight matrix.
        l1cbf, l2cbf: CBF gains.
        omega0, ro: Angular and obstacle parameters.
        ta, tau0, v: Vehicle state.
        x, xo, y, yo: Position and obstacle coordinates.

    Returns:
        psi0: Scalar value.
        psi1: Array of length 2.
    """
    # Extract elements from matrix P
    P1_1 = P[0, 0]
    P1_2 = P[0, 1]
    P2_1 = P[1, 0]
    P2_2 = P[1, 1]

    # Pre-computed terms
    t2 = np.cos(ta)
    t3 = np.sin(ta)
    t4 = P1_2 + P2_1
    t5 = 1.0 / Iwheel
    t7 = 1.0 / Rwheel
    t9 = 1.0 / gamma
    t10 = -xo
    t11 = -yo
    t6 = t2 * v
    t8 = t3 * v
    t14 = t10 + x
    t15 = t11 + y
    t12 = hdx0 + t6
    t13 = hdy0 + t8
    t16 = P1_2 * t14
    t17 = P2_1 * t14
    t18 = P1_2 * t15
    t19 = P2_1 * t15
    t20 = P1_1 * t14 * 2.0
    t21 = P2_2 * t15 * 2.0
    t22 = t16 + t17 + t21
    t23 = t18 + t19 + t20
    t24 = t2 * t23
    t25 = t3 * t22
    t26 = t6 * t22
    t27 = t8 * t23
    t28 = -t27
    t29 = t24 + t25
    t30 = t26 + t28

    # Compute psi0
    psi0 = (
        t29 * (hdv0 - Rwheel * gamma * t5 * (c0 + c1 * t7 * t9 * v))
        + hdta0 * t30
        + hdx1 * t23
        + hdy1 * t22
        + l2cbf * (
            t12 * t23
            + t13 * t22
            + l1cbf * (
                t14 * (t19 + P1_1 * t14)
                + t15 * (t16 + P2_2 * t15)
                - ro**2
            )
        )
        + t12 * (P1_1 * t12 * 2.0 + l1cbf * t23 + t4 * t13)
        + t13 * (P2_2 * t13 * 2.0 + l1cbf * t22 + t4 * t12)
    )

    # Compute psi1
    psi1 = [
        (t30 * v) / L,
        Rwheel * gamma * t5 * t29 * (tau0 - (t7 * t9 * tau0 * v) / omega0)
    ]

    return psi0, psi1


def xdot_dob(x, p_dob, u, dobpara):
    """
    Compute the state derivative (dotx) and control inputs (u).
    
    Parameters:
        x: State vector for system and observer
        u: Control input (alpha, beta)
        phypara, ctrlpara, dobpara, obspara: Parameter dictionaries
        mode: Control and desired mode options
        opts: Options for the optimization solver (e.g., for quadprog)
    
    Returns:
        dotp: Observer State derivatives
    """
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    L =  4.52
    delta = 0.626671  # 35.9 degree

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    # Disturbances and reference
    # varphi, psi, distv, distheta = disturbance(t)

    # Extract state variables
    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]
    p0x, p1x, p0y, p1y = p_dob[0], p_dob[1], p_dob[2], p_dob[3] # EDOB state
    p0theta, p0v = p_dob[4], p_dob[5]

    # # Compute dx and dy
    # dx = v * np.cos(varphi + ta)
    # dy = v * np.sin(varphi + ta)

    # EDOB for dx and dy
    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    dp0x = -l0x * (v * np.cos(ta) + hdx0) + hdx1
    dp1x = -l1x * (v * np.cos(ta) + hdx0)

    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y
    dp0y = -l0y * (v * np.sin(ta) + hdy0) + hdy1
    dp1y = -l1y * (v * np.sin(ta) + hdy0)

    # DOB for theta and v
    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta

    alpha, beta = u[0], u[1]
    u1 = np.tan(beta*delta)

    omegam = v / (Rwheel * gamma)
    f_1fun = -tau0 * omegam / omega0 + tau0
    T_fun = f_1fun * alpha - c1 * omegam - c0
    # dv = T_fun * gamma / Iwheel * Rwheel + distv

    # DOB for dv and dtheta
    dp0theta = -ata * (v / L * u1 + hdta0)
    dp0v = -av * (T_fun * gamma / Iwheel * Rwheel + hdv0)

    # # Filter DOB dynamics
    # dpx = -ax * (v * np.cos(ta) + hdx)
    # dpy = -ay * (v * np.sin(ta) + hdy)
    # dhdxf = -Tf * (hdxf - hdx)
    # dhdyf = -Tf * (hdyf - hdy)

    # Collect state derivatives
    dotp = np.array([dp0x, dp1x, dp0y, dp1y, dp0theta, dp0v])
    hdxyvta = np.array([hdx0, hdy0, hdv0, hdta0])
    return dotp, hdxyvta 
import numpy as np

def CBF_psi01_edob_weighted_acc(Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, 
                                 l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x, xo, y, yo):
    """
    Compute the weighted Control Barrier Function (CBF) components psi0 and psi1 with acceleration consideration.

    Parameters:
        Iwheel, L, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1: System parameters.
        P: Weight matrix (2x2).
        l1cbf, l2cbf: CBF gains.
        omega0, ro: Angular and obstacle parameters.
        ta, tau0, v: System state variables.
        vxo, vyo: Obstacle velocity components.
        x, xo, y, yo: Position variables.
    
    Returns:
        psi0: First component of the CBF.
        psi1: Second component of the CBF.
    """
    # Extract elements from the weight matrix P
    P1_1 = P[0, 0]
    P1_2 = P[0, 1]
    P2_1 = P[1, 0]
    P2_2 = P[1, 1]

    # Pre-compute trigonometric terms
    t2 = np.cos(ta)
    t3 = np.sin(ta)

    # Pre-compute constants and terms
    t4 = ro**2
    t5 = P1_2 + P2_1
    t6 = 2.0 * P1_1 * vxo
    t7 = 2.0 * P2_2 * vyo
    t8 = 1.0 / Iwheel
    t10 = 1.0 / Rwheel
    t12 = 1.0 / gamma

    # Position differences
    t13 = -xo
    t14 = -yo
    t21 = t13 + x
    t22 = t14 + y

    # Velocity components
    t9 = t2 * v
    t11 = t3 * v
    t19 = hdx0 + t9
    t20 = hdy0 + t11

    # Additional terms
    t15 = -t6
    t16 = -t7
    t17 = t5 * vxo
    t18 = t5 * vyo
    t24 = -t17
    t25 = -t18

    # Weighted positional terms
    t27 = P1_2 * t21
    t28 = P2_1 * t21
    t29 = P1_2 * t22
    t30 = P2_1 * t22
    t33 = 2.0 * P1_1 * t21
    t34 = 2.0 * P2_2 * t22
    t31 = 2.0 * P1_1 * t19
    t32 = 2.0 * P2_2 * t20

    # Combine terms
    t35 = t5 * t19
    t36 = t5 * t20
    t37 = t27 + t28 + t34
    t38 = t29 + t30 + t33
    t39 = l1cbf * t37
    t40 = l1cbf * t38

    # Angular and velocity terms
    t41 = t2 * t38
    t42 = t3 * t37
    t43 = t9 * t37
    t44 = t11 * t38
    t45 = -t44
    t46 = t16 + t24 + t32 + t35 + t39
    t47 = t15 + t25 + t31 + t36 + t40
    t49 = -l1cbf * t4 * v**2 + t41 + t42
    t48 = t43 + t45

    # Compute psi0
    psi0 = (
        t49 * (hdv0 - Rwheel * gamma * t8 * (c0 + c1 * t10 * t12 * v))
        + hdta0 * t48
        + hdx1 * t38
        + hdy1 * t37
        + t19 * t47
        + t20 * t46
        - t47 * vxo
        - t46 * vyo
        + l2cbf * (
            t19 * t38
            + t20 * t37
            - t38 * vxo
            - t37 * vyo
            + l1cbf * (-t4 * v**2 + t21 * (t30 + P1_1 * t21) + t22 * (t27 + P2_2 * t22))
        )
    )

    # Compute psi1
    psi1 = [
        (t48 * v) / L,
        Rwheel * gamma * t8 * t49 * (tau0 - (t10 * t12 * tau0 * v) / omega0),
    ]

    return psi0, psi1
import numpy as np

def CBF_psi01_edob_weighted_acc_brake(
    Iwheel, L, P, Rwheel, brake_effect_coef, c0, c1, gamma, 
    hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, 
    ro, ta, v, vxo, vyo, x, xo, y, yo
):
    """
    Compute the weighted Control Barrier Function (CBF) components psi0 and psi1
    with acceleration and braking effects.

    Parameters:
        Iwheel, L, Rwheel, brake_effect_coef, c0, c1, gamma, hdta0, hdv0, hdx0, 
        hdx1, hdy0, hdy1, l1cbf, l2cbf: System and CBF parameters.
        P: Weight matrix (2x2).
        ro: Obstacle size parameter.
        ta: Angle (radians).
        v, vxo, vyo: Velocity and obstacle velocity components.
        x, xo, y, yo: Position variables.

    Returns:
        psi0: First component of the CBF.
        psi1: Second component of the CBF, including braking effects.
    """
    # Extract elements from the weight matrix P
    P1_1 = P[0, 0]
    P1_2 = P[0, 1]
    P2_1 = P[1, 0]
    P2_2 = P[1, 1]

    # Pre-compute trigonometric terms
    t2 = np.cos(ta)
    t3 = np.sin(ta)

    # Pre-compute constants and terms
    t4 = ro**2
    t5 = P1_2 + P2_1
    t6 = 2.0 * P1_1 * vxo
    t7 = 2.0 * P2_2 * vyo

    # Position differences
    t10 = -xo
    t11 = -yo
    t18 = t10 + x
    t19 = t11 + y

    # Velocity components
    t8 = t2 * v
    t9 = t3 * v
    t16 = hdx0 + t8
    t17 = hdy0 + t9

    # Additional terms
    t12 = -t6
    t13 = -t7
    t14 = t5 * vxo
    t15 = t5 * vyo
    t21 = -t14
    t22 = -t15

    # Weighted positional terms
    t24 = P1_2 * t18
    t25 = P2_1 * t18
    t26 = P1_2 * t19
    t27 = P2_1 * t19
    t30 = 2.0 * P1_1 * t18
    t31 = 2.0 * P2_2 * t19
    t28 = 2.0 * P1_1 * t16
    t29 = 2.0 * P2_2 * t17

    # Combine terms
    t32 = t5 * t16
    t33 = t5 * t17
    t34 = t24 + t25 + t31
    t35 = t26 + t27 + t30
    t36 = l1cbf * t34
    t37 = l1cbf * t35

    # Angular and velocity terms
    t38 = t2 * t35
    t39 = t3 * t34
    t40 = t8 * t34
    t41 = t9 * t35
    t42 = -t41
    t43 = t13 + t21 + t29 + t32 + t36
    t44 = t12 + t22 + t28 + t33 + t37
    t46 = -l1cbf * t4 * v**2 + t38 + t39
    t45 = t40 + t42

    # Compute psi0
    psi0 = (
        hdta0 * t45
        + hdx1 * t35
        + hdy1 * t34
        + t16 * t44
        + t17 * t43
        - t44 * vxo
        - t43 * vyo
        + l2cbf * (
            t16 * t35
            + t17 * t34
            - t35 * vxo
            - t34 * vyo
            + l1cbf * (-t4 * v**2 + t18 * (t27 + P1_1 * t18) + t19 * (t24 + P2_2 * t19))
        )
        + t46 * (hdv0 - (Rwheel * gamma * (c0 + (c1 * v) / (Rwheel * gamma))) / Iwheel)
    )

    # Compute psi1 (includes braking effect)
    psi1 = [
        (t45 * v) / L,
        brake_effect_coef * t46 * v
    ]

    return psi0, psi1


def CBF_psi01_edob_weighted_acc_poly_new(hdv0, l2cbf, ro, v, vxo, x, xo):
    """
    CBF_psi01_edob_weighted_acc_poly_new
    Translated from MATLAB to Python.

    Parameters:
        hdv0 : float
        l2cbf : float
        ro : float
        v : float
        vxo : float
        x : float
        xo : float
    
    Returns:
        psi0 : float
        psi1 : float (if applicable)
    """
    t2 = ro ** 2
    t3 = v ** 2
    t4 = x * 2.0
    t5 = xo * 2.0
    t6 = -t5
    t7 = t4 + t6

    psi0 = (
        t7 * v
        - t7 * vxo
        + l2cbf * ((x - xo) ** 2 - t2 * t3)
        - t2 * v * (hdv0 - t3 * 2.39238426e-2 + v * 1.235469 - 1.657656527123383e+1) * 2.0
    )

    psi1 = None
    if hdv0 is not None:  # If nargout > 1 in MATLAB
        psi1 = (
            t2
            * v
            * (
                t3 * 2.1761362e-2
                - v * 1.3437261
                + v ** 3 * 1.59919731e-4
                + 2.158635e+1
            )
            * -2.0
        )

    return psi0, psi1

def CBF_psi01_edob_weighted_acc_poly(
    L, P, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, 
    l1cbf, l2cbf, ro, ta, v, vxo, vyo, x, xo, y, yo
):
    """
    Compute the weighted Control Barrier Function (CBF) components psi0 and psi1
    with polynomial weighting.

    Parameters:
        L: Characteristic length.
        P: Weight matrix (2x2).
        hdta0, hdv0, hdx0, hdx1, hdy0, hdy1: Higher derivatives of the state variables.
        l1cbf, l2cbf: Control barrier function parameters.
        ro: Obstacle size parameter.
        ta: Angle (radians).
        v, vxo, vyo: Velocity and obstacle velocity components.
        x, xo, y, yo: Position variables.

    Returns:
        psi0: First component of the CBF.
        psi1: Second component of the CBF with polynomial terms.
    """
    # Extract elements from the weight matrix P
    P1_1 = P[0, 0]
    P1_2 = P[0, 1]
    P2_1 = P[1, 0]
    P2_2 = P[1, 1]

    # Pre-compute trigonometric terms
    t2 = np.cos(ta)
    t3 = np.sin(ta)

    # Pre-compute constants and terms
    t4 = ro**2
    t5 = v**2
    t6 = P1_2 + P2_1
    t7 = 2.0 * P1_1 * vxo
    t8 = 2.0 * P2_2 * vyo

    # Position differences
    t11 = -xo
    t12 = -yo
    t19 = t11 + x
    t20 = t12 + y

    # Velocity components
    t9 = t2 * v
    t10 = t3 * v
    t17 = hdx0 + t9
    t18 = hdy0 + t10

    # Additional terms
    t13 = -t7
    t14 = -t8
    t15 = t6 * vxo
    t16 = t6 * vyo
    t22 = -t15
    t23 = -t16

    # Weighted positional terms
    t24 = -l1cbf * t4 * v * 2.0
    t25 = P1_2 * t19
    t26 = P2_1 * t19
    t27 = P1_2 * t20
    t28 = P2_1 * t20
    t31 = 2.0 * P1_1 * t19
    t32 = 2.0 * P2_2 * t20
    t29 = 2.0 * P1_1 * t17
    t30 = 2.0 * P2_2 * t18

    # Combine terms
    t33 = t6 * t17
    t34 = t6 * t18
    t35 = t25 + t26 + t32
    t36 = t27 + t28 + t31
    t37 = l1cbf * t35
    t38 = l1cbf * t36

    # Angular and velocity terms
    t39 = t2 * t36
    t40 = t3 * t35
    t41 = t9 * t35
    t42 = t10 * t36
    t43 = -t42
    t44 = t14 + t22 + t30 + t33 + t37
    t45 = t13 + t23 + t29 + t34 + t38
    t47 = t24 + t39 + t40
    t46 = t41 + t43

    # Compute psi0
    psi0 = (
        hdta0 * t46
        + hdx1 * t36
        + hdy1 * t35
        + t17 * t45
        + t18 * t44
        - t45 * vxo
        - t44 * vyo
        + t47 * (hdv0 - t5 * 2.39238426e-2 + v * 1.235469)
        + l2cbf * (
            l1cbf * (-t4 * t5 + t19 * (t28 + P1_1 * t19) + t20 * (t25 + P2_2 * t20))
            + t17 * t36 + t18 * t35 - t36 * vxo - t35 * vyo
        )
    )

    # Compute psi1 with polynomial terms
    psi1 = [
        (t46 * v) / L,
        t47 * (t5 * 2.1761362e-2 - v * 1.3437261 + v**3 * 1.59919731e-4 + 2.158635e+1)
    ]

    return psi0, psi1

def DOBCBF(x, ud, p_dob, ctrlpara, dobpara, weight1, weight2, SV_pos,v_SV, ro):
    # ud = (alpha, beta)
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    # L =  4.52

    L = 2.5
    delta = 0.626671  # 35.9 degree

    # Control parameters
    l1cbf = ctrlpara['l1cbf']
    l2cbf = ctrlpara['l2cbf']

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]

    Rmatt = np.array([[np.cos(ta), np.sin(ta)],[-np.sin(ta), np.cos(ta)]])
    Q = np.array([[weight1, 0.],[0.,weight2]])
    P = Rmatt.T @ Q @ Rmatt
    
    [p0x, p1x, p0y, p1y, p0theta, p0v] = p_dob

    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y

    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta

    hdta0 = 0 
    # hdv0 = 0 
    hdx0 = 0 
    hdx1 = 0
    hdy0 = 0
    hdy1 = 0 # uncomment them if no dob

    # Setting obstacles

    ind_sv = []
    # r_cons = 2 * ro
    for i in range(SV_pos.shape[1]):
        dd = (x_state - SV_pos[0,i])**2 + (y - SV_pos[1,i])**2
        if dd <= 2 * ro * ro / weight1:
            ind_sv.append(i)
    Aineq = []
    bineq = []
    for i in ind_sv:
    # for i in range(SV_pos.shape[1]):
        xo = SV_pos[0,i]
        yo = SV_pos[1,i]
        vxo = v_SV[0,i]
        vyo = v_SV[1,i]
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, x_state, xo, y, yo
        #     )
        psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc_poly_new(hdv0, l2cbf, ro, v, vxo, x_state, xo)
        psi_cbf1 = [0, psi_cbf1]
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_vo(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
        #     )
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
        #     )
        Aineq.append([-i for i in psi_cbf1])
        bineq.append(psi_cbf0)

    steering_max = 0.15
    u1_lim = np.tan(steering_max*delta)

    udd = np.array([np.tan(ud[1]*delta), ud[0]])

    Hcbf = 2 * np.eye(2)
    fcbf = -2 * udd
    # Aineq = -psi_cbf1
    # bineq = psi_cbf0
    Aineq = np.array(Aineq)
    bineq = np.array(bineq)
    # print(Aineq.shape)
    # print(bineq.shape)

    # Solve the QP problem
    def qp_objective(u):
        return 0.5 * u.T @ Hcbf @ u + fcbf.T @ u
    if Aineq.shape[0] > 0:
        Aineq = np.vstack((Aineq, np.array([[1., 0.], [-1., 0.]])))
        bineq = np.hstack((bineq, np.array([u1_lim, u1_lim])))

        constraints = [{'type': 'ineq', 'fun': lambda u: bineq - Aineq @ u}]
        result = minimize(qp_objective, udd, constraints=constraints)
        u = result.x
        if not result.success:
            print('#################    Infeasible!!!   ##################')
            alpha, beta = ud
        else:
            u1, u2 = u[0], u[1]
            alpha = u2
            beta = np.arctan(u1) / delta
    else:
        alpha, beta = ud



    return (alpha, beta)


def CBF_psi01_dob_weighted_dbeta_poly(L, beta, hdv0, l1cbf, l2cbf, ro, ta, v, vxo, vyo, weight1, weight2, x, xo, y, yo):
    """
    CBF_psi01_dob_weighted_dbeta_poly
    Translated from MATLAB to Python.

    Parameters:
        L, beta, hdv0, l1cbf, l2cbf, ro, ta, v, vxo, vyo, weight1, weight2, x, xo, y, yo : float
    
    Returns:
        psi0 : float
        psi1 : list (if applicable)
    """
    t2 = np.tan(beta)
    t3 = np.cos(ta)
    t4 = np.sin(ta)
    t5 = v ** 2
    t8 = 1.0 / L
    t9 = -xo
    t10 = -yo
    t6 = t3 ** 2
    t7 = t4 ** 2
    t15 = t3 * t4 * weight1
    t16 = t3 * t4 * weight2
    t17 = t9 + x
    t18 = t10 + y
    t11 = t6 * weight1
    t12 = t6 * weight2
    t13 = t7 * weight1
    t14 = t7 * weight2
    t23 = t15 * 2.0
    t24 = t16 * 2.0
    t27 = -t16
    t29 = t3 * t17
    t30 = t3 * t18
    t31 = t4 * t17
    t32 = t4 * t18
    t19 = t11 * 2.0
    t20 = t12 * 2.0
    t21 = t13 * 2.0
    t22 = t14 * 2.0
    t25 = -t12
    t26 = -t13
    t28 = -t24
    t33 = -t31
    t34 = t11 + t14
    t35 = t12 + t13
    t38 = t15 + t27
    t45 = t29 + t32
    t36 = t19 + t22
    t37 = t20 + t21
    t39 = t23 + t28
    t53 = t17 * t34
    t54 = t18 * t35
    t55 = t30 + t33
    t56 = t17 * t38
    t57 = t18 * t38
    t60 = t3 * t45 * weight1
    t61 = t3 * t45 * weight2
    t62 = t4 * t45 * weight1
    t63 = t4 * t45 * weight2
    t74 = t25 + t26 + t34
    t40 = t36 * vxo
    t41 = t37 * vyo
    t42 = t39 * vxo
    t43 = t39 * vyo
    t44 = t3 * t36 * v
    t46 = t4 * t37 * v
    t49 = t3 * t39 * v
    t50 = t4 * t39 * v
    t58 = t17 * t39
    t59 = t18 * t39
    t64 = t3 * t55 * weight1
    t65 = t3 * t55 * weight2
    t66 = -t61
    t67 = t4 * t55 * weight1
    t69 = t4 * t55 * weight2
    t70 = -t63
    t75 = t17 * t74
    t76 = t18 * t74
    t48 = -t41
    t51 = -t42
    t71 = -t64
    t73 = -t69
    t77 = -t76
    t78 = t54 + t56 + t62 + t65
    t79 = l1cbf * t78
    t80 = t53 + t57 + t60 + t73
    t81 = t4 * t78
    t84 = t60 + t66 + t67 + t73
    t89 = t58 + t62 + t65 + t70 + t71 + t77
    t82 = l1cbf * t80
    t83 = t3 * t80
    t87 = t18 * t84
    t88 = t59 + t75 + t84
    t91 = t2 * t8 * t89 * v
    t90 = t2 * t8 * t88 * v
    t94 = t2 * t8 * (t87 - t17 * (t62 + t65 + t70 + t71))
    t95 = t46 + t48 + t49 + t51 + t79 + t90
    t97 = t81 + t83 + t94

    et1 = (
        vxo * (t40 + t43 - t44 - t50 - t82 + t91)
        + l2cbf * (
            t81 * v
            + t83 * v
            + t94 * v
            - t80 * vxo
            - t78 * vyo
            + l1cbf * (t18 * (t62 + t65) + t17 * (t60 + t73) - ro ** 2)
        )
        + t97 * (hdv0 - t5 * 2.39238426e-2 + v * 1.235469 - 1.657656527123383e+1)
        - t95 * vyo
        - t3 * v * (t40 + t43 - t44 - t50 - t82 + t91)
        + t4 * t95 * v
    )

    et2 = (
        t2
        * t8
        * v
        * (
            t89 * vxo
            - t88 * vyo
            + l1cbf * (t87 - t17 * (t62 + t65 + t70 + t71))
            + t3 * t78 * v
            - t4 * t80 * v
            - t3 * t89 * v
            + t4 * t88 * v
            - t2
            * t8
            * v
            * (
                t18 * (t62 * 2.0 - t63 * 2.0 - t64 * 2.0 + t65 * 2.0)
                + t17 * (t60 * 2.0 - t61 * 2.0 + t67 * 2.0 - t69 * 2.0)
            )
        )
    )

    psi0 = et1 + et2

    psi1 = None
    if hdv0 is not None:  # If nargout > 1 in MATLAB
        psi1 = [
            t8 * v * (t2 ** 2 + 1.0) * (t87 - t17 * (t62 + t65 + t70 + t71)),
            t97 * (t5 * 2.1761362e-2 - v * 1.3437261 + v ** 3 * 1.59919731e-4 + 2.158635e+1),
        ]

    return psi0, psi1
import numpy as np

def cbf_psi01_dob_weighted_dbeta_poly_bias(L, beta, bias, hdv0, l1cbf, l2cbf, ro, ta, v, vxo, vyo, weight1, weight2, x, xo, y, yo):
    t2 = np.tan(beta)
    t3 = np.cos(ta)
    t4 = np.sin(ta)
    t5 = v**2
    t8 = 1.0 / L
    t9 = -xo
    t10 = -yo
    t6 = t3**2
    t7 = t4**2
    t11 = t9 + x
    t12 = t10 + y
    t17 = t3 * t4 * weight1 * 2.0
    t18 = t3 * t4 * weight2 * 2.0
    t13 = t6 * weight1 * 2.0
    t14 = t6 * weight2 * 2.0
    t15 = t7 * weight1 * 2.0
    t16 = t7 * weight2 * 2.0
    t19 = -t18
    t20 = t3 * t11
    t21 = t3 * t12
    t22 = t4 * t11
    t23 = t4 * t12
    t24 = -t22
    t25 = t13 + t16
    t26 = t14 + t15
    t27 = t17 + t19
    t33 = t20 + t23
    t28 = t25 * vxo
    t29 = t26 * vyo
    t30 = t27 * vxo
    t31 = t27 * vyo
    t32 = t3 * t25 * v
    t34 = t4 * t26 * v
    t36 = t3 * t27 * v
    t37 = t4 * t27 * v
    t39 = bias + t33
    t41 = t21 + t24
    t44 = t3 * t33 * weight2 * 2.0
    t45 = t4 * t33 * weight2 * 2.0
    t35 = -t29
    t38 = -t30
    t40 = -t32
    t42 = -t37
    t43 = t41**2
    t46 = t3 * t39 * weight1 * 2.0
    t47 = t4 * t39 * weight1 * 2.0
    t50 = t3 * t41 * weight1 * 2.0
    t51 = t3 * t41 * weight2 * 2.0
    t52 = t4 * t41 * weight1 * 2.0
    t53 = t4 * t41 * weight2 * 2.0
    t56 = t33 * t41 * weight2 * 2.0
    t57 = t39 * t41 * weight1 * 2.0
    t55 = -t53
    t58 = -t57
    t59 = t47 + t51
    t71 = -t2 * t8 * v * (t44 - t46 - t52 + t53)
    t60 = t46 + t55
    t61 = l1cbf * t59
    t63 = t4 * t59
    t66 = t56 + t58
    t72 = -t2 * t8 * v * (t45 + t50 - t59)
    t62 = l1cbf * t60
    t64 = t3 * t60
    t67 = t2 * t8 * t66
    t74 = t34 + t35 + t36 + t38 + t61 + t71
    t65 = -t62
    t68 = -t67
    t73 = t63 + t64 + t68
    t75 = t28 + t31 + t40 + t42 + t65 + t72
    et1 = (t73 * (hdv0 - t5 * 2.39238426e-2 + v * 1.235469 - 1.657656527123383e+1) + t75 * vxo - t74 * vyo +
          l2cbf * (t63 * v + t64 * v + t68 * v - t60 * vxo - t59 * vyo + l1cbf * (t43 * weight2 + t39**2 * weight1 - ro**2)) -
          t3 * t75 * v + t4 * t74 * v)
    et2 = (-t2 * t8 * v * (l1cbf * t66 - vyo * (t44 - t46 - t52 + t53) + vxo * (t45 + t50 - t59) +
                          t4 * v * (t44 - t46 - t52 + t53) - t3 * v * (t45 + t50 - t59) -
                          t3 * t59 * v + t4 * t60 * v - t2 * t8 * v * (t43 * weight1 * 2.0 - t43 * weight2 * 2.0 +
                                                                      t33**2 * weight2 * 2.0 - t33 * t39 * weight1 * 2.0)))
    psi0 = et1 + et2
    
    psi1 = None
    if psi0 is not None:
        psi1 = [-t8 * t66 * v * (t2**2 + 1.0),
                         t73 * (t5 * 2.1761362e-2 - v * 1.3437261 + v**3 * 1.59919731e-4 + 2.158635e+1)]
    
    return psi0, psi1


def DOBCBF_dbeta(x, ud, upre, p_dob, ctrlpara, dobpara, weight1, weight2, SV_pos,v_SV, ro, r_precpt, bias, dob_mode=False, cbfflavor='balance'):
    # ud = (alpha, dbeta)
    # Physical parameters

    L =  4.52
    delta = 0.626671  # 35.9 degree

    # Control parameters
    l1cbf = ctrlpara['l1cbf']
    l2cbf = ctrlpara['l2cbf']
    kblf = ctrlpara['kblf']

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    x_state = x[0]
    y = x[1]
    ta = x[2]
    beta = x[4]
    v = x[3]

    Rmatt = np.array([[np.cos(ta), np.sin(ta)],[-np.sin(ta), np.cos(ta)]])
    Q = np.array([[weight1, 0.],[0.,weight2]])
    P = Rmatt.T @ Q @ Rmatt
    
    p0v = p_dob



    hdv0 = p0v + av * v

    if not dob_mode:
        hdv0 = 0 


    # Setting obstacles

    ind_sv = []
    # r_cons = 2 * ro
    for i in range(SV_pos.shape[1]):
        dd = (x_state - SV_pos[0,i])**2 + (y - SV_pos[1,i])**2
        diffx = np.array([SV_pos[0,i]- x_state, SV_pos[1,i]- y])
        re_diffx = Rmatt @ diffx
        # if (dd <= 2 * r_precpt * r_precpt) and (re_diffx[0] >= -0.1 * r_precpt):
        if (dd <=  r_precpt * r_precpt):
            ind_sv.append(i)
    # print(ind_sv)


    Aineq = []
    bineq = []
    for i in ind_sv:
    # for i in range(SV_pos.shape[1]):
        xo = SV_pos[0,i]
        yo = SV_pos[1,i]
        vxo = v_SV[0,i]
        vyo = v_SV[1,i]

        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc_poly_new(hdv0, l2cbf, ro, v, vxo, x_state, xo)
        # psi_cbf0, psi_cbf1 = CBF_psi01_dob_weighted_dbeta_poly(L, beta, hdv0, l1cbf, l2cbf, ro, ta, v, vxo, vyo, weight1, weight2, x_state, xo, y, yo)
        psi_cbf0, psi_cbf1 = cbf_psi01_dob_weighted_dbeta_poly_bias(L, beta, bias, hdv0, l1cbf, l2cbf, ro, ta, v, vxo, vyo, weight1, weight2, x_state, xo, y, yo)
        # psi_cbf1 = [0, psi_cbf1]
        Aineq.append([-i for i in psi_cbf1])
        bineq.append(psi_cbf0)

    # steering_max = 0.15
    # u1_lim = np.tan(steering_max*delta)

    # udd = np.array([np.tan(ud[1]*delta), ud[0]])
    udd = np.array([ud[1], ud[0]]) # udd: (dbeta, alpha)

    if cbfflavor == 'balance':
        #### USE THIS FOR STRAIGHT LANE
        Hcbf = 2 * np.eye(2)
        fcbf = -2 * udd
        # Uweight = np.array([[10.,0.],[0.,1.]])
        # Hcbf = Uweight
        # fcbf = -2 * Uweight.T @ udd
    elif cbfflavor == 'throttleonly':
        ### USE THIS FOR COMPLEX LANE
        Uweight = np.array([[500.,0.],[0.,1.]])
        Hcbf = Uweight
        fcbf = -2 * Uweight.T @ udd
    else:
        Hcbf = 2 * np.eye(2)
        fcbf = -2 * udd

    Aineq = np.array(Aineq)
    bineq = np.array(bineq)
    # print(Aineq.shape)
    # print(bineq.shape)

    # Solve the QP problem
    def qp_objective(u):
        return 0.5 * u.T @ Hcbf @ u + fcbf.T @ u
    if Aineq.shape[0] > 0:
        # Aineq = np.vstack((Aineq, np.array([[1., 0.], [-1., 0.]])))
        # bineq = np.hstack((bineq, np.array([u1_lim, u1_lim])))

        constraints = [{'type': 'ineq', 'fun': lambda u: bineq - Aineq @ u}]
        result = minimize(qp_objective, udd, constraints=constraints)
        u = result.x
        if not result.success:
            print('#################    Infeasible!!!   ##################')
            alpha = upre[0]
            dbeta = 0.
            print(f'pre_alpha = {alpha}, pre_dbeta = {dbeta}')
            # alpha, beta = ud
        else:
            # print('********   Solved!!!   *******')
            u1, u2 = u[0], u[1]
            alpha = u2
            dbeta = u1
        flagcbf = True
    else:
        alpha, dbeta = ud
        flagcbf = False


    return (alpha, dbeta), flagcbf

def DOBCBF_ACC(x, ud, upre, p_dob, ctrlpara, dobpara, weight1, weight2, SV_pos,v_SV, ro, r_precpt, dob_mode=False):
    # ud = (alpha, beta)
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    L =  4.52
    delta = 0.626671  # 35.9 degree

    # Control parameters
    l1cbf = ctrlpara['l1cbf']
    l2cbf = ctrlpara['l2cbf']

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]

    Rmatt = np.array([[np.cos(ta), np.sin(ta)],[-np.sin(ta), np.cos(ta)]])
    Q = np.array([[weight1, 0.],[0.,weight2]])
    P = Rmatt.T @ Q @ Rmatt
    
    [p0x, p1x, p0y, p1y, p0theta, p0v] = p_dob

    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y

    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta
    if dob_mode:
        hdta0 = 0 
        # hdv0 = 0 
        hdx0 = 0 
        hdx1 = 0
        hdy0 = 0
        hdy1 = 0 # uncomment them if no dob
    else:
        hdta0 = 0 
        hdv0 = 0 
        hdx0 = 0 
        hdx1 = 0
        hdy0 = 0
        hdy1 = 0 # uncomment them if no dob

    # Setting obstacles

    ind_sv = []
    # r_cons = 2 * ro
    for i in range(SV_pos.shape[1]):
        dd = (x_state - SV_pos[0,i])**2 + (y - SV_pos[1,i])**2
        if dd <= 2 * r_precpt * r_precpt / weight1:
            ind_sv.append(i)
    Aineq = []
    bineq = []
    for i in ind_sv:
    # for i in range(SV_pos.shape[1]):
        xo = SV_pos[0,i]
        yo = SV_pos[1,i]
        vxo = v_SV[0,i]
        vyo = v_SV[1,i]
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, x_state, xo, y, yo
        #     )
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_vo(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
        #     )
        psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc_poly_new(hdv0, l2cbf, ro, v, vxo, x_state, xo)
        psi_cbf1 = [0, psi_cbf1]
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc_poly(
        #         L, P, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, 
        #         l1cbf, l2cbf, ro, ta, v, vxo, vyo, x_state, xo, y, yo
        #     )
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
        #     )
        Aineq.append([-i for i in psi_cbf1])
        bineq.append(psi_cbf0)

    steering_max = 0.15
    u1_lim = np.tan(steering_max*delta)

    udd = np.array([np.tan(ud[1]*delta), ud[0]])

    Hcbf = 2 * np.eye(2)
    fcbf = -2 * udd
    # Aineq = -psi_cbf1
    # bineq = psi_cbf0
    Aineq = np.array(Aineq)
    bineq = np.array(bineq)
    # print(Aineq.shape)
    # print(bineq.shape)

    # Solve the QP problem
    def qp_objective(u):
        return 0.5 * u.T @ Hcbf @ u + fcbf.T @ u
    if Aineq.shape[0] > 0:
        Aineq = np.vstack((Aineq, np.array([[1., 0.], [-1., 0.]])))
        bineq = np.hstack((bineq, np.array([u1_lim, u1_lim])))

        constraints = [{'type': 'ineq', 'fun': lambda u: bineq - Aineq @ u}]
        result = minimize(qp_objective, udd, constraints=constraints)
        u = result.x
        if not result.success:
            print('#################    Infeasible!!!   ##################')
            alpha, beta = upre
            # alpha, beta = ud
        else:
            # print('********   Solved!!!   *******')
            u1, u2 = u[0], u[1]
            alpha = u2
            beta = np.arctan(u1) / delta
    else:
        alpha, beta = ud



    return (alpha, beta)


def DOBCBF_ACC_switch(x, ud, p_dob, ctrlpara, dobpara, weight1, weight2, SV_pos,v_SV, ro, dob_mode=False):
    # ud = (alpha, beta)
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    L =  4.52
    delta = 0.626671  # 35.9 degree
    brake_effect_coef = 0.9

    # Control parameters
    l1cbf = ctrlpara['l1cbf']
    l2cbf = ctrlpara['l2cbf']

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]

    Rmatt = np.array([[np.cos(ta), np.sin(ta)],[-np.sin(ta), np.cos(ta)]])
    Q = np.array([[weight1, 0.],[0.,weight2]])
    P = Rmatt.T @ Q @ Rmatt
    
    [p0x, p1x, p0y, p1y, p0theta, p0v] = p_dob

    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y

    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta
    if dob_mode:
        hdta0 = 0 
        # hdv0 = 0 
        hdx0 = 0 
        hdx1 = 0
        hdy0 = 0
        hdy1 = 0 # uncomment them if no dob
    else:
        hdta0 = 0 
        hdv0 = 0 
        hdx0 = 0 
        hdx1 = 0
        hdy0 = 0
        hdy1 = 0 # uncomment them if no dob
    # Setting obstacles

    ind_sv = []
    # r_cons = 2 * ro
    for i in range(SV_pos.shape[1]):
        dd = (x_state - SV_pos[0,i])**2 + (y - SV_pos[1,i])**2
        if dd <= 2 * ro * ro / weight1:
            ind_sv.append(i)
    Aineq = []
    bineq = []
    Aineq_brake = []
    bineq_brake = []
    # for i in ind_sv:
    for i in range(SV_pos.shape[1]):
        xo = SV_pos[0,i]
        yo = SV_pos[1,i]
        vxo = v_SV[0,i]
        vyo = v_SV[1,i]
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, x_state, xo, y, yo
        #     )
        # psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_vo(
        #     Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
        #     )
        psi_cbf0, psi_cbf1 = CBF_psi01_edob_weighted_acc(
            Iwheel, L, P, Rwheel, c0, c1, gamma, hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, omega0, ro, ta, tau0, v, vxo, vyo, x_state, xo, y, yo
            ) # CBF for accelerating
        psi_cbf0_brake, psi_cbf1_brake = CBF_psi01_edob_weighted_acc_brake(
                                    Iwheel, L, P, Rwheel, brake_effect_coef, c0, c1, gamma, 
                                    hdta0, hdv0, hdx0, hdx1, hdy0, hdy1, l1cbf, l2cbf, 
                                    ro, ta, v, vxo, vyo, x_state, xo, y, yo
                                ) # CBF for braking
        Aineq.append([-i for i in psi_cbf1])
        bineq.append(psi_cbf0)
        Aineq_brake.append([-i for i in psi_cbf1_brake])
        bineq_brake.append(psi_cbf0_brake)

    steering_max = 0.3
    u1_lim = np.tan(steering_max*delta)

    udd = np.array([np.tan(ud[1]*delta), ud[0]])

    Hcbf = 2 * np.eye(2)
    fcbf = -2 * udd
    # Aineq = -psi_cbf1
    # bineq = psi_cbf0
    Aineq = np.array(Aineq)
    bineq = np.array(bineq)
    Aineq_brake = np.array(Aineq_brake)
    bineq_brake = np.array(bineq_brake)
    # print(Aineq.shape)
    # print(bineq.shape)

    # Solve the QP problem
    def qp_objective(u):
        return 0.5 * u.T @ Hcbf @ u + fcbf.T @ u
    if Aineq.shape[0] > 0:
        Aineq = np.vstack((Aineq, np.array([[1., 0.], [-1., 0.]])))
        bineq = np.hstack((bineq, np.array([u1_lim, u1_lim])))
        Aineq = np.vstack((Aineq, np.array([[0., 1.], [0., -1.]])))
        bineq = np.hstack((bineq, np.array([1., 1.])))

        Aineq_brake = np.vstack((Aineq_brake, np.array([[1., 0.], [-1., 0.]])))
        bineq_brake = np.hstack((bineq_brake, np.array([u1_lim, u1_lim])))
        Aineq_brake = np.vstack((Aineq_brake, np.array([[0., 1.], [0., -1.]])))
        bineq_brake = np.hstack((bineq_brake, np.array([1., 1.])))

        # first solve acceleration CBF
        constraints = [{'type': 'ineq', 'fun': lambda u: bineq - Aineq @ u}]
        result = minimize(qp_objective, udd, constraints=constraints)
        u = result.x
        flag_acc = result.success

        if not result.success or u[1] < 0:
            # print('#################    Infeasible!!!   ##################')
            # Then try braking CBF
            constraints = [{'type': 'ineq', 'fun': lambda u: bineq_brake - Aineq_brake @ u}]
            result = minimize(qp_objective, udd, constraints=constraints)
            u = result.x
            flag_brake = result.success
            if not result.success or u[1] > 0:
                print('#################    Infeasible!!!   ##################')
                alpha, beta = ud
            else:
                u1, u2 = u[0], u[1]
                alpha = u2
                beta = np.arctan(u1) / delta
        else:
            u1, u2 = u[0], u[1]
            alpha = u2
            beta = np.arctan(u1) / delta
    else:
        alpha, beta = ud



    return (alpha, beta)
def xdot_dob_dbeta(x, p_dob, u, dobpara):
    """
    Compute the state derivative (dotx) and control inputs (u).
    
    Parameters:
        x: State vector for system and observer
        u: Control input (alpha, beta)
        phypara, ctrlpara, dobpara, obspara: Parameter dictionaries
        mode: Control and desired mode options
        opts: Options for the optimization solver (e.g., for quadprog)
    
    Returns:
        dotp: Observer State derivatives
    """
    # Physical parameters
    L =  4.52
    delta = 0.626671  # 35.9 degree

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']


    # Disturbances and reference
    # varphi, psi, distv, distheta = disturbance(t)

    # Extract state variables
    x_state = x[0]
    y = x[1]
    ta = x[2]
    beta = x[4]
    v = x[3]

    p0v = p_dob

    hdv0 = p0v + av * v

    alpha, dbeta = u[0], u[1]
    dp0v = -av * ( -16.576565271233825 + 1.235469*v + (-2.39238426e-02)*v**2 + 2.15863500e+01*alpha + (-1.34372610)*v*alpha + (2.17613620e-02)*alpha*v**2 + 1.59919731e-04*alpha*v**3 + hdv0)

    # # Filter DOB dynamics
    # dpx = -ax * (v * np.cos(ta) + hdx)
    # dpy = -ay * (v * np.sin(ta) + hdy)
    # dhdxf = -Tf * (hdxf - hdx)
    # dhdyf = -Tf * (hdyf - hdy)

    # Collect state derivatives
    return dp0v, hdv0 

def xdot_dob_poly(x, p_dob, u, dobpara):
    """
    Compute the state derivative (dotx) and control inputs (u).
    
    Parameters:
        x: State vector for system and observer
        u: Control input (alpha, beta)
        phypara, ctrlpara, dobpara, obspara: Parameter dictionaries
        mode: Control and desired mode options
        opts: Options for the optimization solver (e.g., for quadprog)
    
    Returns:
        dotp: Observer State derivatives
    """
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    L =  4.52
    delta = 0.626671  # 35.9 degree
    brake_effect_coef = 0.9

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    # Disturbances and reference
    # varphi, psi, distv, distheta = disturbance(t)

    # Extract state variables
    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]
    p0x, p1x, p0y, p1y = p_dob[0], p_dob[1], p_dob[2], p_dob[3] # EDOB state
    p0theta, p0v = p_dob[4], p_dob[5]

    # # Compute dx and dy
    # dx = v * np.cos(varphi + ta)
    # dy = v * np.sin(varphi + ta)

    # EDOB for dx and dy
    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    dp0x = -l0x * (v * np.cos(ta) + hdx0) + hdx1
    dp1x = -l1x * (v * np.cos(ta) + hdx0)

    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y
    dp0y = -l0y * (v * np.sin(ta) + hdy0) + hdy1
    dp1y = -l1y * (v * np.sin(ta) + hdy0)

    # DOB for theta and v
    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta

    alpha, beta = u[0], u[1]
    u1 = np.tan(beta*delta)
    # if alpha > 0:
    #     omegam = v / (Rwheel * gamma)
    #     f_1fun = -tau0 * omegam / omega0 + tau0
    #     T_fun = f_1fun * alpha - c1 * omegam - c0
    #     # dv = T_fun * gamma / Iwheel * Rwheel + distv

    #     # DOB for dv and dtheta
    #     dp0theta = -ata * (v / L * u1 + hdta0)
    #     dp0v = -av * (T_fun * gamma / Iwheel * Rwheel + hdv0)
    # else:
    #     dp0theta = -ata * (v / L * u1 + hdta0)
    #     dp0v = -av * ( v * brake_effect_coef * alpha + hdv0)

    dp0theta = -ata * (v / L * u1 + hdta0)
    dp0v = -av * ( -16.576565271233825 + 1.235469*v + (-2.39238426e-02)*v**2 + 2.15863500e+01*alpha + (-1.34372610)*v*alpha + (2.17613620e-02)*alpha*v**2 + 1.59919731e-04*alpha*v**3 + hdv0)

    # # Filter DOB dynamics
    # dpx = -ax * (v * np.cos(ta) + hdx)
    # dpy = -ay * (v * np.sin(ta) + hdy)
    # dhdxf = -Tf * (hdxf - hdx)
    # dhdyf = -Tf * (hdyf - hdy)

    # Collect state derivatives
    dotp = np.array([dp0x, dp1x, dp0y, dp1y, dp0theta, dp0v])
    hdxyvta = np.array([hdx0, hdy0, hdv0, hdta0])
    return dotp, hdxyvta 
def xdot_dob_switch(x, p_dob, u, dobpara):
    """
    Compute the state derivative (dotx) and control inputs (u).
    
    Parameters:
        x: State vector for system and observer
        u: Control input (alpha, beta)
        phypara, ctrlpara, dobpara, obspara: Parameter dictionaries
        mode: Control and desired mode options
        opts: Options for the optimization solver (e.g., for quadprog)
    
    Returns:
        dotp: Observer State derivatives
    """
    # Physical parameters
    tau0 = 100.
    omega0 = 1200.
    c0 = 0.01
    c1 = 0.02
    Rwheel = 0.3
    gamma = 1/3
    Iwheel = 0.6
    L =  4.52
    delta = 0.626671  # 35.9 degree
    brake_effect_coef = 0.9

    # DOB parameters
    l0x = dobpara['l0x']
    l1x = dobpara['l1x']
    l0y = dobpara['l0y']
    l1y = dobpara['l1y']
    av = dobpara['av']
    ata = dobpara['ata']

    # Disturbances and reference
    # varphi, psi, distv, distheta = disturbance(t)

    # Extract state variables
    x_state = x[0]
    y = x[1]
    ta = x[2]
    v = x[3]
    p0x, p1x, p0y, p1y = p_dob[0], p_dob[1], p_dob[2], p_dob[3] # EDOB state
    p0theta, p0v = p_dob[4], p_dob[5]

    # # Compute dx and dy
    # dx = v * np.cos(varphi + ta)
    # dy = v * np.sin(varphi + ta)

    # EDOB for dx and dy
    hdx0 = p0x + l0x * x_state
    hdx1 = p1x + l1x * x_state
    dp0x = -l0x * (v * np.cos(ta) + hdx0) + hdx1
    dp1x = -l1x * (v * np.cos(ta) + hdx0)

    hdy0 = p0y + l0y * y
    hdy1 = p1y + l1y * y
    dp0y = -l0y * (v * np.sin(ta) + hdy0) + hdy1
    dp1y = -l1y * (v * np.sin(ta) + hdy0)

    # DOB for theta and v
    hdv0 = p0v + av * v
    hdta0 = p0theta + ata * ta

    alpha, beta = u[0], u[1]
    u1 = np.tan(beta*delta)
    if alpha > 0:
        omegam = v / (Rwheel * gamma)
        f_1fun = -tau0 * omegam / omega0 + tau0
        T_fun = f_1fun * alpha - c1 * omegam - c0
        # dv = T_fun * gamma / Iwheel * Rwheel + distv

        # DOB for dv and dtheta
        dp0theta = -ata * (v / L * u1 + hdta0)
        dp0v = -av * (T_fun * gamma / Iwheel * Rwheel + hdv0)
    else:
        dp0theta = -ata * (v / L * u1 + hdta0)
        dp0v = -av * ( v * brake_effect_coef * alpha + hdv0)
    # # Filter DOB dynamics
    # dpx = -ax * (v * np.cos(ta) + hdx)
    # dpy = -ay * (v * np.sin(ta) + hdy)
    # dhdxf = -Tf * (hdxf - hdx)
    # dhdyf = -Tf * (hdyf - hdy)

    # Collect state derivatives
    dotp = np.array([dp0x, dp1x, dp0y, dp1y, dp0theta, dp0v])
    hdxyvta = np.array([hdx0, hdy0, hdv0, hdta0])
    return dotp, hdxyvta 