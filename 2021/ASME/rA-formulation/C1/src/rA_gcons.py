#!/usr/bin/env python3

"""Implementations of the primitive GCons

"""

import numpy as np
from numpy import cos, sin, pi


# ------------------------------------- Utility functions -----------------------------------------
def skew(vector):
    """Return the skew symmetric cross product matrix of a 3x1 vector."""
    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])


# ------------------------------------- Driving Constraint -----------------------------------------
class DrivingConstraint:
    """Class to define a driving function and its first and second derivatives

    This class stores the driving constraint function f(t) and its first and
    second derivatives as lambda functions.
    """

    def __init__(self, f_string, f_dot_string, f_ddot_string):
        self.f = lambda t: eval(f_string)
        self.f_dot = lambda t: eval(f_dot_string)
        self.f_ddot = lambda t: eval(f_ddot_string)


# ------------------------------------- DP1 Constraint --------------------------------------------
class GConDP1:
    """This class implements the Dot Product 1 (DP1) geometric constraint.

    The DP1 constraint reflects the fact that motion is such that the dot
    product between a vector attached to body i and a second vector attached
    to body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 8, Slide 34
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.a_bar_j = np.array([constraint_dict['a_bar_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def phi(self, t):
        """Return phi(t), the expression for the constraint."""
        # for readability
        a_bar_j = self.a_bar_j
        a_bar_i = self.a_bar_i
        A_i = self.body_i.A
        A_j = self.body_j.A

        return a_bar_i.T @ A_i.T @ A_j @ a_bar_j - self.prescribed_val.f(t)

    def nu(self, t):
        """Return nu, the RHS of the velocity equation."""
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        """Return gamma, the RHS of the acceleration equation."""
        # for readability
        a_bar_j = self.a_bar_j
        a_bar_i = self.a_bar_i
        A_i = self.body_i.A
        A_j = self.body_j.A
        omega_i = self.body_i.omega
        omega_j = self.body_j.omega
        omega_tilde_i = skew(self.body_i.omega)
        omega_tilde_j = skew(self.body_j.omega)

        return -a_bar_j.T \
               @ (A_j.T @ A_i @ omega_tilde_i @ omega_tilde_i + omega_tilde_j @ omega_tilde_j @ A_j.T @ A_i) @ a_bar_i \
               + 2 * omega_j.T @ skew(a_bar_j) @ A_j.T @ A_i @ skew(a_bar_i) @ omega_i \
               + self.prescribed_val.f_ddot(t)

    def r_sensitivity(self):
        """Return the sensitivity of phi with respect to r."""
        r_sensitivity_i = np.zeros((1, 3))
        r_sensitivity_j = np.zeros((1, 3))
        if self.body_i.is_ground:
            return r_sensitivity_j
        if self.body_j.is_ground:
            return r_sensitivity_i
        return [r_sensitivity_i, r_sensitivity_j]

    def theta_sensitivity(self):
        """Return the sensitivity of phi with respect to theta."""
        # for readability
        a_bar_j = self.a_bar_j
        a_bar_i = self.a_bar_i
        a_bar_i_tilde = skew(self.a_bar_i)
        a_bar_j_tilde = skew(self.a_bar_j)
        A_i = self.body_i.A
        A_j = self.body_j.A

        theta_sensitivity_i = -a_bar_j.T @ A_j.T @ A_i @ a_bar_i_tilde
        theta_sensitivity_j = -a_bar_i.T @ A_i.T @ A_j @ a_bar_j_tilde
        if self.body_i.is_ground:
            return theta_sensitivity_j
        if self.body_j.is_ground:
            return theta_sensitivity_i
        return [theta_sensitivity_i, theta_sensitivity_j]


# ------------------------------------- DP2 Constraint --------------------------------------------
class GConDP2:
    """This class implements the Dot Product 2 (DP2) geometric constraint.

    The DP2 constraint reflects the fact that motion is such that the dot product between a vector
    a_bar_i on body i and a second vector P_iQ_j from body i to body j assumes a specified value
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 9
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        """Return phi(t), the expression for the constraint."""
        # for readability
        a_bar_i = self.a_bar_i
        A_i = self.body_i.A

        return a_bar_i.T @ A_i.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        """Return nu, the RHS of the velocity equation."""
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        """Return gamma, the RHS of the acceleration equation."""
        # for readability
        a_bar_i = self.a_bar_i
        s_bar_q_j = self.s_bar_q_j
        s_bar_p_i = self.s_bar_p_i
        A_i = self.body_i.A
        A_j = self.body_j.A
        omega_i = self.body_i.omega
        omega_tilde_i = skew(self.body_i.omega)
        omega_tilde_j = skew(self.body_j.omega)
        r_dot_i = self.body_i.r_dot
        r_dot_j = self.body_j.r_dot

        return 2 * omega_i.T @ skew(a_bar_i) @ A_i.T @ (r_dot_i - r_dot_j) \
               + 2 * s_bar_q_j.T @ omega_tilde_j @ A_j.T @ A_i @ omega_tilde_i @ a_bar_i \
               - s_bar_p_i.T @ omega_tilde_i @ omega_tilde_i @ a_bar_i \
               - s_bar_q_j.T @ omega_tilde_j @ omega_tilde_j @ A_j.T @ A_i @ a_bar_i \
               - self.d_ij().T @ A_i @ omega_tilde_i @ omega_tilde_i @ a_bar_i \
               + self.prescribed_val.f_ddot(t)

    def r_sensitivity(self):
        """Return the sensitivity of phi with respect to r."""
        # no r dependence, so the partial derivatives are zero
        r_sensitivity_i = -self.a_bar_i.T @ self.body_i.A.T
        r_sensitivity_j = self.a_bar_i.T @ self.body_i.A.T
        if self.body_i.is_ground:
            return r_sensitivity_j
        if self.body_j.is_ground:
            return r_sensitivity_i
        else:
            return [r_sensitivity_i, r_sensitivity_j]

    def theta_sensitivity(self):
        """Return the sensitivity of phi with respect to theta."""
        # for readability
        a_bar_i = self.a_bar_i
        a_bar_i_tilde = skew(self.a_bar_i)
        s_bar_q_j = self.s_bar_q_j
        s_bar_p_i = self.s_bar_p_i
        A_i = self.body_i.A
        A_j = self.body_j.A

        theta_sensitivity_i = a_bar_i.T @ skew(s_bar_p_i) - self.d_ij().T @ A_i @ a_bar_i_tilde
        theta_sensitivity_j = -a_bar_i.T @ A_i.T @ A_j @ skew(s_bar_q_j)
        if self.body_i.is_ground:
            return theta_sensitivity_j
        if self.body_j.is_ground:
            return theta_sensitivity_i
        return [theta_sensitivity_i, theta_sensitivity_j]


# -------------------------------------- D Constraint ---------------------------------------------
class GConD:
    """This class implements the Distance (D) geometric constraint.

    The D constraint reflects the fact that motion is such that the distance between point P on
    body i and point Q on body j assumes a specified value greater than zero.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 12
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        """Return phi(t), the expression for the constraint."""
        return self.d_ij().T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        """Return nu, the RHS of the velocity equation."""
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        """Return gamma, the RHS of the acceleration equation."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_dot_i = self.body_i.r_dot
        r_dot_j = self.body_j.r_dot
        omega_tilde_i = skew(self.body_i.omega)
        omega_tilde_j = skew(self.body_j.omega)
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        return - 2 * (r_dot_j - r_dot_i).T @ (r_dot_j - r_dot_i) \
               + 2 * s_bar_q_j.T @ omega_tilde_j @ omega_tilde_j @ s_bar_q_j \
               + 2 * s_bar_p_i.T @ omega_tilde_i @ omega_tilde_i @ s_bar_p_i \
               - 4 * s_bar_q_j.T @ omega_tilde_j @ A_j.T @ A_i @ omega_tilde_i @ s_bar_p_i \
               + 4 * (r_dot_j - r_dot_i).T @ (A_j @ skew(s_bar_q_j) @ self.body_j.omega
                                              - A_i @ skew(s_bar_p_i) @ self.body_i.omega) \
               - 2 * self.d_ij().T @ (A_i @ omega_tilde_i @ skew(s_bar_p_i) @ self.body_i.omega
                                      - A_j @ omega_tilde_j @ skew(s_bar_q_j) @ self.body_j.omega) \
               + self.prescribed_val.f_ddot(t)

    def r_sensitivity(self):
        """Return the sensitivity of phi with respect to r."""
        r_sensitivity_i = -2 * self.d_ij().T
        r_sensitivity_j = 2 * self.d_ij().T
        if self.body_i.is_ground:
            return r_sensitivity_j
        if self.body_j.is_ground:
            return r_sensitivity_i
        else:
            return [r_sensitivity_i, r_sensitivity_j]

    def theta_sensitivity(self):
        """Return the sensitivity of phi with respect to theta."""
        theta_sensitivity_i = 2 * self.d_ij().T @ self.body_i.A @ skew(self.s_bar_p_i)
        theta_sensitivity_j = -2 * self.d_ij().T @ self.body_j.A @ skew(self.s_bar_q_j)
        if self.body_i.is_ground:
            return theta_sensitivity_j
        if self.body_j.is_ground:
            return theta_sensitivity_i
        else:
            return [theta_sensitivity_i, theta_sensitivity_j]


# ------------------------------------- CD Constraint ---------------------------------------------
class GConCD:
    """This class implements the Coordinate Difference (CD) geometric constraint.

    The CD geometric constraint reflects the fact that motion is such that the difference
    between the x (or y or z) coordinate of point P on body i and the x (or y or z) coordinate
    of point Q on body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 15
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.c = np.array([constraint_dict['c']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        """Return phi(t), the expression for the constraint."""
        return self.c.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        """Return nu, the RHS of the velocity equation."""
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        """Return gamma, the RHS of the acceleration equation."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        omega_tilde_i = skew(self.body_i.omega)
        omega_tilde_j = skew(self.body_j.omega)
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        return self.c.T \
               @ (A_i @ omega_tilde_i @ omega_tilde_i @ s_bar_p_i - A_j @ omega_tilde_j @ omega_tilde_j @ s_bar_q_j) \
               + self.prescribed_val.f_ddot(t)

    def r_sensitivity(self):
        """Return the sensitivity of phi with respect to r."""
        r_sensitivity_i = -self.c.T
        r_sensitivity_j = self.c.T
        if self.body_i.is_ground:
            return r_sensitivity_j
        if self.body_j.is_ground:
            return r_sensitivity_i
        else:
            return [r_sensitivity_i, r_sensitivity_j]

    def theta_sensitivity(self):
        """Return the sensitivity of phi with respect to theta."""
        theta_sensitivity_i = self.c.T @ self.body_i.A @ skew(self.s_bar_p_i)
        theta_sensitivity_j = -self.c.T @ self.body_j.A @ skew(self.s_bar_q_j)
        if self.body_i.is_ground:
            return theta_sensitivity_j
        if self.body_j.is_ground:
            return theta_sensitivity_i
        else:
            return [theta_sensitivity_i, theta_sensitivity_j]
