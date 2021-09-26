#!/usr/bin/env python3

"""Implementations of the primitive GCons
"""

import numpy as np
from numpy import cos, sin, pi

I3 = np.eye(3)

# ------------------------------------- Utility functions -----------------------------------------
def skew(vector):
    """ Function to transform 3x1 vector into a skew symmetric cross product matrix """
    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])


# ------------------------------------- Driving Constraint -----------------------------------------
class DrivingConstraint:
    """This class defines a driving function and its first and second derivatives

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
        p_i = self.body_i.p
        p_j = self.body_j.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        return self.a_bar_i.T @ A_i.T @ A_j @ self.a_bar_j - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.a_bar_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        a_bar_j = self.a_bar_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_j = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                  2 * (p_j[1:] @ a_bar_j.T - (
                                          p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)

        p_dot_i = self.body_i.p_dot
        e_tilde_dot_i = skew(p_dot_i[1:])
        b_mat_dot_i = np.concatenate((2 * ((p_dot_i[0] * I3 + e_tilde_dot_i) @ a_bar_i),
                                      2 * (p_dot_i[1:] @ a_bar_i.T - (
                                              p_dot_i[0] * I3 + e_tilde_dot_i) @ a_bar_tilde_i)), axis=1)

        p_dot_j = self.body_j.p_dot
        e_tilde_dot_j = skew(p_dot_j[1:])
        b_mat_dot_j = np.concatenate((2 * ((p_dot_j[0] * I3 + e_tilde_dot_j) @ a_bar_j),
                                      2 * (p_dot_j[1:] @ a_bar_j.T - (
                                              p_dot_j[0] * I3 + e_tilde_dot_j) @ a_bar_tilde_j)), axis=1)

        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        a_i = A_i @ a_bar_i
        a_j = A_j @ a_bar_j
        a_dot_i = b_mat_i @ p_dot_i
        a_dot_j = b_mat_j @ p_dot_j

        return - a_i.T @ b_mat_dot_j @ p_dot_j \
               - a_j.T @ b_mat_dot_i @ p_dot_i \
               - 2 * (a_dot_i.T @ a_dot_j) + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        phi_r_i = np.zeros((1, 3))
        phi_r_j = np.zeros((1, 3))
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_p(self):
        # calculate partial_phi/partial_p
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.a_bar_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        a_bar_j = self.a_bar_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_j = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                  2 * (p_j[1:] @ a_bar_j.T - (
                                          p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)

        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        a_i = A_i @ a_bar_i
        a_j = A_j @ a_bar_j
        phi_p_i = a_j.T @ b_mat_i
        phi_p_j = a_i.T @ b_mat_j
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        return [phi_p_i, phi_p_j]


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
        # calculate d_ij, the distance between point P and point Q
        p_i = self.body_i.p
        p_j = self.body_j.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        p_i = self.body_i.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        return self.a_bar_i.T @ A_i.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.a_bar_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))

        p_dot_i = self.body_i.p_dot
        e_dot_tilde_i = skew(p_dot_i[1:])
        a_bar_i_s = self.s_bar_p_i
        a_bar_tilde_i_s = skew(a_bar_i_s)
        b_mat_dot_i = np.concatenate((2 * ((p_dot_i[0] * I3 + e_dot_tilde_i) @ a_bar_i_s),
                                      2 * (p_dot_i[1:] @ a_bar_i_s.T - (
                                                  p_dot_i[0] * I3 + e_dot_tilde_i) @ a_bar_tilde_i_s)), axis=1)

        p_dot_j = self.body_j.p_dot
        e_dot_tilde_j = skew(p_dot_j[1:])
        a_bar_j_s = self.s_bar_q_j
        a_bar_tilde_j_s = skew(a_bar_j_s)
        b_mat_dot_j = np.concatenate((2 * ((p_dot_j[0] * I3 + e_dot_tilde_j) @ a_bar_j_s),
                                      2 * (p_dot_j[1:] @ a_bar_j_s.T - (
                                              p_dot_j[0] * I3 + e_dot_tilde_j) @ a_bar_tilde_j_s)), axis=1)

        b_mat_i_s = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i_s),
                                  2 * (p_i[1:] @ a_bar_i_s.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i_s)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        b_mat_j_s = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j_s),
                                  2 * (p_j[1:] @ a_bar_j_s.T - (p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j_s)), axis=1)

        b_mat_dot_i_a = np.concatenate((2 * ((p_dot_i[0] * I3 + e_dot_tilde_i) @ a_bar_i),
                                      2 * (p_dot_i[1:] @ a_bar_i.T - (
                                                  p_dot_i[0] * I3 + e_dot_tilde_i) @ a_bar_tilde_i)), axis=1)

        a_i = A_i @ a_bar_i
        a_dot_i = b_mat_i @ p_dot_i
        d_dot_ij = self.body_j.r_dot + b_mat_j_s @ p_dot_j \
                   - self.body_i.r_dot - b_mat_i_s @ p_dot_i


        return - a_i.T @ b_mat_dot_j @ p_dot_j \
               + a_i.T @ b_mat_dot_i @ p_dot_i \
               - self.d_ij().T @ b_mat_dot_i_a @ p_dot_i \
               - 2 * a_dot_i.T @ d_dot_ij + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        p_i = self.body_i.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        a_bar_i = self.a_bar_i
        phi_r_i = -a_bar_i.T @ A_i.T
        phi_r_j = a_bar_i.T @ A_i.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_p(self):
        # calculate partial_phi/partial_p
        p_i = self.body_i.p
        p_j = self.body_j.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))

        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.a_bar_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)
        a_bar_i_s = self.s_bar_p_i
        a_bar_tilde_i_s = skew(a_bar_i_s)
        b_mat_i_s = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i_s),
                                    2 * (p_i[1:] @ a_bar_i_s.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i_s)), axis=1)
        e_tilde_j = skew(p_j[1:])
        a_bar_j_s = self.s_bar_q_j
        a_bar_tilde_j_s = skew(a_bar_j_s)
        b_mat_j_s = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j_s),
                                    2 * (p_j[1:] @ a_bar_j_s.T - (p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j_s)), axis=1)

        phi_p_i = self.d_ij().T @ b_mat_i \
                  - a_bar_i.T @ A_i.T @ b_mat_i_s
        phi_p_j = a_bar_i.T @ A_i.T @ b_mat_j_s
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        return [phi_p_i, phi_p_j]


# ------------------------------------- D Constraint --------------------------------------------
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
        # calculate d_ij, the distance between point P and point Q
        p_i = self.body_i.p
        p_j = self.body_j.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        d_ij = self.d_ij()
        return d_ij.T @ d_ij - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.s_bar_p_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        a_bar_j = self.s_bar_q_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_j = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                  2 * (p_j[1:] @ a_bar_j.T - (
                                          p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)

        p_dot_i = self.body_i.p_dot
        e_tilde_dot_i = skew(p_dot_i[1:])
        b_mat_dot_i = np.concatenate((2 * ((p_dot_i[0] * I3 + e_tilde_dot_i) @ a_bar_i),
                                      2 * (p_dot_i[1:] @ a_bar_i.T - (
                                              p_dot_i[0] * I3 + e_tilde_dot_i) @ a_bar_tilde_i)), axis=1)

        p_dot_j = self.body_j.p_dot
        e_tilde_dot_j = skew(p_dot_j[1:])
        b_mat_dot_j = np.concatenate((2 * ((p_dot_j[0] * I3 + e_tilde_dot_j) @ a_bar_j),
                                      2 * (p_dot_j[1:] @ a_bar_j.T - (
                                              p_dot_j[0] * I3 + e_tilde_dot_j) @ a_bar_tilde_j)), axis=1)

        d_ij = self.d_ij()
        d_dot_ij = self.body_j.r_dot + b_mat_j @ p_dot_j \
                   - self.body_i.r_dot - b_mat_i @ p_dot_i
        return - 2 * d_ij.T @ b_mat_dot_j @ p_dot_j \
               + 2 * d_ij.T @ b_mat_dot_i @ p_dot_i \
               - 2 * d_dot_ij.T @ d_dot_ij + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        d_ij = self.d_ij()
        phi_r_i = -2 * d_ij.T
        phi_r_j = 2 * d_ij.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_p(self):
        # calculate partial_phi/partial_p
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.s_bar_p_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        a_bar_j = self.s_bar_q_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_j = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                  2 * (p_j[1:] @ a_bar_j.T - (
                                          p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)

        d_ij = self.d_ij()
        phi_p_i = -2 * d_ij.T @ b_mat_i
        phi_p_j = 2 * d_ij.T @ b_mat_j
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        return [phi_p_i, phi_p_j]


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
        # calculate d_ij, the distance between point P and point Q
        p_i = self.body_i.p
        p_j = self.body_j.p
        A_i = (p_i[0] ** 2 - p_i[1:].T @ p_i[1:]) * I3 + 2 * (
                p_i[1:] @ p_i[1:].T + p_i[0] * skew(p_i[1:]))
        A_j = (p_j[0] ** 2 - p_j[1:].T @ p_j[1:]) * I3 + 2 * (
                p_j[1:] @ p_j[1:].T + p_j[0] * skew(p_j[1:]))
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.c.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        p_dot_i = self.body_i.p_dot
        e_tilde_i = skew(p_dot_i[1:])
        a_bar_i = self.s_bar_p_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_dot_i = np.concatenate((2 * ((p_dot_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                      2 * (p_dot_i[1:] @ a_bar_i.T - (
                                                  p_dot_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_dot_j = self.body_j.p_dot
        e_tilde_j = skew(p_dot_j[1:])
        a_bar_j = self.s_bar_q_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_dot_j = np.concatenate((2 * ((p_dot_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                      2 * (p_dot_j[1:] @ a_bar_j.T - (
                                              p_dot_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)

        c = self.c
        return c.T @ b_mat_dot_i @ p_dot_i \
               - c.T @ b_mat_dot_j @ p_dot_j \
               + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        c = self.c
        phi_r_i = -c.T
        phi_r_j = c.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_p(self):
        # calculate partial_phi/partial_p
        p_i = self.body_i.p
        e_tilde_i = skew(p_i[1:])
        a_bar_i = self.s_bar_p_i
        a_bar_tilde_i = skew(a_bar_i)
        b_mat_i = np.concatenate((2 * ((p_i[0] * I3 + e_tilde_i) @ a_bar_i),
                                  2 * (p_i[1:] @ a_bar_i.T - (p_i[0] * I3 + e_tilde_i) @ a_bar_tilde_i)), axis=1)

        p_j = self.body_j.p
        e_tilde_j = skew(p_j[1:])
        a_bar_j = self.s_bar_q_j
        a_bar_tilde_j = skew(a_bar_j)
        b_mat_j = np.concatenate((2 * ((p_j[0] * I3 + e_tilde_j) @ a_bar_j),
                                  2 * (p_j[1:] @ a_bar_j.T - (
                                          p_j[0] * I3 + e_tilde_j) @ a_bar_tilde_j)), axis=1)
        c = self.c
        phi_p_i = -c.T @ b_mat_i
        phi_p_j = c.T @ b_mat_j
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        return [phi_p_i, phi_p_j]
